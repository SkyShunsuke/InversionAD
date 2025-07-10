
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
from numpy import ndarray
import pandas as pd

from tqdm import tqdm
from pathlib import Path

import time
import copy
import argparse
import yaml

from src.datasets import build_dataset
from src.utils import get_optimizer, get_lr_scheduler
from src.denoiser import get_denoiser, Denoiser
from src.backbones import get_backbone

from einops import rearrange
from skimage import measure
from sklearn.metrics import roc_auc_score, average_precision_score, auc
from src.utils import AverageMeter
from src.adeval.adeval import EvalAccumulatorCuda

from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F

from anomalib import metrics

MAX_BATCH_SIZE = 64
NUM_WORKERS = 1

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(description="InversionAD Inference")
    
    parser.add_argument('--eval_strategy', type=str, default='inversion', choices=['inversion', 'reconstruction'], help='Evaluation strategy: inversion or reconstruction')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to the directory contais results')
    parser.add_argument('--eval_step', type=int, default=-1, help='Number of steps for evaluation')
    parser.add_argument('--noise_step', type=int, default=8, help='Number of noise steps for evaluation')
    parser.add_argument('--use_ema_model', action='store_true', help='Use EMA model for evaluation')
    parser.add_argument('--use_best_model', action='store_true', help='Use best model for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    assert sum([args.use_best_model, args.use_ema_model]) < 2, "Please specify either --use_best_model or --use_ema_model"
    return args

def postprocess(x):
    x = x / 2 + 0.5
    return x.clamp(0, 1)

def postprocess_lpips(x):
    # -> [-1, 1]
    x = x * 2 - 1  # Assume x is in [0, 1]

def convert2image(x):
    if x.dim() == 3:
        return x.permute(1, 2, 0).cpu().numpy()
    elif x.dim() == 4:
        return x.permute(0, 2, 3, 1).cpu().numpy()
    else:
        return x.cpu().numpy()
    

@torch.no_grad()
def extract_features(x, model):
    b, c, h, w = x.shape
    out = model.forward_features(x)  # (B, 197, d)
    out = out[:, 1:]  # remove the first token
    out = rearrange(out, 'b (h w) d -> b d h w', h=14, w=14)
    return out

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> float:
    """
    Compute the area under the curve of per-region overlapping (PRO) and FPR from 0 to 0.3.

    Args:
        masks (ndarray): Binary ground truth masks. Shape: (num_test_data, h, w)
        amaps (ndarray): Anomaly maps. Shape: (num_test_data, h, w)
        num_th (int, optional): Number of thresholds to evaluate.

    Returns:
        float: PRO AUC (area under the curve)
    """

    # Validations
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(np.unique(masks)).issubset({0, 1}), "masks must contain only 0 and 1"
    assert isinstance(num_th, int), "type(num_th) must be int"

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    records = []

    for th in np.arange(min_th, max_th, delta):
        binary_amaps = (amaps > th).astype(np.uint8)

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            labeled_mask = measure.label(mask)
            for region in measure.regionprops(labeled_mask):
                coords = region.coords
                tp_pixels = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pros.append(tp_pixels / region.area)

        if len(pros) == 0:
            continue  # Skip if there are no regions

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        denom = inverse_masks.sum()
        fpr = fp_pixels / denom if denom != 0 else 0.0

        records.append({
            "pro": np.mean(pros),
            "fpr": fpr,
            "threshold": th
        })

    df = pd.DataFrame.from_records(records)

    # Filter for FPR < 0.3 and normalize FPR to [0, 1]
    df = df[df["fpr"] < 0.3]
    if df.empty or df["fpr"].max() == 0:
        return 0.0  # No valid points for AUC

    df["fpr"] = df["fpr"] / df["fpr"].max()
    pro_auc = auc(df["fpr"], df["pro"])

    return pro_auc

def main(config, args):
    
    # For reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    assert args.save_dir is not None, "Please provide a save directory"

    dataset_config = config['data']
    device = config['meta']['device']
    
    dataset_config['train'] = False
    dataset_config['normal_only'] = False
    dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**dataset_config)
    dataset_config['anom_only'] = False
    dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**dataset_config)
    
    is_multi_class = isinstance(normal_dataset, ConcatDataset) or isinstance(anom_dataset, ConcatDataset)
    if is_multi_class:
        logger.info(f"Using multi-class dataset")
        anom_loader = [
            DataLoader(
                anom_ds,
                batch_size=MAX_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                drop_last=False
            )
            for anom_ds in anom_dataset.datasets
        ]
        normal_loader = [
            DataLoader(
                normal_ds,
                batch_size=MAX_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                drop_last=False
            )
            for normal_ds in normal_dataset.datasets
        ]
    else:
        logger.info(f"Using single-class dataset: {anom_dataset.category}")
        anom_loader = [
            DataLoader(
                anom_dataset,
                batch_size=MAX_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                drop_last=False,
            )
        ]
        normal_loader = [
            DataLoader(
                normal_dataset,
                batch_size=MAX_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                drop_last=False,
            )
        ]
    
    diff_in_sh = (272, 16, 16)  # For EfficientNet-b4
    # diff_in_sh = (272, 24, 24)  # For EfficientNet-b4 with 24x24 input
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    model.to(device).eval()

    backbone_kwargs = config['backbone']
    logger.info(f"Using feature space reconstruction with {backbone_kwargs['model_type']} backbone")
    
    feature_extractor = get_backbone(**backbone_kwargs)
    feature_extractor.to(device).eval()
    
    # Load the model
    if args.use_ema_model:
        checkpoint_path = os.path.join(args.save_dir, 'model_ema_latest.pth')
    elif args.use_best_model:
        checkpoint_path = os.path.join(args.save_dir, 'model_best.pth')
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Best model checkpoint not found at {checkpoint_path}. Using latest model instead.")
            checkpoint_path = os.path.join(args.save_dir, 'model_latest.pth')
    else:
        checkpoint_path = os.path.join(args.save_dir, 'model_latest.pth')
        
    model_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if is_multi_class:
        # for distributed training, the model state dict keys may have a prefix
        # Remove the prefix if it exists
        if 'module.' in list(model_ckpt.keys())[0]:
            model_ckpt = {k.replace('module.', ''): v for k, v in model_ckpt.items()}
    model.load_state_dict(model_ckpt, strict=True)
    logger.info(f"Loaded model from {checkpoint_path}")
    
    # Cout the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in the model: {num_params / 1e6:.2f}M")
    
    if args.eval_strategy == 'reconstruction':
        logger.info("Evaluating reconstruction performance")
        assert args.noise_step < args.eval_step, "Noise step should be less than evaluation step for reconstruction"
        metrics_dict = evaluate_recon(
            model,
            feature_extractor,
            anom_loader,
            normal_loader,
            config, 
            diff_in_sh,
            "Eval",
            args.eval_step if args.eval_step != -1 else config["evaluation"]["eval_step"],
            args.noise_step,
            device
        )
    elif args.eval_strategy == 'inversion':
        metrics_dict = evaluate_inv(
            model,
            feature_extractor,
            anom_loader,
            normal_loader,
            config, 
            diff_in_sh,
            "Eval",
            config["evaluation"]["eval_step"] if args.eval_step == -1 else args.eval_step,
            device,
        )
            
    logger.info(f"{metrics_dict}")
    # Compute Average AUC
    if is_multi_class:
        img_aucs = [metrics_dict[cat]["I-AUROC"] for cat in metrics_dict]
        px_aucs = [metrics_dict[cat]["P-AUROC"] for cat in metrics_dict]
        img_aps = [metrics_dict[cat]["I-AP"] for cat in metrics_dict]
        px_aps = [metrics_dict[cat]["P-AP"] for cat in metrics_dict]
        pros = [metrics_dict[cat]["PRO"] for cat in metrics_dict]
        img_f1s = [metrics_dict[cat]["I-F1Max"] for cat in metrics_dict]
        px_f1s = [metrics_dict[cat]["P-F1Max"] for cat in metrics_dict]
        latencies = [metrics_dict[cat]["latency"] for cat in metrics_dict]    
        memory_usage = [metrics_dict[cat]["memory"] for cat in metrics_dict]
        
        avg_img_auc = np.mean(img_aucs)
        avg_px_auc = np.mean(px_aucs)
        avg_img_ap = np.mean(img_aps)
        avg_px_ap = np.mean(px_aps)
        avg_pro = np.mean(pros)
        avg_img_f1 = np.mean(img_f1s)
        avg_px_f1 = np.mean(px_f1s)
        avg_latency = np.mean(latencies)
        avg_memory_usage = np.mean(memory_usage)
        
        logger.info(f"\nImage-level Metrics:\n ================\n")
        logger.info(f"Average Image AUC: {avg_img_auc}")
        logger.info(f"Average Image AP: {avg_img_ap}")
        logger.info(f"Average Image F1Max: {avg_img_f1}")
        logger.info(f"\nPixel-level Metrics:\n================\n")
        logger.info(f"Average Pixel AUC: {avg_px_auc}")
        logger.info(f"Average Pixel AP: {avg_px_ap}")
        logger.info(f"Average PRO: {avg_pro}")
        logger.info(f"Average Pixel F1Max: {avg_px_f1}")
        logger.info(f"\nEfficiency Metrics:\n ================\n")
        logger.info(f"Average Latency: {avg_latency} ms")
        logger.info(f"Average Memory Usage: {avg_memory_usage} MB")
        
    
def init_denoiser(num_inference_steps, device, config, in_sh, inherit_model=None):
    config["diffusion"]["num_sampling_steps"] = str(num_inference_steps)
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=in_sh)
    
    if inherit_model is not None:
        for p, p_inherit in zip(model.parameters(), inherit_model.parameters()):
            p.data.copy_(p_inherit.data)
    model.to(device).eval()
    return model

def calculate_log_pdf(x):
    ll = -0.5 * (x ** 2 + np.log(2 * np.pi))
    ll = ll.sum(dim=(1, 2, 3))
    return ll

def calculate_log_pdf_spatial(x):
    # Calculate log pdf for each spatial dimension
    ll = -0.5 * (x ** 2 + np.log(2 * np.pi))
    ll = ll.sum(dim=1)  # Sum over the channel dimension
    return ll

@torch.no_grad()
def evaluate_recon(denoiser, feature_extractor, anom_loaders, normal_loaders, config, in_sh, epoch, eval_step, noise_step, device):
    denoiser.eval()
    feature_extractor.eval()
    
    eval_denoiser = init_denoiser(eval_step, device, config, in_sh, inherit_model=denoiser)
    roc_dict = {}
    for normal_loader, anom_loader in zip(normal_loaders, anom_loaders):
        category = anom_loader.dataset.category if hasattr(anom_loader.dataset, 'category') else 'unknown'
        logger.info(f"[{category}] Evaluating on {len(anom_loader)} anomalous samples and {len(normal_loader)} normal samples")
        logger.info(f"[{category}] Evaluation step: {eval_step}")
        logger.info(f"[{category}] Epoch: {epoch}")
        
        losses = []
        mses = []
        mses_sp = []
        noise_steps = torch.tensor([noise_step] * 1, device=device, dtype=torch.long)
        noise = torch.randn((1, *in_sh), device=device, dtype=torch.float32)
        for i, batch in enumerate(normal_loader):
            images = batch["samples"].to(device)
            labels = batch["clslabels"].to(device)
            
            features, features_list = feature_extractor(images)
            loss = denoiser(features, labels)
            losses.append(loss.cpu().numpy())
            
            # Perturb to x_t
            x_t = eval_denoiser.q_sample(features, noise_steps, noise=noise)
            
            # Reconstruct
            x_rec = eval_denoiser.denoise_from_intermediate(x_t, noise_steps, labels, sampler="ddim")
            
            mse = torch.mean((x_rec - features) ** 2, dim=(1, 2, 3))  # (bs, )
            min_mse_spatial = mse.view(mse.shape[0], -1).min(dim=1)[0]  # (bs, )
            max_mse_spatial = mse.view(mse.shape[0], -1).max(dim=1)[0]  # (bs, )
            mse_sp = torch.abs(min_mse_spatial - max_mse_spatial)  # (bs, )
            mses_sp.extend(mse_sp.cpu().numpy())
            mses.extend(mse.cpu().numpy())
        
        for i, batch in enumerate(anom_loader):
            images = batch["samples"].to(device)
            labels = batch["clslabels"].to(device)
            
            features, features_list = feature_extractor(images)
            loss = denoiser(features, labels)
            losses.append(loss.cpu().numpy())
            
            # Perturb to x_t
            x_t = eval_denoiser.q_sample(features, noise_steps, noise=noise)
            
            # Reconstruct
            x_rec = eval_denoiser.denoise_from_intermediate(x_t, noise_steps, labels, sampler="ddim")
            
            mse = torch.mean((x_rec - features) ** 2, dim=(1, 2, 3))
            min_mse_spatial = mse.view(mse.shape[0], -1).min(dim=1)[0]  # (bs, )
            max_mse_spatial = mse.view(mse.shape[0], -1).max(dim=1)[0]  # (bs, )
            mse_sp = torch.abs(min_mse_spatial - max_mse_spatial)  # (bs, )
            mses_sp.extend(mse_sp.cpu().numpy())
            mses.extend(mse.cpu().numpy())
            
        losses = np.array(losses)
        logger.info(f"[{category}] Loss: {losses.mean()} at epoch {epoch}")
        mses = np.array(mses)
        logger.info(f"[{category}] MSE: {mses.mean()} at epoch {epoch}")
        
        normal_mses = mses[:len(normal_loader.dataset)]
        anomaly_mses = mses[len(normal_loader.dataset):]
        normal_mses_sp = mses_sp[:len(normal_loader.dataset)]
        anomaly_mses_sp = mses_sp[len(normal_loader.dataset):]
        normal_mses = np.array(normal_mses)
        anomaly_mses = np.array(anomaly_mses)
        normal_mses_sp = np.array(normal_mses_sp)
        anomaly_mses_sp = np.array(anomaly_mses_sp)
        mses_min = np.min([normal_mses.min(), anomaly_mses.min()])
        mses_max = np.max([normal_mses.max(), anomaly_mses.max()])
        mses_sp_min = np.min([normal_mses_sp.min(), anomaly_mses_sp.min()])
        mses_sp_max = np.max([normal_mses_sp.max(), anomaly_mses_sp.max()])
        eps = 1e-8
        normal_mses = (normal_mses - mses_min) / (mses_max - mses_min + eps)
        anomaly_mses = (anomaly_mses - mses_min) / (mses_max - mses_min + eps)
        normal_mses_sp = (normal_mses_sp - mses_sp_min) / (mses_sp_max - mses_sp_min + eps)
        anomaly_mses_sp = (anomaly_mses_sp - mses_sp_min) / (mses_sp_max - mses_sp_min + eps)

        y_true = np.concatenate([np.zeros(len(normal_mses)), np.ones(len(anomaly_mses))])
        normal_scores = normal_mses + normal_mses_sp
        anomaly_scores = anomaly_mses + anomaly_mses_sp
        y_score = np.concatenate([normal_scores, anomaly_scores])
        from sklearn.metrics import roc_auc_score
    
        roc_auc = roc_auc_score(y_true, y_score)
        roc_dict[category] = roc_auc
        
        logger.info(f"[{category}] AUC: {roc_auc} at epoch {epoch}")
        
    logger.info(f"Evaluation completed for all categories.")
    return roc_dict

@torch.no_grad()
def evaluate_inv(denoiser, feature_extractor, anom_loaders, normal_loaders, config, in_sh, epoch, eval_step, device):
    denoiser.eval()
    feature_extractor.eval()
    
    eval_denoiser = init_denoiser(eval_step, device, config, in_sh, inherit_model=denoiser)
    metrics_dict = {}
    for normal_loader, anom_loader in zip(normal_loaders, anom_loaders):
        category = anom_loader.dataset.category if hasattr(anom_loader.dataset, 'category') else 'unknown'
        if category not in metrics_dict:
            metrics_dict[category] = {}
        logger.info(f"[{category}] Evaluating on {len(anom_loader.dataset)} anomalous samples and {len(normal_loader.dataset)} normal samples")
        logger.info(f"[{category}] Evaluation step: {eval_step}")
        logger.info(f"[{category}] Epoch: {epoch}")
    
        start_t = torch.tensor([0] * 8, device=device, dtype=torch.long)
        normal_ats = []
        normal_nlls = []
        normal_maps = []
        normal_gt_masks = []
        losses = []
        time_meter = AverageMeter()
        memory_meter = AverageMeter()
        c = 0
        for batch in tqdm(normal_loader, total=len(normal_loader)):
            
            images = batch["samples"].to(device)
            org_h, org_w = images.shape[2], images.shape[3]
            labels = batch["clslabels"].to(device)
            normal_gt_masks.append(batch["masks"])
            
            s_time = time.perf_counter()
            features, features_list = feature_extractor(images)
            loss = denoiser(features, labels)
            losses.append(loss.cpu().numpy())
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                latents_last = eval_denoiser.ddim_reverse_sample(
                    features, start_t, labels, eta=0.0
                )
            latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
            ats = torch.abs(latents_last_l2 - torch.sqrt(torch.tensor([0], device=device, dtype=torch.float32))) 
            min_ats_spatial = ats.view(ats.shape[0], -1).min(dim=1)[0]  # (bs, )
            max_ats_spatial = ats.view(ats.shape[0], -1).max(dim=1)[0]  # (bs, )
            ats = torch.abs(min_ats_spatial - max_ats_spatial)  # (bs, )
            nll = calculate_log_pdf(latents_last.cpu()) * -1
            e_time = time.perf_counter()
            time_meter.update(e_time - s_time, n=1)

            normal_map = F.interpolate(latents_last_l2.unsqueeze(0), size=(org_h, org_w), mode='bilinear', align_corners=False).squeeze(0)
            normal_maps.append(normal_map.cpu().numpy())
            normal_nlls.extend(nll.cpu().numpy())
            normal_ats.extend(ats.cpu().numpy())
            
            memory_meter.update(
                torch.cuda.max_memory_allocated() / (1024 * 1024), n=latents_last.shape[0]
            )
            
        # logger.info(f"[{category}] Average time per sample: {time_meter.avg*1000:.4f} [ms]")
        anomaly_ats = []
        anomaly_nlls = []
        anomaly_maps = []
        anomaly_gt_masks = []
        # time_meter = AverageMeter()
        
        c = 0
        for batch in tqdm(anom_loader, total=len(anom_loader)):
            images = batch["samples"].to(device)
            labels = batch["clslabels"].to(device)
            anomaly_gt_masks.append(batch["masks"])
            
            torch.cuda.synchronize()
            s_time = time.perf_counter()
            features, features_list = feature_extractor(images)
            loss = denoiser(features, labels)
            losses.append(loss.cpu().numpy())
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                latents_last = eval_denoiser.ddim_reverse_sample(
                    features, start_t, labels, eta=0.0
                )
            
            latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
            ats = torch.abs(latents_last_l2 - torch.sqrt(torch.tensor([0], device=device, dtype=torch.float32))) 
            min_ats_spatial = ats.view(ats.shape[0], -1).min(dim=1)[0]
            max_ats_spatial = ats.view(ats.shape[0], -1).max(dim=1)[0]
            ats = torch.abs(min_ats_spatial - max_ats_spatial)
            nll = calculate_log_pdf(latents_last.cpu()) * -1
            e_time = time.perf_counter()
            
            time_meter.update(e_time - s_time, n=1)
    
            anomaly_map = F.interpolate(latents_last_l2.unsqueeze(0), size=(org_h, org_w), mode='bilinear', align_corners=False).squeeze(0)
            anomaly_maps.append(anomaly_map.cpu().numpy())
            anomaly_nlls.extend(nll.cpu().numpy())
            anomaly_ats.extend(ats.cpu().numpy())
            
            memory_meter.update(
                torch.cuda.max_memory_allocated() / (1024 * 1024), n=latents_last.shape[0]
            )
        
        losses = np.array(losses)
        logger.info(f"[{category}] Loss: {losses.mean()} at epoch {epoch}")
        logger.info(f"[{category}] Average time per sample: {time_meter.avg*1000:.4f} [ms]")
        logger.info(f"[{category}] Average memory usage: {memory_meter.avg:.2f} [MB]")
        metrics_dict[category]["latency"] = time_meter.avg * 1000  # Convert to milliseconds
        metrics_dict[category]["memory"] = memory_meter.avg  # in MB
        
        normal_ats = np.array(normal_ats)
        anomaly_ats = np.array(anomaly_ats)
        normal_nlls = np.array(normal_nlls)
        anomaly_nlls = np.array(anomaly_nlls)
        normal_maps = np.concatenate(normal_maps, axis=0)
        anomaly_maps = np.concatenate(anomaly_maps, axis=0)
        normal_gt_masks = torch.cat(normal_gt_masks, dim=0)
        anomaly_gt_masks = torch.cat(anomaly_gt_masks, dim=0)

        ats_min = np.min([normal_ats.min(), anomaly_ats.min()])
        ats_max = np.max([normal_ats.max(), anomaly_ats.max()])
        nlls_min = np.min([normal_nlls.min(), anomaly_nlls.min()])
        nlls_max = np.max([normal_nlls.max(), anomaly_nlls.max()])
        eps = 1e-8  # Small constant to avoid division by zero
        normal_ats = (normal_ats - ats_min) / (ats_max - ats_min + eps) 
        anomaly_ats = (anomaly_ats - ats_min) / (ats_max - ats_min + eps) 
        normal_nlls = (normal_nlls - nlls_min) / (nlls_max - nlls_min + eps)
        anomaly_nlls = (anomaly_nlls - nlls_min) / (nlls_max - nlls_min + eps)

        # Calculate Metrics for image-level
        s_time = time.perf_counter()
        y_true = np.concatenate([np.zeros(len(normal_ats)), np.ones(len(anomaly_ats))])
        normal_scores = normal_ats + normal_nlls
        anomaly_scores = anomaly_ats + anomaly_nlls
        y_score = np.concatenate([normal_scores, anomaly_scores])
        roc_auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        f1max = metrics.F1Max()
        f1max_score = f1max(torch.from_numpy(y_score).to(device), torch.from_numpy(y_true).to(device))
        e_time = time.perf_counter()
        logger.info(f"[{category}] Image-level metrics computation time: {e_time - s_time:.4f} seconds")
        
        metrics_dict[category]["I-AUROC"] = roc_auc
        metrics_dict[category]["I-AP"] = ap
        metrics_dict[category]["I-F1Max"] = f1max_score.item()
        
        # Calculate Metics for pixel-level
        y_true_px = np.concatenate([
            normal_gt_masks.cpu().numpy().flatten(),
            anomaly_gt_masks.cpu().numpy().flatten()
        ])
        y_true_map = np.concatenate([
            normal_gt_masks.cpu().squeeze(1).numpy(),
            anomaly_gt_masks.cpu().squeeze(1).numpy()
        ])
        y_true_map = np.where(y_true_map > 0.5, 1, 0)  # Convert masks to binary
        y_score_map = np.concatenate([
            normal_maps, 
            anomaly_maps
        ])
        y_true_px = np.where(y_true_px > 0.5, 1, 0)  # Convert masks to binary
        y_score_px = np.concatenate([
            normal_maps.flatten(),
            anomaly_maps.flatten()
        ])
        
        
        score_min, score_max = y_score.min(), y_score.max()
        anomap_min, anomap_max = y_score_map.min(), y_score_map.max()
        accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max)
        
        s_time = time.perf_counter()
        accum_batch_size = 2048
        num_batches = len(y_true_px) // accum_batch_size + (1 if len(y_true_px) % accum_batch_size > 0 else 0)
        logger.info(f"[{category}] Number of batches for pixel-level evaluation: {num_batches}")
        for i in range(0, len(y_true_px), accum_batch_size):
            end_idx = min(i + accum_batch_size, len(y_true_px))
            batch_y_true_map = torch.from_numpy(y_true_map[i:end_idx]).to(device)
            batch_y_score_map = torch.from_numpy(y_score_map[i:end_idx]).to(device)
            batcy_y_score = torch.from_numpy(y_score[i:end_idx]).to(device)
            batch_y_true = torch.from_numpy(y_true[i:end_idx]).to(device)
            
            accum.add_anomap_batch(batch_y_score_map, batch_y_true_map)
            accum.add_image(batcy_y_score, batch_y_true)
            
        ad_metrics = accum.summary()
        
        e_time = time.perf_counter()
        logger.info(f"[{category}] Pixel-level metrics computation time: {e_time - s_time:.4f} seconds")
        roc_auc_px = ad_metrics["p_auroc"]
        pro = ad_metrics["p_aupro"]
        
        # s_time = time.perf_counter()
        # roc_auc_px = roc_auc_score(y_true_px, y_score_px)
        # e_time = time.perf_counter()
        # logger.info(f"[{category}] Pixel AUC computation time: {e_time - s_time:.4f} seconds")
        
        s_time = time.perf_counter()
        ap_px = average_precision_score(y_true_px, y_score_px)
        e_time = time.perf_counter()
        logger.info(f"[{category}] Pixel AP computation time: {e_time - s_time:.4f} seconds")

        # s_time = time.perf_counter()
        # pro = compute_pro(y_true_map, y_score_map)
        # e_time = time.perf_counter()
        # logger.info(f"[{category}] PRO computation time: {e_time - s_time:.4f} seconds")
        
        s_time = time.perf_counter()
        f1max_px = metrics.F1Max()
        e_time = time.perf_counter()
        logger.info(f"[{category}] Pixel F1Max computation time: {e_time - s_time:.4f} seconds")
        
        s_time = time.perf_counter()
        f1max_px_score = f1max_px(torch.from_numpy(y_score_px).to(device), torch.from_numpy(y_true_px).to(device))
        e_time = time.perf_counter()
        logger.info(f"[{category}] Pixel F1Max score computation time: {e_time - s_time:.4f} seconds")
        
        metrics_dict[category]["P-AUROC"] = roc_auc_px
        metrics_dict[category]["P-AP"] = ap_px
        metrics_dict[category]["PRO"] = pro
        metrics_dict[category]["P-F1Max"] = f1max_px_score.item()
        
        logger.info(f"[{category}] Image AUC: {roc_auc} at epoch {epoch}")
        logger.info(f"[{category}] Pixel AUC: {roc_auc_px} at epoch {epoch}")
        logger.info(f"[{category}] Image AP: {ap} at epoch {epoch}")
        logger.info(f"[{category}] Pixel AP: {ap_px} at epoch {epoch}")
        logger.info(f"[{category}] PRO: {pro} at epoch {epoch}")
        logger.info(f"[{category}] Image F1Max: {f1max_score.item()} at epoch {epoch}")
        logger.info(f"[{category}] Pixel F1Max: {f1max_px_score.item()} at epoch {epoch}")
        
        torch.cuda.empty_cache()
    logger.info(f"Evaluation completed for all categories.")
    return metrics_dict

import torch.distributed as dist
@torch.no_grad()
def concat_all_gather(tensor, world_size):
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor.contiguous())
    return torch.cat(tensor_list, dim=0)

@torch.no_grad()
def evaluate_dist(denoiser, feature_extractor, anom_loader, normal_loader, config, in_sh, epoch, eval_step, device, world_size, rank):
    denoiser.eval()
    feature_extractor.eval()
    category = anom_loader.dataset.category
    
    eval_denoiser = init_denoiser(eval_step, device, config, in_sh, inherit_model=denoiser)
    
    logger.info(f"[{category}] Evaluating on {len(anom_loader.dataset)} anomalous samples and {len(normal_loader.dataset)} normal samples")
    logger.info(f"[{category}] Evaluation step: {eval_step}")
    logger.info(f"[{category}] Epoch: {epoch}")
    
    start_t = torch.tensor([0] * 8, device=device, dtype=torch.long)
    normal_ats = []
    normal_nlls = []
    normal_maps = []
    normal_gt_masks = []
    losses = []
    for i, batch in enumerate(normal_loader):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        normal_gt_masks.append(batch["masks"].to(device))
        
        features, features_list = feature_extractor(images)
        loss = denoiser(features, labels)
        losses.append(loss.cpu().numpy())
        
        latents_last = eval_denoiser.ddim_reverse_sample(
            features, start_t, labels, eta=0.0
        )
        latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
        ats = torch.abs(latents_last_l2 - torch.sqrt(torch.tensor([0], device=device, dtype=torch.float32))) 
        min_ats_spatial = ats.view(ats.shape[0], -1).min(dim=1)[0]  # (bs, )
        max_ats_spatial = ats.view(ats.shape[0], -1).max(dim=1)[0]  # (bs, )
        ats = torch.abs(min_ats_spatial - max_ats_spatial)  # (bs, )
        nll = calculate_log_pdf(latents_last) * -1
        
        normal_map = F.interpolate(latents_last_l2.unsqueeze(0), size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False).squeeze(0)
        normal_maps.append(normal_map)
    
        normal_nlls.append(nll)
        normal_ats.append(ats)
    dist.barrier()  # Ensure all processes have completed the normal data processing
        
    anomaly_ats = []
    anomaly_nlls = []
    anomaly_maps = []
    anomaly_gt_masks = []
    for i, batch in enumerate(anom_loader):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        anomaly_gt_masks.append(batch["masks"].to(device))
        
        features, features_list = feature_extractor(images)
        loss = denoiser(features, labels)
        losses.append(loss.cpu().numpy())
        latents_last = eval_denoiser.ddim_reverse_sample(
            features, start_t, labels, eta=0.0
        )
        latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
        ats = torch.abs(latents_last_l2 - torch.sqrt(torch.tensor([0], device=device, dtype=torch.float32))) 
        min_ats_spatial = ats.view(ats.shape[0], -1).min(dim=1)[0]
        max_ats_spatial = ats.view(ats.shape[0], -1).max(dim=1)[0]
        ats = torch.abs(min_ats_spatial - max_ats_spatial)
        nll = calculate_log_pdf(latents_last) * -1
        
        anomaly_map = F.interpolate(latents_last_l2.unsqueeze(0), size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False).squeeze(0)
        anomaly_maps.append(anomaly_map)
        anomaly_nlls.append(nll)
        anomaly_ats.append(ats)
    dist.barrier()  # Ensure all processes have completed the anomaly data processing
    
    losses = np.array(losses)
    logger.info(f"[{category}] Loss: {losses.mean()} at epoch {epoch}")
    
    normal_ats = torch.cat(normal_ats, dim=0)  
    anomaly_ats = torch.cat(anomaly_ats, dim=0)
    normal_nlls = torch.cat(normal_nlls, dim=0)
    anomaly_nlls = torch.cat(anomaly_nlls, dim=0)
    normal_maps = torch.cat(normal_maps, dim=0)
    anomaly_maps = torch.cat(anomaly_maps, dim=0)
    normal_gt_masks = torch.cat(normal_gt_masks, dim=0).squeeze(1)  # Assuming masks are in shape (bs, 1, h, w)
    anomaly_gt_masks = torch.cat(anomaly_gt_masks, dim=0).squeeze(1)  # Assuming masks are in shape (bs, 1, h, w)
    
    # Gather results from all processes
    def to_numpy(tensor):
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    normal_ats = to_numpy(concat_all_gather(normal_ats, world_size))
    anomaly_ats = to_numpy(concat_all_gather(anomaly_ats, world_size))
    normal_nlls = to_numpy(concat_all_gather(normal_nlls, world_size))
    anomaly_nlls = to_numpy(concat_all_gather(anomaly_nlls, world_size))
    normal_maps = to_numpy(concat_all_gather(normal_maps, world_size))
    anomaly_maps = to_numpy(concat_all_gather(anomaly_maps, world_size))
    normal_gt_masks = to_numpy(concat_all_gather(normal_gt_masks, world_size))
    anomaly_gt_masks = to_numpy(concat_all_gather(anomaly_gt_masks, world_size))
    
    if rank != 0:
        return None
    
    logger.info(f"[{category}] Number of normal samples: {len(normal_ats)}")
    logger.info(f"[{category}] Number of anomaly samples: {len(anomaly_ats)}")
    ats_min = np.min([normal_ats.min(), anomaly_ats.min()])
    ats_max = np.max([normal_ats.max(), anomaly_ats.max()])
    nlls_min = np.min([normal_nlls.min(), anomaly_nlls.min()])
    nlls_max = np.max([normal_nlls.max(), anomaly_nlls.max()])
    eps = 1e-8  # Small constant to avoid division by zero
    normal_ats = (normal_ats - ats_min) / (ats_max - ats_min + eps) 
    anomaly_ats = (anomaly_ats - ats_min) / (ats_max - ats_min + eps) 
    normal_nlls = (normal_nlls - nlls_min) / (nlls_max - nlls_min + eps)
    anomaly_nlls = (anomaly_nlls - nlls_min) / (nlls_max - nlls_min + eps)

    y_true = np.concatenate([np.zeros(len(normal_ats)), np.ones(len(anomaly_ats))])
    normal_scores = normal_ats + normal_nlls
    anomaly_scores = anomaly_ats + anomaly_nlls
    y_score = np.concatenate([normal_scores, anomaly_scores])
    
    # Image-level metrics
    roc_auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    # f1max = metrics.F1Max()
    # f1max.update(torch.from_numpy(y_score).to(device), torch.from_numpy(y_true).to(device))
    # f1max_score = f1max.compute()
    f1max_score = 0
    
    logger.info(f"[{category}] Image-level metrics: AUC: {roc_auc}, AP: {ap}, F1Max: {f1max_score} at epoch {epoch}")
    metrics_dict = {
        "I-AUROC": roc_auc,
        "I-AP": ap,
        "I-F1Max": f1max_score
    }
    
    # Pixel-level metrics
    y_true_px = np.concatenate([
        normal_gt_masks.flatten(),
        anomaly_gt_masks.flatten()
    ])
    y_true_map = np.concatenate([
        normal_gt_masks,
        anomaly_gt_masks
    ])
    y_true_map = np.where(y_true_map > 0.5, 1, 0)
    y_score_map = np.concatenate([
        normal_maps, 
        anomaly_maps
    ])
    y_true_px = np.where(y_true_px > 0.5, 1, 0)
    y_score_px = np.concatenate([
        normal_maps.flatten(),
        anomaly_maps.flatten()
    ])
    score_min, score_max = y_score.min(), y_score.max()
    anomap_min, anomap_max = y_score_map.min(), y_score_map.max()
    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max)
    
    accum_batch_size = 128
    num_batches = len(y_true_px) // accum_batch_size + (1 if len(y_true_px) % accum_batch_size > 0 else 0)
    logger.info(f"[{category}] Number of batches for pixel-level evaluation: {num_batches}")
    for i in range(0, len(y_true_px), accum_batch_size):
        end_idx = min(i + accum_batch_size, len(y_true_px))
        batch_y_true_map = torch.from_numpy(y_true_map[i:end_idx]).to(device)
        batch_y_score_map = torch.from_numpy(y_score_map[i:end_idx]).to(device)
        batcy_y_score = torch.from_numpy(y_score[i:end_idx]).to(device)
        batch_y_true = torch.from_numpy(y_true[i:end_idx]).to(device)
        
        accum.add_anomap_batch(batch_y_score_map, batch_y_true_map)
        accum.add_image(batcy_y_score, batch_y_true)
    
    ad_metrics = accum.summary()
    roc_auc_px = ad_metrics["p_auroc"]
    pro = ad_metrics["p_aupro"]
    ap_px = average_precision_score(y_true_px, y_score_px)
    f1max_px = metrics.F1Max()
    # f1max_px_score = f1max_px(torch.from_numpy(y_score_px).to(device), torch.from_numpy(y_true_px).to(device))
    f1max_px_score = 0
    logger.info(f"[{category}] Pixel-level metrics: AUC: {roc_auc_px}, AP: {ap_px}, PRO: {pro}, F1Max: {f1max_px_score} at epoch {epoch}")
    mad = np.mean([roc_auc, roc_auc_px, ap, ap_px, pro, f1max_score, f1max_px_score])
    logger.info(f"[{category}] mAD: {mad} at epoch {epoch}")
    metrics_dict.update({
        "P-AUROC": roc_auc_px,
        "P-AP": ap_px,
        "PRO": pro,
        "P-F1Max": f1max_px_score,
        "mAD": mad,
    })
    
    return {category: metrics_dict}


if __name__ == "__main__":
    args = parse_args()
    main(args)