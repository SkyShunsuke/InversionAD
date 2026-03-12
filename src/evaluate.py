
import os
import sys
import torch
from torch.utils.data import DataLoader

import numpy as np
from numpy import ndarray
import pandas as pd

from tqdm import tqdm

import time
import argparse
import yaml

from src.datasets import build_dataset
from src.denoiser import get_denoiser, Denoiser
from src.backbones import get_backbone, get_backbone_feature_shape

from skimage import measure
from sklearn.metrics import roc_auc_score, average_precision_score, auc
from src.utils import AverageMeter

from torch.utils.data import ConcatDataset
from torch.nn import functional as F
import matplotlib.pyplot as plt 

from src.adeval.eval_utils import (
    calculate_img_metrics,
    calculate_px_metrics,
    divide_by_class,
    extract_features,
    aggregate_px_values,
    SUPPORTED_METRICS
)

MAX_BATCH_SIZE = 64
NUM_WORKERS = 4

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def parse_args():
    parser = argparse.ArgumentParser(description="InversionAD Inference")
    
    parser.add_argument('--eval_strategy', type=str, default='inversion', choices=['inversion', 'reconstruction'], help='Evaluation strategy: inversion or reconstruction')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to the directory contais results')
    parser.add_argument('--eval_step', type=int, default=-1, help='Number of steps for evaluation')
    parser.add_argument('--noise_step', type=int, default=8, help='Number of noise steps for evaluation')
    parser.add_argument('--use_ema_model', action='store_true', help='Use EMA model for evaluation')
    parser.add_argument('--use_best_model', action='store_true', help='Use best model for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--category', type=str, default=None, help='Category to evaluate on')
    parser.add_argument('--visualize_samples', action='store_true', help='Visualize samples during evaluation')
    
    args = parser.parse_args()
    assert sum([args.use_best_model, args.use_ema_model]) < 2, "Please specify either --use_best_model or --use_ema_model"
    return args

def denormalize(x):
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = x * imagenet_std + imagenet_mean
    return x.clamp(0, 1)

def postprocess(x):
    x = x / 2 + 0.5
    return x.clamp(0, 1)

def convert2image(x):
    if x.dim() == 3:
        return x.permute(1, 2, 0).cpu().numpy()
    elif x.dim() == 4:
        return x.permute(0, 2, 3, 1).cpu().numpy()
    else:
        return x.cpu().numpy()

def main(config, args):
    
    # For reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    assert args.save_dir is not None, "Please provide a save directory"

    dataset_config = config['data']
    device = config['meta']['device']
    
    if args.category is not None:
        dataset_config['category'] = args.category
        dataset_config['dataset_name'] = dataset_config['dataset_name'].split('_all')[0]  # Remove '_all' suffix if exists
        logger.info(f"Evaluating on category: {args.category}")
    
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
    
    diff_in_sh = get_backbone_feature_shape(model_type=config['backbone']['model_type'])
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
        
    model_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
            args
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
def evaluate_inv(denoiser, feature_extractor, anom_loaders, normal_loaders, config, in_sh, epoch, eval_step, device, args):
    denoiser.eval()
    feature_extractor.eval()
    
    eval_denoiser = init_denoiser(eval_step, device, config, in_sh, inherit_model=denoiser)
    metrics_dict = {}
    for normal_loader, anom_loader in zip(normal_loaders, anom_loaders):
        
        samples_dict = {}
        category = anom_loader.dataset.category if hasattr(anom_loader.dataset, 'category') else 'unknown'
        if category not in metrics_dict:
            metrics_dict[category] = {}
        if category not in samples_dict:
            samples_dict[category] = {}
        logger.info(f"[{category}] Evaluating on {len(anom_loader.dataset)} anomalous samples and {len(normal_loader.dataset)} normal samples")
        logger.info(f"[{category}] Evaluation step: {eval_step}")
        logger.info(f"[{category}] Epoch: {epoch}")
    
        start_t = torch.tensor([0] * 8, device=device, dtype=torch.long)
        normal_diffs = []
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
            paths = batch["filenames"]
            labels = batch["clslabels"].to(device)
            anom_labels = batch["labels"]
            normal_gt_masks.append(batch["masks"])
            
            s_time = time.perf_counter()
            features, _ = feature_extractor(images)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                latents_last = eval_denoiser.ddim_reverse_sample(
                    features, start_t, labels, eta=0.0
                )
            latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
            min_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).min(dim=1)[0]  # (bs, )
            max_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).max(dim=1)[0]  # (bs, )
            diffs = max_diffs_spatial - min_diffs_spatial  # (bs, )
            nll = calculate_log_pdf(latents_last.cpu()) * -1
            e_time = time.perf_counter()
            time_meter.update(e_time - s_time, n=1)

            normal_map = F.interpolate(latents_last_l2.unsqueeze(0), size=(org_h, org_w), mode='bilinear', align_corners=False).squeeze(0)
            normal_maps.append(normal_map.cpu().numpy())
            normal_nlls.extend(nll.cpu().numpy())
            normal_diffs.extend(diffs.cpu().numpy())
            
            memory_meter.update(
                torch.cuda.max_memory_allocated() / (1024 * 1024), n=latents_last.shape[0]
            )
            
            # Save the samples for visualization
            org_imgs = images.cpu()
            org_imgs = denormalize(org_imgs)
            org_imgs = convert2image(org_imgs)
            normal_map = normal_map.cpu().numpy()
            for i in range(len(org_imgs)):
                samples_dict[f"{category}"].update({
                    f"{paths[i]}": {
                        "image": org_imgs[i],
                        "anomaly_map": normal_map[i],
                        "gt_mask": batch["masks"][i].cpu().numpy()
                    }
                })
                if anom_labels[i] == 1:
                    logger.info(f"Found anomalous sample: {paths[i]}")
            
        anomaly_diffs = []
        anomaly_nlls = []
        anomaly_maps = []
        anomaly_gt_masks = []
        
        for batch in tqdm(anom_loader, total=len(anom_loader)):
            images = batch["samples"].to(device)
            labels = batch["clslabels"].to(device)
            anomaly_gt_masks.append(batch["masks"])
            anom_labels = batch["labels"]
            
            torch.cuda.synchronize()
            s_time = time.perf_counter()
            features, _ = feature_extractor(images)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                latents_last = eval_denoiser.ddim_reverse_sample(
                    features, start_t, labels, eta=0.0
                )
            latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
            min_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).min(dim=1)[0]
            max_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).max(dim=1)[0]
            diffs = max_diffs_spatial - min_diffs_spatial
            nll = calculate_log_pdf(latents_last.cpu()) * -1
            e_time = time.perf_counter()
            
            time_meter.update(e_time - s_time, n=1)
    
            anomaly_map = F.interpolate(latents_last_l2.unsqueeze(0), size=(org_h, org_w), mode='bilinear', align_corners=False).squeeze(0)
            anomaly_maps.append(anomaly_map.cpu().numpy())
            anomaly_nlls.extend(nll.cpu().numpy())
            anomaly_diffs.extend(diffs.cpu().numpy())
            
            memory_meter.update(
                torch.cuda.max_memory_allocated() / (1024 * 1024), n=latents_last.shape[0]
            )
            
            # Save the samples for visualization
            org_imgs = images.cpu()
            org_imgs = denormalize(org_imgs)
            org_imgs = convert2image(org_imgs)
            anomaly_map = anomaly_map.cpu().numpy()
            paths = batch["filenames"]
            for i in range(len(org_imgs)):
                samples_dict[f"{category}"].update({
                    f"{paths[i]}": {
                        "image": org_imgs[i],
                        "anomaly_map": anomaly_map[i],
                        "gt_mask": batch["masks"][i].cpu().numpy()
                    }
                })
                
                if anom_labels[i] == 0:
                    logger.info(f"Found normal sample: {paths[i]}")
            
        
        logger.info(f"[{category}] Average time per sample: {time_meter.avg*1000:.4f} [ms]")
        logger.info(f"[{category}] Average memory usage: {memory_meter.avg:.2f} [MB]")
        metrics_dict[category]["latency"] = time_meter.avg * 1000  # Convert to milliseconds
        metrics_dict[category]["memory"] = memory_meter.avg  # in MB
        
        normal_diffs = np.array(normal_diffs)
        anomaly_diffs = np.array(anomaly_diffs)
        normal_nlls = np.array(normal_nlls)
        anomaly_nlls = np.array(anomaly_nlls)
        normal_maps = np.concatenate(normal_maps, axis=0)
        anomaly_maps = np.concatenate(anomaly_maps, axis=0)
        normal_gt_masks = torch.cat(normal_gt_masks, dim=0)
        anomaly_gt_masks = torch.cat(anomaly_gt_masks, dim=0)

        diffs_min = np.min([normal_diffs.min(), anomaly_diffs.min()])
        diffs_max = np.max([normal_diffs.max(), anomaly_diffs.max()])
        nlls_min = np.min([normal_nlls.min(), anomaly_nlls.min()])
        nlls_max = np.max([normal_nlls.max(), anomaly_nlls.max()])
        eps = 1e-8  # Small constant to avoid division by zero
        normal_diffs = (normal_diffs - diffs_min) / (diffs_max - diffs_min + eps) 
        anomaly_diffs = (anomaly_diffs - diffs_min) / (diffs_max - diffs_min + eps) 
        normal_nlls = (normal_nlls - nlls_min) / (nlls_max - nlls_min + eps)
        anomaly_nlls = (anomaly_nlls - nlls_min) / (nlls_max - nlls_min + eps)

        # Calculate Metrics for image-level
        y_true = np.concatenate([np.zeros(len(normal_diffs)), np.ones(len(anomaly_diffs))])
        normal_scores = normal_diffs + normal_nlls
        anomaly_scores = anomaly_diffs + anomaly_nlls
        y_score = np.concatenate([normal_scores, anomaly_scores])
        img_metrics = calculate_img_metrics(
            gt_labels=y_true,
            pred_scores=y_score,
            metrics=['img_auroc', 'img_aupr', 'img_f1max', 'img_ap'],
        )
        img_auroc = img_metrics["img_auroc"]
        img_ap = img_metrics["img_ap"]
        img_f1max = img_metrics["img_f1max"]
        
        metrics_dict[category]["I-AUROC"] = img_auroc
        metrics_dict[category]["I-AP"] = img_ap
        metrics_dict[category]["I-F1Max"] = img_f1max
        
        # Calculate Metics for pixel-level
        y_true_px = np.concatenate([
            normal_gt_masks.cpu().numpy().flatten(),
            anomaly_gt_masks.cpu().numpy().flatten()
        ])
        y_true_map = np.concatenate([
            normal_gt_masks.cpu().squeeze(1).numpy(),
            anomaly_gt_masks.cpu().squeeze(1).numpy()
        ])
        y_true_map = np.where(y_true_map > 0., 1, 0)  # Convert masks to binary
        y_score_map = np.concatenate([
            normal_maps, 
            anomaly_maps
        ])
        y_true_px = np.where(y_true_px > 0., 1, 0)  # Convert masks to binary
        
        px_metrics = calculate_px_metrics(
            gt_masks=y_true_map,
            pred_scores=y_score_map,
            metrics=['px_auroc', 'px_aupr', 'px_f1max', 'px_ap', 'px_aupro']
        )
        px_auroc = px_metrics["px_auroc"]
        px_ap = px_metrics["px_ap"]
        px_f1max = px_metrics["px_f1max"]
        px_pro = px_metrics["px_aupro"]
        
        metrics_dict[category]["P-AUROC"] = px_auroc
        metrics_dict[category]["P-AP"] = px_ap
        metrics_dict[category]["PRO"] = px_pro
        metrics_dict[category]["P-F1Max"] = px_f1max
        mad = np.mean([img_auroc, px_auroc, img_ap, px_ap, px_pro, img_f1max, px_f1max])
        metrics_dict[category]["mAD"] = mad
        
        logger.info(f"[{category}] Image AUC: {img_auroc} at epoch {epoch}")
        logger.info(f"[{category}] Pixel AUC: {px_auroc} at epoch {epoch}")
        logger.info(f"[{category}] Image AP: {img_ap} at epoch {epoch}")
        logger.info(f"[{category}] Pixel AP: {px_ap} at epoch {epoch}")
        logger.info(f"[{category}] PRO: {px_pro} at epoch {epoch}")
        logger.info(f"[{category}] Image F1Max: {img_f1max} at epoch {epoch}")
        logger.info(f"[{category}] Pixel F1Max: {px_f1max} at epoch {epoch}")
        logger.info(f"[{category}] mAD: {mad} at epoch {epoch}")    
        
        torch.cuda.empty_cache()
        
        if args.visualize_samples:
            logger.info(f"[{category}] Visualizing samples and anomaly maps")
            map_min, map_max = np.min(y_score_map), np.max(y_score_map)
            logger.info(f"[{category}] Anomaly map min: {map_min}, max: {map_max}")
            # Save the samples
            save_dir = os.path.join('samples', category)
            os.makedirs(save_dir, exist_ok=True)
            for path in tqdm(samples_dict[category]):
                sample = samples_dict[category][path]
                image = sample["image"]
                anomaly_map = sample["anomaly_map"]
                # Save the image and anomaly map
                fig, ax = plt.subplots(1, 3, figsize=(12, 6))
                ax[0].imshow(image)
                ax[0].set_title("Original Image")
                ax[0].axis('off')
                ax[1].imshow(anomaly_map, cmap='viridis', interpolation='nearest', vmin=map_min, vmax=map_max)
                ax[1].set_title("Anomaly Map")
                ax[1].axis('off')
                ax[2].imshow(sample["gt_mask"], cmap='gray', interpolation='nearest')
                ax[2].set_title("Ground Truth Mask")
                ax[2].axis('off')
                
                plt.tight_layout()
                filename = path.split(".png")[0].replace("/", "_")
                save_path = os.path.join(save_dir, f"{filename}.png")
                plt.savefig(save_path)
                plt.close(fig)
            logger.info(f"Saved sample images and anomaly maps to {save_dir}")
        
    logger.info(f"Evaluation completed for all categories.")
    return metrics_dict

import torch.distributed as dist
@torch.no_grad()
def concat_all_gather(array, world_size):
    world_size = dist.get_world_size()
    gather_list = [None] * world_size
    dist.all_gather_object(gather_list, array)  # Gather the arrays from all processes
    return np.concatenate(gather_list, axis=0)

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
    normal_diffs = []
    normal_nlls = []
    normal_maps = []
    normal_gt_masks = []
    losses = []
    for i, batch in enumerate(normal_loader):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        normal_gt_masks.append(batch["masks"])
        
        features, _ = feature_extractor(images)
        loss = denoiser(features, labels)
        losses.append(loss.cpu().numpy())
        
        latents_last = eval_denoiser.ddim_reverse_sample(
            features, start_t, labels, eta=0.0
        )
        latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
        min_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).min(dim=1)[0]  # (bs, )
        max_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).max(dim=1)[0]  # (bs, )
        diffs = min_diffs_spatial - max_diffs_spatial  # (bs, )
        nll = calculate_log_pdf(latents_last) * -1
        
        normal_map = F.interpolate(latents_last_l2.unsqueeze(0), size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False).squeeze(0)
        normal_maps.append(normal_map.cpu())
    
        normal_nlls.append(nll.cpu())
        normal_diffs.append(diffs.cpu())
    dist.barrier()  # Ensure all processes have completed the normal data processing
        
    anomaly_diffs = []
    anomaly_nlls = []
    anomaly_maps = []
    anomaly_gt_masks = []
    for i, batch in enumerate(anom_loader):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        anomaly_gt_masks.append(batch["masks"])
        
        features, _ = feature_extractor(images)
        loss = denoiser(features, labels)
        losses.append(loss.cpu().numpy())
        latents_last = eval_denoiser.ddim_reverse_sample(
            features, start_t, labels, eta=0.0
        )
        latents_last_l2 = torch.sum(latents_last ** 2, dim=1).sqrt()
        min_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).min(dim=1)[0]
        max_diffs_spatial = latents_last_l2.view(latents_last_l2.shape[0], -1).max(dim=1)[0]
        diffs = min_diffs_spatial - max_diffs_spatial
        nll = calculate_log_pdf(latents_last) * -1
        
        anomaly_map = F.interpolate(latents_last_l2.unsqueeze(0), size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False).squeeze(0)
        anomaly_maps.append(anomaly_map.cpu())
        anomaly_nlls.append(nll.cpu())
        anomaly_diffs.append(diffs.cpu())
        del latents_last, latents_last_l2, diffs, nll
        torch.cuda.empty_cache()                     
    dist.barrier()  # Ensure all processes have completed the anomaly data processing
    
    losses = np.array(losses)
    logger.info(f"[{category}] Loss: {losses.mean()} at epoch {epoch}")
    
    normal_diffs = torch.cat(normal_diffs, dim=0)  
    anomaly_diffs = torch.cat(anomaly_diffs, dim=0)
    normal_nlls = torch.cat(normal_nlls, dim=0)
    anomaly_nlls = torch.cat(anomaly_nlls, dim=0)
    normal_maps = torch.cat(normal_maps, dim=0)
    anomaly_maps = torch.cat(anomaly_maps, dim=0)
    normal_gt_masks = torch.cat(normal_gt_masks, dim=0).squeeze(1)  # Assuming masks are in shape (bs, 1, h, w)
    anomaly_gt_masks = torch.cat(anomaly_gt_masks, dim=0).squeeze(1)  # Assuming masks are in shape (bs, 1, h, w)
    
    # Gather results from all processes
    def to_numpy(tensor):
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    normal_diffs = concat_all_gather(normal_diffs, world_size)
    anomaly_diffs = concat_all_gather(anomaly_diffs, world_size)
    normal_nlls = concat_all_gather(normal_nlls, world_size)
    anomaly_nlls = concat_all_gather(anomaly_nlls, world_size)
    normal_maps = concat_all_gather(normal_maps, world_size)
    anomaly_maps = concat_all_gather(anomaly_maps, world_size)
    normal_gt_masks = concat_all_gather(normal_gt_masks, world_size)
    anomaly_gt_masks = concat_all_gather(anomaly_gt_masks, world_size)
    
    if rank != 0:
        return None
    
    logger.info(f"[{category}] Number of normal samples: {len(normal_diffs)}")
    logger.info(f"[{category}] Number of anomaly samples: {len(anomaly_diffs)}")
    diffs_min = np.min([normal_diffs.min(), anomaly_diffs.min()])
    diffs_max = np.max([normal_diffs.max(), anomaly_diffs.max()])
    nlls_min = np.min([normal_nlls.min(), anomaly_nlls.min()])
    nlls_max = np.max([normal_nlls.max(), anomaly_nlls.max()])
    eps = 1e-8  # Small constant to avoid division by zero
    normal_diffs = (normal_diffs - diffs_min) / (diffs_max - diffs_min + eps) 
    anomaly_diffs = (anomaly_diffs - diffs_min) / (diffs_max - diffs_min + eps) 
    normal_nlls = (normal_nlls - nlls_min) / (nlls_max - nlls_min + eps)
    anomaly_nlls = (anomaly_nlls - nlls_min) / (nlls_max - nlls_min + eps)

    y_true = np.concatenate([np.zeros(len(normal_diffs)), np.ones(len(anomaly_diffs))])
    normal_scores = normal_diffs + normal_nlls
    anomaly_scores = anomaly_diffs + anomaly_nlls
    y_score = np.concatenate([normal_scores, anomaly_scores])
    
    # Image-level metrics
    img_metrics = calculate_img_metrics(
        gt_labels=y_true,
        pred_scores=y_score,
        metrics=['img_auroc', 'img_aupr', 'img_f1max', 'img_ap'],
    )
    roc_auc = img_metrics["img_auroc"]
    ap = img_metrics["img_ap"]
    f1max_score = img_metrics["img_f1max"]
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
    px_metrics = calculate_px_metrics(
        gt_masks=y_true_map,
        pred_scores=y_score_map,
        metrics=['px_auroc', 'px_aupr', 'px_f1max', 'px_ap', 'px_aupro']
    )
    roc_auc_px = px_metrics["px_auroc"]
    ap_px = px_metrics["px_ap"]
    f1max_px_score = px_metrics["px_f1max"]
    pro = px_metrics["px_aupro"]
    
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
    def load_config(config_path):
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return config
    
    config = load_config(os.path.join(args.save_dir, "config.yaml"))
    main(config, args)