
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
from tqdm import tqdm
from pathlib import Path

import copy
import argparse
import yaml

from datasets import build_dataset
from utils import get_optimizer, get_lr_scheduler
from denoiser import get_denoiser, Denoiser
from backbones import get_backbone

from einops import rearrange
from sklearn.metrics import roc_curve, roc_auc_score

from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F

import torch.multiprocessing as mp

import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

MAX_BATCH_SIZE = 64
NUM_WORKERS = 1

def parse_args():
    parser = argparse.ArgumentParser(description="InversionAD Inference")
    
    parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'], help='List of devices to use for distributed evaluation')
    parser.add_argument('--eval_strategy', type=str, default='inversion', choices=['inversion', 'reconstruction'], help='Evaluation strategy: inversion or reconstruction')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to the directory contais results')
    parser.add_argument('--eval_step', type=int, default=-1, help='Number of steps for evaluation')
    parser.add_argument('--noise_step', type=int, default=8, help='Number of noise steps for evaluation')
    parser.add_argument('--use_ema_model', action='store_true', help='Use EMA model for evaluation')
    parser.add_argument('--port', type=int, default=12345, help='Port for distributed training')
    args = parser.parse_args()
    return args

def get_unused_indices(dataset, batch_size, world_size, rank):
    dataset_len = len(dataset)
    num_batches = dataset_len // (batch_size * world_size)
    samples_per_rank = num_batches * batch_size

    all_indices = list(range(dataset_len))
    start = rank * samples_per_rank
    end = start + samples_per_rank
    used_indices = all_indices[start:end]

    return used_indices

def get_dropped_indices(dataset, batch_size):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    dataset_len = len(dataset)

    local_used = get_unused_indices(dataset, batch_size, world_size, rank)
    local_used_tensor = torch.tensor(local_used, dtype=torch.long, device="cuda")

    max_len = (dataset_len // (batch_size * world_size)) * batch_size
    padded = torch.full((max_len,), -1, dtype=torch.long, device="cuda")
    padded[:len(local_used_tensor)] = local_used_tensor

    gathered = [torch.empty_like(padded, device="cuda") for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    used_indices = torch.cat(gathered).tolist()
    used_indices = set(i for i in used_indices if i >= 0)

    all_indices = set(range(dataset_len))
    dropped_indices = sorted(all_indices - used_indices)

    # if rank == 0:
    #     print(f"[INFO] Dropped sample indices: {dropped_indices}")
    
    return dropped_indices if rank == 0 else []

def init_distributed(port=12345, rank_and_world_size=(None, None)):
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    
    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'
    
    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            logger.info('SLURM vars not set (distributed training not available)')
            world_size, rank = 1, 0
            return world_size, rank
    
    try:
        os.environ['MASTER_PORT'] = str(port)
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=world_size,
            rank=rank)
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f'distributed training not available {e}')
    
    return world_size, rank

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

def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.info(exc)
    return config

def main(params, args):

    dataset_config = params['data']
    device = params['meta']['device']
    
    dataset_config['train'] = False
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
                num_workers=NUM_WORKERS,
                sampler=DistributedSampler(anom_ds, shuffle=False, drop_last=True)
            )
            for anom_ds in anom_dataset.datasets
        ]
        normal_loader = [
            DataLoader(
                normal_ds,
                batch_size=MAX_BATCH_SIZE,
                num_workers=NUM_WORKERS,
                sampler=DistributedSampler(normal_ds, shuffle=False, drop_last=True)
            )
            for normal_ds in normal_dataset.datasets
        ]
    else:
        logger.info(f"Using single-class dataset: {anom_dataset.category}")
        anom_loader = [
            DataLoader(
                anom_dataset,
                batch_size=MAX_BATCH_SIZE,
                num_workers=NUM_WORKERS,
                sampler=DistributedSampler(anom_dataset, shuffle=False, drop_last=True)
            )
        ]
        normal_loader = [
            DataLoader(
                normal_dataset,
                batch_size=MAX_BATCH_SIZE,
                num_workers=NUM_WORKERS,
                sampler=DistributedSampler(normal_dataset, shuffle=False, drop_last=True)
            )
        ]
    diff_in_sh = (272, 16, 16)  # For EfficientNet-b4
    model: Denoiser = get_denoiser(**params['diffusion'], input_shape=diff_in_sh)
    model.to(device).eval()

    backbone_kwargs = params['backbone']
    logger.info(f"Using feature space reconstruction with {backbone_kwargs['model_type']} backbone")
    
    feature_extractor = get_backbone(**backbone_kwargs)
    feature_extractor.to(device).eval()
    
    # Load the model
    if args.use_ema_model:
        checkpoint_path = os.path.join(args.save_dir, 'model_ema_latest.pth')
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
    
    if args.eval_strategy == 'reconstruction':
        logger.info("Evaluating reconstruction performance")
        assert args.noise_step < args.eval_step, "Noise step should be less than evaluation step for reconstruction"
        pass
    elif args.eval_strategy == 'inversion':
        auc_dict_all = {}
        for anom_loader, normal_loader in zip(anom_loader, normal_loader):
            dist.barrier()  # Ensure all processes have completed the evaluation
            auc_dict = evaluate_dist(
                model,
                feature_extractor,
                anom_loader,
                normal_loader,
                params, 
                diff_in_sh,
                "Eval",
                params["evaluation"]["eval_step"] if args.eval_step == -1 else args.eval_step,
                device,
                world_size=dist.get_world_size(),
                rank=dist.get_rank()
            )
            if auc_dict is None:
                continue
            auc_dict_all.update(auc_dict)
    
    if dist.get_rank() == 0:
        logger.info(f"{auc_dict_all}")
        # Compute Average AUC
        if is_multi_class:
            avg_auc = np.mean(list(auc_dict_all.values()))
            logger.info(f"Average AUC: {avg_auc}")
    
    
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
    auc_dict = {category: {}}
    
    eval_denoiser = init_denoiser(eval_step, device, config, in_sh, inherit_model=denoiser)
    
    logger.info(f"[{category}] Evaluating on {len(anom_loader)} anomalous samples and {len(normal_loader)} normal samples")
    logger.info(f"[{category}] Evaluation step: {eval_step}")
    logger.info(f"[{category}] Epoch: {epoch}")
    
    start_t = torch.tensor([0] * 8, device=device, dtype=torch.long)
    normal_ats = []
    normal_nlls = []
    normal_maps = []
    normal_gt_masks = []
    losses = []
    
    drop_indices_normal = get_dropped_indices(normal_loader.dataset, MAX_BATCH_SIZE)
    drop_indices_anomaly = get_dropped_indices(anom_loader.dataset, MAX_BATCH_SIZE)
    logger.info(f"[{category}] Dropped indices for normal data: {drop_indices_normal}")
    logger.info(f"[{category}] Dropped indices for anomaly data: {drop_indices_anomaly}")
    
    for i, batch in enumerate(normal_loader):
        images = batch["samples"].to(device)
        labels = batch["clslabels"].to(device)
        
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
        normal_gt_masks.append(batch["masks"].to(device))
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
        anomaly_gt_masks.append(batch["masks"].to(device))
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
    normal_gt_masks = torch.cat(normal_gt_masks, dim=0)
    anomaly_gt_masks = torch.cat(anomaly_gt_masks, dim=0)
    
    # Gather results from all processes
    def to_numpy(tensor):
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    normal_ats = to_numpy(concat_all_gather(normal_ats, world_size))
    anomaly_ats = to_numpy(concat_all_gather(anomaly_ats, world_size))
    normal_nlls = to_numpy(concat_all_gather(normal_nlls, world_size))
    anomaly_nlls = to_numpy(concat_all_gather(anomaly_nlls, world_size))
    normal_maps = to_numpy(concat_all_gather(normal_maps, world_size))
    anomaly_maps = to_numpy(concat_all_gather(anomaly_maps, world_size))
    normal_gt_masks = to_numpy(concat_all_gather(normal_gt_masks, world_size)).astype(bool)
    anomaly_gt_masks = to_numpy(concat_all_gather(anomaly_gt_masks, world_size)).astype(bool) 
    
    # Process dropped indices with single process
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
    y_true_px = np.concatenate([
        normal_gt_masks.flatten(),
        anomaly_gt_masks.flatten()
    ])
    normal_scores = normal_ats + normal_nlls
    anomaly_scores = anomaly_ats + anomaly_nlls
    y_score = np.concatenate([normal_scores, anomaly_scores])
    y_score_px = np.concatenate([
        normal_maps.flatten(),
        anomaly_maps.flatten()
    ])
    
    roc_auc = roc_auc_score(y_true, y_score)
    # roc_auc_px = roc_auc_score(y_true_px, y_score_px)
    auc_dict[category]['roc_auc'] = roc_auc
    # auc_dict[category]['roc_auc_px'] = roc_auc_px
    return auc_dict

def process_main(rank, save_dir, world_size, devices, port, args):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(":")[-1])
    
    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
        
    assert save_dir is not None, "Please provide a save directory"
    config_path = os.path.join(save_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    params = load_config(config_path)
    logging.info("loaded params...")
    
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size), port=port)
    logger.info(f"Running... (rank: {rank}/{world_size})")
    main(params, args)
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    
    num_gpus = len(args.devices)
    mp.set_start_method("spawn", True)
    
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_main,
            args=(rank, args.save_dir, num_gpus, args.devices, args.port, args)
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
    logger.info("All processes finished.")