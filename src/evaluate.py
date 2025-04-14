
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
from pprint import pprint

from datasets import build_dataset
from utils import get_optimizer, get_lr_scheduler
from denoiser import get_denoiser, Denoiser
from backbones import get_backbone

from einops import rearrange
from sklearn.metrics import roc_curve, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(description="InversionAD Inference")
    
    parser.add_argument('--save_dir', type=str, default=None, help='Path to the directory contais results')
    parser.add_argument('--eval_step', type=int, default=-1, help='Number of steps for evaluation')
    parser.add_argument('--use_ema_model', action='store_true', help='Use EMA model for evaluation')
    args = parser.parse_args()
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

def main(args):
    
    assert args.save_dir is not None, "Please provide a save directory"
    config_path = os.path.join(args.save_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    def load_config(config_path):
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return config
    
    config = load_config(config_path)
    pprint(config)

    dataset_config = config['data']
    device = config['meta']['device']
    
    dataset_config['train'] = False
    dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**dataset_config)
    dataset_config['anom_only'] = False
    dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**dataset_config)
    anom_loader = DataLoader(anom_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    normal_loader = DataLoader(normal_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    
    diff_in_sh = config['backbone']['feature_shape']
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    model.to(device).eval()

    backbone_kwargs = config['backbone']
    print(f"Using feature space reconstruction with {backbone_kwargs['model_type']} backbone")
    
    feature_extractor = get_backbone(**backbone_kwargs)
    feature_extractor.to(device).eval()
    
    # Load the model
    if args.use_ema_model:
        checkpoint_path = os.path.join(args.save_dir, 'model_ema_latest.pth')
    else:
        checkpoint_path = os.path.join(args.save_dir, 'model_latest.pth')
    
    model_ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_ckpt, strict=True)
    print(f"Loaded model from {checkpoint_path}")
    
    auc = evaluate(
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
    print(f"AUC: {auc}")
    
    
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

@torch.no_grad()
def evaluate(denoiser, feature_extractor, anom_loader, normal_loader, config, in_sh, epoch, eval_step, device):
    denoiser.eval()
    feature_extractor.eval()
    
    eval_denoiser = init_denoiser(eval_step, device, config, in_sh, inherit_model=denoiser)
    
    print(f"Evaluating on {len(anom_loader)} anomalous samples and {len(normal_loader)} normal samples")
    print(f"Evaluation step: {eval_step}")
    print(f"Epoch: {epoch}")
    
    start_t = torch.tensor([0] * 8, device=device, dtype=torch.long)
    normal_ats = []
    normal_nlls = []
    losses = []
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
        nll = calculate_log_pdf(latents_last.cpu()) * -1
    
        normal_nlls.extend(nll.cpu().numpy())
        normal_ats.extend(ats.cpu().numpy())
        
    anomaly_ats = []
    anomaly_nlls = []
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
        nll = calculate_log_pdf(latents_last.cpu()) * -1
        
        anomaly_nlls.extend(nll.cpu().numpy())
        anomaly_ats.extend(ats.cpu().numpy())
    
    losses = np.array(losses)
    print(f"Loss: {losses.mean()} at epoch {epoch}")
    
    normal_ats = np.array(normal_ats)
    anomaly_ats = np.array(anomaly_ats)
    normal_nlls = np.array(normal_nlls)
    anomaly_nlls = np.array(anomaly_nlls)
    

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
    
    roc_auc = roc_auc_score(y_true, y_score)
    return roc_auc

if __name__ == "__main__":
    args = parse_args()
    main(args)