
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from tensorboardX import SummaryWriter

import numpy as np
from tqdm import tqdm
from pathlib import Path

import copy
import argparse
import yaml
from pprint import pprint

from src.datasets import build_dataset
from src.utils import get_optimizer, get_lr_scheduler, AverageMeter
from src.denoiser import get_denoiser, Denoiser
from src.backbones import get_backbone, get_backbone_feature_shape
from src.evaluate import evaluate_cm, evaluate_inv, evaluate_odm
from src.models import ODM

from einops import rearrange
from sklearn.metrics import roc_curve, roc_auc_score

import wandb
from dotenv import load_dotenv

# Load environment variables from .env file
try: 
    load_dotenv()
    use_wandb = (os.getenv("WANDB_API_KEY") is not None)
    if use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
except ImportError:
    pass

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

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def expand_tensor(tensor, target_shape):
    dims = len(target_shape)
    if tensor.dim() == dims:
        return tensor
    else:
        # add new dimensions to the tensor
        for _ in range(dims - tensor.dim()):
            tensor = tensor.unsqueeze(-1)
        return tensor
    
@torch.no_grad()
def extract_features(x, model):
    b, c, h, w = x.shape
    out = model.forward_features(x)  # (B, 197, d)
    out = out[:, 1:]  # remove the first token
    out = rearrange(out, 'b (h w) d -> b d h w', h=14, w=14)
    return out

def init_denoiser(num_inference_steps, device, config, in_sh, inherit_model=None):
    config["diffusion"]["num_sampling_steps"] = str(num_inference_steps)
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=in_sh)
    
    if inherit_model is not None:
        for p, p_inherit in zip(model.parameters(), inherit_model.parameters()):
            p.data.copy_(p_inherit.data)
    model.to(device).eval()
    return model
    
def main(config, args):
    pprint(config)
    
    if use_wandb:
        # create wandb project
        project = os.environ.get("WANDB_PROJECT")
        if project is None:
            raise ValueError("Please set the WANDB_PROJECT environment variable.")
        entity = os.environ.get("WANDB_ENTITY")
        if entity is None:
            raise ValueError("Please set the WANDB_ENTITY environment variable.")
        wandb.init(project=project, entity=entity, config=config)
    
    # set seed
    seed = config['meta']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    dataset_config = config['data']
    device = config['meta']['device']
    batch_size = config['data']['batch_size']
    train_dataset = build_dataset(**config['data'])
    dataset_config['train'] = False
    dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**dataset_config)
    dataset_config['anom_only'] = False
    dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**dataset_config)
    anom_loader = [DataLoader(anom_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)]
    normal_loader = [DataLoader(normal_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)]

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, \
        pin_memory=config['data']['pin_memory'], num_workers=config['data']['num_workers'], drop_last=True)

    diff_in_sh = get_backbone_feature_shape(model_type=config['backbone']['model_type'])
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    # odm: ODM = ODM(
    #     input_size=14,
    #     patch_size=1,
    #     in_channels=272,
    #     hidden_size=config['odm']['width'],
    #     depth=config['odm']['depth'],
    # )
    odm = get_denoiser(**config['odm'], input_shape=diff_in_sh)
    loss_type = config['diffusion']['loss_type']
    huber_c = config.get('diffusion', {}).get('huber_c', 0.1)

    checkpoint_path = os.path.join(args.save_dir, 'model_best.pth')
    model_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    is_multi_class = isinstance(normal_dataset, ConcatDataset) or isinstance(anom_dataset, ConcatDataset)
    is_multi_class = True
    if is_multi_class:
        # for distributed training, the model state dict keys may have a prefix
        # Remove the prefix if it exists
        if 'module.' in list(model_ckpt.keys())[0]:
            model_ckpt = {k.replace('module.', ''): v for k, v in model_ckpt.items()}
            
        if '_orig_mod.' in list(model_ckpt.keys())[0]:
            model_ckpt = {k.replace('_orig_mod.', ''): v for k, v in model_ckpt.items()}
    model.load_state_dict(model_ckpt, strict=True)
    print(f"Model is loaded from {checkpoint_path}")

    ema_decay = config['diffusion']['ema_decay']
    model_ema = copy.deepcopy(model)
    model_ema.to(device)
    model.to(device)
    odm.to(device)

    backbone_kwargs = config['backbone']
    print(f"Using feature space reconstruction with {backbone_kwargs['model_type']} backbone")
    
    feature_extractor = get_backbone(**backbone_kwargs)
    feature_extractor.to(device).eval()

    optimizer = get_optimizer([odm], **config['optimizer'])
    if config['optimizer']['scheduler_type'] == 'none':
        pass
    else:
        scheduler = get_lr_scheduler(optimizer, **config['optimizer'], iter_per_epoch=len(train_loader))

    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(save_dir))

    # save config
    save_path = save_dir / "config.yaml"
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config is saved at {save_path}")
    
    # Number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
    
    model.train()
    print(f"Steps per epoch: {len(train_loader)}")
    
    # First, collate all inversion features
    print("Extracting features from training dataset...")
    features_dict = {}
    model.eval()
    eval_step = config['evaluation']['eval_step']
    eval_denoiser = init_denoiser(eval_step, device, config, diff_in_sh, inherit_model=model)
    # temporaly disable batch drop
    train_loader.batch_sampler.drop_last = False
    for i, data in enumerate(tqdm(train_loader, desc="Extracting features")):
        img, labels = data["samples"], data["clslabels"]
        paths = data["filenames"]
        img = img.to(device)
        labels = labels.to(device)
        start_t = torch.tensor([0] * 8, device=device, dtype=torch.long)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            z, _ = feature_extractor(img)
            latents_last = eval_denoiser.ddim_reverse_sample(
                z, start_t, labels, eta=0.0
            )
        for j in range(len(img)):
            features_dict[paths[j]] = latents_last[j].cpu()
    train_loader.batch_sampler.drop_last = True
    print("Features extracted.")
    
    print(f"Evaluating Teacher Model...")
    metrics_dict = evaluate_inv(
        model,
        feature_extractor,
        anom_loader,
        normal_loader,
        config, 
        diff_in_sh,
        epoch=0,
        eval_step=eval_step,
        device=device,
    )
    print(f"Teacher Model Evaluation Metrics: {metrics_dict}")
    
    best_mad = 0
    meter = AverageMeter()
    for epoch in range(config['optimizer']['num_epochs']):
        for i, data in enumerate(train_loader):
            img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
            paths = data["filenames"]
            img = img.to(device)
            labels = labels.to(device)
            
            targets = []
            for j in range(len(img)):
                if paths[j] not in features_dict:
                    raise ValueError(f"Feature for {paths[j]} not found in the extracted features.")
                z = features_dict[paths[j]].to(device)
                targets.append(z)
            targets = torch.stack(targets, dim=0)  # (B, c, h, w)
            
            with torch.no_grad():
                z, _ = feature_extractor(img)  # (B, c, h, w)
            
            # prediction
            t = torch.tensor([0] * len(img), device=device, dtype=torch.long)
            preds = odm.net(z, t)  # (B, c, h, w)
            
            # compute the consistency loss
            if loss_type == "l2":
                loss = torch.nn.functional.mse_loss(preds.float(), targets.float(), reduction='mean')
            elif loss_type == "huber":
                loss = torch.mean(
                    torch.sqrt((preds.float() - targets.float()) ** 2 + huber_c**2) - huber_c
                )
            else:
                raise ValueError(f"Loss type {loss_type} is not supported.")
            
            # backward
            optimizer.zero_grad()
            loss.backward()
                    
            if config['optimizer']['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['grad_clip'])
            optimizer.step()
            scheduler.step()
            
            # update ema
            for ema_param, model_param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1.0 - ema_decay)
            
            if i % config["logging"]["log_interval"] == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss {loss.item()}")
                # print(f"z_std: {meter.avg:.4f}")
                meter.reset()   
                tb_writer.add_scalar("Loss", loss.item(), epoch * len(train_loader) + i)  
                if use_wandb:
                    wandb.log({"Loss": loss.item(), "LR": scheduler.get_last_lr()})

        if (epoch + 1) % config["logging"]["save_interval"] == 0:
            save_path = save_dir / f"model_latest.pth"
            torch.save(model.state_dict(), save_path)
            save_path = save_dir / f"model_ema_latest.pth"
            torch.save(model_ema.state_dict(), save_path)
            print(f"Model is saved at {save_dir}")
        
        if (epoch + 1) % config["evaluation"]["eval_interval"] == 0:
            current_mad = evaluate_odm(
                odm,
                feature_extractor,
                anom_loader,
                normal_loader,
                config, 
                diff_in_sh,
                epoch + 1,
                config["evaluation"]["eval_step"],
                device,
            )["mAD"]
            
            if current_mad > best_mad:
                best_mad = current_mad
                save_path = save_dir / f"model_best.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Model is saved at {save_dir}")

            if use_wandb:
                wandb.log({"mAD": current_mad})
            print(f"mAD: {current_mad} at epoch {epoch}")
            
    print("Training is done!")
    tb_writer.close()
    
    # save model
    save_path = save_dir / "model_latest.pth"
    torch.save(model.state_dict(), save_path)
    save_path = save_dir / "model_ema_latest.pth"
    torch.save(model_ema.state_dict(), save_path)
    print(f"Model is saved at {save_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)


    
    
    
        
      
            
    
        
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
