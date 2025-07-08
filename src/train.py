
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

from src.datasets import build_dataset
from src.utils import get_optimizer, get_lr_scheduler
from src.denoiser import get_denoiser, Denoiser
from src.backbones import get_backbone, get_backbone_feature_shape
from src.evaluate import evaluate_inv

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

def parse_args():
    parser = argparse.ArgumentParser(description="InversionAD Training")
    
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')

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
    
def main(config):
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
    ema_decay = config['diffusion']['ema_decay']
    model_ema = copy.deepcopy(model)
    model.to(device)
    model_ema.to(device)

    backbone_kwargs = config['backbone']
    print(f"Using feature space reconstruction with {backbone_kwargs['model_type']} backbone")
    
    feature_extractor = get_backbone(**backbone_kwargs)
    feature_extractor.to(device).eval()

    optimizer = get_optimizer([model], **config['optimizer'])
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
    
    es_count = 0
    best_auc = 0
    for epoch in range(config['optimizer']['num_epochs']):
        for i, data in enumerate(train_loader):
            img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
            img = img.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                x, x_list = feature_extractor(img)  # (B, c, h, w)
                
            loss = model(x, labels)  
            
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
            current_auc = evaluate_inv(
                model,
                feature_extractor,
                anom_loader,
                normal_loader,
                config, 
                diff_in_sh,
                epoch + 1,
                config["evaluation"]["eval_step"],
                device,
            )["I-AUROC"]
            
            if current_auc > best_auc:
                best_auc = current_auc
                save_path = save_dir / f"model_best.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Model is saved at {save_dir}")

            if use_wandb:
                wandb.log({"AUC": current_auc})
            print(f"AUC: {current_auc} at epoch {epoch}")
            
            
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


    
    
    
        
      
            
    
        
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
