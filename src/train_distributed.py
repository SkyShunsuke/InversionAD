
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
import time

from datasets import build_dataset
from utils import get_optimizer, get_lr_scheduler
from denoiser import get_denoiser, Denoiser
from backbones import get_backbone, get_backbone_feature_shape
from evaluate import evaluate_dist

from einops import rearrange
from sklearn.metrics import roc_curve, roc_auc_score

import wandb
from dotenv import load_dotenv
import multiprocessing as mp
import logging
import torch.distributed as dist

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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(description="InversionAD Training")
    
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')
    parser.add_argument(
        "--devices", type=str, nargs="+", default=["cuda:0"],
    )
    parser.add_argument(
        "--port", type=int, default=29500,
    )
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

def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.info(exc)
    return config
    
def main(config):

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass
    
    # Initialize distributed training
    world_size, rank = init_distributed()
    logger.info(f"Initalized distributed training with world size {world_size} and rank {rank}")
    if rank > 0:
        logger.setLevel(logging.ERROR)
    else:
        # Load environment variables from .env file
        try: 
            load_dotenv()
            use_wandb = (os.getenv("WANDB_API_KEY") is not None)
            if use_wandb:
                wandb.login(key=os.getenv("WANDB_API_KEY"))
        except ImportError:
            pass
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
    seed = config['meta']['seed'] + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    dataset_config = config['data']
    device = torch.device('cuda:0')
    batch_size = config['data']['batch_size']
    train_dataset = build_dataset(**config['data'])
    dataset_config['train'] = False
    dataset_config['anom_only'] = True
    anom_dataset = build_dataset(**dataset_config)
    dataset_config['anom_only'] = False
    dataset_config['normal_only'] = True
    normal_dataset = build_dataset(**dataset_config)
    
    # anom_loaders = [DataLoader(anom_ds, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True) for anom_ds in anom_dataset.datasets]
    # normal_loaders = [DataLoader(normal_ds, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True) for normal_ds in normal_dataset.datasets]

    # train_loader = DataLoader(train_dataset, batch_size, shuffle=True, \
    #     pin_memory=config['data']['pin_memory'], num_workers=config['data']['num_workers'], drop_last=True, pin_memory=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank)
    anom_samplers = [torch.utils.data.distributed.DistributedSampler(
        dataset=anom_ds,
        num_replicas=world_size,
        rank=rank) for anom_ds in anom_dataset.datasets]
    normal_samplers = [torch.utils.data.distributed.DistributedSampler(
        dataset=normal_ds,
        num_replicas=world_size,
        rank=rank) for normal_ds in normal_dataset.datasets]
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
    )
    anom_loaders = [torch.utils.data.DataLoader(
        anom_ds,
        sampler=anom_sampler,
        batch_size=batch_size//world_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
    ) for anom_ds, anom_sampler in zip(anom_dataset.datasets, anom_samplers)]
    normal_loaders = [torch.utils.data.DataLoader(
        normal_ds,
        sampler=normal_sampler,
        batch_size=batch_size//world_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
    ) for normal_ds, normal_sampler in zip(normal_dataset.datasets, normal_samplers)]
    
    diff_in_sh = get_backbone_feature_shape(model_type=config['backbone']['model_type'])
    logger.info(f"Using input shape {diff_in_sh} for the diffusion model")
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=diff_in_sh)
    ema_decay = config['diffusion']['ema_decay']
    model_ema = copy.deepcopy(model)
    model.to(device)
    model_ema.to(device)
    
    model = torch.nn.parallel.DistributedDataParallel(model, static_graph=True)
    model_ema = torch.nn.parallel.DistributedDataParallel(model_ema, static_graph=True)
    for p in model_ema.parameters():
        p.requires_grad = False

    backbone_kwargs = config['backbone']
    logger.info(f"Using feature space reconstruction with {backbone_kwargs['model_type']} backbone")
    
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
    logger.info(f"Config is saved at {save_path}")
    
    model.train()
    logger.info(f"Steps per epoch: {len(train_loader)}")
    
    es_count = 0
    best_auc = 0
    import time

    for epoch in range(config['optimizer']['num_epochs']):
        for i, data in enumerate(train_loader):
            # --- timing start ---
            t0 = time.time()

            # Data loading
            t1 = time.time()
            img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
            img = img.to(device)
            labels = labels.to(device)
            t2 = time.time()

            # Feature extraction (forward pass of backbone)
            with torch.no_grad():
                x, x_list = feature_extractor(img)  # (B, c, h, w)
            t3 = time.time()

            # Model forward and backward
            loss = model(x, labels)  
            optimizer.zero_grad()
            loss.backward()
            if config['optimizer']['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['grad_clip'])
            optimizer.step()
            t4 = time.time()

            # Scheduler update
            scheduler.step()

            # EMA update
            for ema_param, model_param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1.0 - ema_decay)
            t5 = time.time()

            # Logging
            if i % config["logging"]["log_interval"] == 0 and rank == 0:
                logger.info(f"Epoch {epoch}, Iter {i}, Loss {loss.item():.4f}, LR {scheduler.get_last_lr():.6f}")
                # logger.info(
                #     f"Timings [ms] | Data: {(t2 - t1) * 1000:.2f} | Forward: {(t3 - t2) * 1000:.2f} | "
                #     f"Backward: {(t4 - t3) * 1000:.2f} | EMA+Sched: {(t5 - t4) * 1000:.2f} | Total: {(t5 - t0) * 1000:.2f}"
                # )
                tb_writer.add_scalar("Loss", loss.item(), epoch * len(train_loader) + i)
                if use_wandb:
                    wandb.log({
                        "Loss": loss.item(),
                        "LR": scheduler.get_last_lr(),
                        "Time/Data [ms]": (t2 - t1) * 1000,
                        "Time/Forward [ms]": (t3 - t2) * 1000,
                        "Time/Backward [ms]": (t4 - t3) * 1000,
                        "Time/Total [ms]": (t5 - t0) * 1000
                    })

        if (epoch + 1) % config["logging"]["save_interval"] == 0 and rank == 0:
            save_path = save_dir / f"model_latest.pth"
            torch.save(model.state_dict(), save_path)
            save_path = save_dir / f"model_ema_latest.pth"
            torch.save(model_ema.state_dict(), save_path)
            logger.info(f"Model is saved at {save_dir}")

        
        if (epoch + 1) % config["evaluation"]["eval_interval"] == 0:
            aucs_dict = {}
            for anom_loader, normal_loader in zip(anom_loaders, normal_loaders):
                logger.info(f"Evaluating on {anom_loader.dataset.category} dataset")
                auc = evaluate_dist(
                    model,
                    feature_extractor,
                    anom_loader,
                    normal_loader,
                    config, 
                    diff_in_sh,
                    epoch + 1,
                    config["evaluation"]["eval_step"],
                    device,
                    world_size=world_size,
                    rank=rank,
                )
                aucs_dict[anom_loader.dataset.category] = auc
            logger.info(f"AUCs: {aucs_dict}")
            
            
            if rank == 0:
                current_auc = np.mean(list(aucs_dict.values()))
                if current_auc > best_auc:
                    best_auc = current_auc
                    save_path = save_dir / f"model_best.pth"
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"Model is saved at {save_dir}")

                if use_wandb:
                    wandb.log({"AUC-avg": current_auc})
                    for cat, auc in aucs_dict.items():
                        wandb.log({f"AUC-{cat}": auc})
                logger.info(f"AUC: {current_auc} at epoch {epoch}")
            
            dist.barrier()  # wait for all processes to finish evaluation
    logger.info("Training is done!")
    tb_writer.close()
    
    # save model
    save_path = save_dir / "model_latest.pth"
    torch.save(model.state_dict(), save_path)
    save_path = save_dir / "model_ema_latest.pth"
    torch.save(model_ema.state_dict(), save_path)
    logger.info(f"Model is saved at {save_dir}")
    
def process_main(rank, fname, world_size, devices, port):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(":")[-1])
    
    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
        
    logging.info(f"called-params {fname}")  
    
    # load params
    parms = load_config(fname)
    logging.info("loaded params...")
    
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size), port=port)
    logger.info(f"Running... (rank: {rank}/{world_size})")
    main(parms)
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    
    num_gpus = len(args.devices)
    mp.set_start_method("spawn", True)
    
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_main,
            args=(rank, args.config_path, num_gpus, args.devices, args.port)
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
    logger.info("All processes finished.")


    
    
    
        
      
            
    
        
            
            
            
            
            
            
            
            
    
    
