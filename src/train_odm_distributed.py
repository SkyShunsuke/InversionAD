
import os
import sys
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import ConcatDataset
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from pathlib import Path

import copy
import argparse
import yaml
import time

from src.datasets import build_dataset
from src.utils import get_optimizer, get_lr_scheduler, init_distributed
from src.denoiser import get_denoiser, Denoiser
from src.backbones import get_backbone, get_backbone_feature_shape
import src.evaluate as evaluate

from einops import rearrange
from sklearn.metrics import roc_curve, roc_auc_score

import wandb
from dotenv import load_dotenv
import multiprocessing as mp
import logging
import torch.distributed as dist

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

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

def init_denoiser(num_inference_steps, device, config, in_sh, inherit_model=None):
    config["diffusion"]["num_sampling_steps"] = str(num_inference_steps)
    model: Denoiser = get_denoiser(**config['diffusion'], input_shape=in_sh)
    
    if inherit_model is not None:
        for p, p_inherit in zip(model.parameters(), inherit_model.parameters()):
            p.data.copy_(p_inherit.data)
    model.to(device).eval()
    return model

@torch.no_grad()
def concat_all_gather(array, world_size):
    world_size = dist.get_world_size()
    gather_list = [None] * world_size
    dist.all_gather_object(gather_list, array)  # CPU 配列をそのまま gather
    return np.concatenate(gather_list, axis=0)
    
def main(config, args):

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
    odm: Denoiser = get_denoiser(**config['odm'], input_shape=diff_in_sh)
    
    checkpoint_path = os.path.join(args.save_dir, 'model_best.pth')
    model_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    is_multi_class = isinstance(normal_dataset, ConcatDataset) or isinstance(anom_dataset, ConcatDataset)
    if is_multi_class:
        # for distributed training, the model state dict keys may have a prefix
        # Remove the prefix if it exists
        if 'module.' in list(model_ckpt.keys())[0]:
            model_ckpt = {k.replace('module.', ''): v for k, v in model_ckpt.items()}
            
        if '_orig_mod.' in list(model_ckpt.keys())[0]:
            model_ckpt = {k.replace('_orig_mod.', ''): v for k, v in model_ckpt.items()}
    model.load_state_dict(model_ckpt, strict=True)
    logger.info(f"Model is loaded from {checkpoint_path}")

    ema_decay = config['diffusion']['ema_decay']
    model.to(device)
    odm.to(device)
    
    model = torch.compile(model, fullgraph=True)
    odm = torch.compile(odm, fullgraph=True)
    model = torch.nn.parallel.DistributedDataParallel(model, static_graph=True)
    odm = torch.nn.parallel.DistributedDataParallel(odm, static_graph=True)
    for p in model.parameters():
        p.requires_grad = False

    backbone_kwargs = config['backbone']
    logger.info(f"Using feature space reconstruction with {backbone_kwargs['model_type']} backbone")
    
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
    logger.info(f"Config is saved at {save_path}")
    
    logger.info("Extracting features from training dataset...")
    # First, collate all inversion features
    print("Extracting features from training dataset...")
    features_dict = {}
    model.eval()
    eval_step = config['evaluation']['eval_step']
    eval_denoiser = init_denoiser(eval_step, device, config, diff_in_sh, inherit_model=model)
    # temprily disable dropping last batch
    train_loader.batch_sampler.drop_last = False
    for i, data in enumerate(tqdm(train_loader, desc="Extracting features")):
        img, labels = data["samples"], data["clslabels"]
        paths = data["filenames"]
        img = img.to(device)
        labels = labels.to(device)
        start_t = torch.tensor([0] * img.size(0), device=device, dtype=torch.long)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            z, _ = feature_extractor(img)
            latents_last = eval_denoiser.ddim_reverse_sample(
                z, start_t, labels, eta=0.0
            )
        for j in range(len(img)):
            features_dict[paths[j]] = latents_last[j].cpu()
    dist.barrier()  # wait for all processes to finish feature extraction
    train_loader.batch_sampler.drop_last = True
    local_features = features_dict  
    all_features = [None for _ in range(dist.get_world_size())]

    dist.all_gather_object(all_features, local_features)

    if dist.get_rank() == 0:
        merged_features = {}
        for fdict in all_features:
            merged_features.update(fdict)
    else:
        merged_features = None
    obj_list = [merged_features] if dist.get_rank() == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    merged_features = obj_list[0]
    logger.info("Features extracted.")
    
    logger.info(f"Evaluating Teacher model ...")
    teacher_results = {}
    categories = [ds.category for ds in anom_dataset.datasets]
    for anom_loader, normal_loader in zip(anom_loaders, normal_loaders):
        logger.info(f"Evaluating on {anom_loader.dataset.category} dataset")
        metrics_dict = evaluate.evaluate_dist(
            model,
            feature_extractor,
            anom_loader,
            normal_loader,
            config, 
            diff_in_sh,
            0,
            config["evaluation"]["eval_step"],
            device,
            world_size=world_size,
            rank=rank,
        )
        if rank == 0:
            teacher_results.update(metrics_dict)
        dist.barrier()  # wait for all processes to finish evaluation
    teacher_avg_results = {}
    keys = ["I-AUROC", "I-AP", "I-F1Max", "P-AUROC", "P-AP", "P-F1Max", "PRO", "mAD"]
    for key in keys:
        teacher_avg_results[key] = np.mean([teacher_results[cat][key] for cat in teacher_results.keys()])
    logger.info(f"Average results: {teacher_avg_results}")
    
    if rank == 0:
        # save to wandb 
        if use_wandb:
            for cat in categories:
                wandb.log({
                    f"{cat}/I-AUROC": teacher_results[cat]["I-AUROC"],
                    f"{cat}/I-AP": teacher_results[cat]["I-AP"],
                    f"{cat}/I-F1Max": teacher_results[cat]["I-F1Max"],
                    f"{cat}/P-AUROC": teacher_results[cat]["P-AUROC"],
                    f"{cat}/P-AP": teacher_results[cat]["P-AP"],
                    f"{cat}/P-F1Max": teacher_results[cat]["P-F1Max"],
                    f"{cat}/PRO": teacher_results[cat]["PRO"],
                    f"{cat}/mAD": teacher_results[cat]["mAD"]
                })
            wandb.log({
                "I-AUROC": teacher_avg_results["I-AUROC"],
                "I-AP": teacher_avg_results["I-AP"],
                "I-F1Max": teacher_avg_results["I-F1Max"],
                "P-AUROC": teacher_avg_results["P-AUROC"],
                "P-AP": teacher_avg_results["P-AP"],
                "P-F1Max": teacher_avg_results["P-F1Max"],
                "PRO": teacher_avg_results["PRO"],
                "mAD": teacher_avg_results["mAD"]
            })
    logger.info("Teacher model evaluation done.")
    
    model.train()
    logger.info(f"Steps per epoch: {len(train_loader)}")
    
    es_count = 0
    best_mad = 0
    import time

    for epoch in range(config['optimizer']['num_epochs']):
        for i, data in enumerate(train_loader):
            # --- timing start ---
            t0 = time.time()

            # Data loading
            t1 = time.time()
            img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
            paths = data["filenames"]
            img = img.to(device)
            labels = labels.to(device)
            t2 = time.time()
            
            # Prepare targets
            targets = []
            for j in range(len(img)):
                if paths[j] not in merged_features:
                    raise ValueError(f"Feature for {paths[j]} not found in the extracted features.")
                z = merged_features[paths[j]].to(device)
                targets.append(z)
            targets = torch.stack(targets, dim=0)  # (B, c, h, w)

            # Feature extraction (forward pass of backbone)
            with torch.no_grad():
                z, _ = feature_extractor(img)  # (B, c, h, w)
            t3 = time.time()

            start_t = torch.tensor([0] * img.size(0), device=device, dtype=torch.long)
            # prediction
            preds = odm.module.net(z, start_t)
            loss = torch.nn.functional.mse_loss(preds.float(), targets.float(), reduction='mean')
            
            optimizer.zero_grad()
            loss.backward()
            if config['optimizer']['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['grad_clip'])
            optimizer.step()
            t4 = time.time()

            # Scheduler update
            scheduler.step()

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
            logger.info(f"Model is saved at {save_dir}")

        
        if (epoch + 1) % config["evaluation"]["eval_interval"] == 0:
            all_results = {}
            categories = [ds.category for ds in anom_dataset.datasets]
            for anom_loader, normal_loader in zip(anom_loaders, normal_loaders):
                logger.info(f"Evaluating on {anom_loader.dataset.category} dataset")
                metrics_dict = evaluate.evaluate_dist_odm(
                    odm,
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
                if rank == 0:
                    all_results.update(metrics_dict)
                dist.barrier()  # wait for all processes to finish evaluation
            
            # Compute average AUC across all categories
            avg_results = {}
            keys = ["I-AUROC", "I-AP", "I-F1Max", "P-AUROC", "P-AP", "P-F1Max", "PRO", "mAD"]
            for key in keys:
                avg_results[key] = np.mean([all_results[cat][key] for cat in all_results.keys()])
            logger.info(f"Average results: {avg_results}")
            
            if rank == 0:
                current_mad = avg_results["mAD"]
                if current_mad > best_mad:
                    best_mad = current_mad
                    save_path = save_dir / f"model_best.pth"
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"Model is saved at {save_dir}")

                if use_wandb:
                    for cat in categories:
                        wandb.log({
                            f"{cat}/I-AUROC": all_results[cat]["I-AUROC"],
                            f"{cat}/I-AP": all_results[cat]["I-AP"],
                            f"{cat}/I-F1Max": all_results[cat]["I-F1Max"],
                            f"{cat}/P-AUROC": all_results[cat]["P-AUROC"],
                            f"{cat}/P-AP": all_results[cat]["P-AP"],
                            f"{cat}/P-F1Max": all_results[cat]["P-F1Max"],
                            f"{cat}/PRO": all_results[cat]["PRO"],
                            f"{cat}/mAD": all_results[cat]["mAD"]
                        })
                    
                    wandb.log({
                        "I-AUROC": avg_results["I-AUROC"],
                        "I-AP": avg_results["I-AP"],
                        "I-F1Max": avg_results["I-F1Max"],
                        "P-AUROC": avg_results["P-AUROC"],
                        "P-AP": avg_results["P-AP"],
                        "P-F1Max": avg_results["P-F1Max"],
                        "PRO": avg_results["PRO"],
                        "mAD": avg_results["mAD"]
                    })
                logger.info(f"mAD: {current_mad} at epoch {epoch}")
            
            dist.barrier()  # wait for all processes to finish evaluation
    logger.info("Training is done!")
    tb_writer.close()
    
    # save model
    save_path = save_dir / "model_latest.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model is saved at {save_dir}")


    
    
    
        
      
            
    
        
            
            
            
            
            
            
            
            
    
    
