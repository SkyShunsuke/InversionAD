
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
from src.evaluate import evaluate_cm, evaluate_inv

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
    parser = argparse.ArgumentParser(description="Consistency Model Training")
    
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to the config file')

    args = parser.parse_args()
    return args

def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0

def predicted_last(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
        pred_x_t = alphas * pred_x_0 + sigmas * model_output
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
        pred_x_t = (sample + pred_x_0 * model_output) / alphas
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_t

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

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
    
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0, total_steps=1000):
    timestep = total_steps - (timestep + 1)  # convert to reverse timestep
    ts_scaled = timestep / timestep_scaling
    c_skip = sigma_data**2 / (ts_scaled ** 2 + sigma_data**2)
    c_out = ts_scaled / (ts_scaled ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out

class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_next = torch.concat([self.ddim_alpha_cumprods[1:], torch.tensor([0.0])])
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        self.ddim_alpha_cumprods_next = self.ddim_alpha_cumprods_next.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev
    
    def ddim_reverse_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_next = extract_into_tensor(self.ddim_alpha_cumprods_next, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_next).sqrt() * pred_noise
        x_next = alpha_cumprod_next.sqrt() * pred_x0 + dir_xt
        return x_next

@torch.no_grad()
def extract_features(x, model):
    b, c, h, w = x.shape
    out = model.forward_features(x)  # (B, 197, d)
    out = out[:, 1:]  # remove the first token
    out = rearrange(out, 'b (h w) d -> b d h w', h=14, w=14)
    return out
    
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

    # params
    T = config['diffusion']['T']  # default: 1000
    N = config['diffusion']['N']  # default: 10
    ts = config['diffusion']['timestep_scaling']  # default: 1000
    loss_type = config['diffusion']['loss_type']  # default: 'huber'
    sigma_data = config['diffusion'].get('sigma_data', 0.5)  # default: 0.5
    huber_c = config['diffusion'].get('huber_c', 0.001)  # default: 0.001
    print(f"Using T: {T}, N: {N}, timestep scaling: {ts}, loss type: {loss_type}")
    print(f"Sigma_data: {sigma_data}")
    
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

    checkpoint_path = os.path.join(args.save_dir, 'model_latest.pth')
    model_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    is_multi_class = True # isinstance(normal_dataset, ConcatDataset) or isinstance(anom_dataset, ConcatDataset)
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

    solver = DDIMSolver(
        model.train_diffusion.alphas_cumprod,
        timesteps=T,
        ddim_timesteps=N
    ).to(device)
    
    alpha_schedule = torch.sqrt(torch.from_numpy(model.train_diffusion.alphas_cumprod)).to(device)
    sigma_schedule = torch.sqrt(1 - torch.from_numpy(model.train_diffusion.alphas_cumprod)).to(device)

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
    best_mad = 0
    meter = AverageMeter()
    for epoch in range(config['optimizer']['num_epochs']):
        for i, data in enumerate(train_loader):
            img, labels = data["samples"], data["clslabels"]    # (B, C, H, W), (B,)
            img = img.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                z, _ = feature_extractor(img)  # (B, c, h, w)
                # z_std = z.std(dim=[2, 3], keepdim=True)
                # z_mean = z.mean(dim=[2, 3], keepdim=True)
                # meter.update(z_std.mean().item())
            
            # First sample timesteps from uniform distribution
            # topk = T // N
            # index = torch.randint(0, N, (img.shape[0],), device=device).long()  # (B,)
            # max_timestep = solver.ddim_timesteps[N - 1]
            # start_t = solver.ddim_timesteps[index]  # (B,)
            # t = torch.clamp(start_t + topk, min=0, max=max_timestep)
            t_index = torch.randint(0, len(solver.ddim_timesteps), (img.shape[0],), device=device)
            t       = solver.ddim_timesteps[t_index]
            
            
            # coefs
            # c_skip_start, c_out_start= scalings_for_boundary_conditions(start_t, sigma_data=sigma_data, timestep_scaling=ts)
            # c_skip_start, c_out_start = [append_dims(x, z.ndim) for x in [c_skip_start, c_out_start]]
            # c_skip, c_out = scalings_for_boundary_conditions(t, sigma_data=sigma_data, timestep_scaling=ts)
            # c_skip, c_out = [append_dims(x, z.ndim) for x in [c_skip, c_out]]
            
            # Then, sample x_n sample 
            z_t = model.q_sample(z, t).float()
            
            # online predition
            noise_pred = model.net(z_t, t)  # (B, c, h, w)
            pred_x_T = predicted_last(
                noise_pred,
                t,
                z_t,
                "epsilon",
                alpha_schedule,
                sigma_schedule
            )
            model_pred = pred_x_T # c_skip_start * z_t + c_out_start * pred_x_T
            
            # Target input prediction
            with torch.no_grad():
                # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                teacher_pred = model_ema.net(z_t, t)  # (B, c, h, w)
                pred_x_0 = predicted_origin(
                    teacher_pred,
                    t,
                    z_t,
                    "epsilon",
                    alpha_schedule,
                    sigma_schedule
                )
                # x_prev = solver.ddim_step(pred_x0, teacher_pred, index).float()
                x_next = solver.ddim_reverse_step(pred_x_0, teacher_pred, t_index).float()
            
            # Get target 
            with torch.no_grad():
                # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                target_noise_pred = model_ema.net(x_next, t)  # (B, c, h, w)
                pred_x_T = predicted_last(
                    target_noise_pred,
                    t,
                    x_next,
                    "epsilon",
                    alpha_schedule,
                    sigma_schedule
                )
                boundary_timesteps = torch.tensor([len(solver.ddim_timesteps)] * z.shape[0], device=device, dtype=torch.long)
                boundary_mask = append_dims((t == boundary_timesteps), z.ndim)
                target = boundary_mask.float() * x_next + (~boundary_mask).float() * pred_x_T

            # compute the consistency loss
            if loss_type == "l2":
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction='mean')
            elif loss_type == "huber":
                loss = torch.mean(
                    torch.sqrt((model_pred.float() - target.float()) ** 2 + huber_c**2) - huber_c
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
            # for ema_param, model_param in zip(model_ema.parameters(), model.parameters()):
            #     ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1.0 - ema_decay)
            
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
            current_mad = evaluate_cm(
                model,
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


    
    
    
        
      
            
    
        
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
