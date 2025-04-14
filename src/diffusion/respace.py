import math

import numpy as np
import torch
from torch import nn
from torch import Tensor
import enum

from typing import Tuple, List, Set

from .gaussian_diffusion import GaussianDiffusion

def space_timesteps(num_timesteps: int, section_counts: List[int]) -> Set[int]:
    """Create a list of timesteps to use from an original diffusion process, given the number of timesteps we want to take from 
    equally-sized portions of the original process. E.g., if there's 300 timesteps and the section counts are [10, 15, 20], then
    the first 100 steps are strided by 10, the next 100 steps are strided by 15, and the last 100 steps are strided by 20.
    If the stride is a string starting with 'ddim', then we assume that the number of sections is a divisor of the number of timesteps.
    E.g., if the number of timesteps is 128 and the stride is 'ddim64', then we stride by 2.
    
    Args:
        num_timesteps (int): the number of timesteps in the original diffusion process
        section_counts (List[int] | str): the number of timesteps to take from each section of the original process
    Returns:
        Set[int]: the timesteps to use
    """
    if isinstance(section_counts, str):
        if section_counts.startswith('ddim'):
            desired_count = int(section_counts[len('ddim'):])  # We assume one specified such like 'ddim128'
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"Could not find a divisor of {num_timesteps} that gives {desired_count} sections.")
        section_counts = [int(x) for x in section_counts.split(',')]  # We assume one specified such like '128,64,32'
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"Section {i} has size {size} which is less than the desired size {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else: 
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

class _WrappedModel:
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.original_num_steps = original_num_steps
        
    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=x.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]  # timesteps in an original diffusion process
        return self.model(x, new_ts, **kwargs)

class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion Process which can skip steps in a base diffusio process.
    """
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timesteps_map = []
        self.original_num_steps = len(kwargs['betas'])
        
        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timesteps_map.append(i)
        kwargs['betas'] = np.array(new_betas)
        super().__init__(**kwargs)
        
        self.base_diffusion = base_diffusion
        
    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timesteps_map, self.original_num_steps)
    
    def _scale_timesteps(self, t):
        return t
    
    def _get_original_timesteps(self, t):
        return torch.tensor(self.timesteps_map, device=t.device, dtype=t.dtype)[t]
    
    def p_mean_variance(
        self, model, *args, **kwargs
    ):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)
    
    def p_mean_variance_org(
        self, model, *args, **kwargs
    ):
        return super().p_mean_variance(model, *args, **kwargs)
    
    def q_sample(
        self, x_0, t, noise=None, **kwargs
    ):
        return super().q_sample(x_0, t, noise, **kwargs)
    
    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)
    
    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)
    
    def conditional_score(self, model, *args, **kwargs):
        return super().conditional_score(self._wrap_model(model), *args, **kwargs)
    
    
    
    
    
    
    