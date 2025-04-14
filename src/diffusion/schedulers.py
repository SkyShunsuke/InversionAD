import math

import numpy as np
import torch
import enum

def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    """Return beta schedule with linear warmup from beta_start to beta_end.
    Args:
        beta_start: Starting beta value.
        beta_end: Ending beta value.
        num_diffusion_timesteps: Number of diffusion timesteps (T).
        warmup_frac: Fraction of timesteps to linearly increase beta.
    Returns:
        betas: Beta values at each timestep.
    Examples:
        >>> _warmup_beta(0.1, 0.2, 1000, 0.1)
        array([0.101, 0.102, 0.103, ..., 0.2, 0.2, 0.2])
    """
    
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """Return beta values at each timestep.
    Args:
        beta_schedule: Beta schedule type.
        beta_start (kwargs): Starting beta value.
        beta_end (kwargs): Ending beta value.
        num_diffusion_timesteps (kwargs): Number of diffusion timesteps (T).
    Returns:
        betas: Beta values at each timestep. (T,)
    """
    
    if beta_schedule == 'quad':
        betas = (
            np.linspace(
                beta_start ** 0.5, 
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64
            )
            ** 2
        )   # beta quadratically increases from beta_start to beta_end
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':
        betas = 1 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Create a beta schedule that discretizes the given alpha_bar function,
    which defines the cumulative product of (1 - beta) over time from t = [0, 1].
    Args:
        num_diffusion_timesteps: Number of diffusion timesteps (T).
        alpha_bar: Cumulative product of (1 - beta) over time.
        max_beta: Maximum beta value.
    Returns:
        betas: Beta values at each timestep. (T,)
    """
    
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """Return beta values at each timestep. 
    Args:
        schedule_name: Beta schedule type.
        num_diffusion_timesteps: Number of diffusion timesteps (T).
    Returns:
        betas: Beta values at each timestep. (T,)
    """
    if schedule_name == 'linear':
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            'linear',
            beta_start = scale * 0.0001,
            beta_end = scale * 0.02,
            num_diffusion_timesteps = num_diffusion_timesteps
        )
        # It extended to the case if some latent variables are skipped with the same duration: 
        # i.e. 100 step sampling with original 1000 step diffusion, beta start with beta_min * 1000 / 100.
    elif schedule_name == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.0008) / 1.008 * math.pi / 2) ** 2
        )  # alpha_bar(t) = cos^2(pi/2 * (t + 0.0008) / 1.008), t = [0, 1]
    else:
        return NotImplementedError(schedule_name)

