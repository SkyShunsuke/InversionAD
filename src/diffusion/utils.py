import math

import numpy as np
import torch
from torch import Tensor    
import enum

def normal_kl(mean1: Tensor, logvar1: Tensor, mean2: Tensor, logvar2: Tensor) -> Tensor:
    """Compute KL divergence between two gaussians with diagonal covariance matrices.
    Args:
        mean1 (Tensor): Mean of the first gaussian.
        logvar1 (Tensor): Log variance of the first gaussian.
        mean2 (Tensor): Mean of the second gaussian.
        logvar2 (Tensor): Log variance of the second gaussian.
    Returns:
        Tensor: KL divergence between the two gaussians.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, Tensor):
            tensor = obj
            break
    assert tensor is not None, "At least one of the inputs must be a tensor"
    
    logvar1, logvar2 = [
        x if isinstance(x, Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]    
    
    return 0.5 * (
        -1.0  # constant term
        + logvar2 # variance term (we assume these variance terms always non-negative)
        - logvar1
        + torch.exp(logvar1 - logvar2)  # trace term
        + ((mean2 - mean1) ** 2) * torch.exp(-logvar2)  # mean term
    )

def approx_standard_normal_cdf(x: Tensor) -> Tensor:
    """A fast approximation of the standard normal CDF.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x: Tensor, *, means: Tensor, log_scales: Tensor) -> Tensor:
    """Compute the log-liklihood of a Gaussian distribution discretizing to a given image.
    Args:
        x (Tensor): Input tensor of shape (b, c, h, w), we assume it has uint8 values, rescaled to [-1, 1]
        means (Tensor): Mean tensor of shape (b, c, h, w)
        log_scales (Tensor): Log scale tensor of shape (b, c, h, w)
    Returns:
        Tensor: Log likelihood tensor of shape (b, c, h, w)
    """
    assert x.shape == means.shape == log_scales.shape, "Input shapes must match"
    # TODO: 
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(torch.clamp(cdf_delta, min=1e-12))
        )
    )
    assert log_probs.shape == x.shape, "Log probs shape must match input shape"
    return log_probs
    
    

    
    
    
