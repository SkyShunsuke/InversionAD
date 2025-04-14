import math

import numpy as np
import torch
from torch import Tensor
import enum

from typing import Tuple

def mean_flat(tensor: Tensor) -> Tensor:
    """
    Take the mean over all non-batch dimensions.
    Args:
        tensor (Tensor): tensor of shape (b, ...)
    Returns:
        Tensor: tensor of shape (b, )
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def extract_into_tensor(arr: np.ndarray, timesteps: Tensor, broadcast_shape: Tuple) -> Tensor:
    """Extract values from a 1-D numpy array for a batch of time indices.
    Args:
        arr (np.ndarray): 1-D numpy array of shape (T, )
        timesteps (Tensor): tensor of shape (b, )
        broadcast_shape (Tuple): shape to broadcast the extracted values to
    Returns:
        Tensor: tensor of shape (b, *broadcast_shape)
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)  # add a new dimension
    return res + torch.zeros(broadcast_shape, device=timesteps.device)
