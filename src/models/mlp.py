import math

import numpy as np
import torch
from torch import nn
from torch import Tensor
import enum

from typing import Tuple

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embed_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embed_size = frequency_embed_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embed_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create a sinusodial timestep embedding.
        Args:
            t (Tensor): tensor of shape (B, )
            dim (int): embedding dimension
            max_period (int): maximum period
        Returns:
            Tensor: tensor of shape (B, D)
        """
        half = dim // 2
        freqs = torch.exp(
            -1 * math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)  # (B, 2/D)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, D)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # (B, D)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embed_size)  # (B, D)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
    ):
        super().__init__()
        self.channels = channels
        
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )
    
    def forward(self, x, y):
        # conditioning
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)  # (B, C)
        shift_mlp = shift_mlp.unsqueeze(1)  # (B, 1, C)
        scale_mlp = scale_mlp.unsqueeze(1)  # (B, 1, C)
        gate_mlp = gate_mlp.unsqueeze(1)  # (B, 1, C)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)  # (B, N, C)
        h = self.mlp(h)  # (B, N, C)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=True, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        shift = shift.unsqueeze(1)  # (B, 1, C)
        scale = scale.unsqueeze(1)  # (B, 1, C)
        x = modulate(self.norm_final(x), shift, scale)
        # Apply final linear layer
        x = self.linear(x)
        return x

class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        input_size: int,
        in_channels: int,   
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_blocks: int,
        patch_size: int = 1,
        grad_checkpoint: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.num_blocks = num_blocks
        self.grad_checkpoint = grad_checkpoint
        
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, model_channels)
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(ResBlock(model_channels))
        self.res_blocks = nn.ModuleList(res_blocks)
        
        self.final_layer = FinalLayer(model_channels, out_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)        

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        
        # Zero out adaLN modulation weights
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """Convert sequence of tokens into image-like tensor.
        Args:
            x (Tensor): tensor of shape (B, N, patch_size*patch_size*in_channels)
        Returns:
            Tensor: tensor of shape (B, C, H, W)
        """
        C = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p, p2=p, c=C)
        return x
        
    def forward(self, x, t, y=None, return_tokens=False, **kwargs):
        """Apply model to an input batch
        Args:
            x (Tensor): input tensor (B, C)
            t (Tensor): timestep tensor (B, )
            y (Tensor): condition tensor (B, Z)
        Returns:
            Tensor: output tensor (B, C)
        """
        x = self.x_embedder(x)  # (B, N, C)
        t = self.time_embed(t)
        if y is not None:
            y = self.cond_embed(y)
            y = t + y
        else:
            y = t
        if self.grad_checkpoint and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)
        
        x = self.final_layer(x, y)
        if return_tokens:
            return x  # (B, N, C)
        else:
            return self.unpatchify(x)  # (B, C, H, W)
        
    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[:len(x)//2]  # (B/2, C)
        combined = torch.cat([half, half], dim=0)  # (B, C)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]  
        cond_eps, uncond_eps = torch.split(eps, len(eps)//2, dim=0)  # (B/2, C)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
        