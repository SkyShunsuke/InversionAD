"""Assymetric Encoder and Predictor vision transformer models.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



def indices_to_mask(mask_indices, L):
    """Convert indices to binary mask keeping it's orders.
    Args:
        masks_indices (tensor): The indices of masked patches. shape of (B, M), where M is the number of masked patches.
        L (int): The total number of patches.
    Returns:
        mask (tensor): The binary mask. shape of (B, L), where L is the number of patches.
    """
    B, M = mask_indices.shape
    masks = torch.zeros(B, L, device=mask_indices.device, dtype=torch.long)
    for b in range(B):
        for order, idx in enumerate(mask_indices[b]):
            masks[b, idx] = order + 1 
    masks = (masks != 0)
    return masks.bool()

def mask_to_indices(masks):
    """Convert binary mask to indices keeping it's orders.
    Args:
        masks (tensor): The binary mask. shape of (B, L), where L is the number of patches.
    Returns:
        mask_indices (tensor): The indices of masked patches. shape of (B, M), where M is the number of masked patches.
    """
    B, L = masks.shape
    
    masks = masks.long()
    mask_indices_list = []
    for b in range(B):
        row = masks[b] 
        nonzero_positions = torch.nonzero(row, as_tuple=False).squeeze(1)
        order_values = row[nonzero_positions]
        _, sorted_idx = torch.sort(order_values)
        sorted_positions = nonzero_positions[sorted_idx]
        mask_indices_list.append(sorted_positions)

    mask_indices = torch.stack(mask_indices_list, dim=0)
    
    return mask_indices

def get_unmasked_indices(masked_indices, L):
    mask = indices_to_mask(masked_indices, L)
    inv_mask = ~mask
    unmasked_indices = mask_to_indices(inv_mask)
    return unmasked_indices

class PatchEmbed(nn.Module):
    def __init__(
        self, in_channels, patch_size, emb_size: int = 768
    ):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_size = emb_size
        
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embedding
        Args:
            x (torch.Tensor): tensor of shape (B, c, h, w)
        Returns:
            torch.Tensor: tensor of shape (B, L, d)
        """
        x = self.proj(x)  # (b, e, h, w)
        h, w = x.shape[2:]  
        x = rearrange(x, 'b c h w -> b (h w) c', h=h, w=w)  
        return x

class FeedForwardBlock(nn.Module):  
    def __init__(
        self, in_channels, hidden_channels, out_channels
    ):
        super(FeedForwardBlock, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d) -> (B, L, d)
        return self.net(x)
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, emb_size, num_heads=8):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.to_out = nn.Linear(emb_size, emb_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Multi-head attention block
        Args:
            x (torch.Tensor): tensor of shape (B, L, d)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tensor of shape (B, L, d), tensor of shape (B, h, L, L)
        """
        # x: (B, L, d)
        b, n, e = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 3 * (B, L, d)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)  # (B, h, L, d)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(dots, dim=-1)  # (B, h, L, L)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (B, L, d)
        out = self.to_out(out)
        return out, attn

class PosEmbedding(nn.Module):
    def __init__(self, emb_size, num_tokens):
        super().__init__()
        self.emb_size = emb_size
        self.resolution = num_tokens
        self.pos_embedding = nn.Parameter(torch.randn(num_tokens, emb_size))  # (L, d)
    
    def forward(self, x: torch.Tensor, apply_indices=None) -> torch.Tensor:
        B, _, C = x.shape
        if apply_indices is not None:
            pos_embed = self.pos_embedding.unsqueeze(0).repeat(B, 1, 1)  # (B, L, d)
            x = x + torch.gather(pos_embed, 1, apply_indices.unsqueeze(-1).expand(-1, -1, C))
        else:
            x = x + self.pos_embedding
        return x

class VisionTransformerEncoder(nn.Module):
    def __init__(self, in_resolution, in_channels, patch_size, emb_size, num_layers, num_heads, mlp_ratio, \
                layer_norm=nn.LayerNorm):
        super().__init__()
        self.in_resolutions = in_resolution
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_patches = (in_resolution // patch_size) ** 2
        
        self.patch_embed = PatchEmbed(in_channels, patch_size, emb_size)
        self.pos_embed = PosEmbedding(emb_size, self.num_patches)
        
        self.layer_norm = layer_norm(emb_size)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttentionBlock(emb_size, num_heads),
                FeedForwardBlock(emb_size, int(emb_size * mlp_ratio), emb_size)
            ])
            for _ in range(num_layers)
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Forward pass of the encoder. Encoder takes the output of backbone network. And only handle unmasked patches. 
        Args:
            x (torch.Tensor): tensor of shape (B, c, h, w)
            mask (torch.Tensor): bool tensor of shape (B, L), where n is the number of patches. True for masked patches.
        Returns:
            Tuple[torch.Tensor, list]: tensor of shape (B, V, D), list of tensors of shape (B, h, V, V)
        """
        assert mask.dtype == torch.bool, "mask should be a boolean tensor"
        
        # patch embedding
        x = self.patch_embed(x)  # (B, L, d)
        
        # add position embedding
        x = self.pos_embed(x)  # (B, V, d)
        
        # apply mask
        vis_indices = mask_to_indices(torch.logical_not(mask))  # (B, V)
        x = torch.gather(x, 1, vis_indices.unsqueeze(-1).expand(-1, -1, self.emb_size))  # (B, V, d)
        
        attn_weights_list = []
        for i, (attn, ffn) in enumerate(self.layers):
            # Multi-head attention
            residual = x
            x, attn_weights = attn(self.layer_norm(x))
            attn_weights_list.append(attn_weights)
            x = x + residual
            
            # Feed forward
            residual = x
            x = ffn(self.layer_norm(x))
            x = x + residual
        return x, attn_weights_list

class VisionTransformerPredictor(nn.Module):
    def __init__(self, num_patches, in_channels, out_channels, emb_size, num_layers, num_heads, mlp_ratio, layer_norm=nn.LayerNorm):
        super().__init__()
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.pos_embed = PosEmbedding(emb_size, num_patches)
        self.layer_norm = layer_norm(emb_size)
        self.mask_embed = nn.Embedding(1, emb_size)  
        
        self.embed = nn.Linear(in_channels, emb_size)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadAttentionBlock(emb_size, num_heads),
                FeedForwardBlock(emb_size, int(emb_size * mlp_ratio), emb_size)
            ])
            for _ in range(num_layers)
        ])
        
        self.proj = nn.Linear(emb_size, out_channels)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_all_patches=False) -> Tuple[torch.Tensor, list]:
        """Forward pass of the predictor. Predictor takes the output of encoder. 
        Args:
            x (torch.Tensor): tensor of shape (B, V, D)
            mask (torch.Tensor): bool tensor of shape (B, L), where n is the number of patches. True for masked patches.
            return_all_patches (bool): If True, return the output of all patches. Otherwise, return the output of masked patches.
        Returns:
            if return_all_patches is True:
                Tuple[torch.Tensor, list]: tensor of shape (B, V+M, c*p*p), list of tensors of shape (B, h, L, L)
            else:
                Tuple[torch.Tensor, list]: tensor of shape (B, M, c*p*p), list of tensors of shape (B, h, V, V)
        """
        assert mask.dtype == torch.bool, "mask should be a boolean tensor"
        
        L = mask.shape[1]
        M = L - x.shape[1]
        B = x.shape[0]
        mask_indices = mask_to_indices(mask)
        vis_indices = mask_to_indices(torch.logical_not(mask))  
        
        # mask embedding
        mask_embed = self.mask_embed.weight.unsqueeze(0).repeat(B, M, 1)  # (B, M, d), M is the number of masked patches
        
        # pos encoding
        mask_embed = self.pos_embed(mask_embed, apply_indices=mask_indices)  # (B, M, d)
        vis_embed = self.pos_embed(x, apply_indices=vis_indices)  # (B, V, d)
        
        # concat masked and visible patches
        x = torch.cat([vis_embed, mask_embed], dim=1)  # (B, L, d)
        
        attn_weights_list = []
        for i, (attn, ffn) in enumerate(self.layers):
            # Multi-head attention
            residual = x
            x, attn_weights = attn(self.layer_norm(x))
            attn_weights_list.append(attn_weights)
            x = x + residual
            
            # Feed forward
            residual = x
            x = ffn(self.layer_norm(x))
            x = x + residual
        
        x = self.proj(x)  
        
        if return_all_patches:
            return x, attn_weights_list
        else:
            return x[:, -M:], attn_weights_list
            
        