from .mlp import SimpleMLPAdaLN
from .unet import UNetModel
from .dit import DiT, DiTMoE
from .vae import AutoencoderKL

def create_unet_model(latent_size=32, model_channels=256, num_res_blocks=2, num_heads=8, channel_mult=[1,2,4], context_dim=512, ncls=15):
    model = UNetModel(image_size=latent_size, 
                    in_channels=272,
                    model_channels=model_channels, 
                    out_channels=272, 
                    num_heads=num_heads, 
                    num_res_blocks=num_res_blocks, 
                    dropout=0.,
                    attention_resolutions=[2, 4, 8], 
                    channel_mult = channel_mult,
                    num_head_channels= model_channels//num_heads,
                    use_spatial_transformer= False,
                    ncls=ncls,
                    transformer_depth= 2,
                    context_dim=None,
                )
    return model

def get_vae():
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
    return vae
    
def UNET_XS(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=64, num_heads=4, channel_mult=[1,2,4], context_dim=128, ncls=ncls)

def UNET_S(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=128, num_heads=4, channel_mult=[1,2,4], context_dim=256, ncls=ncls)

def UNET_M(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=192, num_heads=6, channel_mult=[1,2,4], context_dim=384, ncls=ncls)

def UNET_L(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=256, num_heads=8, channel_mult=[1,2,4], context_dim=512, ncls=ncls)

def UNET_XL(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=320, num_heads=12, channel_mult=[1,2,4], context_dim=640, ncls=ncls)

UNET_models = {
'UNet_XS' : UNET_XS, 
'UNet_S' : UNET_S, 
'UNet_M' : UNET_M, 
'UNet_L' : UNET_L, 
'UNet_XL' : UNET_XL, 
}

def create_vae(
    model_type: str,
    embed_dim = 16,
    ch_mult = (1, 1, 2, 2, 4),
    ckpt_path = None,
    **kwargs
):
    assert ckpt_path is not None, "Checkpoint path must be provided"
    if model_type == "vae_kl":
        return AutoencoderKL(
            embed_dim=embed_dim,
            ch_mult=ch_mult,
            ckpt_path=ckpt_path,
        )

def create_denising_model(
    model_type: str,
    in_channels: int,
    in_res: int,
    model_channels: int,
    out_channels: int,
    z_channels: int,
    num_blocks: int,
    patch_size: int = 1,
    num_heads: int = 8,
    mlp_ratio: int = 4,
    class_dropout_prob: float = 0.,
    num_classes: int = 15,
    learn_sigma: bool = False,
    grad_checkpoint: bool = False, 
    num_experts: int = 4,
    conditioning_scheme: str = 'none',
    pos_embed = None,
    channel_mult = (1, 1, 2, 2),
    num_heads_upsample: int = -1,
    attention_resolutions: list = [2, 4, 8],
    **kwargs
):
    if model_type == "mlp":
        return SimpleMLPAdaLN(
            input_size=in_res,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            z_channels=z_channels,
            num_blocks=num_blocks,
            grad_checkpoint=grad_checkpoint
        )
    elif model_type == "unet":
        return create_unet_model(
            latent_size=16,
            ncls=30,
            model_channels=512,
            num_heads=8,
            num_res_blocks=2,
            channel_mult=[1, 2, 3, 4],
            context_dim=None,
        )
    elif model_type == "dit":
        return DiT(
            input_size=in_res,
            patch_size=patch_size,
            in_channels=in_channels,
            cond_channels=z_channels,
            hidden_size=model_channels,
            depth=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
            pos_embed=pos_embed
        )
    elif model_type == "dit_moe":
        return DiTMoE(
            input_size=in_res,
            patch_size=patch_size,
            in_channels=in_channels,
            cond_channels=z_channels,
            hidden_size=model_channels,
            depth=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_experts=num_experts,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
            pos_embed=pos_embed
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")