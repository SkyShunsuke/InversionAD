from .mlp import SimpleMLPAdaLN
from .unet import UNetModel 
from .dit import DiT
from .vae import AutoencoderKL

# unet parameters
tiny  = dict(model_channels=128, channel_mult=(1,2,2,4), num_res_blocks=1, attention_resolutions=(8,))
base  = dict(model_channels=192, channel_mult=(1,2,3,4), num_res_blocks=2, attention_resolutions=(8,4))
# large = dict(model_channels=256, channel_mult=(1,2,3,4,4), num_res_blocks=3, attention_resolutions=(16,8,4))
large = dict(model_channels=512, channel_mult=(1,2,3,4,4), num_res_blocks=3, attention_resolutions=(16,8,4,2))
xl    = dict(model_channels=320, channel_mult=(1,1,2,2,4,4), num_res_blocks=3, attention_resolutions=(16,8,4,2))

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
    conditioning_scheme: str = 'none',
    pos_embed = None,
    channel_mult = (1, 1, 2, 2),
    num_heads_upsample: int = -1,
    attention_resolutions: list = [2, 4, 8],
    **kwargs
):
    if model_type == "mlp":
        return SimpleMLPAdaLN(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            z_channels=z_channels,
            num_blocks=num_blocks,
            grad_checkpoint=grad_checkpoint
        )
    elif model_type == "unet":
        # return UNetModel(
        #     image_size=in_res,
        #     in_channels=in_channels,
        #     model_channels=model_channels,
        #     out_channels=out_channels,
        #     num_res_blocks=num_blocks,
        #     attention_resolutions=attention_resolutions,
        #     channel_mult=channel_mult,
        #     num_classes=num_classes,
        #     num_heads=num_heads,
        #     num_heads_upsample=num_heads_upsample,
        #     use_fp16=False,
        # )
        # use xl model
        return UNetModel(
            image_size=in_res,
            in_channels=in_channels,
            model_channels=large["model_channels"],
            out_channels=out_channels,
            num_res_blocks=large["num_res_blocks"],
            attention_resolutions=large["attention_resolutions"],
            channel_mult=large["channel_mult"],
            num_classes=num_classes,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_fp16=False,
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
    else:
        raise ValueError(f"Model type {model_type} not supported")