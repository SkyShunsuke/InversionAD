"""

"""

import math

import numpy as np
import torch
from torch import nn
from torch import Tensor
import enum
from typing import Tuple

from .utils import *
from .tensor import extract_into_tensor, mean_flat
from .schedulers import *


class ModelMeanType(enum.Enum):
    """
    What is used as the model's output mean.
    """
    PREVIOUS_X = enum.auto()  # the model predicts the previous x_{t-1}
    START_X = enum.auto()  # the model predicts start x_0
    EPSILON = enum.auto()  # the model predicts added noise

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict values b/w FIXED_SMALL and FIXED_LARGE, making its job easier. 
    """
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    """
    The type of loss function used.
    """
    MSE = enum.auto()
    RESCALED_MSE = (
        enum.auto()
    )
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self in [LossType.KL, LossType.RESCALED_KL]

class GaussianDiffusion:
    """"""
    def __init__(
        self, 
        *,
        betas: np.ndarray,
        model_mean_type: ModelMeanType, 
        model_var_type: ModelVarType,
        loss_type: LossType,
    ):
        self.betas = np.array(betas, dtype=np.float64)
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        
        assert len(self.betas.shape) == 1, "betas must be 1D"
        assert (self.betas > 0).all() and (betas <= 1).all(), "betas must be in (0, 1]"
        
        self.num_timesteps = int(betas.shape[0])
        
        # Calculate alphas
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)  # (T,) 
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])  # (T,)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)  # (T,)
        assert self.alphas_cumprod.shape == (self.num_timesteps,)
        
        # Calculate coefficients for diffusion steps q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)  # (T,)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)  # (T,)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)  # (T,)
        
        # Calculate coefficients for relation b/w x_0 and x_t
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)  # (T,)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)  # (T,)
        
        # Calculate coefficients for posterior q(x_{t-1} | x_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # (T,)
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])  # (T, )
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # (T,)
        self.posterior_mean_coef2 = (
            np.sqrt(alphas) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # (T,)
    
    def q_mean_variance(self, x_start: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Get the distribution of q(x_t | x_{t-1}).
        Args:
            x_start (Tensor): tensor of shape (b, c, h, w)
            t (Tensor): tensor of shape (b, ) representing the time indices.
        Returns:
            Tuple[Tensor, Tensor, Tensor]: mean, variance, log_variance of q(x_t | x_{t-1})
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
        
    def q_sample(self, x_start: Tensor, t: Tensor, noise=None) -> Tensor:
        """Sample from q(x_t | x_{t-1}).
        Args:
            x_start (Tensor): tensor of shape (b, c, h, w)
            t (Tensor): tensor of shape (b, ) representing the time indices.
            noise (Tensor): tensor of shape (b, c, h, w) representing the noise. If None, a new noise tensor is created.
        Returns:
            Tensor: tensor of shape (b, c, h, w) representing the sampled x_t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)   
        assert noise.shape == x_start.shape
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start 
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_posterior_mean_variance(self, x_start: Tensor, x_t: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Get the posterior distribution of q(x_{t-1} | x_t).
        Args:
            x_start (Tensor): tensor of shape (b, c, h, w)
            x_t (Tensor): tensor of shape (b, c, h, w)
            t (Tensor): tensor of shape (b, ) representing the time indices.
        Returns:
            Tuple[Tensor, Tensor, Tensor]: mean, variance, log_variance of q(x_{t-1} | x_t)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_start.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_start.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_start.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_start.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def _predict_xstart_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Predict x_0 from x_t and noise.
        """
        assert x_t.shape == eps.shape
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(self, model: nn.Module, x: Tensor, t: Tensor, clip_denoised: bool = True, denoised_fn=None, model_kwargs=None):
        """"""
        if model_kwargs is None:
            model_kwargs = {}
        
        B, C = x.shape[:2]
        assert t.shape == (B, )
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None
        
        # For both variance, the posterior variance is at least beta_t, and is scaled by \bar{beta}_{t-1} / \bar{beta}_t.
        # TODO: 
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])  # Model should output mean and variance
            model_mean_values, model_var_values = torch.split(model_output, C, dim=1)
            min_log = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)  # 
            max_log = extract_into_tensor(np.log(self.betas), t, x.shape)  
            
            # We assume model var values are in the range [-1, 1]
            frac = (model_var_values + 1) / 2    # (B, C, H, W)
            model_log_variance = frac * max_log + (1 - frac) * min_log  # (B, C, H, W)
            model_variance = torch.exp(model_log_variance)
        else:
            model_mean_values = model_output
            model_variances, model_log_variances = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:]))
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped
                ),
            }[self.model_var_type]
            model_variance = extract_into_tensor(model_variances, t, x.shape)
            model_log_variance = extract_into_tensor(model_log_variances, t, x.shape)
        
        def process_xstart(x: Tensor) -> Tensor:
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                x = torch.clamp(x, -1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_mean_values)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_mean_values)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        
        assert model_mean.shape == model_log_variance.shape == x.shape == pred_xstart.shape
        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_xstart': pred_xstart,
            'extra': extra,
            'eps': model_mean_values,
        }
    
    def condition_mean(self, cond_fn: nn.Module, p_mean_var: dict, x: Tensor, t: Tensor, model_kwargs=None) -> Tensor:
        """Compute the mean for the previous step, given a function cond_fn that computes the gradient of a 
        conditional log probability with respect to x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y. i.e., classifier-guidance.
        Args:
            cond_fn (nn.Module): function that computes the gradient of a conditional log probability with respect to x.
            p_mean_var (dict): dictionary containing the mean and variance of the previous step.
            x (Tensor): tensor of shape (B, C, H, W)
            t (Tensor): tensor of shape (B, ) representing the time indices.
            model_kwargs (dict): additional arguments for the cond_fn.
        Returns:
            Tensor: tensor of shape (B, C, H, W) representing the new mean.
        """
        # TODO: 
        gradient = cond_fn(x, t, **model_kwargs)  # (B, C, H, W)
        new_mean = p_mean_var['mean'].float() + p_mean_var['variance'] * gradient.float()
        return new_mean
    
    def conditional_score(self, cond_fn: nn.Module, p_mean_var: dict, x: Tensor, t: Tensor, model_kwargs=None) -> dict:
        """Compute what the p_mean_variance output would have been if the model had been conditioned by cond_fn.
        Args:
            cond_fn (nn.Module): function that computes the gradient of a conditional log probability with respect to x.
            p_mean_var (dict): dictionary containing the mean and variance of the previous step.
            x (Tensor): tensor of shape (B, C, H, W)
            t (Tensor): tensor of shape (B, ) representing the time indices.
            model_kwargs (dict): additional arguments for the cond_fn.
        Returns:
            dict: dictionary containing the mean and variance of the previous step.
        """
        # TODO: 
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)  # (B, C, H, W)
        
        eps = self._predict_eps_from_xstart(x_t=x, t=t, pred_xstart=p_mean_var['pred_xstart'])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)
        
        out = p_mean_var.copy()
        out['pred_xstart'] = self._predict_xstart_from_eps(x_t=x, t=t, eps=eps)
        out['mean'], _, _ = self.q_posterior_mean_variance(x_start=out['pred_xstart'], x_t=x, t=t)
        return out

    def p_sample(
        self, 
        model: nn.Module,
        x: Tensor,
        t: Tensor,
        clip_denoised: bool = True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        temperature: float = 1.0,
    ) -> dict:
        """Sample x_{t-1} from the model at time t.
        Args:
            model (nn.Module): model that predicts the previous step.
            x (Tensor): tensor of shape (B, C, H, W) representing the current step.
            t (Tensor): tensor of shape (B, ) representing the time indices.
            clip_denoised (bool): whether to clip the denoised output.
            denoised_fn: function that denoises the output.
            cond_fn: function that computes the gradient of a conditional log probability with respect to x.
            model_kwargs (dict): additional arguments for the model.
            temperature (float): temperature for sampling.
        Returns:
            dict: dictionary containing the sampled x_{t-1} and the predicted x_0.
        """
        # Predict the mean and variance of the previous step
        out = self.p_mean_variance(model, x, t, clip_denoised, denoised_fn, model_kwargs)
        
        noise = torch.randn_like(x)  # (B, C, H, W)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  
        ) # (B, 1, 1, 1), no noise for t=0
        
        # conditioning
        if cond_fn is not None:
            # Compute the conditional score 
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        
        # sampling
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise * temperature
        return {
            "posterior_mean": out["mean"],
            "posterior_variance": torch.exp(0.5 * out["log_variance"]),
            "eps": out["eps"],
            "sample": sample,
            "pred_xstart": out["pred_xstart"],  # TODO: this is not conditioned
        }
        
    def p_sample_loop_progressive(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        noise: Tensor=None,
        clip_denoised: bool=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress: bool=False,
        temperature: float=1.0,
    ):
        """Generate samples from the model and yield intermediate samples from each timesteps of the diffusion chain.
        Args:
            model (nn.Module): model that predicts the previous step.
            shape (Tuple[int, int, int, int]): shape of the input tensor.
            noise (Tensor): tensor of shape (B, C, H, W) representing the noise. If None, a new noise tensor is created.
            clip_denoised (bool): whether to clip the denoised output.
            denoised_fn: function that denoises the output.
            cond_fn: function that computes the gradient of a conditional log probability with respect to x.
            model_kwargs (dict): additional arguments for the model.
            device: device to use.
            progress (bool): whether to show progress bar.
            temperature (float): temperature for sampling.
        Yields:
            dict: dictionary containing the sampled x_{t-1} and the predicted x_0.
        """
        assert isinstance(shape, (tuple, list))
        
        # sample x_T
        if noise is not None:
            img = noise  # (B, C, H, W)
        else:
            img = torch.randn(shape, device=device)  # (B, C, H, W)
        indices = list(range(self.num_timesteps))[::-1]  # [T-1, T-2, ..., 0]
        
        if progress:
            from tqdm.auto import tqdm
            
            indices = tqdm(indices)
        
        for i in indices:
            t = torch.tensor([i] * shape[0]).to(device)  # (B, )
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img, 
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    temperature=temperature
                )
                yield out
                img = out["sample"]  # update x_t to x_{t-1}
        
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        noise: Tensor=None,
        clip_denoised: bool=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress: bool=False,
        temperature: float=1.0,
    ):
        """Generate samples from the model and return the final sample.
        Args:
            model (nn.Module): model that predicts the previous step.
            shape (Tuple[int, int, int, int]): shape of the input tensor.
            noise (Tensor): tensor of shape (B, C, H, W) representing the noise. If None, a new noise tensor is created.
            clip_denoised (bool): whether to clip the denoised output.
            denoised_fn: function that denoises the output.
            cond_fn: function that computes the gradient of a conditional log probability with respect to x.
            model_kwargs (dict): additional arguments for the model.
            device: device to use.
            progress (bool): whether to show progress bar.
            temperature (float): temperature for sampling.
        Returns:
            Tensor: tensor of shape (B, C, H, W) representing the final sample.
        """
        final: Tensor = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape, 
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            temperature=temperature
        ):
            final = sample
        return final["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        # eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        eps = out["eps"]

        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"], "eps": eps}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape).cuda()
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0]).cuda()
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, 
        model: nn.Module,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        clip_denoised: bool=True,
        denoised_fn=None,
    ) -> dict:
        """Get a term for the variational lower-bound. The resulting units are bits.
        Args:
            model (nn.Module): model that predicts the previous step.
            x_start (Tensor): tensor of shape (B, C, H, W) representing the start step.
            x_t (Tensor): tensor of shape (B, C, H, W) representing the current step.
            t (Tensor): tensor of shape (B, ) representing the time indices.
            clip_denoised (bool): whether to clip the denoised output.
            denoised_fn: function that denoises the output.
        Returns:
            dict: dictionary containing the output and the predicted x_0.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(
            model,
            x_t,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])  # (B, C, H, W)
        kl = mean_flat(kl) / np.log(2.0)  # (B, )  # TODO: why log(2.0)?
        
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=out["log_variance"] * 0.5
        )  # (B, C, H, W)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)  # (B, )
        
        # For the first timestep, we only have the decoder NLL
        output = torch.where((t == 0), decoder_nll, kl)  # (B, )
        return {
            "output": output,
            "pred_xstart": out["pred_xstart"],
        }
        
    def training_losses(
        self, 
        model: nn.Module,
        x_start: Tensor,
        t: Tensor,
        model_kwargs=None,
        noise=None,
    ) -> dict:
        """Compute training losses for a single timestep.
        Args:
            model (nn.Module): model that predicts the previous step.
            x_start (Tensor): tensor of shape (B, C, H, W) representing the start step.
            t (Tensor): tensor of shape (B, ) representing the time indices.
            model_kwargs (dict): additional arguments for the model.
            noise (Tensor): tensor of shape (B, C, H, W) representing the noise. If None, a new noise tensor is created.
        Returns:
            dict: dictionary containing the loss.
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        if noise is None:
            noise = torch.randn_like(x_start)  # (B, C, H, W)
        x_t = self.q_sample(x_start, t, noise)  # (B, C, H, W)
        
        terms = {}
        
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model,
                x_start,
                x_t,
                t,
                **model_kwargs
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps  # rescale the KL loss
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, t, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_mean_values, model_var_values = torch.split(model_output, C, dim=1)
                
                # learn the variance using the ELBO, but don't let it affect our mean prediction.
                frozen_out = torch.cat([model_mean_values.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0  # rescale the KL loss without this, the MSE term is too small than vb.
            else:
                model_mean_values = model_output
            
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_mean_values.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_mean_values) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        return terms
                