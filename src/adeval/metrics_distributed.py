import torch
import torch.distributed as dist
from src.dist_utils import concat_all_gather

@torch.no_grad()
def f1_max_gpu_hist(scores: torch.Tensor,
                    labels: torch.Tensor,
                    n_bins: int = 1001,
                    eps: float = 1e-8,
                    distributed: bool = False):
    """
    Memory-efficient F1-max on GPU.
    scores : (N,)  float32/float16, already in [0,1] (will be min-max normalised again)
    labels : (N,)  bool / {0,1} tensor   (1=anomaly)
    n_bins : number of threshold bins (≥2)
    eps    : numerical stabiliser
    distributed : if True, aggregate histograms across processes (DDP)
    """

    if distributed and dist.is_available() and dist.is_initialized():
        # compute global min/max over scores
        local_minmax = torch.stack([scores.min(), scores.max()])  # (2,)
        gathered = concat_all_gather(local_minmax)                # (world, 2)
        g_min = gathered[:, 0].min()
        g_max = gathered[:, 1].max()
        scores = (scores - g_min) / (g_max - g_min + eps)
    else:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + eps)

    scores = torch.clamp(scores, 0.0, 1.0 - eps)
    bin_idx = (scores * (n_bins - 1)).long()

    pos_per_bin = torch.zeros(n_bins, device=scores.device, dtype=torch.int64)
    neg_per_bin = torch.zeros_like(pos_per_bin)

    labels_bool = labels.bool()
    pos_per_bin.scatter_add_(0, bin_idx[labels_bool],
                             torch.ones_like(bin_idx[labels_bool]))
    neg_per_bin.scatter_add_(0, bin_idx[~labels_bool],
                             torch.ones_like(bin_idx[~labels_bool]))

    if distributed and dist.is_available() and dist.is_initialized():
        # sum histograms over all processes
        dist.all_reduce(pos_per_bin, op=dist.ReduceOp.SUM)
        dist.all_reduce(neg_per_bin, op=dist.ReduceOp.SUM)

    tp_cum = pos_per_bin.flip(0).cumsum(0).flip(0).to(torch.float32)
    fp_cum = neg_per_bin.flip(0).cumsum(0).flip(0).to(torch.float32)

    total_pos = tp_cum[0]
    fn_cum = total_pos - tp_cum

    denom = 2 * tp_cum + fp_cum + fn_cum + eps
    f1 = (2 * tp_cum) / denom

    best = torch.argmax(f1)
    thr = best.float() / (n_bins - 1)

    return f1[best], thr