from .mem_effic import (
    auroc,
    auroc_and_aupr,
    auroc_aupr_aupro,
)
from .iterative import EvalAccumulator

try:
    import torch
except ImportError:
    torch = None

from .utils import HAS_MP_WITH_LOCALS
if torch is not None and torch.cuda.is_available():
    from .cuda_mem_effic import (
        auroc as auroc_cuda,
        auroc_and_aupr as auroc_and_aupr_cuda,
        auroc_aupr_aupro as auroc_aupr_aupro_cuda,
    )
    HAS_CUDA = True
else:
    HAS_CUDA = False

if torch is not None:
    from .iterative_cuda import (
        EvalAccumulator as EvalAccumulatorCuda,
    )

from .metrics_distributed import f1_max_gpu_hist