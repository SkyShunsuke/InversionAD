from typing import (
    Tuple,
    Optional,
    Union,
    Iterable,
    Generic,
)
from collections.abc import Generator
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch

from .mem_effic import (
    _Curve,
    ReusableGenerator,
    T_seed,
    PosWeightStrategy,
    NegWeightStrategy,
    ReusableMap,
    _pro_weight as _pro_weight_np,
    _trapezoid_intep,
)


def _histogram_bins(input: Tensor, bins: int, min: float, max: float) -> Tensor:

    input = input.reshape(-1)
    stat = input.new_zeros(bins + 2, dtype=torch.float64)
    torch.histc(input, bins=bins, min=min, max=max,
                out=stat[1:-1])
    stat[0] = input.less(min).count_nonzero()
    stat[-1] = input.greater_equal(max).count_nonzero()
    return stat


class _AccumulateStatCurve:

    def __init__(self, min_score: float, max_score: float, nstrips: int = 1000) -> None:
        self.min = min_score
        self.max = max_score
        assert self.min < self.max
        self.bins = nstrips
        self.pos_stat = torch.zeros((self.bins + 2),
                                    dtype=torch.double, device='cuda')
        self.neg_stat = torch.zeros_like(self.pos_stat)
        self.weighted_pos_stat = torch.zeros_like(self.pos_stat)
        self.weighted_neg_stat = torch.zeros_like(self.pos_stat)

    def accum(self, preds: Tensor, label: Tensor,
              weight: Union[float, Tensor] = 1.):
        assert preds.shape == label.shape
        if isinstance(weight, Tensor):
            assert weight.ndim == 0 or weight.shape == label.shape
        else:
            weight = float(weight)
            assert weight > 0
        label = torch.not_equal(label, 0)

        preds = preds.reshape(-1)
        label = label.reshape(-1)

        if isinstance(weight, Tensor) and weight.ndim != 0:
            weight = weight.reshape(-1)
            preds_pos = preds[label]
            preds_neg = preds[~label]
            weight_pos = weight[label]
            weight_neg = weight[~label]
            del preds, label
            for wt in torch.unique(weight):
                hist_pos = _histogram_bins(preds_pos[weight_pos == wt], self.bins, self.min, self.max)
                hist_neg = _histogram_bins(preds_neg[weight_neg == wt], self.bins, self.min, self.max)
                self.pos_stat.add_(hist_pos)
                self.neg_stat.add_(hist_neg)
                self.weighted_pos_stat.add_(hist_pos, alpha=float(wt))
                self.weighted_neg_stat.add_(hist_neg, alpha=float(wt))

        else:
            hist_pos = _histogram_bins(preds[label], self.bins, self.min, self.max)
            hist_neg = _histogram_bins(preds[~label], self.bins, self.min, self.max)
            self.pos_stat.add_(hist_pos)
            self.neg_stat.add_(hist_neg)
            self.weighted_pos_stat.add_(hist_pos, alpha=float(weight))
            self.weighted_neg_stat.add_(hist_neg, alpha=float(weight))

    def reset(self, min_score: Optional[float] = None, max_score: Optional[float] = None):
        assert (
            self.min if min_score is None else min_score
        ) < (
            self.max if max_score is None else max_score
        )
        self.min = self.min if min_score is None else min_score
        self.max = self.max if max_score is None else max_score
        self.pos_stat *= 0.
        self.neg_stat *= 0.

    def _summary(self) -> _Curve:

        def cum(val: Tensor) -> Tuple[Tensor, float]:
            cval = torch.cumsum(val, dim=0)
            return cval[:-1], float(cval[-1])

        def asnp(val: Tensor) -> np.ndarray:
            return val.cpu().numpy()

        thr = np.linspace(0., 1., num=self.bins + 1, endpoint=True, dtype=np.float32)
        fn, p = cum(self.pos_stat)
        tn, n = cum(self.neg_stat)
        tp = p - fn
        fp = n - tn

        return _Curve(tp=asnp(tp), fp=asnp(fp), tn=asnp(tn), fn=asnp(fn),
                      thr=thr)

    def _summary_weighted(self) -> _Curve:

        def cum(val: Tensor) -> Tuple[Tensor, float]:
            cval = torch.cumsum(val, dim=0)
            return cval[:-1], float(cval[-1])

        def asnp(val: Tensor) -> np.ndarray:
            return val.cpu().numpy()

        thr = np.linspace(0., 1., num=self.bins + 1, endpoint=True, dtype=np.float32)
        fn, p = cum(self.weighted_pos_stat)
        tn, n = cum(self.weighted_neg_stat)
        tp = p - fn
        fp = n - tn

        return _Curve(tp=asnp(tp), fp=asnp(fp), tn=asnp(tn), fn=asnp(fn),
                      thr=thr)   

    def roc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # fpr, tpr, threshold
        curve = self._summary()
        return curve.fpr, curve.tpr, curve.thr
    
    def pr(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # recall, precision, threshold
        curve = self._summary()
        mask = np.greater(curve.tp + curve.fp, 0)
        recall, precison, thr = curve.tpr, curve.precision, curve.thr
        precison[~mask] = 1.  # set precison to 1. as no predict positive -> no false positive
        return recall, precison, thr

    def weighted_roc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # fpr, tpr, threshold
        curve = self._summary_weighted()
        return curve.fpr, curve.tpr, curve.thr

 
def _minmax(vals: Union[Tensor, Iterable[Tensor]]
            ) -> Tuple[float, float]:
    
    if isinstance(vals, Tensor):
        vals = vals.cuda(non_blocking=True)
        return float(vals.min()), float(vals.max())

    else:
        minmax = torch.tensor([_minmax(val) for val in vals], device='cuda')
        return float(minmax[:, 0].min()), float(minmax[:, 1].max())


def _perform_accum(preds: Union[Tensor, Iterable[Tensor]],
                   targets: Union[Tensor, Iterable[Tensor]],
                   weight_strategy: int = 0, *, nstrips: int = 1000) -> _AccumulateStatCurve:

    assert weight_strategy >= 0
    pos_strategy = weight_strategy & 0xff00
    neg_strategy = weight_strategy & 0x00ff

    assert pos_strategy in PosWeightStrategy.__members__.values()
    assert neg_strategy in NegWeightStrategy.__members__.values()

    if isinstance(preds, Generator):
        raise ValueError('preds of type Generator is not allowed')

    with torch.no_grad():
        min_score, max_score = _minmax(preds)

        if isinstance(preds, Tensor) and (preds.ndim < 2 or preds[0:1].numel() < 64):
            preds = [preds]
        if isinstance(targets, Tensor) and (targets.ndim < 2 or targets[0:1].numel() < 64):
            targets = [targets]

        accum = _AccumulateStatCurve(min_score, max_score, nstrips=nstrips)
        for subpre, subtar in zip(preds, targets):
            subpre = subpre.cuda(non_blocking=True)
            subtar = subtar.cuda(non_blocking=True)

            if pos_strategy == PosWeightStrategy.Nop and neg_strategy == NegWeightStrategy.Nop:
                weight = 1.
            else:
                weight = torch.empty_like(subtar)
                label = torch.not_equal(subtar, 0)
                if pos_strategy == PosWeightStrategy.Nop:
                    weight[label] = 1.
                elif pos_strategy == PosWeightStrategy.CopyFromTragets:
                    weight[label] = subtar[label]
                else:
                    raise NotImplementedError
                if neg_strategy == NegWeightStrategy.Nop:
                    weight[~label] = 1.
                else:
                    raise NotImplementedError

            accum.accum(subpre, subtar, weight)

        return accum


def auroc(preds: Union[Tensor, Iterable[Tensor]],
          targets: Union[Tensor, Iterable[Tensor]],
          *, nstrips: int = 1000) -> float:

    fpr, tpr, _ = _perform_accum(preds, targets, nstrips=nstrips).roc()
    return float(np.trapz(tpr[::-1], fpr[::-1], axis=0))


def aupr(preds: Union[Tensor, Iterable[Tensor]],
         targets: Union[Tensor, Iterable[Tensor]],
         *, nstrips: int = 1000) -> float:

    recall, precision, _ = _perform_accum(preds, targets, nstrips=nstrips).pr()
    return float(np.trapz(precision[::-1], recall[::-1], axis=0))


def auroc_and_aupr(preds: Union[Tensor, Iterable[Tensor]],
                   targets: Union[Tensor, Iterable[Tensor]],
                   *, nstrips: int = 1000) -> Tuple[float, float]:
    accum = _perform_accum(preds, targets, nstrips=nstrips)
    fpr, tpr, _ = accum.roc()
    recall, precision, _ = accum.pr()
    return (float(np.trapz(tpr[::-1], fpr[::-1], axis=0)),
            float(np.trapz(precision[::-1], recall[::-1], axis=0)))


def _pro_weight(label: Tensor) -> Tensor:
    shape = label.shape
    return torch.from_numpy(_pro_weight_np(label.squeeze().cpu().numpy())
                            ).reshape(shape)


class _ProWeightWrapper(Dataset[Tensor]):

    def __init__(self, src: Dataset[Tensor]) -> None:
        super().__init__()
        self._src = src

    def __len__(self) -> int:
        return len(self._src)
    
    def __getitem__(self, index: int) -> Tensor:
        return _pro_weight(self._src[index])


def _add_pro_weight_to_dataloader(data: DataLoader[Tensor]
                                  ) -> DataLoader[Tensor]:

    data.__dict__['dataset'] = _ProWeightWrapper(data.dataset)
    # we know that wraper dataset that has a strictly mapping
    #    to the wrapped dataset is allowed

    return data


def auroc_aupr_aupro(
    preds: Union[Tensor, Iterable[Tensor]],
    targets: Union[Tensor, Iterable[Tensor]],
    *, nstrips: int = 1000
) -> Tuple[float, float, float]:
    LIMIT = 0.3
    accum = _perform_accum(
        preds,
        _add_pro_weight_to_dataloader(targets)
        if isinstance(targets, DataLoader)
        else ReusableMap(targets, _pro_weight),
        PosWeightStrategy.CopyFromTragets,
        nstrips=nstrips
    )
    fpr, pro, _ = accum.weighted_roc()
    fpr, tpr, _ = accum.roc()
    recall, precision, _ = accum.pr()

    mask = fpr <= LIMIT
    lo_pro, lo_fpr = pro[mask][::-1], fpr[mask][::-1]
    hi_pro, hi_fpr = pro[~mask][::-1], fpr[~mask][::-1]
    lo = float(np.trapz(lo_pro, lo_fpr, axis=0))
    hi = float(np.trapz(hi_pro, hi_fpr, axis=0))
    tot = float(np.trapz(pro[::-1], fpr[::-1], axis=0))

    return (
        float(np.trapz(tpr[::-1], fpr[::-1], axis=0)),
        float(np.trapz(precision[::-1], recall[::-1], axis=0)),
        (lo + _trapezoid_intep(
            (LIMIT - lo_fpr[-1]) / (hi_fpr[0] - lo_fpr[-1]),
            tot - (lo + hi), lo_pro[-1], hi_pro[0]
        )) / LIMIT,
    )


class Preloader(Dataset[Tensor], Generic[T_seed]):

    def __init__(self, gen: ReusableGenerator[np.ndarray, T_seed]) -> None:
        super().__init__()
        self._src = gen._src
        self._trans = gen._trans

    def __len__(self) -> int:
        return len(self._src)
    
    def __getitem__(self, index: int) -> Tensor:
        res = self._trans(self._src[index])
        return torch.from_numpy(res)

    def build_loader(self, workers: int = 0,
                     batch_size: int = 1) -> DataLoader[Tensor]:
        from multiprocessing.context import SpawnContext
        from .utils import HAS_MP_WITH_LOCALS

        if not HAS_MP_WITH_LOCALS:
            workers = 0

        assert isinstance(workers, int) and workers >= 0
        assert isinstance(batch_size, int) and batch_size > 0
        return DataLoader(
            self, batch_size=batch_size, shuffle=False,
            pin_memory=True, num_workers=workers,
            multiprocessing_context=SpawnContext()
            if HAS_MP_WITH_LOCALS else None,
        )