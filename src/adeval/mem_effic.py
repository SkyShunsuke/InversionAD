from typing import (
    Iterator,
    Optional,
    Tuple,
    Iterable,
    Union,
    Callable,
    TypeVar,
    Sequence,
    Generic,
)
from collections.abc import Generator
from enum import IntEnum
from dataclasses import dataclass
import numpy as np


T_seed = TypeVar('T_seed', covariant=True)
T_item = TypeVar('T_item', covariant=True)


def _identity(val: T_item) -> T_item:
    return val


class ReusableGenerator(Iterable[T_item], Generic[T_item, T_seed]):

    def __init__(self, source: Sequence[T_seed],
                 transform: Callable[[T_seed], T_item] = _identity) -> None:
        super().__init__()
        self._src = source
        self._trans = transform

    def __iter__(self) -> Iterator[T_item]:
        return (self._trans(item) for item in self._src)


class ReusableMap(Iterable[T_item], Generic[T_item, T_seed]):

    def __init__(self, source: Iterable[T_seed],
                 transform: Callable[[T_seed], T_item] = _identity) -> None:
        self._src = source
        self._trans = transform

    def __iter__(self) -> Iterator[T_item]:
        return map(self._trans, self._src)


@dataclass
class _Curve:
    tp: np.ndarray
    fp: np.ndarray
    tn: np.ndarray
    fn: np.ndarray
    thr: np.ndarray

    @property
    def tpr(self) -> np.ndarray:
        return self.tp / (self.tp + self.fn + 1e-10)  # avoid division by zero

    @property
    def fpr(self) -> np.ndarray:
        return self.fp / (self.fp + self.tn + 1e-10)  # avoid division by zero
    
    @property
    def precision(self) -> np.ndarray:
        return self.tp / (self.tp + self.fp + 1e-10)  # avoid division by zero
    
    @property
    def fdr(self) -> np.ndarray:
        return self.fp / (self.tp + self.fp + 1e-10)  # avoid division by zero

    @property
    def accuracy(self) -> np.ndarray:
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + 1e-10)  # avoid division by zero

    @property
    def f1score(self) -> np.ndarray:
        return 2 * self.tp + (2 * self.tp + self.fp + self.fn)


class _AccumulateStatCurve:

    def __init__(self, min_score: float, max_score: float, nstrips: int = 1000) -> None:
        self.min = min_score
        self.max = max_score
        assert self.min < self.max
        bins = np.linspace(0., 1., num=nstrips + 1, endpoint=True, dtype=np.float32)
        bins = bins * (self.max - self.min) + self.min
        inf = np.array([float('inf')], dtype=np.float32) # for overflow bins
        self.bins = np.concatenate([-inf, bins, inf], axis=0)
        self.pos_stat = np.zeros((self.bins.shape[0] - 1), dtype=np.double)
        self.neg_stat = np.zeros_like(self.pos_stat)
        self.weighted_pos_stat = np.zeros_like(self.pos_stat)
        self.weighted_neg_stat = np.zeros_like(self.pos_stat)

    def accum(self, preds: np.ndarray, label: np.ndarray,
              weight: Union[float, np.ndarray] = 1.):
        assert preds.shape == label.shape
        if isinstance(weight, np.ndarray):
            assert weight.ndim == 0 or weight.shape == label.shape
        else:
            weight = float(weight)
            assert weight > 0
        label = np.not_equal(label, 0)

        preds = preds.reshape(-1)
        label = label.reshape(-1)

        if isinstance(weight, np.ndarray) and weight.ndim != 0:
            weight = weight.reshape(-1)
            preds_pos = preds[label]
            preds_neg = preds[~label]
            weight_pos = weight[label]
            weight_neg = weight[~label]
            del preds, label
            for wt in np.unique(weight):
                hist_pos = np.histogram(preds_pos[weight_pos == wt], self.bins)[0]
                hist_neg = np.histogram(preds_neg[weight_neg == wt], self.bins)[0]
                self.pos_stat += hist_pos
                self.neg_stat += hist_neg
                self.weighted_pos_stat += hist_pos * wt
                self.weighted_neg_stat += hist_neg * wt

        else:
            hist_pos = np.histogram(preds[label], self.bins)[0]
            hist_neg = np.histogram(preds[~label], self.bins)[0]
            self.pos_stat += hist_pos
            self.neg_stat += hist_neg
            self.weighted_pos_stat += hist_pos * weight
            self.weighted_neg_stat += hist_neg * weight

    def reset(self, min_score: Optional[float] = None, max_score: Optional[float] = None):
        assert (
            self.min if min_score is None else min_score
        ) < (
            self.max if max_score is None else max_score
        )
        self.min = self.min if min_score is None else min_score
        self.max = self.max if max_score is None else max_score
        bins = self.bins[1:-1]
        ori_min, ori_max = bins.min(), bins.max()
        self.bins[1:-1] = (bins - ori_min) / (ori_max - ori_min) * (self.max - self.min) + self.min
        self.pos_stat *= 0.
        self.neg_stat *= 0.

    def _summary(self) -> _Curve:

        def cum(val: np.ndarray) -> Tuple[np.ndarray, float]:
            cval = np.cumsum(val)
            return cval[:-1], float(cval[-1])

        thr = self.bins[1:-1]
        fn, p = cum(self.pos_stat)
        tn, n = cum(self.neg_stat)
        tp = p - fn
        fp = n - tn

        return _Curve(tp=tp, fp=fp, tn=tn, fn=fn, thr=thr)

    def _summary_weighted(self) -> _Curve:

        def cum(val: np.ndarray) -> Tuple[np.ndarray, float]:
            cval = np.cumsum(val)
            return cval[:-1], float(cval[-1])

        thr = self.bins[1:-1]
        fn, p = cum(self.weighted_pos_stat)
        tn, n = cum(self.weighted_neg_stat)
        tp = p - fn
        fp = n - tn

        return _Curve(tp=tp, fp=fp, tn=tn, fn=fn, thr=thr)

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


def _minmax(vals: Union[np.ndarray, Iterable[np.ndarray]]
            ) -> Tuple[float, float]:
    
    if isinstance(vals, np.ndarray):
        return float(vals.min()), float(vals.max())

    else:
        minmax = np.array([(val.min(), val.max()) for val in vals])
        return float(minmax[:, 0].min()), float(minmax[:, 1].max())


class PosWeightStrategy(IntEnum):
    Nop = 0  # all ones
    CopyFromTragets = 0x0100


class NegWeightStrategy(IntEnum):
    Nop = 0  # all ones


def _perform_accum(preds: Union[np.ndarray, Iterable[np.ndarray]],
                   targets: Union[np.ndarray, Iterable[np.ndarray]],
                   weight_strategy: int = 0, *, nstrips: int = 1000) -> _AccumulateStatCurve:

    assert isinstance(nstrips, int) and nstrips >= 5, 'nstrips must be integer >= 5'

    assert weight_strategy >= 0
    pos_strategy = weight_strategy & 0xff00
    neg_strategy = weight_strategy & 0x00ff

    assert pos_strategy in PosWeightStrategy.__members__.values()
    assert neg_strategy in NegWeightStrategy.__members__.values()

    if isinstance(preds, Generator):
        raise ValueError('preds of type Generator is not allowed')

    min_score, max_score = _minmax(preds)

    if isinstance(preds, np.ndarray) and (preds.ndim < 2 or preds[0:1].size < 64):
        preds = [preds]
    if isinstance(targets, np.ndarray) and (targets.ndim < 2 or targets[0:1].size < 64):
        targets = [targets]

    accum = _AccumulateStatCurve(min_score, max_score, nstrips=nstrips)
    for subpre, subtar in zip(preds, targets):

        if pos_strategy == PosWeightStrategy.Nop and neg_strategy == NegWeightStrategy.Nop:
            weight = 1.
        else:
            weight = np.empty_like(subtar)
            label = np.not_equal(subtar, 0)
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


def auroc(preds: Union[np.ndarray, Iterable[np.ndarray]],
          targets: Union[np.ndarray, Iterable[np.ndarray]],
          *, nstrips: int = 1000) -> float:

    fpr, tpr, _ = _perform_accum(preds, targets, nstrips=nstrips).roc()
    return float(np.trapz(tpr[::-1], fpr[::-1], axis=0))


def aupr(preds: Union[np.ndarray, Iterable[np.ndarray]],
         targets: Union[np.ndarray, Iterable[np.ndarray]],
         *, nstrips: int = 1000) -> float:

    recall, precision, _ = _perform_accum(preds, targets, nstrips=nstrips).pr()
    return float(np.trapz(precision[::-1], recall[::-1], axis=0))


def auroc_and_aupr(preds: Union[np.ndarray, Iterable[np.ndarray]],
                   targets: Union[np.ndarray, Iterable[np.ndarray]],
                   *, nstrips: int = 1000) -> Tuple[float, float]:
    accum = _perform_accum(preds, targets, nstrips=nstrips)
    fpr, tpr, _ = accum.roc()
    recall, precision, _ = accum.pr()
    return (float(np.trapz(tpr[::-1], fpr[::-1], axis=0)),
            float(np.trapz(precision[::-1], recall[::-1], axis=0)))


def _pro_weight(gt_mask: np.ndarray) -> np.ndarray:
    import cv2
    gt_mask = gt_mask.astype(np.uint8)
    cnt, comp_mask, stat, _ = cv2.connectedComponentsWithStats(gt_mask, connectivity=8, ltype=cv2.CV_16U)
    weights = 1. / stat[1:, cv2.CC_STAT_AREA]
    weighted = np.zeros_like(gt_mask, dtype=np.float32)
    for label, weight in enumerate(weights, start=1):
        weighted[comp_mask == label] = weight
    return weighted


def _trapezoid_intep(interp: float, area: float, y1: float, y2: float
                     ) -> float:
    # assert area >= 0
    # assert interp >= 0
    # assert 0 <= y1 <= y2
    rect_ratio = y1 / (y1 + 0.5 * (y2 - y1))
    rect_area = area * rect_ratio
    tri_area = area * (1 - rect_ratio)
    return rect_area * interp + tri_area * interp * interp


def auroc_aupr_aupro(
    preds: Union[np.ndarray, Iterable[np.ndarray]],
    targets: Union[np.ndarray, Iterable[np.ndarray]],
    *, nstrips: int = 1000
) -> Tuple[float, float, float]:
    LIMIT = 0.3
    accum = _perform_accum(preds, ReusableMap(targets, _pro_weight),
                           PosWeightStrategy.CopyFromTragets, nstrips=nstrips)
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