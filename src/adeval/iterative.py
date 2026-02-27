from typing import Optional, Union, List, Literal, Dict
from numbers import Real
import numpy as np
import cv2

NDArr = np.ndarray
Elements = Union[NDArr, Real, List[NDArr], List[Real]]

from .mem_effic import _AccumulateStatCurve, _pro_weight, _trapezoid_intep


class EvalAccumulator:
    """
    Accumulate results without keeping them integrallty in memory. Note
    that the estimated lower & upper bound of scores should be provided.
    It is recommanded that 99% of predicted scores should be between the
    lower & upper bound so that the evaluation can be calculated accurately
    enough.

    estimated_score_lower: float
        estimated lower bound of image score
    estimated_score_upper: float
        estimated upper bound of image score
    estimated_anomap_lower: Optional[float]
        estimated lower bound of values in anomap, use lower bound of image
        score if not provided
    estimated_anomap_upper: Optional[float]
        estimated lower bound of values in anomap, use lower bound of image
        score if not provided
    ignore_pixel_aupro: bool
        skip pro calculation if True
    nstrips: int
        number of strips sampled on the ROC/PR curve, default to 1000,
        not less than 5
    """

    DEFAULT_NSTRIPS = 1000

    def __init__(self,
                 estimated_score_lower: float,
                 estimated_score_upper: float,
                 estimated_anomap_lower: Optional[float] = None,
                 estimated_anomap_upper: Optional[float] = None,
                 skip_pixel_aupro: bool = False,
                 *, nstrips: Optional[int] = None) -> None:

        assert all(isinstance(val, Real) for val in (
            estimated_score_lower, estimated_score_upper
        ))
        assert estimated_score_lower < estimated_score_upper
        if nstrips is None:
            nstrips = self.DEFAULT_NSTRIPS
        assert isinstance(nstrips, int) and nstrips >= 5

        if estimated_anomap_lower is None:
            estimated_anomap_lower = estimated_score_lower
        if estimated_anomap_upper is None:
            estimated_anomap_upper = estimated_score_upper

        assert all(isinstance(val, Real) for val in (
            estimated_anomap_lower, estimated_anomap_upper
        ))
        assert estimated_anomap_lower < estimated_anomap_upper

        self._img_bound = (estimated_score_lower, estimated_score_upper)
        self._map_bound = (estimated_anomap_lower, estimated_anomap_upper)
        self._has_pro = not bool(skip_pixel_aupro)
        self._nstrips = nstrips

        self.reset()

    def reset(self):
        self._sample_acc = _AccumulateStatCurve(*self._img_bound, nstrips=self._nstrips)
        self._image_acc = _AccumulateStatCurve(*self._img_bound, nstrips=self._nstrips)
        self._pixel_acc = _AccumulateStatCurve(*self._map_bound, nstrips=self._nstrips)

    def add_anomap(self, anomap: NDArr, gtmap: NDArr):
        if anomap.ndim != 2:
            raise ValueError('anomap: 2d array expected')
        if gtmap.ndim != 2:
            raise ValueError('gtmap: 2d array expected')

        weight = np.ones_like(gtmap, dtype=np.float32)
        if self._has_pro:
            gtmap = _pro_weight(gtmap)
            label = np.not_equal(gtmap, 0)
            weight[label] = gtmap[label]

        anomap = cv2.resize(anomap, gtmap.shape[0:2][::-1],
                            interpolation=cv2.INTER_LINEAR)

        self._pixel_acc.accum(anomap, gtmap, weight)

    def add_anomap_batch(self, anomap: Union[NDArr, List[NDArr]],
                         gtmap: Union[NDArr, List[NDArr]]):
        for pred, target in zip(anomap, gtmap):
            self.add_anomap(pred, target)

    def add_image(self, score: Elements, gtlabel: Elements):
        score = np.array(score).reshape(-1)
        gtlabel = np.array(gtlabel).reshape(-1)
        if score.shape != gtlabel.shape:
            raise ValueError('score & gtlabel not matched')
        self._image_acc.accum(score, gtlabel)

    def add_sample(self, score: Elements, gtlabel: Elements):
        score = np.array(score).reshape(-1)
        gtlabel = np.array(gtlabel).reshape(-1)
        if score.shape != gtlabel.shape:
            raise ValueError('score & gtlabel not matched')
        self._sample_acc.accum(score, gtlabel)

    def summary(self) -> Dict[Literal['s_auroc', 's_aupr',
                                      'i_auroc', 'i_aupr',
                                      'p_auroc', 'p_aupr', 'p_aupro'], float]:
        return dict(
            s_auroc=self._auroc(self._sample_acc),
            s_aupr=self._aupr(self._sample_acc),
            i_auroc=self._auroc(self._image_acc),
            i_aupr=self._aupr(self._image_acc),
            p_auroc=self._auroc(self._pixel_acc),
            p_aupr=self._aupr(self._pixel_acc),
            p_aupro=self._aupro(),
        )

    @staticmethod
    def _auroc(acc: _AccumulateStatCurve) -> float:
        fpr, tpr, _ = acc.roc()
        return float(np.trapz(tpr[::-1], fpr[::-1], axis=0))

    @staticmethod
    def _aupr(acc: _AccumulateStatCurve) -> float:
        recall, precision, _ = acc.pr()
        return float(np.trapz(precision[::-1], recall[::-1], axis=0))
    
    def _aupro(self) -> float:
        if not self._has_pro:
            return float('nan')

        LIMIT = 0.3
        fpr, pro, _ = self._pixel_acc.weighted_roc()
        mask = fpr <= LIMIT
        lo_pro, lo_fpr = pro[mask][::-1], fpr[mask][::-1]
        hi_pro, hi_fpr = pro[~mask][::-1], fpr[~mask][::-1]
        lo = float(np.trapz(lo_pro, lo_fpr, axis=0))
        hi = float(np.trapz(hi_pro, hi_fpr, axis=0))
        tot = float(np.trapz(pro[::-1], fpr[::-1], axis=0))

        return (lo + _trapezoid_intep(
            (LIMIT - lo_fpr[-1]) / (hi_fpr[0] - lo_fpr[-1]),
            tot - (lo + hi), lo_pro[-1], hi_pro[0]
        )) / LIMIT
