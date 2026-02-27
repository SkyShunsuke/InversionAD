from typing import Union, List
from numbers import Real
import torch
from torch import Tensor
from torch.nn.functional import interpolate as interp

NDArr = Tensor
Elements = Union[NDArr, Real, List[NDArr], List[Real]]

from .iterative import EvalAccumulator as _Base
from .cuda_mem_effic import _AccumulateStatCurve, _pro_weight


class EvalAccumulator(_Base):

    def reset(self):
        super().reset()
        self._pixel_acc = _AccumulateStatCurve(*self._map_bound, nstrips=self._nstrips)

    def add_anomap(self, anomap: NDArr, gtmap: NDArr):
        if anomap.ndim != 2:
            raise ValueError('anomap: 2d array expected')
        if gtmap.ndim != 2:
            raise ValueError('gtmap: 2d array expected')

        weight = gtmap.new_ones(gtmap.shape, dtype=torch.float32)
        if self._has_pro:
            gtmap = _pro_weight(gtmap).to(gtmap.device)
            label = torch.not_equal(gtmap, 0)
            weight[label] = gtmap[label]

        anomap = interp(anomap.reshape((1, 1,) + anomap.shape),
                        gtmap.shape, mode='bilinear', align_corners=False)
        anomap = anomap.squeeze_()

        self._pixel_acc.accum(anomap, gtmap, weight)

    def add_anomap_batch(self, anomap: Union[NDArr, List[NDArr]],
                         gtmap: Union[NDArr, List[NDArr]]):
        for pred, target in zip(anomap, gtmap):
            self.add_anomap(pred, target)

    def add_image(self, score: Elements, gtlabel: Elements):
        super().add_image(
            score.cpu().numpy() if torch.is_tensor(score) else score,
            gtlabel.cpu().numpy() if torch.is_tensor(gtlabel) else gtlabel
        )

    def add_sample(self, score: Elements, gtlabel: Elements):
        super().add_sample(
            score.cpu().numpy() if torch.is_tensor(score) else score,
            gtlabel.cpu().numpy() if torch.is_tensor(gtlabel) else gtlabel
        )
