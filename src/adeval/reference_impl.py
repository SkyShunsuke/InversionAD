from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Union
from functools import partial

import numpy as np


@dataclass
class _Args:
    anomap: str
    remove_prefix: str
    add_prefix: str
    evaluate_size: Optional[Tuple[int, int]] = None
    sample_key_pat: Optional[str] = None
    expected_views: Optional[int] = None
    more_anomap: Tuple[str, ...] = tuple()

    def __reduce__(self):
        return partial(type, '_Args', (object,)), (asdict(self),)


def parse_args() -> _Args:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('anomap', type=str, help='''
path to a pickle of a dict[str, tuple[ndarray, float, str | None]], whose key is path to image
of dataset, and values are predicted anomap, predicted image-level score and path to the ground 
truth anomaly mask respectively
                        ''')
    parser.add_argument('--remove_prefix', type=str, default=None,
                        help='remove prefix from path to image & mask first')
    parser.add_argument('--add_prefix', type=str, default=None,
                        help='add another prefix to path to image & mask')
    parser.add_argument('--evaluate_size', type=str, default=None,
                        help='''calculate metrics at specific size rather than original size of
                        images, can be [size] (a single integer) or a [w,h] (a tuple of
                        integers)''')
    parser.add_argument('--sample_key_pat', type=str, default=None,
                        help='regex to capture the sample id from image path, '
                        'for evaluating sample-level metrics')
    parser.add_argument('--expected_views_per_sample', type=int, default=None,
                        help='if given, script will check if each sample '
                        'has the expected views, and raise error if not')
    parser.add_argument('-a', '--more_anomap', type=str, action='append', default=[])
    
    args = parser.parse_args()

    evaluate_size = None
    if args.evaluate_size is not None:
        try:
            size = int(args.evaluate_size)
            evaluate_size = (size, size)
        except ValueError:
            evaluate_size = tuple(int(val) for val in args.evaluate_size.split(','))
            if len(evaluate_size) != 2:
                raise ValueError('evaluate_size must be integer or 2-d tuple of integers')

    return _Args(anomap=args.anomap,
                 remove_prefix=args.remove_prefix,
                 add_prefix=args.add_prefix,
                 evaluate_size=evaluate_size,
                 sample_key_pat=args.sample_key_pat,
                 expected_views=args.expected_views_per_sample,
                 more_anomap=tuple(args.more_anomap))


def load_anomap(path: str) -> Dict[str, Tuple[np.ndarray, float, Optional[str]]]:
    import pickle
    from numbers import Real

    with open(path, 'rb') as fp:
        anomap_set = pickle.load(fp)

    assert isinstance(anomap_set, dict)
    assert all(isinstance(key, str) for key in anomap_set.keys())
    assert all(isinstance(anomap, np.ndarray) and anomap.ndim == 2 and
               isinstance(score, Real) and not np.isnan(score) and
               isinstance(mask, (str, None.__class__))
               for anomap, score, mask in anomap_set.values())

    return anomap_set


def get_gt(path: Union[str, Tuple[int, int]], mask: Optional[str]) -> Tuple[np.ndarray, int]:
    import cv2
    from PIL import Image

    w, h = Image.open(path).size if isinstance(path, str) else path
    if mask is None:
        succ, ret = cv2.imencode('.png', np.zeros((h, w), dtype=np.uint8))
        assert succ
        label = 0
    else:
        data = np.fromfile(mask, dtype=np.uint8)
        maskimg = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        assert maskimg is not None
        if maskimg.shape != (h, w):
            maskimg = cv2.resize(maskimg, (w, h), interpolation=cv2.INTER_NEAREST)
        maskimg = np.where(
            maskimg > 127, 255, np.zeros((), dtype=np.uint8)
        )
        label = 1 if np.any(maskimg) else 0
        succ, ret = cv2.imencode('.png', maskimg)
        assert succ
    return ret, label


def decode_gt(enc: np.ndarray) -> np.ndarray:
    import cv2
    gt_mask = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    assert gt_mask is not None
    return gt_mask


def main():
    import re
    from itertools import groupby
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing.context import SpawnContext
    import os

    from .metrics import (
        compute_imagewise_retrieval_metrics,
        compute_pixelwise_retrieval_metrics,
    )
    from .au_pro import calculate_au_pro


    args = parse_args()
    sample_pat = re.compile(args.sample_key_pat) \
        if args.sample_key_pat is not None else None
 
    def replace(path: Optional[str]) -> Optional[str]:
        if path is None:
            return path
        if args.remove_prefix is not None:
            assert path.startswith(args.remove_prefix)
            path = path[len(args.remove_prefix):]
        if args.add_prefix is not None:
            path = args.add_prefix + path
        return path

    if sample_pat is not None:
        if args.expected_views is not None:
            assert args.expected_views > 0

        def get_sample_id(path: str) -> str:
            mat = sample_pat.search(path)
            assert mat is not None, (path, sample_pat)
            assert len(mat.groups()) == 1
            return mat.groups()[0]
    else:
        def get_sample_id(path: str) -> str:
            return ''

    if args.evaluate_size is not None:
        def interpret_result(dat: Tuple[str, Tuple[np.ndarray, float, Optional[str]]]
                             ) -> Tuple[np.ndarray, float, Tuple[np.ndarray, int], Tuple[str, str]]:
            path, (anomap, score, mask) = dat
            return (anomap, score, get_gt(args.evaluate_size, replace(mask)),
                    (path, get_sample_id(replace(path))))
    else:
        def interpret_result(dat: Tuple[str, Tuple[np.ndarray, float, Optional[str]]]
                             ) -> Tuple[np.ndarray, float, Tuple[np.ndarray, int], Tuple[str, str]]:
            path, (anomap, score, mask) = dat
            return (anomap, score, get_gt(replace(path), replace(mask)),
                    (path, get_sample_id(replace(path))))

    result = load_anomap(args.anomap)
    _ = get_sample_id(next(iter(result.keys())))

    pairs = [(anomap, score,
              get_gt(replace(path) if args.evaluate_size is None
                     else args.evaluate_size, replace(mask)),
              (path, get_sample_id(replace(path))))
             for path, (anomap, score, mask) in result.items()]

    if sample_pat is not None:
        sample_pairs = [([score for _, score, _, _ in views],
                         [label for _, _, (_, label), _ in views],
                         [path for _, _, _, (path, _) in views], key)
                        for key, views in [(key, list(views)) for key, views in groupby(
                             sorted(pairs, key=lambda tup: tup[3][1]), key=lambda tup: tup[3][1]
                        )]]
        if args.expected_views is not None:
            assert all(len(score) == args.expected_views for score, _, _, _ in sample_pairs), \
                [(score, label, path, key) for score, label, path, key in sample_pairs
                 if len(score) != args.expected_views]
        metrics = compute_imagewise_retrieval_metrics(
            [max(score) for score, _, _, _ in sample_pairs],
            [max(label) for _, label, _, _ in sample_pairs],
        )
        s_auroc, s_aupr = metrics['auroc'], metrics['ap']
        # s_auroc, s_aupr = auroc_and_aupr(
        #     np.array([max(score) for score, _, _, _ in sample_pairs]),
        #     np.array([max(label) for _, label, _, _ in sample_pairs]),
        # )

    else:
        s_auroc, s_aupr = float('nan'), float('nan')

    metrics = compute_imagewise_retrieval_metrics(
        [score for _, score, _, _ in pairs],
        [label for _, _, (_, label), _ in pairs],
    )
    i_auroc, i_aupr = metrics['auroc'], metrics['ap']
    # i_auroc, i_aupr = auroc_and_aupr(
    #     np.array([score for _, score, _, _ in pairs]),
    #     np.array([label for _, _, (_, label), _ in pairs])
    # )

    # pixel_metrics = compute_pixelwise_retrieval_metrics(
    #     [cv2.resize(anomap, mask.shape[0:2][::-1], interpolation=cv2.INTER_LINEAR)
    #      for anomap, _, mask in pairs],
    #     [mask for _, _, mask in pairs]
    #     # for debug
    #     # [anomap for anomap, _, _ in pairs],
    #     # [cv2.resize(mask.astype(np.uint8), anomap.shape[0:2][::-1], interpolation=cv2.INTER_NEAREST)
    #     #  for anomap, _, mask in pairs]
    # )
    # p_auroc = pixel_metrics['auroc']
    # p_ap = pixel_metrics['ap']

    def resize_anomap(anomap: np.ndarray, enc_gt: np.ndarray) -> np.ndarray:
        from PIL import Image
        import cv2
        from io import BytesIO
        with BytesIO(enc_gt[0:64].tobytes()) as sio:
            size = Image.open(sio).size
        return cv2.resize(anomap, size, interpolation=cv2.INTER_LINEAR)

    # if HAS_CUDA:
    #     p_auroc, p_ap, p_aupro = auroc_aupr_aupro_cuda(
    #         Preloader(
    #             ReusableGenerator(pairs, lambda tup: resize_anomap(tup[0], tup[2][0]))
    #         ).build_loader(4 if HAS_MP_WITH_LOCALS else 0, 1),
    #         Preloader(
    #             ReusableGenerator(pairs, lambda tup: decode_gt(tup[2][0]))
    #         ).build_loader(8 if HAS_MP_WITH_LOCALS else 0, 1),
    #     )

    # else:
    #     p_auroc, p_ap, p_aupro = auroc_aupr_aupro(
    #         ReusableGenerator(pairs, lambda tup: resize_anomap(tup[0], tup[2][0])),
    #         ReusableGenerator(pairs, lambda tup: decode_gt(tup[2][0])),
    #     )

    processed_anomap = [resize_anomap(anomap, mask) for anomap, _, (mask, _), _ in pairs]
    processed_gt = [np.where(decode_gt(mask) != 0, 1, 0) for _, _, (mask, _), _ in pairs]
    metrics = compute_pixelwise_retrieval_metrics(
        processed_anomap, processed_gt
    )
    p_auroc, p_ap = metrics['auroc'], metrics['ap']
    p_aupro, _ = calculate_au_pro(processed_gt, processed_anomap, num_thresholds=1000)

    print(f'{args.anomap},'
          f's_auroc,{s_auroc},s_aupr,{s_aupr},'
          f'i_auroc,{i_auroc},i_aupr,{i_aupr},'
          f'p_auroc,{p_auroc},p_aupr,{p_ap},p_aupro,{p_aupro}', flush=True)

    if len(args.more_anomap) > 0:
        print('skip other anomaps because we are in reference implementation')

if __name__ == '__main__':
    main()
