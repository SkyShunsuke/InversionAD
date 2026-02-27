
import torch
import numpy as np
import torch.nn.functional as F
import numpy as np

from src.dist_utils import concat_all_gather

from sklearn.metrics import roc_auc_score, average_precision_score
from src.adeval import EvalAccumulatorCuda, f1_max_gpu_hist

SUPPORTED_METRICS = [
    'img_auroc', 'img_aupr', 'img_f1max', 'img_ap',
    'px_auroc', 'px_aupr', 'px_f1max', 'px_ap', 'px_aupro'
]

@torch.no_grad()
def extract_features(
    fe: torch.nn.Module,
    imgs: torch.Tensor,
    device: torch.device,
):
    """Extract features using the feature extractor.
    Args:
        fe: Feature extractor model.
        imgs: Input images tensor of shape (B, C, H, W).
        device: Device to perform computation on.
    Returns:
        torch.Tensor: Extracted features tensor.
    """
    imgs = imgs.to(device, non_blocking=True)
    features = fe(imgs)
    return features

def aggregate_px_values(
    agg_method: str,
    px_values: np.ndarray,
):
    """Aggregate pixel-wise values to a single image-level score.
    Args:
        agg_method: Aggregation method ('diff', 'max', 'mean', 'median').
        px_values: Pixel-wise values of shape (N, H, W).
    Returns:
        np.ndarray: Aggregated image-level scores of shape (N,).
    """
    if agg_method == 'diff':
        scores_min = px_values.min(axis=(1, 2))
        scores_max = px_values.max(axis=(1, 2))
        scores = scores_max - scores_min
    elif agg_method == 'sum':
        scores = px_values.sum(axis=(1, 2))
    elif agg_method == 'max':
        scores = px_values.max(axis=(1, 2))
    elif agg_method == 'mean':
        scores = px_values.mean(axis=(1, 2))
    elif agg_method == 'median':
        scores = np.median(px_values, axis=(1, 2))
    elif agg_method == 'diff+sum':
        scores_min = px_values.min(axis=(1, 2))
        scores_max = px_values.max(axis=(1, 2))
        scores_diff = scores_max - scores_min
        scores_sum = px_values.sum(axis=(1, 2))
        normalized_diff = (scores_diff - scores_diff.min()) / (scores_diff.max() - scores_diff.min() + 1e-8)
        normalized_sum = (scores_sum - scores_sum.min()) / (scores_sum.max() - scores_sum.min() + 1e-8)
        scores = normalized_diff + normalized_sum
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")
    return scores

def calculate_img_metrics(
    gt_labels: np.ndarray,
    pred_scores: np.ndarray,
    metrics: list,
):
    """Calculate image-level anomaly detection metrics.
    Args:
        gt_labels: Ground truth labels of shape (N,).
        pred_scores: Predicted anomaly scores of shape (N,).
        metrics: List of metrics to compute ('auroc', 'aupr', 'f1').
    Returns:
        dict: Dictionary containing computed metrics.
    """
    results_dict = {}
    if 'img_auroc' in metrics:
        auroc = roc_auc_score(gt_labels, pred_scores)
        results_dict['img_auroc'] = auroc
    if 'img_aupr' in metrics or 'img_ap' in metrics:
        aupr = average_precision_score(gt_labels, pred_scores)
        if 'img_aupr' in metrics:
            results_dict['img_aupr'] = aupr
        if 'img_ap' in metrics:
            results_dict['img_ap'] = aupr
    if 'img_f1max' in metrics:
        scores_tensor = torch.from_numpy(pred_scores).to(torch.float32).cuda()
        labels_tensor = torch.from_numpy(gt_labels).to(torch.float32).cuda()
        f1, _ = f1_max_gpu_hist(scores_tensor, labels_tensor)
        results_dict['img_f1max'] = f1.item()
        
    return results_dict

def calculate_px_metrics(
    gt_masks: np.ndarray,
    pred_scores: np.ndarray,
    metrics: list,
    device: torch.device = torch.device('cuda'),
    distributed: bool = False,
    accum_size: int = 100,
    eps: float = 1e-8,
):
    """Calculate pixel-level anomaly detection metrics.
    Args:
        gt_masks: Ground truth masks of shape (N, H, W).
        pred_scores: Predicted anomaly scores of shape (N, H, W).
        metrics: List of metrics to compute ('auroc', 'aupr', 'f1').
    Returns:
        dict: Dictionary containing computed metrics.
    """
    results_dict = {}
    gt_masks_flat = gt_masks.reshape(-1)
    pred_scores_flat = pred_scores.reshape(-1)
    score_min, score_max = pred_scores_flat.min() - eps, pred_scores_flat.max() + eps

    # -- use adeval for AUROC and AUPRO, AUPR
    nb = len(gt_masks) // accum_size + (1 if len(gt_masks) % accum_size != 0 else 0)
    evaluator = EvalAccumulatorCuda(score_min, score_max, score_min, score_max)
    for i in range(0, nb):
        start_idx = i * accum_size
        end_idx = min((i + 1) * accum_size, len(gt_masks))
        gt_batch = torch.from_numpy((gt_masks[start_idx:end_idx] > 0).astype(np.uint8)).to(device)
        score_batch = torch.from_numpy(pred_scores[start_idx:end_idx]).to(device)
        evaluator.add_anomap_batch(score_batch, gt_batch)
    results = evaluator.summary()
    if 'px_auroc' in metrics:
        results_dict['px_auroc'] = results['p_auroc']
    if 'px_aupro' in metrics:
        results_dict['px_aupro'] = results['p_aupro']
    if 'px_aupr' in metrics:
        results_dict['px_aupr'] = results['p_aupr']
    
    # -- use skleran for AP
    if 'px_ap' in metrics:
        gt_bin = (gt_masks_flat > 0).astype(int)
        ap = average_precision_score(gt_bin, pred_scores_flat)
        results_dict['px_ap'] = ap
        
    # -- use f1_max_gpu_hist for F1-max
    if 'px_f1max' in metrics:
        gt_masks_flat = (gt_masks_flat > 0).astype(float)
        scores_tensor = torch.from_numpy(pred_scores_flat).to(torch.float32).to(device)
        labels_tensor = torch.from_numpy(gt_masks_flat).to(torch.float32).to(device)
        f1, _ = f1_max_gpu_hist(scores_tensor, labels_tensor, distributed=distributed)
        results_dict['px_f1max'] = f1.item()
        
    return results_dict

def divide_by_class(array, labels):
    """Divide array by class labels.
    Args:
        array: Numpy array of (N, ...).
        labels: Numpy array of class labels (N,).
    Returns:
        dict: Dictionary mapping class label to corresponding array.
    """
    class_dict = {}
    for label in np.unique(labels):
        class_dict[label] = array[labels == label]
    return class_dict

