from __future__ import annotations

import numpy as np


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, target).sum() / union)


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(pred, target).sum() / denom)

