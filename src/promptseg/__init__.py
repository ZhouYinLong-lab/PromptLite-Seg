"""Prompted segmentation utilities for the IntroAI course project."""

from __future__ import annotations

from .algorithms import METHODS, center_color, robust_superpixel
from .dataset import Prompt, Sample, iter_samples, load_sample
from .metrics import dice, iou
from .utils import SEVERITIES, clip_bbox, stable_rng, write_csv
from .visualize import draw_metric_summary, draw_prediction_figure, overlay

__all__ = [
    "METHODS",
    "center_color",
    "robust_superpixel",
    "Prompt",
    "Sample",
    "iter_samples",
    "load_sample",
    "dice",
    "iou",
    "SEVERITIES",
    "clip_bbox",
    "stable_rng",
    "write_csv",
    "draw_metric_summary",
    "draw_prediction_figure",
    "overlay",
]
