"""Small, testable adapter around the Segment Anything predictor contract."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import numpy as np

from .dataset import Prompt


PROMPT_MODES = ("point_only", "box_only", "point_box")


def _inference_context() -> ContextManager:
    """Use inference mode when PyTorch is installed, while keeping mocks lightweight."""

    try:
        import torch
    except ModuleNotFoundError:
        return nullcontext()
    return torch.inference_mode()


def predict_sam(predictor, prompt: Prompt, prompt_mode: str = "point_box") -> tuple[np.ndarray, float]:
    """Run a SAM-compatible predictor and return its highest-scored mask.

    The adapter intentionally depends only on the public ``predict`` contract, so
    every prompt modality can be tested with a CPU-only fake predictor.
    """

    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unknown SAM prompt mode: {prompt_mode}")

    point_coords: np.ndarray | None = None
    point_labels: np.ndarray | None = None
    box: np.ndarray | None = None
    if prompt_mode in {"point_only", "point_box"}:
        point_coords = np.array([prompt.point], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
    if prompt_mode in {"box_only", "point_box"}:
        box = np.array(prompt.bbox, dtype=np.float32)

    with _inference_context():
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool), float(scores[best_idx])

