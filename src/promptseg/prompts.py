"""Deterministic prompt perturbations and observable prompt-quality metrics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .dataset import Prompt
from .metrics import iou
from .utils import SEVERITIES, clip_bbox, stable_rng


NOISE_SOURCES = ("point_noise", "box_noise", "point_box_noise")


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 1.0


def point_hits_target(point: tuple[int, int], target: np.ndarray) -> bool:
    x, y = point
    return bool(0 <= y < target.shape[0] and 0 <= x < target.shape[1] and target[y, x])


def perturb_prompt(
    prompt: Prompt,
    shape: tuple[int, int],
    *,
    point_scale: float,
    box_scale: float,
    noise_source: str,
    trial: int,
    sample_id: str,
    seed_namespace: str = "promptlite-seg-calibration-v1",
) -> Prompt:
    """Perturb a prompt using independently calibrated point and box scales."""

    if noise_source not in NOISE_SOURCES:
        raise ValueError(f"Unknown noise source: {noise_source}")
    height, width = shape
    x0, y0, x1, y1 = prompt.bbox
    box_width = max(1, x1 - x0)
    box_height = max(1, y1 - y0)
    rng = stable_rng(seed_namespace, sample_id, noise_source, trial)

    point_x, point_y = prompt.point
    if noise_source in {"point_noise", "point_box_noise"}:
        point_x = int(np.clip(point_x + round(rng.normal(0, point_scale * box_width)), 0, width - 1))
        point_y = int(np.clip(point_y + round(rng.normal(0, point_scale * box_height)), 0, height - 1))

    bbox = prompt.bbox
    if noise_source in {"box_noise", "point_box_noise"}:
        translate_x = int(round(rng.normal(0, box_scale * box_width)))
        translate_y = int(round(rng.normal(0, box_scale * box_height)))
        grow_left = int(round(rng.normal(0, box_scale * box_width)))
        grow_top = int(round(rng.normal(0, box_scale * box_height)))
        grow_right = int(round(rng.normal(0, box_scale * box_width)))
        grow_bottom = int(round(rng.normal(0, box_scale * box_height)))
        bbox = clip_bbox(
            (
                x0 + translate_x - grow_left,
                y0 + translate_y - grow_top,
                x1 + translate_x + grow_right,
                y1 + translate_y + grow_bottom,
            ),
            shape,
        )
    return replace(prompt, bbox=bbox, point=(point_x, point_y))


def perturb_by_severity(
    prompt: Prompt,
    shape: tuple[int, int],
    severity: str,
    noise_source: str,
    trial: int,
    sample_id: str,
) -> Prompt:
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity: {severity}")
    specification = SEVERITIES[severity]
    return perturb_prompt(
        prompt,
        shape,
        point_scale=specification["point"],
        box_scale=specification["box"],
        noise_source=noise_source,
        trial=trial,
        sample_id=sample_id,
    )


def jitter_around_observed_prompt(
    prompt: Prompt,
    shape: tuple[int, int],
    severity: str,
    trial: int,
    variant_id: int,
    sample_id: str,
) -> Prompt:
    """Sample a local candidate without consulting the unavailable clean prompt."""

    specification = SEVERITIES[severity]
    height, width = shape
    x0, y0, x1, y1 = prompt.bbox
    box_width = max(1, x1 - x0)
    box_height = max(1, y1 - y0)
    rng = stable_rng(sample_id, "ensemble", severity, trial, variant_id)
    point_scale = 0.55 * specification["point"]
    point_x = int(np.clip(prompt.point[0] + round(rng.normal(0, point_scale * box_width)), 0, width - 1))
    point_y = int(np.clip(prompt.point[1] + round(rng.normal(0, point_scale * box_height)), 0, height - 1))
    box_scale = 0.55 * specification["box"]
    translate_x = int(round(rng.normal(0, box_scale * box_width)))
    translate_y = int(round(rng.normal(0, box_scale * box_height)))
    grow_left = int(round(rng.normal(0, box_scale * box_width)))
    grow_top = int(round(rng.normal(0, box_scale * box_height)))
    grow_right = int(round(rng.normal(0, box_scale * box_width)))
    grow_bottom = int(round(rng.normal(0, box_scale * box_height)))
    bbox = clip_bbox(
        (
            x0 + translate_x - grow_left,
            y0 + translate_y - grow_top,
            x1 + translate_x + grow_right,
            y1 + translate_y + grow_bottom,
        ),
        shape,
    )
    return replace(prompt, bbox=bbox, point=(point_x, point_y))


def mask_iou_matrix(masks: list[np.ndarray]) -> np.ndarray:
    count = len(masks)
    output = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            value = iou(masks[first], masks[second])
            output[first, second] = value
            output[second, first] = value
    return output


def select_consistency_medoid(masks: list[np.ndarray]) -> np.ndarray:
    if len(masks) == 1:
        return masks[0]
    pairwise = mask_iou_matrix(masks)
    scores = (pairwise.sum(axis=1) - 1.0) / (len(masks) - 1)
    return masks[int(np.argmax(scores))]


def select_ensemble_predictions(
    candidate_masks: list[np.ndarray],
    candidate_scores: list[float],
    target: np.ndarray,
) -> dict[str, tuple[np.ndarray, float]]:
    """Return deployable ensemble selections plus an explicit oracle bound."""

    if not candidate_masks or len(candidate_masks) != len(candidate_scores):
        raise ValueError("candidate masks and scores must be non-empty and aligned")
    oracle_index = int(np.argmax([iou(mask, target) for mask in candidate_masks]))
    return {
        "sam_single_noisy": (candidate_masks[0], candidate_scores[0]),
        "sam_score_select": (
            candidate_masks[int(np.argmax(candidate_scores))],
            max(candidate_scores),
        ),
        "sam_consistency_medoid": (
            select_consistency_medoid(candidate_masks),
            float(np.mean(candidate_scores)),
        ),
        "sam_vote_consensus": (
            np.mean(np.stack(candidate_masks, axis=0), axis=0) >= 0.5,
            float(np.mean(candidate_scores)),
        ),
        "sam_oracle_best": (candidate_masks[oracle_index], candidate_scores[oracle_index]),
    }
