"""Lightweight prompt segmentation algorithms.

Two baselines are provided:
    - ``center_color``: pixel-distance from a seed region's median Lab color.
    - ``robust_superpixel``: superpixel-level feature clustering with
      multi-ratio consensus.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, measure, morphology, segmentation

from .dataset import Prompt
from .utils import clip_bbox

# ---------------------------------------------------------------------------
# Named algorithm parameters (magic constants extracted for discoverability)
# ---------------------------------------------------------------------------

# center_color ---------------------------------------------------------------
# Fraction of the smaller bbox dimension used as the seed disk radius.
CENTER_COLOR_SEED_RADIUS_RATIO: float = 0.025
# Minimum number of foreground seed pixels; fallback uses the prompt point.
CENTER_COLOR_SEED_MIN_PIXELS: int = 1
# Otsu fallback percentile when threshold_otsu fails.
CENTER_COLOR_OTSU_FALLBACK_PERCENTILE: float = 55.0
# Upper clamp on the threshold (percentile of dist values in the box).
CENTER_COLOR_THRESHOLD_UPPER_PERCENTILE: float = 65.0
# Minimum object size for _clean, as fraction of total pixels.
CENTER_COLOR_MIN_SIZE_RATIO: float = 0.0005

# _superpixel_once -----------------------------------------------------------
# Default number of superpixel segments for SLIC.
SLIC_N_SEGMENTS: int = 280
# SLIC compactness parameter (higher = more shape-regular).
SLIC_COMPACTNESS: float = 14.0
# Gaussian smoothing sigma passed to SLIC.
SLIC_SIGMA: float = 1.0
# Spatial feature scale factor (converts normalized coords back to pixels).
SPATIAL_SCALE: float = 25.0
# Radius of the foreground seed disk, as fraction of bbox minor dimension.
SUPERPIXEL_FG_RADIUS_RATIO: float = 0.035
# Border-band width relative to the seed radius.
BORDER_BAND_RADIUS_MULTIPLIER: int = 2
# Blending weight for the spatial prior in probability.
SPATIAL_WEIGHT: float = 0.75
PROB_WEIGHT: float = 0.25
# Probability threshold for superpixel classification.
SUPERPIXEL_PROB_THRESHOLD: float = 0.52
# Proportion of a superpixel that must lie inside the box to be considered.
SEG_INSIDE_BOX_RATIO: float = 0.5

# robust_superpixel ----------------------------------------------------------
# Bbox expansion ratios for the three ensemble members.
BBOX_EXPANSION_RATIOS: tuple[float, float, float] = (-0.04, 0.0, 0.06)
# Minimum object size for _clean in robust mode, as fraction of total pixels.
ROBUST_MIN_SIZE_RATIO: float = 0.0008
# Absolute minimum _clean object size in pixels.
ROBUST_MIN_SIZE_ABS: int = 24

# ---------------------------------------------------------------------------


def _expand_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], ratio: float
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    return clip_bbox(
        (
            int(x0 - bw * ratio),
            int(y0 - bh * ratio),
            int(x1 + bw * ratio),
            int(y1 + bh * ratio),
        ),
        shape,
    )


def _box_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = clip_bbox(bbox, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _disk_mask(shape: tuple[int, int], point: tuple[int, int], radius: int) -> np.ndarray:
    x, y = point
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius**2


def _component_touching_point(mask: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    if not mask.any():
        return mask
    x, y = point
    labels = measure.label(mask, connectivity=1)
    if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] and labels[y, x] > 0:
        return labels == labels[y, x]
    props = measure.regionprops(labels)
    if not props:
        return mask
    px = np.array([p.centroid[1] for p in props])
    py = np.array([p.centroid[0] for p in props])
    idx = int(np.argmin((px - x) ** 2 + (py - y) ** 2))
    return labels == props[idx].label


def _clean(mask: np.ndarray, point: tuple[int, int], min_size: int = 64) -> np.ndarray:
    # scikit-image 0.25 removes components strictly smaller than ``min_size``.
    # Using N + 1 preserves the 0.26 ``max_size=N`` behavior (remove size <= N)
    # while retaining Python 3.10 support, which scikit-image 0.26 dropped.
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size + 1)
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.closing(mask, morphology.disk(2))
    return _component_touching_point(mask, point)


def center_color(image: np.ndarray, prompt: Prompt) -> np.ndarray:
    """A compact baseline: segment pixels in the box by distance to center color."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = clip_bbox(prompt.bbox, (h, w))
    box = _box_mask((h, w), (x0, y0, x1, y1))
    radius = max(2, int(CENTER_COLOR_SEED_RADIUS_RATIO * min(x1 - x0, y1 - y0)))
    seed = _disk_mask((h, w), prompt.point, radius) & box
    if seed.sum() == 0:
        seed[prompt.point[1], prompt.point[0]] = True
    proto = np.median(image[seed].astype(np.float32), axis=0)
    dist = np.linalg.norm(image.astype(np.float32) - proto, axis=2)
    values = dist[box]
    if values.size == 0:
        return np.zeros((h, w), dtype=bool)
    try:
        threshold = filters.threshold_otsu(values)
    except ValueError:
        threshold = float(np.percentile(values, CENTER_COLOR_OTSU_FALLBACK_PERCENTILE))
    threshold = min(float(threshold), float(np.percentile(values, CENTER_COLOR_THRESHOLD_UPPER_PERCENTILE)))
    pred = (dist <= threshold) & box
    return _clean(pred, prompt.point, min_size=max(16, int(CENTER_COLOR_MIN_SIZE_RATIO * h * w)))


def _superpixel_once(
    image: np.ndarray,
    prompt: Prompt,
    bbox_ratio: float,
    n_segments: int = SLIC_N_SEGMENTS,
) -> np.ndarray:
    h, w = image.shape[:2]
    bbox = _expand_bbox(prompt.bbox, (h, w), bbox_ratio)
    x0, y0, x1, y1 = bbox
    box = _box_mask((h, w), bbox)
    lab = color.rgb2lab(image)
    segments = segmentation.slic(
        image,
        n_segments=n_segments,
        compactness=SLIC_COMPACTNESS,
        sigma=SLIC_SIGMA,
        start_label=0,
        channel_axis=-1,
    )
    num_segments = int(segments.max()) + 1

    # ------------------------------------------------------------------
    # Vectorised per-segment feature extraction using ndimage.mean.
    #
    # Each superpixel is described by a 5-d feature vector:
    #   [L, a, b,  scaled_x, scaled_y]
    #
    # Using ndi.mean over the label map is O(num_segments + h*w) rather
    # than O(num_segments * h*w) as the original Python loop was.
    #
    # Note on np.ogrid vs np.mgrid:
    #   np.ogrid produces broadcastable 1-D arrays (2 * ~8 KB at 1024px)
    #   instead of full (h, w) float64 grids (~16 MB at 1024px).
    # ------------------------------------------------------------------
    yy, xx = np.ogrid[:h, :w]
    features = np.zeros((num_segments, 5), dtype=np.float32)
    for c in range(3):
        features[:, c] = ndi.mean(lab[:, :, c], labels=segments, index=np.arange(num_segments))
    features[:, 3] = SPATIAL_SCALE * ndi.mean(xx, labels=segments, index=np.arange(num_segments)) / max(1, w - 1)
    features[:, 4] = SPATIAL_SCALE * ndi.mean(yy, labels=segments, index=np.arange(num_segments)) / max(1, h - 1)

    radius = max(3, int(SUPERPIXEL_FG_RADIUS_RATIO * min(x1 - x0, y1 - y0)))
    fg_pixels = _disk_mask((h, w), prompt.point, radius) & box
    bg_pixels = ~box
    border = np.zeros((h, w), dtype=bool)
    border[y0 : min(y0 + max(BORDER_BAND_RADIUS_MULTIPLIER, radius), h), x0:x1] = True
    border[max(y1 - max(BORDER_BAND_RADIUS_MULTIPLIER, radius), 0) : y1, x0:x1] = True
    border[y0:y1, x0 : min(x0 + max(BORDER_BAND_RADIUS_MULTIPLIER, radius), w)] = True
    border[y0:y1, max(x1 - max(BORDER_BAND_RADIUS_MULTIPLIER, radius), 0) : x1] = True
    bg_pixels |= border

    fg_ids = np.unique(segments[fg_pixels])
    bg_ids = np.unique(segments[bg_pixels])
    if fg_ids.size == 0:
        fg_ids = np.array([segments[prompt.point[1], prompt.point[0]]])
    if bg_ids.size == 0:
        bg_ids = np.setdiff1d(np.arange(num_segments), fg_ids)
    fg_proto = np.median(features[fg_ids], axis=0)
    bg_proto = np.median(features[bg_ids], axis=0)

    fg_dist = np.linalg.norm(features - fg_proto, axis=1)
    bg_dist = np.linalg.norm(features - bg_proto, axis=1)
    prob = bg_dist / (fg_dist + bg_dist + 1e-6)

    cx, cy = prompt.point
    seg_x = features[:, 3] / SPATIAL_SCALE * max(1, w - 1)
    seg_y = features[:, 4] / SPATIAL_SCALE * max(1, h - 1)
    spatial = np.exp(-(((seg_x - cx) / max(1, x1 - x0)) ** 2 + ((seg_y - cy) / max(1, y1 - y0)) ** 2))
    prob = SPATIAL_WEIGHT * prob + PROB_WEIGHT * spatial

    # For each segment, check whether most of its pixels fall inside the box.
    seg_inside = ndi.mean(box.astype(np.float32), labels=segments, index=np.arange(num_segments)) > SEG_INSIDE_BOX_RATIO
    seg_pred = (prob >= SUPERPIXEL_PROB_THRESHOLD) & seg_inside
    seg_pred[fg_ids] = True
    seg_pred[bg_ids] = False
    return seg_pred[segments]


def robust_superpixel(image: np.ndarray, prompt: Prompt) -> np.ndarray:
    """Training-free robust prompt segmentation with superpixel expansion."""
    color_seed = center_color(image, prompt)
    votes = [_superpixel_once(image, prompt, bbox_ratio=ratio) for ratio in BBOX_EXPANSION_RATIOS]
    superpixel_consensus = np.mean(votes, axis=0) >= 0.5
    pred = color_seed | superpixel_consensus
    h, w = image.shape[:2]
    return _clean(pred, prompt.point, min_size=max(ROBUST_MIN_SIZE_ABS, int(ROBUST_MIN_SIZE_RATIO * h * w)))


METHODS = {
    "center_color": center_color,
    "robust_superpixel": robust_superpixel,
}
