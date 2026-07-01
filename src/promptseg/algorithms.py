from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, measure, morphology, segmentation

from .dataset import Prompt


def _clip_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = bbox
    x0 = int(np.clip(x0, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    y1 = int(np.clip(y1, y0 + 1, h))
    return x0, y0, x1, y1


def _expand_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], ratio: float
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    return _clip_bbox(
        (
            int(x0 - bw * ratio),
            int(y0 - bh * ratio),
            int(x1 + bw * ratio),
            int(y1 + bh * ratio),
        ),
        shape,
    )


def _box_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = _clip_bbox(bbox, shape)
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
    mask = morphology.remove_small_objects(mask.astype(bool), max_size=min_size)
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.closing(mask, morphology.disk(2))
    return _component_touching_point(mask, point)


def center_color(image: np.ndarray, prompt: Prompt) -> np.ndarray:
    """A compact baseline: segment pixels in the box by distance to center color."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(prompt.bbox, (h, w))
    box = _box_mask((h, w), (x0, y0, x1, y1))
    radius = max(2, int(0.025 * min(x1 - x0, y1 - y0)))
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
        threshold = float(np.percentile(values, 55))
    threshold = min(float(threshold), float(np.percentile(values, 65)))
    pred = (dist <= threshold) & box
    return _clean(pred, prompt.point, min_size=max(16, int(0.0005 * h * w)))


def _superpixel_once(
    image: np.ndarray,
    prompt: Prompt,
    bbox_ratio: float,
    n_segments: int = 280,
) -> np.ndarray:
    h, w = image.shape[:2]
    bbox = _expand_bbox(prompt.bbox, (h, w), bbox_ratio)
    x0, y0, x1, y1 = bbox
    box = _box_mask((h, w), bbox)
    lab = color.rgb2lab(image)
    segments = segmentation.slic(
        image,
        n_segments=n_segments,
        compactness=14,
        sigma=1,
        start_label=0,
        channel_axis=-1,
    )
    num_segments = int(segments.max()) + 1
    yy, xx = np.mgrid[:h, :w]
    features = np.zeros((num_segments, 5), dtype=np.float32)
    for seg_id in range(num_segments):
        region = segments == seg_id
        features[seg_id, :3] = lab[region].mean(axis=0)
        features[seg_id, 3] = 25.0 * xx[region].mean() / max(1, w - 1)
        features[seg_id, 4] = 25.0 * yy[region].mean() / max(1, h - 1)

    radius = max(3, int(0.035 * min(x1 - x0, y1 - y0)))
    fg_pixels = _disk_mask((h, w), prompt.point, radius) & box
    bg_pixels = ~box
    border = np.zeros((h, w), dtype=bool)
    border[y0 : min(y0 + max(2, radius), h), x0:x1] = True
    border[max(y1 - max(2, radius), 0) : y1, x0:x1] = True
    border[y0:y1, x0 : min(x0 + max(2, radius), w)] = True
    border[y0:y1, max(x1 - max(2, radius), 0) : x1] = True
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
    seg_x = features[:, 3] / 25.0 * max(1, w - 1)
    seg_y = features[:, 4] / 25.0 * max(1, h - 1)
    spatial = np.exp(-(((seg_x - cx) / max(1, x1 - x0)) ** 2 + ((seg_y - cy) / max(1, y1 - y0)) ** 2))
    prob = 0.75 * prob + 0.25 * spatial

    seg_inside = np.array([box[segments == seg_id].mean() > 0.5 for seg_id in range(num_segments)])
    seg_pred = (prob >= 0.52) & seg_inside
    seg_pred[fg_ids] = True
    seg_pred[bg_ids] = False
    return seg_pred[segments]


def robust_superpixel(image: np.ndarray, prompt: Prompt) -> np.ndarray:
    """Training-free robust prompt segmentation with superpixel expansion."""
    color_seed = center_color(image, prompt)
    votes = [_superpixel_once(image, prompt, bbox_ratio=ratio) for ratio in (-0.04, 0.0, 0.06)]
    superpixel_consensus = np.mean(votes, axis=0) >= 0.5
    pred = color_seed | superpixel_consensus
    h, w = image.shape[:2]
    return _clean(pred, prompt.point, min_size=max(24, int(0.0008 * h * w)))


METHODS = {
    "center_color": center_color,
    "robust_superpixel": robust_superpixel,
}
