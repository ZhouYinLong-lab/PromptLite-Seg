"""ADE20K dataset handler for zero-shot prompted segmentation.

Uses the ``1aurent/ADE20K`` Hugging Face dataset, which provides images,
instance masks, and object metadata. Targets are constructed by selecting
the largest object instance in each image and deriving a clean point
(distance-transform maximum inside the mask) and box (tight bounding box).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def _mask_from_instance_png(png: Image.Image) -> np.ndarray:
    """Convert a grayscale instance PNG to a boolean mask."""
    arr = np.array(png.convert("L"), dtype=np.uint8)
    return arr > 0


def _largest_instance(masks: list[np.ndarray]) -> tuple[int, np.ndarray]:
    """Return (index, mask) of the largest instance by pixel count."""
    best_idx = 0
    best_mask = masks[0]
    best_area = best_mask.sum()
    for i, m in enumerate(masks[1:], start=1):
        area = m.sum()
        if area > best_area:
            best_area = area
            best_idx = i
            best_mask = m
    return best_idx, best_mask


def _clean_point_from_mask(mask: np.ndarray) -> tuple[int, int]:
    """Farthest interior point (distance-transform maximum)."""
    if not mask.any():
        return (0, 0)
    dt: np.ndarray = ndi.distance_transform_edt(mask)
    y, x = np.unravel_index(np.argmax(dt), dt.shape)
    return (int(x), int(y))


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight axis-aligned bounding box of non-zero pixels."""
    if not mask.any():
        return (0, 0, 1, 1)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    # Ensure non-zero area
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return (x0, y0, x1, y1)


class ADE20KSample:
    """A single ADE20K sample ready for prompted segmentation."""

    __slots__ = ("image", "mask", "point", "bbox", "sample_id", "object_name")

    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        point: tuple[int, int],
        bbox: tuple[int, int, int, int],
        sample_id: str,
        object_name: str = "",
    ):
        self.image = image
        self.mask = mask
        self.point = point
        self.bbox = bbox
        self.sample_id = sample_id
        self.object_name = object_name


def iter_ade20k_validation(
    *,
    max_samples: int | None = None,
    min_mask_area: int = 256,
    seed: int = 20260720,
) -> Iterator[ADE20KSample]:
    """Yield ADE20K validation samples for prompted segmentation.

    Parameters
    ----------
    max_samples : int or None
        Cap the number of returned samples (None = all valid samples).
    min_mask_area : int
        Skip instances smaller than this many pixels.
    seed : int
        RNG seed for shuffling (ensures reproducibility).
    """
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    ds = load_dataset("1aurent/ADE20K", split="validation", streaming=True)
    count = 0

    for row in ds:
        if max_samples is not None and count >= max_samples:
            break

        instances_png = row.get("instances")
        objects_meta = row.get("objects")
        if not instances_png or not objects_meta:
            continue

        masks = [_mask_from_instance_png(png) for png in instances_png]
        if not masks:
            continue

        idx, mask = _largest_instance(masks)
        if mask.sum() < min_mask_area:
            continue

        point = _clean_point_from_mask(mask)
        bbox = _tight_bbox(mask)
        image = np.array(row["image"].convert("RGB"), dtype=np.uint8)

        # Derive a stable sample id from the filename
        filename = row.get("filename", f"ade_val_{count:06d}")
        sample_id = filename.replace(".jpg", "")

        obj_name = ""
        if idx < len(objects_meta):
            obj_name = objects_meta[idx].get("name", "")

        yield ADE20KSample(
            image=image,
            mask=mask,
            point=point,
            bbox=bbox,
            sample_id=sample_id,
            object_name=obj_name,
        )
        count += 1


def load_ade20k_samples(
    *,
    max_samples: int | None = None,
    min_mask_area: int = 256,
    seed: int = 20260720,
) -> list[ADE20KSample]:
    """Materialize ADE20K samples into a list (for multi-pass experiments)."""
    return list(
        iter_ade20k_validation(
            max_samples=max_samples,
            min_mask_area=min_mask_area,
            seed=seed,
        )
    )
