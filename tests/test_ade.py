"""Tests for the ADE20K dataset handler.

These tests use local synthetic data only and do not require network access.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from promptseg.ade import (
    _clean_point_from_mask,
    _largest_instance,
    _mask_from_instance_png,
    _tight_bbox,
)


def test_mask_from_instance_png_empty() -> None:
    """An all-black instance PNG yields an all-False mask."""
    png = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")
    mask = _mask_from_instance_png(png)
    assert mask.shape == (32, 32)
    assert mask.dtype == np.bool_
    assert not mask.any()


def test_mask_from_instance_png_partial() -> None:
    """An instance PNG with a white rectangle yields the correct mask."""
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[8:24, 8:24] = 255
    png = Image.fromarray(arr, mode="L")
    mask = _mask_from_instance_png(png)
    assert mask.sum() == 256
    assert mask[8, 8]
    assert not mask[0, 0]


def test_largest_instance_selects_biggest() -> None:
    m1 = np.zeros((32, 32), dtype=bool)
    m1[0:8, 0:8] = True
    m2 = np.zeros((32, 32), dtype=bool)
    m2[0:16, 0:16] = True
    m3 = np.zeros((32, 32), dtype=bool)
    m3[0:4, 0:4] = True
    idx, mask = _largest_instance([m1, m2, m3])
    assert idx == 1
    assert mask.sum() == 256


def test_clean_point_from_square() -> None:
    """Farthest interior point of a solid square is near its centre."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    x, y = _clean_point_from_mask(mask)
    # Should be close to (15.5, 15.5) → (15, 15) or (16, 15)
    assert 14 <= x <= 17
    assert 14 <= y <= 17


def test_clean_point_from_empty_mask() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    x, y = _clean_point_from_mask(mask)
    assert (x, y) == (0, 0)


def test_tight_bbox_square() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    bbox = _tight_bbox(mask)
    assert bbox == (8, 8, 24, 24)


def test_tight_bbox_full() -> None:
    mask = np.ones((32, 32), dtype=bool)
    bbox = _tight_bbox(mask)
    assert bbox == (0, 0, 32, 32)


def test_tight_bbox_empty() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    bbox = _tight_bbox(mask)
    assert bbox == (0, 0, 1, 1)  # fallback non-zero area
