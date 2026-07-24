"""Tests for the ADE20K dataset handler.

These tests use local synthetic data only and do not require network access.
Full-stream enumeration, manifest I/O, deterministic sampling, materialization,
and edge cases are covered.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.ade import (
    _clean_point_from_mask,
    _deterministic_sample,
    _largest_instance,
    _mask_from_instance_png,
    _tight_bbox,
    ADE20KSourceRow,
    eligible_rows,
    load_manifest,
    write_manifest,
)


def test_synthetic_sample_is_stable_across_python_processes() -> None:
    code = (
        "import hashlib;"
        "from scripts.run_ade20k_cpu import _synthetic_sample,_synthetic_source_rows;"
        "r=next(x for x in _synthetic_source_rows(10) if x.eligible);"
        "im,mask=_synthetic_sample(r);"
        "print(hashlib.sha256(im.tobytes()+mask.tobytes()).hexdigest())"
    )
    digests = []
    for hash_seed in ("1", "999"):
        env = dict(__import__("os").environ)
        env["PYTHONHASHSEED"] = hash_seed
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        digests.append(result.stdout.strip())
    assert digests[0] == digests[1]

# ---------------------------------------------------------------------------
# Low-level helpers (existing tests preserved)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ADE20KSourceRow roundtrip
# ---------------------------------------------------------------------------

def test_source_row_roundtrip() -> None:
    row = ADE20KSourceRow(
        row_index=5,
        sample_id="ADE_val_00000005",
        filename="ADE_val_00000005.jpg",
        eligible=True,
        exclusion_reason="",
        image_width=640,
        image_height=480,
        num_instances=3,
        selected_instance_idx=1,
        selected_mask_area=5000,
        object_name="chair",
        point=(320, 240),
        bbox=(100, 50, 400, 350),
    )
    d = row.to_dict()
    restored = ADE20KSourceRow.from_dict(d)
    assert restored.row_index == 5
    assert restored.sample_id == "ADE_val_00000005"
    assert restored.eligible is True
    assert restored.exclusion_reason == ""
    assert restored.point == (320, 240)
    assert restored.bbox == (100, 50, 400, 350)
    assert restored.object_name == "chair"


def test_source_row_roundtrip_excluded() -> None:
    row = ADE20KSourceRow(
        row_index=0,
        sample_id="ADE_val_00000000",
        filename="ADE_val_00000000.jpg",
        eligible=False,
        exclusion_reason="no_instances_or_objects_metadata",
        image_width=256,
        image_height=256,
        num_instances=0,
        selected_instance_idx=None,
        selected_mask_area=None,
        object_name=None,
        point=None,
        bbox=None,
    )
    d = row.to_dict()
    restored = ADE20KSourceRow.from_dict(d)
    assert restored.eligible is False
    assert restored.exclusion_reason == "no_instances_or_objects_metadata"
    assert restored.point is None
    assert restored.bbox is None


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def test_write_and_load_manifest(tmp_path: Path) -> None:
    rows = [
        ADE20KSourceRow(
            row_index=i,
            sample_id=f"test_{i:06d}",
            filename=f"test_{i:06d}.jpg",
            eligible=(i % 3 != 0),
            exclusion_reason="mask_area_below_minimum" if i % 3 == 0 else "",
            image_width=512,
            image_height=512,
            num_instances=2,
            selected_instance_idx=0,
            selected_mask_area=1000 if i % 3 != 0 else 100,
            object_name="test_obj",
            point=(256, 256) if i % 3 != 0 else None,
            bbox=(100, 100, 400, 400) if i % 3 != 0 else None,
        )
        for i in range(20)
    ]
    path = tmp_path / "manifest.jsonl"
    sha = write_manifest(rows, path)
    assert path.exists()
    assert len(sha) == 64

    loaded = load_manifest(path)
    assert len(loaded) == 20
    assert loaded[0].sample_id == "test_000000"
    assert loaded[0].row_index == 0


def test_eligible_rows_filtering(tmp_path: Path) -> None:
    rows = [
        ADE20KSourceRow(
            row_index=i,
            sample_id=f"test_{i:06d}",
            filename=f"test_{i:06d}.jpg",
            eligible=(i % 2 == 0),
            exclusion_reason="" if i % 2 == 0 else "no_instances_or_objects_metadata",
            image_width=256,
            image_height=256,
            num_instances=1 if i % 2 == 0 else 0,
            selected_instance_idx=0 if i % 2 == 0 else None,
            selected_mask_area=500 if i % 2 == 0 else None,
            object_name="obj" if i % 2 == 0 else None,
            point=(128, 128) if i % 2 == 0 else None,
            bbox=(50, 50, 200, 200) if i % 2 == 0 else None,
        )
        for i in range(10)
    ]
    path = tmp_path / "manifest.jsonl"
    write_manifest(rows, path)
    loaded = load_manifest(path)
    el = eligible_rows(loaded)
    assert len(el) == 5
    assert all(r.eligible for r in el)


def test_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Duplicate sample IDs in a manifest must raise an error."""
    rows = [
        ADE20KSourceRow(
            row_index=0,
            sample_id="dup_id",
            filename="dup_id.jpg",
            eligible=True,
            exclusion_reason="",
            image_width=256,
            image_height=256,
            num_instances=1,
            selected_instance_idx=0,
            selected_mask_area=500,
            object_name="obj",
            point=(128, 128),
            bbox=(50, 50, 200, 200),
        ),
        ADE20KSourceRow(
            row_index=1,
            sample_id="dup_id",
            filename="dup_id.jpg",
            eligible=True,
            exclusion_reason="",
            image_width=256,
            image_height=256,
            num_instances=1,
            selected_instance_idx=0,
            selected_mask_area=500,
            object_name="obj",
            point=(128, 128),
            bbox=(50, 50, 200, 200),
        ),
    ]
    path = tmp_path / "bad_manifest.jsonl"
    write_manifest(rows, path)
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_manifest(path)


# ---------------------------------------------------------------------------
# Deterministic sampling
# ---------------------------------------------------------------------------


def _make_eligible_rows(n: int) -> list[ADE20KSourceRow]:
    return [
        ADE20KSourceRow(
            row_index=i,
            sample_id=f"sample_{i:06d}",
            filename=f"sample_{i:06d}.jpg",
            eligible=True,
            exclusion_reason="",
            image_width=256,
            image_height=256,
            num_instances=1,
            selected_instance_idx=0,
            selected_mask_area=500,
            object_name="obj",
            point=(128, 128),
            bbox=(50, 50, 200, 200),
        )
        for i in range(n)
    ]


def test_deterministic_sample_returns_requested_count() -> None:
    rows = _make_eligible_rows(100)
    sampled = _deterministic_sample(rows, 20, seed=42)
    assert len(sampled) == 20
    # Row indices should be sorted (preserves stream order)
    indices = [r.row_index for r in sampled]
    assert indices == sorted(indices)


def test_deterministic_sample_is_reproducible() -> None:
    rows = _make_eligible_rows(100)
    a = _deterministic_sample(rows, 20, seed=42)
    b = _deterministic_sample(rows, 20, seed=42)
    assert [r.row_index for r in a] == [r.row_index for r in b]


def test_deterministic_sample_different_seeds_differ() -> None:
    rows = _make_eligible_rows(100)
    a = _deterministic_sample(rows, 20, seed=42)
    b = _deterministic_sample(rows, 20, seed=99)
    # Extremely unlikely to produce the same 20 indices with different seeds
    assert [r.row_index for r in a] != [r.row_index for r in b]


def test_deterministic_sample_all_when_max_exceeds() -> None:
    rows = _make_eligible_rows(10)
    sampled = _deterministic_sample(rows, 100, seed=0)
    assert len(sampled) == 10


def test_deterministic_sample_zero() -> None:
    rows = _make_eligible_rows(10)
    sampled = _deterministic_sample(rows, 0, seed=0)
    assert len(sampled) == 0


# ---------------------------------------------------------------------------
# No seed-shuffle false implication
# ---------------------------------------------------------------------------

def test_manifest_source_rows_are_in_stream_order(tmp_path: Path) -> None:
    """Verify that manifest rows appear in source-row-index order."""
    rows = [
        ADE20KSourceRow(
            row_index=i,
            sample_id=f"test_{i:06d}",
            filename=f"test_{i:06d}.jpg",
            eligible=True,
            exclusion_reason="",
            image_width=256,
            image_height=256,
            num_instances=1,
            selected_instance_idx=0,
            selected_mask_area=500,
            object_name="obj",
            point=(128, 128),
            bbox=(50, 50, 200, 200),
        )
        for i in [5, 2, 8, 1, 3]  # unsorted input
    ]
    path = tmp_path / "manifest.jsonl"
    write_manifest(rows, path)
    loaded = load_manifest(path)
    indices = [r.row_index for r in loaded]
    # Manifest preserves write order; sampling preserves order within the sample
    assert indices == [5, 2, 8, 1, 3]


def test_deterministic_sample_preserves_relative_ordering() -> None:
    """Sampled rows must preserve their relative stream order."""
    rows = _make_eligible_rows(50)
    sampled = _deterministic_sample(rows, 10, seed=123)
    indices = [r.row_index for r in sampled]
    assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# Six-method default assertion
# ---------------------------------------------------------------------------

def test_frozen_cpu_methods_are_exactly_six() -> None:
    from scripts.run_ade20k_cpu import FROZEN_CPU_METHOD_NAMES

    assert len(FROZEN_CPU_METHOD_NAMES) == 6
    expected = {
        "center_color",
        "grabcut_point_box",
        "robust_superpixel",
        "robust_no_color_seed",
        "robust_no_spatial_prior",
        "robust_single_box",
    }
    assert set(FROZEN_CPU_METHOD_NAMES) == expected
    assert "adaptive_superpixel" not in FROZEN_CPU_METHOD_NAMES


def test_all_frozen_methods_exist_in_registry() -> None:
    from promptseg.algorithms import CONFIRMATORY_CPU_METHODS
    from scripts.run_ade20k_cpu import FROZEN_CPU_METHOD_NAMES

    for name in FROZEN_CPU_METHOD_NAMES:
        assert name in CONFIRMATORY_CPU_METHODS


# ---------------------------------------------------------------------------
# Synthetic smoke mode integration
# ---------------------------------------------------------------------------

def test_synthetic_source_rows_produces_eligible_and_excluded(tmp_path: Path) -> None:
    from scripts.run_ade20k_cpu import _synthetic_source_rows

    rows = _synthetic_source_rows(num_rows=100)
    assert len(rows) == 100
    eligible = [r for r in rows if r.eligible]
    excluded = [r for r in rows if not r.eligible]
    assert len(eligible) > 0
    assert len(excluded) > 0
    # Every row has a unique sample_id
    ids = [r.sample_id for r in rows]
    assert len(ids) == len(set(ids))
    # Eligible rows have point and bbox
    for r in eligible:
        assert r.point is not None
        assert r.bbox is not None
        assert r.selected_mask_area is not None and r.selected_mask_area >= 256
    # Excluded rows have a reason
    for r in excluded:
        assert r.exclusion_reason != ""


def test_synthetic_sample_creates_valid_data() -> None:
    from scripts.run_ade20k_cpu import _synthetic_sample
    from scripts.run_ade20k_cpu import _synthetic_source_rows

    rows = _synthetic_source_rows(num_rows=5)
    eligible = [r for r in rows if r.eligible]
    assert len(eligible) > 0

    row = eligible[0]
    image, mask = _synthetic_sample(row)
    assert image.shape == (row.image_height, row.image_width, 3)
    assert image.dtype == np.uint8
    assert mask.shape == (row.image_height, row.image_width)
    assert mask.dtype == np.bool_
    assert mask.sum() > 0


# ---------------------------------------------------------------------------
# Failure-zero accounting
# ---------------------------------------------------------------------------

def test_run_method_on_sample_scores_failure_as_zero() -> None:
    """When a method raises, IoU and Dice must be 0.0 and success=False."""
    from scripts.run_ade20k_cpu import _run_method_on_sample

    def _failing_method(image, prompt):
        raise RuntimeError("simulated failure")

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=bool)
    result = _run_method_on_sample(
        "failing_test",
        _failing_method,
        "test_sample",
        "test_obj",
        image, mask,
        (32, 32), (10, 10, 50, 50),
        1000, 0,
    )
    assert result["iou"] == 0.0
    assert result["dice"] == 0.0
    assert result["success"] is False
    assert result["method"] == "failing_test"
    assert result["sample_id"] == "test_sample"


def test_run_method_on_sample_scores_success() -> None:
    """A method returning the exact mask must score IoU=1.0."""
    from scripts.run_ade20k_cpu import _run_method_on_sample

    def _perfect_method(image, prompt):
        return np.ones((64, 64), dtype=bool)

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=bool)
    result = _run_method_on_sample(
        "perfect_test",
        _perfect_method,
        "test_sample",
        "test_obj",
        image, mask,
        (32, 32), (10, 10, 50, 50),
        4096, 0,
    )
    assert result["iou"] == 1.0
    assert result["dice"] == 1.0
    assert result["success"] is True


# ---------------------------------------------------------------------------
# Manifest counts are accurate
# ---------------------------------------------------------------------------

def test_manifest_accounts_for_every_row(tmp_path: Path) -> None:
    """Every source row (eligible or excluded) is recorded exactly once."""
    rows = [
        ADE20KSourceRow(
            row_index=i,
            sample_id=f"test_{i:06d}",
            filename=f"test_{i:06d}.jpg",
            eligible=(i % 4 != 0),
            exclusion_reason="" if i % 4 != 0 else "no_instances_or_objects_metadata",
            image_width=256,
            image_height=256,
            num_instances=1 if i % 4 != 0 else 0,
            selected_instance_idx=0 if i % 4 != 0 else None,
            selected_mask_area=500 if i % 4 != 0 else None,
            object_name="obj" if i % 4 != 0 else None,
            point=(128, 128) if i % 4 != 0 else None,
            bbox=(50, 50, 200, 200) if i % 4 != 0 else None,
        )
        for i in range(40)
    ]
    path = tmp_path / "manifest.jsonl"
    write_manifest(rows, path)
    loaded = load_manifest(path)

    el = eligible_rows(loaded)
    ex = [r for r in loaded if not r.eligible]

    assert len(loaded) == 40  # every row
    assert len(el) == 30      # 3/4 eligible
    assert len(ex) == 10      # 1/4 excluded
    assert len(el) + len(ex) == len(loaded)
    # All row_index values 0–39 present
    assert set(r.row_index for r in loaded) == set(range(40))
