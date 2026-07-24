"""ADE20K dataset handler for zero-shot prompted segmentation.

Uses the ``1aurent/ADE20K`` Hugging Face dataset, which provides images,
instance masks, and object metadata. Targets are constructed by selecting
the largest object instance in each image and deriving a clean point
(distance-transform maximum inside the mask) and box (tight bounding box).

The full-stream enumeration scans every source row in the natural streaming
order returned by Hugging Face datasets.  No shuffling or random sampling is
applied during full-stream enumeration — the ``seed`` parameter is only used
when a bounded deterministic subsample of eligible rows is requested via the
materialisation helpers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Internal helpers (unchanged from the original module)
# ---------------------------------------------------------------------------

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
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ADE20KSourceRow:
    """Metadata for a single source row in the ADE20K validation stream.

    Every row that passes through the stream is recorded — eligible or
    excluded — so that the full scan can be audited without holding images
    or masks in memory.
    """

    row_index: int
    sample_id: str
    filename: str
    eligible: bool
    exclusion_reason: str
    image_width: int
    image_height: int
    num_instances: int
    selected_instance_idx: int | None
    selected_mask_area: int | None
    object_name: str | None
    point: tuple[int, int] | None
    bbox: tuple[int, int, int, int] | None

    def to_dict(self) -> dict:
        d = {}
        for field_name in [
            "row_index", "sample_id", "filename", "eligible", "exclusion_reason",
            "image_width", "image_height", "num_instances", "selected_instance_idx",
            "selected_mask_area", "object_name",
        ]:
            val = getattr(self, field_name)
            if isinstance(val, (np.integer,)):
                val = int(val)
            d[field_name] = val
        # point and bbox may contain numpy ints
        if self.point is not None:
            d["point"] = [int(x) for x in self.point]
        else:
            d["point"] = None
        if self.bbox is not None:
            d["bbox"] = [int(x) for x in self.bbox]
        else:
            d["bbox"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ADE20KSourceRow":
        pt = d.get("point")
        bb = d.get("bbox")
        return cls(
            row_index=int(d["row_index"]),
            sample_id=str(d["sample_id"]),
            filename=str(d["filename"]),
            eligible=bool(d["eligible"]),
            exclusion_reason=str(d["exclusion_reason"]),
            image_width=int(d["image_width"]),
            image_height=int(d["image_height"]),
            num_instances=int(d["num_instances"]),
            selected_instance_idx=(
                int(d["selected_instance_idx"])
                if d.get("selected_instance_idx") is not None
                else None
            ),
            selected_mask_area=(
                int(d["selected_mask_area"])
                if d.get("selected_mask_area") is not None
                else None
            ),
            object_name=str(d["object_name"]) if d.get("object_name") is not None else None,
            point=(
                (int(pt[0]), int(pt[1]))
                if pt is not None and len(pt) == 2
                else None
            ),
            bbox=(
                (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                if bb is not None and len(bb) == 4
                else None
            ),
        )


class ADE20KSample:
    """A single ADE20K sample ready for prompted segmentation."""

    __slots__ = ("image", "mask", "point", "bbox", "sample_id", "object_name", "source_row_index")

    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        point: tuple[int, int],
        bbox: tuple[int, int, int, int],
        sample_id: str,
        object_name: str = "",
        source_row_index: int = -1,
    ):
        self.image = image
        self.mask = mask
        self.point = point
        self.bbox = bbox
        self.sample_id = sample_id
        self.object_name = object_name
        self.source_row_index = source_row_index


# ---------------------------------------------------------------------------
# Full-stream enumeration (data-free — no image/mask arrays retained)
# ---------------------------------------------------------------------------

def enumerate_ade20k_stream(
    *,
    min_mask_area: int = 256,
    max_rows: int | None = None,
) -> Iterator[ADE20KSourceRow]:
    """Enumerate every source row in the ADE20K validation split.

    Yields an ``ADE20KSourceRow`` for **every** row encountered in the
    streaming dataset, regardless of eligibility.  No shuffling or random
    sampling is applied — rows appear in the natural order returned by the
    Hugging Face ``datasets`` library.

    Parameters
    ----------
    min_mask_area : int
        Minimum pixel count for the largest-instance mask (default 256).
    max_rows : int or None
        If set, stop scanning after this many source rows (useful for
        bounded pilot runs).  Otherwise scan the entire validation split.

    Yields
    ------
    ADE20KSourceRow
        One record per source row, with eligibility and (when eligible)
        prompt geometry.
    """
    from datasets import load_dataset

    ds = load_dataset("1aurent/ADE20K", split="validation", streaming=True)

    for row_index, row in enumerate(ds):
        if max_rows is not None and row_index >= max_rows:
            break

        filename = row.get("filename", f"ade_val_{row_index:06d}")
        sample_id = filename.replace(".jpg", "")

        instances_png = row.get("instances")
        objects_meta = row.get("objects")

        # Image dimensions from the PIL image (don't keep the array)
        pil_image = row["image"]
        image_width = pil_image.width
        image_height = pil_image.height

        # --- Exclusion: no instances or objects ---
        if not instances_png or not objects_meta:
            yield ADE20KSourceRow(
                row_index=row_index,
                sample_id=sample_id,
                filename=filename,
                eligible=False,
                exclusion_reason="no_instances_or_objects_metadata",
                image_width=image_width,
                image_height=image_height,
                num_instances=0,
                selected_instance_idx=None,
                selected_mask_area=None,
                object_name=None,
                point=None,
                bbox=None,
            )
            continue

        masks = [_mask_from_instance_png(png) for png in instances_png]
        num_instances = len(masks)

        # --- Exclusion: no valid instance masks ---
        if num_instances == 0:
            yield ADE20KSourceRow(
                row_index=row_index,
                sample_id=sample_id,
                filename=filename,
                eligible=False,
                exclusion_reason="no_valid_instance_masks",
                image_width=image_width,
                image_height=image_height,
                num_instances=0,
                selected_instance_idx=None,
                selected_mask_area=None,
                object_name=None,
                point=None,
                bbox=None,
            )
            continue

        idx, mask = _largest_instance(masks)
        mask_area = int(mask.sum())

        # Derive object name early (used even for excluded rows)
        obj_name = ""
        if idx < len(objects_meta):
            obj_name = objects_meta[idx].get("name", "")

        # --- Exclusion: mask too small ---
        if mask_area < min_mask_area:
            yield ADE20KSourceRow(
                row_index=row_index,
                sample_id=sample_id,
                filename=filename,
                eligible=False,
                exclusion_reason=f"mask_area_below_minimum",
                image_width=image_width,
                image_height=image_height,
                num_instances=num_instances,
                selected_instance_idx=idx,
                selected_mask_area=mask_area,
                object_name=obj_name,
                point=None,
                bbox=None,
            )
            continue

        # --- Eligible ---
        point = _clean_point_from_mask(mask)
        bbox = _tight_bbox(mask)

        yield ADE20KSourceRow(
            row_index=row_index,
            sample_id=sample_id,
            filename=filename,
            eligible=True,
            exclusion_reason="",
            image_width=image_width,
            image_height=image_height,
            num_instances=num_instances,
            selected_instance_idx=idx,
            selected_mask_area=mask_area,
            object_name=obj_name,
            point=point,
            bbox=bbox,
        )


# ---------------------------------------------------------------------------
# Manifest I/O (data-free JSONL)
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION = 1


def write_manifest(rows: list[ADE20KSourceRow], path: Path) -> str:
    """Write a data-free JSONL manifest and return its SHA-256 hex digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"schema_version": MANIFEST_SCHEMA_VERSION, **r.to_dict()}, sort_keys=True)
             for r in rows]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> list[ADE20KSourceRow]:
    """Read a JSONL manifest back into ``ADE20KSourceRow`` objects."""
    rows: list[ADE20KSourceRow] = []
    seen_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        row = ADE20KSourceRow.from_dict(obj)
        if row.sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id in manifest: {row.sample_id}")
        seen_ids.add(row.sample_id)
        rows.append(row)
    return rows


def eligible_rows(manifest: list[ADE20KSourceRow]) -> list[ADE20KSourceRow]:
    """Return only the eligible rows from a manifest."""
    return [r for r in manifest if r.eligible]


# ---------------------------------------------------------------------------
# Deterministic bounded sampling (used ONLY for pilot / bounded modes)
# ---------------------------------------------------------------------------

def _deterministic_sample(
    eligible: list[ADE20KSourceRow],
    max_samples: int,
    seed: int,
) -> list[ADE20KSourceRow]:
    """Select *max_samples* rows from *eligible* using a deterministic RNG.

    This is used for bounded pilot runs; it does **not** shuffle the
    streaming order but draws a reproducible subset via index selection.
    """
    if max_samples >= len(eligible):
        return list(eligible)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(eligible), size=max_samples, replace=False)
    # Sort to preserve the original stream order within the sample
    indices_sorted = sorted(indices)
    return [eligible[i] for i in indices_sorted]


# ---------------------------------------------------------------------------
# Materialisation (images + masks → ADE20KSample) with local caching
# ---------------------------------------------------------------------------

def _sample_from_dataset_row(
    row: ADE20KSourceRow,
    ds_row: dict,
) -> ADE20KSample:
    """Convert a streamed dataset row using the selection frozen in the manifest."""
    instances_png = ds_row.get("instances")
    masks = [_mask_from_instance_png(png) for png in instances_png]
    idx = row.selected_instance_idx
    if idx is None or idx >= len(masks):
        raise ValueError(f"Manifest instance index is invalid for {row.sample_id}")
    mask = masks[idx]

    image = np.array(ds_row["image"].convert("RGB"), dtype=np.uint8)

    return ADE20KSample(
        image=image,
        mask=mask,
        point=row.point if row.point else (0, 0),
        bbox=row.bbox if row.bbox else (0, 0, 1, 1),
        sample_id=row.sample_id,
        object_name=row.object_name or "",
        source_row_index=row.row_index,
    )


def _cache_path(cache_dir: Path, row: ADE20KSourceRow) -> Path:
    key = hashlib.sha256(row.sample_id.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{row.row_index:06d}_{key}.npz"


def _load_cached_sample(path: Path, row: ADE20KSourceRow) -> ADE20KSample:
    with np.load(path, allow_pickle=False) as cached:
        source_row_index = int(cached["source_row_index"])
        sample_id = str(cached["sample_id"])
        if source_row_index != row.row_index or sample_id != row.sample_id:
            raise ValueError(f"Cached ADE20K sample does not match manifest row {row.sample_id}")
        return ADE20KSample(
            image=cached["image"],
            mask=cached["mask"].astype(bool),
            point=row.point or (0, 0),
            bbox=row.bbox or (0, 0, 1, 1),
            sample_id=row.sample_id,
            object_name=row.object_name or "",
            source_row_index=row.row_index,
        )


def _write_cached_sample(path: Path, sample: ADE20KSample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        image=sample.image,
        mask=sample.mask.astype(np.uint8),
        sample_id=np.asarray(sample.sample_id),
        source_row_index=np.asarray(sample.source_row_index, dtype=np.int64),
    )
    temporary.replace(path)


def materialize_samples(
    rows: list[ADE20KSourceRow],
    *,
    cache_dir: Path | None = None,
) -> Iterator[ADE20KSample]:
    """Materialize eligible source rows into ``ADE20KSample`` objects.

    Images and masks are fetched from Hugging Face datasets (network
    required) and optionally cached to *cache_dir* for resume safety.

    Parameters
    ----------
    rows : list of ADE20KSourceRow
        Eligible source rows to materialize.
    cache_dir : Path or None
        If provided, downloaded images/masks are saved here so that
        interrupted runs can resume without re-downloading.  This
        directory should be Git-ignored.

    Yields
    ------
    ADE20KSample
    """
    selected = [row for row in rows if row.eligible]
    if not selected:
        return
    if any(a.row_index >= b.row_index for a, b in zip(selected, selected[1:])):
        raise ValueError("Rows must be supplied once each in increasing source-row order")

    cached: dict[int, ADE20KSample] = {}
    missing: dict[int, ADE20KSourceRow] = {}
    if cache_dir is not None:
        for row in selected:
            path = _cache_path(cache_dir, row)
            if path.exists():
                cached[row.row_index] = _load_cached_sample(path, row)
            else:
                missing[row.row_index] = row
    else:
        missing = {row.row_index: row for row in selected}

    streamed: dict[int, ADE20KSample] = {}
    if missing:
        from datasets import load_dataset

        last_needed = max(missing)
        dataset = load_dataset("1aurent/ADE20K", split="validation", streaming=True)
        for row_index, ds_row in enumerate(dataset):
            if row_index > last_needed:
                break
            row = missing.get(row_index)
            if row is None:
                continue
            sample = _sample_from_dataset_row(row, ds_row)
            streamed[row_index] = sample
            if cache_dir is not None:
                _write_cached_sample(_cache_path(cache_dir, row), sample)

        absent = sorted(set(missing) - set(streamed))
        if absent:
            raise RuntimeError(f"ADE20K stream ended before source rows {absent[:5]}")

    for row in selected:
        yield cached.get(row.row_index) or streamed[row.row_index]


# ---------------------------------------------------------------------------
# Backward-compatible convenience wrappers
# ---------------------------------------------------------------------------

def iter_ade20k_validation(
    *,
    max_samples: int | None = None,
    min_mask_area: int = 256,
    seed: int = 20260720,
) -> Iterator[ADE20KSample]:
    """Yield ADE20K validation samples for prompted segmentation.

    .. deprecated::
        Prefer :func:`enumerate_ade20k_stream` for new code.  This wrapper
        exists for backward compatibility only.  The ``seed`` parameter
        does **not** shuffle the streaming order — rows always appear in
        the natural order of the Hugging Face dataset split.

    Parameters
    ----------
    max_samples : int or None
        Cap the number of returned samples (None = all eligible samples).
        When a cap is requested, eligible rows are deterministically
        subsampled using *seed* rather than taking the first *N*.
    min_mask_area : int
        Skip instances smaller than this many pixels.
    seed : int
        RNG seed used only when ``max_samples`` limits the eligible set.
        Full-stream enumeration ignores this parameter.
    """
    # Stream all rows to collect eligible ones (network access is required).
    manifest = list(enumerate_ade20k_stream(min_mask_area=min_mask_area))
    eligible = eligible_rows(manifest)

    if max_samples is not None and max_samples < len(eligible):
        eligible = _deterministic_sample(eligible, max_samples, seed)

    yield from materialize_samples(eligible)


def load_ade20k_samples(
    *,
    max_samples: int | None = None,
    min_mask_area: int = 256,
    seed: int = 20260720,
) -> list[ADE20KSample]:
    """Materialize ADE20K samples into a list (for multi-pass experiments).

    .. deprecated::
        Prefer the manifest + materialize pipeline for new code.
    """
    return list(
        iter_ade20k_validation(
            max_samples=max_samples,
            min_mask_area=min_mask_area,
            seed=seed,
        )
    )
