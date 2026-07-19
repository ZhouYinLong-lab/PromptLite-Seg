"""Shared constants, helpers, and utilities for PromptLite-Seg experiments."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

__all__ = [
    "SEVERITIES",
    "clip_bbox",
    "write_csv",
    "stable_rng",
]

# Severity levels for prompt perturbation experiments.
# Each entry defines the noise scale (as fraction of bbox width/height)
# for point and box perturbations.
SEVERITIES: dict[str, dict[str, float]] = {
    "clean": {"point": 0.00, "box": 0.00},
    "mild": {"point": 0.05, "box": 0.06},
    "moderate": {"point": 0.10, "box": 0.12},
}


def clip_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Clip a bounding box to lie within the given (height, width) shape."""
    h, w = shape
    if h < 1 or w < 1:
        raise ValueError(f"Image shape must be positive, got {shape}.")
    x0, y0, x1, y1 = bbox
    x0 = int(np.clip(x0, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    y1 = int(np.clip(y1, y0 + 1, h))
    return x0, y0, x1, y1


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dict rows as a CSV file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Write an empty file with no header to avoid crashes.
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stable_rng(*parts: object) -> np.random.Generator:
    """Deterministic RNG seeded from a SHA-256 hash of the given parts.

    Unlike ``abs(hash(...))``, this is platform- and interpreter-invariant,
    ensuring reproducible samples across different Python runs.
    """
    key = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    seed = int.from_bytes(digest[:8], byteorder="little") % (2**32)
    return np.random.default_rng(seed)
