from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from promptseg.dataset import iter_samples, load_sample
from promptseg.utils import clip_bbox, stable_rng


@pytest.mark.parametrize(
    ("bbox", "shape", "expected"),
    [
        ((-5, -2, 12, 9), (8, 10), (0, 0, 10, 8)),
        ((3, 3, 3, 3), (5, 5), (3, 3, 4, 4)),
        ((-10, -10, 10, 10), (1, 1), (0, 0, 1, 1)),
    ],
)
def test_clip_bbox_handles_boundaries(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    expected: tuple[int, int, int, int],
) -> None:
    assert clip_bbox(bbox, shape) == expected


def test_clip_bbox_rejects_empty_images() -> None:
    with pytest.raises(ValueError, match="positive"):
        clip_bbox((0, 0, 1, 1), (0, 1))


def test_stable_rng_is_deterministic() -> None:
    first = stable_rng("sample_001", "moderate", 2).normal(size=8)
    second = stable_rng("sample_001", "moderate", 2).normal(size=8)
    different = stable_rng("sample_001", "moderate", 3).normal(size=8)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_load_sample_parses_prompt_and_files(synthetic_data_dir: Path) -> None:
    sample = load_sample(synthetic_data_dir / "sample_000")

    assert sample.sample_id == "sample_000"
    assert sample.image.shape == (48, 48, 3)
    assert sample.mask.shape == (48, 48)
    assert sample.prompt.bbox == (12, 10, 36, 38)
    assert sample.prompt.point == (24, 24)
    assert sample.prompt.label == 1
    assert sample.prompt.class_name == "aeroplane"


def test_iter_samples_uses_sample_directories(synthetic_data_dir: Path) -> None:
    (synthetic_data_dir / "notes").mkdir()
    (synthetic_data_dir / "sample_incomplete").mkdir()

    samples = list(iter_samples(synthetic_data_dir))

    assert [sample.sample_id for sample in samples] == ["sample_000"]


def test_load_sample_rejects_malformed_prompt(synthetic_data_dir: Path) -> None:
    prompt_path = synthetic_data_dir / "sample_000" / "prompt.txt"
    prompt_path.write_text("label=1\npoint=24,24\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed metadata"):
        load_sample(synthetic_data_dir / "sample_000")
