from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from promptseg.algorithms import (
    CONFIRMATORY_CPU_METHODS,
    center_color,
    robust_superpixel,
)
from promptseg.dataset import load_sample


@pytest.mark.parametrize("method", [center_color, robust_superpixel])
def test_algorithm_smoke_on_synthetic_image(method, synthetic_data_dir: Path) -> None:
    sample = load_sample(synthetic_data_dir / "sample_000")

    prediction = method(sample.image, sample.prompt)

    assert prediction.shape == sample.mask.shape
    assert prediction.dtype == np.bool_
    assert prediction[sample.prompt.point[1], sample.prompt.point[0]]


@pytest.mark.parametrize("method_name", CONFIRMATORY_CPU_METHODS)
def test_confirmatory_cpu_method_contract(method_name: str, synthetic_data_dir: Path) -> None:
    sample = load_sample(synthetic_data_dir / "sample_000")

    prediction = CONFIRMATORY_CPU_METHODS[method_name](sample.image, sample.prompt)

    assert prediction.dtype == np.bool_
    assert prediction.shape == sample.mask.shape
    assert prediction[sample.prompt.point[1], sample.prompt.point[0]]
    assert prediction.any()
