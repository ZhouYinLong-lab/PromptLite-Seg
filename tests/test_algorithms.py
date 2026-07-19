from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from promptseg.algorithms import center_color, robust_superpixel
from promptseg.dataset import load_sample


@pytest.mark.parametrize("method", [center_color, robust_superpixel])
def test_algorithm_smoke_on_synthetic_image(method, synthetic_data_dir: Path) -> None:
    sample = load_sample(synthetic_data_dir / "sample_000")

    prediction = method(sample.image, sample.prompt)

    assert prediction.shape == sample.mask.shape
    assert prediction.dtype == np.bool_
    assert prediction[sample.prompt.point[1], sample.prompt.point[0]]
    assert prediction.any()
