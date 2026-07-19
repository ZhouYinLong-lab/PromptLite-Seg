from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    sample_dir = tmp_path / "sample_000"
    sample_dir.mkdir()

    image = np.zeros((48, 48, 3), dtype=np.uint8)
    image[:, :] = (20, 25, 30)
    image[10:38, 12:36] = (210, 45, 35)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[10:38, 12:36] = 255

    Image.fromarray(image).save(sample_dir / "image.jpg", quality=100)
    Image.fromarray(mask).save(sample_dir / "target_mask.png")
    (sample_dir / "prompt.txt").write_text(
        "source_row=0\n"
        "label=1\n"
        "class_name=aeroplane\n"
        "bbox=12,10,36,38\n"
        "point=24,24\n",
        encoding="utf-8",
    )
    return tmp_path
