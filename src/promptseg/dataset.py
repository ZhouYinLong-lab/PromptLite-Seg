from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


VOC_CLASSES = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}


@dataclass(frozen=True)
class Prompt:
    bbox: tuple[int, int, int, int]
    point: tuple[int, int]
    label: int
    class_name: str


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: np.ndarray
    mask: np.ndarray
    prompt: Prompt


def load_sample(sample_dir: Path) -> Sample:
    image = np.asarray(Image.open(sample_dir / "image.jpg").convert("RGB"))
    mask = np.asarray(Image.open(sample_dir / "target_mask.png").convert("L")) > 0
    meta = {}
    for line in (sample_dir / "prompt.txt").read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        meta[key] = value
    bbox = tuple(int(x) for x in meta["bbox"].split(","))
    point = tuple(int(x) for x in meta["point"].split(","))
    label = int(meta["label"])
    prompt = Prompt(
        bbox=bbox,
        point=point,
        label=label,
        class_name=VOC_CLASSES.get(label, f"class_{label}"),
    )
    return Sample(sample_id=sample_dir.name, image=image, mask=mask, prompt=prompt)


def iter_samples(data_dir: str | Path):
    root = Path(data_dir)
    for sample_dir in sorted(root.glob("sample_*")):
        if (sample_dir / "image.jpg").exists() and (sample_dir / "target_mask.png").exists():
            yield load_sample(sample_dir)

