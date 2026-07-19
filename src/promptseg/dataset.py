"""Dataset loading utilities for the PASCAL VOC 2012 subset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = [
    "VOC_CLASSES",
    "Prompt",
    "Sample",
    "load_sample",
    "iter_samples",
]

VOC_CLASSES: dict[int, str] = {
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
    """Load a single sample from its directory.

    Args:
        sample_dir: Path to a ``sample_*`` directory containing
            ``image.jpg``, ``target_mask.png``, and ``prompt.txt``.

    Returns:
        A populated ``Sample`` instance.

    Raises:
        FileNotFoundError: If any required file is missing.
        ValueError: If the prompt metadata is malformed.
    """
    image_path = sample_dir / "image.jpg"
    mask_path = sample_dir / "target_mask.png"
    prompt_path = sample_dir / "prompt.txt"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing image file {image_path} in sample directory {sample_dir}."
        )
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Missing mask file {mask_path} in sample directory {sample_dir}."
        )
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Missing prompt file {prompt_path} in sample directory {sample_dir}."
        )

    try:
        image = np.asarray(Image.open(image_path).convert("RGB"))
    except Exception as exc:
        raise ValueError(
            f"Failed to load or decode image {image_path}: {exc}"
        ) from exc

    try:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    except Exception as exc:
        raise ValueError(
            f"Failed to load or decode mask {mask_path}: {exc}"
        ) from exc

    try:
        meta: dict[str, str] = {}
        for line in prompt_path.read_text(encoding="utf-8").splitlines():
            key, value = line.split("=", 1)
            meta[key] = value
    except Exception as exc:
        raise ValueError(
            f"Failed to parse prompt file {prompt_path}: {exc}"
        ) from exc

    try:
        bbox = tuple(int(x) for x in meta["bbox"].split(","))
        point = tuple(int(x) for x in meta["point"].split(","))
        label = int(meta["label"])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Malformed metadata in {prompt_path}. "
            f"Expected 'bbox', 'point', and 'label' keys. Error: {exc}"
        ) from exc

    prompt = Prompt(
        bbox=bbox,
        point=point,
        label=label,
        class_name=VOC_CLASSES.get(label, f"class_{label}"),
    )
    return Sample(sample_id=sample_dir.name, image=image, mask=mask, prompt=prompt)


def iter_samples(data_dir: str | Path):
    """Yield ``Sample`` instances for every valid sample directory in *data_dir*.

    Directories that fail to load are logged (via print) and skipped so a
    single corrupt sample does not crash the entire experiment loop.
    """
    root = Path(data_dir)
    for sample_dir in sorted(root.glob("sample_*")):
        required = [sample_dir / "image.jpg", sample_dir / "target_mask.png", sample_dir / "prompt.txt"]
        if all(p.exists() for p in required):
            try:
                yield load_sample(sample_dir)
            except (FileNotFoundError, ValueError) as exc:
                print(f"WARNING: Skipping {sample_dir} due to error: {exc}")
