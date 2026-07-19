"""PASCAL VOC parquet decoding and target-construction utilities."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image
from scipy import ndimage as ndi


HF_DATA_ROOT = "https://huggingface.co/datasets/nateraw/pascal-voc-2012/resolve/main/data"
VOC_PARQUET_URLS = {
    "train": f"{HF_DATA_ROOT}/train-00000-of-00001.parquet",
    "val": f"{HF_DATA_ROOT}/val-00000-of-00001.parquet",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    partial = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(out_path)


def read_voc_table(path: Path, columns: list[str] | None = None):
    return pq.read_table(path, columns=columns or ["image", "mask"])


def decode_image(cell: dict) -> Image.Image:
    if cell.get("bytes") is not None:
        return Image.open(io.BytesIO(cell["bytes"]))
    if cell.get("path") is not None:
        return Image.open(cell["path"])
    raise ValueError("Unsupported Hugging Face image cell.")


def voc_palette() -> dict[tuple[int, int, int], int]:
    palette = {}
    for label in range(256):
        red = green = blue = 0
        class_id = label
        for bit in range(8):
            red |= ((class_id >> 0) & 1) << (7 - bit)
            green |= ((class_id >> 1) & 1) << (7 - bit)
            blue |= ((class_id >> 2) & 1) << (7 - bit)
            class_id >>= 3
        palette[(red, green, blue)] = label
    return palette


def decode_voc_mask(image: Image.Image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8)
    rgb = np.asarray(image.convert("RGB"))
    label_map = np.zeros(rgb.shape[:2], dtype=np.uint8)
    palette = voc_palette()
    flat = rgb.reshape(-1, 3)
    output = label_map.reshape(-1)
    for color in np.unique(flat, axis=0):
        key = tuple(int(value) for value in color)
        output[np.all(flat == color, axis=1)] = palette.get(key, 255)
    return label_map


def largest_component(mask: np.ndarray) -> tuple[int, np.ndarray] | None:
    best_label: int | None = None
    best_component: np.ndarray | None = None
    best_area = 0
    for label in sorted(int(value) for value in np.unique(mask) if 1 <= int(value) <= 20):
        labeled, count = ndi.label(mask == label)
        for component_id in range(1, count + 1):
            component = labeled == component_id
            area = int(component.sum())
            if area > best_area:
                best_area = area
                best_label = label
                best_component = component
    if best_label is None or best_component is None:
        return None
    return best_label, best_component


def bbox_and_point(component: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    ys, xs = np.where(component)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    distance = ndi.distance_transform_edt(component)
    point_y, point_x = np.unravel_index(int(distance.argmax()), distance.shape)
    return (x0, y0, x1, y1), (int(point_x), int(point_y))

