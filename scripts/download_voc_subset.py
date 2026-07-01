from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.dataset import VOC_CLASSES


VAL_PARQUET_URL = (
    "https://huggingface.co/datasets/nateraw/pascal-voc-2012/resolve/main/"
    "data/val-00000-of-00001.parquet"
)


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def decode_image(cell: dict) -> Image.Image:
    if cell.get("bytes") is not None:
        return Image.open(io.BytesIO(cell["bytes"]))
    if cell.get("path") is not None:
        return Image.open(cell["path"])
    raise ValueError("Unsupported Hugging Face image cell.")


def voc_palette() -> dict[tuple[int, int, int], int]:
    palette = {}
    for label in range(256):
        r = g = b = 0
        cid = label
        for bit in range(8):
            r |= ((cid >> 0) & 1) << (7 - bit)
            g |= ((cid >> 1) & 1) << (7 - bit)
            b |= ((cid >> 2) & 1) << (7 - bit)
            cid >>= 3
        palette[(r, g, b)] = label
    return palette


def decode_voc_mask(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    arr = np.asarray(image.convert("RGB"))
    label_map = np.zeros(arr.shape[:2], dtype=np.uint8)
    palette = voc_palette()
    flat = arr.reshape(-1, 3)
    out = label_map.reshape(-1)
    colors = np.unique(flat, axis=0)
    for color in colors:
        key = tuple(int(v) for v in color)
        out[np.all(flat == color, axis=1)] = palette.get(key, 255)
    return label_map


def largest_component(mask: np.ndarray) -> tuple[int, np.ndarray] | None:
    best_label = None
    best_component = None
    best_area = 0
    for label in sorted(int(x) for x in np.unique(mask) if 1 <= int(x) <= 20):
        labeled, count = ndi.label(mask == label)
        for idx in range(1, count + 1):
            component = labeled == idx
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
    dist = ndi.distance_transform_edt(component)
    py, px = np.unravel_index(int(dist.argmax()), dist.shape)
    return (x0, y0, x1, y1), (int(px), int(py))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/voc_subset"))
    parser.add_argument("--start-row", type=int, default=0)
    args = parser.parse_args()

    parquet_path = args.cache_dir / "pascal_voc_2012_val.parquet"
    print(f"Downloading or reusing {parquet_path} ...")
    download_file(VAL_PARQUET_URL, parquet_path)

    table = pq.read_table(parquet_path, columns=["image", "mask"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    row_count = table.num_rows
    for row_idx in range(args.start_row, row_count):
        row = table.slice(row_idx, 1).to_pylist()[0]
        image = decode_image(row["image"]).convert("RGB")
        raw_mask = decode_image(row["mask"])
        mask = decode_voc_mask(raw_mask)
        selected = largest_component(mask)
        if selected is None:
            continue
        label, component = selected
        if component.sum() < 256:
            continue
        sample_id = f"sample_{written:03d}"
        sample_dir = args.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        bbox, point = bbox_and_point(component)
        image.save(sample_dir / "image.jpg", quality=95)
        Image.fromarray((component.astype(np.uint8) * 255)).save(sample_dir / "target_mask.png")
        Image.fromarray(mask.astype(np.uint8)).save(sample_dir / "semantic_mask.png")
        (sample_dir / "prompt.txt").write_text(
            "\n".join(
                [
                    f"source_row={row_idx}",
                    f"label={label}",
                    f"class_name={VOC_CLASSES.get(label, f'class_{label}')}",
                    f"bbox={','.join(str(v) for v in bbox)}",
                    f"point={point[0]},{point[1]}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {sample_id} from row {row_idx}: {VOC_CLASSES.get(label, label)}")
        written += 1
        if written >= args.count:
            break
    print(f"Prepared {written} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
