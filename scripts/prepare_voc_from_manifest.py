"""Materialize local VOC samples from a committed data-free protocol manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.dataset import VOC_CLASSES
from promptseg.voc import bbox_and_point, decode_image, decode_voc_mask, largest_component, read_voc_table


def load_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Manifest contains duplicate sample IDs: {path}")
    return rows


def verify_target(row: dict, label: int, component: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    bbox, point = bbox_and_point(component)
    expected = {
        "label": label,
        "class_name": VOC_CLASSES[label],
        "target_area": int(component.sum()),
        "image_height": int(component.shape[0]),
        "image_width": int(component.shape[1]),
        "bbox": list(bbox),
        "point": list(point),
    }
    mismatches = {key: (row.get(key), value) for key, value in expected.items() if row.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Manifest target mismatch for {row['sample_id']}: {mismatches}")
    return bbox, point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-semantic-mask", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    if args.output_dir.exists():
        if not args.replace:
            raise SystemExit(f"Output already exists: {args.output_dir}; pass --replace to rebuild")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    table = read_voc_table(args.parquet, columns=["image", "mask"])
    for index, manifest_row in enumerate(rows, start=1):
        source_row = int(manifest_row["source_row"])
        source = table.slice(source_row, 1).to_pylist()[0]
        image = decode_image(source["image"]).convert("RGB")
        semantic_mask = decode_voc_mask(decode_image(source["mask"]))
        selected = largest_component(semantic_mask)
        if selected is None:
            raise RuntimeError(f"No target for {manifest_row['sample_id']}")
        label, component = selected
        bbox, point = verify_target(manifest_row, label, component)

        sample_dir = args.output_dir / manifest_row["sample_id"]
        sample_dir.mkdir()
        image.save(sample_dir / "image.jpg", quality=95)
        Image.fromarray(component.astype(np.uint8) * 255).save(sample_dir / "target_mask.png")
        if args.include_semantic_mask:
            Image.fromarray(semantic_mask.astype(np.uint8)).save(sample_dir / "semantic_mask.png")
        (sample_dir / "prompt.txt").write_text(
            "\n".join(
                [
                    f"source_split={manifest_row['source_split']}",
                    f"source_row={source_row}",
                    f"label={label}",
                    f"class_name={VOC_CLASSES[label]}",
                    f"bbox={','.join(str(value) for value in bbox)}",
                    f"point={point[0]},{point[1]}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        if index % 100 == 0 or index == len(rows):
            print(f"Prepared {index}/{len(rows)} samples")


if __name__ == "__main__":
    main()

