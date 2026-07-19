"""Build immutable, data-free manifests for tuning and confirmatory VOC splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.dataset import VOC_CLASSES
from promptseg.voc import (
    VOC_PARQUET_URLS,
    bbox_and_point,
    decode_image,
    decode_voc_mask,
    download_file,
    largest_component,
    read_voc_table,
    sha256_file,
)


def inventory_split(parquet_path: Path, split: str) -> list[dict]:
    table = read_voc_table(parquet_path, columns=["mask"])
    rows: list[dict] = []
    for row_index in range(table.num_rows):
        cell = table.slice(row_index, 1).to_pylist()[0]["mask"]
        semantic_mask = decode_voc_mask(decode_image(cell))
        selected = largest_component(semantic_mask)
        if selected is None:
            raise RuntimeError(f"VOC {split} row {row_index} has no foreground component")
        label, component = selected
        bbox, point = bbox_and_point(component)
        rows.append(
            {
                "sample_id": f"{split}_{row_index:06d}",
                "source_split": split,
                "source_row": row_index,
                "label": label,
                "class_name": VOC_CLASSES[label],
                "target_area": int(component.sum()),
                "image_height": int(component.shape[0]),
                "image_width": int(component.shape[1]),
                "bbox": list(bbox),
                "point": list(point),
            }
        )
    return rows


def stratified_tuning_sample(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["label"])].append(row)
    missing = [label for label in range(1, 21) if len(grouped[label]) < per_class]
    if missing:
        raise RuntimeError(f"Not enough tuning candidates for labels: {missing}")

    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    for label in range(1, 21):
        candidates = grouped[label]
        indices = rng.choice(len(candidates), size=per_class, replace=False)
        selected.extend(candidates[int(index)] for index in indices)
    return sorted(selected, key=lambda row: int(row["source_row"]))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def split_summary(rows: list[dict]) -> dict:
    counts = Counter(row["class_name"] for row in rows)
    return {
        "num_samples": len(rows),
        "num_classes": len(counts),
        "class_counts": dict(sorted(counts.items())),
        "source_rows_sha256": __import__("hashlib").sha256(
            ",".join(str(row["source_row"]) for row in rows).encode("ascii")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("protocol/manifests"))
    parser.add_argument("--tuning-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()

    paths = {
        split: args.cache_dir / f"pascal_voc_2012_{split}.parquet"
        for split in ("train", "val")
    }
    for split, path in paths.items():
        download_file(VOC_PARQUET_URLS[split], path)

    train_rows = inventory_split(paths["train"], "train")
    validation_rows = inventory_split(paths["val"], "val")
    tuning_rows = stratified_tuning_sample(train_rows, args.tuning_per_class, args.seed)

    tuning_path = args.output_dir / "tuning_train.jsonl"
    validation_path = args.output_dir / "confirmatory_validation.jsonl"
    write_jsonl(tuning_path, tuning_rows)
    write_jsonl(validation_path, validation_rows)

    payload = {
        "seed": args.seed,
        "tuning_per_class": args.tuning_per_class,
        "sources": {
            split: {
                "url": VOC_PARQUET_URLS[split],
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for split, path in paths.items()
        },
        "tuning": {
            **split_summary(tuning_rows),
            "manifest": tuning_path.as_posix(),
            "manifest_sha256": sha256_file(tuning_path),
        },
        "confirmatory": {
            **split_summary(validation_rows),
            "manifest": validation_path.as_posix(),
            "manifest_sha256": sha256_file(validation_path),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

