from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/voc_subset"))
    parser.add_argument("--start-row", type=int, default=0)
    args = parser.parse_args()

    parquet_path = args.cache_dir / "pascal_voc_2012_val.parquet"
    print(f"Downloading or reusing {parquet_path} ...")
    download_file(VOC_PARQUET_URLS["val"], parquet_path)

    table = read_voc_table(parquet_path, columns=["image", "mask"])
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
