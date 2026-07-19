"""Materialize local VOC samples from a committed data-free protocol manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.dataset import VOC_CLASSES
from promptseg.voc import bbox_and_point, decode_image, decode_voc_mask, largest_component, read_voc_table


SAMPLE_ID_PATTERN = re.compile(r"(?:train|val)_\d{6}\Z")
OUTPUT_MARKER = ".promptseg-materialization.json"
OUTPUT_OWNER = "promptlite-seg.prepare-voc-from-manifest"


def load_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Manifest contains duplicate sample IDs: {path}")
    return rows


def safe_output_root(path: Path) -> Path:
    is_junction = getattr(path, "is_junction", lambda: False)
    if path.is_symlink() or is_junction():
        raise ValueError(f"Output directory must not be a symbolic link or junction: {path}")
    resolved = path.resolve()
    repository = ROOT.resolve()
    data_root = (repository / "data").resolve()
    home = Path.home().resolve()
    forbidden = {Path(resolved.anchor).resolve(), home, repository}
    if (
        resolved in forbidden
        or repository.is_relative_to(resolved)
        or home.is_relative_to(resolved)
        or resolved == data_root
        or not resolved.is_relative_to(data_root)
    ):
        raise ValueError(f"Refusing unsafe output directory: {resolved}")
    return resolved


def validate_managed_output(output_root: Path) -> None:
    marker = output_root / OUTPUT_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise ValueError(
            f"Refusing to replace unmanaged directory: {output_root}; "
            f"expected {OUTPUT_MARKER}"
        )
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid materialization marker: {marker}") from error
    if payload != {"owner": OUTPUT_OWNER, "schema_version": 1}:
        raise ValueError(f"Unrecognized materialization marker: {marker}")


def write_output_marker(output_root: Path) -> None:
    (output_root / OUTPUT_MARKER).write_text(
        json.dumps({"owner": OUTPUT_OWNER, "schema_version": 1}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def safe_sample_dir(output_root: Path, sample_id: str) -> Path:
    if SAMPLE_ID_PATTERN.fullmatch(sample_id) is None:
        raise ValueError(f"Invalid VOC sample_id: {sample_id!r}")
    candidate = (output_root / sample_id).resolve()
    if candidate.parent != output_root:
        raise ValueError(f"Sample path escapes output directory: {sample_id!r}")
    return candidate


def write_source_image(cell: dict, destination: Path) -> None:
    """Preserve the exact source image bytes instead of re-encoding JPEG pixels."""

    raw = cell.get("bytes")
    if raw is not None:
        destination.write_bytes(raw)
        return
    source_path = cell.get("path")
    if source_path is not None:
        shutil.copyfile(source_path, destination)
        return
    raise ValueError("Unsupported Hugging Face image cell")


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
    output_root = safe_output_root(args.output_dir)
    if output_root.exists():
        if not args.replace:
            raise SystemExit(f"Output already exists: {output_root}; pass --replace to rebuild")
        validate_managed_output(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    write_output_marker(output_root)

    table = read_voc_table(args.parquet, columns=["image", "mask"])
    for index, manifest_row in enumerate(rows, start=1):
        source_row = int(manifest_row["source_row"])
        source = table.slice(source_row, 1).to_pylist()[0]
        semantic_mask = decode_voc_mask(decode_image(source["mask"]))
        selected = largest_component(semantic_mask)
        if selected is None:
            raise RuntimeError(f"No target for {manifest_row['sample_id']}")
        label, component = selected
        bbox, point = verify_target(manifest_row, label, component)

        sample_dir = safe_sample_dir(output_root, manifest_row["sample_id"])
        sample_dir.mkdir()
        write_source_image(source["image"], sample_dir / "image.jpg")
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
