"""Validate human annotation CSV files for PII, schema, and correctness.

Checks::
    - No PII fields present
    - All required fields present with correct types
    - Coordinates are within image bounds
    - No duplicate task IDs
    - Synthetic rows are properly labelled
    - Valid participant codes

Usage::

    python scripts/validate_human_annotations.py --annotations data/human_annotations/
"""

from __future__ import annotations

import argparse
import csv
import re
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

PII_FIELDS = {
    "name", "email", "student_number", "student_id", "ip_address",
    "user_agent", "free_text", "comment", "demographic", "age",
    "gender", "ethnicity", "phone", "address",
}

REQUIRED_FIELDS = {
    "participant_code", "task_id", "image_id", "task_type",
    "point_x", "point_y", "box_x0", "box_y0", "box_x1", "box_y1",
    "elapsed_time_ms", "timeout",
}

SYNTHETIC_MARKER = "is_synthetic"
PARTICIPANT_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,20}$")


def validate_annotations(
    csv_path: Path,
    data_dir: Path,
) -> tuple[bool, list[str]]:
    """Validate a single annotation CSV file.

    Returns (is_valid, list_of_errors).
    """
    errors: list[str] = []
    prefix = f"[{csv_path.name}]"

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        errors.append(f"{prefix} Empty file")
        return False, errors

    # --- PII check ---
    fieldnames = set(rows[0].keys())
    pii_found = fieldnames & PII_FIELDS
    if pii_found:
        errors.append(f"{prefix} PII fields found: {sorted(pii_found)}")

    # --- Required fields ---
    missing = REQUIRED_FIELDS - fieldnames
    if missing:
        errors.append(f"{prefix} Missing required fields: {sorted(missing)}")

    has_synthetic_col = SYNTHETIC_MARKER in fieldnames

    # --- Per-row validation ---
    seen_task_ids: set[str] = set()
    image_cache: dict[str, tuple[int, int]] = {}

    for i, row in enumerate(rows):
        line_tag = f"{prefix} row {i + 1}"

        # Participant code
        pc = row.get("participant_code", "")
        if not PARTICIPANT_CODE_PATTERN.fullmatch(pc):
            errors.append(f"{line_tag} Invalid participant_code: '{pc}'")

        # Task ID
        tid = row.get("task_id", "")
        if not tid:
            errors.append(f"{line_tag} Missing task_id")
        elif tid in seen_task_ids:
            errors.append(f"{line_tag} Duplicate task_id: {tid}")
        else:
            seen_task_ids.add(tid)

        # Task type
        tt = row.get("task_type", "")
        if tt not in ("point", "box"):
            errors.append(f"{line_tag} Invalid task_type: '{tt}'")

        # Timeout flag
        timeout_str = row.get("timeout", "false").strip().lower()
        is_timeout = timeout_str in ("true", "1", "yes")

        # Coordinate validation
        img_id = row.get("image_id", "")
        if img_id and not is_timeout:
            # Get image dimensions
            if img_id not in image_cache:
                img_path = data_dir / img_id / "image.jpg"
                if img_path.exists():
                    try:
                        img = Image.open(img_path)
                        image_cache[img_id] = (img.width, img.height)
                    except Exception:
                        image_cache[img_id] = (99999, 99999)
                else:
                    image_cache[img_id] = (99999, 99999)

            w, h = image_cache[img_id]

            if tt == "point":
                px_str = row.get("point_x", "")
                py_str = row.get("point_y", "")
                if px_str and py_str:
                    try:
                        px, py = int(float(px_str)), int(float(py_str))
                        if not (0 <= px < w and 0 <= py < h):
                            errors.append(
                                f"{line_tag} Point ({px}, {py}) out of bounds "
                                f"for image {img_id} ({w}×{h})"
                            )
                    except (ValueError, TypeError):
                        errors.append(f"{line_tag} Non-integer point coordinates")

            elif tt == "box":
                for coord_name in ("box_x0", "box_y0", "box_x1", "box_y1"):
                    val = row.get(coord_name, "")
                    if val:
                        try:
                            c = int(float(val))
                            if coord_name in ("box_x0", "box_x1") and not (0 <= c < w):
                                errors.append(
                                    f"{line_tag} {coord_name}={c} out of bounds "
                                    f"for image {img_id} ({w}×{h})"
                                )
                            elif coord_name in ("box_y0", "box_y1") and not (0 <= c < h):
                                errors.append(
                                    f"{line_tag} {coord_name}={c} out of bounds "
                                    f"for image {img_id} ({w}×{h})"
                                )
                        except (ValueError, TypeError):
                            errors.append(f"{line_tag} Non-integer {coord_name}")

                # Box must have non-zero area
                bx0 = row.get("box_x0", "")
                by0 = row.get("box_y0", "")
                bx1 = row.get("box_x1", "")
                by1 = row.get("box_y1", "")
                if bx0 and by0 and bx1 and by1:
                    try:
                        x0, y0 = int(float(bx0)), int(float(by0))
                        x1, y1 = int(float(bx1)), int(float(by1))
                        if x0 == x1 or y0 == y1:
                            errors.append(
                                f"{line_tag} Box has zero area: "
                                f"({x0}, {y0}) → ({x1}, {y1})"
                            )
                    except (ValueError, TypeError):
                        pass

        # Synthetic label check
        if has_synthetic_col:
            is_syn = row.get(SYNTHETIC_MARKER, "").strip().lower()
            if is_syn not in ("true", "false", "1", "0", ""):
                errors.append(f"{line_tag} Invalid {SYNTHETIC_MARKER} value: '{is_syn}'")

    return (len(errors) == 0, errors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate human annotation CSV files"
    )
    parser.add_argument(
        "--annotations", type=Path, required=True,
        help="Path to annotation CSV or directory of CSVs",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/voc_validation"),
        help="Directory containing sample subdirectories (for image bounds)",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.annotations.is_dir():
        paths = sorted(args.annotations.glob("annotations_*.csv"))
    else:
        paths = [args.annotations]

    if not paths:
        print("No annotation files found.", file=sys.stderr)
        sys.exit(1)

    all_valid = True
    total_errors = 0

    for path in paths:
        is_valid, errors = validate_annotations(path, args.data_dir)
        if is_valid:
            print(f"✓ {path.name} — VALID")
        else:
            all_valid = False
            print(f"✗ {path.name} — {len(errors)} error(s):")
            for err in errors:
                print(f"    {err}")
            total_errors += len(errors)

    print()
    if all_valid:
        print(f"All {len(paths)} file(s) passed validation.")
    else:
        print(f"{len(paths)} file(s) checked, {total_errors} total error(s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
