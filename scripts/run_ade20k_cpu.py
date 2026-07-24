"""Run CPU baselines on the ADE20K validation split.

Three operating modes are supported:

    synthetic   No network — uses fake source rows for smoke testing.
    pilot       Bounded network run (e.g. 20–200 eligible samples).
    full        Complete validation-stream scan and evaluation.

Usage::

    python scripts/run_ade20k_cpu.py --mode synthetic
    python scripts/run_ade20k_cpu.py --mode pilot --max-eligible 50
    python scripts/run_ade20k_cpu.py --mode full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.ade import (
    ADE20KSourceRow,
    _clean_point_from_mask,
    _tight_bbox,
    eligible_rows,
    enumerate_ade20k_stream,
    load_manifest,
    materialize_samples,
    write_manifest,
)
from promptseg.algorithms import CONFIRMATORY_CPU_METHODS
from promptseg.dataset import Prompt
from promptseg.metrics import dice, iou
from promptseg.utils import write_csv

# ---------------------------------------------------------------------------
# Frozen six-method CPU set from the research protocol (excludes adaptive)
# ---------------------------------------------------------------------------
FROZEN_CPU_METHOD_NAMES = [
    "center_color",
    "grabcut_point_box",
    "robust_superpixel",
    "robust_no_color_seed",
    "robust_no_spatial_prior",
    "robust_single_box",
]

FROZEN_CPU_METHODS = {k: CONFIRMATORY_CPU_METHODS[k] for k in FROZEN_CPU_METHOD_NAMES}

SECONDARY_METHOD_NAMES = ["adaptive_superpixel"]

# ---------------------------------------------------------------------------
# Synthetic smoke helpers
# ---------------------------------------------------------------------------


def _synthetic_source_rows(num_rows: int = 100, seed: int = 42) -> list[ADE20KSourceRow]:
    """Generate deterministic synthetic source rows for smoke testing."""
    rng = np.random.default_rng(seed)
    rows: list[ADE20KSourceRow] = []
    for i in range(num_rows):
        eligible = True
        exclusion_reason = ""
        # Make ~15% of rows ineligible for realistic coverage
        if i % 7 == 0:
            eligible = False
            exclusion_reason = "no_instances_or_objects_metadata"
        elif i % 13 == 3:
            eligible = False
            exclusion_reason = "mask_area_below_minimum"
        elif i % 17 == 5:
            eligible = False
            exclusion_reason = "no_valid_instance_masks"

        img_w = rng.integers(200, 800)
        img_h = rng.integers(200, 800)
        mask_area = int(rng.integers(300, 20000)) if eligible else int(rng.integers(1, 255))

        point = None
        bbox = None
        if eligible:
            px = rng.integers(10, img_w - 10)
            py = rng.integers(10, img_h - 10)
            bx0 = rng.integers(5, img_w // 2)
            by0 = rng.integers(5, img_h // 2)
            bx1 = rng.integers(bx0 + 10, img_w - 5)
            by1 = rng.integers(by0 + 10, img_h - 5)
            point = (int(px), int(py))
            bbox = (int(bx0), int(by0), int(bx1), int(by1))

        rows.append(
            ADE20KSourceRow(
                row_index=i,
                sample_id=f"synth_ade_{i:06d}",
                filename=f"synth_ade_{i:06d}.jpg",
                eligible=eligible,
                exclusion_reason=exclusion_reason,
                image_width=img_w,
                image_height=img_h,
                num_instances=rng.integers(1, 8),
                selected_instance_idx=0 if eligible else None,
                selected_mask_area=mask_area,
                object_name=f"synth_obj_{i % 20}" if eligible else None,
                point=point,
                bbox=bbox,
            )
        )
    return rows


def _synthetic_sample(row: ADE20KSourceRow) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic image and mask for a source row (no network)."""
    stable_seed = int.from_bytes(
        hashlib.sha256(row.sample_id.encode("utf-8")).digest()[:8],
        byteorder="big",
    )
    rng = np.random.default_rng(stable_seed)
    h, w = row.image_height, row.image_width
    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    if row.bbox is not None:
        x0, y0, x1, y1 = row.bbox
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))
        mask[y0:y1, x0:x1] = True
    return image, mask


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _run_method_on_sample(
    method_name: str,
    method_fn: Any,
    sample_id: str,
    object_name: str,
    image: np.ndarray,
    mask: np.ndarray,
    point: tuple[int, int],
    bbox: tuple[int, int, int, int],
    mask_area: int,
    row_index: int,
) -> dict[str, Any]:
    """Run one CPU method on one sample; failures score as zero."""
    prompt = Prompt(
        point=point,
        bbox=bbox,
        label=0,
        class_name=object_name or "object",
    )
    t0 = time.perf_counter()
    try:
        prediction = method_fn(image, prompt)
        success = True
        sample_iou_val = iou(prediction, mask)
        sample_dice_val = dice(prediction, mask)
    except Exception:
        success = False
        sample_iou_val = 0.0
        sample_dice_val = 0.0
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "sample_id": sample_id,
        "source_row_index": row_index,
        "method": method_name,
        "object_name": object_name,
        "mask_area_px": mask_area,
        "iou": sample_iou_val,
        "dice": sample_dice_val,
        "success": success,
        "latency_ms": round(latency_ms, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADE20K CPU benchmark (secondary cross-dataset experiment)",
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "pilot", "full"],
        default="pilot",
        help="Operating mode: synthetic (no network), pilot (bounded), full (complete stream)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap on source rows scanned (default: unlimited in full mode, 200 in pilot)",
    )
    parser.add_argument(
        "--max-eligible",
        type=int,
        default=None,
        help="Cap on eligible samples evaluated (default: unlimited)",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=256,
        help="Minimum object mask area in pixels (default: 256)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/secondary/ade20k",
        help="Output directory for data-free artifacts",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/ade20k"),
        help="Git-ignored image/mask cache used for resume-safe materialization",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(CONFIRMATORY_CPU_METHODS),
        default=None,
        help="Override default six-method set (adaptive must be requested explicitly)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260720,
        help="RNG seed used only for deterministic bounded subsampling",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not attempt to resume from existing checkpoints",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve method set
    if args.methods:
        method_names = list(args.methods)
        has_adaptive = "adaptive_superpixel" in method_names
        if not has_adaptive:
            method_names = [m for m in method_names if m not in SECONDARY_METHOD_NAMES]
    else:
        method_names = list(FROZEN_CPU_METHOD_NAMES)
        has_adaptive = False

    if has_adaptive:
        print("NOTE: adaptive_superpixel is a secondary method and is excluded "
              "from frozen-method summaries.", file=sys.stderr)

    methods = {k: CONFIRMATORY_CPU_METHODS[k] for k in method_names}

    # ------------------------------------------------------------------
    # Phase 1 — produce or load the source-row manifest
    # ------------------------------------------------------------------
    manifest_path = out_dir / "source_manifest.jsonl"
    manifest_sha256_path = out_dir / "source_manifest.sha256"

    if args.mode == "synthetic":
        print("=== SYNTHETIC MODE (no network) ===")
        max_rows = args.max_rows or 100
        rows = _synthetic_source_rows(num_rows=max_rows)
        sha = write_manifest(rows, manifest_path)
        manifest_sha256_path.write_text(sha + "\n", encoding="utf-8")
        print(f"  Synthetic manifest: {len(rows)} rows → {manifest_path}")

    elif manifest_path.exists() and not args.no_resume:
        print("=== RESUMING from existing manifest ===")
        rows = load_manifest(manifest_path)
        sha = manifest_sha256_path.read_text(encoding="utf-8").strip() if manifest_sha256_path.exists() else ""
        print(f"  Loaded {len(rows)} rows from {manifest_path}")

    else:
        print("=== STREAMING ADE20K validation split ===")
        max_rows = args.max_rows  # None = scan everything
        if args.mode == "pilot" and max_rows is None:
            max_rows = 200
        t_stream = time.perf_counter()
        rows = list(enumerate_ade20k_stream(min_mask_area=args.min_mask_area, max_rows=max_rows))
        stream_sec = time.perf_counter() - t_stream
        sha = write_manifest(rows, manifest_path)
        manifest_sha256_path.write_text(sha + "\n", encoding="utf-8")
        print(f"  Scanned {len(rows)} source rows in {stream_sec:.1f} s")
        print(f"  Manifest SHA-256: {sha}")

    all_eligible = eligible_rows(rows)
    rows_scanned = len(rows)
    eligible_in_stream = len(all_eligible)
    excluded_count = rows_scanned - eligible_in_stream

    print(f"  Rows scanned:   {rows_scanned}")
    print(f"  Eligible:       {eligible_in_stream}")
    print(f"  Excluded:       {excluded_count}")

    # Exclusion reason breakdown
    from collections import Counter
    reasons = Counter(r.exclusion_reason for r in rows if not r.eligible)
    if reasons:
        print("  Exclusion breakdown:")
        for reason, count in reasons.most_common():
            print(f"    {reason}: {count}")

    # Apply max-eligible bound
    selected = all_eligible
    if args.max_eligible is not None and args.max_eligible < eligible_in_stream:
        from promptseg.ade import _deterministic_sample
        selected = _deterministic_sample(all_eligible, args.max_eligible, args.seed)
        print(f"  Selected {len(selected)} of {eligible_in_stream} eligible rows (seed={args.seed})")

    if not selected:
        print("ERROR: No eligible samples to evaluate.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2 — evaluate (with checkpoint/resume)
    # ------------------------------------------------------------------
    checkpoint_dir = out_dir / "sample_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Run fingerprint
    run_config = {
        "schema_version": 1,
        "mode": args.mode,
        "min_mask_area": args.min_mask_area,
        "manifest_sha256": sha,
        "method_names": method_names,
        "seed": args.seed,
        "selected_sample_ids": [row.sample_id for row in selected],
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    config_path = out_dir / "run_config.json"
    existing_checkpoints = sorted(checkpoint_dir.glob("*.json"))
    if config_path.exists() and not args.no_resume:
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        existing_fp = hashlib.sha256(
            json.dumps({k: existing_config.get(k) for k in run_config}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if existing_fp != run_fingerprint:
            raise SystemExit(
                "Output directory belongs to a different run configuration. "
                "Use --output-dir to specify a fresh directory."
            )
    elif existing_checkpoints and not args.no_resume:
        raise SystemExit(
            "Checkpoint files exist without a matching run_config.json. "
            "Use --output-dir to specify a fresh directory."
        )
    else:
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    completed_before = 0
    t_eval = time.perf_counter()
    pending_rows: list[ADE20KSourceRow] = []
    for row in selected:
        cp_path = checkpoint_dir / f"{row.sample_id}.json"
        if cp_path.exists() and not args.no_resume:
            cp = json.loads(cp_path.read_text(encoding="utf-8"))
            if cp.get("run_fingerprint") != run_fingerprint:
                raise RuntimeError(
                    f"Checkpoint fingerprint mismatch for {row.sample_id}; "
                    f"use --no-resume or a fresh --output-dir"
                )
            cp_rows = cp.get("rows", [])
            if len(cp_rows) != len(method_names):
                raise RuntimeError(f"Incomplete checkpoint for {row.sample_id}")
            all_rows.extend(cp_rows)
            completed_before += 1
        else:
            pending_rows.append(row)

    real_samples = (
        iter(materialize_samples(pending_rows, cache_dir=args.cache_dir))
        if args.mode != "synthetic"
        else None
    )
    for i, row in enumerate(pending_rows):
        cp_path = checkpoint_dir / f"{row.sample_id}.json"
        if args.mode == "synthetic":
            image, mask = _synthetic_sample(row)
        else:
            sample = next(real_samples)
            if sample.sample_id != row.sample_id:
                raise RuntimeError(
                    f"Materialized sample order mismatch: {sample.sample_id} != {row.sample_id}"
                )
            image, mask = sample.image, sample.mask

        sample_rows = []
        for mname, mfn in methods.items():
            result = _run_method_on_sample(
                mname, mfn,
                row.sample_id,
                row.object_name or "",
                image, mask,
                row.point or (0, 0),
                row.bbox or (0, 0, 1, 1),
                row.selected_mask_area or 0,
                row.row_index,
            )
            sample_rows.append(result)

        all_rows.extend(sample_rows)

        # Atomic checkpoint write
        cp_payload = {
            "run_fingerprint": run_fingerprint,
            "sample_id": row.sample_id,
            "rows": sample_rows,
        }
        tmp_path = cp_path.with_name(cp_path.name + ".tmp")
        tmp_path.write_text(json.dumps(cp_payload) + "\n", encoding="utf-8")
        tmp_path.replace(cp_path)

        done = completed_before + i + 1
        if done % 25 == 0 or done == len(selected):
            print(f"  Evaluated {done}/{len(selected)} samples …", flush=True)

    eval_sec = time.perf_counter() - t_eval
    print(f"  Evaluation completed in {eval_sec:.1f} s")

    # ------------------------------------------------------------------
    # Phase 3 — aggregate and write data-free outputs
    # ------------------------------------------------------------------
    metrics_path = out_dir / "metrics.csv"
    write_csv(metrics_path, all_rows)
    metrics_sha256 = hashlib.sha256(metrics_path.read_bytes()).hexdigest()

    # Per-method summary across ALL attempted rows (failures = zero)
    summaries = []
    for mname in method_names:
        mrows = [r for r in all_rows if r["method"] == mname]
        n = len(mrows)
        n_fail = sum(1 for r in mrows if not r["success"])
        ious = np.array([r["iou"] for r in mrows], dtype=np.float64)
        dices = np.array([r["dice"] for r in mrows], dtype=np.float64)
        lats = np.array([r["latency_ms"] for r in mrows], dtype=np.float64)

        summaries.append({
            "method": mname,
            "num_samples": n,
            "num_failures": n_fail,
            "mean_iou": float(np.mean(ious)),
            "median_iou": float(np.median(ious)),
            "mean_dice": float(np.mean(dices)),
            "median_dice": float(np.median(dices)),
            "median_latency_ms": float(np.median(lats)),
            "p95_latency_ms": float(np.percentile(lats, 95)),
            "total_seconds": float(lats.sum() / 1000),
        })

    # Build aggregate summary
    aggregate_summary = {
        "status": "secondary",
        "experiment": "ade20k_cross_dataset_cpu",
        "mode": args.mode,
        "rows_scanned": rows_scanned,
        "eligible_rows_in_stream": eligible_in_stream,
        "selected_samples": len(selected),
        "evaluated_samples": len({row["sample_id"] for row in all_rows}),
        "excluded_count": excluded_count,
        "exclusion_reasons": dict(reasons),
        "min_mask_area": args.min_mask_area,
        "methods": method_names,
        "has_adaptive": has_adaptive,
        "frozen_method_summary_applies": not has_adaptive or len(method_names) == len(FROZEN_CPU_METHOD_NAMES),
        "stream_manifest_sha256": sha,
        "metrics_csv_sha256": metrics_sha256,
        "run_fingerprint": run_fingerprint,
        "completed_before_resume": completed_before,
        "eval_seconds": round(eval_sec, 1),
        "run_config": run_config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "summaries": summaries,
    }

    # Only claim "complete" if full mode + no max-rows + no max-eligible
    is_complete = (
        args.mode == "full"
        and args.max_rows is None
        and args.max_eligible is None
        and len({row["sample_id"] for row in all_rows}) == eligible_in_stream
    )
    aggregate_summary["stream_complete"] = is_complete

    summary_path = out_dir / "aggregate_summary.json"
    summary_path.write_text(json.dumps(aggregate_summary, indent=2) + "\n", encoding="utf-8", newline="\n")

    # Write aggregate checksums
    checksums_path = out_dir / "checksums.json"
    checksums = {
        "source_manifest_sha256": sha,
        "metrics_csv_sha256": metrics_sha256,
        "aggregate_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "command": " ".join(sys.argv),
    }
    checksums_path.write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8", newline="\n")

    # Write README
    readme_path = out_dir / "README.md"
    readme_path.write_text(
        f"# ADE20K Secondary Cross-Dataset Experiment\n\n"
        f"**Status**: secondary (not a confirmatory H1–H3 result)\n\n"
        f"- Rows scanned: {rows_scanned}\n"
        f"- Eligible rows in scanned stream: {eligible_in_stream}\n"
        f"- Selected/evaluated samples: {len(selected)}\n"
        f"- Excluded: {excluded_count}\n"
        f"- Methods: {', '.join(method_names)}\n"
        f"- Mode: {args.mode}\n"
        f"- Stream complete: {is_complete}\n\n"
        f"## Regenerate\n\n"
        f"```bash\n"
        f"python scripts/run_ade20k_cpu.py --mode {args.mode}"
        + (f" --max-rows {args.max_rows}" if args.max_rows else "")
        + (f" --max-eligible {args.max_eligible}" if args.max_eligible else "")
        + "\n```\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Phase 4 — Report
    # ------------------------------------------------------------------
    print()
    print(f"=== Results ({'COMPLETE' if is_complete else 'PARTIAL'}) ===")
    print(f"  Rows scanned:    {rows_scanned}")
    print(f"  Eligible:        {eligible_in_stream}")
    print(f"  Selected:        {len(selected)}")
    print(f"  Evaluated:       {len({row['sample_id'] for row in all_rows})}")
    print(f"  Output:          {out_dir}")
    print()
    header = f"{'Method':<28s} {'Mean IoU':>8s}  {'Mean Dice':>8s}  {'Med Lat':>8s}  {'Fails':>5s}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['method']:<28s} {s['mean_iou']:8.4f}  {s['mean_dice']:8.4f}  "
            f"{s['median_latency_ms']:7.1f}ms  {s['num_failures']:5d}"
        )

    # Final status message
    if is_complete:
        print(f"\nFull-stream evaluation complete: {rows_scanned} rows scanned, "
              f"{eligible_in_stream} eligible, {len(selected)} evaluated.")
    else:
        print(f"\nPARTIAL run: {rows_scanned} rows scanned, {eligible_in_stream} eligible, "
              f"{len(selected)} selected. Rerun with --mode full for complete stream.")


if __name__ == "__main__":
    main()
