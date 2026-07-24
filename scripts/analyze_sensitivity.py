"""Analyze multi-quality point-versus-box sensitivity curve.

Aggregates trials per sample, computes per-target paired deltas with
95% paired-bootstrap confidence intervals, and produces a data-free
sensitivity curve summary.

IMPORTANT: Numeric observable matching (point hit rate vs box IoU) is a
calibration convenience and does NOT imply human-perceptual equivalence.
This is a robustness/sensitivity analysis, not a primary hypothesis test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.utils import write_csv

N_BOOTSTRAP = 10_000
SEED_CI = 20260724


def _sample_level_aggregate(
    rows: list[dict],
    target: str,
    condition: str,
    sample_id: str | None = None,
) -> dict[str, float]:
    """Aggregate IoU across trials for a single sample, target, and condition."""
    matching = [
        r for r in rows
        if r.get("quality_target") == target and r.get("condition") == condition
        and (sample_id is None or r.get("sample_id") == sample_id)
    ]
    if not matching:
        return {}

    iou_vals = []
    failures = 0
    for r in matching:
        val = r.get("iou", "")
        if val == "" or val is None:
            iou_vals.append(0.0)
            failures += 1
        else:
            try:
                iou_vals.append(float(val))
            except (ValueError, TypeError):
                iou_vals.append(0.0)
                failures += 1

    if not iou_vals:
        return {}

    return {
        "sample_id": matching[0].get("sample_id", ""),
        "mean_iou": float(np.mean(iou_vals)),
        "num_trials": len(iou_vals),
        "num_failures": failures,
    }


def _paired_bootstrap_ci(
    deltas: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED_CI,
) -> tuple[float, float, float]:
    """95% paired-bootstrap confidence interval for sample-level deltas."""
    if len(deltas) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.zeros(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[b] = np.mean(deltas[idx])
    lower = float(np.percentile(means, 2.5))
    upper = float(np.percentile(means, 97.5))
    mean_delta = float(np.mean(deltas))
    return (mean_delta, lower, upper)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze multi-quality point-vs-box sensitivity curve"
    )
    parser.add_argument(
        "--metrics", type=Path,
        default=Path("artifacts/secondary/prompt_quality_sensitivity/metrics.csv"),
        help="Path to sensitivity metrics CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/secondary/prompt_quality_sensitivity"),
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("protocol/sensitivity_protocol.json"),
    )
    args = parser.parse_args()

    if not args.metrics.exists():
        raise SystemExit(f"Metrics file not found: {args.metrics}")

    # Read metrics
    import csv
    all_rows: list[dict] = []
    with args.metrics.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if not all_rows:
        raise SystemExit("Metrics CSV is empty")

    # Determine quality targets from data
    targets_raw = sorted(
        set(r.get("quality_target", "") for r in all_rows),
        key=lambda value: float(value or 0),
        reverse=True,
    )
    targets = [t for t in targets_raw if t]

    # Check if data is synthetic
    is_synthetic = any(
        r.get("synthetic", "").lower() in ("true", "1", "yes") for r in all_rows
    )

    # Get unique sample IDs
    sample_ids = sorted(set(r.get("sample_id", "") for r in all_rows))
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in all_rows:
        key = (
            row.get("sample_id", ""),
            row.get("quality_target", ""),
            row.get("condition", ""),
        )
        grouped.setdefault(key, []).append(row)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected_trials = int(protocol.get("validation", {}).get("trials_per_condition", 1))
    incomplete_pairs = 0

    print(f"Analyzing sensitivity curve")
    print(f"  Samples:       {len(sample_ids)}")
    print(f"  Total rows:    {len(all_rows)}")
    print(f"  Quality levels: {targets}")
    print(f"  Synthetic:     {is_synthetic}")
    print()

    # Per-target analysis
    curve_points: list[dict[str, Any]] = []

    for target in targets:
        point_ious: list[float] = []
        box_ious: list[float] = []
        point_qualities: list[float] = []
        box_qualities: list[float] = []
        sample_count = 0
        failure_count = 0

        for sid in sample_ids:
            pt_rows = grouped.get((sid, target, "point_noise"), [])
            bx_rows = grouped.get((sid, target, "box_noise"), [])
            point_agg = _sample_level_aggregate(
                pt_rows, target, "point_noise", sample_id=sid
            )
            box_agg = _sample_level_aggregate(
                bx_rows, target, "box_noise", sample_id=sid
            )

            if not point_agg or not box_agg:
                failure_count += 1
                incomplete_pairs += 1
                continue
            if (
                point_agg["num_trials"] != expected_trials
                or box_agg["num_trials"] != expected_trials
            ):
                incomplete_pairs += 1

            # Get prompt quality values for this sample
            for row in pt_rows:
                ph = row.get("point_hit", "").lower()
                if ph in ("true", "false"):
                    point_qualities.append(1.0 if ph == "true" else 0.0)

            for row in bx_rows:
                bi_str = row.get("box_iou", "")
                try:
                    box_qualities.append(float(bi_str))
                except (ValueError, TypeError):
                    pass

            point_ious.append(point_agg["mean_iou"])
            box_ious.append(box_agg["mean_iou"])
            sample_count += 1

        if sample_count == 0:
            print(f"  Target {target}: no valid samples")
            continue

        point_iou_arr = np.array(point_ious)
        box_iou_arr = np.array(box_ious)
        deltas = point_iou_arr - box_iou_arr  # positive = box noise causes larger loss

        mean_delta, ci_lower, ci_upper = _paired_bootstrap_ci(deltas)

        mean_point_quality = float(np.mean(point_qualities)) if point_qualities else None
        mean_box_quality = float(np.mean(box_qualities)) if box_qualities else None

        curve_points.append({
            "quality_target": target,
            "sample_count": sample_count,
            "failure_count": failure_count,
            "mean_point_iou": float(np.mean(point_iou_arr)),
            "mean_box_iou": float(np.mean(box_iou_arr)),
            "mean_point_quality": mean_point_quality,
            "mean_box_quality": mean_box_quality,
            "paired_delta_mean": mean_delta,
            "paired_delta_ci_lower": ci_lower,
            "paired_delta_ci_upper": ci_upper,
            "interpretation": (
                "box_noise_IoU < point_noise_IoU" if mean_delta > 0
                else "box_noise_IoU > point_noise_IoU" if mean_delta < 0
                else "no_difference"
            ),
        })

        print(
            f"  Target {target}: "
            f"n={sample_count}, "
            f"Δ={mean_delta:.4f} [{ci_lower:.4f}, {ci_upper:.4f}], "
            f"Pt IoU={float(np.mean(point_iou_arr)):.4f}, "
            f"Bx IoU={float(np.mean(box_iou_arr)):.4f}"
        )

    # Write analysis summary
    analysis = {
        "status": "secondary",
        "analysis_type": "multi_quality_sensitivity_curve",
        "is_synthetic": is_synthetic,
        "num_samples_analyzed": len(sample_ids),
        "expected_trials_per_condition": expected_trials,
        "incomplete_sample_target_pairs": incomplete_pairs,
        "analysis_complete": incomplete_pairs == 0 and not is_synthetic,
        "quality_targets": targets,
        "disclaimer": (
            "Numeric observable matching (point hit rate vs box IoU) is a "
            "calibration convenience and does NOT imply human-perceptual "
            "equivalence. This is a robustness/sensitivity analysis, not a "
            "primary hypothesis test. No primary p-values are reported and "
            "H3 is not relabelled."
        ),
        "curve_points": curve_points,
        "bootstrap_config": {
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED_CI,
            "ci_level": 0.95,
        },
    }

    analysis_path = args.output_dir / "sensitivity_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8", newline="\n")

    # Write per-target CSV
    if curve_points:
        write_csv(args.output_dir / "sensitivity_curve.csv", curve_points)

    analysis_sha256 = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    print(f"\nAnalysis written to {analysis_path}")
    print(f"SHA-256: {analysis_sha256}")

    # Print final disclaimer
    print()
    print("=" * 72)
    print("DISCLAIMER: Numeric observable matching (point hit rate vs box IoU)")
    print("is a calibration convenience. It does NOT imply human-perceptual")
    print("equivalence between point and box perturbations.")
    print("=" * 72)


if __name__ == "__main__":
    main()
