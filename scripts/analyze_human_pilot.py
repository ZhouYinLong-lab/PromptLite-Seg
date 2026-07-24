"""Analyze human pilot annotations with participant-clustered statistics.

Reads validated annotation CSVs, computes per-participant and aggregate
metrics, and compares with synthetic perturbation baselines.

CRITICAL: This script will REFUSE to produce a human-result summary from
synthetic annotations. Synthetic rows must contain ``is_synthetic: true``.
Real human results require independently collected annotations with
proper consent.

Usage::

    python scripts/analyze_human_pilot.py
    python scripts/analyze_human_pilot.py --annotations data/human_annotations/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.dataset import load_sample
from promptseg.metrics import iou as compute_iou
from promptseg.prompts import point_hits_target, bbox_iou
from promptseg.utils import write_csv


def _load_annotations(annotations_dir: Path) -> tuple[list[dict], bool]:
    """Load all annotation CSVs from a directory.

    Returns (all_rows, is_synthetic).
    """
    all_rows: list[dict] = []
    is_synthetic = False

    for csv_path in sorted(annotations_dir.glob("annotations_*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check synthetic marker
                syn = row.get("is_synthetic", "").strip().lower()
                if syn in ("true", "1"):
                    is_synthetic = True
                all_rows.append(row)

    return all_rows, is_synthetic


def _compute_point_metrics(
    rows: list[dict], data_dir: Path,
) -> dict[str, Any]:
    """Compute point task metrics: hit rate per participant and overall."""
    participant_hits: dict[str, list[bool]] = defaultdict(list)

    for row in rows:
        if row.get("task_type") != "point":
            continue
        if row.get("timeout", "").strip().lower() in ("true", "1"):
            continue

        pc = row.get("participant_code", "unknown")
        img_id = row.get("image_id", "")

        try:
            px = int(float(row.get("point_x", -1)))
            py = int(float(row.get("point_y", -1)))
        except (ValueError, TypeError):
            continue

        # Load mask
        mask_path = data_dir / img_id / "target_mask.png"
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 0

        hit = point_hits_target((px, py), mask)
        participant_hits[pc].append(hit)

    per_participant = {}
    for pc, hits in sorted(participant_hits.items()):
        per_participant[pc] = {
            "n_tasks": len(hits),
            "hit_rate": float(np.mean(hits)),
        }

    all_hits = [h for hits in participant_hits.values() for h in hits]
    return {
        "per_participant": per_participant,
        "aggregate_hit_rate": float(np.mean(all_hits)) if all_hits else None,
        "total_point_tasks": len(all_hits),
    }


def _compute_box_metrics(
    rows: list[dict], data_dir: Path,
) -> dict[str, Any]:
    """Compute box task metrics: IoU per participant and overall."""
    participant_ious: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        if row.get("task_type") != "box":
            continue
        if row.get("timeout", "").strip().lower() in ("true", "1"):
            continue

        pc = row.get("participant_code", "unknown")
        img_id = row.get("image_id", "")

        try:
            bx0 = int(float(row.get("box_x0", -1)))
            by0 = int(float(row.get("box_y0", -1)))
            bx1 = int(float(row.get("box_x1", -1)))
            by1 = int(float(row.get("box_y1", -1)))
        except (ValueError, TypeError):
            continue

        # Get ground-truth bbox from the sample
        try:
            sample = load_sample(data_dir / img_id)
            gt_bbox = sample.prompt.bbox
        except Exception:
            continue

        bi = bbox_iou((bx0, by0, bx1, by1), gt_bbox)
        participant_ious[pc].append(bi)

    per_participant = {}
    for pc, ious in sorted(participant_ious.items()):
        arr = np.array(ious)
        per_participant[pc] = {
            "n_tasks": len(ious),
            "mean_iou": float(np.mean(arr)),
            "median_iou": float(np.median(arr)),
            "q25_iou": float(np.percentile(arr, 25)),
            "q75_iou": float(np.percentile(arr, 75)),
        }

    all_ious = [v for ious in participant_ious.values() for v in ious]
    arr = np.array(all_ious) if all_ious else np.array([])
    return {
        "per_participant": per_participant,
        "aggregate_mean_iou": float(np.mean(arr)) if len(arr) > 0 else None,
        "aggregate_median_iou": float(np.median(arr)) if len(arr) > 0 else None,
        "total_box_tasks": len(all_ious),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze human pilot annotations"
    )
    parser.add_argument(
        "--annotations", type=Path,
        default=Path("data/human_annotations"),
        help="Directory containing annotation CSVs",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/voc_validation"),
        help="Directory containing sample subdirectories",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/secondary/human_pilot"),
    )
    args = parser.parse_args()

    if not args.annotations.exists():
        raise SystemExit(
            f"Annotations directory not found: {args.annotations}\n"
            f"No human annotations exist yet. Run run_human_collection.py first."
        )

    rows, is_synthetic = _load_annotations(args.annotations)

    if not rows:
        raise SystemExit("No annotation rows found.")

    # Check: refuse to present synthetic data as human results
    if is_synthetic:
        print("=" * 72)
        print("WARNING: Annotations contain is_synthetic markers.")
        print("These are DEMO / TEST data ONLY.")
        print("THIS ANALYSIS DOES NOT CONSTITUTE A HUMAN-RESULT SUMMARY.")
        print("Real participant data is required for any human-pilot claim.")
        print("=" * 72)
        print()

    # Per-participant statistics
    participants = sorted(set(r.get("participant_code", "unknown") for r in rows))
    print(f"Participants: {participants}")
    print(f"Total annotations: {len(rows)}")

    # Compute metrics
    point_metrics = _compute_point_metrics(rows, args.data_dir)
    box_metrics = _compute_box_metrics(rows, args.data_dir)

    # Comparison with synthetic perturbation baselines (if available)
    sensitivity_path = Path("artifacts/secondary/prompt_quality_sensitivity/sensitivity_analysis.json")
    comparison = None
    if sensitivity_path.exists():
        try:
            sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
            # Extract mean point and box IoU at the highest quality target
            curve = sensitivity.get("curve_points", [])
            if curve:
                comparison = {
                    "source": "synthetic_perturbation_curve",
                    "note": (
                        "Synthetic perturbations at calibrated quality targets. "
                        "Human and synthetic prompt qualities are NOT directly "
                        "comparable — different distributions, different tasks."
                    ),
                    "curve_summary": [
                        {
                            "quality_target": cp["quality_target"],
                            "mean_point_iou": cp["mean_point_iou"],
                            "mean_box_iou": cp["mean_box_iou"],
                        }
                        for cp in curve
                    ],
                }
        except Exception:
            pass

    # Build analysis report
    analysis = {
        "status": "secondary_pilot",
        "is_synthetic": is_synthetic,
        "contains_human_results": not is_synthetic,
        "disclaimer": (
            "This is an EXPLORATORY PILOT analysis. Results are observational "
            "only and do NOT constitute confirmatory evidence. No primary "
            "hypothesis tests or p-values are reported. All metrics are "
            "participant-clustered where applicable."
        ),
        "participants": len(participants),
        "participant_codes": participants,
        "total_annotations": len(rows),
        "point_metrics": point_metrics,
        "box_metrics": box_metrics,
        "synthetic_perturbation_comparison": comparison,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_dir / "human_pilot_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8", newline="\n")

    # Write participant-clustered CSV
    participant_rows = []
    for pc in participants:
        pm = point_metrics["per_participant"].get(pc, {})
        bm = box_metrics["per_participant"].get(pc, {})

        participant_rows.append({
            "participant_code": pc,
            "point_tasks": pm.get("n_tasks", 0),
            "point_hit_rate": pm.get("hit_rate"),
            "box_tasks": bm.get("n_tasks", 0),
            "box_mean_iou": bm.get("mean_iou"),
            "box_median_iou": bm.get("median_iou"),
            "is_synthetic": is_synthetic,
        })
    write_csv(args.output_dir / "participant_summary.csv", participant_rows)

    print(f"\nAnalysis written to {analysis_path}")
    print(json.dumps({
        "participants": len(participants),
        "point_hit_rate": point_metrics.get("aggregate_hit_rate"),
        "box_mean_iou": box_metrics.get("aggregate_mean_iou"),
        "is_synthetic": is_synthetic,
    }, indent=2))


if __name__ == "__main__":
    main()
