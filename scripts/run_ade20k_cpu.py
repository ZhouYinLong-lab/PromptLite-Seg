"""Run CPU baselines on ADE20K validation subset.

Downloads ADE20K samples via Hugging Face datasets (streaming), runs all
CPU methods, and writes per-sample metrics and a summary JSON/CSV.

Usage:
    python scripts/run_ade20k_cpu.py                     # 200 samples
    python scripts/run_ade20k_cpu.py --max-samples 500   # 500 samples
    python scripts/run_ade20k_cpu.py --max-samples 100 --output-dir outputs_ade20k
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.ade import ADE20KSample, load_ade20k_samples
from promptseg.algorithms import CONFIRMATORY_CPU_METHODS
from promptseg.dataset import Prompt
from promptseg.metrics import dice, iou
from promptseg.utils import write_csv


def run_method(
    method_name: str,
    method_fn: Any,
    sample: ADE20KSample,
) -> dict[str, Any]:
    """Run one CPU method on one ADE20K sample."""
    prompt = Prompt(
        point=sample.point,
        bbox=sample.bbox,
        label=0,
        class_name=sample.object_name or "object",
    )
    t0 = time.perf_counter()
    try:
        prediction = method_fn(sample.image, prompt)
        success = True
        sample_iou = iou(prediction, sample.mask)
        sample_dice = dice(prediction, sample.mask)
    except Exception as exc:
        prediction = None
        success = False
        sample_iou = 0.0
        sample_dice = 0.0
        print(f"  [{method_name}] FAIL on {sample.sample_id}: {exc}", file=sys.stderr)
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "sample_id": sample.sample_id,
        "method": method_name,
        "object_name": sample.object_name,
        "mask_area_px": int(sample.mask.sum()),
        "iou": sample_iou,
        "dice": sample_dice,
        "success": success,
        "latency_ms": round(latency_ms, 2),
    }


def summarize(rows: list[dict], method_name: str) -> dict[str, Any]:
    """Compute aggregate statistics for one method."""
    method_rows = [r for r in rows if r["method"] == method_name]
    n = len(method_rows)
    n_fail = sum(1 for r in method_rows if not r["success"])
    ious = np.array([r["iou"] for r in method_rows], dtype=np.float64)
    dices = np.array([r["dice"] for r in method_rows], dtype=np.float64)
    lats = np.array([r["latency_ms"] for r in method_rows], dtype=np.float64)

    return {
        "method": method_name,
        "num_samples": n,
        "num_failures": n_fail,
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "mean_dice": float(np.mean(dices)),
        "median_dice": float(np.median(dices)),
        "median_latency_ms": float(np.median(lats)),
        "p95_latency_ms": float(np.percentile(lats, 95)),
        "total_seconds": float(lats.sum() / 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ADE20K CPU benchmark")
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Number of ADE20K validation samples (default: 200)",
    )
    parser.add_argument(
        "--min-mask-area", type=int, default=256,
        help="Minimum object mask area in pixels (default: 256)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs_ade20k",
        help="Output directory for metrics and summaries",
    )
    parser.add_argument(
        "--seed", type=int, default=20260720,
        help="RNG seed for reproducible sample ordering",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load samples
    # ------------------------------------------------------------------
    print(f"Loading up to {args.max_samples} ADE20K validation samples (min area ≥ {args.min_mask_area} px)…")
    t_load = time.perf_counter()
    samples = load_ade20k_samples(
        max_samples=args.max_samples,
        min_mask_area=args.min_mask_area,
        seed=args.seed,
    )
    load_sec = time.perf_counter() - t_load
    print(f"  → {len(samples)} valid samples in {load_sec:.1f} s")

    if not samples:
        print("ERROR: No valid samples loaded.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Run all CPU methods on all samples
    # ------------------------------------------------------------------
    method_names = list(CONFIRMATORY_CPU_METHODS.keys())
    all_rows: list[dict[str, Any]] = []

    print(f"Running {len(method_names)} methods × {len(samples)} samples …")
    t_run = time.perf_counter()
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  sample {i+1}/{len(samples)} …")
        for mname, mfn in CONFIRMATORY_CPU_METHODS.items():
            row = run_method(mname, mfn, sample)
            all_rows.append(row)
    run_sec = time.perf_counter() - t_run
    print(f"  → completed in {run_sec:.1f} s ({run_sec/len(samples):.2f} s/sample)")

    # ------------------------------------------------------------------
    # 3. Write per-sample metrics CSV
    # ------------------------------------------------------------------
    csv_path = out_dir / "ade20k_cpu_metrics.csv"
    write_csv(csv_path, all_rows)
    print(f"Per-sample metrics → {csv_path}")

    # ------------------------------------------------------------------
    # 4. Write summary JSON
    # ------------------------------------------------------------------
    summaries = [summarize(all_rows, m) for m in method_names]
    summary = {
        "dataset": "ADE20K (1aurent/ADE20K on Hugging Face)",
        "num_samples": len(samples),
        "min_mask_area": args.min_mask_area,
        "seed": args.seed,
        "methods": method_names,
        "load_seconds": round(load_sec, 1),
        "run_seconds": round(run_sec, 1),
        "summaries": summaries,
    }
    json_path = out_dir / "ade20k_cpu_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary → {json_path}")

    # ------------------------------------------------------------------
    # 5. Print results table
    # ------------------------------------------------------------------
    print()
    print(f"{'Method':<28s} {'Mean IoU':>8s}  {'Mean Dice':>8s}  {'Med Lat':>8s}  {'Fails':>5s}")
    print("-" * 65)
    for s in summaries:
        print(
            f"{s['method']:<28s} {s['mean_iou']:8.4f}  {s['mean_dice']:8.4f}  "
            f"{s['median_latency_ms']:7.1f}ms  {s['num_failures']:5d}"
        )


if __name__ == "__main__":
    main()
