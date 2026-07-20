"""Run only adaptive_superpixel on VOC validation and compare with baseline."""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.algorithms import adaptive_superpixel, robust_superpixel
from promptseg.dataset import load_sample
from promptseg.metrics import iou, dice
from promptseg.utils import write_csv


def main() -> None:
    data_dir = ROOT / "data" / "voc_validation"
    sample_dirs = sorted(data_dir.glob("val_*"))
    if not sample_dirs:
        print("ERROR: No VOC validation samples found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(sample_dirs)} samples")

    out_dir = ROOT / "outputs_adaptive"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.perf_counter()
    for i, sd in enumerate(sample_dirs):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  {i+1}/{len(sample_dirs)} …")
        sample = load_sample(sd)

        # adaptive
        t1 = time.perf_counter()
        try:
            pred_a = adaptive_superpixel(sample.image, sample.prompt)
            success_a = True
            iou_a = iou(pred_a, sample.mask)
            dice_a = dice(pred_a, sample.mask)
        except Exception as e:
            success_a = False
            iou_a = 0.0
            dice_a = 0.0
            print(f"  FAIL adaptive on {sample.sample_id}: {e}", file=sys.stderr)
        lat_a = (time.perf_counter() - t1) * 1000

        # baseline robust_superpixel
        t2 = time.perf_counter()
        try:
            pred_r = robust_superpixel(sample.image, sample.prompt)
            success_r = True
            iou_r = iou(pred_r, sample.mask)
            dice_r = dice(pred_r, sample.mask)
        except Exception as e:
            success_r = False
            iou_r = 0.0
            dice_r = 0.0
            print(f"  FAIL robust on {sample.sample_id}: {e}", file=sys.stderr)
        lat_r = (time.perf_counter() - t2) * 1000

        rows.append({
            "sample_id": sample.sample_id,
            "class_name": sample.prompt.class_name,
            "mask_area_px": int(sample.mask.sum()),
            "adaptive_iou": iou_a,
            "adaptive_dice": dice_a,
            "adaptive_success": success_a,
            "adaptive_latency_ms": round(lat_a, 2),
            "robust_iou": iou_r,
            "robust_dice": dice_r,
            "robust_success": success_r,
            "robust_latency_ms": round(lat_r, 2),
        })

    elapsed = time.perf_counter() - t0
    print(f"Completed in {elapsed:.1f} s ({elapsed/len(sample_dirs):.2f} s/sample)")

    # Write metrics
    csv_path = out_dir / "adaptive_vs_robust.csv"
    write_csv(csv_path, rows)
    print(f"Metrics → {csv_path}")

    # Summary
    a_ious = np.array([r["adaptive_iou"] for r in rows], dtype=np.float64)
    r_ious = np.array([r["robust_iou"] for r in rows], dtype=np.float64)
    a_fails = sum(1 for r in rows if not r["adaptive_success"])
    r_fails = sum(1 for r in rows if not r["robust_success"])
    delta = a_ious - r_ious
    # Paired bootstrap
    rng = np.random.default_rng(20260720)
    n_boot = 20000
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(delta), len(delta))
        boot_means[b] = np.mean(delta[idx])
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    summary = {
        "num_samples": len(rows),
        "adaptive": {
            "mean_iou": float(np.mean(a_ious)),
            "median_iou": float(np.median(a_ious)),
            "mean_dice": float(np.mean([r["adaptive_dice"] for r in rows])),
            "failures": a_fails,
            "median_latency_ms": float(np.median([r["adaptive_latency_ms"] for r in rows])),
        },
        "robust": {
            "mean_iou": float(np.mean(r_ious)),
            "median_iou": float(np.median(r_ious)),
            "mean_dice": float(np.mean([r["robust_dice"] for r in rows])),
            "failures": r_fails,
            "median_latency_ms": float(np.median([r["robust_latency_ms"] for r in rows])),
        },
        "paired_delta_mean": float(np.mean(delta)),
        "paired_delta_95ci": [ci_lo, ci_hi],
        "significant": bool(ci_lo > 0 or ci_hi < 0),  # CI doesn't cross zero
    }
    json_path = out_dir / "adaptive_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary → {json_path}")

    # Print comparison
    print()
    print(f"{'Method':<25s} {'Mean IoU':>8s}  {'Mean Dice':>8s}  {'Fails':>5s}")
    print("-" * 55)
    print(f"{'adaptive_superpixel':<25s} {summary['adaptive']['mean_iou']:8.4f}  {summary['adaptive']['mean_dice']:8.4f}  {a_fails:5d}")
    print(f"{'robust_superpixel':<25s} {summary['robust']['mean_iou']:8.4f}  {summary['robust']['mean_dice']:8.4f}  {r_fails:5d}")
    print(f"\nPaired Δ (adaptive − robust): {summary['paired_delta_mean']:+.4f} IoU, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"Significant: {summary['significant']}")


if __name__ == "__main__":
    main()
