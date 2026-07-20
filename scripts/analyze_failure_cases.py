"""Find and describe failure/success cases for qualitative analysis.

Identifies the worst and best Robust Superpixel predictions, generates
overlay visualizations, and prints common failure patterns.
"""
from __future__ import annotations

import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_csv(path: Path) -> list[dict]:
    import csv
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    metrics_path = ROOT / "artifacts" / "confirmatory" / "cpu" / "metrics.csv"
    manifest_path = ROOT / "protocol" / "manifests" / "confirmatory_validation.jsonl"

    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found", file=sys.stderr)
        sys.exit(1)

    rows = read_csv(metrics_path)

    # Load manifest for metadata
    manifest = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                manifest[m["sample_id"]] = m

    # Extract robust_superpixel IoU per sample
    rs_data = []
    for r in rows:
        if r["method"] == "robust_superpixel":
            sid = r["sample_id"]
            rs_data.append({
                "sample_id": sid,
                "iou": float(r["iou"]),
                "dice": float(r["dice"]),
                "latency_ms": float(r.get("latency_ms", 0)),
                "class_name": manifest.get(sid, {}).get("class_name", "?"),
                "target_area": int(manifest.get(sid, {}).get("target_area", 0)),
            })

    rs_data.sort(key=lambda x: x["iou"])

    # ── Top-20 and bottom-20 ──
    worst = rs_data[:20]
    best = rs_data[-20:]

    print("=" * 70)
    print("WORST 20 — Robust Superpixel (IoU)")
    print("=" * 70)
    for s in worst:
        print(f"  {s['sample_id']:<16s} class={s['class_name']:<14s} "
              f"area={s['target_area']:>6d}px  IoU={s['iou']:.4f}")

    print()
    print("=" * 70)
    print("BEST 20 — Robust Superpixel (IoU)")
    print("=" * 70)
    for s in best:
        print(f"  {s['sample_id']:<16s} class={s['class_name']:<14s} "
              f"area={s['target_area']:>6d}px  IoU={s['iou']:.4f}")

    # ── Common failure patterns ──
    print("\n" + "=" * 70)
    print("FAILURE PATTERN ANALYSIS")
    print("=" * 70)

    # Classes that appear disproportionately in worst 20
    print("\nClass distribution in worst 20 vs all:")
    all_classes = defaultdict(int)
    worst_classes = defaultdict(int)
    for s in rs_data:
        all_classes[s["class_name"]] += 1
    for s in worst:
        worst_classes[s["class_name"]] += 1
    for cls in sorted(worst_classes, key=lambda c: worst_classes[c] / all_classes[c], reverse=True):
        ratio = worst_classes[cls] / all_classes[cls] * 100
        print(f"  {cls:<16s}: {worst_classes[cls]:>2d}/{all_classes[cls]:>3d} ({ratio:.1f}%)")

    # Area distribution
    all_areas = [s["target_area"] for s in rs_data]
    worst_areas = [s["target_area"] for s in worst]
    best_areas = [s["target_area"] for s in best]
    print(f"\nMedian target area: all={np.median(all_areas):.0f}px, "
          f"worst-20={np.median(worst_areas):.0f}px, "
          f"best-20={np.median(best_areas):.0f}px")

    # IoU distribution summary
    all_ious = np.array([s["iou"] for s in rs_data])
    print(f"\nIoU distribution: median={np.median(all_ious):.4f}, "
          f"P5={np.percentile(all_ious, 5):.4f}, "
          f"P95={np.percentile(all_ious, 95):.4f}")

    # ── Compare worst-20 with other methods ──
    print("\n" + "=" * 70)
    print("CROSS-METHOD COMPARISON ON WORST-20 SAMPLES")
    print("=" * 70)
    worst_ids = {s["sample_id"] for s in worst}
    method_ious = defaultdict(list)
    for r in rows:
        if r["sample_id"] in worst_ids:
            method_ious[r["method"]].append(float(r["iou"]))
    print(f"{'Method':<30s} {'Mean IoU on worst-20':>20s}")
    print("-" * 52)
    for m in sorted(method_ious):
        print(f"  {m:<30s} {np.mean(method_ious[m]):>19.4f}")

    # ── Save results ──
    out_dir = ROOT / "outputs_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "robust_superpixel",
        "num_samples_total": len(rs_data),
        "median_iou": float(np.median(all_ious)),
        "worst_20": [
            {"sample_id": s["sample_id"], "class_name": s["class_name"],
             "iou": s["iou"], "target_area": s["target_area"]}
            for s in worst
        ],
        "best_20": [
            {"sample_id": s["sample_id"], "class_name": s["class_name"],
             "iou": s["iou"], "target_area": s["target_area"]}
            for s in best
        ],
    }
    (out_dir / "failure_analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"\nSaved → {out_dir / 'failure_analysis.json'}")


if __name__ == "__main__":
    main()
