"""Per-class, per-size, and per-aspect-ratio stratified analysis.

Reads the confirmatory CPU metrics CSV and the validation manifest to produce
detailed breakdown tables for the report.
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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    metrics_path = ROOT / "artifacts" / "confirmatory" / "cpu" / "metrics.csv"
    manifest_path = ROOT / "protocol" / "manifests" / "confirmatory_validation.jsonl"

    if not metrics_path.exists():
        print(f"ERROR: Metrics file not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    # Load metrics
    all_rows = read_csv(metrics_path)
    print(f"Loaded {len(all_rows)} metric rows")

    # Load manifest for per-sample metadata
    manifest_samples = {}
    if manifest_path.exists():
        manifest = read_jsonl(manifest_path)
        for entry in manifest:
            sid = entry.get("sample_id", "")
            manifest_samples[sid] = entry

    # Organize: sample_id -> {method: {iou, dice, ...}}
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in all_rows:
        sid = row.get("sample_id", "")
        method = row.get("method", "")
        by_sample[sid][method] = {
            "iou": float(row.get("iou", 0)),
            "dice": float(row.get("dice", 0)),
            "success": row.get("success", "True") == "True",
            "latency_ms": float(row.get("latency_ms", 0)),
        }

    # Enrich with manifest metadata
    for sid in by_sample:
        if sid in manifest_samples:
            m = manifest_samples[sid]
            area_raw = m.get("target_area", 0)
            area = int(area_raw) if area_raw else 0
            if area > 0:
                by_sample[sid]["_area"] = area
            by_sample[sid]["_class"] = m.get("class_name", "unknown")
            by_sample[sid]["_img_h"] = int(m.get("image_height", 0))
            by_sample[sid]["_img_w"] = int(m.get("image_width", 0))

    # Build per-class and per-area-quartile strata
    all_areas = [by_sample[s]["_area"] for s in by_sample if "_area" in by_sample[s]]
    if all_areas:
        area_bins = np.percentile(all_areas, [25, 50, 75])
    else:
        area_bins = [0, 0, 0]

    def area_quartile(area: float) -> str:
        if area <= area_bins[0]:
            return "Q1 (small)"
        elif area <= area_bins[1]:
            return "Q2 (medium-small)"
        elif area <= area_bins[2]:
            return "Q3 (medium-large)"
        else:
            return "Q4 (large)"

    # Compute per-method aggregates per class
    methods = sorted(set(r["method"] for r in all_rows))
    classes = sorted(set(by_sample[s].get("_class", "unknown") for s in by_sample))
    classes = [c for c in classes if c != "unknown"]

    # ── Per-class table ──
    print("\n" + "=" * 80)
    print("PER-CLASS MEAN IoU")
    print("=" * 80)
    header = f"{'Class':<20s}"
    for m in methods:
        header += f" {m:>22s}"
    print(header)
    print("-" * len(header))
    for cls in classes:
        line = f"{cls:<20s}"
        for m in methods:
            vals = []
            for sid, methods_dict in by_sample.items():
                if by_sample[sid].get("_class") == cls and m in methods_dict:
                    vals.append(methods_dict[m]["iou"])
            if vals:
                line += f" {np.mean(vals):>21.4f}"
            else:
                line += f" {'—':>21s}"
        print(line)

    # ── Per-area-quartile table ──
    print("\n" + "=" * 80)
    print("PER-AREA-QUARTILE MEAN IoU")
    print(f"Quartile boundaries (px): {[int(b) for b in area_bins]}")
    print("=" * 80)
    header = f"{'Quartile':<20s} {'Samples':>8s}"
    for m in methods:
        header += f" {m:>22s}"
    print(header)
    print("-" * len(header))
    quartiles = ["Q1 (small)", "Q2 (medium-small)", "Q3 (medium-large)", "Q4 (large)"]
    for qi, qname in enumerate(quartiles):
        q_ious = {m: [] for m in methods}
        for sid, methods_dict in by_sample.items():
            area = by_sample[sid].get("_area", 0)
            if area > 0 and area_quartile(area) == qname:
                for m in methods:
                    if m in methods_dict:
                        q_ious[m].append(methods_dict[m]["iou"])
        n = len(q_ious[methods[0]]) if methods else 0
        line = f"{qname:<20s} {n:>8d}"
        for m in methods:
            if q_ious[m]:
                line += f" {np.mean(q_ious[m]):>21.4f}"
            else:
                line += f" {'—':>21s}"
        print(line)

    # ── Summary statistics for the report ──
    print("\n" + "=" * 80)
    print("KEY FINDINGS FOR REPORT")
    print("=" * 80)

    # Best/worst classes for robust_superpixel
    if "robust_superpixel" in methods:
        rs = "robust_superpixel"
        class_means = {}
        for cls in classes:
            vals = []
            for sid, md in by_sample.items():
                if by_sample[sid].get("_class") == cls and rs in md:
                    vals.append(md[rs]["iou"])
            if vals:
                class_means[cls] = np.mean(vals)
        sorted_classes = sorted(class_means.items(), key=lambda x: x[1])
        print(f"\n{rs} — Best 3 classes:")
        for cls, m in sorted_classes[-3:]:
            print(f"  {cls}: {m:.4f}")
        print(f"{rs} — Worst 3 classes:")
        for cls, m in sorted_classes[:3]:
            print(f"  {cls}: {m:.4f}")

    # Area-quartile spread per method
    print("\nArea-quartile IoU spread (Q4 mean − Q1 mean):")
    for m in methods:
        q1_vals, q4_vals = [], []
        for sid, md in by_sample.items():
            area = by_sample[sid].get("_area", 0)
            q = area_quartile(area) if area > 0 else ""
            if m in md:
                if q == "Q1 (small)":
                    q1_vals.append(md[m]["iou"])
                elif q == "Q4 (large)":
                    q4_vals.append(md[m]["iou"])
        if q1_vals and q4_vals:
            spread = np.mean(q4_vals) - np.mean(q1_vals)
            print(f"  {m:<30s}: {spread:+.4f} IoU")

    # GrabCut failures per class
    print("\nGrabCut failures by class:")
    gc_fails = defaultdict(int)
    gc_totals = defaultdict(int)
    for sid, md in by_sample.items():
        cls = by_sample[sid].get("_class", "unknown")
        if "grabcut_point_box" in md:
            gc_totals[cls] += 1
            if not md["grabcut_point_box"]["success"]:
                gc_fails[cls] += 1
    for cls in sorted(gc_fails):
        if gc_fails[cls] > 0:
            print(f"  {cls}: {gc_fails[cls]}/{gc_totals[cls]}")

    # ── Per-aspect-ratio analysis ──
    print("\n" + "=" * 80)
    print("PER-ASPECT-RATIO MEAN IoU")
    print("=" * 80)
    # Compute aspect ratio for each sample (width/height of target bbox)
    ar_vals = []
    for sid in by_sample:
        if sid in manifest_samples:
            m = manifest_samples[sid]
            bbox_raw = m.get("bbox", [])
            if isinstance(bbox_raw, str):
                bbox_raw = [int(x.strip()) for x in bbox_raw.strip("[]").split(",")]
            if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                x0, y0, x1, y1 = [int(x) for x in bbox_raw]
                bw, bh = x1 - x0, y1 - y0
                if bh > 0:
                    ar_vals.append(bw / bh)
    if ar_vals:
        ar_bins = np.percentile(ar_vals, [25, 50, 75])
        print(f"AR quartile boundaries: {[round(b, 2) for b in ar_bins]}")
        for qi, (lo, hi, label) in enumerate([
            (0, ar_bins[0], "Tall (AR < {:.2f})"),
            (ar_bins[0], ar_bins[1], "Slightly tall"),
            (ar_bins[1], ar_bins[2], "Slightly wide"),
            (ar_bins[2], 999, "Wide (AR > {:.2f})"),
        ]):
            label_str = label.format(lo if qi == 0 else hi)
            vals = {m: [] for m in methods}
            for sid, md in by_sample.items():
                if sid in manifest_samples:
                    mf = manifest_samples[sid]
                    bbox_raw = mf.get("bbox", [])
                    if isinstance(bbox_raw, str):
                        bbox_raw = [int(x.strip()) for x in bbox_raw.strip("[]").split(",")]
                    if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                        x0, y0, x1, y1 = [int(x) for x in bbox_raw]
                        bw, bh = x1 - x0, y1 - y0
                        ar = bw / bh if bh > 0 else 1.0
                        if lo <= ar < hi or (qi == 3 and ar >= lo):
                                for m in methods:
                                    if m in md:
                                        vals[m].append(md[m]["iou"])
            n = len(vals[methods[0]]) if methods else 0
            line = f"{label_str:<28s} {n:>5d}"
            for m in methods:
                if vals[m]:
                    line += f" {np.mean(vals[m]):>21.4f}"
            print(line)

    # ── Save structured JSON summary ──
    out_dir = ROOT / "outputs_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = {
        "num_samples": len(by_sample),
        "num_classes_with_data": len(classes),
        "area_quartile_boundaries_px": [int(b) for b in area_bins],
        "per_class": {
            cls: {
                m: float(np.mean([
                    by_sample[s][m]["iou"]
                    for s in by_sample
                    if by_sample[s].get("_class") == cls and m in by_sample[s]
                ]))
                for m in methods
                if any(
                    by_sample[s].get("_class") == cls and m in by_sample[s]
                    for s in by_sample
                )
            }
            for cls in classes
        },
    }
    (out_dir / "stratified_summary.json").write_text(
        json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nStructured summary → {out_dir / 'stratified_summary.json'}")


if __name__ == "__main__":
    main()
