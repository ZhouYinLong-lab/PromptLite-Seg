"""Report figures: per-class IoU bar chart + per-area line chart (CPU methods)."""
from __future__ import annotations

import csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _figure_style import apply_style, save_figure, METHOD_COLORS
import matplotlib.pyplot as plt

METHODS = ["center_color", "robust_no_color_seed", "robust_no_spatial_prior",
           "robust_single_box", "robust_superpixel", "grabcut_point_box"]
LABELS = ["Center Color", "Robust −color", "Robust −spatial",
          "Robust −consensus", "Robust Superpixel", "GrabCut"]
N = len(METHODS)


def main() -> None:
    apply_style()
    out = ROOT / "outputs_analysis"; out.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    with open(ROOT / "protocol/manifests/confirmatory_validation.jsonl", encoding="utf-8") as f:
        class_map = {json.loads(l)["sample_id"]: json.loads(l)["class_name"] for l in f if l.strip()}

    rows = []
    with open(ROOT / "artifacts/confirmatory/cpu/metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    data: dict = {m: defaultdict(list) for m in METHODS}
    for r in rows:
        m, sid, v = r["method"], r["sample_id"], r.get("iou", "").strip()
        if m in METHODS and sid in class_map and v:
            data[m][class_map[sid]].append(float(v))

    classes = sorted(set(class_map.values()))
    class_means = {m: {c: float(np.mean(data[m].get(c, [0]) or [0])) for c in classes} for m in METHODS}

    # ── Fig 1: Per-class grouped bar ──
    x = np.arange(len(classes)); w = 0.8 / N
    fig, ax = plt.subplots(figsize=(19, 6.5))

    for i, (m, lbl) in enumerate(zip(METHODS, LABELS)):
        means = [class_means[m][c] for c in classes]
        bars = ax.bar(x + i * w, means, w, label=lbl, color=METHOD_COLORS[i], zorder=2)
        for bar, val in zip(bars, means):
            if val > 0.68:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=90, color="#444444")

    ax.set_ylabel("Mean IoU")
    ax.set_title("Per-Class IoU — CPU Methods on PASCAL VOC 2012 Validation")
    ax.set_xticks(x + w * (N - 1) / 2)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8.5)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.set_ylim(0, 0.98)
    ax.axhline(y=0.6044, color="#999999", linestyle="--", linewidth=0.7, zorder=1)
    ax.text(len(classes)-0.5, 0.608, "macro mean (0.604)", fontsize=7, color="#999999", va="bottom")

    save_figure(fig, str(out / "per_class_iou.png"))
    print(f"Saved → {out / 'per_class_iou.png'}")

    # ── Fig 2: Per-area line chart ──
    area_map = {}
    with open(ROOT / "protocol/manifests/confirmatory_validation.jsonl", encoding="utf-8") as f:
        for l in f:
            m = json.loads(l); a = int(m.get("target_area", 0))
            if a > 0: area_map[m["sample_id"]] = a
    bins = np.percentile(list(area_map.values()), [25, 50, 75])

    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    for i, (m, lbl) in enumerate(zip(METHODS, LABELS)):
        qm = []
        for lo, hi in [(0, bins[0]), (bins[0], bins[1]), (bins[1], bins[2]), (bins[2], 1e9)]:
            vv = [float(r["iou"]) for r in rows
                  if r["method"] == m and r["sample_id"] in area_map
                  and lo <= area_map[r["sample_id"]] < hi and r.get("iou", "").strip()]
            qm.append(float(np.mean(vv)) if vv else 0)
        ax2.plot(range(4), qm, "o-", label=lbl, color=METHOD_COLORS[i], linewidth=2.2, markersize=7, zorder=2)

    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f"Q1\n(<{bins[0]/1000:.0f}K px)", f"Q2\n({bins[0]/1000:.0f}–{bins[1]/1000:.0f}K)",
                          f"Q3\n({bins[1]/1000:.0f}–{bins[2]/1000:.0f}K)", f"Q4\n(>{bins[2]/1000:.0f}K px)"], fontsize=9)
    ax2.set_ylabel("Mean IoU")
    ax2.set_title("IoU by Target Area Quartile — CPU Methods")
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.set_ylim(0.22, 0.85)

    save_figure(fig2, str(out / "per_area_iou.png"))
    print(f"Saved → {out / 'per_area_iou.png'}")


if __name__ == "__main__":
    main()
