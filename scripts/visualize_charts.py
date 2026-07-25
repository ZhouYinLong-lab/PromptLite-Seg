"""Report figures: per-class IoU bar chart + per-area line chart (CPU methods)."""
from __future__ import annotations

import csv, json, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt
from _figure_style import apply_style, save_figure, METHOD_COLORS

METHODS = ["center_color", "robust_no_color_seed", "robust_no_spatial_prior",
           "robust_single_box", "robust_superpixel", "grabcut_point_box"]
LABELS = ["Center Color", "Robust w/o color", "Robust w/o spatial",
          "Robust single box", "Robust Superpixel", "GrabCut"]
N = len(METHODS)
SERIES_COLORS = ["#D55E00", "#E69F00", "#0072B2", "#56B4E9", "#009E73", "#CC79A7"]
SERIES_MARKERS = ["o", "s", "D", "^", "v", "P"]


def main() -> None:
    apply_style()
    out = ROOT / "outputs_analysis"; out.mkdir(parents=True, exist_ok=True)
    report_figures = ROOT / "reports" / "figures"
    report_figures.mkdir(parents=True, exist_ok=True)

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

    # ── Fig 1: Full 20-class comparison, split into two readable panels ──
    class_groups = [classes[:10], classes[10:]]
    offsets = np.linspace(-0.30, 0.30, N)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 11.5), sharex=True)

    for panel_index, (ax, panel_classes) in enumerate(zip(axes, class_groups)):
        y = np.arange(len(panel_classes))
        for i, (m, lbl, color, marker) in enumerate(
            zip(METHODS, LABELS, SERIES_COLORS, SERIES_MARKERS)
        ):
            vals = [class_means[m][c] for c in panel_classes]
            ax.scatter(
                vals,
                y + offsets[i],
                s=68,
                marker=marker,
                label=lbl,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(panel_classes, fontsize=13)
        ax.tick_params(axis="x", labelsize=12)
        ax.invert_yaxis()
        ax.set_xlim(0, 0.9)
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", visible=False)
        ax.set_title(
            f"Classes {panel_index * 10 + 1}–{panel_index * 10 + len(panel_classes)}",
            fontsize=14,
            loc="left",
        )

    axes[-1].set_xlabel("Mean IoU", fontsize=14)
    fig.suptitle("CPU Methods by VOC Class", fontsize=16, y=0.995)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        fontsize=11,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(top=0.90, hspace=0.24)

    class_path = report_figures / "per_class_cpu_split.pdf"
    save_figure(fig, str(class_path))
    print(f"Saved → {class_path}")

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
