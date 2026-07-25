"""Report figure: SAM vs CPU per-class IoU comparison."""
from __future__ import annotations

import csv, json, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _figure_style import apply_style, save_figure, FULL_COLORS

CPU_KEYS = ["center_color", "robust_superpixel", "grabcut_point_box"]
CPU_LABELS = ["Center Color (CPU)", "Robust Superpixel (CPU)", "GrabCut (CPU)"]
SAM_KEYS = ["point_only", "box_only", "point_box"]
SAM_LABELS = ["SAM point-only", "SAM box-only", "SAM point+box"]
ALL_KEYS = CPU_KEYS + SAM_KEYS
ALL_LABELS = CPU_LABELS + SAM_LABELS
KEY_CLASSES = ["bicycle", "tvmonitor", "sofa", "bus", "cat", "bird"]
SERIES_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
SERIES_MARKERS = ["o", "s", "D", "^", "v", "P"]


def main() -> None:
    apply_style()
    out = ROOT / "outputs_analysis"; out.mkdir(parents=True, exist_ok=True)
    report_figures = ROOT / "reports" / "figures"
    report_figures.mkdir(parents=True, exist_ok=True)

    # ── Class mapping ──
    with open(ROOT / "protocol/manifests/confirmatory_validation.jsonl", encoding="utf-8") as f:
        class_map = {json.loads(l)["sample_id"]: json.loads(l)["class_name"] for l in f if l.strip()}

    def load_means(path, filter_fn=None):
        data = defaultdict(lambda: defaultdict(list))
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if filter_fn and not filter_fn(r):
                    continue
                sid, v = r["sample_id"], r.get("iou", "").strip()
                if sid in class_map and v:
                    # SAM modality rows share the method name ``sam_vit_b``;
                    # their prompt modality is encoded in ``condition``.
                    series_key = r.get("condition") or r.get("method") or "?"
                    data[series_key][class_map[sid]].append(float(v))
        return data

    cpu = load_means(ROOT / "artifacts/confirmatory/cpu/metrics.csv")
    sam = load_means(ROOT / "artifacts/confirmatory/sam/metrics.csv",
                     lambda r: r.get("experiment") == "modality" and r.get("severity") == "clean")

    classes = sorted(set(class_map.values()))

    def means(source, key):
        vals = source.get(key, {})
        return [float(np.mean(vals.get(c, [0]) or [0])) for c in classes]

    # ── Main-text figure: selected diagnostic classes ──
    y = np.arange(len(KEY_CLASSES))
    offsets = np.linspace(-0.30, 0.30, len(ALL_KEYS))
    fig, ax = plt.subplots(figsize=(11, 6.4))

    for i, (key, lbl, color, marker) in enumerate(
        zip(ALL_KEYS, ALL_LABELS, SERIES_COLORS, SERIES_MARKERS)
    ):
        src = cpu if key in CPU_KEYS else sam
        vals_by_class = dict(zip(classes, means(src, key)))
        vals = [vals_by_class[c] for c in KEY_CLASSES]
        ax.scatter(
            vals,
            y + offsets[i],
            s=82,
            marker=marker,
            label=lbl,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    ax.set_xlabel("Mean IoU", fontsize=14)
    ax.set_yticks(y)
    ax.set_yticklabels(KEY_CLASSES, fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.suptitle("Diagnostic Classes: CPU Baselines vs SAM ViT-B", fontsize=16, y=0.98)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        fontsize=11,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(top=0.76)

    main_path = report_figures / "per_class_key_sam_vs_cpu.pdf"
    save_figure(fig, str(main_path))
    print(f"Saved → {main_path}")

    # ── SAM-only clean chart ──
    x = np.arange(len(classes))
    fig2, ax2 = plt.subplots(figsize=(16, 4.8))
    for i, (key, lbl) in enumerate(zip(SAM_KEYS, SAM_LABELS)):
        vals = means(sam, key)
        ax2.bar(x + i * 0.28, vals, 0.28, label=lbl, color=FULL_COLORS[len(CPU_KEYS)+i], zorder=2)
    ax2.set_ylabel("Mean IoU")
    ax2.set_title("SAM ViT-B Prompt Modalities — Per-Class IoU")
    ax2.set_xticks(x + 0.28)
    ax2.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax2.legend(fontsize=9.5)
    ax2.set_ylim(0, 1.02)

    save_figure(fig2, str(out / "per_class_sam_only.png"))
    print(f"Saved → {out / 'per_class_sam_only.png'}")


if __name__ == "__main__":
    main()
