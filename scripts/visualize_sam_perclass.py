"""Report figure: SAM vs CPU per-class IoU comparison."""
from __future__ import annotations

import csv, json, sys
from collections import defaultdict
from pathlib import Path

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


def main() -> None:
    apply_style()
    out = ROOT / "outputs_analysis"; out.mkdir(parents=True, exist_ok=True)

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
                    data[r.get("method", r.get("condition", "?"))][class_map[sid]].append(float(v))
        return data

    cpu = load_means(ROOT / "artifacts/confirmatory/cpu/metrics.csv")
    sam = load_means(ROOT / "artifacts/confirmatory/sam/metrics.csv",
                     lambda r: r.get("experiment") == "modality" and r.get("severity") == "clean")

    classes = sorted(set(class_map.values()))

    def means(source, key):
        vals = source.get(key, {})
        return [float(np.mean(vals.get(c, [0]) or [0])) for c in classes]

    # ── Figure ──
    x = np.arange(len(classes))
    w = 0.8 / len(ALL_KEYS)
    fig, ax = plt.subplots(figsize=(21, 7))

    for i, (key, lbl) in enumerate(zip(ALL_KEYS, ALL_LABELS)):
        src = cpu if key in CPU_KEYS else sam
        vals = means(src, key)
        ax.bar(x + i * w, vals, w, label=lbl, color=FULL_COLORS[i], zorder=2)

    ax.set_ylabel("Mean IoU")
    ax.set_title("Per-Class IoU — CPU Methods vs SAM ViT-B (VOC 2012 Validation)")
    ax.set_xticks(x + w * (len(ALL_KEYS) - 1) / 2)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8.5)
    ax.legend(loc="lower left", ncol=2, fontsize=7.8)
    ax.set_ylim(0, 1.02)

    save_figure(fig, str(out / "per_class_sam_vs_cpu.png"))
    print(f"Saved → {out / 'per_class_sam_vs_cpu.png'}")

    # ── SAM-only clean chart ──
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
