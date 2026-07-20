"""Report figure: failure/success case error maps for Robust Superpixel."""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from _figure_style import apply_style, save_figure
from promptseg.algorithms import robust_superpixel
from promptseg.dataset import load_sample

# ── Color-blind friendly diff map ──
TP = (26, 152, 80)     # green — correct
FP = (215, 48, 39)     # red — false positive
FN = (69, 117, 180)    # blue — false negative
BG = (245, 245, 245)   # light gray — background


def diff_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    h, w = pred.shape
    dm = np.full((h, w, 3), BG, dtype=np.uint8)
    dm[pred & gt] = TP
    dm[pred & ~gt] = FP
    dm[~pred & gt] = FN
    return dm


def select_diverse(cases, n):
    seen, sel = set(), []
    for c in cases:
        if c["class_name"] not in seen or len(sel) < n:
            sel.append(c); seen.add(c["class_name"])
        if len(sel) >= n: break
    return sel[:n]


def main() -> None:
    apply_style()
    out = ROOT / "outputs_analysis" / "failure_figures"
    out.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data" / "voc_validation"

    fa = json.loads((ROOT / "outputs_analysis/failure_analysis.json").read_text(encoding="utf-8"))
    worst, best = fa["worst_20"], fa["best_20"]
    cases = select_diverse(worst, 3) + select_diverse(best, 3)

    # ── 2×3 grid ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5))
    for idx, case in enumerate(cases):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        sd = data_dir / case["sample_id"]
        if not sd.exists():
            ax.text(0.5, 0.5, "N/A", ha="center", va="center"); ax.axis("off"); continue

        s = load_sample(sd)
        pred = robust_superpixel(s.image, s.prompt)
        ax.imshow(diff_map(pred, s.mask))
        tag = "Worst" if idx < 3 else "Best"
        ax.set_title(f"{tag}: {case['class_name']} | IoU={case['iou']:.3f} | {case['target_area']:,} px",
                     fontsize=10, fontweight="bold" if idx < 3 else "normal",
                     color="#C0392B" if idx < 3 else "#27AE60")
        ax.axis("off")

    # Unified legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(TP)/255, label="True positive (correct)"),
        Patch(facecolor=np.array(FP)/255, label="False positive (over-seg)"),
        Patch(facecolor=np.array(FN)/255, label="False negative (under-seg)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
               frameon=True, edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Robust Superpixel — Failure & Success Cases (Error Maps)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.subplots_adjust(hspace=0.25, wspace=0.05)

    save_figure(fig, str(out / "failure_report_grid.png"), dpi=200)
    print(f"Saved → {out / 'failure_report_grid.png'}")


if __name__ == "__main__":
    main()
