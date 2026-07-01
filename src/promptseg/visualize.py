from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .dataset import Sample


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_prediction_figure(sample: Sample, predictions: dict[str, np.ndarray], out_path: Path) -> None:
    cols = 2 + len(predictions)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(sample.image)
    axes[0].set_title(f"{sample.sample_id}: {sample.prompt.class_name}")
    x0, y0, x1, y1 = sample.prompt.bbox
    axes[0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="yellow", linewidth=2)
    axes[0].scatter([sample.prompt.point[0]], [sample.prompt.point[1]], color="cyan", s=40)

    axes[1].imshow(overlay(sample.image, sample.mask, (0, 210, 80)))
    axes[1].set_title("ground truth")

    for ax, (name, pred) in zip(axes[2:], predictions.items()):
        ax.imshow(overlay(sample.image, pred, (220, 40, 40)))
        ax.set_title(name)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_metric_summary(summary: dict, out_path: Path) -> None:
    methods = [m for m in summary.keys() if isinstance(summary[m], dict)]
    ious = [summary[m]["mean_iou"] for m in methods]
    dices = [summary[m]["mean_dice"] for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, ious, width, label="IoU", color="#4778c7")
    ax.bar(x + width / 2, dices, width, label="Dice", color="#55a868")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, methods, rotation=12)
    ax.set_ylabel("score")
    ax.legend()
    ax.set_title("Prompt segmentation performance on VOC subset")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

