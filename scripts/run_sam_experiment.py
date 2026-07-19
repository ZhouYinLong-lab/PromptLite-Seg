"""Run SAM (Segment Anything Model) on the VOC subset for comparison."""

from __future__ import annotations

import argparse
from itertools import islice
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.dataset import iter_samples
from promptseg.metrics import dice, iou
from promptseg.sam import predict_sam
from promptseg.utils import write_csv
from promptseg.visualize import draw_prediction_figure


def draw_method_comparison(rows: list[dict], out_path: Path) -> None:
    methods = [row["method"] for row in rows]
    ious = [float(row["mean_iou"]) for row in rows]
    dices = [float(row["mean_dice"]) for row in rows]
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - width / 2, ious, width, label="IoU", color="#4c78a8")
    ax.bar(x + width / 2, dices, width, label="Dice", color="#59a96a")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("SAM comparison with lightweight prompt baselines")
    ax.set_xticks(x, methods, rotation=12, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_method_comparison(base_summary_path: Path, sam_summary: dict, out_dir: Path) -> None:
    if not base_summary_path.exists():
        return
    base = json.loads(base_summary_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for method in ("center_color", "robust_superpixel"):
        if method in base:
            rows.append(
                {
                    "method": method,
                    "mean_iou": f"{base[method]['mean_iou']:.6f}",
                    "std_iou": f"{base[method]['std_iou']:.6f}",
                    "mean_dice": f"{base[method]['mean_dice']:.6f}",
                    "std_dice": f"{base[method]['std_dice']:.6f}",
                }
            )
    rows.append(
        {
            "method": sam_summary["method"],
            "mean_iou": f"{sam_summary['mean_iou']:.6f}",
            "std_iou": f"{sam_summary['std_iou']:.6f}",
            "mean_dice": f"{sam_summary['mean_dice']:.6f}",
            "std_dice": f"{sam_summary['std_dice']:.6f}",
        }
    )
    write_csv(out_dir / "method_comparison.csv", rows)
    draw_method_comparison(rows, out_dir / "method_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/voc_subset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sam"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam_vit_b_01ec64.pth"))
    parser.add_argument("--baseline-summary", type=Path, default=Path("outputs/summary.json"))
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be at least 1")

    import torch

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise SystemExit(
            "segment-anything is not installed. Install it in the SAM environment with "
            "`python -m pip install segment-anything`."
        ) from exc

    if not args.checkpoint.exists():
        raise SystemExit(
            f"Missing SAM checkpoint: {args.checkpoint}. Download SAM ViT-B from "
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )

    samples = list(islice(iter_samples(args.data_dir), args.max_samples))
    if not samples:
        raise SystemExit(f"No samples found in {args.data_dir}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    rows: list[dict] = []
    ious: list[float] = []
    dices: list[float] = []

    # Cache the most recent image embedding to avoid redundant set_image calls
    # when the same image is processed multiple times in other experiments.
    cached_image_id: str | None = None

    for sample in samples:
        if sample.sample_id != cached_image_id:
            predictor.set_image(sample.image)
            cached_image_id = sample.sample_id

        pred, score = predict_sam(predictor, sample.prompt)
        sample_iou = iou(pred, sample.mask)
        sample_dice = dice(pred, sample.mask)
        ious.append(sample_iou)
        dices.append(sample_dice)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "class_name": sample.prompt.class_name,
                "method": f"sam_{args.model_type}_point_box",
                "iou": f"{sample_iou:.6f}",
                "dice": f"{sample_dice:.6f}",
                "score": f"{score:.6f}",
                "mask_pixels": int(sample.mask.sum()),
                "pred_pixels": int(pred.sum()),
            }
        )
        draw_prediction_figure(
            sample,
            {f"sam_{args.model_type}": pred},
            args.output_dir / "figures" / f"{sample.sample_id}.png",
        )

    summary = {
        "num_samples": len(samples),
        "method": f"sam_{args.model_type}_point_box",
        "device": args.device,
        "checkpoint": str(args.checkpoint),
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious)),
        "mean_dice": float(np.mean(dices)),
        "std_dice": float(np.std(dices)),
    }
    write_csv(args.output_dir / "metrics.csv", rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_method_comparison(args.baseline_summary, summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
