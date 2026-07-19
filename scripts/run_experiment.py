"""Run the lightweight prompt segmentation baselines on the VOC subset."""

from __future__ import annotations

import argparse
from itertools import islice
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.algorithms import METHODS
from promptseg.dataset import iter_samples
from promptseg.metrics import dice, iou
from promptseg.utils import write_csv
from promptseg.visualize import draw_metric_summary, draw_prediction_figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/voc_subset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    sample_iterator = iter_samples(args.data_dir)
    samples = list(islice(sample_iterator, args.max_samples)) if args.max_samples is not None else list(sample_iterator)
    if not samples:
        raise SystemExit(f"No samples found in {args.data_dir}. Run scripts/download_voc_subset.py first.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    aggregate: dict[str, dict[str, list[float]]] = {name: {"iou": [], "dice": []} for name in METHODS}
    for sample in samples:
        predictions = {}
        for name, method in METHODS.items():
            pred = method(sample.image, sample.prompt)
            predictions[name] = pred
            sample_iou = iou(pred, sample.mask)
            sample_dice = dice(pred, sample.mask)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "class_name": sample.prompt.class_name,
                    "method": name,
                    "iou": f"{sample_iou:.6f}",
                    "dice": f"{sample_dice:.6f}",
                    "mask_pixels": int(sample.mask.sum()),
                    "pred_pixels": int(pred.sum()),
                }
            )
            aggregate[name]["iou"].append(sample_iou)
            aggregate[name]["dice"].append(sample_dice)
        draw_prediction_figure(sample, predictions, args.output_dir / "figures" / f"{sample.sample_id}.png")

    summary = {"num_samples": len(samples)}
    for name, values in aggregate.items():
        summary[name] = {
            "mean_iou": float(np.mean(values["iou"])),
            "std_iou": float(np.std(values["iou"])),
            "mean_dice": float(np.mean(values["dice"])),
            "std_dice": float(np.std(values["dice"])),
        }
    write_csv(args.output_dir / "metrics.csv", rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_metric_summary(summary, args.output_dir / "figures" / "metric_summary.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
