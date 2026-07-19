"""Prompt robustness experiment: perturb point+box and evaluate degradation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.algorithms import center_color, robust_superpixel
from promptseg.dataset import Prompt, iter_samples
from promptseg.metrics import dice, iou
from promptseg.utils import SEVERITIES, clip_bbox, stable_rng, write_csv
from promptseg.visualize import draw_prediction_figure

__all__ = [
    "SEVERITIES",
]


def perturb_prompt(prompt: Prompt, shape: tuple[int, int], severity: str, trial: int, sample_id: str) -> Prompt:
    if severity == "clean":
        return prompt
    spec = SEVERITIES[severity]
    h, w = shape
    x0, y0, x1, y1 = prompt.bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    rng = stable_rng(sample_id, severity, trial)

    point_scale = spec["point"]
    dx = int(round(rng.normal(0, point_scale * bw)))
    dy = int(round(rng.normal(0, point_scale * bh)))
    px = int(np.clip(prompt.point[0] + dx, 0, w - 1))
    py = int(np.clip(prompt.point[1] + dy, 0, h - 1))

    box_scale = spec["box"]
    tx = int(round(rng.normal(0, box_scale * bw)))
    ty = int(round(rng.normal(0, box_scale * bh)))
    grow_l = int(round(rng.normal(0, box_scale * bw)))
    grow_t = int(round(rng.normal(0, box_scale * bh)))
    grow_r = int(round(rng.normal(0, box_scale * bw)))
    grow_b = int(round(rng.normal(0, box_scale * bh)))
    noisy_bbox = clip_bbox((x0 + tx - grow_l, y0 + ty - grow_t, x1 + tx + grow_r, y1 + ty + grow_b), shape)
    return replace(prompt, bbox=noisy_bbox, point=(px, py))


def prompt_from_mask(mask: np.ndarray, fallback: Prompt) -> Prompt:
    if not mask.any():
        return fallback
    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    dist = ndi.distance_transform_edt(mask)
    py, px = np.unravel_index(int(dist.argmax()), dist.shape)
    return replace(fallback, bbox=bbox, point=(int(px), int(py)))


def repaired_superpixel(image: np.ndarray, prompt: Prompt) -> tuple[np.ndarray, Prompt]:
    first = robust_superpixel(image, prompt)
    repaired = prompt_from_mask(first, prompt)
    second = robust_superpixel(image, repaired)
    return second, repaired


def sam_predict(predictor, prompt: Prompt) -> tuple[np.ndarray, float]:
    import torch

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=np.array([prompt.point], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            box=np.array(prompt.bbox, dtype=np.float32),
            multimask_output=True,
        )
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool), float(scores[best_idx])


def load_sam(args):
    if not args.include_sam:
        return None
    import torch

    from segment_anything import SamPredictor, sam_model_registry

    if not args.checkpoint.exists():
        raise SystemExit(f"Missing SAM checkpoint: {args.checkpoint}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor, device


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["severity"], row["method"]), []).append(row)
    summary_rows: list[dict] = []
    for (severity, method), items in sorted(grouped.items()):
        ious = np.array([float(x["iou"]) for x in items], dtype=np.float64)
        dices = np.array([float(x["dice"]) for x in items], dtype=np.float64)
        summary_rows.append(
            {
                "severity": severity,
                "method": method,
                "num_cases": len(items),
                "mean_iou": f"{ious.mean():.6f}",
                "std_iou": f"{ious.std():.6f}",
                "mean_dice": f"{dices.mean():.6f}",
                "std_dice": f"{dices.std():.6f}",
            }
        )
    return summary_rows


def plot_robustness(summary_rows: list[dict], out_path: Path) -> None:
    severity_order = list(SEVERITIES)
    methods = sorted({row["method"] for row in summary_rows})
    lookup = {(row["severity"], row["method"]): float(row["mean_iou"]) for row in summary_rows}
    x = np.arange(len(severity_order))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for method in methods:
        values = [lookup.get((severity, method), np.nan) for severity in severity_order]
        ax.plot(x, values, marker="o", linewidth=2, label=method)
    ax.set_ylim(0, 1)
    ax.set_xticks(x, severity_order)
    ax.set_ylabel("mean IoU")
    ax.set_title("Prompt perturbation robustness")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_markdown(summary_rows: list[dict], out_path: Path) -> None:
    severity_order = list(SEVERITIES)
    methods = sorted({row["method"] for row in summary_rows})
    lookup = {(row["severity"], row["method"]): row for row in summary_rows}
    lines = [
        "# Prompt Robustness Analysis",
        "",
        "This benchmark perturbs point and box prompts, then evaluates how each method degrades.",
        "",
        "| Severity | Method | Mean IoU | Mean Dice | Cases |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for severity in severity_order:
        for method in methods:
            row = lookup.get((severity, method))
            if row:
                lines.append(
                    f"| {severity} | {method} | {row['mean_iou']} | {row['mean_dice']} | {row['num_cases']} |"
                )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "The repaired variants first use the lightweight superpixel mask to infer a new point and box, "
        "then rerun the downstream method. This tests whether a transparent baseline can act as a "
        "prompt repair module rather than only a weak competitor."
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/voc_subset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--include-sam", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam_vit_b_01ec64.pth"))
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    samples = list(iter_samples(args.data_dir))[: args.max_samples]
    predictor_and_device = load_sam(args)
    predictor = predictor_and_device[0] if predictor_and_device else None
    device = predictor_and_device[1] if predictor_and_device else "none"

    rows: list[dict] = []
    examples_written = 0
    for sample in samples:
        if predictor is not None:
            predictor.set_image(sample.image)
        for severity in SEVERITIES:
            trial_count = 1 if severity == "clean" else args.trials
            for trial in range(trial_count):
                noisy_prompt = perturb_prompt(sample.prompt, sample.mask.shape, severity, trial, sample.sample_id)

                predictions: dict[str, np.ndarray] = {}
                for method_name, method in (
                    ("center_color", center_color),
                    ("robust_superpixel", robust_superpixel),
                ):
                    pred = method(sample.image, noisy_prompt)
                    predictions[method_name] = pred
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "class_name": sample.prompt.class_name,
                            "severity": severity,
                            "trial": trial,
                            "method": method_name,
                            "iou": f"{iou(pred, sample.mask):.6f}",
                            "dice": f"{dice(pred, sample.mask):.6f}",
                            "device": "cpu",
                        }
                    )

                repaired_pred, repaired_prompt = repaired_superpixel(sample.image, noisy_prompt)
                predictions["repaired_superpixel"] = repaired_pred
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "class_name": sample.prompt.class_name,
                        "severity": severity,
                        "trial": trial,
                        "method": "repaired_superpixel",
                        "iou": f"{iou(repaired_pred, sample.mask):.6f}",
                        "dice": f"{dice(repaired_pred, sample.mask):.6f}",
                        "device": "cpu",
                    }
                )

                if predictor is not None:
                    sam_pred, sam_score = sam_predict(predictor, noisy_prompt)
                    predictions["sam_noisy"] = sam_pred
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "class_name": sample.prompt.class_name,
                            "severity": severity,
                            "trial": trial,
                            "method": "sam_noisy_prompt",
                            "iou": f"{iou(sam_pred, sample.mask):.6f}",
                            "dice": f"{dice(sam_pred, sample.mask):.6f}",
                            "device": device,
                        }
                    )
                    sam_repaired_pred, sam_repaired_score = sam_predict(predictor, sample.image, repaired_prompt)
                    predictions["sam_repaired"] = sam_repaired_pred
                    sam_repaired_iou = iou(sam_repaired_pred, sample.mask)
                    sam_noisy_iou = iou(sam_pred, sample.mask)
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "class_name": sample.prompt.class_name,
                            "severity": severity,
                            "trial": trial,
                            "method": "sam_repaired_prompt",
                            "iou": f"{iou(sam_repaired_pred, sample.mask):.6f}",
                            "dice": f"{dice(sam_repaired_pred, sample.mask):.6f}",
                            "device": device,
                        }
                    )
                    score_selected = sam_repaired_pred if sam_repaired_score > sam_score else sam_pred
                    oracle_best = sam_repaired_pred if sam_repaired_iou > sam_noisy_iou else sam_pred
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "class_name": sample.prompt.class_name,
                            "severity": severity,
                            "trial": trial,
                            "method": "sam_score_selected_prompt",
                            "iou": f"{iou(score_selected, sample.mask):.6f}",
                            "dice": f"{dice(score_selected, sample.mask):.6f}",
                            "device": device,
                        }
                    )
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "class_name": sample.prompt.class_name,
                            "severity": severity,
                            "trial": trial,
                            "method": "sam_oracle_best_prompt",
                            "iou": f"{iou(oracle_best, sample.mask):.6f}",
                            "dice": f"{dice(oracle_best, sample.mask):.6f}",
                            "device": device,
                        }
                    )

                if severity == "moderate" and trial == 0 and examples_written < 4:
                    draw_prediction_figure(
                        sample,
                        predictions,
                        args.output_dir / "figures" / f"{sample.sample_id}_moderate.png",
                    )
                    examples_written += 1

    summary_rows = summarize(rows)
    write_csv(args.output_dir / "metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    plot_robustness(summary_rows, args.output_dir / "robustness_curve.png")
    write_markdown(summary_rows, args.output_dir / "robustness_analysis.md")
    payload = {
        "num_samples": len(samples),
        "trials": args.trials,
        "include_sam": args.include_sam,
        "device": device,
        "summary": summary_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
