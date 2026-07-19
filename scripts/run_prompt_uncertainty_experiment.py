"""Prompt uncertainty experiment: modality, noise, and ensemble effects for SAM."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.dataset import Prompt, iter_samples
from promptseg.metrics import dice, iou
from promptseg.sam import PROMPT_MODES, predict_sam
from promptseg.utils import SEVERITIES, clip_bbox, stable_rng, write_csv
from promptseg.visualize import draw_prediction_figure

__all__ = [
    "PROMPT_MODES",
    "NOISE_SOURCES",
    "ENSEMBLE_METHODS",
]

NOISE_SOURCES = ("point_noise", "box_noise", "point_box_noise")
ENSEMBLE_METHODS = (
    "sam_single_noisy",
    "sam_score_select",
    "sam_consistency_medoid",
    "sam_vote_consensus",
    "sam_oracle_best",
)


def perturb_prompt(
    prompt: Prompt,
    shape: tuple[int, int],
    severity: str,
    noise_source: str,
    trial: int,
    sample_id: str,
) -> Prompt:
    if severity == "clean":
        return prompt
    if noise_source not in {"point_noise", "box_noise", "point_box_noise"}:
        raise ValueError(f"Unknown noise source: {noise_source}")

    spec = SEVERITIES[severity]
    h, w = shape
    x0, y0, x1, y1 = prompt.bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    rng = stable_rng(sample_id, severity, noise_source, trial)

    px, py = prompt.point
    if noise_source in {"point_noise", "point_box_noise"}:
        dx = int(round(rng.normal(0, spec["point"] * bw)))
        dy = int(round(rng.normal(0, spec["point"] * bh)))
        px = int(np.clip(px + dx, 0, w - 1))
        py = int(np.clip(py + dy, 0, h - 1))

    bbox = prompt.bbox
    if noise_source in {"box_noise", "point_box_noise"}:
        box_scale = spec["box"]
        tx = int(round(rng.normal(0, box_scale * bw)))
        ty = int(round(rng.normal(0, box_scale * bh)))
        grow_l = int(round(rng.normal(0, box_scale * bw)))
        grow_t = int(round(rng.normal(0, box_scale * bh)))
        grow_r = int(round(rng.normal(0, box_scale * bw)))
        grow_b = int(round(rng.normal(0, box_scale * bh)))
        bbox = clip_bbox((x0 + tx - grow_l, y0 + ty - grow_t, x1 + tx + grow_r, y1 + ty + grow_b), shape)

    return replace(prompt, bbox=bbox, point=(px, py))


def jitter_around_observed_prompt(
    prompt: Prompt,
    shape: tuple[int, int],
    severity: str,
    trial: int,
    variant_id: int,
    sample_id: str,
) -> Prompt:
    spec = SEVERITIES[severity]
    h, w = shape
    x0, y0, x1, y1 = prompt.bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    rng = stable_rng(sample_id, "ensemble", severity, trial, variant_id)

    # Search locally around the observed noisy prompt. The original clean prompt is not used.
    point_scale = 0.55 * spec["point"]
    dx = int(round(rng.normal(0, point_scale * bw)))
    dy = int(round(rng.normal(0, point_scale * bh)))
    px = int(np.clip(prompt.point[0] + dx, 0, w - 1))
    py = int(np.clip(prompt.point[1] + dy, 0, h - 1))

    box_scale = 0.55 * spec["box"]
    tx = int(round(rng.normal(0, box_scale * bw)))
    ty = int(round(rng.normal(0, box_scale * bh)))
    grow_l = int(round(rng.normal(0, box_scale * bw)))
    grow_t = int(round(rng.normal(0, box_scale * bh)))
    grow_r = int(round(rng.normal(0, box_scale * bw)))
    grow_b = int(round(rng.normal(0, box_scale * bh)))
    bbox = clip_bbox((x0 + tx - grow_l, y0 + ty - grow_t, x1 + tx + grow_r, y1 + ty + grow_b), shape)
    return replace(prompt, bbox=bbox, point=(px, py))


def mask_iou_matrix(masks: list[np.ndarray]) -> np.ndarray:
    n = len(masks)
    out = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            value = iou(masks[i], masks[j])
            out[i, j] = value
            out[j, i] = value
    return out


def select_consistency_medoid(masks: list[np.ndarray]) -> np.ndarray:
    if len(masks) == 1:
        return masks[0]
    pairwise = mask_iou_matrix(masks)
    scores = (pairwise.sum(axis=1) - 1.0) / (len(masks) - 1)
    return masks[int(np.argmax(scores))]


def select_ensemble_predictions(
    candidate_masks: list[np.ndarray],
    candidate_scores: list[float],
    target: np.ndarray,
) -> dict[str, tuple[np.ndarray, float]]:
    """Return deployable ensemble selections plus an explicit oracle bound."""

    if not candidate_masks or len(candidate_masks) != len(candidate_scores):
        raise ValueError("candidate masks and scores must be non-empty and aligned")
    oracle_idx = int(np.argmax([iou(mask, target) for mask in candidate_masks]))
    return {
        "sam_single_noisy": (candidate_masks[0], candidate_scores[0]),
        "sam_score_select": (
            candidate_masks[int(np.argmax(candidate_scores))],
            max(candidate_scores),
        ),
        "sam_consistency_medoid": (
            select_consistency_medoid(candidate_masks),
            float(np.mean(candidate_scores)),
        ),
        "sam_vote_consensus": (
            np.mean(np.stack(candidate_masks, axis=0), axis=0) >= 0.5,
            float(np.mean(candidate_scores)),
        ),
        "sam_oracle_best": (candidate_masks[oracle_idx], candidate_scores[oracle_idx]),
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["experiment"], row["severity"], row["condition"], row["method"])
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for (experiment, severity, condition, method), items in sorted(grouped.items()):
        ious = np.array([float(item["iou"]) for item in items], dtype=np.float64)
        dices = np.array([float(item["dice"]) for item in items], dtype=np.float64)
        summary_rows.append(
            {
                "experiment": experiment,
                "severity": severity,
                "condition": condition,
                "method": method,
                "num_cases": len(items),
                "mean_iou": f"{ious.mean():.6f}",
                "std_iou": f"{ious.std():.6f}",
                "mean_dice": f"{dices.mean():.6f}",
                "std_dice": f"{dices.std():.6f}",
            }
        )
    return summary_rows


def plot_bar(rows: list[dict], out_path: Path, title: str, ylabel: str = "mean IoU") -> None:
    labels = [row["condition"] if row["condition"] != "point_box_noise" else "point+box_noise" for row in rows]
    values = [float(row["mean_iou"]) for row in rows]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#72b7b2"][: len(rows)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(np.arange(len(rows)), values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(rows)), labels, rotation=15, ha="right")
    for idx, value in enumerate(values):
        ax.text(idx, min(0.98, value + 0.02), f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_noise_decomposition(summary_rows: list[dict], out_path: Path) -> None:
    severities = ["clean", "mild", "moderate"]
    conditions = ["clean", "point_noise", "box_noise", "point_box_noise"]
    lookup = {
        (row["severity"], row["condition"]): float(row["mean_iou"])
        for row in summary_rows
        if row["experiment"] == "noise_decomposition"
    }
    x = np.arange(len(severities))
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for condition, color in zip(conditions, ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]):
        values = [lookup.get((severity, condition), np.nan) for severity in severities]
        if np.all(np.isnan(values)):
            continue
        label = condition.replace("_", " ")
        ax.plot(x, values, marker="o", linewidth=2, label=label, color=color)
    ax.set_ylim(0, 1)
    ax.set_xticks(x, severities)
    ax.set_ylabel("mean IoU")
    ax.set_title("SAM sensitivity to point and box prompt noise")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_markdown(summary_rows: list[dict], out_path: Path) -> None:
    lines = [
        "# Prompt Uncertainty Research Experiment",
        "",
        "This experiment uses SAM ViT-B with one image embedding per sample and multiple prompt queries.",
        "It separates prompt modality, point-vs-box noise, and multi-prompt uncertainty selection.",
        "",
        "## Summary",
        "",
        "| Experiment | Severity | Condition | Method | Mean IoU | Mean Dice | Cases |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['experiment']} | {row['severity']} | {row['condition']} | {row['method']} | "
            f"{row['mean_iou']} | {row['mean_dice']} | {row['num_cases']} |"
        )

    def find(experiment: str, severity: str, condition: str, method: str) -> float | None:
        for row in summary_rows:
            if (
                row["experiment"] == experiment
                and row["severity"] == severity
                and row["condition"] == condition
                and row["method"] == method
            ):
                return float(row["mean_iou"])
        return None

    clean_point_box = find("modality", "clean", "point_box", "sam_vit_b")
    clean_box = find("modality", "clean", "box_only", "sam_vit_b")
    moderate_single = find("uncertainty_ensemble", "moderate", "point_box_noise", "sam_single_noisy")
    moderate_medoid = find("uncertainty_ensemble", "moderate", "point_box_noise", "sam_consistency_medoid")
    moderate_oracle = find("uncertainty_ensemble", "moderate", "point_box_noise", "sam_oracle_best")

    lines.extend(["", "## Findings", ""])
    if clean_point_box is not None and clean_box is not None:
        lines.append(
            f"- Clean point+box SAM reaches {clean_point_box:.4f} mean IoU, while box-only reaches {clean_box:.4f}. "
            "This quantifies how much the extra point contributes beyond localization."
        )
    if moderate_single is not None and moderate_medoid is not None and moderate_oracle is not None:
        delta = moderate_medoid - moderate_single
        gap = moderate_oracle - moderate_single
        lines.append(
            f"- Under moderate point+box noise, consistency-medoid changes IoU by {delta:+.4f} over a single noisy prompt. "
            f"The oracle gap is {gap:+.4f}, estimating the recoverable headroom from prompt uncertainty."
        )
    lines.append(
        "- The key research question is no longer whether SAM is strong under clean prompts, but which prompt channel "
        "fails under realistic annotation noise and whether agreement among prompt variants is a usable reliability signal."
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_row(
    rows: list[dict],
    sample_id: str,
    class_name: str,
    experiment: str,
    severity: str,
    condition: str,
    method: str,
    trial: int,
    pred: np.ndarray,
    target: np.ndarray,
    score: float,
    device: str,
) -> None:
    rows.append(
        {
            "sample_id": sample_id,
            "class_name": class_name,
            "experiment": experiment,
            "severity": severity,
            "condition": condition,
            "method": method,
            "trial": trial,
            "iou": f"{iou(pred, target):.6f}",
            "dice": f"{dice(pred, target):.6f}",
            "sam_score": f"{score:.6f}",
            "pred_pixels": int(pred.sum()),
            "target_pixels": int(target.sum()),
            "device": device,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/voc_subset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prompt_uncertainty"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam_vit_b_01ec64.pth"))
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

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
        raise SystemExit(f"Missing SAM checkpoint: {args.checkpoint}")

    samples = list(iter_samples(args.data_dir))[: args.max_samples]
    if not samples:
        raise SystemExit(f"No samples found in {args.data_dir}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    rows: list[dict] = []
    examples_written = 0

    # Cache SAM image embeddings per sample to avoid redundant set_image calls.
    cached_image_id: str | None = None

    for sample in samples:
        if sample.sample_id != cached_image_id:
            predictor.set_image(sample.image)
            cached_image_id = sample.sample_id

        for prompt_mode in PROMPT_MODES:
            pred, score = predict_sam(predictor, sample.prompt, prompt_mode)
            add_row(
                rows,
                sample.sample_id,
                sample.prompt.class_name,
                "modality",
                "clean",
                prompt_mode,
                f"sam_{args.model_type}",
                0,
                pred,
                sample.mask,
                score,
                args.device,
            )

        clean_pred, clean_score = predict_sam(predictor, sample.prompt, "point_box")
        add_row(
            rows,
            sample.sample_id,
            sample.prompt.class_name,
            "noise_decomposition",
            "clean",
            "clean",
            "sam_single_prompt",
            0,
            clean_pred,
            sample.mask,
            clean_score,
            args.device,
        )

        for severity in ("mild", "moderate"):
            for noise_source in NOISE_SOURCES:
                for trial in range(args.trials):
                    noisy_prompt = perturb_prompt(
                        sample.prompt,
                        sample.mask.shape,
                        severity,
                        noise_source,
                        trial,
                        sample.sample_id,
                    )
                    pred, score = predict_sam(predictor, noisy_prompt, "point_box")
                    add_row(
                        rows,
                        sample.sample_id,
                        sample.prompt.class_name,
                        "noise_decomposition",
                        severity,
                        noise_source,
                        "sam_single_prompt",
                        trial,
                        pred,
                        sample.mask,
                        score,
                        args.device,
                    )

            for trial in range(args.trials):
                noisy_prompt = perturb_prompt(
                    sample.prompt,
                    sample.mask.shape,
                    severity,
                    "point_box_noise",
                    trial,
                    sample.sample_id,
                )
                candidate_prompts = [noisy_prompt]
                candidate_prompts.extend(
                    jitter_around_observed_prompt(
                        noisy_prompt,
                        sample.mask.shape,
                        severity,
                        trial,
                        variant_id,
                        sample.sample_id,
                    )
                    for variant_id in range(args.ensemble_size)
                )

                candidate_masks: list[np.ndarray] = []
                candidate_scores: list[float] = []
                for candidate_prompt in candidate_prompts:
                    mask, score = predict_sam(predictor, candidate_prompt, "point_box")
                    candidate_masks.append(mask)
                    candidate_scores.append(score)

                method_predictions = select_ensemble_predictions(
                    candidate_masks,
                    candidate_scores,
                    sample.mask,
                )
                for method, (pred, score) in method_predictions.items():
                    add_row(
                        rows,
                        sample.sample_id,
                        sample.prompt.class_name,
                        "uncertainty_ensemble",
                        severity,
                        "point_box_noise",
                        method,
                        trial,
                        pred,
                        sample.mask,
                        score,
                        args.device,
                    )

                if severity == "moderate" and trial == 0 and examples_written < 4:
                    single_pred = method_predictions["sam_single_noisy"][0]
                    score_pred = method_predictions["sam_score_select"][0]
                    medoid_pred = method_predictions["sam_consistency_medoid"][0]
                    vote_pred = method_predictions["sam_vote_consensus"][0]
                    oracle_pred = method_predictions["sam_oracle_best"][0]
                    draw_prediction_figure(
                        sample,
                        {
                            "single": single_pred,
                            "score": score_pred,
                            "medoid": medoid_pred,
                            "vote": vote_pred,
                            "oracle": oracle_pred,
                        },
                        args.output_dir / "figures" / f"{sample.sample_id}_uncertainty.png",
                    )
                    examples_written += 1

    summary_rows = summarize(rows)
    write_csv(args.output_dir / "metrics.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_samples": len(samples),
                "trials": args.trials,
                "ensemble_size": args.ensemble_size,
                "model_type": args.model_type,
                "device": args.device,
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    modality_rows = [
        row
        for row in summary_rows
        if row["experiment"] == "modality" and row["severity"] == "clean" and row["method"] == f"sam_{args.model_type}"
    ]
    modality_order = {mode: idx for idx, mode in enumerate(PROMPT_MODES)}
    modality_rows.sort(key=lambda row: modality_order[row["condition"]])
    plot_bar(modality_rows, args.output_dir / "prompt_modality.png", "Clean SAM prompt modality comparison")

    plot_noise_decomposition(summary_rows, args.output_dir / "noise_decomposition.png")

    ensemble_rows = [
        {**row, "condition": row["method"].replace("sam_", "")}
        for row in summary_rows
        if row["experiment"] == "uncertainty_ensemble"
        and row["severity"] == "moderate"
        and row["condition"] == "point_box_noise"
    ]
    ensemble_order = {method: idx for idx, method in enumerate(ENSEMBLE_METHODS)}
    ensemble_rows.sort(key=lambda row: ensemble_order[row["method"]])
    plot_bar(ensemble_rows, args.output_dir / "uncertainty_ensemble.png", "Moderate noise multi-prompt selection")

    write_markdown(summary_rows, args.output_dir / "research_findings.md")
    print(json.dumps({"num_rows": len(rows), "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
