"""Resume-safe confirmatory SAM benchmark with no image-bearing outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.dataset import load_sample
from promptseg.metrics import dice, iou
from promptseg.prompts import (
    bbox_iou,
    jitter_around_observed_prompt,
    perturb_by_severity,
    point_hits_target,
    select_ensemble_predictions,
)
from promptseg.sam import PROMPT_MODES, predict_sam
from promptseg.utils import write_csv


CHECKPOINT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def metric_row(sample, experiment: str, condition: str, method: str, trial: int, pred, score: float, **extra) -> dict:
    return {
        "sample_id": sample.sample_id,
        "class_name": sample.prompt.class_name,
        "experiment": experiment,
        "severity": "clean" if experiment == "modality" else "moderate",
        "condition": condition,
        "method": method,
        "trial": trial,
        "iou": f"{iou(pred, sample.mask):.6f}",
        "dice": f"{dice(pred, sample.mask):.6f}",
        "sam_score": f"{score:.6f}",
        "point_hit": extra.get("point_hit", ""),
        "box_iou": extra.get("box_iou", ""),
    }


def evaluate_sample(predictor, sample, trials: int, ensemble_size: int) -> list[dict]:
    rows: list[dict] = []
    predictor.set_image(sample.image)

    for mode in PROMPT_MODES:
        pred, score = predict_sam(predictor, sample.prompt, mode)
        rows.append(metric_row(sample, "modality", mode, "sam_vit_b", 0, pred, score))

    for noise_source in ("point_noise", "box_noise"):
        for trial in range(trials):
            noisy = perturb_by_severity(
                sample.prompt,
                sample.mask.shape,
                "moderate",
                noise_source,
                trial,
                sample.sample_id,
            )
            pred, score = predict_sam(predictor, noisy)
            rows.append(
                metric_row(
                    sample,
                    "noise_decomposition",
                    noise_source,
                    "sam_single_prompt",
                    trial,
                    pred,
                    score,
                    point_hit=str(point_hits_target(noisy.point, sample.mask)).lower(),
                    box_iou=f"{bbox_iou(sample.prompt.bbox, noisy.bbox):.6f}",
                )
            )

    for trial in range(trials):
        noisy = perturb_by_severity(
            sample.prompt,
            sample.mask.shape,
            "moderate",
            "point_box_noise",
            trial,
            sample.sample_id,
        )
        candidates = [noisy]
        candidates.extend(
            jitter_around_observed_prompt(
                noisy,
                sample.mask.shape,
                "moderate",
                trial,
                variant_id,
                sample.sample_id,
            )
            for variant_id in range(ensemble_size)
        )
        candidate_predictions = [predict_sam(predictor, candidate) for candidate in candidates]
        masks = [item[0] for item in candidate_predictions]
        scores = [item[1] for item in candidate_predictions]
        selections = select_ensemble_predictions(masks, scores, sample.mask)
        for method, (pred, score) in selections.items():
            rows.append(
                metric_row(
                    sample,
                    "uncertainty_ensemble",
                    "point_box_noise",
                    method,
                    trial,
                    pred,
                    score,
                    point_hit=str(point_hits_target(noisy.point, sample.mask)).lower(),
                    box_iou=f"{bbox_iou(sample.prompt.bbox, noisy.bbox):.6f}",
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/voc_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_confirmatory/sam"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam_vit_b_01ec64.pth"))
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    import torch
    from segment_anything import SamPredictor, sam_model_registry

    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("SAM checkpoint SHA-256 does not match the frozen protocol")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "sample_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    sample_dirs = [
        path
        for path in sorted(args.data_dir.iterdir())
        if path.is_dir()
        and (path / "image.jpg").is_file()
        and (path / "target_mask.png").is_file()
        and (path / "prompt.txt").is_file()
    ]
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]

    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=args.device)
    predictor = SamPredictor(sam)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    completed_before = 0
    for index, sample_dir in enumerate(sample_dirs, start=1):
        sample_path = checkpoint_dir / f"{sample_dir.name}.json"
        if sample_path.exists():
            completed_before += 1
            continue
        sample = load_sample(sample_dir)
        rows = evaluate_sample(predictor, sample, args.trials, args.ensemble_size)
        sample_path.write_text(json.dumps(rows) + "\n", encoding="utf-8")
        if index % 25 == 0 or index == len(sample_dirs):
            print(f"Completed {index}/{len(sample_dirs)} samples", flush=True)

    rows: list[dict] = []
    for sample_dir in sample_dirs:
        path = checkpoint_dir / f"{sample_dir.name}.json"
        if not path.exists():
            raise RuntimeError(f"Missing completed checkpoint for {sample_dir.name}")
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    write_csv(args.output_dir / "metrics.csv", rows)

    process = psutil.Process()
    payload = {
        "git_commit": git_commit(),
        "confirmatory": args.max_samples is None and len(sample_dirs) == 1449,
        "num_samples": len(sample_dirs),
        "num_rows": len(rows),
        "trials": args.trials,
        "ensemble_size": args.ensemble_size,
        "device": args.device,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "completed_before_resume": completed_before,
        "elapsed_seconds_this_run": perf_counter() - start,
        "process_rss_mb": process.memory_info().rss / (1024 * 1024),
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if args.device.startswith("cuda") else None
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
