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
from promptseg.protocol import (
    atomic_write_text,
    canonical_json_sha256,
    dataset_fingerprint,
    git_is_dirty,
    manifest_sample_ids,
    sha256_file,
)
from promptseg.utils import write_csv


CHECKPOINT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
ENSEMBLE_METHODS = (
    "sam_single_noisy",
    "sam_score_select",
    "sam_consistency_medoid",
    "sam_vote_consensus",
    "sam_oracle_best",
)


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


def expected_sample_keys(trials: int) -> set[tuple[str, str, str, int]]:
    keys = {("modality", mode, "sam_vit_b", 0) for mode in PROMPT_MODES}
    keys.update(
        ("noise_decomposition", condition, "sam_single_prompt", trial)
        for condition in ("point_noise", "box_noise")
        for trial in range(trials)
    )
    keys.update(
        ("uncertainty_ensemble", "point_box_noise", method, trial)
        for method in ENSEMBLE_METHODS
        for trial in range(trials)
    )
    return keys


def validate_checkpoint_payload(
    payload: object,
    *,
    sample_id: str,
    run_fingerprint: str,
    trials: int,
) -> list[dict]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Legacy or malformed checkpoint for {sample_id}; use a fresh output directory")
    if payload.get("run_fingerprint") != run_fingerprint or payload.get("sample_id") != sample_id:
        raise RuntimeError(f"Checkpoint fingerprint mismatch for {sample_id}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Checkpoint rows are malformed for {sample_id}")
    observed_keys: list[tuple[str, str, str, int]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("sample_id") != sample_id:
            raise RuntimeError(f"Checkpoint contains misaligned rows for {sample_id}")
        observed_keys.append(
            (
                str(row.get("experiment")),
                str(row.get("condition")),
                str(row.get("method")),
                int(row.get("trial")),
            )
        )
    expected = expected_sample_keys(trials)
    if len(observed_keys) != len(expected) or set(observed_keys) != expected:
        raise RuntimeError(f"Checkpoint design is incomplete or duplicated for {sample_id}")
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
    parser.add_argument("--protocol", type=Path, default=Path("protocol/research_protocol.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("protocol/manifests/confirmatory_validation.jsonl"),
    )
    args = parser.parse_args()
    if args.trials < 1 or args.ensemble_size < 1:
        parser.error("--trials and --ensemble-size must be at least 1")

    import torch
    from segment_anything import SamPredictor, sam_model_registry

    observed_checkpoint_sha256 = sha256_file(args.checkpoint)
    if observed_checkpoint_sha256 != CHECKPOINT_SHA256:
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

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected_ids = manifest_sample_ids(args.manifest)
    observed_ids = [path.name for path in sample_dirs]
    run_config = {
        "schema_version": 1,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(ROOT),
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_sha256": sha256_file(args.manifest),
        "dataset_sha256": dataset_fingerprint(sample_dirs),
        "checkpoint_sha256": observed_checkpoint_sha256,
        "model_type": args.model_type,
        "device": args.device,
        "trials": args.trials,
        "ensemble_size": args.ensemble_size,
        "sample_ids": observed_ids,
    }
    run_fingerprint = canonical_json_sha256(run_config)

    config_path = args.output_dir / "run_config.json"
    existing_checkpoints = list(checkpoint_dir.glob("*.json"))
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise SystemExit("Output directory belongs to a different SAM run configuration")
    elif existing_checkpoints:
        raise SystemExit("Checkpoint files exist without a matching run_config.json")
    else:
        atomic_write_text(config_path, json.dumps(run_config, indent=2) + "\n")

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
            validate_checkpoint_payload(
                json.loads(sample_path.read_text(encoding="utf-8")),
                sample_id=sample_dir.name,
                run_fingerprint=run_fingerprint,
                trials=args.trials,
            )
            completed_before += 1
            continue
        sample = load_sample(sample_dir)
        rows = evaluate_sample(predictor, sample, args.trials, args.ensemble_size)
        checkpoint_payload = {
            "run_fingerprint": run_fingerprint,
            "sample_id": sample.sample_id,
            "rows": rows,
        }
        validate_checkpoint_payload(
            checkpoint_payload,
            sample_id=sample.sample_id,
            run_fingerprint=run_fingerprint,
            trials=args.trials,
        )
        atomic_write_text(sample_path, json.dumps(checkpoint_payload) + "\n")
        if index % 25 == 0 or index == len(sample_dirs):
            print(f"Completed {index}/{len(sample_dirs)} samples", flush=True)

    rows: list[dict] = []
    for sample_dir in sample_dirs:
        path = checkpoint_dir / f"{sample_dir.name}.json"
        if not path.exists():
            raise RuntimeError(f"Missing completed checkpoint for {sample_dir.name}")
        rows.extend(
            validate_checkpoint_payload(
                json.loads(path.read_text(encoding="utf-8")),
                sample_id=sample_dir.name,
                run_fingerprint=run_fingerprint,
                trials=args.trials,
            )
        )
    write_csv(args.output_dir / "metrics.csv", rows)

    process = psutil.Process()
    frozen_trials = int(protocol["methods"]["sam_confirmatory_trials_per_sample"])
    frozen_ensemble_size = int(protocol["methods"]["sam_ensemble_additional_candidates"])
    payload = {
        "git_commit": git_commit(),
        "run_fingerprint": run_fingerprint,
        "protocol_sha256": run_config["protocol_sha256"],
        "manifest_sha256": run_config["manifest_sha256"],
        "dataset_sha256": run_config["dataset_sha256"],
        "confirmatory": (
            args.max_samples is None
            and observed_ids == expected_ids
            and args.trials == frozen_trials
            and args.ensemble_size == frozen_ensemble_size
            and args.model_type == protocol["methods"]["sam_model"]
            and not run_config["git_dirty"]
        ),
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
