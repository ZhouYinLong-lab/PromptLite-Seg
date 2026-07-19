"""Resume-safe confirmatory SAM benchmark with no image-bearing outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
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
    base_runtime_environment,
    canonical_json_sha256,
    dataset_fingerprint,
    git_commit,
    git_is_dirty,
    manifest_sample_ids,
    module_source_fingerprint,
    sha256_file,
    verify_runtime_sources,
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
        if not isinstance(row.get("class_name"), str) or not row["class_name"]:
            raise RuntimeError(f"Checkpoint contains an invalid class label for {sample_id}")
        trial_value = row.get("trial")
        if type(trial_value) is not int:
            raise RuntimeError(f"Checkpoint contains a non-integer trial for {sample_id}")
        experiment = str(row.get("experiment"))
        expected_severity = "clean" if experiment == "modality" else "moderate"
        if row.get("severity") != expected_severity:
            raise RuntimeError(f"Checkpoint contains an invalid severity for {sample_id}")
        for metric in ("iou", "dice", "sam_score"):
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Checkpoint contains invalid {metric} for {sample_id}") from error
            if not np.isfinite(value) or (metric in {"iou", "dice"} and not 0.0 <= value <= 1.0):
                raise RuntimeError(f"Checkpoint contains out-of-range {metric} for {sample_id}")
        if experiment == "modality":
            if row.get("point_hit", "") != "" or row.get("box_iou", "") != "":
                raise RuntimeError(f"Checkpoint modality rows contain noisy-prompt quality for {sample_id}")
        else:
            if str(row.get("point_hit")).lower() not in {"true", "false"}:
                raise RuntimeError(f"Checkpoint contains invalid point_hit for {sample_id}")
            try:
                box_iou_value = float(row["box_iou"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Checkpoint contains invalid box_iou for {sample_id}") from error
            if not np.isfinite(box_iou_value) or not 0.0 <= box_iou_value <= 1.0:
                raise RuntimeError(f"Checkpoint contains out-of-range box_iou for {sample_id}")
        observed_keys.append(
            (
                experiment,
                str(row.get("condition")),
                str(row.get("method")),
                trial_value,
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
    parser.add_argument(
        "--dataset-fingerprints",
        type=Path,
        default=Path("protocol/dataset_fingerprints.json"),
    )
    parser.add_argument(
        "--runtime-sources",
        type=Path,
        default=Path("protocol/runtime_sources.json"),
    )
    args = parser.parse_args()
    if args.trials < 1 or args.ensemble_size < 1:
        parser.error("--trials and --ensemble-size must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")

    import torch
    import segment_anything
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
    dataset_spec = json.loads(args.dataset_fingerprints.read_text(encoding="utf-8"))
    expected_dataset_sha256 = dataset_spec["confirmatory"]["dataset_sha256"]
    initial_sources = verify_runtime_sources(ROOT, args.runtime_sources)
    initial_commit = git_commit(ROOT)
    initial_dirty = git_is_dirty(ROOT)
    runtime_environment = base_runtime_environment(("numpy", "Pillow", "torch", "torchvision"))
    runtime_environment["torch"] = {
        "version": getattr(torch, "__version__", None),
        "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn": (
            getattr(getattr(getattr(torch, "backends", None), "cudnn", None), "version", lambda: None)()
        ),
    }
    if args.device.startswith("cuda"):
        runtime_environment["torch"]["device_name"] = getattr(
            torch.cuda, "get_device_name", lambda *_args: None
        )(args.device)
        get_device_capability = getattr(torch.cuda, "get_device_capability", lambda *_args: ())
        runtime_environment["torch"]["device_capability"] = list(get_device_capability(args.device))
    runtime_environment["segment_anything_source_sha256"] = module_source_fingerprint(segment_anything)
    run_config = {
        "schema_version": 2,
        "git_commit": initial_commit,
        "git_dirty": initial_dirty,
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_sha256": sha256_file(args.manifest),
        "dataset_sha256": dataset_fingerprint(sample_dirs),
        "expected_dataset_sha256": expected_dataset_sha256,
        "dataset_fingerprints_sha256": sha256_file(args.dataset_fingerprints),
        "runtime_sources_sha256": initial_sources["specification_sha256"],
        "source_tree_sha256": initial_sources["fingerprint"],
        "source_tree_matches": initial_sources["matches"],
        "runtime_environment": runtime_environment,
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
    process = psutil.Process()
    final_dataset_sha256 = dataset_fingerprint(sample_dirs)
    final_sources = verify_runtime_sources(ROOT, args.runtime_sources)
    final_commit = git_commit(ROOT)
    final_dirty = git_is_dirty(ROOT)
    if final_dataset_sha256 != run_config["dataset_sha256"]:
        raise RuntimeError("Dataset changed while the SAM confirmatory run was executing")
    if final_sources["fingerprint"] != initial_sources["fingerprint"]:
        raise RuntimeError("Runtime sources changed while the SAM confirmatory run was executing")
    if not final_sources["matches"]:
        raise RuntimeError("Runtime sources no longer match the frozen source specification")
    git_stable = initial_commit == final_commit and final_dirty in (False, None)
    write_csv(args.output_dir / "metrics.csv", rows)
    frozen_trials = int(protocol["methods"]["sam_confirmatory_trials_per_sample"])
    frozen_ensemble_size = int(protocol["methods"]["sam_ensemble_additional_candidates"])
    payload = {
        "git_commit": initial_commit,
        "git_dirty": final_dirty,
        "run_fingerprint": run_fingerprint,
        "protocol_sha256": run_config["protocol_sha256"],
        "manifest_sha256": run_config["manifest_sha256"],
        "dataset_sha256": run_config["dataset_sha256"],
        "metrics_sha256": sha256_file(args.output_dir / "metrics.csv"),
        "confirmatory": (
            args.max_samples is None
            and observed_ids == expected_ids
            and args.trials == frozen_trials
            and args.ensemble_size == frozen_ensemble_size
            and args.model_type == protocol["methods"]["sam_model"]
            and final_dataset_sha256 == expected_dataset_sha256
            and initial_sources["matches"]
            and final_sources["matches"]
            and initial_dirty in (False, None)
            and git_stable
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
