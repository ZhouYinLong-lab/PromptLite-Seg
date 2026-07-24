"""Resume-safe SAM evaluation for multi-quality sensitivity curve.

Evaluates SAM ViT-B under independently calibrated point-only and box-only
perturbations at five quality targets.  The other prompt channel is kept
clean for each condition.

Usage::

    python scripts/run_sensitivity_sam.py                    # full 1,449-sample run
    python scripts/run_sensitivity_sam.py --max-samples 10   # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.dataset import load_sample
from promptseg.metrics import dice, iou
from promptseg.prompts import bbox_iou, perturb_prompt, point_hits_target
from promptseg.sam import predict_sam
from promptseg.protocol import (
    atomic_write_text,
    base_runtime_environment,
    dataset_fingerprint,
    git_commit,
    git_is_dirty,
    manifest_sample_ids,
    sha256_file,
    verify_runtime_sources,
)
from promptseg.utils import write_csv

CHECKPOINT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"


def _select_best_masks(
    masks: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one complete 2-D mask per batch item using its highest score."""
    if masks.ndim != 4 or scores.ndim != 2:
        raise ValueError("Expected masks [B,C,H,W] and scores [B,C]")
    if masks.shape[:2] != scores.shape:
        raise ValueError("Mask and score batch/candidate dimensions must match")
    best = np.argmax(scores, axis=1)
    batch = np.arange(masks.shape[0])
    return masks[batch, best], scores[batch, best]


def evaluate_sample_sam(
    predictor,
    sample,
    calibrated: dict,
    seed_namespace: str,
    trials: int,
) -> list[dict]:
    """Run SAM on all quality target × condition combinations for one sample."""
    rows = []
    predictor.set_image(sample.image)

    for target_str, scales in calibrated.items():
        point_scale = scales["point"]["scale"]
        box_scale = scales["box"]["scale"]

        for condition in ("point_noise", "box_noise"):
            for trial in range(trials):
                if condition == "point_noise":
                    perturbed = perturb_prompt(
                        sample.prompt,
                        sample.mask.shape,
                        point_scale=point_scale,
                        box_scale=0.0,
                        noise_source="point_noise",
                        trial=trial,
                        sample_id=sample.sample_id,
                        seed_namespace=seed_namespace + f"-target{target_str}",
                    )
                else:
                    perturbed = perturb_prompt(
                        sample.prompt,
                        sample.mask.shape,
                        point_scale=0.0,
                        box_scale=box_scale,
                        noise_source="box_noise",
                        trial=trial,
                        sample_id=sample.sample_id,
                        seed_namespace=seed_namespace + f"-target{target_str}",
                    )

                ph = point_hits_target(perturbed.point, sample.mask)
                bi = bbox_iou(sample.prompt.bbox, perturbed.bbox)

                try:
                    pred, score = predict_sam(predictor, perturbed, prompt_mode="point_box")
                    rows.append({
                    "sample_id": sample.sample_id,
                    "class_name": sample.prompt.class_name,
                    "quality_target": target_str,
                    "condition": condition,
                    "point_scale": point_scale,
                    "box_scale": box_scale,
                    "trial": trial,
                    "point_hit": str(ph).lower(),
                    "box_iou": f"{bi:.6f}",
                    "iou": f"{iou(pred, sample.mask):.6f}",
                    "dice": f"{dice(pred, sample.mask):.6f}",
                    "sam_score": f"{score:.6f}",
                    })
                except Exception as exc:
                    rows.append({
                    "sample_id": sample.sample_id,
                    "class_name": sample.prompt.class_name,
                    "quality_target": target_str,
                    "condition": condition,
                    "point_scale": point_scale,
                    "box_scale": box_scale,
                    "trial": trial,
                    "point_hit": str(ph).lower(),
                    "box_iou": f"{bi:.6f}",
                    "iou": "",
                    "dice": "",
                    "sam_score": "",
                    "error": f"sam_failure:{type(exc).__name__}",
                    })

    return rows


def evaluate_sample_sam_batched(
    predictor,
    sample,
    calibrated: dict,
    seed_namespace: str,
    trials: int,
    batch_size: int,
) -> list[dict]:
    """Evaluate all prompt trials with SAM's public batched torch API."""
    import torch

    predictor.set_image(sample.image)
    cases: list[tuple[dict, object]] = []
    for target_str, scales in calibrated.items():
        point_scale = scales["point"]["scale"]
        box_scale = scales["box"]["scale"]
        for condition in ("point_noise", "box_noise"):
            for trial in range(trials):
                perturbed = perturb_prompt(
                    sample.prompt,
                    sample.mask.shape,
                    point_scale=point_scale if condition == "point_noise" else 0.0,
                    box_scale=box_scale if condition == "box_noise" else 0.0,
                    noise_source=condition,
                    trial=trial,
                    sample_id=sample.sample_id,
                    seed_namespace=seed_namespace + f"-target{target_str}",
                )
                cases.append((
                    {
                        "sample_id": sample.sample_id,
                        "class_name": sample.prompt.class_name,
                        "quality_target": target_str,
                        "condition": condition,
                        "point_scale": point_scale if condition == "point_noise" else 0.0,
                        "box_scale": box_scale if condition == "box_noise" else 0.0,
                        "trial": trial,
                        "point_hit": str(
                            point_hits_target(perturbed.point, sample.mask)
                        ).lower(),
                        "box_iou": f"{bbox_iou(sample.prompt.bbox, perturbed.bbox):.6f}",
                    },
                    perturbed,
                ))

    rows: list[dict] = []
    image_shape = sample.image.shape[:2]
    for start in range(0, len(cases), batch_size):
        chunk = cases[start:start + batch_size]
        prompts = [case[1] for case in chunk]
        point_coords = np.asarray([[prompt.point] for prompt in prompts], dtype=np.float32)
        boxes = np.asarray([prompt.bbox for prompt in prompts], dtype=np.float32)
        point_coords = predictor.transform.apply_coords(point_coords, image_shape)
        boxes = predictor.transform.apply_boxes(boxes, image_shape)
        point_coords_t = torch.as_tensor(
            point_coords, dtype=torch.float32, device=predictor.device
        )
        point_labels_t = torch.ones(
            (len(chunk), 1), dtype=torch.int64, device=predictor.device
        )
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32, device=predictor.device)
        with torch.inference_mode():
            masks, scores, _ = predictor.predict_torch(
                point_coords=point_coords_t,
                point_labels=point_labels_t,
                boxes=boxes_t,
                mask_input=None,
                multimask_output=True,
                return_logits=False,
            )
        masks_np = masks.detach().cpu().numpy().astype(bool)
        scores_np = scores.detach().cpu().numpy()
        selected_masks, selected_scores = _select_best_masks(masks_np, scores_np)
        for (metadata, _), prediction, score in zip(
            chunk, selected_masks, selected_scores
        ):
            rows.append({
                **metadata,
                "iou": f"{iou(prediction, sample.mask):.6f}",
                "dice": f"{dice(prediction, sample.mask):.6f}",
                "sam_score": f"{float(score):.6f}",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM evaluation for multi-quality sensitivity curve"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/voc_validation"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/secondary/prompt_quality_sensitivity"),
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("protocol/sensitivity_protocol.json"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/sam_vit_b_01ec64.pth"),
    )
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="SAM mask-decoder prompt batch size (real evaluation only)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Explicitly generate smoke-test data without SAM; never treated as evidence",
    )
    parser.add_argument("--max-samples", type=int, default=None)
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
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    # Load calibrated protocol
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    calibrated = protocol.get("calibrated_scales")
    if not calibrated:
        raise SystemExit(
            "Protocol has no calibrated scales. Run calibrate_sensitivity.py first."
        )
    seed_namespace = protocol["calibration"]["seed_namespace"]

    trials = int(protocol.get("validation", {}).get("trials_per_condition", 20))
    if trials < 1:
        raise SystemExit("Protocol validation.trials_per_condition must be at least 1")

    # Check runtime capability
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except (ImportError, Exception):
        pass

    if not args.synthetic and args.device.startswith("cuda") and not cuda_available:
        raise SystemExit("CUDA was requested but is unavailable; use --device cpu or --synthetic")

    checkpoint_available = args.checkpoint.exists()
    segment_anything_available = False
    try:
        import segment_anything  # noqa: F401
        segment_anything_available = True
    except ImportError:
        pass
    if not args.synthetic and not checkpoint_available:
        raise SystemExit(f"SAM checkpoint not found: {args.checkpoint}")
    if not args.synthetic and not segment_anything_available:
        raise SystemExit("segment_anything is not installed; use the pinned SAM environment")

    # Dataset preparation
    sample_dirs = [
        path for path in sorted(args.data_dir.iterdir())
        if path.is_dir()
        and (path / "image.jpg").is_file()
        and (path / "target_mask.png").is_file()
        and (path / "prompt.txt").is_file()
    ]
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]

    expected_ids = manifest_sample_ids(args.manifest)
    observed_ids = [path.name for path in sample_dirs]
    dataset_spec = json.loads(args.dataset_fingerprints.read_text(encoding="utf-8"))
    expected_dataset_sha256 = dataset_spec["confirmatory"]["dataset_sha256"]
    initial_dataset_sha256 = dataset_fingerprint(sample_dirs)

    initial_sources = verify_runtime_sources(ROOT, args.runtime_sources)
    initial_commit = git_commit(ROOT)
    initial_dirty = git_is_dirty(ROOT)

    # Build run config fingerprint
    run_config = {
        "schema_version": 1,
        "experiment": "prompt_quality_sensitivity",
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_sha256": sha256_file(args.manifest),
        "dataset_sha256": initial_dataset_sha256,
        "expected_dataset_sha256": expected_dataset_sha256,
        "dataset_fingerprints_sha256": sha256_file(args.dataset_fingerprints),
        "runtime_sources_sha256": initial_sources["specification_sha256"],
        "source_tree_sha256": initial_sources["fingerprint"],
        "source_tree_matches": initial_sources["matches"],
        "git_commit": initial_commit,
        "git_dirty": initial_dirty,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_type": args.model_type,
        "device": args.device,
        "synthetic": args.synthetic,
        "trials_per_condition": trials,
        "batch_size": args.batch_size,
        "sample_ids": observed_ids,
        "calibrated_scales": calibrated,
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(run_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "sample_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    config_path = args.output_dir / "run_config.json"
    existing_checkpoints = sorted(checkpoint_dir.glob("*.json"))
    if config_path.exists() and not args.no_resume:
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise SystemExit(
                "Output directory belongs to a different run configuration."
            )
    elif existing_checkpoints and not args.no_resume:
        raise SystemExit("Checkpoint files exist without a matching run_config.json")
    else:
        config_path.write_text(json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8")

    # === Evaluation ===
    all_rows: list[dict] = []
    completed_before = 0
    is_synthetic = args.synthetic
    run_label = "MOCK/SYNTHETIC" if is_synthetic else "REAL"

    if is_synthetic:
        print("=== MOCK/SYNTHETIC SENSITIVITY EVALUATION (no real SAM inference) ===")
        # Generate synthetic evaluation rows with mock IoU values
        rng = np.random.default_rng(42)
        for i, sample_dir in enumerate(sample_dirs):
            sample = load_sample(sample_dir)
            for target_str in calibrated:
                for condition in ("point_noise", "box_noise"):
                    scales = calibrated[target_str]
                    ps = scales["point"]["scale"] if condition == "point_noise" else 0.0
                    bs = scales["box"]["scale"] if condition == "box_noise" else 0.0
                    for trial in range(trials):
                        base_iou = 0.85 if condition == "point_noise" else 0.80
                        mock_iou = base_iou - 0.05 * list(calibrated.keys()).index(target_str)
                        mock_iou = float(np.clip(mock_iou + rng.normal(0, 0.02), 0.0, 1.0))
                        all_rows.append({
                        "sample_id": sample.sample_id,
                        "class_name": sample.prompt.class_name,
                        "quality_target": target_str,
                        "condition": condition,
                        "point_scale": ps,
                        "box_scale": bs,
                        "trial": trial,
                        "point_hit": "true",
                        "box_iou": f"{1.0 - bs:.6f}",
                        "iou": f"{mock_iou:.6f}",
                        "dice": f"{(2*mock_iou)/(1+mock_iou):.6f}",
                        "sam_score": f"{0.95:.6f}",
                        "synthetic": True,
                        })
            if (i + 1) % 100 == 0 or i + 1 == len(sample_dirs):
                print(f"  Mock-evaluated {i + 1}/{len(sample_dirs)} samples", flush=True)

    else:
        # Real SAM evaluation
        print("=== REAL SAM EVALUATION ===")
        from segment_anything import SamPredictor, sam_model_registry

        observed_cp_sha256 = sha256_file(args.checkpoint)
        if observed_cp_sha256 != CHECKPOINT_SHA256:
            raise SystemExit("SAM checkpoint SHA-256 does not match the frozen protocol")

        sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
        sam.to(device=args.device)
        predictor = SamPredictor(sam)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        t_start = perf_counter()
        for i, sample_dir in enumerate(sample_dirs):
            cp_path = checkpoint_dir / f"{sample_dir.name}.json"
            if cp_path.exists() and not args.no_resume:
                cp = json.loads(cp_path.read_text(encoding="utf-8"))
                if cp.get("run_fingerprint") != run_fingerprint:
                    raise RuntimeError(f"Checkpoint fingerprint mismatch for {sample_dir.name}")
                all_rows.extend(cp.get("rows", []))
                completed_before += 1
                continue

            sample = load_sample(sample_dir)
            sample_rows = evaluate_sample_sam_batched(
                predictor,
                sample,
                calibrated,
                seed_namespace,
                trials,
                args.batch_size,
            )
            all_rows.extend(sample_rows)

            cp_payload = {
                "run_fingerprint": run_fingerprint,
                "sample_id": sample.sample_id,
                "rows": sample_rows,
            }
            atomic_write_text(cp_path, json.dumps(cp_payload) + "\n")

            if (i + 1) % 50 == 0 or i + 1 == len(sample_dirs):
                print(f"  Evaluated {i + 1}/{len(sample_dirs)} samples", flush=True)

        elapsed = perf_counter() - t_start
        print(f"  Completed in {elapsed:.1f} s")

    # === Write outputs ===
    metrics_path = args.output_dir / "metrics.csv"
    write_csv(metrics_path, all_rows)
    metrics_sha256 = sha256_file(metrics_path)

    expected_row_count = len(sample_dirs) * len(calibrated) * 2 * trials
    failure_rows = sum(1 for row in all_rows if row.get("error"))
    is_complete = (
        args.max_samples is None
        and observed_ids == expected_ids
        and not is_synthetic
        and initial_dataset_sha256 == expected_dataset_sha256
        and len(all_rows) == expected_row_count
        and failure_rows == 0
    )

    final_dataset_sha256 = dataset_fingerprint(sample_dirs)
    final_commit = git_commit(ROOT)
    final_dirty = git_is_dirty(ROOT)

    summary = {
        "status": "secondary",
        "experiment": "prompt_quality_sensitivity",
        "is_synthetic": is_synthetic,
        "evaluation_complete": is_complete,
        "num_samples": len(sample_dirs),
        "num_rows": len(all_rows),
        "expected_rows": expected_row_count,
        "failure_rows": failure_rows,
        "trials_per_condition": trials,
        "quality_targets": list(calibrated.keys()),
        "conditions": ["point_noise", "box_noise"],
        "protocol_sha256": sha256_file(args.protocol),
        "metrics_sha256": metrics_sha256,
        "run_fingerprint": run_fingerprint,
        "completed_before_resume": completed_before,
        "dataset_sha256": final_dataset_sha256,
        "git_commit": final_commit,
        "git_dirty": final_dirty,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_available": cuda_available,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    # Write README
    (args.output_dir / "README.md").write_text(
        f"# Multi-Quality Point-vs-Box Sensitivity Curve\n\n"
        f"**Status**: secondary (not a confirmatory H1–H3 result)\n\n"
        f"This is a robustness/sensitivity analysis. Numeric observable matching "
        f"(point hit rate vs box IoU) does NOT imply human-perceptual equivalence.\n\n"
        f"- Samples: {len(sample_dirs)}\n"
        f"- Quality targets: {list(calibrated.keys())}\n"
        f"- Conditions: point_noise, box_noise\n"
        f"- Synthetic: {is_synthetic}\n"
        f"- Trials per sample/target/condition: {trials}\n"
        f"- Evaluation complete: {is_complete}\n\n"
        f"## Regenerate\n\n"
        f"```bash\n"
        f"python scripts/calibrate_sensitivity.py\n"
        f"python scripts/run_sensitivity_sam.py\n"
        f"python scripts/analyze_sensitivity.py\n"
        f"```\n",
        encoding="utf-8",
    )

    print()
    print(f"Output written to {args.output_dir}")
    print(f"  Samples: {len(sample_dirs)}")
    print(f"  Rows:    {len(all_rows)}")
    print(f"  Complete: {is_complete}")


if __name__ == "__main__":
    main()
