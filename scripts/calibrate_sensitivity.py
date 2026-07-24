"""Calibrate point and box perturbation scales for multi-quality sensitivity curve.

Calibrates independently per quality target using the frozen VOC tuning split.
Output is written into the secondary sensitivity protocol file, which is then
frozen before any validation results are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.dataset import iter_samples
from promptseg.prompts import bbox_iou, perturb_prompt, point_hits_target


QUALITY_TARGETS = [0.90, 0.80, 0.70, 0.60, 0.50]
DEFAULT_TRIALS = 20
DEFAULT_GRID_STEP = 0.0025


def measure_point_quality(
    samples, point_scale: float, trials: int,
    seed_namespace: str = "promptlite-seg-sensitivity-calibration-v1",
) -> float:
    """Aggregate target-mask hit rate for a given point perturbation scale."""
    hits: list[bool] = []
    for sample in samples:
        for trial in range(trials):
            noisy = perturb_prompt(
                sample.prompt,
                sample.mask.shape,
                point_scale=point_scale,
                box_scale=0.0,
                noise_source="point_noise",
                trial=trial,
                sample_id=sample.sample_id,
                seed_namespace=seed_namespace,
            )
            hits.append(point_hits_target(noisy.point, sample.mask))
    return float(np.mean(hits))


def measure_box_quality(
    samples, box_scale: float, trials: int,
    seed_namespace: str = "promptlite-seg-sensitivity-calibration-v1",
) -> float:
    """Mean IoU with clean tight box for a given box perturbation scale."""
    values: list[float] = []
    for sample in samples:
        for trial in range(trials):
            noisy = perturb_prompt(
                sample.prompt,
                sample.mask.shape,
                point_scale=0.0,
                box_scale=box_scale,
                noise_source="box_noise",
                trial=trial,
                sample_id=sample.sample_id,
                seed_namespace=seed_namespace,
            )
            values.append(bbox_iou(sample.prompt.bbox, noisy.bbox))
    return float(np.mean(values))


def choose_scale(
    samples,
    target_quality: float,
    trials: int,
    kind: str,
    grid: np.ndarray,
    seed_namespace: str,
) -> dict[str, Any]:
    """Find the scale that best matches the target quality."""
    measure = measure_point_quality if kind == "point" else measure_box_quality
    best_scale = 0.0
    best_error = float("inf")
    best_observed = 0.0

    for scale in grid:
        observed = measure(samples, float(scale), trials, seed_namespace=seed_namespace)
        error = abs(observed - target_quality)
        if error < best_error:
            best_error = error
            best_scale = float(scale)
            best_observed = observed

    return {
        "scale": best_scale,
        "target_quality": target_quality,
        "observed_quality": round(best_observed, 6),
        "absolute_error": round(best_error, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate sensitivity curve perturbation scales"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/voc_tuning"),
        help="VOC tuning split directory",
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("protocol/sensitivity_protocol.json"),
        help="Secondary sensitivity protocol file",
    )
    parser.add_argument(
        "--grid-step", type=float, default=DEFAULT_GRID_STEP,
        help="Grid search step size",
    )
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help="Deterministic trials per target",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if not 0 < args.grid_step <= 0.5:
        parser.error("--grid-step must be in (0, 0.5]")

    # Load protocol
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))

    # Load tuning samples
    samples = list(iter_samples(args.data_dir))
    expected = protocol["calibration"]["samples"]
    if len(samples) != expected:
        raise SystemExit(
            f"Expected {expected} tuning samples in {args.data_dir}, found {len(samples)}"
        )

    seed_namespace = protocol["calibration"]["seed_namespace"]
    trials = args.trials
    targets = protocol["calibration"]["quality_targets"]

    # Build point grid and box grid separately
    point_grid = np.arange(0.0, 0.5000001, args.grid_step)
    box_grid = np.arange(0.0, 0.5000001, args.grid_step)

    print(f"Calibrating on {len(samples)} tuning samples × {trials} trials each")
    print(f"Quality targets: {targets}")
    print()

    calibrated = {}
    for target in targets:
        print(f"Target quality = {target:.2f}")
        point_result = choose_scale(
            samples, target, trials, "point", point_grid, seed_namespace,
        )
        box_result = choose_scale(
            samples, target, trials, "box", box_grid, seed_namespace,
        )
        calibrated[str(target)] = {
            "point": point_result,
            "box": box_result,
        }
        print(f"  point: scale={point_result['scale']:.4f}, "
              f"observed={point_result['observed_quality']:.4f}, "
              f"error={point_result['absolute_error']:.4f}")
        print(f"  box:   scale={box_result['scale']:.4f}, "
              f"observed={box_result['observed_quality']:.4f}, "
              f"error={box_result['absolute_error']:.4f}")
        print()

    # Write calibrated scales into the protocol
    protocol["calibrated_scales"] = calibrated
    protocol["calibration_meta"] = {
        "num_tuning_samples": len(samples),
        "trials_per_target": trials,
        "grid_step": args.grid_step,
        "seed_namespace": seed_namespace,
    }

    protocol["status"] = "secondary_sensitivity_frozen"
    protocol["frozen_at"] = datetime.now(timezone.utc).isoformat()
    protocol.pop("_sha256", None)
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    protocol["_sha256"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding _sha256",
        "value": protocol_sha256,
    }

    args.protocol.write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Protocol written to {args.protocol}")
    print(f"SHA-256: {protocol_sha256}")


if __name__ == "__main__":
    main()
