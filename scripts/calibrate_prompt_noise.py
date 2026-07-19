"""Calibrate point and box perturbations on the frozen VOC tuning split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.dataset import iter_samples
from promptseg.prompts import bbox_iou, perturb_prompt, point_hits_target


def measure_point_quality(samples, scale: float, trials: int) -> float:
    hits: list[bool] = []
    for sample in samples:
        for trial in range(trials):
            noisy = perturb_prompt(
                sample.prompt,
                sample.mask.shape,
                point_scale=scale,
                box_scale=0.0,
                noise_source="point_noise",
                trial=trial,
                sample_id=sample.sample_id,
            )
            hits.append(point_hits_target(noisy.point, sample.mask))
    return float(np.mean(hits))


def measure_box_quality(samples, scale: float, trials: int) -> float:
    values: list[float] = []
    for sample in samples:
        for trial in range(trials):
            noisy = perturb_prompt(
                sample.prompt,
                sample.mask.shape,
                point_scale=0.0,
                box_scale=scale,
                noise_source="box_noise",
                trial=trial,
                sample_id=sample.sample_id,
            )
            values.append(bbox_iou(sample.prompt.bbox, noisy.bbox))
    return float(np.mean(values))


def choose_scale(samples, target: float, trials: int, kind: str, grid: np.ndarray) -> dict:
    measure = measure_point_quality if kind == "point" else measure_box_quality
    candidates = [(float(scale), measure(samples, float(scale), trials)) for scale in grid]
    scale, observed = min(candidates, key=lambda item: (abs(item[1] - target), item[0]))
    return {
        "scale": scale,
        "target_quality": target,
        "observed_quality": observed,
        "absolute_error": abs(observed - target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/voc_tuning"))
    parser.add_argument("--protocol", type=Path, default=Path("protocol/research_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("protocol/noise_calibration.json"))
    parser.add_argument("--grid-step", type=float, default=0.0025)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    samples = list(iter_samples(args.data_dir))
    expected = protocol["splits"]["tuning"]["samples"]
    if len(samples) != expected:
        raise SystemExit(f"Expected {expected} tuning samples, found {len(samples)}")
    calibration = protocol["noise_calibration"]
    trials = int(calibration["trials_per_target"])
    grid = np.arange(0.0, 0.5000001, args.grid_step)

    results = {}
    for severity, target in calibration["quality_levels"].items():
        results[severity] = {
            "point": choose_scale(samples, float(target), trials, "point", grid),
            "box": choose_scale(samples, float(target), trials, "box", grid),
        }

    payload = {
        "protocol_version": protocol["protocol_version"],
        "tuning_manifest_sha256": protocol["splits"]["tuning"]["manifest_sha256"],
        "num_tuning_samples": len(samples),
        "trials_per_target": trials,
        "grid_step": args.grid_step,
        "seed_namespace": calibration["seed_namespace"],
        "quality_definition": {
            "point": calibration["point_quality"],
            "box": calibration["box_quality"],
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

