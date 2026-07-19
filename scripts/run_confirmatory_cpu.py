"""Run frozen CPU baselines and ablations without emitting copyrighted images."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.algorithms import CONFIRMATORY_CPU_METHODS
from promptseg.dataset import load_sample
from promptseg.metrics import dice, iou
from promptseg.protocol import dataset_fingerprint, git_is_dirty, manifest_sample_ids, sha256_file
from promptseg.utils import write_csv


class PeakRssMonitor:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self.stop_event = Event()
        self.process = psutil.Process()
        self.peak_bytes = self.process.memory_info().rss
        self.thread = Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            processes = [self.process, *self.process.children(recursive=True)]
            rss = 0
            for process in processes:
                try:
                    rss += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_bytes = max(self.peak_bytes, rss)

    def __enter__(self) -> "PeakRssMonitor":
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop_event.set()
        self.thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def run_sample_task(task: tuple[str, Path]) -> dict:
    method_name, sample_dir = task
    method = CONFIRMATORY_CPU_METHODS[method_name]
    sample = load_sample(sample_dir)
    call_start = perf_counter()
    try:
        prediction = method(sample.image, sample.prompt)
    except Exception as error:
        latency_ms = (perf_counter() - call_start) * 1000
        return {
            "sample_id": sample.sample_id,
            "class_name": sample.prompt.class_name,
            "method": method_name,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "iou": "",
            "dice": "",
            "latency_ms": f"{latency_ms:.6f}",
        }
    latency_ms = (perf_counter() - call_start) * 1000
    return {
        "sample_id": sample.sample_id,
        "class_name": sample.prompt.class_name,
        "method": method_name,
        "status": "ok",
        "error": "",
        "iou": f"{iou(prediction, sample.mask):.6f}",
        "dice": f"{dice(prediction, sample.mask):.6f}",
        "latency_ms": f"{latency_ms:.6f}",
    }


def sample_directories(data_dir: Path, max_samples: int | None) -> list[Path]:
    directories = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_dir()
        and (path / "image.jpg").is_file()
        and (path / "target_mask.png").is_file()
        and (path / "prompt.txt").is_file()
    ]
    return directories if max_samples is None else directories[:max_samples]


def run_method(
    method_name: str,
    data_dir: Path,
    max_samples: int | None,
    workers: int,
) -> tuple[list[dict], dict]:
    directories = sample_directories(data_dir, max_samples)
    tasks = [(method_name, sample_dir) for sample_dir in directories]
    start = perf_counter()
    with PeakRssMonitor() as memory:
        if workers == 1:
            rows = [run_sample_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(run_sample_task, tasks, chunksize=4))
    elapsed = perf_counter() - start
    valid = [row for row in rows if row["status"] == "ok"]
    latencies = [float(row["latency_ms"]) for row in valid]
    success_iou = [float(row["iou"]) for row in valid]
    success_dice = [float(row["dice"]) for row in valid]
    summary = {
        "method": method_name,
        "num_samples": len(rows),
        "num_success": len(valid),
        "num_failures": len(rows) - len(valid),
        "mean_iou_success_only": statistics.fmean(success_iou) if valid else None,
        "mean_dice_success_only": statistics.fmean(success_dice) if valid else None,
        "mean_iou_failure_zero": sum(success_iou) / len(rows) if rows else None,
        "mean_dice_failure_zero": sum(success_dice) / len(rows) if rows else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95) if latencies else None,
        "total_seconds": elapsed,
        "peak_rss_mb": memory.peak_bytes / (1024 * 1024),
        "workers": workers,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/voc_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_confirmatory/cpu"))
    parser.add_argument("--methods", nargs="+", choices=tuple(CONFIRMATORY_CPU_METHODS), default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--protocol", type=Path, default=Path("protocol/research_protocol.json"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("protocol/manifests/confirmatory_validation.jsonl"),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frozen_methods = [
        *protocol["methods"]["cpu_baselines"],
        protocol["methods"]["proposed"],
        *protocol["methods"]["ablations"],
    ]
    methods = args.methods or list(CONFIRMATORY_CPU_METHODS)
    directories = sample_directories(args.data_dir, args.max_samples)
    observed_ids = [path.name for path in directories]
    expected_ids = manifest_sample_ids(args.manifest)
    is_confirmatory = (
        args.max_samples is None
        and observed_ids == expected_ids
        and methods == frozen_methods
        and not git_is_dirty(ROOT)
    )
    all_rows: list[dict] = []
    summaries: list[dict] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method_name in methods:
        rows, summary = run_method(method_name, args.data_dir, args.max_samples, args.workers)
        all_rows.extend(rows)
        summaries.append(summary)
        write_csv(args.output_dir / "metrics.csv", all_rows)
        write_csv(args.output_dir / "summary.csv", summaries)
        print(json.dumps(summary, indent=2))

    payload = {
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(ROOT),
        "data_dir": str(args.data_dir),
        "dataset_sha256": dataset_fingerprint(directories),
        "manifest_sha256": sha256_file(args.manifest),
        "protocol_sha256": sha256_file(args.protocol),
        "confirmatory": is_confirmatory,
        "expected_confirmatory_samples": len(expected_ids),
        "max_samples": args.max_samples,
        "workers": args.workers,
        "methods": methods,
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "confirmatory": payload["confirmatory"]}, indent=2))


if __name__ == "__main__":
    main()
