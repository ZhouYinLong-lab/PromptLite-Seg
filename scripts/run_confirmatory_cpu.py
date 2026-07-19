"""Run frozen CPU baselines and ablations without emitting copyrighted images."""

from __future__ import annotations

import argparse
import json
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
from promptseg.dataset import iter_samples
from promptseg.metrics import dice, iou
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
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

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


def run_method(method_name: str, data_dir: Path, max_samples: int | None) -> tuple[list[dict], dict]:
    method = CONFIRMATORY_CPU_METHODS[method_name]
    rows: list[dict] = []
    latencies: list[float] = []
    start = perf_counter()
    with PeakRssMonitor() as memory:
        for index, sample in enumerate(iter_samples(data_dir)):
            if max_samples is not None and index >= max_samples:
                break
            call_start = perf_counter()
            try:
                prediction = method(sample.image, sample.prompt)
            except Exception as error:  # Preserve failures in the artifact instead of dropping them.
                latency_ms = (perf_counter() - call_start) * 1000
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "class_name": sample.prompt.class_name,
                        "method": method_name,
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                        "iou": "",
                        "dice": "",
                        "latency_ms": f"{latency_ms:.6f}",
                    }
                )
                continue
            latency_ms = (perf_counter() - call_start) * 1000
            latencies.append(latency_ms)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "class_name": sample.prompt.class_name,
                    "method": method_name,
                    "status": "ok",
                    "error": "",
                    "iou": f"{iou(prediction, sample.mask):.6f}",
                    "dice": f"{dice(prediction, sample.mask):.6f}",
                    "latency_ms": f"{latency_ms:.6f}",
                }
            )
    elapsed = perf_counter() - start
    valid = [row for row in rows if row["status"] == "ok"]
    summary = {
        "method": method_name,
        "num_samples": len(rows),
        "num_success": len(valid),
        "num_failures": len(rows) - len(valid),
        "mean_iou": statistics.fmean(float(row["iou"]) for row in valid) if valid else None,
        "mean_dice": statistics.fmean(float(row["dice"]) for row in valid) if valid else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, 0.95) if latencies else None,
        "total_seconds": elapsed,
        "peak_rss_mb": memory.peak_bytes / (1024 * 1024),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/voc_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_confirmatory/cpu"))
    parser.add_argument("--methods", nargs="+", choices=tuple(CONFIRMATORY_CPU_METHODS), default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    methods = args.methods or list(CONFIRMATORY_CPU_METHODS)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for method_name in methods:
        rows, summary = run_method(method_name, args.data_dir, args.max_samples)
        all_rows.extend(rows)
        summaries.append(summary)
        write_csv(args.output_dir / "metrics.csv", all_rows)
        write_csv(args.output_dir / "summary.csv", summaries)
        print(json.dumps(summary, indent=2))

    sample_counts = {summary["num_samples"] for summary in summaries}
    payload = {
        "git_commit": git_commit(),
        "data_dir": str(args.data_dir),
        "confirmatory": args.max_samples is None and sample_counts == {1449},
        "expected_confirmatory_samples": 1449,
        "max_samples": args.max_samples,
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

