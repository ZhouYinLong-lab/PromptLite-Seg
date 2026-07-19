from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    "script",
    ["scripts/analyze_results.py", "scripts/analyze_statistics.py"],
)
def test_analysis_scripts_start(script: str) -> None:
    result = run_script(script, "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_one_sample_cpu_experiment(synthetic_data_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "cpu-output"
    result = run_script(
        "scripts/run_experiment.py",
        "--data-dir",
        str(synthetic_data_dir),
        "--output-dir",
        str(output_dir),
        "--max-samples",
        "1",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_samples"] == 1
    assert set(summary) == {"num_samples", "center_color", "robust_superpixel"}
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "figures" / "sample_000.png").is_file()


def test_confirmatory_cpu_smoke_has_metrics_without_images(synthetic_data_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "confirmatory-output"
    result = run_script(
        "scripts/run_confirmatory_cpu.py",
        "--data-dir",
        str(synthetic_data_dir),
        "--output-dir",
        str(output_dir),
        "--max-samples",
        "1",
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["confirmatory"] is False
    assert len(summary["summaries"]) == 6
    assert all(item["num_success"] == 1 for item in summary["summaries"])
    assert all(item["num_failures"] == 0 for item in summary["summaries"])
    assert not list(output_dir.rglob("*.png"))
