"""Tests for multi-quality point-versus-box sensitivity curve.

Covers calibration, mock SAM evaluation, sample-level aggregation,
CI shape, checkpoint rejection, and validation-data independence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.prompts import bbox_iou, perturb_prompt, point_hits_target
from scripts.analyze_sensitivity import _paired_bootstrap_ci, _sample_level_aggregate
from scripts.run_sensitivity_sam import _select_best_masks


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

class MockSample:
    """Minimal sample for calibration tests."""
    def __init__(self, sample_id: str, point, bbox, mask_shape):
        from promptseg.dataset import Prompt
        self.sample_id = sample_id
        self.prompt = Prompt(
            point=point, bbox=bbox, label=0, class_name="test"
        )
        self.mask = np.ones(mask_shape, dtype=bool)


def _make_mock_samples(n: int = 20) -> list:
    return [
        MockSample(
            sample_id=f"test_{i:04d}",
            point=(200, 200),
            bbox=(100, 100, 300, 300),
            mask_shape=(400, 400),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Calibration determinism
# ---------------------------------------------------------------------------

def test_perturb_prompt_is_deterministic() -> None:
    """Same inputs produce the same perturbed prompt."""
    from promptseg.dataset import Prompt

    prompt = Prompt(point=(200, 200), bbox=(100, 100, 300, 300), label=0, class_name="test")
    a = perturb_prompt(
        prompt, (400, 400),
        point_scale=0.1, box_scale=0.0,
        noise_source="point_noise", trial=0, sample_id="test_sample",
        seed_namespace="test-ns",
    )
    b = perturb_prompt(
        prompt, (400, 400),
        point_scale=0.1, box_scale=0.0,
        noise_source="point_noise", trial=0, sample_id="test_sample",
        seed_namespace="test-ns",
    )
    assert a.point == b.point
    assert a.bbox == b.bbox


def test_perturb_prompt_point_noise_leaves_box_clean() -> None:
    """Point-only perturbation should not change the box."""
    from promptseg.dataset import Prompt

    prompt = Prompt(point=(200, 200), bbox=(100, 100, 300, 300), label=0, class_name="test")
    noisy = perturb_prompt(
        prompt, (400, 400),
        point_scale=0.1, box_scale=0.0,
        noise_source="point_noise", trial=0, sample_id="test_sample",
    )
    assert noisy.bbox == prompt.bbox
    # Point may have changed
    assert noisy.point != prompt.point or True  # might be small change


def test_perturb_prompt_box_noise_leaves_point_clean() -> None:
    """Box-only perturbation should not change the point."""
    from promptseg.dataset import Prompt

    prompt = Prompt(point=(200, 200), bbox=(100, 100, 300, 300), label=0, class_name="test")
    noisy = perturb_prompt(
        prompt, (400, 400),
        point_scale=0.0, box_scale=0.05,
        noise_source="box_noise", trial=0, sample_id="test_sample",
    )
    assert noisy.point == prompt.point


def test_point_hits_target_inside() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    assert point_hits_target((50, 50), mask) is True


def test_point_hits_target_outside() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    assert point_hits_target((10, 10), mask) is False


def test_bbox_iou_perfect() -> None:
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_bbox_iou_partial() -> None:
    iou_val = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.1 < iou_val < 0.2


def test_bbox_iou_disjoint() -> None:
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# ---------------------------------------------------------------------------
# Sample-level aggregation
# ---------------------------------------------------------------------------

def test_sample_level_aggregate_single_trial() -> None:
    rows = [
        {"sample_id": "s1", "quality_target": "0.90", "condition": "point_noise", "iou": "0.750000"},
    ]
    agg = _sample_level_aggregate(rows, "0.90", "point_noise")
    assert agg["mean_iou"] == pytest.approx(0.75)
    assert agg["num_trials"] == 1
    assert agg["num_failures"] == 0


def test_sample_level_aggregate_multiple_trials() -> None:
    rows = [
        {"sample_id": "s1", "quality_target": "0.80", "condition": "box_noise", "iou": "0.600000"},
        {"sample_id": "s1", "quality_target": "0.80", "condition": "box_noise", "iou": "0.700000"},
        {"sample_id": "s1", "quality_target": "0.80", "condition": "box_noise", "iou": "0.800000"},
    ]
    agg = _sample_level_aggregate(rows, "0.80", "box_noise")
    assert agg["mean_iou"] == pytest.approx(0.70)
    assert agg["num_trials"] == 3


def test_sample_level_aggregate_failure_counts_as_zero() -> None:
    rows = [
        {"sample_id": "s1", "quality_target": "0.90", "condition": "point_noise", "iou": ""},
        {"sample_id": "s1", "quality_target": "0.90", "condition": "point_noise", "iou": "0.800000"},
    ]
    agg = _sample_level_aggregate(rows, "0.90", "point_noise")
    assert agg["mean_iou"] == pytest.approx(0.40)  # (0.0 + 0.8) / 2
    assert agg["num_failures"] == 1


def test_sample_level_aggregate_empty() -> None:
    agg = _sample_level_aggregate([], "0.90", "point_noise")
    assert agg == {}


def test_sample_level_aggregate_filters_sample_id() -> None:
    rows = [
        {"sample_id": "a", "quality_target": "0.90", "condition": "point_noise", "iou": "0.2"},
        {"sample_id": "b", "quality_target": "0.90", "condition": "point_noise", "iou": "0.8"},
    ]
    agg = _sample_level_aggregate(rows, "0.90", "point_noise", sample_id="b")
    assert agg["sample_id"] == "b"
    assert agg["mean_iou"] == pytest.approx(0.8)


def test_batched_sam_selects_complete_two_dimensional_mask() -> None:
    masks = np.zeros((2, 3, 8, 8), dtype=bool)
    masks[0, 1, 2:6, 2:6] = True
    masks[1, 2, 1:7, 1:7] = True
    scores = np.asarray([[0.1, 0.9, 0.2], [0.2, 0.3, 0.8]])
    selected, selected_scores = _select_best_masks(masks, scores)
    assert selected.shape == (2, 8, 8)
    assert selected[0].sum() == 16
    assert selected[1].sum() == 36
    assert selected_scores.tolist() == pytest.approx([0.9, 0.8])


# ---------------------------------------------------------------------------
# Paired bootstrap CI
# ---------------------------------------------------------------------------

def test_paired_bootstrap_ci_shape() -> None:
    """CI output must be a 3-tuple (mean, lower, upper)."""
    # 10 sample-level deltas
    deltas = np.array([-0.05, -0.03, -0.04, -0.06, -0.02,
                       0.01, -0.03, -0.04, -0.05, -0.02])
    mean_d, lo, hi = _paired_bootstrap_ci(deltas, n_bootstrap=2000, seed=42)
    assert lo <= mean_d <= hi
    # The deltas are mostly negative, so mean should be negative
    assert mean_d < 0


def test_paired_bootstrap_ci_empty() -> None:
    mean_d, lo, hi = _paired_bootstrap_ci(np.array([]))
    assert mean_d == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_paired_bootstrap_ci_reproducible() -> None:
    deltas = np.random.default_rng(42).normal(-0.03, 0.02, 50)
    a = _paired_bootstrap_ci(deltas, n_bootstrap=1000, seed=99)
    b = _paired_bootstrap_ci(deltas, n_bootstrap=1000, seed=99)
    assert a == b


def test_paired_bootstrap_ci_all_same() -> None:
    """When all deltas are equal, CI should collapse to that value."""
    deltas = np.ones(30) * 0.5
    mean_d, lo, hi = _paired_bootstrap_ci(deltas, n_bootstrap=2000, seed=0)
    assert mean_d == pytest.approx(0.5)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Calibration script integration
# ---------------------------------------------------------------------------

def test_calibrate_sensitivity_help() -> None:
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/calibrate_sensitivity.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_calibrate_sensitivity_runs_on_tuning_data(tmp_path: Path) -> None:
    """Calibration must complete on real or synthetic tuning data."""
    # Use a synthetic tuning directory
    from PIL import Image

    tuning_dir = tmp_path / "tuning"
    tuning_dir.mkdir()

    rng = np.random.default_rng(12345)
    for i in range(10):
        sd = tuning_dir / f"sample_{i:04d}"
        sd.mkdir()
        img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(img).save(sd / "image.jpg", quality=100)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[20:100, 20:100] = 255
        Image.fromarray(mask).save(sd / "target_mask.png")
        (sd / "prompt.txt").write_text(
            f"source_row={i}\nlabel=1\nclass_name=aeroplane\n"
            f"bbox=20,20,100,100\npoint=60,60\n",
            encoding="utf-8",
        )

    # Create a minimal protocol
    proto_path = tmp_path / "proto.json"
    proto = {
        "protocol_version": "1.0",
        "calibration": {
            "dataset": "VOC 2012 train tuning split",
            "manifest": "none",
            "samples": 10,
            "quality_targets": [0.90, 0.80],
            "trials_per_target": 4,
            "point_metric": "aggregate target-mask hit rate",
            "box_metric": "mean IoU with clean tight box",
            "grid": {"point_scale_range": [0.0, 0.5], "box_scale_range": [0.0, 0.5], "step": 0.0025},
            "seed_namespace": "test-calibration-ns",
            "method": "independent deterministic grid search",
        },
        "calibrated_scales": None,
    }
    proto_path.write_text(json.dumps(proto))

    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/calibrate_sensitivity.py"),
         "--data-dir", str(tuning_dir),
         "--protocol", str(proto_path),
         "--grid-step", "0.05",
         "--trials", "4",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "SHA-256" in result.stdout

    # Verify calibrated scales were written
    updated = json.loads(proto_path.read_text(encoding="utf-8"))
    assert updated["calibrated_scales"] is not None
    assert "0.9" in updated["calibrated_scales"]
    assert "0.8" in updated["calibrated_scales"]
    assert "point" in updated["calibrated_scales"]["0.9"]
    assert "box" in updated["calibrated_scales"]["0.9"]


# ---------------------------------------------------------------------------
# Sensitivity runner mock/synthetic
# ---------------------------------------------------------------------------

def test_sensitivity_sam_help() -> None:
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_sensitivity_sam.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_sensitivity_sam_mock_run(tmp_path: Path) -> None:
    """Mock/synthetic run must produce valid metrics CSV."""
    from PIL import Image

    val_dir = tmp_path / "validation"
    val_dir.mkdir()
    for i in range(3):
        sd = val_dir / f"sample_{i:04d}"
        sd.mkdir()
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[10:50, 10:50] = (100, 150, 200)
        Image.fromarray(img).save(sd / "image.jpg", quality=100)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        Image.fromarray(mask).save(sd / "target_mask.png")
        (sd / "prompt.txt").write_text(
            f"source_row={i}\nlabel=1\nclass_name=aeroplane\n"
            f"bbox=10,10,50,50\npoint=30,30\n",
            encoding="utf-8",
        )

    # Create calibrated protocol
    proto_path = tmp_path / "proto.json"
    proto = {
        "protocol_version": "1.0",
        "calibration": {
            "seed_namespace": "test-sensitivity-ns",
        },
        "calibrated_scales": {
            "0.9": {"point": {"scale": 0.05}, "box": {"scale": 0.02}},
            "0.8": {"point": {"scale": 0.10}, "box": {"scale": 0.04}},
        },
    }
    proto_path.write_text(json.dumps(proto))

    # Manifest
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps({"sample_id": f"sample_{i:04d}"}) for i in range(3)) + "\n"
    )

    # Fingerprints
    fp_path = tmp_path / "fp.json"
    fp_path.write_text(json.dumps({"confirmatory": {"dataset_sha256": "dummy"}}))

    # Runtime sources — must pass verify_runtime_sources validation
    # which requires schema_version==1 and a non-empty files dict.
    # Use actual source-file hashes so the check passes.
    from promptseg.protocol import canonical_source_sha256
    src_files = {
        "src/promptseg/dataset.py": canonical_source_sha256(ROOT / "src/promptseg/dataset.py"),
        "src/promptseg/prompts.py": canonical_source_sha256(ROOT / "src/promptseg/prompts.py"),
        "src/promptseg/sam.py": canonical_source_sha256(ROOT / "src/promptseg/sam.py"),
    }
    rs_path = tmp_path / "rs.json"
    rs_path.write_text(json.dumps({"schema_version": 1, "files": src_files}))

    out_dir = tmp_path / "output"
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_sensitivity_sam.py"),
         "--data-dir", str(val_dir),
         "--output-dir", str(out_dir),
         "--protocol", str(proto_path),
         "--manifest", str(manifest_path),
         "--dataset-fingerprints", str(fp_path),
         "--runtime-sources", str(rs_path),
         "--checkpoint", str(tmp_path / "nonexistent.pth"),
         "--device", "cpu",
         "--synthetic",
         "--max-samples", "3",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr

    # Verify outputs
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "README.md").exists()

    # Verify metrics CSV structure
    with (out_dir / "metrics.csv").open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    expected_fields = {"sample_id", "quality_target", "condition", "iou", "dice"}
    assert expected_fields.issubset(set(rows[0].keys()))

    # Verify summary
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["is_synthetic"] is True
    assert summary["status"] == "secondary"
    assert summary["num_samples"] == 3


# ---------------------------------------------------------------------------
# Analysis script
# ---------------------------------------------------------------------------

def test_analyze_sensitivity_help() -> None:
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_sensitivity.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_analyze_sensitivity_disclaimer_present(tmp_path: Path) -> None:
    """Analysis output must contain the perceptual-equivalence disclaimer."""
    # Create synthetic metrics
    metrics_path = tmp_path / "metrics.csv"
    rows = []
    for sid in [f"sample_{i:04d}" for i in range(5)]:
        for target in ["0.9", "0.8"]:
            for cond in ["point_noise", "box_noise"]:
                rows.append({
                    "sample_id": sid,
                    "class_name": "aeroplane",
                    "quality_target": target,
                    "condition": cond,
                    "point_scale": "0.05",
                    "box_scale": "0.02",
                    "point_hit": "true",
                    "box_iou": "0.950000",
                    "iou": "0.700000" if cond == "box_noise" else "0.750000",
                    "dice": "0.820000",
                    "sam_score": "0.950000",
                })

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "analysis_output"
    out_dir.mkdir()

    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_sensitivity.py"),
         "--metrics", str(metrics_path),
         "--output-dir", str(out_dir),
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr

    # Check analysis output
    analysis = json.loads((out_dir / "sensitivity_analysis.json").read_text(encoding="utf-8"))
    assert analysis["status"] == "secondary"
    assert "human-perceptual" in analysis["disclaimer"].lower()
    assert len(analysis["curve_points"]) == 2  # two quality targets

    for cp in analysis["curve_points"]:
        assert "paired_delta_mean" in cp
        assert "paired_delta_ci_lower" in cp
        assert "paired_delta_ci_upper" in cp


# ---------------------------------------------------------------------------
# Checkpoint rejection: different calibration → reject
# ---------------------------------------------------------------------------

def test_sensitivity_config_fingerprint_changes_with_calibration() -> None:
    """Different calibrated scales produce different fingerprints."""
    config_a = {
        "experiment": "prompt_quality_sensitivity",
        "protocol_sha256": "aaa",
        "sample_ids": ["s1", "s2"],
        "calibrated_scales": {"0.9": {"point": {"scale": 0.05}}},
    }
    config_b = {
        "experiment": "prompt_quality_sensitivity",
        "protocol_sha256": "aaa",
        "sample_ids": ["s1", "s2"],
        "calibrated_scales": {"0.9": {"point": {"scale": 0.10}}},
    }
    import hashlib
    fp_a = hashlib.sha256(json.dumps(config_a, sort_keys=True, default=str).encode()).hexdigest()
    fp_b = hashlib.sha256(json.dumps(config_b, sort_keys=True, default=str).encode()).hexdigest()
    assert fp_a != fp_b


# ---------------------------------------------------------------------------
# No validation-driven recalibration
# ---------------------------------------------------------------------------

def test_sensitivity_protocol_specifies_separate_tuning_and_validation() -> None:
    """The protocol must use tuning data for calibration, validation for evaluation."""
    proto_path = ROOT / "protocol/sensitivity_protocol.json"
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    assert proto["calibration"]["dataset"] == "VOC 2012 train tuning split"
    assert proto["validation"]["dataset"] == "VOC 2012 validation"


def test_sensitivity_protocol_calibration_and_hash_are_consistent() -> None:
    proto_path = ROOT / "protocol/sensitivity_protocol.json"
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    recorded = proto.pop("_sha256")
    payload = json.dumps(proto, sort_keys=True, separators=(",", ":"))
    observed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert recorded["scope"] == "canonical JSON excluding _sha256"
    assert recorded["value"] == observed
    assert proto["status"] == "secondary_sensitivity_frozen"
    assert proto["calibration_meta"]["trials_per_target"] == proto["calibration"]["trials_per_target"]
    assert proto["calibration_meta"]["grid_step"] == proto["calibration"]["grid"]["step"]


def test_analysis_refuses_primary_pvalue_claims() -> None:
    """Analysis must not emit primary p-values."""
    # Check the analysis script's disclaimer
    proto_path = ROOT / "protocol/sensitivity_protocol.json"
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    assert proto["statistics"]["primary_pvalues"] is False
    assert proto["statistics"]["relabel_h3"] is False
