"""Tests for the privacy-preserving human prompt pilot toolkit.

Covers annotation schema, PII rejection, coordinate validation,
duplicate detection, synthetic labelling, and participant-clustered analysis.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic dataset with 4 images for testing."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()

    rng = np.random.default_rng(42)
    for i in range(4):
        sd = data_dir / f"sample_{i:04d}"
        sd.mkdir()
        img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(img).save(sd / "image.jpg", quality=100)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[30:90, 30:90] = 255
        Image.fromarray(mask).save(sd / "target_mask.png")
        (sd / "prompt.txt").write_text(
            f"source_row={i}\nlabel=1\nclass_name=aeroplane\n"
            f"bbox=30,30,90,90\npoint=60,60\n",
            encoding="utf-8",
        )
    return data_dir


@pytest.fixture
def synthetic_annotations_dir(tmp_path: Path, synthetic_data_dir: Path) -> Path:
    """Generate synthetic annotation CSV files for testing."""
    from scripts.run_human_collection import _load_task_list, run_synthetic_demo

    # Create a minimal protocol
    proto_path = tmp_path / "proto.json"
    proto = {
        "design": {
            "sample": {
                "sampling_seed": 42,
                "images_per_class": 2,
                "total_images": 4,
            },
        },
    }
    proto_path.write_text(json.dumps(proto))

    ann_dir = tmp_path / "annotations"
    tasks = _load_task_list(proto_path, synthetic_data_dir)

    # Limit tasks for testing
    limited_tasks = tasks[:8]  # 4 images × 2 tasks
    run_synthetic_demo(limited_tasks, "P001", ann_dir)

    return ann_dir


# ---------------------------------------------------------------------------
# Schema and field validation
# ---------------------------------------------------------------------------

def test_synthetic_annotations_have_correct_schema(synthetic_annotations_dir: Path) -> None:
    csv_path = synthetic_annotations_dir / "annotations_P001.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    assert rows[0]["participant_code"] == "P001"
    assert "is_synthetic" in rows[0]
    assert rows[0]["is_synthetic"] == "True"

    for row in rows:
        assert row["task_type"] in ("point", "box")
        if row["task_type"] == "point":
            assert row["point_x"] != ""
            assert row["point_y"] != ""
            assert row["box_x0"] == ""
        else:
            assert row["box_x0"] != ""
            assert row["box_y0"] != ""
            assert row["box_x1"] != ""
            assert row["box_y1"] != ""


def test_synthetic_annotations_contain_is_synthetic_true(synthetic_annotations_dir: Path) -> None:
    csv_path = synthetic_annotations_dir / "annotations_P001.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        assert row.get("is_synthetic", "").strip().lower() in ("true", "1")


# ---------------------------------------------------------------------------
# Validator: PII rejection
# ---------------------------------------------------------------------------

def test_validator_rejects_pii_fields(tmp_path: Path, synthetic_data_dir: Path) -> None:
    """CSV containing PII field names must fail validation."""
    csv_path = tmp_path / "bad.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "participant_code", "task_id", "image_id", "task_type",
            "point_x", "point_y", "box_x0", "box_y0", "box_x1", "box_y1",
            "elapsed_time_ms", "timeout", "email",
        ])
        writer.writeheader()
        writer.writerow({
            "participant_code": "P001",
            "task_id": "task_0000",
            "image_id": "sample_0000",
            "task_type": "point",
            "point_x": "60",
            "point_y": "60",
            "box_x0": "", "box_y0": "", "box_x1": "", "box_y1": "",
            "elapsed_time_ms": "1000",
            "timeout": "False",
            "email": "test" + chr(64) + "example.com",
        })

    from scripts.validate_human_annotations import validate_annotations
    is_valid, errors = validate_annotations(csv_path, synthetic_data_dir)
    assert not is_valid
    assert any("PII" in e or "email" in e for e in errors)


def test_validator_accepts_clean_annotations(synthetic_annotations_dir: Path, synthetic_data_dir: Path) -> None:
    from scripts.validate_human_annotations import validate_annotations
    csv_path = synthetic_annotations_dir / "annotations_P001.csv"
    is_valid, errors = validate_annotations(csv_path, synthetic_data_dir)
    assert is_valid, errors


def test_validator_rejects_duplicate_task_ids(tmp_path: Path, synthetic_data_dir: Path) -> None:
    csv_path = tmp_path / "dup.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "participant_code", "task_id", "image_id", "task_type",
            "point_x", "point_y", "box_x0", "box_y0", "box_x1", "box_y1",
            "elapsed_time_ms", "timeout",
        ])
        writer.writeheader()
        for _ in range(2):  # duplicate
            writer.writerow({
                "participant_code": "P001",
                "task_id": "task_0000",
                "image_id": "sample_0000",
                "task_type": "point",
                "point_x": "60", "point_y": "60",
                "box_x0": "", "box_y0": "", "box_x1": "", "box_y1": "",
                "elapsed_time_ms": "1000",
                "timeout": "False",
            })

    from scripts.validate_human_annotations import validate_annotations
    is_valid, errors = validate_annotations(csv_path, synthetic_data_dir)
    assert not is_valid
    assert any("Duplicate" in e or "duplicate" in e for e in errors)


def test_validator_rejects_out_of_bounds_coordinates(tmp_path: Path, synthetic_data_dir: Path) -> None:
    csv_path = tmp_path / "oob.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "participant_code", "task_id", "image_id", "task_type",
            "point_x", "point_y", "box_x0", "box_y0", "box_x1", "box_y1",
            "elapsed_time_ms", "timeout",
        ])
        writer.writeheader()
        writer.writerow({
            "participant_code": "P001",
            "task_id": "task_0000",
            "image_id": "sample_0000",
            "task_type": "point",
            "point_x": "9999",  # out of bounds
            "point_y": "9999",
            "box_x0": "", "box_y0": "", "box_x1": "", "box_y1": "",
            "elapsed_time_ms": "1000",
            "timeout": "False",
        })

    from scripts.validate_human_annotations import validate_annotations
    is_valid, errors = validate_annotations(csv_path, synthetic_data_dir)
    assert not is_valid
    assert any("out of bounds" in e.lower() for e in errors)


def test_validator_rejects_zero_area_box(tmp_path: Path, synthetic_data_dir: Path) -> None:
    csv_path = tmp_path / "zero.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "participant_code", "task_id", "image_id", "task_type",
            "point_x", "point_y", "box_x0", "box_y0", "box_x1", "box_y1",
            "elapsed_time_ms", "timeout",
        ])
        writer.writeheader()
        writer.writerow({
            "participant_code": "P001",
            "task_id": "task_0000",
            "image_id": "sample_0000",
            "task_type": "box",
            "point_x": "", "point_y": "",
            "box_x0": "50", "box_y0": "50",
            "box_x1": "50", "box_y1": "50",  # zero area
            "elapsed_time_ms": "1000",
            "timeout": "False",
        })

    from scripts.validate_human_annotations import validate_annotations
    is_valid, errors = validate_annotations(csv_path, synthetic_data_dir)
    assert not is_valid
    assert any("zero area" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Synthetic labelling check
# ---------------------------------------------------------------------------

def test_validator_allows_synthetic_column(synthetic_annotations_dir: Path, synthetic_data_dir: Path) -> None:
    """Validator should allow is_synthetic column (it's not PII)."""
    from scripts.validate_human_annotations import validate_annotations
    csv_path = synthetic_annotations_dir / "annotations_P001.csv"
    is_valid, errors = validate_annotations(csv_path, synthetic_data_dir)
    assert is_valid, errors


def test_validator_rejects_mislabelled_synthetic(tmp_path: Path, synthetic_data_dir: Path) -> None:
    """is_synthetic must be true/false, not arbitrary values."""
    csv_path = tmp_path / "bad_syn.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "participant_code", "task_id", "image_id", "task_type",
            "point_x", "point_y", "box_x0", "box_y0", "box_x1", "box_y1",
            "elapsed_time_ms", "timeout", "is_synthetic",
        ])
        writer.writeheader()
        writer.writerow({
            "participant_code": "P001",
            "task_id": "task_0000",
            "image_id": "sample_0000",
            "task_type": "point",
            "point_x": "60", "point_y": "60",
            "box_x0": "", "box_y0": "", "box_x1": "", "box_y1": "",
            "elapsed_time_ms": "1000",
            "timeout": "False",
            "is_synthetic": "maybe",  # invalid
        })

    from scripts.validate_human_annotations import validate_annotations
    is_valid, errors = validate_annotations(csv_path, synthetic_data_dir)
    assert not is_valid


# ---------------------------------------------------------------------------
# Collection script
# ---------------------------------------------------------------------------

def test_collection_help() -> None:
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_human_collection.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_collection_synthetic_mode(synthetic_data_dir: Path, tmp_path: Path) -> None:
    """Synthetic mode must produce valid annotation CSV without GUI."""
    proto_path = tmp_path / "proto.json"
    proto = {
        "design": {
            "sample": {
                "sampling_seed": 42,
                "images_per_class": 4,
                "total_images": 4,
            },
        },
    }
    proto_path.write_text(json.dumps(proto))

    out_dir = tmp_path / "annotations"

    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_human_collection.py"),
         "--participant", "P999",
         "--data-dir", str(synthetic_data_dir),
         "--protocol", str(proto_path),
         "--output-dir", str(out_dir),
         "--synthetic",
         "--max-images", "4",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr

    # Verify output
    csv_path = out_dir / "annotations_P999.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    for row in rows:
        assert row["is_synthetic"] == "True"
        assert row["participant_code"] == "P999"


# ---------------------------------------------------------------------------
# Validator CLI integration
# ---------------------------------------------------------------------------

def test_validator_help() -> None:
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_human_annotations.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_validator_cli_with_synthetic_data(
    synthetic_annotations_dir: Path, synthetic_data_dir: Path,
) -> None:
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_human_annotations.py"),
         "--annotations", str(synthetic_annotations_dir),
         "--data-dir", str(synthetic_data_dir),
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "VALID" in result.stdout


# ---------------------------------------------------------------------------
# Analysis: participant-clustered and synthetic refusal
# ---------------------------------------------------------------------------

def test_analyze_human_pilot_help() -> None:
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_human_pilot.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_analyze_refuses_synthetic_as_human_result(
    synthetic_annotations_dir: Path, synthetic_data_dir: Path, tmp_path: Path,
) -> None:
    """Analysis on synthetic data must refuse to present as human result."""
    out_dir = tmp_path / "analysis_output"

    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_human_pilot.py"),
         "--annotations", str(synthetic_annotations_dir),
         "--data-dir", str(synthetic_data_dir),
         "--output-dir", str(out_dir),
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr

    # Analysis output must mark synthetic
    analysis = json.loads((out_dir / "human_pilot_analysis.json").read_text(encoding="utf-8"))
    assert analysis["is_synthetic"] is True
    assert analysis["contains_human_results"] is False
    # Must not claim confirmatory evidence
    assert "confirmatory" not in analysis.get("disclaimer", "").lower() or \
           "not" in analysis.get("disclaimer", "").lower()


def test_analyze_produces_participant_clustered_output(
    synthetic_annotations_dir: Path, synthetic_data_dir: Path, tmp_path: Path,
) -> None:
    """Analysis must produce per-participant metrics."""
    out_dir = tmp_path / "analysis_output"

    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_human_pilot.py"),
         "--annotations", str(synthetic_annotations_dir),
         "--data-dir", str(synthetic_data_dir),
         "--output-dir", str(out_dir),
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr

    # Check participant-clustered CSV
    csv_path = out_dir / "participant_summary.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    # Each row should be per-participant
    pcs = [r["participant_code"] for r in rows]
    assert len(pcs) == len(set(pcs)), "participant_summary must have unique participants"


def test_analyze_no_annotations_errors(tmp_path: Path) -> None:
    """Analysis must error cleanly when no annotations exist."""
    import subprocess
    out_dir = tmp_path / "analysis_output"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/analyze_human_pilot.py"),
         "--annotations", str(tmp_path / "nonexistent"),
         "--output-dir", str(out_dir),
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Protocol checks
# ---------------------------------------------------------------------------

def test_protocol_contains_ethics_checkpoint() -> None:
    proto_path = ROOT / "protocol/human_pilot_protocol.json"
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    assert "ethics_checkpoint" in proto
    assert proto["ethics_checkpoint"]["required_before_recruitment"] is True
    assert proto["ethics_checkpoint"]["status"] == "NOT_YET_REVIEWED"


def test_protocol_excludes_pii_fields() -> None:
    proto_path = ROOT / "protocol/human_pilot_protocol.json"
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    excluded = proto["data_collection"]["explicitly_excluded"]
    pii_indicators = ["name", "email", "ip_address", "student_number"]
    for pii in pii_indicators:
        assert pii in excluded


def test_protocol_specifies_exploratory_status() -> None:
    proto_path = ROOT / "protocol/human_pilot_protocol.json"
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    assert proto["status_exploratory"] is True
    assert proto["pilot_only"] is True
    assert proto["analysis"]["confirmatory_claims"] is False


# ---------------------------------------------------------------------------
# Git-ignore verification
# ---------------------------------------------------------------------------

def test_human_annotations_dir_is_gitignored() -> None:
    """data/human_annotations/ must be in .gitignore."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "human_annotations" in gitignore or "data/human" in gitignore


# ---------------------------------------------------------------------------
# Randomization determinism
# ---------------------------------------------------------------------------

def test_task_list_is_deterministic(synthetic_data_dir: Path, tmp_path: Path) -> None:
    """Same seed and data must produce the same task list."""
    from scripts.run_human_collection import _load_task_list

    proto_path = tmp_path / "proto.json"
    proto = {
        "design": {
            "sample": {
                "sampling_seed": 42,
                "images_per_class": 2,
                "total_images": 4,
            },
        },
    }
    proto_path.write_text(json.dumps(proto))

    tasks_a = _load_task_list(proto_path, synthetic_data_dir)
    tasks_b = _load_task_list(proto_path, synthetic_data_dir)

    assert len(tasks_a) == len(tasks_b)
    for a, b in zip(tasks_a, tasks_b):
        assert a["task_id"] == b["task_id"]
        assert a["image_id"] == b["image_id"]
        assert a["task_type"] == b["task_type"]
