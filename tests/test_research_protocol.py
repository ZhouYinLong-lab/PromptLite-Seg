from __future__ import annotations

import hashlib
import json
from pathlib import Path

from promptseg.utils import SEVERITIES


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_frozen_protocol_matches_manifests_and_algorithm_sources() -> None:
    protocol = json.loads((ROOT / "protocol/research_protocol.json").read_text(encoding="utf-8"))

    assert protocol["status"] == "frozen_before_confirmatory_evaluation"
    assert len(protocol["primary_hypotheses"]) == 3
    assert protocol["statistics"]["primary_family_correction"].startswith("Holm")
    assert "upper bounds" in protocol["statistics"]["oracle_policy"]

    for split_name in ("tuning", "confirmatory"):
        split = protocol["splits"][split_name]
        manifest_path = ROOT / split["manifest"]
        rows = read_jsonl(manifest_path)
        assert len(rows) == split["samples"]
        assert sha256(manifest_path) == split["manifest_sha256"]

    for relative_path, expected_hash in protocol["frozen_algorithm_sources"].items():
        assert sha256(ROOT / relative_path) == expected_hash

    calibration = protocol["noise_calibration"]
    calibration_path = ROOT / calibration["artifact"]
    assert sha256(calibration_path) == calibration["artifact_sha256"]


def test_runtime_noise_scales_match_tuning_calibration() -> None:
    calibration = json.loads((ROOT / "protocol/noise_calibration.json").read_text(encoding="utf-8"))

    for severity in ("mild", "moderate"):
        assert SEVERITIES[severity]["point"] == calibration["results"][severity]["point"]["scale"]
        assert SEVERITIES[severity]["box"] == calibration["results"][severity]["box"]["scale"]
        assert calibration["results"][severity]["point"]["absolute_error"] < 0.005
        assert calibration["results"][severity]["box"]["absolute_error"] < 0.005


def test_tuning_and_confirmatory_manifests_are_disjoint_and_representative() -> None:
    tuning = read_jsonl(ROOT / "protocol/manifests/tuning_train.jsonl")
    confirmatory = read_jsonl(ROOT / "protocol/manifests/confirmatory_validation.jsonl")

    assert len(tuning) == 100
    assert len(confirmatory) == 1449
    assert {row["source_split"] for row in tuning} == {"train"}
    assert {row["source_split"] for row in confirmatory} == {"val"}
    assert {row["class_name"] for row in tuning} == {row["class_name"] for row in confirmatory}
    assert {(row["source_split"], row["source_row"]) for row in tuning}.isdisjoint(
        {(row["source_split"], row["source_row"]) for row in confirmatory}
    )

    tuning_counts = {
        class_name: sum(row["class_name"] == class_name for row in tuning)
        for class_name in {row["class_name"] for row in tuning}
    }
    assert set(tuning_counts.values()) == {5}


def test_manifest_contains_no_image_or_mask_payloads() -> None:
    for path in (ROOT / "protocol/manifests").glob("*.jsonl"):
        text = path.read_text(encoding="utf-8")
        assert '"bytes"' not in text
        assert '"image"' not in text
        assert '"mask"' not in text
