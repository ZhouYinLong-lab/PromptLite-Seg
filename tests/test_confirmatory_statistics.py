from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_confirmatory import holm_adjust, load_manifest, validate_metric_design


ROOT = Path(__file__).resolve().parents[1]


def test_holm_adjustment_is_monotone_and_step_down() -> None:
    rows = [
        {"hypothesis": "H1", "p_raw": 0.01},
        {"hypothesis": "H2", "p_raw": 0.03},
        {"hypothesis": "H3", "p_raw": 0.04},
    ]

    adjusted = holm_adjust(rows)

    assert [row["p_holm"] for row in adjusted] == [0.03, 0.06, 0.06]
    assert [row["reject_holm_005"] for row in adjusted] == [True, False, False]


def test_cpu_summary_distinguishes_failure_aggregation() -> None:
    payload = json.loads(
        (ROOT / "artifacts/confirmatory/cpu/summary.json").read_text(encoding="utf-8")
    )
    grabcut = next(row for row in payload["summaries"] if row["method"] == "grabcut_point_box")

    assert grabcut["num_failures"] == 8
    assert grabcut["mean_iou_success_only"] > grabcut["mean_iou_failure_zero"]
    assert grabcut["mean_iou_failure_zero"] == 0.6859115017253278


def test_confirmatory_design_rejects_a_missing_method_row() -> None:
    cpu = pd.read_csv(ROOT / "artifacts/confirmatory/cpu/metrics.csv", keep_default_na=False)
    sam = pd.read_csv(ROOT / "artifacts/confirmatory/sam/metrics.csv", keep_default_na=False)
    manifest = load_manifest(ROOT / "protocol/manifests/confirmatory_validation.jsonl")
    protocol = json.loads((ROOT / "protocol/research_protocol.json").read_text(encoding="utf-8"))
    partial_cpu = cpu.drop(cpu[(cpu["sample_id"] == "val_000000") & (cpu["method"] == "robust_superpixel")].index)

    with pytest.raises(ValueError, match="exactly one row"):
        validate_metric_design(partial_cpu, sam, manifest, protocol)


def test_confirmatory_design_types_optional_sam_quality_columns() -> None:
    cpu = pd.read_csv(ROOT / "artifacts/confirmatory/cpu/metrics.csv", keep_default_na=False)
    sam = pd.read_csv(ROOT / "artifacts/confirmatory/sam/metrics.csv", keep_default_na=False)
    manifest = load_manifest(ROOT / "protocol/manifests/confirmatory_validation.jsonl")
    protocol = json.loads((ROOT / "protocol/research_protocol.json").read_text(encoding="utf-8"))

    _, validated = validate_metric_design(cpu, sam, manifest, protocol)

    noisy = validated[validated["experiment"] != "modality"]
    assert noisy["box_iou"].notna().all()
    assert noisy["point_hit"].isin({True, False}).all()
