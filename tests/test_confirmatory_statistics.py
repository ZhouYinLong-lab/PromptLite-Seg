from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_confirmatory import holm_adjust


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
    assert grabcut["mean_iou_failure_zero"] == 0.6877666894409938
