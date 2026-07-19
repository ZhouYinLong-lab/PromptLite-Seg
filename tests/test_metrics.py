from __future__ import annotations

import numpy as np
import pytest

from promptseg.metrics import dice, iou


def test_metrics_normal_overlap() -> None:
    pred = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[1, 0], [1, 0]], dtype=bool)

    assert iou(pred, target) == pytest.approx(1 / 3)
    assert dice(pred, target) == pytest.approx(1 / 2)


def test_metrics_empty_masks_are_perfect_match() -> None:
    empty = np.zeros((3, 3), dtype=bool)

    assert iou(empty, empty) == 1.0
    assert dice(empty, empty) == 1.0


def test_metrics_disjoint_masks_are_zero() -> None:
    pred = np.array([[1, 0], [0, 0]], dtype=bool)
    target = np.array([[0, 0], [0, 1]], dtype=bool)

    assert iou(pred, target) == 0.0
    assert dice(pred, target) == 0.0
