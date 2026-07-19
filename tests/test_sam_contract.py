from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from promptseg.dataset import Prompt
from promptseg.sam import PROMPT_MODES, predict_sam
from scripts.run_prompt_uncertainty_experiment import ENSEMBLE_METHODS, select_ensemble_predictions
from scripts.run_robustness_experiment import evaluate_sam_prompt_pair


class FakeSamPredictor:
    """CPU-only implementation of the public SamPredictor.predict contract."""

    def __init__(self, responses: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        masks, scores = self.responses.pop(0)
        return masks, scores, np.zeros_like(masks, dtype=np.float32)


@pytest.fixture
def prompt() -> Prompt:
    return Prompt(bbox=(2, 3, 8, 9), point=(5, 6), label=1, class_name="aeroplane")


def masks(*active: tuple[int, int]) -> np.ndarray:
    result = np.zeros((3, 12, 12), dtype=bool)
    for index, (y, x) in enumerate(active):
        result[index, y, x] = True
    return result


@pytest.mark.parametrize(
    ("mode", "expects_point", "expects_box"),
    [
        ("point_only", True, False),
        ("box_only", False, True),
        ("point_box", True, True),
    ],
)
def test_predict_sam_contract_for_every_prompt_mode(
    prompt: Prompt,
    mode: str,
    expects_point: bool,
    expects_box: bool,
) -> None:
    predictor = FakeSamPredictor([(masks((1, 1), (2, 2), (3, 3)), np.array([0.1, 0.9, 0.3]))])

    prediction, score = predict_sam(predictor, prompt, mode)

    call = predictor.calls[0]
    assert (call["point_coords"] is not None) is expects_point
    assert (call["point_labels"] is not None) is expects_point
    assert (call["box"] is not None) is expects_box
    assert call["multimask_output"] is True
    assert prediction[2, 2]
    assert prediction.sum() == 1
    assert score == pytest.approx(0.9)


def test_predict_sam_rejects_unknown_mode(prompt: Prompt) -> None:
    predictor = FakeSamPredictor([])

    with pytest.raises(ValueError, match="Unknown SAM prompt mode"):
        predict_sam(predictor, prompt, "scribble")


def test_robustness_sam_variants_cover_repair_score_and_oracle(prompt: Prompt) -> None:
    noisy_masks = masks((1, 1), (2, 2), (3, 3))
    repaired_masks = masks((4, 4), (5, 5), (6, 6))
    predictor = FakeSamPredictor(
        [
            (noisy_masks, np.array([0.1, 0.8, 0.2])),
            (repaired_masks, np.array([0.1, 0.2, 0.95])),
        ]
    )
    target = np.zeros((12, 12), dtype=bool)
    target[2, 2] = True

    variants = evaluate_sam_prompt_pair(predictor, prompt, prompt, target)

    assert set(variants) == {
        "sam_noisy_prompt",
        "sam_repaired_prompt",
        "sam_score_selected_prompt",
        "sam_oracle_best_prompt",
    }
    assert variants["sam_score_selected_prompt"][0][6, 6]
    assert variants["sam_oracle_best_prompt"][0][2, 2]
    assert len(predictor.calls) == 2


def test_uncertainty_ensemble_covers_every_selection_branch() -> None:
    target = np.zeros((3, 3), dtype=bool)
    target[1, 1] = True
    first = np.zeros_like(target)
    first[0, 0] = True
    second = target.copy()
    third = target.copy()

    variants = select_ensemble_predictions(
        [first, second, third],
        [0.2, 0.9, 0.4],
        target,
    )

    assert tuple(variants) == (
        "sam_single_noisy",
        "sam_score_select",
        "sam_consistency_medoid",
        "sam_vote_consensus",
        "sam_oracle_best",
    )
    assert variants["sam_single_noisy"][0][0, 0]
    assert variants["sam_score_select"][0][1, 1]
    assert variants["sam_consistency_medoid"][0][1, 1]
    assert variants["sam_vote_consensus"][0][1, 1]
    assert variants["sam_oracle_best"][0][1, 1]


def test_prompt_mode_registry_is_stable() -> None:
    assert PROMPT_MODES == ("point_only", "box_only", "point_box")


class FakeSamModel:
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.device: str | None = None

    def to(self, device: str) -> "FakeSamModel":
        self.device = device
        return self


class FakeCliPredictor:
    def __init__(self, model: FakeSamModel) -> None:
        self.model = model
        self.shape: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        self.shape = image.shape[:2]

    def predict(self, *, point_coords, point_labels, box, multimask_output):
        assert self.shape is not None
        assert multimask_output is True
        height, width = self.shape
        result = np.zeros((3, height, width), dtype=bool)
        if box is not None:
            x0, y0, x1, y1 = np.asarray(box, dtype=int)
            result[1, max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
        if point_coords is not None:
            x, y = np.asarray(point_coords[0], dtype=int)
            result[2, max(0, y - 2) : min(height, y + 3), max(0, x - 2) : min(width, x + 3)] = True
        scores = np.array([0.1, 0.9 if box is not None else 0.2, 0.8], dtype=np.float32)
        return result, scores, np.zeros_like(result, dtype=np.float32)


def install_fake_sam_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)
    torch_module.inference_mode = nullcontext
    torch_module.no_grad = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    sam_module = ModuleType("segment_anything")
    sam_module.SamPredictor = FakeCliPredictor
    sam_module.sam_model_registry = {
        model_type: (lambda checkpoint, _model_type=model_type: FakeSamModel(checkpoint))
        for model_type in ("vit_b", "vit_l", "vit_h")
    }
    monkeypatch.setitem(sys.modules, "segment_anything", sam_module)


@pytest.mark.parametrize(
    ("module_name", "extra_args", "expected_key"),
    [
        ("scripts.run_sam_experiment", [], "method"),
        ("scripts.run_robustness_experiment", ["--include-sam", "--trials", "1"], "include_sam"),
        (
            "scripts.run_prompt_uncertainty_experiment",
            ["--trials", "1", "--ensemble-size", "1"],
            "ensemble_size",
        ),
    ],
)
def test_sam_cli_branches_run_with_contract_runtime(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_data_dir: Path,
    tmp_path: Path,
    module_name: str,
    extra_args: list[str],
    expected_key: str,
) -> None:
    install_fake_sam_runtime(monkeypatch)
    module = __import__(module_name, fromlist=["main"])
    checkpoint = tmp_path / "fake-sam.pth"
    checkpoint.write_bytes(b"contract-test-only")
    output_dir = tmp_path / module_name.rsplit(".", 1)[-1]
    argv = [
        module_name,
        "--data-dir",
        str(synthetic_data_dir),
        "--output-dir",
        str(output_dir),
        "--checkpoint",
        str(checkpoint),
        "--max-samples",
        "1",
        "--device",
        "cpu",
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_samples"] == 1
    assert expected_key in summary
    assert (output_dir / "metrics.csv").is_file()

    if module_name.endswith("run_robustness_experiment"):
        methods = {row["method"] for row in summary["summary"]}
        assert {
            "sam_noisy_prompt",
            "sam_repaired_prompt",
            "sam_score_selected_prompt",
            "sam_oracle_best_prompt",
        } <= methods
    if module_name.endswith("run_prompt_uncertainty_experiment"):
        methods = {row["method"] for row in summary["summary"]}
        assert set(ENSEMBLE_METHODS) <= methods
