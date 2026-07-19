from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_voc_from_manifest import safe_output_root, safe_sample_dir, write_source_image


ROOT = Path(__file__).resolve().parents[1]


def test_output_root_rejects_repository_and_ancestor() -> None:
    with pytest.raises(ValueError, match="unsafe output directory"):
        safe_output_root(ROOT)
    with pytest.raises(ValueError, match="unsafe output directory"):
        safe_output_root(ROOT.parent)


@pytest.mark.parametrize("sample_id", ["../escape", "val_000001/child", "C:/escape", "sample_000001"])
def test_sample_id_cannot_escape_output(tmp_path: Path, sample_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid VOC sample_id"):
        safe_sample_dir(tmp_path.resolve(), sample_id)


def test_source_image_bytes_are_preserved(tmp_path: Path) -> None:
    payload = b"\xff\xd8source-jpeg-bytes\xff\xd9"
    destination = tmp_path / "image.jpg"

    write_source_image({"bytes": payload, "path": None}, destination)

    assert destination.read_bytes() == payload
