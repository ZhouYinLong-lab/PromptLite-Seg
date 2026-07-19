from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_voc_from_manifest import (
    OUTPUT_MARKER,
    safe_output_root,
    safe_sample_dir,
    validate_managed_output,
    write_output_marker,
    write_source_image,
)


ROOT = Path(__file__).resolve().parents[1]


def test_output_root_rejects_repository_and_ancestor() -> None:
    with pytest.raises(ValueError, match="unsafe output directory"):
        safe_output_root(ROOT)
    with pytest.raises(ValueError, match="unsafe output directory"):
        safe_output_root(ROOT.parent)
    with pytest.raises(ValueError, match="unsafe output directory"):
        safe_output_root(Path.home().parent)


def test_output_root_rejects_arbitrary_external_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe output directory"):
        safe_output_root(tmp_path / "materialized")

    assert safe_output_root(ROOT / "data" / "safe-materialized") == (
        ROOT / "data" / "safe-materialized"
    ).resolve()


def test_replace_requires_owned_materialization_marker(tmp_path: Path) -> None:
    unmanaged = tmp_path / "important-data"
    unmanaged.mkdir()
    (unmanaged / "keep.txt").write_text("do not delete", encoding="utf-8")

    with pytest.raises(ValueError, match="unmanaged directory"):
        validate_managed_output(unmanaged)

    write_output_marker(unmanaged)
    validate_managed_output(unmanaged)
    assert (unmanaged / OUTPUT_MARKER).is_file()


@pytest.mark.parametrize("sample_id", ["../escape", "val_000001/child", "C:/escape", "sample_000001"])
def test_sample_id_cannot_escape_output(tmp_path: Path, sample_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid VOC sample_id"):
        safe_sample_dir(tmp_path.resolve(), sample_id)


def test_source_image_bytes_are_preserved(tmp_path: Path) -> None:
    payload = b"\xff\xd8source-jpeg-bytes\xff\xd9"
    destination = tmp_path / "image.jpg"

    write_source_image({"bytes": payload, "path": None}, destination)

    assert destination.read_bytes() == payload
