from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import zipfile

from pypdf import PdfReader, PdfWriter
import pytest

import scripts.export_anonymous_artifact as exporter
from scripts.export_anonymous_artifact import ALLOWED_SUFFIXES, DEFAULT_BANNED, build_archive


def test_anonymous_report_meets_course_page_and_blinding_rules() -> None:
    pdf_path = Path("reports/report_anonymous.pdf")
    reader = PdfReader(pdf_path)
    page_text = [page.extract_text() or "" for page in reader.pages]
    reference_pages = [
        page_number
        for page_number, text in enumerate(page_text, start=1)
        if "References" in text
    ]

    assert reference_pages, "The anonymous report has no References boundary"
    assert reference_pages[0] - 1 <= 7, "Main text exceeds the seven-page limit"
    assert "acknowledg" not in "\n".join(page_text[: reference_pages[0] - 1]).lower()
    assert not exporter.identity_hits(
        exporter.pdf_text_and_metadata(pdf_path),
        DEFAULT_BANNED,
    )


def test_anonymous_export_has_no_identity_data_or_git_history(tmp_path: Path) -> None:
    output = tmp_path / "anonymous.zip"

    build_archive(output)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        git_prefix = "." + "git/"
        assert "README.md" in names
        assert "reports/report_anonymous.pdf" in names
        assert "artifacts/secondary/adaptive_summary.json" in names
        assert not any(name.startswith(git_prefix) for name in names)
        assert not any(".egg-info/" in name for name in names)
        assert "scripts/export_anonymous_artifact.py" not in names
        assert "tests/test_anonymous_artifact.py" not in names
        assert not any(Path(name).suffix.lower() not in ALLOWED_SUFFIXES for name in names)
        for name in names:
            if Path(name).suffix.lower() in {".md", ".py", ".toml", ".txt", ".json", ".jsonl", ".csv"}:
                text = archive.read(name).decode("utf-8").lower()
                assert not any(token in text for token in DEFAULT_BANNED)


def test_pdf_metadata_identity_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "anonymous.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_metadata({"/Author": "Zhou" + "YinLong"})
    with pdf_path.open("wb") as handle:
        writer.write(handle)
    monkeypatch.setattr(exporter, "ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="Identity-bearing"):
        exporter.audit_entry(pdf_path, "anonymous.pdf", DEFAULT_BANNED)


def test_pdf_metadata_identity_variants_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "anonymous.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_metadata({"/Author": "Yinlong Zhou"})
    with pdf_path.open("wb") as handle:
        writer.write(handle)
    monkeypatch.setattr(exporter, "ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="Identity-bearing"):
        exporter.audit_entry(pdf_path, "anonymous.pdf", DEFAULT_BANNED)


def test_personal_home_path_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text_path = tmp_path / "log.txt"
    text_path.write_text(r"checkpoint=C:\Users\Alice\models\sam.pth", encoding="utf-8")
    monkeypatch.setattr(exporter, "ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="Identity-bearing"):
        exporter.audit_entry(text_path, "log.txt", DEFAULT_BANNED)


def test_svg_and_pdf_attachments_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    svg_path = tmp_path / "leak.svg"
    svg_path.write_text('<svg><image href="data:image/jpeg;base64,AA=="/></svg>', encoding="utf-8")
    with pytest.raises(RuntimeError, match="not allowed"):
        exporter.audit_entry(svg_path, "leak.svg", DEFAULT_BANNED)

    pdf_path = tmp_path / "attached.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_attachment("voc.jpg", b"image payload")
    with pdf_path.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(RuntimeError, match="attachments are forbidden"):
        exporter.audit_entry(pdf_path, "attached.pdf", DEFAULT_BANNED)


def test_extracted_anonymous_artifact_runs_its_documented_tests(tmp_path: Path) -> None:
    output = tmp_path / "anonymous.zip"
    extracted = tmp_path / "extracted"
    build_archive(output)
    with zipfile.ZipFile(output) as archive:
        archive.extractall(extracted)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=extracted,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
