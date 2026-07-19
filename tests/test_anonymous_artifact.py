from __future__ import annotations

from pathlib import Path
import zipfile

from scripts.export_anonymous_artifact import DEFAULT_BANNED, FORBIDDEN_SUFFIXES, build_archive


def test_anonymous_export_has_no_identity_data_or_git_history(tmp_path: Path) -> None:
    output = tmp_path / "anonymous.zip"

    build_archive(output)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        git_prefix = "." + "git/"
        assert "README.md" in names
        assert "reports/report_anonymous.pdf" in names
        assert not any(name.startswith(git_prefix) for name in names)
        assert not any(".egg-info/" in name for name in names)
        assert not any(Path(name).suffix.lower() in FORBIDDEN_SUFFIXES for name in names)
        for name in names:
            if Path(name).suffix.lower() in {".md", ".py", ".toml", ".txt", ".json", ".jsonl", ".csv"}:
                text = archive.read(name).decode("utf-8").lower()
                assert not any(token in text for token in DEFAULT_BANNED)
