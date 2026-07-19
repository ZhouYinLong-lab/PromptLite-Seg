from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_requirements_match_project_dependencies() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"(?ms)^dependencies = \[(.*?)^\]", pyproject)
    assert match is not None
    project_dependencies = set(re.findall(r'"([^"]+)"', match.group(1)))
    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert requirements == project_dependencies


def test_sam_requirements_extend_cpu_requirements() -> None:
    lines = [
        line.strip()
        for line in (ROOT / "requirements-sam.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert lines[0] == "-r requirements.txt"
    assert any(line.startswith("segment-anything") for line in lines[1:])
