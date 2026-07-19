from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def test_readme_local_links_exist() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    missing: list[str] = []

    for raw_target in LINK_PATTERN.findall(text):
        target = raw_target.strip().strip("<>").split("#", 1)[0]
        if not target or re.match(r"^(?:https?://|mailto:)", target):
            continue
        path = ROOT / unquote(target)
        if not path.exists():
            missing.append(raw_target)

    assert not missing, f"Missing local README targets: {missing}"
