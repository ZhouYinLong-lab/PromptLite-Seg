"""Integrity helpers for frozen confirmatory experiment execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


SAMPLE_FILENAMES = ("image.jpg", "target_mask.png", "prompt.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_is_dirty(repository: Path) -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository,
        text=True,
    )
    return bool(output.strip())


def manifest_sample_ids(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    sample_ids = [str(row["sample_id"]) for row in rows]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Manifest must contain unique sample IDs: {path}")
    return sample_ids


def dataset_fingerprint(sample_dirs: list[Path]) -> str:
    """Hash ordered sample IDs plus the exact image, mask, and prompt bytes."""

    digest = hashlib.sha256()
    for sample_dir in sample_dirs:
        digest.update(sample_dir.name.encode("utf-8"))
        digest.update(b"\0")
        for name in SAMPLE_FILENAMES:
            path = sample_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            digest.update(name.encode("ascii"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)
