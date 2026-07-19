"""Integrity helpers for frozen confirmatory experiment execution."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


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


def canonical_source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git_commit(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def git_is_dirty(repository: Path) -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(output.strip())


def verify_runtime_sources(repository: Path, specification: Path) -> dict:
    payload = json.loads(specification.read_text(encoding="utf-8"))
    expected = payload.get("files")
    if payload.get("schema_version") != 1 or not isinstance(expected, dict) or not expected:
        raise ValueError(f"Invalid runtime source specification: {specification}")
    observed = {
        relative: canonical_source_sha256(repository / relative)
        for relative in sorted(expected)
    }
    mismatches = {
        relative: {"expected": expected[relative], "observed": observed[relative]}
        for relative in observed
        if observed[relative] != expected[relative]
    }
    return {
        "matches": not mismatches,
        "fingerprint": canonical_json_sha256(observed),
        "specification_sha256": sha256_file(specification),
        "mismatches": mismatches,
    }


def module_source_fingerprint(module: object) -> str | None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    root = Path(module_file).resolve().parent
    if not root.is_dir():
        return None
    sources = {
        path.relative_to(root).as_posix(): canonical_source_sha256(path)
        for path in sorted(root.rglob("*.py"))
        if path.is_file() and not path.is_symlink()
    }
    return canonical_json_sha256(sources) if sources else None


def package_versions(distributions: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def base_runtime_environment(distributions: tuple[str, ...]) -> dict:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable_architecture": platform.machine(),
        "packages": package_versions(distributions),
        "python_cache_tag": sys.implementation.cache_tag,
    }


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
