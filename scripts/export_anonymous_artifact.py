"""Build and audit a deterministic, identity-free supplementary ZIP."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import zipfile

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pth", ".pt"}
EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
LOCAL_PATH_PATTERNS = (re.compile(r"[a-z]:\\users\\", re.IGNORECASE), re.compile(r"/home/[^/\s]+/"))
DEFAULT_BANNED = (
    "zhouyinlong",
    "github.com/zhouyinlong-lab",
    "nanjing university",
    "南京大学",
    ".git/",
)


def source_files() -> list[tuple[Path, str]]:
    tracked_output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
    )
    tracked = sorted(item.decode("utf-8") for item in tracked_output.split(b"\0") if item)
    prefixes = ("src/promptseg/", "scripts/", "protocol/", "artifacts/confirmatory/", "tests/")
    selected = [
        name
        for name in tracked
        if name.startswith(prefixes) and name != "scripts/export_anonymous_artifact.py"
    ]
    fixed = (
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "requirements.txt",
        "requirements-sam.txt",
        "requirements-sam-cu128.txt",
        "pyproject.toml",
        "reports/report_anonymous.pdf",
    )
    required_tracked = {*fixed, "reports/ANONYMOUS_README.md"}
    missing = sorted(required_tracked - set(tracked))
    if missing:
        raise RuntimeError(f"Required anonymous-artifact files are not tracked: {missing}")
    entries = [(ROOT / name, name) for name in [*selected, *fixed]]
    entries.append((ROOT / "reports/ANONYMOUS_README.md", "README.md"))
    return entries


def identity_hits(text: str, banned: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    found = [token for token in banned if token in lowered]
    if EMAIL_PATTERN.search(text):
        found.append("email-address")
    if any(pattern.search(text) for pattern in LOCAL_PATH_PATTERNS):
        found.append("local-home-path")
    return found


def pdf_text_and_metadata(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = [str(dict(reader.metadata or {}))]
    parts.extend(page.extract_text() or "" for page in reader.pages)
    xmp = reader.xmp_metadata
    if xmp is not None:
        for attribute in ("dc_creator", "dc_title", "dc_subject", "pdf_keywords", "xmp_creator_tool"):
            parts.append(str(getattr(xmp, attribute, "")))
    attachments = getattr(reader, "attachments", {})
    for name, payloads in attachments.items():
        parts.append(str(name))
        if isinstance(payloads, bytes):
            payloads = [payloads]
        for payload in payloads:
            parts.append(payload.decode("utf-8", errors="ignore"))
    return "\n".join(parts)


def audit_entry(path: Path, archive_name: str, banned: tuple[str, ...]) -> None:
    root = ROOT.resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise RuntimeError(f"Anonymous artifact path is a symlink or escapes the repository: {archive_name}")
    lowered_name = archive_name.lower()
    if Path(archive_name).suffix.lower() in FORBIDDEN_SUFFIXES:
        raise RuntimeError(f"Forbidden binary asset in anonymous artifact: {archive_name}")
    if any(token in lowered_name for token in banned):
        raise RuntimeError(f"Identity-bearing archive path: {archive_name}")
    text = pdf_text_and_metadata(path) if path.suffix.lower() == ".pdf" else path.read_text(
        encoding="utf-8", errors="strict"
    )
    found = identity_hits(text, banned)
    if found:
        raise RuntimeError(f"Identity-bearing text in {archive_name}: {found}")


def build_archive(output: Path, banned: tuple[str, ...] = DEFAULT_BANNED) -> None:
    entries = source_files()
    for path, archive_name in entries:
        if not path.is_file():
            raise FileNotFoundError(path)
        audit_entry(path, archive_name, banned)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, archive_name in entries:
            info = zipfile.ZipInfo(archive_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Anonymous artifact contains duplicate paths")
        if any(Path(name).suffix.lower() in FORBIDDEN_SUFFIXES for name in names):
            raise RuntimeError("Anonymous artifact contains forbidden binary assets")
    print(f"Built {output} with {len(entries)} audited files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/promptlite-seg-anonymous.zip"))
    args = parser.parse_args()
    build_archive(args.output)


if __name__ == "__main__":
    main()
