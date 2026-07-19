"""Build and audit a deterministic, identity-free supplementary ZIP."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pth", ".pt"}
DEFAULT_BANNED = (
    "zhouyinlong",
    "github.com/zhouyinlong-lab",
    "nanjing university",
    "南京大学",
    ".git/",
)


def source_files() -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    for directory in ("src/promptseg", "scripts", "protocol", "artifacts/confirmatory", "tests"):
        for path in sorted((ROOT / directory).rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.name != Path(__file__).name
            ):
                entries.append((path, path.relative_to(ROOT).as_posix()))
    for name in (
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "requirements.txt",
        "requirements-sam.txt",
        "pyproject.toml",
        "reports/report_anonymous.pdf",
    ):
        entries.append((ROOT / name, name))
    entries.append((ROOT / "reports/ANONYMOUS_README.md", "README.md"))
    return entries


def audit_entry(path: Path, archive_name: str, banned: tuple[str, ...]) -> None:
    lowered_name = archive_name.lower()
    if Path(archive_name).suffix.lower() in FORBIDDEN_SUFFIXES:
        raise RuntimeError(f"Forbidden binary asset in anonymous artifact: {archive_name}")
    if any(token in lowered_name for token in banned):
        raise RuntimeError(f"Identity-bearing archive path: {archive_name}")
    if path.suffix.lower() != ".pdf":
        text = path.read_text(encoding="utf-8", errors="strict").lower()
        found = [token for token in banned if token in text]
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
