"""Download protocol assets and reject any byte-level source drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.voc import download_file, sha256_file


SAM_CHECKPOINT = {
    "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "sha256": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    "path": "checkpoints/sam_vit_b_01ec64.pth",
}


def fetch_and_verify(url: str, destination: Path, expected_sha256: str) -> None:
    download_file(url, destination)
    observed = sha256_file(destination)
    if observed != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {destination}: expected {expected_sha256}, observed {observed}"
        )
    print(f"Verified {destination}: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-summary",
        type=Path,
        default=Path("protocol/manifests/manifest_summary.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--include-sam", action="store_true")
    args = parser.parse_args()

    summary = json.loads(args.manifest_summary.read_text(encoding="utf-8"))
    for split, source in summary["sources"].items():
        destination = args.cache_dir / f"pascal_voc_2012_{split}.parquet"
        fetch_and_verify(source["url"], destination, source["sha256"])
    if args.include_sam:
        fetch_and_verify(
            SAM_CHECKPOINT["url"],
            Path(SAM_CHECKPOINT["path"]),
            SAM_CHECKPOINT["sha256"],
        )


if __name__ == "__main__":
    main()
