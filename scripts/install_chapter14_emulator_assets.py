#!/usr/bin/env python3
"""Install Chapter 14's immutable inference assets from a Payne Zero checkout.

Only runtime checkpoints and their two manifests are copied.  Training
corpora are intentionally excluded: inference must not depend on them.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path.home() / "payne-zero" / "source_data_files"
DESTINATION_ROOT = REPOSITORY_ROOT / "data/static"

ASSETS = {
    "atmosphere_emulator/release_manifest.json": (
        "fe8093a6c0260f2524efd35a974790797e2cc8f921e9dd109477f03d845a789c"
    ),
    "atmosphere_emulator/five_label/checkpoint.pt": (
        "c32717016d4f9047ab37bc17b6900faf9c514de3407292a07a745c35a50784e5"
    ),
    "atmosphere_emulator/cno8/checkpoint.pt": (
        "be97c0b729490b9e18f092c59f836adb421e2e99359b44436edc3ae5632f8a02"
    ),
    "atmosphere_emulator/direct_abundance/manifest.json": (
        "fb59da5e6bd3f8fcba06e0c4c284137e90aab5c4e93165daa74d8ce2ae268710"
    ),
    "atmosphere_emulator/direct_abundance/checkpoint.pt": (
        "1b8e1db1514956dfbf890eb5ae96e01bd918acfc86be538b6b77230332104243"
    ),
}


def sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_assets(source_root: Path) -> None:
    """Copy only verified inference files into the textbook data tree."""

    for relative, expected_sha256 in ASSETS.items():
        source = source_root / relative
        destination = DESTINATION_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing source runtime asset: {source}")
        observed_source = sha256(source)
        if observed_source != expected_sha256:
            raise RuntimeError(f"source runtime asset changed: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and sha256(destination) == expected_sha256:
            continue
        shutil.copyfile(source, destination)
        if sha256(destination) != expected_sha256:
            raise RuntimeError(f"copied runtime asset failed verification: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Payne Zero source_data_files directory (read only)",
    )
    args = parser.parse_args()
    install_assets(args.source_root.resolve())
    print("Chapter 14 emulator assets: installed and verified")


if __name__ == "__main__":
    main()
