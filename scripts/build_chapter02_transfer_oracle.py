#!/usr/bin/env python3
"""Generate Chapter 2 transfer goldens with the pinned read-only source."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"


def parse_args() -> argparse.Namespace:
    """Parse the read-only source root."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/Users/ysting/payne-zero"),
    )
    return parser.parse_args()


def main() -> None:
    """Verify the pin and run the isolated oracle worker."""

    arguments = parse_args()
    source_root = arguments.source_root.expanduser().resolve()
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"expected {EXPECTED_COMMIT}, found {commit}")

    output_path = (
        REPOSITORY_ROOT
        / "data"
        / "golden"
        / "payne_zero"
        / "chapter02_transfer_outputs.npz"
    )
    environment = os.environ.copy()
    environment["NUMBA_CACHE_DIR"] = str(
        REPOSITORY_ROOT / ".cache" / "chapter02-pinned-oracle"
    )
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "chapter02_transfer_oracle_worker.py"),
            "--source-root",
            str(source_root),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
    )
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"{output_path.relative_to(REPOSITORY_ROOT)} {digest}")


if __name__ == "__main__":
    main()
