#!/usr/bin/env python3
"""Build ordered Chapter 3 comparison outputs from the pinned read-only oracle."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_ROOT = Path("/Users/ysting/payne-zero")
EXPECTED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
FIXTURE = REPOSITORY_ROOT / "data" / "fixtures" / "chapter03_atom_only_inputs.npz"
OUTPUT_DIR = REPOSITORY_ROOT / "data" / "golden" / "payne_zero"
WORKER = REPOSITORY_ROOT / "scripts" / "chapter03_oracle_worker.py"
OUTPUT_NAMES = (
    "chapter03_atmosphere_saha_outputs.npz",
    "chapter03_atmosphere_atomic_state.npz",
    "chapter03_packed_bridge_outputs.npz",
    "chapter03_synthesis_atomic_state_cpu_float64.npz",
)


def parse_args() -> argparse.Namespace:
    """Parse pinned checkout and frozen fixture overrides for auditing."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--pinned-root", type=Path, default=PINNED_ROOT)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_commit(root: Path) -> str:
    """Read the oracle commit without changing the checkout."""

    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_outputs(fixture_hash: str) -> None:
    """Require every golden to bind itself to the frozen source and fixture."""

    for name in OUTPUT_NAMES:
        path = OUTPUT_DIR / name
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["payne_zero_commit"]) != EXPECTED_COMMIT:
                raise AssertionError(f"{name} has the wrong source commit")
            if str(archive["fixture_sha256"]) != fixture_hash:
                raise AssertionError(f"{name} has the wrong fixture identity")


def main() -> None:
    """Run the isolated oracle worker with a fresh temporary Numba cache."""

    arguments = parse_args()
    root = arguments.pinned_root.expanduser().resolve()
    fixture = arguments.fixture.expanduser().resolve()
    if root != PINNED_ROOT.resolve():
        raise ValueError(
            "the Chapter 3 golden contract permits only "
            f"{PINNED_ROOT.resolve()}"
        )
    actual_commit = pinned_commit(root)
    if actual_commit != EXPECTED_COMMIT:
        raise ValueError(
            f"pinned checkout is {actual_commit}; expected {EXPECTED_COMMIT}"
        )
    fixture_hash = sha256(fixture)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="chapter03-numba-") as cache:
        environment = os.environ.copy()
        environment.update(
            {
                "NUMBA_CACHE_DIR": cache,
                "NUMBA_NUM_THREADS": "2",
                "PAYNE_ZERO_DATA_ROOT": str(root / "source_data_files"),
                "PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE": str(
                    root
                    / "source_data_files"
                    / "synthesis_tables"
                    / "atomic_masses.npz"
                ),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join(
                    (str(root), str(REPOSITORY_ROOT))
                ),
            }
        )
        subprocess.run(
            [
                sys.executable,
                str(WORKER),
                "--fixture",
                str(fixture),
                "--output-dir",
                str(OUTPUT_DIR),
            ],
            check=True,
            cwd=REPOSITORY_ROOT,
            env=environment,
        )

    validate_outputs(fixture_hash)
    for name in OUTPUT_NAMES:
        path = OUTPUT_DIR / name
        print(
            f"{path.relative_to(REPOSITORY_ROOT)} "
            f"{sha256(path)} ({path.stat().st_size} bytes)"
        )


if __name__ == "__main__":
    main()
