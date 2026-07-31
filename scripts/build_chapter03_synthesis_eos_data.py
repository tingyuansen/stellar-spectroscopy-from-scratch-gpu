#!/usr/bin/env python3
"""Split the pinned mixed synthesis EOS bundle into honest data roles."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from deterministic_npz import write_npz


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_SHA256 = (
    "0e235e7f1edecf39630690f4c68f4fc952f55785a08174562bc9575100fc4e27"
)
DEFAULT_SOURCE = (
    Path("/Users/ysting/payne-zero")
    / "source_data_files"
    / "synthesis_tables"
    / "partition_saha_inputs.npz"
)
STATIC_OUTPUT = (
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "synthesis_tables"
    / "partition_saha_tables.npz"
)
FIXTURE_OUTPUT = (
    REPOSITORY_ROOT
    / "data"
    / "fixtures"
    / "chapter03_synthesis_eos_state.npz"
)

FIXTURE_ARRAYS = (
    "temperature",
    "thermal_energy_ev",
    "thermal_energy_erg",
    "hc_over_kt",
    "natural_log_temperature",
    "gas_pressure",
    "electron_density",
    "total_nuclei_number_density",
    "elemental_abundances",
    "ion_stage_count",
    "ground_partition_table",
)


def parse_args() -> argparse.Namespace:
    """Parse an optional read-only source-bundle path."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="pinned mixed partition_saha_inputs.npz development oracle",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_arrays(source: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return source arrays split into invariant tables and one state fixture."""

    source = source.expanduser().resolve()
    actual_sha256 = sha256(source)
    if actual_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"{source} has SHA-256 {actual_sha256}; "
            f"expected {EXPECTED_SOURCE_SHA256}"
        )

    with np.load(source, allow_pickle=False) as archive:
        source_names = set(archive.files)
        fixture_names = set(FIXTURE_ARRAYS)
        missing = sorted(fixture_names - source_names)
        if missing:
            raise KeyError(f"source EOS bundle is missing fixture arrays: {missing}")
        fixture = {
            name: np.asarray(archive[name]).copy() for name in FIXTURE_ARRAYS
        }
        static = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
            if name not in fixture_names
        }

    if set(static).intersection(fixture):
        raise AssertionError("static and fixture array roles overlap")
    if set(static).union(fixture) != source_names:
        raise AssertionError("split does not exhaust the source EOS bundle")
    return static, fixture


def main() -> None:
    """Write both role-separated archives with deterministic NPZ bytes."""

    arguments = parse_args()
    static, fixture = split_arrays(arguments.source)
    write_npz(STATIC_OUTPUT, static)
    write_npz(FIXTURE_OUTPUT, fixture)
    print(
        f"{STATIC_OUTPUT.relative_to(REPOSITORY_ROOT)} "
        f"{sha256(STATIC_OUTPUT)} ({len(static)} arrays)"
    )
    print(
        f"{FIXTURE_OUTPUT.relative_to(REPOSITORY_ROOT)} "
        f"{sha256(FIXTURE_OUTPUT)} ({len(fixture)} arrays)"
    )


if __name__ == "__main__":
    main()
