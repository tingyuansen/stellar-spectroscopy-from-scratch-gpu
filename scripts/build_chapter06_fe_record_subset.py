#!/usr/bin/env python3
"""Extract the accepted one-record Chapter 6 Fe I teaching subset."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

try:
    from .deterministic_npz import write_npz
except ImportError:  # direct ``python scripts/...`` execution
    from deterministic_npz import write_npz


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
SOURCE_ARCHIVE_RELATIVE_PATH = Path(
    "source_data_files/source_catalogs/lines/atomic_source_lines_parsed.npz"
)
SOURCE_ARCHIVE_SHA256 = (
    "4eafa927c02a4f74401523149a44e35239f2aaecb4a64f2905a4cd5530c2dde7"
)
SOURCE_ARCHIVE_BYTES = 258_021_389
SOURCE_ARCHIVE_ROW_COUNT = 1_939_975
SOURCE_ROW_INDEX = 873_702
SUBSET_SCHEMA_VERSION = 1
OUTPUT = REPOSITORY_ROOT / "data/subsets/chapter06_fe_i_source_row_873702.npz"

RAW_FIELDS = (
    "stored_wavelength_nm",
    "raw_log_oscillator_strength",
    "species_code",
    "first_energy_column_cm",
    "second_energy_column_cm",
    "radiative_damping_log",
    "stark_damping_log",
    "van_der_waals_damping_log",
    "lower_principal_quantum_number",
    "upper_principal_quantum_number",
    "primary_isotope_number",
    "primary_isotope_log_correction",
    "secondary_isotope_log_correction",
    "energy_shift_field",
    "isotope_shift_units",
    "line_size",
    "line_category_tag",
)


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_source_archive() -> Path:
    """Return the optional full-catalog location used for regeneration."""

    source_root = Path(
        os.environ.get("PAYNE_ZERO_READONLY_ROOT", "/Users/ysting/payne-zero")
    ).expanduser()
    return source_root / SOURCE_ARCHIVE_RELATIVE_PATH


def build_arrays(source_archive: Path) -> dict[str, np.ndarray]:
    """Return one exact source row plus provenance, never computed outputs."""

    source_archive = Path(source_archive)
    if not source_archive.is_file():
        raise FileNotFoundError(
            "the optional full atomic source archive is required to regenerate "
            f"this subset: {source_archive}"
        )
    if source_archive.stat().st_size != SOURCE_ARCHIVE_BYTES:
        raise ValueError(
            f"{source_archive} has {source_archive.stat().st_size} bytes; "
            f"expected {SOURCE_ARCHIVE_BYTES}"
        )
    actual_sha256 = sha256(source_archive)
    if actual_sha256 != SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            f"{source_archive} has SHA-256 {actual_sha256}; "
            f"expected {SOURCE_ARCHIVE_SHA256}"
        )

    with np.load(source_archive, allow_pickle=False) as source:
        if tuple(source.files) != RAW_FIELDS:
            raise ValueError(
                "the atomic source archive field order changed: "
                f"{tuple(source.files)!r}"
            )
        row_counts = {int(np.asarray(source[name]).shape[0]) for name in RAW_FIELDS}
        if row_counts != {SOURCE_ARCHIVE_ROW_COUNT}:
            raise ValueError(
                "the atomic source archive row counts changed: "
                f"{sorted(row_counts)!r}"
            )
        raw_arrays = {
            name: np.array(
                source[name][SOURCE_ROW_INDEX : SOURCE_ROW_INDEX + 1],
                copy=True,
            )
            for name in RAW_FIELDS
        }

    for name, values in raw_arrays.items():
        if values.shape != (1,):
            raise ValueError(f"{name} extraction has shape {values.shape}, expected (1,)")
        if values.dtype.hasobject:
            raise ValueError(f"{name} extraction has forbidden object dtype")

    return {
        **raw_arrays,
        "builder_command": np.asarray(
            "python scripts/build_chapter06_fe_record_subset.py "
            "--source-archive "
            "<optional-full-source>/source_data_files/source_catalogs/lines/"
            "atomic_source_lines_parsed.npz"
        ),
        "payne_zero_commit": np.asarray(PAYNE_ZERO_COMMIT),
        "source_archive_bytes": np.asarray(SOURCE_ARCHIVE_BYTES, dtype=np.int64),
        "source_archive_relative_path": np.asarray(
            SOURCE_ARCHIVE_RELATIVE_PATH.as_posix()
        ),
        "source_archive_row_count": np.asarray(
            SOURCE_ARCHIVE_ROW_COUNT,
            dtype=np.int64,
        ),
        "source_archive_sha256": np.asarray(SOURCE_ARCHIVE_SHA256),
        "source_field_count": np.asarray(len(RAW_FIELDS), dtype=np.int64),
        "source_row_index": np.asarray(SOURCE_ROW_INDEX, dtype=np.int64),
        "subset_role": np.asarray(
            "teaching subset: one immutable raw source record; no computed outputs"
        ),
        "subset_schema_version": np.asarray(
            SUBSET_SCHEMA_VERSION,
            dtype=np.int64,
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse the explicit optional-source and output locations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=default_source_archive(),
        help="read-only full atomic source archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="deterministic one-record subset destination",
    )
    return parser.parse_args()


def main() -> None:
    """Write the deterministic static teaching subset."""

    arguments = parse_args()
    write_npz(arguments.output, build_arrays(arguments.source_archive))
    try:
        display_path = arguments.output.relative_to(REPOSITORY_ROOT)
    except ValueError:
        display_path = arguments.output
    print(f"{display_path} {sha256(arguments.output)}")


if __name__ == "__main__":
    main()
