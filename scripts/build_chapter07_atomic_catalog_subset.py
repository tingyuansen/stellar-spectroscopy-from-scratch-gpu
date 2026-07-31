#!/usr/bin/env python3
"""Build the compact, source-faithful Chapter 7 atomic teaching catalog."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
PINNED_SOURCE_SHA256 = (
    "4eafa927c02a4f74401523149a44e35239f2aaecb4a64f2905a4cd5530c2dde7"
)
CHAPTER06_SOURCE_ROW = 873702
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
    """Return one file SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_type(raw: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the exact catalog routing labels without transforming line physics."""

    species = np.asarray(raw["species_code"], dtype=np.float64)
    atomic_number = (species + 1.0e-6).astype(np.int64)
    fraction = species - atomic_number
    ion_stage = np.where(
        fraction > 1.0e-6,
        np.rint(fraction * 100.0).astype(np.int64) + 1,
        1,
    )
    tag = raw["line_category_tag"]
    isotope = raw["primary_isotope_number"]
    result = np.zeros(species.shape, dtype=np.int64)
    result = np.where(tag == b"AUT", 1, result)
    result = np.where(tag == b"COR", 2, result)
    result = np.where(tag == b"PRD", 3, result)
    standard = ~np.isin(tag, np.asarray([b"AUT", b"COR", b"PRD"]))
    result = np.where(
        standard & (atomic_number == 1) & (ion_stage == 1) & (isotope == 2),
        -2,
        result,
    )
    result = np.where(
        standard & (atomic_number == 1) & (ion_stage == 1) & (isotope != 2),
        -1,
        result,
    )
    result = np.where(
        standard & (atomic_number == 2) & (ion_stage == 1),
        -3,
        result,
    )
    result = np.where(
        standard & (atomic_number == 2) & (ion_stage == 2),
        -6,
        result,
    )
    return result


def nearest_index(
    wavelength_nm: np.ndarray,
    routing_type: np.ndarray,
    target_type: int,
    target_wavelength_nm: float,
) -> int:
    """Select the source row of one nearest special-profile example."""

    candidates = np.flatnonzero(routing_type == target_type)
    if candidates.size == 0:
        raise RuntimeError(f"source catalog has no line type {target_type}")
    return int(
        candidates[
            np.argmin(np.abs(wavelength_nm[candidates] - target_wavelength_nm))
        ]
    )


def selected_rows(raw: dict[str, np.ndarray]) -> np.ndarray:
    """Choose a small ordinary forest plus one example per wired special family."""

    wavelength = np.asarray(raw["stored_wavelength_nm"], dtype=np.float64)
    routing = line_type(raw)
    ordinary = np.flatnonzero(
        (routing == 0) & (wavelength >= 498.8) & (wavelength <= 499.3)
    )
    corrected_log_gf = (
        np.asarray(raw["raw_log_oscillator_strength"], dtype=np.float64)
        + np.asarray(raw["primary_isotope_log_correction"], dtype=np.float64)
        + np.asarray(raw["secondary_isotope_log_correction"], dtype=np.float64)
    )
    strongest_order = np.lexsort((ordinary, -corrected_log_gf[ordinary]))
    chosen_ordinary = ordinary[strongest_order[:128]]
    chosen_ordinary = np.unique(
        np.concatenate(
            (
                chosen_ordinary,
                np.asarray([CHAPTER06_SOURCE_ROW], dtype=np.int64),
            )
        )
    )
    special = np.asarray(
        [
            nearest_index(wavelength, routing, 1, 632.0),
            nearest_index(wavelength, routing, -1, 656.3),
            nearest_index(wavelength, routing, -2, 656.3),
            nearest_index(wavelength, routing, -3, 587.6),
            nearest_index(wavelength, routing, -6, 468.6),
        ],
        dtype=np.int64,
    )
    return np.unique(np.concatenate((chosen_ordinary, special)))


def build(source: Path, destination: Path) -> None:
    """Write one compact NPZ while preserving every selected raw field exactly."""

    if sha256(source) != PINNED_SOURCE_SHA256:
        raise RuntimeError("atomic source catalog does not match the pinned identity")
    with np.load(source, allow_pickle=False) as archive:
        raw = {name: np.asarray(archive[name]) for name in RAW_FIELDS}
    rows = selected_rows(raw)
    payload = {name: np.asarray(raw[name][rows]) for name in RAW_FIELDS}
    payload.update(
        source_row_index=rows.astype(np.int64),
        source_catalog_sha256=np.asarray(PINNED_SOURCE_SHA256),
        payne_zero_commit=np.asarray(PINNED_COMMIT),
        selection_description=np.asarray(
            "128 strongest ordinary stored-wavelength rows in 498.8-499.3 nm, "
            "the Chapter 6 row, and nearest AUT/H I/D I/He I/He II examples"
        ),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    build(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
