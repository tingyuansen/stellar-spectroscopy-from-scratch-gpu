#!/usr/bin/env python3
"""Build the compact, source-bound inputs for Chapter 11.

The builder reads the pinned Payne Zero checkout without modifying it.  It
derives one 80-layer fixed-column seed from the published solar atmosphere,
keeps a small strong-line view of the observed packed catalog, and keeps a
small detailed-transition view.  The products are inputs, not parity goldens.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from deterministic_npz import write_npz


PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
SOURCE_HASHES = {
    "sun_structured_atmosphere.npz": (
        "d686ea7107d60bf1707607e3d6377d283fb3eb7115c170ac2aeef54fbaa6abdb"
    ),
    "observed_atomic_lines.npy": (
        "649b039474ca6cda6e9fe0dea3052e5d47c24320ba8be1f48cd2ec39b7bcf84f"
    ),
    "detailed_transition_lines.npz": (
        "57b3ff244698965b0c507c3ab23f767d85c37bf4705eb97f0a91c597e2330232"
    ),
}
TRANSITION_FIELDS = (
    "vacuum_wavelength_nm",
    "lower_excitation_cm",
    "oscillator_strength",
    "lower_hydrogen_level",
    "upper_hydrogen_level",
    "packed_species_slot",
    "line_type",
    "hydrogen_continuum_selector_index",
    "continuum_species_slot",
    "radiative_damping",
    "stark_damping",
    "van_der_waals_damping",
    "packed_wavelength_index",
    "line_limit",
)


def sha256(path: Path) -> str:
    """Return one file SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(path: Path) -> None:
    """Fail closed unless a source input has its pinned identity."""

    expected = SOURCE_HASHES[path.name]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{path} has SHA-256 {actual}, expected {expected}")


def build_seed(source: Path, destination: Path) -> None:
    """Write the 80-layer supplied seed and pass-2 support controls."""

    with np.load(source, allow_pickle=False) as archive:
        source_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if source_arrays["temperature"].shape != (80,):
        raise RuntimeError("published solar atmosphere is no longer 80 layers")

    abundance = np.asarray(source_arrays["elemental_abundances"], dtype=np.float64)
    if abundance.shape != (99,) or np.any(abundance <= 0.0):
        raise RuntimeError("published solar atmosphere lacks 99 positive abundances")
    fixed_abundance = np.log10(abundance)
    fixed_abundance[:2] = abundance[:2]
    column_mass = np.asarray(source_arrays["column_mass"], dtype=np.float64)
    gravity = 10.0**4.44

    write_npz(
        destination,
        {
            "column_mass": column_mass,
            "temperature": np.asarray(source_arrays["temperature"], dtype=np.float64),
            "gas_pressure": np.asarray(
                source_arrays["gas_pressure"], dtype=np.float64
            ),
            "electron_density": np.asarray(
                source_arrays["electron_density"], dtype=np.float64
            ),
            "rosseland_opacity": np.geomspace(2.0e-4, 1.2, 80).astype(np.float64),
            "radiative_acceleration": np.zeros(80, dtype=np.float64),
            "microturbulence": np.zeros(80, dtype=np.float64),
            "convective_flux": np.zeros(80, dtype=np.float64),
            "convective_velocity": np.zeros(80, dtype=np.float64),
            "fixed_column_abundance_values": fixed_abundance,
            "effective_temperature": np.asarray(5778.0, dtype=np.float64),
            "log_surface_gravity": np.asarray(4.44, dtype=np.float64),
            "opacity_flags": np.asarray(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                dtype=np.int8,
            ),
            "pressure_iteration_enabled": np.asarray(1, dtype=np.int8),
            "previous_integrated_radiation_pressure": (
                0.02 * gravity * column_mass
            ).astype(np.float64),
            "previous_turbulent_pressure": np.zeros(80, dtype=np.float64),
            "source_sha256": np.asarray(SOURCE_HASHES[source.name]),
            "payne_zero_commit": np.asarray(PINNED_COMMIT),
            "fixture_role": np.asarray(
                "supplied 80-layer first-pass seed; not a convergence claim"
            ),
        },
    )


def build_observed_subset(source: Path, destination: Path) -> None:
    """Keep the strongest packed observed rows, preserving wavelength order."""

    words = np.load(source, mmap_mode="r", allow_pickle=False)
    if words.dtype != np.int32 or words.ndim != 2 or words.shape[1] != 4:
        raise RuntimeError("observed atomic source is not packed (N, 4) int32")
    packed_halves = np.ascontiguousarray(words[:, 1:4]).view(np.int16).reshape(-1, 6)
    strength = packed_halves[:, 2].astype(np.int32)
    keep_count = min(2048, words.shape[0])
    strongest = np.argpartition(strength, -keep_count)[-keep_count:]
    rows = strongest[np.argsort(np.asarray(words[strongest, 0]), kind="stable")]
    subset = np.ascontiguousarray(words[rows], dtype=np.int32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, subset, allow_pickle=False)

    write_npz(
        destination.with_suffix(".provenance.npz"),
        {
            "source_row_index": rows.astype(np.int64),
            "source_sha256": np.asarray(SOURCE_HASHES[source.name]),
            "payne_zero_commit": np.asarray(PINNED_COMMIT),
            "selection": np.asarray(
                "2048 largest packed log-strength codes; source wavelength order"
            ),
        },
    )


def build_transition_subset(source: Path, destination: Path) -> None:
    """Keep strong ordinary detailed transitions in their original order."""

    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in TRANSITION_FIELDS}
    oscillator_strength = np.asarray(arrays["oscillator_strength"], dtype=np.float64)
    wavelength = np.asarray(arrays["vacuum_wavelength_nm"], dtype=np.float64)
    line_type = np.asarray(arrays["line_type"], dtype=np.int32)
    candidates = np.flatnonzero(
        (line_type == 0)
        & (wavelength >= 100.0)
        & (wavelength <= 3000.0)
        & np.isfinite(oscillator_strength)
        & (oscillator_strength > 0.0)
    )
    if candidates.size < 64:
        raise RuntimeError("not enough ordinary detailed transitions")
    strongest = candidates[
        np.argpartition(oscillator_strength[candidates], -64)[-64:]
    ]
    rows = np.sort(strongest)
    payload = {name: np.asarray(values[rows]) for name, values in arrays.items()}
    payload.update(
        source_row_index=rows.astype(np.int64),
        source_sha256=np.asarray(SOURCE_HASHES[source.name]),
        payne_zero_commit=np.asarray(PINNED_COMMIT),
        selection=np.asarray(
            "64 strongest positive-gf type-0 transitions from 100-3000 nm"
        ),
    )
    write_npz(destination, payload)


def build(source_root: Path, repository_root: Path) -> None:
    """Build all Chapter 11 compact inputs from one pinned source root."""

    source_root = source_root.resolve()
    repository_root = repository_root.resolve()
    sources = {
        "sun_structured_atmosphere.npz": (
            source_root / "examples/data/sun_structured_atmosphere.npz"
        ),
        "observed_atomic_lines.npy": (
            source_root
            / "source_data_files/source_catalogs/lines/observed_atomic_lines.npy"
        ),
        "detailed_transition_lines.npz": (
            source_root
            / "source_data_files/source_catalogs/lines/detailed_transition_lines.npz"
        ),
    }
    for source in sources.values():
        verify_source(source)

    build_seed(
        sources["sun_structured_atmosphere.npz"],
        repository_root / "data/fixtures/chapter11_solar_seed.npz",
    )
    build_observed_subset(
        sources["observed_atomic_lines.npy"],
        repository_root / "data/subsets/chapter11_observed_atomic_subset.npy",
    )
    build_transition_subset(
        sources["detailed_transition_lines.npz"],
        repository_root / "data/subsets/chapter11_detailed_transition_subset.npz",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/Users/ysting/payne-zero"),
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args()
    build(arguments.source_root, arguments.repository_root)


if __name__ == "__main__":
    main()
