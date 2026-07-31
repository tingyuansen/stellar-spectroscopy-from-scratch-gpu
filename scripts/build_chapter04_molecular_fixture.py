#!/usr/bin/env python3
"""Build the input-only Chapter 4 molecular thermochemistry fixture."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from deterministic_npz import write_npz


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHAPTER03_INPUTS = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter03_atom_only_inputs.npz"
)
OUTPUT = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter04_molecular_inputs.npz"
)
EXPECTED_CHAPTER03_SHA256 = (
    "3ed0d65431fc9e284a77011b82267241b25cc56cdffa73e1bc86eec15f9b5219"
)
REFERENCE_BOLTZMANN_ERG_PER_K = 1.38054e-16


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def next_float(value: float) -> float:
    """Return the first float64 value strictly above a branch boundary."""

    return float(np.nextafter(np.float64(value), np.float64(np.inf)))


def build_arrays() -> dict[str, np.ndarray]:
    """Return controlled chemistry inputs without any backend output."""

    if sha256(CHAPTER03_INPUTS) != EXPECTED_CHAPTER03_SHA256:
        raise ValueError("the accepted Chapter 3 abundance fixture has changed")
    with np.load(CHAPTER03_INPUTS, allow_pickle=False) as source:
        elemental_abundances = np.asarray(
            source["elemental_abundances"], dtype=np.float64
        ).copy()

    temperature = np.array(
        [3300.0, 3800.0, 4500.0, 5500.0, 7000.0, 9000.0],
        dtype=np.float64,
    )
    gas_pressure = np.logspace(2.0, 7.0, temperature.size, dtype=np.float64)
    electron_density_seed = (
        0.1
        * gas_pressure
        / (REFERENCE_BOLTZMANN_ERG_PER_K * temperature)
    )

    return {
        "atmosphere_h2_boundary_temperature": np.array(
            [100.0, 101.0, 19899.0, 19900.0, 20000.0, next_float(20000.0)]
        ),
        "atmosphere_polynomial_boundary_temperature": np.array(
            [10000.0, next_float(10000.0)]
        ),
        "column_mass": np.geomspace(1.0e-6, 1.0e2, temperature.size),
        "electron_density_seed": electron_density_seed,
        "elemental_abundances": elemental_abundances,
        "gas_pressure": gas_pressure,
        "microturbulence": np.full(temperature.size, 2.0e5, dtype=np.float64),
        "named_molecule_codes": np.array(
            [101.0, 606.0, 607.0, 608.0, 707.0, 708.0, 808.0]
        ),
        "pressure_control_gas_pressure": np.array([1.0e2, 1.0e4, 1.0e6]),
        "pressure_control_temperature": np.full(3, 3500.0),
        "source_abundance_fixture_sha256": np.array(
            EXPECTED_CHAPTER03_SHA256
        ),
        "synthesis_named_h2_boundary_temperature": np.array(
            [9000.0, next_float(9000.0)]
        ),
        "synthesis_polynomial_boundary_temperature": np.array(
            [10000.0, next_float(10000.0)]
        ),
        "temperature": temperature,
        "temperature_control_gas_pressure": np.full(3, 1.0e5),
        "temperature_control_temperature": np.array(
            [3500.0, 6000.0, 9000.0]
        ),
    }


def main() -> None:
    """Write deterministic input-only fixture bytes."""

    write_npz(OUTPUT, build_arrays())
    print(f"{OUTPUT.relative_to(REPOSITORY_ROOT)} {sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
