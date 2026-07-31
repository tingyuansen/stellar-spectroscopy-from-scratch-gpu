#!/usr/bin/env python3
"""Build the small, role-honest atom-only integration fixture for Chapter 3."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from deterministic_npz import write_npz


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_STATE = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter03_synthesis_eos_state.npz"
)
OUTPUT = REPOSITORY_ROOT / "data" / "fixtures" / "chapter03_atom_only_inputs.npz"
EXPECTED_SOURCE_STATE_SHA256 = (
    "ecc2856d69c7d96bcdfb6d50988addcd06c68f6e492d9c1f08d21492984ad6c9"
)
SOURCE_DEPTH_INDICES = np.array([0, 16, 32, 48, 64, 79], dtype=np.int64)


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_arrays() -> dict[str, np.ndarray]:
    """Return controlled closure and one-element inputs with explicit roles."""

    if sha256(SOURCE_STATE) != EXPECTED_SOURCE_STATE_SHA256:
        raise ValueError("the extracted synthesis EOS state has changed")
    with np.load(SOURCE_STATE, allow_pickle=False) as source:
        temperature = np.asarray(
            source["temperature"][SOURCE_DEPTH_INDICES], np.float64
        )
        gas_pressure = np.asarray(
            source["gas_pressure"][SOURCE_DEPTH_INDICES], np.float64
        )
        electron_density_seed = np.asarray(
            source["electron_density"][SOURCE_DEPTH_INDICES], np.float64
        )
        elemental_abundances = np.asarray(
            source["elemental_abundances"], np.float64
        )

    saha_temperature = np.array([3500.0, 5000.0, 10000.0, 30000.0])
    saha_gas_pressure = np.array([1.0e2, 1.0e3, 1.0e5, 1.0e6])
    saha_electron_density = np.full(4, 1.0e13, dtype=np.float64)
    saha_charge_square_density = np.array(
        [2.0e13, 2.0e14, 2.0e16, 1.0e21], dtype=np.float64
    )
    thermal_energy = 1.38054e-16 * saha_temperature
    saha_total_nuclei_number_density = (
        saha_gas_pressure / thermal_energy - saha_electron_density
    )
    if np.any(saha_total_nuclei_number_density <= 0.0):
        raise AssertionError("the one-element fixture must have positive nuclei density")

    return {
        "column_mass": np.geomspace(1.0e-6, 1.0e2, temperature.size),
        "electron_density_seed": electron_density_seed,
        "elemental_abundances": elemental_abundances,
        "gas_pressure": gas_pressure,
        "microturbulence": np.array(
            [0.0, 1.0e5, 2.0e5, 2.0e5, 3.0e5, 4.0e5], dtype=np.float64
        ),
        "saha_atomic_number": np.array([1, 10, 26], dtype=np.int64),
        "saha_charge_square_density": saha_charge_square_density,
        "saha_electron_density": saha_electron_density,
        "saha_gas_pressure": saha_gas_pressure,
        "saha_ion_stage_count": np.array([2, 6, 5], dtype=np.int64),
        "saha_population_mode": np.array([11, 12, 13], dtype=np.int64),
        "saha_temperature": saha_temperature,
        "saha_total_nuclei_number_density": (
            saha_total_nuclei_number_density
        ),
        "source_depth_indices": SOURCE_DEPTH_INDICES,
        "temperature": temperature,
    }


def main() -> None:
    """Write deterministic fixture bytes."""

    write_npz(OUTPUT, build_arrays())
    print(f"{OUTPUT.relative_to(REPOSITORY_ROOT)} {sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
