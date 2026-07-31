#!/usr/bin/env python3
"""Build the tiny schema-v4 interface fixture used by Chapter 2."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from deterministic_npz import write_npz


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter02_schema_v4_minimal.npz"
)


def build_fixture() -> dict[str, np.ndarray]:
    """Return a deterministic four-depth interface fixture, not a model star."""

    depth_count = 4
    stage_count = 6
    species_count = 139

    temperature = np.array([4800.0, 5200.0, 6100.0, 7600.0], dtype=np.float64)
    column_mass = np.array([1.0e-6, 1.0e-4, 1.0e-2, 1.0], dtype=np.float64)
    gas_pressure = np.array([2.0e-2, 2.0, 2.0e2, 2.0e4], dtype=np.float64)
    electron_density = np.array([1.0e8, 4.0e9, 2.0e11, 8.0e12], dtype=np.float64)
    mass_density = np.array([1.0e-12, 8.0e-11, 5.0e-9, 3.0e-7], dtype=np.float64)

    population_shape = (depth_count, stage_count, species_count)
    partition_normalized_populations = np.full(
        population_shape, 2.0e-12, dtype=np.float64
    )
    ion_stage_populations = 2.5 * partition_normalized_populations
    fractional_doppler_widths = np.full(
        population_shape, 1.0e-5, dtype=np.float64
    )

    elemental_abundances = np.full(99, 1.0e-12, dtype=np.float64)
    elemental_abundances[0] = 0.92
    elemental_abundances[1] = 0.079
    elemental_abundances[25] = 1.0e-3
    elemental_abundances /= elemental_abundances.sum()

    edge_wavelength = np.array([300.0, 400.0, 500.0, 650.0, 900.0], dtype=np.float64)
    edge_frequency = 2.99792458e17 / edge_wavelength
    edge_midpoint = 0.5 * (edge_wavelength[:-1] + edge_wavelength[1:])
    edge_width_squared_over_two = 0.5 * np.diff(edge_wavelength) ** 2

    return {
        "atmosphere_schema_version": np.array(4, dtype=np.int64),
        "temperature": temperature,
        "gas_pressure": gas_pressure,
        "electron_density": electron_density,
        "mass_density": mass_density,
        "column_mass": column_mass,
        "partition_normalized_populations": partition_normalized_populations,
        "ion_stage_populations": ion_stage_populations,
        "fractional_doppler_widths": fractional_doppler_widths,
        "hydrogen_neutral_population": np.full(depth_count, 1.0e12, dtype=np.float64),
        "helium_neutral_population": np.full(depth_count, 8.0e10, dtype=np.float64),
        "helium_singly_ionized_population": np.full(
            depth_count, 2.0e8, dtype=np.float64
        ),
        "molecular_hydrogen_population": np.full(
            depth_count, 1.0e7, dtype=np.float64
        ),
        "hydrogen_partition_normalized_ion_stage_populations": np.full(
            (depth_count, 2), 1.0e9, dtype=np.float64
        ),
        "carbon_partition_normalized_ion_stage_populations": np.full(
            (depth_count, 2), 1.0e6, dtype=np.float64
        ),
        "magnesium_neutral_partition_normalized_population": np.full(
            depth_count, 1.0e5, dtype=np.float64
        ),
        "aluminum_neutral_partition_normalized_population": np.full(
            depth_count, 1.0e4, dtype=np.float64
        ),
        "silicon_neutral_partition_normalized_population": np.full(
            depth_count, 1.0e5, dtype=np.float64
        ),
        "iron_neutral_partition_normalized_population": np.full(
            depth_count, 1.0e5, dtype=np.float64
        ),
        # hc/kT in centimetres from the exact cgs constants used by synthesis.
        # The corresponding constant is 1.4387768775039338 cm K;
        # multiplying it by 1e7 would instead produce nanometres.
        "hc_over_kt": 1.4387768775039338 / temperature,
        "microturbulence": np.full(depth_count, 2.0e5, dtype=np.float64),
        "elemental_abundances": elemental_abundances,
        "signed_continuum_edge_frequency_hz": edge_frequency,
        "continuum_edge_wavelength_nm": edge_wavelength,
        "continuum_edge_midpoint_wavelength_nm": edge_midpoint,
        "continuum_edge_interval_width_squared_over_two_nm2": (
            edge_width_squared_over_two
        ),
    }


def main() -> None:
    """Write the fixture deterministically and report its content hash."""

    write_npz(OUTPUT_PATH, build_fixture())
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    print(f"{OUTPUT_PATH.relative_to(REPOSITORY_ROOT)} {digest}")


if __name__ == "__main__":
    main()
