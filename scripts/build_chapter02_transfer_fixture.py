#!/usr/bin/env python3
"""Build the compact exact-transfer integration fixture for Chapter 2."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from deterministic_npz import write_npz


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter02_transfer_inputs.npz"
)


def build_fixture() -> dict[str, np.ndarray]:
    """Return small positive transfer inputs with exact production layouts."""

    depth_count = 8
    frequency_count = 4
    column_mass = np.geomspace(1.0e-5, 10.0, depth_count).astype(np.float64)
    temperature = np.linspace(4800.0, 7000.0, depth_count, dtype=np.float64)
    frequency_hz = np.linspace(4.0e14, 6.0e14, frequency_count, dtype=np.float64)
    frequency_weights = np.full(frequency_count, 1.0 / frequency_count)

    planck_all = np.empty((frequency_count, depth_count), dtype=np.float64)
    for frequency_index in range(frequency_count):
        planck_all[frequency_index] = (
            1.0e-5
            * (temperature / temperature[0])
            * (frequency_hz[frequency_index] / frequency_hz[0]) ** 2
        )

    continuum_absorption_slab = np.empty(
        (depth_count, frequency_count), dtype=np.float64
    )
    for frequency_index in range(frequency_count):
        continuum_absorption_slab[:, frequency_index] = (
            0.2 + 0.05 * frequency_index + 0.01 * np.arange(depth_count)
        )

    return {
        "frequency_hz": frequency_hz,
        "frequency_weights": frequency_weights,
        "planck_all": planck_all,
        "stimulated_all": np.ones_like(planck_all),
        "continuum_absorption_slab": continuum_absorption_slab,
        "continuum_scattering_slab": np.zeros_like(continuum_absorption_slab),
        "continuum_source_slab": np.ascontiguousarray(planck_all.T),
        "line_mass_absorption_coefficient_slab": np.zeros(
            (depth_count, frequency_count), dtype=np.float32
        ),
        "column_mass": column_mass,
        "h_over_kt": 6.62607015e-27 / (1.380649e-16 * temperature),
        "temperature": temperature,
        "target_integrated_eddington_flux": np.array(1.0, dtype=np.float64),
        "effective_temperature": np.array(5772.0, dtype=np.float64),
    }


def main() -> None:
    """Write the deterministic fixture and report its hash."""

    write_npz(OUTPUT_PATH, build_fixture())
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    print(f"{OUTPUT_PATH.relative_to(REPOSITORY_ROOT)} {digest}")


if __name__ == "__main__":
    main()
