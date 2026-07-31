#!/usr/bin/env python3
"""Build four Chapter 5 continuum input states from accepted Chapter 4 code."""

from __future__ import annotations

from dataclasses import fields
import hashlib
from pathlib import Path
import warnings

import numpy as np

from deterministic_npz import write_npz


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHAPTER04_INPUTS = REPOSITORY_ROOT / "data/fixtures/chapter04_molecular_inputs.npz"
EDGE_GRID = REPOSITORY_ROOT / "data/static/synthesis_tables/continuum_edge_grid.npz"
OUTPUT = REPOSITORY_ROOT / "data/fixtures/chapter05_continuum_states.npz"
PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
EXPECTED_CHAPTER04_SHA256 = (
    "351bba75dca1fa502f5cc2a108035f69f2e31c760a90133480f2e7fe31e45f79"
)
EXPECTED_EDGE_SHA256 = (
    "11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e"
)
REFERENCE_BOLTZMANN_ERG_PER_K = 1.38054e-16

REGIMES = {
    "hot_dwarf": {
        "effective_temperature": 30000.0,
        "temperature": [8000.0, 10000.0, 13000.0, 18000.0, 25000.0, 32000.0],
        "gas_pressure_bounds": (1.0e2, 1.0e7),
    },
    "solar_dwarf": {
        "effective_temperature": 5772.0,
        "temperature": [4000.0, 4500.0, 5200.0, 5800.0, 6500.0, 8000.0],
        "gas_pressure_bounds": (1.0e2, 1.0e7),
    },
    "low_gravity_giant": {
        "effective_temperature": 4500.0,
        "temperature": [3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6500.0],
        "gas_pressure_bounds": (1.0, 1.0e5),
    },
    "cool_molecule_rich": {
        "effective_temperature": 3200.0,
        "temperature": [2800.0, 3200.0, 3600.0, 4200.0, 5000.0, 6000.0],
        "gas_pressure_bounds": (1.0e2, 1.0e8),
    },
}


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def regime_inputs(
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    *,
    column_mass: np.ndarray,
    microturbulence: np.ndarray,
    elemental_abundances: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return one positive six-depth thermodynamic track."""

    return {
        "column_mass": column_mass.copy(),
        "temperature": temperature,
        "gas_pressure": gas_pressure,
        "electron_density_seed": (
            0.1 * gas_pressure / (REFERENCE_BOLTZMANN_ERG_PER_K * temperature)
        ),
        "microturbulence": microturbulence.copy(),
        "elemental_abundances": elemental_abundances.copy(),
    }


def build_arrays() -> dict[str, np.ndarray]:
    """Return deterministic Chapter 5 inputs and accepted upstream states."""

    if sha256(CHAPTER04_INPUTS) != EXPECTED_CHAPTER04_SHA256:
        raise ValueError("the accepted Chapter 4 input fixture has changed")
    if sha256(EDGE_GRID) != EXPECTED_EDGE_SHA256:
        raise ValueError("the accepted continuum edge grid has changed")

    from book.chapter04_runtime import (
        build_public_molecular_lane_checkpoint,
        compute_atmosphere_molecular_state,
        configure_local_data_paths,
    )

    configure_local_data_paths()
    from payne_zero_atmosphere.continuum_opacity import (
        build_continuum_atmosphere_state,
    )

    with np.load(CHAPTER04_INPUTS, allow_pickle=False) as source:
        column_mass = np.asarray(source["column_mass"], dtype=np.float64)
        microturbulence = np.asarray(source["microturbulence"], dtype=np.float64)
        elemental_abundances = np.asarray(
            source["elemental_abundances"],
            dtype=np.float64,
        )
    with np.load(EDGE_GRID, allow_pickle=False) as edge:
        edge_wavelength = np.asarray(
            edge["continuum_edge_wavelength_nm"],
            dtype=np.float64,
        )

    arrays: dict[str, np.ndarray] = {
        "payne_zero_commit": np.asarray(PAYNE_ZERO_COMMIT),
        "regime_names": np.asarray(tuple(REGIMES)),
        "source_chapter04_fixture_sha256": np.asarray(EXPECTED_CHAPTER04_SHA256),
        "source_continuum_edge_grid_sha256": np.asarray(EXPECTED_EDGE_SHA256),
        "atmosphere_runner_opacity_flags": np.asarray(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0],
            dtype=np.int64,
        ),
        "hminus_edge_frequency_hz": np.asarray(
            [
                np.nextafter(1.82365e14, -np.inf),
                1.82365e14,
                np.nextafter(1.82365e14, np.inf),
            ]
        ),
        "h2_policy_temperature_k": np.asarray(
            [
                100.0,
                101.0,
                8999.0,
                9000.0,
                19899.0,
                19900.0,
                20000.0,
                np.nextafter(20000.0, np.inf),
            ]
        ),
        "atmosphere_grid_boundary_temperature_k": np.asarray(
            [
                4499.0,
                4500.0,
                7249.0,
                7250.0,
                12999.0,
                13000.0,
                29999.0,
                30000.0,
            ]
        ),
        "synthesis_edge_probe_wavelength_nm": np.asarray(
            [
                np.nextafter(edge_wavelength[170], -np.inf),
                edge_wavelength[170],
                np.nextafter(edge_wavelength[170], np.inf),
            ]
        ),
    }

    for regime_name, specification in REGIMES.items():
        temperature = np.asarray(specification["temperature"], dtype=np.float64)
        lower, upper = specification["gas_pressure_bounds"]
        gas_pressure = np.geomspace(lower, upper, temperature.size)
        inputs = regime_inputs(
            temperature,
            gas_pressure,
            column_mass=column_mass,
            microturbulence=microturbulence,
            elemental_abundances=elemental_abundances,
        )
        prefix = f"{regime_name}__"
        for name, values in inputs.items():
            arrays[f"{prefix}input__{name}"] = np.asarray(values).copy()
        arrays[f"{prefix}effective_temperature"] = np.asarray(
            specification["effective_temperature"],
            dtype=np.float64,
        )

        atmosphere_result = compute_atmosphere_molecular_state(inputs)
        continuum_state = build_continuum_atmosphere_state(
            atmosphere_result.setup.atmosphere,
            atmosphere_result.runtime_state,
        )
        for field in fields(continuum_state):
            arrays[f"{prefix}atmosphere__{field.name}"] = np.asarray(
                getattr(continuum_state, field.name)
            ).copy()

        release_inputs = dict(inputs)
        release_inputs["electron_density_seed"] = (
            atmosphere_result.runtime_state.electron_density.copy()
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow encountered in exp")
            public_lane = build_public_molecular_lane_checkpoint(release_inputs)
        for name, values in public_lane.structured_atmosphere.items():
            arrays[f"{prefix}synthesis__{name}"] = np.asarray(values).copy()

    return arrays


def main() -> None:
    """Write deterministic Chapter 5 fixture bytes."""

    write_npz(OUTPUT, build_arrays())
    print(f"{OUTPUT.relative_to(REPOSITORY_ROOT)} {sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
