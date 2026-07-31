"""Manifest-bound support for Chapter 2's exact transfer integration fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_FIXTURE_PATH = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter02_transfer_inputs.npz"
)
TRANSFER_TABLE_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "atmosphere_tables"
    / "radiative_transfer_tables.npz"
)
TRANSFER_GOLDEN_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "golden"
    / "payne_zero"
    / "chapter02_transfer_outputs.npz"
)

TRANSFER_OUTPUT_NAMES = (
    "rosseland_accumulator",
    "radiation_energy_density",
    "integrated_eddington_flux",
    "radiative_acceleration",
    "surface_radiation_pressure_constant",
    "temperature_correction_heating_derivative",
    "temperature_correction_mean_intensity_minus_source_integral",
    "temperature_correction_integrated_eddington_flux",
    "temperature_correction_diagonal_lambda",
)


def fresh_transfer_outputs(depth_count: int) -> tuple[np.ndarray, ...]:
    """Allocate the exact nine output accumulator families."""

    return (
        np.zeros(depth_count, dtype=np.float64),
        np.zeros(depth_count, dtype=np.float64),
        np.zeros(depth_count, dtype=np.float64),
        np.zeros(depth_count, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(depth_count, dtype=np.float64),
        np.zeros(depth_count, dtype=np.float64),
        np.zeros(depth_count, dtype=np.float64),
        np.zeros(depth_count, dtype=np.float64),
    )


def load_transfer_fixture() -> tuple[tuple[object, ...], int]:
    """Load declared fixture/table inputs in the exact production argument order."""

    with np.load(TRANSFER_FIXTURE_PATH, allow_pickle=False) as fixture:
        inputs = {name: np.asarray(fixture[name]) for name in fixture.files}
    with np.load(TRANSFER_TABLE_PATH, allow_pickle=False) as tables:
        transfer_grid = np.asarray(
            tables["transfer_optical_depth_grid"], dtype=np.float32
        )
        mean_intensity_operator = np.asarray(
            tables["mean_intensity_operator"], dtype=np.float32
        )
        eddington_flux_operator = np.asarray(
            tables["eddington_flux_operator"], dtype=np.float32
        )
        second_moment_weights = np.asarray(
            tables["second_moment_weights"], dtype=np.float32
        )

    frequency_count = inputs["frequency_hz"].size
    common_arguments = (
        0,
        frequency_count,
        inputs["frequency_hz"],
        inputs["frequency_weights"],
        inputs["planck_all"],
        inputs["stimulated_all"],
        inputs["continuum_absorption_slab"],
        inputs["continuum_scattering_slab"],
        inputs["continuum_source_slab"],
        inputs["line_mass_absorption_coefficient_slab"],
        inputs["column_mass"],
        inputs["h_over_kt"],
        inputs["temperature"],
        transfer_grid,
        mean_intensity_operator,
        eddington_flux_operator,
        second_moment_weights,
        float(inputs["target_integrated_eddington_flux"]),
        float(inputs["effective_temperature"]),
        frequency_count,
    )
    return common_arguments, inputs["column_mass"].size


def run_transfer_fixture(
    *, chunk_count: int = 2
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run the exact serial and fixed-chunk compiled transfer paths."""

    from payne_zero_atmosphere.transfer_kernels import (
        accumulate_transfer_range_compiled,
        accumulate_transfer_range_parallel,
    )

    common_arguments, depth_count = load_transfer_fixture()
    serial_outputs = fresh_transfer_outputs(depth_count)
    parallel_outputs = fresh_transfer_outputs(depth_count)
    accumulate_transfer_range_compiled(*common_arguments, *serial_outputs)
    accumulate_transfer_range_parallel(
        int(chunk_count), *common_arguments, *parallel_outputs
    )
    serial = dict(zip(TRANSFER_OUTPUT_NAMES, serial_outputs, strict=True))
    parallel = dict(zip(TRANSFER_OUTPUT_NAMES, parallel_outputs, strict=True))
    return serial, parallel
