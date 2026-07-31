#!/usr/bin/env python3
"""Isolated worker that imports only the pinned Payne Zero checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from deterministic_npz import write_npz


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAMES = (
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


def parse_args() -> argparse.Namespace:
    """Parse source and destination paths."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def fresh_outputs(depth_count: int) -> tuple[np.ndarray, ...]:
    """Allocate the nine exact accumulator families."""

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


def main() -> None:
    """Run serial and two-chunk exact accumulation and save comparison outputs."""

    arguments = parse_args()
    sys.path.insert(0, str(arguments.source_root))
    from payne_zero_atmosphere.transfer_kernels import (
        accumulate_transfer_range_compiled,
        accumulate_transfer_range_parallel,
    )

    fixture_path = (
        REPOSITORY_ROOT / "data" / "fixtures" / "chapter02_transfer_inputs.npz"
    )
    table_path = (
        REPOSITORY_ROOT
        / "data"
        / "static"
        / "atmosphere_tables"
        / "radiative_transfer_tables.npz"
    )
    with np.load(fixture_path, allow_pickle=False) as fixture:
        inputs = {name: np.asarray(fixture[name]) for name in fixture.files}
    with np.load(table_path, allow_pickle=False) as tables:
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
    depth_count = inputs["column_mass"].size
    common = (
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
    serial = fresh_outputs(depth_count)
    parallel = fresh_outputs(depth_count)
    accumulate_transfer_range_compiled(*common, *serial)
    accumulate_transfer_range_parallel(2, *common, *parallel)

    arrays = {}
    for prefix, values in (("serial", serial), ("parallel_chunk2", parallel)):
        arrays.update(
            {
                f"{prefix}_{name}": value
                for name, value in zip(OUTPUT_NAMES, values, strict=True)
            }
        )
    arrays["payne_zero_commit"] = np.array(
        "9c44001feae40b85146630499e6f8a5fed42e5af"
    )
    arrays["fixture_sha256"] = np.array(
        __import__("hashlib").sha256(fixture_path.read_bytes()).hexdigest()
    )
    arrays["transfer_tables_sha256"] = np.array(
        __import__("hashlib").sha256(table_path.read_bytes()).hexdigest()
    )
    write_npz(arguments.output, arrays)


if __name__ == "__main__":
    main()
