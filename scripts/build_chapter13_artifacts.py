#!/usr/bin/env python3
"""Build the compact Chapter 13 correction fixture and pinned comparison.

The input is analytic and contains no computed answer.  The comparison is
evaluated by the exact staged ``temperature_correction.py`` whose byte identity
is pinned below.  The reader-facing runtime always computes before it opens the
comparison archive.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
STATIC_ROOT = REPOSITORY_ROOT / "data/static"
os.environ.setdefault("PAYNE_ZERO_DATA_ROOT", str(STATIC_ROOT))
os.environ.setdefault(
    "PAYNE_ZERO_ATMOSPHERE_DATA_ROOT",
    str(STATIC_ROOT / "atmosphere_tables"),
)
os.environ.setdefault(
    "PAYNE_ZERO_SYNTHESIS_DATA_ROOT",
    str(STATIC_ROOT / "synthesis_tables"),
)

from scripts.deterministic_npz import write_npz  # noqa: E402


PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
PINNED_SOURCE_SHA256 = {
    "src/payne_zero_atmosphere/temperature_correction.py": (
        "67728389ba857511979d0f82ea59f0bf41ee635b8151ae26673dace02b195d21"
    ),
    "src/payne_zero_atmosphere/convergence.py": (
        "6b4c674deda148baab6fd90e8a25eed2921581ce7d4bace489c024bc1c2748cb"
    ),
}
FIXTURE_PATH = REPOSITORY_ROOT / "data/fixtures/chapter13_correction_inputs.npz"
GOLDEN_PATH = (
    REPOSITORY_ROOT
    / "data/golden/payne_zero/chapter13/chapter13_correction_outputs.npz"
)
MANIFEST_PATH = REPOSITORY_ROOT / "data/chapter13_artifacts.json"


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def analytic_inputs() -> dict[str, np.ndarray]:
    """Return a smooth 64-layer correction state with all three terms active."""

    layer_count = 64
    layer = np.arange(layer_count, dtype=np.float64)
    rosseland_optical_depth = 10.0 ** (-6.875 + 0.125 * layer)
    column_mass = (
        3.0e-5 * (rosseland_optical_depth / rosseland_optical_depth[0]) ** 0.72
    )
    temperature = 4300.0 + 46.0 * layer + 160.0 * np.tanh((layer - 35.0) / 12.0)
    rosseland_opacity = 0.12 + 0.28 * (layer / (layer_count - 1.0)) ** 1.4
    effective_temperature = 5777.0
    target_flux = 5.6697e-5 / 12.5664 * effective_temperature**4
    integrated_flux = target_flux * (
        0.88 + 0.16 * layer / (layer_count - 1.0) + 0.025 * np.sin(layer / 7.0)
    )
    convective_flux = target_flux * 0.10 / (1.0 + np.exp(-(layer - 39.0) / 4.0))

    from payne_zero_atmosphere.radiative_transfer import (
        differentiate_on_depth_grid,
    )

    absorption_heating_derivative = (
        integrated_flux
        * differentiate_on_depth_grid(column_mass, rosseland_opacity)
        / rosseland_opacity
    )
    return {
        "column_mass": column_mass,
        "temperature_k": temperature,
        "rosseland_optical_depth": rosseland_optical_depth,
        "rosseland_opacity": rosseland_opacity,
        "target_integrated_eddington_flux": np.asarray([target_flux], np.float64),
        "effective_temperature": np.asarray([effective_temperature], np.float64),
        "integrated_eddington_flux": integrated_flux,
        "mean_intensity_minus_source_integral": (
            -0.012 * target_flux * rosseland_opacity * np.exp(-rosseland_optical_depth)
        ),
        "absorption_heating_derivative": absorption_heating_derivative,
        "diagonal_lambda_accumulator": -target_flux * rosseland_opacity / 220.0,
        "previous_temperature_correction": np.linspace(
            25.0, -15.0, layer_count, dtype=np.float64
        ),
        "convective_flux": convective_flux,
        "previous_convective_flux": np.zeros(layer_count, dtype=np.float64),
        "logarithmic_temperature_pressure_gradient": np.full(
            layer_count, 0.30, dtype=np.float64
        ),
        "adiabatic_gradient": np.full(layer_count, 0.25, dtype=np.float64),
        "pressure_scale_height": np.full(layer_count, 1.0e8, dtype=np.float64),
        "total_pressure": 1.0e4 * column_mass + 100.0,
        "mass_density": np.full(layer_count, 1.0e-7, dtype=np.float64),
        "log_density_temperature_derivative_at_constant_total_pressure": np.full(
            layer_count, -1.0, dtype=np.float64
        ),
        "heat_capacity": np.full(layer_count, 1.0e8, dtype=np.float64),
        "integrated_radiation_pressure": 0.02 * 1.0e4 * column_mass,
        "turbulent_pressure": np.zeros(layer_count, dtype=np.float64),
        "lookup_temperature_k": np.stack(
            [0.85 * temperature, temperature, 1.15 * temperature]
        ),
        "lookup_gas_pressure": np.stack(
            [
                0.85 * 1.0e4 * column_mass,
                1.00 * 1.0e4 * column_mass,
                1.15 * 1.0e4 * column_mass,
            ]
        ),
        "lookup_rosseland_opacity": np.stack(
            [rosseland_opacity, rosseland_opacity, rosseland_opacity]
        ),
        "iteration_index": np.asarray([2], np.int32),
        "convection_enabled": np.asarray([1], np.int32),
        "frequency_count": np.asarray([30_000], np.int32),
        "smooth_start_layer": np.asarray([10], np.int32),
        "smooth_stop_layer": np.asarray([25], np.int32),
        "mixing_length": np.asarray([1.25], np.float64),
        "surface_gravity_cgs": np.asarray([1.0e4], np.float64),
    }


def evaluate(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Evaluate the exact mode-3 correction from the analytic inputs."""

    from payne_zero_atmosphere.temperature_correction import (
        apply_temperature_correction,
        ingest_temperature_correction_rosseland_table,
        initialize_temperature_correction_state,
    )

    layer_count = int(inputs["temperature_k"].size)
    state = initialize_temperature_correction_state(layer_count)
    for name in (
        "integrated_eddington_flux",
        "mean_intensity_minus_source_integral",
        "absorption_heating_derivative",
        "diagonal_lambda_accumulator",
        "previous_temperature_correction",
    ):
        getattr(state, name)[:] = inputs[name]
    for index in range(inputs["lookup_temperature_k"].shape[0]):
        ingest_temperature_correction_rosseland_table(
            state,
            temperature_k=inputs["lookup_temperature_k"][index],
            gas_pressure=inputs["lookup_gas_pressure"][index],
            rosseland_opacity=inputs["lookup_rosseland_opacity"][index],
        )

    zeros = np.zeros(layer_count, dtype=np.float64)
    ones = np.ones(layer_count, dtype=np.float64)
    result = apply_temperature_correction(
        state,
        mode=3,
        frequency_weight=0.0,
        column_mass=inputs["column_mass"],
        total_opacity=ones,
        monochromatic_eddington_flux=zeros,
        mean_intensity_minus_source=zeros,
        monochromatic_optical_depth=zeros,
        planck_source=zeros,
        frequency_hz=0.0,
        h_over_kt=ones,
        temperature_k=inputs["temperature_k"],
        stimulated_emission=ones,
        scattering_fraction=zeros,
        target_integrated_eddington_flux=float(
            inputs["target_integrated_eddington_flux"][0]
        ),
        effective_temperature=float(inputs["effective_temperature"][0]),
        frequency_count=int(inputs["frequency_count"][0]),
        rosseland_optical_depth=inputs["rosseland_optical_depth"],
        rosseland_opacity=inputs["rosseland_opacity"],
        iteration_index=int(inputs["iteration_index"][0]),
        convection_enabled=int(inputs["convection_enabled"][0]),
        convective_flux=inputs["convective_flux"],
        previous_convective_flux=inputs["previous_convective_flux"],
        logarithmic_temperature_pressure_gradient=inputs[
            "logarithmic_temperature_pressure_gradient"
        ],
        adiabatic_gradient=inputs["adiabatic_gradient"],
        pressure_scale_height=inputs["pressure_scale_height"],
        total_pressure=inputs["total_pressure"],
        mass_density=inputs["mass_density"],
        log_density_temperature_derivative_at_constant_total_pressure=inputs[
            "log_density_temperature_derivative_at_constant_total_pressure"
        ],
        heat_capacity=inputs["heat_capacity"],
        mixing_length=float(inputs["mixing_length"][0]),
        smooth_start_layer=int(inputs["smooth_start_layer"][0]),
        smooth_stop_layer=int(inputs["smooth_stop_layer"][0]),
        integrated_radiation_pressure=inputs["integrated_radiation_pressure"],
        turbulent_pressure=inputs["turbulent_pressure"],
        surface_gravity_cgs=float(inputs["surface_gravity_cgs"][0]),
    )
    if result is None:
        raise RuntimeError("mode 3 returned no correction result")
    outputs = {
        name: np.asarray(getattr(result, name))
        for name in result.__dataclass_fields__
    }
    outputs["state_previous_temperature_correction"] = np.asarray(
        state.previous_temperature_correction
    )
    outputs["state_rosseland_entry_count"] = np.asarray(
        [state.rosseland_opacity_table.entry_count], dtype=np.int32
    )
    return outputs


def array_manifest(path: Path) -> dict[str, dict[str, object]]:
    """Return shape, dtype, and content identity for every NPZ member."""

    with np.load(path, allow_pickle=False) as archive:
        return {
            name: {
                "shape": list(archive[name].shape),
                "dtype": str(archive[name].dtype),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(archive[name]).tobytes()
                ).hexdigest(),
            }
            for name in sorted(archive.files)
        }


def main() -> None:
    """Write deterministic artifacts and their local role manifest."""

    for relative, expected in PINNED_SOURCE_SHA256.items():
        path = REPOSITORY_ROOT / relative
        if sha256(path) != expected:
            raise RuntimeError(f"pinned Chapter 13 source changed: {relative}")

    inputs = analytic_inputs()
    outputs = evaluate(inputs)
    write_npz(FIXTURE_PATH, inputs)
    write_npz(GOLDEN_PATH, outputs)
    manifest = {
        "schema_version": 1,
        "payne_zero_commit": PINNED_PAYNE_ZERO_COMMIT,
        "source_sha256": PINNED_SOURCE_SHA256,
        "artifacts": {
            str(FIXTURE_PATH.relative_to(REPOSITORY_ROOT)): {
                "role": "analytic correction input fixture; contains no answer",
                "sha256": sha256(FIXTURE_PATH),
                "arrays": array_manifest(FIXTURE_PATH),
            },
            str(GOLDEN_PATH.relative_to(REPOSITORY_ROOT)): {
                "role": "comparison-only exact mode-3 correction output",
                "sha256": sha256(GOLDEN_PATH),
                "arrays": array_manifest(GOLDEN_PATH),
            },
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {FIXTURE_PATH.relative_to(REPOSITORY_ROOT)}")
    print(f"wrote {GOLDEN_PATH.relative_to(REPOSITORY_ROOT)}")
    print(f"wrote {MANIFEST_PATH.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
