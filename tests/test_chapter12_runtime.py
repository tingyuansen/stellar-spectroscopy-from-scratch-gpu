"""Focused source, reduction, restoration, and book gates for Chapter 12."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from book.chapter12_runtime import (
    FIXED_CHUNK_COUNT,
    FREQUENCY_COUNT,
    LAYER_COUNT,
    REDUCTION_OUTPUT_NAMES,
    TRANSFER_TABLES_SHA256,
    chapter11_finalization_checkpoint,
    chapter11_handoff_checkpoint,
    configure_chapter12_runtime,
    convection_checkpoint,
    finalization_checkpoint,
    input_checkpoint,
    one_frequency_checkpoint,
    persistence_checkpoint,
    reduction_checkpoint,
    restore_checkpoint,
)
from book.chapter12_teaching import (
    central_logical_derivative,
    chunk_ledger,
    physical_flux_from_eddington_flux,
    require_all_or_none,
)
from book.chapters.chapter_12 import build_notebook


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_pinned_source_verifier_recomputes_local_identities() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/verify_chapter12_source.py"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Chapter 12 source: verified" in completed.stdout


def test_input_handoff_has_exact_axes_count_and_dtypes() -> None:
    checkpoint = input_checkpoint()
    assert checkpoint.layer_count == LAYER_COUNT == 6
    assert checkpoint.frequency_count == FREQUENCY_COUNT == 30_000
    assert checkpoint.wavelength_increases
    assert checkpoint.frequency_decreases
    assert checkpoint.weights_positive
    assert checkpoint.planck_shape == (FREQUENCY_COUNT, LAYER_COUNT)
    assert checkpoint.continuum_shape == (LAYER_COUNT, FREQUENCY_COUNT)
    assert checkpoint.line_dtype == "float32"
    assert checkpoint.transfer_table_sha256 == TRANSFER_TABLES_SHA256


def test_exact_chapter11_state_reaches_chapter12_finalization() -> None:
    handoff = chapter11_handoff_checkpoint()
    assert handoff.opacity_state_type == "OpacityState"
    assert handoff.depth_count == 80
    assert handoff.frequency_count == FREQUENCY_COUNT
    assert handoff.continuum_shape == (80, FREQUENCY_COUNT)
    assert handoff.line_shape == (80, FREQUENCY_COUNT)
    assert handoff.continuum_dtype == "float64"
    assert handoff.line_dtype == "float32"
    assert handoff.molecules_enabled
    assert handoff.selected_line_count > 0
    assert handoff.source_is_chapter11_cache

    finalization = chapter11_finalization_checkpoint()
    assert finalization.finalization_type == "IterationFinalization"
    assert finalization.correction_type == "TemperatureCorrectionResult"
    assert finalization.depth_count == 80
    assert finalization.frequency_count == FREQUENCY_COUNT
    for values in (
        finalization.rosseland_opacity,
        finalization.rosseland_optical_depth,
        finalization.integrated_eddington_flux,
        finalization.radiative_acceleration,
        finalization.correction_temperature,
        finalization.correction_column_mass,
    ):
        assert values.shape == (80,)
        assert np.all(np.isfinite(values))
    assert np.all(finalization.rosseland_opacity > 0.0)
    assert np.all(np.diff(finalization.rosseland_optical_depth) > 0.0)
    assert finalization.correction_result_finite
    assert finalization.correction_column_mass_positive
    assert finalization.correction_column_mass_strictly_increasing
    assert finalization.rosseland_aliases_accumulator
    assert finalization.radiative_state_aliases_accumulator
    assert finalization.lookup_entry_count == 80
    assert finalization.source_is_chapter11_cache


def test_one_frequency_readable_and_compiled_deposits_are_exact() -> None:
    checkpoint = one_frequency_checkpoint()
    assert checkpoint.frequency_index == 15_400
    assert checkpoint.output_names == REDUCTION_OUTPUT_NAMES
    assert len(checkpoint.compiled_outputs) == 9
    for compiled, helper in zip(
        checkpoint.compiled_outputs,
        checkpoint.helper_outputs,
        strict=True,
    ):
        np.testing.assert_array_equal(compiled, helper)
    assert checkpoint.maximum_absolute_difference == 0.0
    assert checkpoint.maximum_relative_difference == 0.0
    np.testing.assert_array_equal(
        checkpoint.compiled_outputs[2],
        checkpoint.compiled_outputs[7],
    )
    assert not checkpoint.post_guard_identity_holds


def test_full_frequency_reduction_is_repeatable_with_measured_grouping_delta() -> None:
    checkpoint = reduction_checkpoint()
    assert checkpoint.frequency_count == FREQUENCY_COUNT
    assert checkpoint.layer_count == LAYER_COUNT
    assert checkpoint.chunk_count == FIXED_CHUNK_COUNT
    np.testing.assert_array_equal(
        checkpoint.bounds,
        np.asarray([0, 15_000, 30_000], dtype=np.int64),
    )
    assert checkpoint.private_bytes == (
        8 * FIXED_CHUNK_COUNT * (8 * LAYER_COUNT + 1)
    )
    assert checkpoint.fixed_policy_repeatable
    for first, repeated in zip(
        checkpoint.fixed_chunk_outputs,
        checkpoint.repeated_outputs,
        strict=True,
    ):
        np.testing.assert_array_equal(first, repeated)
        assert np.all(np.isfinite(first))
    assert checkpoint.maximum_absolute_difference > 0.0
    assert checkpoint.maximum_relative_difference < 2.0e-14


def test_mode1_persistence_and_exact_mode3_finalization() -> None:
    persistence = persistence_checkpoint()
    assert persistence.reset_names == (
        "mean_intensity_minus_source_integral",
        "absorption_heating_derivative",
        "diagonal_lambda_accumulator",
        "integrated_eddington_flux",
    )
    assert persistence.reset_arrays_zero
    assert persistence.previous_correction_preserved
    assert persistence.table_identity_preserved
    assert persistence.table_entry_count_preserved
    assert persistence.radiative_mode1_preserves_final_pressures

    final = finalization_checkpoint()
    assert final.rosseland_opacity.shape == (LAYER_COUNT,)
    assert np.all(np.isfinite(final.rosseland_opacity))
    assert np.all(final.rosseland_opacity > 0.0)
    assert np.all(np.diff(final.rosseland_optical_depth) > 0.0)
    for values in (
        final.integrated_eddington_flux,
        final.radiation_energy_density,
        final.radiative_acceleration,
        final.integrated_radiation_pressure,
        final.absolute_radiation_pressure,
    ):
        assert values.shape == (LAYER_COUNT,)
        assert np.all(np.isfinite(values))
    assert np.isfinite(final.surface_radiation_pressure_constant)
    assert final.correction_result_finite
    assert final.rosseland_aliases_accumulator
    assert final.radiative_state_aliases_accumulator
    assert final.lookup_entry_count == LAYER_COUNT
    assert final.correction_field_names == (
        "temperature",
        "flux_error_percent",
        "flux_derivative",
        "flux_temperature_derivative",
        "lambda_temperature_derivative",
        "surface_temperature_derivative",
        "temperature_correction",
        "flux_ratio",
        "convective_flux",
        "column_mass",
        "column_mass_correction",
    )


def test_atomic_perturbation_restore_audit_exposes_only_source_deltas() -> None:
    checkpoint = restore_checkpoint()
    assert checkpoint.sample_field_names == (
        "specific_internal_energy_plus_temperature",
        "specific_internal_energy_minus_temperature",
        "specific_internal_energy_plus_pressure",
        "specific_internal_energy_minus_pressure",
        "density_plus_temperature",
        "density_minus_temperature",
        "density_plus_pressure",
        "density_minus_pressure",
    )
    assert checkpoint.temperature_restored
    assert checkpoint.gas_pressure_restored
    assert checkpoint.electron_density_restored
    assert checkpoint.total_nuclei_density_restored
    assert checkpoint.mass_density_restored
    assert checkpoint.populations_restored
    assert checkpoint.cache_restored
    assert not checkpoint.charge_square_density_restored
    assert checkpoint.zero_energy_replaced
    assert np.all(np.isfinite(checkpoint.density_temperature_derivative))
    assert np.all(np.isfinite(checkpoint.energy_temperature_derivative))
    assert np.any(checkpoint.density_temperature_derivative != 0.0)


def test_convection_fields_suppression_and_disabled_diagnostic_are_distinct() -> None:
    checkpoint = convection_checkpoint()
    assert checkpoint.result_field_names == (
        "geometric_depth_below_surface_km",
        "logarithmic_temperature_pressure_gradient",
        "heat_capacity",
        "log_density_temperature_derivative_at_constant_total_pressure",
        "sound_speed",
        "adiabatic_gradient",
        "pressure_scale_height",
        "convective_flux",
        "convective_velocity",
        "raw_convective_flux",
        "overshoot_convective_flux",
    )
    assert checkpoint.geometric_depth_below_surface_km[0] == 0.0
    assert np.all(np.diff(checkpoint.geometric_depth_below_surface_km) > 0.0)
    for values in (
        checkpoint.logarithmic_gradient,
        checkpoint.adiabatic_gradient,
        checkpoint.heat_capacity,
        checkpoint.raw_convective_flux,
        checkpoint.returned_convective_flux,
        checkpoint.convective_velocity,
        checkpoint.disabled_flux,
        checkpoint.disabled_velocity,
    ):
        assert values.shape == (LAYER_COUNT,)
        assert np.all(np.isfinite(values))
    assert checkpoint.first_two_suppressed
    assert checkpoint.standard_six_layer_fixture_fully_suppressed
    assert checkpoint.disabled_can_be_nonzero
    assert np.any(checkpoint.disabled_flux != 0.0)
    assert np.any(checkpoint.disabled_velocity != 0.0)


def test_teaching_ledgers_enforce_exact_bookkeeping_boundaries() -> None:
    ledger = chunk_ledger(start=3, stop=22, chunk_count=6, layer_count=7)
    np.testing.assert_array_equal(
        ledger.bounds,
        np.asarray([3, 6, 9, 12, 15, 18, 22]),
    )
    assert ledger.private_depth_array_count == 8
    assert ledger.private_surface_array_count == 1
    assert ledger.private_bytes == 8 * 6 * (8 * 7 + 1)

    coordinate = np.asarray([2.0, 4.0, 8.0])
    plus = 1.001 * coordinate**2
    minus = 0.999 * coordinate**2
    np.testing.assert_allclose(
        central_logical_derivative(plus, minus, coordinate),
        coordinate,
        rtol=2.0e-13,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        physical_flux_from_eddington_flux(np.asarray([1.0, 2.0])),
        4.0 * np.pi * np.asarray([1.0, 2.0]),
    )
    assert not require_all_or_none(None, None)
    assert require_all_or_none(np.ones(1), np.ones(1))
    with pytest.raises(ValueError, match="all eight"):
        require_all_or_none(np.ones(1), None)


def test_exact_extracted_interfaces_and_notebook_twenty_cell_spine() -> None:
    configure_chapter12_runtime()
    from payne_zero_atmosphere.convection import (
        ConvectionFiniteDifferenceSamples,
        ConvectionResult,
    )
    from payne_zero_atmosphere.runner import (
        IterationFinalization,
        TransferAccumulation,
    )

    assert tuple(field.name for field in fields(TransferAccumulation)) == (
        "opacity_state",
        "frequency_start_index",
        "frequency_stop_index",
        "rosseland_accumulator",
        "radiative_pressure_state",
        "temperature_correction_state",
    )
    assert tuple(field.name for field in fields(IterationFinalization)) == (
        "transfer_accumulation",
        "rosseland_opacity",
        "rosseland_optical_depth",
        "radiative_pressure_state",
        "temperature_correction_result",
        "convection_result",
        "convection_finite_difference_samples",
    )
    assert len(fields(ConvectionFiniteDifferenceSamples)) == 8
    assert len(fields(ConvectionResult)) == 11

    document = build_notebook()
    visible = [
        cell
        for cell in document["cells"]
        if cell["cell_type"] == "code"
        and "hide-input" not in cell.get("metadata", {}).get("tags", ())
        and "book-setup" not in cell.get("metadata", {}).get("tags", ())
    ]
    assert len(visible) == 21
    markdown_source = "\n".join(
        cell["source"]
        for cell in document["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "# Chapter 12" in markdown_source
    assert "## 12.24 Chapter summary" in markdown_source
    assert "/reader.html?ch=13" in markdown_source
    assert "/Users/ysting/payne-zero" not in markdown_source
    assert "```python" not in markdown_source
