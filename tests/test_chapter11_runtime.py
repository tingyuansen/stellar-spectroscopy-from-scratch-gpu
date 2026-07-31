"""Focused source, state, and chapter gates for Chapter 11."""

from __future__ import annotations

from dataclasses import fields
import hashlib
from pathlib import Path

import numpy as np

from book.chapter11_runtime import (
    CONTINUUM_LEVEL_TABLES,
    INPUT_HASHES,
    blanketing_checkpoint,
    configure_chapter11_runtime,
    continuum_checkpoint,
    hydrostatic_checkpoint,
    opacity_pass_checkpoint,
    population_checkpoint,
    quantization_checkpoint,
    remap_checkpoint,
    reuse_checkpoint,
    sampling_grid_checkpoint,
    seed_checkpoint,
    selection_checkpoint,
    setup_checkpoint,
)
from book.chapter11_teaching import opacity_grid_start_index, opacity_memory_ledger
from book.chapters.chapter_11 import build_notebook
from book.registry import BY_NUMBER


configure_chapter11_runtime()

from payne_zero_atmosphere.continuum_opacity import ContinuumAtmosphereState
from payne_zero_atmosphere.runner import OpacityState


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_owned_inputs_and_small_required_table_are_exact() -> None:
    for path, expected in INPUT_HASHES.items():
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert CONTINUUM_LEVEL_TABLES.stat().st_size == 9998
    with np.load(CONTINUUM_LEVEL_TABLES, allow_pickle=False) as archive:
        assert len(archive.files) == 26
        assert archive["element_block_offsets"].shape == (10,)


def test_seed_validation_setup_and_microturbulence_boundaries() -> None:
    seed = seed_checkpoint()
    assert seed.layers == 80
    assert seed.abundance_count == 99
    assert seed.column_mass_strictly_increasing
    assert seed.field_names == (
        "column_mass",
        "temperature",
        "gas_pressure",
        "electron_density",
        "rosseland_opacity",
        "radiative_acceleration",
        "microturbulence",
        "convective_flux",
        "convective_velocity",
    )
    assert len(seed.error_cases) == len(seed.error_messages) == 4
    assert all(message.startswith("atmosphere seed") for message in seed.error_messages)

    setup = setup_checkpoint()
    assert setup.iterations == 1
    assert setup.effective_temperature == 5778.0
    assert setup.log_surface_gravity == 4.44
    assert len(setup.opacity_flags) == 20
    assert setup.opacity_flags[14] == 1
    assert setup.opacity_flags[16] == 0
    assert setup.molecules_enabled
    assert setup.pressure_iteration_enabled
    assert setup.standard_rosseland_optical_depth.shape == (80,)
    np.testing.assert_allclose(
        np.diff(np.log10(setup.standard_rosseland_optical_depth)),
        0.125,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.all(setup.all_zero_profile_after > 0.0)
    assert setup.partly_positive_unchanged
    assert np.count_nonzero(setup.partly_positive_profile_after) == 1


def test_hydrostatic_quantization_and_scalar_remap_are_explicit() -> None:
    hydro = hydrostatic_checkpoint()
    np.testing.assert_allclose(
        hydro.balance_residual,
        0.0,
        rtol=0.0,
        atol=2.0e-10,
    )
    assert hydro.pressure_constant == 0.0
    assert hydro.warning_message.startswith("Hydrostatic P non-positive")
    assert hydro.warning_floor_positive

    quantized = quantization_checkpoint()
    assert quantized.maximum_absolute_delta.shape == (9,)
    assert np.any(quantized.maximum_absolute_delta > 0.0)
    assert quantized.second_application_bitwise_equal

    remapped = remap_checkpoint()
    interior = (
        (remapped.target_grid >= remapped.source_grid[0])
        & (remapped.target_grid <= remapped.source_grid[-1])
    )
    assert np.all(remapped.constant_remap[interior] == 3.5)
    assert remapped.constant_final_source_interval_index == 8
    assert remapped.monotone_final_source_interval_index == 8


def test_strict_sampling_grid_and_memory_ledger() -> None:
    grid = sampling_grid_checkpoint()
    assert grid.wavelength_nm.shape == (30_000,)
    assert grid.frequency_hz.shape == (30_000,)
    assert grid.frequency_weights.shape == (30_000,)
    assert grid.wavelength_increasing
    assert grid.frequency_decreasing
    assert np.all(grid.frequency_weights > 0.0)
    assert opacity_grid_start_index(30_000.0) == 1
    assert opacity_grid_start_index(np.nextafter(30_000.0, -np.inf)) == 3_577
    assert opacity_grid_start_index(13_000.0) == 3_577
    assert opacity_grid_start_index(np.nextafter(13_000.0, -np.inf)) == 7_027
    assert opacity_grid_start_index(7_250.0) == 7_027
    assert opacity_grid_start_index(np.nextafter(7_250.0, -np.inf)) == 9_599
    assert opacity_grid_start_index(4_500.0) == 9_599
    assert opacity_grid_start_index(np.nextafter(4_500.0, -np.inf)) == 11_601

    ledger = opacity_memory_ledger()
    assert ledger.one_float64_slab_bytes == 80 * 30_000 * 8
    assert ledger.one_float32_slab_bytes == 80 * 30_000 * 4
    assert ledger.complete_four_slab_bytes == (
        3 * ledger.one_float64_slab_bytes + ledger.one_float32_slab_bytes
    )


def test_complete_molecule_enabled_80_by_30000_pretransfer_state() -> None:
    population = population_checkpoint()
    assert population.layers == 80
    assert population.packed_slot_count == 1006
    assert population.molecular_state_present
    assert population.molecule_count > 0
    assert population.fractional_doppler_widths.shape == (80, 1006)
    assert population.line_strength_population_factors.shape == (80, 1006)
    assert np.all(np.isfinite(population.mass_density))
    assert np.all(population.mass_density > 0.0)

    continuum = continuum_checkpoint()
    assert continuum.adapter_field_names == tuple(
        field.name for field in fields(ContinuumAtmosphereState)
    )
    for values in (continuum.absorption, continuum.scattering, continuum.source):
        assert values.shape == (80, 30_000)
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))
    assert np.all(continuum.absorption >= 0.0)
    assert np.all(continuum.scattering >= 0.0)
    assert continuum.threshold.shape == (80, 344)
    assert continuum.threshold.dtype == np.float32
    assert continuum.reference_wavelength_nm[343] == (
        continuum.reference_wavelength_nm[342]
    )
    assert continuum.packed_reference_indices[343] == 2**30

    selected = selection_checkpoint()
    assert selected.selected_line_count > 0
    assert selected.detailed_line_count == 64
    assert not selected.detailed_accumulation_enabled
    assert selected.contributing_line_count > 0
    assert selected.line_mass_absorption_coefficient.shape == (80, 30_000)
    assert selected.line_opacity_dtype == "float32"
    assert np.all(np.isfinite(selected.line_mass_absorption_coefficient))
    assert np.all(selected.line_mass_absorption_coefficient >= 0.0)
    assert np.any(selected.line_mass_absorption_coefficient > 0.0)

    opacity = opacity_pass_checkpoint()
    assert opacity.opacity_state_field_names == tuple(
        field.name for field in fields(OpacityState)
    )
    assert opacity.continuum_shape == opacity.line_shape == (80, 30_000)
    assert opacity.rosseland_entry_count == 0
    assert not opacity.schema_v4_product

    blanketing = blanketing_checkpoint()
    assert blanketing.wavelength_nm.shape == (1000,)
    assert np.all(blanketing.blanketed_extinction >= blanketing.continuum_extinction)
    assert blanketing.line_to_continuum_peak_ratio > 1.0

    reused = reuse_checkpoint()
    assert reused.selected_catalog_same_object
    assert reused.detailed_branch_inactive
    assert reused.temperature_changed
    assert reused.line_opacity_changed
    assert reused.maximum_line_opacity_change > 0.0


def test_chapter_has_exactly_18_visible_cells_and_stops_before_transfer() -> None:
    document = build_notebook()
    visible = [
        cell
        for cell in document["cells"]
        if cell["cell_type"] == "code"
        and "hide-input" not in cell.get("metadata", {}).get("tags", [])
    ]
    markdown_text = "\n".join(
        cell["source"]
        for cell in document["cells"]
        if cell["cell_type"] == "markdown"
    )
    code_text = "\n".join(cell["source"] for cell in document["cells"])
    assert len(visible) == 19
    assert "## 11.14 Chapter summary" in markdown_text
    assert "/reader.html?ch=12" in markdown_text
    assert "exercise" not in markdown_text.lower()
    assert "accumulate_transfer_state(" not in code_text
    assert "finalize_transfer_state(" not in code_text
    assert "save_product_structured_atmosphere(" not in code_text
    assert BY_NUMBER[11].available
