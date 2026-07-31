"""Focused scientific and pedagogical gates for Chapter 8."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from book.chapter08_runtime import (
    COMPILED_FIELDS,
    LOCAL_INPUT_HASHES,
    atmosphere_family_checkpoint,
    cache_checkpoint,
    catalog_checkpoint,
    doppler_checkpoint,
    feature_status_rows,
    invariant_checkpoint,
    manifest_rug_checkpoint,
    molecular_opacity_checkpoint,
    packed_compiler_checkpoint,
    population_and_band_checkpoint,
    source_checkpoint,
    sparse_oracle_checkpoint,
    text_compiler_checkpoint,
    text_record_checkpoint,
)
from book.chapters.chapter_08 import build_notebook
from book.registry import BY_NUMBER


def test_local_source_inputs_are_hash_bound_and_small() -> None:
    checkpoint = source_checkpoint()
    assert checkpoint.band_count == 32
    assert checkpoint.full_text_record_count == 22_377_706
    assert checkpoint.full_tio_record_count == 37_744_499
    assert checkpoint.full_h2o_record_count == 65_912_356
    assert checkpoint.full_diatomic_record_count == 12_488_322
    assert checkpoint.compact_text_record_count == 220
    assert checkpoint.compact_tio_record_count == 64
    assert checkpoint.compact_h2o_record_count == 28
    assert checkpoint.compact_diatomic_record_count == 96
    assert len(checkpoint.local_hashes) == len(LOCAL_INPUT_HASHES)
    assert all(path.is_file() for path in LOCAL_INPUT_HASHES)
    assert all("/Users/ysting/payne-zero" not in str(path) for path in LOCAL_INPUT_HASHES)


def test_species_codes_map_to_exact_stage5_columns_and_co_band() -> None:
    checkpoint = population_and_band_checkpoint()
    np.testing.assert_array_equal(checkpoint.species_code, [240, 276, 366, 534])
    np.testing.assert_array_equal(checkpoint.population_column, [39, 45, 60, 88])
    assert checkpoint.co_wavelength_nm.shape == (6,)
    assert np.all(checkpoint.co_classical_line_strength > 0.0)
    assert np.all(checkpoint.co_population_cm3 > 0.0)


def test_text_parser_dispatch_has_pair_fallback_and_true_rejection() -> None:
    checkpoint = text_record_checkpoint()
    assert checkpoint.stored_wavelength_nm == 1000.0
    assert checkpoint.energy_wavelength_nm == 1000.0
    assert checkpoint.fallback_wavelength_nm == 1000.0
    assert checkpoint.pair_dispatch_species_code == 276
    assert checkpoint.code_fallback_species_code == 276
    assert checkpoint.missing_dispatch_is_none


def test_scalar_and_serial_numba_text_compilers_are_record_exact() -> None:
    checkpoint = text_compiler_checkpoint()
    assert checkpoint.scalar_line_count == 93
    assert checkpoint.compiled_line_count == 93
    assert checkpoint.band_order_exact
    assert set(checkpoint.field_exact) == set(COMPILED_FIELDS)
    assert all(checkpoint.field_exact.values())
    assert checkpoint.dtypes == {
        "center_index_1based": "int32",
        "classical_line_strength": "float32",
        "species_code": "int16",
        "lower_excitation_cm": "float32",
        "radiative_damping": "float32",
        "stark_damping": "float32",
        "van_der_waals_damping": "float32",
        "margin_class": "int16",
    }


def test_fresh_text_build_preserves_manifest_concatenation() -> None:
    checkpoint = manifest_rug_checkpoint()
    assert checkpoint.concatenation_exact
    assert checkpoint.line_count_by_band.shape == (32,)
    assert checkpoint.line_count_by_band.sum() == 93
    assert checkpoint.wavelength_nm.size == 93
    assert np.all(np.diff(checkpoint.manifest_index) >= 0)


def test_packed_compilers_keep_family_owned_semantics() -> None:
    checkpoint = packed_compiler_checkpoint()
    np.testing.assert_array_equal(np.unique(checkpoint.tio_isotope_index), [1, 2, 3, 4, 5])
    np.testing.assert_allclose(
        checkpoint.tio_isotope_fraction,
        [0.0793, 0.0728, 0.7394, 0.0551, 0.0534],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(np.unique(checkpoint.h2o_isotope_index), [0, 1, 2, 3])
    np.testing.assert_allclose(
        checkpoint.h2o_isotope_fraction,
        [0.9976, 0.0004, 0.0020, 0.00001],
        rtol=0.0,
        atol=0.0,
    )
    assert checkpoint.tio_line_count == 64
    assert checkpoint.h2o_line_count == 28
    assert checkpoint.fractional_coefficient_difference == (
        0.01502 / (0.026538 / 1.77245) - 1.0
    )
    assert checkpoint.tio_air_changed_fields == (
        "center_index_1based",
        "classical_line_strength",
        "radiative_damping",
        "stark_damping",
        "van_der_waals_damping",
    )
    assert checkpoint.h2o_air_changed_fields == (
        "center_index_1based",
        "radiative_damping",
    )


def test_atmosphere_family_corrections_and_routing_are_separate() -> None:
    checkpoint = atmosphere_family_checkpoint()
    np.testing.assert_array_equal(checkpoint.source_count, [96, 64, 28, 2])
    np.testing.assert_array_equal(checkpoint.selected_count, [96, 64, 28, 2])
    np.testing.assert_array_equal(
        checkpoint.offset_examples["TiO"],
        [-1272, -1259, -1138, -1101, -131],
    )
    np.testing.assert_array_equal(
        checkpoint.water_packed_species,
        [-9403, -9402, -9401, -9400],
    )
    assert not checkpoint.h3plus_probe_is_scientific_catalog

    status = {row["family"]: row for row in feature_status_rows()}
    assert status["H2O"]["atmosphere_default_source"]
    assert status["H2O"]["synthesis_compiler_exists"]
    assert not status["H2O"]["synthesis_standard_deposits"]
    assert status["H3+"]["atmosphere_selector_exists"]
    assert not status["H3+"]["atmosphere_default_source"]
    assert not status["H3+"]["synthesis_compiler_exists"]


def test_standard_catalog_is_text_then_tio_with_no_water() -> None:
    checkpoint = catalog_checkpoint()
    assert checkpoint.text_line_count == 93
    assert checkpoint.tio_line_count == 64
    assert checkpoint.combined_line_count == 157
    assert checkpoint.first_tio_row == 93
    assert checkpoint.concatenation_exact
    assert checkpoint.center_reconstruction_max_abs_nm < 2.0e-5
    assert 534 not in checkpoint.species_codes
    np.testing.assert_array_equal(
        checkpoint.species_population_columns,
        checkpoint.species_codes // 6 - 1,
    )


def test_derived_cache_reloads_and_unreadable_bytes_rebuild() -> None:
    checkpoint = cache_checkpoint()
    assert checkpoint.cache_file_count == 1
    assert checkpoint.first_reload_exact
    assert checkpoint.cached_reload_exact
    assert checkpoint.corrupt_cache_rebuilt_exact
    assert not checkpoint.combined_persistent_cache_fingerprints_manifest


def test_invariants_and_doppler_state_have_native_shapes() -> None:
    invariants = invariant_checkpoint()
    assert len(invariants.field_names) == 17
    assert all(device == "cpu" for device in invariants.devices if device != "host scalar")
    assert 299_999.9 < invariants.local_resolving_power_min < 300_000.1
    assert 299_999.9 < invariants.local_resolving_power_max < 300_000.1

    doppler = doppler_checkpoint()
    assert doppler.doppler_fraction.shape == (
        6,
        doppler.species_code.size,
    )
    assert doppler.population_doppler_ratio.shape == doppler.doppler_fraction.shape
    assert doppler.thermal_width_heavier_is_smaller
    assert doppler.microturbulent_widths_converge
    assert np.all(np.isfinite(doppler.population_doppler_ratio))
    assert np.all(doppler.population_doppler_ratio >= 0.0)


def test_sparse_oracle_and_complete_opacity_checks() -> None:
    oracle = sparse_oracle_checkpoint()
    np.testing.assert_array_equal(oracle.center_index_1based, [2, 2])
    np.testing.assert_array_equal(oracle.dense_sum, oracle.sparse_sum)
    assert oracle.maximum_absolute_difference == 0.0
    assert oracle.maximum_relative_difference == 0.0

    opacity = molecular_opacity_checkpoint()
    assert opacity.text_opacity.shape == (6, 121)
    assert opacity.tio_opacity.shape == (6, 121)
    assert opacity.combined_opacity.shape == (6, 121)
    assert opacity.combined_opacity.dtype == np.float32
    assert np.all(np.isfinite(opacity.combined_opacity))
    assert np.all(opacity.combined_opacity >= 0.0)
    assert opacity.separate_sum_max_abs <= 2.0e-8
    assert opacity.chunk_regrouping_max_abs <= 6.0e-8
    assert opacity.stimulation_ratio_max_abs == 0.0
    assert opacity.compiler_only_h2o_line_count == 28
    assert opacity.standard_h2o_line_count == 0
    assert np.all(np.diff(opacity.integrated_opacity) > 0.0)


def test_chapter_has_17_visible_code_cells_and_causal_close() -> None:
    document = build_notebook()
    cells = document["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    visible_code = [
        cell
        for cell in code_cells
        if "hide-input" not in cell.get("metadata", {}).get("tags", [])
    ]
    markdown_text = "\n".join(
        cell["source"] for cell in cells if cell["cell_type"] == "markdown"
    )
    assert len(visible_code) == 17
    assert "## 8.18 Chapter summary" in markdown_text
    assert "/reader.html?ch=9" in markdown_text
    assert "detached exercise" not in markdown_text.lower()
    assert BY_NUMBER[8].available
    assert BY_NUMBER[8].title == "Molecular Bands and Source Compilation"
    for path in (
        "ch08-one-molecule-band-v1.png",
        "ch08-encodings-to-record-v1.png",
        "ch08-host-to-device-v1.png",
        "ch08-two-lanes-v1.png",
    ):
        assert (
            Path("assets/schematics/textbook") / path
        ).is_file()
