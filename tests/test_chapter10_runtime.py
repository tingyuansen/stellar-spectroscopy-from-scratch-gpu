"""Focused source, systems, and scientific gates for Chapter 10."""

from __future__ import annotations

from dataclasses import fields
import inspect
from pathlib import Path

import numpy as np
import torch

from book.chapter10_runtime import (
    REGIMES,
    cache_checkpoint,
    configure_chapter10_runtime,
    end_to_end_checkpoint,
    fixture_checkpoint,
    four_regime_checkpoint,
    grid_checkpoint,
    hydrogen_checkpoint,
    invariant_checkpoint,
    memory_checkpoint,
    public_spectrum,
    roundtrip_checkpoint,
    runtime_policy_checkpoint,
    timing_checkpoint,
)
from book.chapter10_teaching import allocation_ledger
from book.chapters.chapter_10 import build_notebook
from book.registry import BY_NUMBER


configure_chapter10_runtime()

from payne_zero_synthesis import Spectrum, synthesize
from payne_zero_synthesis.pipeline import (
    SpectrumResult,
    SynthesisPipeline,
    WindowInvariants,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_exact_public_interfaces_and_runtime_policy() -> None:
    assert tuple(field.name for field in fields(SpectrumResult)) == (
        "wavelength_nm",
        "eddington_flux_total_per_frequency",
        "eddington_flux_continuum_per_frequency",
        "normalized_flux",
        "continuum_absorption",
        "continuum_scattering",
        "line_mass_absorption_coefficient",
        "line_source",
        "spectral_operator_seconds",
        "spectral_operator_name",
    )
    assert tuple(field.name for field in fields(Spectrum)) == (
        "wavelength_nm",
        "flux_total",
        "flux_continuum",
        "normalized_flux",
        "seconds",
    )
    assert tuple(field.name for field in fields(WindowInvariants)) == (
        "key",
        "device",
        "dtype",
        "molecular_lines",
        "metal_chunk",
        "grid_obj",
        "synthesis_wavelength_nm",
        "wavelength_nm",
        "output_slice",
        "n_synthesis_wl",
        "n_wl",
        "n_atomic",
        "atomic_kernel_catalog",
        "has_metal",
        "has_helium",
        "has_hydrogen",
        "continuum_tables",
        "transfer_tables",
        "metal_invariant_chunks",
        "helium_invariants",
        "hydrogen_invariants_template",
        "molecular_invariants",
        "n_molecular",
        "build_profile",
    )
    assert str(inspect.signature(SynthesisPipeline.run)) == (
        "(self, keep_slabs: 'bool' = False, spectral_operator=None) "
        "-> 'SpectrumResult'"
    )
    assert "resolution" in inspect.signature(synthesize).parameters
    assert "r_grid" not in inspect.signature(synthesize).parameters

    policy = runtime_policy_checkpoint()
    assert policy.cpu_default_dtype == "torch.float64"
    assert policy.mps_float64_rejected


def test_fixtures_grid_and_window_invariants_activate_every_family() -> None:
    fixture = fixture_checkpoint()
    assert fixture.regimes == REGIMES
    assert fixture.schema_version == 4
    assert fixture.required_array_count == 25
    assert fixture.depth_count == 6
    assert np.unique(fixture.electron_density_surface_cm3).size == len(REGIMES)

    grid = grid_checkpoint()
    assert grid.synthesis_count == grid.requested_count + 32
    assert grid.context_each_side == 16
    assert grid.output_slice == slice(16, 16 + grid.requested_count)
    assert grid.interior_bitwise_exact

    invariants = invariant_checkpoint()
    assert len(invariants.field_names) == 24
    assert invariants.atomic_line_count == 133
    assert invariants.molecular_line_count > 0
    assert all(
        (
            invariants.has_metal,
            invariants.has_helium,
            invariants.has_hydrogen,
        )
    )
    assert invariants.metal_chunk == 40_000
    assert invariants.device == "cpu"
    assert invariants.dtype == "torch.float64"


def test_process_cache_hydrogen_replacement_and_memory_boundaries() -> None:
    cache = cache_checkpoint()
    assert cache.process_hit_same_object
    assert cache.disabled_build_new_object
    assert cache.disabled_values_equal
    assert cache.perturbed_key_is_distinct
    assert cache.clear_removes_entries

    hydrogen = hydrogen_checkpoint()
    assert hydrogen.template_depth_count == 1
    assert hydrogen.hot_depth_count == fixture_checkpoint().depth_count
    assert hydrogen.cool_depth_count == fixture_checkpoint().depth_count
    assert hydrogen.hot_cool_state_differs
    assert hydrogen.template_unchanged

    memory = memory_checkpoint()
    assert memory.float32_line_slab_bytes == (
        memory.depth_count * memory.synthesis_wavelength_count * 4
    )
    assert not memory.dense_depth_line_wavelength_allocated
    assert memory.hypothetical_dense_depth_line_wavelength_bytes > (
        memory.float32_line_slab_bytes
    )
    ledger = allocation_ledger(
        depth_count=memory.depth_count,
        wavelength_count=memory.synthesis_wavelength_count,
        line_count=memory.line_count,
    )
    assert ledger.float32_line_slab_bytes == memory.float32_line_slab_bytes


def test_complete_pipeline_has_exact_order_shapes_and_flux_semantics() -> None:
    checkpoint = end_to_end_checkpoint("solar_dwarf")
    depth_count = fixture_checkpoint().depth_count
    wavelength_count = grid_checkpoint().requested_count
    assert checkpoint.continuum_absorption.shape == (depth_count, wavelength_count)
    assert checkpoint.continuum_scattering.shape == (depth_count, wavelength_count)
    assert checkpoint.line_absorption.shape == (depth_count, wavelength_count)
    assert checkpoint.line_source.shape == (depth_count, wavelength_count)
    assert checkpoint.wavelength_nm.shape == (wavelength_count,)
    assert checkpoint.stage_order == (
        "continuum absorption + scattering",
        "float32 shared line slab",
        "ordinary/autoionizing metal chunks",
        "helium",
        "hydrogen with star-specific merge state",
        "molecular text + TiO",
        "LTE Planck line source + zero line scattering",
        "total/continuum transfer",
        "device-side requested-grid crop",
        "final host result construction",
    )
    for values in (
        checkpoint.eddington_flux_total_per_frequency,
        checkpoint.eddington_flux_continuum_per_frequency,
        checkpoint.flux_total,
        checkpoint.flux_continuum,
        checkpoint.normalized_flux,
    ):
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))
    np.testing.assert_allclose(
        checkpoint.normalized_flux,
        checkpoint.flux_total / checkpoint.flux_continuum,
        rtol=3.0e-7,
        atol=0.0,
    )


def test_public_api_builder_roundtrip_timing_and_four_regimes() -> None:
    spectrum = public_spectrum("solar_dwarf")
    assert isinstance(spectrum, Spectrum)
    assert spectrum.wavelength_nm.size == grid_checkpoint().requested_count
    np.testing.assert_allclose(
        spectrum.normalized_flux,
        spectrum.flux_total / spectrum.flux_continuum,
        rtol=3.0e-7,
        atol=0.0,
    )

    roundtrip = roundtrip_checkpoint()
    assert roundtrip.saved_field_count == 25
    assert roundtrip.field_names_exact
    assert roundtrip.arrays_exact
    assert roundtrip.fixed_electron_density_seed

    timing = timing_checkpoint()
    assert timing.outputs_equal
    assert timing.seconds.shape == (3,)
    assert np.all(timing.seconds > 0.0)

    four = four_regime_checkpoint()
    assert four.regimes == REGIMES
    assert four.normalized_flux.shape == (
        len(REGIMES),
        grid_checkpoint().requested_count,
    )
    assert np.all(np.isfinite(four.normalized_flux))
    assert np.unique(four.minimum_normalized_flux).size == len(REGIMES)
    assert four.backend == "cpu"
    assert four.dtype == "torch.float64"


def test_chapter_has_18_visible_cells_no_exercises_and_causal_close() -> None:
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
    normalized_markdown = " ".join(markdown_text.split())
    assert len(visible) == 18
    assert len(markdown_text.split()) >= 3_500
    assert "## 10.12 Chapter summary" in markdown_text
    assert "/reader.html?ch=11" in markdown_text
    assert "exercise" not in markdown_text.lower()
    assert "```" not in markdown_text
    assert "every quantitative spectrum on **CPU in float64**" in normalized_markdown
    assert (
        "same synthesis implementation can select CUDA or Apple Metal"
        in normalized_markdown
    )
    assert "portable reference" in normalized_markdown
    assert BY_NUMBER[10].available


def test_chapter10_source_catalog_view_never_resolves_external_checkout() -> None:
    checkpoint = end_to_end_checkpoint("hot_dwarf")
    assert checkpoint.normalized_flux.size == grid_checkpoint().requested_count
    external_root = Path("/Users/ysting/payne-zero").resolve()
    for name, module in tuple(__import__("sys").modules.items()):
        if not name.startswith(("payne_zero_synthesis", "payne_zero_atmosphere")):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is not None:
            assert not Path(module_path).resolve().is_relative_to(external_root)
