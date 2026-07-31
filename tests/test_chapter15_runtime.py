"""Focused workflow, provenance, schema, and honesty gates for Chapter 15."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import subprocess
import sys

import numpy as np

from book.chapter15_runtime import (
    EXPECTED_CASE_NAMES,
    EXPECTED_RUNNER_SYMBOLS,
    INITIALIZER_EXPORTS_REQUIRED_BY_SYNTHESIS_API,
    dependency_checkpoint,
    direct_mixture_checkpoint,
    four_regime_capstone,
    initializer_route_checkpoints,
    load_case_requests,
    public_interface_checkpoint,
    reproducibility_checkpoint,
    safety_type_checkpoint,
    verified_solar_checkpoint,
)
from book.chapter15_teaching import (
    AcceptanceRow,
    StellarRequest,
    acceptance_rows,
    canonical_digest,
    request_digest,
    summarize_atmosphere,
    summarize_spectrum,
)
from book.chapters.chapter_15 import build_notebook


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_artifact_verifier_recomputes_every_pinned_identity() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/verify_chapter15_artifacts.py"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Chapter 15 artifacts: verified" in completed.stdout
    assert "12 identities, 4 requests" in completed.stdout


def test_exact_public_types_and_signatures_remain_unwrapped() -> None:
    checkpoint = public_interface_checkpoint()
    assert checkpoint.forward_timing_fields == (
        "initializer_seconds",
        "population_bridge_seconds",
        "synthesis_seconds",
        "total_seconds",
    )
    assert checkpoint.initialized_atmosphere_fields == (
        "structured_atmosphere",
        "initializer_family",
        "labels",
        "provenance",
        "timings",
        "atmosphere_converged",
        "atmosphere_closure_required",
    )
    assert checkpoint.label_spectrum_fields == (
        "wavelength_nm",
        "flux_total",
        "flux_continuum",
        "normalized_flux",
        "seconds",
        "initializer_family",
        "labels",
        "provenance",
        "timings",
        "initialized_atmosphere",
        "atmosphere_converged",
        "atmosphere_closure_required",
    )
    assert checkpoint.spectrum_fields == (
        "wavelength_nm",
        "flux_total",
        "flux_continuum",
        "normalized_flux",
        "seconds",
    )
    assert "x_over_h" in checkpoint.initialize_signature
    assert "r_grid" in checkpoint.synthesize_from_labels_signature
    assert "resolution" in checkpoint.synthesize_signature
    assert checkpoint.validation_signature.endswith("-> 'tuple[str, ...]'")
    assert checkpoint.metadata_signature.endswith(
        "-> 'dict[str, object] | None'"
    )


def test_four_requests_preserve_exact_labels_and_route_deterministically() -> None:
    requests = load_case_requests()
    assert tuple(request.name for request in requests) == EXPECTED_CASE_NAMES
    assert all(isinstance(request, StellarRequest) for request in requests)
    assert [request_digest(request) for request in requests] == [
        "20213efb00814453f6974d5dd1af7501ebcdf5a4784befbdf0d99f895cf329a1",
        "8d41091248457499e352a24d4cc1f92e3b537cea95f4ff72e47195a80ecf5953",
        "e3e9d0d1031a107424823d4891e786214eec654c8d7db61af75ff0f4a627a5c8",
        "91d3b8889a576d000dd0a95336836771c3c4d2c81b853931bb06dcc278367ca7",
    ]

    routes = initializer_route_checkpoints()
    assert tuple(route.name for route in routes) == EXPECTED_CASE_NAMES
    assert tuple(route.routed_family for route in routes) == (
        "five_label",
        "five_label",
        "five_label",
        "cno8",
    )
    assert tuple(route.candidate_count for route in routes) == (2, 2, 2, 2)
    assert tuple(route.projection_used for route in routes) == (
        False,
        False,
        False,
        True,
    )
    assert all(route.request_unchanged for route in routes)
    assert routes[-1].first_initializer_query is not None
    assert (
        routes[-1].first_initializer_query["effective_temperature"]
        != requests[-1].effective_temperature
    )
    assert routes[-1].requested_labels == requests[-1].label_payload()
    assert "retaining the exact requested labels" in routes[-1].warning_messages[0]


def test_direct_81_to_97_mixture_is_exact_but_not_release_promoted() -> None:
    checkpoint = direct_mixture_checkpoint()
    assert checkpoint.public_atomic_number_count == 81
    assert checkpoint.solver_slot_count == 97
    assert checkpoint.sentinel_slot_count == 16
    assert checkpoint.iron_abundance == -0.2
    assert checkpoint.selected_public_values == checkpoint.selected_solver_values
    assert checkpoint.sentinels_inherit_iron
    assert checkpoint.centidex_lattice_exact
    assert checkpoint.mixture_sha256 == (
        "bd012ff8718dd1838c44d74ff6e05d28ba309671689642a7c3d6a15b73ec5f99"
    )
    assert not checkpoint.release_gate_passed
    assert checkpoint.exact_closure_required


def test_dependency_status_is_derived_and_names_every_current_blocker() -> None:
    checkpoint = dependency_checkpoint()
    assert checkpoint.initializer_export_names == (
        INITIALIZER_EXPORTS_REQUIRED_BY_SYNTHESIS_API
    )
    assert checkpoint.runner_required_symbols == EXPECTED_RUNNER_SYMBOLS
    assert checkpoint.exploratory_boundary.available == (
        not checkpoint.missing_initializer_exports
    )
    assert checkpoint.verified_boundary.available == (
        not checkpoint.missing_runner_symbols
        and checkpoint.solve_structured_atmosphere_staged
        and checkpoint.exact_source_catalogs_ready
    )
    if checkpoint.missing_initializer_exports:
        assert checkpoint.exploratory_boundary.blocker.startswith(
            "initializer_export_seam_blocked:"
        )
        assert all(
            name in checkpoint.exploratory_boundary.blocker
            for name in checkpoint.missing_initializer_exports
        )
    if not checkpoint.verified_boundary.available:
        if checkpoint.missing_runner_symbols or not checkpoint.solve_structured_atmosphere_staged:
            assert checkpoint.verified_boundary.blocker.startswith(
                "exact_runner_seam_blocked:"
            )
            assert all(
                name in checkpoint.verified_boundary.blocker
                for name in checkpoint.missing_runner_symbols
            )
        else:
            assert not checkpoint.exact_source_catalogs_ready
            assert checkpoint.verified_boundary.blocker.startswith(
                "exact_catalog_data_unavailable:"
            )


def test_exact_exploratory_types_cannot_claim_physical_closure() -> None:
    checkpoint = safety_type_checkpoint()
    assert not checkpoint.probe_only
    assert checkpoint.initializer_family == "five_label"
    assert checkpoint.initialized_flags == (False, True)
    assert checkpoint.label_spectrum_flags == (False, True)
    assert checkpoint.wavelength_count == 8
    assert np.isfinite(checkpoint.minimum_normalized_flux)
    assert checkpoint.initializer_seconds > 0.0
    assert checkpoint.population_bridge_seconds > 0.0
    assert checkpoint.synthesis_seconds > 0.0
    assert len(checkpoint.saved_validated_names) == 25
    assert checkpoint.loaded_field_count == 26
    assert checkpoint.saved_product_role == "learned_initializer_prediction"
    assert checkpoint.saved_initializer_family == "five_label"
    assert not checkpoint.metadata_converged
    assert checkpoint.metadata_closure_required
    assert not checkpoint.eligible_as_verified_physical_product
    assert checkpoint.mapping_arrays_equal


def test_four_regime_compact_schema_and_spectra_pass_only_owned_gates() -> None:
    checkpoint = four_regime_capstone()
    assert checkpoint.case_names == EXPECTED_CASE_NAMES
    assert checkpoint.wavelength_nm.shape == (6746,)
    assert np.all(np.diff(checkpoint.wavelength_nm) > 0.0)
    assert checkpoint.backend == "cpu"
    assert checkpoint.dtype == "torch.float64"
    assert checkpoint.compact_fixture_only

    for row in checkpoint.rows:
        assert row.atmosphere.required_field_count == 25
        assert row.atmosphere.missing_fields == ()
        assert row.atmosphere.depth_count == 6
        assert row.atmosphere.all_depth_axes_match
        assert row.atmosphere.all_public_arrays_float64
        assert row.atmosphere.finite
        assert row.atmosphere.column_mass_positive
        assert row.atmosphere.column_mass_strictly_increasing
        assert row.atmosphere.positive_thermodynamic_columns
        assert row.spectrum.wavelength_count == checkpoint.wavelength_nm.size
        assert row.spectrum.wavelength_strictly_increasing
        assert row.spectrum.arrays_aligned
        assert row.spectrum.finite
        assert row.spectrum.continuum_nonzero
        assert row.spectrum.normalized_ratio_matches

        exploratory = {gate.gate: gate.passed for gate in row.exploratory_acceptance}
        assert exploratory["initializer support and asset integrity"] is True
        assert exploratory["schema-v4 mapping"] is True
        assert exploratory["finite wavelength-aligned spectrum"] is True
        assert exploratory["structural fixed-point convergence"] is None
        assert exploratory["declared flux-error threshold"] is None
        assert exploratory["hydrostatic residual"] is None

        verified = {gate.gate: gate.passed for gate in row.verified_acceptance}
        assert verified["structural fixed-point convergence"] is None
        assert verified["schema-v4 mapping"] is None
        assert verified["finite wavelength-aligned spectrum"] is None
        assert "fixture" not in row.verified_status.lower()


def test_verified_solar_checkpoint_preserves_exact_parity_and_open_gates() -> None:
    checkpoint = verified_solar_checkpoint()
    assert checkpoint.status == "accepted_exact_three_run_array_parity"
    assert checkpoint.run_roles == (
        "staged",
        "staged_repeat",
        "pinned_read_only_oracle",
    )
    assert checkpoint.run_count == 3
    assert checkpoint.converged_on_pass == 4
    assert checkpoint.deep_temperature_change < (
        checkpoint.deep_temperature_change_threshold
    )
    assert checkpoint.atmosphere_array_count == 27
    assert checkpoint.atmosphere_depth_count == 80
    assert checkpoint.atmosphere_archives_byte_identical
    assert checkpoint.wavelength_count == 8
    assert checkpoint.wavelength_start_nm >= 498.95
    assert checkpoint.wavelength_stop_nm <= 499.15
    assert checkpoint.spectrum_physical_arrays_bitwise_identical
    assert checkpoint.catalogs_required_to_rebuild
    assert not checkpoint.catalogs_required_to_validate

    accepted = {row.gate: row.passed for row in checkpoint.acceptance}
    assert accepted["structural fixed-point convergence"] is True
    assert accepted["schema-v4 mapping"] is True
    assert accepted["finite wavelength-aligned spectrum"] is True
    assert accepted["golden spectral parity"] is True
    assert accepted["provenance and limitations"] is True
    assert accepted["declared flux-error threshold"] is None
    assert accepted["hydrostatic residual"] is None
    assert accepted["standard optical-depth grid"] is None


def test_schema_and_spectrum_failures_are_independent() -> None:
    from book.chapter10_runtime import load_regime_atmosphere, public_spectrum
    from payne_zero_synthesis.atmosphere import REQUIRED_ATMOSPHERE_ARRAYS

    atmosphere = load_regime_atmosphere("solar_dwarf")
    broken_column_mass = dict(atmosphere)
    broken_column_mass["column_mass"] = atmosphere["column_mass"][::-1].copy()
    atmosphere_summary = summarize_atmosphere(
        broken_column_mass,
        required_fields=REQUIRED_ATMOSPHERE_ARRAYS,
    )
    assert not atmosphere_summary.column_mass_strictly_increasing
    assert atmosphere_summary.finite

    spectrum = public_spectrum("solar_dwarf")
    broken_normalized = spectrum.normalized_flux.copy()
    broken_normalized[0] += 0.01
    spectrum_summary = summarize_spectrum(
        wavelength_nm=spectrum.wavelength_nm,
        flux_total=spectrum.flux_total,
        flux_continuum=spectrum.flux_continuum,
        normalized_flux=broken_normalized,
        ratio_rtol=3.0e-7,
        ratio_atol=0.0,
    )
    assert spectrum_summary.finite
    assert not spectrum_summary.normalized_ratio_matches


def test_acceptance_rows_preserve_none_instead_of_coercing_to_failure() -> None:
    rows = acceptance_rows(
        workflow="exploratory",
        initializer_assets_valid=True,
        seed_valid=True,
        structural_convergence=None,
        flux_accepted=None,
        hydrostatic_accepted=None,
        optical_grid_accepted=True,
        schema_valid=True,
        spectrum_valid=True,
        golden_spectral_parity=None,
        provenance_valid=True,
        blockers=("physical closure not requested",),
    )
    assert all(isinstance(row, AcceptanceRow) for row in rows)
    status = {row.gate: row.passed for row in rows}
    assert status["structural fixed-point convergence"] is None
    assert status["declared flux-error threshold"] is None
    assert status["hydrostatic residual"] is None
    assert status["schema-v4 mapping"] is True


def test_provenance_digest_is_repeatable_complete_and_timing_free() -> None:
    checkpoint = reproducibility_checkpoint()
    assert checkpoint.repeatable
    assert checkpoint.sha256 == checkpoint.repeated_sha256
    assert checkpoint.timing_fields_excluded
    assert canonical_digest(checkpoint.provenance) == checkpoint.sha256
    assert checkpoint.provenance["pinned_payne_zero_commit"] == (
        "9c44001feae40b85146630499e6f8a5fed42e5af"
    )
    assert tuple(checkpoint.provenance["requests"]) == EXPECTED_CASE_NAMES
    assert checkpoint.provenance["availability"]["compact_fixture_only"]
    assert len(checkpoint.provenance["limitations"]) >= 6


def test_notebook_is_a_compact_capstone_with_four_case_cells() -> None:
    document = build_notebook()
    visible = [
        cell
        for cell in document["cells"]
        if cell["cell_type"] == "code"
        and "hide-input" not in cell.get("metadata", {}).get("tags", ())
        and "book-setup" not in cell.get("metadata", {}).get("tags", ())
    ]
    assert len(visible) == 16
    markdown_source = "\n".join(
        cell["source"]
        for cell in document["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "# Chapter 15" in markdown_source
    assert "Hot dwarfs" in markdown_source
    assert "solar-like dwarfs" in markdown_source
    assert "low-gravity giants" in markdown_source
    assert "cool" in markdown_source and "molecule-rich dwarfs" in markdown_source
    assert "## 15.18 Chapter summary" in markdown_source
    assert "```python" not in markdown_source
    assert "/Users/ysting/payne-zero" not in markdown_source
    assert "## Exercises" not in markdown_source
