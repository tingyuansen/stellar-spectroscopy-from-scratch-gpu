"""Focused exact-correction and iteration-control gates for Chapter 13."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from book.chapter13_runtime import (
    ARTIFACT_MANIFEST,
    FIXTURE_PATH,
    GOLDEN_PATH,
    PINNED_SOURCE_SHA256,
    RUNNER_PASS_ORDER,
    RUNNER_REQUIRED_SYMBOLS,
    cache_contract_checkpoint,
    chapter12_handoff_checkpoint,
    chunk_checkpoint,
    correction_checkpoint,
    damping_checkpoint,
    iteration_control_checkpoint,
    output_contract_checkpoint,
    prewarm_contract_checkpoint,
    quantization_checkpoint,
    reset_checkpoint,
    runner_patch_plan,
)
from book.chapters.chapter_13 import build_notebook


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_standalone_source_and_artifact_identities() -> None:
    manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    for relative, digest in PINNED_SOURCE_SHA256.items():
        assert _sha256(REPOSITORY_ROOT / relative) == digest
    assert _sha256(FIXTURE_PATH) == manifest["artifacts"][
        str(FIXTURE_PATH.relative_to(REPOSITORY_ROOT))
    ]["sha256"]
    assert _sha256(GOLDEN_PATH) == manifest["artifacts"][
        str(GOLDEN_PATH.relative_to(REPOSITORY_ROOT))
    ]["sha256"]


def test_artifact_verifier_recomputes_before_comparison() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/verify_chapter13_artifacts.py"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Chapter 13 artifacts: verified" in completed.stdout


def test_mode3_correction_matches_every_pinned_output() -> None:
    checkpoint = correction_checkpoint()
    assert set(checkpoint.computed) == set(checkpoint.golden)
    for name in checkpoint.computed:
        np.testing.assert_array_equal(
            checkpoint.computed[name],
            checkpoint.golden[name],
        )
    assert checkpoint.maximum_absolute_difference == 0.0
    assert checkpoint.raw_three_term_identity


def test_chapter12_finalization_is_consumed_by_the_exact_remapper() -> None:
    checkpoint = chapter12_handoff_checkpoint()
    assert checkpoint.finalization_type == "IterationFinalization"
    assert checkpoint.correction_type == "TemperatureCorrectionResult"
    assert checkpoint.remap_type == "IterationRemap"
    assert checkpoint.depth_count == 80
    assert checkpoint.frequency_count == 30_000
    assert checkpoint.correction_field_names == (
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
    for values in (
        checkpoint.correction_temperature,
        checkpoint.correction_column_mass,
        checkpoint.remapped_temperature,
        checkpoint.remapped_column_mass,
        checkpoint.standard_rosseland_optical_depth,
    ):
        assert values.shape == (80,)
        assert np.all(np.isfinite(values))
    assert checkpoint.correction_result_finite
    assert checkpoint.rosseland_optical_depth_strictly_increasing
    assert checkpoint.correction_column_mass_positive
    assert checkpoint.correction_column_mass_strictly_increasing
    assert checkpoint.remapped_fields_finite
    assert checkpoint.remapped_column_mass_strictly_increasing
    assert checkpoint.source_finalization_is_chapter12_cache


def test_correction_result_is_finite_positive_and_monotone() -> None:
    checkpoint = correction_checkpoint()
    result = checkpoint.computed
    for name, values in result.items():
        assert np.all(np.isfinite(values)), name
    assert np.all(result["temperature"] >= 1.0)
    assert checkpoint.minimum_inward_temperature_rise_k >= 1.0
    assert checkpoint.column_mass_positive
    assert checkpoint.column_mass_strictly_increasing
    np.testing.assert_array_equal(
        result["state_previous_temperature_correction"],
        result["temperature_correction"],
    )


def test_mode1_resets_only_per_pass_accumulators() -> None:
    checkpoint = reset_checkpoint()
    assert checkpoint.reset_fields == (
        "mean_intensity_minus_source_integral",
        "absorption_heating_derivative",
        "diagonal_lambda_accumulator",
        "integrated_eddington_flux",
    )
    np.testing.assert_array_equal(checkpoint.reset_sums, np.zeros(4))
    assert checkpoint.previous_correction_unchanged
    assert checkpoint.lookup_same_object
    assert checkpoint.lookup_entry_count_unchanged


def test_previous_step_damping_branches_are_exact() -> None:
    checkpoint = damping_checkpoint()
    np.testing.assert_array_equal(
        checkpoint.same_sign,
        checkpoint.same_sign_expected,
    )
    np.testing.assert_array_equal(
        checkpoint.sign_flip,
        checkpoint.sign_flip_expected,
    )


def test_deep_norm_uses_exact_zero_based_slice_and_new_denominator() -> None:
    from payne_zero_atmosphere.convergence import (
        deep_layer_relative_temperature_change,
    )

    before = np.linspace(4000.0, 8000.0, 80)
    shallow_after = before.copy()
    shallow_after[:39] += 1000.0
    assert deep_layer_relative_temperature_change(before, shallow_after) == 0.0

    after = before.copy()
    after[39] += 4.0
    expected = abs(after[39] - before[39]) / abs(after[39])
    assert deep_layer_relative_temperature_change(before, after) == expected


def test_minimum_and_consecutive_state_machine_stops_on_exact_pass() -> None:
    checkpoint = iteration_control_checkpoint()
    assert checkpoint.pass_order == RUNNER_PASS_ORDER
    assert checkpoint.standard_trace.stopping_pass == 4
    assert checkpoint.standard_trace.converged
    np.testing.assert_array_equal(
        checkpoint.standard_trace.consecutive_count,
        np.asarray([0, 0, 1, 2], dtype=np.int32),
    )
    assert checkpoint.interrupted_trace.stopping_pass == 5
    np.testing.assert_array_equal(
        checkpoint.interrupted_trace.consecutive_count,
        np.asarray([0, 1, 0, 1, 2], dtype=np.int32),
    )
    assert not checkpoint.stopping_disabled_trace.converged
    assert checkpoint.stopping_disabled_trace.stopping_pass is None


def test_fixed_chunks_cover_frequency_range_once_and_reduce_in_order() -> None:
    checkpoint = chunk_checkpoint(start=3, stop=22, chunk_count=6)
    assert checkpoint.bounds[0] == checkpoint.start
    assert checkpoint.bounds[-1] == checkpoint.stop
    assert np.all(np.diff(checkpoint.bounds) >= 0)
    assert checkpoint.covers_each_frequency_once
    np.testing.assert_array_equal(
        checkpoint.reduction_order,
        np.arange(checkpoint.chunk_count),
    )


def test_terminal_quantization_is_one_idempotent_round_trip() -> None:
    checkpoint = quantization_checkpoint()
    assert checkpoint.terminal_format_parse_calls == 1
    assert checkpoint.idempotence_probe_format_parse_calls == 1
    assert checkpoint.idempotent
    assert np.max(
        np.abs(checkpoint.quantized_temperature - checkpoint.input_temperature)
    ) > 0.0


def test_cache_precedence_and_runner_seam_are_reported_without_false_prewarm() -> None:
    checkpoint = cache_contract_checkpoint()
    assert checkpoint.existing_numba_cache == Path(
        "/tmp/chapter13-existing-numba"
    )
    assert checkpoint.requested_payne_zero_cache == Path(
        "/tmp/chapter13-requested"
    )
    assert set(checkpoint.runner_symbols_available).isdisjoint(
        checkpoint.runner_symbols_missing
    )
    assert set(checkpoint.runner_symbols_available).union(
        checkpoint.runner_symbols_missing
    ) == set(RUNNER_REQUIRED_SYMBOLS)
    if checkpoint.prewarm_executable:
        assert not checkpoint.runner_symbols_missing


def test_prewarm_and_output_contracts_do_not_overclaim_pending_runner_work() -> None:
    prewarm = prewarm_contract_checkpoint()
    assert prewarm.branch_names == ("hot", "sun", "giant", "sun_atomic_only")
    assert prewarm.representative_iterations_per_branch == 1
    if prewarm.executable:
        assert prewarm.runner_ready

    outputs = output_contract_checkpoint()
    assert len(outputs.base_diagnostic_keys) == 23
    assert len(outputs.iteration_timing_keys) == 14
    assert not outputs.diagnostics_path_written
    assert not outputs.debug_uses_terminal_quantized_atmosphere
    assert outputs.product_requires_structural_convergence
    assert outputs.product_population_source == "final_fixed_column_quantized_arrays"


def test_deferred_shared_runner_patch_plan_freezes_nonnegotiable_edges() -> None:
    plan = runner_patch_plan()
    assert plan["required_symbols"] == RUNNER_REQUIRED_SYMBOLS
    assert plan["pass_order"] == RUNNER_PASS_ORDER
    assert len(plan["pass_order"]) == 15
    assert plan["interior_edge"] == "correction -> complete remap -> next pass"
    assert plan["forbidden_interior_call"] == "fixed-column format/parse"
    if plan["missing_symbols"]:
        assert plan["blocked_by"] == "Chapter 11/12 shared runner seams"


def test_chapter_is_causal_self_contained_and_closes_toward_chapter14() -> None:
    document = build_notebook()
    cells = document["cells"]
    visible = [
        cell
        for cell in cells
        if cell["cell_type"] == "code"
        and "hide-input" not in cell.get("metadata", {}).get("tags", [])
    ]
    markdown_text = "\n".join(
        cell["source"] for cell in cells if cell["cell_type"] == "markdown"
    )
    assert 10 <= len(visible) <= 20
    assert "## 13.12 Chapter summary" in markdown_text
    assert "### Next:" in markdown_text
    assert "](/reader.html?ch=14)" in markdown_text
    assert "```python" not in markdown_text
    assert "## Exercises" not in markdown_text
    assert "/Users/ysting/payne-zero" not in markdown_text
    assert "/Users/ysting/Source_Files_Not_For_Review" not in markdown_text
    assert "correction -> complete remap -> next pass" in markdown_text
