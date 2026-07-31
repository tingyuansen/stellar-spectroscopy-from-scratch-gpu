"""Parity and pedagogy gates for Chapter 14's initializer reader core."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from book.chapter14_runtime import (
    ARTIFACT_MANIFEST,
    FIVE_LABEL_EXAMPLE,
    FIXTURE_PATH,
    GOLDEN_PATH,
    PINNED_ASSET_SHA256,
    PINNED_SOURCE_SHA256,
    asset_identity_checkpoint,
    candidate_checkpoint,
    closure_seam_checkpoint,
    decoder_checkpoint,
    direct_decoder_checkpoint,
    direct_mixture_checkpoint,
    direct_safety_checkpoint,
    direct_set_encoder_checkpoint,
    direct_warm_start_checkpoint,
    training_data_checkpoint,
    warm_start_checkpoint,
)
from book.chapter14_teaching import (
    fixed_point_contraction_trace,
    pca_sentinel_trace,
    profile_transform_round_trip,
    quantize_centidex,
    set_encoder_token_inputs,
)
from book.chapters.chapter_14 import build_notebook


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_and_runtime_assets_are_pinned_without_training_corpora() -> None:
    checkpoint = asset_identity_checkpoint()
    assert checkpoint.source_sha256 == PINNED_SOURCE_SHA256
    assert checkpoint.asset_sha256 == PINNED_ASSET_SHA256
    assert checkpoint.release == "v1.3"
    assert len(checkpoint.five_label_features) == 5
    assert len(checkpoint.cno8_features) == 8
    assert checkpoint.exact_solver_is_final_authority
    assert not checkpoint.training_corpora_packaged


def test_six_profile_transforms_round_trip_and_keep_declared_constraints() -> None:
    decoded = decoder_checkpoint("five_label")
    profile = np.column_stack(tuple(decoded.prediction.values()))
    from payne_zero_atmosphere.warm_start import (
        INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH,
    )

    round_trip = profile_transform_round_trip(
        profile,
        effective_temperature=FIVE_LABEL_EXAMPLE["effective_temperature"],
        rosseland_optical_depth=INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH,
        acceleration_scale=decoded.acceleration_scale,
    )
    assert round_trip.coordinates.shape == (80, 6)
    assert round_trip.reconstructed.dtype == np.float64
    assert np.max(round_trip.maximum_relative_difference) < 2.0e-14
    assert np.all(np.diff(round_trip.reconstructed[:, 0]) > 0.0)
    assert np.all(round_trip.reconstructed[:, :5] > 0.0)


def test_pca_orientation_and_dtype_boundary_are_exact() -> None:
    sentinel = pca_sentinel_trace(component_index=7, layer_index=23, field_index=4)
    assert sentinel.flattened_index == 23 * 6 + 4
    assert sentinel.recovered_layer_index == 23
    assert sentinel.recovered_field_index == 4

    for family, input_dim in (("five_label", 5), ("cno8", 8)):
        checkpoint = decoder_checkpoint(family)
        assert checkpoint.model_config["input_dim"] == input_dim
        assert checkpoint.model_config["output_dim"] == 160
        assert checkpoint.standardized_coefficients.shape == (160,)
        assert checkpoint.standardized_coefficients.dtype == np.float32
        assert checkpoint.pca_trace.network_output_dtype == "float32"
        assert checkpoint.pca_trace.coefficients_dtype == "float64"
        assert checkpoint.pca_trace.standardized_coordinates_dtype == "float64"
        assert checkpoint.pca_trace.coordinates_dtype == "float64"
        assert checkpoint.pca_trace.coordinates.shape == (80, 6)
        assert checkpoint.maximum_absolute_decode_difference == 0.0
        assert set(checkpoint.output_shapes.values()) == {(80,)}
        assert set(checkpoint.output_dtypes.values()) == {"float64"}


def test_training_objective_weights_match_each_family() -> None:
    five = decoder_checkpoint("five_label")
    cno = decoder_checkpoint("cno8")
    direct = direct_decoder_checkpoint()
    assert five.derivative_loss_weight == cno.derivative_loss_weight == 0.1
    assert direct.derivative_loss_weight == 0.2
    for checkpoint in (five, cno, direct):
        assert checkpoint.optical_depth_loss_weight == 0.05
        assert checkpoint.hydrostatic_loss_weight == 0.05


def test_training_roles_are_explicit_and_counts_close_exactly() -> None:
    checkpoint = training_data_checkpoint()
    assert (
        checkpoint.five_label_train
        + checkpoint.five_label_fit_validation
        + checkpoint.five_label_internal_check
        == checkpoint.five_label_total
        == 52199
    )
    assert (
        checkpoint.cno8_frozen_train
        + checkpoint.cno8_appended_training
        + checkpoint.cno8_fit_validation
        + checkpoint.cno8_internal_check
        == checkpoint.cno8_total
        == 53824
    )
    assert (
        checkpoint.direct_train
        + checkpoint.direct_fit_validation
        + checkpoint.direct_internal_check
        + checkpoint.direct_unused_external_gate
        == checkpoint.direct_total
        == 82016
    )
    assert checkpoint.direct_optimizer == "AdamW"
    assert checkpoint.direct_learning_rate == 0.0007
    assert checkpoint.direct_weight_decay == 0.0001
    assert checkpoint.direct_seed == 314159


def test_family_routing_projection_and_candidate_order_are_deterministic() -> None:
    checkpoint = candidate_checkpoint()
    assert checkpoint.five_label_family == "five_label"
    assert checkpoint.cno_relative_family == "cno8"
    assert checkpoint.cno_absolute_family == "cno8"
    assert checkpoint.in_support_candidates[0] is None
    assert checkpoint.projected_candidates[0] is not None
    assert (
        checkpoint.first_projected_temperature
        < checkpoint.projected_request_temperature
    )
    assert len(checkpoint.projected_candidates) == 3
    assert checkpoint.deterministic
    assert checkpoint.projected_candidates == checkpoint.repeated_candidates


def test_fixed_column_seed_preserves_target_labels_but_quantizes_profiles() -> None:
    for family in ("five_label", "cno8"):
        checkpoint = warm_start_checkpoint(family)
        assert checkpoint.requested_effective_temperature == (
            checkpoint.parsed_effective_temperature
        )
        assert checkpoint.requested_log_surface_gravity == (
            checkpoint.parsed_log_surface_gravity
        )
        assert not checkpoint.has_converged_field
        assert len(checkpoint.deck_sha256) == 64
        assert max(checkpoint.maximum_relative_quantization_difference.values()) > 0.0
        assert np.all(checkpoint.quantized_prediction["column_mass"] > 0.0)
        assert np.all(np.diff(checkpoint.quantized_prediction["column_mass"]) > 0.0)


def test_direct_layout_lattice_sentinels_and_hash_are_exact() -> None:
    checkpoint = direct_mixture_checkpoint()
    layout = checkpoint.layout
    assert layout.public_abundance_count == 81
    assert layout.network_feature_count == 84
    assert layout.exact_mixture_count == 97
    assert layout.sentinel_count == 16
    assert checkpoint.feature_vector.shape == (84,)
    assert checkpoint.exact_mixture.shape == (97,)
    assert checkpoint.public_values_quantized
    assert layout.maximum_sentinel_difference_from_iron == 0.0
    assert len(checkpoint.mixture_sha256) == 64


def test_centidex_half_steps_use_numpy_ties_to_even() -> None:
    values = np.asarray([-0.025, -0.015, -0.005, 0.005, 0.015, 0.025])
    np.testing.assert_array_equal(
        quantize_centidex(values),
        np.asarray([-0.02, -0.02, 0.0, 0.0, 0.02, 0.02]),
    )


def test_direct_set_encoder_matches_exact_forward_and_keeps_identity_paired() -> None:
    checkpoint = direct_set_encoder_checkpoint()
    assert checkpoint.state_shape == (1, 4)
    assert checkpoint.relative_abundance_shape == (1, 80)
    assert checkpoint.element_embedding_shape == (80, 64)
    assert checkpoint.token_input_shape == (1, 80, 70)
    assert checkpoint.response_law_shape == (1, 80, 2, 128)
    assert checkpoint.summed_response_shape == (1, 128)
    assert checkpoint.output_shape == (1, 160)
    assert checkpoint.maximum_manual_output_difference == 0.0
    assert checkpoint.paired_permutation_output_difference < 5.0e-7
    assert checkpoint.abundance_only_permutation_output_difference > 1.0e-3

    token = set_encoder_token_inputs(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        np.asarray([-0.2, 0.3]),
    )
    assert token.shape == (2, 8)
    np.testing.assert_allclose(token[:, -2], [-0.2, 0.3])
    np.testing.assert_allclose(token[:, -1], [0.04, 0.09])


def test_direct_decoder_uses_the_same_common_float64_decoder() -> None:
    checkpoint = direct_decoder_checkpoint()
    assert checkpoint.family == "direct_abundance"
    assert len(checkpoint.checkpoint_feature_fields) == 84
    assert checkpoint.model_config["architecture"] == "set_encoded"
    assert checkpoint.model_config["output_dim"] == 160
    assert checkpoint.standardized_coefficients.dtype == np.float32
    assert checkpoint.pca_trace.coordinates.dtype == np.float64
    assert checkpoint.maximum_absolute_decode_difference == 0.0


def test_direct_seed_crosses_the_same_fixed_column_boundary_once() -> None:
    checkpoint = direct_warm_start_checkpoint()
    assert checkpoint.family == "direct_abundance"
    assert checkpoint.requested_effective_temperature == (
        checkpoint.parsed_effective_temperature
    )
    assert checkpoint.requested_log_surface_gravity == (
        checkpoint.parsed_log_surface_gravity
    )
    assert not checkpoint.has_converged_field
    assert max(checkpoint.maximum_relative_quantization_difference.values()) > 0.0
    assert np.all(checkpoint.quantized_prediction["column_mass"] > 0.0)
    assert np.all(np.diff(checkpoint.quantized_prediction["column_mass"]) > 0.0)


def test_direct_public_boundaries_fail_closed_and_keep_provenance() -> None:
    checkpoint = direct_safety_checkpoint()
    assert checkpoint.opt_in_rejected_by_default
    assert checkpoint.incomplete_public_vector_rejected
    assert checkpoint.unsupported_state_rejected
    assert not checkpoint.release_gate_passed
    assert checkpoint.role == "experimental_direct_xh_optimizer_surrogate"
    assert checkpoint.exact_closure_required
    assert not checkpoint.is_final_atmosphere
    assert not checkpoint.realized_mixture_writeable
    assert checkpoint.public_deck_type == "str"
    assert not checkpoint.public_deck_exposes_model
    assert checkpoint.exact_trial_count == 1
    for digest in (
        checkpoint.checkpoint_sha256,
        checkpoint.manifest_sha256,
        checkpoint.realized_mixture_sha256,
        checkpoint.deck_sha256,
        checkpoint.surrogate_identity_sha256,
    ):
        assert len(digest) == 64


def test_fixed_point_example_separates_a_good_start_from_zero_residual() -> None:
    trace = fixed_point_contraction_trace(4.0, fixed_point=1.0, contraction=0.5)
    assert trace.values[0] == 4.0
    assert trace.values[-1] != trace.fixed_point
    assert np.all(np.abs(trace.residuals[1:]) < np.abs(trace.residuals[:-1]))


def test_artifacts_are_pinned_and_verifier_recomputes_before_comparison() -> None:
    assert ARTIFACT_MANIFEST.is_file()
    assert FIXTURE_PATH.is_file()
    assert GOLDEN_PATH.is_file()
    first_fixture = _sha256(FIXTURE_PATH)
    first_golden = _sha256(GOLDEN_PATH)
    completed = subprocess.run(
        [sys.executable, "scripts/verify_chapter14_artifacts.py"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Chapter 14 artifacts: verified" in completed.stdout
    assert _sha256(FIXTURE_PATH) == first_fixture
    assert _sha256(GOLDEN_PATH) == first_golden


def test_shared_runner_seam_is_reported_without_false_chapter13_closure() -> None:
    checkpoint = closure_seam_checkpoint()
    assert checkpoint.initializer_reader_core_executable
    assert set(checkpoint.runner_symbols_available).isdisjoint(
        checkpoint.runner_symbols_missing
    )
    assert not checkpoint.runner_symbols_missing
    assert checkpoint.status == "ready"
    assert checkpoint.exact_restart_trajectory_executable


def test_chapter_is_causal_self_contained_and_closes_toward_chapter15() -> None:
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
    assert "## 14.9 Chapter summary" in markdown_text
    assert "### Next:" in markdown_text
    assert "](/reader.html?ch=15)" in markdown_text
    assert "```python" not in markdown_text
    assert "## Exercises" not in markdown_text
    assert "/Users/ysting/payne-zero" not in markdown_text
    assert "/Users/ysting/Source_Files_Not_For_Review" not in markdown_text
    assert "81" in markdown_text and "84" in markdown_text and "97" in markdown_text
    assert "restart trajectory executable" in markdown_text
