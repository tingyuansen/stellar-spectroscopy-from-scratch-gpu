"""Focused scientific, source-resolution, and chapter gates for Chapter 9."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from book.chapter09_runtime import (
    ATMOSPHERE_TRANSFER_TABLES,
    SYNTHESIS_TRANSFER_TABLES,
    atmosphere_moment_checkpoint,
    atmosphere_parallel_checkpoint,
    contribution_checkpoint,
    equal_extinction_checkpoint,
    flux_conversion_checkpoint,
    hand_sweep_checkpoint,
    optical_depth_checkpoint,
    prepared_window_checkpoint,
    remap_checkpoint,
    saturated_route_checkpoint,
    source_sweep_checkpoint,
    transfer_table_checkpoint,
)
from book.chapters.chapter_09 import build_notebook
from book.registry import BY_NUMBER


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_fresh_process_resolves_staged_source_before_first_checkpoint() -> None:
    script = r"""
import json
from pathlib import Path
from book.chapters.chapter_09 import build_notebook

namespace = {}
for cell in build_notebook()["cells"]:
    if cell["cell_type"] != "code":
        continue
    exec(cell["source"], namespace, namespace)
    if "source_and_alpha" in namespace:
        break

namespace["equal_extinction_checkpoint"]()
namespace["prepared_window_checkpoint"]()
import payne_zero_atmosphere.transfer_kernels as atmosphere_transfer
import payne_zero_synthesis.constants as synthesis_constants
import payne_zero_synthesis.radiative_transfer as synthesis_transfer

print(json.dumps({
    "synthesis_constants": str(Path(synthesis_constants.__file__).resolve()),
    "synthesis_transfer": str(Path(synthesis_transfer.__file__).resolve()),
    "atmosphere_transfer": str(Path(atmosphere_transfer.__file__).resolve()),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = json.loads(completed.stdout)
    staged_root = (REPOSITORY_ROOT / "src").resolve()
    assert all(
        Path(path).resolve().is_relative_to(staged_root)
        for path in resolved.values()
    )


def test_equal_extinction_does_not_imply_equal_flux() -> None:
    checkpoint = equal_extinction_checkpoint()
    np.testing.assert_array_equal(checkpoint.extinction[0], checkpoint.extinction[1])
    assert checkpoint.scattering_fraction[0, 0] == pytest.approx(0.1)
    assert checkpoint.scattering_fraction[1, 0] == pytest.approx(0.9)
    assert checkpoint.eddington_flux_per_frequency[0] != pytest.approx(
        checkpoint.eddington_flux_per_frequency[1]
    )


def test_compact_window_has_monotone_optical_depth_and_outward_line_formation() -> None:
    optical = optical_depth_checkpoint()
    assert optical.optical_depth.shape == (2, optical.column_mass.size)
    assert np.all(np.diff(optical.optical_depth, axis=1) >= 0.0)
    np.testing.assert_array_equal(
        optical.optical_depth[:, 0],
        optical.extinction[:, 0] * optical.column_mass[0],
    )
    contribution = contribution_checkpoint()
    assert contribution.line_peak_column_mass < contribution.continuum_peak_column_mass


def test_two_static_table_products_keep_exact_shared_and_distinct_members() -> None:
    checkpoint = transfer_table_checkpoint()
    assert SYNTHESIS_TRANSFER_TABLES.is_file()
    assert ATMOSPHERE_TRANSFER_TABLES.is_file()
    assert checkpoint.synthesis_shapes == (
        (51,),
        (51, 51),
        (51,),
        (51,),
    )
    assert checkpoint.shared_grid_exact
    assert checkpoint.shared_surface_weights_exact
    assert checkpoint.mean_operator_differing_entries == 4
    assert checkpoint.mean_operator_max_abs_difference <= 1.0e-8


def test_remap_and_backward_source_sweep_match_exact_boundaries() -> None:
    remap = remap_checkpoint()
    assert remap.exact_point_max_abs < 3.0e-15
    assert remap.above_atmosphere_count > 0
    assert remap.above_atmosphere_source_exact
    assert remap.above_atmosphere_scattering_exact

    sweep = hand_sweep_checkpoint()
    np.testing.assert_array_equal(sweep.hand_source, sweep.exact_source)


def test_eight_synthesis_sweeps_are_positive_and_float32() -> None:
    checkpoint = source_sweep_checkpoint()
    assert checkpoint.source_dtype == "torch.float32"
    assert checkpoint.minimum_after_each_sweep.shape == (8,)
    assert np.all(checkpoint.minimum_after_each_sweep > 0.0)
    assert not np.array_equal(
        checkpoint.thermal_source,
        checkpoint.eight_sweep_source,
    )


def test_saturated_boundary_has_strict_failure_and_allowed_positive_fallback() -> None:
    checkpoint = saturated_route_checkpoint()
    assert checkpoint.threshold == 20.0
    assert checkpoint.strict_failure_seen
    np.testing.assert_array_equal(
        checkpoint.saturated_mask,
        checkpoint.first_layer_optical_depth > checkpoint.threshold,
    )
    assert np.all(np.isfinite(checkpoint.eddington_flux_per_frequency))
    assert np.all(checkpoint.eddington_flux_per_frequency > 0.0)


def test_total_continuum_stack_and_zero_line_limit() -> None:
    checkpoint = prepared_window_checkpoint()
    assert checkpoint.normalized_flux.shape == checkpoint.wavelength_nm.shape
    assert checkpoint.stacked_max_abs_difference < 1.0e-12
    assert np.max(np.abs(checkpoint.zero_line_normalized_flux - 1.0)) < 1.0e-9
    assert np.min(checkpoint.normalized_flux) < 0.9
    assert np.all(np.isfinite(checkpoint.normalized_flux))


def test_surface_flux_conversion_has_four_pi_jacobian_and_ratio_invariance() -> None:
    checkpoint = flux_conversion_checkpoint()
    np.testing.assert_array_equal(
        checkpoint.flux_per_wavelength_nm,
        checkpoint.helper_flux_per_wavelength_nm,
    )
    np.testing.assert_allclose(
        checkpoint.flux_per_frequency,
        4.0 * np.pi * checkpoint.eddington_flux_per_frequency,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        checkpoint.normalized_before,
        checkpoint.normalized_after,
        rtol=2.0e-16,
        atol=0.0,
    )


def test_atmosphere_moments_and_fixed_chunk_policy() -> None:
    fixed = atmosphere_moment_checkpoint(deep=False)
    deep = atmosphere_moment_checkpoint(deep=True)
    assert fixed.mapped_layer_count > deep.mapped_layer_count
    for checkpoint in (fixed, deep):
        assert np.all(np.diff(checkpoint.optical_depth) >= 0.0)
        assert np.all(np.isfinite(checkpoint.mean_intensity))
        assert np.all(np.isfinite(checkpoint.eddington_flux))
        expected_line = (
            checkpoint.gross_line_mass_absorption_coefficient.astype(np.float64)
            * checkpoint.stimulated
        )
        np.testing.assert_array_equal(
            checkpoint.stimulated_line_mass_absorption_coefficient,
            expected_line,
        )

    parallel = atmosphere_parallel_checkpoint()
    assert parallel.fixed_policy_repeatable
    assert parallel.worst_absolute_difference <= 1.0e-12
    assert parallel.worst_relative_difference <= 1.0e-12


def test_chapter_has_17_visible_cells_four_schematics_and_causal_close() -> None:
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
    assert len(visible) == 18
    assert "## 9.17 Chapter summary" in markdown_text
    assert "/reader.html?ch=10" in markdown_text
    assert "```python" not in markdown_text
    assert BY_NUMBER[9].available
    for name in (
        "ch09-rays-moments-boundaries-v1.png",
        "ch09-scattering-fixed-point-v1.png",
        "ch09-total-continuum-stack-v1.png",
        "ch09-two-transfer-lanes-v1.png",
    ):
        assert f"assets/schematics/textbook/{name}" in markdown_text
        assert (REPOSITORY_ROOT / "assets/schematics/textbook" / name).is_file()
