"""Focused scientific and interface checks for Chapter 7."""

from __future__ import annotations

import unittest

import numpy as np

from book.chapter07_runtime import (
    atomic_source_checkpoint,
    autoionizing_checkpoint,
    catalog_layout_checkpoint,
    compare_forest_wing_modes,
    hydrogen_checkpoint,
    record_transform_checkpoint,
    route_ledger,
    run_atomic_forest,
    scatter_add_checkpoint,
    window_selection_checkpoint,
)


class Chapter07RuntimeTests(unittest.TestCase):
    """Protect the compact catalog-to-forest construction."""

    def test_compact_source_has_ordinary_and_each_available_special_example(self) -> None:
        checkpoint = atomic_source_checkpoint()
        self.assertEqual(checkpoint.record_count, 133)
        self.assertEqual(
            checkpoint.route_counts,
            {-6: 1, -3: 1, -2: 1, -1: 1, 0: 128, 1: 1},
        )
        self.assertEqual(sum(checkpoint.full_catalog_route_counts.values()), 1_939_975)

    def test_isotope_correction_and_blank_damping_fields_are_observable(self) -> None:
        checkpoint = record_transform_checkpoint()
        self.assertEqual(checkpoint.isotope_changed_count, 21)
        self.assertEqual(checkpoint.default_radiative_count, 25)
        self.assertEqual(checkpoint.default_stark_count, 26)
        self.assertEqual(checkpoint.default_van_der_waals_count, 27)
        self.assertAlmostEqual(checkpoint.deuterium_strength_ratio, 1.0e-5, places=14)

    def test_window_filter_keeps_reaching_wings_outside_the_requested_centers(self) -> None:
        checkpoint = window_selection_checkpoint()
        self.assertEqual(checkpoint.center_inside_count, 45)
        self.assertEqual(checkpoint.selected_count, 100)
        self.assertEqual(checkpoint.outside_center_but_selected_count, 55)
        expected = (
            checkpoint.wavelength_nm
            >= checkpoint.start_wavelength_nm - checkpoint.margin_nm
        ) & (
            checkpoint.wavelength_nm
            <= checkpoint.end_wavelength_nm + checkpoint.margin_nm
        )
        np.testing.assert_array_equal(checkpoint.selected_with_margin, expected)

    def test_compiled_catalog_records_real_center_collisions(self) -> None:
        checkpoint = catalog_layout_checkpoint()
        self.assertEqual(checkpoint.wavelength_nm.size, 121)
        self.assertEqual(checkpoint.line_count, 100)
        self.assertEqual(checkpoint.type_segments, {0: (0, 100)})
        self.assertEqual(checkpoint.on_grid_center_count, 45)
        self.assertEqual(checkpoint.unique_on_grid_center_count, 30)
        self.assertEqual(checkpoint.colliding_on_grid_record_count, 15)

    def test_scatter_add_and_private_reduction_preserve_collisions(self) -> None:
        checkpoint = scatter_add_checkpoint()
        self.assertFalse(np.array_equal(checkpoint.overwritten, checkpoint.scatter_added))
        np.testing.assert_array_equal(checkpoint.reduced, checkpoint.scatter_added)
        self.assertEqual(checkpoint.overwritten[4], 3.0)
        self.assertEqual(checkpoint.scatter_added[4], 5.5)

    def test_exact_cpu_forest_is_finite_nonnegative_and_once_stimulated(self) -> None:
        checkpoint = run_atomic_forest()
        self.assertEqual(checkpoint.selected_line_count, 100)
        self.assertEqual(checkpoint.metal_line_count, 100)
        self.assertEqual(checkpoint.center_collision_count, 15)
        self.assertEqual(checkpoint.net_line_mass_absorption_coefficient.dtype, np.float32)
        self.assertTrue(np.isfinite(checkpoint.net_line_mass_absorption_coefficient).all())
        self.assertGreaterEqual(
            float(np.min(checkpoint.net_line_mass_absorption_coefficient)),
            0.0,
        )
        self.assertTrue(
            np.all(
                checkpoint.net_line_mass_absorption_coefficient
                <= checkpoint.gross_line_mass_absorption_coefficient
            )
        )
        self.assertGreater(checkpoint.peak_net_opacity_cm2_per_g, 0.0)

    def test_batched_and_explicit_wing_walks_agree_on_the_compact_forest(self) -> None:
        comparison = compare_forest_wing_modes()
        self.assertLessEqual(comparison.maximum_absolute_difference, 5.0e-7)
        self.assertLessEqual(comparison.maximum_relative_difference, 2.0e-6)

    def test_hydrogen_fine_structure_and_support_limits_are_explicit(self) -> None:
        checkpoint = hydrogen_checkpoint()
        self.assertEqual(checkpoint.component_offset_hz.size, 7)
        self.assertAlmostEqual(float(np.sum(checkpoint.component_weight)), 1.0)
        self.assertGreater(checkpoint.isotope_separation_nm, 0.0)
        self.assertEqual(checkpoint.synthesis_minimum_supported_lower_level, 2)
        self.assertEqual(checkpoint.atmosphere_supported_lower_level_range, (1, 100))
        self.assertTrue(np.all(np.diff(checkpoint.merge_wavenumber_cm) > 0.0))

    def test_route_ledger_keeps_cor_prd_and_do_metal_semantics_honest(self) -> None:
        by_code = {int(row[0]): row for row in route_ledger()}
        self.assertIn("ordinary LTE", by_code[3][2])
        self.assertIn("skipped", by_code[2][2])
        self.assertEqual(by_code[2][4], 0)
        self.assertEqual(by_code[1][4], 1)

    def test_autoionizing_example_uses_raw_shore_parameters(self) -> None:
        checkpoint = autoionizing_checkpoint()
        self.assertEqual(checkpoint.source_row, 1_012_945)
        self.assertGreater(checkpoint.radiative_width, 0.0)
        self.assertGreater(checkpoint.shore_baseline, 0.0)
        self.assertGreaterEqual(float(np.min(checkpoint.positive_profile_ratio)), 0.0)
        self.assertAlmostEqual(float(np.max(checkpoint.positive_profile_ratio)), 1.0)


if __name__ == "__main__":
    unittest.main()
