"""Transparent numerical checks that precede Chapter 5 production routes."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from book.chapter05_runtime import edge_triplet_checkpoint
from book.chapter05_teaching import (
    njit_frequency_columns,
    prange_frequency_columns,
    python_frequency_columns,
    reconstruct_positive_opacity,
    three_point_edge_basis,
)


class Chapter05TeachingTests(unittest.TestCase):
    """Keep the small reader-built kernels honest and exact."""

    def test_three_point_basis_matches_the_exact_edge_checkpoint(self) -> None:
        requested = np.asarray([400.0, 500.0, 650.0, 900.0])
        exact = edge_triplet_checkpoint(requested)
        with np.load(
            "data/static/synthesis_tables/continuum_edge_grid.npz",
            allow_pickle=False,
        ) as edge:
            wavelength = np.asarray(edge["continuum_edge_wavelength_nm"])
        left = wavelength[exact.interval_index]
        right = wavelength[exact.interval_index + 1]
        basis = three_point_edge_basis(requested, left, right)
        np.testing.assert_array_equal(basis.left, exact.left_basis)
        np.testing.assert_array_equal(basis.midpoint, exact.midpoint_basis)
        np.testing.assert_array_equal(basis.right, exact.right_basis)
        np.testing.assert_allclose(basis.sum, 1.0, rtol=0.0, atol=5.0e-15)

    def test_log_reconstruction_reproduces_all_three_nodes(self) -> None:
        left = 400.0
        right = 500.0
        midpoint = 0.5 * (left + right)
        target = np.asarray([left, midpoint, right])
        sampled = np.asarray([1.0e-4, 3.0e-2, 2.0e-1])
        basis = three_point_edge_basis(target, left, right)
        reconstructed = reconstruct_positive_opacity(sampled, basis)
        np.testing.assert_allclose(
            reconstructed,
            sampled,
            rtol=3.0e-15,
            atol=0.0,
        )
        self.assertTrue(np.all(reconstructed > 0.0))

    def test_log_reconstruction_applies_floor_only_before_logarithm(self) -> None:
        basis = three_point_edge_basis(np.asarray([450.0]), 400.0, 500.0)
        reconstructed = reconstruct_positive_opacity(
            np.asarray([0.0, 1.0e-20, 1.0e-10]),
            basis,
        )
        self.assertEqual(float(reconstructed[0]), 1.0e-20)

    def test_edge_helpers_reject_out_of_interval_and_negative_opacity(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "inside its interval"):
            three_point_edge_basis(np.asarray([399.0]), 400.0, 500.0)
        basis = three_point_edge_basis(np.asarray([450.0]), 400.0, 500.0)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            reconstruct_positive_opacity(
                np.asarray([1.0e-4, -1.0e-5, 1.0e-3]),
                basis,
            )

    def test_python_njit_and_prange_own_the_same_columns(self) -> None:
        number_density = np.geomspace(1.0e10, 1.0e17, 12)
        mass_density = np.geomspace(1.0e-12, 1.0e-5, 12)
        cross_section = np.geomspace(1.0e-22, 1.0e-17, 31)
        expected = python_frequency_columns(
            number_density,
            cross_section,
            mass_density,
        )
        serial = njit_frequency_columns(
            number_density,
            cross_section,
            mass_density,
        )
        parallel = prange_frequency_columns(
            number_density,
            cross_section,
            mass_density,
        )
        np.testing.assert_array_equal(serial, expected)
        np.testing.assert_array_equal(parallel, expected)
        np.testing.assert_array_equal(
            parallel[:, 7],
            number_density * cross_section[7] / mass_density,
        )

    def test_parallel_source_uses_prange_only_for_frequency_owner(self) -> None:
        source = inspect.getsource(prange_frequency_columns.py_func)
        self.assertIn("for frequency_index in prange", source)
        self.assertIn("for depth_index in range", source)
        self.assertNotIn("for depth_index in prange", source)

    def test_transparent_python_kernel_rejects_nonphysical_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            python_frequency_columns(
                np.asarray([1.0, 0.0]),
                np.asarray([1.0]),
                np.asarray([1.0, 1.0]),
            )
        with self.assertRaisesRegex(ValueError, "depth shapes"):
            python_frequency_columns(
                np.asarray([1.0, 2.0]),
                np.asarray([1.0]),
                np.asarray([1.0]),
            )


if __name__ == "__main__":
    unittest.main()
