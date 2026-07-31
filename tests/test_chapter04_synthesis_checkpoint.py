"""Focused tests for the exact Chapter 4 synthesis Newton checkpoint."""

from __future__ import annotations

import ast
import inspect
import unittest

import numpy as np
import torch

from book.chapter04_synthesis_checkpoint import (
    ATMOSPHERE_CATALOG,
    CHAPTER04_INPUTS,
    GROUND_PARTITION_FIXTURE,
    POSITIVITY_FLOOR_DIVISOR,
    SYNTHESIS_CATALOG,
    SYNTHESIS_TABLES,
    load_checkpoint_inputs,
    synthesis_newton_checkpoint,
)
from payne_zero_synthesis import molecular_equilibrium


class Chapter04SynthesisCheckpointTests(unittest.TestCase):
    """Pin the physical, numerical, and orchestration evidence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint = synthesis_newton_checkpoint()

    def test_uses_only_repository_owned_inputs_and_exact_local_sources(
        self,
    ) -> None:
        for path in (
            ATMOSPHERE_CATALOG,
            CHAPTER04_INPUTS,
            GROUND_PARTITION_FIXTURE,
            SYNTHESIS_CATALOG,
            SYNTHESIS_TABLES,
        ):
            self.assertTrue(path.is_file())
            self.assertTrue(path.is_relative_to(CHAPTER04_INPUTS.parents[2]))
        source = inspect.getsource(
            __import__(
                "book.chapter04_synthesis_checkpoint",
                fromlist=["synthesis_newton_checkpoint"],
            )
        )
        self.assertNotIn("/Users/", source)
        self.assertNotIn("data/goldens", source.lower())
        numeric_literals = {
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, float)
        }
        self.assertIn(1.0e-300, numeric_literals)
        self.assertNotIn(1.0e-30, numeric_literals)

    def test_checkpoint_is_exact_cpu_float64_density_space(self) -> None:
        checkpoint = self.checkpoint
        self.assertEqual(checkpoint.depth_index, 0)
        self.assertEqual(checkpoint.device_type, "cpu")
        self.assertEqual(checkpoint.torch_dtype, "torch.float64")
        self.assertEqual(checkpoint.density_units, "cm^-3")
        self.assertEqual(checkpoint.equation_species_codes.shape, (23,))
        self.assertEqual(checkpoint.residual_cm3.shape, (23,))
        self.assertEqual(checkpoint.jacrev_jacobian.shape, (23, 23))
        self.assertTrue(np.all(np.isfinite(checkpoint.residual_cm3)))
        self.assertTrue(np.all(np.isfinite(checkpoint.jacrev_jacobian)))
        self.assertTrue(np.all(checkpoint.density_before_cm3 > 0.0))

    def test_independent_finite_difference_checks_jacrev_at_physical_state(
        self,
    ) -> None:
        checkpoint = self.checkpoint
        self.assertEqual(
            checkpoint.finite_difference_jacobian.shape,
            checkpoint.jacrev_jacobian.shape,
        )
        self.assertLess(
            checkpoint.max_finite_difference_scale_relative_error,
            1.0e-7,
        )
        self.assertGreater(
            checkpoint.max_finite_difference_absolute_error,
            0.0,
        )

    def test_column_scaled_solve_is_the_exact_physical_step(self) -> None:
        checkpoint = self.checkpoint
        np.testing.assert_array_equal(
            checkpoint.physical_step_cm3,
            checkpoint.column_scale_cm3 * checkpoint.fractional_step,
        )
        np.testing.assert_allclose(
            checkpoint.scaled_jacobian,
            checkpoint.jacrev_jacobian
            * checkpoint.column_scale_cm3[None, :],
            rtol=0.0,
            atol=0.0,
        )
        self.assertLess(
            checkpoint.unscaled_linear_residual_relative_norm,
            1.0e-14,
        )
        self.assertLess(
            checkpoint.scaled_linear_residual_relative_norm,
            1.0e-14,
        )
        expected_relative_update = np.abs(checkpoint.physical_step_cm3) / (
            np.maximum(
                np.abs(checkpoint.density_before_cm3),
                torch.finfo(torch.float64).tiny,
            )
        )
        np.testing.assert_array_equal(
            checkpoint.undamped_relative_update,
            expected_relative_update,
        )
        self.assertEqual(
            checkpoint.needs_more_iterations,
            bool(
                np.any(
                    expected_relative_update
                    > checkpoint.convergence_tolerance
                )
            ),
        )
        self.assertEqual(
            checkpoint.maximum_fractional_update,
            float(np.max(expected_relative_update)),
        )

    def test_first_physical_step_uses_exact_multiplicative_floor(self) -> None:
        checkpoint = self.checkpoint
        expected_threshold = (
            checkpoint.density_before_cm3 / POSITIVITY_FLOOR_DIVISOR
        )
        expected_mask = (
            checkpoint.candidate_density_cm3 < expected_threshold
        )
        expected_after = np.where(
            expected_mask,
            expected_threshold,
            checkpoint.candidate_density_cm3,
        )
        np.testing.assert_array_equal(
            checkpoint.positivity_threshold_cm3,
            expected_threshold,
        )
        np.testing.assert_array_equal(
            checkpoint.positivity_floor_mask,
            expected_mask,
        )
        np.testing.assert_array_equal(
            checkpoint.density_after_cm3,
            expected_after,
        )
        self.assertEqual(int(expected_mask.sum()), 2)
        self.assertTrue(np.all(checkpoint.density_after_cm3 > 0.0))
        self.assertFalse(checkpoint.sign_change_mask.any())

    def test_controlled_opposite_sign_step_executes_exact_damping(self) -> None:
        witness = self.checkpoint.sign_damping
        self.assertNotEqual(witness.undamped_step_cm3, 0.0)
        self.assertLess(
            witness.previous_step_cm3 * witness.undamped_step_cm3,
            0.0,
        )
        self.assertTrue(witness.sign_change)
        self.assertEqual(witness.damping_factor, 0.69)
        self.assertEqual(
            witness.damped_step_cm3,
            0.69 * witness.undamped_step_cm3,
        )

    def test_physical_and_controlled_nonfinite_paths_are_separate(self) -> None:
        checkpoint = self.checkpoint
        self.assertTrue(checkpoint.physical_log_products_are_finite)
        self.assertTrue(
            checkpoint.physical_pre_replacement_products_are_finite
        )
        self.assertEqual(checkpoint.physical_nonfinite_replacement_count, 0)

        overflow = checkpoint.nonfinite_product
        self.assertTrue(overflow.natural_log_product_is_finite)
        self.assertFalse(overflow.pre_replacement_product_is_finite)
        self.assertTrue(overflow.replacement_applied)
        self.assertEqual(overflow.post_replacement_product, 0.0)
        self.assertTrue(overflow.residual_after_replacement_is_finite)

    def test_real_short_solve_observes_jacrev_newton_and_final_vmap(
        self,
    ) -> None:
        order = self.checkpoint.call_order
        self.assertEqual(
            order.observed_events,
            (
                "jacrev transform",
                "jacrev evaluation",
                "Newton step",
                "jacrev evaluation",
                "Newton step",
                "final vmap transform",
                "final vmap evaluation",
            ),
        )
        self.assertEqual(order.jacrev_evaluation_count, 2)
        self.assertEqual(order.newton_step_call_count, 2)
        self.assertEqual(order.final_vmap_evaluation_count, 1)
        self.assertTrue(order.jacrev_precedes_each_newton_step)
        self.assertTrue(order.final_vmap_follows_all_newton_steps)
        self.assertTrue(order.patched_identities_restored)
        self.assertIn(
            "undamped relative-update convergence test",
            order.operation_order,
        )

    def test_real_short_solve_executes_a_true_chain_restart(self) -> None:
        order = self.checkpoint.call_order
        np.testing.assert_array_equal(
            order.executed_depth_indices,
            np.array([0, 1], dtype=np.int64),
        )
        self.assertEqual(order.explicit_chain_length, 1)
        self.assertEqual(order.explicit_restart_depth_index, 1)
        self.assertEqual(
            order.observed_step_input_densities_cm3.shape,
            (2, 23),
        )
        np.testing.assert_array_equal(
            order.observed_step_input_densities_cm3[0],
            self.checkpoint.density_before_cm3,
        )
        np.testing.assert_array_equal(
            order.observed_step_input_densities_cm3[1],
            order.expected_restart_density_cm3,
        )
        self.assertTrue(order.explicit_restart_seed_matches)

    def test_scoped_production_instrumentation_restores_exact_identities(
        self,
    ) -> None:
        original_jacrev = torch.func.jacrev
        original_vmap = torch.func.vmap
        original_step = molecular_equilibrium._newton_step
        synthesis_newton_checkpoint()
        self.assertIs(torch.func.jacrev, original_jacrev)
        self.assertIs(torch.func.vmap, original_vmap)
        self.assertIs(molecular_equilibrium._newton_step, original_step)

    def test_both_catalogs_independently_decode_the_same_14_negative_ions(
        self,
    ) -> None:
        evidence = self.checkpoint.shared_negative_ions
        expected_codes = np.array(
            [
                100.0,
                600.0,
                800.0,
                900.0,
                1300.0,
                1600.0,
                1700.0,
                2600.0,
                10100.0,
                10800.0,
                60600.0,
                60700.0,
                70700.0,
                80800.0,
            ]
        )
        np.testing.assert_array_equal(evidence.molecule_codes, expected_codes)
        self.assertEqual(evidence.molecule_codes.shape, (14,))
        np.testing.assert_array_equal(
            evidence.atmosphere_row_indices,
            np.array(
                [
                    94,
                    115,
                    131,
                    142,
                    144,
                    147,
                    151,
                    152,
                    153,
                    163,
                    165,
                    166,
                    168,
                    169,
                ],
                dtype=np.int64,
            ),
        )
        np.testing.assert_array_equal(
            evidence.synthesis_row_indices,
            np.array(
                [
                    94,
                    116,
                    132,
                    143,
                    145,
                    148,
                    152,
                    153,
                    154,
                    164,
                    169,
                    173,
                    178,
                    181,
                ],
                dtype=np.int64,
            ),
        )
        expected_components_by_record = (
            (1, 100),
            (6, 100),
            (8, 100),
            (9, 100),
            (13, 100),
            (16, 100),
            (17, 100),
            (26, 100),
            (1, 1, 100),
            (1, 8, 100),
            (6, 6, 100),
            (6, 7, 100),
            (7, 7, 100),
            (8, 8, 100),
        )
        expected_offsets = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.cumsum(
                    [len(record) for record in expected_components_by_record]
                ),
            )
        )
        expected_components = np.asarray(
            [
                component
                for record in expected_components_by_record
                for component in record
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(
            evidence.component_offsets,
            expected_offsets,
        )
        np.testing.assert_array_equal(
            evidence.atmosphere_component_species_codes,
            expected_components,
        )
        np.testing.assert_array_equal(
            evidence.atmosphere_component_species_codes,
            evidence.synthesis_component_species_codes,
        )
        self.assertTrue(
            np.all(evidence.atmosphere_ordinary_electron_count == 1)
        )
        self.assertTrue(
            np.all(evidence.synthesis_ordinary_electron_count == 1)
        )
        self.assertTrue(
            np.all(evidence.atmosphere_inverse_electron_count == 0)
        )
        self.assertTrue(
            np.all(evidence.synthesis_inverse_electron_count == 0)
        )
        self.assertTrue(evidence.atmosphere_ordinary_electron_is_last.all())
        self.assertTrue(evidence.synthesis_ordinary_electron_is_last.all())
        for record_index in range(14):
            stop = evidence.component_offsets[record_index + 1]
            self.assertEqual(
                evidence.synthesis_component_species_codes[stop - 1],
                100,
            )
        self.assertFalse(
            np.array_equal(
                evidence.atmosphere_row_indices,
                evidence.synthesis_row_indices,
            )
        )

    def test_net_electron_coefficient_and_controlled_residual_signs(
        self,
    ) -> None:
        evidence = self.checkpoint.shared_negative_ions
        expected_negative_coefficient = (
            evidence.synthesis_electron_component_multiplicity
            + evidence.synthesis_inverse_electron_power
            - 2.0 * evidence.synthesis_negative_ion_flag
        )
        np.testing.assert_array_equal(
            evidence.synthesis_electron_component_multiplicity,
            evidence.synthesis_ordinary_electron_count,
        )
        np.testing.assert_array_equal(
            evidence.synthesis_inverse_electron_power,
            evidence.synthesis_inverse_electron_count,
        )
        np.testing.assert_array_equal(
            evidence.synthesis_negative_ion_flag,
            np.ones(14),
        )
        np.testing.assert_array_equal(
            evidence.synthesis_net_electron_coefficient,
            expected_negative_coefficient,
        )
        np.testing.assert_array_equal(
            evidence.synthesis_net_electron_coefficient,
            -np.ones(14),
        )

        negative = self.checkpoint.negative_ion_residual
        self.assertEqual(negative.molecule_code, 100.0)
        np.testing.assert_array_equal(
            negative.component_species_codes,
            np.array([1, 100]),
        )
        self.assertEqual(negative.ordinary_electron_count, 1)
        self.assertEqual(negative.inverse_electron_count, 0)
        self.assertEqual(negative.component_species_codes[-1], 100)
        self.assertEqual(negative.net_electron_coefficient, -1.0)
        self.assertEqual(
            negative.observed_electron_residual_delta,
            -negative.molecular_term,
        )

        positive = self.checkpoint.positive_ion_residual
        self.assertEqual(positive.molecule_code, 1.01)
        np.testing.assert_array_equal(
            positive.component_species_codes,
            np.array([1, 101]),
        )
        self.assertEqual(positive.ordinary_electron_count, 0)
        self.assertGreater(positive.inverse_electron_count, 0)
        self.assertEqual(positive.component_species_codes[-1], 101)
        self.assertGreater(positive.net_electron_coefficient, 0.0)
        self.assertEqual(
            positive.observed_electron_residual_delta,
            positive.net_electron_coefficient * positive.molecular_term,
        )

    def test_later_depth_requires_an_explicit_ordered_chain_restart(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit_chain_restart"):
            synthesis_newton_checkpoint(depth_index=1)
        restarted = synthesis_newton_checkpoint(
            depth_index=1,
            explicit_chain_restart=True,
        )
        self.assertEqual(restarted.depth_index, 1)
        self.assertEqual(restarted.call_order.explicit_chain_length, 1)
        self.assertTrue(restarted.call_order.explicit_restart_seed_matches)

    def test_input_loader_returns_independent_arrays(self) -> None:
        first = load_checkpoint_inputs()
        second = load_checkpoint_inputs()
        first["temperature"][0] = -1.0
        self.assertGreater(second["temperature"][0], 0.0)
        self.assertEqual(second["elemental_abundances"].shape, (99,))

    def test_all_exposed_numpy_evidence_is_read_only(self) -> None:
        checkpoint = self.checkpoint
        arrays = (
            checkpoint.density_before_cm3,
            checkpoint.residual_cm3,
            checkpoint.jacrev_jacobian,
            checkpoint.finite_difference_jacobian,
            checkpoint.column_scale_cm3,
            checkpoint.physical_step_cm3,
            checkpoint.density_after_cm3,
            checkpoint.undamped_relative_update,
            checkpoint.shared_negative_ions.molecule_codes,
            checkpoint.call_order.observed_step_input_densities_cm3,
        )
        self.assertTrue(all(not array.flags.writeable for array in arrays))


if __name__ == "__main__":
    unittest.main()
