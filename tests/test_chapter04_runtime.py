"""Reader-runtime gates for the progressive Chapter 4 molecular routes."""

from __future__ import annotations

import unittest
import warnings
from unittest import mock

import numpy as np

from book.chapter04_runtime import (
    atmosphere_continuation_checkpoint,
    atmosphere_jacobian_checkpoint,
    atmosphere_linear_solve_checkpoint,
    atmosphere_newton_update_witness,
    build_molecular_atmosphere,
    build_public_molecular_lane_checkpoint,
    compute_atmosphere_molecular_state,
    compute_synthesis_molecular_state,
    h2_formation_policy_curves,
    h2_partition_table_probe,
    load_molecular_inputs,
    molecular_catalog_summary,
    molecular_route_boundary_checkpoint,
    SYNTHESIS_CONTINUUM_EDGE_GRID,
    validate_molecular_inputs,
)


class Chapter04RuntimeTests(unittest.TestCase):
    """Require self-contained inputs and the exact local state shapes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_molecular_inputs()

    def test_fixture_is_input_only_and_depth_ordered(self) -> None:
        forbidden = (
            "molecular_populations",
            "molecular_equation_densities",
            "mass_density",
            "total_nuclei_number_density",
        )
        self.assertTrue(np.all(np.diff(self.inputs["column_mass"]) > 0.0))
        for name in forbidden:
            self.assertNotIn(name, self.inputs)

    def test_validation_rejects_bad_shape_order_and_abundance(self) -> None:
        bad_shape = dict(self.inputs)
        bad_shape["temperature"] = self.inputs["temperature"][:-1]
        with self.assertRaises(ValueError):
            validate_molecular_inputs(bad_shape)

        bad_order = {
            name: np.asarray(value).copy() for name, value in self.inputs.items()
        }
        bad_order["column_mass"][[1, 2]] = bad_order["column_mass"][[2, 1]]
        with self.assertRaises(ValueError):
            validate_molecular_inputs(bad_order)

        bad_abundance = {
            name: np.asarray(value).copy() for name, value in self.inputs.items()
        }
        bad_abundance["elemental_abundances"][5] = -1.0
        with self.assertRaises(ValueError):
            validate_molecular_inputs(bad_abundance)

    def test_zero_microturbulence_is_physical_but_negative_is_not(self) -> None:
        zero_microturbulence = {
            name: np.asarray(value).copy() for name, value in self.inputs.items()
        }
        zero_microturbulence["microturbulence"][2] = 0.0
        validated = validate_molecular_inputs(zero_microturbulence)
        self.assertEqual(validated["microturbulence"][2], 0.0)

        negative_microturbulence = {
            name: np.asarray(value).copy() for name, value in self.inputs.items()
        }
        negative_microturbulence["microturbulence"][2] = -1.0
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            validate_molecular_inputs(negative_microturbulence)

    def test_atmosphere_builder_preserves_declared_columns(self) -> None:
        atmosphere = build_molecular_atmosphere(self.inputs)
        np.testing.assert_array_equal(
            atmosphere.temperature,
            self.inputs["temperature"],
        )
        np.testing.assert_array_equal(
            atmosphere.electron_density,
            self.inputs["electron_density_seed"],
        )
        self.assertTrue(np.all(atmosphere.rosseland_opacity == 1.0))
        self.assertEqual(
            atmosphere.metadata["pressure_iteration_enabled"],
            "1",
        )

    def test_catalogs_have_distinct_exact_active_extents(self) -> None:
        summary = molecular_catalog_summary()
        np.testing.assert_array_equal(
            summary["atmosphere_counts"],
            [170, 23, 481],
        )
        np.testing.assert_array_equal(
            summary["synthesis_counts"],
            [190, 23, 548],
        )
        self.assertEqual(summary["synthesis_only_code_keys"].size, 20)
        self.assertEqual(summary["shared_code_keys"].size, 170)
        self.assertEqual(np.count_nonzero(summary["shared_semantic_mismatch"]), 0)
        self.assertEqual(np.count_nonzero(summary["shared_row_reordered"]), 64)

    def test_h2_plot_adapter_preserves_all_three_exact_boundaries(self) -> None:
        edges = np.array(
            [
                100.0,
                101.0,
                9000.0,
                np.nextafter(9000.0, np.inf),
                10000.0,
                np.nextafter(10000.0, np.inf),
                19899.0,
                19900.0,
                20000.0,
                np.nextafter(20000.0, np.inf),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            policies = h2_formation_policy_curves(edges)
        np.testing.assert_array_equal(
            policies["provisional public H2"] > 0.0,
            [True, True, True, False, False, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            policies["synthesis catalog policy"] > 0.0,
            [True, True, True, True, True, False, False, False, False, False],
        )
        np.testing.assert_array_equal(
            policies["atmosphere H2 policy"] > 0.0,
            [True, True, True, True, True, True, True, True, True, False],
        )
        self.assertNotEqual(
            policies["atmosphere H2 policy"][6],
            policies["atmosphere H2 policy"][7],
        )
        np.testing.assert_allclose(
            h2_partition_table_probe(edges)[[0, 1, 6, 7]],
            [0.667, 0.67373, 658.10609, 658.16],
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertEqual(
            h2_partition_table_probe(np.array([99.0, 100.0]))[0],
            h2_partition_table_probe(np.array([99.0, 100.0]))[1],
        )
        self.assertEqual(
            h2_partition_table_probe(np.array([19900.0, 19901.0]))[0],
            h2_partition_table_probe(np.array([19900.0, 19901.0]))[1],
        )
        with self.assertRaises(ValueError):
            h2_formation_policy_curves(np.array([0.0]))
        with self.assertRaises(ValueError):
            h2_partition_table_probe(np.array([np.nan]))

    def test_atmosphere_route_returns_active_unpadded_arrays(self) -> None:
        result = compute_atmosphere_molecular_state(self.inputs)
        self.assertIsNotNone(result.molecular_state)
        molecular = result.molecular_state
        self.assertEqual(molecular.molecular_populations.shape, (6, 170))
        self.assertEqual(
            molecular.partition_normalized_molecular_populations.shape,
            (6, 170),
        )
        self.assertEqual(
            molecular.molecular_equation_densities.shape,
            (6, 23),
        )
        self.assertTrue(np.all(np.isfinite(molecular.molecular_populations)))

    def test_diagnostic_h2_and_atom_only_fallback_routes_execute(self) -> None:
        route = molecular_route_boundary_checkpoint(self.inputs)
        self.assertEqual(route.live_shape_error_type, "ValueError")
        self.assertEqual(route.padded_molecule_code_shape, (200,))
        self.assertEqual(route.active_molecular_population_shape, (6, 170))
        np.testing.assert_array_equal(
            route.h2_mixed_output,
            route.h2_mixed_catalog_input,
        )
        self.assertEqual(route.h2_mixed_output[1], 0.0)
        np.testing.assert_array_equal(
            route.h2_all_zero_output,
            route.h2_no_catalog_output,
        )
        self.assertEqual(route.fallback_fixed_eos_call_count, 1)
        self.assertEqual(route.fallback_molecular_solve_call_count, 1)

    def test_atmosphere_structural_doppler_infinities_are_exact(self) -> None:
        result = compute_atmosphere_molecular_state(self.inputs)
        widths = result.fractional_doppler_widths
        nonfinite_rows, nonfinite_slots = np.where(~np.isfinite(widths))
        np.testing.assert_array_equal(
            np.unique(nonfinite_slots),
            [919, 927],
        )
        self.assertEqual(nonfinite_rows.size, 12)
        self.assertTrue(np.all(np.isposinf(widths[:, [919, 927]])))
        runtime = result.runtime_state
        np.testing.assert_array_equal(
            runtime.major_isotope_mass_amu[[919, 927]],
            [0.0, 0.0],
        )
        np.testing.assert_array_equal(
            runtime.ion_stage_populations_by_packed_slot[:, [919, 927]],
            0.0,
        )
        np.testing.assert_array_equal(
            runtime.partition_normalized_populations_by_packed_slot[:, [919, 927]],
            0.0,
        )

    def test_synthesis_full_and_fixed_routes_keep_their_public_claims(self) -> None:
        full = compute_synthesis_molecular_state(
            self.inputs,
            fixed_electron_density=False,
        )
        fixed = compute_synthesis_molecular_state(
            self.inputs,
            fixed_electron_density=True,
        )
        self.assertEqual(full.molecular_populations.shape, (6, 200))
        self.assertEqual(fixed.molecular_equation_densities.shape, (6, 30))
        np.testing.assert_array_equal(
            fixed.electron_density,
            self.inputs["electron_density_seed"],
        )
        self.assertFalse(
            np.array_equal(
                full.electron_density,
                self.inputs["electron_density_seed"],
            )
        )

    def test_public_builder_maps_only_the_owned_normalized_stage5_cells(
        self,
    ) -> None:
        checkpoint = build_public_molecular_lane_checkpoint(self.inputs)
        self.assertEqual(checkpoint.fixed_eos_call_count, 1)
        self.assertEqual(checkpoint.molecular_solve_call_count, 1)
        self.assertTrue(checkpoint.reused_fixed_molecular_arrays)
        self.assertEqual(checkpoint.edge_grid_call_count, 1)
        self.assertEqual(checkpoint.line_species_codes.shape, (54,))
        self.assertEqual(
            checkpoint.equilibrium_code_offsets.shape,
            (55,),
        )
        self.assertEqual(checkpoint.equilibrium_codes.shape, (54,))
        self.assertEqual(
            np.unique(checkpoint.line_species_codes).size,
            54,
        )
        self.assertEqual(np.unique(checkpoint.public_columns).size, 54)
        np.testing.assert_array_equal(
            checkpoint.public_columns,
            checkpoint.line_species_codes // 6 - 1,
        )
        self.assertTrue(np.all(checkpoint.public_columns >= 0))
        self.assertTrue(np.all(checkpoint.public_columns < 139))
        self.assertEqual(checkpoint.equilibrium_code_offsets[0], 0)
        self.assertEqual(checkpoint.equilibrium_code_offsets[-1], 54)
        self.assertTrue(np.all(np.diff(checkpoint.equilibrium_code_offsets) == 1))
        from payne_zero_synthesis import molecular_equilibrium

        for index, line_species_code in enumerate(checkpoint.line_species_codes):
            start = checkpoint.equilibrium_code_offsets[index]
            stop = checkpoint.equilibrium_code_offsets[index + 1]
            np.testing.assert_array_equal(
                checkpoint.equilibrium_codes[start:stop],
                molecular_equilibrium._SPECIES_CODE_TO_MOLECULE_CODES[
                    int(line_species_code)
                ],
            )
        self.assertEqual(
            checkpoint.structured_atmosphere["partition_normalized_populations"].shape,
            (6, 6, 139),
        )
        self.assertEqual(
            np.count_nonzero(checkpoint.public_columns <= 98),
            51,
        )
        np.testing.assert_array_equal(
            checkpoint.public_columns[checkpoint.public_columns > 98],
            [129, 130, 131],
        )
        self.assertEqual(checkpoint.co_line_species_code, 276)
        self.assertEqual(checkpoint.co_equilibrium_code, 608.0)
        self.assertEqual(checkpoint.co_public_column, 45)
        np.testing.assert_array_equal(
            checkpoint.co_component_species_codes,
            [6, 8],
        )
        self.assertEqual(checkpoint.co_leading_coefficient, 11.091)
        self.assertEqual(
            checkpoint.co_molecular_mass_amu,
            28.009999999999998,
        )
        np.testing.assert_array_equal(
            checkpoint.ion_cube_after,
            checkpoint.ion_cube_before,
        )
        np.testing.assert_array_equal(
            checkpoint.normalized_delta[:, :5, :],
            0.0,
        )
        np.testing.assert_array_equal(
            checkpoint.normalized_delta[:, 5, ~checkpoint.owned_stage5_mask],
            0.0,
        )
        np.testing.assert_array_equal(
            checkpoint.partition_cube_after[:, 5, checkpoint.public_columns],
            checkpoint.no_ground_line_populations,
        )
        self.assertEqual(
            np.count_nonzero(checkpoint.ground_discrimination_mask),
            11,
        )
        self.assertTrue(
            np.all(checkpoint.co_raw_population != checkpoint.co_normalized_population)
        )
        np.testing.assert_array_equal(
            checkpoint.co_independent_population,
            checkpoint.co_normalized_population,
        )
        with np.load(
            SYNTHESIS_CONTINUUM_EDGE_GRID,
            allow_pickle=False,
        ) as edge_archive:
            edge_names = {
                "signed_continuum_edge_frequency_hz": (
                    "signed_continuum_edge_frequency_hz"
                ),
                "continuum_edge_wavelength_nm": ("continuum_edge_wavelength_nm"),
                "continuum_edge_midpoint_wavelength_nm": (
                    "continuum_edge_midpoint_wavelength_nm"
                ),
                "continuum_edge_interval_width_squared_over_two_nm2": (
                    "continuum_edge_interval_width_squared_over_two_nm2"
                ),
            }
            for structured_name, archive_name in edge_names.items():
                np.testing.assert_array_equal(
                    checkpoint.structured_atmosphere[structured_name],
                    edge_archive[archive_name],
                )
        self.assertEqual(
            checkpoint.hard_coded_molecular_atomic_masses_sha256,
            "b6b870f0cdb3ea49fc4977dfc1fcff0ed1c16747922940d302644aefe28b7636",
        )

    def test_public_builder_observers_restore_after_failure(self) -> None:
        from payne_zero_synthesis import equation_of_state
        from payne_zero_synthesis import molecular_equilibrium
        from payne_zero_synthesis import pipeline

        original_fixed = equation_of_state.solve_population_state_at_electron_density
        original_molecular = molecular_equilibrium.solve_molecular_equilibrium
        original_edges = pipeline._build_edge_grid
        with mock.patch.object(
            pipeline,
            "build_structured_atmosphere_from_columns",
            side_effect=RuntimeError("controlled public-builder failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "controlled"):
                build_public_molecular_lane_checkpoint(self.inputs)
        self.assertIs(
            equation_of_state.solve_population_state_at_electron_density,
            original_fixed,
        )
        self.assertIs(
            molecular_equilibrium.solve_molecular_equilibrium,
            original_molecular,
        )
        self.assertIs(pipeline._build_edge_grid, original_edges)

    def test_exact_atmosphere_jacobian_matches_finite_difference(self) -> None:
        checkpoint = atmosphere_jacobian_checkpoint(
            self.inputs,
            depth_index=2,
        )
        self.assertEqual(checkpoint.depth_index, 2)
        self.assertEqual(checkpoint.temperature_k, 4500.0)
        self.assertEqual(checkpoint.gas_pressure, 1.0e4)
        self.assertEqual(checkpoint.residual.shape, (23,))
        self.assertEqual(checkpoint.analytic_jacobian.shape, (23, 23))
        self.assertEqual(
            checkpoint.finite_difference_jacobian.shape,
            (23, 23),
        )
        np.testing.assert_array_equal(
            checkpoint.row_species_codes,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                11,
                12,
                13,
                14,
                16,
                17,
                19,
                20,
                22,
                23,
                24,
                26,
                100,
            ],
        )
        np.testing.assert_array_equal(
            checkpoint.particle_carbon_oxygen_indices,
            [0, 6, 8],
        )
        self.assertLess(
            checkpoint.max_column_scale_relative_error,
            5.0e-4,
        )
        np.testing.assert_allclose(
            checkpoint.particle_carbon_oxygen_analytic_block,
            checkpoint.particle_carbon_oxygen_finite_difference_block,
            rtol=5.0e-5,
            atol=5.0e-5,
        )
        self.assertLess(
            np.max(np.abs(checkpoint.residual) / checkpoint.residual_scale),
            1.0e-12,
        )
        self.assertEqual(checkpoint.row_labels[0], "particle budget")
        self.assertEqual(checkpoint.column_labels[-1], "electrons")

    def test_exact_linear_solver_and_controlled_fallback_are_distinct(
        self,
    ) -> None:
        direct, fallback = atmosphere_linear_solve_checkpoint(self.inputs)
        self.assertEqual(direct.branch, "solve")
        self.assertEqual(direct.matrix_rank, 23)
        self.assertEqual(direct.step.shape, (23,))
        self.assertLess(direct.linear_residual_norm, 1.0e-12)

        self.assertEqual(fallback.branch, "lstsq")
        self.assertEqual(fallback.matrix_rank, 22)
        self.assertEqual(fallback.step.shape, (23,))
        self.assertLess(fallback.linear_residual_norm, 1.0e-10)

    def test_exact_newton_update_witness_triggers_every_branch(self) -> None:
        witness = atmosphere_newton_update_witness()
        self.assertEqual(
            witness.branches,
            (
                "accept",
                "sign-damped accept",
                "absolute reflection",
                "one-percent fallback",
                "shared-scale fallback",
            ),
        )
        np.testing.assert_array_equal(
            witness.sign_damped,
            [False, True, False, True, False],
        )
        np.testing.assert_array_equal(
            witness.effective_delta,
            [1.0, -1.38, 15.0, 10.0, 20.0],
        )
        np.testing.assert_allclose(
            witness.candidate,
            [9.0, 11.38, -5.0, 0.0, 0.0],
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            witness.returned_density,
            [9.0, 11.38, 5.0, 0.1, 2.0],
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_array_equal(
            witness.scale_before,
            [100.0, 100.0, 100.0, 100.0, 10.0],
        )
        np.testing.assert_array_equal(
            witness.scale_after,
            [100.0, 100.0, 100.0, 10.0, 10.0],
        )
        self.assertTrue(witness.still_iterating)
        np.testing.assert_array_equal(
            witness.convergence_probe_relative_update.view(np.uint64),
            [
                np.float64(1.0e-4).view(np.uint64),
                np.float64(1.0e-4).view(np.uint64) + np.uint64(1),
            ],
        )
        np.testing.assert_array_equal(
            witness.convergence_probe_still_iterating,
            [False, True],
        )
        np.testing.assert_array_equal(
            witness.convergence_probe_returned_density,
            1.0 - witness.convergence_probe_relative_update,
        )

    def test_pressure_continuation_is_exact_and_restarts_are_independent(
        self,
    ) -> None:
        checkpoint = atmosphere_continuation_checkpoint(self.inputs)
        np.testing.assert_array_equal(
            checkpoint.pressure_ratio,
            [1.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        )
        np.testing.assert_array_equal(
            checkpoint.continuation_seed_equal,
            True,
        )
        np.testing.assert_array_equal(
            checkpoint.continuation_iteration_count,
            [9, 5, 5, 5, 7, 7],
        )
        np.testing.assert_array_equal(
            checkpoint.restart_iteration_count,
            [9, 9, 7, 7, 6, 8],
        )
        self.assertTrue(np.all(checkpoint.continuation_converged))
        self.assertTrue(np.all(checkpoint.restart_converged))
        self.assertLess(
            np.max(checkpoint.continuation_residual_max_scaled),
            1.0e-10,
        )
        self.assertLess(
            np.max(checkpoint.restart_residual_max_scaled),
            1.0e-10,
        )
        full = compute_atmosphere_molecular_state(self.inputs)
        np.testing.assert_array_equal(
            checkpoint.continuation_solution,
            full.molecular_state.previous_molecular_equation_densities,
        )
        self.assertFalse(
            np.array_equal(
                checkpoint.continuation_seed[1:],
                checkpoint.restart_seed[1:],
            )
        )


if __name__ == "__main__":
    unittest.main()
