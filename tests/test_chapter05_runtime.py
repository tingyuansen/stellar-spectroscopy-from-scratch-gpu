"""Analytic and route checks for the progressive Chapter 5 runtime."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from book.chapter05_runtime import (
    SYNTHESIS_CONTINUUM_FIELDS,
    atmosphere_component_budget,
    atmosphere_grid_boundary_checkpoint,
    atmosphere_grid_checkpoint,
    continuum_table_preflight,
    edge_triplet_checkpoint,
    edge_reconstruction_checkpoint,
    edge_use_trace_checkpoint,
    hminus_table_preflight,
    hminus_edge_checkpoint,
    hminus_component_checkpoint,
    line_reference_checkpoint,
    load_regime_state,
    mass_opacity_from_cross_section,
    metal_population_ownership_checkpoint,
    molecular_continuum_checkpoint,
    numba_timing_checkpoint,
    opacity_scaling_checkpoint,
    project_synthesis_continuum_state,
    run_atmosphere_continuum,
    run_full_atmosphere_continuum,
    run_sampled_synthesis_continuum,
    run_sampled_synthesis_continuum_at_frequency,
    run_synthesis_continuum,
    scattering_checkpoint,
    standard_synthesis_component_checkpoint,
    state_projection_checkpoint,
    stimulated_emission_checkpoint,
    stimulated_emission_factor,
    synthesis_stored_h2_invariance_checkpoint,
)


class Chapter05RuntimeTests(unittest.TestCase):
    """Require local physics before any oracle comparison is possible."""

    def test_mass_opacity_units_and_scaling_are_executable(self) -> None:
        value = mass_opacity_from_cross_section(2.0e14, 3.0e-18, 4.0e-8)
        self.assertAlmostEqual(float(value), 1.5e4)
        checkpoint = opacity_scaling_checkpoint()
        np.testing.assert_array_equal(checkpoint.population_ratio, 2.0)
        np.testing.assert_array_equal(checkpoint.density_ratio, 0.5)

    def test_continuum_tables_are_manifest_bound_before_first_use(self) -> None:
        checkpoint = continuum_table_preflight()
        self.assertEqual(len(checkpoint.roles), 5)
        self.assertTrue(all(checkpoint.manifest_verified))
        self.assertTrue(all(len(value) == 64 for value in checkpoint.sha256))
        self.assertEqual(checkpoint.hminus_boundfree_wavelength_shape, (85,))
        self.assertEqual(checkpoint.hminus_boundfree_cross_section_shape, (85,))
        self.assertEqual(checkpoint.hminus_freefree_wavelength_shape, (22,))
        self.assertEqual(checkpoint.hminus_freefree_temperature_shape, (11,))
        self.assertIn("1e-18 cm^2", checkpoint.hminus_stored_unit)

    def test_hminus_preflight_is_just_in_time_and_fail_closed(self) -> None:
        checkpoint = hminus_table_preflight()
        self.assertEqual(checkpoint.roles, ("atmosphere continuum",))
        self.assertEqual(checkpoint.manifest_verified, (True,))
        with patch("book.chapter05_runtime.hashlib.sha256") as digest:
            digest.return_value.hexdigest.return_value = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                hminus_table_preflight()

    def test_mass_opacity_rejects_nonphysical_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            mass_opacity_from_cross_section(0.0, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            mass_opacity_from_cross_section(1.0, np.nan, 1.0)

    def test_stimulated_emission_has_both_limits(self) -> None:
        checkpoint = stimulated_emission_checkpoint()
        self.assertTrue(np.all(checkpoint.factor > 0.0))
        self.assertTrue(np.all(checkpoint.factor <= 1.0))
        self.assertAlmostEqual(
            checkpoint.factor[0] / checkpoint.low_energy_approximation[0],
            1.0,
            places=6,
        )
        self.assertAlmostEqual(checkpoint.factor[-1], 1.0, places=4)
        value = stimulated_emission_factor(5.0e14, 5772.0)
        self.assertGreater(float(value), 0.0)
        self.assertLessEqual(float(value), 1.0)

    def test_all_regime_states_load_without_a_golden(self) -> None:
        for regime in (
            "hot_dwarf",
            "solar_dwarf",
            "low_gravity_giant",
            "cool_molecule_rich",
        ):
            with self.subTest(regime=regime):
                atmosphere = load_regime_state(regime, "atmosphere")
                synthesis = load_regime_state(regime, "synthesis")
                self.assertEqual(len(atmosphere), 18)
                self.assertEqual(len(synthesis), 27)
                self.assertEqual(atmosphere["temperature"].shape, (6,))
                self.assertEqual(synthesis["temperature"].shape, (6,))

    def test_atmosphere_grid_checkpoint_covers_five_exact_branches(self) -> None:
        checkpoint = atmosphere_grid_checkpoint()
        np.testing.assert_array_equal(checkpoint.sample_count, 30000)
        np.testing.assert_allclose(
            checkpoint.first_wavelength_nm,
            [
                144.577263387,
                91.1800865275,
                50.4312810266,
                22.7876741143,
                10.0023028502,
            ],
            rtol=0.0,
            atol=5.0e-10,
        )
        self.assertTrue(np.all(checkpoint.first_frequency_weight_hz > 0.0))
        self.assertTrue(np.all(checkpoint.interior_frequency_weight_hz > 0.0))
        self.assertTrue(np.all(checkpoint.last_frequency_weight_hz > 0.0))
        np.testing.assert_array_equal(
            checkpoint.active_reference_count,
            [226, 240, 263, 299, 338],
        )

    def test_atmosphere_grid_checks_both_sides_of_every_boundary(self) -> None:
        checkpoint = atmosphere_grid_boundary_checkpoint()
        np.testing.assert_array_equal(
            checkpoint.effective_temperature_k,
            [4499.0, 4500.0, 7249.0, 7250.0, 12999.0, 13000.0, 29999.0, 30000.0],
        )
        np.testing.assert_allclose(
            checkpoint.first_wavelength_nm,
            [
                144.577263387,
                91.1800865275,
                91.1800865275,
                50.4312810266,
                50.4312810266,
                22.7876741143,
                22.7876741143,
                10.0023028502,
            ],
            rtol=0.0,
            atol=5.0e-10,
        )
        np.testing.assert_array_equal(
            checkpoint.active_reference_count,
            [226, 240, 240, 263, 263, 299, 299, 338],
        )

    def test_edge_triplet_basis_and_stored_samples_are_exact(self) -> None:
        requested = np.asarray([400.0, 500.0, 650.0, 900.0])
        checkpoint = edge_triplet_checkpoint(requested)
        np.testing.assert_allclose(checkpoint.basis_sum, 1.0, atol=2.0e-15)
        self.assertTrue(checkpoint.packaged_samples_bitwise_equal)
        self.assertTrue(checkpoint.sign_flip_invariant)
        self.assertEqual(
            checkpoint.sample_frequency_hz.size,
            3 * checkpoint.used_interval_index.size,
        )

    def test_standard_synthesis_route_is_finite_and_nonnegative(self) -> None:
        checkpoint = run_synthesis_continuum(
            "solar_dwarf",
            [400.0, 500.0, 650.0, 900.0],
        )
        self.assertEqual(checkpoint.absorption_cm2_per_g.shape, (6, 4))
        self.assertEqual(checkpoint.scattering_cm2_per_g.shape, (6, 4))
        self.assertTrue(np.all(np.isfinite(checkpoint.absorption_cm2_per_g)))
        self.assertTrue(np.all(np.isfinite(checkpoint.scattering_cm2_per_g)))
        self.assertTrue(np.all(checkpoint.absorption_cm2_per_g >= 0.0))
        self.assertTrue(np.all(checkpoint.scattering_cm2_per_g >= 0.0))
        np.testing.assert_array_equal(
            checkpoint.continuum_opacity,
            checkpoint.absorption_cm2_per_g
            + checkpoint.scattering_cm2_per_g,
        )
        self.assertEqual(checkpoint.dtype, "torch.float64")

    def test_standard_synthesis_consumes_only_the_exact_18_field_view(
        self,
    ) -> None:
        handoff = load_regime_state("solar_dwarf", "synthesis")
        continuum_state = project_synthesis_continuum_state(handoff)
        self.assertEqual(tuple(continuum_state), SYNTHESIS_CONTINUUM_FIELDS)
        self.assertNotIn("hydrogen_ionized_population", continuum_state)
        self.assertNotIn("molecular_hydrogen_population", continuum_state)

        changed_handoff = dict(handoff)
        changed_handoff["hydrogen_ionized_population"] = (
            10.0 * handoff["hydrogen_ionized_population"]
        )
        wavelengths = [400.0, 500.0, 650.0, 900.0]
        with patch(
            "book.chapter05_runtime.load_regime_state",
            return_value=handoff,
        ):
            baseline = run_synthesis_continuum("solar_dwarf", wavelengths)
        with patch(
            "book.chapter05_runtime.load_regime_state",
            return_value=changed_handoff,
        ):
            changed = run_synthesis_continuum("solar_dwarf", wavelengths)
        np.testing.assert_array_equal(
            baseline.absorption_cm2_per_g,
            changed.absorption_cm2_per_g,
        )
        np.testing.assert_array_equal(
            baseline.scattering_cm2_per_g,
            changed.scattering_cm2_per_g,
        )

    def test_edge_reconstruction_uses_the_same_trimmed_state(self) -> None:
        handoff = load_regime_state("solar_dwarf", "synthesis")
        changed_handoff = dict(handoff)
        changed_handoff["hydrogen_ionized_population"] = (
            10.0 * handoff["hydrogen_ionized_population"]
        )
        with patch(
            "book.chapter05_runtime.load_regime_state",
            return_value=handoff,
        ):
            baseline = edge_reconstruction_checkpoint(sample_count=24)
        with patch(
            "book.chapter05_runtime.load_regime_state",
            return_value=changed_handoff,
        ):
            changed = edge_reconstruction_checkpoint(sample_count=24)
        np.testing.assert_array_equal(
            baseline.absorption_samples_cm2_per_g,
            changed.absorption_samples_cm2_per_g,
        )
        np.testing.assert_array_equal(
            baseline.exact_absorption_cm2_per_g,
            changed.exact_absorption_cm2_per_g,
        )

    def test_atmosphere_route_is_finite_without_a_repair_clamp(self) -> None:
        checkpoint = run_atmosphere_continuum(
            "cool_molecule_rich",
            [3.0e13, 1.0e14, 5.0e14],
        )
        self.assertEqual(checkpoint.absorption_cm2_per_g.shape, (6, 3))
        self.assertEqual(checkpoint.scattering_cm2_per_g.shape, (6, 3))
        self.assertEqual(checkpoint.continuum_source.shape, (6, 3))
        self.assertTrue(np.all(np.isfinite(checkpoint.absorption_cm2_per_g)))
        self.assertTrue(np.all(np.isfinite(checkpoint.scattering_cm2_per_g)))
        self.assertTrue(np.all(np.isfinite(checkpoint.continuum_source)))
        self.assertTrue(np.all(checkpoint.absorption_cm2_per_g >= 0.0))
        self.assertTrue(np.all(checkpoint.scattering_cm2_per_g >= 0.0))
        np.testing.assert_array_equal(
            checkpoint.continuum_opacity,
            checkpoint.absorption_cm2_per_g
            + checkpoint.scattering_cm2_per_g,
        )

    def test_full_atmosphere_route_uses_direct_30000_point_grid(self) -> None:
        checkpoint = run_full_atmosphere_continuum("solar_dwarf", 5772.0)
        self.assertEqual(checkpoint.wavelength_nm.shape, (30000,))
        self.assertEqual(checkpoint.absorption_cm2_per_g.shape, (6, 30000))
        self.assertEqual(checkpoint.scattering_cm2_per_g.shape, (6, 30000))
        self.assertEqual(checkpoint.continuum_source.shape, (6, 30000))
        self.assertEqual(
            checkpoint.route,
            "atmosphere direct 30,000-point product",
        )
        self.assertEqual(checkpoint.dtype, "float64")
        self.assertTrue(np.all(np.isfinite(checkpoint.absorption_cm2_per_g)))
        self.assertTrue(np.all(np.isfinite(checkpoint.scattering_cm2_per_g)))
        self.assertTrue(np.all(np.isfinite(checkpoint.continuum_source)))

    def test_hminus_edge_separates_boundfree_and_freefree(self) -> None:
        checkpoint = hminus_edge_checkpoint()
        np.testing.assert_array_equal(
            checkpoint.stored_boundfree_cross_section_1e_minus_18_cm2[:2],
            0.0,
        )
        self.assertGreater(
            checkpoint.stored_boundfree_cross_section_1e_minus_18_cm2[2],
            0.0,
        )
        np.testing.assert_array_equal(
            checkpoint.boundfree_absorption_cm2_per_g[:, :2],
            0.0,
        )
        self.assertTrue(
            np.all(checkpoint.boundfree_absorption_cm2_per_g[:, 2] > 0.0)
        )
        self.assertTrue(
            np.all(checkpoint.freefree_absorption_cm2_per_g > 0.0)
        )
        np.testing.assert_array_equal(
            checkpoint.total_absorption_cm2_per_g,
            checkpoint.boundfree_absorption_cm2_per_g
            + checkpoint.freefree_absorption_cm2_per_g,
        )
        self.assertGreater(
            2.99792458e17 / checkpoint.last_table_wavelength_nm,
            checkpoint.threshold_hz,
        )

        curve = hminus_component_checkpoint(
            "solar_dwarf",
            np.linspace(800.0, 2_200.0, 101),
        )
        self.assertEqual(curve.total_absorption_cm2_per_g.shape, (6, 101))
        np.testing.assert_array_equal(
            curve.total_absorption_cm2_per_g,
            curve.boundfree_absorption_cm2_per_g
            + curve.freefree_absorption_cm2_per_g,
        )
        longward = curve.wavelength_nm >= (
            2.99792458e17 / curve.threshold_hz
        )
        np.testing.assert_array_equal(
            curve.boundfree_absorption_cm2_per_g[:, longward],
            0.0,
        )
        np.testing.assert_array_equal(
            checkpoint.physical_boundfree_cross_section_cm2,
            1.0e-18
            * checkpoint.stored_boundfree_cross_section_1e_minus_18_cm2,
        )
        np.testing.assert_allclose(
            checkpoint.transparent_unit_boundfree_absorption_cm2_per_g,
            checkpoint.exact_unit_boundfree_absorption_cm2_per_g,
            rtol=6.0e-11,
            atol=1.0e-30,
        )
        np.testing.assert_allclose(
            checkpoint.transparent_fixture_boundfree_absorption_cm2_per_g,
            checkpoint.boundfree_absorption_cm2_per_g,
            rtol=6.0e-11,
            atol=1.0e-30,
        )
        np.testing.assert_array_equal(
            checkpoint.fixture_hminus_population_cm3,
            checkpoint.unit_hminus_population_cm3
            * checkpoint.fixture_hydrogen_departure_coefficient,
        )

    def test_molecular_continuum_keeps_three_distinct_processes(self) -> None:
        checkpoint = molecular_continuum_checkpoint()
        np.testing.assert_array_equal(
            checkpoint.total_absorption_cm2_per_g,
            checkpoint.ch_absorption_cm2_per_g
            + checkpoint.oh_absorption_cm2_per_g
            + checkpoint.collision_induced_absorption_cm2_per_g,
        )
        self.assertTrue(
            np.any(checkpoint.collision_induced_absorption_cm2_per_g[:, -2] > 0.0)
        )
        np.testing.assert_array_equal(
            checkpoint.collision_induced_absorption_cm2_per_g[:, -1],
            0.0,
        )
        np.testing.assert_allclose(
            checkpoint.collision_induced_absorption_cm2_per_g,
            checkpoint.h2h2_absorption_cm2_per_g
            + checkpoint.h2he_absorption_cm2_per_g,
            rtol=4.0e-15,
            atol=1.0e-30,
        )
        self.assertTrue(np.all(checkpoint.h2h2_absorption_cm2_per_g >= 0.0))
        self.assertTrue(np.all(checkpoint.h2he_absorption_cm2_per_g >= 0.0))
        cool = checkpoint.temperature_gate_k < 9000.0
        warm = ~cool
        self.assertTrue(np.all(checkpoint.ch_at_temperature_gate_cm2_per_g[cool] > 0.0))
        self.assertTrue(np.all(checkpoint.oh_at_temperature_gate_cm2_per_g[cool] > 0.0))
        np.testing.assert_array_equal(
            checkpoint.ch_at_temperature_gate_cm2_per_g[warm],
            0.0,
        )
        np.testing.assert_array_equal(
            checkpoint.oh_at_temperature_gate_cm2_per_g[warm],
            0.0,
        )
        np.testing.assert_array_equal(
            checkpoint.h2_equilibrium_temperature_k,
            [100.0, 101.0, 19899.0, 19900.0],
        )
        self.assertTrue(
            np.all(np.isfinite(checkpoint.h2_equilibrium_population_cm3))
        )
        self.assertTrue(np.all(checkpoint.h2_equilibrium_population_cm3 > 0.0))
        self.assertGreater(
            checkpoint.h2_equilibrium_population_cm3[0],
            checkpoint.h2_equilibrium_population_cm3[1],
        )
        self.assertGreater(
            checkpoint.h2_equilibrium_population_cm3[2],
            checkpoint.h2_equilibrium_population_cm3[3],
        )
        cutoff_active = checkpoint.h2_cutoff_temperature_k <= 20000.0
        self.assertTrue(
            np.all(
                checkpoint.h2_rayleigh_at_cutoff_cm2_per_g[cutoff_active] > 0.0
            )
        )
        np.testing.assert_array_equal(
            checkpoint.h2_rayleigh_at_cutoff_cm2_per_g[~cutoff_active],
            0.0,
        )
        np.testing.assert_array_equal(
            checkpoint.cia_lower_column_weight,
            [0.0, 0.5, 0.0],
        )
        np.testing.assert_array_equal(
            checkpoint.cia_upper_column_weight,
            [1.0, 0.5, 1.0],
        )
        with np.load(
            "data/static/atmosphere_tables/continuum_opacity_tables.npz",
            allow_pickle=False,
        ) as tables:
            for actual, table_name in (
                (
                    checkpoint.cia_h2h2_log10_coefficient,
                    "hydrogen_molecule_h2_collision_table",
                ),
                (
                    checkpoint.cia_h2he_log10_coefficient,
                    "hydrogen_molecule_he_collision_table",
                ),
            ):
                table = np.asarray(tables[table_name])
                expected = np.asarray(
                    [
                        table[40, 3],
                        0.5 * (table[40, 2] + table[40, 3]),
                        table[40, 4],
                    ]
                )
                np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            checkpoint.all_warm_absorption_cm2_per_g,
            0.0,
        )
        self.assertTrue(np.any(checkpoint.mixed_column_absorption_cm2_per_g > 0.0))
        self.assertTrue(
            np.any(checkpoint.molecule_disabled_absorption_cm2_per_g > 0.0)
        )
        np.testing.assert_array_equal(
            checkpoint.ifop13_alone_scattering_cm2_per_g,
            0.0,
        )
        self.assertTrue(
            np.any(checkpoint.ifop4_and13_h2_increment_cm2_per_g > 0.0)
        )

    def test_scattering_components_preserve_redirection_budget(self) -> None:
        checkpoint = scattering_checkpoint()
        named_sum = (
            checkpoint.electron_scattering_cm2_per_g
            + checkpoint.hydrogen_rayleigh_cm2_per_g
            + checkpoint.helium_rayleigh_cm2_per_g
            + checkpoint.molecular_hydrogen_rayleigh_cm2_per_g
        )
        np.testing.assert_allclose(
            checkpoint.total_scattering_cm2_per_g,
            named_sum,
            rtol=3.0e-15,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            np.ptp(checkpoint.electron_scattering_cm2_per_g, axis=1),
            0.0,
        )
        self.assertTrue(
            np.all(
                checkpoint.hydrogen_rayleigh_cm2_per_g[:, 0]
                > checkpoint.hydrogen_rayleigh_cm2_per_g[:, -1]
            )
        )
        np.testing.assert_array_equal(
            checkpoint.cap_component_value_cm2_per_g,
            checkpoint.above_cap_component_value_cm2_per_g,
        )

    def test_line_reference_is_a_subroute_not_a_raw_continuum(self) -> None:
        checkpoint = line_reference_checkpoint("solar_dwarf", 5772.0)
        self.assertEqual(checkpoint.active_count, 240)
        self.assertEqual(checkpoint.inactive_count, 103)
        self.assertEqual(checkpoint.threshold_cm2_per_g.shape, (6, 344))
        self.assertEqual(checkpoint.dtype, "float32")
        self.assertTrue(checkpoint.duplicated_last_column)
        self.assertEqual(checkpoint.packed_sentinel, 2**30)
        self.assertTrue(checkpoint.inactive_placeholder_matches)
        self.assertEqual(checkpoint.inactive_placeholder_max_abs_residual, 0.0)

    def test_named_atmosphere_budget_reconstructs_the_exact_product(self) -> None:
        checkpoint = atmosphere_component_budget(
            "solar_dwarf",
            [3.0e14, 5.0e14, 1.0e15, 2.0e15, 4.0e15, 8.0e15],
        )
        self.assertEqual(len(checkpoint.component_names), 14)
        self.assertEqual(
            checkpoint.component_absorption_cm2_per_g.shape,
            (14, 6, 6),
        )
        np.testing.assert_allclose(
            checkpoint.reconstructed_absorption_cm2_per_g,
            checkpoint.exact_absorption_cm2_per_g,
            rtol=3.0e-15,
            atol=1.0e-25,
        )
        self.assertTrue(
            np.all(
                np.abs(checkpoint.absorption_residual_cm2_per_g)
                <= (
                    4.0e-15 * np.abs(checkpoint.exact_absorption_cm2_per_g)
                    + 1.0e-25
                )
            )
        )
        np.testing.assert_allclose(
            checkpoint.reconstructed_source,
            checkpoint.exact_source,
            rtol=3.0e-15,
            atol=1.0e-30,
        )
        self.assertEqual(
            checkpoint.component_source_numerator.shape,
            (14, 6, 6),
        )
        np.testing.assert_allclose(
            checkpoint.ordered_absorption_partial_sum_cm2_per_g[-1],
            checkpoint.reconstructed_absorption_cm2_per_g,
            rtol=4.0e-15,
            atol=1.0e-30,
        )
        np.testing.assert_allclose(
            checkpoint.reconstructed_source_numerator,
            checkpoint.exact_source_numerator,
            rtol=4.0e-15,
            atol=1.0e-30,
        )
        self.assertTrue(
            np.all(
                np.abs(checkpoint.source_numerator_residual)
                <= (
                    5.0e-15 * np.abs(checkpoint.exact_source_numerator)
                    + 1.0e-30
                )
            )
        )
        for required in (
            "hydrogen",
            "hminus",
            "h2plus",
            "molecular",
            "hot_metals",
        ):
            self.assertIn(required, checkpoint.component_names)

    def test_hot_metal_population_views_have_independent_linear_limits(self) -> None:
        checkpoint = metal_population_ownership_checkpoint()
        active_normalized = np.isfinite(checkpoint.normalized_ratio)
        self.assertTrue(np.any(active_normalized))
        np.testing.assert_array_equal(
            checkpoint.normalized_ratio[active_normalized],
            2.0,
        )
        np.testing.assert_array_equal(checkpoint.actual_ratio, 2.0)
        self.assertTrue(
            np.all(
                checkpoint.normalized_only_absorption_cm2_per_g[:, :2]
                == 0.0
            )
        )
        self.assertTrue(
            np.all(checkpoint.actual_only_absorption_cm2_per_g > 0.0)
        )
        self.assertTrue(
            checkpoint.normalized_perturbation_preserves_charge_square
        )
        self.assertTrue(
            checkpoint.actual_perturbation_preserves_hot_metal_populations
        )
        np.testing.assert_array_equal(
            checkpoint.doubled_normalized_hot_metal_populations[:, 0],
            2.0 * checkpoint.baseline_hot_metal_populations[:, 0],
        )
        np.testing.assert_array_equal(
            checkpoint.doubled_normalized_hot_metal_populations[:, 1:],
            checkpoint.baseline_hot_metal_populations[:, 1:],
        )
        charge_delta = (
            checkpoint.doubled_actual_charge_square_population_sum
            - checkpoint.baseline_charge_square_population_sum
        )
        np.testing.assert_allclose(
            charge_delta[:, 0],
            checkpoint.expected_actual_charge_square_delta,
            rtol=1.0e-15,
            atol=0.0,
        )
        np.testing.assert_array_equal(charge_delta[:, 1:], 0.0)

    def test_synthesis_handoff_projects_to_a_distinct_consumer_view(self) -> None:
        checkpoint = state_projection_checkpoint()
        self.assertEqual(len(checkpoint.synthesis_handoff_field_names), 27)
        self.assertEqual(len(checkpoint.synthesis_continuum_field_names), 18)
        self.assertEqual(len(checkpoint.atmosphere_continuum_field_names), 18)
        self.assertNotEqual(
            set(checkpoint.synthesis_continuum_field_names),
            set(checkpoint.atmosphere_continuum_field_names),
        )
        self.assertNotIn(
            "molecular_hydrogen_population",
            checkpoint.synthesis_continuum_field_names,
        )
        self.assertNotIn(
            "hydrogen_ionized_population",
            checkpoint.synthesis_continuum_field_names,
        )
        self.assertIn(
            "hydrogen_ionized_population",
            checkpoint.atmosphere_continuum_field_names,
        )
        self.assertEqual(
            checkpoint.synthesis_continuum_shapes[
                checkpoint.synthesis_continuum_field_names.index(
                    "ion_stage_populations"
                )
            ],
            (6, 6, 139),
        )
        self.assertEqual(
            checkpoint.atmosphere_continuum_shapes[
                checkpoint.atmosphere_continuum_field_names.index(
                    "ion_stage_populations_by_packed_slot"
                )
            ],
            (6, 1006),
        )

    def test_stored_schema_h2_is_not_a_standard_synthesis_input(self) -> None:
        checkpoint = synthesis_stored_h2_invariance_checkpoint()
        self.assertEqual(checkpoint.stored_h2_scale_factor, 1.0e6)
        self.assertTrue(checkpoint.bitwise_equal)
        np.testing.assert_array_equal(
            checkpoint.baseline_absorption_cm2_per_g,
            checkpoint.changed_absorption_cm2_per_g,
        )
        np.testing.assert_array_equal(
            checkpoint.baseline_scattering_cm2_per_g,
            checkpoint.changed_scattering_cm2_per_g,
        )

    def test_numba_timing_separates_compile_warm_cache_and_threads(self) -> None:
        checkpoint = numba_timing_checkpoint(maximum_threads=2)
        self.assertTrue(checkpoint.all_outputs_bitwise_equal)
        self.assertGreater(checkpoint.cache_file_count, 0)
        self.assertEqual(len(checkpoint.output_fingerprint), 64)
        for capture in (checkpoint.first_process, checkpoint.cached_process):
            self.assertEqual(capture["shape"], [80, 30000])
            self.assertEqual(capture["dtype"], "float64")
            self.assertEqual(capture["used_threads"], 2)
            for name in (
                "python_seconds",
                "serial_first_seconds",
                "serial_warm_seconds",
                "parallel_first_seconds",
                "parallel_one_thread_seconds",
                "parallel_many_thread_seconds",
            ):
                self.assertGreater(capture[name], 0.0)

    def test_sampled_diagnostic_and_extension_remain_labelled_lanes(self) -> None:
        wavelength = [100.0, 125.0, 200.0, 400.0, 1_000.0, 2_500.0]
        diagnostic = run_sampled_synthesis_continuum(
            "solar_dwarf",
            wavelength,
            precomputed=False,
        )
        extension = run_sampled_synthesis_continuum(
            "solar_dwarf",
            wavelength,
            precomputed=True,
        )
        self.assertEqual(diagnostic.route, "sampled diagnostic")
        self.assertEqual(extension.route, "sampled precomputed extension")
        self.assertTrue(diagnostic.coulomb_table_energy_first)
        self.assertTrue(extension.coulomb_table_energy_first)
        self.assertEqual(diagnostic.frequency_invariant_shapes, {})
        self.assertEqual(diagnostic.tensor_cache_entries_after_reuse, 0)
        self.assertGreater(len(extension.frequency_invariant_shapes), 20)
        self.assertGreater(extension.tensor_cache_entries_after_reuse, 0)
        for checkpoint in (diagnostic, extension):
            self.assertEqual(checkpoint.absorption_cm2_per_g.shape, (6, 6))
            self.assertEqual(checkpoint.scattering_cm2_per_g.shape, (6, 6))
            self.assertEqual(checkpoint.source_bnu.shape, (6, 6))
            self.assertEqual(checkpoint.dtype, "torch.float64")

            self.assertTrue(np.all(np.isfinite(checkpoint.absorption_cm2_per_g)))
            self.assertTrue(np.all(np.isfinite(checkpoint.scattering_cm2_per_g)))
            self.assertTrue(np.all(checkpoint.absorption_cm2_per_g >= 0.0))
            self.assertTrue(np.all(checkpoint.scattering_cm2_per_g >= 0.0))
            self.assertTrue(checkpoint.source_matches_direct)
            np.testing.assert_allclose(
                checkpoint.source_bnu,
                checkpoint.direct_source_bnu,
                rtol=3.0e-15,
                atol=0.0,
            )
        self.assertIsNone(diagnostic.supported_wavelength_bounds_nm)
        self.assertIsNone(diagnostic.supported_wavelength_nm)
        np.testing.assert_array_equal(
            extension.supported_wavelength_nm,
            [
                100.0,
                125.0,
                160.0,
                200.0,
                250.0,
                320.0,
                400.0,
                500.0,
                700.0,
                1_000.0,
                1_600.0,
                2_500.0,
            ],
        )
        self.assertEqual(
            extension.supported_wavelength_bounds_nm,
            (100.0, 2500.0),
        )
        np.testing.assert_array_equal(diagnostic.source_bnu, extension.source_bnu)
        self.assertFalse(
            np.array_equal(
                diagnostic.absorption_cm2_per_g,
                extension.absorption_cm2_per_g,
            )
        )
        with self.assertRaisesRegex(ValueError, "validated only"):
            run_sampled_synthesis_continuum(
                "solar_dwarf",
                [99.0, 400.0],
                precomputed=True,
            )
        with self.assertRaisesRegex(ValueError, "twelve-point"):
            run_sampled_synthesis_continuum(
                "solar_dwarf",
                [150.0, 400.0],
                precomputed=True,
            )

    def test_frequency_native_diagnostic_preserves_exact_threshold_probes(
        self,
    ) -> None:
        threshold = np.float64(4.8359395487147644e14)
        frequency = np.asarray(
            [
                np.nextafter(threshold, -np.inf),
                threshold,
                np.nextafter(threshold, np.inf),
            ],
            dtype=np.float64,
        )
        checkpoint = run_sampled_synthesis_continuum_at_frequency(
            "solar_dwarf",
            frequency,
        )
        np.testing.assert_array_equal(checkpoint.frequency_hz, frequency)
        self.assertEqual(checkpoint.route, "sampled diagnostic")
        self.assertFalse(checkpoint.frequency_invariant_shapes)

    def test_standard_synthesis_calls_only_used_edge_triplets(self) -> None:
        checkpoint = edge_use_trace_checkpoint()
        np.testing.assert_array_equal(
            checkpoint.used_interval_index,
            [100, 220],
        )
        self.assertEqual(checkpoint.exact_internal_edge_assigned_interval, 100)
        self.assertEqual(checkpoint.called_frequency_count, 6)
        self.assertEqual(checkpoint.total_interval_count, 340)
        self.assertEqual(checkpoint.unused_interval_count, 338)
        np.testing.assert_array_equal(
            checkpoint.called_frequency_hz,
            checkpoint.expected_called_frequency_hz,
        )
        self.assertFalse(checkpoint.coulomb_table_energy_first)
        self.assertTrue(checkpoint.frequency_invariants_was_none)

    def test_standard_synthesis_retains_named_components_and_pops(self) -> None:
        checkpoint = standard_synthesis_component_checkpoint()
        self.assertEqual(
            checkpoint.absorption_components_cm2_per_g.shape,
            (7, 6, 12),
        )
        self.assertEqual(
            checkpoint.scattering_components_cm2_per_g.shape,
            (3, 6, 12),
        )
        self.assertEqual(
            checkpoint.build_pops_hot_metal_populations.shape,
            (6, 21),
        )
        self.assertEqual(
            checkpoint.build_pops_charge_square_population_sum.shape,
            (6, 5),
        )
        synthesis_handoff = load_regime_state("solar_dwarf", "synthesis")
        np.testing.assert_array_equal(
            checkpoint.build_pops_hydrogen_ionized_population,
            synthesis_handoff[
                "hydrogen_partition_normalized_ion_stage_populations"
            ][:, 1],
        )
        np.testing.assert_array_equal(
            checkpoint.absorption_partial_sums_cm2_per_g[-1],
            checkpoint.exact_sampled_absorption_cm2_per_g,
        )
        np.testing.assert_array_equal(
            checkpoint.scattering_partial_sums_cm2_per_g[-1],
            checkpoint.exact_sampled_scattering_cm2_per_g,
        )
        np.testing.assert_array_equal(
            checkpoint.minor_absorption_subcomponents_cm2_per_g.sum(axis=0),
            checkpoint.absorption_components_cm2_per_g[2],
        )
        np.testing.assert_array_equal(
            checkpoint.minor_scattering_subcomponents_cm2_per_g.sum(axis=0),
            checkpoint.scattering_components_cm2_per_g[2],
        )
        product = run_synthesis_continuum(
            "solar_dwarf",
            checkpoint.requested_wavelength_nm,
        )
        np.testing.assert_array_equal(
            checkpoint.final_absorption_cm2_per_g,
            product.absorption_cm2_per_g,
        )
        np.testing.assert_array_equal(
            checkpoint.final_scattering_cm2_per_g,
            product.scattering_cm2_per_g,
        )
        self.assertFalse(checkpoint.coulomb_table_energy_first)
        self.assertTrue(checkpoint.frequency_invariants_was_none)

    def test_one_standard_edge_is_reconstructed_from_three_samples(self) -> None:
        checkpoint = edge_reconstruction_checkpoint()
        self.assertLess(
            checkpoint.left_wavelength_nm,
            checkpoint.midpoint_wavelength_nm,
        )
        self.assertLess(
            checkpoint.midpoint_wavelength_nm,
            checkpoint.right_wavelength_nm,
        )
        self.assertEqual(checkpoint.absorption_samples_cm2_per_g.shape, (6, 3))
        self.assertEqual(
            checkpoint.reconstructed_absorption_cm2_per_g.shape,
            (6, 240),
        )
        np.testing.assert_allclose(
            checkpoint.basis_sum,
            1.0,
            rtol=0.0,
            atol=5.0e-15,
        )
        np.testing.assert_allclose(
            checkpoint.reconstructed_absorption_cm2_per_g,
            checkpoint.exact_absorption_cm2_per_g,
            rtol=3.0e-15,
            atol=1.0e-25,
        )
        np.testing.assert_allclose(
            checkpoint.reconstructed_scattering_cm2_per_g,
            checkpoint.exact_scattering_cm2_per_g,
            rtol=3.0e-15,
            atol=1.0e-25,
        )


if __name__ == "__main__":
    unittest.main()
