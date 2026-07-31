"""Physics and boundary tests for the progressive Chapter 6 runtime."""

from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from book.chapter05_runtime import load_regime_state
import book.chapter06_runtime as chapter06


class Chapter06RuntimeTests(unittest.TestCase):
    """Keep the readable one-line build causal, dimensional, and exact."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.state = load_regime_state("solar_dwarf", "synthesis")
        cls.fe_population = np.asarray(
            cls.state["partition_normalized_populations"][:, 0, 25],
            dtype=np.float64,
        )
        cls.fe_width = np.asarray(
            cls.state["fractional_doppler_widths"][:, 0, 25],
            dtype=np.float64,
        )

    def test_teaching_subset_is_one_raw_record_and_not_a_golden(self) -> None:
        raw = chapter06.load_teaching_line_raw()
        self.assertEqual(tuple(raw), chapter06.RAW_LINE_FIELDS)
        self.assertTrue(all(values.shape == (1,) for values in raw.values()))
        self.assertEqual(raw["energy_shift_field"][0], b"          ")
        self.assertEqual(raw["line_category_tag"][0], b"")
        self.assertNotIn("golden", str(chapter06.TEACHING_LINE_SUBSET))
        self.assertNotIn(
            "golden",
            inspect.getsource(chapter06.load_teaching_line_raw).lower(),
        )

    def test_transition_follows_energy_separation_before_line_strength(self) -> None:
        result = chapter06.transition_checkpoint()
        self.assertEqual(result.lower_excitation_cm, 33507.123)
        self.assertEqual(result.upper_excitation_cm, 53545.833)
        self.assertEqual(result.energy_separation_cm, 20038.71)
        self.assertEqual(result.wavelength_nm, 499.03411946178176)
        self.assertEqual(result.stored_wavelength_nm, 499.0341)
        self.assertEqual(
            result.stored_minus_derived_nm,
            -1.946178173284352e-05,
        )
        self.assertEqual(result.oscillator_strength, 0.1380384264602885)
        self.assertEqual(
            (result.atomic_number, result.ion_stage, result.line_type),
            (26, 1, 0),
        )

    def test_transformed_teaching_record_is_defensively_copied(self) -> None:
        first = chapter06.teaching_line_record()
        first["oscillator_strength"][0] = 999.0
        second = chapter06.teaching_line_record()
        self.assertEqual(second["oscillator_strength"][0], 0.1380384264602885)
        self.assertEqual(
            chapter06.transition_checkpoint().oscillator_strength,
            0.1380384264602885,
        )

    def test_fastex_lanes_keep_their_distinct_domain_contracts(self) -> None:
        result = chapter06.fast_exponential_checkpoint()
        import payne_zero_atmosphere.line_profile_math as atmosphere_profile_math
        import payne_zero_synthesis.line_opacity as synthesis_line_opacity

        source_root = (chapter06.REPOSITORY_ROOT / "src").resolve()
        self.assertTrue(
            Path(atmosphere_profile_math.__file__).resolve().is_relative_to(source_root)
        )
        self.assertTrue(
            Path(synthesis_line_opacity.__file__).resolve().is_relative_to(source_root)
        )
        negative, zero = 0, 1
        boundary, positive_infinity, not_a_number = 7, 8, 9
        self.assertEqual(result.atmosphere_lookup[negative], 0.0)
        self.assertEqual(
            result.synthesis_float64_lookup[negative],
            np.exp(1.0),
        )
        np.testing.assert_array_equal(
            result.atmosphere_lookup,
            result.atmosphere_compiled_lookup,
        )
        self.assertEqual(result.atmosphere_lookup[zero], 1.0)
        self.assertEqual(result.synthesis_float64_lookup[zero], 1.0)
        self.assertEqual(result.synthesis_float32_lookup[zero], 1.0)
        self.assertEqual(result.atmosphere_lookup[boundary], 0.0)
        self.assertEqual(result.synthesis_float64_lookup[boundary], 0.0)
        self.assertEqual(result.synthesis_float32_lookup[boundary], 0.0)
        self.assertEqual(result.atmosphere_lookup[positive_infinity], 0.0)
        self.assertEqual(result.synthesis_float64_lookup[positive_infinity], 0.0)
        self.assertEqual(result.synthesis_float32_lookup[positive_infinity], 0.0)
        self.assertEqual(result.atmosphere_lookup[not_a_number], 0.0)
        self.assertTrue(np.isnan(result.synthesis_float64_lookup[not_a_number]))
        self.assertTrue(np.isnan(result.synthesis_float32_lookup[not_a_number]))
        self.assertEqual(result.atmosphere_lookup[2], np.exp(-0.001))
        self.assertEqual(result.synthesis_float64_lookup[2], np.exp(-0.001))
        self.assertEqual(
            result.atmosphere_lookup[3],
            np.exp(-np.float64(239) * np.float64(0.001)),
        )
        self.assertEqual(
            result.synthesis_float64_lookup[3],
            np.exp(-np.float64(239) * np.float64(0.001)),
        )

    def test_grid_first_call_uses_the_textbooks_staged_source(self) -> None:
        code = "\n".join(
            (
                "from pathlib import Path",
                "import book.chapter06_runtime as chapter06",
                "chapter06.build_synthesis_wavelength_grid()",
                "import payne_zero_synthesis.atomic_lines as atomic_lines",
                "print(Path(atomic_lines.__file__).resolve())",
            )
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=chapter06.REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        imported_path = Path(completed.stdout.strip())
        self.assertTrue(
            imported_path.is_relative_to((chapter06.REPOSITORY_ROOT / "src").resolve()),
            imported_path,
        )

    def test_line_strength_has_expected_linear_and_inverse_scalings(self) -> None:
        baseline = chapter06.gross_line_strength_checkpoint(
            partition_normalized_population_cm3=self.fe_population,
            mass_density_g_cm3=self.state["mass_density"],
            fractional_doppler_width=self.fe_width,
            hc_over_kt_cm=self.state["hc_over_kt"],
        )
        doubled_gf = chapter06.gross_line_strength_checkpoint(
            partition_normalized_population_cm3=self.fe_population,
            mass_density_g_cm3=self.state["mass_density"],
            fractional_doppler_width=self.fe_width,
            hc_over_kt_cm=self.state["hc_over_kt"],
            oscillator_strength=2.0 * baseline.oscillator_strength,
        )
        doubled_population = chapter06.gross_line_strength_checkpoint(
            partition_normalized_population_cm3=2.0 * self.fe_population,
            mass_density_g_cm3=self.state["mass_density"],
            fractional_doppler_width=self.fe_width,
            hc_over_kt_cm=self.state["hc_over_kt"],
        )
        doubled_density = chapter06.gross_line_strength_checkpoint(
            partition_normalized_population_cm3=self.fe_population,
            mass_density_g_cm3=2.0 * self.state["mass_density"],
            fractional_doppler_width=self.fe_width,
            hc_over_kt_cm=self.state["hc_over_kt"],
        )
        np.testing.assert_allclose(
            doubled_gf.integrated_strength_cm2_hz_per_g,
            2.0 * baseline.integrated_strength_cm2_hz_per_g,
            rtol=2e-15,
            atol=0.0,
        )
        np.testing.assert_allclose(
            doubled_population.line_amplitude_cm2_per_g,
            2.0 * baseline.line_amplitude_cm2_per_g,
            rtol=2e-15,
            atol=0.0,
        )
        np.testing.assert_allclose(
            doubled_density.line_amplitude_cm2_per_g,
            0.5 * baseline.line_amplitude_cm2_per_g,
            rtol=2e-15,
            atol=0.0,
        )
        expected_excitation_factor = (
            self.fe_population
            * baseline.lower_level_boltzmann_factor
            * baseline.oscillator_strength
        )
        np.testing.assert_allclose(
            baseline.gf_weighted_excitation_factor_cm3,
            expected_excitation_factor,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            baseline.excitation_weighted_partition_normalized_population_cm3,
            self.fe_population * baseline.lower_level_boltzmann_factor,
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(hasattr(baseline, "lower_level_population_cm3"))

    def test_damping_ledger_separates_electron_and_neutral_perturbers(self) -> None:
        collision = chapter06.collision_density_proxy(self.state)
        baseline = chapter06.damping_checkpoint(
            electron_density_cm3=self.state["electron_density"],
            collision_density_proxy_cm3=collision,
            fractional_doppler_width=self.fe_width,
        )
        more_electrons = chapter06.damping_checkpoint(
            electron_density_cm3=2.0 * self.state["electron_density"],
            collision_density_proxy_cm3=collision,
            fractional_doppler_width=self.fe_width,
        )
        more_neutrals = chapter06.damping_checkpoint(
            electron_density_cm3=self.state["electron_density"],
            collision_density_proxy_cm3=2.0 * collision,
            fractional_doppler_width=self.fe_width,
        )
        np.testing.assert_array_equal(
            more_electrons.radiative_term,
            baseline.radiative_term,
        )
        np.testing.assert_array_equal(
            more_electrons.van_der_waals_term,
            baseline.van_der_waals_term,
        )
        np.testing.assert_allclose(
            more_electrons.stark_term,
            2.0 * baseline.stark_term,
        )
        np.testing.assert_array_equal(
            more_neutrals.radiative_term,
            baseline.radiative_term,
        )
        np.testing.assert_array_equal(
            more_neutrals.stark_term,
            baseline.stark_term,
        )
        np.testing.assert_allclose(
            more_neutrals.van_der_waals_term,
            2.0 * baseline.van_der_waals_term,
        )

    def test_continuous_voigt_reference_preserves_area_on_a_wide_domain(
        self,
    ) -> None:
        result = chapter06.profile_normalization_checkpoint(
            damping_ratio=np.asarray([0.0, 0.05, 0.2]),
            integration_limit_doppler_widths=100.0,
        )
        self.assertLess(float(np.max(result.relative_missing_area)), 1.3e-3)
        self.assertGreaterEqual(float(np.min(result.relative_missing_area)), -1e-12)
        np.testing.assert_allclose(
            result.measured_integral_phi_dnu,
            result.measured_integral_h_du / np.sqrt(np.pi),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            result.exact_integral_phi_dnu,
            np.ones(3),
        )
        with self.assertRaisesRegex(ValueError, "greater than 12"):
            chapter06.profile_normalization_checkpoint(
                damping_ratio=np.asarray([0.1]),
                integration_limit_doppler_widths=12.0,
            )

    def test_dense_line_is_finite_nonnegative_and_applies_stimulation_once(
        self,
    ) -> None:
        depth_index = 2
        strength = chapter06.gross_line_strength_checkpoint(
            partition_normalized_population_cm3=self.fe_population,
            mass_density_g_cm3=self.state["mass_density"],
            fractional_doppler_width=self.fe_width,
            hc_over_kt_cm=self.state["hc_over_kt"],
        )
        damping = chapter06.damping_checkpoint(
            electron_density_cm3=self.state["electron_density"],
            collision_density_proxy_cm3=chapter06.collision_density_proxy(self.state),
            fractional_doppler_width=self.fe_width,
        )
        center = chapter06.transition_checkpoint().wavelength_nm
        wavelength = np.linspace(center - 0.08, center + 0.08, 1201)
        result = chapter06.dense_line_checkpoint(
            wavelength_nm=wavelength,
            line_amplitude_cm2_per_g=float(
                strength.line_amplitude_cm2_per_g[depth_index]
            ),
            fractional_doppler_width=float(self.fe_width[depth_index]),
            damping_ratio=float(damping.damping_ratio[depth_index]),
            temperature_k=float(self.state["temperature"][depth_index]),
            depth_index=depth_index,
        )
        self.assertEqual(result.wavelength_nm.shape, (1201,))
        self.assertEqual(
            result.profile_contract,
            "full synthesis Harris H(a,u) reference",
        )
        self.assertTrue(
            np.all(np.isfinite(result.gross_line_mass_absorption_coefficient))
        )
        self.assertTrue(np.all(result.gross_line_mass_absorption_coefficient >= 0.0))
        np.testing.assert_allclose(
            result.net_line_mass_absorption_coefficient,
            result.gross_line_mass_absorption_coefficient
            * result.stimulated_emission_factor,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(
            np.all(
                result.net_line_mass_absorption_coefficient
                <= result.gross_line_mass_absorption_coefficient
            )
        )

    def test_staged_atmosphere_lane_returns_one_gross_float32_line(self) -> None:
        result = chapter06.run_atmosphere_one_line()
        pre = result.pre_stimulated_line_mass_absorption_coefficient
        post = result.post_stimulated_line_mass_absorption_coefficient
        self.assertEqual(pre.shape, (80, 30_000))
        self.assertEqual(pre.dtype, np.float32)
        self.assertEqual(post.dtype, np.float64)
        self.assertEqual(result.selected_line_count, 1)
        np.testing.assert_array_equal(
            result.nonzero_count_per_depth,
            np.full(80, 3, dtype=np.int64),
        )
        self.assertEqual(result.peak_pre_stimulated_cm2_per_g, 0.3731258809566498)
        self.assertTrue(np.all(np.isfinite(pre)))
        self.assertTrue(np.all(pre >= 0.0))
        self.assertTrue(np.all(post <= pre.astype(np.float64)))
        self.assertEqual(
            result.stimulation_owner,
            "downstream atmosphere transfer accumulator",
        )

    def test_stored_doppler_width_conversions_and_area_scaling(self) -> None:
        from payne_zero_synthesis.pipeline import (
            compute_doppler_per_ion,
            load_atomic_masses,
        )

        result = chapter06.stored_doppler_checkpoint(
            temperature_k=self.state["temperature"],
            fractional_doppler_width=self.fe_width,
        )
        center = chapter06.transition_checkpoint().wavelength_nm
        np.testing.assert_allclose(
            result.doppler_width_nm,
            center * self.fe_width,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            result.doppler_width_km_per_s,
            2.99792458e5 * self.fe_width,
            rtol=0.0,
            atol=0.0,
        )
        self.assertIn("Chapter 3", result.width_source)

        narrow = chapter06.gross_line_strength_checkpoint(
            partition_normalized_population_cm3=self.fe_population,
            mass_density_g_cm3=self.state["mass_density"],
            fractional_doppler_width=self.fe_width,
            hc_over_kt_cm=self.state["hc_over_kt"],
        )
        broad = chapter06.gross_line_strength_checkpoint(
            partition_normalized_population_cm3=self.fe_population,
            mass_density_g_cm3=self.state["mass_density"],
            fractional_doppler_width=2.0 * self.fe_width,
            hc_over_kt_cm=self.state["hc_over_kt"],
        )
        np.testing.assert_allclose(
            broad.line_amplitude_cm2_per_g,
            0.5 * narrow.line_amplitude_cm2_per_g,
            rtol=2e-15,
            atol=0.0,
        )
        frequency_hz = chapter06.LIGHT_SPEED_NM_PER_S / center
        narrow_area = (
            narrow.line_amplitude_cm2_per_g
            * frequency_hz
            * self.fe_width
            * chapter06.SQRT_PI_REFERENCE
        )
        broad_area = (
            broad.line_amplitude_cm2_per_g
            * frequency_hz
            * (2.0 * self.fe_width)
            * chapter06.SQRT_PI_REFERENCE
        )
        np.testing.assert_allclose(
            narrow_area,
            narrow.integrated_strength_cm2_hz_per_g,
            rtol=2e-15,
            atol=0.0,
        )
        np.testing.assert_allclose(broad_area, narrow_area, rtol=2e-15, atol=0.0)
        masses = load_atomic_masses(
            chapter06.REPOSITORY_ROOT / "data/static/synthesis_tables/atomic_masses.npz"
        )
        baseline_width = compute_doppler_per_ion(
            self.state["temperature"],
            self.state["microturbulence"],
            masses,
        )[:, 0, 25]
        larger_microturbulence_width = compute_doppler_per_ion(
            self.state["temperature"],
            2.0 * self.state["microturbulence"],
            masses,
        )[:, 0, 25]
        self.assertTrue(np.all(larger_microturbulence_width > baseline_width))
        baseline_damping = chapter06.damping_checkpoint(
            electron_density_cm3=self.state["electron_density"],
            collision_density_proxy_cm3=chapter06.collision_density_proxy(self.state),
            fractional_doppler_width=baseline_width,
        )
        broader_damping = chapter06.damping_checkpoint(
            electron_density_cm3=self.state["electron_density"],
            collision_density_proxy_cm3=chapter06.collision_density_proxy(self.state),
            fractional_doppler_width=larger_microturbulence_width,
        )
        self.assertTrue(
            np.all(broader_damping.damping_ratio < baseline_damping.damping_ratio)
        )

    def test_harris_branch_seams_and_two_authorities_are_explicit(self) -> None:
        below_02 = np.nextafter(0.2, 0.0)
        above_02 = np.nextafter(0.2, np.inf)
        below_14 = np.nextafter(1.4, 0.0)
        above_14 = np.nextafter(1.4, np.inf)
        below_10 = np.nextafter(10.0, 0.0)
        above_10 = np.nextafter(10.0, np.inf)
        offset = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                np.nextafter(2.7, 0.0),
                np.nextafter(2.7, np.inf),
                below_10,
                above_10,
                0.15,
                0.15,
            ]
        )
        damping = np.asarray(
            [
                below_02,
                above_02,
                below_14,
                above_14,
                0.5,
                0.5,
                0.1,
                0.1,
                0.1,
                np.nextafter(100.0, np.inf),
            ]
        )
        result = chapter06.harris_branch_checkpoint(offset, damping)
        np.testing.assert_array_equal(
            result.atmosphere_scalar_profile,
            result.atmosphere_compiled_profile,
        )
        np.testing.assert_array_equal(
            result.atmosphere_scalar_profile,
            result.atmosphere_compiled_float64_profile,
        )
        np.testing.assert_allclose(
            result.synthesis_full_profile,
            result.synthesis_scalar_reference,
            rtol=3e-15,
            atol=0.0,
        )
        self.assertEqual(
            result.synthesis_wing_branch[6],
            "low-damping H0+aH1 wing",
        )
        self.assertEqual(
            result.synthesis_wing_branch[7],
            "low-damping u^-2 wing",
        )
        self.assertNotEqual(
            result.atmosphere_scalar_profile[8],
            result.synthesis_full_profile[8],
        )
        self.assertNotEqual(
            result.synthesis_ordinary_wing_profile[8],
            result.synthesis_full_profile[8],
        )

    def test_production_center_policy_reconstructs_all_twenty_four_depths(
        self,
    ) -> None:
        expected_masks = {
            "hot_dwarf": [True, True, True, False, False, False],
            "solar_dwarf": [True, True, True, True, True, True],
            "low_gravity_giant": [True, True, True, True, True, True],
            "cool_molecule_rich": [True, True, True, True, True, True],
        }
        checkpoints = {}
        for regime, expected in expected_masks.items():
            checkpoint = chapter06.synthesis_center_policy_checkpoint(regime)
            checkpoints[regime] = checkpoint
            self.assertEqual(
                checkpoint.passes_post_fastex_cutoff.tolist(),
                expected,
            )
            np.testing.assert_array_equal(
                checkpoint.float32_center_deposit_cm2_per_g,
                checkpoint.production_center_deposit_cm2_per_g,
            )
            np.testing.assert_allclose(
                checkpoint.pre_excitation_strength_cm2_per_g
                * checkpoint.fast_ex_weight,
                checkpoint.post_fastex_line_amplitude_cm2_per_g,
                rtol=0.0,
                atol=0.0,
            )
            self.assertTrue(np.all(checkpoint.center_cutoff_cm2_per_g >= 0.0))
        np.testing.assert_allclose(
            checkpoints["solar_dwarf"].damping_ratio,
            np.asarray(
                [
                    0.00522449344,
                    0.00580802287,
                    0.0117496888,
                    0.0694896007,
                    0.633170875,
                    8.53045131,
                ]
            ),
            rtol=2e-8,
            atol=0.0,
        )
        self.assertEqual(
            checkpoints["hot_dwarf"].passes_pre_excitation_cutoff.tolist(),
            [True, True, True, True, False, False],
        )
        self.assertGreater(
            np.max(
                np.abs(
                    checkpoints["hot_dwarf"].fast_ex_weight
                    / checkpoints["hot_dwarf"].direct_exp_weight
                    - 1.0
                )
            ),
            1.0e-5,
        )
        self.assertEqual(
            checkpoints["solar_dwarf"].float32_classical_line_strength_cm2,
            3.4403587321228807e-18,
        )
        self.assertEqual(
            checkpoints["solar_dwarf"].host_classical_line_strength_cm2,
            3.4403587659200408e-18,
        )
        solar = checkpoints["solar_dwarf"]
        low_damping = solar.damping_ratio < 0.2
        np.testing.assert_array_equal(
            solar.selected_center_profile[low_damping],
            solar.shortcut_center_profile[low_damping],
        )
        self.assertTrue(
            np.any(
                solar.full_harris_center_profile[low_damping]
                != solar.shortcut_center_profile[low_damping]
            )
        )
        self.assertEqual(
            solar.selected_branch.tolist(),
            [
                "low-damping center shortcut",
                "low-damping center shortcut",
                "low-damping center shortcut",
                "low-damping center shortcut",
                "full Harris center",
                "full Harris center",
            ],
        )

    def test_exact_synthesis_grid_and_constructor_mapping_are_complete(
        self,
    ) -> None:
        grid = chapter06.build_synthesis_wavelength_grid()
        self.assertEqual(grid.shape, (6000,))
        self.assertEqual(grid.dtype, np.dtype("float64"))
        self.assertEqual(grid[0], 495.0009387906341)
        self.assertEqual(grid[-1], 504.9989209057178)

        mapping = chapter06.one_line_synthesis_mapping()
        expected_mapping_keys = {
            "line_type",
            "atomic_number",
            "ion_stage",
            "wavelength_nm",
            "index_wavelength_nm",
            "oscillator_strength",
            "lower_excitation_cm",
            "radiative_damping",
            "stark_damping",
            "van_der_waals_damping",
            "raw_radiative_damping_log",
            "raw_stark_damping_log",
            "raw_van_der_waals_damping_log",
            "helium_line_type",
            "helium_line_center_cutoff_ratio",
            "harris_profile_h0_table",
            "harris_profile_h1_table",
            "harris_profile_h2_table",
        }
        self.assertEqual(set(mapping), expected_mapping_keys)
        self.assertEqual(mapping["line_type"].tolist(), [0])
        self.assertEqual(mapping["atomic_number"].tolist(), [26])
        self.assertEqual(mapping["ion_stage"].tolist(), [1])
        self.assertEqual(mapping["helium_line_type"].shape, (0,))
        self.assertEqual(
            float(mapping["helium_line_center_cutoff_ratio"]),
            1.0e-3,
        )
        for name in (
            "harris_profile_h0_table",
            "harris_profile_h1_table",
            "harris_profile_h2_table",
        ):
            self.assertEqual(mapping[name].shape, (2001,))
            self.assertEqual(mapping[name].dtype, np.dtype("float64"))

    def test_exact_synthesis_lane_recovers_activity_and_loop_parity(
        self,
    ) -> None:
        import torch

        expected_masks = {
            "hot_dwarf": [True, True, True, False, False, False],
            "solar_dwarf": [True, True, True, True, True, True],
            "low_gravity_giant": [True, True, True, True, True, True],
            "cool_molecule_rich": [True, True, True, True, True, True],
        }
        maximum_reach = 0
        for regime, expected in expected_masks.items():
            with self.subTest(regime=regime):
                batched = chapter06.run_synthesis_one_line(regime, wing_mode="batched")
                loop = chapter06.run_synthesis_one_line(regime, wing_mode="loop")
                self.assertEqual(
                    batched.gross_line_mass_absorption_coefficient.shape,
                    (6, 6000),
                )
                self.assertEqual(
                    batched.gross_line_mass_absorption_coefficient.dtype,
                    np.dtype("float32"),
                )
                self.assertEqual(
                    batched.net_line_mass_absorption_coefficient.dtype,
                    np.dtype("float32"),
                )
                self.assertEqual(batched.work_dtype, "torch.float64")
                self.assertEqual(batched.accumulation_dtype, "torch.float32")
                self.assertEqual(batched.cutoff_dtype, "torch.float32")
                self.assertEqual(batched.stimulation_dtype, "torch.float32")
                self.assertEqual(batched.device, "cpu")
                self.assertEqual(
                    batched.gross_line_mass_absorption_tensor.dtype,
                    torch.float32,
                )
                self.assertEqual(
                    batched.net_line_mass_absorption_tensor.device.type,
                    "cpu",
                )
                self.assertEqual(
                    (
                        batched.metal_line_count,
                        batched.auto_line_count,
                        batched.helium_line_count,
                    ),
                    (1, 0, 0),
                )
                self.assertEqual(
                    (
                        batched.population_ion_stage_index,
                        batched.population_element_index,
                    ),
                    (0, 25),
                )
                self.assertEqual(batched.metal_center_index, 2434)
                self.assertEqual(batched.metal_wing_index, 2434)
                self.assertEqual(batched.activity_mask.tolist(), expected)
                expected_nonzero = np.where(
                    batched.activity_mask,
                    2 * batched.wing_reach + 1,
                    0,
                )
                np.testing.assert_array_equal(
                    batched.nonzero_count,
                    expected_nonzero,
                )
                maximum_reach = max(
                    maximum_reach,
                    int(np.max(batched.wing_reach)),
                )
                np.testing.assert_array_equal(
                    batched.gross_line_mass_absorption_coefficient,
                    loop.gross_line_mass_absorption_coefficient,
                )
                np.testing.assert_array_equal(
                    batched.net_line_mass_absorption_coefficient,
                    loop.net_line_mass_absorption_coefficient,
                )
                self.assertTrue(
                    np.all(np.isfinite(batched.net_line_mass_absorption_coefficient))
                )
                self.assertTrue(
                    np.all(batched.net_line_mass_absorption_coefficient >= 0.0)
                )
                state = load_regime_state(regime, "synthesis")
                frequency = torch.as_tensor(
                    chapter06.LIGHT_SPEED_NM_PER_S / batched.wavelength_nm,
                    dtype=torch.float32,
                )
                photon_temperature_factor = (
                    chapter06.PLANCK_ERG_SECOND
                    / (
                        chapter06.BOLTZMANN_ERG_PER_K
                        * torch.as_tensor(
                            state["temperature"],
                            dtype=torch.float64,
                        )
                    )
                ).to(torch.float32)
                stimulated = 1.0 - torch.exp(
                    -frequency[None, :] * photon_temperature_factor[:, None]
                )
                torch.testing.assert_close(
                    batched.net_line_mass_absorption_tensor,
                    batched.gross_line_mass_absorption_tensor * stimulated,
                    rtol=0.0,
                    atol=0.0,
                )
        self.assertEqual(maximum_reach, 163)

    def test_synthesis_wing_reach_clamps_at_the_production_maximum(self) -> None:
        import torch

        from payne_zero_synthesis.line_opacity import (
            MAX_WING_PROFILE_STEPS,
            _wing_reach_batched,
            precompute_invariants,
        )

        grid = chapter06.build_synthesis_wavelength_grid()
        invariants = precompute_invariants(
            chapter06.one_line_synthesis_mapping(),
            grid,
            runtime_device=torch.device("cpu"),
        )
        reach, *_ = _wing_reach_batched(
            invariants,
            invariants.metal_wing_index,
            torch.full((1, 1), 1.0e20, dtype=torch.float64),
            torch.full((1, 1), 0.1, dtype=torch.float64),
            torch.full((1, 1), 0.004, dtype=torch.float64),
            invariants.metal_wavelength_nm,
            torch.full((1, 1), 1.0e-30, dtype=torch.float64),
            torch.ones((1, 1), dtype=torch.bool),
        )
        self.assertEqual(int(reach[0, 0]), MAX_WING_PROFILE_STEPS)


if __name__ == "__main__":
    unittest.main()
