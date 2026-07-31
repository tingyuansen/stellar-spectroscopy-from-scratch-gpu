"""Analytic checks for the small Chapter 3 teaching constructions."""

from __future__ import annotations

import unittest

import numpy as np

from book.chapter03_support import (
    REFERENCE_HC_OVER_K_CM_K,
    damped_hydrogen_electron_fixed_point,
    fractional_doppler_width,
    neutral_collision_density_proxy,
    two_level_lte_populations,
    two_stage_saha_fractions,
)
from book.chapter03_runtime import backend_parity_profile
from payne_zero_synthesis.constants import (
    REFERENCE_ATOMIC_MASS_GRAM,
    REFERENCE_BOLTZMANN_ERG_PER_K,
    REFERENCE_BOLTZMANN_EV_PER_K,
    REFERENCE_PLANCK_ERG_SECOND,
    REFERENCE_SAHA_COEFFICIENT,
)


class Chapter03TeachingTests(unittest.TestCase):
    def test_backend_profile_separates_resolved_error_from_zero_leakage(self) -> None:
        reference = {
            "field": np.array([0.0, 1.0e-7, 1.0e6]),
            "provenance_sha256": np.array("metadata-not-a-number"),
        }
        candidate = {"field": np.array([1.0e-6, 1.0e-7, 1.0e6 + 1.0])}
        profile = backend_parity_profile(reference, candidate)
        self.assertEqual(profile["field_count"], 1)
        self.assertAlmostEqual(profile["resolved_relative"], 1.0e-6)
        self.assertAlmostEqual(profile["zero_leakage"], 1.0e-12)

    def test_backend_profile_rejects_missing_or_nonfinite_fields(self) -> None:
        reference = {
            "first": np.array([1.0]),
            "second": np.array([2.0]),
            "provenance": np.array("not a scientific field"),
        }
        with self.assertRaisesRegex(ValueError, "missing=\\['second'\\]"):
            backend_parity_profile(
                reference,
                {"first": np.array([1.0])},
            )
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            backend_parity_profile(
                reference,
                {
                    "first": np.array([np.nan]),
                    "second": np.array([2.0]),
                },
            )

    def test_teaching_helpers_pin_the_rounded_eos_constant_tier(self) -> None:
        self.assertEqual(REFERENCE_BOLTZMANN_ERG_PER_K, 1.38054e-16)
        self.assertEqual(REFERENCE_BOLTZMANN_EV_PER_K, 8.6171e-5)
        self.assertEqual(REFERENCE_PLANCK_ERG_SECOND, 6.6256e-27)
        self.assertEqual(REFERENCE_SAHA_COEFFICIENT, 2.4148e15)
        expected_hc_over_k = 6.6256e-27 * 2.99792458e10 / 1.38054e-16
        self.assertEqual(REFERENCE_HC_OVER_K_CM_K, expected_hc_over_k)

    def test_two_level_state_normalizes_and_recovers_boltzmann_ratio(self) -> None:
        temperature = np.array([3000.0, 6000.0, 12000.0])
        energy = np.array([0.0, 10000.0])
        weight = np.array([2.0, 4.0])
        partition, fraction = two_level_lte_populations(
            temperature, energy, weight
        )
        expected_ratio = (
            weight[1] / weight[0]
            * np.exp(-REFERENCE_HC_OVER_K_CM_K * energy[1] / temperature)
        )
        np.testing.assert_allclose(fraction.sum(axis=1), 1.0)
        np.testing.assert_allclose(fraction[:, 1] / fraction[:, 0], expected_ratio)
        self.assertTrue(np.all(np.diff(partition) > 0.0))
        self.assertTrue(np.all(np.diff(fraction[:, 1]) > 0.0))

    def test_two_stage_saha_limits_follow_temperature_and_electron_density(self) -> None:
        temperature = np.array([5000.0, 10000.0, 20000.0])
        electron_density = np.full(3, 1.0e13)
        lower, upper, _ = two_stage_saha_fractions(
            temperature,
            electron_density,
            np.full(3, 2.0),
            np.ones(3),
            13.5985,
        )
        np.testing.assert_allclose(lower + upper, 1.0)
        self.assertTrue(np.all(np.diff(upper) > 0.0))
        _, crowded_upper, _ = two_stage_saha_fractions(
            temperature,
            100.0 * electron_density,
            np.full(3, 2.0),
            np.ones(3),
            13.5985,
        )
        self.assertTrue(np.all(crowded_upper < upper))

        scalar_temperature = 10000.0
        scalar_electron_density = 1.0e13
        scalar_lower_partition = 2.0
        scalar_upper_partition = 1.0
        _, _, scalar_log_ratio = two_stage_saha_fractions(
            np.array([scalar_temperature]),
            np.array([scalar_electron_density]),
            np.array([scalar_lower_partition]),
            np.array([scalar_upper_partition]),
            13.5985,
        )
        expected_ratio = (
            2.0
            * 2.4148e15
            * scalar_temperature**1.5
            / scalar_electron_density
            * scalar_upper_partition
            / scalar_lower_partition
            * np.exp(-13.5985 / (8.6171e-5 * scalar_temperature))
        )
        np.testing.assert_allclose(
            np.exp(scalar_log_ratio[0]), expected_ratio, rtol=2.0e-15
        )

    def test_saha_keeps_a_nonzero_minority_stage_in_log_space(self) -> None:
        lower, upper, log_ratio = two_stage_saha_fractions(
            np.array([20000.0]),
            np.array([1.0e-8]),
            np.array([2.0]),
            np.array([1.0]),
            13.5985,
        )
        self.assertGreater(log_ratio[0], 40.0)
        self.assertGreater(lower[0], 0.0)
        self.assertEqual(upper[0], 1.0)

    def test_hydrogen_fixed_point_converges_under_declared_rule(self) -> None:
        result = damped_hydrogen_electron_fixed_point(
            temperature_k=9000.0,
            gas_pressure_dyn_cm2=2.0e4,
            electron_density_seed_cm3=1.0e12,
            ionization_energy_ev=13.5985,
        )
        history = np.asarray(result["history"])
        self.assertTrue(result["converged"])
        self.assertLess(history[-1, 3], 1.0e-4)
        self.assertGreater(result["electron_density_cm3"], 0.0)
        self.assertLess(
            result["electron_density_cm3"],
            result["total_particle_density_cm3"],
        )
        old = history[:, 0]
        raw = history[:, 1]
        updated = history[:, 2]
        expected_updated = 0.5 * (np.maximum(raw, 0.5 * old) + old)
        expected_residual = np.abs(
            (old - expected_updated) / np.maximum(expected_updated, 1.0e-300)
        )
        np.testing.assert_array_equal(updated, expected_updated)
        np.testing.assert_array_equal(history[:, 3], expected_residual)
        self.assertTrue(np.all(updated >= 0.75 * old))

    def test_hydrogen_fixed_point_rejects_an_impossible_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive nuclei density"):
            damped_hydrogen_electron_fixed_point(
                temperature_k=9000.0,
                gas_pressure_dyn_cm2=2.0e4,
                electron_density_seed_cm3=1.0e30,
                ionization_energy_ev=13.5985,
            )

    def test_collision_proxy_and_doppler_limits(self) -> None:
        temperature = np.array([5000.0, 10000.0])
        hydrogen = np.array([1.0e14, 2.0e14])
        helium = np.array([1.0e13, 2.0e13])
        molecular = np.zeros(2)
        proxy = neutral_collision_density_proxy(
            temperature, hydrogen, helium, molecular
        )
        expected = (hydrogen + 0.42 * helium) * (temperature / 1.0e4) ** 0.3
        np.testing.assert_array_equal(proxy, expected)

        zero_micro = np.zeros(2)
        hydrogen_width = fractional_doppler_width(
            temperature, zero_micro, REFERENCE_ATOMIC_MASS_GRAM
        )
        iron_width = fractional_doppler_width(
            temperature, zero_micro, 56.0 * REFERENCE_ATOMIC_MASS_GRAM
        )
        self.assertTrue(np.all(hydrogen_width > iron_width))
        self.assertGreater(hydrogen_width[1], hydrogen_width[0])

        dominant_microturbulence = np.full(2, 1.0e8)
        hydrogen_micro = fractional_doppler_width(
            temperature,
            dominant_microturbulence,
            REFERENCE_ATOMIC_MASS_GRAM,
        )
        iron_micro = fractional_doppler_width(
            temperature,
            dominant_microturbulence,
            56.0 * REFERENCE_ATOMIC_MASS_GRAM,
        )
        np.testing.assert_allclose(hydrogen_micro, iron_micro, rtol=2.0e-4)


if __name__ == "__main__":
    unittest.main()
