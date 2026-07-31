"""Exact-name and numerical gates for the first reconstructed functions."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from payne_zero_atmosphere.run_setup import (
    standard_rosseland_optical_depth_grid,
)
from payne_zero_synthesis.constants import (
    BOLTZMANN_ERG_PER_K,
    LIGHT_SPEED_NM_PER_S,
    PLANCK_ERG_SECOND,
)
from payne_zero_synthesis.radiative_transfer import PLANCK_PREFACTOR, planck_bnu


class Chapter01ExactNameTests(unittest.TestCase):
    def test_standard_depth_grid_matches_pinned_expression(self) -> None:
        standard_rosseland_optical_depth = (
            standard_rosseland_optical_depth_grid(80)
        )
        expected = 10.0 ** (-6.875 + 0.125 * np.arange(80, dtype=np.float64))
        np.testing.assert_array_equal(standard_rosseland_optical_depth, expected)

    def test_planck_bnu_shape_and_temperature_response(self) -> None:
        wavelength_nm = torch.tensor([500.0, 600.0], dtype=torch.float64)
        temperature = torch.tensor([4500.0, 5772.0, 7500.0], dtype=torch.float64)
        planck = planck_bnu(wavelength_nm, temperature)
        self.assertEqual(tuple(planck.shape), (3, 2))
        self.assertTrue(bool(torch.all(torch.diff(planck[:, 0]) > 0.0)))

    def test_planck_bnu_matches_direct_equation(self) -> None:
        wavelength_nm = torch.tensor([505.0], dtype=torch.float64)
        temperature = torch.tensor([5772.0], dtype=torch.float64)
        frequency_hz = LIGHT_SPEED_NM_PER_S / wavelength_nm
        photon_energy_over_thermal_energy = (
            PLANCK_ERG_SECOND
            * frequency_hz
            / (BOLTZMANN_ERG_PER_K * temperature)
        )
        boltzmann_factor = torch.exp(-photon_energy_over_thermal_energy)
        expected = (
            PLANCK_PREFACTOR
            * (frequency_hz / 1e15) ** 3
            * boltzmann_factor
            / (1.0 - boltzmann_factor)
        )
        torch.testing.assert_close(
            planck_bnu(wavelength_nm, temperature).reshape(-1),
            expected,
            rtol=0.0,
            atol=0.0,
        )

        fully_exact = (
            2.0
            * PLANCK_ERG_SECOND
            * frequency_hz**3
            / (2.99792458e10**2)
            / torch.expm1(photon_energy_over_thermal_energy)
        )
        relative_difference = torch.abs(expected / fully_exact - 1.0)
        self.assertLess(float(relative_difference), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
