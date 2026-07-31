"""Analytic and numerical gates for the transparent Chapter 4 network."""

from __future__ import annotations

import unittest

import numpy as np

from book.chapter04_teaching import (
    decode_base100_molecule_code,
    equal_abundance_closed_form,
    finite_difference_jacobian,
    fixed_nuclei_molecule_density,
    mass_action_population,
    solve_two_element_molecule,
    two_element_molecule_jacobian,
    two_element_molecule_residual,
)


class Chapter04TeachingTests(unittest.TestCase):
    """Keep the reader-built molecule simple, physical, and exact."""

    def test_source_native_base100_examples_decode_without_new_notation(self) -> None:
        self.assertEqual(decode_base100_molecule_code(608.0), ((6, 8), 0))
        self.assertEqual(decode_base100_molecule_code(101.0), ((1, 1), 0))
        self.assertEqual(decode_base100_molecule_code(100.0), ((1, 100), 0))
        self.assertEqual(decode_base100_molecule_code(608.01), ((6, 8), 1))

    def test_mass_action_handles_ordinary_and_inverse_electrons(self) -> None:
        basis = {1: 2.0e15, 6: 3.0e15, 8: 4.0e15}
        electron_density = 1.0e13
        co = mass_action_population(
            2.0e-20,
            (6, 8),
            basis,
            electron_density=electron_density,
        )
        negative_hydrogen = mass_action_population(
            5.0e-30,
            (1, 100),
            basis,
            electron_density=electron_density,
        )
        positive_co = mass_action_population(
            2.0e-7,
            (6, 8),
            basis,
            electron_density=electron_density,
            positive_charge=1,
        )
        self.assertEqual(co, 2.0e-20 * basis[6] * basis[8])
        self.assertEqual(
            negative_hydrogen,
            5.0e-30 * basis[1] * electron_density,
        )
        self.assertEqual(
            positive_co,
            2.0e-7 * basis[6] * basis[8] / electron_density,
        )

    def test_equal_abundance_closed_form_closes_all_three_ledgers(self) -> None:
        particle_density = 1.0e17
        for formation_constant in (0.0, 1.0e-19, 1.0e-17, 1.0e-15):
            with self.subTest(formation_constant=formation_constant):
                solution = equal_abundance_closed_form(
                    total_particle_density=particle_density,
                    formation_constant=formation_constant,
                )
                self.assertLess(
                    np.max(np.abs(solution.residual)) / particle_density,
                    3.0e-15,
                )
                self.assertGreater(solution.free_a_density, 0.0)
                self.assertGreater(solution.free_b_density, 0.0)
                self.assertGreaterEqual(solution.molecule_density, 0.0)
                self.assertAlmostEqual(
                    solution.free_a_density + solution.molecule_density,
                    0.5 * solution.total_nuclei_density,
                    delta=64.0,
                )

    def test_fixed_nuclei_quadratic_stays_inside_the_limiting_budget(self) -> None:
        nuclei_density = 1.0e17
        abundances = np.array([0.7, 0.3])
        populations = [
            fixed_nuclei_molecule_density(
                total_nuclei_density=nuclei_density,
                elemental_abundances=abundances,
                formation_constant=constant,
            )
            for constant in (0.0, 1.0e-20, 1.0e-17, 1.0e-10)
        ]
        self.assertTrue(np.all(np.diff(populations) > 0.0))
        self.assertGreaterEqual(populations[0], 0.0)
        self.assertLessEqual(populations[-1], 0.3 * nuclei_density)
        molecule_density = populations[2]
        residual = molecule_density - 1.0e-17 * (
            0.7 * nuclei_density - molecule_density
        ) * (0.3 * nuclei_density - molecule_density)
        self.assertLess(abs(residual) / nuclei_density, 2.0e-16)

    def test_analytic_jacobian_matches_independent_central_difference(self) -> None:
        densities = np.array([1.2e17, 4.0e16, 3.0e16])
        particle_density = 8.0e16
        abundances = np.array([0.6, 0.4])
        formation_constant = 2.0e-17

        def residual(point):
            return two_element_molecule_residual(
                point,
                total_particle_density=particle_density,
                elemental_abundances=abundances,
                formation_constant=formation_constant,
            )

        analytic = two_element_molecule_jacobian(
            densities,
            total_particle_density=particle_density,
            elemental_abundances=abundances,
            formation_constant=formation_constant,
        )
        numerical = finite_difference_jacobian(residual, densities)
        np.testing.assert_allclose(analytic, numerical, rtol=2.0e-10, atol=1.0e-10)

    def test_density_space_newton_matches_the_closed_form(self) -> None:
        particle_density = 1.0e17
        formation_constant = 1.0e-17
        numerical = solve_two_element_molecule(
            total_particle_density=particle_density,
            elemental_abundances=np.array([0.5, 0.5]),
            formation_constant=formation_constant,
        )
        analytic = equal_abundance_closed_form(
            total_particle_density=particle_density,
            formation_constant=formation_constant,
        )
        self.assertTrue(numerical.converged)
        self.assertLessEqual(numerical.iterations, 10)
        for field in (
            "total_nuclei_density",
            "free_a_density",
            "free_b_density",
            "molecule_density",
        ):
            with self.subTest(field=field):
                self.assertAlmostEqual(
                    getattr(numerical, field) / getattr(analytic, field),
                    1.0,
                    places=13,
                )

    def test_unequal_abundances_close_both_elemental_budgets(self) -> None:
        solution = solve_two_element_molecule(
            total_particle_density=2.0e16,
            elemental_abundances=np.array([0.7, 0.3]),
            formation_constant=4.0e-17,
        )
        self.assertTrue(solution.converged)
        self.assertLess(
            np.max(np.abs(solution.residual)) / 2.0e16,
            2.0e-14,
        )
        np.testing.assert_allclose(
            [
                solution.free_a_density + solution.molecule_density,
                solution.free_b_density + solution.molecule_density,
            ],
            np.array([0.7, 0.3]) * solution.total_nuclei_density,
            rtol=2.0e-14,
            atol=0.0,
        )

    def test_molecule_fraction_grows_with_constant_and_pressure_scale(self) -> None:
        constants = (1.0e-19, 1.0e-18, 1.0e-17)
        fractions_at_fixed_density = []
        for constant in constants:
            solution = equal_abundance_closed_form(
                total_particle_density=1.0e17,
                formation_constant=constant,
            )
            fractions_at_fixed_density.append(
                2.0
                * solution.molecule_density
                / solution.total_nuclei_density
            )
        self.assertTrue(np.all(np.diff(fractions_at_fixed_density) > 0.0))

        fractions_at_fixed_constant = []
        for particle_density in (1.0e14, 1.0e16, 1.0e18):
            solution = equal_abundance_closed_form(
                total_particle_density=particle_density,
                formation_constant=1.0e-17,
            )
            fractions_at_fixed_constant.append(
                2.0
                * solution.molecule_density
                / solution.total_nuclei_density
            )
        self.assertTrue(np.all(np.diff(fractions_at_fixed_constant) > 0.0))

    def test_invalid_inputs_fail_before_newton(self) -> None:
        invalid = (
            {"total_particle_density": 0.0},
            {"formation_constant": -1.0},
            {"elemental_abundances": np.array([0.4, 0.4])},
            {"elemental_abundances": np.array([0.5, np.nan])},
        )
        baseline = {
            "total_particle_density": 1.0e17,
            "elemental_abundances": np.array([0.5, 0.5]),
            "formation_constant": 1.0e-17,
        }
        for change in invalid:
            arguments = {**baseline, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                solve_two_element_molecule(**arguments)


if __name__ == "__main__":
    unittest.main()
