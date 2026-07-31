"""Transparent molecular-equilibrium constructions for Chapter 4.

These helpers expose the mathematics before the fixed catalogs and production
solvers appear.  They do not read goldens or the external Payne Zero checkout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwoElementMoleculeSolution:
    """One positive solution of the transparent A + B <-> AB network."""

    total_nuclei_density: float
    free_a_density: float
    free_b_density: float
    molecule_density: float
    residual: np.ndarray
    iterations: int
    converged: bool


def decode_base100_molecule_code(code: float) -> tuple[tuple[int, ...], int]:
    """Decode the source-native molecule code into components and charge.

    Species 100 is an ordinary electron component.  The returned positive
    charge counts one inverse-electron sentinel (source code 101) per charge.
    """

    molecule_code = float(code)
    if not np.isfinite(molecule_code) or molecule_code <= 0.0:
        raise ValueError("molecule code must be positive and finite")
    place_values = np.array(
        [1.0e14, 1.0e12, 1.0e10, 1.0e8, 1.0e6, 1.0e4, 1.0e2, 1.0],
        dtype=np.float64,
    )
    active = np.flatnonzero(molecule_code >= place_values)
    if active.size == 0:
        raise ValueError("molecule code has no active base-100 field")

    remainder = molecule_code
    components: list[int] = []
    for place_value in place_values[int(active[0]) :]:
        species_code = int(remainder / place_value + 0.5)
        remainder -= float(species_code) * place_value
        components.append(100 if species_code == 0 else species_code)
    positive_charge = int(remainder * 100.0 + 0.5)
    return tuple(components), positive_charge


def mass_action_population(
    formation_constant: float,
    component_species_codes: tuple[int, ...],
    basis_density_by_species: dict[int, float],
    *,
    electron_density: float,
    positive_charge: int = 0,
) -> float:
    """Evaluate one ordinary-density mass-action product in exact notation."""

    population = float(formation_constant)
    if not np.isfinite(population) or population < 0.0:
        raise ValueError("formation_constant must be finite and nonnegative")
    if not np.isfinite(electron_density) or electron_density <= 0.0:
        raise ValueError("electron_density must be positive and finite")
    for species_code in component_species_codes:
        density = (
            electron_density
            if species_code == 100
            else float(basis_density_by_species[species_code])
        )
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("component densities must be positive and finite")
        population *= density
    population /= electron_density ** int(positive_charge)
    return population


def fixed_nuclei_molecule_density(
    *,
    total_nuclei_density: float,
    elemental_abundances: np.ndarray,
    formation_constant: float,
) -> float:
    """Return the physical quadratic root with the nuclei scale held fixed.

    For ``A + B <-> AB``, conservation gives
    ``n_AB = K (A_A*x0 - n_AB) (A_B*x0 - n_AB)``.
    """

    nuclei_density = float(total_nuclei_density)
    abundance_a, abundance_b = _validated_two_element_inputs(
        nuclei_density,
        elemental_abundances,
        formation_constant,
    )
    constant = float(formation_constant)
    if constant == 0.0:
        return 0.0
    budget_a = abundance_a * nuclei_density
    budget_b = abundance_b * nuclei_density
    linear_scale = 1.0 + constant * (budget_a + budget_b)
    discriminant = linear_scale**2 - 4.0 * constant**2 * budget_a * budget_b
    molecule_density = (
        2.0
        * constant
        * budget_a
        * budget_b
        / (linear_scale + np.sqrt(max(discriminant, 0.0)))
    )
    return min(float(molecule_density), budget_a, budget_b)


def two_element_molecule_residual(
    densities: np.ndarray,
    *,
    total_particle_density: float,
    elemental_abundances: np.ndarray,
    formation_constant: float,
) -> np.ndarray:
    """Return the A+B<->AB particle and two elemental residuals in cm^-3."""

    total_nuclei_density, free_a_density, free_b_density = np.asarray(
        densities,
        dtype=np.float64,
    )
    abundance_a, abundance_b = _validated_two_element_inputs(
        total_particle_density,
        elemental_abundances,
        formation_constant,
    )
    molecule_density = float(formation_constant) * free_a_density * free_b_density
    return np.array(
        [
            -float(total_particle_density)
            + free_a_density
            + free_b_density
            + molecule_density,
            free_a_density - abundance_a * total_nuclei_density + molecule_density,
            free_b_density - abundance_b * total_nuclei_density + molecule_density,
        ],
        dtype=np.float64,
    )


def two_element_molecule_jacobian(
    densities: np.ndarray,
    *,
    total_particle_density: float,
    elemental_abundances: np.ndarray,
    formation_constant: float,
) -> np.ndarray:
    """Return the analytic Jacobian of :func:`two_element_molecule_residual`."""

    _, free_a_density, free_b_density = np.asarray(
        densities,
        dtype=np.float64,
    )
    abundance_a, abundance_b = _validated_two_element_inputs(
        total_particle_density,
        elemental_abundances,
        formation_constant,
    )
    derivative_a = float(formation_constant) * free_b_density
    derivative_b = float(formation_constant) * free_a_density
    return np.array(
        [
            [0.0, 1.0 + derivative_a, 1.0 + derivative_b],
            [-abundance_a, 1.0 + derivative_a, derivative_b],
            [-abundance_b, derivative_a, 1.0 + derivative_b],
        ],
        dtype=np.float64,
    )


def finite_difference_jacobian(
    function,
    point: np.ndarray,
    *,
    relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Estimate a small square Jacobian with positive central differences."""

    point = np.asarray(point, dtype=np.float64)
    baseline = np.asarray(function(point), dtype=np.float64)
    jacobian = np.empty((baseline.size, point.size), dtype=np.float64)
    for column in range(point.size):
        step = relative_step * max(abs(float(point[column])), 1.0)
        if point[column] - step <= 0.0:
            step = 0.5 * point[column]
        offset = np.zeros_like(point)
        offset[column] = step
        jacobian[:, column] = (
            np.asarray(function(point + offset)) - np.asarray(function(point - offset))
        ) / (2.0 * step)
    return jacobian


def equal_abundance_closed_form(
    *,
    total_particle_density: float,
    formation_constant: float,
) -> TwoElementMoleculeSolution:
    """Solve the equal-A/B transparent network without iteration."""

    _validated_two_element_inputs(
        total_particle_density,
        np.array([0.5, 0.5]),
        formation_constant,
    )
    particle_density = float(total_particle_density)
    constant = float(formation_constant)
    if constant == 0.0:
        free_density = particle_density / 2.0
    else:
        free_density = particle_density / (
            np.sqrt(1.0 + constant * particle_density) + 1.0
        )
    molecule_density = constant * free_density**2
    nuclei_density = 2.0 * (free_density + molecule_density)
    residual = two_element_molecule_residual(
        np.array([nuclei_density, free_density, free_density]),
        total_particle_density=particle_density,
        elemental_abundances=np.array([0.5, 0.5]),
        formation_constant=constant,
    )
    return TwoElementMoleculeSolution(
        total_nuclei_density=nuclei_density,
        free_a_density=free_density,
        free_b_density=free_density,
        molecule_density=molecule_density,
        residual=residual,
        iterations=0,
        converged=True,
    )


def solve_two_element_molecule(
    *,
    total_particle_density: float,
    elemental_abundances: np.ndarray,
    formation_constant: float,
    relative_tolerance: float = 1.0e-12,
    maximum_iterations: int = 50,
) -> TwoElementMoleculeSolution:
    """Solve the transparent density-space root with positive backtracking."""

    abundances = np.asarray(elemental_abundances, dtype=np.float64)
    abundance_a, abundance_b = _validated_two_element_inputs(
        total_particle_density,
        abundances,
        formation_constant,
    )
    particle_density = float(total_particle_density)
    densities = np.array(
        [
            particle_density,
            abundance_a * particle_density,
            abundance_b * particle_density,
        ],
        dtype=np.float64,
    )
    residual = np.zeros(3, dtype=np.float64)
    for iteration in range(1, int(maximum_iterations) + 1):
        residual = two_element_molecule_residual(
            densities,
            total_particle_density=particle_density,
            elemental_abundances=abundances,
            formation_constant=formation_constant,
        )
        jacobian = two_element_molecule_jacobian(
            densities,
            total_particle_density=particle_density,
            elemental_abundances=abundances,
            formation_constant=formation_constant,
        )
        update = np.linalg.solve(jacobian, residual)
        relative_update = np.max(
            np.abs(update) / np.maximum(np.abs(densities), 1.0e-300)
        )
        step_fraction = 1.0
        residual_norm = np.max(np.abs(residual))
        while step_fraction >= 2.0**-20:
            candidate = densities - step_fraction * update
            if np.all(candidate > 0.0):
                candidate_residual = two_element_molecule_residual(
                    candidate,
                    total_particle_density=particle_density,
                    elemental_abundances=abundances,
                    formation_constant=formation_constant,
                )
                if np.max(np.abs(candidate_residual)) <= residual_norm:
                    densities = candidate
                    residual = candidate_residual
                    break
            step_fraction *= 0.5
        else:
            break
        if relative_update <= relative_tolerance:
            return _solution_from_densities(
                densities,
                residual,
                formation_constant,
                iteration,
                True,
            )
    return _solution_from_densities(
        densities,
        residual,
        formation_constant,
        int(maximum_iterations),
        False,
    )


def _solution_from_densities(
    densities: np.ndarray,
    residual: np.ndarray,
    formation_constant: float,
    iterations: int,
    converged: bool,
) -> TwoElementMoleculeSolution:
    """Package one transparent-network state."""

    nuclei, free_a, free_b = np.asarray(densities, dtype=np.float64)
    return TwoElementMoleculeSolution(
        total_nuclei_density=float(nuclei),
        free_a_density=float(free_a),
        free_b_density=float(free_b),
        molecule_density=float(formation_constant) * free_a * free_b,
        residual=np.asarray(residual, dtype=np.float64),
        iterations=int(iterations),
        converged=bool(converged),
    )


def _validated_two_element_inputs(
    total_particle_density: float,
    elemental_abundances: np.ndarray,
    formation_constant: float,
) -> tuple[float, float]:
    """Validate the transparent network without silently renormalizing it."""

    particle_density = float(total_particle_density)
    constant = float(formation_constant)
    abundances = np.asarray(elemental_abundances, dtype=np.float64)
    if not np.isfinite(particle_density) or particle_density <= 0.0:
        raise ValueError("total_particle_density must be positive and finite")
    if not np.isfinite(constant) or constant < 0.0:
        raise ValueError("formation_constant must be finite and nonnegative")
    if abundances.shape != (2,):
        raise ValueError("elemental_abundances must have shape (2,)")
    if np.any(~np.isfinite(abundances)) or np.any(abundances <= 0.0):
        raise ValueError("both elemental abundances must be positive and finite")
    if not np.isclose(np.sum(abundances), 1.0, rtol=0.0, atol=1.0e-14):
        raise ValueError("two-element abundances must sum to one")
    return float(abundances[0]), float(abundances[1])
