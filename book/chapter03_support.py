"""Small teaching helpers for Chapter 3's analytic thermochemistry checks.

These functions use the parity-pinned EOS constants but are deliberately
book-defined. Production atmosphere and synthesis states are computed by the
exact Payne Zero modules staged under ``src/``.
"""

from __future__ import annotations

import numpy as np

from payne_zero_synthesis.constants import (
    LIGHT_SPEED_CM_PER_S,
    REFERENCE_BOLTZMANN_ERG_PER_K,
    REFERENCE_BOLTZMANN_EV_PER_K,
    REFERENCE_PLANCK_ERG_SECOND,
    REFERENCE_SAHA_COEFFICIENT,
)


REFERENCE_HC_OVER_K_CM_K = (
    REFERENCE_PLANCK_ERG_SECOND
    * LIGHT_SPEED_CM_PER_S
    / REFERENCE_BOLTZMANN_ERG_PER_K
)


def two_level_lte_populations(
    temperature_k: np.ndarray,
    energy_cm: np.ndarray,
    statistical_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return partition function and normalized level fractions."""

    temperature = np.asarray(temperature_k, np.float64)
    energy = np.asarray(energy_cm, np.float64)
    weight = np.asarray(statistical_weight, np.float64)
    if temperature.ndim != 1 or np.any(temperature <= 0.0):
        raise ValueError("temperature_k must be a positive one-dimensional array")
    if energy.ndim != 1 or weight.shape != energy.shape:
        raise ValueError("energy_cm and statistical_weight must share one axis")
    boltzmann_weight = weight[None, :] * np.exp(
        -REFERENCE_HC_OVER_K_CM_K * energy[None, :] / temperature[:, None]
    )
    partition_function = np.sum(boltzmann_weight, axis=1)
    level_fraction = boltzmann_weight / partition_function[:, None]
    return partition_function, level_fraction


def two_stage_saha_fractions(
    temperature_k: np.ndarray,
    electron_density_cm3: np.ndarray,
    lower_partition: np.ndarray,
    upper_partition: np.ndarray,
    ionization_energy_ev: float,
    lowering_energy_ev: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper fractions and their natural-log Saha ratio."""

    temperature = np.asarray(temperature_k, np.float64)
    electron_density = np.asarray(electron_density_cm3, np.float64)
    lower = np.asarray(lower_partition, np.float64)
    upper = np.asarray(upper_partition, np.float64)
    lowering = np.asarray(lowering_energy_ev, np.float64)
    if np.any(temperature <= 0.0):
        raise ValueError("temperature_k must be positive")
    if np.any(electron_density <= 0.0):
        raise ValueError("electron_density_cm3 must be positive")
    if np.any(lower <= 0.0) or np.any(upper <= 0.0):
        raise ValueError("partition functions must be positive")
    log_ratio = (
        np.log(2.0 * REFERENCE_SAHA_COEFFICIENT)
        + 1.5 * np.log(temperature)
        - np.log(electron_density)
        + np.log(upper)
        - np.log(lower)
        - (float(ionization_energy_ev) - lowering)
        / (REFERENCE_BOLTZMANN_EV_PER_K * temperature)
    )
    log_normalizer = np.logaddexp(0.0, log_ratio)
    lower_fraction = np.exp(-log_normalizer)
    upper_fraction = np.exp(log_ratio - log_normalizer)
    return lower_fraction, upper_fraction, log_ratio


def damped_hydrogen_electron_fixed_point(
    *,
    temperature_k: float,
    gas_pressure_dyn_cm2: float,
    electron_density_seed_cm3: float,
    ionization_energy_ev: float,
    lower_partition: float = 2.0,
    upper_partition: float = 1.0,
    tolerance: float = 1.0e-4,
    max_iterations: int = 200,
) -> dict[str, np.ndarray | float | bool]:
    """Iterate the exact half-step damping rule for a hydrogen-only gas."""

    if temperature_k <= 0.0 or gas_pressure_dyn_cm2 <= 0.0:
        raise ValueError("temperature and gas pressure must be positive")
    if electron_density_seed_cm3 <= 0.0:
        raise ValueError("electron-density seed must be positive")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    total_particle_density = (
        gas_pressure_dyn_cm2
        / (REFERENCE_BOLTZMANN_ERG_PER_K * temperature_k)
    )
    if electron_density_seed_cm3 >= total_particle_density:
        raise ValueError(
            "electron-density seed must leave a positive nuclei density"
        )
    electron_density = float(electron_density_seed_cm3)
    history = []
    for _ in range(max_iterations):
        nuclei_density = total_particle_density - electron_density
        _, upper_fraction, _ = two_stage_saha_fractions(
            np.array([temperature_k]),
            np.array([electron_density]),
            np.array([lower_partition]),
            np.array([upper_partition]),
            ionization_energy_ev,
        )
        raw_charge = nuclei_density * float(upper_fraction[0])
        bounded_charge = max(raw_charge, 0.5 * electron_density)
        updated = 0.5 * (bounded_charge + electron_density)
        residual = abs((electron_density - updated) / max(updated, 1.0e-300))
        history.append((electron_density, raw_charge, updated, residual))
        electron_density = updated
        if residual < tolerance:
            break
    values = np.asarray(history, np.float64)
    return {
        "electron_density_cm3": electron_density,
        "total_particle_density_cm3": total_particle_density,
        "history": values,
        "converged": bool(values.size and values[-1, 3] < tolerance),
    }


def neutral_collision_density_proxy(
    temperature_k: np.ndarray,
    hydrogen_neutral_cm3: np.ndarray,
    helium_neutral_cm3: np.ndarray,
    molecular_hydrogen_cm3: np.ndarray,
) -> np.ndarray:
    """Return the ordinary-line neutral-perturber density proxy."""

    temperature = np.asarray(temperature_k, np.float64)
    neutral_sum = (
        np.asarray(hydrogen_neutral_cm3, np.float64)
        + 0.42 * np.asarray(helium_neutral_cm3, np.float64)
        + 0.85 * np.asarray(molecular_hydrogen_cm3, np.float64)
    )
    return neutral_sum * (temperature / 1.0e4) ** 0.3


def fractional_doppler_width(
    temperature_k: np.ndarray,
    microturbulence_cm_s: np.ndarray,
    particle_mass_gram: float,
) -> np.ndarray:
    """Return the thermal-plus-microturbulent Doppler width divided by c."""

    thermal_energy = REFERENCE_BOLTZMANN_ERG_PER_K * np.asarray(
        temperature_k, np.float64
    )
    microturbulence = np.asarray(microturbulence_cm_s, np.float64)
    return (
        np.sqrt(2.0 * thermal_energy / float(particle_mass_gram) + microturbulence**2)
        / LIGHT_SPEED_CM_PER_S
    )
