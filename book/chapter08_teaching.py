"""Readable scalar helpers for Chapter 8's molecular text compiler.

The production compiler lives in the exact staged Payne Zero module.  This
module keeps one short scalar mirror so the notebook can expose the decisions
before comparing them with the serial cached-Numba kernel.
"""

from __future__ import annotations

import math

import numpy as np


def compile_text_rows_scalar(
    *,
    stored_wavelength_nm: np.ndarray,
    log_oscillator_strength: np.ndarray,
    first_energy_cm: np.ndarray,
    second_energy_cm: np.ndarray,
    source_code: np.ndarray,
    isotope_index: np.ndarray,
    radiative_damping_log_scaled: np.ndarray,
    upper_label_is_ground_state: np.ndarray,
    start_wavelength_nm: float,
    end_wavelength_nm: float,
    resolution: float,
    use_energy_level_wavelengths: bool = True,
    include_predicted_lines: bool = False,
) -> dict[str, np.ndarray]:
    """Compile one source-ordered band with the exact scalar arithmetic."""

    from payne_zero_synthesis.constants import (
        CLASSICAL_LINE_STRENGTH_COEFFICIENT,
        LIGHT_SPEED_NM_PER_S,
        NATURAL_LOG_10,
    )
    from payne_zero_synthesis.source_catalog_molecular_compiler import (
        _compiled_molecular_arrays,
        _dispatch_molecule,
        _molecular_text_wavelength_nm,
    )

    log_grid_ratio = math.log(1.0 + 1.0 / float(resolution))
    grid_origin_index = math.floor(
        math.log(float(start_wavelength_nm)) / log_grid_ratio
    )
    if math.exp(grid_origin_index * log_grid_ratio) < start_wavelength_nm:
        grid_origin_index += 1
    window_min_nm = float(start_wavelength_nm) - 0.01
    window_max_nm = float(end_wavelength_nm) + 0.1

    output: list[list[float | int]] = [[] for _ in range(8)]
    for row in range(np.asarray(stored_wavelength_nm).size):
        stored = float(stored_wavelength_nm[row])
        first_energy = float(first_energy_cm[row])
        second_energy = float(second_energy_cm[row])
        if abs(stored) == 0.0:
            continue
        if not include_predicted_lines and (
            first_energy < 0.0 or second_energy < 0.0
        ):
            continue
        wavelength_nm = _molecular_text_wavelength_nm(
            stored,
            first_energy,
            second_energy,
            use_energy_level_wavelengths,
        )
        if wavelength_nm < window_min_nm:
            continue
        if wavelength_nm > window_max_nm:
            break
        if use_energy_level_wavelengths:
            source_position_nm = abs(stored)
            if source_position_nm > window_max_nm + 10.0 or (
                0.0 < source_position_nm < window_min_nm - 10.0
            ):
                continue

        dispatch = _dispatch_molecule(
            int(source_code[row]),
            int(isotope_index[row]),
        )
        if dispatch is None:
            continue
        species, _, _, primary_log_weight, secondary_log_weight = dispatch
        oscillator_strength = math.exp(
            (
                float(log_oscillator_strength[row])
                + primary_log_weight
                + secondary_log_weight
            )
            * NATURAL_LOG_10
        )
        lower_excitation = min(abs(first_energy), abs(second_energy))
        grid_position = math.log(max(wavelength_nm, 1.0e-30)) / log_grid_ratio + 0.5
        center_index_1based = int(grid_position) - grid_origin_index + 1
        frequency_hz = LIGHT_SPEED_NM_PER_S / max(wavelength_nm, 1.0e-30)
        classical_strength = (
            CLASSICAL_LINE_STRENGTH_COEFFICIENT
            * oscillator_strength
            / frequency_hz
        )
        frequency_4pi = frequency_hz * 12.5664
        radiative_gamma = 10.0 ** (
            float(radiative_damping_log_scaled[row]) * 0.01
        )
        stark_gamma = 3.0e-5
        van_der_waals_gamma = 1.0e-7
        if bool(upper_label_is_ground_state[row]):
            stark_gamma = 3.0e-8
            van_der_waals_gamma = 1.0e-8
        values = (
            center_index_1based,
            classical_strength,
            species,
            lower_excitation,
            radiative_gamma / frequency_4pi,
            stark_gamma / frequency_4pi,
            van_der_waals_gamma / frequency_4pi,
            7,
        )
        for field, value in zip(output, values, strict=True):
            field.append(value)

    return _compiled_molecular_arrays(*output)
