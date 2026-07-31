"""Small transparent continuum constructions used before exact route calls.

The functions in this module expose two numerical ideas that are easy to hide
inside the full continuum implementation:

* one frequency worker owns one complete depth column; and
* three one-sided edge samples reconstruct positive opacity in log space.

They use only local arrays.  They do not read a fixture, a golden archive, or
the external Payne Zero checkout.
"""

from __future__ import annotations

from dataclasses import dataclass

from numba import njit, prange
import numpy as np


@dataclass(frozen=True)
class ThreePointBasis:
    """Quadratic basis values for one continuum-edge interval."""

    left: np.ndarray
    midpoint: np.ndarray
    right: np.ndarray

    @property
    def sum(self) -> np.ndarray:
        """Return the partition-of-unity check."""

        return self.left + self.midpoint + self.right


def three_point_edge_basis(
    target_wavelength_nm: np.ndarray,
    left_wavelength_nm: np.ndarray | float,
    right_wavelength_nm: np.ndarray | float,
) -> ThreePointBasis:
    """Return the exact midpoint Lagrange basis used by continuum sampling."""

    target = np.asarray(target_wavelength_nm, dtype=np.float64)
    left = np.asarray(left_wavelength_nm, dtype=np.float64)
    right = np.asarray(right_wavelength_nm, dtype=np.float64)
    try:
        target, left, right = np.broadcast_arrays(target, left, right)
    except ValueError as error:
        raise ValueError("target, left, and right wavelengths must broadcast") from error
    if not np.all(np.isfinite(target)):
        raise ValueError("target_wavelength_nm must be finite")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("interval wavelengths must be finite")
    if np.any(left <= 0.0) or np.any(right <= left):
        raise ValueError("every interval must satisfy 0 < left < right")
    if np.any(target < left) or np.any(target > right):
        raise ValueError("every target wavelength must lie inside its interval")
    midpoint = 0.5 * (left + right)
    denominator = 0.5 * (right - left) ** 2
    return ThreePointBasis(
        left=(target - midpoint) * (target - right) / denominator,
        midpoint=2.0 * (left - target) * (target - right) / denominator,
        right=(target - left) * (target - midpoint) / denominator,
    )


def reconstruct_positive_opacity(
    sampled_opacity_cm2_per_g: np.ndarray,
    basis: ThreePointBasis,
    *,
    floor_cm2_per_g: float = 1.0e-30,
) -> np.ndarray:
    """Interpolate three positive opacity samples in base-10 logarithmic space."""

    sampled = np.asarray(sampled_opacity_cm2_per_g, dtype=np.float64)
    if sampled.ndim < 1 or sampled.shape[-1] != 3:
        raise ValueError("sampled opacity must have a final left/midpoint/right axis")
    if not np.all(np.isfinite(sampled)):
        raise ValueError("sampled opacity must be finite")
    if np.any(sampled < 0.0):
        raise ValueError("sampled opacity cannot be negative")
    if not np.isfinite(floor_cm2_per_g) or floor_cm2_per_g <= 0.0:
        raise ValueError("floor_cm2_per_g must be positive and finite")

    logarithm = np.log10(np.maximum(sampled, floor_cm2_per_g))
    return 10.0 ** (
        logarithm[..., 0] * basis.left
        + logarithm[..., 1] * basis.midpoint
        + logarithm[..., 2] * basis.right
    )


def python_frequency_columns(
    absorber_number_density_cm3: np.ndarray,
    cross_section_cm2: np.ndarray,
    mass_density_g_cm3: np.ndarray,
) -> np.ndarray:
    """Compute ``n[d] * sigma[f] / rho[d]`` with explicit Python loops."""

    number_density, cross_section, mass_density = _column_inputs(
        absorber_number_density_cm3,
        cross_section_cm2,
        mass_density_g_cm3,
    )
    opacity = np.empty(
        (number_density.size, cross_section.size),
        dtype=np.float64,
    )
    for frequency_index in range(cross_section.size):
        for depth_index in range(number_density.size):
            opacity[depth_index, frequency_index] = (
                number_density[depth_index]
                * cross_section[frequency_index]
                / mass_density[depth_index]
            )
    return opacity


@njit(cache=True, nogil=True)
def njit_frequency_columns(
    absorber_number_density_cm3: np.ndarray,
    cross_section_cm2: np.ndarray,
    mass_density_g_cm3: np.ndarray,
) -> np.ndarray:
    """Compile the same ordered frequency-column loop."""

    opacity = np.empty(
        (absorber_number_density_cm3.size, cross_section_cm2.size),
        dtype=np.float64,
    )
    for frequency_index in range(cross_section_cm2.size):
        for depth_index in range(absorber_number_density_cm3.size):
            opacity[depth_index, frequency_index] = (
                absorber_number_density_cm3[depth_index]
                * cross_section_cm2[frequency_index]
                / mass_density_g_cm3[depth_index]
            )
    return opacity


@njit(cache=True, nogil=True, parallel=True)
def prange_frequency_columns(
    absorber_number_density_cm3: np.ndarray,
    cross_section_cm2: np.ndarray,
    mass_density_g_cm3: np.ndarray,
) -> np.ndarray:
    """Parallelize only the independent frequency-column owner."""

    opacity = np.empty(
        (absorber_number_density_cm3.size, cross_section_cm2.size),
        dtype=np.float64,
    )
    for frequency_index in prange(cross_section_cm2.size):
        for depth_index in range(absorber_number_density_cm3.size):
            opacity[depth_index, frequency_index] = (
                absorber_number_density_cm3[depth_index]
                * cross_section_cm2[frequency_index]
                / mass_density_g_cm3[depth_index]
            )
    return opacity


def _column_inputs(
    absorber_number_density_cm3: np.ndarray,
    cross_section_cm2: np.ndarray,
    mass_density_g_cm3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the transparent column-kernel inputs outside compiled code."""

    number_density = np.asarray(absorber_number_density_cm3, dtype=np.float64)
    cross_section = np.asarray(cross_section_cm2, dtype=np.float64)
    mass_density = np.asarray(mass_density_g_cm3, dtype=np.float64)
    if number_density.ndim != 1 or mass_density.ndim != 1:
        raise ValueError("depth inputs must be one-dimensional")
    if cross_section.ndim != 1:
        raise ValueError("cross_section_cm2 must be one-dimensional")
    if number_density.shape != mass_density.shape:
        raise ValueError("number-density and mass-density depth shapes must match")
    for name, values in (
        ("absorber_number_density_cm3", number_density),
        ("cross_section_cm2", cross_section),
        ("mass_density_g_cm3", mass_density),
    ):
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{name} must be positive and finite")
    return number_density, cross_section, mass_density


__all__ = [
    "ThreePointBasis",
    "njit_frequency_columns",
    "prange_frequency_columns",
    "python_frequency_columns",
    "reconstruct_positive_opacity",
    "three_point_edge_basis",
]
