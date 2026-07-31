"""Small transparent constructions used before Chapter 9's exact kernels."""

from __future__ import annotations

import numpy as np


def formal_surface_intensity(
    optical_depth: np.ndarray,
    source: np.ndarray,
    direction_cosine: float,
    *,
    bottom_intensity: float = 0.0,
) -> float:
    """Integrate the outward formal solution for one prescribed source."""

    tau = np.asarray(optical_depth, dtype=np.float64)
    source_array = np.asarray(source, dtype=np.float64)
    mu = float(direction_cosine)
    if tau.ndim != 1 or source_array.shape != tau.shape:
        raise ValueError("optical_depth and source must be equal one-dimensional arrays")
    if tau.size < 2 or np.any(np.diff(tau) <= 0.0):
        raise ValueError("optical_depth must contain at least two increasing points")
    if not 0.0 < mu <= 1.0:
        raise ValueError("direction_cosine must lie in (0, 1]")
    contribution = source_array * np.exp(-tau / mu) / mu
    return float(
        bottom_intensity * np.exp(-tau[-1] / mu)
        + np.trapezoid(contribution, tau)
    )


def angular_moments(
    direction_cosine: np.ndarray,
    quadrature_weight: np.ndarray,
    intensity: np.ndarray,
) -> tuple[float, float, float]:
    """Return teaching-quadrature estimates of ``J_nu``, ``H_nu``, and ``K_nu``."""

    mu = np.asarray(direction_cosine, dtype=np.float64)
    weight = np.asarray(quadrature_weight, dtype=np.float64)
    intensity_array = np.asarray(intensity, dtype=np.float64)
    if mu.shape != weight.shape or mu.shape != intensity_array.shape:
        raise ValueError("direction, weight, and intensity arrays must have one shape")
    return (
        float(0.5 * np.sum(weight * intensity_array)),
        float(0.5 * np.sum(weight * mu * intensity_array)),
        float(0.5 * np.sum(weight * mu * mu * intensity_array)),
    )


def one_backward_source_sweep(
    thermal_source: np.ndarray,
    scattering_fraction: np.ndarray,
    lambda_operator: np.ndarray,
) -> np.ndarray:
    """Expose one deepest-to-surface Gauss--Seidel source sweep."""

    import torch

    thermal = torch.as_tensor(
        np.asarray(thermal_source, dtype=np.float32)[None, :]
    )
    alpha = torch.as_tensor(
        np.asarray(scattering_fraction, dtype=np.float32)[None, :]
    )
    operator = torch.as_tensor(
        np.asarray(lambda_operator, dtype=np.float32)
    )
    if thermal.ndim != 2 or alpha.shape != thermal.shape:
        raise ValueError("thermal_source and scattering_fraction need one shared axis")
    if operator.shape != (thermal.shape[1], thermal.shape[1]):
        raise ValueError("lambda_operator must be square on the source axis")
    diagonal = torch.diagonal(operator).contiguous()
    thermal_emission = thermal * (1.0 - alpha)
    denominator = 1.0 - alpha * diagonal
    alpha_by_depth = alpha.transpose(0, 1).contiguous()
    thermal_by_depth = thermal_emission.transpose(0, 1).contiguous()
    denominator_by_depth = denominator.transpose(0, 1).contiguous()
    source_by_depth = thermal.transpose(0, 1).contiguous()
    floor = torch.tensor(1.0e-38, dtype=torch.float32)
    for depth_index in range(source_by_depth.shape[0] - 1, -1, -1):
        mean_intensity = torch.matmul(operator[depth_index], source_by_depth)
        correction = (
            mean_intensity * alpha_by_depth[depth_index]
            + thermal_by_depth[depth_index]
            - source_by_depth[depth_index]
        ) / denominator_by_depth[depth_index]
        source_by_depth[depth_index] = torch.maximum(
            source_by_depth[depth_index] + correction,
            floor,
        )
    return source_by_depth.transpose(0, 1)[0].numpy()
