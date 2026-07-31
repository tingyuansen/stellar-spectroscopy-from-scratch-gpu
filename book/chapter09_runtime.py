"""Self-contained Chapter 9 transfer checkpoints.

The compact spectral window reuses the continuum, atomic, and molecular
products built by Chapters 5--8, then calls the exact staged transfer
implementations.  No helper opens a comparison golden or an external source
checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

from book.chapter09_teaching import one_backward_source_sweep


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYNTHESIS_TRANSFER_TABLES = (
    REPOSITORY_ROOT / "data/static/synthesis_tables/transfer_tables.npz"
)
ATMOSPHERE_TRANSFER_TABLES = (
    REPOSITORY_ROOT / "data/static/atmosphere_tables/radiative_transfer_tables.npz"
)
SYNTHESIS_TRANSFER_TABLES_SHA256 = (
    "64f75de9af02697c0b97b7bbf919f6fed9d646622f18859eb5f66ff66e7f7a7b"
)
ATMOSPHERE_TRANSFER_TABLES_SHA256 = (
    "d69fcad9e22dd8dd42634e5720df717f0298849d98ee2cd93236009e22391e56"
)


@dataclass(frozen=True)
class EqualExtinctionCheckpoint:
    extinction: np.ndarray
    scattering_fraction: np.ndarray
    thermal_source: np.ndarray
    eddington_flux_per_frequency: np.ndarray


@dataclass(frozen=True)
class OpticalDepthCheckpoint:
    column_mass: np.ndarray
    extinction: np.ndarray
    optical_depth: np.ndarray
    line_center_index: int
    continuum_index: int


@dataclass(frozen=True)
class TransferTableCheckpoint:
    synthesis_fields: tuple[str, ...]
    synthesis_shapes: tuple[tuple[int, ...], ...]
    synthesis_dtype: str
    synthesis_device: str
    atmosphere_shapes: tuple[tuple[int, ...], ...]
    atmosphere_dtypes: tuple[str, ...]
    shared_grid_exact: bool
    shared_surface_weights_exact: bool
    mean_operator_differing_entries: int
    mean_operator_max_abs_difference: float


@dataclass(frozen=True)
class RemapCheckpoint:
    optical_depth: np.ndarray
    transfer_grid: np.ndarray
    thermal_source_grid: np.ndarray
    scattering_fraction_grid: np.ndarray
    exact_point_max_abs: float
    above_atmosphere_count: int
    above_atmosphere_source_exact: bool
    above_atmosphere_scattering_exact: bool


@dataclass(frozen=True)
class HandSweepCheckpoint:
    thermal_source: np.ndarray
    scattering_fraction: np.ndarray
    lambda_operator: np.ndarray
    hand_source: np.ndarray
    exact_source: np.ndarray


@dataclass(frozen=True)
class SourceSweepCheckpoint:
    transfer_grid: np.ndarray
    thermal_source: np.ndarray
    one_sweep_source: np.ndarray
    eight_sweep_source: np.ndarray
    minimum_after_each_sweep: np.ndarray
    source_dtype: str


@dataclass(frozen=True)
class SaturatedRouteCheckpoint:
    first_layer_optical_depth: np.ndarray
    eddington_flux_per_frequency: np.ndarray
    strict_failure_seen: bool
    saturated_mask: np.ndarray
    threshold: float


@dataclass(frozen=True)
class PreparedWindowCheckpoint:
    wavelength_nm: np.ndarray
    column_mass: np.ndarray
    temperature: np.ndarray
    continuum_absorption: np.ndarray
    continuum_scattering: np.ndarray
    atomic_line_absorption: np.ndarray
    molecular_line_absorption: np.ndarray
    line_mass_absorption_coefficient: np.ndarray
    planck_source: np.ndarray
    eddington_flux_total_per_frequency: np.ndarray
    eddington_flux_continuum_per_frequency: np.ndarray
    normalized_flux: np.ndarray
    zero_line_normalized_flux: np.ndarray
    stacked_max_abs_difference: float


@dataclass(frozen=True)
class ContributionCheckpoint:
    column_mass: np.ndarray
    continuum_contribution: np.ndarray
    line_contribution: np.ndarray
    continuum_index: int
    line_center_index: int
    continuum_peak_column_mass: float
    line_peak_column_mass: float


@dataclass(frozen=True)
class FluxConversionCheckpoint:
    wavelength_nm: np.ndarray
    eddington_flux_per_frequency: np.ndarray
    flux_per_frequency: np.ndarray
    flux_per_wavelength_nm: np.ndarray
    helper_flux_per_wavelength_nm: np.ndarray
    normalized_before: np.ndarray
    normalized_after: np.ndarray


@dataclass(frozen=True)
class AtmosphereMomentCheckpoint:
    column_mass: np.ndarray
    optical_depth: np.ndarray
    source: np.ndarray
    eddington_flux: np.ndarray
    mean_intensity: np.ndarray
    mean_intensity_minus_source: np.ndarray
    total_opacity: np.ndarray
    scattering_fraction: np.ndarray
    surface_second_moment: float
    mapped_layer_count: int
    gross_line_mass_absorption_coefficient: np.ndarray
    stimulated: np.ndarray
    stimulated_line_mass_absorption_coefficient: np.ndarray


@dataclass(frozen=True)
class AtmosphereParallelCheckpoint:
    output_names: tuple[str, ...]
    serial_outputs: tuple[np.ndarray, ...]
    parallel_outputs: tuple[np.ndarray, ...]
    repeated_outputs: tuple[np.ndarray, ...]
    worst_absolute_difference: float
    worst_relative_difference: float
    worst_output_name: str
    fixed_policy_repeatable: bool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_staged_source() -> None:
    """Resolve every Payne Zero import from this repository's staged source."""

    from book.chapter06_runtime import configure_local_data_paths

    configure_local_data_paths()


def validate_transfer_inputs() -> None:
    """Fail closed if either lane's static operator archive changes."""

    expected = {
        SYNTHESIS_TRANSFER_TABLES: SYNTHESIS_TRANSFER_TABLES_SHA256,
        ATMOSPHERE_TRANSFER_TABLES: ATMOSPHERE_TRANSFER_TABLES_SHA256,
    }
    for path, digest in expected.items():
        if _sha256(path) != digest:
            raise RuntimeError(f"Chapter 9 transfer table changed: {path}")


@lru_cache(maxsize=1)
def synthesis_transfer_tables():
    """Load the exact synthesis operators on CPU in float64."""

    _configure_staged_source()
    validate_transfer_inputs()
    import torch

    from payne_zero_synthesis.radiative_transfer import TransferTables

    return TransferTables.from_npz(
        SYNTHESIS_TRANSFER_TABLES,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def equal_extinction_checkpoint() -> EqualExtinctionCheckpoint:
    """Compare absorption-rich and scattering-rich columns at fixed extinction."""

    _configure_staged_source()
    import torch

    from payne_zero_synthesis.radiative_transfer import (
        _solve_flux_rows,
        source_and_alpha,
    )

    column_mass = torch.logspace(-5, 1, 18, dtype=torch.float64)
    depth_shape = (2, column_mass.numel())
    extinction_target = torch.linspace(0.15, 1.2, column_mass.numel())[None, :]
    extinction_target = extinction_target.expand(depth_shape).contiguous()
    absorption_fraction = torch.tensor([0.9, 0.1], dtype=torch.float64)[:, None]
    continuum_absorption = extinction_target * absorption_fraction
    continuum_scattering = extinction_target - continuum_absorption
    thermal_source = torch.linspace(0.35, 2.0, column_mass.numel())[None, :]
    thermal_source = thermal_source.expand(depth_shape).contiguous()
    zeros = torch.zeros_like(extinction_target)
    extinction, alpha, recovered_thermal = source_and_alpha(
        continuum_absorption,
        thermal_source,
        zeros,
        thermal_source,
        continuum_scattering,
        zeros,
    )
    flux = _solve_flux_rows(
        continuum_absorption,
        thermal_source,
        zeros,
        thermal_source,
        continuum_scattering,
        zeros,
        column_mass,
        synthesis_transfer_tables(),
        assert_no_saturated_core=True,
    )
    return EqualExtinctionCheckpoint(
        extinction=extinction.detach().numpy(),
        scattering_fraction=alpha.detach().numpy(),
        thermal_source=recovered_thermal.detach().numpy(),
        eddington_flux_per_frequency=flux.detach().numpy(),
    )


@lru_cache(maxsize=1)
def prepared_window_checkpoint() -> PreparedWindowCheckpoint:
    """Solve one compact Chapter 5--8 opacity window through both branches."""

    _configure_staged_source()
    import torch

    from book.chapter05_runtime import (
        load_regime_state,
        run_synthesis_continuum,
    )
    from book.chapter07_runtime import run_atomic_forest
    from book.chapter08_runtime import molecular_opacity_checkpoint
    from payne_zero_synthesis.radiative_transfer import (
        _solve_flux_rows,
        planck_bnu,
        solve_spectrum,
    )

    forest = run_atomic_forest("cool_molecule_rich")
    molecular = molecular_opacity_checkpoint()
    if not np.array_equal(forest.wavelength_nm, molecular.wavelength_nm):
        raise RuntimeError("Chapter 7 and 8 windows no longer share one wavelength grid")
    continuum = run_synthesis_continuum(
        "cool_molecule_rich",
        forest.wavelength_nm,
    )
    state = load_regime_state("cool_molecule_rich", "synthesis")
    wavelength = torch.as_tensor(forest.wavelength_nm, dtype=torch.float64)
    column_mass = torch.as_tensor(state["column_mass"], dtype=torch.float64)
    temperature = torch.as_tensor(state["temperature"], dtype=torch.float64)
    continuum_absorption = torch.as_tensor(
        continuum.absorption_cm2_per_g,
        dtype=torch.float64,
    )
    continuum_scattering = torch.as_tensor(
        continuum.scattering_cm2_per_g,
        dtype=torch.float64,
    )
    atomic_line = np.asarray(
        forest.net_line_mass_absorption_coefficient,
        dtype=np.float64,
    )
    molecular_line = np.asarray(molecular.combined_opacity, dtype=np.float64)
    line_absorption = torch.as_tensor(
        atomic_line + molecular_line,
        dtype=torch.float64,
    )
    planck = planck_bnu(wavelength, temperature)
    zeros = torch.zeros_like(line_absorption)
    total_h, continuum_h, normalized = solve_spectrum(
        continuum_absorption,
        continuum_scattering,
        line_absorption,
        zeros,
        planck,
        column_mass,
        synthesis_transfer_tables(),
        assert_no_saturated_core=False,
    )
    zero_total_h, zero_continuum_h, zero_normalized = solve_spectrum(
        continuum_absorption,
        continuum_scattering,
        zeros,
        zeros,
        planck,
        column_mass,
        synthesis_transfer_tables(),
        assert_no_saturated_core=False,
    )
    continuum_rows = continuum_absorption.T.contiguous()
    scattering_rows = continuum_scattering.T.contiguous()
    line_rows = line_absorption.T.contiguous()
    source_rows = planck.T.contiguous()
    total_independent = _solve_flux_rows(
        continuum_rows,
        source_rows,
        line_rows,
        source_rows,
        scattering_rows,
        torch.zeros_like(line_rows),
        column_mass,
        synthesis_transfer_tables(),
        assert_no_saturated_core=False,
    )
    continuum_independent = _solve_flux_rows(
        continuum_rows,
        source_rows,
        torch.zeros_like(line_rows),
        source_rows,
        scattering_rows,
        torch.zeros_like(line_rows),
        column_mass,
        synthesis_transfer_tables(),
        assert_no_saturated_core=False,
    )
    stacked_difference = torch.max(
        torch.abs(
            torch.cat((total_h, continuum_h))
            - torch.cat((total_independent, continuum_independent))
        )
    )
    if not torch.allclose(zero_total_h, zero_continuum_h, rtol=1.0e-9, atol=1.0e-15):
        raise RuntimeError("zero-line total and continuum transfer diverged")
    return PreparedWindowCheckpoint(
        wavelength_nm=forest.wavelength_nm.copy(),
        column_mass=np.asarray(state["column_mass"], dtype=np.float64).copy(),
        temperature=np.asarray(state["temperature"], dtype=np.float64).copy(),
        continuum_absorption=continuum_absorption.numpy(),
        continuum_scattering=continuum_scattering.numpy(),
        atomic_line_absorption=atomic_line.copy(),
        molecular_line_absorption=molecular_line.copy(),
        line_mass_absorption_coefficient=line_absorption.numpy(),
        planck_source=planck.numpy(),
        eddington_flux_total_per_frequency=total_h.detach().numpy(),
        eddington_flux_continuum_per_frequency=continuum_h.detach().numpy(),
        normalized_flux=normalized.detach().numpy(),
        zero_line_normalized_flux=zero_normalized.detach().numpy(),
        stacked_max_abs_difference=float(stacked_difference),
    )


def optical_depth_checkpoint() -> OpticalDepthCheckpoint:
    """Integrate one continuum row and the strongest line-center row."""

    _configure_staged_source()
    import torch

    from payne_zero_synthesis.radiative_transfer import integrate_optical_depth

    window = prepared_window_checkpoint()
    line_strength = np.max(
        window.line_mass_absorption_coefficient,
        axis=0,
    )
    line_center_index = int(np.argmax(line_strength))
    continuum_index = int(np.argmin(line_strength))
    indices = np.asarray([continuum_index, line_center_index], dtype=np.int64)
    extinction = (
        window.continuum_absorption
        + window.continuum_scattering
        + window.line_mass_absorption_coefficient
    )[:, indices].T
    column_mass = torch.as_tensor(window.column_mass, dtype=torch.float64)
    extinction_tensor = torch.as_tensor(extinction, dtype=torch.float64)
    surface_tau = extinction_tensor[:, 0] * column_mass[0]
    optical_depth = integrate_optical_depth(
        column_mass,
        extinction_tensor,
        surface_tau,
    )
    return OpticalDepthCheckpoint(
        column_mass=window.column_mass.copy(),
        extinction=extinction.copy(),
        optical_depth=optical_depth.numpy(),
        line_center_index=line_center_index,
        continuum_index=continuum_index,
    )


def transfer_table_checkpoint() -> TransferTableCheckpoint:
    """Compare the two exact product table contracts without merging them."""

    _configure_staged_source()
    import torch

    from payne_zero_atmosphere.radiative_transfer import (
        load_radiative_transfer_tables,
    )

    synthesis = synthesis_transfer_tables()
    atmosphere = load_radiative_transfer_tables(
        ATMOSPHERE_TRANSFER_TABLES,
        force_reload=True,
    )
    with np.load(SYNTHESIS_TRANSFER_TABLES, allow_pickle=False) as archive:
        synthesis_operator_stored = np.asarray(
            archive["mean_intensity_operator"],
            dtype=np.float64,
        )
        synthesis_grid_stored = np.asarray(
            archive["transfer_optical_depth_grid"],
            dtype=np.float64,
        )
        synthesis_surface_stored = np.asarray(
            archive["surface_eddington_flux_weights"],
            dtype=np.float64,
        )
    with np.load(ATMOSPHERE_TRANSFER_TABLES, allow_pickle=False) as archive:
        atmosphere_operator_stored = np.asarray(
            archive["mean_intensity_operator"],
            dtype=np.float64,
        )
        atmosphere_grid_stored = np.asarray(
            archive["transfer_optical_depth_grid"],
            dtype=np.float64,
        )
        atmosphere_surface_stored = np.asarray(
            archive["surface_eddington_flux_weights"],
            dtype=np.float64,
        )
    difference = np.abs(synthesis_operator_stored - atmosphere_operator_stored)
    fields = (
        "transfer_optical_depth_grid",
        "mean_intensity_operator",
        "mean_intensity_diagonal",
        "surface_eddington_flux_weights",
    )
    tensors = tuple(getattr(synthesis, field) for field in fields)
    atmosphere_arrays = (
        atmosphere.surface_eddington_flux_weights,
        atmosphere.second_moment_weights,
        atmosphere.transfer_optical_depth_grid,
        atmosphere.mean_intensity_operator,
        atmosphere.eddington_flux_operator,
    )
    return TransferTableCheckpoint(
        synthesis_fields=fields,
        synthesis_shapes=tuple(tuple(tensor.shape) for tensor in tensors),
        synthesis_dtype=str(tensors[0].dtype),
        synthesis_device=str(tensors[0].device),
        atmosphere_shapes=tuple(array.shape for array in atmosphere_arrays),
        atmosphere_dtypes=tuple(str(array.dtype) for array in atmosphere_arrays),
        shared_grid_exact=np.array_equal(
            synthesis_grid_stored,
            atmosphere_grid_stored,
        ),
        shared_surface_weights_exact=np.array_equal(
            synthesis_surface_stored,
            atmosphere_surface_stored,
        ),
        mean_operator_differing_entries=int(np.count_nonzero(difference)),
        mean_operator_max_abs_difference=float(np.max(difference)),
    )


def remap_checkpoint() -> RemapCheckpoint:
    """Exercise exact transfer points and the explicit above-atmosphere rule."""

    _configure_staged_source()
    import torch

    from payne_zero_synthesis.radiative_transfer import (
        _interpolate_to_transfer_grid,
    )

    grid = synthesis_transfer_tables().transfer_optical_depth_grid
    optical_depth = grid[None, :].clone()
    thermal = (0.8 + 0.06 * grid)[None, :]
    scattering = (0.15 + 0.01 * grid)[None, :].clamp(max=0.8)
    exact_thermal = _interpolate_to_transfer_grid(optical_depth, thermal, grid)
    shifted_optical_depth = optical_depth + 1.0
    remapped_thermal = _interpolate_to_transfer_grid(
        shifted_optical_depth,
        thermal,
        grid,
    )
    remapped_scattering = _interpolate_to_transfer_grid(
        shifted_optical_depth,
        scattering,
        grid,
    )
    above = grid[None, :] < shifted_optical_depth[:, :1]
    surface_source = thermal[:, :1]
    surface_scattering = scattering[:, :1]
    remapped_thermal = torch.where(
        above,
        surface_source.expand_as(remapped_thermal),
        remapped_thermal,
    )
    remapped_scattering = torch.where(
        above,
        surface_scattering.expand_as(remapped_scattering),
        remapped_scattering,
    )
    return RemapCheckpoint(
        optical_depth=shifted_optical_depth.numpy(),
        transfer_grid=grid.numpy(),
        thermal_source_grid=remapped_thermal.numpy(),
        scattering_fraction_grid=remapped_scattering.numpy(),
        exact_point_max_abs=float(torch.max(torch.abs(exact_thermal - thermal))),
        above_atmosphere_count=int(torch.count_nonzero(above)),
        above_atmosphere_source_exact=bool(
            torch.all(remapped_thermal[above] == surface_source.item())
        ),
        above_atmosphere_scattering_exact=bool(
            torch.all(remapped_scattering[above] == surface_scattering.item())
        ),
    )


def hand_sweep_checkpoint() -> HandSweepCheckpoint:
    """Match one readable backward correction to the exact Torch kernel."""

    _configure_staged_source()
    import torch

    from payne_zero_synthesis.radiative_transfer import solve_scattering_source

    thermal = np.asarray([0.8, 1.1, 1.5], dtype=np.float32)
    alpha = np.asarray([0.2, 0.45, 0.7], dtype=np.float32)
    operator = np.asarray(
        [
            [0.45, 0.18, 0.04],
            [0.20, 0.50, 0.16],
            [0.05, 0.22, 0.55],
        ],
        dtype=np.float32,
    )
    hand = one_backward_source_sweep(thermal, alpha, operator)
    exact = solve_scattering_source(
        torch.as_tensor(thermal[None, :]),
        torch.as_tensor(alpha[None, :]),
        torch.as_tensor(operator),
        torch.as_tensor(np.diag(operator).copy()),
        sweeps=1,
    )[0]
    return HandSweepCheckpoint(
        thermal_source=thermal,
        scattering_fraction=alpha,
        lambda_operator=operator,
        hand_source=hand,
        exact_source=exact.numpy(),
    )


def source_sweep_checkpoint() -> SourceSweepCheckpoint:
    """Record the real source after one through eight fixed float32 sweeps."""

    _configure_staged_source()
    import torch

    from payne_zero_synthesis.radiative_transfer import (
        _interpolate_to_transfer_grid,
        integrate_optical_depth,
        solve_scattering_source,
        source_and_alpha,
    )

    window = prepared_window_checkpoint()
    scattering_strength = np.max(
        window.continuum_scattering
        / np.maximum(
            window.continuum_absorption + window.continuum_scattering,
            1.0e-38,
        ),
        axis=0,
    )
    wavelength_index = int(np.argmax(scattering_strength))
    column_mass = torch.as_tensor(window.column_mass, dtype=torch.float64)
    absorption = torch.as_tensor(
        window.continuum_absorption[:, wavelength_index][None, :],
        dtype=torch.float64,
    )
    scattering = torch.as_tensor(
        window.continuum_scattering[:, wavelength_index][None, :],
        dtype=torch.float64,
    )
    planck = torch.as_tensor(
        window.planck_source[:, wavelength_index][None, :],
        dtype=torch.float64,
    )
    zeros = torch.zeros_like(absorption)
    extinction, alpha, thermal = source_and_alpha(
        absorption,
        planck,
        zeros,
        planck,
        scattering,
        zeros,
    )
    optical_depth = integrate_optical_depth(
        column_mass,
        extinction,
        extinction[:, 0] * column_mass[0],
    )
    grid = synthesis_transfer_tables().transfer_optical_depth_grid
    thermal_grid = _interpolate_to_transfer_grid(optical_depth, thermal, grid)
    alpha_grid = _interpolate_to_transfer_grid(optical_depth, alpha, grid).clamp(0.0, 1.0)
    above = grid[None, :] < optical_depth[:, :1]
    thermal_grid = torch.where(above, thermal[:, :1], thermal_grid)
    alpha_grid = torch.where(above, alpha[:, :1], alpha_grid)
    checkpoints = [
        solve_scattering_source(
            thermal_grid,
            alpha_grid,
            synthesis_transfer_tables().mean_intensity_operator,
            synthesis_transfer_tables().mean_intensity_diagonal,
            sweeps=sweeps,
        )
        for sweeps in range(1, 9)
    ]
    return SourceSweepCheckpoint(
        transfer_grid=grid.numpy(),
        thermal_source=thermal_grid[0].numpy(),
        one_sweep_source=checkpoints[0][0].numpy(),
        eight_sweep_source=checkpoints[-1][0].numpy(),
        minimum_after_each_sweep=np.asarray(
            [float(source.min()) for source in checkpoints],
            dtype=np.float64,
        ),
        source_dtype=str(checkpoints[-1].dtype),
    )


def saturated_route_checkpoint() -> SaturatedRouteCheckpoint:
    """Cross the strict ``tau_surface > 20`` route boundary."""

    _configure_staged_source()
    import torch

    from payne_zero_synthesis.radiative_transfer import _solve_flux_rows

    target_surface_tau = torch.linspace(18.0, 22.0, 17, dtype=torch.float64)
    column_mass = torch.as_tensor([1.0, 1.5, 2.4, 4.0, 7.0], dtype=torch.float64)
    extinction = target_surface_tau[:, None].expand(-1, column_mass.numel())
    thermal = torch.as_tensor(
        [0.6, 0.9, 1.4, 2.2, 3.4],
        dtype=torch.float64,
    )[None, :].expand_as(extinction)
    zeros = torch.zeros_like(extinction)
    strict_failure_seen = False
    try:
        _solve_flux_rows(
            extinction,
            thermal,
            zeros,
            thermal,
            zeros,
            zeros,
            column_mass,
            synthesis_transfer_tables(),
            assert_no_saturated_core=True,
        )
    except NotImplementedError:
        strict_failure_seen = True
    flux = _solve_flux_rows(
        extinction,
        thermal,
        zeros,
        thermal,
        zeros,
        zeros,
        column_mass,
        synthesis_transfer_tables(),
        assert_no_saturated_core=False,
    )
    threshold = float(synthesis_transfer_tables().transfer_optical_depth_grid[-1])
    return SaturatedRouteCheckpoint(
        first_layer_optical_depth=target_surface_tau.numpy(),
        eddington_flux_per_frequency=flux.numpy(),
        strict_failure_seen=strict_failure_seen,
        saturated_mask=(target_surface_tau > threshold).numpy(),
        threshold=threshold,
    )


def contribution_checkpoint() -> ContributionCheckpoint:
    """Locate broad continuum and line-center flux contributions in column mass."""

    optical = optical_depth_checkpoint()
    window = prepared_window_checkpoint()
    indices = (optical.continuum_index, optical.line_center_index)
    log_column_mass = np.linspace(
        np.log(optical.column_mass[0]),
        np.log(optical.column_mass[-1]),
        500,
    )
    column_mass = np.exp(log_column_mass)
    mu, weight = np.polynomial.legendre.leggauss(8)
    outward = mu > 0.0
    mu = mu[outward]
    weight = weight[outward]
    contributions = []
    complete_extinction = (
        window.continuum_absorption
        + window.continuum_scattering
        + window.line_mass_absorption_coefficient
    )
    for wavelength_index in indices:
        extinction = np.interp(
            log_column_mass,
            np.log(optical.column_mass),
            complete_extinction[:, wavelength_index],
        )
        source = np.interp(
            log_column_mass,
            np.log(optical.column_mass),
            window.planck_source[:, wavelength_index],
        )
        optical_depth_per_log_mass = extinction * column_mass
        tau = np.empty_like(column_mass)
        tau[0] = extinction[0] * column_mass[0]
        tau[1:] = tau[0] + np.cumsum(
            0.5
            * (
                optical_depth_per_log_mass[1:]
                + optical_depth_per_log_mass[:-1]
            )
            * np.diff(log_column_mass)
        )
        angular = np.zeros_like(tau)
        for mu_value, weight_value in zip(mu, weight, strict=True):
            angular += weight_value * source * np.exp(-tau / mu_value)
        per_log_column_mass = angular * optical_depth_per_log_mass
        contributions.append(
            per_log_column_mass / max(float(np.max(per_log_column_mass)), 1.0e-300)
        )
    continuum, line = contributions
    return ContributionCheckpoint(
        column_mass=column_mass,
        continuum_contribution=continuum,
        line_contribution=line,
        continuum_index=optical.continuum_index,
        line_center_index=optical.line_center_index,
        continuum_peak_column_mass=float(
            column_mass[int(np.argmax(continuum))]
        ),
        line_peak_column_mass=float(column_mass[int(np.argmax(line))]),
    )


def flux_conversion_checkpoint() -> FluxConversionCheckpoint:
    """Check ``H_nu -> F_nu -> F_lambda`` and the cancelling ratio."""

    _configure_staged_source()
    from payne_zero_synthesis.api import (
        FOUR_PI,
        SPEED_OF_LIGHT_NM_S,
        _surface_flux_per_wavelength_nm,
    )

    window = prepared_window_checkpoint()
    indices = np.asarray([10, 60, 110], dtype=np.int64)
    wavelength = window.wavelength_nm[indices]
    total_h = window.eddington_flux_total_per_frequency[indices]
    continuum_h = window.eddington_flux_continuum_per_frequency[indices]
    flux_nu = FOUR_PI * total_h
    flux_lambda = flux_nu * SPEED_OF_LIGHT_NM_S / np.square(wavelength)
    helper_total = _surface_flux_per_wavelength_nm(wavelength, total_h)
    helper_continuum = _surface_flux_per_wavelength_nm(wavelength, continuum_h)
    return FluxConversionCheckpoint(
        wavelength_nm=wavelength.copy(),
        eddington_flux_per_frequency=total_h.copy(),
        flux_per_frequency=flux_nu,
        flux_per_wavelength_nm=flux_lambda,
        helper_flux_per_wavelength_nm=helper_total,
        normalized_before=total_h / continuum_h,
        normalized_after=helper_total / helper_continuum,
    )


def atmosphere_moment_checkpoint(*, deep: bool = True) -> AtmosphereMomentCheckpoint:
    """Run the exact one-frequency atmosphere moment kernel."""

    _configure_staged_source()
    from payne_zero_atmosphere.radiative_transfer import (
        load_radiative_transfer_tables,
    )
    from payne_zero_atmosphere.transfer_kernels import _transfer_moments_compiled

    tables = load_radiative_transfer_tables(
        ATMOSPHERE_TRANSFER_TABLES,
        force_reload=True,
    )
    column_mass = np.logspace(-4.0, 2.0 if deep else -0.1, 12)
    continuum_absorption = np.linspace(0.35, 0.8, column_mass.size)
    continuum_scattering = np.linspace(0.08, 0.16, column_mass.size)
    continuum_source = np.linspace(0.7, 2.4, column_mass.size)
    gross_line = np.linspace(0.12, 0.3, column_mass.size).astype(np.float32)
    stimulated = np.linspace(0.72, 0.93, column_mass.size)
    net_line = gross_line.astype(np.float64) * stimulated
    planck = np.linspace(0.75, 2.6, column_mass.size)
    outputs = [np.empty(column_mass.size, dtype=np.float64) for _ in range(7)]
    surface_second_moment, mapped_layer_count = _transfer_moments_compiled(
        continuum_absorption,
        continuum_source,
        net_line,
        planck,
        continuum_scattering,
        column_mass,
        planck,
        tables.transfer_optical_depth_grid,
        tables.mean_intensity_operator,
        tables.eddington_flux_operator,
        tables.second_moment_weights,
        *outputs,
    )
    (
        optical_depth,
        source,
        eddington_flux,
        mean_intensity,
        mean_intensity_minus_source,
        total_opacity,
        scattering_fraction,
    ) = outputs
    return AtmosphereMomentCheckpoint(
        column_mass=column_mass,
        optical_depth=optical_depth,
        source=source,
        eddington_flux=eddington_flux,
        mean_intensity=mean_intensity,
        mean_intensity_minus_source=mean_intensity_minus_source,
        total_opacity=total_opacity,
        scattering_fraction=scattering_fraction,
        surface_second_moment=float(surface_second_moment),
        mapped_layer_count=int(mapped_layer_count),
        gross_line_mass_absorption_coefficient=gross_line,
        stimulated=stimulated,
        stimulated_line_mass_absorption_coefficient=net_line,
    )


def _run_atmosphere_accumulator(*, parallel: bool) -> tuple[np.ndarray, ...]:
    _configure_staged_source()
    from payne_zero_atmosphere.radiative_transfer import (
        load_radiative_transfer_tables,
    )
    from payne_zero_atmosphere.transfer_kernels import (
        accumulate_transfer_range_compiled,
        accumulate_transfer_range_parallel,
    )

    tables = load_radiative_transfer_tables(
        ATMOSPHERE_TRANSFER_TABLES,
        force_reload=True,
    )
    frequency_count = 2
    layer_count = 8
    frequency_hz = np.asarray([4.9e14, 6.1e14], dtype=np.float64)
    frequency_weights = np.asarray([0.45, 0.55], dtype=np.float64)
    column_mass = np.logspace(-4.0, 1.0, layer_count)
    temperature = np.linspace(4300.0, 6100.0, layer_count)
    planck_all = np.vstack(
        (
            np.linspace(0.65, 2.0, layer_count),
            np.linspace(0.8, 2.5, layer_count),
        )
    )
    stimulated_all = np.vstack(
        (
            np.linspace(0.7, 0.9, layer_count),
            np.linspace(0.75, 0.95, layer_count),
        )
    )
    depth = np.arange(layer_count, dtype=np.float64)[:, None]
    frequency = np.arange(frequency_count, dtype=np.float64)[None, :]
    continuum_absorption = 0.3 + 0.04 * depth + 0.03 * frequency
    continuum_scattering = 0.06 + 0.01 * depth + 0.005 * frequency
    continuum_source = 0.7 + 0.18 * depth + 0.1 * frequency
    gross_line = (0.09 + 0.015 * depth + 0.02 * frequency).astype(np.float32)
    h_over_kt = np.full(layer_count, 1.1e-14, dtype=np.float64)
    output = tuple(np.zeros(layer_count, dtype=np.float64) for _ in range(4))
    surface = np.zeros(1, dtype=np.float64)
    correction = tuple(np.zeros(layer_count, dtype=np.float64) for _ in range(4))
    arguments = (
        0,
        frequency_count,
        frequency_hz,
        frequency_weights,
        planck_all,
        stimulated_all,
        continuum_absorption,
        continuum_scattering,
        continuum_source,
        gross_line,
        column_mass,
        h_over_kt,
        temperature,
        tables.transfer_optical_depth_grid,
        tables.mean_intensity_operator,
        tables.eddington_flux_operator,
        tables.second_moment_weights,
        1.0,
        5100.0,
        frequency_count,
        *output,
        surface,
        *correction,
    )
    if parallel:
        accumulate_transfer_range_parallel(2, *arguments)
    else:
        accumulate_transfer_range_compiled(*arguments)
    return (*output, surface, *correction)


@lru_cache(maxsize=1)
def atmosphere_parallel_checkpoint() -> AtmosphereParallelCheckpoint:
    """Compare serial and fixed two-chunk atmosphere accumulation."""

    names = (
        "rosseland_accumulator",
        "radiation_energy_density",
        "integrated_eddington_flux",
        "radiative_acceleration",
        "surface_radiation_pressure_constant",
        "temperature_correction_heating_derivative",
        "temperature_correction_mean_intensity_minus_source_integral",
        "temperature_correction_integrated_eddington_flux",
        "temperature_correction_diagonal_lambda",
    )
    serial = _run_atmosphere_accumulator(parallel=False)
    parallel = _run_atmosphere_accumulator(parallel=True)
    repeated = _run_atmosphere_accumulator(parallel=True)
    worst_absolute = -1.0
    worst_relative = -1.0
    worst_name = ""
    for name, left, right in zip(names, serial, parallel, strict=True):
        absolute = np.abs(left - right)
        relative = absolute / np.maximum(np.abs(left), 1.0e-300)
        local_absolute = float(np.max(absolute))
        local_relative = float(np.max(relative))
        if local_absolute > worst_absolute:
            worst_absolute = local_absolute
            worst_relative = local_relative
            worst_name = name
    repeatable = all(
        np.array_equal(left, right)
        for left, right in zip(parallel, repeated, strict=True)
    )
    return AtmosphereParallelCheckpoint(
        output_names=names,
        serial_outputs=tuple(array.copy() for array in serial),
        parallel_outputs=tuple(array.copy() for array in parallel),
        repeated_outputs=tuple(array.copy() for array in repeated),
        worst_absolute_difference=worst_absolute,
        worst_relative_difference=worst_relative,
        worst_output_name=worst_name,
        fixed_policy_repeatable=repeatable,
    )
