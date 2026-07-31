"""Executable Chapter 12 checkpoints for radiation and convection.

The chapter-level handoff consumes Chapter 11's exact 80-layer
``OpacityState`` and calls the canonical accumulation/finalization routines.
A deterministic six-layer, 30,000-frequency microfixture remains available
only where a smaller state makes individual deposits, reduction grouping,
restoration, or convection easier to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
from pathlib import Path
import sys

import numpy as np

from book.chapter12_teaching import (
    central_logical_derivative,
    chunk_ledger,
    require_all_or_none,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
TRANSFER_TABLES = (
    REPOSITORY_ROOT / "data/static/atmosphere_tables/radiative_transfer_tables.npz"
)
TRANSFER_TABLES_SHA256 = (
    "d69fcad9e22dd8dd42634e5720df717f0298849d98ee2cd93236009e22391e56"
)
SOURCE_HASHES = {
    "rosseland_mean.py": (
        "91071248fd903e05322b7163d37566e9f894daefc1d7ba018d4850d362f1fc86"
    ),
    "radiative_pressure.py": (
        "c61a256892282d9a0d6cb19714ea5ce6135f1b6f7f573e5761d216d552f321ec"
    ),
    "temperature_correction.py": (
        "67728389ba857511979d0f82ea59f0bf41ee635b8151ae26673dace02b195d21"
    ),
    "convection.py": (
        "9099af3ce97123a88cfee554cefb55b2b47a52085e3cb6cda19e6869e0fef9fd"
    ),
    "transfer_kernels.py": (
        "50e759a085e6aefdb7819a3dbe3ef5e83405834f4b07e0a4de2f3c0e7354d3b9"
    ),
}

LAYER_COUNT = 6
FREQUENCY_COUNT = 30_000
EFFECTIVE_TEMPERATURE = 5778.0
SURFACE_GRAVITY_CGS = 1.0e4
FIXED_CHUNK_COUNT = 2
REDUCTION_OUTPUT_NAMES = (
    "rosseland_accumulator",
    "radiation_energy_density",
    "integrated_eddington_flux",
    "radiative_acceleration",
    "surface_radiation_pressure_constant",
    "absorption_heating_derivative",
    "mean_intensity_minus_source_integral",
    "correction_integrated_eddington_flux",
    "diagonal_lambda_accumulator",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_chapter12_runtime() -> None:
    """Select staged source, immutable tables, and exact source identities."""

    from book.chapter06_runtime import configure_local_data_paths

    configure_local_data_paths()
    staged_root = (REPOSITORY_ROOT / "src").resolve()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("payne_zero_atmosphere"):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            continue
        if not Path(module_path).resolve().is_relative_to(staged_root):
            raise RuntimeError(
                f"{name} resolved outside staged source: {module_path}"
            )
    if _sha256(TRANSFER_TABLES) != TRANSFER_TABLES_SHA256:
        raise RuntimeError("Chapter 12 transfer-table identity changed")
    source_root = REPOSITORY_ROOT / "src/payne_zero_atmosphere"
    for name, expected in SOURCE_HASHES.items():
        if _sha256(source_root / name) != expected:
            raise RuntimeError(f"Chapter 12 staged source changed: {name}")


@dataclass(frozen=True)
class RadiationFixture:
    """Deterministic Chapter 11-shaped input for the Chapter 12 spine."""

    column_mass: np.ndarray
    temperature: np.ndarray
    gas_pressure: np.ndarray
    electron_density: np.ndarray
    wavelength_nm: np.ndarray
    frequency_hz: np.ndarray
    frequency_weights: np.ndarray
    planck_all: np.ndarray
    stimulated_all: np.ndarray
    continuum_absorption: np.ndarray
    continuum_scattering: np.ndarray
    continuum_source: np.ndarray
    line_mass_absorption_coefficient: np.ndarray


@dataclass(frozen=True)
class Chapter11HandoffCheckpoint:
    """Compact identity of Chapter 11's exact pre-transfer state."""

    opacity_state_type: str
    depth_count: int
    frequency_count: int
    continuum_shape: tuple[int, int]
    line_shape: tuple[int, int]
    continuum_dtype: str
    line_dtype: str
    molecules_enabled: bool
    selected_line_count: int
    source_is_chapter11_cache: bool


@dataclass(frozen=True)
class Chapter11FinalizationCheckpoint:
    """Small view of the exact Chapter 11 state after Chapter 12 finalization."""

    finalization_type: str
    correction_type: str
    depth_count: int
    frequency_count: int
    rosseland_opacity: np.ndarray
    rosseland_optical_depth: np.ndarray
    integrated_eddington_flux: np.ndarray
    radiative_acceleration: np.ndarray
    correction_temperature: np.ndarray
    correction_column_mass: np.ndarray
    correction_field_names: tuple[str, ...]
    correction_result_finite: bool
    correction_column_mass_positive: bool
    correction_column_mass_strictly_increasing: bool
    rosseland_aliases_accumulator: bool
    radiative_state_aliases_accumulator: bool
    lookup_entry_count: int
    source_is_chapter11_cache: bool


@dataclass(frozen=True)
class InputCheckpoint:
    layer_count: int
    frequency_count: int
    wavelength_increases: bool
    frequency_decreases: bool
    weights_positive: bool
    planck_shape: tuple[int, int]
    continuum_shape: tuple[int, int]
    line_dtype: str
    transfer_table_sha256: str


@dataclass(frozen=True)
class OneFrequencyCheckpoint:
    frequency_index: int
    frequency_hz: float
    output_names: tuple[str, ...]
    compiled_outputs: tuple[np.ndarray, ...]
    helper_outputs: tuple[np.ndarray, ...]
    maximum_absolute_difference: float
    maximum_relative_difference: float
    post_guard_identity_holds: bool


@dataclass(frozen=True)
class ReductionCheckpoint:
    frequency_count: int
    layer_count: int
    chunk_count: int
    bounds: np.ndarray
    private_bytes: int
    output_names: tuple[str, ...]
    one_chunk_outputs: tuple[np.ndarray, ...]
    fixed_chunk_outputs: tuple[np.ndarray, ...]
    repeated_outputs: tuple[np.ndarray, ...]
    fixed_policy_repeatable: bool
    maximum_absolute_difference: float
    maximum_relative_difference: float


@dataclass(frozen=True)
class PersistenceCheckpoint:
    reset_names: tuple[str, ...]
    reset_arrays_zero: bool
    previous_correction_preserved: bool
    table_identity_preserved: bool
    table_entry_count_preserved: bool
    radiative_mode1_preserves_final_pressures: bool


@dataclass(frozen=True)
class FinalizationCheckpoint:
    rosseland_opacity: np.ndarray
    rosseland_optical_depth: np.ndarray
    integrated_eddington_flux: np.ndarray
    radiation_energy_density: np.ndarray
    radiative_acceleration: np.ndarray
    integrated_radiation_pressure: np.ndarray
    absolute_radiation_pressure: np.ndarray
    surface_radiation_pressure_constant: float
    correction_field_names: tuple[str, ...]
    correction_result_finite: bool
    rosseland_aliases_accumulator: bool
    radiative_state_aliases_accumulator: bool
    lookup_entry_count: int


@dataclass(frozen=True)
class RestoreCheckpoint:
    sample_field_names: tuple[str, ...]
    temperature_restored: bool
    gas_pressure_restored: bool
    electron_density_restored: bool
    total_nuclei_density_restored: bool
    mass_density_restored: bool
    populations_restored: bool
    cache_restored: bool
    charge_square_density_restored: bool
    zero_energy_replaced: bool
    density_temperature_derivative: np.ndarray
    energy_temperature_derivative: np.ndarray


@dataclass(frozen=True)
class ConvectionCheckpoint:
    result_field_names: tuple[str, ...]
    geometric_depth_below_surface_km: np.ndarray
    logarithmic_gradient: np.ndarray
    adiabatic_gradient: np.ndarray
    heat_capacity: np.ndarray
    raw_convective_flux: np.ndarray
    returned_convective_flux: np.ndarray
    standard_suppressed_flux: np.ndarray
    convective_velocity: np.ndarray
    first_two_suppressed: bool
    standard_six_layer_fixture_fully_suppressed: bool
    disabled_flux: np.ndarray
    disabled_velocity: np.ndarray
    disabled_can_be_nonzero: bool


@lru_cache(maxsize=1)
def chapter11_opacity_state():
    """Return Chapter 11's exact cached ``OpacityState`` without copying slabs."""

    configure_chapter12_runtime()
    from book.chapter11_runtime import (
        configure_chapter11_runtime,
        full_opacity_state,
    )

    configure_chapter11_runtime()
    return full_opacity_state()


def chapter11_handoff_checkpoint() -> Chapter11HandoffCheckpoint:
    """Expose the exact Chapter 11 handoff without copying its large arrays."""

    from book.chapter11_runtime import full_opacity_state

    state = chapter11_opacity_state()
    atmosphere = state.population_state.setup.atmosphere
    return Chapter11HandoffCheckpoint(
        opacity_state_type=type(state).__name__,
        depth_count=atmosphere.layers,
        frequency_count=int(state.opacity_frequency_hz.size),
        continuum_shape=tuple(state.continuum_absorption.shape),
        line_shape=tuple(
            state.line_opacity.line_mass_absorption_coefficient.shape
        ),
        continuum_dtype=str(state.continuum_absorption.dtype),
        line_dtype=str(
            state.line_opacity.line_mass_absorption_coefficient.dtype
        ),
        molecules_enabled=state.population_state.molecular_state is not None,
        selected_line_count=(
            0
            if state.selected_line_catalog is None
            else int(state.selected_line_catalog.line_count)
        ),
        source_is_chapter11_cache=state is full_opacity_state(),
    )


@lru_cache(maxsize=1)
def chapter11_iteration_finalization():
    """Run the exact Chapter 11 state through Chapter 12's canonical calls."""

    configure_chapter12_runtime()
    from payne_zero_atmosphere.runner import (
        accumulate_transfer_state,
        finalize_transfer_state,
    )

    transfer = accumulate_transfer_state(chapter11_opacity_state())
    return finalize_transfer_state(
        transfer,
        iteration_index=1,
        convection_enabled=False,
    )


@lru_cache(maxsize=1)
def chapter11_finalization_checkpoint() -> Chapter11FinalizationCheckpoint:
    """Return a compact, copied view of the exact composed finalization."""

    from book.chapter11_runtime import full_opacity_state

    finalization = chapter11_iteration_finalization()
    transfer = finalization.transfer_accumulation
    correction = finalization.temperature_correction_result
    pressure = finalization.radiative_pressure_state
    correction_arrays = [
        np.asarray(getattr(correction, field.name))
        for field in fields(correction)
        if isinstance(getattr(correction, field.name), np.ndarray)
    ]
    return Chapter11FinalizationCheckpoint(
        finalization_type=type(finalization).__name__,
        correction_type=type(correction).__name__,
        depth_count=int(finalization.rosseland_opacity.size),
        frequency_count=int(
            transfer.opacity_state.opacity_frequency_hz.size
        ),
        rosseland_opacity=finalization.rosseland_opacity.copy(),
        rosseland_optical_depth=finalization.rosseland_optical_depth.copy(),
        integrated_eddington_flux=(
            pressure.integrated_eddington_flux.copy()
        ),
        radiative_acceleration=pressure.radiative_acceleration.copy(),
        correction_temperature=correction.temperature.copy(),
        correction_column_mass=correction.column_mass.copy(),
        correction_field_names=tuple(field.name for field in fields(correction)),
        correction_result_finite=all(
            np.all(np.isfinite(array)) for array in correction_arrays
        ),
        correction_column_mass_positive=bool(
            np.all(correction.column_mass > 0.0)
        ),
        correction_column_mass_strictly_increasing=bool(
            np.all(np.diff(correction.column_mass) > 0.0)
        ),
        rosseland_aliases_accumulator=(
            finalization.rosseland_opacity
            is transfer.rosseland_accumulator
        ),
        radiative_state_aliases_accumulator=(
            finalization.radiative_pressure_state
            is transfer.radiative_pressure_state
        ),
        lookup_entry_count=(
            transfer.temperature_correction_state.rosseland_opacity_table.entry_count
        ),
        source_is_chapter11_cache=(
            transfer.opacity_state is full_opacity_state()
        ),
    )


@lru_cache(maxsize=1)
def radiation_fixture() -> RadiationFixture:
    """Build a deterministic 30,000-frequency, six-layer transfer input."""

    configure_chapter12_runtime()
    from payne_zero_atmosphere.continuum_opacity import build_opacity_sampling_grid

    wavelength_nm, frequency_weights = build_opacity_sampling_grid(
        EFFECTIVE_TEMPERATURE
    )
    frequency_hz = 2.99792458e17 / wavelength_nm
    column_mass = np.logspace(-4.0, 2.0, LAYER_COUNT, dtype=np.float64)
    temperature = np.asarray(
        [4300.0, 4700.0, 5200.0, 5800.0, 6400.0, 7000.0],
        dtype=np.float64,
    )
    gas_pressure = SURFACE_GRAVITY_CGS * column_mass + 250.0
    electron_density = (
        gas_pressure
        / np.maximum(temperature * 1.38054e-16, 1.0e-300)
        * np.linspace(3.0e-5, 8.0e-5, LAYER_COUNT)
    )
    h_over_kt = 6.6256e-27 / np.maximum(
        temperature * 1.38054e-16,
        1.0e-300,
    )
    exponential = np.exp(-frequency_hz[:, None] * h_over_kt[None, :])
    stimulated = np.maximum(1.0 - exponential, 1.0e-300)
    planck = (
        1.47439e-2
        * (frequency_hz[:, None] / 1.0e15) ** 3
        * exponential
        / stimulated
    )

    depth = np.arange(LAYER_COUNT, dtype=np.float64)[:, None]
    phase = np.linspace(0.0, 12.0 * np.pi, FREQUENCY_COUNT)[None, :]
    continuum_absorption = (
        0.18 + 0.075 * depth + 0.025 * (1.0 + np.sin(phase))
    )
    continuum_scattering = (
        0.012 + 0.003 * depth + 0.002 * (1.0 + np.cos(phase * 0.5))
    )
    continuum_source = np.ascontiguousarray(planck.T, dtype=np.float64)

    index = np.arange(FREQUENCY_COUNT, dtype=np.float64)[None, :]
    line_profile = (
        0.10 * np.exp(-0.5 * ((index - 7_200.0) / 120.0) ** 2)
        + 0.18 * np.exp(-0.5 * ((index - 15_400.0) / 260.0) ** 2)
        + 0.07 * np.exp(-0.5 * ((index - 24_600.0) / 90.0) ** 2)
    )
    line_opacity = np.asarray(
        (1.0 + 0.15 * depth) * line_profile,
        dtype=np.float32,
    )
    return RadiationFixture(
        column_mass=column_mass,
        temperature=temperature,
        gas_pressure=gas_pressure,
        electron_density=electron_density,
        wavelength_nm=np.asarray(wavelength_nm, dtype=np.float64),
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
        frequency_weights=np.asarray(frequency_weights, dtype=np.float64),
        planck_all=np.asarray(planck, dtype=np.float64),
        stimulated_all=np.asarray(stimulated, dtype=np.float64),
        continuum_absorption=np.ascontiguousarray(
            continuum_absorption,
            dtype=np.float64,
        ),
        continuum_scattering=np.ascontiguousarray(
            continuum_scattering,
            dtype=np.float64,
        ),
        continuum_source=continuum_source,
        line_mass_absorption_coefficient=np.ascontiguousarray(
            line_opacity,
            dtype=np.float32,
        ),
    )


def input_checkpoint() -> InputCheckpoint:
    fixture = radiation_fixture()
    return InputCheckpoint(
        layer_count=fixture.column_mass.size,
        frequency_count=fixture.frequency_hz.size,
        wavelength_increases=bool(np.all(np.diff(fixture.wavelength_nm) > 0.0)),
        frequency_decreases=bool(np.all(np.diff(fixture.frequency_hz) < 0.0)),
        weights_positive=bool(np.all(fixture.frequency_weights > 0.0)),
        planck_shape=fixture.planck_all.shape,
        continuum_shape=fixture.continuum_absorption.shape,
        line_dtype=str(fixture.line_mass_absorption_coefficient.dtype),
        transfer_table_sha256=_sha256(TRANSFER_TABLES),
    )


def _model_and_runtime():
    configure_chapter12_runtime()
    from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere
    from payne_zero_atmosphere.runtime_state import (
        build_runtime_state,
        update_charge_square_density,
    )

    fixture = radiation_fixture()
    zeros = np.zeros(LAYER_COUNT, dtype=np.float64)
    atmosphere = ModelAtmosphere(
        column_mass=fixture.column_mass.copy(),
        temperature=fixture.temperature.copy(),
        gas_pressure=fixture.gas_pressure.copy(),
        electron_density=fixture.electron_density.copy(),
        rosseland_opacity=np.full(LAYER_COUNT, 0.35, dtype=np.float64),
        radiative_acceleration=zeros.copy(),
        microturbulence=np.full(LAYER_COUNT, 2.0e5, dtype=np.float64),
        convective_flux=zeros.copy(),
        convective_velocity=zeros.copy(),
        metadata={
            "effective_temperature": str(EFFECTIVE_TEMPERATURE),
            "log_surface_gravity": "4.0",
        },
    )
    runtime_state = build_runtime_state(atmosphere)
    update_charge_square_density(
        thermal_energy_erg=atmosphere.thermal_energy_erg,
        state=runtime_state,
    )
    return atmosphere, runtime_state


def make_opacity_state():
    """Return a fresh exact ``OpacityState`` around the deterministic slabs."""

    configure_chapter12_runtime()
    from payne_zero_atmosphere.continuum_opacity import (
        build_continuum_atmosphere_state,
        create_rosseland_opacity_table,
    )
    from payne_zero_atmosphere.line_opacity import LineOpacityState
    from payne_zero_atmosphere.run_setup import (
        ConvectionSettings,
        RunSetup,
        TurbulenceSettings,
    )
    from payne_zero_atmosphere.runner import AtmospherePopulationState, OpacityState

    fixture = radiation_fixture()
    atmosphere, runtime_state = _model_and_runtime()
    setup = RunSetup(
        atmosphere=atmosphere,
        iterations=1,
        enable_convergence_stop=False,
        minimum_iterations_before_convergence=3,
        required_consecutive_converged_iterations=1,
        maximum_deep_layer_relative_temperature_change=5.0e-4,
        maximum_all_layer_relative_temperature_change=None,
        surface_gravity_cgs=SURFACE_GRAVITY_CGS,
        opacity_flags=[0] * 20,
        molecules_enabled=False,
        pressure_iteration_enabled=True,
        convection=ConvectionSettings(
            enabled=True,
            mixing_length=1.25,
            overshoot_weight=0.0,
            zero_top_layer_count=0,
        ),
        turbulence=TurbulenceSettings(
            enabled=False,
            density_coefficient=0.0,
            density_power=0.0,
            sound_speed_fraction=0.0,
            constant_velocity_km_s=0.0,
        ),
        surface_radiation_pressure_constant=0.0,
        effective_temperature=EFFECTIVE_TEMPERATURE,
        log_surface_gravity=4.0,
        standard_rosseland_optical_depth=np.logspace(
            -6.875,
            3.0,
            LAYER_COUNT,
        ),
    )
    population = AtmospherePopulationState(
        setup=setup,
        runtime_state=runtime_state,
        fractional_doppler_widths=np.zeros((LAYER_COUNT, 1), dtype=np.float64),
        partition_normalized_population_over_mass_density_and_fractional_doppler_width=np.zeros(
            (LAYER_COUNT, 1),
            dtype=np.float64,
        ),
        temperature_iteration_cache={},
        molecular_state=None,
    )
    continuum_atmosphere = build_continuum_atmosphere_state(
        atmosphere,
        runtime_state,
    )
    return OpacityState(
        population_state=population,
        continuum_atmosphere=continuum_atmosphere,
        opacity_wavelength_grid_nm=fixture.wavelength_nm,
        opacity_frequency_hz=fixture.frequency_hz,
        frequency_weights=fixture.frequency_weights,
        active_continuum_indices=np.arange(343, dtype=np.int64),
        active_continuum_frequency_hz=fixture.frequency_hz[:343],
        continuum_absorption=fixture.continuum_absorption,
        continuum_scattering=fixture.continuum_scattering,
        continuum_source=fixture.continuum_source,
        continuum_line_selection_threshold=np.zeros(
            (LAYER_COUNT, 344),
            dtype=np.float32,
        ),
        continuum_reference_wavelength_nm=fixture.wavelength_nm[:344],
        wavelength_bin_edges=np.arange(344, dtype=np.int64),
        line_opacity=LineOpacityState(
            line_mass_absorption_coefficient=fixture.line_mass_absorption_coefficient,
            selected_line_count=3,
        ),
        rosseland_table=create_rosseland_opacity_table(LAYER_COUNT),
        selected_line_catalog=None,
        transition_line_catalog=None,
    )


def _run_low_level(chunk_count: int) -> tuple[np.ndarray, ...]:
    configure_chapter12_runtime()
    from payne_zero_atmosphere.radiative_transfer import (
        load_radiative_transfer_tables,
    )
    from payne_zero_atmosphere.transfer_kernels import (
        accumulate_transfer_range_parallel,
    )

    fixture = radiation_fixture()
    tables = load_radiative_transfer_tables(force_reload=True)
    outputs = [np.zeros(LAYER_COUNT, dtype=np.float64) for _ in range(4)]
    surface = np.zeros(1, dtype=np.float64)
    correction = [np.zeros(LAYER_COUNT, dtype=np.float64) for _ in range(4)]
    target = 5.6697e-5 / 12.5664 * EFFECTIVE_TEMPERATURE**4
    h_over_kt = 6.6256e-27 / np.maximum(
        fixture.temperature * 1.38054e-16,
        1.0e-300,
    )
    accumulate_transfer_range_parallel(
        int(chunk_count),
        0,
        FREQUENCY_COUNT,
        fixture.frequency_hz,
        fixture.frequency_weights,
        fixture.planck_all,
        fixture.stimulated_all,
        fixture.continuum_absorption,
        fixture.continuum_scattering,
        fixture.continuum_source,
        fixture.line_mass_absorption_coefficient,
        fixture.column_mass,
        h_over_kt,
        fixture.temperature,
        tables.transfer_optical_depth_grid,
        tables.mean_intensity_operator,
        tables.eddington_flux_operator,
        tables.second_moment_weights,
        target,
        EFFECTIVE_TEMPERATURE,
        FREQUENCY_COUNT,
        *outputs,
        surface,
        *correction,
    )
    return (*outputs, surface, *correction)


@lru_cache(maxsize=1)
def reduction_checkpoint() -> ReductionCheckpoint:
    one = _run_low_level(1)
    fixed = _run_low_level(FIXED_CHUNK_COUNT)
    repeated = _run_low_level(FIXED_CHUNK_COUNT)
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for left, right in zip(one, fixed, strict=True):
        absolute = np.abs(left - right)
        relative = absolute / np.maximum(np.abs(left), 1.0e-300)
        maximum_absolute = max(maximum_absolute, float(np.max(absolute)))
        maximum_relative = max(maximum_relative, float(np.max(relative)))
    ledger = chunk_ledger(
        start=0,
        stop=FREQUENCY_COUNT,
        chunk_count=FIXED_CHUNK_COUNT,
        layer_count=LAYER_COUNT,
    )
    repeatable = all(
        np.array_equal(left, right)
        for left, right in zip(fixed, repeated, strict=True)
    )
    return ReductionCheckpoint(
        frequency_count=FREQUENCY_COUNT,
        layer_count=LAYER_COUNT,
        chunk_count=FIXED_CHUNK_COUNT,
        bounds=ledger.bounds,
        private_bytes=ledger.private_bytes,
        output_names=REDUCTION_OUTPUT_NAMES,
        one_chunk_outputs=tuple(value.copy() for value in one),
        fixed_chunk_outputs=tuple(value.copy() for value in fixed),
        repeated_outputs=tuple(value.copy() for value in repeated),
        fixed_policy_repeatable=repeatable,
        maximum_absolute_difference=maximum_absolute,
        maximum_relative_difference=maximum_relative,
    )


@lru_cache(maxsize=1)
def one_frequency_checkpoint() -> OneFrequencyCheckpoint:
    configure_chapter12_runtime()
    from payne_zero_atmosphere.runner import (
        _planck_source_and_stimulated_emission,
    )
    from payne_zero_atmosphere.radiative_pressure import (
        accumulate_radiative_pressure,
        initialize_radiative_pressure_state,
    )
    from payne_zero_atmosphere.radiative_transfer import (
        load_radiative_transfer_tables,
    )
    from payne_zero_atmosphere.rosseland_mean import rosseland_mean_step
    from payne_zero_atmosphere.temperature_correction import (
        apply_temperature_correction,
        initialize_temperature_correction_state,
    )
    from payne_zero_atmosphere.transfer_kernels import (
        _transfer_moments_compiled,
        accumulate_transfer_range_compiled,
    )

    fixture = radiation_fixture()
    tables = load_radiative_transfer_tables(force_reload=True)
    index = 15_400
    frequency = float(fixture.frequency_hz[index])
    weight = float(fixture.frequency_weights[index])
    h_over_kt = 6.6256e-27 / np.maximum(
        fixture.temperature * 1.38054e-16,
        1.0e-300,
    )
    scalar_planck, scalar_stimulated = _planck_source_and_stimulated_emission(
        frequency_hz=frequency,
        h_over_kt=h_over_kt,
    )
    np.testing.assert_array_equal(scalar_planck, fixture.planck_all[index])
    np.testing.assert_array_equal(scalar_stimulated, fixture.stimulated_all[index])

    moment_outputs = [
        np.empty(LAYER_COUNT, dtype=np.float64) for _ in range(7)
    ]
    net_line = (
        fixture.line_mass_absorption_coefficient[:, index].astype(np.float64)
        * scalar_stimulated
    )
    surface_second_moment, _ = _transfer_moments_compiled(
        fixture.continuum_absorption[:, index],
        fixture.continuum_source[:, index],
        net_line,
        scalar_planck,
        fixture.continuum_scattering[:, index],
        fixture.column_mass,
        scalar_planck,
        tables.transfer_optical_depth_grid,
        tables.mean_intensity_operator,
        tables.eddington_flux_operator,
        tables.second_moment_weights,
        *moment_outputs,
    )
    (
        optical_depth,
        source,
        eddington_flux,
        mean_intensity,
        mean_minus_source,
        total_opacity,
        scattering_fraction,
    ) = moment_outputs
    if np.any(eddington_flux < 0.0):
        eddington_flux[:] = np.maximum(eddington_flux, 1.0e-99)
        mean_intensity[:] = np.maximum(mean_intensity, 1.0e-99)
        source[:] = np.maximum(source, 1.0e-99)

    target = 5.6697e-5 / 12.5664 * EFFECTIVE_TEMPERATURE**4
    rosseland = np.zeros(LAYER_COUNT, dtype=np.float64)
    rosseland, _ = rosseland_mean_step(
        rosseland,
        mode=2,
        frequency_weight=weight,
        planck_source=scalar_planck,
        frequency_hz=frequency,
        h_over_kt=h_over_kt,
        temperature_k=fixture.temperature,
        stimulated_emission=scalar_stimulated,
        total_opacity=total_opacity,
        frequency_count=FREQUENCY_COUNT,
        column_mass=fixture.column_mass,
    )
    pressure = initialize_radiative_pressure_state(LAYER_COUNT)
    accumulate_radiative_pressure(
        pressure,
        mode=2,
        frequency_weight=weight,
        total_opacity=total_opacity,
        monochromatic_eddington_flux=eddington_flux,
        mean_intensity=mean_intensity,
        surface_second_moment=surface_second_moment,
        target_integrated_eddington_flux=target,
        column_mass=fixture.column_mass,
    )
    correction = initialize_temperature_correction_state(LAYER_COUNT)
    apply_temperature_correction(
        correction,
        mode=2,
        frequency_weight=weight,
        column_mass=fixture.column_mass,
        total_opacity=total_opacity,
        monochromatic_eddington_flux=eddington_flux,
        mean_intensity_minus_source=mean_minus_source,
        monochromatic_optical_depth=optical_depth,
        planck_source=scalar_planck,
        frequency_hz=frequency,
        h_over_kt=h_over_kt,
        temperature_k=fixture.temperature,
        stimulated_emission=scalar_stimulated,
        scattering_fraction=scattering_fraction,
        target_integrated_eddington_flux=target,
        effective_temperature=EFFECTIVE_TEMPERATURE,
        frequency_count=FREQUENCY_COUNT,
    )
    helper = (
        rosseland,
        pressure.radiation_energy_density,
        pressure.integrated_eddington_flux,
        pressure.radiative_acceleration,
        np.asarray([pressure.surface_radiation_pressure_constant]),
        correction.absorption_heating_derivative,
        correction.mean_intensity_minus_source_integral,
        correction.integrated_eddington_flux,
        correction.diagonal_lambda_accumulator,
    )

    compiled_arrays = [
        np.zeros(LAYER_COUNT, dtype=np.float64) for _ in range(4)
    ]
    compiled_surface = np.zeros(1, dtype=np.float64)
    compiled_correction = [
        np.zeros(LAYER_COUNT, dtype=np.float64) for _ in range(4)
    ]
    accumulate_transfer_range_compiled(
        index,
        index + 1,
        fixture.frequency_hz,
        fixture.frequency_weights,
        fixture.planck_all,
        fixture.stimulated_all,
        fixture.continuum_absorption,
        fixture.continuum_scattering,
        fixture.continuum_source,
        fixture.line_mass_absorption_coefficient,
        fixture.column_mass,
        h_over_kt,
        fixture.temperature,
        tables.transfer_optical_depth_grid,
        tables.mean_intensity_operator,
        tables.eddington_flux_operator,
        tables.second_moment_weights,
        target,
        EFFECTIVE_TEMPERATURE,
        FREQUENCY_COUNT,
        *compiled_arrays,
        compiled_surface,
        *compiled_correction,
    )
    compiled = (*compiled_arrays, compiled_surface, *compiled_correction)
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for left, right in zip(compiled, helper, strict=True):
        absolute = np.abs(left - right)
        relative = absolute / np.maximum(np.abs(left), 1.0e-300)
        maximum_absolute = max(maximum_absolute, float(np.max(absolute)))
        maximum_relative = max(maximum_relative, float(np.max(relative)))
    return OneFrequencyCheckpoint(
        frequency_index=index,
        frequency_hz=frequency,
        output_names=REDUCTION_OUTPUT_NAMES,
        compiled_outputs=tuple(value.copy() for value in compiled),
        helper_outputs=tuple(value.copy() for value in helper),
        maximum_absolute_difference=maximum_absolute,
        maximum_relative_difference=maximum_relative,
        post_guard_identity_holds=bool(
            np.array_equal(mean_intensity, source + mean_minus_source)
        ),
    )


def persistence_checkpoint() -> PersistenceCheckpoint:
    configure_chapter12_runtime()
    from payne_zero_atmosphere.radiative_pressure import (
        accumulate_radiative_pressure,
        initialize_radiative_pressure_state,
    )
    from payne_zero_atmosphere.temperature_correction import (
        apply_temperature_correction,
        ingest_temperature_correction_rosseland_table,
        initialize_temperature_correction_state,
    )

    fixture = radiation_fixture()
    state = initialize_temperature_correction_state(LAYER_COUNT)
    state.mean_intensity_minus_source_integral[:] = 1.0
    state.absorption_heating_derivative[:] = 2.0
    state.diagonal_lambda_accumulator[:] = 3.0
    state.integrated_eddington_flux[:] = 4.0
    state.previous_temperature_correction[:] = np.arange(LAYER_COUNT)
    ingest_temperature_correction_rosseland_table(
        state,
        temperature_k=fixture.temperature,
        gas_pressure=fixture.gas_pressure,
        rosseland_opacity=np.full(LAYER_COUNT, 0.4),
    )
    previous = state.previous_temperature_correction.copy()
    table = state.rosseland_opacity_table
    entry_count = table.entry_count
    zeros = np.zeros(LAYER_COUNT, dtype=np.float64)
    ones = np.ones(LAYER_COUNT, dtype=np.float64)
    apply_temperature_correction(
        state,
        mode=1,
        frequency_weight=0.0,
        column_mass=fixture.column_mass,
        total_opacity=ones,
        monochromatic_eddington_flux=zeros,
        mean_intensity_minus_source=zeros,
        monochromatic_optical_depth=zeros,
        planck_source=zeros,
        frequency_hz=0.0,
        h_over_kt=ones,
        temperature_k=fixture.temperature,
        stimulated_emission=ones,
        scattering_fraction=zeros,
        target_integrated_eddington_flux=1.0,
        effective_temperature=EFFECTIVE_TEMPERATURE,
        frequency_count=FREQUENCY_COUNT,
    )

    pressure = initialize_radiative_pressure_state(LAYER_COUNT)
    pressure.integrated_radiation_pressure[:] = 7.0
    pressure.absolute_radiation_pressure[:] = 8.0
    accumulate_radiative_pressure(
        pressure,
        mode=1,
        frequency_weight=0.0,
        total_opacity=ones,
        monochromatic_eddington_flux=zeros,
        mean_intensity=zeros,
        surface_second_moment=0.0,
        target_integrated_eddington_flux=1.0,
        column_mass=fixture.column_mass,
    )
    reset_arrays = (
        state.mean_intensity_minus_source_integral,
        state.absorption_heating_derivative,
        state.diagonal_lambda_accumulator,
        state.integrated_eddington_flux,
    )
    return PersistenceCheckpoint(
        reset_names=(
            "mean_intensity_minus_source_integral",
            "absorption_heating_derivative",
            "diagonal_lambda_accumulator",
            "integrated_eddington_flux",
        ),
        reset_arrays_zero=all(not np.any(array) for array in reset_arrays),
        previous_correction_preserved=bool(
            np.array_equal(previous, state.previous_temperature_correction)
        ),
        table_identity_preserved=state.rosseland_opacity_table is table,
        table_entry_count_preserved=table.entry_count == entry_count,
        radiative_mode1_preserves_final_pressures=bool(
            np.all(pressure.integrated_radiation_pressure == 7.0)
            and np.all(pressure.absolute_radiation_pressure == 8.0)
        ),
    )


@lru_cache(maxsize=1)
def finalization_checkpoint() -> FinalizationCheckpoint:
    configure_chapter12_runtime()
    from payne_zero_atmosphere.runner import (
        accumulate_transfer_state,
        finalize_transfer_state,
    )

    transfer = accumulate_transfer_state(make_opacity_state())
    finalization = finalize_transfer_state(
        transfer,
        iteration_index=1,
        convection_enabled=False,
    )
    pressure = finalization.radiative_pressure_state
    correction = finalization.temperature_correction_result
    correction_arrays = [
        np.asarray(getattr(correction, field.name))
        for field in fields(correction)
        if isinstance(getattr(correction, field.name), np.ndarray)
    ]
    return FinalizationCheckpoint(
        rosseland_opacity=finalization.rosseland_opacity.copy(),
        rosseland_optical_depth=finalization.rosseland_optical_depth.copy(),
        integrated_eddington_flux=pressure.integrated_eddington_flux.copy(),
        radiation_energy_density=pressure.radiation_energy_density.copy(),
        radiative_acceleration=pressure.radiative_acceleration.copy(),
        integrated_radiation_pressure=pressure.integrated_radiation_pressure.copy(),
        absolute_radiation_pressure=pressure.absolute_radiation_pressure.copy(),
        surface_radiation_pressure_constant=float(
            pressure.surface_radiation_pressure_constant
        ),
        correction_field_names=tuple(field.name for field in fields(correction)),
        correction_result_finite=all(
            np.all(np.isfinite(array)) for array in correction_arrays
        ),
        rosseland_aliases_accumulator=(
            finalization.rosseland_opacity
            is transfer.rosseland_accumulator
        ),
        radiative_state_aliases_accumulator=(
            finalization.radiative_pressure_state
            is transfer.radiative_pressure_state
        ),
        lookup_entry_count=(
            transfer.temperature_correction_state.rosseland_opacity_table.entry_count
        ),
    )


@lru_cache(maxsize=1)
def restore_checkpoint() -> RestoreCheckpoint:
    configure_chapter12_runtime()
    from payne_zero_atmosphere.runner import (
        compute_convection_finite_difference_samples,
    )

    atmosphere, runtime_state = _model_and_runtime()
    cache: dict[str, int] = {}
    before = {
        "temperature": atmosphere.temperature.copy(),
        "gas_pressure": runtime_state.gas_pressure.copy(),
        "electron_density": runtime_state.electron_density.copy(),
        "total_nuclei": runtime_state.total_nuclei_number_density.copy(),
        "mass_density": runtime_state.mass_density.copy(),
        "populations": runtime_state.ion_stage_populations_by_packed_slot.copy(),
        "normalized": (
            runtime_state.partition_normalized_populations_by_packed_slot.copy()
        ),
        "charge_square": runtime_state.charge_square_density.copy(),
        "energy": runtime_state.specific_internal_energy.copy(),
        "cache": dict(cache),
    }
    final = finalization_checkpoint()
    samples = compute_convection_finite_difference_samples(
        atmosphere=atmosphere,
        runtime_state=runtime_state,
        absolute_radiation_pressure=final.absolute_radiation_pressure,
        rosseland_optical_depth=final.rosseland_optical_depth,
        temperature_iteration_seed=10,
        temperature_iteration_cache=cache,
        molecules_enabled=False,
        molecular_state=None,
    )
    density_temperature_derivative = central_logical_derivative(
        samples.density_plus_temperature,
        samples.density_minus_temperature,
        atmosphere.temperature,
    )
    energy_temperature_derivative = central_logical_derivative(
        samples.specific_internal_energy_plus_temperature,
        samples.specific_internal_energy_minus_temperature,
        atmosphere.temperature,
    )
    return RestoreCheckpoint(
        sample_field_names=tuple(field.name for field in fields(samples)),
        temperature_restored=np.array_equal(
            before["temperature"],
            atmosphere.temperature,
        ),
        gas_pressure_restored=np.array_equal(
            before["gas_pressure"],
            runtime_state.gas_pressure,
        ),
        electron_density_restored=np.array_equal(
            before["electron_density"],
            runtime_state.electron_density,
        ),
        total_nuclei_density_restored=np.array_equal(
            before["total_nuclei"],
            runtime_state.total_nuclei_number_density,
        ),
        mass_density_restored=np.array_equal(
            before["mass_density"],
            runtime_state.mass_density,
        ),
        populations_restored=bool(
            np.array_equal(
                before["populations"],
                runtime_state.ion_stage_populations_by_packed_slot,
            )
            and np.array_equal(
                before["normalized"],
                runtime_state.partition_normalized_populations_by_packed_slot,
            )
        ),
        cache_restored=cache == before["cache"],
        charge_square_density_restored=np.array_equal(
            before["charge_square"],
            runtime_state.charge_square_density,
        ),
        zero_energy_replaced=bool(
            not np.any(before["energy"])
            and np.any(runtime_state.specific_internal_energy)
        ),
        density_temperature_derivative=density_temperature_derivative,
        energy_temperature_derivative=energy_temperature_derivative,
    )


@lru_cache(maxsize=1)
def convection_checkpoint() -> ConvectionCheckpoint:
    configure_chapter12_runtime()
    from payne_zero_atmosphere.convection import (
        compute_convection,
        compute_disabled_convection_diagnostics,
    )
    from payne_zero_atmosphere.continuum_opacity import (
        create_rosseland_opacity_table,
        ingest_rosseland_opacity_table,
    )
    from payne_zero_atmosphere.runner import (
        compute_convection_finite_difference_samples,
    )

    atmosphere, runtime_state = _model_and_runtime()
    final = finalization_checkpoint()
    table = create_rosseland_opacity_table(LAYER_COUNT)
    ingest_rosseland_opacity_table(
        table,
        temperature_k=atmosphere.temperature,
        gas_pressure=runtime_state.gas_pressure,
        rosseland_opacity=final.rosseland_opacity,
    )
    samples = compute_convection_finite_difference_samples(
        atmosphere=atmosphere,
        runtime_state=runtime_state,
        absolute_radiation_pressure=final.absolute_radiation_pressure,
        rosseland_optical_depth=final.rosseland_optical_depth,
        temperature_iteration_seed=20,
        temperature_iteration_cache={},
    )
    fd_values = tuple(getattr(samples, field.name) for field in fields(samples))
    assert require_all_or_none(*fd_values)
    total_pressure = SURFACE_GRAVITY_CGS * atmosphere.column_mass
    target = 5.6697e-5 / 12.5664 * EFFECTIVE_TEMPERATURE**4
    keywords = dict(
        rosseland_table=table,
        column_mass=atmosphere.column_mass,
        rosseland_optical_depth=final.rosseland_optical_depth,
        temperature_k=atmosphere.temperature,
        gas_pressure=runtime_state.gas_pressure,
        mass_density=runtime_state.mass_density,
        rosseland_opacity=final.rosseland_opacity,
        microturbulence=atmosphere.microturbulence,
        absolute_radiation_pressure=final.absolute_radiation_pressure,
        total_pressure=total_pressure,
        surface_gravity_cgs=SURFACE_GRAVITY_CGS,
        target_integrated_eddington_flux=target,
        mixing_length=1.25,
        overshoot_weight=0.0,
        convection_enabled=True,
        specific_internal_energy_plus_temperature=(
            samples.specific_internal_energy_plus_temperature
        ),
        specific_internal_energy_minus_temperature=(
            samples.specific_internal_energy_minus_temperature
        ),
        specific_internal_energy_plus_pressure=(
            samples.specific_internal_energy_plus_pressure
        ),
        specific_internal_energy_minus_pressure=(
            samples.specific_internal_energy_minus_pressure
        ),
        density_plus_temperature=samples.density_plus_temperature,
        density_minus_temperature=samples.density_minus_temperature,
        density_plus_pressure=samples.density_plus_pressure,
        density_minus_pressure=samples.density_minus_pressure,
    )
    result = compute_convection(zero_top_layer_count=2, **keywords)
    standard = compute_convection(zero_top_layer_count=36, **keywords)
    # The production fixture is stably stratified.  A deliberately steep
    # controlled temperature column demonstrates the source's separate
    # disabled-diagnostic branch without relabelling it as a production
    # ConvectionResult.
    disabled_temperature = np.asarray(
        [4_000.0, 4_800.0, 6_400.0, 10_000.0, 16_000.0, 28_000.0],
        dtype=np.float64,
    )
    disabled_table = create_rosseland_opacity_table(LAYER_COUNT)
    ingest_rosseland_opacity_table(
        disabled_table,
        temperature_k=disabled_temperature,
        gas_pressure=runtime_state.gas_pressure,
        rosseland_opacity=final.rosseland_opacity,
    )
    disabled = compute_disabled_convection_diagnostics(
        column_mass=atmosphere.column_mass,
        rosseland_optical_depth=final.rosseland_optical_depth,
        temperature_k=disabled_temperature,
        gas_pressure=runtime_state.gas_pressure,
        mass_density=runtime_state.mass_density,
        rosseland_opacity=final.rosseland_opacity,
        absolute_radiation_pressure=final.absolute_radiation_pressure,
        total_pressure=total_pressure,
        surface_gravity_cgs=SURFACE_GRAVITY_CGS,
        target_integrated_eddington_flux=target,
        mixing_length=1.25,
        rosseland_table=disabled_table,
        overshoot_weight=0.0,
        zero_top_layer_count=2,
    )
    return ConvectionCheckpoint(
        result_field_names=tuple(field.name for field in fields(result)),
        geometric_depth_below_surface_km=result.geometric_depth_below_surface_km.copy(),
        logarithmic_gradient=result.logarithmic_temperature_pressure_gradient.copy(),
        adiabatic_gradient=result.adiabatic_gradient.copy(),
        heat_capacity=result.heat_capacity.copy(),
        raw_convective_flux=result.raw_convective_flux.copy(),
        returned_convective_flux=result.convective_flux.copy(),
        standard_suppressed_flux=standard.convective_flux.copy(),
        convective_velocity=result.convective_velocity.copy(),
        first_two_suppressed=bool(not np.any(result.convective_flux[:2])),
        standard_six_layer_fixture_fully_suppressed=bool(
            not np.any(standard.convective_flux)
        ),
        disabled_flux=disabled.convective_flux.copy(),
        disabled_velocity=disabled.convective_velocity.copy(),
        disabled_can_be_nonzero=bool(
            np.any(disabled.convective_flux) or np.any(disabled.convective_velocity)
        ),
    )
