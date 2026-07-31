"""Repository-only checkpoints for Chapter 11's first atmosphere pass.

The helpers load a manifest-owned 80-layer supplied seed, call the exact
staged Payne Zero setup/population/continuum/line implementations, and stop
at ``OpacityState``.  They never open the external source checkout, execute
transfer or corrections, or promote a live state to schema v4.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
from pathlib import Path
import warnings

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
SEED_FIXTURE = REPOSITORY_ROOT / "data/fixtures/chapter11_solar_seed.npz"
MOLECULAR_EQUILIBRIUM = (
    REPOSITORY_ROOT
    / "data/static/source_catalogs/lines/molecular_equilibrium_atmosphere.npz"
)
OBSERVED_ATOMIC_SUBSET = (
    REPOSITORY_ROOT / "data/subsets/chapter11_observed_atomic_subset.npy"
)
DIATOMIC_SUBSET = (
    REPOSITORY_ROOT / "data/subsets/chapter08_atmosphere_diatomic_subset.npy"
)
TITANIUM_OXIDE_SUBSET = (
    REPOSITORY_ROOT / "data/subsets/chapter08_tio_subset.npy"
)
WATER_SUBSET = REPOSITORY_ROOT / "data/subsets/chapter08_h2o_subset.npy"
DETAILED_TRANSITION_SUBSET = (
    REPOSITORY_ROOT / "data/subsets/chapter11_detailed_transition_subset.npz"
)
CONTINUUM_LEVEL_TABLES = (
    REPOSITORY_ROOT / "data/static/atmosphere_tables/continuum_level_tables.npz"
)

INPUT_HASHES = {
    SEED_FIXTURE: "a14e32467dc381a6ceed8da362c6c5a7e118276a964e461faa3b8843f77374da",
    MOLECULAR_EQUILIBRIUM: (
        "971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644"
    ),
    OBSERVED_ATOMIC_SUBSET: (
        "68ee3d3775cd8496b73d1500a490836f888c33364922d6541b1ec7fd8f603a5e"
    ),
    DIATOMIC_SUBSET: (
        "ebc43f107b8e046cf6494eb587a4234fa791e2d85b4a05817cd1935f969e45db"
    ),
    TITANIUM_OXIDE_SUBSET: (
        "204c2aa286b173c7a8125e7aa67139155522f7594acb44eabc1adac11bb6ab13"
    ),
    WATER_SUBSET: (
        "26bf33be3859dcd3ed601f820a88f7bf66fe7fc132a69ada0ad2aefba2b43409"
    ),
    DETAILED_TRANSITION_SUBSET: (
        "49752ed1627134145f12768feb9e10f3c578ba986e1be90fc75c3a6348601b25"
    ),
    CONTINUUM_LEVEL_TABLES: (
        "35a6839be4ff3dd824206c7a6b851b987132313374ede7ea5441f9d0bd69888f"
    ),
}

SEED_FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
    "microturbulence",
    "convective_flux",
    "convective_velocity",
)


@dataclass(frozen=True)
class SeedCheckpoint:
    """Validation surface of the supplied fixed-column seed."""

    layers: int
    field_names: tuple[str, ...]
    field_shapes: tuple[tuple[int, ...], ...]
    abundance_count: int
    column_mass_strictly_increasing: bool
    error_cases: tuple[str, ...]
    error_messages: tuple[str, ...]


@dataclass(frozen=True)
class SetupCheckpoint:
    """Resolved run controls and the narrow microturbulence fill rule."""

    setup_field_names: tuple[str, ...]
    iterations: int
    effective_temperature: float
    log_surface_gravity: float
    surface_gravity_cgs: float
    opacity_flags: tuple[int, ...]
    molecules_enabled: bool
    pressure_iteration_enabled: bool
    standard_rosseland_optical_depth: np.ndarray
    all_zero_profile_after: np.ndarray
    partly_positive_profile_after: np.ndarray
    partly_positive_unchanged: bool


@dataclass(frozen=True)
class HydrostaticCheckpoint:
    """Next-pass pressure support and the exact nonpositive-pressure floor."""

    column_mass: np.ndarray
    gravity_pressure: np.ndarray
    integrated_radiation_pressure: np.ndarray
    turbulent_pressure: np.ndarray
    pressure_constant: float
    next_pass_gas_pressure: np.ndarray
    balance_residual: np.ndarray
    warning_message: str
    warning_floor_positive: bool


@dataclass(frozen=True)
class QuantizationCheckpoint:
    """One and two applications of fixed-column format/parse quantization."""

    field_names: tuple[str, ...]
    maximum_absolute_delta: np.ndarray
    maximum_relative_delta: np.ndarray
    second_application_bitwise_equal: bool
    deck_line_count: int


@dataclass(frozen=True)
class RemapCheckpoint:
    """Scalar remap behavior and the returned final source-interval index."""

    source_grid: np.ndarray
    target_grid: np.ndarray
    constant_remap: np.ndarray
    monotone_remap: np.ndarray
    constant_final_source_interval_index: int
    monotone_final_source_interval_index: int


@dataclass(frozen=True)
class PopulationCheckpoint:
    """The exact current EOS/population/Doppler state."""

    population_field_names: tuple[str, ...]
    runtime_field_names: tuple[str, ...]
    layers: int
    packed_slot_count: int
    molecule_count: int
    molecular_state_present: bool
    mass_density: np.ndarray
    electron_density: np.ndarray
    fractional_doppler_widths: np.ndarray
    line_strength_population_factors: np.ndarray


@dataclass(frozen=True)
class SamplingGridCheckpoint:
    """The direct 30,000-frequency atmosphere sampling grid."""

    effective_temperature: float
    wavelength_nm: np.ndarray
    frequency_hz: np.ndarray
    frequency_weights: np.ndarray
    strict_threshold_temperatures: np.ndarray
    strict_threshold_start_indices: np.ndarray
    wavelength_increasing: bool
    frequency_decreasing: bool


@dataclass(frozen=True)
class ContinuumCheckpoint:
    """Continuum adapter and full depth-by-frequency slabs."""

    adapter_field_names: tuple[str, ...]
    absorption: np.ndarray
    scattering: np.ndarray
    source: np.ndarray
    active_reference_indices: np.ndarray
    active_reference_frequency_hz: np.ndarray
    threshold: np.ndarray
    reference_wavelength_nm: np.ndarray
    packed_reference_indices: np.ndarray


@dataclass(frozen=True)
class SelectionCheckpoint:
    """Compact catalog membership and current line-opacity result."""

    selected_catalog_field_names: tuple[str, ...]
    transition_catalog_field_names: tuple[str, ...]
    selected_line_count: int
    detailed_line_count: int
    detailed_accumulation_enabled: bool
    contributing_line_count: int
    line_opacity_dtype: str
    line_mass_absorption_coefficient: np.ndarray


@dataclass(frozen=True)
class OpacityPassCheckpoint:
    """Complete pre-transfer state and its exact memory/axis contract."""

    opacity_state_field_names: tuple[str, ...]
    depth_count: int
    frequency_count: int
    continuum_shape: tuple[int, int]
    line_shape: tuple[int, int]
    continuum_dtype: str
    line_dtype: str
    continuum_bytes: int
    line_bytes: int
    rosseland_entry_count: int
    schema_v4_product: bool


@dataclass(frozen=True)
class ReuseCheckpoint:
    """Caller-threaded catalog reuse across a perturbed atmosphere state."""

    selected_catalog_same_object: bool
    detailed_branch_inactive: bool
    temperature_changed: bool
    line_opacity_changed: bool
    maximum_line_opacity_change: float


@dataclass(frozen=True)
class BlanketingCheckpoint:
    """A compact plot view of continuum and continuum-plus-line extinction."""

    wavelength_nm: np.ndarray
    continuum_extinction: np.ndarray
    blanketed_extinction: np.ndarray
    depth_index: int
    line_to_continuum_peak_ratio: float
    temperature_k: np.ndarray
    rosseland_continuum_only: np.ndarray
    rosseland_blanketed: np.ndarray


def _rosseland_mean(
    opacity: np.ndarray,
    frequency_hz: np.ndarray,
    frequency_weights: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    """Return the harmonic Rosseland mean of ``opacity`` at every depth.

    The weight is the temperature derivative of the Planck function, so
    transparent windows dominate. This is the same definition Chapter 12
    finalizes; it is evaluated here only to show what line opacity does to it.
    """

    planck_h = 6.62607015e-27
    boltzmann_k = 1.380649e-16
    light_c = 2.99792458e10

    exponent = (
        planck_h * frequency_hz[None, :]
        / (boltzmann_k * temperature_k[:, None])
    )
    # dB/dT, up to constants that cancel between numerator and denominator.
    stable = np.exp(-exponent)
    weight = (
        frequency_hz[None, :] ** 4
        * stable
        / np.square(1.0 - stable)
        / np.square(temperature_k[:, None])
    )
    weight = weight * frequency_weights[None, :] * (2.0 * planck_h**2 / (light_c**2 * boltzmann_k))

    floor = np.finfo(np.float64).tiny
    numerator = np.sum(weight / np.maximum(opacity, floor), axis=1)
    denominator = np.sum(weight, axis=1)
    return denominator / np.maximum(numerator, floor)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_chapter11_runtime() -> None:
    """Resolve imports locally and fail closed if any Chapter 11 input changed."""

    from book.chapter06_runtime import configure_local_data_paths

    configure_local_data_paths()
    for path, expected in INPUT_HASHES.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Chapter 11 input identity changed for {path}: {actual}"
            )


def load_seed_atmosphere():
    """Return a fresh mutable ``ModelAtmosphere`` from the supplied fixture."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere

    with np.load(SEED_FIXTURE, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in SEED_FIELDS}
        abundances = np.asarray(
            archive["fixed_column_abundance_values"], dtype=np.float64
        )
        opacity_flags = " ".join(str(int(value)) for value in archive["opacity_flags"])
        metadata = {
            "effective_temperature": f"{float(archive['effective_temperature']):.6f}",
            "log_surface_gravity": f"{float(archive['log_surface_gravity']):.6f}",
            "opacity_flags": f"OPACITY IFOP {opacity_flags}",
            "pressure_iteration_enabled": str(
                int(archive["pressure_iteration_enabled"])
            ),
            "title": "Chapter 11 supplied solar seed",
            "surface_radiation_pressure_line": "PRADK 0.000000E+00",
        }
    return ModelAtmosphere(
        **arrays,
        metadata=metadata,
        fixed_column_abundance_values={
            atomic_number: float(abundances[atomic_number - 1])
            for atomic_number in range(1, 100)
        },
    )


def _copy_atmosphere(atmosphere):
    """Return a deep-enough mutable copy of one atmosphere seed."""

    from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere

    return ModelAtmosphere(
        **{name: np.asarray(getattr(atmosphere, name)).copy() for name in SEED_FIELDS},
        metadata=dict(atmosphere.metadata),
        fixed_column_abundance_values=dict(
            atmosphere.fixed_column_abundance_values
        ),
    )


def build_chapter11_config(atmosphere=None):
    """Return the explicit, supported one-pass configuration."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.config import (
        AtmosphereConfig,
        AtmosphereInput,
        AtmosphereOutput,
    )

    seed = load_seed_atmosphere() if atmosphere is None else atmosphere
    return AtmosphereConfig(
        inputs=AtmosphereInput(
            initial_atmosphere=seed,
            molecules_path=MOLECULAR_EQUILIBRIUM,
            observed_atomic_lines_path=OBSERVED_ATOMIC_SUBSET,
            diatomic_lines_path=DIATOMIC_SUBSET,
            titanium_oxide_lines_path=TITANIUM_OXIDE_SUBSET,
            water_lines_path=WATER_SUBSET,
            detailed_line_catalog_path=DETAILED_TRANSITION_SUBSET,
        ),
        outputs=AtmosphereOutput(),
        iterations=1,
        enable_molecules=True,
        enable_convection=False,
        enable_convergence_stop=False,
    )


def seed_checkpoint() -> SeedCheckpoint:
    """Validate the canonical seed and a one-change-at-a-time failure gallery."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.run_setup import validate_atmosphere_seed

    seed = load_seed_atmosphere()
    validate_atmosphere_seed(seed)
    cases = (
        ("nonfinite temperature", "temperature", lambda values: values.__setitem__(4, np.nan)),
        ("zero gas pressure", "gas_pressure", lambda values: values.__setitem__(2, 0.0)),
        (
            "nonmonotone column mass",
            "column_mass",
            lambda values: values.__setitem__(3, values[2]),
        ),
        (
            "negative microturbulence",
            "microturbulence",
            lambda values: values.__setitem__(1, -1.0),
        ),
    )
    messages = []
    for _, field_name, mutation in cases:
        bad = _copy_atmosphere(seed)
        mutation(getattr(bad, field_name))
        try:
            validate_atmosphere_seed(bad)
        except ValueError as exc:
            messages.append(str(exc))
        else:
            raise AssertionError(f"invalid {field_name} seed unexpectedly passed")
    return SeedCheckpoint(
        layers=seed.layers,
        field_names=SEED_FIELDS,
        field_shapes=tuple(getattr(seed, name).shape for name in SEED_FIELDS),
        abundance_count=len(seed.fixed_column_abundance_values),
        column_mass_strictly_increasing=bool(np.all(np.diff(seed.column_mass) > 0.0)),
        error_cases=tuple(case[0] for case in cases),
        error_messages=tuple(messages),
    )


def setup_checkpoint() -> SetupCheckpoint:
    """Resolve controls and contrast all-zero with partly positive turbulence."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.run_setup import RunSetup, resolve_run_setup

    all_zero = load_seed_atmosphere()
    setup = resolve_run_setup(build_chapter11_config(all_zero))
    partly_positive = load_seed_atmosphere()
    partly_positive.microturbulence[17] = 123_456.0
    before = partly_positive.microturbulence.copy()
    partial_setup = resolve_run_setup(build_chapter11_config(partly_positive))
    return SetupCheckpoint(
        setup_field_names=tuple(field.name for field in fields(RunSetup)),
        iterations=setup.iterations,
        effective_temperature=setup.effective_temperature,
        log_surface_gravity=setup.log_surface_gravity,
        surface_gravity_cgs=setup.surface_gravity_cgs,
        opacity_flags=tuple(setup.opacity_flags),
        molecules_enabled=setup.molecules_enabled,
        pressure_iteration_enabled=setup.pressure_iteration_enabled,
        standard_rosseland_optical_depth=setup.standard_rosseland_optical_depth.copy(),
        all_zero_profile_after=setup.atmosphere.microturbulence.copy(),
        partly_positive_profile_after=partial_setup.atmosphere.microturbulence.copy(),
        partly_positive_unchanged=bool(
            np.array_equal(before, partial_setup.atmosphere.microturbulence)
        ),
    )


def hydrostatic_checkpoint() -> HydrostaticCheckpoint:
    """Evaluate the next-pass helper and force its warning/floor branch once."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.hydrostatic import integrate_hydrostatic_pressure

    atmosphere = load_seed_atmosphere()
    with np.load(SEED_FIXTURE, allow_pickle=False) as archive:
        radiation = np.asarray(
            archive["previous_integrated_radiation_pressure"], dtype=np.float64
        )
        turbulent = np.asarray(
            archive["previous_turbulent_pressure"], dtype=np.float64
        )
    gravity = 10.0 ** float(atmosphere.metadata["log_surface_gravity"])
    pressure = integrate_hydrostatic_pressure(
        atmosphere,
        surface_gravity_cgs=gravity,
        integrated_radiation_pressure=radiation,
        turbulent_pressure=turbulent,
        pressure_constant=0.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        floored = integrate_hydrostatic_pressure(
            atmosphere,
            surface_gravity_cgs=gravity,
            integrated_radiation_pressure=1.01 * gravity * atmosphere.column_mass,
            turbulent_pressure=turbulent,
            pressure_constant=0.0,
        )
    return HydrostaticCheckpoint(
        column_mass=atmosphere.column_mass.copy(),
        gravity_pressure=gravity * atmosphere.column_mass,
        integrated_radiation_pressure=radiation.copy(),
        turbulent_pressure=turbulent.copy(),
        pressure_constant=0.0,
        next_pass_gas_pressure=pressure,
        balance_residual=(
            pressure + radiation + turbulent - gravity * atmosphere.column_mass
        ),
        warning_message=str(caught[0].message) if caught else "",
        warning_floor_positive=bool(np.all(floored > 0.0)),
    )


def quantization_checkpoint() -> QuantizationCheckpoint:
    """Apply the literal fixed-column formatter/parser once and twice."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.atmosphere_io import (
        format_atmosphere_deck,
        parse_atmosphere_deck,
    )

    seed = load_seed_atmosphere()
    first_text = format_atmosphere_deck(seed)
    once = parse_atmosphere_deck(first_text, source="Chapter 11 in-memory pass 1")
    second_text = format_atmosphere_deck(once)
    twice = parse_atmosphere_deck(second_text, source="Chapter 11 in-memory pass 2")
    absolute = []
    relative = []
    exact = True
    for name in SEED_FIELDS:
        source = np.asarray(getattr(seed, name), dtype=np.float64)
        first = np.asarray(getattr(once, name), dtype=np.float64)
        second = np.asarray(getattr(twice, name), dtype=np.float64)
        absolute.append(float(np.max(np.abs(first - source))))
        relative.append(
            float(np.max(np.abs(first - source) / np.maximum(np.abs(source), 1.0e-300)))
        )
        exact = exact and np.array_equal(first, second)
    return QuantizationCheckpoint(
        field_names=SEED_FIELDS,
        maximum_absolute_delta=np.asarray(absolute, dtype=np.float64),
        maximum_relative_delta=np.asarray(relative, dtype=np.float64),
        second_application_bitwise_equal=bool(exact),
        deck_line_count=len(first_text.splitlines()),
    )


def remap_checkpoint() -> RemapCheckpoint:
    """Probe exact scalar remapping with constant and monotone fields."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.radiative_transfer import remap_to_grid

    source_grid = np.linspace(-6.0, 2.0, 9, dtype=np.float64)
    target_grid = np.linspace(-6.5, 2.5, 19, dtype=np.float64)
    constant, constant_interval = remap_to_grid(
        source_grid, np.full(9, 3.5), target_grid
    )
    monotone, monotone_interval = remap_to_grid(
        source_grid, np.exp(0.2 * source_grid), target_grid
    )
    return RemapCheckpoint(
        source_grid=source_grid,
        target_grid=target_grid,
        constant_remap=constant,
        monotone_remap=monotone,
        constant_final_source_interval_index=constant_interval,
        monotone_final_source_interval_index=monotone_interval,
    )


@lru_cache(maxsize=1)
def _population_bundle():
    configure_chapter11_runtime()
    from payne_zero_atmosphere.runner import prepare_population_state

    config = build_chapter11_config()
    population = prepare_population_state(config, temperature_iteration_index=1)
    return config, population


def population_checkpoint() -> PopulationCheckpoint:
    """Return the current molecule-enabled population state."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.runner import AtmospherePopulationState
    from payne_zero_atmosphere.runtime_state import AtmosphereRuntimeState

    _, population = _population_bundle()
    runtime = population.runtime_state
    molecular_count = (
        0
        if population.molecular_state is None
        else int(population.molecular_state.molecular_populations.shape[1])
    )
    return PopulationCheckpoint(
        population_field_names=tuple(
            field.name for field in fields(AtmospherePopulationState)
        ),
        runtime_field_names=tuple(field.name for field in fields(AtmosphereRuntimeState)),
        layers=population.setup.atmosphere.layers,
        packed_slot_count=int(
            runtime.ion_stage_populations_by_packed_slot.shape[1]
        ),
        molecule_count=molecular_count,
        molecular_state_present=population.molecular_state is not None,
        mass_density=runtime.mass_density.copy(),
        electron_density=runtime.electron_density.copy(),
        fractional_doppler_widths=population.fractional_doppler_widths.copy(),
        line_strength_population_factors=(
            population.partition_normalized_population_over_mass_density_and_fractional_doppler_width.copy()
        ),
    )


def sampling_grid_checkpoint() -> SamplingGridCheckpoint:
    """Build the exact grid and expose every strict threshold sentinel."""

    configure_chapter11_runtime()
    from book.chapter11_teaching import opacity_grid_start_index
    from payne_zero_atmosphere.continuum_opacity import build_opacity_sampling_grid

    effective_temperature = 5778.0
    wavelength, weights = build_opacity_sampling_grid(effective_temperature)
    frequency = 2.99792458e17 / wavelength
    sentinels = np.asarray(
        [
            30_000.0,
            np.nextafter(30_000.0, -np.inf),
            13_000.0,
            np.nextafter(13_000.0, -np.inf),
            7_250.0,
            np.nextafter(7_250.0, -np.inf),
            4_500.0,
            np.nextafter(4_500.0, -np.inf),
        ],
        dtype=np.float64,
    )
    return SamplingGridCheckpoint(
        effective_temperature=effective_temperature,
        wavelength_nm=wavelength.copy(),
        frequency_hz=frequency,
        frequency_weights=weights.copy(),
        strict_threshold_temperatures=sentinels,
        strict_threshold_start_indices=np.asarray(
            [opacity_grid_start_index(value) for value in sentinels],
            dtype=np.int64,
        ),
        wavelength_increasing=bool(np.all(np.diff(wavelength) > 0.0)),
        frequency_decreasing=bool(np.all(np.diff(frequency) < 0.0)),
    )


@lru_cache(maxsize=1)
def full_opacity_state():
    """Build and cache exactly one complete pre-transfer opacity state."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.runner import prepare_opacity_state

    config, population = _population_bundle()
    return prepare_opacity_state(config, population_state=population)


def continuum_checkpoint() -> ContinuumCheckpoint:
    """Return the adapter fields, full continuum, and selection reference."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.continuum_opacity import ContinuumAtmosphereState

    state = full_opacity_state()
    return ContinuumCheckpoint(
        adapter_field_names=tuple(
            field.name for field in fields(ContinuumAtmosphereState)
        ),
        absorption=state.continuum_absorption,
        scattering=state.continuum_scattering,
        source=state.continuum_source,
        active_reference_indices=state.active_continuum_indices,
        active_reference_frequency_hz=state.active_continuum_frequency_hz,
        threshold=state.continuum_line_selection_threshold,
        reference_wavelength_nm=state.continuum_reference_wavelength_nm,
        packed_reference_indices=state.wavelength_bin_edges,
    )


def selection_checkpoint() -> SelectionCheckpoint:
    """Return both catalog interfaces and the final current line slab."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.line_catalog import (
        LineTransitionCatalog,
        SelectedLineCatalog,
        read_line_transition_catalog,
    )

    state = full_opacity_state()
    detailed_catalog = read_line_transition_catalog(DETAILED_TRANSITION_SUBSET)
    selected_count = (
        0 if state.selected_line_catalog is None else state.selected_line_catalog.line_count
    )
    return SelectionCheckpoint(
        selected_catalog_field_names=tuple(
            field.name for field in fields(SelectedLineCatalog)
        ),
        transition_catalog_field_names=tuple(
            field.name for field in fields(LineTransitionCatalog)
        ),
        selected_line_count=selected_count,
        detailed_line_count=detailed_catalog.line_count,
        detailed_accumulation_enabled=bool(
            state.population_state.setup.opacity_flags[16]
        ),
        contributing_line_count=state.line_opacity.selected_line_count,
        line_opacity_dtype=str(
            state.line_opacity.line_mass_absorption_coefficient.dtype
        ),
        line_mass_absorption_coefficient=(
            state.line_opacity.line_mass_absorption_coefficient
        ),
    )


def opacity_pass_checkpoint() -> OpacityPassCheckpoint:
    """Describe the complete Chapter 11 handoff without calling transfer."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.runner import OpacityState

    state = full_opacity_state()
    continuum_shape = tuple(int(value) for value in state.continuum_absorption.shape)
    line_shape = tuple(
        int(value)
        for value in state.line_opacity.line_mass_absorption_coefficient.shape
    )
    return OpacityPassCheckpoint(
        opacity_state_field_names=tuple(field.name for field in fields(OpacityState)),
        depth_count=continuum_shape[0],
        frequency_count=continuum_shape[1],
        continuum_shape=continuum_shape,
        line_shape=line_shape,
        continuum_dtype=str(state.continuum_absorption.dtype),
        line_dtype=str(state.line_opacity.line_mass_absorption_coefficient.dtype),
        continuum_bytes=sum(
            array.nbytes
            for array in (
                state.continuum_absorption,
                state.continuum_scattering,
                state.continuum_source,
            )
        ),
        line_bytes=state.line_opacity.line_mass_absorption_coefficient.nbytes,
        rosseland_entry_count=state.rosseland_table.entry_count,
        schema_v4_product=False,
    )


@lru_cache(maxsize=1)
def reuse_checkpoint() -> ReuseCheckpoint:
    """Perturb temperature and thread both catalog objects into a new pass."""

    configure_chapter11_runtime()
    from payne_zero_atmosphere.runner import (
        prepare_opacity_state,
        prepare_population_state,
    )

    first = full_opacity_state()
    changed_atmosphere = load_seed_atmosphere()
    changed_atmosphere.temperature *= 1.001
    changed_config = build_chapter11_config(changed_atmosphere)
    changed_population = prepare_population_state(
        changed_config, temperature_iteration_index=2
    )
    second = prepare_opacity_state(
        changed_config,
        population_state=changed_population,
        selected_line_catalog=first.selected_line_catalog,
        transition_line_catalog=first.transition_line_catalog,
    )
    difference = np.abs(
        np.asarray(second.line_opacity.line_mass_absorption_coefficient)
        - np.asarray(first.line_opacity.line_mass_absorption_coefficient)
    )
    return ReuseCheckpoint(
        selected_catalog_same_object=(
            second.selected_line_catalog is first.selected_line_catalog
        ),
        detailed_branch_inactive=(
            first.transition_line_catalog is None
            and second.transition_line_catalog is None
        ),
        temperature_changed=bool(
            np.any(
                changed_population.setup.atmosphere.temperature
                != first.population_state.setup.atmosphere.temperature
            )
        ),
        line_opacity_changed=bool(np.any(difference != 0.0)),
        maximum_line_opacity_change=float(np.max(difference)),
    )


def blanketing_checkpoint(
    *, depth_index: int = 40, stride: int = 30
) -> BlanketingCheckpoint:
    """Return a downsampled one-panel view while retaining full calculations."""

    state = full_opacity_state()
    depth = int(depth_index)
    step = int(stride)
    if not 0 <= depth < state.continuum_absorption.shape[0]:
        raise ValueError("depth_index is outside the atmosphere")
    if step <= 0:
        raise ValueError("stride must be positive")
    continuum = (
        state.continuum_absorption[depth] + state.continuum_scattering[depth]
    )
    blanketed = (
        continuum + state.line_opacity.line_mass_absorption_coefficient[depth]
    )
    ratio = np.max(
        blanketed / np.maximum(continuum, np.finfo(np.float64).tiny)
    )
    temperature = load_seed_atmosphere().temperature
    all_continuum = (
        state.continuum_absorption + state.continuum_scattering
    )
    all_blanketed = (
        all_continuum + state.line_opacity.line_mass_absorption_coefficient
    )
    rosseland_arguments = (
        state.opacity_frequency_hz,
        state.frequency_weights,
        temperature,
    )
    return BlanketingCheckpoint(
        wavelength_nm=state.opacity_wavelength_grid_nm[::step],
        continuum_extinction=continuum[::step],
        blanketed_extinction=blanketed[::step],
        depth_index=depth,
        line_to_continuum_peak_ratio=float(ratio),
        temperature_k=temperature,
        rosseland_continuum_only=_rosseland_mean(
            all_continuum, *rosseland_arguments
        ),
        rosseland_blanketed=_rosseland_mean(
            all_blanketed, *rosseland_arguments
        ),
    )
