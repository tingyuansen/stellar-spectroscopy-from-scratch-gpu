"""Progressive Chapter 5 helpers for continuum opacity and scattering.

The reader-facing notebook uses the small analytic helpers directly and calls
the exact staged Payne Zero sources only through the route checkpoints below.
No helper opens a golden file.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = REPOSITORY_ROOT / "data" / "static"
CONTINUUM_FIXTURE = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter05_continuum_states.npz"
)
SYNTHESIS_CONTINUUM_TABLES = (
    STATIC_ROOT / "synthesis_tables" / "continuum_tables.npz"
)
SYNTHESIS_EDGE_GRID = (
    STATIC_ROOT / "synthesis_tables" / "continuum_edge_grid.npz"
)
ATMOSPHERE_CONTINUUM_TABLES = (
    STATIC_ROOT / "atmosphere_tables" / "continuum_opacity_tables.npz"
)
ATMOSPHERE_KARZAS_TABLES = (
    STATIC_ROOT / "atmosphere_tables" / "karzas_latter_tables.npz"
)
ATMOSPHERE_MOLECULAR_TABLES = (
    STATIC_ROOT / "atmosphere_tables" / "molecular_equilibrium_tables.npz"
)
RUNNER_OPACITY_FLAGS = (
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
)
SYNTHESIS_CONTINUUM_FIELDS = (
    "temperature",
    "mass_density",
    "electron_density",
    "hydrogen_partition_normalized_ion_stage_populations",
    "hydrogen_neutral_population",
    "helium_neutral_population",
    "helium_singly_ionized_population",
    "carbon_partition_normalized_ion_stage_populations",
    "magnesium_neutral_partition_normalized_population",
    "aluminum_neutral_partition_normalized_population",
    "silicon_neutral_partition_normalized_population",
    "iron_neutral_partition_normalized_population",
    "partition_normalized_populations",
    "ion_stage_populations",
    "signed_continuum_edge_frequency_hz",
    "continuum_edge_wavelength_nm",
    "continuum_edge_midpoint_wavelength_nm",
    "continuum_edge_interval_width_squared_over_two_nm2",
)
PLANCK_ERG_SECOND = 6.62607015e-27
BOLTZMANN_ERG_PER_K = 1.380649e-16
LIGHT_SPEED_NM_PER_S = 2.99792458e17
HMINUS_BOUNDFREE_THRESHOLD_HZ = 1.82365e14
SAMPLED_EXTENSION_SUPPORTED_WAVELENGTH_NM = (
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    320.0,
    400.0,
    500.0,
    700.0,
    1_000.0,
    1_600.0,
    2_500.0,
)


@dataclass(frozen=True)
class OpacityScalingCheckpoint:
    """Linearity and inverse-density evidence for ``n sigma / rho``."""

    baseline_cm2_per_g: np.ndarray
    doubled_population_cm2_per_g: np.ndarray
    doubled_density_cm2_per_g: np.ndarray
    population_ratio: np.ndarray
    density_ratio: np.ndarray


@dataclass(frozen=True)
class StimulatedEmissionCheckpoint:
    """Exact values and limiting approximations for the LTE net factor."""

    photon_energy_over_kt: np.ndarray
    factor: np.ndarray
    low_energy_approximation: np.ndarray
    high_energy_limit: np.ndarray


@dataclass(frozen=True)
class AtmosphereGridCheckpoint:
    """The five exact direct-sampling regimes and boundary assignments."""

    effective_temperature_k: np.ndarray
    first_wavelength_nm: np.ndarray
    last_wavelength_nm: np.ndarray
    sample_count: np.ndarray
    first_frequency_weight_hz: np.ndarray
    interior_frequency_weight_hz: np.ndarray
    last_frequency_weight_hz: np.ndarray
    active_reference_count: np.ndarray


@dataclass(frozen=True)
class EdgeTripletCheckpoint:
    """Used synthesis intervals, their sample frequencies, and basis checks."""

    requested_wavelength_nm: np.ndarray
    interval_index: np.ndarray
    used_interval_index: np.ndarray
    sample_frequency_hz: np.ndarray
    left_basis: np.ndarray
    midpoint_basis: np.ndarray
    right_basis: np.ndarray
    basis_sum: np.ndarray
    packaged_samples_bitwise_equal: bool
    sign_flip_invariant: bool


@dataclass(frozen=True)
class ContinuumRouteCheckpoint:
    """One exact continuum route evaluated on a compact requested grid."""

    regime: str
    wavelength_nm: np.ndarray
    absorption_cm2_per_g: np.ndarray
    scattering_cm2_per_g: np.ndarray
    continuum_opacity: np.ndarray
    continuum_source: np.ndarray | None
    route: str
    dtype: str


@dataclass(frozen=True)
class HMinusEdgeCheckpoint:
    """Exact H-minus components across the production bound-free edge."""

    frequency_hz: np.ndarray
    wavelength_nm: np.ndarray
    stored_boundfree_cross_section_1e_minus_18_cm2: np.ndarray
    boundfree_absorption_cm2_per_g: np.ndarray
    freefree_absorption_cm2_per_g: np.ndarray
    total_absorption_cm2_per_g: np.ndarray
    threshold_hz: float
    last_table_wavelength_nm: float
    temperature_k: np.ndarray
    electron_density_cm3: np.ndarray
    mass_density_g_cm3: np.ndarray
    hydrogen_partition_normalized_population_cm3: np.ndarray
    fixture_hydrogen_departure_coefficient: np.ndarray
    unit_hminus_population_cm3: np.ndarray
    fixture_hminus_population_cm3: np.ndarray
    physical_boundfree_cross_section_cm2: np.ndarray
    transparent_unit_boundfree_absorption_cm2_per_g: np.ndarray
    transparent_fixture_boundfree_absorption_cm2_per_g: np.ndarray
    exact_unit_boundfree_absorption_cm2_per_g: np.ndarray


@dataclass(frozen=True)
class MolecularContinuumCheckpoint:
    """Exact atmosphere-only CH, OH, and collision-induced absorption."""

    wavenumber_cm1: np.ndarray
    wavelength_nm: np.ndarray
    ch_absorption_cm2_per_g: np.ndarray
    oh_absorption_cm2_per_g: np.ndarray
    h2h2_absorption_cm2_per_g: np.ndarray
    h2he_absorption_cm2_per_g: np.ndarray
    collision_induced_absorption_cm2_per_g: np.ndarray
    total_absorption_cm2_per_g: np.ndarray
    temperature_gate_k: np.ndarray
    ch_at_temperature_gate_cm2_per_g: np.ndarray
    oh_at_temperature_gate_cm2_per_g: np.ndarray
    h2_equilibrium_temperature_k: np.ndarray
    h2_equilibrium_population_cm3: np.ndarray
    h2_cutoff_temperature_k: np.ndarray
    h2_rayleigh_at_cutoff_cm2_per_g: np.ndarray
    cia_temperature_k: np.ndarray
    cia_lower_column_weight: np.ndarray
    cia_upper_column_weight: np.ndarray
    cia_h2h2_log10_coefficient: np.ndarray
    cia_h2he_log10_coefficient: np.ndarray
    all_warm_absorption_cm2_per_g: np.ndarray
    mixed_column_absorption_cm2_per_g: np.ndarray
    molecule_disabled_absorption_cm2_per_g: np.ndarray
    ifop13_alone_scattering_cm2_per_g: np.ndarray
    ifop4_and13_h2_increment_cm2_per_g: np.ndarray


@dataclass(frozen=True)
class ScatteringCheckpoint:
    """Exact Thomson and atmosphere Rayleigh component columns."""

    wavelength_nm: np.ndarray
    electron_scattering_cm2_per_g: np.ndarray
    hydrogen_rayleigh_cm2_per_g: np.ndarray
    helium_rayleigh_cm2_per_g: np.ndarray
    molecular_hydrogen_rayleigh_cm2_per_g: np.ndarray
    total_scattering_cm2_per_g: np.ndarray
    cap_frequency_hz: np.ndarray
    cap_component_value_cm2_per_g: np.ndarray
    above_cap_component_value_cm2_per_g: np.ndarray


@dataclass(frozen=True)
class LineReferenceCheckpoint:
    """The exact atmosphere continuum subroute used by later line selection."""

    effective_temperature_k: float
    active_count: int
    active_index: np.ndarray
    active_frequency_hz: np.ndarray
    threshold_cm2_per_g: np.ndarray
    wavelength_nm: np.ndarray
    packed_wavelength_index: np.ndarray
    inactive_count: int
    duplicated_last_column: bool
    packed_sentinel: int
    dtype: str
    inactive_placeholder_matches: bool
    inactive_placeholder_max_abs_residual: float


@dataclass(frozen=True)
class AtmosphereComponentBudgetCheckpoint:
    """Named exact atmosphere absorption components before their total hides them."""

    regime: str
    frequency_hz: np.ndarray
    component_names: tuple[str, ...]
    component_absorption_cm2_per_g: np.ndarray
    component_source: np.ndarray
    component_source_numerator: np.ndarray
    ordered_absorption_partial_sum_cm2_per_g: np.ndarray
    ordered_source_numerator_partial_sum: np.ndarray
    reconstructed_absorption_cm2_per_g: np.ndarray
    exact_absorption_cm2_per_g: np.ndarray
    absorption_residual_cm2_per_g: np.ndarray
    exact_scattering_cm2_per_g: np.ndarray
    reconstructed_source: np.ndarray
    exact_source: np.ndarray
    reconstructed_source_numerator: np.ndarray
    exact_source_numerator: np.ndarray
    source_numerator_residual: np.ndarray


@dataclass(frozen=True)
class MetalPopulationOwnershipCheckpoint:
    """Independent normalized bound-free and actual charge-square perturbations."""

    frequency_hz: np.ndarray
    normalized_only_absorption_cm2_per_g: np.ndarray
    doubled_normalized_only_absorption_cm2_per_g: np.ndarray
    actual_only_absorption_cm2_per_g: np.ndarray
    doubled_actual_only_absorption_cm2_per_g: np.ndarray
    normalized_ratio: np.ndarray
    actual_ratio: np.ndarray
    synthesis_normalized_owner_index: tuple[int, int]
    synthesis_actual_owner_index: tuple[int, int]
    baseline_hot_metal_populations: np.ndarray
    doubled_normalized_hot_metal_populations: np.ndarray
    baseline_charge_square_population_sum: np.ndarray
    doubled_actual_charge_square_population_sum: np.ndarray
    expected_actual_charge_square_delta: np.ndarray
    normalized_perturbation_preserves_charge_square: bool
    actual_perturbation_preserves_hot_metal_populations: bool


@dataclass(frozen=True)
class StateProjectionCheckpoint:
    """The distinct Chapter 4 handoff and two exact continuum consumer views."""

    regime: str
    synthesis_handoff_field_names: tuple[str, ...]
    synthesis_continuum_field_names: tuple[str, ...]
    atmosphere_continuum_field_names: tuple[str, ...]
    synthesis_continuum_shapes: tuple[tuple[int, ...], ...]
    atmosphere_continuum_shapes: tuple[tuple[int, ...], ...]
    shared_field_names: tuple[str, ...]


@dataclass(frozen=True)
class StoredH2InvarianceCheckpoint:
    """Standard synthesis products before and after changing stored schema H2."""

    wavelength_nm: np.ndarray
    baseline_absorption_cm2_per_g: np.ndarray
    changed_absorption_cm2_per_g: np.ndarray
    baseline_scattering_cm2_per_g: np.ndarray
    changed_scattering_cm2_per_g: np.ndarray
    stored_h2_scale_factor: float
    bitwise_equal: bool


@dataclass(frozen=True)
class NumbaTimingCheckpoint:
    """Two isolated-process timing records sharing one external cache."""

    first_process: Mapping[str, object]
    cached_process: Mapping[str, object]
    cache_file_count: int
    output_fingerprint: str
    all_outputs_bitwise_equal: bool


@dataclass(frozen=True)
class SampledSynthesisCheckpoint:
    """One explicitly labelled sampled diagnostic or precomputed extension."""

    regime: str
    wavelength_nm: np.ndarray
    frequency_hz: np.ndarray
    absorption_cm2_per_g: np.ndarray
    scattering_cm2_per_g: np.ndarray
    source_bnu: np.ndarray
    direct_source_bnu: np.ndarray
    source_residual: np.ndarray
    source_matches_direct: bool
    route: str
    coulomb_table_energy_first: bool
    frequency_invariant_shapes: Mapping[str, tuple[int, ...]]
    tensor_cache_entries_after_reuse: int
    supported_wavelength_nm: np.ndarray | None
    supported_wavelength_bounds_nm: tuple[float, float] | None
    dtype: str


@dataclass(frozen=True)
class EdgeUseTraceCheckpoint:
    """Observed standard-route sampling of only the requested edge intervals."""

    requested_wavelength_nm: np.ndarray
    interval_index: np.ndarray
    used_interval_index: np.ndarray
    called_frequency_hz: np.ndarray
    expected_called_frequency_hz: np.ndarray
    called_frequency_count: int
    total_interval_count: int
    unused_interval_count: int
    exact_internal_edge_nm: float
    exact_internal_edge_assigned_interval: int
    coulomb_table_energy_first: bool
    frequency_invariants_was_none: bool


@dataclass(frozen=True)
class ContinuumTablePreflight:
    """Manifest-bound identities and next-use shapes for continuum static data."""

    roles: tuple[str, ...]
    relative_paths: tuple[str, ...]
    sha256: tuple[str, ...]
    manifest_verified: tuple[bool, ...]
    hminus_boundfree_wavelength_shape: tuple[int, ...]
    hminus_boundfree_cross_section_shape: tuple[int, ...]
    hminus_freefree_wavelength_shape: tuple[int, ...]
    hminus_freefree_temperature_shape: tuple[int, ...]
    hminus_stored_unit: str


@dataclass(frozen=True)
class EdgeReconstructionCheckpoint:
    """One exact standard synthesis interval and its transparent reconstruction."""

    interval_index: int
    left_wavelength_nm: float
    midpoint_wavelength_nm: float
    right_wavelength_nm: float
    sample_frequency_hz: np.ndarray
    sample_wavelength_nm: np.ndarray
    target_wavelength_nm: np.ndarray
    absorption_samples_cm2_per_g: np.ndarray
    scattering_samples_cm2_per_g: np.ndarray
    reconstructed_absorption_cm2_per_g: np.ndarray
    reconstructed_scattering_cm2_per_g: np.ndarray
    exact_absorption_cm2_per_g: np.ndarray
    exact_scattering_cm2_per_g: np.ndarray
    basis_sum: np.ndarray


@dataclass(frozen=True)
class StandardSynthesisComponentCheckpoint:
    """Canonical standard-route samples, named owners, and ordered sums."""

    regime: str
    requested_wavelength_nm: np.ndarray
    used_interval_index: np.ndarray
    sample_frequency_hz: np.ndarray
    build_pops_hydrogen_ionized_population: np.ndarray
    build_pops_hot_metal_populations: np.ndarray
    build_pops_charge_square_population_sum: np.ndarray
    absorption_component_names: tuple[str, ...]
    absorption_components_cm2_per_g: np.ndarray
    absorption_partial_sums_cm2_per_g: np.ndarray
    exact_sampled_absorption_cm2_per_g: np.ndarray
    absorption_residual_cm2_per_g: np.ndarray
    scattering_component_names: tuple[str, ...]
    scattering_components_cm2_per_g: np.ndarray
    scattering_partial_sums_cm2_per_g: np.ndarray
    exact_sampled_scattering_cm2_per_g: np.ndarray
    scattering_residual_cm2_per_g: np.ndarray
    minor_absorption_subcomponent_names: tuple[str, ...]
    minor_absorption_subcomponents_cm2_per_g: np.ndarray
    minor_scattering_subcomponent_names: tuple[str, ...]
    minor_scattering_subcomponents_cm2_per_g: np.ndarray
    final_absorption_cm2_per_g: np.ndarray
    final_scattering_cm2_per_g: np.ndarray
    coulomb_table_energy_first: bool
    frequency_invariants_was_none: bool


def configure_local_data_paths() -> None:
    """Point exact staged loaders at this repository's immutable data."""

    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(STATIC_ROOT)


def continuum_table_preflight() -> ContinuumTablePreflight:
    """Fail closed after verifying the complete immutable continuum bundle."""

    return _continuum_table_preflight(
        (
            ("atmosphere continuum", ATMOSPHERE_CONTINUUM_TABLES),
            ("atmosphere hydrogenic", ATMOSPHERE_KARZAS_TABLES),
            ("atmosphere local H2", ATMOSPHERE_MOLECULAR_TABLES),
            ("synthesis continuum", SYNTHESIS_CONTINUUM_TABLES),
            ("synthesis edge geometry", SYNTHESIS_EDGE_GRID),
        )
    )


def hminus_table_preflight() -> ContinuumTablePreflight:
    """Fail closed on only the table first consumed by the H-minus lesson."""

    return _continuum_table_preflight(
        (("atmosphere continuum", ATMOSPHERE_CONTINUUM_TABLES),)
    )


def _continuum_table_preflight(
    paths: tuple[tuple[str, Path], ...],
) -> ContinuumTablePreflight:
    """Verify a declared just-in-time subset against the data manifest."""

    if not paths:
        raise ValueError("at least one continuum table must be preflighted")
    all_paths = (
        ("atmosphere continuum", ATMOSPHERE_CONTINUUM_TABLES),
        ("atmosphere hydrogenic", ATMOSPHERE_KARZAS_TABLES),
        ("atmosphere local H2", ATMOSPHERE_MOLECULAR_TABLES),
        ("synthesis continuum", SYNTHESIS_CONTINUUM_TABLES),
        ("synthesis edge geometry", SYNTHESIS_EDGE_GRID),
    )
    manifest = json.loads(
        (REPOSITORY_ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8")
    )
    by_path = {entry["path"]: entry for entry in manifest["entries"]}
    relative_paths = tuple(
        path.relative_to(REPOSITORY_ROOT).as_posix() for _, path in paths
    )
    digests = tuple(
        hashlib.sha256(path.read_bytes()).hexdigest() for _, path in paths
    )
    verified = tuple(
        relative in by_path and by_path[relative]["sha256"] == digest
        for relative, digest in zip(relative_paths, digests)
    )
    if not all(verified):
        failed = [
            relative
            for relative, matches in zip(relative_paths, verified)
            if not matches
        ]
        raise RuntimeError(
            "continuum table identity mismatch: " + ", ".join(failed)
        )
    with np.load(ATMOSPHERE_CONTINUUM_TABLES, allow_pickle=False) as table:
        boundfree_wavelength_shape = table[
            "hminus_boundfree_wavelength_nm"
        ].shape
        boundfree_cross_section_shape = table[
            "hminus_boundfree_cross_section_cm2"
        ].shape
        freefree_wavelength_shape = table[
            "hminus_freefree_inverse_wavelength_grid"
        ].shape
        freefree_temperature_shape = table["hminus_freefree_theta_grid"].shape
    atmosphere_relative = next(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for _, path in all_paths
        if path == ATMOSPHERE_CONTINUUM_TABLES
    )
    hminus_unit = by_path[atmosphere_relative]["arrays"][
        "hminus_boundfree_cross_section_cm2"
    ]["unit"]
    return ContinuumTablePreflight(
        roles=tuple(role for role, _ in paths),
        relative_paths=relative_paths,
        sha256=digests,
        manifest_verified=verified,
        hminus_boundfree_wavelength_shape=boundfree_wavelength_shape,
        hminus_boundfree_cross_section_shape=boundfree_cross_section_shape,
        hminus_freefree_wavelength_shape=freefree_wavelength_shape,
        hminus_freefree_temperature_shape=freefree_temperature_shape,
        hminus_stored_unit=str(hminus_unit),
    )


def _positive_finite(name: str, values: np.ndarray) -> np.ndarray:
    """Return a float64 array after a reader-facing physical-domain check."""

    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def mass_opacity_from_cross_section(
    absorber_number_density_cm3: np.ndarray | float,
    cross_section_cm2: np.ndarray | float,
    mass_density_g_cm3: np.ndarray | float,
) -> np.ndarray:
    """Convert microscopic area to mass opacity with no repair clamp."""

    number_density = _positive_finite(
        "absorber_number_density_cm3",
        absorber_number_density_cm3,
    )
    cross_section = _positive_finite("cross_section_cm2", cross_section_cm2)
    mass_density = _positive_finite("mass_density_g_cm3", mass_density_g_cm3)
    try:
        number_density, cross_section, mass_density = np.broadcast_arrays(
            number_density,
            cross_section,
            mass_density,
        )
    except ValueError as error:
        raise ValueError("opacity inputs must be broadcast-compatible") from error
    return number_density * cross_section / mass_density


def opacity_scaling_checkpoint() -> OpacityScalingCheckpoint:
    """Return a small live factor-of-two experiment."""

    number_density = np.asarray([2.0e13, 7.0e14])
    cross_section = np.asarray([4.0e-18, 9.0e-19])
    mass_density = np.asarray([2.0e-8, 5.0e-7])
    baseline = mass_opacity_from_cross_section(
        number_density,
        cross_section,
        mass_density,
    )
    doubled_population = mass_opacity_from_cross_section(
        2.0 * number_density,
        cross_section,
        mass_density,
    )
    doubled_density = mass_opacity_from_cross_section(
        number_density,
        cross_section,
        2.0 * mass_density,
    )
    return OpacityScalingCheckpoint(
        baseline_cm2_per_g=baseline,
        doubled_population_cm2_per_g=doubled_population,
        doubled_density_cm2_per_g=doubled_density,
        population_ratio=doubled_population / baseline,
        density_ratio=doubled_density / baseline,
    )


def stimulated_emission_factor(
    frequency_hz: np.ndarray | float,
    temperature_k: np.ndarray | float,
) -> np.ndarray:
    """Return ``1 - exp(-h nu / kT)`` using its stable ``expm1`` form."""

    frequency = _positive_finite("frequency_hz", frequency_hz)
    temperature = _positive_finite("temperature_k", temperature_k)
    try:
        frequency, temperature = np.broadcast_arrays(frequency, temperature)
    except ValueError as error:
        raise ValueError("frequency_hz and temperature_k must broadcast") from error
    ratio = PLANCK_ERG_SECOND * frequency / (
        BOLTZMANN_ERG_PER_K * temperature
    )
    return -np.expm1(-ratio)


def stimulated_emission_checkpoint(
    photon_energy_over_kt: Sequence[float] = (1.0e-6, 1.0e-3, 0.1, 1.0, 10.0),
) -> StimulatedEmissionCheckpoint:
    """Evaluate the dimensionless law without introducing arbitrary units."""

    ratio = _positive_finite(
        "photon_energy_over_kt",
        np.asarray(photon_energy_over_kt, dtype=np.float64),
    )
    factor = -np.expm1(-ratio)
    return StimulatedEmissionCheckpoint(
        photon_energy_over_kt=ratio,
        factor=factor,
        low_energy_approximation=ratio.copy(),
        high_energy_limit=np.ones_like(ratio),
    )


def load_regime_state(regime: str, lane: str) -> dict[str, np.ndarray]:
    """Load one input-only Chapter 5 state and strip its archive prefix."""

    if lane not in {"atmosphere", "synthesis"}:
        raise ValueError("lane must be 'atmosphere' or 'synthesis'")
    with np.load(CONTINUUM_FIXTURE, allow_pickle=False) as fixture:
        regimes = tuple(str(value) for value in fixture["regime_names"].tolist())
        if regime not in regimes:
            raise ValueError(f"unknown regime {regime!r}; expected one of {regimes}")
        prefix = f"{regime}__{lane}__"
        state = {
            name.removeprefix(prefix): np.asarray(fixture[name]).copy()
            for name in fixture.files
            if name.startswith(prefix)
        }
    return state


def load_atmosphere_continuum_state(regime: str):
    """Return the exact 18-field atmosphere continuum adapter for one regime."""

    configure_local_data_paths()
    from payne_zero_atmosphere.continuum_opacity import ContinuumAtmosphereState

    raw_state = load_regime_state(regime, "atmosphere")
    allowed = {field.name for field in fields(ContinuumAtmosphereState)}
    return ContinuumAtmosphereState(
        **{name: values for name, values in raw_state.items() if name in allowed}
    )


def project_synthesis_continuum_state(
    synthesis_handoff: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return the exact 18-field view consumed by standard synthesis.

    The complete schema-v4 handoff has 27 fields.  Standard continuum
    synthesis deliberately receives only the tuple declared in
    ``SYNTHESIS_CONTINUUM_FIELDS``; in particular, it reconstructs H II from
    the normalized hydrogen stages instead of reading the optional stored
    ``hydrogen_ionized_population`` field.
    """

    missing = tuple(
        name for name in SYNTHESIS_CONTINUUM_FIELDS if name not in synthesis_handoff
    )
    if missing:
        raise ValueError(
            "synthesis handoff is missing continuum fields: "
            + ", ".join(missing)
        )
    return {
        name: np.asarray(synthesis_handoff[name])
        for name in SYNTHESIS_CONTINUUM_FIELDS
    }


def state_projection_checkpoint(
    regime: str = "solar_dwarf",
) -> StateProjectionCheckpoint:
    """Expose the 27-field handoff and its two noninterchangeable consumers."""

    synthesis_handoff = load_regime_state(regime, "synthesis")
    synthesis_view = project_synthesis_continuum_state(synthesis_handoff)
    atmosphere_view = load_atmosphere_continuum_state(regime)
    atmosphere_names = tuple(field.name for field in fields(type(atmosphere_view)))
    return StateProjectionCheckpoint(
        regime=regime,
        synthesis_handoff_field_names=tuple(sorted(synthesis_handoff)),
        synthesis_continuum_field_names=SYNTHESIS_CONTINUUM_FIELDS,
        atmosphere_continuum_field_names=atmosphere_names,
        synthesis_continuum_shapes=tuple(
            np.asarray(synthesis_view[name]).shape
            for name in SYNTHESIS_CONTINUUM_FIELDS
        ),
        atmosphere_continuum_shapes=tuple(
            np.asarray(getattr(atmosphere_view, name)).shape
            for name in atmosphere_names
        ),
        shared_field_names=tuple(
            sorted(set(SYNTHESIS_CONTINUUM_FIELDS).intersection(atmosphere_names))
        ),
    )


def synthesis_stored_h2_invariance_checkpoint(
    regime: str = "cool_molecule_rich",
    wavelength_nm: Sequence[float] = (400.0, 800.0, 1_600.0, 2_500.0),
    scale_factor: float = 1.0e6,
) -> StoredH2InvarianceCheckpoint:
    """Prove that the standard synthesis continuum does not read stored H2."""

    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("scale_factor must be positive and finite")
    wavelength = _positive_finite(
        "wavelength_nm",
        np.asarray(wavelength_nm, dtype=np.float64),
    )
    if wavelength.ndim != 1:
        raise ValueError("wavelength_nm must be one-dimensional")
    state = load_regime_state(regime, "synthesis")
    changed_state = dict(state)
    changed_state["molecular_hydrogen_population"] = (
        scale_factor * state["molecular_hydrogen_population"]
    )
    baseline_absorption, baseline_scattering = _evaluate_synthesis_continuum_state(
        state,
        wavelength,
    )
    changed_absorption, changed_scattering = _evaluate_synthesis_continuum_state(
        changed_state,
        wavelength,
    )
    return StoredH2InvarianceCheckpoint(
        wavelength_nm=wavelength,
        baseline_absorption_cm2_per_g=baseline_absorption,
        changed_absorption_cm2_per_g=changed_absorption,
        baseline_scattering_cm2_per_g=baseline_scattering,
        changed_scattering_cm2_per_g=changed_scattering,
        stored_h2_scale_factor=float(scale_factor),
        bitwise_equal=bool(
            np.array_equal(baseline_absorption, changed_absorption)
            and np.array_equal(baseline_scattering, changed_scattering)
        ),
    )


def numba_timing_checkpoint(
    maximum_threads: int = 4,
) -> NumbaTimingCheckpoint:
    """Measure first-call, warm, cached-process, and thread cases honestly."""

    if isinstance(maximum_threads, bool) or int(maximum_threads) != maximum_threads:
        raise ValueError("maximum_threads must be a positive integer")
    maximum_threads = int(maximum_threads)
    if maximum_threads <= 0:
        raise ValueError("maximum_threads must be a positive integer")

    import json
    import subprocess
    import sys
    import tempfile

    worker = REPOSITORY_ROOT / "scripts" / "chapter05_numba_timing_worker.py"
    with tempfile.TemporaryDirectory(prefix="chapter05-numba-cache-") as temporary:
        cache = Path(temporary)
        environment = os.environ.copy()
        environment.update(
            {
                "LC_ALL": "C",
                "NUMBA_CACHE_DIR": str(cache),
                "NUMBA_NUM_THREADS": str(maximum_threads),
                "OMP_NUM_THREADS": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        local_pythonpath = os.pathsep.join(
            [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")]
        )
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            local_pythonpath
            if not existing_pythonpath
            else os.pathsep.join([local_pythonpath, existing_pythonpath])
        )
        command = [
            sys.executable,
            str(worker),
            "--maximum-threads",
            str(maximum_threads),
        ]
        captures = []
        for _ in range(2):
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            captures.append(json.loads(completed.stdout))
        cache_file_count = sum(path.is_file() for path in cache.rglob("*"))

    fingerprints = {
        str(capture[name])
        for capture in captures
        for name in (
            "python_fingerprint",
            "serial_fingerprint",
            "parallel_one_fingerprint",
            "parallel_many_fingerprint",
        )
    }
    equality_keys = (
        "serial_first_equal",
        "serial_warm_equal",
        "parallel_first_equal",
        "parallel_one_equal",
        "parallel_many_equal",
    )
    all_equal = (
        len(fingerprints) == 1
        and all(bool(capture[name]) for capture in captures for name in equality_keys)
    )
    if cache_file_count == 0:
        raise RuntimeError("Numba timing produced no external cache files")
    if not all_equal:
        raise RuntimeError("Numba timing kernels changed the opacity columns")
    return NumbaTimingCheckpoint(
        first_process=captures[0],
        cached_process=captures[1],
        cache_file_count=cache_file_count,
        output_fingerprint=fingerprints.pop(),
        all_outputs_bitwise_equal=all_equal,
    )


def hminus_edge_checkpoint(
    regime: str = "solar_dwarf",
    relative_offset: float = 1.0e-5,
) -> HMinusEdgeCheckpoint:
    """Separate exact H-minus bound-free and free-free absorption at its edge."""

    if not np.isfinite(relative_offset) or not 0.0 < relative_offset < 1.0:
        raise ValueError("relative_offset must be finite and between zero and one")

    frequency = HMINUS_BOUNDFREE_THRESHOLD_HZ * np.asarray(
        [1.0 - relative_offset, 1.0, 1.0 + relative_offset],
        dtype=np.float64,
    )
    return _hminus_component_checkpoint(regime, frequency)


def hminus_component_checkpoint(
    regime: str,
    wavelength_nm: Sequence[float],
) -> HMinusEdgeCheckpoint:
    """Separate exact H-minus components on a caller wavelength grid."""

    wavelength = _positive_finite(
        "wavelength_nm",
        np.asarray(wavelength_nm, dtype=np.float64),
    )
    if wavelength.ndim != 1:
        raise ValueError("wavelength_nm must be one-dimensional")
    return _hminus_component_checkpoint(
        regime,
        LIGHT_SPEED_NM_PER_S / wavelength,
    )


def _hminus_component_checkpoint(
    regime: str,
    frequency_hz: np.ndarray,
) -> HMinusEdgeCheckpoint:
    """Assemble one exact H-minus decomposition on a validated frequency grid."""

    configure_local_data_paths()
    from payne_zero_atmosphere.continuum_opacity import (
        BOLTZMANN_EV_PER_K_REFERENCE,
        _piecewise_quadratic_remap,
        _planck_frequency_exact,
        compute_hminus_opacity_columns,
        load_continuum_opacity_tables,
    )

    atmosphere = load_atmosphere_continuum_state(regime)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    wavelength = LIGHT_SPEED_NM_PER_S / frequency
    tables = load_continuum_opacity_tables()
    total, _ = compute_hminus_opacity_columns(
        atmosphere,
        frequency,
        continuum_tables=tables,
    )
    freefree_tables = replace(
        tables,
        hminus_boundfree_cross_section_cm2=np.zeros_like(
            tables.hminus_boundfree_cross_section_cm2
        ),
    )
    freefree, _ = compute_hminus_opacity_columns(
        atmosphere,
        frequency,
        continuum_tables=freefree_tables,
    )
    stored_cross_section = np.zeros_like(frequency)
    active = frequency > HMINUS_BOUNDFREE_THRESHOLD_HZ
    stored_cross_section[active] = _piecewise_quadratic_remap(
        tables.hminus_boundfree_wavelength_nm,
        tables.hminus_boundfree_cross_section_cm2,
        wavelength[active],
    )
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    electron_density = np.asarray(atmosphere.electron_density, dtype=np.float64)
    mass_density = np.asarray(atmosphere.mass_density, dtype=np.float64)
    normalized_hydrogen = np.asarray(
        atmosphere.hydrogen_partition_normalized_ion_stage_populations[:, 0],
        dtype=np.float64,
    )
    fixture_hydrogen_departure = np.asarray(
        atmosphere.hydrogen_departure_coefficients[:, 0],
        dtype=np.float64,
    )
    thermal_energy_ev = BOLTZMANN_EV_PER_K_REFERENCE * temperature
    common_population_factor = (
        np.exp(0.754209 / thermal_energy_ev)
        / (2.0 * 2.4148e15 * temperature * np.sqrt(temperature))
        * normalized_hydrogen
        * electron_density
    )
    unit_hminus_population = common_population_factor
    fixture_hminus_population = (
        common_population_factor * fixture_hydrogen_departure
    )
    physical_cross_section = stored_cross_section * 1.0e-18
    _, photon_boltzmann_factor, stimulated_emission = _planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequency,
    )
    transparent_unit_boundfree = (
        physical_cross_section[None, :]
        * stimulated_emission
        * unit_hminus_population[:, None]
        / mass_density[:, None]
    )
    transparent_fixture_boundfree = (
        physical_cross_section[None, :]
        * stimulated_emission
        * fixture_hminus_population[:, None]
        / mass_density[:, None]
    )
    unit_departure_coefficients = np.ones_like(
        atmosphere.hydrogen_departure_coefficients
    )
    unit_state = replace(
        atmosphere,
        hydrogen_departure_coefficients=unit_departure_coefficients,
    )
    unit_total, _ = compute_hminus_opacity_columns(
        unit_state,
        frequency,
        continuum_tables=tables,
    )
    unit_freefree, _ = compute_hminus_opacity_columns(
        unit_state,
        frequency,
        continuum_tables=freefree_tables,
    )
    exact_unit_boundfree = unit_total - unit_freefree
    if not np.all(np.isfinite(photon_boltzmann_factor)):
        raise RuntimeError("H-minus photon Boltzmann factor became nonfinite")
    return HMinusEdgeCheckpoint(
        frequency_hz=frequency,
        wavelength_nm=wavelength,
        stored_boundfree_cross_section_1e_minus_18_cm2=stored_cross_section,
        boundfree_absorption_cm2_per_g=total - freefree,
        freefree_absorption_cm2_per_g=freefree,
        total_absorption_cm2_per_g=total,
        threshold_hz=HMINUS_BOUNDFREE_THRESHOLD_HZ,
        last_table_wavelength_nm=float(
            tables.hminus_boundfree_wavelength_nm[-1]
        ),
        temperature_k=temperature,
        electron_density_cm3=electron_density,
        mass_density_g_cm3=mass_density,
        hydrogen_partition_normalized_population_cm3=normalized_hydrogen,
        fixture_hydrogen_departure_coefficient=fixture_hydrogen_departure,
        unit_hminus_population_cm3=unit_hminus_population,
        fixture_hminus_population_cm3=fixture_hminus_population,
        physical_boundfree_cross_section_cm2=physical_cross_section,
        transparent_unit_boundfree_absorption_cm2_per_g=(
            transparent_unit_boundfree
        ),
        transparent_fixture_boundfree_absorption_cm2_per_g=(
            transparent_fixture_boundfree
        ),
        exact_unit_boundfree_absorption_cm2_per_g=exact_unit_boundfree,
    )


def molecular_continuum_checkpoint(
    regime: str = "cool_molecule_rich",
    wavenumber_cm1: Sequence[float] = (
        1_000.0,
        5_000.0,
        10_000.0,
        19_999.0,
        20_000.0,
        20_001.0,
    ),
) -> MolecularContinuumCheckpoint:
    """Separate exact CH, OH, and H2 collision-induced absorption."""

    configure_local_data_paths()
    from payne_zero_atmosphere.continuum_opacity import (
        compute_continuum_scattering_columns,
        compute_molecular_continuum_opacity_columns,
        compute_molecular_hydrogen_population,
        load_continuum_opacity_tables,
    )

    wavenumber = _positive_finite(
        "wavenumber_cm1",
        np.asarray(wavenumber_cm1, dtype=np.float64),
    )
    if wavenumber.ndim != 1:
        raise ValueError("wavenumber_cm1 must be one-dimensional")
    frequency = wavenumber * 2.99792458e10
    atmosphere = load_atmosphere_continuum_state(regime)
    zero = np.zeros_like(atmosphere.temperature)

    ch_state = replace(
        atmosphere,
        oh_population=zero,
        hydrogen_neutral_population=zero,
    )
    oh_state = replace(
        atmosphere,
        ch_population=zero,
        hydrogen_neutral_population=zero,
    )
    collision_state = replace(
        atmosphere,
        ch_population=zero,
        oh_population=zero,
    )
    ch, _ = compute_molecular_continuum_opacity_columns(ch_state, frequency)
    oh, _ = compute_molecular_continuum_opacity_columns(oh_state, frequency)
    collision, _ = compute_molecular_continuum_opacity_columns(
        collision_state,
        frequency,
    )
    h2h2_state = replace(
        collision_state,
        helium_neutral_population=np.zeros_like(
            atmosphere.helium_neutral_population
        ),
    )
    h2h2, _ = compute_molecular_continuum_opacity_columns(
        h2h2_state,
        frequency,
    )
    h2he = collision - h2h2
    total, _ = compute_molecular_continuum_opacity_columns(
        atmosphere,
        frequency,
    )

    temperature_gate = np.asarray(
        [8999.0, 9000.0, 8999.0, 9000.0, 8999.0, 9000.0],
        dtype=np.float64,
    )
    gate_state = replace(atmosphere, temperature=temperature_gate)
    gate_ch_state = replace(
        gate_state,
        oh_population=zero,
        hydrogen_neutral_population=zero,
    )
    gate_oh_state = replace(
        gate_state,
        ch_population=zero,
        hydrogen_neutral_population=zero,
    )
    # CH and OH begin above roughly two electron volts, so this temperature
    # gate is probed at 30,000 cm^-1 rather than on the CIA-only infrared grid.
    gate_frequency = np.asarray([30_000.0 * 2.99792458e10])
    gate_ch, _ = compute_molecular_continuum_opacity_columns(
        gate_ch_state,
        gate_frequency,
    )
    gate_oh, _ = compute_molecular_continuum_opacity_columns(
        gate_oh_state,
        gate_frequency,
    )

    h2_equilibrium_temperature = np.asarray(
        [100.0, 101.0, 19899.0, 19900.0],
        dtype=np.float64,
    )
    h2_equilibrium_population = compute_molecular_hydrogen_population(
        temperature_k=h2_equilibrium_temperature,
        hydrogen_neutral_partition_normalized_population=np.ones(4),
        hydrogen_departure_coefficient=np.ones(4),
    )

    h2_cutoff_temperature = np.asarray(
        [19999.0, 20000.0, 20001.0, 19999.0, 20000.0, 20001.0],
        dtype=np.float64,
    )
    cutoff_state = replace(atmosphere, temperature=h2_cutoff_temperature)
    cutoff_frequency = np.asarray([1.0e14], dtype=np.float64)
    flags_hydrogen = [0] * 20
    flags_hydrogen[3] = 1
    flags_hydrogen_and_h2 = list(flags_hydrogen)
    flags_hydrogen_and_h2[12] = 1
    cutoff_hydrogen = compute_continuum_scattering_columns(
        cutoff_state,
        cutoff_frequency,
        opacity_flags=flags_hydrogen,
    )
    cutoff_hydrogen_and_h2 = compute_continuum_scattering_columns(
        cutoff_state,
        cutoff_frequency,
        opacity_flags=flags_hydrogen_and_h2,
    )
    h2_rayleigh_at_cutoff = (
        cutoff_hydrogen_and_h2 - cutoff_hydrogen
    )[:, 0]

    cia_temperature = np.asarray([3000.0, 3500.0, 4000.0], dtype=np.float64)
    cia_temperature_index = np.clip(
        np.asarray(cia_temperature / 1000.0, dtype=np.int64),
        1,
        6,
    )
    cia_lower_weight = np.clip(
        (
            cia_temperature
            - 1000.0 * cia_temperature_index.astype(np.float64)
        )
        / 1000.0,
        0.0,
        1.0,
    )
    cia_upper_weight = 1.0 - cia_lower_weight
    continuum_tables = load_continuum_opacity_tables()
    cia_row = 40  # exactly 10,000 cm^-1 on the 250 cm^-1 table grid

    def interpolate_cia_log(table: np.ndarray) -> np.ndarray:
        return (
            table[cia_row, cia_temperature_index - 1] * cia_lower_weight
            + table[cia_row, cia_temperature_index] * cia_upper_weight
        )

    cia_h2h2_log = interpolate_cia_log(
        continuum_tables.hydrogen_molecule_h2_collision_table
    )
    cia_h2he_log = interpolate_cia_log(
        continuum_tables.hydrogen_molecule_he_collision_table
    )

    seam_frequency = np.asarray([10_000.0 * 2.99792458e10])
    all_warm_state = replace(
        atmosphere,
        temperature=np.full(atmosphere.layers, 9000.0),
    )
    all_warm, _ = compute_molecular_continuum_opacity_columns(
        all_warm_state,
        seam_frequency,
    )
    mixed_temperature = np.asarray(
        [8999.0, 9000.0, 10000.0, 15000.0, 20000.0, 20001.0],
        dtype=np.float64,
    )
    mixed_state = replace(atmosphere, temperature=mixed_temperature)
    mixed, _ = compute_molecular_continuum_opacity_columns(
        mixed_state,
        seam_frequency,
    )
    molecule_disabled_state = replace(
        atmosphere,
        ch_population=zero,
        oh_population=zero,
    )
    molecule_disabled, _ = compute_molecular_continuum_opacity_columns(
        molecule_disabled_state,
        seam_frequency,
    )

    flags_h2_only = [0] * 20
    flags_h2_only[12] = 1
    ifop13_alone = compute_continuum_scattering_columns(
        atmosphere,
        cutoff_frequency,
        opacity_flags=flags_h2_only,
    )
    ifop4 = compute_continuum_scattering_columns(
        atmosphere,
        cutoff_frequency,
        opacity_flags=flags_hydrogen,
    )
    ifop4_and13 = compute_continuum_scattering_columns(
        atmosphere,
        cutoff_frequency,
        opacity_flags=flags_hydrogen_and_h2,
    )
    return MolecularContinuumCheckpoint(
        wavenumber_cm1=wavenumber,
        wavelength_nm=1.0e7 / wavenumber,
        ch_absorption_cm2_per_g=ch,
        oh_absorption_cm2_per_g=oh,
        h2h2_absorption_cm2_per_g=h2h2,
        h2he_absorption_cm2_per_g=h2he,
        collision_induced_absorption_cm2_per_g=collision,
        total_absorption_cm2_per_g=total,
        temperature_gate_k=temperature_gate,
        ch_at_temperature_gate_cm2_per_g=gate_ch[:, 0],
        oh_at_temperature_gate_cm2_per_g=gate_oh[:, 0],
        h2_equilibrium_temperature_k=h2_equilibrium_temperature,
        h2_equilibrium_population_cm3=h2_equilibrium_population,
        h2_cutoff_temperature_k=h2_cutoff_temperature,
        h2_rayleigh_at_cutoff_cm2_per_g=h2_rayleigh_at_cutoff,
        cia_temperature_k=cia_temperature,
        cia_lower_column_weight=cia_lower_weight,
        cia_upper_column_weight=cia_upper_weight,
        cia_h2h2_log10_coefficient=cia_h2h2_log,
        cia_h2he_log10_coefficient=cia_h2he_log,
        all_warm_absorption_cm2_per_g=all_warm,
        mixed_column_absorption_cm2_per_g=mixed,
        molecule_disabled_absorption_cm2_per_g=molecule_disabled,
        ifop13_alone_scattering_cm2_per_g=ifop13_alone,
        ifop4_and13_h2_increment_cm2_per_g=ifop4_and13 - ifop4,
    )


def scattering_checkpoint(
    regime: str = "solar_dwarf",
    wavelength_nm: Sequence[float] = (
        200.0,
        300.0,
        400.0,
        500.0,
        700.0,
        1_000.0,
        1_600.0,
        2_500.0,
    ),
) -> ScatteringCheckpoint:
    """Return separately gated atmosphere scattering components."""

    configure_local_data_paths()
    from payne_zero_atmosphere.continuum_opacity import (
        compute_continuum_scattering_columns,
    )

    wavelength = _positive_finite(
        "wavelength_nm",
        np.asarray(wavelength_nm, dtype=np.float64),
    )
    if wavelength.ndim != 1:
        raise ValueError("wavelength_nm must be one-dimensional")
    frequency = LIGHT_SPEED_NM_PER_S / wavelength
    atmosphere = load_atmosphere_continuum_state(regime)

    def isolated(*enabled: int, at_frequency: np.ndarray = frequency) -> np.ndarray:
        flags = [0] * 20
        for index in enabled:
            flags[index] = 1
        return compute_continuum_scattering_columns(
            atmosphere,
            at_frequency,
            opacity_flags=flags,
        )

    electron = isolated(11)
    hydrogen = isolated(3)
    helium = isolated(7)
    molecular_hydrogen = isolated(3, 12) - hydrogen
    total = compute_continuum_scattering_columns(
        atmosphere,
        frequency,
        opacity_flags=RUNNER_OPACITY_FLAGS,
    )

    cap_frequency = np.asarray([2.463e15, 5.15e15, 2.922e15])
    cap_component = np.empty((3, atmosphere.layers), dtype=np.float64)
    above_cap_component = np.empty_like(cap_component)
    component_flags = ((3,), (7,), (3, 12))
    for row, (cap, enabled) in enumerate(zip(cap_frequency, component_flags)):
        probes = np.asarray([cap, 1.1 * cap])
        values = isolated(*enabled, at_frequency=probes)
        if enabled == (3, 12):
            hydrogen_values = isolated(3, at_frequency=probes)
            values = values - hydrogen_values
        cap_component[row] = values[:, 0]
        above_cap_component[row] = values[:, 1]

    return ScatteringCheckpoint(
        wavelength_nm=wavelength,
        electron_scattering_cm2_per_g=electron,
        hydrogen_rayleigh_cm2_per_g=hydrogen,
        helium_rayleigh_cm2_per_g=helium,
        molecular_hydrogen_rayleigh_cm2_per_g=molecular_hydrogen,
        total_scattering_cm2_per_g=total,
        cap_frequency_hz=cap_frequency,
        cap_component_value_cm2_per_g=cap_component,
        above_cap_component_value_cm2_per_g=above_cap_component,
    )


def atmosphere_component_budget(
    regime: str,
    frequency_hz: Sequence[float],
) -> AtmosphereComponentBudgetCheckpoint:
    """Retain every named standard atmosphere absorption component."""

    configure_local_data_paths()
    from payne_zero_atmosphere.continuum_opacity import (
        compute_aluminum_neutral_opacity_columns,
        compute_carbon_neutral_opacity_columns,
        compute_continuum_opacity_columns,
        compute_heminus_opacity_columns,
        compute_helium_ionized_opacity_columns,
        compute_helium_neutral_opacity_columns,
        compute_hminus_opacity_columns,
        compute_hot_metal_opacity_columns,
        compute_hydrogen_opacity_columns,
        compute_iron_neutral_opacity_columns,
        compute_lukewarm_metal_opacity_columns,
        compute_magnesium_neutral_opacity_columns,
        compute_molecular_continuum_opacity_columns,
        compute_molecular_hydrogen_ion_opacity_columns,
        compute_silicon_neutral_opacity_columns,
    )

    frequency = _positive_finite(
        "frequency_hz",
        np.asarray(frequency_hz, dtype=np.float64),
    )
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")
    atmosphere = load_atmosphere_continuum_state(regime)
    component_functions = (
        ("hydrogen", compute_hydrogen_opacity_columns),
        ("hminus", compute_hminus_opacity_columns),
        ("h2plus", compute_molecular_hydrogen_ion_opacity_columns),
        ("helium_neutral", compute_helium_neutral_opacity_columns),
        ("helium_ionized", compute_helium_ionized_opacity_columns),
        ("heminus", compute_heminus_opacity_columns),
        ("molecular", compute_molecular_continuum_opacity_columns),
        ("carbon_neutral", compute_carbon_neutral_opacity_columns),
        ("magnesium_neutral", compute_magnesium_neutral_opacity_columns),
        ("aluminum_neutral", compute_aluminum_neutral_opacity_columns),
        ("silicon_neutral", compute_silicon_neutral_opacity_columns),
        ("iron_neutral", compute_iron_neutral_opacity_columns),
        ("lukewarm_metals", compute_lukewarm_metal_opacity_columns),
        ("hot_metals", compute_hot_metal_opacity_columns),
    )
    absorption_by_name: dict[str, np.ndarray] = {}
    source_by_name: dict[str, np.ndarray] = {}
    for name, function in component_functions:
        component_absorption, component_source = function(atmosphere, frequency)
        absorption_by_name[name] = component_absorption
        source_by_name[name] = component_source

    nonthermal_absorption = (
        absorption_by_name["hydrogen"] + absorption_by_name["hminus"]
    )
    thermal_absorption = np.zeros_like(nonthermal_absorption)
    for name in (
        "h2plus",
        "helium_neutral",
        "helium_ionized",
        "heminus",
        "molecular",
        "carbon_neutral",
        "magnesium_neutral",
        "aluminum_neutral",
        "silicon_neutral",
        "iron_neutral",
        "lukewarm_metals",
        "hot_metals",
    ):
        thermal_absorption += absorption_by_name[name]
    reconstructed_absorption = nonthermal_absorption + thermal_absorption
    source_numerator = (
        absorption_by_name["hydrogen"] * source_by_name["hydrogen"]
        + absorption_by_name["hminus"] * source_by_name["hminus"]
        + thermal_absorption * source_by_name["h2plus"]
    )
    reconstructed_source = source_by_name["h2plus"].copy()
    active = reconstructed_absorption > 0.0
    reconstructed_source[active] = (
        source_numerator[active] / reconstructed_absorption[active]
    )
    exact_absorption, exact_scattering, exact_source = (
        compute_continuum_opacity_columns(
            atmosphere,
            frequency,
            opacity_flags=RUNNER_OPACITY_FLAGS,
        )
    )
    component_names = tuple(name for name, _ in component_functions)
    component_absorption = np.stack(
        [absorption_by_name[name] for name in component_names],
        axis=0,
    )
    component_source = np.stack(
        [source_by_name[name] for name in component_names],
        axis=0,
    )
    component_source_numerator = component_absorption * component_source
    ordered_absorption_partial_sum = np.cumsum(component_absorption, axis=0)
    ordered_source_numerator_partial_sum = np.cumsum(
        component_source_numerator,
        axis=0,
    )
    reconstructed_source_numerator = ordered_source_numerator_partial_sum[-1]
    exact_source_numerator = exact_absorption * exact_source
    return AtmosphereComponentBudgetCheckpoint(
        regime=regime,
        frequency_hz=frequency,
        component_names=component_names,
        component_absorption_cm2_per_g=component_absorption,
        component_source=component_source,
        component_source_numerator=component_source_numerator,
        ordered_absorption_partial_sum_cm2_per_g=(
            ordered_absorption_partial_sum
        ),
        ordered_source_numerator_partial_sum=(
            ordered_source_numerator_partial_sum
        ),
        reconstructed_absorption_cm2_per_g=reconstructed_absorption,
        exact_absorption_cm2_per_g=exact_absorption,
        absorption_residual_cm2_per_g=(
            reconstructed_absorption - exact_absorption
        ),
        exact_scattering_cm2_per_g=exact_scattering,
        reconstructed_source=reconstructed_source,
        exact_source=exact_source,
        reconstructed_source_numerator=reconstructed_source_numerator,
        exact_source_numerator=exact_source_numerator,
        source_numerator_residual=(
            reconstructed_source_numerator - exact_source_numerator
        ),
    )


def metal_population_ownership_checkpoint(
    regime: str = "hot_dwarf",
    frequency_hz: Sequence[float] = (2.0e15, 4.0e15, 8.0e15),
) -> MetalPopulationOwnershipCheckpoint:
    """Double the two physical population views in independent hot-metal limits."""

    configure_local_data_paths()
    import torch

    from payne_zero_atmosphere.continuum_opacity import (
        compute_hot_metal_opacity_columns,
    )
    from payne_zero_synthesis.continuum import build_pops

    frequency = _positive_finite(
        "frequency_hz",
        np.asarray(frequency_hz, dtype=np.float64),
    )
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")
    atmosphere = load_atmosphere_continuum_state(regime)
    zero_actual = np.zeros_like(
        atmosphere.ion_stage_populations_by_packed_slot
    )
    zero_normalized = np.zeros_like(
        atmosphere.partition_normalized_populations_by_packed_slot
    )

    normalized_only_state = replace(
        atmosphere,
        ion_stage_populations_by_packed_slot=zero_actual,
    )
    doubled_normalized_state = replace(
        normalized_only_state,
        partition_normalized_populations_by_packed_slot=(
            2.0
            * atmosphere.partition_normalized_populations_by_packed_slot
        ),
    )
    actual_only_state = replace(
        atmosphere,
        partition_normalized_populations_by_packed_slot=zero_normalized,
    )
    doubled_actual_state = replace(
        actual_only_state,
        ion_stage_populations_by_packed_slot=(
            2.0 * atmosphere.ion_stage_populations_by_packed_slot
        ),
    )
    normalized_only, _ = compute_hot_metal_opacity_columns(
        normalized_only_state,
        frequency,
    )
    doubled_normalized, _ = compute_hot_metal_opacity_columns(
        doubled_normalized_state,
        frequency,
    )
    actual_only, _ = compute_hot_metal_opacity_columns(
        actual_only_state,
        frequency,
    )
    doubled_actual, _ = compute_hot_metal_opacity_columns(
        doubled_actual_state,
        frequency,
    )
    normalized_ratio = np.divide(
        doubled_normalized,
        normalized_only,
        out=np.full_like(normalized_only, np.nan),
        where=normalized_only > 0.0,
    )
    actual_ratio = np.divide(
        doubled_actual,
        actual_only,
        out=np.full_like(actual_only, np.nan),
        where=actual_only > 0.0,
    )

    synthesis_state = project_synthesis_continuum_state(
        load_regime_state(regime, "synthesis")
    )
    baseline_pops = build_pops(
        synthesis_state,
        device="cpu",
        dtype=torch.float64,
    )
    normalized_owner_index = (0, 5)
    doubled_normalized_state = dict(synthesis_state)
    doubled_normalized_cube = np.asarray(
        synthesis_state["partition_normalized_populations"]
    ).copy()
    doubled_normalized_cube[:, normalized_owner_index[0], normalized_owner_index[1]] *= (
        2.0
    )
    doubled_normalized_state["partition_normalized_populations"] = (
        doubled_normalized_cube
    )
    doubled_normalized_pops = build_pops(
        doubled_normalized_state,
        device="cpu",
        dtype=torch.float64,
    )

    actual_owner_index = (1, 5)
    doubled_actual_state = dict(synthesis_state)
    doubled_actual_cube = np.asarray(
        synthesis_state["ion_stage_populations"]
    ).copy()
    expected_actual_delta = doubled_actual_cube[
        :,
        actual_owner_index[0],
        actual_owner_index[1],
    ].copy()
    doubled_actual_cube[:, actual_owner_index[0], actual_owner_index[1]] *= 2.0
    doubled_actual_state["ion_stage_populations"] = doubled_actual_cube
    doubled_actual_pops = build_pops(
        doubled_actual_state,
        device="cpu",
        dtype=torch.float64,
    )

    def host(name: str, pops: Mapping[str, object]) -> np.ndarray:
        return pops[name].detach().cpu().numpy()

    baseline_hot = host("hot_metal_populations", baseline_pops)
    baseline_charge = host("charge_square_population_sum", baseline_pops)
    doubled_normalized_hot = host(
        "hot_metal_populations",
        doubled_normalized_pops,
    )
    doubled_normalized_charge = host(
        "charge_square_population_sum",
        doubled_normalized_pops,
    )
    doubled_actual_hot = host("hot_metal_populations", doubled_actual_pops)
    doubled_actual_charge = host(
        "charge_square_population_sum",
        doubled_actual_pops,
    )
    return MetalPopulationOwnershipCheckpoint(
        frequency_hz=frequency,
        normalized_only_absorption_cm2_per_g=normalized_only,
        doubled_normalized_only_absorption_cm2_per_g=doubled_normalized,
        actual_only_absorption_cm2_per_g=actual_only,
        doubled_actual_only_absorption_cm2_per_g=doubled_actual,
        normalized_ratio=normalized_ratio,
        actual_ratio=actual_ratio,
        synthesis_normalized_owner_index=normalized_owner_index,
        synthesis_actual_owner_index=actual_owner_index,
        baseline_hot_metal_populations=baseline_hot,
        doubled_normalized_hot_metal_populations=doubled_normalized_hot,
        baseline_charge_square_population_sum=baseline_charge,
        doubled_actual_charge_square_population_sum=doubled_actual_charge,
        expected_actual_charge_square_delta=expected_actual_delta,
        normalized_perturbation_preserves_charge_square=bool(
            np.array_equal(baseline_charge, doubled_normalized_charge)
        ),
        actual_perturbation_preserves_hot_metal_populations=bool(
            np.array_equal(baseline_hot, doubled_actual_hot)
        ),
    )


def line_reference_checkpoint(
    regime: str,
    effective_temperature_k: float,
) -> LineReferenceCheckpoint:
    """Run the exact 343/344-point atmosphere line-reference subroute."""

    if not np.isfinite(effective_temperature_k) or effective_temperature_k <= 0.0:
        raise ValueError("effective_temperature_k must be positive and finite")

    configure_local_data_paths()
    from payne_zero_atmosphere.continuum_opacity import (
        BOLTZMANN_ERG_PER_K_REFERENCE,
        LIGHT_SPEED_NM_PER_S as ATMOSPHERE_LIGHT_SPEED_NM_PER_S,
        PLANCK_ERG_SECOND_REFERENCE,
        active_continuum_reference_frequencies,
        assemble_continuum_line_selection_threshold,
        compute_continuum_opacity_columns,
    )

    atmosphere = load_atmosphere_continuum_state(regime)
    active_index, active_frequency = active_continuum_reference_frequencies(
        float(effective_temperature_k)
    )
    absorption, scattering, _ = compute_continuum_opacity_columns(
        atmosphere,
        active_frequency,
        opacity_flags=RUNNER_OPACITY_FLAGS,
    )
    threshold, wavelength, packed = assemble_continuum_line_selection_threshold(
        effective_temperature=float(effective_temperature_k),
        temperature_k=atmosphere.temperature,
        active_continuum_absorption=absorption,
        active_continuum_scattering=scattering,
    )
    inactive_index = np.setdiff1d(
        np.arange(343, dtype=np.int64),
        active_index,
        assume_unique=True,
    )
    if inactive_index.size:
        inactive_frequency = (
            ATMOSPHERE_LIGHT_SPEED_NM_PER_S / wavelength[inactive_index]
        )
        h_over_kt = PLANCK_ERG_SECOND_REFERENCE / (
            BOLTZMANN_ERG_PER_K_REFERENCE * atmosphere.temperature
        )
        stimulated = np.maximum(
            1.0 - np.exp(-np.outer(h_over_kt, inactive_frequency)),
            1.0e-300,
        )
        expected_inactive = np.asarray(1.0e7 / stimulated, dtype=np.float32)
        inactive_residual = (
            threshold[:, inactive_index].astype(np.float64)
            - expected_inactive.astype(np.float64)
        )
        inactive_matches = np.array_equal(
            threshold[:, inactive_index],
            expected_inactive,
        )
        inactive_max_residual = float(np.max(np.abs(inactive_residual)))
    else:
        inactive_matches = True
        inactive_max_residual = 0.0
    return LineReferenceCheckpoint(
        effective_temperature_k=float(effective_temperature_k),
        active_count=int(active_index.size),
        active_index=active_index,
        active_frequency_hz=active_frequency,
        threshold_cm2_per_g=threshold,
        wavelength_nm=wavelength,
        packed_wavelength_index=packed,
        inactive_count=int(343 - active_index.size),
        duplicated_last_column=bool(
            np.array_equal(threshold[:, 343], threshold[:, 342])
        ),
        packed_sentinel=int(packed[343]),
        dtype=str(threshold.dtype),
        inactive_placeholder_matches=bool(inactive_matches),
        inactive_placeholder_max_abs_residual=inactive_max_residual,
    )


def atmosphere_grid_checkpoint() -> AtmosphereGridCheckpoint:
    """Evaluate one representative of every direct-grid branch."""

    configure_local_data_paths()
    from payne_zero_atmosphere.continuum_opacity import (
        active_continuum_reference_frequencies,
        build_opacity_sampling_grid,
    )

    effective_temperature = np.asarray(
        [4499.0, 4500.0, 7250.0, 13000.0, 30000.0],
        dtype=np.float64,
    )
    return _build_atmosphere_grid_checkpoint(
        effective_temperature,
        build_opacity_sampling_grid,
        active_continuum_reference_frequencies,
    )


def atmosphere_grid_boundary_checkpoint() -> AtmosphereGridCheckpoint:
    """Evaluate both sides of all four effective-temperature boundaries."""

    configure_local_data_paths()
    from payne_zero_atmosphere.continuum_opacity import (
        active_continuum_reference_frequencies,
        build_opacity_sampling_grid,
    )

    effective_temperature = np.asarray(
        [4499.0, 4500.0, 7249.0, 7250.0, 12999.0, 13000.0, 29999.0, 30000.0],
        dtype=np.float64,
    )
    return _build_atmosphere_grid_checkpoint(
        effective_temperature,
        build_opacity_sampling_grid,
        active_continuum_reference_frequencies,
    )


def _build_atmosphere_grid_checkpoint(
    effective_temperature: np.ndarray,
    build_grid,
    active_reference,
) -> AtmosphereGridCheckpoint:
    """Assemble one exact grid checkpoint for a declared temperature vector."""

    grids = [
        build_grid(float(value))
        for value in effective_temperature
    ]
    return AtmosphereGridCheckpoint(
        effective_temperature_k=effective_temperature,
        first_wavelength_nm=np.asarray([grid[0][0] for grid in grids]),
        last_wavelength_nm=np.asarray([grid[0][-1] for grid in grids]),
        sample_count=np.asarray([grid[0].size for grid in grids], dtype=np.int64),
        first_frequency_weight_hz=np.asarray([grid[1][0] for grid in grids]),
        interior_frequency_weight_hz=np.asarray([grid[1][15000] for grid in grids]),
        last_frequency_weight_hz=np.asarray([grid[1][-1] for grid in grids]),
        active_reference_count=np.asarray(
            [
                active_reference(float(value))[0].size
                for value in effective_temperature
            ],
            dtype=np.int64,
        ),
    )


def edge_triplet_checkpoint(
    requested_wavelength_nm: Sequence[float],
) -> EdgeTripletCheckpoint:
    """Rebuild exact edge triplets and the corresponding interpolation basis."""

    configure_local_data_paths()
    from payne_zero_synthesis.continuum import build_edge_sample_frequencies

    wavelength = _positive_finite(
        "requested_wavelength_nm",
        np.asarray(requested_wavelength_nm, dtype=np.float64),
    )
    if wavelength.ndim != 1:
        raise ValueError("requested_wavelength_nm must be one-dimensional")
    with np.load(SYNTHESIS_EDGE_GRID, allow_pickle=False) as edge:
        signed_frequency = np.asarray(
            edge["signed_continuum_edge_frequency_hz"],
            dtype=np.float64,
        )
        edge_wavelength = np.asarray(
            edge["continuum_edge_wavelength_nm"],
            dtype=np.float64,
        )
        midpoint = np.asarray(
            edge["continuum_edge_midpoint_wavelength_nm"],
            dtype=np.float64,
        )
        half_width_squared = np.asarray(
            edge["continuum_edge_interval_width_squared_over_two_nm2"],
            dtype=np.float64,
        )
        packaged_samples = np.asarray(
            edge["continuum_edge_sample_frequency_hz"],
            dtype=np.float64,
        )

    rebuilt = build_edge_sample_frequencies(signed_frequency, edge_wavelength)
    flipped = build_edge_sample_frequencies(-signed_frequency, edge_wavelength)
    interval = np.clip(
        np.searchsorted(edge_wavelength, wavelength, side="right") - 1,
        0,
        edge_wavelength.size - 2,
    )
    left = edge_wavelength[interval]
    middle = midpoint[interval]
    right = edge_wavelength[interval + 1]
    denominator = half_width_squared[interval]
    left_basis = (wavelength - middle) * (wavelength - right) / denominator
    midpoint_basis = (
        2.0 * (left - wavelength) * (wavelength - right) / denominator
    )
    right_basis = (wavelength - left) * (wavelength - middle) / denominator
    used = np.unique(interval)
    sample_indices = np.asarray(
        [
            3 * edge_index + offset
            for edge_index in used
            for offset in range(3)
        ],
        dtype=np.int64,
    )
    return EdgeTripletCheckpoint(
        requested_wavelength_nm=wavelength,
        interval_index=interval,
        used_interval_index=used,
        sample_frequency_hz=rebuilt[sample_indices],
        left_basis=left_basis,
        midpoint_basis=midpoint_basis,
        right_basis=right_basis,
        basis_sum=left_basis + midpoint_basis + right_basis,
        packaged_samples_bitwise_equal=np.array_equal(rebuilt, packaged_samples),
        sign_flip_invariant=np.array_equal(rebuilt, flipped),
    )


def edge_use_trace_checkpoint() -> EdgeUseTraceCheckpoint:
    """Observe the standard route calling opacity only for used intervals."""

    configure_local_data_paths()
    import importlib
    from types import SimpleNamespace

    import torch

    continuum_module = importlib.import_module("payne_zero_synthesis.continuum")

    with np.load(SYNTHESIS_EDGE_GRID, allow_pickle=False) as edge:
        edge_wavelength = np.asarray(
            edge["continuum_edge_wavelength_nm"],
            dtype=np.float64,
        )
        signed_frequency = np.asarray(
            edge["signed_continuum_edge_frequency_hz"],
            dtype=np.float64,
        )
        midpoint = np.asarray(
            edge["continuum_edge_midpoint_wavelength_nm"],
            dtype=np.float64,
        )
        width_squared_over_two = np.asarray(
            edge["continuum_edge_interval_width_squared_over_two_nm2"],
            dtype=np.float64,
        )
    exact_edge_index = 100
    requested = np.asarray(
        [
            edge_wavelength[exact_edge_index],
            0.5
            * (
                edge_wavelength[exact_edge_index]
                + edge_wavelength[exact_edge_index + 1]
            ),
            0.5 * (edge_wavelength[220] + edge_wavelength[221]),
        ],
        dtype=np.float64,
    )
    interval = np.clip(
        np.searchsorted(edge_wavelength, requested, side="right") - 1,
        0,
        edge_wavelength.size - 2,
    )
    used = np.unique(interval)
    sample_frequency = continuum_module.build_edge_sample_frequencies(
        signed_frequency,
        edge_wavelength,
    )
    expected_indices = np.asarray(
        [
            3 * edge_index + offset
            for edge_index in used
            for offset in range(3)
        ],
        dtype=np.int64,
    )
    observed: dict[str, object] = {}
    original_compute = continuum_module._compute_at_freqs

    def observe_compute(
        continuum_tables,
        frequencies_hz,
        pops,
        coulomb_table_energy_first=False,
        frequency_invariants=None,
    ):
        del continuum_tables
        frequency = np.asarray(frequencies_hz, dtype=np.float64)
        observed["frequency_hz"] = frequency.copy()
        observed["coulomb_table_energy_first"] = bool(
            coulomb_table_energy_first
        )
        observed["frequency_invariants_was_none"] = frequency_invariants is None
        depth_count = int(pops["temperature"].shape[0])
        return (
            torch.full(
                (depth_count, frequency.size),
                2.0,
                dtype=torch.float64,
            ),
            torch.full(
                (depth_count, frequency.size),
                3.0,
                dtype=torch.float64,
            ),
        )

    atmosphere = {
        "signed_continuum_edge_frequency_hz": signed_frequency,
        "continuum_edge_wavelength_nm": edge_wavelength,
        "continuum_edge_midpoint_wavelength_nm": midpoint,
        "continuum_edge_interval_width_squared_over_two_nm2": (
            width_squared_over_two
        ),
    }
    tables = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float64)
    pops = {"temperature": torch.ones(2, dtype=torch.float64)}
    try:
        continuum_module._compute_at_freqs = observe_compute
        continuum_module.continuum(
            requested,
            atmosphere,
            tables,
            pops=pops,
        )
    finally:
        continuum_module._compute_at_freqs = original_compute

    called = np.asarray(observed["frequency_hz"], dtype=np.float64)
    return EdgeUseTraceCheckpoint(
        requested_wavelength_nm=requested,
        interval_index=interval,
        used_interval_index=used,
        called_frequency_hz=called,
        expected_called_frequency_hz=sample_frequency[expected_indices],
        called_frequency_count=int(called.size),
        total_interval_count=int(edge_wavelength.size - 1),
        unused_interval_count=int(edge_wavelength.size - 1 - used.size),
        exact_internal_edge_nm=float(edge_wavelength[exact_edge_index]),
        exact_internal_edge_assigned_interval=int(interval[0]),
        coulomb_table_energy_first=bool(
            observed["coulomb_table_energy_first"]
        ),
        frequency_invariants_was_none=bool(
            observed["frequency_invariants_was_none"]
        ),
    )


def standard_synthesis_component_checkpoint(
    regime: str = "solar_dwarf",
    requested_wavelength_nm: Sequence[float] = (400.0, 500.0, 650.0, 900.0),
) -> StandardSynthesisComponentCheckpoint:
    """Retain canonical standard-route components before interpolation hides them."""

    configure_local_data_paths()
    import importlib

    import torch

    continuum_module = importlib.import_module("payne_zero_synthesis.continuum")
    requested = _positive_finite(
        "requested_wavelength_nm",
        np.asarray(requested_wavelength_nm, dtype=np.float64),
    )
    if requested.ndim != 1:
        raise ValueError("requested_wavelength_nm must be one-dimensional")
    state = project_synthesis_continuum_state(
        load_regime_state(regime, "synthesis")
    )
    tables = continuum_module.ContinuumTables.from_npz(
        SYNTHESIS_CONTINUUM_TABLES,
        device="cpu",
        dtype=torch.float64,
    )
    pops = continuum_module.build_pops(
        state,
        device="cpu",
        dtype=torch.float64,
    )
    edge_wavelength = np.asarray(
        state["continuum_edge_wavelength_nm"],
        dtype=np.float64,
    )
    interval = np.clip(
        np.searchsorted(edge_wavelength, requested, side="right") - 1,
        0,
        edge_wavelength.size - 2,
    )
    used = np.unique(interval)
    all_sample_frequency = continuum_module.build_edge_sample_frequencies(
        state["signed_continuum_edge_frequency_hz"],
        edge_wavelength,
    )
    sample_index = np.asarray(
        [
            3 * edge_index + offset
            for edge_index in used
            for offset in range(3)
        ],
        dtype=np.int64,
    )
    sample_frequency = all_sample_frequency[sample_index]

    recorded: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "hminus",
            "hydrogen",
            "minor",
            "helium_neutral",
            "helium_ionized",
            "hot_metals",
            "silicon_singly_ionized",
            "hydrogen_rayleigh",
            "electron_scattering",
            "minor_scattering",
            "minor_h2plus",
            "minor_heminus",
            "minor_carbon",
            "minor_magnesium",
            "minor_aluminum",
            "minor_silicon",
            "minor_helium_rayleigh",
            "minor_h2_rayleigh",
        )
    }

    def keep(name: str, value) -> None:
        recorded[name].append(value.detach().cpu().numpy().copy())

    originals = {
        name: getattr(continuum_module, name)
        for name in (
            "_hminus_opacity",
            "_hydrogen_opacity",
            "_minor_terms",
            "_helium_opacity",
            "_hot_metal_and_silicon_singly_ionized_opacity",
            "_scattering_opacity",
        )
    }

    def isolate_pops(base: Mapping[str, object], keep_keys: set[str]) -> dict:
        isolated = dict(base)
        owned_keys = {
            "hydrogen_partition_normalized_ion_stage_populations",
            "helium_neutral_population",
            "carbon_partition_normalized_ion_stage_populations",
            "magnesium_neutral_partition_normalized_population",
            "aluminum_neutral_partition_normalized_population",
            "silicon_neutral_partition_normalized_population",
        }
        for key in owned_keys - keep_keys:
            isolated[key] = torch.zeros_like(base[key])
        return isolated

    def wrap_single(record_name: str, source_name: str):
        source = originals[source_name]

        def wrapped(*args, **kwargs):
            value = source(*args, **kwargs)
            keep(record_name, value)
            return value

        return wrapped

    original_minor = originals["_minor_terms"]

    def wrapped_minor(*args, **kwargs):
        value = original_minor(*args, **kwargs)
        keep("minor", value[0])
        keep("minor_scattering", value[1])
        base_pops = args[2]
        absorption_isolations = (
            (
                "minor_h2plus",
                {"hydrogen_partition_normalized_ion_stage_populations"},
            ),
            ("minor_heminus", {"helium_neutral_population"}),
            (
                "minor_carbon",
                {"carbon_partition_normalized_ion_stage_populations"},
            ),
            (
                "minor_magnesium",
                {"magnesium_neutral_partition_normalized_population"},
            ),
            (
                "minor_aluminum",
                {"aluminum_neutral_partition_normalized_population"},
            ),
            (
                "minor_silicon",
                {"silicon_neutral_partition_normalized_population"},
            ),
        )
        for name, keep_keys in absorption_isolations:
            isolated_args = list(args)
            isolated_args[2] = isolate_pops(base_pops, keep_keys)
            isolated_value = original_minor(*isolated_args, **kwargs)
            keep(name, isolated_value[0])
        helium_args = list(args)
        helium_pops = dict(base_pops)
        helium_pops["hydrogen_neutral_population"] = torch.zeros_like(
            base_pops["hydrogen_neutral_population"]
        )
        helium_args[2] = helium_pops
        keep(
            "minor_helium_rayleigh",
            original_minor(*helium_args, **kwargs)[1],
        )
        h2_args = list(args)
        h2_pops = dict(base_pops)
        h2_pops["helium_neutral_population"] = torch.zeros_like(
            base_pops["helium_neutral_population"]
        )
        h2_args[2] = h2_pops
        keep("minor_h2_rayleigh", original_minor(*h2_args, **kwargs)[1])
        return value

    def wrapped_helium(*args, **kwargs):
        value = originals["_helium_opacity"](*args, **kwargs)
        keep("helium_neutral", value[0])
        keep("helium_ionized", value[1])
        return value

    def wrapped_hot(*args, **kwargs):
        value = originals[
            "_hot_metal_and_silicon_singly_ionized_opacity"
        ](*args, **kwargs)
        keep("hot_metals", value[0])
        keep("silicon_singly_ionized", value[1])
        return value

    def wrapped_scattering(*args, **kwargs):
        value = originals["_scattering_opacity"](*args, **kwargs)
        keep("hydrogen_rayleigh", value[0])
        keep("electron_scattering", value[1])
        return value

    continuum_module._hminus_opacity = wrap_single(
        "hminus",
        "_hminus_opacity",
    )
    continuum_module._hydrogen_opacity = wrap_single(
        "hydrogen",
        "_hydrogen_opacity",
    )
    continuum_module._minor_terms = wrapped_minor
    continuum_module._helium_opacity = wrapped_helium
    continuum_module._hot_metal_and_silicon_singly_ionized_opacity = wrapped_hot
    continuum_module._scattering_opacity = wrapped_scattering
    try:
        exact_sampled_absorption_t, exact_sampled_scattering_t = (
            continuum_module._compute_at_freqs(
                tables,
                sample_frequency,
                pops,
                coulomb_table_energy_first=False,
                frequency_invariants=None,
            )
        )
    finally:
        for name, source in originals.items():
            setattr(continuum_module, name, source)

    def grid(name: str) -> np.ndarray:
        return np.stack(recorded[name], axis=1)

    absorption_names = (
        "hminus",
        "hydrogen",
        "minor",
        "helium_neutral",
        "helium_ionized",
        "hot_metals",
        "silicon_singly_ionized",
    )
    absorption_components = np.stack(
        [grid(name) for name in absorption_names],
        axis=0,
    )
    absorption_partial = np.cumsum(absorption_components, axis=0)
    exact_sampled_absorption = (
        exact_sampled_absorption_t.detach().cpu().numpy()
    )
    scattering_names = (
        "hydrogen_rayleigh",
        "electron_scattering",
        "minor_scattering",
    )
    scattering_components = np.stack(
        [grid(name) for name in scattering_names],
        axis=0,
    )
    scattering_partial = np.cumsum(scattering_components, axis=0)
    exact_sampled_scattering = (
        exact_sampled_scattering_t.detach().cpu().numpy()
    )
    minor_absorption_names = (
        "h2plus",
        "heminus",
        "carbon",
        "magnesium",
        "aluminum",
        "silicon",
    )
    minor_absorption_components = np.stack(
        [
            grid(f"minor_{name}")
            for name in minor_absorption_names
        ],
        axis=0,
    )
    minor_scattering_names = ("helium_rayleigh", "h2_rayleigh")
    minor_scattering_components = np.stack(
        [
            grid(f"minor_{name}")
            for name in minor_scattering_names
        ],
        axis=0,
    )
    final_absorption_t, final_scattering_t = continuum_module.continuum(
        requested,
        state,
        tables,
        pops=pops,
    )

    def host_pop(name: str) -> np.ndarray:
        return pops[name].detach().cpu().numpy().copy()

    return StandardSynthesisComponentCheckpoint(
        regime=regime,
        requested_wavelength_nm=requested,
        used_interval_index=used,
        sample_frequency_hz=sample_frequency,
        build_pops_hydrogen_ionized_population=host_pop(
            "hydrogen_ionized_population"
        ),
        build_pops_hot_metal_populations=host_pop("hot_metal_populations"),
        build_pops_charge_square_population_sum=host_pop(
            "charge_square_population_sum"
        ),
        absorption_component_names=absorption_names,
        absorption_components_cm2_per_g=absorption_components,
        absorption_partial_sums_cm2_per_g=absorption_partial,
        exact_sampled_absorption_cm2_per_g=exact_sampled_absorption,
        absorption_residual_cm2_per_g=(
            absorption_partial[-1] - exact_sampled_absorption
        ),
        scattering_component_names=scattering_names,
        scattering_components_cm2_per_g=scattering_components,
        scattering_partial_sums_cm2_per_g=scattering_partial,
        exact_sampled_scattering_cm2_per_g=exact_sampled_scattering,
        scattering_residual_cm2_per_g=(
            scattering_partial[-1] - exact_sampled_scattering
        ),
        minor_absorption_subcomponent_names=minor_absorption_names,
        minor_absorption_subcomponents_cm2_per_g=minor_absorption_components,
        minor_scattering_subcomponent_names=minor_scattering_names,
        minor_scattering_subcomponents_cm2_per_g=minor_scattering_components,
        final_absorption_cm2_per_g=final_absorption_t.detach().cpu().numpy(),
        final_scattering_cm2_per_g=final_scattering_t.detach().cpu().numpy(),
        coulomb_table_energy_first=False,
        frequency_invariants_was_none=True,
    )


def edge_reconstruction_checkpoint(
    regime: str = "solar_dwarf",
    wavelength_near_nm: float = 500.0,
    sample_count: int = 240,
) -> EdgeReconstructionCheckpoint:
    """Reconstruct one exact standard interval from its three opacity samples."""

    if not np.isfinite(wavelength_near_nm) or wavelength_near_nm <= 0.0:
        raise ValueError("wavelength_near_nm must be positive and finite")
    if isinstance(sample_count, bool) or int(sample_count) != sample_count:
        raise ValueError("sample_count must be an integer of at least three")
    sample_count = int(sample_count)
    if sample_count < 3:
        raise ValueError("sample_count must be an integer of at least three")

    configure_local_data_paths()
    import importlib

    import torch

    from book.chapter05_teaching import (
        reconstruct_positive_opacity,
        three_point_edge_basis,
    )

    continuum_module = importlib.import_module("payne_zero_synthesis.continuum")
    state = project_synthesis_continuum_state(
        load_regime_state(regime, "synthesis")
    )
    tables = continuum_module.ContinuumTables.from_npz(
        SYNTHESIS_CONTINUUM_TABLES,
        device="cpu",
        dtype=torch.float64,
    )
    pops = continuum_module.build_pops(
        state,
        device="cpu",
        dtype=torch.float64,
    )
    edge_wavelength = np.asarray(
        state["continuum_edge_wavelength_nm"],
        dtype=np.float64,
    )
    edge_index = int(
        np.clip(
            np.searchsorted(edge_wavelength, wavelength_near_nm, side="right")
            - 1,
            0,
            edge_wavelength.size - 2,
        )
    )
    left = float(edge_wavelength[edge_index])
    right = float(edge_wavelength[edge_index + 1])
    midpoint = float(
        state["continuum_edge_midpoint_wavelength_nm"][edge_index]
    )
    all_sample_frequency = continuum_module.build_edge_sample_frequencies(
        state["signed_continuum_edge_frequency_hz"],
        edge_wavelength,
    )
    sample_frequency = all_sample_frequency[
        3 * edge_index : 3 * edge_index + 3
    ]
    absorption_samples_t, scattering_samples_t = (
        continuum_module._compute_at_freqs(
            tables,
            sample_frequency,
            pops,
        )
    )
    absorption_samples = absorption_samples_t.detach().cpu().numpy()
    scattering_samples = scattering_samples_t.detach().cpu().numpy()
    target = np.linspace(
        left,
        np.nextafter(right, left),
        sample_count,
        dtype=np.float64,
    )
    basis = three_point_edge_basis(target, left, right)
    reconstructed_absorption = reconstruct_positive_opacity(
        absorption_samples[:, None, :],
        basis,
    )
    reconstructed_scattering = reconstruct_positive_opacity(
        scattering_samples[:, None, :],
        basis,
    )
    exact_absorption_t, exact_scattering_t = continuum_module.continuum(
        target,
        state,
        tables,
        pops=pops,
    )
    return EdgeReconstructionCheckpoint(
        interval_index=edge_index,
        left_wavelength_nm=left,
        midpoint_wavelength_nm=midpoint,
        right_wavelength_nm=right,
        sample_frequency_hz=sample_frequency,
        sample_wavelength_nm=LIGHT_SPEED_NM_PER_S / sample_frequency,
        target_wavelength_nm=target,
        absorption_samples_cm2_per_g=absorption_samples,
        scattering_samples_cm2_per_g=scattering_samples,
        reconstructed_absorption_cm2_per_g=reconstructed_absorption,
        reconstructed_scattering_cm2_per_g=reconstructed_scattering,
        exact_absorption_cm2_per_g=exact_absorption_t.detach().cpu().numpy(),
        exact_scattering_cm2_per_g=exact_scattering_t.detach().cpu().numpy(),
        basis_sum=basis.sum,
    )


def _evaluate_synthesis_continuum_state(
    state: Mapping[str, np.ndarray],
    wavelength_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the exact standard 18-field synthesis view on CPU float64."""

    configure_local_data_paths()
    import torch

    from payne_zero_synthesis.continuum import ContinuumTables, continuum

    tables = ContinuumTables.from_npz(
        SYNTHESIS_CONTINUUM_TABLES,
        device="cpu",
        dtype=torch.float64,
    )
    continuum_state = project_synthesis_continuum_state(state)
    absorption, scattering = continuum(wavelength_nm, continuum_state, tables)
    return (
        absorption.detach().cpu().numpy(),
        scattering.detach().cpu().numpy(),
    )


def run_synthesis_continuum(
    regime: str,
    wavelength_nm: Sequence[float],
) -> ContinuumRouteCheckpoint:
    """Run the standard synthesis edge-triplet product on CPU float64."""

    wavelength = _positive_finite(
        "wavelength_nm",
        np.asarray(wavelength_nm, dtype=np.float64),
    )
    if wavelength.ndim != 1:
        raise ValueError("wavelength_nm must be one-dimensional")
    state = load_regime_state(regime, "synthesis")
    absorption, scattering = _evaluate_synthesis_continuum_state(
        state,
        wavelength,
    )
    return ContinuumRouteCheckpoint(
        regime=regime,
        wavelength_nm=wavelength,
        absorption_cm2_per_g=absorption,
        scattering_cm2_per_g=scattering,
        continuum_opacity=absorption + scattering,
        continuum_source=None,
        route="standard synthesis edge triplets",
        dtype="torch.float64",
    )


def run_sampled_synthesis_continuum(
    regime: str,
    wavelength_nm: Sequence[float],
    *,
    precomputed: bool,
) -> SampledSynthesisCheckpoint:
    """Run a wavelength-labelled diagnostic or precomputed extension."""

    wavelength = _positive_finite(
        "wavelength_nm",
        np.asarray(wavelength_nm, dtype=np.float64),
    )
    if wavelength.ndim != 1:
        raise ValueError("wavelength_nm must be one-dimensional")
    support_bounds = (
        float(SAMPLED_EXTENSION_SUPPORTED_WAVELENGTH_NM[0]),
        float(SAMPLED_EXTENSION_SUPPORTED_WAVELENGTH_NM[-1]),
    )
    supported_wavelength = np.asarray(
        SAMPLED_EXTENSION_SUPPORTED_WAVELENGTH_NM,
        dtype=np.float64,
    )
    if precomputed and not np.all(np.isin(wavelength, supported_wavelength)):
        raise ValueError(
            "precomputed sampled extension is validated only at the declared "
            "twelve-point wavelength vector spanning "
            f"{support_bounds[0]:g} to {support_bounds[1]:g} nm"
        )
    frequency = LIGHT_SPEED_NM_PER_S / wavelength
    return _run_sampled_synthesis_continuum(
        regime,
        wavelength,
        frequency,
        precomputed=precomputed,
    )


def run_sampled_synthesis_continuum_at_frequency(
    regime: str,
    frequency_hz: Sequence[float],
) -> SampledSynthesisCheckpoint:
    """Run the sampled diagnostic without a wavelength round trip.

    This frequency-native teaching boundary is needed for exact nextafter
    probes at component thresholds.  The precomputed extension remains
    wavelength-labelled because its accepted support is an exact twelve-point
    wavelength vector.
    """

    frequency = _positive_finite(
        "frequency_hz",
        np.asarray(frequency_hz, dtype=np.float64),
    )
    if frequency.ndim != 1:
        raise ValueError("frequency_hz must be one-dimensional")
    wavelength = LIGHT_SPEED_NM_PER_S / frequency
    return _run_sampled_synthesis_continuum(
        regime,
        wavelength,
        frequency,
        precomputed=False,
    )


def _run_sampled_synthesis_continuum(
    regime: str,
    wavelength: np.ndarray,
    frequency: np.ndarray,
    *,
    precomputed: bool,
) -> SampledSynthesisCheckpoint:
    """Evaluate one already-validated sampled-continuum coordinate pair."""

    configure_local_data_paths()
    import torch

    from payne_zero_synthesis.continuum import (
        ContinuumTables,
        build_frequency_invariants,
        build_pops,
        compute_sampled_continuum,
    )

    support_bounds = (
        float(SAMPLED_EXTENSION_SUPPORTED_WAVELENGTH_NM[0]),
        float(SAMPLED_EXTENSION_SUPPORTED_WAVELENGTH_NM[-1]),
    )
    supported_wavelength = np.asarray(
        SAMPLED_EXTENSION_SUPPORTED_WAVELENGTH_NM,
        dtype=np.float64,
    )
    state = project_synthesis_continuum_state(
        load_regime_state(regime, "synthesis")
    )
    tables = ContinuumTables.from_npz(
        SYNTHESIS_CONTINUUM_TABLES,
        device="cpu",
        dtype=torch.float64,
    )
    pops = build_pops(state, device="cpu", dtype=torch.float64)
    invariants = (
        build_frequency_invariants(
            tables,
            frequency,
            coulomb_table_energy_first=True,
        )
        if precomputed
        else None
    )
    absorption, scattering, source = compute_sampled_continuum(
        tables,
        frequency,
        pops,
        invariants,
    )
    source_host = source.detach().cpu().numpy()
    temperature = np.asarray(state["temperature"], dtype=np.float64)[:, None]
    frequency_grid = frequency[None, :]
    light_speed_cm_per_s = LIGHT_SPEED_NM_PER_S * 1.0e-7
    photon_energy_over_kt = (
        PLANCK_ERG_SECOND
        * frequency_grid
        / (BOLTZMANN_ERG_PER_K * temperature)
    )
    direct_source = (
        2.0
        * PLANCK_ERG_SECOND
        * frequency_grid**3
        / light_speed_cm_per_s**2
        / np.expm1(photon_energy_over_kt)
    )
    source_residual = source_host - direct_source
    invariant_shapes: dict[str, tuple[int, ...]] = {}
    cache_entries = 0
    if invariants is not None:
        invariant_shapes = {
            field.name: np.asarray(getattr(invariants, field.name)).shape
            for field in fields(type(invariants))
            if field.name != "_tensor_cache"
            and isinstance(getattr(invariants, field.name), np.ndarray)
        }
        first_tensor = invariants.tensor("frequencies_hz", torch.float64, "cpu")
        second_tensor = invariants.tensor("frequencies_hz", torch.float64, "cpu")
        if first_tensor.data_ptr() != second_tensor.data_ptr():
            raise RuntimeError("FrequencyInvariants did not reuse its tensor view")
        cache_entries = len(invariants._tensor_cache)
    return SampledSynthesisCheckpoint(
        regime=regime,
        wavelength_nm=wavelength,
        frequency_hz=frequency,
        absorption_cm2_per_g=absorption.detach().cpu().numpy(),
        scattering_cm2_per_g=scattering.detach().cpu().numpy(),
        source_bnu=source_host,
        direct_source_bnu=direct_source,
        source_residual=source_residual,
        source_matches_direct=bool(
            np.allclose(source_host, direct_source, rtol=3.0e-15, atol=0.0)
        ),
        route=(
            "sampled precomputed extension"
            if precomputed
            else "sampled diagnostic"
        ),
        coulomb_table_energy_first=True,
        frequency_invariant_shapes=invariant_shapes,
        tensor_cache_entries_after_reuse=cache_entries,
        supported_wavelength_nm=(
            supported_wavelength.copy() if precomputed else None
        ),
        supported_wavelength_bounds_nm=support_bounds if precomputed else None,
        dtype=str(absorption.dtype),
    )


def run_atmosphere_continuum(
    regime: str,
    frequency_hz: Sequence[float],
) -> ContinuumRouteCheckpoint:
    """Run the exact atmosphere product on a compact diagnostic grid."""

    configure_local_data_paths()
    from payne_zero_atmosphere.continuum_opacity import (
        ContinuumAtmosphereState,
        compute_continuum_opacity_columns,
    )

    frequency = _positive_finite(
        "frequency_hz",
        np.asarray(frequency_hz, dtype=np.float64),
    )
    raw_state = load_regime_state(regime, "atmosphere")
    allowed = {field.name for field in fields(ContinuumAtmosphereState)}
    state = ContinuumAtmosphereState(
        **{name: values for name, values in raw_state.items() if name in allowed}
    )
    absorption, scattering, source = compute_continuum_opacity_columns(
        state,
        frequency,
        opacity_flags=RUNNER_OPACITY_FLAGS,
    )
    wavelength = 2.99792458e17 / frequency
    return ContinuumRouteCheckpoint(
        regime=regime,
        wavelength_nm=wavelength,
        absorption_cm2_per_g=absorption,
        scattering_cm2_per_g=scattering,
        continuum_opacity=absorption + scattering,
        continuum_source=source,
        route="atmosphere direct frequency columns",
        dtype=str(absorption.dtype),
    )


def run_full_atmosphere_continuum(
    regime: str,
    effective_temperature_k: float,
) -> ContinuumRouteCheckpoint:
    """Run the exact atmosphere product on its direct 30,000-point grid."""

    if not np.isfinite(effective_temperature_k) or effective_temperature_k <= 0.0:
        raise ValueError("effective_temperature_k must be positive and finite")
    configure_local_data_paths()
    from payne_zero_atmosphere.continuum_opacity import (
        build_opacity_sampling_grid,
        compute_continuum_opacity_columns,
    )

    atmosphere = load_atmosphere_continuum_state(regime)
    wavelength, _ = build_opacity_sampling_grid(float(effective_temperature_k))
    frequency = LIGHT_SPEED_NM_PER_S / wavelength
    absorption, scattering, source = compute_continuum_opacity_columns(
        atmosphere,
        frequency,
        opacity_flags=RUNNER_OPACITY_FLAGS,
    )
    return ContinuumRouteCheckpoint(
        regime=regime,
        wavelength_nm=wavelength,
        absorption_cm2_per_g=absorption,
        scattering_cm2_per_g=scattering,
        continuum_opacity=absorption + scattering,
        continuum_source=source,
        route="atmosphere direct 30,000-point product",
        dtype=str(absorption.dtype),
    )


def state_shapes(state: Mapping[str, np.ndarray]) -> dict[str, tuple[int, ...]]:
    """Return exact field shapes for compact reader-facing inspections."""

    return {name: np.asarray(value).shape for name, value in state.items()}


__all__ = [
    "AtmosphereComponentBudgetCheckpoint",
    "AtmosphereGridCheckpoint",
    "ContinuumRouteCheckpoint",
    "ContinuumTablePreflight",
    "EdgeTripletCheckpoint",
    "EdgeUseTraceCheckpoint",
    "EdgeReconstructionCheckpoint",
    "HMinusEdgeCheckpoint",
    "LineReferenceCheckpoint",
    "MetalPopulationOwnershipCheckpoint",
    "MolecularContinuumCheckpoint",
    "NumbaTimingCheckpoint",
    "OpacityScalingCheckpoint",
    "RUNNER_OPACITY_FLAGS",
    "ScatteringCheckpoint",
    "SampledSynthesisCheckpoint",
    "StandardSynthesisComponentCheckpoint",
    "StateProjectionCheckpoint",
    "StoredH2InvarianceCheckpoint",
    "SYNTHESIS_CONTINUUM_FIELDS",
    "SAMPLED_EXTENSION_SUPPORTED_WAVELENGTH_NM",
    "StimulatedEmissionCheckpoint",
    "atmosphere_grid_checkpoint",
    "atmosphere_grid_boundary_checkpoint",
    "atmosphere_component_budget",
    "configure_local_data_paths",
    "continuum_table_preflight",
    "edge_triplet_checkpoint",
    "edge_reconstruction_checkpoint",
    "edge_use_trace_checkpoint",
    "hminus_edge_checkpoint",
    "hminus_component_checkpoint",
    "hminus_table_preflight",
    "line_reference_checkpoint",
    "load_atmosphere_continuum_state",
    "load_regime_state",
    "mass_opacity_from_cross_section",
    "metal_population_ownership_checkpoint",
    "molecular_continuum_checkpoint",
    "numba_timing_checkpoint",
    "opacity_scaling_checkpoint",
    "project_synthesis_continuum_state",
    "run_atmosphere_continuum",
    "run_full_atmosphere_continuum",
    "run_synthesis_continuum",
    "run_sampled_synthesis_continuum",
    "run_sampled_synthesis_continuum_at_frequency",
    "scattering_checkpoint",
    "standard_synthesis_component_checkpoint",
    "state_projection_checkpoint",
    "state_shapes",
    "stimulated_emission_checkpoint",
    "stimulated_emission_factor",
    "synthesis_stored_h2_invariance_checkpoint",
]
