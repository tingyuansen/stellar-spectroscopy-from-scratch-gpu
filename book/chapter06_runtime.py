"""Progressive Chapter 6 helpers for one ordinary atomic line.

The reader-facing notebook first builds the physics with small NumPy
expressions, then calls the exact staged atmosphere and synthesis kernels at
their production boundaries.  No helper in this module opens a golden file.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEACHING_LINE_SUBSET = (
    REPOSITORY_ROOT / "data/subsets/chapter06_fe_i_source_row_873702.npz"
)
CHAPTER05_CONTINUUM_FIXTURE = (
    REPOSITORY_ROOT / "data/fixtures/chapter05_continuum_states.npz"
)
ATMOSPHERE_LINE_TABLES = (
    REPOSITORY_ROOT / "data/static/atmosphere_tables/line_opacity_tables.npz"
)
SYNTHESIS_LINE_TABLES = (
    REPOSITORY_ROOT / "data/static/synthesis_tables/line_profile_tables.npz"
)
ATMOSPHERE_ONE_LINE_FIXTURE = (
    REPOSITORY_ROOT / "data/fixtures/chapter06_atmosphere_one_line_inputs.npz"
)
ATMOSPHERE_ONE_LINE_FIXTURE_SHA256 = (
    "1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff"
)

LIGHT_SPEED_NM_PER_S = 2.99792458e17
PLANCK_ERG_SECOND = 6.62607015e-27
BOLTZMANN_ERG_PER_K = 1.380649e-16
CLASSICAL_INTEGRATED_LINE_COEFFICIENT = 0.026538
SQRT_PI_REFERENCE = 1.77245
DAMPING_NORMALIZATION = 12.5664
LINE_CENTER_CUTOFF_RATIO = 1.0e-3

RAW_LINE_FIELDS = (
    "stored_wavelength_nm",
    "raw_log_oscillator_strength",
    "species_code",
    "first_energy_column_cm",
    "second_energy_column_cm",
    "radiative_damping_log",
    "stark_damping_log",
    "van_der_waals_damping_log",
    "lower_principal_quantum_number",
    "upper_principal_quantum_number",
    "primary_isotope_number",
    "primary_isotope_log_correction",
    "secondary_isotope_log_correction",
    "energy_shift_field",
    "isotope_shift_units",
    "line_size",
    "line_category_tag",
)


@dataclass(frozen=True)
class TransitionCheckpoint:
    """A two-level energy difference converted into one source record."""

    lower_excitation_cm: float
    upper_excitation_cm: float
    energy_separation_cm: float
    stored_wavelength_nm: float
    wavelength_nm: float
    stored_minus_derived_nm: float
    log_oscillator_strength: float
    oscillator_strength: float
    atomic_number: int
    ion_stage: int
    line_type: int


@dataclass(frozen=True)
class FastExponentialCheckpoint:
    """The two exact table contracts beside the mathematical exponential."""

    exponent_argument: np.ndarray
    exact_exp_minus_x: np.ndarray
    atmosphere_lookup: np.ndarray
    atmosphere_compiled_lookup: np.ndarray
    synthesis_float64_lookup: np.ndarray
    synthesis_float32_lookup: np.ndarray
    atmosphere_minus_exact: np.ndarray
    synthesis_float64_minus_exact: np.ndarray
    synthesis_float32_minus_exact: np.ndarray


@dataclass(frozen=True)
class StrengthCheckpoint:
    """Factor ledger for the gross, pre-stimulated line amplitude."""

    oscillator_strength: float
    partition_normalized_population_cm3: np.ndarray
    lower_level_boltzmann_factor: np.ndarray
    excitation_weighted_partition_normalized_population_cm3: np.ndarray
    gf_weighted_excitation_factor_cm3: np.ndarray
    mass_density_g_cm3: np.ndarray
    fractional_doppler_width: np.ndarray
    integrated_strength_cm2_hz_per_g: np.ndarray
    line_amplitude_cm2_per_g: np.ndarray


@dataclass(frozen=True)
class StoredDopplerCheckpoint:
    """Unit conversions for widths already computed and owned by Chapter 3."""

    temperature_k_context: np.ndarray
    fractional_doppler_width: np.ndarray
    doppler_width_nm: np.ndarray
    doppler_width_km_per_s: np.ndarray
    width_source: str


@dataclass(frozen=True)
class DampingCheckpoint:
    """Radiative, electron, and neutral contributions to the damping ratio."""

    radiative_term: np.ndarray
    stark_term: np.ndarray
    van_der_waals_term: np.ndarray
    total_damping_fraction: np.ndarray
    fractional_doppler_width: np.ndarray
    damping_ratio: np.ndarray


@dataclass(frozen=True)
class ProfileNormalizationCheckpoint:
    """Finite-domain evidence that a Voigt profile redistributes fixed area."""

    damping_ratio: np.ndarray
    integration_limit_doppler_widths: float
    measured_integral_h_du: np.ndarray
    exact_integral_h_du: np.ndarray
    measured_integral_phi_dnu: np.ndarray
    exact_integral_phi_dnu: np.ndarray
    relative_missing_area: np.ndarray


@dataclass(frozen=True)
class DenseLineCheckpoint:
    """A readable one-line calculation before production cutoff/deposition."""

    wavelength_nm: np.ndarray
    doppler_offset: np.ndarray
    profile_h: np.ndarray
    profile_contract: str
    gross_line_mass_absorption_coefficient: np.ndarray
    stimulated_emission_factor: np.ndarray
    net_line_mass_absorption_coefficient: np.ndarray
    line_center_wavelength_nm: float
    depth_index: int


@dataclass(frozen=True)
class SynthesisOneLineCheckpoint:
    """The exact one-record synthesis lane on one six-depth atmosphere."""

    regime: str
    wavelength_nm: np.ndarray
    continuum_opacity_cm2_per_g: np.ndarray
    gross_line_mass_absorption_tensor: object
    net_line_mass_absorption_tensor: object
    gross_line_mass_absorption_coefficient: np.ndarray
    net_line_mass_absorption_coefficient: np.ndarray
    activity_mask: np.ndarray
    nonzero_count: np.ndarray
    wing_reach: np.ndarray
    metal_center_index: int
    metal_wing_index: int
    work_dtype: str
    accumulation_dtype: str
    cutoff_dtype: str
    stimulation_dtype: str
    device: str
    metal_line_count: int
    auto_line_count: int
    helium_line_count: int
    population_ion_stage_index: int
    population_element_index: int
    wing_mode: str


@dataclass(frozen=True)
class AtmosphereOneLineCheckpoint:
    """The exact staged CPU atmosphere lane for one selected Fe I record."""

    effective_temperature: float
    wavelength_nm: np.ndarray
    temperature: np.ndarray
    pre_stimulated_line_mass_absorption_coefficient: np.ndarray
    post_stimulated_line_mass_absorption_coefficient: np.ndarray
    selected_line_count: int
    nonzero_count_per_depth: np.ndarray
    peak_pre_stimulated_cm2_per_g: float
    accumulation_dtype: str
    device: str
    stimulation_owner: str


@dataclass(frozen=True)
class ProductionCenterPolicyCheckpoint:
    """Every exact synthesis factor and branch at the one-line center."""

    regime: str
    depth_index: np.ndarray
    excitation_exponent: np.ndarray
    direct_exp_weight: np.ndarray
    fast_ex_weight: np.ndarray
    pre_excitation_strength_cm2_per_g: np.ndarray
    post_fastex_line_amplitude_cm2_per_g: np.ndarray
    center_cutoff_cm2_per_g: np.ndarray
    passes_pre_excitation_cutoff: np.ndarray
    passes_post_fastex_cutoff: np.ndarray
    damping_ratio: np.ndarray
    full_harris_center_profile: np.ndarray
    shortcut_center_profile: np.ndarray
    selected_center_profile: np.ndarray
    selected_branch: np.ndarray
    float32_center_deposit_cm2_per_g: np.ndarray
    production_center_deposit_cm2_per_g: np.ndarray
    host_classical_line_strength_cm2: float
    float32_classical_line_strength_cm2: float


@dataclass(frozen=True)
class HarrisBranchCheckpoint:
    """Scalar/compiled Harris values and the synthesis ordinary-wing policy."""

    doppler_offset: np.ndarray
    damping_ratio: np.ndarray
    atmosphere_scalar_profile: np.ndarray
    atmosphere_compiled_profile: np.ndarray
    atmosphere_compiled_float64_profile: np.ndarray
    synthesis_full_profile: np.ndarray
    synthesis_scalar_reference: np.ndarray
    synthesis_ordinary_wing_profile: np.ndarray
    synthesis_wing_branch: np.ndarray


def configure_local_data_paths() -> None:
    """Prefer this book's staged source and point it at immutable local data."""

    source_path = (REPOSITORY_ROOT / "src").resolve()
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith(
            ("payne_zero_atmosphere", "payne_zero_synthesis")
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        if not Path(module_file).resolve().is_relative_to(source_path):
            raise RuntimeError(
                f"{module_name} was imported from outside the textbook's staged "
                f"source tree: {module_file}"
            )

    source_root = str(source_path)
    if source_root in sys.path:
        sys.path.remove(source_root)
    sys.path.insert(0, source_root)
    static_root = REPOSITORY_ROOT / "data/static"
    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(static_root)
    os.environ["PAYNE_ZERO_ATMOSPHERE_DATA_ROOT"] = str(
        static_root / "atmosphere_tables"
    )
    os.environ["PAYNE_ZERO_SYNTHESIS_DATA_ROOT"] = str(static_root / "synthesis_tables")


def load_teaching_line_raw() -> dict[str, np.ndarray]:
    """Load the one immutable raw source row, never a computed line product."""

    if not TEACHING_LINE_SUBSET.is_file():
        raise FileNotFoundError(
            "The manifest-bound Chapter 6 teaching subset is missing: "
            f"{TEACHING_LINE_SUBSET}"
        )
    with np.load(TEACHING_LINE_SUBSET, allow_pickle=False) as archive:
        missing = sorted(set(RAW_LINE_FIELDS).difference(archive.files))
        if missing:
            raise ValueError(
                "The Chapter 6 teaching subset is missing raw fields: "
                + ", ".join(missing)
            )
        raw = {name: np.asarray(archive[name]).copy() for name in RAW_LINE_FIELDS}
    for name, values in raw.items():
        if values.shape != (1,):
            raise ValueError(
                f"Chapter 6 raw field {name!r} has shape {values.shape}; "
                "expected one source row"
            )
    return raw


@lru_cache(maxsize=1)
def _canonical_teaching_line_record() -> dict[str, np.ndarray]:
    """Cache a private read-only source transformation."""

    configure_local_data_paths()
    from payne_zero_synthesis.atomic_lines import _build_records

    record = {}
    for name, values in _build_records(load_teaching_line_raw()).items():
        frozen = np.asarray(values).copy()
        frozen.flags.writeable = False
        record[name] = frozen
    return record


def teaching_line_record() -> dict[str, np.ndarray]:
    """Return defensive copies of the exact transformed teaching row."""

    return {
        name: values.copy()
        for name, values in _canonical_teaching_line_record().items()
    }


def transition_checkpoint() -> TransitionCheckpoint:
    """Return the causal two-level-to-record conversion."""

    raw = load_teaching_line_raw()
    record = teaching_line_record()
    first = abs(float(raw["first_energy_column_cm"][0]))
    second = abs(float(raw["second_energy_column_cm"][0]))
    lower = min(first, second)
    upper = max(first, second)
    separation = upper - lower
    wavelength = float(record["wavelength_nm"][0])
    stored = float(raw["stored_wavelength_nm"][0])
    return TransitionCheckpoint(
        lower_excitation_cm=lower,
        upper_excitation_cm=upper,
        energy_separation_cm=separation,
        stored_wavelength_nm=stored,
        wavelength_nm=wavelength,
        stored_minus_derived_nm=stored - wavelength,
        log_oscillator_strength=float(record["log_oscillator_strength"][0]),
        oscillator_strength=float(record["oscillator_strength"][0]),
        atomic_number=int(record["atomic_number"][0]),
        ion_stage=int(record["ion_stage"][0]),
        line_type=int(record["line_type"][0]),
    )


def fast_exponential_checkpoint(
    exponent_argument: np.ndarray | None = None,
) -> FastExponentialCheckpoint:
    """Evaluate the exact atmosphere and synthesis FASTEX policies."""

    configure_local_data_paths()
    from payne_zero_atmosphere.line_profile_math import (
        build_fast_exponential_tables,
        fast_exponential_lookup,
    )
    from payne_zero_atmosphere.line_opacity import (
        _fast_exponential_lookup_compiled,
    )
    from payne_zero_synthesis.line_opacity import (
        _fastex_tables,
        fast_ex,
    )
    import torch

    if exponent_argument is None:
        exponent_argument = np.asarray(
            [
                -1.0,
                0.0,
                0.0005,
                0.2385,
                1.0,
                12.3455,
                1000.999,
                1001.0,
                np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )
    argument = np.asarray(exponent_argument, dtype=np.float64)
    atmosphere_tables = build_fast_exponential_tables()
    atmosphere = np.asarray(
        [
            fast_exponential_lookup(float(value), atmosphere_tables)
            for value in argument
        ],
        dtype=np.float64,
    )
    atmosphere_compiled = np.asarray(
        [
            _fast_exponential_lookup_compiled(
                float(value),
                atmosphere_tables.integer_step,
                atmosphere_tables.fractional_step,
            )
            for value in argument
        ],
        dtype=np.float64,
    )
    synthesis_integer_f64, synthesis_fraction_f64 = _fastex_tables(
        torch.device("cpu"), torch.float64
    )
    synthesis_float64 = (
        fast_ex(
            torch.as_tensor(argument, dtype=torch.float64),
            synthesis_integer_f64,
            synthesis_fraction_f64,
        )
        .cpu()
        .numpy()
    )
    synthesis_integer_f32, synthesis_fraction_f32 = _fastex_tables(
        torch.device("cpu"), torch.float32
    )
    synthesis_float32 = (
        fast_ex(
            torch.as_tensor(argument, dtype=torch.float32),
            synthesis_integer_f32,
            synthesis_fraction_f32,
        )
        .cpu()
        .numpy()
    )
    with np.errstate(over="ignore", invalid="ignore"):
        exact = np.exp(-argument)
    return FastExponentialCheckpoint(
        exponent_argument=argument,
        exact_exp_minus_x=exact,
        atmosphere_lookup=atmosphere,
        atmosphere_compiled_lookup=atmosphere_compiled,
        synthesis_float64_lookup=synthesis_float64,
        synthesis_float32_lookup=synthesis_float32,
        atmosphere_minus_exact=atmosphere - exact,
        synthesis_float64_minus_exact=synthesis_float64 - exact,
        synthesis_float32_minus_exact=synthesis_float32.astype(np.float64) - exact,
    )


def lower_level_boltzmann_factor(
    lower_excitation_cm: float,
    hc_over_kt_cm: np.ndarray,
) -> np.ndarray:
    """Return ``exp(-E_l hc/kT)`` with an explicitly dimensionless exponent."""

    exponent = float(lower_excitation_cm) * np.asarray(hc_over_kt_cm, dtype=np.float64)
    return np.exp(-exponent)


def gross_line_strength_checkpoint(
    *,
    partition_normalized_population_cm3: np.ndarray,
    mass_density_g_cm3: np.ndarray,
    fractional_doppler_width: np.ndarray,
    hc_over_kt_cm: np.ndarray,
    oscillator_strength: float | None = None,
    lower_excitation_cm: float | None = None,
) -> StrengthCheckpoint:
    """Build the analytic direct-exponential line-strength reference."""

    transition = transition_checkpoint()
    gf = (
        transition.oscillator_strength
        if oscillator_strength is None
        else float(oscillator_strength)
    )
    excitation = (
        transition.lower_excitation_cm
        if lower_excitation_cm is None
        else float(lower_excitation_cm)
    )
    normalized_population = np.asarray(
        partition_normalized_population_cm3, dtype=np.float64
    )
    density = np.asarray(mass_density_g_cm3, dtype=np.float64)
    doppler_fraction = np.asarray(fractional_doppler_width, dtype=np.float64)
    boltzmann = lower_level_boltzmann_factor(excitation, hc_over_kt_cm)
    excitation_weighted_population = normalized_population * boltzmann
    gf_weighted_excitation_factor = gf * excitation_weighted_population
    valid = (
        np.isfinite(excitation_weighted_population)
        & np.isfinite(density)
        & np.isfinite(doppler_fraction)
        & (excitation_weighted_population >= 0.0)
        & (density > 0.0)
        & (doppler_fraction > 0.0)
    )
    gf_weighted_factor_per_mass = np.divide(
        gf_weighted_excitation_factor,
        density,
        out=np.zeros_like(excitation_weighted_population),
        where=valid,
    )
    integrated = CLASSICAL_INTEGRATED_LINE_COEFFICIENT * gf_weighted_factor_per_mass
    frequency_hz = LIGHT_SPEED_NM_PER_S / transition.wavelength_nm
    amplitude = np.divide(
        CLASSICAL_INTEGRATED_LINE_COEFFICIENT
        / SQRT_PI_REFERENCE
        / frequency_hz
        * gf_weighted_excitation_factor,
        density * doppler_fraction,
        out=np.zeros_like(excitation_weighted_population),
        where=valid,
    )
    return StrengthCheckpoint(
        oscillator_strength=gf,
        partition_normalized_population_cm3=normalized_population,
        lower_level_boltzmann_factor=boltzmann,
        excitation_weighted_partition_normalized_population_cm3=(
            excitation_weighted_population
        ),
        gf_weighted_excitation_factor_cm3=gf_weighted_excitation_factor,
        mass_density_g_cm3=density,
        fractional_doppler_width=doppler_fraction,
        integrated_strength_cm2_hz_per_g=integrated,
        line_amplitude_cm2_per_g=amplitude,
    )


def stored_doppler_checkpoint(
    *,
    temperature_k: np.ndarray,
    fractional_doppler_width: np.ndarray,
) -> StoredDopplerCheckpoint:
    """Convert the stored dimensionless width without recomputing Chapter 3."""

    temperature = np.asarray(temperature_k, dtype=np.float64)
    fraction = np.asarray(fractional_doppler_width, dtype=np.float64)
    wavelength = transition_checkpoint().wavelength_nm
    return StoredDopplerCheckpoint(
        temperature_k_context=temperature,
        fractional_doppler_width=fraction,
        doppler_width_nm=wavelength * fraction,
        doppler_width_km_per_s=2.99792458e5 * fraction,
        width_source="supplied Chapter 3 fractional_doppler_widths",
    )


def damping_checkpoint(
    *,
    electron_density_cm3: np.ndarray,
    collision_density_proxy_cm3: np.ndarray,
    fractional_doppler_width: np.ndarray,
) -> DampingCheckpoint:
    """Form the three production damping terms and their dimensionless ratio."""

    record = teaching_line_record()
    electron_density = np.asarray(electron_density_cm3, dtype=np.float64)
    collision_density = np.asarray(collision_density_proxy_cm3, dtype=np.float64)
    doppler_fraction = np.asarray(fractional_doppler_width, dtype=np.float64)
    radiative = np.full_like(
        doppler_fraction,
        float(record["radiative_damping"][0]),
        dtype=np.float64,
    )
    stark = float(record["stark_damping"][0]) * electron_density
    van_der_waals = float(record["van_der_waals_damping"][0]) * collision_density
    total = radiative + stark + van_der_waals
    ratio = np.divide(
        total,
        doppler_fraction,
        out=np.zeros_like(total),
        where=doppler_fraction > 0.0,
    )
    return DampingCheckpoint(
        radiative_term=radiative,
        stark_term=stark,
        van_der_waals_term=van_der_waals,
        total_damping_fraction=total,
        fractional_doppler_width=doppler_fraction,
        damping_ratio=ratio,
    )


def stimulated_emission_factor(
    wavelength_nm: np.ndarray | float,
    temperature_k: np.ndarray | float,
) -> np.ndarray:
    """Return the one LTE gross-to-net line-opacity factor."""

    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    frequency_hz = LIGHT_SPEED_NM_PER_S / wavelength
    exponent = PLANCK_ERG_SECOND * frequency_hz / (BOLTZMANN_ERG_PER_K * temperature)
    return -np.expm1(-exponent)


def continuous_voigt_h(
    doppler_offset: np.ndarray,
    damping_ratio: float,
    *,
    integration_half_width: float = 10.0,
) -> np.ndarray:
    r"""Evaluate the defining Gaussian–Lorentz convolution for \(H(a,u)\).

    This is deliberately a transparent reference, not the production Harris
    approximation.  The integration step resolves the Lorentzian half-width
    with at least twelve samples for the damping ratios used in the chapter.
    """

    offset = np.asarray(doppler_offset, dtype=np.float64)
    damping = float(damping_ratio)
    if damping < 0.0 or not np.isfinite(damping):
        raise ValueError("damping_ratio must be finite and nonnegative")
    if damping == 0.0:
        return np.exp(-(offset * offset))
    step = min(0.01, max(damping / 12.0, 2.5e-4))
    integration_grid = np.arange(
        -float(integration_half_width),
        float(integration_half_width) + 0.5 * step,
        step,
        dtype=np.float64,
    )
    gaussian_weight = np.exp(-(integration_grid * integration_grid))
    flat_offset = offset.reshape(-1)
    result = np.empty_like(flat_offset)
    chunk_size = 128
    for start in range(0, flat_offset.size, chunk_size):
        stop = min(start + chunk_size, flat_offset.size)
        separation = flat_offset[start:stop, None] - integration_grid[None, :]
        integrand = gaussian_weight[None, :] / (
            separation * separation + damping * damping
        )
        result[start:stop] = (
            damping / np.pi * np.trapezoid(integrand, integration_grid, axis=1)
        )
    return result.reshape(offset.shape)


def profile_normalization_checkpoint(
    damping_ratio: np.ndarray | None = None,
    *,
    integration_limit_doppler_widths: float = 400.0,
) -> ProfileNormalizationCheckpoint:
    """Measure the finite-domain area of the defining Voigt convolution."""

    damping = (
        np.asarray([0.0, 0.05, 0.2, 1.0], dtype=np.float64)
        if damping_ratio is None
        else np.asarray(damping_ratio, dtype=np.float64)
    )
    limit = float(integration_limit_doppler_widths)
    if not np.isfinite(limit) or limit <= 12.0:
        raise ValueError(
            "integration_limit_doppler_widths must be finite and greater than 12"
        )
    # A dense core plus logarithmically expanding wings measures both parts
    # without pretending that a finite wavelength window contains all wings.
    positive_core = np.linspace(0.0, 12.0, 24_001, dtype=np.float64)
    positive_wing = np.geomspace(
        np.nextafter(12.0, np.inf),
        limit,
        12_000,
        dtype=np.float64,
    )
    positive = np.concatenate((positive_core, positive_wing))
    measured = np.empty_like(damping)
    for index, value in enumerate(damping):
        profile = continuous_voigt_h(positive, float(value))
        measured[index] = 2.0 * np.trapezoid(profile, positive)
    exact = np.full_like(damping, np.sqrt(np.pi))
    return ProfileNormalizationCheckpoint(
        damping_ratio=damping,
        integration_limit_doppler_widths=limit,
        measured_integral_h_du=measured,
        exact_integral_h_du=exact,
        measured_integral_phi_dnu=measured / np.sqrt(np.pi),
        exact_integral_phi_dnu=np.ones_like(damping),
        relative_missing_area=(exact - measured) / exact,
    )


def atmosphere_harris_profile(
    doppler_offset: np.ndarray,
    damping_ratio: np.ndarray | float,
) -> np.ndarray:
    """Evaluate the exact scalar atmosphere Harris authority elementwise."""

    configure_local_data_paths()
    from payne_zero_atmosphere.line_profile_math import (
        build_voigt_profile_basis,
        evaluate_voigt_profile,
        load_line_opacity_tables,
    )

    load_line_opacity_tables(ATMOSPHERE_LINE_TABLES, force_reload=True)
    build_voigt_profile_basis.cache_clear()
    basis = build_voigt_profile_basis()
    offset, damping = np.broadcast_arrays(
        np.abs(np.asarray(doppler_offset, dtype=np.float64)),
        np.asarray(damping_ratio, dtype=np.float64),
    )
    result = np.empty_like(offset)
    for index in np.ndindex(offset.shape):
        result[index] = evaluate_voigt_profile(
            float(offset[index]), float(damping[index]), basis
        )
    return result


def synthesis_harris_profile(
    doppler_offset: np.ndarray,
    damping_ratio: np.ndarray | float,
) -> np.ndarray:
    """Evaluate the exact full synthesis Harris authority on CPU float64."""

    configure_local_data_paths()
    from payne_zero_synthesis.line_opacity import interpolate_harris_profile
    import torch

    with np.load(SYNTHESIS_LINE_TABLES, allow_pickle=False) as archive:
        h0 = torch.as_tensor(
            np.asarray(archive["harris_profile_h0_table"]),
            dtype=torch.float64,
        )
        h1 = torch.as_tensor(
            np.asarray(archive["harris_profile_h1_table"]),
            dtype=torch.float64,
        )
        h2 = torch.as_tensor(
            np.asarray(archive["harris_profile_h2_table"]),
            dtype=torch.float64,
        )
    offset, damping = np.broadcast_arrays(
        np.asarray(doppler_offset, dtype=np.float64),
        np.asarray(damping_ratio, dtype=np.float64),
    )
    result = interpolate_harris_profile(
        torch.as_tensor(offset, dtype=torch.float64),
        torch.as_tensor(damping, dtype=torch.float64),
        h0,
        h1,
        h2,
    )
    return result.cpu().numpy()


def _synthesis_harris_scalar(
    doppler_offset: float,
    damping_ratio: float,
    h0_table: np.ndarray,
    h1_table: np.ndarray,
    h2_table: np.ndarray,
) -> float:
    """Independent scalar transcription of the full synthesis Harris rule."""

    offset = abs(float(doppler_offset))
    damping = float(damping_ratio)
    table_index = int(offset * 200.0 + 0.5)
    table_index = min(max(table_index, 0), h0_table.size - 1)
    h0 = float(h0_table[table_index])
    h1 = float(h1_table[table_index])
    h2 = float(h2_table[table_index])
    damping_squared = damping * damping
    offset_squared = offset * offset
    if damping < 0.2:
        if offset > 10.0:
            return 0.5642 * damping / offset_squared
        return (h2 * damping + h1) * damping + h0
    if damping > 1.4 or damping + offset > 3.2:
        denominator = (damping_squared + offset_squared) * 1.4142
        base = damping * 0.79788 / denominator
        if damping > 100.0:
            return base
        damping_fraction = damping_squared / denominator
        offset_fraction = offset_squared / denominator
        correction = (
            (damping_fraction - 10.0 * offset_fraction) * damping_fraction * 3.0
            + 15.0 * offset_fraction * offset_fraction
            + 3.0 * offset_squared
            - damping_squared
        )
        return (correction / (denominator * denominator) + 1.0) * base

    blend_1 = h1 + h0 * 1.12838
    blend_2 = h2 + blend_1 * 1.12838 - h0
    blend_3 = (
        (1.0 - h2) * 0.37613 - blend_1 * 0.66667 * offset_squared + blend_2 * 1.12838
    )
    blend_4 = (
        3.0 * blend_3 - blend_1
    ) * 0.37613 + h0 * 0.66667 * offset_squared * offset_squared
    polynomial = (
        ((blend_4 * damping + blend_3) * damping + blend_2) * damping + blend_1
    ) * damping + h0
    scale = (
        (-0.122727278 * damping + 0.532770573) * damping - 0.96284325
    ) * damping + 0.979895032
    return polynomial * scale


def harris_branch_checkpoint(
    doppler_offset: np.ndarray,
    damping_ratio: np.ndarray,
) -> HarrisBranchCheckpoint:
    """Evaluate both authorities and the exact synthesis wing shortcut."""

    configure_local_data_paths()
    from payne_zero_atmosphere.line_opacity import (
        _voigt_profile_compiled,
        _voigt_profile_f64_compiled,
    )
    from payne_zero_atmosphere.line_profile_math import (
        build_voigt_profile_basis,
        load_line_opacity_tables,
    )
    from payne_zero_synthesis.line_opacity import (
        harris_wing_walk_profile,
        precompute_invariants,
    )
    import torch

    offset, damping = np.broadcast_arrays(
        np.asarray(doppler_offset, dtype=np.float64),
        np.asarray(damping_ratio, dtype=np.float64),
    )
    absolute_offset = np.abs(offset)
    load_line_opacity_tables(ATMOSPHERE_LINE_TABLES, force_reload=True)
    build_voigt_profile_basis.cache_clear()
    basis = build_voigt_profile_basis()
    atmosphere_scalar = atmosphere_harris_profile(absolute_offset, damping)
    atmosphere_compiled = np.empty_like(absolute_offset)
    atmosphere_compiled_f64 = np.empty_like(absolute_offset)
    for index in np.ndindex(absolute_offset.shape):
        arguments = (
            float(absolute_offset[index]),
            float(damping[index]),
            basis.gaussian_profile,
            basis.first_correction,
            basis.second_correction,
        )
        atmosphere_compiled[index] = _voigt_profile_compiled(*arguments)
        atmosphere_compiled_f64[index] = _voigt_profile_f64_compiled(*arguments)

    with np.load(SYNTHESIS_LINE_TABLES, allow_pickle=False) as archive:
        h0 = np.asarray(archive["harris_profile_h0_table"], dtype=np.float64)
        h1 = np.asarray(archive["harris_profile_h1_table"], dtype=np.float64)
        h2 = np.asarray(archive["harris_profile_h2_table"], dtype=np.float64)
    synthesis_full = synthesis_harris_profile(offset, damping)
    synthesis_scalar = np.empty_like(offset)
    for index in np.ndindex(offset.shape):
        synthesis_scalar[index] = _synthesis_harris_scalar(
            float(offset[index]),
            float(damping[index]),
            h0,
            h1,
            h2,
        )

    invariants = precompute_invariants(
        one_line_synthesis_mapping(),
        build_synthesis_wavelength_grid(),
        runtime_device=torch.device("cpu"),
    )
    offset_tensor = torch.as_tensor(offset, dtype=torch.float64)
    damping_tensor = torch.as_tensor(damping, dtype=torch.float64)
    low_damping = damping_tensor < 0.2
    ordinary_wing = harris_wing_walk_profile(
        offset_tensor,
        damping_tensor,
        invariants,
        low_damping,
    )
    branch = np.where(
        damping < 0.2,
        np.where(
            absolute_offset > 10.0,
            "low-damping u^-2 wing",
            "low-damping H0+aH1 wing",
        ),
        "full Harris wing",
    )
    return HarrisBranchCheckpoint(
        doppler_offset=offset.copy(),
        damping_ratio=damping.copy(),
        atmosphere_scalar_profile=atmosphere_scalar,
        atmosphere_compiled_profile=atmosphere_compiled,
        atmosphere_compiled_float64_profile=atmosphere_compiled_f64,
        synthesis_full_profile=synthesis_full,
        synthesis_scalar_reference=synthesis_scalar,
        synthesis_ordinary_wing_profile=ordinary_wing.cpu().numpy().copy(),
        synthesis_wing_branch=np.asarray(branch),
    )


def dense_line_checkpoint(
    *,
    wavelength_nm: np.ndarray,
    line_amplitude_cm2_per_g: float,
    fractional_doppler_width: float,
    damping_ratio: float,
    temperature_k: float,
    depth_index: int,
    profile_lane: str = "synthesis",
) -> DenseLineCheckpoint:
    """Evaluate one readable line on a dense local wavelength grid."""

    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    center = transition_checkpoint().wavelength_nm
    doppler_width_nm = center * float(fractional_doppler_width)
    if doppler_width_nm <= 0.0:
        raise ValueError("fractional_doppler_width must be positive")
    offset = np.abs(wavelength - center) / doppler_width_nm
    if profile_lane == "synthesis":
        profile = synthesis_harris_profile(offset, float(damping_ratio))
        profile_contract = "full synthesis Harris H(a,u) reference"
    elif profile_lane == "atmosphere":
        profile = atmosphere_harris_profile(offset, float(damping_ratio))
        profile_contract = "atmosphere Harris H(a,u) reference"
    elif profile_lane == "continuous":
        profile = continuous_voigt_h(offset, float(damping_ratio))
        profile_contract = "defining continuous Voigt H(a,u) reference"
    else:
        raise ValueError(
            "profile_lane must be 'synthesis', 'atmosphere', or 'continuous'"
        )
    gross = float(line_amplitude_cm2_per_g) * profile
    stimulated = stimulated_emission_factor(wavelength, float(temperature_k))
    return DenseLineCheckpoint(
        wavelength_nm=wavelength,
        doppler_offset=offset,
        profile_h=profile,
        profile_contract=profile_contract,
        gross_line_mass_absorption_coefficient=gross,
        stimulated_emission_factor=stimulated,
        net_line_mass_absorption_coefficient=gross * stimulated,
        line_center_wavelength_nm=center,
        depth_index=int(depth_index),
    )


def collision_density_proxy(
    state: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Rebuild the exact synthesis neutral-collision proxy from named fields."""

    temperature = np.asarray(state["temperature"], dtype=np.float64)
    hydrogen = np.asarray(state["hydrogen_neutral_population"], dtype=np.float64)
    helium = np.asarray(state["helium_neutral_population"], dtype=np.float64)
    molecular_hydrogen = np.asarray(
        state["molecular_hydrogen_population"], dtype=np.float64
    )
    return (hydrogen + 0.42 * helium + 0.85 * molecular_hydrogen) * (
        temperature / 10_000.0
    ) ** 0.3


def build_synthesis_wavelength_grid(
    *,
    start_wavelength_nm: float = 495.0,
    end_wavelength_nm: float = 505.0,
    resolution: float = 300_000.0,
) -> np.ndarray:
    """Build the exact geometric synthesis grid for the one-line chapter."""

    configure_local_data_paths()
    from payne_zero_synthesis.atomic_lines import Grid

    return Grid(
        start_wavelength_nm=float(start_wavelength_nm),
        end_wavelength_nm=float(end_wavelength_nm),
        resolution=float(resolution),
    ).build()


def one_line_synthesis_mapping() -> dict[str, np.ndarray]:
    """Return one physical record plus the exact constructor support fields."""

    physical_keys = (
        "line_type",
        "atomic_number",
        "ion_stage",
        "wavelength_nm",
        "index_wavelength_nm",
        "oscillator_strength",
        "lower_excitation_cm",
        "radiative_damping",
        "stark_damping",
        "van_der_waals_damping",
        "raw_radiative_damping_log",
        "raw_stark_damping_log",
        "raw_van_der_waals_damping_log",
    )
    record = teaching_line_record()
    mapping = {name: np.asarray(record[name]).copy() for name in physical_keys}
    with np.load(SYNTHESIS_LINE_TABLES, allow_pickle=False) as archive:
        mapping.update(
            helium_line_type=np.zeros(0, dtype=np.int64),
            helium_line_center_cutoff_ratio=np.asarray(
                LINE_CENTER_CUTOFF_RATIO,
                dtype=np.float64,
            ),
            harris_profile_h0_table=np.asarray(
                archive["harris_profile_h0_table"],
                dtype=np.float64,
            ).copy(),
            harris_profile_h1_table=np.asarray(
                archive["harris_profile_h1_table"],
                dtype=np.float64,
            ).copy(),
            harris_profile_h2_table=np.asarray(
                archive["harris_profile_h2_table"],
                dtype=np.float64,
            ).copy(),
        )
    return mapping


def synthesis_line_state(
    regime: str,
    wavelength_nm: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compose the existing state with a freshly recomputed continuum cutoff."""

    from book.chapter05_runtime import (
        load_regime_state,
        run_synthesis_continuum,
    )

    state = load_regime_state(regime, "synthesis")
    continuum = run_synthesis_continuum(regime, wavelength_nm)
    line_state = {
        "partition_normalized_populations": np.asarray(
            state["partition_normalized_populations"], dtype=np.float64
        ),
        "fractional_doppler_widths": np.asarray(
            state["fractional_doppler_widths"], dtype=np.float64
        ),
        "mass_density": np.asarray(state["mass_density"], dtype=np.float64),
        "electron_density": np.asarray(state["electron_density"], dtype=np.float64),
        "temperature": np.asarray(state["temperature"], dtype=np.float64),
        "hc_over_kt": np.asarray(state["hc_over_kt"], dtype=np.float64),
        "collision_density_proxy": collision_density_proxy(state),
        "continuum_opacity": np.asarray(continuum.continuum_opacity, dtype=np.float64),
    }
    return line_state, np.asarray(continuum.continuum_opacity, dtype=np.float64)


def _one_line_reach(
    line_mass_absorption_coefficient: np.ndarray,
    wing_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nonzero counts and the farthest populated offset per depth."""

    slab = np.asarray(line_mass_absorption_coefficient)
    nonzero_count = np.count_nonzero(slab, axis=1).astype(np.int64)
    reach = np.zeros(slab.shape[0], dtype=np.int64)
    for depth_index in range(slab.shape[0]):
        columns = np.flatnonzero(slab[depth_index] != 0.0)
        if columns.size:
            reach[depth_index] = int(np.max(np.abs(columns - int(wing_index))))
    return nonzero_count, reach


def run_synthesis_one_line(
    regime: str,
    *,
    wavelength_nm: np.ndarray | None = None,
    wing_mode: str = "batched",
    runtime_device: object = "cpu",
) -> SynthesisOneLineCheckpoint:
    """Run the exact ordinary Fe I synthesis path on one Torch device.

    This function computes both gross and net slabs before any comparison
    artifact is opened. The general constructor still observes its empty
    autoionizing and helium routes.
    """

    configure_local_data_paths()
    from payne_zero_synthesis.line_opacity import (
        accumulate_atomic,
        precompute_invariants,
    )
    import torch

    if wing_mode not in {"batched", "loop"}:
        raise ValueError("wing_mode must be 'batched' or 'loop'")
    compute_device = torch.device(runtime_device)
    wavelength = (
        build_synthesis_wavelength_grid()
        if wavelength_nm is None
        else np.asarray(wavelength_nm, dtype=np.float64)
    )
    state, continuum = synthesis_line_state(regime, wavelength)
    invariants = precompute_invariants(
        one_line_synthesis_mapping(),
        wavelength,
        runtime_device=compute_device,
    )
    gross_tensor = accumulate_atomic(
        invariants,
        state,
        do_metal=True,
        do_helium=False,
        apply_stim=False,
        wing_mode=wing_mode,
    )
    net_tensor = accumulate_atomic(
        invariants,
        state,
        do_metal=True,
        do_helium=False,
        apply_stim=True,
        wing_mode=wing_mode,
    )
    gross = gross_tensor.detach().cpu().numpy().copy()
    net = net_tensor.detach().cpu().numpy().copy()
    wing_index = int(invariants.metal_wing_index[0].item())
    nonzero_count, reach = _one_line_reach(gross, wing_index)
    return SynthesisOneLineCheckpoint(
        regime=regime,
        wavelength_nm=wavelength,
        continuum_opacity_cm2_per_g=continuum,
        gross_line_mass_absorption_tensor=gross_tensor.detach().clone(),
        net_line_mass_absorption_tensor=net_tensor.detach().clone(),
        gross_line_mass_absorption_coefficient=gross,
        net_line_mass_absorption_coefficient=net,
        activity_mask=np.any(gross != 0.0, axis=1),
        nonzero_count=nonzero_count,
        wing_reach=reach,
        metal_center_index=int(invariants.metal_center_index[0].item()),
        metal_wing_index=wing_index,
        work_dtype=str(invariants.wavelength_grid.dtype),
        accumulation_dtype=str(gross_tensor.dtype),
        cutoff_dtype="torch.float32",
        stimulation_dtype="torch.float32",
        device=str(gross_tensor.device),
        metal_line_count=int(invariants.metal_catalog_index.numel()),
        auto_line_count=int(invariants.auto_catalog_index.numel()),
        helium_line_count=int(invariants.helium_line_type.numel()),
        population_ion_stage_index=int(
            invariants.metal_population_ion_stage_index[0].item()
        ),
        population_element_index=int(
            invariants.metal_population_element_index[0].item()
        ),
        wing_mode=wing_mode,
    )


def run_atmosphere_one_line() -> AtmosphereOneLineCheckpoint:
    """Run the staged serial-compiled atmosphere lane from its compact fixture.

    The fixture contains only the columns read by the chosen line.  This
    function reconstructs the three public dense inputs in memory, calls the
    textbook's staged source, and forms a separate stimulated view.  It opens
    no comparison golden.
    """

    if (
        hashlib.sha256(ATMOSPHERE_ONE_LINE_FIXTURE.read_bytes()).hexdigest()
        != ATMOSPHERE_ONE_LINE_FIXTURE_SHA256
    ):
        raise RuntimeError("the Chapter 6 atmosphere input fixture changed")
    with np.load(ATMOSPHERE_ONE_LINE_FIXTURE, allow_pickle=False) as archive:
        fixture = {name: np.asarray(archive[name]).copy() for name in archive.files}

    configure_local_data_paths()
    from payne_zero_atmosphere.line_catalog import SelectedLineCatalog
    from payne_zero_atmosphere.line_opacity import (
        accumulate_selected_line_opacity,
    )

    selected = SelectedLineCatalog(
        packed_wavelength_index=fixture["packed_wavelength_index"],
        packed_species_slot=fixture["packed_species_slot"],
        lower_excitation_index=fixture["lower_excitation_index"],
        log_strength_index=fixture["log_strength_index"],
        radiative_damping_index=fixture["radiative_damping_index"],
        stark_damping_index=fixture["stark_damping_index"],
        van_der_waals_damping_index=fixture["van_der_waals_damping_index"],
    )
    depth_count = int(fixture["temperature"].size)
    population_slot_count = 1006
    line_slot = int(fixture["line_population_slot_zero_based"])

    actual = np.zeros((depth_count, population_slot_count), dtype=np.float64)
    actual[:, fixture["actual_population_slot_indices"]] = fixture[
        "actual_population_slot_values"
    ]
    line_support = np.zeros_like(actual)
    line_support[:, line_slot] = fixture[
        "partition_normalized_population_over_mass_density_and_"
        "fractional_doppler_width_at_line_slot"
    ]
    fractional_widths = np.zeros_like(actual)
    fractional_widths[:, line_slot] = fixture[
        "fractional_doppler_widths_at_line_slot"
    ]

    result = accumulate_selected_line_opacity(
        selected_lines=selected,
        opacity_wavelength_grid_nm=fixture["opacity_wavelength_grid_nm"],
        wavelength_bin_edges=fixture["wavelength_bin_edges"],
        continuum_line_selection_threshold=fixture[
            "continuum_line_selection_threshold"
        ],
        temperature=fixture["temperature"],
        hc_over_kt=fixture["hc_over_kt"],
        electron_density=fixture["electron_density"],
        ion_stage_populations_by_packed_slot=actual,
        partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
            line_support
        ),
        fractional_doppler_widths=fractional_widths,
    )
    pre = np.asarray(result.line_mass_absorption_coefficient).copy()
    stimulation = stimulated_emission_factor(
        fixture["opacity_wavelength_grid_nm"][None, :],
        fixture["temperature"][:, None],
    )
    post = pre.astype(np.float64) * stimulation
    return AtmosphereOneLineCheckpoint(
        effective_temperature=float(fixture["effective_temperature"]),
        wavelength_nm=fixture["opacity_wavelength_grid_nm"],
        temperature=fixture["temperature"],
        pre_stimulated_line_mass_absorption_coefficient=pre,
        post_stimulated_line_mass_absorption_coefficient=post,
        selected_line_count=int(result.selected_line_count),
        nonzero_count_per_depth=np.count_nonzero(pre, axis=1).astype(np.int64),
        peak_pre_stimulated_cm2_per_g=float(np.max(pre)),
        accumulation_dtype=str(pre.dtype),
        device="cpu",
        stimulation_owner="downstream atmosphere transfer accumulator",
    )


def synthesis_center_policy_checkpoint(
    regime: str,
) -> ProductionCenterPolicyCheckpoint:
    """Expose the exact FASTEX, cutoff, damping, and center-profile bridge."""

    configure_local_data_paths()
    from payne_zero_synthesis.constants import (
        CLASSICAL_LINE_STRENGTH_COEFFICIENT,
    )
    from payne_zero_synthesis.line_opacity import (
        accumulate_atomic,
        fast_ex,
        harris_profile_at_line_center,
        precompute_invariants,
    )
    import torch

    wavelength = build_synthesis_wavelength_grid()
    state, _ = synthesis_line_state(regime, wavelength)
    invariants = precompute_invariants(
        one_line_synthesis_mapping(),
        wavelength,
        runtime_device=torch.device("cpu"),
    )
    work_dtype = invariants.wavelength_grid.dtype
    population = torch.as_tensor(
        state["partition_normalized_populations"],
        dtype=work_dtype,
    )[:, 0, 25]
    doppler_fraction = torch.as_tensor(
        state["fractional_doppler_widths"],
        dtype=work_dtype,
    )[:, 0, 25]
    mass_density = torch.as_tensor(state["mass_density"], dtype=work_dtype)
    electron_density = torch.as_tensor(
        state["electron_density"],
        dtype=torch.float32,
    ).to(work_dtype)
    hc_over_kt = torch.as_tensor(state["hc_over_kt"], dtype=work_dtype)
    collision_density = torch.as_tensor(
        state["collision_density_proxy"],
        dtype=torch.float32,
    ).to(work_dtype)
    continuum = torch.as_tensor(
        state["continuum_opacity"],
        dtype=torch.float32,
    ).to(work_dtype)

    valid = (population > 0.0) & (doppler_fraction > 0.0) & (mass_density > 0.0)
    safe_width = torch.where(
        doppler_fraction > 0.0,
        doppler_fraction,
        torch.ones_like(doppler_fraction),
    )
    safe_density = torch.where(
        mass_density > 0.0,
        mass_density,
        torch.ones_like(mass_density),
    )
    population_doppler_ratio = torch.where(
        valid,
        population / (safe_density * safe_width),
        torch.zeros_like(population),
    )
    classical_strength = invariants.metal_classical_strength[0].to(work_dtype)
    pre_excitation_strength = classical_strength * population_doppler_ratio
    excitation_exponent = invariants.metal_lower_excitation_cm[0] * hc_over_kt
    fast_weight = fast_ex(
        excitation_exponent,
        invariants.exponential_integer_table,
        invariants.exponential_fraction_table,
    )
    direct_weight = torch.exp(-excitation_exponent)
    line_amplitude = pre_excitation_strength * fast_weight

    center_index = int(invariants.metal_center_clamped[0].item())
    center_cutoff = continuum[:, center_index] * LINE_CENTER_CUTOFF_RATIO
    passes_pre = valid & (pre_excitation_strength >= center_cutoff)
    passes_post = (
        passes_pre & (line_amplitude >= center_cutoff) & (line_amplitude > 0.0)
    )

    radiative = invariants.metal_radiative_damping[0].to(work_dtype)
    stark = invariants.metal_stark_damping[0].to(work_dtype) * electron_density
    van_der_waals = (
        invariants.metal_van_der_waals_damping[0].to(work_dtype) * collision_density
    )
    damping_ratio = (radiative + stark + van_der_waals) / safe_width
    full_profile = harris_profile_at_line_center(
        damping_ratio.to(torch.float32),
        invariants.harris_profile_h0_table,
        invariants.harris_profile_h1_table,
        invariants.harris_profile_h2_table,
    ).to(work_dtype)
    shortcut_profile = 1.0 - 1.128 * damping_ratio
    use_shortcut = damping_ratio < 0.2
    selected_profile = torch.where(
        use_shortcut,
        shortcut_profile,
        full_profile,
    )
    center_deposit = torch.where(
        passes_post & (damping_ratio >= 0.0),
        line_amplitude * selected_profile,
        torch.zeros_like(line_amplitude),
    ).to(torch.float32)

    production = accumulate_atomic(
        invariants,
        state,
        do_metal=True,
        do_helium=False,
        apply_stim=False,
        wing_mode="batched",
    )
    production_center = production[:, center_index]

    transition = transition_checkpoint()
    frequency_hz = LIGHT_SPEED_NM_PER_S / transition.wavelength_nm
    host_classical_strength = (
        CLASSICAL_LINE_STRENGTH_COEFFICIENT
        * transition.oscillator_strength
        / frequency_hz
    )
    branch = np.where(
        use_shortcut.cpu().numpy(),
        "low-damping center shortcut",
        "full Harris center",
    )

    def as_numpy(values):
        return values.detach().cpu().numpy().copy()

    return ProductionCenterPolicyCheckpoint(
        regime=regime,
        depth_index=np.arange(population.numel(), dtype=np.int64),
        excitation_exponent=as_numpy(excitation_exponent),
        direct_exp_weight=as_numpy(direct_weight),
        fast_ex_weight=as_numpy(fast_weight),
        pre_excitation_strength_cm2_per_g=as_numpy(pre_excitation_strength),
        post_fastex_line_amplitude_cm2_per_g=as_numpy(line_amplitude),
        center_cutoff_cm2_per_g=as_numpy(center_cutoff),
        passes_pre_excitation_cutoff=as_numpy(passes_pre),
        passes_post_fastex_cutoff=as_numpy(passes_post),
        damping_ratio=as_numpy(damping_ratio),
        full_harris_center_profile=as_numpy(full_profile),
        shortcut_center_profile=as_numpy(shortcut_profile),
        selected_center_profile=as_numpy(selected_profile),
        selected_branch=np.asarray(branch),
        float32_center_deposit_cm2_per_g=as_numpy(center_deposit),
        production_center_deposit_cm2_per_g=as_numpy(production_center),
        host_classical_line_strength_cm2=float(host_classical_strength),
        float32_classical_line_strength_cm2=float(
            invariants.metal_classical_strength[0].item()
        ),
    )


__all__ = [
    "ATMOSPHERE_LINE_TABLES",
    "ATMOSPHERE_ONE_LINE_FIXTURE",
    "ATMOSPHERE_ONE_LINE_FIXTURE_SHA256",
    "AtmosphereOneLineCheckpoint",
    "CHAPTER05_CONTINUUM_FIXTURE",
    "CLASSICAL_INTEGRATED_LINE_COEFFICIENT",
    "DAMPING_NORMALIZATION",
    "DampingCheckpoint",
    "DenseLineCheckpoint",
    "FastExponentialCheckpoint",
    "HarrisBranchCheckpoint",
    "LINE_CENTER_CUTOFF_RATIO",
    "ProductionCenterPolicyCheckpoint",
    "ProfileNormalizationCheckpoint",
    "RAW_LINE_FIELDS",
    "REPOSITORY_ROOT",
    "SQRT_PI_REFERENCE",
    "SYNTHESIS_LINE_TABLES",
    "StrengthCheckpoint",
    "StoredDopplerCheckpoint",
    "SynthesisOneLineCheckpoint",
    "TEACHING_LINE_SUBSET",
    "TransitionCheckpoint",
    "atmosphere_harris_profile",
    "build_synthesis_wavelength_grid",
    "collision_density_proxy",
    "configure_local_data_paths",
    "continuous_voigt_h",
    "damping_checkpoint",
    "dense_line_checkpoint",
    "fast_exponential_checkpoint",
    "gross_line_strength_checkpoint",
    "harris_branch_checkpoint",
    "load_teaching_line_raw",
    "lower_level_boltzmann_factor",
    "one_line_synthesis_mapping",
    "profile_normalization_checkpoint",
    "run_atmosphere_one_line",
    "run_synthesis_one_line",
    "stored_doppler_checkpoint",
    "stimulated_emission_factor",
    "synthesis_center_policy_checkpoint",
    "synthesis_line_state",
    "synthesis_harris_profile",
    "teaching_line_record",
    "transition_checkpoint",
]
