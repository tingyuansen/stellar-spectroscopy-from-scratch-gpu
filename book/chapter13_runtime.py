"""Executable Chapter 13 correction and iteration-control checkpoints.

The exact correction, convergence, and runner modules are staged in ``src``.
The chapter first consumes Chapter 12's live ``IterationFinalization`` through
the canonical complete-state remapper. Its analytic correction fixture remains
a controlled parity lens for inspecting individual terms and safeguards.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

from book.chapter13_teaching import (
    StructuralConvergenceTrace,
    fixed_chunk_bounds,
    trace_structural_convergence,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
FIXTURE_PATH = REPOSITORY_ROOT / "data/fixtures/chapter13_correction_inputs.npz"
GOLDEN_PATH = (
    REPOSITORY_ROOT
    / "data/golden/payne_zero/chapter13/chapter13_correction_outputs.npz"
)
ARTIFACT_MANIFEST = REPOSITORY_ROOT / "data/chapter13_artifacts.json"
PINNED_SOURCE_SHA256 = {
    "src/payne_zero_atmosphere/temperature_correction.py": (
        "67728389ba857511979d0f82ea59f0bf41ee635b8151ae26673dace02b195d21"
    ),
    "src/payne_zero_atmosphere/convergence.py": (
        "6b4c674deda148baab6fd90e8a25eed2921581ce7d4bace489c024bc1c2748cb"
    ),
}

RESET_FIELDS = (
    "mean_intensity_minus_source_integral",
    "absorption_heating_derivative",
    "diagonal_lambda_accumulator",
    "integrated_eddington_flux",
)
CARRIED_FIELDS = (
    "previous_temperature_correction",
    "rosseland_opacity_table",
)
RUNNER_PASS_ORDER = (
    "seed copy or prior remap plus optional hydrostatic pressure",
    "pass-local setup with carried surface radiation pressure",
    "populations, molecules, internal energy, widths, and strengths",
    "continuum and line opacity",
    "first-pass catalog construction or later object reuse",
    "reset pass accumulators while history and lookup persist",
    "frequency-chunk transfer and fixed-order reduction",
    "Rosseland finalization",
    "radiation-pressure finalization and new surface constant",
    "Rosseland lookup ingest",
    "four EOS perturbations and convection when enabled",
    "temperature and column-mass correction",
    "disabled-convection diagnostics when needed, then complete remap",
    "carry lookup and surface constant",
    "structural and flux diagnostics, counter, return or terminal exit",
)
RUNNER_REQUIRED_SYMBOLS = (
    "TransferAccumulation",
    "IterationFinalization",
    "IterationRemap",
    "AtmosphereRunResult",
    "finalize_transfer_state",
    "remap_finalized_iteration_state",
    "finalize_remapped_iteration",
    "run_atmosphere_model",
)
PREWARM_REPRESENTATIVE_BRANCHES = (
    {
        "name": "hot",
        "effective_temperature": 9000.0,
        "log_surface_gravity": 4.0,
        "metallicity": 0.0,
        "alpha_enhancement": 0.0,
        "microturbulence_km_s": 2.0,
        "enable_molecules": True,
    },
    {
        "name": "sun",
        "effective_temperature": 5777.0,
        "log_surface_gravity": 4.44,
        "metallicity": 0.0,
        "alpha_enhancement": 0.0,
        "microturbulence_km_s": 2.0,
        "enable_molecules": True,
    },
    {
        "name": "giant",
        "effective_temperature": 4500.0,
        "log_surface_gravity": 2.0,
        "metallicity": -0.5,
        "alpha_enhancement": 0.2,
        "microturbulence_km_s": 2.0,
        "enable_molecules": True,
    },
    {
        "name": "sun_atomic_only",
        "effective_temperature": 5777.0,
        "log_surface_gravity": 4.44,
        "metallicity": 0.0,
        "alpha_enhancement": 0.0,
        "microturbulence_km_s": 2.0,
        "enable_molecules": False,
        "numba_threads": 1,
    },
)
BASE_DIAGNOSTIC_KEYS = (
    "supported_branch",
    "layer_count",
    "frequency_count",
    "frequency_start_index",
    "frequency_stop_index",
    "line_selection_enabled",
    "detailed_line_enabled",
    "molecules_enabled",
    "convection_enabled",
    "deep_layer_relative_temperature_change",
    "all_layer_relative_temperature_change",
    "median_absolute_flux_error_percent",
    "p95_absolute_flux_error_percent",
    "maximum_absolute_flux_error_percent",
    "maximum_deep_layer_relative_temperature_change",
    "maximum_all_layer_relative_temperature_change",
    "enable_convergence_stop",
    "minimum_iterations_before_convergence",
    "required_consecutive_converged_iterations",
    "consecutive_converged_iterations",
    "setup_seconds",
    "total_seconds",
    "iteration_timings",
)
ITERATION_TIMING_KEYS = (
    "iteration",
    "prepare_iteration_seconds",
    "population_seconds",
    "opacity_seconds",
    "transfer_seconds",
    "finalization_seconds",
    "remap_seconds",
    "deep_layer_relative_temperature_change",
    "all_layer_relative_temperature_change",
    "median_absolute_flux_error_percent",
    "p95_absolute_flux_error_percent",
    "maximum_absolute_flux_error_percent",
    "convergence_seconds",
    "total_seconds",
)


@dataclass(frozen=True)
class Chapter12HandoffCheckpoint:
    """Exact Chapter 12 finalization consumed at Chapter 13's remap boundary."""

    finalization_type: str
    correction_type: str
    remap_type: str
    depth_count: int
    frequency_count: int
    correction_field_names: tuple[str, ...]
    correction_temperature: np.ndarray
    correction_column_mass: np.ndarray
    correction_result_finite: bool
    rosseland_optical_depth_strictly_increasing: bool
    correction_column_mass_positive: bool
    correction_column_mass_strictly_increasing: bool
    remapped_temperature: np.ndarray
    remapped_column_mass: np.ndarray
    standard_rosseland_optical_depth: np.ndarray
    remapped_fields_finite: bool
    remapped_column_mass_strictly_increasing: bool
    source_finalization_is_chapter12_cache: bool


@dataclass(frozen=True)
class CorrectionCheckpoint:
    """Computed correction plus comparison-only exact parity evidence."""

    input_temperature: np.ndarray
    rosseland_optical_depth: np.ndarray
    computed: dict[str, np.ndarray]
    golden: dict[str, np.ndarray]
    maximum_absolute_difference: float
    raw_three_term_identity: bool
    minimum_inward_temperature_rise_k: float
    column_mass_positive: bool
    column_mass_strictly_increasing: bool
    fixture_sha256: str
    golden_sha256: str


@dataclass(frozen=True)
class ResetCheckpoint:
    """Mode-1 reset evidence for mixed-lifetime correction state."""

    reset_fields: tuple[str, ...]
    carried_fields: tuple[str, ...]
    reset_sums: np.ndarray
    previous_correction_unchanged: bool
    lookup_same_object: bool
    lookup_entry_count_unchanged: bool


@dataclass(frozen=True)
class DampingCheckpoint:
    """Three exact previous-correction branches with convection disabled."""

    undamped: np.ndarray
    same_sign: np.ndarray
    sign_flip: np.ndarray
    same_sign_expected: np.ndarray
    sign_flip_expected: np.ndarray


@dataclass(frozen=True)
class IterationControlCheckpoint:
    """Executable structural stopping traces and the frozen pass order."""

    standard_trace: StructuralConvergenceTrace
    interrupted_trace: StructuralConvergenceTrace
    stopping_disabled_trace: StructuralConvergenceTrace
    pass_order: tuple[str, ...]


@dataclass(frozen=True)
class ChunkCheckpoint:
    """Fixed contiguous chunk membership and reduction-order evidence."""

    start: int
    stop: int
    chunk_count: int
    bounds: np.ndarray
    membership: np.ndarray
    covers_each_frequency_once: bool
    reduction_order: np.ndarray


@dataclass(frozen=True)
class QuantizationCheckpoint:
    """One terminal fixed-column round trip and its idempotence."""

    input_temperature: np.ndarray
    quantized_temperature: np.ndarray
    first_deck: str
    second_deck: str
    idempotent: bool
    terminal_format_parse_calls: int
    idempotence_probe_format_parse_calls: int


@dataclass(frozen=True)
class CacheContractCheckpoint:
    """Observed cache precedence plus the not-yet-runnable prewarm boundary."""

    existing_numba_cache: Path
    requested_payne_zero_cache: Path
    default_cache: Path
    runner_symbols_available: tuple[str, ...]
    runner_symbols_missing: tuple[str, ...]
    prewarm_executable: bool


@dataclass(frozen=True)
class PrewarmContractCheckpoint:
    """Exact representative branches without a false full-run claim."""

    representative_branches: tuple[dict[str, object], ...]
    representative_iterations_per_branch: int
    branch_names: tuple[str, ...]
    runner_ready: bool
    executable: bool


@dataclass(frozen=True)
class OutputContractCheckpoint:
    """Frozen runner diagnostics and terminal product boundaries."""

    base_diagnostic_keys: tuple[str, ...]
    iteration_timing_keys: tuple[str, ...]
    diagnostics_path_written: bool
    debug_uses_terminal_quantized_atmosphere: bool
    product_requires_structural_convergence: bool
    product_population_source: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_chapter13_runtime() -> Path:
    """Resolve exact staged modules and fail closed on local artifact identity."""

    staged_root = SOURCE_ROOT.resolve()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith(("payne_zero_atmosphere", "payne_zero_synthesis")):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is not None and not Path(module_path).resolve().is_relative_to(
            staged_root
        ):
            raise RuntimeError(
                f"{name} resolved outside the staged source tree: {module_path}"
            )
    staged = str(staged_root)
    if staged in sys.path:
        sys.path.remove(staged)
    sys.path.insert(0, staged)
    static_root = REPOSITORY_ROOT / "data/static"
    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(static_root)
    os.environ["PAYNE_ZERO_ATMOSPHERE_DATA_ROOT"] = str(
        static_root / "atmosphere_tables"
    )
    os.environ["PAYNE_ZERO_SYNTHESIS_DATA_ROOT"] = str(
        static_root / "synthesis_tables"
    )
    manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("payne_zero_commit") != PINNED_PAYNE_ZERO_COMMIT:
        raise RuntimeError("Chapter 13 manifest names the wrong Payne Zero commit")
    for relative, expected in PINNED_SOURCE_SHA256.items():
        if _sha256(REPOSITORY_ROOT / relative) != expected:
            raise RuntimeError(f"Chapter 13 source identity changed: {relative}")
    for path in (FIXTURE_PATH, GOLDEN_PATH):
        record = manifest["artifacts"][str(path.relative_to(REPOSITORY_ROOT))]
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Chapter 13 artifact identity changed: {path}")
    return SOURCE_ROOT


@lru_cache(maxsize=1)
def chapter12_handoff_checkpoint() -> Chapter12HandoffCheckpoint:
    """Consume Chapter 12's live finalization through the canonical remapper."""

    configure_chapter13_runtime()
    from book.chapter12_runtime import chapter11_iteration_finalization
    from payne_zero_atmosphere.runner import remap_finalized_iteration_state

    finalization = chapter11_iteration_finalization()
    correction = finalization.temperature_correction_result
    remapped = remap_finalized_iteration_state(
        finalization,
        completed_iterations=1,
    )
    atmosphere = remapped.atmosphere
    correction_arrays = [
        np.asarray(getattr(correction, field.name))
        for field in fields(correction)
        if isinstance(getattr(correction, field.name), np.ndarray)
    ]
    remapped_arrays = (
        atmosphere.column_mass,
        atmosphere.temperature,
        atmosphere.gas_pressure,
        atmosphere.electron_density,
        atmosphere.rosseland_opacity,
        atmosphere.radiative_acceleration,
        atmosphere.microturbulence,
        atmosphere.convective_flux,
        atmosphere.convective_velocity,
    )
    return Chapter12HandoffCheckpoint(
        finalization_type=type(finalization).__name__,
        correction_type=type(correction).__name__,
        remap_type=type(remapped).__name__,
        depth_count=atmosphere.layers,
        frequency_count=int(
            finalization.transfer_accumulation.opacity_state.opacity_frequency_hz.size
        ),
        correction_field_names=tuple(field.name for field in fields(correction)),
        correction_temperature=correction.temperature.copy(),
        correction_column_mass=correction.column_mass.copy(),
        correction_result_finite=all(
            np.all(np.isfinite(array)) for array in correction_arrays
        ),
        rosseland_optical_depth_strictly_increasing=bool(
            np.all(np.diff(finalization.rosseland_optical_depth) > 0.0)
        ),
        correction_column_mass_positive=bool(
            np.all(correction.column_mass > 0.0)
        ),
        correction_column_mass_strictly_increasing=bool(
            np.all(np.diff(correction.column_mass) > 0.0)
        ),
        remapped_temperature=atmosphere.temperature.copy(),
        remapped_column_mass=atmosphere.column_mass.copy(),
        standard_rosseland_optical_depth=(
            remapped.standard_rosseland_optical_depth.copy()
        ),
        remapped_fields_finite=all(
            np.all(np.isfinite(values)) for values in remapped_arrays
        ),
        remapped_column_mass_strictly_increasing=bool(
            np.all(np.diff(atmosphere.column_mass) > 0.0)
        ),
        source_finalization_is_chapter12_cache=(
            finalization is chapter11_iteration_finalization()
        ),
    )


def load_correction_inputs() -> dict[str, np.ndarray]:
    """Load defensive copies of the analytic input fixture."""

    configure_chapter13_runtime()
    with np.load(FIXTURE_PATH, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def run_correction_arrays(
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Evaluate exact mode 3 without reading the comparison-only golden."""

    configure_chapter13_runtime()
    from payne_zero_atmosphere.temperature_correction import (
        apply_temperature_correction,
        ingest_temperature_correction_rosseland_table,
        initialize_temperature_correction_state,
    )

    layer_count = int(np.asarray(inputs["temperature_k"]).size)
    state = initialize_temperature_correction_state(layer_count)
    for name in (
        "integrated_eddington_flux",
        "mean_intensity_minus_source_integral",
        "absorption_heating_derivative",
        "diagonal_lambda_accumulator",
        "previous_temperature_correction",
    ):
        getattr(state, name)[:] = inputs[name]
    for index in range(np.asarray(inputs["lookup_temperature_k"]).shape[0]):
        ingest_temperature_correction_rosseland_table(
            state,
            temperature_k=inputs["lookup_temperature_k"][index],
            gas_pressure=inputs["lookup_gas_pressure"][index],
            rosseland_opacity=inputs["lookup_rosseland_opacity"][index],
        )

    zeros = np.zeros(layer_count, dtype=np.float64)
    ones = np.ones(layer_count, dtype=np.float64)
    result = apply_temperature_correction(
        state,
        mode=3,
        frequency_weight=0.0,
        column_mass=inputs["column_mass"],
        total_opacity=ones,
        monochromatic_eddington_flux=zeros,
        mean_intensity_minus_source=zeros,
        monochromatic_optical_depth=zeros,
        planck_source=zeros,
        frequency_hz=0.0,
        h_over_kt=ones,
        temperature_k=inputs["temperature_k"],
        stimulated_emission=ones,
        scattering_fraction=zeros,
        target_integrated_eddington_flux=float(
            inputs["target_integrated_eddington_flux"][0]
        ),
        effective_temperature=float(inputs["effective_temperature"][0]),
        frequency_count=int(inputs["frequency_count"][0]),
        rosseland_optical_depth=inputs["rosseland_optical_depth"],
        rosseland_opacity=inputs["rosseland_opacity"],
        iteration_index=int(inputs["iteration_index"][0]),
        convection_enabled=int(inputs["convection_enabled"][0]),
        convective_flux=inputs["convective_flux"],
        previous_convective_flux=inputs["previous_convective_flux"],
        logarithmic_temperature_pressure_gradient=inputs[
            "logarithmic_temperature_pressure_gradient"
        ],
        adiabatic_gradient=inputs["adiabatic_gradient"],
        pressure_scale_height=inputs["pressure_scale_height"],
        total_pressure=inputs["total_pressure"],
        mass_density=inputs["mass_density"],
        log_density_temperature_derivative_at_constant_total_pressure=inputs[
            "log_density_temperature_derivative_at_constant_total_pressure"
        ],
        heat_capacity=inputs["heat_capacity"],
        mixing_length=float(inputs["mixing_length"][0]),
        smooth_start_layer=int(inputs["smooth_start_layer"][0]),
        smooth_stop_layer=int(inputs["smooth_stop_layer"][0]),
        integrated_radiation_pressure=inputs["integrated_radiation_pressure"],
        turbulent_pressure=inputs["turbulent_pressure"],
        surface_gravity_cgs=float(inputs["surface_gravity_cgs"][0]),
    )
    if result is None:
        raise RuntimeError("mode 3 returned no correction result")
    outputs = {
        name: np.asarray(getattr(result, name)).copy()
        for name in result.__dataclass_fields__
    }
    outputs["state_previous_temperature_correction"] = (
        state.previous_temperature_correction.copy()
    )
    outputs["state_rosseland_entry_count"] = np.asarray(
        [state.rosseland_opacity_table.entry_count], dtype=np.int32
    )
    return outputs


def correction_checkpoint() -> CorrectionCheckpoint:
    """Compute first, then compare with the pinned exact output archive."""

    inputs = load_correction_inputs()
    computed = run_correction_arrays(inputs)
    with np.load(GOLDEN_PATH, allow_pickle=False) as archive:
        golden = {name: np.asarray(archive[name]).copy() for name in archive.files}
    differences = [
        float(np.max(np.abs(computed[name] - golden[name])))
        for name in computed
        if np.issubdtype(computed[name].dtype, np.number)
    ]
    three_term_sum = (
        computed["flux_temperature_derivative"]
        + computed["lambda_temperature_derivative"]
        + computed["surface_temperature_derivative"]
    )
    return CorrectionCheckpoint(
        input_temperature=inputs["temperature_k"],
        rosseland_optical_depth=inputs["rosseland_optical_depth"],
        computed=computed,
        golden=golden,
        maximum_absolute_difference=max(differences, default=0.0),
        raw_three_term_identity=np.array_equal(
            computed["temperature_correction"], three_term_sum
        ),
        minimum_inward_temperature_rise_k=float(
            np.min(np.diff(computed["temperature"]))
        ),
        column_mass_positive=bool(np.all(computed["column_mass"] > 0.0)),
        column_mass_strictly_increasing=bool(
            np.all(np.diff(computed["column_mass"]) > 0.0)
        ),
        fixture_sha256=_sha256(FIXTURE_PATH),
        golden_sha256=_sha256(GOLDEN_PATH),
    )


def reset_checkpoint() -> ResetCheckpoint:
    """Prove mode 1 resets accumulators but preserves history and lookup."""

    configure_chapter13_runtime()
    from payne_zero_atmosphere.temperature_correction import (
        apply_temperature_correction,
        ingest_temperature_correction_rosseland_table,
        initialize_temperature_correction_state,
    )

    inputs = load_correction_inputs()
    layer_count = inputs["temperature_k"].size
    state = initialize_temperature_correction_state(layer_count)
    for offset, name in enumerate(RESET_FIELDS, start=1):
        getattr(state, name)[:] = float(offset)
    state.previous_temperature_correction[:] = np.linspace(2.0, 3.0, layer_count)
    ingest_temperature_correction_rosseland_table(
        state,
        temperature_k=inputs["temperature_k"],
        gas_pressure=inputs["lookup_gas_pressure"][1],
        rosseland_opacity=inputs["rosseland_opacity"],
    )
    previous = state.previous_temperature_correction.copy()
    table = state.rosseland_opacity_table
    entry_count = int(table.entry_count)
    zeros = np.zeros(layer_count, dtype=np.float64)
    ones = np.ones(layer_count, dtype=np.float64)
    apply_temperature_correction(
        state,
        mode=1,
        frequency_weight=0.0,
        column_mass=inputs["column_mass"],
        total_opacity=ones,
        monochromatic_eddington_flux=zeros,
        mean_intensity_minus_source=zeros,
        monochromatic_optical_depth=zeros,
        planck_source=zeros,
        frequency_hz=0.0,
        h_over_kt=ones,
        temperature_k=inputs["temperature_k"],
        stimulated_emission=ones,
        scattering_fraction=zeros,
        target_integrated_eddington_flux=float(
            inputs["target_integrated_eddington_flux"][0]
        ),
        effective_temperature=float(inputs["effective_temperature"][0]),
        frequency_count=int(inputs["frequency_count"][0]),
    )
    return ResetCheckpoint(
        reset_fields=RESET_FIELDS,
        carried_fields=CARRIED_FIELDS,
        reset_sums=np.asarray(
            [np.sum(getattr(state, name)) for name in RESET_FIELDS],
            dtype=np.float64,
        ),
        previous_correction_unchanged=np.array_equal(
            previous, state.previous_temperature_correction
        ),
        lookup_same_object=table is state.rosseland_opacity_table,
        lookup_entry_count_unchanged=entry_count
        == state.rosseland_opacity_table.entry_count,
    )


def damping_checkpoint() -> DampingCheckpoint:
    """Exercise first-pass, same-sign shrinkage, and sign-reversal branches."""

    inputs = load_correction_inputs()
    inputs["convection_enabled"] = np.asarray([0], dtype=np.int32)
    inputs["iteration_index"] = np.asarray([1], dtype=np.int32)
    inputs["previous_temperature_correction"] = np.zeros_like(
        inputs["temperature_k"]
    )
    undamped = run_correction_arrays(inputs)["temperature_correction"]

    inputs["iteration_index"] = np.asarray([2], dtype=np.int32)
    inputs["previous_temperature_correction"] = 2.0 * undamped
    same_sign = run_correction_arrays(inputs)["temperature_correction"]
    inputs["previous_temperature_correction"] = -undamped
    sign_flip = run_correction_arrays(inputs)["temperature_correction"]
    return DampingCheckpoint(
        undamped=undamped,
        same_sign=same_sign,
        sign_flip=sign_flip,
        same_sign_expected=1.25 * undamped,
        sign_flip_expected=0.5 * undamped,
    )


def _temperature_passes(
    relative_changes: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    base = np.linspace(4300.0, 7800.0, 80, dtype=np.float64)
    before = np.stack([base + 2.0 * index for index in range(len(relative_changes))])
    after = np.stack(
        [
            values * (1.0 + relative_change)
            for values, relative_change in zip(before, relative_changes, strict=True)
        ]
    )
    return before, after


def iteration_control_checkpoint() -> IterationControlCheckpoint:
    """Run exact norm/counter sequences without pretending physics seams exist."""

    before, after = _temperature_passes((1.0e-3, 4.0e-4, 3.0e-4, 2.0e-4))
    standard = trace_structural_convergence(
        before,
        after,
        enable_convergence_stop=True,
        minimum_iterations_before_convergence=3,
        required_consecutive_converged_iterations=2,
        maximum_deep_layer_relative_temperature_change=5.0e-4,
        maximum_all_layer_relative_temperature_change=5.0e-4,
    )
    before, after = _temperature_passes(
        (8.0e-4, 4.0e-4, 7.0e-4, 3.0e-4, 2.0e-4)
    )
    interrupted = trace_structural_convergence(
        before,
        after,
        enable_convergence_stop=True,
        minimum_iterations_before_convergence=2,
        required_consecutive_converged_iterations=2,
        maximum_deep_layer_relative_temperature_change=5.0e-4,
        maximum_all_layer_relative_temperature_change=5.0e-4,
    )
    disabled = trace_structural_convergence(
        before,
        after,
        enable_convergence_stop=False,
        minimum_iterations_before_convergence=1,
        required_consecutive_converged_iterations=1,
        maximum_deep_layer_relative_temperature_change=1.0,
        maximum_all_layer_relative_temperature_change=None,
    )
    return IterationControlCheckpoint(
        standard_trace=standard,
        interrupted_trace=interrupted,
        stopping_disabled_trace=disabled,
        pass_order=RUNNER_PASS_ORDER,
    )


def chunk_checkpoint(
    *,
    start: int = 0,
    stop: int = 17,
    chunk_count: int = 4,
) -> ChunkCheckpoint:
    """Show contiguous private work and the declared reduction order."""

    bounds = fixed_chunk_bounds(start, stop, chunk_count)
    membership = np.full(stop - start, -1, dtype=np.int64)
    visits = np.zeros(stop - start, dtype=np.int64)
    for chunk in range(chunk_count):
        left, right = int(bounds[chunk]), int(bounds[chunk + 1])
        membership[left - start : right - start] = chunk
        visits[left - start : right - start] += 1
    return ChunkCheckpoint(
        start=start,
        stop=stop,
        chunk_count=chunk_count,
        bounds=bounds,
        membership=membership,
        covers_each_frequency_once=bool(np.all(visits == 1)),
        reduction_order=np.arange(chunk_count, dtype=np.int64),
    )


def quantization_checkpoint() -> QuantizationCheckpoint:
    """Apply the exact fixed-column operator once, then check idempotence."""

    configure_chapter13_runtime()
    from payne_zero_atmosphere.atmosphere_io import (
        ModelAtmosphere,
        format_atmosphere_deck,
        parse_atmosphere_deck,
    )

    correction = correction_checkpoint()
    count = 8
    temperature = correction.computed["temperature"][:count] + np.linspace(
        0.001, 0.009, count
    )
    atmosphere = ModelAtmosphere(
        column_mass=correction.computed["column_mass"][:count],
        temperature=temperature,
        gas_pressure=np.geomspace(1.0e2, 1.0e6, count),
        electron_density=np.geomspace(1.0e9, 1.0e13, count),
        rosseland_opacity=np.linspace(0.12, 0.22, count),
        radiative_acceleration=np.linspace(0.0, 15.0, count),
        microturbulence=np.full(count, 2.0e5),
        convective_flux=np.zeros(count),
        convective_velocity=np.zeros(count),
        metadata={
            "effective_temperature": "5777.0",
            "log_surface_gravity": "4.44",
            "begin_line": "BEGIN                    ITERATION   4 COMPLETED",
        },
        fixed_column_abundance_values={1: 0.92, 2: 0.08},
    )
    first_deck = format_atmosphere_deck(atmosphere)
    quantized = parse_atmosphere_deck(first_deck, source="Chapter 13 terminal")
    second_deck = format_atmosphere_deck(quantized)
    requantized = parse_atmosphere_deck(second_deck, source="Chapter 13 repeat")
    return QuantizationCheckpoint(
        input_temperature=temperature,
        quantized_temperature=quantized.temperature.copy(),
        first_deck=first_deck,
        second_deck=second_deck,
        idempotent=bool(
            np.array_equal(quantized.temperature, requantized.temperature)
            and second_deck == format_atmosphere_deck(requantized)
        ),
        terminal_format_parse_calls=1,
        idempotence_probe_format_parse_calls=1,
    )


def cache_contract_checkpoint() -> CacheContractCheckpoint:
    """Observe precedence and report whether the complete prewarm seam exists."""

    configure_chapter13_runtime()
    from payne_zero_atmosphere._numba_cache import configure_numba_cache

    saved = {
        name: os.environ.get(name)
        for name in ("NUMBA_CACHE_DIR", "PAYNE_ZERO_NUMBA_CACHE_DIR")
    }
    live_numba_cache = None
    if "numba" in sys.modules:
        from numba import config as numba_config

        live_numba_cache = numba_config.CACHE_DIR
    try:
        os.environ["NUMBA_CACHE_DIR"] = "/tmp/chapter13-existing-numba"
        os.environ["PAYNE_ZERO_NUMBA_CACHE_DIR"] = "/tmp/chapter13-requested"
        existing = configure_numba_cache()
        os.environ.pop("NUMBA_CACHE_DIR", None)
        requested = configure_numba_cache()
        os.environ.pop("NUMBA_CACHE_DIR", None)
        os.environ.pop("PAYNE_ZERO_NUMBA_CACHE_DIR", None)
        default = configure_numba_cache()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if live_numba_cache is not None:
            from numba import config as numba_config

            numba_config.CACHE_DIR = live_numba_cache

    runner = importlib.import_module("payne_zero_atmosphere.runner")
    available = tuple(
        name for name in RUNNER_REQUIRED_SYMBOLS if hasattr(runner, name)
    )
    missing = tuple(
        name for name in RUNNER_REQUIRED_SYMBOLS if not hasattr(runner, name)
    )
    prewarm_spec = importlib.util.find_spec("payne_zero_atmosphere.prewarm")
    return CacheContractCheckpoint(
        existing_numba_cache=existing,
        requested_payne_zero_cache=requested,
        default_cache=default,
        runner_symbols_available=available,
        runner_symbols_missing=missing,
        prewarm_executable=prewarm_spec is not None and not missing,
    )


def prewarm_contract_checkpoint() -> PrewarmContractCheckpoint:
    """Return exact branch coverage while honoring the unresolved runner seam."""

    cache = cache_contract_checkpoint()
    branches = tuple(dict(branch) for branch in PREWARM_REPRESENTATIVE_BRANCHES)
    return PrewarmContractCheckpoint(
        representative_branches=branches,
        representative_iterations_per_branch=1,
        branch_names=tuple(str(branch["name"]) for branch in branches),
        runner_ready=not cache.runner_symbols_missing,
        executable=cache.prewarm_executable,
    )


def output_contract_checkpoint() -> OutputContractCheckpoint:
    """Return exact output semantics that the deferred runner patch must retain."""

    return OutputContractCheckpoint(
        base_diagnostic_keys=BASE_DIAGNOSTIC_KEYS,
        iteration_timing_keys=ITERATION_TIMING_KEYS,
        diagnostics_path_written=False,
        debug_uses_terminal_quantized_atmosphere=False,
        product_requires_structural_convergence=True,
        product_population_source="final_fixed_column_quantized_arrays",
    )


def runner_patch_plan() -> dict[str, object]:
    """Return the exact deferred insertion contract for the shared runner."""

    runner = importlib.import_module("payne_zero_atmosphere.runner")
    missing = tuple(
        name for name in RUNNER_REQUIRED_SYMBOLS if not hasattr(runner, name)
    )
    return {
        "blocked_by": (
            "Chapter 11/12 shared runner seams"
            if missing
            else None
        ),
        "required_symbols": RUNNER_REQUIRED_SYMBOLS,
        "missing_symbols": missing,
        "pass_order": RUNNER_PASS_ORDER,
        "interior_edge": "correction -> complete remap -> next pass",
        "terminal_edge": (
            "one fixed-column format/parse -> AtmosphereRunResult -> "
            "converged-only product rebuild"
        ),
        "forbidden_interior_call": "fixed-column format/parse",
    }
