"""Chapter 4 orchestration around the exact progressive molecular sources.

The notebook uses these helpers to keep visible cells focused on one physical
claim.  They load only manifest-bound textbook data and never open a golden or
the external Payne Zero checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "data"
STATIC_DATA_ROOT = DATA_ROOT / "static"
MOLECULAR_INPUT_FIXTURE = DATA_ROOT / "fixtures" / "chapter04_molecular_inputs.npz"
ATMOSPHERE_MOLECULAR_TABLES = (
    STATIC_DATA_ROOT / "atmosphere_tables" / "molecular_equilibrium_tables.npz"
)
ATMOSPHERE_MOLECULE_CATALOG = (
    STATIC_DATA_ROOT
    / "source_catalogs"
    / "lines"
    / "molecular_equilibrium_atmosphere.npz"
)
SYNTHESIS_MOLECULE_CATALOG = (
    STATIC_DATA_ROOT
    / "source_catalogs"
    / "lines"
    / "molecular_equilibrium_synthesis.npz"
)
SYNTHESIS_EOS_TABLES = (
    STATIC_DATA_ROOT / "synthesis_tables" / "partition_saha_tables.npz"
)
SYNTHESIS_ATOMIC_MASSES = STATIC_DATA_ROOT / "synthesis_tables" / "atomic_masses.npz"
SYNTHESIS_CONTINUUM_EDGE_GRID = (
    STATIC_DATA_ROOT / "synthesis_tables" / "continuum_edge_grid.npz"
)


@dataclass(frozen=True)
class AtmosphereJacobianCheckpoint:
    """Exact one-depth residual/Jacobian and its numerical derivative check."""

    depth_index: int
    temperature_k: float
    gas_pressure: float
    density_cm3: np.ndarray
    formation_constants: np.ndarray
    residual: np.ndarray
    residual_scale: np.ndarray
    analytic_jacobian: np.ndarray
    finite_difference_jacobian: np.ndarray
    row_species_codes: np.ndarray
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    absolute_error: np.ndarray
    column_scale_relative_error: np.ndarray
    max_absolute_error: float
    max_column_scale_relative_error: float
    particle_carbon_oxygen_indices: np.ndarray
    particle_carbon_oxygen_analytic_block: np.ndarray
    particle_carbon_oxygen_finite_difference_block: np.ndarray


@dataclass(frozen=True)
class AtmosphereLinearSolveEvidence:
    """One exact linear-solver branch and its algebraic diagnostics."""

    branch: str
    matrix_rank: int
    matrix: np.ndarray
    right_hand_side: np.ndarray
    step: np.ndarray
    step_norm: float
    linear_residual_norm: float


@dataclass(frozen=True)
class AtmosphereNewtonUpdateWitness:
    """Per-index evidence from one controlled exact Newton density update."""

    labels: tuple[str, ...]
    branches: tuple[str, ...]
    old_density: np.ndarray
    previous_delta: np.ndarray
    raw_delta: np.ndarray
    effective_delta: np.ndarray
    candidate: np.ndarray
    returned_density: np.ndarray
    sign_damped: np.ndarray
    scale_before: np.ndarray
    scale_after: np.ndarray
    still_iterating: bool
    convergence_probe_relative_update: np.ndarray
    convergence_probe_returned_density: np.ndarray
    convergence_probe_still_iterating: np.ndarray


@dataclass(frozen=True)
class AtmosphereContinuationCheckpoint:
    """Exact ordered-continuation and independent-restart diagnostics."""

    depth_index: np.ndarray
    pressure_ratio: np.ndarray
    continuation_seed: np.ndarray
    expected_continuation_seed: np.ndarray
    continuation_seed_equal: np.ndarray
    continuation_solution: np.ndarray
    continuation_iteration_count: np.ndarray
    continuation_converged: np.ndarray
    continuation_residual_max_abs: np.ndarray
    continuation_residual_max_scaled: np.ndarray
    restart_seed: np.ndarray
    restart_solution: np.ndarray
    restart_iteration_count: np.ndarray
    restart_converged: np.ndarray
    restart_residual_max_abs: np.ndarray
    restart_residual_max_scaled: np.ndarray


@dataclass(frozen=True)
class PublicMolecularLaneCheckpoint:
    """Exact release mapping and the counterfactual needed to interpret it."""

    structured_atmosphere: dict[str, np.ndarray]
    line_species_codes: np.ndarray
    equilibrium_code_offsets: np.ndarray
    equilibrium_codes: np.ndarray
    public_columns: np.ndarray
    partition_cube_before: np.ndarray
    partition_cube_after: np.ndarray
    ion_cube_before: np.ndarray
    ion_cube_after: np.ndarray
    normalized_delta: np.ndarray
    no_ground_line_populations: np.ndarray
    grounded_line_populations: np.ndarray
    ground_discrimination_mask: np.ndarray
    owned_stage5_mask: np.ndarray
    co_line_species_code: int
    co_equilibrium_code: float
    co_public_column: int
    co_catalog_index: int
    co_component_equation_indices: np.ndarray
    co_raw_population: np.ndarray
    co_normalized_population: np.ndarray
    co_independent_population: np.ndarray
    co_component_species_codes: np.ndarray
    co_component_atomic_masses_amu: np.ndarray
    co_molecular_mass_amu: float
    co_leading_coefficient: float
    co_temperature: np.ndarray
    co_equation_densities: np.ndarray
    co_transformed_equation_densities: np.ndarray
    co_no_ground_neutral_partitions: np.ndarray
    co_normalization: np.ndarray
    hard_coded_molecular_atomic_masses_sha256: str
    fixed_eos_call_count: int
    molecular_solve_call_count: int
    reused_fixed_molecular_arrays: bool
    edge_grid_call_count: int
    fixed_population_state: Any
    molecular_call_arguments: dict[str, Any]
    molecular_call_outputs: tuple[Any, Any, Any, Any]
    molecular_call_diagnostics: dict[str, Any]
    molecular_call_caller_name: str
    molecular_call_caller_module: str
    molecular_call_caller_file: str


@dataclass(frozen=True)
class MolecularRouteBoundaryCheckpoint:
    """Executed evidence for the diagnostic and atom-only fallback seams."""

    live_shape_error_type: str
    live_shape_error_message: str
    padded_molecule_code_shape: tuple[int, ...]
    active_molecular_population_shape: tuple[int, ...]
    h2_temperature_k: np.ndarray
    h2_mixed_catalog_input: np.ndarray
    h2_all_zero_catalog_input: np.ndarray
    h2_mixed_output: np.ndarray
    h2_all_zero_output: np.ndarray
    h2_no_catalog_output: np.ndarray
    fallback_fixed_eos_call_count: int
    fallback_molecular_solve_call_count: int


def configure_local_data_paths() -> None:
    """Point exact loaders at this repository's self-contained static data."""

    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(STATIC_DATA_ROOT)
    os.environ["PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE"] = str(SYNTHESIS_ATOMIC_MASSES)


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load independent arrays without leaving an NPZ handle open."""

    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def validate_molecular_inputs(
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate the Chapter 4 input-only depth track without renormalizing it."""

    required_depth_fields = (
        "column_mass",
        "temperature",
        "gas_pressure",
        "electron_density_seed",
        "microturbulence",
    )
    missing = [
        name
        for name in (*required_depth_fields, "elemental_abundances")
        if name not in inputs
    ]
    if missing:
        raise ValueError(f"molecular input fields are missing: {missing}")

    arrays = {name: np.asarray(value).copy() for name, value in inputs.items()}
    depth_count = int(np.asarray(arrays["temperature"]).size)
    for name in required_depth_fields:
        values = np.asarray(arrays[name], dtype=np.float64)
        if values.shape != (depth_count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite ({depth_count},) array")
        if name == "microturbulence":
            if np.any(values < 0.0):
                raise ValueError("microturbulence must be nonnegative")
        elif np.any(values <= 0.0):
            raise ValueError(f"{name} must be strictly positive")
        arrays[name] = values

    if np.any(np.diff(arrays["column_mass"]) <= 0.0):
        raise ValueError("column_mass must increase from outer to inner depth")

    abundances = np.asarray(
        arrays["elemental_abundances"],
        dtype=np.float64,
    )
    if abundances.shape != (99,):
        raise ValueError("elemental_abundances must have shape (99,)")
    if np.any(~np.isfinite(abundances)) or np.any(abundances < 0.0):
        raise ValueError("elemental_abundances must be finite and nonnegative")
    if not np.isfinite(np.sum(abundances)) or np.sum(abundances) <= 0.0:
        raise ValueError("elemental_abundances must have a positive sum")
    arrays["elemental_abundances"] = abundances
    return arrays


def load_molecular_inputs() -> dict[str, np.ndarray]:
    """Return the controlled six-depth, input-only molecular fixture."""

    return validate_molecular_inputs(load_npz_arrays(MOLECULAR_INPUT_FIXTURE))


def build_molecular_atmosphere(
    inputs: Mapping[str, np.ndarray],
    *,
    pressure_iteration_enabled: bool = True,
):
    """Build the exact atmosphere input object from declared linear arrays."""

    configure_local_data_paths()
    arrays = validate_molecular_inputs(inputs)
    from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere

    temperature = arrays["temperature"]
    abundances = arrays["elemental_abundances"]
    abundance_deck = {
        atomic_number: (
            float(abundances[atomic_number - 1])
            if atomic_number <= 2
            else float(np.log10(abundances[atomic_number - 1]))
        )
        for atomic_number in range(1, 100)
    }
    zeros = np.zeros_like(temperature)
    return ModelAtmosphere(
        column_mass=arrays["column_mass"].copy(),
        temperature=temperature.copy(),
        gas_pressure=arrays["gas_pressure"].copy(),
        electron_density=arrays["electron_density_seed"].copy(),
        rosseland_opacity=np.ones_like(temperature),
        radiative_acceleration=zeros.copy(),
        microturbulence=arrays["microturbulence"].copy(),
        convective_flux=zeros.copy(),
        convective_velocity=zeros.copy(),
        metadata={
            "pressure_iteration_enabled": ("1" if pressure_iteration_enabled else "0")
        },
        fixed_column_abundance_values=abundance_deck,
    )


def build_atmosphere_molecular_config(
    inputs: Mapping[str, np.ndarray],
    *,
    pressure_iteration_enabled: bool = True,
):
    """Build the exact molecule-enabled population-stage configuration."""

    configure_local_data_paths()
    from payne_zero_atmosphere.config import (
        AtmosphereConfig,
        AtmosphereInput,
        AtmosphereOutput,
    )

    atmosphere = build_molecular_atmosphere(
        inputs,
        pressure_iteration_enabled=pressure_iteration_enabled,
    )
    return AtmosphereConfig(
        inputs=AtmosphereInput(
            initial_atmosphere=atmosphere,
            molecules_path=ATMOSPHERE_MOLECULE_CATALOG,
        ),
        outputs=AtmosphereOutput(),
        iterations=1,
        enable_molecules=True,
        enable_convection=False,
    )


def compute_atmosphere_molecular_state(
    inputs: Mapping[str, np.ndarray],
    *,
    structured_handoff: bool = False,
    pressure_iteration_enabled: bool = True,
):
    """Run the exact local atmosphere molecular population stage."""

    configure_local_data_paths()
    from payne_zero_atmosphere.runner import (
        prepare_population_state,
        prepare_structured_handoff_population_state,
    )

    config = build_atmosphere_molecular_config(
        inputs,
        pressure_iteration_enabled=pressure_iteration_enabled,
    )
    prepare = (
        prepare_structured_handoff_population_state
        if structured_handoff
        else prepare_population_state
    )
    return prepare(config, temperature_iteration_index=1)


def _initialize_unsolved_atmosphere_molecular_state(
    inputs: Mapping[str, np.ndarray],
):
    """Allocate the exact local atmosphere state without starting chemistry."""

    configure_local_data_paths()
    from payne_zero_atmosphere.molecular_data import (
        read_molecular_equilibrium_catalog,
    )
    from payne_zero_atmosphere.molecular_equilibrium import (
        initialize_molecular_equilibrium_state,
    )
    from payne_zero_atmosphere.run_setup import resolve_run_setup
    from payne_zero_atmosphere.runtime_state import (
        build_runtime_state,
        update_charge_square_density,
    )

    config = build_atmosphere_molecular_config(inputs)
    setup = resolve_run_setup(config)
    runtime_state = build_runtime_state(setup.atmosphere)
    update_charge_square_density(
        thermal_energy_erg=setup.atmosphere.thermal_energy_erg,
        state=runtime_state,
    )
    catalog = read_molecular_equilibrium_catalog(ATMOSPHERE_MOLECULE_CATALOG)
    molecular_state = initialize_molecular_equilibrium_state(
        temperature_k=setup.atmosphere.temperature,
        thermal_energy_erg=setup.atmosphere.thermal_energy_erg,
        gas_pressure=runtime_state.gas_pressure,
        runtime_state=runtime_state,
        catalog=catalog,
    )
    return setup, runtime_state, molecular_state


def _atmosphere_restart_seed(
    molecular_state,
    layer_index: int,
) -> np.ndarray:
    """Construct the exact first-layer density seed at any declared depth."""

    from payne_zero_atmosphere.constants import (
        BOLTZMANN_ERG_PER_K_REFERENCE,
    )
    from payne_zero_atmosphere.molecular_equilibrium import (
        _abundance_vector_for_layer,
    )

    catalog = molecular_state.catalog
    equation_count = int(catalog.equation_count)
    temperature = float(molecular_state.temperature_k[layer_index])
    particle_density = float(
        molecular_state.gas_pressure[layer_index]
        / max(temperature * BOLTZMANN_ERG_PER_K_REFERENCE, 1.0e-300)
    )
    nuclei_density = (
        particle_density if temperature < 4000.0 else particle_density / 2.0
    )
    electron_density = nuclei_density / 10.0
    abundance = _abundance_vector_for_layer(molecular_state, layer_index)
    seed = np.zeros(equation_count, dtype=np.float64)
    seed[0] = nuclei_density
    seed[1:] = electron_density * abundance[1:]
    if int(catalog.equation_species_codes[equation_count - 1]) == 100:
        seed[equation_count - 1] = electron_density
    return seed


def _atmosphere_residual_and_jacobian(
    molecular_state,
    layer_index: int,
    density_cm3: np.ndarray,
    formation_constants: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the exact compiled 23-row atmosphere linearization."""

    from payne_zero_atmosphere.constants import (
        BOLTZMANN_ERG_PER_K_REFERENCE,
    )
    from payne_zero_atmosphere.molecular_equilibrium import (
        _abundance_vector_for_layer,
        _catalog_kernel_cache,
        _newton_matrix_kernel,
    )

    catalog = molecular_state.catalog
    equation_count = int(catalog.equation_count)
    abundance = _abundance_vector_for_layer(molecular_state, layer_index)
    cache = _catalog_kernel_cache(catalog)
    particle_density = float(
        molecular_state.gas_pressure[layer_index]
        / max(
            molecular_state.temperature_k[layer_index] * BOLTZMANN_ERG_PER_K_REFERENCE,
            1.0e-300,
        )
    )
    jacobian, residual = _newton_matrix_kernel(
        equation_count,
        int(catalog.molecule_count),
        np.ascontiguousarray(density_cm3, dtype=np.float64),
        abundance,
        np.ascontiguousarray(formation_constants, dtype=np.float64),
        -particle_density,
        cache.equation_species_codes,
        cache.component_start_indices,
        cache.component_equation_indices,
    )
    residual_scale = np.maximum(
        np.abs(abundance * float(density_cm3[0])),
        1.0,
    )
    residual_scale[0] = max(particle_density, 1.0)
    if int(catalog.equation_species_codes[equation_count - 1]) == 100:
        residual_scale[equation_count - 1] = max(
            abs(float(density_cm3[equation_count - 1])),
            1.0,
        )
    return (
        np.asarray(residual, dtype=np.float64),
        np.asarray(jacobian, dtype=np.float64),
        residual_scale,
    )


def _central_difference_jacobian(
    residual_function: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    *,
    relative_step: float,
) -> np.ndarray:
    """Differentiate a vector function without using its analytic Jacobian."""

    density = np.asarray(point, dtype=np.float64)
    jacobian = np.empty((density.size, density.size), dtype=np.float64)
    for column_index in range(density.size):
        step = relative_step * max(abs(float(density[column_index])), 1.0)
        plus = density.copy()
        minus = density.copy()
        plus[column_index] += step
        minus[column_index] -= step
        jacobian[:, column_index] = (
            residual_function(plus) - residual_function(minus)
        ) / (2.0 * step)
    return jacobian


def _equation_labels(
    species_codes: np.ndarray,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    """Name the exact particle, elemental, and electron rows and columns."""

    rows: list[str] = []
    columns: list[str] = []
    for species_code in np.asarray(species_codes, dtype=np.int64):
        if species_code == 0:
            rows.append("particle budget")
            columns.append("total nuclei")
        elif species_code == 100:
            rows.append("charge budget")
            columns.append("electrons")
        else:
            rows.append(f"Z={species_code} budget")
            columns.append(f"Z={species_code} free density")
    return tuple(rows), tuple(columns)


def atmosphere_jacobian_checkpoint(
    inputs: Mapping[str, np.ndarray],
    *,
    depth_index: int = 2,
    finite_difference_relative_step: float = 1.0e-5,
) -> AtmosphereJacobianCheckpoint:
    """Check the exact atmosphere Jacobian at one independent physical depth.

    The density vector is first solved with the same formation constants that
    are then held fixed in both derivative evaluations.  The finite difference
    treats the exact residual kernel as a black box and never reads a golden.
    """

    arrays = validate_molecular_inputs(inputs)
    if not 0 <= int(depth_index) < arrays["temperature"].size:
        raise IndexError("depth_index is outside the molecular input track")
    if (
        not np.isfinite(finite_difference_relative_step)
        or finite_difference_relative_step <= 0.0
    ):
        raise ValueError("finite_difference_relative_step must be positive")

    _, runtime_state, molecular_state = _initialize_unsolved_atmosphere_molecular_state(
        arrays
    )
    from payne_zero_atmosphere.molecular_equilibrium import (
        compute_equilibrium_constants_for_layer,
        solve_molecular_equilibrium_layer,
    )

    layer_index = int(depth_index)
    seed = _atmosphere_restart_seed(molecular_state, layer_index)
    runtime_state.electron_density[layer_index] = seed[-1]
    formation_constants = compute_equilibrium_constants_for_layer(
        molecular_state,
        layer_index,
    )
    density = solve_molecular_equilibrium_layer(
        molecular_state,
        layer_index,
        seed,
    )
    residual, analytic, residual_scale = _atmosphere_residual_and_jacobian(
        molecular_state,
        layer_index,
        density,
        formation_constants,
    )

    def residual_at(candidate: np.ndarray) -> np.ndarray:
        return _atmosphere_residual_and_jacobian(
            molecular_state,
            layer_index,
            candidate,
            formation_constants,
        )[0]

    finite_difference = _central_difference_jacobian(
        residual_at,
        density,
        relative_step=float(finite_difference_relative_step),
    )
    absolute_error = np.abs(analytic - finite_difference)
    column_scale = np.maximum(
        np.max(np.abs(analytic), axis=0),
        1.0e-300,
    )
    relative_error = absolute_error / column_scale[np.newaxis, :]
    species_codes = np.asarray(
        molecular_state.catalog.equation_species_codes[
            : molecular_state.catalog.equation_count
        ],
        dtype=np.int64,
    )
    rows, columns = _equation_labels(species_codes)
    block_indices = np.asarray(
        [
            0,
            int(np.flatnonzero(species_codes == 6)[0]),
            int(np.flatnonzero(species_codes == 8)[0]),
        ],
        dtype=np.int64,
    )
    return AtmosphereJacobianCheckpoint(
        depth_index=layer_index,
        temperature_k=float(arrays["temperature"][layer_index]),
        gas_pressure=float(arrays["gas_pressure"][layer_index]),
        density_cm3=np.asarray(density, dtype=np.float64),
        formation_constants=np.asarray(
            formation_constants,
            dtype=np.float64,
        ),
        residual=residual,
        residual_scale=residual_scale,
        analytic_jacobian=analytic,
        finite_difference_jacobian=finite_difference,
        row_species_codes=species_codes,
        row_labels=rows,
        column_labels=columns,
        absolute_error=absolute_error,
        column_scale_relative_error=relative_error,
        max_absolute_error=float(np.max(absolute_error)),
        max_column_scale_relative_error=float(np.max(relative_error)),
        particle_carbon_oxygen_indices=block_indices,
        particle_carbon_oxygen_analytic_block=analytic[
            np.ix_(block_indices, block_indices)
        ],
        particle_carbon_oxygen_finite_difference_block=finite_difference[
            np.ix_(block_indices, block_indices)
        ],
    )


def _linear_solve_evidence(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
) -> AtmosphereLinearSolveEvidence:
    """Execute the production direct-solve/least-squares branch boundary."""

    coefficients = np.asarray(matrix, dtype=np.float64)
    values = np.asarray(right_hand_side, dtype=np.float64)
    try:
        step = np.linalg.solve(coefficients, values)
        branch = "solve"
    except np.linalg.LinAlgError:
        step, *_ = np.linalg.lstsq(coefficients, values, rcond=None)
        branch = "lstsq"
    linear_residual = coefficients @ step - values
    return AtmosphereLinearSolveEvidence(
        branch=branch,
        matrix_rank=int(np.linalg.matrix_rank(coefficients)),
        matrix=coefficients.copy(),
        right_hand_side=values.copy(),
        step=np.asarray(step, dtype=np.float64),
        step_norm=float(np.linalg.norm(step)),
        linear_residual_norm=float(np.linalg.norm(linear_residual)),
    )


def atmosphere_linear_solve_checkpoint(
    inputs: Mapping[str, np.ndarray],
    *,
    depth_index: int = 2,
) -> tuple[
    AtmosphereLinearSolveEvidence,
    AtmosphereLinearSolveEvidence,
]:
    """Run the physical direct branch and a controlled singular fallback."""

    checkpoint = atmosphere_jacobian_checkpoint(
        inputs,
        depth_index=depth_index,
    )
    direct = _linear_solve_evidence(
        checkpoint.analytic_jacobian,
        checkpoint.residual,
    )
    rank_deficient = checkpoint.analytic_jacobian.copy()
    rank_deficient_right_hand_side = checkpoint.residual.copy()
    rank_deficient[-1] = 0.0
    rank_deficient_right_hand_side[-1] = 0.0
    fallback = _linear_solve_evidence(
        rank_deficient,
        rank_deficient_right_hand_side,
    )
    if direct.branch != "solve" or fallback.branch != "lstsq":
        raise RuntimeError("controlled atmosphere linear branches were not taken")
    return direct, fallback


def atmosphere_newton_update_witness() -> AtmosphereNewtonUpdateWitness:
    """Trigger every exact atmosphere damping/positivity branch in order."""

    configure_local_data_paths()
    from payne_zero_atmosphere.molecular_equilibrium import (
        _newton_update_kernel,
    )

    labels = (
        "ordinary accept",
        "0.69 sign damping",
        "absolute reflection",
        "one-percent fallback",
        "later shared-scale fallback",
    )
    branches = (
        "accept",
        "sign-damped accept",
        "absolute reflection",
        "one-percent fallback",
        "shared-scale fallback",
    )
    old_density = np.asarray(
        [10.0, 10.0, 10.0, 10.0, 20.0],
        dtype=np.float64,
    )
    previous_delta = np.asarray(
        [0.0, 1.0, 0.0, -1.0, 0.0],
        dtype=np.float64,
    )
    raw_delta = np.asarray(
        [1.0, -2.0, 15.0, 10.0 / 0.69, 20.0],
        dtype=np.float64,
    )

    sign_damped = previous_delta * raw_delta < 0.0
    effective_delta = raw_delta.copy()
    effective_delta[sign_damped] *= 0.69
    candidate = old_density - effective_delta
    scale_before = np.empty(old_density.size, dtype=np.float64)
    scale_after = np.empty(old_density.size, dtype=np.float64)
    expected_density = np.empty(old_density.size, dtype=np.float64)
    scale = 100.0
    for index in range(old_density.size):
        scale_before[index] = scale
        if abs(candidate[index]) >= old_density[index] / 100.0:
            expected_density[index] = abs(candidate[index])
        else:
            expected_density[index] = old_density[index] / scale
            if sign_damped[index]:
                scale = float(np.sqrt(scale))
        scale_after[index] = scale

    returned_density = old_density.copy()
    returned_previous_delta = previous_delta.copy()
    returned_delta = raw_delta.copy()
    still_iterating = bool(
        _newton_update_kernel(
            old_density.size,
            returned_density,
            returned_previous_delta,
            returned_delta,
        )
    )
    exact_trace_matches = (
        np.array_equal(returned_density, expected_density)
        and np.array_equal(returned_delta, effective_delta)
        and np.array_equal(returned_previous_delta, effective_delta)
    )
    if not exact_trace_matches:
        raise RuntimeError("exact Newton update disagrees with its branch trace")
    convergence_probe_relative_update = np.asarray(
        [
            1.0e-4,
            np.nextafter(np.float64(1.0e-4), np.float64(np.inf)),
        ],
        dtype=np.float64,
    )
    convergence_probe_still_iterating = np.zeros(2, dtype=np.bool_)
    convergence_probe_returned_density = np.zeros(2, dtype=np.float64)
    for index, relative_update in enumerate(convergence_probe_relative_update):
        probe_density = np.ones(1, dtype=np.float64)
        probe_previous = np.zeros(1, dtype=np.float64)
        probe_delta = np.asarray([relative_update], dtype=np.float64)
        convergence_probe_still_iterating[index] = _newton_update_kernel(
            1,
            probe_density,
            probe_previous,
            probe_delta,
        )
        convergence_probe_returned_density[index] = probe_density[0]
    return AtmosphereNewtonUpdateWitness(
        labels=labels,
        branches=branches,
        old_density=old_density,
        previous_delta=previous_delta,
        raw_delta=raw_delta,
        effective_delta=returned_delta,
        candidate=candidate,
        returned_density=returned_density,
        sign_damped=sign_damped,
        scale_before=scale_before,
        scale_after=scale_after,
        still_iterating=still_iterating,
        convergence_probe_relative_update=(convergence_probe_relative_update),
        convergence_probe_returned_density=(convergence_probe_returned_density),
        convergence_probe_still_iterating=(convergence_probe_still_iterating),
    )


def _solve_atmosphere_layer_with_trace(
    molecular_state,
    layer_index: int,
    seed: np.ndarray,
) -> tuple[np.ndarray, int, bool, float, float]:
    """Run the exact one-depth kernels while retaining convergence evidence."""

    from payne_zero_atmosphere.molecular_equilibrium import (
        _MAX_NEWTON_ITERATIONS,
        _abundance_vector_for_layer,
        _catalog_kernel_cache,
        _compute_equilibrium_constants_for_layer_compiled,
        _newton_matrix_kernel,
        _newton_update_kernel,
    )

    catalog = molecular_state.catalog
    equation_count = int(catalog.equation_count)
    cache = _catalog_kernel_cache(catalog)
    density = np.asarray(seed, dtype=np.float64).copy()
    previous_delta = np.zeros(equation_count, dtype=np.float64)
    abundance = _abundance_vector_for_layer(molecular_state, layer_index)
    constants = _compute_equilibrium_constants_for_layer_compiled(
        molecular_state,
        layer_index,
    )
    from payne_zero_atmosphere.constants import (
        BOLTZMANN_ERG_PER_K_REFERENCE,
    )

    particle_density = float(
        molecular_state.gas_pressure[layer_index]
        / max(
            molecular_state.temperature_k[layer_index] * BOLTZMANN_ERG_PER_K_REFERENCE,
            1.0e-300,
        )
    )
    converged = False
    iteration_count = 0
    for iteration_count in range(1, int(_MAX_NEWTON_ITERATIONS) + 1):
        jacobian, residual = _newton_matrix_kernel(
            equation_count,
            int(catalog.molecule_count),
            density,
            abundance,
            constants,
            -particle_density,
            cache.equation_species_codes,
            cache.component_start_indices,
            cache.component_equation_indices,
        )
        try:
            delta = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
        still_iterating = _newton_update_kernel(
            equation_count,
            density,
            previous_delta,
            np.ascontiguousarray(delta, dtype=np.float64),
        )
        if not still_iterating:
            converged = True
            break

    final_residual, _, residual_scale = _atmosphere_residual_and_jacobian(
        molecular_state,
        layer_index,
        density,
        constants,
    )
    return (
        density,
        iteration_count,
        converged,
        float(np.max(np.abs(final_residual))),
        float(np.max(np.abs(final_residual) / residual_scale)),
    )


def _one_depth_inputs(
    inputs: Mapping[str, np.ndarray],
    depth_index: int,
) -> dict[str, np.ndarray]:
    """Extract one physical depth while retaining the shared abundance deck."""

    depth_fields = {
        "column_mass",
        "temperature",
        "gas_pressure",
        "electron_density_seed",
        "microturbulence",
    }
    return {
        name: (
            np.asarray(value)[depth_index : depth_index + 1].copy()
            if name in depth_fields
            else np.asarray(value).copy()
        )
        for name, value in inputs.items()
    }


def atmosphere_continuation_checkpoint(
    inputs: Mapping[str, np.ndarray],
) -> AtmosphereContinuationCheckpoint:
    """Compare exact ordered pressure continuation with six cold restarts."""

    arrays = validate_molecular_inputs(inputs)
    _, runtime_state, molecular_state = _initialize_unsolved_atmosphere_molecular_state(
        arrays
    )
    from payne_zero_atmosphere.constants import ATOMIC_MASS_GRAM_REFERENCE

    depth_count = arrays["temperature"].size
    equation_count = int(molecular_state.catalog.equation_count)
    continuation_seed = np.zeros(
        (depth_count, equation_count),
        dtype=np.float64,
    )
    continuation_solution = np.zeros_like(continuation_seed)
    continuation_iterations = np.zeros(depth_count, dtype=np.int64)
    continuation_converged = np.zeros(depth_count, dtype=np.bool_)
    continuation_residual_abs = np.zeros(depth_count, dtype=np.float64)
    continuation_residual_scaled = np.zeros(depth_count, dtype=np.float64)

    current_seed = _atmosphere_restart_seed(molecular_state, 0)
    runtime_state.electron_density[0] = current_seed[-1]
    for depth_index in range(depth_count):
        if depth_index > 0:
            pressure_ratio = float(
                arrays["gas_pressure"][depth_index]
                / arrays["gas_pressure"][depth_index - 1]
            )
            current_seed = continuation_solution[depth_index - 1] * (pressure_ratio)
            runtime_state.electron_density[depth_index] = (
                runtime_state.electron_density[depth_index - 1] * pressure_ratio
            )
        continuation_seed[depth_index] = current_seed
        (
            solved,
            iteration_count,
            converged,
            residual_abs,
            residual_scaled,
        ) = _solve_atmosphere_layer_with_trace(
            molecular_state,
            depth_index,
            current_seed,
        )
        continuation_solution[depth_index] = solved
        continuation_iterations[depth_index] = iteration_count
        continuation_converged[depth_index] = converged
        continuation_residual_abs[depth_index] = residual_abs
        continuation_residual_scaled[depth_index] = residual_scaled
        runtime_state.total_nuclei_number_density[depth_index] = solved[0]
        runtime_state.mass_density[depth_index] = (
            solved[0]
            * runtime_state.mean_nuclear_mass_amu[depth_index]
            * ATOMIC_MASS_GRAM_REFERENCE
        )
        runtime_state.electron_density[depth_index] = solved[-1]

    expected_continuation_seed = continuation_seed.copy()
    pressure_ratios = np.ones(depth_count, dtype=np.float64)
    continuation_seed_equal = np.ones(depth_count, dtype=np.bool_)
    for depth_index in range(1, depth_count):
        pressure_ratios[depth_index] = (
            arrays["gas_pressure"][depth_index]
            / arrays["gas_pressure"][depth_index - 1]
        )
        expected_continuation_seed[depth_index] = (
            continuation_solution[depth_index - 1] * pressure_ratios[depth_index]
        )
        continuation_seed_equal[depth_index] = np.array_equal(
            continuation_seed[depth_index],
            expected_continuation_seed[depth_index],
        )

    restart_seed = np.zeros_like(continuation_seed)
    restart_solution = np.zeros_like(continuation_seed)
    restart_iterations = np.zeros(depth_count, dtype=np.int64)
    restart_converged = np.zeros(depth_count, dtype=np.bool_)
    restart_residual_abs = np.zeros(depth_count, dtype=np.float64)
    restart_residual_scaled = np.zeros(depth_count, dtype=np.float64)
    for depth_index in range(depth_count):
        one_depth = _one_depth_inputs(arrays, depth_index)
        _, one_runtime, one_molecular = _initialize_unsolved_atmosphere_molecular_state(
            one_depth
        )
        seed = _atmosphere_restart_seed(one_molecular, 0)
        one_runtime.electron_density[0] = seed[-1]
        restart_seed[depth_index] = seed
        (
            solved,
            iteration_count,
            converged,
            residual_abs,
            residual_scaled,
        ) = _solve_atmosphere_layer_with_trace(
            one_molecular,
            0,
            seed,
        )
        restart_solution[depth_index] = solved
        restart_iterations[depth_index] = iteration_count
        restart_converged[depth_index] = converged
        restart_residual_abs[depth_index] = residual_abs
        restart_residual_scaled[depth_index] = residual_scaled

    return AtmosphereContinuationCheckpoint(
        depth_index=np.arange(depth_count, dtype=np.int64),
        pressure_ratio=pressure_ratios,
        continuation_seed=continuation_seed,
        expected_continuation_seed=expected_continuation_seed,
        continuation_seed_equal=continuation_seed_equal,
        continuation_solution=continuation_solution,
        continuation_iteration_count=continuation_iterations,
        continuation_converged=continuation_converged,
        continuation_residual_max_abs=continuation_residual_abs,
        continuation_residual_max_scaled=continuation_residual_scaled,
        restart_seed=restart_seed,
        restart_solution=restart_solution,
        restart_iteration_count=restart_iterations,
        restart_converged=restart_converged,
        restart_residual_max_abs=restart_residual_abs,
        restart_residual_max_scaled=restart_residual_scaled,
    )


def load_synthesis_eos_tables():
    """Load exact synthesis EOS tables on CPU in explicit `float64`."""

    configure_local_data_paths()
    import torch

    from book.chapter03_runtime import load_synthesis_tables

    return load_synthesis_tables(
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def compute_synthesis_molecular_state(
    inputs: Mapping[str, np.ndarray],
    *,
    fixed_electron_density: bool,
    mass_density: np.ndarray | None = None,
):
    """Run the exact local synthesis full or fixed-electron molecular route."""

    arrays = validate_molecular_inputs(inputs)
    configure_local_data_paths()
    from payne_zero_synthesis.equation_of_state import (
        solve_population_state,
        solve_population_state_at_electron_density,
    )

    tables = load_synthesis_eos_tables()
    common = {
        "temperature": arrays["temperature"],
        "gas_pressure": arrays["gas_pressure"],
        "elemental_abundances": arrays["elemental_abundances"],
        "tables": tables,
        "mean_nuclear_mass_amu": None,
        "molecules": True,
        "molecules_path": SYNTHESIS_MOLECULE_CATALOG,
    }
    if fixed_electron_density:
        return solve_population_state_at_electron_density(
            **common,
            electron_density=arrays["electron_density_seed"],
            mass_density=mass_density,
        )
    if mass_density is not None:
        raise ValueError("mass_density belongs only to the fixed route")
    return solve_population_state(
        **common,
        electron_density_seed=arrays["electron_density_seed"],
        max_iter=200,
        tol=1.0e-4,
    )


def molecular_catalog_summary() -> dict[str, np.ndarray]:
    """Return exact active extents and identities for both molecular catalogs."""

    configure_local_data_paths()
    from payne_zero_atmosphere.molecular_data import (
        read_molecular_equilibrium_catalog,
    )
    from payne_zero_synthesis.molecular_equilibrium import read_molecule_table

    atmosphere = read_molecular_equilibrium_catalog(ATMOSPHERE_MOLECULE_CATALOG)
    synthesis = read_molecule_table(SYNTHESIS_MOLECULE_CATALOG)
    atmosphere_codes = np.asarray(
        atmosphere.molecule_codes[: atmosphere.molecule_count],
        dtype=np.float64,
    )
    synthesis_codes = np.asarray(
        synthesis.molecule_codes[: synthesis.molecule_count],
        dtype=np.float64,
    )
    atmosphere_keys = {int(round(float(code) * 100.0)) for code in atmosphere_codes}
    synthesis_keys = {int(round(float(code) * 100.0)) for code in synthesis_codes}
    synthesis_only = np.array(
        sorted(synthesis_keys - atmosphere_keys),
        dtype=np.int64,
    )
    synthesis_component_count = int(
        synthesis.component_start_indices[synthesis.molecule_count]
    )
    atmosphere_rows = {
        int(round(float(code) * 100.0)): row
        for row, code in enumerate(atmosphere_codes)
    }
    synthesis_rows = {
        int(round(float(code) * 100.0)): row for row, code in enumerate(synthesis_codes)
    }
    shared_keys = np.asarray(
        sorted(atmosphere_keys & synthesis_keys),
        dtype=np.int64,
    )

    def record_semantics(catalog, row: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(catalog.component_start_indices[row])
        stop = int(catalog.component_start_indices[row + 1])
        equation_indices = np.asarray(
            catalog.component_equation_indices[start:stop],
            dtype=np.int64,
        )
        component_species = np.full(equation_indices.size, 101, dtype=np.int32)
        ordinary = equation_indices < int(catalog.equation_count)
        component_species[ordinary] = np.asarray(
            catalog.equation_species_codes[equation_indices[ordinary]],
            dtype=np.int32,
        )
        coefficients = np.asarray(
            catalog.equilibrium_coefficients[:, row],
            dtype=np.float64,
        )
        return component_species, coefficients

    semantic_mismatch = np.zeros(shared_keys.size, dtype=np.bool_)
    row_reordered = np.zeros(shared_keys.size, dtype=np.bool_)
    for index, key in enumerate(shared_keys):
        atmosphere_row = atmosphere_rows[int(key)]
        synthesis_row = synthesis_rows[int(key)]
        atmosphere_record = record_semantics(atmosphere, atmosphere_row)
        synthesis_record = record_semantics(synthesis, synthesis_row)
        semantic_mismatch[index] = not (
            np.array_equal(atmosphere_record[0], synthesis_record[0])
            and np.array_equal(atmosphere_record[1], synthesis_record[1])
        )
        row_reordered[index] = atmosphere_row != synthesis_row
    return {
        "atmosphere_counts": np.array(
            [
                atmosphere.molecule_count,
                atmosphere.equation_count,
                atmosphere.component_count,
            ],
            dtype=np.int64,
        ),
        "synthesis_counts": np.array(
            [
                synthesis.molecule_count,
                synthesis.equation_count,
                synthesis_component_count,
            ],
            dtype=np.int64,
        ),
        "atmosphere_molecule_codes": atmosphere_codes,
        "synthesis_molecule_codes": synthesis_codes,
        "synthesis_only_code_keys": synthesis_only,
        "shared_code_keys": shared_keys,
        "shared_semantic_mismatch": semantic_mismatch,
        "shared_row_reordered": row_reordered,
    }


def h2_formation_policy_curves(
    temperature: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the three exact H2 policies on one declared plotting grid."""

    values = np.asarray(temperature, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("temperature must be a nonempty positive finite vector")
    configure_local_data_paths()
    from payne_zero_atmosphere.molecular_equilibrium import (
        hydrogen_molecule_equilibrium_constant,
    )
    from payne_zero_synthesis import molecular_equilibrium
    from payne_zero_synthesis.constants import (
        REFERENCE_BOLTZMANN_EV_PER_K,
    )

    atmosphere = np.asarray(
        [hydrogen_molecule_equilibrium_constant(value) for value in values],
        dtype=np.float64,
    )
    atmosphere[values > 20000.0] = 0.0

    molecule_table = molecular_equilibrium.read_molecule_table(
        SYNTHESIS_MOLECULE_CATALOG
    )
    active_codes = molecule_table.molecule_codes[: molecule_table.molecule_count]
    h2_index = int(np.flatnonzero(np.isclose(active_codes, 101.0))[0])
    # NumPy evaluates both arms of the source's ``where`` expression.  Other
    # catalog rows can therefore overflow above their activation temperature
    # even though the returned inactive values are exactly zero.  That warning
    # is a useful source-level edge test, but it is not part of this plotting
    # adapter's public result.
    with np.errstate(over="ignore", invalid="ignore"):
        synthesis = molecular_equilibrium.polynomial_formation_constants(
            values,
            molecule_table,
        )[:, h2_index]

    thermal_energy_ev = values * REFERENCE_BOLTZMANN_EV_PER_K
    natural_log_temperature = np.log(values)
    provisional = np.exp(
        4.478 / thermal_energy_ev
        - 4.64584e1
        + (
            1.63660e-3
            + (
                -4.93992e-7
                + (
                    1.11822e-10
                    + (-1.49567e-14 + (1.06206e-18 - 3.08720e-23 * values) * values)
                    * values
                )
                * values
            )
            * values
        )
        * values
        - 1.5 * natural_log_temperature
    )
    provisional[values > 9000.0] = 0.0
    return {
        "atmosphere H2 policy": atmosphere,
        "synthesis catalog policy": synthesis,
        "provisional public H2": provisional,
    }


def h2_partition_table_probe(temperature: np.ndarray) -> np.ndarray:
    """Evaluate the exact atmosphere H2 partition-table interpolation policy."""

    values = np.asarray(temperature, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("temperature must be a nonempty positive finite vector")
    configure_local_data_paths()
    from payne_zero_atmosphere.molecular_equilibrium import (
        _interp_hydrogen_molecule_partition,
    )

    return np.asarray(
        [_interp_hydrogen_molecule_partition(value) for value in values],
        dtype=np.float64,
    )


def molecular_species_mapping() -> dict[str, np.ndarray]:
    """Return the lossless line-species to equilibrium-code address map."""

    configure_local_data_paths()
    from payne_zero_synthesis import molecular_equilibrium

    line_species_codes = np.asarray(
        molecular_equilibrium.supported_molecular_species_codes(),
        dtype=np.int64,
    )
    offsets = [0]
    equilibrium_codes: list[float] = []
    for line_species_code in line_species_codes:
        mapped_codes = molecular_equilibrium._SPECIES_CODE_TO_MOLECULE_CODES[
            int(line_species_code)
        ]
        equilibrium_codes.extend(float(code) for code in mapped_codes)
        offsets.append(len(equilibrium_codes))
    return {
        "line_species_codes": line_species_codes,
        "equilibrium_code_offsets": np.asarray(offsets, dtype=np.int64),
        "equilibrium_codes": np.asarray(
            equilibrium_codes,
            dtype=np.float64,
        ),
        "public_columns": line_species_codes // 6 - 1,
    }


def molecular_route_boundary_checkpoint(
    inputs: Mapping[str, np.ndarray],
    atmosphere_result: Any | None = None,
) -> MolecularRouteBoundaryCheckpoint:
    """Execute the live/debug, global-H2, and atom-only fallback policies."""

    arrays = validate_molecular_inputs(inputs)
    configure_local_data_paths()
    from payne_zero_atmosphere import synthesis_bridge
    from payne_zero_synthesis import equation_of_state
    from payne_zero_synthesis import molecular_equilibrium
    from payne_zero_synthesis import pipeline

    result = (
        compute_atmosphere_molecular_state(arrays)
        if atmosphere_result is None
        else atmosphere_result
    )
    live_error: ValueError | None = None
    try:
        synthesis_bridge.structured_atmosphere_from_runtime_state(
            atmosphere=result.setup.atmosphere,
            runtime_state=result.runtime_state,
            molecular_state=result.molecular_state,
        )
    except ValueError as error:
        live_error = error
    if live_error is None:
        raise RuntimeError("live bridge unexpectedly accepted active/padded shapes")

    temperature = np.asarray([8000.0, 9500.0, 7000.0], dtype=np.float64)
    packed = np.zeros((temperature.size, 1006), dtype=np.float64)
    packed[:, 0] = np.asarray([2.0e14, 3.0e14, 4.0e14], dtype=np.float64)
    codes = np.asarray([101.0], dtype=np.float64)
    mixed = np.asarray([[5.0e8], [0.0], [7.0e8]], dtype=np.float64)
    zero = np.zeros_like(mixed)
    h2_arguments = {
        "temperature": temperature,
        "ion_stage_populations_by_packed_slot": packed,
        "molecule_codes": codes,
    }
    mixed_output = synthesis_bridge._molecular_hydrogen_population(
        **h2_arguments,
        molecular_populations=mixed,
    )
    zero_output = synthesis_bridge._molecular_hydrogen_population(
        **h2_arguments,
        molecular_populations=zero,
    )
    no_catalog_output = synthesis_bridge._molecular_hydrogen_population(
        temperature=temperature,
        ion_stage_populations_by_packed_slot=packed,
        molecule_codes=None,
        molecular_populations=None,
    )

    fixed_calls = []
    molecular_calls = []
    original_fixed_solver = equation_of_state.solve_population_state_at_electron_density
    original_molecular_solver = molecular_equilibrium.solve_molecular_equilibrium
    original_edge_builder = pipeline._build_edge_grid

    def force_atom_only_state(*args, **kwargs):
        atom_only_kwargs = dict(kwargs)
        atom_only_kwargs["molecules"] = False
        state = original_fixed_solver(*args, **atom_only_kwargs)
        fixed_calls.append(state)
        return state

    def observe_fallback_solve(*args, **kwargs):
        result = original_molecular_solver(*args, **kwargs)
        molecular_calls.append(result)
        return result

    def build_local_edge_grid():
        return original_edge_builder(SYNTHESIS_CONTINUUM_EDGE_GRID)

    equation_of_state.solve_population_state_at_electron_density = force_atom_only_state
    molecular_equilibrium.solve_molecular_equilibrium = observe_fallback_solve
    pipeline._build_edge_grid = build_local_edge_grid
    try:
        pipeline.build_structured_atmosphere_from_columns(
            temperature=arrays["temperature"],
            column_mass=arrays["column_mass"],
            gas_pressure=arrays["gas_pressure"],
            electron_density=arrays["electron_density_seed"],
            elemental_abundances=arrays["elemental_abundances"],
            mean_nuclear_mass_amu=None,
            microturbulence=arrays["microturbulence"],
            eos_tables=load_synthesis_eos_tables(),
            electron_density_seed=None,
            tol=1.0e-4,
            atomic_masses=pipeline.load_atomic_masses(SYNTHESIS_ATOMIC_MASSES),
            mass_density=None,
            molecular_species_codes=np.asarray([240], dtype=np.int64),
            molecules_path=SYNTHESIS_MOLECULE_CATALOG,
        )
    finally:
        equation_of_state.solve_population_state_at_electron_density = (
            original_fixed_solver
        )
        molecular_equilibrium.solve_molecular_equilibrium = original_molecular_solver
        pipeline._build_edge_grid = original_edge_builder

    return MolecularRouteBoundaryCheckpoint(
        live_shape_error_type=type(live_error).__name__,
        live_shape_error_message=str(live_error),
        padded_molecule_code_shape=tuple(
            result.molecular_state.catalog.molecule_codes.shape
        ),
        active_molecular_population_shape=tuple(
            result.molecular_state.molecular_populations.shape
        ),
        h2_temperature_k=temperature,
        h2_mixed_catalog_input=mixed[:, 0],
        h2_all_zero_catalog_input=zero[:, 0],
        h2_mixed_output=np.asarray(mixed_output, dtype=np.float64),
        h2_all_zero_output=np.asarray(zero_output, dtype=np.float64),
        h2_no_catalog_output=np.asarray(no_catalog_output, dtype=np.float64),
        fallback_fixed_eos_call_count=len(fixed_calls),
        fallback_molecular_solve_call_count=len(molecular_calls),
    )


def build_public_molecular_lane_checkpoint(
    inputs: Mapping[str, np.ndarray],
) -> PublicMolecularLaneCheckpoint:
    """Build the exact public cube once and expose only interpretive evidence."""

    arrays = validate_molecular_inputs(inputs)
    configure_local_data_paths()
    from payne_zero_synthesis import equation_of_state
    from payne_zero_synthesis import molecular_equilibrium
    from payne_zero_synthesis import pipeline

    tables = load_synthesis_eos_tables()
    mapping = molecular_species_mapping()
    line_species_codes = mapping["line_species_codes"]
    public_columns = mapping["public_columns"]
    fixed_states = []
    molecular_solve_calls = []
    edge_grids = []
    original_fixed_solver = equation_of_state.solve_population_state_at_electron_density
    original_edge_builder = pipeline._build_edge_grid
    original_molecular_solver = molecular_equilibrium.solve_molecular_equilibrium

    def observe_fixed_state(*args, **kwargs):
        state = original_fixed_solver(*args, **kwargs)
        fixed_states.append(state)
        return state

    def build_local_edge_grid():
        edge_grid = original_edge_builder(SYNTHESIS_CONTINUUM_EDGE_GRID)
        edge_grids.append(edge_grid)
        return edge_grid

    def observe_molecular_solve(*args, **kwargs):
        caller = inspect.currentframe().f_back
        bound = inspect.signature(original_molecular_solver).bind_partial(
            *args,
            **kwargs,
        )
        bound.apply_defaults()
        effective = dict(bound.arguments)
        caller_requested_diagnostics = bool(effective["return_diagnostics"])
        diagnostic_kwargs = dict(kwargs)
        diagnostic_kwargs["return_diagnostics"] = True
        result = original_molecular_solver(*args, **diagnostic_kwargs)
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("public molecular diagnostic solve returned a bad tuple")
        molecular_solve_calls.append(
            {
                "arguments": effective,
                "outputs": result[:4],
                "diagnostics": result[4],
                "caller_name": caller.f_code.co_name,
                "caller_module": caller.f_globals.get("__name__", ""),
                "caller_file": caller.f_code.co_filename,
            }
        )
        return result if caller_requested_diagnostics else result[:4]

    equation_of_state.solve_population_state_at_electron_density = observe_fixed_state
    pipeline._build_edge_grid = build_local_edge_grid
    molecular_equilibrium.solve_molecular_equilibrium = observe_molecular_solve
    try:
        structured = pipeline.build_structured_atmosphere_from_columns(
            temperature=arrays["temperature"],
            column_mass=arrays["column_mass"],
            gas_pressure=arrays["gas_pressure"],
            electron_density=arrays["electron_density_seed"],
            elemental_abundances=arrays["elemental_abundances"],
            mean_nuclear_mass_amu=None,
            microturbulence=arrays["microturbulence"],
            eos_tables=tables,
            electron_density_seed=None,
            tol=1.0e-4,
            atomic_masses=pipeline.load_atomic_masses(SYNTHESIS_ATOMIC_MASSES),
            mass_density=None,
            molecular_species_codes=line_species_codes,
            molecules_path=SYNTHESIS_MOLECULE_CATALOG,
        )
    finally:
        equation_of_state.solve_population_state_at_electron_density = (
            original_fixed_solver
        )
        pipeline._build_edge_grid = original_edge_builder
        molecular_equilibrium.solve_molecular_equilibrium = original_molecular_solver
    if len(fixed_states) != 1:
        raise RuntimeError("public builder did not own exactly one fixed EOS call")
    if len(molecular_solve_calls) != 1:
        raise RuntimeError("public builder did not own exactly one molecular solve")
    if len(edge_grids) != 1:
        raise RuntimeError("public builder did not load exactly one edge grid")
    fixed_state = fixed_states[0]
    molecular_solve_call = molecular_solve_calls[0]
    molecular_solve_result = molecular_solve_call["outputs"]

    def host_float64(values) -> np.ndarray:
        if hasattr(values, "detach"):
            values = values.detach().cpu().double().numpy()
        return np.asarray(values, dtype=np.float64)

    reused_fixed_molecular_arrays = np.array_equal(
        fixed_state.molecular_populations,
        host_float64(molecular_solve_result[1]),
    ) and np.array_equal(
        fixed_state.molecular_equation_densities,
        host_float64(molecular_solve_result[2]),
    )

    metadata = molecular_equilibrium.molecular_equilibrium_metadata(
        SYNTHESIS_MOLECULE_CATALOG
    )
    molecular_elements = [
        int(atomic_number)
        for atomic_number in metadata.equation_species_codes[: metadata.equation_count]
        if 1 <= int(atomic_number) <= 99
    ]
    grounded_partition_cube = (
        fixed_state.eos.partition_functions.detach().cpu().double().numpy()
    )
    bridge_partition_cube = grounded_partition_cube.copy()
    without_ground = equation_of_state.partition_functions_for_elements(
        arrays["temperature"],
        arrays["gas_pressure"],
        np.asarray(fixed_state.electron_density, dtype=np.float64),
        tables=tables,
        elements=molecular_elements,
        nion=6,
        apply_ground_partition=False,
    )
    for atomic_number, partition_by_stage in without_ground.items():
        stage_count = min(
            partition_by_stage.shape[1],
            bridge_partition_cube.shape[2],
        )
        bridge_partition_cube[:, atomic_number - 1, :stage_count] = partition_by_stage[
            :, :stage_count
        ]

    line_arguments = {
        "temperature": arrays["temperature"],
        "equation_densities": fixed_state.molecular_equation_densities,
        "species_codes": line_species_codes,
        "molecules_path": SYNTHESIS_MOLECULE_CATALOG,
    }
    no_ground_by_species = (
        molecular_equilibrium.molecular_line_populations_by_species_code(
            **line_arguments,
            neutral_partition=bridge_partition_cube[:, :, 0],
        )
    )
    grounded_by_species = (
        molecular_equilibrium.molecular_line_populations_by_species_code(
            **line_arguments,
            neutral_partition=grounded_partition_cube[:, :, 0],
        )
    )
    no_ground = np.column_stack(
        [no_ground_by_species[int(code)] for code in line_species_codes]
    )
    grounded = np.column_stack(
        [grounded_by_species[int(code)] for code in line_species_codes]
    )

    partition_before = np.asarray(
        fixed_state.partition_normalized_populations,
        dtype=np.float64,
    )
    ion_before = np.asarray(
        fixed_state.ion_stage_populations,
        dtype=np.float64,
    )
    partition_after = np.asarray(
        structured["partition_normalized_populations"],
        dtype=np.float64,
    )
    ion_after = np.asarray(
        structured["ion_stage_populations"],
        dtype=np.float64,
    )
    normalized_delta = partition_after - partition_before
    owned_stage5_mask = np.zeros(139, dtype=np.bool_)
    owned_stage5_mask[public_columns] = True

    co_species_index = int(np.flatnonzero(line_species_codes == 276)[0])
    co_offset_start = int(mapping["equilibrium_code_offsets"][co_species_index])
    co_offset_stop = int(mapping["equilibrium_code_offsets"][co_species_index + 1])
    co_codes = mapping["equilibrium_codes"][co_offset_start:co_offset_stop]
    catalog = molecular_equilibrium.read_molecule_table(SYNTHESIS_MOLECULE_CATALOG)
    co_catalog_index = int(
        np.flatnonzero(
            np.abs(catalog.molecule_codes[: catalog.molecule_count] - co_codes[0])
            < 1.0e-3
        )[0]
    )
    hard_coded_masses = np.asarray(
        molecular_equilibrium._ATOMIC_MASSES_FOR_MOLECULES,
        dtype=np.float64,
    )
    co_component_start = int(catalog.component_start_indices[co_catalog_index])
    co_component_stop = int(catalog.component_start_indices[co_catalog_index + 1])
    co_component_equation_indices = np.asarray(
        catalog.component_equation_indices[co_component_start:co_component_stop],
        dtype=np.int64,
    )
    co_component_species_codes = np.asarray(
        [
            catalog.equation_species_codes[equation_index]
            for equation_index in co_component_equation_indices
        ],
        dtype=np.int64,
    )
    co_component_masses = hard_coded_masses[co_component_species_codes - 1]
    co_molecular_mass = float(np.sum(co_component_masses))
    transformed_co_components = np.empty(
        (arrays["temperature"].size, co_component_species_codes.size),
        dtype=np.float64,
    )
    for component_index, (equation_index, atomic_number) in enumerate(
        zip(
            co_component_equation_indices,
            co_component_species_codes,
        )
    ):
        atomic_mass = float(hard_coded_masses[atomic_number - 1])
        denominator = (
            bridge_partition_cube[:, atomic_number - 1, 0]
            * 1.8786e20
            * np.sqrt(
                np.maximum(
                    (atomic_mass * arrays["temperature"]) ** 3,
                    1.0e-300,
                )
            )
        )
        transformed_co_components[:, component_index] = (
            fixed_state.molecular_equation_densities[:, equation_index]
            / np.maximum(denominator, 1.0e-300)
        )
    co_leading_coefficient = float(
        catalog.equilibrium_coefficients[0, co_catalog_index]
    )
    co_independent_population = np.exp(
        co_leading_coefficient / np.maximum(arrays["temperature"] / 11604.5, 1.0e-300)
    )
    for component_index in range(transformed_co_components.shape[1]):
        co_independent_population = (
            co_independent_population * transformed_co_components[:, component_index]
        )
    co_thermal_mass_factor = np.sqrt(
        np.maximum(
            (co_molecular_mass * arrays["temperature"]) ** 3,
            1.0e-300,
        )
    )
    co_normalization = 1.8786e20 * co_thermal_mass_factor
    co_independent_population = co_independent_population * 1.8786e20
    co_independent_population = co_independent_population * co_thermal_mass_factor
    return PublicMolecularLaneCheckpoint(
        structured_atmosphere={
            name: np.asarray(values).copy() for name, values in structured.items()
        },
        line_species_codes=line_species_codes,
        equilibrium_code_offsets=mapping["equilibrium_code_offsets"],
        equilibrium_codes=mapping["equilibrium_codes"],
        public_columns=public_columns,
        partition_cube_before=partition_before.copy(),
        partition_cube_after=partition_after.copy(),
        ion_cube_before=ion_before.copy(),
        ion_cube_after=ion_after.copy(),
        normalized_delta=normalized_delta,
        no_ground_line_populations=no_ground,
        grounded_line_populations=grounded,
        ground_discrimination_mask=no_ground != grounded,
        owned_stage5_mask=owned_stage5_mask,
        co_line_species_code=276,
        co_equilibrium_code=float(co_codes[0]),
        co_public_column=int(public_columns[co_species_index]),
        co_catalog_index=co_catalog_index,
        co_component_equation_indices=co_component_equation_indices,
        co_raw_population=np.asarray(
            fixed_state.molecular_populations[:, co_catalog_index],
            dtype=np.float64,
        ).copy(),
        co_normalized_population=partition_after[
            :, 5, public_columns[co_species_index]
        ].copy(),
        co_independent_population=co_independent_population,
        co_component_species_codes=co_component_species_codes,
        co_component_atomic_masses_amu=co_component_masses,
        co_molecular_mass_amu=co_molecular_mass,
        co_leading_coefficient=co_leading_coefficient,
        co_temperature=arrays["temperature"].copy(),
        co_equation_densities=np.asarray(
            fixed_state.molecular_equation_densities[:, co_component_equation_indices],
            dtype=np.float64,
        ).copy(),
        co_transformed_equation_densities=(transformed_co_components.copy()),
        co_no_ground_neutral_partitions=bridge_partition_cube[
            :, co_component_species_codes - 1, 0
        ].copy(),
        co_normalization=co_normalization,
        hard_coded_molecular_atomic_masses_sha256=hashlib.sha256(
            hard_coded_masses.tobytes(order="C")
        ).hexdigest(),
        fixed_eos_call_count=len(fixed_states),
        molecular_solve_call_count=len(molecular_solve_calls),
        reused_fixed_molecular_arrays=reused_fixed_molecular_arrays,
        edge_grid_call_count=len(edge_grids),
        fixed_population_state=fixed_state,
        molecular_call_arguments=molecular_solve_call["arguments"],
        molecular_call_outputs=molecular_solve_result,
        molecular_call_diagnostics=molecular_solve_call["diagnostics"],
        molecular_call_caller_name=molecular_solve_call["caller_name"],
        molecular_call_caller_module=molecular_solve_call["caller_module"],
        molecular_call_caller_file=molecular_solve_call["caller_file"],
    )
