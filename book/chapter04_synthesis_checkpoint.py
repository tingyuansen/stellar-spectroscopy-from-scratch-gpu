"""Exact one-depth synthesis Newton evidence for Chapter 4.

This adapter deliberately stops after one Newton update.  It calls the staged
production residual and step functions, uses the production synthesis catalog,
and derives its physical state only from repository-owned input fixtures.  It
does not read a golden or an external checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from payne_zero_synthesis import equation_of_state
from payne_zero_synthesis import molecular_equilibrium


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHAPTER04_INPUTS = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter04_molecular_inputs.npz"
)
SYNTHESIS_CATALOG = (
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "source_catalogs"
    / "lines"
    / "molecular_equilibrium_synthesis.npz"
)
ATMOSPHERE_CATALOG = (
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "source_catalogs"
    / "lines"
    / "molecular_equilibrium_atmosphere.npz"
)
SYNTHESIS_TABLES = (
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "synthesis_tables"
    / "partition_saha_tables.npz"
)
GROUND_PARTITION_FIXTURE = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter03_synthesis_eos_state.npz"
)

CPU = torch.device("cpu")
DTYPE = torch.float64
LOG_ZERO_SENTINEL = -700.0
NETWORK_ABUNDANCE_FLOOR = 1.0e-20
POSITIVITY_FLOOR_DIVISOR = 100.0


def _frozen(array, *, dtype=None) -> np.ndarray:
    """Return an independent, read-only NumPy array."""

    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SynthesisNonfiniteProductEvidence:
    """Controlled witness for the residual's nonfinite-product replacement."""

    molecule_index: int
    molecule_code: float
    natural_log_product: float
    natural_log_product_is_finite: bool
    pre_replacement_product: float
    pre_replacement_product_is_finite: bool
    replacement_applied: bool
    post_replacement_product: float
    residual_after_replacement_is_finite: bool


@dataclass(frozen=True)
class SynthesisCallOrderEvidence:
    """Observed events from a real, short staged production solve."""

    operation_order: tuple[str, ...]
    observed_events: tuple[str, ...]
    executed_depth_indices: np.ndarray
    explicit_chain_length: int
    explicit_restart_depth_index: int
    observed_step_input_densities_cm3: np.ndarray
    expected_restart_density_cm3: np.ndarray
    explicit_restart_seed_matches: bool
    jacrev_evaluation_count: int
    newton_step_call_count: int
    final_vmap_evaluation_count: int
    jacrev_precedes_each_newton_step: bool
    final_vmap_follows_all_newton_steps: bool
    patched_identities_restored: bool


@dataclass(frozen=True)
class SynthesisSignDampingEvidence:
    """Controlled opposite-sign witness for the exact 0.69 branch."""

    equation_index: int
    previous_step_cm3: float
    undamped_step_cm3: float
    sign_change: bool
    damping_factor: float
    damped_step_cm3: float


@dataclass(frozen=True)
class IonResidualSignEvidence:
    """One catalog record's isolated contribution to the electron residual."""

    ion_kind: str
    molecule_code: float
    component_species_codes: np.ndarray
    ordinary_electron_count: int
    inverse_electron_count: int
    negative_ion_flag: int
    net_electron_coefficient: float
    molecular_term: float
    observed_electron_residual_delta: float


@dataclass(frozen=True)
class SharedNegativeIonEvidence:
    """Independent atmosphere/synthesis decoding of all shared negative ions."""

    molecule_codes: np.ndarray
    atmosphere_row_indices: np.ndarray
    synthesis_row_indices: np.ndarray
    component_offsets: np.ndarray
    atmosphere_component_species_codes: np.ndarray
    synthesis_component_species_codes: np.ndarray
    atmosphere_ordinary_electron_count: np.ndarray
    synthesis_ordinary_electron_count: np.ndarray
    atmosphere_inverse_electron_count: np.ndarray
    synthesis_inverse_electron_count: np.ndarray
    atmosphere_ordinary_electron_is_last: np.ndarray
    synthesis_ordinary_electron_is_last: np.ndarray
    synthesis_electron_component_multiplicity: np.ndarray
    synthesis_inverse_electron_power: np.ndarray
    synthesis_negative_ion_flag: np.ndarray
    synthesis_net_electron_coefficient: np.ndarray


@dataclass(frozen=True)
class _DecodedCatalog:
    """Minimal independently decoded catalog representation."""

    molecule_codes: np.ndarray
    component_offsets: np.ndarray
    component_species_codes: np.ndarray
    row_indices: np.ndarray
    ordinary_electron_count: np.ndarray
    inverse_electron_count: np.ndarray
    ordinary_electron_is_last: np.ndarray


@dataclass(frozen=True)
class SynthesisNewtonCheckpoint:
    """Frozen evidence from one exact CPU-float64 synthesis Newton update."""

    depth_index: int
    temperature_k: float
    gas_pressure_dyn_cm2: float
    device_type: str
    torch_dtype: str
    density_units: str
    equation_species_codes: np.ndarray
    equation_abundance: np.ndarray
    abundance_floor_mask: np.ndarray
    total_particle_density_cm3: float
    density_before_cm3: np.ndarray
    residual_cm3: np.ndarray
    jacrev_jacobian: np.ndarray
    finite_difference_jacobian: np.ndarray
    finite_difference_absolute_error: np.ndarray
    finite_difference_scale_relative_error: np.ndarray
    max_finite_difference_absolute_error: float
    max_finite_difference_scale_relative_error: float
    column_scale_cm3: np.ndarray
    scaled_jacobian: np.ndarray
    fractional_step: np.ndarray
    physical_step_cm3: np.ndarray
    unscaled_linear_residual_cm3: np.ndarray
    scaled_linear_residual_cm3: np.ndarray
    unscaled_linear_residual_relative_norm: float
    scaled_linear_residual_relative_norm: float
    convergence_tolerance: float
    undamped_relative_update: np.ndarray
    needs_more_iterations: bool
    maximum_fractional_update: float
    previous_step_cm3: np.ndarray
    sign_change_mask: np.ndarray
    damped_step_cm3: np.ndarray
    candidate_density_cm3: np.ndarray
    positivity_threshold_cm3: np.ndarray
    positivity_floor_mask: np.ndarray
    density_after_cm3: np.ndarray
    physical_log_products_are_finite: bool
    physical_pre_replacement_products_are_finite: bool
    physical_nonfinite_replacement_count: int
    sign_damping: SynthesisSignDampingEvidence
    shared_negative_ions: SharedNegativeIonEvidence
    negative_ion_residual: IonResidualSignEvidence
    positive_ion_residual: IonResidualSignEvidence
    nonfinite_product: SynthesisNonfiniteProductEvidence
    call_order: SynthesisCallOrderEvidence


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load independent arrays without retaining an archive handle."""

    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def load_checkpoint_inputs() -> dict[str, np.ndarray]:
    """Load and validate the ordered Chapter 4 input-only depth track."""

    inputs = _load_npz(CHAPTER04_INPUTS)
    required = {
        "temperature",
        "gas_pressure",
        "electron_density_seed",
        "elemental_abundances",
    }
    missing = sorted(required - set(inputs))
    if missing:
        raise ValueError(f"Chapter 4 synthesis inputs are missing {missing}")

    for name in ("temperature", "gas_pressure", "electron_density_seed"):
        values = np.asarray(inputs[name], dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name} must be a nonempty one-dimensional array")
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{name} must be finite and strictly positive")
        inputs[name] = values
    depth_count = inputs["temperature"].size
    if any(
        inputs[name].shape != (depth_count,)
        for name in ("gas_pressure", "electron_density_seed")
    ):
        raise ValueError("thermodynamic inputs must share one depth axis")
    if np.any(np.diff(inputs["gas_pressure"]) <= 0.0):
        raise ValueError("gas_pressure must preserve outer-to-inner depth order")

    abundance = np.asarray(inputs["elemental_abundances"], dtype=np.float64)
    if abundance.shape != (99,):
        raise ValueError("elemental_abundances must have shape (99,)")
    if np.any(~np.isfinite(abundance)) or np.any(abundance < 0.0):
        raise ValueError("elemental_abundances must be finite and nonnegative")
    inputs["elemental_abundances"] = abundance
    return inputs


def _load_eos_tables() -> equation_of_state.EOSTables:
    """Assemble the role-separated local EOS tables on CPU in float64."""

    table_arrays = _load_npz(SYNTHESIS_TABLES)
    ground_arrays = _load_npz(GROUND_PARTITION_FIXTURE)
    table_arrays["ground_partition_table"] = ground_arrays["ground_partition_table"]
    return equation_of_state.EOSTables.from_dict(
        table_arrays,
        device=CPU,
        dtype=DTYPE,
    )


def _formation_problem(
    inputs: Mapping[str, np.ndarray],
) -> tuple[
    molecular_equilibrium.MoleculeTable,
    molecular_equilibrium.MolecularStructure,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build the exact formation constants and equation abundances."""

    molecule_table = molecular_equilibrium.read_molecule_table(SYNTHESIS_CATALOG)
    metadata = molecular_equilibrium.molecular_equilibrium_metadata(SYNTHESIS_CATALOG)
    ion_formation_constants = (
        equation_of_state.molecular_ion_formation_constants_from_seed(
            inputs["temperature"],
            inputs["gas_pressure"],
            inputs["electron_density_seed"],
            tables=_load_eos_tables(),
            meta=metadata,
        )
    )
    molecule_count = molecule_table.molecule_count
    formation_constants = molecular_equilibrium.polynomial_formation_constants(
        inputs["temperature"],
        molecule_table,
    )
    component_counts = (
        molecule_table.component_start_indices[1 : molecule_count + 1]
        - molecule_table.component_start_indices[:molecule_count]
    )
    ion_mask = molecule_table.equilibrium_coefficients[0, :molecule_count] == 0.0
    single_mask = ion_mask & (component_counts == 1)
    ion_ratio_mask = ion_mask & (component_counts > 1)
    formation_constants[:, single_mask] = 1.0
    formation_constants[:, ion_ratio_mask] = ion_formation_constants[
        :, :molecule_count
    ][:, ion_ratio_mask]
    with np.errstate(divide="ignore"):
        natural_log_formation_constants = np.where(
            formation_constants[:, :molecule_count] > 0.0,
            np.log(
                np.maximum(
                    formation_constants[:, :molecule_count],
                    1.0e-300,
                )
            ),
            LOG_ZERO_SENTINEL,
        )

    structure = molecular_equilibrium.MolecularStructure.build(
        molecule_table,
        device=CPU,
        dtype=DTYPE,
    )
    equation_abundance = np.zeros(
        molecule_table.equation_count,
        dtype=np.float64,
    )
    for equation_index in range(1, molecule_table.equation_count):
        element_id = int(molecule_table.equation_species_codes[equation_index])
        if element_id < 100:
            equation_abundance[equation_index] = max(
                float(inputs["elemental_abundances"][element_id - 1]),
                NETWORK_ABUNDANCE_FLOOR,
            )
    return (
        molecule_table,
        structure,
        natural_log_formation_constants,
        equation_abundance,
        ion_formation_constants,
    )


def _chain_start_density(
    temperature_k: float,
    gas_pressure: float,
    equation_abundance: np.ndarray,
    electron_equation_index: int,
) -> tuple[float, np.ndarray]:
    """Construct the exact synthesis seed used at a chain boundary."""

    total_particle_density = gas_pressure / (
        temperature_k * molecular_equilibrium.BOLTZMANN_ERG_PER_K
    )
    initial_total_density = total_particle_density / 2.0
    if temperature_k < 4000.0:
        initial_total_density = total_particle_density
    base_density = initial_total_density / 10.0
    density = np.zeros(equation_abundance.size, dtype=np.float64)
    density[0] = initial_total_density
    density[1:] = base_density * equation_abundance[1:]
    if electron_equation_index >= 0:
        density[electron_equation_index] = base_density
    return total_particle_density, density


def _finite_difference_jacobian(
    residual_function,
    density: torch.Tensor,
    *,
    relative_step: float,
) -> np.ndarray:
    """Differentiate the exact residual independently with central differences."""

    point = density.detach().cpu().numpy()
    jacobian = np.empty((point.size, point.size), dtype=np.float64)
    for column_index in range(point.size):
        step = relative_step * max(abs(float(point[column_index])), 1.0)
        plus = point.copy()
        minus = point.copy()
        plus[column_index] += step
        minus[column_index] -= step
        with torch.no_grad():
            plus_residual = residual_function(
                torch.as_tensor(plus, dtype=DTYPE, device=CPU)
            )
            minus_residual = residual_function(
                torch.as_tensor(minus, dtype=DTYPE, device=CPU)
            )
        jacobian[:, column_index] = (
            plus_residual.detach().cpu().numpy() - minus_residual.detach().cpu().numpy()
        ) / (2.0 * step)
    return jacobian


def _molecular_product_terms(
    density: torch.Tensor,
    natural_log_formation_constants: torch.Tensor,
    structure: molecular_equilibrium.MolecularStructure,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose the exact three stages around the residual's finiteness gate."""

    log_density = molecular_equilibrium._safe_log(density)
    natural_log_product = natural_log_formation_constants + (
        structure.component_multiplicity * log_density
    ).sum(dim=1)
    if structure.electron_equation_index >= 0:
        natural_log_product = (
            natural_log_product
            - structure.inverse_electron_power
            * log_density[structure.electron_equation_index]
        )
    pre_replacement = torch.exp(natural_log_product) * structure.active_molecule_mask
    post_replacement = torch.where(
        torch.isfinite(pre_replacement),
        pre_replacement,
        torch.zeros_like(pre_replacement),
    )
    return natural_log_product, pre_replacement, post_replacement


def _nonfinite_product_evidence(
    density: torch.Tensor,
    natural_log_formation_constants: torch.Tensor,
    equation_abundance: torch.Tensor,
    total_particle_density: torch.Tensor,
    molecule_table: molecular_equilibrium.MoleculeTable,
    structure: molecular_equilibrium.MolecularStructure,
) -> SynthesisNonfiniteProductEvidence:
    """Drive one active product through the exact nonfinite replacement."""

    active_indices = torch.nonzero(
        structure.active_molecule_mask > 0.0,
        as_tuple=False,
    ).flatten()
    molecule_index = int(active_indices[0].item())
    controlled_log_constants = natural_log_formation_constants.clone()
    controlled_log_constants[molecule_index] = 1000.0
    log_product, pre_replacement, post_replacement = _molecular_product_terms(
        density,
        controlled_log_constants,
        structure,
    )
    residual = molecular_equilibrium._residual(
        density,
        controlled_log_constants,
        equation_abundance,
        total_particle_density,
        structure,
    )
    pre_is_finite = bool(torch.isfinite(pre_replacement[molecule_index]).item())
    return SynthesisNonfiniteProductEvidence(
        molecule_index=molecule_index,
        molecule_code=float(molecule_table.molecule_codes[molecule_index]),
        natural_log_product=float(log_product[molecule_index].item()),
        natural_log_product_is_finite=bool(
            torch.isfinite(log_product[molecule_index]).item()
        ),
        pre_replacement_product=float(pre_replacement[molecule_index].item()),
        pre_replacement_product_is_finite=pre_is_finite,
        replacement_applied=not pre_is_finite,
        post_replacement_product=float(post_replacement[molecule_index].item()),
        residual_after_replacement_is_finite=bool(
            torch.isfinite(residual).all().item()
        ),
    )


def _observed_call_order_evidence(
    inputs: Mapping[str, np.ndarray],
    ion_formation_constants: np.ndarray,
    equation_abundance: np.ndarray,
    electron_equation_index: int,
) -> SynthesisCallOrderEvidence:
    """Trace a real two-depth, one-iteration solve and restore every patch."""

    events: list[str] = []
    step_inputs: list[np.ndarray] = []
    original_jacrev = torch.func.jacrev
    original_vmap = torch.func.vmap
    original_newton_step = molecular_equilibrium._newton_step

    def observed_jacrev(*args, **kwargs):
        events.append("jacrev transform")
        transformed = original_jacrev(*args, **kwargs)

        def evaluate(*evaluate_args, **evaluate_kwargs):
            events.append("jacrev evaluation")
            return transformed(*evaluate_args, **evaluate_kwargs)

        return evaluate

    def observed_vmap(*args, **kwargs):
        events.append("final vmap transform")
        transformed = original_vmap(*args, **kwargs)

        def evaluate(*evaluate_args, **evaluate_kwargs):
            events.append("final vmap evaluation")
            return transformed(*evaluate_args, **evaluate_kwargs)

        return evaluate

    def observed_newton_step(jacobian, residual, densities):
        events.append("Newton step")
        step_inputs.append(densities.detach().cpu().to(torch.float64).numpy().copy())
        return original_newton_step(jacobian, residual, densities)

    try:
        torch.func.jacrev = observed_jacrev
        torch.func.vmap = observed_vmap
        molecular_equilibrium._newton_step = observed_newton_step
        molecular_equilibrium.solve_molecular_equilibrium(
            inputs["temperature"][:2],
            inputs["gas_pressure"][:2],
            inputs["electron_density_seed"][:2],
            inputs["elemental_abundances"],
            ion_formation_constants[:2],
            molecules_path=SYNTHESIS_CATALOG,
            device=CPU,
            dtype=DTYPE,
            max_iter=1,
            tol=1.0e-3,
            chain_length=1,
        )
    finally:
        torch.func.jacrev = original_jacrev
        torch.func.vmap = original_vmap
        molecular_equilibrium._newton_step = original_newton_step

    identities_restored = (
        torch.func.jacrev is original_jacrev
        and torch.func.vmap is original_vmap
        and molecular_equilibrium._newton_step is original_newton_step
    )
    if len(step_inputs) != 2:
        raise RuntimeError("short production trace did not execute two depths")
    _, expected_restart = _chain_start_density(
        float(inputs["temperature"][1]),
        float(inputs["gas_pressure"][1]),
        equation_abundance,
        electron_equation_index,
    )
    expected_events = (
        "jacrev transform",
        "jacrev evaluation",
        "Newton step",
        "jacrev evaluation",
        "Newton step",
        "final vmap transform",
        "final vmap evaluation",
    )
    observed_events = tuple(events)
    if observed_events != expected_events:
        raise RuntimeError(f"short production call order changed: {observed_events}")
    return SynthesisCallOrderEvidence(
        operation_order=(
            "production residual",
            "torch.func.jacrev Jacobian",
            "density-column-scaled solve",
            "undamped relative-update convergence test",
            "0.69 sign damping",
            "candidate density",
            "current_density/100 positivity floor",
        ),
        observed_events=observed_events,
        executed_depth_indices=_frozen([0, 1], dtype=np.int64),
        explicit_chain_length=1,
        explicit_restart_depth_index=1,
        observed_step_input_densities_cm3=_frozen(np.stack(step_inputs)),
        expected_restart_density_cm3=_frozen(expected_restart),
        explicit_restart_seed_matches=bool(
            np.array_equal(step_inputs[1], expected_restart)
        ),
        jacrev_evaluation_count=events.count("jacrev evaluation"),
        newton_step_call_count=events.count("Newton step"),
        final_vmap_evaluation_count=events.count("final vmap evaluation"),
        jacrev_precedes_each_newton_step=(
            events.index("jacrev evaluation") < events.index("Newton step")
            and events.index("jacrev evaluation", 2) < events.index("Newton step", 3)
        ),
        final_vmap_follows_all_newton_steps=(
            events.index("final vmap evaluation")
            > max(index for index, event in enumerate(events) if event == "Newton step")
        ),
        patched_identities_restored=identities_restored,
    )


def _decode_negative_ions(path: Path) -> _DecodedCatalog:
    """Decode negative ions directly from one local fixed-buffer catalog."""

    catalog = _load_npz(path)
    molecule_count = int(catalog["molecule_count"])
    equation_count = int(catalog["equation_count"])
    molecule_codes = np.asarray(
        catalog["molecule_codes"][:molecule_count],
        dtype=np.float64,
    )
    starts = np.asarray(catalog["component_start_indices"], dtype=np.int64)
    equation_indices = np.asarray(
        catalog["component_equation_indices"],
        dtype=np.int64,
    )
    equation_species_codes = np.asarray(
        catalog["equation_species_codes"][:equation_count],
        dtype=np.int64,
    )

    rows: list[int] = []
    flattened: list[int] = []
    offsets = [0]
    ordinary_counts: list[int] = []
    inverse_counts: list[int] = []
    electron_last: list[bool] = []
    for molecule_index in range(molecule_count):
        component_indices = equation_indices[
            starts[molecule_index] : starts[molecule_index + 1]
        ]
        if np.any(component_indices < 0) or np.any(component_indices > equation_count):
            raise ValueError("catalog component index is outside its encoding")
        semantics = np.asarray(
            [
                101
                if equation_index == equation_count
                else equation_species_codes[equation_index]
                for equation_index in component_indices
            ],
            dtype=np.int64,
        )
        ordinary_count = int(np.count_nonzero(semantics == 100))
        inverse_count = int(np.count_nonzero(semantics == 101))
        is_negative_ion = (
            semantics.size > 1
            and ordinary_count == 1
            and inverse_count == 0
            and int(semantics[-1]) == 100
        )
        if not is_negative_ion:
            continue
        rows.append(molecule_index)
        flattened.extend(semantics.tolist())
        offsets.append(len(flattened))
        ordinary_counts.append(ordinary_count)
        inverse_counts.append(inverse_count)
        electron_last.append(bool(semantics[-1] == 100))

    row_indices = np.asarray(rows, dtype=np.int64)
    return _DecodedCatalog(
        molecule_codes=_frozen(molecule_codes[row_indices]),
        component_offsets=_frozen(offsets, dtype=np.int64),
        component_species_codes=_frozen(flattened, dtype=np.int64),
        row_indices=_frozen(row_indices, dtype=np.int64),
        ordinary_electron_count=_frozen(ordinary_counts, dtype=np.int64),
        inverse_electron_count=_frozen(inverse_counts, dtype=np.int64),
        ordinary_electron_is_last=_frozen(electron_last, dtype=np.bool_),
    )


def _shared_negative_ion_evidence(
    molecule_table: molecular_equilibrium.MoleculeTable,
    structure: molecular_equilibrium.MolecularStructure,
) -> SharedNegativeIonEvidence:
    """Align independently decoded negative ions and verify residual coefficients."""

    atmosphere = _decode_negative_ions(ATMOSPHERE_CATALOG)
    synthesis = _decode_negative_ions(SYNTHESIS_CATALOG)
    atmosphere_keys = np.rint(atmosphere.molecule_codes * 100.0).astype(np.int64)
    synthesis_keys = np.rint(synthesis.molecule_codes * 100.0).astype(np.int64)
    synthesis_by_key = {int(key): index for index, key in enumerate(synthesis_keys)}
    synthesis_order = np.asarray(
        [synthesis_by_key[int(key)] for key in atmosphere_keys],
        dtype=np.int64,
    )
    aligned_synthesis_codes = synthesis.molecule_codes[synthesis_order]
    if not np.array_equal(atmosphere.molecule_codes, aligned_synthesis_codes):
        raise RuntimeError("negative-ion codes differ across local catalogs")

    synthesis_components: list[int] = []
    atmosphere_components: list[int] = []
    offsets = [0]
    for atmosphere_index, synthesis_index in enumerate(synthesis_order):
        atmosphere_slice = slice(
            atmosphere.component_offsets[atmosphere_index],
            atmosphere.component_offsets[atmosphere_index + 1],
        )
        synthesis_slice = slice(
            synthesis.component_offsets[synthesis_index],
            synthesis.component_offsets[synthesis_index + 1],
        )
        atmosphere_record = atmosphere.component_species_codes[atmosphere_slice]
        synthesis_record = synthesis.component_species_codes[synthesis_slice]
        if not np.array_equal(atmosphere_record, synthesis_record):
            raise RuntimeError("negative-ion component order differs")
        atmosphere_components.extend(atmosphere_record.tolist())
        synthesis_components.extend(synthesis_record.tolist())
        offsets.append(len(atmosphere_components))

    synthesis_rows = synthesis.row_indices[synthesis_order]
    electron_index = structure.electron_equation_index
    net_coefficient = (
        structure.component_multiplicity[synthesis_rows, electron_index]
        + structure.inverse_electron_power[synthesis_rows]
        - 2.0 * structure.negative_ion_flag[synthesis_rows]
    )
    electron_component_multiplicity = structure.component_multiplicity[
        synthesis_rows,
        electron_index,
    ]
    inverse_electron_power = structure.inverse_electron_power[synthesis_rows]
    negative_ion_flag = structure.negative_ion_flag[synthesis_rows]
    table_codes = molecule_table.molecule_codes[synthesis_rows]
    if not np.array_equal(table_codes, atmosphere.molecule_codes):
        raise RuntimeError("production synthesis table lost negative-ion rows")
    return SharedNegativeIonEvidence(
        molecule_codes=_frozen(atmosphere.molecule_codes),
        atmosphere_row_indices=_frozen(atmosphere.row_indices),
        synthesis_row_indices=_frozen(synthesis_rows),
        component_offsets=_frozen(offsets, dtype=np.int64),
        atmosphere_component_species_codes=_frozen(
            atmosphere_components,
            dtype=np.int64,
        ),
        synthesis_component_species_codes=_frozen(
            synthesis_components,
            dtype=np.int64,
        ),
        atmosphere_ordinary_electron_count=_frozen(atmosphere.ordinary_electron_count),
        synthesis_ordinary_electron_count=_frozen(
            synthesis.ordinary_electron_count[synthesis_order]
        ),
        atmosphere_inverse_electron_count=_frozen(atmosphere.inverse_electron_count),
        synthesis_inverse_electron_count=_frozen(
            synthesis.inverse_electron_count[synthesis_order]
        ),
        atmosphere_ordinary_electron_is_last=_frozen(
            atmosphere.ordinary_electron_is_last
        ),
        synthesis_ordinary_electron_is_last=_frozen(
            synthesis.ordinary_electron_is_last[synthesis_order]
        ),
        synthesis_electron_component_multiplicity=_frozen(
            electron_component_multiplicity.detach().cpu().numpy()
        ),
        synthesis_inverse_electron_power=_frozen(
            inverse_electron_power.detach().cpu().numpy()
        ),
        synthesis_negative_ion_flag=_frozen(negative_ion_flag.detach().cpu().numpy()),
        synthesis_net_electron_coefficient=_frozen(
            net_coefficient.detach().cpu().numpy()
        ),
    )


def _component_semantics(
    molecule_table: molecular_equilibrium.MoleculeTable,
    molecule_index: int,
) -> np.ndarray:
    """Decode one staged synthesis record without using a semantic helper."""

    start = int(molecule_table.component_start_indices[molecule_index])
    stop = int(molecule_table.component_start_indices[molecule_index + 1])
    indices = molecule_table.component_equation_indices[start:stop]
    return np.asarray(
        [
            101
            if int(index) == molecule_table.equation_count
            else int(molecule_table.equation_species_codes[int(index)])
            for index in indices
        ],
        dtype=np.int64,
    )


def _isolated_ion_residual_evidence(
    molecule_table: molecular_equilibrium.MoleculeTable,
    structure: molecular_equilibrium.MolecularStructure,
    molecule_index: int,
    *,
    ion_kind: str,
) -> IonResidualSignEvidence:
    """Isolate one real catalog record and measure its electron-row sign."""

    selector = torch.zeros_like(structure.active_molecule_mask)
    selector[molecule_index] = 1.0
    selected_structure = molecular_equilibrium.MolecularStructure(
        equation_count=structure.equation_count,
        electron_equation_index=structure.electron_equation_index,
        component_multiplicity=(
            structure.component_multiplicity * selector.unsqueeze(1)
        ),
        inverse_electron_power=(structure.inverse_electron_power * selector),
        negative_ion_flag=structure.negative_ion_flag,
        active_molecule_mask=selector,
        full_component_multiplicity=structure.full_component_multiplicity,
        full_inverse_electron_power=structure.full_inverse_electron_power,
    )
    baseline_structure = molecular_equilibrium.MolecularStructure(
        equation_count=structure.equation_count,
        electron_equation_index=structure.electron_equation_index,
        component_multiplicity=torch.zeros_like(structure.component_multiplicity),
        inverse_electron_power=torch.zeros_like(structure.inverse_electron_power),
        negative_ion_flag=torch.zeros_like(structure.negative_ion_flag),
        active_molecule_mask=torch.zeros_like(structure.active_molecule_mask),
        full_component_multiplicity=structure.full_component_multiplicity,
        full_inverse_electron_power=structure.full_inverse_electron_power,
    )
    density = torch.ones(structure.equation_count, dtype=DTYPE, device=CPU)
    log_constants = torch.zeros(
        structure.active_molecule_mask.size(0),
        dtype=DTYPE,
        device=CPU,
    )
    equation_abundance = torch.zeros_like(density)
    total_particle_density = torch.as_tensor(1.0, dtype=DTYPE, device=CPU)
    selected_residual = molecular_equilibrium._residual(
        density,
        log_constants,
        equation_abundance,
        total_particle_density,
        selected_structure,
    )
    baseline_residual = molecular_equilibrium._residual(
        density,
        log_constants,
        equation_abundance,
        total_particle_density,
        baseline_structure,
    )
    electron_index = structure.electron_equation_index
    component_count = float(
        structure.component_multiplicity[molecule_index, electron_index]
        .detach()
        .cpu()
        .item()
    )
    inverse_count = float(
        structure.inverse_electron_power[molecule_index].detach().cpu().item()
    )
    negative_flag = float(
        structure.negative_ion_flag[molecule_index].detach().cpu().item()
    )
    coefficient = component_count + inverse_count - 2.0 * negative_flag
    semantics = _component_semantics(molecule_table, molecule_index)
    return IonResidualSignEvidence(
        ion_kind=ion_kind,
        molecule_code=float(molecule_table.molecule_codes[molecule_index]),
        component_species_codes=_frozen(semantics, dtype=np.int64),
        ordinary_electron_count=int(np.count_nonzero(semantics == 100)),
        inverse_electron_count=int(np.count_nonzero(semantics == 101)),
        negative_ion_flag=int(negative_flag),
        net_electron_coefficient=coefficient,
        molecular_term=1.0,
        observed_electron_residual_delta=float(
            (selected_residual[electron_index] - baseline_residual[electron_index])
            .detach()
            .cpu()
            .item()
        ),
    )


def _ion_residual_sign_evidence(
    molecule_table: molecular_equilibrium.MoleculeTable,
    structure: molecular_equilibrium.MolecularStructure,
) -> tuple[IonResidualSignEvidence, IonResidualSignEvidence]:
    """Return one controlled negative-ion and positive-ion residual witness."""

    negative_indices = torch.nonzero(
        structure.negative_ion_flag > 0.0,
        as_tuple=False,
    ).flatten()
    positive_indices = torch.nonzero(
        (structure.inverse_electron_power > 0.0) & (structure.negative_ion_flag == 0.0),
        as_tuple=False,
    ).flatten()
    negative = _isolated_ion_residual_evidence(
        molecule_table,
        structure,
        int(negative_indices[0].item()),
        ion_kind="negative ion",
    )
    positive = _isolated_ion_residual_evidence(
        molecule_table,
        structure,
        int(positive_indices[0].item()),
        ion_kind="positive ion",
    )
    return negative, positive


def _sign_damping_evidence(
    physical_step: torch.Tensor,
) -> SynthesisSignDampingEvidence:
    """Trigger exactly one opposite-sign 0.69 damping branch."""

    equation_index = int(torch.argmax(physical_step.abs()).item())
    undamped = float(physical_step[equation_index].item())
    if undamped == 0.0:
        raise RuntimeError("controlled damping witness needs a nonzero step")
    previous = -undamped
    sign_change = (previous > 0.0 and undamped < 0.0) or (
        previous < 0.0 and undamped > 0.0
    )
    damped = undamped * 0.69 if sign_change else undamped
    return SynthesisSignDampingEvidence(
        equation_index=equation_index,
        previous_step_cm3=previous,
        undamped_step_cm3=undamped,
        sign_change=sign_change,
        damping_factor=0.69,
        damped_step_cm3=damped,
    )


def synthesis_newton_checkpoint(
    *,
    depth_index: int = 0,
    explicit_chain_restart: bool = False,
    finite_difference_relative_step: float = 1.0e-4,
) -> SynthesisNewtonCheckpoint:
    """Execute one exact synthesis Newton update at an ordered physical depth.

    Depth zero is the natural start of the production continuation chain.  A
    later depth is accepted only when the caller explicitly declares a chain
    restart; this prevents a one-depth teaching calculation from masquerading
    as independent production depth parallelism.
    """

    inputs = load_checkpoint_inputs()
    depth_count = int(inputs["temperature"].size)
    if not 0 <= depth_index < depth_count:
        raise IndexError(f"depth_index must be in [0, {depth_count})")
    if depth_index > 0 and not explicit_chain_restart:
        raise ValueError(
            "a later one-depth checkpoint requires explicit_chain_restart=True"
        )
    if (
        not np.isfinite(finite_difference_relative_step)
        or finite_difference_relative_step <= 0.0
    ):
        raise ValueError("finite_difference_relative_step must be positive")

    (
        molecule_table,
        structure,
        natural_log_formation_constants,
        equation_abundance,
        ion_formation_constants,
    ) = _formation_problem(inputs)
    temperature_k = float(inputs["temperature"][depth_index])
    gas_pressure = float(inputs["gas_pressure"][depth_index])
    total_particle_density, initial_density = _chain_start_density(
        temperature_k,
        gas_pressure,
        equation_abundance,
        structure.electron_equation_index,
    )

    density = torch.as_tensor(initial_density, dtype=DTYPE, device=CPU)
    log_constants = torch.as_tensor(
        natural_log_formation_constants[depth_index],
        dtype=DTYPE,
        device=CPU,
    )
    abundance = torch.as_tensor(
        equation_abundance,
        dtype=DTYPE,
        device=CPU,
    )
    total_density = torch.as_tensor(
        total_particle_density,
        dtype=DTYPE,
        device=CPU,
    )

    def residual_function(density_row: torch.Tensor) -> torch.Tensor:
        return molecular_equilibrium._residual(
            density_row,
            log_constants,
            abundance,
            total_density,
            structure,
        )

    residual = residual_function(density)
    from torch.func import jacrev

    jacobian = jacrev(residual_function, argnums=0)(density)
    finite_difference = _finite_difference_jacobian(
        residual_function,
        density,
        relative_step=finite_difference_relative_step,
    )

    column_scale = density.abs().clamp_min(torch.finfo(DTYPE).tiny)
    scaled_jacobian = jacobian * column_scale.unsqueeze(0)
    fractional_step = torch.linalg.solve(scaled_jacobian, residual)
    physical_step = molecular_equilibrium._newton_step(
        jacobian,
        residual,
        density,
    )
    if not torch.equal(physical_step, column_scale * fractional_step):
        raise RuntimeError("exact _newton_step no longer matches its scaled system")

    unscaled_linear_residual = jacobian @ physical_step - residual
    scaled_linear_residual = scaled_jacobian @ fractional_step - residual
    residual_norm = torch.linalg.vector_norm(residual).clamp_min(
        torch.finfo(DTYPE).tiny
    )

    convergence_tolerance = 1.0e-3
    undamped_relative_update = physical_step.abs() / density.abs().clamp_min(
        torch.finfo(DTYPE).tiny
    )
    needs_more_iterations = bool(
        (undamped_relative_update > convergence_tolerance).any().item()
    )
    previous_step = torch.zeros_like(density)
    sign_change = ((previous_step > 0.0) & (physical_step < 0.0)) | (
        (previous_step < 0.0) & (physical_step > 0.0)
    )
    damped_step = torch.where(
        sign_change,
        physical_step * 0.69,
        physical_step,
    )
    candidate = density - damped_step
    positivity_threshold = density / POSITIVITY_FLOOR_DIVISOR
    positivity_floor_mask = candidate < positivity_threshold
    density_after = torch.where(
        positivity_floor_mask,
        positivity_threshold,
        candidate,
    )

    jacobian_numpy = jacobian.detach().cpu().numpy()
    residual_numpy = residual.detach().cpu().numpy()
    column_scale_numpy = column_scale.detach().cpu().numpy()
    absolute_error = np.abs(finite_difference - jacobian_numpy)
    response_scale = np.maximum(
        np.maximum(
            np.abs(residual_numpy)[:, None],
            np.abs(jacobian_numpy) * column_scale_numpy[None, :],
        ),
        1.0,
    )
    scale_relative_error = absolute_error * column_scale_numpy[None, :] / response_scale

    log_product, pre_replacement, _ = _molecular_product_terms(
        density,
        log_constants,
        structure,
    )
    physical_replacement_mask = ~torch.isfinite(pre_replacement)
    call_order = _observed_call_order_evidence(
        inputs,
        ion_formation_constants,
        equation_abundance,
        structure.electron_equation_index,
    )
    shared_negative_ions = _shared_negative_ion_evidence(
        molecule_table,
        structure,
    )
    negative_ion_residual, positive_ion_residual = _ion_residual_sign_evidence(
        molecule_table, structure
    )
    sign_damping = _sign_damping_evidence(physical_step)
    overflow_evidence = _nonfinite_product_evidence(
        density,
        log_constants,
        abundance,
        total_density,
        molecule_table,
        structure,
    )

    return SynthesisNewtonCheckpoint(
        depth_index=depth_index,
        temperature_k=temperature_k,
        gas_pressure_dyn_cm2=gas_pressure,
        device_type=density.device.type,
        torch_dtype=str(density.dtype),
        density_units="cm^-3",
        equation_species_codes=_frozen(
            molecule_table.equation_species_codes[: molecule_table.equation_count],
            dtype=np.int64,
        ),
        equation_abundance=_frozen(equation_abundance),
        abundance_floor_mask=_frozen(
            (equation_abundance == NETWORK_ABUNDANCE_FLOOR)
            & (np.arange(equation_abundance.size) > 0),
            dtype=np.bool_,
        ),
        total_particle_density_cm3=total_particle_density,
        density_before_cm3=_frozen(initial_density),
        residual_cm3=_frozen(residual_numpy),
        jacrev_jacobian=_frozen(jacobian_numpy),
        finite_difference_jacobian=_frozen(finite_difference),
        finite_difference_absolute_error=_frozen(absolute_error),
        finite_difference_scale_relative_error=_frozen(scale_relative_error),
        max_finite_difference_absolute_error=float(np.max(absolute_error)),
        max_finite_difference_scale_relative_error=float(np.max(scale_relative_error)),
        column_scale_cm3=_frozen(column_scale_numpy),
        scaled_jacobian=_frozen(scaled_jacobian.detach().cpu().numpy()),
        fractional_step=_frozen(fractional_step.detach().cpu().numpy()),
        physical_step_cm3=_frozen(physical_step.detach().cpu().numpy()),
        unscaled_linear_residual_cm3=_frozen(
            unscaled_linear_residual.detach().cpu().numpy()
        ),
        scaled_linear_residual_cm3=_frozen(
            scaled_linear_residual.detach().cpu().numpy()
        ),
        unscaled_linear_residual_relative_norm=float(
            (torch.linalg.vector_norm(unscaled_linear_residual) / residual_norm)
            .detach()
            .cpu()
            .item()
        ),
        scaled_linear_residual_relative_norm=float(
            (torch.linalg.vector_norm(scaled_linear_residual) / residual_norm)
            .detach()
            .cpu()
            .item()
        ),
        convergence_tolerance=convergence_tolerance,
        undamped_relative_update=_frozen(
            undamped_relative_update.detach().cpu().numpy()
        ),
        needs_more_iterations=needs_more_iterations,
        maximum_fractional_update=float(undamped_relative_update.max().item()),
        previous_step_cm3=_frozen(previous_step.detach().cpu().numpy()),
        sign_change_mask=_frozen(
            sign_change.detach().cpu().numpy(),
            dtype=np.bool_,
        ),
        damped_step_cm3=_frozen(damped_step.detach().cpu().numpy()),
        candidate_density_cm3=_frozen(candidate.detach().cpu().numpy()),
        positivity_threshold_cm3=_frozen(positivity_threshold.detach().cpu().numpy()),
        positivity_floor_mask=_frozen(
            positivity_floor_mask.detach().cpu().numpy(),
            dtype=np.bool_,
        ),
        density_after_cm3=_frozen(density_after.detach().cpu().numpy()),
        physical_log_products_are_finite=bool(torch.isfinite(log_product).all().item()),
        physical_pre_replacement_products_are_finite=bool(
            torch.isfinite(pre_replacement).all().item()
        ),
        physical_nonfinite_replacement_count=int(
            physical_replacement_mask.sum().item()
        ),
        sign_damping=sign_damping,
        shared_negative_ions=shared_negative_ions,
        negative_ion_residual=negative_ion_residual,
        positive_ion_residual=positive_ion_residual,
        nonfinite_product=overflow_evidence,
        call_order=call_order,
    )
