"""Compute the complete Chapter 4 molecular state from repository data.

This module deliberately has only one direction of dependency: input fixtures
and progressive source modules enter, detached NumPy arrays leave.  Expected
results and publication machinery do not participate in the calculation.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import inspect
from typing import Any, Callable, Mapping

import numpy as np

from book.chapter04_runtime import (
    ATMOSPHERE_MOLECULE_CATALOG,
    SYNTHESIS_ATOMIC_MASSES,
    SYNTHESIS_CONTINUUM_EDGE_GRID,
    SYNTHESIS_MOLECULE_CATALOG,
    _initialize_unsolved_atmosphere_molecular_state,
    atmosphere_continuation_checkpoint,
    build_public_molecular_lane_checkpoint,
    compute_atmosphere_molecular_state,
    compute_synthesis_molecular_state,
    configure_local_data_paths,
    load_molecular_inputs,
    load_npz_arrays,
    load_synthesis_eos_tables,
)


def _array(value: Any) -> np.ndarray:
    """Return one detached, owned, object-free array."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value).copy()
    if result.dtype.hasobject:
        raise TypeError("Chapter 4 parity state cannot contain object arrays")
    result.setflags(write=False)
    return result


def _freeze_mapping(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Detach every value and impose a stable lexical key order."""

    return {name: _array(values[name]) for name in sorted(values)}


def _flatten_population_state(
    state: Any,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Flatten a molecular population state, keeping its EOS fields distinct."""

    if not is_dataclass(state):
        raise TypeError("population state must be a dataclass")
    arrays: dict[str, np.ndarray] = {}
    for field in fields(state):
        value = getattr(state, field.name)
        if field.name == "eos":
            if not is_dataclass(value):
                raise TypeError("population state's EOS field must be a dataclass")
            for eos_field in fields(value):
                arrays[f"{prefix}eos__{eos_field.name}"] = _array(
                    getattr(value, eos_field.name)
                )
        elif value is None:
            raise RuntimeError(f"molecule-enabled state omitted {field.name}")
        else:
            arrays[f"{prefix}{field.name}"] = _array(value)
    return arrays


def _molecular_residual_diagnostics(
    effective: Mapping[str, Any],
    outputs: tuple[Any, ...],
    diagnostics: Mapping[str, Any],
    molecular_module: Any,
) -> dict[str, np.ndarray]:
    """Independently evaluate the final residual and its physical row scale."""

    torch = molecular_module.torch
    structure = diagnostics["structure"]
    equation_count = int(diagnostics["equation_count"])
    molecule_count = int(diagnostics["molecule_count"])
    equation_codes = np.asarray(
        diagnostics["equation_species_codes"],
        dtype=np.int64,
    )
    abundance = np.asarray(
        effective["elemental_abundances"],
        dtype=np.float64,
    )
    temperature = np.asarray(effective["temperature"], dtype=np.float64)
    gas_pressure = np.asarray(effective["gas_pressure"], dtype=np.float64)
    equation_abundance = np.zeros(equation_count, dtype=np.float64)
    for index in range(1, equation_count):
        species_code = int(equation_codes[index])
        if species_code < 100:
            equation_abundance[index] = max(
                float(abundance[species_code - 1]),
                1.0e-20,
            )

    densities = outputs[2][:, :equation_count]
    natural_log_formation = diagnostics["natural_log_formation_constants"][
        :, :molecule_count
    ]
    equation_abundance_tensor = torch.as_tensor(
        equation_abundance,
        dtype=densities.dtype,
        device=densities.device,
    )
    total_particle_density = torch.as_tensor(
        gas_pressure / (temperature * molecular_module.BOLTZMANN_ERG_PER_K),
        dtype=densities.dtype,
        device=densities.device,
    )

    residual_rows = []
    normalized_rows = []
    log_term_rows = []
    nonfinite_rows = []
    with torch.no_grad():
        for depth_index in range(temperature.size):
            density = densities[depth_index]
            log_constant = natural_log_formation[depth_index]
            particle_density = total_particle_density[depth_index]
            residual = molecular_module._residual(
                density,
                log_constant,
                equation_abundance_tensor,
                particle_density,
                structure,
            )

            log_density = molecular_module._safe_log(density)
            log_term = log_constant + (
                structure.component_multiplicity * log_density
            ).sum(dim=1)
            electron_index = int(structure.electron_equation_index)
            if electron_index >= 0:
                log_term = (
                    log_term
                    - structure.inverse_electron_power * log_density[electron_index]
                )
            raw_term = torch.exp(log_term) * structure.active_molecule_mask
            nonfinite = ~torch.isfinite(raw_term)
            finite_term = torch.where(
                nonfinite,
                torch.zeros_like(raw_term),
                raw_term,
            )

            scale = torch.empty_like(density)
            scale[0] = (
                particle_density.abs()
                + density[1:].abs().sum()
                + finite_term.abs().sum()
            )
            for equation_index in range(1, equation_count):
                row_scale = (
                    density[equation_index].abs()
                    + (equation_abundance_tensor[equation_index] * density[0]).abs()
                    + (
                        structure.component_multiplicity[:, equation_index]
                        * finite_term
                    )
                    .abs()
                    .sum()
                )
                if equation_index == electron_index:
                    row_scale = (
                        density[equation_index].abs()
                        + (
                            structure.component_multiplicity[:, equation_index]
                            * finite_term
                        )
                        .abs()
                        .sum()
                        + (structure.inverse_electron_power * finite_term).abs().sum()
                        + (2.0 * structure.negative_ion_flag * finite_term).abs().sum()
                    )
                scale[equation_index] = row_scale
            scale = scale.clamp_min(1.0)

            residual_rows.append(residual)
            normalized_rows.append(residual / scale)
            log_term_rows.append(log_term)
            nonfinite_rows.append(nonfinite)

    residual_array = torch.stack(residual_rows)
    normalized_array = torch.stack(normalized_rows)
    log_term_array = torch.stack(log_term_rows)
    nonfinite_array = torch.stack(nonfinite_rows)
    return _freeze_mapping(
        {
            "equation_abundance": equation_abundance,
            "total_particle_density": total_particle_density,
            "residual": residual_array,
            "normalized_residual": normalized_array,
            "pre_replacement_log_term": log_term_array,
            "pre_replacement_term_nonfinite_mask": nonfinite_array,
            "pre_replacement_term_nonfinite_count": nonfinite_array.sum(dim=1),
            "pre_replacement_log_term_min": log_term_array.min(dim=1).values,
            "pre_replacement_log_term_max": log_term_array.max(dim=1).values,
        }
    )


def _molecular_call_arrays(
    *,
    effective: Mapping[str, Any],
    outputs: tuple[Any, Any, Any, Any],
    diagnostics: Mapping[str, Any],
    caller_name: str,
    caller_module: str,
    caller_file: str,
    molecular_module: Any,
) -> dict[str, np.ndarray]:
    """Serialize one locally observed call without importing oracle machinery."""

    iterations = _array(diagnostics["iterations_completed"])
    call: dict[str, Any] = {
        "caller_name": caller_name,
        "caller_module": caller_module,
        "caller_file": caller_file,
        "device": str(effective["device"]),
        "dtype": str(effective["dtype"]),
        "max_iter": effective["max_iter"],
        "tol": effective["tol"],
        "chain_length": (
            -1 if effective["chain_length"] is None else effective["chain_length"]
        ),
        "molecules_path": str(effective["molecules_path"].resolve()),
        "input__temperature": effective["temperature"],
        "input__gas_pressure": effective["gas_pressure"],
        "input__electron_density": effective["electron_density"],
        "input__elemental_abundances": effective["elemental_abundances"],
        "input__ion_formation_constants": effective["ion_formation_constants"],
        "output__total_nuclei_number_density": outputs[0],
        "output__molecular_populations": outputs[1],
        "output__equation_densities": outputs[2],
        "output__electron_density": outputs[3],
        "diag__molecule_count": diagnostics["molecule_count"],
        "diag__equation_count": diagnostics["equation_count"],
        "diag__equation_species_codes": diagnostics["equation_species_codes"],
        "diag__molecule_codes": diagnostics["molecule_codes"],
        "diag__iterations_completed": iterations,
        "diag__exhausted_mask": (iterations >= int(effective["max_iter"])),
        "diag__natural_log_formation_constants": diagnostics[
            "natural_log_formation_constants"
        ],
    }
    structure = diagnostics["structure"]
    for name in (
        "component_multiplicity",
        "inverse_electron_power",
        "negative_ion_flag",
        "active_molecule_mask",
        "full_component_multiplicity",
        "full_inverse_electron_power",
    ):
        call[f"diag__structure__{name}"] = getattr(structure, name)
    call["diag__structure__electron_equation_index"] = structure.electron_equation_index
    residual = _molecular_residual_diagnostics(
        effective,
        outputs,
        diagnostics,
        molecular_module,
    )
    for name, values in residual.items():
        call[f"diag__{name}"] = values
    return _freeze_mapping(call)


def _capture_molecular_calls(
    calculation: Callable[[], Any],
) -> tuple[Any, tuple[dict[str, np.ndarray], ...]]:
    """Observe exact local molecular calls without changing returned values."""

    from payne_zero_synthesis import molecular_equilibrium

    original = molecular_equilibrium.solve_molecular_equilibrium
    calls: list[dict[str, np.ndarray]] = []

    def observed(*args, **kwargs):
        caller = inspect.currentframe().f_back
        bound = inspect.signature(original).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        effective = dict(bound.arguments)
        caller_requested_diagnostics = bool(effective["return_diagnostics"])
        diagnostic_kwargs = dict(kwargs)
        diagnostic_kwargs["return_diagnostics"] = True
        result = original(*args, **diagnostic_kwargs)
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("diagnostic molecular solve returned a bad tuple")
        diagnostics = result[4]
        calls.append(
            _molecular_call_arrays(
                effective=effective,
                outputs=result[:4],
                diagnostics=diagnostics,
                caller_name=caller.f_code.co_name,
                caller_module=caller.f_globals.get("__name__", ""),
                caller_file=caller.f_code.co_filename,
                molecular_module=molecular_equilibrium,
            )
        )
        return result if caller_requested_diagnostics else result[:4]

    molecular_equilibrium.solve_molecular_equilibrium = observed
    try:
        state = calculation()
    finally:
        molecular_equilibrium.solve_molecular_equilibrium = original
    return state, tuple(calls)


def _component_semantics(
    catalog: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode component equation indices, preserving the electron sentinel."""

    molecule_count = int(np.asarray(catalog["molecule_count"]))
    equation_count = int(np.asarray(catalog["equation_count"]))
    starts = np.asarray(catalog["component_start_indices"], dtype=np.int64)
    indices = np.asarray(
        catalog["component_equation_indices"],
        dtype=np.int64,
    )
    species = np.asarray(
        catalog["equation_species_codes"],
        dtype=np.int64,
    )
    component_count = int(starts[molecule_count])
    active = indices[:component_count]
    sentinel = active == equation_count
    if np.any(active < 0) or np.any(active > equation_count):
        raise RuntimeError("component equation index is outside the active table")
    semantic = np.full(indices.shape, -1, dtype=np.int32)
    semantic[:component_count][sentinel] = 101
    semantic[:component_count][~sentinel] = species[active[~sentinel]]
    sentinel_mask = np.zeros(indices.shape, dtype=np.bool_)
    sentinel_mask[:component_count] = sentinel
    return _array(semantic), _array(sentinel_mask)


def _catalog_state() -> dict[str, np.ndarray]:
    """Return fixed buffers, active extents, and order-independent overlap."""

    from payne_zero_synthesis import molecular_equilibrium

    atmosphere = load_npz_arrays(ATMOSPHERE_MOLECULE_CATALOG)
    synthesis = load_npz_arrays(SYNTHESIS_MOLECULE_CATALOG)
    atmosphere_count = int(atmosphere["molecule_count"])
    synthesis_count = int(synthesis["molecule_count"])
    synthesis_component_count = int(
        synthesis["component_start_indices"][synthesis_count]
    )
    atmosphere_semantic, atmosphere_sentinel = _component_semantics(atmosphere)
    synthesis_semantic, synthesis_sentinel = _component_semantics(synthesis)

    atmosphere_codes = np.asarray(
        atmosphere["molecule_codes"][:atmosphere_count],
        dtype=np.float64,
    )
    synthesis_codes = np.asarray(
        synthesis["molecule_codes"][:synthesis_count],
        dtype=np.float64,
    )
    atmosphere_keys = np.rint(atmosphere_codes * 100.0).astype(np.int64)
    synthesis_keys = np.rint(synthesis_codes * 100.0).astype(np.int64)
    shared_keys = np.intersect1d(synthesis_keys, atmosphere_keys)
    atmosphere_index = {int(key): index for index, key in enumerate(atmosphere_keys)}
    synthesis_index = {int(key): index for index, key in enumerate(synthesis_keys)}
    atmosphere_shared = np.asarray(
        [atmosphere_index[int(key)] for key in shared_keys],
        dtype=np.int64,
    )
    synthesis_shared = np.asarray(
        [synthesis_index[int(key)] for key in shared_keys],
        dtype=np.int64,
    )
    synthesis_only_keys = np.setdiff1d(synthesis_keys, atmosphere_keys)
    synthesis_only = np.asarray(
        [synthesis_index[int(key)] for key in synthesis_only_keys],
        dtype=np.int64,
    )
    atmosphere_only_keys = np.setdiff1d(atmosphere_keys, synthesis_keys)
    atmosphere_only = np.asarray(
        [atmosphere_index[int(key)] for key in atmosphere_only_keys],
        dtype=np.int64,
    )

    atmosphere_starts = np.asarray(
        atmosphere["component_start_indices"],
        dtype=np.int64,
    )
    synthesis_starts = np.asarray(
        synthesis["component_start_indices"],
        dtype=np.int64,
    )
    offsets = [0]
    atmosphere_shared_components: list[int] = []
    synthesis_shared_components: list[int] = []
    component_mismatch: list[bool] = []
    for synthesis_row, atmosphere_row in zip(
        synthesis_shared,
        atmosphere_shared,
        strict=True,
    ):
        synthesis_values = synthesis_semantic[
            synthesis_starts[synthesis_row] : synthesis_starts[synthesis_row + 1]
        ]
        atmosphere_values = atmosphere_semantic[
            atmosphere_starts[atmosphere_row] : atmosphere_starts[atmosphere_row + 1]
        ]
        synthesis_shared_components.extend(synthesis_values.tolist())
        atmosphere_shared_components.extend(atmosphere_values.tolist())
        offsets.append(len(synthesis_shared_components))
        component_mismatch.append(
            not np.array_equal(synthesis_values, atmosphere_values)
        )
    coefficient_mismatch = np.any(
        np.asarray(synthesis["equilibrium_coefficients"])[:, synthesis_shared]
        != np.asarray(atmosphere["equilibrium_coefficients"])[:, atmosphere_shared],
        axis=0,
    )
    component_mismatch_array = np.asarray(
        component_mismatch,
        dtype=np.bool_,
    )
    semantic_mismatch = coefficient_mismatch | component_mismatch_array

    arrays: dict[str, Any] = {}
    for name, values in atmosphere.items():
        arrays[f"catalog__atmosphere__{name}"] = values
    for name, values in synthesis.items():
        arrays[f"catalog__synthesis__{name}"] = values
    arrays.update(
        {
            "catalog__atmosphere__component_semantic_species_codes": (
                atmosphere_semantic
            ),
            "catalog__atmosphere__component_inverse_electron_sentinel_mask": (
                atmosphere_sentinel
            ),
            "catalog__synthesis__component_count": synthesis_component_count,
            "catalog__synthesis__component_semantic_species_codes": (
                synthesis_semantic
            ),
            "catalog__synthesis__component_inverse_electron_sentinel_mask": (
                synthesis_sentinel
            ),
            "catalog__synthesis__active_molecule_mask": (
                np.arange(synthesis["molecule_codes"].size) < synthesis_count
            ),
            "catalog__synthesis__active_component_mask": (
                np.arange(synthesis["component_equation_indices"].size)
                < synthesis_component_count
            ),
            "catalog__synthesis__active_equation_mask": (
                np.arange(synthesis["equation_species_codes"].size)
                < int(synthesis["equation_count"])
            ),
            "catalog__synthesis__hard_coded_molecular_atomic_masses_amu": (
                molecular_equilibrium._ATOMIC_MASSES_FOR_MOLECULES
            ),
            "catalog__synthesis__hard_coded_molecular_atomic_masses_sha256": (
                hashlib.sha256(
                    np.asarray(
                        molecular_equilibrium._ATOMIC_MASSES_FOR_MOLECULES,
                        dtype=np.float64,
                    ).tobytes(order="C")
                ).hexdigest()
            ),
            "alignment__shared_rounded_code_keys": shared_keys,
            "alignment__shared_molecule_codes": (synthesis_codes[synthesis_shared]),
            "alignment__synthesis_shared_row_indices": synthesis_shared,
            "alignment__atmosphere_shared_row_indices": atmosphere_shared,
            "alignment__shared_row_indices_differ_mask": (
                synthesis_shared != atmosphere_shared
            ),
            "alignment__shared_component_offsets": np.asarray(
                offsets,
                dtype=np.int64,
            ),
            "alignment__synthesis_shared_component_semantics": np.asarray(
                synthesis_shared_components,
                dtype=np.int32,
            ),
            "alignment__atmosphere_shared_component_semantics": np.asarray(
                atmosphere_shared_components,
                dtype=np.int32,
            ),
            "alignment__coefficient_mismatch_mask": coefficient_mismatch,
            "alignment__component_mismatch_mask": component_mismatch_array,
            "alignment__semantic_mismatch_mask": semantic_mismatch,
            "alignment__shared_count": shared_keys.size,
            "alignment__atmosphere_only_rounded_code_keys": (atmosphere_only_keys),
            "alignment__atmosphere_only_row_indices": atmosphere_only,
            "alignment__atmosphere_only_count": atmosphere_only_keys.size,
            "alignment__synthesis_only_rounded_code_keys": synthesis_only_keys,
            "alignment__synthesis_only_row_indices": synthesis_only,
            "alignment__synthesis_only_molecule_codes": (
                synthesis_codes[synthesis_only]
            ),
            "alignment__synthesis_only_count": synthesis_only_keys.size,
            "alignment__semantic_mismatch_count": semantic_mismatch.sum(),
        }
    )
    return _freeze_mapping(arrays)


def _deterministic_supplied_mass_density(
    inputs: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Derive the supplied-density branch from inputs and local mass data."""

    atomic_masses = load_npz_arrays(SYNTHESIS_ATOMIC_MASSES)["atomic_mass_amu"][
        :99
    ].astype(np.float64, copy=False)
    abundance = np.asarray(inputs["elemental_abundances"], dtype=np.float64)
    mean_mass_amu = float(np.sum(abundance * atomic_masses) / np.sum(abundance))
    nuclei_density_scale = np.asarray(
        inputs["gas_pressure"],
        dtype=np.float64,
    ) / (np.asarray(inputs["temperature"], dtype=np.float64) * 1.38054e-16)
    return _array(nuclei_density_scale * mean_mass_amu * 1.660e-24)


def _atmosphere_state(
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute main, handoff, disabled, and independent control states."""

    result = compute_atmosphere_molecular_state(inputs)
    molecular = result.molecular_state
    if molecular is None:
        raise RuntimeError("molecule-enabled atmosphere omitted molecular state")
    runtime = result.runtime_state
    from payne_zero_atmosphere import molecular_equilibrium

    continuation = atmosphere_continuation_checkpoint(inputs)
    arrays: dict[str, Any] = {
        "molecular_populations": molecular.molecular_populations,
        "partition_normalized_molecular_populations": (
            molecular.partition_normalized_molecular_populations
        ),
        "molecular_equation_densities": (molecular.molecular_equation_densities),
        "previous_molecular_equation_densities": (
            molecular.previous_molecular_equation_densities
        ),
        "electron_density": runtime.electron_density,
        "total_nuclei_number_density": runtime.total_nuclei_number_density,
        "mass_density": runtime.mass_density,
        "charge_square_density": runtime.charge_square_density,
        "ion_stage_populations_by_packed_slot": (
            runtime.ion_stage_populations_by_packed_slot
        ),
        "partition_normalized_populations_by_packed_slot": (
            runtime.partition_normalized_populations_by_packed_slot
        ),
        "fractional_doppler_widths": result.fractional_doppler_widths,
        "partition_normalized_population_over_mass_density_and_width": (
            result.partition_normalized_population_over_mass_density_and_fractional_doppler_width
        ),
        "specific_internal_energy": runtime.specific_internal_energy,
        "major_isotope_mass_amu": runtime.major_isotope_mass_amu,
        "temperature_iteration_cache_value": (
            result.temperature_iteration_cache.get("pops_itemp", -1)
        ),
        "population_constants": np.stack(
            [
                molecular_equilibrium.compute_equilibrium_constants_for_layer(
                    molecular,
                    layer_index,
                )
                for layer_index in range(np.asarray(inputs["temperature"]).size)
            ]
        ),
        "continuation_solution": continuation.continuation_solution,
        "continuation_seed": continuation.continuation_seed,
        "continuation_seed_equal": continuation.continuation_seed_equal,
        "continuation_iteration_count": (continuation.continuation_iteration_count),
        "continuation_converged": continuation.continuation_converged,
        "continuation_residual_max_abs": (continuation.continuation_residual_max_abs),
        "continuation_residual_max_scaled": (
            continuation.continuation_residual_max_scaled
        ),
    }
    structural_mask = ~np.isfinite(result.fractional_doppler_widths)
    arrays["fractional_doppler_width_structural_infinity_mask"] = structural_mask
    arrays["fractional_doppler_width_structural_infinity_slots"] = np.unique(
        np.where(structural_mask)[1]
    )
    arrays["continuation_exhausted"] = continuation.continuation_iteration_count >= 200

    handoff_inputs = {
        name: np.asarray(values).copy() for name, values in inputs.items()
    }
    handoff_inputs["electron_density_seed"] = np.asarray(
        runtime.electron_density,
        dtype=np.float64,
    ).copy()
    handoff = compute_atmosphere_molecular_state(
        handoff_inputs,
        structured_handoff=True,
    )
    handoff_molecular = handoff.molecular_state
    if handoff_molecular is None:
        raise RuntimeError("structured handoff omitted molecular state")
    handoff_runtime = handoff.runtime_state
    arrays.update(
        {
            "handoff__input_electron_density": handoff_inputs["electron_density_seed"],
            "handoff__output_electron_density": (handoff_runtime.electron_density),
            "handoff__charge_square_density": (handoff_runtime.charge_square_density),
            "handoff__fractional_doppler_widths": (handoff.fractional_doppler_widths),
            "handoff__ion_stage_populations_by_packed_slot": (
                handoff_runtime.ion_stage_populations_by_packed_slot
            ),
            "handoff__molecular_populations": (handoff_molecular.molecular_populations),
            "handoff__partition_normalized_molecular_populations": (
                handoff_molecular.partition_normalized_molecular_populations
            ),
            "handoff__partition_normalized_populations_by_packed_slot": (
                handoff_runtime.partition_normalized_populations_by_packed_slot
            ),
            "handoff__equation_densities_after_normalization": (
                handoff_molecular.molecular_equation_densities
            ),
            "handoff__transformed_molecular_equation_densities": (
                handoff_molecular.molecular_equation_densities
            ),
            "handoff__physical_equation_densities_saved_before_fill": (
                handoff_molecular.previous_molecular_equation_densities
            ),
            "handoff__raw_populations_before_fill": (
                handoff_molecular.molecular_populations
            ),
            "handoff__normalized_populations_after_fill": (
                handoff_molecular.partition_normalized_molecular_populations
            ),
            "handoff__temperature_iteration_cache_value": (
                handoff.temperature_iteration_cache.get("pops_itemp", -1)
            ),
        }
    )

    disabled = compute_atmosphere_molecular_state(
        inputs,
        pressure_iteration_enabled=False,
    )
    disabled_molecular = disabled.molecular_state
    if disabled_molecular is None:
        raise RuntimeError("disabled atmosphere route omitted molecular state")
    disabled_runtime = disabled.runtime_state
    arrays.update(
        {
            "disabled__input_electron_density": inputs["electron_density_seed"],
            "disabled__output_electron_density": (disabled_runtime.electron_density),
            "disabled__charge_square_density": (disabled_runtime.charge_square_density),
            "disabled__fractional_doppler_widths": (disabled.fractional_doppler_widths),
            "disabled__ion_stage_populations_by_packed_slot": (
                disabled_runtime.ion_stage_populations_by_packed_slot
            ),
            "disabled__molecular_populations": (
                disabled_molecular.molecular_populations
            ),
            "disabled__molecular_equation_densities": (
                disabled_molecular.molecular_equation_densities
            ),
            "disabled__partition_normalized_molecular_populations": (
                disabled_molecular.partition_normalized_molecular_populations
            ),
            "disabled__partition_normalized_populations_by_packed_slot": (
                disabled_runtime.partition_normalized_populations_by_packed_slot
            ),
            "disabled__partition_normalized_population_over_mass_density_and_width": (
                disabled.partition_normalized_population_over_mass_density_and_fractional_doppler_width
            ),
            "disabled__previous_molecular_equation_densities": (
                disabled_molecular.previous_molecular_equation_densities
            ),
            "disabled__specific_internal_energy": (
                disabled_runtime.specific_internal_energy
            ),
            "disabled__total_nuclei_number_density": (
                disabled_runtime.total_nuclei_number_density
            ),
            "disabled__pressure_iteration_enabled": (
                disabled.setup.pressure_iteration_enabled
            ),
            "disabled__temperature_iteration_cache_key_count": len(
                disabled.temperature_iteration_cache
            ),
            "disabled__temperature_iteration_cache_pops_itemp_present": (
                "pops_itemp" in disabled.temperature_iteration_cache
            ),
        }
    )

    atmosphere_catalog = load_npz_arrays(ATMOSPHERE_MOLECULE_CATALOG)
    active_codes = atmosphere_catalog["molecule_codes"][
        : int(atmosphere_catalog["molecule_count"])
    ]
    named_indices = np.asarray(
        [
            int(np.flatnonzero(np.abs(active_codes - float(code)) < 1.0e-3)[0])
            for code in inputs["named_molecule_codes"]
        ],
        dtype=np.int64,
    )

    def independent_controls(
        temperatures: np.ndarray,
        gas_pressures: np.ndarray,
        selected_indices: np.ndarray = named_indices,
    ) -> dict[str, np.ndarray]:
        records: list[dict[str, np.ndarray]] = []
        for temperature, gas_pressure in zip(
            temperatures,
            gas_pressures,
            strict=True,
        ):
            temperature_value = float(temperature)
            pressure_value = float(gas_pressure)
            one_depth = {
                "column_mass": np.asarray(
                    [inputs["column_mass"][0]],
                    dtype=np.float64,
                ),
                "temperature": np.asarray(
                    [temperature_value],
                    dtype=np.float64,
                ),
                "gas_pressure": np.asarray(
                    [pressure_value],
                    dtype=np.float64,
                ),
                "electron_density_seed": np.asarray(
                    [0.1 * pressure_value / (1.38054e-16 * temperature_value)],
                    dtype=np.float64,
                ),
                "microturbulence": np.asarray(
                    [inputs["microturbulence"][0]],
                    dtype=np.float64,
                ),
                "elemental_abundances": inputs["elemental_abundances"],
            }
            control = compute_atmosphere_molecular_state(one_depth)
            control_molecular = control.molecular_state
            if control_molecular is None:
                raise RuntimeError("independent control omitted molecular state")
            records.append(
                {
                    "raw_named_populations": (
                        control_molecular.molecular_populations[
                            0,
                            selected_indices,
                        ]
                    ),
                    "normalized_named_populations": (
                        control_molecular.partition_normalized_molecular_populations[
                            0,
                            selected_indices,
                        ]
                    ),
                    "physical_equation_densities": (
                        control_molecular.previous_molecular_equation_densities[0]
                    ),
                    "transformed_equation_densities": (
                        control_molecular.molecular_equation_densities[0]
                    ),
                    "population_constants": (
                        molecular_equilibrium.compute_equilibrium_constants_for_layer(
                            control_molecular,
                            0,
                        )
                    ),
                }
            )
        return {
            name: np.stack([record[name] for record in records]) for name in records[0]
        }

    for control_name in ("temperature_control", "pressure_control"):
        temperatures = np.asarray(inputs[f"{control_name}_temperature"])
        gas_pressures = np.asarray(inputs[f"{control_name}_gas_pressure"])
        arrays[f"{control_name}__temperature"] = temperatures
        arrays[f"{control_name}__gas_pressure"] = gas_pressures
        for name, values in independent_controls(
            temperatures,
            gas_pressures,
        ).items():
            arrays[f"{control_name}__{name}"] = values

    boundary_temperature = np.concatenate(
        (
            inputs["atmosphere_polynomial_boundary_temperature"],
            inputs["atmosphere_h2_boundary_temperature"][-2:],
        )
    )
    boundary_pressure = np.full(
        boundary_temperature.shape,
        float(inputs["temperature_control_gas_pressure"][0]),
        dtype=np.float64,
    )
    boundary_control = independent_controls(
        boundary_temperature,
        boundary_pressure,
        np.arange(active_codes.size, dtype=np.int64),
    )
    arrays["boundary__temperature"] = boundary_temperature
    arrays["boundary__temperature_uint64_bits"] = boundary_temperature.view(np.uint64)
    arrays["boundary__gas_pressure"] = boundary_pressure
    for name, values in boundary_control.items():
        arrays[f"boundary__{name}"] = values
    for name in (
        "normalized_named_populations",
        "population_constants",
        "raw_named_populations",
    ):
        arrays[f"boundary__{name}_uint64_bits"] = np.asarray(
            boundary_control[name],
            dtype=np.float64,
        ).view(np.uint64)
    arrays["boundary__polynomial_active_branch_mask"] = boundary_temperature <= 10000.0
    arrays["boundary__polynomial_inactive_branch_mask"] = boundary_temperature > 10000.0
    arrays["boundary__h2_catalog_active_branch_mask"] = boundary_temperature <= 20000.0
    arrays["boundary__h2_catalog_inactive_branch_mask"] = boundary_temperature > 20000.0

    h2_temperature = np.asarray(
        inputs["atmosphere_h2_boundary_temperature"],
        dtype=np.float64,
    )
    h2_pressure = float(inputs["temperature_control_gas_pressure"][0])
    h2_catalog_index = int(named_indices[0])
    h2_catalog_constants = []
    h2_helper_constants = []
    h2_partitions = []
    for temperature in h2_temperature:
        h2_inputs = {
            "column_mass": np.asarray(
                [inputs["column_mass"][0]],
                dtype=np.float64,
            ),
            "temperature": np.asarray([temperature], dtype=np.float64),
            "gas_pressure": np.asarray([h2_pressure], dtype=np.float64),
            "electron_density_seed": np.asarray(
                [0.1 * h2_pressure / (1.38054e-16 * float(temperature))],
                dtype=np.float64,
            ),
            "microturbulence": np.asarray(
                [inputs["microturbulence"][0]],
                dtype=np.float64,
            ),
            "elemental_abundances": inputs["elemental_abundances"],
        }
        _, _, h2_molecular = _initialize_unsolved_atmosphere_molecular_state(h2_inputs)
        h2_catalog_constants.append(
            molecular_equilibrium.compute_equilibrium_constants_for_layer(
                h2_molecular,
                0,
            )[h2_catalog_index]
        )
        h2_helper_constants.append(
            molecular_equilibrium.hydrogen_molecule_equilibrium_constant(
                float(temperature)
            )
        )
        h2_partitions.append(
            molecular_equilibrium._interp_hydrogen_molecule_partition(
                float(temperature)
            )
        )
    h2_catalog_constants_array = np.asarray(
        h2_catalog_constants,
        dtype=np.float64,
    )
    h2_helper_constants_array = np.asarray(
        h2_helper_constants,
        dtype=np.float64,
    )
    h2_partitions_array = np.asarray(h2_partitions, dtype=np.float64)
    arrays.update(
        {
            "h2_probe__temperature": h2_temperature,
            "h2_probe__temperature_uint64_bits": h2_temperature.view(np.uint64),
            "h2_probe__catalog_equilibrium_constant_gated": (
                h2_catalog_constants_array
            ),
            "h2_probe__catalog_equilibrium_constant_gated_uint64_bits": (
                h2_catalog_constants_array.view(np.uint64)
            ),
            "h2_probe__helper_equilibrium_constant_ungated": (
                h2_helper_constants_array
            ),
            "h2_probe__helper_equilibrium_constant_ungated_uint64_bits": (
                h2_helper_constants_array.view(np.uint64)
            ),
            "h2_probe__interpolated_partition": h2_partitions_array,
            "h2_probe__interpolated_partition_uint64_bits": (
                h2_partitions_array.view(np.uint64)
            ),
            "h2_probe__partition_low_clamp_branch_mask": (h2_temperature <= 100.0),
            "h2_probe__partition_interpolation_branch_mask": np.logical_and(
                h2_temperature > 100.0,
                h2_temperature < 19900.0,
            ),
            "h2_probe__partition_high_clamp_branch_mask": (h2_temperature >= 19900.0),
            "h2_probe__helper_finite_positive_input_branch_mask": (
                np.logical_and(
                    np.isfinite(h2_temperature),
                    h2_temperature > 0.0,
                )
            ),
            "h2_probe__catalog_active_branch_mask": (h2_temperature <= 20000.0),
            "h2_probe__catalog_inactive_branch_mask": (h2_temperature > 20000.0),
        }
    )

    mode_temperature = float(inputs["temperature_control_temperature"][0])
    mode_pressure = float(inputs["temperature_control_gas_pressure"][0])
    mode_inputs = {
        "column_mass": np.asarray(
            [inputs["column_mass"][0]],
            dtype=np.float64,
        ),
        "temperature": np.asarray([mode_temperature], dtype=np.float64),
        "gas_pressure": np.asarray([mode_pressure], dtype=np.float64),
        "electron_density_seed": np.asarray(
            [0.1 * mode_pressure / (1.38054e-16 * mode_temperature)],
            dtype=np.float64,
        ),
        "microturbulence": np.asarray(
            [inputs["microturbulence"][0]],
            dtype=np.float64,
        ),
        "elemental_abundances": inputs["elemental_abundances"],
    }
    for mode_name, population_mode in (("mode2", 2), ("mode12", 12)):
        _, mode_runtime, mode_molecular = (
            _initialize_unsolved_atmosphere_molecular_state(mode_inputs)
        )
        previous_before = mode_molecular.previous_molecular_equation_densities.copy()
        normalized_before = (
            mode_molecular.partition_normalized_molecular_populations.copy()
        )
        energy_before = mode_runtime.specific_internal_energy.copy()
        molecular_equilibrium.solve_molecular_equilibrium(
            mode_molecular,
            population_mode=population_mode,
        )
        arrays.update(
            {
                f"{mode_name}__population_mode": population_mode,
                f"{mode_name}__molecular_equation_densities_after": (
                    mode_molecular.molecular_equation_densities
                ),
                f"{mode_name}__normalized_populations_before": (normalized_before),
                f"{mode_name}__normalized_populations_after": (
                    mode_molecular.partition_normalized_molecular_populations
                ),
                f"{mode_name}__previous_equation_densities_before": (previous_before),
                f"{mode_name}__previous_equation_densities_after": (
                    mode_molecular.previous_molecular_equation_densities
                ),
                f"{mode_name}__raw_populations_after": (
                    mode_molecular.molecular_populations
                ),
                f"{mode_name}__specific_internal_energy_before": (energy_before),
                f"{mode_name}__specific_internal_energy_after": (
                    mode_runtime.specific_internal_energy
                ),
            }
        )

    _, energy_runtime, energy_molecular = (
        _initialize_unsolved_atmosphere_molecular_state(inputs)
    )
    energy_saved_rows = molecular.previous_molecular_equation_densities.copy()
    energy_normalized_seed = molecular.partition_normalized_molecular_populations.copy()
    molecular_equilibrium.restore_molecular_equation_density(
        energy_molecular,
        energy_saved_rows.copy(),
    )
    energy_molecular.partition_normalized_molecular_populations[:] = (
        energy_normalized_seed
    )
    energy_saved_before = energy_molecular.previous_molecular_equation_densities.copy()
    energy_normalized_before = (
        energy_molecular.partition_normalized_molecular_populations.copy()
    )
    energy_specific_before = energy_runtime.specific_internal_energy.copy()
    molecular_equilibrium.set_molecular_specific_internal_energy_mode(
        energy_molecular,
        True,
    )
    molecular_equilibrium.solve_molecular_equilibrium(
        energy_molecular,
        population_mode=1,
    )
    arrays.update(
        {
            "energy__population_mode": 1,
            "energy__molecular_equation_densities_after": (
                energy_molecular.molecular_equation_densities
            ),
            "energy__normalized_populations_before": (energy_normalized_before),
            "energy__normalized_populations_after": (
                energy_molecular.partition_normalized_molecular_populations
            ),
            "energy__raw_populations_after": (energy_molecular.molecular_populations),
            "energy__saved_physical_rows_input": energy_saved_rows,
            "energy__saved_physical_rows_before": energy_saved_before,
            "energy__saved_physical_rows_after": (
                energy_molecular.previous_molecular_equation_densities
            ),
            "energy__specific_internal_energy_before": (energy_specific_before),
            "energy__specific_internal_energy_after": (
                energy_runtime.specific_internal_energy
            ),
            "energy__specific_internal_energy_mode_enabled": (
                energy_molecular.specific_internal_energy_mode_enabled
            ),
            "energy__direct_specific_internal_energy_reference": (
                molecular_equilibrium.compute_molecular_specific_internal_energy(
                    energy_molecular
                )
            ),
        }
    )

    from payne_zero_atmosphere import synthesis_bridge
    from payne_zero_atmosphere.population_layout import (
        population_job_schedule,
    )

    bridge_error: ValueError | None = None
    try:
        synthesis_bridge.structured_atmosphere_from_runtime_state(
            atmosphere=result.setup.atmosphere,
            runtime_state=runtime,
            molecular_state=molecular,
        )
    except ValueError as error:
        bridge_error = error
    if bridge_error is None:
        raise RuntimeError(
            "the exact live/debug bridge unexpectedly accepted packed shapes"
        )
    bridge_temperature = np.asarray(
        [8000.0, 9500.0, 7000.0],
        dtype=np.float64,
    )
    bridge_packed = np.zeros(
        (bridge_temperature.size, 1006),
        dtype=np.float64,
    )
    bridge_packed[:, 0] = np.asarray(
        [2.0e14, 3.0e14, 4.0e14],
        dtype=np.float64,
    )
    bridge_codes = np.asarray([101.0], dtype=np.float64)
    bridge_mixed = np.asarray(
        [[5.0e8], [0.0], [7.0e8]],
        dtype=np.float64,
    )
    bridge_zero = np.zeros_like(bridge_mixed)
    bridge_arguments = {
        "temperature": bridge_temperature,
        "ion_stage_populations_by_packed_slot": bridge_packed,
        "molecule_codes": bridge_codes,
    }
    arrays.update(
        {
            "bridge__h2_temperature": bridge_temperature,
            "bridge__h2_packed_neutral_hydrogen": bridge_packed[:, 0],
            "bridge__h2_mixed_catalog_input": bridge_mixed,
            "bridge__h2_all_zero_catalog_input": bridge_zero,
            "bridge__h2_mixed_output": (
                synthesis_bridge._molecular_hydrogen_population(
                    **bridge_arguments,
                    molecular_populations=bridge_mixed,
                )
            ),
            "bridge__h2_all_zero_output": (
                synthesis_bridge._molecular_hydrogen_population(
                    **bridge_arguments,
                    molecular_populations=bridge_zero,
                )
            ),
            "bridge__h2_no_catalog_output": (
                synthesis_bridge._molecular_hydrogen_population(
                    temperature=bridge_temperature,
                    ion_stage_populations_by_packed_slot=bridge_packed,
                    molecule_codes=None,
                    molecular_populations=None,
                )
            ),
            "bridge__live_shape_error_message": str(bridge_error),
            "bridge__live_shape_error_type": type(bridge_error).__name__,
            "bridge__padded_molecule_code_shape": (
                molecular.catalog.molecule_codes.shape
            ),
            "bridge__active_molecular_population_shape": (
                molecular.molecular_populations.shape
            ),
        }
    )

    jobs = population_job_schedule(include_molecules=True)
    schedule_codes = np.asarray([job.code for job in jobs], dtype=np.float64)
    schedule_modes = np.asarray([job.mode for job in jobs], dtype=np.int64)
    schedule_starts = np.asarray(
        [job.start_slot for job in jobs],
        dtype=np.int64,
    )
    schedule_output_slots = np.asarray(
        [job.output_slots for job in jobs],
        dtype=np.int64,
    )
    schedule_targets = np.asarray(
        [
            0 if job.target == "partition_normalized_populations_by_packed_slot" else 1
            for job in jobs
        ],
        dtype=np.int64,
    )
    molecular_mask = schedule_codes >= 100.0
    atomic_mask = ~molecular_mask
    molecular_codes = schedule_codes[molecular_mask][::2]
    arrays.update(
        {
            "schedule_inventory__code": schedule_codes,
            "schedule_inventory__mode": schedule_modes,
            "schedule_inventory__packed_start_slot_zero_based": (schedule_starts),
            "schedule_inventory__packed_start_slot_one_based": (schedule_starts + 1),
            "schedule_inventory__output_slots": schedule_output_slots,
            "schedule_inventory__target_index": schedule_targets,
            "schedule_inventory__target_name": np.asarray(
                [
                    (
                        "partition_normalized_populations_by_packed_slot"
                        if target == 0
                        else "ion_stage_populations_by_packed_slot"
                    )
                    for target in schedule_targets
                ],
                dtype=np.str_,
            ),
            "schedule_inventory__molecular_job_mask": molecular_mask,
            "schedule_inventory__atomic_job_mask": atomic_mask,
            "schedule_inventory__job_count": schedule_codes.size,
            "schedule_inventory__molecular_job_count": np.count_nonzero(molecular_mask),
            "schedule_inventory__atomic_job_count": np.count_nonzero(atomic_mask),
            "schedule_inventory__molecular_unique_code": molecular_codes,
            "schedule_inventory__molecular_unique_code_count": (molecular_codes.size),
            "schedule_inventory__molecular_unique_mode_pair": (
                schedule_modes[molecular_mask].reshape(-1, 2)
            ),
            "schedule_inventory__molecular_unique_packed_start_slot_one_based": (
                schedule_starts[molecular_mask][::2] + 1
            ),
        }
    )
    return _freeze_mapping(arrays)


def _synthesis_state(
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute full, fixed-derived, and fixed-supplied synthesis branches."""

    full, full_calls = _capture_molecular_calls(
        lambda: compute_synthesis_molecular_state(
            inputs,
            fixed_electron_density=False,
        )
    )
    fixed_derived, derived_calls = _capture_molecular_calls(
        lambda: compute_synthesis_molecular_state(
            inputs,
            fixed_electron_density=True,
        )
    )
    supplied_mass = _deterministic_supplied_mass_density(inputs)
    fixed_supplied, supplied_calls = _capture_molecular_calls(
        lambda: compute_synthesis_molecular_state(
            inputs,
            fixed_electron_density=True,
            mass_density=supplied_mass,
        )
    )
    if len(full_calls) != 2:
        raise RuntimeError("full synthesis route must own two molecular calls")
    if len(derived_calls) != 1 or len(supplied_calls) != 1:
        raise RuntimeError("each fixed synthesis route must own one molecular call")

    arrays: dict[str, Any] = {}
    arrays.update(_flatten_population_state(full, prefix="full__state__"))
    arrays.update(
        _flatten_population_state(
            fixed_derived,
            prefix="derived__state__",
        )
    )
    arrays.update(
        _flatten_population_state(
            fixed_supplied,
            prefix="supplied__state__",
        )
    )
    arrays["supplied__input__mass_density"] = supplied_mass
    for branch, calls in (
        ("full", full_calls),
        ("derived", derived_calls),
        ("supplied", supplied_calls),
    ):
        for index, call in enumerate(calls):
            for name, values in call.items():
                arrays[f"{branch}__call_{index}__{name}"] = values
    arrays["full__molecular_call_count"] = len(full_calls)
    arrays["derived__molecular_call_count"] = len(derived_calls)
    arrays["supplied__molecular_call_count"] = len(supplied_calls)
    return _freeze_mapping(arrays)


def _synthesis_boundary_state(
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rebuild the exact 10000 K and provisional-H2 branch contracts."""

    import torch

    from payne_zero_synthesis import equation_of_state
    from payne_zero_synthesis import molecular_equilibrium
    from payne_zero_synthesis.constants import (
        REFERENCE_BOLTZMANN_ERG_PER_K,
        REFERENCE_BOLTZMANN_EV_PER_K,
    )

    polynomial_temperature = np.asarray(
        inputs["synthesis_polynomial_boundary_temperature"],
        dtype=np.float64,
    )
    polynomial_gas_pressure = np.full(
        polynomial_temperature.size,
        1.0e5,
        dtype=np.float64,
    )
    polynomial_electron_density = (
        0.1
        * polynomial_gas_pressure
        / (REFERENCE_BOLTZMANN_ERG_PER_K * polynomial_temperature)
    )
    tables = load_synthesis_eos_tables()
    metadata = molecular_equilibrium.molecular_equilibrium_metadata(
        SYNTHESIS_MOLECULE_CATALOG
    )
    molecule_table = molecular_equilibrium.read_molecule_table(
        SYNTHESIS_MOLECULE_CATALOG
    )
    ion_constants = equation_of_state.molecular_ion_formation_constants_from_seed(
        polynomial_temperature,
        polynomial_gas_pressure,
        polynomial_electron_density,
        tables=tables,
        meta=metadata,
    )
    direct_polynomial = molecular_equilibrium.polynomial_formation_constants(
        polynomial_temperature,
        molecule_table,
    )
    formation_constants = direct_polynomial.copy()
    polynomial_mask = (
        metadata.equilibrium_coefficients[0, : metadata.molecule_count] != 0.0
    )
    component_counts = (
        metadata.component_start_indices[1 : metadata.molecule_count + 1]
        - metadata.component_start_indices[: metadata.molecule_count]
    )
    ion_mask = ~polynomial_mask
    single_mask = ion_mask & (component_counts == 1)
    ion_ratio_mask = ion_mask & (component_counts > 1)
    formation_constants[:, single_mask] = 1.0
    formation_constants[:, ion_ratio_mask] = ion_constants[
        :,
        : metadata.molecule_count,
    ][:, ion_ratio_mask]
    with np.errstate(divide="ignore"):
        natural_log_constants = np.where(
            formation_constants > 0.0,
            np.log(np.maximum(formation_constants, 1.0e-300)),
            -700.0,
        )

    h2_temperature = np.asarray(
        inputs["synthesis_named_h2_boundary_temperature"],
        dtype=np.float64,
    )
    h2_gas_pressure = np.full(
        h2_temperature.size,
        1.0e5,
        dtype=np.float64,
    )
    h2_electron_density = (
        0.1 * h2_gas_pressure / (REFERENCE_BOLTZMANN_ERG_PER_K * h2_temperature)
    )
    boundary_inputs = dict(inputs)
    boundary_inputs.update(
        {
            "column_mass": np.asarray(inputs["column_mass"][:2], dtype=np.float64),
            "temperature": h2_temperature,
            "gas_pressure": h2_gas_pressure,
            "electron_density_seed": h2_electron_density,
            "microturbulence": np.asarray(
                inputs["microturbulence"][:2],
                dtype=np.float64,
            ),
        }
    )
    h2_state, h2_calls = _capture_molecular_calls(
        lambda: compute_synthesis_molecular_state(
            boundary_inputs,
            fixed_electron_density=True,
        )
    )
    if len(h2_calls) != 1:
        raise RuntimeError("provisional-H2 boundary must own one molecular call")
    h2_call = h2_calls[0]
    neutral_hydrogen = np.asarray(
        h2_state.hydrogen_neutral_population,
        dtype=np.float64,
    )
    thermal_energy_ev = h2_temperature * REFERENCE_BOLTZMANN_EV_PER_K
    natural_log_temperature = np.log(h2_temperature)
    equilibrium_factor = np.exp(
        4.478 / thermal_energy_ev
        - 4.64584e1
        + (
            1.63660e-3
            + (
                -4.93992e-7
                + (
                    1.11822e-10
                    + (
                        -1.49567e-14
                        + (1.06206e-18 - 3.08720e-23 * h2_temperature) * h2_temperature
                    )
                    * h2_temperature
                )
                * h2_temperature
            )
            * h2_temperature
        )
        * h2_temperature
        - 1.5 * natural_log_temperature
    )
    provisional_h2 = neutral_hydrogen**2 * equilibrium_factor
    provisional_h2[h2_temperature > 9000.0] = 0.0

    polynomial_threshold_bits = np.asarray([np.float64(10000.0).view(np.uint64)])
    h2_threshold_bits = np.asarray([np.float64(9000.0).view(np.uint64)])
    arrays = {
        "boundary__synthesis_polynomial_temperature": polynomial_temperature,
        "boundary__synthesis_polynomial_temperature_bits": (
            polynomial_temperature.view(np.uint64)
        ),
        "boundary__synthesis_polynomial_threshold_bits": (polynomial_threshold_bits),
        "boundary__synthesis_polynomial_temperature_branch_mask": (
            polynomial_temperature <= 10000.0
        ),
        "boundary__synthesis_polynomial_gas_pressure": (polynomial_gas_pressure),
        "boundary__synthesis_polynomial_electron_density": (
            polynomial_electron_density
        ),
        "boundary__synthesis_polynomial_mask": polynomial_mask,
        "boundary__synthesis_polynomial_molecule_codes": (
            metadata.molecule_codes[: metadata.molecule_count]
        ),
        "boundary__synthesis_polynomial_direct_constants": (direct_polynomial),
        "boundary__synthesis_pre_newton_formation_constants": (formation_constants),
        "boundary__synthesis_pre_newton_log_formation_constants": (
            natural_log_constants
        ),
        "boundary__provisional_h2_temperature": h2_temperature,
        "boundary__provisional_h2_temperature_bits": (h2_temperature.view(np.uint64)),
        "boundary__provisional_h2_threshold_bits": h2_threshold_bits,
        "boundary__provisional_h2_temperature_branch_mask": (h2_temperature <= 9000.0),
        "boundary__provisional_h2_active_population_mask": (provisional_h2 > 0.0),
        "boundary__provisional_h2_gas_pressure": h2_gas_pressure,
        "boundary__provisional_h2_electron_density": h2_electron_density,
        "boundary__provisional_h2_neutral_hydrogen": neutral_hydrogen,
        "boundary__provisional_h2_equilibrium_factor": equilibrium_factor,
        "boundary__provisional_h2_population": provisional_h2,
        "boundary__provisional_h2_molecular_iterations": (
            h2_call["diag__iterations_completed"]
        ),
        "boundary__provisional_h2_molecular_exhausted_mask": (
            h2_call["diag__exhausted_mask"]
        ),
    }
    if tables.device.type != "cpu" or tables.dtype != torch.float64:
        raise RuntimeError("synthesis boundary parity requires CPU float64")
    return _freeze_mapping(arrays)


def _public_state(
    inputs: Mapping[str, np.ndarray],
    synthesis: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute all public structured fields and molecular-lane diagnostics."""

    checkpoint = build_public_molecular_lane_checkpoint(inputs)
    structured = checkpoint.structured_atmosphere
    from payne_zero_synthesis import equation_of_state
    from payne_zero_synthesis import molecular_equilibrium

    public_lanes = structured["partition_normalized_populations"][
        :, 5, checkpoint.public_columns
    ]
    h2_index = int(
        np.flatnonzero(
            np.abs(synthesis["derived__call_0__diag__molecule_codes"] - 101.0) < 1.0e-3
        )[0]
    )
    arrays: dict[str, Any] = {
        "mapping__species_codes": checkpoint.line_species_codes,
        "mapping__molecule_code_offsets": (checkpoint.equilibrium_code_offsets),
        "mapping__molecule_codes": checkpoint.equilibrium_codes,
        "mapping__public_columns": checkpoint.public_columns,
        "partition_cube__before": checkpoint.partition_cube_before,
        "partition_cube__after": checkpoint.partition_cube_after,
        "ion_cube__before": checkpoint.ion_cube_before,
        "ion_cube__after": checkpoint.ion_cube_after,
        "line_population__public": public_lanes,
        "line_population__independent_no_ground": (
            checkpoint.no_ground_line_populations
        ),
        "line_population__grounded": (checkpoint.grounded_line_populations),
        "line_population__ground_discrimination_mask": (
            checkpoint.ground_discrimination_mask
        ),
        "molecular_hydrogen__catalog_index": h2_index,
        "molecular_hydrogen__solved_code_101": (
            synthesis["derived__call_0__output__molecular_populations"][:, h2_index]
        ),
        "co_reconstruction__line_species_code": (checkpoint.co_line_species_code),
        "co_reconstruction__equilibrium_code": (checkpoint.co_equilibrium_code),
        "co_reconstruction__public_stage_index": 5,
        "co_reconstruction__public_column_index": checkpoint.co_public_column,
        "co_reconstruction__catalog_index": checkpoint.co_catalog_index,
        "co_reconstruction__component_equation_indices": (
            checkpoint.co_component_equation_indices.astype(
                np.int32,
                copy=False,
            )
        ),
        "co_reconstruction__component_species_codes": (
            checkpoint.co_component_species_codes.astype(
                np.int32,
                copy=False,
            )
        ),
        "co_reconstruction__component_atomic_masses_amu": (
            checkpoint.co_component_atomic_masses_amu
        ),
        "co_reconstruction__molecular_mass_amu": (checkpoint.co_molecular_mass_amu),
        "co_reconstruction__leading_coefficient": (checkpoint.co_leading_coefficient),
        "co_reconstruction__raw_equilibrium_population": (checkpoint.co_raw_population),
        "co_reconstruction__reference_public_lane": (
            checkpoint.co_normalized_population
        ),
        "co_reconstruction__independent_population": (
            checkpoint.co_independent_population
        ),
        "co_reconstruction__raw_discrimination_mask": (
            checkpoint.co_raw_population != checkpoint.co_independent_population
        ),
        "co_reconstruction__temperature": checkpoint.co_temperature,
        "co_reconstruction__equation_densities": (checkpoint.co_equation_densities),
        "co_reconstruction__transformed_equation_densities": (
            checkpoint.co_transformed_equation_densities
        ),
        "co_reconstruction__no_ground_neutral_partitions": (
            checkpoint.co_no_ground_neutral_partitions
        ),
        "co_reconstruction__normalization": checkpoint.co_normalization,
        "owned_stage5_mask": checkpoint.owned_stage5_mask,
        "normalized_delta": checkpoint.normalized_delta,
        "line_population__ground_discrimination_max_abs": np.max(
            np.abs(
                checkpoint.no_ground_line_populations
                - checkpoint.grounded_line_populations
            )
        ),
        "fixed_eos_call_count": checkpoint.fixed_eos_call_count,
        "molecular_solve_call_count": checkpoint.molecular_solve_call_count,
        "reused_fixed_molecular_arrays": (checkpoint.reused_fixed_molecular_arrays),
        "edge_grid_call_count": checkpoint.edge_grid_call_count,
    }
    edge_arrays = load_npz_arrays(SYNTHESIS_CONTINUUM_EDGE_GRID)
    for name, values in edge_arrays.items():
        loader_name = (
            "edge_interval_width_squared_over_two_nm2"
            if name == "continuum_edge_interval_width_squared_over_two_nm2"
            else name
        )
        arrays[f"edge_loader__{loader_name}"] = values

    grounded_partitions = synthesis["derived__state__eos__partition_functions"]
    metadata = molecular_equilibrium.molecular_equilibrium_metadata(
        SYNTHESIS_MOLECULE_CATALOG
    )
    partition_elements = np.asarray(
        sorted(
            {
                int(code)
                for code in metadata.equation_species_codes[: metadata.equation_count]
                if 1 <= int(code) <= 99
            }
        ),
        dtype=np.int64,
    )
    without_ground = equation_of_state.partition_functions_for_elements(
        inputs["temperature"],
        inputs["gas_pressure"],
        inputs["electron_density_seed"],
        tables=load_synthesis_eos_tables(),
        elements=partition_elements,
        nion=6,
        apply_ground_partition=False,
    )
    stage_counts = np.asarray(
        [without_ground[int(code)].shape[1] for code in partition_elements],
        dtype=np.int64,
    )
    offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(stage_counts, dtype=np.int64),
        )
    )
    bridge_partitions = np.asarray(grounded_partitions).copy()
    for atomic_number, values in without_ground.items():
        copy_count = min(values.shape[1], bridge_partitions.shape[2])
        bridge_partitions[:, int(atomic_number) - 1, :copy_count] = values[
            :, :copy_count
        ]
    arrays.update(
        {
            "partition__elements_without_ground_floor": partition_elements,
            "partition__without_ground_floor_stage_counts": stage_counts,
            "partition__without_ground_floor_offsets": offsets,
            "partition__without_ground_floor": np.concatenate(
                [without_ground[int(code)] for code in partition_elements],
                axis=1,
            ),
            "partition__grounded_cube": grounded_partitions,
            "partition__bridge_cube": bridge_partitions,
        }
    )
    for name, values in structured.items():
        arrays[f"structured__{name}"] = values

    arrays.update(
        _flatten_population_state(
            checkpoint.fixed_population_state,
            prefix="fixed_state__",
        )
    )
    public_call = _molecular_call_arrays(
        effective=checkpoint.molecular_call_arguments,
        outputs=checkpoint.molecular_call_outputs,
        diagnostics=checkpoint.molecular_call_diagnostics,
        caller_name=checkpoint.molecular_call_caller_name,
        caller_module=checkpoint.molecular_call_caller_module,
        caller_file=checkpoint.molecular_call_caller_file,
        molecular_module=molecular_equilibrium,
    )
    for name, values in public_call.items():
        arrays[f"call_0__{name}"] = values
    return _freeze_mapping(arrays)


def compute_chapter04_local_parity_state() -> dict[str, Any]:
    """Complete every local scientific calculation and detach all outputs."""

    configure_local_data_paths()
    inputs = load_molecular_inputs()
    catalog = _catalog_state()
    atmosphere = _atmosphere_state(inputs)
    synthesis = _synthesis_state(inputs)
    synthesis.update(_synthesis_boundary_state(inputs))
    public = _public_state(inputs, synthesis)
    return {
        "inputs": _freeze_mapping(inputs),
        "catalog": catalog,
        "atmosphere": atmosphere,
        "synthesis": synthesis,
        "public": public,
        "coverage": (
            "catalog_active_buffers_and_semantic_overlap",
            "atmosphere_final_molecular_packed_energy_and_continuation",
            "synthesis_full_two_call_fixed_derived_fixed_supplied",
            "public_27_fields_mapping_owned_lanes_co_h2_edges",
        ),
        "excluded_oracle_trace_ownership": (
            "archive_publication_metadata",
            "runtime_environment_and_executed_source_provenance",
            "atmosphere_per_iteration_and_lifecycle_instrumentation",
            "synthesis_local_path_identity",
            "public_builder_trace_instrumentation",
        ),
    }
