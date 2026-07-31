"""Small Chapter 3 orchestration helpers around the exact progressive sources.

The functions in this module keep notebook cells readable. They do not replace
the physical kernels: every population, closure, Doppler, energy, and mapping
result is produced by the exact local Payne Zero definitions in ``src/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "data"
STATIC_DATA_ROOT = DATA_ROOT / "static"
ATOM_ONLY_FIXTURE = DATA_ROOT / "fixtures" / "chapter03_atom_only_inputs.npz"
SYNTHESIS_STATE_FIXTURE = (
    DATA_ROOT / "fixtures" / "chapter03_synthesis_eos_state.npz"
)
SYNTHESIS_TABLES = (
    STATIC_DATA_ROOT / "synthesis_tables" / "partition_saha_tables.npz"
)
ATOMIC_MASSES = STATIC_DATA_ROOT / "synthesis_tables" / "atomic_masses.npz"


def configure_local_data_paths() -> None:
    """Point exact package loaders at this repository's manifest-bound data."""

    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(STATIC_DATA_ROOT)
    os.environ["PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE"] = str(ATOMIC_MASSES)


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load an NPZ into independent arrays so no archive handle escapes."""

    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def load_atom_only_fixture() -> dict[str, np.ndarray]:
    """Return the controlled six-depth atom-only integration fixture."""

    return load_npz_arrays(ATOM_ONLY_FIXTURE)


def build_atmosphere_state(
    inputs: Mapping[str, np.ndarray],
):
    """Build the exact atmosphere runtime state from declared linear inputs."""

    configure_local_data_paths()
    from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere
    from payne_zero_atmosphere.runtime_state import build_runtime_state

    temperature = np.asarray(inputs["temperature"], np.float64)
    abundance = np.asarray(inputs["elemental_abundances"], np.float64)
    abundance_deck = {
        atomic_number: (
            float(abundance[atomic_number - 1])
            if atomic_number <= 2
            else float(np.log10(abundance[atomic_number - 1]))
        )
        for atomic_number in range(1, 100)
    }
    zeros = np.zeros_like(temperature)
    atmosphere = ModelAtmosphere(
        column_mass=np.asarray(inputs["column_mass"], np.float64),
        temperature=temperature,
        gas_pressure=np.asarray(inputs["gas_pressure"], np.float64),
        electron_density=np.asarray(inputs["electron_density_seed"], np.float64),
        rosseland_opacity=zeros,
        radiative_acceleration=zeros,
        microturbulence=np.asarray(inputs["microturbulence"], np.float64),
        convective_flux=zeros,
        convective_velocity=zeros,
        fixed_column_abundance_values=abundance_deck,
    )
    return atmosphere, build_runtime_state(atmosphere)


def compute_atmosphere_saha_modes(
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Evaluate exact scalar and batch atmosphere modes 11, 12, and 13."""

    configure_local_data_paths()
    from payne_zero_atmosphere.equation_of_state import (
        saha_partition_depth,
        saha_partition_depth_batch,
    )

    temperature = np.asarray(inputs["saha_temperature"], np.float64)
    electron_density = np.asarray(inputs["saha_electron_density"], np.float64)
    charge_square = np.asarray(
        inputs["saha_charge_square_density"], np.float64
    )
    nuclei_density = np.asarray(
        inputs["saha_total_nuclei_number_density"], np.float64
    )
    atomic_numbers = np.asarray(inputs["saha_atomic_number"], np.int64)
    stage_counts = np.asarray(inputs["saha_ion_stage_count"], np.int64)
    modes = np.asarray(inputs["saha_population_mode"], np.int64)
    abundance = np.asarray(inputs["elemental_abundances"], np.float64)

    outputs: dict[str, np.ndarray] = {}
    for atomic_number, stage_count in zip(atomic_numbers, stage_counts):
        for mode in modes:
            prefix = f"z{int(atomic_number):02d}_mode{int(mode)}"
            batch = saha_partition_depth_batch(
                temperature,
                electron_density,
                int(atomic_number),
                int(stage_count),
                int(mode),
                charge_square,
            )
            scalar = np.stack(
                [
                    saha_partition_depth(
                        temperature_k=float(temperature[index]),
                        electron_density_cm3=float(electron_density[index]),
                        total_nuclei_number_density_cm3=float(
                            nuclei_density[index]
                        ),
                        elemental_abundance=float(
                            abundance[int(atomic_number) - 1]
                        ),
                        atomic_number=int(atomic_number),
                        ion_stage_count=int(stage_count),
                        population_mode=int(mode),
                        charge_square_density_cm3=float(charge_square[index]),
                    )
                    for index in range(temperature.size)
                ]
            )
            outputs[f"{prefix}_batch"] = batch
            outputs[f"{prefix}_scalar"] = scalar
    return outputs


def compute_atmosphere_atomic_state(
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run atom-only closure, refresh packed populations, and build support."""

    configure_local_data_paths()
    from payne_zero_atmosphere.doppler import (
        update_doppler_line_strength_factors,
    )
    from payne_zero_atmosphere.equation_of_state import populate_all_species
    from payne_zero_atmosphere.runtime_state import (
        update_charge_square_density,
    )
    from payne_zero_atmosphere.specific_internal_energy import (
        compute_atomic_specific_internal_energy,
    )

    atmosphere, state = build_atmosphere_state(inputs)
    update_charge_square_density(
        thermal_energy_erg=atmosphere.thermal_energy_erg,
        state=state,
    )
    populate_all_species(
        temperature_k=atmosphere.temperature,
        thermal_energy_erg=atmosphere.thermal_energy_erg,
        state=state,
        molecules_enabled=False,
        pressure_iteration_enabled=True,
        temperature_iteration_index=1,
        temperature_iteration_cache={},
    )
    doppler, population_over_density_and_width = (
        update_doppler_line_strength_factors(
            thermal_energy_erg=atmosphere.thermal_energy_erg,
            microturbulence=atmosphere.microturbulence,
            state=state,
        )
    )
    atomic_energy = compute_atomic_specific_internal_energy(
        temperature_k=atmosphere.temperature,
        state=state,
    )
    return {
        "atomic_specific_internal_energy": atomic_energy,
        "charge_square_density": state.charge_square_density.copy(),
        "electron_density": state.electron_density.copy(),
        "fractional_doppler_widths": doppler,
        "ion_stage_populations_by_packed_slot": (
            state.ion_stage_populations_by_packed_slot.copy()
        ),
        "mass_density": state.mass_density.copy(),
        (
            "partition_normalized_population_over_mass_density_"
            "and_fractional_doppler_width"
        ): population_over_density_and_width,
        "partition_normalized_populations_by_packed_slot": (
            state.partition_normalized_populations_by_packed_slot.copy()
        ),
        "total_nuclei_number_density": (
            state.total_nuclei_number_density.copy()
        ),
    }


def compute_atmosphere_atomic_energy_breakdown(
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rebuild the exact atomic energy as translation, ionization, excitation."""

    atomic_state = compute_atmosphere_atomic_state(inputs)
    configure_local_data_paths()
    from payne_zero_atmosphere.constants import (
        BOLTZMANN_ERG_PER_K_REFERENCE,
        LIGHT_SPEED_CM_PER_S_EXACT,
        PLANCK_ERG_SECOND_REFERENCE,
    )
    from payne_zero_atmosphere.equation_of_state import (
        saha_partition_depth_batch,
    )
    from payne_zero_atmosphere.population_layout import (
        atomic_population_slot_start,
        ion_stage_count_for_atomic_number,
    )
    from payne_zero_atmosphere.specific_internal_energy import (
        _ionization_potential_sums,
    )

    temperature = np.asarray(inputs["temperature"], np.float64)
    layer_count = temperature.size
    slot_count = 840
    partition_plus = np.ones((layer_count, slot_count), dtype=np.float64)
    partition_minus = np.ones((layer_count, slot_count), dtype=np.float64)
    for atomic_number in range(1, 100):
        stage_count = (
            ion_stage_count_for_atomic_number(atomic_number)
            if atomic_number <= 30
            else 3
        )
        start = atomic_population_slot_start(atomic_number)
        count = min(stage_count, slot_count - start)
        if count <= 0:
            continue
        for target, factor in (
            (partition_plus, 1.001),
            (partition_minus, 0.999),
        ):
            values = saha_partition_depth_batch(
                np.maximum(temperature, 1.0) * factor,
                atomic_state["electron_density"],
                atomic_number,
                stage_count,
                13,
                atomic_state["charge_square_density"],
            )
            target[:, start : start + count] = values[:, :count]

    logarithmic_partition_response = (
        (partition_plus - partition_minus)
        / np.maximum(partition_plus + partition_minus, 1.0e-30)
        * 1000.0
    )
    populations = atomic_state["ion_stage_populations_by_packed_slot"][
        :, :slot_count
    ]
    mass_density = atomic_state["mass_density"]
    thermal_energy = BOLTZMANN_ERG_PER_K_REFERENCE * temperature
    translation = (
        1.5
        * (
            atomic_state["electron_density"]
            + atomic_state["total_nuclei_number_density"]
        )
        * thermal_energy
        / mass_density
    )
    ionization = (
        np.sum(populations * _ionization_potential_sums()[:slot_count], axis=1)
        * PLANCK_ERG_SECOND_REFERENCE
        * LIGHT_SPEED_CM_PER_S_EXACT
        / mass_density
    )
    excitation = (
        np.sum(
            populations
            * thermal_energy[:, None]
            * logarithmic_partition_response,
            axis=1,
        )
        / mass_density
    )
    return {
        "translation_specific_energy": translation,
        "cumulative_ionization_specific_energy": ionization,
        "partition_excitation_specific_energy": excitation,
        "reconstructed_specific_energy": translation + ionization + excitation,
        "exact_specific_energy": atomic_state["atomic_specific_internal_energy"],
    }


def compute_atmosphere_atomic_state_by_depth(
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Repeat the full refreshed route as independent one-depth calls."""

    depth_count = np.asarray(inputs["temperature"]).size
    depth_keys = {
        "column_mass",
        "electron_density_seed",
        "gas_pressure",
        "microturbulence",
        "temperature",
    }
    results = []
    for depth_index in range(depth_count):
        one_depth = {
            name: (
                np.asarray(values)[depth_index : depth_index + 1]
                if name in depth_keys
                else np.asarray(values)
            )
            for name, values in inputs.items()
        }
        results.append(compute_atmosphere_atomic_state(one_depth))
    return {
        name: np.concatenate([result[name] for result in results], axis=0)
        for name in results[0]
    }


def compute_packed_bridge(
    atmospheric_state: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map exact packed arrays to public cubes and preserve the supplied ne."""

    configure_local_data_paths()
    from payne_zero_atmosphere.synthesis_bridge import _packed_atomic_cube

    actual_packed = np.asarray(
        atmospheric_state["ion_stage_populations_by_packed_slot"], np.float64
    )
    normalized_packed = np.asarray(
        atmospheric_state[
            "partition_normalized_populations_by_packed_slot"
        ],
        np.float64,
    )
    doppler_packed = np.asarray(
        atmospheric_state["fractional_doppler_widths"], np.float64
    )
    actual, normalized, doppler = _packed_atomic_cube(
        ion_stage_populations_by_packed_slot=actual_packed,
        partition_normalized_populations_by_packed_slot=normalized_packed,
        fractional_doppler_widths_by_packed_slot=doppler_packed,
    )
    electron_density = np.asarray(
        atmospheric_state["electron_density"], np.float64
    ).copy()
    return {
        "aluminum_neutral_partition_normalized_population": (
            normalized_packed[:, 90].copy()
        ),
        "carbon_partition_normalized_ion_stage_populations": (
            normalized_packed[:, 20:22].copy()
        ),
        "electron_density": electron_density,
        "fixed_input_electron_density": electron_density.copy(),
        "fractional_doppler_widths": doppler,
        "helium_neutral_population": actual_packed[:, 2].copy(),
        "helium_singly_ionized_population": actual_packed[:, 3].copy(),
        "hydrogen_ionized_population": actual_packed[:, 1].copy(),
        "hydrogen_neutral_population": actual_packed[:, 0].copy(),
        "hydrogen_partition_normalized_ion_stage_populations": (
            normalized_packed[:, 0:2].copy()
        ),
        "ion_stage_populations": actual,
        "iron_neutral_partition_normalized_population": (
            normalized_packed[:, 350].copy()
        ),
        "magnesium_neutral_partition_normalized_population": (
            normalized_packed[:, 77].copy()
        ),
        "partition_normalized_populations": normalized,
        "silicon_neutral_partition_normalized_population": (
            normalized_packed[:, 104].copy()
        ),
    }


def compute_atmosphere_fixed_handoff_state(
    inputs: Mapping[str, np.ndarray],
    electron_density: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run the exact atom-only structured handoff at one retained density."""

    configure_local_data_paths()
    from payne_zero_atmosphere.config import (
        AtmosphereConfig,
        AtmosphereInput,
        AtmosphereOutput,
    )
    from payne_zero_atmosphere.runner import (
        prepare_structured_handoff_population_state,
    )

    fixed_inputs = {
        name: np.asarray(values).copy() for name, values in inputs.items()
    }
    fixed_input_electron_density = np.asarray(
        electron_density, np.float64
    ).copy()
    if fixed_input_electron_density.shape != np.asarray(
        inputs["temperature"]
    ).shape:
        raise ValueError("electron_density must have one value per depth")
    fixed_inputs["electron_density_seed"] = fixed_input_electron_density
    atmosphere, _ = build_atmosphere_state(fixed_inputs)

    # Chapter 1's unit-opacity scaffold satisfies the structured seed
    # validator without pretending that Chapter 3 has computed opacity.
    atmosphere.rosseland_opacity[:] = 1.0
    config = AtmosphereConfig(
        inputs=AtmosphereInput(initial_atmosphere=atmosphere),
        outputs=AtmosphereOutput(),
        enable_molecules=False,
        enable_convection=False,
    )
    handoff = prepare_structured_handoff_population_state(config)
    runtime_state = handoff.runtime_state
    packed_state = {
        "electron_density": runtime_state.electron_density.copy(),
        "fractional_doppler_widths": (
            handoff.fractional_doppler_widths.copy()
        ),
        "ion_stage_populations_by_packed_slot": (
            runtime_state.ion_stage_populations_by_packed_slot.copy()
        ),
        "partition_normalized_populations_by_packed_slot": (
            runtime_state.partition_normalized_populations_by_packed_slot.copy()
        ),
    }
    public_state = compute_packed_bridge(packed_state)
    public_state["fixed_input_electron_density"] = (
        fixed_input_electron_density
    )
    public_state[
        "partition_normalized_population_over_mass_density_"
        "and_fractional_doppler_width"
    ] = (
        handoff
        .partition_normalized_population_over_mass_density_and_fractional_doppler_width
        .copy()
    )
    return public_state


def load_synthesis_tables(*, device=None, dtype=None):
    """Construct exact synthesis EOS tables under one declared runtime policy."""

    configure_local_data_paths()
    import torch
    from payne_zero_synthesis.equation_of_state import EOSTables

    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float64
    static = load_npz_arrays(SYNTHESIS_TABLES)
    fixture = load_npz_arrays(SYNTHESIS_STATE_FIXTURE)
    static["ground_partition_table"] = fixture["ground_partition_table"]
    return EOSTables.from_dict(
        static,
        device=device,
        dtype=dtype,
    )


def probe_partition_branches(
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Probe special, packed-ordinary, and PFIRON exact partition routes."""

    configure_local_data_paths()
    from payne_zero_synthesis.equation_of_state import (
        partition_functions_for_elements,
    )
    from payne_zero_synthesis.ground_partition_table import (
        ground_partition_value,
    )

    atmosphere_modes = compute_atmosphere_saha_modes(inputs)
    tables = load_synthesis_tables()
    atomic_numbers = np.asarray(inputs["saha_atomic_number"], np.int64)
    synthesis = partition_functions_for_elements(
        np.asarray(inputs["saha_temperature"], np.float64),
        np.asarray(inputs["saha_gas_pressure"], np.float64),
        np.asarray(inputs["saha_electron_density"], np.float64),
        tables=tables,
        elements=[int(value) for value in atomic_numbers],
        nion=6,
    )
    ground_temperature = np.array([3000.0], dtype=np.float64)
    ground_pressure = np.array([1.0e4], dtype=np.float64)
    ground_electron_density = np.array([1.0e13], dtype=np.float64)
    ground_enabled = partition_functions_for_elements(
        ground_temperature,
        ground_pressure,
        ground_electron_density,
        tables=load_synthesis_tables(),
        elements=[8],
        nion=6,
        apply_ground_partition=True,
    )[8]
    ground_disabled = partition_functions_for_elements(
        ground_temperature,
        ground_pressure,
        ground_electron_density,
        tables=load_synthesis_tables(),
        elements=[8],
        nion=6,
        apply_ground_partition=False,
    )[8]
    outputs = {
        "atomic_number": atomic_numbers.copy(),
        "branch_label": np.asarray(
            ["special ordered levels", "packed ordinary", "PFIRON"]
        ),
        "charge_square_density": np.asarray(
            inputs["saha_charge_square_density"], np.float64
        ).copy(),
        "electron_density": np.asarray(
            inputs["saha_electron_density"], np.float64
        ).copy(),
        "gas_pressure": np.asarray(
            inputs["saha_gas_pressure"], np.float64
        ).copy(),
        "temperature": np.asarray(
            inputs["saha_temperature"], np.float64
        ).copy(),
        "ground_floor_atomic_number": np.asarray(8, np.int64),
        "ground_floor_ion_stage": np.asarray(2, np.int64),
        "ground_floor_label": np.asarray(44, np.int64),
        "ground_floor_temperature": ground_temperature,
        "ground_floor_ordered_value": np.asarray(
            [ground_partition_value(44, float(ground_temperature[0]))],
            np.float64,
        ),
        "ground_floor_enabled_partition": ground_enabled[:, 1].copy(),
        "ground_floor_disabled_partition": ground_disabled[:, 1].copy(),
    }
    for atomic_number in atomic_numbers:
        key = f"z{int(atomic_number):02d}"
        outputs[f"atmosphere_{key}_partition"] = atmosphere_modes[
            f"{key}_mode13_batch"
        ]
        outputs[f"synthesis_{key}_partition"] = synthesis[
            int(atomic_number)
        ].copy()
    return outputs


def probe_pressure_lowering() -> dict[str, np.ndarray]:
    """Expose one controlled exact low/high-density partition-policy crossing."""

    configure_local_data_paths()
    from payne_zero_atmosphere.equation_of_state import (
        iron_group_partition_function,
        saha_partition_depth_batch,
    )
    from payne_zero_atmosphere.constants import (
        BOLTZMANN_ERG_PER_K_REFERENCE as ATMOSPHERE_BOLTZMANN,
        WAVENUMBER_PER_EV_REFERENCE,
    )
    from payne_zero_synthesis.constants import (
        REFERENCE_BOLTZMANN_ERG_PER_K as SYNTHESIS_BOLTZMANN,
    )
    from payne_zero_synthesis.equation_of_state import (
        _debye_lowering_np,
        partition_functions_for_elements,
    )

    temperature = np.array([30000.0, 30000.0], dtype=np.float64)
    electron_density = np.array([1.0e13, 1.0e20], dtype=np.float64)
    thermal_atmosphere = ATMOSPHERE_BOLTZMANN * temperature
    gas_pressure = 1.5 * electron_density * thermal_atmosphere
    particle_density = gas_pressure / thermal_atmosphere
    excess = 2.0 * electron_density - particle_density
    charge_square_density = 2.0 * electron_density
    charge_square_density = np.where(
        excess > 0.0,
        charge_square_density + 2.0 * excess,
        charge_square_density,
    )
    atmosphere_radius = np.sqrt(
        thermal_atmosphere
        / (
            12.5664
            * (4.801e-10) ** 2
            * np.maximum(charge_square_density, 1.0)
        )
    )
    atmosphere_lowering = np.minimum(1.0, 1.44e-7 / atmosphere_radius)
    atmosphere_h = saha_partition_depth_batch(
        temperature,
        electron_density,
        1,
        2,
        13,
        charge_square_density,
    )
    atmosphere_fe = saha_partition_depth_batch(
        temperature,
        electron_density,
        26,
        5,
        13,
        charge_square_density,
    )

    thermal_synthesis = SYNTHESIS_BOLTZMANN * temperature
    synthesis_lowering = _debye_lowering_np(
        electron_density,
        thermal_synthesis,
        gas_pressure,
    )
    synthesis_effective_density = 2.0 * electron_density
    synthesis_excess = (
        2.0 * electron_density - gas_pressure / thermal_synthesis
    )
    synthesis_effective_density = np.where(
        synthesis_excess > 0.0,
        synthesis_effective_density + 2.0 * synthesis_excess,
        synthesis_effective_density,
    )
    synthesis_radius = np.sqrt(
        thermal_synthesis
        / (2.8965e-18 * np.maximum(synthesis_effective_density, 1.0))
    )
    synthesis = partition_functions_for_elements(
        temperature,
        gas_pressure,
        electron_density,
        tables=load_synthesis_tables(),
        elements=[1, 26],
        nion=5,
    )
    pfiron_temperature = np.array([50000.0], dtype=np.float64)
    pfiron_electron_density = np.array([1.0e20], dtype=np.float64)
    pfiron_pressure = (
        1.5 * pfiron_electron_density * SYNTHESIS_BOLTZMANN
        * pfiron_temperature
    )
    pfiron_lowering_coordinate = 4.0 * WAVENUMBER_PER_EV_REFERENCE
    pfiron_atmosphere_partition = iron_group_partition_function(
        atomic_number=25,
        ion_stage=4,
        log10_temperature=float(np.log10(pfiron_temperature[0])),
        lowering_energy_cm=float(pfiron_lowering_coordinate),
    )
    pfiron_synthesis_partition = partition_functions_for_elements(
        pfiron_temperature,
        pfiron_pressure,
        pfiron_electron_density,
        tables=load_synthesis_tables(),
        elements=[25],
        nion=4,
    )[25][0, 3]
    return {
        "atmosphere_charge_square_density": charge_square_density,
        "atmosphere_debye_radius_cm": atmosphere_radius,
        "atmosphere_fe_partition": atmosphere_fe,
        "atmosphere_h_partition": atmosphere_h,
        "atmosphere_lowering_per_charge_ev": atmosphere_lowering,
        "atmosphere_fe_v_lowering_coordinate_cm": (
            atmosphere_lowering * 5.0 * WAVENUMBER_PER_EV_REFERENCE
        ),
        "atmosphere_policy_label": np.asarray(
            "explicit charge-square density; PFIRON clamps above final node"
        ),
        "electron_density": electron_density,
        "gas_pressure": gas_pressure,
        "pfiron_edge_atomic_number": np.asarray(25, np.int64),
        "pfiron_edge_ion_stage": np.asarray(4, np.int64),
        "pfiron_edge_temperature": pfiron_temperature,
        "pfiron_edge_lowering_coordinate_cm": np.asarray(
            [pfiron_lowering_coordinate], np.float64
        ),
        "pfiron_edge_atmosphere_partition": np.asarray(
            [pfiron_atmosphere_partition], np.float64
        ),
        "pfiron_edge_synthesis_partition": np.asarray(
            [pfiron_synthesis_partition], np.float64
        ),
        "synthesis_debye_radius_cm": synthesis_radius,
        "synthesis_effective_charge_density": synthesis_effective_density,
        "synthesis_fe_partition": synthesis[26].copy(),
        "synthesis_h_partition": synthesis[1].copy(),
        "synthesis_lowering_per_charge_ev": synthesis_lowering,
        "synthesis_fe_v_lowering_coordinate_cm": (
            synthesis_lowering * 5.0 * WAVENUMBER_PER_EV_REFERENCE
        ),
        "synthesis_policy_label": np.asarray(
            "pressure/electron proxy; PFIRON extrapolates above final node"
        ),
        "temperature": temperature,
    }


def _population_state_arrays(prefix: str, state) -> dict[str, np.ndarray]:
    """Flatten one exact synthesis PopulationState into comparison arrays."""

    import torch

    return {
        f"{prefix}_carbon_partition_normalized_ion_stage_populations": (
            state.carbon_partition_normalized_ion_stage_populations.copy()
        ),
        f"{prefix}_electron_density": state.electron_density.copy(),
        f"{prefix}_eos_ion_stage_fractions_over_partition": (
            state.eos.ion_stage_fractions_over_partition.detach()
            .cpu()
            .to(dtype=torch.float64)
            .numpy()
        ),
        f"{prefix}_eos_partition_functions": (
            state.eos.partition_functions.detach()
            .cpu()
            .to(dtype=torch.float64)
            .numpy()
        ),
        f"{prefix}_helium_neutral_population": (
            state.helium_neutral_population.copy()
        ),
        f"{prefix}_helium_singly_ionized_population": (
            state.helium_singly_ionized_population.copy()
        ),
        f"{prefix}_hydrogen_ionized_population": (
            state.hydrogen_ionized_population.copy()
        ),
        f"{prefix}_hydrogen_neutral_population": (
            state.hydrogen_neutral_population.copy()
        ),
        f"{prefix}_hydrogen_partition_normalized_ion_stage_populations": (
            state.hydrogen_partition_normalized_ion_stage_populations.copy()
        ),
        f"{prefix}_ion_stage_populations": (
            state.ion_stage_populations.copy()
        ),
        f"{prefix}_mass_density": state.mass_density.copy(),
        f"{prefix}_partition_normalized_populations": (
            state.partition_normalized_populations.copy()
        ),
        f"{prefix}_total_nuclei_number_density": (
            state.total_nuclei_number_density.copy()
        ),
    }


def compute_synthesis_atomic_states(
    inputs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run CPU-float64 full closure, then fixed-ne at that exact solution."""

    import torch

    return compute_synthesis_atomic_states_on_backend(
        inputs,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def compute_synthesis_atomic_states_on_backend(
    inputs: Mapping[str, np.ndarray],
    *,
    device,
    dtype,
) -> dict[str, np.ndarray]:
    """Run full then fixed atom-only closure on one declared Torch backend."""

    configure_local_data_paths()
    from payne_zero_synthesis.equation_of_state import (
        solve_population_state,
        solve_population_state_at_electron_density,
    )
    from payne_zero_synthesis.pipeline import (
        compute_doppler_per_ion,
        load_atomic_masses,
    )

    tables = load_synthesis_tables(device=device, dtype=dtype)
    temperature = np.asarray(inputs["temperature"], np.float64)
    gas_pressure = np.asarray(inputs["gas_pressure"], np.float64)
    abundance = np.asarray(inputs["elemental_abundances"], np.float64)
    full = solve_population_state(
        temperature,
        gas_pressure,
        abundance,
        tables=tables,
        electron_density_seed=np.asarray(
            inputs["electron_density_seed"], np.float64
        ),
        molecules=False,
    )
    fixed_input = full.electron_density.copy()
    fixed = solve_population_state_at_electron_density(
        temperature,
        gas_pressure,
        abundance,
        tables=tables,
        electron_density=fixed_input,
        molecules=False,
    )
    outputs = _population_state_arrays("full", full)
    outputs.update(_population_state_arrays("fixed", fixed))
    outputs["fixed_input_electron_density"] = fixed_input
    outputs["fractional_doppler_widths"] = compute_doppler_per_ion(
        temperature,
        np.asarray(inputs["microturbulence"], np.float64),
        load_atomic_masses(),
    )
    return outputs


def backend_parity_profile(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    resolved_floor_fraction: float = 1.0e-12,
) -> dict[str, float | int]:
    """Return scale-aware aggregate differences for one backend state.

    A direct elementwise relative error is uninformative where the reference
    stores exact zeros. We therefore report a resolved-value error above a
    declared fraction of each field's scale and, separately, leakage only at
    entries where the reference is exactly zero.
    """

    def numeric_field_names(values: Mapping[str, np.ndarray]) -> set[str]:
        return {
            name
            for name, value in values.items()
            if np.issubdtype(np.asarray(value).dtype, np.number)
        }

    reference_fields = numeric_field_names(reference)
    candidate_fields = numeric_field_names(candidate)
    if candidate_fields != reference_fields:
        missing = sorted(reference_fields - candidate_fields)
        unexpected = sorted(candidate_fields - reference_fields)
        raise ValueError(
            "backend numeric field set mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    maximum_absolute = 0.0
    scale_relative = 0.0
    resolved_relative = 0.0
    zero_leakage = 0.0
    for name in sorted(reference_fields):
        reference_array = np.asarray(reference[name], np.float64)
        candidate_array = np.asarray(candidate[name], np.float64)
        if reference_array.shape != candidate_array.shape:
            raise ValueError(f"{name}: backend shape mismatch")
        if not np.all(np.isfinite(reference_array)):
            raise ValueError(f"{name}: reference contains a nonfinite value")
        if not np.all(np.isfinite(candidate_array)):
            raise ValueError(f"{name}: backend contains a nonfinite value")
        difference = np.abs(candidate_array - reference_array)
        maximum_absolute = max(maximum_absolute, float(np.max(difference)))
        scale = max(float(np.max(np.abs(reference_array))), 1.0e-300)
        scale_relative = max(scale_relative, float(np.max(difference)) / scale)
        resolved = np.abs(reference_array) > scale * resolved_floor_fraction
        if np.any(resolved):
            resolved_relative = max(
                resolved_relative,
                float(
                    np.max(
                        difference[resolved] / np.abs(reference_array[resolved])
                    )
                ),
            )
        exact_zero = reference_array == 0.0
        if np.any(exact_zero):
            zero_leakage = max(
                zero_leakage,
                float(np.max(np.abs(candidate_array[exact_zero]))) / scale,
            )
    return {
        "field_count": len(reference_fields),
        "maximum_absolute": maximum_absolute,
        "scale_relative": scale_relative,
        "resolved_relative": resolved_relative,
        "zero_leakage": zero_leakage,
    }


def available_backend_parity_profiles(
    inputs: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    cpu_candidate: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    """Measure exact CPU and every available device under separate policies."""

    import torch

    rows: list[dict[str, object]] = []
    policies = (
        ("CPU", torch.device("cpu"), torch.float64, True, 0.0, 0.0),
        (
            "CUDA",
            torch.device("cuda"),
            torch.float64,
            torch.cuda.is_available(),
            5.0e-10,
            5.0e-12,
        ),
        (
            "MPS",
            torch.device("mps"),
            torch.float32,
            bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ),
            2.0e-5,
            2.0e-12,
        ),
    )
    for name, device, dtype, available, relative_limit, leak_limit in policies:
        row: dict[str, object] = {
            "backend": name,
            "dtype": str(dtype).split(".")[-1],
            "available": available,
            "relative_limit": relative_limit,
            "zero_leakage_limit": leak_limit,
        }
        if not available:
            row["status"] = "unavailable"
            rows.append(row)
            continue
        candidate = (
            cpu_candidate
            if name == "CPU"
            else compute_synthesis_atomic_states_on_backend(
                inputs,
                device=device,
                dtype=dtype,
            )
        )
        profile = backend_parity_profile(reference, candidate)
        row.update(profile)
        row["status"] = (
            "pass"
            if profile["scale_relative"] <= relative_limit
            and profile["resolved_relative"] <= relative_limit
            and profile["zero_leakage"] <= leak_limit
            else "FAIL"
        )
        rows.append(row)
    return rows
