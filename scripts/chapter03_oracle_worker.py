#!/usr/bin/env python3
"""Produce Chapter 3 comparison arrays with the pinned Payne Zero checkout."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np

from deterministic_npz import write_npz


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_ROOT = Path("/Users/ysting/payne-zero")
PINNED_DATA_ROOT = PINNED_ROOT / "source_data_files"
PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
ATMOSPHERE_TABLE_NAMES = (
    "special_partition_tables.npz",
    "iron_group_partition_tables.npz",
    "ionization_potential_tables.npz",
    "packed_level_metadata.npz",
    "isotope_tables.npz",
)


def parse_args() -> argparse.Namespace:
    """Parse the frozen local input and repository output directory."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_arrays(fixture_sha256: str) -> dict[str, np.ndarray]:
    """Return source, input, and exact table identities stored in each golden."""

    arrays = {
        "fixture_sha256": np.asarray(fixture_sha256),
        "payne_zero_commit": np.asarray(PAYNE_ZERO_COMMIT),
    }
    for name in ATMOSPHERE_TABLE_NAMES:
        key = name.removesuffix(".npz") + "_sha256"
        arrays[key] = np.asarray(
            sha256(PINNED_DATA_ROOT / "atmosphere_tables" / name)
        )
    arrays["synthesis_atomic_masses_sha256"] = np.asarray(
        sha256(PINNED_DATA_ROOT / "synthesis_tables" / "atomic_masses.npz")
    )
    arrays["synthesis_eos_bundle_sha256"] = np.asarray(
        sha256(
            PINNED_DATA_ROOT
            / "synthesis_tables"
            / "partition_saha_inputs.npz"
        )
    )
    return arrays


def configure_oracle_runtime(fixture: Path):
    """Import local orchestration with exact packages selected from the pin."""

    sys.path.insert(0, str(PINNED_ROOT))
    sys.path.insert(1, str(REPOSITORY_ROOT))
    from book import chapter03_runtime as runtime

    runtime.ATOM_ONLY_FIXTURE = fixture
    runtime.STATIC_DATA_ROOT = PINNED_DATA_ROOT
    runtime.SYNTHESIS_TABLES = (
        PINNED_DATA_ROOT
        / "synthesis_tables"
        / "partition_saha_inputs.npz"
    )
    runtime.SYNTHESIS_STATE_FIXTURE = (
        PINNED_DATA_ROOT
        / "synthesis_tables"
        / "partition_saha_inputs.npz"
    )
    runtime.ATOMIC_MASSES = (
        PINNED_DATA_ROOT / "synthesis_tables" / "atomic_masses.npz"
    )
    runtime.configure_local_data_paths()
    import payne_zero_atmosphere.equation_of_state as atmosphere_eos
    import payne_zero_synthesis.equation_of_state as synthesis_eos

    for module in (atmosphere_eos, synthesis_eos):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(PINNED_ROOT.resolve()):
            raise RuntimeError(
                "oracle module resolved outside pinned checkout: "
                f"{module_path}"
            )
    return runtime


def sentinel_bridge(runtime) -> dict[str, np.ndarray]:
    """Return exact interface-only packed sentinels after canonical mapping."""

    packed = np.arange(1, 1007, dtype=np.float64)[None, :]
    state = {
        "electron_density": np.array([7.0e12], dtype=np.float64),
        "fractional_doppler_widths": packed + 20000.0,
        "ion_stage_populations_by_packed_slot": packed,
        "partition_normalized_populations_by_packed_slot": packed + 10000.0,
    }
    return {
        f"sentinel_{name}": values
        for name, values in runtime.compute_packed_bridge(state).items()
    }


def main() -> None:
    """Compute local-first ordered oracle outputs and write deterministic NPZs."""

    arguments = parse_args()
    fixture = arguments.fixture.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_hash = sha256(fixture)
    runtime = configure_oracle_runtime(fixture)
    inputs = runtime.load_atom_only_fixture()
    identities = identity_arrays(fixture_hash)

    saha = runtime.compute_atmosphere_saha_modes(inputs)
    saha.update(identities)
    write_npz(output_dir / "chapter03_atmosphere_saha_outputs.npz", saha)

    atmosphere = runtime.compute_atmosphere_atomic_state(inputs)
    scalar = runtime.compute_atmosphere_atomic_state_by_depth(inputs)
    atmosphere.update(
        {f"scalar_depth_{name}": values for name, values in scalar.items()}
    )
    import numba

    atmosphere.update(identities)
    atmosphere["numba_cache_state"] = np.asarray(
        "fresh temporary cache; Saha modes evaluated before closure"
    )
    atmosphere["numba_thread_count"] = np.asarray(
        numba.get_num_threads(), dtype=np.int64
    )
    atmosphere["numba_version"] = np.asarray(numba.__version__)
    atmosphere_path = output_dir / "chapter03_atmosphere_atomic_state.npz"
    write_npz(atmosphere_path, atmosphere)

    physical_bridge = runtime.compute_atmosphere_fixed_handoff_state(
        inputs,
        atmosphere["electron_density"],
    )
    physical_bridge.update(sentinel_bridge(runtime))
    physical_bridge.update(identities)
    physical_bridge["atmosphere_state_golden_sha256"] = np.asarray(
        sha256(atmosphere_path)
    )
    write_npz(
        output_dir / "chapter03_packed_bridge_outputs.npz",
        physical_bridge,
    )

    synthesis = runtime.compute_synthesis_atomic_states(inputs)
    synthesis.update(identities)
    import torch

    synthesis["torch_device"] = np.asarray("cpu")
    synthesis["torch_dtype"] = np.asarray("float64")
    synthesis["torch_version"] = np.asarray(torch.__version__)
    write_npz(
        output_dir / "chapter03_synthesis_atomic_state_cpu_float64.npz",
        synthesis,
    )


if __name__ == "__main__":
    main()
