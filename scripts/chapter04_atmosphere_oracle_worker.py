#!/usr/bin/env python3
"""Build the unpublished Chapter 4 atmosphere oracle in memory.

This worker imports the pinned Payne Zero checkout, observes the exact
atmosphere molecular routes without changing their arguments or return
values, and returns NumPy arrays.  Golden publication is its only deferred
scope: it never writes or reads a golden product.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_ROOT = Path("/Users/ysting/payne-zero")
PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
FIXTURE_PATH = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter04_molecular_inputs.npz"
)
FIXTURE_SHA256 = "351bba75dca1fa502f5cc2a108035f69f2e31c760a90133480f2e7fe31e45f79"
REFERENCE_BOLTZMANN_ERG_PER_K = 1.38054e-16

PINNED_SOURCE_HASHES = {
    "atmosphere_io.py": (
        "95c4d2cab230f6925e9404639ecb05b25af8c0c85755ac1ca70d760156a8683e"
    ),
    "config.py": "51e19846fb81c832ae57334faf3da2c1e4fc2ef9edf6e08467ef7296e4640b45",
    "equation_of_state.py": (
        "719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2"
    ),
    "molecular_data.py": (
        "705c3072d79c8019c948ce0fa2c82052f232816d453e10a7c8e5fc5a8f5ce249"
    ),
    "molecular_equilibrium.py": (
        "4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86"
    ),
    "population_layout.py": (
        "36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0"
    ),
    "run_setup.py": "de7cf08b936585dbcfa2e572c026fafa3f10282a99c27b834b62db0f3f2888c9",
    "runner.py": "05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7",
    "runtime_state.py": (
        "fae240ec00f6f89d7c2a7ef721ce6e6539be234e523291fd6e8a096d731430e8"
    ),
    "synthesis_bridge.py": (
        "142a960b5e710823754b02766803b3c1dd8c48c9945fdfabe560b4ee7e1acb50"
    ),
}

PINNED_ASSET_HASHES = {
    "atmosphere_tables/iron_group_partition_tables.npz": (
        "137629dea64eca46f77ea3656c18305ade912a468d7eb27029544c0106cc3296"
    ),
    "atmosphere_tables/ionization_potential_tables.npz": (
        "82a2e82f2015da02c3d2bce77ca5337aa2b9c4e23d8d6219da07895896ca8a50"
    ),
    "atmosphere_tables/isotope_tables.npz": (
        "53c8d315fb53f1e051dc2752b028fc270d7c17a2c1042279c04ffcb750aef5c6"
    ),
    "atmosphere_tables/molecular_equilibrium_tables.npz": (
        "1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe"
    ),
    "atmosphere_tables/packed_level_metadata.npz": (
        "de5f17b6a9eaec1d1b07e96fd02ff014279cd8eaa9f976fefde0e2a153961bc3"
    ),
    "atmosphere_tables/special_partition_tables.npz": (
        "7d737524aacda1cc2281e5b18ff49f240ca34665dbe6c96d4dd0f39db4aedd22"
    ),
    "source_catalogs/lines/molecular_equilibrium_atmosphere.npz": (
        "971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644"
    ),
}

ONE_THREAD_ENVIRONMENT = {
    "LC_ALL": "C",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "TZ": "UTC",
    "VECLIB_MAXIMUM_THREADS": "1",
}

FIXTURE_KEYS = {
    "atmosphere_h2_boundary_temperature",
    "atmosphere_polynomial_boundary_temperature",
    "column_mass",
    "electron_density_seed",
    "elemental_abundances",
    "gas_pressure",
    "microturbulence",
    "named_molecule_codes",
    "pressure_control_gas_pressure",
    "pressure_control_temperature",
    "source_abundance_fixture_sha256",
    "synthesis_named_h2_boundary_temperature",
    "synthesis_polynomial_boundary_temperature",
    "temperature",
    "temperature_control_gas_pressure",
    "temperature_control_temperature",
}

DEFERRED_CAPTURE_FIELDS = (
    "golden_publication",
)


class OracleIdentityError(RuntimeError):
    """Raised when the worker cannot prove the exact source/input identity."""


class OracleEnvironmentError(RuntimeError):
    """Raised when the worker is not running under the one-thread contract."""


class IncompleteOracleScopeError(RuntimeError):
    """Raised when a caller requests publication from this in-memory worker."""


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pinned_root(pinned_root: Path = PINNED_ROOT) -> dict[str, str]:
    """Fail unless the exact read-only checkout and source/data hashes are present."""

    root = Path(pinned_root).expanduser().resolve()
    expected_root = PINNED_ROOT.resolve()
    if root != expected_root:
        raise OracleIdentityError(
            f"Chapter 4 permits only pinned root {expected_root}; received {root}"
        )
    if not root.is_dir():
        raise OracleIdentityError(f"pinned root does not exist: {root}")

    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != PINNED_COMMIT:
        raise OracleIdentityError(
            f"pinned checkout is {commit}; expected {PINNED_COMMIT}"
        )

    identities = {"payne_zero_commit": commit}
    source_root = root / "payne_zero_atmosphere"
    for name, expected_hash in PINNED_SOURCE_HASHES.items():
        actual_hash = sha256(source_root / name)
        if actual_hash != expected_hash:
            raise OracleIdentityError(f"pinned source hash changed: {name}")
        identities[f"source_{name.removesuffix('.py')}_sha256"] = actual_hash

    data_root = root / "source_data_files"
    for relative_name, expected_hash in PINNED_ASSET_HASHES.items():
        actual_hash = sha256(data_root / relative_name)
        if actual_hash != expected_hash:
            raise OracleIdentityError(f"pinned asset hash changed: {relative_name}")
        key = relative_name.replace("/", "_").removesuffix(".npz")
        identities[f"asset_{key}_sha256"] = actual_hash
    return identities


def load_input_fixture(path: Path = FIXTURE_PATH) -> dict[str, np.ndarray]:
    """Load and validate the input-only Chapter 4 fixture."""

    fixture_path = Path(path).expanduser().resolve()
    actual_hash = sha256(fixture_path)
    if actual_hash != FIXTURE_SHA256:
        raise OracleIdentityError(
            f"fixture hash is {actual_hash}; expected {FIXTURE_SHA256}"
        )
    with np.load(fixture_path, allow_pickle=False) as archive:
        if set(archive.files) != FIXTURE_KEYS:
            raise OracleIdentityError("Chapter 4 fixture key set changed")
        arrays = {
            name: np.asarray(archive[name]).copy() for name in sorted(archive.files)
        }

    depth_count = 6
    for name in (
        "column_mass",
        "electron_density_seed",
        "gas_pressure",
        "microturbulence",
        "temperature",
    ):
        values = arrays[name]
        if values.shape != (depth_count,) or values.dtype != np.float64:
            raise OracleIdentityError(f"fixture {name} must be float64[6]")
        if not np.all(np.isfinite(values)):
            raise OracleIdentityError(f"fixture {name} contains nonfinite values")
    for name in (
        "column_mass",
        "electron_density_seed",
        "gas_pressure",
        "temperature",
    ):
        if np.any(arrays[name] <= 0.0):
            raise OracleIdentityError(f"fixture {name} must be positive")
    if np.any(np.diff(arrays["column_mass"]) <= 0.0):
        raise OracleIdentityError("fixture column_mass must increase inward")

    abundances = arrays["elemental_abundances"]
    if (
        abundances.shape != (99,)
        or abundances.dtype != np.float64
        or not np.all(np.isfinite(abundances))
        or np.any(abundances < 0.0)
        or not float(np.sum(abundances)) > 0.0
    ):
        raise OracleIdentityError("fixture elemental_abundances contract failed")
    return arrays


def validate_one_thread_environment() -> dict[str, np.ndarray]:
    """Require and describe the fresh-process one-thread environment."""

    missing_or_wrong = {
        name: os.environ.get(name)
        for name, expected in ONE_THREAD_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if missing_or_wrong:
        raise OracleEnvironmentError(
            "one-thread oracle environment is incomplete: "
            + json.dumps(missing_or_wrong, sort_keys=True)
        )
    cache_value = os.environ.get("NUMBA_CACHE_DIR", "")
    if not cache_value:
        raise OracleEnvironmentError("NUMBA_CACHE_DIR must name a fresh directory")
    cache_path = Path(cache_value).expanduser().resolve()
    pinned_root = PINNED_ROOT.resolve()
    if cache_path == pinned_root or pinned_root in cache_path.parents:
        raise OracleEnvironmentError("NUMBA_CACHE_DIR must not be inside the pin")
    cache_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        f"environment_{name.lower()}": np.asarray(value)
        for name, value in sorted(ONE_THREAD_ENVIRONMENT.items())
    }
    metadata["environment_numba_cache_policy"] = np.asarray(
        "fresh external temporary directory"
    )
    metadata["environment_cpu_only"] = np.asarray(True, dtype=np.bool_)
    metadata["environment_numpy_float_dtype"] = np.asarray("float64")
    return metadata


def _is_under(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def _configure_pinned_modules(pinned_root: Path) -> SimpleNamespace:
    """Import all executed Payne Zero modules from the pin, never local ``src``."""

    root = Path(pinned_root).resolve()
    data_root = (root / "source_data_files").resolve()
    configured_data_root = os.environ.get("PAYNE_ZERO_DATA_ROOT")
    if configured_data_root is not None:
        if Path(configured_data_root).expanduser().resolve() != data_root:
            raise OracleIdentityError(
                "PAYNE_ZERO_DATA_ROOT points outside the pinned checkout"
            )
    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(data_root)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    for name, module in tuple(sys.modules.items()):
        if name != "payne_zero_atmosphere" and not name.startswith(
            "payne_zero_atmosphere."
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None and not _is_under(Path(module_file), root):
            raise OracleIdentityError(
                f"{name} was imported outside the pinned checkout: {module_file}"
            )

    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    modules = SimpleNamespace(
        atmosphere_io=importlib.import_module(
            "payne_zero_atmosphere.atmosphere_io"
        ),
        config=importlib.import_module("payne_zero_atmosphere.config"),
        equation_of_state=importlib.import_module(
            "payne_zero_atmosphere.equation_of_state"
        ),
        molecular_data=importlib.import_module(
            "payne_zero_atmosphere.molecular_data"
        ),
        molecular_equilibrium=importlib.import_module(
            "payne_zero_atmosphere.molecular_equilibrium"
        ),
        runner=importlib.import_module("payne_zero_atmosphere.runner"),
        runtime_state=importlib.import_module(
            "payne_zero_atmosphere.runtime_state"
        ),
        synthesis_bridge=importlib.import_module(
            "payne_zero_atmosphere.synthesis_bridge"
        ),
    )
    for module in vars(modules).values():
        module_path = Path(module.__file__).resolve()
        if not _is_under(module_path, root):
            raise OracleIdentityError(
                f"oracle module resolved outside pinned checkout: {module_path}"
            )

    import numba

    if int(numba.get_num_threads()) != 1:
        raise OracleEnvironmentError(
            f"Numba reports {numba.get_num_threads()} threads, expected 1"
        )
    modules.numba = numba
    return modules


def _abundance_deck(elemental_abundances: np.ndarray) -> dict[int, float]:
    """Encode the exact mixed linear/log atmosphere-deck abundance convention."""

    return {
        atomic_number: (
            float(elemental_abundances[atomic_number - 1])
            if atomic_number <= 2
            else float(np.log10(elemental_abundances[atomic_number - 1]))
        )
        for atomic_number in range(1, 100)
    }


def _build_atmosphere(
    modules: SimpleNamespace,
    *,
    column_mass: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
    microturbulence: np.ndarray,
    elemental_abundances: np.ndarray,
) -> Any:
    """Construct one exact structured atmosphere from declared fixture values."""

    temperature_array = np.asarray(temperature, dtype=np.float64)
    zeros = np.zeros_like(temperature_array)
    return modules.atmosphere_io.ModelAtmosphere(
        column_mass=np.asarray(column_mass, dtype=np.float64).copy(),
        temperature=temperature_array.copy(),
        gas_pressure=np.asarray(gas_pressure, dtype=np.float64).copy(),
        electron_density=np.asarray(electron_density, dtype=np.float64).copy(),
        rosseland_opacity=np.ones_like(temperature_array),
        radiative_acceleration=zeros.copy(),
        microturbulence=np.asarray(microturbulence, dtype=np.float64).copy(),
        convective_flux=zeros.copy(),
        convective_velocity=zeros.copy(),
        fixed_column_abundance_values=_abundance_deck(elemental_abundances),
    )


def _expected_charge_square_density(
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    electron_density: np.ndarray,
) -> np.ndarray:
    thermal_energy = np.asarray(temperature, np.float64) * (
        REFERENCE_BOLTZMANN_ERG_PER_K
    )
    electrons = np.asarray(electron_density, np.float64)
    pressure = np.asarray(gas_pressure, np.float64)
    charge_square = 2.0 * electrons
    excess = 2.0 * electrons - pressure / np.maximum(thermal_energy, 1.0e-300)
    charge_square = charge_square.copy()
    charge_square[excess > 0.0] += 2.0 * excess[excess > 0.0]
    return charge_square


class PopulationScheduleCapture(
    AbstractContextManager["PopulationScheduleCapture"]
):
    """Observe the real zero-code priming call and cached packed schedule."""

    def __init__(
        self,
        runner_module: Any,
        equation_of_state_module: Any,
        molecular_capture: "MolecularSolveCapture",
    ):
        self.runner = runner_module
        self.equation_of_state = equation_of_state_module
        self.molecular_capture = molecular_capture
        self.originals: dict[str, Any] = {}
        self.priming_records: list[dict[str, Any]] = []
        self.populate_all_records: list[dict[str, Any]] = []
        self.job_records: list[dict[str, Any]] = []

    def __enter__(self) -> "PopulationScheduleCapture":
        self.originals = {
            "runner_populate_species": self.runner.populate_species,
            "runner_populate_all_species": self.runner.populate_all_species,
            "eos_populate_species": self.equation_of_state.populate_species,
        }

        def runner_populate_species_wrapper(*args: Any, **kwargs: Any) -> None:
            cache = kwargs["temperature_iteration_cache"]
            record = {
                "cache_before": int(cache.get("pops_itemp", -1)),
                "code": float(kwargs["code"]),
                "mode": int(kwargs["population_mode"]),
                "output_shape": np.asarray(
                    kwargs["output"].shape, dtype=np.int64
                ),
                "solve_before": int(self.molecular_capture.solve_call_count),
                "temperature_iteration_index": int(
                    kwargs["temperature_iteration_index"]
                ),
            }
            result = self.originals["runner_populate_species"](*args, **kwargs)
            record["cache_after"] = int(cache.get("pops_itemp", -1))
            record["solve_after"] = int(self.molecular_capture.solve_call_count)
            self.priming_records.append(record)
            return result

        def runner_populate_all_species_wrapper(
            *args: Any, **kwargs: Any
        ) -> None:
            cache = kwargs["temperature_iteration_cache"]
            record = {
                "cache_before": int(cache.get("pops_itemp", -1)),
                "job_count_before": len(self.job_records),
                "solve_before": int(self.molecular_capture.solve_call_count),
                "temperature_iteration_index": int(
                    kwargs["temperature_iteration_index"]
                ),
            }
            result = self.originals["runner_populate_all_species"](
                *args, **kwargs
            )
            record["cache_after"] = int(cache.get("pops_itemp", -1))
            record["job_count_after"] = len(self.job_records)
            record["solve_after"] = int(self.molecular_capture.solve_call_count)
            self.populate_all_records.append(record)
            return result

        def eos_populate_species_wrapper(*args: Any, **kwargs: Any) -> None:
            state = kwargs["state"]
            output = np.asarray(kwargs["output"])
            targets = (
                state.partition_normalized_populations_by_packed_slot,
                state.ion_stage_populations_by_packed_slot,
            )
            matching_targets = [
                index
                for index, target in enumerate(targets)
                if np.shares_memory(output, target)
            ]
            if len(matching_targets) != 1:
                raise RuntimeError("packed population target ownership changed")
            target_index = matching_targets[0]
            target = np.asarray(targets[target_index])
            byte_offset = int(output.ctypes.data - target.ctypes.data)
            if byte_offset < 0 or byte_offset % target.itemsize != 0:
                raise RuntimeError("packed population destination is misaligned")
            start_slot = byte_offset // target.itemsize
            cache = kwargs["temperature_iteration_cache"]
            record = {
                "cache_before": int(cache.get("pops_itemp", -1)),
                "code": float(kwargs["code"]),
                "mode": int(kwargs["population_mode"]),
                "output_slots": int(output.shape[1]),
                "solve_before": int(self.molecular_capture.solve_call_count),
                "start_slot": int(start_slot),
                "target_index": int(target_index),
                "temperature_iteration_index": int(
                    kwargs["temperature_iteration_index"]
                ),
            }
            result = self.originals["eos_populate_species"](*args, **kwargs)
            record["cache_after"] = int(cache.get("pops_itemp", -1))
            record["solve_after"] = int(self.molecular_capture.solve_call_count)
            self.job_records.append(record)
            return result

        wrappers = {
            "runner_populate_species": runner_populate_species_wrapper,
            "runner_populate_all_species": runner_populate_all_species_wrapper,
            "eos_populate_species": eos_populate_species_wrapper,
        }
        for wrapper in wrappers.values():
            setattr(wrapper, "__chapter04_capture_wrapper__", True)
        self.runner.populate_species = runner_populate_species_wrapper
        self.runner.populate_all_species = runner_populate_all_species_wrapper
        self.equation_of_state.populate_species = eos_populate_species_wrapper
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.runner.populate_species = self.originals["runner_populate_species"]
        self.runner.populate_all_species = self.originals[
            "runner_populate_all_species"
        ]
        self.equation_of_state.populate_species = self.originals[
            "eos_populate_species"
        ]
        self.originals.clear()
        return None

    def arrays(self) -> dict[str, np.ndarray]:
        """Return ownership and schedule inventory as deliberately separate keys."""

        if (
            len(self.priming_records) != 1
            or len(self.populate_all_records) != 1
            or len(self.job_records) != 230
        ):
            raise RuntimeError("zero-code priming or packed schedule count changed")
        priming = self.priming_records[0]
        populate_all = self.populate_all_records[0]
        codes = np.asarray(
            [record["code"] for record in self.job_records], dtype=np.float64
        )
        modes = np.asarray(
            [record["mode"] for record in self.job_records], dtype=np.int64
        )
        start_slots = np.asarray(
            [record["start_slot"] for record in self.job_records],
            dtype=np.int64,
        )
        output_slots = np.asarray(
            [record["output_slots"] for record in self.job_records],
            dtype=np.int64,
        )
        target_indices = np.asarray(
            [record["target_index"] for record in self.job_records],
            dtype=np.int64,
        )
        cache_before = np.asarray(
            [record["cache_before"] for record in self.job_records],
            dtype=np.int64,
        )
        cache_after = np.asarray(
            [record["cache_after"] for record in self.job_records],
            dtype=np.int64,
        )
        solve_before = np.asarray(
            [record["solve_before"] for record in self.job_records],
            dtype=np.int64,
        )
        solve_after = np.asarray(
            [record["solve_after"] for record in self.job_records],
            dtype=np.int64,
        )
        temperature_indices = np.asarray(
            [
                record["temperature_iteration_index"]
                for record in self.job_records
            ],
            dtype=np.int64,
        )
        molecular_mask = codes >= 100.0
        atomic_mask = np.logical_not(molecular_mask)
        molecular_codes = codes[molecular_mask]
        unique_molecular_codes = molecular_codes[::2]
        return {
            "priming_cache_additional_solve_count_during_schedule": np.asarray(
                int(populate_all["solve_after"])
                - int(populate_all["solve_before"]),
                dtype=np.int64,
            ),
            "priming_cache_cache_value_after_priming": np.asarray(
                priming["cache_after"], dtype=np.int64
            ),
            "priming_cache_cache_value_after_schedule": np.asarray(
                populate_all["cache_after"], dtype=np.int64
            ),
            "priming_cache_cache_value_before_priming": np.asarray(
                priming["cache_before"], dtype=np.int64
            ),
            "priming_cache_cache_value_before_schedule": np.asarray(
                populate_all["cache_before"], dtype=np.int64
            ),
            "priming_cache_job_count_during_schedule": np.asarray(
                int(populate_all["job_count_after"])
                - int(populate_all["job_count_before"]),
                dtype=np.int64,
            ),
            "priming_cache_molecular_solve_count_during_zero_code": np.asarray(
                int(priming["solve_after"]) - int(priming["solve_before"]),
                dtype=np.int64,
            ),
            "priming_cache_populate_all_call_count": np.asarray(
                1, dtype=np.int64
            ),
            "priming_cache_schedule_solve_count_after": np.asarray(
                populate_all["solve_after"], dtype=np.int64
            ),
            "priming_cache_schedule_solve_count_before": np.asarray(
                populate_all["solve_before"], dtype=np.int64
            ),
            "priming_cache_schedule_job_cache_value_after": cache_after,
            "priming_cache_schedule_job_cache_value_before": cache_before,
            "priming_cache_schedule_job_solve_count_after": solve_after,
            "priming_cache_schedule_job_solve_count_before": solve_before,
            "priming_cache_schedule_job_temperature_iteration_index": (
                temperature_indices
            ),
            "priming_cache_temperature_iteration_index": np.asarray(
                priming["temperature_iteration_index"], dtype=np.int64
            ),
            "priming_cache_zero_code": np.asarray(
                priming["code"], dtype=np.float64
            ),
            "priming_cache_zero_code_call_count": np.asarray(
                1, dtype=np.int64
            ),
            "priming_cache_zero_code_mode": np.asarray(
                priming["mode"], dtype=np.int64
            ),
            "priming_cache_zero_code_output_shape": np.asarray(
                priming["output_shape"], dtype=np.int64
            ),
            "schedule_inventory_atomic_job_count": np.asarray(
                np.count_nonzero(atomic_mask), dtype=np.int64
            ),
            "schedule_inventory_atomic_job_mask": atomic_mask,
            "schedule_inventory_code": codes,
            "schedule_inventory_job_count": np.asarray(
                codes.size, dtype=np.int64
            ),
            "schedule_inventory_mode": modes,
            "schedule_inventory_molecular_job_count": np.asarray(
                np.count_nonzero(molecular_mask), dtype=np.int64
            ),
            "schedule_inventory_molecular_job_mask": molecular_mask,
            "schedule_inventory_molecular_unique_code": (
                unique_molecular_codes.copy()
            ),
            "schedule_inventory_molecular_unique_code_count": np.asarray(
                unique_molecular_codes.size, dtype=np.int64
            ),
            "schedule_inventory_molecular_unique_mode_pair": modes[
                molecular_mask
            ].reshape(-1, 2),
            (
                "schedule_inventory_"
                "molecular_unique_packed_start_slot_one_based"
            ): (start_slots[molecular_mask][::2] + 1),
            "schedule_inventory_output_slots": output_slots,
            "schedule_inventory_packed_start_slot_one_based": (
                start_slots + 1
            ),
            "schedule_inventory_packed_start_slot_zero_based": start_slots,
            "schedule_inventory_target_index": target_indices,
            "schedule_inventory_target_name": np.asarray(
                [
                    (
                        "partition_normalized_populations_by_packed_slot"
                        if index == 0
                        else "ion_stage_populations_by_packed_slot"
                    )
                    for index in target_indices
                ],
                dtype=np.str_,
            ),
        }


class MolecularSolveCapture(AbstractContextManager["MolecularSolveCapture"]):
    """Observe one exact molecular solve without changing its numerical calls."""

    def __init__(self, molecular_module: Any):
        self.module = molecular_module
        self.originals: dict[str, Any] = {}
        self.solve_call_count = 0
        self.solve_population_modes: list[int] = []
        self.current_solve: int | None = None
        self.current_layer: int | None = None
        self.layer_order: list[tuple[int, int]] = []
        self.layer_seeds: dict[tuple[int, int], np.ndarray] = {}
        self.frozen_constants: dict[tuple[int, int], np.ndarray] = {}
        self.population_constants: dict[tuple[int, int], np.ndarray] = {}
        self.relative_updates: dict[tuple[int, int], list[float]] = {}
        self.still_iterating: dict[tuple[int, int], list[bool]] = {}
        self.preupdate_densities: dict[
            tuple[int, int], list[np.ndarray]
        ] = {}
        self.raw_corrections: dict[tuple[int, int], list[np.ndarray]] = {}
        self.linear_solve_attempts: dict[tuple[int, int], int] = {}
        self.linear_lstsq_calls: dict[tuple[int, int], int] = {}
        self.linear_branches: dict[tuple[int, int], list[int]] = {}
        self.residuals: dict[tuple[int, int], np.ndarray] = {}
        self.scaled_residuals: dict[tuple[int, int], np.ndarray] = {}
        self.fill_before: list[dict[str, np.ndarray]] = []
        self.fill_after: list[dict[str, np.ndarray]] = []
        self.linalg_originals: dict[str, Any] = {}

    def __enter__(self) -> "MolecularSolveCapture":
        names = (
            "solve_molecular_equilibrium",
            "solve_molecular_equilibrium_layer",
            "_compute_equilibrium_constants_for_layer_compiled",
            "_newton_update_kernel",
            "_fill_partition_normalized_molecular_densities",
        )
        self.originals = {name: getattr(self.module, name) for name in names}
        self.linalg_originals = {
            "solve": np.linalg.solve,
            "lstsq": np.linalg.lstsq,
        }

        def linalg_solve_wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
            if self.current_solve is None or self.current_layer is None:
                return self.linalg_originals["solve"](*args, **kwargs)
            key = (self.current_solve, self.current_layer)
            self.linear_solve_attempts[key] = (
                self.linear_solve_attempts.get(key, 0) + 1
            )
            result = self.linalg_originals["solve"](*args, **kwargs)
            self.linear_branches.setdefault(key, []).append(0)
            return result

        def linalg_lstsq_wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.current_solve is None or self.current_layer is None:
                return self.linalg_originals["lstsq"](*args, **kwargs)
            key = (self.current_solve, self.current_layer)
            self.linear_lstsq_calls[key] = (
                self.linear_lstsq_calls.get(key, 0) + 1
            )
            result = self.linalg_originals["lstsq"](*args, **kwargs)
            self.linear_branches.setdefault(key, []).append(1)
            return result

        def solve_wrapper(molecular_state: Any, *, population_mode: int = 1) -> None:
            if self.current_solve is not None:
                raise RuntimeError("nested molecular solve is not supported")
            solve_index = self.solve_call_count
            self.solve_call_count += 1
            self.solve_population_modes.append(int(population_mode))
            self.current_solve = solve_index
            try:
                return self.originals["solve_molecular_equilibrium"](
                    molecular_state,
                    population_mode=population_mode,
                )
            finally:
                self.current_solve = None

        def layer_wrapper(
            molecular_state: Any,
            layer_index: int,
            equation_density_seed: np.ndarray,
        ) -> np.ndarray:
            if self.current_solve is None or self.current_layer is not None:
                raise RuntimeError("invalid molecular layer capture state")
            layer = int(layer_index)
            key = (self.current_solve, layer)
            self.current_layer = layer
            self.layer_order.append(key)
            self.layer_seeds[key] = np.asarray(
                equation_density_seed, dtype=np.float64
            ).copy()
            try:
                result = self.originals["solve_molecular_equilibrium_layer"](
                    molecular_state,
                    layer,
                    equation_density_seed,
                )
                self._capture_residual(molecular_state, layer, result, key)
                return result
            finally:
                self.current_layer = None

        def constants_wrapper(
            molecular_state: Any,
            layer_index: int,
        ) -> np.ndarray:
            values = self.originals[
                "_compute_equilibrium_constants_for_layer_compiled"
            ](molecular_state, layer_index)
            if self.current_solve is not None:
                key = (self.current_solve, int(layer_index))
                destination = (
                    self.frozen_constants
                    if self.current_layer is not None
                    else self.population_constants
                )
                if key in destination:
                    raise RuntimeError("duplicate equilibrium-constant capture")
                destination[key] = np.asarray(values, dtype=np.float64).copy()
            return values

        def update_wrapper(
            equation_count: int,
            equation_density: np.ndarray,
            previous_delta: np.ndarray,
            delta: np.ndarray,
        ) -> bool:
            if self.current_solve is None or self.current_layer is None:
                raise RuntimeError("Newton update occurred outside a captured layer")
            key = (self.current_solve, self.current_layer)
            density_before = np.asarray(equation_density, np.float64).copy()
            delta_before = np.asarray(delta, np.float64).copy()
            ratio = float(
                np.max(
                    np.abs(delta_before)
                    / np.maximum(np.abs(density_before), 1.0e-300)
                )
            )
            result = self.originals["_newton_update_kernel"](
                equation_count,
                equation_density,
                previous_delta,
                delta,
            )
            self.preupdate_densities.setdefault(key, []).append(density_before)
            self.raw_corrections.setdefault(key, []).append(delta_before)
            self.relative_updates.setdefault(key, []).append(ratio)
            self.still_iterating.setdefault(key, []).append(bool(result))
            return result

        def fill_wrapper(molecular_state: Any) -> None:
            self.fill_before.append(
                {
                    "equation_densities": (
                        molecular_state.molecular_equation_densities.copy()
                    ),
                    "previous_equation_densities": (
                        molecular_state.previous_molecular_equation_densities.copy()
                    ),
                    "raw_populations": molecular_state.molecular_populations.copy(),
                    "normalized_populations": (
                        molecular_state
                        .partition_normalized_molecular_populations
                        .copy()
                    ),
                }
            )
            result = self.originals[
                "_fill_partition_normalized_molecular_densities"
            ](molecular_state)
            self.fill_after.append(
                {
                    "equation_densities": (
                        molecular_state.molecular_equation_densities.copy()
                    ),
                    "normalized_populations": (
                        molecular_state
                        .partition_normalized_molecular_populations
                        .copy()
                    ),
                }
            )
            return result

        wrappers = {
            "solve_molecular_equilibrium": solve_wrapper,
            "solve_molecular_equilibrium_layer": layer_wrapper,
            "_compute_equilibrium_constants_for_layer_compiled": constants_wrapper,
            "_newton_update_kernel": update_wrapper,
            "_fill_partition_normalized_molecular_densities": fill_wrapper,
        }
        for name, wrapper in wrappers.items():
            setattr(wrapper, "__chapter04_capture_wrapper__", True)
            setattr(self.module, name, wrapper)
        setattr(linalg_solve_wrapper, "__chapter04_capture_wrapper__", True)
        setattr(linalg_lstsq_wrapper, "__chapter04_capture_wrapper__", True)
        np.linalg.solve = linalg_solve_wrapper
        np.linalg.lstsq = linalg_lstsq_wrapper
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for name, original in self.originals.items():
            setattr(self.module, name, original)
        for name, original in self.linalg_originals.items():
            setattr(np.linalg, name, original)
        self.originals.clear()
        self.linalg_originals.clear()
        return None

    def _capture_residual(
        self,
        molecular_state: Any,
        layer_index: int,
        equation_density: np.ndarray,
        key: tuple[int, int],
    ) -> None:
        """Re-evaluate the exact residual once at the returned physical density."""

        constants = self.frozen_constants.get(key)
        if constants is None:
            raise RuntimeError("frozen constants were not observed")
        catalog = molecular_state.catalog
        cache = self.module._catalog_kernel_cache(catalog)
        abundance = self.module._abundance_vector_for_layer(
            molecular_state, layer_index
        )
        temperature = float(molecular_state.temperature_k[layer_index])
        particle_density = float(
            molecular_state.gas_pressure[layer_index]
            / max(temperature * REFERENCE_BOLTZMANN_ERG_PER_K, 1.0e-300)
        )
        _, residual = self.module._newton_matrix_kernel(
            int(catalog.equation_count),
            int(catalog.molecule_count),
            np.asarray(equation_density, dtype=np.float64),
            abundance,
            constants,
            -particle_density,
            cache.equation_species_codes,
            cache.component_start_indices,
            cache.component_equation_indices,
        )
        scale = np.maximum(
            np.abs(abundance * float(equation_density[0])),
            1.0e-300,
        )
        scale[0] = max(abs(particle_density), 1.0e-300)
        if int(catalog.equation_species_codes[catalog.equation_count - 1]) == 100:
            scale[catalog.equation_count - 1] = max(
                abs(float(equation_density[catalog.equation_count - 1])),
                1.0e-300,
            )
        self.residuals[key] = np.asarray(residual, dtype=np.float64).copy()
        self.scaled_residuals[key] = (
            np.abs(np.asarray(residual, dtype=np.float64)) / scale
        )

    def trace_arrays(
        self,
        *,
        depth_count: int,
        molecule_count: int,
        equation_count: int,
        expected_population_mode: int,
        expected_fill_call_count: int,
    ) -> dict[str, np.ndarray]:
        """Return the common ordered trace for one exact molecular solve."""

        expected_keys = [(0, layer) for layer in range(depth_count)]
        if self.solve_call_count != 1 or self.layer_order != expected_keys:
            raise RuntimeError(
                "capture expected one ordered solve over every declared depth"
            )
        if self.solve_population_modes != [int(expected_population_mode)]:
            raise RuntimeError("captured molecular population mode changed")
        for mapping in (
            self.layer_seeds,
            self.frozen_constants,
            self.population_constants,
            self.preupdate_densities,
            self.raw_corrections,
            self.relative_updates,
            self.still_iterating,
            self.linear_solve_attempts,
            self.linear_branches,
            self.residuals,
            self.scaled_residuals,
        ):
            if list(mapping) != expected_keys:
                raise RuntimeError("captured molecular layer key set is incomplete")
        if (
            len(self.fill_before) != int(expected_fill_call_count)
            or len(self.fill_after) != int(expected_fill_call_count)
        ):
            raise RuntimeError("molecular normalized-fill call count changed")

        iteration_count = np.asarray(
            [len(self.relative_updates[key]) for key in expected_keys],
            dtype=np.int64,
        )
        final_still_iterating = np.asarray(
            [self.still_iterating[key][-1] for key in expected_keys],
            dtype=np.bool_,
        )
        history_offsets = np.zeros(depth_count + 1, dtype=np.int64)
        history_offsets[1:] = np.cumsum(iteration_count)
        history_layer_index = np.repeat(
            np.arange(depth_count, dtype=np.int64),
            iteration_count,
        )
        history_iteration_index = np.concatenate(
            [
                np.arange(int(count), dtype=np.int64)
                for count in iteration_count
            ]
        )
        preupdate_densities = np.concatenate(
            [self.preupdate_densities[key] for key in expected_keys],
            axis=0,
        ).reshape(-1, equation_count)
        raw_corrections = np.concatenate(
            [self.raw_corrections[key] for key in expected_keys],
            axis=0,
        ).reshape(-1, equation_count)
        relative_update_history = np.concatenate(
            [
                np.asarray(self.relative_updates[key], dtype=np.float64)
                for key in expected_keys
            ]
        )
        still_iterating_history = np.concatenate(
            [
                np.asarray(self.still_iterating[key], dtype=np.bool_)
                for key in expected_keys
            ]
        )
        linear_branch_history = np.concatenate(
            [
                np.asarray(self.linear_branches[key], dtype=np.int8)
                for key in expected_keys
            ]
        )
        solve_calls_by_layer = np.asarray(
            [self.linear_solve_attempts[key] for key in expected_keys],
            dtype=np.int64,
        )
        lstsq_calls_by_layer = np.asarray(
            [self.linear_lstsq_calls.get(key, 0) for key in expected_keys],
            dtype=np.int64,
        )
        if not np.array_equal(solve_calls_by_layer, iteration_count):
            raise RuntimeError("np.linalg.solve call count changed")
        if linear_branch_history.size != int(np.sum(iteration_count)):
            raise RuntimeError("linear-solver branch history is incomplete")
        if int(np.count_nonzero(linear_branch_history == 1)) != int(
            np.sum(lstsq_calls_by_layer)
        ):
            raise RuntimeError("np.linalg.lstsq branch history is inconsistent")
        return {
            "fill_call_count": np.asarray(
                expected_fill_call_count, dtype=np.int64
            ),
            "frozen_newton_constants": np.stack(
                [self.frozen_constants[key] for key in expected_keys]
            ).reshape(depth_count, molecule_count),
            "layer_order": np.asarray(
                [layer for _, layer in expected_keys], dtype=np.int64
            ),
            "layer_seed_equation_densities": np.stack(
                [self.layer_seeds[key] for key in expected_keys]
            ).reshape(depth_count, equation_count),
            "newton_converged": np.logical_not(final_still_iterating),
            "newton_exhausted": np.logical_and(
                iteration_count == 200,
                final_still_iterating,
            ),
            "newton_iteration_count": iteration_count,
            "newton_history_iteration_index": history_iteration_index,
            "newton_history_layer_index": history_layer_index,
            "newton_history_layer_offsets": history_offsets,
            "newton_linear_direct_solve_branch_mask": (
                linear_branch_history == 0
            ),
            "newton_linear_lstsq_branch_mask": linear_branch_history == 1,
            "newton_np_linalg_direct_solve_branch_count": np.asarray(
                np.count_nonzero(linear_branch_history == 0),
                dtype=np.int64,
            ),
            "newton_np_linalg_lstsq_call_count": np.asarray(
                np.sum(lstsq_calls_by_layer), dtype=np.int64
            ),
            "newton_np_linalg_lstsq_call_count_by_layer": (
                lstsq_calls_by_layer
            ),
            "newton_np_linalg_solve_call_count": np.asarray(
                np.sum(solve_calls_by_layer), dtype=np.int64
            ),
            "newton_np_linalg_solve_call_count_by_layer": (
                solve_calls_by_layer
            ),
            "newton_preupdate_equation_densities": preupdate_densities,
            "newton_raw_corrections": raw_corrections,
            "newton_relative_update_max_history": relative_update_history,
            "newton_still_iterating_history": still_iterating_history,
            "population_mode": np.asarray(
                expected_population_mode, dtype=np.int64
            ),
            "population_constants": np.stack(
                [self.population_constants[key] for key in expected_keys]
            ).reshape(depth_count, molecule_count),
            "postsolve_residual": np.stack(
                [self.residuals[key] for key in expected_keys]
            ).reshape(depth_count, equation_count),
            "postsolve_row_scaled_residual": np.stack(
                [self.scaled_residuals[key] for key in expected_keys]
            ).reshape(depth_count, equation_count),
            "postsolve_row_scaled_residual_max": np.asarray(
                [
                    np.max(self.scaled_residuals[key])
                    for key in expected_keys
                ],
                dtype=np.float64,
            ),
            "solve_call_count": np.asarray(1, dtype=np.int64),
            "stopping_relative_update_max": np.asarray(
                [self.relative_updates[key][-1] for key in expected_keys],
                dtype=np.float64,
            ),
        }

    def arrays(
        self,
        *,
        depth_count: int,
        molecule_count: int,
        equation_count: int,
    ) -> dict[str, np.ndarray]:
        """Return the ordered trace and mode-1 normalized-fill lifecycle."""

        arrays = self.trace_arrays(
            depth_count=depth_count,
            molecule_count=molecule_count,
            equation_count=equation_count,
            expected_population_mode=1,
            expected_fill_call_count=1,
        )
        arrays.update(
            {
                "equation_densities_before_normalization": self.fill_before[0][
                    "equation_densities"
                ],
                "equation_densities_after_normalization": self.fill_after[0][
                    "equation_densities"
                ],
                "normalized_populations_after_fill": self.fill_after[0][
                    "normalized_populations"
                ],
                "normalized_populations_before_fill": self.fill_before[0][
                    "normalized_populations"
                ],
                "physical_equation_densities_saved_before_fill": self.fill_before[0][
                    "previous_equation_densities"
                ],
                "raw_populations_before_fill": self.fill_before[0][
                    "raw_populations"
                ],
            }
        )
        return arrays


def _prefix(
    destination: dict[str, np.ndarray],
    prefix: str,
    arrays: Mapping[str, np.ndarray],
) -> None:
    for name, values in arrays.items():
        destination[f"{prefix}_{name}"] = np.asarray(values).copy()


def _initialize_molecular_state(
    modules: SimpleNamespace,
    catalog: Any,
    atmosphere: Any,
) -> tuple[Any, Any]:
    """Build one fresh runtime and molecule state without solving either."""

    runtime_state = modules.runtime_state.build_runtime_state(atmosphere)
    modules.runtime_state.update_charge_square_density(
        thermal_energy_erg=atmosphere.thermal_energy_erg,
        state=runtime_state,
    )
    molecular_state = (
        modules.molecular_equilibrium.initialize_molecular_equilibrium_state(
            temperature_k=atmosphere.temperature,
            thermal_energy_erg=atmosphere.thermal_energy_erg,
            gas_pressure=runtime_state.gas_pressure,
            runtime_state=runtime_state,
            catalog=catalog,
        )
    )
    return runtime_state, molecular_state


def _one_layer_mode_probe(
    modules: SimpleNamespace,
    catalog: Any,
    fixture: Mapping[str, np.ndarray],
    *,
    population_mode: int,
) -> dict[str, np.ndarray]:
    """Run one independent 3500 K, 1e5 dyn cm^-2 lifecycle-mode probe."""

    temperature = float(fixture["temperature_control_temperature"][0])
    gas_pressure = float(fixture["temperature_control_gas_pressure"][0])
    electron_density = (
        0.1
        * gas_pressure
        / (REFERENCE_BOLTZMANN_ERG_PER_K * temperature)
    )
    atmosphere = _build_atmosphere(
        modules,
        column_mass=np.asarray([fixture["column_mass"][0]], dtype=np.float64),
        temperature=np.asarray([temperature], dtype=np.float64),
        gas_pressure=np.asarray([gas_pressure], dtype=np.float64),
        electron_density=np.asarray([electron_density], dtype=np.float64),
        microturbulence=np.asarray(
            [fixture["microturbulence"][0]], dtype=np.float64
        ),
        elemental_abundances=fixture["elemental_abundances"],
    )
    runtime_state, molecular_state = _initialize_molecular_state(
        modules, catalog, atmosphere
    )
    previous_before = (
        molecular_state.previous_molecular_equation_densities.copy()
    )
    normalized_before = (
        molecular_state.partition_normalized_molecular_populations.copy()
    )
    energy_before = runtime_state.specific_internal_energy.copy()
    with MolecularSolveCapture(modules.molecular_equilibrium) as capture:
        modules.molecular_equilibrium.solve_molecular_equilibrium(
            molecular_state,
            population_mode=int(population_mode),
        )
    arrays = capture.trace_arrays(
        depth_count=1,
        molecule_count=int(catalog.molecule_count),
        equation_count=int(catalog.equation_count),
        expected_population_mode=int(population_mode),
        expected_fill_call_count=0,
    )
    arrays.update(
        {
            "molecular_equation_densities_after": (
                molecular_state.molecular_equation_densities.copy()
            ),
            "normalized_populations_after": (
                molecular_state
                .partition_normalized_molecular_populations
                .copy()
            ),
            "normalized_populations_before": normalized_before,
            "previous_equation_densities_after": (
                molecular_state
                .previous_molecular_equation_densities
                .copy()
            ),
            "previous_equation_densities_before": previous_before,
            "raw_populations_after": (
                molecular_state.molecular_populations.copy()
            ),
            "specific_internal_energy_after": (
                runtime_state.specific_internal_energy.copy()
            ),
            "specific_internal_energy_before": energy_before,
        }
    )
    return arrays


def _direct_probe(
    modules: SimpleNamespace,
    catalog: Any,
    fixture: Mapping[str, np.ndarray],
    *,
    temperature: float,
    gas_pressure: float,
) -> tuple[Any, dict[str, np.ndarray]]:
    """Run one independent one-layer mode-1 molecular solve."""

    electron_density = (
        0.1
        * float(gas_pressure)
        / (REFERENCE_BOLTZMANN_ERG_PER_K * float(temperature))
    )
    atmosphere = _build_atmosphere(
        modules,
        column_mass=np.asarray([fixture["column_mass"][0]], dtype=np.float64),
        temperature=np.asarray([temperature], dtype=np.float64),
        gas_pressure=np.asarray([gas_pressure], dtype=np.float64),
        electron_density=np.asarray([electron_density], dtype=np.float64),
        microturbulence=np.asarray(
            [fixture["microturbulence"][0]], dtype=np.float64
        ),
        elemental_abundances=fixture["elemental_abundances"],
    )
    _, molecular_state = _initialize_molecular_state(
        modules, catalog, atmosphere
    )
    with MolecularSolveCapture(modules.molecular_equilibrium) as capture:
        modules.molecular_equilibrium.solve_molecular_equilibrium(
            molecular_state,
            population_mode=1,
        )
    arrays = capture.arrays(
        depth_count=1,
        molecule_count=int(catalog.molecule_count),
        equation_count=int(catalog.equation_count),
    )
    return molecular_state, arrays


def _stack_control_probes(
    modules: SimpleNamespace,
    catalog: Any,
    fixture: Mapping[str, np.ndarray],
    *,
    temperatures: np.ndarray,
    gas_pressures: np.ndarray,
    named_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run independent one-layer controls and stack them after each solve."""

    records: list[dict[str, np.ndarray]] = []
    histories: list[dict[str, np.ndarray]] = []
    for temperature, gas_pressure in zip(temperatures, gas_pressures):
        molecular_state, capture = _direct_probe(
            modules,
            catalog,
            fixture,
            temperature=float(temperature),
            gas_pressure=float(gas_pressure),
        )
        histories.append(capture)
        records.append(
            {
                "frozen_newton_constants": capture[
                    "frozen_newton_constants"
                ][0],
                "newton_converged": capture["newton_converged"][0],
                "newton_exhausted": capture["newton_exhausted"][0],
                "newton_iteration_count": capture[
                    "newton_iteration_count"
                ][0],
                "newton_np_linalg_direct_solve_branch_count": capture[
                    "newton_np_linalg_direct_solve_branch_count"
                ],
                "newton_np_linalg_lstsq_call_count": capture[
                    "newton_np_linalg_lstsq_call_count"
                ],
                "newton_np_linalg_solve_call_count": capture[
                    "newton_np_linalg_solve_call_count"
                ],
                "normalized_named_populations": (
                    molecular_state
                    .partition_normalized_molecular_populations[0, named_indices]
                ),
                "physical_equation_densities": (
                    molecular_state
                    .previous_molecular_equation_densities[0]
                ),
                "population_constants": capture["population_constants"][0],
                "postsolve_row_scaled_residual_max": capture[
                    "postsolve_row_scaled_residual_max"
                ][0],
                "raw_named_populations": (
                    molecular_state.molecular_populations[0, named_indices]
                ),
                "stopping_relative_update_max": capture[
                    "stopping_relative_update_max"
                ][0],
                "transformed_equation_densities": (
                    molecular_state.molecular_equation_densities[0]
                ),
            }
        )
    stacked = {
        name: np.stack([np.asarray(record[name]) for record in records])
        for name in sorted(records[0])
    }
    iteration_count = np.asarray(
        stacked["newton_iteration_count"], dtype=np.int64
    )
    history_offsets = np.zeros(len(histories) + 1, dtype=np.int64)
    history_offsets[1:] = np.cumsum(iteration_count)
    stacked.update(
        {
            "newton_history_iteration_index": np.concatenate(
                [
                    np.arange(int(count), dtype=np.int64)
                    for count in iteration_count
                ]
            ),
            "newton_history_layer_index": np.repeat(
                np.arange(len(histories), dtype=np.int64),
                iteration_count,
            ),
            "newton_history_layer_offsets": history_offsets,
        }
    )
    for name in (
        "newton_linear_direct_solve_branch_mask",
        "newton_linear_lstsq_branch_mask",
        "newton_preupdate_equation_densities",
        "newton_raw_corrections",
        "newton_relative_update_max_history",
        "newton_still_iterating_history",
    ):
        stacked[name] = np.concatenate(
            [np.asarray(capture[name]) for capture in histories],
            axis=0,
        )
    return stacked


def _catalog_identity_arrays(catalog: Any) -> dict[str, np.ndarray]:
    return {
        "atmosphere_component_count": np.asarray(
            catalog.component_count, dtype=np.int64
        ),
        "atmosphere_component_equation_indices": np.asarray(
            catalog.component_equation_indices
        ).copy(),
        "atmosphere_component_start_indices": np.asarray(
            catalog.component_start_indices
        ).copy(),
        "atmosphere_equation_count": np.asarray(
            catalog.equation_count, dtype=np.int64
        ),
        "atmosphere_equation_species_codes": np.asarray(
            catalog.equation_species_codes
        ).copy(),
        "atmosphere_equilibrium_coefficients": np.asarray(
            catalog.equilibrium_coefficients
        ).copy(),
        "atmosphere_molecule_codes": np.asarray(catalog.molecule_codes).copy(),
        "atmosphere_molecule_count": np.asarray(
            catalog.molecule_count, dtype=np.int64
        ),
        "atmosphere_species_to_equation_index": np.asarray(
            catalog.species_to_equation_index
        ).copy(),
    }


def _environment_identity_arrays(
    modules: SimpleNamespace,
    declared_environment: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    arrays = {
        name: np.asarray(value).copy()
        for name, value in declared_environment.items()
    }
    try:
        numpy_configuration = np.show_config(mode="dicts")
    except TypeError:
        numpy_configuration = {}
    build_dependencies = numpy_configuration.get("Build Dependencies", {})
    blas = build_dependencies.get("blas", {})
    lapack = build_dependencies.get("lapack", {})
    arrays.update(
        {
            "blas_name": np.asarray(str(blas.get("name", "unknown"))),
            "blas_version": np.asarray(str(blas.get("version", "unknown"))),
            "lapack_name": np.asarray(str(lapack.get("name", "unknown"))),
            "lapack_version": np.asarray(
                str(lapack.get("version", "unknown"))
            ),
            "numpy_build_dependencies_json": np.asarray(
                json.dumps(
                    build_dependencies,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            ),
            "numba_thread_count": np.asarray(
                modules.numba.get_num_threads(), dtype=np.int64
            ),
            "numba_version": np.asarray(modules.numba.__version__),
            "numpy_version": np.asarray(np.__version__),
            "platform": np.asarray(platform.platform()),
            "platform_machine": np.asarray(platform.machine()),
            "platform_release": np.asarray(platform.release()),
            "platform_system": np.asarray(platform.system()),
            "python_executable": np.asarray(str(Path(sys.executable).resolve())),
            "python_implementation": np.asarray(platform.python_implementation()),
            "python_version": np.asarray(platform.python_version()),
            "system_byteorder": np.asarray(sys.byteorder),
        }
    )
    return arrays


def _executed_source_identity_arrays(
    pinned_root: Path,
) -> dict[str, np.ndarray]:
    """Record every imported atmosphere source module from the pinned checkout."""

    root = Path(pinned_root).resolve()
    records: list[tuple[str, str, str]] = []
    for module_name, module in sorted(sys.modules.items()):
        if module_name != "payne_zero_atmosphere" and not module_name.startswith(
            "payne_zero_atmosphere."
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        source_path = Path(module_file).resolve()
        if not _is_under(source_path, root):
            raise OracleIdentityError(
                f"executed module resolved outside pinned checkout: {source_path}"
            )
        if source_path.suffix != ".py":
            continue
        records.append(
            (
                module_name,
                str(source_path.relative_to(root)),
                sha256(source_path),
            )
        )
    if not records:
        raise OracleIdentityError("no executed atmosphere source modules found")
    module_names, relative_paths, hashes = zip(*records)
    return {
        "executed_atmosphere_source_count": np.asarray(
            len(records), dtype=np.int64
        ),
        "executed_atmosphere_source_module_names": np.asarray(
            module_names, dtype=np.str_
        ),
        "executed_atmosphere_source_relative_paths": np.asarray(
            relative_paths, dtype=np.str_
        ),
        "executed_atmosphere_source_sha256": np.asarray(
            hashes, dtype=np.str_
        ),
    }


def _named_catalog_indices(catalog: Any, codes: np.ndarray) -> np.ndarray:
    active_codes = np.asarray(
        catalog.molecule_codes[: catalog.molecule_count],
        np.float64,
    )
    indices = []
    for code in np.asarray(codes, np.float64):
        matches = np.flatnonzero(np.abs(active_codes - float(code)) < 0.005)
        if matches.size != 1:
            raise OracleIdentityError(
                f"named molecule code {code:g} has {matches.size} catalog matches"
            )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64)


def build_atmosphere_oracle_results(
    *,
    pinned_root: Path = PINNED_ROOT,
    fixture_path: Path = FIXTURE_PATH,
    require_complete: bool = False,
) -> dict[str, np.ndarray]:
    """Return the deterministic, unpublished atmosphere oracle arrays."""

    if require_complete:
        raise IncompleteOracleScopeError(
            "deferred Chapter 4 captures: " + ", ".join(DEFERRED_CAPTURE_FIELDS)
        )

    source_identities = validate_pinned_root(pinned_root)
    fixture = load_input_fixture(fixture_path)
    environment = validate_one_thread_environment()
    modules = _configure_pinned_modules(Path(pinned_root))
    original_numpy_linalg_solve = np.linalg.solve
    original_numpy_linalg_lstsq = np.linalg.lstsq
    catalog_path = (
        Path(pinned_root)
        / "source_data_files"
        / "source_catalogs"
        / "lines"
        / "molecular_equilibrium_atmosphere.npz"
    )
    catalog = modules.molecular_data.read_molecular_equilibrium_catalog(
        catalog_path
    )
    named_codes = np.asarray(fixture["named_molecule_codes"], np.float64)
    named_indices = _named_catalog_indices(catalog, named_codes)

    results: dict[str, np.ndarray] = {
        "fixture_sha256": np.asarray(FIXTURE_SHA256),
            "oracle_role": np.asarray(
                "comparison-only complete in-memory atmosphere capture; "
                "publication deferred"
            ),
            "payne_zero_commit": np.asarray(PINNED_COMMIT),
            "schema_version": np.asarray(2, dtype=np.int64),
    }
    for name, value in source_identities.items():
        results[name] = np.asarray(value)
    results.update(_environment_identity_arrays(modules, environment))
    results.update(_catalog_identity_arrays(catalog))
    results["named_molecule_catalog_indices"] = named_indices
    results["named_molecule_codes"] = named_codes.copy()

    atmosphere = _build_atmosphere(
        modules,
        column_mass=fixture["column_mass"],
        temperature=fixture["temperature"],
        gas_pressure=fixture["gas_pressure"],
        electron_density=fixture["electron_density_seed"],
        microturbulence=fixture["microturbulence"],
        elemental_abundances=fixture["elemental_abundances"],
    )
    config = modules.config.AtmosphereConfig(
        inputs=modules.config.AtmosphereInput(
            initial_atmosphere=atmosphere,
            molecules_path=catalog_path,
        ),
        outputs=modules.config.AtmosphereOutput(),
        enable_molecules=True,
        enable_convection=False,
    )
    with MolecularSolveCapture(
        modules.molecular_equilibrium
    ) as full_capture, PopulationScheduleCapture(
        modules.runner,
        modules.equation_of_state,
        full_capture,
    ) as population_schedule_capture:
        population_state = modules.runner.prepare_population_state(
            config,
            temperature_iteration_index=1,
            molecular_thermal_energy_erg=None,
        )
    molecular_state = population_state.molecular_state
    if molecular_state is None:
        raise RuntimeError("molecule-enabled route returned no molecular state")
    full_arrays = full_capture.arrays(
        depth_count=fixture["temperature"].size,
        molecule_count=int(catalog.molecule_count),
        equation_count=int(catalog.equation_count),
    )
    _prefix(results, "full", full_arrays)
    results.update(population_schedule_capture.arrays())

    runtime_state = population_state.runtime_state
    results.update(
        {
            "full_charge_square_density": runtime_state.charge_square_density.copy(),
            "full_column_mass": fixture["column_mass"].copy(),
            "full_fractional_doppler_widths": (
                population_state.fractional_doppler_widths.copy()
            ),
            "full_fractional_doppler_width_structural_infinity_mask": (
                np.logical_not(
                    np.isfinite(population_state.fractional_doppler_widths)
                )
            ),
            "full_fractional_doppler_width_structural_infinity_slots": (
                np.asarray([919, 927], dtype=np.int64)
            ),
            "full_gas_pressure": runtime_state.gas_pressure.copy(),
            "full_input_charge_square_density": (
                _expected_charge_square_density(
                    fixture["temperature"],
                    fixture["gas_pressure"],
                    fixture["electron_density_seed"],
                )
            ),
            "full_input_electron_density": fixture[
                "electron_density_seed"
            ].copy(),
            "full_ion_stage_populations_by_packed_slot": (
                runtime_state.ion_stage_populations_by_packed_slot.copy()
            ),
            "full_mass_density": runtime_state.mass_density.copy(),
            "full_major_isotope_mass_amu": (
                runtime_state.major_isotope_mass_amu.copy()
            ),
            "full_molecular_populations": (
                molecular_state.molecular_populations.copy()
            ),
            "full_partition_normalized_molecular_populations": (
                molecular_state.partition_normalized_molecular_populations.copy()
            ),
            "full_partition_normalized_population_over_mass_density_and_width": (
                population_state
                .partition_normalized_population_over_mass_density_and_fractional_doppler_width
                .copy()
            ),
            "full_partition_normalized_populations_by_packed_slot": (
                runtime_state.partition_normalized_populations_by_packed_slot.copy()
            ),
            "full_previous_molecular_equation_densities": (
                molecular_state.previous_molecular_equation_densities.copy()
            ),
            "full_specific_internal_energy": (
                runtime_state.specific_internal_energy.copy()
            ),
            "full_temperature": fixture["temperature"].copy(),
            "full_temperature_iteration_cache_value": np.asarray(
                population_state.temperature_iteration_cache["pops_itemp"],
                dtype=np.int64,
            ),
            "full_total_nuclei_number_density": (
                runtime_state.total_nuclei_number_density.copy()
            ),
            "full_transformed_molecular_equation_densities": (
                molecular_state.molecular_equation_densities.copy()
            ),
            "full_output_electron_density": runtime_state.electron_density.copy(),
        }
    )

    handoff_input_electron_density = (
        runtime_state.electron_density.copy()
    )
    handoff_atmosphere = _build_atmosphere(
        modules,
        column_mass=fixture["column_mass"],
        temperature=fixture["temperature"],
        gas_pressure=fixture["gas_pressure"],
        electron_density=handoff_input_electron_density,
        microturbulence=fixture["microturbulence"],
        elemental_abundances=fixture["elemental_abundances"],
    )
    handoff_config = modules.config.AtmosphereConfig(
        inputs=modules.config.AtmosphereInput(
            initial_atmosphere=handoff_atmosphere,
            molecules_path=catalog_path,
        ),
        outputs=modules.config.AtmosphereOutput(),
        enable_molecules=True,
        enable_convection=False,
    )
    with MolecularSolveCapture(
        modules.molecular_equilibrium
    ) as handoff_capture:
        handoff_population_state = (
            modules.runner.prepare_structured_handoff_population_state(
                handoff_config,
                temperature_iteration_index=1,
                molecular_thermal_energy_erg=None,
            )
        )
    handoff_molecular_state = handoff_population_state.molecular_state
    if handoff_molecular_state is None:
        raise RuntimeError("structured handoff returned no molecular state")
    _prefix(
        results,
        "handoff",
        handoff_capture.arrays(
            depth_count=fixture["temperature"].size,
            molecule_count=int(catalog.molecule_count),
            equation_count=int(catalog.equation_count),
        ),
    )
    handoff_runtime = handoff_population_state.runtime_state
    results.update(
        {
            "handoff_charge_square_density": (
                handoff_runtime.charge_square_density.copy()
            ),
            "handoff_fractional_doppler_widths": (
                handoff_population_state.fractional_doppler_widths.copy()
            ),
            "handoff_input_electron_density": (
                handoff_input_electron_density.copy()
            ),
            "handoff_ion_stage_populations_by_packed_slot": (
                handoff_runtime.ion_stage_populations_by_packed_slot.copy()
            ),
            "handoff_molecular_populations": (
                handoff_molecular_state.molecular_populations.copy()
            ),
            "handoff_output_electron_density": (
                handoff_runtime.electron_density.copy()
            ),
            "handoff_partition_normalized_molecular_populations": (
                handoff_molecular_state
                .partition_normalized_molecular_populations
                .copy()
            ),
            "handoff_partition_normalized_populations_by_packed_slot": (
                handoff_runtime
                .partition_normalized_populations_by_packed_slot
                .copy()
            ),
            "handoff_temperature_iteration_cache_value": np.asarray(
                handoff_population_state
                .temperature_iteration_cache["pops_itemp"],
                dtype=np.int64,
            ),
            "handoff_transformed_molecular_equation_densities": (
                handoff_molecular_state.molecular_equation_densities.copy()
            ),
        }
    )

    disabled_atmosphere = _build_atmosphere(
        modules,
        column_mass=fixture["column_mass"],
        temperature=fixture["temperature"],
        gas_pressure=fixture["gas_pressure"],
        electron_density=fixture["electron_density_seed"],
        microturbulence=fixture["microturbulence"],
        elemental_abundances=fixture["elemental_abundances"],
    )
    disabled_atmosphere.metadata["pressure_iteration_enabled"] = "0"
    disabled_config = modules.config.AtmosphereConfig(
        inputs=modules.config.AtmosphereInput(
            initial_atmosphere=disabled_atmosphere,
            molecules_path=catalog_path,
        ),
        outputs=modules.config.AtmosphereOutput(),
        enable_molecules=True,
        enable_convection=False,
    )
    with MolecularSolveCapture(
        modules.molecular_equilibrium
    ) as disabled_capture:
        disabled_population_state = modules.runner.prepare_population_state(
            disabled_config,
            temperature_iteration_index=1,
            molecular_thermal_energy_erg=None,
        )
    disabled_molecular_state = disabled_population_state.molecular_state
    if disabled_molecular_state is None:
        raise RuntimeError("disabled route returned no molecular state")
    disabled_runtime = disabled_population_state.runtime_state
    results.update(
        {
            "disabled_charge_square_density": (
                disabled_runtime.charge_square_density.copy()
            ),
            "disabled_fill_call_count": np.asarray(
                len(disabled_capture.fill_before), dtype=np.int64
            ),
            "disabled_fractional_doppler_widths": (
                disabled_population_state.fractional_doppler_widths.copy()
            ),
            "disabled_input_electron_density": (
                fixture["electron_density_seed"].copy()
            ),
            "disabled_ion_stage_populations_by_packed_slot": (
                disabled_runtime.ion_stage_populations_by_packed_slot.copy()
            ),
            "disabled_layer_call_count": np.asarray(
                len(disabled_capture.layer_order), dtype=np.int64
            ),
            "disabled_molecular_equation_densities": (
                disabled_molecular_state.molecular_equation_densities.copy()
            ),
            "disabled_molecular_populations": (
                disabled_molecular_state.molecular_populations.copy()
            ),
            "disabled_output_electron_density": (
                disabled_runtime.electron_density.copy()
            ),
            "disabled_partition_normalized_molecular_populations": (
                disabled_molecular_state
                .partition_normalized_molecular_populations
                .copy()
            ),
            "disabled_partition_normalized_population_over_mass_density_and_width": (
                disabled_population_state
                .partition_normalized_population_over_mass_density_and_fractional_doppler_width
                .copy()
            ),
            "disabled_partition_normalized_populations_by_packed_slot": (
                disabled_runtime
                .partition_normalized_populations_by_packed_slot
                .copy()
            ),
            "disabled_pressure_iteration_enabled": np.asarray(
                disabled_population_state.setup.pressure_iteration_enabled,
                dtype=np.bool_,
            ),
            "disabled_previous_molecular_equation_densities": (
                disabled_molecular_state
                .previous_molecular_equation_densities
                .copy()
            ),
            "disabled_solve_call_count": np.asarray(
                disabled_capture.solve_call_count, dtype=np.int64
            ),
            "disabled_specific_internal_energy": (
                disabled_runtime.specific_internal_energy.copy()
            ),
            "disabled_temperature_iteration_cache_key_count": np.asarray(
                len(disabled_population_state.temperature_iteration_cache),
                dtype=np.int64,
            ),
            "disabled_temperature_iteration_cache_pops_itemp_present": (
                np.asarray(
                    "pops_itemp"
                    in disabled_population_state.temperature_iteration_cache,
                    dtype=np.bool_,
                )
            ),
            "disabled_total_nuclei_number_density": (
                disabled_runtime.total_nuclei_number_density.copy()
            ),
        }
    )

    _prefix(
        results,
        "mode2",
        _one_layer_mode_probe(
            modules,
            catalog,
            fixture,
            population_mode=2,
        ),
    )
    _prefix(
        results,
        "mode12",
        _one_layer_mode_probe(
            modules,
            catalog,
            fixture,
            population_mode=12,
        ),
    )

    energy_atmosphere = _build_atmosphere(
        modules,
        column_mass=fixture["column_mass"],
        temperature=fixture["temperature"],
        gas_pressure=fixture["gas_pressure"],
        electron_density=fixture["electron_density_seed"],
        microturbulence=fixture["microturbulence"],
        elemental_abundances=fixture["elemental_abundances"],
    )
    energy_runtime, energy_molecular_state = _initialize_molecular_state(
        modules, catalog, energy_atmosphere
    )
    energy_saved_rows = (
        molecular_state.previous_molecular_equation_densities.copy()
    )
    energy_normalized_seed = (
        molecular_state.partition_normalized_molecular_populations.copy()
    )
    modules.molecular_equilibrium.restore_molecular_equation_density(
        energy_molecular_state,
        energy_saved_rows.copy(),
    )
    energy_molecular_state.partition_normalized_molecular_populations[:] = (
        energy_normalized_seed
    )
    energy_saved_before = (
        energy_molecular_state.previous_molecular_equation_densities.copy()
    )
    energy_normalized_before = (
        energy_molecular_state
        .partition_normalized_molecular_populations
        .copy()
    )
    energy_specific_before = (
        energy_runtime.specific_internal_energy.copy()
    )
    modules.molecular_equilibrium.set_molecular_specific_internal_energy_mode(
        energy_molecular_state,
        True,
    )
    with MolecularSolveCapture(
        modules.molecular_equilibrium
    ) as energy_capture:
        modules.molecular_equilibrium.solve_molecular_equilibrium(
            energy_molecular_state,
            population_mode=1,
        )
    energy_trace = energy_capture.trace_arrays(
        depth_count=fixture["temperature"].size,
        molecule_count=int(catalog.molecule_count),
        equation_count=int(catalog.equation_count),
        expected_population_mode=1,
        expected_fill_call_count=0,
    )
    _prefix(results, "energy", energy_trace)

    equation_count = int(catalog.equation_count)
    energy_ordinary_seeds = np.zeros(
        (fixture["temperature"].size, equation_count),
        dtype=np.float64,
    )
    first_total_density = float(
        fixture["gas_pressure"][0]
        / (
            fixture["temperature"][0]
            * REFERENCE_BOLTZMANN_ERG_PER_K
        )
    )
    first_nuclei_seed = (
        first_total_density
        if float(fixture["temperature"][0]) < 4000.0
        else first_total_density / 2.0
    )
    first_electron_seed = first_nuclei_seed / 10.0
    first_abundance = (
        modules.molecular_equilibrium._abundance_vector_for_layer(
            energy_molecular_state,
            0,
        )
    )
    energy_ordinary_seeds[0, 0] = first_nuclei_seed
    energy_ordinary_seeds[0, 1:] = (
        first_electron_seed * first_abundance[1:]
    )
    if int(catalog.equation_species_codes[equation_count - 1]) == 100:
        energy_ordinary_seeds[0, equation_count - 1] = (
            first_electron_seed
        )
    for layer_index in range(1, fixture["temperature"].size):
        pressure_ratio = float(
            fixture["gas_pressure"][layer_index]
            / fixture["gas_pressure"][layer_index - 1]
        )
        energy_ordinary_seeds[layer_index] = (
            energy_molecular_state.molecular_equation_densities[
                layer_index - 1
            ]
            * pressure_ratio
        )
    energy_direct_reference = (
        modules.molecular_equilibrium
        .compute_molecular_specific_internal_energy(
            energy_molecular_state
        )
    )
    results.update(
        {
            "energy_actual_seed_override": (
                energy_trace["layer_seed_equation_densities"].copy()
            ),
            "energy_direct_specific_internal_energy_reference": (
                energy_direct_reference.copy()
            ),
            "energy_molecular_equation_densities_after": (
                energy_molecular_state
                .molecular_equation_densities
                .copy()
            ),
            "energy_normalized_populations_after": (
                energy_molecular_state
                .partition_normalized_molecular_populations
                .copy()
            ),
            "energy_normalized_populations_before": (
                energy_normalized_before
            ),
            "energy_ordinary_continuation_seed": (
                energy_ordinary_seeds
            ),
            "energy_raw_populations_after": (
                energy_molecular_state.molecular_populations.copy()
            ),
            "energy_saved_physical_rows_after": (
                energy_molecular_state
                .previous_molecular_equation_densities
                .copy()
            ),
            "energy_saved_physical_rows_before": energy_saved_before,
            "energy_saved_physical_rows_input": energy_saved_rows,
            "energy_specific_internal_energy_after": (
                energy_runtime.specific_internal_energy.copy()
            ),
            "energy_specific_internal_energy_before": (
                energy_specific_before
            ),
            "energy_specific_internal_energy_mode_enabled": np.asarray(
                energy_molecular_state
                .specific_internal_energy_mode_enabled,
                dtype=np.bool_,
            ),
        }
    )

    bridge_error: ValueError | None = None
    try:
        modules.synthesis_bridge.structured_atmosphere_from_runtime_state(
            atmosphere=atmosphere,
            runtime_state=runtime_state,
            molecular_state=molecular_state,
        )
    except ValueError as exc:
        bridge_error = exc
    if bridge_error is None:
        raise RuntimeError("pinned live/debug bridge unexpectedly accepted shapes")
    bridge_h2_temperature = np.asarray(
        [8000.0, 9500.0, 7000.0], dtype=np.float64
    )
    bridge_h2_packed_populations = np.zeros(
        (bridge_h2_temperature.size, 1006),
        dtype=np.float64,
    )
    bridge_h2_packed_populations[:, 0] = np.asarray(
        [2.0e14, 3.0e14, 4.0e14], dtype=np.float64
    )
    bridge_h2_codes = np.asarray([101.0], dtype=np.float64)
    bridge_h2_mixed_catalog = np.asarray(
        [[5.0e8], [0.0], [7.0e8]],
        dtype=np.float64,
    )
    bridge_h2_all_zero_catalog = np.zeros_like(bridge_h2_mixed_catalog)
    bridge_h2_mixed_output = (
        modules.synthesis_bridge._molecular_hydrogen_population(
            temperature=bridge_h2_temperature,
            ion_stage_populations_by_packed_slot=(
                bridge_h2_packed_populations
            ),
            molecule_codes=bridge_h2_codes,
            molecular_populations=bridge_h2_mixed_catalog,
        )
    )
    bridge_h2_all_zero_output = (
        modules.synthesis_bridge._molecular_hydrogen_population(
            temperature=bridge_h2_temperature,
            ion_stage_populations_by_packed_slot=(
                bridge_h2_packed_populations
            ),
            molecule_codes=bridge_h2_codes,
            molecular_populations=bridge_h2_all_zero_catalog,
        )
    )
    bridge_h2_no_catalog_output = (
        modules.synthesis_bridge._molecular_hydrogen_population(
            temperature=bridge_h2_temperature,
            ion_stage_populations_by_packed_slot=(
                bridge_h2_packed_populations
            ),
            molecule_codes=None,
            molecular_populations=None,
        )
    )
    results.update(
        {
            "bridge_h2_all_zero_catalog_input": (
                bridge_h2_all_zero_catalog.copy()
            ),
            "bridge_h2_all_zero_output": (
                bridge_h2_all_zero_output.copy()
            ),
            "bridge_h2_mixed_catalog_input": (
                bridge_h2_mixed_catalog.copy()
            ),
            "bridge_h2_mixed_output": bridge_h2_mixed_output.copy(),
            "bridge_h2_no_catalog_output": (
                bridge_h2_no_catalog_output.copy()
            ),
            "bridge_h2_packed_neutral_hydrogen": (
                bridge_h2_packed_populations[:, 0].copy()
            ),
            "bridge_h2_temperature": bridge_h2_temperature.copy(),
            "bridge_live_shape_error_message": np.asarray(
                str(bridge_error)
            ),
            "bridge_live_shape_error_type": np.asarray(
                type(bridge_error).__name__
            ),
            "bridge_padded_molecule_code_shape": np.asarray(
                catalog.molecule_codes.shape, dtype=np.int64
            ),
            "bridge_active_molecular_population_shape": np.asarray(
                molecular_state.molecular_populations.shape,
                dtype=np.int64,
            ),
        }
    )

    temperature_control = _stack_control_probes(
        modules,
        catalog,
        fixture,
        temperatures=fixture["temperature_control_temperature"],
        gas_pressures=fixture["temperature_control_gas_pressure"],
        named_indices=named_indices,
    )
    _prefix(results, "temperature_control", temperature_control)
    results["temperature_control_temperature"] = fixture[
        "temperature_control_temperature"
    ].copy()
    results["temperature_control_gas_pressure"] = fixture[
        "temperature_control_gas_pressure"
    ].copy()

    pressure_control = _stack_control_probes(
        modules,
        catalog,
        fixture,
        temperatures=fixture["pressure_control_temperature"],
        gas_pressures=fixture["pressure_control_gas_pressure"],
        named_indices=named_indices,
    )
    _prefix(results, "pressure_control", pressure_control)
    results["pressure_control_temperature"] = fixture[
        "pressure_control_temperature"
    ].copy()
    results["pressure_control_gas_pressure"] = fixture[
        "pressure_control_gas_pressure"
    ].copy()

    boundary_temperature = np.concatenate(
        (
            fixture["atmosphere_polynomial_boundary_temperature"],
            fixture["atmosphere_h2_boundary_temperature"][-2:],
        )
    )
    boundary_pressure = np.full(
        boundary_temperature.shape,
        float(fixture["temperature_control_gas_pressure"][0]),
        dtype=np.float64,
    )
    boundary = _stack_control_probes(
        modules,
        catalog,
        fixture,
        temperatures=boundary_temperature,
        gas_pressures=boundary_pressure,
        named_indices=np.arange(catalog.molecule_count, dtype=np.int64),
    )
    _prefix(results, "boundary", boundary)
    for name in (
        "frozen_newton_constants",
        "normalized_named_populations",
        "population_constants",
        "raw_named_populations",
    ):
        results[f"boundary_{name}_uint64_bits"] = np.asarray(
            boundary[name], dtype=np.float64
        ).view(np.uint64).copy()
    results["boundary_temperature"] = boundary_temperature
    results["boundary_temperature_uint64_bits"] = (
        boundary_temperature.view(np.uint64).copy()
    )
    results["boundary_polynomial_active_branch_mask"] = (
        boundary_temperature <= 10000.0
    )
    results["boundary_polynomial_inactive_branch_mask"] = (
        boundary_temperature > 10000.0
    )
    results["boundary_h2_catalog_active_branch_mask"] = (
        boundary_temperature <= 20000.0
    )
    results["boundary_h2_catalog_inactive_branch_mask"] = (
        boundary_temperature > 20000.0
    )
    results["boundary_gas_pressure"] = boundary_pressure

    h2_temperature = fixture["atmosphere_h2_boundary_temperature"].copy()
    h2_pressure = float(fixture["temperature_control_gas_pressure"][0])
    h2_constants = []
    h2_partitions = []
    h2_helper_constants = []
    for temperature in h2_temperature:
        one_atmosphere = _build_atmosphere(
            modules,
            column_mass=np.asarray([fixture["column_mass"][0]], np.float64),
            temperature=np.asarray([temperature], np.float64),
            gas_pressure=np.asarray([h2_pressure], np.float64),
            electron_density=np.asarray(
                [
                    0.1
                    * h2_pressure
                    / (
                        REFERENCE_BOLTZMANN_ERG_PER_K
                        * float(temperature)
                    )
                ],
                np.float64,
            ),
            microturbulence=np.asarray(
                [fixture["microturbulence"][0]], np.float64
            ),
            elemental_abundances=fixture["elemental_abundances"],
        )
        one_runtime = modules.runtime_state.build_runtime_state(one_atmosphere)
        modules.runtime_state.update_charge_square_density(
            thermal_energy_erg=one_atmosphere.thermal_energy_erg,
            state=one_runtime,
        )
        one_molecular = (
            modules.molecular_equilibrium.initialize_molecular_equilibrium_state(
                temperature_k=one_atmosphere.temperature,
                thermal_energy_erg=one_atmosphere.thermal_energy_erg,
                gas_pressure=one_runtime.gas_pressure,
                runtime_state=one_runtime,
                catalog=catalog,
            )
        )
        h2_constants.append(
            modules.molecular_equilibrium.compute_equilibrium_constants_for_layer(
                one_molecular, 0
            )
        )
        h2_partitions.append(
            modules.molecular_equilibrium._interp_hydrogen_molecule_partition(
                float(temperature)
            )
        )
        h2_helper_constants.append(
            modules.molecular_equilibrium.hydrogen_molecule_equilibrium_constant(
                float(temperature)
            )
        )
    h2_catalog_index = int(named_indices[0])
    results["h2_probe_catalog_equilibrium_constant_gated"] = np.asarray(
        [values[h2_catalog_index] for values in h2_constants],
        dtype=np.float64,
    )
    results[
        "h2_probe_catalog_equilibrium_constant_gated_uint64_bits"
    ] = results["h2_probe_catalog_equilibrium_constant_gated"].view(
        np.uint64
    ).copy()
    results["h2_probe_helper_equilibrium_constant_ungated"] = np.asarray(
        h2_helper_constants, dtype=np.float64
    )
    results[
        "h2_probe_helper_equilibrium_constant_ungated_uint64_bits"
    ] = results["h2_probe_helper_equilibrium_constant_ungated"].view(
        np.uint64
    ).copy()
    results["h2_probe_interpolated_partition"] = np.asarray(
        h2_partitions, dtype=np.float64
    )
    results["h2_probe_interpolated_partition_uint64_bits"] = results[
        "h2_probe_interpolated_partition"
    ].view(np.uint64).copy()
    results["h2_probe_temperature"] = h2_temperature
    results["h2_probe_temperature_uint64_bits"] = (
        h2_temperature.view(np.uint64).copy()
    )
    results["h2_probe_partition_low_clamp_branch_mask"] = (
        h2_temperature <= 100.0
    )
    results["h2_probe_partition_interpolation_branch_mask"] = np.logical_and(
        h2_temperature > 100.0,
        h2_temperature < 19900.0,
    )
    results["h2_probe_partition_high_clamp_branch_mask"] = (
        h2_temperature >= 19900.0
    )
    results["h2_probe_helper_finite_positive_input_branch_mask"] = (
        np.logical_and(np.isfinite(h2_temperature), h2_temperature > 0.0)
    )
    results["h2_probe_catalog_active_branch_mask"] = (
        h2_temperature <= 20000.0
    )
    results["h2_probe_catalog_inactive_branch_mask"] = (
        h2_temperature > 20000.0
    )

    results["capture_main_full_route_implemented"] = np.asarray(
        True, dtype=np.bool_
    )
    results["capture_temperature_controls_implemented"] = np.asarray(
        True, dtype=np.bool_
    )
    results["capture_pressure_controls_implemented"] = np.asarray(
        True, dtype=np.bool_
    )
    results["capture_boundary_lifecycle_implemented"] = np.asarray(
        True, dtype=np.bool_
    )
    results["capture_structured_handoff_implemented"] = np.asarray(
        True, dtype=np.bool_
    )
    results["capture_pressure_iteration_disabled_implemented"] = np.asarray(
        True, dtype=np.bool_
    )
    results["capture_population_modes_2_and_12_implemented"] = np.asarray(
        True, dtype=np.bool_
    )
    results["capture_specific_internal_energy_mode_implemented"] = np.asarray(
        True, dtype=np.bool_
    )
    results["capture_live_debug_bridge_implemented"] = np.asarray(
        True, dtype=np.bool_
    )
    results[
        "capture_zero_code_priming_and_cached_schedule_implemented"
    ] = np.asarray(True, dtype=np.bool_)
    for name in DEFERRED_CAPTURE_FIELDS:
        results[f"deferred_{name}"] = np.asarray(True, dtype=np.bool_)
    results["capture_in_memory_scope_complete"] = np.asarray(
        True, dtype=np.bool_
    )
    results["capture_scope_complete"] = np.asarray(False, dtype=np.bool_)
    results["capture_wrappers_restored"] = np.asarray(
        all(
            not getattr(
                getattr(modules.molecular_equilibrium, name),
                "__chapter04_capture_wrapper__",
                False,
            )
            for name in (
                "solve_molecular_equilibrium",
                "solve_molecular_equilibrium_layer",
                "_compute_equilibrium_constants_for_layer_compiled",
                "_newton_update_kernel",
                "_fill_partition_normalized_molecular_densities",
            )
        ),
        dtype=np.bool_,
    )
    results["capture_numpy_linalg_wrappers_restored"] = np.asarray(
        np.linalg.solve is original_numpy_linalg_solve
        and np.linalg.lstsq is original_numpy_linalg_lstsq,
        dtype=np.bool_,
    )
    results["capture_population_schedule_wrappers_restored"] = np.asarray(
        all(
            not getattr(binding, "__chapter04_capture_wrapper__", False)
            for binding in (
                modules.runner.populate_species,
                modules.runner.populate_all_species,
                modules.equation_of_state.populate_species,
            )
        ),
        dtype=np.bool_,
    )
    results.update(_executed_source_identity_arrays(Path(pinned_root)))

    ordered = {
        name: np.asarray(results[name]).copy() for name in sorted(results)
    }
    self_check_results(ordered)
    return ordered


def _self_check_newton_history(
    results: Mapping[str, np.ndarray],
    *,
    prefix: str,
    layer_count: int,
    equation_count: int = 23,
) -> None:
    """Verify that one prefixed trace is a lossless ordered Newton history."""

    iteration_count = np.asarray(
        results[f"{prefix}_newton_iteration_count"], dtype=np.int64
    )
    offsets = np.asarray(
        results[f"{prefix}_newton_history_layer_offsets"], dtype=np.int64
    )
    history_count = int(np.sum(iteration_count))
    expected_offsets = np.zeros(layer_count + 1, dtype=np.int64)
    expected_offsets[1:] = np.cumsum(iteration_count)
    if iteration_count.shape != (layer_count,):
        raise AssertionError(f"{prefix} iteration-count shape changed")
    if not np.array_equal(offsets, expected_offsets):
        raise AssertionError(f"{prefix} Newton history offsets changed")
    if not np.array_equal(
        results[f"{prefix}_newton_history_layer_index"],
        np.repeat(np.arange(layer_count, dtype=np.int64), iteration_count),
    ):
        raise AssertionError(f"{prefix} Newton history layer index changed")
    expected_iteration_index = np.concatenate(
        [
            np.arange(int(count), dtype=np.int64)
            for count in iteration_count
        ]
    )
    if not np.array_equal(
        results[f"{prefix}_newton_history_iteration_index"],
        expected_iteration_index,
    ):
        raise AssertionError(f"{prefix} Newton iteration index changed")
    for name in (
        "newton_preupdate_equation_densities",
        "newton_raw_corrections",
    ):
        if results[f"{prefix}_{name}"].shape != (
            history_count,
            equation_count,
        ):
            raise AssertionError(f"{prefix} {name} shape changed")
    for name in (
        "newton_linear_direct_solve_branch_mask",
        "newton_linear_lstsq_branch_mask",
        "newton_relative_update_max_history",
        "newton_still_iterating_history",
    ):
        if results[f"{prefix}_{name}"].shape != (history_count,):
            raise AssertionError(f"{prefix} {name} length changed")
    direct_mask = np.asarray(
        results[f"{prefix}_newton_linear_direct_solve_branch_mask"],
        dtype=np.bool_,
    )
    lstsq_mask = np.asarray(
        results[f"{prefix}_newton_linear_lstsq_branch_mask"],
        dtype=np.bool_,
    )
    if np.any(np.logical_and(direct_mask, lstsq_mask)) or not np.all(
        np.logical_or(direct_mask, lstsq_mask)
    ):
        raise AssertionError(f"{prefix} linear branch masks are not exclusive")
    solve_calls = np.asarray(
        results[f"{prefix}_newton_np_linalg_solve_call_count"],
        dtype=np.int64,
    )
    lstsq_calls = np.asarray(
        results[f"{prefix}_newton_np_linalg_lstsq_call_count"],
        dtype=np.int64,
    )
    direct_count = np.asarray(
        results[f"{prefix}_newton_np_linalg_direct_solve_branch_count"],
        dtype=np.int64,
    )
    if int(np.sum(solve_calls)) != history_count:
        raise AssertionError(f"{prefix} np.linalg.solve count changed")
    if int(np.sum(lstsq_calls)) != int(np.count_nonzero(lstsq_mask)):
        raise AssertionError(f"{prefix} np.linalg.lstsq count changed")
    if int(np.sum(direct_count)) != int(np.count_nonzero(direct_mask)):
        raise AssertionError(f"{prefix} direct-solve branch count changed")
    still_iterating = np.asarray(
        results[f"{prefix}_newton_still_iterating_history"],
        dtype=np.bool_,
    )
    for layer in range(layer_count):
        start = int(offsets[layer])
        stop = int(offsets[layer + 1])
        if stop <= start:
            raise AssertionError(f"{prefix} layer {layer} has no Newton history")
        if np.any(np.logical_not(still_iterating[start : stop - 1])):
            raise AssertionError(f"{prefix} layer {layer} stopped before its final step")
        final_expected = not bool(results[f"{prefix}_newton_converged"][layer])
        if bool(still_iterating[stop - 1]) != final_expected:
            raise AssertionError(f"{prefix} final still-iterating state changed")
        if (
            results[f"{prefix}_newton_stopping_relative_update_max"]
            if f"{prefix}_newton_stopping_relative_update_max" in results
            else results[f"{prefix}_stopping_relative_update_max"]
        )[layer] != results[
            f"{prefix}_newton_relative_update_max_history"
        ][stop - 1]:
            raise AssertionError(f"{prefix} final relative update changed")


def self_check_results(results: Mapping[str, np.ndarray]) -> None:
    """Fail before publication if this partial in-memory result is inconsistent."""

    if list(results) != sorted(results):
        raise AssertionError("oracle result keys must be sorted")
    for name, value in results.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise AssertionError(f"oracle result {name} uses object dtype")
        if name in (
            "disabled_fractional_doppler_widths",
            "full_fractional_doppler_widths",
            "handoff_fractional_doppler_widths",
        ):
            continue
        if np.issubdtype(array.dtype, np.number) and not np.all(
            np.isfinite(array)
        ):
            raise AssertionError(f"oracle result {name} is nonfinite")

    if bool(results["capture_scope_complete"]):
        raise AssertionError("publication-deferred scope cannot be complete")
    if not bool(results["capture_in_memory_scope_complete"]):
        raise AssertionError("in-memory atmosphere capture is incomplete")
    if not bool(results["capture_wrappers_restored"]):
        raise AssertionError("molecular capture wrappers escaped their context")
    if not bool(results["capture_numpy_linalg_wrappers_restored"]):
        raise AssertionError("NumPy linear-algebra wrappers escaped their context")
    if not bool(results["capture_population_schedule_wrappers_restored"]):
        raise AssertionError("population schedule wrappers escaped their context")
    for name in DEFERRED_CAPTURE_FIELDS:
        if not bool(results[f"deferred_{name}"]):
            raise AssertionError(f"deferred scope marker was cleared: {name}")
    for name in (
        "capture_structured_handoff_implemented",
        "capture_pressure_iteration_disabled_implemented",
        "capture_population_modes_2_and_12_implemented",
        "capture_specific_internal_energy_mode_implemented",
        "capture_live_debug_bridge_implemented",
        "capture_zero_code_priming_and_cached_schedule_implemented",
    ):
        if not bool(results[name]):
            raise AssertionError(f"implemented capture marker was cleared: {name}")

    for prefix, layer_count in (
        ("full", 6),
        ("handoff", 6),
        ("mode2", 1),
        ("mode12", 1),
        ("energy", 6),
        ("temperature_control", 3),
        ("pressure_control", 3),
        ("boundary", 4),
    ):
        _self_check_newton_history(
            results,
            prefix=prefix,
            layer_count=layer_count,
        )

    if (
        int(results["priming_cache_zero_code_call_count"]) != 1
        or float(results["priming_cache_zero_code"]) != 0.0
        or int(results["priming_cache_zero_code_mode"]) != 1
        or not np.array_equal(
            results["priming_cache_zero_code_output_shape"],
            np.asarray([6, 1], dtype=np.int64),
        )
    ):
        raise AssertionError("zero-code priming call changed")
    if (
        int(results["priming_cache_temperature_iteration_index"]) != 1
        or int(results["priming_cache_cache_value_before_priming"]) != -1
        or int(results["priming_cache_cache_value_after_priming"]) != 1
        or int(results["priming_cache_molecular_solve_count_during_zero_code"])
        != 1
    ):
        raise AssertionError("zero-code solve/cache ownership changed")
    if (
        int(results["priming_cache_populate_all_call_count"]) != 1
        or int(results["priming_cache_job_count_during_schedule"]) != 230
        or int(results["priming_cache_cache_value_before_schedule"]) != 1
        or int(results["priming_cache_cache_value_after_schedule"]) != 1
        or int(results["priming_cache_schedule_solve_count_before"]) != 1
        or int(results["priming_cache_schedule_solve_count_after"]) != 1
        or int(results["priming_cache_additional_solve_count_during_schedule"])
        != 0
    ):
        raise AssertionError("cached population schedule ownership changed")
    for name in (
        "priming_cache_schedule_job_cache_value_before",
        "priming_cache_schedule_job_cache_value_after",
        "priming_cache_schedule_job_solve_count_before",
        "priming_cache_schedule_job_solve_count_after",
        "priming_cache_schedule_job_temperature_iteration_index",
    ):
        if not np.array_equal(
            results[name],
            np.ones(230, dtype=np.int64),
        ):
            raise AssertionError(f"{name} changed")

    schedule_codes = np.asarray(
        results["schedule_inventory_code"], dtype=np.float64
    )
    schedule_modes = np.asarray(
        results["schedule_inventory_mode"], dtype=np.int64
    )
    molecular_mask = np.asarray(
        results["schedule_inventory_molecular_job_mask"], dtype=np.bool_
    )
    atomic_mask = np.asarray(
        results["schedule_inventory_atomic_job_mask"], dtype=np.bool_
    )
    if (
        int(results["schedule_inventory_job_count"]) != 230
        or schedule_codes.shape != (230,)
        or schedule_modes.shape != (230,)
        or int(results["schedule_inventory_atomic_job_count"]) != 198
        or int(results["schedule_inventory_molecular_job_count"]) != 32
        or not np.array_equal(atomic_mask, np.logical_not(molecular_mask))
        or np.count_nonzero(atomic_mask) != 198
        or np.count_nonzero(molecular_mask) != 32
    ):
        raise AssertionError("packed schedule composition changed")
    expected_molecular_codes = np.asarray(
        [
            101.0,
            106.0,
            107.0,
            108.0,
            112.0,
            114.0,
            120.0,
            124.0,
            126.0,
            606.0,
            607.0,
            608.0,
            814.0,
            822.0,
            823.0,
            10108.0,
        ],
        dtype=np.float64,
    )
    expected_molecular_slots = np.asarray(
        [
            841,
            846,
            847,
            848,
            851,
            853,
            858,
            862,
            864,
            868,
            869,
            870,
            889,
            895,
            896,
            940,
        ],
        dtype=np.int64,
    )
    if (
        int(results["schedule_inventory_molecular_unique_code_count"]) != 16
        or not np.array_equal(
            results["schedule_inventory_molecular_unique_code"],
            expected_molecular_codes,
        )
        or not np.array_equal(
            results["schedule_inventory_molecular_unique_mode_pair"],
            np.tile(np.asarray([1, 11], dtype=np.int64), (16, 1)),
        )
        or not np.array_equal(
            results[
                "schedule_inventory_"
                "molecular_unique_packed_start_slot_one_based"
            ],
            expected_molecular_slots,
        )
    ):
        raise AssertionError("molecular packed schedule inventory changed")
    if not np.array_equal(
        schedule_codes[molecular_mask],
        np.repeat(expected_molecular_codes, 2),
    ):
        raise AssertionError("molecular packed schedule order changed")
    if not np.array_equal(
        results["schedule_inventory_target_index"][molecular_mask],
        np.zeros(32, dtype=np.int64),
    ) or not np.array_equal(
        results["schedule_inventory_output_slots"][molecular_mask],
        np.ones(32, dtype=np.int64),
    ):
        raise AssertionError("molecular packed schedule target changed")

    if results["full_molecular_populations"].shape != (6, 170):
        raise AssertionError("full molecular population shape changed")
    if results["full_transformed_molecular_equation_densities"].shape != (
        6,
        23,
    ):
        raise AssertionError("full molecular equation shape changed")
    if int(results["full_solve_call_count"]) != 1:
        raise AssertionError("full route must call molecular equilibrium once")
    if not np.array_equal(results["full_layer_order"], np.arange(6)):
        raise AssertionError("full route depth order changed")
    if not np.all(results["full_newton_converged"]):
        raise AssertionError("full route contains a nonconverged layer")
    if np.any(results["full_newton_exhausted"]):
        raise AssertionError("full route reached the 200-iteration limit")
    if np.any(results["full_newton_iteration_count"] > 200):
        raise AssertionError("invalid molecular iteration count")
    if np.any(results["full_stopping_relative_update_max"] > 1.0e-4):
        raise AssertionError("source stopping ratio exceeds 1e-4")
    if not np.array_equal(
        results["full_input_charge_square_density"],
        results["full_charge_square_density"],
    ):
        raise AssertionError("molecular solve changed charge_square_density")
    doppler = results["full_fractional_doppler_widths"]
    structural_slots = results[
        "full_fractional_doppler_width_structural_infinity_slots"
    ]
    if not np.array_equal(
        structural_slots, np.asarray([919, 927], dtype=np.int64)
    ):
        raise AssertionError("Doppler structural infinity slots changed")
    expected_structural_mask = np.zeros((6, 1006), dtype=np.bool_)
    expected_structural_mask[:, structural_slots] = True
    structural_mask = results[
        "full_fractional_doppler_width_structural_infinity_mask"
    ]
    if not np.array_equal(structural_mask, expected_structural_mask):
        raise AssertionError("stored Doppler structural infinity mask changed")
    actual_nonfinite = np.logical_not(np.isfinite(doppler))
    if not np.array_equal(actual_nonfinite, expected_structural_mask):
        raise AssertionError("unexpected Doppler nonfinite structure")
    if not np.all(np.isposinf(doppler[structural_mask])):
        raise AssertionError("known structural Doppler entries must be +inf")
    if not np.all(np.isfinite(doppler[np.logical_not(structural_mask)])):
        raise AssertionError("Doppler widths outside structural slots must be finite")
    if np.any(results["full_major_isotope_mass_amu"][structural_slots]):
        raise AssertionError("structural Doppler slots must have zero isotope mass")
    if np.any(
        results["full_ion_stage_populations_by_packed_slot"][
            structural_mask
        ]
    ) or np.any(
        results["full_partition_normalized_populations_by_packed_slot"][
            structural_mask
        ]
    ):
        raise AssertionError("structural Doppler slots must have zero populations")
    if not np.array_equal(
        results["full_equation_densities_before_normalization"],
        results["full_previous_molecular_equation_densities"],
    ):
        raise AssertionError("saved physical equation densities changed")
    if not np.array_equal(
        results["full_equation_densities_after_normalization"],
        results["full_transformed_molecular_equation_densities"],
    ):
        raise AssertionError("transformed equation-density capture changed")
    if not np.array_equal(
        results["full_raw_populations_before_fill"],
        results["full_molecular_populations"],
    ):
        raise AssertionError("raw molecular population capture changed")
    if not np.array_equal(
        results["full_normalized_populations_after_fill"],
        results["full_partition_normalized_molecular_populations"],
    ):
        raise AssertionError("normalized molecular population capture changed")

    if int(results["handoff_solve_call_count"]) != 1:
        raise AssertionError("structured handoff must solve molecules once")
    if int(results["handoff_fill_call_count"]) != 1:
        raise AssertionError("structured handoff must normalize once")
    if not np.array_equal(
        results["handoff_input_electron_density"],
        results["full_output_electron_density"],
    ):
        raise AssertionError("structured handoff electron input changed")
    if np.array_equal(
        results["handoff_input_electron_density"],
        results["handoff_output_electron_density"],
    ):
        raise AssertionError("structured handoff incorrectly preserved electrons")
    if not np.array_equal(results["handoff_layer_order"], np.arange(6)):
        raise AssertionError("structured handoff depth order changed")
    if not np.all(results["handoff_newton_converged"]):
        raise AssertionError("structured handoff contains a nonconverged layer")
    if np.any(results["handoff_newton_exhausted"]):
        raise AssertionError("structured handoff exhausted Newton iterations")
    if int(results["handoff_temperature_iteration_cache_value"]) != 1:
        raise AssertionError("structured handoff cache value changed")
    if not np.array_equal(
        results["handoff_charge_square_density"],
        _expected_charge_square_density(
            results["full_temperature"],
            results["full_gas_pressure"],
            results["handoff_input_electron_density"],
        ),
    ):
        raise AssertionError("structured handoff charge-square seed changed")
    if not np.array_equal(
        results["handoff_equation_densities_after_normalization"],
        results["handoff_transformed_molecular_equation_densities"],
    ):
        raise AssertionError("structured handoff transformed state changed")
    if not np.array_equal(
        results["handoff_normalized_populations_after_fill"],
        results["handoff_partition_normalized_molecular_populations"],
    ):
        raise AssertionError("structured handoff normalized state changed")

    if bool(results["disabled_pressure_iteration_enabled"]):
        raise AssertionError("disabled route resolved pressure iteration as enabled")
    for name in (
        "disabled_solve_call_count",
        "disabled_layer_call_count",
        "disabled_fill_call_count",
        "disabled_temperature_iteration_cache_key_count",
    ):
        if int(results[name]) != 0:
            raise AssertionError(f"disabled route unexpectedly changed {name}")
    if bool(
        results["disabled_temperature_iteration_cache_pops_itemp_present"]
    ):
        raise AssertionError("disabled route populated the temperature cache")
    if not np.array_equal(
        results["disabled_input_electron_density"],
        results["disabled_output_electron_density"],
    ):
        raise AssertionError("disabled route changed electron density")
    if not np.array_equal(
        results["disabled_charge_square_density"],
        results["full_input_charge_square_density"],
    ):
        raise AssertionError("disabled route charge-square seed changed")
    for name in (
        "disabled_ion_stage_populations_by_packed_slot",
        "disabled_molecular_equation_densities",
        "disabled_molecular_populations",
        "disabled_partition_normalized_molecular_populations",
        "disabled_partition_normalized_population_over_mass_density_and_width",
        "disabled_partition_normalized_populations_by_packed_slot",
        "disabled_previous_molecular_equation_densities",
        "disabled_specific_internal_energy",
    ):
        if np.any(results[name]):
            raise AssertionError(f"disabled route unexpectedly populated {name}")

    for branch_doppler_name in (
        "handoff_fractional_doppler_widths",
        "disabled_fractional_doppler_widths",
    ):
        branch_doppler = results[branch_doppler_name]
        if not np.array_equal(
            np.logical_not(np.isfinite(branch_doppler)),
            expected_structural_mask,
        ):
            raise AssertionError(
                f"{branch_doppler_name} changed the structural infinity mask"
            )
        if not np.all(np.isposinf(branch_doppler[expected_structural_mask])):
            raise AssertionError(
                f"{branch_doppler_name} structural entries must be +inf"
            )
        if not np.all(
            np.isfinite(branch_doppler[np.logical_not(expected_structural_mask)])
        ):
            raise AssertionError(
                f"{branch_doppler_name} has an unexpected nonfinite entry"
            )

    for prefix, mode in (("mode2", 2), ("mode12", 12)):
        if int(results[f"{prefix}_population_mode"]) != mode:
            raise AssertionError(f"{prefix} population mode changed")
        if int(results[f"{prefix}_solve_call_count"]) != 1:
            raise AssertionError(f"{prefix} must solve once")
        if int(results[f"{prefix}_fill_call_count"]) != 0:
            raise AssertionError(f"{prefix} must return before normalized fill")
        if not np.all(results[f"{prefix}_newton_converged"]):
            raise AssertionError(f"{prefix} did not converge")
        if np.any(results[f"{prefix}_newton_exhausted"]):
            raise AssertionError(f"{prefix} exhausted Newton iterations")
        if np.any(results[f"{prefix}_previous_equation_densities_before"]):
            raise AssertionError(f"{prefix} did not start from fresh saved rows")
        if not np.array_equal(
            results[f"{prefix}_molecular_equation_densities_after"],
            results[f"{prefix}_previous_equation_densities_after"],
        ):
            raise AssertionError(f"{prefix} transformed physical equation rows")
        if np.any(results[f"{prefix}_normalized_populations_before"]) or np.any(
            results[f"{prefix}_normalized_populations_after"]
        ):
            raise AssertionError(f"{prefix} changed normalized populations")
        if np.any(results[f"{prefix}_specific_internal_energy_before"]) or np.any(
            results[f"{prefix}_specific_internal_energy_after"]
        ):
            raise AssertionError(f"{prefix} changed specific internal energy")
        if not np.any(results[f"{prefix}_raw_populations_after"] > 0.0):
            raise AssertionError(f"{prefix} did not calculate raw populations")

    if int(results["energy_solve_call_count"]) != 1:
        raise AssertionError("specific-energy mode must solve once")
    if int(results["energy_fill_call_count"]) != 0:
        raise AssertionError("specific-energy mode must skip normalized fill")
    if not bool(results["energy_specific_internal_energy_mode_enabled"]):
        raise AssertionError("specific-energy mode flag was not enabled")
    if not np.all(results["energy_newton_converged"]):
        raise AssertionError("specific-energy mode contains a nonconverged layer")
    if np.any(results["energy_newton_exhausted"]):
        raise AssertionError("specific-energy mode exhausted Newton iterations")
    if not np.array_equal(
        results["energy_saved_physical_rows_input"],
        results["full_previous_molecular_equation_densities"],
    ):
        raise AssertionError("specific-energy saved-row source changed")
    if not np.array_equal(
        results["energy_actual_seed_override"],
        results["energy_saved_physical_rows_input"],
    ) or not np.array_equal(
        results["energy_layer_seed_equation_densities"],
        results["energy_saved_physical_rows_input"],
    ):
        raise AssertionError("specific-energy solve did not use saved rows")
    if np.array_equal(
        results["energy_actual_seed_override"],
        results["energy_ordinary_continuation_seed"],
    ):
        raise AssertionError("specific-energy seed override was not exercised")
    if not np.array_equal(
        results["energy_saved_physical_rows_before"],
        results["energy_saved_physical_rows_input"],
    ) or not np.array_equal(
        results["energy_saved_physical_rows_after"],
        results["energy_saved_physical_rows_input"],
    ):
        raise AssertionError("specific-energy mode changed saved physical rows")
    if not np.array_equal(
        results["energy_normalized_populations_before"],
        results["full_partition_normalized_molecular_populations"],
    ) or not np.array_equal(
        results["energy_normalized_populations_after"],
        results["energy_normalized_populations_before"],
    ):
        raise AssertionError("specific-energy mode changed normalized arrays")
    if np.any(results["energy_specific_internal_energy_before"]):
        raise AssertionError("specific-energy runtime did not start fresh")
    if not np.array_equal(
        results["energy_specific_internal_energy_after"],
        results["energy_direct_specific_internal_energy_reference"],
    ):
        raise AssertionError("specific-energy output changed")
    if not np.all(results["energy_specific_internal_energy_after"] > 0.0):
        raise AssertionError("specific-energy output must be positive")

    if str(results["bridge_live_shape_error_type"]) != "ValueError":
        raise AssertionError("live/debug bridge error type changed")
    if str(results["bridge_live_shape_error_message"]) != (
        "molecular_populations must have shape (n_depth, n_molecule) "
        "matching molecule_codes"
    ):
        raise AssertionError(
            "live/debug bridge shape error changed: "
            f"{str(results['bridge_live_shape_error_message'])!r}"
        )
    if not np.array_equal(
        results["bridge_padded_molecule_code_shape"],
        np.asarray([200], dtype=np.int64),
    ) or not np.array_equal(
        results["bridge_active_molecular_population_shape"],
        np.asarray([6, 170], dtype=np.int64),
    ):
        raise AssertionError("live/debug bridge shape seam changed")
    if not np.array_equal(
        results["bridge_h2_mixed_output"],
        results["bridge_h2_mixed_catalog_input"][:, 0],
    ):
        raise AssertionError("mixed H2 vector was filled depth by depth")
    if not np.array_equal(
        results["bridge_h2_all_zero_output"],
        results["bridge_h2_no_catalog_output"],
    ):
        raise AssertionError("all-zero H2 vector did not select analytic fallback")
    fallback_h2 = results["bridge_h2_all_zero_output"]
    if (
        fallback_h2[0] <= 0.0
        or fallback_h2[1] != 0.0
        or fallback_h2[2] <= 0.0
    ):
        raise AssertionError("analytic H2 fallback 9000 K gate changed")

    for prefix, count in (
        ("temperature_control", 3),
        ("pressure_control", 3),
        ("boundary", 4),
    ):
        if results[f"{prefix}_newton_iteration_count"].shape != (count,):
            raise AssertionError(f"{prefix} control count changed")
        if not np.all(results[f"{prefix}_newton_converged"]):
            raise AssertionError(f"{prefix} contains a nonconverged solve")
        if np.any(results[f"{prefix}_newton_exhausted"]):
            raise AssertionError(f"{prefix} exhausted Newton iterations")

    named_codes = np.rint(results["named_molecule_codes"]).astype(np.int64)
    named_indices = results["named_molecule_catalog_indices"]
    code_to_index = {
        int(code): int(index) for code, index in zip(named_codes, named_indices)
    }
    boundary_raw = results["boundary_raw_named_populations"]
    boundary_normalized = results["boundary_normalized_named_populations"]
    co_index = code_to_index[608]
    h2_index = code_to_index[101]
    if not boundary_raw[0, co_index] > 0.0 or boundary_raw[1, co_index] != 0.0:
        raise AssertionError("10000 K polynomial gate changed")
    if not boundary_raw[2, h2_index] > 0.0 or boundary_raw[3, h2_index] != 0.0:
        raise AssertionError("20000 K H2 activity gate changed")
    if boundary_normalized[1, co_index] == 0.0:
        raise AssertionError("normalized polynomial population was incorrectly gated")
    if boundary_normalized[3, h2_index] == 0.0:
        raise AssertionError("normalized H2 population was incorrectly gated")

    boundary_temperature = np.asarray(
        results["boundary_temperature"], dtype=np.float64
    )
    if not np.array_equal(
        results["boundary_temperature_uint64_bits"],
        boundary_temperature.view(np.uint64),
    ):
        raise AssertionError("boundary temperature bit patterns changed")
    polynomial_active = boundary_temperature <= 10000.0
    h2_active = boundary_temperature <= 20000.0
    for name, expected in (
        ("boundary_polynomial_active_branch_mask", polynomial_active),
        (
            "boundary_polynomial_inactive_branch_mask",
            np.logical_not(polynomial_active),
        ),
        ("boundary_h2_catalog_active_branch_mask", h2_active),
        (
            "boundary_h2_catalog_inactive_branch_mask",
            np.logical_not(h2_active),
        ),
    ):
        if not np.array_equal(results[name], expected):
            raise AssertionError(f"{name} changed")
    if (
        int(results["boundary_temperature_uint64_bits"][1])
        != int(results["boundary_temperature_uint64_bits"][0]) + 1
        or int(results["boundary_temperature_uint64_bits"][3])
        != int(results["boundary_temperature_uint64_bits"][2]) + 1
    ):
        raise AssertionError("boundary nextafter bit adjacency changed")

    h2_probe_temperature = np.asarray(
        results["h2_probe_temperature"], dtype=np.float64
    )
    if not np.array_equal(
        results["h2_probe_temperature_uint64_bits"],
        h2_probe_temperature.view(np.uint64),
    ):
        raise AssertionError("H2 helper temperature bit patterns changed")
    low_clamp = h2_probe_temperature <= 100.0
    interpolate = np.logical_and(
        h2_probe_temperature > 100.0,
        h2_probe_temperature < 19900.0,
    )
    high_clamp = h2_probe_temperature >= 19900.0
    catalog_active = h2_probe_temperature <= 20000.0
    for name, expected in (
        ("h2_probe_partition_low_clamp_branch_mask", low_clamp),
        ("h2_probe_partition_interpolation_branch_mask", interpolate),
        ("h2_probe_partition_high_clamp_branch_mask", high_clamp),
        (
            "h2_probe_helper_finite_positive_input_branch_mask",
            np.ones(h2_probe_temperature.shape, dtype=np.bool_),
        ),
        ("h2_probe_catalog_active_branch_mask", catalog_active),
        (
            "h2_probe_catalog_inactive_branch_mask",
            np.logical_not(catalog_active),
        ),
    ):
        if not np.array_equal(results[name], expected):
            raise AssertionError(f"{name} changed")
    if int(results["h2_probe_temperature_uint64_bits"][-1]) != (
        int(results["h2_probe_temperature_uint64_bits"][-2]) + 1
    ):
        raise AssertionError("H2 nextafter bit adjacency changed")
    if not np.array_equal(
        results["h2_probe_catalog_equilibrium_constant_gated"] != 0.0,
        catalog_active,
    ):
        raise AssertionError("H2 catalog activity mask does not match values")
    for value_name, bit_name in (
        (
            "boundary_frozen_newton_constants",
            "boundary_frozen_newton_constants_uint64_bits",
        ),
        (
            "boundary_normalized_named_populations",
            "boundary_normalized_named_populations_uint64_bits",
        ),
        (
            "boundary_population_constants",
            "boundary_population_constants_uint64_bits",
        ),
        (
            "boundary_raw_named_populations",
            "boundary_raw_named_populations_uint64_bits",
        ),
        (
            "h2_probe_catalog_equilibrium_constant_gated",
            "h2_probe_catalog_equilibrium_constant_gated_uint64_bits",
        ),
        (
            "h2_probe_helper_equilibrium_constant_ungated",
            "h2_probe_helper_equilibrium_constant_ungated_uint64_bits",
        ),
        (
            "h2_probe_interpolated_partition",
            "h2_probe_interpolated_partition_uint64_bits",
        ),
    ):
        if not np.array_equal(
            results[bit_name],
            np.asarray(results[value_name], dtype=np.float64).view(np.uint64),
        ):
            raise AssertionError(f"{bit_name} changed")

    executed_count = int(results["executed_atmosphere_source_count"])
    for name in (
        "executed_atmosphere_source_module_names",
        "executed_atmosphere_source_relative_paths",
        "executed_atmosphere_source_sha256",
    ):
        if results[name].shape != (executed_count,):
            raise AssertionError(f"{name} count changed")
    if len(set(results["executed_atmosphere_source_module_names"].tolist())) != (
        executed_count
    ):
        raise AssertionError("executed atmosphere module names are not unique")
    if not all(
        len(str(value)) == 64
        for value in results["executed_atmosphere_source_sha256"]
    ):
        raise AssertionError("executed atmosphere source hash shape changed")


def result_fingerprint(results: Mapping[str, np.ndarray]) -> str:
    """Hash sorted in-memory NPY members without creating an NPZ product."""

    digest = hashlib.sha256()
    for name in sorted(results):
        buffer = io.BytesIO()
        np.lib.format.write_array(
            buffer,
            np.asarray(results[name]),
            allow_pickle=False,
        )
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(buffer.getvalue())
    return digest.hexdigest()


def summarize_results(results: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Return a small JSON-safe self-check report, never the physical arrays."""

    return {
        "boundary_shape": list(results["boundary_raw_named_populations"].shape),
        "boundary_temperature_uint64_bits": [
            int(value) for value in results["boundary_temperature_uint64_bits"]
        ],
        "boundary_polynomial_active_branch_mask": [
            bool(value)
            for value in results["boundary_polynomial_active_branch_mask"]
        ],
        "boundary_h2_catalog_active_branch_mask": [
            bool(value)
            for value in results["boundary_h2_catalog_active_branch_mask"]
        ],
        "bridge_live_shape_error_type": str(
            results["bridge_live_shape_error_type"]
        ),
        "capture_scope_complete": bool(results["capture_scope_complete"]),
        "capture_in_memory_scope_complete": bool(
            results["capture_in_memory_scope_complete"]
        ),
        "deferred_fields": list(DEFERRED_CAPTURE_FIELDS),
        "disabled_solve_call_count": int(
            results["disabled_solve_call_count"]
        ),
        "doppler_structural_infinity_count": int(
            np.count_nonzero(
                results[
                    "full_fractional_doppler_width_structural_infinity_mask"
                ]
            )
        ),
        "executed_atmosphere_source_count": int(
            results["executed_atmosphere_source_count"]
        ),
        "fingerprint": result_fingerprint(results),
        "full_equation_shape": list(
            results["full_transformed_molecular_equation_densities"].shape
        ),
        "full_iteration_count": [
            int(value) for value in results["full_newton_iteration_count"]
        ],
        "full_newton_history_count": int(
            results["full_newton_history_layer_offsets"][-1]
        ),
        "full_np_linalg_lstsq_call_count": int(
            results["full_newton_np_linalg_lstsq_call_count"]
        ),
        "full_np_linalg_solve_call_count": int(
            results["full_newton_np_linalg_solve_call_count"]
        ),
        "full_molecule_shape": list(results["full_molecular_populations"].shape),
        "full_solve_call_count": int(results["full_solve_call_count"]),
        "handoff_iteration_count": [
            int(value) for value in results["handoff_newton_iteration_count"]
        ],
        "handoff_solve_call_count": int(results["handoff_solve_call_count"]),
        "key_count": len(results),
        "mode12_fill_call_count": int(results["mode12_fill_call_count"]),
        "mode12_solve_call_count": int(results["mode12_solve_call_count"]),
        "mode2_fill_call_count": int(results["mode2_fill_call_count"]),
        "mode2_solve_call_count": int(results["mode2_solve_call_count"]),
        "numba_thread_count": int(results["numba_thread_count"]),
        "population_schedule_wrappers_restored": bool(
            results["capture_population_schedule_wrappers_restored"]
        ),
        "priming_cache_additional_solve_count_during_schedule": int(
            results["priming_cache_additional_solve_count_during_schedule"]
        ),
        "priming_cache_molecular_solve_count_during_zero_code": int(
            results["priming_cache_molecular_solve_count_during_zero_code"]
        ),
        "pressure_control_shape": list(
            results["pressure_control_raw_named_populations"].shape
        ),
        "temperature_control_shape": list(
            results["temperature_control_raw_named_populations"].shape
        ),
        "schedule_inventory_atomic_job_count": int(
            results["schedule_inventory_atomic_job_count"]
        ),
        "schedule_inventory_job_count": int(
            results["schedule_inventory_job_count"]
        ),
        "schedule_inventory_molecular_job_count": int(
            results["schedule_inventory_molecular_job_count"]
        ),
        "schedule_inventory_molecular_unique_code": [
            float(value)
            for value in results[
                "schedule_inventory_molecular_unique_code"
            ]
        ],
        "schedule_inventory_molecular_unique_packed_start_slot_one_based": [
            int(value)
            for value in results[
                "schedule_inventory_"
                "molecular_unique_packed_start_slot_one_based"
            ]
        ],
        "energy_iteration_count": [
            int(value) for value in results["energy_newton_iteration_count"]
        ],
        "energy_solve_call_count": int(results["energy_solve_call_count"]),
        "wrappers_restored": bool(results["capture_wrappers_restored"]),
        "numpy_linalg_wrappers_restored": bool(
            results["capture_numpy_linalg_wrappers_restored"]
        ),
        "h2_probe_temperature_uint64_bits": [
            int(value) for value in results["h2_probe_temperature_uint64_bits"]
        ],
        "h2_probe_partition_interpolation_branch_mask": [
            bool(value)
            for value in results[
                "h2_probe_partition_interpolation_branch_mask"
            ]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="self-check the unpublished Chapter 4 atmosphere oracle"
    )
    parser.add_argument("--pinned-root", type=Path, default=PINNED_ROOT)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail closed until every deferred capture is implemented",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    results = build_atmosphere_oracle_results(
        pinned_root=arguments.pinned_root,
        fixture_path=arguments.fixture,
        require_complete=arguments.require_complete,
    )
    print(json.dumps(summarize_results(results), sort_keys=True))


if __name__ == "__main__":
    main()
