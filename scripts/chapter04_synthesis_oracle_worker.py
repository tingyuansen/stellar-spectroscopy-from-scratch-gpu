#!/usr/bin/env python3
"""Capture Chapter 4 synthesis chemistry from the pinned Payne Zero checkout.

This module deliberately builds in-memory arrays only.  A later, separately
reviewed publisher may write those arrays as comparison-only goldens.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping, Optional


THREAD_ENVIRONMENT = {
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

ORACLE_PROCESS_ENVIRONMENT = {
    **THREAD_ENVIRONMENT,
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "MKL_DYNAMIC": "FALSE",
    "LC_ALL": "C",
    "TZ": "UTC",
}

# A directly executed worker controls BLAS before importing NumPy.  Imported
# callers must instead establish and verify the environment explicitly.
if __name__ == "__main__":
    os.environ.update(THREAD_ENVIRONMENT)

import numpy as np  # noqa: E402  (thread controls must precede this import)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_ROOT = Path("/Users/ysting/payne-zero")
PINNED_DATA_ROOT = PINNED_ROOT / "source_data_files"
PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
DEFAULT_FIXTURE = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter04_molecular_inputs.npz"
)
FIXTURE_SHA256 = (
    "351bba75dca1fa502f5cc2a108035f69f2e31c760a90133480f2e7fe31e45f79"
)
MOLECULE_CATALOG = (
    PINNED_DATA_ROOT
    / "source_catalogs"
    / "lines"
    / "molecular_equilibrium_synthesis.npz"
)
ATMOSPHERE_MOLECULE_CATALOG = (
    PINNED_DATA_ROOT
    / "source_catalogs"
    / "lines"
    / "molecular_equilibrium_atmosphere.npz"
)
EOS_TABLES = (
    PINNED_DATA_ROOT
    / "synthesis_tables"
    / "partition_saha_inputs.npz"
)
ATOMIC_MASSES = (
    PINNED_DATA_ROOT / "synthesis_tables" / "atomic_masses.npz"
)
CONTINUUM_EDGE_GRID = (
    PINNED_DATA_ROOT / "synthesis_tables" / "continuum_edge_grid.npz"
)
ATMOSPHERE_SCHEMA = (
    PINNED_ROOT / "payne_zero_synthesis" / "atmosphere_schema.json"
)

EXPECTED_SOURCE_SHA256 = {
    "molecular_equilibrium_source": (
        PINNED_ROOT / "payne_zero_synthesis" / "molecular_equilibrium.py",
        "df01757c160b2bff4390cc2148cff9d1ba6e5a2bc7cab4515b46f38e868d2714",
    ),
    "equation_of_state_source": (
        PINNED_ROOT / "payne_zero_synthesis" / "equation_of_state.py",
        "6497c29abb954e0b55d918cc22fa7b660952812c548faf1d7b1053345ef13562",
    ),
    "pipeline_source": (
        PINNED_ROOT / "payne_zero_synthesis" / "pipeline.py",
        "465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22",
    ),
    "molecule_catalog": (
        MOLECULE_CATALOG,
        "3e8c1ea69fe672b9886bda38922f868c6d2ac2b43c4eb0d7750620241c238d28",
    ),
    "atmosphere_molecule_catalog": (
        ATMOSPHERE_MOLECULE_CATALOG,
        "971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644",
    ),
    "eos_tables": (
        EOS_TABLES,
        "0e235e7f1edecf39630690f4c68f4fc952f55785a08174562bc9575100fc4e27",
    ),
    "atomic_masses": (
        ATOMIC_MASSES,
        "d4739fef7e03964aea5a7b2604f9585fd9095c26c58f5b7d5d040aaafeb5d117",
    ),
    "continuum_edge_grid": (
        CONTINUUM_EDGE_GRID,
        "11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e",
    ),
    "atmosphere_schema": (
        ATMOSPHERE_SCHEMA,
        "2ba8d637e613be12ff43ce319a752616323f0341ea69f8e2391c3c244939777a",
    ),
}

FIXTURE_KEYS = frozenset(
    {
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
)

FULL_ROUTE_CALLERS = (
    "solve_electron_density",
    "_molecule_backed_population_state",
)
FIXED_ROUTE_CALLERS = ("_molecule_backed_population_state",)
EOS_MODULE_NAME = "payne_zero_synthesis.equation_of_state"
PUBLIC_BUILDER_PARAMETERS = (
    "temperature",
    "column_mass",
    "gas_pressure",
    "electron_density",
    "elemental_abundances",
    "mean_nuclear_mass_amu",
    "microturbulence",
    "eos_tables",
    "electron_density_seed",
    "tol",
    "atomic_masses",
    "mass_density",
    "molecular_species_codes",
    "molecules_path",
)
EXPECTED_SYNTHESIS_ONLY_CODES = np.asarray(
    [
        111.0,
        10811.0,
        10812.0,
        10820.0,
        60606.0,
        60608.0,
        60614.0,
        60816.0,
        61414.0,
        61616.0,
        70708.0,
        70808.0,
        80814.0,
        80816.0,
        1010106.0,
        1010107.0,
        1010606.0,
        6060707.0,
        101010106.0,
        101010114.0,
    ],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceIdentity:
    """Verified source and data identity for one oracle process."""

    root: Path
    commit: str
    sha256_by_name: Mapping[str, str]


def verify_pinned_source_identity(root: Path = PINNED_ROOT) -> SourceIdentity:
    """Fail unless ``root`` is the one frozen checkout and every hash matches."""

    resolved = Path(root).expanduser().resolve()
    if resolved != PINNED_ROOT.resolve():
        raise ValueError(
            "Chapter 4 synthesis oracles permit only "
            f"{PINNED_ROOT.resolve()}, not {resolved}"
        )
    completed = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(
            f"pinned checkout is {commit}; expected {PINNED_COMMIT}"
        )

    actual_hashes: dict[str, str] = {}
    for name, (path, expected) in EXPECTED_SOURCE_SHA256.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"{path} has SHA-256 {actual}; expected {expected}"
            )
        actual_hashes[name] = actual
    return SourceIdentity(resolved, commit, actual_hashes)


def require_oracle_process_environment() -> None:
    """Require deterministic process controls that cannot be repaired late."""

    mismatches = {
        name: os.environ.get(name)
        for name, value in ORACLE_PROCESS_ENVIRONMENT.items()
        if os.environ.get(name) != value
    }
    if mismatches:
        rendered = ", ".join(
            f"{name}={value!r}" for name, value in sorted(mismatches.items())
        )
        raise RuntimeError(
            "fresh oracle process controls are missing or wrong: " + rendered
        )


def load_input_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, np.ndarray]:
    """Load and validate the frozen input-only molecular fixture."""

    resolved = Path(path).expanduser().resolve()
    actual_hash = sha256(resolved)
    if resolved == DEFAULT_FIXTURE.resolve() and actual_hash != FIXTURE_SHA256:
        raise RuntimeError(
            f"{resolved} has SHA-256 {actual_hash}; expected {FIXTURE_SHA256}"
        )
    with np.load(resolved, allow_pickle=False) as archive:
        names = set(archive.files)
        if names != FIXTURE_KEYS:
            missing = sorted(FIXTURE_KEYS - names)
            unexpected = sorted(names - FIXTURE_KEYS)
            raise ValueError(
                "Chapter 4 input fixture key mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        arrays = {
            name: np.asarray(archive[name]).copy() for name in sorted(names)
        }

    temperature = arrays["temperature"]
    gas_pressure = arrays["gas_pressure"]
    electron_density = arrays["electron_density_seed"]
    abundance = arrays["elemental_abundances"]
    if temperature.shape != (6,) or gas_pressure.shape != (6,):
        raise ValueError("the controlled thermochemical track must have six depths")
    if electron_density.shape != temperature.shape:
        raise ValueError("electron_density_seed must follow the depth axis")
    if abundance.shape != (99,):
        raise ValueError("elemental_abundances must be depth independent, shape (99,)")
    numeric_inputs = (temperature, gas_pressure, electron_density, abundance)
    if any(values.dtype != np.dtype(np.float64) for values in numeric_inputs):
        raise TypeError("scientific fixture inputs must be NumPy float64")
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature must be finite and strictly positive")
    if np.any(~np.isfinite(gas_pressure)) or np.any(gas_pressure <= 0.0):
        raise ValueError("gas_pressure must be finite and strictly positive")
    if np.any(np.diff(gas_pressure) <= 0.0):
        raise ValueError("gas_pressure must follow the intended outer-to-inner order")
    if np.any(~np.isfinite(electron_density)) or np.any(electron_density <= 0.0):
        raise ValueError("electron_density_seed must be finite and positive")
    if np.any(~np.isfinite(abundance)) or np.any(abundance < 0.0):
        raise ValueError("elemental_abundances must be finite and nonnegative")
    if float(abundance.sum()) <= 0.0:
        raise ValueError("elemental_abundances must have a positive sum")
    return arrays


@dataclass(frozen=True)
class PinnedRuntime:
    """Imported pinned modules and one CPU-float64 EOS table bundle."""

    identity: SourceIdentity
    torch: ModuleType
    eos: ModuleType
    molecular: ModuleType
    tables: Any


def _assert_module_under_root(module: ModuleType, root: Path) -> None:
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(root.resolve()):
        raise RuntimeError(
            f"{module.__name__} resolved outside pinned checkout: {module_path}"
        )


def load_pinned_runtime(
    root: Path = PINNED_ROOT,
    *,
    require_process_controls: bool = True,
) -> PinnedRuntime:
    """Import the exact synthesis modules and create CPU-float64 tables."""

    if require_process_controls:
        require_oracle_process_environment()
    identity = verify_pinned_source_identity(root)

    for name, module in tuple(sys.modules.items()):
        if name != "payne_zero_synthesis" and not name.startswith(
            "payne_zero_synthesis."
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None or not Path(module_file).resolve().is_relative_to(
            identity.root
        ):
            raise RuntimeError(
                f"{name} was imported before the pinned oracle path: {module_file}"
            )

    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(PINNED_DATA_ROOT)
    os.environ["PAYNE_ZERO_SYNTHESIS_SOURCE_CATALOG_ROOT"] = str(
        PINNED_DATA_ROOT / "source_catalogs"
    )
    os.environ["PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE"] = str(ATOMIC_MASSES)
    root_text = str(identity.root)
    if not sys.path or sys.path[0] != root_text:
        sys.path.insert(0, root_text)

    torch = importlib.import_module("torch")
    eos = importlib.import_module("payne_zero_synthesis.equation_of_state")
    molecular = importlib.import_module(
        "payne_zero_synthesis.molecular_equilibrium"
    )
    package = importlib.import_module("payne_zero_synthesis")
    for module in (package, eos, molecular):
        _assert_module_under_root(module, identity.root)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Torch did not accept the one-thread oracle policy")

    device = torch.device("cpu")
    tables = eos.EOSTables.from_npz(
        EOS_TABLES, device=device, dtype=torch.float64
    )
    if tables.device.type != "cpu" or tables.dtype != torch.float64:
        raise RuntimeError("EOS tables are not explicit CPU float64")
    for name, module in tuple(sys.modules.items()):
        if name == "payne_zero_synthesis" or name.startswith(
            "payne_zero_synthesis."
        ):
            _assert_module_under_root(module, identity.root)
    return PinnedRuntime(identity, torch, eos, molecular, tables)


@dataclass
class CapturedMolecularCall:
    """One untouched molecular solve plus diagnostics requested by the wrapper."""

    caller_name: str
    caller_module: str
    caller_file: str
    effective_arguments: dict[str, Any]
    outputs: tuple[Any, Any, Any, Any]
    diagnostics: dict[str, Any]
    exhausted_mask: np.ndarray


def _exhaustion_assignment_line(function: Any) -> int:
    """Locate the exact ``for ... else`` exhaustion assignment."""

    source, first_line = inspect.getsourcelines(function)
    matches = [
        first_line + offset
        for offset, line in enumerate(source)
        if line.strip() == "converged_iters[depth_index] = max_iter"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "could not identify the unique molecular-iteration exhaustion line"
        )
    return matches[0]


class MolecularSolveCapture(AbstractContextManager["MolecularSolveCapture"]):
    """Temporarily request diagnostics without changing returned EOS values."""

    def __init__(
        self,
        molecular_module: ModuleType,
        *,
        trace_exhaustion: bool = True,
    ) -> None:
        self.module = molecular_module
        self.original = molecular_module.solve_molecular_equilibrium
        self.trace_exhaustion = trace_exhaustion
        self.calls: list[CapturedMolecularCall] = []
        self._exhaustion_line = (
            _exhaustion_assignment_line(self.original)
            if trace_exhaustion
            else -1
        )

    def __enter__(self) -> "MolecularSolveCapture":
        self.module.solve_molecular_equilibrium = self._wrapped
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Optional[bool]:
        self.module.solve_molecular_equilibrium = self.original
        return None

    def _call_with_exhaustion_trace(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, list[int]]:
        if not self.trace_exhaustion:
            return self.original(*args, **kwargs), []

        exhausted_depths: list[int] = []
        previous_trace = sys.gettrace()

        def local_trace(frame, event, argument):
            if event == "line" and frame.f_lineno == self._exhaustion_line:
                exhausted_depths.append(int(frame.f_locals["depth_index"]))
            return local_trace

        def global_trace(frame, event, argument):
            if frame.f_code is self.original.__code__:
                return local_trace
            if previous_trace is not None:
                return previous_trace(frame, event, argument)
            return None

        sys.settrace(global_trace)
        try:
            result = self.original(*args, **kwargs)
        finally:
            sys.settrace(previous_trace)
        return result, exhausted_depths

    def _wrapped(self, *args, **kwargs):
        caller = inspect.currentframe().f_back
        caller_name = caller.f_code.co_name
        caller_module = str(caller.f_globals.get("__name__", ""))
        caller_file = str(Path(caller.f_code.co_filename).resolve())

        bound = inspect.signature(self.original).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        effective_arguments = dict(bound.arguments)
        caller_requested_diagnostics = bool(
            effective_arguments["return_diagnostics"]
        )
        diagnostic_kwargs = dict(kwargs)
        diagnostic_kwargs["return_diagnostics"] = True
        result, exhausted_depths = self._call_with_exhaustion_trace(
            args, diagnostic_kwargs
        )
        if not isinstance(result, tuple) or len(result) != 5:
            raise RuntimeError("diagnostic molecular solve did not return five items")
        outputs = result[:4]
        diagnostics = result[4]
        depth_count = int(np.asarray(effective_arguments["temperature"]).size)
        exhausted_mask = np.zeros(depth_count, dtype=np.bool_)
        for depth_index in exhausted_depths:
            exhausted_mask[depth_index] = True
        self.calls.append(
            CapturedMolecularCall(
                caller_name=caller_name,
                caller_module=caller_module,
                caller_file=caller_file,
                effective_arguments=effective_arguments,
                outputs=outputs,
                diagnostics=diagnostics,
                exhausted_mask=exhausted_mask,
            )
        )
        return result if caller_requested_diagnostics else outputs


@dataclass
class CapturedFunctionCall:
    """One unmodified helper call made by the public structured builder."""

    caller_name: str
    caller_module: str
    caller_file: str
    effective_arguments: dict[str, Any]
    result: Any


def _captured_caller(
    *,
    extra_frames: int = 0,
) -> tuple[str, str, str]:
    """Return the caller above a wrapper method."""

    frame = inspect.currentframe()
    caller = frame
    for _ in range(2 + extra_frames):
        caller = caller.f_back
    return (
        caller.f_code.co_name,
        str(caller.f_globals.get("__name__", "")),
        str(Path(caller.f_code.co_filename).resolve()),
    )


class PublicBuilderCapture(AbstractContextManager["PublicBuilderCapture"]):
    """Capture the public builder's fixed/reuse/mapping/edge calls."""

    def __init__(
        self,
        eos_module: ModuleType,
        molecular_module: ModuleType,
        pipeline_module: ModuleType,
    ) -> None:
        self.eos = eos_module
        self.molecular = molecular_module
        self.pipeline = pipeline_module
        self.fixed_original = (
            eos_module.solve_population_state_at_electron_density
        )
        self.partition_original = eos_module.partition_functions_for_elements
        self.line_original = (
            molecular_module.molecular_line_populations_by_species_code
        )
        self.edge_original = pipeline_module._build_edge_grid
        self.fixed_calls: list[CapturedFunctionCall] = []
        self.partition_calls: list[CapturedFunctionCall] = []
        self.line_calls: list[CapturedFunctionCall] = []
        self.edge_calls: list[CapturedFunctionCall] = []
        self.edge_cache_present_before = hasattr(
            self.edge_original, "_cache"
        )
        self.edge_wrapper = None

    def __enter__(self) -> "PublicBuilderCapture":
        if self.edge_cache_present_before:
            raise RuntimeError(
                "public oracle requires a fresh process with no edge-grid cache"
            )
        self.eos.solve_population_state_at_electron_density = (
            self._fixed_wrapped
        )
        self.eos.partition_functions_for_elements = self._partition_wrapped
        self.molecular.molecular_line_populations_by_species_code = (
            self._line_wrapped
        )

        # The pinned loader refers to the module-global `_build_edge_grid`
        # object when storing its cache.  A closure remains a normal function
        # with a writable `__dict__`; a bound method would make that exact
        # source assignment fail.
        def edge_wrapper(*args, **kwargs):
            return self._edge_wrapped(*args, **kwargs)

        self.edge_wrapper = edge_wrapper
        self.pipeline._build_edge_grid = edge_wrapper
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Optional[bool]:
        self.pipeline._build_edge_grid = self.edge_original
        self.molecular.molecular_line_populations_by_species_code = (
            self.line_original
        )
        self.eos.partition_functions_for_elements = self.partition_original
        self.eos.solve_population_state_at_electron_density = (
            self.fixed_original
        )
        return None

    @staticmethod
    def _bound_arguments(function: Any, args, kwargs) -> dict[str, Any]:
        bound = inspect.signature(function).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def _fixed_wrapped(self, *args, **kwargs):
        caller_name, caller_module, caller_file = _captured_caller()
        arguments = self._bound_arguments(
            self.fixed_original, args, kwargs
        )
        result = self.fixed_original(*args, **kwargs)
        self.fixed_calls.append(
            CapturedFunctionCall(
                caller_name,
                caller_module,
                caller_file,
                arguments,
                result,
            )
        )
        return result

    def _partition_wrapped(self, *args, **kwargs):
        caller_name, caller_module, caller_file = _captured_caller()
        arguments = self._bound_arguments(
            self.partition_original, args, kwargs
        )
        result = self.partition_original(*args, **kwargs)
        copied_result = {
            int(code): np.asarray(values).copy()
            for code, values in result.items()
        }
        self.partition_calls.append(
            CapturedFunctionCall(
                caller_name,
                caller_module,
                caller_file,
                arguments,
                copied_result,
            )
        )
        return result

    def _line_wrapped(self, *args, **kwargs):
        caller_name, caller_module, caller_file = _captured_caller()
        arguments = self._bound_arguments(self.line_original, args, kwargs)
        copied_arguments = dict(arguments)
        for name in (
            "temperature",
            "equation_densities",
            "neutral_partition",
            "species_codes",
        ):
            copied_arguments[name] = np.asarray(arguments[name]).copy()
        result = self.line_original(*args, **kwargs)
        copied_result = {
            int(code): np.asarray(values).copy()
            for code, values in result.items()
        }
        self.line_calls.append(
            CapturedFunctionCall(
                caller_name,
                caller_module,
                caller_file,
                copied_arguments,
                copied_result,
            )
        )
        return result

    def _edge_wrapped(self, *args, **kwargs):
        caller_name, caller_module, caller_file = _captured_caller(
            extra_frames=1
        )
        arguments = self._bound_arguments(self.edge_original, args, kwargs)
        result = self.edge_original(*args, **kwargs)
        copied_result = {
            name: np.asarray(values).copy()
            for name, values in result.items()
        }
        self.edge_calls.append(
            CapturedFunctionCall(
                caller_name,
                caller_module,
                caller_file,
                arguments,
                copied_result,
            )
        )
        return result


def _as_numpy(value: Any) -> np.ndarray:
    """Copy a NumPy, Torch, scalar, or text value without object dtype."""

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).copy()
    if array.dtype.hasobject:
        raise TypeError("oracle arrays may not use object dtype")
    return array


def deterministic_result(
    arrays: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return copied, object-free arrays in stable lexical key order."""

    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise TypeError("oracle result keys must be nonempty strings")
        normalized[name] = _as_numpy(value)
    return {name: normalized[name] for name in sorted(normalized)}


def _residual_diagnostics(
    call: CapturedMolecularCall,
    molecular_module: ModuleType,
) -> dict[str, np.ndarray]:
    """Evaluate the exact final residual and pre-replacement molecular terms."""

    torch = molecular_module.torch
    arguments = call.effective_arguments
    diagnostics = call.diagnostics
    structure = diagnostics["structure"]
    equation_count = int(diagnostics["equation_count"])
    molecule_count = int(diagnostics["molecule_count"])
    equation_codes = np.asarray(
        diagnostics["equation_species_codes"], dtype=np.int64
    )
    abundance = np.asarray(arguments["elemental_abundances"], dtype=np.float64)
    temperature = np.asarray(arguments["temperature"], dtype=np.float64)
    gas_pressure = np.asarray(arguments["gas_pressure"], dtype=np.float64)
    equation_abundance = np.zeros(equation_count, dtype=np.float64)
    for index in range(1, equation_count):
        species_code = int(equation_codes[index])
        if species_code < 100:
            equation_abundance[index] = max(
                float(abundance[species_code - 1]), 1.0e-20
            )

    densities = call.outputs[2][:, :equation_count]
    log_formation = diagnostics["natural_log_formation_constants"][
        :, :molecule_count
    ]
    equation_abundance_t = torch.as_tensor(
        equation_abundance,
        dtype=densities.dtype,
        device=densities.device,
    )
    particle_density_t = torch.as_tensor(
        gas_pressure / (temperature * molecular_module.BOLTZMANN_ERG_PER_K),
        dtype=densities.dtype,
        device=densities.device,
    )

    residual_rows = []
    normalized_rows = []
    raw_log_terms = []
    raw_nonfinite = []
    with torch.no_grad():
        for depth_index in range(temperature.size):
            density = densities[depth_index]
            log_constant = log_formation[depth_index]
            particle_density = particle_density_t[depth_index]
            residual = molecular_module._residual(
                density,
                log_constant,
                equation_abundance_t,
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
                    - structure.inverse_electron_power
                    * log_density[electron_index]
                )
            raw_term = (
                torch.exp(log_term) * structure.active_molecule_mask
            )
            nonfinite = ~torch.isfinite(raw_term)
            finite_term = torch.where(
                nonfinite, torch.zeros_like(raw_term), raw_term
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
                    + (
                        equation_abundance_t[equation_index] * density[0]
                    ).abs()
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
                        + (
                            structure.inverse_electron_power * finite_term
                        )
                        .abs()
                        .sum()
                        + (
                            2.0 * structure.negative_ion_flag * finite_term
                        )
                        .abs()
                        .sum()
                    )
                scale[equation_index] = row_scale
            scale = scale.clamp_min(1.0)

            residual_rows.append(residual)
            normalized_rows.append(residual / scale)
            raw_log_terms.append(log_term)
            raw_nonfinite.append(nonfinite)

    residual_array = torch.stack(residual_rows)
    normalized_array = torch.stack(normalized_rows)
    log_term_array = torch.stack(raw_log_terms)
    nonfinite_array = torch.stack(raw_nonfinite)
    return {
        "equation_abundance": equation_abundance,
        "total_particle_density": particle_density_t,
        "residual": residual_array,
        "normalized_residual": normalized_array,
        "pre_replacement_log_term": log_term_array,
        "pre_replacement_term_nonfinite_mask": nonfinite_array,
        "pre_replacement_term_nonfinite_count": nonfinite_array.sum(dim=1),
        "pre_replacement_log_term_min": log_term_array.min(dim=1).values,
        "pre_replacement_log_term_max": log_term_array.max(dim=1).values,
    }


def _assert_call_structure(call: CapturedMolecularCall) -> None:
    """Require the exact active and padded synthesis extents."""

    diagnostics = call.diagnostics
    molecule_count = int(diagnostics["molecule_count"])
    equation_count = int(diagnostics["equation_count"])
    depth_count = int(
        np.asarray(call.effective_arguments["temperature"]).size
    )
    molecular_populations = _as_numpy(call.outputs[1])
    equation_densities = _as_numpy(call.outputs[2])
    if molecule_count != 190 or equation_count != 23:
        raise AssertionError(
            f"unexpected active synthesis extents: {molecule_count}, "
            f"{equation_count}"
        )
    if molecular_populations.shape != (depth_count, 200):
        raise AssertionError("molecular populations lost their padded shape")
    if equation_densities.shape != (depth_count, 30):
        raise AssertionError("equation densities lost their padded shape")
    if not np.array_equal(
        molecular_populations[:, molecule_count:],
        np.zeros((depth_count, 200 - molecule_count)),
    ):
        raise AssertionError("inactive molecular-population padding is not zero")
    if not np.array_equal(
        equation_densities[:, equation_count:],
        np.zeros((depth_count, 30 - equation_count)),
    ):
        raise AssertionError("inactive equation-density padding is not zero")
    for output in call.outputs:
        if np.any(~np.isfinite(_as_numpy(output))):
            raise AssertionError("molecular solve returned nonfinite output")
    if np.any(call.exhausted_mask):
        depths = np.flatnonzero(call.exhausted_mask).tolist()
        raise AssertionError(
            f"molecular Newton iteration exhausted at depths {depths}"
        )


def assert_route_call_ownership(
    route: str,
    calls: list[CapturedMolecularCall],
    *,
    enforce_pinned_caller: bool = True,
) -> None:
    """Require exact molecular solve count, order, and EOS ownership."""

    if route == "full":
        expected = FULL_ROUTE_CALLERS
    elif route == "fixed":
        expected = FIXED_ROUTE_CALLERS
    else:
        raise ValueError(f"unknown synthesis route {route!r}")
    actual = tuple(call.caller_name for call in calls)
    if actual != expected:
        raise AssertionError(
            f"{route} route molecular callers are {actual}; expected {expected}"
        )
    if enforce_pinned_caller:
        expected_file = str(
            (PINNED_ROOT / "payne_zero_synthesis" / "equation_of_state.py").resolve()
        )
        for call in calls:
            if call.caller_module != EOS_MODULE_NAME:
                raise AssertionError(
                    f"molecular caller module is {call.caller_module!r}"
                )
            if call.caller_file != expected_file:
                raise AssertionError(
                    f"molecular caller file is {call.caller_file!r}"
                )


def _serialize_call(
    prefix: str,
    call: CapturedMolecularCall,
    molecular_module: ModuleType,
) -> dict[str, Any]:
    """Convert one captured call and its exact diagnostics to flat arrays."""

    arguments = call.effective_arguments
    diagnostics = call.diagnostics
    residual = _residual_diagnostics(call, molecular_module)
    if np.any(~np.isfinite(_as_numpy(residual["residual"]))):
        raise AssertionError("final molecular residual is nonfinite")
    if np.any(~np.isfinite(_as_numpy(residual["normalized_residual"]))):
        raise AssertionError("normalized molecular residual is nonfinite")
    if np.any(_as_numpy(residual["pre_replacement_term_nonfinite_mask"])):
        raise AssertionError(
            "molecular residual required nonfinite-term replacement"
        )
    device = arguments["device"]
    dtype = arguments["dtype"]
    arrays: dict[str, Any] = {
        f"{prefix}caller_name": call.caller_name,
        f"{prefix}caller_module": call.caller_module,
        f"{prefix}caller_file": call.caller_file,
        f"{prefix}device": str(device),
        f"{prefix}dtype": str(dtype),
        f"{prefix}max_iter": arguments["max_iter"],
        f"{prefix}tol": arguments["tol"],
        f"{prefix}chain_length": (
            -1
            if arguments["chain_length"] is None
            else arguments["chain_length"]
        ),
        f"{prefix}molecules_path": str(
            Path(arguments["molecules_path"]).resolve()
        ),
        f"{prefix}input__temperature": arguments["temperature"],
        f"{prefix}input__gas_pressure": arguments["gas_pressure"],
        f"{prefix}input__electron_density": arguments["electron_density"],
        f"{prefix}input__elemental_abundances": arguments[
            "elemental_abundances"
        ],
        f"{prefix}input__ion_formation_constants": arguments[
            "ion_formation_constants"
        ],
        f"{prefix}output__total_nuclei_number_density": call.outputs[0],
        f"{prefix}output__molecular_populations": call.outputs[1],
        f"{prefix}output__equation_densities": call.outputs[2],
        f"{prefix}output__electron_density": call.outputs[3],
        f"{prefix}diag__molecule_count": diagnostics["molecule_count"],
        f"{prefix}diag__equation_count": diagnostics["equation_count"],
        f"{prefix}diag__equation_species_codes": diagnostics[
            "equation_species_codes"
        ],
        f"{prefix}diag__molecule_codes": diagnostics["molecule_codes"],
        f"{prefix}diag__iterations_completed": diagnostics[
            "iterations_completed"
        ],
        f"{prefix}diag__exhausted_mask": call.exhausted_mask,
        f"{prefix}diag__natural_log_formation_constants": diagnostics[
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
        arrays[f"{prefix}diag__structure__{name}"] = getattr(structure, name)
    arrays[f"{prefix}diag__structure__electron_equation_index"] = (
        structure.electron_equation_index
    )
    for name, values in residual.items():
        arrays[f"{prefix}diag__{name}"] = values
    return arrays


def _serialize_state(state: Any, prefix: str = "state__") -> dict[str, Any]:
    """Flatten the public population dataclass, including its nested EOS state."""

    if not is_dataclass(state):
        raise TypeError("population state must be a dataclass instance")
    arrays: dict[str, Any] = {}
    for field in fields(state):
        value = getattr(state, field.name)
        if field.name == "eos":
            if not is_dataclass(value):
                raise TypeError("population state's eos field is not a dataclass")
            for eos_field in fields(value):
                arrays[f"{prefix}eos__{eos_field.name}"] = getattr(
                    value, eos_field.name
                )
        elif value is None:
            raise AssertionError(
                f"molecule-enabled population field {field.name} is absent"
            )
        else:
            arrays[f"{prefix}{field.name}"] = value
    return arrays


def _catalog_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load one fixed-buffer molecular catalog without changing row order."""

    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.asarray(archive[name]).copy()
            for name in sorted(archive.files)
        }


def _component_semantic_codes(
    catalog: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode equation indices to species codes, preserving sentinel 101."""

    molecule_count = int(catalog["molecule_count"])
    equation_count = int(catalog["equation_count"])
    starts = np.asarray(catalog["component_start_indices"], dtype=np.int64)
    equation_indices = np.asarray(
        catalog["component_equation_indices"], dtype=np.int64
    )
    equation_species_codes = np.asarray(
        catalog["equation_species_codes"], dtype=np.int64
    )
    component_count = int(starts[molecule_count])
    active_indices = equation_indices[:component_count]
    if np.any(active_indices < 0) or np.any(active_indices > equation_count):
        raise AssertionError("catalog component equation index is out of range")
    sentinel_mask = active_indices == equation_count
    ordinary_indices = active_indices[~sentinel_mask]
    if np.any(ordinary_indices >= equation_species_codes.size):
        raise AssertionError("catalog equation table is too short")

    semantic = np.full(equation_indices.shape, -1, dtype=np.int32)
    semantic[:component_count][sentinel_mask] = 101
    semantic[:component_count][~sentinel_mask] = equation_species_codes[
        ordinary_indices
    ]
    padded_sentinel_mask = np.zeros(equation_indices.shape, dtype=np.bool_)
    padded_sentinel_mask[:component_count] = sentinel_mask
    return semantic, padded_sentinel_mask


def _rounded_molecule_code_keys(codes: np.ndarray) -> np.ndarray:
    """Return exact integer hundredth-code identities for catalog alignment."""

    values = np.asarray(codes, dtype=np.float64)
    keys = np.rint(values * 100.0).astype(np.int64)
    if not np.array_equal(values, keys.astype(np.float64) / 100.0):
        raise AssertionError("molecule codes are not exact hundredth identities")
    if np.unique(keys).size != keys.size:
        raise AssertionError("molecule-code identities are not unique")
    return keys


def _catalog_identity_arrays(runtime: PinnedRuntime) -> dict[str, Any]:
    """Capture both catalog buffers and their code-semantic exact alignment."""

    synthesis = _catalog_arrays(MOLECULE_CATALOG)
    atmosphere = _catalog_arrays(ATMOSPHERE_MOLECULE_CATALOG)
    synthesis_table = runtime.molecular.read_molecule_table(MOLECULE_CATALOG)
    for name in (
        "molecule_count",
        "equation_count",
        "molecule_codes",
        "equilibrium_coefficients",
        "component_start_indices",
        "component_equation_indices",
        "equation_species_codes",
    ):
        if not np.array_equal(
            np.asarray(getattr(synthesis_table, name)),
            np.asarray(synthesis[name]),
        ):
            raise AssertionError(f"executed synthesis reader changed {name}")

    synthesis_count = int(synthesis["molecule_count"])
    atmosphere_count = int(atmosphere["molecule_count"])
    synthesis_component_count = int(
        synthesis["component_start_indices"][synthesis_count]
    )
    atmosphere_component_count = int(atmosphere["component_count"])
    if (synthesis_count, atmosphere_count) != (190, 170):
        raise AssertionError("molecular catalog active counts changed")
    if (synthesis_component_count, atmosphere_component_count) != (548, 481):
        raise AssertionError("molecular catalog component counts changed")
    if synthesis_component_count != int(
        synthesis["component_start_indices"][synthesis_count]
    ):
        raise AssertionError("synthesis component extent is inconsistent")
    if atmosphere_component_count != int(
        atmosphere["component_start_indices"][atmosphere_count]
    ):
        raise AssertionError("atmosphere component extent is inconsistent")

    synthesis_semantic, synthesis_sentinel = _component_semantic_codes(
        synthesis
    )
    atmosphere_semantic, atmosphere_sentinel = _component_semantic_codes(
        atmosphere
    )
    synthesis_codes = np.asarray(
        synthesis["molecule_codes"][:synthesis_count], dtype=np.float64
    )
    atmosphere_codes = np.asarray(
        atmosphere["molecule_codes"][:atmosphere_count], dtype=np.float64
    )
    synthesis_keys = _rounded_molecule_code_keys(synthesis_codes)
    atmosphere_keys = _rounded_molecule_code_keys(atmosphere_codes)
    shared_keys = np.intersect1d(synthesis_keys, atmosphere_keys)
    atmosphere_only_keys = np.setdiff1d(atmosphere_keys, synthesis_keys)
    synthesis_only_keys = np.setdiff1d(synthesis_keys, atmosphere_keys)
    synthesis_index_by_key = {
        int(key): index for index, key in enumerate(synthesis_keys)
    }
    atmosphere_index_by_key = {
        int(key): index for index, key in enumerate(atmosphere_keys)
    }
    synthesis_shared_indices = np.asarray(
        [synthesis_index_by_key[int(key)] for key in shared_keys],
        dtype=np.int64,
    )
    atmosphere_shared_indices = np.asarray(
        [atmosphere_index_by_key[int(key)] for key in shared_keys],
        dtype=np.int64,
    )
    synthesis_only_indices = np.asarray(
        [synthesis_index_by_key[int(key)] for key in synthesis_only_keys],
        dtype=np.int64,
    )
    atmosphere_only_indices = np.asarray(
        [atmosphere_index_by_key[int(key)] for key in atmosphere_only_keys],
        dtype=np.int64,
    )
    synthesis_only_codes = synthesis_codes[synthesis_only_indices]
    if shared_keys.size != 170 or atmosphere_only_keys.size != 0:
        raise AssertionError("catalogs are not exactly 170-shared/0-atmosphere-only")
    if not np.array_equal(
        synthesis_only_codes, EXPECTED_SYNTHESIS_ONLY_CODES
    ):
        raise AssertionError("the exact 20 synthesis-only codes changed")
    if np.array_equal(synthesis_shared_indices, atmosphere_shared_indices):
        raise AssertionError("catalog comparison accidentally relies on row order")

    synthesis_coefficients = np.asarray(
        synthesis["equilibrium_coefficients"]
    )[:, synthesis_shared_indices]
    atmosphere_coefficients = np.asarray(
        atmosphere["equilibrium_coefficients"]
    )[:, atmosphere_shared_indices]
    coefficient_mismatch_mask = np.any(
        synthesis_coefficients != atmosphere_coefficients, axis=0
    )

    synthesis_starts = np.asarray(
        synthesis["component_start_indices"], dtype=np.int64
    )
    atmosphere_starts = np.asarray(
        atmosphere["component_start_indices"], dtype=np.int64
    )
    shared_component_offsets = [0]
    synthesis_shared_components: list[int] = []
    atmosphere_shared_components: list[int] = []
    component_mismatch_mask = []
    for synthesis_index, atmosphere_index in zip(
        synthesis_shared_indices, atmosphere_shared_indices, strict=True
    ):
        synthesis_components = synthesis_semantic[
            synthesis_starts[synthesis_index] : synthesis_starts[
                synthesis_index + 1
            ]
        ]
        atmosphere_components = atmosphere_semantic[
            atmosphere_starts[atmosphere_index] : atmosphere_starts[
                atmosphere_index + 1
            ]
        ]
        synthesis_shared_components.extend(synthesis_components.tolist())
        atmosphere_shared_components.extend(atmosphere_components.tolist())
        shared_component_offsets.append(len(synthesis_shared_components))
        component_mismatch_mask.append(
            not np.array_equal(synthesis_components, atmosphere_components)
        )
    component_mismatch_mask_array = np.asarray(
        component_mismatch_mask, dtype=np.bool_
    )
    semantic_mismatch_mask = (
        coefficient_mismatch_mask | component_mismatch_mask_array
    )
    if np.any(semantic_mismatch_mask):
        raise AssertionError("shared molecule codes have semantic mismatches")

    arrays: dict[str, Any] = {
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
            np.asarray(
                runtime.molecular._ATOMIC_MASSES_FOR_MOLECULES,
                dtype=np.float64,
            ).copy()
        ),
        "catalog__synthesis__hard_coded_molecular_atomic_masses_sha256": (
            hashlib.sha256(
                np.asarray(
                    runtime.molecular._ATOMIC_MASSES_FOR_MOLECULES,
                    dtype=np.float64,
                ).tobytes(order="C")
            ).hexdigest()
        ),
        "catalog__atmosphere__component_semantic_species_codes": (
            atmosphere_semantic
        ),
        "catalog__atmosphere__component_inverse_electron_sentinel_mask": (
            atmosphere_sentinel
        ),
        "alignment__shared_rounded_code_keys": shared_keys,
        "alignment__shared_molecule_codes": (
            synthesis_codes[synthesis_shared_indices]
        ),
        "alignment__synthesis_shared_row_indices": synthesis_shared_indices,
        "alignment__atmosphere_shared_row_indices": atmosphere_shared_indices,
        "alignment__shared_row_indices_differ_mask": (
            synthesis_shared_indices != atmosphere_shared_indices
        ),
        "alignment__shared_component_offsets": np.asarray(
            shared_component_offsets, dtype=np.int64
        ),
        "alignment__synthesis_shared_component_semantics": np.asarray(
            synthesis_shared_components, dtype=np.int32
        ),
        "alignment__atmosphere_shared_component_semantics": np.asarray(
            atmosphere_shared_components, dtype=np.int32
        ),
        "alignment__coefficient_mismatch_mask": coefficient_mismatch_mask,
        "alignment__component_mismatch_mask": component_mismatch_mask_array,
        "alignment__semantic_mismatch_mask": semantic_mismatch_mask,
        "alignment__shared_count": shared_keys.size,
        "alignment__atmosphere_only_rounded_code_keys": atmosphere_only_keys,
        "alignment__atmosphere_only_row_indices": atmosphere_only_indices,
        "alignment__atmosphere_only_count": atmosphere_only_keys.size,
        "alignment__synthesis_only_rounded_code_keys": synthesis_only_keys,
        "alignment__synthesis_only_row_indices": synthesis_only_indices,
        "alignment__synthesis_only_molecule_codes": synthesis_only_codes,
        "alignment__synthesis_only_count": synthesis_only_keys.size,
        "alignment__semantic_mismatch_count": int(
            semantic_mismatch_mask.sum()
        ),
    }
    for name, values in synthesis.items():
        arrays[f"catalog__synthesis__{name}"] = values
    for name, values in atmosphere.items():
        arrays[f"catalog__atmosphere__{name}"] = values
    return arrays


def _numpy_build_identity() -> dict[str, str]:
    """Return available NumPy BLAS/LAPACK identities without free-form output."""

    try:
        configuration = np.show_config(mode="dicts")
    except TypeError:
        configuration = {}
    dependencies = configuration.get("Build Dependencies", {})
    blas = dependencies.get("blas", {})
    lapack = dependencies.get("lapack", {})
    return {
        "blas_name": str(blas.get("name", "unknown")),
        "blas_version": str(blas.get("version", "unknown")),
        "lapack_name": str(lapack.get("name", "unknown")),
        "lapack_version": str(lapack.get("version", "unknown")),
    }


def _executed_source_arrays(runtime: PinnedRuntime) -> dict[str, Any]:
    """Record every imported synthesis source under the pinned checkout."""

    records = []
    for name, module in tuple(sys.modules.items()):
        if name != "payne_zero_synthesis" and not name.startswith(
            "payne_zero_synthesis."
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"executed synthesis module {name} has no source file")
        path = Path(module_file).resolve()
        if not path.is_relative_to(runtime.identity.root):
            raise RuntimeError(f"executed synthesis module escaped pin: {path}")
        records.append(
            (
                name,
                str(path.relative_to(runtime.identity.root)),
                sha256(path),
            )
        )
    records.sort()
    if not records:
        raise RuntimeError("no executed synthesis source modules were recorded")
    return {
        "meta__executed_source_module_names": np.asarray(
            [record[0] for record in records]
        ),
        "meta__executed_source_relative_paths": np.asarray(
            [record[1] for record in records]
        ),
        "meta__executed_source_sha256": np.asarray(
            [record[2] for record in records]
        ),
        "meta__executed_source_count": len(records),
    }


def _identity_arrays(runtime: PinnedRuntime, fixture_path: Path) -> dict[str, Any]:
    require_oracle_process_environment()
    build_identity = _numpy_build_identity()
    arrays: dict[str, Any] = {
        "meta__payne_zero_commit": runtime.identity.commit,
        "meta__fixture_sha256": sha256(fixture_path),
        "meta__device": "cpu",
        "meta__dtype": "torch.float64",
        "meta__torch_version": runtime.torch.__version__,
        "meta__numpy_version": np.__version__,
        "meta__python_version": sys.version.split()[0],
        "meta__platform": platform.platform(),
        "meta__system_byteorder": sys.byteorder,
        "meta__blas_name": build_identity["blas_name"],
        "meta__blas_version": build_identity["blas_version"],
        "meta__lapack_name": build_identity["lapack_name"],
        "meta__lapack_version": build_identity["lapack_version"],
        "meta__torch_num_threads": runtime.torch.get_num_threads(),
        "meta__torch_num_interop_threads": (
            runtime.torch.get_num_interop_threads()
        ),
    }
    for name, value in sorted(runtime.identity.sha256_by_name.items()):
        arrays[f"meta__source_sha256__{name}"] = value
    for name in sorted(ORACLE_PROCESS_ENVIRONMENT):
        arrays[f"meta__environment__{name}"] = os.environ[name]
    arrays.update(_executed_source_arrays(runtime))
    arrays.update(_catalog_identity_arrays(runtime))
    return arrays


def _input_arrays(inputs: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "input__temperature": inputs["temperature"],
        "input__gas_pressure": inputs["gas_pressure"],
        "input__electron_density_seed": inputs["electron_density_seed"],
        "input__elemental_abundances": inputs["elemental_abundances"],
    }


def _assert_cpu_float64_calls(
    calls: list[CapturedMolecularCall],
    runtime: PinnedRuntime,
) -> None:
    for call in calls:
        arguments = call.effective_arguments
        if arguments["device"] != runtime.torch.device("cpu"):
            raise AssertionError("molecular solve did not run on CPU")
        if arguments["dtype"] != runtime.torch.float64:
            raise AssertionError("molecular solve did not run in float64")
        if Path(arguments["molecules_path"]).resolve() != MOLECULE_CATALOG.resolve():
            raise AssertionError("molecular solve used the wrong catalog")
        if arguments["max_iter"] != 200:
            raise AssertionError("molecular solve did not use max_iter=200")
        if arguments["tol"] != 1.0e-4:
            raise AssertionError("molecular solve did not use tol=1e-4")
        if arguments["chain_length"] is not None:
            raise AssertionError("production route unexpectedly reset its chain")
        if arguments["return_diagnostics"]:
            raise AssertionError("EOS route itself unexpectedly requested diagnostics")


def build_full_route_result(
    runtime: PinnedRuntime,
    inputs: Mapping[str, np.ndarray],
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, np.ndarray]:
    """Run and capture the production full route's two molecular solves."""

    with MolecularSolveCapture(runtime.molecular) as capture:
        state = runtime.eos.solve_population_state(
            inputs["temperature"],
            inputs["gas_pressure"],
            inputs["elemental_abundances"],
            tables=runtime.tables,
            mean_nuclear_mass_amu=None,
            electron_density_seed=inputs["electron_density_seed"],
            max_iter=200,
            tol=1.0e-4,
            molecules=True,
            molecules_path=MOLECULE_CATALOG,
        )
    assert_route_call_ownership("full", capture.calls)
    _assert_cpu_float64_calls(capture.calls, runtime)
    for call in capture.calls:
        _assert_call_structure(call)
    first, second = capture.calls
    if not np.array_equal(
        state.electron_density, _as_numpy(first.outputs[3])
    ):
        raise AssertionError("full public electron density is not solve 0 output")
    if not np.array_equal(
        state.total_nuclei_number_density, _as_numpy(second.outputs[0])
    ):
        raise AssertionError("full nuclei density is not solve 1 output")
    if not np.array_equal(
        state.molecular_populations, _as_numpy(second.outputs[1])
    ):
        raise AssertionError("full molecular populations are not solve 1 output")
    if not np.array_equal(
        state.molecular_equation_densities, _as_numpy(second.outputs[2])
    ):
        raise AssertionError("full equation densities are not solve 1 output")

    arrays: dict[str, Any] = {}
    arrays.update(_identity_arrays(runtime, Path(fixture_path).resolve()))
    arrays.update(_input_arrays(inputs))
    arrays.update(_serialize_state(state))
    for index, call in enumerate(capture.calls):
        arrays.update(
            _serialize_call(f"call_{index}__", call, runtime.molecular)
        )
    arrays["trace__route"] = "full"
    arrays["trace__molecular_call_count"] = len(capture.calls)
    arrays["trace__published_electron_from_call"] = 0
    arrays["trace__published_molecules_from_call"] = 1
    return deterministic_result(arrays)


def build_fixed_route_result(
    runtime: PinnedRuntime,
    inputs: Mapping[str, np.ndarray],
    *,
    mass_density: Optional[np.ndarray] = None,
    fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, np.ndarray]:
    """Run and capture one fixed-public-electron-density molecular solve."""

    if mass_density is not None:
        supplied_mass = np.asarray(mass_density, dtype=np.float64)
        if supplied_mass.shape != np.asarray(inputs["temperature"]).shape:
            raise ValueError("supplied mass_density must follow the depth axis")
        if np.any(~np.isfinite(supplied_mass)) or np.any(supplied_mass <= 0.0):
            raise ValueError("supplied mass_density must be finite and positive")
    else:
        supplied_mass = None

    with MolecularSolveCapture(runtime.molecular) as capture:
        state = runtime.eos.solve_population_state_at_electron_density(
            inputs["temperature"],
            inputs["gas_pressure"],
            inputs["elemental_abundances"],
            tables=runtime.tables,
            electron_density=inputs["electron_density_seed"],
            mean_nuclear_mass_amu=None,
            mass_density=supplied_mass,
            molecules=True,
            molecules_path=MOLECULE_CATALOG,
        )
    assert_route_call_ownership("fixed", capture.calls)
    _assert_cpu_float64_calls(capture.calls, runtime)
    _assert_call_structure(capture.calls[0])
    call = capture.calls[0]
    if not np.array_equal(
        state.electron_density, np.asarray(inputs["electron_density_seed"])
    ):
        raise AssertionError("fixed route did not preserve public electron density")
    if not np.array_equal(
        state.total_nuclei_number_density, _as_numpy(call.outputs[0])
    ):
        raise AssertionError("fixed nuclei density is not its one solve output")
    if not np.array_equal(
        state.molecular_populations, _as_numpy(call.outputs[1])
    ):
        raise AssertionError("fixed molecular populations are not its solve output")
    if not np.array_equal(
        state.molecular_equation_densities, _as_numpy(call.outputs[2])
    ):
        raise AssertionError("fixed equation densities are not its solve output")
    if supplied_mass is not None and not np.array_equal(
        state.mass_density, supplied_mass
    ):
        raise AssertionError("fixed route did not preserve supplied mass density")

    arrays: dict[str, Any] = {}
    arrays.update(_identity_arrays(runtime, Path(fixture_path).resolve()))
    arrays.update(_input_arrays(inputs))
    arrays.update(_serialize_state(state))
    arrays.update(_serialize_call("call_0__", call, runtime.molecular))
    arrays["trace__route"] = (
        "fixed_supplied_mass" if supplied_mass is not None else "fixed_derived_mass"
    )
    arrays["trace__molecular_call_count"] = len(capture.calls)
    arrays["trace__published_electron_from_input"] = True
    arrays["trace__internal_electron_density"] = call.outputs[3]
    if supplied_mass is not None:
        arrays["input__mass_density"] = supplied_mass
    return deterministic_result(arrays)


def assert_public_builder_signature(pipeline_module: ModuleType) -> None:
    """Pin the real public structured-builder interface."""

    function = pipeline_module.build_structured_atmosphere_from_columns
    signature = inspect.signature(function)
    actual = tuple(signature.parameters)
    if actual != PUBLIC_BUILDER_PARAMETERS:
        raise RuntimeError(
            f"public builder parameters are {actual}; "
            f"expected {PUBLIC_BUILDER_PARAMETERS}"
        )
    if any(
        parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    ):
        raise RuntimeError("public structured-builder inputs must remain keyword-only")


def molecular_species_mapping_arrays(
    molecular_module: ModuleType,
) -> dict[str, np.ndarray]:
    """Return the exact lossless line-species to equilibrium-code mapping."""

    species_codes = np.asarray(
        molecular_module.supported_molecular_species_codes(),
        dtype=np.int64,
    )
    if species_codes.shape != (54,):
        raise AssertionError(
            f"expected 54 molecular species codes, got {species_codes.shape}"
        )
    if not np.array_equal(species_codes, np.unique(species_codes)):
        raise AssertionError("supported molecular species codes are not unique")
    mapping = molecular_module._SPECIES_CODE_TO_MOLECULE_CODES
    if set(mapping) != set(species_codes.tolist()):
        raise AssertionError("supported species list and mapping keys disagree")

    offsets = [0]
    flattened_codes: list[float] = []
    for species_code in species_codes:
        molecule_codes = tuple(mapping[int(species_code)])
        if not molecule_codes:
            raise AssertionError(
                f"species code {int(species_code)} has no equilibrium mapping"
            )
        flattened_codes.extend(float(code) for code in molecule_codes)
        offsets.append(len(flattened_codes))
    public_columns = species_codes // 6 - 1
    if np.unique(public_columns).size != species_codes.size:
        raise AssertionError("molecular species do not map to distinct columns")
    if np.any(public_columns < 0) or np.any(public_columns >= 139):
        raise AssertionError("molecular public column lies outside the schema cube")

    co_index = int(np.flatnonzero(species_codes == 276)[0])
    co_codes = flattened_codes[offsets[co_index] : offsets[co_index + 1]]
    if co_codes != [608.0] or int(public_columns[co_index]) != 45:
        raise AssertionError("CO mapping is not 276 -> 608 -> column 45")
    return {
        "species_codes": species_codes,
        "molecule_code_offsets": np.asarray(offsets, dtype=np.int64),
        "molecule_codes": np.asarray(flattened_codes, dtype=np.float64),
        "public_columns": public_columns.astype(np.int64),
    }


def _line_population_matrix(
    values_by_species: Mapping[int, np.ndarray],
    species_codes: np.ndarray,
) -> np.ndarray:
    """Stack a complete species-keyed result in the declared species order."""

    expected = {int(code) for code in species_codes}
    if set(values_by_species) != expected:
        raise AssertionError(
            "molecular line mapping returned a missing or unexpected species"
        )
    return np.column_stack(
        [
            np.asarray(values_by_species[int(code)], dtype=np.float64)
            for code in species_codes
        ]
    )


def _independent_line_population_matrix(
    *,
    runtime: PinnedRuntime,
    temperature: np.ndarray,
    equation_densities: np.ndarray,
    neutral_partition: np.ndarray,
    mapping: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[float, dict[str, np.ndarray]]]:
    """Reconstruct requested N/U lanes directly from catalog arithmetic."""

    table = runtime.molecular.read_molecule_table(MOLECULE_CATALOG)
    temperature_array = np.asarray(temperature, dtype=np.float64)
    equation_density_array = np.asarray(
        equation_densities, dtype=np.float64
    )
    neutral_partition_array = np.asarray(
        neutral_partition, dtype=np.float64
    )
    transformed = equation_density_array.copy()
    sqrt_temperature = np.sqrt(np.maximum(temperature_array, 1.0e-300))
    hard_coded_masses = np.asarray(
        runtime.molecular._ATOMIC_MASSES_FOR_MOLECULES, dtype=np.float64
    )
    for equation_index in range(1, int(table.equation_count)):
        element_id = int(table.equation_species_codes[equation_index])
        if element_id == 100:
            transformed[:, equation_index] = transformed[
                :, equation_index
            ] / (
                2.0
                * runtime.molecular.REFERENCE_SAHA_COEFFICIENT
                * temperature_array
                * sqrt_temperature
            )
            continue
        atomic_mass = (
            float(hard_coded_masses[element_id - 1])
            if 1 <= element_id <= hard_coded_masses.size
            else float(element_id)
        )
        partition = (
            neutral_partition_array[:, element_id - 1]
            if 1 <= element_id <= neutral_partition_array.shape[1]
            else np.ones(temperature_array.size, dtype=np.float64)
        )
        denominator = np.maximum(
            partition
            * 1.8786e20
            * np.sqrt(
                np.maximum(
                    (atomic_mass * temperature_array) ** 3, 1.0e-300
                )
            ),
            1.0e-300,
        )
        transformed[:, equation_index] = (
            transformed[:, equation_index] / denominator
        )

    requested_codes = np.asarray(mapping["molecule_codes"], dtype=np.float64)
    unique_codes: list[float] = []
    for code in requested_codes:
        if float(code) not in unique_codes:
            unique_codes.append(float(code))
    reconstructed_by_code: dict[float, np.ndarray] = {}
    diagnostics_by_code: dict[float, dict[str, np.ndarray]] = {}
    thermal_energy_ev = temperature_array / 11604.5
    active_codes = np.asarray(
        table.molecule_codes[: table.molecule_count], dtype=np.float64
    )
    for code in unique_codes:
        hits = np.flatnonzero(np.abs(active_codes - code) < 1.0e-3)
        if hits.size > 1:
            raise AssertionError(
                f"mapped equilibrium code {code:g} has {hits.size} catalog hits"
            )
        if hits.size == 0:
            zero_population = np.zeros(
                temperature_array.size, dtype=np.float64
            )
            reconstructed_by_code[code] = zero_population
            diagnostics_by_code[code] = {
                "catalog_index": np.asarray(-1, dtype=np.int64),
                "leading_coefficient": np.asarray(0.0, dtype=np.float64),
                "component_equation_indices": np.empty(0, dtype=np.int32),
                "component_species_codes": np.empty(0, dtype=np.int32),
                "component_atomic_masses_amu": np.empty(0, dtype=np.float64),
                "molecular_mass_amu": np.asarray(0.0, dtype=np.float64),
                "transformed_equation_densities": np.empty(
                    (temperature_array.size, 0), dtype=np.float64
                ),
                "normalization": np.zeros(
                    temperature_array.size, dtype=np.float64
                ),
                "population": zero_population,
            }
            continue
        molecule_index = int(hits[0])
        leading_coefficient = float(
            table.equilibrium_coefficients[0, molecule_index]
        )
        if leading_coefficient == 0.0:
            raise AssertionError(
                f"mapped line-population code {code:g} is not polynomial"
            )
        component_start = int(
            table.component_start_indices[molecule_index]
        )
        component_stop = int(
            table.component_start_indices[molecule_index + 1]
        )
        component_equation_indices = np.asarray(
            table.component_equation_indices[
                component_start:component_stop
            ],
            dtype=np.int32,
        )
        component_species_codes = []
        component_masses = []
        population = np.exp(
            leading_coefficient
            / np.maximum(thermal_energy_ev, 1.0e-300)
        )
        for equation_index in component_equation_indices:
            if int(equation_index) >= int(table.equation_count):
                component_species_codes.append(101)
                component_masses.append(0.0)
                population = population / np.maximum(
                    transformed[:, int(table.equation_count) - 1], 1.0e-300
                )
            else:
                element_id = int(
                    table.equation_species_codes[int(equation_index)]
                )
                component_species_codes.append(element_id)
                component_mass = (
                    float(hard_coded_masses[element_id - 1])
                    if 1 <= element_id <= hard_coded_masses.size
                    else 0.0
                )
                component_masses.append(component_mass)
                population = (
                    population * transformed[:, int(equation_index)]
                )
        component_masses_array = np.asarray(
            component_masses, dtype=np.float64
        )
        molecular_mass = float(component_masses_array.sum())
        mass_temperature_root = np.sqrt(
            np.maximum(
                (molecular_mass * temperature_array) ** 3, 1.0e-300
            )
        )
        population = population * 1.8786e20 * mass_temperature_root
        reconstructed_by_code[code] = population
        diagnostics_by_code[code] = {
            "catalog_index": np.asarray(molecule_index, dtype=np.int64),
            "leading_coefficient": np.asarray(
                leading_coefficient, dtype=np.float64
            ),
            "component_equation_indices": component_equation_indices,
            "component_species_codes": np.asarray(
                component_species_codes, dtype=np.int32
            ),
            "component_atomic_masses_amu": component_masses_array,
            "molecular_mass_amu": np.asarray(
                molecular_mass, dtype=np.float64
            ),
            "transformed_equation_densities": transformed[
                :, component_equation_indices
            ],
            "normalization": 1.8786e20 * mass_temperature_root,
            "population": population,
        }

    offsets = np.asarray(mapping["molecule_code_offsets"], dtype=np.int64)
    species_codes = np.asarray(mapping["species_codes"], dtype=np.int64)
    lanes = []
    for species_index in range(species_codes.size):
        start = int(offsets[species_index])
        stop = int(offsets[species_index + 1])
        lane = np.zeros(temperature_array.size, dtype=np.float64)
        for code in requested_codes[start:stop]:
            lane += reconstructed_by_code[float(code)]
        lanes.append(lane)
    return np.column_stack(lanes), diagnostics_by_code


def _ground_discrimination_mask(
    no_ground_lanes: np.ndarray,
    grounded_lanes: np.ndarray,
) -> np.ndarray:
    """Require a finite fixture that distinguishes the partition policies."""

    no_ground = np.asarray(no_ground_lanes, dtype=np.float64)
    grounded = np.asarray(grounded_lanes, dtype=np.float64)
    if no_ground.shape != grounded.shape:
        raise AssertionError("ground-policy lane arrays have different shapes")
    if np.any(~np.isfinite(no_ground)) or np.any(~np.isfinite(grounded)):
        raise AssertionError("molecular public-lane counterfactual is nonfinite")
    discrimination = no_ground != grounded
    if not np.any(discrimination):
        raise AssertionError(
            "fixture does not discriminate no-ground and grounded lanes"
        )
    return discrimination


def _assert_public_helper_call(
    call: CapturedFunctionCall,
    *,
    helper_name: str,
) -> None:
    """Require one helper to be owned by the exact pinned public builder."""

    expected_file = str(
        (PINNED_ROOT / "payne_zero_synthesis" / "pipeline.py").resolve()
    )
    if call.caller_name != "build_structured_atmosphere_from_columns":
        raise AssertionError(
            f"{helper_name} caller is {call.caller_name!r}, not public builder"
        )
    if call.caller_module != "payne_zero_synthesis.pipeline":
        raise AssertionError(
            f"{helper_name} caller module is {call.caller_module!r}"
        )
    if call.caller_file != expected_file:
        raise AssertionError(
            f"{helper_name} caller file is {call.caller_file!r}"
        )


def _assert_structured_schema(
    structured: Mapping[str, np.ndarray],
    atmosphere_module: ModuleType,
) -> tuple[str, ...]:
    """Require schema-v4 fields, shapes, dtypes, and finite numeric arrays."""

    schema = json.loads(ATMOSPHERE_SCHEMA.read_text(encoding="utf-8"))
    required = tuple(item["name"] for item in schema["required_arrays"])
    missing = [name for name in required if name not in structured]
    if missing:
        raise AssertionError(f"structured builder omitted schema arrays: {missing}")
    version = np.asarray(structured["atmosphere_schema_version"])
    expected_version = np.asarray(
        [atmosphere_module.ATMOSPHERE_SCHEMA_VERSION], dtype=np.int32
    )
    if not np.array_equal(version, expected_version):
        raise AssertionError("structured builder did not return schema version 4")

    depth_count = np.asarray(structured["temperature"]).size
    cube_shape = (depth_count, 6, 139)
    for name in (
        "partition_normalized_populations",
        "ion_stage_populations",
        "fractional_doppler_widths",
    ):
        if np.asarray(structured[name]).shape != cube_shape:
            raise AssertionError(f"{name} has the wrong public cube shape")
    for name, values in structured.items():
        array = np.asarray(values)
        if array.dtype.hasobject:
            raise AssertionError(f"{name} has forbidden object dtype")
        if np.issubdtype(array.dtype, np.number) and np.any(
            ~np.isfinite(array)
        ):
            raise AssertionError(f"{name} contains a nonfinite value")
    return tuple(sorted(structured))


def build_public_mapping_result(
    runtime: PinnedRuntime,
    inputs: Mapping[str, np.ndarray],
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, np.ndarray]:
    """Capture the exact public one-solve/reuse molecular mapping route."""

    pipeline = importlib.import_module("payne_zero_synthesis.pipeline")
    atmosphere = importlib.import_module("payne_zero_synthesis.atmosphere")
    _assert_module_under_root(pipeline, runtime.identity.root)
    _assert_module_under_root(atmosphere, runtime.identity.root)
    assert_public_builder_signature(pipeline)
    mapping = molecular_species_mapping_arrays(runtime.molecular)
    species_codes = mapping["species_codes"]
    atomic_masses = pipeline.load_atomic_masses(ATOMIC_MASSES)

    with (
        MolecularSolveCapture(runtime.molecular) as molecular_capture,
        PublicBuilderCapture(
            runtime.eos, runtime.molecular, pipeline
        ) as public_capture,
    ):
        structured = pipeline.build_structured_atmosphere_from_columns(
            temperature=inputs["temperature"],
            column_mass=inputs["column_mass"],
            gas_pressure=inputs["gas_pressure"],
            electron_density=inputs["electron_density_seed"],
            elemental_abundances=inputs["elemental_abundances"],
            mean_nuclear_mass_amu=None,
            microturbulence=inputs["microturbulence"],
            eos_tables=runtime.tables,
            electron_density_seed=None,
            tol=1.0e-4,
            atomic_masses=atomic_masses,
            mass_density=None,
            molecular_species_codes=species_codes,
            molecules_path=MOLECULE_CATALOG,
        )

    assert_route_call_ownership("fixed", molecular_capture.calls)
    _assert_cpu_float64_calls(molecular_capture.calls, runtime)
    _assert_call_structure(molecular_capture.calls[0])
    traced_groups = (
        ("fixed EOS", public_capture.fixed_calls),
        ("no-ground partition", public_capture.partition_calls),
        ("molecular line mapping", public_capture.line_calls),
        ("edge grid", public_capture.edge_calls),
    )
    for helper_name, calls in traced_groups:
        if len(calls) != 1:
            raise AssertionError(
                f"public builder made {len(calls)} {helper_name} calls"
            )
        _assert_public_helper_call(calls[0], helper_name=helper_name)

    fixed_call = public_capture.fixed_calls[0]
    partition_call = public_capture.partition_calls[0]
    line_call = public_capture.line_calls[0]
    edge_call = public_capture.edge_calls[0]
    fixed_state = fixed_call.result
    molecular_call = molecular_capture.calls[0]

    fixed_arguments = fixed_call.effective_arguments
    if fixed_arguments["tables"] is not runtime.tables:
        raise AssertionError("public builder did not reuse the explicit EOS tables")
    if not fixed_arguments["molecules"]:
        raise AssertionError("54 species codes did not activate molecular mode")
    if Path(fixed_arguments["molecules_path"]).resolve() != (
        MOLECULE_CATALOG.resolve()
    ):
        raise AssertionError("public fixed EOS used the wrong molecular catalog")
    if not np.array_equal(
        fixed_arguments["electron_density"],
        inputs["electron_density_seed"],
    ):
        raise AssertionError("public builder changed the fixed electron input")

    partition_arguments = partition_call.effective_arguments
    if partition_arguments["tables"] is not runtime.tables:
        raise AssertionError("no-ground partition used different EOS tables")
    if partition_arguments["nion"] != 6:
        raise AssertionError("no-ground partition did not request six stages")
    if partition_arguments["apply_ground_partition"] is not False:
        raise AssertionError("public molecular lane retained the ground floor")

    line_arguments = line_call.effective_arguments
    if not np.array_equal(line_arguments["species_codes"], species_codes):
        raise AssertionError("public line mapper did not receive all 54 codes")
    if Path(line_arguments["molecules_path"]).resolve() != (
        MOLECULE_CATALOG.resolve()
    ):
        raise AssertionError("public line mapper used the wrong catalog")
    if not np.array_equal(
        line_arguments["equation_densities"],
        fixed_state.molecular_equation_densities,
    ):
        raise AssertionError(
            "public builder did not reuse fixed-state equation densities"
        )
    if not np.array_equal(
        fixed_state.molecular_populations,
        _as_numpy(molecular_call.outputs[1]),
    ):
        raise AssertionError(
            "public fixed state did not retain its one molecular solve"
        )
    if not np.array_equal(
        fixed_state.molecular_equation_densities,
        _as_numpy(molecular_call.outputs[2]),
    ):
        raise AssertionError(
            "public fixed state equation densities are not its solve output"
        )

    edge_path = Path(edge_call.effective_arguments["edge_table_path"]).resolve()
    if edge_path != CONTINUUM_EDGE_GRID.resolve():
        raise AssertionError(f"public builder loaded the wrong edge grid: {edge_path}")

    grounded_partitions = (
        fixed_state.eos.partition_functions.detach()
        .cpu()
        .to(runtime.torch.float64)
        .numpy()
    )
    reconstructed_bridge_partitions = grounded_partitions.copy()
    partition_elements = np.asarray(
        sorted(partition_call.result), dtype=np.int64
    )
    partition_stage_counts = np.asarray(
        [
            partition_call.result[int(code)].shape[1]
            for code in partition_elements
        ],
        dtype=np.int64,
    )
    partition_offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(partition_stage_counts, dtype=np.int64),
        )
    )
    partition_without_ground = np.concatenate(
        [partition_call.result[int(code)] for code in partition_elements],
        axis=1,
    )
    for atomic_number, values in partition_call.result.items():
        copy_count = min(
            values.shape[1], reconstructed_bridge_partitions.shape[2]
        )
        reconstructed_bridge_partitions[
            :, int(atomic_number) - 1, :copy_count
        ] = values[:, :copy_count]
    no_ground_neutral = np.asarray(
        line_arguments["neutral_partition"], dtype=np.float64
    )
    if not np.array_equal(
        no_ground_neutral, reconstructed_bridge_partitions[:, :, 0]
    ):
        raise AssertionError("captured no-ground partition bridge is inconsistent")

    no_ground_lanes = _line_population_matrix(
        line_call.result, species_codes
    )
    independently_reconstructed_no_ground, no_ground_diagnostics = (
        _independent_line_population_matrix(
            runtime=runtime,
            temperature=inputs["temperature"],
            equation_densities=fixed_state.molecular_equation_densities,
            neutral_partition=no_ground_neutral,
            mapping=mapping,
        )
    )
    if not np.array_equal(
        independently_reconstructed_no_ground, no_ground_lanes
    ):
        raise AssertionError(
            "independent no-ground reconstruction differs from public mapper"
        )
    grounded_lanes, grounded_diagnostics = (
        _independent_line_population_matrix(
            runtime=runtime,
            temperature=inputs["temperature"],
            equation_densities=fixed_state.molecular_equation_densities,
            neutral_partition=grounded_partitions[:, :, 0],
            mapping=mapping,
        )
    )
    discrimination = _ground_discrimination_mask(
        no_ground_lanes, grounded_lanes
    )

    co_species_index = int(np.flatnonzero(species_codes == 276)[0])
    co_equilibrium_code = 608.0
    co_diagnostics = no_ground_diagnostics[co_equilibrium_code]
    co_grounded_diagnostics = grounded_diagnostics[co_equilibrium_code]
    co_public = no_ground_lanes[:, co_species_index]
    co_reconstructed = np.asarray(
        co_diagnostics["population"], dtype=np.float64
    )
    if not np.array_equal(co_public, co_reconstructed):
        raise AssertionError("CO public lane is not the independent 608 result")
    co_catalog_index = int(co_diagnostics["catalog_index"])
    co_raw_population = np.asarray(
        fixed_state.molecular_populations[:, co_catalog_index],
        dtype=np.float64,
    )
    co_raw_discrimination = co_raw_population != co_reconstructed
    if not np.any(co_raw_discrimination):
        raise AssertionError("fixture does not distinguish raw CO from CO N/U")
    if int(mapping["public_columns"][co_species_index]) != 45:
        raise AssertionError("CO public destination is no longer column 45")
    if not np.array_equal(
        co_diagnostics["component_species_codes"],
        np.asarray([6, 8], dtype=np.int32),
    ):
        raise AssertionError("CO components are no longer carbon and oxygen")
    if not np.array_equal(
        co_grounded_diagnostics["component_species_codes"],
        co_diagnostics["component_species_codes"],
    ):
        raise AssertionError("CO grounded reconstruction changed components")

    pre_partition = np.asarray(
        fixed_state.partition_normalized_populations, dtype=np.float64
    )
    post_partition = np.asarray(
        structured["partition_normalized_populations"], dtype=np.float64
    )
    public_columns = mapping["public_columns"]
    public_lanes = post_partition[:, 5, public_columns]
    if not np.array_equal(public_lanes, no_ground_lanes):
        raise AssertionError("public molecular lanes are not no-ground values")
    expected_post_partition = pre_partition.copy()
    expected_post_partition[:, 5, public_columns] = no_ground_lanes
    if not np.array_equal(post_partition, expected_post_partition):
        raise AssertionError("public builder changed an unowned partition cell")

    pre_ion = np.asarray(fixed_state.ion_stage_populations, dtype=np.float64)
    post_ion = np.asarray(structured["ion_stage_populations"], dtype=np.float64)
    if not np.array_equal(post_ion, pre_ion):
        raise AssertionError("public builder changed the actual ion-stage cube")

    metadata = runtime.molecular.molecular_equilibrium_metadata(
        MOLECULE_CATALOG
    )
    h2_indices = np.flatnonzero(
        np.abs(metadata.molecule_codes[: metadata.molecule_count] - 101.0)
        < 1.0e-3
    )
    if h2_indices.size != 1:
        raise AssertionError("synthesis catalog does not contain one H2 code 101")
    h2_solved = np.asarray(
        fixed_state.molecular_populations[:, int(h2_indices[0])],
        dtype=np.float64,
    )
    if not np.array_equal(
        structured["molecular_hydrogen_population"], h2_solved
    ):
        raise AssertionError("public named H2 did not use solved code 101")

    structured_keyset = _assert_structured_schema(structured, atmosphere)
    edge_public_names = {
        "signed_continuum_edge_frequency_hz": (
            "signed_continuum_edge_frequency_hz"
        ),
        "continuum_edge_wavelength_nm": "continuum_edge_wavelength_nm",
        "continuum_edge_midpoint_wavelength_nm": (
            "continuum_edge_midpoint_wavelength_nm"
        ),
        "continuum_edge_interval_width_squared_over_two_nm2": (
            "edge_interval_width_squared_over_two_nm2"
        ),
    }
    for public_name, loader_name in edge_public_names.items():
        if not np.array_equal(
            structured[public_name], edge_call.result[loader_name]
        ):
            raise AssertionError(f"structured edge array {public_name} changed")

    arrays: dict[str, Any] = {}
    arrays.update(_identity_arrays(runtime, Path(fixture_path).resolve()))
    arrays.update(_input_arrays(inputs))
    arrays.update(_serialize_state(fixed_state, "fixed_state__"))
    arrays.update(
        _serialize_call(
            "call_0__", molecular_call, runtime.molecular
        )
    )
    for name, values in structured.items():
        arrays[f"structured__{name}"] = values
    for name, values in mapping.items():
        arrays[f"mapping__{name}"] = values
    for name, values in edge_call.result.items():
        arrays[f"edge_loader__{name}"] = values
    arrays.update(
        {
            "schema__structured_keyset": np.asarray(structured_keyset),
            "partition__elements_without_ground_floor": partition_elements,
            "partition__without_ground_floor_offsets": partition_offsets,
            "partition__without_ground_floor_stage_counts": (
                partition_stage_counts
            ),
            "partition__without_ground_floor": partition_without_ground,
            "partition__grounded_cube": grounded_partitions,
            "partition__bridge_cube": reconstructed_bridge_partitions,
            "partition_cube__before": pre_partition,
            "partition_cube__after": post_partition,
            "ion_cube__before": pre_ion,
            "ion_cube__after": post_ion,
            "line_population__public": public_lanes,
            "line_population__no_ground": no_ground_lanes,
            "line_population__independent_no_ground": (
                independently_reconstructed_no_ground
            ),
            "line_population__grounded": grounded_lanes,
            "line_population__ground_discrimination_mask": discrimination,
            "line_population__ground_discrimination_max_abs": np.max(
                np.abs(no_ground_lanes - grounded_lanes)
            ),
            "molecular_hydrogen__catalog_index": int(h2_indices[0]),
            "molecular_hydrogen__solved_code_101": h2_solved,
            "co_reconstruction__line_species_code": 276,
            "co_reconstruction__equilibrium_code": co_equilibrium_code,
            "co_reconstruction__public_stage_index": 5,
            "co_reconstruction__public_column_index": 45,
            "co_reconstruction__catalog_index": co_catalog_index,
            "co_reconstruction__component_equation_indices": (
                co_diagnostics["component_equation_indices"]
            ),
            "co_reconstruction__component_species_codes": (
                co_diagnostics["component_species_codes"]
            ),
            "co_reconstruction__component_atomic_masses_amu": (
                co_diagnostics["component_atomic_masses_amu"]
            ),
            "co_reconstruction__molecular_mass_amu": (
                co_diagnostics["molecular_mass_amu"]
            ),
            "co_reconstruction__leading_coefficient": (
                co_diagnostics["leading_coefficient"]
            ),
            "co_reconstruction__temperature": inputs["temperature"],
            "co_reconstruction__equation_densities": (
                fixed_state.molecular_equation_densities[
                    :, co_diagnostics["component_equation_indices"]
                ]
            ),
            "co_reconstruction__transformed_equation_densities": (
                co_diagnostics["transformed_equation_densities"]
            ),
            "co_reconstruction__no_ground_neutral_partitions": (
                no_ground_neutral[
                    :, co_diagnostics["component_species_codes"] - 1
                ]
            ),
            "co_reconstruction__normalization": (
                co_diagnostics["normalization"]
            ),
            "co_reconstruction__reference_public_lane": co_public,
            "co_reconstruction__independent_population": co_reconstructed,
            "co_reconstruction__difference": co_reconstructed - co_public,
            "co_reconstruction__raw_equilibrium_population": (
                co_raw_population
            ),
            "co_reconstruction__raw_discrimination_mask": (
                co_raw_discrimination
            ),
            "co_reconstruction__grounded_population": (
                co_grounded_diagnostics["population"]
            ),
            "trace__route": "public_structured_mapping",
            "trace__fixed_eos_call_count": len(
                public_capture.fixed_calls
            ),
            "trace__molecular_solve_call_count": len(
                molecular_capture.calls
            ),
            "trace__partition_no_ground_call_count": len(
                public_capture.partition_calls
            ),
            "trace__line_mapping_production_call_count": len(
                public_capture.line_calls
            ),
            "trace__line_mapping_grounded_diagnostic_call_count": 0,
            "trace__independent_line_reconstruction_count": 2,
            "trace__edge_grid_call_count": len(public_capture.edge_calls),
            "trace__edge_grid_cache_present_before": (
                public_capture.edge_cache_present_before
            ),
            "trace__edge_grid_path": str(edge_path),
            "trace__fallback_resolve_call_count": 0,
            "trace__reused_fixed_molecular_arrays": True,
            "trace__fixed_eos_caller": fixed_call.caller_name,
            "trace__partition_no_ground_caller": (
                partition_call.caller_name
            ),
            "trace__line_mapping_caller": line_call.caller_name,
            "trace__edge_grid_caller": edge_call.caller_name,
        }
    )
    return deterministic_result(arrays)


class _BoundaryTraceComplete(Exception):
    """Private control-flow marker used to stop before an expensive next stage."""


def _unique_source_line(function: Any, text: str) -> int:
    """Return the unique source line whose stripped text equals ``text``."""

    source, first_line = inspect.getsourcelines(function)
    matches = [
        first_line + offset
        for offset, line in enumerate(source)
        if line.strip() == text
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {text!r} trace anchor, found {len(matches)}"
        )
    return matches[0]


def _capture_locals_at_line(
    function: Any,
    target_line: int,
    names: tuple[str, ...],
    *args,
    **kwargs,
) -> dict[str, Any]:
    """Call exact source and abort after copying named locals at one line."""

    captured: dict[str, Any] = {}
    previous_trace = sys.gettrace()

    def local_trace(frame, event, argument):
        if event == "line" and frame.f_lineno == target_line:
            for name in names:
                value = frame.f_locals[name]
                if isinstance(value, np.ndarray):
                    value = value.copy()
                captured[name] = value
            raise _BoundaryTraceComplete
        return local_trace

    def global_trace(frame, event, argument):
        if frame.f_code is function.__code__:
            return local_trace
        if previous_trace is not None:
            return previous_trace(frame, event, argument)
        return None

    sys.settrace(global_trace)
    try:
        function(*args, **kwargs)
    except _BoundaryTraceComplete:
        pass
    finally:
        sys.settrace(previous_trace)
    if set(captured) != set(names):
        raise RuntimeError("exact boundary trace did not reach its source anchor")
    return captured


def _synthesis_formation_boundary(
    runtime: PinnedRuntime,
    inputs: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Capture the exact 10000 K polynomial gate and log sentinel."""

    temperature = np.asarray(
        inputs["synthesis_polynomial_boundary_temperature"],
        dtype=np.float64,
    )
    expected = np.array(
        [10000.0, np.nextafter(np.float64(10000.0), np.float64(np.inf))]
    )
    if not np.array_equal(temperature, expected):
        raise AssertionError("fixture lost the exact synthesis 10000 K boundary")
    temperature_bits = temperature.view(np.uint64)
    expected_bits = np.asarray(
        [np.float64(10000.0).view(np.uint64)]
    )
    if (
        temperature_bits[0] != expected_bits[0]
        or temperature_bits[1] != temperature_bits[0] + np.uint64(1)
    ):
        raise AssertionError("synthesis 10000 K probes are not bit-adjacent")
    temperature_branch_mask = temperature <= 10000.0
    if not np.array_equal(
        temperature_branch_mask, np.asarray([True, False])
    ):
        raise AssertionError("synthesis 10000 K branch mask changed")
    gas_pressure = np.full(temperature.size, 1.0e5, dtype=np.float64)
    electron_density = (
        0.1
        * gas_pressure
        / (
            runtime.eos.REFERENCE_BOLTZMANN_ERG_PER_K
            * temperature
        )
    )
    metadata = runtime.molecular.molecular_equilibrium_metadata(
        MOLECULE_CATALOG
    )
    ion_constants = runtime.eos.molecular_ion_formation_constants_from_seed(
        temperature,
        gas_pressure,
        electron_density,
        tables=runtime.tables,
        meta=metadata,
    )
    molecule_table = runtime.molecular.read_molecule_table(MOLECULE_CATALOG)
    direct_polynomial = runtime.molecular.polynomial_formation_constants(
        temperature, molecule_table
    )
    trace_line = _unique_source_line(
        runtime.molecular.solve_molecular_equilibrium,
        "structure = MolecularStructure.build(molecule_table, device=device, "
        "dtype=dtype)",
    )
    traced = _capture_locals_at_line(
        runtime.molecular.solve_molecular_equilibrium,
        trace_line,
        (
            "formation_constants",
            "natural_log_formation_constants",
        ),
        temperature,
        gas_pressure,
        electron_density,
        inputs["elemental_abundances"],
        ion_constants,
        molecules_path=MOLECULE_CATALOG,
        device=runtime.torch.device("cpu"),
        dtype=runtime.torch.float64,
        max_iter=200,
        tol=1.0e-4,
        chain_length=None,
        return_diagnostics=False,
    )
    formation_constants = np.asarray(
        traced["formation_constants"], dtype=np.float64
    )[:, : metadata.molecule_count]
    natural_log_constants = np.asarray(
        traced["natural_log_formation_constants"], dtype=np.float64
    )
    polynomial_mask = (
        metadata.equilibrium_coefficients[0, : metadata.molecule_count]
        != 0.0
    )
    if not np.array_equal(
        formation_constants[1, polynomial_mask],
        np.zeros(int(polynomial_mask.sum()), dtype=np.float64),
    ):
        raise AssertionError("polynomial formation constants survived above 10000 K")
    if not np.array_equal(
        natural_log_constants[1, polynomial_mask],
        np.full(int(polynomial_mask.sum()), -700.0, dtype=np.float64),
    ):
        raise AssertionError("above-gate polynomial logs did not use -700")
    if not np.array_equal(
        direct_polynomial[:, polynomial_mask],
        formation_constants[:, polynomial_mask],
    ):
        raise AssertionError("pre-Newton polynomial constants changed")
    return {
        "boundary__synthesis_polynomial_temperature": temperature,
        "boundary__synthesis_polynomial_temperature_bits": temperature_bits,
        "boundary__synthesis_polynomial_threshold_bits": expected_bits,
        "boundary__synthesis_polynomial_temperature_branch_mask": (
            temperature_branch_mask
        ),
        "boundary__synthesis_polynomial_gas_pressure": gas_pressure,
        "boundary__synthesis_polynomial_electron_density": electron_density,
        "boundary__synthesis_polynomial_mask": polynomial_mask,
        "boundary__synthesis_polynomial_molecule_codes": (
            metadata.molecule_codes[: metadata.molecule_count]
        ),
        "boundary__synthesis_polynomial_direct_constants": direct_polynomial,
        "boundary__synthesis_pre_newton_formation_constants": (
            formation_constants
        ),
        "boundary__synthesis_pre_newton_log_formation_constants": (
            natural_log_constants
        ),
    }


def _provisional_h2_boundary(
    runtime: PinnedRuntime,
    inputs: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Trace the exact inline provisional H2 value before code-101 replacement."""

    pipeline = importlib.import_module("payne_zero_synthesis.pipeline")
    _assert_module_under_root(pipeline, runtime.identity.root)
    assert_public_builder_signature(pipeline)
    temperature = np.asarray(
        inputs["synthesis_named_h2_boundary_temperature"],
        dtype=np.float64,
    )
    expected = np.array(
        [9000.0, np.nextafter(np.float64(9000.0), np.float64(np.inf))]
    )
    if not np.array_equal(temperature, expected):
        raise AssertionError("fixture lost the exact named-H2 9000 K boundary")
    temperature_bits = temperature.view(np.uint64)
    expected_bits = np.asarray([np.float64(9000.0).view(np.uint64)])
    if (
        temperature_bits[0] != expected_bits[0]
        or temperature_bits[1] != temperature_bits[0] + np.uint64(1)
    ):
        raise AssertionError("synthesis 9000 K probes are not bit-adjacent")
    temperature_branch_mask = temperature <= 9000.0
    if not np.array_equal(
        temperature_branch_mask, np.asarray([True, False])
    ):
        raise AssertionError("provisional-H2 branch mask changed")
    gas_pressure = np.full(temperature.size, 1.0e5, dtype=np.float64)
    electron_density = (
        0.1
        * gas_pressure
        / (
            runtime.eos.REFERENCE_BOLTZMANN_ERG_PER_K
            * temperature
        )
    )
    target_line = _unique_source_line(
        pipeline.build_structured_atmosphere_from_columns,
        "hydrogen_ionized_population = np.asarray(",
    )
    atomic_masses = pipeline.load_atomic_masses(ATOMIC_MASSES)
    species_codes = runtime.molecular.supported_molecular_species_codes()
    with MolecularSolveCapture(runtime.molecular) as molecular_capture:
        traced = _capture_locals_at_line(
            pipeline.build_structured_atmosphere_from_columns,
            target_line,
            (
                "temperature_array",
                "hydrogen_neutral_population",
                "molecular_hydrogen_equilibrium_factor",
                "molecular_hydrogen_population",
            ),
            temperature=temperature,
            column_mass=inputs["column_mass"][:2],
            gas_pressure=gas_pressure,
            electron_density=electron_density,
            elemental_abundances=inputs["elemental_abundances"],
            mean_nuclear_mass_amu=None,
            microturbulence=inputs["microturbulence"][:2],
            eos_tables=runtime.tables,
            electron_density_seed=None,
            tol=1.0e-4,
            atomic_masses=atomic_masses,
            mass_density=None,
            molecular_species_codes=species_codes,
            molecules_path=MOLECULE_CATALOG,
        )
    assert_route_call_ownership("fixed", molecular_capture.calls)
    _assert_cpu_float64_calls(molecular_capture.calls, runtime)
    _assert_call_structure(molecular_capture.calls[0])
    provisional = np.asarray(
        traced["molecular_hydrogen_population"], dtype=np.float64
    )
    if not provisional[0] > 0.0:
        raise AssertionError("provisional H2 is not positive at exactly 9000 K")
    if provisional[1] != 0.0:
        raise AssertionError("provisional H2 did not turn off above 9000 K")
    return {
        "boundary__provisional_h2_temperature": temperature,
        "boundary__provisional_h2_temperature_bits": temperature_bits,
        "boundary__provisional_h2_threshold_bits": expected_bits,
        "boundary__provisional_h2_temperature_branch_mask": (
            temperature_branch_mask
        ),
        "boundary__provisional_h2_active_population_mask": provisional > 0.0,
        "boundary__provisional_h2_gas_pressure": gas_pressure,
        "boundary__provisional_h2_electron_density": electron_density,
        "boundary__provisional_h2_neutral_hydrogen": traced[
            "hydrogen_neutral_population"
        ],
        "boundary__provisional_h2_equilibrium_factor": traced[
            "molecular_hydrogen_equilibrium_factor"
        ],
        "boundary__provisional_h2_population": provisional,
        "boundary__provisional_h2_molecular_iterations": (
            molecular_capture.calls[0].diagnostics["iterations_completed"]
        ),
        "boundary__provisional_h2_molecular_exhausted_mask": (
            molecular_capture.calls[0].exhausted_mask
        ),
        "trace__provisional_h2_fixed_eos_call_count": 1,
        "trace__provisional_h2_molecular_solve_call_count": len(
            molecular_capture.calls
        ),
    }


def build_boundary_probe_result(
    runtime: PinnedRuntime,
    inputs: Mapping[str, np.ndarray],
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, np.ndarray]:
    """Build exact synthesis 10000 K and provisional-H2 9000 K probes."""

    arrays: dict[str, Any] = {}
    arrays.update(_synthesis_formation_boundary(runtime, inputs))
    arrays.update(_provisional_h2_boundary(runtime, inputs))
    arrays.update(_identity_arrays(runtime, Path(fixture_path).resolve()))
    arrays["trace__route"] = "synthesis_boundary_probes"
    return deterministic_result(arrays)


def capture_exotic_boundary_traces(
    runtime: PinnedRuntime,
    inputs: Mapping[str, np.ndarray],
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
) -> dict[str, np.ndarray]:
    """Compatibility spelling for the now-implemented exact boundary probes."""

    return build_boundary_probe_result(
        runtime, inputs, fixture_path=fixture_path
    )


def _result_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash an in-memory result without claiming it is a published archive."""

    digest = hashlib.sha256()
    for name, values in arrays.items():
        array = np.asarray(values)
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _deterministic_supplied_mass_density(
    inputs: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Derive an input-only positive mass track for the supplied-mass branch."""

    with np.load(ATOMIC_MASSES, allow_pickle=False) as archive:
        atomic_masses = np.asarray(
            archive["atomic_mass_amu"], dtype=np.float64
        )[:99]
    abundance = np.asarray(inputs["elemental_abundances"], dtype=np.float64)
    abundance_sum = float(abundance.sum())
    mean_nuclear_mass_amu = float(
        np.sum(abundance * atomic_masses) / abundance_sum
    )
    nuclei_density_scale = np.asarray(
        inputs["gas_pressure"], dtype=np.float64
    ) / (
        np.asarray(inputs["temperature"], dtype=np.float64)
        * 1.38054e-16
    )
    supplied = (
        nuclei_density_scale * mean_nuclear_mass_amu * 1.660e-24
    )
    if np.any(~np.isfinite(supplied)) or np.any(supplied <= 0.0):
        raise AssertionError("deterministic supplied mass is not finite and positive")
    return supplied


def _route_summary(
    requested_route: str,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Return a compact regression record for one in-memory-only CLI route."""

    exhausted_keys = [
        name for name in arrays if name.endswith("exhausted_mask")
    ]
    exhaustion_count = int(
        sum(np.count_nonzero(arrays[name]) for name in exhausted_keys)
    )
    if requested_route == "full":
        principal_shape = list(np.asarray(
            arrays["state__molecular_populations"]
        ).shape)
        call_count = int(arrays["trace__molecular_call_count"])
    elif requested_route in {"fixed", "fixed-supplied"}:
        principal_shape = list(np.asarray(
            arrays["state__molecular_populations"]
        ).shape)
        call_count = int(arrays["trace__molecular_call_count"])
    elif requested_route == "public":
        principal_shape = list(np.asarray(
            arrays["structured__partition_normalized_populations"]
        ).shape)
        call_count = int(arrays["trace__molecular_solve_call_count"])
    elif requested_route == "boundaries":
        principal_shape = list(np.asarray(
            arrays["boundary__synthesis_pre_newton_formation_constants"]
        ).shape)
        call_count = int(
            arrays["trace__provisional_h2_molecular_solve_call_count"]
        )
    else:
        raise ValueError(f"cannot summarize route {requested_route!r}")
    summary: dict[str, Any] = {
        "requested_route": requested_route,
        "captured_route": str(np.asarray(arrays["trace__route"]).item()),
        "key_count": len(arrays),
        "digest": _result_digest(arrays),
        "molecular_call_count": call_count,
        "exhaustion_count": exhaustion_count,
        "principal_shape": principal_shape,
        "shared_catalog_count": int(arrays["alignment__shared_count"]),
        "synthesis_only_catalog_count": int(
            arrays["alignment__synthesis_only_count"]
        ),
        "catalog_semantic_mismatch_count": int(
            arrays["alignment__semantic_mismatch_count"]
        ),
        "synthesis_only_catalog_codes": np.asarray(
            arrays["alignment__synthesis_only_molecule_codes"],
            dtype=np.float64,
        ).tolist(),
        "catalog_row_reordering_count": int(
            np.count_nonzero(
                arrays["alignment__shared_row_indices_differ_mask"]
            )
        ),
        "executed_source_count": int(
            arrays["meta__executed_source_count"]
        ),
        "platform": str(np.asarray(arrays["meta__platform"]).item()),
        "system_byteorder": str(
            np.asarray(arrays["meta__system_byteorder"]).item()
        ),
        "blas_name": str(np.asarray(arrays["meta__blas_name"]).item()),
        "process_environment_count": sum(
            name.startswith("meta__environment__") for name in arrays
        ),
    }
    iteration_keys = sorted(
        name
        for name in arrays
        if name.endswith("diag__iterations_completed")
    )
    summary["iteration_vectors"] = [
        np.asarray(arrays[name], dtype=np.int64).tolist()
        for name in iteration_keys
    ]
    if requested_route == "fixed-supplied":
        summary["supplied_mass_declared"] = bool(
            "input__mass_density" in arrays
            and np.all(np.asarray(arrays["input__mass_density"]) > 0.0)
        )
    if requested_route == "public":
        summary["production_line_mapping_call_count"] = int(
            arrays["trace__line_mapping_production_call_count"]
        )
        summary["grounded_production_line_mapping_call_count"] = int(
            arrays["trace__line_mapping_grounded_diagnostic_call_count"]
        )
        summary["co_exact_reconstruction"] = bool(
            np.array_equal(
                arrays["co_reconstruction__independent_population"],
                arrays["co_reconstruction__reference_public_lane"],
            )
        )
        summary["co_raw_discrimination_count"] = int(
            np.count_nonzero(
                arrays["co_reconstruction__raw_discrimination_mask"]
            )
        )
    if requested_route == "boundaries":
        summary["polynomial_temperature_bits"] = np.asarray(
            arrays["boundary__synthesis_polynomial_temperature_bits"],
            dtype=np.uint64,
        ).tolist()
        summary["polynomial_branch_mask"] = np.asarray(
            arrays[
                "boundary__synthesis_polynomial_temperature_branch_mask"
            ],
            dtype=np.bool_,
        ).tolist()
        summary["provisional_h2_temperature_bits"] = np.asarray(
            arrays["boundary__provisional_h2_temperature_bits"],
            dtype=np.uint64,
        ).tolist()
        summary["provisional_h2_branch_mask"] = np.asarray(
            arrays[
                "boundary__provisional_h2_temperature_branch_mask"
            ],
            dtype=np.bool_,
        ).tolist()
    return summary


def parse_args() -> argparse.Namespace:
    """Parse an identity check or an explicitly requested route run."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--route",
        choices=(
            "identity",
            "full",
            "fixed",
            "fixed-supplied",
            "public",
            "boundaries",
        ),
        default="identity",
    )
    return parser.parse_args()


def main() -> None:
    """Verify identity or build one route in memory without publishing files."""

    arguments = parse_args()
    inputs = load_input_fixture(arguments.fixture)
    if arguments.route == "identity":
        identity = verify_pinned_source_identity()
        print(
            f"verified {identity.commit} and input fixture "
            f"{sha256(arguments.fixture.resolve())}"
        )
        return

    runtime = load_pinned_runtime()
    if arguments.route == "full":
        arrays = build_full_route_result(
            runtime, inputs, fixture_path=arguments.fixture
        )
    elif arguments.route == "fixed":
        arrays = build_fixed_route_result(
            runtime, inputs, fixture_path=arguments.fixture
        )
    elif arguments.route == "fixed-supplied":
        arrays = build_fixed_route_result(
            runtime,
            inputs,
            mass_density=_deterministic_supplied_mass_density(inputs),
            fixture_path=arguments.fixture,
        )
    elif arguments.route == "public":
        arrays = build_public_mapping_result(
            runtime, inputs, fixture_path=arguments.fixture
        )
    else:
        arrays = build_boundary_probe_result(
            runtime, inputs, fixture_path=arguments.fixture
        )
    print(json.dumps(_route_summary(arguments.route, arrays), sort_keys=True))


if __name__ == "__main__":
    main()
