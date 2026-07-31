#!/usr/bin/env python3
"""Build Chapter 4 comparison archives from two isolated pinned captures.

The scientific workers remain in-memory observers.  This publisher launches
each scientific route in its own process, writes only temporary raw captures,
assembles compact public archives twice, and requires byte identity before an
atomic first publication.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
repository_text = str(REPOSITORY_ROOT)
if repository_text not in sys.path:
    sys.path.insert(0, repository_text)

from scripts.deterministic_npz import write_npz  # noqa: E402


PINNED_ROOT = Path("/Users/ysting/payne-zero")
PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
FIXTURE = REPOSITORY_ROOT / "data" / "fixtures" / "chapter04_molecular_inputs.npz"
FIXTURE_SHA256 = (
    "351bba75dca1fa502f5cc2a108035f69f2e31c760a90133480f2e7fe31e45f79"
)
OUTPUT_DIR = (
    REPOSITORY_ROOT / "data" / "golden" / "payne_zero" / "chapter04"
)
CAPTURE_CONTRACT = REPOSITORY_ROOT / "design" / "chapter04_oracle_capture_contract.md"
ACCEPTANCE_REPORT = (
    REPOSITORY_ROOT / "design" / "chapter04_oracle_worker_acceptance.md"
)
ATMOSPHERE_WORKER = (
    REPOSITORY_ROOT / "scripts" / "chapter04_atmosphere_oracle_worker.py"
)
SYNTHESIS_WORKER = (
    REPOSITORY_ROOT / "scripts" / "chapter04_synthesis_oracle_worker.py"
)
DETERMINISTIC_WRITER = REPOSITORY_ROOT / "scripts" / "deterministic_npz.py"

PINNED_PUBLISHER_INPUT_SHA256 = {
    "atmosphere_worker": (
        "5c33a5064fbfe187527fcc257b50ca3cc2cde40af6802d4a965f9d07c689f31f"
    ),
    "synthesis_worker": (
        "a9b20f2149cfc3a07dc283007b775e2036b00958d1b343e8f3528a5b63fa1b50"
    ),
    "deterministic_npz": (
        "f4886766524d79623e648d28ab9d24215da42f8bf1f859f69381c546f9c96e49"
    ),
    "capture_contract": (
        "ac170894a3ca5f7c8eda6dc1a3cd19688b3254e10c32a729f46b09c36344efff"
    ),
    "oracle_acceptance": (
        "6416e5890f29faf6defab48702ce76d972f40f1f37148fd9362cf3c433c8b76b"
    ),
}
PINNED_PUBLISHER_INPUT_PATHS = {
    "atmosphere_worker": ATMOSPHERE_WORKER,
    "synthesis_worker": SYNTHESIS_WORKER,
    "deterministic_npz": DETERMINISTIC_WRITER,
    "capture_contract": CAPTURE_CONTRACT,
    "oracle_acceptance": ACCEPTANCE_REPORT,
}

ACCEPTED_RAW_DIGESTS = {
    "atmosphere": (
        "a6116c5f73c7ed3b0ee51907c419a307de2e1477be09f0b9969fab888c0b7682"
    ),
    "synthesis-boundaries": (
        "49b633ef299deb2ad3d37009506a79daa1931c941997e742b7fdd7fd4b8c62f1"
    ),
    "synthesis-full": (
        "f4e36d39a9b736ade95972d810801e7ecaad66a5c7c0985b83ca897f0485b8d0"
    ),
    "synthesis-fixed-derived": (
        "ce0cb3c2a2d323011dfe4b7a0a4ea1dc9a37df2ad5dbad22bd6482539995a56f"
    ),
    "synthesis-fixed-supplied": (
        "1e9782da2c9990bbefa74fd34d26dd72d8a8445773e8b646d773a7a9935a4444"
    ),
    "synthesis-public": (
        "2e87bbc3b8bf09b03af232515518513daa0d1886e120b677355496c333bfee96"
    ),
}

OUTPUT_NAMES = (
    "chapter04_molecular_constants_cpu_float64.npz",
    "chapter04_atmosphere_molecular_state_cpu_float64.npz",
    "chapter04_synthesis_molecular_full_cpu_float64.npz",
    "chapter04_synthesis_molecular_fixed_cpu_float64.npz",
    "chapter04_molecular_public_mapping_cpu_float64.npz",
)
CONSTANTS_NAME = OUTPUT_NAMES[0]
EXPECTED_ARCHIVE_KINDS = {
    OUTPUT_NAMES[0]: "molecular_constants",
    OUTPUT_NAMES[1]: "atmosphere_molecular_state",
    OUTPUT_NAMES[2]: "synthesis_molecular_full",
    OUTPUT_NAMES[3]: "synthesis_molecular_fixed",
    OUTPUT_NAMES[4]: "molecular_public_mapping",
}

ROUTES = (
    "atmosphere",
    "synthesis-boundaries",
    "synthesis-full",
    "synthesis-fixed-derived",
    "synthesis-fixed-supplied",
    "synthesis-public",
)
RAW_NAMES = tuple(f"{route}.npz" for route in ROUTES)

PROCESS_CONTROLS = {
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
EXPECTED_ATMOSPHERE_SCHEDULE_CODES = np.asarray(
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
EXPECTED_ATMOSPHERE_SCHEDULE_SLOTS = np.asarray(
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

ATMOSPHERE_ROUTE_PREFIXES = (
    "full_",
    "priming_cache_",
    "schedule_inventory_",
    "handoff_",
    "disabled_",
    "mode2_",
    "mode12_",
    "energy_",
    "bridge_",
    "temperature_control_",
    "pressure_control_",
)
ATMOSPHERE_IDENTITY_PREFIXES = (
    "asset_",
    "environment_",
    "executed_atmosphere_source_",
    "source_",
)
ATMOSPHERE_IDENTITY_NAMES = frozenset(
    {
        "blas_name",
        "blas_version",
        "lapack_name",
        "lapack_version",
        "numba_thread_count",
        "numba_version",
        "numpy_build_dependencies_json",
        "numpy_version",
        "platform",
        "platform_machine",
        "platform_release",
        "platform_system",
        "python_executable",
        "python_implementation",
        "python_version",
        "system_byteorder",
    }
)

COMMON_CALL_INPUT_ALIASES = {
    "input__temperature": "input__temperature",
    "input__gas_pressure": "input__gas_pressure",
    "input__elemental_abundances": "input__elemental_abundances",
}
STATE_CALL_ALIASES = {
    "state__electron_density": "call_0__output__electron_density",
    "state__total_nuclei_number_density": (
        "call_1__output__total_nuclei_number_density"
    ),
    "state__molecular_populations": "call_1__output__molecular_populations",
    "state__molecular_equation_densities": (
        "call_1__output__equation_densities"
    ),
}
FIXED_STATE_CALL_ALIASES = {
    "state__electron_density": "input__electron_density_seed",
    "state__total_nuclei_number_density": (
        "call_0__output__total_nuclei_number_density"
    ),
    "state__molecular_populations": "call_0__output__molecular_populations",
    "state__molecular_equation_densities": (
        "call_0__output__equation_densities"
    ),
}
PUBLIC_FIXED_STATE_CALL_ALIASES = {
    key.replace("state__", "fixed_state__"): value
    for key, value in FIXED_STATE_CALL_ALIASES.items()
}


@dataclass(frozen=True)
class RawSchema:
    """Fail-closed shape of one reviewed worker result."""

    key_count: int
    required: frozenset[str]
    allowed_prefixes: tuple[str, ...]


RAW_SCHEMAS = {
    "atmosphere": RawSchema(
        460,
        frozenset(
            {
                "capture_in_memory_scope_complete",
                "capture_scope_complete",
                "capture_wrappers_restored",
                "capture_numpy_linalg_wrappers_restored",
                "capture_population_schedule_wrappers_restored",
                "deferred_golden_publication",
                "full_molecular_populations",
                "full_transformed_molecular_equation_densities",
                "full_newton_preupdate_equation_densities",
                "full_newton_raw_corrections",
                "full_newton_still_iterating_history",
                "full_newton_linear_direct_solve_branch_mask",
                "full_newton_linear_lstsq_branch_mask",
                "full_fractional_doppler_widths",
                "full_fractional_doppler_width_structural_infinity_mask",
                "full_fractional_doppler_width_structural_infinity_slots",
                "priming_cache_zero_code",
                "schedule_inventory_code",
                "handoff_molecular_populations",
                "disabled_solve_call_count",
                "mode2_population_mode",
                "mode12_population_mode",
                "energy_saved_physical_rows_input",
                "bridge_live_shape_error_type",
                "temperature_control_raw_named_populations",
                "pressure_control_raw_named_populations",
                "boundary_temperature_uint64_bits",
                "h2_probe_temperature_uint64_bits",
                "named_molecule_codes",
                "named_molecule_catalog_indices",
            }
        ),
        (
            "asset_",
            "atmosphere_",
            "boundary_",
            "bridge_",
            "capture_",
            "deferred_",
            "disabled_",
            "energy_",
            "environment_",
            "executed_atmosphere_source_",
            "full_",
            "h2_probe_",
            "handoff_",
            "mode12_",
            "mode2_",
            "named_",
            "pressure_control_",
            "priming_cache_",
            "schedule_inventory_",
            "source_",
            "temperature_control_",
        ),
    ),
    "synthesis-boundaries": RawSchema(
        112,
        frozenset(
            {
                "alignment__shared_count",
                "alignment__synthesis_only_count",
                "alignment__semantic_mismatch_count",
                "boundary__synthesis_polynomial_temperature_bits",
                "boundary__synthesis_polynomial_temperature_branch_mask",
                "boundary__provisional_h2_temperature_bits",
                "boundary__provisional_h2_temperature_branch_mask",
                "boundary__provisional_h2_molecular_exhausted_mask",
                "trace__provisional_h2_molecular_solve_call_count",
                "trace__route",
            }
        ),
        ("alignment__", "boundary__", "catalog__", "meta__", "trace__"),
    ),
    "synthesis-full": RawSchema(
        196,
        frozenset(
            {
                "alignment__shared_count",
                "call_0__caller_name",
                "call_0__diag__exhausted_mask",
                "call_1__caller_name",
                "call_1__diag__exhausted_mask",
                "state__electron_density",
                "state__molecular_populations",
                "trace__molecular_call_count",
                "trace__published_electron_from_call",
                "trace__published_molecules_from_call",
                "trace__route",
            }
        ),
        (
            "alignment__",
            "call_0__",
            "call_1__",
            "catalog__",
            "input__",
            "meta__",
            "state__",
            "trace__",
        ),
    ),
    "synthesis-fixed-derived": RawSchema(
        155,
        frozenset(
            {
                "call_0__diag__exhausted_mask",
                "state__electron_density",
                "state__mass_density",
                "trace__internal_electron_density",
                "trace__molecular_call_count",
                "trace__published_electron_from_input",
                "trace__route",
            }
        ),
        (
            "alignment__",
            "call_0__",
            "catalog__",
            "input__",
            "meta__",
            "state__",
            "trace__",
        ),
    ),
    "synthesis-fixed-supplied": RawSchema(
        156,
        frozenset(
            {
                "call_0__diag__exhausted_mask",
                "input__mass_density",
                "state__electron_density",
                "state__mass_density",
                "trace__internal_electron_density",
                "trace__molecular_call_count",
                "trace__published_electron_from_input",
                "trace__route",
            }
        ),
        (
            "alignment__",
            "call_0__",
            "catalog__",
            "input__",
            "meta__",
            "state__",
            "trace__",
        ),
    ),
    "synthesis-public": RawSchema(
        244,
        frozenset(
            {
                "call_0__diag__exhausted_mask",
                "co_reconstruction__independent_population",
                "co_reconstruction__raw_discrimination_mask",
                "fixed_state__molecular_populations",
                "ion_cube__after",
                "ion_cube__before",
                "line_population__grounded",
                "line_population__independent_no_ground",
                "line_population__no_ground",
                "line_population__public",
                "mapping__public_columns",
                "mapping__species_codes",
                "partition_cube__after",
                "partition_cube__before",
                "structured__ion_stage_populations",
                "structured__partition_normalized_populations",
                "trace__fallback_resolve_call_count",
                "trace__molecular_solve_call_count",
                "trace__route",
            }
        ),
        (
            "alignment__",
            "call_0__",
            "catalog__",
            "co_reconstruction__",
            "edge_loader__",
            "fixed_state__",
            "input__",
            "ion_cube__",
            "line_population__",
            "mapping__",
            "meta__",
            "molecular_hydrogen__",
            "partition__",
            "partition_cube__",
            "schema__",
            "structured__",
            "trace__",
        ),
    ),
}


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def atmosphere_raw_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Match the accepted atmosphere worker's sorted NPY-member fingerprint."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        buffer = io.BytesIO()
        np.lib.format.write_array(
            buffer,
            np.asarray(arrays[name]),
            allow_pickle=False,
        )
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(buffer.getvalue())
    return digest.hexdigest()


def synthesis_raw_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Match the accepted synthesis name/dtype/shape/C-bytes digest."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def raw_result_digest(
    route: str,
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Dispatch to the exact accepted algorithm for one worker family."""

    if route == "atmosphere":
        return atmosphere_raw_digest(arrays)
    if route in ACCEPTED_RAW_DIGESTS:
        return synthesis_raw_digest(arrays)
    raise ValueError(f"unknown raw digest route {route!r}")


def validate_raw_digest(
    route: str,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Reject any value, dtype, shape, or member-name drift."""

    actual = raw_result_digest(route, arrays)
    expected = ACCEPTED_RAW_DIGESTS[route]
    if actual != expected:
        raise AssertionError(
            f"{route} raw digest is {actual}; accepted digest is {expected}"
        )


def _scalar(value: Any) -> Any:
    """Return one scalar from a zero-dimensional array."""

    return np.asarray(value).item()


def _copy_mapping(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return a sorted, detached, object-free array mapping."""

    copied: dict[str, np.ndarray] = {}
    for name in sorted(arrays):
        if not isinstance(name, str) or not name:
            raise TypeError("archive member names must be nonempty strings")
        value = np.asarray(arrays[name])
        if value.dtype.hasobject:
            raise TypeError(f"{name} has forbidden object dtype")
        copied[name] = value.copy()
    return copied


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load one pickle-free archive into detached arrays."""

    with np.load(path, allow_pickle=False) as archive:
        if archive.files != sorted(archive.files):
            raise AssertionError(f"{path} members are not lexically ordered")
        return _copy_mapping({name: archive[name] for name in archive.files})


def _require_equal(
    left: np.ndarray,
    right: np.ndarray,
    description: str,
) -> None:
    if not np.array_equal(np.asarray(left), np.asarray(right)):
        raise AssertionError(f"{description} is not bitwise equal")


def _require_false(values: np.ndarray, description: str) -> None:
    if np.any(np.asarray(values, dtype=np.bool_)):
        raise AssertionError(f"{description} contains a true value")


def _require_member_alias(
    arrays: Mapping[str, np.ndarray],
    alias_name: str,
    authority_name: str,
) -> None:
    if alias_name not in arrays:
        raise AssertionError(f"archive lacks alias {alias_name}")
    if str(_scalar(arrays[alias_name])) != authority_name:
        raise AssertionError(
            f"{alias_name} points to {str(_scalar(arrays[alias_name]))!r}; "
            f"expected {authority_name!r}"
        )


def verify_static_identity() -> None:
    """Reject any source root, commit, fixture, or publisher input drift."""

    if PINNED_ROOT.resolve() != Path("/Users/ysting/payne-zero").resolve():
        raise RuntimeError("publisher pin path changed")
    completed = subprocess.run(
        ["git", "-C", str(PINNED_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != PINNED_COMMIT:
        raise RuntimeError("pinned Payne Zero commit changed")
    if sha256(FIXTURE) != FIXTURE_SHA256:
        raise RuntimeError("Chapter 4 input fixture changed")
    for name, path in PINNED_PUBLISHER_INPUT_PATHS.items():
        if not path.is_file():
            raise RuntimeError(f"required publisher input is missing: {path}")
        actual = sha256(path)
        expected = PINNED_PUBLISHER_INPUT_SHA256[name]
        if actual != expected:
            raise RuntimeError(
                f"publisher input {name} has SHA-256 {actual}; "
                f"expected {expected}"
            )


def validate_raw_schema(route: str, arrays: Mapping[str, np.ndarray]) -> None:
    """Require one exact reviewed worker envelope and scientific invariants."""

    schema = RAW_SCHEMAS[route]
    names = set(arrays)
    missing = sorted(schema.required - names)
    if missing:
        raise AssertionError(f"{route} raw capture is missing {missing}")
    if len(arrays) != schema.key_count:
        raise AssertionError(
            f"{route} raw capture has {len(arrays)} keys; "
            f"expected {schema.key_count}"
        )
    validate_raw_digest(route, arrays)
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"{route}:{name} has object dtype")
        if not name.startswith(schema.allowed_prefixes) and name not in {
            "blas_name",
            "blas_version",
            "fixture_sha256",
            "lapack_name",
            "lapack_version",
            "numba_thread_count",
            "numba_version",
            "numpy_build_dependencies_json",
            "numpy_version",
            "oracle_role",
            "payne_zero_commit",
            "platform",
            "platform_machine",
            "platform_release",
            "platform_system",
            "python_executable",
            "python_implementation",
            "python_version",
            "schema_version",
            "system_byteorder",
        }:
            raise AssertionError(f"{route} has unowned raw member {name}")
        if name.endswith("exhausted_mask") or name.endswith("_newton_exhausted"):
            _require_false(value, f"{route}:{name}")
        if np.issubdtype(array.dtype, np.inexact):
            allowed_nonfinite = route == "atmosphere" and name in {
                "disabled_fractional_doppler_widths",
                "full_fractional_doppler_widths",
                "handoff_fractional_doppler_widths",
            }
            if not allowed_nonfinite and np.any(~np.isfinite(array)):
                raise AssertionError(f"{route}:{name} contains a nonfinite value")

    if route == "atmosphere":
        _validate_atmosphere_raw(arrays)
    else:
        _validate_synthesis_raw(route, arrays)


def _validate_atmosphere_raw(arrays: Mapping[str, np.ndarray]) -> None:
    if not bool(_scalar(arrays["capture_in_memory_scope_complete"])):
        raise AssertionError("atmosphere in-memory capture is incomplete")
    if bool(_scalar(arrays["capture_scope_complete"])):
        raise AssertionError("worker unexpectedly claimed publication")
    for name in (
        "capture_wrappers_restored",
        "capture_numpy_linalg_wrappers_restored",
        "capture_population_schedule_wrappers_restored",
        "deferred_golden_publication",
    ):
        if not bool(_scalar(arrays[name])):
            raise AssertionError(f"atmosphere worker flag {name} is false")
    if arrays["full_molecular_populations"].shape != (6, 170):
        raise AssertionError("atmosphere molecular shape changed")
    if arrays["full_transformed_molecular_equation_densities"].shape != (6, 23):
        raise AssertionError("atmosphere equation shape changed")
    if arrays["full_newton_preupdate_equation_densities"].shape != (38, 23):
        raise AssertionError("atmosphere Newton history changed")
    if arrays["full_newton_raw_corrections"].shape != (38, 23):
        raise AssertionError("atmosphere Newton correction history changed")
    if int(_scalar(arrays["full_newton_np_linalg_solve_call_count"])) != 38:
        raise AssertionError("atmosphere direct solve count changed")
    if int(_scalar(arrays["full_newton_np_linalg_lstsq_call_count"])) != 0:
        raise AssertionError("atmosphere fallback solve count changed")
    if int(_scalar(arrays["priming_cache_zero_code_call_count"])) != 1:
        raise AssertionError("zero-code priming count changed")
    if int(_scalar(arrays["priming_cache_additional_solve_count_during_schedule"])) != 0:
        raise AssertionError("packed schedule performed a molecular re-solve")
    if int(_scalar(arrays["schedule_inventory_job_count"])) != 230:
        raise AssertionError("packed schedule size changed")
    if int(_scalar(arrays["schedule_inventory_atomic_job_count"])) != 198:
        raise AssertionError("packed atomic schedule size changed")
    if int(_scalar(arrays["schedule_inventory_molecular_job_count"])) != 32:
        raise AssertionError("packed molecular schedule size changed")
    _require_equal(
        arrays["schedule_inventory_molecular_unique_code"],
        EXPECTED_ATMOSPHERE_SCHEDULE_CODES,
        "atmosphere molecular schedule codes",
    )
    _require_equal(
        arrays[
            "schedule_inventory_molecular_unique_packed_start_slot_one_based"
        ],
        EXPECTED_ATMOSPHERE_SCHEDULE_SLOTS,
        "atmosphere molecular schedule packed slots",
    )
    _require_equal(
        arrays["schedule_inventory_molecular_unique_mode_pair"],
        np.tile(np.asarray([1, 11], dtype=np.int64), (16, 1)),
        "atmosphere molecular schedule mode pairs",
    )
    _require_equal(
        arrays["boundary_temperature_uint64_bits"],
        np.asarray(
            [
                4666723172467343360,
                4666723172467343361,
                4671226772094713856,
                4671226772094713857,
            ],
            dtype=np.uint64,
        ),
        "atmosphere boundary bit patterns",
    )
    _require_equal(
        arrays["boundary_polynomial_active_branch_mask"],
        np.asarray([True, False, False, False]),
        "atmosphere polynomial branch mask",
    )
    _require_equal(
        arrays["boundary_h2_catalog_active_branch_mask"],
        np.asarray([True, True, True, False]),
        "atmosphere H2 catalog branch mask",
    )

    for prefix in ("full", "handoff", "disabled"):
        doppler = np.asarray(arrays[f"{prefix}_fractional_doppler_widths"])
        expected = np.zeros(doppler.shape, dtype=np.bool_)
        expected[:, [919, 927]] = True
        if prefix == "full":
            mask = np.asarray(
                arrays[
                    "full_fractional_doppler_width_structural_infinity_mask"
                ],
                dtype=np.bool_,
            )
            _require_equal(mask, expected, "atmosphere structural infinity mask")
        if not np.all(np.isposinf(doppler[expected])):
            raise AssertionError(
                f"{prefix} structural Doppler slots are not positive infinity"
            )
        if np.any(~np.isfinite(doppler[~expected])):
            raise AssertionError(f"unexpected nonfinite {prefix} Doppler value")


def _validate_synthesis_raw(
    route: str,
    arrays: Mapping[str, np.ndarray],
) -> None:
    if int(_scalar(arrays["alignment__shared_count"])) != 170:
        raise AssertionError("shared molecular catalog count changed")
    if int(_scalar(arrays["alignment__synthesis_only_count"])) != 20:
        raise AssertionError("synthesis-only molecular catalog count changed")
    if int(_scalar(arrays["alignment__semantic_mismatch_count"])) != 0:
        raise AssertionError("shared molecular catalog semantics changed")
    if int(_scalar(arrays["alignment__atmosphere_only_count"])) != 0:
        raise AssertionError("an atmosphere-only molecular record appeared")
    _require_equal(
        arrays["alignment__synthesis_only_molecule_codes"],
        EXPECTED_SYNTHESIS_ONLY_CODES,
        "synthesis-only molecular codes",
    )
    if not np.any(arrays["alignment__shared_row_indices_differ_mask"]):
        raise AssertionError("catalog alignment accidentally became row based")

    if route == "synthesis-boundaries":
        if str(_scalar(arrays["trace__route"])) != "synthesis_boundary_probes":
            raise AssertionError("synthesis boundary route label changed")
        _require_false(
            arrays["boundary__provisional_h2_molecular_exhausted_mask"],
            "provisional H2 boundary solve",
        )
        _require_equal(
            arrays["boundary__synthesis_polynomial_temperature_bits"],
            np.asarray(
                [4666723172467343360, 4666723172467343361],
                dtype=np.uint64,
            ),
            "synthesis polynomial boundary bits",
        )
        _require_equal(
            arrays["boundary__synthesis_polynomial_temperature_branch_mask"],
            np.asarray([True, False]),
            "synthesis polynomial boundary mask",
        )
        _require_equal(
            arrays["boundary__provisional_h2_temperature_bits"],
            np.asarray(
                [4666173416653455360, 4666173416653455361],
                dtype=np.uint64,
            ),
            "synthesis H2 boundary bits",
        )
        _require_equal(
            arrays["boundary__provisional_h2_temperature_branch_mask"],
            np.asarray([True, False]),
            "synthesis H2 boundary mask",
        )
        return

    if route == "synthesis-full":
        if int(_scalar(arrays["trace__molecular_call_count"])) != 2:
            raise AssertionError("full synthesis route is no longer two-solve")
        if str(_scalar(arrays["call_0__caller_name"])) != "solve_electron_density":
            raise AssertionError("full synthesis call 0 owner changed")
        if (
            str(_scalar(arrays["call_1__caller_name"]))
            != "_molecule_backed_population_state"
        ):
            raise AssertionError("full synthesis call 1 owner changed")
        _require_equal(
            arrays["state__electron_density"],
            arrays["call_0__output__electron_density"],
            "full public electron ownership",
        )
        _require_equal(
            arrays["state__molecular_populations"],
            arrays["call_1__output__molecular_populations"],
            "full public molecule ownership",
        )
        if arrays["state__molecular_populations"].shape != (6, 200):
            raise AssertionError("full synthesis molecular padding changed")
        if arrays["state__molecular_equation_densities"].shape != (6, 30):
            raise AssertionError("full synthesis equation padding changed")
        return

    if route in {"synthesis-fixed-derived", "synthesis-fixed-supplied"}:
        if int(_scalar(arrays["trace__molecular_call_count"])) != 1:
            raise AssertionError(f"{route} is no longer one-solve")
        if not bool(_scalar(arrays["trace__published_electron_from_input"])):
            raise AssertionError(f"{route} lost fixed-electron ownership")
        _require_equal(
            arrays["state__electron_density"],
            arrays["input__electron_density_seed"],
            f"{route} public electron density",
        )
        if route == "synthesis-fixed-supplied":
            _require_equal(
                arrays["state__mass_density"],
                arrays["input__mass_density"],
                "supplied fixed mass density",
            )
        return

    if int(_scalar(arrays["trace__fixed_eos_call_count"])) != 1:
        raise AssertionError("public fixed EOS call count changed")
    if int(_scalar(arrays["trace__molecular_solve_call_count"])) != 1:
        raise AssertionError("public molecular call count changed")
    if int(_scalar(arrays["trace__partition_no_ground_call_count"])) != 1:
        raise AssertionError("public no-ground partition call count changed")
    if int(_scalar(arrays["trace__line_mapping_production_call_count"])) != 1:
        raise AssertionError("public line-mapping call count changed")
    if int(_scalar(arrays["trace__fallback_resolve_call_count"])) != 0:
        raise AssertionError("public builder used a fallback molecular solve")
    species = np.asarray(arrays["mapping__species_codes"])
    columns = np.asarray(arrays["mapping__public_columns"])
    if species.shape != (54,) or np.unique(species).size != 54:
        raise AssertionError("public molecular species mapping changed")
    if int(np.count_nonzero(columns < 99)) != 51:
        raise AssertionError("public molecular mapping lost the 51/3 split")
    if int(np.count_nonzero(columns >= 99)) != 3:
        raise AssertionError("public molecular mapping lost the 51/3 split")
    _require_equal(
        arrays["ion_cube__before"],
        arrays["ion_cube__after"],
        "public actual ion cube",
    )
    _require_equal(
        arrays["line_population__public"],
        arrays["line_population__no_ground"],
        "public no-ground molecular lanes",
    )
    _require_equal(
        arrays["line_population__public"],
        arrays["line_population__independent_no_ground"],
        "independent molecular lanes",
    )
    if int(np.count_nonzero(arrays["co_reconstruction__raw_discrimination_mask"])) != 6:
        raise AssertionError("CO raw/normalized discrimination changed")
    if int(_scalar(arrays["co_reconstruction__line_species_code"])) != 276:
        raise AssertionError("CO public line code changed")
    if float(_scalar(arrays["co_reconstruction__equilibrium_code"])) != 608.0:
        raise AssertionError("CO equilibrium code changed")
    if int(_scalar(arrays["co_reconstruction__public_stage_index"])) != 5:
        raise AssertionError("CO public stage changed")
    if int(_scalar(arrays["co_reconstruction__public_column_index"])) != 45:
        raise AssertionError("CO public column changed")


def _publisher_metadata(kind: str) -> dict[str, np.ndarray]:
    return {
        "meta__archive_kind": np.asarray(kind),
        "meta__archive_schema_version": np.asarray(1, dtype=np.int64),
        "meta__atmosphere_worker_sha256": np.asarray(
            PINNED_PUBLISHER_INPUT_SHA256["atmosphere_worker"]
        ),
        "meta__capture_contract_sha256": np.asarray(
            PINNED_PUBLISHER_INPUT_SHA256["capture_contract"]
        ),
        "meta__deterministic_npz_sha256": np.asarray(
            PINNED_PUBLISHER_INPUT_SHA256["deterministic_npz"]
        ),
        "meta__fixture_sha256": np.asarray(FIXTURE_SHA256),
        "meta__oracle_acceptance_sha256": np.asarray(
            PINNED_PUBLISHER_INPUT_SHA256["oracle_acceptance"]
        ),
        "meta__payne_zero_commit": np.asarray(PINNED_COMMIT),
        "meta__publisher_sha256": np.asarray(sha256(Path(__file__).resolve())),
        "meta__synthesis_worker_sha256": np.asarray(
            PINNED_PUBLISHER_INPUT_SHA256["synthesis_worker"]
        ),
    }


def _prefix_selected(
    arrays: Mapping[str, np.ndarray],
    prefix: str,
    predicate: Callable[[str], bool],
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}{name}": np.asarray(value).copy()
        for name, value in arrays.items()
        if predicate(name)
    }


def _validate_catalog_overlap(
    atmosphere: Mapping[str, np.ndarray],
    synthesis: Mapping[str, np.ndarray],
) -> None:
    for name in (
        "component_count",
        "component_equation_indices",
        "component_start_indices",
        "equation_count",
        "equation_species_codes",
        "equilibrium_coefficients",
        "molecule_codes",
        "molecule_count",
        "species_to_equation_index",
    ):
        _require_equal(
            atmosphere[f"atmosphere_{name}"],
            synthesis[f"catalog__atmosphere__{name}"],
            f"atmosphere catalog {name}",
        )


def build_constants_payload(
    atmosphere: Mapping[str, np.ndarray],
    boundaries: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Own catalogs, semantic alignment, exact boundaries, and identities."""

    _validate_catalog_overlap(atmosphere, boundaries)
    payload = _publisher_metadata("molecular_constants")
    payload.update(
        _prefix_selected(
            atmosphere,
            "atmosphere__",
            lambda name: (
                name.startswith(ATMOSPHERE_IDENTITY_PREFIXES)
                or name in ATMOSPHERE_IDENTITY_NAMES
                or name.startswith(("boundary_", "h2_probe_", "named_"))
            ),
        )
    )
    payload.update(
        _prefix_selected(
            boundaries,
            "synthesis__",
            lambda name: name.startswith(
                ("alignment__", "boundary__", "catalog__", "meta__", "trace__")
            ),
        )
    )
    return _copy_mapping(payload)


def build_atmosphere_payload(
    atmosphere: Mapping[str, np.ndarray],
    constants_sha256: str,
) -> dict[str, np.ndarray]:
    """Remove invariant catalogs and boundary probes from the route archive."""

    payload = _publisher_metadata("atmosphere_molecular_state")
    payload["meta__constants_archive_sha256"] = np.asarray(constants_sha256)
    for name, value in atmosphere.items():
        if name.startswith(ATMOSPHERE_ROUTE_PREFIXES):
            payload[name] = np.asarray(value).copy()
        elif name.startswith(ATMOSPHERE_IDENTITY_PREFIXES) or (
            name in ATMOSPHERE_IDENTITY_NAMES
        ):
            payload[f"oracle__{name}"] = np.asarray(value).copy()
    return _copy_mapping(payload)


def _remove_common_call_inputs(
    payload: dict[str, np.ndarray],
    *,
    call_prefixes: Sequence[str],
    route_specific_aliases: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    aliases_by_call = route_specific_aliases or {}
    for call_prefix in call_prefixes:
        aliases = dict(COMMON_CALL_INPUT_ALIASES)
        aliases.update(aliases_by_call.get(call_prefix, {}))
        for suffix, common_name in aliases.items():
            name = f"{call_prefix}{suffix}"
            if name not in payload:
                raise AssertionError(f"missing common call input {name}")
            _require_equal(
                payload[name],
                payload[common_name],
                f"{name} common route input",
            )
            del payload[name]
            payload[f"{name}_member"] = np.asarray(common_name)


def _remove_state_aliases(
    payload: dict[str, np.ndarray],
    aliases: Mapping[str, str],
) -> None:
    for state_name, authority_name in aliases.items():
        _require_equal(
            payload[state_name],
            payload[authority_name],
            f"{state_name} authority",
        )
        del payload[state_name]
        payload[f"alias__{state_name}_member"] = np.asarray(authority_name)


def _route_without_constants(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value).copy()
        for name, value in arrays.items()
        if not name.startswith(("alignment__", "catalog__", "meta__"))
    }


def _route_provenance(
    arrays: Mapping[str, np.ndarray],
    prefix: str = "oracle__",
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}{name}": np.asarray(value).copy()
        for name, value in arrays.items()
        if name.startswith("meta__")
    }


def build_full_payload(
    full: Mapping[str, np.ndarray],
    constants_sha256: str,
) -> dict[str, np.ndarray]:
    payload = _publisher_metadata("synthesis_molecular_full")
    payload["meta__constants_archive_sha256"] = np.asarray(constants_sha256)
    payload.update(_route_provenance(full))
    payload.update(_route_without_constants(full))
    _remove_common_call_inputs(
        payload,
        call_prefixes=("call_0__", "call_1__"),
        route_specific_aliases={
            "call_1__": {
                "input__electron_density": (
                    "call_0__output__electron_density"
                )
            },
        },
    )
    _remove_state_aliases(payload, STATE_CALL_ALIASES)
    return _copy_mapping(payload)


def _common_fixed_inputs(
    derived: Mapping[str, np.ndarray],
    supplied: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {}
    for name, value in derived.items():
        if not name.startswith("input__"):
            continue
        if name == "input__mass_density":
            continue
        _require_equal(value, supplied[name], f"fixed common input {name}")
        result[name] = np.asarray(value).copy()
    return result


def _require_common_fixed_provenance(
    derived: Mapping[str, np.ndarray],
    supplied: Mapping[str, np.ndarray],
) -> None:
    derived_meta = {name: value for name, value in derived.items() if name.startswith("meta__")}
    supplied_meta = {
        name: value for name, value in supplied.items() if name.startswith("meta__")
    }
    if set(derived_meta) != set(supplied_meta):
        raise AssertionError("fixed route provenance keys differ")
    for name in derived_meta:
        _require_equal(derived_meta[name], supplied_meta[name], f"fixed provenance {name}")


def _fixed_branch(
    arrays: Mapping[str, np.ndarray],
    branch: str,
) -> dict[str, np.ndarray]:
    branch_payload = _route_without_constants(arrays)
    for name in tuple(branch_payload):
        if name.startswith("input__") and name != "input__mass_density":
            del branch_payload[name]
    common_inputs = {
        name: np.asarray(value).copy()
        for name, value in arrays.items()
        if name.startswith("input__") and name != "input__mass_density"
    }
    branch_payload.update(common_inputs)
    _remove_common_call_inputs(
        branch_payload,
        call_prefixes=("call_0__",),
        route_specific_aliases={
            "call_0__": {
                "input__electron_density": "input__electron_density_seed"
            }
        },
    )
    _remove_state_aliases(branch_payload, FIXED_STATE_CALL_ALIASES)
    _remove_state_aliases(
        branch_payload,
        {
            "trace__internal_electron_density": (
                "call_0__output__electron_density"
            )
        },
    )
    for name in tuple(branch_payload):
        if name in common_inputs:
            del branch_payload[name]
    prefixed = {}
    for name, value in branch_payload.items():
        copied = np.asarray(value).copy()
        if name.startswith("alias__") and copied.shape == ():
            authority = str(copied.item())
            if authority.startswith("call_0__"):
                copied = np.asarray(f"{branch}__{authority}")
        prefixed[f"{branch}__{name}"] = copied
    return prefixed


def build_fixed_payload(
    derived: Mapping[str, np.ndarray],
    supplied: Mapping[str, np.ndarray],
    constants_sha256: str,
) -> dict[str, np.ndarray]:
    _require_common_fixed_provenance(derived, supplied)
    payload = _publisher_metadata("synthesis_molecular_fixed")
    payload["meta__constants_archive_sha256"] = np.asarray(constants_sha256)
    payload.update(_route_provenance(derived))
    payload.update(_common_fixed_inputs(derived, supplied))
    payload.update(_fixed_branch(derived, "derived"))
    payload.update(_fixed_branch(supplied, "supplied"))
    return _copy_mapping(payload)


def _deduplicate_public_payload(payload: dict[str, np.ndarray]) -> None:
    _remove_common_call_inputs(
        payload,
        call_prefixes=("call_0__",),
        route_specific_aliases={
            "call_0__": {
                "input__electron_density": "input__electron_density_seed"
            }
        },
    )
    _remove_state_aliases(payload, PUBLIC_FIXED_STATE_CALL_ALIASES)

    aliases = {
        "ion_cube__before": "fixed_state__ion_stage_populations",
        "ion_cube__after": "structured__ion_stage_populations",
        "partition_cube__before": (
            "fixed_state__partition_normalized_populations"
        ),
        "partition_cube__after": (
            "structured__partition_normalized_populations"
        ),
        "line_population__no_ground": "line_population__public",
        "co_reconstruction__reference_public_lane": (
            "co_reconstruction__independent_population"
        ),
    }
    for duplicate, authority in aliases.items():
        _require_equal(payload[duplicate], payload[authority], duplicate)
        del payload[duplicate]
        payload[f"alias__{duplicate}_member"] = np.asarray(authority)

    if np.any(payload["co_reconstruction__difference"]):
        raise AssertionError("CO independent reconstruction difference is nonzero")
    del payload["co_reconstruction__difference"]

    species_index = int(
        np.flatnonzero(payload["mapping__species_codes"] == 276)[0]
    )
    _require_equal(
        payload["co_reconstruction__grounded_population"],
        payload["line_population__grounded"][:, species_index],
        "CO grounded population",
    )
    del payload["co_reconstruction__grounded_population"]
    payload["alias__co_reconstruction__grounded_population_member"] = np.asarray(
        f"line_population__grounded[:,{species_index}]"
    )

    h2_index = int(_scalar(payload["molecular_hydrogen__catalog_index"]))
    _require_equal(
        payload["molecular_hydrogen__solved_code_101"],
        payload["call_0__output__molecular_populations"][:, h2_index],
        "solved molecular hydrogen",
    )
    del payload["molecular_hydrogen__solved_code_101"]
    payload["alias__molecular_hydrogen__solved_code_101_member"] = np.asarray(
        f"call_0__output__molecular_populations[:,{h2_index}]"
    )

    edge_aliases = {
        "edge_loader__signed_continuum_edge_frequency_hz": (
            "structured__signed_continuum_edge_frequency_hz"
        ),
        "edge_loader__continuum_edge_wavelength_nm": (
            "structured__continuum_edge_wavelength_nm"
        ),
        "edge_loader__continuum_edge_midpoint_wavelength_nm": (
            "structured__continuum_edge_midpoint_wavelength_nm"
        ),
        "edge_loader__edge_interval_width_squared_over_two_nm2": (
            "structured__continuum_edge_interval_width_squared_over_two_nm2"
        ),
    }
    for duplicate, authority in edge_aliases.items():
        _require_equal(payload[duplicate], payload[authority], duplicate)
        del payload[duplicate]
        payload[f"alias__{duplicate}_member"] = np.asarray(authority)


def build_public_payload(
    public: Mapping[str, np.ndarray],
    constants_sha256: str,
) -> dict[str, np.ndarray]:
    payload = _publisher_metadata("molecular_public_mapping")
    payload["meta__constants_archive_sha256"] = np.asarray(constants_sha256)
    payload.update(_route_provenance(public))
    payload.update(_route_without_constants(public))
    _deduplicate_public_payload(payload)
    return _copy_mapping(payload)


def validate_final_payload(
    name: str,
    arrays: Mapping[str, np.ndarray],
    *,
    constants_sha256: str | None,
) -> None:
    """Validate ownership, provenance, and deduplication in one final payload."""

    required_meta = {
        "meta__archive_kind",
        "meta__archive_schema_version",
        "meta__atmosphere_worker_sha256",
        "meta__capture_contract_sha256",
        "meta__deterministic_npz_sha256",
        "meta__fixture_sha256",
        "meta__oracle_acceptance_sha256",
        "meta__payne_zero_commit",
        "meta__publisher_sha256",
        "meta__synthesis_worker_sha256",
    }
    missing = sorted(required_meta - set(arrays))
    if missing:
        raise AssertionError(f"{name} lacks publisher metadata {missing}")
    if str(_scalar(arrays["meta__payne_zero_commit"])) != PINNED_COMMIT:
        raise AssertionError(f"{name} has the wrong source commit")
    if str(_scalar(arrays["meta__fixture_sha256"])) != FIXTURE_SHA256:
        raise AssertionError(f"{name} has the wrong fixture identity")
    expected_kind = EXPECTED_ARCHIVE_KINDS[name]
    if str(_scalar(arrays["meta__archive_kind"])) != expected_kind:
        raise AssertionError(
            f"{name} has archive kind "
            f"{str(_scalar(arrays['meta__archive_kind']))!r}; "
            f"expected {expected_kind!r}"
        )
    if int(_scalar(arrays["meta__archive_schema_version"])) != 1:
        raise AssertionError(f"{name} has the wrong archive schema version")
    exact_metadata = {
        "meta__atmosphere_worker_sha256": (
            PINNED_PUBLISHER_INPUT_SHA256["atmosphere_worker"]
        ),
        "meta__capture_contract_sha256": (
            PINNED_PUBLISHER_INPUT_SHA256["capture_contract"]
        ),
        "meta__deterministic_npz_sha256": (
            PINNED_PUBLISHER_INPUT_SHA256["deterministic_npz"]
        ),
        "meta__oracle_acceptance_sha256": (
            PINNED_PUBLISHER_INPUT_SHA256["oracle_acceptance"]
        ),
        "meta__publisher_sha256": sha256(Path(__file__).resolve()),
        "meta__synthesis_worker_sha256": (
            PINNED_PUBLISHER_INPUT_SHA256["synthesis_worker"]
        ),
    }
    for member, expected in exact_metadata.items():
        if str(_scalar(arrays[member])) != expected:
            raise AssertionError(f"{name} has the wrong {member}")
    for member, values in arrays.items():
        if np.asarray(values).dtype.hasobject:
            raise TypeError(f"{name}:{member} has object dtype")

    if name == CONSTANTS_NAME:
        if "meta__constants_archive_sha256" in arrays:
            raise AssertionError("constants archive contains a cyclic self-hash")
        for prefix in (
            "atmosphere__boundary_",
            "atmosphere__h2_probe_",
            "synthesis__alignment__",
            "synthesis__boundary__",
            "synthesis__catalog__",
        ):
            if not any(member.startswith(prefix) for member in arrays):
                raise AssertionError(f"constants archive lacks {prefix}")
        return

    if constants_sha256 is None:
        raise AssertionError(f"{name} was validated without constants identity")
    if str(_scalar(arrays["meta__constants_archive_sha256"])) != constants_sha256:
        raise AssertionError(f"{name} has the wrong constants archive identity")
    if any(member.startswith(("catalog__", "alignment__")) for member in arrays):
        raise AssertionError(f"{name} duplicates constants-owned catalog arrays")

    if name == OUTPUT_NAMES[1]:
        if any(member.startswith(("boundary_", "h2_probe_")) for member in arrays):
            raise AssertionError("atmosphere state duplicates constants boundaries")
        if any(member.startswith("named_") for member in arrays):
            raise AssertionError("atmosphere state duplicates named catalog mapping")
        if not all(
            any(member.startswith(prefix) for member in arrays)
            for prefix in ATMOSPHERE_ROUTE_PREFIXES
        ):
            raise AssertionError("atmosphere state route coverage is incomplete")
    elif name == OUTPUT_NAMES[2]:
        for member in (
            "alias__state__electron_density_member",
            "alias__state__molecular_populations_member",
            "call_0__diag__iterations_completed",
            "call_1__diag__iterations_completed",
        ):
            if member not in arrays:
                raise AssertionError(f"full synthesis archive lacks {member}")
        for duplicate in (
            "state__electron_density",
            "call_1__input__electron_density",
        ):
            if duplicate in arrays:
                raise AssertionError(
                    f"full synthesis archive retained duplicate {duplicate}"
                )
        _require_member_alias(
            arrays,
            "alias__state__electron_density_member",
            "call_0__output__electron_density",
        )
        _require_member_alias(
            arrays,
            "call_1__input__electron_density_member",
            "call_0__output__electron_density",
        )
    elif name == OUTPUT_NAMES[3]:
        if not any(member.startswith("derived__") for member in arrays):
            raise AssertionError("fixed archive lacks derived branch")
        if not any(member.startswith("supplied__") for member in arrays):
            raise AssertionError("fixed archive lacks supplied branch")
        if "supplied__input__mass_density" not in arrays:
            raise AssertionError("fixed archive lacks supplied mass input")
        for branch in ("derived", "supplied"):
            for duplicate in (
                "state__electron_density",
                "call_0__input__electron_density",
                "trace__internal_electron_density",
            ):
                member = f"{branch}__{duplicate}"
                if member in arrays:
                    raise AssertionError(
                        f"fixed archive retained duplicate {member}"
                    )
            for alias in (
                "alias__state__electron_density_member",
                "call_0__input__electron_density_member",
                "alias__trace__internal_electron_density_member",
            ):
                member = f"{branch}__{alias}"
                if member not in arrays:
                    raise AssertionError(f"fixed archive lacks alias {member}")
            _require_member_alias(
                arrays,
                f"{branch}__alias__state__electron_density_member",
                "input__electron_density_seed",
            )
            _require_member_alias(
                arrays,
                f"{branch}__call_0__input__electron_density_member",
                "input__electron_density_seed",
            )
            _require_member_alias(
                arrays,
                (
                    f"{branch}__alias__"
                    "trace__internal_electron_density_member"
                ),
                f"{branch}__call_0__output__electron_density",
            )
    elif name == OUTPUT_NAMES[4]:
        if sum(member.startswith("structured__") for member in arrays) != 27:
            raise AssertionError("public archive does not contain 27 structured arrays")
        for forbidden in (
            "ion_cube__before",
            "ion_cube__after",
            "partition_cube__before",
            "partition_cube__after",
            "line_population__no_ground",
            "co_reconstruction__difference",
            "molecular_hydrogen__solved_code_101",
        ):
            if forbidden in arrays:
                raise AssertionError(f"public archive retained duplicate {forbidden}")
        for duplicate in (
            "fixed_state__electron_density",
            "call_0__input__electron_density",
        ):
            if duplicate in arrays:
                raise AssertionError(
                    f"public archive retained duplicate {duplicate}"
                )
        for alias in (
            "alias__fixed_state__electron_density_member",
            "call_0__input__electron_density_member",
        ):
            if alias not in arrays:
                raise AssertionError(f"public archive lacks alias {alias}")
        _require_member_alias(
            arrays,
            "alias__fixed_state__electron_density_member",
            "input__electron_density_seed",
        )
        _require_member_alias(
            arrays,
            "call_0__input__electron_density_member",
            "input__electron_density_seed",
        )


def assemble_capture_set(raw_dir: Path, final_dir: Path) -> None:
    """Assemble five compact deterministic archives from six raw routes."""

    raw = {}
    for route in ROUTES:
        arrays = load_npz(raw_dir / f"{route}.npz")
        validate_raw_schema(route, arrays)
        raw[route] = arrays

    final_dir.mkdir(parents=True, exist_ok=False)
    constants = build_constants_payload(
        raw["atmosphere"],
        raw["synthesis-boundaries"],
    )
    validate_final_payload(CONSTANTS_NAME, constants, constants_sha256=None)
    constants_path = final_dir / CONSTANTS_NAME
    write_npz(constants_path, constants)
    constants_hash = sha256(constants_path)

    payloads = {
        OUTPUT_NAMES[1]: build_atmosphere_payload(
            raw["atmosphere"], constants_hash
        ),
        OUTPUT_NAMES[2]: build_full_payload(
            raw["synthesis-full"], constants_hash
        ),
        OUTPUT_NAMES[3]: build_fixed_payload(
            raw["synthesis-fixed-derived"],
            raw["synthesis-fixed-supplied"],
            constants_hash,
        ),
        OUTPUT_NAMES[4]: build_public_payload(
            raw["synthesis-public"], constants_hash
        ),
    }
    for name in OUTPUT_NAMES[1:]:
        validate_final_payload(
            name,
            payloads[name],
            constants_sha256=constants_hash,
        )
        write_npz(final_dir / name, payloads[name])
    validate_final_directory(final_dir)


def validate_final_directory(directory: Path) -> None:
    """Require exactly five independently valid final archives."""

    actual = tuple(sorted(path.name for path in directory.iterdir()))
    expected = tuple(sorted(OUTPUT_NAMES))
    if actual != expected:
        raise AssertionError(
            f"final archive set is {actual}; expected {expected}"
        )
    constants_hash = sha256(directory / CONSTANTS_NAME)
    for name in OUTPUT_NAMES:
        validate_final_payload(
            name,
            load_npz(directory / name),
            constants_sha256=None if name == CONSTANTS_NAME else constants_hash,
        )


def _child_environment(cache_dir: Path) -> dict[str, str]:
    """Return a complete pre-import environment for one route child."""

    environment = os.environ.copy()
    environment.update(PROCESS_CONTROLS)
    environment.update(
        {
            "NUMBA_CACHE_DIR": str(cache_dir.resolve()),
            "PAYNE_ZERO_DATA_ROOT": str(PINNED_ROOT / "source_data_files"),
            "PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE": str(
                PINNED_ROOT
                / "source_data_files"
                / "synthesis_tables"
                / "atomic_masses.npz"
            ),
            "PYTHONPATH": os.pathsep.join(
                (str(PINNED_ROOT), str(REPOSITORY_ROOT))
            ),
        }
    )
    return environment


def run_route_child(route: str, raw_path: Path, cache_dir: Path) -> None:
    """Launch one and only one scientific route in a fresh process."""

    if cache_dir.exists():
        raise RuntimeError(f"route cache is not fresh: {cache_dir}")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-route",
            route,
            "--internal-output",
            str(raw_path),
        ],
        cwd=REPOSITORY_ROOT,
        env=_child_environment(cache_dir),
        check=True,
        timeout=300,
    )


RouteRunner = Callable[[str, Path, Path], None]
Assembler = Callable[[Path, Path], None]


def build_capture_set(
    capture_root: Path,
    *,
    route_runner: RouteRunner = run_route_child,
    assembler: Assembler = assemble_capture_set,
) -> None:
    """Run all six isolated routes and assemble one five-archive capture."""

    raw_dir = capture_root / "raw"
    cache_root = capture_root / "cache"
    final_dir = capture_root / "final"
    raw_dir.mkdir(parents=True, exist_ok=False)
    cache_root.mkdir(parents=True, exist_ok=False)
    cache_paths = []
    for route in ROUTES:
        cache_dir = cache_root / route
        cache_paths.append(cache_dir.resolve())
        route_runner(route, raw_dir / f"{route}.npz", cache_dir)
    if len(set(cache_paths)) != len(ROUTES):
        raise AssertionError("route children did not receive distinct caches")
    assembler(raw_dir, final_dir)


def compare_file_sets(
    first: Path,
    second: Path,
    names: Sequence[str],
    *,
    label: str,
) -> None:
    """Require byte equality for an exact named file set."""

    expected = set(names)
    first_names = {path.name for path in first.iterdir() if path.is_file()}
    second_names = {path.name for path in second.iterdir() if path.is_file()}
    if first_names != expected or second_names != expected:
        raise AssertionError(
            f"{label} file set differs: first={sorted(first_names)}, "
            f"second={sorted(second_names)}, expected={sorted(expected)}"
        )
    for name in names:
        left = (first / name).read_bytes()
        right = (second / name).read_bytes()
        if left != right:
            limit = min(len(left), len(right))
            offset = next(
                (index for index in range(limit) if left[index] != right[index]),
                limit,
            )
            raise AssertionError(
                f"{label} {name} differs at byte {offset}; "
                f"sizes are {len(left)} and {len(right)}"
            )


DirectoryValidator = Callable[[Path], None]


def publish_verified_directory(
    source: Path,
    destination: Path = OUTPUT_DIR,
    *,
    validator: DirectoryValidator = validate_final_directory,
) -> str:
    """Atomically publish one exact five-file directory or refuse replacement."""

    validator(source)
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"publication target is not a directory: {destination}")
        compare_file_sets(
            source,
            destination,
            OUTPUT_NAMES,
            label="existing Chapter 4 publication",
        )
        validator(destination)
        return "identical-existing"

    with tempfile.TemporaryDirectory(
        prefix=".chapter04-stage-",
        dir=destination_parent,
    ) as temporary:
        stage_parent = Path(temporary)
        stage = stage_parent / "chapter04"
        shutil.copytree(source, stage)
        validator(stage)
        for name in OUTPUT_NAMES:
            with (stage / name).open("rb") as handle:
                os.fsync(handle.fileno())
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(stage, destination)
        parent_fd = os.open(destination_parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    return "published"


CaptureBuilder = Callable[[Path], None]
Publisher = Callable[[Path, Path], str]


def generate_and_maybe_publish(
    *,
    verify_only: bool,
    destination: Path = OUTPUT_DIR,
    capture_builder: CaptureBuilder = build_capture_set,
    publisher: Publisher = publish_verified_directory,
) -> dict[str, Any]:
    """Build two complete sets, compare raw and final bytes, then maybe publish."""

    with (
        tempfile.TemporaryDirectory(prefix="chapter04-capture-a-") as first_tmp,
        tempfile.TemporaryDirectory(prefix="chapter04-capture-b-") as second_tmp,
    ):
        first = Path(first_tmp)
        second = Path(second_tmp)
        capture_builder(first)
        capture_builder(second)
        compare_file_sets(
            first / "raw",
            second / "raw",
            RAW_NAMES,
            label="raw capture",
        )
        compare_file_sets(
            first / "final",
            second / "final",
            OUTPUT_NAMES,
            label="assembled capture",
        )
        archive_records = {
            name: {
                "sha256": sha256(first / "final" / name),
                "bytes": (first / "final" / name).stat().st_size,
            }
            for name in OUTPUT_NAMES
        }
        status = (
            "verified-only"
            if verify_only
            else publisher(first / "final", destination)
        )
    return {"status": status, "archives": archive_records}


def _capture_internal_route(route: str, output: Path) -> None:
    """Run one reviewed worker route and write one temporary raw NPZ."""

    mismatches = {
        name: os.environ.get(name)
        for name, expected in PROCESS_CONTROLS.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "internal route lacks frozen pre-import controls: "
            + json.dumps(mismatches, sort_keys=True)
        )
    cache_value = os.environ.get("NUMBA_CACHE_DIR")
    if not cache_value:
        raise RuntimeError("internal route lacks a fresh Numba cache")
    cache_dir = Path(cache_value)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise RuntimeError("internal route Numba cache is not fresh")
    root_text = str(PINNED_ROOT)
    if not sys.path or sys.path[0] != root_text:
        sys.path.insert(0, root_text)

    if route == "atmosphere":
        from scripts import chapter04_atmosphere_oracle_worker as worker

        arrays = worker.build_atmosphere_oracle_results(
            pinned_root=PINNED_ROOT,
            fixture_path=FIXTURE,
        )
    else:
        from scripts import chapter04_synthesis_oracle_worker as worker

        inputs = worker.load_input_fixture(FIXTURE)
        runtime = worker.load_pinned_runtime(PINNED_ROOT)
        if route == "synthesis-boundaries":
            arrays = worker.build_boundary_probe_result(
                runtime, inputs, fixture_path=FIXTURE
            )
        elif route == "synthesis-full":
            arrays = worker.build_full_route_result(
                runtime, inputs, fixture_path=FIXTURE
            )
        elif route == "synthesis-fixed-derived":
            arrays = worker.build_fixed_route_result(
                runtime, inputs, fixture_path=FIXTURE
            )
        elif route == "synthesis-fixed-supplied":
            arrays = worker.build_fixed_route_result(
                runtime,
                inputs,
                mass_density=worker._deterministic_supplied_mass_density(inputs),
                fixture_path=FIXTURE,
            )
        elif route == "synthesis-public":
            arrays = worker.build_public_mapping_result(
                runtime, inputs, fixture_path=FIXTURE
            )
        else:
            raise ValueError(f"unknown internal route {route!r}")
    normalized = _copy_mapping(arrays)
    validate_raw_schema(route, normalized)
    write_npz(output, normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="build byte-reproducible Chapter 4 pinned comparison archives"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="perform both complete captures and comparisons without publishing",
    )
    parser.add_argument(
        "--internal-route",
        choices=ROUTES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-output",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.internal_route is not None:
        if arguments.internal_output is None:
            raise SystemExit("--internal-output is required for an internal route")
        verify_static_identity()
        _capture_internal_route(
            arguments.internal_route,
            arguments.internal_output.resolve(),
        )
        return
    if arguments.internal_output is not None:
        raise SystemExit("--internal-output requires --internal-route")

    verify_static_identity()
    summary = generate_and_maybe_publish(verify_only=arguments.verify_only)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
