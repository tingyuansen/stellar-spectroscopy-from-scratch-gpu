#!/usr/bin/env python3
"""Observe the pinned Chapter 6 atmosphere one-line route in memory.

The worker accepts only the canonical, manifest-bound 19-member atmosphere
fixture.  It reconstructs the three public-call arrays, executes the standard
selected-line route, derives the once-stimulated downstream view, and records
the algorithmic seam evidence required by the Chapter 6 oracle plan.

There is deliberately no output-path, archive-writing, golden-reading, or
publication interface.  The only persistent writes permitted during a run are
Numba's files below a caller-owned, fresh, external ``NUMBA_CACHE_DIR``.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence


THREAD_ENVIRONMENT = {
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

ORACLE_ENVIRONMENT = {
    **THREAD_ENVIRONMENT,
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "TZ": "UTC",
}


class OracleEnvironmentError(RuntimeError):
    """Raised when deterministic fresh-process controls are absent."""


class OracleIdentityError(RuntimeError):
    """Raised when an accepted source, fixture, or manifest binding changed."""


def _require_bootstrap_environment() -> None:
    """Reject process-control drift before NumPy can initialize."""

    wrong = {
        name: os.environ.get(name)
        for name, expected in ORACLE_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if wrong:
        raise OracleEnvironmentError(
            "pre-NumPy atmosphere-oracle controls are missing or changed: "
            + json.dumps(wrong, sort_keys=True)
        )


if __name__ == "__main__":
    _require_bootstrap_environment()

import numpy as np  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_ROOT = Path("/Users/ysting/payne-zero")
PINNED_DATA_ROOT = PINNED_ROOT / "source_data_files"
WORKER_PATH = Path(__file__).resolve()

PLAN_PATH = REPOSITORY_ROOT / "design/chapter06_atmosphere_fixture_oracle_plan.md"
SOURCE_CONTRACT_PATH = REPOSITORY_ROOT / "design/chapter06_exact_source_contract.md"
CAUSAL_OUTLINE_PATH = REPOSITORY_ROOT / "design/chapter06_causal_outline.md"
RUNTIME_AUDIT_PATH = REPOSITORY_ROOT / "design/chapter06_runtime_core_audit.md"
FIXTURE_WORKER_PATH = REPOSITORY_ROOT / "scripts/chapter06_atmosphere_fixture_worker.py"
FIXTURE_WORKER_AUDIT_PATH = (
    REPOSITORY_ROOT / "design/chapter06_atmosphere_fixture_worker_independent_audit.md"
)
CONVERTER_PATH = REPOSITORY_ROOT / "scripts/chapter06_atmosphere_line_converter.py"
CONVERTER_AUDIT_PATH = (
    REPOSITORY_ROOT / "design/chapter06_atmosphere_converter_independent_audit.md"
)
FIXTURE_ACCEPTANCE_PATH = (
    REPOSITORY_ROOT / "design/chapter06_atmosphere_fixture_publication_acceptance.json"
)
FIXTURE_RECORD_REVIEW_PATH = (
    REPOSITORY_ROOT
    / "design/chapter06_atmosphere_fixture_publication_record_review.json"
)
MANIFEST_PATH = REPOSITORY_ROOT / "data/MANIFEST.json"
FIXTURE_PATH = (
    REPOSITORY_ROOT / "data/fixtures/chapter06_atmosphere_one_line_inputs.npz"
)
SUBSET_PATH = REPOSITORY_ROOT / "data/subsets/chapter06_fe_i_source_row_873702.npz"
GOLDEN_ROOT = REPOSITORY_ROOT / "data/golden"
LINE_TABLE_PATH = PINNED_DATA_ROOT / "atmosphere_tables/line_opacity_tables.npz"
MOLECULAR_EQUILIBRIUM_TABLE_PATH = (
    PINNED_DATA_ROOT / "atmosphere_tables/molecular_equilibrium_tables.npz"
)

PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
PLAN_SHA256 = "cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97"
SOURCE_CONTRACT_SHA256 = (
    "ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1"
)
CAUSAL_OUTLINE_SHA256 = (
    "1b66df5d548f2854f83289fcf9de5109058f1482a7b64aadaff3505d1f57e019"
)
RUNTIME_AUDIT_SHA256 = (
    "83366bed79293bb8d02ccdf67b54d0350dcc638888e6678d9dedc7b1d58f3313"
)
FIXTURE_WORKER_SHA256 = (
    "db107c9f67c5f074e0aa77b3f523c781b8f63dce2819ee5a1248ff9e6fb1ec84"
)
FIXTURE_WORKER_AUDIT_SHA256 = (
    "336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e"
)
CONVERTER_SHA256 = "4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb"
CONVERTER_AUDIT_SHA256 = (
    "60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75"
)
FIXTURE_ACCEPTANCE_SHA256 = (
    "a756fe395bbe9d598fcc4748e7b604920e615a4de9a2c2dabca5942e8a50b9eb"
)
FIXTURE_RECORD_REVIEW_SHA256 = (
    "07518979af4d60d8cbd2321ea6c976a52ba04de53c2fc528af51888a6c42f37b"
)

FIXTURE_SHA256 = "1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff"
FIXTURE_BYTES = 363_050
FIXTURE_MEMBER_COUNT = 19
FIXTURE_SCHEMA_DIGEST = (
    "f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698"
)
FIXTURE_CAPTURE_PAYLOAD_FINGERPRINT = (
    "f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663"
)
SUBSET_SHA256 = "bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04"
SUBSET_BYTES = 8_665
SUBSET_ROW_INDEX = 873_702
SUBSET_SCHEMA_VERSION = 1
SUBSET_BUILDER_SHA256 = (
    "25bcf4662740155e8b08615b9522f3f4517e1a5ddc4627c68686620ccfff4d6c"
)
CONVERSION_VERSION = 1

LINE_TABLE_SHA256 = "89f486122cb8939b23dc5423145a46d88a77df8daf57a1def35055b7b8205f16"
LINE_TABLE_BYTES = 1_359
LINE_TABLE_SCHEMA = {
    "hydrogen_profile_table": ((81,), np.dtype(np.float64)),
    "voigt_interpolation_table": ((81,), np.dtype(np.float64)),
}
MOLECULAR_EQUILIBRIUM_TABLE_SHA256 = (
    "1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe"
)
MOLECULAR_EQUILIBRIUM_TABLE_BYTES = 1_935
MOLECULAR_EQUILIBRIUM_TABLE_SCHEMA = {
    "atomic_mass_amu": ((99,), np.dtype(np.float64)),
    "h2_partition_function": ((200,), np.dtype(np.float64)),
}

CONTRADICTORY_MANIFEST_LABELS = {
    "arrays.actual_population_slot_values.unit": "cm^-3 per partition function",
    "arrays.packed_species_slot.unit": "zero-based packed species slot",
    "arrays.wavelength_bin_edges.unit": (
        "zero-based opacity-wavelength boundary index"
    ),
}

EXPECTED_DENSE_PRE_SHA256 = (
    "43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58"
)
EXPECTED_DENSE_POST_SHA256 = (
    "a695164eea303f76f249b1a81ce179426d16e9047cefa079b09a9ff3cfae35ab"
)
EXPECTED_SPARSE_INDEX_SHA256 = (
    "9ff0b9443c843eb582ef7d69a1d46ffab6690d938cd213215e6affda680a40f4"
)
EXPECTED_SPARSE_PRE_SHA256 = (
    "8ad01776bee089b24f3190f45e1bcaee106e7438b7c804c1c263bb78c8fac040"
)
EXPECTED_SPARSE_POST_SHA256 = (
    "669e5c9928edb9ed7f05b1711b2661e1d4fdfd5b91b06508c94b78745fe996eb"
)
EXPECTED_RECONSTRUCTED_WAVELENGTH_NM = 499.03410878793585
EXPECTED_FIRST_RED_INDEX = 7_383
EXPECTED_CONTINUUM_COLUMN = 169
EXPECTED_NONZERO_COUNT = 240
EXPECTED_PEAK_PRE = np.float32(0.3731258809566498)

CAPTURE_SCHEMA_VERSION = 1
POPULATION_SLOT_COUNT = 1_006
DEPTH_COUNT = 80
LINE_SLOT_ZERO_BASED = 350
GATE_DEPTHS_1BASED = np.arange(8, 81, 8, dtype=np.int16)
RATIO_LOG_STEP = np.log(1.0 + 1.0 / 2_000_000.0)

FIXTURE_MEMBER_CONTRACT = {
    "actual_population_slot_indices": (
        (3,),
        np.dtype(np.int16),
        "eff6cc5c731be5128ac078be458469978b6ac7c823665abd47098485291a5af2",
    ),
    "actual_population_slot_values": (
        (80, 3),
        np.dtype(np.float64),
        "ae17c6b96dc9eb824051707d83eca2bbdb22483af4c143b18d832326292e86ce",
    ),
    "continuum_line_selection_threshold": (
        (80, 344),
        np.dtype(np.float32),
        "76cb7ef18554149b61b97e32f2a69bbcdba3eb8b7e6b7da8ba3c69b8775b7293",
    ),
    "effective_temperature": (
        (),
        np.dtype(np.float64),
        "0ae804feb5a3896c0d30fd25ee9853a10ee08fed330969967ce16a4d07a7329b",
    ),
    "electron_density": (
        (80,),
        np.dtype(np.float64),
        "1e36bdf5aac263da2eac448f1b4c91e43e48a48afc133c8a8ec5e15735c85c18",
    ),
    "fractional_doppler_widths_at_line_slot": (
        (80,),
        np.dtype(np.float64),
        "f8018929e269db3e6b9a572d36b528bc0577a3a93bcc25166b277824ffb2e785",
    ),
    "hc_over_kt": (
        (80,),
        np.dtype(np.float64),
        "fdbd300b88163af886fe2905a883fa319c32abcc89db1099e4b0df7965d8174e",
    ),
    "line_population_slot_zero_based": (
        (),
        np.dtype(np.int16),
        "8d6dd9f330421e48e73e97c0819d55a9965bac90b1f2f36057b4c906d5791f05",
    ),
    "log_strength_index": (
        (1,),
        np.dtype(np.int16),
        "3fe821d54660a0c51b42d19a42571e208791b3c5ff6bc0cac16b6553a226515a",
    ),
    "lower_excitation_index": (
        (1,),
        np.dtype(np.int16),
        "9b5bf0b74e2e212b57bcb2b9f2712eaab4c6169595f4e82767793f4534365648",
    ),
    "opacity_wavelength_grid_nm": (
        (30_000,),
        np.dtype(np.float64),
        "8944f1dd701ba27f50d37a16e48ab9375e1bef1b444ed405b671ba91fde8132b",
    ),
    "packed_species_slot": (
        (1,),
        np.dtype(np.int16),
        "c2126161f7488b7d198ea310da9a8694786a18a2317eea5e30379ee118d34743",
    ),
    "packed_wavelength_index": (
        (1,),
        np.dtype(np.int32),
        "c423f4ac4a3825c6fad5336a1e15c0038ab17087f930916ad550ad45b4990dfc",
    ),
    (
        "partition_normalized_population_over_mass_density_and_"
        "fractional_doppler_width_at_line_slot"
    ): (
        (80,),
        np.dtype(np.float64),
        "8625e8d942892d142384d51fc0625701374f79dfa471a78ff593d88479b3056f",
    ),
    "radiative_damping_index": (
        (1,),
        np.dtype(np.int16),
        "5cdf3b75730a5b45c3f24da2c1030103143981191d2844aa7374e948b9abaeea",
    ),
    "stark_damping_index": (
        (1,),
        np.dtype(np.int16),
        "91d82e7b89ae43b29bb673bb416697d005e093d0011c9d7af501630c6a502141",
    ),
    "temperature": (
        (80,),
        np.dtype(np.float64),
        "7b76410e731a6772cb6de12f923aea8a26e345778dbab67db3e8938278ce270f",
    ),
    "van_der_waals_damping_index": (
        (1,),
        np.dtype(np.int16),
        "c0346f09a0362e8e9d29c01a9fc7c292a7395b524a90319bb921608b9fdb1b60",
    ),
    "wavelength_bin_edges": (
        (344,),
        np.dtype(np.int64),
        "ae50e9c0bafdcbfc39e242fd5029bf53b9e712d43008c6cb78a4493b89272cd5",
    ),
}

# Section 3.1 of the plan contains one synthesis source used to interpret the
# raw row and fifteen atmosphere sources.  The atmosphere package's ordinary
# import path loads a larger, accepted 35-file closure; that closure is parsed
# from the exact accepted fixture worker before any pinned module is imported.
CRITICAL_SOURCE_HASHES = {
    "payne_zero_synthesis/atomic_lines.py": (
        "0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0"
    ),
    "payne_zero_atmosphere/constants.py": (
        "ac1f1fbd345dc816eb3e70a8f97ebebc7a4c744fd2759b32ec19f8c88d987036"
    ),
    "payne_zero_atmosphere/atmosphere_io.py": (
        "95c4d2cab230f6925e9404639ecb05b25af8c0c85755ac1ca70d760156a8683e"
    ),
    "payne_zero_atmosphere/runtime_state.py": (
        "fae240ec00f6f89d7c2a7ef721ce6e6539be234e523291fd6e8a096d731430e8"
    ),
    "payne_zero_atmosphere/population_layout.py": (
        "36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0"
    ),
    "payne_zero_atmosphere/equation_of_state.py": (
        "719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2"
    ),
    "payne_zero_atmosphere/molecular_data.py": (
        "705c3072d79c8019c948ce0fa2c82052f232816d453e10a7c8e5fc5a8f5ce249"
    ),
    "payne_zero_atmosphere/molecular_equilibrium.py": (
        "4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86"
    ),
    "payne_zero_atmosphere/doppler.py": (
        "e118a78bf5250ef5e1f77d652c9e78fbb7b92acf5c069f717faed7a3b3ea98f0"
    ),
    "payne_zero_atmosphere/continuum_opacity.py": (
        "1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81"
    ),
    "payne_zero_atmosphere/runner.py": (
        "05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7"
    ),
    "payne_zero_atmosphere/line_profile_math.py": (
        "9a5794140f00ff3c3fb6c2e3b28461bbc22b471f962d275055c066ad7f8acd15"
    ),
    "payne_zero_atmosphere/line_catalog.py": (
        "2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92"
    ),
    "payne_zero_atmosphere/line_selection.py": (
        "b2c62fdf5e1fe43f33022184bfeff88985b13331354e3c745c7dab3a6b634fef"
    ),
    "payne_zero_atmosphere/line_opacity.py": (
        "d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6"
    ),
    "payne_zero_atmosphere/transfer_kernels.py": (
        "50e759a085e6aefdb7819a3dbe3ef5e83405834f4b07e0a4de2f3c0e7354d3b9"
    ),
}

FINGERPRINT_FIELDS = frozenset(
    {
        "meta__capture_schema_digest",
        "meta__oracle_payload_fingerprint",
        "meta__full_capture_fingerprint",
    }
)


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Hash one array's contiguous C-order bytes."""

    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(values)).tobytes(order="C")
    ).hexdigest()


def _is_under(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    parent = root.resolve()
    return resolved == parent or parent in resolved.parents


def _display_input_path(path: Path) -> str:
    """Return one deterministic authority-qualified path label."""

    resolved = Path(path).resolve()
    if _is_under(resolved, REPOSITORY_ROOT):
        return (
            "repository:" + resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
        )
    if _is_under(resolved, PINNED_ROOT):
        return "pin:" + resolved.relative_to(PINNED_ROOT.resolve()).as_posix()
    raise OracleIdentityError(f"dynamic input lies outside accepted roots: {resolved}")


def _require_regular_nonsymlink(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise OracleIdentityError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise OracleIdentityError(f"{label} is not a regular file: {resolved}")
    return resolved


def _canonical_input(path: Path, expected: Path, label: str) -> Path:
    """Accept only one canonical non-golden input pathname."""

    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    resolved = candidate.resolve()
    if _is_under(absolute, GOLDEN_ROOT) or _is_under(resolved, GOLDEN_ROOT):
        raise OracleIdentityError(f"{label} may not read a golden artifact")
    if candidate.is_symlink():
        raise OracleIdentityError(f"{label} path must not be a symlink: {candidate}")
    canonical = expected.resolve()
    if absolute != canonical:
        raise OracleIdentityError(
            f"{label} accepts only the canonical path {canonical}, not {absolute}"
        )
    return _require_regular_nonsymlink(expected, label)


def _literal_assignment(path: Path, name: str) -> Any:
    """Read one literal module assignment without importing the module."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        value = node.value
        if value is None:
            continue
        matches.append(ast.literal_eval(value))
    if len(matches) != 1:
        raise OracleIdentityError(
            f"{path.name} must define exactly one literal {name}, got {len(matches)}"
        )
    return matches[0]


def _duplicate_free_json(path: Path) -> Any:
    """Parse JSON while rejecting duplicate object keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OracleIdentityError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                OracleIdentityError(f"nonfinite JSON token {token!r} in {path}")
            ),
        )
    except UnicodeDecodeError as error:
        raise OracleIdentityError(f"{path} is not valid UTF-8") from error


def schema_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash sorted names, dtype strings, and int64 shapes."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise OracleIdentityError(f"object dtype is forbidden: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    return digest.hexdigest()


def mapping_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash sorted names, schemas, and contiguous bytes."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise OracleIdentityError(f"object dtype is forbidden: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _validate_fixture_mapping(arrays: Mapping[str, np.ndarray]) -> None:
    actual_names = set(arrays)
    expected_names = set(FIXTURE_MEMBER_CONTRACT)
    if actual_names != expected_names:
        raise OracleIdentityError(
            "fixture member set changed; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    for name, (
        expected_shape,
        expected_dtype,
        expected_hash,
    ) in FIXTURE_MEMBER_CONTRACT.items():
        values = np.asarray(arrays[name])
        if values.shape != expected_shape or values.dtype != expected_dtype:
            raise OracleIdentityError(
                f"fixture member schema changed: {name} "
                f"shape={values.shape}, dtype={values.dtype}"
            )
        if values.dtype.hasobject or not values.flags.c_contiguous:
            raise OracleIdentityError(
                f"fixture member is not object-free C-contiguous: {name}"
            )
        actual_hash = array_sha256(values)
        if actual_hash != expected_hash:
            raise OracleIdentityError(
                f"fixture member bytes changed: {name} {actual_hash}"
            )
    if schema_digest(arrays) != FIXTURE_SCHEMA_DIGEST:
        raise OracleIdentityError("fixture scientific schema digest changed")

    if not np.array_equal(
        arrays["actual_population_slot_indices"],
        np.asarray([0, 2, 840], dtype=np.int16),
    ):
        raise OracleIdentityError("fixture actual-population projection changed")
    if int(arrays["line_population_slot_zero_based"]) != LINE_SLOT_ZERO_BASED:
        raise OracleIdentityError("fixture line-population slot changed")
    if float(arrays["effective_temperature"]) != 5778.0:
        raise OracleIdentityError("fixture effective-temperature configuration changed")
    if int(arrays["wavelength_bin_edges"][-1]) != 2**30:
        raise OracleIdentityError("fixture wavelength-bin sentinel changed")
    grid = arrays["opacity_wavelength_grid_nm"]
    if np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise OracleIdentityError("fixture wavelength grid is not finite/increasing")
    for name, values in arrays.items():
        if np.issubdtype(values.dtype, np.floating) and np.any(~np.isfinite(values)):
            raise OracleIdentityError(f"fixture contains nonfinite values: {name}")


def _manifest_entry(manifest: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise OracleIdentityError("manifest entries must be a list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("path") == target
    ]
    if len(matches) != 1:
        raise OracleIdentityError(
            f"manifest must contain exactly one {target!r} entry, got {len(matches)}"
        )
    return matches[0]


def verify_manifest_bindings(
    *, treat_array_labels_as_authority: bool = False
) -> dict[str, str]:
    """Verify the fixture and raw-subset entries without pinning append order."""

    manifest_path = _require_regular_nonsymlink(MANIFEST_PATH, "data manifest")
    manifest = _duplicate_free_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise OracleIdentityError("data manifest schema version changed")
    if manifest.get("payne_zero_commit") != PINNED_COMMIT:
        raise OracleIdentityError("data manifest commit changed")

    fixture_relative = FIXTURE_PATH.relative_to(REPOSITORY_ROOT).as_posix()
    fixture_entry = _manifest_entry(manifest, fixture_relative)
    expected_fixture_scalars = {
        "role": "fixture",
        "source_commit": PINNED_COMMIT,
        "archive_kind": "atmosphere_one_line_input_fixture",
        "fixture_capture_schema_version": 1,
        "archive_contains_embedded_schema_version": False,
        "member_count": FIXTURE_MEMBER_COUNT,
        "scientific_fixture_schema_digest": FIXTURE_SCHEMA_DIGEST,
        "scientific_payload_fingerprint": FIXTURE_CAPTURE_PAYLOAD_FINGERPRINT,
        "format": "npz",
        "sha256": FIXTURE_SHA256,
        "bytes": FIXTURE_BYTES,
        "publication_acceptance_sha256": FIXTURE_ACCEPTANCE_SHA256,
        "publication_record_review_sha256": FIXTURE_RECORD_REVIEW_SHA256,
    }
    for name, expected in expected_fixture_scalars.items():
        if fixture_entry.get(name) != expected:
            raise OracleIdentityError(f"fixture manifest binding changed: {name}")
    entry_arrays = fixture_entry.get("arrays")
    if not isinstance(entry_arrays, dict) or set(entry_arrays) != set(
        FIXTURE_MEMBER_CONTRACT
    ):
        raise OracleIdentityError("fixture manifest array member set changed")
    for name, (shape, dtype, member_hash) in FIXTURE_MEMBER_CONTRACT.items():
        record = entry_arrays[name]
        if (
            record.get("shape") != list(shape)
            or record.get("dtype") != dtype.name
            or record.get("sha256") != member_hash
        ):
            raise OracleIdentityError(
                f"fixture manifest member binding changed: {name}"
            )

    observed_contradictions = {
        "arrays.actual_population_slot_values.unit": entry_arrays[
            "actual_population_slot_values"
        ].get("unit"),
        "arrays.packed_species_slot.unit": entry_arrays["packed_species_slot"].get(
            "unit"
        ),
        "arrays.wavelength_bin_edges.unit": entry_arrays["wavelength_bin_edges"].get(
            "unit"
        ),
    }
    if observed_contradictions != CONTRADICTORY_MANIFEST_LABELS:
        raise OracleIdentityError(
            "published contradictory manifest labels changed; a separate "
            "manifest-correction lifecycle is required"
        )
    if treat_array_labels_as_authority:
        raise OracleIdentityError(
            "the three published fixture labels contradict the pinned source "
            "semantics and may not be treated as scientific authority"
        )

    subset_relative = SUBSET_PATH.relative_to(REPOSITORY_ROOT).as_posix()
    subset_entry = _manifest_entry(manifest, subset_relative)
    expected_subset_scalars = {
        "role": "subset",
        "source_commit": PINNED_COMMIT,
        "source_row_index": SUBSET_ROW_INDEX,
        "source_field_count": 17,
        "subset_schema_version": SUBSET_SCHEMA_VERSION,
        "builder_sha256": SUBSET_BUILDER_SHA256,
        "sha256": SUBSET_SHA256,
        "bytes": SUBSET_BYTES,
        "format": "npz",
    }
    for name, expected in expected_subset_scalars.items():
        if subset_entry.get(name) != expected:
            raise OracleIdentityError(f"raw-subset manifest binding changed: {name}")
    return {
        "fixture_entry_digest": hashlib.sha256(
            json.dumps(fixture_entry, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "subset_entry_digest": hashlib.sha256(
            json.dumps(subset_entry, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, np.ndarray]:
    """Load and exhaustively validate the canonical atmosphere input fixture."""

    resolved = _canonical_input(path, FIXTURE_PATH, "Chapter 6 atmosphere fixture")
    if resolved.stat().st_size != FIXTURE_BYTES:
        raise OracleIdentityError("atmosphere fixture byte size changed")
    actual_hash = sha256(resolved)
    if actual_hash != FIXTURE_SHA256:
        raise OracleIdentityError(
            f"atmosphere fixture SHA-256 is {actual_hash}; expected {FIXTURE_SHA256}"
        )
    with np.load(resolved, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise OracleIdentityError("fixture contains duplicate member names")
        arrays = {
            name: np.array(archive[name], copy=True, order="C")
            for name in archive.files
        }
    _validate_fixture_mapping(arrays)
    return arrays


def _accepted_python_manifest() -> dict[str, str]:
    """Read the exact accepted package-import closure as inert source data."""

    manifest = _literal_assignment(FIXTURE_WORKER_PATH, "FROZEN_PINNED_PYTHON_MANIFEST")
    if not isinstance(manifest, dict) or len(manifest) != 35:
        raise OracleIdentityError("accepted pinned Python manifest changed")
    normalized = {str(name): str(value) for name, value in manifest.items()}
    if not all(
        name.startswith("payne_zero_atmosphere/") and len(value) == 64
        for name, value in normalized.items()
    ):
        raise OracleIdentityError("accepted pinned Python manifest is malformed")
    for name, expected_hash in CRITICAL_SOURCE_HASHES.items():
        if name.startswith("payne_zero_atmosphere/"):
            if normalized.get(name) != expected_hash:
                raise OracleIdentityError(
                    f"critical source is absent from accepted import closure: {name}"
                )
    return normalized


def verify_preimport_identities() -> dict[str, str]:
    """Verify every authority and source before importing Payne Zero."""

    fixed_files = {
        PLAN_PATH: PLAN_SHA256,
        SOURCE_CONTRACT_PATH: SOURCE_CONTRACT_SHA256,
        CAUSAL_OUTLINE_PATH: CAUSAL_OUTLINE_SHA256,
        RUNTIME_AUDIT_PATH: RUNTIME_AUDIT_SHA256,
        FIXTURE_WORKER_PATH: FIXTURE_WORKER_SHA256,
        FIXTURE_WORKER_AUDIT_PATH: FIXTURE_WORKER_AUDIT_SHA256,
        CONVERTER_PATH: CONVERTER_SHA256,
        CONVERTER_AUDIT_PATH: CONVERTER_AUDIT_SHA256,
        FIXTURE_ACCEPTANCE_PATH: FIXTURE_ACCEPTANCE_SHA256,
        FIXTURE_RECORD_REVIEW_PATH: FIXTURE_RECORD_REVIEW_SHA256,
    }
    identities: dict[str, str] = {}
    for path, expected_hash in fixed_files.items():
        resolved = _require_regular_nonsymlink(path, f"accepted input {path.name}")
        actual_hash = sha256(resolved)
        if actual_hash != expected_hash:
            raise OracleIdentityError(
                f"accepted input changed: {path} has {actual_hash}"
            )
        identities[f"accepted__{path.name}__sha256"] = actual_hash

    converter_version = _literal_assignment(CONVERTER_PATH, "CONVERSION_VERSION")
    if converter_version != CONVERSION_VERSION:
        raise OracleIdentityError("atmosphere conversion version changed")

    root = PINNED_ROOT.resolve()
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
    identities["payne_zero_commit"] = commit

    accepted_manifest = _accepted_python_manifest()
    for relative_name, expected_hash in accepted_manifest.items():
        source = _require_regular_nonsymlink(
            PINNED_ROOT / relative_name, f"pinned source {relative_name}"
        )
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise OracleIdentityError(
                f"pinned source changed for {relative_name}: {actual_hash}"
            )
        identities[f"source__{relative_name}__sha256"] = actual_hash
    for relative_name, expected_hash in CRITICAL_SOURCE_HASHES.items():
        source = _require_regular_nonsymlink(
            PINNED_ROOT / relative_name, f"critical source {relative_name}"
        )
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise OracleIdentityError(
                f"critical source changed for {relative_name}: {actual_hash}"
            )
        identities[f"critical__{relative_name}__sha256"] = actual_hash

    table = _require_regular_nonsymlink(LINE_TABLE_PATH, "line-opacity table")
    if table.stat().st_size != LINE_TABLE_BYTES or sha256(table) != LINE_TABLE_SHA256:
        raise OracleIdentityError("line-opacity table identity changed")
    with np.load(table, allow_pickle=False) as archive:
        if set(archive.files) != set(LINE_TABLE_SCHEMA):
            raise OracleIdentityError("line-opacity table member set changed")
        for name, (shape, dtype) in LINE_TABLE_SCHEMA.items():
            values = np.asarray(archive[name])
            if values.shape != shape or values.dtype != dtype:
                raise OracleIdentityError(
                    f"line-opacity table member schema changed: {name}"
                )
    identities["table__line_opacity_tables__sha256"] = LINE_TABLE_SHA256

    molecular_table = _require_regular_nonsymlink(
        MOLECULAR_EQUILIBRIUM_TABLE_PATH,
        "eager molecular-equilibrium table",
    )
    if (
        molecular_table.stat().st_size != MOLECULAR_EQUILIBRIUM_TABLE_BYTES
        or sha256(molecular_table) != MOLECULAR_EQUILIBRIUM_TABLE_SHA256
    ):
        raise OracleIdentityError("molecular-equilibrium table identity changed")
    with np.load(molecular_table, allow_pickle=False) as archive:
        if set(archive.files) != set(MOLECULAR_EQUILIBRIUM_TABLE_SCHEMA):
            raise OracleIdentityError("molecular-equilibrium table member set changed")
        for name, (shape, dtype) in MOLECULAR_EQUILIBRIUM_TABLE_SCHEMA.items():
            values = np.asarray(archive[name])
            if values.shape != shape or values.dtype != dtype:
                raise OracleIdentityError(
                    f"molecular-equilibrium table member schema changed: {name}"
                )
    identities["table__molecular_equilibrium_tables__sha256"] = (
        MOLECULAR_EQUILIBRIUM_TABLE_SHA256
    )

    subset = _require_regular_nonsymlink(SUBSET_PATH, "raw teaching subset")
    if subset.stat().st_size != SUBSET_BYTES or sha256(subset) != SUBSET_SHA256:
        raise OracleIdentityError("raw teaching subset identity changed")
    identities["raw_subset_sha256"] = SUBSET_SHA256
    identities.update(verify_manifest_bindings())
    return identities


def require_environment() -> dict[str, np.ndarray]:
    """Require one-thread controls and a fresh external Numba cache."""

    wrong = {
        name: os.environ.get(name)
        for name, expected in ORACLE_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if wrong:
        raise OracleEnvironmentError(
            "fresh one-thread atmosphere-oracle controls are missing: "
            + json.dumps(wrong, sort_keys=True)
        )
    cache_text = os.environ.get("NUMBA_CACHE_DIR", "")
    if not cache_text:
        raise OracleEnvironmentError("NUMBA_CACHE_DIR must name a fresh directory")
    cache_candidate = Path(cache_text).expanduser()
    if cache_candidate.exists() or cache_candidate.is_symlink():
        cache_candidate.lstat()
        if cache_candidate.is_symlink():
            raise OracleEnvironmentError("NUMBA_CACHE_DIR must not be a symlink")
    else:
        cache_candidate.mkdir(parents=True, exist_ok=False)
    cache_path = cache_candidate.resolve()
    for forbidden_root in (
        PINNED_ROOT,
        REPOSITORY_ROOT,
        GOLDEN_ROOT,
    ):
        if _is_under(cache_path, forbidden_root):
            raise OracleEnvironmentError(
                "NUMBA_CACHE_DIR must be external to source, repository, and golden trees"
            )
    if any(cache_path.iterdir()):
        raise OracleEnvironmentError(
            "NUMBA_CACHE_DIR must be truly empty at process start"
        )
    return {
        f"environment__{name.lower()}": np.asarray(value)
        for name, value in sorted(ORACLE_ENVIRONMENT.items())
    } | {
        "environment__cache_policy": np.asarray(
            "verified empty external nonsymlink directory", dtype="<U48"
        ),
        "environment__cpu_only": np.asarray(True, dtype=np.bool_),
        "environment__work_backend": np.asarray("NumPy/Numba CPU", dtype="<U32"),
    }


def _loaded_pinned_python_manifest() -> dict[str, str]:
    loaded: dict[str, str] = {}
    root = PINNED_ROOT.resolve()
    for module in sys.modules.values():
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        path = Path(module_file).resolve()
        if path.suffix != ".py" or not _is_under(path, root):
            continue
        loaded[str(path.relative_to(root))] = sha256(path)
    return loaded


def _assert_loaded_manifest(expected: Mapping[str, str]) -> dict[str, str]:
    loaded = _loaded_pinned_python_manifest()
    if loaded != dict(expected):
        expected_names = set(expected)
        actual_names = set(loaded)
        changed = sorted(
            name
            for name in expected_names & actual_names
            if loaded[name] != expected[name]
        )
        raise OracleIdentityError(
            "loaded pinned Python manifest changed; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}, changed={changed}"
        )
    return loaded


def load_pinned_modules() -> SimpleNamespace:
    """Import the standard pinned package route and verify its full closure."""

    configured_data_root = os.environ.get("PAYNE_ZERO_DATA_ROOT")
    if configured_data_root is not None:
        raise OracleIdentityError(
            "PAYNE_ZERO_DATA_ROOT must be absent before the oracle binds the pin"
        )
    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(PINNED_DATA_ROOT.resolve())

    for name, module in tuple(sys.modules.items()):
        if name != "payne_zero_atmosphere" and not name.startswith(
            "payne_zero_atmosphere."
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None and not _is_under(Path(module_file), PINNED_ROOT):
            raise OracleIdentityError(
                f"{name} was already imported outside the pin: {module_file}"
            )

    root_text = str(PINNED_ROOT.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    line_catalog = importlib.import_module("payne_zero_atmosphere.line_catalog")
    line_opacity = importlib.import_module("payne_zero_atmosphere.line_opacity")
    line_profile_math = importlib.import_module(
        "payne_zero_atmosphere.line_profile_math"
    )
    runner = importlib.import_module("payne_zero_atmosphere.runner")
    import numba

    for module in (line_catalog, line_opacity, line_profile_math, runner):
        if not _is_under(Path(module.__file__), PINNED_ROOT):
            raise OracleIdentityError(
                f"atmosphere module resolved outside the pin: {module.__file__}"
            )
    loaded_manifest = _assert_loaded_manifest(_accepted_python_manifest())
    if int(numba.get_num_threads()) != 1:
        raise OracleEnvironmentError("Numba did not honor NUMBA_NUM_THREADS=1")
    if line_profile_math._TABLE_CACHE is not None:
        raise OracleEnvironmentError("line-opacity table cache was populated early")
    if line_profile_math.build_voigt_profile_basis.cache_info().currsize != 0:
        raise OracleEnvironmentError("Voigt basis cache was populated early")
    return SimpleNamespace(
        line_catalog=line_catalog,
        line_opacity=line_opacity,
        line_profile_math=line_profile_math,
        runner=runner,
        numba=numba,
        loaded_python_manifest=loaded_manifest,
    )


def deterministic_result(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return a sorted detached mapping and reject object dtypes."""

    result: dict[str, np.ndarray] = {}
    for name in sorted(arrays):
        values = np.asarray(arrays[name])
        if values.dtype.hasobject:
            raise TypeError(f"oracle results forbid object dtype: {name}")
        result[name] = np.array(values, copy=True, order="C")
    return result


def _selected_catalog(
    modules: SimpleNamespace, fixture: Mapping[str, np.ndarray]
) -> Any:
    selected = modules.line_catalog.SelectedLineCatalog(
        packed_wavelength_index=fixture["packed_wavelength_index"].copy(),
        packed_species_slot=fixture["packed_species_slot"].copy(),
        lower_excitation_index=fixture["lower_excitation_index"].copy(),
        log_strength_index=fixture["log_strength_index"].copy(),
        radiative_damping_index=fixture["radiative_damping_index"].copy(),
        stark_damping_index=fixture["stark_damping_index"].copy(),
        van_der_waals_damping_index=fixture["van_der_waals_damping_index"].copy(),
    )
    if selected.line_count != 1:
        raise RuntimeError("canonical selected catalog must contain exactly one line")
    return selected


def reconstruct_public_call_arrays(
    fixture: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct only the three dense arrays read by the selected record."""

    actual = np.zeros((DEPTH_COUNT, POPULATION_SLOT_COUNT), dtype=np.float64)
    actual[:, fixture["actual_population_slot_indices"]] = fixture[
        "actual_population_slot_values"
    ]
    line_support = np.zeros((DEPTH_COUNT, POPULATION_SLOT_COUNT), dtype=np.float64)
    line_support[:, LINE_SLOT_ZERO_BASED] = fixture[
        "partition_normalized_population_over_mass_density_and_"
        "fractional_doppler_width_at_line_slot"
    ]
    fractional_widths = np.zeros((DEPTH_COUNT, POPULATION_SLOT_COUNT), dtype=np.float64)
    fractional_widths[:, LINE_SLOT_ZERO_BASED] = fixture[
        "fractional_doppler_widths_at_line_slot"
    ]
    return actual, line_support, fractional_widths


def _public_call(
    modules: SimpleNamespace,
    selected: Any,
    *,
    grid: np.ndarray,
    bin_edges: np.ndarray,
    threshold: np.ndarray,
    temperature: np.ndarray,
    hc_over_kt: np.ndarray,
    electron_density: np.ndarray,
    actual: np.ndarray,
    line_support: np.ndarray,
    fractional_widths: np.ndarray,
) -> Any:
    return modules.line_opacity.accumulate_selected_line_opacity(
        selected_lines=selected,
        opacity_wavelength_grid_nm=grid,
        wavelength_bin_edges=bin_edges,
        continuum_line_selection_threshold=threshold,
        temperature=temperature,
        hc_over_kt=hc_over_kt,
        electron_density=electron_density,
        ion_stage_populations_by_packed_slot=actual,
        partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
            line_support
        ),
        fractional_doppler_widths=fractional_widths,
    )


@contextmanager
def _record_npz_reads() -> Iterator[set[str]]:
    """Record direct ``numpy.load`` paths during the scientific call."""

    reads: set[str] = set()
    original = np.load

    def recording_load(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            reads.add(str(Path(file).expanduser().resolve()))
        return original(file, *args, **kwargs)

    np.load = recording_load  # type: ignore[assignment]
    try:
        yield reads
    finally:
        np.load = original  # type: ignore[assignment]


def _route_source_evidence() -> dict[str, np.ndarray]:
    """Verify the exact source-level serial call topology and ordering."""

    path = PINNED_ROOT / "payne_zero_atmosphere/line_opacity.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "accumulate_selected_line_opacity",
        "_accumulate_selected_line_opacity_compiled",
        "_accumulate_selected_line_wings_compiled",
        "_voigt_profile_compiled",
        "_fast_exponential_lookup_compiled",
        "_accumulate_selected_line_opacity_parallel",
    }
    if not required.issubset(functions):
        raise RuntimeError("serial route source functions changed")

    def called_names(function_name: str) -> list[str]:
        calls: list[tuple[int, int, str]] = []
        for node in ast.walk(functions[function_name]):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                calls.append(
                    (
                        getattr(node, "lineno", -1),
                        getattr(node, "col_offset", -1),
                        node.func.id,
                    )
                )
        return [name for _, _, name in sorted(calls)]

    public_calls = called_names("accumulate_selected_line_opacity")
    compiled_calls = called_names("_accumulate_selected_line_opacity_compiled")
    wing_calls = called_names("_accumulate_selected_line_wings_compiled")
    required_public = {
        "_accumulate_selected_line_opacity_parallel",
        "_accumulate_selected_line_opacity_compiled",
    }
    if not required_public.issubset(public_calls):
        raise RuntimeError("public selected-line routing source changed")
    if "_accumulate_selected_line_wings_compiled" not in compiled_calls:
        raise RuntimeError("compiled selected-line route no longer calls wing helper")
    if "_fast_exponential_lookup_compiled" not in compiled_calls:
        raise RuntimeError("compiled selected-line route no longer calls FASTEX")
    if "_voigt_profile_compiled" not in wing_calls:
        raise RuntimeError("compiled wing route no longer calls Voigt helper")
    first_fast = compiled_calls.index("_fast_exponential_lookup_compiled")
    first_wing = compiled_calls.index("_accumulate_selected_line_wings_compiled")
    if first_fast >= first_wing:
        raise RuntimeError("FASTEX no longer precedes the first wing deposit")
    return {
        "route__public_serial_dispatch_present": np.asarray(True, dtype=np.bool_),
        "route__public_parallel_dispatch_guard_present": np.asarray(
            True, dtype=np.bool_
        ),
        "route__compiled_calls_fast_exponential": np.asarray(True, dtype=np.bool_),
        "route__compiled_calls_wing_helper": np.asarray(True, dtype=np.bool_),
        "route__wing_calls_voigt_helper": np.asarray(True, dtype=np.bool_),
        "route__fast_exponential_precedes_first_wing": np.asarray(True, dtype=np.bool_),
    }


def _line_parameter_ledger(modules: SimpleNamespace, selected: Any) -> dict[str, float]:
    lookup = np.ascontiguousarray(
        modules.line_profile_math.build_selection_log_lookup(), dtype=np.float32
    )
    packed = int(selected.packed_wavelength_index[0])
    wavelength = float(np.exp(np.float64(packed) * RATIO_LOG_STEP))
    wavelength_f32 = np.float32(wavelength)
    log_strength_index = int(selected.log_strength_index[0])
    lower_index = int(selected.lower_excitation_index[0])
    radiative_index = int(selected.radiative_damping_index[0])
    stark_index = int(selected.stark_damping_index[0])
    vdw_index = int(selected.van_der_waals_damping_index[0])
    classical = np.float32(
        np.float32(modules.line_opacity._CLASSICAL_LINE_STRENGTH_SCALE)
        * wavelength_f32
        * lookup[log_strength_index - 1]
    )
    damping_scale = np.float32(modules.line_opacity._DAMPING_SCALE)
    radiative = np.float32(lookup[radiative_index - 1] * wavelength_f32 * damping_scale)
    stark = np.float32(lookup[stark_index - 1] * wavelength_f32 * damping_scale)
    vdw = np.float32(lookup[vdw_index - 1] * wavelength_f32 * damping_scale)
    return {
        "wavelength_nm": wavelength,
        "classical_strength": float(classical),
        "lower_excitation_cm": float(lookup[lower_index - 1]),
        "radiative_damping": float(radiative),
        "stark_damping": float(stark),
        "van_der_waals_damping": float(vdw),
    }


def _controlled_inputs(
    center_nm: float,
    *,
    step_nm: float = 0.001,
    support_value: float = 1.0e18,
    fractional_width: float = 1.0e-3,
    hc_over_kt_value: float = 0.0,
    threshold_value: float = 0.0,
) -> dict[str, np.ndarray]:
    grid = center_nm + (np.arange(401, dtype=np.float64) - 200.0) * step_nm
    grid[200] = center_nm
    actual = np.zeros((80, 1006), dtype=np.float64)
    support = np.zeros((80, 1006), dtype=np.float64)
    support[:, LINE_SLOT_ZERO_BASED] = support_value
    widths = np.zeros((80, 1006), dtype=np.float64)
    widths[:, LINE_SLOT_ZERO_BASED] = fractional_width
    return {
        "grid": grid,
        "bin_edges": np.asarray([2**30], dtype=np.int64),
        "threshold": np.full((80, 1), threshold_value, dtype=np.float32),
        "temperature": np.full(80, 5000.0, dtype=np.float64),
        "hc_over_kt": np.full(80, hc_over_kt_value, dtype=np.float64),
        "electron_density": np.zeros(80, dtype=np.float64),
        "actual": actual,
        "line_support": support,
        "fractional_widths": widths,
    }


def _run_controlled(
    modules: SimpleNamespace,
    selected: Any,
    values: Mapping[str, np.ndarray],
) -> Any:
    return _public_call(
        modules,
        selected,
        grid=values["grid"],
        bin_edges=values["bin_edges"],
        threshold=values["threshold"],
        temperature=values["temperature"],
        hc_over_kt=values["hc_over_kt"],
        electron_density=values["electron_density"],
        actual=values["actual"],
        line_support=values["line_support"],
        fractional_widths=values["fractional_widths"],
    )


def _depth_resize(
    values: Mapping[str, np.ndarray], count: int
) -> dict[str, np.ndarray]:
    resized: dict[str, np.ndarray] = {}
    depth_names = {
        "threshold",
        "temperature",
        "hc_over_kt",
        "electron_density",
        "actual",
        "line_support",
        "fractional_widths",
    }
    for name, array in values.items():
        if name not in depth_names:
            resized[name] = array.copy()
            continue
        if count <= array.shape[0]:
            resized[name] = array[:count].copy()
        else:
            extra = np.repeat(array[-1:], count - array.shape[0], axis=0)
            resized[name] = np.concatenate([array, extra], axis=0)
    return resized


def _depth_contract_evidence(
    modules: SimpleNamespace, selected: Any, center_nm: float
) -> dict[str, np.ndarray]:
    base = _controlled_inputs(center_nm)
    succeeded = _run_controlled(modules, selected, base)
    if succeeded.line_mass_absorption_coefficient.shape != (80, 401):
        raise RuntimeError("controlled 80-depth public call failed")
    messages = []
    for count in (79, 81):
        try:
            _run_controlled(modules, selected, _depth_resize(base, count))
        except ValueError as error:
            expected = f"Selected-line opacity expects 80 depth layers, got {count}."
            if str(error) != expected:
                raise RuntimeError(
                    f"{count}-depth failure message changed: {error}"
                ) from error
            messages.append(str(error))
        else:
            raise RuntimeError(f"{count}-depth public call unexpectedly succeeded")

    malformed_fixture = {
        name: np.array(values, copy=True) for name, values in load_fixture().items()
    }
    malformed_fixture["temperature"] = malformed_fixture["temperature"][:-1]
    try:
        _validate_fixture_mapping(malformed_fixture)
    except OracleIdentityError:
        fixture_shape_rejected = True
    else:
        fixture_shape_rejected = False
    if not fixture_shape_rejected:
        raise RuntimeError("fixture depth mismatch was not rejected")
    return {
        "seam__depth__accepted_count": np.asarray(80, dtype=np.int16),
        "seam__depth__rejected_counts": np.asarray([79, 81], dtype=np.int16),
        "seam__depth__rejection_messages": np.asarray(messages, dtype="<U64"),
        "seam__depth__fixture_axis_mismatch_rejected": np.asarray(
            fixture_shape_rejected, dtype=np.bool_
        ),
    }


def _wing_cap_evidence(
    modules: SimpleNamespace, selected: Any, center_nm: float
) -> dict[str, np.ndarray]:
    values = _controlled_inputs(
        center_nm,
        support_value=1.0e18,
        fractional_width=1.0e-3,
        threshold_value=0.0,
    )
    state = _run_controlled(modules, selected, values)
    slab = state.line_mass_absorption_coefficient
    support = np.flatnonzero(slab[7] != 0.0)
    expected = np.arange(101, 302, dtype=np.int64)
    if not np.array_equal(support, expected):
        raise RuntimeError(
            "controlled public call no longer owns 101 red plus 100 blue samples"
        )
    if slab[7, 100] != 0.0 or slab[7, 302] != 0.0:
        raise RuntimeError("controlled public call deposited beyond the exact caps")
    if state.selected_line_count != 1:
        raise RuntimeError("controlled wing-cap line did not contribute")
    return {
        "seam__caps__center_grid_index": np.asarray(200, dtype=np.int16),
        "seam__caps__first_red_index": np.asarray(201, dtype=np.int16),
        "seam__caps__first_blue_index": np.asarray(200, dtype=np.int16),
        "seam__caps__red_indices": np.arange(201, 302, dtype=np.int16),
        "seam__caps__blue_indices": np.arange(200, 100, -1, dtype=np.int16),
        "seam__caps__support_indices": support.astype(np.int16),
        "seam__caps__red_count": np.asarray(101, dtype=np.int16),
        "seam__caps__blue_count": np.asarray(100, dtype=np.int16),
        "seam__caps__outside_is_zero": np.asarray(True, dtype=np.bool_),
    }


def _scalar_wing_contributions(
    modules: SimpleNamespace,
    values: Mapping[str, np.ndarray],
    ledger: Mapping[str, float],
    *,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    width = np.float32(values["fractional_widths"][7, LINE_SLOT_ZERO_BASED])
    center_absorption = np.float32(
        np.float32(ledger["classical_strength"])
        * np.float32(values["line_support"][7, LINE_SLOT_ZERO_BASED])
    )
    radiative = np.float32(ledger["radiative_damping"])
    damping = np.float64(np.float32(radiative / width))
    doppler_wavelength_width = float(width) * float(ledger["wavelength_nm"])
    basis = modules.line_profile_math.build_voigt_profile_basis()
    if side == "red":
        indices = np.arange(201, 302, dtype=np.int64)
        offsets = values["grid"][indices] - ledger["wavelength_nm"]
    elif side == "blue":
        indices = np.arange(200, 100, -1, dtype=np.int64)
        offsets = ledger["wavelength_nm"] - values["grid"][indices]
    else:
        raise ValueError(f"unknown wing side {side!r}")
    contributions = np.empty(indices.size, dtype=np.float64)
    for position, offset in enumerate(offsets):
        voigt_offset = float(offset) / doppler_wavelength_width
        if damping <= 0.2 and voigt_offset > 10.0:
            profile = 0.5642 * float(damping) / (voigt_offset * voigt_offset)
        else:
            profile = modules.line_profile_math.evaluate_voigt_profile(
                voigt_offset, float(damping), basis
            )
        contributions[position] = float(center_absorption) * float(profile)
    return indices, contributions


def _find_first_below_case(contributions: np.ndarray) -> tuple[int, np.float32]:
    for j in range(2, contributions.size - 2):
        q = np.nextafter(
            np.float32(contributions[j + 1]),
            np.float32(np.inf),
            dtype=np.float32,
        )
        if (
            np.isfinite(q)
            and contributions[j] >= float(q)
            and contributions[j + 1] < float(q)
            and float(q) > 0.0
        ):
            return j, q
    raise RuntimeError("could not construct a representable first-below-cutoff seam")


def _first_below_evidence(
    modules: SimpleNamespace,
    selected: Any,
    ledger: Mapping[str, float],
) -> dict[str, np.ndarray]:
    values = _controlled_inputs(
        ledger["wavelength_nm"],
        support_value=1.0e18,
        fractional_width=1.0e-5,
        threshold_value=0.0,
    )
    output: dict[str, np.ndarray] = {}
    for side in ("red", "blue"):
        indices, contributions = _scalar_wing_contributions(
            modules, values, ledger, side=side
        )
        j, threshold = _find_first_below_case(contributions)
        trial = {name: array.copy() for name, array in values.items()}
        trial["threshold"][:] = threshold
        state = _run_controlled(modules, selected, trial)
        row = state.line_mass_absorption_coefficient[7]
        retained = int(indices[j + 1])
        absent = int(indices[j + 2])
        if row[retained] == 0.0 or row[absent] != 0.0:
            raise RuntimeError(f"{side} first-below-cutoff ownership changed")
        output[f"seam__first_below__{side}__threshold_float32"] = np.asarray(
            threshold, dtype=np.float32
        )
        output[f"seam__first_below__{side}__above_index"] = np.asarray(
            indices[j], dtype=np.int16
        )
        output[f"seam__first_below__{side}__retained_below_index"] = np.asarray(
            retained, dtype=np.int16
        )
        output[f"seam__first_below__{side}__next_absent_index"] = np.asarray(
            absent, dtype=np.int16
        )
        output[f"seam__first_below__{side}__scalar_triplet"] = np.asarray(
            contributions[j : j + 3], dtype=np.float64
        )

    slab_equal = np.zeros((1, 401), dtype=np.float32)
    gaussian = np.ones(2001, dtype=np.float64)
    corrections = np.zeros(2001, dtype=np.float64)
    modules.line_opacity._accumulate_selected_line_wings_compiled(
        slab_equal,
        0,
        201,
        ledger["wavelength_nm"],
        1.0,
        0.1,
        1.0,
        1.0,
        values["grid"],
        gaussian,
        corrections,
        corrections,
    )
    equal_support = np.flatnonzero(slab_equal[0] != 0.0)
    if not np.array_equal(equal_support, np.arange(101, 302)):
        raise RuntimeError("wing equality no longer continues to both caps")

    slab_above = np.zeros((1, 401), dtype=np.float32)
    above_one = np.nextafter(np.float64(1.0), np.float64(np.inf))
    modules.line_opacity._accumulate_selected_line_wings_compiled(
        slab_above,
        0,
        201,
        ledger["wavelength_nm"],
        1.0,
        0.1,
        1.0,
        above_one,
        values["grid"],
        gaussian,
        corrections,
        corrections,
    )
    above_support = np.flatnonzero(slab_above[0] != 0.0)
    if not np.array_equal(above_support, np.asarray([200, 201])):
        raise RuntimeError("first strictly-below equality helper behavior changed")
    output["seam__wing_equality__support_indices"] = equal_support.astype(np.int16)
    output["seam__wing_nextafter__support_indices"] = above_support.astype(np.int16)
    output["seam__wing_equality__continues_to_caps"] = np.asarray(True, dtype=np.bool_)
    output["seam__wing_nextafter__retains_first_each_side"] = np.asarray(
        True, dtype=np.bool_
    )
    return output


def _gate_case(
    modules: SimpleNamespace,
    selected: Any,
    center_nm: float,
    active_depths_1based: Sequence[int],
) -> Any:
    values = _controlled_inputs(
        center_nm,
        support_value=0.0,
        fractional_width=1.0e-3,
        threshold_value=1.0e-18,
    )
    for depth in active_depths_1based:
        values["line_support"][depth - 1, LINE_SLOT_ZERO_BASED] = 1.0
    return _run_controlled(modules, selected, values)


def _gate_evidence(
    modules: SimpleNamespace, selected: Any, center_nm: float
) -> dict[str, np.ndarray]:
    cases = (
        ("interior_only", (10,), 10, False, 0),
        ("interior_plus_gate8", (8, 10), 10, True, 1),
        ("interior_plus_gate16", (10, 16), 10, True, 1),
        ("first_block_interior_only", (1,), 1, False, 0),
        ("first_block_plus_gate8", (1, 8), 1, True, 1),
        ("final_block_interior_only", (75,), 75, False, 0),
        ("final_block_plus_gate72", (72, 75), 75, True, 1),
        ("final_block_plus_gate80", (75, 80), 75, True, 1),
    )
    labels = []
    interior_deposited = []
    counts = []
    for (
        label,
        active_depths,
        inspected_depth,
        expected_deposit,
        expected_count,
    ) in cases:
        state = _gate_case(modules, selected, center_nm, active_depths)
        deposited = bool(
            np.any(state.line_mass_absorption_coefficient[inspected_depth - 1] != 0.0)
        )
        if deposited != expected_deposit or state.selected_line_count != expected_count:
            raise RuntimeError(f"8-layer gate topology changed for {label}")
        labels.append(label)
        interior_deposited.append(deposited)
        counts.append(state.selected_line_count)
    return {
        "seam__gates__case": np.asarray(labels, dtype="<U32"),
        "seam__gates__interior_deposited": np.asarray(
            interior_deposited, dtype=np.bool_
        ),
        "seam__gates__selected_line_count": np.asarray(counts, dtype=np.int16),
        "seam__gates__topology_verified": np.asarray(True, dtype=np.bool_),
    }


def _center_cutoff_evidence(
    modules: SimpleNamespace,
    selected: Any,
    ledger: Mapping[str, float],
) -> dict[str, np.ndarray]:
    base = _controlled_inputs(
        ledger["wavelength_nm"],
        support_value=0.0,
        fractional_width=1.0e-3,
        threshold_value=0.0,
    )
    base["line_support"][7, LINE_SLOT_ZERO_BASED] = 1.0
    pre = np.float32(
        np.float32(ledger["classical_strength"])
        * np.float32(base["line_support"][7, LINE_SLOT_ZERO_BASED])
    )
    exponential_tables = modules.line_profile_math.build_fast_exponential_tables()
    hc_value = np.float32(1.0e-4)
    x = float(np.float32(ledger["lower_excitation_cm"])) * float(hc_value)
    fast = modules.line_opacity._fast_exponential_lookup_compiled(
        x,
        exponential_tables.integer_step,
        exponential_tables.fractional_step,
    )
    post = np.float32(pre * fast)
    cases = (
        (
            "pre_below",
            np.nextafter(pre, np.float32(np.inf), dtype=np.float32),
            np.float32(0.0),
            False,
        ),
        ("pre_equal", pre, np.float32(0.0), True),
        (
            "post_below",
            np.nextafter(post, np.float32(np.inf), dtype=np.float32),
            hc_value,
            False,
        ),
        ("post_equal", post, hc_value, True),
    )
    labels = []
    thresholds = []
    survived = []
    counts = []
    for label, threshold, hc_over_kt, expected in cases:
        values = {name: array.copy() for name, array in base.items()}
        values["threshold"][:] = threshold
        values["hc_over_kt"][:] = hc_over_kt
        state = _run_controlled(modules, selected, values)
        deposited = bool(np.any(state.line_mass_absorption_coefficient[7] != 0.0))
        if deposited != expected or state.selected_line_count != int(expected):
            raise RuntimeError(f"center-cutoff ordering changed for {label}")
        labels.append(label)
        thresholds.append(threshold)
        survived.append(deposited)
        counts.append(state.selected_line_count)
    if not (pre > post > 0.0):
        raise RuntimeError("controlled center-cutoff pre/post ordering is invalid")
    return {
        "seam__center_cutoff__case": np.asarray(labels, dtype="<U16"),
        "seam__center_cutoff__pre_excitation_center_float32": np.asarray(
            pre, dtype=np.float32
        ),
        "seam__center_cutoff__post_fastex_center_float32": np.asarray(
            post, dtype=np.float32
        ),
        "seam__center_cutoff__threshold_float32": np.asarray(
            thresholds, dtype=np.float32
        ),
        "seam__center_cutoff__survived": np.asarray(survived, dtype=np.bool_),
        "seam__center_cutoff__selected_line_count": np.asarray(counts, dtype=np.int16),
    }


def _projection_lifecycle_evidence(
    modules: SimpleNamespace,
    selected: Any,
    ledger: Mapping[str, float],
    baseline_pre: np.ndarray,
    stimulated: np.ndarray,
    post: np.ndarray,
    actual: np.ndarray,
    line_support: np.ndarray,
    fractional_widths: np.ndarray,
    fixture: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    unowned_actual = actual.copy()
    unowned_actual[:, 1] = 1.0e99
    unowned_support = line_support.copy()
    unowned_support[:, 349] = 1.0e99
    unowned_widths = fractional_widths.copy()
    unowned_widths[:, 349] = 1.0
    unowned_state = _public_call(
        modules,
        selected,
        grid=fixture["opacity_wavelength_grid_nm"],
        bin_edges=fixture["wavelength_bin_edges"],
        threshold=fixture["continuum_line_selection_threshold"],
        temperature=fixture["temperature"],
        hc_over_kt=fixture["hc_over_kt"],
        electron_density=fixture["electron_density"],
        actual=unowned_actual,
        line_support=unowned_support,
        fractional_widths=unowned_widths,
    )
    if not np.array_equal(baseline_pre, unowned_state.line_mass_absorption_coefficient):
        raise RuntimeError("an unowned packed column changed the one-line output")

    controlled = _controlled_inputs(
        ledger["wavelength_nm"],
        support_value=1.0e18,
        fractional_width=1.0e-5,
        threshold_value=0.0,
    )
    controlled_base = _run_controlled(modules, selected, controlled)
    actual_changed = []
    for slot in (0, 2, 840):
        trial = {name: array.copy() for name, array in controlled.items()}
        trial["actual"][:, slot] = 1.0e30
        changed = _run_controlled(modules, selected, trial)
        actual_changed.append(
            not np.array_equal(
                controlled_base.line_mass_absorption_coefficient,
                changed.line_mass_absorption_coefficient,
            )
        )
    if not all(actual_changed):
        raise RuntimeError("an owned neutral-collision slot did not change the route")

    support_trial = {name: array.copy() for name, array in controlled.items()}
    support_trial["line_support"][:, LINE_SLOT_ZERO_BASED] *= 2.0
    support_changed = _run_controlled(modules, selected, support_trial)
    width_trial = {name: array.copy() for name, array in controlled.items()}
    width_trial["fractional_widths"][:, LINE_SLOT_ZERO_BASED] *= 2.0
    width_changed = _run_controlled(modules, selected, width_trial)
    if np.array_equal(
        controlled_base.line_mass_absorption_coefficient,
        support_changed.line_mass_absorption_coefficient,
    ):
        raise RuntimeError("owned line-support slot perturbation had no effect")
    if np.array_equal(
        controlled_base.line_mass_absorption_coefficient,
        width_changed.line_mass_absorption_coefficient,
    ):
        raise RuntimeError("owned fractional-width slot perturbation had no effect")

    reconstructed_post = baseline_pre.astype(np.float64) * stimulated
    if not np.array_equal(reconstructed_post, post):
        raise RuntimeError("post-stimulated view is not exactly pre times stimulation")
    if not np.array_equal(np.flatnonzero(baseline_pre), np.flatnonzero(post)):
        raise RuntimeError("pre/post sparse support changed under stimulation")
    double_stimulated = post * stimulated
    if np.array_equal(double_stimulated, post):
        raise RuntimeError("double stimulation was not detectably different")
    return {
        "seam__projection__accepted_fixture_worker_parity_bound": np.asarray(
            True, dtype=np.bool_
        ),
        "seam__projection__unowned_columns_invariant": np.asarray(True, dtype=np.bool_),
        "seam__projection__owned_actual_slots": np.asarray([0, 2, 840], dtype=np.int16),
        "seam__projection__owned_actual_slots_change_route": np.asarray(
            actual_changed, dtype=np.bool_
        ),
        "seam__projection__actual_slots_change_only_neutral_collision_input": (
            np.asarray(True, dtype=np.bool_)
        ),
        "seam__projection__line_support_slot_changes_line": np.asarray(
            True, dtype=np.bool_
        ),
        "seam__projection__fractional_width_slot_changes_line": np.asarray(
            True, dtype=np.bool_
        ),
        "seam__lifecycle__post_equals_pre_times_stimulated": np.asarray(
            True, dtype=np.bool_
        ),
        "seam__lifecycle__pre_post_support_identical": np.asarray(True, dtype=np.bool_),
        "seam__lifecycle__double_stimulation_detected": np.asarray(
            True, dtype=np.bool_
        ),
        "seam__lifecycle__golden_read_performed": np.asarray(False, dtype=np.bool_),
    }


def _dtype_evidence(
    modules: SimpleNamespace,
    selected: Any,
    fixture: Mapping[str, np.ndarray],
    pre: np.ndarray,
    post: np.ndarray,
) -> dict[str, np.ndarray]:
    allocated = modules.line_opacity.allocate_line_opacity_state(
        layer_count=80, wavelength_count=401
    )
    selected_dtypes = np.asarray(
        [
            selected.packed_wavelength_index.dtype.str,
            selected.packed_species_slot.dtype.str,
            selected.lower_excitation_index.dtype.str,
            selected.log_strength_index.dtype.str,
            selected.radiative_damping_index.dtype.str,
            selected.stark_damping_index.dtype.str,
            selected.van_der_waals_damping_index.dtype.str,
        ],
        dtype="<U4",
    )
    expected_selected = np.asarray(
        ["<i4", "<i2", "<i2", "<i2", "<i2", "<i2", "<i2"], dtype="<U4"
    )
    if not np.array_equal(selected_dtypes, expected_selected):
        raise RuntimeError("selected-record dtype boundary changed")
    sanitized_float32 = (
        modules.line_opacity._sanitize_float32(
            fixture["continuum_line_selection_threshold"]
        ),
        modules.line_opacity._sanitize_float32(fixture["electron_density"]),
        modules.line_opacity._sanitize_float32(
            fixture[
                "partition_normalized_population_over_mass_density_and_"
                "fractional_doppler_width_at_line_slot"
            ]
        ),
        modules.line_opacity._sanitize_float32(fixture["hc_over_kt"], ceiling=1.0e10),
        modules.line_opacity._sanitize_float32(
            fixture["fractional_doppler_widths_at_line_slot"], ceiling=1.0e10
        ),
    )
    if any(values.dtype != np.float32 for values in sanitized_float32):
        raise RuntimeError("public-call float32 sanitation boundary changed")
    if (
        allocated.line_mass_absorption_coefficient.dtype != np.float64
        or pre.dtype != np.float32
        or post.dtype != np.float64
    ):
        raise RuntimeError("line-state or downstream dtype boundary changed")
    return {
        "seam__dtype__allocated_empty_slab": np.asarray(
            allocated.line_mass_absorption_coefficient.dtype.str, dtype="<U4"
        ),
        "seam__dtype__nonempty_pre_slab": np.asarray(pre.dtype.str, dtype="<U4"),
        "seam__dtype__post_stimulated_view": np.asarray(post.dtype.str, dtype="<U4"),
        "seam__dtype__selected_fields": selected_dtypes,
        "seam__dtype__compiled_float32_input_count": np.asarray(
            len(sanitized_float32), dtype=np.int16
        ),
        "seam__dtype__actual_population_input": np.asarray("<f8", dtype="<U4"),
        "seam__dtype__wavelength_grid_input": np.asarray("<f8", dtype="<U4"),
    }


def _stimulated_view(
    modules: SimpleNamespace,
    pre: np.ndarray,
    grid: np.ndarray,
    temperature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frequency_hz = 2.99792458e17 / np.asarray(grid, dtype=np.float64)
    h_over_kt = 6.6256e-27 / (np.asarray(temperature, dtype=np.float64) * 1.38054e-16)
    stimulated = np.maximum(
        1.0 - np.exp(-h_over_kt[:, np.newaxis] * frequency_hz[np.newaxis, :]),
        1.0e-300,
    )
    helper_stimulated = np.empty_like(stimulated)
    for wavelength_index, frequency in enumerate(frequency_hz):
        _, column = modules.runner._planck_source_and_stimulated_emission(
            frequency_hz=float(frequency), h_over_kt=h_over_kt
        )
        helper_stimulated[:, wavelength_index] = column
    if not np.array_equal(stimulated, helper_stimulated):
        maximum = float(np.max(np.abs(stimulated - helper_stimulated)))
        raise RuntimeError(
            "broadcast stimulation differs from runner column helper; "
            f"maximum absolute difference={maximum}"
        )
    post = pre.astype(np.float64) * stimulated
    return stimulated, post


def _capture_schema_digest(results: Mapping[str, np.ndarray]) -> str:
    return schema_digest(results)


def _fingerprint(
    results: Mapping[str, np.ndarray], *, oracle_payload_only: bool
) -> str:
    selected: dict[str, np.ndarray] = {}
    for name, values in results.items():
        if name in FINGERPRINT_FIELDS:
            continue
        if oracle_payload_only and name.startswith(("meta__", "identity__")):
            continue
        selected[name] = values
    return mapping_digest(selected)


def _validate_complete_result(results: Mapping[str, np.ndarray]) -> None:
    for name, values in results.items():
        array = np.asarray(values)
        if array.dtype.hasobject:
            raise RuntimeError(f"capture contains object dtype: {name}")
        if not array.flags.c_contiguous:
            raise RuntimeError(f"capture contains noncontiguous member: {name}")
    if not bool(results["meta__capture_scope_complete"]):
        raise RuntimeError("oracle capture is not complete")
    if bool(results["meta__golden_read_performed"]):
        raise RuntimeError("oracle capture read a golden")
    if bool(results["meta__serialization_performed"]):
        raise RuntimeError("oracle capture serialized output")
    if bool(results["meta__publication_performed"]):
        raise RuntimeError("oracle capture performed publication")
    if results["pre_stimulated_dense_float32"].shape != (80, 30_000):
        raise RuntimeError("pre-stimulated dense shape changed")
    if results["pre_stimulated_dense_float32"].dtype != np.float32:
        raise RuntimeError("pre-stimulated dense dtype changed")
    if results["post_stimulated_dense_float64"].dtype != np.float64:
        raise RuntimeError("post-stimulated dense dtype changed")
    if int(results["selected_line_count"]) != 1:
        raise RuntimeError("selected-line count changed")
    if not np.array_equal(results["gate_depth_1based"], GATE_DEPTHS_1BASED):
        raise RuntimeError("physical gate-depth topology changed")
    if not np.all(results["gate_active"]):
        raise RuntimeError("physical fixture no longer activates all gates")


def build_oracle_results(*, fixture_path: Path = FIXTURE_PATH) -> dict[str, np.ndarray]:
    """Execute the complete scientific observation without persistence."""

    # Every authority, file, and process check precedes pinned package import.
    identities = verify_preimport_identities()
    fixture = load_fixture(fixture_path)
    environment = require_environment()
    with _record_npz_reads() as import_reads:
        modules = load_pinned_modules()
    selected = _selected_catalog(modules, fixture)
    actual, line_support, fractional_widths = reconstruct_public_call_arrays(fixture)
    ledger = _line_parameter_ledger(modules, selected)

    if ledger["wavelength_nm"] != EXPECTED_RECONSTRUCTED_WAVELENGTH_NM:
        raise RuntimeError("reconstructed selected-line wavelength changed")
    first_red = int(
        np.searchsorted(
            fixture["opacity_wavelength_grid_nm"],
            ledger["wavelength_nm"],
            side="right",
        )
    )
    if first_red != EXPECTED_FIRST_RED_INDEX:
        raise RuntimeError("physical fixture first-red grid index changed")
    continuum_column = int(
        np.searchsorted(
            fixture["wavelength_bin_edges"],
            int(selected.packed_wavelength_index[0]),
            side="right",
        )
    )
    if continuum_column != EXPECTED_CONTINUUM_COLUMN:
        raise RuntimeError("physical fixture continuum reference column changed")

    parallel_original = modules.line_opacity._accumulate_selected_line_opacity_parallel

    def forbidden_parallel(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("parallel selected-line wrapper was called for one line")

    modules.line_opacity._accumulate_selected_line_opacity_parallel = forbidden_parallel
    try:
        with _record_npz_reads() as call_reads:
            state = _public_call(
                modules,
                selected,
                grid=fixture["opacity_wavelength_grid_nm"],
                bin_edges=fixture["wavelength_bin_edges"],
                threshold=fixture["continuum_line_selection_threshold"],
                temperature=fixture["temperature"],
                hc_over_kt=fixture["hc_over_kt"],
                electron_density=fixture["electron_density"],
                actual=actual,
                line_support=line_support,
                fractional_widths=fractional_widths,
            )
    finally:
        modules.line_opacity._accumulate_selected_line_opacity_parallel = (
            parallel_original
        )
    # The canonical fixture was loaded before package import.  The two traced
    # sets own all package/runtime table reads; adding that already-validated
    # fixture path yields the complete NPZ input closure.
    dynamic_reads = import_reads | call_reads | {str(FIXTURE_PATH.resolve())}
    expected_reads = {
        str(FIXTURE_PATH.resolve()),
        str(LINE_TABLE_PATH.resolve()),
        str(MOLECULAR_EQUILIBRIUM_TABLE_PATH.resolve()),
    }
    if dynamic_reads != expected_reads:
        raise RuntimeError(
            "oracle dynamic NPZ read closure changed; "
            f"expected={sorted(expected_reads)}, actual={sorted(dynamic_reads)}"
        )

    pre = np.array(state.line_mass_absorption_coefficient, copy=True, order="C")
    if pre.shape != (80, 30_000) or pre.dtype != np.float32:
        raise RuntimeError("public one-line output shape/dtype changed")
    if state.selected_line_count != 1:
        raise RuntimeError("public one-line selected count changed")
    pre_hash = array_sha256(pre)
    if pre_hash != EXPECTED_DENSE_PRE_SHA256:
        raise RuntimeError(f"pre-stimulated dense hash changed: {pre_hash}")
    if np.max(pre) != EXPECTED_PEAK_PRE:
        raise RuntimeError("pre-stimulated peak changed")

    stimulated, post = _stimulated_view(
        modules,
        pre,
        fixture["opacity_wavelength_grid_nm"],
        fixture["temperature"],
    )
    post_hash = array_sha256(post)
    if post_hash != EXPECTED_DENSE_POST_SHA256:
        raise RuntimeError(f"post-stimulated dense hash changed: {post_hash}")
    nonzero = np.flatnonzero(pre).astype(np.int64)
    pre_sparse = np.ascontiguousarray(pre.ravel(order="C")[nonzero])
    post_sparse = np.ascontiguousarray(post.ravel(order="C")[nonzero])
    if (
        nonzero.size != EXPECTED_NONZERO_COUNT
        or array_sha256(nonzero) != EXPECTED_SPARSE_INDEX_SHA256
        or array_sha256(pre_sparse) != EXPECTED_SPARSE_PRE_SHA256
        or array_sha256(post_sparse) != EXPECTED_SPARSE_POST_SHA256
    ):
        raise RuntimeError("canonical sparse encoding changed")
    if not np.array_equal(nonzero, np.flatnonzero(post)):
        raise RuntimeError("pre/post sparse support differs")
    nonzero_per_depth = np.count_nonzero(pre, axis=1)
    if not np.array_equal(nonzero_per_depth, np.full(80, 3, dtype=np.int64)):
        raise RuntimeError("physical fixture no longer has three pixels per depth")
    gate_active = np.any(pre[GATE_DEPTHS_1BASED.astype(np.int64) - 1] != 0.0, axis=1)
    if not np.all(gate_active):
        raise RuntimeError("physical fixture no longer activates all ten gates")

    results: dict[str, Any] = {
        "meta__capture_schema_version": np.asarray(
            CAPTURE_SCHEMA_VERSION, dtype=np.int16
        ),
        "meta__capture_scope_complete": np.asarray(True, dtype=np.bool_),
        "meta__capture_status": np.asarray(
            "candidate complete; independent review pending", dtype="<U64"
        ),
        "meta__fixture_build_performed": np.asarray(False, dtype=np.bool_),
        "meta__golden_read_performed": np.asarray(False, dtype=np.bool_),
        "meta__serialization_performed": np.asarray(False, dtype=np.bool_),
        "meta__publication_performed": np.asarray(False, dtype=np.bool_),
        "meta__capture_schema_digest": np.asarray(b"0" * 64, dtype="|S64"),
        "meta__oracle_payload_fingerprint": np.asarray(b"0" * 64, dtype="|S64"),
        "meta__full_capture_fingerprint": np.asarray(b"0" * 64, dtype="|S64"),
        "identity__worker_sha256": np.asarray(sha256(WORKER_PATH), dtype="<U64"),
        "identity__fixture_sha256": np.asarray(FIXTURE_SHA256, dtype="<U64"),
        "identity__fixture_schema_digest": np.asarray(
            FIXTURE_SCHEMA_DIGEST, dtype="<U64"
        ),
        "identity__fixture_capture_payload_fingerprint": np.asarray(
            FIXTURE_CAPTURE_PAYLOAD_FINGERPRINT, dtype="<U64"
        ),
        "identity__raw_subset_sha256": np.asarray(SUBSET_SHA256, dtype="<U64"),
        "identity__raw_subset_row_index": np.asarray(SUBSET_ROW_INDEX, dtype=np.int64),
        "identity__raw_subset_schema_version": np.asarray(
            SUBSET_SCHEMA_VERSION, dtype=np.int16
        ),
        "identity__conversion_version": np.asarray(CONVERSION_VERSION, dtype=np.int16),
        "identity__line_table_sha256": np.asarray(LINE_TABLE_SHA256, dtype="<U64"),
        "identity__fixture_manifest_entry_digest": np.asarray(
            identities["fixture_entry_digest"], dtype="<U64"
        ),
        "identity__subset_manifest_entry_digest": np.asarray(
            identities["subset_entry_digest"], dtype="<U64"
        ),
        "identity__loaded_pinned_python_source_count": np.asarray(
            len(modules.loaded_python_manifest), dtype=np.int16
        ),
        "identity__loaded_pinned_python_names": np.asarray(
            sorted(modules.loaded_python_manifest), dtype="<U64"
        ),
        "identity__loaded_pinned_python_hashes": np.asarray(
            [
                modules.loaded_python_manifest[name]
                for name in sorted(modules.loaded_python_manifest)
            ],
            dtype="<U64",
        ),
        "identity__critical_source_names": np.asarray(
            sorted(CRITICAL_SOURCE_HASHES), dtype="<U64"
        ),
        "identity__critical_source_hashes": np.asarray(
            [CRITICAL_SOURCE_HASHES[name] for name in sorted(CRITICAL_SOURCE_HASHES)],
            dtype="<U64",
        ),
        "identity__fixture_member_names": np.asarray(
            sorted(FIXTURE_MEMBER_CONTRACT), dtype="<U128"
        ),
        "identity__fixture_member_hashes": np.asarray(
            [
                FIXTURE_MEMBER_CONTRACT[name][2]
                for name in sorted(FIXTURE_MEMBER_CONTRACT)
            ],
            dtype="<U64",
        ),
        "identity__dynamic_npz_reads": np.asarray(
            sorted(_display_input_path(Path(path)) for path in dynamic_reads),
            dtype="<U96",
        ),
        "identity__manifest_semantic_labels_authoritative": np.asarray(
            False, dtype=np.bool_
        ),
        "identity__contradictory_manifest_label_paths": np.asarray(
            sorted(CONTRADICTORY_MANIFEST_LABELS), dtype="<U64"
        ),
        "identity__contradictory_manifest_label_values": np.asarray(
            [
                CONTRADICTORY_MANIFEST_LABELS[name]
                for name in sorted(CONTRADICTORY_MANIFEST_LABELS)
            ],
            dtype="<U64",
        ),
        "identity__source_semantic_actual_population_unit": np.asarray(
            "cm^-3", dtype="<U16"
        ),
        "identity__source_semantic_packed_species_code": np.asarray(
            int(selected.packed_species_slot[0]), dtype=np.int16
        ),
        "identity__source_semantic_decoded_population_slot_1based": np.asarray(
            abs(int(selected.packed_species_slot[0])) // 10, dtype=np.int16
        ),
        "identity__source_semantic_wavelength_bin_kind": np.asarray(
            "packed wavelength codes with final sentinel", dtype="<U48"
        ),
        "identity__python_version": np.asarray(platform.python_version(), dtype="<U32"),
        "identity__numpy_version": np.asarray(np.__version__, dtype="<U32"),
        "identity__numba_version": np.asarray(modules.numba.__version__, dtype="<U32"),
        "identity__command": np.asarray(
            "python scripts/chapter06_atmosphere_oracle_worker.py", dtype="<U96"
        ),
        "reconstructed_wavelength_nm": np.asarray(
            ledger["wavelength_nm"], dtype=np.float64
        ),
        "first_red_grid_index_zero_based": np.asarray(first_red, dtype=np.int64),
        "continuum_reference_column_zero_based": np.asarray(
            continuum_column, dtype=np.int64
        ),
        "pre_stimulated_dense_float32": pre,
        "stimulated_emission_factor_float64": stimulated,
        "post_stimulated_dense_float64": post,
        "dense_shape": np.asarray(pre.shape, dtype=np.int64),
        "nonzero_flat_index": nonzero,
        "pre_stimulated_nonzero_value": pre_sparse,
        "post_stimulated_nonzero_value": post_sparse,
        "selected_line_count": np.asarray(state.selected_line_count, dtype=np.int64),
        "gate_depth_1based": GATE_DEPTHS_1BASED,
        "gate_active": gate_active.astype(np.bool_),
        "dense_pre_stimulated_sha256": np.asarray(
            pre_hash.encode("ascii"), dtype="|S64"
        ),
        "dense_post_stimulated_sha256": np.asarray(
            post_hash.encode("ascii"), dtype="|S64"
        ),
        "physical_nonzero_count_per_depth": nonzero_per_depth.astype(np.int16),
        "physical_peak_pre_stimulated_float32": np.asarray(
            np.max(pre), dtype=np.float32
        ),
        "route__parallel_wrapper_forbidden_and_dormant": np.asarray(
            True, dtype=np.bool_
        ),
        **environment,
        **_route_source_evidence(),
        **_depth_contract_evidence(modules, selected, ledger["wavelength_nm"]),
        **_wing_cap_evidence(modules, selected, ledger["wavelength_nm"]),
        **_first_below_evidence(modules, selected, ledger),
        **_gate_evidence(modules, selected, ledger["wavelength_nm"]),
        **_center_cutoff_evidence(modules, selected, ledger),
        **_projection_lifecycle_evidence(
            modules,
            selected,
            ledger,
            pre,
            stimulated,
            post,
            actual,
            line_support,
            fractional_widths,
            fixture,
        ),
        **_dtype_evidence(modules, selected, fixture, pre, post),
    }
    results = deterministic_result(results)
    results["meta__capture_schema_digest"] = np.asarray(
        _capture_schema_digest(results).encode("ascii"), dtype="|S64"
    )
    results["meta__oracle_payload_fingerprint"] = np.asarray(
        _fingerprint(results, oracle_payload_only=True).encode("ascii"),
        dtype="|S64",
    )
    results["meta__full_capture_fingerprint"] = np.asarray(
        _fingerprint(results, oracle_payload_only=False).encode("ascii"),
        dtype="|S64",
    )
    results = deterministic_result(results)
    _validate_complete_result(results)
    return results


def summarize(results: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Return a compact JSON-safe double-capture comparison summary."""

    return {
        "capture_scope_complete": bool(results["meta__capture_scope_complete"]),
        "capture_status": str(results["meta__capture_status"]),
        "key_count": len(results),
        "capture_schema_digest": bytes(results["meta__capture_schema_digest"]).decode(
            "ascii"
        ),
        "oracle_payload_fingerprint": bytes(
            results["meta__oracle_payload_fingerprint"]
        ).decode("ascii"),
        "full_capture_fingerprint": bytes(
            results["meta__full_capture_fingerprint"]
        ).decode("ascii"),
        "worker_sha256": str(results["identity__worker_sha256"]),
        "fixture_sha256": str(results["identity__fixture_sha256"]),
        "fixture_schema_digest": str(results["identity__fixture_schema_digest"]),
        "conversion_version": int(results["identity__conversion_version"]),
        "loaded_pinned_python_source_count": int(
            results["identity__loaded_pinned_python_source_count"]
        ),
        "dynamic_npz_reads": results["identity__dynamic_npz_reads"].tolist(),
        "reconstructed_wavelength_nm": float(results["reconstructed_wavelength_nm"]),
        "first_red_grid_index_zero_based": int(
            results["first_red_grid_index_zero_based"]
        ),
        "continuum_reference_column_zero_based": int(
            results["continuum_reference_column_zero_based"]
        ),
        "dense_shape": results["dense_shape"].tolist(),
        "pre_dtype": str(results["pre_stimulated_dense_float32"].dtype),
        "post_dtype": str(results["post_stimulated_dense_float64"].dtype),
        "selected_line_count": int(results["selected_line_count"]),
        "nonzero_count": int(results["nonzero_flat_index"].size),
        "nonzero_count_per_depth": results["physical_nonzero_count_per_depth"].tolist(),
        "gate_depth_1based": results["gate_depth_1based"].tolist(),
        "gate_active": results["gate_active"].tolist(),
        "peak_pre_stimulated": float(results["physical_peak_pre_stimulated_float32"]),
        "dense_pre_stimulated_sha256": bytes(
            results["dense_pre_stimulated_sha256"]
        ).decode("ascii"),
        "dense_post_stimulated_sha256": bytes(
            results["dense_post_stimulated_sha256"]
        ).decode("ascii"),
        "sparse_index_sha256": array_sha256(results["nonzero_flat_index"]),
        "sparse_pre_value_sha256": array_sha256(
            results["pre_stimulated_nonzero_value"]
        ),
        "sparse_post_value_sha256": array_sha256(
            results["post_stimulated_nonzero_value"]
        ),
        "serial_route_proved": bool(
            results["route__parallel_wrapper_forbidden_and_dormant"]
            and results["route__fast_exponential_precedes_first_wing"]
            and results["route__wing_calls_voigt_helper"]
        ),
        "depth_rejections": results["seam__depth__rejected_counts"].tolist(),
        "wing_cap_counts": [
            int(results["seam__caps__red_count"]),
            int(results["seam__caps__blue_count"]),
        ],
        "first_below_red_retained": int(
            results["seam__first_below__red__retained_below_index"]
        ),
        "first_below_blue_retained": int(
            results["seam__first_below__blue__retained_below_index"]
        ),
        "gate_topology_verified": bool(results["seam__gates__topology_verified"]),
        "center_cutoff_survived": results["seam__center_cutoff__survived"].tolist(),
        "dtype_boundary_verified": bool(
            str(results["seam__dtype__allocated_empty_slab"]) == "<f8"
            and str(results["seam__dtype__nonempty_pre_slab"]) == "<f4"
            and str(results["seam__dtype__post_stimulated_view"]) == "<f8"
        ),
        "projection_lifecycle_verified": bool(
            results["seam__projection__unowned_columns_invariant"]
            and results["seam__lifecycle__post_equals_pre_times_stimulated"]
            and results["seam__lifecycle__pre_post_support_identical"]
            and results["seam__lifecycle__double_stimulation_detected"]
        ),
        "fixture_build_performed": bool(results["meta__fixture_build_performed"]),
        "golden_read_performed": bool(results["meta__golden_read_performed"]),
        "serialization_performed": bool(results["meta__serialization_performed"]),
        "publication_performed": bool(results["meta__publication_performed"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse canonical input checks only; no output path is accepted."""

    parser = argparse.ArgumentParser(
        description="self-check the unpublished Chapter 6 atmosphere oracle"
    )
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="validate authorities and the canonical fixture without importing Payne Zero",
    )
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_args()
    if arguments.identity_only:
        identities = verify_preimport_identities()
        fixture = load_fixture(arguments.fixture)
        print(
            json.dumps(
                {
                    "commit": identities["payne_zero_commit"],
                    "fixture_member_count": len(fixture),
                    "fixture_sha256": FIXTURE_SHA256,
                    "fixture_manifest_entry_digest": identities["fixture_entry_digest"],
                    "conversion_version": CONVERSION_VERSION,
                },
                sort_keys=True,
            )
        )
        return
    results = build_oracle_results(fixture_path=arguments.fixture)
    print(json.dumps(summarize(results), sort_keys=True))


if __name__ == "__main__":
    main()
