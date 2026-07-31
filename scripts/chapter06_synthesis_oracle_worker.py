#!/usr/bin/env python3
"""Observe the pinned Chapter 6 one-line synthesis route entirely in memory.

This worker validates the accepted Chapter 5 state fixture, the immutable
one-row Fe I subset, all static tables, and the pinned Payne Zero checkout
before importing or executing synthesis code.  It returns detached,
object-free NumPy arrays.  It has no serialization or publication interface
and never reads a golden artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


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

# Establish thread controls before NumPy is imported when the worker is
# executed directly. Hash randomization must already be fixed by the parent.
if __name__ == "__main__":
    os.environ.update(THREAD_ENVIRONMENT)

import numpy as np  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_ROOT = Path("/Users/ysting/payne-zero")
PINNED_DATA_ROOT = PINNED_ROOT / "source_data_files"
WORKER_PATH = Path(__file__).resolve()
DESIGN_PATH = REPOSITORY_ROOT / "design/chapter06_synthesis_fixture_oracle_plan.md"
SOURCE_CONTRACT_PATH = REPOSITORY_ROOT / "design/chapter06_exact_source_contract.md"
GOLDEN_ROOT = REPOSITORY_ROOT / "data/golden"

PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"

FIXTURE_PATH = REPOSITORY_ROOT / "data/fixtures/chapter05_continuum_states.npz"
FIXTURE_SHA256 = "ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae"
FIXTURE_PAYLOAD_DIGEST = (
    "4bcf0bbd8d61e58334c4c7ef6caaaf9ca47e6fb4536ad0098d5a541d540ec048"
)
FIXTURE_KEY_COUNT = 217

SUBSET_PATH = REPOSITORY_ROOT / "data/subsets/chapter06_fe_i_source_row_873702.npz"
SUBSET_SHA256 = "bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04"
SUBSET_BYTES = 8665
SUBSET_KEY_COUNT = 27
SOURCE_ARCHIVE_RELATIVE_PATH = Path(
    "source_data_files/source_catalogs/lines/atomic_source_lines_parsed.npz"
)
SOURCE_ARCHIVE_PATH = PINNED_ROOT / SOURCE_ARCHIVE_RELATIVE_PATH
SOURCE_ARCHIVE_SHA256 = (
    "4eafa927c02a4f74401523149a44e35239f2aaecb4a64f2905a4cd5530c2dde7"
)
SOURCE_ARCHIVE_BYTES = 258_021_389
SOURCE_ARCHIVE_ROW_COUNT = 1_939_975
SOURCE_ROW_INDEX = 873_702

CONTINUUM_TABLES_PATH = (
    REPOSITORY_ROOT / "data/static/synthesis_tables/continuum_tables.npz"
)
CONTINUUM_EDGE_GRID_PATH = (
    REPOSITORY_ROOT / "data/static/synthesis_tables/continuum_edge_grid.npz"
)
LINE_PROFILE_TABLES_PATH = (
    REPOSITORY_ROOT / "data/static/synthesis_tables/line_profile_tables.npz"
)
STATIC_TABLE_IDENTITIES = {
    "continuum_tables": (
        CONTINUUM_TABLES_PATH,
        PINNED_DATA_ROOT / "synthesis_tables/continuum_tables.npz",
        "406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d",
    ),
    "continuum_edge_grid": (
        CONTINUUM_EDGE_GRID_PATH,
        PINNED_DATA_ROOT / "synthesis_tables/continuum_edge_grid.npz",
        "11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e",
    ),
    "line_profile_tables": (
        LINE_PROFILE_TABLES_PATH,
        PINNED_DATA_ROOT / "synthesis_tables/line_profile_tables.npz",
        "87b47fc76bed10455218f43c4b6686525b961002e72d6a5ef01255a08deb27d4",
    ),
}

REGIME_NAMES = (
    "hot_dwarf",
    "solar_dwarf",
    "low_gravity_giant",
    "cool_molecule_rich",
)
EXPECTED_ACTIVITY_MASK = np.asarray(
    [
        [True, True, True, False, False, False],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
    ],
    dtype=np.bool_,
)

GRID_SPECS = {
    "canonical": {
        "start_wavelength_nm": 495.0,
        "end_wavelength_nm": 505.0,
        "resolution": 300_000.0,
        "count": 6000,
        "first_wavelength_nm": 495.0009387906341,
        "last_wavelength_nm": 504.9989209057178,
        "center_index": 2434,
        "center_wavelength_nm": 499.0333758196059,
    },
    "coarse": {
        "start_wavelength_nm": 495.0,
        "end_wavelength_nm": 505.0,
        "resolution": 20_000.0,
        "count": 400,
        "center_wavelength_nm": 499.04420746429804,
    },
}

PIPELINE_CONTINUUM_FIELDS = (
    "temperature",
    "mass_density",
    "electron_density",
    "hydrogen_partition_normalized_ion_stage_populations",
    "hydrogen_neutral_population",
    "helium_neutral_population",
    "helium_singly_ionized_population",
    "carbon_partition_normalized_ion_stage_populations",
    "magnesium_neutral_partition_normalized_population",
    "aluminum_neutral_partition_normalized_population",
    "silicon_neutral_partition_normalized_population",
    "iron_neutral_partition_normalized_population",
    "partition_normalized_populations",
    "ion_stage_populations",
    "signed_continuum_edge_frequency_hz",
    "continuum_edge_wavelength_nm",
    "continuum_edge_midpoint_wavelength_nm",
    "continuum_edge_interval_width_squared_over_two_nm2",
)

LINE_STATE_FIELDS = (
    "partition_normalized_populations",
    "fractional_doppler_widths",
    "mass_density",
    "electron_density",
    "temperature",
    "hc_over_kt",
)

SYNTHESIS_FIXTURE_FIELDS = (
    "aluminum_neutral_partition_normalized_population",
    "atmosphere_schema_version",
    "carbon_partition_normalized_ion_stage_populations",
    "column_mass",
    "continuum_edge_interval_width_squared_over_two_nm2",
    "continuum_edge_midpoint_wavelength_nm",
    "continuum_edge_wavelength_nm",
    "electron_density",
    "elemental_abundances",
    "fractional_doppler_widths",
    "gas_pressure",
    "hc_over_kt",
    "helium_neutral_population",
    "helium_singly_ionized_population",
    "hydrogen_ionized_population",
    "hydrogen_neutral_population",
    "hydrogen_partition_normalized_ion_stage_populations",
    "ion_stage_populations",
    "iron_neutral_partition_normalized_population",
    "magnesium_neutral_partition_normalized_population",
    "mass_density",
    "microturbulence",
    "molecular_hydrogen_population",
    "partition_normalized_populations",
    "signed_continuum_edge_frequency_hz",
    "silicon_neutral_partition_normalized_population",
    "temperature",
)

ATMOSPHERE_FIXTURE_FIELDS = (
    "temperature",
    "mass_density",
    "electron_density",
    "gas_pressure",
    "hydrogen_partition_normalized_ion_stage_populations",
    "hydrogen_neutral_population",
    "hydrogen_ionized_population",
    "helium_neutral_population",
    "helium_singly_ionized_population",
    "helium_neutral_partition_normalized_population",
    "helium_singly_ionized_partition_normalized_population",
    "elemental_abundances_by_layer",
    "hydrogen_departure_coefficients",
    "microturbulence",
    "ion_stage_populations_by_packed_slot",
    "partition_normalized_populations_by_packed_slot",
    "ch_population",
    "oh_population",
)

FIXTURE_INPUT_FIELDS = (
    "column_mass",
    "electron_density_seed",
    "elemental_abundances",
    "gas_pressure",
    "microturbulence",
    "temperature",
)

FIXTURE_GLOBAL_FIELDS = (
    "atmosphere_grid_boundary_temperature_k",
    "atmosphere_runner_opacity_flags",
    "h2_policy_temperature_k",
    "hminus_edge_frequency_hz",
    "payne_zero_commit",
    "regime_names",
    "source_chapter04_fixture_sha256",
    "source_continuum_edge_grid_sha256",
    "synthesis_edge_probe_wavelength_nm",
)

RAW_FIELDS = (
    "stored_wavelength_nm",
    "raw_log_oscillator_strength",
    "species_code",
    "first_energy_column_cm",
    "second_energy_column_cm",
    "radiative_damping_log",
    "stark_damping_log",
    "van_der_waals_damping_log",
    "lower_principal_quantum_number",
    "upper_principal_quantum_number",
    "primary_isotope_number",
    "primary_isotope_log_correction",
    "secondary_isotope_log_correction",
    "energy_shift_field",
    "isotope_shift_units",
    "line_size",
    "line_category_tag",
)

RAW_FIELD_DTYPES = {
    "stored_wavelength_nm": np.dtype("<f8"),
    "raw_log_oscillator_strength": np.dtype("<f8"),
    "species_code": np.dtype("<f8"),
    "first_energy_column_cm": np.dtype("<f8"),
    "second_energy_column_cm": np.dtype("<f8"),
    "radiative_damping_log": np.dtype("<f8"),
    "stark_damping_log": np.dtype("<f8"),
    "van_der_waals_damping_log": np.dtype("<f8"),
    "lower_principal_quantum_number": np.dtype("<i8"),
    "upper_principal_quantum_number": np.dtype("<i8"),
    "primary_isotope_number": np.dtype("<i8"),
    "primary_isotope_log_correction": np.dtype("<f8"),
    "secondary_isotope_log_correction": np.dtype("<f8"),
    "energy_shift_field": np.dtype("S10"),
    "isotope_shift_units": np.dtype("<f8"),
    "line_size": np.dtype("<i8"),
    "line_category_tag": np.dtype("S3"),
}

SUBSET_PROVENANCE_FIELDS = (
    "builder_command",
    "payne_zero_commit",
    "source_archive_bytes",
    "source_archive_relative_path",
    "source_archive_row_count",
    "source_archive_sha256",
    "source_field_count",
    "source_row_index",
    "subset_role",
    "subset_schema_version",
)

CATALOG_LINE_FIELDS = (
    "line_type",
    "atomic_number",
    "ion_stage",
    "wavelength_nm",
    "index_wavelength_nm",
    "oscillator_strength",
    "lower_excitation_cm",
    "radiative_damping",
    "stark_damping",
    "van_der_waals_damping",
    "raw_radiative_damping_log",
    "raw_stark_damping_log",
    "raw_van_der_waals_damping_log",
)

CATALOG_SUPPORT_FIELDS = (
    "helium_line_type",
    "helium_line_center_cutoff_ratio",
    "harris_profile_h0_table",
    "harris_profile_h1_table",
    "harris_profile_h2_table",
)

ATOMIC_INVARIANT_FIELDS = (
    "wavelength_grid",
    "n_wavelengths",
    "grid_resolution",
    "metal_catalog_index",
    "metal_classical_strength",
    "metal_lower_excitation_cm",
    "metal_radiative_damping",
    "metal_stark_damping",
    "metal_van_der_waals_damping",
    "metal_wavelength_nm",
    "metal_population_ion_stage_index",
    "metal_population_element_index",
    "metal_center_index",
    "metal_wing_index",
    "metal_center_clamped",
    "metal_wing_clamped",
    "auto_catalog_index",
    "auto_oscillator_strength",
    "auto_lower_excitation_cm",
    "auto_radiative_damping",
    "auto_stark_damping",
    "auto_van_der_waals_damping",
    "auto_wavelength_nm",
    "auto_population_ion_stage_index",
    "auto_population_element_index",
    "auto_center_index",
    "auto_center_clamped",
    "helium_classical_strength",
    "helium_lower_excitation_cm",
    "helium_radiative_damping",
    "helium_stark_damping",
    "helium_van_der_waals_damping",
    "helium_wavelength_nm",
    "helium_population_ion_stage_index",
    "helium_population_element_index",
    "helium_center_index",
    "helium_line_type",
    "helium_cutoff",
    "harris_profile_h0_table",
    "harris_profile_h1_table",
    "harris_profile_h2_table",
    "exponential_integer_table",
    "exponential_fraction_table",
)

METAL_INVARIANT_FIELDS = tuple(
    name for name in ATOMIC_INVARIANT_FIELDS if name.startswith("metal_")
)
AUTO_INVARIANT_FIELDS = tuple(
    name for name in ATOMIC_INVARIANT_FIELDS if name.startswith("auto_")
)
HELIUM_ARRAY_INVARIANT_FIELDS = tuple(
    name
    for name in ATOMIC_INVARIANT_FIELDS
    if name.startswith("helium_") and name != "helium_cutoff"
)

# Importing any synthesis submodule executes the package API imports. This is
# the exact module set observed and accepted at the pinned commit.
FROZEN_PINNED_PYTHON_MANIFEST = {
    "payne_zero_synthesis/__init__.py": (
        "7e94560ab51be3baa3950a93e8b6f998c849a6d1d78356c5f662f6cfaba927db"
    ),
    "payne_zero_synthesis/api.py": (
        "77718303c1e0052a520ece7fab277b3b1922c21d09b35a288596592d03310940"
    ),
    "payne_zero_synthesis/atmosphere.py": (
        "06b79770e4d9472093655022d53ee7fddf7cc6727206f34c0f60c57151e2cf9b"
    ),
    "payne_zero_synthesis/atomic_lines.py": (
        "0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0"
    ),
    "payne_zero_synthesis/constants.py": (
        "ed58004196790f9fb4a2871044c9cd36bf7bc42046a923f9314f7b8ea7456798"
    ),
    "payne_zero_synthesis/continuum.py": (
        "ab0d4eb771ee04101f6936253f633ed60d845e2816854a06b1b059e8b91dce1b"
    ),
    "payne_zero_synthesis/device.py": (
        "22e769ebed60ad3a0f2060264247e469a99afd20ec5cadb69a01b6e5fa82ea3c"
    ),
    "payne_zero_synthesis/equation_of_state.py": (
        "6497c29abb954e0b55d918cc22fa7b660952812c548faf1d7b1053345ef13562"
    ),
    "payne_zero_synthesis/ground_partition_table.py": (
        "6950686c89ea51e301b4b11256d9413dad58d82741a51580f3547aa012ade832"
    ),
    "payne_zero_synthesis/hydrogen_lines.py": (
        "81ab3ee2ca9ecd1994ddde8f01e09535c5b74f7beec5afe98a3c63b44677dcca"
    ),
    "payne_zero_synthesis/line_opacity.py": (
        "639b95c3812f1a7d227b797fa89a4d6ef9725d5f0e1284f3d49cf86844278275"
    ),
    "payne_zero_synthesis/molecular_equilibrium.py": (
        "df01757c160b2bff4390cc2148cff9d1ba6e5a2bc7cab4515b46f38e868d2714"
    ),
    "payne_zero_synthesis/molecular_lines.py": (
        "14c9d07e431fa73e6d6938e9db2d11c6688e52348234e0aac37cc76e8be3dc32"
    ),
    "payne_zero_synthesis/paths.py": (
        "2bca3284eb1765449ab3fc87439eb603e3941213d9c1205c71aee3fd1ad30b5d"
    ),
    "payne_zero_synthesis/pipeline.py": (
        "465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22"
    ),
    "payne_zero_synthesis/radiative_transfer.py": (
        "52e0d1a0c4a2713294ce1b43130c5d900e54c4cf1f8b2b05058fc2d6831ff62b"
    ),
    "payne_zero_synthesis/synthesis.py": (
        "590e430b6582fbcf601a52b721d8f65073432903773a99238073a1d821fe0d0c"
    ),
}

# These are the staged files directly consumed by this worker. The progressive
# package has intentionally diverged elsewhere, so broader byte equality would
# be an incorrect source claim.
STAGED_EXECUTED_SOURCE_FILES = (
    "atomic_lines.py",
    "constants.py",
    "continuum.py",
    "device.py",
    "line_opacity.py",
)

FINGERPRINT_FIELDS = frozenset(
    {
        "meta__physical_payload_fingerprint",
        "meta__full_capture_fingerprint",
    }
)

# Frozen only after two fresh-process captures and the independent review in
# ``design/chapter06_synthesis_worker_independent_audit.md``.
ACCEPTED_CAPTURE_KEY_COUNT: int | None = 754
ACCEPTED_CAPTURE_SCHEMA_DIGEST: str | None = (
    "d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178"
)
CAPTURE_SCHEMA_VERSION = 1


class OracleIdentityError(RuntimeError):
    """Raised when an accepted source, fixture, subset, or table changed."""


class OracleEnvironmentError(RuntimeError):
    """Raised when fresh one-thread execution cannot be established."""


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_under(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    parent = root.resolve()
    return resolved == parent or parent in resolved.parents


def _require_regular_nonsymlink(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise OracleIdentityError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise OracleIdentityError(f"{label} is not a regular file: {resolved}")
    return resolved


def _canonical_input(path: Path, expected: Path, label: str) -> Path:
    """Resolve one input and reject every alternate or golden path."""

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


def array_mapping_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash sorted array names, dtypes, shapes, and contiguous bytes."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _expected_fixture_keys() -> set[str]:
    expected = set(FIXTURE_GLOBAL_FIELDS)
    for regime in REGIME_NAMES:
        expected.add(f"{regime}__effective_temperature")
        expected.update(f"{regime}__input__{name}" for name in FIXTURE_INPUT_FIELDS)
        expected.update(
            f"{regime}__atmosphere__{name}" for name in ATMOSPHERE_FIXTURE_FIELDS
        )
        expected.update(
            f"{regime}__synthesis__{name}" for name in SYNTHESIS_FIXTURE_FIELDS
        )
    return expected


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, np.ndarray]:
    """Load and exhaustively validate the accepted Chapter 5 input fixture."""

    resolved = _canonical_input(path, FIXTURE_PATH, "Chapter 5 fixture")
    actual_hash = sha256(resolved)
    if actual_hash != FIXTURE_SHA256:
        raise OracleIdentityError(
            f"fixture SHA-256 is {actual_hash}; expected {FIXTURE_SHA256}"
        )
    with np.load(resolved, allow_pickle=False) as archive:
        actual_keys = set(archive.files)
        expected_keys = _expected_fixture_keys()
        if actual_keys != expected_keys:
            raise OracleIdentityError(
                "fixture schema changed; "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if len(arrays) != FIXTURE_KEY_COUNT:
        raise OracleIdentityError("fixture key count changed")
    if array_mapping_digest(arrays) != FIXTURE_PAYLOAD_DIGEST:
        raise OracleIdentityError("fixture physical payload digest changed")
    if str(arrays["payne_zero_commit"]) != PINNED_COMMIT:
        raise OracleIdentityError("fixture commit changed")
    if tuple(str(value) for value in arrays["regime_names"].tolist()) != REGIME_NAMES:
        raise OracleIdentityError("fixture regime order changed")
    for regime in REGIME_NAMES:
        for field_name in SYNTHESIS_FIXTURE_FIELDS:
            values = arrays[f"{regime}__synthesis__{field_name}"]
            if values.dtype.hasobject:
                raise OracleIdentityError(
                    f"fixture object dtype is forbidden: {regime}/{field_name}"
                )
        for field_name in ("temperature", "mass_density", "electron_density"):
            values = arrays[f"{regime}__synthesis__{field_name}"]
            if values.shape != (6,) or values.dtype != np.float64:
                raise OracleIdentityError(
                    f"fixture field contract changed: {regime}/{field_name}"
                )
            if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
                raise OracleIdentityError(
                    f"fixture field is not finite/positive: {regime}/{field_name}"
                )
        populations = arrays[f"{regime}__synthesis__partition_normalized_populations"]
        widths = arrays[f"{regime}__synthesis__fractional_doppler_widths"]
        if populations.shape != (6, 6, 139) or populations.dtype != np.float64:
            raise OracleIdentityError("partition-normalized population shape changed")
        if widths.shape != (6, 6, 139) or widths.dtype != np.float64:
            raise OracleIdentityError("fractional Doppler-width shape changed")
    return arrays


def load_subset(path: Path = SUBSET_PATH) -> dict[str, np.ndarray]:
    """Load and validate the immutable raw Fe I source record."""

    resolved = _canonical_input(path, SUBSET_PATH, "Chapter 6 source subset")
    if resolved.stat().st_size != SUBSET_BYTES:
        raise OracleIdentityError("source subset byte size changed")
    actual_hash = sha256(resolved)
    if actual_hash != SUBSET_SHA256:
        raise OracleIdentityError(
            f"subset SHA-256 is {actual_hash}; expected {SUBSET_SHA256}"
        )
    expected = set(RAW_FIELDS) | set(SUBSET_PROVENANCE_FIELDS)
    with np.load(resolved, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise OracleIdentityError(
                "subset schema changed; "
                f"missing={sorted(expected - set(archive.files))}, "
                f"extra={sorted(set(archive.files) - expected)}"
            )
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if len(arrays) != SUBSET_KEY_COUNT:
        raise OracleIdentityError("source subset key count changed")
    for name in RAW_FIELDS:
        values = arrays[name]
        if values.shape != (1,) or values.dtype != RAW_FIELD_DTYPES[name]:
            raise OracleIdentityError(
                f"raw subset field changed: {name} "
                f"shape={values.shape}, dtype={values.dtype}"
            )
    expected_scalars = {
        "payne_zero_commit": PINNED_COMMIT,
        "source_archive_bytes": SOURCE_ARCHIVE_BYTES,
        "source_archive_relative_path": SOURCE_ARCHIVE_RELATIVE_PATH.as_posix(),
        "source_archive_row_count": SOURCE_ARCHIVE_ROW_COUNT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_field_count": len(RAW_FIELDS),
        "source_row_index": SOURCE_ROW_INDEX,
        "subset_schema_version": 1,
    }
    for name, expected_value in expected_scalars.items():
        if arrays[name].item() != expected_value:
            raise OracleIdentityError(f"subset provenance changed: {name}")
    if "no computed outputs" not in str(arrays["subset_role"]):
        raise OracleIdentityError("subset role no longer excludes computed output")
    return arrays


def verify_identity() -> dict[str, str]:
    """Fail closed on the checkout, sources, source archive, and tables."""

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
    identities = {"payne_zero_commit": commit}

    for relative_name, expected_hash in FROZEN_PINNED_PYTHON_MANIFEST.items():
        source_path = _require_regular_nonsymlink(
            root / relative_name, f"pinned source {relative_name}"
        )
        actual = sha256(source_path)
        if actual != expected_hash:
            raise OracleIdentityError(
                f"pinned source changed for {relative_name}: {actual}"
            )
        identities[f"source__{relative_name}__sha256"] = actual

    for filename in STAGED_EXECUTED_SOURCE_FILES:
        staged = _require_regular_nonsymlink(
            REPOSITORY_ROOT / "src/payne_zero_synthesis" / filename,
            f"staged source {filename}",
        )
        upstream = root / "payne_zero_synthesis" / filename
        if staged.read_bytes() != upstream.read_bytes():
            raise OracleIdentityError(
                f"staged source is not byte-identical to the pin: {filename}"
            )
        identities[f"staged_source__{filename}__sha256"] = sha256(staged)

    source_archive = _require_regular_nonsymlink(
        SOURCE_ARCHIVE_PATH,
        "oracle-provenance full atomic source archive "
        "(reader/runtime code uses only the compact subset)",
    )
    if source_archive.stat().st_size != SOURCE_ARCHIVE_BYTES:
        raise OracleIdentityError(
            "oracle-provenance full atomic source archive size changed"
        )
    source_archive_hash = sha256(source_archive)
    if source_archive_hash != SOURCE_ARCHIVE_SHA256:
        raise OracleIdentityError(
            "oracle-provenance full atomic source archive hash changed"
        )
    identities["source_archive_sha256"] = source_archive_hash

    for name, (staged, upstream, expected_hash) in STATIC_TABLE_IDENTITIES.items():
        staged_resolved = _require_regular_nonsymlink(staged, f"staged {name}")
        upstream_resolved = _require_regular_nonsymlink(upstream, f"upstream {name}")
        staged_hash = sha256(staged_resolved)
        upstream_hash = sha256(upstream_resolved)
        if staged_hash != expected_hash or upstream_hash != expected_hash:
            raise OracleIdentityError(
                f"{name} identity changed: staged={staged_hash}, "
                f"upstream={upstream_hash}"
            )
        if staged_resolved.read_bytes() != upstream_resolved.read_bytes():
            raise OracleIdentityError(f"{name} staged/upstream bytes differ")
        identities[f"table__{name}__sha256"] = expected_hash
    return identities


def require_environment() -> dict[str, np.ndarray]:
    """Require deterministic controls and an external empty cache directory."""

    wrong = {
        name: os.environ.get(name)
        for name, expected in ORACLE_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if wrong:
        raise OracleEnvironmentError(
            "fresh one-thread oracle controls are missing: "
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
    for forbidden_root in (PINNED_ROOT, REPOSITORY_ROOT, GOLDEN_ROOT):
        if _is_under(cache_path, forbidden_root):
            raise OracleEnvironmentError(
                "NUMBA_CACHE_DIR must be external to source and repository trees"
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
            "verified empty external nonsymlink directory"
        ),
        "environment__cpu_only": np.asarray(True, dtype=np.bool_),
        "environment__work_dtype": np.asarray("torch.float64"),
        "environment__accumulation_dtype": np.asarray("torch.float32"),
    }


def _loaded_pinned_python_manifest() -> dict[str, str]:
    """Return every loaded Python source below the pinned checkout."""

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


def _assert_loaded_manifest() -> dict[str, str]:
    loaded = _loaded_pinned_python_manifest()
    if loaded != FROZEN_PINNED_PYTHON_MANIFEST:
        expected = set(FROZEN_PINNED_PYTHON_MANIFEST)
        actual = set(loaded)
        changed = sorted(
            name
            for name in expected & actual
            if loaded[name] != FROZEN_PINNED_PYTHON_MANIFEST[name]
        )
        raise OracleIdentityError(
            "loaded pinned Python manifest changed; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}, changed={changed}"
        )
    return loaded


def load_pinned_modules() -> SimpleNamespace:
    """Import the pinned synthesis route and verify its complete module set."""

    configured_data_root = os.environ.get("PAYNE_ZERO_DATA_ROOT")
    if configured_data_root is not None:
        if (
            Path(configured_data_root).expanduser().resolve()
            != PINNED_DATA_ROOT.resolve()
        ):
            raise OracleIdentityError("PAYNE_ZERO_DATA_ROOT points outside the pin")
    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(PINNED_DATA_ROOT.resolve())

    for name, module in tuple(sys.modules.items()):
        if name != "payne_zero_synthesis" and not name.startswith(
            "payne_zero_synthesis."
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
    atomic_lines = importlib.import_module("payne_zero_synthesis.atomic_lines")
    continuum = importlib.import_module("payne_zero_synthesis.continuum")
    line_opacity = importlib.import_module("payne_zero_synthesis.line_opacity")
    import numba
    import torch

    for module in (atomic_lines, continuum, line_opacity):
        if not _is_under(Path(module.__file__), PINNED_ROOT):
            raise OracleIdentityError(
                f"synthesis module resolved outside the pin: {module.__file__}"
            )
    loaded_manifest = _assert_loaded_manifest()
    if int(numba.get_num_threads()) != 1:
        raise OracleEnvironmentError("Numba did not honor NUMBA_NUM_THREADS=1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        raise OracleEnvironmentError(
            "Torch interop threads were initialized before the oracle"
        ) from error
    torch.use_deterministic_algorithms(True)
    if int(torch.get_num_threads()) != 1:
        raise OracleEnvironmentError("Torch CPU thread count is not one")
    if int(torch.get_num_interop_threads()) != 1:
        raise OracleEnvironmentError("Torch CPU interop thread count is not one")
    if line_opacity.highp_dtype(torch.device("cpu")) != torch.float64:
        raise OracleEnvironmentError("canonical CPU work dtype is not float64")
    if line_opacity.ACCUMULATION_DTYPE != torch.float32:
        raise OracleEnvironmentError("canonical accumulation dtype is not float32")

    continuum_tables = continuum.ContinuumTables.from_npz(
        CONTINUUM_TABLES_PATH,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    return SimpleNamespace(
        atomic_lines=atomic_lines,
        continuum=continuum,
        continuum_tables=continuum_tables,
        line_opacity=line_opacity,
        numba=numba,
        torch=torch,
        loaded_python_manifest=loaded_manifest,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("oracle results forbid object dtype")
    return np.array(array, copy=True, order="C")


def _prefix(
    destination: dict[str, np.ndarray],
    prefix: str,
    arrays: Mapping[str, Any],
) -> None:
    for name, value in arrays.items():
        destination[f"{prefix}__{name}"] = _to_numpy(value)


def deterministic_result(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return a sorted detached object-free NumPy mapping."""

    result: dict[str, np.ndarray] = {}
    for name in sorted(arrays):
        if not isinstance(name, str) or not name:
            raise TypeError("oracle result names must be nonempty strings")
        result[name] = _to_numpy(arrays[name])
    return result


def _build_catalog(
    modules: SimpleNamespace,
    subset: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return exact derived records and the 18-entry constructor mapping."""

    raw = {name: np.asarray(subset[name]).copy() for name in RAW_FIELDS}
    records = modules.atomic_lines._build_records(raw)
    catalog = {name: np.asarray(records[name]).copy() for name in CATALOG_LINE_FIELDS}
    with np.load(LINE_PROFILE_TABLES_PATH, allow_pickle=False) as archive:
        for name in CATALOG_SUPPORT_FIELDS[-3:]:
            catalog[name] = np.asarray(archive[name]).copy()
    catalog["helium_line_type"] = np.empty(0, dtype=np.int64)
    catalog["helium_line_center_cutoff_ratio"] = np.asarray(
        modules.line_opacity.LINE_CENTER_CUTOFF_RATIO,
        dtype=np.float64,
    )
    if tuple(catalog) != CATALOG_LINE_FIELDS + (
        "harris_profile_h0_table",
        "harris_profile_h1_table",
        "harris_profile_h2_table",
        "helium_line_type",
        "helium_line_center_cutoff_ratio",
    ):
        raise RuntimeError("internal catalog construction order changed")
    if set(catalog) != set(CATALOG_LINE_FIELDS) | set(CATALOG_SUPPORT_FIELDS):
        raise RuntimeError("one-record catalog is not the exact 18-entry mapping")
    expected_record = {
        "line_type": 0,
        "atomic_number": 26,
        "ion_stage": 1,
        "wavelength_nm": 499.03411946178176,
        "index_wavelength_nm": 499.03411946178176,
        "oscillator_strength": 0.1380384264602885,
        "lower_excitation_cm": 33507.123,
        "radiative_damping": 3.909296919359072e-08,
        "stark_damping": 4.700008650504819e-21,
        "van_der_waals_damping": 4.093536407068315e-24,
    }
    for name, expected in expected_record.items():
        if catalog[name].shape != (1,) or catalog[name][0].item() != expected:
            raise RuntimeError(f"derived one-record mapping changed: {name}")
    if catalog["helium_line_type"].shape != (0,):
        raise RuntimeError("helium_line_type is not exactly empty")
    for name in CATALOG_SUPPORT_FIELDS[-3:]:
        if catalog[name].shape != (2001,) or catalog[name].dtype != np.float64:
            raise RuntimeError(f"synthesis Harris support changed: {name}")
    return records, catalog


def _build_grid(modules: SimpleNamespace, grid_name: str) -> np.ndarray:
    spec = GRID_SPECS[grid_name]
    wavelength = modules.atomic_lines.Grid(
        spec["start_wavelength_nm"],
        spec["end_wavelength_nm"],
        spec["resolution"],
    ).build()
    if wavelength.shape != (spec["count"],) or wavelength.dtype != np.float64:
        raise RuntimeError(f"{grid_name} wavelength-grid contract changed")
    if grid_name == "canonical":
        if wavelength[0] != spec["first_wavelength_nm"]:
            raise RuntimeError("canonical first wavelength changed")
        if wavelength[-1] != spec["last_wavelength_nm"]:
            raise RuntimeError("canonical last wavelength changed")
    return wavelength


def _validate_invariants(
    modules: SimpleNamespace,
    invariants: Any,
    grid_name: str,
) -> None:
    torch = modules.torch
    actual_fields = tuple(field.name for field in fields(invariants))
    if actual_fields != ATOMIC_INVARIANT_FIELDS:
        raise RuntimeError("AtomicInvariants field set/order changed")
    if invariants.wavelength_grid.dtype != torch.float64:
        raise RuntimeError("CPU invariant wavelength grid is not float64")
    if invariants.n_wavelengths != GRID_SPECS[grid_name]["count"]:
        raise RuntimeError("invariant wavelength count changed")
    for name in METAL_INVARIANT_FIELDS:
        value = getattr(invariants, name)
        if value.shape != (1,):
            raise RuntimeError(f"ordinary invariant shape changed: {name}")
    expected_float32 = {
        "metal_classical_strength",
        "metal_radiative_damping",
        "metal_stark_damping",
        "metal_van_der_waals_damping",
    }
    expected_int64 = {
        "metal_catalog_index",
        "metal_population_ion_stage_index",
        "metal_population_element_index",
        "metal_center_index",
        "metal_wing_index",
        "metal_center_clamped",
        "metal_wing_clamped",
    }
    for name in expected_float32:
        if getattr(invariants, name).dtype != torch.float32:
            raise RuntimeError(f"ordinary invariant dtype changed: {name}")
    for name in expected_int64:
        if getattr(invariants, name).dtype != torch.int64:
            raise RuntimeError(f"ordinary invariant dtype changed: {name}")
    if invariants.metal_lower_excitation_cm.dtype != torch.float64:
        raise RuntimeError("ordinary excitation invariant is not float64")
    if invariants.metal_wavelength_nm.dtype != torch.float64:
        raise RuntimeError("ordinary wavelength invariant is not float64")
    if int(invariants.metal_population_ion_stage_index[0]) != 0:
        raise RuntimeError("Fe I population-stage index changed")
    if int(invariants.metal_population_element_index[0]) != 25:
        raise RuntimeError("Fe population-element index changed")
    for name in AUTO_INVARIANT_FIELDS + HELIUM_ARRAY_INVARIANT_FIELDS:
        value = getattr(invariants, name)
        if value.shape != (0,):
            raise RuntimeError(f"empty invariant route changed: {name}")
    empty_int64 = {
        "auto_catalog_index",
        "auto_population_ion_stage_index",
        "auto_population_element_index",
        "auto_center_index",
        "auto_center_clamped",
        "helium_population_ion_stage_index",
        "helium_population_element_index",
        "helium_center_index",
        "helium_line_type",
    }
    empty_float32 = {
        "auto_oscillator_strength",
        "auto_radiative_damping",
        "auto_stark_damping",
        "auto_van_der_waals_damping",
        "helium_classical_strength",
        "helium_radiative_damping",
        "helium_stark_damping",
        "helium_van_der_waals_damping",
    }
    empty_float64 = (
        (set(AUTO_INVARIANT_FIELDS) | set(HELIUM_ARRAY_INVARIANT_FIELDS))
        - empty_int64
        - empty_float32
    )
    for name in empty_int64:
        if getattr(invariants, name).dtype != torch.int64:
            raise RuntimeError(f"empty invariant dtype changed: {name}")
    for name in empty_float32:
        if getattr(invariants, name).dtype != torch.float32:
            raise RuntimeError(f"empty invariant dtype changed: {name}")
    for name in empty_float64:
        if getattr(invariants, name).dtype != torch.float64:
            raise RuntimeError(f"empty invariant dtype changed: {name}")
    if invariants.helium_cutoff != 1.0e-3:
        raise RuntimeError("empty helium cutoff changed")
    for name in (
        "harris_profile_h0_table",
        "harris_profile_h1_table",
        "harris_profile_h2_table",
    ):
        value = getattr(invariants, name)
        if value.shape != (2001,) or value.dtype != torch.float64:
            raise RuntimeError(f"Harris invariant changed: {name}")
    for name in ("exponential_integer_table", "exponential_fraction_table"):
        value = getattr(invariants, name)
        if value.shape != (1001,) or value.dtype != torch.float64:
            raise RuntimeError(f"FASTEX invariant changed: {name}")


def _invariant_arrays(invariants: Any) -> dict[str, np.ndarray]:
    return {
        name: _to_numpy(getattr(invariants, name)) for name in ATOMIC_INVARIANT_FIELDS
    }


def _regime_state(
    fixture: Mapping[str, np.ndarray],
    regime: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    handoff = {
        name: np.asarray(fixture[f"{regime}__synthesis__{name}"]).copy()
        for name in SYNTHESIS_FIXTURE_FIELDS
    }
    continuum_state = {name: handoff[name].copy() for name in PIPELINE_CONTINUUM_FIELDS}
    collision_density_proxy = (
        handoff["hydrogen_neutral_population"]
        + 0.42 * handoff["helium_neutral_population"]
        + 0.85 * handoff["molecular_hydrogen_population"]
    ) * (handoff["temperature"] / 1.0e4) ** 0.3
    if collision_density_proxy.shape != (6,):
        raise RuntimeError("collision-density proxy shape changed")
    if collision_density_proxy.dtype != np.float64:
        raise RuntimeError("collision-density proxy must be NumPy float64")
    line_state = {name: handoff[name].copy() for name in LINE_STATE_FIELDS}
    line_state["collision_density_proxy"] = collision_density_proxy
    return continuum_state, line_state


def _evaluate_continuum(
    modules: SimpleNamespace,
    wavelength_nm: np.ndarray,
    state: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    absorption, scattering = modules.continuum.continuum(
        wavelength_nm,
        state,
        modules.continuum_tables,
    )
    absorption_host = _to_numpy(absorption)
    scattering_host = _to_numpy(scattering)
    total_float64 = absorption_host + scattering_host
    total_float32 = total_float64.astype(np.float32)
    expected_shape = (6, wavelength_nm.size)
    for name, values in (
        ("absorption", absorption_host),
        ("scattering", scattering_host),
        ("total", total_float64),
    ):
        if values.shape != expected_shape or values.dtype != np.float64:
            raise RuntimeError(f"continuum {name} contract changed")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise RuntimeError(f"continuum {name} is not finite/nonnegative")
    if total_float32.shape != expected_shape:
        raise RuntimeError("float32 continuum fence changed")
    return absorption_host, scattering_host, total_float64, total_float32


def _stimulated_factor(
    modules: SimpleNamespace,
    invariants: Any,
    temperature: np.ndarray,
) -> Any:
    torch = modules.torch
    frequency_grid_hz = (
        modules.line_opacity.LIGHT_SPEED_NM_PER_S / invariants.wavelength_grid
    ).to(torch.float32)
    temperature_t = torch.as_tensor(
        temperature,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    photon_temperature_factor = (
        modules.line_opacity.PLANCK_ERG_SECOND
        / (modules.line_opacity.BOLTZMANN_ERG_PER_K * temperature_t)
    ).to(torch.float32)
    return 1.0 - torch.exp(
        -frequency_grid_hz[None, :] * photon_temperature_factor[:, None]
    )


def _factor_ledger(
    modules: SimpleNamespace,
    invariants: Any,
    state: Mapping[str, np.ndarray],
    gross: Any,
    net: Any,
) -> dict[str, np.ndarray]:
    """Reconstruct the exact one-line factor, branch, and reach ledger."""

    torch = modules.torch
    line_opacity = modules.line_opacity
    work_dtype = torch.float64
    device = torch.device("cpu")
    partition = torch.as_tensor(
        state["partition_normalized_populations"],
        dtype=work_dtype,
        device=device,
    )
    widths = torch.as_tensor(
        state["fractional_doppler_widths"],
        dtype=work_dtype,
        device=device,
    )
    mass_density = torch.as_tensor(
        state["mass_density"], dtype=work_dtype, device=device
    )
    electron_density = torch.as_tensor(
        state["electron_density"], dtype=torch.float32, device=device
    )
    hc_over_kt = torch.as_tensor(state["hc_over_kt"], dtype=work_dtype, device=device)
    collision_proxy = torch.as_tensor(
        state["collision_density_proxy"], dtype=torch.float32, device=device
    )
    continuum = torch.as_tensor(
        state["continuum_opacity"], dtype=torch.float32, device=device
    )
    stage = int(invariants.metal_population_ion_stage_index[0])
    element = int(invariants.metal_population_element_index[0])
    selected_population = partition[:, stage, element][:, None]
    selected_width = widths[:, stage, element][:, None]
    mass_column = mass_density[:, None]
    valid = (selected_population > 0.0) & (selected_width > 0.0) & (mass_column > 0.0)
    width_safe = torch.where(
        selected_width > 0.0,
        selected_width,
        torch.ones_like(selected_width),
    )
    mass_safe = torch.where(
        mass_column > 0.0,
        mass_column,
        torch.ones_like(mass_column),
    )
    population_doppler_ratio = torch.where(
        valid,
        selected_population / (mass_safe * width_safe),
        torch.zeros_like(selected_population),
    )
    classical_strength = invariants.metal_classical_strength.to(work_dtype)[None, :]
    pre_excitation_strength = classical_strength * population_doppler_ratio
    excitation_exponent = (
        invariants.metal_lower_excitation_cm[None, :] * hc_over_kt[:, None]
    )
    fastex_weight = line_opacity.fast_ex(
        excitation_exponent,
        invariants.exponential_integer_table,
        invariants.exponential_fraction_table,
    )
    line_amplitude = pre_excitation_strength * fastex_weight
    center_index = int(invariants.metal_center_clamped[0])
    wing_index = int(invariants.metal_wing_clamped[0])
    center_cutoff = (
        continuum[:, invariants.metal_center_clamped].to(work_dtype)
        * line_opacity.LINE_CENTER_CUTOFF_RATIO
    )
    pre_cutoff_active = valid & (pre_excitation_strength >= center_cutoff)
    post_cutoff_active = (
        pre_cutoff_active & (line_amplitude >= center_cutoff) & (line_amplitude > 0.0)
    )
    radiative_term = (
        invariants.metal_radiative_damping[None, :].to(work_dtype).expand(6, -1)
    )
    stark_term = (
        invariants.metal_stark_damping[None, :].to(work_dtype)
        * electron_density.to(work_dtype)[:, None]
    )
    van_der_waals_term = (
        invariants.metal_van_der_waals_damping[None, :].to(work_dtype)
        * collision_proxy.to(work_dtype)[:, None]
    )
    total_damping = radiative_term + stark_term + van_der_waals_term
    damping_ratio = torch.where(
        selected_width > 0.0,
        total_damping / selected_width,
        torch.zeros_like(total_damping),
    )
    low_center_profile = 1.0 - 1.128 * damping_ratio
    full_center_profile = line_opacity.harris_profile_at_line_center(
        damping_ratio.to(torch.float32),
        invariants.harris_profile_h0_table,
        invariants.harris_profile_h1_table,
        invariants.harris_profile_h2_table,
    ).to(work_dtype)
    center_profile = torch.where(
        damping_ratio < 0.2,
        low_center_profile,
        full_center_profile,
    )
    center_deposits = (
        post_cutoff_active & (damping_ratio >= 0.0) & (line_amplitude > 0.0)
    )
    center_opacity_work = torch.where(
        center_deposits,
        line_amplitude * center_profile,
        torch.zeros_like(line_amplitude),
    )
    wing_damping = torch.clamp(damping_ratio, min=1.0e-12)
    wing_center_profile = line_opacity.harris_profile_at_line_center(
        wing_damping.to(torch.float32),
        invariants.harris_profile_h0_table,
        invariants.harris_profile_h1_table,
        invariants.harris_profile_h2_table,
    ).to(work_dtype)
    wing_profile_amplitude = torch.where(
        center_opacity_work > 0.0,
        center_opacity_work / wing_center_profile,
        torch.zeros_like(center_opacity_work),
    )
    wing_cutoff = torch.maximum(
        continuum[:, invariants.metal_wing_clamped].to(work_dtype)
        * line_opacity.LINE_CENTER_CUTOFF_RATIO,
        continuum[:, invariants.metal_wing_clamped].to(work_dtype)
        * line_opacity.WING_CUTOFF_FLOOR_RATIO,
    )
    doppler_width_nm = selected_width * invariants.metal_wavelength_nm[None, :]
    (
        reach,
        use_far_wing,
        ten_doppler_steps,
        doppler_offset_per_pixel,
        far_wing_coefficient,
    ) = line_opacity._wing_reach_batched(
        invariants,
        invariants.metal_wing_index,
        torch.where(
            center_deposits,
            wing_profile_amplitude,
            torch.zeros_like(wing_profile_amplitude),
        ),
        wing_damping,
        torch.where(
            center_deposits,
            doppler_width_nm,
            torch.zeros_like(doppler_width_nm),
        ),
        invariants.metal_wavelength_nm,
        wing_cutoff,
        center_deposits,
    )
    stimulation = _stimulated_factor(
        modules, invariants, np.asarray(state["temperature"])
    )
    reconstructed_net = gross * stimulation
    if not torch.equal(reconstructed_net, net):
        raise RuntimeError("net line slab is not one exact float32 stimulation")
    gross_center = gross[:, center_index]
    net_center = net[:, center_index]
    nonzero_count = torch.count_nonzero(gross > 0.0, dim=1)
    expected_nonzero = 2 * reach[:, 0] + center_deposits[:, 0].to(torch.int64)
    if not torch.equal(nonzero_count, expected_nonzero):
        raise RuntimeError("unclipped one-line reach/count identity changed")
    if not torch.equal(
        gross_center,
        center_opacity_work[:, 0].to(torch.float32),
    ):
        raise RuntimeError("gross center does not reconstruct the center deposit")
    if not torch.equal(
        center_deposits[:, 0],
        torch.any(gross > 0.0, dim=1),
    ):
        raise RuntimeError("center branch does not reconstruct activity")

    electron_perturbed_radiative = (
        invariants.metal_radiative_damping[None, :].to(work_dtype).expand(6, -1)
    )
    doubled_electron_stark = (
        invariants.metal_stark_damping[None, :].to(work_dtype)
        * (2.0 * electron_density).to(work_dtype)[:, None]
    )
    electron_perturbed_van_der_waals = (
        invariants.metal_van_der_waals_damping[None, :].to(work_dtype)
        * collision_proxy.to(work_dtype)[:, None]
    )
    collision_perturbed_radiative = (
        invariants.metal_radiative_damping[None, :].to(work_dtype).expand(6, -1)
    )
    collision_perturbed_stark = (
        invariants.metal_stark_damping[None, :].to(work_dtype)
        * electron_density.to(work_dtype)[:, None]
    )
    doubled_collision_vdw = (
        invariants.metal_van_der_waals_damping[None, :].to(work_dtype)
        * (2.0 * collision_proxy).to(work_dtype)[:, None]
    )
    electron_perturbed_total = (
        electron_perturbed_radiative
        + doubled_electron_stark
        + electron_perturbed_van_der_waals
    )
    collision_perturbed_total = (
        collision_perturbed_radiative
        + collision_perturbed_stark
        + doubled_collision_vdw
    )
    electron_total_delta = electron_perturbed_total - total_damping
    collision_total_delta = collision_perturbed_total - total_damping
    expected_stark_delta = doubled_electron_stark - stark_term
    expected_van_der_waals_delta = doubled_collision_vdw - van_der_waals_term
    electron_perturbation_isolated = (
        torch.equal(electron_perturbed_radiative, radiative_term)
        and torch.equal(electron_perturbed_van_der_waals, van_der_waals_term)
        and not torch.equal(doubled_electron_stark, stark_term)
    )
    collision_perturbation_isolated = (
        torch.equal(collision_perturbed_radiative, radiative_term)
        and torch.equal(collision_perturbed_stark, stark_term)
        and not torch.equal(doubled_collision_vdw, van_der_waals_term)
    )
    if not electron_perturbation_isolated:
        raise RuntimeError("electron perturbation is not isolated to Stark damping")
    if not collision_perturbation_isolated:
        raise RuntimeError(
            "collision perturbation is not isolated to van der Waals damping"
        )
    return {
        "selected_partition_normalized_population": selected_population[:, 0],
        "selected_fractional_doppler_width": selected_width[:, 0],
        "doppler_width_nm": doppler_width_nm[:, 0],
        "mass_density": mass_density,
        "population_over_mass_and_fractional_width": population_doppler_ratio[:, 0],
        "classical_strength_float32": invariants.metal_classical_strength,
        "pre_excitation_strength": pre_excitation_strength[:, 0],
        "lower_excitation_exponent": excitation_exponent[:, 0],
        "fastex_weight": fastex_weight[:, 0],
        "post_fastex_line_amplitude": line_amplitude[:, 0],
        "center_continuum_float32": continuum[:, center_index],
        "center_cutoff_work_float64": center_cutoff[:, 0],
        "pre_cutoff_active": pre_cutoff_active[:, 0],
        "post_cutoff_active": post_cutoff_active[:, 0],
        "radiative_damping_term": radiative_term[:, 0],
        "stark_damping_term": stark_term[:, 0],
        "van_der_waals_damping_term": van_der_waals_term[:, 0],
        "total_damping": total_damping[:, 0],
        "damping_ratio": damping_ratio[:, 0],
        "center_profile": center_profile[:, 0],
        "center_opacity_work_float64": center_opacity_work[:, 0],
        "wing_center_profile": wing_center_profile[:, 0],
        "wing_profile_amplitude": wing_profile_amplitude[:, 0],
        "wing_continuum_float32": continuum[:, wing_index],
        "wing_cutoff_work_float64": wing_cutoff[:, 0],
        "wing_reach": reach[:, 0],
        "use_far_wing": use_far_wing[:, 0],
        "ten_doppler_steps": ten_doppler_steps[:, 0],
        "doppler_offset_per_pixel": doppler_offset_per_pixel[:, 0],
        "far_wing_coefficient": far_wing_coefficient[:, 0],
        "gross_center_opacity_float32": gross_center,
        "net_center_opacity_float32": net_center,
        "stimulated_emission_factor_float32": stimulation,
        "stimulated_center_factor_float32": stimulation[:, center_index],
        "nonzero_count": nonzero_count,
        "electron_density_float32": electron_density,
        "collision_density_proxy_float32": collision_proxy,
        "doubled_electron_stark_term": doubled_electron_stark[:, 0],
        "doubled_collision_van_der_waals_term": doubled_collision_vdw[:, 0],
        "electron_perturbed_total_damping": electron_perturbed_total[:, 0],
        "collision_perturbed_total_damping": collision_perturbed_total[:, 0],
        "electron_total_damping_delta": electron_total_delta[:, 0],
        "collision_total_damping_delta": collision_total_delta[:, 0],
        "expected_stark_damping_delta": expected_stark_delta[:, 0],
        "expected_van_der_waals_damping_delta": (expected_van_der_waals_delta[:, 0]),
        "electron_perturbation_other_terms_unchanged": np.asarray(
            electron_perturbation_isolated,
            dtype=np.bool_,
        ),
        "collision_perturbation_other_terms_unchanged": np.asarray(
            collision_perturbation_isolated,
            dtype=np.bool_,
        ),
    }


def _capture_grid_regime(
    modules: SimpleNamespace,
    invariants: Any,
    wavelength_nm: np.ndarray,
    continuum_state: Mapping[str, np.ndarray],
    base_line_state: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    absorption, scattering, total_float64, total_float32 = _evaluate_continuum(
        modules, wavelength_nm, continuum_state
    )
    line_state = {
        name: np.asarray(value).copy() for name, value in base_line_state.items()
    }
    line_state["continuum_opacity"] = total_float64
    calls = {}
    for stimulation_name, apply_stim in (("gross", False), ("net", True)):
        for wing_mode in ("batched", "loop"):
            calls[f"{stimulation_name}_{wing_mode}"] = (
                modules.line_opacity.accumulate_atomic(
                    invariants,
                    line_state,
                    do_metal=True,
                    do_helium=False,
                    apply_stim=apply_stim,
                    wing_mode=wing_mode,
                )
            )
    for name, tensor in calls.items():
        if tensor.shape != (6, wavelength_nm.size):
            raise RuntimeError(f"line slab shape changed: {name}")
        if tensor.dtype != modules.torch.float32:
            raise RuntimeError(f"line slab dtype changed: {name}")
        if not bool(modules.torch.isfinite(tensor).all()):
            raise RuntimeError(f"line slab is nonfinite: {name}")
        if bool((tensor < 0.0).any()):
            raise RuntimeError(f"line slab is negative: {name}")
    for stimulation_name in ("gross", "net"):
        if not modules.torch.equal(
            calls[f"{stimulation_name}_batched"],
            calls[f"{stimulation_name}_loop"],
        ):
            raise RuntimeError(
                f"CPU loop/batched {stimulation_name} slabs are not bitwise equal"
            )
    ledger = _factor_ledger(
        modules,
        invariants,
        line_state,
        calls["gross_batched"],
        calls["net_batched"],
    )
    return {
        "continuum_absorption_float64": absorption,
        "continuum_scattering_float64": scattering,
        "continuum_total_float64": total_float64,
        "continuum_line_input_float32": total_float32,
        "gross_batched_float32": _to_numpy(calls["gross_batched"]),
        "net_batched_float32": _to_numpy(calls["net_batched"]),
        "gross_loop_float32": _to_numpy(calls["gross_loop"]),
        "net_loop_float32": _to_numpy(calls["net_loop"]),
        **{f"ledger__{name}": value for name, value in ledger.items()},
    }


def _capture_schema_digest(results: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(results):
        array = np.asarray(results[name])
        if array.dtype.hasobject:
            raise RuntimeError(f"capture schema forbids object dtype: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _fingerprint(
    results: Mapping[str, np.ndarray],
    *,
    physical_payload_only: bool,
) -> str:
    digest = hashlib.sha256()
    for name in sorted(results):
        if name in FINGERPRINT_FIELDS:
            continue
        if physical_payload_only and name.startswith(("meta__", "identity__")):
            continue
        array = np.asarray(results[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validate_complete_result(results: Mapping[str, np.ndarray]) -> None:
    """Validate cross-regime activity, dtype, and route contracts."""

    canonical_activity = np.stack(
        [
            results[f"{regime}__canonical__ledger__post_cutoff_active"]
            for regime in REGIME_NAMES
        ]
    )
    coarse_activity = np.stack(
        [
            results[f"{regime}__coarse__ledger__post_cutoff_active"]
            for regime in REGIME_NAMES
        ]
    )
    if not np.array_equal(canonical_activity, EXPECTED_ACTIVITY_MASK):
        raise RuntimeError("canonical four-regime activity mask changed")
    if not np.array_equal(coarse_activity, EXPECTED_ACTIVITY_MASK):
        raise RuntimeError("coarse-grid four-regime activity mask changed")
    for regime in REGIME_NAMES:
        for grid_name, spec in GRID_SPECS.items():
            shape = (6, spec["count"])
            for name in (
                "gross_batched_float32",
                "net_batched_float32",
                "gross_loop_float32",
                "net_loop_float32",
            ):
                array = results[f"{regime}__{grid_name}__{name}"]
                if array.shape != shape or array.dtype != np.float32:
                    raise RuntimeError(
                        f"captured line-array contract changed: "
                        f"{regime}/{grid_name}/{name}"
                    )
            if not np.array_equal(
                results[f"{regime}__{grid_name}__gross_batched_float32"],
                results[f"{regime}__{grid_name}__gross_loop_float32"],
            ):
                raise RuntimeError("captured gross loop/batched mismatch")
            if not np.array_equal(
                results[f"{regime}__{grid_name}__net_batched_float32"],
                results[f"{regime}__{grid_name}__net_loop_float32"],
            ):
                raise RuntimeError("captured net loop/batched mismatch")
            stimulation = results[
                f"{regime}__{grid_name}__ledger__stimulated_emission_factor_float32"
            ]
            if stimulation.shape != shape or stimulation.dtype != np.float32:
                raise RuntimeError(
                    "captured stimulated-emission factor contract changed"
                )
            if (
                np.any(~np.isfinite(stimulation))
                or np.any(stimulation <= 0.0)
                or np.any(stimulation > 1.0)
            ):
                raise RuntimeError(
                    "captured stimulated-emission factor is outside (0, 1]"
                )
            for wing_mode in ("batched", "loop"):
                gross = results[f"{regime}__{grid_name}__gross_{wing_mode}_float32"]
                net = results[f"{regime}__{grid_name}__net_{wing_mode}_float32"]
                reconstructed = np.multiply(gross, stimulation, dtype=np.float32)
                if not np.array_equal(reconstructed, net):
                    raise RuntimeError(
                        "captured net slab is not exactly gross times the full "
                        f"stimulated factor: {regime}/{grid_name}/{wing_mode}"
                    )
    for name, value in results.items():
        if np.asarray(value).dtype.hasobject:
            raise RuntimeError(f"capture contains object dtype: {name}")


def build_oracle_results(
    *,
    fixture_path: Path = FIXTURE_PATH,
    subset_path: Path = SUBSET_PATH,
) -> dict[str, np.ndarray]:
    """Execute the complete in-memory Chapter 6 synthesis observation."""

    # Every file and process check precedes importing pinned Payne Zero.
    identities = verify_identity()
    fixture = load_fixture(fixture_path)
    subset = load_subset(subset_path)
    environment = require_environment()
    modules = load_pinned_modules()
    records, catalog = _build_catalog(modules, subset)

    grids = {grid_name: _build_grid(modules, grid_name) for grid_name in GRID_SPECS}
    invariants = {
        grid_name: modules.line_opacity.precompute_invariants(
            catalog,
            wavelength,
            runtime_device=modules.torch.device("cpu"),
        )
        for grid_name, wavelength in grids.items()
    }
    for grid_name, invariant_block in invariants.items():
        _validate_invariants(modules, invariant_block, grid_name)

    canonical_center = int(invariants["canonical"].metal_center_index[0])
    if canonical_center != GRID_SPECS["canonical"]["center_index"]:
        raise RuntimeError("canonical mapped line-center index changed")
    if (
        grids["canonical"][canonical_center]
        != GRID_SPECS["canonical"]["center_wavelength_nm"]
    ):
        raise RuntimeError("canonical mapped center wavelength changed")
    coarse_center = int(invariants["coarse"].metal_center_index[0])
    if grids["coarse"][coarse_center] != GRID_SPECS["coarse"]["center_wavelength_nm"]:
        raise RuntimeError("coarse mapped center wavelength changed")

    results: dict[str, Any] = {
        "meta__capture_scope_complete": np.asarray(False, dtype=np.bool_),
        "meta__capture_scope": np.asarray(
            "candidate complete; independent schema review pending",
            dtype="<U128",
        ),
        "meta__golden_publication_performed": np.asarray(False, dtype=np.bool_),
        "meta__golden_read_performed": np.asarray(False, dtype=np.bool_),
        "meta__capture_schema_version": np.asarray(
            CAPTURE_SCHEMA_VERSION, dtype=np.int64
        ),
        "meta__fixture_path": np.asarray(str(FIXTURE_PATH.resolve())),
        "meta__fixture_sha256": np.asarray(FIXTURE_SHA256),
        "meta__fixture_key_count": np.asarray(FIXTURE_KEY_COUNT, dtype=np.int64),
        "meta__fixture_payload_digest": np.asarray(FIXTURE_PAYLOAD_DIGEST),
        "meta__subset_path": np.asarray(str(SUBSET_PATH.resolve())),
        "meta__subset_sha256": np.asarray(SUBSET_SHA256),
        "meta__subset_key_count": np.asarray(SUBSET_KEY_COUNT, dtype=np.int64),
        "meta__source_row_index": np.asarray(SOURCE_ROW_INDEX, dtype=np.int64),
        "meta__platform": np.asarray(platform.platform()),
        "meta__architecture": np.asarray(platform.machine()),
        "meta__processor": np.asarray(platform.processor()),
        "meta__system_byteorder": np.asarray(sys.byteorder),
        "meta__python_version": np.asarray(platform.python_version()),
        "meta__numpy_version": np.asarray(np.__version__),
        "meta__torch_version": np.asarray(modules.torch.__version__),
        "meta__numba_version": np.asarray(modules.numba.__version__),
        "meta__torch_cpu_thread_count": np.asarray(
            modules.torch.get_num_threads(), dtype=np.int64
        ),
        "meta__torch_cpu_interop_thread_count": np.asarray(
            modules.torch.get_num_interop_threads(), dtype=np.int64
        ),
        "meta__numba_thread_count": np.asarray(
            modules.numba.get_num_threads(), dtype=np.int64
        ),
        "meta__torch_deterministic_algorithms": np.asarray(
            modules.torch.are_deterministic_algorithms_enabled(),
            dtype=np.bool_,
        ),
        "meta__pinned_data_root": np.asarray(str(PINNED_DATA_ROOT.resolve())),
        "meta__regime_names": np.asarray(REGIME_NAMES),
        "meta__pipeline_continuum_fields": np.asarray(PIPELINE_CONTINUUM_FIELDS),
        "meta__catalog_line_fields": np.asarray(CATALOG_LINE_FIELDS),
        "meta__catalog_support_fields": np.asarray(CATALOG_SUPPORT_FIELDS),
        "meta__loaded_pinned_python_source_count": np.asarray(
            len(modules.loaded_python_manifest), dtype=np.int64
        ),
        "meta__loaded_pinned_python_manifest_digest": np.asarray(
            hashlib.sha256(
                json.dumps(
                    modules.loaded_python_manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        ),
        "meta__worker_sha256": np.asarray(sha256(WORKER_PATH)),
        "meta__design_sha256": np.asarray(sha256(DESIGN_PATH)),
        "meta__source_contract_sha256": np.asarray(sha256(SOURCE_CONTRACT_PATH)),
        "fixture_payload__field_names": np.asarray(sorted(fixture)),
        "fixture_payload__content_digest": np.asarray(array_mapping_digest(fixture)),
        "subset_payload__field_names": np.asarray(sorted(subset)),
        "subset_payload__content_digest": np.asarray(array_mapping_digest(subset)),
        "record__field_names": np.asarray(CATALOG_LINE_FIELDS),
    }
    for name, value in identities.items():
        results[f"identity__{name}"] = np.asarray(value)
    _prefix(results, "meta", environment)
    _prefix(
        results,
        "record",
        {name: records[name] for name in CATALOG_LINE_FIELDS},
    )
    _prefix(results, "catalog", catalog)
    for grid_name, wavelength in grids.items():
        results[f"grid__{grid_name}__wavelength_nm"] = wavelength
        results[f"grid__{grid_name}__sha256"] = np.asarray(
            hashlib.sha256(np.ascontiguousarray(wavelength).tobytes()).hexdigest()
        )
        results[f"grid__{grid_name}__line_center_index"] = np.asarray(
            int(invariants[grid_name].metal_center_index[0]), dtype=np.int64
        )
        results[f"grid__{grid_name}__line_wing_index"] = np.asarray(
            int(invariants[grid_name].metal_wing_index[0]), dtype=np.int64
        )
        _prefix(
            results,
            f"invariant__{grid_name}",
            _invariant_arrays(invariants[grid_name]),
        )

    for regime in REGIME_NAMES:
        continuum_state, line_state = _regime_state(fixture, regime)
        _prefix(results, f"{regime}__continuum_state", continuum_state)
        _prefix(results, f"{regime}__line_state", line_state)
        for grid_name, wavelength in grids.items():
            capture = _capture_grid_regime(
                modules,
                invariants[grid_name],
                wavelength,
                continuum_state,
                line_state,
            )
            _prefix(results, f"{regime}__{grid_name}", capture)

    post_lane_manifest = _assert_loaded_manifest()
    results["meta__post_lane_loaded_manifest_verified"] = np.asarray(
        True, dtype=np.bool_
    )
    results["meta__post_lane_loaded_source_count"] = np.asarray(
        len(post_lane_manifest), dtype=np.int64
    )
    results["meta__post_lane_loaded_manifest_digest"] = np.asarray(
        hashlib.sha256(
            json.dumps(
                post_lane_manifest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    results["meta__capture_schema_digest"] = np.asarray("0" * 64)
    results["meta__physical_payload_fingerprint"] = np.asarray("0" * 64)
    results["meta__full_capture_fingerprint"] = np.asarray("0" * 64)
    results = deterministic_result(results)
    _validate_complete_result(results)

    schema_digest = _capture_schema_digest(results)
    results["meta__capture_schema_digest"] = np.asarray(schema_digest)
    if (
        ACCEPTED_CAPTURE_KEY_COUNT is not None
        and ACCEPTED_CAPTURE_SCHEMA_DIGEST is not None
    ):
        if len(results) != ACCEPTED_CAPTURE_KEY_COUNT:
            raise RuntimeError(
                f"capture key count is {len(results)}, "
                f"expected {ACCEPTED_CAPTURE_KEY_COUNT}"
            )
        if schema_digest != ACCEPTED_CAPTURE_SCHEMA_DIGEST:
            raise RuntimeError(
                f"capture schema digest is {schema_digest}, "
                f"expected {ACCEPTED_CAPTURE_SCHEMA_DIGEST}"
            )
        results["meta__capture_scope_complete"] = np.asarray(True, dtype=np.bool_)
        results["meta__capture_scope"] = np.asarray(
            "accepted exhaustive Chapter 6 CPU one-line synthesis capture",
            dtype="<U128",
        )
    results["meta__physical_payload_fingerprint"] = np.asarray(
        _fingerprint(results, physical_payload_only=True)
    )
    results["meta__full_capture_fingerprint"] = np.asarray(
        _fingerprint(results, physical_payload_only=False)
    )
    if str(results["meta__physical_payload_fingerprint"]) != _fingerprint(
        results, physical_payload_only=True
    ):
        raise RuntimeError("physical-payload fingerprint is inconsistent")
    if str(results["meta__full_capture_fingerprint"]) != _fingerprint(
        results, physical_payload_only=False
    ):
        raise RuntimeError("full-capture fingerprint is inconsistent")
    return deterministic_result(results)


def summarize(results: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Return a compact JSON-safe candidate report without serialization."""

    canonical_activity = np.stack(
        [
            results[f"{regime}__canonical__ledger__post_cutoff_active"]
            for regime in REGIME_NAMES
        ]
    )
    coarse_activity = np.stack(
        [
            results[f"{regime}__coarse__ledger__post_cutoff_active"]
            for regime in REGIME_NAMES
        ]
    )
    canonical_reach = np.stack(
        [results[f"{regime}__canonical__ledger__wing_reach"] for regime in REGIME_NAMES]
    )
    loop_batched_max_abs = max(
        float(
            np.max(
                np.abs(
                    results[f"{regime}__{grid_name}__gross_batched_float32"]
                    - results[f"{regime}__{grid_name}__gross_loop_float32"]
                )
            )
        )
        for regime in REGIME_NAMES
        for grid_name in GRID_SPECS
    )
    stimulation_members = [
        results[f"{regime}__{grid_name}__ledger__stimulated_emission_factor_float32"]
        for regime in REGIME_NAMES
        for grid_name in GRID_SPECS
    ]
    batched_stimulation_reconstructs = all(
        np.array_equal(
            np.multiply(
                results[f"{regime}__{grid_name}__gross_batched_float32"],
                results[
                    f"{regime}__{grid_name}__ledger__stimulated_emission_factor_float32"
                ],
                dtype=np.float32,
            ),
            results[f"{regime}__{grid_name}__net_batched_float32"],
        )
        for regime in REGIME_NAMES
        for grid_name in GRID_SPECS
    )
    loop_stimulation_reconstructs = all(
        np.array_equal(
            np.multiply(
                results[f"{regime}__{grid_name}__gross_loop_float32"],
                results[
                    f"{regime}__{grid_name}__ledger__stimulated_emission_factor_float32"
                ],
                dtype=np.float32,
            ),
            results[f"{regime}__{grid_name}__net_loop_float32"],
        )
        for regime in REGIME_NAMES
        for grid_name in GRID_SPECS
    )
    return {
        "capture_scope_complete": bool(results["meta__capture_scope_complete"]),
        "key_count": len(results),
        "capture_schema_digest": str(results["meta__capture_schema_digest"]),
        "physical_payload_fingerprint": str(
            results["meta__physical_payload_fingerprint"]
        ),
        "full_capture_fingerprint": str(results["meta__full_capture_fingerprint"]),
        "worker_sha256": str(results["meta__worker_sha256"]),
        "fixture_sha256": str(results["meta__fixture_sha256"]),
        "subset_sha256": str(results["meta__subset_sha256"]),
        "loaded_pinned_python_source_count": int(
            results["meta__loaded_pinned_python_source_count"]
        ),
        "canonical_grid_count": int(results["grid__canonical__wavelength_nm"].size),
        "coarse_grid_count": int(results["grid__coarse__wavelength_nm"].size),
        "canonical_activity_mask": canonical_activity.tolist(),
        "coarse_activity_mask": coarse_activity.tolist(),
        "canonical_reach_minimum": int(canonical_reach[canonical_activity].min()),
        "canonical_reach_maximum": int(canonical_reach[canonical_activity].max()),
        "maximum_loop_batched_absolute_difference": loop_batched_max_abs,
        "full_stimulated_factor_member_count": len(stimulation_members),
        "full_stimulated_factor_minimum": min(
            float(values.min()) for values in stimulation_members
        ),
        "full_stimulated_factor_maximum": max(
            float(values.max()) for values in stimulation_members
        ),
        "batched_net_reconstructed_from_full_stimulated_factor": (
            batched_stimulation_reconstructs
        ),
        "loop_net_reconstructed_from_full_stimulated_factor": (
            loop_stimulation_reconstructs
        ),
        "golden_read_performed": bool(results["meta__golden_read_performed"]),
        "golden_publication_performed": bool(
            results["meta__golden_publication_performed"]
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse only canonical input checks; no output path is accepted."""

    parser = argparse.ArgumentParser(
        description="self-check the unpublished Chapter 6 synthesis oracle"
    )
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--subset", type=Path, default=SUBSET_PATH)
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="validate inputs and identities without importing synthesis code",
    )
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_args()
    if arguments.identity_only:
        identities = verify_identity()
        fixture = load_fixture(arguments.fixture)
        subset = load_subset(arguments.subset)
        print(
            json.dumps(
                {
                    "commit": identities["payne_zero_commit"],
                    "fixture_key_count": len(fixture),
                    "fixture_sha256": FIXTURE_SHA256,
                    "subset_key_count": len(subset),
                    "subset_sha256": SUBSET_SHA256,
                },
                sort_keys=True,
            )
        )
        return
    results = build_oracle_results(
        fixture_path=arguments.fixture,
        subset_path=arguments.subset,
    )
    print(json.dumps(summarize(results), sort_keys=True))


if __name__ == "__main__":
    main()
