#!/usr/bin/env python3
"""Build reviewed Chapter 5 comparison archives without implicit publication.

The scientific worker remains an in-memory observer.  This publisher launches
it twice in isolated CPU-float64 children, compares deterministic temporary raw
captures, independently assembles a compact reader archive and a deduplicated
integration archive, and publishes only after a separate explicit ``--publish``
request.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

CANONICAL_REPOSITORY_ROOT = Path(
    "/Users/ysting/stellar-spectroscopy-from-scratch-gpu"
)
PUBLISHER_RELATIVE_PATH = "scripts/build_chapter05_payne_zero_goldens.py"
PUBLISHER_PATH = CANONICAL_REPOSITORY_ROOT / PUBLISHER_RELATIVE_PATH
try:
    EXECUTING_PUBLISHER_PATH = Path(__file__).resolve(strict=True)
except (FileNotFoundError, OSError) as error:
    raise RuntimeError(
        "executing Chapter 5 publisher path is absent or unreadable"
    ) from error
if EXECUTING_PUBLISHER_PATH != PUBLISHER_PATH:
    raise RuntimeError(
        "executing Chapter 5 publisher is not the reviewed canonical path"
    )

REPOSITORY_ROOT = CANONICAL_REPOSITORY_ROOT
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402

from scripts import chapter05_oracle_worker as oracle_worker  # noqa: E402
from scripts.deterministic_npz import write_npz  # noqa: E402


PINNED_ROOT = Path("/Users/ysting/payne-zero")
PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
PINNED_DATA_ROOT = PINNED_ROOT / "source_data_files"

WORKER_PATH = REPOSITORY_ROOT / "scripts" / "chapter05_oracle_worker.py"
CAPTURE_CONTRACT_PATH = (
    REPOSITORY_ROOT / "design" / "chapter05_oracle_capture_contract.md"
)
EXACT_SOURCE_CONTRACT_PATH = (
    REPOSITORY_ROOT / "design" / "chapter05_exact_source_contract.md"
)
PUBLISHER_CONTRACT_PATH = (
    REPOSITORY_ROOT / "design" / "chapter05_publisher_contract.md"
)
ACCEPTANCE_PATH = (
    REPOSITORY_ROOT / "design" / "chapter05_oracle_worker_acceptance.md"
)
DETERMINISTIC_WRITER_PATH = REPOSITORY_ROOT / "scripts" / "deterministic_npz.py"
FIXTURE_PATH = (
    REPOSITORY_ROOT / "data" / "fixtures" / "chapter05_continuum_states.npz"
)
OUTPUT_DIR = REPOSITORY_ROOT / "data" / "golden" / "payne_zero" / "chapter05"
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "MANIFEST.json"
PUBLICATION_ACCEPTANCE_RECORD_PATH = (
    REPOSITORY_ROOT / "design" / "chapter05_publication_acceptance.json"
)

READER_NAME = "chapter05_continuum_reader_cpu_float64.npz"
INTEGRATION_NAME = "chapter05_continuum_integration_cpu_float64.npz"
OUTPUT_NAMES = (READER_NAME, INTEGRATION_NAME)
RAW_NAME = "chapter05_raw_capture_cpu_float64.npz"
PUBLISHER_CONTRACT_RELATIVE_PATH = "design/chapter05_publisher_contract.md"
MANIFEST_RELATIVE_PATH = "data/MANIFEST.json"
OUTPUT_RELATIVE_PATHS = {
    READER_NAME: f"data/golden/payne_zero/chapter05/{READER_NAME}",
    INTEGRATION_NAME: f"data/golden/payne_zero/chapter05/{INTEGRATION_NAME}",
}

ACCEPTED_INPUT_SHA256 = {
    "worker": "429252d5fefd2b911ce4321578820aa67b505d5fe37b174cb647d4b6177d7389",
    "capture_contract": (
        "4198f76419f102efbb3468b5f2ed7ddca7ff6af776ffc566cfa4058b6164fdaf"
    ),
    "exact_source_contract": (
        "ec1c84519a898a454a408780bd6b36fa723fc7bade83a98e11eede1343bf2956"
    ),
    "publisher_contract": (
        "8a58866b315b2cdd1ca769436aa0ccdbed3b08e4ef180bd309179d6cc7e04c08"
    ),
    "oracle_acceptance": (
        "892279cd2dfa850c3eabfb8ce94953e4723db5270da319bcedb9bc90c9597474"
    ),
    "deterministic_npz": (
        "f4886766524d79623e648d28ab9d24215da42f8bf1f859f69381c546f9c96e49"
    ),
    "fixture": "ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae",
}
PUBLISHER_INPUT_PATHS = {
    "worker": WORKER_PATH,
    "capture_contract": CAPTURE_CONTRACT_PATH,
    "exact_source_contract": EXACT_SOURCE_CONTRACT_PATH,
    "publisher_contract": PUBLISHER_CONTRACT_PATH,
    "oracle_acceptance": ACCEPTANCE_PATH,
    "deterministic_npz": DETERMINISTIC_WRITER_PATH,
    "fixture": FIXTURE_PATH,
}

STAGED_SOURCE_HASHES = {
    REPOSITORY_ROOT / "src" / "payne_zero_atmosphere" / "continuum_opacity.py": (
        "1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81"
    ),
    REPOSITORY_ROOT / "src" / "payne_zero_synthesis" / "continuum.py": (
        "ab0d4eb771ee04101f6936253f633ed60d845e2816854a06b1b059e8b91dce1b"
    ),
}
STAGED_STATIC_HASHES = {
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "atmosphere_tables"
    / "continuum_opacity_tables.npz": (
        "6fd4c556418870c28d3fcc9a050252af58ac4cc433cae979477355c8c7d593e3"
    ),
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "atmosphere_tables"
    / "karzas_latter_tables.npz": (
        "23805dc17c47af45b8ae63b2e278e1fb6c584a01c87d1eb3c31306e4555e6d15"
    ),
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "atmosphere_tables"
    / "molecular_equilibrium_tables.npz": (
        "1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe"
    ),
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "synthesis_tables"
    / "continuum_tables.npz": (
        "406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d"
    ),
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "synthesis_tables"
    / "continuum_edge_grid.npz": (
        "11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e"
    ),
}

ACCEPTED_RAW_SCHEMA_VERSION = 2
ACCEPTED_RAW_KEY_COUNT = 1161
ACCEPTED_RAW_SCHEMA_DIGEST = (
    "652c110dc79a6f6dfca6893bee35416289675b4920a5d0dcfe6b2cb262dacf3d"
)
ACCEPTED_FIXTURE_PAYLOAD_DIGEST = (
    "4bcf0bbd8d61e58334c4c7ef6caaaf9ca47e6fb4536ad0098d5a541d540ec048"
)
ACCEPTED_PHYSICAL_FINGERPRINT = (
    "d223351fa2c51dc24a1b01896da9ab9a82fc475f4082c47fde34734d8dc03343"
)
ACCEPTED_FULL_FINGERPRINT = (
    "3d2c131711e1c0dc6aa088892193bb24d41a76d005bc20dd1c42d3e84f66e656"
)

# Frozen from the first real accepted-raw assembly.  The independent publisher
# review still decides whether these candidate final schemas may be published.
ACCEPTED_READER_KEY_COUNT: int | None = 257
ACCEPTED_READER_SCHEMA_DIGEST: str | None = (
    "058389fcd0e944dd4c1ad1208adbaac44d53ab35e61e5366d37a8a141ad91f88"
)
ACCEPTED_INTEGRATION_KEY_COUNT: int | None = 1079
ACCEPTED_INTEGRATION_SCHEMA_DIGEST: str | None = (
    "e09a8932d97f8a2756aca2b779c4b8fdd822ca239197fd908ee575d09df081ca"
)
ACCEPTED_INVENTORY_MAPPING_DIGEST = (
    "b02a6a2896d3468d1052441f8def0841b608eb5d5816ccd72539a0ec982c452c"
)

# Publication deliberately remains unavailable until a later independent
# review installs the strict external record at the one canonical path above.
# The publisher never pins that record's hash and never embeds it in an
# artifact, avoiding a publisher-hash/artifact-hash/record-hash cycle.
PUBLICATION_ACCEPTANCE_SCHEMA_VERSION = 1
PUBLICATION_ACCEPTANCE_KIND = "chapter05_publication_acceptance"

READER_SIZE_LIMIT_BYTES = 4 * 1024 * 1024
INTEGRATION_SIZE_LIMIT_BYTES = 32 * 1024 * 1024
LIGHT_SPEED_NM_PER_S = np.float64(2.99792458e17)

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

REGIME_NAMES = tuple(oracle_worker.REGIME_NAMES)
ATMOSPHERE_ABSORPTION_COMPONENTS = tuple(
    name for name, _ in oracle_worker.ATMOSPHERE_COMPONENT_ROUTINES
)
ATMOSPHERE_SCATTERING_COMPONENTS = (
    "electron",
    "hydrogen_rayleigh",
    "helium_rayleigh",
    "h2_rayleigh",
)
ATMOSPHERE_MOLECULAR_COMPONENTS = ("ch", "oh", "h2_cia")
SYNTHESIS_COMPONENT_NAMES = (
    "hminus_absorption",
    "hydrogen_absorption",
    "minor_absorption",
    "helium_neutral_absorption",
    "helium_ionized_absorption",
    "hot_metal_absorption",
    "light_element_and_si2_absorption",
    "hydrogen_rayleigh_scattering",
    "electron_scattering",
    "minor_scattering",
)
SYNTHESIS_MINOR_CASES = (
    "h2plus",
    "heminus_and_helium_rayleigh",
    "compact_carbon",
    "compact_magnesium",
    "compact_aluminum",
    "compact_silicon",
    "h2_rayleigh",
)
READER_STATE_FIELDS = (
    "temperature",
    "mass_density",
    "electron_density",
    "hydrogen_partition_normalized_ion_stage_populations",
    "hydrogen_ionized_population",
    "helium_neutral_partition_normalized_population",
    "helium_singly_ionized_partition_normalized_population",
    "hot_metal_populations",
    "charge_square_population_sum",
)
EXTENSION_INVARIANT_FIELDS = (
    "frequencies_hz",
    "coulomb_table_energy_first",
    "natural_log_frequency",
    "hminus_freefree_rows",
    "hminus_boundfree_cross_section_cm2",
    "silicon_singly_ionized_peach_frequency_rows",
    "silicon_singly_ionized_peach_natural_log_temperature_grid",
    "rayleigh_factor",
    "hydrogen_high_level_photoionization_cross_sections",
    "hydrogen_low_level_photoionization_cross_sections",
    "hydrogen_ground_level_photoionization_cross_section",
    "hydrogen_tail_edge",
    "neutral_helium_low_level_photoionization_cross_sections",
    "neutral_helium_high_level_photoionization_cross_sections",
    "nitrogen_edge_cross_sections",
    "oxygen_911_cross_section",
    "calcium_edge_cross_sections",
    "magnesium_ionized_cross_section_rows",
    "carbon_boundfree_cross_section_rows",
    "carbon_freefree_prefactor",
    "carbon_freefree_threshold",
    "magnesium_boundfree_cross_section_rows",
    "magnesium_freefree_prefactor",
    "magnesium_freefree_threshold",
    "silicon_boundfree_cross_section_rows",
    "silicon_freefree_prefactor",
    "silicon_freefree_threshold",
    "aluminum_boundfree_cross_section",
    "iron_boundfree_cross_section_rows",
)
STANDARD_TRACE_FIELDS = (
    "called_frequency_hz",
    "compute_at_freqs_call_count",
    "unused_edge_mask",
    "unused_sample_call_count",
    "unused_sample_index",
    "unused_sample_was_called",
)
COUNTERFACTUAL_STANDARD_OUTPUT_ALIASES = (
    ("rich_hii_absorption", "absorption"),
    ("rich_hii_scattering", "scattering"),
    ("rich_hii_perturbed_scattering", "scattering"),
    ("schema_h2_original_absorption", "absorption"),
    ("schema_h2_original_scattering", "scattering"),
    ("schema_h2_perturbed_absorption", "absorption"),
    ("schema_h2_perturbed_scattering", "scattering"),
    ("signed_edge_flipped_absorption", "absorption"),
    ("signed_edge_flipped_scattering", "scattering"),
)
READER_MOLECULAR_BOUNDARY_FIELDS = frozenset(
    {
        "ch_oh_frequency_hz",
        "ch_oh_frequency_family_index",
        "ch_oh_temperature_k",
        "ch_cross_section_times_partition",
        "oh_cross_section_times_partition",
        "cia_wavenumber_cm_inverse",
        "cia_frequency_hz",
        "cia_absorption",
        "cia_temperature_fraction",
        "cia_lower_column_weight",
        "cia_upper_column_weight",
        "cia_active_frequency_mask",
    }
)
GRID_POLICY_LABELS = (
    "teff_below_4500_k",
    "teff_4500_to_7249_k",
    "teff_7250_to_12999_k",
    "teff_13000_to_29999_k",
    "teff_30000_k_and_above",
)

FINGERPRINT_FIELDS = frozenset(
    {
        "meta__physical_payload_fingerprint",
        "meta__full_capture_fingerprint",
    }
)


class PublisherIdentityError(RuntimeError):
    """Raised when a reviewed source, fixture, or acceptance identity drifts."""


class PublisherSchemaError(RuntimeError):
    """Raised when a raw or final archive no longer has its reviewed schema."""


class PublicationAcceptanceError(RuntimeError):
    """Raised when detached publication authorization is absent or invalid."""


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scalar(value: Any) -> Any:
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
    """Load a deterministic pickle-free archive into detached arrays."""

    with np.load(path, allow_pickle=False) as archive:
        if archive.files != sorted(archive.files):
            raise PublisherSchemaError(f"{path} members are not lexically ordered")
        return _copy_mapping({name: archive[name] for name in archive.files})


def schema_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash exact names, dtypes, and shapes without array values."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise TypeError(f"{name} has forbidden object dtype")
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    return digest.hexdigest()


def capture_fingerprint(
    arrays: Mapping[str, np.ndarray],
    *,
    physical_payload_only: bool,
) -> str:
    """Reproduce the accepted Chapter 5 worker fingerprint."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        if name in FINGERPRINT_FIELDS:
            continue
        if physical_payload_only and name.startswith(("meta__", "identity__")):
            continue
        array = np.asarray(arrays[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def loaded_source_manifest_digest() -> str:
    """Return the accepted canonical digest of the 52 loaded Python sources."""

    text = json.dumps(
        oracle_worker.FROZEN_PINNED_PYTHON_MANIFEST,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode()).hexdigest()


def inventory_mapping_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash exact raw-name, disposition, and published-member ownership."""

    digest = hashlib.sha256()
    for member in (
        "inventory__raw_member_name",
        "inventory__disposition",
        "inventory__published_member",
    ):
        value = np.asarray(arrays[member])
        digest.update(member.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _verify_file_hashes(
    files: Mapping[Path, str],
    *,
    label: str,
) -> None:
    for path, expected in files.items():
        if not path.is_file():
            raise PublisherIdentityError(f"{label} is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise PublisherIdentityError(
                f"{label} {path} has SHA-256 {actual}; expected {expected}"
            )


def validate_fixture_payload(arrays: Mapping[str, np.ndarray]) -> None:
    """Require the canonical 217-field fixture schema and payload digest."""

    copied = _copy_mapping(arrays)
    expected_names = oracle_worker._expected_fixture_keys()
    if set(copied) != expected_names:
        missing = sorted(expected_names - set(copied))
        extra = sorted(set(copied) - expected_names)
        raise PublisherIdentityError(
            f"fixture payload schema changed; missing={missing}, extra={extra}"
        )
    if len(copied) != 217:
        raise PublisherIdentityError(
            f"fixture has {len(copied)} members; expected 217"
        )
    actual = oracle_worker.array_mapping_digest(copied)
    if actual != ACCEPTED_FIXTURE_PAYLOAD_DIGEST:
        raise PublisherIdentityError(
            f"fixture payload digest is {actual}; "
            f"expected {ACCEPTED_FIXTURE_PAYLOAD_DIGEST}"
        )


def verify_static_identity() -> dict[str, Any]:
    """Fail closed before a scientific child is allowed to start."""

    try:
        _canonical_publisher_bytes()
    except PublicationAcceptanceError as error:
        raise PublisherIdentityError(
            "Chapter 5 publisher is not executing from its canonical path"
        ) from error
    if PINNED_ROOT.resolve() != Path("/Users/ysting/payne-zero").resolve():
        raise PublisherIdentityError("publisher pin path changed")
    completed = subprocess.run(
        ["git", "-C", str(PINNED_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != PINNED_COMMIT:
        raise PublisherIdentityError("pinned Payne Zero commit changed")

    _verify_file_hashes(
        {
            PUBLISHER_INPUT_PATHS[name]: expected
            for name, expected in ACCEPTED_INPUT_SHA256.items()
        },
        label="reviewed publisher input",
    )
    _verify_file_hashes(STAGED_SOURCE_HASHES, label="staged continuum source")
    _verify_file_hashes(STAGED_STATIC_HASHES, label="staged continuum static file")
    _verify_file_hashes(
        {
            PINNED_ROOT / relative: expected
            for relative, expected in oracle_worker.FROZEN_PINNED_PYTHON_MANIFEST.items()
        },
        label="pinned loaded Python source",
    )
    _verify_file_hashes(
        {
            path: expected
            for path, expected in (
                value for value in oracle_worker.EXPECTED_DATA_HASHES.values()
            )
        },
        label="pinned continuum static file",
    )
    fixture = oracle_worker.load_fixture(FIXTURE_PATH)
    validate_fixture_payload(fixture)
    return {
        "commit": PINNED_COMMIT,
        "loaded_python_source_count": len(
            oracle_worker.FROZEN_PINNED_PYTHON_MANIFEST
        ),
        "fixture_sha256": ACCEPTED_INPUT_SHA256["fixture"],
        "acceptance_sha256": ACCEPTED_INPUT_SHA256["oracle_acceptance"],
    }


def validate_raw_capture(arrays: Mapping[str, np.ndarray]) -> None:
    """Require the exact independently accepted worker result."""

    copied = _copy_mapping(arrays)
    if len(copied) != ACCEPTED_RAW_KEY_COUNT:
        raise PublisherSchemaError(
            f"raw capture has {len(copied)} keys; expected {ACCEPTED_RAW_KEY_COUNT}"
        )
    actual_schema = schema_digest(copied)
    if actual_schema != ACCEPTED_RAW_SCHEMA_DIGEST:
        raise PublisherSchemaError(
            f"raw schema digest is {actual_schema}; "
            f"expected {ACCEPTED_RAW_SCHEMA_DIGEST}"
        )
    exact_scalars = {
        "meta__frozen_capture_schema_version": ACCEPTED_RAW_SCHEMA_VERSION,
        "meta__worker_sha256": ACCEPTED_INPUT_SHA256["worker"],
        "meta__capture_contract_sha256": ACCEPTED_INPUT_SHA256[
            "capture_contract"
        ],
        "meta__fixture_sha256": ACCEPTED_INPUT_SHA256["fixture"],
        "fixture_payload__content_digest": ACCEPTED_FIXTURE_PAYLOAD_DIGEST,
        "meta__capture_schema_digest": ACCEPTED_RAW_SCHEMA_DIGEST,
        "meta__physical_payload_fingerprint": ACCEPTED_PHYSICAL_FINGERPRINT,
        "meta__full_capture_fingerprint": ACCEPTED_FULL_FINGERPRINT,
    }
    for name, expected in exact_scalars.items():
        if name not in copied or _scalar(copied[name]) != expected:
            raise PublisherSchemaError(f"raw capture has the wrong {name}")
    for name in (
        "meta__capture_scope_complete",
        "meta__post_lane_loaded_manifest_verified",
        "meta__runner_continuum_binding_verified",
        "meta__pipeline_continuum_binding_verified",
    ):
        if not bool(_scalar(copied[name])):
            raise PublisherSchemaError(f"raw capture flag is false: {name}")
    if bool(_scalar(copied["meta__golden_publication_performed"])):
        raise PublisherSchemaError("scientific worker claimed golden publication")
    if capture_fingerprint(
        copied, physical_payload_only=True
    ) != ACCEPTED_PHYSICAL_FINGERPRINT:
        raise PublisherSchemaError("raw physical fingerprint does not recompute")
    if capture_fingerprint(
        copied, physical_payload_only=False
    ) != ACCEPTED_FULL_FINGERPRINT:
        raise PublisherSchemaError("raw full fingerprint does not recompute")


def _publisher_metadata(
    kind: str,
    raw: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return publication and accepted-oracle identity metadata."""

    return {
        "meta__archive_kind": np.asarray(kind),
        "meta__archive_schema_version": np.asarray(1, dtype=np.int64),
        "meta__payne_zero_commit": np.asarray(PINNED_COMMIT),
        "meta__worker_sha256": np.asarray(ACCEPTED_INPUT_SHA256["worker"]),
        "meta__capture_contract_sha256": np.asarray(
            ACCEPTED_INPUT_SHA256["capture_contract"]
        ),
        "meta__exact_source_contract_sha256": np.asarray(
            ACCEPTED_INPUT_SHA256["exact_source_contract"]
        ),
        "meta__publisher_contract_sha256": np.asarray(
            ACCEPTED_INPUT_SHA256["publisher_contract"]
        ),
        "meta__publisher_sha256": np.asarray(_canonical_publisher_sha256()),
        "meta__deterministic_npz_sha256": np.asarray(
            ACCEPTED_INPUT_SHA256["deterministic_npz"]
        ),
        "meta__oracle_acceptance_sha256": np.asarray(
            ACCEPTED_INPUT_SHA256["oracle_acceptance"]
        ),
        "meta__fixture_sha256": np.asarray(ACCEPTED_INPUT_SHA256["fixture"]),
        "meta__fixture_payload_digest": np.asarray(
            ACCEPTED_FIXTURE_PAYLOAD_DIGEST
        ),
        "meta__raw_capture_schema_version": np.asarray(
            ACCEPTED_RAW_SCHEMA_VERSION, dtype=np.int64
        ),
        "meta__raw_capture_key_count": np.asarray(
            ACCEPTED_RAW_KEY_COUNT, dtype=np.int64
        ),
        "meta__raw_capture_schema_digest": np.asarray(
            ACCEPTED_RAW_SCHEMA_DIGEST
        ),
        "meta__accepted_physical_payload_fingerprint": np.asarray(
            ACCEPTED_PHYSICAL_FINGERPRINT
        ),
        "meta__accepted_full_capture_fingerprint": np.asarray(
            ACCEPTED_FULL_FINGERPRINT
        ),
        "meta__loaded_pinned_python_source_count": np.asarray(
            int(_scalar(raw["meta__loaded_pinned_python_source_count"])),
            dtype=np.int64,
        ),
        "meta__loaded_pinned_python_manifest_digest": np.asarray(
            _scalar(raw["meta__post_lane_loaded_manifest_digest"])
        ),
        "meta__cpu_only": np.asarray(True, dtype=np.bool_),
        "meta__work_dtype": np.asarray("float64"),
        "meta__regime_names": np.asarray(REGIME_NAMES),
        "meta__process_control_name": np.asarray(sorted(PROCESS_CONTROLS)),
        "meta__process_control_value": np.asarray(
            [PROCESS_CONTROLS[name] for name in sorted(PROCESS_CONTROLS)]
        ),
    }


def _stack(raw: Mapping[str, np.ndarray], pattern: str) -> np.ndarray:
    """Stack one per-regime raw member in accepted regime order."""

    return np.stack([np.asarray(raw[pattern.format(regime=regime)]) for regime in REGIME_NAMES])


def _bitwise_equal(first: Any, second: Any) -> bool:
    """Return whether two arrays have identical dtype, shape, and bytes."""

    left = np.asarray(first)
    right = np.asarray(second)
    return (
        left.dtype.str == right.dtype.str
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _require_bitwise_equal(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    if not _bitwise_equal(actual, expected):
        raise PublisherSchemaError(f"bit-identical deduplication route changed: {label}")


def _require_regime_invariant(
    raw: Mapping[str, np.ndarray],
    pattern: str,
) -> np.ndarray:
    """Return one raw member after requiring bit equality across regimes."""

    values = [np.asarray(raw[pattern.format(regime=regime)]) for regime in REGIME_NAMES]
    first = values[0]
    for regime, value in zip(REGIME_NAMES[1:], values[1:]):
        if not _bitwise_equal(value, first):
            raise PublisherSchemaError(
                f"regime-invariant member changed for {regime}: {pattern}"
            )
    return first.copy()


def _mark(
    ownership: dict[str, tuple[str, str]],
    raw_name: str,
    disposition: str,
    published_name: str,
) -> None:
    if raw_name in ownership:
        raise PublisherSchemaError(f"raw member received two owners: {raw_name}")
    ownership[raw_name] = (disposition, published_name)


def _copy_common_oracle_metadata(
    raw: Mapping[str, np.ndarray],
    reader: dict[str, np.ndarray],
    integration: dict[str, np.ndarray],
    ownership: dict[str, tuple[str, str]],
) -> None:
    reader_owned_meta = {
        "meta__pipeline_continuum_fields",
        "meta__sampled_extension_wavelength_nm",
    }
    for name in sorted(raw):
        if not name.startswith(("meta__", "identity__")):
            continue
        if name in reader_owned_meta:
            continue
        target = f"oracle__{name}"
        reader[target] = np.asarray(raw[name]).copy()
        integration[target] = np.asarray(raw[name]).copy()
        _mark(ownership, name, "common_identity_metadata", target)


def _add_reader_member(
    reader: dict[str, np.ndarray],
    ownership: dict[str, tuple[str, str]],
    raw: Mapping[str, np.ndarray],
    raw_name: str,
    published_name: str,
    *,
    disposition: str = "reader",
) -> None:
    reader[published_name] = np.asarray(raw[raw_name]).copy()
    _mark(ownership, raw_name, disposition, published_name)


def _add_reader_axes(
    raw: Mapping[str, np.ndarray],
    reader: dict[str, np.ndarray],
    ownership: dict[str, tuple[str, str]],
) -> None:
    reader["axis__regime_name"] = np.asarray(REGIME_NAMES)
    reader["axis__regime_index"] = np.arange(len(REGIME_NAMES), dtype=np.int64)
    reader["axis__depth_index"] = np.arange(6, dtype=np.int64)
    for name in sorted(raw):
        if name.startswith("frequency_probe__"):
            _add_reader_member(
                reader,
                ownership,
                raw,
                name,
                name.replace("frequency_probe__", "axis__diagnostic__", 1),
            )
        elif name == "ifop__frequency_hz":
            if not np.array_equal(
                raw[name], reader["axis__diagnostic__frequency_hz"]
            ):
                raise PublisherSchemaError("IFOP diagnostic frequency changed")
            _mark(
                ownership,
                name,
                "reader_alias",
                "axis__diagnostic__frequency_hz",
            )
        elif name.startswith("ifop__"):
            _add_reader_member(
                reader,
                ownership,
                raw,
                name,
                name.replace("ifop__", "seam__ifop__", 1),
            )
    for suffix in sorted(READER_MOLECULAR_BOUNDARY_FIELDS):
        raw_name = f"molecular_boundary__{suffix}"
        _add_reader_member(
            reader,
            ownership,
            raw,
            raw_name,
            f"seam__molecular_boundary__{suffix}",
        )


def _add_reader_atmosphere(
    raw: Mapping[str, np.ndarray],
    reader: dict[str, np.ndarray],
    ownership: dict[str, tuple[str, str]],
) -> None:
    for suffix in ("absorption", "scattering", "source"):
        output = f"atmosphere__compact__{suffix}"
        reader[output] = _stack(raw, f"{{regime}}__atmosphere__{suffix}")
        for regime in REGIME_NAMES:
            _mark(
                ownership,
                f"{regime}__atmosphere__{suffix}",
                "reader",
                f"{output}[{REGIME_NAMES.index(regime)}]",
            )

    runner_flags = _require_regime_invariant(
        raw, "{regime}__atmosphere__runner_opacity_flags"
    )
    reader["atmosphere__runner_opacity_flags"] = runner_flags
    for regime in REGIME_NAMES:
        _mark(
            ownership,
            f"{regime}__atmosphere__runner_opacity_flags",
            "reader",
            "atmosphere__runner_opacity_flags",
        )
        _mark(
            ownership,
            f"{regime}__atmosphere__frequency_hz",
            "reader_alias",
            "axis__diagnostic__frequency_hz",
        )

    reader["atmosphere__component__absorption_name"] = np.asarray(
        ATMOSPHERE_ABSORPTION_COMPONENTS
    )
    reader["atmosphere__component__absorption"] = np.stack(
        [
            np.stack(
                [
                    raw[
                        f"{regime}__atmosphere__component__{component}__absorption"
                    ]
                    for component in ATMOSPHERE_ABSORPTION_COMPONENTS
                ]
            )
            for regime in REGIME_NAMES
        ]
    )
    reader["atmosphere__component__source_name"] = np.asarray(
        ATMOSPHERE_ABSORPTION_COMPONENTS
    )
    reader["atmosphere__component__source"] = np.stack(
        [
            np.stack(
                [
                    raw[f"{regime}__atmosphere__component__{component}__source"]
                    for component in ATMOSPHERE_ABSORPTION_COMPONENTS
                ]
            )
            for regime in REGIME_NAMES
        ]
    )
    for regime_index, regime in enumerate(REGIME_NAMES):
        for component_index, component in enumerate(
            ATMOSPHERE_ABSORPTION_COMPONENTS
        ):
            for quantity in ("absorption", "source"):
                _mark(
                    ownership,
                    (
                        f"{regime}__atmosphere__component__"
                        f"{component}__{quantity}"
                    ),
                    "reader",
                    (
                        f"atmosphere__component__{quantity}"
                        f"[{regime_index},{component_index}]"
                    ),
                )

    reader["atmosphere__component__scattering_name"] = np.asarray(
        ATMOSPHERE_SCATTERING_COMPONENTS
    )
    reader["atmosphere__component__scattering"] = np.stack(
        [
            np.stack(
                [
                    raw[
                        f"{regime}__atmosphere__component__"
                        f"{component}__scattering"
                    ]
                    for component in ATMOSPHERE_SCATTERING_COMPONENTS
                ]
            )
            for regime in REGIME_NAMES
        ]
    )
    for regime_index, regime in enumerate(REGIME_NAMES):
        for component_index, component in enumerate(
            ATMOSPHERE_SCATTERING_COMPONENTS
        ):
            _mark(
                ownership,
                (
                    f"{regime}__atmosphere__component__"
                    f"{component}__scattering"
                ),
                "reader",
                (
                    "atmosphere__component__scattering"
                    f"[{regime_index},{component_index}]"
                ),
            )

    molecular_raw_suffix = {
        "ch": "ch_absorption",
        "oh": "oh_absorption",
        "h2_cia": "h2_cia_absorption",
    }
    reader["atmosphere__molecular_component__absorption_name"] = np.asarray(
        ATMOSPHERE_MOLECULAR_COMPONENTS
    )
    reader["atmosphere__molecular_component__absorption"] = np.stack(
        [
            np.stack(
                [
                    raw[
                        f"{regime}__atmosphere__molecular_component__"
                        f"{molecular_raw_suffix[component]}"
                    ]
                    for component in ATMOSPHERE_MOLECULAR_COMPONENTS
                ]
            )
            for regime in REGIME_NAMES
        ]
    )
    for regime_index, regime in enumerate(REGIME_NAMES):
        for component_index, component in enumerate(
            ATMOSPHERE_MOLECULAR_COMPONENTS
        ):
            _mark(
                ownership,
                (
                    f"{regime}__atmosphere__molecular_component__"
                    f"{molecular_raw_suffix[component]}"
                ),
                "reader",
                (
                    "atmosphere__molecular_component__absorption"
                    f"[{regime_index},{component_index}]"
                ),
            )

    simple_component_suffixes = (
        "component__ordered_absorption_sum",
        "component__ordered_scattering_sum",
        "component__ordered_source_numerator_sum",
        "component__absorption_residual",
        "component__scattering_residual",
        "component__source_residual",
        "molecular_component__ordered_absorption_sum",
    )
    for suffix in simple_component_suffixes:
        output = f"atmosphere__{suffix}"
        reader[output] = _stack(raw, f"{{regime}}__atmosphere__{suffix}")
        for regime_index, regime in enumerate(REGIME_NAMES):
            _mark(
                ownership,
                f"{regime}__atmosphere__{suffix}",
                "reader",
                f"{output}[{regime_index}]",
            )

    reference_wavelength = _require_regime_invariant(
        raw, "{regime}__atmosphere__line_reference_wavelength_nm"
    )
    reference_index = _require_regime_invariant(
        raw, "{regime}__atmosphere__line_reference_packed_wavelength_index"
    )
    reader["line_reference__wavelength_nm"] = reference_wavelength
    reader["line_reference__packed_wavelength_index"] = reference_index
    for regime in REGIME_NAMES:
        for suffix, output in (
            ("line_reference_wavelength_nm", "line_reference__wavelength_nm"),
            (
                "line_reference_packed_wavelength_index",
                "line_reference__packed_wavelength_index",
            ),
        ):
            _mark(
                ownership,
                f"{regime}__atmosphere__{suffix}",
                "reader",
                output,
            )

    threshold = _stack(
        raw, "{regime}__atmosphere__line_reference_threshold"
    )
    reader["line_reference__threshold"] = threshold
    counts = np.asarray(
        [
            raw[f"{regime}__atmosphere__line_reference_active_index"].size
            for regime in REGIME_NAMES
        ],
        dtype=np.int64,
    )
    reader["line_reference__active_count"] = counts
    maximum = int(np.max(counts))
    active_index = np.zeros((4, maximum), dtype=np.int64)
    active_frequency = np.zeros((4, maximum), dtype=np.float64)
    valid = np.zeros((4, maximum), dtype=np.bool_)
    active_products = {
        name: np.zeros((4, 6, maximum), dtype=np.float64)
        for name in ("absorption", "scattering", "source")
    }
    for regime_index, regime in enumerate(REGIME_NAMES):
        count = int(counts[regime_index])
        valid[regime_index, :count] = True
        active_index[regime_index, :count] = raw[
            f"{regime}__atmosphere__line_reference_active_index"
        ]
        active_frequency[regime_index, :count] = raw[
            f"{regime}__atmosphere__line_reference_active_frequency_hz"
        ]
        for name in active_products:
            active_products[name][regime_index, :, :count] = raw[
                f"{regime}__atmosphere__line_reference_active_{name}"
            ]
        _mark(
            ownership,
            f"{regime}__atmosphere__line_reference_threshold",
            "reader",
            f"line_reference__threshold[{regime_index}]",
        )
        for suffix, output in (
            ("active_index", "line_reference__active_index"),
            ("active_frequency_hz", "line_reference__active_frequency_hz"),
            ("active_absorption", "line_reference__active_absorption"),
            ("active_scattering", "line_reference__active_scattering"),
            ("active_source", "line_reference__active_source"),
        ):
            _mark(
                ownership,
                f"{regime}__atmosphere__line_reference_{suffix}",
                "reader",
                f"{output}[{regime_index},valid]",
            )
    reader["line_reference__active_index"] = active_index
    reader["line_reference__active_frequency_hz"] = active_frequency
    reader["line_reference__active_valid"] = valid
    for name, value in active_products.items():
        reader[f"line_reference__active_{name}"] = value


def _add_reader_synthesis(
    raw: Mapping[str, np.ndarray],
    reader: dict[str, np.ndarray],
    ownership: dict[str, tuple[str, str]],
) -> None:
    first_prefix = f"{REGIME_NAMES[0]}__synthesis__route__"
    route_suffixes = sorted(
        name.removeprefix(first_prefix)
        for name in raw
        if name.startswith(first_prefix)
    )
    for suffix in route_suffixes:
        if suffix == "packaged_sample_frequency_hz":
            continue
        output = (
            "synthesis__edge__sample_frequency_hz"
            if suffix == "all_sample_frequency_hz"
            else f"synthesis__edge__{suffix}"
        )
        reader[output] = _require_regime_invariant(
            raw, f"{{regime}}__synthesis__route__{suffix}"
        )
        for regime in REGIME_NAMES:
            _mark(
                ownership,
                f"{regime}__synthesis__route__{suffix}",
                "reader",
                output,
            )
    packaged_pattern = (
        "{regime}__synthesis__route__packaged_sample_frequency_hz"
    )
    packaged_frequency = _require_regime_invariant(raw, packaged_pattern)
    if not np.array_equal(
        packaged_frequency, reader["synthesis__edge__sample_frequency_hz"]
    ):
        raise PublisherSchemaError(
            "rebuilt and packaged synthesis sample frequencies differ"
        )
    for regime in REGIME_NAMES:
        _mark(
            ownership,
            packaged_pattern.format(regime=regime),
            "reader_alias",
            "synthesis__edge__sample_frequency_hz",
        )

    standard_suffixes = (
        "absorption",
        "scattering",
        "sample_absorption",
        "sample_scattering",
        "reconstructed_absorption",
        "reconstructed_scattering",
        "interpolation_absorption_residual",
        "interpolation_scattering_residual",
        "coulomb_table_energy_first",
        "frequency_invariants_supplied",
    )
    for suffix in standard_suffixes:
        output = f"synthesis__standard__{suffix}"
        reader[output] = _stack(
            raw, f"{{regime}}__synthesis__standard__{suffix}"
        )
        for regime_index, regime in enumerate(REGIME_NAMES):
            _mark(
                ownership,
                f"{regime}__synthesis__standard__{suffix}",
                "reader",
                f"{output}[{regime_index}]",
            )

    reader["synthesis__standard__component_name"] = np.asarray(
        SYNTHESIS_COMPONENT_NAMES
    )
    reader["synthesis__standard__component"] = np.stack(
        [
            np.stack(
                [
                    raw[
                        f"{regime}__synthesis__standard__component__{component}"
                    ]
                    for component in SYNTHESIS_COMPONENT_NAMES
                ]
            )
            for regime in REGIME_NAMES
        ]
    )
    reader["synthesis__diagnostic__component_name"] = np.asarray(
        SYNTHESIS_COMPONENT_NAMES
    )
    reader["synthesis__diagnostic__component"] = np.stack(
        [
            np.stack(
                [
                    raw[
                        f"{regime}__synthesis__diagnostic__component__{component}"
                    ]
                    for component in SYNTHESIS_COMPONENT_NAMES
                ]
            )
            for regime in REGIME_NAMES
        ]
    )
    for lane in ("standard", "diagnostic"):
        for regime_index, regime in enumerate(REGIME_NAMES):
            for component_index, component in enumerate(SYNTHESIS_COMPONENT_NAMES):
                _mark(
                    ownership,
                    f"{regime}__synthesis__{lane}__component__{component}",
                    "reader",
                    (
                        f"synthesis__{lane}__component"
                        f"[{regime_index},{component_index}]"
                    ),
                )
        for suffix in (
            "component__ordered_absorption_sum",
            "component__ordered_scattering_sum",
            "component__absorption_residual",
            "component__scattering_residual",
        ):
            output = f"synthesis__{lane}__{suffix}"
            reader[output] = _stack(
                raw, f"{{regime}}__synthesis__{lane}__{suffix}"
            )
            for regime_index, regime in enumerate(REGIME_NAMES):
                _mark(
                    ownership,
                    f"{regime}__synthesis__{lane}__{suffix}",
                    "reader",
                    f"{output}[{regime_index}]",
                )

    reader["synthesis__standard__isolated_minor__case_name"] = np.asarray(
        SYNTHESIS_MINOR_CASES
    )
    for quantity in ("absorption", "scattering"):
        output = f"synthesis__standard__isolated_minor__{quantity}"
        reader[output] = np.stack(
            [
                np.stack(
                    [
                        raw[
                            f"{regime}__synthesis__standard__isolated_minor__"
                            f"{case}__{quantity}"
                        ]
                        for case in SYNTHESIS_MINOR_CASES
                    ]
                )
                for regime in REGIME_NAMES
            ]
        )
        for regime_index, regime in enumerate(REGIME_NAMES):
            for case_index, case in enumerate(SYNTHESIS_MINOR_CASES):
                _mark(
                    ownership,
                    (
                        f"{regime}__synthesis__standard__isolated_minor__"
                        f"{case}__{quantity}"
                    ),
                    "reader",
                    f"{output}[{regime_index},{case_index}]",
                )
    for suffix in (
        "ordered_absorption_sum",
        "ordered_scattering_sum",
        "absorption_residual",
        "scattering_residual",
    ):
        output = f"synthesis__standard__isolated_minor__{suffix}"
        reader[output] = _stack(
            raw,
            f"{{regime}}__synthesis__standard__isolated_minor__{suffix}",
        )
        for regime_index, regime in enumerate(REGIME_NAMES):
            _mark(
                ownership,
                (
                    f"{regime}__synthesis__standard__isolated_minor__"
                    f"{suffix}"
                ),
                "reader",
                f"{output}[{regime_index}]",
            )

    for lane in ("diagnostic", "extension"):
        for suffix in (
            "absorption",
            "scattering",
            "source",
            "coulomb_table_energy_first",
            "frequency_invariants_supplied",
        ):
            raw_name = f"{{regime}}__synthesis__{lane}__{suffix}"
            if raw_name.format(regime=REGIME_NAMES[0]) not in raw:
                continue
            output = f"synthesis__{lane}__{suffix}"
            reader[output] = _stack(raw, raw_name)
            for regime_index, regime in enumerate(REGIME_NAMES):
                _mark(
                    ownership,
                    raw_name.format(regime=regime),
                    "reader",
                    f"{output}[{regime_index}]",
                )

    for suffix in (
        "frequency_hz",
        "support_wavelength_min_nm",
        "support_wavelength_max_nm",
        "wavelength_nm",
        "continuum_atmosphere_field_names",
        "pops_field_names",
        "invariant_field_names",
    ):
        pattern = f"{{regime}}__synthesis__extension__{suffix}"
        if pattern.format(regime=REGIME_NAMES[0]) not in raw:
            continue
        output = f"synthesis__extension__{suffix}"
        reader[output] = _require_regime_invariant(raw, pattern)
        for regime in REGIME_NAMES:
            _mark(ownership, pattern.format(regime=regime), "reader", output)

    invariant_names = tuple(
        str(name)
        for name in reader["synthesis__extension__invariant_field_names"].tolist()
    )
    if invariant_names != EXTENSION_INVARIANT_FIELDS:
        raise PublisherSchemaError(
            "extension FrequencyInvariants field inventory changed"
        )

    meta_aliases = {
        "meta__sampled_extension_wavelength_nm": (
            "synthesis__extension__wavelength_nm"
        ),
        "meta__pipeline_continuum_fields": (
            "synthesis__extension__continuum_atmosphere_field_names"
        ),
    }
    for raw_name, published_name in meta_aliases.items():
        if not np.array_equal(raw[raw_name], reader[published_name]):
            raise PublisherSchemaError(
                f"reader-owned metadata alias changed: {raw_name}"
            )
        _mark(ownership, raw_name, "reader_alias", published_name)

    diagnostic_frequency = _require_regime_invariant(
        raw, "{regime}__synthesis__diagnostic__frequency_hz"
    )
    if not np.array_equal(
        diagnostic_frequency, reader["axis__diagnostic__frequency_hz"]
    ):
        raise PublisherSchemaError(
            "synthesis diagnostic frequency differs from the shared axis"
        )
    for regime in REGIME_NAMES:
        _mark(
            ownership,
            f"{regime}__synthesis__diagnostic__frequency_hz",
            "reader_alias",
            "axis__diagnostic__frequency_hz",
        )

    reader["reader_state__field_name"] = np.asarray(READER_STATE_FIELDS)
    for field in READER_STATE_FIELDS:
        pattern = f"{{regime}}__synthesis__extension__input__pops__{field}"
        output = f"reader_state__{field}"
        reader[output] = _stack(raw, pattern)
        for regime_index, regime in enumerate(REGIME_NAMES):
            _mark(
                ownership,
                pattern.format(regime=regime),
                "reader",
                f"{output}[{regime_index}]",
            )

    for suffix in (
        "counterfactual__rich_hii_matches_trimmed",
        "counterfactual__schema_h2_bit_invariant",
        "counterfactual__signed_edge_bit_invariant",
    ):
        output = f"synthesis__seam__{suffix}"
        reader[output] = _stack(raw, f"{{regime}}__synthesis__{suffix}")
        for regime_index, regime in enumerate(REGIME_NAMES):
            _mark(
                ownership,
                f"{regime}__synthesis__{suffix}",
                "reader",
                f"{output}[{regime_index}]",
            )


def _add_reader_alias_evidence(
    raw: Mapping[str, np.ndarray],
    reader: Mapping[str, np.ndarray],
    ownership: dict[str, tuple[str, str]],
) -> None:
    """Route exact reader-owned evidence to string aliases only."""

    direct_axes = (
        (
            "sampling_boundary__reference_wavelength_nm",
            "line_reference__wavelength_nm",
        ),
        (
            "sampling_boundary__reference_packed_wavelength_index",
            "line_reference__packed_wavelength_index",
        ),
        ("ifop19__frequency_hz", "axis__diagnostic__frequency_hz"),
        ("molecular_entry__frequency_hz", "axis__diagnostic__frequency_hz"),
    )
    for raw_name, reader_name in direct_axes:
        _require_bitwise_equal(
            raw[raw_name],
            reader[reader_name],
            label=f"{raw_name} -> {reader_name}",
        )
        _mark(ownership, raw_name, "reader_alias", reader_name)

    for regime_index, regime in enumerate(REGIME_NAMES):
        prefix = f"{regime}__synthesis__"
        routes = (
            (
                f"{prefix}extension__input__invariant__frequencies_hz",
                "synthesis__extension__frequency_hz",
                reader["synthesis__extension__frequency_hz"],
            ),
            (
                f"{prefix}extension__input__invariant__"
                "coulomb_table_energy_first",
                f"synthesis__extension__coulomb_table_energy_first[{regime_index}]",
                reader["synthesis__extension__coulomb_table_energy_first"][
                    regime_index
                ],
            ),
            (
                f"{prefix}standard__trace__called_frequency_hz",
                "synthesis__edge__selected_sample_frequency_hz",
                reader["synthesis__edge__selected_sample_frequency_hz"],
            ),
            (
                f"{prefix}counterfactual__signed_edge_original",
                "synthesis__edge__signed_edge_frequency_hz",
                reader["synthesis__edge__signed_edge_frequency_hz"],
            ),
        )
        for raw_name, reader_descriptor, expected in routes:
            _require_bitwise_equal(
                raw[raw_name],
                expected,
                label=f"{raw_name} -> {reader_descriptor}",
            )
            _mark(
                ownership,
                raw_name,
                "reader_alias",
                reader_descriptor,
            )

        for suffix, standard_quantity in COUNTERFACTUAL_STANDARD_OUTPUT_ALIASES:
            raw_name = f"{prefix}counterfactual__{suffix}"
            reader_descriptor = (
                f"synthesis__standard__{standard_quantity}[{regime_index}]"
            )
            expected = reader[f"synthesis__standard__{standard_quantity}"][
                regime_index
            ]
            _require_bitwise_equal(
                raw[raw_name],
                expected,
                label=f"{raw_name} -> {reader_descriptor}",
            )
            _mark(
                ownership,
                raw_name,
                "reader_alias",
                reader_descriptor,
            )


def _add_coalesced_integration_evidence(
    raw: Mapping[str, np.ndarray],
    integration: dict[str, np.ndarray],
    ownership: dict[str, tuple[str, str]],
) -> None:
    """Store one physical copy of each declared regime-invariant field."""

    reader_alias_fields = {
        "frequencies_hz",
        "coulomb_table_energy_first",
    }
    for field in EXTENSION_INVARIANT_FIELDS:
        if field in reader_alias_fields:
            continue
        pattern = (
            "{regime}__synthesis__extension__input__invariant__" + field
        )
        published_name = (
            "evidence__synthesis__extension__input__invariant__" + field
        )
        integration[published_name] = _require_regime_invariant(raw, pattern)
        for regime in REGIME_NAMES:
            _mark(
                ownership,
                pattern.format(regime=regime),
                "integration",
                published_name,
            )

    signed_edge_pattern = (
        "{regime}__synthesis__counterfactual__signed_edge_flipped"
    )
    signed_edge_member = (
        "evidence__synthesis__counterfactual__signed_edge_flipped"
    )
    integration[signed_edge_member] = _require_regime_invariant(
        raw, signed_edge_pattern
    )
    for regime in REGIME_NAMES:
        _mark(
            ownership,
            signed_edge_pattern.format(regime=regime),
            "integration",
            signed_edge_member,
        )

    for field in STANDARD_TRACE_FIELDS:
        if field == "called_frequency_hz":
            continue
        pattern = f"{{regime}}__synthesis__standard__trace__{field}"
        published_name = f"evidence__synthesis__standard__trace__{field}"
        integration[published_name] = _require_regime_invariant(raw, pattern)
        for regime in REGIME_NAMES:
            _mark(
                ownership,
                pattern.format(regime=regime),
                "integration",
                published_name,
            )


def _grid_bank(
    raw: Mapping[str, np.ndarray],
    integration: dict[str, np.ndarray],
    ownership: dict[str, tuple[str, str]],
) -> None:
    """Coalesce every exact 30,000-point sampling grid into five rows."""

    boundary_wavelength = np.asarray(raw["sampling_boundary__wavelength_nm"])
    boundary_weight = np.asarray(raw["sampling_boundary__frequency_weight_hz"])
    if boundary_wavelength.shape != (8, 30000) or boundary_weight.shape != (
        8,
        30000,
    ):
        raise PublisherSchemaError("sampling-boundary grid shape changed")

    bank_wavelength: list[np.ndarray] = []
    bank_weight: list[np.ndarray] = []

    def locate(wavelength: np.ndarray, weight: np.ndarray) -> int:
        for index, (known_wavelength, known_weight) in enumerate(
            zip(bank_wavelength, bank_weight)
        ):
            if np.array_equal(wavelength, known_wavelength):
                if not np.array_equal(weight, known_weight):
                    raise PublisherSchemaError(
                        "equal wavelength grids acquired unequal weights"
                    )
                return index
        bank_wavelength.append(np.asarray(wavelength).copy())
        bank_weight.append(np.asarray(weight).copy())
        return len(bank_wavelength) - 1

    boundary_selector = np.asarray(
        [
            locate(boundary_wavelength[row], boundary_weight[row])
            for row in range(boundary_wavelength.shape[0])
        ],
        dtype=np.int64,
    )
    regime_selector = np.empty(len(REGIME_NAMES), dtype=np.int64)
    for regime_index, regime in enumerate(REGIME_NAMES):
        wavelength = np.asarray(
            raw[f"{regime}__atmosphere__sampling_wavelength_nm"]
        )
        weight = np.asarray(
            raw[f"{regime}__atmosphere__sampling_frequency_weight_hz"]
        )
        regime_selector[regime_index] = locate(wavelength, weight)
        frequency = LIGHT_SPEED_NM_PER_S / np.maximum(wavelength, 1.0e-300)
        captured_frequency = np.asarray(
            raw[f"{regime}__atmosphere__product_frequency_hz"]
        )
        if not np.array_equal(frequency, captured_frequency):
            raise PublisherSchemaError(
                f"derived product frequency changed for {regime}"
            )

    if len(bank_wavelength) != 5:
        raise PublisherSchemaError(
            f"sampling grid has {len(bank_wavelength)} policies; expected 5"
        )
    integration["grid_bank__wavelength_nm"] = np.stack(bank_wavelength)
    integration["grid_bank__frequency_weight_hz"] = np.stack(bank_weight)
    integration["grid_bank__policy_index"] = np.arange(5, dtype=np.int64)
    integration["grid_bank__policy_label"] = np.asarray(GRID_POLICY_LABELS)
    integration["grid_bank__light_speed_nm_per_s"] = np.asarray(
        LIGHT_SPEED_NM_PER_S
    )
    integration["sampling_boundary__grid_bank_index"] = boundary_selector
    integration["atmosphere_product__grid_bank_index"] = regime_selector
    frequency_digests = []
    for wavelength in bank_wavelength:
        derived = LIGHT_SPEED_NM_PER_S / np.maximum(wavelength, 1.0e-300)
        frequency_digests.append(hashlib.sha256(derived.tobytes()).hexdigest())
    integration["grid_bank__derived_frequency_sha256"] = np.asarray(
        frequency_digests
    )

    _mark(
        ownership,
        "sampling_boundary__wavelength_nm",
        "coalesced_grid_bank",
        "grid_bank__wavelength_nm[sampling_boundary__grid_bank_index]",
    )
    _mark(
        ownership,
        "sampling_boundary__frequency_weight_hz",
        "coalesced_grid_bank",
        "grid_bank__frequency_weight_hz[sampling_boundary__grid_bank_index]",
    )
    for regime_index, regime in enumerate(REGIME_NAMES):
        for suffix, published in (
            ("sampling_wavelength_nm", "grid_bank__wavelength_nm"),
            (
                "sampling_frequency_weight_hz",
                "grid_bank__frequency_weight_hz",
            ),
            ("product_frequency_hz", "grid_bank__derived_frequency"),
        ):
            _mark(
                ownership,
                f"{regime}__atmosphere__{suffix}",
                "coalesced_grid_bank",
                f"{published}[atmosphere_product__grid_bank_index[{regime_index}]]",
            )


def _add_integration_products(
    raw: Mapping[str, np.ndarray],
    integration: dict[str, np.ndarray],
    ownership: dict[str, tuple[str, str]],
) -> None:
    for quantity in ("absorption", "scattering", "source"):
        output = f"atmosphere_product__{quantity}"
        integration[output] = _stack(
            raw, f"{{regime}}__atmosphere__product_{quantity}"
        )
        for regime_index, regime in enumerate(REGIME_NAMES):
            _mark(
                ownership,
                f"{regime}__atmosphere__product_{quantity}",
                "integration",
                f"{output}[{regime_index}]",
            )


def _complete_integration_evidence(
    raw: Mapping[str, np.ndarray],
    reader: Mapping[str, np.ndarray],
    integration: dict[str, np.ndarray],
    ownership: dict[str, tuple[str, str]],
) -> None:
    for raw_name in sorted(raw):
        if raw_name in ownership:
            continue
        published_name = f"evidence__{raw_name}"
        integration[published_name] = np.asarray(raw[raw_name]).copy()
        _mark(ownership, raw_name, "integration", published_name)

    for raw_name, (disposition, published_name) in sorted(ownership.items()):
        if disposition not in {"reader", "reader_alias"}:
            continue
        alias_name = f"alias__{raw_name}__reader_member"
        integration[alias_name] = np.asarray(published_name)

    raw_names = sorted(raw)
    integration["inventory__raw_member_name"] = np.asarray(raw_names)
    integration["inventory__disposition"] = np.asarray(
        [ownership[name][0] for name in raw_names]
    )
    integration["inventory__published_member"] = np.asarray(
        [ownership[name][1] for name in raw_names]
    )
    if set(ownership) != set(raw):
        missing = sorted(set(raw) - set(ownership))
        extra = sorted(set(ownership) - set(raw))
        raise PublisherSchemaError(
            f"logical raw coverage changed; missing={missing}, extra={extra}"
        )
    integration["meta__logical_raw_capture_coverage_complete"] = np.asarray(
        True, dtype=np.bool_
    )


def _finalize_schema_metadata(payload: dict[str, np.ndarray]) -> None:
    payload["meta__archive_schema_digest"] = np.asarray("0" * 64)
    payload["meta__archive_key_count"] = np.asarray(
        len(payload) + (0 if "meta__archive_key_count" in payload else 1),
        dtype=np.int64,
    )
    payload["meta__archive_schema_digest"] = np.asarray(schema_digest(payload))


def assemble_payloads(
    raw: Mapping[str, np.ndarray],
    *,
    validate_raw: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Project one accepted raw capture into the two reviewed payloads."""

    raw_copy = _copy_mapping(raw)
    if validate_raw:
        validate_raw_capture(raw_copy)
    reader = _publisher_metadata("continuum_reader", raw_copy)
    integration = _publisher_metadata("continuum_integration", raw_copy)
    ownership: dict[str, tuple[str, str]] = {}
    _copy_common_oracle_metadata(raw_copy, reader, integration, ownership)
    _add_reader_axes(raw_copy, reader, ownership)
    _add_reader_atmosphere(raw_copy, reader, ownership)
    _add_reader_synthesis(raw_copy, reader, ownership)
    _add_reader_alias_evidence(raw_copy, reader, ownership)
    _add_integration_products(raw_copy, integration, ownership)
    _grid_bank(raw_copy, integration, ownership)
    _add_coalesced_integration_evidence(
        raw_copy, integration, ownership
    )
    _complete_integration_evidence(
        raw_copy, reader, integration, ownership
    )
    _finalize_schema_metadata(reader)
    reader = _copy_mapping(reader)
    integration["meta__reader_archive_schema_digest"] = np.asarray(
        _scalar(reader["meta__archive_schema_digest"])
    )
    # The byte hash is added after the reader archive is serialized.
    integration["meta__reader_archive_sha256"] = np.asarray("0" * 64)
    _finalize_schema_metadata(integration)
    return _copy_mapping(reader), _copy_mapping(integration)


def _validate_common_final_semantics(
    name: str,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Validate final-only metadata that is not part of the raw fingerprint."""

    if not bool(_scalar(arrays["meta__cpu_only"])):
        raise PublisherSchemaError(f"{name} is not marked CPU-only")
    if _scalar(arrays["meta__work_dtype"]) != "float64":
        raise PublisherSchemaError(f"{name} has the wrong work dtype")
    if tuple(arrays["meta__regime_names"].tolist()) != REGIME_NAMES:
        raise PublisherSchemaError(f"{name} has the wrong regime order")
    control_names = tuple(arrays["meta__process_control_name"].tolist())
    control_values = tuple(arrays["meta__process_control_value"].tolist())
    if control_names != tuple(sorted(PROCESS_CONTROLS)):
        raise PublisherSchemaError(f"{name} has the wrong process controls")
    if control_values != tuple(
        PROCESS_CONTROLS[control] for control in sorted(PROCESS_CONTROLS)
    ):
        raise PublisherSchemaError(f"{name} has the wrong process-control values")
    if int(_scalar(arrays["meta__raw_capture_schema_version"])) != (
        ACCEPTED_RAW_SCHEMA_VERSION
    ):
        raise PublisherSchemaError(f"{name} has the wrong raw schema version")
    if int(_scalar(arrays["meta__raw_capture_key_count"])) != (
        ACCEPTED_RAW_KEY_COUNT
    ):
        raise PublisherSchemaError(f"{name} has the wrong raw key count")
    if int(_scalar(arrays["meta__loaded_pinned_python_source_count"])) != 52:
        raise PublisherSchemaError(f"{name} has the wrong loaded-source count")
    if _scalar(arrays["meta__loaded_pinned_python_manifest_digest"]) != (
        loaded_source_manifest_digest()
    ):
        raise PublisherSchemaError(f"{name} has the wrong loaded-source digest")


def _validate_reader_semantics(arrays: Mapping[str, np.ndarray]) -> None:
    """Validate reader-only names, axes, padding, and threshold conventions."""

    exact_vectors = {
        "axis__regime_name": np.asarray(REGIME_NAMES),
        "axis__regime_index": np.arange(4, dtype=np.int64),
        "axis__depth_index": np.arange(6, dtype=np.int64),
        "atmosphere__component__absorption_name": np.asarray(
            ATMOSPHERE_ABSORPTION_COMPONENTS
        ),
        "atmosphere__component__source_name": np.asarray(
            ATMOSPHERE_ABSORPTION_COMPONENTS
        ),
        "atmosphere__component__scattering_name": np.asarray(
            ATMOSPHERE_SCATTERING_COMPONENTS
        ),
        "atmosphere__molecular_component__absorption_name": np.asarray(
            ATMOSPHERE_MOLECULAR_COMPONENTS
        ),
        "synthesis__standard__component_name": np.asarray(
            SYNTHESIS_COMPONENT_NAMES
        ),
        "synthesis__diagnostic__component_name": np.asarray(
            SYNTHESIS_COMPONENT_NAMES
        ),
        "synthesis__standard__isolated_minor__case_name": np.asarray(
            SYNTHESIS_MINOR_CASES
        ),
        "reader_state__field_name": np.asarray(READER_STATE_FIELDS),
    }
    for member, expected in exact_vectors.items():
        if not np.array_equal(arrays[member], expected):
            raise PublisherSchemaError(f"reader semantic vector changed: {member}")

    counts = np.asarray(arrays["line_reference__active_count"], dtype=np.int64)
    expected_counts = np.asarray([338, 240, 240, 226], dtype=np.int64)
    if not np.array_equal(counts, expected_counts):
        raise PublisherSchemaError("reader active line-reference counts changed")
    valid = np.asarray(arrays["line_reference__active_valid"], dtype=np.bool_)
    expected_valid = (
        np.arange(valid.shape[1], dtype=np.int64)[None, :] < counts[:, None]
    )
    if not np.array_equal(valid, expected_valid):
        raise PublisherSchemaError("reader active validity mask changed")
    if not np.array_equal(np.sum(valid, axis=1), counts):
        raise PublisherSchemaError("reader active counts do not match validity")
    invalid = ~valid
    if np.any(np.asarray(arrays["line_reference__active_index"])[invalid] != 0):
        raise PublisherSchemaError("reader active-index padding is not zero")
    if np.any(
        np.asarray(arrays["line_reference__active_frequency_hz"])[invalid] != 0.0
    ):
        raise PublisherSchemaError("reader active-frequency padding is not zero")
    for quantity in ("absorption", "scattering", "source"):
        values = np.asarray(arrays[f"line_reference__active_{quantity}"])
        for regime_index in range(4):
            if np.any(values[regime_index, :, invalid[regime_index]] != 0.0):
                raise PublisherSchemaError(
                    f"reader active-{quantity} padding is not zero"
                )
    packed = np.asarray(arrays["line_reference__packed_wavelength_index"])
    if packed.shape != (344,) or int(packed[-1]) != 2**30:
        raise PublisherSchemaError("reader packed line-reference sentinel changed")
    threshold = np.asarray(arrays["line_reference__threshold"])
    if not np.array_equal(threshold[:, :, -1], threshold[:, :, -2]):
        raise PublisherSchemaError("reader final threshold column is not duplicated")


def _validate_deduplicated_inventory(
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Require every frozen physical-deduplication route exactly."""

    names = [str(name) for name in arrays["inventory__raw_member_name"].tolist()]
    dispositions = [
        str(value) for value in arrays["inventory__disposition"].tolist()
    ]
    published = [
        str(value) for value in arrays["inventory__published_member"].tolist()
    ]
    routes = dict(zip(names, zip(dispositions, published)))

    def require(raw_name: str, disposition: str, member: str) -> None:
        actual = routes.get(raw_name)
        expected = (disposition, member)
        if actual != expected:
            raise PublisherSchemaError(
                f"deduplication route for {raw_name} is {actual}; "
                f"expected {expected}"
            )

    for raw_name, reader_name in (
        (
            "sampling_boundary__reference_wavelength_nm",
            "line_reference__wavelength_nm",
        ),
        (
            "sampling_boundary__reference_packed_wavelength_index",
            "line_reference__packed_wavelength_index",
        ),
        ("ifop19__frequency_hz", "axis__diagnostic__frequency_hz"),
        ("molecular_entry__frequency_hz", "axis__diagnostic__frequency_hz"),
    ):
        require(raw_name, "reader_alias", reader_name)

    for regime_index, regime in enumerate(REGIME_NAMES):
        prefix = f"{regime}__synthesis__"
        require(
            f"{prefix}extension__input__invariant__frequencies_hz",
            "reader_alias",
            "synthesis__extension__frequency_hz",
        )
        require(
            f"{prefix}extension__input__invariant__"
            "coulomb_table_energy_first",
            "reader_alias",
            f"synthesis__extension__coulomb_table_energy_first[{regime_index}]",
        )
        require(
            f"{prefix}standard__trace__called_frequency_hz",
            "reader_alias",
            "synthesis__edge__selected_sample_frequency_hz",
        )
        require(
            f"{prefix}counterfactual__signed_edge_original",
            "reader_alias",
            "synthesis__edge__signed_edge_frequency_hz",
        )
        for suffix, standard_quantity in COUNTERFACTUAL_STANDARD_OUTPUT_ALIASES:
            require(
                f"{prefix}counterfactual__{suffix}",
                "reader_alias",
                f"synthesis__standard__{standard_quantity}[{regime_index}]",
            )

    for field in EXTENSION_INVARIANT_FIELDS:
        if field in {"frequencies_hz", "coulomb_table_energy_first"}:
            continue
        member = (
            "evidence__synthesis__extension__input__invariant__" + field
        )
        if member not in arrays:
            raise PublisherSchemaError(
                f"coalesced extension invariant is missing: {member}"
            )
        for regime in REGIME_NAMES:
            require(
                f"{regime}__synthesis__extension__input__invariant__{field}",
                "integration",
                member,
            )

    signed_member = (
        "evidence__synthesis__counterfactual__signed_edge_flipped"
    )
    if signed_member not in arrays:
        raise PublisherSchemaError("coalesced signed-edge evidence is missing")
    for regime in REGIME_NAMES:
        require(
            f"{regime}__synthesis__counterfactual__signed_edge_flipped",
            "integration",
            signed_member,
        )

    for field in STANDARD_TRACE_FIELDS:
        if field == "called_frequency_hz":
            continue
        member = f"evidence__synthesis__standard__trace__{field}"
        if member not in arrays:
            raise PublisherSchemaError(
                f"coalesced standard trace is missing: {member}"
            )
        for regime in REGIME_NAMES:
            require(
                f"{regime}__synthesis__standard__trace__{field}",
                "integration",
                member,
            )


def _validate_integration_semantics(
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Validate integration-only grid, reader, and inventory metadata."""

    if tuple(arrays["grid_bank__policy_label"].tolist()) != GRID_POLICY_LABELS:
        raise PublisherSchemaError("integration grid-policy labels changed")
    if not np.array_equal(
        arrays["grid_bank__policy_index"], np.arange(5, dtype=np.int64)
    ):
        raise PublisherSchemaError("integration grid-policy indices changed")
    boundary_selector = np.asarray(
        arrays["sampling_boundary__grid_bank_index"], dtype=np.int64
    )
    if not np.array_equal(
        boundary_selector,
        np.asarray([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int64),
    ):
        raise PublisherSchemaError("integration boundary-grid selectors changed")
    regime_selector = np.asarray(
        arrays["atmosphere_product__grid_bank_index"], dtype=np.int64
    )
    if not np.array_equal(
        regime_selector, np.asarray([4, 1, 1, 0], dtype=np.int64)
    ):
        raise PublisherSchemaError("integration regime-grid selectors changed")
    light_speed = float(_scalar(arrays["grid_bank__light_speed_nm_per_s"]))
    if light_speed != float(LIGHT_SPEED_NM_PER_S):
        raise PublisherSchemaError("integration light-speed identity changed")
    expected_frequency_digests = []
    for wavelength in np.asarray(arrays["grid_bank__wavelength_nm"]):
        frequency = light_speed / np.maximum(wavelength, 1.0e-300)
        expected_frequency_digests.append(
            hashlib.sha256(frequency.tobytes()).hexdigest()
        )
    if tuple(arrays["grid_bank__derived_frequency_sha256"].tolist()) != tuple(
        expected_frequency_digests
    ):
        raise PublisherSchemaError("integration frequency digests changed")
    if _scalar(arrays["meta__reader_archive_schema_digest"]) != (
        ACCEPTED_READER_SCHEMA_DIGEST
    ):
        raise PublisherSchemaError("integration reader-schema identity changed")
    if arrays["inventory__raw_member_name"].shape != (ACCEPTED_RAW_KEY_COUNT,):
        raise PublisherSchemaError("integration raw inventory length changed")
    names = arrays["inventory__raw_member_name"].tolist()
    if names != sorted(names):
        raise PublisherSchemaError("integration raw inventory is not sorted")
    _validate_deduplicated_inventory(arrays)
    actual_inventory_digest = inventory_mapping_digest(arrays)
    if actual_inventory_digest != ACCEPTED_INVENTORY_MAPPING_DIGEST:
        raise PublisherSchemaError(
            "integration inventory ownership digest is "
            f"{actual_inventory_digest}; expected "
            f"{ACCEPTED_INVENTORY_MAPPING_DIGEST}"
        )


def validate_final_payload(
    name: str,
    arrays: Mapping[str, np.ndarray],
    *,
    reader_sha256: str | None = None,
) -> None:
    """Validate one final archive's identity, shape, and frozen schema."""

    copied = _copy_mapping(arrays)
    expected_kind = {
        READER_NAME: "continuum_reader",
        INTEGRATION_NAME: "continuum_integration",
    }[name]
    required = {
        "meta__archive_kind",
        "meta__archive_schema_version",
        "meta__archive_schema_digest",
        "meta__archive_key_count",
        "meta__payne_zero_commit",
        "meta__worker_sha256",
        "meta__capture_contract_sha256",
        "meta__exact_source_contract_sha256",
        "meta__publisher_contract_sha256",
        "meta__publisher_sha256",
        "meta__deterministic_npz_sha256",
        "meta__oracle_acceptance_sha256",
        "meta__fixture_sha256",
        "meta__fixture_payload_digest",
        "meta__raw_capture_schema_digest",
        "meta__accepted_physical_payload_fingerprint",
        "meta__accepted_full_capture_fingerprint",
    }
    missing = sorted(required - set(copied))
    if missing:
        raise PublisherSchemaError(f"{name} lacks metadata {missing}")
    if _scalar(copied["meta__archive_kind"]) != expected_kind:
        raise PublisherSchemaError(f"{name} has the wrong archive kind")
    if int(_scalar(copied["meta__archive_schema_version"])) != 1:
        raise PublisherSchemaError(f"{name} has the wrong archive version")
    if int(_scalar(copied["meta__archive_key_count"])) != len(copied):
        raise PublisherSchemaError(f"{name} has the wrong member count metadata")
    actual_schema = schema_digest(copied)
    if _scalar(copied["meta__archive_schema_digest"]) != actual_schema:
        raise PublisherSchemaError(f"{name} schema digest does not recompute")
    exact_metadata = {
        "meta__payne_zero_commit": PINNED_COMMIT,
        "meta__worker_sha256": ACCEPTED_INPUT_SHA256["worker"],
        "meta__capture_contract_sha256": ACCEPTED_INPUT_SHA256[
            "capture_contract"
        ],
        "meta__exact_source_contract_sha256": ACCEPTED_INPUT_SHA256[
            "exact_source_contract"
        ],
        "meta__publisher_contract_sha256": ACCEPTED_INPUT_SHA256[
            "publisher_contract"
        ],
        "meta__publisher_sha256": _canonical_publisher_sha256(),
        "meta__deterministic_npz_sha256": ACCEPTED_INPUT_SHA256[
            "deterministic_npz"
        ],
        "meta__oracle_acceptance_sha256": ACCEPTED_INPUT_SHA256[
            "oracle_acceptance"
        ],
        "meta__fixture_sha256": ACCEPTED_INPUT_SHA256["fixture"],
        "meta__fixture_payload_digest": ACCEPTED_FIXTURE_PAYLOAD_DIGEST,
        "meta__raw_capture_schema_digest": ACCEPTED_RAW_SCHEMA_DIGEST,
        "meta__accepted_physical_payload_fingerprint": (
            ACCEPTED_PHYSICAL_FINGERPRINT
        ),
        "meta__accepted_full_capture_fingerprint": ACCEPTED_FULL_FINGERPRINT,
    }
    for member, expected in exact_metadata.items():
        if _scalar(copied[member]) != expected:
            raise PublisherSchemaError(f"{name} has the wrong {member}")
    _validate_common_final_semantics(name, copied)

    if name == READER_NAME:
        if any(
            np.asarray(value).ndim > 0 and 30000 in np.asarray(value).shape
            for value in copied.values()
        ):
            raise PublisherSchemaError("reader archive contains a 30000-point array")
        expected_count = ACCEPTED_READER_KEY_COUNT
        expected_schema = ACCEPTED_READER_SCHEMA_DIGEST
        if copied["line_reference__threshold"].shape != (4, 6, 344):
            raise PublisherSchemaError("reader line-reference shape changed")
        if copied["line_reference__threshold"].dtype != np.float32:
            raise PublisherSchemaError("reader line-reference dtype changed")
        _validate_reader_semantics(copied)
    else:
        expected_count = ACCEPTED_INTEGRATION_KEY_COUNT
        expected_schema = ACCEPTED_INTEGRATION_SCHEMA_DIGEST
        for quantity in ("absorption", "scattering", "source"):
            member = f"atmosphere_product__{quantity}"
            if copied[member].shape != (4, 6, 30000):
                raise PublisherSchemaError(f"integration {member} shape changed")
        if copied["grid_bank__wavelength_nm"].shape != (5, 30000):
            raise PublisherSchemaError("integration wavelength bank changed")
        if copied["grid_bank__frequency_weight_hz"].shape != (5, 30000):
            raise PublisherSchemaError("integration weight bank changed")
        if not bool(
            _scalar(copied["meta__logical_raw_capture_coverage_complete"])
        ):
            raise PublisherSchemaError("integration logical coverage is false")
        if reader_sha256 is None:
            raise PublisherSchemaError("integration validated without reader hash")
        if _scalar(copied["meta__reader_archive_sha256"]) != reader_sha256:
            raise PublisherSchemaError("integration reader identity changed")
        _validate_integration_semantics(copied)

    if expected_count is not None and len(copied) != expected_count:
        raise PublisherSchemaError(
            f"{name} has {len(copied)} members; expected {expected_count}"
        )
    if expected_schema is not None and actual_schema != expected_schema:
        raise PublisherSchemaError(
            f"{name} schema is {actual_schema}; expected {expected_schema}"
        )


def _resolve_published_member(
    published: str,
    reader: Mapping[str, np.ndarray],
    integration: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Resolve one inventory descriptor back to its exact raw array."""

    if "[" not in published:
        if published in reader:
            return np.asarray(reader[published]).copy()
        if published in integration:
            return np.asarray(integration[published]).copy()
        raise PublisherSchemaError(
            f"logical inventory points to missing member {published!r}"
        )
    if not published.endswith("]"):
        raise PublisherSchemaError(
            f"logical inventory has malformed descriptor {published!r}"
        )
    base, selector_text = published[:-1].split("[", 1)
    if selector_text == "sampling_boundary__grid_bank_index":
        selector = np.asarray(
            integration["sampling_boundary__grid_bank_index"], dtype=np.int64
        )
        return np.asarray(integration[base])[selector].copy()
    if selector_text.startswith("atmosphere_product__grid_bank_index["):
        regime_index = int(
            selector_text.removeprefix(
                "atmosphere_product__grid_bank_index["
            ).removesuffix("]")
        )
        grid_index = int(
            integration["atmosphere_product__grid_bank_index"][regime_index]
        )
        if base == "grid_bank__derived_frequency":
            wavelength = np.asarray(integration["grid_bank__wavelength_nm"])[
                grid_index
            ]
            light_speed = float(
                _scalar(integration["grid_bank__light_speed_nm_per_s"])
            )
            if light_speed != float(LIGHT_SPEED_NM_PER_S):
                raise PublisherSchemaError(
                    "logical reconstruction received the wrong light speed"
                )
            return (
                light_speed / np.maximum(wavelength, 1.0e-300)
            ).copy()
        return np.asarray(integration[base])[grid_index].copy()

    payload = reader if base in reader else integration
    if base not in payload:
        raise PublisherSchemaError(
            f"logical inventory points to missing sliced member {base!r}"
        )
    values = np.asarray(payload[base])
    selectors = selector_text.split(",")
    if selectors[-1] == "valid":
        regime_index = int(selectors[0])
        mask = np.asarray(reader["line_reference__active_valid"])[regime_index]
        if values.ndim == 2:
            return values[regime_index][mask].copy()
        if values.ndim == 3:
            return values[regime_index][:, mask].copy()
        raise PublisherSchemaError(
            f"valid-mask descriptor has unsupported rank: {published}"
        )
    indices = tuple(int(selector) for selector in selectors)
    return np.asarray(values[indices]).copy()


def reconstruct_logical_raw_capture(
    reader: Mapping[str, np.ndarray],
    integration: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rebuild all 1,161 accepted raw members from the published pair."""

    names = np.asarray(integration["inventory__raw_member_name"]).tolist()
    dispositions = np.asarray(integration["inventory__disposition"]).tolist()
    published = np.asarray(integration["inventory__published_member"]).tolist()
    if not (len(names) == len(dispositions) == len(published)):
        raise PublisherSchemaError("logical inventory vector lengths differ")
    if len(names) != len(set(names)):
        raise PublisherSchemaError("logical inventory contains duplicate raw names")
    allowed = {
        "reader",
        "integration",
        "coalesced_grid_bank",
        "reader_alias",
        "common_identity_metadata",
        "publisher_control_metadata",
    }
    unknown = sorted(set(dispositions) - allowed)
    if unknown:
        raise PublisherSchemaError(
            f"logical inventory has unknown dispositions: {unknown}"
        )
    expected_aliases: dict[str, str] = {}
    for raw_name, disposition, member in zip(
        names, dispositions, published
    ):
        raw_name = str(raw_name)
        disposition = str(disposition)
        member = str(member)
        base = member.split("[", 1)[0]
        if disposition in {"reader", "reader_alias"}:
            if base not in reader:
                raise PublisherSchemaError(
                    f"{raw_name} is reader-owned but points to {base!r}"
                )
            expected_aliases[
                f"alias__{raw_name}__reader_member"
            ] = member
        elif disposition == "common_identity_metadata":
            if base not in reader or base not in integration:
                raise PublisherSchemaError(
                    f"{raw_name} common identity is not present in both archives"
                )
            if not np.array_equal(reader[base], integration[base]):
                raise PublisherSchemaError(
                    f"{raw_name} common identity differs between archives"
                )
        elif disposition in {
            "integration",
            "coalesced_grid_bank",
            "publisher_control_metadata",
        }:
            if base != "grid_bank__derived_frequency" and base not in integration:
                raise PublisherSchemaError(
                    f"{raw_name} is integration-owned but points to {base!r}"
                )
        else:
            raise PublisherSchemaError(
                f"{raw_name} has unsupported disposition {disposition!r}"
            )
    actual_aliases = {
        name
        for name in integration
        if name.startswith("alias__") and name.endswith("__reader_member")
    }
    if actual_aliases != set(expected_aliases):
        missing = sorted(set(expected_aliases) - actual_aliases)
        extra = sorted(actual_aliases - set(expected_aliases))
        raise PublisherSchemaError(
            f"reader alias set changed; missing={missing}, extra={extra}"
        )
    for alias_name, expected in expected_aliases.items():
        if _scalar(integration[alias_name]) != expected:
            raise PublisherSchemaError(
                f"reader alias target changed: {alias_name}"
            )
    reconstructed = {
        str(name): _resolve_published_member(
            str(member), reader, integration
        )
        for name, member in zip(names, published)
    }
    return _copy_mapping(reconstructed)


def assemble_capture(raw_path: Path, final_dir: Path) -> None:
    """Assemble and validate one deterministic two-artifact directory."""

    raw = load_npz(raw_path)
    validate_raw_capture(raw)
    reader, integration = assemble_payloads(raw, validate_raw=False)
    final_dir.mkdir(parents=True, exist_ok=False)
    reader_path = final_dir / READER_NAME
    write_npz(reader_path, reader)
    reader_hash = sha256(reader_path)
    integration["meta__reader_archive_sha256"] = np.asarray(reader_hash)
    # The value changed, not the schema; its accepted schema digest is stable.
    write_npz(final_dir / INTEGRATION_NAME, integration)
    validate_final_directory(final_dir, enforce_size=True)


def validate_final_directory(
    directory: Path,
    *,
    enforce_size: bool = True,
    validate_logical_raw: bool = True,
) -> None:
    """Require exactly two independently valid Chapter 5 archives."""

    actual = tuple(sorted(path.name for path in directory.iterdir()))
    expected = tuple(sorted(OUTPUT_NAMES))
    if actual != expected:
        raise PublisherSchemaError(
            f"final archive set is {actual}; expected {expected}"
        )
    reader_path = directory / READER_NAME
    integration_path = directory / INTEGRATION_NAME
    reader = load_npz(reader_path)
    integration = load_npz(integration_path)
    validate_final_payload(READER_NAME, reader)
    validate_final_payload(
        INTEGRATION_NAME,
        integration,
        reader_sha256=sha256(reader_path),
    )
    if validate_logical_raw:
        reconstructed = reconstruct_logical_raw_capture(reader, integration)
        validate_raw_capture(reconstructed)
    if enforce_size and reader_path.stat().st_size > READER_SIZE_LIMIT_BYTES:
        raise PublisherSchemaError("reader archive exceeds its 4 MiB limit")
    if (
        enforce_size
        and integration_path.stat().st_size > INTEGRATION_SIZE_LIMIT_BYTES
    ):
        raise PublisherSchemaError("integration archive exceeds its 32 MiB limit")


def _child_environment(cache_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(PROCESS_CONTROLS)
    environment.update(
        {
            "NUMBA_CACHE_DIR": str(cache_dir.resolve()),
            "PAYNE_ZERO_DATA_ROOT": str(PINNED_DATA_ROOT.resolve()),
            "PYTHONPATH": os.pathsep.join(
                (str(PINNED_ROOT.resolve()), str(REPOSITORY_ROOT.resolve()))
            ),
        }
    )
    return environment


def _require_fresh_cache_target(cache_dir: Path) -> None:
    unresolved = Path(cache_dir)
    if unresolved.is_symlink():
        raise RuntimeError("capture cache path must not be a symlink")
    if unresolved.exists():
        if any(unresolved.iterdir()):
            raise RuntimeError("capture cache path is populated")
        raise RuntimeError("publisher requires an absent cache path")
    if not unresolved.parent.is_dir():
        raise RuntimeError("capture cache parent does not exist")


def run_capture_child(
    raw_path: Path,
    cache_dir: Path,
    *,
    timeout_seconds: int = 1800,
) -> None:
    """Launch one accepted worker in a fresh deterministic process."""

    _require_fresh_cache_target(cache_dir)
    if raw_path.resolve().is_relative_to(REPOSITORY_ROOT.resolve()):
        raise RuntimeError("temporary raw capture must be outside the repository")
    publisher_path = _canonical_publisher_path()
    subprocess.run(
        [
            sys.executable,
            str(publisher_path),
            "--internal-capture",
            "--internal-output",
            str(raw_path),
        ],
        cwd=REPOSITORY_ROOT,
        env=_child_environment(cache_dir),
        check=True,
        timeout=timeout_seconds,
    )


CaptureRunner = Callable[[Path, Path], None]
Assembler = Callable[[Path, Path], None]


def build_capture_set(
    capture_root: Path,
    *,
    capture_runner: CaptureRunner = run_capture_child,
    assembler: Assembler = assemble_capture,
) -> None:
    """Build one raw capture and one independently assembled final pair."""

    raw_dir = capture_root / "raw"
    cache_root = capture_root / "cache"
    final_dir = capture_root / "final"
    raw_dir.mkdir(parents=True, exist_ok=False)
    cache_root.mkdir(parents=True, exist_ok=False)
    raw_path = raw_dir / RAW_NAME
    cache_path = cache_root / "numba"
    capture_runner(raw_path, cache_path)
    assembler(raw_path, final_dir)


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


def _read_canonical_regular_file(path: Path, *, label: str) -> bytes:
    """Read one absolute canonical regular file without following symlinks."""

    candidate = Path(path)
    if not candidate.is_absolute():
        raise PublicationAcceptanceError(f"{label} path is not absolute")
    try:
        resolved = candidate.resolve(strict=True)
        before = candidate.lstat()
    except (FileNotFoundError, OSError) as error:
        raise PublicationAcceptanceError(f"{label} is absent or unreadable") from error
    if resolved != candidate:
        raise PublicationAcceptanceError(f"{label} path is not canonical")
    if not stat.S_ISREG(before.st_mode):
        raise PublicationAcceptanceError(f"{label} is not a regular file")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise PublicationAcceptanceError(
            f"{label} could not be opened without following symlinks"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise PublicationAcceptanceError(f"{label} is not a regular file")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise PublicationAcceptanceError(f"{label} changed while opening")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(descriptor)

    try:
        after = candidate.lstat()
    except OSError as error:
        raise PublicationAcceptanceError(f"{label} changed while reading") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(payload) != after.st_size:
        raise PublicationAcceptanceError(f"{label} changed while reading")
    return payload


def _canonical_publisher_path() -> Path:
    """Require execution from the one reviewed canonical publisher path."""

    expected = Path(PUBLISHER_PATH)
    if not expected.is_absolute():
        raise PublicationAcceptanceError("canonical publisher path is not absolute")
    try:
        executing = Path(__file__).resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PublicationAcceptanceError(
            "executing Chapter 5 publisher path is absent or unreadable"
        ) from error
    if executing != expected:
        raise PublicationAcceptanceError(
            "executing Chapter 5 publisher is not the canonical publisher path"
        )
    _read_canonical_regular_file(expected, label="Chapter 5 publisher")
    return expected


def _canonical_publisher_bytes() -> bytes:
    """Read the reviewed publisher only through its canonical repository path."""

    path = _canonical_publisher_path()
    return _read_canonical_regular_file(path, label="Chapter 5 publisher")


def _canonical_publisher_sha256() -> str:
    """Hash the canonical publisher bytes used by metadata and authorization."""

    return hashlib.sha256(_canonical_publisher_bytes()).hexdigest()


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    """Decode one duplicate-free, finite JSON object."""

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PublicationAcceptanceError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PublicationAcceptanceError(
            f"{label} contains non-finite JSON constant {value}"
        )

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_nonfinite,
        )
    except PublicationAcceptanceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationAcceptanceError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise PublicationAcceptanceError(f"{label} must contain one JSON object")
    return value


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublicationAcceptanceError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise PublicationAcceptanceError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _require_sha256_hex(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationAcceptanceError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_plain_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PublicationAcceptanceError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _artifact_acceptance_specifications() -> dict[str, dict[str, Any]]:
    """Return cycle-free schema expectations for the two accepted candidates."""

    return {
        "reader": {
            "name": READER_NAME,
            "relative_path": OUTPUT_RELATIVE_PATHS[READER_NAME],
            "archive_kind": "continuum_reader",
            "archive_schema_version": 1,
            "key_count": ACCEPTED_READER_KEY_COUNT,
            "schema_digest": ACCEPTED_READER_SCHEMA_DIGEST,
            "size_limit": READER_SIZE_LIMIT_BYTES,
        },
        "integration": {
            "name": INTEGRATION_NAME,
            "relative_path": OUTPUT_RELATIVE_PATHS[INTEGRATION_NAME],
            "archive_kind": "continuum_integration",
            "archive_schema_version": 1,
            "key_count": ACCEPTED_INTEGRATION_KEY_COUNT,
            "schema_digest": ACCEPTED_INTEGRATION_SCHEMA_DIGEST,
            "size_limit": INTEGRATION_SIZE_LIMIT_BYTES,
        },
    }


def _validate_prepublication_manifest(
    acceptance: dict[str, Any],
) -> None:
    """Bind authorization to one reviewed manifest state with no Chapter 5 entry."""

    expected_paths = [OUTPUT_RELATIVE_PATHS[name] for name in OUTPUT_NAMES]
    if acceptance["path"] != MANIFEST_RELATIVE_PATH:
        raise PublicationAcceptanceError("acceptance record has the wrong manifest path")
    expected_manifest_sha = _require_sha256_hex(
        acceptance["prepublication_sha256"],
        label="manifest prepublication SHA-256",
    )
    if type(acceptance["chapter05_entries_present"]) is not bool:
        raise PublicationAcceptanceError(
            "manifest chapter05_entries_present must be boolean"
        )
    if acceptance["chapter05_entries_present"]:
        raise PublicationAcceptanceError(
            "acceptance record must describe a prepublication manifest"
        )
    if acceptance["chapter05_golden_paths"] != expected_paths:
        raise PublicationAcceptanceError(
            "acceptance record has the wrong Chapter 5 manifest paths"
        )
    if acceptance["payne_zero_commit"] != PINNED_COMMIT:
        raise PublicationAcceptanceError(
            "acceptance record has the wrong manifest Payne Zero commit"
        )

    manifest_bytes = _read_canonical_regular_file(
        MANIFEST_PATH,
        label="data/MANIFEST.json",
    )
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_sha != expected_manifest_sha:
        raise PublicationAcceptanceError(
            "data/MANIFEST.json differs from the accepted prepublication identity"
        )
    manifest = _strict_json_object(manifest_bytes, label="data/MANIFEST.json")
    manifest_schema_version = _require_plain_int(
        manifest.get("schema_version"),
        label="data/MANIFEST.json schema version",
        minimum=1,
    )
    if manifest_schema_version != acceptance["schema_version"]:
        raise PublicationAcceptanceError(
            "data/MANIFEST.json schema version disagrees with acceptance"
        )
    if manifest.get("payne_zero_commit") != PINNED_COMMIT:
        raise PublicationAcceptanceError(
            "data/MANIFEST.json Payne Zero commit disagrees with acceptance"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise PublicationAcceptanceError("data/MANIFEST.json entries must be a list")
    target_prefix = "data/golden/payne_zero/chapter05/"
    chapter_entries: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PublicationAcceptanceError(
                f"data/MANIFEST.json entry {index} is not an object"
            )
        path = entry.get("path")
        if not isinstance(path, str):
            raise PublicationAcceptanceError(
                f"data/MANIFEST.json entry {index} has no string path"
            )
        if path.startswith(target_prefix) or path in expected_paths:
            chapter_entries.append(path)
    if chapter_entries:
        raise PublicationAcceptanceError(
            "data/MANIFEST.json already contains Chapter 5 publication entries"
        )


def load_publication_acceptance_record() -> dict[str, Any]:
    """Load and fully validate the detached, cycle-free publication record."""

    if PUBLICATION_ACCEPTANCE_RECORD_PATH.is_relative_to(OUTPUT_DIR):
        raise PublicationAcceptanceError(
            "publication acceptance record must remain outside the golden directory"
        )
    record_bytes = _read_canonical_regular_file(
        PUBLICATION_ACCEPTANCE_RECORD_PATH,
        label="Chapter 5 publication acceptance record",
    )
    record = _strict_json_object(
        record_bytes,
        label="Chapter 5 publication acceptance record",
    )
    _require_exact_keys(
        record,
        {
            "schema_version",
            "record_kind",
            "publisher",
            "publisher_contract",
            "manifest",
            "artifacts",
        },
        label="publication acceptance record",
    )
    if (
        _require_plain_int(
            record["schema_version"],
            label="publication acceptance schema version",
            minimum=1,
        )
        != PUBLICATION_ACCEPTANCE_SCHEMA_VERSION
    ):
        raise PublicationAcceptanceError(
            "publication acceptance schema version is unsupported"
        )
    if record["record_kind"] != PUBLICATION_ACCEPTANCE_KIND:
        raise PublicationAcceptanceError("publication acceptance kind is wrong")

    publisher_identity = _require_exact_keys(
        record["publisher"],
        {"path", "sha256"},
        label="publisher identity",
    )
    if publisher_identity["path"] != PUBLISHER_RELATIVE_PATH:
        raise PublicationAcceptanceError("acceptance record has the wrong publisher path")
    accepted_publisher_sha = _require_sha256_hex(
        publisher_identity["sha256"],
        label="accepted publisher SHA-256",
    )
    publisher_bytes = _canonical_publisher_bytes()
    if hashlib.sha256(publisher_bytes).hexdigest() != accepted_publisher_sha:
        raise PublicationAcceptanceError(
            "current publisher differs from the accepted publisher identity"
        )

    contract_identity = _require_exact_keys(
        record["publisher_contract"],
        {"path", "sha256"},
        label="publisher contract identity",
    )
    if contract_identity["path"] != PUBLISHER_CONTRACT_RELATIVE_PATH:
        raise PublicationAcceptanceError(
            "acceptance record has the wrong publisher contract path"
        )
    accepted_contract_sha = _require_sha256_hex(
        contract_identity["sha256"],
        label="accepted publisher contract SHA-256",
    )
    contract_bytes = _read_canonical_regular_file(
        PUBLISHER_CONTRACT_PATH,
        label="Chapter 5 publisher contract",
    )
    actual_contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    if (
        actual_contract_sha != accepted_contract_sha
        or actual_contract_sha != ACCEPTED_INPUT_SHA256["publisher_contract"]
    ):
        raise PublicationAcceptanceError(
            "publisher contract identity disagrees with publisher or acceptance"
        )

    manifest_acceptance = _require_exact_keys(
        record["manifest"],
        {
            "path",
            "prepublication_sha256",
            "schema_version",
            "payne_zero_commit",
            "chapter05_golden_paths",
            "chapter05_entries_present",
        },
        label="manifest acceptance",
    )
    _require_plain_int(
        manifest_acceptance["schema_version"],
        label="accepted manifest schema version",
        minimum=1,
    )
    _validate_prepublication_manifest(manifest_acceptance)

    artifacts = _require_exact_keys(
        record["artifacts"],
        {"reader", "integration"},
        label="artifact acceptance",
    )
    for role, expected in _artifact_acceptance_specifications().items():
        artifact = _require_exact_keys(
            artifacts[role],
            {
                "name",
                "relative_path",
                "sha256",
                "bytes",
                "archive_kind",
                "archive_schema_version",
                "key_count",
                "schema_digest",
            },
            label=f"{role} artifact acceptance",
        )
        for field in (
            "name",
            "relative_path",
            "archive_kind",
            "schema_digest",
        ):
            if artifact[field] != expected[field]:
                raise PublicationAcceptanceError(
                    f"{role} artifact {field} disagrees with the accepted schema"
                )
        for field in ("archive_schema_version", "key_count"):
            actual_integer = _require_plain_int(
                artifact[field],
                label=f"{role} artifact {field}",
                minimum=1,
            )
            if actual_integer != expected[field]:
                raise PublicationAcceptanceError(
                    f"{role} artifact {field} disagrees with the accepted schema"
                )
        _require_sha256_hex(
            artifact["schema_digest"],
            label=f"{role} artifact schema digest",
        )
        _require_sha256_hex(
            artifact["sha256"],
            label=f"{role} artifact SHA-256",
        )
        size = _require_plain_int(
            artifact["bytes"],
            label=f"{role} artifact byte size",
            minimum=1,
        )
        if size > expected["size_limit"]:
            raise PublicationAcceptanceError(
                f"{role} artifact exceeds its accepted size limit"
            )
    return record


def publication_gate_ready() -> bool:
    """Return whether the detached record and all bound identities validate."""

    try:
        load_publication_acceptance_record()
    except PublicationAcceptanceError:
        return False
    return True


def _require_publication_gate() -> dict[str, Any]:
    try:
        return load_publication_acceptance_record()
    except PublicationAcceptanceError as error:
        raise PublicationAcceptanceError(
            "Chapter 5 publication is disabled: detached acceptance, publisher, "
            "candidate, or prepublication manifest identity is unavailable"
        ) from error


def _candidate_records_from_directory(
    directory: Path,
) -> dict[str, dict[str, Any]]:
    """Hash an exact canonical pair while rejecting symlink/nonregular members."""

    root = Path(directory)
    if not root.is_absolute():
        raise PublicationAcceptanceError("candidate directory path is not absolute")
    try:
        resolved = root.resolve(strict=True)
        root_metadata = root.lstat()
    except (FileNotFoundError, OSError) as error:
        raise PublicationAcceptanceError("candidate directory is absent") from error
    if resolved != root or root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise PublicationAcceptanceError(
            "candidate directory is not a canonical regular directory"
        )
    actual = tuple(sorted(path.name for path in root.iterdir()))
    expected = tuple(sorted(OUTPUT_NAMES))
    if actual != expected:
        raise PublicationAcceptanceError(
            f"candidate archive set is {actual}; expected {expected}"
        )
    records: dict[str, dict[str, Any]] = {}
    for name in OUTPUT_NAMES:
        payload = _read_canonical_regular_file(
            root / name,
            label=f"candidate artifact {name}",
        )
        records[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return records


def validate_candidate_against_acceptance(
    directory: Path,
    acceptance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Require exact accepted candidate bytes and sizes for both artifacts."""

    records = _candidate_records_from_directory(directory)
    artifacts = acceptance["artifacts"]
    for role, expected in _artifact_acceptance_specifications().items():
        name = expected["name"]
        accepted = artifacts[role]
        if records[name] != {
            "sha256": accepted["sha256"],
            "bytes": accepted["bytes"],
        }:
            raise PublicationAcceptanceError(
                f"{role} candidate bytes differ from detached acceptance"
            )
    return records


def _require_canonical_publication_destination(destination: Path) -> Path:
    """Require the one canonical repository destination before any write."""

    expected = Path(OUTPUT_DIR)
    requested = Path(destination)
    if (
        not expected.is_absolute()
        or expected.resolve(strict=False) != expected
    ):
        raise PublicationAcceptanceError(
            "configured Chapter 5 publication destination is not canonical"
        )
    if requested != expected:
        raise PublicationAcceptanceError(
            "Chapter 5 publication destination must be the canonical OUTPUT_DIR"
        )
    return expected


def _atomic_rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing any existing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            -2,  # Darwin AT_FDCWD
            source_bytes,
            -2,
            destination_bytes,
            0x00000004,  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            -100,  # Linux AT_FDCWD
            source_bytes,
            -100,
            destination_bytes,
            1,  # RENAME_NOREPLACE
        )
    else:
        raise RuntimeError(
            "this platform lacks the reviewed atomic no-replace rename primitive"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number, os.strerror(error_number), str(destination)
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def publish_verified_directory(
    source: Path,
    destination: Path = OUTPUT_DIR,
) -> str:
    """Reauthorize and atomically publish one exact pair without replacement."""

    destination = _require_canonical_publication_destination(destination)
    acceptance = _require_publication_gate()
    verify_static_identity()
    validate_final_directory(source)
    validate_candidate_against_acceptance(source, acceptance)
    destination_parent = destination.parent
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink():
            raise FileExistsError(
                f"publication target is not a directory: {destination}"
            )
        acceptance = _require_publication_gate()
        verify_static_identity()
        validate_final_directory(destination)
        validate_candidate_against_acceptance(destination, acceptance)
        compare_file_sets(
            source,
            destination,
            OUTPUT_NAMES,
            label="existing Chapter 5 publication",
        )
        return "identical-existing"

    destination_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".chapter05-stage-",
        dir=destination_parent,
    ) as temporary:
        stage_parent = Path(temporary)
        stage = stage_parent / "chapter05"
        shutil.copytree(source, stage)
        validate_final_directory(stage)
        for name in OUTPUT_NAMES:
            with (stage / name).open("rb") as handle:
                os.fsync(handle.fileno())
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        verify_static_identity()
        acceptance = _require_publication_gate()
        validate_candidate_against_acceptance(stage, acceptance)
        try:
            _atomic_rename_directory_no_replace(stage, destination)
        except FileExistsError:
            if not destination.is_dir() or destination.is_symlink():
                raise
            acceptance = _require_publication_gate()
            verify_static_identity()
            validate_final_directory(destination)
            validate_candidate_against_acceptance(destination, acceptance)
            compare_file_sets(
                stage,
                destination,
                OUTPUT_NAMES,
                label="raced Chapter 5 publication",
            )
            return "identical-existing"
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
    publish: bool,
    destination: Path = OUTPUT_DIR,
    capture_builder: CaptureBuilder = build_capture_set,
    publisher: Publisher = publish_verified_directory,
) -> dict[str, Any]:
    """Build two complete candidates, compare bytes, then optionally publish."""

    acceptance: dict[str, Any] | None = None
    if publish:
        destination = _require_canonical_publication_destination(destination)
        acceptance = _require_publication_gate()
    verify_static_identity()
    with (
        tempfile.TemporaryDirectory(prefix="chapter05-capture-a-") as first_tmp,
        tempfile.TemporaryDirectory(prefix="chapter05-capture-b-") as second_tmp,
    ):
        first = Path(first_tmp).resolve()
        second = Path(second_tmp).resolve()
        capture_builder(first)
        capture_builder(second)
        compare_file_sets(
            first / "raw",
            second / "raw",
            (RAW_NAME,),
            label="raw capture",
        )
        compare_file_sets(
            first / "final",
            second / "final",
            OUTPUT_NAMES,
            label="assembled capture",
        )
        records = {
            name: {
                "sha256": sha256(first / "final" / name),
                "bytes": (first / "final" / name).stat().st_size,
            }
            for name in OUTPUT_NAMES
        }
        if publish:
            if acceptance is None:
                raise AssertionError("publication acceptance was not loaded")
            accepted_records = validate_candidate_against_acceptance(
                first / "final",
                acceptance,
            )
            if records != accepted_records:
                raise PublicationAcceptanceError(
                    "post-capture candidate summary changed during validation"
                )
            if publisher is not publish_verified_directory:
                raise PublicationAcceptanceError(
                    "publication mode forbids an injected publication function"
                )
        status = (
            publisher(first / "final", destination)
            if publish
            else "verified-only"
        )
    return {"status": status, "archives": records}


def _capture_internal(output: Path) -> None:
    """Serialize one temporary accepted worker result outside the repository."""

    verify_static_identity()
    resolved = output.expanduser().resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise RuntimeError("internal raw output must be outside the repository")
    if resolved.exists():
        raise FileExistsError(f"internal raw output already exists: {resolved}")
    if resolved.suffix != ".npz" or not resolved.parent.is_dir():
        raise RuntimeError("internal raw output must be an NPZ in an existing directory")
    arrays = oracle_worker.build_oracle_results(fixture_path=FIXTURE_PATH)
    validate_raw_capture(arrays)
    write_npz(resolved, _copy_mapping(arrays))
    reloaded = load_npz(resolved)
    validate_raw_capture(reloaded)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "verify Chapter 5 publisher identity or explicitly run a full "
            "deterministic double capture"
        )
    )
    parser.add_argument(
        "--full-capture",
        action="store_true",
        help="run the expensive two-process scientific capture and assembly",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--verify-only",
        action="store_true",
        help="with --full-capture, compare both candidates without publishing",
    )
    action.add_argument(
        "--publish",
        action="store_true",
        help="with --full-capture, atomically publish the reviewed pair",
    )
    parser.add_argument("--internal-capture", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-output", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.internal_capture:
        if arguments.internal_output is None:
            parser.error("--internal-capture requires --internal-output")
        if arguments.full_capture or arguments.publish or arguments.verify_only:
            parser.error("internal capture cannot be combined with public actions")
        return arguments
    if arguments.internal_output is not None:
        parser.error("--internal-output is private to --internal-capture")
    if arguments.publish and not arguments.full_capture:
        parser.error("--publish requires --full-capture")
    if arguments.publish and not publication_gate_ready():
        parser.error(
            "--publish is disabled pending reviewed publication identities"
        )
    if arguments.verify_only and not arguments.full_capture:
        parser.error("--verify-only requires --full-capture")
    if arguments.full_capture and not (arguments.publish or arguments.verify_only):
        parser.error("--full-capture requires --verify-only or --publish")
    return arguments


def main() -> None:
    arguments = parse_args()
    if arguments.internal_capture:
        _capture_internal(arguments.internal_output)
        return
    if not arguments.full_capture:
        print(json.dumps({"status": "identity-only", **verify_static_identity()}, sort_keys=True))
        return
    result = generate_and_maybe_publish(
        publish=bool(arguments.publish),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
