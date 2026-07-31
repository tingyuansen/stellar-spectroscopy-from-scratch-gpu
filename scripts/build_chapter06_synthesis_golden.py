#!/usr/bin/env python3
"""Fixed-path publisher for the Chapter 6 synthesis comparison golden.

This module is deliberately fail-closed.  Candidate-byte acceptance is not
publication authority.  The canonical publisher may construct candidate bytes
only after a detached authorization and its independent JSON review exist at
their one fixed location and bind the exact accepted publisher, contract,
candidate, atmosphere-first manifest state, and manifest-entry template.

There is no caller-selected repository, destination, record, repair, force,
overwrite, or partial-lane interface.  Verification-only execution never
calls a publication primitive.  The mutation boundary is reachable only from
``--publish`` after the same complete authorization gate has passed.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import subprocess
import sys
from typing import Any
import unicodedata
import zipfile

import numpy as np


MODULE_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_REPOSITORY_ROOT))


CANONICAL_REPOSITORY_ROOT = Path("/Users/ysting/stellar-spectroscopy-from-scratch-gpu")
PUBLISHER_RELATIVE_PATH = "scripts/build_chapter06_synthesis_golden.py"
PUBLISHER_TESTS_RELATIVE_PATH = "tests/test_chapter06_synthesis_golden_publisher.py"
PUBLISHER_ACCEPTANCE_RELATIVE_PATH = (
    "design/chapter06_synthesis_golden_publisher_independent_audit.md"
)
AUTHORIZATION_RELATIVE_PATH = "design/chapter06_synthesis_publication_acceptance.json"
RECORD_REVIEW_RELATIVE_PATH = (
    "design/chapter06_synthesis_publication_record_review.json"
)
POSTPUBLICATION_AUDIT_RELATIVE_PATH = (
    "design/chapter06_synthesis_postpublication_audit.md"
)
CONTRACT_RELATIVE_PATH = "design/chapter06_lane_artifact_publisher_contract.md"
CONTRACT_AUDIT_RELATIVE_PATH = (
    "design/chapter06_lane_artifact_publisher_contract_rebind_independent_audit.md"
)
CANDIDATE_ACCEPTANCE_RELATIVE_PATH = (
    "design/chapter06_synthesis_candidate_byte_acceptance.md"
)
ATMOSPHERE_CANDIDATE_ACCEPTANCE_RELATIVE_PATH = (
    "design/chapter06_atmosphere_fixture_byte_acceptance.md"
)
WRITER_RELATIVE_PATH = "scripts/chapter06_synthesis_compact_writer.py"
WRITER_TESTS_RELATIVE_PATH = "tests/test_chapter06_synthesis_compact_writer.py"
WRITER_CANDIDATE_RELATIVE_PATH = "design/chapter06_synthesis_writer_rebind_candidate.md"
WRITER_ACCEPTANCE_RELATIVE_PATH = (
    "design/chapter06_synthesis_writer_rebind_independent_audit.md"
)
MANIFEST_RELATIVE_PATH = "data/MANIFEST.json"
DATA_README_RELATIVE_PATH = "data/README.md"
ATMOSPHERE_DESTINATION_RELATIVE_PATH = (
    "data/fixtures/chapter06_atmosphere_one_line_inputs.npz"
)
DESTINATION_RELATIVE_PATH = (
    "data/golden/payne_zero/chapter06/synthesis/"
    "chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz"
)
DESTINATION_PARENT_RELATIVE_PATH = "data/golden/payne_zero/chapter06/synthesis"
DESTINATION_FILENAME = (
    "chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz"
)

PUBLISHER_PATH = CANONICAL_REPOSITORY_ROOT / PUBLISHER_RELATIVE_PATH
PUBLISHER_TESTS_PATH = CANONICAL_REPOSITORY_ROOT / PUBLISHER_TESTS_RELATIVE_PATH
PUBLISHER_ACCEPTANCE_PATH = (
    CANONICAL_REPOSITORY_ROOT / PUBLISHER_ACCEPTANCE_RELATIVE_PATH
)
AUTHORIZATION_PATH = CANONICAL_REPOSITORY_ROOT / AUTHORIZATION_RELATIVE_PATH
RECORD_REVIEW_PATH = CANONICAL_REPOSITORY_ROOT / RECORD_REVIEW_RELATIVE_PATH
CONTRACT_PATH = CANONICAL_REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH
CONTRACT_AUDIT_PATH = CANONICAL_REPOSITORY_ROOT / CONTRACT_AUDIT_RELATIVE_PATH
CANDIDATE_ACCEPTANCE_PATH = (
    CANONICAL_REPOSITORY_ROOT / CANDIDATE_ACCEPTANCE_RELATIVE_PATH
)
ATMOSPHERE_CANDIDATE_ACCEPTANCE_PATH = (
    CANONICAL_REPOSITORY_ROOT / ATMOSPHERE_CANDIDATE_ACCEPTANCE_RELATIVE_PATH
)
MANIFEST_PATH = CANONICAL_REPOSITORY_ROOT / MANIFEST_RELATIVE_PATH
DESTINATION_PATH = CANONICAL_REPOSITORY_ROOT / DESTINATION_RELATIVE_PATH
DATA_PATH = CANONICAL_REPOSITORY_ROOT / "data"

PINNED_PAYNE_ZERO_ROOT = Path("/Users/ysting/payne-zero")
PAPER_ROOT = Path("/Users/ysting/Source_Files_Not_For_Review")
PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"

CONTRACT_SHA256 = "3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b"
CONTRACT_AUDIT_SHA256 = (
    "fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc"
)
CANDIDATE_ACCEPTANCE_SHA256 = (
    "434088cff95ed60d65dc6c9749d18c2e74e45d114787c03728ed7ae9cf0bd9c9"
)
ATMOSPHERE_CANDIDATE_ACCEPTANCE_SHA256 = (
    "8298b9473cf89161441bbd72a881c744e38fba699aa088eb876014642c91ed71"
)
DATA_README_SHA256 = "1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b"

EXPECTED_ARCHIVE_BYTES = 1_294_865
EXPECTED_ARCHIVE_SHA256 = (
    "a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955"
)
EXPECTED_MEMBER_COUNT = 213
EXPECTED_COMPACT_SCHEMA_VERSION = 1
EXPECTED_COMPACT_SCHEMA_DIGEST = (
    "911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde"
)
EXPECTED_COMPACT_PAYLOAD_FINGERPRINT = (
    "e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b"
)
EXPECTED_RAW_OWNERSHIP_DIGEST = (
    "5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675"
)
EXPECTED_RAW_MEMBER_COUNT = 754
EXPECTED_RAW_SCHEMA_DIGEST = (
    "d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178"
)
EXPECTED_RAW_PHYSICAL_FINGERPRINT = (
    "51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc"
)
EXPECTED_RAW_FULL_FINGERPRINT = (
    "8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893"
)
EXPECTED_RAW_MAPPING_DIGEST = (
    "09072fb51bd3425f6e635275db4f08c6a4fb33c367c9be1a85cdb6c62bc7b06c"
)
EXPECTED_RAW_ARCHIVE_SHA256 = (
    "e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e"
)
EXPECTED_RAW_ARCHIVE_BYTES = 8_689_108
EXPECTED_COMPACT_ARRAY_BYTES = 1_235_275
EXPECTED_RAW_OWNERSHIP_COUNTS = {
    "derived_digest_only": 123,
    "final": 250,
    "intentionally_ephemeral": 381,
}
EXPECTED_ARCHIVE_KIND = "synthesis_one_line_comparison_candidate"
EXPECTED_NPY_VERSION = (2, 0)
EXPECTED_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
EXPECTED_ZIP_EXTERNAL_ATTR = 0o100600 << 16
EXPECTED_MEMBER_METADATA_DIGEST = (
    "39d319b577a59ea06de475a14da9fe68d5fcced24119964a4e3e322226b2b78a"
)

AUTHORIZATION_SCHEMA_VERSION = 1
AUTHORIZATION_RECORD_KIND = "chapter06_synthesis_publication_acceptance"
RECORD_REVIEW_SCHEMA_VERSION = 1
RECORD_REVIEW_KIND = "chapter06_synthesis_publication_record_review"
LATE_AUTHORIZATION_SHA256 = "__LATE_BOUND_AUTHORIZATION_SHA256__"
LATE_RECORD_REVIEW_SHA256 = "__LATE_BOUND_RECORD_REVIEW_SHA256__"
NO_REPLACE_PRIMITIVE = "hard_link_from_same_filesystem_stage"
DESTINATION_PARENT_POLICY = "create-only-missing-chapter06-then-synthesis-with-fsync"

STAGE_PREFIX = ".chapter06-synthesis-artifact-stage-"
MANIFEST_TEMP_PREFIX = ".MANIFEST.json.chapter06-synthesis-temporary-"
QUARANTINE_CLEANUP_RELATIVE_PATH = (
    "design/chapter06_publication_quarantine_cleanup_acceptance.json"
)
PREPUBLICATION_DATA_DIRECTORIES = (
    "data",
    "data/fixtures",
    "data/golden",
    "data/golden/payne_zero",
    "data/golden/payne_zero/chapter04",
    "data/golden/payne_zero/chapter05",
    "data/static",
    "data/static/atmosphere_tables",
    "data/static/schemas",
    "data/static/source_catalogs",
    "data/static/source_catalogs/lines",
    "data/static/synthesis_tables",
    "data/subsets",
)
SYNTHESIS_NEW_DATA_DIRECTORIES = (
    "data/golden/payne_zero/chapter06",
    DESTINATION_PARENT_RELATIVE_PATH,
)

FIXED_UPSTREAM_IDENTITIES: dict[str, tuple[str, str]] = {
    "fixture_oracle_plan": (
        "design/chapter06_synthesis_fixture_oracle_plan.md",
        "413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856",
    ),
    "phase1_candidate": (
        "design/chapter06_synthesis_plan_rebind_candidate.md",
        "dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93",
    ),
    "phase1_audit": (
        "design/chapter06_synthesis_plan_rebind_independent_audit.md",
        "9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7",
    ),
    "worker": (
        "scripts/chapter06_synthesis_oracle_worker.py",
        "36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68",
    ),
    "worker_tests": (
        "tests/test_chapter06_synthesis_oracle_worker.py",
        "1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189",
    ),
    "worker_audit": (
        "design/chapter06_synthesis_worker_independent_audit.md",
        "a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334",
    ),
    "assembler": (
        "scripts/chapter06_synthesis_compact_assembler.py",
        "583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8",
    ),
    "assembler_tests": (
        "tests/test_chapter06_synthesis_compact_assembler.py",
        "25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153",
    ),
    "phase2_candidate": (
        "design/chapter06_synthesis_compact_rebind_candidate.md",
        "54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c",
    ),
    "phase2_audit": (
        "design/chapter06_synthesis_compact_rebind_independent_audit.md",
        "739854db2b5c4c0c0fe5e9db71d8a52958ce401ded7e7a80a8ab90e15172ddcb",
    ),
    "writer": (
        WRITER_RELATIVE_PATH,
        "57aa7147afee4a7366cb2a075715d3607fa20507c23c07ec978b0698368ae47b",
    ),
    "writer_tests": (
        WRITER_TESTS_RELATIVE_PATH,
        "7c41a74f9d2e38a23d988c990af4040ac262a8066cb3cd9feae4e29f0bdc0a4e",
    ),
    "writer_candidate": (
        WRITER_CANDIDATE_RELATIVE_PATH,
        "6ab1f346a409b0302550a0923c35b71a84d6b2899f2c356070c8d76aa8145e5a",
    ),
    "writer_acceptance": (
        WRITER_ACCEPTANCE_RELATIVE_PATH,
        "467fdc810f14302dba80f0dd18ba34239dfedb7579b48280899f6f9b6e3b3653",
    ),
    "publisher_contract": (CONTRACT_RELATIVE_PATH, CONTRACT_SHA256),
    "publisher_contract_acceptance": (
        CONTRACT_AUDIT_RELATIVE_PATH,
        CONTRACT_AUDIT_SHA256,
    ),
    "candidate_byte_acceptance": (
        CANDIDATE_ACCEPTANCE_RELATIVE_PATH,
        CANDIDATE_ACCEPTANCE_SHA256,
    ),
    "atmosphere_candidate_byte_acceptance": (
        ATMOSPHERE_CANDIDATE_ACCEPTANCE_RELATIVE_PATH,
        ATMOSPHERE_CANDIDATE_ACCEPTANCE_SHA256,
    ),
}

AUTHORIZATION_KEYS = (
    "schema_version",
    "record_kind",
    "lane",
    "role",
    "payne_zero_commit",
    "identities",
    "artifact",
    "prepublication_manifest",
    "manifest_entry_template",
    "manifest_entry_template_sha256",
    "source_data_snapshot_sha256",
    "destination_parent_policy",
    "no_replace_primitive",
)
IDENTITY_KEYS = ("path", "sha256")
ARTIFACT_KEYS = (
    "path",
    "filename",
    "role",
    "purpose",
    "bytes",
    "sha256",
    "member_count",
    "archive_kind",
    "comparison_only",
    "compact_schema_version",
    "npy_member_format_version",
    "schema_digest",
    "payload_fingerprint",
    "raw_ownership_digest",
    "raw_member_count",
    "raw_schema_digest",
    "raw_physical_fingerprint",
    "raw_full_fingerprint",
    "contains_copied_input_state",
    "contains_atmosphere_lane_members",
)
PREPUBLICATION_MANIFEST_KEYS = (
    "path",
    "sha256",
    "schema_version",
    "payne_zero_commit",
    "ordered_entry_path_digest",
    "destination_entry_absent",
    "atmosphere_phase",
    "atmosphere_entry_path",
    "atmosphere_artifact_sha256",
    "atmosphere_entry_digest",
)
RECORD_REVIEW_KEYS = (
    "schema_version",
    "record_kind",
    "authorization_path",
    "authorization_sha256",
    "candidate_byte_acceptance_path",
    "candidate_byte_acceptance_sha256",
    "publisher_acceptance_path",
    "publisher_acceptance_sha256",
    "manifest_entry_template_sha256",
    "disposition",
)
MANIFEST_TOP_LEVEL_KEYS = ("schema_version", "payne_zero_commit", "entries")
ARRAY_RECORD_KEYS = ("shape", "dtype", "unit", "sha256", "axes", "ownership")
TEMPLATE_KEYS = (
    "path",
    "role",
    "format",
    "scope",
    "builder",
    "builder_sha256",
    "publisher_contract_sha256",
    "publisher_contract_acceptance_sha256",
    "candidate_byte_acceptance_sha256",
    "atmosphere_candidate_byte_acceptance_sha256",
    "publisher_acceptance_sha256",
    "writer",
    "writer_sha256",
    "writer_tests_sha256",
    "writer_candidate_sha256",
    "writer_acceptance_sha256",
    "publication_acceptance_sha256",
    "publication_record_review_sha256",
    "payne_zero_commit",
    "source_data_snapshot_sha256",
    "archive_kind",
    "comparison_only",
    "compact_schema_version",
    "npy_member_format_version",
    "compact_schema_digest",
    "compact_payload_fingerprint",
    "raw_ownership_digest",
    "raw_member_count",
    "raw_schema_digest",
    "raw_physical_fingerprint",
    "raw_full_fingerprint",
    "upstream_identities",
    "reproducibility_environment",
    "arrays",
    "sha256",
    "bytes",
)
REPRODUCIBILITY_ENVIRONMENT = {
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
    "cpu_only": True,
    "work_dtype": "torch.float64",
    "accumulation_dtype": "torch.float32",
    "archive_compression": "ZIP_STORED",
    "npy_member_format_version": "2.0",
}
REQUIRED_IDENTITY_LABELS = (
    *FIXED_UPSTREAM_IDENTITIES,
    "publisher",
    "publisher_tests",
    "publisher_acceptance",
)
WRITER_SUMMARY_KEYS = (
    "accepted_identity_count",
    "writer_sha256",
    "raw_a_key_count",
    "raw_b_key_count",
    "raw_a_b_bitwise_equal",
    "raw_a_b_mapping_digest",
    "raw_a_archive_sha256",
    "raw_b_archive_sha256",
    "raw_a_b_archive_byte_equal",
    "raw_archive_bytes",
    "raw_schema_digest",
    "raw_physical_fingerprint",
    "raw_full_fingerprint",
    "compact_a_b_schema_equal",
    "compact_a_b_payload_equal",
    "compact_a_b_ownership_equal",
    "compact_key_count",
    "compact_array_bytes",
    "compact_schema_digest",
    "compact_payload_fingerprint",
    "raw_ownership_digest",
    "raw_ownership_counts",
    "archive_a_sha256",
    "archive_b_sha256",
    "archive_a_b_byte_equal",
    "archive_bytes",
    "archive_member_count",
    "archive_member_order",
    "archive_compression",
    "npy_version",
    "fixed_zip_date_time",
    "capture_process_evidence",
    "capture_process_evidence_sha256",
    "disposable_cache_directories_created",
    "publication_authorized",
    "golden_publication_performed",
    "manifest_mutation_performed",
    "artifact_file_write_performed",
)
WRITER_PROCESS_EVIDENCE_KEYS = (
    "capture_a_origin_token_sha256",
    "capture_b_origin_token_sha256",
    "capture_a_child_pid",
    "capture_b_child_pid",
    "capture_a_cache_root_sha256",
    "capture_b_cache_root_sha256",
    "capture_origins_distinct",
    "capture_child_processes_distinct",
    "capture_cache_roots_distinct",
    "capture_cache_roots_external",
    "capture_cache_roots_nonsymlink",
    "capture_cache_roots_empty_before",
    "capture_cache_roots_empty_after",
    "capture_cache_roots_disposed",
)


class PublisherError(RuntimeError):
    """Base class for a fail-closed publisher rejection."""


class PublisherIdentityError(PublisherError):
    """Raised when a fixed path or accepted byte identity changes."""


class PublisherSchemaError(PublisherError):
    """Raised for hostile, duplicate, reordered, or semantically wrong data."""


class PublicationGateError(PublisherError):
    """Raised before candidate construction when authority is incomplete."""


class PublicationStateError(PublisherError):
    """Raised when filesystem state is neither complete nor recoverable."""


class PublicationIOError(PublisherError):
    """Raised when a durability or atomicity operation is incomplete."""


@dataclass(frozen=True)
class FileFact:
    """No-follow identity for one regular file."""

    relative_path: str
    device: int
    inode: int
    owner: int
    group: int
    mode: int
    link_count: int
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class DirectoryFact:
    """No-follow identity for one directory."""

    relative_path: str
    device: int
    inode: int
    owner: int
    group: int
    mode: int


@dataclass(frozen=True)
class ManifestFileFact:
    """Exact manifest-to-file join for one registered artifact."""

    relative_path: str
    role: str
    entry_position: int
    entry_sha256: str
    byte_count: int
    file_sha256: str


@dataclass(frozen=True)
class NonmanifestSupportFact:
    """One regular data file governed outside the legacy Chapter 6 manifest."""

    relative_path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class DataSnapshot:
    """Complete data-tree inventory plus the exact legacy-manifest join."""

    directories: tuple[DirectoryFact, ...]
    files: tuple[FileFact, ...]
    manifest_files: tuple[ManifestFileFact, ...]
    nonmanifest_support: tuple[NonmanifestSupportFact, ...]
    manifest_sha256: str
    manifest_device: int
    manifest_inode: int
    ordered_entry_path_digest: str
    aggregate_sha256: str


@dataclass(frozen=True)
class OwnedManifestTemporary:
    """Exact retained identity of this invocation's one manifest temporary."""

    path: Path
    descriptor: int
    parent_descriptor: int
    details: os.stat_result
    intended_bytes: bytes
    intended_sha256: str
    mode: int


@dataclass(frozen=True)
class Candidate:
    """Accepted in-memory bytes and deterministic publisher evidence."""

    archive_bytes: bytes
    archive_sha256: str
    summary: dict[str, Any]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class Authority:
    """Validated forward-only authorization and review bytes."""

    authorization: dict[str, Any]
    authorization_bytes: bytes
    authorization_sha256: str
    review: dict[str, Any]
    review_bytes: bytes
    review_sha256: str


@dataclass(frozen=True)
class InvocationDirectory:
    """One directory created and durably identified by this invocation."""

    parent: Path
    path: Path
    device: int
    inode: int


def sha256_bytes(payload: bytes) -> str:
    """Return lowercase SHA-256 for immutable bytes."""

    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Any, expected: tuple[str, ...], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublisherSchemaError(f"{label} must be a JSON object")
    if tuple(value) != expected:
        raise PublisherSchemaError(
            f"{label} keys/order changed: {tuple(value)!r}; expected {expected!r}"
        )
    return value


def _plain_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PublisherSchemaError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _plain_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise PublisherSchemaError(f"{label} must be boolean")
    return value


def _sha256_hex(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublisherSchemaError(f"{label} must be lowercase 64-hex SHA-256")
    return value


def _identity_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise PublisherSchemaError(f"{label} must be nonempty ASCII")
    if unicodedata.normalize("NFC", value) != value:
        raise PublisherSchemaError(f"{label} must be NFC")
    if (
        value.startswith("/")
        or "\\" in value
        or "%" in value
        or "//" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise PublisherSchemaError(f"{label} is not a safe POSIX identity")
    if PurePosixPath(value).as_posix() != value:
        raise PublisherSchemaError(f"{label} is not a literal POSIX identity")
    return value


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    """Decode finite UTF-8 JSON while rejecting duplicate keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PublisherSchemaError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise PublisherSchemaError(f"{label} contains nonfinite value {value}")

    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PublisherSchemaError(f"{label} is not UTF-8") from error
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise PublisherSchemaError(f"{label} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise PublisherSchemaError(f"{label} top level must be an object")
    return value


def _manifest_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            separators=(",", ": "),
        )
    except (TypeError, ValueError) as error:
        raise PublisherSchemaError(f"manifest is not encodable: {error}") from error
    return (text + "\n").encode("utf-8")


def _template_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise PublisherSchemaError(f"template is not encodable: {error}") from error
    return text.encode("utf-8")


def _entry_digest(entry: Mapping[str, Any]) -> str:
    return sha256_bytes(_template_bytes(entry))


def _ordered_path_digest(paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        encoded = path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _require_canonical_execution() -> None:
    try:
        executing = Path(__file__).resolve(strict=True)
        repository = CANONICAL_REPOSITORY_ROOT.resolve(strict=True)
    except OSError as error:
        raise PublisherIdentityError(
            "canonical publisher or repository is unavailable"
        ) from error
    if repository != CANONICAL_REPOSITORY_ROOT:
        raise PublisherIdentityError("canonical repository lexical identity changed")
    if executing != PUBLISHER_PATH:
        raise PublisherIdentityError(
            "publisher is not executing from its one canonical path"
        )
    if PINNED_PAYNE_ZERO_ROOT == repository or PAPER_ROOT == repository:
        raise PublisherIdentityError("forbidden external roots alias repository")


def _open_flags(*, directory: bool = False) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    return flags


def _read_all_fd(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_regular_nofollow(path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    try:
        fd = os.open(path, _open_flags())
    except OSError as error:
        raise PublisherIdentityError(
            f"{label} is missing, symlinked, or unreadable: {path}"
        ) from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PublisherIdentityError(f"{label} is not a regular file")
        payload = _read_all_fd(fd)
        after = os.fstat(fd)
        named = os.lstat(path)
        if (
            (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
            or not stat.S_ISREG(named.st_mode)
            or named.st_size != len(payload)
        ):
            raise PublisherIdentityError(f"{label} changed while being read")
        return payload, after
    finally:
        os.close(fd)


def _read_repository_file(
    relative_path: str,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    """Walk one fixed repository identity with directory-relative no-follow opens."""

    identity = _identity_path(relative_path, label=f"{label} path")
    components = identity.split("/")
    root_fd = os.open(
        CANONICAL_REPOSITORY_ROOT,
        _open_flags(directory=True),
    )
    current_fd = root_fd
    try:
        root_details = os.fstat(root_fd)
        named_root = os.lstat(CANONICAL_REPOSITORY_ROOT)
        if (root_details.st_dev, root_details.st_ino) != (
            named_root.st_dev,
            named_root.st_ino,
        ):
            raise PublisherIdentityError("canonical repository root changed")
        for component in components[:-1]:
            try:
                child_fd = os.open(
                    component,
                    _open_flags(directory=True),
                    dir_fd=current_fd,
                )
            except OSError as error:
                raise PublisherIdentityError(
                    f"{label} parent is missing or symlinked: {component}"
                ) from error
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd
        try:
            file_fd = os.open(
                components[-1],
                _open_flags(),
                dir_fd=current_fd,
            )
        except OSError as error:
            raise PublisherIdentityError(
                f"{label} is missing, symlinked, or unreadable"
            ) from error
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode):
                raise PublisherIdentityError(f"{label} is not a regular file")
            payload = _read_all_fd(file_fd)
            after = os.fstat(file_fd)
            named = os.stat(
                components[-1],
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if (
                (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
                or not stat.S_ISREG(named.st_mode)
                or named.st_size != len(payload)
            ):
                raise PublisherIdentityError(f"{label} changed while being read")
            return payload, after
        finally:
            os.close(file_fd)
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def _open_repository_directory(relative_path: str, *, label: str) -> int:
    """Open one repository directory through an all-component no-follow walk."""

    identity = _identity_path(relative_path, label=f"{label} path")
    root_fd = os.open(
        CANONICAL_REPOSITORY_ROOT,
        _open_flags(directory=True),
    )
    current_fd = root_fd
    try:
        for component in identity.split("/"):
            child_fd = os.open(
                component,
                _open_flags(directory=True),
                dir_fd=current_fd,
            )
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd
        result = os.dup(current_fd)
    except OSError as error:
        raise PublisherIdentityError(
            f"{label} is missing, symlinked, or not a directory"
        ) from error
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    return result


def _revalidate_repository_directory(
    relative_path: str,
    retained_fd: int,
    *,
    label: str,
) -> None:
    reopened_fd = _open_repository_directory(relative_path, label=label)
    try:
        retained = os.fstat(retained_fd)
        reopened = os.fstat(reopened_fd)
        if (retained.st_dev, retained.st_ino) != (
            reopened.st_dev,
            reopened.st_ino,
        ):
            raise PublicationStateError(f"{label} directory identity changed")
    finally:
        os.close(reopened_fd)


def _rebind_retained_directory_path(
    path: Path,
    retained_fd: int,
    *,
    label: str,
) -> None:
    """Require a retained mutated directory to remain at its canonical name."""

    try:
        reopened_fd = os.open(path, _open_flags(directory=True))
    except OSError as error:
        raise PublicationStateError(
            f"{label} canonical directory identity is unavailable"
        ) from error
    try:
        retained = os.fstat(retained_fd)
        reopened = os.fstat(reopened_fd)
        if (
            not stat.S_ISDIR(retained.st_mode)
            or not stat.S_ISDIR(reopened.st_mode)
            or (retained.st_dev, retained.st_ino) != (reopened.st_dev, reopened.st_ino)
        ):
            raise PublicationStateError(f"{label} canonical directory identity changed")
    finally:
        os.close(reopened_fd)


def _read_named_regular_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    """Read a regular child through one retained, already validated parent."""

    if "/" in name or name in {"", ".", ".."}:
        raise PublisherIdentityError(f"{label} child name is unsafe")
    try:
        fd = os.open(name, _open_flags(), dir_fd=directory_fd)
    except OSError as error:
        raise PublisherIdentityError(
            f"{label} is missing, symlinked, or unreadable"
        ) from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PublisherIdentityError(f"{label} is not a regular file")
        payload = _read_all_fd(fd)
        after = os.fstat(fd)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or (after.st_dev, after.st_ino) != (named.st_dev, named.st_ino)
            or not stat.S_ISREG(named.st_mode)
            or named.st_size != len(payload)
        ):
            raise PublisherIdentityError(f"{label} changed while being read")
        return payload, after
    finally:
        os.close(fd)


def _verify_fixed_upstream_identities() -> dict[str, str]:
    verified: dict[str, str] = {}
    for label, (relative_path, expected_hash) in FIXED_UPSTREAM_IDENTITIES.items():
        _identity_path(relative_path, label=f"{label} path")
        payload, _ = _read_repository_file(
            relative_path,
            label=label,
        )
        actual_hash = sha256_bytes(payload)
        if actual_hash != expected_hash:
            raise PublisherIdentityError(
                f"{label} identity changed: {actual_hash}; expected {expected_hash}"
            )
        verified[label] = actual_hash
    return verified


def _file_fact(path: Path, *, relative_path: str) -> FileFact:
    payload, details = _read_regular_nofollow(path, label=relative_path)
    return FileFact(
        relative_path=relative_path,
        device=details.st_dev,
        inode=details.st_ino,
        owner=details.st_uid,
        group=details.st_gid,
        mode=stat.S_IMODE(details.st_mode),
        link_count=details.st_nlink,
        byte_count=len(payload),
        sha256=sha256_bytes(payload),
    )


def _directory_fact(path: Path, *, relative_path: str) -> DirectoryFact:
    try:
        details = os.lstat(path)
    except OSError as error:
        raise PublisherIdentityError(
            f"data directory is unavailable: {relative_path}"
        ) from error
    if not stat.S_ISDIR(details.st_mode) or path.is_symlink():
        raise PublisherIdentityError(
            f"data component is not a nonsymlink directory: {relative_path}"
        )
    return DirectoryFact(
        relative_path=relative_path,
        device=details.st_dev,
        inode=details.st_ino,
        owner=details.st_uid,
        group=details.st_gid,
        mode=stat.S_IMODE(details.st_mode),
    )


def _parse_manifest(payload: bytes) -> dict[str, Any]:
    manifest = _strict_json(payload, label=MANIFEST_RELATIVE_PATH)
    _exact_keys(manifest, MANIFEST_TOP_LEVEL_KEYS, label="manifest")
    if _plain_int(manifest["schema_version"], label="manifest schema") != 1:
        raise PublisherSchemaError("manifest schema version changed")
    if manifest["payne_zero_commit"] != PINNED_PAYNE_ZERO_COMMIT:
        raise PublisherSchemaError("manifest Payne Zero commit changed")
    entries = manifest["entries"]
    if not isinstance(entries, list):
        raise PublisherSchemaError("manifest entries must be a list")
    paths: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PublisherSchemaError(f"manifest entry {index} is not an object")
        if "path" not in entry or "role" not in entry:
            raise PublisherSchemaError(f"manifest entry {index} lacks path/role")
        path = _identity_path(entry["path"], label=f"manifest entry {index} path")
        if entry["role"] not in {"fixture", "golden", "static", "subset"}:
            raise PublisherSchemaError(
                f"manifest entry {index} has an invalid data role"
            )
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise PublisherSchemaError("manifest contains duplicate entry paths")
    if _manifest_bytes(manifest) != payload:
        raise PublisherSchemaError("manifest violates exact ordered byte encoding")
    return manifest


def _inventory_quarantine(root: Path) -> tuple[FileFact, ...]:
    found: list[FileFact] = []
    if not root.exists():
        return ()
    for path in root.rglob("*"):
        if not path.name.startswith((STAGE_PREFIX, MANIFEST_TEMP_PREFIX)):
            continue
        relative = path.relative_to(CANONICAL_REPOSITORY_ROOT).as_posix()
        try:
            details = os.lstat(path)
        except OSError as error:
            raise PublicationStateError(
                f"quarantine object changed during inventory: {relative}"
            ) from error
        if not stat.S_ISREG(details.st_mode):
            raise PublicationStateError(
                f"quarantine object is not a regular file: {relative}"
            )
        found.append(_file_fact(path, relative_path=relative))
    return tuple(sorted(found, key=lambda item: item.relative_path))


def _validate_owned_manifest_temporary(
    owned: OwnedManifestTemporary,
) -> FileFact:
    """Prove one temporary is the exact live inode retained by this invocation."""

    if (
        owned.path.parent != DATA_PATH
        or not owned.path.name.startswith(MANIFEST_TEMP_PREFIX)
        or "/" in owned.path.name
        or owned.path.name in {"", ".", ".."}
    ):
        raise PublicationStateError(
            "owned manifest temporary is outside its exact canonical namespace"
        )
    _revalidate_repository_directory(
        "data",
        owned.parent_descriptor,
        label="owned manifest temporary parent",
    )
    try:
        retained = os.fstat(owned.descriptor)
        named = os.stat(
            owned.path.name,
            dir_fd=owned.parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise PublicationStateError(
            "owned manifest temporary identity is unavailable"
        ) from error
    expected_identity = (owned.details.st_dev, owned.details.st_ino)
    if (
        not stat.S_ISREG(retained.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or (retained.st_dev, retained.st_ino) != expected_identity
        or (named.st_dev, named.st_ino) != expected_identity
        or retained.st_nlink != 1
        or named.st_nlink != 1
        or retained.st_nlink != owned.details.st_nlink
        or retained.st_size != len(owned.intended_bytes)
        or named.st_size != len(owned.intended_bytes)
        or retained.st_size != owned.details.st_size
        or stat.S_IMODE(retained.st_mode) != owned.mode
        or stat.S_IMODE(named.st_mode) != owned.mode
        or stat.S_IMODE(owned.details.st_mode) != owned.mode
        or retained.st_uid != owned.details.st_uid
        or named.st_uid != owned.details.st_uid
        or retained.st_gid != owned.details.st_gid
        or named.st_gid != owned.details.st_gid
    ):
        raise PublicationStateError(
            "owned manifest temporary descriptor/name identity changed"
        )
    if len(owned.intended_sha256) != 64 or owned.intended_sha256 != sha256_bytes(
        owned.intended_bytes
    ):
        raise PublicationStateError("owned manifest temporary intended hash changed")
    os.lseek(owned.descriptor, 0, os.SEEK_SET)
    retained_payload = _read_all_fd(owned.descriptor)
    named_payload, named_details = _read_named_regular_at(
        owned.parent_descriptor,
        owned.path.name,
        label="owned manifest temporary",
    )
    if (
        retained_payload != owned.intended_bytes
        or named_payload != owned.intended_bytes
        or sha256_bytes(retained_payload) != owned.intended_sha256
        or sha256_bytes(named_payload) != owned.intended_sha256
        or (named_details.st_dev, named_details.st_ino) != expected_identity
    ):
        raise PublicationStateError("owned manifest temporary bytes or hash changed")
    relative_path = owned.path.relative_to(CANONICAL_REPOSITORY_ROOT).as_posix()
    return FileFact(
        relative_path=relative_path,
        device=retained.st_dev,
        inode=retained.st_ino,
        owner=retained.st_uid,
        group=retained.st_gid,
        mode=stat.S_IMODE(retained.st_mode),
        link_count=retained.st_nlink,
        byte_count=len(retained_payload),
        sha256=sha256_bytes(retained_payload),
    )


def _snapshot_data(
    *,
    allow_exact_unregistered_target: str | None = None,
    owned_manifest_temporary: OwnedManifestTemporary | None = None,
) -> DataSnapshot:
    """Return and validate the complete closed canonical data inventory."""

    owned_manifest_fact = (
        _validate_owned_manifest_temporary(owned_manifest_temporary)
        if owned_manifest_temporary is not None
        else None
    )
    quarantine = _inventory_quarantine(DATA_PATH)
    if owned_manifest_fact is None:
        unexpected_quarantine = quarantine
    elif quarantine == (owned_manifest_fact,):
        unexpected_quarantine = ()
    else:
        unexpected_quarantine = tuple(
            item for item in quarantine if item != owned_manifest_fact
        )
        if not unexpected_quarantine:
            raise PublicationStateError(
                "owned manifest temporary is not the unique exact quarantine object"
            )
    if unexpected_quarantine:
        raise PublicationStateError(
            "crash-left publication temporary is quarantined; separately "
            f"review {QUARANTINE_CLEANUP_RELATIVE_PATH}"
        )
    manifest_payload, manifest_details = _read_repository_file(
        MANIFEST_RELATIVE_PATH,
        label=MANIFEST_RELATIVE_PATH,
    )
    manifest = _parse_manifest(manifest_payload)
    directories: list[DirectoryFact] = []
    files: list[FileFact] = []
    for current_root, directory_names, file_names in os.walk(
        DATA_PATH,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        directory_names.sort()
        file_names.sort()
        relative_directory = current.relative_to(CANONICAL_REPOSITORY_ROOT).as_posix()
        directories.append(_directory_fact(current, relative_path=relative_directory))
        for name in directory_names:
            child = current / name
            details = os.lstat(child)
            if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
                raise PublicationStateError(
                    f"data tree contains a non-directory component: {child}"
                )
        for name in file_names:
            child = current / name
            relative_file = child.relative_to(CANONICAL_REPOSITORY_ROOT).as_posix()
            details = os.lstat(child)
            if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
                raise PublicationStateError(
                    f"data tree contains a symlink or special node: {relative_file}"
                )
            files.append(_file_fact(child, relative_path=relative_file))

    by_path = {item.relative_path: item for item in files}
    if len(by_path) != len(files):
        raise PublicationStateError("data inventory contains duplicate path identity")
    directory_by_path = {item.relative_path: item for item in directories}
    actual_directories = set(directory_by_path)
    # Historical empty base directories are an explicit part of the original
    # publisher state.  Every later directory must be justified by an actual
    # regular file, so arbitrary empty-directory drift remains rejected.
    expected_directories = set(PREPUBLICATION_DATA_DIRECTORIES)
    for item in files:
        parent = PurePosixPath(item.relative_path).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_directories != expected_directories:
        raise PublicationStateError(
            "data directory inventory changed: "
            f"missing={sorted(expected_directories - actual_directories)!r}, "
            f"extra={sorted(actual_directories - expected_directories)!r}"
        )
    for relative_path in sorted(actual_directories - {"data"}):
        item = directory_by_path[relative_path]
        parent = directory_by_path[str(PurePosixPath(relative_path).parent)]
        if (
            item.mode != 0o755
            or item.owner != parent.owner
            or item.group != parent.group
        ):
            raise PublicationStateError(
                f"manifest-derived data directory metadata changed: {relative_path}"
            )
    readme = by_path.get(DATA_README_RELATIVE_PATH)
    if readme is None or readme.sha256 != DATA_README_SHA256:
        raise PublicationStateError("closed nonmanifest data/README.md changed")

    entry_paths: list[str] = []
    manifest_files: list[ManifestFileFact] = []
    for index, entry in enumerate(manifest["entries"]):
        path = entry["path"]
        entry_paths.append(path)
        fact = by_path.get(path)
        if fact is None:
            raise PublicationStateError(
                f"manifest entry {index} has no regular file: {path}"
            )
        if entry.get("sha256") != fact.sha256 or entry.get("bytes") != fact.byte_count:
            raise PublicationStateError(
                f"manifest entry/file identity mismatch: {path}"
            )
        manifest_files.append(
            ManifestFileFact(
                relative_path=path,
                role=entry["role"],
                entry_position=index,
                entry_sha256=_entry_digest(entry),
                byte_count=fact.byte_count,
                file_sha256=fact.sha256,
            )
        )

    normal_nonmanifest = {
        MANIFEST_RELATIVE_PATH,
        DATA_README_RELATIVE_PATH,
    }
    if allow_exact_unregistered_target is not None:
        normal_nonmanifest.add(allow_exact_unregistered_target)
    if owned_manifest_fact is not None:
        observed_owned = by_path.get(owned_manifest_fact.relative_path)
        if observed_owned != owned_manifest_fact:
            raise PublicationStateError(
                "owned manifest temporary changed during closed data inventory"
            )
        normal_nonmanifest.add(owned_manifest_fact.relative_path)
    # Later chapters deliberately use chapter-scoped artifact records instead
    # of extending the legacy append-only MANIFEST.json.  Keep every regular
    # file in the no-follow snapshot and aggregate digest, but reserve the
    # strict manifest/file join for paths that MANIFEST.json actually owns.
    external_support_paths = (
        set(by_path) - set(entry_paths) - normal_nonmanifest
    )
    target = by_path.get(DESTINATION_RELATIVE_PATH)
    if target is not None:
        registered = DESTINATION_RELATIVE_PATH in entry_paths
        if not registered and (
            allow_exact_unregistered_target != DESTINATION_RELATIVE_PATH
        ):
            raise PublicationStateError(
                "synthesis target is unregistered outside recovery state"
            )
        if (
            target.byte_count != EXPECTED_ARCHIVE_BYTES
            or target.sha256 != EXPECTED_ARCHIVE_SHA256
            or target.link_count != 1
            or target.mode != 0o600
        ):
            raise PublicationStateError(
                "synthesis target is not the exact single-link mode-0600 artifact"
            )
        if not set(SYNTHESIS_NEW_DATA_DIRECTORIES).issubset(actual_directories):
            raise PublicationStateError(
                "synthesis target exists without its exact destination directories"
            )

    digest = hashlib.sha256()
    # The authorization binds stable prepublication data.  The manifest has
    # its own exact hash/inode fields, while the only recoverable mutation
    # delta is the exact target plus the two allowlisted destination
    # directories.  Excluding precisely those objects makes M1, an exact
    # unregistered crash state, and reconstructed M1 from registered M2 share
    # one authority digest without hiding any other data delta.
    stable_directories = [
        item
        for item in directories
        if item.relative_path not in SYNTHESIS_NEW_DATA_DIRECTORIES
    ]
    stable_files = [
        item
        for item in files
        if item.relative_path
        not in {
            MANIFEST_RELATIVE_PATH,
            DESTINATION_RELATIVE_PATH,
            (
                owned_manifest_fact.relative_path
                if owned_manifest_fact is not None
                else ""
            ),
        }
    ]
    stable_manifest_files = [
        item
        for item in manifest_files
        if item.relative_path != DESTINATION_RELATIVE_PATH
    ]
    for item in sorted(stable_directories, key=lambda value: value.relative_path):
        encoded = json.dumps(
            asdict(item), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    for item in sorted(stable_files, key=lambda value: value.relative_path):
        encoded = json.dumps(
            asdict(item), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    for item in stable_manifest_files:
        encoded = json.dumps(
            asdict(item), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    support = tuple(
        NonmanifestSupportFact(
            relative_path=by_path[relative_path].relative_path,
            byte_count=by_path[relative_path].byte_count,
            sha256=by_path[relative_path].sha256,
        )
        for relative_path in sorted(
            {DATA_README_RELATIVE_PATH, *external_support_paths}
        )
    )
    for item in support:
        encoded = json.dumps(
            asdict(item), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    if owned_manifest_temporary is not None:
        final_owned_fact = _validate_owned_manifest_temporary(owned_manifest_temporary)
        if final_owned_fact != owned_manifest_fact:
            raise PublicationStateError(
                "owned manifest temporary changed across closed data inventory"
            )
    return DataSnapshot(
        directories=tuple(sorted(directories, key=lambda value: value.relative_path)),
        files=tuple(sorted(files, key=lambda value: value.relative_path)),
        manifest_files=tuple(manifest_files),
        nonmanifest_support=support,
        manifest_sha256=sha256_bytes(manifest_payload),
        manifest_device=manifest_details.st_dev,
        manifest_inode=manifest_details.st_ino,
        ordered_entry_path_digest=_ordered_path_digest(entry_paths),
        aggregate_sha256=digest.hexdigest(),
    )


def _verify_source_identity() -> dict[str, str]:
    """Run the accepted worker's closed pinned/staged/source-data identity gate."""

    from scripts import chapter06_synthesis_oracle_worker as worker

    if (
        Path(worker.__file__).resolve(strict=True)
        != CANONICAL_REPOSITORY_ROOT / FIXED_UPSTREAM_IDENTITIES["worker"][0]
    ):
        raise PublisherIdentityError("synthesis worker imported from wrong path")
    try:
        identities = worker.verify_identity()
    except (
        OSError,
        subprocess.SubprocessError,
        worker.OracleIdentityError,
    ) as error:
        raise PublisherIdentityError(
            f"accepted synthesis source identity failed: {error}"
        ) from error
    if not isinstance(identities, dict) or not identities:
        raise PublisherIdentityError("accepted synthesis source inventory is empty")
    return {name: identities[name] for name in sorted(identities)}


def _source_data_snapshot_digest(
    snapshot: DataSnapshot,
    source_identities: Mapping[str, str],
    *,
    manifest_sha256: str | None = None,
    ordered_entry_path_digest: str | None = None,
) -> str:
    """Bind the complete source inventory to the complete data inventory."""

    payload = {
        "schema_version": 1,
        "source_identities": dict(source_identities),
        "data_aggregate_sha256": snapshot.aggregate_sha256,
        "manifest_sha256": (
            snapshot.manifest_sha256 if manifest_sha256 is None else manifest_sha256
        ),
        "ordered_entry_path_digest": (
            snapshot.ordered_entry_path_digest
            if ordered_entry_path_digest is None
            else ordered_entry_path_digest
        ),
    }
    return sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _validate_identity_record(
    label: str,
    record: Any,
    *,
    exact_path: str,
    exact_sha256: str | None,
) -> str:
    value = _exact_keys(record, IDENTITY_KEYS, label=f"identity {label}")
    path = _identity_path(value["path"], label=f"identity {label} path")
    if path != exact_path:
        raise PublicationGateError(f"identity {label} has the wrong path")
    digest = _sha256_hex(value["sha256"], label=f"identity {label} hash")
    if exact_sha256 is not None and digest != exact_sha256:
        raise PublicationGateError(f"identity {label} has a stale accepted hash")
    payload, _ = _read_repository_file(
        path,
        label=f"identity {label}",
    )
    if sha256_bytes(payload) != digest:
        raise PublicationGateError(f"identity {label} bytes changed")
    return digest


def _validate_template_shape(template: Any) -> dict[str, Any]:
    template = _exact_keys(
        template,
        TEMPLATE_KEYS,
        label="manifest entry template",
    )
    placeholders: list[tuple[str, str]] = []

    def walk(value: Any, trail: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, (*trail, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*trail, str(index)))
        elif value in {LATE_AUTHORIZATION_SHA256, LATE_RECORD_REVIEW_SHA256}:
            placeholders.append((".".join(trail), value))

    walk(template, ())
    if placeholders != [
        ("publication_acceptance_sha256", LATE_AUTHORIZATION_SHA256),
        ("publication_record_review_sha256", LATE_RECORD_REVIEW_SHA256),
    ]:
        raise PublisherSchemaError(
            "template late-bound placeholders changed position, order, or count"
        )
    if template.get("path") != DESTINATION_RELATIVE_PATH:
        raise PublisherSchemaError("template has the wrong artifact path")
    if template.get("role") != "golden":
        raise PublisherSchemaError("template has the wrong manifest role")
    fixed_values = {
        "format": "npz",
        "scope": (
            "Comparison-only Chapter 6 synthesis one-line oracle; opened "
            "only after the reader-built result exists."
        ),
        "builder": PUBLISHER_RELATIVE_PATH,
        "publisher_contract_sha256": CONTRACT_SHA256,
        "publisher_contract_acceptance_sha256": CONTRACT_AUDIT_SHA256,
        "candidate_byte_acceptance_sha256": CANDIDATE_ACCEPTANCE_SHA256,
        "atmosphere_candidate_byte_acceptance_sha256": (
            ATMOSPHERE_CANDIDATE_ACCEPTANCE_SHA256
        ),
        "writer": WRITER_RELATIVE_PATH,
        "writer_sha256": FIXED_UPSTREAM_IDENTITIES["writer"][1],
        "writer_tests_sha256": FIXED_UPSTREAM_IDENTITIES["writer_tests"][1],
        "writer_candidate_sha256": FIXED_UPSTREAM_IDENTITIES["writer_candidate"][1],
        "writer_acceptance_sha256": FIXED_UPSTREAM_IDENTITIES["writer_acceptance"][1],
        "payne_zero_commit": PINNED_PAYNE_ZERO_COMMIT,
        "archive_kind": EXPECTED_ARCHIVE_KIND,
        "comparison_only": True,
        "compact_schema_version": EXPECTED_COMPACT_SCHEMA_VERSION,
        "npy_member_format_version": "2.0",
        "compact_schema_digest": EXPECTED_COMPACT_SCHEMA_DIGEST,
        "compact_payload_fingerprint": (EXPECTED_COMPACT_PAYLOAD_FINGERPRINT),
        "raw_ownership_digest": EXPECTED_RAW_OWNERSHIP_DIGEST,
        "raw_member_count": EXPECTED_RAW_MEMBER_COUNT,
        "raw_schema_digest": EXPECTED_RAW_SCHEMA_DIGEST,
        "raw_physical_fingerprint": EXPECTED_RAW_PHYSICAL_FINGERPRINT,
        "raw_full_fingerprint": EXPECTED_RAW_FULL_FINGERPRINT,
        "reproducibility_environment": REPRODUCIBILITY_ENVIRONMENT,
    }
    for field, expected in fixed_values.items():
        if template[field] != expected:
            raise PublisherSchemaError(f"template field changed: {field}")
    for field in (
        "builder_sha256",
        "publisher_acceptance_sha256",
        "source_data_snapshot_sha256",
    ):
        _sha256_hex(template[field], label=f"template {field}")
    expected_upstream = {
        label: {"path": path, "sha256": digest}
        for label, (path, digest) in FIXED_UPSTREAM_IDENTITIES.items()
    }
    upstream = template["upstream_identities"]
    if (
        not isinstance(upstream, dict)
        or tuple(upstream) != tuple(expected_upstream)
        or upstream != expected_upstream
    ):
        raise PublisherSchemaError("template upstream identity mapping changed")
    for label, identity in upstream.items():
        _exact_keys(
            identity,
            IDENTITY_KEYS,
            label=f"template upstream identity {label}",
        )
    environment = template["reproducibility_environment"]
    if not isinstance(environment, dict) or tuple(environment) != tuple(
        REPRODUCIBILITY_ENVIRONMENT
    ):
        raise PublisherSchemaError("template reproducibility environment order changed")
    if template.get("sha256") != EXPECTED_ARCHIVE_SHA256:
        raise PublisherSchemaError("template has the wrong artifact hash")
    if template.get("bytes") != EXPECTED_ARCHIVE_BYTES:
        raise PublisherSchemaError("template has the wrong artifact size")
    arrays = template.get("arrays")
    if not isinstance(arrays, dict) or len(arrays) != EXPECTED_MEMBER_COUNT:
        raise PublisherSchemaError("template arrays mapping has the wrong size")
    if tuple(arrays) != tuple(sorted(arrays)):
        raise PublisherSchemaError("template array names are not lexical")
    for name, record in arrays.items():
        _identity_path(name, label=f"template array name {name}")
        checked = _exact_keys(
            record,
            ARRAY_RECORD_KEYS,
            label=f"template array record {name}",
        )
        if (
            not isinstance(checked["shape"], list)
            or any(type(size) is not int or size < 0 for size in checked["shape"])
            or not isinstance(checked["dtype"], str)
            or not isinstance(checked["unit"], str)
            or not checked["unit"].strip()
            or not isinstance(checked["axes"], list)
            or any(
                not isinstance(axis, str) or not axis.strip()
                for axis in checked["axes"]
            )
            or not isinstance(checked["ownership"], str)
            or not checked["ownership"].strip()
        ):
            raise PublisherSchemaError(
                f"template array metadata has wrong types: {name}"
            )
        _sha256_hex(checked["sha256"], label=f"template array hash {name}")
        if len(checked["axes"]) != len(checked["shape"]):
            raise PublisherSchemaError(f"template array axis rank changed: {name}")
    forbidden = {
        "realized_entry_sha256",
        "postpublication_manifest_sha256",
        "postpublication_audit_sha256",
        "authorization_review_sha256",
    }
    if forbidden.intersection(template):
        raise PublisherSchemaError("template contains a cycle-forming field")
    return template


def _validate_authorization(payload: bytes) -> dict[str, Any]:
    record = _strict_json(payload, label=AUTHORIZATION_RELATIVE_PATH)
    _exact_keys(record, AUTHORIZATION_KEYS, label="authorization")
    if (
        _plain_int(record["schema_version"], label="authorization schema")
        != AUTHORIZATION_SCHEMA_VERSION
        or record["record_kind"] != AUTHORIZATION_RECORD_KIND
        or record["lane"] != "synthesis"
        or record["role"] != "golden"
        or record["payne_zero_commit"] != PINNED_PAYNE_ZERO_COMMIT
    ):
        raise PublicationGateError("authorization lane identity changed")

    identities = record["identities"]
    if (
        not isinstance(identities, dict)
        or tuple(identities) != REQUIRED_IDENTITY_LABELS
    ):
        raise PublicationGateError("authorization identity set/order changed")
    for label, (path, digest) in FIXED_UPSTREAM_IDENTITIES.items():
        _validate_identity_record(
            label,
            identities[label],
            exact_path=path,
            exact_sha256=digest,
        )
    _validate_identity_record(
        "publisher",
        identities["publisher"],
        exact_path=PUBLISHER_RELATIVE_PATH,
        exact_sha256=None,
    )
    _validate_identity_record(
        "publisher_tests",
        identities["publisher_tests"],
        exact_path=PUBLISHER_TESTS_RELATIVE_PATH,
        exact_sha256=None,
    )
    _validate_identity_record(
        "publisher_acceptance",
        identities["publisher_acceptance"],
        exact_path=PUBLISHER_ACCEPTANCE_RELATIVE_PATH,
        exact_sha256=None,
    )

    artifact = _exact_keys(record["artifact"], ARTIFACT_KEYS, label="artifact")
    exact_artifact_values = {
        "path": DESTINATION_RELATIVE_PATH,
        "filename": DESTINATION_FILENAME,
        "role": "golden",
        "purpose": "comparison-only",
        "bytes": EXPECTED_ARCHIVE_BYTES,
        "sha256": EXPECTED_ARCHIVE_SHA256,
        "member_count": EXPECTED_MEMBER_COUNT,
        "archive_kind": EXPECTED_ARCHIVE_KIND,
        "comparison_only": True,
        "compact_schema_version": EXPECTED_COMPACT_SCHEMA_VERSION,
        "npy_member_format_version": "2.0",
        "schema_digest": EXPECTED_COMPACT_SCHEMA_DIGEST,
        "payload_fingerprint": EXPECTED_COMPACT_PAYLOAD_FINGERPRINT,
        "raw_ownership_digest": EXPECTED_RAW_OWNERSHIP_DIGEST,
        "raw_member_count": EXPECTED_RAW_MEMBER_COUNT,
        "raw_schema_digest": EXPECTED_RAW_SCHEMA_DIGEST,
        "raw_physical_fingerprint": EXPECTED_RAW_PHYSICAL_FINGERPRINT,
        "raw_full_fingerprint": EXPECTED_RAW_FULL_FINGERPRINT,
        "contains_copied_input_state": False,
        "contains_atmosphere_lane_members": False,
    }
    if artifact != exact_artifact_values:
        raise PublicationGateError("authorization artifact contract changed")

    premanifest = _exact_keys(
        record["prepublication_manifest"],
        PREPUBLICATION_MANIFEST_KEYS,
        label="prepublication manifest",
    )
    _identity_path(premanifest["path"], label="prepublication manifest path")
    _sha256_hex(premanifest["sha256"], label="prepublication manifest hash")
    _sha256_hex(
        premanifest["ordered_entry_path_digest"],
        label="prepublication ordered path digest",
    )
    _sha256_hex(
        premanifest["atmosphere_artifact_sha256"],
        label="atmosphere artifact hash",
    )
    _sha256_hex(
        premanifest["atmosphere_entry_digest"],
        label="atmosphere entry digest",
    )
    if (
        premanifest["path"] != MANIFEST_RELATIVE_PATH
        or _plain_int(
            premanifest["schema_version"], label="prepublication manifest schema"
        )
        != 1
        or premanifest["payne_zero_commit"] != PINNED_PAYNE_ZERO_COMMIT
        or _plain_bool(
            premanifest["destination_entry_absent"],
            label="destination entry absence",
        )
        is not True
        or premanifest["atmosphere_phase"] != "M1_registered_fixture"
        or premanifest["atmosphere_entry_path"] != ATMOSPHERE_DESTINATION_RELATIVE_PATH
    ):
        raise PublicationGateError(
            "authorization does not bind exact atmosphere-first manifest M1"
        )

    template = _validate_template_shape(record["manifest_entry_template"])
    if (
        template["builder_sha256"] != identities["publisher"]["sha256"]
        or template["publisher_acceptance_sha256"]
        != identities["publisher_acceptance"]["sha256"]
        or template["source_data_snapshot_sha256"]
        != record["source_data_snapshot_sha256"]
    ):
        raise PublicationGateError(
            "template does not bind publisher, publisher acceptance, or "
            "source/data snapshot"
        )
    template_hash = _sha256_hex(
        record["manifest_entry_template_sha256"],
        label="manifest template hash",
    )
    if sha256_bytes(_template_bytes(template)) != template_hash:
        raise PublicationGateError("manifest template digest does not recompute")
    _sha256_hex(
        record["source_data_snapshot_sha256"],
        label="source/data snapshot hash",
    )
    if (
        record["destination_parent_policy"] != DESTINATION_PARENT_POLICY
        or record["no_replace_primitive"] != NO_REPLACE_PRIMITIVE
    ):
        raise PublicationGateError("destination or no-replace policy changed")
    forbidden = {
        "authorization_sha256",
        "record_review_sha256",
        "realized_entry_sha256",
        "postpublication_manifest_sha256",
        "postpublication_audit_sha256",
    }
    if forbidden.intersection(record):
        raise PublicationGateError("authorization contains a backward hash edge")
    return record


def _validate_record_review(
    payload: bytes,
    *,
    authorization_sha256: str,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    review = _strict_json(payload, label=RECORD_REVIEW_RELATIVE_PATH)
    _exact_keys(review, RECORD_REVIEW_KEYS, label="authorization record review")
    for field in (
        "authorization_path",
        "candidate_byte_acceptance_path",
        "publisher_acceptance_path",
    ):
        _identity_path(review[field], label=f"review {field}")
    for field in (
        "authorization_sha256",
        "candidate_byte_acceptance_sha256",
        "publisher_acceptance_sha256",
        "manifest_entry_template_sha256",
    ):
        _sha256_hex(review[field], label=f"review {field}")
    publisher_acceptance_hash = authorization["identities"]["publisher_acceptance"][
        "sha256"
    ]
    exact = {
        "schema_version": RECORD_REVIEW_SCHEMA_VERSION,
        "record_kind": RECORD_REVIEW_KIND,
        "authorization_path": AUTHORIZATION_RELATIVE_PATH,
        "authorization_sha256": authorization_sha256,
        "candidate_byte_acceptance_path": CANDIDATE_ACCEPTANCE_RELATIVE_PATH,
        "candidate_byte_acceptance_sha256": CANDIDATE_ACCEPTANCE_SHA256,
        "publisher_acceptance_path": PUBLISHER_ACCEPTANCE_RELATIVE_PATH,
        "publisher_acceptance_sha256": publisher_acceptance_hash,
        "manifest_entry_template_sha256": authorization[
            "manifest_entry_template_sha256"
        ],
        "disposition": "ACCEPT",
    }
    if review != exact:
        raise PublicationGateError("authorization record review binding changed")
    return review


def _load_authority() -> Authority:
    """Load the complete authority chain before any candidate construction."""

    authorization_bytes, _ = _read_repository_file(
        AUTHORIZATION_RELATIVE_PATH,
        label=AUTHORIZATION_RELATIVE_PATH,
    )
    authorization = _validate_authorization(authorization_bytes)
    authorization_hash = sha256_bytes(authorization_bytes)
    review_bytes, _ = _read_repository_file(
        RECORD_REVIEW_RELATIVE_PATH,
        label=RECORD_REVIEW_RELATIVE_PATH,
    )
    review = _validate_record_review(
        review_bytes,
        authorization_sha256=authorization_hash,
        authorization=authorization,
    )
    return Authority(
        authorization=authorization,
        authorization_bytes=authorization_bytes,
        authorization_sha256=authorization_hash,
        review=review,
        review_bytes=review_bytes,
        review_sha256=sha256_bytes(review_bytes),
    )


def _verify_atmosphere_phase(
    manifest: Mapping[str, Any],
    premanifest: Mapping[str, Any],
) -> None:
    entries = manifest["entries"]
    matches = [
        entry
        for entry in entries
        if entry.get("path") == ATMOSPHERE_DESTINATION_RELATIVE_PATH
    ]
    if len(matches) != 1:
        raise PublicationGateError(
            "synthesis requires exactly one registered atmosphere fixture"
        )
    atmosphere = matches[0]
    if (
        atmosphere.get("role") != "fixture"
        or atmosphere.get("sha256") != premanifest["atmosphere_artifact_sha256"]
        or _entry_digest(atmosphere) != premanifest["atmosphere_entry_digest"]
    ):
        raise PublicationGateError("registered atmosphere fixture M1 changed")
    payload, _ = _read_repository_file(
        ATMOSPHERE_DESTINATION_RELATIVE_PATH,
        label=ATMOSPHERE_DESTINATION_RELATIVE_PATH,
    )
    if sha256_bytes(payload) != atmosphere.get("sha256"):
        raise PublicationGateError("registered atmosphere fixture bytes changed")


def _validate_prepublication_state(
    authority: Authority,
) -> tuple[
    dict[str, Any],
    bytes,
    DataSnapshot,
    dict[str, str],
    str,
    bytes,
]:
    current_manifest_bytes, _ = _read_repository_file(
        MANIFEST_RELATIVE_PATH,
        label=MANIFEST_RELATIVE_PATH,
    )
    current_manifest = _parse_manifest(current_manifest_bytes)
    premanifest = authority.authorization["prepublication_manifest"]
    destination_indices = [
        index
        for index, entry in enumerate(current_manifest["entries"])
        if entry["path"] == DESTINATION_RELATIVE_PATH
    ]
    if not destination_indices:
        manifest = current_manifest
        manifest_bytes = current_manifest_bytes
        lifecycle = "prepublication"
    elif destination_indices == [len(current_manifest["entries"]) - 1]:
        expected_entry = _realize_template(
            authority.authorization["manifest_entry_template"],
            authorization_sha256=authority.authorization_sha256,
            review_sha256=authority.review_sha256,
        )
        if current_manifest["entries"][-1] != expected_entry:
            raise PublicationStateError(
                "registered synthesis entry differs from unique realization"
            )
        manifest = deepcopy(current_manifest)
        manifest["entries"].pop()
        manifest_bytes = _manifest_bytes(manifest)
        lifecycle = "registered"
    else:
        raise PublicationStateError(
            "synthesis destination entry is duplicated or not append-last"
        )
    if (
        sha256_bytes(manifest_bytes) != premanifest["sha256"]
        or _ordered_path_digest(entry["path"] for entry in manifest["entries"])
        != premanifest["ordered_entry_path_digest"]
    ):
        raise PublicationGateError("authorization is stale for current manifest M1")
    _verify_atmosphere_phase(manifest, premanifest)
    snapshot = _snapshot_data(
        allow_exact_unregistered_target=(
            DESTINATION_RELATIVE_PATH
            if lifecycle == "prepublication" and DESTINATION_PATH.exists()
            else None
        )
    )
    source_identities = _verify_source_identity()
    reconstructed_ordered_digest = _ordered_path_digest(
        entry["path"] for entry in manifest["entries"]
    )
    if (
        _source_data_snapshot_digest(
            snapshot,
            source_identities,
            manifest_sha256=sha256_bytes(manifest_bytes),
            ordered_entry_path_digest=reconstructed_ordered_digest,
        )
        != authority.authorization["source_data_snapshot_sha256"]
    ):
        raise PublicationGateError("accepted source/data snapshot changed")
    return (
        manifest,
        manifest_bytes,
        snapshot,
        source_identities,
        lifecycle,
        current_manifest_bytes,
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    if array.dtype.hasobject or not array.flags.c_contiguous:
        raise PublisherSchemaError("NPY member must be object-free and C-contiguous")
    stream = BytesIO()
    np.lib.format.write_array(
        stream,
        array,
        version=EXPECTED_NPY_VERSION,
        allow_pickle=False,
    )
    return stream.getvalue()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=EXPECTED_ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.flag_bits = 0
    info.volume = 0
    info.internal_attr = 0
    info.external_attr = EXPECTED_ZIP_EXTERNAL_ATTR
    info.extra = b""
    info.comment = b""
    return info


def _encode_canonical_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    names = sorted(arrays)
    if names != list(arrays) or len(names) != len(set(names)):
        raise PublisherSchemaError("archive member mapping is not unique lexical")
    if any(
        not isinstance(name, str)
        or not name
        or "\\" in name
        or name.endswith(".npy")
        or any(part in {"", ".", ".."} for part in name.split("/"))
        for name in names
    ):
        raise PublisherSchemaError("archive mapping contains an unsafe member name")
    stream = BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for name in names:
            archive.writestr(
                _zip_info(f"{name}.npy"),
                _npy_bytes(np.asarray(arrays[name])),
                compress_type=zipfile.ZIP_STORED,
            )
    return stream.getvalue()


def _decode_canonical_npz(payload: bytes) -> dict[str, np.ndarray]:
    """Decode untrusted ZIP/NPY and require the one canonical encoding."""

    if not isinstance(payload, bytes) or not payload:
        raise PublisherSchemaError("candidate archive must be nonempty bytes")
    arrays: dict[str, np.ndarray] = {}
    try:
        with zipfile.ZipFile(BytesIO(payload), mode="r") as archive:
            if archive.comment:
                raise PublisherSchemaError("archive comment must be empty")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if names != sorted(names) or len(names) != len(set(names)):
                raise PublisherSchemaError("archive members must be unique and lexical")
            for info in infos:
                if (
                    not info.filename.endswith(".npy")
                    or "\\" in info.filename
                    or any(
                        part in {"", ".", ".."}
                        for part in info.filename.removesuffix(".npy").split("/")
                    )
                ):
                    raise PublisherSchemaError("archive contains unsafe member name")
                checks = (
                    info.date_time == EXPECTED_ZIP_DATE_TIME,
                    info.compress_type == zipfile.ZIP_STORED,
                    info.create_system == 3,
                    info.create_version == 20,
                    info.extract_version == 20,
                    info.flag_bits == 0,
                    info.volume == 0,
                    info.internal_attr == 0,
                    info.external_attr == EXPECTED_ZIP_EXTERNAL_ATTR,
                    info.extra == b"",
                    info.comment == b"",
                    info.compress_size == info.file_size,
                    not info.is_dir(),
                )
                if not all(checks):
                    raise PublisherSchemaError(
                        f"archive metadata changed: {info.filename}"
                    )
                npy_payload = archive.read(info)
                if not npy_payload.startswith(b"\x93NUMPY\x02\x00"):
                    raise PublisherSchemaError(f"NPY format changed: {info.filename}")
                array = np.lib.format.read_array(
                    BytesIO(npy_payload), allow_pickle=False
                )
                if array.dtype.hasobject or not array.flags.c_contiguous:
                    raise PublisherSchemaError(
                        f"unsafe array representation: {info.filename}"
                    )
                if _npy_bytes(array) != npy_payload:
                    raise PublisherSchemaError(
                        f"noncanonical NPY bytes: {info.filename}"
                    )
                arrays[info.filename.removesuffix(".npy")] = np.array(
                    array, copy=True, order="C"
                )
    except (
        EOFError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as error:
        raise PublisherSchemaError(f"invalid candidate archive: {error}") from error
    if _encode_canonical_npz(arrays) != payload:
        raise PublisherSchemaError("candidate is not the unique canonical NPZ")
    return arrays


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in arrays or np.asarray(arrays[name]).shape != ():
        raise PublisherSchemaError(f"required scalar is missing: {name}")
    return np.asarray(arrays[name]).item()


def _member_sha256(array: np.ndarray) -> str:
    """Hash only one logical array's contiguous C-order data bytes."""

    value = np.asarray(array)
    return sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def _member_metadata_digest(arrays: Mapping[str, Any]) -> str:
    """Bind every lexical member name to its exact scientific unit and axes."""

    metadata: dict[str, dict[str, Any]] = {}
    for name in sorted(arrays):
        record = arrays[name]
        if not isinstance(record, Mapping):
            raise PublisherSchemaError(f"array metadata is not an object: {name}")
        unit = record.get("unit")
        axes = record.get("axes")
        if (
            not isinstance(unit, str)
            or not unit.strip()
            or not isinstance(axes, list)
            or any(not isinstance(axis, str) or not axis.strip() for axis in axes)
        ):
            raise PublisherSchemaError(
                f"array unit/axis semantics are empty or invalid: {name}"
            )
        metadata[name] = {"unit": unit, "axes": list(axes)}
    return sha256_bytes(
        json.dumps(
            metadata,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _validate_expected_member_metadata(arrays: Mapping[str, Any]) -> None:
    if _member_metadata_digest(arrays) != EXPECTED_MEMBER_METADATA_DIGEST:
        raise PublisherSchemaError("template scientific units/axes mapping changed")


def _schema_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _payload_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        if name == "meta__compact_payload_fingerprint":
            continue
        value = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _validate_candidate_semantics(
    payload: bytes,
    *,
    template: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    if len(payload) != EXPECTED_ARCHIVE_BYTES:
        raise PublisherSchemaError("candidate archive size changed")
    if sha256_bytes(payload) != EXPECTED_ARCHIVE_SHA256:
        raise PublisherSchemaError("candidate whole-byte identity changed")
    arrays = _decode_canonical_npz(payload)
    if len(arrays) != EXPECTED_MEMBER_COUNT:
        raise PublisherSchemaError("candidate member count changed")
    exact_scalars = {
        "meta__archive_kind": EXPECTED_ARCHIVE_KIND,
        "meta__compact_schema_version": EXPECTED_COMPACT_SCHEMA_VERSION,
        "meta__compact_key_count": EXPECTED_MEMBER_COUNT,
        "meta__compact_schema_digest": EXPECTED_COMPACT_SCHEMA_DIGEST,
        "meta__compact_payload_fingerprint": (EXPECTED_COMPACT_PAYLOAD_FINGERPRINT),
        "meta__raw_ownership_digest": EXPECTED_RAW_OWNERSHIP_DIGEST,
        "meta__raw_ownership_complete": True,
        "meta__publication_authorized": False,
        "meta__golden_publication_performed": False,
    }
    for name, expected in exact_scalars.items():
        if _scalar(arrays, name) != expected:
            raise PublisherSchemaError(f"candidate semantic changed: {name}")
    if _schema_digest(arrays) != EXPECTED_COMPACT_SCHEMA_DIGEST:
        raise PublisherSchemaError("candidate schema digest does not recompute")
    if _payload_fingerprint(arrays) != EXPECTED_COMPACT_PAYLOAD_FINGERPRINT:
        raise PublisherSchemaError("candidate payload fingerprint does not recompute")
    forbidden_fragments = (
        "atmosphere_fixture",
        "fixture_role",
        "continuum_state_input",
        "copied_input",
    )
    if any(fragment in name for name in arrays for fragment in forbidden_fragments):
        raise PublisherSchemaError("candidate crossed fixture/input ownership")
    if template is None:
        return arrays
    template_arrays = template["arrays"]
    if tuple(template_arrays) != tuple(arrays):
        raise PublisherSchemaError("template and archive member sets disagree")
    _validate_expected_member_metadata(template_arrays)
    for name, array in arrays.items():
        record = template_arrays[name]
        if (
            record["shape"] != list(array.shape)
            or record["dtype"] != str(array.dtype)
            or record["sha256"] != _member_sha256(array)
            or len(record["axes"]) != array.ndim
            or record["ownership"] != "synthesis comparison golden"
        ):
            raise PublisherSchemaError(
                f"template metadata disagrees with candidate member: {name}"
            )
    return arrays


def _deterministic_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, dict) or tuple(summary) != WRITER_SUMMARY_KEYS:
        raise PublisherSchemaError("writer summary schema/order changed")
    evidence = summary.get("capture_process_evidence")
    if (
        not isinstance(evidence, dict)
        or tuple(evidence) != WRITER_PROCESS_EVIDENCE_KEYS
    ):
        raise PublisherSchemaError("writer process evidence schema/order changed")
    if summary["capture_process_evidence_sha256"] != sha256_bytes(
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ):
        raise PublisherSchemaError("writer process evidence digest changed")
    for field in (
        "capture_a_origin_token_sha256",
        "capture_b_origin_token_sha256",
        "capture_a_cache_root_sha256",
        "capture_b_cache_root_sha256",
    ):
        _sha256_hex(evidence[field], label=f"writer evidence {field}")
    for field in ("capture_a_child_pid", "capture_b_child_pid"):
        _plain_int(evidence[field], label=f"writer evidence {field}", minimum=1)
    if (
        evidence["capture_a_origin_token_sha256"]
        == evidence["capture_b_origin_token_sha256"]
        or evidence["capture_a_cache_root_sha256"]
        == evidence["capture_b_cache_root_sha256"]
        or evidence["capture_a_child_pid"] == evidence["capture_b_child_pid"]
    ):
        raise PublisherSchemaError("writer process origins are not distinct")
    topology_fields = (
        "capture_origins_distinct",
        "capture_child_processes_distinct",
        "capture_cache_roots_distinct",
        "capture_cache_roots_external",
        "capture_cache_roots_nonsymlink",
        "capture_cache_roots_empty_before",
        "capture_cache_roots_empty_after",
        "capture_cache_roots_disposed",
    )
    if any(evidence.get(field) is not True for field in topology_fields):
        raise PublisherSchemaError("writer topology/cache evidence changed")
    reduced = {
        key: deepcopy(value)
        for key, value in summary.items()
        if key not in {"capture_process_evidence", "capture_process_evidence_sha256"}
    }
    reduced["capture_topology"] = {field: True for field in topology_fields}
    exact = {
        "accepted_identity_count": 10,
        "writer_sha256": FIXED_UPSTREAM_IDENTITIES["writer"][1],
        "archive_a_sha256": EXPECTED_ARCHIVE_SHA256,
        "archive_b_sha256": EXPECTED_ARCHIVE_SHA256,
        "archive_a_b_byte_equal": True,
        "archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "archive_member_count": EXPECTED_MEMBER_COUNT,
        "compact_schema_digest": EXPECTED_COMPACT_SCHEMA_DIGEST,
        "compact_payload_fingerprint": EXPECTED_COMPACT_PAYLOAD_FINGERPRINT,
        "raw_ownership_digest": EXPECTED_RAW_OWNERSHIP_DIGEST,
        "raw_a_key_count": EXPECTED_RAW_MEMBER_COUNT,
        "raw_b_key_count": EXPECTED_RAW_MEMBER_COUNT,
        "raw_a_b_bitwise_equal": True,
        "raw_a_b_mapping_digest": EXPECTED_RAW_MAPPING_DIGEST,
        "raw_a_archive_sha256": EXPECTED_RAW_ARCHIVE_SHA256,
        "raw_b_archive_sha256": EXPECTED_RAW_ARCHIVE_SHA256,
        "raw_a_b_archive_byte_equal": True,
        "raw_archive_bytes": EXPECTED_RAW_ARCHIVE_BYTES,
        "raw_schema_digest": EXPECTED_RAW_SCHEMA_DIGEST,
        "raw_physical_fingerprint": EXPECTED_RAW_PHYSICAL_FINGERPRINT,
        "raw_full_fingerprint": EXPECTED_RAW_FULL_FINGERPRINT,
        "compact_a_b_schema_equal": True,
        "compact_a_b_payload_equal": True,
        "compact_a_b_ownership_equal": True,
        "compact_key_count": EXPECTED_MEMBER_COUNT,
        "compact_array_bytes": EXPECTED_COMPACT_ARRAY_BYTES,
        "raw_ownership_counts": EXPECTED_RAW_OWNERSHIP_COUNTS,
        "archive_member_order": "lexical",
        "archive_compression": "ZIP_STORED",
        "npy_version": [2, 0],
        "fixed_zip_date_time": [1980, 1, 1, 0, 0, 0],
        "disposable_cache_directories_created": True,
        "publication_authorized": False,
        "golden_publication_performed": False,
        "manifest_mutation_performed": False,
        "artifact_file_write_performed": False,
    }
    for name, expected in exact.items():
        if reduced.get(name) != expected:
            raise PublisherSchemaError(f"writer accepted summary changed: {name}")
    return reduced


@contextmanager
def _accepted_writer_environment() -> Iterable[None]:
    """Install the accepted child controls for two direct zero-argument calls."""

    from scripts import chapter06_synthesis_oracle_worker as worker

    controlled = dict(worker.ORACLE_ENVIRONMENT)
    controlled["PAYNE_ZERO_DATA_ROOT"] = str(worker.PINNED_DATA_ROOT)
    previous = {name: os.environ.get(name) for name in controlled}
    try:
        os.environ.update(controlled)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _build_candidate_twice(authority: Authority | None) -> Candidate:
    """Call the exact zero-argument writer twice and reduce nondeterminism."""

    from scripts import chapter06_synthesis_compact_writer as writer

    if (
        Path(writer.__file__).resolve(strict=True)
        != CANONICAL_REPOSITORY_ROOT / WRITER_RELATIVE_PATH
    ):
        raise PublisherIdentityError("synthesis writer imported from wrong path")
    with _accepted_writer_environment():
        first = writer.build_deterministic_compact_archive()
        second = writer.build_deterministic_compact_archive()
    if first.archive_bytes != second.archive_bytes:
        raise PublisherSchemaError("top-level candidate builds disagree")
    first_summary = _deterministic_summary(first.summary)
    second_summary = _deterministic_summary(second.summary)
    if first_summary != second_summary:
        raise PublisherSchemaError("top-level accepted summaries disagree")
    arrays = _validate_candidate_semantics(
        first.archive_bytes,
        template=(
            authority.authorization["manifest_entry_template"]
            if authority is not None
            else None
        ),
    )
    return Candidate(
        archive_bytes=bytes(first.archive_bytes),
        archive_sha256=sha256_bytes(first.archive_bytes),
        summary=first_summary,
        arrays=arrays,
    )


def _target_state(candidate_bytes: bytes) -> str:
    if not DESTINATION_PATH.exists() and not DESTINATION_PATH.is_symlink():
        return "absent"
    payload, details = _read_regular_nofollow(
        DESTINATION_PATH,
        label=DESTINATION_RELATIVE_PATH,
    )
    if details.st_nlink != 1:
        raise PublicationStateError("existing target must have exactly one link")
    if payload != candidate_bytes:
        raise PublicationStateError("existing target is not the authorized artifact")
    return "exact_unregistered"


def _realize_template(
    template: Mapping[str, Any],
    *,
    authorization_sha256: str,
    review_sha256: str,
) -> dict[str, Any]:
    before = deepcopy(template)
    realized = deepcopy(template)
    realized["publication_acceptance_sha256"] = authorization_sha256
    realized["publication_record_review_sha256"] = review_sha256
    differences: list[tuple[tuple[str, ...], Any, Any]] = []

    def compare(left: Any, right: Any, trail: tuple[str, ...]) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            if tuple(left) != tuple(right):
                differences.append((trail, tuple(left), tuple(right)))
                return
            for key in left:
                compare(left[key], right[key], (*trail, key))
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append((trail, len(left), len(right)))
                return
            for index, (first, second) in enumerate(zip(left, right, strict=True)):
                compare(first, second, (*trail, str(index)))
        elif left != right:
            differences.append((trail, left, right))

    compare(before, realized, ())
    expected = [
        (
            ("publication_acceptance_sha256",),
            LATE_AUTHORIZATION_SHA256,
            authorization_sha256,
        ),
        (
            ("publication_record_review_sha256",),
            LATE_RECORD_REVIEW_SHA256,
            review_sha256,
        ),
    ]
    if differences != expected:
        raise PublisherSchemaError("template realization changed more than two values")
    return realized


def _construct_postmanifest(
    premanifest: Mapping[str, Any],
    *,
    premanifest_bytes: bytes,
    entry: Mapping[str, Any],
) -> bytes:
    result = deepcopy(premanifest)
    result["entries"].append(deepcopy(entry))
    encoded = _manifest_bytes(result)
    reconstructed = deepcopy(result)
    removed = reconstructed["entries"].pop()
    if removed != entry or _manifest_bytes(reconstructed) != premanifest_bytes:
        raise PublisherSchemaError("manifest delete-last reconstruction failed")
    if reconstructed != premanifest:
        raise PublisherSchemaError("pre-existing manifest tree changed")
    return encoded


def _deterministic_report(
    authority: Authority,
    candidate: Candidate,
    *,
    manifest_before_sha256: str,
    target_state: str,
    intended_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "publisher": PUBLISHER_RELATIVE_PATH,
        "lane": "synthesis",
        "role": "golden",
        "destination": DESTINATION_RELATIVE_PATH,
        "authorization_sha256": authority.authorization_sha256,
        "record_review_sha256": authority.review_sha256,
        "contract_sha256": CONTRACT_SHA256,
        "candidate_byte_acceptance_sha256": CANDIDATE_ACCEPTANCE_SHA256,
        "artifact_sha256": candidate.archive_sha256,
        "artifact_bytes": len(candidate.archive_bytes),
        "archive_member_count": len(candidate.arrays),
        "manifest_before_sha256": manifest_before_sha256,
        "manifest_after_sha256": intended_manifest_sha256,
        "target_state": target_state,
        "atmosphere_phase": "M1_registered_fixture",
        "candidate_built_twice": True,
        "candidate_bytes_equal": True,
        "accepted_summaries_equal": True,
        "untrusted_decode_valid": True,
        "semantic_ownership_valid": True,
        "canonical_reencode_equal": True,
        "dry_run_repository_delta": "zero",
        "publication_performed": False,
    }


def _verification_report(
    candidate: Candidate,
    *,
    snapshot: DataSnapshot,
    target_state: str,
) -> dict[str, Any]:
    manifest = _parse_manifest(
        _read_repository_file(
            MANIFEST_RELATIVE_PATH,
            label=MANIFEST_RELATIVE_PATH,
        )[0]
    )
    atmosphere_entries = [
        entry
        for entry in manifest["entries"]
        if entry["path"] == ATMOSPHERE_DESTINATION_RELATIVE_PATH
    ]
    return {
        "schema_version": 1,
        "report_kind": "chapter06_synthesis_candidate_verification_only",
        "publisher": PUBLISHER_RELATIVE_PATH,
        "lane": "synthesis",
        "role": "golden",
        "destination": DESTINATION_RELATIVE_PATH,
        "authorization_checked": False,
        "publication_authorized": False,
        "publication_performed": False,
        "contract_sha256": CONTRACT_SHA256,
        "candidate_byte_acceptance_sha256": CANDIDATE_ACCEPTANCE_SHA256,
        "artifact_sha256": candidate.archive_sha256,
        "artifact_bytes": len(candidate.archive_bytes),
        "archive_member_count": len(candidate.arrays),
        "manifest_sha256": snapshot.manifest_sha256,
        "target_state": target_state,
        "atmosphere_phase_ready": (
            len(atmosphere_entries) == 1
            and atmosphere_entries[0].get("role") == "fixture"
        ),
        "candidate_built_twice": True,
        "candidate_bytes_equal": True,
        "accepted_summaries_equal": True,
        "untrusted_decode_valid": True,
        "canonical_reencode_equal": True,
        "verification_repository_delta": "zero",
        "decision": "VERIFIED_CANDIDATE_ONLY_NOT_AUTHORIZED",
    }


def _prepare_authorized() -> tuple[
    Authority,
    Candidate,
    dict[str, Any],
    bytes,
    DataSnapshot,
    dict[str, Any],
    bytes,
    str,
]:
    _require_canonical_execution()
    _verify_fixed_upstream_identities()
    authority = _load_authority()
    (
        manifest,
        manifest_bytes,
        snapshot,
        _source_identities,
        lifecycle,
        current_manifest_bytes,
    ) = _validate_prepublication_state(authority)
    candidate = _build_candidate_twice(authority)
    target = _target_state(candidate.archive_bytes)
    if lifecycle == "registered":
        if target != "exact_unregistered":
            raise PublicationStateError(
                "registered synthesis entry lacks its exact artifact"
            )
        state = "exact_registered"
    else:
        state = target
    realized = _realize_template(
        authority.authorization["manifest_entry_template"],
        authorization_sha256=authority.authorization_sha256,
        review_sha256=authority.review_sha256,
    )
    intended_manifest = _construct_postmanifest(
        manifest,
        premanifest_bytes=manifest_bytes,
        entry=realized,
    )
    if lifecycle == "registered" and intended_manifest != current_manifest_bytes:
        raise PublicationStateError(
            "registered manifest is not the unique M1-plus-entry result"
        )
    return (
        authority,
        candidate,
        manifest,
        manifest_bytes,
        snapshot,
        realized,
        intended_manifest,
        state,
    )


def verify_only() -> dict[str, Any]:
    """Verify accepted candidate bytes twice without loading publication authority."""

    _require_canonical_execution()
    upstream_before = _verify_fixed_upstream_identities()
    source_before = _verify_source_identity()
    snapshot_before = _snapshot_data(
        allow_exact_unregistered_target=(
            DESTINATION_RELATIVE_PATH if DESTINATION_PATH.exists() else None
        )
    )
    candidate = _build_candidate_twice(None)
    target = _target_state(candidate.archive_bytes)
    upstream_after = _verify_fixed_upstream_identities()
    source_after = _verify_source_identity()
    snapshot_after = _snapshot_data(
        allow_exact_unregistered_target=(
            DESTINATION_RELATIVE_PATH if DESTINATION_PATH.exists() else None
        )
    )
    if (
        upstream_after != upstream_before
        or source_after != source_before
        or snapshot_after != snapshot_before
    ):
        raise PublicationStateError(
            "verification-only execution observed a repository delta"
        )
    return _verification_report(
        candidate,
        snapshot=snapshot_after,
        target_state=target,
    )


def dry_run() -> dict[str, Any]:
    """Run the complete authorized verification path with no mutation calls."""

    before = _snapshot_data(
        allow_exact_unregistered_target=(
            DESTINATION_RELATIVE_PATH if DESTINATION_PATH.exists() else None
        )
    )
    (
        authority,
        candidate,
        _manifest,
        manifest_bytes,
        _snapshot,
        _entry,
        intended_manifest,
        state,
    ) = _prepare_authorized()
    after = _snapshot_data(
        allow_exact_unregistered_target=(
            DESTINATION_RELATIVE_PATH if DESTINATION_PATH.exists() else None
        )
    )
    if before != after:
        raise PublicationStateError("dry run changed canonical data state")
    return _deterministic_report(
        authority,
        candidate,
        manifest_before_sha256=sha256_bytes(manifest_bytes),
        target_state=state,
        intended_manifest_sha256=sha256_bytes(intended_manifest),
    )


@contextmanager
def _exclusive_data_lock() -> Iterable[int]:
    fd = _open_repository_directory("data", label="canonical data")
    try:
        details = os.fstat(fd)
        named = os.lstat(DATA_PATH)
        if not stat.S_ISDIR(details.st_mode) or (details.st_dev, details.st_ino) != (
            named.st_dev,
            named.st_ino,
        ):
            raise PublicationStateError("stable data-directory lock identity changed")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as error:
            raise PublicationIOError("platform data-directory lock failed") from error
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as error:
            raise PublicationIOError("data-directory lock release failed") from error
        finally:
            os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, _open_flags(directory=True))
    try:
        os.fsync(fd)
    except OSError as error:
        raise PublicationIOError(f"directory fsync failed: {path}") from error
    finally:
        os.close(fd)


def _ensure_destination_parent(
    invocation_created: list[InvocationDirectory] | None = None,
) -> tuple[Path, list[InvocationDirectory]]:
    """Create/adopt the two allowlisted components using retained descriptors."""

    created = invocation_created if invocation_created is not None else []
    root_fd = os.open(
        CANONICAL_REPOSITORY_ROOT,
        _open_flags(directory=True),
    )
    current_fd = root_fd
    current_path = CANONICAL_REPOSITORY_ROOT
    try:
        for component in ("data", "golden", "payne_zero"):
            child_fd = os.open(
                component,
                _open_flags(directory=True),
                dir_fd=current_fd,
            )
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd
            current_path /= component

        for component in ("chapter06", "synthesis"):
            parent_path = current_path
            parent_before = os.fstat(current_fd)
            candidate_path = parent_path / component
            made = False
            try:
                os.mkdir(component, mode=0o755, dir_fd=current_fd)
                made = True
            except FileExistsError:
                pass
            except OSError as error:
                raise PublicationIOError(
                    f"exclusive directory creation failed: {candidate_path}"
                ) from error

            try:
                child_fd = os.open(
                    component,
                    _open_flags(directory=True),
                    dir_fd=current_fd,
                )
            except OSError as error:
                raise PublicationIOError(
                    f"created/adopted directory cannot be opened: {candidate_path}"
                ) from error
            details = os.fstat(child_fd)
            if (
                not stat.S_ISDIR(details.st_mode)
                or stat.S_IMODE(details.st_mode) != 0o755
                or details.st_uid != parent_before.st_uid
                or details.st_gid != parent_before.st_gid
            ):
                os.close(child_fd)
                raise PublicationStateError(
                    f"destination directory metadata changed: {candidate_path}"
                )
            if made:
                created.append(
                    InvocationDirectory(
                        parent=parent_path,
                        path=candidate_path,
                        device=details.st_dev,
                        inode=details.st_ino,
                    )
                )

            allowed_names = (
                {"synthesis"} if component == "chapter06" else {DESTINATION_FILENAME}
            )
            actual_names = set(os.listdir(child_fd))
            if not actual_names.issubset(allowed_names):
                os.close(child_fd)
                raise PublicationStateError(
                    "existing destination directory has unexpected contents: "
                    f"{candidate_path}"
                )
            try:
                os.fsync(child_fd)
                os.fsync(current_fd)
            except OSError as error:
                os.close(child_fd)
                raise PublicationIOError(
                    f"nested destination directory fsync failed: {candidate_path}"
                ) from error

            named = os.stat(
                component,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            reopened_fd = os.open(
                component,
                _open_flags(directory=True),
                dir_fd=current_fd,
            )
            reopened = os.fstat(reopened_fd)
            if (
                (named.st_dev, named.st_ino) != (details.st_dev, details.st_ino)
                or (reopened.st_dev, reopened.st_ino)
                != (details.st_dev, details.st_ino)
                or reopened.st_uid != details.st_uid
                or reopened.st_gid != details.st_gid
                or stat.S_IMODE(reopened.st_mode) != stat.S_IMODE(details.st_mode)
                or (os.fstat(current_fd).st_dev, os.fstat(current_fd).st_ino)
                != (parent_before.st_dev, parent_before.st_ino)
            ):
                os.close(child_fd)
                os.close(reopened_fd)
                raise PublicationStateError(
                    "destination directory or parent changed during descent"
                )
            os.close(child_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = reopened_fd
            current_path = candidate_path
        return current_path, created
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def _cleanup_invocation_directories(
    created: Sequence[InvocationDirectory],
) -> None:
    for item in reversed(created):
        try:
            details = os.lstat(item.path)
        except FileNotFoundError:
            continue
        if (details.st_dev, details.st_ino) != (
            item.device,
            item.inode,
        ) or not stat.S_ISDIR(details.st_mode):
            raise PublicationIOError("refusing cleanup of changed invocation directory")
        try:
            os.rmdir(item.path)
        except OSError as error:
            raise PublicationIOError(
                f"invocation directory is not safely empty: {item.path}"
            ) from error
        _fsync_directory(item.parent)


def _write_all(fd: int, payload: bytes) -> None:
    try:
        written = os.write(fd, payload)
    except OSError as error:
        raise PublicationIOError("artifact write failed") from error
    if written != len(payload):
        raise PublicationIOError("short artifact write")


def _create_stage(
    parent: Path,
    payload: bytes,
    *,
    prefix: str,
    mode: int = 0o600,
    parent_fd: int | None = None,
) -> tuple[Path, int, os.stat_result]:
    name = prefix + secrets.token_hex(24)
    path = parent / name
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(
            name if parent_fd is not None else path,
            flags,
            mode,
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise PublicationIOError("exclusive same-filesystem stage failed") from error
    details: os.stat_result | None = None
    try:
        details = os.fstat(fd)
        try:
            os.fchmod(fd, mode)
        except OSError as error:
            raise PublicationIOError("stage mode enforcement failed") from error
        mode_details = os.fstat(fd)
        if (mode_details.st_dev, mode_details.st_ino) != (
            details.st_dev,
            details.st_ino,
        ):
            raise PublicationIOError("stage inode changed during mode enforcement")
        details = mode_details
        parent_details = (
            os.fstat(parent_fd) if parent_fd is not None else os.lstat(parent)
        )
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or details.st_dev != parent_details.st_dev
            or stat.S_IMODE(details.st_mode) != mode
        ):
            raise PublicationIOError("stage type/link/filesystem/mode is unsafe")
        _write_all(fd, payload)
        os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        readback = _read_all_fd(fd)
        retained = os.fstat(fd)
        named = (
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if parent_fd is not None
            else os.lstat(path)
        )
        if (
            readback != payload
            or (retained.st_dev, retained.st_ino) != (details.st_dev, details.st_ino)
            or (named.st_dev, named.st_ino) != (details.st_dev, details.st_ino)
            or retained.st_nlink != 1
            or named.st_nlink != 1
            or retained.st_size != len(payload)
            or named.st_size != len(payload)
            or stat.S_IMODE(retained.st_mode) != mode
            or stat.S_IMODE(named.st_mode) != mode
        ):
            raise PublicationIOError("stage readback or identity changed")
        details = retained
        return path, fd, details
    except BaseException:
        try:
            if details is not None and _stage_name_matches(
                path,
                details,
                parent_fd=parent_fd,
            ):
                _unlink_owned_stage(
                    path,
                    details,
                    parent_fd=parent_fd,
                )
        finally:
            os.close(fd)
        raise


def _stage_name_matches(
    path: Path,
    details: os.stat_result,
    *,
    parent_fd: int | None = None,
) -> bool:
    try:
        named = (
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if parent_fd is not None
            else os.lstat(path)
        )
    except FileNotFoundError:
        return False
    return stat.S_ISREG(named.st_mode) and (named.st_dev, named.st_ino) == (
        details.st_dev,
        details.st_ino,
    )


def _validate_retained_stage(
    path: Path,
    descriptor: int,
    details: os.stat_result,
    *,
    expected_bytes: bytes,
    expected_mode: int,
    parent_fd: int | None = None,
    expected_links: int = 1,
) -> None:
    retained = os.fstat(descriptor)
    named = (
        os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if parent_fd is not None
        else os.lstat(path)
    )
    if (
        not stat.S_ISREG(retained.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or (retained.st_dev, retained.st_ino) != (details.st_dev, details.st_ino)
        or (named.st_dev, named.st_ino) != (details.st_dev, details.st_ino)
        or retained.st_nlink != expected_links
        or named.st_nlink != expected_links
        or retained.st_size != len(expected_bytes)
        or named.st_size != len(expected_bytes)
        or stat.S_IMODE(retained.st_mode) != expected_mode
        or stat.S_IMODE(named.st_mode) != expected_mode
    ):
        raise PublicationIOError("retained stage/name identity changed")
    os.lseek(descriptor, 0, os.SEEK_SET)
    if _read_all_fd(descriptor) != expected_bytes:
        raise PublicationIOError("retained stage bytes changed")


def _unlink_owned_stage(
    path: Path,
    details: os.stat_result,
    *,
    parent_fd: int | None = None,
) -> None:
    named = (
        os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if parent_fd is not None
        else os.lstat(path)
    )
    if (named.st_dev, named.st_ino) != (details.st_dev, details.st_ino):
        raise PublicationIOError("refusing to unlink a changed staging inode")
    if parent_fd is None:
        os.unlink(path)
        _fsync_directory(path.parent)
    else:
        os.unlink(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)


def _link_validated_stage_no_replace(
    *,
    stage: Path,
    stage_fd: int,
    stage_details: os.stat_result,
    candidate_bytes: bytes,
    source_parent_fd: int,
    destination_parent_fd: int,
) -> None:
    """Bind the retained stage inode/name immediately to the no-replace syscall."""

    _validate_retained_stage(
        stage,
        stage_fd,
        stage_details,
        expected_bytes=candidate_bytes,
        expected_mode=0o600,
        parent_fd=source_parent_fd,
    )
    os.link(
        stage.name,
        DESTINATION_FILENAME,
        src_dir_fd=source_parent_fd,
        dst_dir_fd=destination_parent_fd,
        follow_symlinks=False,
    )


def _install_no_replace(
    parent: Path,
    *,
    stage: Path,
    stage_fd: int,
    stage_details: os.stat_result,
    candidate_bytes: bytes,
    parent_fd: int | None = None,
) -> str:
    source_parent_fd = (
        os.dup(parent_fd)
        if parent_fd is not None
        else os.open(parent, _open_flags(directory=True))
    )
    destination_parent_fd = os.dup(source_parent_fd)
    link_installed = False
    try:
        if parent_fd is not None:
            _revalidate_repository_directory(
                DESTINATION_PARENT_RELATIVE_PATH,
                parent_fd,
                label="destination parent",
            )
        try:
            _link_validated_stage_no_replace(
                stage=stage,
                stage_fd=stage_fd,
                stage_details=stage_details,
                candidate_bytes=candidate_bytes,
                source_parent_fd=source_parent_fd,
                destination_parent_fd=destination_parent_fd,
            )
        except FileExistsError:
            payload, details = _read_named_regular_at(
                destination_parent_fd,
                DESTINATION_FILENAME,
                label=DESTINATION_RELATIVE_PATH,
            )
            if payload != candidate_bytes or details.st_nlink != 1:
                raise PublicationStateError("no-replace race produced nonexact target")
            result = "exact_race_noop"
        except OSError as error:
            raise PublicationIOError("atomic no-replace hard-link failed") from error
        else:
            link_installed = True
            result = "installed"
            _validate_retained_stage(
                stage,
                stage_fd,
                stage_details,
                expected_bytes=candidate_bytes,
                expected_mode=0o600,
                parent_fd=source_parent_fd,
                expected_links=2,
            )
            installed_details = os.stat(
                DESTINATION_FILENAME,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
            if (installed_details.st_dev, installed_details.st_ino) != (
                stage_details.st_dev,
                stage_details.st_ino,
            ):
                raise PublicationIOError(
                    "no-replace target is not the retained stage inode"
                )
        os.fsync(destination_parent_fd)
        _rebind_retained_directory_path(
            parent,
            destination_parent_fd,
            label="destination parent after artifact mutation fsync",
        )
        _unlink_owned_stage(
            stage,
            stage_details,
            parent_fd=source_parent_fd,
        )
        _rebind_retained_directory_path(
            parent,
            destination_parent_fd,
            label="destination parent after stage cleanup fsync",
        )
        final_payload, final_details = _read_named_regular_at(
            destination_parent_fd,
            DESTINATION_FILENAME,
            label=DESTINATION_RELATIVE_PATH,
        )
        try:
            canonical_details = os.lstat(parent / DESTINATION_FILENAME)
        except OSError as error:
            raise PublicationStateError(
                "canonical no-replace target identity is unavailable"
            ) from error
        if (
            final_payload != candidate_bytes
            or final_details.st_nlink != 1
            or len(final_payload) != EXPECTED_ARCHIVE_BYTES
            or sha256_bytes(final_payload) != EXPECTED_ARCHIVE_SHA256
            or not stat.S_ISREG(canonical_details.st_mode)
            or (canonical_details.st_dev, canonical_details.st_ino)
            != (final_details.st_dev, final_details.st_ino)
        ):
            raise PublicationStateError("installed target failed complete validation")
        return result
    except BaseException:
        if not link_installed and _stage_name_matches(
            stage,
            stage_details,
            parent_fd=source_parent_fd,
        ):
            _unlink_owned_stage(
                stage,
                stage_details,
                parent_fd=source_parent_fd,
            )
        raise
    finally:
        os.close(stage_fd)
        os.close(destination_parent_fd)
        os.close(source_parent_fd)


def _revalidate_source_data_before_manifest(
    authority: Authority,
    *,
    expected_manifest_bytes: bytes,
    owned_manifest_temporary: OwnedManifestTemporary | None = None,
) -> None:
    """Recheck accepted sources and the exact normalized M1 data state."""

    _verify_fixed_upstream_identities()
    source_identities = _verify_source_identity()
    snapshot = _snapshot_data(
        allow_exact_unregistered_target=DESTINATION_RELATIVE_PATH,
        owned_manifest_temporary=owned_manifest_temporary,
    )
    expected_manifest = _parse_manifest(expected_manifest_bytes)
    expected_order = _ordered_path_digest(
        entry["path"] for entry in expected_manifest["entries"]
    )
    if (
        snapshot.manifest_sha256 != sha256_bytes(expected_manifest_bytes)
        or snapshot.ordered_entry_path_digest != expected_order
        or _source_data_snapshot_digest(
            snapshot,
            source_identities,
            manifest_sha256=sha256_bytes(expected_manifest_bytes),
            ordered_entry_path_digest=expected_order,
        )
        != authority.authorization["source_data_snapshot_sha256"]
    ):
        raise PublicationStateError(
            "accepted source or non-target data changed before manifest replacement"
        )


def _replace_validated_manifest_stage(
    *,
    stage: Path,
    stage_fd: int,
    stage_details: os.stat_result,
    intended_bytes: bytes,
    manifest_mode: int,
    data_fd: int | None,
) -> None:
    """Bind the retained temporary inode/name immediately to atomic replace."""

    _validate_retained_stage(
        stage,
        stage_fd,
        stage_details,
        expected_bytes=intended_bytes,
        expected_mode=manifest_mode,
        parent_fd=data_fd,
    )
    if data_fd is not None:
        _revalidate_repository_directory(
            "data",
            data_fd,
            label="canonical data",
        )
        os.replace(
            stage.name,
            Path(MANIFEST_RELATIVE_PATH).name,
            src_dir_fd=data_fd,
            dst_dir_fd=data_fd,
        )
    else:
        os.replace(stage, MANIFEST_PATH)


def _validate_replaced_manifest_from_retained(
    *,
    stage_fd: int,
    stage_details: os.stat_result,
    intended_bytes: bytes,
    manifest_mode: int,
    data_fd: int | None,
) -> None:
    retained = os.fstat(stage_fd)
    named = (
        os.stat(
            Path(MANIFEST_RELATIVE_PATH).name,
            dir_fd=data_fd,
            follow_symlinks=False,
        )
        if data_fd is not None
        else os.lstat(MANIFEST_PATH)
    )
    if (
        not stat.S_ISREG(retained.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or (retained.st_dev, retained.st_ino)
        != (stage_details.st_dev, stage_details.st_ino)
        or (named.st_dev, named.st_ino) != (stage_details.st_dev, stage_details.st_ino)
        or retained.st_nlink != 1
        or named.st_nlink != 1
        or retained.st_size != len(intended_bytes)
        or named.st_size != len(intended_bytes)
        or stat.S_IMODE(retained.st_mode) != manifest_mode
        or stat.S_IMODE(named.st_mode) != manifest_mode
    ):
        raise PublicationIOError(
            "replaced manifest is not the retained temporary inode"
        )
    os.lseek(stage_fd, 0, os.SEEK_SET)
    if _read_all_fd(stage_fd) != intended_bytes:
        raise PublicationIOError("replaced manifest retained bytes changed")


def _replace_manifest(
    intended_bytes: bytes,
    *,
    expected_before: bytes,
    candidate_bytes: bytes,
    authority: Authority | None = None,
    data_fd: int | None = None,
) -> None:
    manifest_named = (
        os.stat(
            Path(MANIFEST_RELATIVE_PATH).name,
            dir_fd=data_fd,
            follow_symlinks=False,
        )
        if data_fd is not None
        else os.lstat(MANIFEST_PATH)
    )
    manifest_mode = stat.S_IMODE(manifest_named.st_mode)
    stage, stage_fd, stage_details = _create_stage(
        DATA_PATH,
        intended_bytes,
        prefix=MANIFEST_TEMP_PREFIX,
        mode=manifest_mode,
        parent_fd=data_fd,
    )
    if data_fd is None:
        try:
            owned_parent_fd = os.open(DATA_PATH, _open_flags(directory=True))
        except OSError as error:
            try:
                if _stage_name_matches(stage, stage_details):
                    _unlink_owned_stage(stage, stage_details)
            finally:
                os.close(stage_fd)
            raise PublicationIOError(
                "manifest temporary parent identity is unavailable"
            ) from error
        close_owned_parent_fd = True
    else:
        owned_parent_fd = data_fd
        close_owned_parent_fd = False
    owned_manifest_temporary = OwnedManifestTemporary(
        path=stage,
        descriptor=stage_fd,
        parent_descriptor=owned_parent_fd,
        details=stage_details,
        intended_bytes=intended_bytes,
        intended_sha256=sha256_bytes(intended_bytes),
        mode=manifest_mode,
    )
    replaced = False
    try:
        _validate_retained_stage(
            stage,
            stage_fd,
            stage_details,
            expected_bytes=intended_bytes,
            expected_mode=manifest_mode,
            parent_fd=data_fd,
        )
        _parse_manifest(intended_bytes)
        current, current_details = (
            _read_named_regular_at(
                data_fd,
                Path(MANIFEST_RELATIVE_PATH).name,
                label=MANIFEST_RELATIVE_PATH,
            )
            if data_fd is not None
            else _read_regular_nofollow(
                MANIFEST_PATH,
                label=MANIFEST_RELATIVE_PATH,
            )
        )
        if current != expected_before:
            raise PublicationStateError("manifest changed before atomic replacement")
        final_artifact, _ = (
            _read_repository_file(
                DESTINATION_RELATIVE_PATH,
                label=DESTINATION_RELATIVE_PATH,
            )
            if data_fd is not None
            else _read_regular_nofollow(
                DESTINATION_PATH,
                label=DESTINATION_RELATIVE_PATH,
            )
        )
        if final_artifact != candidate_bytes:
            raise PublicationStateError("artifact changed before manifest replacement")
        if authority is not None:
            authorization_bytes, _ = _read_repository_file(
                AUTHORIZATION_RELATIVE_PATH,
                label=AUTHORIZATION_RELATIVE_PATH,
            )
            review_bytes, _ = _read_repository_file(
                RECORD_REVIEW_RELATIVE_PATH,
                label=RECORD_REVIEW_RELATIVE_PATH,
            )
            if (
                authorization_bytes != authority.authorization_bytes
                or review_bytes != authority.review_bytes
                or sha256_bytes(
                    _template_bytes(authority.authorization["manifest_entry_template"])
                )
                != authority.authorization["manifest_entry_template_sha256"]
                or _manifest_bytes(_parse_manifest(intended_bytes)) != intended_bytes
            ):
                raise PublicationStateError(
                    "authorization, review, template, or intended manifest "
                    "changed before replacement"
                )
            _revalidate_source_data_before_manifest(
                authority,
                expected_manifest_bytes=expected_before,
                owned_manifest_temporary=owned_manifest_temporary,
            )
        named_manifest = (
            os.stat(
                Path(MANIFEST_RELATIVE_PATH).name,
                dir_fd=data_fd,
                follow_symlinks=False,
            )
            if data_fd is not None
            else os.lstat(MANIFEST_PATH)
        )
        if (named_manifest.st_dev, named_manifest.st_ino) != (
            current_details.st_dev,
            current_details.st_ino,
        ):
            raise PublicationStateError("manifest inode changed before replacement")
        _replace_validated_manifest_stage(
            stage=stage,
            stage_fd=stage_fd,
            stage_details=stage_details,
            intended_bytes=intended_bytes,
            manifest_mode=manifest_mode,
            data_fd=data_fd,
        )
        replaced = True
        if data_fd is not None:
            os.fsync(data_fd)
            _rebind_retained_directory_path(
                DATA_PATH,
                data_fd,
                label="data after manifest replacement fsync",
            )
        else:
            _fsync_directory(DATA_PATH)
        _validate_replaced_manifest_from_retained(
            stage_fd=stage_fd,
            stage_details=stage_details,
            intended_bytes=intended_bytes,
            manifest_mode=manifest_mode,
            data_fd=data_fd,
        )
        try:
            canonical_manifest = os.lstat(MANIFEST_PATH)
        except OSError as error:
            raise PublicationStateError(
                "canonical replaced manifest identity is unavailable"
            ) from error
        if not stat.S_ISREG(canonical_manifest.st_mode) or (
            canonical_manifest.st_dev,
            canonical_manifest.st_ino,
        ) != (stage_details.st_dev, stage_details.st_ino):
            raise PublicationStateError(
                "canonical replaced manifest is not the retained temporary inode"
            )
    except BaseException:
        if not replaced and _stage_name_matches(
            stage,
            stage_details,
            parent_fd=data_fd,
        ):
            _unlink_owned_stage(
                stage,
                stage_details,
                parent_fd=data_fd,
            )
        raise
    finally:
        if close_owned_parent_fd:
            os.close(owned_parent_fd)
        os.close(stage_fd)
    final, _ = (
        _read_named_regular_at(
            data_fd,
            Path(MANIFEST_RELATIVE_PATH).name,
            label=MANIFEST_RELATIVE_PATH,
        )
        if data_fd is not None
        else _read_regular_nofollow(
            MANIFEST_PATH,
            label=MANIFEST_RELATIVE_PATH,
        )
    )
    if final != intended_bytes:
        raise PublicationIOError("manifest replacement readback changed")


def _registered_validation_report() -> dict[str, Any]:
    """Run the complete fixed-root registered-state validator."""

    (
        authority,
        candidate,
        _manifest,
        _manifest_bytes,
        snapshot,
        entry,
        intended_manifest,
        state,
    ) = _prepare_authorized()
    if state != "exact_registered":
        raise PublicationStateError(
            "fresh validator did not observe unique registered M2"
        )
    target, details = _read_repository_file(
        DESTINATION_RELATIVE_PATH,
        label=DESTINATION_RELATIVE_PATH,
    )
    if (
        target != candidate.archive_bytes
        or details.st_nlink != 1
        or stat.S_IMODE(details.st_mode) != 0o600
    ):
        raise PublicationStateError(
            "fresh validator found a nonexact registered artifact"
        )
    manifest_now, _ = _read_repository_file(
        MANIFEST_RELATIVE_PATH,
        label=MANIFEST_RELATIVE_PATH,
    )
    if manifest_now != intended_manifest:
        raise PublicationStateError(
            "fresh validator found a changed registered manifest"
        )
    return {
        "schema_version": 1,
        "state": "exact_registered",
        "lane": "synthesis",
        "role": "golden",
        "artifact_sha256": candidate.archive_sha256,
        "manifest_sha256": sha256_bytes(manifest_now),
        "realized_entry_sha256": _entry_digest(entry),
        "authorization_sha256": authority.authorization_sha256,
        "record_review_sha256": authority.review_sha256,
        "source_data_snapshot_sha256": authority.authorization[
            "source_data_snapshot_sha256"
        ],
        "data_aggregate_sha256": snapshot.aggregate_sha256,
        "member_metadata_digest": _member_metadata_digest(
            authority.authorization["manifest_entry_template"]["arrays"]
        ),
        "atmosphere_phase": "M1_registered_fixture",
        "complete_validation": True,
    }


def verify_published() -> dict[str, Any]:
    """Validate the immutable Chapter 6 result in an evolving textbook tree.

    Publication authority records historical builder identities.  After the
    transition succeeds, later chapters may add data and the maintenance
    implementation may evolve.  This validator therefore checks the closed
    authority -> review -> realized manifest entry -> exact artifact graph,
    without pretending that the original prepublication snapshot is still the
    live repository state.
    """

    _require_canonical_execution()
    snapshot = _snapshot_data()
    authorization_bytes, _ = _read_repository_file(
        AUTHORIZATION_RELATIVE_PATH,
        label=AUTHORIZATION_RELATIVE_PATH,
    )
    authorization = _strict_json(
        authorization_bytes,
        label=AUTHORIZATION_RELATIVE_PATH,
    )
    _exact_keys(authorization, AUTHORIZATION_KEYS, label="authorization")
    if (
        authorization["schema_version"] != AUTHORIZATION_SCHEMA_VERSION
        or authorization["record_kind"] != AUTHORIZATION_RECORD_KIND
        or authorization["lane"] != "synthesis"
        or authorization["role"] != "golden"
        or authorization["payne_zero_commit"] != PINNED_PAYNE_ZERO_COMMIT
    ):
        raise PublicationGateError("published authorization lane identity changed")
    artifact = _exact_keys(
        authorization["artifact"],
        ARTIFACT_KEYS,
        label="published artifact",
    )
    if (
        artifact["path"] != DESTINATION_RELATIVE_PATH
        or artifact["bytes"] != EXPECTED_ARCHIVE_BYTES
        or artifact["sha256"] != EXPECTED_ARCHIVE_SHA256
        or artifact["member_count"] != EXPECTED_MEMBER_COUNT
        or artifact["archive_kind"] != EXPECTED_ARCHIVE_KIND
        or artifact["schema_digest"] != EXPECTED_COMPACT_SCHEMA_DIGEST
        or artifact["payload_fingerprint"] != EXPECTED_COMPACT_PAYLOAD_FINGERPRINT
        or artifact["raw_ownership_digest"] != EXPECTED_RAW_OWNERSHIP_DIGEST
    ):
        raise PublicationGateError("published artifact authority changed")
    template = _validate_template_shape(authorization["manifest_entry_template"])
    if (
        sha256_bytes(_template_bytes(template))
        != authorization["manifest_entry_template_sha256"]
    ):
        raise PublicationGateError("published manifest template digest changed")
    identities = authorization["identities"]
    if not isinstance(identities, dict) or tuple(identities) != REQUIRED_IDENTITY_LABELS:
        raise PublicationGateError("published authorization identity set changed")
    if (
        template["builder_sha256"] != identities["publisher"]["sha256"]
        or template["publisher_acceptance_sha256"]
        != identities["publisher_acceptance"]["sha256"]
        or template["source_data_snapshot_sha256"]
        != authorization["source_data_snapshot_sha256"]
    ):
        raise PublicationGateError("published template authority joins changed")

    authorization_sha256 = sha256_bytes(authorization_bytes)
    review_bytes, _ = _read_repository_file(
        RECORD_REVIEW_RELATIVE_PATH,
        label=RECORD_REVIEW_RELATIVE_PATH,
    )
    review = _validate_record_review(
        review_bytes,
        authorization_sha256=authorization_sha256,
        authorization=authorization,
    )
    review_sha256 = sha256_bytes(review_bytes)
    expected_entry = _realize_template(
        template,
        authorization_sha256=authorization_sha256,
        review_sha256=review_sha256,
    )
    manifest_payload, _ = _read_repository_file(
        MANIFEST_RELATIVE_PATH,
        label=MANIFEST_RELATIVE_PATH,
    )
    manifest = _parse_manifest(manifest_payload)
    matches = [
        entry
        for entry in manifest["entries"]
        if entry["path"] == DESTINATION_RELATIVE_PATH
    ]
    if matches != [expected_entry]:
        raise PublicationStateError(
            "published synthesis entry differs from its unique realization"
        )

    target, target_details = _read_repository_file(
        DESTINATION_RELATIVE_PATH,
        label=DESTINATION_RELATIVE_PATH,
    )
    if (
        len(target) != EXPECTED_ARCHIVE_BYTES
        or sha256_bytes(target) != EXPECTED_ARCHIVE_SHA256
        or target_details.st_nlink != 1
        or stat.S_IMODE(target_details.st_mode) != 0o600
    ):
        raise PublicationStateError("published synthesis artifact identity changed")
    arrays = _validate_candidate_semantics(target, template=template)
    atmosphere_matches = [
        entry
        for entry in manifest["entries"]
        if entry["path"] == ATMOSPHERE_DESTINATION_RELATIVE_PATH
    ]
    if len(atmosphere_matches) != 1 or atmosphere_matches[0]["role"] != "fixture":
        raise PublicationStateError("published atmosphere prerequisite changed")
    atmosphere_payload, _ = _read_repository_file(
        ATMOSPHERE_DESTINATION_RELATIVE_PATH,
        label=ATMOSPHERE_DESTINATION_RELATIVE_PATH,
    )
    if (
        sha256_bytes(atmosphere_payload) != atmosphere_matches[0]["sha256"]
        or len(atmosphere_payload) != atmosphere_matches[0]["bytes"]
    ):
        raise PublicationStateError("published atmosphere prerequisite bytes changed")
    return {
        "schema_version": 1,
        "report_kind": "chapter06_synthesis_postpublication_verification",
        "state": "exact_registered",
        "artifact_sha256": EXPECTED_ARCHIVE_SHA256,
        "artifact_bytes": EXPECTED_ARCHIVE_BYTES,
        "archive_member_count": len(arrays),
        "authorization_sha256": authorization_sha256,
        "record_review_sha256": review_sha256,
        "realized_entry_sha256": _entry_digest(expected_entry),
        "manifest_sha256": snapshot.manifest_sha256,
        "later_chapter_support_file_count": max(
            0, len(snapshot.nonmanifest_support) - 1
        ),
        "complete_validation": True,
    }


def _fresh_validate_expected_final(
    *,
    authority: Authority,
    intended_manifest: bytes,
    candidate_bytes: bytes,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CANONICAL_REPOSITORY_ROOT / PUBLISHER_RELATIVE_PATH),
            "--internal-validate-published",
        ],
        cwd=CANONICAL_REPOSITORY_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(CANONICAL_REPOSITORY_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0 or completed.stderr:
        raise PublicationStateError("fresh-process final validation failed")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PublicationStateError(
            "fresh-process final validation returned invalid JSON"
        ) from error
    expected = {
        "schema_version": 1,
        "state": "exact_registered",
        "lane": "synthesis",
        "role": "golden",
        "artifact_sha256": sha256_bytes(candidate_bytes),
        "manifest_sha256": sha256_bytes(intended_manifest),
        "realized_entry_sha256": _entry_digest(
            _realize_template(
                authority.authorization["manifest_entry_template"],
                authorization_sha256=authority.authorization_sha256,
                review_sha256=authority.review_sha256,
            )
        ),
        "authorization_sha256": authority.authorization_sha256,
        "record_review_sha256": authority.review_sha256,
        "source_data_snapshot_sha256": authority.authorization[
            "source_data_snapshot_sha256"
        ],
        "member_metadata_digest": EXPECTED_MEMBER_METADATA_DIGEST,
        "atmosphere_phase": "M1_registered_fixture",
        "complete_validation": True,
    }
    if (
        not isinstance(report, dict)
        or set(report) != {*expected, "data_aggregate_sha256"}
        or any(report.get(key) != value for key, value in expected.items())
        or not isinstance(report.get("data_aggregate_sha256"), str)
        or len(report["data_aggregate_sha256"]) != 64
    ):
        raise PublicationStateError("fresh-process final validation changed")
    if authority.review["authorization_sha256"] != authority.authorization_sha256:
        raise PublicationStateError("authority changed after publication")


def publish() -> dict[str, Any]:
    """Publish exactly one authorized golden and append exactly one entry."""

    (
        authority,
        candidate,
        manifest,
        manifest_bytes,
        initial_snapshot,
        _entry,
        intended_manifest,
        initial_state,
    ) = _prepare_authorized()
    created: list[InvocationDirectory] = []
    with _exclusive_data_lock() as data_fd:
        # The lock is acquired only after authorization and candidate validation.
        (
            current_authority,
            current_candidate,
            current_manifest,
            current_manifest_bytes,
            current_snapshot,
            _current_entry,
            current_intended_manifest,
            current_state,
        ) = _prepare_authorized()
        if (
            current_authority != authority
            or current_candidate.archive_bytes != candidate.archive_bytes
            or current_manifest != manifest
            or current_manifest_bytes != manifest_bytes
            or current_snapshot != initial_snapshot
            or current_intended_manifest != intended_manifest
            or current_state != initial_state
        ):
            raise PublicationStateError("authorized state changed under data lock")
        if current_state == "exact_registered":
            installation = "exact_registered_validation_noop"
        else:
            parent_fd: int | None = None
            try:
                parent, created = _ensure_destination_parent(created)
                parent_fd = _open_repository_directory(
                    DESTINATION_PARENT_RELATIVE_PATH,
                    label="destination parent",
                )
                if current_state == "absent":
                    stage, stage_fd, stage_details = _create_stage(
                        parent,
                        candidate.archive_bytes,
                        prefix=STAGE_PREFIX,
                        parent_fd=parent_fd,
                    )
                    try:
                        os.lseek(stage_fd, 0, os.SEEK_SET)
                        staged_bytes = _read_all_fd(stage_fd)
                        _validate_candidate_semantics(
                            staged_bytes,
                            template=authority.authorization["manifest_entry_template"],
                        )
                    except BaseException:
                        if _stage_name_matches(
                            stage,
                            stage_details,
                            parent_fd=parent_fd,
                        ):
                            _unlink_owned_stage(
                                stage,
                                stage_details,
                                parent_fd=parent_fd,
                            )
                        os.close(stage_fd)
                        raise
                    installation = _install_no_replace(
                        parent,
                        stage=stage,
                        stage_fd=stage_fd,
                        stage_details=stage_details,
                        candidate_bytes=candidate.archive_bytes,
                        parent_fd=parent_fd,
                    )
                else:
                    installation = "exact_unregistered_recovery"
                _replace_manifest(
                    intended_manifest,
                    expected_before=manifest_bytes,
                    candidate_bytes=candidate.archive_bytes,
                    authority=authority,
                    data_fd=data_fd,
                )
            except BaseException:
                # Never delete an installed exact artifact.  Only invocation-owned
                # empty directories can be removed, and only if still empty/exact.
                if not DESTINATION_PATH.exists() and created:
                    _cleanup_invocation_directories(created)
                raise
            finally:
                if parent_fd is not None:
                    os.close(parent_fd)
        _fresh_validate_expected_final(
            authority=authority,
            intended_manifest=intended_manifest,
            candidate_bytes=candidate.archive_bytes,
        )
    report = _deterministic_report(
        authority,
        candidate,
        manifest_before_sha256=sha256_bytes(manifest_bytes),
        target_state=initial_state,
        intended_manifest_sha256=sha256_bytes(intended_manifest),
    )
    report["publication_performed"] = installation != "exact_registered_validation_noop"
    report["installation_result"] = installation
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify or publish the fixed Chapter 6 synthesis golden."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="rebuild and validate the candidate without publication authority",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="perform authorized verification without any repository write",
    )
    mode.add_argument(
        "--publish",
        action="store_true",
        help="perform the reviewed fixed-path no-replace publication",
    )
    mode.add_argument(
        "--verify-published",
        action="store_true",
        help="validate the registered artifact and historical authority graph",
    )
    mode.add_argument(
        "--internal-validate-published",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        if arguments.verify_only:
            result = verify_only()
        elif arguments.dry_run:
            result = dry_run()
        elif arguments.publish:
            result = publish()
        elif arguments.verify_published:
            result = verify_published()
        else:
            result = _registered_validation_report()
    except PublisherError as error:
        print(
            json.dumps(
                {
                    "status": "REJECT",
                    "reason": str(error),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_RELATIVE_PATH",
    "CANDIDATE_ACCEPTANCE_SHA256",
    "CONTRACT_SHA256",
    "DESTINATION_RELATIVE_PATH",
    "EXPECTED_ARCHIVE_BYTES",
    "EXPECTED_ARCHIVE_SHA256",
    "EXPECTED_MEMBER_COUNT",
    "MANIFEST_RELATIVE_PATH",
    "PUBLISHER_RELATIVE_PATH",
    "PublisherError",
    "PublisherIdentityError",
    "PublisherSchemaError",
    "PublicationGateError",
    "PublicationIOError",
    "PublicationStateError",
    "RECORD_REVIEW_RELATIVE_PATH",
    "dry_run",
    "main",
    "parse_args",
    "publish",
    "verify_only",
]
