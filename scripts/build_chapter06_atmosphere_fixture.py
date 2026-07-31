#!/usr/bin/env python3
"""Verify and, only when separately authorized, publish the Chapter 6 fixture.

This module is deliberately narrower than a conventional data builder:

* the accepted scientific writer is invoked twice in unrelated top-level
  interpreter processes;
* its returned NPZ bytes are treated as untrusted and independently decoded,
  validated, and canonically re-encoded;
* verification-only mode cannot reach a publication primitive; and
* publication has one repository, one destination, one role, and no
  caller-selected path, force, replace, repair, or merge option.

The detached authorization and its independent JSON review do not exist while
this implementation candidate is being authored.  Consequently ``publish()``
fails closed before it creates a lock or a staging inode.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence
import unicodedata

_BOOTSTRAP_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPOSITORY_ROOT))

from scripts import chapter06_atmosphere_fixture_worker as worker  # noqa: E402
from scripts import chapter06_atmosphere_fixture_writer as writer  # noqa: E402


CANONICAL_REPOSITORY_ROOT = Path("/Users/ysting/stellar-spectroscopy-from-scratch-gpu")
PINNED_PAYNE_ZERO_ROOT = Path("/Users/ysting/payne-zero")
PAPER_ROOT = Path("/Users/ysting/Source_Files_Not_For_Review")
PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"

PUBLISHER_IDENTITY = "scripts/build_chapter06_atmosphere_fixture.py"
PUBLISHER_TEST_IDENTITY = "tests/test_chapter06_atmosphere_fixture_publisher.py"
PUBLISHER_ACCEPTANCE_IDENTITY = (
    "design/chapter06_atmosphere_fixture_publisher_independent_audit.md"
)
AUTHORIZATION_IDENTITY = (
    "design/chapter06_atmosphere_fixture_publication_acceptance.json"
)
RECORD_REVIEW_IDENTITY = (
    "design/chapter06_atmosphere_fixture_publication_record_review.json"
)
POSTPUBLICATION_AUDIT_IDENTITY = (
    "design/chapter06_atmosphere_fixture_postpublication_audit.md"
)
QUARANTINE_CLEANUP_IDENTITY = (
    "design/chapter06_publication_quarantine_cleanup_acceptance.json"
)
MANIFEST_IDENTITY = "data/MANIFEST.json"
DESTINATION_IDENTITY = "data/fixtures/chapter06_atmosphere_one_line_inputs.npz"

CONTRACT_IDENTITY = "design/chapter06_lane_artifact_publisher_contract.md"
CONTRACT_SHA256 = "3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b"
CONTRACT_AUDIT_IDENTITY = (
    "design/chapter06_lane_artifact_publisher_contract_rebind_independent_audit.md"
)
CONTRACT_AUDIT_SHA256 = (
    "fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc"
)
CANDIDATE_BYTE_ACCEPTANCE_IDENTITY = (
    "design/chapter06_atmosphere_fixture_byte_acceptance.md"
)
CANDIDATE_BYTE_ACCEPTANCE_SHA256 = (
    "8298b9473cf89161441bbd72a881c744e38fba699aa088eb876014642c91ed71"
)
SYNTHESIS_CANDIDATE_BYTE_ACCEPTANCE_IDENTITY = (
    "design/chapter06_synthesis_candidate_byte_acceptance.md"
)
SYNTHESIS_CANDIDATE_BYTE_ACCEPTANCE_SHA256 = (
    "434088cff95ed60d65dc6c9749d18c2e74e45d114787c03728ed7ae9cf0bd9c9"
)

WRITER_IDENTITY = "scripts/chapter06_atmosphere_fixture_writer.py"
WRITER_SHA256 = "0c6a3300c6ce98e5d8b9a31fec1dec6783e9928f7f8df1ecccec33d9feda2538"
WRITER_TEST_IDENTITY = "tests/test_chapter06_atmosphere_fixture_writer.py"
WRITER_TEST_SHA256 = "c741d06abaae2e09c9fa0736d6abd26f609e01e9414cae131309bbc202e27a4a"
WRITER_CANDIDATE_IDENTITY = "design/chapter06_atmosphere_fixture_writer_candidate.md"
WRITER_CANDIDATE_SHA256 = (
    "e6d9bc2120eee12d5776e48953a64da68e8aa0e8812d6ce5e0b43c792f2571ee"
)
WRITER_ACCEPTANCE_IDENTITY = (
    "design/chapter06_atmosphere_fixture_writer_independent_audit.md"
)
WRITER_ACCEPTANCE_SHA256 = (
    "b946dcef0beeacf49a3da9ac036e21af7cd7b44d15092639cd3be744fb42f0f9"
)

EXPECTED_ARCHIVE_SHA256 = (
    "1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff"
)
EXPECTED_ARCHIVE_BYTES = 363_050
EXPECTED_ARRAY_BYTES = 357_984
EXPECTED_MEMBER_COUNT = 19
EXPECTED_FIXTURE_SCHEMA_DIGEST = (
    "f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698"
)
EXPECTED_PAYLOAD_FINGERPRINT = (
    "f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663"
)
EXPECTED_MAPPING_DIGEST = (
    "f533e3e327c879b1d367a89822bcd1847b15a73ba3d234a9c577c81997f75e0a"
)
EXPECTED_EVIDENCE_MAPPING_DIGEST = (
    "f5e49f9f7f49c4604c08a53def65a9ee3b6ddff37376130eb274646ee375af4f"
)
EXPECTED_FULL_CAPTURE_SCHEMA_DIGEST = (
    "cdf470038e67301b4c19b0691e672cd97df3233a3decf1b88e32ce3ac0dc1371"
)
EXPECTED_FULL_CAPTURE_FINGERPRINT = (
    "a25875097c7084ffe2577de65c2913d8775f29613823d9e6e0ab0d9db4644654"
)
EXPECTED_CAPTURE_TRANSPORT_SHA256 = (
    "0523ed254a78edaa07480bc30f23082f535e885f46d867de10f92cba1acd5b16"
)
EXPECTED_CAPTURE_TRANSPORT_BYTES = 33_147_771
EXPECTED_EVIDENCE_MEMBER_COUNT = 89
EXTERNAL_CAPTURE_SCHEMA_VERSION = 1
NPY_MEMBER_FORMAT_VERSION = "2.0"
ARCHIVE_KIND = "atmosphere_one_line_input_fixture"
MANIFEST_ROLE = "fixture"

AUTHORIZATION_PLACEHOLDER = "__LATE_BOUND_AUTHORIZATION_SHA256__"
RECORD_REVIEW_PLACEHOLDER = "__LATE_BOUND_RECORD_REVIEW_SHA256__"
ARTIFACT_STAGE_PREFIX = ".chapter06-atmosphere-stage-"
MANIFEST_TEMPORARY_PREFIX = ".chapter06-atmosphere-manifest-"

MANIFEST_TOP_LEVEL_KEYS = (
    "schema_version",
    "payne_zero_commit",
    "entries",
)
AUTHORIZATION_KEYS = (
    "schema_version",
    "record_kind",
    "lane",
    "manifest_role",
    "payne_zero_commit",
    "writer",
    "writer_tests",
    "writer_candidate",
    "writer_acceptance",
    "candidate_byte_acceptance",
    "publisher",
    "publisher_tests",
    "publisher_contract",
    "publisher_contract_audit",
    "publisher_acceptance",
    "artifact",
    "manifest",
    "manifest_entry_template",
    "manifest_entry_template_sha256",
    "source_snapshot_sha256",
    "data_snapshot_sha256",
    "destination_parent_policy",
    "no_replace_primitive",
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
IDENTITY_KEYS = ("path", "sha256")
ARTIFACT_AUTHORIZATION_KEYS = (
    "path",
    "filename",
    "bytes",
    "sha256",
    "member_count",
    "archive_kind",
    "fixture_capture_schema_version",
    "archive_contains_embedded_schema_version",
    "npy_member_format_version",
    "scientific_fixture_schema_digest",
    "scientific_payload_fingerprint",
)
MANIFEST_AUTHORIZATION_KEYS = (
    "path",
    "prepublication_sha256",
    "schema_version",
    "payne_zero_commit",
    "ordered_entry_path_sha256",
    "destination_entry_absent",
)
ARRAY_RECORD_KEYS = (
    "shape",
    "dtype",
    "unit",
    "sha256",
    "axes",
    "ownership",
)

EXPECTED_PREPUBLICATION_DIRECTORY_IDENTITIES = (
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
README_IDENTITY = "data/README.md"
README_SHA256 = "1a1028744b7e72e24e5a0831a68900d93345c8a946335a6f6de0358754b5bf2b"

# The common writer gate is intentionally checked by either lane publisher.
# The accepted scientific writer rechecks its deeper Payne Zero/static inputs.
ACCEPTED_SOURCE_IDENTITIES: tuple[tuple[str, str], ...] = (
    (CONTRACT_IDENTITY, CONTRACT_SHA256),
    (CONTRACT_AUDIT_IDENTITY, CONTRACT_AUDIT_SHA256),
    (
        CANDIDATE_BYTE_ACCEPTANCE_IDENTITY,
        CANDIDATE_BYTE_ACCEPTANCE_SHA256,
    ),
    (
        SYNTHESIS_CANDIDATE_BYTE_ACCEPTANCE_IDENTITY,
        SYNTHESIS_CANDIDATE_BYTE_ACCEPTANCE_SHA256,
    ),
    (
        "design/chapter06_atmosphere_fixture_oracle_plan.md",
        "cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97",
    ),
    (
        "design/chapter06_exact_source_contract.md",
        "ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1",
    ),
    (
        "scripts/chapter06_atmosphere_line_converter.py",
        "4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb",
    ),
    (
        "tests/test_chapter06_atmosphere_line_converter.py",
        "254d796b7ab761ca806c372d0bcdd935067ff1a89b2acfebcfa3007fe3f549dc",
    ),
    (
        "design/chapter06_atmosphere_converter_candidate.md",
        "acae959e9b01f986a553e9806fa7f60c6bb770ce3f356fab11c2d7509b63d03a",
    ),
    (
        "design/chapter06_atmosphere_converter_independent_audit.md",
        "60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75",
    ),
    (
        "scripts/chapter06_atmosphere_fixture_worker.py",
        "21f373f32df75ef3d172ac35e871b37d8e1daaf0efa5f3f5cd1fd38825e10531",
    ),
    (
        "tests/test_chapter06_atmosphere_fixture_worker.py",
        "611639a05178209d304e7c64a9756162c61a14fe20f2dd312ae49b555340cb42",
    ),
    (
        "design/chapter06_atmosphere_fixture_worker_candidate.md",
        "da53e4846ee91f814c437994fff604ae91ed94a4da4db7856f8f47bb61cf72dc",
    ),
    (
        "design/chapter06_atmosphere_fixture_worker_independent_audit.md",
        "336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e",
    ),
    (WRITER_IDENTITY, WRITER_SHA256),
    (WRITER_TEST_IDENTITY, WRITER_TEST_SHA256),
    (WRITER_CANDIDATE_IDENTITY, WRITER_CANDIDATE_SHA256),
    (WRITER_ACCEPTANCE_IDENTITY, WRITER_ACCEPTANCE_SHA256),
    (
        "design/chapter06_synthesis_fixture_oracle_plan.md",
        "413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856",
    ),
    (
        "design/chapter06_synthesis_plan_rebind_candidate.md",
        "dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93",
    ),
    (
        "design/chapter06_synthesis_plan_rebind_independent_audit.md",
        "9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7",
    ),
    (
        "scripts/chapter06_synthesis_oracle_worker.py",
        "36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68",
    ),
    (
        "tests/test_chapter06_synthesis_oracle_worker.py",
        "1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189",
    ),
    (
        "design/chapter06_synthesis_worker_independent_audit.md",
        "a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334",
    ),
    (
        "scripts/chapter06_synthesis_compact_assembler.py",
        "583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8",
    ),
    (
        "tests/test_chapter06_synthesis_compact_assembler.py",
        "25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153",
    ),
    (
        "design/chapter06_synthesis_compact_rebind_candidate.md",
        "54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c",
    ),
    (
        "design/chapter06_synthesis_compact_rebind_independent_audit.md",
        "739854db2b5c4c0c0fe5e9db71d8a52958ce401ded7e7a80a8ab90e15172ddcb",
    ),
    (
        "scripts/chapter06_synthesis_compact_writer.py",
        "57aa7147afee4a7366cb2a075715d3607fa20507c23c07ec978b0698368ae47b",
    ),
    (
        "tests/test_chapter06_synthesis_compact_writer.py",
        "7c41a74f9d2e38a23d988c990af4040ac262a8066cb3cd9feae4e29f0bdc0a4e",
    ),
    (
        "design/chapter06_synthesis_writer_rebind_candidate.md",
        "6ab1f346a409b0302550a0923c35b71a84d6b2899f2c356070c8d76aa8145e5a",
    ),
    (
        "design/chapter06_synthesis_writer_rebind_independent_audit.md",
        "467fdc810f14302dba80f0dd18ba34239dfedb7579b48280899f6f9b6e3b3653",
    ),
)


class AtmosphereFixturePublicationError(RuntimeError):
    """Base class for a rejected verification or publication transition."""


class IdentityError(AtmosphereFixturePublicationError):
    """An exact file, path, type, or hash identity changed."""


class CandidateError(AtmosphereFixturePublicationError):
    """The reconstructed in-memory candidate failed validation."""


class AuthorizationError(AtmosphereFixturePublicationError):
    """Detached authorization or its independent review is unavailable."""


class ManifestError(AtmosphereFixturePublicationError):
    """The manifest bytes, schema, role join, or intended delta changed."""


class QuarantineError(AtmosphereFixturePublicationError):
    """A crash-left hidden object requires separate reviewed cleanup."""


class PublicationIOError(AtmosphereFixturePublicationError):
    """A durable no-replace publication operation did not complete."""


@dataclass(frozen=True)
class RuntimePaths:
    """Fixed production access paths.

    Tests may monkeypatch the private ``_runtime_paths`` function to exercise
    the transition in an isolated tree.  No public API accepts a root or
    destination.
    """

    repository_root: Path
    data_root: Path
    manifest: Path
    destination: Path
    authorization: Path
    record_review: Path
    publisher_acceptance: Path


@dataclass(frozen=True)
class RegularFile:
    identity: str
    payload: bytes
    device: int
    inode: int
    owner: int
    group: int
    mode: int
    links: int

    @property
    def sha256(self) -> str:
        return _sha256(self.payload)

    @property
    def size(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class TreeSnapshot:
    directories: tuple[dict[str, Any], ...]
    files: tuple[dict[str, Any], ...]
    manifest_backed: tuple[dict[str, Any], ...]
    nonmanifest_support: tuple[dict[str, Any], ...]
    manifest_sha256: str
    ordered_entry_path_sha256: str
    aggregate_sha256: str


@dataclass(frozen=True)
class SourceSnapshot:
    records: tuple[dict[str, Any], ...]
    aggregate_sha256: str


@dataclass(frozen=True)
class Candidate:
    archive_bytes: bytes
    arrays: dict[str, Any]
    stable_writer_summary: dict[str, Any]
    topology_summary: dict[str, Any]


@dataclass(frozen=True)
class Authority:
    authorization: dict[str, Any]
    authorization_file: RegularFile
    review: dict[str, Any]
    review_file: RegularFile


@dataclass(frozen=True)
class Stage:
    path: Path
    device: int
    inode: int


def _runtime_paths() -> RuntimePaths:
    root = CANONICAL_REPOSITORY_ROOT
    return RuntimePaths(
        repository_root=root,
        data_root=root / "data",
        manifest=root / MANIFEST_IDENTITY,
        destination=root / DESTINATION_IDENTITY,
        authorization=root / AUTHORIZATION_IDENTITY,
        record_review=root / RECORD_REVIEW_IDENTITY,
        publisher_acceptance=root / PUBLISHER_ACCEPTANCE_IDENTITY,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_plain_int(
    value: Any,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    if not _is_plain_int(value):
        raise IdentityError(f"{label} must be a plain JSON integer")
    if minimum is not None and value < minimum:
        raise IdentityError(f"{label} is below {minimum}")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IdentityError(f"{label} must be lowercase 64-hex SHA-256")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    actual = tuple(value)
    if actual != tuple(expected):
        raise IdentityError(
            f"{label} key order is {actual!r}; expected {tuple(expected)!r}"
        )


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IdentityError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise IdentityError(f"nonfinite JSON value is forbidden: {token}")


def _parse_strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise IdentityError(f"{label} is not UTF-8") from error
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise IdentityError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise IdentityError(f"{label} top level must be an object")
    return parsed


def _manifest_encode(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=False,
                separators=(",", ": "),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ManifestError(f"manifest cannot be encoded exactly: {error}") from error


def _template_encode(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AuthorizationError(
            f"manifest template cannot be encoded exactly: {error}"
        ) from error


def _validate_repository_identity(identity: Any, *, label: str) -> str:
    if not isinstance(identity, str) or not identity:
        raise IdentityError(f"{label} must be a nonempty string")
    if not identity.isascii() or unicodedata.normalize("NFC", identity) != identity:
        raise IdentityError(f"{label} must be an NFC ASCII identity")
    if "\\" in identity or "%" in identity or identity.startswith("/"):
        raise IdentityError(f"{label} is not an allowlisted POSIX identity")
    pure = PurePosixPath(identity)
    parts = pure.parts
    if (
        not parts
        or any(part in {"", ".", ".."} for part in parts)
        or str(pure) != identity
    ):
        raise IdentityError(f"{label} has an unsafe path component")
    return identity


def _canonical_repository_guard(paths: RuntimePaths) -> None:
    root = paths.repository_root
    if not root.is_absolute():
        raise IdentityError("repository root is not absolute")
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise IdentityError("canonical repository root is unavailable") from error
    if (
        resolved != root
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise IdentityError("canonical repository root is relocated or symlinked")

    production = paths.repository_root == CANONICAL_REPOSITORY_ROOT
    if production:
        publisher_path = root / PUBLISHER_IDENTITY
        try:
            if Path(__file__).resolve(strict=True) != publisher_path:
                raise IdentityError("publisher is executing from a relocated path")
        except OSError as error:
            raise IdentityError("publisher path is unavailable") from error
        expected_pairs = {
            "data root": (paths.data_root, root / "data"),
            "manifest": (paths.manifest, root / MANIFEST_IDENTITY),
            "destination": (paths.destination, root / DESTINATION_IDENTITY),
            "authorization": (
                paths.authorization,
                root / AUTHORIZATION_IDENTITY,
            ),
            "record review": (
                paths.record_review,
                root / RECORD_REVIEW_IDENTITY,
            ),
            "publisher acceptance": (
                paths.publisher_acceptance,
                root / PUBLISHER_ACCEPTANCE_IDENTITY,
            ),
        }
        for label, (actual, expected) in expected_pairs.items():
            if actual != expected:
                raise IdentityError(f"{label} is not the fixed production path")


def _read_repository_regular(
    paths: RuntimePaths,
    identity: str,
    *,
    label: str,
    require_single_link: bool = True,
) -> RegularFile:
    identity = _validate_repository_identity(identity, label=f"{label} identity")
    parts = PurePosixPath(identity).parts
    flags_directory = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags_directory |= getattr(os, "O_NOFOLLOW", 0)
    flags_file = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        root_fd = os.open(paths.repository_root, flags_directory)
        descriptors.append(root_fd)
        current_fd = root_fd
        for component in parts[:-1]:
            next_fd = os.open(component, flags_directory, dir_fd=current_fd)
            descriptors.append(next_fd)
            current_fd = next_fd
        file_fd = os.open(parts[-1], flags_file, dir_fd=current_fd)
        descriptors.append(file_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise IdentityError(f"{label} is not a regular file")
        if require_single_link and before.st_nlink != 1:
            raise IdentityError(f"{label} must have exactly one hard link")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise IdentityError(f"{label} changed while being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise IdentityError(f"{label} read was short")
        return RegularFile(
            identity=identity,
            payload=payload,
            device=before.st_dev,
            inode=before.st_ino,
            owner=before.st_uid,
            group=before.st_gid,
            mode=stat.S_IMODE(before.st_mode),
            links=before.st_nlink,
        )
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        if isinstance(error, IdentityError):
            raise
        raise IdentityError(
            f"{label} is unavailable without following links"
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _optional_repository_regular(
    paths: RuntimePaths,
    identity: str,
    *,
    label: str,
) -> RegularFile | None:
    try:
        return _read_repository_regular(paths, identity, label=label)
    except IdentityError as error:
        candidate = paths.repository_root / identity
        try:
            candidate.lstat()
        except FileNotFoundError:
            return None
        raise error


def _file_record(file: RegularFile) -> dict[str, Any]:
    return {
        "path": file.identity,
        "type": "regular",
        "nonsymlink": True,
        "device": file.device,
        "inode": file.inode,
        "owner": file.owner,
        "group": file.group,
        "mode": file.mode,
        "links": file.links,
        "bytes": file.size,
        "sha256": file.sha256,
    }


def _verify_accepted_source_identities(paths: RuntimePaths) -> SourceSnapshot:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for identity, expected_hash in ACCEPTED_SOURCE_IDENTITIES:
        if identity in seen:
            raise IdentityError(f"duplicate accepted source identity: {identity}")
        seen.add(identity)
        source = _read_repository_regular(
            paths,
            identity,
            label=f"accepted source {identity}",
        )
        if source.sha256 != expected_hash:
            raise IdentityError(
                f"accepted source hash changed: {identity} "
                f"{source.sha256} != {expected_hash}"
            )
        records.append(_file_record(source))

    # The candidate publisher and tests are not pre-accepted identities, but
    # their exact current bytes belong to the source snapshot later bound by A.
    for identity in (PUBLISHER_IDENTITY, PUBLISHER_TEST_IDENTITY):
        source = _read_repository_regular(
            paths,
            identity,
            label=f"publisher candidate source {identity}",
        )
        records.append(_file_record(source))
    publisher_acceptance = _optional_repository_regular(
        paths,
        PUBLISHER_ACCEPTANCE_IDENTITY,
        label="publisher implementation acceptance",
    )
    if publisher_acceptance is not None:
        records.append(_file_record(publisher_acceptance))

    # The accepted scientific worker owns the complete pinned-source and
    # dynamic-data vocabulary.  Re-run that exact gate here as part of the
    # publisher snapshot instead of copying a shallower, divergent manifest.
    try:
        deep_identities = worker.verify_preimport_identities()
    except (RuntimeError, OSError, ValueError) as error:
        raise IdentityError(
            f"accepted writer deep source gate failed: {error}"
        ) from error
    if deep_identities.get("payne_zero_commit") != PINNED_PAYNE_ZERO_COMMIT:
        raise IdentityError("deep source gate returned the wrong pinned commit")
    deep_record = {
        "scope": "accepted_atmosphere_writer_deep_source_gate",
        "payne_zero_root": str(PINNED_PAYNE_ZERO_ROOT),
        "payne_zero_commit": PINNED_PAYNE_ZERO_COMMIT,
        "paper_root_read_only_boundary": str(PAPER_ROOT),
        "identity_count": len(deep_identities),
        "identities": {key: deep_identities[key] for key in sorted(deep_identities)},
        "dynamic_data_read_set": list(worker.EXPECTED_DYNAMIC_READ_SET),
    }
    records.append(deep_record)

    encoded = json.dumps(
        records,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return SourceSnapshot(tuple(records), _sha256(encoded))


def _directory_record(identity: str, metadata: os.stat_result) -> dict[str, Any]:
    return {
        "path": identity,
        "type": "directory",
        "nonsymlink": True,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "owner": metadata.st_uid,
        "group": metadata.st_gid,
        "mode": stat.S_IMODE(metadata.st_mode),
    }


def _walk_data_tree(
    paths: RuntimePaths,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    def descend(directory: Path, identity: str) -> None:
        try:
            before = directory.lstat()
        except OSError as error:
            raise IdentityError(f"data directory is unavailable: {identity}") from error
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise IdentityError(
                f"data component is not a nonsymlink directory: {identity}"
            )
        directories.append(_directory_record(identity, before))
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise IdentityError(
                f"cannot inventory data directory: {identity}"
            ) from error
        for child in children:
            child_identity = f"{identity}/{child.name}"
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as error:
                raise IdentityError(
                    f"cannot inspect data descendant: {child_identity}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise IdentityError(f"symlink is forbidden in data: {child_identity}")
            if stat.S_ISDIR(metadata.st_mode):
                descend(Path(child.path), child_identity)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise IdentityError(
                    f"special node is forbidden in data: {child_identity}"
                )
            regular = _read_repository_regular(
                paths,
                child_identity,
                label=f"data file {child_identity}",
            )
            files.append(_file_record(regular))
        after = directory.lstat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise IdentityError(f"data directory changed during inventory: {identity}")

    descend(paths.data_root, "data")
    return tuple(directories), tuple(files)


def _parse_manifest_file(
    manifest_file: RegularFile,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    manifest = _parse_strict_json(
        manifest_file.payload,
        label="data manifest",
    )
    _require_exact_keys(
        manifest,
        MANIFEST_TOP_LEVEL_KEYS,
        label="data manifest top level",
    )
    if manifest.get("schema_version") != 1:
        raise ManifestError("data manifest schema version changed")
    if manifest.get("payne_zero_commit") != PINNED_PAYNE_ZERO_COMMIT:
        raise ManifestError("data manifest Payne Zero commit changed")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ManifestError("data manifest entries must be a list")
    paths: list[str] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"manifest entry {position} is not an object")
        identity = _validate_repository_identity(
            entry.get("path"),
            label=f"manifest entry {position} path",
        )
        if not identity.startswith("data/"):
            raise ManifestError(f"manifest entry leaves data/: {identity}")
        if identity in paths:
            raise ManifestError(f"duplicate manifest path: {identity}")
        role = entry.get("role")
        if role not in {"static", "subset", "fixture", "golden"}:
            raise ManifestError(f"manifest entry has invalid role: {identity}")
        _require_sha256(entry.get("sha256"), label=f"{identity} SHA-256")
        _require_plain_int(
            entry.get("bytes"),
            label=f"{identity} byte size",
            minimum=0,
        )
        paths.append(identity)
    if _manifest_encode(manifest) != manifest_file.payload:
        raise ManifestError(
            "data manifest does not round-trip under the accepted unsorted encoder"
        )
    return manifest, tuple(paths)


def _entry_digest(entry: Mapping[str, Any]) -> str:
    return _sha256(
        json.dumps(
            entry,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _ordered_path_digest(paths: Sequence[str]) -> str:
    return _sha256(
        json.dumps(
            list(paths),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _prepublication_manifest_identity(
    paths: RuntimePaths,
) -> tuple[str, str]:
    """Return M's hash/path digest from either M or its exact append-shaped N."""

    manifest_file = _read_repository_regular(
        paths,
        MANIFEST_IDENTITY,
        label="manifest used to derive prepublication identity",
    )
    manifest, ordered_paths = _parse_manifest_file(manifest_file)
    if DESTINATION_IDENTITY not in ordered_paths:
        return manifest_file.sha256, _ordered_path_digest(ordered_paths)
    if ordered_paths.count(DESTINATION_IDENTITY) != 1:
        raise ManifestError("destination occurs more than once in manifest")
    if ordered_paths[-1] != DESTINATION_IDENTITY:
        raise ManifestError("destination registration is not the final append")
    reconstructed = deepcopy(manifest)
    reconstructed["entries"].pop()
    pre_paths = ordered_paths[:-1]
    return _sha256(_manifest_encode(reconstructed)), _ordered_path_digest(pre_paths)


def _snapshot_data(
    paths: RuntimePaths,
    *,
    allow_exact_unregistered_target: bool = False,
    owned_manifest_temporary: Stage | None = None,
    owned_manifest_bytes: bytes | None = None,
    owned_manifest_mode: int | None = None,
) -> TreeSnapshot:
    if (owned_manifest_temporary is None) != (owned_manifest_bytes is None) or (
        owned_manifest_temporary is None
    ) != (owned_manifest_mode is None):
        raise IdentityError("owned manifest-temporary identity is incomplete")
    manifest_file = _read_repository_regular(
        paths,
        MANIFEST_IDENTITY,
        label="data manifest",
    )
    manifest, ordered_paths = _parse_manifest_file(manifest_file)
    directories, files = _walk_data_tree(paths)
    directory_paths = tuple(record["path"] for record in directories)
    if directory_paths != tuple(sorted(EXPECTED_PREPUBLICATION_DIRECTORY_IDENTITIES)):
        raise IdentityError(
            f"closed data directory inventory changed: {directory_paths!r}"
        )

    file_by_path = {record["path"]: record for record in files}
    if len(file_by_path) != len(files):
        raise IdentityError("data inventory contains duplicate file identities")
    manifest_backed: list[dict[str, Any]] = []
    for position, entry in enumerate(manifest["entries"]):
        identity = entry["path"]
        record = file_by_path.get(identity)
        if record is None:
            raise ManifestError(f"registered data file is absent: {identity}")
        if record["bytes"] != entry["bytes"] or record["sha256"] != entry["sha256"]:
            raise ManifestError(
                f"registered data file disagrees with manifest: {identity}"
            )
        manifest_backed.append(
            {
                "path": identity,
                "role": entry["role"],
                "entry_position": position,
                "entry_digest": _entry_digest(entry),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            }
        )

    support: list[dict[str, Any]] = []
    manifest_paths = set(ordered_paths)
    for record in files:
        identity = record["path"]
        if identity in manifest_paths or identity == MANIFEST_IDENTITY:
            continue
        if identity == README_IDENTITY and record["sha256"] == README_SHA256:
            support.append(
                {
                    "path": identity,
                    "classification": "closed_nonmanifest_support",
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
            )
            continue
        if (
            allow_exact_unregistered_target
            and identity == DESTINATION_IDENTITY
            and record["bytes"] == EXPECTED_ARCHIVE_BYTES
            and record["sha256"] == EXPECTED_ARCHIVE_SHA256
            and record["links"] == 1
        ):
            support.append(
                {
                    "path": identity,
                    "classification": "exact_unregistered_recovery_target",
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
            )
            continue
        if (
            owned_manifest_temporary is not None
            and owned_manifest_bytes is not None
            and owned_manifest_mode is not None
            and identity == f"data/{owned_manifest_temporary.path.name}"
            and (record["device"], record["inode"])
            == (
                owned_manifest_temporary.device,
                owned_manifest_temporary.inode,
            )
            and record["mode"] == owned_manifest_mode
            and record["links"] == 1
            and record["bytes"] == len(owned_manifest_bytes)
            and record["sha256"] == _sha256(owned_manifest_bytes)
        ):
            support.append(
                {
                    "path": identity,
                    "classification": "invocation_owned_manifest_temporary",
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
            )
            continue
        if Path(identity).name.startswith(
            (ARTIFACT_STAGE_PREFIX, MANIFEST_TEMPORARY_PREFIX)
        ):
            raise QuarantineError(
                f"crash-left hidden object requires reviewed cleanup: {identity}"
            )
        raise IdentityError(f"unexpected nonmanifest data file: {identity}")

    if tuple(
        item["path"]
        for item in support
        if item["classification"] == "closed_nonmanifest_support"
    ) != (README_IDENTITY,):
        raise IdentityError("closed data support inventory changed")

    exact_unregistered = any(
        item["classification"] == "exact_unregistered_recovery_target"
        for item in support
    )
    invocation_temporary_paths = {
        item["path"]
        for item in support
        if item["classification"] == "invocation_owned_manifest_temporary"
    }
    aggregate_files = (
        tuple(record for record in files if record["path"] != DESTINATION_IDENTITY)
        if exact_unregistered
        else files
    )
    if invocation_temporary_paths:
        aggregate_files = tuple(
            record
            for record in aggregate_files
            if record["path"] not in invocation_temporary_paths
        )
    aggregate_support = (
        tuple(
            item
            for item in support
            if item["classification"] != "exact_unregistered_recovery_target"
        )
        if exact_unregistered
        else tuple(support)
    )
    if invocation_temporary_paths:
        aggregate_support = tuple(
            item
            for item in aggregate_support
            if item["classification"] != "invocation_owned_manifest_temporary"
        )
    snapshot_object = {
        "directories": directories,
        "regular_files": aggregate_files,
        "manifest_backed_files": manifest_backed,
        "nonmanifest_support": aggregate_support,
        "manifest_sha256": manifest_file.sha256,
        "ordered_entry_path_sha256": _ordered_path_digest(ordered_paths),
    }
    aggregate = _sha256(
        json.dumps(
            snapshot_object,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return TreeSnapshot(
        directories,
        files,
        tuple(manifest_backed),
        tuple(support),
        manifest_file.sha256,
        snapshot_object["ordered_entry_path_sha256"],
        aggregate,
    )


def _snapshot_is_registered(snapshot: TreeSnapshot) -> bool:
    return any(
        record["path"] == DESTINATION_IDENTITY for record in snapshot.manifest_backed
    )


def _stable_writer_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    stable = dict(summary)
    stable.pop("capture_process_evidence", None)
    stable.pop("capture_process_evidence_sha256", None)
    return stable


def _validate_writer_summary(
    summary: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    required = {
        "writer_sha256": WRITER_SHA256,
        "fixture_a_member_count": EXPECTED_MEMBER_COUNT,
        "fixture_b_member_count": EXPECTED_MEMBER_COUNT,
        "fixture_a_b_bitwise_equal": True,
        "fixture_a_b_mapping_digest": EXPECTED_MAPPING_DIGEST,
        "evidence_a_member_count": EXPECTED_EVIDENCE_MEMBER_COUNT,
        "evidence_b_member_count": EXPECTED_EVIDENCE_MEMBER_COUNT,
        "evidence_a_b_bitwise_equal": True,
        "evidence_a_b_mapping_digest": EXPECTED_EVIDENCE_MAPPING_DIGEST,
        "capture_a_transport_sha256": EXPECTED_CAPTURE_TRANSPORT_SHA256,
        "capture_b_transport_sha256": EXPECTED_CAPTURE_TRANSPORT_SHA256,
        "capture_a_b_transport_byte_equal": True,
        "capture_transport_bytes": EXPECTED_CAPTURE_TRANSPORT_BYTES,
        "fixture_schema_digest": EXPECTED_FIXTURE_SCHEMA_DIGEST,
        "payload_fingerprint": EXPECTED_PAYLOAD_FINGERPRINT,
        "full_capture_schema_digest": EXPECTED_FULL_CAPTURE_SCHEMA_DIGEST,
        "full_capture_fingerprint": EXPECTED_FULL_CAPTURE_FINGERPRINT,
        "archive_a_sha256": EXPECTED_ARCHIVE_SHA256,
        "archive_b_sha256": EXPECTED_ARCHIVE_SHA256,
        "archive_a_b_byte_equal": True,
        "archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "archive_array_bytes": EXPECTED_ARRAY_BYTES,
        "archive_member_count": EXPECTED_MEMBER_COUNT,
        "archive_member_order": "lexical",
        "archive_compression": "ZIP_STORED",
        "npy_version": [2, 0],
        "fixed_zip_date_time": [1980, 1, 1, 0, 0, 0],
        "disposable_cache_directories_created": True,
        "publication_authorized": False,
        "fixture_publication_performed": False,
        "golden_publication_performed": False,
        "manifest_mutation_performed": False,
        "artifact_file_write_performed": False,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise CandidateError(
                f"accepted writer summary changed at {key}: "
                f"{summary.get(key)!r} != {expected!r}"
            )
    evidence = summary.get("capture_process_evidence")
    if not isinstance(evidence, dict):
        raise CandidateError("writer process evidence is absent")
    topology_required = {
        "capture_origins_distinct": True,
        "capture_child_processes_distinct": True,
        "capture_cache_roots_distinct": True,
        "capture_cache_roots_external": True,
        "capture_cache_roots_nonsymlink": True,
        "capture_cache_roots_empty_before": True,
        "capture_a_cache_entry_count_after": 37,
        "capture_b_cache_entry_count_after": 37,
        "capture_cache_roots_disposed": True,
    }
    for key, expected in topology_required.items():
        if evidence.get(key) != expected:
            raise CandidateError(
                f"accepted writer topology changed at {key}: "
                f"{evidence.get(key)!r} != {expected!r}"
            )
    topology = {
        "fresh_child_processes": True,
        "distinct_origins": True,
        "distinct_external_cache_roots": True,
        "cache_roots_empty_before": True,
        "cache_entries_after_each_child": 37,
        "cache_package_directories_after_each_child": 1,
        "cache_nbi_files_after_each_child": 18,
        "cache_nbc_files_after_each_child": 18,
        "cache_symlinks_after_each_child": 0,
        "cache_other_files_after_each_child": 0,
        "cache_roots_disposed_before_return": True,
    }
    return _stable_writer_summary(summary), topology


_TOP_LEVEL_CHILD_CODE = r"""
import json
import struct
import sys
from scripts.chapter06_atmosphere_fixture_writer import (
    build_deterministic_atmosphere_fixture_archive,
)

result = build_deterministic_atmosphere_fixture_archive()
header = json.dumps(
    result.summary,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=False,
    separators=(",", ":"),
).encode("utf-8")
sys.stdout.buffer.write(b"ATM6PUB1")
sys.stdout.buffer.write(struct.pack(">Q", len(header)))
sys.stdout.buffer.write(header)
sys.stdout.buffer.write(result.archive_bytes)
sys.stdout.buffer.flush()
"""


def _invoke_accepted_writer_top_level(
    paths: RuntimePaths,
) -> tuple[
    bytes,
    dict[str, Any],
]:
    environment = dict(os.environ)
    environment.update(worker.CAPTURE_ENVIRONMENT)
    environment.pop("PAYNE_ZERO_DATA_ROOT", None)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(paths.repository_root)
        if not existing_pythonpath
        else str(paths.repository_root) + os.pathsep + existing_pythonpath
    )
    completed = subprocess.run(
        [sys.executable, "-c", _TOP_LEVEL_CHILD_CODE],
        cwd=paths.repository_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise CandidateError(
            f"accepted top-level writer failed with {completed.returncode}: {detail}"
        )
    output = completed.stdout
    if len(output) < 16 or output[:8] != b"ATM6PUB1":
        raise CandidateError("top-level writer returned an invalid binary frame")
    header_size = int.from_bytes(output[8:16], "big")
    if header_size <= 0 or header_size > 1_000_000:
        raise CandidateError("top-level writer header size is invalid")
    boundary = 16 + header_size
    if boundary > len(output):
        raise CandidateError("top-level writer frame is truncated")
    summary = _parse_strict_json(
        output[16:boundary],
        label="top-level writer summary",
    )
    archive = bytes(output[boundary:])
    if len(archive) != EXPECTED_ARCHIVE_BYTES:
        raise CandidateError("top-level writer archive byte size changed")
    return archive, summary


def _array_sha256(array: Any) -> str:
    contiguous = worker.np.ascontiguousarray(array)
    return _sha256(contiguous.tobytes(order="C"))


_MEMBER_CONVENTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "actual_population_slot_indices": (
        "zero-based packed population slot",
        ("population_projection",),
    ),
    "actual_population_slot_values": (
        "cm^-3 per partition function",
        ("depth", "population_projection"),
    ),
    "continuum_line_selection_threshold": (
        "cm^2 g^-1",
        ("depth", "wavelength_bin"),
    ),
    "effective_temperature": ("K", ()),
    "electron_density": ("cm^-3", ("depth",)),
    "fractional_doppler_widths_at_line_slot": (
        "dimensionless Delta-lambda/lambda",
        ("depth",),
    ),
    "hc_over_kt": ("cm", ("depth",)),
    "line_population_slot_zero_based": (
        "zero-based packed population slot",
        (),
    ),
    "log_strength_index": ("packed line-table integer", ("selected_line",)),
    "lower_excitation_index": (
        "packed line-table integer",
        ("selected_line",),
    ),
    "opacity_wavelength_grid_nm": ("nm", ("opacity_wavelength",)),
    "packed_species_slot": (
        "zero-based packed species slot",
        ("selected_line",),
    ),
    "packed_wavelength_index": (
        "packed logarithmic wavelength integer",
        ("selected_line",),
    ),
    (
        "partition_normalized_population_over_mass_density_and_"
        "fractional_doppler_width_at_line_slot"
    ): ("g^-1", ("depth",)),
    "radiative_damping_index": (
        "packed line-table integer",
        ("selected_line",),
    ),
    "stark_damping_index": ("packed line-table integer", ("selected_line",)),
    "temperature": ("K", ("depth",)),
    "van_der_waals_damping_index": (
        "packed line-table integer",
        ("selected_line",),
    ),
    "wavelength_bin_edges": (
        "zero-based opacity-wavelength boundary index",
        ("wavelength_bin_edge",),
    ),
}


def _validate_untrusted_archive(archive_bytes: bytes) -> dict[str, Any]:
    if not isinstance(archive_bytes, bytes):
        raise CandidateError("candidate archive must be immutable bytes")
    if len(archive_bytes) != EXPECTED_ARCHIVE_BYTES:
        raise CandidateError("candidate archive byte size changed")
    if _sha256(archive_bytes) != EXPECTED_ARCHIVE_SHA256:
        raise CandidateError("candidate archive whole-byte identity changed")
    try:
        arrays = writer._deserialize_canonical_mapping(
            archive_bytes,
            size_ceiling=writer.FIXTURE_ARCHIVE_SIZE_CEILING_BYTES,
        )
        worker._validate_fixture_mapping(arrays)
    except (
        RuntimeError,
        TypeError,
        ValueError,
        writer.AtmosphereFixtureWriterError,
    ) as error:
        raise CandidateError(f"candidate archive validation failed: {error}") from error
    expected_names = tuple(sorted(worker.FIXTURE_SCHEMA))
    if tuple(arrays) != expected_names or len(arrays) != EXPECTED_MEMBER_COUNT:
        raise CandidateError("candidate archive member allowlist changed")
    if worker.mapping_schema_digest(arrays) != EXPECTED_FIXTURE_SCHEMA_DIGEST:
        raise CandidateError("candidate scientific schema digest changed")
    # The accepted physical payload fingerprint also covers transient
    # ``payload__`` evidence intentionally excluded from the 19-member
    # archive.  It is revalidated by each accepted top-level writer summary;
    # inventing an archive-only substitute would change its meaning.
    if (
        writer._serialize_mapping(
            arrays,
            size_ceiling=writer.FIXTURE_ARCHIVE_SIZE_CEILING_BYTES,
        )
        != archive_bytes
    ):
        raise CandidateError("candidate canonical re-encoding changed")
    if sum(array.nbytes for array in arrays.values()) != EXPECTED_ARRAY_BYTES:
        raise CandidateError("candidate scientific array-byte total changed")
    if set(_MEMBER_CONVENTIONS) != set(arrays):
        raise CandidateError("manifest member convention coverage changed")

    records: dict[str, Any] = {}
    for name in expected_names:
        array = arrays[name]
        if array.dtype.kind in {"f", "c"} and not worker.np.all(
            worker.np.isfinite(array)
        ):
            raise CandidateError(f"candidate contains nonfinite values: {name}")
        unit, axes = _MEMBER_CONVENTIONS[name]
        record = {
            "shape": list(array.shape),
            "dtype": array.dtype.name,
            "unit": unit,
            "sha256": _array_sha256(array),
            "axes": list(axes),
            "ownership": "atmosphere fixture owns this input member",
        }
        _require_exact_keys(record, ARRAY_RECORD_KEYS, label=f"{name} array record")
        records[name] = record
    return records


def _build_candidate_twice(paths: RuntimePaths) -> Candidate:
    archive_a, summary_a = _invoke_accepted_writer_top_level(paths)
    archive_b, summary_b = _invoke_accepted_writer_top_level(paths)
    stable_a, topology_a = _validate_writer_summary(summary_a)
    stable_b, topology_b = _validate_writer_summary(summary_b)
    if archive_a != archive_b:
        raise CandidateError("unrelated top-level writer archives disagree")
    if stable_a != stable_b:
        raise CandidateError("unrelated top-level stable writer summaries disagree")
    if topology_a != topology_b:
        raise CandidateError("unrelated top-level topology decisions disagree")
    arrays = _validate_untrusted_archive(archive_a)
    return Candidate(archive_a, arrays, stable_a, topology_a)


def _identity_object(path: str, sha256: str) -> dict[str, Any]:
    value = {"path": path, "sha256": sha256}
    _require_exact_keys(value, IDENTITY_KEYS, label=f"{path} identity object")
    return value


def _current_candidate_identities(paths: RuntimePaths) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label, identity in (
        ("publisher", PUBLISHER_IDENTITY),
        ("publisher_tests", PUBLISHER_TEST_IDENTITY),
    ):
        file = _read_repository_regular(paths, identity, label=label)
        result[label] = _identity_object(identity, file.sha256)
    return result


def _build_manifest_template(
    candidate: Candidate,
    *,
    publisher_sha256: str,
    publisher_test_sha256: str,
    publisher_acceptance_sha256: str,
) -> dict[str, Any]:
    template = {
        "path": DESTINATION_IDENTITY,
        "role": MANIFEST_ROLE,
        "source": str(PINNED_PAYNE_ZERO_ROOT),
        "source_commit": PINNED_PAYNE_ZERO_COMMIT,
        "scope": (
            "Input-only 80-depth fixture for building the selected-line "
            "atmosphere kernel from scratch; contains no line-opacity output."
        ),
        "builder": PUBLISHER_IDENTITY,
        "publisher_sha256": publisher_sha256,
        "publisher_test_sha256": publisher_test_sha256,
        "writer_sha256": WRITER_SHA256,
        "writer_test_sha256": WRITER_TEST_SHA256,
        "writer_candidate_sha256": WRITER_CANDIDATE_SHA256,
        "writer_acceptance_sha256": WRITER_ACCEPTANCE_SHA256,
        "publisher_contract_sha256": CONTRACT_SHA256,
        "publisher_contract_audit_sha256": CONTRACT_AUDIT_SHA256,
        "candidate_byte_acceptance_sha256": CANDIDATE_BYTE_ACCEPTANCE_SHA256,
        "publisher_acceptance_sha256": publisher_acceptance_sha256,
        "publication_acceptance_path": AUTHORIZATION_IDENTITY,
        "publication_acceptance_sha256": AUTHORIZATION_PLACEHOLDER,
        "publication_record_review_path": RECORD_REVIEW_IDENTITY,
        "publication_record_review_sha256": RECORD_REVIEW_PLACEHOLDER,
        "archive_kind": ARCHIVE_KIND,
        "fixture_capture_schema_version": EXTERNAL_CAPTURE_SCHEMA_VERSION,
        "archive_contains_embedded_schema_version": False,
        "npy_member_format_version": NPY_MEMBER_FORMAT_VERSION,
        "member_count": EXPECTED_MEMBER_COUNT,
        "scientific_fixture_schema_digest": EXPECTED_FIXTURE_SCHEMA_DIGEST,
        "scientific_payload_fingerprint": EXPECTED_PAYLOAD_FINGERPRINT,
        "capture_policy": (
            "two unrelated top-level builds; each owns two fresh children and "
            "two distinct external caches with 1 directory + 18 .nbi + 18 .nbc"
        ),
        "format": "npz",
        "arrays": deepcopy(candidate.arrays),
        "sha256": EXPECTED_ARCHIVE_SHA256,
        "bytes": EXPECTED_ARCHIVE_BYTES,
    }
    if tuple(template["arrays"]) != tuple(sorted(template["arrays"])):
        raise ManifestError("manifest template arrays are not lexical")
    for name, record in template["arrays"].items():
        _require_exact_keys(
            record,
            ARRAY_RECORD_KEYS,
            label=f"manifest template array {name}",
        )
    return template


def _realize_template(
    template: Mapping[str, Any],
    *,
    authorization_sha256: str,
    review_sha256: str,
) -> dict[str, Any]:
    _require_sha256(authorization_sha256, label="authorization SHA-256")
    _require_sha256(review_sha256, label="record-review SHA-256")
    before = json.dumps(
        template,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=False,
        separators=(",", ":"),
    )
    if before.count(AUTHORIZATION_PLACEHOLDER) != 1:
        raise AuthorizationError(
            "authorization placeholder must occur exactly once in template"
        )
    if before.count(RECORD_REVIEW_PLACEHOLDER) != 1:
        raise AuthorizationError(
            "record-review placeholder must occur exactly once in template"
        )
    if template.get("publication_acceptance_sha256") != AUTHORIZATION_PLACEHOLDER:
        raise AuthorizationError("authorization placeholder is in the wrong field")
    if template.get("publication_record_review_sha256") != RECORD_REVIEW_PLACEHOLDER:
        raise AuthorizationError("record-review placeholder is in the wrong field")
    realized = deepcopy(template)
    realized["publication_acceptance_sha256"] = authorization_sha256
    realized["publication_record_review_sha256"] = review_sha256
    if tuple(realized) != tuple(template):
        raise AuthorizationError("template realization changed object key order")
    differences = [key for key in template if realized[key] != template[key]]
    if differences != [
        "publication_acceptance_sha256",
        "publication_record_review_sha256",
    ]:
        raise AuthorizationError(
            f"template realization changed fields other than two hashes: {differences}"
        )
    return realized


def _validate_identity_binding(
    value: Any,
    *,
    expected_path: str,
    expected_sha256: str,
    label: str,
) -> None:
    if not isinstance(value, dict):
        raise AuthorizationError(f"{label} must be an identity object")
    _require_exact_keys(value, IDENTITY_KEYS, label=label)
    if value["path"] != expected_path:
        raise AuthorizationError(f"{label} path changed")
    if value["sha256"] != expected_sha256:
        raise AuthorizationError(f"{label} SHA-256 changed")


def _validate_authorization_record(
    record: dict[str, Any],
    *,
    paths: RuntimePaths,
    candidate: Candidate,
    source_snapshot: SourceSnapshot,
    data_snapshot: TreeSnapshot,
    publisher_acceptance_file: RegularFile,
    enforce_data_snapshot: bool = True,
) -> None:
    _require_exact_keys(record, AUTHORIZATION_KEYS, label="authorization")
    scalar_expected = {
        "schema_version": 1,
        "record_kind": "chapter06_atmosphere_fixture_publication_acceptance",
        "lane": "atmosphere_fixture",
        "manifest_role": MANIFEST_ROLE,
        "payne_zero_commit": PINNED_PAYNE_ZERO_COMMIT,
        "source_snapshot_sha256": source_snapshot.aggregate_sha256,
        "destination_parent_policy": (
            "existing data/fixtures directory only; publisher creates no directory"
        ),
        "no_replace_primitive": (
            "same-filesystem hard-link from create-exclusive stage"
        ),
    }
    for key, expected in scalar_expected.items():
        if record[key] != expected:
            raise AuthorizationError(
                f"authorization {key} changed: {record[key]!r} != {expected!r}"
            )
    _require_sha256(
        record["data_snapshot_sha256"],
        label="authorization data snapshot SHA-256",
    )
    if (
        enforce_data_snapshot
        and record["data_snapshot_sha256"] != data_snapshot.aggregate_sha256
    ):
        raise AuthorizationError("authorization data snapshot is stale")

    candidates = _current_candidate_identities(paths)
    identity_expectations = {
        "writer": (WRITER_IDENTITY, WRITER_SHA256),
        "writer_tests": (WRITER_TEST_IDENTITY, WRITER_TEST_SHA256),
        "writer_candidate": (
            WRITER_CANDIDATE_IDENTITY,
            WRITER_CANDIDATE_SHA256,
        ),
        "writer_acceptance": (
            WRITER_ACCEPTANCE_IDENTITY,
            WRITER_ACCEPTANCE_SHA256,
        ),
        "candidate_byte_acceptance": (
            CANDIDATE_BYTE_ACCEPTANCE_IDENTITY,
            CANDIDATE_BYTE_ACCEPTANCE_SHA256,
        ),
        "publisher": (
            PUBLISHER_IDENTITY,
            candidates["publisher"]["sha256"],
        ),
        "publisher_tests": (
            PUBLISHER_TEST_IDENTITY,
            candidates["publisher_tests"]["sha256"],
        ),
        "publisher_contract": (CONTRACT_IDENTITY, CONTRACT_SHA256),
        "publisher_contract_audit": (
            CONTRACT_AUDIT_IDENTITY,
            CONTRACT_AUDIT_SHA256,
        ),
        "publisher_acceptance": (
            PUBLISHER_ACCEPTANCE_IDENTITY,
            publisher_acceptance_file.sha256,
        ),
    }
    for key, (expected_path, expected_hash) in identity_expectations.items():
        _validate_identity_binding(
            record[key],
            expected_path=expected_path,
            expected_sha256=expected_hash,
            label=f"authorization {key}",
        )

    artifact = record["artifact"]
    if not isinstance(artifact, dict):
        raise AuthorizationError("authorization artifact must be an object")
    _require_exact_keys(
        artifact,
        ARTIFACT_AUTHORIZATION_KEYS,
        label="authorization artifact",
    )
    expected_artifact = {
        "path": DESTINATION_IDENTITY,
        "filename": Path(DESTINATION_IDENTITY).name,
        "bytes": len(candidate.archive_bytes),
        "sha256": _sha256(candidate.archive_bytes),
        "member_count": len(candidate.arrays),
        "archive_kind": ARCHIVE_KIND,
        "fixture_capture_schema_version": EXTERNAL_CAPTURE_SCHEMA_VERSION,
        "archive_contains_embedded_schema_version": False,
        "npy_member_format_version": NPY_MEMBER_FORMAT_VERSION,
        "scientific_fixture_schema_digest": EXPECTED_FIXTURE_SCHEMA_DIGEST,
        "scientific_payload_fingerprint": EXPECTED_PAYLOAD_FINGERPRINT,
    }
    if artifact != expected_artifact:
        raise AuthorizationError("authorization artifact binding changed")

    manifest = record["manifest"]
    if not isinstance(manifest, dict):
        raise AuthorizationError("authorization manifest must be an object")
    _require_exact_keys(
        manifest,
        MANIFEST_AUTHORIZATION_KEYS,
        label="authorization manifest",
    )
    pre_manifest_hash, pre_ordered_path_digest = _prepublication_manifest_identity(
        paths
    )
    expected_manifest = {
        "path": MANIFEST_IDENTITY,
        "prepublication_sha256": pre_manifest_hash,
        "schema_version": 1,
        "payne_zero_commit": PINNED_PAYNE_ZERO_COMMIT,
        "ordered_entry_path_sha256": pre_ordered_path_digest,
        "destination_entry_absent": True,
    }
    if manifest != expected_manifest:
        raise AuthorizationError("authorization prepublication manifest changed")

    template = record["manifest_entry_template"]
    if not isinstance(template, dict):
        raise AuthorizationError("authorization manifest template must be an object")
    expected_template = _build_manifest_template(
        candidate,
        publisher_sha256=candidates["publisher"]["sha256"],
        publisher_test_sha256=candidates["publisher_tests"]["sha256"],
        publisher_acceptance_sha256=publisher_acceptance_file.sha256,
    )
    if template != expected_template or tuple(template) != tuple(expected_template):
        raise AuthorizationError("authorization manifest template changed")
    template_digest = _sha256(_template_encode(template))
    if record["manifest_entry_template_sha256"] != template_digest:
        raise AuthorizationError("authorization manifest template digest changed")


def _load_authority(
    paths: RuntimePaths,
    *,
    candidate: Candidate,
    source_snapshot: SourceSnapshot,
    data_snapshot: TreeSnapshot,
    enforce_data_snapshot: bool = True,
) -> Authority:
    # All three objects must exist before any lock or stage can be created.
    publisher_acceptance = _optional_repository_regular(
        paths,
        PUBLISHER_ACCEPTANCE_IDENTITY,
        label="publisher implementation acceptance",
    )
    authorization_file = _optional_repository_regular(
        paths,
        AUTHORIZATION_IDENTITY,
        label="detached publication authorization",
    )
    review_file = _optional_repository_regular(
        paths,
        RECORD_REVIEW_IDENTITY,
        label="authorization-record review",
    )
    absent = [
        label
        for label, value in (
            ("publisher implementation acceptance", publisher_acceptance),
            ("detached publication authorization", authorization_file),
            ("authorization-record review", review_file),
        )
        if value is None
    ]
    if absent:
        raise AuthorizationError(
            "publication is disabled before lock/staging creation; absent: "
            + ", ".join(absent)
        )
    assert publisher_acceptance is not None
    assert authorization_file is not None
    assert review_file is not None

    authorization = _parse_strict_json(
        authorization_file.payload,
        label="detached publication authorization",
    )
    review = _parse_strict_json(
        review_file.payload,
        label="authorization-record review",
    )
    _validate_authorization_record(
        authorization,
        paths=paths,
        candidate=candidate,
        source_snapshot=source_snapshot,
        data_snapshot=data_snapshot,
        publisher_acceptance_file=publisher_acceptance,
        enforce_data_snapshot=enforce_data_snapshot,
    )
    _require_exact_keys(review, RECORD_REVIEW_KEYS, label="record review")
    expected_review = {
        "schema_version": 1,
        "record_kind": ("chapter06_atmosphere_fixture_publication_record_review"),
        "authorization_path": AUTHORIZATION_IDENTITY,
        "authorization_sha256": authorization_file.sha256,
        "candidate_byte_acceptance_path": CANDIDATE_BYTE_ACCEPTANCE_IDENTITY,
        "candidate_byte_acceptance_sha256": CANDIDATE_BYTE_ACCEPTANCE_SHA256,
        "publisher_acceptance_path": PUBLISHER_ACCEPTANCE_IDENTITY,
        "publisher_acceptance_sha256": publisher_acceptance.sha256,
        "manifest_entry_template_sha256": authorization[
            "manifest_entry_template_sha256"
        ],
        "disposition": "ACCEPT",
    }
    if review != expected_review:
        raise AuthorizationError("authorization-record review binding changed")
    return Authority(
        authorization=authorization,
        authorization_file=authorization_file,
        review=review,
        review_file=review_file,
    )


def _preflight_authority_presence(paths: RuntimePaths) -> None:
    """Cheap fail-closed gate used before the expensive two-build step."""

    absent: list[str] = []
    for identity, label in (
        (PUBLISHER_ACCEPTANCE_IDENTITY, "publisher implementation acceptance"),
        (AUTHORIZATION_IDENTITY, "detached publication authorization"),
        (RECORD_REVIEW_IDENTITY, "authorization-record review"),
    ):
        if _optional_repository_regular(paths, identity, label=label) is None:
            absent.append(label)
    if absent:
        raise AuthorizationError(
            "publication is disabled before candidate, lock, parent, or stage "
            "creation; absent: " + ", ".join(absent)
        )


def _candidate_report(
    *,
    paths: RuntimePaths,
    candidate: Candidate,
    source_snapshot: SourceSnapshot,
    data_snapshot: TreeSnapshot,
    authority: Authority | None,
) -> dict[str, Any]:
    identities = _current_candidate_identities(paths)
    destination_state = (
        "registered"
        if any(
            record["path"] == DESTINATION_IDENTITY
            for record in data_snapshot.manifest_backed
        )
        else "exact-unregistered"
        if any(
            record["path"] == DESTINATION_IDENTITY
            for record in data_snapshot.nonmanifest_support
        )
        else "absent"
    )
    report = {
        "report_schema_version": 1,
        "report_kind": "chapter06_atmosphere_fixture_publisher_dry_run",
        "mode": (
            "authorized-verification-only"
            if authority is not None
            else "candidate-verification-only"
        ),
        "repository": str(paths.repository_root),
        "destination": DESTINATION_IDENTITY,
        "manifest_role": MANIFEST_ROLE,
        "publication_authorized": authority is not None,
        "publication_performed": False,
        "lock_created": False,
        "stage_created": False,
        "manifest_mutated": False,
        "top_level_writer_invocations": 2,
        "top_level_archives_byte_identical": True,
        "accepted_writer_topology": candidate.topology_summary,
        "archive": {
            "bytes": len(candidate.archive_bytes),
            "sha256": _sha256(candidate.archive_bytes),
            "member_count": len(candidate.arrays),
            "scientific_array_bytes": EXPECTED_ARRAY_BYTES,
            "archive_kind": ARCHIVE_KIND,
            "fixture_capture_schema_version": EXTERNAL_CAPTURE_SCHEMA_VERSION,
            "archive_contains_embedded_schema_version": False,
            "npy_member_format_version": NPY_MEMBER_FORMAT_VERSION,
            "scientific_fixture_schema_digest": EXPECTED_FIXTURE_SCHEMA_DIGEST,
            "scientific_payload_fingerprint": EXPECTED_PAYLOAD_FINGERPRINT,
            "canonical_decode_validate_reencode": True,
        },
        "trust_identities": {
            "publisher_contract": _identity_object(
                CONTRACT_IDENTITY,
                CONTRACT_SHA256,
            ),
            "publisher_contract_audit": _identity_object(
                CONTRACT_AUDIT_IDENTITY,
                CONTRACT_AUDIT_SHA256,
            ),
            "candidate_byte_acceptance": _identity_object(
                CANDIDATE_BYTE_ACCEPTANCE_IDENTITY,
                CANDIDATE_BYTE_ACCEPTANCE_SHA256,
            ),
            "common_synthesis_candidate_byte_acceptance": _identity_object(
                SYNTHESIS_CANDIDATE_BYTE_ACCEPTANCE_IDENTITY,
                SYNTHESIS_CANDIDATE_BYTE_ACCEPTANCE_SHA256,
            ),
            "writer": _identity_object(WRITER_IDENTITY, WRITER_SHA256),
            "writer_tests": _identity_object(
                WRITER_TEST_IDENTITY,
                WRITER_TEST_SHA256,
            ),
            "writer_candidate": _identity_object(
                WRITER_CANDIDATE_IDENTITY,
                WRITER_CANDIDATE_SHA256,
            ),
            "writer_acceptance": _identity_object(
                WRITER_ACCEPTANCE_IDENTITY,
                WRITER_ACCEPTANCE_SHA256,
            ),
            "publisher": identities["publisher"],
            "publisher_tests": identities["publisher_tests"],
        },
        "source_snapshot_sha256": source_snapshot.aggregate_sha256,
        "data_snapshot_sha256": data_snapshot.aggregate_sha256,
        "manifest": {
            "path": MANIFEST_IDENTITY,
            "sha256": data_snapshot.manifest_sha256,
            "ordered_entry_path_sha256": (data_snapshot.ordered_entry_path_sha256),
            "encoding": "preserve-unsorted-append-only-indent-2",
        },
        "destination_state": destination_state,
        "authorization_state": (
            "accepted"
            if authority is not None
            else "absent-and-required-before-publication"
        ),
        "decision": (
            "AUTHORIZED_DRY_RUN_ONLY"
            if authority is not None
            else "VERIFIED_CANDIDATE_ONLY_NOT_AUTHORIZED"
        ),
    }
    return report


def _report_bytes(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=False,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def verify_only() -> dict[str, Any]:
    """Rebuild and validate the accepted candidate twice without authorization.

    This function has no call edge to any lock, stage, link, unlink, or
    manifest-replacement helper.
    """

    paths = _runtime_paths()
    _canonical_repository_guard(paths)
    source_before = _verify_accepted_source_identities(paths)
    data_before = _snapshot_data(
        paths,
        allow_exact_unregistered_target=True,
    )
    candidate = _build_candidate_twice(paths)
    source_after = _verify_accepted_source_identities(paths)
    data_after = _snapshot_data(
        paths,
        allow_exact_unregistered_target=True,
    )
    if source_after != source_before:
        raise IdentityError("accepted source snapshot changed during dry run")
    if data_after != data_before:
        raise IdentityError("data snapshot changed during dry run")
    return _candidate_report(
        paths=paths,
        candidate=candidate,
        source_snapshot=source_after,
        data_snapshot=data_after,
        authority=None,
    )


def authorized_dry_run() -> dict[str, Any]:
    """Run every authorization and candidate gate, but perform no writes."""

    paths = _runtime_paths()
    _canonical_repository_guard(paths)
    _preflight_authority_presence(paths)
    source_before = _verify_accepted_source_identities(paths)
    data_before = _snapshot_data(
        paths,
        allow_exact_unregistered_target=True,
    )
    candidate = _build_candidate_twice(paths)
    authority = _load_authority(
        paths,
        candidate=candidate,
        source_snapshot=source_before,
        data_snapshot=data_before,
        enforce_data_snapshot=not _snapshot_is_registered(data_before),
    )
    source_after = _verify_accepted_source_identities(paths)
    data_after = _snapshot_data(
        paths,
        allow_exact_unregistered_target=True,
    )
    if source_after != source_before or data_after != data_before:
        raise IdentityError("repository changed during authorized dry run")
    return _candidate_report(
        paths=paths,
        candidate=candidate,
        source_snapshot=source_after,
        data_snapshot=data_after,
        authority=authority,
    )


def _open_retained_repository_directory(
    paths: RuntimePaths,
    identity: str,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    """Open one fixed repository directory by component without following links."""

    identity = _validate_repository_identity(identity, label=f"{label} identity")
    parts = PurePosixPath(identity).parts
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current = os.open(paths.repository_root, flags)
        descriptors.append(current)
        for component in parts:
            current = os.open(component, flags, dir_fd=current)
            descriptors.append(current)
        retained = descriptors.pop()
        metadata = os.fstat(retained)
        named = (paths.repository_root / identity).lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino)
        ):
            os.close(retained)
            raise IdentityError(f"{label} retained directory identity changed")
        return retained, metadata
    except OSError as error:
        raise IdentityError(
            f"{label} is unavailable through the fixed no-follow path"
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _revalidate_retained_directory(
    path: Path,
    descriptor: int,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    """Bind a retained directory descriptor to its still-canonical name."""

    try:
        retained = os.fstat(descriptor)
        named = path.lstat()
    except OSError as error:
        raise IdentityError(f"{label} directory identity is unavailable") from error
    expected_identity = (expected.st_dev, expected.st_ino)
    if (
        not stat.S_ISDIR(retained.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or (retained.st_dev, retained.st_ino) != expected_identity
        or (named.st_dev, named.st_ino) != expected_identity
    ):
        raise IdentityError(f"{label} directory inode changed")


def _fsync_retained_directory(
    path: Path,
    descriptor: int,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    """Durably flush the exact retained directory and then rebind its name."""

    try:
        os.fsync(descriptor)
    except OSError as error:
        raise PublicationIOError(f"{label} directory fsync failed") from error
    _revalidate_retained_directory(
        path,
        descriptor,
        expected,
        label=f"{label} after fsync",
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise PublicationIOError(f"directory fsync failed: {directory}") from error
    finally:
        os.close(descriptor)


def _write_exact(descriptor: int, payload: bytes) -> None:
    written = os.write(descriptor, payload)
    if written != len(payload):
        raise PublicationIOError(
            f"short write rejected: {written} of {len(payload)} bytes"
        )


def _readback_descriptor(descriptor: int, expected_size: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(remaining, 1 << 20))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    extra = os.read(descriptor, 1)
    result = b"".join(chunks)
    if remaining or extra:
        raise PublicationIOError("descriptor readback length changed")
    return result


def _create_exclusive_stage(
    parent: Path,
    *,
    prefix: str,
    payload: bytes,
    mode: int = 0o600,
    directory_fd: int | None = None,
    parent_metadata: os.stat_result | None = None,
) -> Stage:
    parent_before = (
        os.fstat(directory_fd) if directory_fd is not None else parent.lstat()
    )
    if not stat.S_ISDIR(parent_before.st_mode) or (
        directory_fd is None
        and (
            stat.S_ISLNK(parent_before.st_mode)
            or not stat.S_ISDIR(parent_before.st_mode)
        )
    ):
        raise IdentityError(f"stage parent is not a nonsymlink directory: {parent}")
    if parent_metadata is not None and (
        parent_before.st_dev,
        parent_before.st_ino,
    ) != (parent_metadata.st_dev, parent_metadata.st_ino):
        raise IdentityError("retained stage-parent descriptor changed")
    if directory_fd is not None and parent_metadata is not None:
        _revalidate_retained_directory(
            parent,
            directory_fd,
            parent_metadata,
            label="stage parent before creation",
        )
    name = prefix + secrets.token_hex(16)
    path = parent / name
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        name if directory_fd is not None else path,
        flags,
        mode,
        dir_fd=directory_fd,
    )
    metadata: os.stat_result | None = None
    try:
        os.fchmod(descriptor, mode)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != mode
            or metadata.st_dev != parent_before.st_dev
        ):
            raise PublicationIOError("exclusive stage metadata is invalid")
        _write_exact(descriptor, payload)
        os.fsync(descriptor)
        if _readback_descriptor(descriptor, len(payload)) != payload:
            raise PublicationIOError("exclusive stage readback changed")
        metadata_after = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) != (
            metadata_after.st_dev,
            metadata_after.st_ino,
        ):
            raise PublicationIOError("exclusive stage inode changed")
        return Stage(path, metadata.st_dev, metadata.st_ino)
    except BaseException:
        try:
            metadata_now = (
                os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if directory_fd is not None
                else path.lstat()
            )
            if (
                metadata is not None
                and metadata_now.st_dev == metadata.st_dev
                and metadata_now.st_ino == metadata.st_ino
            ):
                if directory_fd is None:
                    path.unlink()
                    _fsync_directory(parent)
                else:
                    if parent_metadata is not None:
                        _revalidate_retained_directory(
                            parent,
                            directory_fd,
                            parent_metadata,
                            label="stage parent before failed-stage cleanup",
                        )
                    os.unlink(name, dir_fd=directory_fd)
                    if parent_metadata is None:
                        os.fsync(directory_fd)
                    else:
                        _fsync_retained_directory(
                            parent,
                            directory_fd,
                            parent_metadata,
                            label="stage parent after failed-stage cleanup",
                        )
        except (FileNotFoundError, OSError, AtmosphereFixturePublicationError):
            pass
        raise
    finally:
        os.close(descriptor)


def _unlink_owned_stage(stage: Stage) -> None:
    try:
        metadata = stage.path.lstat()
    except OSError as error:
        raise PublicationIOError("invocation-owned stage disappeared") from error
    if (
        metadata.st_dev != stage.device
        or metadata.st_ino != stage.inode
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise PublicationIOError("invocation-owned stage inode changed")
    stage.path.unlink()
    _fsync_directory(stage.path.parent)


def _unlink_owned_stage_at(
    stage: Stage,
    directory_fd: int,
    *,
    parent_path: Path | None = None,
    parent_metadata: os.stat_result | None = None,
    label: str = "stage parent",
) -> None:
    if (parent_path is None) != (parent_metadata is None):
        raise PublicationIOError("retained stage-parent identity is incomplete")
    if parent_path is not None and parent_metadata is not None:
        _revalidate_retained_directory(
            parent_path,
            directory_fd,
            parent_metadata,
            label=f"{label} before stage unlink",
        )
    try:
        metadata = os.stat(
            stage.path.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise PublicationIOError(
            "invocation-owned stage disappeared from retained parent"
        ) from error
    if (
        metadata.st_dev != stage.device
        or metadata.st_ino != stage.inode
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise PublicationIOError("invocation-owned stage inode changed")
    os.unlink(stage.path.name, dir_fd=directory_fd)
    if parent_path is None or parent_metadata is None:
        os.fsync(directory_fd)
    else:
        _fsync_retained_directory(
            parent_path,
            directory_fd,
            parent_metadata,
            label=f"{label} after stage unlink",
        )


def _read_regular_at(
    directory_fd: int,
    name: str,
    *,
    identity: str,
    require_single_link: bool,
) -> RegularFile:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise IdentityError(f"{identity} is not a regular file")
        if require_single_link and before.st_nlink != 1:
            raise IdentityError(f"{identity} must have one hard link")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise IdentityError(f"{identity} changed while being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise IdentityError(f"{identity} read was short")
        return RegularFile(
            identity=identity,
            payload=payload,
            device=before.st_dev,
            inode=before.st_ino,
            owner=before.st_uid,
            group=before.st_gid,
            mode=stat.S_IMODE(before.st_mode),
            links=before.st_nlink,
        )
    finally:
        os.close(descriptor)


def _validate_stage_at(
    stage: Stage,
    directory_fd: int,
    *,
    expected_payload: bytes,
    expected_mode: int,
    label: str,
) -> RegularFile:
    """Rebind a staged name to its retained inode and exact intended bytes."""

    staged = _read_regular_at(
        directory_fd,
        stage.path.name,
        identity=label,
        require_single_link=True,
    )
    if (
        (staged.device, staged.inode) != (stage.device, stage.inode)
        or staged.mode != expected_mode
        or staged.size != len(expected_payload)
        or staged.sha256 != _sha256(expected_payload)
        or staged.payload != expected_payload
    ):
        raise PublicationIOError(f"{label} identity, mode, or bytes changed")
    return staged


def _require_same_directory_inode(
    path: Path,
    expected: os.stat_result,
    *,
    label: str,
) -> None:
    current = path.lstat()
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise IdentityError(f"{label} directory inode changed")


def _scan_quarantine(paths: RuntimePaths) -> None:
    parents = {paths.data_root, paths.destination.parent}
    for parent in parents:
        try:
            children = sorted(os.scandir(parent), key=lambda item: item.name)
        except OSError as error:
            raise IdentityError(f"cannot scan publication parent: {parent}") from error
        for child in children:
            if child.name.startswith(
                (ARTIFACT_STAGE_PREFIX, MANIFEST_TEMPORARY_PREFIX)
            ):
                metadata = child.stat(follow_symlinks=False)
                kind = (
                    "artifact_stage"
                    if child.name.startswith(ARTIFACT_STAGE_PREFIX)
                    else "manifest_temporary"
                )
                raise QuarantineError(
                    "crash-left quarantine object hard-stops publication: "
                    f"{child.path} kind={kind} dev={metadata.st_dev} "
                    f"ino={metadata.st_ino} mode={stat.S_IMODE(metadata.st_mode):04o} "
                    f"links={metadata.st_nlink} bytes={metadata.st_size}"
                )


def _read_target(
    paths: RuntimePaths,
    *,
    require_single_link: bool = True,
) -> RegularFile | None:
    try:
        return _read_repository_regular(
            paths,
            DESTINATION_IDENTITY,
            label="canonical atmosphere fixture target",
            require_single_link=require_single_link,
        )
    except IdentityError as error:
        try:
            paths.destination.lstat()
        except FileNotFoundError:
            return None
        raise error


def _validate_exact_target(
    target: RegularFile,
    candidate: Candidate,
    *,
    accepted_link_counts: tuple[int, ...] = (1,),
) -> None:
    if target.links not in accepted_link_counts:
        raise IdentityError(
            "canonical target has an invalid hard-link count: "
            f"{target.links} not in {accepted_link_counts}"
        )
    if (
        target.size != len(candidate.archive_bytes)
        or target.sha256 != _sha256(candidate.archive_bytes)
        or target.payload != candidate.archive_bytes
    ):
        raise IdentityError("canonical target is not the exact authorized artifact")
    _validate_untrusted_archive(target.payload)


def _atomic_install_no_replace(
    paths: RuntimePaths,
    candidate: Candidate,
    *,
    data_fd: int,
    data_metadata: os.stat_result,
) -> str:
    parent = paths.destination.parent
    _revalidate_retained_directory(
        paths.data_root,
        data_fd,
        data_metadata,
        label="locked data before destination-parent open",
    )
    directory_fd, parent_before = _open_retained_repository_directory(
        paths,
        "data/fixtures",
        label="destination parent",
    )
    stage: Stage | None = None
    installed = False
    try:
        _revalidate_retained_directory(
            parent,
            directory_fd,
            parent_before,
            label="destination parent before stage creation",
        )
        stage = _create_exclusive_stage(
            parent,
            prefix=ARTIFACT_STAGE_PREFIX,
            payload=candidate.archive_bytes,
            directory_fd=directory_fd,
            parent_metadata=parent_before,
        )
        _fsync_retained_directory(
            parent,
            directory_fd,
            parent_before,
            label="destination parent after stage creation",
        )
        _validate_stage_at(
            stage,
            directory_fd,
            expected_payload=candidate.archive_bytes,
            expected_mode=0o600,
            label="artifact stage before install",
        )
        _revalidate_retained_directory(
            paths.data_root,
            data_fd,
            data_metadata,
            label="locked data before artifact install",
        )
        _revalidate_retained_directory(
            parent,
            directory_fd,
            parent_before,
            label="destination parent before artifact install",
        )
        try:
            os.link(
                stage.path.name,
                paths.destination.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            installed = True
        except FileExistsError:
            installed = False
        _fsync_retained_directory(
            parent,
            directory_fd,
            parent_before,
            label="destination parent after artifact install decision",
        )
        _revalidate_retained_directory(
            paths.data_root,
            data_fd,
            data_metadata,
            label="locked data after artifact install decision",
        )
        target = _read_regular_at(
            directory_fd,
            paths.destination.name,
            identity=DESTINATION_IDENTITY,
            require_single_link=False,
        )
        _validate_exact_target(
            target,
            candidate,
            accepted_link_counts=(2,) if installed else (1,),
        )
        if installed and (target.device, target.inode) != (stage.device, stage.inode):
            raise PublicationIOError(
                "installed target is not the validated stage inode"
            )
        _revalidate_retained_directory(
            parent,
            directory_fd,
            parent_before,
            label="destination parent before stage cleanup",
        )
        _unlink_owned_stage_at(
            stage,
            directory_fd,
            parent_path=parent,
            parent_metadata=parent_before,
            label="destination parent",
        )
        _revalidate_retained_directory(
            parent,
            directory_fd,
            parent_before,
            label="destination parent after stage cleanup",
        )
        final_target = _read_regular_at(
            directory_fd,
            paths.destination.name,
            identity=DESTINATION_IDENTITY,
            require_single_link=True,
        )
        _validate_exact_target(final_target, candidate)
        _revalidate_retained_directory(
            paths.data_root,
            data_fd,
            data_metadata,
            label="locked data after artifact publication",
        )
        return "installed" if installed else "identical-existing"
    except BaseException:
        if stage is not None:
            try:
                os.stat(
                    stage.path.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                _unlink_owned_stage_at(
                    stage,
                    directory_fd,
                    parent_path=parent,
                    parent_metadata=parent_before,
                    label="destination parent during failure cleanup",
                )
            except (OSError, AtmosphereFixturePublicationError):
                pass
        raise
    finally:
        os.close(directory_fd)


@contextmanager
def _exclusive_data_lock(
    paths: RuntimePaths,
) -> Iterator[tuple[int, os.stat_result]]:
    """Lock the one stable canonical data-directory inode shared by both lanes."""

    descriptor, data_metadata = _open_retained_repository_directory(
        paths,
        "data",
        label="canonical data publication lock",
    )
    locked = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
        except OSError as error:
            raise PublicationIOError(
                "exclusive canonical data-directory lock is unavailable"
            ) from error
        _revalidate_retained_directory(
            paths.data_root,
            descriptor,
            data_metadata,
            label="canonical data after lock acquisition",
        )
        try:
            yield descriptor, data_metadata
        finally:
            _revalidate_retained_directory(
                paths.data_root,
                descriptor,
                data_metadata,
                label="canonical data before lock release",
            )
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError as error:
            raise PublicationIOError(
                "canonical data-directory lock release failed"
            ) from error
        finally:
            os.close(descriptor)


def _construct_postpublication_manifest(
    manifest_file: RegularFile,
    authority: Authority,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    manifest, ordered_paths = _parse_manifest_file(manifest_file)
    if DESTINATION_IDENTITY in ordered_paths:
        raise ManifestError("destination already occurs in prepublication manifest")
    expected_pre_hash = authority.authorization["manifest"]["prepublication_sha256"]
    if manifest_file.sha256 != expected_pre_hash:
        raise ManifestError("prepublication manifest hash is stale")
    template = authority.authorization["manifest_entry_template"]
    if (
        _sha256(_template_encode(template))
        != authority.authorization["manifest_entry_template_sha256"]
    ):
        raise ManifestError("authorization template digest changed")
    realized = _realize_template(
        template,
        authorization_sha256=authority.authorization_file.sha256,
        review_sha256=authority.review_file.sha256,
    )
    before = deepcopy(manifest)
    manifest["entries"].append(realized)
    intended_bytes = _manifest_encode(manifest)
    reconstructed = deepcopy(manifest)
    removed = reconstructed["entries"].pop()
    if removed != realized or _manifest_encode(reconstructed) != manifest_file.payload:
        raise ManifestError("delete-last reconstruction did not reproduce M")
    if reconstructed != before:
        raise ManifestError("pre-existing manifest tree changed")
    parsed_intended = _parse_strict_json(
        intended_bytes,
        label="intended postpublication manifest",
    )
    if parsed_intended != manifest or tuple(parsed_intended) != tuple(manifest):
        raise ManifestError("intended manifest exact parse changed")
    return manifest, intended_bytes, realized


def _revalidate_manifest_transition_authority(
    paths: RuntimePaths,
    *,
    expected_manifest: RegularFile,
    intended_bytes: bytes,
    expected_authority: Authority,
    candidate: Candidate,
    source_snapshot: SourceSnapshot,
    data_snapshot: TreeSnapshot,
) -> None:
    """Rebind every authority-derived input at the final manifest boundary."""

    authority_now = _load_authority(
        paths,
        candidate=candidate,
        source_snapshot=source_snapshot,
        data_snapshot=data_snapshot,
        enforce_data_snapshot=True,
    )
    if authority_now != expected_authority:
        raise AuthorizationError(
            "detached authority identity changed at manifest replacement boundary"
        )
    authorization_file_now = _read_repository_regular(
        paths,
        AUTHORIZATION_IDENTITY,
        label="authorization at manifest replacement boundary",
    )
    review_file_now = _read_repository_regular(
        paths,
        RECORD_REVIEW_IDENTITY,
        label="record review at manifest replacement boundary",
    )
    if (
        authorization_file_now != expected_authority.authorization_file
        or review_file_now != expected_authority.review_file
    ):
        raise AuthorizationError(
            "detached authority bytes changed at manifest replacement boundary"
        )

    template = authority_now.authorization["manifest_entry_template"]
    template_digest = _sha256(_template_encode(template))
    if (
        template_digest != authority_now.authorization["manifest_entry_template_sha256"]
        or template_digest
        != expected_authority.authorization["manifest_entry_template_sha256"]
    ):
        raise AuthorizationError(
            "authorization template digest changed at manifest replacement boundary"
        )

    target = _read_target(paths)
    if target is None:
        raise PublicationIOError(
            "authorized artifact disappeared at manifest replacement boundary"
        )
    _validate_exact_target(target, candidate)

    _, intended_now, _ = _construct_postpublication_manifest(
        expected_manifest,
        authority_now,
    )
    if intended_now != intended_bytes:
        raise ManifestError(
            "intended manifest changed at manifest replacement boundary"
        )


def _replace_manifest_atomically(
    paths: RuntimePaths,
    *,
    expected_manifest: RegularFile,
    intended_bytes: bytes,
    data_fd: int,
    data_metadata: os.stat_result,
    expected_source_snapshot: SourceSnapshot | None = None,
    expected_data_snapshot: TreeSnapshot | None = None,
    expected_authority: Authority | None = None,
    candidate: Candidate | None = None,
) -> None:
    boundary_inputs = (
        expected_source_snapshot,
        expected_data_snapshot,
        expected_authority,
        candidate,
    )
    if any(value is None for value in boundary_inputs) and any(
        value is not None for value in boundary_inputs
    ):
        raise IdentityError(
            "pre-replacement source/data/authority identity is incomplete"
        )
    _revalidate_retained_directory(
        paths.data_root,
        data_fd,
        data_metadata,
        label="locked data before manifest staging",
    )
    stage = _create_exclusive_stage(
        paths.data_root,
        prefix=MANIFEST_TEMPORARY_PREFIX,
        payload=intended_bytes,
        mode=expected_manifest.mode,
        directory_fd=data_fd,
        parent_metadata=data_metadata,
    )
    _fsync_retained_directory(
        paths.data_root,
        data_fd,
        data_metadata,
        label="locked data after manifest staging",
    )
    replaced = False
    try:
        live = _read_regular_at(
            data_fd,
            paths.manifest.name,
            identity="manifest immediately before replacement",
            require_single_link=True,
        )
        if (
            live.device,
            live.inode,
            live.sha256,
        ) != (
            expected_manifest.device,
            expected_manifest.inode,
            expected_manifest.sha256,
        ):
            raise ManifestError("manifest raced before atomic replacement")
        _validate_stage_at(
            stage,
            data_fd,
            expected_payload=intended_bytes,
            expected_mode=expected_manifest.mode,
            label="manifest temporary before replacement",
        )
        _revalidate_retained_directory(
            paths.data_root,
            data_fd,
            data_metadata,
            label="locked data before manifest replacement",
        )
        if (
            expected_source_snapshot is not None
            and expected_data_snapshot is not None
            and expected_authority is not None
            and candidate is not None
        ):
            source_now = _verify_accepted_source_identities(paths)
            data_now = _snapshot_data(
                paths,
                allow_exact_unregistered_target=True,
                owned_manifest_temporary=stage,
                owned_manifest_bytes=intended_bytes,
                owned_manifest_mode=expected_manifest.mode,
            )
            if source_now != expected_source_snapshot:
                raise IdentityError(
                    "accepted source snapshot changed immediately before "
                    "manifest replacement"
                )
            if (
                data_now.aggregate_sha256 != expected_data_snapshot.aggregate_sha256
                or data_now.manifest_sha256 != expected_data_snapshot.manifest_sha256
                or data_now.ordered_entry_path_sha256
                != expected_data_snapshot.ordered_entry_path_sha256
            ):
                raise IdentityError(
                    "closed data snapshot changed immediately before "
                    "manifest replacement"
                )
            _revalidate_manifest_transition_authority(
                paths,
                expected_manifest=expected_manifest,
                intended_bytes=intended_bytes,
                expected_authority=expected_authority,
                candidate=candidate,
                source_snapshot=source_now,
                data_snapshot=data_now,
            )
            _validate_stage_at(
                stage,
                data_fd,
                expected_payload=intended_bytes,
                expected_mode=expected_manifest.mode,
                label="manifest temporary after final source/data snapshot",
            )
            _revalidate_retained_directory(
                paths.data_root,
                data_fd,
                data_metadata,
                label="locked data at manifest replacement boundary",
            )
        os.replace(
            stage.path.name,
            paths.manifest.name,
            src_dir_fd=data_fd,
            dst_dir_fd=data_fd,
        )
        replaced = True
        _fsync_retained_directory(
            paths.data_root,
            data_fd,
            data_metadata,
            label="locked data after manifest replacement",
        )
        final_manifest = _read_regular_at(
            data_fd,
            paths.manifest.name,
            identity="postpublication manifest",
            require_single_link=True,
        )
        if final_manifest.payload != intended_bytes:
            raise ManifestError("postpublication manifest readback changed")
        if final_manifest.mode != expected_manifest.mode:
            raise ManifestError("postpublication manifest mode changed")
        _parse_manifest_file(final_manifest)
    except BaseException:
        if not replaced:
            try:
                _unlink_owned_stage_at(
                    stage,
                    data_fd,
                    parent_path=paths.data_root,
                    parent_metadata=data_metadata,
                    label="locked data during manifest-stage cleanup",
                )
            except (OSError, AtmosphereFixturePublicationError):
                pass
        raise


def _registered_state_matches(
    paths: RuntimePaths,
    *,
    authority: Authority,
    candidate: Candidate,
) -> bool:
    manifest_file = _read_repository_regular(
        paths,
        MANIFEST_IDENTITY,
        label="current manifest",
    )
    manifest, ordered = _parse_manifest_file(manifest_file)
    if DESTINATION_IDENTITY not in ordered:
        return False
    if ordered.count(DESTINATION_IDENTITY) != 1 or ordered[-1] != DESTINATION_IDENTITY:
        raise ManifestError("registered target is duplicate or not final append")
    final_entry = manifest["entries"][-1]
    reconstructed = deepcopy(manifest)
    removed = reconstructed["entries"].pop()
    pre_bytes = _manifest_encode(reconstructed)
    if (
        _sha256(pre_bytes)
        != authority.authorization["manifest"]["prepublication_sha256"]
    ):
        raise ManifestError("registered state does not reconstruct authorized M")
    expected_entry = _realize_template(
        authority.authorization["manifest_entry_template"],
        authorization_sha256=authority.authorization_file.sha256,
        review_sha256=authority.review_file.sha256,
    )
    if removed != expected_entry or final_entry != expected_entry:
        raise ManifestError("registered state entry differs from realized E")
    target = _read_target(paths)
    if target is None:
        raise ManifestError("registered target is missing")
    _validate_exact_target(target, candidate)
    return True


def _fresh_postpublication_validation(paths: RuntimePaths) -> None:
    if paths.repository_root != CANONICAL_REPOSITORY_ROOT:
        # Isolated adversarial tests validate in-process; production always
        # uses a new interpreter with no path argument.
        target = _read_target(paths)
        if target is None:
            raise ManifestError("isolated postpublication target is absent")
        return
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(paths.repository_root / PUBLISHER_IDENTITY),
            "--internal-validate-published",
        ],
        cwd=paths.repository_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise ManifestError(f"fresh postpublication validation failed: {detail}")


def verify_published() -> dict[str, Any]:
    """Validate the registered fixture without replaying its retired publisher.

    The detached records preserve the byte identities of the implementation
    that performed publication.  Live maintenance code and later chapter data
    are allowed to evolve; the authority/review/manifest/artifact graph is not.
    """

    paths = _runtime_paths()
    _canonical_repository_guard(paths)
    authorization_file = _read_repository_regular(
        paths,
        AUTHORIZATION_IDENTITY,
        label="published authorization",
    )
    review_file = _read_repository_regular(
        paths,
        RECORD_REVIEW_IDENTITY,
        label="published authorization review",
    )
    publisher_acceptance = _read_repository_regular(
        paths,
        PUBLISHER_ACCEPTANCE_IDENTITY,
        label="historical publisher acceptance",
    )
    authorization = _parse_strict_json(
        authorization_file.payload,
        label="published authorization",
    )
    _require_exact_keys(authorization, AUTHORIZATION_KEYS, label="authorization")
    if (
        authorization["schema_version"] != 1
        or authorization["record_kind"]
        != "chapter06_atmosphere_fixture_publication_acceptance"
        or authorization["lane"] != "atmosphere_fixture"
        or authorization["manifest_role"] != MANIFEST_ROLE
        or authorization["payne_zero_commit"] != PINNED_PAYNE_ZERO_COMMIT
    ):
        raise AuthorizationError("published authorization lane identity changed")
    if authorization["publisher_acceptance"] != {
        "path": PUBLISHER_ACCEPTANCE_IDENTITY,
        "sha256": publisher_acceptance.sha256,
    }:
        raise AuthorizationError("historical publisher acceptance binding changed")

    target = _read_target(paths)
    if target is None:
        raise ManifestError("published atmosphere fixture is absent")
    if (
        target.size != EXPECTED_ARCHIVE_BYTES
        or target.sha256 != EXPECTED_ARCHIVE_SHA256
        or target.mode != 0o600
        or target.links != 1
    ):
        raise CandidateError("published atmosphere fixture identity changed")
    array_records = _validate_untrusted_archive(target.payload)
    artifact = authorization["artifact"]
    _require_exact_keys(
        artifact,
        ARTIFACT_AUTHORIZATION_KEYS,
        label="authorization artifact",
    )
    expected_artifact = {
        "path": DESTINATION_IDENTITY,
        "filename": Path(DESTINATION_IDENTITY).name,
        "bytes": EXPECTED_ARCHIVE_BYTES,
        "sha256": EXPECTED_ARCHIVE_SHA256,
        "member_count": EXPECTED_MEMBER_COUNT,
        "archive_kind": ARCHIVE_KIND,
        "fixture_capture_schema_version": EXTERNAL_CAPTURE_SCHEMA_VERSION,
        "archive_contains_embedded_schema_version": False,
        "npy_member_format_version": NPY_MEMBER_FORMAT_VERSION,
        "scientific_fixture_schema_digest": EXPECTED_FIXTURE_SCHEMA_DIGEST,
        "scientific_payload_fingerprint": EXPECTED_PAYLOAD_FINGERPRINT,
    }
    if artifact != expected_artifact:
        raise AuthorizationError("published artifact authority changed")

    template = authorization["manifest_entry_template"]
    if not isinstance(template, dict):
        raise AuthorizationError("published manifest template is not an object")
    if (
        _sha256(_template_encode(template))
        != authorization["manifest_entry_template_sha256"]
        or template.get("path") != DESTINATION_IDENTITY
        or template.get("role") != MANIFEST_ROLE
        or template.get("sha256") != EXPECTED_ARCHIVE_SHA256
        or template.get("bytes") != EXPECTED_ARCHIVE_BYTES
        or template.get("arrays") != array_records
    ):
        raise AuthorizationError("published manifest template changed")
    if (
        template.get("publisher_sha256")
        != authorization["publisher"]["sha256"]
        or template.get("publisher_test_sha256")
        != authorization["publisher_tests"]["sha256"]
        or template.get("publisher_acceptance_sha256")
        != authorization["publisher_acceptance"]["sha256"]
    ):
        raise AuthorizationError("published template identity joins changed")

    review = _parse_strict_json(
        review_file.payload,
        label="published authorization review",
    )
    _require_exact_keys(review, RECORD_REVIEW_KEYS, label="record review")
    expected_review = {
        "schema_version": 1,
        "record_kind": "chapter06_atmosphere_fixture_publication_record_review",
        "authorization_path": AUTHORIZATION_IDENTITY,
        "authorization_sha256": authorization_file.sha256,
        "candidate_byte_acceptance_path": CANDIDATE_BYTE_ACCEPTANCE_IDENTITY,
        "candidate_byte_acceptance_sha256": CANDIDATE_BYTE_ACCEPTANCE_SHA256,
        "publisher_acceptance_path": PUBLISHER_ACCEPTANCE_IDENTITY,
        "publisher_acceptance_sha256": publisher_acceptance.sha256,
        "manifest_entry_template_sha256": authorization[
            "manifest_entry_template_sha256"
        ],
        "disposition": "ACCEPT",
    }
    if review != expected_review:
        raise AuthorizationError("published authorization review changed")
    expected_entry = _realize_template(
        template,
        authorization_sha256=authorization_file.sha256,
        review_sha256=review_file.sha256,
    )
    manifest_file = _read_repository_regular(
        paths,
        MANIFEST_IDENTITY,
        label="live manifest",
    )
    manifest, ordered_paths = _parse_manifest_file(manifest_file)
    matches = [
        entry
        for entry in manifest["entries"]
        if entry["path"] == DESTINATION_IDENTITY
    ]
    if matches != [expected_entry] or ordered_paths.count(DESTINATION_IDENTITY) != 1:
        raise ManifestError(
            "published atmosphere manifest entry differs from its realization"
        )
    return {
        "schema_version": 1,
        "report_kind": "chapter06_atmosphere_postpublication_verification",
        "state": "exact_registered",
        "artifact_sha256": target.sha256,
        "artifact_bytes": target.size,
        "archive_member_count": len(array_records),
        "authorization_sha256": authorization_file.sha256,
        "record_review_sha256": review_file.sha256,
        "realized_entry_sha256": _entry_digest(expected_entry),
        "manifest_sha256": manifest_file.sha256,
        "later_manifest_entry_count": (
            len(ordered_paths) - ordered_paths.index(DESTINATION_IDENTITY) - 1
        ),
        "complete_validation": True,
    }


def publish() -> str:
    """Publish the one exact fixture only under detached reviewed authority."""

    paths = _runtime_paths()
    _canonical_repository_guard(paths)
    # This deliberate early check proves that the current candidate cannot
    # create a lock, parent, stage, target, or manifest temporary.
    _preflight_authority_presence(paths)

    source_before = _verify_accepted_source_identities(paths)
    data_before = _snapshot_data(
        paths,
        allow_exact_unregistered_target=True,
    )
    candidate = _build_candidate_twice(paths)
    authority = _load_authority(
        paths,
        candidate=candidate,
        source_snapshot=source_before,
        data_snapshot=data_before,
        enforce_data_snapshot=not _snapshot_is_registered(data_before),
    )

    with _exclusive_data_lock(paths) as (data_fd, data_metadata):
        _scan_quarantine(paths)
        source_locked = _verify_accepted_source_identities(paths)
        data_locked = _snapshot_data(
            paths,
            allow_exact_unregistered_target=True,
        )
        if source_locked != source_before or data_locked != data_before:
            raise IdentityError("source or data snapshot changed before mutation")
        authority_locked = _load_authority(
            paths,
            candidate=candidate,
            source_snapshot=source_locked,
            data_snapshot=data_locked,
            enforce_data_snapshot=not _snapshot_is_registered(data_locked),
        )
        if (
            authority_locked.authorization_file.sha256
            != authority.authorization_file.sha256
            or authority_locked.review_file.sha256 != authority.review_file.sha256
        ):
            raise AuthorizationError("authority changed after lock acquisition")

        if _registered_state_matches(
            paths,
            authority=authority_locked,
            candidate=candidate,
        ):
            _fresh_postpublication_validation(paths)
            return "identical-registered-no-op"

        target = _read_target(paths)
        if target is None:
            install_result = _atomic_install_no_replace(
                paths,
                candidate,
                data_fd=data_fd,
                data_metadata=data_metadata,
            )
        else:
            _validate_exact_target(target, candidate)
            install_result = "exact-unregistered-recovery"

        manifest_file = _read_repository_regular(
            paths,
            MANIFEST_IDENTITY,
            label="prepublication manifest",
        )
        _, intended_bytes, _ = _construct_postpublication_manifest(
            manifest_file,
            authority_locked,
        )

        # Last rechecks before the one allowed manifest replacement.
        source_pre_manifest = _verify_accepted_source_identities(paths)
        data_pre_manifest = _snapshot_data(
            paths,
            allow_exact_unregistered_target=True,
        )
        if source_pre_manifest != source_locked:
            raise IdentityError("source snapshot changed after artifact installation")
        if data_pre_manifest.aggregate_sha256 != data_locked.aggregate_sha256:
            raise IdentityError(
                "nontarget data or manifest changed after artifact installation"
            )
        if data_pre_manifest.manifest_sha256 != data_locked.manifest_sha256:
            raise ManifestError("manifest changed after artifact installation")
        if _sha256(candidate.archive_bytes) != EXPECTED_ARCHIVE_SHA256:
            raise CandidateError("candidate bytes changed before manifest transition")
        if (
            _read_repository_regular(
                paths,
                AUTHORIZATION_IDENTITY,
                label="authorization before manifest transition",
            ).sha256
            != authority_locked.authorization_file.sha256
        ):
            raise AuthorizationError("authorization changed before manifest transition")
        if (
            _read_repository_regular(
                paths,
                RECORD_REVIEW_IDENTITY,
                label="record review before manifest transition",
            ).sha256
            != authority_locked.review_file.sha256
        ):
            raise AuthorizationError("record review changed before manifest transition")
        final_target = _read_target(paths)
        if final_target is None:
            raise PublicationIOError("installed artifact disappeared")
        _validate_exact_target(final_target, candidate)
        _replace_manifest_atomically(
            paths,
            expected_manifest=manifest_file,
            intended_bytes=intended_bytes,
            data_fd=data_fd,
            data_metadata=data_metadata,
            expected_source_snapshot=source_locked,
            expected_data_snapshot=data_locked,
            expected_authority=authority_locked,
            candidate=candidate,
        )
        if not _registered_state_matches(
            paths,
            authority=authority_locked,
            candidate=candidate,
        ):
            raise ManifestError("postpublication state is not registered")
        _fresh_postpublication_validation(paths)
        return f"{install_result}-and-registered"


def _internal_validate_published() -> None:
    paths = _runtime_paths()
    _canonical_repository_guard(paths)
    source = _verify_accepted_source_identities(paths)
    data = _snapshot_data(paths, allow_exact_unregistered_target=False)
    # Reconstructing four scientific children is unnecessary here.  The
    # accepted exact candidate bytes are independently decoded and the strict
    # authority/entry graph is rechecked against an equivalent Candidate.
    target = _read_target(paths)
    if target is None:
        raise ManifestError("fresh validator found no canonical target")
    arrays = _validate_untrusted_archive(target.payload)
    candidate = Candidate(
        target.payload,
        arrays,
        {},
        {},
    )
    authority = _load_authority(
        paths,
        candidate=candidate,
        source_snapshot=source,
        data_snapshot=data,
        enforce_data_snapshot=False,
    )
    if not _registered_state_matches(
        paths,
        authority=authority,
        candidate=candidate,
    ):
        raise ManifestError("fresh validator found an incomplete registered state")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the fixed Chapter 6 atmosphere fixture candidate or run "
            "its separately authorized fixed-path publisher."
        )
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--verify-only",
        action="store_true",
        help="rebuild twice and validate without requiring publication authority",
    )
    modes.add_argument(
        "--authorized-dry-run",
        action="store_true",
        help="require detached authority and verify without writing",
    )
    modes.add_argument(
        "--publish",
        action="store_true",
        help="run the fixed-path no-replace publisher",
    )
    modes.add_argument(
        "--verify-published",
        action="store_true",
        help="validate the registered artifact and historical authority graph",
    )
    modes.add_argument(
        "--internal-validate-published",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    try:
        if arguments.verify_only:
            sys.stdout.buffer.write(_report_bytes(verify_only()))
        elif arguments.authorized_dry_run:
            sys.stdout.buffer.write(_report_bytes(authorized_dry_run()))
        elif arguments.publish:
            print(publish())
        elif arguments.verify_published:
            sys.stdout.buffer.write(_report_bytes(verify_published()))
        else:
            _internal_validate_published()
    except AtmosphereFixturePublicationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
