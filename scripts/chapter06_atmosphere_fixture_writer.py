"""Deterministic in-memory NPZ bytes for the accepted Chapter 6 atmosphere fixture.

This gate binds the independently accepted scientific worker, owns two fresh
child captures, validates their complete fixture and ephemeral-evidence
mappings, and returns one canonical byte string containing only the nineteen
accepted fixture arrays.

Process evidence and the worker's scientific audit evidence remain outside the
nineteen-member archive.  This module has no output path, filesystem artifact
writer, command-line entry point, manifest mutation, authorization parser,
publisher, canonical fixture, golden, overwrite, or alternate-root operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np

from scripts import chapter06_atmosphere_fixture_worker as worker


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRITER_PATH = Path(__file__).resolve()
PINNED_ROOT = Path("/Users/ysting/payne-zero")
PAPER_ROOT = Path("/Users/ysting/Source_Files_Not_For_Review")
FORBIDDEN_CACHE_ROOTS = (REPOSITORY_ROOT, PINNED_ROOT, PAPER_ROOT)
CACHE_CAPTURE_TOKEN_ENV = "CHAPTER06_ATMOSPHERE_CAPTURE_TOKEN"

ACCEPTED_FILE_IDENTITIES = {
    "scientific_worker": (
        REPOSITORY_ROOT / "scripts/chapter06_atmosphere_fixture_worker.py",
        "db107c9f67c5f074e0aa77b3f523c781b8f63dce2819ee5a1248ff9e6fb1ec84",
    ),
    "scientific_worker_tests": (
        REPOSITORY_ROOT / "tests/test_chapter06_atmosphere_fixture_worker.py",
        "069d13277e380cc70706da4ec3db71ef6e862c055432e6578169596414ccb408",
    ),
    "scientific_worker_candidate": (
        REPOSITORY_ROOT / "design/chapter06_atmosphere_fixture_worker_candidate.md",
        "da53e4846ee91f814c437994fff604ae91ed94a4da4db7856f8f47bb61cf72dc",
    ),
    "scientific_worker_independent_audit": (
        REPOSITORY_ROOT
        / "design/chapter06_atmosphere_fixture_worker_independent_audit.md",
        "336372a0d37f5f46b2e53dbadd382ac87a24f3b76129307a4753e01be414d52e",
    ),
}

ACCEPTED_FIXTURE_MEMBER_COUNT = 19
ACCEPTED_EVIDENCE_MEMBER_COUNT = 89
ACCEPTED_FIXTURE_SCHEMA_DIGEST = (
    "f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698"
)
ACCEPTED_PAYLOAD_FINGERPRINT = (
    "f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663"
)
ACCEPTED_FULL_CAPTURE_SCHEMA_DIGEST = (
    "cdf470038e67301b4c19b0691e672cd97df3233a3decf1b88e32ce3ac0dc1371"
)
ACCEPTED_FULL_CAPTURE_FINGERPRINT = (
    "14a2ff95ed468b87cc92b638f7bfdec25e67e9d6d5569bf3ea2f8e06cd4b5a07"
)
ACCEPTED_LINE_OUTPUT_SHA256 = (
    "43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58"
)

FIXTURE_ARCHIVE_SIZE_CEILING_BYTES = 1_048_576
CAPTURE_TRANSPORT_SIZE_CEILING_BYTES = 128 * 1_048_576
NPY_VERSION = (2, 0)
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
ZIP_CREATE_SYSTEM = 3
ZIP_CREATE_VERSION = 20
ZIP_EXTRACT_VERSION = 20
ZIP_EXTERNAL_ATTR = 0o100600 << 16
ZIP_INTERNAL_ATTR = 0
ZIP_FLAG_BITS = 0
ZIP_COMPRESSION = zipfile.ZIP_STORED

_FIXTURE_TRANSPORT_PREFIX = "fixture__"
_EVIDENCE_TRANSPORT_PREFIX = "evidence__"


class AtmosphereFixtureWriterError(RuntimeError):
    """Raised when accepted authority, A/B parity, or archive bytes drift."""


@dataclass(frozen=True)
class DeterministicAtmosphereFixtureArchive:
    """One immutable canonical fixture byte string and JSON-safe evidence."""

    archive_bytes: bytes
    summary: dict[str, Any]


@dataclass(frozen=True)
class _FreshAtmosphereCapture:
    """One child-owned accepted capture plus non-scientific origin evidence."""

    fixture_arrays: dict[str, np.ndarray]
    ephemeral_evidence: dict[str, np.ndarray]
    capture_transport_bytes: bytes
    capture_transport_sha256: str
    origin_token_sha256: str
    child_pid: int
    cache_root_sha256: str
    cache_empty_before: bool
    cache_entry_count_after: int
    cache_external: bool
    cache_nonsymlink: bool


def sha256_bytes(value: bytes) -> str:
    """Return one immutable byte string's SHA-256 identity."""

    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_regular_nonsymlink(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise AtmosphereFixtureWriterError(
            f"{label} must not be a symlink: {candidate}"
        )
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise AtmosphereFixtureWriterError(f"{label} is missing: {candidate}")
    return resolved


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_cache_directory(path: Path, *, require_empty: bool) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise AtmosphereFixtureWriterError(
            f"capture cache must be an existing nonsymlink directory: {candidate}"
        )
    resolved = candidate.resolve()
    if any(_is_under(resolved, root) for root in FORBIDDEN_CACHE_ROOTS):
        raise AtmosphereFixtureWriterError(
            f"capture cache must be external to all source/data trees: {resolved}"
        )
    if require_empty and any(resolved.iterdir()):
        raise AtmosphereFixtureWriterError(
            f"capture cache must be truly empty: {resolved}"
        )
    return resolved


def _require_distinct_cache_roots(cache_a: Path, cache_b: Path) -> None:
    if cache_a.resolve() == cache_b.resolve():
        raise AtmosphereFixtureWriterError(
            "fresh atmosphere capture A/B cache roots must be distinct"
        )


def _require_nonconflicting_capture_environment() -> dict[str, str]:
    """Reject inherited conflicts, then complete absent accepted controls."""

    conflicts = {
        name: os.environ[name]
        for name, expected in worker.CAPTURE_ENVIRONMENT.items()
        if name in os.environ and os.environ[name] != expected
    }
    if "PAYNE_ZERO_DATA_ROOT" in os.environ:
        conflicts["PAYNE_ZERO_DATA_ROOT"] = os.environ["PAYNE_ZERO_DATA_ROOT"]
    if conflicts:
        raise AtmosphereFixtureWriterError(
            "inherited atmosphere capture controls conflict with the accepted "
            "worker environment: " + json.dumps(conflicts, sort_keys=True)
        )
    return {
        **dict(os.environ),
        **worker.CAPTURE_ENVIRONMENT,
    }


def verify_accepted_identities() -> dict[str, str]:
    """Require the exact independently accepted atmosphere-worker snapshot."""

    identities: dict[str, str] = {}
    for label, (path, expected_hash) in sorted(ACCEPTED_FILE_IDENTITIES.items()):
        resolved = _require_regular_nonsymlink(path, label)
        actual_hash = _file_sha256(resolved)
        if actual_hash != expected_hash:
            raise AtmosphereFixtureWriterError(
                f"accepted identity changed for {label}: "
                f"{actual_hash}; expected {expected_hash}"
            )
        identities[label] = actual_hash
    if (
        worker.WORKER_PATH.resolve()
        != ACCEPTED_FILE_IDENTITIES["scientific_worker"][0].resolve()
    ):
        raise AtmosphereFixtureWriterError(
            "atmosphere scientific worker imported from an unexpected path"
        )
    return identities


def _bitwise_equal(left: Any, right: Any) -> bool:
    first = np.asarray(left)
    second = np.asarray(right)
    return (
        first.dtype.str == second.dtype.str
        and first.shape == second.shape
        and np.ascontiguousarray(first).tobytes(order="C")
        == np.ascontiguousarray(second).tobytes(order="C")
    )


def _mapping_digest(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.asarray(values[name])
        if array.dtype.hasobject:
            raise AtmosphereFixtureWriterError(f"object dtype is forbidden: {name}")
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _combined_worker_mapping(
    fixture: Mapping[str, np.ndarray],
    evidence: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        **{f"fixture__{name}": np.asarray(value) for name, value in fixture.items()},
        **{f"evidence__{name}": np.asarray(value) for name, value in evidence.items()},
    }


def _scalar_text(values: Mapping[str, np.ndarray], name: str) -> str:
    if name not in values:
        raise AtmosphereFixtureWriterError(f"accepted evidence is missing: {name}")
    value = np.asarray(values[name])
    if value.shape != ():
        raise AtmosphereFixtureWriterError(
            f"accepted evidence member is not scalar: {name}"
        )
    return str(value.item())


def _scalar_bool(values: Mapping[str, np.ndarray], name: str) -> bool:
    if name not in values:
        raise AtmosphereFixtureWriterError(f"accepted evidence is missing: {name}")
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.kind != "b":
        raise AtmosphereFixtureWriterError(
            f"accepted evidence member is not a Boolean scalar: {name}"
        )
    return bool(value.item())


def _validate_accepted_capture(
    fixture: Mapping[str, np.ndarray],
    evidence: Mapping[str, np.ndarray],
) -> None:
    """Validate all accepted worker evidence before fixture serialization."""

    if len(fixture) != ACCEPTED_FIXTURE_MEMBER_COUNT:
        raise AtmosphereFixtureWriterError("accepted fixture member count changed")
    if set(fixture) != set(worker.FIXTURE_SCHEMA):
        raise AtmosphereFixtureWriterError("accepted fixture member set changed")
    if len(evidence) != ACCEPTED_EVIDENCE_MEMBER_COUNT:
        raise AtmosphereFixtureWriterError("accepted evidence member count changed")

    for group_name, values in (("fixture", fixture), ("evidence", evidence)):
        for name, value in values.items():
            if not isinstance(name, str) or not name:
                raise AtmosphereFixtureWriterError(
                    f"{group_name} contains an invalid member name"
                )
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise AtmosphereFixtureWriterError(
                    f"{group_name} member has object dtype: {name}"
                )
            if not array.flags.c_contiguous:
                raise AtmosphereFixtureWriterError(
                    f"{group_name} member is not C-contiguous: {name}"
                )

    try:
        worker._validate_fixture_mapping(fixture)
    except (RuntimeError, TypeError, ValueError) as error:
        raise AtmosphereFixtureWriterError(
            f"accepted nineteen-member fixture validation failed: {error}"
        ) from error

    expected_scalars = {
        "meta__fixture_schema_digest": ACCEPTED_FIXTURE_SCHEMA_DIGEST,
        "meta__payload_fingerprint": ACCEPTED_PAYLOAD_FINGERPRINT,
        "meta__full_capture_schema_digest": ACCEPTED_FULL_CAPTURE_SCHEMA_DIGEST,
        "meta__full_capture_fingerprint": ACCEPTED_FULL_CAPTURE_FINGERPRINT,
        "meta__worker_sha256": ACCEPTED_FILE_IDENTITIES["scientific_worker"][1],
        "line__full_output_sha256": ACCEPTED_LINE_OUTPUT_SHA256,
        "line__projected_output_sha256": ACCEPTED_LINE_OUTPUT_SHA256,
        "line__placeholder_output_sha256": ACCEPTED_LINE_OUTPUT_SHA256,
    }
    for name, expected in expected_scalars.items():
        if _scalar_text(evidence, name) != expected:
            raise AtmosphereFixtureWriterError(
                f"accepted worker evidence changed: {name}"
            )

    required_true = (
        "line__full_projected_bitwise_equal",
        "line__placeholder_bitwise_equal",
        "placeholder__all_fixture_members_bitwise_equal",
        "payload__full_projected_line_equal",
        "payload__placeholder_fixture_equal",
        "payload__placeholder_line_equal",
    )
    required_false = (
        "meta__capture_scope_complete",
        "meta__fixture_publication_performed",
        "meta__golden_read_performed",
        "meta__golden_write_performed",
        "meta__manifest_write_performed",
    )
    for name in required_true:
        if not _scalar_bool(evidence, name):
            raise AtmosphereFixtureWriterError(
                f"accepted worker parity evidence changed: {name}"
            )
    for name in required_false:
        if _scalar_bool(evidence, name):
            raise AtmosphereFixtureWriterError(
                f"worker no-publication evidence changed: {name}"
            )

    environment_values = {
        name.removeprefix("environment__"): _scalar_text(evidence, name)
        for name in evidence
        if name.startswith("environment__")
        and name
        not in {
            "environment__cache_policy",
            "environment__cpu_only",
        }
    }
    expected_environment = {
        name.lower(): expected for name, expected in worker.CAPTURE_ENVIRONMENT.items()
    }
    if environment_values != expected_environment:
        raise AtmosphereFixtureWriterError(
            "complete accepted worker environment evidence changed"
        )
    if not _scalar_bool(evidence, "environment__cpu_only"):
        raise AtmosphereFixtureWriterError("worker CPU-only evidence changed")

    if worker.mapping_schema_digest(fixture) != ACCEPTED_FIXTURE_SCHEMA_DIGEST:
        raise AtmosphereFixtureWriterError(
            "fixture schema digest recomputation changed"
        )

    payload_mapping = {
        **{f"fixture__{name}": value for name, value in fixture.items()},
        **{
            name: value
            for name, value in evidence.items()
            if name.startswith("payload__")
        },
    }
    if worker.mapping_fingerprint(payload_mapping) != ACCEPTED_PAYLOAD_FINGERPRINT:
        raise AtmosphereFixtureWriterError(
            "physical payload fingerprint recomputation changed"
        )
    combined = _combined_worker_mapping(fixture, evidence)
    if worker.mapping_schema_digest(combined) != ACCEPTED_FULL_CAPTURE_SCHEMA_DIGEST:
        raise AtmosphereFixtureWriterError(
            "full capture schema digest recomputation changed"
        )
    evidence_without_fingerprint = {
        name: value
        for name, value in evidence.items()
        if name != "meta__full_capture_fingerprint"
    }
    if (
        worker.mapping_fingerprint(
            _combined_worker_mapping(fixture, evidence_without_fingerprint)
        )
        != ACCEPTED_FULL_CAPTURE_FINGERPRINT
    ):
        raise AtmosphereFixtureWriterError(
            "full capture fingerprint recomputation changed"
        )


def _require_mapping_pair_identity(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    *,
    label: str,
) -> str:
    if tuple(sorted(first)) != tuple(sorted(second)):
        raise AtmosphereFixtureWriterError(f"{label} A/B member sets disagree")
    for name in sorted(first):
        if not _bitwise_equal(first[name], second[name]):
            raise AtmosphereFixtureWriterError(
                f"{label} A/B dtype, shape, or C bytes disagree: {name}"
            )
    first_digest = _mapping_digest(first)
    if first_digest != _mapping_digest(second):
        raise AtmosphereFixtureWriterError(f"{label} A/B mapping digests disagree")
    return first_digest


def _npy_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise AtmosphereFixtureWriterError(
            "object dtype is forbidden in deterministic NPZ"
        )
    if not array.flags.c_contiguous:
        raise AtmosphereFixtureWriterError(
            "non-C-contiguous arrays are forbidden in deterministic NPZ"
        )
    stream = BytesIO()
    np.lib.format.write_array(
        stream,
        array,
        version=NPY_VERSION,
        allow_pickle=False,
    )
    return stream.getvalue()


def _zip_info(member_name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(member_name, date_time=ZIP_DATE_TIME)
    info.compress_type = ZIP_COMPRESSION
    info.create_system = ZIP_CREATE_SYSTEM
    info.create_version = ZIP_CREATE_VERSION
    info.extract_version = ZIP_EXTRACT_VERSION
    info.flag_bits = ZIP_FLAG_BITS
    info.volume = 0
    info.internal_attr = ZIP_INTERNAL_ATTR
    info.external_attr = ZIP_EXTERNAL_ATTR
    info.extra = b""
    info.comment = b""
    return info


def _validate_base_member_names(names: list[str]) -> None:
    if len(names) != len(set(names)):
        raise AtmosphereFixtureWriterError(
            "deterministic mapping has duplicate member names"
        )
    if any(
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\\" in name
        or name.endswith(".npy")
        or name in {".", ".."}
        for name in names
    ):
        raise AtmosphereFixtureWriterError(
            "deterministic mapping has an unsafe or path-like member name"
        )


def _serialize_mapping(
    values: Mapping[str, np.ndarray],
    *,
    size_ceiling: int,
) -> bytes:
    names = sorted(values)
    _validate_base_member_names(names)
    stream = BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=ZIP_COMPRESSION,
        allowZip64=False,
        strict_timestamps=True,
    ) as archive:
        archive.comment = b""
        for name in names:
            archive.writestr(
                _zip_info(f"{name}.npy"),
                _npy_bytes(values[name]),
                compress_type=ZIP_COMPRESSION,
            )
    result = stream.getvalue()
    if len(result) > size_ceiling:
        raise AtmosphereFixtureWriterError(
            f"deterministic archive is {len(result)} bytes; ceiling is {size_ceiling}"
        )
    return result


def _validate_zip_info(info: zipfile.ZipInfo, expected_name: str) -> None:
    checks = {
        "filename": info.filename == expected_name,
        "date_time": info.date_time == ZIP_DATE_TIME,
        "compression": info.compress_type == ZIP_COMPRESSION,
        "create_system": info.create_system == ZIP_CREATE_SYSTEM,
        "create_version": info.create_version == ZIP_CREATE_VERSION,
        "extract_version": info.extract_version == ZIP_EXTRACT_VERSION,
        "flag_bits": info.flag_bits == ZIP_FLAG_BITS,
        "volume": info.volume == 0,
        "internal_attr": info.internal_attr == ZIP_INTERNAL_ATTR,
        "external_attr": info.external_attr == ZIP_EXTERNAL_ATTR,
        "extra": info.extra == b"",
        "comment": info.comment == b"",
        "stored_size": info.compress_size == info.file_size,
        "not_directory": not info.is_dir(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AtmosphereFixtureWriterError(
            f"noncanonical ZIP metadata for {expected_name}: {failed}"
        )


def _deserialize_canonical_mapping(
    archive_bytes: bytes,
    *,
    size_ceiling: int,
) -> dict[str, np.ndarray]:
    if not isinstance(archive_bytes, bytes):
        raise AtmosphereFixtureWriterError(
            "deterministic archive must be immutable bytes"
        )
    if not archive_bytes:
        raise AtmosphereFixtureWriterError("deterministic archive is empty")
    if len(archive_bytes) > size_ceiling:
        raise AtmosphereFixtureWriterError(
            f"deterministic archive exceeds the {size_ceiling}-byte ceiling"
        )

    loaded: dict[str, np.ndarray] = {}
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes), mode="r") as archive:
            if archive.comment != b"":
                raise AtmosphereFixtureWriterError(
                    "deterministic archive comment is not empty"
                )
            infos = archive.infolist()
            actual_names = [info.filename for info in infos]
            if actual_names != sorted(actual_names):
                raise AtmosphereFixtureWriterError(
                    "deterministic archive member order is not lexical"
                )
            if len(actual_names) != len(set(actual_names)):
                raise AtmosphereFixtureWriterError(
                    "deterministic archive has duplicate members"
                )
            if any(
                not member_name.endswith(".npy")
                or member_name == ".npy"
                or "/" in member_name.removesuffix(".npy")
                or "\\" in member_name
                or member_name.removesuffix(".npy") in {".", ".."}
                for member_name in actual_names
            ):
                raise AtmosphereFixtureWriterError(
                    "deterministic archive has an unsafe or path-like member name"
                )
            for info in infos:
                member_name = info.filename
                name = member_name.removesuffix(".npy")
                _validate_zip_info(info, member_name)
                payload = archive.read(info)
                if not payload.startswith(b"\x93NUMPY\x02\x00"):
                    raise AtmosphereFixtureWriterError(
                        f"NPY version changed for {name}"
                    )
                array = np.lib.format.read_array(
                    BytesIO(payload),
                    allow_pickle=False,
                )
                if array.dtype.hasobject:
                    raise AtmosphereFixtureWriterError(
                        f"object dtype entered serialized member: {name}"
                    )
                if not array.flags.c_contiguous:
                    raise AtmosphereFixtureWriterError(
                        f"serialized member is not C-contiguous: {name}"
                    )
                if payload != _npy_bytes(array):
                    raise AtmosphereFixtureWriterError(
                        f"serialized NPY bytes are noncanonical for {name}"
                    )
                loaded[name] = np.array(array, copy=True, order="C")
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise AtmosphereFixtureWriterError(
            f"invalid deterministic NPZ bytes: {error}"
        ) from error

    canonical = _serialize_mapping(loaded, size_ceiling=size_ceiling)
    if archive_bytes != canonical:
        raise AtmosphereFixtureWriterError(
            "archive bytes are not the unique canonical deterministic encoding"
        )
    return loaded


def _capture_transport_mapping(
    fixture: Mapping[str, np.ndarray],
    evidence: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        **{
            f"{_FIXTURE_TRANSPORT_PREFIX}{name}": np.asarray(value)
            for name, value in fixture.items()
        },
        **{
            f"{_EVIDENCE_TRANSPORT_PREFIX}{name}": np.asarray(value)
            for name, value in evidence.items()
        },
    }


def _split_capture_transport(
    values: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fixture: dict[str, np.ndarray] = {}
    evidence: dict[str, np.ndarray] = {}
    for name, value in values.items():
        if name.startswith(_FIXTURE_TRANSPORT_PREFIX):
            destination = fixture
            base_name = name.removeprefix(_FIXTURE_TRANSPORT_PREFIX)
        elif name.startswith(_EVIDENCE_TRANSPORT_PREFIX):
            destination = evidence
            base_name = name.removeprefix(_EVIDENCE_TRANSPORT_PREFIX)
        else:
            raise AtmosphereFixtureWriterError(
                f"capture transport contains an unscoped member: {name}"
            )
        if not base_name or base_name in destination:
            raise AtmosphereFixtureWriterError(
                "capture transport contains an ambiguous member"
            )
        destination[base_name] = np.array(value, copy=True, order="C")
    _validate_accepted_capture(fixture, evidence)
    return fixture, evidence


def _capture_in_fresh_child(
    cache_directory: Path,
    *,
    capture_label: str,
    base_environment: Mapping[str, str],
) -> _FreshAtmosphereCapture:
    cache = _validate_cache_directory(cache_directory, require_empty=True)
    cache_root_sha256 = sha256_bytes(str(cache).encode("utf-8"))
    capture_token = secrets.token_hex(32)
    capture_token_sha256 = sha256_bytes(capture_token.encode("ascii"))
    accepted_environment_json = json.dumps(
        worker.CAPTURE_ENVIRONMENT,
        sort_keys=True,
    )
    child_source = f"""
import json
import os
expected_environment = json.loads({accepted_environment_json!r})
wrong = {{
    name: os.environ.get(name)
    for name, expected in expected_environment.items()
    if os.environ.get(name) != expected
}}
if wrong:
    raise SystemExit(
        "pre-NumPy atmosphere capture controls changed: "
        + json.dumps(wrong, sort_keys=True)
    )

import hashlib
from pathlib import Path
import sys
from scripts import chapter06_atmosphere_fixture_writer as writer
from scripts import chapter06_atmosphere_fixture_worker as worker
writer.verify_accepted_identities()
capture = worker.build_fixture_capture()
writer._validate_accepted_capture(
    capture.fixture_arrays,
    capture.ephemeral_evidence,
)
transport = writer._serialize_mapping(
    writer._capture_transport_mapping(
        capture.fixture_arrays,
        capture.ephemeral_evidence,
    ),
    size_ceiling=writer.CAPTURE_TRANSPORT_SIZE_CEILING_BYTES,
)
cache = Path(os.environ["NUMBA_CACHE_DIR"]).expanduser().resolve()
header = json.dumps({{
    "cache_root_sha256": hashlib.sha256(
        str(cache).encode("utf-8")
    ).hexdigest(),
    "capture_token": os.environ[writer.CACHE_CAPTURE_TOKEN_ENV],
    "child_pid": os.getpid(),
    "capture_transport_bytes": len(transport),
    "capture_transport_sha256": hashlib.sha256(transport).hexdigest(),
}}, sort_keys=True, separators=(",", ":")).encode("utf-8")
sys.stdout.buffer.write(len(header).to_bytes(8, "big"))
sys.stdout.buffer.write(header)
sys.stdout.buffer.write(transport)
"""
    environment = dict(base_environment)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    environment["NUMBA_CACHE_DIR"] = str(cache)
    environment[CACHE_CAPTURE_TOKEN_ENV] = capture_token
    completed = subprocess.run(
        [sys.executable, "-c", child_source],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        timeout=240,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AtmosphereFixtureWriterError(
            f"fresh atmosphere capture {capture_label} child failed with "
            f"{completed.returncode}: {message}"
        )
    if completed.stderr:
        raise AtmosphereFixtureWriterError(
            f"fresh atmosphere capture {capture_label} child wrote stderr: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    cache_after = _validate_cache_directory(cache, require_empty=False)
    if cache_after != cache:
        raise AtmosphereFixtureWriterError(
            f"fresh atmosphere capture {capture_label} cache identity changed"
        )
    cache_entry_count_after = sum(1 for _ in cache.rglob("*"))

    payload = bytes(completed.stdout)
    if len(payload) < 8:
        raise AtmosphereFixtureWriterError(
            f"fresh atmosphere capture {capture_label} returned a truncated frame"
        )
    header_size = int.from_bytes(payload[:8], "big")
    if header_size <= 0 or header_size > 64 * 1024 or len(payload) < 8 + header_size:
        raise AtmosphereFixtureWriterError(
            f"fresh atmosphere capture {capture_label} returned an invalid header"
        )
    try:
        header = json.loads(payload[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AtmosphereFixtureWriterError(
            f"fresh atmosphere capture {capture_label} returned invalid evidence JSON"
        ) from error
    expected_header_names = {
        "cache_root_sha256",
        "capture_token",
        "child_pid",
        "capture_transport_bytes",
        "capture_transport_sha256",
    }
    if not isinstance(header, dict) or set(header) != expected_header_names:
        raise AtmosphereFixtureWriterError(
            f"fresh atmosphere capture {capture_label} evidence schema changed"
        )
    transport = payload[8 + header_size :]
    transport_hash = sha256_bytes(transport)
    if (
        header["capture_token"] != capture_token
        or header["cache_root_sha256"] != cache_root_sha256
        or not isinstance(header["child_pid"], int)
        or header["child_pid"] <= 0
        or header["capture_transport_bytes"] != len(transport)
        or header["capture_transport_sha256"] != transport_hash
    ):
        raise AtmosphereFixtureWriterError(
            f"fresh atmosphere capture {capture_label} origin evidence changed"
        )
    decoded = _deserialize_canonical_mapping(
        transport,
        size_ceiling=CAPTURE_TRANSPORT_SIZE_CEILING_BYTES,
    )
    fixture, evidence = _split_capture_transport(decoded)
    return _FreshAtmosphereCapture(
        fixture_arrays=fixture,
        ephemeral_evidence=evidence,
        capture_transport_bytes=transport,
        capture_transport_sha256=transport_hash,
        origin_token_sha256=capture_token_sha256,
        child_pid=int(header["child_pid"]),
        cache_root_sha256=cache_root_sha256,
        cache_empty_before=True,
        cache_entry_count_after=cache_entry_count_after,
        cache_external=True,
        cache_nonsymlink=True,
    )


def _require_independent_capture_pair(
    capture_a: _FreshAtmosphereCapture,
    capture_b: _FreshAtmosphereCapture,
) -> tuple[str, str]:
    if capture_a is capture_b:
        raise AtmosphereFixtureWriterError(
            "fresh atmosphere capture A/B reused one capture object"
        )
    if capture_a.origin_token_sha256 == capture_b.origin_token_sha256:
        raise AtmosphereFixtureWriterError(
            "fresh atmosphere capture A/B reused one origin token"
        )
    if capture_a.child_pid == capture_b.child_pid:
        raise AtmosphereFixtureWriterError(
            "fresh atmosphere capture A/B reused one child process"
        )
    if capture_a.cache_root_sha256 == capture_b.cache_root_sha256:
        raise AtmosphereFixtureWriterError(
            "fresh atmosphere capture A/B reused one cache root"
        )
    for label, capture in (("A", capture_a), ("B", capture_b)):
        if not (
            capture.cache_empty_before
            and capture.cache_external
            and capture.cache_nonsymlink
        ):
            raise AtmosphereFixtureWriterError(
                f"fresh atmosphere capture {label} cache evidence is incomplete"
            )
        if capture.cache_entry_count_after < 0:
            raise AtmosphereFixtureWriterError(
                f"fresh atmosphere capture {label} cache entry count is invalid"
            )
        if capture.capture_transport_sha256 != sha256_bytes(
            capture.capture_transport_bytes
        ):
            raise AtmosphereFixtureWriterError(
                f"fresh atmosphere capture {label} transport identity changed"
            )
        _validate_accepted_capture(
            capture.fixture_arrays,
            capture.ephemeral_evidence,
        )
    if capture_a.capture_transport_bytes != capture_b.capture_transport_bytes:
        raise AtmosphereFixtureWriterError(
            "complete accepted atmosphere capture A/B bytes disagree"
        )
    fixture_digest = _require_mapping_pair_identity(
        capture_a.fixture_arrays,
        capture_b.fixture_arrays,
        label="nineteen-member fixture",
    )
    evidence_digest = _require_mapping_pair_identity(
        capture_a.ephemeral_evidence,
        capture_b.ephemeral_evidence,
        label="complete worker evidence",
    )
    return fixture_digest, evidence_digest


def _decode_and_validate_fixture_archive(
    archive_bytes: bytes,
    expected: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Decode final bytes and require exact accepted fixture semantics."""

    loaded = _deserialize_canonical_mapping(
        archive_bytes,
        size_ceiling=FIXTURE_ARCHIVE_SIZE_CEILING_BYTES,
    )
    if tuple(loaded) != tuple(sorted(worker.FIXTURE_SCHEMA)):
        raise AtmosphereFixtureWriterError(
            "serialized atmosphere fixture member set or order changed"
        )
    try:
        worker._validate_fixture_mapping(loaded)
    except (RuntimeError, TypeError, ValueError) as error:
        raise AtmosphereFixtureWriterError(
            f"serialized atmosphere fixture validation failed: {error}"
        ) from error
    if tuple(sorted(expected)) != tuple(loaded):
        raise AtmosphereFixtureWriterError(
            "serialized atmosphere fixture disagrees with accepted capture members"
        )
    for name in loaded:
        if not _bitwise_equal(loaded[name], expected[name]):
            raise AtmosphereFixtureWriterError(
                f"serialized atmosphere fixture payload changed: {name}"
            )
    if worker.mapping_schema_digest(loaded) != ACCEPTED_FIXTURE_SCHEMA_DIGEST:
        raise AtmosphereFixtureWriterError(
            "serialized atmosphere fixture schema digest changed"
        )
    return loaded


def build_deterministic_atmosphere_fixture_archive() -> (
    DeterministicAtmosphereFixtureArchive
):
    """Build canonical fixture bytes from two builder-owned fresh captures."""

    base_environment = _require_nonconflicting_capture_environment()
    identities = verify_accepted_identities()
    cache_parent = _validate_cache_directory(
        Path(tempfile.gettempdir()),
        require_empty=False,
    )
    with (
        tempfile.TemporaryDirectory(
            prefix="chapter06-atmosphere-capture-a-",
            dir=cache_parent,
        ) as cache_a_text,
        tempfile.TemporaryDirectory(
            prefix="chapter06-atmosphere-capture-b-",
            dir=cache_parent,
        ) as cache_b_text,
    ):
        cache_a = _validate_cache_directory(Path(cache_a_text), require_empty=True)
        cache_b = _validate_cache_directory(Path(cache_b_text), require_empty=True)
        _require_distinct_cache_roots(cache_a, cache_b)
        capture_a = _capture_in_fresh_child(
            cache_a,
            capture_label="A",
            base_environment=base_environment,
        )
        capture_b = _capture_in_fresh_child(
            cache_b,
            capture_label="B",
            base_environment=base_environment,
        )
        fixture_digest, evidence_digest = _require_independent_capture_pair(
            capture_a,
            capture_b,
        )
        process_evidence = {
            "capture_a_origin_token_sha256": capture_a.origin_token_sha256,
            "capture_b_origin_token_sha256": capture_b.origin_token_sha256,
            "capture_a_child_pid": capture_a.child_pid,
            "capture_b_child_pid": capture_b.child_pid,
            "capture_a_cache_root_sha256": capture_a.cache_root_sha256,
            "capture_b_cache_root_sha256": capture_b.cache_root_sha256,
            "capture_origins_distinct": True,
            "capture_child_processes_distinct": True,
            "capture_cache_roots_distinct": True,
            "capture_cache_roots_external": True,
            "capture_cache_roots_nonsymlink": True,
            "capture_cache_roots_empty_before": True,
            "capture_a_cache_entry_count_after": (capture_a.cache_entry_count_after),
            "capture_b_cache_entry_count_after": (capture_b.cache_entry_count_after),
        }
        transport_a = capture_a.capture_transport_bytes
        transport_b = capture_b.capture_transport_bytes
        fixture_a = capture_a.fixture_arrays
        fixture_b = capture_b.fixture_arrays

    if Path(cache_a_text).exists() or Path(cache_b_text).exists():
        raise AtmosphereFixtureWriterError(
            "disposable atmosphere capture cache cleanup failed"
        )
    process_evidence["capture_cache_roots_disposed"] = True

    archive_a = _serialize_mapping(
        fixture_a,
        size_ceiling=FIXTURE_ARCHIVE_SIZE_CEILING_BYTES,
    )
    archive_b = _serialize_mapping(
        fixture_b,
        size_ceiling=FIXTURE_ARCHIVE_SIZE_CEILING_BYTES,
    )
    _decode_and_validate_fixture_archive(archive_a, fixture_a)
    _decode_and_validate_fixture_archive(archive_b, fixture_b)
    if archive_a != archive_b:
        raise AtmosphereFixtureWriterError(
            "capture A/B produced different final atmosphere NPZ bytes"
        )

    archive_hash = sha256_bytes(archive_a)
    process_evidence_digest = sha256_bytes(
        json.dumps(
            process_evidence,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    summary = {
        "accepted_identity_count": len(identities),
        "writer_sha256": _file_sha256(WRITER_PATH),
        "fixture_a_member_count": len(fixture_a),
        "fixture_b_member_count": len(fixture_b),
        "fixture_a_b_bitwise_equal": True,
        "fixture_a_b_mapping_digest": fixture_digest,
        "evidence_a_member_count": ACCEPTED_EVIDENCE_MEMBER_COUNT,
        "evidence_b_member_count": ACCEPTED_EVIDENCE_MEMBER_COUNT,
        "evidence_a_b_bitwise_equal": True,
        "evidence_a_b_mapping_digest": evidence_digest,
        "capture_a_transport_sha256": sha256_bytes(transport_a),
        "capture_b_transport_sha256": sha256_bytes(transport_b),
        "capture_a_b_transport_byte_equal": True,
        "capture_transport_bytes": len(transport_a),
        "fixture_schema_digest": ACCEPTED_FIXTURE_SCHEMA_DIGEST,
        "payload_fingerprint": ACCEPTED_PAYLOAD_FINGERPRINT,
        "full_capture_schema_digest": ACCEPTED_FULL_CAPTURE_SCHEMA_DIGEST,
        "full_capture_fingerprint": ACCEPTED_FULL_CAPTURE_FINGERPRINT,
        "archive_a_sha256": archive_hash,
        "archive_b_sha256": archive_hash,
        "archive_a_b_byte_equal": True,
        "archive_bytes": len(archive_a),
        "archive_array_bytes": sum(
            np.asarray(value).nbytes for value in fixture_a.values()
        ),
        "archive_member_count": ACCEPTED_FIXTURE_MEMBER_COUNT,
        "archive_member_order": "lexical",
        "archive_compression": "ZIP_STORED",
        "npy_version": list(NPY_VERSION),
        "fixed_zip_date_time": list(ZIP_DATE_TIME),
        "capture_process_evidence": process_evidence,
        "capture_process_evidence_sha256": process_evidence_digest,
        "disposable_cache_directories_created": True,
        "publication_authorized": False,
        "fixture_publication_performed": False,
        "golden_publication_performed": False,
        "manifest_mutation_performed": False,
        "artifact_file_write_performed": False,
    }
    return DeterministicAtmosphereFixtureArchive(
        archive_bytes=bytes(archive_a),
        summary=summary,
    )


__all__ = [
    "ACCEPTED_EVIDENCE_MEMBER_COUNT",
    "ACCEPTED_FILE_IDENTITIES",
    "ACCEPTED_FIXTURE_MEMBER_COUNT",
    "ACCEPTED_FIXTURE_SCHEMA_DIGEST",
    "ACCEPTED_FULL_CAPTURE_FINGERPRINT",
    "ACCEPTED_FULL_CAPTURE_SCHEMA_DIGEST",
    "ACCEPTED_PAYLOAD_FINGERPRINT",
    "AtmosphereFixtureWriterError",
    "DeterministicAtmosphereFixtureArchive",
    "FIXTURE_ARCHIVE_SIZE_CEILING_BYTES",
    "NPY_VERSION",
    "ZIP_COMPRESSION",
    "ZIP_CREATE_SYSTEM",
    "ZIP_DATE_TIME",
    "ZIP_EXTERNAL_ATTR",
    "build_deterministic_atmosphere_fixture_archive",
    "sha256_bytes",
    "verify_accepted_identities",
]
