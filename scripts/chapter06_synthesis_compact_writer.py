"""Deterministic in-memory NPZ bytes for the accepted Chapter 6 synthesis candidate.

This module is the serialization gate immediately after the independently
accepted compact assembler.  It binds the exact accepted worker/assembler
review snapshot, requires two accepted raw observations and two independently
validated compact assemblies, and returns one canonical byte string plus a
JSON-safe summary.

It has no output path, filesystem writer, manifest mutation, authorization
parser, publisher, canonical archive, command-line entry point, or overwrite
operation.
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

from scripts import chapter06_synthesis_compact_assembler as compact
from scripts import chapter06_synthesis_oracle_worker as oracle_worker


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRITER_PATH = Path(__file__).resolve()
PINNED_ROOT = Path("/Users/ysting/payne-zero")
PAPER_ROOT = Path("/Users/ysting/Source_Files_Not_For_Review")
FORBIDDEN_CACHE_ROOTS = (REPOSITORY_ROOT, PINNED_ROOT, PAPER_ROOT)
CACHE_CAPTURE_TOKEN_ENV = "CHAPTER06_SYNTHESIS_CAPTURE_TOKEN"

ACCEPTED_FILE_IDENTITIES = {
    "fixture_oracle_plan": (
        REPOSITORY_ROOT / "design/chapter06_synthesis_fixture_oracle_plan.md",
        "413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856",
    ),
    "scientific_worker": (
        REPOSITORY_ROOT / "scripts/chapter06_synthesis_oracle_worker.py",
        "36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68",
    ),
    "scientific_worker_tests": (
        REPOSITORY_ROOT / "tests/test_chapter06_synthesis_oracle_worker.py",
        "1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189",
    ),
    "scientific_worker_independent_audit": (
        REPOSITORY_ROOT / "design/chapter06_synthesis_worker_independent_audit.md",
        "a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334",
    ),
    "plan_rebind_candidate": (
        REPOSITORY_ROOT / "design/chapter06_synthesis_plan_rebind_candidate.md",
        "dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93",
    ),
    "plan_rebind_independent_audit": (
        REPOSITORY_ROOT / "design/chapter06_synthesis_plan_rebind_independent_audit.md",
        "9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7",
    ),
    "compact_assembler": (
        REPOSITORY_ROOT / "scripts/chapter06_synthesis_compact_assembler.py",
        "583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8",
    ),
    "compact_assembler_tests": (
        REPOSITORY_ROOT / "tests/test_chapter06_synthesis_compact_assembler.py",
        "25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153",
    ),
    "compact_rebind_candidate": (
        REPOSITORY_ROOT / "design/chapter06_synthesis_compact_rebind_candidate.md",
        "54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c",
    ),
    "compact_rebind_independent_audit": (
        REPOSITORY_ROOT
        / "design/chapter06_synthesis_compact_rebind_independent_audit.md",
        "739854db2b5c4c0c0fe5e9db71d8a52958ce401ded7e7a80a8ab90e15172ddcb",
    ),
}

ACCEPTED_COMPACT_KEY_COUNT = 213
ACCEPTED_COMPACT_ARRAY_BYTES = 1_235_275
ACCEPTED_COMPACT_SCHEMA_DIGEST = (
    "911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde"
)
ACCEPTED_COMPACT_PAYLOAD_FINGERPRINT = (
    "e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b"
)
ACCEPTED_RAW_OWNERSHIP_DIGEST = (
    "5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675"
)
ACCEPTED_RAW_OWNERSHIP_COUNTS = {
    "derived_digest_only": 123,
    "final": 250,
    "intentionally_ephemeral": 381,
}

NPY_VERSION = (2, 0)
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
ZIP_CREATE_SYSTEM = 3
ZIP_CREATE_VERSION = 20
ZIP_EXTRACT_VERSION = 20
ZIP_EXTERNAL_ATTR = 0o100600 << 16
ZIP_INTERNAL_ATTR = 0
ZIP_FLAG_BITS = 0
ZIP_COMPRESSION = zipfile.ZIP_STORED


class CompactWriterError(RuntimeError):
    """Raised when accepted identities, A/B parity, or archive bytes drift."""


@dataclass(frozen=True)
class DeterministicCompactArchive:
    """One immutable canonical byte string and its JSON-safe review summary."""

    archive_bytes: bytes
    summary: dict[str, Any]


@dataclass(frozen=True)
class _FreshRawCapture:
    """One child-owned raw observation plus non-scientific origin evidence."""

    raw_mapping: dict[str, np.ndarray]
    raw_archive_bytes: bytes
    raw_archive_sha256: str
    origin_token_sha256: str
    child_pid: int
    cache_root_sha256: str
    cache_empty_before: bool
    cache_empty_after: bool
    cache_external: bool
    cache_nonsymlink: bool


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 identity of one immutable byte string."""

    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_regular_nonsymlink(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise CompactWriterError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise CompactWriterError(f"{label} is missing: {candidate}")
    return resolved


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_cache_directory(
    path: Path,
    *,
    require_empty: bool,
) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise CompactWriterError(
            f"capture cache must be an existing nonsymlink directory: {candidate}"
        )
    resolved = candidate.resolve()
    if any(_is_under(resolved, root) for root in FORBIDDEN_CACHE_ROOTS):
        raise CompactWriterError(
            f"capture cache must be external to all source/data trees: {resolved}"
        )
    if require_empty and any(resolved.iterdir()):
        raise CompactWriterError(f"capture cache must be empty: {resolved}")
    return resolved


def _require_distinct_cache_roots(cache_a: Path, cache_b: Path) -> None:
    if cache_a.resolve() == cache_b.resolve():
        raise CompactWriterError("fresh capture A/B cache roots must be distinct")


def verify_accepted_identities() -> dict[str, str]:
    """Require the exact independently accepted worker/assembler snapshot."""

    identities: dict[str, str] = {}
    for label, (path, expected_hash) in sorted(ACCEPTED_FILE_IDENTITIES.items()):
        resolved = _require_regular_nonsymlink(path, label)
        actual_hash = _file_sha256(resolved)
        if actual_hash != expected_hash:
            raise CompactWriterError(
                f"accepted identity changed for {label}: "
                f"{actual_hash}; expected {expected_hash}"
            )
        identities[label] = actual_hash
    if (
        compact.ASSEMBLER_PATH.resolve()
        != ACCEPTED_FILE_IDENTITIES["compact_assembler"][0].resolve()
    ):
        raise CompactWriterError("compact assembler imported from an unexpected path")
    if (
        oracle_worker.WORKER_PATH.resolve()
        != ACCEPTED_FILE_IDENTITIES["scientific_worker"][0].resolve()
    ):
        raise CompactWriterError("scientific worker imported from an unexpected path")
    return identities


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in arrays:
        raise CompactWriterError(f"accepted compact scalar is missing: {name}")
    value = np.asarray(arrays[name])
    if value.shape != ():
        raise CompactWriterError(f"accepted compact scalar is not scalar: {name}")
    return value.item()


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
            raise CompactWriterError(f"object dtype is forbidden: {name}")
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _validate_accepted_assembly(assembly: compact.CompactAssembly) -> None:
    try:
        compact.validate_compact_candidate(assembly)
    except compact.CompactAssemblyError as error:
        raise CompactWriterError(
            f"compact assembly validation failed: {error}"
        ) from error

    arrays = assembly.arrays
    exact_scalars = {
        "meta__compact_key_count": ACCEPTED_COMPACT_KEY_COUNT,
        "meta__compact_schema_digest": ACCEPTED_COMPACT_SCHEMA_DIGEST,
        "meta__compact_payload_fingerprint": (ACCEPTED_COMPACT_PAYLOAD_FINGERPRINT),
        "meta__raw_ownership_digest": ACCEPTED_RAW_OWNERSHIP_DIGEST,
    }
    for name, expected in exact_scalars.items():
        if _scalar(arrays, name) != expected:
            raise CompactWriterError(f"accepted compact identity changed: {name}")
    if len(arrays) != ACCEPTED_COMPACT_KEY_COUNT:
        raise CompactWriterError("accepted compact member count changed")
    array_bytes = sum(np.asarray(value).nbytes for value in arrays.values())
    if array_bytes != ACCEPTED_COMPACT_ARRAY_BYTES:
        raise CompactWriterError(f"accepted compact array bytes changed: {array_bytes}")
    if compact.schema_digest(arrays) != ACCEPTED_COMPACT_SCHEMA_DIGEST:
        raise CompactWriterError("accepted compact schema digest changed")
    if compact.payload_fingerprint(arrays) != ACCEPTED_COMPACT_PAYLOAD_FINGERPRINT:
        raise CompactWriterError("accepted compact payload fingerprint changed")
    if compact.ownership_digest(assembly.raw_ownership) != (
        ACCEPTED_RAW_OWNERSHIP_DIGEST
    ):
        raise CompactWriterError("accepted raw ownership digest changed")
    actual_counts = {
        disposition: sum(
            item.disposition == disposition for item in assembly.raw_ownership
        )
        for disposition in sorted(compact.RAW_DISPOSITIONS)
    }
    if actual_counts != ACCEPTED_RAW_OWNERSHIP_COUNTS:
        raise CompactWriterError(
            f"accepted raw ownership counts changed: {actual_counts}"
        )


def _require_raw_pair_identity(
    raw_a: Mapping[str, np.ndarray],
    raw_b: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    first = oracle_worker.deterministic_result(raw_a)
    second = oracle_worker.deterministic_result(raw_b)
    if tuple(first) != tuple(second):
        raise CompactWriterError("raw A/B member sets or lexical order disagree")
    for name in first:
        if not _bitwise_equal(first[name], second[name]):
            raise CompactWriterError(f"raw A/B payload disagrees: {name}")
    first_digest = _mapping_digest(first)
    second_digest = _mapping_digest(second)
    if first_digest != second_digest:
        raise CompactWriterError("raw A/B mapping digests disagree")
    return first, second, first_digest


def _require_assembly_pair_identity(
    assembly_a: compact.CompactAssembly,
    assembly_b: compact.CompactAssembly,
) -> None:
    if assembly_a.schema != assembly_b.schema:
        raise CompactWriterError("compact A/B schema descriptions disagree")
    if assembly_a.raw_ownership != assembly_b.raw_ownership:
        raise CompactWriterError("compact A/B raw ownership disagrees")
    if tuple(assembly_a.arrays) != tuple(assembly_b.arrays):
        raise CompactWriterError("compact A/B member sets or lexical order disagree")
    for name in assembly_a.arrays:
        if not _bitwise_equal(assembly_a.arrays[name], assembly_b.arrays[name]):
            raise CompactWriterError(f"compact A/B payload disagrees: {name}")


def _npy_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise CompactWriterError("object dtype is forbidden in deterministic NPZ")
    if not array.flags.c_contiguous:
        raise CompactWriterError("non-C-contiguous arrays are forbidden")
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


def _serialize_mapping(
    values: Mapping[str, np.ndarray],
    *,
    size_ceiling: int | None,
) -> bytes:
    names = sorted(values)
    if len(names) != len(set(names)):
        raise CompactWriterError("deterministic mapping has duplicate names")
    if any(
        not isinstance(name, str)
        or not name
        or "\\" in name
        or name.endswith(".npy")
        or any(part in {"", ".", ".."} for part in name.split("/"))
        for name in names
    ):
        raise CompactWriterError("deterministic mapping has an unsafe member name")

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
    if size_ceiling is not None and len(result) > size_ceiling:
        raise CompactWriterError(
            f"deterministic archive is {len(result)} bytes; ceiling is {size_ceiling}"
        )
    return result


def _serialize_accepted_assembly(assembly: compact.CompactAssembly) -> bytes:
    _validate_accepted_assembly(assembly)
    return _serialize_mapping(
        assembly.arrays,
        size_ceiling=compact.COMPACT_SIZE_CEILING_BYTES,
    )


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
        raise CompactWriterError(
            f"noncanonical ZIP metadata for {expected_name}: {failed}"
        )


def _deserialize_canonical_mapping(
    archive_bytes: bytes,
    *,
    size_ceiling: int | None,
) -> dict[str, np.ndarray]:
    if not isinstance(archive_bytes, bytes):
        raise CompactWriterError("deterministic archive must be immutable bytes")
    if not archive_bytes:
        raise CompactWriterError("deterministic archive is empty")
    if size_ceiling is not None and len(archive_bytes) > size_ceiling:
        raise CompactWriterError(
            f"deterministic archive exceeds the {size_ceiling}-byte ceiling"
        )

    loaded: dict[str, np.ndarray] = {}
    try:
        with zipfile.ZipFile(BytesIO(archive_bytes), mode="r") as archive:
            if archive.comment != b"":
                raise CompactWriterError("deterministic archive comment is not empty")
            infos = archive.infolist()
            actual_names = [info.filename for info in infos]
            if actual_names != sorted(actual_names):
                raise CompactWriterError(
                    "deterministic archive member order is not lexical"
                )
            if len(actual_names) != len(set(actual_names)):
                raise CompactWriterError("deterministic archive has duplicate members")
            if any(
                not member_name.endswith(".npy")
                or "\\" in member_name
                or member_name == ".npy"
                or any(
                    part in {"", ".", ".."}
                    for part in member_name.removesuffix(".npy").split("/")
                )
                for member_name in actual_names
            ):
                raise CompactWriterError(
                    "deterministic archive has an unsafe member name"
                )
            for info in infos:
                member_name = info.filename
                name = member_name.removesuffix(".npy")
                _validate_zip_info(info, member_name)
                payload = archive.read(info)
                if not payload.startswith(b"\x93NUMPY\x02\x00"):
                    raise CompactWriterError(f"NPY version changed for {name}")
                array = np.lib.format.read_array(
                    BytesIO(payload),
                    allow_pickle=False,
                )
                if array.dtype.hasobject:
                    raise CompactWriterError(
                        f"object dtype entered serialized member: {name}"
                    )
                if not array.flags.c_contiguous:
                    raise CompactWriterError(
                        f"serialized member is not C-contiguous: {name}"
                    )
                canonical_payload = _npy_bytes(array)
                if payload != canonical_payload:
                    raise CompactWriterError(
                        f"serialized NPY bytes are noncanonical for {name}"
                    )
                loaded[name] = np.array(array, copy=True, order="C")
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise CompactWriterError(f"invalid deterministic NPZ bytes: {error}") from error

    canonical_archive = _serialize_mapping(loaded, size_ceiling=size_ceiling)
    if archive_bytes != canonical_archive:
        raise CompactWriterError(
            "archive bytes are not the unique canonical deterministic encoding"
        )
    return loaded


def _deserialize_and_validate(
    archive_bytes: bytes,
    expected: compact.CompactAssembly,
) -> dict[str, np.ndarray]:
    _validate_accepted_assembly(expected)
    loaded = _deserialize_canonical_mapping(
        archive_bytes,
        size_ceiling=compact.COMPACT_SIZE_CEILING_BYTES,
    )
    if tuple(loaded) != tuple(expected.arrays):
        raise CompactWriterError("serialized compact member set changed")
    for name in expected.arrays:
        if not _bitwise_equal(loaded[name], expected.arrays[name]):
            raise CompactWriterError(
                f"serialized dtype/shape/C-bytes changed for {name}"
            )
    reconstructed = compact.CompactAssembly(
        arrays=loaded,
        schema=expected.schema,
        raw_ownership=expected.raw_ownership,
    )
    _validate_accepted_assembly(reconstructed)
    return loaded


def _serialize_ephemeral_raw_mapping(raw: Mapping[str, np.ndarray]) -> bytes:
    deterministic = oracle_worker.deterministic_result(raw)
    return _serialize_mapping(deterministic, size_ceiling=None)


def _capture_raw_in_fresh_child(
    cache_directory: Path,
    *,
    capture_label: str,
) -> _FreshRawCapture:
    cache = _validate_cache_directory(cache_directory, require_empty=True)
    cache_root_sha256 = sha256_bytes(str(cache).encode("utf-8"))
    capture_token = secrets.token_hex(32)
    capture_token_sha256 = sha256_bytes(capture_token.encode("ascii"))
    child_source = """
import hashlib
import json
import os
from pathlib import Path
import sys
from scripts import chapter06_synthesis_compact_writer as writer
from scripts import chapter06_synthesis_oracle_worker as worker
writer.verify_accepted_identities()
raw = worker.build_oracle_results()
raw_bytes = writer._serialize_ephemeral_raw_mapping(raw)
cache = Path(os.environ["NUMBA_CACHE_DIR"]).expanduser().resolve()
header = json.dumps({
    "cache_root_sha256": hashlib.sha256(
        str(cache).encode("utf-8")
    ).hexdigest(),
    "capture_token": os.environ[writer.CACHE_CAPTURE_TOKEN_ENV],
    "child_pid": os.getpid(),
    "raw_archive_bytes": len(raw_bytes),
    "raw_archive_sha256": hashlib.sha256(raw_bytes).hexdigest(),
}, sort_keys=True, separators=(",", ":")).encode("utf-8")
sys.stdout.buffer.write(len(header).to_bytes(8, "big"))
sys.stdout.buffer.write(header)
sys.stdout.buffer.write(raw_bytes)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    environment["NUMBA_CACHE_DIR"] = str(cache)
    environment[CACHE_CAPTURE_TOKEN_ENV] = capture_token
    completed = subprocess.run(
        [sys.executable, "-c", child_source],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CompactWriterError(
            f"fresh raw-capture child failed with {completed.returncode}: {message}"
        )
    if completed.stderr:
        raise CompactWriterError(
            "fresh raw-capture child wrote unexpected stderr: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    cache_after = _validate_cache_directory(cache, require_empty=True)
    if cache_after != cache:
        raise CompactWriterError(
            f"fresh capture {capture_label} cache identity changed"
        )

    payload = bytes(completed.stdout)
    if len(payload) < 8:
        raise CompactWriterError(
            f"fresh capture {capture_label} returned a truncated evidence frame"
        )
    header_size = int.from_bytes(payload[:8], "big")
    if header_size <= 0 or header_size > 64 * 1024 or len(payload) < 8 + header_size:
        raise CompactWriterError(
            f"fresh capture {capture_label} returned an invalid evidence header"
        )
    try:
        header = json.loads(payload[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompactWriterError(
            f"fresh capture {capture_label} returned invalid evidence JSON"
        ) from error
    expected_header_names = {
        "cache_root_sha256",
        "capture_token",
        "child_pid",
        "raw_archive_bytes",
        "raw_archive_sha256",
    }
    if not isinstance(header, dict) or set(header) != expected_header_names:
        raise CompactWriterError(
            f"fresh capture {capture_label} evidence schema changed"
        )
    raw_bytes = payload[8 + header_size :]
    raw_hash = sha256_bytes(raw_bytes)
    if (
        header["capture_token"] != capture_token
        or header["cache_root_sha256"] != cache_root_sha256
        or not isinstance(header["child_pid"], int)
        or header["child_pid"] <= 0
        or header["raw_archive_bytes"] != len(raw_bytes)
        or header["raw_archive_sha256"] != raw_hash
    ):
        raise CompactWriterError(
            f"fresh capture {capture_label} origin evidence changed"
        )
    raw = _deserialize_canonical_mapping(raw_bytes, size_ceiling=None)
    return _FreshRawCapture(
        raw_mapping=raw,
        raw_archive_bytes=raw_bytes,
        raw_archive_sha256=raw_hash,
        origin_token_sha256=capture_token_sha256,
        child_pid=int(header["child_pid"]),
        cache_root_sha256=cache_root_sha256,
        cache_empty_before=True,
        cache_empty_after=True,
        cache_external=True,
        cache_nonsymlink=True,
    )


def _require_independent_capture_pair(
    capture_a: _FreshRawCapture,
    capture_b: _FreshRawCapture,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    if capture_a is capture_b:
        raise CompactWriterError("fresh capture A/B reused one capture object")
    if capture_a.origin_token_sha256 == capture_b.origin_token_sha256:
        raise CompactWriterError("fresh capture A/B reused one origin token")
    if capture_a.child_pid == capture_b.child_pid:
        raise CompactWriterError("fresh capture A/B reused one child process")
    if capture_a.cache_root_sha256 == capture_b.cache_root_sha256:
        raise CompactWriterError("fresh capture A/B reused one cache root")
    for label, capture in (("A", capture_a), ("B", capture_b)):
        if not (
            capture.cache_empty_before
            and capture.cache_empty_after
            and capture.cache_external
            and capture.cache_nonsymlink
        ):
            raise CompactWriterError(
                f"fresh capture {label} cache policy evidence is incomplete"
            )
        if capture.raw_archive_sha256 != sha256_bytes(capture.raw_archive_bytes):
            raise CompactWriterError(f"fresh capture {label} raw byte identity changed")
    if capture_a.raw_archive_bytes != capture_b.raw_archive_bytes:
        raise CompactWriterError("raw A/raw B deterministic bytes disagree")
    return _require_raw_pair_identity(
        capture_a.raw_mapping,
        capture_b.raw_mapping,
    )


def build_deterministic_compact_archive() -> DeterministicCompactArchive:
    """Build and prove one canonical archive from two accepted raw observations."""

    identities = verify_accepted_identities()
    cache_parent = _validate_cache_directory(
        Path(tempfile.gettempdir()),
        require_empty=False,
    )
    with (
        tempfile.TemporaryDirectory(
            prefix="chapter06-synthesis-capture-a-",
            dir=cache_parent,
        ) as cache_a_text,
        tempfile.TemporaryDirectory(
            prefix="chapter06-synthesis-capture-b-",
            dir=cache_parent,
        ) as cache_b_text,
    ):
        cache_a = _validate_cache_directory(Path(cache_a_text), require_empty=True)
        cache_b = _validate_cache_directory(Path(cache_b_text), require_empty=True)
        _require_distinct_cache_roots(cache_a, cache_b)
        capture_a = _capture_raw_in_fresh_child(
            cache_a,
            capture_label="A",
        )
        capture_b = _capture_raw_in_fresh_child(
            cache_b,
            capture_label="B",
        )
        raw_first, raw_second, raw_pair_digest = _require_independent_capture_pair(
            capture_a, capture_b
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
            "capture_cache_roots_empty_after": True,
        }
        raw_archive_a = capture_a.raw_archive_bytes
        raw_archive_b = capture_b.raw_archive_bytes

    if Path(cache_a_text).exists() or Path(cache_b_text).exists():
        raise CompactWriterError("disposable capture cache cleanup failed")
    process_evidence["capture_cache_roots_disposed"] = True

    try:
        assembly_a = compact.assemble_compact_candidate(raw_first)
        assembly_b = compact.assemble_compact_candidate(raw_second)
    except compact.CompactAssemblyError as error:
        raise CompactWriterError(f"raw capture was not accepted: {error}") from error
    _validate_accepted_assembly(assembly_a)
    _validate_accepted_assembly(assembly_b)
    _require_assembly_pair_identity(assembly_a, assembly_b)

    archive_a = _serialize_accepted_assembly(assembly_a)
    archive_b = _serialize_accepted_assembly(assembly_b)
    _deserialize_and_validate(archive_a, assembly_a)
    _deserialize_and_validate(archive_b, assembly_b)
    if archive_a != archive_b:
        raise CompactWriterError("raw A/raw B produced different final NPZ bytes")

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
        "raw_a_key_count": len(raw_first),
        "raw_b_key_count": len(raw_second),
        "raw_a_b_bitwise_equal": True,
        "raw_a_b_mapping_digest": raw_pair_digest,
        "raw_a_archive_sha256": sha256_bytes(raw_archive_a),
        "raw_b_archive_sha256": sha256_bytes(raw_archive_b),
        "raw_a_b_archive_byte_equal": True,
        "raw_archive_bytes": len(raw_archive_a),
        "raw_schema_digest": compact.ACCEPTED_RAW_SCHEMA_DIGEST,
        "raw_physical_fingerprint": compact.ACCEPTED_RAW_PHYSICAL_FINGERPRINT,
        "raw_full_fingerprint": compact.ACCEPTED_RAW_FULL_FINGERPRINT,
        "compact_a_b_schema_equal": True,
        "compact_a_b_payload_equal": True,
        "compact_a_b_ownership_equal": True,
        "compact_key_count": len(assembly_a.arrays),
        "compact_array_bytes": sum(
            np.asarray(value).nbytes for value in assembly_a.arrays.values()
        ),
        "compact_schema_digest": ACCEPTED_COMPACT_SCHEMA_DIGEST,
        "compact_payload_fingerprint": ACCEPTED_COMPACT_PAYLOAD_FINGERPRINT,
        "raw_ownership_digest": ACCEPTED_RAW_OWNERSHIP_DIGEST,
        "raw_ownership_counts": dict(ACCEPTED_RAW_OWNERSHIP_COUNTS),
        "archive_a_sha256": archive_hash,
        "archive_b_sha256": archive_hash,
        "archive_a_b_byte_equal": True,
        "archive_bytes": len(archive_a),
        "archive_member_count": ACCEPTED_COMPACT_KEY_COUNT,
        "archive_member_order": "lexical",
        "archive_compression": "ZIP_STORED",
        "npy_version": list(NPY_VERSION),
        "fixed_zip_date_time": list(ZIP_DATE_TIME),
        "capture_process_evidence": process_evidence,
        "capture_process_evidence_sha256": process_evidence_digest,
        "disposable_cache_directories_created": True,
        "publication_authorized": False,
        "golden_publication_performed": False,
        "manifest_mutation_performed": False,
        "artifact_file_write_performed": False,
    }
    return DeterministicCompactArchive(
        archive_bytes=bytes(archive_a),
        summary=summary,
    )


__all__ = [
    "ACCEPTED_COMPACT_ARRAY_BYTES",
    "ACCEPTED_COMPACT_KEY_COUNT",
    "ACCEPTED_COMPACT_PAYLOAD_FINGERPRINT",
    "ACCEPTED_COMPACT_SCHEMA_DIGEST",
    "ACCEPTED_FILE_IDENTITIES",
    "ACCEPTED_RAW_OWNERSHIP_COUNTS",
    "ACCEPTED_RAW_OWNERSHIP_DIGEST",
    "CompactWriterError",
    "DeterministicCompactArchive",
    "NPY_VERSION",
    "ZIP_COMPRESSION",
    "ZIP_CREATE_SYSTEM",
    "ZIP_DATE_TIME",
    "ZIP_EXTERNAL_ATTR",
    "build_deterministic_compact_archive",
    "sha256_bytes",
    "verify_accepted_identities",
]
