"""Adversarial tests for the fixed Chapter 6 atmosphere-fixture publisher."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Iterator

import pytest

from scripts import build_chapter06_atmosphere_fixture as publisher


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
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


def _write(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)


def _minimal_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
        "entries": [],
    }


def _candidate(
    payload: bytes = b"isolated-atmosphere-candidate",
) -> publisher.Candidate:
    return publisher.Candidate(
        archive_bytes=payload,
        arrays={},
        stable_writer_summary={"accepted": True},
        topology_summary={
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
        },
    )


def _runtime_paths(root: Path) -> publisher.RuntimePaths:
    return publisher.RuntimePaths(
        repository_root=root,
        data_root=root / "data",
        manifest=root / publisher.MANIFEST_IDENTITY,
        destination=root / publisher.DESTINATION_IDENTITY,
        authorization=root / publisher.AUTHORIZATION_IDENTITY,
        record_review=root / publisher.RECORD_REVIEW_IDENTITY,
        publisher_acceptance=root / publisher.PUBLISHER_ACCEPTANCE_IDENTITY,
    )


def _initialize_isolated_tree(root: Path) -> publisher.RuntimePaths:
    paths = _runtime_paths(root)
    (root / "data/fixtures").mkdir(parents=True)
    _write(root / publisher.README_IDENTITY, b"isolated data readme\n")
    _write(paths.manifest, publisher._manifest_encode(_minimal_manifest()))
    _write(root / publisher.PUBLISHER_IDENTITY, b"# isolated publisher\n")
    _write(root / publisher.PUBLISHER_TEST_IDENTITY, b"# isolated tests\n")
    return paths


def _isolated_source_snapshot() -> publisher.SourceSnapshot:
    records = (
        {
            "path": "isolated/source",
            "type": "regular",
            "nonsymlink": True,
            "bytes": 1,
            "sha256": "0" * 64,
        },
    )
    return publisher.SourceSnapshot(records, _sha256(b"isolated-source-snapshot"))


def _patch_isolated_policy(
    monkeypatch: pytest.MonkeyPatch,
    paths: publisher.RuntimePaths,
    candidate: publisher.Candidate,
) -> publisher.SourceSnapshot:
    source = _isolated_source_snapshot()
    monkeypatch.setattr(publisher, "_runtime_paths", lambda: paths)
    monkeypatch.setattr(
        publisher,
        "EXPECTED_PREPUBLICATION_DIRECTORY_IDENTITIES",
        ("data", "data/fixtures"),
    )
    monkeypatch.setattr(
        publisher,
        "README_SHA256",
        _sha256((paths.repository_root / publisher.README_IDENTITY).read_bytes()),
    )
    monkeypatch.setattr(
        publisher,
        "EXPECTED_ARCHIVE_SHA256",
        _sha256(candidate.archive_bytes),
    )
    monkeypatch.setattr(
        publisher,
        "EXPECTED_ARCHIVE_BYTES",
        len(candidate.archive_bytes),
    )
    monkeypatch.setattr(
        publisher,
        "_verify_accepted_source_identities",
        lambda unused_paths: source,
    )
    monkeypatch.setattr(
        publisher,
        "_build_candidate_twice",
        lambda unused_paths: candidate,
    )
    monkeypatch.setattr(
        publisher,
        "_validate_untrusted_archive",
        lambda payload: {}
        if payload == candidate.archive_bytes
        else (_ for _ in ()).throw(publisher.CandidateError("wrong isolated bytes")),
    )
    return source


def _identity(path: str, sha256: str) -> dict[str, Any]:
    return {"path": path, "sha256": sha256}


def _create_isolated_authority(
    paths: publisher.RuntimePaths,
    candidate: publisher.Candidate,
    source: publisher.SourceSnapshot,
) -> tuple[bytes, bytes]:
    acceptance_payload = b"isolated publisher implementation acceptance\n"
    _write(paths.publisher_acceptance, acceptance_payload)
    data = publisher._snapshot_data(paths, allow_exact_unregistered_target=True)
    publisher_file = publisher._read_repository_regular(
        paths,
        publisher.PUBLISHER_IDENTITY,
        label="isolated publisher",
    )
    tests_file = publisher._read_repository_regular(
        paths,
        publisher.PUBLISHER_TEST_IDENTITY,
        label="isolated publisher tests",
    )
    acceptance_file = publisher._read_repository_regular(
        paths,
        publisher.PUBLISHER_ACCEPTANCE_IDENTITY,
        label="isolated publisher acceptance",
    )
    template = publisher._build_manifest_template(
        candidate,
        publisher_sha256=publisher_file.sha256,
        publisher_test_sha256=tests_file.sha256,
        publisher_acceptance_sha256=acceptance_file.sha256,
    )
    authorization = {
        "schema_version": 1,
        "record_kind": "chapter06_atmosphere_fixture_publication_acceptance",
        "lane": "atmosphere_fixture",
        "manifest_role": "fixture",
        "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
        "writer": _identity(publisher.WRITER_IDENTITY, publisher.WRITER_SHA256),
        "writer_tests": _identity(
            publisher.WRITER_TEST_IDENTITY,
            publisher.WRITER_TEST_SHA256,
        ),
        "writer_candidate": _identity(
            publisher.WRITER_CANDIDATE_IDENTITY,
            publisher.WRITER_CANDIDATE_SHA256,
        ),
        "writer_acceptance": _identity(
            publisher.WRITER_ACCEPTANCE_IDENTITY,
            publisher.WRITER_ACCEPTANCE_SHA256,
        ),
        "candidate_byte_acceptance": _identity(
            publisher.CANDIDATE_BYTE_ACCEPTANCE_IDENTITY,
            publisher.CANDIDATE_BYTE_ACCEPTANCE_SHA256,
        ),
        "publisher": _identity(
            publisher.PUBLISHER_IDENTITY,
            publisher_file.sha256,
        ),
        "publisher_tests": _identity(
            publisher.PUBLISHER_TEST_IDENTITY,
            tests_file.sha256,
        ),
        "publisher_contract": _identity(
            publisher.CONTRACT_IDENTITY,
            publisher.CONTRACT_SHA256,
        ),
        "publisher_contract_audit": _identity(
            publisher.CONTRACT_AUDIT_IDENTITY,
            publisher.CONTRACT_AUDIT_SHA256,
        ),
        "publisher_acceptance": _identity(
            publisher.PUBLISHER_ACCEPTANCE_IDENTITY,
            acceptance_file.sha256,
        ),
        "artifact": {
            "path": publisher.DESTINATION_IDENTITY,
            "filename": Path(publisher.DESTINATION_IDENTITY).name,
            "bytes": len(candidate.archive_bytes),
            "sha256": _sha256(candidate.archive_bytes),
            "member_count": len(candidate.arrays),
            "archive_kind": publisher.ARCHIVE_KIND,
            "fixture_capture_schema_version": (
                publisher.EXTERNAL_CAPTURE_SCHEMA_VERSION
            ),
            "archive_contains_embedded_schema_version": False,
            "npy_member_format_version": publisher.NPY_MEMBER_FORMAT_VERSION,
            "scientific_fixture_schema_digest": (
                publisher.EXPECTED_FIXTURE_SCHEMA_DIGEST
            ),
            "scientific_payload_fingerprint": (publisher.EXPECTED_PAYLOAD_FINGERPRINT),
        },
        "manifest": {
            "path": publisher.MANIFEST_IDENTITY,
            "prepublication_sha256": data.manifest_sha256,
            "schema_version": 1,
            "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
            "ordered_entry_path_sha256": data.ordered_entry_path_sha256,
            "destination_entry_absent": True,
        },
        "manifest_entry_template": template,
        "manifest_entry_template_sha256": _sha256(publisher._template_encode(template)),
        "source_snapshot_sha256": source.aggregate_sha256,
        "data_snapshot_sha256": data.aggregate_sha256,
        "destination_parent_policy": (
            "existing data/fixtures directory only; publisher creates no directory"
        ),
        "no_replace_primitive": (
            "same-filesystem hard-link from create-exclusive stage"
        ),
    }
    assert tuple(authorization) == publisher.AUTHORIZATION_KEYS
    authorization_payload = _json_bytes(authorization)
    _write(paths.authorization, authorization_payload)
    review = {
        "schema_version": 1,
        "record_kind": ("chapter06_atmosphere_fixture_publication_record_review"),
        "authorization_path": publisher.AUTHORIZATION_IDENTITY,
        "authorization_sha256": _sha256(authorization_payload),
        "candidate_byte_acceptance_path": (
            publisher.CANDIDATE_BYTE_ACCEPTANCE_IDENTITY
        ),
        "candidate_byte_acceptance_sha256": (
            publisher.CANDIDATE_BYTE_ACCEPTANCE_SHA256
        ),
        "publisher_acceptance_path": publisher.PUBLISHER_ACCEPTANCE_IDENTITY,
        "publisher_acceptance_sha256": acceptance_file.sha256,
        "manifest_entry_template_sha256": authorization[
            "manifest_entry_template_sha256"
        ],
        "disposition": "ACCEPT",
    }
    assert tuple(review) == publisher.RECORD_REVIEW_KEYS
    review_payload = _json_bytes(review)
    _write(paths.record_review, review_payload)
    return authorization_payload, review_payload


@pytest.fixture
def isolated_authorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[
    tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ]
]:
    root = tmp_path / "textbook"
    paths = _initialize_isolated_tree(root)
    candidate = _candidate()
    source = _patch_isolated_policy(monkeypatch, paths, candidate)
    _create_isolated_authority(paths, candidate, source)
    yield paths, candidate, source


def test_production_surfaces_accept_no_root_or_destination() -> None:
    assert tuple(inspect.signature(publisher.verify_only).parameters) == ()
    assert tuple(inspect.signature(publisher.authorized_dry_run).parameters) == ()
    assert tuple(inspect.signature(publisher.publish).parameters) == ()
    parser = publisher._build_argument_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--root" not in options
    assert "--destination" not in options
    assert "--force" not in options
    assert "--replace" not in options
    assert "--repair" not in options


def test_exact_canonical_constants_and_trust_hashes() -> None:
    assert publisher.CANONICAL_REPOSITORY_ROOT == Path(
        "/Users/ysting/stellar-spectroscopy-from-scratch-gpu"
    )
    assert (
        publisher.DESTINATION_IDENTITY
        == "data/fixtures/chapter06_atmosphere_one_line_inputs.npz"
    )
    assert publisher.MANIFEST_ROLE == "fixture"
    assert publisher.CONTRACT_SHA256 == (
        "3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b"
    )
    assert publisher.CONTRACT_AUDIT_SHA256 == (
        "fe48eb57f1f665a3f41756344c631a365a0b9260905918e80a1c2e58f8e335cc"
    )
    assert publisher.CANDIDATE_BYTE_ACCEPTANCE_SHA256 == (
        "8298b9473cf89161441bbd72a881c744e38fba699aa088eb876014642c91ed71"
    )
    assert not hasattr(publisher, "LOCK_PATH")


def test_member_hash_is_exact_c_order_bytes_only() -> None:
    array = publisher.worker.np.asarray([1.0, 2.0], dtype=publisher.worker.np.float64)
    expected = hashlib.sha256(
        publisher.worker.np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()
    typed_shape_digest = hashlib.sha256()
    typed_shape_digest.update(array.dtype.str.encode("ascii"))
    typed_shape_digest.update(
        publisher.worker.np.asarray(
            array.shape, dtype=publisher.worker.np.int64
        ).tobytes()
    )
    typed_shape_digest.update(array.tobytes(order="C"))
    assert publisher._array_sha256(array) == expected
    assert publisher._array_sha256(array) != typed_shape_digest.hexdigest()


def test_member_units_and_axes_are_exact_nonempty_scientific_conventions() -> None:
    assert set(publisher._MEMBER_CONVENTIONS) == set(publisher.worker.FIXTURE_SCHEMA)
    for name, (unit, axes) in publisher._MEMBER_CONVENTIONS.items():
        assert unit
        assert all(axis for axis in axes)
        expected_shape, _ = publisher.worker.FIXTURE_SCHEMA[name]
        assert len(axes) == len(expected_shape)


@pytest.mark.parametrize(
    "identity",
    [
        "/absolute/path",
        "../escape",
        "data/../escape",
        "data\\escape",
        "data//escape",
        "data/%2e%2e/escape",
        "dáta/file",
    ],
)
def test_repository_identity_rejects_aliases(identity: str) -> None:
    with pytest.raises(publisher.IdentityError):
        publisher._validate_repository_identity(identity, label="mutation")


def test_strict_json_rejects_duplicates_and_nonfinite() -> None:
    with pytest.raises(publisher.IdentityError, match="duplicate"):
        publisher._parse_strict_json(b'{"x":1,"x":2}', label="duplicate")
    with pytest.raises(publisher.IdentityError, match="nonfinite"):
        publisher._parse_strict_json(b'{"x":NaN}', label="nonfinite")


def test_review_and_authorization_key_orders_are_closed(
    isolated_authorized: tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ],
) -> None:
    paths, candidate, source = isolated_authorized
    data = publisher._snapshot_data(paths, allow_exact_unregistered_target=True)
    authority = publisher._load_authority(
        paths,
        candidate=candidate,
        source_snapshot=source,
        data_snapshot=data,
    )
    assert tuple(authority.authorization) == publisher.AUTHORIZATION_KEYS
    assert tuple(authority.review) == publisher.RECORD_REVIEW_KEYS
    mutated = deepcopy(authority.review)
    value = mutated.pop("disposition")
    reordered = {"disposition": value, **mutated}
    with pytest.raises(publisher.IdentityError, match="key order"):
        publisher._require_exact_keys(
            reordered,
            publisher.RECORD_REVIEW_KEYS,
            label="mutated review",
        )


def test_manifest_roundtrip_preserves_live_unsorted_bytes() -> None:
    paths = publisher._runtime_paths()
    file = publisher._read_repository_regular(
        paths,
        publisher.MANIFEST_IDENTITY,
        label="live manifest",
    )
    manifest, identities = publisher._parse_manifest_file(file)
    assert publisher._manifest_encode(manifest) == file.payload
    assert identities.count(publisher.DESTINATION_IDENTITY) == 1
    assert identities.index(publisher.DESTINATION_IDENTITY) < len(identities) - 1
    assert identities != tuple(sorted(identities))
    report = publisher.verify_published()
    assert report["manifest_sha256"] == file.sha256
    assert report["later_manifest_entry_count"] == (
        len(identities) - identities.index(publisher.DESTINATION_IDENTITY) - 1
    )


def test_exact_unregistered_recovery_normalizes_only_exact_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    candidate = _candidate()
    _patch_isolated_policy(monkeypatch, paths, candidate)
    absent = publisher._snapshot_data(
        paths,
        allow_exact_unregistered_target=True,
    )
    _write(paths.destination, candidate.archive_bytes, mode=0o600)
    recovery = publisher._snapshot_data(
        paths,
        allow_exact_unregistered_target=True,
    )
    assert recovery.aggregate_sha256 == absent.aggregate_sha256
    assert any(
        item["path"] == publisher.DESTINATION_IDENTITY
        and item["classification"] == "exact_unregistered_recovery_target"
        for item in recovery.nonmanifest_support
    )
    paths.destination.write_bytes(b"nonexact")
    with pytest.raises(publisher.IdentityError, match="unexpected nonmanifest"):
        publisher._snapshot_data(
            paths,
            allow_exact_unregistered_target=True,
        )


def test_closed_data_directory_inventory_rejects_unexpected_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    candidate = _candidate()
    _patch_isolated_policy(monkeypatch, paths, candidate)
    (paths.data_root / "rogue-empty-directory").mkdir()
    with pytest.raises(publisher.IdentityError, match="directory inventory"):
        publisher._snapshot_data(paths, allow_exact_unregistered_target=True)


def test_no_follow_reader_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target"
    target.write_bytes(b"payload")
    (root / "alias").symlink_to(target)
    paths = _runtime_paths(root)
    with pytest.raises(publisher.IdentityError):
        publisher._read_repository_regular(paths, "alias", label="symlink")


def test_template_has_lexical_arrays_and_exact_late_placeholders() -> None:
    arrays = {
        "a": {
            "shape": [],
            "dtype": "float64",
            "unit": "1",
            "sha256": "a" * 64,
            "axes": [],
            "ownership": "atmosphere fixture owns this input member",
        },
        "b": {
            "shape": [1],
            "dtype": "int16",
            "unit": "index",
            "sha256": "b" * 64,
            "axes": ["selected_line"],
            "ownership": "atmosphere fixture owns this input member",
        },
    }
    candidate = publisher.Candidate(b"bytes", arrays, {}, {})
    template = publisher._build_manifest_template(
        candidate,
        publisher_sha256="1" * 64,
        publisher_test_sha256="2" * 64,
        publisher_acceptance_sha256="3" * 64,
    )
    encoded = publisher._template_encode(template).decode()
    assert tuple(template["arrays"]) == ("a", "b")
    assert encoded.count(publisher.AUTHORIZATION_PLACEHOLDER) == 1
    assert encoded.count(publisher.RECORD_REVIEW_PLACEHOLDER) == 1
    realized = publisher._realize_template(
        template,
        authorization_sha256="4" * 64,
        review_sha256="5" * 64,
    )
    changed = [key for key in template if template[key] != realized[key]]
    assert changed == [
        "publication_acceptance_sha256",
        "publication_record_review_sha256",
    ]


def test_append_only_manifest_delete_last_recovers_exact_m(
    isolated_authorized: tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ],
) -> None:
    paths, candidate, source = isolated_authorized
    data = publisher._snapshot_data(paths, allow_exact_unregistered_target=True)
    authority = publisher._load_authority(
        paths,
        candidate=candidate,
        source_snapshot=source,
        data_snapshot=data,
    )
    manifest_file = publisher._read_repository_regular(
        paths,
        publisher.MANIFEST_IDENTITY,
        label="isolated manifest",
    )
    post, post_bytes, realized = publisher._construct_postpublication_manifest(
        manifest_file,
        authority,
    )
    assert post["entries"][-1] == realized
    reconstructed = deepcopy(post)
    reconstructed["entries"].pop()
    assert publisher._manifest_encode(reconstructed) == manifest_file.payload
    assert publisher._manifest_encode(post) == post_bytes


def test_real_canonical_postpublication_verification_avoids_candidate_and_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = publisher._runtime_paths()
    manifest_before = paths.manifest.read_bytes()
    destination_before = paths.destination.exists()
    called = False

    def forbidden_build(unused_paths: publisher.RuntimePaths) -> publisher.Candidate:
        nonlocal called
        called = True
        raise AssertionError("postpublication verification must not rebuild")

    monkeypatch.setattr(publisher, "_build_candidate_twice", forbidden_build)
    report = publisher.verify_published()
    assert report["state"] == "exact_registered"
    assert report["complete_validation"] is True
    assert not called
    assert paths.manifest.read_bytes() == manifest_before
    assert paths.destination.exists() == destination_before
    assert not any(
        child.name.startswith(
            (publisher.ARTIFACT_STAGE_PREFIX, publisher.MANIFEST_TEMPORARY_PREFIX)
        )
        for parent in (paths.data_root, paths.destination.parent)
        for child in parent.iterdir()
    )


def test_verification_only_has_no_publication_call_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(b"dry-run")
    source = _isolated_source_snapshot()
    data = publisher.TreeSnapshot(
        (),
        (),
        (),
        (),
        "1" * 64,
        "2" * 64,
        "3" * 64,
    )
    monkeypatch.setattr(
        publisher,
        "_verify_accepted_source_identities",
        lambda unused_paths: source,
    )
    monkeypatch.setattr(
        publisher,
        "_snapshot_data",
        lambda unused_paths, allow_exact_unregistered_target: data,
    )
    monkeypatch.setattr(
        publisher,
        "_build_candidate_twice",
        lambda unused_paths: candidate,
    )
    monkeypatch.setattr(
        publisher,
        "_current_candidate_identities",
        lambda unused_paths: {
            "publisher": _identity(publisher.PUBLISHER_IDENTITY, "4" * 64),
            "publisher_tests": _identity(
                publisher.PUBLISHER_TEST_IDENTITY,
                "5" * 64,
            ),
        },
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("verification-only reached a mutation primitive")

    for name in (
        "_exclusive_data_lock",
        "_create_exclusive_stage",
        "_atomic_install_no_replace",
        "_replace_manifest_atomically",
    ):
        monkeypatch.setattr(publisher, name, forbidden)
    first = publisher.verify_only()
    second = publisher.verify_only()
    assert publisher._report_bytes(first) == publisher._report_bytes(second)
    assert first["publication_authorized"] is False
    assert first["publication_performed"] is False
    assert first["decision"] == "VERIFIED_CANDIDATE_ONLY_NOT_AUTHORIZED"


def test_create_exclusive_stage_rejects_short_write_and_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = os.write

    def short_write(descriptor: int, payload: bytes) -> int:
        return original_write(descriptor, payload[:-1])

    monkeypatch.setattr(publisher.os, "write", short_write)
    with pytest.raises(publisher.PublicationIOError, match="short write"):
        publisher._create_exclusive_stage(
            tmp_path,
            prefix=publisher.ARTIFACT_STAGE_PREFIX,
            payload=b"long-enough",
        )
    assert list(tmp_path.iterdir()) == []


def test_create_exclusive_stage_fsync_failure_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fsync = os.fsync
    calls = 0

    def failed_file_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(publisher.os, "fsync", failed_file_fsync)
    with pytest.raises(OSError, match="injected"):
        publisher._create_exclusive_stage(
            tmp_path,
            prefix=publisher.ARTIFACT_STAGE_PREFIX,
            payload=b"payload",
        )
    assert list(tmp_path.iterdir()) == []


def test_create_exclusive_stage_readback_failure_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_readback(descriptor: int, expected_size: int) -> bytes:
        raise publisher.PublicationIOError("injected readback failure")

    monkeypatch.setattr(publisher, "_readback_descriptor", failed_readback)
    with pytest.raises(publisher.PublicationIOError, match="readback"):
        publisher._create_exclusive_stage(
            tmp_path,
            prefix=publisher.ARTIFACT_STAGE_PREFIX,
            payload=b"payload",
        )
    assert list(tmp_path.iterdir()) == []


def test_create_exclusive_stage_enforces_0600_independently_of_umask(
    tmp_path: Path,
) -> None:
    previous_umask = os.umask(0o777)
    try:
        stage = publisher._create_exclusive_stage(
            tmp_path,
            prefix=publisher.ARTIFACT_STAGE_PREFIX,
            payload=b"payload",
        )
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(stage.path.stat().st_mode) == 0o600
    publisher._unlink_owned_stage(stage)
    assert list(tmp_path.iterdir()) == []


def test_quarantine_object_hard_stops_without_cleanup(tmp_path: Path) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    quarantine = (
        paths.destination.parent / f"{publisher.ARTIFACT_STAGE_PREFIX}crash-left"
    )
    quarantine.write_bytes(b"looks exact but is never promoted")
    before = quarantine.read_bytes()
    with pytest.raises(publisher.QuarantineError, match="hard-stops"):
        publisher._scan_quarantine(paths)
    assert quarantine.read_bytes() == before


def test_atomic_no_replace_loses_nonexact_target_race_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    candidate = _candidate()
    monkeypatch.setattr(
        publisher,
        "_validate_untrusted_archive",
        lambda payload: {}
        if payload == candidate.archive_bytes
        else (_ for _ in ()).throw(publisher.CandidateError("wrong bytes")),
    )
    paths.destination.write_bytes(b"racing nonexact target")
    before = paths.destination.read_bytes()
    with publisher._exclusive_data_lock(paths) as (data_fd, data_metadata):
        with pytest.raises(publisher.IdentityError):
            publisher._atomic_install_no_replace(
                paths,
                candidate,
                data_fd=data_fd,
                data_metadata=data_metadata,
            )
    assert paths.destination.read_bytes() == before
    assert not any(
        child.name.startswith(publisher.ARTIFACT_STAGE_PREFIX)
        for child in paths.destination.parent.iterdir()
    )


def test_atomic_no_replace_exact_target_race_becomes_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    candidate = _candidate()
    monkeypatch.setattr(publisher, "_validate_untrusted_archive", lambda payload: {})
    paths.destination.write_bytes(candidate.archive_bytes)
    with publisher._exclusive_data_lock(paths) as (data_fd, data_metadata):
        result = publisher._atomic_install_no_replace(
            paths,
            candidate,
            data_fd=data_fd,
            data_metadata=data_metadata,
        )
    assert result == "identical-existing"
    assert paths.destination.read_bytes() == candidate.archive_bytes
    assert paths.destination.stat().st_nlink == 1


def test_atomic_no_replace_rejects_preexisting_multilink_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    candidate = _candidate()
    monkeypatch.setattr(publisher, "_validate_untrusted_archive", lambda payload: {})
    paths.destination.write_bytes(candidate.archive_bytes)
    alias = tmp_path / "target-hard-link"
    os.link(paths.destination, alias)
    with publisher._exclusive_data_lock(paths) as (data_fd, data_metadata):
        with pytest.raises(publisher.IdentityError, match="hard-link count"):
            publisher._atomic_install_no_replace(
                paths,
                candidate,
                data_fd=data_fd,
                data_metadata=data_metadata,
            )
    assert paths.destination.read_bytes() == candidate.archive_bytes
    assert paths.destination.stat().st_nlink == 2
    assert not any(
        child.name.startswith(publisher.ARTIFACT_STAGE_PREFIX)
        for child in paths.destination.parent.iterdir()
    )


def test_artifact_stage_inode_substitution_rejects_before_target_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    candidate = _candidate()
    monkeypatch.setattr(publisher, "_validate_untrusted_archive", lambda payload: {})
    create_stage = publisher._create_exclusive_stage

    def substituted_stage(*args: Any, **kwargs: Any) -> publisher.Stage:
        stage = create_stage(*args, **kwargs)
        stage.path.unlink()
        _write(stage.path, b"foreign-stage-bytes", mode=0o600)
        return stage

    monkeypatch.setattr(publisher, "_create_exclusive_stage", substituted_stage)
    with publisher._exclusive_data_lock(paths) as (data_fd, data_metadata):
        with pytest.raises(publisher.PublicationIOError, match="stage"):
            publisher._atomic_install_no_replace(
                paths,
                candidate,
                data_fd=data_fd,
                data_metadata=data_metadata,
            )
    assert not paths.destination.exists()
    leftovers = [
        child
        for child in paths.destination.parent.iterdir()
        if child.name.startswith(publisher.ARTIFACT_STAGE_PREFIX)
    ]
    assert len(leftovers) == 1
    assert leftovers[0].read_bytes() == b"foreign-stage-bytes"


def test_artifact_parent_swap_fsyncs_retained_mutated_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    candidate = _candidate()
    monkeypatch.setattr(publisher, "_validate_untrusted_archive", lambda payload: {})
    parent = paths.destination.parent
    mutated_inode = parent.stat().st_ino
    moved = parent.with_name("fixtures-moved-after-link")
    real_link = publisher.os.link
    real_fsync = publisher.os.fsync
    linked = False
    fsynced_after_link: list[int] = []

    def link_then_swap(*args: Any, **kwargs: Any) -> None:
        nonlocal linked
        real_link(*args, **kwargs)
        parent.rename(moved)
        parent.mkdir()
        linked = True

    def record_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if linked and stat.S_ISDIR(metadata.st_mode):
            fsynced_after_link.append(metadata.st_ino)
        real_fsync(descriptor)

    monkeypatch.setattr(publisher.os, "link", link_then_swap)
    monkeypatch.setattr(publisher.os, "fsync", record_fsync)
    with publisher._exclusive_data_lock(paths) as (data_fd, data_metadata):
        with pytest.raises(publisher.IdentityError, match="inode changed"):
            publisher._atomic_install_no_replace(
                paths,
                candidate,
                data_fd=data_fd,
                data_metadata=data_metadata,
            )
    assert mutated_inode in fsynced_after_link
    assert parent.stat().st_ino not in fsynced_after_link
    assert (moved / paths.destination.name).read_bytes() == candidate.archive_bytes
    assert not paths.destination.exists()


def test_retained_parent_identity_detects_directory_swap(tmp_path: Path) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    parent = paths.destination.parent
    before = parent.lstat()
    moved = parent.with_name("fixtures-moved-by-racer")
    parent.rename(moved)
    parent.mkdir()
    with pytest.raises(publisher.IdentityError, match="inode changed"):
        publisher._require_same_directory_inode(
            parent,
            before,
            label="adversarial parent",
        )


def test_manifest_race_rejects_and_cleans_own_temporary(
    tmp_path: Path,
) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    expected = publisher._read_repository_regular(
        paths,
        publisher.MANIFEST_IDENTITY,
        label="expected manifest",
    )
    raced = deepcopy(_minimal_manifest())
    raced["entries"].append(
        {
            "path": "data/fixtures/race.npz",
            "role": "fixture",
            "sha256": "1" * 64,
            "bytes": 1,
        }
    )
    _write(paths.manifest, publisher._manifest_encode(raced))
    raced_bytes = paths.manifest.read_bytes()
    with publisher._exclusive_data_lock(paths) as (data_fd, data_metadata):
        with pytest.raises(publisher.ManifestError, match="raced"):
            publisher._replace_manifest_atomically(
                paths,
                expected_manifest=expected,
                intended_bytes=publisher._manifest_encode(_minimal_manifest()),
                data_fd=data_fd,
                data_metadata=data_metadata,
            )
    assert paths.manifest.read_bytes() == raced_bytes
    assert not any(
        child.name.startswith(publisher.MANIFEST_TEMPORARY_PREFIX)
        for child in paths.data_root.iterdir()
    )


def test_manifest_temporary_inode_substitution_preserves_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    expected = publisher._read_repository_regular(
        paths,
        publisher.MANIFEST_IDENTITY,
        label="expected manifest",
    )
    manifest_before = paths.manifest.read_bytes()
    intended = publisher._manifest_encode(_minimal_manifest())
    create_stage = publisher._create_exclusive_stage

    def substituted_stage(*args: Any, **kwargs: Any) -> publisher.Stage:
        stage = create_stage(*args, **kwargs)
        stage.path.unlink()
        _write(stage.path, b'{"foreign":true}\n', mode=expected.mode)
        return stage

    monkeypatch.setattr(publisher, "_create_exclusive_stage", substituted_stage)
    with publisher._exclusive_data_lock(paths) as (data_fd, data_metadata):
        with pytest.raises(publisher.PublicationIOError, match="temporary"):
            publisher._replace_manifest_atomically(
                paths,
                expected_manifest=expected,
                intended_bytes=intended,
                data_fd=data_fd,
                data_metadata=data_metadata,
            )
    assert paths.manifest.read_bytes() == manifest_before
    leftovers = [
        child
        for child in paths.data_root.iterdir()
        if child.name.startswith(publisher.MANIFEST_TEMPORARY_PREFIX)
    ]
    assert len(leftovers) == 1
    assert leftovers[0].read_bytes() == b'{"foreign":true}\n'


def test_manifest_parent_swap_fsyncs_retained_mutated_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    expected = publisher._read_repository_regular(
        paths,
        publisher.MANIFEST_IDENTITY,
        label="expected manifest",
    )
    intended_manifest = deepcopy(_minimal_manifest())
    intended_manifest["entries"].append(
        {
            "path": "data/fixtures/registered.npz",
            "role": "fixture",
            "sha256": "1" * 64,
            "bytes": 1,
        }
    )
    intended = publisher._manifest_encode(intended_manifest)
    data_root = paths.data_root
    mutated_inode = data_root.stat().st_ino
    moved = data_root.with_name("data-moved-after-replace")
    real_replace = publisher.os.replace
    real_fsync = publisher.os.fsync
    replaced = False
    fsynced_after_replace: list[int] = []

    def replace_then_swap(*args: Any, **kwargs: Any) -> None:
        nonlocal replaced
        real_replace(*args, **kwargs)
        data_root.rename(moved)
        data_root.mkdir()
        replaced = True

    def record_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if replaced and stat.S_ISDIR(metadata.st_mode):
            fsynced_after_replace.append(metadata.st_ino)
        real_fsync(descriptor)

    monkeypatch.setattr(publisher.os, "replace", replace_then_swap)
    monkeypatch.setattr(publisher.os, "fsync", record_fsync)
    with pytest.raises(publisher.IdentityError, match="inode changed"):
        with publisher._exclusive_data_lock(paths) as (data_fd, data_metadata):
            publisher._replace_manifest_atomically(
                paths,
                expected_manifest=expected,
                intended_bytes=intended,
                data_fd=data_fd,
                data_metadata=data_metadata,
            )
    assert mutated_inode in fsynced_after_replace
    assert data_root.stat().st_ino not in fsynced_after_replace
    assert (moved / paths.manifest.name).read_bytes() == intended


def test_canonical_data_directory_lock_excludes_second_process(tmp_path: Path) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    code = """
import fcntl
import os
import sys
fd = os.open(sys.argv[1], os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(0)
raise SystemExit(7)
"""
    with publisher._exclusive_data_lock(paths):
        completed = subprocess.run(
            [sys.executable, "-c", code, str(paths.data_root)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    assert completed.returncode == 0


def test_data_directory_lock_failure_occurs_before_any_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _initialize_isolated_tree(tmp_path / "root")
    manifest_before = paths.manifest.read_bytes()

    def unavailable_lock(descriptor: int, operation: int) -> None:
        raise OSError("injected lock failure")

    monkeypatch.setattr(publisher.fcntl, "flock", unavailable_lock)
    with pytest.raises(publisher.PublicationIOError, match="unavailable"):
        with publisher._exclusive_data_lock(paths):
            raise AssertionError("unavailable lock yielded")
    assert paths.manifest.read_bytes() == manifest_before
    assert not paths.destination.exists()
    assert not any(
        child.name.startswith(
            (publisher.ARTIFACT_STAGE_PREFIX, publisher.MANIFEST_TEMPORARY_PREFIX)
        )
        for parent in (paths.data_root, paths.destination.parent)
        for child in parent.iterdir()
    )


def test_mutation_sequences_do_not_reopen_parent_for_fsync() -> None:
    assert (
        "_fsync_directory" not in publisher._atomic_install_no_replace.__code__.co_names
    )
    assert (
        "_fsync_directory"
        not in publisher._replace_manifest_atomically.__code__.co_names
    )
    assert "_fsync_retained_directory" in (
        publisher._atomic_install_no_replace.__code__.co_names
    )
    assert "_fsync_retained_directory" in (
        publisher._replace_manifest_atomically.__code__.co_names
    )


def test_authorized_isolated_full_transition_and_idempotent_noop(
    isolated_authorized: tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, candidate, _ = isolated_authorized
    manifest_before = paths.manifest.read_bytes()
    real_replace = publisher.os.replace
    replacements: list[tuple[Any, ...]] = []

    def record_real_replace(*args: Any, **kwargs: Any) -> None:
        replacements.append(args)
        real_replace(*args, **kwargs)

    monkeypatch.setattr(publisher.os, "replace", record_real_replace)
    assert publisher.publish() == "installed-and-registered"
    assert len(replacements) == 1
    assert paths.destination.read_bytes() == candidate.archive_bytes
    assert paths.destination.stat().st_nlink == 1
    manifest_after = paths.manifest.read_bytes()
    assert manifest_after != manifest_before
    parsed = publisher._parse_strict_json(
        manifest_after,
        label="postpublication manifest",
    )
    assert [entry["path"] for entry in parsed["entries"]] == [
        publisher.DESTINATION_IDENTITY
    ]
    entry = parsed["entries"][0]
    assert entry["role"] == "fixture"
    assert entry["publication_acceptance_sha256"] == _sha256(
        paths.authorization.read_bytes()
    )
    assert entry["publication_record_review_sha256"] == _sha256(
        paths.record_review.read_bytes()
    )
    publisher._internal_validate_published()

    artifact_inode = paths.destination.stat().st_ino
    manifest_inode = paths.manifest.stat().st_ino
    assert publisher.publish() == "identical-registered-no-op"
    assert len(replacements) == 1
    assert paths.destination.stat().st_ino == artifact_inode
    assert paths.manifest.stat().st_ino == manifest_inode
    assert paths.manifest.read_bytes() == manifest_after


@pytest.mark.parametrize(
    "mutation",
    (
        "authorization-whitespace",
        "coherent-authorization-review-rebind",
        "review-only-whitespace",
        "target-bytes",
        "authorization-template",
        "intended-manifest-stage",
    ),
)
def test_final_manifest_boundary_rejects_ordinary_presyscall_mutations(
    isolated_authorized: tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    paths, candidate, _ = isolated_authorized
    manifest_before = paths.manifest.read_bytes()
    authorization_before = paths.authorization.read_bytes()
    review_before = paths.record_review.read_bytes()
    real_snapshot = publisher._snapshot_data
    real_replace = publisher.os.replace
    boundary_reached = False
    replacements = 0
    named_mutations: dict[Path, bytes] = {}

    def whitespace(payload: bytes) -> bytes:
        assert payload.endswith(b"\n")
        return payload[:-1] + b" \n"

    def mutate_after_final_snapshot(stage: publisher.Stage) -> None:
        if mutation == "authorization-whitespace":
            payload = whitespace(authorization_before)
            _write(paths.authorization, payload)
            named_mutations[paths.authorization] = payload
            return
        if mutation == "coherent-authorization-review-rebind":
            authorization_payload = whitespace(authorization_before)
            _write(paths.authorization, authorization_payload)
            review = publisher._parse_strict_json(review_before, label="review")
            review["authorization_sha256"] = _sha256(authorization_payload)
            review_payload = _json_bytes(review)
            _write(paths.record_review, review_payload)
            named_mutations[paths.authorization] = authorization_payload
            named_mutations[paths.record_review] = review_payload
            return
        if mutation == "review-only-whitespace":
            payload = whitespace(review_before)
            _write(paths.record_review, payload)
            named_mutations[paths.record_review] = payload
            return
        if mutation == "target-bytes":
            payload = b"ordinary target mutation after final snapshot"
            _write(paths.destination, payload)
            named_mutations[paths.destination] = payload
            return
        if mutation == "authorization-template":
            authorization = publisher._parse_strict_json(
                authorization_before,
                label="authorization",
            )
            authorization["manifest_entry_template"]["scope"] += " mutated"
            template_digest = _sha256(
                publisher._template_encode(
                    authorization["manifest_entry_template"],
                )
            )
            authorization["manifest_entry_template_sha256"] = template_digest
            authorization_payload = _json_bytes(authorization)
            _write(paths.authorization, authorization_payload)
            review = publisher._parse_strict_json(review_before, label="review")
            review["authorization_sha256"] = _sha256(authorization_payload)
            review["manifest_entry_template_sha256"] = template_digest
            review_payload = _json_bytes(review)
            _write(paths.record_review, review_payload)
            named_mutations[paths.authorization] = authorization_payload
            named_mutations[paths.record_review] = review_payload
            return
        if mutation == "intended-manifest-stage":
            _write(stage.path, b'{"mutated_intended_manifest":true}\n')
            return
        raise AssertionError(f"unknown boundary mutation: {mutation}")

    def snapshot_then_mutate(
        call_paths: publisher.RuntimePaths,
        **kwargs: Any,
    ) -> publisher.TreeSnapshot:
        nonlocal boundary_reached
        snapshot = real_snapshot(call_paths, **kwargs)
        stage = kwargs.get("owned_manifest_temporary")
        if stage is not None and not boundary_reached:
            boundary_reached = True
            mutate_after_final_snapshot(stage)
        return snapshot

    def record_real_replace(*args: Any, **kwargs: Any) -> None:
        nonlocal replacements
        replacements += 1
        real_replace(*args, **kwargs)

    monkeypatch.setattr(publisher, "_snapshot_data", snapshot_then_mutate)
    monkeypatch.setattr(publisher.os, "replace", record_real_replace)
    with pytest.raises(publisher.AtmosphereFixturePublicationError):
        publisher.publish()

    assert boundary_reached
    assert replacements == 0
    assert paths.manifest.read_bytes() == manifest_before
    assert not any(
        child.name.startswith(publisher.MANIFEST_TEMPORARY_PREFIX)
        for child in paths.data_root.iterdir()
    )
    for path, expected_payload in named_mutations.items():
        assert path.read_bytes() == expected_payload
    if paths.destination not in named_mutations:
        assert paths.destination.read_bytes() == candidate.archive_bytes


def test_manifest_failure_leaves_exact_inert_file_then_recovers(
    isolated_authorized: tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, candidate, _ = isolated_authorized
    manifest_before = paths.manifest.read_bytes()
    real_replace = publisher._replace_manifest_atomically

    def injected_failure(*args: Any, **kwargs: Any) -> None:
        raise publisher.PublicationIOError("injected manifest failure")

    monkeypatch.setattr(
        publisher,
        "_replace_manifest_atomically",
        injected_failure,
    )
    with pytest.raises(publisher.PublicationIOError, match="injected"):
        publisher.publish()
    assert paths.destination.read_bytes() == candidate.archive_bytes
    assert paths.manifest.read_bytes() == manifest_before

    monkeypatch.setattr(
        publisher,
        "_replace_manifest_atomically",
        real_replace,
    )
    assert publisher.publish() == "exact-unregistered-recovery-and-registered"
    assert paths.destination.read_bytes() == candidate.archive_bytes


def test_nontarget_data_change_after_artifact_install_blocks_manifest(
    isolated_authorized: tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, candidate, _ = isolated_authorized
    manifest_before = paths.manifest.read_bytes()
    real_install = publisher._atomic_install_no_replace

    def install_then_mutate(
        call_paths: publisher.RuntimePaths,
        call_candidate: publisher.Candidate,
        *,
        data_fd: int,
        data_metadata: os.stat_result,
    ) -> str:
        result = real_install(
            call_paths,
            call_candidate,
            data_fd=data_fd,
            data_metadata=data_metadata,
        )
        (call_paths.repository_root / publisher.README_IDENTITY).write_bytes(
            b"mutated after artifact installation\n"
        )
        return result

    monkeypatch.setattr(
        publisher,
        "_atomic_install_no_replace",
        install_then_mutate,
    )
    with pytest.raises(publisher.IdentityError):
        publisher.publish()
    assert paths.destination.read_bytes() == candidate.archive_bytes
    assert paths.manifest.read_bytes() == manifest_before


def test_nontarget_data_change_after_manifest_staging_blocks_replacement(
    isolated_authorized: tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, candidate, _ = isolated_authorized
    manifest_before = paths.manifest.read_bytes()
    real_create = publisher._create_exclusive_stage

    def stage_then_mutate(*args: Any, **kwargs: Any) -> publisher.Stage:
        stage = real_create(*args, **kwargs)
        if kwargs.get("prefix") == publisher.MANIFEST_TEMPORARY_PREFIX:
            (paths.repository_root / publisher.README_IDENTITY).write_bytes(
                b"mutated after manifest staging\n"
            )
        return stage

    monkeypatch.setattr(
        publisher,
        "_create_exclusive_stage",
        stage_then_mutate,
    )
    with pytest.raises(publisher.IdentityError):
        publisher.publish()
    assert paths.destination.read_bytes() == candidate.archive_bytes
    assert paths.manifest.read_bytes() == manifest_before
    assert not any(
        child.name.startswith(publisher.MANIFEST_TEMPORARY_PREFIX)
        for child in paths.data_root.iterdir()
    )


def test_source_change_after_artifact_install_blocks_manifest(
    isolated_authorized: tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, candidate, accepted_source = isolated_authorized
    manifest_before = paths.manifest.read_bytes()
    calls = 0

    def changing_source(
        unused_paths: publisher.RuntimePaths,
    ) -> publisher.SourceSnapshot:
        nonlocal calls
        calls += 1
        if calls >= 3:
            return publisher.SourceSnapshot(
                accepted_source.records,
                "f" * 64,
            )
        return accepted_source

    monkeypatch.setattr(
        publisher,
        "_verify_accepted_source_identities",
        changing_source,
    )
    with pytest.raises(publisher.IdentityError, match="source snapshot changed"):
        publisher.publish()
    assert calls >= 3
    assert paths.destination.read_bytes() == candidate.archive_bytes
    assert paths.manifest.read_bytes() == manifest_before


def test_source_change_after_manifest_staging_blocks_replacement(
    isolated_authorized: tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, candidate, accepted_source = isolated_authorized
    manifest_before = paths.manifest.read_bytes()
    real_create = publisher._create_exclusive_stage
    manifest_staged = False

    def source_at_boundary(
        unused_paths: publisher.RuntimePaths,
    ) -> publisher.SourceSnapshot:
        if manifest_staged:
            return publisher.SourceSnapshot(accepted_source.records, "e" * 64)
        return accepted_source

    def stage_then_change_source(*args: Any, **kwargs: Any) -> publisher.Stage:
        nonlocal manifest_staged
        stage = real_create(*args, **kwargs)
        if kwargs.get("prefix") == publisher.MANIFEST_TEMPORARY_PREFIX:
            manifest_staged = True
        return stage

    monkeypatch.setattr(
        publisher,
        "_verify_accepted_source_identities",
        source_at_boundary,
    )
    monkeypatch.setattr(
        publisher,
        "_create_exclusive_stage",
        stage_then_change_source,
    )
    with pytest.raises(publisher.IdentityError, match="immediately before"):
        publisher.publish()
    assert manifest_staged
    assert paths.destination.read_bytes() == candidate.archive_bytes
    assert paths.manifest.read_bytes() == manifest_before
    assert not any(
        child.name.startswith(publisher.MANIFEST_TEMPORARY_PREFIX)
        for child in paths.data_root.iterdir()
    )


def test_fresh_postpublication_validation_runs_while_data_lock_is_held(
    isolated_authorized: tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _, _ = isolated_authorized
    real_fresh = publisher._fresh_postpublication_validation
    observed = False

    def assert_locked(call_paths: publisher.RuntimePaths) -> None:
        nonlocal observed
        code = """
import fcntl
import os
import sys
fd = os.open(sys.argv[1], os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(0)
raise SystemExit(9)
"""
        completed = subprocess.run(
            [sys.executable, "-c", code, str(call_paths.data_root)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert completed.returncode == 0
        observed = True
        real_fresh(call_paths)

    monkeypatch.setattr(
        publisher,
        "_fresh_postpublication_validation",
        assert_locked,
    )
    assert publisher.publish() == "installed-and-registered"
    assert observed


def test_stale_authorization_rejects_before_stage_or_target(
    isolated_authorized: tuple[
        publisher.RuntimePaths,
        publisher.Candidate,
        publisher.SourceSnapshot,
    ],
) -> None:
    paths, _, _ = isolated_authorized
    authorization = publisher._parse_strict_json(
        paths.authorization.read_bytes(),
        label="authorization",
    )
    authorization["artifact"]["sha256"] = "f" * 64
    stale_payload = _json_bytes(authorization)
    _write(paths.authorization, stale_payload)
    review = publisher._parse_strict_json(
        paths.record_review.read_bytes(),
        label="review",
    )
    review["authorization_sha256"] = _sha256(stale_payload)
    _write(paths.record_review, _json_bytes(review))
    manifest_before = paths.manifest.read_bytes()
    with pytest.raises(publisher.AuthorizationError, match="artifact binding"):
        publisher.publish()
    assert not paths.destination.exists()
    assert paths.manifest.read_bytes() == manifest_before
    assert not any(
        child.name.startswith(
            (publisher.ARTIFACT_STAGE_PREFIX, publisher.MANIFEST_TEMPORARY_PREFIX)
        )
        for parent in (paths.data_root, paths.destination.parent)
        for child in parent.iterdir()
    )


def test_untrusted_archive_validator_rejects_simple_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publisher, "EXPECTED_ARCHIVE_BYTES", 4)
    monkeypatch.setattr(publisher, "EXPECTED_ARCHIVE_SHA256", _sha256(b"good"))
    with pytest.raises(publisher.CandidateError, match="whole-byte"):
        publisher._validate_untrusted_archive(b"evil")
    with pytest.raises(publisher.CandidateError, match="byte size"):
        publisher._validate_untrusted_archive(b"bad")


@pytest.mark.skipif(
    os.environ.get("CHAPTER06_RUN_LIVE_PUBLISHER_TEST") != "1",
    reason="the live two-top-level reconstruction is run explicitly in acceptance QA",
)
def test_live_verify_only_rebuilds_four_fresh_children_without_repo_delta() -> None:
    paths = publisher._runtime_paths()
    manifest_before = paths.manifest.read_bytes()
    destination_before = paths.destination.exists()
    report = publisher.verify_only()
    assert report["top_level_writer_invocations"] == 2
    assert report["archive"]["sha256"] == publisher.EXPECTED_ARCHIVE_SHA256
    assert report["archive"]["member_count"] == 19
    assert report["accepted_writer_topology"]["cache_entries_after_each_child"] == 37
    assert paths.manifest.read_bytes() == manifest_before
    assert paths.destination.exists() == destination_before
