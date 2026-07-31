"""Adversarial tests for the fail-closed Chapter 6 synthesis publisher."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from scripts import build_chapter06_synthesis_golden as publisher
from scripts import chapter06_synthesis_compact_writer as writer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLISHER_PATH = REPOSITORY_ROOT / "scripts/build_chapter06_synthesis_golden.py"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_file_snapshot() -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): (
            path.stat().st_size,
            _hash(path),
        )
        for path in sorted((REPOSITORY_ROOT / "data").rglob("*"))
        if path.is_file()
    }


def _canonical_directory_snapshot() -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): (
            path.stat().st_mode,
            path.stat().st_ino,
        )
        for path in sorted((REPOSITORY_ROOT / "data").rglob("*"))
        if path.is_dir()
    }


def _array_record(
    array: np.ndarray,
    *,
    unit: str = "accepted compact-member unit",
    axes: list[str] | None = None,
) -> dict[str, object]:
    if axes is None:
        axes = [f"axis_{index}" for index in range(array.ndim)]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "unit": unit,
        "sha256": hashlib.sha256(
            np.ascontiguousarray(array).tobytes(order="C")
        ).hexdigest(),
        "axes": axes,
        "ownership": "synthesis comparison golden",
    }


def _template(
    arrays: dict[str, np.ndarray] | None = None,
    *,
    member_metadata: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    if arrays is None:
        arrays = {
            f"member_{index:03d}": np.asarray(index, dtype=np.int64)
            for index in range(publisher.EXPECTED_MEMBER_COUNT)
        }
    return {
        "path": publisher.DESTINATION_RELATIVE_PATH,
        "role": "golden",
        "format": "npz",
        "scope": (
            "Comparison-only Chapter 6 synthesis one-line oracle; opened "
            "only after the reader-built result exists."
        ),
        "builder": publisher.PUBLISHER_RELATIVE_PATH,
        "builder_sha256": "1" * 64,
        "publisher_contract_sha256": publisher.CONTRACT_SHA256,
        "publisher_contract_acceptance_sha256": (publisher.CONTRACT_AUDIT_SHA256),
        "candidate_byte_acceptance_sha256": (publisher.CANDIDATE_ACCEPTANCE_SHA256),
        "atmosphere_candidate_byte_acceptance_sha256": (
            publisher.ATMOSPHERE_CANDIDATE_ACCEPTANCE_SHA256
        ),
        "publisher_acceptance_sha256": "3" * 64,
        "writer": publisher.WRITER_RELATIVE_PATH,
        "writer_sha256": publisher.FIXED_UPSTREAM_IDENTITIES["writer"][1],
        "writer_tests_sha256": publisher.FIXED_UPSTREAM_IDENTITIES["writer_tests"][1],
        "writer_candidate_sha256": publisher.FIXED_UPSTREAM_IDENTITIES[
            "writer_candidate"
        ][1],
        "writer_acceptance_sha256": publisher.FIXED_UPSTREAM_IDENTITIES[
            "writer_acceptance"
        ][1],
        "publication_acceptance_sha256": (publisher.LATE_AUTHORIZATION_SHA256),
        "publication_record_review_sha256": (publisher.LATE_RECORD_REVIEW_SHA256),
        "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
        "source_data_snapshot_sha256": "8" * 64,
        "archive_kind": publisher.EXPECTED_ARCHIVE_KIND,
        "comparison_only": True,
        "compact_schema_version": publisher.EXPECTED_COMPACT_SCHEMA_VERSION,
        "npy_member_format_version": "2.0",
        "compact_schema_digest": publisher.EXPECTED_COMPACT_SCHEMA_DIGEST,
        "compact_payload_fingerprint": (publisher.EXPECTED_COMPACT_PAYLOAD_FINGERPRINT),
        "raw_ownership_digest": publisher.EXPECTED_RAW_OWNERSHIP_DIGEST,
        "raw_member_count": publisher.EXPECTED_RAW_MEMBER_COUNT,
        "raw_schema_digest": publisher.EXPECTED_RAW_SCHEMA_DIGEST,
        "raw_physical_fingerprint": (publisher.EXPECTED_RAW_PHYSICAL_FINGERPRINT),
        "raw_full_fingerprint": publisher.EXPECTED_RAW_FULL_FINGERPRINT,
        "upstream_identities": {
            label: {"path": path, "sha256": digest}
            for label, (
                path,
                digest,
            ) in publisher.FIXED_UPSTREAM_IDENTITIES.items()
        },
        "reproducibility_environment": deepcopy(publisher.REPRODUCIBILITY_ENVIRONMENT),
        "arrays": {
            name: _array_record(
                arrays[name],
                unit=(
                    str(member_metadata[name]["unit"])
                    if member_metadata is not None
                    else "accepted compact-member unit"
                ),
                axes=(
                    list(member_metadata[name]["axes"])  # type: ignore[arg-type]
                    if member_metadata is not None
                    else None
                ),
            )
            for name in sorted(arrays)
        },
        "sha256": publisher.EXPECTED_ARCHIVE_SHA256,
        "bytes": publisher.EXPECTED_ARCHIVE_BYTES,
    }


def _identity_records() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for label, (path, digest) in publisher.FIXED_UPSTREAM_IDENTITIES.items():
        result[label] = {"path": path, "sha256": digest}
    result["publisher"] = {
        "path": publisher.PUBLISHER_RELATIVE_PATH,
        "sha256": "1" * 64,
    }
    result["publisher_tests"] = {
        "path": publisher.PUBLISHER_TESTS_RELATIVE_PATH,
        "sha256": "2" * 64,
    }
    result["publisher_acceptance"] = {
        "path": publisher.PUBLISHER_ACCEPTANCE_RELATIVE_PATH,
        "sha256": "3" * 64,
    }
    return result


def _authorization(template: dict[str, object] | None = None) -> dict[str, object]:
    if template is None:
        template = _template()
    return {
        "schema_version": 1,
        "record_kind": publisher.AUTHORIZATION_RECORD_KIND,
        "lane": "synthesis",
        "role": "golden",
        "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
        "identities": _identity_records(),
        "artifact": {
            "path": publisher.DESTINATION_RELATIVE_PATH,
            "filename": publisher.DESTINATION_FILENAME,
            "role": "golden",
            "purpose": "comparison-only",
            "bytes": publisher.EXPECTED_ARCHIVE_BYTES,
            "sha256": publisher.EXPECTED_ARCHIVE_SHA256,
            "member_count": publisher.EXPECTED_MEMBER_COUNT,
            "archive_kind": publisher.EXPECTED_ARCHIVE_KIND,
            "comparison_only": True,
            "compact_schema_version": (publisher.EXPECTED_COMPACT_SCHEMA_VERSION),
            "npy_member_format_version": "2.0",
            "schema_digest": publisher.EXPECTED_COMPACT_SCHEMA_DIGEST,
            "payload_fingerprint": (publisher.EXPECTED_COMPACT_PAYLOAD_FINGERPRINT),
            "raw_ownership_digest": (publisher.EXPECTED_RAW_OWNERSHIP_DIGEST),
            "raw_member_count": publisher.EXPECTED_RAW_MEMBER_COUNT,
            "raw_schema_digest": publisher.EXPECTED_RAW_SCHEMA_DIGEST,
            "raw_physical_fingerprint": (publisher.EXPECTED_RAW_PHYSICAL_FINGERPRINT),
            "raw_full_fingerprint": (publisher.EXPECTED_RAW_FULL_FINGERPRINT),
            "contains_copied_input_state": False,
            "contains_atmosphere_lane_members": False,
        },
        "prepublication_manifest": {
            "path": publisher.MANIFEST_RELATIVE_PATH,
            "sha256": "4" * 64,
            "schema_version": 1,
            "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
            "ordered_entry_path_digest": "5" * 64,
            "destination_entry_absent": True,
            "atmosphere_phase": "M1_registered_fixture",
            "atmosphere_entry_path": (publisher.ATMOSPHERE_DESTINATION_RELATIVE_PATH),
            "atmosphere_artifact_sha256": "6" * 64,
            "atmosphere_entry_digest": "7" * 64,
        },
        "manifest_entry_template": template,
        "manifest_entry_template_sha256": publisher.sha256_bytes(
            publisher._template_bytes(template)
        ),
        "source_data_snapshot_sha256": "8" * 64,
        "destination_parent_policy": publisher.DESTINATION_PARENT_POLICY,
        "no_replace_primitive": publisher.NO_REPLACE_PRIMITIVE,
    }


def _authority(template: dict[str, object]) -> publisher.Authority:
    authorization = {"manifest_entry_template": template}
    return publisher.Authority(
        authorization=authorization,
        authorization_bytes=b"authorization",
        authorization_sha256="a" * 64,
        review={},
        review_bytes=b"review",
        review_sha256="b" * 64,
    )


def _fake_snapshot() -> publisher.DataSnapshot:
    return publisher.DataSnapshot(
        directories=(),
        files=(),
        manifest_files=(),
        nonmanifest_support=(),
        manifest_sha256="1" * 64,
        manifest_device=1,
        manifest_inode=2,
        ordered_entry_path_digest="2" * 64,
        aggregate_sha256="3" * 64,
    )


def _small_arrays() -> dict[str, np.ndarray]:
    return {
        "a": np.asarray([1.0, 2.0], dtype=np.float64),
        "b": np.asarray(3, dtype=np.int64),
    }


def _zip_with(
    arrays: dict[str, np.ndarray],
    *,
    compression: int = zipfile.ZIP_STORED,
    reverse: bool = False,
    archive_comment: bytes = b"",
    member_comment: bytes = b"",
    date_time: tuple[int, int, int, int, int, int] = (
        1980,
        1,
        1,
        0,
        0,
        0,
    ),
    external_attr: int = publisher.EXPECTED_ZIP_EXTERNAL_ATTR,
) -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        archive.comment = archive_comment
        names = sorted(arrays, reverse=reverse)
        for name in names:
            info = publisher._zip_info(f"{name}.npy")
            info.compress_type = compression
            info.date_time = date_time
            info.external_attr = external_attr
            info.comment = member_comment
            archive.writestr(
                info,
                publisher._npy_bytes(arrays[name]),
                compress_type=compression,
            )
    return stream.getvalue()


class FixedBoundaryTests(unittest.TestCase):
    """The real entry points expose no caller-selected filesystem authority."""

    def test_exact_contract_chain_and_canonical_paths(self) -> None:
        self.assertEqual(
            _hash(REPOSITORY_ROOT / publisher.CONTRACT_RELATIVE_PATH),
            publisher.CONTRACT_SHA256,
        )
        self.assertEqual(
            _hash(REPOSITORY_ROOT / publisher.CONTRACT_AUDIT_RELATIVE_PATH),
            publisher.CONTRACT_AUDIT_SHA256,
        )
        self.assertEqual(
            _hash(REPOSITORY_ROOT / publisher.CANDIDATE_ACCEPTANCE_RELATIVE_PATH),
            publisher.CANDIDATE_ACCEPTANCE_SHA256,
        )
        self.assertEqual(
            _hash(
                REPOSITORY_ROOT
                / publisher.ATMOSPHERE_CANDIDATE_ACCEPTANCE_RELATIVE_PATH
            ),
            publisher.ATMOSPHERE_CANDIDATE_ACCEPTANCE_SHA256,
        )
        self.assertEqual(
            publisher.PUBLISHER_PATH,
            PUBLISHER_PATH,
        )
        self.assertEqual(
            publisher.DESTINATION_RELATIVE_PATH,
            "data/golden/payne_zero/chapter06/synthesis/"
            "chapter06_synthesis_one_line_cpu_float64_work_"
            "float32_accumulation.npz",
        )
        verified = publisher._verify_fixed_upstream_identities()
        self.assertEqual(set(verified), set(publisher.FIXED_UPSTREAM_IDENTITIES))

    def test_each_fixed_upstream_hash_is_a_hard_gate(self) -> None:
        for label in publisher.FIXED_UPSTREAM_IDENTITIES:
            with self.subTest(label=label):
                changed = dict(publisher.FIXED_UPSTREAM_IDENTITIES)
                path, _digest = changed[label]
                changed[label] = (path, "0" * 64)
                with (
                    mock.patch.object(
                        publisher,
                        "FIXED_UPSTREAM_IDENTITIES",
                        changed,
                    ),
                    self.assertRaises(publisher.PublisherIdentityError),
                ):
                    publisher._verify_fixed_upstream_identities()

    def test_cli_has_verify_authorized_dry_run_or_publish(self) -> None:
        verify = publisher.parse_args(["--verify-only"])
        dry = publisher.parse_args(["--dry-run"])
        publish = publisher.parse_args(["--publish"])
        self.assertTrue(verify.verify_only)
        self.assertTrue(dry.dry_run)
        self.assertTrue(publish.publish)
        for forbidden in (
            "--root",
            "--destination",
            "--output",
            "--force",
            "--replace",
            "--repair",
            "--merge",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(SystemExit):
                    publisher.parse_args([forbidden, "x"])
        self.assertEqual(publisher.dry_run.__code__.co_argcount, 0)
        self.assertEqual(publisher.publish.__code__.co_argcount, 0)
        self.assertEqual(publisher.verify_only.__code__.co_argcount, 0)

    def test_published_authority_verifies_without_writer_or_mutation(self) -> None:
        self.assertTrue(publisher.AUTHORIZATION_PATH.exists())
        self.assertTrue(publisher.RECORD_REVIEW_PATH.exists())
        self.assertTrue(publisher.DESTINATION_PATH.exists())
        files_before = _canonical_file_snapshot()
        directories_before = _canonical_directory_snapshot()
        forbidden = AssertionError("candidate reconstruction/mutation was reached")
        patches = (
            mock.patch.object(
                writer,
                "build_deterministic_compact_archive",
                side_effect=forbidden,
            ),
            mock.patch.object(
                publisher,
                "_ensure_destination_parent",
                side_effect=forbidden,
            ),
            mock.patch.object(
                publisher,
                "_create_stage",
                side_effect=forbidden,
            ),
            mock.patch.object(
                publisher,
                "_replace_manifest",
                side_effect=forbidden,
            ),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            report = publisher.verify_published()
        self.assertEqual(report["state"], "exact_registered")
        self.assertTrue(report["complete_validation"])
        self.assertEqual(
            report["artifact_sha256"],
            publisher.EXPECTED_ARCHIVE_SHA256,
        )
        self.assertEqual(files_before, _canonical_file_snapshot())
        self.assertEqual(directories_before, _canonical_directory_snapshot())
        self.assertTrue(publisher.DESTINATION_PATH.exists())

    def test_published_cli_report_is_machine_readable_and_zero_delta(self) -> None:
        before = _canonical_file_snapshot()
        completed = subprocess.run(
            [sys.executable, str(PUBLISHER_PATH), "--verify-published"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.stderr, "")
        report = json.loads(completed.stdout)
        self.assertEqual(
            report["report_kind"],
            "chapter06_synthesis_postpublication_verification",
        )
        self.assertEqual(report["state"], "exact_registered")
        self.assertTrue(report["complete_validation"])
        self.assertEqual(before, _canonical_file_snapshot())

    def test_dry_run_call_graph_has_no_publication_primitive(self) -> None:
        names = set(publisher.dry_run.__code__.co_names)
        self.assertNotIn("publish", names)
        for mutation in (
            "_ensure_destination_parent",
            "_create_stage",
            "_install_no_replace",
            "_replace_manifest",
        ):
            self.assertNotIn(mutation, names)
        source = PUBLISHER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("os.rename(", source)
        self.assertNotIn("shutil.", source)
        install_source = source[
            source.index("def _link_validated_stage_no_replace") :
        ].split("def _revalidate_source_data_before_manifest", maxsplit=1)[0]
        self.assertIn("os.link(", install_source)
        self.assertNotIn("os.replace(", install_source)

    def test_verify_only_is_authorization_independent_and_mutation_unreachable(
        self,
    ) -> None:
        names = set(publisher.verify_only.__code__.co_names)
        self.assertNotIn("_load_authority", names)
        self.assertNotIn("_prepare_authorized", names)
        for mutation in (
            "_exclusive_data_lock",
            "_ensure_destination_parent",
            "_create_stage",
            "_install_no_replace",
            "_replace_manifest",
            "publish",
        ):
            self.assertNotIn(mutation, names)
        self.assertIn("_build_candidate_twice", names)
        self.assertIn("_verify_source_identity", names)
        self.assertIn("_snapshot_data", names)
        candidate = publisher.Candidate(
            archive_bytes=b"candidate",
            archive_sha256=hashlib.sha256(b"candidate").hexdigest(),
            summary={},
            arrays={},
        )
        snapshot = _fake_snapshot()
        with (
            mock.patch.object(publisher, "_require_canonical_execution"),
            mock.patch.object(
                publisher,
                "_verify_fixed_upstream_identities",
                return_value={"upstream": "exact"},
            ),
            mock.patch.object(
                publisher,
                "_verify_source_identity",
                return_value={"source": "exact"},
            ),
            mock.patch.object(
                publisher,
                "_snapshot_data",
                return_value=snapshot,
            ),
            mock.patch.object(
                publisher,
                "_build_candidate_twice",
                return_value=candidate,
            ) as build,
            mock.patch.object(
                publisher,
                "_target_state",
                return_value="absent",
            ),
            mock.patch.object(
                publisher,
                "_verification_report",
                return_value={"decision": "verified"},
            ),
            mock.patch.object(
                publisher,
                "_load_authority",
                side_effect=AssertionError("authorization was read"),
            ),
        ):
            self.assertEqual(
                publisher.verify_only(),
                {"decision": "verified"},
            )
        build.assert_called_once_with(None)

    def test_fresh_validator_is_complete_and_internal(self) -> None:
        names = set(publisher._registered_validation_report.__code__.co_names)
        for required in (
            "_prepare_authorized",
            "_read_repository_file",
            "_entry_digest",
            "_member_metadata_digest",
        ):
            self.assertIn(required, names)
        source = PUBLISHER_PATH.read_text(encoding="utf-8")
        fresh = source[source.index("def _fresh_validate_expected_final") :].split(
            "def publish", maxsplit=1
        )[0]
        self.assertIn("--internal-validate-published", fresh)


class StrictSchemaTests(unittest.TestCase):
    """Every trust object is duplicate-free, ordered, and lane-exact."""

    def test_strict_json_rejects_duplicate_nonfinite_and_nonobject(self) -> None:
        hostile = (
            b'{"x":1,"x":2}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b"[]",
            b"\xff",
        )
        for payload in hostile:
            with self.subTest(payload=payload):
                with self.assertRaises(publisher.PublisherSchemaError):
                    publisher._strict_json(payload, label="hostile")

    def test_identity_path_rejects_alias_and_escape_forms(self) -> None:
        accepted = "data/golden/payne_zero/chapter06/synthesis/x.npz"
        self.assertEqual(
            publisher._identity_path(accepted, label="path"),
            accepted,
        )
        for value in (
            "",
            "/absolute",
            ".",
            "..",
            "data/../design",
            "data//x",
            "data\\x",
            "data/%2e%2e/x",
            "dáta/x",
        ):
            with self.subTest(value=value):
                with self.assertRaises(publisher.PublisherSchemaError):
                    publisher._identity_path(value, label="path")

    def test_repository_reader_rejects_symlinked_parent_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            safe = root / "safe"
            safe.mkdir()
            (safe / "file").write_bytes(b"safe")
            (root / "link").symlink_to(safe, target_is_directory=True)
            with mock.patch.object(
                publisher,
                "CANONICAL_REPOSITORY_ROOT",
                root,
            ):
                payload, _details = publisher._read_repository_file(
                    "safe/file",
                    label="safe",
                )
                self.assertEqual(payload, b"safe")
                with self.assertRaises(publisher.PublisherIdentityError):
                    publisher._read_repository_file(
                        "link/file",
                        label="symlink parent",
                    )

    def test_authorization_exact_schema_and_role_contract(self) -> None:
        record = _authorization()
        payload = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        with mock.patch.object(
            publisher,
            "_validate_identity_record",
            return_value="0" * 64,
        ):
            validated = publisher._validate_authorization(payload)
        self.assertEqual(validated, record)

        mutations: list[tuple[str, dict[str, object]]] = []
        wrong_role = deepcopy(record)
        wrong_role["role"] = "fixture"
        mutations.append(("role", wrong_role))
        wrong_lane = deepcopy(record)
        wrong_lane["lane"] = "atmosphere"
        mutations.append(("lane", wrong_lane))
        copied = deepcopy(record)
        copied["artifact"]["contains_copied_input_state"] = True  # type: ignore[index]
        mutations.append(("copied input", copied))
        wrong_primitive = deepcopy(record)
        wrong_primitive["no_replace_primitive"] = "rename"
        mutations.append(("replace primitive", wrong_primitive))
        stale_atmosphere = deepcopy(record)
        stale_atmosphere["prepublication_manifest"]["atmosphere_phase"] = "M0"  # type: ignore[index]
        mutations.append(("atmosphere phase", stale_atmosphere))
        unknown = deepcopy(record)
        unknown["unknown"] = True
        mutations.append(("unknown key", unknown))
        reordered = OrderedDict(record)
        reordered.move_to_end("schema_version")
        mutations.append(("reordered key", dict(reordered)))
        for label, mutation in mutations:
            with self.subTest(label=label):
                encoded = json.dumps(
                    mutation,
                    separators=(",", ":"),
                ).encode()
                with mock.patch.object(
                    publisher,
                    "_validate_identity_record",
                    return_value="0" * 64,
                ):
                    with self.assertRaises(publisher.PublisherError):
                        publisher._validate_authorization(encoded)

    def test_authorization_duplicate_key_rejects_before_identity_access(self) -> None:
        payload = b'{"schema_version":1,"schema_version":1}'
        with mock.patch.object(
            publisher,
            "_validate_identity_record",
            side_effect=AssertionError("identity access reached"),
        ):
            with self.assertRaises(publisher.PublisherSchemaError):
                publisher._validate_authorization(payload)

    def test_record_review_is_exact_ordered_and_acyclic(self) -> None:
        authorization = _authorization()
        authorization_sha = "a" * 64
        review = {
            "schema_version": 1,
            "record_kind": publisher.RECORD_REVIEW_KIND,
            "authorization_path": publisher.AUTHORIZATION_RELATIVE_PATH,
            "authorization_sha256": authorization_sha,
            "candidate_byte_acceptance_path": (
                publisher.CANDIDATE_ACCEPTANCE_RELATIVE_PATH
            ),
            "candidate_byte_acceptance_sha256": (publisher.CANDIDATE_ACCEPTANCE_SHA256),
            "publisher_acceptance_path": (publisher.PUBLISHER_ACCEPTANCE_RELATIVE_PATH),
            "publisher_acceptance_sha256": "3" * 64,
            "manifest_entry_template_sha256": authorization[
                "manifest_entry_template_sha256"
            ],
            "disposition": "ACCEPT",
        }
        payload = json.dumps(review, separators=(",", ":")).encode()
        self.assertEqual(
            publisher._validate_record_review(
                payload,
                authorization_sha256=authorization_sha,
                authorization=authorization,
            ),
            review,
        )
        for label, mutation in (
            ("wrong auth", {**review, "authorization_sha256": "b" * 64}),
            ("reject", {**review, "disposition": "REJECT"}),
            ("unknown", {**review, "review_sha256": "c" * 64}),
        ):
            with self.subTest(label=label):
                with self.assertRaises(publisher.PublisherError):
                    publisher._validate_record_review(
                        json.dumps(mutation, separators=(",", ":")).encode(),
                        authorization_sha256=authorization_sha,
                        authorization=authorization,
                    )


class ManifestContractTests(unittest.TestCase):
    """The only allowed manifest delta is append-last and two substitutions."""

    def test_manifest_roundtrip_preserves_intentionally_unsorted_entries(self) -> None:
        manifest = {
            "schema_version": 1,
            "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
            "entries": [
                {"path": "data/z.npz", "role": "fixture"},
                {"path": "data/a.npz", "role": "static"},
            ],
        }
        payload = publisher._manifest_bytes(manifest)
        parsed = publisher._parse_manifest(payload)
        self.assertEqual(parsed, manifest)
        self.assertEqual(
            [entry["path"] for entry in parsed["entries"]],
            ["data/z.npz", "data/a.npz"],
        )
        wrong_top_order = {
            "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
            "schema_version": 1,
            "entries": [],
        }
        with self.assertRaises(publisher.PublisherSchemaError):
            publisher._parse_manifest(publisher._manifest_bytes(wrong_top_order))
        duplicate = deepcopy(manifest)
        duplicate["entries"].append(deepcopy(duplicate["entries"][0]))
        with self.assertRaises(publisher.PublisherSchemaError):
            publisher._parse_manifest(publisher._manifest_bytes(duplicate))

    def test_template_requires_exact_two_complete_placeholders(self) -> None:
        template = _template()
        self.assertIs(
            publisher._validate_template_shape(template),
            template,
        )
        hostile: list[tuple[str, dict[str, object]]] = []
        missing = deepcopy(template)
        missing["publication_record_review_sha256"] = "a" * 64
        hostile.append(("missing", missing))
        duplicate = deepcopy(template)
        duplicate["extra"] = publisher.LATE_AUTHORIZATION_SHA256
        hostile.append(("duplicate", duplicate))
        nested = deepcopy(template)
        nested["publication_acceptance_sha256"] = {
            "value": publisher.LATE_AUTHORIZATION_SHA256
        }
        hostile.append(("nested", nested))
        reordered = OrderedDict(template)
        reordered.move_to_end("publication_acceptance_sha256")
        hostile.append(("reordered", dict(reordered)))
        bad_array_order = deepcopy(template)
        first_name = next(iter(bad_array_order["arrays"]))  # type: ignore[arg-type]
        first = bad_array_order["arrays"][first_name]  # type: ignore[index]
        first["dtype"], first["shape"] = first["shape"], first["dtype"]
        hostile.append(("array metadata", bad_array_order))
        for label, value in hostile:
            with self.subTest(label=label):
                with self.assertRaises(publisher.PublisherSchemaError):
                    publisher._validate_template_shape(value)

    def test_units_and_axes_are_nonempty_and_exactly_digest_bound(self) -> None:
        arrays = {
            f"member_{index:03d}": (
                np.asarray([index], dtype=np.int64)
                if index == 0
                else np.asarray(index, dtype=np.int64)
            )
            for index in range(publisher.EXPECTED_MEMBER_COUNT)
        }
        template = _template(arrays)
        first_name = next(iter(template["arrays"]))  # type: ignore[arg-type]
        empty_unit = deepcopy(template)
        empty_unit["arrays"][first_name]["unit"] = ""  # type: ignore[index]
        with self.assertRaises(publisher.PublisherSchemaError):
            publisher._validate_template_shape(empty_unit)
        empty_axis = deepcopy(template)
        empty_axis["arrays"][first_name]["axes"][0] = ""  # type: ignore[index]
        with self.assertRaises(publisher.PublisherSchemaError):
            publisher._validate_template_shape(empty_axis)

        records = template["arrays"]
        digest = publisher._member_metadata_digest(records)  # type: ignore[arg-type]
        with mock.patch.object(
            publisher,
            "EXPECTED_MEMBER_METADATA_DIGEST",
            digest,
        ):
            publisher._validate_expected_member_metadata(records)  # type: ignore[arg-type]
            wrong = deepcopy(records)
            wrong[first_name]["unit"] = "incorrect unit"  # type: ignore[index]
            with self.assertRaises(publisher.PublisherSchemaError):
                publisher._validate_expected_member_metadata(wrong)

    def test_realization_and_append_delete_last_are_exact(self) -> None:
        template = _template()
        realized = publisher._realize_template(
            template,
            authorization_sha256="a" * 64,
            review_sha256="b" * 64,
        )
        self.assertEqual(
            template["publication_acceptance_sha256"],
            publisher.LATE_AUTHORIZATION_SHA256,
        )
        self.assertEqual(realized["publication_acceptance_sha256"], "a" * 64)
        self.assertEqual(
            realized["publication_record_review_sha256"],
            "b" * 64,
        )
        premanifest = {
            "schema_version": 1,
            "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
            "entries": [
                {"path": "data/z.npz", "role": "fixture"},
                {"path": "data/a.npz", "role": "static"},
            ],
        }
        before = publisher._manifest_bytes(premanifest)
        after = publisher._construct_postmanifest(
            premanifest,
            premanifest_bytes=before,
            entry=realized,
        )
        parsed = publisher._parse_manifest(after)
        self.assertEqual(parsed["entries"][:-1], premanifest["entries"])
        self.assertEqual(parsed["entries"][-1], realized)
        stripped = deepcopy(parsed)
        stripped["entries"].pop()
        self.assertEqual(publisher._manifest_bytes(stripped), before)

    def test_registered_manifest_reconstructs_authorized_m1(self) -> None:
        template = _template()
        authority = _authority(template)
        atmosphere = {
            "path": publisher.ATMOSPHERE_DESTINATION_RELATIVE_PATH,
            "role": "fixture",
            "sha256": "c" * 64,
            "bytes": 10,
        }
        base = {
            "schema_version": 1,
            "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
            "entries": [atmosphere],
        }
        base_bytes = publisher._manifest_bytes(base)
        authority.authorization.update(
            {
                "prepublication_manifest": {
                    "sha256": publisher.sha256_bytes(base_bytes),
                    "ordered_entry_path_digest": (
                        publisher._ordered_path_digest(
                            entry["path"] for entry in base["entries"]
                        )
                    ),
                },
                "source_data_snapshot_sha256": "d" * 64,
            }
        )
        realized = publisher._realize_template(
            template,
            authorization_sha256=authority.authorization_sha256,
            review_sha256=authority.review_sha256,
        )
        current = deepcopy(base)
        current["entries"].append(realized)
        current_bytes = publisher._manifest_bytes(current)
        snapshot = _fake_snapshot()
        authority.authorization["source_data_snapshot_sha256"] = (
            publisher._source_data_snapshot_digest(
                snapshot,
                {"source": "exact"},
                manifest_sha256=publisher.sha256_bytes(base_bytes),
                ordered_entry_path_digest=publisher._ordered_path_digest(
                    entry["path"] for entry in base["entries"]
                ),
            )
        )
        with (
            mock.patch.object(
                publisher,
                "_read_repository_file",
                return_value=(current_bytes, mock.Mock()),
            ),
            mock.patch.object(publisher, "_verify_atmosphere_phase"),
            mock.patch.object(
                publisher,
                "_snapshot_data",
                return_value=snapshot,
            ),
            mock.patch.object(
                publisher,
                "_verify_source_identity",
                return_value={"source": "exact"},
            ),
        ):
            (
                recovered,
                recovered_bytes,
                actual_snapshot,
                _sources,
                lifecycle,
                actual_current,
            ) = publisher._validate_prepublication_state(authority)
        self.assertEqual(recovered, base)
        self.assertEqual(recovered_bytes, base_bytes)
        self.assertEqual(actual_snapshot, snapshot)
        self.assertEqual(lifecycle, "registered")
        self.assertEqual(actual_current, current_bytes)

    def test_atmosphere_phase_requires_one_exact_fixture(self) -> None:
        artifact = b"atmosphere fixture bytes"
        entry = {
            "path": publisher.ATMOSPHERE_DESTINATION_RELATIVE_PATH,
            "role": "fixture",
            "sha256": publisher.sha256_bytes(artifact),
            "bytes": len(artifact),
        }
        manifest = {
            "schema_version": 1,
            "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
            "entries": [entry],
        }
        premanifest = {
            "atmosphere_artifact_sha256": publisher.sha256_bytes(artifact),
            "atmosphere_entry_digest": publisher._entry_digest(entry),
        }
        with mock.patch.object(
            publisher,
            "_read_repository_file",
            return_value=(artifact, mock.Mock()),
        ):
            publisher._verify_atmosphere_phase(manifest, premanifest)
        wrong_role = deepcopy(manifest)
        wrong_role["entries"][0]["role"] = "golden"
        with self.assertRaises(publisher.PublicationGateError):
            publisher._verify_atmosphere_phase(wrong_role, premanifest)
        duplicate = deepcopy(manifest)
        duplicate["entries"].append(deepcopy(entry))
        with self.assertRaises(publisher.PublicationGateError):
            publisher._verify_atmosphere_phase(duplicate, premanifest)


class ArchiveBoundaryTests(unittest.TestCase):
    """Untrusted ZIP/NPY bytes are accepted only in one canonical form."""

    def test_manifest_member_hash_is_raw_c_order_bytes_only(self) -> None:
        array = np.asarray([1.0, 2.0], dtype=np.float64)
        expected = hashlib.sha256(
            np.ascontiguousarray(array).tobytes(order="C")
        ).hexdigest()
        self.assertEqual(
            expected,
            "dc91ce9a50ddc828740aa26743716897fdb2bb64f1db662fe263a59be56145ae",
        )
        self.assertEqual(publisher._member_sha256(array), expected)

    def test_small_canonical_archive_roundtrip(self) -> None:
        arrays = _small_arrays()
        encoded = publisher._encode_canonical_npz(arrays)
        decoded = publisher._decode_canonical_npz(encoded)
        self.assertEqual(tuple(decoded), tuple(arrays))
        for name in arrays:
            self.assertEqual(decoded[name].dtype, arrays[name].dtype)
            self.assertEqual(decoded[name].shape, arrays[name].shape)
            self.assertEqual(decoded[name].tobytes(), arrays[name].tobytes())
        self.assertEqual(publisher._encode_canonical_npz(decoded), encoded)

    def test_archive_metadata_and_corruption_matrix(self) -> None:
        arrays = _small_arrays()
        canonical = publisher._encode_canonical_npz(arrays)
        hostile = {
            "truncated": canonical[:-1],
            "trailing bytes": canonical + b"x",
            "compression": _zip_with(arrays, compression=zipfile.ZIP_DEFLATED),
            "order": _zip_with(arrays, reverse=True),
            "timestamp": _zip_with(
                arrays,
                date_time=(1981, 1, 1, 0, 0, 0),
            ),
            "permissions": _zip_with(arrays, external_attr=0o100644 << 16),
            "member comment": _zip_with(arrays, member_comment=b"x"),
            "archive comment": _zip_with(arrays, archive_comment=b"x"),
        }
        for label, payload in hostile.items():
            with self.subTest(label=label):
                with self.assertRaises(publisher.PublisherSchemaError):
                    publisher._decode_canonical_npz(payload)

    def test_unsafe_object_noncontiguous_and_member_names_reject(self) -> None:
        hostile = (
            {"../escape": np.asarray(1)},
            {"x.npy": np.asarray(1)},
            {"x": np.asarray([object()])},
            {"x": np.arange(8).reshape(2, 4)[:, ::2]},
        )
        for arrays in hostile:
            with self.subTest(names=tuple(arrays)):
                with self.assertRaises(publisher.PublisherSchemaError):
                    publisher._encode_canonical_npz(arrays)

    def test_wrong_npy_version_and_duplicate_members_reject(self) -> None:
        arrays = {"a": np.asarray([1.0], dtype=np.float64)}
        wrong_version_stream = BytesIO()
        np.lib.format.write_array(
            wrong_version_stream,
            arrays["a"],
            version=(1, 0),
            allow_pickle=False,
        )
        archive_stream = BytesIO()
        with zipfile.ZipFile(archive_stream, "w") as archive:
            archive.writestr(
                publisher._zip_info("a.npy"),
                wrong_version_stream.getvalue(),
            )
        with self.assertRaises(publisher.PublisherSchemaError):
            publisher._decode_canonical_npz(archive_stream.getvalue())

        duplicate_stream = BytesIO()
        with (
            self.assertWarns(UserWarning),
            zipfile.ZipFile(duplicate_stream, "w") as archive,
        ):
            info = publisher._zip_info("a.npy")
            archive.writestr(info, publisher._npy_bytes(arrays["a"]))
            archive.writestr(info, publisher._npy_bytes(arrays["a"]))
        with self.assertRaises(publisher.PublisherSchemaError):
            publisher._decode_canonical_npz(duplicate_stream.getvalue())

    @unittest.skipUnless(
        writer.PINNED_ROOT.is_dir(),
        "pinned Payne Zero checkout is unavailable",
    )
    def test_exact_candidate_two_call_gate_semantics_and_zero_delta(self) -> None:
        files_before = _canonical_file_snapshot()
        captured_schemas: list[tuple[object, ...]] = []
        original_assembler = writer.compact.assemble_compact_candidate

        def capture_schema(raw: object) -> object:
            assembly = original_assembler(raw)  # type: ignore[arg-type]
            captured_schemas.append(assembly.schema)
            return assembly

        with (
            publisher._accepted_writer_environment(),
            mock.patch.object(
                writer.compact,
                "assemble_compact_candidate",
                side_effect=capture_schema,
            ),
        ):
            result = writer.build_deterministic_compact_archive()
        self.assertEqual(len(captured_schemas), 2)
        self.assertEqual(captured_schemas[0], captured_schemas[1])
        member_metadata = {
            spec.name: {"unit": spec.unit, "axes": list(spec.axes)}
            for spec in captured_schemas[0]
        }
        self.assertEqual(
            publisher._member_metadata_digest(member_metadata),
            publisher.EXPECTED_MEMBER_METADATA_DIGEST,
        )
        arrays = publisher._decode_canonical_npz(result.archive_bytes)
        template = _template(arrays, member_metadata=member_metadata)
        authority = _authority(template)
        with mock.patch.object(
            writer,
            "build_deterministic_compact_archive",
            return_value=result,
        ) as build:
            candidate = publisher._build_candidate_twice(authority)
        self.assertEqual(build.call_count, 2)
        self.assertEqual(candidate.archive_sha256, publisher.EXPECTED_ARCHIVE_SHA256)
        self.assertEqual(len(candidate.archive_bytes), publisher.EXPECTED_ARCHIVE_BYTES)
        self.assertEqual(len(candidate.arrays), publisher.EXPECTED_MEMBER_COUNT)
        rendered_summary = json.dumps(
            candidate.summary,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertNotIn("child_pid", rendered_summary)
        self.assertNotIn("cache_root_sha256", rendered_summary)
        self.assertTrue(
            candidate.summary["capture_topology"]["capture_cache_roots_empty_after"]
        )
        self.assertEqual(files_before, _canonical_file_snapshot())

    def test_two_call_gate_rejects_byte_disagreement(self) -> None:
        summary = {
            "capture_process_evidence": {
                field: True
                for field in (
                    "capture_origins_distinct",
                    "capture_child_processes_distinct",
                    "capture_cache_roots_distinct",
                    "capture_cache_roots_external",
                    "capture_cache_roots_nonsymlink",
                    "capture_cache_roots_empty_before",
                    "capture_cache_roots_empty_after",
                    "capture_cache_roots_disposed",
                )
            }
        }
        first = writer.DeterministicCompactArchive(b"a", summary)
        second = writer.DeterministicCompactArchive(b"b", summary)
        with mock.patch.object(
            writer,
            "build_deterministic_compact_archive",
            side_effect=[first, second],
        ):
            with self.assertRaisesRegex(
                publisher.PublisherSchemaError,
                "top-level candidate builds disagree",
            ):
                publisher._build_candidate_twice(_authority(_template()))


class FilesystemPrimitiveTests(unittest.TestCase):
    """Private isolated harnesses test no-replace, durability, and quarantine."""

    def _exercise_authorized_manifest_transition(
        self,
        *,
        mutation: str | None,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            fixtures = data / "fixtures"
            destination = root / publisher.DESTINATION_RELATIVE_PATH
            fixtures.mkdir(parents=True)
            destination.parent.mkdir(parents=True)
            for directory in (
                data,
                fixtures,
                data / "golden",
                data / "golden/payne_zero",
                data / "golden/payne_zero/chapter06",
                destination.parent,
            ):
                directory.chmod(0o755)

            readme = data / "README.md"
            readme_before = b"closed nonmanifest support"
            readme.write_bytes(readme_before)
            atmosphere = root / publisher.ATMOSPHERE_DESTINATION_RELATIVE_PATH
            atmosphere_bytes = b"registered atmosphere fixture"
            atmosphere.write_bytes(atmosphere_bytes)
            atmosphere_entry = {
                "path": publisher.ATMOSPHERE_DESTINATION_RELATIVE_PATH,
                "role": "fixture",
                "sha256": hashlib.sha256(atmosphere_bytes).hexdigest(),
                "bytes": len(atmosphere_bytes),
            }
            before_value = {
                "schema_version": 1,
                "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
                "entries": [atmosphere_entry],
            }
            before = publisher._manifest_bytes(before_value)
            manifest = data / "MANIFEST.json"
            manifest.write_bytes(before)
            manifest.chmod(0o644)

            candidate = b"exact isolated synthesis candidate"
            destination.write_bytes(candidate)
            destination.chmod(0o600)
            target_entry = {
                "path": publisher.DESTINATION_RELATIVE_PATH,
                "role": "golden",
                "sha256": hashlib.sha256(candidate).hexdigest(),
                "bytes": len(candidate),
            }
            after_value = deepcopy(before_value)
            after_value["entries"].append(target_entry)
            after = publisher._manifest_bytes(after_value)

            source = root / "accepted-source.bin"
            source.write_bytes(b"accepted source")

            def source_identity() -> dict[str, str]:
                return {
                    "accepted-source": hashlib.sha256(source.read_bytes()).hexdigest()
                }

            authorization_bytes = b'{"isolated":"authorization"}\n'
            review_bytes = b'{"isolated":"review"}\n'
            authorization_path = root / publisher.AUTHORIZATION_RELATIVE_PATH
            review_path = root / publisher.RECORD_REVIEW_RELATIVE_PATH
            authorization_path.parent.mkdir(parents=True, exist_ok=True)
            authorization_path.write_bytes(authorization_bytes)
            review_path.write_bytes(review_bytes)
            template = _template()
            authority = publisher.Authority(
                authorization={
                    "manifest_entry_template": template,
                    "manifest_entry_template_sha256": publisher.sha256_bytes(
                        publisher._template_bytes(template)
                    ),
                },
                authorization_bytes=authorization_bytes,
                authorization_sha256=hashlib.sha256(authorization_bytes).hexdigest(),
                review={},
                review_bytes=review_bytes,
                review_sha256=hashlib.sha256(review_bytes).hexdigest(),
            )

            base_directories = (
                "data",
                "data/fixtures",
                "data/golden",
                "data/golden/payne_zero",
            )
            replacement_calls = 0
            real_replace = os.replace
            real_create_stage = publisher._create_stage
            rogue_temporary: Path | None = None

            def track_replace(*args: object, **kwargs: object) -> None:
                nonlocal replacement_calls
                replacement_calls += 1
                real_replace(*args, **kwargs)  # type: ignore[arg-type]

            def create_then_mutate(
                *args: object,
                **kwargs: object,
            ) -> tuple[Path, int, os.stat_result]:
                nonlocal rogue_temporary
                result = real_create_stage(*args, **kwargs)  # type: ignore[arg-type]
                if mutation == "source":
                    source.write_bytes(b"changed source")
                elif mutation == "nontarget":
                    readme.write_bytes(b"changed nonmanifest support")
                elif mutation == "second_temporary":
                    rogue_temporary = data / f"{publisher.MANIFEST_TEMP_PREFIX}foreign"
                    rogue_temporary.write_bytes(b"foreign temporary")
                return result

            with (
                mock.patch.object(
                    publisher,
                    "CANONICAL_REPOSITORY_ROOT",
                    root,
                ),
                mock.patch.object(publisher, "DATA_PATH", data),
                mock.patch.object(publisher, "MANIFEST_PATH", manifest),
                mock.patch.object(publisher, "DESTINATION_PATH", destination),
                mock.patch.object(
                    publisher,
                    "PREPUBLICATION_DATA_DIRECTORIES",
                    base_directories,
                ),
                mock.patch.object(
                    publisher,
                    "DATA_README_SHA256",
                    hashlib.sha256(readme_before).hexdigest(),
                ),
                mock.patch.object(
                    publisher,
                    "EXPECTED_ARCHIVE_BYTES",
                    len(candidate),
                ),
                mock.patch.object(
                    publisher,
                    "EXPECTED_ARCHIVE_SHA256",
                    hashlib.sha256(candidate).hexdigest(),
                ),
                mock.patch.object(
                    publisher,
                    "_verify_fixed_upstream_identities",
                    return_value={},
                ),
                mock.patch.object(
                    publisher,
                    "_verify_source_identity",
                    side_effect=source_identity,
                ),
            ):
                before_snapshot = publisher._snapshot_data(
                    allow_exact_unregistered_target=(
                        publisher.DESTINATION_RELATIVE_PATH
                    )
                )
                authority.authorization["source_data_snapshot_sha256"] = (
                    publisher._source_data_snapshot_digest(
                        before_snapshot,
                        source_identity(),
                        manifest_sha256=hashlib.sha256(before).hexdigest(),
                        ordered_entry_path_digest=publisher._ordered_path_digest(
                            entry["path"] for entry in before_value["entries"]
                        ),
                    )
                )
                data_fd = publisher._open_repository_directory(
                    "data",
                    label="data",
                )
                try:
                    with (
                        mock.patch.object(
                            publisher,
                            "_create_stage",
                            side_effect=create_then_mutate,
                        ),
                        mock.patch.object(
                            publisher.os,
                            "replace",
                            side_effect=track_replace,
                        ),
                    ):
                        if mutation is None:
                            publisher._replace_manifest(
                                after,
                                expected_before=before,
                                candidate_bytes=candidate,
                                authority=authority,
                                data_fd=data_fd,
                            )
                        else:
                            with self.assertRaises(publisher.PublicationStateError):
                                publisher._replace_manifest(
                                    after,
                                    expected_before=before,
                                    candidate_bytes=candidate,
                                    authority=authority,
                                    data_fd=data_fd,
                                )
                finally:
                    os.close(data_fd)

            self.assertEqual(destination.read_bytes(), candidate)
            if mutation is None:
                self.assertEqual(replacement_calls, 1)
                self.assertEqual(manifest.read_bytes(), after)
                self.assertFalse(
                    any(
                        child.name.startswith(publisher.MANIFEST_TEMP_PREFIX)
                        for child in data.iterdir()
                    )
                )
            else:
                self.assertEqual(replacement_calls, 0)
                self.assertEqual(manifest.read_bytes(), before)
                if mutation == "second_temporary":
                    self.assertIsNotNone(rogue_temporary)
                    assert rogue_temporary is not None
                    self.assertEqual(
                        rogue_temporary.read_bytes(),
                        b"foreign temporary",
                    )
                else:
                    self.assertFalse(
                        any(
                            child.name.startswith(publisher.MANIFEST_TEMP_PREFIX)
                            for child in data.iterdir()
                        )
                    )

    def test_authorized_manifest_transition_reaches_real_atomic_replace(self) -> None:
        self._exercise_authorized_manifest_transition(mutation=None)

    def test_authorized_manifest_transition_rejects_source_mutation(self) -> None:
        self._exercise_authorized_manifest_transition(mutation="source")

    def test_authorized_manifest_transition_rejects_nontarget_mutation(self) -> None:
        self._exercise_authorized_manifest_transition(mutation="nontarget")

    def test_authorized_manifest_transition_rejects_second_temporary(self) -> None:
        self._exercise_authorized_manifest_transition(mutation="second_temporary")

    def test_write_all_rejects_zero_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with (
                    mock.patch.object(os, "write", return_value=0),
                    self.assertRaises(publisher.PublicationIOError),
                ):
                    publisher._write_all(fd, b"payload")
            finally:
                os.close(fd)

    def test_write_all_rejects_partial_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with (
                    mock.patch.object(os, "write", return_value=2),
                    self.assertRaisesRegex(
                        publisher.PublicationIOError,
                        "short artifact write",
                    ),
                ):
                    publisher._write_all(fd, b"payload")
            finally:
                os.close(fd)

    def test_stage_is_exclusive_mode_0600_and_readback_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            payload = b"candidate"
            stage, fd, details = publisher._create_stage(
                parent,
                payload,
                prefix=publisher.STAGE_PREFIX,
            )
            os.close(fd)
            self.assertEqual(stage.read_bytes(), payload)
            self.assertEqual(stat.S_IMODE(stage.stat().st_mode), 0o600)
            self.assertEqual(stage.stat().st_nlink, 1)
            publisher._unlink_owned_stage(stage, details)
            self.assertFalse(stage.exists())

    def test_stage_mode_is_0600_independent_of_umask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            previous = os.umask(0o777)
            try:
                stage, fd, details = publisher._create_stage(
                    parent,
                    b"candidate",
                    prefix=publisher.STAGE_PREFIX,
                )
            finally:
                os.umask(previous)
            try:
                self.assertEqual(stat.S_IMODE(os.fstat(fd).st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(stage.stat().st_mode), 0o600)
            finally:
                os.close(fd)
                publisher._unlink_owned_stage(stage, details)

    def test_handled_stage_readback_failure_cleans_owned_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            with (
                mock.patch.object(
                    publisher,
                    "_read_all_fd",
                    side_effect=publisher.PublisherIdentityError("readback"),
                ),
                self.assertRaises(publisher.PublisherIdentityError),
            ):
                publisher._create_stage(
                    parent,
                    b"candidate",
                    prefix=publisher.STAGE_PREFIX,
                )
            self.assertFalse(
                any(
                    child.name.startswith(publisher.STAGE_PREFIX)
                    for child in parent.iterdir()
                )
            )

    def test_handled_stage_mode_failure_cleans_owned_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            with (
                mock.patch.object(
                    os,
                    "fchmod",
                    side_effect=OSError("mode failure"),
                ),
                self.assertRaises(publisher.PublicationIOError),
            ):
                publisher._create_stage(
                    parent,
                    b"candidate",
                    prefix=publisher.STAGE_PREFIX,
                )
            self.assertFalse(
                any(
                    child.name.startswith(publisher.STAGE_PREFIX)
                    for child in parent.iterdir()
                )
            )

    def test_stage_name_substitution_rejects_before_no_replace_install(self) -> None:
        payload = b"authorized"
        foreign = b"foreign-race-bytes"
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(publisher, "EXPECTED_ARCHIVE_BYTES", len(payload)),
            mock.patch.object(
                publisher,
                "EXPECTED_ARCHIVE_SHA256",
                publisher.sha256_bytes(payload),
            ),
        ):
            parent = Path(temporary)
            stage, stage_fd, details = publisher._create_stage(
                parent,
                payload,
                prefix=publisher.STAGE_PREFIX,
            )
            original = publisher._link_validated_stage_no_replace

            def substitute_then_link(**kwargs: object) -> None:
                stage.unlink()
                stage.write_bytes(foreign)
                stage.chmod(0o600)
                original(**kwargs)  # type: ignore[arg-type]

            with (
                mock.patch.object(
                    publisher,
                    "_link_validated_stage_no_replace",
                    side_effect=substitute_then_link,
                ),
                self.assertRaises(publisher.PublicationIOError),
            ):
                publisher._install_no_replace(
                    parent,
                    stage=stage,
                    stage_fd=stage_fd,
                    stage_details=details,
                    candidate_bytes=payload,
                )
            self.assertFalse((parent / publisher.DESTINATION_FILENAME).exists())
            self.assertEqual(stage.read_bytes(), foreign)

    def test_no_replace_install_and_exact_race_noop(self) -> None:
        payload = b"candidate"
        digest = publisher.sha256_bytes(payload)
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(publisher, "EXPECTED_ARCHIVE_BYTES", len(payload)),
            mock.patch.object(publisher, "EXPECTED_ARCHIVE_SHA256", digest),
        ):
            parent = Path(temporary)
            stage, fd, details = publisher._create_stage(
                parent,
                payload,
                prefix=publisher.STAGE_PREFIX,
            )
            self.assertEqual(
                publisher._install_no_replace(
                    parent,
                    stage=stage,
                    stage_fd=fd,
                    stage_details=details,
                    candidate_bytes=payload,
                ),
                "installed",
            )
            target = parent / publisher.DESTINATION_FILENAME
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(target.stat().st_nlink, 1)

            next_stage, fd, next_details = publisher._create_stage(
                parent,
                payload,
                prefix=publisher.STAGE_PREFIX,
            )
            self.assertEqual(
                publisher._install_no_replace(
                    parent,
                    stage=next_stage,
                    stage_fd=fd,
                    stage_details=next_details,
                    candidate_bytes=payload,
                ),
                "exact_race_noop",
            )
            self.assertFalse(next_stage.exists())
            self.assertEqual(target.read_bytes(), payload)

    def test_nonexact_race_never_overwrites_and_cleans_owned_stage(self) -> None:
        payload = b"authorized"
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(publisher, "EXPECTED_ARCHIVE_BYTES", len(payload)),
            mock.patch.object(
                publisher,
                "EXPECTED_ARCHIVE_SHA256",
                publisher.sha256_bytes(payload),
            ),
        ):
            parent = Path(temporary)
            target = parent / publisher.DESTINATION_FILENAME
            target.write_bytes(b"attacker")
            stage, fd, details = publisher._create_stage(
                parent,
                payload,
                prefix=publisher.STAGE_PREFIX,
            )
            with self.assertRaises(publisher.PublicationStateError):
                publisher._install_no_replace(
                    parent,
                    stage=stage,
                    stage_fd=fd,
                    stage_details=details,
                    candidate_bytes=payload,
                )
            self.assertEqual(target.read_bytes(), b"attacker")
            self.assertFalse(stage.exists())

    def test_existing_target_symlink_or_multiple_links_rejects(self) -> None:
        candidate = b"candidate"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_bytes(candidate)
            alias = root / "alias"
            os.link(target, alias)
            with mock.patch.object(publisher, "DESTINATION_PATH", target):
                with self.assertRaises(publisher.PublicationStateError):
                    publisher._target_state(candidate)
            alias.unlink()
            target.unlink()
            target.symlink_to(root / "missing")
            with mock.patch.object(publisher, "DESTINATION_PATH", target):
                with self.assertRaises(publisher.PublisherIdentityError):
                    publisher._target_state(candidate)

    def test_retained_parent_swap_rejects_before_no_replace_link(self) -> None:
        payload = b"authorized"
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(publisher, "EXPECTED_ARCHIVE_BYTES", len(payload)),
            mock.patch.object(
                publisher,
                "EXPECTED_ARCHIVE_SHA256",
                publisher.sha256_bytes(payload),
            ),
        ):
            root = Path(temporary)
            parent = root / publisher.DESTINATION_PARENT_RELATIVE_PATH
            parent.mkdir(parents=True)
            with mock.patch.object(
                publisher,
                "CANONICAL_REPOSITORY_ROOT",
                root,
            ):
                parent_fd = publisher._open_repository_directory(
                    publisher.DESTINATION_PARENT_RELATIVE_PATH,
                    label="destination parent",
                )
                try:
                    stage, stage_fd, details = publisher._create_stage(
                        parent,
                        payload,
                        prefix=publisher.STAGE_PREFIX,
                        parent_fd=parent_fd,
                    )
                    moved = parent.with_name("synthesis-moved")
                    parent.rename(moved)
                    parent.mkdir()
                    with self.assertRaises(publisher.PublicationStateError):
                        publisher._install_no_replace(
                            parent,
                            stage=stage,
                            stage_fd=stage_fd,
                            stage_details=details,
                            candidate_bytes=payload,
                            parent_fd=parent_fd,
                        )
                    self.assertFalse((moved / publisher.DESTINATION_FILENAME).exists())
                    self.assertFalse((parent / publisher.DESTINATION_FILENAME).exists())
                    self.assertFalse((moved / stage.name).exists())
                finally:
                    os.close(parent_fd)

    def test_post_link_parent_swap_rejects_before_helper_success(self) -> None:
        payload = b"authorized"
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(publisher, "EXPECTED_ARCHIVE_BYTES", len(payload)),
            mock.patch.object(
                publisher,
                "EXPECTED_ARCHIVE_SHA256",
                publisher.sha256_bytes(payload),
            ),
        ):
            root = Path(temporary)
            parent = root / publisher.DESTINATION_PARENT_RELATIVE_PATH
            parent.mkdir(parents=True)
            moved = parent.with_name("synthesis-moved-after-link")
            with mock.patch.object(
                publisher,
                "CANONICAL_REPOSITORY_ROOT",
                root,
            ):
                parent_fd = publisher._open_repository_directory(
                    publisher.DESTINATION_PARENT_RELATIVE_PATH,
                    label="destination parent",
                )
                try:
                    stage, stage_fd, details = publisher._create_stage(
                        parent,
                        payload,
                        prefix=publisher.STAGE_PREFIX,
                        parent_fd=parent_fd,
                    )
                    real_link = publisher.os.link

                    def link_then_swap(*args: object, **kwargs: object) -> None:
                        real_link(*args, **kwargs)  # type: ignore[arg-type]
                        parent.rename(moved)
                        parent.mkdir()

                    with (
                        mock.patch.object(
                            publisher.os,
                            "link",
                            side_effect=link_then_swap,
                        ),
                        self.assertRaisesRegex(
                            publisher.PublicationStateError,
                            "canonical directory identity changed",
                        ),
                    ):
                        publisher._install_no_replace(
                            parent,
                            stage=stage,
                            stage_fd=stage_fd,
                            stage_details=details,
                            candidate_bytes=payload,
                            parent_fd=parent_fd,
                        )
                    self.assertFalse((parent / publisher.DESTINATION_FILENAME).exists())
                    self.assertEqual(
                        (moved / publisher.DESTINATION_FILENAME).read_bytes(),
                        payload,
                    )
                    self.assertEqual((moved / stage.name).read_bytes(), payload)
                finally:
                    os.close(parent_fd)

    def test_nested_directory_creation_and_owned_reverse_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "data/golden/payne_zero"
            base.mkdir(parents=True)
            with mock.patch.object(
                publisher,
                "CANONICAL_REPOSITORY_ROOT",
                root,
            ):
                parent, created = publisher._ensure_destination_parent()
                self.assertEqual(
                    parent,
                    root / publisher.DESTINATION_PARENT_RELATIVE_PATH,
                )
                self.assertEqual(
                    [item.path.name for item in created],
                    ["chapter06", "synthesis"],
                )
                publisher._cleanup_invocation_directories(created)
                self.assertFalse((base / "chapter06").exists())

    def test_unexpected_crash_left_directory_content_hard_stops(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chapter = root / "data/golden/payne_zero/chapter06"
            chapter.mkdir(parents=True)
            (chapter / "unexpected").write_bytes(b"x")
            with (
                mock.patch.object(
                    publisher,
                    "CANONICAL_REPOSITORY_ROOT",
                    root,
                ),
                self.assertRaises(publisher.PublicationStateError),
            ):
                publisher._ensure_destination_parent()

    def test_nested_directory_fsync_failure_occurs_before_any_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "data/golden/payne_zero"
            base.mkdir(parents=True)
            created: list[publisher.InvocationDirectory] = []
            with (
                mock.patch.object(
                    publisher,
                    "CANONICAL_REPOSITORY_ROOT",
                    root,
                ),
                mock.patch.object(os, "fsync", side_effect=OSError("fsync")),
                self.assertRaises(publisher.PublicationIOError),
            ):
                publisher._ensure_destination_parent(created)
            self.assertFalse(
                (base / "chapter06" / publisher.DESTINATION_FILENAME).exists()
            )
            publisher._cleanup_invocation_directories(created)

    def test_quarantine_inventory_is_inode_bound_and_never_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            stage = data / f"{publisher.STAGE_PREFIX}orphan"
            stage.write_bytes(b"looks exactly valid")
            with mock.patch.object(
                publisher,
                "CANONICAL_REPOSITORY_ROOT",
                root,
            ):
                inventory = publisher._inventory_quarantine(data)
            self.assertEqual(len(inventory), 1)
            self.assertEqual(inventory[0].relative_path, f"data/{stage.name}")
            self.assertEqual(inventory[0].sha256, _hash(stage))
            self.assertTrue(stage.exists())

    def test_manifest_replacement_changes_only_manifest_in_isolated_harness(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            manifest = data / "MANIFEST.json"
            artifact = data / "artifact.npz"
            before = publisher._manifest_bytes(
                {
                    "schema_version": 1,
                    "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
                    "entries": [],
                }
            )
            after = publisher._manifest_bytes(
                {
                    "schema_version": 1,
                    "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
                    "entries": [{"path": "data/x.npz", "role": "static"}],
                }
            )
            candidate = b"candidate"
            manifest.write_bytes(before)
            manifest.chmod(0o644)
            artifact.write_bytes(candidate)
            artifact_before = artifact.read_bytes()
            with (
                mock.patch.object(publisher, "DATA_PATH", data),
                mock.patch.object(publisher, "MANIFEST_PATH", manifest),
                mock.patch.object(publisher, "DESTINATION_PATH", artifact),
            ):
                previous = os.umask(0o777)
                try:
                    publisher._replace_manifest(
                        after,
                        expected_before=before,
                        candidate_bytes=candidate,
                    )
                finally:
                    os.umask(previous)
            self.assertEqual(manifest.read_bytes(), after)
            self.assertEqual(stat.S_IMODE(manifest.stat().st_mode), 0o644)
            self.assertEqual(artifact.read_bytes(), artifact_before)
            self.assertFalse(
                any(
                    child.name.startswith(publisher.MANIFEST_TEMP_PREFIX)
                    for child in data.iterdir()
                )
            )

    def test_manifest_temporary_substitution_preserves_original_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            manifest = data / "MANIFEST.json"
            artifact = data / "artifact.npz"
            before = publisher._manifest_bytes(
                {
                    "schema_version": 1,
                    "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
                    "entries": [],
                }
            )
            after = publisher._manifest_bytes(
                {
                    "schema_version": 1,
                    "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
                    "entries": [{"path": "data/x.npz", "role": "static"}],
                }
            )
            manifest.write_bytes(before)
            manifest.chmod(0o644)
            artifact.write_bytes(b"candidate")
            original = publisher._replace_validated_manifest_stage

            def substitute_then_replace(**kwargs: object) -> None:
                stage = kwargs["stage"]
                assert isinstance(stage, Path)
                stage.unlink()
                stage.write_bytes(b'{"foreign":true}\n')
                stage.chmod(0o644)
                original(**kwargs)  # type: ignore[arg-type]

            with (
                mock.patch.object(publisher, "DATA_PATH", data),
                mock.patch.object(publisher, "MANIFEST_PATH", manifest),
                mock.patch.object(publisher, "DESTINATION_PATH", artifact),
                mock.patch.object(
                    publisher,
                    "_replace_validated_manifest_stage",
                    side_effect=substitute_then_replace,
                ),
                self.assertRaises(publisher.PublicationIOError),
            ):
                publisher._replace_manifest(
                    after,
                    expected_before=before,
                    candidate_bytes=b"candidate",
                )
            self.assertEqual(manifest.read_bytes(), before)
            foreign = [
                child
                for child in data.iterdir()
                if child.name.startswith(publisher.MANIFEST_TEMP_PREFIX)
            ]
            self.assertEqual(len(foreign), 1)
            self.assertEqual(foreign[0].read_bytes(), b'{"foreign":true}\n')

    def test_registered_publish_path_is_validation_only_noop(self) -> None:
        authority = _authority(_template())
        candidate = publisher.Candidate(
            archive_bytes=b"candidate",
            archive_sha256=publisher.sha256_bytes(b"candidate"),
            summary={},
            arrays={},
        )
        manifest = {
            "schema_version": 1,
            "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
            "entries": [],
        }
        prepared = (
            authority,
            candidate,
            manifest,
            b"manifest-before",
            _fake_snapshot(),
            {},
            b"manifest-after",
            "exact_registered",
        )
        lock_state = {"held": False}

        class Lock:
            def __enter__(self) -> int:
                lock_state["held"] = True
                return 1

            def __exit__(self, *_args: object) -> None:
                lock_state["held"] = False

        def fresh_while_locked(**_kwargs: object) -> None:
            self.assertTrue(lock_state["held"])

        with (
            mock.patch.object(
                publisher,
                "_prepare_authorized",
                side_effect=[prepared, prepared],
            ),
            mock.patch.object(
                publisher,
                "_exclusive_data_lock",
                return_value=Lock(),
            ),
            mock.patch.object(
                publisher,
                "_fresh_validate_expected_final",
                side_effect=fresh_while_locked,
            ),
            mock.patch.object(
                publisher,
                "_ensure_destination_parent",
                side_effect=AssertionError("directory mutation reached"),
            ),
            mock.patch.object(
                publisher,
                "_create_stage",
                side_effect=AssertionError("artifact mutation reached"),
            ),
            mock.patch.object(
                publisher,
                "_replace_manifest",
                side_effect=AssertionError("manifest mutation reached"),
            ),
        ):
            report = publisher.publish()
        self.assertFalse(lock_state["held"])
        self.assertFalse(report["publication_performed"])
        self.assertEqual(
            report["installation_result"],
            "exact_registered_validation_noop",
        )

    def test_manifest_replace_uses_retained_locked_data_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            manifest = data / "MANIFEST.json"
            before = publisher._manifest_bytes(
                {
                    "schema_version": 1,
                    "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
                    "entries": [],
                }
            )
            after = publisher._manifest_bytes(
                {
                    "schema_version": 1,
                    "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
                    "entries": [{"path": "data/x.npz", "role": "static"}],
                }
            )
            manifest.write_bytes(before)
            artifact = root / publisher.DESTINATION_RELATIVE_PATH
            artifact.parent.mkdir(parents=True)
            candidate = b"candidate"
            artifact.write_bytes(candidate)
            with (
                mock.patch.object(
                    publisher,
                    "CANONICAL_REPOSITORY_ROOT",
                    root,
                ),
                mock.patch.object(publisher, "DATA_PATH", data),
                mock.patch.object(publisher, "MANIFEST_PATH", manifest),
            ):
                data_fd = publisher._open_repository_directory(
                    "data",
                    label="data",
                )
                try:
                    publisher._replace_manifest(
                        after,
                        expected_before=before,
                        candidate_bytes=candidate,
                        data_fd=data_fd,
                    )
                finally:
                    os.close(data_fd)
            self.assertEqual(manifest.read_bytes(), after)
            self.assertEqual(artifact.read_bytes(), candidate)

    def test_post_replace_data_swap_rejects_before_helper_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            manifest = data / "MANIFEST.json"
            before = publisher._manifest_bytes(
                {
                    "schema_version": 1,
                    "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
                    "entries": [],
                }
            )
            after = publisher._manifest_bytes(
                {
                    "schema_version": 1,
                    "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
                    "entries": [{"path": "data/x.npz", "role": "static"}],
                }
            )
            manifest.write_bytes(before)
            artifact = root / publisher.DESTINATION_RELATIVE_PATH
            artifact.parent.mkdir(parents=True)
            candidate = b"candidate"
            artifact.write_bytes(candidate)
            moved = root / "data-moved-after-replace"
            real_replace = publisher.os.replace

            def replace_then_swap(*args: object, **kwargs: object) -> None:
                real_replace(*args, **kwargs)  # type: ignore[arg-type]
                data.rename(moved)
                data.mkdir()

            with (
                mock.patch.object(
                    publisher,
                    "CANONICAL_REPOSITORY_ROOT",
                    root,
                ),
                mock.patch.object(publisher, "DATA_PATH", data),
                mock.patch.object(publisher, "MANIFEST_PATH", manifest),
                mock.patch.object(publisher, "DESTINATION_PATH", artifact),
            ):
                data_fd = publisher._open_repository_directory(
                    "data",
                    label="data",
                )
                try:
                    with (
                        mock.patch.object(
                            publisher.os,
                            "replace",
                            side_effect=replace_then_swap,
                        ),
                        self.assertRaisesRegex(
                            publisher.PublicationStateError,
                            "canonical directory identity changed",
                        ),
                    ):
                        publisher._replace_manifest(
                            after,
                            expected_before=before,
                            candidate_bytes=candidate,
                            data_fd=data_fd,
                        )
                finally:
                    os.close(data_fd)
            self.assertFalse(manifest.exists())
            self.assertEqual((moved / "MANIFEST.json").read_bytes(), after)
            moved_artifact = moved / Path(
                publisher.DESTINATION_RELATIVE_PATH
            ).relative_to("data")
            self.assertEqual(moved_artifact.read_bytes(), candidate)


class CanonicalInventoryTests(unittest.TestCase):
    """The current repository is closed and publication failures leave it so."""

    def test_current_data_snapshot_is_closed_and_manifest_join_exact(self) -> None:
        snapshot = publisher._snapshot_data()
        manifest = publisher._parse_manifest(
            (REPOSITORY_ROOT / publisher.MANIFEST_RELATIVE_PATH).read_bytes()
        )
        entry_paths = {entry["path"] for entry in manifest["entries"]}
        file_paths = {fact.relative_path for fact in snapshot.files}
        self.assertTrue(entry_paths.issubset(file_paths))
        self.assertEqual(
            file_paths - entry_paths,
            {
                publisher.MANIFEST_RELATIVE_PATH,
                *(item.relative_path for item in snapshot.nonmanifest_support),
            },
        )
        self.assertIn(
            publisher.DATA_README_RELATIVE_PATH,
            {item.relative_path for item in snapshot.nonmanifest_support},
        )
        self.assertEqual(
            snapshot.manifest_sha256,
            _hash(REPOSITORY_ROOT / publisher.MANIFEST_RELATIVE_PATH),
        )
        self.assertEqual(
            snapshot.ordered_entry_path_digest,
            publisher._ordered_path_digest(
                entry["path"] for entry in manifest["entries"]
            ),
        )

    def test_source_data_digest_is_order_stable_and_source_sensitive(self) -> None:
        snapshot = publisher._snapshot_data()
        first = {"a": "1", "b": "2"}
        second = {"a": "1", "b": "3"}
        digest = publisher._source_data_snapshot_digest(snapshot, first)
        self.assertEqual(
            digest,
            publisher._source_data_snapshot_digest(snapshot, first),
        )
        self.assertNotEqual(
            digest,
            publisher._source_data_snapshot_digest(snapshot, second),
        )

    def test_unexpected_empty_directory_is_rejected_by_closed_inventory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            readme = data / "README.md"
            readme.write_bytes(b"closed")
            manifest = data / "MANIFEST.json"
            manifest.write_bytes(
                publisher._manifest_bytes(
                    {
                        "schema_version": 1,
                        "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
                        "entries": [],
                    }
                )
            )
            (data / "rogue-empty-directory").mkdir()
            with (
                mock.patch.object(
                    publisher,
                    "CANONICAL_REPOSITORY_ROOT",
                    root,
                ),
                mock.patch.object(publisher, "DATA_PATH", data),
                mock.patch.object(
                    publisher,
                    "PREPUBLICATION_DATA_DIRECTORIES",
                    ("data",),
                ),
                mock.patch.object(
                    publisher,
                    "DATA_README_SHA256",
                    hashlib.sha256(b"closed").hexdigest(),
                ),
                self.assertRaisesRegex(
                    publisher.PublicationStateError,
                    "directory inventory changed",
                ),
            ):
                publisher._snapshot_data()

    def test_exact_unregistered_recovery_normalizes_only_reviewed_delta(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            fixtures = data / "fixtures"
            base_parent = data / "golden/payne_zero"
            fixtures.mkdir(parents=True)
            base_parent.mkdir(parents=True)
            readme = data / "README.md"
            readme.write_bytes(b"closed")
            atmosphere = root / publisher.ATMOSPHERE_DESTINATION_RELATIVE_PATH
            atmosphere.write_bytes(b"atmosphere")
            atmosphere_entry = {
                "path": publisher.ATMOSPHERE_DESTINATION_RELATIVE_PATH,
                "role": "fixture",
                "sha256": hashlib.sha256(b"atmosphere").hexdigest(),
                "bytes": len(b"atmosphere"),
            }
            manifest_value = {
                "schema_version": 1,
                "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
                "entries": [atmosphere_entry],
            }
            manifest_bytes = publisher._manifest_bytes(manifest_value)
            manifest = data / "MANIFEST.json"
            manifest.write_bytes(manifest_bytes)
            candidate = b"exact-recovery-candidate"
            destination = root / publisher.DESTINATION_RELATIVE_PATH
            base_directories = (
                "data",
                "data/fixtures",
                "data/golden",
                "data/golden/payne_zero",
            )
            sources = {"accepted-source": "f" * 64}
            with (
                mock.patch.object(
                    publisher,
                    "CANONICAL_REPOSITORY_ROOT",
                    root,
                ),
                mock.patch.object(publisher, "DATA_PATH", data),
                mock.patch.object(publisher, "MANIFEST_PATH", manifest),
                mock.patch.object(publisher, "DESTINATION_PATH", destination),
                mock.patch.object(
                    publisher,
                    "PREPUBLICATION_DATA_DIRECTORIES",
                    base_directories,
                ),
                mock.patch.object(
                    publisher,
                    "DATA_README_SHA256",
                    hashlib.sha256(b"closed").hexdigest(),
                ),
                mock.patch.object(
                    publisher,
                    "EXPECTED_ARCHIVE_BYTES",
                    len(candidate),
                ),
                mock.patch.object(
                    publisher,
                    "EXPECTED_ARCHIVE_SHA256",
                    hashlib.sha256(candidate).hexdigest(),
                ),
                mock.patch.object(
                    publisher,
                    "_verify_source_identity",
                    return_value=sources,
                ),
            ):
                before = publisher._snapshot_data()
                bound_digest = publisher._source_data_snapshot_digest(
                    before,
                    sources,
                )
                destination.parent.mkdir(parents=True)
                destination.write_bytes(candidate)
                destination.chmod(0o600)
                recovery = publisher._snapshot_data(
                    allow_exact_unregistered_target=(
                        publisher.DESTINATION_RELATIVE_PATH
                    )
                )
                self.assertEqual(
                    recovery.aggregate_sha256,
                    before.aggregate_sha256,
                )
                authority = _authority(_template())
                authority.authorization.update(
                    {
                        "prepublication_manifest": {
                            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                            "ordered_entry_path_digest": (
                                publisher._ordered_path_digest(
                                    entry["path"] for entry in manifest_value["entries"]
                                )
                            ),
                            "atmosphere_artifact_sha256": atmosphere_entry["sha256"],
                            "atmosphere_entry_digest": publisher._entry_digest(
                                atmosphere_entry
                            ),
                        },
                        "source_data_snapshot_sha256": bound_digest,
                    }
                )
                (
                    _manifest,
                    _manifest_bytes,
                    recovered_snapshot,
                    _source,
                    lifecycle,
                    _current,
                ) = publisher._validate_prepublication_state(authority)
                self.assertEqual(lifecycle, "prepublication")
                self.assertEqual(
                    recovered_snapshot.aggregate_sha256,
                    before.aggregate_sha256,
                )
                destination.write_bytes(b"nonexact")
                with self.assertRaises(publisher.PublicationStateError):
                    publisher._snapshot_data(
                        allow_exact_unregistered_target=(
                            publisher.DESTINATION_RELATIVE_PATH
                        )
                    )

    def test_immediate_pre_manifest_recheck_binds_source_and_nontarget_data(
        self,
    ) -> None:
        manifest = {
            "schema_version": 1,
            "payne_zero_commit": publisher.PINNED_PAYNE_ZERO_COMMIT,
            "entries": [],
        }
        manifest_bytes = publisher._manifest_bytes(manifest)
        ordered = publisher._ordered_path_digest(())
        snapshot = publisher.DataSnapshot(
            directories=(),
            files=(),
            manifest_files=(),
            nonmanifest_support=(),
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            manifest_device=1,
            manifest_inode=2,
            ordered_entry_path_digest=ordered,
            aggregate_sha256="3" * 64,
        )
        source = {"accepted-source": "4" * 64}
        authority = _authority(_template())
        authority.authorization["source_data_snapshot_sha256"] = (
            publisher._source_data_snapshot_digest(
                snapshot,
                source,
                manifest_sha256=snapshot.manifest_sha256,
                ordered_entry_path_digest=ordered,
            )
        )
        with (
            mock.patch.object(
                publisher,
                "_verify_fixed_upstream_identities",
                return_value={},
            ),
            mock.patch.object(
                publisher,
                "_verify_source_identity",
                return_value=source,
            ),
            mock.patch.object(
                publisher,
                "_snapshot_data",
                return_value=snapshot,
            ),
        ):
            publisher._revalidate_source_data_before_manifest(
                authority,
                expected_manifest_bytes=manifest_bytes,
            )
        with (
            mock.patch.object(
                publisher,
                "_verify_fixed_upstream_identities",
                return_value={},
            ),
            mock.patch.object(
                publisher,
                "_verify_source_identity",
                return_value={"accepted-source": "5" * 64},
            ),
            mock.patch.object(
                publisher,
                "_snapshot_data",
                return_value=snapshot,
            ),
            self.assertRaises(publisher.PublicationStateError),
        ):
            publisher._revalidate_source_data_before_manifest(
                authority,
                expected_manifest_bytes=manifest_bytes,
            )
        changed_snapshot = publisher.DataSnapshot(
            **{
                **snapshot.__dict__,
                "aggregate_sha256": "6" * 64,
            }
        )
        with (
            mock.patch.object(
                publisher,
                "_verify_fixed_upstream_identities",
                return_value={},
            ),
            mock.patch.object(
                publisher,
                "_verify_source_identity",
                return_value=source,
            ),
            mock.patch.object(
                publisher,
                "_snapshot_data",
                return_value=changed_snapshot,
            ),
            self.assertRaises(publisher.PublicationStateError),
        ):
            publisher._revalidate_source_data_before_manifest(
                authority,
                expected_manifest_bytes=manifest_bytes,
            )


if __name__ == "__main__":
    unittest.main()
