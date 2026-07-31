"""Focused contracts for deterministic in-memory Chapter 6 atmosphere bytes."""

from __future__ import annotations

from dataclasses import replace
import ast
from contextlib import contextmanager
import copy
import hashlib
from io import BytesIO
import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

import numpy as np

from scripts import chapter06_atmosphere_fixture_worker as worker
from scripts import chapter06_atmosphere_fixture_writer as writer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRITER_PATH = REPOSITORY_ROOT / "scripts/chapter06_atmosphere_fixture_writer.py"

ACCEPTED_ARCHIVE_SHA256 = (
    "1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff"
)
ACCEPTED_ARCHIVE_BYTES = 363_050
ACCEPTED_ARCHIVE_ARRAY_BYTES = 357_984
ACCEPTED_CAPTURE_TRANSPORT_SHA256 = (
    "6feba1394d200e627a36df181b9ca18a257eaac05fdaf101256c88f6d3b78f67"
)
ACCEPTED_CAPTURE_TRANSPORT_BYTES = 33_147_771
ACCEPTED_FIXTURE_MAPPING_DIGEST = (
    "f533e3e327c879b1d367a89822bcd1847b15a73ba3d234a9c577c81997f75e0a"
)
ACCEPTED_EVIDENCE_MAPPING_DIGEST = (
    "98ea10dbe0973466b45811cd4d5927a24d42593056cc25f5fdd03849cab6784a"
)


def file_sha256(path: Path) -> str:
    """Return one file's SHA-256 identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_snapshot() -> dict[str, tuple[int, str]]:
    """Return identities for every canonical textbook data file."""

    return {
        str(path.relative_to(REPOSITORY_ROOT)): (
            path.stat().st_size,
            file_sha256(path),
        )
        for path in sorted((REPOSITORY_ROOT / "data").rglob("*"))
        if path.is_file()
    }


def accepted_source_snapshot() -> dict[str, tuple[int, str]]:
    """Return all accepted local and pinned scientific input identities."""

    snapshot = {
        label: (path.stat().st_size, file_sha256(path))
        for label, (path, _) in writer.ACCEPTED_FILE_IDENTITIES.items()
    }
    for relative_path in worker.FROZEN_PINNED_PYTHON_MANIFEST:
        path = worker.PINNED_ROOT / relative_path
        snapshot[f"pinned-source::{relative_path}"] = (
            path.stat().st_size,
            file_sha256(path),
        )
    for relative_path in worker.DATA_IDENTITIES:
        path = worker.PINNED_ROOT / relative_path
        snapshot[f"pinned-data::{relative_path}"] = (
            path.stat().st_size,
            file_sha256(path),
        )
    for relative_path in worker.STAGED_CONVERTER_DEPENDENCIES:
        path = worker.STAGED_SOURCE_ROOT / relative_path
        snapshot[f"staged-source::{relative_path}"] = (
            path.stat().st_size,
            file_sha256(path),
        )
    return snapshot


@contextmanager
def accepted_process_environment():
    """Temporarily present the exact complete worker capture environment."""

    with mock.patch.dict(os.environ, worker.CAPTURE_ENVIRONMENT, clear=False):
        inherited_data_root = os.environ.pop("PAYNE_ZERO_DATA_ROOT", None)
        try:
            yield
        finally:
            if inherited_data_root is not None:
                os.environ["PAYNE_ZERO_DATA_ROOT"] = inherited_data_root


def array_copy_mapping(
    values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return a detached copy of one array mapping."""

    return {
        name: np.array(value, copy=True, order="C") for name, value in values.items()
    }


def npy_bytes(value: np.ndarray, *, version: tuple[int, int] = (2, 0)) -> bytes:
    """Encode one test array with a selected NPY version."""

    stream = BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(value),
        version=version,
        allow_pickle=True,
    )
    return stream.getvalue()


def raw_zip(
    members: list[tuple[zipfile.ZipInfo, bytes]],
    *,
    comment: bytes = b"",
) -> bytes:
    """Build deliberately selectable ZIP bytes for rejection tests."""

    stream = BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(
            stream,
            mode="w",
            compression=writer.ZIP_COMPRESSION,
            allowZip64=False,
        ) as archive:
            archive.comment = comment
            for info, payload in members:
                archive.writestr(
                    info,
                    payload,
                    compress_type=info.compress_type,
                )
    return stream.getvalue()


class Chapter06AtmosphereFixtureWriterTests(unittest.TestCase):
    """Freeze deterministic bytes without granting filesystem publication."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_before = data_snapshot()
        cls.sources_before = accepted_source_snapshot()
        recorded: dict[str, writer._FreshAtmosphereCapture] = {}
        original_pair_gate = writer._require_independent_capture_pair

        def record_pair(
            capture_a: writer._FreshAtmosphereCapture,
            capture_b: writer._FreshAtmosphereCapture,
        ) -> tuple[str, str]:
            recorded["a"] = capture_a
            recorded["b"] = capture_b
            return original_pair_gate(capture_a, capture_b)

        with (
            accepted_process_environment(),
            mock.patch.object(
                writer,
                "_require_independent_capture_pair",
                side_effect=record_pair,
            ),
        ):
            cls.result = writer.build_deterministic_atmosphere_fixture_archive()
        cls.capture_a = recorded["a"]
        cls.capture_b = recorded["b"]
        cls.data_after = data_snapshot()
        cls.sources_after = accepted_source_snapshot()
        cls.loaded = writer._decode_and_validate_fixture_archive(
            cls.result.archive_bytes,
            cls.capture_a.fixture_arrays,
        )

    def test_public_builder_is_zero_argument_and_owns_both_captures(self) -> None:
        signature = inspect.signature(
            writer.build_deterministic_atmosphere_fixture_archive
        )
        self.assertEqual(tuple(signature.parameters), ())
        with self.assertRaises(TypeError):
            writer.build_deterministic_atmosphere_fixture_archive({})  # type: ignore[call-arg]
        source = WRITER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("raw_a=None", source)
        self.assertNotIn("capture_a=None", source)
        self.assertEqual(source.count("_capture_in_fresh_child("), 3)

    def test_exact_accepted_inputs_are_hash_gated_before_capture(self) -> None:
        self.assertEqual(
            writer.verify_accepted_identities(),
            {
                label: expected_hash
                for label, (_, expected_hash) in sorted(
                    writer.ACCEPTED_FILE_IDENTITIES.items()
                )
            },
        )
        scientific_path, expected_hash = writer.ACCEPTED_FILE_IDENTITIES[
            "scientific_worker"
        ]
        changed = {
            **writer.ACCEPTED_FILE_IDENTITIES,
            "scientific_worker": (scientific_path, "0" * 64),
        }
        with (
            accepted_process_environment(),
            mock.patch.object(writer, "ACCEPTED_FILE_IDENTITIES", changed),
            mock.patch.object(writer.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(
                writer.AtmosphereFixtureWriterError,
                "accepted identity changed",
            ):
                writer.build_deterministic_atmosphere_fixture_archive()
            run.assert_not_called()
        self.assertEqual(file_sha256(scientific_path), expected_hash)

    def test_all_inherited_environment_conflicts_fail_before_child_launch(
        self,
    ) -> None:
        for name, expected in worker.CAPTURE_ENVIRONMENT.items():
            changed = "changed" if expected != "changed" else "different"
            with self.subTest(name=name):
                with (
                    mock.patch.dict(os.environ, {name: changed}, clear=True),
                    mock.patch.object(writer.subprocess, "run") as run,
                ):
                    with self.assertRaisesRegex(
                        writer.AtmosphereFixtureWriterError,
                        name,
                    ):
                        writer.build_deterministic_atmosphere_fixture_archive()
                    run.assert_not_called()

        with mock.patch.dict(os.environ, {}, clear=True):
            completed = writer._require_nonconflicting_capture_environment()
        self.assertEqual(
            {name: completed[name] for name in worker.CAPTURE_ENVIRONMENT},
            worker.CAPTURE_ENVIRONMENT,
        )
        with (
            mock.patch.dict(
                os.environ,
                {"PAYNE_ZERO_DATA_ROOT": "/private/tmp/changed"},
                clear=True,
            ),
            mock.patch.object(writer.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(
                writer.AtmosphereFixtureWriterError,
                "PAYNE_ZERO_DATA_ROOT",
            ):
                writer.build_deterministic_atmosphere_fixture_archive()
            run.assert_not_called()

    def test_capture_pair_binds_distinct_origin_process_and_cache(self) -> None:
        first = self.capture_a
        second = self.capture_b
        self.assertIsNot(first, second)
        self.assertNotEqual(first.origin_token_sha256, second.origin_token_sha256)
        self.assertNotEqual(first.child_pid, second.child_pid)
        self.assertNotEqual(first.cache_root_sha256, second.cache_root_sha256)
        self.assertEqual(
            first.capture_transport_bytes,
            second.capture_transport_bytes,
        )
        self.assertGreater(first.cache_entry_count_after, 0)
        self.assertEqual(
            first.cache_entry_count_after,
            second.cache_entry_count_after,
        )
        self.assertTrue(first.cache_empty_before)
        self.assertTrue(second.cache_empty_before)
        self.assertEqual(
            writer._require_independent_capture_pair(first, second),
            (
                ACCEPTED_FIXTURE_MAPPING_DIGEST,
                ACCEPTED_EVIDENCE_MAPPING_DIGEST,
            ),
        )

    def test_rejects_same_object_and_copied_same_origin_capture(self) -> None:
        with self.assertRaisesRegex(
            writer.AtmosphereFixtureWriterError,
            "one capture object",
        ):
            writer._require_independent_capture_pair(
                self.capture_a,
                self.capture_a,
            )

        copied_same_origin = replace(
            self.capture_a,
            fixture_arrays=array_copy_mapping(self.capture_a.fixture_arrays),
            ephemeral_evidence=array_copy_mapping(self.capture_a.ephemeral_evidence),
            capture_transport_bytes=bytes(
                bytearray(self.capture_a.capture_transport_bytes)
            ),
        )
        self.assertIsNot(copied_same_origin, self.capture_a)
        with self.assertRaisesRegex(
            writer.AtmosphereFixtureWriterError,
            "one origin token",
        ):
            writer._require_independent_capture_pair(
                self.capture_a,
                copied_same_origin,
            )

    def test_rejects_reused_pid_shared_cache_and_incomplete_cache_evidence(
        self,
    ) -> None:
        cases = {
            "child process": replace(
                self.capture_b,
                child_pid=self.capture_a.child_pid,
            ),
            "cache root": replace(
                self.capture_b,
                cache_root_sha256=self.capture_a.cache_root_sha256,
            ),
            "cache evidence": replace(
                self.capture_b,
                cache_empty_before=False,
            ),
        }
        for message, changed in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    writer.AtmosphereFixtureWriterError,
                    message,
                ):
                    writer._require_independent_capture_pair(
                        self.capture_a,
                        changed,
                    )

    def test_rejects_occupied_symlink_forbidden_and_shared_cache_roots(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="chapter06-atmosphere-cache-policy-",
            dir="/private/tmp",
        ) as root_text:
            root = Path(root_text)
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "prior-cache-entry").touch()
            with self.assertRaisesRegex(
                writer.AtmosphereFixtureWriterError,
                "truly empty",
            ):
                writer._validate_cache_directory(occupied, require_empty=True)

            target = root / "target"
            target.mkdir()
            symlink = root / "symlink"
            symlink.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(
                writer.AtmosphereFixtureWriterError,
                "nonsymlink",
            ):
                writer._validate_cache_directory(symlink, require_empty=True)

            first = root / "first"
            first.mkdir()
            with self.assertRaisesRegex(
                writer.AtmosphereFixtureWriterError,
                "must be distinct",
            ):
                writer._require_distinct_cache_roots(first, first)

            with (
                accepted_process_environment(),
                mock.patch.object(writer.subprocess, "run") as run,
            ):
                with self.assertRaisesRegex(
                    writer.AtmosphereFixtureWriterError,
                    "truly empty",
                ):
                    writer._capture_in_fresh_child(
                        occupied,
                        capture_label="test",
                        base_environment=dict(os.environ),
                    )
                run.assert_not_called()

        with self.assertRaisesRegex(
            writer.AtmosphereFixtureWriterError,
            "external",
        ):
            writer._validate_cache_directory(
                writer.REPOSITORY_ROOT,
                require_empty=False,
            )

    def test_complete_capture_and_exact_nineteen_member_payload_are_bound(
        self,
    ) -> None:
        writer._validate_accepted_capture(
            self.capture_a.fixture_arrays,
            self.capture_a.ephemeral_evidence,
        )
        changed_fixture = array_copy_mapping(self.capture_a.fixture_arrays)
        changed_fixture["temperature"][0] = np.nextafter(
            changed_fixture["temperature"][0],
            np.inf,
        )
        with self.assertRaisesRegex(
            writer.AtmosphereFixtureWriterError,
            "nineteen-member fixture validation failed",
        ):
            writer._validate_accepted_capture(
                changed_fixture,
                self.capture_a.ephemeral_evidence,
            )

        changed_evidence = array_copy_mapping(self.capture_a.ephemeral_evidence)
        original_version = str(changed_evidence["meta__python_version"])
        changed_evidence["meta__python_version"] = np.asarray(
            ("x" if original_version[0] != "x" else "y") + original_version[1:]
        )
        with self.assertRaisesRegex(
            writer.AtmosphereFixtureWriterError,
            "full capture fingerprint recomputation changed",
        ):
            writer._validate_accepted_capture(
                self.capture_a.fixture_arrays,
                changed_evidence,
            )

    def test_canonical_archive_has_exact_bytes_members_and_metadata(self) -> None:
        archive_bytes = self.result.archive_bytes
        summary = self.result.summary
        self.assertIsInstance(archive_bytes, bytes)
        self.assertEqual(len(archive_bytes), ACCEPTED_ARCHIVE_BYTES)
        self.assertEqual(writer.sha256_bytes(archive_bytes), ACCEPTED_ARCHIVE_SHA256)
        self.assertEqual(summary["archive_bytes"], ACCEPTED_ARCHIVE_BYTES)
        self.assertEqual(summary["archive_array_bytes"], ACCEPTED_ARCHIVE_ARRAY_BYTES)
        self.assertEqual(summary["archive_member_count"], 19)
        self.assertEqual(
            summary["capture_transport_bytes"],
            ACCEPTED_CAPTURE_TRANSPORT_BYTES,
        )
        self.assertEqual(
            summary["capture_a_transport_sha256"],
            ACCEPTED_CAPTURE_TRANSPORT_SHA256,
        )
        self.assertEqual(
            summary["capture_b_transport_sha256"],
            ACCEPTED_CAPTURE_TRANSPORT_SHA256,
        )
        self.assertLessEqual(
            len(archive_bytes),
            writer.FIXTURE_ARCHIVE_SIZE_CEILING_BYTES,
        )

        with zipfile.ZipFile(BytesIO(archive_bytes), mode="r") as archive:
            self.assertEqual(archive.comment, b"")
            infos = archive.infolist()
            expected = [f"{name}.npy" for name in sorted(worker.FIXTURE_SCHEMA)]
            self.assertEqual([info.filename for info in infos], expected)
            for info, expected_name in zip(infos, expected, strict=True):
                writer._validate_zip_info(info, expected_name)
                payload = archive.read(info)
                self.assertTrue(payload.startswith(b"\x93NUMPY\x02\x00"))

        self.assertEqual(tuple(self.loaded), tuple(sorted(worker.FIXTURE_SCHEMA)))
        for name, (shape, dtype) in worker.FIXTURE_SCHEMA.items():
            self.assertEqual(self.loaded[name].shape, shape)
            self.assertEqual(self.loaded[name].dtype, dtype)
            self.assertEqual(
                worker.array_sha256(self.loaded[name]),
                worker.EXPECTED_FIXTURE_MEMBER_HASHES[name],
            )

    def test_decode_reencode_identity_and_valid_crc_mutations_are_rejected(
        self,
    ) -> None:
        reencoded = writer._serialize_mapping(
            self.loaded,
            size_ceiling=writer.FIXTURE_ARCHIVE_SIZE_CEILING_BYTES,
        )
        self.assertEqual(reencoded, self.result.archive_bytes)

        payload_changed = array_copy_mapping(self.loaded)
        payload_changed["temperature"][0] = np.nextafter(
            payload_changed["temperature"][0],
            np.inf,
        )
        changed_bytes = writer._serialize_mapping(
            payload_changed,
            size_ceiling=writer.FIXTURE_ARCHIVE_SIZE_CEILING_BYTES,
        )
        with self.assertRaisesRegex(
            writer.AtmosphereFixtureWriterError,
            "serialized atmosphere fixture validation failed",
        ):
            writer._decode_and_validate_fixture_archive(
                changed_bytes,
                self.capture_a.fixture_arrays,
            )

        shape_changed = array_copy_mapping(self.loaded)
        shape_changed["temperature"] = shape_changed["temperature"].reshape(40, 2)
        changed_bytes = writer._serialize_mapping(
            shape_changed,
            size_ceiling=writer.FIXTURE_ARCHIVE_SIZE_CEILING_BYTES,
        )
        with self.assertRaisesRegex(
            writer.AtmosphereFixtureWriterError,
            "serialized atmosphere fixture validation failed",
        ):
            writer._decode_and_validate_fixture_archive(
                changed_bytes,
                self.capture_a.fixture_arrays,
            )

    def test_malformed_noncanonical_and_path_like_archives_are_rejected(
        self,
    ) -> None:
        with zipfile.ZipFile(BytesIO(self.result.archive_bytes), mode="r") as archive:
            canonical_members = [
                (copy.copy(info), archive.read(info)) for info in archive.infolist()
            ]

        mutations: dict[str, tuple[str, bytes]] = {}
        reversed_members = list(reversed(canonical_members))
        mutations["order"] = ("order", raw_zip(reversed_members))

        timestamp_members = copy.deepcopy(canonical_members)
        timestamp_members[0][0].date_time = (2001, 2, 3, 4, 5, 6)
        mutations["timestamp"] = ("metadata", raw_zip(timestamp_members))

        permission_members = copy.deepcopy(canonical_members)
        permission_members[0][0].external_attr = 0o100644 << 16
        mutations["permissions"] = ("metadata", raw_zip(permission_members))

        compressed_members = copy.deepcopy(canonical_members)
        compressed_members[0][0].compress_type = zipfile.ZIP_DEFLATED
        mutations["compression"] = ("metadata", raw_zip(compressed_members))

        npy_v1_members = copy.deepcopy(canonical_members)
        first_base = npy_v1_members[0][0].filename.removesuffix(".npy")
        npy_v1_members[0] = (
            npy_v1_members[0][0],
            npy_bytes(self.loaded[first_base], version=(1, 0)),
        )
        mutations["NPY version"] = ("NPY version", raw_zip(npy_v1_members))

        path_members = copy.deepcopy(canonical_members)
        path_members[0][0].filename = "../escape.npy"
        mutations["path-like"] = ("path-like", raw_zip(path_members))

        duplicate_members = copy.deepcopy(canonical_members)
        duplicate_members.append(copy.deepcopy(duplicate_members[0]))
        duplicate_members.sort(key=lambda item: item[0].filename)
        mutations["duplicate"] = ("duplicate", raw_zip(duplicate_members))

        mutations["comment"] = (
            "comment",
            raw_zip(canonical_members, comment=b"changed"),
        )
        mutations["truncated"] = ("invalid", self.result.archive_bytes[:-19])
        corrupted = bytearray(self.result.archive_bytes)
        corrupted[len(corrupted) // 2] ^= 0x01
        mutations["corrupted"] = ("invalid", bytes(corrupted))

        for label, (expected_message, archive_bytes) in mutations.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    writer.AtmosphereFixtureWriterError,
                    expected_message,
                ):
                    writer._decode_and_validate_fixture_archive(
                        archive_bytes,
                        self.capture_a.fixture_arrays,
                    )

    def test_writer_boundaries_reject_object_dtype_and_oversize_bytes(self) -> None:
        with self.assertRaisesRegex(
            writer.AtmosphereFixtureWriterError,
            "object dtype",
        ):
            writer._serialize_mapping(
                {"bad": np.asarray([object()], dtype=object)},
                size_ceiling=1024,
            )
        with self.assertRaisesRegex(
            writer.AtmosphereFixtureWriterError,
            "ceiling",
        ):
            writer._deserialize_canonical_mapping(
                b"x" * (writer.FIXTURE_ARCHIVE_SIZE_CEILING_BYTES + 1),
                size_ceiling=writer.FIXTURE_ARCHIVE_SIZE_CEILING_BYTES,
            )
        with self.assertRaisesRegex(
            writer.AtmosphereFixtureWriterError,
            "immutable bytes",
        ):
            writer._deserialize_canonical_mapping(  # type: ignore[arg-type]
                bytearray(self.result.archive_bytes),
                size_ceiling=writer.FIXTURE_ARCHIVE_SIZE_CEILING_BYTES,
            )

    def test_process_evidence_is_outside_scientific_bytes_and_caches_dispose(
        self,
    ) -> None:
        summary = self.result.summary
        process = summary["capture_process_evidence"]
        self.assertTrue(process["capture_origins_distinct"])
        self.assertTrue(process["capture_child_processes_distinct"])
        self.assertTrue(process["capture_cache_roots_distinct"])
        self.assertTrue(process["capture_cache_roots_external"])
        self.assertTrue(process["capture_cache_roots_nonsymlink"])
        self.assertTrue(process["capture_cache_roots_empty_before"])
        self.assertTrue(process["capture_cache_roots_disposed"])
        self.assertGreater(process["capture_a_cache_entry_count_after"], 0)
        self.assertEqual(
            process["capture_a_cache_entry_count_after"],
            process["capture_b_cache_entry_count_after"],
        )
        self.assertFalse(summary["publication_authorized"])
        self.assertFalse(summary["fixture_publication_performed"])
        self.assertFalse(summary["golden_publication_performed"])
        self.assertFalse(summary["manifest_mutation_performed"])
        self.assertFalse(summary["artifact_file_write_performed"])

        decoded_names = set(self.loaded)
        forbidden_fragments = {
            "origin",
            "pid",
            "cache",
            "evidence",
            "transport",
            "worker",
            "sha256",
        }
        self.assertFalse(
            any(
                fragment in name
                for name in decoded_names
                for fragment in forbidden_fragments
            )
        )

    def test_source_and_data_are_immutable_and_writer_has_no_publisher(self) -> None:
        self.assertEqual(self.data_before, self.data_after)
        self.assertEqual(self.sources_before, self.sources_after)

        source = WRITER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_calls = {
            "save",
            "savez",
            "savez_compressed",
            "tofile",
            "write_bytes",
            "write_text",
            "replace",
            "rename",
        }
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(forbidden_calls.isdisjoint(called_attributes))
        for forbidden_text in (
            "data/fixtures",
            "data/golden",
            "data/MANIFEST.json",
            "--output",
            "argparse",
            "authorization",
            "publisher",
        ):
            with self.subTest(forbidden_text=forbidden_text):
                if forbidden_text in {"authorization", "publisher"}:
                    self.assertNotIn(f"def {forbidden_text}", source.lower())
                else:
                    self.assertNotIn(forbidden_text, source)


if __name__ == "__main__":
    unittest.main()
