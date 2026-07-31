"""Focused contracts for deterministic in-memory Chapter 6 synthesis NPZ bytes."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import chapter06_synthesis_compact_writer as writer
from scripts import chapter06_synthesis_oracle_worker as oracle_worker


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRITER_PATH = REPOSITORY_ROOT / "scripts/chapter06_synthesis_compact_writer.py"
EXPECTED_ARCHIVE_SHA256 = (
    "a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955"
)
EXPECTED_ARCHIVE_BYTES = 1_294_865
EXPECTED_RAW_ARCHIVE_SHA256 = (
    "e69398c9a3fd367cabbca6e6e1c16819e7cd3318289a3c6c4c1006df3b915e5e"
)
EXPECTED_RAW_ARCHIVE_BYTES = 8_689_108
EXPECTED_RAW_MAPPING_DIGEST = (
    "09072fb51bd3425f6e635275db4f08c6a4fb33c367c9be1a85cdb6c62bc7b06c"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_data_snapshot() -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(REPOSITORY_ROOT)): (
            path.stat().st_size,
            _file_sha256(path),
        )
        for path in sorted((REPOSITORY_ROOT / "data").rglob("*"))
        if path.is_file()
    }


def _fresh_child(code: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(
        prefix="chapter06-synthesis-compact-writer-cache-"
    ) as cache:
        environment = os.environ.copy()
        environment.update(oracle_worker.ORACLE_ENVIRONMENT)
        environment["NUMBA_CACHE_DIR"] = cache
        environment["PAYNE_ZERO_DATA_ROOT"] = str(oracle_worker.PINNED_DATA_ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if any(Path(cache).iterdir()):
            raise AssertionError("deterministic writer populated the Numba cache")
    return completed


class Chapter06SynthesisCompactWriterTests(unittest.TestCase):
    """Pin byte determinism without granting filesystem/publication authority."""

    def test_exact_identity_gate_and_no_publication_surface(self) -> None:
        identities = writer.verify_accepted_identities()
        self.assertEqual(set(identities), set(writer.ACCEPTED_FILE_IDENTITIES))
        self.assertEqual(len(identities), 10)

        stale_identities = {
            "fixture_oracle_plan": (
                "d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565"
            ),
            "compact_assembler": (
                "62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a"
            ),
            "plan_rebind_independent_audit": "0" * 64,
            "compact_rebind_independent_audit": "0" * 64,
        }
        for label, stale_sha256 in stale_identities.items():
            with self.subTest(label=label):
                changed = dict(writer.ACCEPTED_FILE_IDENTITIES)
                path, _accepted = changed[label]
                changed[label] = (path, stale_sha256)
                with mock.patch.object(writer, "ACCEPTED_FILE_IDENTITIES", changed):
                    with self.assertRaisesRegex(
                        writer.CompactWriterError,
                        f"accepted identity changed for {label}",
                    ):
                        writer.verify_accepted_identities()

        self.assertEqual(
            tuple(
                inspect.signature(writer.build_deterministic_compact_archive).parameters
            ),
            (),
        )
        self.assertFalse(hasattr(writer, "validate_deterministic_compact_archive"))
        with self.assertRaises(TypeError):
            writer.build_deterministic_compact_archive({}, {})
        self.assertFalse(hasattr(writer, "main"))
        self.assertFalse(hasattr(writer, "parse_args"))
        source = WRITER_PATH.read_text()
        tree = ast.parse(source)
        forbidden_attributes = {
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
        self.assertTrue(forbidden_attributes.isdisjoint(called_attributes))
        self.assertNotIn("data/golden", source)
        self.assertNotIn("data/MANIFEST.json", source)
        self.assertNotIn("--publish", source)
        self.assertNotIn("--output", source)

    @unittest.skipUnless(
        oracle_worker.PINNED_ROOT.is_dir(), "pinned Payne Zero checkout absent"
    )
    def test_two_fresh_top_level_builds_reproduce_raw_and_final_bytes(self) -> None:
        code = """
import json
from scripts import chapter06_synthesis_compact_writer as writer
result = writer.build_deterministic_compact_archive()
assert isinstance(result.archive_bytes, bytes)
print(json.dumps(result.summary, sort_keys=True))
"""
        data_before = _canonical_data_snapshot()
        first = _fresh_child(code)
        second = _fresh_child(code)
        data_after = _canonical_data_snapshot()
        self.assertEqual(first.stderr, "")
        self.assertEqual(second.stderr, "")
        self.assertEqual(data_before, data_after)

        first_summary = json.loads(first.stdout)
        second_summary = json.loads(second.stdout)
        nondeterministic_evidence = {
            "capture_process_evidence",
            "capture_process_evidence_sha256",
        }
        self.assertEqual(
            {
                name: value
                for name, value in first_summary.items()
                if name not in nondeterministic_evidence
            },
            {
                name: value
                for name, value in second_summary.items()
                if name not in nondeterministic_evidence
            },
        )
        summary = first_summary
        self.assertEqual(summary["archive_a_sha256"], EXPECTED_ARCHIVE_SHA256)
        self.assertEqual(summary["archive_b_sha256"], EXPECTED_ARCHIVE_SHA256)
        self.assertEqual(summary["archive_bytes"], EXPECTED_ARCHIVE_BYTES)
        self.assertEqual(summary["archive_member_count"], 213)
        self.assertTrue(summary["archive_a_b_byte_equal"])
        self.assertEqual(
            summary["raw_a_archive_sha256"],
            EXPECTED_RAW_ARCHIVE_SHA256,
        )
        self.assertEqual(
            summary["raw_b_archive_sha256"],
            EXPECTED_RAW_ARCHIVE_SHA256,
        )
        self.assertEqual(summary["raw_archive_bytes"], EXPECTED_RAW_ARCHIVE_BYTES)
        self.assertEqual(
            summary["raw_a_b_mapping_digest"],
            EXPECTED_RAW_MAPPING_DIGEST,
        )
        self.assertTrue(summary["raw_a_b_archive_byte_equal"])
        self.assertTrue(summary["raw_a_b_bitwise_equal"])
        self.assertTrue(summary["compact_a_b_schema_equal"])
        self.assertTrue(summary["compact_a_b_payload_equal"])
        self.assertTrue(summary["compact_a_b_ownership_equal"])
        self.assertFalse(summary["publication_authorized"])
        self.assertFalse(summary["golden_publication_performed"])
        self.assertFalse(summary["manifest_mutation_performed"])
        self.assertFalse(summary["artifact_file_write_performed"])
        self.assertTrue(summary["disposable_cache_directories_created"])

        all_tokens: set[str] = set()
        all_caches: set[str] = set()
        all_pids: set[int] = set()
        for candidate in (first_summary, second_summary):
            evidence = candidate["capture_process_evidence"]
            self.assertTrue(evidence["capture_origins_distinct"])
            self.assertTrue(evidence["capture_child_processes_distinct"])
            self.assertTrue(evidence["capture_cache_roots_distinct"])
            self.assertTrue(evidence["capture_cache_roots_external"])
            self.assertTrue(evidence["capture_cache_roots_nonsymlink"])
            self.assertTrue(evidence["capture_cache_roots_empty_before"])
            self.assertTrue(evidence["capture_cache_roots_empty_after"])
            self.assertTrue(evidence["capture_cache_roots_disposed"])
            tokens = {
                evidence["capture_a_origin_token_sha256"],
                evidence["capture_b_origin_token_sha256"],
            }
            caches = {
                evidence["capture_a_cache_root_sha256"],
                evidence["capture_b_cache_root_sha256"],
            }
            pids = {
                evidence["capture_a_child_pid"],
                evidence["capture_b_child_pid"],
            }
            self.assertEqual(len(tokens), 2)
            self.assertEqual(len(caches), 2)
            self.assertEqual(len(pids), 2)
            all_tokens.update(tokens)
            all_caches.update(caches)
            all_pids.update(pids)
            encoded = json.dumps(
                evidence,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            self.assertEqual(
                candidate["capture_process_evidence_sha256"],
                hashlib.sha256(encoded).hexdigest(),
            )
        self.assertEqual(len(all_tokens), 4)
        self.assertEqual(len(all_caches), 4)
        self.assertGreaterEqual(len(all_pids), 2)

    @unittest.skipUnless(
        oracle_worker.PINNED_ROOT.is_dir(), "pinned Payne Zero checkout absent"
    )
    def test_archive_has_exact_lexical_members_and_fixed_zip_metadata(self) -> None:
        code = """
import hashlib
from io import BytesIO
import json
import zipfile
from scripts import chapter06_synthesis_compact_writer as writer

result = writer.build_deterministic_compact_archive()
loaded = writer._deserialize_canonical_mapping(
    result.archive_bytes,
    size_ceiling=4 * 1024 * 1024,
)
with zipfile.ZipFile(BytesIO(result.archive_bytes), "r") as archive:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    payload_versions = [
        list(archive.read(info)[:8])
        for info in infos
    ]
    metadata_exact = all(
        info.date_time == writer.ZIP_DATE_TIME
        and info.compress_type == writer.ZIP_COMPRESSION
        and info.create_system == writer.ZIP_CREATE_SYSTEM
        and info.create_version == writer.ZIP_CREATE_VERSION
        and info.extract_version == writer.ZIP_EXTRACT_VERSION
        and info.flag_bits == writer.ZIP_FLAG_BITS
        and info.volume == 0
        and info.internal_attr == writer.ZIP_INTERNAL_ATTR
        and info.external_attr == writer.ZIP_EXTERNAL_ATTR
        and info.extra == b""
        and info.comment == b""
        and info.compress_size == info.file_size
        for info in infos
    )
    archive_comment_empty = archive.comment == b""
print(json.dumps({
    "archive_sha256": hashlib.sha256(result.archive_bytes).hexdigest(),
    "member_count": len(names),
    "names_lexical": names == sorted(names),
    "names_unique": len(names) == len(set(names)),
    "loaded_names_match": list(loaded) == [
        name.removesuffix(".npy") for name in names
    ],
    "all_npy_v2": all(
        version == [147, 78, 85, 77, 80, 89, 2, 0]
        for version in payload_versions
    ),
    "metadata_exact": metadata_exact,
    "archive_comment_empty": archive_comment_empty,
}, sort_keys=True))
"""
        completed = _fresh_child(code)
        self.assertEqual(completed.stderr, "")
        report = json.loads(completed.stdout)
        self.assertEqual(report["archive_sha256"], EXPECTED_ARCHIVE_SHA256)
        self.assertEqual(report["member_count"], 213)
        self.assertTrue(report["names_lexical"])
        self.assertTrue(report["names_unique"])
        self.assertTrue(report["loaded_names_match"])
        self.assertTrue(report["all_npy_v2"])
        self.assertTrue(report["metadata_exact"])
        self.assertTrue(report["archive_comment_empty"])

    @unittest.skipUnless(
        oracle_worker.PINNED_ROOT.is_dir(), "pinned Payne Zero checkout absent"
    )
    def test_adversarial_archive_and_pair_mutations_fail_closed(self) -> None:
        code = """
from io import BytesIO
import json
import struct
from types import SimpleNamespace
import zipfile
import numpy as np
from scripts import chapter06_synthesis_compact_assembler as compact
from scripts import chapter06_synthesis_compact_writer as writer
from scripts import chapter06_synthesis_oracle_worker as worker

result = writer.build_deterministic_compact_archive()
accepted = result.archive_bytes
failures = {}
def expect(label, callback):
    try:
        callback()
    except writer.CompactWriterError as error:
        failures[label] = str(error)
    else:
        raise AssertionError(f"{label} unexpectedly passed")

raw = worker.build_oracle_results()
assembly = compact.assemble_compact_candidate(raw)
live_payload_fingerprint = writer.ACCEPTED_COMPACT_PAYLOAD_FINGERPRINT
writer.ACCEPTED_COMPACT_PAYLOAD_FINGERPRINT = (
    "ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9"
)
expect(
    "historical_compact_payload",
    lambda: writer._validate_accepted_assembly(assembly),
)
writer.ACCEPTED_COMPACT_PAYLOAD_FINGERPRINT = live_payload_fingerprint

expect(
    "object_dtype",
    lambda: writer._serialize_mapping(
        {"bad": np.asarray([object()], dtype=object)},
        size_ceiling=None,
    ),
)
expect(
    "truncated_archive",
    lambda: writer._deserialize_canonical_mapping(
        accepted[:-17],
        size_ceiling=4 * 1024 * 1024,
    ),
)

mutated = bytearray(accepted)
with zipfile.ZipFile(BytesIO(accepted), "r") as archive:
    first_info = archive.infolist()[0]
offset = first_info.header_offset
name_length, extra_length = struct.unpack_from("<HH", accepted, offset + 26)
data_offset = offset + 30 + name_length + extra_length
mutated[data_offset + 12] ^= 1
expect(
    "mutated_archive",
    lambda: writer._deserialize_canonical_mapping(
        bytes(mutated),
        size_ceiling=4 * 1024 * 1024,
    ),
)

with zipfile.ZipFile(BytesIO(accepted), "r") as source:
    members = [(info.filename, source.read(info)) for info in source.infolist()]

def rebuild(entries, *, changed_date=False):
    stream = BytesIO()
    with zipfile.ZipFile(
        stream,
        "w",
        compression=writer.ZIP_COMPRESSION,
        allowZip64=False,
        strict_timestamps=True,
    ) as archive:
        for index, (name, payload) in enumerate(entries):
            info = writer._zip_info(name)
            if changed_date and index == 0:
                info.date_time = (1981, 1, 1, 0, 0, 0)
            archive.writestr(
                info,
                payload,
                compress_type=writer.ZIP_COMPRESSION,
            )
    return stream.getvalue()

expect(
    "nonlexical_order",
    lambda: writer._deserialize_canonical_mapping(
        rebuild(list(reversed(members))),
        size_ceiling=4 * 1024 * 1024,
    ),
)
expect(
    "changed_zip_metadata",
    lambda: writer._deserialize_canonical_mapping(
        rebuild(members, changed_date=True),
        size_ceiling=4 * 1024 * 1024,
    ),
)

expect(
    "raw_a_b_disagreement",
    lambda: writer._require_raw_pair_identity(
        {"a": np.asarray([1], dtype=np.int16)},
        {"a": np.asarray([2], dtype=np.int16)},
    ),
)
base = SimpleNamespace(
    schema=("same",),
    raw_ownership=("same",),
    arrays={"a": np.asarray([1], dtype=np.int16)},
)
expect(
    "assembly_schema_disagreement",
    lambda: writer._require_assembly_pair_identity(
        base,
        SimpleNamespace(
            schema=("different",),
            raw_ownership=base.raw_ownership,
            arrays=base.arrays,
        ),
    ),
)
expect(
    "assembly_ownership_disagreement",
    lambda: writer._require_assembly_pair_identity(
        base,
        SimpleNamespace(
            schema=base.schema,
            raw_ownership=("different",),
            arrays=base.arrays,
        ),
    ),
)
expect(
    "assembly_payload_disagreement",
    lambda: writer._require_assembly_pair_identity(
        base,
        SimpleNamespace(
            schema=base.schema,
            raw_ownership=base.raw_ownership,
            arrays={"a": np.asarray([2], dtype=np.int16)},
        ),
    ),
)
print(json.dumps({"failures": sorted(failures)}, sort_keys=True))
"""
        completed = _fresh_child(code)
        self.assertEqual(completed.stderr, "")
        report = json.loads(completed.stdout)
        self.assertEqual(
            report["failures"],
            [
                "assembly_ownership_disagreement",
                "assembly_payload_disagreement",
                "assembly_schema_disagreement",
                "changed_zip_metadata",
                "historical_compact_payload",
                "mutated_archive",
                "nonlexical_order",
                "object_dtype",
                "raw_a_b_disagreement",
                "truncated_archive",
            ],
        )

    def test_origin_gate_rejects_alias_copy_child_reuse_and_shared_cache(
        self,
    ) -> None:
        raw = {"a": np.asarray([1], dtype=np.int16)}
        raw_bytes = writer._serialize_mapping(raw, size_ceiling=None)
        base = writer._FreshRawCapture(
            raw_mapping=raw,
            raw_archive_bytes=raw_bytes,
            raw_archive_sha256=writer.sha256_bytes(raw_bytes),
            origin_token_sha256="1" * 64,
            child_pid=101,
            cache_root_sha256="a" * 64,
            cache_empty_before=True,
            cache_empty_after=True,
            cache_external=True,
            cache_nonsymlink=True,
        )

        cases = {
            "same_object": base,
            "copied_same_origin": replace(
                base,
                raw_mapping={"a": raw["a"].copy()},
            ),
            "one_child_reused": replace(
                base,
                origin_token_sha256="2" * 64,
                cache_root_sha256="b" * 64,
            ),
            "shared_cache": replace(
                base,
                origin_token_sha256="2" * 64,
                child_pid=202,
            ),
        }
        messages = {
            "same_object": "reused one capture object",
            "copied_same_origin": "reused one origin token",
            "one_child_reused": "reused one child process",
            "shared_cache": "reused one cache root",
        }
        for label, second in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    writer.CompactWriterError,
                    messages[label],
                ):
                    writer._require_independent_capture_pair(base, second)

        with mock.patch.object(
            writer,
            "_capture_raw_in_fresh_child",
            return_value=base,
        ):
            with self.assertRaisesRegex(
                writer.CompactWriterError,
                "reused one capture object",
            ):
                writer.build_deterministic_compact_archive()

        copied = replace(base, raw_mapping={"a": raw["a"].copy()})
        with mock.patch.object(
            writer,
            "_capture_raw_in_fresh_child",
            side_effect=[base, copied],
        ):
            with self.assertRaisesRegex(
                writer.CompactWriterError,
                "reused one origin token",
            ):
                writer.build_deterministic_compact_archive()

        shared_cache = replace(
            base,
            origin_token_sha256="2" * 64,
            child_pid=202,
        )
        with mock.patch.object(
            writer,
            "_capture_raw_in_fresh_child",
            side_effect=[base, shared_cache],
        ):
            with self.assertRaisesRegex(
                writer.CompactWriterError,
                "reused one cache root",
            ):
                writer.build_deterministic_compact_archive()

    def test_cache_policy_rejects_occupied_symlink_shared_and_forbidden_roots(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="chapter06-writer-cache-policy-"
        ) as parent_text:
            parent = Path(parent_text)
            first = parent / "first"
            second = parent / "second"
            first.mkdir()
            second.mkdir()
            self.assertEqual(
                writer._validate_cache_directory(first, require_empty=True),
                first.resolve(),
            )
            writer._require_distinct_cache_roots(first, second)
            with self.assertRaisesRegex(
                writer.CompactWriterError,
                "must be distinct",
            ):
                writer._require_distinct_cache_roots(first, first)

            occupied = parent / "occupied"
            occupied.mkdir()
            (occupied / "entry").write_bytes(b"occupied")
            with (
                self.assertRaisesRegex(
                    writer.CompactWriterError,
                    "must be empty",
                ),
                mock.patch.object(writer.subprocess, "run") as run,
            ):
                writer._capture_raw_in_fresh_child(
                    occupied,
                    capture_label="occupied",
                )
            run.assert_not_called()

            target = parent / "target"
            target.mkdir()
            symlink = parent / "symlink"
            symlink.symlink_to(target, target_is_directory=True)
            with (
                self.assertRaisesRegex(
                    writer.CompactWriterError,
                    "nonsymlink",
                ),
                mock.patch.object(writer.subprocess, "run") as run,
            ):
                writer._capture_raw_in_fresh_child(
                    symlink,
                    capture_label="symlink",
                )
            run.assert_not_called()

            with mock.patch.object(
                writer,
                "FORBIDDEN_CACHE_ROOTS",
                (parent,),
            ):
                with self.assertRaisesRegex(
                    writer.CompactWriterError,
                    "external to all source/data trees",
                ):
                    writer._validate_cache_directory(
                        first,
                        require_empty=True,
                    )


if __name__ == "__main__":
    unittest.main()
