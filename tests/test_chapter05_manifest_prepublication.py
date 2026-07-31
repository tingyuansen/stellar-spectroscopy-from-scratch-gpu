"""Fail-closed manifest gates for the two accepted Chapter 5 candidates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import sync_data_manifest as manifest_sync


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_GOLDEN_DIR = (
    REPOSITORY_ROOT / "data" / "golden" / "payne_zero" / "chapter05"
)
REPAIRED_REVIEW_CANDIDATE = Path(
    "/tmp/ch05-candidate-repaired.MAqeZl/capture/final"
)


def _candidate_root() -> Path | None:
    override = os.environ.get("CHAPTER05_CANDIDATE_DIR")
    if override is not None:
        return Path(override)
    if REPAIRED_REVIEW_CANDIDATE.is_dir():
        return REPAIRED_REVIEW_CANDIDATE
    if CANONICAL_GOLDEN_DIR.is_dir():
        return CANONICAL_GOLDEN_DIR
    return None


def _member_name_digest(names: list[str]) -> str:
    return hashlib.sha256("\0".join(names).encode()).hexdigest()


class Chapter05ManifestPrepublicationTests(unittest.TestCase):
    """Require exact publication state, identities, units, and ownership."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.candidate_root = _candidate_root()
        cls.specifications: tuple[dict[str, object], ...] = ()
        if cls.candidate_root is not None:
            cls.specifications = manifest_sync.chapter05_golden_specs(
                cls.candidate_root
            )

    def require_candidate(self) -> Path:
        if self.candidate_root is None:
            self.skipTest(
                "set CHAPTER05_CANDIDATE_DIR to exercise the accepted pair"
            )
        return self.candidate_root

    def test_candidate_override_does_not_modify_live_manifest(self) -> None:
        root = self.require_candidate()
        before = manifest_sync.MANIFEST_PATH.read_bytes()
        specifications = manifest_sync.chapter05_golden_specs(root)
        self.assertEqual(len(specifications), 2)
        self.assertEqual(manifest_sync.MANIFEST_PATH.read_bytes(), before)

    def test_candidate_override_builds_exact_two_entries(self) -> None:
        root = self.require_candidate()
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "MANIFEST.json"
            shutil.copyfile(manifest_sync.MANIFEST_PATH, manifest_path)
            live_before = manifest_sync.MANIFEST_PATH.read_bytes()
            with mock.patch.object(
                manifest_sync,
                "MANIFEST_PATH",
                manifest_path,
            ):
                manifest_sync.main(chapter05_archive_root=root)
            built = json.loads(manifest_path.read_text(encoding="utf-8"))
            prefix = "data/golden/payne_zero/chapter05/"
            entries = [
                entry
                for entry in built["entries"]
                if entry["path"].startswith(prefix)
            ]
            self.assertEqual(len(entries), 2)
            self.assertEqual(
                {Path(entry["path"]).name for entry in entries},
                set(manifest_sync.CHAPTER05_GOLDEN_ARCHIVES),
            )
            for entry in entries:
                name = Path(entry["path"]).name
                self.assertEqual(
                    entry["builder"],
                    manifest_sync.CHAPTER05_GOLDEN_BUILDER,
                )
                self.assertEqual(
                    len(entry["arrays"]),
                    manifest_sync.CHAPTER05_GOLDEN_ARCHIVES[name]["key_count"],
                )
            self.assertEqual(manifest_sync.MANIFEST_PATH.read_bytes(), live_before)

    def test_specs_pin_exact_pair_and_every_reviewed_identity(self) -> None:
        self.require_candidate()
        specifications = {
            Path(specification["path"]).name: specification
            for specification in self.specifications
        }
        self.assertEqual(
            set(specifications),
            set(manifest_sync.CHAPTER05_GOLDEN_ARCHIVES),
        )
        required = {
            "archive_kind",
            "archive_schema_version",
            "key_count",
            "schema_digest",
            "member_name_digest",
            "sha256",
            "bytes",
            "builder",
            "publisher_sha256",
            "worker_sha256",
            "capture_contract_sha256",
            "exact_source_contract_sha256",
            "publisher_contract_sha256",
            "deterministic_npz_sha256",
            "oracle_acceptance_sha256",
            "publication_acceptance_sha256",
            "fixture_sha256",
            "fixture_payload_digest",
            "raw_capture_schema_version",
            "raw_capture_key_count",
            "raw_capture_schema_digest",
            "accepted_physical_payload_fingerprint",
            "accepted_full_capture_fingerprint",
            "loaded_pinned_python_source_count",
            "loaded_pinned_python_manifest_digest",
            "capture_policy",
        }
        for name, expected in manifest_sync.CHAPTER05_GOLDEN_ARCHIVES.items():
            with self.subTest(archive=name):
                specification = specifications[name]
                self.assertTrue(required <= set(specification))
                for field in (
                    "archive_kind",
                    "archive_schema_version",
                    "key_count",
                    "schema_digest",
                    "member_name_digest",
                    "sha256",
                    "bytes",
                ):
                    self.assertEqual(specification[field], expected[field])
                self.assertEqual(
                    specification["builder"],
                    manifest_sync.CHAPTER05_GOLDEN_BUILDER,
                )
                self.assertEqual(
                    specification["publication_acceptance_sha256"],
                    manifest_sync.CHAPTER05_PUBLICATION_ACCEPTANCE_SHA256,
                )
                self.assertIn("Single owner", specification["scope"])
        integration = specifications[manifest_sync.CHAPTER05_INTEGRATION_NAME]
        self.assertEqual(
            integration["reader_archive_sha256"],
            manifest_sync.CHAPTER05_GOLDEN_ARCHIVES[
                manifest_sync.CHAPTER05_READER_NAME
            ]["sha256"],
        )
        self.assertEqual(
            integration["inventory_mapping_digest"],
            manifest_sync.CHAPTER05_INVENTORY_MAPPING_DIGEST,
        )

    def test_exact_hash_size_schema_and_member_name_digests(self) -> None:
        root = self.require_candidate()
        for name, expected in manifest_sync.CHAPTER05_GOLDEN_ARCHIVES.items():
            path = root / name
            with self.subTest(archive=name):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    expected["sha256"],
                )
                self.assertEqual(path.stat().st_size, expected["bytes"])
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(archive.files, sorted(archive.files))
                    self.assertEqual(len(archive.files), expected["key_count"])
                    self.assertEqual(
                        _member_name_digest(archive.files),
                        expected["member_name_digest"],
                    )
                    self.assertEqual(
                        np.asarray(archive["meta__archive_schema_digest"]).item(),
                        expected["schema_digest"],
                    )

    def test_all_1336_members_have_unit_axes_ownership_and_hash(self) -> None:
        root = self.require_candidate()
        seen = 0
        for name in manifest_sync.CHAPTER05_GOLDEN_ARCHIVES:
            relative_path = f"data/golden/payne_zero/chapter05/{name}"
            inventory = manifest_sync.npz_inventory(
                root / name,
                relative_path,
            )
            self.assertEqual(
                len(inventory),
                manifest_sync.CHAPTER05_GOLDEN_ARCHIVES[name]["key_count"],
            )
            for member, metadata in inventory.items():
                with self.subTest(archive=name, member=member):
                    self.assertEqual(
                        set(metadata),
                        {
                            "shape",
                            "dtype",
                            "unit",
                            "sha256",
                            "axes",
                            "ownership",
                        },
                    )
                    self.assertEqual(
                        len(metadata["axes"]),
                        len(metadata["shape"]),
                    )
                    self.assertTrue(metadata["unit"].strip())
                    self.assertNotIn("unknown", metadata["unit"].lower())
                    self.assertNotIn("unspecified", metadata["unit"].lower())
                    self.assertTrue(metadata["ownership"].strip())
                    self.assertRegex(metadata["sha256"], r"^[0-9a-f]{64}$")
                    seen += 1
        self.assertEqual(seen, 1336)

    def test_representative_units_axes_and_ownership_are_semantic(self) -> None:
        root = self.require_candidate()
        reader_path = (
            "data/golden/payne_zero/chapter05/"
            f"{manifest_sync.CHAPTER05_READER_NAME}"
        )
        integration_path = (
            "data/golden/payne_zero/chapter05/"
            f"{manifest_sync.CHAPTER05_INTEGRATION_NAME}"
        )
        reader = manifest_sync.npz_inventory(
            root / manifest_sync.CHAPTER05_READER_NAME,
            reader_path,
        )
        integration = manifest_sync.npz_inventory(
            root / manifest_sync.CHAPTER05_INTEGRATION_NAME,
            integration_path,
        )
        exact_cases = {
            reader["atmosphere__compact__source"]["unit"]: (
                "erg s^-1 cm^-2 sr^-1 Hz^-1"
            ),
            reader["line_reference__threshold"]["unit"]: (
                "cm^2 g^-1 with embedded 1e-3 and "
                "stimulated-emission division"
            ),
            reader[
                "seam__molecular_boundary__ch_cross_section_times_partition"
            ]["unit"]: "cm^2 times partition-function convention",
            integration[
                "evidence__synthesis__extension__input__invariant__"
                "carbon_boundfree_cross_section_rows"
            ]["unit"]: "cm^2",
            integration[
                "evidence__synthesis__extension__input__invariant__"
                "carbon_freefree_prefactor"
            ]["unit"]: (
                "source-native Kramers free-free prefactor; combined with "
                "hc/kT to yield cm^2"
            ),
        }
        for actual, expected in exact_cases.items():
            self.assertEqual(actual, expected)
        self.assertIn(
            "absorption-weighted source numerator",
            reader[
                "atmosphere__component__ordered_source_numerator_sum"
            ]["unit"],
        )
        self.assertEqual(
            reader["atmosphere__compact__absorption"]["axes"],
            ["regime", "depth", "diagnostic_frequency"],
        )
        self.assertEqual(
            integration["atmosphere_product__source"]["axes"],
            ["regime", "depth", "atmosphere_frequency"],
        )
        exact_audit_repairs = (
            (
                "line_reference__active_index",
                reader["line_reference__active_index"]["unit"],
                "zero-based index",
            ),
            (
                "cia_temperature_fraction",
                reader["seam__molecular_boundary__cia_temperature_fraction"][
                    "unit"
                ],
                "dimensionless",
            ),
            (
                "cia_temperature_table_index",
                integration[
                    "evidence__molecular_boundary__cia_temperature_table_index"
                ]["unit"],
                "zero-based index",
            ),
            (
                "count_263_present",
                integration[
                    "evidence__sampling_boundary__count_263_present"
                ]["unit"],
                "boolean",
            ),
            (
                "count_299_present",
                integration[
                    "evidence__sampling_boundary__count_299_present"
                ]["unit"],
                "boolean",
            ),
            (
                "ifop19_bolometric_like_source",
                integration[
                    "evidence__ifop19__bolometric_like_source"
                ]["unit"],
                (
                    "erg s^-1 cm^-2 sr^-1; "
                    "bolometric sigma*T^4/pi convention"
                ),
            ),
        )
        for member, actual, expected in exact_audit_repairs:
            with self.subTest(audit_repair=member):
                self.assertEqual(actual, expected)
        self.assertNotIn(
            "Hz^-1",
            integration["evidence__ifop19__bolometric_like_source"]["unit"],
        )
        self.assertEqual(
            reader["atmosphere__component__absorption_name"]["axes"],
            ["absorption_component"],
        )
        self.assertEqual(
            reader["atmosphere__component__source_name"]["axes"],
            ["absorption_component"],
        )
        self.assertEqual(
            reader["atmosphere__component__scattering_name"]["axes"],
            ["scattering_component"],
        )
        alias = next(
            member
            for member in integration
            if member.startswith("alias__")
        )
        self.assertIn("reader-owned", integration[alias]["ownership"])
        self.assertEqual(integration[alias]["unit"], "ordered identifier")

    def test_fake_paths_names_regimes_and_fields_are_rejected(self) -> None:
        self.require_candidate()
        reader_path = (
            "data/golden/payne_zero/chapter05/"
            f"{manifest_sync.CHAPTER05_READER_NAME}"
        )
        integration_path = (
            "data/golden/payne_zero/chapter05/"
            f"{manifest_sync.CHAPTER05_INTEGRATION_NAME}"
        )
        rejected = (
            (
                f"elsewhere/{manifest_sync.CHAPTER05_READER_NAME}",
                "atmosphere__compact__absorption",
            ),
            (reader_path, "future__apparently_valid__absorption"),
            (
                integration_path,
                "alias__future_regime__atmosphere__absorption__reader_member",
            ),
            (
                integration_path,
                "evidence__hot_dwarf__synthesis__future_field__absorption",
            ),
            (
                integration_path,
                "oracle__identity__source__future_module__sha256",
            ),
        )
        for path, member in rejected:
            with self.subTest(path=path, member=member):
                with self.assertRaises(KeyError):
                    manifest_sync.chapter05_golden_unit(path, member)

    def test_empty_publication_state_keeps_manifest_bytes_unchanged(self) -> None:
        manifest = json.loads(
            manifest_sync.MANIFEST_PATH.read_text(encoding="utf-8")
        )
        prefix = "data/golden/payne_zero/chapter05/"
        manifest["entries"] = [
            entry
            for entry in manifest["entries"]
            if not entry["path"].startswith(prefix)
        ]
        text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            manifest_path = temporary_root / "MANIFEST.json"
            empty_golden = temporary_root / "empty"
            empty_golden.mkdir()
            manifest_path.write_text(text, encoding="utf-8")
            before = manifest_path.read_bytes()
            with (
                mock.patch.object(
                    manifest_sync, "MANIFEST_PATH", manifest_path
                ),
                mock.patch.object(
                    manifest_sync,
                    "CHAPTER05_GOLDEN_DIRECTORY",
                    empty_golden,
                ),
            ):
                manifest_sync.main()
            self.assertEqual(manifest_path.read_bytes(), before)

    def test_partial_pair_fails_before_manifest_mutation(self) -> None:
        root = self.require_candidate()
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            partial = temporary_root / "chapter05"
            partial.mkdir()
            shutil.copyfile(
                root / manifest_sync.CHAPTER05_READER_NAME,
                partial / manifest_sync.CHAPTER05_READER_NAME,
            )
            manifest_path = temporary_root / "MANIFEST.json"
            shutil.copyfile(manifest_sync.MANIFEST_PATH, manifest_path)
            before = manifest_path.read_bytes()
            with (
                mock.patch.object(
                    manifest_sync, "MANIFEST_PATH", manifest_path
                ),
                mock.patch.object(
                    manifest_sync, "CHAPTER05_GOLDEN_DIRECTORY", partial
                ),
                self.assertRaisesRegex(RuntimeError, "partial Chapter 5"),
            ):
                manifest_sync.main()
            self.assertEqual(manifest_path.read_bytes(), before)

    def test_bad_candidate_bytes_fail_before_manifest_mutation(self) -> None:
        root = self.require_candidate()
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            bad_pair = temporary_root / "chapter05"
            bad_pair.mkdir()
            for name in manifest_sync.CHAPTER05_GOLDEN_ARCHIVES:
                shutil.copyfile(root / name, bad_pair / name)
            reader_path = bad_pair / manifest_sync.CHAPTER05_READER_NAME
            corrupted = bytearray(reader_path.read_bytes())
            corrupted[0] ^= 1
            reader_path.write_bytes(corrupted)
            manifest_path = temporary_root / "MANIFEST.json"
            shutil.copyfile(manifest_sync.MANIFEST_PATH, manifest_path)
            before = manifest_path.read_bytes()
            with (
                mock.patch.object(
                    manifest_sync, "MANIFEST_PATH", manifest_path
                ),
                mock.patch.object(
                    manifest_sync, "CHAPTER05_GOLDEN_DIRECTORY", bad_pair
                ),
                self.assertRaisesRegex(RuntimeError, "has SHA-256"),
            ):
                manifest_sync.main()
            self.assertEqual(manifest_path.read_bytes(), before)

    def test_atomic_replace_failure_keeps_original_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "MANIFEST.json"
            manifest_path.write_text("original\n", encoding="utf-8")
            before = manifest_path.read_bytes()
            with (
                mock.patch.object(
                    manifest_sync, "MANIFEST_PATH", manifest_path
                ),
                mock.patch.object(
                    manifest_sync.os,
                    "replace",
                    side_effect=OSError("injected replace failure"),
                ),
                self.assertRaisesRegex(OSError, "replace failure"),
            ):
                manifest_sync._atomic_write_manifest("replacement\n")
            self.assertEqual(manifest_path.read_bytes(), before)
            self.assertEqual(
                list(manifest_path.parent.glob(".MANIFEST.json.*.tmp")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
