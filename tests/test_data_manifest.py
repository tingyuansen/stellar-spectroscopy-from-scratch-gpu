"""Repository-wide role and byte-identity checks for textbook data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DataManifestTests(unittest.TestCase):
    """Keep every active textbook data artifact owned and reproducible."""

    def test_every_manifest_entry_has_an_allowed_role_and_matching_hash(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "data" / "MANIFEST.json").read_text()
        )
        self.assertEqual(manifest["schema_version"], 1)
        allowed_roles = {"static", "subset", "fixture", "golden"}
        seen_paths = set()
        for entry in manifest["entries"]:
            relative_path = entry["path"]
            with self.subTest(path=relative_path):
                self.assertNotIn(relative_path, seen_paths)
                seen_paths.add(relative_path)
                self.assertIn(entry["role"], allowed_roles)
                path = REPOSITORY_ROOT / relative_path
                self.assertTrue(path.is_file())
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(actual, entry["sha256"])
                self.assertEqual(path.stat().st_size, entry["bytes"])

    def test_role_matches_physical_directory(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "data" / "MANIFEST.json").read_text()
        )
        role_directory = {
            "static": "static",
            "subset": "subsets",
            "fixture": "fixtures",
            "golden": "golden",
        }
        for entry in manifest["entries"]:
            parts = Path(entry["path"]).parts
            with self.subTest(path=entry["path"]):
                self.assertGreaterEqual(len(parts), 3)
                self.assertEqual(parts[0], "data")
                self.assertEqual(parts[1], role_directory[entry["role"]])

    def test_npz_inventory_is_exhaustive_and_matches_stored_arrays(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "data" / "MANIFEST.json").read_text()
        )
        for entry in manifest["entries"]:
            if entry["format"] != "npz":
                continue
            path = REPOSITORY_ROOT / entry["path"]
            with self.subTest(path=entry["path"]), np.load(
                path, allow_pickle=False
            ) as archive:
                self.assertEqual(set(entry["arrays"]), set(archive.files))
                for name in archive.files:
                    values = np.asarray(archive[name])
                    record = entry["arrays"][name]
                    self.assertEqual(record["shape"], list(values.shape))
                    self.assertEqual(record["dtype"], str(values.dtype))
                    self.assertTrue(record["unit"])
                    array_sha256 = hashlib.sha256(
                        np.ascontiguousarray(values).tobytes(order="C")
                    ).hexdigest()
                    self.assertEqual(record["sha256"], array_sha256)

    def test_schema_manifest_inventory_matches_all_declared_arrays(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "data" / "MANIFEST.json").read_text()
        )
        entry = next(
            item
            for item in manifest["entries"]
            if item["path"].endswith("atmosphere_schema.json")
        )
        schema = json.loads((REPOSITORY_ROOT / entry["path"]).read_text())
        expected = {field["name"] for field in schema["required_arrays"]}
        self.assertEqual(set(entry["declared_arrays"]), expected)
        for field in schema["required_arrays"]:
            record = entry["declared_arrays"][field["name"]]
            self.assertEqual(record["shape"], field["shape"])
            self.assertEqual(record["unit"], field["unit"])

    def test_locally_built_npz_artifacts_name_their_builder(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "data" / "MANIFEST.json").read_text()
        )
        for entry in manifest["entries"]:
            if entry["role"] not in {"fixture", "golden"}:
                continue
            with self.subTest(path=entry["path"]):
                builder = REPOSITORY_ROOT / entry["builder"]
                self.assertTrue(builder.is_file())

    def test_every_named_builder_exists(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "data" / "MANIFEST.json").read_text()
        )
        for entry in manifest["entries"]:
            if "builder" not in entry:
                continue
            with self.subTest(path=entry["path"]):
                self.assertTrue((REPOSITORY_ROOT / entry["builder"]).is_file())

    def test_chapter03_source_derived_assets_record_provenance(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "data" / "MANIFEST.json").read_text()
        )
        chapter03_names = {
            "special_partition_tables.npz",
            "iron_group_partition_tables.npz",
            "ionization_potential_tables.npz",
            "packed_level_metadata.npz",
            "isotope_tables.npz",
            "atomic_masses.npz",
            "partition_saha_tables.npz",
            "chapter03_synthesis_eos_state.npz",
        }
        entries = [
            entry
            for entry in manifest["entries"]
            if Path(entry["path"]).name in chapter03_names
        ]
        self.assertEqual(len(entries), len(chapter03_names))
        for entry in entries:
            with self.subTest(path=entry["path"]):
                self.assertEqual(
                    entry["source_commit"],
                    "9c44001feae40b85146630499e6f8a5fed42e5af",
                )
                self.assertFalse(entry["requires_optional_full_catalog"])
                self.assertTrue(
                    entry.get("source_sha256")
                    or entry.get("source_bundle_sha256")
                )

    def test_chapter03_harness_assets_record_roles_inputs_and_builders(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "data" / "MANIFEST.json").read_text()
        )
        fixture_name = "chapter03_atom_only_inputs.npz"
        golden_names = {
            "chapter03_atmosphere_saha_outputs.npz",
            "chapter03_atmosphere_atomic_state.npz",
            "chapter03_packed_bridge_outputs.npz",
            "chapter03_synthesis_atomic_state_cpu_float64.npz",
        }
        entries = {
            Path(entry["path"]).name: entry for entry in manifest["entries"]
        }
        fixture = entries[fixture_name]
        self.assertEqual(fixture["role"], "fixture")
        self.assertEqual(
            fixture["source_sha256"],
            "ecc2856d69c7d96bcdfb6d50988addcd06c68f6e492d9c1f08d21492984ad6c9",
        )
        self.assertEqual(
            fixture["builder"],
            "scripts/build_chapter03_atom_only_fixture.py",
        )
        for name in golden_names:
            with self.subTest(golden=name):
                entry = entries[name]
                self.assertEqual(entry["role"], "golden")
                self.assertEqual(
                    entry["source_commit"],
                    "9c44001feae40b85146630499e6f8a5fed42e5af",
                )
                self.assertEqual(entry["fixture_sha256"], fixture["sha256"])
                self.assertEqual(
                    entry["builder"],
                    "scripts/build_chapter03_payne_zero_goldens.py",
                )
                self.assertFalse(entry["requires_optional_full_catalog"])


if __name__ == "__main__":
    unittest.main()
