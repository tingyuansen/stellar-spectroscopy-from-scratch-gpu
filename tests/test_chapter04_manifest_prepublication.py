"""Prepublication identity and unit gates for the five Chapter 4 goldens."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import sync_data_manifest as manifest_sync


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_GOLDEN_DIR = REPOSITORY_ROOT / "data" / "golden" / "payne_zero" / "chapter04"

EXPECTED_ARCHIVES = {
    "chapter04_molecular_constants_cpu_float64.npz": (
        "cf742ecc5181589e2b7f8b56c7b2d82bd203c9f303de263a5b1c863adaba40a0",
        235675,
        "molecular_constants",
    ),
    "chapter04_atmosphere_molecular_state_cpu_float64.npz": (
        "e0c80f66bf74776d29947c6ced204402c36678edce9b3bdddebd9bd79713ec2a",
        1047808,
        "atmosphere_molecular_state",
    ),
    "chapter04_synthesis_molecular_full_cpu_float64.npz": (
        "c78393d8525bb637706b5f4dbb17aae7f24660e468bbbfd84dff659ceb8edf31",
        471636,
        "synthesis_molecular_full",
    ),
    "chapter04_synthesis_molecular_fixed_cpu_float64.npz": (
        "c4aeff4b3afba423e3ab9613bd0eef813b3487f7bf792e6b95bcbbf0db46ca12",
        649052,
        "synthesis_molecular_fixed",
    ),
    "chapter04_molecular_public_mapping_cpu_float64.npz": (
        "a40dcd105641e4620a7c8e9362f0270437e3fe87e7e1e9ed3209b9b566f32628",
        590138,
        "molecular_public_mapping",
    ),
}


def _archive_root() -> Path | None:
    candidate = os.environ.get("CHAPTER04_CANDIDATE_DIR")
    if candidate is not None:
        return Path(candidate)
    if REPOSITORY_GOLDEN_DIR.is_dir():
        return REPOSITORY_GOLDEN_DIR
    return None


class Chapter04ManifestPrepublicationTests(unittest.TestCase):
    """Reject stale identities, missing ownership, and ambiguous units."""

    def test_static_specs_pin_five_single_owner_archives(self) -> None:
        specifications = {
            Path(specification["path"]).name: specification
            for specification in manifest_sync.CHAPTER04_ENTRY_SPECS
            if specification["role"] == "golden"
        }
        self.assertEqual(set(specifications), set(EXPECTED_ARCHIVES))
        scopes = set()
        for name, (expected_sha256, expected_bytes, _) in EXPECTED_ARCHIVES.items():
            with self.subTest(archive=name):
                specification = specifications[name]
                self.assertEqual(specification["sha256"], expected_sha256)
                self.assertEqual(specification["bytes"], expected_bytes)
                self.assertEqual(
                    specification["source_commit"],
                    manifest_sync.PAYNE_ZERO_COMMIT,
                )
                self.assertEqual(
                    specification["fixture_sha256"],
                    manifest_sync.CHAPTER04_FIXTURE_SHA256,
                )
                self.assertEqual(
                    specification["publisher_sha256"],
                    manifest_sync.CHAPTER04_PUBLISHER_SHA256,
                )
                self.assertEqual(
                    specification["oracle_acceptance_sha256"],
                    manifest_sync.CHAPTER04_ACCEPTANCE_SHA256,
                )
                self.assertEqual(
                    specification["capture_contract_sha256"],
                    manifest_sync.CHAPTER04_CAPTURE_CONTRACT_SHA256,
                )
                self.assertEqual(
                    specification["builder"],
                    manifest_sync.CHAPTER04_GOLDEN_BUILDER,
                )
                self.assertEqual(
                    specification["source"],
                    "/Users/ysting/payne-zero",
                )
                self.assertEqual(
                    specification["capture_policy"],
                    "two fresh byte-identical capture sets",
                )
                self.assertFalse(specification["requires_optional_full_catalog"])
                self.assertIn("Single owner", specification["scope"])
                scopes.add(specification["scope"])
                if name != "chapter04_molecular_constants_cpu_float64.npz":
                    self.assertEqual(
                        specification["constants_archive_sha256"],
                        manifest_sync.CHAPTER04_CONSTANTS_SHA256,
                    )
        self.assertEqual(len(scopes), len(EXPECTED_ARCHIVES))

    def test_all_candidate_or_published_members_have_exact_units(self) -> None:
        root = _archive_root()
        if root is None:
            self.skipTest(
                "Chapter 4 goldens are not published; set "
                "CHAPTER04_CANDIDATE_DIR for the prepublication gate"
            )
        self.assertTrue(root.is_dir())
        self.assertEqual(
            {path.name for path in root.glob("*.npz")},
            set(EXPECTED_ARCHIVES),
        )
        seen_members = 0
        for name in EXPECTED_ARCHIVES:
            path = root / name
            with self.subTest(archive=name):
                self.assertTrue(path.is_file())
                relative = f"data/golden/payne_zero/chapter04/{name}"
                with np.load(path, allow_pickle=False) as archive:
                    for member in archive.files:
                        unit = manifest_sync.chapter04_golden_unit(relative, member)
                        self.assertEqual(manifest_sync.unit_for(relative, member), unit)
                        self.assertTrue(unit.strip())
                        self.assertNotIn("unknown", unit.lower())
                        self.assertNotIn("unspecified", unit.lower())
                        seen_members += 1
        self.assertEqual(seen_members, 1171)

    def test_candidate_or_published_archives_match_all_pinned_identities(
        self,
    ) -> None:
        root = _archive_root()
        if root is None:
            self.skipTest(
                "Chapter 4 goldens are not published; set "
                "CHAPTER04_CANDIDATE_DIR for the prepublication gate"
            )
        for name, (
            expected_sha256,
            expected_bytes,
            expected_kind,
        ) in EXPECTED_ARCHIVES.items():
            path = root / name
            with self.subTest(archive=name):
                self.assertEqual(path.stat().st_size, expected_bytes)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    expected_sha256,
                )
                with np.load(path, allow_pickle=False) as archive:

                    def scalar(member: str) -> object:
                        return np.asarray(archive[member]).item()

                    self.assertEqual(scalar("meta__archive_kind"), expected_kind)
                    self.assertEqual(
                        scalar("meta__payne_zero_commit"),
                        manifest_sync.PAYNE_ZERO_COMMIT,
                    )
                    self.assertEqual(
                        scalar("meta__fixture_sha256"),
                        manifest_sync.CHAPTER04_FIXTURE_SHA256,
                    )
                    self.assertEqual(
                        scalar("meta__publisher_sha256"),
                        manifest_sync.CHAPTER04_PUBLISHER_SHA256,
                    )
                    self.assertEqual(
                        scalar("meta__oracle_acceptance_sha256"),
                        manifest_sync.CHAPTER04_ACCEPTANCE_SHA256,
                    )
                    self.assertEqual(
                        scalar("meta__capture_contract_sha256"),
                        manifest_sync.CHAPTER04_CAPTURE_CONTRACT_SHA256,
                    )
                    if name == ("chapter04_molecular_constants_cpu_float64.npz"):
                        self.assertNotIn(
                            "meta__constants_archive_sha256", archive.files
                        )
                    else:
                        self.assertEqual(
                            scalar("meta__constants_archive_sha256"),
                            manifest_sync.CHAPTER04_CONSTANTS_SHA256,
                        )

    def test_unit_router_is_fail_closed(self) -> None:
        path = (
            "data/golden/payne_zero/chapter04/"
            "chapter04_synthesis_molecular_full_cpu_float64.npz"
        )
        rejected = (
            (path, "future_unreviewed_density"),
            (path, "alias__future_unreviewed"),
            (path, "oracle__future_unreviewed_sha256"),
            (
                "data/golden/payne_zero/chapter04/"
                "chapter04_molecular_public_mapping_cpu_float64.npz",
                "alias__future_unreviewed",
            ),
            (
                "data/golden/payne_zero/chapter04/"
                "chapter04_molecular_constants_cpu_float64.npz",
                "atmosphere__source_future_unreviewed",
            ),
            (
                "data/golden/payne_zero/chapter04/not_declared.npz",
                "meta__archive_kind",
            ),
            (
                "/wrong/location/chapter04_synthesis_molecular_full_cpu_float64.npz",
                "meta__archive_kind",
            ),
            (
                "data/golden/payne_zero/chapter04/"
                "chapter04_atmosphere_molecular_state_cpu_float64.npz",
                "oracle__source_sha256__atomic_masses",
            ),
            (path, "oracle__source_atmosphere_io_sha256"),
            (
                "data/golden/payne_zero/chapter04/"
                "chapter04_molecular_constants_cpu_float64.npz",
                "atmosphere__source_sha256__pipeline_source",
            ),
            (
                "data/golden/payne_zero/chapter04/"
                "chapter04_molecular_constants_cpu_float64.npz",
                "synthesis__meta__source_atmosphere_io_sha256",
            ),
            (
                "data/golden/payne_zero/chapter04/"
                "chapter04_molecular_constants_cpu_float64.npz",
                "atmosphere__boundary_future_uint64_bits",
            ),
            (
                "data/golden/payne_zero/chapter04/"
                "chapter04_molecular_constants_cpu_float64.npz",
                "atmosphere__boundary_future_branch_mask",
            ),
            (
                "data/golden/payne_zero/chapter04/"
                "chapter04_molecular_constants_cpu_float64.npz",
                "atmosphere__h2_probe_future_uint64_bits",
            ),
            (
                "data/golden/payne_zero/chapter04/"
                "chapter04_molecular_constants_cpu_float64.npz",
                "synthesis__catalog__future__active_molecule_mask",
            ),
            (
                "data/golden/payne_zero/chapter04/"
                "chapter04_molecular_constants_cpu_float64.npz",
                "synthesis__catalog__future__hard_coded_molecular_atomic_masses_sha256",
            ),
        )
        for rejected_path, member in rejected:
            with self.subTest(path=rejected_path, member=member):
                with self.assertRaises(KeyError):
                    manifest_sync.chapter04_golden_unit(
                        rejected_path,
                        member,
                    )

    def test_representative_scientific_units_are_semantically_exact(self) -> None:
        root = "data/golden/payne_zero/chapter04"
        cases = {
            (
                f"{root}/chapter04_molecular_constants_cpu_float64.npz",
                "synthesis__boundary__synthesis_pre_newton_formation_constants",
            ): "source-native molecule-dependent formation factor",
            (
                f"{root}/chapter04_atmosphere_molecular_state_cpu_float64.npz",
                "full_charge_square_density",
            ): "cm^-3, charge-square weighted",
            (
                f"{root}/chapter04_atmosphere_molecular_state_cpu_float64.npz",
                "full_transformed_molecular_equation_densities",
            ): (
                "source-native species-dependent N/U equation factor after "
                "partition and thermal normalization; not cm^-3"
            ),
            (
                f"{root}/chapter04_atmosphere_molecular_state_cpu_float64.npz",
                "full_postsolve_row_scaled_residual",
            ): "dimensionless row-scaled molecular conservation residual",
            (
                f"{root}/chapter04_synthesis_molecular_full_cpu_float64.npz",
                "call_0__diag__residual",
            ): "cm^-3 molecular conservation residual",
            (
                f"{root}/chapter04_synthesis_molecular_full_cpu_float64.npz",
                "call_0__diag__normalized_residual",
            ): "dimensionless row-scaled molecular conservation residual",
            (
                f"{root}/chapter04_synthesis_molecular_full_cpu_float64.npz",
                "alias__state__electron_density_member",
            ): "NPZ member-name reference for a deduplicated route array",
            (
                f"{root}/chapter04_molecular_public_mapping_cpu_float64.npz",
                "co_reconstruction__transformed_equation_densities",
            ): (
                "source-native species-dependent N/U equation factor after "
                "partition and thermal normalization; not cm^-3"
            ),
            (
                f"{root}/chapter04_molecular_public_mapping_cpu_float64.npz",
                "mapping__public_columns",
            ): "zero-based public atmosphere-cube column index",
            (
                f"{root}/chapter04_molecular_public_mapping_cpu_float64.npz",
                "line_population__ground_discrimination_mask",
            ): "boolean grounded versus no-ground partition-policy mask",
        }
        for (path, member), expected in cases.items():
            with self.subTest(path=path, member=member):
                self.assertEqual(
                    manifest_sync.chapter04_golden_unit(path, member),
                    expected,
                )

    def test_hash_mismatch_fails_before_manifest_mutation(self) -> None:
        fixture_relative = "data/fixtures/chapter04_molecular_inputs.npz"
        fixture = REPOSITORY_ROOT / fixture_relative
        self.assertTrue(fixture.is_file())
        specification = {
            "path": fixture_relative,
            "role": "golden",
            "sha256": "0" * 64,
            "bytes": fixture.stat().st_size,
        }
        manifest = {
            "schema_version": 1,
            "entries": [
                {
                    **specification,
                    "format": "npz",
                    "arrays": {},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "MANIFEST.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            before = manifest_path.read_bytes()
            with (
                mock.patch.object(
                    manifest_sync,
                    "MANIFEST_PATH",
                    manifest_path,
                ),
                mock.patch.object(
                    manifest_sync,
                    "CHAPTER04_ENTRY_SPECS",
                    (specification,),
                ),
                self.assertRaisesRegex(RuntimeError, "has SHA-256"),
            ):
                manifest_sync.main()
            self.assertEqual(manifest_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
