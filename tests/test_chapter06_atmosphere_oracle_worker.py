"""Focused contracts for the in-memory Chapter 6 atmosphere oracle worker."""

from __future__ import annotations

import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import chapter06_atmosphere_oracle_worker as worker


def _data_snapshot() -> dict[str, tuple[int, str]]:
    """Hash every canonical data file without following links."""

    snapshot: dict[str, tuple[int, str]] = {}
    for path in sorted(worker.REPOSITORY_ROOT.joinpath("data").rglob("*")):
        if path.is_symlink():
            snapshot[path.relative_to(worker.REPOSITORY_ROOT).as_posix()] = (
                -1,
                "symlink",
            )
        elif path.is_file():
            snapshot[path.relative_to(worker.REPOSITORY_ROOT).as_posix()] = (
                path.stat().st_size,
                worker.sha256(path),
            )
    return snapshot


def _fresh_environment(cache: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(worker.ORACLE_ENVIRONMENT)
    environment["NUMBA_CACHE_DIR"] = str(cache)
    environment.pop("PAYNE_ZERO_DATA_ROOT", None)
    return environment


def _fresh_detailed_capture(cache: Path) -> dict[str, object]:
    """Collect detailed oracle evidence in a true pre-Numba child process."""

    child_source = """
import json
from scripts import chapter06_atmosphere_oracle_worker as worker
results = worker.build_oracle_results()
print(json.dumps({
    "summary": worker.summarize(results),
    "caps_support_indices": results["seam__caps__support_indices"].tolist(),
    "wing_nextafter_support_indices": (
        results["seam__wing_nextafter__support_indices"].tolist()
    ),
    "gates_interior_deposited": (
        results["seam__gates__interior_deposited"].tolist()
    ),
    "projection_owned_slots_change_route": (
        results["seam__projection__owned_actual_slots_change_route"].tolist()
    ),
    "manifest_semantic_labels_authoritative": bool(
        results["identity__manifest_semantic_labels_authoritative"]
    ),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", child_source],
        cwd=worker.REPOSITORY_ROOT,
        env=_fresh_environment(cache),
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if completed.stderr:
        raise AssertionError(completed.stderr)
    return json.loads(completed.stdout)


class Chapter06AtmosphereOracleWorkerTests(unittest.TestCase):
    """Pin source authority, science seams, and the no-write lifecycle."""

    def test_canonical_fixture_and_manifest_bindings_are_exact(self) -> None:
        fixture = worker.load_fixture()
        self.assertEqual(set(fixture), set(worker.FIXTURE_MEMBER_CONTRACT))
        self.assertEqual(len(fixture), worker.FIXTURE_MEMBER_COUNT)
        self.assertEqual(worker.schema_digest(fixture), worker.FIXTURE_SCHEMA_DIGEST)
        for name, (shape, dtype, member_hash) in worker.FIXTURE_MEMBER_CONTRACT.items():
            self.assertEqual(fixture[name].shape, shape, name)
            self.assertEqual(fixture[name].dtype, dtype, name)
            self.assertEqual(worker.array_sha256(fixture[name]), member_hash, name)

        identities = worker.verify_manifest_bindings()
        self.assertEqual(
            identities["fixture_entry_digest"],
            "cfaf118c11c76a7b97198cb7fe7e3c0f78863f5eee71ea04bbe78c223d3653af",
        )
        self.assertRegex(identities["subset_entry_digest"], r"^[0-9a-f]{64}$")

    def test_three_manifest_label_conflicts_are_never_semantic_authority(
        self,
    ) -> None:
        manifest = worker._duplicate_free_json(worker.MANIFEST_PATH)
        entry = worker._manifest_entry(
            manifest,
            worker.FIXTURE_PATH.relative_to(worker.REPOSITORY_ROOT).as_posix(),
        )
        arrays = entry["arrays"]
        observed = {
            "arrays.actual_population_slot_values.unit": arrays[
                "actual_population_slot_values"
            ]["unit"],
            "arrays.packed_species_slot.unit": arrays["packed_species_slot"]["unit"],
            "arrays.wavelength_bin_edges.unit": arrays["wavelength_bin_edges"]["unit"],
        }
        self.assertEqual(observed, worker.CONTRADICTORY_MANIFEST_LABELS)
        with self.assertRaisesRegex(
            worker.OracleIdentityError, "may not be treated as scientific authority"
        ):
            worker.verify_manifest_bindings(treat_array_labels_as_authority=True)

        fixture = worker.load_fixture()
        self.assertEqual(int(fixture["packed_species_slot"][0]), 3510)
        self.assertEqual(abs(int(fixture["packed_species_slot"][0])) // 10, 351)
        self.assertEqual(int(fixture["wavelength_bin_edges"][-1]), 2**30)
        self.assertEqual(fixture["actual_population_slot_values"].dtype, np.float64)

    def test_fixture_reconstruction_has_only_declared_owned_columns(self) -> None:
        fixture = worker.load_fixture()
        actual, support, widths = worker.reconstruct_public_call_arrays(fixture)
        self.assertEqual(actual.shape, (80, 1006))
        self.assertEqual(support.shape, (80, 1006))
        self.assertEqual(widths.shape, (80, 1006))
        self.assertEqual(actual.dtype, np.float64)
        self.assertEqual(support.dtype, np.float64)
        self.assertEqual(widths.dtype, np.float64)
        np.testing.assert_array_equal(
            actual[:, [0, 2, 840]],
            fixture["actual_population_slot_values"],
        )
        actual_without_owned = actual.copy()
        actual_without_owned[:, [0, 2, 840]] = 0.0
        self.assertFalse(np.any(actual_without_owned))
        self.assertEqual(np.flatnonzero(np.any(support != 0.0, axis=0)).tolist(), [350])
        self.assertEqual(np.flatnonzero(np.any(widths != 0.0, axis=0)).tolist(), [350])

    def test_golden_alternate_and_symlink_fixture_paths_fail_before_read(self) -> None:
        forbidden_golden = (
            worker.GOLDEN_ROOT
            / "payne_zero/chapter06/chapter06_atmosphere_one_line_cpu.npz"
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            alias = temporary_root / "fixture-alias.npz"
            alias.symlink_to(worker.FIXTURE_PATH)
            alternate = temporary_root / "alternate.npz"
            with mock.patch.object(
                worker.np,
                "load",
                side_effect=AssertionError("np.load must not be reached"),
            ):
                with self.assertRaisesRegex(
                    worker.OracleIdentityError, "golden artifact"
                ):
                    worker.load_fixture(forbidden_golden)
                with self.assertRaisesRegex(worker.OracleIdentityError, "symlink"):
                    worker.load_fixture(alias)
                with self.assertRaisesRegex(
                    worker.OracleIdentityError, "canonical path"
                ):
                    worker.load_fixture(alternate)

    def test_worker_exposes_no_writer_or_publication_interface(self) -> None:
        parameters = inspect.signature(worker.build_oracle_results).parameters
        forbidden = ("output", "destination", "publish", "serialize", "golden")
        self.assertFalse(
            any(fragment in name for name in parameters for fragment in forbidden)
        )
        with mock.patch("sys.stderr", new=io.StringIO()):
            for arguments in (
                ["--output", "/tmp/forbidden.npz"],
                ["--publish"],
                ["--golden", "/tmp/forbidden.npz"],
            ):
                with self.subTest(arguments=arguments):
                    with self.assertRaises(SystemExit):
                        worker.parse_args(arguments)

        source = worker.WORKER_PATH.read_text(encoding="utf-8")
        for forbidden_call in (
            "np.save(",
            "np.savez(",
            "np.savez_compressed(",
            ".write_bytes(",
            ".write_text(",
            "os.replace(",
            "os.rename(",
        ):
            self.assertNotIn(forbidden_call, source)

    def test_deterministic_result_and_fingerprints_are_order_independent(
        self,
    ) -> None:
        source = np.asarray([1.0, 2.0], dtype=np.float64)
        first = worker.deterministic_result({"z": np.asarray("value"), "a": source})
        second = worker.deterministic_result(
            {"a": np.asarray([1.0, 2.0]), "z": np.asarray("value")}
        )
        self.assertEqual(list(first), ["a", "z"])
        self.assertEqual(worker.mapping_digest(first), worker.mapping_digest(second))
        self.assertEqual(worker.schema_digest(first), worker.schema_digest(second))
        source[0] = -1.0
        np.testing.assert_array_equal(
            first["a"], np.asarray([1.0, 2.0], dtype=np.float64)
        )
        with self.assertRaisesRegex(TypeError, "object dtype"):
            worker.deterministic_result({"bad": np.asarray([object()], dtype=object)})

    def test_environment_rejects_populated_symlink_and_inherited_data_root(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            occupied = root / "occupied"
            occupied.mkdir()
            occupied.joinpath("sentinel").write_text("occupied", encoding="utf-8")
            environment = dict(worker.ORACLE_ENVIRONMENT)
            environment["NUMBA_CACHE_DIR"] = str(occupied)
            with mock.patch.dict(os.environ, environment, clear=False):
                with self.assertRaisesRegex(
                    worker.OracleEnvironmentError, "truly empty"
                ):
                    worker.require_environment()

            target = root / "target"
            target.mkdir()
            alias = root / "cache-link"
            alias.symlink_to(target, target_is_directory=True)
            environment["NUMBA_CACHE_DIR"] = str(alias)
            with mock.patch.dict(os.environ, environment, clear=False):
                with self.assertRaisesRegex(worker.OracleEnvironmentError, "symlink"):
                    worker.require_environment()

        with mock.patch.dict(
            os.environ,
            {"PAYNE_ZERO_DATA_ROOT": str(worker.PINNED_DATA_ROOT)},
            clear=False,
        ):
            with mock.patch.object(
                worker.importlib,
                "import_module",
                side_effect=AssertionError("import must not be reached"),
            ):
                with self.assertRaisesRegex(
                    worker.OracleIdentityError, "must be absent"
                ):
                    worker.load_pinned_modules()

    @unittest.skipUnless(
        worker.PINNED_ROOT.is_dir(), "pinned Payne Zero checkout absent"
    )
    def test_preimport_source_table_and_authority_closure_is_exact(self) -> None:
        identities = worker.verify_preimport_identities()
        self.assertEqual(identities["payne_zero_commit"], worker.PINNED_COMMIT)
        self.assertEqual(
            identities["table__line_opacity_tables__sha256"],
            worker.LINE_TABLE_SHA256,
        )
        self.assertEqual(
            identities["table__molecular_equilibrium_tables__sha256"],
            worker.MOLECULAR_EQUILIBRIUM_TABLE_SHA256,
        )
        self.assertEqual(identities["raw_subset_sha256"], worker.SUBSET_SHA256)
        accepted = worker._accepted_python_manifest()
        self.assertEqual(len(accepted), 35)
        for name, expected_hash in worker.CRITICAL_SOURCE_HASHES.items():
            self.assertEqual(worker.sha256(worker.PINNED_ROOT / name), expected_hash)

    @unittest.skipUnless(
        worker.PINNED_ROOT.is_dir(), "pinned Payne Zero checkout absent"
    )
    def test_complete_in_memory_capture_contains_exact_science_and_seams(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as cache:
            detailed = _fresh_detailed_capture(Path(cache))
        summary = detailed["summary"]
        self.assertTrue(summary["capture_scope_complete"])
        self.assertEqual(summary["dense_shape"], [80, 30000])
        self.assertEqual(summary["pre_dtype"], "float32")
        self.assertEqual(summary["post_dtype"], "float64")
        self.assertEqual(summary["selected_line_count"], 1)
        self.assertEqual(summary["nonzero_count"], 240)
        self.assertEqual(summary["nonzero_count_per_depth"], [3] * 80)
        self.assertEqual(summary["gate_depth_1based"], list(range(8, 81, 8)))
        self.assertEqual(summary["gate_active"], [True] * 10)
        self.assertEqual(
            summary["dense_pre_stimulated_sha256"],
            worker.EXPECTED_DENSE_PRE_SHA256,
        )
        self.assertEqual(
            summary["dense_post_stimulated_sha256"],
            worker.EXPECTED_DENSE_POST_SHA256,
        )
        self.assertEqual(
            summary["sparse_index_sha256"],
            worker.EXPECTED_SPARSE_INDEX_SHA256,
        )
        self.assertEqual(
            summary["sparse_pre_value_sha256"],
            worker.EXPECTED_SPARSE_PRE_SHA256,
        )
        self.assertEqual(
            summary["sparse_post_value_sha256"],
            worker.EXPECTED_SPARSE_POST_SHA256,
        )
        self.assertTrue(summary["serial_route_proved"])
        self.assertEqual(summary["depth_rejections"], [79, 81])
        self.assertEqual(summary["wing_cap_counts"], [101, 100])
        self.assertTrue(summary["gate_topology_verified"])
        self.assertEqual(summary["center_cutoff_survived"], [False, True, False, True])
        self.assertTrue(summary["dtype_boundary_verified"])
        self.assertTrue(summary["projection_lifecycle_verified"])
        self.assertFalse(summary["fixture_build_performed"])
        self.assertFalse(summary["golden_read_performed"])
        self.assertFalse(summary["serialization_performed"])
        self.assertFalse(summary["publication_performed"])
        self.assertEqual(
            summary["dynamic_npz_reads"],
            [
                "pin:source_data_files/atmosphere_tables/line_opacity_tables.npz",
                (
                    "pin:source_data_files/atmosphere_tables/"
                    "molecular_equilibrium_tables.npz"
                ),
                ("repository:data/fixtures/chapter06_atmosphere_one_line_inputs.npz"),
            ],
        )

        self.assertEqual(
            detailed["caps_support_indices"],
            list(range(101, 302)),
        )
        self.assertEqual(
            detailed["wing_nextafter_support_indices"],
            [200, 201],
        )
        self.assertEqual(
            detailed["gates_interior_deposited"],
            [False, True, True, False, True, False, True, True],
        )
        self.assertEqual(
            detailed["projection_owned_slots_change_route"],
            [True, True, True],
        )
        self.assertFalse(
            detailed["manifest_semantic_labels_authoritative"]
        )

    @unittest.skipUnless(
        worker.PINNED_ROOT.is_dir(), "pinned Payne Zero checkout absent"
    )
    def test_two_fresh_processes_match_and_leave_canonical_data_unchanged(
        self,
    ) -> None:
        before = _data_snapshot()
        summaries = []
        cache_inventories = []
        for capture_name in ("a", "b"):
            with self.subTest(capture=capture_name):
                with tempfile.TemporaryDirectory() as temporary:
                    cache = Path(temporary) / f"cache-{capture_name}"
                    cache.mkdir()
                    completed = subprocess.run(
                        [sys.executable, str(worker.WORKER_PATH)],
                        cwd=worker.REPOSITORY_ROOT,
                        env=_fresh_environment(cache),
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=90,
                    )
                    cache_inventories.append(
                        sorted(
                            path.relative_to(cache).as_posix()
                            for path in cache.rglob("*")
                        )
                    )
                self.assertEqual(completed.stderr, "")
                summaries.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        after = _data_snapshot()
        self.assertEqual(before, after)
        self.assertEqual(summaries[0], summaries[1])
        self.assertTrue(cache_inventories[0])
        self.assertTrue(cache_inventories[1])
        self.assertEqual(cache_inventories[0], cache_inventories[1])
        summary = summaries[0]
        self.assertRegex(summary["capture_schema_digest"], r"^[0-9a-f]{64}$")
        self.assertRegex(summary["oracle_payload_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertRegex(summary["full_capture_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            summary["dense_pre_stimulated_sha256"],
            worker.EXPECTED_DENSE_PRE_SHA256,
        )
        self.assertEqual(
            summary["dense_post_stimulated_sha256"],
            worker.EXPECTED_DENSE_POST_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
