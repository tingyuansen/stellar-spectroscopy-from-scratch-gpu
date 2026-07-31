"""Focused contracts for the in-memory Chapter 6 synthesis oracle worker."""

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

from scripts import chapter06_synthesis_oracle_worker as worker


class Chapter06SynthesisOracleWorkerTests(unittest.TestCase):
    """Pin input roles, forbidden paths, and the fresh-process science result."""

    def test_fixture_and_subset_are_canonical_input_only_artifacts(self) -> None:
        fixture = worker.load_fixture()
        subset = worker.load_subset()
        self.assertEqual(len(fixture), worker.FIXTURE_KEY_COUNT)
        self.assertEqual(
            worker.array_mapping_digest(fixture),
            worker.FIXTURE_PAYLOAD_DIGEST,
        )
        self.assertEqual(len(subset), worker.SUBSET_KEY_COUNT)
        self.assertEqual(
            set(worker.RAW_FIELDS) | set(worker.SUBSET_PROVENANCE_FIELDS), set(subset)
        )
        self.assertEqual(int(subset["source_row_index"]), worker.SOURCE_ROW_INDEX)
        self.assertIn("no computed outputs", str(subset["subset_role"]))
        for name in worker.RAW_FIELDS:
            self.assertEqual(subset[name].shape, (1,))
            self.assertEqual(subset[name].dtype, worker.RAW_FIELD_DTYPES[name])

    def test_golden_paths_are_rejected_before_any_read(self) -> None:
        forbidden_fixture = (
            worker.GOLDEN_ROOT
            / "payne_zero/chapter05/chapter05_continuum_reader_cpu_float64.npz"
        )
        forbidden_subset = (
            worker.GOLDEN_ROOT / "payne_zero/chapter06/synthesis/not_yet_published.npz"
        )
        with mock.patch.object(
            worker.np,
            "load",
            side_effect=AssertionError("np.load must not be reached"),
        ):
            with self.assertRaisesRegex(worker.OracleIdentityError, "golden artifact"):
                worker.load_fixture(forbidden_fixture)
            with self.assertRaisesRegex(worker.OracleIdentityError, "golden artifact"):
                worker.load_subset(forbidden_subset)

    def test_alternate_and_symlink_alias_input_paths_fail_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            fixture_alias = temporary_root / "fixture-alias.npz"
            subset_alias = temporary_root / "subset-alias.npz"
            fixture_alias.symlink_to(worker.FIXTURE_PATH)
            subset_alias.symlink_to(worker.SUBSET_PATH)
            alternate = temporary_root / "ordinary-alternate.npz"
            with mock.patch.object(
                worker.np,
                "load",
                side_effect=AssertionError("np.load must not be reached"),
            ):
                for loader, path in (
                    (worker.load_fixture, fixture_alias),
                    (worker.load_subset, subset_alias),
                ):
                    with self.subTest(path=path.name):
                        with self.assertRaisesRegex(
                            worker.OracleIdentityError, "symlink"
                        ):
                            loader(path)
                for loader in (worker.load_fixture, worker.load_subset):
                    with self.subTest(loader=loader.__name__):
                        with self.assertRaisesRegex(
                            worker.OracleIdentityError, "canonical path"
                        ):
                            loader(alternate)

    def test_worker_has_no_output_or_publication_interface(self) -> None:
        build_parameters = inspect.signature(worker.build_oracle_results).parameters
        forbidden_fragments = (
            "output",
            "destination",
            "publish",
            "golden",
            "serialize",
        )
        self.assertFalse(
            any(
                fragment in name
                for name in build_parameters
                for fragment in forbidden_fragments
            )
        )
        with mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                worker.parse_args(["--output", "/tmp/forbidden.npz"])
            with self.assertRaises(SystemExit):
                worker.parse_args(["--publish"])

    def test_deterministic_result_sorts_copies_and_rejects_objects(self) -> None:
        source = np.asarray([1.0, 2.0], dtype=np.float64)
        result = worker.deterministic_result({"z": np.asarray("value"), "a": source})
        self.assertEqual(list(result), ["a", "z"])
        self.assertEqual(result["z"].shape, ())
        source[0] = -1.0
        np.testing.assert_array_equal(
            result["a"], np.asarray([1.0, 2.0], dtype=np.float64)
        )
        with self.assertRaisesRegex(TypeError, "object dtype"):
            worker.deterministic_result({"bad": np.asarray([object()], dtype=object)})

    def test_exact_mapping_and_invariant_field_contracts_are_declared(self) -> None:
        self.assertEqual(len(worker.CATALOG_LINE_FIELDS), 13)
        self.assertEqual(len(worker.CATALOG_SUPPORT_FIELDS), 5)
        self.assertEqual(
            set(worker.CATALOG_SUPPORT_FIELDS),
            {
                "helium_line_type",
                "helium_line_center_cutoff_ratio",
                "harris_profile_h0_table",
                "harris_profile_h1_table",
                "harris_profile_h2_table",
            },
        )
        self.assertEqual(len(worker.PIPELINE_CONTINUUM_FIELDS), 18)
        self.assertEqual(len(worker.AUTO_INVARIANT_FIELDS), 11)
        self.assertEqual(len(worker.HELIUM_ARRAY_INVARIANT_FIELDS), 10)
        self.assertEqual(worker.GRID_SPECS["canonical"]["count"], 6000)
        self.assertEqual(worker.GRID_SPECS["coarse"]["count"], 400)

    def test_populated_and_symlink_cache_paths_fail_closed(self) -> None:
        environment = dict(worker.ORACLE_ENVIRONMENT)
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "cache"
            cache.mkdir()
            (cache / "sentinel").write_text("occupied", encoding="utf-8")
            environment["NUMBA_CACHE_DIR"] = str(cache)
            with mock.patch.dict(os.environ, environment, clear=False):
                with self.assertRaisesRegex(
                    worker.OracleEnvironmentError, "truly empty"
                ):
                    worker.require_environment()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            cache = root / "cache-link"
            cache.symlink_to(target, target_is_directory=True)
            environment["NUMBA_CACHE_DIR"] = str(cache)
            with mock.patch.dict(os.environ, environment, clear=False):
                with self.assertRaisesRegex(worker.OracleEnvironmentError, "symlink"):
                    worker.require_environment()

    @unittest.skipUnless(
        worker.PINNED_ROOT.is_dir(), "pinned Payne Zero checkout absent"
    )
    def test_pinned_sources_and_static_inputs_are_exact(self) -> None:
        identities = worker.verify_identity()
        self.assertEqual(identities["payne_zero_commit"], worker.PINNED_COMMIT)
        self.assertEqual(
            identities["source_archive_sha256"],
            worker.SOURCE_ARCHIVE_SHA256,
        )
        for filename in worker.STAGED_EXECUTED_SOURCE_FILES:
            self.assertIn(
                f"staged_source__{filename}__sha256",
                identities,
            )
        for name in worker.STATIC_TABLE_IDENTITIES:
            self.assertIn(f"table__{name}__sha256", identities)

    @unittest.skipUnless(
        worker.PINNED_ROOT.is_dir(), "pinned Payne Zero checkout absent"
    )
    def test_two_fresh_processes_have_identical_complete_summaries(self) -> None:
        summaries = []
        for capture_name in ("a", "b"):
            with self.subTest(capture=capture_name):
                with tempfile.TemporaryDirectory() as cache:
                    environment = os.environ.copy()
                    environment.update(worker.ORACLE_ENVIRONMENT)
                    environment["NUMBA_CACHE_DIR"] = cache
                    environment["PAYNE_ZERO_DATA_ROOT"] = str(worker.PINNED_DATA_ROOT)
                    completed = subprocess.run(
                        [sys.executable, str(worker.WORKER_PATH)],
                        cwd=worker.REPOSITORY_ROOT,
                        env=environment,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                self.assertEqual(completed.stderr, "")
                summaries.append(json.loads(completed.stdout.strip().splitlines()[-1]))
        self.assertEqual(summaries[0], summaries[1])
        summary = summaries[0]
        self.assertTrue(summary["capture_scope_complete"])
        self.assertEqual(summary["key_count"], worker.ACCEPTED_CAPTURE_KEY_COUNT)
        self.assertEqual(
            summary["capture_schema_digest"],
            worker.ACCEPTED_CAPTURE_SCHEMA_DIGEST,
        )
        self.assertEqual(summary["canonical_grid_count"], 6000)
        self.assertEqual(summary["coarse_grid_count"], 400)
        self.assertEqual(
            summary["canonical_activity_mask"],
            worker.EXPECTED_ACTIVITY_MASK.tolist(),
        )
        self.assertEqual(
            summary["coarse_activity_mask"],
            worker.EXPECTED_ACTIVITY_MASK.tolist(),
        )
        self.assertEqual(summary["canonical_reach_minimum"], 5)
        self.assertEqual(summary["canonical_reach_maximum"], 163)
        self.assertEqual(summary["maximum_loop_batched_absolute_difference"], 0.0)
        self.assertEqual(summary["full_stimulated_factor_member_count"], 8)
        self.assertGreater(summary["full_stimulated_factor_minimum"], 0.0)
        self.assertLessEqual(summary["full_stimulated_factor_maximum"], 1.0)
        self.assertTrue(
            summary["batched_net_reconstructed_from_full_stimulated_factor"]
        )
        self.assertTrue(summary["loop_net_reconstructed_from_full_stimulated_factor"])
        self.assertEqual(
            summary["loaded_pinned_python_source_count"],
            len(worker.FROZEN_PINNED_PYTHON_MANIFEST),
        )
        self.assertFalse(summary["golden_read_performed"])
        self.assertFalse(summary["golden_publication_performed"])
        self.assertRegex(summary["capture_schema_digest"], r"^[0-9a-f]{64}$")
        self.assertRegex(summary["physical_payload_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertRegex(summary["full_capture_fingerprint"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
