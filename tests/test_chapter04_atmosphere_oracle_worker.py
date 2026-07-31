"""Focused structural tests for the unpublished Chapter 4 atmosphere oracle."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import chapter04_atmosphere_oracle_worker as worker


class Chapter04AtmosphereOracleWorkerTests(unittest.TestCase):
    """Keep identity and the publication-deferred in-memory capture fail-closed."""

    def test_fixture_is_exact_input_only_contract(self) -> None:
        arrays = worker.load_input_fixture()
        self.assertEqual(set(arrays), worker.FIXTURE_KEYS)
        self.assertEqual(arrays["elemental_abundances"].shape, (99,))
        self.assertEqual(arrays["temperature"].shape, (6,))
        self.assertTrue(np.all(np.diff(arrays["column_mass"]) > 0.0))
        self.assertFalse(any("output" in name or "golden" in name for name in arrays))

    def test_wrong_pinned_root_is_rejected_before_import(self) -> None:
        with tempfile.TemporaryDirectory(prefix="chapter04-wrong-pin-") as directory:
            with self.assertRaises(worker.OracleIdentityError):
                worker.validate_pinned_root(Path(directory))

    def test_one_thread_environment_fails_closed(self) -> None:
        environment = {
            name: value for name, value in worker.ONE_THREAD_ENVIRONMENT.items()
        }
        environment.pop("NUMBA_NUM_THREADS")
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(worker.OracleEnvironmentError):
                worker.validate_one_thread_environment()

    def test_complete_scope_request_fails_before_solving(self) -> None:
        with self.assertRaises(worker.IncompleteOracleScopeError) as context:
            worker.build_atmosphere_oracle_results(require_complete=True)
        for field in worker.DEFERRED_CAPTURE_FIELDS:
            self.assertIn(field, str(context.exception))

    def test_fingerprint_is_independent_of_mapping_order(self) -> None:
        first = {
            "a": np.asarray([1.0, 2.0], dtype=np.float64),
            "b": np.asarray([3], dtype=np.int64),
        }
        second = {"b": first["b"].copy(), "a": first["a"].copy()}
        self.assertEqual(
            worker.result_fingerprint(first),
            worker.result_fingerprint(second),
        )

    def test_fresh_subprocess_builds_and_self_checks_requested_slice(self) -> None:
        summaries = []
        for _ in range(2):
            with tempfile.TemporaryDirectory(prefix="chapter04-numba-") as cache:
                environment = os.environ.copy()
                environment.update(worker.ONE_THREAD_ENVIRONMENT)
                environment.update(
                    {
                        "NUMBA_CACHE_DIR": cache,
                        "PAYNE_ZERO_DATA_ROOT": str(
                            worker.PINNED_ROOT / "source_data_files"
                        ),
                        "PYTHONPATH": os.pathsep.join(
                            (str(worker.PINNED_ROOT), str(worker.REPOSITORY_ROOT))
                        ),
                    }
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(worker.__file__).resolve()),
                    ],
                    check=True,
                    cwd=worker.REPOSITORY_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
            summaries.append(json.loads(completed.stdout))
        summary = summaries[0]
        self.assertEqual(
            summary["fingerprint"],
            summaries[1]["fingerprint"],
        )
        self.assertEqual(summary["key_count"], summaries[1]["key_count"])
        self.assertEqual(summary["key_count"], 460)
        self.assertEqual(
            summary["fingerprint"],
            "a6116c5f73c7ed3b0ee51907c419a307de2e1477be09f0b9969fab888c0b7682",
        )
        self.assertEqual(summary["full_solve_call_count"], 1)
        self.assertEqual(summary["full_molecule_shape"], [6, 170])
        self.assertEqual(summary["full_equation_shape"], [6, 23])
        self.assertEqual(summary["temperature_control_shape"], [3, 7])
        self.assertEqual(summary["pressure_control_shape"], [3, 7])
        self.assertEqual(summary["boundary_shape"], [4, 170])
        self.assertEqual(summary["bridge_live_shape_error_type"], "ValueError")
        self.assertEqual(summary["disabled_solve_call_count"], 0)
        self.assertEqual(summary["doppler_structural_infinity_count"], 12)
        self.assertEqual(summary["energy_solve_call_count"], 1)
        self.assertEqual(len(summary["energy_iteration_count"]), 6)
        self.assertEqual(len(summary["full_iteration_count"]), 6)
        self.assertEqual(summary["full_newton_history_count"], 38)
        self.assertEqual(summary["full_np_linalg_solve_call_count"], 38)
        self.assertEqual(summary["full_np_linalg_lstsq_call_count"], 0)
        self.assertEqual(summary["handoff_solve_call_count"], 1)
        self.assertEqual(len(summary["handoff_iteration_count"]), 6)
        self.assertEqual(summary["mode2_solve_call_count"], 1)
        self.assertEqual(summary["mode2_fill_call_count"], 0)
        self.assertEqual(summary["mode12_solve_call_count"], 1)
        self.assertEqual(summary["mode12_fill_call_count"], 0)
        self.assertEqual(summary["numba_thread_count"], 1)
        self.assertTrue(summary["wrappers_restored"])
        self.assertTrue(summary["numpy_linalg_wrappers_restored"])
        self.assertTrue(summary["population_schedule_wrappers_restored"])
        self.assertEqual(
            summary["priming_cache_molecular_solve_count_during_zero_code"],
            1,
        )
        self.assertEqual(
            summary["priming_cache_additional_solve_count_during_schedule"],
            0,
        )
        self.assertEqual(summary["schedule_inventory_job_count"], 230)
        self.assertEqual(summary["schedule_inventory_atomic_job_count"], 198)
        self.assertEqual(summary["schedule_inventory_molecular_job_count"], 32)
        self.assertEqual(
            summary["schedule_inventory_molecular_unique_code"],
            [
                101.0,
                106.0,
                107.0,
                108.0,
                112.0,
                114.0,
                120.0,
                124.0,
                126.0,
                606.0,
                607.0,
                608.0,
                814.0,
                822.0,
                823.0,
                10108.0,
            ],
        )
        self.assertEqual(
            summary[
                "schedule_inventory_"
                "molecular_unique_packed_start_slot_one_based"
            ],
            [
                841,
                846,
                847,
                848,
                851,
                853,
                858,
                862,
                864,
                868,
                869,
                870,
                889,
                895,
                896,
                940,
            ],
        )
        self.assertEqual(
            summary["boundary_temperature_uint64_bits"],
            [
                4666723172467343360,
                4666723172467343361,
                4671226772094713856,
                4671226772094713857,
            ],
        )
        self.assertEqual(
            summary["boundary_polynomial_active_branch_mask"],
            [True, False, False, False],
        )
        self.assertEqual(
            summary["boundary_h2_catalog_active_branch_mask"],
            [True, True, True, False],
        )
        self.assertEqual(
            summary["h2_probe_partition_interpolation_branch_mask"],
            [False, True, True, False, False, False],
        )
        self.assertEqual(summary["executed_atmosphere_source_count"], 35)
        self.assertFalse(summary["capture_scope_complete"])
        self.assertTrue(summary["capture_in_memory_scope_complete"])
        self.assertEqual(
            set(summary["deferred_fields"]),
            set(worker.DEFERRED_CAPTURE_FIELDS),
        )
        self.assertRegex(summary["fingerprint"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
