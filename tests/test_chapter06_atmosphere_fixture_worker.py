"""Focused contracts for the in-memory Chapter 6 atmosphere fixture worker."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import chapter06_atmosphere_fixture_worker as worker


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = REPOSITORY_ROOT / "scripts/chapter06_atmosphere_fixture_worker.py"
CONVERTER_CANDIDATE_PATH = (
    REPOSITORY_ROOT / "design/chapter06_atmosphere_converter_candidate.md"
)


def file_sha256(path: Path) -> str:
    """Return one local file's SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_staged_import_source_closure(entry_source: Path) -> set[str]:
    """Derive staged submodules and every implicitly executed initializer."""

    package_name = "payne_zero_atmosphere"
    discovered: set[str] = set()
    pending_sources = [entry_source]
    scanned_sources: set[Path] = set()
    while pending_sources:
        source_path = pending_sources.pop()
        if source_path in scanned_sources:
            continue
        scanned_sources.add(source_path)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    raise AssertionError(
                        f"relative staged import needs explicit discovery: {source_path}"
                    )
                if node.module is not None:
                    imported_modules.add(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
        for module_name in imported_modules:
            if module_name == package_name:
                relative_sources = [f"{package_name}/__init__.py"]
            elif module_name.startswith(package_name + "."):
                parts = module_name.split(".")
                relative_sources = [
                    "/".join(parts[:package_depth]) + "/__init__.py"
                    for package_depth in range(1, len(parts))
                ]
                relative_sources.append("/".join(parts) + ".py")
            else:
                continue
            for relative_source in relative_sources:
                if relative_source in discovered:
                    continue
                discovered.add(relative_source)
                pending_sources.append(worker.STAGED_SOURCE_ROOT / relative_source)
    return discovered


def canonical_data_snapshot() -> dict[str, tuple[int, str]]:
    """Return all canonical data identities before a no-write worker run."""

    return {
        str(path.relative_to(REPOSITORY_ROOT)): (
            path.stat().st_size,
            file_sha256(path),
        )
        for path in sorted((REPOSITORY_ROOT / "data").rglob("*"))
        if path.is_file()
    }


def fresh_capture_summary() -> dict[str, object]:
    """Run one exact fresh-process capture and return its JSON summary."""

    with tempfile.TemporaryDirectory(
        prefix="chapter06-atmosphere-fixture-cache-",
        dir="/private/tmp",
    ) as cache_directory:
        environment = dict(os.environ)
        environment.update(worker.CAPTURE_ENVIRONMENT)
        environment["NUMBA_CACHE_DIR"] = cache_directory
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
        environment.pop("PAYNE_ZERO_DATA_ROOT", None)
        completed = subprocess.run(
            [sys.executable, str(WORKER_PATH)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    if completed.stderr:
        raise AssertionError(
            f"fixture worker emitted unexpected stderr: {completed.stderr}"
        )
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise AssertionError(f"fixture worker emitted {len(lines)} stdout lines")
    return json.loads(lines[0])


def launch_with_thread_control_mutation(
    control_name: str,
    changed_value: str,
) -> subprocess.CompletedProcess[str]:
    """Launch the executable with one changed inherited thread control."""

    with tempfile.TemporaryDirectory(
        prefix="chapter06-atmosphere-mutated-control-cache-",
        dir="/private/tmp",
    ) as cache_directory:
        environment = dict(os.environ)
        environment.update(worker.CAPTURE_ENVIRONMENT)
        environment[control_name] = changed_value
        environment["NUMBA_CACHE_DIR"] = cache_directory
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
        environment["PYTHONPROFILEIMPORTTIME"] = "1"
        environment.pop("PAYNE_ZERO_DATA_ROOT", None)
        return subprocess.run(
            [sys.executable, str(WORKER_PATH)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )


class Chapter06AtmosphereFixtureWorkerTests(unittest.TestCase):
    """Freeze safe-sequence steps 4--6 without publishing any artifact."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_before = canonical_data_snapshot()
        cls.first = fresh_capture_summary()
        cls.second = fresh_capture_summary()
        cls.data_after = canonical_data_snapshot()

    def test_converter_candidate_record_tracks_the_accepted_repair(self) -> None:
        text = CONVERTER_CANDIDATE_PATH.read_text()
        self.assertIn("independently accepted", text)
        self.assertIn(worker.CONVERTER_SHA256, text)
        self.assertIn(worker.CONVERTER_TEST_SHA256, text)
        self.assertIn(worker.CONVERTER_AUDIT_SHA256, text)
        self.assertNotIn(
            "12c172ef206ded0d9c42e03abfdc5d22cd93b3bc07387953047953dbf93cb4eb",
            text,
        )
        self.assertNotIn(
            "82543e88ebbbc7609d41803cf786d95245c7ada63565e6a2dd660eec4fc4fa11",
            text,
        )

    def test_preimport_authority_covers_commit_source_data_and_converter(
        self,
    ) -> None:
        identities = worker.verify_preimport_identities()
        self.assertEqual(identities["payne_zero_commit"], worker.PINNED_COMMIT)
        self.assertEqual(
            identities["accepted_converter__sha256"],
            worker.CONVERTER_SHA256,
        )
        self.assertEqual(
            identities["accepted_converter_tests__sha256"],
            worker.CONVERTER_TEST_SHA256,
        )
        self.assertEqual(
            identities["accepted_converter_audit__sha256"],
            worker.CONVERTER_AUDIT_SHA256,
        )
        source_names = [name for name in identities if name.startswith("source__")]
        data_names = [name for name in identities if name.startswith("data__")]
        self.assertEqual(
            len(source_names),
            len(worker.FROZEN_PINNED_PYTHON_MANIFEST),
        )
        self.assertEqual(len(data_names), len(worker.DATA_IDENTITIES))
        self.assertEqual(
            self.first["staged_converter_dependency_hashes"],
            worker.STAGED_CONVERTER_DEPENDENCIES,
        )
        self.assertEqual(
            self.first["pinned_converter_dependency_hashes"],
            worker.PINNED_CONVERTER_DEPENDENCY_HASHES,
        )
        self.assertEqual(
            self.first["converter_child_loaded_source_hashes"],
            worker.STAGED_CONVERTER_DEPENDENCIES,
        )
        self.assertEqual(
            worker.STAGED_CONVERTER_BYTE_IDENTICAL_TO_PINNED,
            frozenset(worker.STAGED_CONVERTER_DEPENDENCIES),
        )
        for (
            relative_path,
            expected_hash,
        ) in worker.STAGED_CONVERTER_DEPENDENCIES.items():
            identity_name = f"staged_converter_source__{relative_path}__sha256"
            self.assertEqual(identities[identity_name], expected_hash)
            self.assertEqual(
                expected_hash,
                worker.PINNED_CONVERTER_DEPENDENCY_HASHES[relative_path],
            )

    def test_accepted_converter_child_reproduces_the_frozen_record(self) -> None:
        converted = worker.run_accepted_converter()
        self.assertEqual(
            converted["conversion_version"],
            worker.CONVERSION_VERSION,
        )
        self.assertEqual(
            converted["observed_row_role"],
            "non_authoritative_packing_corroboration",
        )
        self.assertEqual(converted["observed_row_index"], 780_108)
        self.assertEqual(
            {name: converted["fields"][name]["value"] for name in converted["fields"]},
            worker.EXPECTED_CONVERTED_VALUES,
        )
        self.assertEqual(
            converted["words"],
            [[12_425_352, 1_370_295_734, 1_628_847_268, 581_578_398]],
        )
        self.assertEqual(
            converted["loaded_staged_source_sha256"],
            worker.STAGED_CONVERTER_DEPENDENCIES,
        )

    def test_executable_rejects_each_changed_thread_control_before_import(
        self,
    ) -> None:
        source = WORKER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("os.environ.update", source)
        self.assertIn(
            "    _require_bootstrap_environment()\n\nimport numpy",
            source,
        )
        mutations = {
            "MKL_DYNAMIC": "TRUE",
            "MKL_NUM_THREADS": "7",
            "NUMBA_NUM_THREADS": "3",
            "NUMEXPR_NUM_THREADS": "7",
            "OMP_NUM_THREADS": "7",
            "OPENBLAS_NUM_THREADS": "7",
            "VECLIB_MAXIMUM_THREADS": "7",
        }
        self.assertEqual(set(mutations), set(worker.THREAD_ENVIRONMENT))
        for control_name, changed_value in mutations.items():
            with self.subTest(control_name=control_name):
                completed = launch_with_thread_control_mutation(
                    control_name,
                    changed_value,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(completed.stdout, "")
                self.assertIn(control_name, completed.stderr)
                self.assertIn("import time:", completed.stderr)
                self.assertNotRegex(
                    completed.stderr,
                    r"(?m)^import time:.*\|\s*payne_zero_atmosphere(?:\.|$)",
                )

    def test_environment_requires_every_control_and_a_fresh_external_cache(
        self,
    ) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                worker.FixtureEnvironmentError,
                "controls are missing",
            ):
                worker.require_environment()

        with tempfile.TemporaryDirectory(
            prefix="chapter06-atmosphere-environment-",
            dir="/private/tmp",
        ) as cache_directory:
            environment = {
                **worker.CAPTURE_ENVIRONMENT,
                "NUMBA_CACHE_DIR": cache_directory,
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                result = worker.require_environment()
                self.assertTrue(bool(result["environment__cpu_only"]))
                for control_name in sorted(
                    set(worker.CAPTURE_ENVIRONMENT) - set(worker.THREAD_ENVIRONMENT)
                ):
                    changed = dict(environment)
                    changed[control_name] = "deliberately-changed"
                    with (
                        self.subTest(control_name=control_name),
                        mock.patch.dict(os.environ, changed, clear=True),
                        self.assertRaisesRegex(
                            worker.FixtureEnvironmentError,
                            control_name,
                        ),
                    ):
                        worker.require_environment()
                Path(cache_directory, "not-empty").write_text("occupied")
                with self.assertRaisesRegex(
                    worker.FixtureEnvironmentError,
                    "truly empty",
                ):
                    worker.require_environment()

    def test_output_preserving_staged_dependency_drift_fails_before_child_import(
        self,
    ) -> None:
        original_root = worker.STAGED_SOURCE_ROOT
        for relative_path in worker.STAGED_CONVERTER_DEPENDENCIES:
            with self.subTest(relative_path=relative_path):
                with tempfile.TemporaryDirectory(
                    prefix="chapter06-mutated-staged-dependency-",
                    dir="/private/tmp",
                ) as temporary_directory:
                    staged_root = Path(temporary_directory) / "src"
                    for staged_relative_path in worker.STAGED_CONVERTER_DEPENDENCIES:
                        source = original_root / staged_relative_path
                        target = staged_root / staged_relative_path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(source.read_bytes())
                    mutated_path = staged_root / relative_path
                    original_text = mutated_path.read_text(encoding="utf-8")
                    mutated_text = (
                        original_text + "\n# output-preserving adversarial byte drift\n"
                    )
                    self.assertEqual(
                        ast.dump(ast.parse(original_text), include_attributes=False),
                        ast.dump(ast.parse(mutated_text), include_attributes=False),
                    )
                    mutated_path.write_text(mutated_text, encoding="utf-8")
                    with (
                        mock.patch.object(
                            worker,
                            "STAGED_SOURCE_ROOT",
                            staged_root,
                        ),
                        mock.patch.object(worker.subprocess, "run") as child_run,
                        self.assertRaisesRegex(
                            worker.FixtureIdentityError,
                            relative_path,
                        ),
                    ):
                        worker.run_accepted_converter()
                    child_run.assert_not_called()

        for symlinked_relative_path in worker.STAGED_CONVERTER_DEPENDENCIES:
            with self.subTest(symlinked_relative_path=symlinked_relative_path):
                with tempfile.TemporaryDirectory(
                    prefix="chapter06-symlinked-staged-dependency-",
                    dir="/private/tmp",
                ) as temporary_directory:
                    staged_root = Path(temporary_directory) / "src"
                    for relative_path in worker.STAGED_CONVERTER_DEPENDENCIES:
                        target = staged_root / relative_path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if relative_path == symlinked_relative_path:
                            target.symlink_to(original_root / relative_path)
                        else:
                            target.write_bytes(
                                (original_root / relative_path).read_bytes()
                            )
                    with (
                        mock.patch.object(
                            worker,
                            "STAGED_SOURCE_ROOT",
                            staged_root,
                        ),
                        mock.patch.object(worker.subprocess, "run") as child_run,
                        self.assertRaisesRegex(
                            worker.FixtureIdentityError,
                            "must not be a symlink",
                        ),
                    ):
                        worker.run_accepted_converter()
                    child_run.assert_not_called()

    def test_fixture_schema_is_exactly_the_proposed_nineteen_members(self) -> None:
        self.assertEqual(len(worker.FIXTURE_SCHEMA), 19)
        self.assertEqual(set(self.first["fixture_schema"]), set(worker.FIXTURE_SCHEMA))
        self.assertEqual(self.first["fixture_member_count"], 19)
        for name, (shape, dtype) in worker.FIXTURE_SCHEMA.items():
            with self.subTest(name=name):
                metadata = self.first["fixture_schema"][name]
                self.assertEqual(metadata["shape"], list(shape))
                self.assertEqual(metadata["dtype"], dtype.str)
                self.assertEqual(
                    metadata["sha256"],
                    worker.EXPECTED_FIXTURE_MEMBER_HASHES[name],
                )
        forbidden = {
            "line_mass_absorption_coefficient",
            "rosseland_opacity",
            "radiative_acceleration",
            "convective_flux",
            "convective_velocity",
        }
        self.assertTrue(forbidden.isdisjoint(self.first["fixture_schema"]))

    def test_projection_and_placeholder_nonownership_are_bitwise(self) -> None:
        for summary in (self.first, self.second):
            self.assertTrue(summary["full_projected_bitwise_equal"])
            self.assertTrue(summary["placeholder_fixture_bitwise_equal"])
            self.assertTrue(summary["placeholder_line_bitwise_equal"])
            self.assertEqual(
                summary["full_array_hashes"],
                worker.EXPECTED_FULL_ARRAY_HASHES,
            )
            self.assertEqual(
                summary["projected_array_hashes"],
                worker.EXPECTED_PROJECTED_ARRAY_HASHES,
            )
            self.assertEqual(
                summary["line_output_sha256"],
                worker.EXPECTED_LINE_OUTPUT_SHA256,
            )

    def test_dynamic_read_set_is_complete_declared_and_excludes_two_witnesses(
        self,
    ) -> None:
        self.assertEqual(
            self.first["dynamic_read_set"],
            list(worker.EXPECTED_DYNAMIC_READ_SET),
        )
        self.assertEqual(
            self.second["dynamic_read_set"],
            list(worker.EXPECTED_DYNAMIC_READ_SET),
        )
        joined = "\n".join(self.first["dynamic_read_set"])
        self.assertNotIn("continuum_level_tables.npz", joined)
        self.assertNotIn("observed_atomic_lines.npy", joined)
        self.assertEqual(self.first["loaded_python_source_count"], 35)
        self.assertEqual(self.second["loaded_python_source_count"], 35)

    def test_two_fresh_processes_reproduce_all_three_capture_identities(
        self,
    ) -> None:
        for name in (
            "fixture_schema_digest",
            "payload_fingerprint",
            "full_capture_schema_digest",
            "full_capture_fingerprint",
        ):
            with self.subTest(name=name):
                self.assertEqual(self.first[name], self.second[name])
                self.assertRegex(self.first[name], r"^[0-9a-f]{64}$")
        self.assertEqual(self.first["fixture_schema"], self.second["fixture_schema"])
        self.assertEqual(
            self.first["evidence_member_count"],
            self.second["evidence_member_count"],
        )

    def test_worker_performs_no_canonical_or_publication_writes(self) -> None:
        self.assertEqual(self.data_before, self.data_after)
        for summary in (self.first, self.second):
            self.assertFalse(summary["capture_scope_complete"])
            self.assertFalse(summary["fixture_publication_performed"])
            self.assertFalse(summary["golden_read_performed"])
            self.assertFalse(summary["golden_write_performed"])
            self.assertFalse(summary["manifest_write_performed"])

        source = WORKER_PATH.read_text()
        tree = ast.parse(source)
        forbidden_calls = {
            "save",
            "savez",
            "savez_compressed",
            "tofile",
            "write_bytes",
            "write_text",
        }
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(forbidden_calls.isdisjoint(called_attributes))
        self.assertNotIn("data/fixtures", source)
        self.assertNotIn("data/golden", source)
        self.assertNotIn("data/MANIFEST.json", source)
        self.assertNotIn("--output", source)

    def test_mapping_helpers_reject_object_dtype_and_are_order_independent(
        self,
    ) -> None:
        first = {
            "b": np.asarray([2], dtype=np.int16),
            "a": np.asarray([1.0], dtype=np.float64),
        }
        second = {"a": first["a"], "b": first["b"]}
        self.assertEqual(
            worker.mapping_schema_digest(first),
            worker.mapping_schema_digest(second),
        )
        self.assertEqual(
            worker.mapping_fingerprint(first),
            worker.mapping_fingerprint(second),
        )
        with self.assertRaisesRegex(TypeError, "object dtype"):
            worker.mapping_fingerprint({"bad": np.asarray([object()], dtype=object)})


if __name__ == "__main__":
    unittest.main()
