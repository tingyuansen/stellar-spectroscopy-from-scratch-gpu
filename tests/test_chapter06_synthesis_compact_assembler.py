"""Contracts for the pure in-memory Chapter 6 synthesis compact assembler."""

from __future__ import annotations

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

from scripts import chapter06_synthesis_compact_assembler as assembler
from scripts import chapter06_synthesis_oracle_worker as oracle_worker


def _fresh_child(code: str) -> subprocess.CompletedProcess[str]:
    """Run one oracle/assembler probe in a fresh isolated process."""

    with tempfile.TemporaryDirectory() as cache:
        environment = os.environ.copy()
        environment.update(oracle_worker.ORACLE_ENVIRONMENT)
        environment["NUMBA_CACHE_DIR"] = cache
        environment["PAYNE_ZERO_DATA_ROOT"] = str(oracle_worker.PINNED_DATA_ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=oracle_worker.REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if any(Path(cache).iterdir()):
            raise AssertionError("fresh compact-assembly cache was not empty")
    return completed


class Chapter06SynthesisCompactAssemblerTests(unittest.TestCase):
    """Pin the in-memory boundary, complete ownership, and reduction proofs."""

    def test_module_exposes_no_output_serialization_or_publication_api(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(assembler.assemble_compact_candidate).parameters),
            ("raw",),
        )
        self.assertFalse(hasattr(assembler, "main"))
        self.assertFalse(hasattr(assembler, "parse_args"))
        forbidden_fragments = {
            "output",
            "destination",
            "serialize",
            "publish",
            "manifest",
            "authorization",
        }
        public_callables = {
            name
            for name in assembler.__all__
            if callable(getattr(assembler, name, None))
        }
        self.assertFalse(
            {
                name
                for name in public_callables
                if any(fragment in name for fragment in forbidden_fragments)
            }
        )

    def test_unaccepted_mapping_fails_without_running_the_external_oracle(self) -> None:
        with self.assertRaisesRegex(
            assembler.CompactAssemblyError,
            "raw capture has 0 keys",
        ):
            assembler.assemble_compact_candidate({})
        with self.assertRaisesRegex(
            assembler.CompactAssemblyError,
            "object dtype",
        ):
            assembler.schema_digest({"bad": np.asarray([object()], dtype=object)})

    def test_plan_rebind_authority_is_exact_and_fails_closed(self) -> None:
        assembler._verify_rebind_authority()
        self.assertEqual(
            assembler.sha256(assembler.PLAN_PATH),
            assembler.ACCEPTED_PLAN_SHA256,
        )
        self.assertEqual(
            assembler.sha256(assembler.PLAN_REBIND_CANDIDATE_PATH),
            assembler.ACCEPTED_PLAN_REBIND_CANDIDATE_SHA256,
        )
        self.assertEqual(
            assembler.sha256(assembler.PLAN_REBIND_AUDIT_PATH),
            assembler.ACCEPTED_PLAN_REBIND_AUDIT_SHA256,
        )
        with mock.patch.object(
            assembler,
            "ACCEPTED_PLAN_REBIND_AUDIT_SHA256",
            "0" * 64,
        ):
            with self.assertRaisesRegex(
                assembler.CompactAssemblyError,
                "plan-rebind independent audit identity changed",
            ):
                assembler._verify_rebind_authority()

    @unittest.skipUnless(
        oracle_worker.PINNED_ROOT.is_dir(), "pinned Payne Zero checkout absent"
    )
    def test_two_fresh_processes_assemble_identical_compact_summaries(self) -> None:
        code = """
import json
from scripts import chapter06_synthesis_compact_assembler as compact
from scripts import chapter06_synthesis_oracle_worker as worker
raw = worker.build_oracle_results()
assembly = compact.assemble_compact_candidate(raw)
print(json.dumps(compact.summarize_compact_assembly(assembly), sort_keys=True))
"""
        summaries = []
        for capture_name in ("a", "b"):
            with self.subTest(capture=capture_name):
                completed = _fresh_child(code)
                self.assertEqual(completed.stderr, "")
                summaries.append(json.loads(completed.stdout.strip()))
        self.assertEqual(summaries[0], summaries[1])
        summary = summaries[0]
        self.assertEqual(summary["candidate_key_count"], 213)
        self.assertLess(
            summary["candidate_array_bytes"],
            assembler.COMPACT_SIZE_CEILING_BYTES,
        )
        self.assertEqual(summary["gross_shape"], [4, 6, 6000])
        self.assertEqual(summary["net_shape"], [4, 6, 6000])
        self.assertEqual(sum(summary["raw_ownership_counts"].values()), 754)
        self.assertTrue(summary["raw_ownership_complete"])
        self.assertFalse(summary["publication_authorized"])
        self.assertFalse(summary["golden_publication_performed"])
        self.assertRegex(summary["candidate_schema_digest"], r"^[0-9a-f]{64}$")
        self.assertRegex(summary["candidate_payload_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertRegex(summary["raw_ownership_digest"], r"^[0-9a-f]{64}$")

    @unittest.skipUnless(
        oracle_worker.PINNED_ROOT.is_dir(), "pinned Payne Zero checkout absent"
    )
    def test_fresh_process_checks_ownership_reconstruction_and_mutations(self) -> None:
        code = """
import json
from dataclasses import replace
import numpy as np
from scripts import chapter06_synthesis_compact_assembler as compact
from scripts import chapter06_synthesis_oracle_worker as worker

raw = worker.build_oracle_results()
assembly = compact.assemble_compact_candidate(raw)
compact.validate_compact_candidate(assembly)
second = compact.assemble_compact_candidate(raw)
assert tuple(assembly.arrays) == tuple(second.arrays)
assert assembly.schema == second.schema
assert assembly.raw_ownership == second.raw_ownership
for name in assembly.arrays:
    assert np.array_equal(assembly.arrays[name], second.arrays[name])

failures = {}
def expect_failure(label, callback):
    try:
        callback()
    except compact.CompactAssemblyError as error:
        failures[label] = str(error)
    else:
        raise AssertionError(f"{label} unexpectedly passed")

removed = dict(raw)
removed.pop("grid__canonical__wavelength_nm")
expect_failure("removed_raw_member", lambda: compact.assemble_compact_candidate(removed))

extra = dict(raw)
extra["unexpected"] = np.asarray(1, dtype=np.int64)
expect_failure("extra_raw_member", lambda: compact.assemble_compact_candidate(extra))

wrong_scope = dict(raw)
wrong_scope["meta__capture_scope_complete"] = np.asarray(False, dtype=np.bool_)
expect_failure("unaccepted_scope", lambda: compact.assemble_compact_candidate(wrong_scope))

old_raw_boundary = {name: np.asarray(value).copy() for name, value in raw.items()}
old_raw_boundary["meta__design_sha256"] = np.asarray(
    "d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565"
)
old_raw_boundary["meta__full_capture_fingerprint"] = np.asarray(
    "33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b"
)
expect_failure(
    "historical_raw_boundary",
    lambda: compact.assemble_compact_candidate(old_raw_boundary),
)

wrong_full_fingerprint = {
    name: np.asarray(value).copy() for name, value in raw.items()
}
wrong_full_fingerprint["meta__full_capture_fingerprint"] = np.asarray("0" * 64)
expect_failure(
    "wrong_live_full_fingerprint",
    lambda: compact.assemble_compact_candidate(wrong_full_fingerprint),
)

wrong_live_design = {name: np.asarray(value).copy() for name, value in raw.items()}
wrong_live_design["meta__design_sha256"] = np.asarray(
    "d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565"
)
expect_failure(
    "wrong_live_design_sha256",
    lambda: compact.assemble_compact_candidate(wrong_live_design),
)

wrong_physical_fingerprint = {
    name: np.asarray(value).copy() for name, value in raw.items()
}
wrong_physical_fingerprint["meta__physical_payload_fingerprint"] = np.asarray(
    "0" * 64
)
expect_failure(
    "wrong_physical_fingerprint",
    lambda: compact.assemble_compact_candidate(wrong_physical_fingerprint),
)

changed_raw = {name: np.asarray(value).copy() for name, value in raw.items()}
changed_raw["solar_dwarf__canonical__gross_batched_float32"][0, 0] = np.nextafter(
    changed_raw["solar_dwarf__canonical__gross_batched_float32"][0, 0],
    np.float32(np.inf),
)
expect_failure("raw_fingerprint", lambda: compact.assemble_compact_candidate(changed_raw))

bad_factor = {name: np.asarray(value).copy() for name, value in raw.items()}
bad_factor[
    "solar_dwarf__canonical__ledger__stimulated_emission_factor_float32"
][0, 2434] = np.float32(0.5)
expect_failure(
    "factor_reconstruction",
    lambda: compact._validate_route_deduplication(
        bad_factor, "solar_dwarf", "canonical"
    ),
)

bad_loop = {name: np.asarray(value).copy() for name, value in raw.items()}
bad_loop["solar_dwarf__canonical__gross_loop_float32"][0, 0] = np.float32(1.0)
expect_failure(
    "loop_deduplication",
    lambda: compact._validate_route_deduplication(
        bad_loop, "solar_dwarf", "canonical"
    ),
)

bad_continuum = {name: np.asarray(value).copy() for name, value in raw.items()}
bad_continuum[
    "solar_dwarf__canonical__continuum_absorption_float64"
][0, 0] = np.nextafter(
    bad_continuum[
        "solar_dwarf__canonical__continuum_absorption_float64"
    ][0, 0],
    np.float64(np.inf),
)
expect_failure(
    "continuum_reconstruction",
    lambda: compact._validate_continuum(
        bad_continuum, "solar_dwarf", "canonical"
    ),
)

bad_harris = {name: np.asarray(value).copy() for name, value in raw.items()}
bad_harris["invariant__canonical__harris_profile_h0_table"][0] = np.nextafter(
    bad_harris["invariant__canonical__harris_profile_h0_table"][0],
    np.float64(np.inf),
)
expect_failure(
    "harris_deduplication",
    lambda: compact._add_invariants(
        bad_harris, compact._Builder(), {}
    ),
)

bad_candidate_arrays = {
    name: np.asarray(value).copy() for name, value in assembly.arrays.items()
}
bad_candidate_arrays["opacity__gross_float32"][0, 0, 0] = np.nextafter(
    bad_candidate_arrays["opacity__gross_float32"][0, 0, 0],
    np.float32(np.inf),
)
expect_failure(
    "compact_payload_mutation",
    lambda: compact.validate_compact_candidate(
        replace(assembly, arrays=bad_candidate_arrays)
    ),
)
expect_failure(
    "compact_schema_mutation",
    lambda: compact.validate_compact_candidate(
        replace(assembly, schema=assembly.schema[:-1])
    ),
)
expect_failure(
    "compact_ownership_mutation",
    lambda: compact.validate_compact_candidate(
        replace(assembly, raw_ownership=assembly.raw_ownership[:-1])
    ),
)

for forbidden_shape in ((6, 6, 139), (6, 400), (4, 6, 400), (2001,), (1001,)):
    assert not any(np.asarray(value).shape == forbidden_shape for value in assembly.arrays.values())
assert [
    name
    for name, value in assembly.arrays.items()
    if np.asarray(value).shape == (4, 6, 6000)
] == ["opacity__gross_float32", "opacity__net_float32"]
assert len(assembly.raw_ownership) == 754
assert set(item.disposition for item in assembly.raw_ownership) == compact.RAW_DISPOSITIONS
for item in assembly.raw_ownership:
    if item.disposition == "final":
        assert item.target.split("[", 1)[0] in assembly.arrays

gross_before = assembly.arrays["opacity__gross_float32"].copy()
raw["hot_dwarf__canonical__gross_batched_float32"][0, 0] = np.float32(99.0)
assert np.array_equal(gross_before, assembly.arrays["opacity__gross_float32"])

print(json.dumps({
    "failure_labels": sorted(failures),
    "ownership_count": len(assembly.raw_ownership),
    "ownership_digest": compact.ownership_digest(assembly.raw_ownership),
    "candidate_key_count": len(assembly.arrays),
    "candidate_bytes": sum(np.asarray(value).nbytes for value in assembly.arrays.values()),
    "schema_digest": str(assembly.arrays["meta__compact_schema_digest"]),
    "payload_fingerprint": str(assembly.arrays["meta__compact_payload_fingerprint"]),
}, sort_keys=True))
"""
        completed = _fresh_child(code)
        self.assertEqual(completed.stderr, "")
        report = json.loads(completed.stdout.strip())
        self.assertEqual(
            report["failure_labels"],
            [
                "compact_ownership_mutation",
                "compact_payload_mutation",
                "compact_schema_mutation",
                "continuum_reconstruction",
                "extra_raw_member",
                "factor_reconstruction",
                "harris_deduplication",
                "historical_raw_boundary",
                "loop_deduplication",
                "raw_fingerprint",
                "removed_raw_member",
                "unaccepted_scope",
                "wrong_live_design_sha256",
                "wrong_live_full_fingerprint",
                "wrong_physical_fingerprint",
            ],
        )
        self.assertEqual(report["ownership_count"], 754)
        self.assertEqual(report["candidate_key_count"], 213)
        self.assertLess(report["candidate_bytes"], assembler.COMPACT_SIZE_CEILING_BYTES)
        self.assertRegex(report["ownership_digest"], r"^[0-9a-f]{64}$")
        self.assertRegex(report["schema_digest"], r"^[0-9a-f]{64}$")
        self.assertRegex(report["payload_fingerprint"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
