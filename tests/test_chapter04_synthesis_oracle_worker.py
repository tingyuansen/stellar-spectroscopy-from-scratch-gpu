"""Focused structural tests for the Chapter 4 synthesis oracle worker."""

from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from scripts import chapter04_synthesis_oracle_worker as worker


PINNED_PIPELINE = (
    worker.PINNED_ROOT / "payne_zero_synthesis" / "pipeline.py"
)
EXPECTED_FRESH_ROUTE_RESULTS = {
    "full": {
        "captured_route": "full",
        "key_count": 196,
        "digest": (
            "f4e36d39a9b736ade95972d810801e7e"
            "caad66a5c7c0985b83ca897f0485b8d0"
        ),
        "molecular_call_count": 2,
        "principal_shape": [6, 200],
        "iteration_vectors": [[8, 5, 5, 4, 6, 7], [8, 5, 5, 4, 6, 7]],
    },
    "fixed": {
        "captured_route": "fixed_derived_mass",
        "key_count": 155,
        "digest": (
            "ce0cb3c2a2d323011dfe4b7a0a4ea1dc"
            "9a37df2ad5dbad22bd6482539995a56f"
        ),
        "molecular_call_count": 1,
        "principal_shape": [6, 200],
        "iteration_vectors": [[8, 5, 5, 5, 7, 7]],
    },
    "fixed-supplied": {
        "captured_route": "fixed_supplied_mass",
        "key_count": 156,
        "digest": (
            "1e9782da2c9990bbefa74fd34d26dd72"
            "d8a8445773e8b646d773a7a9935a4444"
        ),
        "molecular_call_count": 1,
        "principal_shape": [6, 200],
        "iteration_vectors": [[8, 5, 5, 5, 7, 7]],
        "supplied_mass_declared": True,
    },
    "public": {
        "captured_route": "public_structured_mapping",
        "key_count": 244,
        "digest": (
            "2e87bbc3b8bf09b03af232515518513d"
            "aa0d1886e120b677355496c333bfee96"
        ),
        "molecular_call_count": 1,
        "principal_shape": [6, 6, 139],
        "iteration_vectors": [[8, 5, 5, 5, 7, 7]],
        "production_line_mapping_call_count": 1,
        "grounded_production_line_mapping_call_count": 0,
        "co_exact_reconstruction": True,
        "co_raw_discrimination_count": 6,
    },
    "boundaries": {
        "captured_route": "synthesis_boundary_probes",
        "key_count": 112,
        "digest": (
            "49b633ef299deb2ad3d37009506a79daa"
            "1931c941997e742b7fdd7fd4b8c62f1"
        ),
        "molecular_call_count": 1,
        "principal_shape": [2, 190],
        "iteration_vectors": [],
        "polynomial_temperature_bits": [
            4666723172467343360,
            4666723172467343361,
        ],
        "polynomial_branch_mask": [True, False],
        "provisional_h2_temperature_bits": [
            4666173416653455360,
            4666173416653455361,
        ],
        "provisional_h2_branch_mask": [True, False],
    },
}


def fake_solve_molecular_equilibrium(
    temperature,
    gas_pressure,
    electron_density,
    elemental_abundances,
    ion_formation_constants,
    *,
    molecules_path=None,
    device=None,
    dtype=torch.float32,
    max_iter=200,
    tol=1.0e-3,
    chain_length=None,
    return_diagnostics=False,
):
    """Small direct-solver double with the real return and exhaustion seams."""

    depth_count = np.asarray(temperature).size
    converged_iters = np.zeros(depth_count, dtype=np.int64)
    for depth_index in range(depth_count):
        for _ in range(max_iter):
            converged_iters[depth_index] = 1
            break
        else:
            converged_iters[depth_index] = max_iter

    heavy = torch.arange(
        1, depth_count + 1, dtype=dtype, device=device
    )
    molecular = torch.zeros(
        (depth_count, 200), dtype=dtype, device=device
    )
    equations = torch.zeros(
        (depth_count, 30), dtype=dtype, device=device
    )
    equations[:, 0] = heavy
    equations[:, 1] = heavy / 10.0
    electron = equations[:, 1]
    structure = SimpleNamespace(
        equation_count=23,
        electron_equation_index=22,
        component_multiplicity=torch.zeros(
            (190, 23), dtype=dtype, device=device
        ),
        inverse_electron_power=torch.zeros(
            190, dtype=dtype, device=device
        ),
        negative_ion_flag=torch.zeros(190, dtype=dtype, device=device),
        active_molecule_mask=torch.zeros(190, dtype=dtype, device=device),
        full_component_multiplicity=torch.zeros(
            (190, 23), dtype=dtype, device=device
        ),
        full_inverse_electron_power=torch.zeros(
            190, dtype=dtype, device=device
        ),
    )
    diagnostics = {
        "molecule_count": 190,
        "equation_count": 23,
        "equation_species_codes": np.array(
            [0, *range(1, 22), 100], dtype=np.int64
        ),
        "molecule_codes": np.arange(190, dtype=np.float64),
        "iterations_completed": converged_iters,
        "natural_log_formation_constants": torch.zeros(
            (depth_count, 190), dtype=dtype, device=device
        ),
        "structure": structure,
    }
    outputs = (heavy, molecular, equations, electron)
    return (*outputs, diagnostics) if return_diagnostics else outputs


def solve_electron_density(module, *, max_iter=2):
    """Call through the exact first full-route owner name."""

    temperature = np.array([4000.0, 5000.0])
    gas_pressure = np.array([1.0e4, 1.0e5])
    return module.solve_molecular_equilibrium(
        temperature,
        gas_pressure,
        np.ones(2),
        np.ones(99),
        np.ones((2, 190)),
        molecules_path=worker.MOLECULE_CATALOG,
        device=torch.device("cpu"),
        dtype=torch.float64,
        max_iter=max_iter,
        tol=1.0e-4,
    )


def _molecule_backed_population_state(module):
    """Call through the exact fixed/second-full owner name."""

    temperature = np.array([4000.0, 5000.0])
    gas_pressure = np.array([1.0e4, 1.0e5])
    return module.solve_molecular_equilibrium(
        temperature,
        gas_pressure,
        np.ones(2),
        np.ones(99),
        np.ones((2, 190)),
        molecules_path=worker.MOLECULE_CATALOG,
        device=torch.device("cpu"),
        dtype=torch.float64,
        tol=1.0e-4,
    )


def boundary_trace_target(values):
    """Tiny source-trace target used to prove locals are copied before abort."""

    doubled = np.asarray(values) * 2.0
    marker = doubled + 1.0
    return marker


class Chapter04SynthesisOracleWorkerTests(unittest.TestCase):
    """Pin identity, capture behavior, diagnostics, and exact boundaries."""

    @unittest.skipUnless(
        worker.PINNED_ROOT.is_dir(), "pinned Payne Zero checkout absent"
    )
    def test_pinned_source_identity_checks_commit_and_every_hash(self) -> None:
        identity = worker.verify_pinned_source_identity()
        self.assertEqual(identity.root, worker.PINNED_ROOT.resolve())
        self.assertEqual(identity.commit, worker.PINNED_COMMIT)
        self.assertEqual(
            set(identity.sha256_by_name), set(worker.EXPECTED_SOURCE_SHA256)
        )
        for name, (_, expected) in worker.EXPECTED_SOURCE_SHA256.items():
            self.assertEqual(identity.sha256_by_name[name], expected)

    def test_identity_rejects_any_other_root_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "permit only"):
                worker.verify_pinned_source_identity(Path(temporary))

    def test_fixture_is_exactly_input_only(self) -> None:
        inputs = worker.load_input_fixture()
        self.assertEqual(set(inputs), set(worker.FIXTURE_KEYS))
        self.assertEqual(inputs["temperature"].shape, (6,))
        self.assertEqual(inputs["elemental_abundances"].shape, (99,))
        self.assertEqual(inputs["temperature"].dtype, np.dtype(np.float64))

        contaminated = dict(inputs)
        contaminated["state__electron_density"] = np.ones(6)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contaminated.npz"
            np.savez(path, **contaminated)
            with self.assertRaisesRegex(ValueError, "unexpected"):
                worker.load_input_fixture(path)

    def test_supplied_mass_vector_is_input_only_and_declared(self) -> None:
        inputs = worker.load_input_fixture()
        supplied = worker._deterministic_supplied_mass_density(inputs)
        changed_output_seed = dict(inputs)
        changed_output_seed["electron_density_seed"] = (
            7.0 * inputs["electron_density_seed"]
        )
        np.testing.assert_array_equal(
            supplied,
            worker._deterministic_supplied_mass_density(changed_output_seed),
        )
        self.assertEqual(supplied.shape, (6,))
        self.assertEqual(supplied.dtype, np.dtype(np.float64))
        self.assertTrue(np.all(np.isfinite(supplied)))
        self.assertTrue(np.all(supplied > 0.0))

    def test_capture_requests_diagnostics_but_preserves_four_item_return(self) -> None:
        module = SimpleNamespace(
            solve_molecular_equilibrium=fake_solve_molecular_equilibrium
        )
        original = module.solve_molecular_equilibrium
        with worker.MolecularSolveCapture(module) as capture:
            result = solve_electron_density(module)

        self.assertIs(module.solve_molecular_equilibrium, original)
        self.assertEqual(len(result), 4)
        self.assertEqual(len(capture.calls), 1)
        call = capture.calls[0]
        self.assertEqual(call.caller_name, "solve_electron_density")
        self.assertEqual(call.effective_arguments["max_iter"], 2)
        self.assertEqual(call.effective_arguments["tol"], 1.0e-4)
        self.assertFalse(call.exhausted_mask.any())
        self.assertIn("natural_log_formation_constants", call.diagnostics)

    def test_trace_distinguishes_true_exhaustion_from_iteration_count(self) -> None:
        module = SimpleNamespace(
            solve_molecular_equilibrium=fake_solve_molecular_equilibrium
        )
        with worker.MolecularSolveCapture(module) as capture:
            solve_electron_density(module, max_iter=0)
        np.testing.assert_array_equal(
            capture.calls[0].exhausted_mask,
            np.array([True, True]),
        )

    def test_full_and_fixed_call_ownership_are_exact_and_ordered(self) -> None:
        module = SimpleNamespace(
            solve_molecular_equilibrium=fake_solve_molecular_equilibrium
        )
        with worker.MolecularSolveCapture(module) as capture:
            solve_electron_density(module)
            _molecule_backed_population_state(module)
        worker.assert_route_call_ownership(
            "full", capture.calls, enforce_pinned_caller=False
        )
        with self.assertRaisesRegex(AssertionError, "callers"):
            worker.assert_route_call_ownership(
                "full",
                list(reversed(capture.calls)),
                enforce_pinned_caller=False,
            )
        worker.assert_route_call_ownership(
            "fixed", [capture.calls[1]], enforce_pinned_caller=False
        )
        with self.assertRaisesRegex(AssertionError, "callers"):
            worker.assert_route_call_ownership(
                "fixed", capture.calls, enforce_pinned_caller=False
            )

    def test_final_residual_diagnostics_remain_call_local(self) -> None:
        dtype = torch.float64
        structure = SimpleNamespace(
            electron_equation_index=1,
            component_multiplicity=torch.tensor(
                [[0.0, 1.0], [0.0, 0.0]], dtype=dtype
            ),
            inverse_electron_power=torch.zeros(2, dtype=dtype),
            negative_ion_flag=torch.zeros(2, dtype=dtype),
            active_molecule_mask=torch.ones(2, dtype=dtype),
        )
        equation_densities = torch.zeros((2, 30), dtype=dtype)
        equation_densities[:, :2] = torch.tensor(
            [[10.0, 1.0], [20.0, 2.0]], dtype=dtype
        )
        call = worker.CapturedMolecularCall(
            caller_name="solve_electron_density",
            caller_module="test",
            caller_file=__file__,
            effective_arguments={
                "temperature": np.array([4000.0, 5000.0]),
                "gas_pressure": np.array([1.0e4, 1.0e5]),
                "electron_density": np.ones(2),
                "elemental_abundances": np.ones(99),
                "ion_formation_constants": np.ones((2, 2)),
                "molecules_path": worker.MOLECULE_CATALOG,
                "device": torch.device("cpu"),
                "dtype": dtype,
                "max_iter": 200,
                "tol": 1.0e-4,
                "chain_length": None,
                "return_diagnostics": False,
            },
            outputs=(
                equation_densities[:, 0],
                torch.zeros((2, 200), dtype=dtype),
                equation_densities,
                equation_densities[:, 1],
            ),
            diagnostics={
                "molecule_count": 2,
                "equation_count": 2,
                "equation_species_codes": np.array([0, 100]),
                "molecule_codes": np.array([101.0, 608.0]),
                "iterations_completed": np.ones(2, dtype=np.int64),
                "natural_log_formation_constants": torch.tensor(
                    [[0.0, 0.0], [1000.0, 0.0]], dtype=dtype
                ),
                "structure": structure,
            },
            exhausted_mask=np.zeros(2, dtype=np.bool_),
        )

        def safe_log(values):
            return torch.log(values.clamp_min(torch.finfo(values.dtype).tiny))

        def residual(
            densities,
            natural_log_formation_constants,
            equation_abundance,
            total_particle_density,
            structure,
        ):
            return densities - torch.tensor(
                [total_particle_density, 0.0], dtype=densities.dtype
            )

        molecular = SimpleNamespace(
            torch=torch,
            BOLTZMANN_ERG_PER_K=1.380649e-16,
            _safe_log=safe_log,
            _residual=residual,
        )
        diagnostics = worker._residual_diagnostics(call, molecular)
        self.assertEqual(diagnostics["residual"].shape, (2, 2))
        self.assertEqual(diagnostics["normalized_residual"].shape, (2, 2))
        np.testing.assert_array_equal(
            worker._as_numpy(
                diagnostics["pre_replacement_term_nonfinite_count"]
            ),
            np.array([0, 1]),
        )
        self.assertFalse(
            np.shares_memory(
                worker._as_numpy(diagnostics["residual"]),
                worker._as_numpy(call.outputs[2]),
            )
        )

    def test_deterministic_result_sorts_copies_and_rejects_objects(self) -> None:
        source = np.array([1.0, 2.0])
        result = worker.deterministic_result(
            {"z": np.array(3), "a": source}
        )
        self.assertEqual(list(result), ["a", "z"])
        source[0] = -1.0
        np.testing.assert_array_equal(result["a"], np.array([1.0, 2.0]))
        with self.assertRaisesRegex(TypeError, "object dtype"):
            worker.deterministic_result(
                {"bad": np.array([object()], dtype=object)}
            )

    @unittest.skipUnless(
        PINNED_PIPELINE.is_file(), "pinned Payne Zero checkout absent"
    )
    def test_public_builder_pins_the_real_signature(self) -> None:
        tree = ast.parse(
            PINNED_PIPELINE.read_text(encoding="utf-8"),
            filename=str(PINNED_PIPELINE),
        )
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_structured_atmosphere_from_columns"
        )
        actual = tuple(argument.arg for argument in function.args.kwonlyargs)
        self.assertEqual(actual, worker.PUBLIC_BUILDER_PARAMETERS)
        self.assertIn("eos_tables", actual)
        self.assertIn("molecular_species_codes", actual)
        self.assertIn("molecules_path", actual)
        self.assertNotIn("device", actual)
        self.assertNotIn("dtype", actual)
        self.assertNotIn("molecular_lines", actual)

        def wrong_builder(*, temperature, device):
            return None

        fake_pipeline = SimpleNamespace(
            build_structured_atmosphere_from_columns=wrong_builder
        )
        with self.assertRaisesRegex(RuntimeError, "parameters"):
            worker.assert_public_builder_signature(fake_pipeline)

    def test_supported_species_mapping_is_lossless_and_pins_co(self) -> None:
        import payne_zero_synthesis.molecular_equilibrium as molecular

        mapping = worker.molecular_species_mapping_arrays(molecular)
        self.assertEqual(mapping["species_codes"].shape, (54,))
        self.assertEqual(mapping["molecule_code_offsets"].shape, (55,))
        self.assertEqual(mapping["molecule_code_offsets"][0], 0)
        self.assertEqual(
            mapping["molecule_code_offsets"][-1],
            mapping["molecule_codes"].size,
        )
        co_index = int(np.flatnonzero(mapping["species_codes"] == 276)[0])
        start = mapping["molecule_code_offsets"][co_index]
        stop = mapping["molecule_code_offsets"][co_index + 1]
        np.testing.assert_array_equal(
            mapping["molecule_codes"][start:stop], np.array([608.0])
        )
        self.assertEqual(mapping["public_columns"][co_index], 45)

    def test_ground_policy_fixture_must_discriminate(self) -> None:
        with self.assertRaisesRegex(AssertionError, "does not discriminate"):
            worker._ground_discrimination_mask(
                np.ones((2, 3)), np.ones((2, 3))
            )
        mask = worker._ground_discrimination_mask(
            np.array([[1.0, 2.0]]),
            np.array([[1.0, 3.0]]),
        )
        np.testing.assert_array_equal(mask, np.array([[False, True]]))

    def test_public_helper_wrappers_restore_and_copy_results(self) -> None:
        fixed_result = SimpleNamespace(name="fixed")

        def fixed(*, electron_density, tables):
            return fixed_result

        def partition(*, elements, nion=1, apply_ground_partition=True):
            return {1: np.ones((2, nion))}

        def line(
            *,
            temperature,
            equation_densities,
            neutral_partition,
            species_codes,
            molecules_path=None,
        ):
            return {
                int(code): np.ones(2) * int(code)
                for code in species_codes
            }

        def edge(edge_table_path=worker.CONTINUUM_EDGE_GRID):
            return {"continuum_edge_wavelength_nm": np.array([1.0, 2.0])}

        eos = SimpleNamespace(
            solve_population_state_at_electron_density=fixed,
            partition_functions_for_elements=partition,
        )
        molecular = SimpleNamespace(
            molecular_line_populations_by_species_code=line
        )
        pipeline = SimpleNamespace(_build_edge_grid=edge)
        originals = (
            eos.solve_population_state_at_electron_density,
            eos.partition_functions_for_elements,
            molecular.molecular_line_populations_by_species_code,
            pipeline._build_edge_grid,
        )
        with worker.PublicBuilderCapture(
            eos, molecular, pipeline
        ) as capture:
            self.assertIs(
                eos.solve_population_state_at_electron_density(
                    electron_density=np.ones(2), tables=object()
                ),
                fixed_result,
            )
            eos.partition_functions_for_elements(
                elements=[1], nion=6, apply_ground_partition=False
            )
            line_result = (
                molecular.molecular_line_populations_by_species_code(
                    temperature=np.ones(2),
                    equation_densities=np.ones((2, 2)),
                    neutral_partition=np.ones((2, 99)),
                    species_codes=np.array([240]),
                    molecules_path=worker.MOLECULE_CATALOG,
                )
            )
            pipeline._build_edge_grid()
            line_result[240][0] = -1.0

        self.assertIs(
            eos.solve_population_state_at_electron_density, originals[0]
        )
        self.assertIs(eos.partition_functions_for_elements, originals[1])
        self.assertIs(
            molecular.molecular_line_populations_by_species_code,
            originals[2],
        )
        self.assertIs(pipeline._build_edge_grid, originals[3])
        self.assertEqual(len(capture.fixed_calls), 1)
        self.assertEqual(len(capture.partition_calls), 1)
        self.assertEqual(len(capture.line_calls), 1)
        self.assertEqual(len(capture.edge_calls), 1)
        self.assertEqual(capture.line_calls[0].result[240][0], 240.0)

    def test_exact_boundary_trace_copies_locals_and_aborts(self) -> None:
        target_line = worker._unique_source_line(
            boundary_trace_target, "return marker"
        )
        captured = worker._capture_locals_at_line(
            boundary_trace_target,
            target_line,
            ("doubled", "marker"),
            np.array([2.0, 3.0]),
        )
        np.testing.assert_array_equal(captured["doubled"], np.array([4.0, 6.0]))
        np.testing.assert_array_equal(captured["marker"], np.array([5.0, 7.0]))

    def test_boundary_entry_point_is_implemented_and_requires_inputs(self) -> None:
        signature = inspect.signature(worker.capture_exotic_boundary_traces)
        self.assertEqual(
            tuple(signature.parameters)[:2], ("runtime", "inputs")
        )

    @unittest.skipUnless(
        PINNED_PIPELINE.is_file(), "pinned Payne Zero checkout absent"
    )
    def test_all_real_routes_are_stable_in_fresh_pinned_processes(self) -> None:
        environment = os.environ.copy()
        environment.update(worker.ORACLE_PROCESS_ENVIRONMENT)
        worker_path = (
            worker.REPOSITORY_ROOT
            / "scripts"
            / "chapter04_synthesis_oracle_worker.py"
        )
        for route, expected in EXPECTED_FRESH_ROUTE_RESULTS.items():
            with self.subTest(route=route):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(worker_path),
                        "--route",
                        route,
                    ],
                    cwd=worker.REPOSITORY_ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                actual = json.loads(completed.stdout.strip().splitlines()[-1])
                for name, value in expected.items():
                    self.assertEqual(actual[name], value)
                self.assertEqual(actual["requested_route"], route)
                self.assertEqual(actual["exhaustion_count"], 0)
                self.assertEqual(actual["shared_catalog_count"], 170)
                self.assertEqual(actual["synthesis_only_catalog_count"], 20)
                self.assertEqual(actual["catalog_semantic_mismatch_count"], 0)
                self.assertEqual(
                    actual["synthesis_only_catalog_codes"],
                    worker.EXPECTED_SYNTHESIS_ONLY_CODES.tolist(),
                )
                self.assertGreater(actual["catalog_row_reordering_count"], 0)
                self.assertGreater(actual["executed_source_count"], 0)
                self.assertEqual(actual["process_environment_count"], 12)
                self.assertEqual(len(worker.ORACLE_PROCESS_ENVIRONMENT), 12)
                self.assertEqual(
                    worker.ORACLE_PROCESS_ENVIRONMENT["NUMBA_NUM_THREADS"],
                    "1",
                )
                self.assertTrue(actual["platform"])
                self.assertIn(actual["system_byteorder"], {"little", "big"})
                self.assertTrue(actual["blas_name"])


if __name__ == "__main__":
    unittest.main()
