"""Cheap structural and transaction tests for the Chapter 4 publisher."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import build_chapter04_payne_zero_goldens as publisher
from scripts.deterministic_npz import write_npz


def _write_named_files(directory: Path, marker: int = 1) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for index, name in enumerate(publisher.OUTPUT_NAMES):
        write_npz(
            directory / name,
            {"marker": np.asarray(marker + index, dtype=np.int64)},
        )


def _write_fake_capture(root: Path, marker: int = 1) -> None:
    raw = root / "raw"
    final = root / "final"
    raw.mkdir(parents=True)
    final.mkdir()
    for index, name in enumerate(publisher.RAW_NAMES):
        write_npz(
            raw / name,
            {"marker": np.asarray(marker + index, dtype=np.int64)},
        )
    for index, name in enumerate(publisher.OUTPUT_NAMES):
        write_npz(
            final / name,
            {"marker": np.asarray(marker + index, dtype=np.int64)},
        )


def _minimal_constants_payload() -> dict[str, np.ndarray]:
    payload = publisher._publisher_metadata("molecular_constants")
    payload.update(
        {
            "atmosphere__boundary_marker": np.asarray(1),
            "atmosphere__h2_probe_marker": np.asarray(1),
            "synthesis__alignment__marker": np.asarray(1),
            "synthesis__boundary__marker": np.asarray(1),
            "synthesis__catalog__marker": np.asarray(1),
        }
    )
    return payload


def _synthetic_full_raw() -> dict[str, np.ndarray]:
    temperature = np.asarray([4000.0])
    gas_pressure = np.asarray([1.0e4])
    abundance = np.asarray([1.0, 0.1])
    electron_seed = np.asarray([10.0])
    closure_electron_input = np.asarray([15.0])
    public_electron = np.asarray([20.0])
    total_nuclei = np.asarray([30.0])
    molecules = np.asarray([[4.0, 5.0]])
    equations = np.asarray([[30.0, 20.0]])
    arrays = {
        "input__temperature": temperature,
        "input__gas_pressure": gas_pressure,
        "input__elemental_abundances": abundance,
        "input__electron_density_seed": electron_seed,
        "state__electron_density": public_electron,
        "state__total_nuclei_number_density": total_nuclei,
        "state__molecular_populations": molecules,
        "state__molecular_equation_densities": equations,
    }
    for index, electron_input in enumerate(
        (closure_electron_input, public_electron)
    ):
        prefix = f"call_{index}__"
        arrays.update(
            {
                f"{prefix}input__temperature": temperature,
                f"{prefix}input__gas_pressure": gas_pressure,
                f"{prefix}input__elemental_abundances": abundance,
                f"{prefix}input__electron_density": electron_input,
            }
        )
    arrays.update(
        {
            "call_0__output__electron_density": public_electron,
            "call_1__output__electron_density": np.asarray([21.0]),
            "call_1__output__total_nuclei_number_density": total_nuclei,
            "call_1__output__molecular_populations": molecules,
            "call_1__output__equation_densities": equations,
        }
    )
    return arrays


def _synthetic_fixed_raw(*, supplied: bool) -> dict[str, np.ndarray]:
    temperature = np.asarray([4000.0])
    gas_pressure = np.asarray([1.0e4])
    abundance = np.asarray([1.0, 0.1])
    electron_seed = np.asarray([10.0])
    internal_electron = np.asarray([12.0])
    total_nuclei = np.asarray([30.0])
    molecules = np.asarray([[4.0, 5.0]])
    equations = np.asarray([[30.0, 12.0]])
    mass = np.asarray([2.0e-9 if supplied else 3.0e-9])
    arrays = {
        "input__temperature": temperature,
        "input__gas_pressure": gas_pressure,
        "input__elemental_abundances": abundance,
        "input__electron_density_seed": electron_seed,
        "call_0__input__temperature": temperature,
        "call_0__input__gas_pressure": gas_pressure,
        "call_0__input__elemental_abundances": abundance,
        "call_0__input__electron_density": electron_seed,
        "call_0__output__electron_density": internal_electron,
        "call_0__output__total_nuclei_number_density": total_nuclei,
        "call_0__output__molecular_populations": molecules,
        "call_0__output__equation_densities": equations,
        "state__electron_density": electron_seed,
        "state__total_nuclei_number_density": total_nuclei,
        "state__molecular_populations": molecules,
        "state__molecular_equation_densities": equations,
        "state__mass_density": mass,
        "trace__internal_electron_density": internal_electron,
    }
    if supplied:
        arrays["input__mass_density"] = mass
    return arrays


def _synthetic_public_payload() -> dict[str, np.ndarray]:
    temperature = np.asarray([4000.0])
    gas_pressure = np.asarray([1.0e4])
    abundance = np.asarray([1.0, 0.1])
    electron_seed = np.asarray([10.0])
    internal_electron = np.asarray([12.0])
    total_nuclei = np.asarray([30.0])
    molecules = np.asarray([[9.0]])
    equations = np.asarray([[30.0, 12.0]])
    ion_cube = np.asarray([[[1.0]]])
    partition_cube = np.asarray([[[2.0]]])
    public_lane = np.asarray([[7.0]])
    grounded_lane = np.asarray([[8.0]])
    edge_frequency = np.asarray([1.0, 2.0])
    edge_wavelength = np.asarray([3.0, 4.0])
    edge_midpoint = np.asarray([3.5])
    edge_width = np.asarray([0.5])
    return {
        "input__temperature": temperature,
        "input__gas_pressure": gas_pressure,
        "input__elemental_abundances": abundance,
        "input__electron_density_seed": electron_seed,
        "call_0__input__temperature": temperature,
        "call_0__input__gas_pressure": gas_pressure,
        "call_0__input__elemental_abundances": abundance,
        "call_0__input__electron_density": electron_seed,
        "call_0__output__electron_density": internal_electron,
        "call_0__output__total_nuclei_number_density": total_nuclei,
        "call_0__output__molecular_populations": molecules,
        "call_0__output__equation_densities": equations,
        "fixed_state__electron_density": electron_seed,
        "fixed_state__total_nuclei_number_density": total_nuclei,
        "fixed_state__molecular_populations": molecules,
        "fixed_state__molecular_equation_densities": equations,
        "fixed_state__ion_stage_populations": ion_cube,
        "fixed_state__partition_normalized_populations": partition_cube,
        "structured__ion_stage_populations": ion_cube,
        "structured__partition_normalized_populations": partition_cube,
        "ion_cube__before": ion_cube,
        "ion_cube__after": ion_cube,
        "partition_cube__before": partition_cube,
        "partition_cube__after": partition_cube,
        "line_population__public": public_lane,
        "line_population__no_ground": public_lane,
        "line_population__independent_no_ground": public_lane,
        "line_population__grounded": grounded_lane,
        "mapping__species_codes": np.asarray([276]),
        "co_reconstruction__reference_public_lane": public_lane[:, 0],
        "co_reconstruction__independent_population": public_lane[:, 0],
        "co_reconstruction__difference": np.asarray([0.0]),
        "co_reconstruction__grounded_population": grounded_lane[:, 0],
        "molecular_hydrogen__catalog_index": np.asarray(0),
        "molecular_hydrogen__solved_code_101": molecules[:, 0],
        "edge_loader__signed_continuum_edge_frequency_hz": edge_frequency,
        "structured__signed_continuum_edge_frequency_hz": edge_frequency,
        "edge_loader__continuum_edge_wavelength_nm": edge_wavelength,
        "structured__continuum_edge_wavelength_nm": edge_wavelength,
        "edge_loader__continuum_edge_midpoint_wavelength_nm": edge_midpoint,
        "structured__continuum_edge_midpoint_wavelength_nm": edge_midpoint,
        "edge_loader__edge_interval_width_squared_over_two_nm2": edge_width,
        "structured__continuum_edge_interval_width_squared_over_two_nm2": (
            edge_width
        ),
    }


class Chapter04GoldenPublisherTests(unittest.TestCase):
    """Pin subprocess isolation, schemas, and no-partial publication."""

    def test_exact_names_routes_and_atomic_subdirectory_are_pinned(self) -> None:
        self.assertEqual(len(publisher.OUTPUT_NAMES), 5)
        self.assertEqual(len(set(publisher.OUTPUT_NAMES)), 5)
        self.assertEqual(len(publisher.ROUTES), 6)
        self.assertEqual(len(set(publisher.ROUTES)), 6)
        self.assertEqual(
            publisher.OUTPUT_DIR.parts[-4:],
            ("data", "golden", "payne_zero", "chapter04"),
        )
        self.assertEqual(
            publisher.CONSTANTS_NAME,
            "chapter04_molecular_constants_cpu_float64.npz",
        )

    def test_all_twelve_controls_are_exact_and_child_caches_are_distinct(self) -> None:
        self.assertEqual(len(publisher.PROCESS_CONTROLS), 12)
        with tempfile.TemporaryDirectory() as temporary:
            first_cache = Path(temporary) / "a"
            second_cache = Path(temporary) / "b"
            first = publisher._child_environment(first_cache)
            second = publisher._child_environment(second_cache)
        for name, expected in publisher.PROCESS_CONTROLS.items():
            self.assertEqual(first[name], expected)
            self.assertEqual(second[name], expected)
        self.assertNotEqual(first["NUMBA_CACHE_DIR"], second["NUMBA_CACHE_DIR"])
        self.assertEqual(
            first["PYTHONPATH"].split(os.pathsep)[:2],
            [str(publisher.PINNED_ROOT), str(publisher.REPOSITORY_ROOT)],
        )

    def test_raw_schemas_pin_current_key_counts_and_scientific_seams(self) -> None:
        self.assertEqual(
            {route: schema.key_count for route, schema in publisher.RAW_SCHEMAS.items()},
            {
                "atmosphere": 460,
                "synthesis-boundaries": 112,
                "synthesis-full": 196,
                "synthesis-fixed-derived": 155,
                "synthesis-fixed-supplied": 156,
                "synthesis-public": 244,
            },
        )
        for route in publisher.ROUTES:
            with self.subTest(route=route):
                schema = publisher.RAW_SCHEMAS[route]
                self.assertTrue(schema.required)
                self.assertTrue(schema.allowed_prefixes)
        self.assertIn(
            "full_newton_preupdate_equation_densities",
            publisher.RAW_SCHEMAS["atmosphere"].required,
        )
        self.assertIn(
            "co_reconstruction__independent_population",
            publisher.RAW_SCHEMAS["synthesis-public"].required,
        )

    def test_accepted_route_digests_and_input_hashes_are_frozen(self) -> None:
        self.assertEqual(set(publisher.ACCEPTED_RAW_DIGESTS), set(publisher.ROUTES))
        self.assertTrue(
            all(
                len(value) == 64
                for value in publisher.ACCEPTED_RAW_DIGESTS.values()
            )
        )
        for name, path in publisher.PINNED_PUBLISHER_INPUT_PATHS.items():
            with self.subTest(name=name):
                self.assertEqual(
                    publisher.sha256(path),
                    publisher.PINNED_PUBLISHER_INPUT_SHA256[name],
                )

    def test_route_digest_accepts_exact_mapping_and_rejects_value_and_name_drift(
        self,
    ) -> None:
        for route in ("atmosphere", "synthesis-full"):
            with self.subTest(route=route):
                arrays = {
                    "a": np.asarray([1, 2], dtype=np.uint8),
                    "b": np.asarray([3.5], dtype=np.float64),
                }
                accepted = publisher.raw_result_digest(route, arrays)
                expected = {
                    "atmosphere": (
                        "472cd9c3775c3afa37790b9a2ac67ff01"
                        "46c553de7fc51e25b6629b6f65d2fd2"
                    ),
                    "synthesis-full": (
                        "abf3fe4dc3478c59b9a835b9a03c20dd"
                        "d659e251f45012067ba5aa5160eed31e"
                    ),
                }[route]
                self.assertEqual(accepted, expected)
                with mock.patch.dict(
                    publisher.ACCEPTED_RAW_DIGESTS,
                    {route: accepted},
                ):
                    publisher.validate_raw_digest(route, arrays)
                    changed_value = {
                        **arrays,
                        "a": np.asarray([1, 3], dtype=np.uint8),
                    }
                    with self.assertRaisesRegex(AssertionError, "raw digest"):
                        publisher.validate_raw_digest(route, changed_value)
                    changed_name = {"a": arrays["a"], "c": arrays["b"]}
                    with self.assertRaisesRegex(AssertionError, "raw digest"):
                        publisher.validate_raw_digest(route, changed_name)

    def test_full_fixed_and_public_electron_aliases_preserve_internal_values(
        self,
    ) -> None:
        full = publisher.build_full_payload(
            _synthetic_full_raw(),
            "0" * 64,
        )
        self.assertNotIn("state__electron_density", full)
        self.assertIn("call_0__input__electron_density", full)
        self.assertNotIn("call_1__input__electron_density", full)
        np.testing.assert_array_equal(
            full["call_0__input__electron_density"],
            np.asarray([15.0]),
        )
        self.assertFalse(
            np.array_equal(
                full["call_0__input__electron_density"],
                full["input__electron_density_seed"],
            )
        )
        np.testing.assert_array_equal(
            full["call_1__output__electron_density"],
            np.asarray([21.0]),
        )
        self.assertEqual(
            str(full["call_1__input__electron_density_member"]),
            "call_0__output__electron_density",
        )

        fixed = publisher.build_fixed_payload(
            _synthetic_fixed_raw(supplied=False),
            _synthetic_fixed_raw(supplied=True),
            "0" * 64,
        )
        for branch in ("derived", "supplied"):
            self.assertNotIn(f"{branch}__state__electron_density", fixed)
            self.assertNotIn(
                f"{branch}__call_0__input__electron_density",
                fixed,
            )
            self.assertNotIn(
                f"{branch}__trace__internal_electron_density",
                fixed,
            )
            np.testing.assert_array_equal(
                fixed[f"{branch}__call_0__output__electron_density"],
                np.asarray([12.0]),
            )
            self.assertEqual(
                str(
                    fixed[
                        f"{branch}__alias__"
                        "trace__internal_electron_density_member"
                    ]
                ),
                f"{branch}__call_0__output__electron_density",
            )
        np.testing.assert_array_equal(
            fixed["input__electron_density_seed"],
            np.asarray([10.0]),
        )

        public = _synthetic_public_payload()
        publisher._deduplicate_public_payload(public)
        self.assertNotIn("fixed_state__electron_density", public)
        self.assertNotIn("call_0__input__electron_density", public)
        np.testing.assert_array_equal(
            public["call_0__output__electron_density"],
            np.asarray([12.0]),
        )

    def test_final_metadata_kind_schema_and_hash_mutations_are_rejected(self) -> None:
        payload = _minimal_constants_payload()
        publisher.validate_final_payload(
            publisher.CONSTANTS_NAME,
            payload,
            constants_sha256=None,
        )
        mutations = {
            "meta__archive_kind": np.asarray("wrong"),
            "meta__archive_schema_version": np.asarray(2, dtype=np.int64),
            "meta__oracle_acceptance_sha256": np.asarray("f" * 64),
            "meta__atmosphere_worker_sha256": np.asarray("e" * 64),
        }
        for member, changed in mutations.items():
            with self.subTest(member=member):
                mutated = {name: value.copy() for name, value in payload.items()}
                mutated[member] = changed
                with self.assertRaises(AssertionError):
                    publisher.validate_final_payload(
                        publisher.CONSTANTS_NAME,
                        mutated,
                        constants_sha256=None,
                    )

    def test_capture_set_calls_six_routes_in_order_with_distinct_caches(self) -> None:
        calls = []

        def fake_route(route: str, raw_path: Path, cache_path: Path) -> None:
            calls.append((route, raw_path, cache_path))
            write_npz(raw_path, {"route": np.asarray(route)})

        def fake_assembler(raw_dir: Path, final_dir: Path) -> None:
            self.assertEqual(
                sorted(path.name for path in raw_dir.iterdir()),
                sorted(publisher.RAW_NAMES),
            )
            _write_named_files(final_dir)

        with tempfile.TemporaryDirectory() as temporary:
            publisher.build_capture_set(
                Path(temporary) / "capture",
                route_runner=fake_route,
                assembler=fake_assembler,
            )
        self.assertEqual([call[0] for call in calls], list(publisher.ROUTES))
        self.assertEqual(len({call[2] for call in calls}), 6)
        self.assertTrue(all(not call[2].exists() for call in calls))

    def test_raw_and_final_byte_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            _write_named_files(first)
            _write_named_files(second)
            publisher.compare_file_sets(
                first,
                second,
                publisher.OUTPUT_NAMES,
                label="fake final",
            )
            write_npz(
                second / publisher.OUTPUT_NAMES[-1],
                {"marker": np.asarray(-1, dtype=np.int64)},
            )
            with self.assertRaisesRegex(AssertionError, "differs at byte"):
                publisher.compare_file_sets(
                    first,
                    second,
                    publisher.OUTPUT_NAMES,
                    label="fake final",
                )

    def test_failed_second_capture_never_calls_publisher(self) -> None:
        calls = 0

        def fake_capture(root: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected second-capture failure")
            _write_fake_capture(root)

        fake_publish = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "injected"):
            publisher.generate_and_maybe_publish(
                verify_only=False,
                capture_builder=fake_capture,
                publisher=fake_publish,
            )
        fake_publish.assert_not_called()

    def test_verify_only_compares_both_sets_but_never_publishes(self) -> None:
        calls = 0

        def fake_capture(root: Path) -> None:
            nonlocal calls
            calls += 1
            _write_fake_capture(root)

        fake_publish = mock.Mock()
        result = publisher.generate_and_maybe_publish(
            verify_only=True,
            capture_builder=fake_capture,
            publisher=fake_publish,
        )
        self.assertEqual(calls, 2)
        self.assertEqual(result["status"], "verified-only")
        self.assertEqual(set(result["archives"]), set(publisher.OUTPUT_NAMES))
        for record in result["archives"].values():
            self.assertEqual(set(record), {"sha256", "bytes"})
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(record["bytes"], 0)
        fake_publish.assert_not_called()

    def test_double_capture_mismatch_never_calls_publisher(self) -> None:
        calls = 0

        def fake_capture(root: Path) -> None:
            nonlocal calls
            calls += 1
            _write_fake_capture(root, marker=calls)

        fake_publish = mock.Mock()
        with self.assertRaisesRegex(AssertionError, "differs at byte"):
            publisher.generate_and_maybe_publish(
                verify_only=False,
                capture_builder=fake_capture,
                publisher=fake_publish,
            )
        fake_publish.assert_not_called()

    def test_atomic_first_publication_and_identical_noop(self) -> None:
        def validator(path: Path) -> None:
            self.assertTrue(path.is_dir())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "golden" / "payne_zero" / "chapter04"
            _write_named_files(source)
            status = publisher.publish_verified_directory(
                source,
                target,
                validator=validator,
            )
            self.assertEqual(status, "published")
            self.assertEqual(
                sorted(path.name for path in target.iterdir()),
                sorted(publisher.OUTPUT_NAMES),
            )
            status = publisher.publish_verified_directory(
                source,
                target,
                validator=validator,
            )
            self.assertEqual(status, "identical-existing")

    def test_existing_different_publication_is_left_untouched(self) -> None:
        def validator(path: Path) -> None:
            self.assertTrue(path.is_dir())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "golden" / "payne_zero" / "chapter04"
            _write_named_files(source, marker=1)
            _write_named_files(target, marker=100)
            before = {
                name: (target / name).read_bytes()
                for name in publisher.OUTPUT_NAMES
            }
            with self.assertRaises(AssertionError):
                publisher.publish_verified_directory(
                    source,
                    target,
                    validator=validator,
                )
            after = {
                name: (target / name).read_bytes()
                for name in publisher.OUTPUT_NAMES
            }
            self.assertEqual(before, after)

    def test_staging_validation_failure_leaves_no_target(self) -> None:
        calls = 0

        def validator(path: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise AssertionError("injected staging validation failure")
            self.assertTrue(path.is_dir())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "golden" / "payne_zero" / "chapter04"
            _write_named_files(source)
            with self.assertRaisesRegex(AssertionError, "injected staging"):
                publisher.publish_verified_directory(
                    source,
                    target,
                    validator=validator,
                )
            self.assertFalse(target.exists())

    def test_final_schema_forbids_catalog_duplication_outside_constants(self) -> None:
        source = Path(publisher.__file__).read_text(encoding="utf-8")
        self.assertIn('name.startswith(("alignment__", "catalog__", "meta__"))', source)
        self.assertIn("meta__constants_archive_sha256", source)
        self.assertIn("os.replace(stage, destination)", source)
        self.assertIn("os.fsync(parent_fd)", source)
        self.assertNotIn("np.savez(", source)
        self.assertNotIn("np.savez_compressed(", source)


if __name__ == "__main__":
    unittest.main()
