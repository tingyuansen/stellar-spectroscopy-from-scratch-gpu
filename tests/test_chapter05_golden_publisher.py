"""Structural and scientific gates for the unpublished Chapter 5 publisher."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import build_chapter05_payne_zero_goldens as publisher
from scripts.deterministic_npz import write_npz


def _write_prepublication_manifest(root: Path) -> Path:
    path = root / "MANIFEST.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "payne_zero_commit": publisher.PINNED_COMMIT,
                "entries": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _publication_record(source: Path, manifest_path: Path) -> dict[str, object]:
    artifacts: dict[str, dict[str, object]] = {}
    for role, specification in (
        publisher._artifact_acceptance_specifications().items()
    ):
        name = specification["name"]
        artifact_path = source / name
        artifacts[role] = {
            "name": name,
            "relative_path": specification["relative_path"],
            "sha256": publisher.sha256(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "archive_kind": specification["archive_kind"],
            "archive_schema_version": specification[
                "archive_schema_version"
            ],
            "key_count": specification["key_count"],
            "schema_digest": specification["schema_digest"],
        }
    return {
        "schema_version": publisher.PUBLICATION_ACCEPTANCE_SCHEMA_VERSION,
        "record_kind": publisher.PUBLICATION_ACCEPTANCE_KIND,
        "publisher": {
            "path": publisher.PUBLISHER_RELATIVE_PATH,
            "sha256": publisher.sha256(publisher.PUBLISHER_PATH),
        },
        "publisher_contract": {
            "path": publisher.PUBLISHER_CONTRACT_RELATIVE_PATH,
            "sha256": publisher.sha256(publisher.PUBLISHER_CONTRACT_PATH),
        },
        "manifest": {
            "path": publisher.MANIFEST_RELATIVE_PATH,
            "prepublication_sha256": publisher.sha256(manifest_path),
            "schema_version": 1,
            "payne_zero_commit": publisher.PINNED_COMMIT,
            "chapter05_golden_paths": [
                publisher.OUTPUT_RELATIVE_PATHS[name]
                for name in publisher.OUTPUT_NAMES
            ],
            "chapter05_entries_present": False,
        },
        "artifacts": artifacts,
    }


def _write_publication_record(
    path: Path,
    record: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _f64(shape: tuple[int, ...], value: float) -> np.ndarray:
    return np.full(shape, value, dtype=np.float64)


def synthetic_raw_capture() -> dict[str, np.ndarray]:
    """Return a compact-valued capture with every assembly-owned key family."""

    raw: dict[str, np.ndarray] = {
        "meta__loaded_pinned_python_source_count": np.asarray(52, dtype=np.int64),
        "meta__post_lane_loaded_manifest_digest": np.asarray(
            publisher.loaded_source_manifest_digest()
        ),
        "meta__capture_scope_complete": np.asarray(True, dtype=np.bool_),
        "meta__pipeline_continuum_fields": np.asarray(
            ["temperature", "mass_density"]
        ),
        "meta__sampled_extension_wavelength_nm": np.linspace(
            100.0, 2500.0, 12
        ),
        "identity__payne_zero_commit": np.asarray(publisher.PINNED_COMMIT),
        "frequency_probe__frequency_hz": np.linspace(1.0, 27.0, 27),
        "frequency_probe__family": np.asarray([f"f{i}" for i in range(9)]),
        "frequency_probe__family_index": np.repeat(
            np.arange(9, dtype=np.int64), 3
        ),
        "frequency_probe__side": np.tile(
            np.asarray([-1, 0, 1], dtype=np.int8), 9
        ),
        "ifop__frequency_hz": np.linspace(1.0, 27.0, 27),
        "ifop__ifop4_only_scattering": _f64((6, 27), 1.0),
        "ifop__ifop13_only_scattering": _f64((6, 27), 0.0),
        "ifop__ifop4_and_13_scattering": _f64((6, 27), 1.2),
        "ifop__h2_rayleigh_increment": _f64((6, 27), 0.2),
        "ifop__ifop13_only_is_zero": np.asarray(True, dtype=np.bool_),
        "ifop19__frequency_hz": np.linspace(1.0, 27.0, 27),
        "ifop19__scattering": _f64((6, 27), 0.0),
        "molecular_entry__frequency_hz": np.linspace(1.0, 27.0, 27),
        "molecular_entry__all_warm_absorption": _f64((6, 27), 0.0),
        "molecular_entry__molecule_disabled_ch_absorption": _f64(
            (6, 27), 0.0
        ),
        "molecular_entry__molecule_disabled_oh_absorption": _f64(
            (6, 27), 0.0
        ),
        "molecular_entry__mixed_entry_active": np.asarray(
            True, dtype=np.bool_
        ),
        "sampling_boundary__count_263_present": np.asarray(
            True, dtype=np.bool_
        ),
        "sampling_boundary__count_299_present": np.asarray(
            True, dtype=np.bool_
        ),
    }
    molecular_values: dict[str, np.ndarray] = {
        "ch_oh_frequency_hz": np.linspace(1.0, 15.0, 15),
        "ch_oh_frequency_family_index": np.arange(15, dtype=np.int64),
        "ch_oh_temperature_k": np.asarray([8999.0, 9000.0]),
        "ch_cross_section_times_partition": _f64((2, 15), 1.0e-18),
        "oh_cross_section_times_partition": _f64((2, 15), 2.0e-18),
        "cia_wavenumber_cm_inverse": np.asarray(
            [5000.0, 19999.0, 20000.0, 20001.0]
        ),
        "cia_frequency_hz": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "cia_absorption": _f64((3, 4), 3.0e-4),
        "cia_temperature_fraction": np.asarray([0.0, 0.5, 0.0]),
        "cia_lower_column_weight": np.asarray([0.0, 0.5, 0.0]),
        "cia_upper_column_weight": np.asarray([1.0, 0.5, 1.0]),
        "cia_active_frequency_mask": np.asarray(
            [True, True, True, False], dtype=np.bool_
        ),
    }
    raw.update(
        {
            f"molecular_boundary__{name}": value
            for name, value in molecular_values.items()
        }
    )
    raw["molecular_boundary__integration_only_probe"] = np.asarray([7.0])

    grid_rows = [
        np.geomspace(10.0 + index, 1000.0 + index, 30000).astype(np.float64)
        for index in range(5)
    ]
    weight_rows = [
        np.linspace(1.0 + index, 2.0 + index, 30000, dtype=np.float64)
        for index in range(5)
    ]
    boundary_selector = np.asarray([0, 1, 1, 2, 2, 3, 3, 4])
    raw["sampling_boundary__wavelength_nm"] = np.stack(
        [grid_rows[index] for index in boundary_selector]
    )
    raw["sampling_boundary__frequency_weight_hz"] = np.stack(
        [weight_rows[index] for index in boundary_selector]
    )
    raw["sampling_boundary__effective_temperature_k"] = np.asarray(
        [4499.0, 4500.0, 7249.0, 7250.0, 12999.0, 13000.0, 29999.0, 30000.0]
    )
    raw["sampling_boundary__active_line_reference_count"] = np.asarray(
        [226, 240, 240, 263, 263, 299, 299, 338], dtype=np.int64
    )

    regime_grid = {
        "hot_dwarf": 4,
        "solar_dwarf": 1,
        "low_gravity_giant": 1,
        "cool_molecule_rich": 0,
    }
    active_counts = {
        "hot_dwarf": 338,
        "solar_dwarf": 240,
        "low_gravity_giant": 240,
        "cool_molecule_rich": 226,
    }
    edge_geometry = {
        "probe_interval_index": np.asarray([1, 2], dtype=np.int64),
        "requested_wavelength_nm": np.linspace(100.0, 200.0, 36),
        "requested_edge_index": np.arange(36, dtype=np.int64) % 12,
        "used_edge_index": np.arange(12, dtype=np.int64),
        "all_sample_frequency_hz": np.linspace(1.0, 1020.0, 1020),
        "packaged_sample_frequency_hz": np.linspace(1.0, 1020.0, 1020),
        "packaged_sample_frequency_bit_equal": np.asarray(
            True, dtype=np.bool_
        ),
        "selected_sample_index": np.arange(36, dtype=np.int64),
        "selected_sample_frequency_hz": np.linspace(1.0, 36.0, 36),
        "selected_sample_edge_index": np.repeat(np.arange(12), 3),
        "selected_sample_side": np.tile(np.asarray([-1, 0, 1]), 12),
        "left_basis": _f64((36,), 0.25),
        "center_basis": _f64((36,), 0.5),
        "right_basis": _f64((36,), 0.25),
        "basis_sum": _f64((36,), 1.0),
        "signed_edge_frequency_hz": np.linspace(-1.0, -341.0, 341),
        "edge_wavelength_nm": np.linspace(10.0, 351.0, 341),
        "edge_midpoint_wavelength_nm": np.linspace(10.5, 350.5, 340),
        "edge_interval_width_squared_over_two_nm2": _f64((340,), 0.5),
    }

    for regime_index, regime in enumerate(publisher.REGIME_NAMES):
        value = float(regime_index + 1)
        prefix = f"{regime}__atmosphere__"
        raw[f"{prefix}frequency_hz"] = raw[
            "frequency_probe__frequency_hz"
        ].copy()
        raw[f"{prefix}runner_opacity_flags"] = np.ones(20, dtype=np.int64)
        raw[f"{prefix}absorption"] = _f64((6, 27), value)
        raw[f"{prefix}scattering"] = _f64((6, 27), value / 10.0)
        raw[f"{prefix}source"] = _f64((6, 27), value * 100.0)
        for component_index, component in enumerate(
            publisher.ATMOSPHERE_ABSORPTION_COMPONENTS
        ):
            raw[
                f"{prefix}component__{component}__absorption"
            ] = _f64((6, 27), value + component_index)
            raw[f"{prefix}component__{component}__source"] = _f64(
                (6, 27), value * 100.0 + component_index
            )
        for component_index, component in enumerate(
            publisher.ATMOSPHERE_SCATTERING_COMPONENTS
        ):
            raw[f"{prefix}component__{component}__scattering"] = _f64(
                (6, 27), value / 10.0 + component_index
            )
        for component_index, component in enumerate(
            publisher.ATMOSPHERE_MOLECULAR_COMPONENTS
        ):
            suffix = {
                "ch": "ch_absorption",
                "oh": "oh_absorption",
                "h2_cia": "h2_cia_absorption",
            }[component]
            raw[f"{prefix}molecular_component__{suffix}"] = _f64(
                (6, 27), value + component_index
            )
        for suffix in (
            "component__ordered_absorption_sum",
            "component__ordered_scattering_sum",
            "component__ordered_source_numerator_sum",
            "component__absorption_residual",
            "component__scattering_residual",
            "component__source_residual",
            "molecular_component__ordered_absorption_sum",
        ):
            raw[f"{prefix}{suffix}"] = _f64((6, 27), value)

        count = active_counts[regime]
        raw[f"{prefix}line_reference_wavelength_nm"] = np.linspace(
            100.0, 443.0, 344
        )
        packed = np.arange(344, dtype=np.int64)
        packed[-1] = 2**30
        raw[f"{prefix}line_reference_packed_wavelength_index"] = packed
        threshold = _f64((6, 344), value).astype(np.float32)
        threshold[:, -1] = threshold[:, -2]
        raw[f"{prefix}line_reference_threshold"] = threshold
        raw[f"{prefix}line_reference_active_index"] = np.arange(
            count, dtype=np.int64
        )
        raw[f"{prefix}line_reference_active_frequency_hz"] = np.linspace(
            1.0, float(count), count
        )
        for quantity in ("absorption", "scattering", "source"):
            raw[f"{prefix}line_reference_active_{quantity}"] = _f64(
                (6, count), value
            )

        grid_index = regime_grid[regime]
        wavelength = grid_rows[grid_index]
        raw[f"{prefix}sampling_wavelength_nm"] = wavelength.copy()
        raw[f"{prefix}sampling_frequency_weight_hz"] = weight_rows[
            grid_index
        ].copy()
        raw[f"{prefix}product_frequency_hz"] = (
            publisher.LIGHT_SPEED_NM_PER_S
            / np.maximum(wavelength, 1.0e-300)
        )
        for quantity in ("absorption", "scattering", "source"):
            raw[f"{prefix}product_{quantity}"] = _f64(
                (6, 30000), value
            )

        synthesis_prefix = f"{regime}__synthesis__"
        for suffix, geometry in edge_geometry.items():
            raw[f"{synthesis_prefix}route__{suffix}"] = geometry.copy()
        for suffix in (
            "absorption",
            "scattering",
            "sample_absorption",
            "sample_scattering",
            "reconstructed_absorption",
            "reconstructed_scattering",
            "interpolation_absorption_residual",
            "interpolation_scattering_residual",
        ):
            raw[f"{synthesis_prefix}standard__{suffix}"] = _f64(
                (6, 36), value
            )
        raw[
            f"{synthesis_prefix}standard__coulomb_table_energy_first"
        ] = np.asarray(False, dtype=np.bool_)
        raw[
            f"{synthesis_prefix}standard__frequency_invariants_supplied"
        ] = np.asarray(False, dtype=np.bool_)
        raw[
            f"{synthesis_prefix}standard__trace__called_frequency_hz"
        ] = edge_geometry["selected_sample_frequency_hz"].copy()
        raw[
            f"{synthesis_prefix}standard__trace__compute_at_freqs_call_count"
        ] = np.asarray(1, dtype=np.int64)
        raw[
            f"{synthesis_prefix}standard__trace__unused_edge_mask"
        ] = np.ones(340, dtype=np.bool_)
        raw[
            f"{synthesis_prefix}standard__trace__unused_sample_call_count"
        ] = np.asarray(0, dtype=np.int64)
        raw[
            f"{synthesis_prefix}standard__trace__unused_sample_index"
        ] = np.arange(36, 1020, dtype=np.int64)
        raw[
            f"{synthesis_prefix}standard__trace__unused_sample_was_called"
        ] = np.zeros(984, dtype=np.bool_)
        for lane, width in (("standard", 36), ("diagnostic", 27)):
            for component_index, component in enumerate(
                publisher.SYNTHESIS_COMPONENT_NAMES
            ):
                raw[
                    f"{synthesis_prefix}{lane}__component__{component}"
                ] = _f64((6, width), value + component_index)
            for suffix in (
                "ordered_absorption_sum",
                "ordered_scattering_sum",
                "absorption_residual",
                "scattering_residual",
            ):
                raw[
                    f"{synthesis_prefix}{lane}__component__{suffix}"
                ] = _f64((6, width), value)
        for case_index, case in enumerate(publisher.SYNTHESIS_MINOR_CASES):
            for quantity in ("absorption", "scattering"):
                raw[
                    f"{synthesis_prefix}standard__isolated_minor__"
                    f"{case}__{quantity}"
                ] = _f64((6, 36), value + case_index)
        for suffix in (
            "ordered_absorption_sum",
            "ordered_scattering_sum",
            "absorption_residual",
            "scattering_residual",
        ):
            raw[
                f"{synthesis_prefix}standard__isolated_minor__{suffix}"
            ] = _f64((6, 36), value)

        raw[f"{synthesis_prefix}diagnostic__frequency_hz"] = np.linspace(
            1.0, 27.0, 27
        )
        for lane, width in (("diagnostic", 27), ("extension", 12)):
            for quantity in ("absorption", "scattering", "source"):
                raw[f"{synthesis_prefix}{lane}__{quantity}"] = _f64(
                    (6, width), value
                )
            raw[
                f"{synthesis_prefix}{lane}__coulomb_table_energy_first"
            ] = np.asarray(True, dtype=np.bool_)
            raw[
                f"{synthesis_prefix}{lane}__frequency_invariants_supplied"
            ] = np.asarray(lane == "extension", dtype=np.bool_)
        raw[f"{synthesis_prefix}extension__frequency_hz"] = np.linspace(
            1.0, 12.0, 12
        )
        raw[
            f"{synthesis_prefix}extension__support_wavelength_min_nm"
        ] = np.asarray(100.0)
        raw[
            f"{synthesis_prefix}extension__support_wavelength_max_nm"
        ] = np.asarray(2500.0)
        raw[f"{synthesis_prefix}extension__wavelength_nm"] = np.linspace(
            100.0, 2500.0, 12
        )
        raw[
            f"{synthesis_prefix}extension__continuum_atmosphere_field_names"
        ] = np.asarray(["temperature", "mass_density"])
        raw[
            f"{synthesis_prefix}extension__pops_field_names"
        ] = np.asarray(["temperature", "electron_density"])
        raw[
            f"{synthesis_prefix}extension__invariant_field_names"
        ] = np.asarray(publisher.EXTENSION_INVARIANT_FIELDS)
        state_shapes = {
            "temperature": (6,),
            "mass_density": (6,),
            "electron_density": (6,),
            "hydrogen_partition_normalized_ion_stage_populations": (6, 2),
            "hydrogen_ionized_population": (6,),
            "helium_neutral_partition_normalized_population": (6,),
            "helium_singly_ionized_partition_normalized_population": (6,),
            "hot_metal_populations": (6, 21),
            "charge_square_population_sum": (6, 5),
        }
        for field, shape in state_shapes.items():
            raw[
                f"{synthesis_prefix}extension__input__pops__{field}"
            ] = _f64(shape, value)
        raw[
            f"{synthesis_prefix}extension__input__pops__integration_only"
        ] = _f64((6,), value)
        raw[
            f"{synthesis_prefix}extension__input__invariant__frequencies_hz"
        ] = np.linspace(1.0, 12.0, 12)
        raw[
            f"{synthesis_prefix}extension__input__invariant__"
            "coulomb_table_energy_first"
        ] = np.asarray(True, dtype=np.bool_)
        for field_index, field in enumerate(
            publisher.EXTENSION_INVARIANT_FIELDS[2:], start=2
        ):
            raw[
                f"{synthesis_prefix}extension__input__invariant__{field}"
            ] = np.asarray(
                [field_index, field_index + 0.5], dtype=np.float64
            )
        for suffix, flag in (
            ("rich_hii_matches_trimmed", False),
            ("schema_h2_bit_invariant", True),
            ("signed_edge_bit_invariant", True),
        ):
            raw[
                f"{synthesis_prefix}counterfactual__{suffix}"
            ] = np.asarray(flag, dtype=np.bool_)
        raw[
            f"{synthesis_prefix}counterfactual__integration_only"
        ] = _f64((6,), value)
        raw[
            f"{synthesis_prefix}counterfactual__"
            "rich_minus_trimmed_hii_absorption"
        ] = raw[
            f"{synthesis_prefix}standard__component__absorption_residual"
        ].copy()
        raw[
            f"{synthesis_prefix}counterfactual__signed_edge_original"
        ] = edge_geometry["signed_edge_frequency_hz"].copy()
        raw[
            f"{synthesis_prefix}counterfactual__signed_edge_flipped"
        ] = -edge_geometry["signed_edge_frequency_hz"]
        for suffix, standard_quantity in (
            publisher.COUNTERFACTUAL_STANDARD_OUTPUT_ALIASES
        ):
            raw[
                f"{synthesis_prefix}counterfactual__{suffix}"
            ] = raw[
                f"{synthesis_prefix}standard__{standard_quantity}"
            ].copy()
        raw[f"{synthesis_prefix}activation__active"] = np.ones(
            10, dtype=np.bool_
        )
    raw["sampling_boundary__reference_wavelength_nm"] = raw[
        "hot_dwarf__atmosphere__line_reference_wavelength_nm"
    ].copy()
    raw["sampling_boundary__reference_packed_wavelength_index"] = raw[
        "hot_dwarf__atmosphere__line_reference_packed_wavelength_index"
    ].copy()
    while len(raw) < publisher.ACCEPTED_RAW_KEY_COUNT:
        index = len(raw)
        raw[f"synthetic_extra__{index:04d}"] = np.asarray(
            index, dtype=np.int64
        )
    return raw


class Chapter05GoldenPublisherTests(unittest.TestCase):
    def test_reviewed_identity_tuple_and_output_names_are_frozen(self) -> None:
        self.assertEqual(publisher.ACCEPTED_RAW_SCHEMA_VERSION, 2)
        self.assertEqual(publisher.ACCEPTED_RAW_KEY_COUNT, 1161)
        self.assertEqual(
            publisher.ACCEPTED_RAW_SCHEMA_DIGEST,
            "652c110dc79a6f6dfca6893bee35416289675b4920a5d0dcfe6b2cb262dacf3d",
        )
        self.assertEqual(
            publisher.ACCEPTED_PHYSICAL_FINGERPRINT,
            "d223351fa2c51dc24a1b01896da9ab9a82fc475f4082c47fde34734d8dc03343",
        )
        self.assertEqual(
            publisher.ACCEPTED_FULL_FINGERPRINT,
            "3d2c131711e1c0dc6aa088892193bb24d41a76d005bc20dd1c42d3e84f66e656",
        )
        self.assertEqual(
            publisher.ACCEPTED_INPUT_SHA256["oracle_acceptance"],
            "892279cd2dfa850c3eabfb8ce94953e4723db5270da319bcedb9bc90c9597474",
        )
        self.assertEqual(publisher.ACCEPTED_READER_KEY_COUNT, 257)
        self.assertEqual(
            publisher.ACCEPTED_READER_SCHEMA_DIGEST,
            "058389fcd0e944dd4c1ad1208adbaac44d53ab35e61e5366d37a8a141ad91f88",
        )
        self.assertEqual(publisher.ACCEPTED_INTEGRATION_KEY_COUNT, 1079)
        self.assertEqual(
            publisher.ACCEPTED_INTEGRATION_SCHEMA_DIGEST,
            "e09a8932d97f8a2756aca2b779c4b8fdd822ca239197fd908ee575d09df081ca",
        )
        self.assertEqual(
            publisher.ACCEPTED_INVENTORY_MAPPING_DIGEST,
            "b02a6a2896d3468d1052441f8def0841b608eb5d5816ccd72539a0ec982c452c",
        )
        self.assertEqual(len(publisher.OUTPUT_NAMES), 2)
        self.assertNotIn("raw", " ".join(publisher.OUTPUT_NAMES))

    def test_static_identity_and_one_byte_mutation_fail_closed(self) -> None:
        summary = publisher.verify_static_identity()
        self.assertEqual(summary["loaded_python_source_count"], 52)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.bin"
            path.write_bytes(b"accepted")
            expected = hashlib.sha256(b"accepted").hexdigest()
            publisher._verify_file_hashes({path: expected}, label="test")
            path.write_bytes(b"mutated")
            with self.assertRaises(publisher.PublisherIdentityError):
                publisher._verify_file_hashes({path: expected}, label="test")

    def test_fixture_schema_and_payload_are_checked_before_capture(self) -> None:
        fixture = publisher.oracle_worker.load_fixture(publisher.FIXTURE_PATH)
        publisher.validate_fixture_payload(fixture)
        changed = {name: value.copy() for name, value in fixture.items()}
        changed["solar_dwarf__atmosphere__temperature"][0] += 1.0
        with self.assertRaisesRegex(
            publisher.PublisherIdentityError, "payload digest"
        ):
            publisher.validate_fixture_payload(changed)
        changed = {name: value.copy() for name, value in fixture.items()}
        changed["unexpected"] = np.asarray(1)
        with self.assertRaisesRegex(
            publisher.PublisherIdentityError, "schema"
        ):
            publisher.validate_fixture_payload(changed)

    def test_capture_fingerprint_is_mapping_order_independent(self) -> None:
        first = {
            "z": np.asarray([1.0]),
            "a": np.asarray([2], dtype=np.int64),
        }
        second = dict(reversed(list(first.items())))
        self.assertEqual(
            publisher.capture_fingerprint(first, physical_payload_only=False),
            publisher.capture_fingerprint(second, physical_payload_only=False),
        )
        second["a"] = np.asarray([3], dtype=np.int64)
        self.assertNotEqual(
            publisher.capture_fingerprint(first, physical_payload_only=False),
            publisher.capture_fingerprint(second, physical_payload_only=False),
        )

    def test_raw_validator_rejects_incomplete_or_object_capture(self) -> None:
        with self.assertRaises(publisher.PublisherSchemaError):
            publisher.validate_raw_capture({"only": np.asarray(1)})
        with self.assertRaises(TypeError):
            publisher.validate_raw_capture(
                {"bad": np.asarray([object()], dtype=object)}
            )

    def test_reader_integration_projection_and_grid_deduplication(self) -> None:
        raw = synthetic_raw_capture()
        reader, integration = publisher.assemble_payloads(
            raw, validate_raw=False
        )
        self.assertFalse(
            any(30000 in np.asarray(value).shape for value in reader.values())
        )
        self.assertEqual(
            integration["atmosphere_product__absorption"].shape,
            (4, 6, 30000),
        )
        self.assertEqual(integration["grid_bank__wavelength_nm"].shape, (5, 30000))
        self.assertEqual(
            integration["grid_bank__frequency_weight_hz"].shape, (5, 30000)
        )
        self.assertEqual(
            tuple(integration["grid_bank__policy_label"].tolist()),
            publisher.GRID_POLICY_LABELS,
        )
        boundary_selector = integration[
            "sampling_boundary__grid_bank_index"
        ]
        np.testing.assert_array_equal(
            integration["grid_bank__wavelength_nm"][boundary_selector],
            raw["sampling_boundary__wavelength_nm"],
        )
        for regime_index, regime in enumerate(publisher.REGIME_NAMES):
            selector = integration["atmosphere_product__grid_bank_index"][
                regime_index
            ]
            np.testing.assert_array_equal(
                integration["grid_bank__wavelength_nm"][selector],
                raw[f"{regime}__atmosphere__sampling_wavelength_nm"],
            )
        self.assertNotIn(
            "evidence__sampling_boundary__wavelength_nm", integration
        )
        self.assertNotIn(
            "evidence__hot_dwarf__atmosphere__product_frequency_hz",
            integration,
        )
        self.assertNotIn(
            "oracle__meta__sampled_extension_wavelength_nm", reader
        )
        self.assertNotIn(
            "oracle__meta__sampled_extension_wavelength_nm", integration
        )
        self.assertIn(
            "alias__meta__sampled_extension_wavelength_nm__reader_member",
            integration,
        )
        self.assertEqual(
            set(integration["inventory__raw_member_name"].tolist()), set(raw)
        )
        self.assertTrue(
            bool(integration["meta__logical_raw_capture_coverage_complete"])
        )
        reconstructed = publisher.reconstruct_logical_raw_capture(
            reader, integration
        )
        self.assertEqual(set(reconstructed), set(raw))
        for name in raw:
            self.assertEqual(reconstructed[name].dtype, raw[name].dtype, name)
            self.assertEqual(reconstructed[name].shape, raw[name].shape, name)
            np.testing.assert_array_equal(reconstructed[name], raw[name])

    def test_scientific_deduplication_routes_are_explicit_and_physical(
        self,
    ) -> None:
        raw = synthetic_raw_capture()
        reader, integration = publisher.assemble_payloads(
            raw, validate_raw=False
        )
        routes = {
            str(name): (str(disposition), str(member))
            for name, disposition, member in zip(
                integration["inventory__raw_member_name"].tolist(),
                integration["inventory__disposition"].tolist(),
                integration["inventory__published_member"].tolist(),
            )
        }

        axis_routes = {
            "sampling_boundary__reference_wavelength_nm": (
                "line_reference__wavelength_nm"
            ),
            "sampling_boundary__reference_packed_wavelength_index": (
                "line_reference__packed_wavelength_index"
            ),
            "ifop19__frequency_hz": "axis__diagnostic__frequency_hz",
            "molecular_entry__frequency_hz": "axis__diagnostic__frequency_hz",
        }
        for raw_name, member in axis_routes.items():
            self.assertEqual(routes[raw_name], ("reader_alias", member))
            self.assertNotIn(f"evidence__{raw_name}", integration)
            alias = f"alias__{raw_name}__reader_member"
            self.assertEqual(str(integration[alias].item()), member)

        for regime_index, regime in enumerate(publisher.REGIME_NAMES):
            prefix = f"{regime}__synthesis__"
            reader_aliases = {
                (
                    f"{prefix}extension__input__invariant__frequencies_hz"
                ): "synthesis__extension__frequency_hz",
                (
                    f"{prefix}extension__input__invariant__"
                    "coulomb_table_energy_first"
                ): (
                    "synthesis__extension__coulomb_table_energy_first"
                    f"[{regime_index}]"
                ),
                (
                    f"{prefix}standard__trace__called_frequency_hz"
                ): "synthesis__edge__selected_sample_frequency_hz",
                (
                    f"{prefix}counterfactual__signed_edge_original"
                ): "synthesis__edge__signed_edge_frequency_hz",
            }
            for suffix, standard_quantity in (
                publisher.COUNTERFACTUAL_STANDARD_OUTPUT_ALIASES
            ):
                reader_aliases[
                    f"{prefix}counterfactual__{suffix}"
                ] = (
                    f"synthesis__standard__{standard_quantity}"
                    f"[{regime_index}]"
                )
            for raw_name, member in reader_aliases.items():
                self.assertEqual(routes[raw_name], ("reader_alias", member))
                self.assertNotIn(f"evidence__{raw_name}", integration)

        for field in publisher.EXTENSION_INVARIANT_FIELDS[2:]:
            member = (
                "evidence__synthesis__extension__input__invariant__" + field
            )
            self.assertIn(member, integration)
            for regime in publisher.REGIME_NAMES:
                raw_name = (
                    f"{regime}__synthesis__extension__input__invariant__"
                    f"{field}"
                )
                self.assertEqual(routes[raw_name], ("integration", member))
                self.assertNotIn(f"evidence__{raw_name}", integration)

        signed_member = (
            "evidence__synthesis__counterfactual__signed_edge_flipped"
        )
        for regime in publisher.REGIME_NAMES:
            raw_name = (
                f"{regime}__synthesis__counterfactual__signed_edge_flipped"
            )
            self.assertEqual(
                routes[raw_name], ("integration", signed_member)
            )
        for field in publisher.STANDARD_TRACE_FIELDS[1:]:
            member = f"evidence__synthesis__standard__trace__{field}"
            for regime in publisher.REGIME_NAMES:
                raw_name = (
                    f"{regime}__synthesis__standard__trace__{field}"
                )
                self.assertEqual(routes[raw_name], ("integration", member))

        common = "oracle__meta__loaded_pinned_python_source_count"
        self.assertTrue(publisher._bitwise_equal(reader[common], integration[common]))

        # Equal numbers do not imply equal meaning.  These arrays deliberately
        # remain separate because activation evidence is regime-specific and
        # the rich-minus-trimmed result is not a standard-route residual.
        activation_members = [
            f"evidence__{regime}__synthesis__activation__active"
            for regime in publisher.REGIME_NAMES
        ]
        self.assertTrue(
            all(
                publisher._bitwise_equal(
                    integration[activation_members[0]], integration[member]
                )
                for member in activation_members[1:]
            )
        )
        rich_difference = (
            "evidence__hot_dwarf__synthesis__counterfactual__"
            "rich_minus_trimmed_hii_absorption"
        )
        reader_residual = (
            reader["synthesis__standard__component__absorption_residual"][0]
        )
        self.assertTrue(
            publisher._bitwise_equal(
                integration[rich_difference], reader_residual
            )
        )
        self.assertEqual(
            routes[
                "hot_dwarf__synthesis__counterfactual__"
                "rich_minus_trimmed_hii_absorption"
            ][0],
            "integration",
        )
        zero_members = [
            "evidence__ifop19__scattering",
            "evidence__molecular_entry__all_warm_absorption",
            "evidence__molecular_entry__molecule_disabled_ch_absorption",
            "evidence__molecular_entry__molecule_disabled_oh_absorption",
        ]
        self.assertTrue(
            all(
                publisher._bitwise_equal(
                    integration[zero_members[0]], integration[member]
                )
                for member in zero_members[1:]
            )
        )
        self.assertTrue(
            publisher._bitwise_equal(
                integration[zero_members[0]],
                reader["seam__ifop__ifop13_only_scattering"],
            )
        )
        self.assertTrue(np.all(integration[zero_members[0]] == 0.0))
        self.assertTrue(all(member in integration for member in zero_members))
        scalar_members = [
            "evidence__molecular_entry__mixed_entry_active",
            "evidence__sampling_boundary__count_263_present",
            "evidence__sampling_boundary__count_299_present",
        ]
        self.assertTrue(
            all(bool(integration[member].item()) for member in scalar_members)
        )
        self.assertTrue(all(member in integration for member in scalar_members))

    def test_declared_deduplication_fails_closed_on_one_bitwise_drift(
        self,
    ) -> None:
        raw = synthetic_raw_capture()
        raw[
            "solar_dwarf__synthesis__extension__input__invariant__"
            "natural_log_frequency"
        ][0] += 1.0
        with self.assertRaisesRegex(
            publisher.PublisherSchemaError, "regime-invariant"
        ):
            publisher.assemble_payloads(raw, validate_raw=False)

        raw = synthetic_raw_capture()
        raw[
            "solar_dwarf__synthesis__counterfactual__rich_hii_absorption"
        ][0, 0] += 1.0
        with self.assertRaisesRegex(
            publisher.PublisherSchemaError, "bit-identical"
        ):
            publisher.assemble_payloads(raw, validate_raw=False)

        raw = synthetic_raw_capture()
        raw["sampling_boundary__reference_wavelength_nm"][0] += 1.0
        with self.assertRaisesRegex(
            publisher.PublisherSchemaError, "bit-identical"
        ):
            publisher.assemble_payloads(raw, validate_raw=False)

    def test_component_state_threshold_and_edge_banks_are_interpretable(self) -> None:
        raw = synthetic_raw_capture()
        reader, integration = publisher.assemble_payloads(
            raw, validate_raw=False
        )
        self.assertEqual(
            reader["atmosphere__component__absorption"].shape,
            (4, 14, 6, 27),
        )
        self.assertEqual(
            reader["atmosphere__component__scattering"].shape,
            (4, 4, 6, 27),
        )
        self.assertEqual(
            reader["synthesis__standard__component"].shape,
            (4, 10, 6, 36),
        )
        self.assertEqual(
            reader["synthesis__standard__isolated_minor__absorption"].shape,
            (4, 7, 6, 36),
        )
        self.assertEqual(reader["line_reference__threshold"].dtype, np.float32)
        np.testing.assert_array_equal(
            reader["line_reference__threshold"][:, :, -1],
            reader["line_reference__threshold"][:, :, -2],
        )
        self.assertEqual(
            int(reader["line_reference__packed_wavelength_index"][-1]),
            2**30,
        )
        valid = reader["line_reference__active_valid"]
        self.assertTrue(
            np.all(reader["line_reference__active_index"][~valid] == 0)
        )
        self.assertTrue(
            np.all(
                reader["line_reference__active_frequency_hz"][~valid] == 0.0
            )
        )
        self.assertEqual(reader["reader_state__hot_metal_populations"].shape, (4, 6, 21))
        self.assertIn("synthesis__edge__basis_sum", reader)
        alias = (
            "alias__hot_dwarf__atmosphere__absorption__reader_member"
        )
        self.assertIn(alias, integration)

    def test_final_pair_round_trip_and_reader_hash_binding(self) -> None:
        raw = synthetic_raw_capture()
        reader, integration = publisher.assemble_payloads(
            raw, validate_raw=False
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reader_path = directory / publisher.READER_NAME
            write_npz(reader_path, reader)
            integration["meta__reader_archive_sha256"] = np.asarray(
                publisher.sha256(reader_path)
            )
            write_npz(directory / publisher.INTEGRATION_NAME, integration)
            with (
                mock.patch.object(
                    publisher, "ACCEPTED_READER_KEY_COUNT", len(reader)
                ),
                mock.patch.object(
                    publisher,
                    "ACCEPTED_READER_SCHEMA_DIGEST",
                    publisher.schema_digest(reader),
                ),
                mock.patch.object(
                    publisher,
                    "ACCEPTED_INTEGRATION_KEY_COUNT",
                    len(integration),
                ),
                mock.patch.object(
                    publisher,
                    "ACCEPTED_INTEGRATION_SCHEMA_DIGEST",
                    publisher.schema_digest(integration),
                ),
                mock.patch.object(
                    publisher,
                    "ACCEPTED_INVENTORY_MAPPING_DIGEST",
                    publisher.inventory_mapping_digest(integration),
                ),
            ):
                publisher.validate_final_directory(
                    directory, validate_logical_raw=False
                )
            loaded = publisher.load_npz(directory / publisher.INTEGRATION_NAME)
            loaded["meta__reader_archive_sha256"] = np.asarray("0" * 64)
            with self.assertRaises(publisher.PublisherSchemaError):
                publisher.validate_final_payload(
                    publisher.INTEGRATION_NAME,
                    loaded,
                    reader_sha256=publisher.sha256(reader_path),
                )

    def test_final_only_semantic_mutations_fail_closed(self) -> None:
        raw = synthetic_raw_capture()
        reader, integration = publisher.assemble_payloads(
            raw, validate_raw=False
        )
        reader_hash = "r" * 64
        integration["meta__reader_archive_sha256"] = np.asarray(reader_hash)
        reader_count = len(reader)
        reader_schema = publisher.schema_digest(reader)
        integration_count = len(integration)
        integration_schema = publisher.schema_digest(integration)
        patches = (
            mock.patch.object(
                publisher, "ACCEPTED_READER_KEY_COUNT", reader_count
            ),
            mock.patch.object(
                publisher, "ACCEPTED_READER_SCHEMA_DIGEST", reader_schema
            ),
            mock.patch.object(
                publisher,
                "ACCEPTED_INTEGRATION_KEY_COUNT",
                integration_count,
            ),
            mock.patch.object(
                publisher,
                "ACCEPTED_INTEGRATION_SCHEMA_DIGEST",
                integration_schema,
            ),
            mock.patch.object(
                publisher,
                "ACCEPTED_INVENTORY_MAPPING_DIGEST",
                publisher.inventory_mapping_digest(integration),
            ),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            publisher.validate_final_payload(publisher.READER_NAME, reader)
            publisher.validate_final_payload(
                publisher.INTEGRATION_NAME,
                integration,
                reader_sha256=reader_hash,
            )
            mutated_reader = {
                name: value.copy() for name, value in reader.items()
            }
            mutated_reader["meta__cpu_only"] = np.asarray(
                False, dtype=np.bool_
            )
            with self.assertRaises(publisher.PublisherSchemaError):
                publisher.validate_final_payload(
                    publisher.READER_NAME, mutated_reader
                )
            mutated_reader = {
                name: value.copy() for name, value in reader.items()
            }
            mutated_reader[
                "atmosphere__component__absorption_name"
            ][0] = "wrong"
            with self.assertRaises(publisher.PublisherSchemaError):
                publisher.validate_final_payload(
                    publisher.READER_NAME, mutated_reader
                )
            mutated_integration = {
                name: value.copy() for name, value in integration.items()
            }
            mutated_integration["grid_bank__light_speed_nm_per_s"] = np.asarray(
                1.0
            )
            with self.assertRaises(publisher.PublisherSchemaError):
                publisher.validate_final_payload(
                    publisher.INTEGRATION_NAME,
                    mutated_integration,
                    reader_sha256=reader_hash,
                )
            mutated_integration = {
                name: value.copy() for name, value in integration.items()
            }
            mutated_integration[
                "meta__reader_archive_schema_digest"
            ] = np.asarray("0" * 64)
            with self.assertRaises(publisher.PublisherSchemaError):
                publisher.validate_final_payload(
                    publisher.INTEGRATION_NAME,
                    mutated_integration,
                    reader_sha256=reader_hash,
                )
            mutated_integration = {
                name: value.copy() for name, value in integration.items()
            }
            alias = next(
                name
                for name in mutated_integration
                if name.startswith("alias__")
            )
            mutated_integration[alias] = np.asarray("wrong")
            with self.assertRaises(publisher.PublisherSchemaError):
                publisher.reconstruct_logical_raw_capture(
                    reader, mutated_integration
                )
            for raw_name, replacement in (
                (
                    "cool_molecule_rich__atmosphere__absorption",
                    "reader_alias",
                ),
                ("identity__payne_zero_commit", "integration"),
            ):
                mutated_integration = {
                    name: value.copy() for name, value in integration.items()
                }
                names = mutated_integration[
                    "inventory__raw_member_name"
                ].tolist()
                index = names.index(raw_name)
                mutated_integration["inventory__disposition"][
                    index
                ] = replacement
                with self.assertRaises(publisher.PublisherSchemaError):
                    publisher.validate_final_payload(
                        publisher.INTEGRATION_NAME,
                        mutated_integration,
                        reader_sha256=reader_hash,
                    )

    def test_fresh_cache_policy_rejects_empty_populated_and_symlink_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            absent = root / "absent"
            publisher._require_fresh_cache_target(absent)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(RuntimeError, "absent"):
                publisher._require_fresh_cache_target(empty)
            populated = root / "populated"
            populated.mkdir()
            (populated / "entry").write_text("x")
            with self.assertRaisesRegex(RuntimeError, "populated"):
                publisher._require_fresh_cache_target(populated)
            target = root / "target"
            target.mkdir()
            symlink = root / "cache-link"
            symlink.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                publisher._require_fresh_cache_target(symlink)

    def test_build_capture_set_uses_absent_cache_and_ordered_assembly(self) -> None:
        calls: list[tuple[str, Path]] = []

        def runner(raw_path: Path, cache_path: Path) -> None:
            self.assertFalse(cache_path.exists())
            raw_path.write_bytes(b"raw")
            calls.append(("capture", cache_path))

        def assembler(raw_path: Path, final_dir: Path) -> None:
            self.assertEqual(raw_path.read_bytes(), b"raw")
            final_dir.mkdir()
            for name in publisher.OUTPUT_NAMES:
                (final_dir / name).write_bytes(name.encode())
            calls.append(("assemble", raw_path))

        with tempfile.TemporaryDirectory() as temporary:
            publisher.build_capture_set(
                Path(temporary),
                capture_runner=runner,
                assembler=assembler,
            )
        self.assertEqual([call[0] for call in calls], ["capture", "assemble"])

    def test_child_environment_pins_all_controls_and_distinct_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            first_environment = publisher._child_environment(first)
            second_environment = publisher._child_environment(second)
        for name, expected in publisher.PROCESS_CONTROLS.items():
            self.assertEqual(first_environment[name], expected)
        self.assertNotEqual(
            first_environment["NUMBA_CACHE_DIR"],
            second_environment["NUMBA_CACHE_DIR"],
        )
        self.assertEqual(
            first_environment["PAYNE_ZERO_DATA_ROOT"],
            str(publisher.PINNED_DATA_ROOT.resolve()),
        )

    def test_verify_only_double_capture_never_calls_publisher(self) -> None:
        build_count = 0
        publish_called = False

        def builder(root: Path) -> None:
            nonlocal build_count
            build_count += 1
            (root / "raw").mkdir()
            (root / "final").mkdir()
            (root / "raw" / publisher.RAW_NAME).write_bytes(b"raw")
            for name in publisher.OUTPUT_NAMES:
                (root / "final" / name).write_bytes(name.encode())

        def should_not_publish(source: Path, destination: Path) -> str:
            nonlocal publish_called
            publish_called = True
            return "unexpected"

        with mock.patch.object(publisher, "verify_static_identity", return_value={}):
            result = publisher.generate_and_maybe_publish(
                publish=False,
                capture_builder=builder,
                publisher=should_not_publish,
            )
        self.assertEqual(result["status"], "verified-only")
        self.assertEqual(build_count, 2)
        self.assertFalse(publish_called)

    def test_double_capture_mismatch_prevents_publication(self) -> None:
        build_count = 0
        publish_called = False

        def builder(root: Path) -> None:
            nonlocal build_count
            build_count += 1
            (root / "raw").mkdir()
            (root / "final").mkdir()
            (root / "raw" / publisher.RAW_NAME).write_bytes(
                f"raw-{build_count}".encode()
            )
            for name in publisher.OUTPUT_NAMES:
                (root / "final" / name).write_bytes(name.encode())

        def should_not_publish(source: Path, destination: Path) -> str:
            nonlocal publish_called
            publish_called = True
            return "unexpected"

        with (
            mock.patch.object(publisher, "verify_static_identity", return_value={}),
            self.assertRaises(AssertionError),
        ):
            publisher.generate_and_maybe_publish(
                publish=False,
                capture_builder=builder,
                publisher=should_not_publish,
            )
        self.assertFalse(publish_called)

    def test_second_capture_failure_prevents_publication(self) -> None:
        build_count = 0
        publish_called = False

        def builder(root: Path) -> None:
            nonlocal build_count
            build_count += 1
            if build_count == 2:
                raise RuntimeError("second capture failed")
            (root / "raw").mkdir()
            (root / "final").mkdir()
            (root / "raw" / publisher.RAW_NAME).write_bytes(b"raw")
            for name in publisher.OUTPUT_NAMES:
                (root / "final" / name).write_bytes(name.encode())

        def should_not_publish(source: Path, destination: Path) -> str:
            nonlocal publish_called
            publish_called = True
            return "unexpected"

        with (
            mock.patch.object(publisher, "verify_static_identity", return_value={}),
            self.assertRaisesRegex(RuntimeError, "second capture"),
        ):
            publisher.generate_and_maybe_publish(
                publish=False,
                capture_builder=builder,
                publisher=should_not_publish,
            )
        self.assertFalse(publish_called)

    def test_final_byte_mismatch_prevents_publication(self) -> None:
        build_count = 0
        publish_called = False

        def builder(root: Path) -> None:
            nonlocal build_count
            build_count += 1
            (root / "raw").mkdir()
            (root / "final").mkdir()
            (root / "raw" / publisher.RAW_NAME).write_bytes(b"same-raw")
            for name in publisher.OUTPUT_NAMES:
                payload = name.encode()
                if build_count == 2 and name == publisher.READER_NAME:
                    payload += b"-different"
                (root / "final" / name).write_bytes(payload)

        def should_not_publish(source: Path, destination: Path) -> str:
            nonlocal publish_called
            publish_called = True
            return "unexpected"

        with (
            mock.patch.object(publisher, "verify_static_identity", return_value={}),
            self.assertRaises(AssertionError),
        ):
            publisher.generate_and_maybe_publish(
                publish=False,
                capture_builder=builder,
                publisher=should_not_publish,
            )
        self.assertFalse(publish_called)

    def test_atomic_first_publication_identical_noop_and_refusal(self) -> None:
        def validator(directory: Path) -> None:
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                set(publisher.OUTPUT_NAMES),
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            for name in publisher.OUTPUT_NAMES:
                (source / name).write_bytes(name.encode())
            manifest_path = _write_prepublication_manifest(root)
            acceptance = _publication_record(source, manifest_path)
            destination = root / "published" / "chapter05"
            with (
                mock.patch.object(
                    publisher,
                    "_require_publication_gate",
                    return_value=acceptance,
                ),
                mock.patch.object(
                    publisher, "verify_static_identity", return_value={}
                ),
                mock.patch.object(
                    publisher,
                    "validate_final_directory",
                    side_effect=validator,
                ),
                mock.patch.object(publisher, "OUTPUT_DIR", destination),
            ):
                status = publisher.publish_verified_directory(
                    source,
                    destination,
                )
                self.assertEqual(status, "published")
                before = {
                    name: (destination / name).read_bytes()
                    for name in publisher.OUTPUT_NAMES
                }
                status = publisher.publish_verified_directory(
                    source,
                    destination,
                )
                self.assertEqual(status, "identical-existing")
                (source / publisher.READER_NAME).write_bytes(b"different")
                with self.assertRaises(publisher.PublicationAcceptanceError):
                    publisher.publish_verified_directory(
                        source,
                        destination,
                    )
                self.assertEqual(
                    {
                        name: (destination / name).read_bytes()
                        for name in publisher.OUTPUT_NAMES
                    },
                    before,
                )

    def test_atomic_race_never_clobbers_new_destination(self) -> None:
        def validator(directory: Path) -> None:
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                set(publisher.OUTPUT_NAMES),
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            for name in publisher.OUTPUT_NAMES:
                (source / name).write_bytes(b"candidate")
            manifest_path = _write_prepublication_manifest(root)
            acceptance = _publication_record(source, manifest_path)
            destination = root / "published" / "chapter05"

            def raced_create(stage: Path, target: Path) -> None:
                target.mkdir()
                for name in publisher.OUTPUT_NAMES:
                    (target / name).write_bytes(b"raced-owner")
                raise FileExistsError("raced")

            with (
                mock.patch.object(
                    publisher,
                    "_atomic_rename_directory_no_replace",
                    side_effect=raced_create,
                ),
                mock.patch.object(
                    publisher,
                    "_require_publication_gate",
                    return_value=acceptance,
                ),
                mock.patch.object(
                    publisher, "verify_static_identity", return_value={}
                ),
                mock.patch.object(
                    publisher,
                    "validate_final_directory",
                    side_effect=validator,
                ),
                mock.patch.object(publisher, "OUTPUT_DIR", destination),
                self.assertRaises(publisher.PublicationAcceptanceError),
            ):
                publisher.publish_verified_directory(
                    source,
                    destination,
                )
            for name in publisher.OUTPUT_NAMES:
                self.assertEqual(
                    (destination / name).read_bytes(), b"raced-owner"
                )

    def test_staging_failure_leaves_no_destination(self) -> None:
        call_count = 0

        def validator(directory: Path) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("staged validation failed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            for name in publisher.OUTPUT_NAMES:
                (source / name).write_bytes(name.encode())
            manifest_path = _write_prepublication_manifest(root)
            acceptance = _publication_record(source, manifest_path)
            destination = root / "destination" / "chapter05"
            with (
                mock.patch.object(
                    publisher,
                    "_require_publication_gate",
                    return_value=acceptance,
                ),
                mock.patch.object(
                    publisher, "verify_static_identity", return_value={}
                ),
                mock.patch.object(
                    publisher,
                    "validate_final_directory",
                    side_effect=validator,
                ),
                mock.patch.object(publisher, "OUTPUT_DIR", destination),
                self.assertRaisesRegex(RuntimeError, "staged"),
            ):
                publisher.publish_verified_directory(source, destination)
            self.assertFalse(destination.exists())

    def test_detached_acceptance_record_is_strict_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            for name in publisher.OUTPUT_NAMES:
                (source / name).write_bytes(name.encode())
            manifest_path = _write_prepublication_manifest(root)
            record_path = root / "chapter05_publication_acceptance.json"
            baseline = _publication_record(source, manifest_path)
            with (
                mock.patch.object(
                    publisher,
                    "MANIFEST_PATH",
                    manifest_path,
                ),
                mock.patch.object(
                    publisher,
                    "PUBLICATION_ACCEPTANCE_RECORD_PATH",
                    record_path,
                ),
            ):
                _write_publication_record(record_path, baseline)
                loaded = publisher.load_publication_acceptance_record()
                self.assertEqual(
                    loaded["publisher"]["sha256"],
                    publisher.sha256(publisher.PUBLISHER_PATH),
                )
                self.assertTrue(publisher.publication_gate_ready())

                mutations: list[tuple[str, dict[str, object]]] = []
                extra = json.loads(json.dumps(baseline))
                extra["unexpected"] = True
                mutations.append(("extra field", extra))
                malformed_hex = json.loads(json.dumps(baseline))
                malformed_hex["artifacts"]["reader"]["sha256"] = "g" * 64
                mutations.append(("malformed hex", malformed_hex))
                wrong_publisher = json.loads(json.dumps(baseline))
                wrong_publisher["publisher"]["sha256"] = "0" * 64
                mutations.append(("wrong publisher", wrong_publisher))
                wrong_schema = json.loads(json.dumps(baseline))
                wrong_schema["artifacts"]["integration"]["key_count"] += 1
                mutations.append(("wrong schema", wrong_schema))
                wrong_size = json.loads(json.dumps(baseline))
                wrong_size["artifacts"]["reader"]["bytes"] = 0
                mutations.append(("wrong size", wrong_size))
                wrong_manifest = json.loads(json.dumps(baseline))
                wrong_manifest["manifest"]["prepublication_sha256"] = "1" * 64
                mutations.append(("wrong manifest", wrong_manifest))
                for label, mutation in mutations:
                    with self.subTest(label=label):
                        _write_publication_record(record_path, mutation)
                        self.assertFalse(publisher.publication_gate_ready())
                        with self.assertRaises(
                            publisher.PublicationAcceptanceError
                        ):
                            publisher._require_publication_gate()

                record_path.write_text(
                    '{"schema_version":1,"schema_version":1}\n',
                    encoding="utf-8",
                )
                self.assertFalse(publisher.publication_gate_ready())
                with self.assertRaisesRegex(
                    publisher.PublicationAcceptanceError,
                    "duplicate",
                ):
                    publisher.load_publication_acceptance_record()

    def test_acceptance_record_rejects_symlink_and_nonregular_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            for name in publisher.OUTPUT_NAMES:
                (source / name).write_bytes(name.encode())
            manifest_path = _write_prepublication_manifest(root)
            real_record = root / "reviewed.json"
            _write_publication_record(
                real_record,
                _publication_record(source, manifest_path),
            )
            record_path = root / "chapter05_publication_acceptance.json"
            record_path.symlink_to(real_record)
            with (
                mock.patch.object(publisher, "MANIFEST_PATH", manifest_path),
                mock.patch.object(
                    publisher,
                    "PUBLICATION_ACCEPTANCE_RECORD_PATH",
                    record_path,
                ),
                self.assertRaises(publisher.PublicationAcceptanceError),
            ):
                publisher.load_publication_acceptance_record()
            record_path.unlink()
            record_path.mkdir()
            with (
                mock.patch.object(publisher, "MANIFEST_PATH", manifest_path),
                mock.patch.object(
                    publisher,
                    "PUBLICATION_ACCEPTANCE_RECORD_PATH",
                    record_path,
                ),
                self.assertRaises(publisher.PublicationAcceptanceError),
            ):
                publisher.load_publication_acceptance_record()
            record_path.rmdir()
            _write_publication_record(
                record_path,
                _publication_record(source, manifest_path),
            )
            real_manifest = root / "real-MANIFEST.json"
            real_manifest.write_bytes(manifest_path.read_bytes())
            manifest_path.unlink()
            manifest_path.symlink_to(real_manifest)
            with (
                mock.patch.object(publisher, "MANIFEST_PATH", manifest_path),
                mock.patch.object(
                    publisher,
                    "PUBLICATION_ACCEPTANCE_RECORD_PATH",
                    record_path,
                ),
                self.assertRaises(publisher.PublicationAcceptanceError),
            ):
                publisher.load_publication_acceptance_record()

    def test_direct_api_revalidates_record_candidate_and_manifest_no_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            original_bytes: dict[str, bytes] = {}
            for name in publisher.OUTPUT_NAMES:
                original_bytes[name] = name.encode()
                (source / name).write_bytes(original_bytes[name])
            manifest_path = _write_prepublication_manifest(root)
            record_path = root / "chapter05_publication_acceptance.json"
            baseline = _publication_record(source, manifest_path)
            _write_publication_record(record_path, baseline)
            destination = root / "must-remain-absent" / "chapter05"
            patches = (
                mock.patch.object(publisher, "MANIFEST_PATH", manifest_path),
                mock.patch.object(
                    publisher,
                    "PUBLICATION_ACCEPTANCE_RECORD_PATH",
                    record_path,
                ),
                mock.patch.object(
                    publisher, "verify_static_identity", return_value={}
                ),
                mock.patch.object(
                    publisher, "validate_final_directory", return_value=None
                ),
                mock.patch.object(publisher, "OUTPUT_DIR", destination),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                (source / publisher.READER_NAME).write_bytes(b"wrong-candidate")
                with self.assertRaisesRegex(
                    publisher.PublicationAcceptanceError,
                    "candidate bytes",
                ):
                    publisher.publish_verified_directory(source, destination)
                self.assertFalse(destination.parent.exists())

                (source / publisher.READER_NAME).write_bytes(
                    original_bytes[publisher.READER_NAME]
                )
                symlink_target = root / "reader-target.npz"
                symlink_target.write_bytes(
                    original_bytes[publisher.READER_NAME]
                )
                (source / publisher.READER_NAME).unlink()
                (source / publisher.READER_NAME).symlink_to(symlink_target)
                with self.assertRaises(publisher.PublicationAcceptanceError):
                    publisher.publish_verified_directory(source, destination)
                self.assertFalse(destination.parent.exists())
                (source / publisher.READER_NAME).unlink()
                (source / publisher.READER_NAME).write_bytes(
                    original_bytes[publisher.READER_NAME]
                )
                manifest_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "payne_zero_commit": publisher.PINNED_COMMIT,
                            "entries": [{"path": "unreviewed"}],
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(publisher.PublicationAcceptanceError):
                    publisher.publish_verified_directory(source, destination)
                self.assertFalse(destination.parent.exists())

                record_path.unlink()
                with (
                    mock.patch.object(
                        publisher,
                        "publication_gate_ready",
                        return_value=True,
                    ),
                    self.assertRaises(publisher.PublicationAcceptanceError),
                ):
                    publisher.publish_verified_directory(source, destination)
                self.assertFalse(destination.parent.exists())

    def test_alternate_byte_identical_publisher_path_fails_before_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            source.mkdir()
            for name in publisher.OUTPUT_NAMES:
                (source / name).write_bytes(name.encode())
            manifest_path = _write_prepublication_manifest(root)
            record_path = root / "chapter05_publication_acceptance.json"
            _write_publication_record(
                record_path,
                _publication_record(source, manifest_path),
            )
            alternate = root / "byte-identical-publisher.py"
            alternate.write_bytes(publisher.PUBLISHER_PATH.read_bytes())
            destination = root / "must-remain-absent" / "chapter05"
            with (
                mock.patch.object(publisher, "__file__", str(alternate)),
                mock.patch.object(publisher, "MANIFEST_PATH", manifest_path),
                mock.patch.object(
                    publisher,
                    "PUBLICATION_ACCEPTANCE_RECORD_PATH",
                    record_path,
                ),
                mock.patch.object(publisher, "OUTPUT_DIR", destination),
            ):
                self.assertFalse(publisher.publication_gate_ready())
                with self.assertRaisesRegex(
                    publisher.PublisherIdentityError,
                    "canonical path",
                ):
                    publisher.verify_static_identity()
                with self.assertRaisesRegex(
                    publisher.PublicationAcceptanceError,
                    "disabled",
                ):
                    publisher.publish_verified_directory(source, destination)
            self.assertFalse(destination.parent.exists())

    def test_relocated_publisher_subprocess_fails_before_import_or_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            alternate_root = Path(temporary).resolve() / "relocated-textbook"
            alternate_scripts = alternate_root / "scripts"
            alternate_scripts.mkdir(parents=True)
            alternate_publisher = alternate_scripts / publisher.PUBLISHER_PATH.name
            shutil.copy2(publisher.PUBLISHER_PATH, alternate_publisher)
            alternate_output = (
                alternate_root
                / "data"
                / "golden"
                / "payne_zero"
                / "chapter05"
            )

            for arguments in (
                (),
                ("--full-capture", "--publish"),
            ):
                with self.subTest(arguments=arguments):
                    completed = subprocess.run(
                        [sys.executable, str(alternate_publisher), *arguments],
                        cwd=alternate_root,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(
                        "not the reviewed canonical path",
                        completed.stderr,
                    )
                    self.assertFalse(alternate_output.exists())
                    self.assertFalse(alternate_output.parent.exists())

    def test_post_double_capture_rejects_unaccepted_candidate_before_publish(
        self,
    ) -> None:
        publish_called = False

        def builder(root: Path) -> None:
            (root / "raw").mkdir()
            (root / "final").mkdir()
            (root / "raw" / publisher.RAW_NAME).write_bytes(b"same-raw")
            for name in publisher.OUTPUT_NAMES:
                (root / "final" / name).write_bytes(name.encode())

        def should_not_publish(source: Path, destination: Path) -> str:
            nonlocal publish_called
            publish_called = True
            return "unexpected"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            template = root / "template"
            template.mkdir()
            for name in publisher.OUTPUT_NAMES:
                (template / name).write_bytes(name.encode())
            manifest_path = _write_prepublication_manifest(root)
            record = _publication_record(template, manifest_path)
            accepted_record = json.loads(json.dumps(record))
            record["artifacts"]["integration"]["sha256"] = "2" * 64
            record_path = root / "chapter05_publication_acceptance.json"
            _write_publication_record(record_path, record)
            destination = root / "must-remain-absent" / "chapter05"
            with (
                mock.patch.object(publisher, "MANIFEST_PATH", manifest_path),
                mock.patch.object(
                    publisher,
                    "PUBLICATION_ACCEPTANCE_RECORD_PATH",
                    record_path,
                ),
                mock.patch.object(
                    publisher, "verify_static_identity", return_value={}
                ),
                mock.patch.object(publisher, "OUTPUT_DIR", destination),
                self.assertRaises(publisher.PublicationAcceptanceError),
            ):
                publisher.generate_and_maybe_publish(
                    publish=True,
                    destination=destination,
                    capture_builder=builder,
                    publisher=should_not_publish,
                )
            _write_publication_record(record_path, accepted_record)
            with (
                mock.patch.object(publisher, "MANIFEST_PATH", manifest_path),
                mock.patch.object(
                    publisher,
                    "PUBLICATION_ACCEPTANCE_RECORD_PATH",
                    record_path,
                ),
                mock.patch.object(
                    publisher, "verify_static_identity", return_value={}
                ),
                mock.patch.object(publisher, "OUTPUT_DIR", destination),
                self.assertRaisesRegex(
                    publisher.PublicationAcceptanceError,
                    "injected",
                ),
            ):
                publisher.generate_and_maybe_publish(
                    publish=True,
                    destination=destination,
                    capture_builder=builder,
                    publisher=should_not_publish,
                )
            self.assertFalse(publish_called)
            self.assertFalse(destination.parent.exists())

    def test_publication_rejects_noncanonical_destinations_before_any_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            canonical = root / "canonical" / "chapter05"
            alternate = root / "alternate" / "chapter05"
            relative = Path(f".chapter05-relative-{root.name}") / "chapter05"
            self.assertFalse(relative.parent.exists())
            with mock.patch.object(publisher, "OUTPUT_DIR", canonical):
                for destination in (relative, alternate):
                    with (
                        self.subTest(destination=str(destination)),
                        self.assertRaisesRegex(
                            publisher.PublicationAcceptanceError,
                            "canonical OUTPUT_DIR",
                        ),
                    ):
                        publisher.publish_verified_directory(
                            root / "unused-source",
                            destination,
                        )
                    self.assertFalse(destination.exists())
                    if destination.is_absolute():
                        self.assertFalse(destination.parent.exists())
            self.assertFalse(canonical.parent.exists())
            self.assertFalse(relative.parent.exists())

    def test_cli_requires_explicit_full_capture_and_has_no_force(self) -> None:
        defaults = publisher.parse_args([])
        self.assertFalse(defaults.full_capture)
        self.assertFalse(defaults.publish)
        accepted = publisher.parse_args(["--full-capture", "--verify-only"])
        self.assertTrue(accepted.full_capture)
        with self.assertRaises(SystemExit):
            publisher.parse_args(["--publish"])
        with self.assertRaises(SystemExit):
            publisher.parse_args(["--full-capture"])
        if publisher.publication_gate_ready():
            publish = publisher.parse_args(["--full-capture", "--publish"])
            self.assertTrue(publish.full_capture)
            self.assertTrue(publish.publish)
        else:
            with self.assertRaises(SystemExit):
                publisher.parse_args(["--full-capture", "--publish"])
        with self.assertRaises(SystemExit):
            publisher.parse_args(["--force"])
        with self.assertRaises(SystemExit):
            publisher.parse_args(["--destination", "relative-chapter05"])
        with tempfile.TemporaryDirectory() as temporary:
            alternate = Path(temporary).resolve() / "alternate" / "chapter05"
            with self.assertRaises(SystemExit):
                publisher.parse_args(["--destination", str(alternate)])
            self.assertFalse(alternate.parent.exists())
        if publisher.publication_gate_ready():
            self.assertEqual(
                publisher._require_publication_gate()["record_kind"],
                publisher.PUBLICATION_ACCEPTANCE_KIND,
            )
        else:
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                publisher._require_publication_gate()

    @unittest.skipUnless(
        os.environ.get("CHAPTER05_RUN_FULL_PUBLISHER") == "1",
        "set CHAPTER05_RUN_FULL_PUBLISHER=1 for the real double capture",
    )
    def test_real_double_capture_verify_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = publisher.generate_and_maybe_publish(
                publish=False,
                destination=Path(temporary) / "must-not-exist",
            )
        self.assertEqual(result["status"], "verified-only")


if __name__ == "__main__":
    unittest.main()
