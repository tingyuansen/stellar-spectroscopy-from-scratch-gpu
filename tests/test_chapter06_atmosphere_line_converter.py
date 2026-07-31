"""Exhaustive gates for the pure Chapter 6 atmosphere line converter."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import chapter06_atmosphere_line_converter as converter
from payne_zero_atmosphere.line_catalog import (
    SelectedLineCatalog,
    decode_selected_line_words,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SELECTED_VALUES = {
    "packed_wavelength_index": 12_425_352,
    "packed_species_slot": 3_510,
    "lower_excitation_index": 20_909,
    "log_strength_index": 15_524,
    "radiative_damping_index": 24_854,
    "stark_damping_index": 11_934,
    "van_der_waals_damping_index": 8_874,
}
EXPECTED_MEMBER_HASHES = {
    "packed_wavelength_index": (
        "c423f4ac4a3825c6fad5336a1e15c0038ab17087f930916ad550ad45b4990dfc"
    ),
    "packed_species_slot": (
        "c2126161f7488b7d198ea310da9a8694786a18a2317eea5e30379ee118d34743"
    ),
    "lower_excitation_index": (
        "9b5bf0b74e2e212b57bcb2b9f2712eaab4c6169595f4e82767793f4534365648"
    ),
    "log_strength_index": (
        "3fe821d54660a0c51b42d19a42571e208791b3c5ff6bc0cac16b6553a226515a"
    ),
    "radiative_damping_index": (
        "5cdf3b75730a5b45c3f24da2c1030103143981191d2844aa7374e948b9abaeea"
    ),
    "stark_damping_index": (
        "91d82e7b89ae43b29bb673bb416697d005e093d0011c9d7af501630c6a502141"
    ),
    "van_der_waals_damping_index": (
        "c0346f09a0362e8e9d29c01a9fc7c292a7395b524a90319bb921608b9fdb1b60"
    ),
}
EXPECTED_WORDS = np.asarray(
    [[12_425_352, 1_370_295_734, 1_628_847_268, 581_578_398]],
    dtype=np.int32,
)
EXPECTED_WORD_HASH = "1769c9ad8d33e847a099bd6d50df85a2f478f98d554b2fc39db8121ba93158d2"


def array_sha256(values: np.ndarray) -> str:
    """Return one array's canonical C-byte identity."""

    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


class Chapter06AtmosphereLineConverterTests(unittest.TestCase):
    """Freeze conversion steps 1--3 without creating a fixture or golden."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = converter.load_verified_canonical_raw_row()
        cls.archive_members = converter._load_verified_subset_archive(
            converter.CANONICAL_SUBSET_PATH
        )
        cls.manifest = converter._load_manifest_without_duplicate_keys(
            converter.MANIFEST_PATH
        )
        cls.selected = converter.convert_raw_row_to_selected_line_catalog(cls.raw)

    def test_canonical_subset_hash_schema_and_manifest_binding(self) -> None:
        subset = converter.CANONICAL_SUBSET_PATH
        self.assertEqual(subset.stat().st_size, converter.CANONICAL_SUBSET_BYTES)
        self.assertEqual(
            hashlib.sha256(subset.read_bytes()).hexdigest(),
            converter.CANONICAL_SUBSET_SHA256,
        )
        with np.load(subset, allow_pickle=False) as archive:
            self.assertEqual(
                tuple(archive.files),
                tuple(
                    sorted(
                        set(converter.RAW_FIELD_NAMES) | set(converter.PROVENANCE_SPEC)
                    )
                ),
            )
            self.assertTrue(
                all(
                    not np.asarray(archive[name]).dtype.hasobject
                    for name in archive.files
                )
            )

        manifest = json.loads((REPOSITORY_ROOT / "data/MANIFEST.json").read_text())
        entry = {item["path"]: item for item in manifest["entries"]}[
            "data/subsets/chapter06_fe_i_source_row_873702.npz"
        ]
        self.assertEqual(entry["sha256"], converter.CANONICAL_SUBSET_SHA256)
        self.assertEqual(entry["bytes"], converter.CANONICAL_SUBSET_BYTES)
        self.assertEqual(entry["source_row_index"], converter.SOURCE_ROW_INDEX)
        self.assertEqual(
            entry["subset_schema_version"],
            converter.SUBSET_SCHEMA_VERSION,
        )
        self.assertEqual(entry["source_commit"], converter.PAYNE_ZERO_COMMIT)
        self.assertEqual(
            entry["source_sha256"],
            converter.SOURCE_ARCHIVE_SHA256,
        )
        self.assertEqual(
            entry["source_archive_bytes"],
            converter.SOURCE_ARCHIVE_BYTES,
        )
        self.assertEqual(
            entry["source_archive_row_count"],
            converter.SOURCE_ARCHIVE_ROW_COUNT,
        )
        self.assertEqual(entry["source_field_count"], len(converter.RAW_FIELD_NAMES))
        self.assertEqual(
            entry["builder"],
            converter.SUBSET_BUILDER_RELATIVE_PATH,
        )
        self.assertEqual(
            entry["builder_sha256"],
            converter.SUBSET_BUILDER_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(converter.SUBSET_BUILDER_PATH.read_bytes()).hexdigest(),
            converter.SUBSET_BUILDER_SHA256,
        )
        converter._validate_manifest_authority(
            self.manifest,
            self.archive_members,
        )

    def test_all_seventeen_raw_fields_have_exact_dtypes_values_and_bytes(
        self,
    ) -> None:
        self.assertEqual(
            tuple(converter.RAW_FIELD_SPEC),
            converter.RAW_FIELD_NAMES,
        )
        self.assertEqual(len(self.raw), 17)
        for name, (expected_dtype, expected_value) in converter.RAW_FIELD_SPEC.items():
            with self.subTest(name=name):
                values = self.raw[name]
                self.assertEqual(values.shape, (1,))
                self.assertEqual(values.dtype, expected_dtype)
                self.assertEqual(values[0].item(), expected_value)
        self.assertEqual(self.raw["energy_shift_field"].tobytes(), b" " * 10)
        self.assertEqual(
            self.raw["line_category_tag"].tobytes(),
            b"\x00" * 3,
        )

    def test_loader_rejects_any_noncanonical_file_bytes(self) -> None:
        original = converter.CANONICAL_SUBSET_PATH.read_bytes()
        with tempfile.TemporaryDirectory() as temporary_directory:
            candidate = Path(temporary_directory) / "candidate.npz"
            candidate.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                converter._load_verified_subset_archive(candidate)
            candidate.write_bytes(original[:-1])
            with self.assertRaisesRegex(ValueError, "bytes"):
                converter._load_verified_subset_archive(candidate)

    def test_public_loader_has_no_alternate_path_and_rejects_symlinked_canonical(
        self,
    ) -> None:
        self.assertEqual(
            tuple(
                inspect.signature(converter.load_verified_canonical_raw_row).parameters
            ),
            (),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            exact_copy = temporary_root / "same-bytes-unregistered.npz"
            exact_copy.write_bytes(converter.CANONICAL_SUBSET_PATH.read_bytes())
            with self.assertRaises(TypeError):
                converter.load_verified_canonical_raw_row(exact_copy)

            subset_alias = temporary_root / "subset-alias.npz"
            subset_alias.symlink_to(converter.CANONICAL_SUBSET_PATH)
            with mock.patch.object(
                converter,
                "CANONICAL_SUBSET_PATH",
                subset_alias,
            ):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    converter.load_verified_canonical_raw_row()

    def test_manifest_authority_mutations_fail_closed(self) -> None:
        target_path = converter.CANONICAL_SUBSET_RELATIVE_PATH

        def target_entry(document):
            return next(
                entry
                for entry in document["entries"]
                if entry.get("path") == target_path
            )

        for name, changed_value, expected_message in (
            (
                "schema_version",
                converter.MANIFEST_SCHEMA_VERSION + 1,
                "schema version",
            ),
            ("payne_zero_commit", "0" * 40, "Payne Zero commit"),
        ):
            with self.subTest(manifest_root=name):
                changed = copy.deepcopy(self.manifest)
                changed[name] = changed_value
                with self.assertRaisesRegex(ValueError, expected_message):
                    converter._validate_manifest_authority(
                        changed,
                        self.archive_members,
                    )

        missing = copy.deepcopy(self.manifest)
        missing["entries"] = [
            entry for entry in missing["entries"] if entry.get("path") != target_path
        ]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            converter._validate_manifest_authority(missing, self.archive_members)

        duplicate = copy.deepcopy(self.manifest)
        duplicate["entries"].append(copy.deepcopy(target_entry(duplicate)))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            converter._validate_manifest_authority(duplicate, self.archive_members)

        scalar_mutations = {
            "path": "data/subsets/not-the-canonical-row.npz",
            "sha256": "0" * 64,
            "builder": "scripts/not-the-builder.py",
            "builder_sha256": "0" * 64,
            "source_sha256": "0" * 64,
            "source_archive_bytes": converter.SOURCE_ARCHIVE_BYTES + 1,
            "source_archive_row_count": converter.SOURCE_ARCHIVE_ROW_COUNT + 1,
            "source_field_count": len(converter.RAW_FIELD_NAMES) + 1,
        }
        for name, changed_value in scalar_mutations.items():
            with self.subTest(field=name):
                changed = copy.deepcopy(self.manifest)
                target_entry(changed)[name] = changed_value
                expected_message = "exactly one" if name == "path" else "manifest field"
                with self.assertRaisesRegex(ValueError, expected_message):
                    converter._validate_manifest_authority(
                        changed,
                        self.archive_members,
                    )

        for field, changed_value in (
            ("shape", [2]),
            ("dtype", "float32"),
            ("sha256", "0" * 64),
        ):
            with self.subTest(member_metadata=field):
                changed = copy.deepcopy(self.manifest)
                target_entry(changed)["arrays"]["stored_wavelength_nm"][field] = (
                    changed_value
                )
                with self.assertRaisesRegex(
                    ValueError, "SHA-256" if field == "sha256" else field
                ):
                    converter._validate_manifest_authority(
                        changed,
                        self.archive_members,
                    )

        missing_member = copy.deepcopy(self.manifest)
        target_entry(missing_member)["arrays"].pop("stored_wavelength_nm")
        with self.assertRaisesRegex(ValueError, "member set"):
            converter._validate_manifest_authority(
                missing_member,
                self.archive_members,
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            changed_builder = Path(temporary_directory) / "builder.py"
            changed_builder.write_bytes(
                converter.SUBSET_BUILDER_PATH.read_bytes() + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "builder SHA-256"):
                converter._validate_manifest_authority(
                    self.manifest,
                    self.archive_members,
                    builder_path=changed_builder,
                )

    def test_raw_validator_rejects_missing_extra_shape_and_dtype_mutations(
        self,
    ) -> None:
        missing = copy.deepcopy(self.raw)
        missing.pop(converter.RAW_FIELD_NAMES[0])
        with self.assertRaisesRegex(ValueError, "missing="):
            converter.validate_canonical_raw_row(missing)

        extra = copy.deepcopy(self.raw)
        extra["invented_field"] = np.asarray([0.0])
        with self.assertRaisesRegex(ValueError, "extra="):
            converter.validate_canonical_raw_row(extra)

        for name in converter.RAW_FIELD_NAMES:
            with self.subTest(kind="shape", name=name):
                changed = copy.deepcopy(self.raw)
                changed[name] = np.repeat(changed[name], 2)
                with self.assertRaisesRegex(ValueError, "shape"):
                    converter.validate_canonical_raw_row(changed)
            with self.subTest(kind="dtype", name=name):
                changed = copy.deepcopy(self.raw)
                values = changed[name]
                if values.dtype.kind == "f":
                    changed[name] = values.astype(np.float32)
                elif values.dtype.kind == "i":
                    changed[name] = values.astype(np.int32)
                else:
                    changed[name] = values.astype(f"S{values.dtype.itemsize + 1}")
                with self.assertRaisesRegex(ValueError, "dtype"):
                    converter.validate_canonical_raw_row(changed)

    def test_raw_validator_rejects_a_value_mutation_in_every_field(self) -> None:
        for name in converter.RAW_FIELD_NAMES:
            with self.subTest(name=name):
                changed = copy.deepcopy(self.raw)
                values = changed[name]
                if values.dtype.kind == "f":
                    changed[name][0] = np.nextafter(values[0], np.inf)
                elif values.dtype.kind == "i":
                    changed[name][0] += 1
                elif name == "energy_shift_field":
                    changed[name][0] = b"1         "
                else:
                    changed[name][0] = b"AUT"
                with self.assertRaisesRegex(ValueError, "value"):
                    converter.validate_canonical_raw_row(changed)

    def test_quantizers_reject_all_nonfinite_and_nonpositive_inputs(self) -> None:
        for value in (np.nan, np.inf, -np.inf, 0.0, -1.0):
            with self.subTest(quantizer="wavelength", value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite and positive",
                ):
                    converter.packed_wavelength_code(value)
            with self.subTest(quantizer="TABLOG", value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite and positive",
                ):
                    converter.tablog_code(value)

    def test_quantizers_enforce_rounding_and_storage_domains_exactly(self) -> None:
        self.assertEqual(
            int(converter.packed_wavelength_code(499.03411946178176)),
            12_425_352,
        )
        for code in (1, 16_384, 32_767):
            value = 10.0 ** ((code - 16_384.0) * 0.001)
            with self.subTest(TABLOG_code=code):
                self.assertEqual(int(converter.tablog_code(value)), code)
        for code in (0, 32_768):
            value = 10.0 ** ((code - 16_384.0) * 0.001)
            with self.subTest(rejected_TABLOG_code=code):
                with self.assertRaisesRegex(ValueError, "does not fit int16"):
                    converter.tablog_code(value)

        # Positive finite float64 wavelengths cannot reach the int32 guard with
        # the physical ratio.  Tightening only the step exercises the fail-closed
        # cast guard without changing the production constant.
        with mock.patch.object(
            converter,
            "RATIO_LOG_STEP",
            np.float64(1.0e-12),
        ):
            with self.assertRaisesRegex(ValueError, "does not fit int32"):
                converter.packed_wavelength_code(np.finfo(np.float64).max)

    def test_converter_returns_the_exact_seven_values_dtypes_and_hashes(
        self,
    ) -> None:
        self.assertIsInstance(self.selected, SelectedLineCatalog)
        self.assertEqual(self.selected.line_count, 1)
        for name, expected_value in EXPECTED_SELECTED_VALUES.items():
            with self.subTest(name=name):
                values = np.asarray(getattr(self.selected, name))
                self.assertEqual(values.shape, (1,))
                self.assertEqual(
                    values.dtype,
                    converter.SELECTED_FIELD_DTYPES[name],
                )
                self.assertEqual(int(values[0]), expected_value)
                self.assertEqual(
                    array_sha256(values),
                    EXPECTED_MEMBER_HASHES[name],
                )
        self.assertEqual(
            converter.selected_line_member_hashes(self.selected),
            EXPECTED_MEMBER_HASHES,
        )
        self.assertEqual(
            abs(int(self.selected.packed_species_slot[0])) // 10,
            351,
        )

    def test_independent_scalar_reproduction_matches_all_seven_fields(
        self,
    ) -> None:
        first = abs(float(self.raw["first_energy_column_cm"][0]))
        second = abs(float(self.raw["second_energy_column_cm"][0]))
        wavelength_nm = 1.0e7 / abs(first - second)
        lower_excitation_cm = min(first, second)
        oscillator_strength = 10.0 ** (
            float(self.raw["raw_log_oscillator_strength"][0])
            + float(self.raw["primary_isotope_log_correction"][0])
            + float(self.raw["secondary_isotope_log_correction"][0])
        )
        ratio_log_step = np.log(1.0 + 1.0 / 2_000_000.0)

        def independent_tablog(value: float) -> int:
            return int(np.floor(np.log10(value) * 1000.0 + 16384.5))

        independent_values = {
            "packed_wavelength_index": int(
                np.floor(np.log(wavelength_nm) / ratio_log_step + 0.5)
            ),
            "packed_species_slot": 3_510,
            "lower_excitation_index": independent_tablog(lower_excitation_cm),
            "log_strength_index": independent_tablog(oscillator_strength),
            "radiative_damping_index": independent_tablog(
                10.0 ** float(self.raw["radiative_damping_log"][0])
            ),
            "stark_damping_index": independent_tablog(
                10.0 ** float(self.raw["stark_damping_log"][0])
            ),
            "van_der_waals_damping_index": independent_tablog(
                10.0 ** float(self.raw["van_der_waals_damping_log"][0])
            ),
        }
        self.assertEqual(independent_values, EXPECTED_SELECTED_VALUES)

        halfwords = np.asarray(
            [
                [
                    independent_values["packed_species_slot"],
                    independent_values["lower_excitation_index"],
                    independent_values["log_strength_index"],
                    independent_values["radiative_damping_index"],
                    independent_values["stark_damping_index"],
                    independent_values["van_der_waals_damping_index"],
                ]
            ],
            dtype=np.int16,
        )
        independently_packed = np.column_stack(
            [
                np.asarray(
                    [independent_values["packed_wavelength_index"]],
                    dtype=np.int32,
                ),
                halfwords.view(np.int32).reshape(1, 3),
            ]
        )
        np.testing.assert_array_equal(independently_packed, EXPECTED_WORDS)
        self.assertEqual(array_sha256(independently_packed), EXPECTED_WORD_HASH)

    def test_native_four_word_payload_and_no_swap_decode_are_exact(self) -> None:
        words = converter.pack_selected_line_words(self.selected)
        self.assertEqual(words.shape, (1, 4))
        self.assertEqual(words.dtype, np.dtype("<i4"))
        np.testing.assert_array_equal(words, EXPECTED_WORDS)
        self.assertEqual(array_sha256(words), EXPECTED_WORD_HASH)

        decoded = decode_selected_line_words(
            words,
            detect_swapped_layout=False,
        )
        for name in converter.SELECTED_FIELD_DTYPES:
            with self.subTest(name=name):
                np.testing.assert_array_equal(
                    getattr(decoded, name),
                    getattr(self.selected, name),
                )
                self.assertEqual(
                    getattr(decoded, name).dtype,
                    converter.SELECTED_FIELD_DTYPES[name],
                )

    def test_packer_rejects_silent_shape_or_dtype_casts(self) -> None:
        for name in converter.SELECTED_FIELD_DTYPES:
            values = {
                field: np.array(getattr(self.selected, field), copy=True)
                for field in converter.SELECTED_FIELD_DTYPES
            }
            with self.subTest(kind="shape", name=name):
                values[name] = np.repeat(values[name], 2)
                malformed = SelectedLineCatalog(**values)
                with self.assertRaisesRegex(ValueError, "shape"):
                    converter.pack_selected_line_words(malformed)
            with self.subTest(kind="dtype", name=name):
                values = {
                    field: np.array(getattr(self.selected, field), copy=True)
                    for field in converter.SELECTED_FIELD_DTYPES
                }
                values[name] = values[name].astype(np.int64)
                malformed = SelectedLineCatalog(**values)
                with self.assertRaisesRegex(ValueError, "dtype"):
                    converter.pack_selected_line_words(malformed)

    def test_decoded_physical_ledger_matches_float32_kernel_order(self) -> None:
        ledger = converter.decoded_physical_ledger(self.raw, self.selected)
        self.assertEqual(
            ledger.reconstructed_wavelength_nm,
            499.03410878793585,
        )
        self.assertEqual(
            ledger.unquantized_wavelength_nm,
            499.03411946178176,
        )
        self.assertEqual(
            ledger.reconstructed_minus_unquantized_nm,
            -1.0673845906694623e-05,
        )
        self.assertEqual(ledger.population_slot_one_based, 351)
        self.assertEqual(ledger.lower_excitation_cm, 33496.54296875)
        self.assertEqual(
            ledger.oscillator_strength,
            0.13803842663764954,
        )
        self.assertEqual(
            ledger.raw_radiative_damping_s_inverse,
            295120928.0,
        )
        self.assertEqual(
            ledger.raw_stark_damping_cm3_s_inverse,
            3.548133827280253e-05,
        )
        self.assertEqual(
            ledger.raw_van_der_waals_damping_cm3_s_inverse,
            3.090295308538771e-08,
        )
        self.assertEqual(
            ledger.classical_strength_cm2,
            3.440358938918034e-18,
        )
        self.assertEqual(
            ledger.radiative_damping,
            3.909297063842132e-08,
        )
        self.assertEqual(
            ledger.stark_damping_cm3,
            4.700008332680409e-21,
        )
        self.assertEqual(
            ledger.van_der_waals_damping_cm3,
            4.093536498989339e-24,
        )

    def test_observed_row_is_not_an_input_or_authority(self) -> None:
        self.assertEqual(
            converter.NON_AUTHORITATIVE_OBSERVED_ROW_INDEX,
            780_108,
        )
        module_text = (
            REPOSITORY_ROOT / "scripts/chapter06_atmosphere_line_converter.py"
        ).read_text()
        self.assertNotIn("observed_atomic_lines.npy", module_text)
        self.assertNotIn("np.load(SOURCE_ARCHIVE_PATH_TEXT", module_text)
        self.assertNotIn("Path(SOURCE_ARCHIVE_PATH_TEXT", module_text)


if __name__ == "__main__":
    unittest.main()
