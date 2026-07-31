"""Deterministic source, provenance, and visibility gates for the Chapter 6 line."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from book.chapter05_runtime import load_regime_state, run_synthesis_continuum
from payne_zero_synthesis import atomic_lines, line_opacity
from scripts import build_chapter06_fe_record_subset as subset_builder
from scripts.deterministic_npz import write_npz


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUBSET_PATH = (
    REPOSITORY_ROOT / "data/subsets/chapter06_fe_i_source_row_873702.npz"
)
SUBSET_SHA256 = (
    "bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04"
)
SUBSET_BYTES = 8665
BUILDER_PATH = REPOSITORY_ROOT / "scripts/build_chapter06_fe_record_subset.py"
BUILDER_SHA256 = (
    "25bcf4662740155e8b08615b9522f3f4517e1a5ddc4627c68686620ccfff4d6c"
)
SOURCE_ARCHIVE = (
    Path("/Users/ysting/payne-zero")
    / subset_builder.SOURCE_ARCHIVE_RELATIVE_PATH
)
LINE_PROFILE_TABLES = (
    REPOSITORY_ROOT / "data/static/synthesis_tables/line_profile_tables.npz"
)

RAW_EXPECTED = {
    "stored_wavelength_nm": (np.dtype("<f8"), 499.0341),
    "raw_log_oscillator_strength": (np.dtype("<f8"), -0.86),
    "species_code": (np.dtype("<f8"), 26.0),
    "first_energy_column_cm": (np.dtype("<f8"), 53545.833),
    "second_energy_column_cm": (np.dtype("<f8"), 33507.123),
    "radiative_damping_log": (np.dtype("<f8"), 8.47),
    "stark_damping_log": (np.dtype("<f8"), -4.45),
    "van_der_waals_damping_log": (np.dtype("<f8"), -7.51),
    "lower_principal_quantum_number": (np.dtype("<i8"), 0),
    "upper_principal_quantum_number": (np.dtype("<i8"), 0),
    "primary_isotope_number": (np.dtype("<i8"), 0),
    "primary_isotope_log_correction": (np.dtype("<f8"), 0.0),
    "secondary_isotope_log_correction": (np.dtype("<f8"), 0.0),
    "energy_shift_field": (np.dtype("S10"), b"          "),
    "isotope_shift_units": (np.dtype("<f8"), 0.0),
    "line_size": (np.dtype("<i8"), 0),
    "line_category_tag": (np.dtype("S3"), b""),
}

PROVENANCE_FIELDS = {
    "builder_command",
    "payne_zero_commit",
    "source_archive_bytes",
    "source_archive_relative_path",
    "source_archive_row_count",
    "source_archive_sha256",
    "source_field_count",
    "source_row_index",
    "subset_role",
    "subset_schema_version",
}

REGIME_ACTIVE_MASKS = {
    "hot_dwarf": (True, True, True, False, False, False),
    "solar_dwarf": (True, True, True, True, True, True),
    "low_gravity_giant": (True, True, True, True, True, True),
    "cool_molecule_rich": (True, True, True, True, True, True),
}


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_subset() -> dict[str, np.ndarray]:
    """Load independent subset arrays without retaining an NPZ handle."""

    with np.load(SUBSET_PATH, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def one_record_catalog(
    records: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Build the exact one-record production mapping from static inputs."""

    per_line_fields = (
        "line_type",
        "atomic_number",
        "ion_stage",
        "wavelength_nm",
        "index_wavelength_nm",
        "oscillator_strength",
        "lower_excitation_cm",
        "radiative_damping",
        "stark_damping",
        "van_der_waals_damping",
        "raw_radiative_damping_log",
        "raw_stark_damping_log",
        "raw_van_der_waals_damping_log",
    )
    catalog = {name: np.array(records[name], copy=True) for name in per_line_fields}
    with np.load(LINE_PROFILE_TABLES, allow_pickle=False) as tables:
        for name in (
            "harris_profile_h0_table",
            "harris_profile_h1_table",
            "harris_profile_h2_table",
        ):
            catalog[name] = np.array(tables[name], copy=True)
    catalog["helium_line_type"] = np.empty(0, dtype=np.int64)
    catalog["helium_line_center_cutoff_ratio"] = np.float64(
        line_opacity.LINE_CENTER_CUTOFF_RATIO
    )
    return catalog


class Chapter06FeRecordSubsetTests(unittest.TestCase):
    """Freeze one raw Fe I source row without mixing in computed outputs."""

    def test_subset_and_builder_have_pinned_byte_identities(self) -> None:
        self.assertEqual(SUBSET_PATH.stat().st_size, SUBSET_BYTES)
        self.assertEqual(sha256(SUBSET_PATH), SUBSET_SHA256)
        self.assertEqual(sha256(BUILDER_PATH), BUILDER_SHA256)

    def test_archive_members_are_deterministic_and_role_honest(self) -> None:
        expected_members = set(RAW_EXPECTED) | PROVENANCE_FIELDS
        with zipfile.ZipFile(SUBSET_PATH) as archive:
            infos = archive.infolist()
            self.assertEqual(
                [info.filename for info in infos],
                [f"{name}.npy" for name in sorted(expected_members)],
            )
            self.assertTrue(
                all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in infos)
            )
            self.assertTrue(
                all(info.compress_type == zipfile.ZIP_STORED for info in infos)
            )

        arrays = load_subset()
        self.assertEqual(set(arrays), expected_members)
        forbidden_fragments = ("continuum", "golden", "opacity", "output")
        for name in arrays:
            self.assertFalse(any(fragment in name for fragment in forbidden_fragments))

    def test_all_seventeen_raw_fields_preserve_values_dtypes_and_bytes(self) -> None:
        arrays = load_subset()
        self.assertEqual(set(RAW_EXPECTED), set(subset_builder.RAW_FIELDS))
        for name, (expected_dtype, expected_value) in RAW_EXPECTED.items():
            with self.subTest(name=name):
                values = arrays[name]
                self.assertEqual(values.shape, (1,))
                self.assertEqual(values.dtype, expected_dtype)
                self.assertEqual(values[0].item(), expected_value)
        self.assertEqual(arrays["energy_shift_field"][0].tobytes(), b" " * 10)
        self.assertEqual(arrays["line_category_tag"].tobytes(), b"\x00" * 3)

    def test_provenance_pins_source_row_archive_and_static_role(self) -> None:
        arrays = load_subset()
        self.assertEqual(int(arrays["subset_schema_version"]), 1)
        self.assertEqual(int(arrays["source_row_index"]), 873702)
        self.assertEqual(int(arrays["source_field_count"]), 17)
        self.assertEqual(int(arrays["source_archive_row_count"]), 1_939_975)
        self.assertEqual(int(arrays["source_archive_bytes"]), 258_021_389)
        self.assertEqual(
            str(arrays["source_archive_sha256"]),
            subset_builder.SOURCE_ARCHIVE_SHA256,
        )
        self.assertEqual(
            str(arrays["source_archive_relative_path"]),
            subset_builder.SOURCE_ARCHIVE_RELATIVE_PATH.as_posix(),
        )
        self.assertEqual(
            str(arrays["payne_zero_commit"]),
            subset_builder.PAYNE_ZERO_COMMIT,
        )
        self.assertIn("teaching subset", str(arrays["subset_role"]))
        self.assertIn("no computed outputs", str(arrays["subset_role"]))

    def test_builder_reextracts_identical_bytes_without_changing_source(self) -> None:
        if not SOURCE_ARCHIVE.is_file():
            self.skipTest("optional full atomic source archive is not installed")
        source_hash_before = sha256(SOURCE_ARCHIVE)
        self.assertEqual(source_hash_before, subset_builder.SOURCE_ARCHIVE_SHA256)
        with tempfile.TemporaryDirectory() as temporary_directory:
            candidate = Path(temporary_directory) / SUBSET_PATH.name
            write_npz(candidate, subset_builder.build_arrays(SOURCE_ARCHIVE))
            self.assertEqual(candidate.read_bytes(), SUBSET_PATH.read_bytes())
        self.assertEqual(sha256(SOURCE_ARCHIVE), source_hash_before)

    def test_zero_corrections_and_exact_derived_record(self) -> None:
        arrays = load_subset()
        raw = {name: arrays[name] for name in subset_builder.RAW_FIELDS}
        first_shift, second_shift = atomic_lines._parse_energy_shift_subfields(
            raw["energy_shift_field"]
        )
        np.testing.assert_array_equal(first_shift, np.zeros(1, dtype=np.float64))
        np.testing.assert_array_equal(second_shift, np.zeros(1, dtype=np.float64))
        self.assertEqual(float(raw["primary_isotope_log_correction"][0]), 0.0)
        self.assertEqual(float(raw["secondary_isotope_log_correction"][0]), 0.0)
        self.assertEqual(float(raw["isotope_shift_units"][0]), 0.0)
        self.assertTrue(
            all(
                float(raw[name][0]) != 0.0
                for name in (
                    "radiative_damping_log",
                    "stark_damping_log",
                    "van_der_waals_damping_log",
                )
            )
        )

        records = atomic_lines._build_records(raw)
        exact_values = {
            "wavelength_nm": 499.03411946178176,
            "index_wavelength_nm": 499.03411946178176,
            "log_oscillator_strength": -0.86,
            "oscillator_strength": 0.1380384264602885,
            "lower_excitation_cm": 33507.123,
            "radiative_damping": 3.909296919359072e-08,
            "stark_damping": 4.700008650504819e-21,
            "van_der_waals_damping": 4.093536407068315e-24,
        }
        for name, expected in exact_values.items():
            with self.subTest(name=name):
                self.assertEqual(float(records[name][0]), expected)
        self.assertEqual(int(records["atomic_number"][0]), 26)
        self.assertEqual(int(records["ion_stage"][0]), 1)
        self.assertEqual(int(records["line_type"][0]), 0)

        frequency_hz = (
            atomic_lines.LIGHT_SPEED_NM_PER_S / records["wavelength_nm"][0]
        )
        classical_strength = (
            atomic_lines.CLASSICAL_LINE_STRENGTH_COEFFICIENT
            * records["oscillator_strength"][0]
            / frequency_hz
        )
        self.assertEqual(classical_strength, 3.4403587659200408e-18)

    def test_manifest_owns_subset_and_every_member(self) -> None:
        manifest = json.loads((REPOSITORY_ROOT / "data/MANIFEST.json").read_text())
        entries = {entry["path"]: entry for entry in manifest["entries"]}
        relative_path = str(SUBSET_PATH.relative_to(REPOSITORY_ROOT))
        entry = entries[relative_path]
        self.assertEqual(entry["role"], "subset")
        self.assertEqual(entry["source_commit"], subset_builder.PAYNE_ZERO_COMMIT)
        self.assertEqual(
            entry["source_sha256"],
            subset_builder.SOURCE_ARCHIVE_SHA256,
        )
        self.assertEqual(entry["source_row_index"], 873702)
        self.assertEqual(entry["source_field_count"], 17)
        self.assertEqual(entry["subset_schema_version"], 1)
        self.assertTrue(entry["requires_optional_full_catalog"])
        self.assertEqual(entry["builder"], str(BUILDER_PATH.relative_to(REPOSITORY_ROOT)))
        self.assertEqual(entry["builder_sha256"], BUILDER_SHA256)
        self.assertEqual(entry["sha256"], SUBSET_SHA256)
        self.assertEqual(entry["bytes"], SUBSET_BYTES)

        arrays = load_subset()
        self.assertEqual(set(entry["arrays"]), set(arrays))
        for name, values in arrays.items():
            with self.subTest(name=name):
                metadata = entry["arrays"][name]
                self.assertEqual(metadata["shape"], list(values.shape))
                self.assertEqual(metadata["dtype"], str(values.dtype))
                self.assertTrue(metadata["unit"])
                self.assertEqual(len(metadata["axes"]), values.ndim)
                self.assertTrue(metadata["ownership"])

    def test_exact_route_is_visible_in_every_regime_without_hand_tuning(
        self,
    ) -> None:
        arrays = load_subset()
        raw = {name: arrays[name] for name in subset_builder.RAW_FIELDS}
        records = atomic_lines._build_records(raw)
        wavelength_grid_nm = atomic_lines.Grid(495.0, 505.0, 300_000.0).build()
        invariants = line_opacity.precompute_invariants(
            one_record_catalog(records),
            wavelength_grid_nm,
            runtime_device="cpu",
        )
        self.assertEqual(int(invariants.metal_catalog_index.numel()), 1)
        self.assertEqual(int(invariants.auto_catalog_index.numel()), 0)
        self.assertEqual(int(invariants.helium_line_type.numel()), 0)
        self.assertEqual(int(invariants.metal_population_ion_stage_index[0]), 0)
        self.assertEqual(int(invariants.metal_population_element_index[0]), 25)

        for regime, expected_mask in REGIME_ACTIVE_MASKS.items():
            with self.subTest(regime=regime):
                state = load_regime_state(regime, "synthesis")
                continuum = run_synthesis_continuum(
                    regime,
                    wavelength_grid_nm,
                ).continuum_opacity
                collision_density_proxy = (
                    state["hydrogen_neutral_population"]
                    + 0.42 * state["helium_neutral_population"]
                    + 0.85 * state["molecular_hydrogen_population"]
                ) * (state["temperature"] / 1.0e4) ** 0.3
                atomic_state = {
                    name: state[name]
                    for name in (
                        "partition_normalized_populations",
                        "fractional_doppler_widths",
                        "mass_density",
                        "electron_density",
                        "temperature",
                        "hc_over_kt",
                    )
                }
                atomic_state["collision_density_proxy"] = collision_density_proxy
                atomic_state["continuum_opacity"] = continuum
                gross_opacity = line_opacity.accumulate_atomic(
                    invariants,
                    atomic_state,
                    do_metal=True,
                    do_helium=False,
                    apply_stim=False,
                    wing_mode="batched",
                )
                self.assertEqual(gross_opacity.shape, (6, wavelength_grid_nm.size))
                self.assertEqual(gross_opacity.dtype, torch.float32)
                active_mask = tuple(
                    bool(value)
                    for value in torch.any(gross_opacity > 0.0, dim=1).tolist()
                )
                self.assertEqual(active_mask, expected_mask)
                self.assertGreater(sum(active_mask), 0)


if __name__ == "__main__":
    unittest.main()
