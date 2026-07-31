"""Focused source, role-split data, and import checks for Chapter 3."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_ROOT = Path("/Users/ysting/payne-zero")
PINNED_MIXED_EOS = (
    PINNED_ROOT
    / "source_data_files"
    / "synthesis_tables"
    / "partition_saha_inputs.npz"
)
STATIC_SYNTHESIS_EOS = (
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "synthesis_tables"
    / "partition_saha_tables.npz"
)
FIXTURE_SYNTHESIS_EOS = (
    REPOSITORY_ROOT
    / "data"
    / "fixtures"
    / "chapter03_synthesis_eos_state.npz"
)
ATOMIC_MASSES = (
    REPOSITORY_ROOT
    / "data"
    / "static"
    / "synthesis_tables"
    / "atomic_masses.npz"
)


class Chapter03SourceDataTests(unittest.TestCase):
    """Keep the atomic EOS layer exact, importable, and role honest."""

    def test_dependency_complete_modules_import(self) -> None:
        expected_symbols = {
            "payne_zero_atmosphere.equation_of_state": "saha_partition_depth",
            "payne_zero_atmosphere.population_layout": "population_job_schedule",
            "payne_zero_atmosphere.runtime_state": "AtmosphereRuntimeState",
            "payne_zero_atmosphere.doppler": "update_doppler_line_strength_factors",
            "payne_zero_atmosphere.specific_internal_energy": (
                "compute_atomic_specific_internal_energy"
            ),
            "payne_zero_atmosphere.synthesis_bridge": "_packed_atomic_cube",
            "payne_zero_synthesis.equation_of_state": (
                "solve_population_state_at_electron_density"
            ),
            "payne_zero_synthesis.ground_partition_table": "ground_partition_value",
            "payne_zero_synthesis.paths": "data_root",
            "payne_zero_synthesis.pipeline": "compute_doppler_per_ion",
            "payne_zero_synthesis.synthesis": "compute_mean_nuclear_mass_amu",
        }
        for module_name, symbol in expected_symbols.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, symbol))

    @unittest.skipUnless(PINNED_MIXED_EOS.is_file(), "pinned source checkout absent")
    def test_role_split_is_array_identical_and_exhaustive(self) -> None:
        with (
            np.load(PINNED_MIXED_EOS, allow_pickle=False) as source,
            np.load(STATIC_SYNTHESIS_EOS, allow_pickle=False) as static,
            np.load(FIXTURE_SYNTHESIS_EOS, allow_pickle=False) as fixture,
        ):
            self.assertFalse(set(static.files).intersection(fixture.files))
            self.assertEqual(
                set(source.files), set(static.files).union(fixture.files)
            )
            for name in source.files:
                local = static[name] if name in static.files else fixture[name]
                with self.subTest(array=name):
                    self.assertEqual(local.dtype, source[name].dtype)
                    self.assertEqual(local.shape, source[name].shape)
                    np.testing.assert_array_equal(local, source[name])

    def test_ground_field_remains_fixture_data(self) -> None:
        with (
            np.load(STATIC_SYNTHESIS_EOS, allow_pickle=False) as static,
            np.load(FIXTURE_SYNTHESIS_EOS, allow_pickle=False) as fixture,
        ):
            self.assertNotIn("ground_partition_table", static.files)
            self.assertIn("ground_partition_table", fixture.files)
            self.assertEqual(fixture["ground_partition_table"].shape, (605, 80))

    def test_atmosphere_exact_loaders_resolve_local_static_tables(self) -> None:
        from payne_zero_atmosphere.equation_of_state import (
            load_iron_group_partition_grid,
            load_ionization_potential_table_cm,
            load_packed_level_metadata,
            load_special_partition_tables,
        )
        from payne_zero_atmosphere.runtime_state import (
            load_major_isotope_masses_amu,
        )

        data_root = REPOSITORY_ROOT / "data" / "static"
        for loader in (
            load_iron_group_partition_grid,
            load_ionization_potential_table_cm,
            load_packed_level_metadata,
            load_special_partition_tables,
        ):
            loader.cache_clear()
        with mock.patch.dict(
            "os.environ",
            {"PAYNE_ZERO_DATA_ROOT": str(data_root)},
        ):
            self.assertEqual(
                load_iron_group_partition_grid().shape, (7, 56, 10, 9)
            )
            self.assertEqual(load_ionization_potential_table_cm().shape, (999,))
            self.assertEqual(load_packed_level_metadata().shape, (6, 365))
            special = load_special_partition_tables()
            self.assertEqual(special.element_block_offsets.shape, (29,))
        isotope_path = (
            data_root / "atmosphere_tables" / "isotope_tables.npz"
        )
        self.assertEqual(
            load_major_isotope_masses_amu(isotope_path).shape, (1006,)
        )

    def test_role_split_builds_exact_cpu_tables_and_fixed_ne_state(self) -> None:
        from payne_zero_synthesis.equation_of_state import (
            EOSTables,
            solve_population_state_at_electron_density,
        )

        with (
            np.load(STATIC_SYNTHESIS_EOS, allow_pickle=False) as static,
            np.load(FIXTURE_SYNTHESIS_EOS, allow_pickle=False) as fixture,
        ):
            table_input = {
                name: np.asarray(static[name]).copy() for name in static.files
            }
            table_input["ground_partition_table"] = np.asarray(
                fixture["ground_partition_table"]
            ).copy()
            temperature = np.asarray(fixture["temperature"][:2])
            gas_pressure = np.asarray(fixture["gas_pressure"][:2])
            electron_density = np.asarray(fixture["electron_density"][:2])
            elemental_abundances = np.asarray(fixture["elemental_abundances"])

        tables = EOSTables.from_dict(
            table_input,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        self.assertEqual(tuple(tables.packed_partition_table.shape), (6, 374))
        self.assertEqual(
            tuple(tables.iron_group_partition_grid.shape), (7, 56, 10, 9)
        )

        with mock.patch.dict(
            "os.environ",
            {"PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE": str(ATOMIC_MASSES)},
        ):
            state = solve_population_state_at_electron_density(
                temperature,
                gas_pressure,
                elemental_abundances,
                tables=tables,
                electron_density=electron_density,
                molecules=False,
            )
        np.testing.assert_array_equal(state.electron_density, electron_density)
        self.assertEqual(state.ion_stage_populations.shape, (2, 6, 139))
        self.assertEqual(
            state.partition_normalized_populations.shape, (2, 6, 139)
        )

    def test_packed_bridge_maps_only_atomic_public_columns(self) -> None:
        from payne_zero_atmosphere.synthesis_bridge import _packed_atomic_cube

        packed = np.arange(1, 1007, dtype=np.float64)[None, :]
        actual, normalized, doppler = _packed_atomic_cube(
            ion_stage_populations_by_packed_slot=packed,
            partition_normalized_populations_by_packed_slot=packed + 2000.0,
            fractional_doppler_widths_by_packed_slot=packed + 4000.0,
        )
        self.assertEqual(actual.shape, (1, 6, 139))
        self.assertEqual(normalized.shape, (1, 6, 139))
        self.assertEqual(doppler.shape, (1, 6, 139))
        self.assertEqual(actual[0, 0, 0], 1.0)
        self.assertEqual(normalized[0, 0, 0], 2001.0)
        self.assertTrue(np.all(actual[:, :, 99:] == 0.0))
        self.assertTrue(np.all(normalized[:, :, 99:] == 0.0))
        self.assertTrue(np.all(doppler[:, :, 99:] == 0.0))

    def test_synthesis_doppler_uses_six_by_139_public_axes(self) -> None:
        from payne_zero_synthesis.pipeline import compute_doppler_per_ion

        with np.load(ATOMIC_MASSES, allow_pickle=False) as archive:
            masses = np.asarray(archive["atomic_mass_amu"])
        temperature = np.array([5000.0, 10000.0])
        microturbulence = np.array([2.0e5, 2.0e5])
        widths = compute_doppler_per_ion(
            temperature,
            microturbulence,
            masses,
        )
        self.assertEqual(widths.shape, (2, 6, 139))
        np.testing.assert_array_equal(widths[:, 0, :99], widths[:, 5, :99])
        self.assertTrue(np.all(widths[:, :, 99:] == 0.0))
        self.assertTrue(np.all(widths[1, :, :99] > widths[0, :, :99]))

    def test_synthesis_mean_mass_normalizes_the_abundance_scale(self) -> None:
        from payne_zero_synthesis.synthesis import compute_mean_nuclear_mass_amu

        with np.load(ATOMIC_MASSES, allow_pickle=False) as archive:
            masses = np.asarray(archive["atomic_mass_amu"])
        abundances = np.zeros(99, dtype=np.float64)
        abundances[:2] = (0.9, 0.1)
        expected = float(np.sum(abundances * masses) / np.sum(abundances))
        self.assertEqual(
            compute_mean_nuclear_mass_amu(abundances, masses),
            expected,
        )
        self.assertEqual(
            compute_mean_nuclear_mass_amu(7.0 * abundances, masses),
            expected,
        )

    def test_copied_static_file_hashes_match_pinned_contract(self) -> None:
        expected = {
            "special_partition_tables.npz": (
                "7d737524aacda1cc2281e5b18ff49f240ca34665dbe6c96d4dd0f39db4aedd22"
            ),
            "iron_group_partition_tables.npz": (
                "137629dea64eca46f77ea3656c18305ade912a468d7eb27029544c0106cc3296"
            ),
            "ionization_potential_tables.npz": (
                "82a2e82f2015da02c3d2bce77ca5337aa2b9c4e23d8d6219da07895896ca8a50"
            ),
            "packed_level_metadata.npz": (
                "de5f17b6a9eaec1d1b07e96fd02ff014279cd8eaa9f976fefde0e2a153961bc3"
            ),
            "isotope_tables.npz": (
                "53c8d315fb53f1e051dc2752b028fc270d7c17a2c1042279c04ffcb750aef5c6"
            ),
        }
        table_dir = REPOSITORY_ROOT / "data" / "static" / "atmosphere_tables"
        for name, expected_sha256 in expected.items():
            with self.subTest(file=name):
                actual = hashlib.sha256((table_dir / name).read_bytes()).hexdigest()
                self.assertEqual(actual, expected_sha256)
        self.assertEqual(
            hashlib.sha256(ATOMIC_MASSES.read_bytes()).hexdigest(),
            "d4739fef7e03964aea5a7b2604f9585fd9095c26c58f5b7d5d040aaafeb5d117",
        )


if __name__ == "__main__":
    unittest.main()
