"""Focused exact-source and atom-only import gates for the Chapter 3 handoff."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_ROOT = Path("/Users/ysting/payne-zero")
STATIC_DATA_ROOT = REPOSITORY_ROOT / "data" / "static"


def top_level_definition(path: Path, name: str) -> ast.AST:
    """Return one named top-level function or class."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} is missing from {path}")


class Chapter03ExactHandoffSourceTests(unittest.TestCase):
    """Keep the progressive handoff exact and executable without molecules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_data_root = os.environ.get("PAYNE_ZERO_DATA_ROOT")
        os.environ["PAYNE_ZERO_DATA_ROOT"] = str(STATIC_DATA_ROOT)
        from payne_zero_atmosphere import runtime_state

        cls.runtime_state_module = runtime_state
        cls.previous_isotope_table_path = runtime_state._ISOTOPE_TABLE_PATH
        runtime_state._ISOTOPE_TABLE_PATH = (
            STATIC_DATA_ROOT / "atmosphere_tables" / "isotope_tables.npz"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.runtime_state_module._ISOTOPE_TABLE_PATH = (
            cls.previous_isotope_table_path
        )
        if cls.previous_data_root is None:
            os.environ.pop("PAYNE_ZERO_DATA_ROOT", None)
        else:
            os.environ["PAYNE_ZERO_DATA_ROOT"] = cls.previous_data_root

    def test_full_setup_dependencies_are_byte_identical(self) -> None:
        for name in ("run_setup.py", "microturbulence.py"):
            with self.subTest(module=name):
                local = REPOSITORY_ROOT / "src" / "payne_zero_atmosphere" / name
                source = PINNED_ROOT / "payne_zero_atmosphere" / name
                self.assertEqual(local.read_bytes(), source.read_bytes())

    def test_handoff_definitions_are_ast_exact(self) -> None:
        local = (
            REPOSITORY_ROOT / "src" / "payne_zero_atmosphere" / "runner.py"
        )
        source = PINNED_ROOT / "payne_zero_atmosphere" / "runner.py"
        for name in (
            "AtmospherePopulationState",
            "prepare_structured_handoff_population_state",
        ):
            with self.subTest(definition=name):
                local_ast = ast.dump(
                    top_level_definition(local, name), include_attributes=False
                )
                source_ast = ast.dump(
                    top_level_definition(source, name), include_attributes=False
                )
                self.assertEqual(local_ast, source_ast)

    def test_atom_only_handoff_executes_at_fixed_electron_density(self) -> None:
        from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere
        from payne_zero_atmosphere.config import (
            AtmosphereConfig,
            AtmosphereInput,
            AtmosphereOutput,
        )
        from payne_zero_atmosphere.runner import (
            prepare_structured_handoff_population_state,
        )

        fixture_path = (
            REPOSITORY_ROOT
            / "data"
            / "fixtures"
            / "chapter03_atom_only_inputs.npz"
        )
        with np.load(fixture_path, allow_pickle=False) as archive:
            inputs = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
        temperature = np.asarray(inputs["temperature"], np.float64)
        abundances = np.asarray(inputs["elemental_abundances"], np.float64)
        fixed_abundances = {
            atomic_number: (
                float(abundances[atomic_number - 1])
                if atomic_number <= 2
                else float(np.log10(abundances[atomic_number - 1]))
            )
            for atomic_number in range(1, 100)
        }
        zeros = np.zeros_like(temperature)
        atmosphere = ModelAtmosphere(
            column_mass=np.asarray(inputs["column_mass"], np.float64),
            temperature=temperature,
            gas_pressure=np.asarray(inputs["gas_pressure"], np.float64),
            electron_density=np.asarray(
                inputs["electron_density_seed"], np.float64
            ),
            rosseland_opacity=np.ones_like(temperature),
            radiative_acceleration=zeros.copy(),
            microturbulence=np.asarray(inputs["microturbulence"], np.float64),
            convective_flux=zeros.copy(),
            convective_velocity=zeros.copy(),
            metadata={
                "effective_temperature": "5778",
                "log_surface_gravity": "4.44",
            },
            fixed_column_abundance_values=fixed_abundances,
        )
        electron_density = atmosphere.electron_density.copy()
        config = AtmosphereConfig(
            inputs=AtmosphereInput(initial_atmosphere=atmosphere),
            outputs=AtmosphereOutput(),
            enable_molecules=False,
        )

        handoff = prepare_structured_handoff_population_state(config)

        populations = (
            handoff.runtime_state.ion_stage_populations_by_packed_slot
        )
        active_slots = np.any(populations > 0.0, axis=0)
        self.assertFalse(handoff.setup.molecules_enabled)
        np.testing.assert_array_equal(
            handoff.runtime_state.electron_density, electron_density
        )
        self.assertEqual(populations.shape, (6, 1006))
        self.assertEqual(handoff.fractional_doppler_widths.shape, (6, 1006))
        self.assertTrue(
            np.all(
                np.isfinite(
                    handoff.fractional_doppler_widths[:, active_slots]
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
