"""Exact source, catalog, and fixture gates for Chapter 4."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from payne_zero_atmosphere.config import AtmosphereConfig
from payne_zero_atmosphere.molecular_data import (
    read_molecular_equilibrium_catalog,
)
from payne_zero_synthesis.molecular_equilibrium import (
    _ATOMIC_MASSES_FOR_MOLECULES,
    read_molecule_table,
    solve_molecular_equilibrium,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ATMOSPHERE_CATALOG_PATH = (
    REPOSITORY_ROOT
    / "data/static/source_catalogs/lines/"
    / "molecular_equilibrium_atmosphere.npz"
)
SYNTHESIS_CATALOG_PATH = (
    REPOSITORY_ROOT
    / "data/static/source_catalogs/lines/"
    / "molecular_equilibrium_synthesis.npz"
)
MOLECULAR_TABLE_PATH = (
    REPOSITORY_ROOT / "data/static/atmosphere_tables/molecular_equilibrium_tables.npz"
)
CONTINUUM_EDGE_PATH = (
    REPOSITORY_ROOT / "data/static/synthesis_tables/continuum_edge_grid.npz"
)
FIXTURE_PATH = REPOSITORY_ROOT / "data/fixtures/chapter04_molecular_inputs.npz"
PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"

EXPECTED_FILE_HASHES = {
    ATMOSPHERE_CATALOG_PATH: (
        "971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644"
    ),
    SYNTHESIS_CATALOG_PATH: (
        "3e8c1ea69fe672b9886bda38922f868c6d2ac2b43c4eb0d7750620241c238d28"
    ),
    MOLECULAR_TABLE_PATH: (
        "1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe"
    ),
    CONTINUUM_EDGE_PATH: (
        "11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e"
    ),
    FIXTURE_PATH: ("351bba75dca1fa502f5cc2a108035f69f2e31c760a90133480f2e7fe31e45f79"),
}

EXPECTED_SYNTHESIS_ONLY_CODES = {
    111,
    10811,
    10812,
    10820,
    60606,
    60608,
    60614,
    60816,
    61414,
    61616,
    70708,
    70808,
    80814,
    80816,
    1010106,
    1010107,
    1010606,
    6060707,
    101010106,
    101010114,
}


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_count(archive: np.lib.npyio.NpzFile) -> int:
    """Return the catalog's scalar active molecule count."""

    return int(np.asarray(archive["molecule_count"]).item())


def component_signature(
    archive: np.lib.npyio.NpzFile,
    molecule_index: int,
) -> tuple[int, ...]:
    """Decode one catalog row into its equation-species sequence."""

    starts = archive["component_start_indices"]
    start = int(starts[molecule_index])
    stop = int(starts[molecule_index + 1])
    equation_indices = archive["component_equation_indices"][start:stop]
    return tuple(
        int(value) for value in archive["equation_species_codes"][equation_indices]
    )


def physical_record_counter(
    archive: np.lib.npyio.NpzFile,
) -> Counter[tuple[int, tuple[int, ...], tuple[float, ...]]]:
    """Key records by code, decoded components, and exact coefficients."""

    records = []
    for index in range(active_count(archive)):
        code = int(round(float(archive["molecule_codes"][index])))
        coefficients = tuple(
            float(value) for value in archive["equilibrium_coefficients"][:, index]
        )
        records.append((code, component_signature(archive, index), coefficients))
    return Counter(records)


class Chapter04SourceDataTests(unittest.TestCase):
    """Freeze the exact molecular inputs before any chapter prose is drafted."""

    def test_source_derived_files_have_pinned_byte_identities(self) -> None:
        for path, expected in EXPECTED_FILE_HASHES.items():
            with self.subTest(path=path.name):
                self.assertEqual(sha256(path), expected)

    def test_data_manifest_owns_every_chapter04_asset(self) -> None:
        manifest = json.loads((REPOSITORY_ROOT / "data/MANIFEST.json").read_text())
        entries = {entry["path"]: entry for entry in manifest["entries"]}
        for path, expected_hash in EXPECTED_FILE_HASHES.items():
            relative = str(path.relative_to(REPOSITORY_ROOT))
            with self.subTest(path=relative):
                entry = entries[relative]
                self.assertEqual(entry["sha256"], expected_hash)
                self.assertFalse(entry["requires_optional_full_catalog"])
                if entry["role"] == "static":
                    self.assertEqual(entry["source_commit"], PINNED_COMMIT)
                    self.assertEqual(entry["source_sha256"], expected_hash)
                else:
                    self.assertEqual(entry["role"], "fixture")
                    self.assertEqual(
                        entry["builder"],
                        "scripts/build_chapter04_molecular_fixture.py",
                    )

    def test_exact_loaders_preserve_active_counts_and_fixed_buffers(self) -> None:
        atmosphere = read_molecular_equilibrium_catalog(ATMOSPHERE_CATALOG_PATH)
        synthesis = read_molecule_table(SYNTHESIS_CATALOG_PATH)
        self.assertEqual(
            (
                atmosphere.molecule_count,
                atmosphere.equation_count,
                atmosphere.component_count,
            ),
            (170, 23, 481),
        )
        self.assertEqual(
            (
                synthesis.molecule_count,
                synthesis.equation_count,
                int(synthesis.component_start_indices[190]),
            ),
            (190, 23, 548),
        )
        self.assertEqual(atmosphere.molecule_codes.shape, (200,))
        self.assertEqual(synthesis.molecule_codes.shape, (200,))
        self.assertEqual(atmosphere.equilibrium_coefficients.shape, (7, 200))
        self.assertEqual(synthesis.equilibrium_coefficients.shape, (7, 200))
        self.assertEqual(atmosphere.component_start_indices.shape, (201,))
        self.assertEqual(synthesis.component_start_indices.shape, (201,))
        self.assertEqual(atmosphere.component_equation_indices.shape, (600,))
        self.assertEqual(synthesis.component_equation_indices.shape, (600,))

    def test_catalog_bounds_padding_and_atmosphere_seventh_row(self) -> None:
        for path, molecule_count, component_count in (
            (ATMOSPHERE_CATALOG_PATH, 170, 481),
            (SYNTHESIS_CATALOG_PATH, 190, 548),
        ):
            with (
                self.subTest(path=path.name),
                np.load(path, allow_pickle=False) as archive,
            ):
                starts = archive["component_start_indices"]
                indices = archive["component_equation_indices"]
                equation_count = int(archive["equation_count"])
                self.assertEqual(int(starts[0]), 0)
                self.assertEqual(int(starts[molecule_count]), component_count)
                self.assertTrue(np.all(np.diff(starts[: molecule_count + 1]) >= 0))
                self.assertTrue(np.all(indices[:component_count] >= 0))
                self.assertTrue(np.all(indices[:component_count] <= equation_count))
                self.assertFalse(np.any(archive["molecule_codes"][molecule_count:]))
                self.assertFalse(
                    np.any(archive["equilibrium_coefficients"][:, molecule_count:])
                )
                self.assertFalse(np.any(starts[molecule_count + 1 :]))
                self.assertFalse(np.any(indices[component_count:]))

        with np.load(ATMOSPHERE_CATALOG_PATH, allow_pickle=False) as atmosphere:
            self.assertFalse(np.any(atmosphere["equilibrium_coefficients"][6, :170]))
            self.assertEqual(
                tuple(atmosphere["equation_species_codes"][:23]),
                (
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    11,
                    12,
                    13,
                    14,
                    16,
                    17,
                    19,
                    20,
                    22,
                    23,
                    24,
                    26,
                    100,
                ),
            )
            reverse = atmosphere["species_to_equation_index"]
            self.assertEqual(np.count_nonzero(reverse == -1), 79)
            self.assertEqual(int(reverse[100]), 22)
            self.assertEqual(int(reverse[101]), 23)

    def test_catalogs_share_170_records_by_semantics_not_row_order(self) -> None:
        with np.load(ATMOSPHERE_CATALOG_PATH, allow_pickle=False) as atmosphere:
            atmosphere_records = physical_record_counter(atmosphere)
        with np.load(SYNTHESIS_CATALOG_PATH, allow_pickle=False) as synthesis:
            synthesis_records = physical_record_counter(synthesis)

        self.assertEqual(
            sum((atmosphere_records & synthesis_records).values()),
            170,
        )
        self.assertFalse(atmosphere_records - synthesis_records)
        synthesis_only = synthesis_records - atmosphere_records
        self.assertEqual(sum(synthesis_only.values()), 20)
        self.assertEqual(
            {record[0] for record in synthesis_only},
            EXPECTED_SYNTHESIS_ONLY_CODES,
        )

        with np.load(ATMOSPHERE_CATALOG_PATH, allow_pickle=False) as atmosphere:
            atmosphere_codes = atmosphere["molecule_codes"][:170]
        with np.load(SYNTHESIS_CATALOG_PATH, allow_pickle=False) as synthesis:
            synthesis_codes = synthesis["molecule_codes"][:170]
        self.assertFalse(np.array_equal(atmosphere_codes, synthesis_codes))

    def test_fourteen_negative_ions_end_in_an_ordinary_electron(self) -> None:
        with np.load(ATMOSPHERE_CATALOG_PATH, allow_pickle=False) as atmosphere:
            signatures = [
                component_signature(atmosphere, index)
                for index in range(active_count(atmosphere))
            ]
        negative_ion_signatures = [
            signature for signature in signatures if 100 in signature
        ]
        self.assertEqual(len(negative_ion_signatures), 14)
        for signature in negative_ion_signatures:
            with self.subTest(signature=signature):
                self.assertEqual(signature[-1], 100)
                self.assertNotIn(0, signature)
                self.assertEqual(signature.count(100), 1)

    def test_h2_table_is_finite_positive_and_keeps_its_exact_shape(self) -> None:
        with np.load(MOLECULAR_TABLE_PATH, allow_pickle=False) as archive:
            masses = archive["atomic_mass_amu"]
            h2_partition = archive["h2_partition_function"]
        self.assertEqual(masses.shape, (99,))
        self.assertEqual(h2_partition.shape, (200,))
        self.assertTrue(np.all(np.isfinite(masses)))
        self.assertTrue(np.all(np.isfinite(h2_partition)))
        self.assertTrue(np.all(masses > 0.0))
        self.assertTrue(np.all(h2_partition > 0.0))

    def test_structured_builder_edge_grid_is_exact_and_exhaustive(self) -> None:
        with np.load(CONTINUUM_EDGE_PATH, allow_pickle=False) as archive:
            self.assertEqual(
                set(archive.files),
                {
                    "__meta__",
                    "signed_continuum_edge_frequency_hz",
                    "continuum_edge_wavelength_nm",
                    "continuum_edge_wavenumber_cm",
                    "continuum_edge_sample_frequency_hz",
                    "continuum_edge_midpoint_wavelength_nm",
                    "continuum_edge_interval_width_squared_over_two_nm2",
                },
            )
            self.assertEqual(
                archive["signed_continuum_edge_frequency_hz"].shape,
                (341,),
            )
            self.assertEqual(
                archive["continuum_edge_sample_frequency_hz"].shape,
                (1020,),
            )
            self.assertEqual(
                archive["continuum_edge_midpoint_wavelength_nm"].shape,
                (340,),
            )
            self.assertEqual(
                archive["continuum_edge_interval_width_squared_over_two_nm2"].shape,
                (340,),
            )
            metadata = json.loads(bytes(archive["__meta__"]).decode())
        self.assertEqual(metadata["schema"], 1)
        self.assertIn("continuum-edge", metadata["description"])

    def test_synthesis_hard_coded_molecular_masses_and_dtype_default(self) -> None:
        masses = np.ascontiguousarray(_ATOMIC_MASSES_FOR_MOLECULES)
        self.assertEqual(masses.shape, (99,))
        self.assertEqual(masses.dtype, np.float64)
        self.assertEqual(
            hashlib.sha256(masses.tobytes()).hexdigest(),
            "b6b870f0cdb3ea49fc4977dfc1fcff0ed1c16747922940d302644aefe28b7636",
        )
        signature = inspect.signature(solve_molecular_equilibrium)
        self.assertIs(signature.parameters["dtype"].default, torch.float32)

    def test_molecular_energy_tracking_defaults_and_forwarding_are_pinned(self) -> None:
        public_default = AtmosphereConfig.__dataclass_fields__[
            "molecular_convection_thermal_tracks_perturbation"
        ].default
        self.assertIs(public_default, True)

        manifest = json.loads(
            (REPOSITORY_ROOT / "src/PAYNE_ZERO_SOURCE_MANIFEST.json").read_text()
        )
        runner_path = Path(manifest["source_root"]) / "payne_zero_atmosphere/runner.py"
        tree = ast.parse(runner_path.read_text())
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        for function_name, parameter_name in (
            (
                "compute_convection_finite_difference_samples",
                "molecular_thermal_energy_tracks_perturbation",
            ),
            (
                "finalize_transfer_state",
                "molecular_convection_thermal_tracks_perturbation",
            ),
        ):
            function = functions[function_name]
            defaults = dict(
                zip(
                    (argument.arg for argument in function.args.kwonlyargs),
                    function.args.kw_defaults,
                )
            )
            self.assertIsInstance(defaults[parameter_name], ast.Constant)
            self.assertIs(defaults[parameter_name].value, False)

        runner = functions["run_atmosphere_model"]
        finalizer_calls = [
            node
            for node in ast.walk(runner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "finalize_transfer_state"
        ]
        self.assertEqual(len(finalizer_calls), 1)
        forwarded = {
            keyword.arg: keyword.value for keyword in finalizer_calls[0].keywords
        }
        self.assertEqual(
            ast.unparse(forwarded["molecular_convection_thermal_tracks_perturbation"]),
            "bool(config.molecular_convection_thermal_tracks_perturbation)",
        )

    def test_fixture_is_input_only_monotone_and_rebuildable(self) -> None:
        with np.load(FIXTURE_PATH, allow_pickle=False) as fixture:
            names = set(fixture.files)
            temperature = fixture["temperature"]
            gas_pressure = fixture["gas_pressure"]
            column_mass = fixture["column_mass"]
            electron_seed = fixture["electron_density_seed"]
            source_identity = str(fixture["source_abundance_fixture_sha256"].item())
        for forbidden in ("population", "residual", "converged", "output"):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(any(forbidden in name for name in names))
        self.assertTrue(np.all(np.diff(temperature) > 0.0))
        self.assertTrue(np.all(np.diff(gas_pressure) > 0.0))
        self.assertTrue(np.all(np.diff(column_mass) > 0.0))
        expected_seed = 0.1 * gas_pressure / (1.38054e-16 * temperature)
        np.testing.assert_array_equal(electron_seed, expected_seed)
        self.assertEqual(
            source_identity,
            "3ed0d65431fc9e284a77011b82267241b25cc56cdffa73e1bc86eec15f9b5219",
        )

    def test_fixture_straddles_every_exact_temperature_boundary(self) -> None:
        with np.load(FIXTURE_PATH, allow_pickle=False) as fixture:
            atmosphere_polynomial = fixture[
                "atmosphere_polynomial_boundary_temperature"
            ]
            synthesis_polynomial = fixture["synthesis_polynomial_boundary_temperature"]
            synthesis_h2 = fixture["synthesis_named_h2_boundary_temperature"]
            atmosphere_h2 = fixture["atmosphere_h2_boundary_temperature"]

        np.testing.assert_array_equal(
            atmosphere_polynomial,
            np.array([10000.0, np.nextafter(10000.0, np.inf)]),
        )
        np.testing.assert_array_equal(
            synthesis_polynomial,
            np.array([10000.0, np.nextafter(10000.0, np.inf)]),
        )
        np.testing.assert_array_equal(
            synthesis_h2,
            np.array([9000.0, np.nextafter(9000.0, np.inf)]),
        )
        np.testing.assert_array_equal(
            atmosphere_h2,
            np.array(
                [
                    100.0,
                    101.0,
                    19899.0,
                    19900.0,
                    20000.0,
                    np.nextafter(20000.0, np.inf),
                ]
            ),
        )

    def test_source_manifest_pins_all_molecular_implementation_files(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "src/PAYNE_ZERO_SOURCE_MANIFEST.json").read_text()
        )
        self.assertEqual(manifest["payne_zero_commit"], PINNED_COMMIT)
        entries = {entry["local_path"]: entry for entry in manifest["entries"]}
        expected = {
            "src/payne_zero_atmosphere/molecular_data.py": (
                "705c3072d79c8019c948ce0fa2c82052f232816d453e10a7c8e5fc5a8f5ce249"
            ),
            "src/payne_zero_atmosphere/molecular_equilibrium.py": (
                "4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86"
            ),
            "src/payne_zero_synthesis/molecular_equilibrium.py": (
                "df01757c160b2bff4390cc2148cff9d1ba6e5a2bc7cab4515b46f38e868d2714"
            ),
        }
        for local_path, source_hash in expected.items():
            with self.subTest(local_path=local_path):
                entry = entries[local_path]
                self.assertEqual(entry["copy_mode"], "byte-identical file")
                self.assertEqual(entry["source_file_sha256"], source_hash)
                self.assertEqual(
                    sha256(REPOSITORY_ROOT / local_path),
                    source_hash,
                )


if __name__ == "__main__":
    unittest.main()
