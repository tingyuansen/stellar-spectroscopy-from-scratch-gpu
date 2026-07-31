"""Pinned source, table, grid, and route gates for Chapter 5."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = REPOSITORY_ROOT / "data/static"
os.environ["PAYNE_ZERO_DATA_ROOT"] = str(STATIC_ROOT)

from payne_zero_atmosphere import continuum_opacity as atmosphere_continuum  # noqa: E402
from payne_zero_synthesis import continuum as synthesis_continuum  # noqa: E402


PINNED_HASHES = {
    "src/payne_zero_atmosphere/continuum_opacity.py": (
        "1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81"
    ),
    "src/payne_zero_synthesis/continuum.py": (
        "ab0d4eb771ee04101f6936253f633ed60d845e2816854a06b1b059e8b91dce1b"
    ),
    "data/static/atmosphere_tables/continuum_opacity_tables.npz": (
        "6fd4c556418870c28d3fcc9a050252af58ac4cc433cae979477355c8c7d593e3"
    ),
    "data/static/atmosphere_tables/karzas_latter_tables.npz": (
        "23805dc17c47af45b8ae63b2e278e1fb6c584a01c87d1eb3c31306e4555e6d15"
    ),
    "data/static/synthesis_tables/continuum_tables.npz": (
        "406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d"
    ),
}
CONTINUUM_FIXTURE = REPOSITORY_ROOT / "data/fixtures/chapter05_continuum_states.npz"
CONTINUUM_FIXTURE_SHA256 = (
    "ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae"
)


class Chapter05SourceDataTests(unittest.TestCase):
    """Freeze the four-lane continuum construction before chapter prose."""

    def test_sources_and_active_tables_match_pinned_bytes(self) -> None:
        for relative_path, expected in PINNED_HASHES.items():
            with self.subTest(path=relative_path):
                path = REPOSITORY_ROOT / relative_path
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    expected,
                )

    def test_manifest_declares_hminus_stored_unit_and_all_arrays(self) -> None:
        manifest = json.loads((REPOSITORY_ROOT / "data/MANIFEST.json").read_text())
        entries = {entry["path"]: entry for entry in manifest["entries"]}
        for relative_path in PINNED_HASHES:
            if not relative_path.startswith("data/"):
                continue
            with self.subTest(path=relative_path):
                entry = entries[relative_path]
                self.assertEqual(entry["role"], "static")
                self.assertEqual(entry["sha256"], PINNED_HASHES[relative_path])
                with np.load(REPOSITORY_ROOT / relative_path) as archive:
                    self.assertEqual(set(entry["arrays"]), set(archive.files))
        stored_unit = entries["data/static/synthesis_tables/continuum_tables.npz"][
            "arrays"
        ]["hminus_boundfree_cross_section_cm2"]["unit"]
        self.assertIn("1e-18 cm^2", stored_unit)
        self.assertIn("multiply by 1e-18", stored_unit)

    def test_exact_table_loaders_preserve_shapes_and_host_float64(self) -> None:
        atmosphere_tables = atmosphere_continuum.load_continuum_opacity_tables()
        karzas_tables = atmosphere_continuum.load_karzas_latter_tables()
        self.assertEqual(
            atmosphere_tables.hot_metal_boundfree_transition_table.shape,
            (60, 7),
        )
        self.assertEqual(
            atmosphere_tables.ch_cross_section_table.shape,
            (106, 15),
        )
        self.assertEqual(
            karzas_tables.karzas_latter_angular_log10_cross_section_cm2.shape,
            (6, 6, 29),
        )
        self.assertEqual(
            atmosphere_tables.hminus_boundfree_cross_section_cm2.dtype,
            np.float64,
        )

    def test_synthesis_loader_default_and_product_dtype_are_distinct(self) -> None:
        table_path = STATIC_ROOT / "synthesis_tables/continuum_tables.npz"
        default_tables = synthesis_continuum.ContinuumTables.from_npz(
            table_path,
            device="cpu",
        )
        product_tables = synthesis_continuum.ContinuumTables.from_npz(
            table_path,
            device="cpu",
            dtype=torch.float64,
        )
        self.assertIs(default_tables.dtype, torch.float32)
        self.assertIs(product_tables.dtype, torch.float64)
        self.assertEqual(
            product_tables.coulomb_freefree_gaunt_table_device.dtype,
            torch.float64,
        )

    def test_packaged_edge_samples_equal_exact_reconstruction(self) -> None:
        edge_path = STATIC_ROOT / "synthesis_tables/continuum_edge_grid.npz"
        with np.load(edge_path, allow_pickle=False) as edge:
            rebuilt = synthesis_continuum.build_edge_sample_frequencies(
                edge["signed_continuum_edge_frequency_hz"],
                edge["continuum_edge_wavelength_nm"],
            )
            np.testing.assert_array_equal(
                rebuilt,
                edge["continuum_edge_sample_frequency_hz"],
            )
            sign_flipped = synthesis_continuum.build_edge_sample_frequencies(
                -edge["signed_continuum_edge_frequency_hz"],
                edge["continuum_edge_wavelength_nm"],
            )
        np.testing.assert_array_equal(sign_flipped, rebuilt)

    def test_all_five_atmosphere_sampling_regimes_are_exact(self) -> None:
        cases = (
            (4499.0, 11601, 144.577263387),
            (4500.0, 9599, 91.1800865275),
            (7250.0, 7027, 50.4312810266),
            (13000.0, 3577, 22.7876741143),
            (30000.0, 1, 10.0023028502),
        )
        for temperature, start_index, first_wavelength in cases:
            with self.subTest(temperature=temperature):
                wavelength, weights = atmosphere_continuum.build_opacity_sampling_grid(
                    temperature
                )
                self.assertEqual(wavelength.shape, (30000,))
                self.assertEqual(weights.shape, (30000,))
                self.assertAlmostEqual(
                    wavelength[0],
                    first_wavelength,
                    places=9,
                )
                expected = 10.0 ** (1.0 + 1.0e-4 * start_index)
                self.assertEqual(wavelength[0], expected)

    def test_line_reference_grid_has_physical_tail_and_sentinel(self) -> None:
        wavelength, packed = (
            atmosphere_continuum.build_continuum_reference_wavelength_grid()
        )
        self.assertEqual(wavelength.shape, (344,))
        self.assertEqual(packed.shape, (344,))
        self.assertEqual(wavelength[-1], wavelength[-2])
        self.assertEqual(packed[-1], 2**30)

    def test_four_regime_fixture_is_a_closed_chapter04_to_05_boundary(
        self,
    ) -> None:
        self.assertEqual(
            hashlib.sha256(CONTINUUM_FIXTURE.read_bytes()).hexdigest(),
            CONTINUUM_FIXTURE_SHA256,
        )
        with np.load(CONTINUUM_FIXTURE, allow_pickle=False) as fixture:
            regimes = tuple(fixture["regime_names"].tolist())
            self.assertEqual(
                regimes,
                (
                    "hot_dwarf",
                    "solar_dwarf",
                    "low_gravity_giant",
                    "cool_molecule_rich",
                ),
            )
            for regime in regimes:
                with self.subTest(regime=regime):
                    atmosphere_fields = [
                        name
                        for name in fixture.files
                        if name.startswith(f"{regime}__atmosphere__")
                    ]
                    synthesis_fields = [
                        name
                        for name in fixture.files
                        if name.startswith(f"{regime}__synthesis__")
                    ]
                    self.assertEqual(len(atmosphere_fields), 18)
                    self.assertEqual(len(synthesis_fields), 27)
                    self.assertEqual(
                        fixture[f"{regime}__atmosphere__temperature"].shape,
                        (6,),
                    )
                    self.assertEqual(
                        fixture[
                            f"{regime}__atmosphere__"
                            "ion_stage_populations_by_packed_slot"
                        ].shape,
                        (6, 1006),
                    )
                    self.assertEqual(
                        fixture[f"{regime}__synthesis__ion_stage_populations"].shape,
                        (6, 6, 139),
                    )
                    self.assertTrue(
                        np.all(fixture[f"{regime}__atmosphere__mass_density"] > 0.0)
                    )


if __name__ == "__main__":
    unittest.main()
