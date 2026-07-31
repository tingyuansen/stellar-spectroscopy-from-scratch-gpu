"""Focused local-first parity checks for the Chapter 3 atomic EOS harness."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = REPOSITORY_ROOT / "data" / "golden" / "payne_zero"
PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"


def sha256(path: Path) -> str:
    """Return one file's SHA-256 hexadecimal digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load comparison arrays only after local computation has finished."""

    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


class Chapter03ParityTests(unittest.TestCase):
    """Keep local exact computations independent from comparison-only goldens."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._cache = tempfile.TemporaryDirectory(prefix="chapter03-test-numba-")
        cls._previous_numba_cache = os.environ.get("NUMBA_CACHE_DIR")
        os.environ["NUMBA_CACHE_DIR"] = cls._cache.name
        os.environ["PAYNE_ZERO_DATA_ROOT"] = str(
            REPOSITORY_ROOT / "data" / "static"
        )
        os.environ["PAYNE_ZERO_SYNTHESIS_ATOMIC_MASS_TABLE"] = str(
            REPOSITORY_ROOT
            / "data"
            / "static"
            / "synthesis_tables"
            / "atomic_masses.npz"
        )
        try:
            from numba import config as numba_config

            cls._previous_numba_config_cache = numba_config.CACHE_DIR
            numba_config.CACHE_DIR = cls._cache.name
        except ImportError:
            cls._previous_numba_config_cache = None

        from book import chapter03_runtime as runtime

        cls.runtime = runtime
        cls.inputs = runtime.load_atom_only_fixture()

        # Local calculations are complete before any golden archive is opened.
        cls.local_saha = runtime.compute_atmosphere_saha_modes(cls.inputs)
        cls.local_atmosphere = runtime.compute_atmosphere_atomic_state(cls.inputs)
        cls.local_energy_breakdown = (
            runtime.compute_atmosphere_atomic_energy_breakdown(cls.inputs)
        )
        cls.local_scalar_atmosphere = (
            runtime.compute_atmosphere_atomic_state_by_depth(cls.inputs)
        )
        cls.local_bridge = runtime.compute_atmosphere_fixed_handoff_state(
            cls.inputs,
            cls.local_atmosphere["electron_density"],
        )
        packed = np.arange(1, 1007, dtype=np.float64)[None, :]
        cls.local_sentinel_bridge = runtime.compute_packed_bridge(
            {
                "electron_density": np.array([7.0e12]),
                "fractional_doppler_widths": packed + 20000.0,
                "ion_stage_populations_by_packed_slot": packed,
                "partition_normalized_populations_by_packed_slot": (
                    packed + 10000.0
                ),
            }
        )
        cls.local_synthesis = runtime.compute_synthesis_atomic_states(cls.inputs)

        cls.saha_golden = load_npz(
            GOLDEN_ROOT / "chapter03_atmosphere_saha_outputs.npz"
        )
        cls.atmosphere_golden = load_npz(
            GOLDEN_ROOT / "chapter03_atmosphere_atomic_state.npz"
        )
        cls.bridge_golden = load_npz(
            GOLDEN_ROOT / "chapter03_packed_bridge_outputs.npz"
        )
        cls.synthesis_golden = load_npz(
            GOLDEN_ROOT
            / "chapter03_synthesis_atomic_state_cpu_float64.npz"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous_numba_cache is None:
            os.environ.pop("NUMBA_CACHE_DIR", None)
        else:
            os.environ["NUMBA_CACHE_DIR"] = cls._previous_numba_cache
        try:
            from numba import config as numba_config

            numba_config.CACHE_DIR = cls._previous_numba_config_cache
        except ImportError:
            pass
        cls._cache.cleanup()

    def test_every_golden_binds_to_the_frozen_fixture_and_commit(self) -> None:
        fixture_path = (
            REPOSITORY_ROOT
            / "data"
            / "fixtures"
            / "chapter03_atom_only_inputs.npz"
        )
        fixture_hash = sha256(fixture_path)
        for golden in (
            self.saha_golden,
            self.atmosphere_golden,
            self.bridge_golden,
            self.synthesis_golden,
        ):
            self.assertEqual(str(golden["fixture_sha256"]), fixture_hash)
            self.assertEqual(str(golden["payne_zero_commit"]), PAYNE_ZERO_COMMIT)

    def test_atmosphere_saha_scalar_batch_and_golden_are_exact(self) -> None:
        for name, local in self.local_saha.items():
            with self.subTest(array=name):
                np.testing.assert_array_equal(local, self.saha_golden[name])
        for name, batch in self.local_saha.items():
            if not name.endswith("_batch"):
                continue
            scalar_name = name.removesuffix("_batch") + "_scalar"
            with self.subTest(pair=name):
                np.testing.assert_array_equal(batch, self.local_saha[scalar_name])

    def test_atmosphere_full_state_is_refreshed_and_matches_golden(self) -> None:
        for name, local in self.local_atmosphere.items():
            with self.subTest(array=name):
                np.testing.assert_array_equal(local, self.atmosphere_golden[name])
        for name, local in self.local_scalar_atmosphere.items():
            with self.subTest(scalar_depth_array=name):
                np.testing.assert_array_equal(
                    local,
                    self.atmosphere_golden[f"scalar_depth_{name}"],
                )
                np.testing.assert_array_equal(
                    local,
                    self.local_atmosphere[name],
                )

        from payne_zero_atmosphere.equation_of_state import (
            saha_partition_depth_batch,
        )

        atmosphere, exact_seed_state = self.runtime.build_atmosphere_state(
            self.inputs
        )
        final = self.local_atmosphere
        fractions = saha_partition_depth_batch(
            atmosphere.temperature,
            final["electron_density"],
            1,
            2,
            12,
            final["charge_square_density"],
        )
        expected_hydrogen = (
            fractions
            * (
                final["total_nuclei_number_density"]
                * exact_seed_state.elemental_abundances_by_layer[:, 0]
            )[:, None]
        )
        np.testing.assert_array_equal(
            final["ion_stage_populations_by_packed_slot"][:, :2],
            expected_hydrogen,
        )

        particle_density = (
            np.asarray(self.inputs["gas_pressure"])
            / (1.38054e-16 * np.asarray(self.inputs["temperature"]))
        )
        np.testing.assert_allclose(
            final["electron_density"]
            + final["total_nuclei_number_density"],
            particle_density,
            rtol=2.0e-16,
            atol=0.0,
        )
        packed_population_scale = final[
            "ion_stage_populations_by_packed_slot"
        ].sum(axis=1) / final["total_nuclei_number_density"]
        self.assertLessEqual(
            float(
                np.max(
                    np.abs(
                        packed_population_scale
                        - np.sum(self.inputs["elemental_abundances"])
                    )
                )
            ),
            4.0e-16,
        )
        self.assertTrue(np.all(final["atomic_specific_internal_energy"] > 0.0))

    def test_packed_bridge_routes_sentinels_and_physical_values(self) -> None:
        for name, local in self.local_bridge.items():
            with self.subTest(array=name):
                np.testing.assert_array_equal(local, self.bridge_golden[name])
        for name, local in self.local_sentinel_bridge.items():
            with self.subTest(sentinel_array=name):
                np.testing.assert_array_equal(
                    local,
                    self.bridge_golden[f"sentinel_{name}"],
                )

        self.assertEqual(
            self.local_bridge["ion_stage_populations"].shape,
            (6, 6, 139),
        )
        self.assertTrue(
            np.all(
                self.local_bridge["ion_stage_populations"][:, :, 99:] == 0.0
            )
        )
        np.testing.assert_array_equal(
            self.local_bridge["electron_density"],
            self.local_bridge["fixed_input_electron_density"],
        )
        support = self.local_bridge[
            "partition_normalized_population_over_mass_density_"
            "and_fractional_doppler_width"
        ]
        self.assertEqual(support.shape, (6, 1006))
        self.assertTrue(np.all(np.isfinite(support[:, :356])))

    def test_atomic_energy_is_rebuilt_from_three_physical_contributions(self) -> None:
        breakdown = self.local_energy_breakdown
        for name, values in breakdown.items():
            with self.subTest(field=name):
                self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(breakdown["translation_specific_energy"] > 0.0))
        self.assertTrue(
            np.all(breakdown["cumulative_ionization_specific_energy"] >= 0.0)
        )
        np.testing.assert_allclose(
            breakdown["reconstructed_specific_energy"],
            breakdown["exact_specific_energy"],
            # The pedagogical sum groups the three physical contributions,
            # whereas the pinned kernel accumulates packed slots in source
            # order.  Their largest elementwise difference is roundoff.
            rtol=1.0e-15,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            breakdown["exact_specific_energy"],
            self.atmosphere_golden["atomic_specific_internal_energy"],
        )

    def test_synthesis_cpu_float64_full_and_fixed_states_match_golden(self) -> None:
        for name, local in self.local_synthesis.items():
            with self.subTest(array=name):
                np.testing.assert_array_equal(local, self.synthesis_golden[name])

        fixed_input = self.local_synthesis["fixed_input_electron_density"]
        np.testing.assert_array_equal(
            self.local_synthesis["fixed_electron_density"],
            fixed_input,
        )
        np.testing.assert_array_equal(
            self.local_synthesis["full_electron_density"],
            fixed_input,
        )
        self.assertEqual(
            self.local_synthesis["full_ion_stage_populations"].shape,
            (6, 6, 139),
        )
        self.assertEqual(
            self.local_synthesis[
                "full_eos_ion_stage_fractions_over_partition"
            ].shape,
            (6, 99, 6),
        )

        actual = self.local_synthesis["full_ion_stage_populations"][:, :, :99]
        ion_charge = np.arange(6, dtype=np.float64)[None, :, None]
        implied_electron_density = np.sum(actual * ion_charge, axis=(1, 2))
        relative_residual = np.abs(
            implied_electron_density
            - self.local_synthesis["full_electron_density"]
        ) / self.local_synthesis["full_electron_density"]
        self.assertLess(float(np.max(relative_residual)), 2.0e-4)


if __name__ == "__main__":
    unittest.main()
