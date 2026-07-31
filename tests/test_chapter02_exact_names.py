"""Exact numerical and interface checks introduced in Chapter 2."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import unittest

import numpy as np
import torch

from payne_zero_atmosphere.radiative_transfer import integrate_on_depth_grid
from payne_zero_atmosphere.transfer_kernels import (
    _integrate_on_depth_grid_compiled,
)
from payne_zero_synthesis.device import resolve_runtime
from payne_zero_synthesis.radiative_transfer import integrate_optical_depth

from book.chapter02_support import (
    TRANSFER_FIXTURE_PATH,
    TRANSFER_GOLDEN_PATH,
    TRANSFER_OUTPUT_NAMES,
    TRANSFER_TABLE_PATH,
    run_transfer_fixture,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class Chapter02ExactNameTests(unittest.TestCase):
    """Keep the Chapter 2 numerical and provenance contracts executable."""

    def test_numpy_depth_integral_matches_linear_analytic_case(self) -> None:
        grid = np.array([0.0, 0.25, 0.75, 1.5], dtype=np.float64)
        values = 2.0 * grid + 1.0
        result = integrate_on_depth_grid(grid, values, surface_value=0.3)
        expected = 0.3 + grid**2 + grid
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=2.0e-15)

    def test_compiled_depth_integral_matches_exact_numpy_order(self) -> None:
        grid = np.geomspace(1.0e-6, 1.0, 80).astype(np.float64)
        values = (0.2 + np.sqrt(grid) + 0.1 * grid**2).astype(np.float64)
        expected = integrate_on_depth_grid(grid, values, surface_value=1.0e-8)
        actual = _integrate_on_depth_grid_compiled(grid, values, 1.0e-8)
        np.testing.assert_array_equal(actual, expected)

    def test_torch_optical_depth_preserves_wavelength_depth_layout(self) -> None:
        column_mass = torch.tensor(
            [0.0, 0.25, 0.75, 1.5], dtype=torch.float64
        )
        extinction = torch.stack(
            (2.0 * column_mass + 1.0, 0.5 * column_mass + 2.0)
        )
        surface_tau = torch.tensor([0.3, 0.1], dtype=torch.float64)
        optical_depth = integrate_optical_depth(
            column_mass, extinction, surface_tau
        )
        self.assertEqual(tuple(optical_depth.shape), (2, 4))
        for row in range(2):
            expected = integrate_on_depth_grid(
                column_mass.numpy(),
                extinction[row].numpy(),
                surface_value=float(surface_tau[row]),
            )
            np.testing.assert_allclose(
                optical_depth[row].numpy(), expected, rtol=0.0, atol=2.0e-15
            )

    def test_explicit_cpu_runtime_uses_reference_precision(self) -> None:
        runtime_device, runtime_dtype = resolve_runtime("cpu", None)
        self.assertEqual(runtime_device.type, "cpu")
        self.assertEqual(runtime_dtype, torch.float64)

    def test_transfer_inputs_match_manifest(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "data" / "MANIFEST.json").read_text()
        )
        entries = {
            Path(item["path"]).name: item
            for item in manifest["entries"]
        }
        for path in (TRANSFER_FIXTURE_PATH, TRANSFER_TABLE_PATH):
            with self.subTest(path=path.name):
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(actual, entries[path.name]["sha256"])

    def test_exact_transfer_fixture_matches_pinned_serial_and_parallel_goldens(
        self,
    ) -> None:
        serial, parallel = run_transfer_fixture(chunk_count=2)
        with np.load(TRANSFER_GOLDEN_PATH, allow_pickle=False) as golden:
            for name in TRANSFER_OUTPUT_NAMES:
                np.testing.assert_array_equal(serial[name], golden[f"serial_{name}"])
                np.testing.assert_array_equal(
                    parallel[name], golden[f"parallel_chunk2_{name}"]
                )


if __name__ == "__main__":
    unittest.main()
