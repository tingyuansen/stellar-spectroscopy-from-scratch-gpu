"""Unit checks for the read-only Chapter 6 backend measurement."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

import scripts.measure_chapter06_synthesis_backends as measurement


class Chapter06BackendMeasurementTests(unittest.TestCase):
    def test_float32_ulp_distance_is_exact_for_adjacent_positive_values(self) -> None:
        value = np.float32(1.0)
        neighbor = np.nextafter(value, np.float32(np.inf), dtype=np.float32)
        distance = measurement._float32_ulp_distance(
            np.asarray([value, neighbor]),
            np.asarray([value, value]),
        )
        np.testing.assert_array_equal(distance, np.asarray([0, 1], dtype=np.uint64))

    def test_relative_error_uses_union_of_nonzero_cells(self) -> None:
        observed = np.asarray([0.0, 2.0, 2.0], dtype=np.float32)
        reference = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(
            measurement._relative_error(observed, reference),
            np.asarray([1.0, 1.0, 0.5]),
        )

    def test_frequency_integral_is_orientation_independent(self) -> None:
        wavelength = np.asarray([400.0, 500.0, 600.0], dtype=np.float64)
        slab = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        forward = measurement._frequency_integrals(slab, wavelength)
        reverse = measurement._frequency_integrals(slab[:, ::-1], wavelength[::-1])
        np.testing.assert_allclose(forward, reverse, rtol=0.0, atol=0.0)

    def test_command_is_read_only_and_has_two_repeats(self) -> None:
        source = inspect.getsource(measurement)
        self.assertEqual(measurement.REPEATS, 2)
        for forbidden in (
            "write_bytes(",
            "write_text(",
            "np.save(",
            "np.savez(",
            "np.savez_compressed(",
            "shutil.copy",
            "os.replace",
            "Path.unlink",
        ):
            self.assertNotIn(forbidden, source)

    def test_pinned_authority_exists_with_expected_hash(self) -> None:
        self.assertTrue(measurement.GOLDEN_PATH.is_file())
        self.assertEqual(
            measurement._sha256(measurement.GOLDEN_PATH),
            measurement.EXPECTED_GOLDEN_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
