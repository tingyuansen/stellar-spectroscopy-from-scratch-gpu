"""Small transparent calculations used by the Chapter 12 narrative.

The exact radiation, EOS, and convection algorithms live in the staged
``payne_zero_atmosphere`` modules.  These helpers expose only bookkeeping and
central-difference algebra that is clearer when written beside the chapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChunkLedger:
    """Deterministic contiguous frequency bounds and private storage."""

    bounds: np.ndarray
    private_depth_array_count: int
    private_surface_array_count: int
    private_bytes: int


def chunk_ledger(
    *,
    start: int,
    stop: int,
    chunk_count: int,
    layer_count: int,
) -> ChunkLedger:
    """Return the exact transfer partition and its private-buffer byte count."""

    start = int(start)
    stop = int(stop)
    chunk_count = int(chunk_count)
    layer_count = int(layer_count)
    if start < 0 or stop < start:
        raise ValueError("frequency interval must satisfy 0 <= start <= stop")
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    span = stop - start
    bounds = start + (span * np.arange(chunk_count + 1, dtype=np.int64)) // chunk_count
    private_bytes = 8 * chunk_count * (8 * layer_count + 1)
    return ChunkLedger(
        bounds=bounds,
        private_depth_array_count=8,
        private_surface_array_count=1,
        private_bytes=private_bytes,
    )


def central_logical_derivative(
    plus: np.ndarray,
    minus: np.ndarray,
    central_coordinate: np.ndarray,
    *,
    relative_step: float = 0.001,
) -> np.ndarray:
    """Return ``(f+ - f-) / (2 * relative_step * coordinate)``."""

    plus = np.asarray(plus, dtype=np.float64)
    minus = np.asarray(minus, dtype=np.float64)
    coordinate = np.asarray(central_coordinate, dtype=np.float64)
    if plus.shape != minus.shape or plus.shape != coordinate.shape:
        raise ValueError("plus, minus, and central_coordinate must match")
    step = float(relative_step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("relative_step must be positive and finite")
    return (plus - minus) / np.maximum(
        2.0 * step * coordinate,
        1.0e-300,
    )


def physical_flux_from_eddington_flux(
    integrated_eddington_flux: np.ndarray,
) -> np.ndarray:
    """Convert the stored integrated-H scale to physical flux."""

    return 4.0 * np.pi * np.asarray(integrated_eddington_flux, dtype=np.float64)


def require_all_or_none(*arrays: np.ndarray | None) -> bool:
    """Reject a partial finite-difference sample set and return its mode."""

    present = tuple(array is not None for array in arrays)
    if any(present) and not all(present):
        raise ValueError(
            "finite-difference convection requires all eight sample arrays"
        )
    return all(present)
