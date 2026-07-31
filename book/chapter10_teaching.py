"""Small transparent calculations used by the Chapter 10 narrative.

This module contains no synthesis stages. It only turns declared shapes and
dtypes into the allocation ledger shown beside the exact pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AllocationLedger:
    """Byte counts for bounded slabs and the forbidden dense alternative."""

    float32_line_slab_bytes: int
    four_float64_depth_wavelength_slabs_bytes: int
    hypothetical_dense_depth_line_wavelength_bytes: int


def allocation_ledger(
    *,
    depth_count: int,
    wavelength_count: int,
    line_count: int,
) -> AllocationLedger:
    """Return shape-derived byte counts without inspecting an allocator."""

    dimensions = {
        "depth_count": int(depth_count),
        "wavelength_count": int(wavelength_count),
        "line_count": int(line_count),
    }
    invalid = [name for name, value in dimensions.items() if value <= 0]
    if invalid:
        raise ValueError(
            "allocation dimensions must be positive: " + ", ".join(invalid)
        )
    depth_count = dimensions["depth_count"]
    wavelength_count = dimensions["wavelength_count"]
    line_count = dimensions["line_count"]
    return AllocationLedger(
        float32_line_slab_bytes=depth_count * wavelength_count * 4,
        four_float64_depth_wavelength_slabs_bytes=(
            4 * depth_count * wavelength_count * 8
        ),
        hypothetical_dense_depth_line_wavelength_bytes=(
            depth_count * line_count * wavelength_count * 4
        ),
    )
