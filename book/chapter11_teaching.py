"""Small transparent calculations for Chapter 11.

These helpers explain shape, memory, and strict temperature-branch behavior.
They do not reproduce any opacity or atmosphere-solver stage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpacityMemoryLedger:
    """Byte counts for the full first-pass depth-by-frequency state."""

    one_float64_slab_bytes: int
    one_float32_slab_bytes: int
    three_float64_continuum_slabs_bytes: int
    complete_four_slab_bytes: int


def opacity_memory_ledger(
    *, depth_count: int = 80, frequency_count: int = 30_000
) -> OpacityMemoryLedger:
    """Return shape-derived allocation sizes without inspecting an allocator."""

    depth = int(depth_count)
    frequency = int(frequency_count)
    if depth <= 0 or frequency <= 0:
        raise ValueError("depth_count and frequency_count must be positive")
    float64_slab = depth * frequency * 8
    float32_slab = depth * frequency * 4
    return OpacityMemoryLedger(
        one_float64_slab_bytes=float64_slab,
        one_float32_slab_bytes=float32_slab,
        three_float64_continuum_slabs_bytes=3 * float64_slab,
        complete_four_slab_bytes=3 * float64_slab + float32_slab,
    )


def opacity_grid_start_index(effective_temperature: float) -> int:
    """Expose the exact strict-threshold branch table in one readable function."""

    start_index = 1
    temperature = float(effective_temperature)
    if temperature < 30_000.0:
        start_index = 3_577
    if temperature < 13_000.0:
        start_index = 7_027
    if temperature < 7_250.0:
        start_index = 9_599
    if temperature < 4_500.0:
        start_index = 11_601
    return start_index
