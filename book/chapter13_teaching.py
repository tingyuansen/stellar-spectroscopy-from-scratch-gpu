"""Small transparent control calculations for Chapter 13.

These helpers do not replace any Payne Zero physics routine.  They expose the
two pieces of orchestration that are easiest to verify by hand: fixed
frequency-chunk boundaries and the minimum/consecutive structural stopping
counter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StructuralConvergenceTrace:
    """Exact per-pass structural norms and consecutive-counter history."""

    deep_layer_change: np.ndarray
    all_layer_change: np.ndarray
    within_limits: np.ndarray
    eligible: np.ndarray
    consecutive_count: np.ndarray
    stopping_pass: int | None
    converged: bool


def fixed_chunk_bounds(
    start: int,
    stop: int,
    chunk_count: int,
) -> np.ndarray:
    """Return the exact contiguous bounds used by transfer chunking."""

    start = int(start)
    stop = int(stop)
    chunk_count = int(chunk_count)
    if stop < start:
        raise ValueError("stop must not precede start")
    if chunk_count < 1:
        raise ValueError("chunk_count must be positive")
    span = stop - start
    return np.asarray(
        [start + (span * chunk) // chunk_count for chunk in range(chunk_count + 1)],
        dtype=np.int64,
    )


def trace_structural_convergence(
    before_by_pass: np.ndarray,
    after_by_pass: np.ndarray,
    *,
    enable_convergence_stop: bool,
    minimum_iterations_before_convergence: int,
    required_consecutive_converged_iterations: int,
    maximum_deep_layer_relative_temperature_change: float,
    maximum_all_layer_relative_temperature_change: float | None,
) -> StructuralConvergenceTrace:
    """Execute the exact runner's structural stopping state machine."""

    from payne_zero_atmosphere.convergence import (
        deep_layer_relative_temperature_change,
        max_normalized_column_delta,
        temperature_changes_within_limits,
    )

    before = np.asarray(before_by_pass, dtype=np.float64)
    after = np.asarray(after_by_pass, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError("before_by_pass and after_by_pass must share shape (P,L)")
    pass_count = before.shape[0]
    deep = np.empty(pass_count, dtype=np.float64)
    all_layer = np.empty(pass_count, dtype=np.float64)
    within = np.zeros(pass_count, dtype=np.bool_)
    eligible = np.zeros(pass_count, dtype=np.bool_)
    counts = np.zeros(pass_count, dtype=np.int32)
    consecutive = 0
    stopping_pass: int | None = None

    for pass_offset in range(pass_count):
        pass_index = pass_offset + 1
        deep[pass_offset] = deep_layer_relative_temperature_change(
            before[pass_offset], after[pass_offset]
        )
        all_layer[pass_offset] = max_normalized_column_delta(
            before[pass_offset],
            after[pass_offset],
            floor=1.0,
            symmetric=True,
        )
        within[pass_offset] = temperature_changes_within_limits(
            deep_layer_change=float(deep[pass_offset]),
            all_layer_change=float(all_layer[pass_offset]),
            maximum_deep_layer_change=maximum_deep_layer_relative_temperature_change,
            maximum_all_layer_change=maximum_all_layer_relative_temperature_change,
        )
        eligible[pass_offset] = bool(
            enable_convergence_stop
            and pass_index >= int(minimum_iterations_before_convergence)
        )
        consecutive = (
            consecutive + 1 if eligible[pass_offset] and within[pass_offset] else 0
        )
        counts[pass_offset] = consecutive
        if (
            enable_convergence_stop
            and consecutive >= int(required_consecutive_converged_iterations)
        ):
            stopping_pass = pass_index
            break

    return StructuralConvergenceTrace(
        deep_layer_change=deep,
        all_layer_change=all_layer,
        within_limits=within,
        eligible=eligible,
        consecutive_count=counts,
        stopping_pass=stopping_pass,
        converged=stopping_pass is not None,
    )
