"""Small Chapter 7 kernels used to teach Numba selection semantics."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


def python_keep_mask(population_ratio, center_ratio, boltzmann, minimum_ratio):
    """Evaluate the three exact keep clauses with an interpreted Python loop."""

    keep = np.zeros(population_ratio.size, dtype=np.bool_)
    for line_index in range(population_ratio.size):
        keep[line_index] = (
            population_ratio[line_index] > minimum_ratio
            and center_ratio[line_index] >= 1.0
            and center_ratio[line_index] * boltzmann[line_index] >= 1.0
        )
    return keep


@njit(cache=True, nogil=True)
def njit_keep_mask(population_ratio, center_ratio, boltzmann, minimum_ratio):
    """Compile the same ordered clauses into one serial machine-code loop."""

    keep = np.zeros(population_ratio.size, dtype=np.bool_)
    for line_index in range(population_ratio.size):
        keep[line_index] = (
            population_ratio[line_index] > minimum_ratio
            and center_ratio[line_index] >= 1.0
            and center_ratio[line_index] * boltzmann[line_index] >= 1.0
        )
    return keep


@njit(cache=True, nogil=True, parallel=True)
def prange_keep_mask(population_ratio, center_ratio, boltzmann, minimum_ratio):
    """Parallelize independent line decisions without changing their order."""

    keep = np.zeros(population_ratio.size, dtype=np.bool_)
    for line_index in prange(population_ratio.size):
        keep[line_index] = (
            population_ratio[line_index] > minimum_ratio
            and center_ratio[line_index] >= 1.0
            and center_ratio[line_index] * boltzmann[line_index] >= 1.0
        )
    return keep
