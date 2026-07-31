#!/usr/bin/env python3
"""Measure one transparent Chapter 5 column kernel in an isolated process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def _elapsed(function, *arguments) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    result = function(*arguments)
    return result, time.perf_counter() - start


def _fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def capture(maximum_threads: int) -> dict[str, object]:
    """Return cold, warm, cached-process, and thread timing evidence."""

    cache_text = os.environ.get("NUMBA_CACHE_DIR", "")
    if not cache_text:
        raise RuntimeError("NUMBA_CACHE_DIR must name an external directory")
    cache_path = Path(cache_text)
    if not cache_path.is_dir():
        raise RuntimeError("NUMBA_CACHE_DIR must already exist")

    from numba import config, set_num_threads

    from book.chapter05_teaching import (
        njit_frequency_columns,
        prange_frequency_columns,
        python_frequency_columns,
    )

    number_density = np.geomspace(1.0e9, 1.0e18, 80)
    mass_density = np.geomspace(1.0e-13, 1.0e-4, 80)
    cross_section = np.geomspace(1.0e-23, 1.0e-17, 30_000)
    arguments = (number_density, cross_section, mass_density)

    python_result, python_seconds = _elapsed(
        python_frequency_columns,
        *arguments,
    )
    serial_first, serial_first_seconds = _elapsed(
        njit_frequency_columns,
        *arguments,
    )
    serial_warm, serial_warm_seconds = _elapsed(
        njit_frequency_columns,
        *arguments,
    )

    set_num_threads(1)
    parallel_first, parallel_first_seconds = _elapsed(
        prange_frequency_columns,
        *arguments,
    )
    parallel_one, parallel_one_seconds = _elapsed(
        prange_frequency_columns,
        *arguments,
    )
    available_threads = int(config.NUMBA_NUM_THREADS)
    used_threads = max(
        1,
        min(int(maximum_threads), int(os.cpu_count() or 1), available_threads),
    )
    set_num_threads(used_threads)
    parallel_many, parallel_many_seconds = _elapsed(
        prange_frequency_columns,
        *arguments,
    )

    return {
        "shape": list(python_result.shape),
        "dtype": str(python_result.dtype),
        "python_seconds": python_seconds,
        "serial_first_seconds": serial_first_seconds,
        "serial_warm_seconds": serial_warm_seconds,
        "parallel_first_seconds": parallel_first_seconds,
        "parallel_one_thread_seconds": parallel_one_seconds,
        "parallel_many_thread_seconds": parallel_many_seconds,
        "maximum_requested_threads": int(maximum_threads),
        "available_threads": available_threads,
        "used_threads": used_threads,
        "python_fingerprint": _fingerprint(python_result),
        "serial_fingerprint": _fingerprint(serial_warm),
        "parallel_one_fingerprint": _fingerprint(parallel_one),
        "parallel_many_fingerprint": _fingerprint(parallel_many),
        "serial_first_equal": bool(np.array_equal(serial_first, python_result)),
        "serial_warm_equal": bool(np.array_equal(serial_warm, python_result)),
        "parallel_first_equal": bool(np.array_equal(parallel_first, python_result)),
        "parallel_one_equal": bool(np.array_equal(parallel_one, python_result)),
        "parallel_many_equal": bool(np.array_equal(parallel_many, python_result)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-threads", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.maximum_threads <= 0:
        parser.error("--maximum-threads must be positive")
    print(json.dumps(capture(arguments.maximum_threads), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
