#!/usr/bin/env python3
"""Measure first and warm calls for Chapter 2 in one fresh Python process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from payne_zero_atmosphere.transfer_kernels import (  # noqa: E402
    _integrate_on_depth_grid_compiled,
)


def parse_args() -> argparse.Namespace:
    """Parse the fixed depth count."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-count", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    """Print JSON timings without including input construction."""

    arguments = parse_args()
    grid = np.geomspace(1.0e-8, 1.0, arguments.depth_count)
    values = 0.2 + np.sqrt(grid) + 0.1 * grid**2

    start = time.perf_counter()
    first = _integrate_on_depth_grid_compiled(grid, values, 1.0e-10)
    first_seconds = time.perf_counter() - start

    start = time.perf_counter()
    warm = _integrate_on_depth_grid_compiled(grid, values, 1.0e-10)
    warm_seconds = time.perf_counter() - start
    if not np.array_equal(first, warm):
        raise RuntimeError("first and warm compiled results differ")

    print(
        json.dumps(
            {
                "depth_count": arguments.depth_count,
                "first_seconds": first_seconds,
                "warm_seconds": warm_seconds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
