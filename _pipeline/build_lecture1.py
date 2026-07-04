#!/usr/bin/env python
"""Compatibility wrapper for the GPU-native Lecture 1 assembler."""
import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("build_lecture1_gpu.py")), run_name="__main__")
