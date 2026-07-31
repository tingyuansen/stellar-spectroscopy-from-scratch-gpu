#!/usr/bin/env python3
"""Verify Chapter 13 artifact identity and recompute its exact correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.build_chapter13_artifacts import (  # noqa: E402
    FIXTURE_PATH,
    GOLDEN_PATH,
    MANIFEST_PATH,
    PINNED_SOURCE_SHA256,
    evaluate,
)


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Fail closed on identity, schema, or exact-array disagreement."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported Chapter 13 artifact manifest")
    for relative, expected in PINNED_SOURCE_SHA256.items():
        if sha256(REPOSITORY_ROOT / relative) != expected:
            raise RuntimeError(f"Chapter 13 source identity changed: {relative}")
    for path in (FIXTURE_PATH, GOLDEN_PATH):
        record = manifest["artifacts"][str(path.relative_to(REPOSITORY_ROOT))]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"Chapter 13 artifact identity changed: {path}")

    with np.load(FIXTURE_PATH, allow_pickle=False) as archive:
        inputs = {name: np.asarray(archive[name]) for name in archive.files}
    computed = evaluate(inputs)
    with np.load(GOLDEN_PATH, allow_pickle=False) as archive:
        if set(computed) != set(archive.files):
            raise RuntimeError("Chapter 13 comparison field set changed")
        for name, values in computed.items():
            np.testing.assert_array_equal(values, archive[name])
    print("Chapter 13 artifacts: verified")


if __name__ == "__main__":
    main()
