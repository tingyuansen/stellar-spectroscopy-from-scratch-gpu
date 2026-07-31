#!/usr/bin/env python3
"""Recompute Chapter 14 outputs before opening their pinned comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from book.chapter14_runtime import (  # noqa: E402
    ARTIFACT_MANIFEST,
    FIXTURE_PATH,
    GOLDEN_PATH,
    asset_identity_checkpoint,
    closure_seam_checkpoint,
    computed_artifact_arrays,
    configure_chapter14_runtime,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    configure_chapter14_runtime(require_built_artifacts=True)
    manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    identity = asset_identity_checkpoint()
    if identity.training_corpora_packaged:
        raise RuntimeError("Chapter 14 runtime unexpectedly contains training corpora")

    with np.load(FIXTURE_PATH, allow_pickle=False) as fixture:
        if fixture["five_label_values"].shape != (5,):
            raise RuntimeError("five-label fixture changed shape")
        if fixture["cno8_label_values"].shape != (8,):
            raise RuntimeError("CNO fixture changed shape")
        if fixture["direct_public_atomic_numbers"].shape != (81,):
            raise RuntimeError("direct public layout changed shape")

    computed = computed_artifact_arrays()
    with np.load(GOLDEN_PATH, allow_pickle=False) as archive:
        if set(computed) != set(archive.files):
            raise RuntimeError("Chapter 14 comparison inventory changed")
        for name, values in computed.items():
            np.testing.assert_array_equal(values, archive[name], err_msg=name)

    for relative, record in manifest["artifacts"].items():
        path = REPOSITORY_ROOT / relative
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"Chapter 14 artifact hash changed: {relative}")
    seam = closure_seam_checkpoint()
    if seam.status != manifest["exact_restart_status"]:
        raise RuntimeError("Chapter 14 shared-runner status changed")
    print(
        "Chapter 14 artifacts: verified; "
        f"exact restart status={seam.status}"
    )


if __name__ == "__main__":
    main()
