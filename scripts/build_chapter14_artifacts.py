#!/usr/bin/env python3
"""Build compact Chapter 14 inputs and comparison-only decoder outputs."""

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
    CNO8_EXAMPLE,
    FIVE_LABEL_EXAMPLE,
    FIXTURE_PATH,
    GOLDEN_PATH,
    PINNED_ASSET_SHA256,
    PINNED_PAYNE_ZERO_COMMIT,
    PINNED_SOURCE_SHA256,
    _direct_public_input,
    closure_seam_checkpoint,
    computed_artifact_arrays,
    configure_chapter14_runtime,
)
from scripts.deterministic_npz import write_npz  # noqa: E402


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixture_arrays() -> dict[str, np.ndarray]:
    """Return small requested-label inputs with no decoded answer inside."""

    public_xh, _mixture = _direct_public_input()
    return {
        "five_label_values": np.asarray(
            tuple(FIVE_LABEL_EXAMPLE.values()), dtype=np.float64
        ),
        "cno8_label_values": np.asarray(
            tuple(CNO8_EXAMPLE.values()), dtype=np.float64
        ),
        "direct_public_atomic_numbers": np.asarray(
            tuple(public_xh), dtype=np.int16
        ),
        "direct_public_xh": np.asarray(tuple(public_xh.values()), dtype=np.float64),
        "pca_sentinel_indices": np.asarray([7, 23, 4], dtype=np.int16),
        "centidex_half_step_probe": np.asarray(
            [-0.025, -0.015, -0.005, 0.005, 0.015, 0.025],
            dtype=np.float64,
        ),
    }


def array_record(values: np.ndarray, unit: str) -> dict[str, object]:
    """Describe one stored deterministic array."""

    contiguous = np.ascontiguousarray(values)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "unit": unit,
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def main() -> None:
    configure_chapter14_runtime()
    fixture = fixture_arrays()
    outputs = computed_artifact_arrays()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_npz(FIXTURE_PATH, fixture)
    write_npz(GOLDEN_PATH, outputs)

    fixture_units = {
        "five_label_values": "ordered public physical labels",
        "cno8_label_values": "ordered public physical labels",
        "direct_public_atomic_numbers": "atomic number",
        "direct_public_xh": "dex [X/H]",
        "pca_sentinel_indices": "component, layer, field index",
        "centidex_half_step_probe": "dex",
    }
    output_units = {
        name: (
            "standardized PCA coefficient"
            if name.endswith("standardized_coefficients")
            else "dex or dimensionless network feature"
            if name == "direct_feature_vector"
            else "dex [X/H]"
            if name == "direct_exact_mixture"
            else "profile field in source-declared cgs units"
        )
        for name in outputs
    }
    artifacts = {
        str(FIXTURE_PATH.relative_to(REPOSITORY_ROOT)): {
            "role": "fixture",
            "sha256": sha256(FIXTURE_PATH),
            "bytes": FIXTURE_PATH.stat().st_size,
            "format": "npz",
            "builder": "scripts/build_chapter14_artifacts.py",
            "arrays": {
                name: array_record(values, fixture_units[name])
                for name, values in fixture.items()
            },
        },
        str(GOLDEN_PATH.relative_to(REPOSITORY_ROOT)): {
            "role": "golden",
            "sha256": sha256(GOLDEN_PATH),
            "bytes": GOLDEN_PATH.stat().st_size,
            "format": "npz",
            "builder": "scripts/build_chapter14_artifacts.py",
            "fixture_sha256": sha256(FIXTURE_PATH),
            "arrays": {
                name: array_record(values, output_units[name])
                for name, values in outputs.items()
            },
        },
    }
    manifest = {
        "schema_version": 1,
        "payne_zero_commit": PINNED_PAYNE_ZERO_COMMIT,
        "scope": (
            "Chapter 14 inference, decoder, quantized-seed, and direct-layout "
            "parity; training corpora excluded"
        ),
        "source_sha256": PINNED_SOURCE_SHA256,
        "static_asset_sha256": PINNED_ASSET_SHA256,
        "artifacts": artifacts,
        "exact_restart_status": closure_seam_checkpoint().status,
    }
    ARTIFACT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Chapter 14 artifacts: built")


if __name__ == "__main__":
    main()
