#!/usr/bin/env python3
"""Recompute Chapter 15 source, asset, schema, and request identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "data/chapter15_artifacts.json"
EXPECTED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
EXPECTED_CASES = (
    "hot_dwarf",
    "solar_dwarf",
    "low_gravity_giant",
    "cool_molecule_rich",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema") != "chapter15_artifacts_v1":
        raise SystemExit("unsupported Chapter 15 artifact manifest schema")
    if manifest.get("pinned_payne_zero_commit") != EXPECTED_COMMIT:
        raise SystemExit("Chapter 15 pinned commit changed")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise SystemExit("Chapter 15 manifest has no artifacts")
    for relative, record in artifacts.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise SystemExit(f"missing Chapter 15 artifact: {relative}")
        actual = _sha256(path)
        expected = record.get("sha256")
        if actual != expected:
            raise SystemExit(
                f"{relative}: expected SHA-256 {expected}, observed {actual}"
            )

    request_path = REPOSITORY_ROOT / "data/fixtures/chapter15_case_requests.json"
    requests = json.loads(request_path.read_text(encoding="utf-8"))
    if requests.get("schema") != "chapter15_case_requests_v1":
        raise SystemExit("unsupported Chapter 15 request schema")
    names = tuple(case.get("name") for case in requests.get("cases", ()))
    if names != EXPECTED_CASES:
        raise SystemExit(f"Chapter 15 case order changed: {names}")
    if len(set(names)) != len(names):
        raise SystemExit("Chapter 15 case names are not unique")

    tolerance = manifest.get("tolerance_profile", {})
    if (
        tolerance.get("normalized_flux_ratio_rtol") != 3.0e-7
        or tolerance.get("normalized_flux_ratio_atol") != 0.0
    ):
        raise SystemExit("Chapter 15 tolerance profile changed")

    print(
        "Chapter 15 artifacts: verified "
        f"({len(artifacts)} identities, {len(names)} requests)"
    )


if __name__ == "__main__":
    main()
