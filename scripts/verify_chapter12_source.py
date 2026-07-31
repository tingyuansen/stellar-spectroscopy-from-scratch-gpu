#!/usr/bin/env python3
"""Verify the pinned Chapter 12 source and extracted runner blocks."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WHOLE_FILE_SHA256 = {
    "src/payne_zero_atmosphere/rosseland_mean.py": (
        "91071248fd903e05322b7163d37566e9f894daefc1d7ba018d4850d362f1fc86"
    ),
    "src/payne_zero_atmosphere/radiative_pressure.py": (
        "c61a256892282d9a0d6cb19714ea5ce6135f1b6f7f573e5761d216d552f321ec"
    ),
    "src/payne_zero_atmosphere/temperature_correction.py": (
        "67728389ba857511979d0f82ea59f0bf41ee635b8151ae26673dace02b195d21"
    ),
    "src/payne_zero_atmosphere/convection.py": (
        "9099af3ce97123a88cfee554cefb55b2b47a52085e3cb6cda19e6869e0fef9fd"
    ),
}
EXTRACTED_SYMBOL_SHA256 = {
    "src/payne_zero_atmosphere/runner.py": {
        "compute_convection_finite_difference_samples": (
            "29112ad5aa101fdb5cfde53b23ca70282ebcb2e85243ec2d5d9dedaeaf571731"
        ),
        "TransferAccumulation": (
            "985f919db915b2eade0a898b032961c40cdc57607213dc214a123726c5b272c8"
        ),
        "IterationFinalization": (
            "cce612ee9925ca3d50a73bdc74bd0509bc2483bab2eea47a9a46963a9ee7f1d7"
        ),
        "_planck_source_and_stimulated_emission": (
            "8ff800c1f43567ae4b48ab52dc26405b6d62a8fa232037934d11d90a63d5ba8c"
        ),
        "accumulate_transfer_state": (
            "8b0a9a9652717739cbed1671a65afd4c871b728e0e2a999755bb4e78ef68cb66"
        ),
        "finalize_transfer_state": (
            "b7413ed8afe9d63c090749f853a81bdaa4880ef0a738aca0e65a9a7406d91429"
        ),
    },
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _symbol_digest(node: ast.AST) -> str:
    canonical = ast.dump(node, annotate_fields=True, include_attributes=False)
    return _sha256(canonical.encode("utf-8"))


def main() -> None:
    for relative, expected in WHOLE_FILE_SHA256.items():
        actual = _sha256((REPOSITORY_ROOT / relative).read_bytes())
        if actual != expected:
            raise SystemExit(
                f"{relative}: expected SHA-256 {expected}, observed {actual}"
            )

    for relative, expected_symbols in EXTRACTED_SYMBOL_SHA256.items():
        path = REPOSITORY_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        symbols = {
            node.name: node
            for node in tree.body
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            )
        }
        for name, expected in expected_symbols.items():
            if name not in symbols:
                raise SystemExit(f"{relative}: missing extracted symbol {name}")
            actual = _symbol_digest(symbols[name])
            if actual != expected:
                raise SystemExit(
                    f"{relative}:{name}: expected AST SHA-256 {expected}, "
                    f"observed {actual}"
                )

    print(
        "Chapter 12 source: verified "
        f"({len(WHOLE_FILE_SHA256)} exact files, "
        f"{sum(map(len, EXTRACTED_SYMBOL_SHA256.values()))} extracted symbols)"
    )


if __name__ == "__main__":
    main()
