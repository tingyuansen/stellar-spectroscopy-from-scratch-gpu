#!/usr/bin/env python3
"""Verify that displayed chapter-stage code matches pinned Payne Zero source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"

EXACT_FILES = (
    (
        Path("src/payne_zero_synthesis/__init__.py"),
        Path("payne_zero_synthesis/__init__.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/line_profile_math.py"),
        Path("payne_zero_atmosphere/line_profile_math.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/line_catalog.py"),
        Path("payne_zero_atmosphere/line_catalog.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/line_selection.py"),
        Path("payne_zero_atmosphere/line_selection.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/hydrogen_line_profile.py"),
        Path("payne_zero_atmosphere/hydrogen_line_profile.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/line_opacity.py"),
        Path("payne_zero_atmosphere/line_opacity.py"),
    ),
    (
        Path("src/payne_zero_synthesis/atomic_lines.py"),
        Path("payne_zero_synthesis/atomic_lines.py"),
    ),
    (
        Path("src/payne_zero_synthesis/line_opacity.py"),
        Path("payne_zero_synthesis/line_opacity.py"),
    ),
    (
        Path("src/payne_zero_synthesis/hydrogen_lines.py"),
        Path("payne_zero_synthesis/hydrogen_lines.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/continuum_opacity.py"),
        Path("payne_zero_atmosphere/continuum_opacity.py"),
    ),
    (
        Path("src/payne_zero_synthesis/continuum.py"),
        Path("payne_zero_synthesis/continuum.py"),
    ),
    (
        Path("src/payne_zero_synthesis/constants.py"),
        Path("payne_zero_synthesis/constants.py"),
    ),
    (
        Path("src/payne_zero_synthesis/radiative_transfer.py"),
        Path("payne_zero_synthesis/radiative_transfer.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/_numba_cache.py"),
        Path("payne_zero_atmosphere/_numba_cache.py"),
    ),
    (
        Path("src/payne_zero_synthesis/device.py"),
        Path("payne_zero_synthesis/device.py"),
    ),
    (
        Path("src/payne_zero_synthesis/atmosphere.py"),
        Path("payne_zero_synthesis/atmosphere.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/data_files.py"),
        Path("payne_zero_atmosphere/data_files.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/radiative_transfer.py"),
        Path("payne_zero_atmosphere/radiative_transfer.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/source_catalogs.py"),
        Path("payne_zero_atmosphere/source_catalogs.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/transfer_kernels.py"),
        Path("payne_zero_atmosphere/transfer_kernels.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/atmosphere_io.py"),
        Path("payne_zero_atmosphere/atmosphere_io.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/config.py"),
        Path("payne_zero_atmosphere/config.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/run_setup.py"),
        Path("payne_zero_atmosphere/run_setup.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/hydrostatic.py"),
        Path("payne_zero_atmosphere/hydrostatic.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/microturbulence.py"),
        Path("payne_zero_atmosphere/microturbulence.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/warm_start.py"),
        Path("payne_zero_atmosphere/warm_start.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/direct_abundance.py"),
        Path("payne_zero_atmosphere/direct_abundance.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/constants.py"),
        Path("payne_zero_atmosphere/constants.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/equation_of_state.py"),
        Path("payne_zero_atmosphere/equation_of_state.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/population_layout.py"),
        Path("payne_zero_atmosphere/population_layout.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/runtime_state.py"),
        Path("payne_zero_atmosphere/runtime_state.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/doppler.py"),
        Path("payne_zero_atmosphere/doppler.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/specific_internal_energy.py"),
        Path("payne_zero_atmosphere/specific_internal_energy.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/synthesis_bridge.py"),
        Path("payne_zero_atmosphere/synthesis_bridge.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/molecular_data.py"),
        Path("payne_zero_atmosphere/molecular_data.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/molecular_equilibrium.py"),
        Path("payne_zero_atmosphere/molecular_equilibrium.py"),
    ),
    (
        Path("src/payne_zero_synthesis/equation_of_state.py"),
        Path("payne_zero_synthesis/equation_of_state.py"),
    ),
    (
        Path("src/payne_zero_synthesis/molecular_equilibrium.py"),
        Path("payne_zero_synthesis/molecular_equilibrium.py"),
    ),
    (
        Path("src/payne_zero_synthesis/source_catalog_molecular_compiler.py"),
        Path("payne_zero_synthesis/source_catalog_molecular_compiler.py"),
    ),
    (
        Path("src/payne_zero_synthesis/molecular_lines.py"),
        Path("payne_zero_synthesis/molecular_lines.py"),
    ),
    (
        Path("src/payne_zero_synthesis/ground_partition_table.py"),
        Path("payne_zero_synthesis/ground_partition_table.py"),
    ),
    (
        Path("src/payne_zero_synthesis/paths.py"),
        Path("payne_zero_synthesis/paths.py"),
    ),
    (
        Path("src/payne_zero_synthesis/pipeline.py"),
        Path("payne_zero_synthesis/pipeline.py"),
    ),
    (
        Path("src/payne_zero_synthesis/prewarm.py"),
        Path("payne_zero_synthesis/prewarm.py"),
    ),
    (
        Path("src/payne_zero_synthesis/synthesis.py"),
        Path("payne_zero_synthesis/synthesis.py"),
    ),
    (
        Path("src/payne_zero_synthesis/api.py"),
        Path("payne_zero_synthesis/api.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/runner.py"),
        Path("payne_zero_atmosphere/runner.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/__init__.py"),
        Path("payne_zero_atmosphere/__init__.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/__main__.py"),
        Path("payne_zero_atmosphere/__main__.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/cli.py"),
        Path("payne_zero_atmosphere/cli.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/prewarm.py"),
        Path("payne_zero_atmosphere/prewarm.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/install_runtime_data.py"),
        Path("payne_zero_atmosphere/install_runtime_data.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/convergence.py"),
        Path("payne_zero_atmosphere/convergence.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/convection.py"),
        Path("payne_zero_atmosphere/convection.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/radiative_pressure.py"),
        Path("payne_zero_atmosphere/radiative_pressure.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/rosseland_mean.py"),
        Path("payne_zero_atmosphere/rosseland_mean.py"),
    ),
    (
        Path("src/payne_zero_atmosphere/temperature_correction.py"),
        Path("payne_zero_atmosphere/temperature_correction.py"),
    ),
)

EXACT_DEFINITIONS: tuple[
    tuple[Path, Path, tuple[str, ...]], ...
] = ()


def parse_args() -> argparse.Namespace:
    """Parse the read-only source-root override."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/Users/ysting/payne-zero"),
        help="read-only pinned Payne Zero checkout",
    )
    return parser.parse_args()


def git_commit(source_root: Path) -> str:
    """Return the checkout's exact current commit."""

    return subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def top_level_definitions(path: Path) -> dict[str, ast.AST]:
    """Return top-level functions and classes keyed by exact source name."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def main() -> None:
    """Check the pin, exact files, and exact displayed definitions."""

    arguments = parse_args()
    source_root = arguments.source_root.expanduser().resolve()
    commit = git_commit(source_root)
    if commit != EXPECTED_COMMIT:
        raise SystemExit(
            f"Payne Zero source is at {commit}; expected {EXPECTED_COMMIT}"
        )

    manifest_path = REPOSITORY_ROOT / "src" / "PAYNE_ZERO_SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["payne_zero_commit"] != EXPECTED_COMMIT:
        raise SystemExit("source manifest commit does not match the executable pin")
    manifest_entries = {
        Path(entry["local_path"]): entry for entry in manifest["entries"]
    }
    if len(manifest_entries) != len(manifest["entries"]):
        raise SystemExit("source manifest contains duplicate local paths")
    for local_relative, entry in manifest_entries.items():
        source_relative = Path(entry["source_path"])
        source_path = source_root / source_relative
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_sha256 != entry["source_file_sha256"]:
            raise SystemExit(
                f"manifest SHA-256 for {source_relative} is "
                f"{entry['source_file_sha256']}; pinned source is {source_sha256}"
            )
        if not (REPOSITORY_ROOT / local_relative).is_file():
            raise SystemExit(f"manifest local source is missing: {local_relative}")

    for local_relative, source_relative in EXACT_FILES:
        entry = manifest_entries.get(local_relative)
        if entry is None or entry["copy_mode"] != "byte-identical file":
            raise SystemExit(
                f"{local_relative} is missing its byte-identical manifest contract"
            )
        local_path = REPOSITORY_ROOT / local_relative
        source_path = source_root / source_relative
        if local_path.read_bytes() != source_path.read_bytes():
            raise SystemExit(
                f"{local_relative} is not byte-identical to pinned {source_relative}"
            )
        print(f"exact file: {local_relative}")

    for local_relative, source_relative, names in EXACT_DEFINITIONS:
        entry = manifest_entries.get(local_relative)
        if entry is None or entry["copy_mode"] != (
            "exact top-level definitions in a progressive module"
        ):
            raise SystemExit(
                f"{local_relative} is missing its progressive-definition contract"
            )
        if set(entry["exact_definitions"]) != set(names):
            raise SystemExit(
                f"{local_relative} manifest definitions differ from verifier"
            )
        local_path = REPOSITORY_ROOT / local_relative
        source_path = source_root / source_relative
        local_definitions = top_level_definitions(local_path)
        source_definitions = top_level_definitions(source_path)
        for name in names:
            if name not in local_definitions or name not in source_definitions:
                raise SystemExit(
                    f"{name} missing from {local_relative} or {source_relative}"
                )
            local_ast = ast.dump(local_definitions[name], include_attributes=False)
            source_ast = ast.dump(source_definitions[name], include_attributes=False)
            if local_ast != source_ast:
                raise SystemExit(
                    f"{local_relative}:{name} differs from pinned {source_relative}"
                )
            print(f"exact definition: {local_relative}:{name}")

    print(f"all chapter-stage fragments match Payne Zero {commit}")


if __name__ == "__main__":
    main()
