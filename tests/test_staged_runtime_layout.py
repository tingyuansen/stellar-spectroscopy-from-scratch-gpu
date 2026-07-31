"""Self-contained staged-package data and import contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
STATIC_ROOT = REPOSITORY_ROOT / "data" / "static"
STAGED_DATA_LINK = SOURCE_ROOT / "source_data_files"
DATA_ROOT_ENVIRONMENT_VARIABLES = (
    "PAYNE_ZERO_DATA_ROOT",
    "PAYNE_ZERO_ATMOSPHERE_DATA_ROOT",
    "PAYNE_ZERO_SYNTHESIS_DATA_ROOT",
)


def _clean_staged_environment() -> dict[str, str]:
    """Return an environment that can resolve only the staged source tree."""

    environment = os.environ.copy()
    for name in DATA_ROOT_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def test_staged_default_data_root_is_the_canonical_relative_link() -> None:
    """Keep the pinned package's default data lookup self-contained."""

    assert STAGED_DATA_LINK.is_symlink()
    link_target = Path(os.readlink(STAGED_DATA_LINK))
    assert not link_target.is_absolute()
    assert link_target == Path("../data/static")
    assert STAGED_DATA_LINK.resolve(strict=True) == STATIC_ROOT.resolve(strict=True)

    required_default_assets = (
        "atmosphere_tables/molecular_equilibrium_tables.npz",
        "atmosphere_emulator/release_manifest.json",
        "synthesis_tables/partition_saha_tables.npz",
        "source_catalogs/lines/molecular_equilibrium_atmosphere.npz",
    )
    for relative in required_default_assets:
        assert (STAGED_DATA_LINK / relative).is_file(), relative


def test_package_test_command_runs_the_pytest_suite() -> None:
    """Prevent the package command from falling back to zero-test discovery."""

    package = json.loads((REPOSITORY_ROOT / "package.json").read_text())
    command = package["scripts"]["test"]
    assert "python3 -m pytest" in command
    assert "unittest discover" not in command


def test_staged_packages_import_without_data_root_overrides() -> None:
    """Import both packages through ``src`` and prove their provenance."""

    program = """
import json
from pathlib import Path

import payne_zero_atmosphere
import payne_zero_synthesis
from payne_zero_atmosphere.data_files import atmosphere_table_path, data_root

print(json.dumps({
    "atmosphere_module": str(Path(payne_zero_atmosphere.__file__).resolve()),
    "synthesis_module": str(Path(payne_zero_synthesis.__file__).resolve()),
    "data_root": str(data_root().resolve()),
    "molecular_table": str(
        atmosphere_table_path("molecular_equilibrium_tables.npz").resolve()
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        env=_clean_staged_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert Path(payload["atmosphere_module"]).is_relative_to(SOURCE_ROOT.resolve())
    assert Path(payload["synthesis_module"]).is_relative_to(SOURCE_ROOT.resolve())
    assert Path(payload["data_root"]) == STATIC_ROOT.resolve()
    assert Path(payload["molecular_table"]) == (
        STATIC_ROOT / "atmosphere_tables/molecular_equilibrium_tables.npz"
    ).resolve()


def test_staged_cli_help_works_without_data_root_overrides() -> None:
    """The default CLI must reach argument parsing without external data."""

    completed = subprocess.run(
        [sys.executable, "-m", "payne_zero_atmosphere", "--help"],
        cwd=REPOSITORY_ROOT,
        env=_clean_staged_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage: python -m payne_zero_atmosphere" in completed.stdout
    assert "--effective-temperature" in completed.stdout
    assert "--log-surface-gravity" in completed.stdout
    assert "--out OUT" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_chapter10_and_15_rebind_paths_after_contaminated_staged_imports() -> None:
    """A prior data root must not poison exact synthesis/initializer defaults."""

    program = """
import inspect
import json
import os
from pathlib import Path
import tempfile

import torch

with tempfile.TemporaryDirectory(prefix="payne-zero-poison-root-") as directory:
    poison_root = Path(directory)
    static_root = Path.cwd() / "data/static"
    for child in (
        "atmosphere_tables",
        "atmosphere_emulator",
        "synthesis_tables",
        "source_catalogs",
    ):
        (poison_root / child).symlink_to(
            (static_root / child).resolve(),
            target_is_directory=True,
        )
    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(poison_root)

    import payne_zero_atmosphere
    import payne_zero_atmosphere.warm_start as warm_start
    import payne_zero_synthesis
    import payne_zero_synthesis.equation_of_state as equation_of_state
    import payne_zero_synthesis.pipeline as pipeline

    eos_class_before = equation_of_state.EOSTables
    initialize_before = payne_zero_synthesis.initialize_atmosphere_from_labels
    poisoned_eos_default = (
        equation_of_state.EOSTables.from_npz.__func__.__defaults__[0]
    )
    poisoned_warm_start = warm_start.DEFAULT_FIVE_LABEL_WEIGHTS_PATH

    from book.chapter15_runtime import configure_chapter15_runtime
    configure_chapter15_runtime()

    rebound_eos_default = (
        equation_of_state.EOSTables.from_npz.__func__.__defaults__[0]
    )
    rebound_pipeline_default = inspect.signature(
        pipeline.SynthesisPipeline
    ).parameters["tables_path"].default
    tables = equation_of_state.EOSTables.from_npz(
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    initialized = payne_zero_synthesis.initialize_atmosphere_from_labels(
        effective_temperature=5777.0,
        log_surface_gravity=4.44,
        metallicity=0.0,
        alpha_enhancement=0.0,
        microturbulence_km_s=1.0,
        initializer_family="five_label",
        molecular_lines=True,
        device="cpu",
        dtype="float64",
    )

    print(json.dumps({
        "poisoned_eos_default": str(poisoned_eos_default),
        "poisoned_eos_exists": poisoned_eos_default.exists(),
        "poisoned_warm_start": str(poisoned_warm_start),
        "poisoned_warm_start_exists": poisoned_warm_start.exists(),
        "rebound_eos_default": str(rebound_eos_default),
        "rebound_eos_exists": rebound_eos_default.exists(),
        "rebound_pipeline_default": str(rebound_pipeline_default),
        "rebound_pipeline_exists": rebound_pipeline_default.exists(),
        "rebound_warm_start": str(warm_start.DEFAULT_FIVE_LABEL_WEIGHTS_PATH),
        "rebound_warm_start_exists": (
            warm_start.DEFAULT_FIVE_LABEL_WEIGHTS_PATH.exists()
        ),
        "eos_class_identity_preserved": (
            equation_of_state.EOSTables is eos_class_before
        ),
        "initialize_identity_preserved": (
            payne_zero_synthesis.initialize_atmosphere_from_labels
            is initialize_before
        ),
        "partition_shape": list(tables.packed_partition_table.shape),
        "initializer_family": initialized.initializer_family,
        "initializer_flags": [
            initialized.atmosphere_converged,
            initialized.atmosphere_closure_required,
        ],
    }))
"""
    environment = _clean_staged_environment()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(SOURCE_ROOT), str(REPOSITORY_ROOT))
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert not payload["poisoned_eos_exists"]
    assert payload["poisoned_warm_start_exists"]
    assert payload["rebound_eos_exists"]
    assert payload["rebound_pipeline_exists"]
    assert payload["rebound_warm_start_exists"]
    assert payload["eos_class_identity_preserved"]
    assert payload["initialize_identity_preserved"]
    assert payload["partition_shape"] == [6, 374]
    assert payload["initializer_family"] == "five_label"
    assert payload["initializer_flags"] == [False, True]
    assert payload["rebound_eos_default"].endswith(
        "/data/synthesis_tables/partition_saha_inputs.npz"
    )
    assert payload["rebound_pipeline_default"].endswith(
        "/data/synthesis_tables/line_profile_tables.npz"
    )
    assert payload["rebound_warm_start"].endswith(
        "/data/atmosphere_emulator/five_label/checkpoint.pt"
    )
    assert payload["poisoned_warm_start"] != payload["rebound_warm_start"]
    assert not (
        STATIC_ROOT / "synthesis_tables/partition_saha_inputs.npz"
    ).exists()
