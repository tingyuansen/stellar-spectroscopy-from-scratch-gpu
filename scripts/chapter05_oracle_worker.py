#!/usr/bin/env python3
"""Capture unpublished Chapter 5 continuum comparison oracles in memory.

The worker executes the pinned Payne Zero atmosphere and synthesis continuum
implementations on the input-only Chapter 5 fixture.  It returns deterministic
NumPy arrays and metadata; it does not read or publish a golden product.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import fields
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping


THREAD_ENVIRONMENT = {
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

ORACLE_ENVIRONMENT = {
    **THREAD_ENVIRONMENT,
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "TZ": "UTC",
}

# A directly executed worker can still establish BLAS/Numba controls before
# importing NumPy.  Hash randomization must already be fixed by the parent
# process and is checked below.
if __name__ == "__main__":
    os.environ.update(THREAD_ENVIRONMENT)

import numpy as np  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_ROOT = Path("/Users/ysting/payne-zero")
PINNED_DATA_ROOT = PINNED_ROOT / "source_data_files"
WORKER_PATH = Path(__file__).resolve()
CAPTURE_CONTRACT_PATH = (
    REPOSITORY_ROOT / "design" / "chapter05_oracle_capture_contract.md"
)
PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
FIXTURE_PATH = REPOSITORY_ROOT / "data" / "fixtures" / "chapter05_continuum_states.npz"
FIXTURE_SHA256 = "ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae"

FROZEN_CAPTURE_SCHEMA_VERSION = 2
# Accepted only after two independent empty-cache candidate captures agreed.
ACCEPTED_CAPTURE_KEY_COUNT = 1161
ACCEPTED_CAPTURE_SCHEMA_DIGEST = (
    "652c110dc79a6f6dfca6893bee35416289675b4920a5d0dcfe6b2cb262dacf3d"
)

RESIDUAL_RELATIVE_TOLERANCE = 2.0e-12
RESIDUAL_ABSOLUTE_FLOORS = {
    "opacity": 1.0e-20,
    "source": 1.0e-30,
    "interpolation": 1.0e-20,
}

REGIME_NAMES = (
    "hot_dwarf",
    "solar_dwarf",
    "low_gravity_giant",
    "cool_molecule_rich",
)

ATMOSPHERE_FIELDS = (
    "temperature",
    "mass_density",
    "electron_density",
    "gas_pressure",
    "hydrogen_partition_normalized_ion_stage_populations",
    "hydrogen_neutral_population",
    "hydrogen_ionized_population",
    "helium_neutral_population",
    "helium_singly_ionized_population",
    "helium_neutral_partition_normalized_population",
    "helium_singly_ionized_partition_normalized_population",
    "elemental_abundances_by_layer",
    "hydrogen_departure_coefficients",
    "microturbulence",
    "ion_stage_populations_by_packed_slot",
    "partition_normalized_populations_by_packed_slot",
    "ch_population",
    "oh_population",
)

SYNTHESIS_FIELDS = (
    "aluminum_neutral_partition_normalized_population",
    "atmosphere_schema_version",
    "carbon_partition_normalized_ion_stage_populations",
    "column_mass",
    "continuum_edge_interval_width_squared_over_two_nm2",
    "continuum_edge_midpoint_wavelength_nm",
    "continuum_edge_wavelength_nm",
    "electron_density",
    "elemental_abundances",
    "fractional_doppler_widths",
    "gas_pressure",
    "hc_over_kt",
    "helium_neutral_population",
    "helium_singly_ionized_population",
    "hydrogen_ionized_population",
    "hydrogen_neutral_population",
    "hydrogen_partition_normalized_ion_stage_populations",
    "ion_stage_populations",
    "iron_neutral_partition_normalized_population",
    "magnesium_neutral_partition_normalized_population",
    "mass_density",
    "microturbulence",
    "molecular_hydrogen_population",
    "partition_normalized_populations",
    "signed_continuum_edge_frequency_hz",
    "silicon_neutral_partition_normalized_population",
    "temperature",
)

PIPELINE_CONTINUUM_FIELDS = (
    "temperature",
    "mass_density",
    "electron_density",
    "hydrogen_partition_normalized_ion_stage_populations",
    "hydrogen_neutral_population",
    "helium_neutral_population",
    "helium_singly_ionized_population",
    "carbon_partition_normalized_ion_stage_populations",
    "magnesium_neutral_partition_normalized_population",
    "aluminum_neutral_partition_normalized_population",
    "silicon_neutral_partition_normalized_population",
    "iron_neutral_partition_normalized_population",
    "partition_normalized_populations",
    "ion_stage_populations",
    "signed_continuum_edge_frequency_hz",
    "continuum_edge_wavelength_nm",
    "continuum_edge_midpoint_wavelength_nm",
    "continuum_edge_interval_width_squared_over_two_nm2",
)

INPUT_FIELDS = (
    "column_mass",
    "electron_density_seed",
    "elemental_abundances",
    "gas_pressure",
    "microturbulence",
    "temperature",
)

FIXTURE_GLOBAL_FIELDS = (
    "atmosphere_grid_boundary_temperature_k",
    "atmosphere_runner_opacity_flags",
    "h2_policy_temperature_k",
    "hminus_edge_frequency_hz",
    "payne_zero_commit",
    "regime_names",
    "source_chapter04_fixture_sha256",
    "source_continuum_edge_grid_sha256",
    "synthesis_edge_probe_wavelength_nm",
)

# Frozen after importing the two continuum engines plus the atmosphere runner
# and synthesis pipeline at the pinned commit.  Exact-set verification after
# imports makes an added, removed, redirected, or edited Payne Zero module a
# hard identity failure.
FROZEN_PINNED_PYTHON_MANIFEST = {
    "payne_zero_atmosphere/__init__.py": (
        "dbb7734cab4f3e98b9b88d1f1b5ec27afd02fd7fd003ba7931b43d3750049d61"
    ),
    "payne_zero_atmosphere/_numba_cache.py": (
        "b8988812ea92fd5db1e7f092d06ce685e13fbda3b0b7910eba937eb7a4ddeb82"
    ),
    "payne_zero_atmosphere/atmosphere_io.py": (
        "95c4d2cab230f6925e9404639ecb05b25af8c0c85755ac1ca70d760156a8683e"
    ),
    "payne_zero_atmosphere/cli.py": (
        "62974d80798faefbe91f76f2785543303d8d9875866adb9617d21d50e33416ac"
    ),
    "payne_zero_atmosphere/config.py": (
        "51e19846fb81c832ae57334faf3da2c1e4fc2ef9edf6e08467ef7296e4640b45"
    ),
    "payne_zero_atmosphere/constants.py": (
        "ac1f1fbd345dc816eb3e70a8f97ebebc7a4c744fd2759b32ec19f8c88d987036"
    ),
    "payne_zero_atmosphere/continuum_opacity.py": (
        "1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81"
    ),
    "payne_zero_atmosphere/convection.py": (
        "9099af3ce97123a88cfee554cefb55b2b47a52085e3cb6cda19e6869e0fef9fd"
    ),
    "payne_zero_atmosphere/convergence.py": (
        "6b4c674deda148baab6fd90e8a25eed2921581ce7d4bace489c024bc1c2748cb"
    ),
    "payne_zero_atmosphere/data_files.py": (
        "bf89c32977fc2db0454cf597718d99b3f3d15487529ecddbacf717ad6dc245c2"
    ),
    "payne_zero_atmosphere/direct_abundance.py": (
        "ec65683eb344c4c3fd77340c084e780f58c6401e77c9f0d6db05ef6753131445"
    ),
    "payne_zero_atmosphere/doppler.py": (
        "e118a78bf5250ef5e1f77d652c9e78fbb7b92acf5c069f717faed7a3b3ea98f0"
    ),
    "payne_zero_atmosphere/equation_of_state.py": (
        "719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2"
    ),
    "payne_zero_atmosphere/hydrogen_line_profile.py": (
        "6a48f43afee9e326d2f86282f22f44f5654e243a335cc0490c99f86c41451be0"
    ),
    "payne_zero_atmosphere/hydrostatic.py": (
        "f59f7b807152b74f1cf85ed208c612454aa82f62369f5d7baebe3d1a46740fef"
    ),
    "payne_zero_atmosphere/line_catalog.py": (
        "2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92"
    ),
    "payne_zero_atmosphere/line_opacity.py": (
        "d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6"
    ),
    "payne_zero_atmosphere/line_profile_math.py": (
        "9a5794140f00ff3c3fb6c2e3b28461bbc22b471f962d275055c066ad7f8acd15"
    ),
    "payne_zero_atmosphere/line_selection.py": (
        "b2c62fdf5e1fe43f33022184bfeff88985b13331354e3c745c7dab3a6b634fef"
    ),
    "payne_zero_atmosphere/microturbulence.py": (
        "3692062f1d6877e745ed84bba4fc2fdf04c60a7c52bc27856fc696416a0283cb"
    ),
    "payne_zero_atmosphere/molecular_data.py": (
        "705c3072d79c8019c948ce0fa2c82052f232816d453e10a7c8e5fc5a8f5ce249"
    ),
    "payne_zero_atmosphere/molecular_equilibrium.py": (
        "4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86"
    ),
    "payne_zero_atmosphere/population_layout.py": (
        "36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0"
    ),
    "payne_zero_atmosphere/radiative_pressure.py": (
        "c61a256892282d9a0d6cb19714ea5ce6135f1b6f7f573e5761d216d552f321ec"
    ),
    "payne_zero_atmosphere/radiative_transfer.py": (
        "df8970ca629487537a7c4849278eab5d755b527002d8fc58360c9264a3aa45db"
    ),
    "payne_zero_atmosphere/rosseland_mean.py": (
        "91071248fd903e05322b7163d37566e9f894daefc1d7ba018d4850d362f1fc86"
    ),
    "payne_zero_atmosphere/run_setup.py": (
        "de7cf08b936585dbcfa2e572c026fafa3f10282a99c27b834b62db0f3f2888c9"
    ),
    "payne_zero_atmosphere/runner.py": (
        "05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7"
    ),
    "payne_zero_atmosphere/runtime_state.py": (
        "fae240ec00f6f89d7c2a7ef721ce6e6539be234e523291fd6e8a096d731430e8"
    ),
    "payne_zero_atmosphere/source_catalogs.py": (
        "a9ea21735c9d4964b785d76c89c9fc976a30ed75f8b6f9d4f7c6aaa4e77dae36"
    ),
    "payne_zero_atmosphere/specific_internal_energy.py": (
        "de06ba732ce1333d111a52223e39f5b4f80eece8cfc4ff2f30de9739e16d7ec5"
    ),
    "payne_zero_atmosphere/synthesis_bridge.py": (
        "142a960b5e710823754b02766803b3c1dd8c48c9945fdfabe560b4ee7e1acb50"
    ),
    "payne_zero_atmosphere/temperature_correction.py": (
        "67728389ba857511979d0f82ea59f0bf41ee635b8151ae26673dace02b195d21"
    ),
    "payne_zero_atmosphere/transfer_kernels.py": (
        "50e759a085e6aefdb7819a3dbe3ef5e83405834f4b07e0a4de2f3c0e7354d3b9"
    ),
    "payne_zero_atmosphere/warm_start.py": (
        "3a83af3d68be52a35bfc3f55f5912770661be8251cc28f28da3250b2e83e0ad3"
    ),
    "payne_zero_synthesis/__init__.py": (
        "7e94560ab51be3baa3950a93e8b6f998c849a6d1d78356c5f662f6cfaba927db"
    ),
    "payne_zero_synthesis/api.py": (
        "77718303c1e0052a520ece7fab277b3b1922c21d09b35a288596592d03310940"
    ),
    "payne_zero_synthesis/atmosphere.py": (
        "06b79770e4d9472093655022d53ee7fddf7cc6727206f34c0f60c57151e2cf9b"
    ),
    "payne_zero_synthesis/atomic_lines.py": (
        "0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0"
    ),
    "payne_zero_synthesis/constants.py": (
        "ed58004196790f9fb4a2871044c9cd36bf7bc42046a923f9314f7b8ea7456798"
    ),
    "payne_zero_synthesis/continuum.py": (
        "ab0d4eb771ee04101f6936253f633ed60d845e2816854a06b1b059e8b91dce1b"
    ),
    "payne_zero_synthesis/device.py": (
        "22e769ebed60ad3a0f2060264247e469a99afd20ec5cadb69a01b6e5fa82ea3c"
    ),
    "payne_zero_synthesis/equation_of_state.py": (
        "6497c29abb954e0b55d918cc22fa7b660952812c548faf1d7b1053345ef13562"
    ),
    "payne_zero_synthesis/ground_partition_table.py": (
        "6950686c89ea51e301b4b11256d9413dad58d82741a51580f3547aa012ade832"
    ),
    "payne_zero_synthesis/hydrogen_lines.py": (
        "81ab3ee2ca9ecd1994ddde8f01e09535c5b74f7beec5afe98a3c63b44677dcca"
    ),
    "payne_zero_synthesis/line_opacity.py": (
        "639b95c3812f1a7d227b797fa89a4d6ef9725d5f0e1284f3d49cf86844278275"
    ),
    "payne_zero_synthesis/molecular_equilibrium.py": (
        "df01757c160b2bff4390cc2148cff9d1ba6e5a2bc7cab4515b46f38e868d2714"
    ),
    "payne_zero_synthesis/molecular_lines.py": (
        "14c9d07e431fa73e6d6938e9db2d11c6688e52348234e0aac37cc76e8be3dc32"
    ),
    "payne_zero_synthesis/paths.py": (
        "2bca3284eb1765449ab3fc87439eb603e3941213d9c1205c71aee3fd1ad30b5d"
    ),
    "payne_zero_synthesis/pipeline.py": (
        "465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22"
    ),
    "payne_zero_synthesis/radiative_transfer.py": (
        "52e0d1a0c4a2713294ce1b43130c5d900e54c4cf1f8b2b05058fc2d6831ff62b"
    ),
    "payne_zero_synthesis/synthesis.py": (
        "590e430b6582fbcf601a52b721d8f65073432903773a99238073a1d821fe0d0c"
    ),
}

EXPECTED_DATA_HASHES = {
    "atmosphere__continuum_opacity_tables.npz": (
        PINNED_DATA_ROOT / "atmosphere_tables" / "continuum_opacity_tables.npz",
        "6fd4c556418870c28d3fcc9a050252af58ac4cc433cae979477355c8c7d593e3",
    ),
    "atmosphere__karzas_latter_tables.npz": (
        PINNED_DATA_ROOT / "atmosphere_tables" / "karzas_latter_tables.npz",
        "23805dc17c47af45b8ae63b2e278e1fb6c584a01c87d1eb3c31306e4555e6d15",
    ),
    "atmosphere__molecular_equilibrium_tables.npz": (
        PINNED_DATA_ROOT / "atmosphere_tables" / "molecular_equilibrium_tables.npz",
        "1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe",
    ),
    "synthesis__continuum_edge_grid.npz": (
        PINNED_DATA_ROOT / "synthesis_tables" / "continuum_edge_grid.npz",
        "11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e",
    ),
    "synthesis__continuum_tables.npz": (
        PINNED_DATA_ROOT / "synthesis_tables" / "continuum_tables.npz",
        "406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d",
    ),
}

ATMOSPHERE_COMPONENT_ROUTINES = (
    ("hydrogen", "compute_hydrogen_opacity_columns"),
    ("hminus", "compute_hminus_opacity_columns"),
    ("h2plus", "compute_molecular_hydrogen_ion_opacity_columns"),
    ("helium_neutral", "compute_helium_neutral_opacity_columns"),
    ("helium_ionized", "compute_helium_ionized_opacity_columns"),
    ("heminus", "compute_heminus_opacity_columns"),
    ("molecular", "compute_molecular_continuum_opacity_columns"),
    ("carbon_neutral", "compute_carbon_neutral_opacity_columns"),
    ("magnesium_neutral", "compute_magnesium_neutral_opacity_columns"),
    ("aluminum_neutral", "compute_aluminum_neutral_opacity_columns"),
    ("silicon_neutral", "compute_silicon_neutral_opacity_columns"),
    ("iron_neutral", "compute_iron_neutral_opacity_columns"),
    ("lukewarm_metals", "compute_lukewarm_metal_opacity_columns"),
    ("hot_metals", "compute_hot_metal_opacity_columns"),
)

SYNTHESIS_COMPONENT_OUTPUTS = {
    "_hminus_opacity": ("hminus_absorption",),
    "_hydrogen_opacity": ("hydrogen_absorption",),
    "_minor_terms": ("minor_absorption", "minor_scattering"),
    "_helium_opacity": (
        "helium_neutral_absorption",
        "helium_ionized_absorption",
    ),
    "_hot_metal_and_silicon_singly_ionized_opacity": (
        "hot_metal_absorption",
        "light_element_and_si2_absorption",
    ),
    "_scattering_opacity": (
        "hydrogen_rayleigh_scattering",
        "electron_scattering",
    ),
}

SYNTHESIS_PROBE_INTERVALS = np.asarray(
    [50, 75, 100, 125, 150, 170, 200, 225, 250, 275, 300],
    dtype=np.int64,
)

SAMPLED_EXTENSION_WAVELENGTH_NM = np.asarray(
    [
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        320.0,
        400.0,
        500.0,
        700.0,
        1000.0,
        1600.0,
        2500.0,
    ],
    dtype=np.float64,
)


class OracleIdentityError(RuntimeError):
    """Raised when the pin, fixture, source, or table identity has changed."""


class OracleEnvironmentError(RuntimeError):
    """Raised when a deterministic CPU-float64 process cannot be established."""


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_mapping_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash names, dtypes, shapes, and raw bytes for an array mapping."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _expected_fixture_keys() -> set[str]:
    expected = set(FIXTURE_GLOBAL_FIELDS)
    for regime in REGIME_NAMES:
        expected.add(f"{regime}__effective_temperature")
        expected.update(f"{regime}__input__{name}" for name in INPUT_FIELDS)
        expected.update(f"{regime}__atmosphere__{name}" for name in ATMOSPHERE_FIELDS)
        expected.update(f"{regime}__synthesis__{name}" for name in SYNTHESIS_FIELDS)
    return expected


def verify_identity() -> dict[str, str]:
    """Fail closed unless every executed source and static input is pinned."""

    completed = subprocess.run(
        ["git", "-C", str(PINNED_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != PINNED_COMMIT:
        raise OracleIdentityError(
            f"pinned checkout is {commit}; expected {PINNED_COMMIT}"
        )
    identities = {"payne_zero_commit": commit}
    for relative_name, expected in FROZEN_PINNED_PYTHON_MANIFEST.items():
        path = PINNED_ROOT / relative_name
        actual = sha256(path)
        if actual != expected:
            raise OracleIdentityError(
                f"source identity changed for {relative_name}: {actual}"
            )
        key = relative_name.replace("/", "__").removesuffix(".py")
        identities[f"source__{key}__sha256"] = actual
    for name, (path, expected) in EXPECTED_DATA_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise OracleIdentityError(f"data identity changed for {path}: {actual}")
        identities[f"data__{name}__sha256"] = actual
    return identities


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, np.ndarray]:
    """Load the one declared input artifact and validate its complete schema."""

    resolved = Path(path).expanduser().resolve()
    if resolved != FIXTURE_PATH.resolve():
        raise OracleIdentityError(
            f"Chapter 5 accepts only {FIXTURE_PATH.resolve()}, not {resolved}"
        )
    actual_hash = sha256(resolved)
    if actual_hash != FIXTURE_SHA256:
        raise OracleIdentityError(
            f"fixture hash is {actual_hash}; expected {FIXTURE_SHA256}"
        )
    with np.load(resolved, allow_pickle=False) as archive:
        if set(archive.files) != _expected_fixture_keys():
            missing = sorted(_expected_fixture_keys() - set(archive.files))
            extra = sorted(set(archive.files) - _expected_fixture_keys())
            raise OracleIdentityError(
                f"fixture schema changed; missing={missing}, extra={extra}"
            )
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if str(arrays["payne_zero_commit"]) != PINNED_COMMIT:
        raise OracleIdentityError("fixture commit identity changed")
    if tuple(arrays["regime_names"].tolist()) != REGIME_NAMES:
        raise OracleIdentityError("fixture regime order changed")
    flags = arrays["atmosphere_runner_opacity_flags"]
    if flags.shape != (20,) or flags.dtype != np.int64:
        raise OracleIdentityError("runner opacity flags must be int64[20]")
    for regime in REGIME_NAMES:
        for lane, field_names in (
            ("atmosphere", ATMOSPHERE_FIELDS),
            ("synthesis", SYNTHESIS_FIELDS),
        ):
            for field_name in field_names:
                array = arrays[f"{regime}__{lane}__{field_name}"]
                if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                    raise OracleIdentityError(
                        f"fixture contains nonfinite {regime}/{lane}/{field_name}"
                    )
        for field_name in (
            "temperature",
            "gas_pressure",
            "mass_density",
            "electron_density",
        ):
            for lane in ("atmosphere", "synthesis"):
                values = arrays[f"{regime}__{lane}__{field_name}"]
                if values.shape != (6,) or np.any(values <= 0.0):
                    raise OracleIdentityError(
                        f"invalid physical field {regime}/{lane}/{field_name}"
                    )
    return arrays


def require_environment() -> dict[str, np.ndarray]:
    """Require fresh-process determinism and return its recorded metadata."""

    wrong = {
        name: os.environ.get(name)
        for name, expected in ORACLE_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if wrong:
        raise OracleEnvironmentError(
            "fresh one-thread oracle controls are missing: "
            + json.dumps(wrong, sort_keys=True)
        )
    cache_text = os.environ.get("NUMBA_CACHE_DIR", "")
    if not cache_text:
        raise OracleEnvironmentError("NUMBA_CACHE_DIR must name a fresh directory")
    unresolved_cache_path = Path(cache_text).expanduser()
    if unresolved_cache_path.exists() or unresolved_cache_path.is_symlink():
        unresolved_cache_path.lstat()
        if unresolved_cache_path.is_symlink():
            raise OracleEnvironmentError("NUMBA_CACHE_DIR must not be a symlink")
    else:
        unresolved_cache_path.mkdir(parents=True, exist_ok=False)
    cache_path = unresolved_cache_path.resolve()
    if cache_path == PINNED_ROOT or PINNED_ROOT in cache_path.parents:
        raise OracleEnvironmentError("NUMBA_CACHE_DIR must be outside the pin")
    if any(cache_path.iterdir()):
        raise OracleEnvironmentError(
            "NUMBA_CACHE_DIR must be truly empty at process start"
        )
    return {
        f"environment__{name.lower()}": np.asarray(value)
        for name, value in sorted(ORACLE_ENVIRONMENT.items())
    } | {
        "environment__cpu_only": np.asarray(True, dtype=np.bool_),
        "environment__numba_cache_policy": np.asarray(
            "verified empty external temporary directory"
        ),
        "environment__work_dtype": np.asarray("torch.float64"),
    }


def _is_under(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    parent = root.resolve()
    return resolved == parent or parent in resolved.parents


def _loaded_pinned_python_manifest() -> dict[str, str]:
    """Return every currently loaded pinned Payne Zero Python source."""

    loaded: dict[str, str] = {}
    root = PINNED_ROOT.resolve()
    for module in sys.modules.values():
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        path = Path(module_file).resolve()
        if path.suffix != ".py" or not _is_under(path, root):
            continue
        relative_name = str(path.relative_to(root))
        loaded[relative_name] = sha256(path)
    return loaded


def load_pinned_modules() -> SimpleNamespace:
    """Import continuum callers and verify the exact loaded-source manifest."""

    configured_root = os.environ.get("PAYNE_ZERO_DATA_ROOT")
    if configured_root is not None:
        if Path(configured_root).expanduser().resolve() != PINNED_DATA_ROOT.resolve():
            raise OracleIdentityError("PAYNE_ZERO_DATA_ROOT points outside the pin")
    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(PINNED_DATA_ROOT.resolve())

    for package in ("payne_zero_atmosphere", "payne_zero_synthesis"):
        for name, module in tuple(sys.modules.items()):
            if name != package and not name.startswith(f"{package}."):
                continue
            module_file = getattr(module, "__file__", None)
            if module_file is not None and not _is_under(
                Path(module_file), PINNED_ROOT
            ):
                raise OracleIdentityError(
                    f"{name} was imported outside the pin: {module_file}"
                )

    root_text = str(PINNED_ROOT.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    atmosphere = importlib.import_module("payne_zero_atmosphere.continuum_opacity")
    atmosphere_runner = importlib.import_module("payne_zero_atmosphere.runner")
    synthesis = importlib.import_module("payne_zero_synthesis.continuum")
    synthesis_pipeline = importlib.import_module("payne_zero_synthesis.pipeline")
    import numba
    import torch

    for module in (
        atmosphere,
        atmosphere_runner,
        synthesis,
        synthesis_pipeline,
    ):
        if not _is_under(Path(module.__file__), PINNED_ROOT):
            raise OracleIdentityError(
                f"continuum module resolved outside pin: {module.__file__}"
            )
    loaded_manifest = _loaded_pinned_python_manifest()
    if loaded_manifest != FROZEN_PINNED_PYTHON_MANIFEST:
        missing = sorted(set(FROZEN_PINNED_PYTHON_MANIFEST) - set(loaded_manifest))
        extra = sorted(set(loaded_manifest) - set(FROZEN_PINNED_PYTHON_MANIFEST))
        changed = sorted(
            name
            for name in set(loaded_manifest) & set(FROZEN_PINNED_PYTHON_MANIFEST)
            if loaded_manifest[name] != FROZEN_PINNED_PYTHON_MANIFEST[name]
        )
        raise OracleIdentityError(
            "loaded pinned Python manifest changed; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    if (
        atmosphere_runner.compute_continuum_opacity_columns
        is not atmosphere.compute_continuum_opacity_columns
    ):
        raise OracleIdentityError("atmosphere runner continuum binding changed")
    if synthesis_pipeline.continuum_engine is not synthesis:
        raise OracleIdentityError("synthesis pipeline continuum binding changed")
    if int(numba.get_num_threads()) != 1:
        raise OracleEnvironmentError("Numba did not honor NUMBA_NUM_THREADS=1")
    torch.set_num_threads(1)
    if int(torch.get_num_threads()) != 1:
        raise OracleEnvironmentError("Torch CPU thread count is not one")
    tables = synthesis.ContinuumTables.from_npz(
        EXPECTED_DATA_HASHES["synthesis__continuum_tables.npz"][0],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    return SimpleNamespace(
        atmosphere=atmosphere,
        atmosphere_runner=atmosphere_runner,
        synthesis=synthesis,
        synthesis_pipeline=synthesis_pipeline,
        synthesis_tables=tables,
        loaded_python_manifest=loaded_manifest,
        numba=numba,
        torch=torch,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _prefix(
    destination: dict[str, np.ndarray],
    prefix: str,
    arrays: Mapping[str, Any],
) -> None:
    for name, value in arrays.items():
        destination[f"{prefix}__{name}"] = _to_numpy(value)


def _atmosphere_state(
    module: Any,
    fixture: Mapping[str, np.ndarray],
    regime: str,
) -> Any:
    values = {
        field_name: fixture[f"{regime}__atmosphere__{field_name}"].copy()
        for field_name in ATMOSPHERE_FIELDS
    }
    state = module.ContinuumAtmosphereState(**values)
    if tuple(field.name for field in fields(state)) != ATMOSPHERE_FIELDS:
        raise OracleIdentityError("atmosphere continuum dataclass fields changed")
    return state


def _replace_atmosphere_state(
    module: Any,
    state: Any,
    *,
    layer_indices: np.ndarray | None = None,
    **updates: np.ndarray,
) -> Any:
    """Return a detached continuum state with selected layers and overrides."""

    if layer_indices is None:
        layer_indices = np.arange(state.layers, dtype=np.int64)
    indices = np.asarray(layer_indices, dtype=np.int64)
    values: dict[str, np.ndarray] = {}
    for field_name in ATMOSPHERE_FIELDS:
        source = np.asarray(getattr(state, field_name))
        values[field_name] = source[indices].copy()
    for field_name, value in updates.items():
        if field_name not in values:
            raise KeyError(f"unknown atmosphere field {field_name}")
        values[field_name] = np.asarray(value).copy()
    return module.ContinuumAtmosphereState(**values)


def _synthesis_state(
    fixture: Mapping[str, np.ndarray],
    regime: str,
) -> dict[str, np.ndarray]:
    full = {
        field_name: fixture[f"{regime}__synthesis__{field_name}"].copy()
        for field_name in SYNTHESIS_FIELDS
    }
    return {name: full[name] for name in PIPELINE_CONTINUUM_FIELDS}


def _frequency_triplet(center: float) -> np.ndarray:
    return np.asarray(
        [
            np.nextafter(center, -np.inf),
            center,
            np.nextafter(center, np.inf),
        ],
        dtype=np.float64,
    )


def threshold_frequency_probe(
    atmosphere_module: Any,
    fixture: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return named nextafter probes at continuum branch thresholds."""

    c_cm = float(atmosphere_module.LIGHT_SPEED_CM_PER_S_EXACT)
    wavenumber_per_ev = float(atmosphere_module.WAVENUMBER_PER_EV_REFERENCE)
    centers = np.asarray(
        [
            fixture["hminus_edge_frequency_hz"][1],
            c_cm * 20_000.0,
            c_cm * wavenumber_per_ev * 2.0,
            c_cm * wavenumber_per_ev * 10.5,
            c_cm * wavenumber_per_ev * 2.1,
            c_cm * wavenumber_per_ev * 15.0,
            3.28805e15,
            2.463e15,
            2.922e15,
        ],
        dtype=np.float64,
    )
    names = np.asarray(
        [
            "hminus_bound_free",
            "h2_cia_wavenumber",
            "ch_lower_photon_energy",
            "ch_upper_photon_energy",
            "oh_lower_photon_energy",
            "oh_upper_photon_energy",
            "h2plus_upper_frequency",
            "hydrogen_rayleigh_cap",
            "h2_rayleigh_cap",
        ]
    )
    return {
        "center_hz": centers,
        "family": names,
        "frequency_hz": np.concatenate(
            [_frequency_triplet(float(center)) for center in centers]
        ),
        "family_index": np.repeat(np.arange(centers.size, dtype=np.int64), 3),
        "side": np.tile(np.asarray([-1, 0, 1], dtype=np.int8), centers.size),
    }


def _ordered_add(arrays: list[np.ndarray]) -> np.ndarray:
    result = np.zeros_like(arrays[0])
    for array in arrays:
        result = result + array
    return result


def _atmosphere_components(
    module: Any,
    state: Any,
    frequencies_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    absorption_terms: list[np.ndarray] = []
    source_numerator_terms: list[np.ndarray] = []
    for component_name, routine_name in ATMOSPHERE_COMPONENT_ROUTINES:
        absorption, source = getattr(module, routine_name)(state, frequencies_hz)
        absorption = np.asarray(absorption, dtype=np.float64)
        source = np.asarray(source, dtype=np.float64)
        arrays[f"{component_name}__absorption"] = absorption
        arrays[f"{component_name}__source"] = source
        absorption_terms.append(absorption)
        source_numerator_terms.append(absorption * source)
    arrays["ordered_absorption_sum"] = _ordered_add(absorption_terms)
    arrays["ordered_source_numerator_sum"] = _ordered_add(source_numerator_terms)

    flags = np.zeros(20, dtype=np.int64)
    scattering_terms: list[np.ndarray] = []
    for component_name, flag_index in (
        ("electron", 11),
        ("hydrogen_rayleigh", 3),
        ("helium_rayleigh", 7),
    ):
        flags[:] = 0
        flags[flag_index] = 1
        values = module.compute_continuum_scattering_columns(
            state, frequencies_hz, opacity_flags=flags
        )
        arrays[f"{component_name}__scattering"] = values
        scattering_terms.append(values)
    flags[:] = 0
    flags[3] = 1
    hydrogen_only = module.compute_continuum_scattering_columns(
        state, frequencies_hz, opacity_flags=flags
    )
    flags[12] = 1
    hydrogen_and_h2 = module.compute_continuum_scattering_columns(
        state, frequencies_hz, opacity_flags=flags
    )
    h2_scattering = hydrogen_and_h2 - hydrogen_only
    arrays["h2_rayleigh__scattering"] = h2_scattering
    scattering_terms.append(h2_scattering)
    arrays["ordered_scattering_sum"] = _ordered_add(scattering_terms)
    return arrays


def _atmosphere_molecular_components(
    module: Any,
    state: Any,
    frequencies_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    tables = module.load_continuum_opacity_tables()
    molecular_tables = module.load_molecular_equilibrium_tables()
    temperature = np.asarray(state.temperature, dtype=np.float64)
    mass_density = np.maximum(
        np.asarray(state.mass_density, dtype=np.float64), 1.0e-300
    )
    _, _, stimulated = module._planck_frequency_exact(
        temperature_k=temperature,
        frequency_hz=frequencies_hz,
    )
    ch_cross_section = module._ch_molecular_cross_section_grid(
        frequencies_hz, temperature, tables=tables
    )
    oh_cross_section = module._oh_molecular_cross_section_grid(
        frequencies_hz, temperature, tables=tables
    )
    ch_absorption = (
        ch_cross_section
        * (np.asarray(state.ch_population) / mass_density)[:, None]
        * stimulated
    )
    oh_absorption = (
        oh_cross_section
        * (np.asarray(state.oh_population) / mass_density)[:, None]
        * stimulated
    )
    hydrogen_normalized = (
        module._hydrogen_neutral_partition_normalized_population_from_neutral(
            temperature_k=temperature,
            hydrogen_neutral_population=state.hydrogen_neutral_population,
        )
    )
    h2_cia_absorption = module._h2_collision_absorption_grid(
        frequencies_hz,
        temperature_k=temperature,
        hydrogen_neutral_partition_normalized_population=hydrogen_normalized,
        hydrogen_departure_coefficient=np.asarray(
            state.hydrogen_departure_coefficients[:, 0],
            dtype=np.float64,
        ),
        helium_neutral_population=np.asarray(
            state.helium_neutral_population, dtype=np.float64
        ),
        mass_density=mass_density,
        stimulated_emission=stimulated,
        continuum_tables=tables,
        molecular_tables=molecular_tables,
    )
    return {
        "ch_cross_section_times_partition": ch_cross_section,
        "ch_absorption": ch_absorption,
        "oh_cross_section_times_partition": oh_cross_section,
        "oh_absorption": oh_absorption,
        "h2_cia_absorption": h2_cia_absorption,
        "ordered_absorption_sum": _ordered_add(
            [ch_absorption, oh_absorption, h2_cia_absorption]
        ),
    }


def _capture_molecular_entry_counterfactuals(
    module: Any,
    fixture: Mapping[str, np.ndarray],
    frequencies_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    """Exercise global entry, molecule-disable, and owner-level H2 gates."""

    solar = _atmosphere_state(module, fixture, "solar_dwarf")
    all_warm_temperature = np.asarray(
        [9000.0, 9500.0, 10000.0, 11000.0, 12000.0, 13000.0]
    )
    mixed_temperature = all_warm_temperature.copy()
    mixed_temperature[0] = 8999.0
    all_warm = _replace_atmosphere_state(
        module, solar, temperature=all_warm_temperature
    )
    mixed = _replace_atmosphere_state(module, solar, temperature=mixed_temperature)
    all_warm_absorption, _ = module.compute_molecular_continuum_opacity_columns(
        all_warm, frequencies_hz
    )
    mixed_absorption, _ = module.compute_molecular_continuum_opacity_columns(
        mixed, frequencies_hz
    )

    cool = _atmosphere_state(module, fixture, "cool_molecule_rich")
    molecule_disabled = _replace_atmosphere_state(
        module,
        cool,
        ch_population=np.zeros(cool.layers, dtype=np.float64),
        oh_population=np.zeros(cool.layers, dtype=np.float64),
    )
    disabled_total, _ = module.compute_molecular_continuum_opacity_columns(
        molecule_disabled, frequencies_hz
    )
    disabled_components = _atmosphere_molecular_components(
        module, molecule_disabled, frequencies_hz
    )

    cutoff_temperature = np.asarray(
        [19999.0, 20000.0, np.nextafter(20000.0, np.inf)],
        dtype=np.float64,
    )
    cutoff_state = _replace_atmosphere_state(
        module,
        cool,
        layer_indices=np.asarray([0, 1, 2]),
        temperature=cutoff_temperature,
        ch_population=np.zeros(3, dtype=np.float64),
        oh_population=np.zeros(3, dtype=np.float64),
    )
    cutoff_components = _atmosphere_molecular_components(
        module, cutoff_state, frequencies_hz
    )
    flags = np.zeros(20, dtype=np.int64)
    flags[3] = 1
    hydrogen_rayleigh = module.compute_continuum_scattering_columns(
        cutoff_state, frequencies_hz, opacity_flags=flags
    )
    flags[12] = 1
    hydrogen_and_h2_rayleigh = module.compute_continuum_scattering_columns(
        cutoff_state, frequencies_hz, opacity_flags=flags
    )
    h2_rayleigh = hydrogen_and_h2_rayleigh - hydrogen_rayleigh
    expected_owner_active = cutoff_temperature <= 20000.0
    cia_active_by_layer = np.any(cutoff_components["h2_cia_absorption"] > 0.0, axis=1)
    rayleigh_active_by_layer = np.any(h2_rayleigh > 0.0, axis=1)

    if np.count_nonzero(all_warm_absorption) != 0:
        raise RuntimeError("all-warm molecular composite unexpectedly entered")
    if np.count_nonzero(mixed_absorption) == 0:
        raise RuntimeError("mixed molecular composite failed to enter")
    if np.count_nonzero(disabled_components["ch_absorption"]) != 0:
        raise RuntimeError("molecule-disabled CH absorption is nonzero")
    if np.count_nonzero(disabled_components["oh_absorption"]) != 0:
        raise RuntimeError("molecule-disabled OH absorption is nonzero")
    if not np.array_equal(disabled_total, disabled_components["h2_cia_absorption"]):
        raise RuntimeError("molecule-disabled local H2 ownership changed")
    if not np.array_equal(cia_active_by_layer, expected_owner_active):
        raise RuntimeError("H2-CIA 20000 K owner cutoff changed")
    if not np.array_equal(rayleigh_active_by_layer, expected_owner_active):
        raise RuntimeError("H2 Rayleigh 20000 K owner cutoff changed")
    return {
        "frequency_hz": frequencies_hz,
        "all_warm_temperature_k": all_warm_temperature,
        "all_warm_absorption": all_warm_absorption,
        "mixed_temperature_k": mixed_temperature,
        "mixed_absorption": mixed_absorption,
        "all_warm_entry_active": np.asarray(False, dtype=np.bool_),
        "mixed_entry_active": np.asarray(True, dtype=np.bool_),
        "molecule_disabled_ch_population": molecule_disabled.ch_population,
        "molecule_disabled_oh_population": molecule_disabled.oh_population,
        "molecule_disabled_total_absorption": disabled_total,
        "molecule_disabled_ch_absorption": disabled_components["ch_absorption"],
        "molecule_disabled_oh_absorption": disabled_components["oh_absorption"],
        "molecule_disabled_local_h2_cia_absorption": disabled_components[
            "h2_cia_absorption"
        ],
        "owner_cutoff_temperature_k": cutoff_temperature,
        "owner_cutoff_expected_active": expected_owner_active,
        "owner_cutoff_h2_cia_absorption": cutoff_components["h2_cia_absorption"],
        "owner_cutoff_h2_cia_active_by_layer": cia_active_by_layer,
        "owner_cutoff_h2_rayleigh_scattering": h2_rayleigh,
        "owner_cutoff_h2_rayleigh_active_by_layer": (rayleigh_active_by_layer),
    }


def _capture_atmosphere_regime(
    module: Any,
    fixture: Mapping[str, np.ndarray],
    regime: str,
    frequencies_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    state = _atmosphere_state(module, fixture, regime)
    flags = np.asarray(fixture["atmosphere_runner_opacity_flags"], dtype=np.int64)
    absorption, scattering, source = module.compute_continuum_opacity_columns(
        state, frequencies_hz, opacity_flags=flags
    )
    arrays = {
        "frequency_hz": frequencies_hz,
        "runner_opacity_flags": flags,
        "absorption": absorption,
        "scattering": scattering,
        "source": source,
    }
    components = _atmosphere_components(module, state, frequencies_hz)
    _prefix(arrays, "component", components)
    molecular = _atmosphere_molecular_components(module, state, frequencies_hz)
    _prefix(arrays, "molecular_component", molecular)
    arrays["component__absorption_residual"] = (
        absorption - components["ordered_absorption_sum"]
    )
    arrays["component__scattering_residual"] = (
        scattering - components["ordered_scattering_sum"]
    )
    independent_planck, _, _ = module._planck_frequency_exact(
        temperature_k=state.temperature,
        frequency_hz=frequencies_hz,
    )
    reconstructed_source = np.asarray(independent_planck).copy()
    active = absorption > 0.0
    reconstructed_source[active] = (
        components["ordered_source_numerator_sum"][active] / absorption[active]
    )
    arrays["component__source_residual"] = source - reconstructed_source
    activation_names = np.asarray(
        [
            *(f"{name}_absorption" for name, _ in ATMOSPHERE_COMPONENT_ROUTINES),
            "electron_scattering",
            "hydrogen_rayleigh_scattering",
            "helium_rayleigh_scattering",
            "h2_rayleigh_scattering",
            "ch_absorption",
            "oh_absorption",
            "h2_cia_absorption",
        ]
    )
    activation_arrays = [
        *(
            components[f"{name}__absorption"]
            for name, _ in ATMOSPHERE_COMPONENT_ROUTINES
        ),
        components["electron__scattering"],
        components["hydrogen_rayleigh__scattering"],
        components["helium_rayleigh__scattering"],
        components["h2_rayleigh__scattering"],
        molecular["ch_absorption"],
        molecular["oh_absorption"],
        molecular["h2_cia_absorption"],
    ]
    arrays["activation__component_name"] = activation_names
    arrays["activation__active"] = np.asarray(
        [np.any(values > 0.0) for values in activation_arrays],
        dtype=np.bool_,
    )
    arrays["activation__nonzero_count"] = np.asarray(
        [np.count_nonzero(values) for values in activation_arrays],
        dtype=np.int64,
    )
    arrays["activation__maximum"] = np.asarray(
        [np.max(values) for values in activation_arrays], dtype=np.float64
    )

    effective_temperature = float(fixture[f"{regime}__effective_temperature"])
    sampling_wavelength, sampling_weight = module.build_opacity_sampling_grid(
        effective_temperature
    )
    product_frequency = float(module.LIGHT_SPEED_NM_PER_S) / np.maximum(
        sampling_wavelength, 1.0e-300
    )
    product_absorption, product_scattering, product_source = (
        module.compute_continuum_opacity_columns(
            state,
            product_frequency,
            opacity_flags=flags,
        )
    )
    arrays["sampling_wavelength_nm"] = sampling_wavelength
    arrays["sampling_frequency_weight_hz"] = sampling_weight
    arrays["product_frequency_hz"] = product_frequency
    arrays["product_absorption"] = product_absorption
    arrays["product_scattering"] = product_scattering
    arrays["product_source"] = product_source
    active_indices, active_frequencies = module.active_continuum_reference_frequencies(
        effective_temperature
    )
    active_absorption, active_scattering, active_source = (
        module.compute_continuum_opacity_columns(
            state, active_frequencies, opacity_flags=flags
        )
    )
    threshold, reference_wavelength, packed_index = (
        module.assemble_continuum_line_selection_threshold(
            effective_temperature=effective_temperature,
            temperature_k=state.temperature,
            active_continuum_absorption=active_absorption,
            active_continuum_scattering=active_scattering,
        )
    )
    arrays.update(
        {
            "line_reference_active_index": active_indices,
            "line_reference_active_frequency_hz": active_frequencies,
            "line_reference_active_absorption": active_absorption,
            "line_reference_active_scattering": active_scattering,
            "line_reference_active_source": active_source,
            "line_reference_threshold": threshold,
            "line_reference_wavelength_nm": reference_wavelength,
            "line_reference_packed_wavelength_index": packed_index,
        }
    )
    return arrays


class SynthesisComponentCapture(AbstractContextManager["SynthesisComponentCapture"]):
    """Observe scalar synthesis components without changing their values."""

    def __init__(self, module: Any):
        self.module = module
        self.originals: dict[str, Any] = {}
        self.records: dict[str, list[np.ndarray]] = {
            output_name: []
            for output_names in SYNTHESIS_COMPONENT_OUTPUTS.values()
            for output_name in output_names
        }

    def __enter__(self) -> "SynthesisComponentCapture":
        for routine_name, output_names in SYNTHESIS_COMPONENT_OUTPUTS.items():
            original = getattr(self.module, routine_name)
            self.originals[routine_name] = original

            def wrapper(
                *args: Any,
                __original: Any = original,
                __output_names: tuple[str, ...] = output_names,
                **kwargs: Any,
            ) -> Any:
                result = __original(*args, **kwargs)
                values = result if len(__output_names) > 1 else (result,)
                for output_name, value in zip(__output_names, values):
                    self.records[output_name].append(_to_numpy(value))
                return result

            setattr(wrapper, "__chapter05_capture_wrapper__", True)
            setattr(self.module, routine_name, wrapper)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for name, original in self.originals.items():
            setattr(self.module, name, original)
        self.originals.clear()
        return None

    def arrays(self, frequency_count: int) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for output_name, records in self.records.items():
            if len(records) != frequency_count:
                raise RuntimeError(
                    f"{output_name} captured {len(records)} columns, "
                    f"expected {frequency_count}"
                )
            arrays[output_name] = np.stack(records, axis=1)
        absorption = _ordered_add(
            [
                arrays["hminus_absorption"],
                arrays["hydrogen_absorption"],
                arrays["minor_absorption"],
                arrays["helium_neutral_absorption"],
                arrays["helium_ionized_absorption"],
                arrays["hot_metal_absorption"],
                arrays["light_element_and_si2_absorption"],
            ]
        )
        scattering = _ordered_add(
            [
                arrays["hydrogen_rayleigh_scattering"],
                arrays["electron_scattering"],
                arrays["minor_scattering"],
            ]
        )
        arrays["ordered_absorption_sum"] = absorption
        arrays["ordered_scattering_sum"] = scattering
        return arrays


def synthesis_route_probe(
    module: Any,
    state: Mapping[str, np.ndarray],
    fixture: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Construct an edge-rich requested grid and its exact interpolation basis."""

    edge = np.asarray(state["continuum_edge_wavelength_nm"], np.float64)
    midpoint = np.asarray(state["continuum_edge_midpoint_wavelength_nm"], np.float64)
    width2 = np.asarray(
        state["continuum_edge_interval_width_squared_over_two_nm2"],
        np.float64,
    )
    requested_parts: list[np.ndarray] = []
    for index in SYNTHESIS_PROBE_INTERVALS:
        left = float(edge[index])
        right = float(edge[index + 1])
        requested_parts.append(
            np.asarray(
                [
                    np.nextafter(left, right),
                    midpoint[index],
                    np.nextafter(right, left),
                ],
                dtype=np.float64,
            )
        )
    requested_parts.append(
        np.asarray(fixture["synthesis_edge_probe_wavelength_nm"], np.float64)
    )
    requested = np.concatenate(requested_parts)
    edge_indices = np.clip(
        np.searchsorted(edge, np.abs(requested), side="right") - 1,
        0,
        edge.size - 2,
    )
    used = np.unique(edge_indices)
    signed_frequency = np.asarray(
        state["signed_continuum_edge_frequency_hz"], np.float64
    )
    all_sample_frequency = module.build_edge_sample_frequencies(signed_frequency, edge)
    edge_path = EXPECTED_DATA_HASHES["synthesis__continuum_edge_grid.npz"][0]
    with np.load(edge_path, allow_pickle=False) as archive:
        packaged_sample_frequency = np.asarray(
            archive["continuum_edge_sample_frequency_hz"], dtype=np.float64
        )
    packaged_sample_bit_equal = np.array_equal(
        all_sample_frequency.view(np.uint64),
        packaged_sample_frequency.view(np.uint64),
    )
    if not packaged_sample_bit_equal:
        raise RuntimeError("packaged 1020 edge-sample frequencies changed")
    sample_indices = np.concatenate(
        [np.asarray([3 * index, 3 * index + 1, 3 * index + 2]) for index in used]
    ).astype(np.int64)

    left_basis = np.empty(requested.size, dtype=np.float64)
    center_basis = np.empty_like(left_basis)
    right_basis = np.empty_like(left_basis)
    for output_index, (wavelength, index) in enumerate(zip(requested, edge_indices)):
        denominator = width2[index] if width2[index] != 0.0 else 1.0e-20
        left_basis[output_index] = (
            (wavelength - midpoint[index])
            * (wavelength - edge[index + 1])
            / denominator
        )
        center_basis[output_index] = (
            (edge[index] - wavelength)
            * (wavelength - edge[index + 1])
            * 2.0
            / denominator
        )
        right_basis[output_index] = (
            (wavelength - edge[index]) * (wavelength - midpoint[index]) / denominator
        )
    return {
        "probe_interval_index": SYNTHESIS_PROBE_INTERVALS,
        "requested_wavelength_nm": requested,
        "requested_edge_index": edge_indices,
        "used_edge_index": used,
        "all_sample_frequency_hz": all_sample_frequency,
        "packaged_sample_frequency_hz": packaged_sample_frequency,
        "packaged_sample_frequency_bit_equal": np.asarray(
            packaged_sample_bit_equal, dtype=np.bool_
        ),
        "selected_sample_index": sample_indices,
        "selected_sample_frequency_hz": all_sample_frequency[sample_indices],
        "selected_sample_edge_index": np.repeat(used, 3),
        "selected_sample_side": np.tile(
            np.asarray([-1, 0, 1], dtype=np.int8), used.size
        ),
        "left_basis": left_basis,
        "center_basis": center_basis,
        "right_basis": right_basis,
        "basis_sum": left_basis + center_basis + right_basis,
        "signed_edge_frequency_hz": signed_frequency,
        "edge_wavelength_nm": edge,
        "edge_midpoint_wavelength_nm": midpoint,
        "edge_interval_width_squared_over_two_nm2": width2,
    }


def _reconstruct_synthesis_interpolation(
    sample_values: np.ndarray,
    route: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Independently reproduce the standard log-parabolic interpolation."""

    values = np.asarray(sample_values, dtype=np.float64)
    used_edges = np.asarray(route["used_edge_index"], dtype=np.int64)
    requested_edges = np.asarray(route["requested_edge_index"], dtype=np.int64)
    left_basis = np.asarray(route["left_basis"], dtype=np.float64)
    center_basis = np.asarray(route["center_basis"], dtype=np.float64)
    right_basis = np.asarray(route["right_basis"], dtype=np.float64)
    reconstructed = np.empty((values.shape[0], requested_edges.size), dtype=np.float64)
    edge_to_block = {
        int(edge_index): block_index
        for block_index, edge_index in enumerate(used_edges)
    }
    for wavelength_index, edge_index in enumerate(requested_edges):
        block = edge_to_block[int(edge_index)]
        sample_slice = slice(3 * block, 3 * block + 3)
        logarithm = np.log10(np.maximum(values[:, sample_slice], 1.0e-30))
        interpolated_logarithm = (
            logarithm[:, 0] * left_basis[wavelength_index]
            + logarithm[:, 1] * center_basis[wavelength_index]
            + logarithm[:, 2] * right_basis[wavelength_index]
        )
        reconstructed[:, wavelength_index] = 10.0**interpolated_logarithm
    return reconstructed


def _capture_sampled_extension(
    module: Any,
    tables: Any,
    pops: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Run the explicit finite precomputed-invariants extension lane."""

    wavelength = SAMPLED_EXTENSION_WAVELENGTH_NM.copy()
    frequency = float(module.LIGHT_SPEED_NM_PER_S) / wavelength
    invariants = module.build_frequency_invariants(
        tables,
        frequency,
        coulomb_table_energy_first=True,
    )
    absorption, scattering, source = module.compute_sampled_continuum(
        tables,
        frequency,
        dict(pops),
        frequency_invariants=invariants,
    )
    arrays: dict[str, np.ndarray] = {
        "support_wavelength_min_nm": np.asarray(wavelength[0]),
        "support_wavelength_max_nm": np.asarray(wavelength[-1]),
        "wavelength_nm": wavelength,
        "frequency_hz": frequency,
        "coulomb_table_energy_first": np.asarray(True, dtype=np.bool_),
        "frequency_invariants_supplied": np.asarray(True, dtype=np.bool_),
        "continuum_atmosphere_field_names": np.asarray(PIPELINE_CONTINUUM_FIELDS),
        "pops_field_names": np.asarray(tuple(pops)),
        "absorption": _to_numpy(absorption),
        "scattering": _to_numpy(scattering),
        "source": _to_numpy(source),
    }
    for name, value in pops.items():
        arrays[f"input__pops__{name}"] = _to_numpy(value)
    invariant_field_names: list[str] = []
    for field in fields(invariants):
        value = getattr(invariants, field.name)
        if isinstance(value, dict):
            continue
        invariant_field_names.append(field.name)
        arrays[f"input__invariant__{field.name}"] = _to_numpy(value)
    arrays["invariant_field_names"] = np.asarray(invariant_field_names)
    return arrays


def _capture_isolated_minor_terms(
    module: Any,
    tables: Any,
    pops: Mapping[str, Any],
    frequencies_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split `_minor_terms` through isolated population-state calls."""

    controlled_fields = (
        "electron_density",
        "hydrogen_partition_normalized_ion_stage_populations",
        "hydrogen_neutral_population",
        "helium_neutral_population",
        "carbon_partition_normalized_ion_stage_populations",
        "magnesium_neutral_partition_normalized_population",
        "aluminum_neutral_partition_normalized_population",
        "silicon_neutral_partition_normalized_population",
    )
    cases = {
        "h2plus": ("hydrogen_partition_normalized_ion_stage_populations",),
        "heminus_and_helium_rayleigh": (
            "electron_density",
            "helium_neutral_population",
        ),
        "compact_carbon": ("carbon_partition_normalized_ion_stage_populations",),
        "compact_magnesium": ("magnesium_neutral_partition_normalized_population",),
        "compact_aluminum": ("aluminum_neutral_partition_normalized_population",),
        "compact_silicon": ("silicon_neutral_partition_normalized_population",),
        "h2_rayleigh": ("hydrogen_neutral_population",),
    }
    temperature = pops["temperature"]
    hc_over_kt = (
        module.PLANCK_ERG_SECOND
        / (module.BOLTZMANN_ERG_PER_K * temperature)
        * module.LIGHT_SPEED_CM_PER_S
    )
    hydrogen_partition = module._hydrogen_partition(tables, temperature)
    arrays: dict[str, np.ndarray] = {}
    for case_name, active_fields in cases.items():
        isolated = dict(pops)
        for field_name in controlled_fields:
            original = pops[field_name]
            isolated[field_name] = (
                original.clone()
                if field_name in active_fields
                else module.torch.zeros_like(original)
            )
        absorption_columns: list[np.ndarray] = []
        scattering_columns: list[np.ndarray] = []
        for frequency in frequencies_hz:
            photon_boltzmann = module.torch.exp(
                -module.PLANCK_ERG_SECOND
                * float(frequency)
                / (module.BOLTZMANN_ERG_PER_K * temperature)
            )
            stimulated = 1.0 - photon_boltzmann
            absorption, scattering = module._minor_terms(
                tables,
                float(frequency),
                isolated,
                photon_boltzmann,
                stimulated,
                hc_over_kt,
                coulomb_table_energy_first=False,
                hydrogen_partition=hydrogen_partition,
            )
            absorption_columns.append(_to_numpy(absorption))
            scattering_columns.append(_to_numpy(scattering))
        arrays[f"{case_name}__absorption"] = np.stack(absorption_columns, axis=1)
        arrays[f"{case_name}__scattering"] = np.stack(scattering_columns, axis=1)
    arrays["ordered_absorption_sum"] = _ordered_add(
        [
            arrays["h2plus__absorption"],
            arrays["heminus_and_helium_rayleigh__absorption"],
            arrays["compact_carbon__absorption"],
            arrays["compact_magnesium__absorption"],
            arrays["compact_aluminum__absorption"],
            arrays["compact_silicon__absorption"],
        ]
    )
    arrays["ordered_scattering_sum"] = _ordered_add(
        [
            arrays["heminus_and_helium_rayleigh__scattering"],
            arrays["h2_rayleigh__scattering"],
        ]
    )
    return arrays


def _capture_synthesis_regime(
    module: Any,
    tables: Any,
    fixture: Mapping[str, np.ndarray],
    regime: str,
    diagnostic_frequency_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    state = _synthesis_state(fixture, regime)
    route = synthesis_route_probe(module, state, fixture)
    requested = route["requested_wavelength_nm"]
    original_compute_at_freqs = module._compute_at_freqs
    standard_frequency_calls: list[np.ndarray] = []

    def traced_compute_at_freqs(*args: Any, **kwargs: Any) -> Any:
        standard_frequency_calls.append(np.asarray(args[1], dtype=np.float64).copy())
        return original_compute_at_freqs(*args, **kwargs)

    module._compute_at_freqs = traced_compute_at_freqs
    try:
        standard_absorption, standard_scattering = module.continuum(
            requested, state, tables
        )
    finally:
        module._compute_at_freqs = original_compute_at_freqs
    if len(standard_frequency_calls) != 1:
        raise RuntimeError("standard continuum sampled more than one block")
    traced_frequency = standard_frequency_calls[0]
    if not np.array_equal(
        traced_frequency.view(np.uint64),
        route["selected_sample_frequency_hz"].view(np.uint64),
    ):
        raise RuntimeError("standard continuum sampled an unused edge interval")
    unused_edge_mask = np.ones(340, dtype=np.bool_)
    unused_edge_mask[route["used_edge_index"]] = False
    unused_sample_indices = np.concatenate(
        [
            np.asarray([3 * index, 3 * index + 1, 3 * index + 2])
            for index in np.nonzero(unused_edge_mask)[0]
        ]
    )
    unused_frequency = route["all_sample_frequency_hz"][unused_sample_indices]
    unused_frequency_was_called = np.isin(
        unused_frequency.view(np.uint64), traced_frequency.view(np.uint64)
    )
    if np.any(unused_frequency_was_called):
        raise RuntimeError("unused synthesis edge sample was evaluated")

    rich_hii_state = dict(state)
    rich_hii_state["hydrogen_ionized_population"] = fixture[
        f"{regime}__synthesis__hydrogen_ionized_population"
    ].copy()
    rich_hii_absorption, rich_hii_scattering = module.continuum(
        requested, rich_hii_state, tables
    )
    perturbed_rich_hii_state = dict(rich_hii_state)
    perturbed_rich_hii_state["hydrogen_ionized_population"] = (
        np.asarray(rich_hii_state["hydrogen_ionized_population"]) * 1.5
    )
    perturbed_rich_hii_absorption, perturbed_rich_hii_scattering = module.continuum(
        requested, perturbed_rich_hii_state, tables
    )
    schema_h2_state = dict(state)
    schema_h2 = fixture[f"{regime}__synthesis__molecular_hydrogen_population"].copy()
    schema_h2_state["molecular_hydrogen_population"] = schema_h2
    schema_h2_absorption, schema_h2_scattering = module.continuum(
        requested, schema_h2_state, tables
    )
    perturbed_h2_state = dict(schema_h2_state)
    perturbed_h2 = schema_h2 * 17.0 + 1.0
    perturbed_h2_state["molecular_hydrogen_population"] = perturbed_h2
    perturbed_h2_absorption, perturbed_h2_scattering = module.continuum(
        requested, perturbed_h2_state, tables
    )
    flipped_edge_state = dict(state)
    flipped_edge_state["signed_continuum_edge_frequency_hz"] = -np.asarray(
        state["signed_continuum_edge_frequency_hz"]
    )
    flipped_edge_absorption, flipped_edge_scattering = module.continuum(
        requested, flipped_edge_state, tables
    )

    standard_absorption_numpy = _to_numpy(standard_absorption)
    standard_scattering_numpy = _to_numpy(standard_scattering)
    rich_hii_absorption_numpy = _to_numpy(rich_hii_absorption)
    rich_hii_scattering_numpy = _to_numpy(rich_hii_scattering)
    perturbed_rich_hii_absorption_numpy = _to_numpy(perturbed_rich_hii_absorption)
    perturbed_rich_hii_scattering_numpy = _to_numpy(perturbed_rich_hii_scattering)
    schema_h2_absorption_numpy = _to_numpy(schema_h2_absorption)
    schema_h2_scattering_numpy = _to_numpy(schema_h2_scattering)
    perturbed_h2_absorption_numpy = _to_numpy(perturbed_h2_absorption)
    perturbed_h2_scattering_numpy = _to_numpy(perturbed_h2_scattering)
    flipped_edge_absorption_numpy = _to_numpy(flipped_edge_absorption)
    flipped_edge_scattering_numpy = _to_numpy(flipped_edge_scattering)
    if not np.array_equal(rich_hii_scattering_numpy, standard_scattering_numpy):
        raise RuntimeError("H II view unexpectedly changed scattering")
    if np.array_equal(perturbed_rich_hii_absorption_numpy, rich_hii_absorption_numpy):
        raise RuntimeError("rich H II population is not consumed")
    if not np.array_equal(
        perturbed_rich_hii_scattering_numpy, rich_hii_scattering_numpy
    ):
        raise RuntimeError("perturbed rich H II changed scattering")
    if not np.array_equal(
        schema_h2_absorption_numpy, perturbed_h2_absorption_numpy
    ) or not np.array_equal(schema_h2_scattering_numpy, perturbed_h2_scattering_numpy):
        raise RuntimeError("stored schema H2 became a continuum input")
    if not np.array_equal(
        flipped_edge_absorption_numpy, standard_absorption_numpy
    ) or not np.array_equal(flipped_edge_scattering_numpy, standard_scattering_numpy):
        raise RuntimeError("signed-edge magnitude invariance changed")

    pops = module.build_pops(state, device=tables.device, dtype=tables.dtype)
    selected_frequency = route["selected_sample_frequency_hz"]
    with SynthesisComponentCapture(module) as capture:
        sample_absorption, sample_scattering = module._compute_at_freqs(
            tables,
            selected_frequency,
            pops,
            coulomb_table_energy_first=False,
            frequency_invariants=None,
        )
    standard_components = capture.arrays(selected_frequency.size)

    with SynthesisComponentCapture(module) as diagnostic_capture:
        diagnostic_absorption, diagnostic_scattering, diagnostic_source = (
            module.compute_sampled_continuum(
                tables,
                diagnostic_frequency_hz,
                pops,
                frequency_invariants=None,
            )
        )
    diagnostic_components = diagnostic_capture.arrays(diagnostic_frequency_hz.size)
    wrappers_restored = not any(
        getattr(
            getattr(module, routine_name),
            "__chapter05_capture_wrapper__",
            False,
        )
        for routine_name in SYNTHESIS_COMPONENT_OUTPUTS
    )
    if not wrappers_restored:
        raise RuntimeError("synthesis component wrappers were not restored")
    reconstructed_absorption = _reconstruct_synthesis_interpolation(
        _to_numpy(sample_absorption), route
    )
    reconstructed_scattering = _reconstruct_synthesis_interpolation(
        _to_numpy(sample_scattering), route
    )
    extension = _capture_sampled_extension(module, tables, pops)
    isolated_minor = _capture_isolated_minor_terms(
        module, tables, pops, selected_frequency
    )
    isolated_minor["absorption_residual"] = (
        standard_components["minor_absorption"]
        - isolated_minor["ordered_absorption_sum"]
    )
    isolated_minor["scattering_residual"] = (
        standard_components["minor_scattering"]
        - isolated_minor["ordered_scattering_sum"]
    )

    arrays: dict[str, np.ndarray] = {}
    _prefix(arrays, "route", route)
    arrays.update(
        {
            "standard__absorption": _to_numpy(standard_absorption),
            "standard__scattering": _to_numpy(standard_scattering),
            "standard__trace__compute_at_freqs_call_count": np.asarray(
                len(standard_frequency_calls), dtype=np.int64
            ),
            "standard__trace__called_frequency_hz": traced_frequency,
            "standard__trace__unused_edge_mask": unused_edge_mask,
            "standard__trace__unused_sample_index": unused_sample_indices,
            "standard__trace__unused_sample_was_called": (unused_frequency_was_called),
            "standard__trace__unused_sample_call_count": np.asarray(
                np.count_nonzero(unused_frequency_was_called), dtype=np.int64
            ),
            "standard__sample_absorption": _to_numpy(sample_absorption),
            "standard__sample_scattering": _to_numpy(sample_scattering),
            "standard__reconstructed_absorption": reconstructed_absorption,
            "standard__reconstructed_scattering": reconstructed_scattering,
            "standard__interpolation_absorption_residual": (
                _to_numpy(standard_absorption) - reconstructed_absorption
            ),
            "standard__interpolation_scattering_residual": (
                _to_numpy(standard_scattering) - reconstructed_scattering
            ),
            "standard__coulomb_table_energy_first": np.asarray(False, dtype=np.bool_),
            "standard__frequency_invariants_supplied": np.asarray(
                False, dtype=np.bool_
            ),
            "diagnostic__frequency_hz": diagnostic_frequency_hz,
            "diagnostic__absorption": _to_numpy(diagnostic_absorption),
            "diagnostic__scattering": _to_numpy(diagnostic_scattering),
            "diagnostic__source": _to_numpy(diagnostic_source),
            "diagnostic__coulomb_table_energy_first": np.asarray(True, dtype=np.bool_),
            "diagnostic__frequency_invariants_supplied": np.asarray(
                False, dtype=np.bool_
            ),
            "counterfactual__rich_hii_population": rich_hii_state[
                "hydrogen_ionized_population"
            ],
            "counterfactual__trimmed_hii_fallback_population": state[
                "hydrogen_partition_normalized_ion_stage_populations"
            ][:, 1],
            "counterfactual__rich_hii_absorption": rich_hii_absorption_numpy,
            "counterfactual__rich_hii_scattering": rich_hii_scattering_numpy,
            "counterfactual__rich_minus_trimmed_hii_absorption": (
                rich_hii_absorption_numpy - standard_absorption_numpy
            ),
            "counterfactual__rich_hii_matches_trimmed": np.asarray(
                np.array_equal(rich_hii_absorption_numpy, standard_absorption_numpy),
                dtype=np.bool_,
            ),
            "counterfactual__rich_hii_perturbed_population": (
                perturbed_rich_hii_state["hydrogen_ionized_population"]
            ),
            "counterfactual__rich_hii_perturbed_absorption": (
                perturbed_rich_hii_absorption_numpy
            ),
            "counterfactual__rich_hii_perturbed_scattering": (
                perturbed_rich_hii_scattering_numpy
            ),
            "counterfactual__schema_h2_original": schema_h2,
            "counterfactual__schema_h2_perturbed": perturbed_h2,
            "counterfactual__schema_h2_original_absorption": (
                schema_h2_absorption_numpy
            ),
            "counterfactual__schema_h2_perturbed_absorption": (
                perturbed_h2_absorption_numpy
            ),
            "counterfactual__schema_h2_original_scattering": (
                schema_h2_scattering_numpy
            ),
            "counterfactual__schema_h2_perturbed_scattering": (
                perturbed_h2_scattering_numpy
            ),
            "counterfactual__schema_h2_bit_invariant": np.asarray(True, dtype=np.bool_),
            "counterfactual__signed_edge_original": state[
                "signed_continuum_edge_frequency_hz"
            ],
            "counterfactual__signed_edge_flipped": flipped_edge_state[
                "signed_continuum_edge_frequency_hz"
            ],
            "counterfactual__signed_edge_flipped_absorption": (
                flipped_edge_absorption_numpy
            ),
            "counterfactual__signed_edge_flipped_scattering": (
                flipped_edge_scattering_numpy
            ),
            "counterfactual__signed_edge_bit_invariant": np.asarray(
                True, dtype=np.bool_
            ),
        }
    )
    _prefix(arrays, "extension", extension)
    _prefix(arrays, "standard__isolated_minor", isolated_minor)
    _prefix(arrays, "standard__component", standard_components)
    _prefix(arrays, "diagnostic__component", diagnostic_components)
    arrays["standard__component__absorption_residual"] = (
        _to_numpy(sample_absorption) - standard_components["ordered_absorption_sum"]
    )
    arrays["standard__component__scattering_residual"] = (
        _to_numpy(sample_scattering) - standard_components["ordered_scattering_sum"]
    )
    arrays["diagnostic__component__absorption_residual"] = (
        _to_numpy(diagnostic_absorption)
        - diagnostic_components["ordered_absorption_sum"]
    )
    arrays["diagnostic__component__scattering_residual"] = (
        _to_numpy(diagnostic_scattering)
        - diagnostic_components["ordered_scattering_sum"]
    )
    activation_names = np.asarray(
        [
            "hminus_absorption",
            "hydrogen_absorption",
            "minor_absorption",
            "helium_neutral_absorption",
            "helium_ionized_absorption",
            "hot_metal_absorption",
            "light_element_and_si2_absorption",
            "hydrogen_rayleigh_scattering",
            "electron_scattering",
            "minor_scattering",
        ]
    )
    activation_arrays = [
        standard_components[name] for name in activation_names.tolist()
    ]
    arrays["activation__component_name"] = activation_names
    arrays["activation__active"] = np.asarray(
        [np.any(values > 0.0) for values in activation_arrays],
        dtype=np.bool_,
    )
    arrays["activation__nonzero_count"] = np.asarray(
        [np.count_nonzero(values) for values in activation_arrays],
        dtype=np.int64,
    )
    arrays["activation__maximum"] = np.asarray(
        [np.max(values) for values in activation_arrays], dtype=np.float64
    )
    return arrays


def _capture_sampling_boundaries(
    module: Any,
    fixture: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    temperatures = np.asarray(
        fixture["atmosphere_grid_boundary_temperature_k"], np.float64
    )
    grids = [
        module.build_opacity_sampling_grid(float(temperature))
        for temperature in temperatures
    ]
    active_reference_counts = np.asarray(
        [
            module.active_continuum_reference_frequencies(float(temperature))[0].size
            for temperature in temperatures
        ],
        dtype=np.int64,
    )
    expected_counts = np.asarray(
        [226, 240, 240, 263, 263, 299, 299, 338], dtype=np.int64
    )
    if not np.array_equal(active_reference_counts, expected_counts):
        raise RuntimeError(
            "atmosphere active line-reference counts changed: "
            f"{active_reference_counts.tolist()}"
        )
    reference_wavelength, packed_index = (
        module.build_continuum_reference_wavelength_grid()
    )
    return {
        "effective_temperature_k": temperatures,
        "wavelength_nm": np.stack([grid[0] for grid in grids]),
        "frequency_weight_hz": np.stack([grid[1] for grid in grids]),
        "active_line_reference_count": active_reference_counts,
        "expected_active_line_reference_count": expected_counts,
        "count_263_present": np.asarray(
            np.count_nonzero(active_reference_counts == 263) == 2,
            dtype=np.bool_,
        ),
        "count_299_present": np.asarray(
            np.count_nonzero(active_reference_counts == 299) == 2,
            dtype=np.bool_,
        ),
        "reference_wavelength_nm": reference_wavelength,
        "reference_packed_wavelength_index": packed_index,
    }


def _capture_h2_ch_oh_cia_boundaries(
    module: Any,
    fixture: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    continuum_tables = module.load_continuum_opacity_tables()
    molecular_tables = module.load_molecular_equilibrium_tables()
    h2_temperature = np.asarray(fixture["h2_policy_temperature_k"], dtype=np.float64)
    h2_hydrogen_normalized = np.full(h2_temperature.shape, 1.0e15)
    h2_departure = np.ones_like(h2_temperature)
    h2_equilibrium = module._h2_equilibrium_constant(
        h2_temperature, tables=molecular_tables
    )
    h2_population = module.compute_molecular_hydrogen_population(
        temperature_k=h2_temperature,
        hydrogen_neutral_partition_normalized_population=(h2_hydrogen_normalized),
        hydrogen_departure_coefficient=h2_departure,
        tables=molecular_tables,
    )

    edge_frequency = threshold_frequency_probe(module, fixture)
    molecular_family_mask = np.isin(
        edge_frequency["family_index"], np.asarray([1, 2, 3, 4, 5])
    )
    molecular_frequency = edge_frequency["frequency_hz"][molecular_family_mask]
    molecular_family_index = edge_frequency["family_index"][molecular_family_mask]
    photo_temperature = h2_temperature[np.isin(h2_temperature, [8999.0, 9000.0])]
    ch_cross_section = module._ch_molecular_cross_section_grid(
        molecular_frequency,
        photo_temperature,
        tables=continuum_tables,
    )
    oh_cross_section = module._oh_molecular_cross_section_grid(
        molecular_frequency,
        photo_temperature,
        tables=continuum_tables,
    )

    cia_temperature = np.asarray([3000.0, 3500.0, 4000.0], dtype=np.float64)
    cia_wavenumber = np.asarray(
        [
            5000.0,
            np.nextafter(20000.0, -np.inf),
            20000.0,
            np.nextafter(20000.0, np.inf),
        ],
        dtype=np.float64,
    )
    cia_frequency = cia_wavenumber * float(module.LIGHT_SPEED_CM_PER_S_EXACT)
    _, _, cia_stimulated = module._planck_frequency_exact(
        temperature_k=cia_temperature,
        frequency_hz=cia_frequency,
    )
    cia_hydrogen_normalized = np.full(cia_temperature.shape, 1.0e15)
    cia_departure = np.ones_like(cia_temperature)
    cia_helium = np.full(cia_temperature.shape, 2.0e14)
    cia_mass_density = np.full(cia_temperature.shape, 1.0e-8)
    cia_absorption = module._h2_collision_absorption_grid(
        cia_frequency,
        temperature_k=cia_temperature,
        hydrogen_neutral_partition_normalized_population=(cia_hydrogen_normalized),
        hydrogen_departure_coefficient=cia_departure,
        helium_neutral_population=cia_helium,
        mass_density=cia_mass_density,
        stimulated_emission=cia_stimulated,
        continuum_tables=continuum_tables,
        molecular_tables=molecular_tables,
    )
    temperature_index = np.clip(
        np.asarray(cia_temperature / 1000.0, dtype=np.int64), 1, 6
    )
    temperature_fraction = np.clip(
        (cia_temperature - 1000.0 * temperature_index) / 1000.0,
        0.0,
        1.0,
    )

    active_wavenumber = cia_wavenumber[:3]
    wave_index = np.minimum(np.asarray(active_wavenumber / 250.0, dtype=np.int64), 79)
    wave_fraction = (active_wavenumber - 250.0 * wave_index.astype(np.float64)) / 250.0
    idx0 = np.minimum(wave_index, 80)
    idx1 = np.minimum(wave_index + 1, 80)
    h2h2_by_temperature = (
        continuum_tables.hydrogen_molecule_h2_collision_table[idx0, :]
        * (1.0 - wave_fraction[:, None])
        + continuum_tables.hydrogen_molecule_h2_collision_table[idx1, :]
        * wave_fraction[:, None]
    )
    h2he_by_temperature = (
        continuum_tables.hydrogen_molecule_he_collision_table[idx0, :]
        * (1.0 - wave_fraction[:, None])
        + continuum_tables.hydrogen_molecule_he_collision_table[idx1, :]
        * wave_fraction[:, None]
    )
    h2h2_selected_log = np.empty(
        (cia_temperature.size, active_wavenumber.size), dtype=np.float64
    )
    h2he_selected_log = np.empty_like(h2h2_selected_log)
    for layer, index in enumerate(temperature_index):
        fraction = temperature_fraction[layer]
        h2h2_selected_log[layer] = h2h2_by_temperature[
            :, index - 1
        ] * fraction + h2h2_by_temperature[:, index] * (1.0 - fraction)
        h2he_selected_log[layer] = h2he_by_temperature[
            :, index - 1
        ] * fraction + h2he_by_temperature[:, index] * (1.0 - fraction)
    return {
        "h2_temperature_k": h2_temperature,
        "h2_temperature_uint64_bits": h2_temperature.view(np.uint64),
        "h2_hydrogen_partition_normalized_population": h2_hydrogen_normalized,
        "h2_departure_coefficient": h2_departure,
        "h2_equilibrium_constant": h2_equilibrium,
        "h2_population": h2_population,
        "h2_table_safe_temperature_k": np.minimum(
            np.where(h2_temperature > 100.0, h2_temperature, 100.0),
            19900.0,
        ),
        "ch_oh_frequency_hz": molecular_frequency,
        "ch_oh_frequency_family_index": molecular_family_index,
        "ch_oh_temperature_k": photo_temperature,
        "ch_cross_section_times_partition": ch_cross_section,
        "oh_cross_section_times_partition": oh_cross_section,
        "cia_temperature_k": cia_temperature,
        "cia_wavenumber_cm_inverse": cia_wavenumber,
        "cia_frequency_hz": cia_frequency,
        "cia_hydrogen_partition_normalized_population": (cia_hydrogen_normalized),
        "cia_departure_coefficient": cia_departure,
        "cia_helium_neutral_population": cia_helium,
        "cia_mass_density": cia_mass_density,
        "cia_stimulated_emission": cia_stimulated,
        "cia_absorption": cia_absorption,
        "cia_temperature_table_index": temperature_index,
        "cia_lower_column_index": temperature_index - 1,
        "cia_upper_column_index": temperature_index,
        "cia_temperature_fraction": temperature_fraction,
        "cia_lower_column_weight": temperature_fraction,
        "cia_upper_column_weight": 1.0 - temperature_fraction,
        "cia_h2h2_interpolated_log10_coefficient": h2h2_selected_log,
        "cia_h2he_interpolated_log10_coefficient": h2he_selected_log,
        "cia_active_frequency_mask": cia_wavenumber <= 20000.0,
    }


def _capture_ifop_coupling(
    module: Any,
    fixture: Mapping[str, np.ndarray],
    frequencies_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    state = _atmosphere_state(module, fixture, "cool_molecule_rich")
    flags = np.zeros(20, dtype=np.int64)
    flags[3] = 1
    ifop4 = module.compute_continuum_scattering_columns(
        state, frequencies_hz, opacity_flags=flags
    )
    flags[:] = 0
    flags[12] = 1
    ifop13 = module.compute_continuum_scattering_columns(
        state, frequencies_hz, opacity_flags=flags
    )
    flags[3] = 1
    both = module.compute_continuum_scattering_columns(
        state, frequencies_hz, opacity_flags=flags
    )
    return {
        "frequency_hz": frequencies_hz,
        "ifop4_only_scattering": ifop4,
        "ifop13_only_scattering": ifop13,
        "ifop4_and_13_scattering": both,
        "h2_rayleigh_increment": both - ifop4,
        "ifop13_only_is_zero": np.asarray(
            np.count_nonzero(ifop13) == 0, dtype=np.bool_
        ),
    }


def _capture_ifop19_surrogate(
    module: Any,
    fixture: Mapping[str, np.ndarray],
    frequencies_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    """Execute the separately owned Rosseland-style IFOP 19 surrogate."""

    state = _atmosphere_state(module, fixture, "solar_dwarf")
    rosseland_input = np.geomspace(1.0e-3, 1.0, state.layers)
    table = module.create_rosseland_opacity_table(state.layers)
    module.ingest_rosseland_opacity_table(
        table,
        temperature_k=state.temperature,
        gas_pressure=state.gas_pressure,
        rosseland_opacity=rosseland_input,
    )
    flags = np.zeros(20, dtype=np.int64)
    flags[18] = 1
    absorption, scattering, source = module.compute_continuum_opacity_columns(
        state,
        frequencies_hz,
        opacity_flags=flags,
        rosseland_table=table,
    )
    if np.count_nonzero(absorption) == 0:
        raise RuntimeError("IFOP 19 surrogate absorption is inactive")
    if np.count_nonzero(scattering) != 0:
        raise RuntimeError("IFOP 19 surrogate produced scattering")
    if not np.all(source == source[:, :1]):
        raise RuntimeError("IFOP 19 surrogate source gained frequency structure")
    return {
        "frequency_hz": frequencies_hz,
        "opacity_flags": flags,
        "input_rosseland_opacity": rosseland_input,
        "table_entry_count": np.asarray(table.entry_count, dtype=np.int64),
        "absorption": absorption,
        "scattering": scattering,
        "bolometric_like_source": source,
    }


def _assert_elementwise_residual(
    *,
    name: str,
    residual: np.ndarray,
    reference: np.ndarray,
    absolute_floor: float,
) -> None:
    """Fail unless every residual respects an explicit local error scale."""

    difference = np.abs(np.asarray(residual, dtype=np.float64))
    scale = np.maximum(
        np.abs(np.asarray(reference, dtype=np.float64)),
        float(absolute_floor),
    )
    allowed = float(absolute_floor) + RESIDUAL_RELATIVE_TOLERANCE * scale
    failed = difference > allowed
    if np.any(failed):
        flat_index = int(np.flatnonzero(failed)[0])
        raise RuntimeError(
            f"elementwise residual failed for {name} at flat index "
            f"{flat_index}: residual={difference.flat[flat_index]}, "
            f"allowed={allowed.flat[flat_index]}"
        )


def _validate_main_products(results: Mapping[str, np.ndarray]) -> None:
    for regime in REGIME_NAMES:
        for lane, names in (
            (
                "atmosphere",
                (
                    "absorption",
                    "scattering",
                    "source",
                    "product_absorption",
                    "product_scattering",
                    "product_source",
                ),
            ),
            ("synthesis", ("standard__absorption", "standard__scattering")),
            (
                "synthesis",
                (
                    "diagnostic__absorption",
                    "diagnostic__scattering",
                    "diagnostic__source",
                ),
            ),
            (
                "synthesis",
                (
                    "extension__absorption",
                    "extension__scattering",
                    "extension__source",
                ),
            ),
        ):
            for name in names:
                key = f"{regime}__{lane}__{name}"
                values = np.asarray(results[key])
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(f"nonfinite main product: {key}")
                if "source" not in name and np.any(values < 0.0):
                    raise RuntimeError(f"negative main opacity: {key}")
    if not bool(results["ifop__ifop13_only_is_zero"]):
        raise RuntimeError("IFOP 13 unexpectedly works without IFOP 4")
    residual_specs = (
        (
            "atmosphere__component__absorption_residual",
            "atmosphere__absorption",
            "opacity",
        ),
        (
            "atmosphere__component__scattering_residual",
            "atmosphere__scattering",
            "opacity",
        ),
        (
            "atmosphere__component__source_residual",
            "atmosphere__source",
            "source",
        ),
        (
            "synthesis__standard__component__absorption_residual",
            "synthesis__standard__sample_absorption",
            "opacity",
        ),
        (
            "synthesis__standard__component__scattering_residual",
            "synthesis__standard__sample_scattering",
            "opacity",
        ),
        (
            "synthesis__diagnostic__component__absorption_residual",
            "synthesis__diagnostic__absorption",
            "opacity",
        ),
        (
            "synthesis__diagnostic__component__scattering_residual",
            "synthesis__diagnostic__scattering",
            "opacity",
        ),
        (
            "synthesis__standard__interpolation_absorption_residual",
            "synthesis__standard__absorption",
            "interpolation",
        ),
        (
            "synthesis__standard__interpolation_scattering_residual",
            "synthesis__standard__scattering",
            "interpolation",
        ),
        (
            "synthesis__standard__isolated_minor__absorption_residual",
            "synthesis__standard__component__minor_absorption",
            "opacity",
        ),
        (
            "synthesis__standard__isolated_minor__scattering_residual",
            "synthesis__standard__component__minor_scattering",
            "opacity",
        ),
    )
    for regime in REGIME_NAMES:
        product_shape = results[f"{regime}__atmosphere__product_absorption"].shape
        if product_shape != (6, 30000):
            raise RuntimeError(
                f"atmosphere product shape changed for {regime}: {product_shape}"
            )
        atmosphere_activation = np.asarray(
            results[f"{regime}__atmosphere__activation__active"],
            dtype=np.bool_,
        )
        if not np.all(atmosphere_activation):
            names = np.asarray(
                results[f"{regime}__atmosphere__activation__component_name"]
            )
            raise RuntimeError(
                f"atmosphere regime activation changed for {regime}: "
                f"inactive={names[~atmosphere_activation].tolist()}"
            )
        synthesis_activation = np.asarray(
            results[f"{regime}__synthesis__activation__active"],
            dtype=np.bool_,
        )
        if not np.all(synthesis_activation):
            names = np.asarray(
                results[f"{regime}__synthesis__activation__component_name"]
            )
            raise RuntimeError(
                f"synthesis regime activation changed for {regime}: "
                f"inactive={names[~synthesis_activation].tolist()}"
            )
        product_absorption = results[f"{regime}__atmosphere__product_absorption"]
        product_scattering = results[f"{regime}__atmosphere__product_scattering"]
        if not np.all(np.any(product_absorption > 0.0, axis=1)):
            raise RuntimeError(f"atmosphere absorption inactive by layer for {regime}")
        if not np.all(np.any(product_scattering > 0.0, axis=1)):
            raise RuntimeError(f"atmosphere scattering inactive by layer for {regime}")
        for residual_name, scale_name, floor_name in residual_specs:
            residual = np.asarray(results[f"{regime}__{residual_name}"])
            _assert_elementwise_residual(
                name=f"{regime}/{residual_name}",
                residual=residual,
                reference=results[f"{regime}__{scale_name}"],
                absolute_floor=RESIDUAL_ABSOLUTE_FLOORS[floor_name],
            )
        basis_sum = results[f"{regime}__synthesis__route__basis_sum"]
        if not np.allclose(basis_sum, 1.0, rtol=0.0, atol=2.0e-14):
            raise RuntimeError(f"synthesis interpolation basis changed for {regime}")


FINGERPRINT_FIELDS = frozenset(
    {
        "meta__physical_payload_fingerprint",
        "meta__full_capture_fingerprint",
    }
)


def _capture_schema_digest(results: Mapping[str, np.ndarray]) -> str:
    """Hash the exact key, dtype, and shape schema without array values."""

    digest = hashlib.sha256()
    for name in sorted(results):
        array = np.asarray(results[name])
        if array.dtype.hasobject:
            raise RuntimeError(f"capture schema forbids object dtype: {name}")
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _require_array_contract(
    results: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> None:
    array = np.asarray(results[name])
    expected_dtype = np.dtype(dtype)
    if array.shape != shape or array.dtype != expected_dtype:
        raise RuntimeError(
            f"capture contract changed for {name}: "
            f"shape={array.shape}, dtype={array.dtype}; "
            f"expected shape={shape}, dtype={expected_dtype}"
        )


def _validate_explicit_schema_contracts(
    results: Mapping[str, np.ndarray],
) -> None:
    """Validate principal contracts before the frozen whole-schema digest."""

    _require_array_contract(
        results,
        "sampling_boundary__wavelength_nm",
        (8, 30000),
        np.float64,
    )
    _require_array_contract(
        results,
        "sampling_boundary__frequency_weight_hz",
        (8, 30000),
        np.float64,
    )
    for regime in REGIME_NAMES:
        for name in ("absorption", "scattering", "source"):
            _require_array_contract(
                results,
                f"{regime}__atmosphere__{name}",
                (6, 27),
                np.float64,
            )
            _require_array_contract(
                results,
                f"{regime}__atmosphere__product_{name}",
                (6, 30000),
                np.float64,
            )
        _require_array_contract(
            results,
            f"{regime}__atmosphere__line_reference_threshold",
            (6, 344),
            np.float32,
        )
        for name in ("absorption", "scattering"):
            _require_array_contract(
                results,
                f"{regime}__synthesis__standard__{name}",
                (6, 36),
                np.float64,
            )
        for lane, width in (("diagnostic", 27), ("extension", 12)):
            for name in ("absorption", "scattering", "source"):
                _require_array_contract(
                    results,
                    f"{regime}__synthesis__{lane}__{name}",
                    (6, width),
                    np.float64,
                )
    for name, value in results.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise RuntimeError(f"capture contains object dtype: {name}")


def _fingerprint(
    results: Mapping[str, np.ndarray],
    *,
    physical_payload_only: bool,
) -> str:
    """Hash either physical arrays or the complete reproducibility capture."""

    digest = hashlib.sha256()
    for name in sorted(results):
        if name in FINGERPRINT_FIELDS:
            continue
        if physical_payload_only and name.startswith(("meta__", "identity__")):
            continue
        array = np.asarray(results[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def build_oracle_results(
    *,
    fixture_path: Path = FIXTURE_PATH,
) -> dict[str, np.ndarray]:
    """Execute every Chapter 5 capture lane and return arrays in memory."""

    identities = verify_identity()
    fixture = load_fixture(fixture_path)
    environment = require_environment()
    modules = load_pinned_modules()
    frequency_probe = threshold_frequency_probe(modules.atmosphere, fixture)
    frequency = frequency_probe["frequency_hz"]

    results: dict[str, np.ndarray] = {
        "meta__capture_scope_complete": np.asarray(False, dtype=np.bool_),
        "meta__capture_scope": np.asarray(
            "pending exhaustive validation", dtype="<U128"
        ),
        "meta__golden_publication_performed": np.asarray(False, dtype=np.bool_),
        "meta__fixture_path": np.asarray(str(FIXTURE_PATH.resolve())),
        "meta__fixture_sha256": np.asarray(FIXTURE_SHA256),
        "meta__fixture_key_count": np.asarray(len(fixture), dtype=np.int64),
        "meta__platform": np.asarray(platform.platform()),
        "meta__python_version": np.asarray(platform.python_version()),
        "meta__numpy_version": np.asarray(np.__version__),
        "meta__torch_version": np.asarray(modules.torch.__version__),
        "meta__numba_version": np.asarray(modules.numba.__version__),
        "meta__torch_cpu_thread_count": np.asarray(
            modules.torch.get_num_threads(), dtype=np.int64
        ),
        "meta__numba_thread_count": np.asarray(
            modules.numba.get_num_threads(), dtype=np.int64
        ),
        "meta__regime_names": np.asarray(REGIME_NAMES),
        "meta__pipeline_continuum_fields": np.asarray(PIPELINE_CONTINUUM_FIELDS),
        "meta__loaded_pinned_python_source_count": np.asarray(
            len(modules.loaded_python_manifest), dtype=np.int64
        ),
        "meta__runner_continuum_binding_verified": np.asarray(True, dtype=np.bool_),
        "meta__pipeline_continuum_binding_verified": np.asarray(True, dtype=np.bool_),
        "meta__atmosphere_product_frequency_count": np.asarray(30000, dtype=np.int64),
        "meta__sampled_extension_wavelength_nm": (SAMPLED_EXTENSION_WAVELENGTH_NM),
        "meta__worker_sha256": np.asarray(sha256(WORKER_PATH)),
        "meta__capture_contract_sha256": np.asarray(sha256(CAPTURE_CONTRACT_PATH)),
        "meta__frozen_capture_schema_version": np.asarray(
            FROZEN_CAPTURE_SCHEMA_VERSION, dtype=np.int64
        ),
        "fixture_payload__content_digest": np.asarray(array_mapping_digest(fixture)),
        "fixture_payload__field_count": np.asarray(len(fixture), dtype=np.int64),
        "fixture_payload__field_names": np.asarray(sorted(fixture)),
    }
    for name, value in identities.items():
        results[f"identity__{name}"] = np.asarray(value)
    _prefix(results, "meta", environment)
    _prefix(results, "frequency_probe", frequency_probe)
    _prefix(
        results,
        "sampling_boundary",
        _capture_sampling_boundaries(modules.atmosphere, fixture),
    )
    _prefix(
        results,
        "molecular_boundary",
        _capture_h2_ch_oh_cia_boundaries(modules.atmosphere, fixture),
    )
    _prefix(
        results,
        "ifop",
        _capture_ifop_coupling(modules.atmosphere, fixture, frequency),
    )
    _prefix(
        results,
        "ifop19",
        _capture_ifop19_surrogate(modules.atmosphere, fixture, frequency),
    )
    _prefix(
        results,
        "molecular_entry",
        _capture_molecular_entry_counterfactuals(
            modules.atmosphere, fixture, frequency
        ),
    )

    for regime in REGIME_NAMES:
        _prefix(
            results,
            f"{regime}__atmosphere",
            _capture_atmosphere_regime(modules.atmosphere, fixture, regime, frequency),
        )
        _prefix(
            results,
            f"{regime}__synthesis",
            _capture_synthesis_regime(
                modules.synthesis,
                modules.synthesis_tables,
                fixture,
                regime,
                frequency,
            ),
        )
    _validate_main_products(results)
    _validate_explicit_schema_contracts(results)
    post_lane_manifest = _loaded_pinned_python_manifest()
    if post_lane_manifest != FROZEN_PINNED_PYTHON_MANIFEST:
        missing = sorted(set(FROZEN_PINNED_PYTHON_MANIFEST) - set(post_lane_manifest))
        extra = sorted(set(post_lane_manifest) - set(FROZEN_PINNED_PYTHON_MANIFEST))
        changed = sorted(
            name
            for name in set(post_lane_manifest) & set(FROZEN_PINNED_PYTHON_MANIFEST)
            if post_lane_manifest[name] != FROZEN_PINNED_PYTHON_MANIFEST[name]
        )
        raise OracleIdentityError(
            "post-lane loaded source manifest changed; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    results["meta__post_lane_loaded_manifest_verified"] = np.asarray(
        True, dtype=np.bool_
    )
    results["meta__post_lane_loaded_source_count"] = np.asarray(
        len(post_lane_manifest), dtype=np.int64
    )
    post_lane_manifest_text = json.dumps(
        post_lane_manifest, sort_keys=True, separators=(",", ":")
    )
    results["meta__post_lane_loaded_manifest_digest"] = np.asarray(
        hashlib.sha256(post_lane_manifest_text.encode()).hexdigest()
    )

    results["meta__physical_payload_fingerprint"] = np.asarray("0" * 64)
    results["meta__full_capture_fingerprint"] = np.asarray("0" * 64)
    results["meta__capture_schema_digest"] = np.asarray("0" * 64)
    schema_digest = _capture_schema_digest(results)
    results["meta__capture_schema_digest"] = np.asarray(schema_digest)
    if ACCEPTED_CAPTURE_KEY_COUNT is None or ACCEPTED_CAPTURE_SCHEMA_DIGEST is None:
        results["meta__capture_scope"] = np.asarray(
            "candidate complete; accepted schema constants pending",
            dtype="<U128",
        )
    else:
        if len(results) != ACCEPTED_CAPTURE_KEY_COUNT:
            raise RuntimeError(
                f"capture key count is {len(results)}, expected "
                f"{ACCEPTED_CAPTURE_KEY_COUNT}"
            )
        if schema_digest != ACCEPTED_CAPTURE_SCHEMA_DIGEST:
            raise RuntimeError(
                f"capture schema digest is {schema_digest}, expected "
                f"{ACCEPTED_CAPTURE_SCHEMA_DIGEST}"
            )
        results["meta__capture_scope_complete"] = np.asarray(True, dtype=np.bool_)
        results["meta__capture_scope"] = np.asarray(
            "exhaustive Chapter 5 atmosphere and synthesis continuum "
            "product, ownership, counterfactual, and boundary capture",
            dtype="<U128",
        )

    results["meta__physical_payload_fingerprint"] = np.asarray(
        _fingerprint(results, physical_payload_only=True)
    )
    results["meta__full_capture_fingerprint"] = np.asarray(
        _fingerprint(results, physical_payload_only=False)
    )
    if str(results["meta__physical_payload_fingerprint"]) != _fingerprint(
        results, physical_payload_only=True
    ):
        raise RuntimeError("physical-payload fingerprint is inconsistent")
    if str(results["meta__full_capture_fingerprint"]) != _fingerprint(
        results, physical_payload_only=False
    ):
        raise RuntimeError("full-capture fingerprint is inconsistent")
    return results


def summarize(results: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Return a compact JSON-safe report without serializing physical arrays."""

    maximum_atmosphere_source_residual = max(
        float(
            np.max(np.abs(results[f"{regime}__atmosphere__component__source_residual"]))
        )
        for regime in REGIME_NAMES
    )
    maximum_interpolation_residual = max(
        float(
            np.max(
                np.abs(
                    results[
                        f"{regime}__synthesis__standard__interpolation_{kind}_residual"
                    ]
                )
            )
        )
        for regime in REGIME_NAMES
        for kind in ("absorption", "scattering")
    )
    return {
        "capture_scope_complete": bool(results["meta__capture_scope_complete"]),
        "fixture_sha256": str(results["meta__fixture_sha256"]),
        "key_count": len(results),
        "capture_schema_digest": str(results["meta__capture_schema_digest"]),
        "physical_payload_fingerprint": str(
            results["meta__physical_payload_fingerprint"]
        ),
        "full_capture_fingerprint": str(results["meta__full_capture_fingerprint"]),
        "loaded_pinned_python_source_count": int(
            results["meta__loaded_pinned_python_source_count"]
        ),
        "fixture_payload_content_digest": str(
            results["fixture_payload__content_digest"]
        ),
        "worker_sha256": str(results["meta__worker_sha256"]),
        "capture_contract_sha256": str(results["meta__capture_contract_sha256"]),
        "regimes": list(REGIME_NAMES),
        "atmosphere_product_shapes": {
            regime: list(results[f"{regime}__atmosphere__product_absorption"].shape)
            for regime in REGIME_NAMES
        },
        "synthesis_standard_shapes": {
            regime: list(results[f"{regime}__synthesis__standard__absorption"].shape)
            for regime in REGIME_NAMES
        },
        "sampled_diagnostic_shapes": {
            regime: list(results[f"{regime}__synthesis__diagnostic__absorption"].shape)
            for regime in REGIME_NAMES
        },
        "sampled_extension_shapes": {
            regime: list(results[f"{regime}__synthesis__extension__absorption"].shape)
            for regime in REGIME_NAMES
        },
        "sampling_boundary_shape": list(
            results["sampling_boundary__wavelength_nm"].shape
        ),
        "line_reference_shape": list(
            results["solar_dwarf__atmosphere__line_reference_threshold"].shape
        ),
        "maximum_atmosphere_source_residual": (maximum_atmosphere_source_residual),
        "maximum_synthesis_interpolation_residual": (maximum_interpolation_residual),
        "ifop13_only_is_zero": bool(results["ifop__ifop13_only_is_zero"]),
        "cia_lower_weights": results[
            "molecular_boundary__cia_lower_column_weight"
        ].tolist(),
        "cia_upper_weights": results[
            "molecular_boundary__cia_upper_column_weight"
        ].tolist(),
        "golden_publication_performed": bool(
            results["meta__golden_publication_performed"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="self-check the unpublished Chapter 5 continuum oracle"
    )
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="verify the pin and fixture without executing continuum kernels",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.identity_only:
        identities = verify_identity()
        fixture = load_fixture(arguments.fixture)
        print(
            json.dumps(
                {
                    "commit": identities["payne_zero_commit"],
                    "fixture_sha256": FIXTURE_SHA256,
                    "fixture_key_count": len(fixture),
                },
                sort_keys=True,
            )
        )
        return
    results = build_oracle_results(fixture_path=arguments.fixture)
    print(json.dumps(summarize(results), sort_keys=True))


if __name__ == "__main__":
    main()
