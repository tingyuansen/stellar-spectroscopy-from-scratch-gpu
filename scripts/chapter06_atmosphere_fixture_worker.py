#!/usr/bin/env python3
"""Build the Chapter 6 atmosphere one-line fixture candidate in memory.

This scientific worker implements safe-sequence steps 4--6 only.  It verifies
all fixed authority and process inputs before importing the pinned atmosphere
package, executes the structured fixed-electron-density route, and returns:

1. exactly nineteen detached object-free fixture arrays; and
2. a separate detached ephemeral evidence mapping.

It has no serialization, output-path, manifest, golden, or publication
interface.  The CLI prints only a JSON summary of the in-memory capture.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterator, Mapping


THREAD_ENVIRONMENT = {
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
CAPTURE_ENVIRONMENT = {
    **THREAD_ENVIRONMENT,
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "TZ": "UTC",
}


class FixtureEnvironmentError(RuntimeError):
    """Raised when deterministic fresh-process controls are absent."""


def _require_bootstrap_environment() -> None:
    """Reject inherited process-control drift before NumPy can initialize."""

    wrong = {
        name: os.environ.get(name)
        for name, expected in CAPTURE_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if wrong:
        raise FixtureEnvironmentError(
            "pre-NumPy fixture controls are missing or changed: "
            + json.dumps(wrong, sort_keys=True)
        )


if __name__ == "__main__":
    _require_bootstrap_environment()

import numpy as np  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STAGED_SOURCE_ROOT = REPOSITORY_ROOT / "src"
PINNED_ROOT = Path("/Users/ysting/payne-zero")
PINNED_DATA_ROOT = PINNED_ROOT / "source_data_files"
PAPER_ROOT = Path("/Users/ysting/Source_Files_Not_For_Review")
WORKER_PATH = Path(__file__).resolve()
PLAN_PATH = REPOSITORY_ROOT / "design/chapter06_atmosphere_fixture_oracle_plan.md"
SOURCE_CONTRACT_PATH = REPOSITORY_ROOT / "design/chapter06_exact_source_contract.md"
CONVERTER_PATH = REPOSITORY_ROOT / "scripts/chapter06_atmosphere_line_converter.py"
CONVERTER_TEST_PATH = (
    REPOSITORY_ROOT / "tests/test_chapter06_atmosphere_line_converter.py"
)
CONVERTER_AUDIT_PATH = (
    REPOSITORY_ROOT / "design/chapter06_atmosphere_converter_independent_audit.md"
)

PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
PLAN_SHA256 = "cccc1c47e79c1d41fcae04a0a681cfaf7afe6552457f9d02a60a3c43e36bfb97"
SOURCE_CONTRACT_SHA256 = (
    "ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1"
)
CONVERTER_SHA256 = "4e59e730fc07b2cf447fa227059cb2ccff30ef30f2f01eb0ec78977588d83bbb"
CONVERTER_TEST_SHA256 = (
    "254d796b7ab761ca806c372d0bcdd935067ff1a89b2acfebcfa3007fe3f549dc"
)
CONVERTER_AUDIT_SHA256 = (
    "60e273fd8b8062200718a295b6a73d3408065effd592ff76f674563689377e75"
)
STAGED_CONVERTER_DEPENDENCIES = {
    "payne_zero_atmosphere/__init__.py": (
        "dbb7734cab4f3e98b9b88d1f1b5ec27afd02fd7fd003ba7931b43d3750049d61"
    ),
    "payne_zero_atmosphere/line_catalog.py": (
        "2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92"
    ),
    "payne_zero_atmosphere/population_layout.py": (
        "36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0"
    ),
}
PINNED_CONVERTER_DEPENDENCY_HASHES = {
    "payne_zero_atmosphere/__init__.py": (
        "dbb7734cab4f3e98b9b88d1f1b5ec27afd02fd7fd003ba7931b43d3750049d61"
    ),
    "payne_zero_atmosphere/line_catalog.py": (
        "2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92"
    ),
    "payne_zero_atmosphere/population_layout.py": (
        "36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0"
    ),
}
STAGED_CONVERTER_BYTE_IDENTICAL_TO_PINNED = frozenset(
    {
        "payne_zero_atmosphere/__init__.py",
        "payne_zero_atmosphere/line_catalog.py",
        "payne_zero_atmosphere/population_layout.py",
    }
)
CONVERSION_VERSION = 1
CAPTURE_SCHEMA_VERSION = 1
NON_AUTHORITATIVE_OBSERVED_ROW_INDEX = 780_108

STRUCTURED_SOURCE_PATH = PINNED_ROOT / "examples/data/sun_structured_atmosphere.npz"
MOLECULAR_CATALOG_PATH = (
    PINNED_DATA_ROOT / "source_catalogs/lines/molecular_equilibrium_atmosphere.npz"
)

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
}

DATA_IDENTITIES = {
    "examples/data/sun_structured_atmosphere.npz": (
        1_633_632,
        "d686ea7107d60bf1707607e3d6377d283fb3eb7115c170ac2aeef54fbaa6abdb",
    ),
    "source_data_files/source_catalogs/lines/molecular_equilibrium_atmosphere.npz": (
        19_040,
        "971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644",
    ),
    "source_data_files/atmosphere_tables/isotope_tables.npz": (
        169_568,
        "53c8d315fb53f1e051dc2752b028fc270d7c17a2c1042279c04ffcb750aef5c6",
    ),
    "source_data_files/atmosphere_tables/iron_group_partition_tables.npz": (
        45_573,
        "137629dea64eca46f77ea3656c18305ade912a468d7eb27029544c0106cc3296",
    ),
    "source_data_files/atmosphere_tables/ionization_potential_tables.npz": (
        8_292,
        "82a2e82f2015da02c3d2bce77ca5337aa2b9c4e23d8d6219da07895896ca8a50",
    ),
    "source_data_files/atmosphere_tables/packed_level_metadata.npz": (
        17_816,
        "de5f17b6a9eaec1d1b07e96fd02ff014279cd8eaa9f976fefde0e2a153961bc3",
    ),
    "source_data_files/atmosphere_tables/special_partition_tables.npz": (
        11_364,
        "7d737524aacda1cc2281e5b18ff49f240ca34665dbe6c96d4dd0f39db4aedd22",
    ),
    "source_data_files/atmosphere_tables/molecular_equilibrium_tables.npz": (
        1_935,
        "1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe",
    ),
    "source_data_files/atmosphere_tables/continuum_opacity_tables.npz": (
        57_456,
        "6fd4c556418870c28d3fcc9a050252af58ac4cc433cae979477355c8c7d593e3",
    ),
    "source_data_files/atmosphere_tables/karzas_latter_tables.npz": (
        16_826,
        "23805dc17c47af45b8ae63b2e278e1fb6c584a01c87d1eb3c31306e4555e6d15",
    ),
    "source_data_files/atmosphere_tables/line_opacity_tables.npz": (
        1_359,
        "89f486122cb8939b23dc5423145a46d88a77df8daf57a1def35055b7b8205f16",
    ),
}

EXPECTED_DYNAMIC_READ_SET = tuple(sorted(DATA_IDENTITIES))

FIXTURE_SCHEMA = {
    "packed_wavelength_index": ((1,), np.dtype("<i4")),
    "packed_species_slot": ((1,), np.dtype("<i2")),
    "lower_excitation_index": ((1,), np.dtype("<i2")),
    "log_strength_index": ((1,), np.dtype("<i2")),
    "radiative_damping_index": ((1,), np.dtype("<i2")),
    "stark_damping_index": ((1,), np.dtype("<i2")),
    "van_der_waals_damping_index": ((1,), np.dtype("<i2")),
    "temperature": ((80,), np.dtype("<f8")),
    "hc_over_kt": ((80,), np.dtype("<f8")),
    "electron_density": ((80,), np.dtype("<f8")),
    "actual_population_slot_indices": ((3,), np.dtype("<i2")),
    "actual_population_slot_values": ((80, 3), np.dtype("<f8")),
    "line_population_slot_zero_based": ((), np.dtype("<i2")),
    (
        "partition_normalized_population_over_mass_density_and_"
        "fractional_doppler_width_at_line_slot"
    ): ((80,), np.dtype("<f8")),
    "fractional_doppler_widths_at_line_slot": ((80,), np.dtype("<f8")),
    "opacity_wavelength_grid_nm": ((30_000,), np.dtype("<f8")),
    "wavelength_bin_edges": ((344,), np.dtype("<i8")),
    "continuum_line_selection_threshold": ((80, 344), np.dtype("<f4")),
    "effective_temperature": ((), np.dtype("<f8")),
}

EXPECTED_FIXTURE_MEMBER_HASHES = {
    "packed_wavelength_index": (
        "c423f4ac4a3825c6fad5336a1e15c0038ab17087f930916ad550ad45b4990dfc"
    ),
    "packed_species_slot": (
        "c2126161f7488b7d198ea310da9a8694786a18a2317eea5e30379ee118d34743"
    ),
    "lower_excitation_index": (
        "9b5bf0b74e2e212b57bcb2b9f2712eaab4c6169595f4e82767793f4534365648"
    ),
    "log_strength_index": (
        "3fe821d54660a0c51b42d19a42571e208791b3c5ff6bc0cac16b6553a226515a"
    ),
    "radiative_damping_index": (
        "5cdf3b75730a5b45c3f24da2c1030103143981191d2844aa7374e948b9abaeea"
    ),
    "stark_damping_index": (
        "91d82e7b89ae43b29bb673bb416697d005e093d0011c9d7af501630c6a502141"
    ),
    "van_der_waals_damping_index": (
        "c0346f09a0362e8e9d29c01a9fc7c292a7395b524a90319bb921608b9fdb1b60"
    ),
    "temperature": ("7b76410e731a6772cb6de12f923aea8a26e345778dbab67db3e8938278ce270f"),
    "hc_over_kt": ("fdbd300b88163af886fe2905a883fa319c32abcc89db1099e4b0df7965d8174e"),
    "electron_density": (
        "1e36bdf5aac263da2eac448f1b4c91e43e48a48afc133c8a8ec5e15735c85c18"
    ),
    "actual_population_slot_indices": (
        "eff6cc5c731be5128ac078be458469978b6ac7c823665abd47098485291a5af2"
    ),
    "actual_population_slot_values": (
        "ae17c6b96dc9eb824051707d83eca2bbdb22483af4c143b18d832326292e86ce"
    ),
    "line_population_slot_zero_based": (
        "8d6dd9f330421e48e73e97c0819d55a9965bac90b1f2f36057b4c906d5791f05"
    ),
    (
        "partition_normalized_population_over_mass_density_and_"
        "fractional_doppler_width_at_line_slot"
    ): "8625e8d942892d142384d51fc0625701374f79dfa471a78ff593d88479b3056f",
    "fractional_doppler_widths_at_line_slot": (
        "f8018929e269db3e6b9a572d36b528bc0577a3a93bcc25166b277824ffb2e785"
    ),
    "opacity_wavelength_grid_nm": (
        "8944f1dd701ba27f50d37a16e48ab9375e1bef1b444ed405b671ba91fde8132b"
    ),
    "wavelength_bin_edges": (
        "ae50e9c0bafdcbfc39e242fd5029bf53b9e712d43008c6cb78a4493b89272cd5"
    ),
    "continuum_line_selection_threshold": (
        "76cb7ef18554149b61b97e32f2a69bbcdba3eb8b7e6b7da8ba3c69b8775b7293"
    ),
    "effective_temperature": (
        "0ae804feb5a3896c0d30fd25ee9853a10ee08fed330969967ce16a4d07a7329b"
    ),
}

EXPECTED_FULL_ARRAY_HASHES = {
    "ion_stage_populations_by_packed_slot": (
        "53ff8546517e744862abf16c490f1a3b4bcd761d681f5fde0cfb8ad62e0812b6"
    ),
    (
        "partition_normalized_population_over_mass_density_and_fractional_doppler_width"
    ): "706c8db98522bacb0c428279a29b270ec53772174207cceadd27e701947030c0",
    "fractional_doppler_widths": (
        "23c57f545bdfd74af3dd62ef57b8f347570953ea50c4e19289a9aa9715beb868"
    ),
}
EXPECTED_PROJECTED_ARRAY_HASHES = {
    "ion_stage_populations_by_packed_slot": (
        "30fa30338c773905b57023bdd6b102c205569207346227128f4e1ce585500166"
    ),
    (
        "partition_normalized_population_over_mass_density_and_fractional_doppler_width"
    ): "60db3e4a9c56e347e764b46a7d34e3c33bc1e2256dd44041d0621c1953a63524",
    "fractional_doppler_widths": (
        "d93c56e13576a3a6dce3071026ca9479cd6a94de1518d68514ed1866c2656606"
    ),
}
EXPECTED_LINE_OUTPUT_SHA256 = (
    "43636ea863ed801c36b86c8f3e15ac863583422da87a44e41ae46a5ae43f2c58"
)

EXPECTED_CONVERTED_VALUES = {
    "packed_wavelength_index": 12_425_352,
    "packed_species_slot": 3_510,
    "lower_excitation_index": 20_909,
    "log_strength_index": 15_524,
    "radiative_damping_index": 24_854,
    "stark_damping_index": 11_934,
    "van_der_waals_damping_index": 8_874,
}


class FixtureIdentityError(RuntimeError):
    """Raised when a source, data, or accepted-converter identity changes."""


@dataclass(frozen=True)
class InMemoryFixtureCapture:
    """Nineteen candidate arrays plus separately owned ephemeral evidence."""

    fixture_arrays: dict[str, np.ndarray]
    ephemeral_evidence: dict[str, np.ndarray]


def sha256(path: Path) -> str:
    """Return one file's SHA-256 identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(values: Any) -> str:
    """Return one array's canonical C-byte SHA-256."""

    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.expanduser().resolve())
    except ValueError:
        return False
    return True


def _require_regular_nonsymlink(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise FixtureIdentityError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FixtureIdentityError(f"{label} is missing: {candidate}")
    return resolved


def _resolve_git_head(root: Path) -> str:
    """Resolve a loose or packed Git HEAD without invoking Git filters."""

    git_dir = root / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise FixtureIdentityError("pinned checkout .git must be a real directory")
    head_path = _require_regular_nonsymlink(git_dir / "HEAD", "pinned Git HEAD")
    head_text = head_path.read_text(encoding="ascii").strip()
    if not head_text.startswith("ref: "):
        return head_text
    reference = head_text[5:]
    loose_reference = git_dir / reference
    if loose_reference.is_file() and not loose_reference.is_symlink():
        return loose_reference.read_text(encoding="ascii").strip()
    packed_refs = _require_regular_nonsymlink(
        git_dir / "packed-refs",
        "pinned packed refs",
    )
    for line in packed_refs.read_text(encoding="ascii").splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        object_name, ref_name = line.split(" ", 1)
        if ref_name == reference:
            return object_name
    raise FixtureIdentityError(f"unable to resolve pinned Git reference {reference}")


def verify_staged_converter_dependencies() -> dict[str, str]:
    """Hash-gate every staged atmosphere source imported by the converter."""

    identities: dict[str, str] = {}
    pinned_root = PINNED_ROOT.resolve()
    if set(STAGED_CONVERTER_DEPENDENCIES) != set(PINNED_CONVERTER_DEPENDENCY_HASHES):
        raise FixtureIdentityError(
            "staged and pinned converter dependency manifests disagree"
        )
    for relative_path, expected_staged_hash in STAGED_CONVERTER_DEPENDENCIES.items():
        expected_pinned_hash = PINNED_CONVERTER_DEPENDENCY_HASHES[relative_path]
        if FROZEN_PINNED_PYTHON_MANIFEST.get(relative_path) != expected_pinned_hash:
            raise FixtureIdentityError(
                f"staged dependency pin disagrees with pinned manifest: {relative_path}"
            )
        staged_source = _require_regular_nonsymlink(
            STAGED_SOURCE_ROOT / relative_path,
            f"staged converter dependency {relative_path}",
        )
        pinned_source = _require_regular_nonsymlink(
            pinned_root / relative_path,
            f"pinned converter dependency {relative_path}",
        )
        staged_hash = sha256(staged_source)
        pinned_hash = sha256(pinned_source)
        if pinned_hash != expected_pinned_hash:
            raise FixtureIdentityError(
                f"pinned converter dependency changed for {relative_path}: "
                f"{pinned_hash}; expected {expected_pinned_hash}"
            )
        if staged_hash != expected_staged_hash:
            raise FixtureIdentityError(
                f"staged converter dependency changed for {relative_path}: "
                f"{staged_hash}; expected {expected_staged_hash}"
            )
        if (
            relative_path in STAGED_CONVERTER_BYTE_IDENTICAL_TO_PINNED
            and staged_source.read_bytes() != pinned_source.read_bytes()
        ):
            raise FixtureIdentityError(
                "staged converter dependency is not byte-identical to its "
                f"pinned source: {relative_path}"
            )
        identities[f"staged_converter_source__{relative_path}__sha256"] = staged_hash
    return identities


def verify_preimport_identities() -> dict[str, str]:
    """Verify commit, source, data, design, and converter before physics import."""

    root = PINNED_ROOT.resolve()
    commit = _resolve_git_head(root)
    if commit != PINNED_COMMIT:
        raise FixtureIdentityError(
            f"pinned checkout is {commit}; expected {PINNED_COMMIT}"
        )
    identities = {"payne_zero_commit": commit}

    fixed_local_files = {
        "fixture_plan": (PLAN_PATH, PLAN_SHA256),
        "source_contract": (SOURCE_CONTRACT_PATH, SOURCE_CONTRACT_SHA256),
        "accepted_converter": (CONVERTER_PATH, CONVERTER_SHA256),
        "accepted_converter_tests": (
            CONVERTER_TEST_PATH,
            CONVERTER_TEST_SHA256,
        ),
        "accepted_converter_audit": (
            CONVERTER_AUDIT_PATH,
            CONVERTER_AUDIT_SHA256,
        ),
    }
    for name, (path, expected_hash) in fixed_local_files.items():
        actual = sha256(_require_regular_nonsymlink(path, name))
        if actual != expected_hash:
            raise FixtureIdentityError(
                f"{name} SHA-256 changed: {actual}; expected {expected_hash}"
            )
        identities[f"{name}__sha256"] = actual

    for relative_path, expected_hash in FROZEN_PINNED_PYTHON_MANIFEST.items():
        source = _require_regular_nonsymlink(
            root / relative_path,
            f"pinned source {relative_path}",
        )
        actual = sha256(source)
        if actual != expected_hash:
            raise FixtureIdentityError(
                f"pinned source changed for {relative_path}: {actual}"
            )
        identities[f"source__{relative_path}__sha256"] = actual

    identities.update(verify_staged_converter_dependencies())

    for relative_path, (expected_bytes, expected_hash) in DATA_IDENTITIES.items():
        data_path = _require_regular_nonsymlink(
            root / relative_path,
            f"pinned data {relative_path}",
        )
        if data_path.stat().st_size != expected_bytes:
            raise FixtureIdentityError(
                f"pinned data byte count changed for {relative_path}"
            )
        actual = sha256(data_path)
        if actual != expected_hash:
            raise FixtureIdentityError(
                f"pinned data changed for {relative_path}: {actual}"
            )
        identities[f"data__{relative_path}__sha256"] = actual
    return identities


def require_environment() -> dict[str, np.ndarray]:
    """Require deterministic controls and an already-empty external cache."""

    wrong = {
        name: os.environ.get(name)
        for name, expected in CAPTURE_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if wrong:
        raise FixtureEnvironmentError(
            "fresh one-thread fixture controls are missing: "
            + json.dumps(wrong, sort_keys=True)
        )
    cache_text = os.environ.get("NUMBA_CACHE_DIR", "")
    if not cache_text:
        raise FixtureEnvironmentError("NUMBA_CACHE_DIR must name an empty directory")
    cache = Path(cache_text).expanduser()
    if cache.is_symlink() or not cache.is_dir():
        raise FixtureEnvironmentError(
            "NUMBA_CACHE_DIR must be an existing nonsymlink directory"
        )
    resolved = cache.resolve()
    for forbidden in (PINNED_ROOT, REPOSITORY_ROOT, PAPER_ROOT):
        if _is_under(resolved, forbidden):
            raise FixtureEnvironmentError(
                "NUMBA_CACHE_DIR must be external to all source trees"
            )
    if any(resolved.iterdir()):
        raise FixtureEnvironmentError(
            "NUMBA_CACHE_DIR must be truly empty at process start"
        )
    return {
        **{
            f"environment__{name.lower()}": np.asarray(value)
            for name, value in sorted(CAPTURE_ENVIRONMENT.items())
        },
        "environment__cache_policy": np.asarray(
            "caller-created empty external nonsymlink directory"
        ),
        "environment__cpu_only": np.asarray(True, dtype=np.bool_),
    }


def run_accepted_converter() -> dict[str, Any]:
    """Run the accepted staged converter in an isolated no-file child process."""

    verify_staged_converter_dependencies()
    child_source = r"""
import hashlib
import json
import numpy as np
from pathlib import Path
import sys
from scripts import chapter06_atmosphere_line_converter as converter
raw = converter.load_verified_canonical_raw_row()
selected = converter.convert_raw_row_to_selected_line_catalog(raw)
words = converter.pack_selected_line_words(selected)
loaded_staged_source_sha256 = {}
for module_name, module in tuple(sys.modules.items()):
    if not module_name.startswith("payne_zero_atmosphere"):
        continue
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        continue
    source_path = Path(module_file).resolve()
    try:
        relative_path = source_path.relative_to(converter.LOCAL_SOURCE_ROOT)
    except ValueError as error:
        raise RuntimeError(
            f"converter loaded {module_name} outside the staged source root"
        ) from error
    relative_name = str(relative_path)
    if relative_name not in {
        "payne_zero_atmosphere/__init__.py",
        "payne_zero_atmosphere/line_catalog.py",
        "payne_zero_atmosphere/population_layout.py",
    }:
        continue
    loaded_staged_source_sha256[relative_name] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
fields = {}
for name in converter.SELECTED_FIELD_DTYPES:
    values = np.asarray(getattr(selected, name))
    fields[name] = {
        "value": int(values[0]),
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "sha256": hashlib.sha256(
            np.ascontiguousarray(values).tobytes(order="C")
        ).hexdigest(),
    }
print(json.dumps({
    "conversion_version": converter.CONVERSION_VERSION,
    "subset_sha256": converter.CANONICAL_SUBSET_SHA256,
    "observed_row_role": "non_authoritative_packing_corroboration",
    "observed_row_index": converter.NON_AUTHORITATIVE_OBSERVED_ROW_INDEX,
    "loaded_staged_source_sha256": loaded_staged_source_sha256,
    "fields": fields,
    "words": words.tolist(),
    "words_dtype": words.dtype.str,
    "words_shape": list(words.shape),
    "words_sha256": hashlib.sha256(
        np.ascontiguousarray(words).tobytes(order="C")
    ).hexdigest(),
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", child_source],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise FixtureIdentityError(
            "accepted converter child did not return exact JSON"
        ) from error
    if completed.stderr:
        raise FixtureIdentityError(
            "accepted converter child wrote unexpected stderr: "
            + completed.stderr.strip()
        )
    if result["conversion_version"] != CONVERSION_VERSION:
        raise FixtureIdentityError("accepted converter version changed")
    if result["observed_row_role"] != "non_authoritative_packing_corroboration":
        raise FixtureIdentityError("observed row acquired an authoritative role")
    if result["observed_row_index"] != NON_AUTHORITATIVE_OBSERVED_ROW_INDEX:
        raise FixtureIdentityError("non-authoritative observed row index changed")
    if result["loaded_staged_source_sha256"] != STAGED_CONVERTER_DEPENDENCIES:
        raise FixtureIdentityError(
            "converter child staged loaded-source boundary changed"
        )
    if result["words"] != [[12_425_352, 1_370_295_734, 1_628_847_268, 581_578_398]]:
        raise FixtureIdentityError("accepted converter native payload changed")
    if result["words_dtype"] != "<i4" or result["words_shape"] != [1, 4]:
        raise FixtureIdentityError("accepted converter payload schema changed")
    if (
        result["words_sha256"]
        != "1769c9ad8d33e847a099bd6d50df85a2f478f98d554b2fc39db8121ba93158d2"
    ):
        raise FixtureIdentityError("accepted converter payload hash changed")
    if set(result["fields"]) != set(EXPECTED_CONVERTED_VALUES):
        raise FixtureIdentityError("accepted converter field set changed")
    for name, expected_value in EXPECTED_CONVERTED_VALUES.items():
        field = result["fields"][name]
        expected_dtype = FIXTURE_SCHEMA[name][1].str
        if (
            field["value"] != expected_value
            or field["shape"] != [1]
            or field["dtype"] != expected_dtype
            or field["sha256"] != EXPECTED_FIXTURE_MEMBER_HASHES[name]
        ):
            raise FixtureIdentityError(f"accepted converted record changed for {name}")
    return result


def _loaded_pinned_python_manifest() -> dict[str, str]:
    loaded: dict[str, str] = {}
    root = PINNED_ROOT.resolve()
    for module in sys.modules.values():
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        path = Path(module_file).resolve()
        if path.suffix != ".py" or not _is_under(path, root):
            continue
        loaded[str(path.relative_to(root))] = sha256(path)
    return loaded


def _assert_loaded_manifest() -> dict[str, str]:
    loaded = _loaded_pinned_python_manifest()
    if loaded != FROZEN_PINNED_PYTHON_MANIFEST:
        expected = set(FROZEN_PINNED_PYTHON_MANIFEST)
        actual = set(loaded)
        changed = sorted(
            name
            for name in expected & actual
            if loaded[name] != FROZEN_PINNED_PYTHON_MANIFEST[name]
        )
        raise FixtureIdentityError(
            "loaded pinned Python manifest changed; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}, changed={changed}"
        )
    return loaded


def load_pinned_modules() -> SimpleNamespace:
    """Import the already-verified read-only pinned atmosphere package."""

    configured_root = os.environ.get("PAYNE_ZERO_DATA_ROOT")
    if configured_root is not None and (
        Path(configured_root).expanduser().resolve() != PINNED_DATA_ROOT.resolve()
    ):
        raise FixtureIdentityError("PAYNE_ZERO_DATA_ROOT points outside the pin")
    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(PINNED_DATA_ROOT.resolve())

    for name, module in tuple(sys.modules.items()):
        if name != "payne_zero_atmosphere" and not name.startswith(
            "payne_zero_atmosphere."
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None and not _is_under(Path(module_file), PINNED_ROOT):
            raise FixtureIdentityError(
                f"{name} was already imported outside the pin: {module_file}"
            )
    root_text = str(PINNED_ROOT.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)

    atmosphere_io = importlib.import_module("payne_zero_atmosphere.atmosphere_io")
    config = importlib.import_module("payne_zero_atmosphere.config")
    line_catalog = importlib.import_module("payne_zero_atmosphere.line_catalog")
    line_opacity = importlib.import_module("payne_zero_atmosphere.line_opacity")
    runner = importlib.import_module("payne_zero_atmosphere.runner")
    import numba

    for module in (
        atmosphere_io,
        config,
        line_catalog,
        line_opacity,
        runner,
    ):
        if not _is_under(Path(module.__file__), PINNED_ROOT):
            raise FixtureIdentityError(
                f"atmosphere module resolved outside the pin: {module.__file__}"
            )
    loaded = _assert_loaded_manifest()
    if int(numba.get_num_threads()) != 1:
        raise FixtureEnvironmentError("Numba did not honor NUMBA_NUM_THREADS=1")
    return SimpleNamespace(
        atmosphere_io=atmosphere_io,
        config=config,
        line_catalog=line_catalog,
        line_opacity=line_opacity,
        runner=runner,
        numba=numba,
        loaded_python_manifest=loaded,
    )


@contextmanager
def track_numpy_data_reads() -> Iterator[list[Path]]:
    """Track every path passed to NumPy's data loader during the physics route."""

    original_load = np.load
    paths: list[Path] = []

    def tracked_load(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            paths.append(Path(file).expanduser().resolve())
        return original_load(file, *args, **kwargs)

    np.load = tracked_load
    try:
        yield paths
    finally:
        np.load = original_load


def _detached_mapping(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return a sorted detached mapping with no object dtype."""

    result: dict[str, np.ndarray] = {}
    for name in sorted(values):
        if not isinstance(name, str) or not name:
            raise TypeError("capture names must be nonempty strings")
        array = np.asarray(values[name])
        if array.dtype.hasobject:
            raise TypeError(f"capture member {name!r} has object dtype")
        result[name] = np.array(array, copy=True, order="C")
    return result


def _load_structured_source() -> dict[str, np.ndarray]:
    required = (
        "temperature",
        "column_mass",
        "gas_pressure",
        "electron_density",
        "microturbulence",
        "elemental_abundances",
    )
    with np.load(STRUCTURED_SOURCE_PATH, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in required}
    expected = {
        "temperature": ((80,), np.dtype("<f8")),
        "column_mass": ((80,), np.dtype("<f8")),
        "gas_pressure": ((80,), np.dtype("<f8")),
        "electron_density": ((80,), np.dtype("<f8")),
        "microturbulence": ((80,), np.dtype("<f8")),
        "elemental_abundances": ((99,), np.dtype("<f8")),
    }
    for name, (shape, dtype) in expected.items():
        values = arrays[name]
        if values.shape != shape or values.dtype != dtype:
            raise RuntimeError(
                f"structured source schema changed for {name}: "
                f"{values.shape}, {values.dtype}"
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"structured source is nonfinite: {name}")
    if np.any(np.diff(arrays["column_mass"]) <= 0.0):
        raise RuntimeError("structured source depth order changed")
    if np.any(arrays["elemental_abundances"] <= 0.0):
        raise RuntimeError("structured elemental abundances must be positive")
    return arrays


def _fixed_column_abundances(
    elemental_abundances: np.ndarray,
) -> dict[int, float]:
    values = np.asarray(elemental_abundances, dtype=np.float64)
    result = {1: float(values[0]), 2: float(values[1])}
    result.update(
        {
            atomic_number: float(np.log10(values[atomic_number - 1]))
            for atomic_number in range(3, 100)
        }
    )
    return result


def _opacity_flags(modules: SimpleNamespace) -> list[int]:
    flags = list(modules.config.DEFAULT_OPACITY_FLAGS)
    if len(flags) != 20:
        raise RuntimeError("default opacity flag count changed")
    flags[14] = 0
    flags[16] = 0
    return flags


def _build_model_and_config(
    modules: SimpleNamespace,
    source: Mapping[str, np.ndarray],
    *,
    perturb_validator_only_fields: bool,
) -> tuple[Any, Any]:
    layer_count = 80
    if perturb_validator_only_fields:
        rosseland = np.linspace(2.0, 3.0, layer_count, dtype=np.float64)
        radiative = np.linspace(-11.0, 7.0, layer_count, dtype=np.float64)
        convective_flux = np.linspace(101.0, 181.0, layer_count, dtype=np.float64)
        convective_velocity = np.linspace(-3.0, 5.0, layer_count, dtype=np.float64)
    else:
        rosseland = np.ones(layer_count, dtype=np.float64)
        radiative = np.zeros(layer_count, dtype=np.float64)
        convective_flux = np.zeros(layer_count, dtype=np.float64)
        convective_velocity = np.zeros(layer_count, dtype=np.float64)

    flags = _opacity_flags(modules)
    model = modules.atmosphere_io.ModelAtmosphere(
        column_mass=np.array(source["column_mass"], copy=True),
        temperature=np.array(source["temperature"], copy=True),
        gas_pressure=np.array(source["gas_pressure"], copy=True),
        electron_density=np.array(source["electron_density"], copy=True),
        rosseland_opacity=rosseland,
        radiative_acceleration=radiative,
        microturbulence=np.array(source["microturbulence"], copy=True),
        convective_flux=convective_flux,
        convective_velocity=convective_velocity,
        metadata={
            "effective_temperature": "5778.0",
            "log_surface_gravity": "4.44",
            "pressure_iteration_enabled": "0",
            "opacity_flags": "OPACITY IFOP " + " ".join(str(value) for value in flags),
        },
        fixed_column_abundance_values=_fixed_column_abundances(
            source["elemental_abundances"]
        ),
    )
    decoded_abundances = modules.atmosphere_io.linear_elemental_abundances(model)
    if not np.array_equal(decoded_abundances, source["elemental_abundances"]):
        raise RuntimeError("fixed-column abundance round trip changed")

    inputs = modules.config.AtmosphereInput(
        initial_atmosphere=model,
        molecules_path=MOLECULAR_CATALOG_PATH,
    )
    outputs = modules.config.AtmosphereOutput()
    if any(
        getattr(outputs, name) is not None
        for name in (
            "structured_atmosphere_path",
            "diagnostics_path",
            "debug_state_path",
        )
    ):
        raise RuntimeError("fixture config unexpectedly has an output path")
    config = modules.config.AtmosphereConfig(
        inputs=inputs,
        outputs=outputs,
        iterations=1,
        enable_molecules=True,
        enable_convection=False,
    )
    return model, config


def _run_fixed_state_route(
    modules: SimpleNamespace,
    source: Mapping[str, np.ndarray],
    *,
    perturb_validator_only_fields: bool,
) -> tuple[Any, Any, Any]:
    model, config = _build_model_and_config(
        modules,
        source,
        perturb_validator_only_fields=perturb_validator_only_fields,
    )
    population = modules.runner.prepare_structured_handoff_population_state(
        config,
        temperature_iteration_index=1,
    )
    opacity = modules.runner.prepare_opacity_state(
        config,
        population_state=population,
        temperature_iteration_index=1,
    )
    setup = population.setup
    if (
        setup.atmosphere.layers != 80
        or setup.effective_temperature != 5778.0
        or setup.log_surface_gravity != 4.44
        or setup.pressure_iteration_enabled
        or not setup.molecules_enabled
        or setup.convection.enabled
        or setup.opacity_flags[14] != 0
        or setup.opacity_flags[16] != 0
    ):
        raise RuntimeError("fixed-state setup contract changed")
    if opacity.line_opacity.selected_line_count != 0 or np.any(
        opacity.line_opacity.line_mass_absorption_coefficient
    ):
        raise RuntimeError("line-disabled opacity preparation accumulated a line")
    return model, population, opacity


def _selected_catalog(
    modules: SimpleNamespace,
    converted: Mapping[str, Any],
) -> Any:
    arrays: dict[str, np.ndarray] = {}
    for name, expected_value in EXPECTED_CONVERTED_VALUES.items():
        dtype = FIXTURE_SCHEMA[name][1]
        field = converted["fields"][name]
        if field["value"] != expected_value:
            raise RuntimeError(f"converted field changed before construction: {name}")
        arrays[name] = np.asarray([field["value"]], dtype=dtype)
    return modules.line_catalog.SelectedLineCatalog(**arrays)


def _accumulate_line(
    modules: SimpleNamespace,
    selected_lines: Any,
    model: Any,
    population: Any,
    opacity: Any,
    *,
    actual_populations: np.ndarray,
    line_support: np.ndarray,
    fractional_widths: np.ndarray,
) -> Any:
    return modules.line_opacity.accumulate_selected_line_opacity(
        selected_lines=selected_lines,
        opacity_wavelength_grid_nm=opacity.opacity_wavelength_grid_nm,
        wavelength_bin_edges=opacity.wavelength_bin_edges,
        continuum_line_selection_threshold=(opacity.continuum_line_selection_threshold),
        temperature=model.temperature,
        hc_over_kt=model.hc_over_kt,
        electron_density=population.runtime_state.electron_density,
        ion_stage_populations_by_packed_slot=actual_populations,
        partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
            line_support
        ),
        fractional_doppler_widths=fractional_widths,
    )


def _fixture_mapping(
    selected_lines: Any,
    model: Any,
    population: Any,
    opacity: Any,
) -> dict[str, np.ndarray]:
    full_actual = population.runtime_state.ion_stage_populations_by_packed_slot
    full_support = population.partition_normalized_population_over_mass_density_and_fractional_doppler_width
    full_widths = population.fractional_doppler_widths
    return _detached_mapping(
        {
            "packed_wavelength_index": selected_lines.packed_wavelength_index,
            "packed_species_slot": selected_lines.packed_species_slot,
            "lower_excitation_index": selected_lines.lower_excitation_index,
            "log_strength_index": selected_lines.log_strength_index,
            "radiative_damping_index": selected_lines.radiative_damping_index,
            "stark_damping_index": selected_lines.stark_damping_index,
            "van_der_waals_damping_index": (selected_lines.van_der_waals_damping_index),
            "temperature": model.temperature,
            "hc_over_kt": model.hc_over_kt,
            "electron_density": population.runtime_state.electron_density,
            "actual_population_slot_indices": np.asarray([0, 2, 840], dtype=np.int16),
            "actual_population_slot_values": full_actual[:, [0, 2, 840]],
            "line_population_slot_zero_based": np.asarray(350, dtype=np.int16),
            (
                "partition_normalized_population_over_mass_density_and_"
                "fractional_doppler_width_at_line_slot"
            ): full_support[:, 350],
            "fractional_doppler_widths_at_line_slot": full_widths[:, 350],
            "opacity_wavelength_grid_nm": opacity.opacity_wavelength_grid_nm,
            "wavelength_bin_edges": opacity.wavelength_bin_edges,
            "continuum_line_selection_threshold": (
                opacity.continuum_line_selection_threshold
            ),
            "effective_temperature": np.asarray(5778.0, dtype=np.float64),
        }
    )


def _validate_fixture_mapping(fixture: Mapping[str, np.ndarray]) -> None:
    if set(fixture) != set(FIXTURE_SCHEMA) or len(fixture) != 19:
        raise RuntimeError("fixture mapping is not exactly the proposed 19 members")
    for name, (expected_shape, expected_dtype) in FIXTURE_SCHEMA.items():
        values = np.asarray(fixture[name])
        if values.shape != expected_shape or values.dtype != expected_dtype:
            raise RuntimeError(
                f"fixture member schema changed: {name} {values.shape} {values.dtype}"
            )
        if values.dtype.hasobject:
            raise RuntimeError(f"fixture member has object dtype: {name}")
        if array_sha256(values) != EXPECTED_FIXTURE_MEMBER_HASHES[name]:
            raise RuntimeError(f"fixture feasibility hash changed: {name}")
    if not np.array_equal(
        fixture["actual_population_slot_indices"],
        np.asarray([0, 2, 840], dtype=np.int16),
    ):
        raise RuntimeError("actual-population projection slots changed")
    if int(fixture["line_population_slot_zero_based"]) != 350:
        raise RuntimeError("Fe I line-population slot changed")
    if int(fixture["wavelength_bin_edges"][-1]) != 2**30:
        raise RuntimeError("continuum-bin final sentinel changed")
    if np.any(np.diff(fixture["opacity_wavelength_grid_nm"]) <= 0.0):
        raise RuntimeError("opacity wavelength grid is not increasing")
    if np.any(~np.isfinite(fixture["continuum_line_selection_threshold"])):
        raise RuntimeError("continuum threshold contains nonfinite values")


def mapping_schema_digest(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.asarray(values[name])
        if array.dtype.hasobject:
            raise TypeError(f"schema digest forbids object dtype: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    return digest.hexdigest()


def mapping_fingerprint(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.asarray(values[name])
        if array.dtype.hasobject:
            raise TypeError(f"fingerprint forbids object dtype: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _combined_mapping(
    fixture: Mapping[str, np.ndarray],
    evidence: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return _detached_mapping(
        {
            **{f"fixture__{name}": value for name, value in fixture.items()},
            **{f"evidence__{name}": value for name, value in evidence.items()},
        }
    )


def build_fixture_capture() -> InMemoryFixtureCapture:
    """Execute the complete in-memory candidate construction and evidence pass."""

    identities = verify_preimport_identities()
    environment = require_environment()
    converted = run_accepted_converter()

    with track_numpy_data_reads() as dynamic_reads:
        modules = load_pinned_modules()
        structured = _load_structured_source()
        model, population, opacity = _run_fixed_state_route(
            modules,
            structured,
            perturb_validator_only_fields=False,
        )
        selected = _selected_catalog(modules, converted)
        fixture = _fixture_mapping(
            selected,
            model,
            population,
            opacity,
        )
        _validate_fixture_mapping(fixture)

        full_actual = np.asarray(
            population.runtime_state.ion_stage_populations_by_packed_slot
        )
        full_support = np.asarray(
            population.partition_normalized_population_over_mass_density_and_fractional_doppler_width
        )
        full_widths = np.asarray(population.fractional_doppler_widths)
        full_line = _accumulate_line(
            modules,
            selected,
            model,
            population,
            opacity,
            actual_populations=full_actual,
            line_support=full_support,
            fractional_widths=full_widths,
        )

        projected_actual = np.zeros((80, 1006), dtype=np.float64)
        projected_actual[:, [0, 2, 840]] = fixture["actual_population_slot_values"]
        projected_support = np.zeros((80, 1006), dtype=np.float64)
        projected_support[:, 350] = fixture[
            "partition_normalized_population_over_mass_density_and_"
            "fractional_doppler_width_at_line_slot"
        ]
        projected_widths = np.zeros((80, 1006), dtype=np.float64)
        projected_widths[:, 350] = fixture["fractional_doppler_widths_at_line_slot"]
        projected_line = _accumulate_line(
            modules,
            selected,
            model,
            population,
            opacity,
            actual_populations=projected_actual,
            line_support=projected_support,
            fractional_widths=projected_widths,
        )

        placeholder_model, placeholder_population, placeholder_opacity = (
            _run_fixed_state_route(
                modules,
                structured,
                perturb_validator_only_fields=True,
            )
        )
        placeholder_fixture = _fixture_mapping(
            selected,
            placeholder_model,
            placeholder_population,
            placeholder_opacity,
        )
        _validate_fixture_mapping(placeholder_fixture)
        placeholder_line = _accumulate_line(
            modules,
            selected,
            placeholder_model,
            placeholder_population,
            placeholder_opacity,
            actual_populations=(
                placeholder_population.runtime_state.ion_stage_populations_by_packed_slot
            ),
            line_support=(
                placeholder_population.partition_normalized_population_over_mass_density_and_fractional_doppler_width
            ),
            fractional_widths=placeholder_population.fractional_doppler_widths,
        )

    full_output = np.asarray(full_line.line_mass_absorption_coefficient)
    projected_output = np.asarray(projected_line.line_mass_absorption_coefficient)
    placeholder_output = np.asarray(placeholder_line.line_mass_absorption_coefficient)
    if (
        int(full_line.selected_line_count) != 1
        or int(projected_line.selected_line_count) != 1
        or int(placeholder_line.selected_line_count) != 1
    ):
        raise RuntimeError("one-line selected count changed")
    if full_output.shape != (80, 30_000) or full_output.dtype != np.float32:
        raise RuntimeError("full selected-line output contract changed")
    if (
        projected_output.shape != full_output.shape
        or projected_output.dtype != full_output.dtype
        or not np.array_equal(projected_output, full_output)
    ):
        raise RuntimeError("full and projected selected-line outputs differ")
    if not np.array_equal(placeholder_output, full_output):
        raise RuntimeError("validator-only placeholder fields changed line output")
    if array_sha256(full_output) != EXPECTED_LINE_OUTPUT_SHA256:
        raise RuntimeError("selected-line feasibility output hash changed")
    for name, expected_hash in EXPECTED_FULL_ARRAY_HASHES.items():
        values = {
            "ion_stage_populations_by_packed_slot": full_actual,
            (
                "partition_normalized_population_over_mass_density_and_"
                "fractional_doppler_width"
            ): full_support,
            "fractional_doppler_widths": full_widths,
        }[name]
        if values.shape != (80, 1006) or values.dtype != np.float64:
            raise RuntimeError(f"full upstream array contract changed: {name}")
        if array_sha256(values) != expected_hash:
            raise RuntimeError(f"full upstream feasibility hash changed: {name}")

    for name in FIXTURE_SCHEMA:
        if not np.array_equal(placeholder_fixture[name], fixture[name]):
            raise RuntimeError(f"validator-only placeholder owns fixture member {name}")

    relative_reads: list[str] = []
    for path in sorted(set(dynamic_reads)):
        try:
            relative_reads.append(str(path.relative_to(PINNED_ROOT.resolve())))
        except ValueError as error:
            raise RuntimeError(
                f"undeclared data read occurred outside the pin: {path}"
            ) from error
    if tuple(relative_reads) != EXPECTED_DYNAMIC_READ_SET:
        raise RuntimeError(
            "dynamic data read set changed; "
            f"expected={EXPECTED_DYNAMIC_READ_SET!r}, "
            f"actual={tuple(relative_reads)!r}"
        )
    if any(
        forbidden in relative_path
        for relative_path in relative_reads
        for forbidden in (
            "continuum_level_tables.npz",
            "observed_atomic_lines.npy",
        )
    ):
        raise RuntimeError("forbidden source entered the dynamic read set")
    loaded_manifest = _assert_loaded_manifest()

    fixture_hashes = {name: array_sha256(value) for name, value in fixture.items()}
    full_hashes = {
        "ion_stage_populations_by_packed_slot": array_sha256(full_actual),
        (
            "partition_normalized_population_over_mass_density_and_"
            "fractional_doppler_width"
        ): array_sha256(full_support),
        "fractional_doppler_widths": array_sha256(full_widths),
    }
    projected_hashes = {
        "ion_stage_populations_by_packed_slot": array_sha256(projected_actual),
        (
            "partition_normalized_population_over_mass_density_and_"
            "fractional_doppler_width"
        ): array_sha256(projected_support),
        "fractional_doppler_widths": array_sha256(projected_widths),
    }
    if projected_hashes != EXPECTED_PROJECTED_ARRAY_HASHES:
        raise RuntimeError("projected reconstruction feasibility hashes changed")
    payload_evidence: dict[str, Any] = {
        "payload__structured_source_sha256": DATA_IDENTITIES[
            "examples/data/sun_structured_atmosphere.npz"
        ][1],
        "payload__conversion_version": np.asarray(CONVERSION_VERSION, dtype=np.int64),
        "payload__converter_sha256": CONVERTER_SHA256,
        "payload__full_array_names": np.asarray(sorted(full_hashes)),
        "payload__full_array_sha256": np.asarray(
            [full_hashes[name] for name in sorted(full_hashes)]
        ),
        "payload__projected_array_names": np.asarray(sorted(projected_hashes)),
        "payload__projected_array_sha256": np.asarray(
            [projected_hashes[name] for name in sorted(projected_hashes)]
        ),
        "payload__full_projected_line_equal": np.asarray(True, dtype=np.bool_),
        "payload__placeholder_fixture_equal": np.asarray(True, dtype=np.bool_),
        "payload__placeholder_line_equal": np.asarray(True, dtype=np.bool_),
        "payload__selected_line_count": np.asarray(1, dtype=np.int64),
        "payload__line_output_sha256": EXPECTED_LINE_OUTPUT_SHA256,
    }
    evidence_values: dict[str, Any] = {
        **payload_evidence,
        **environment,
        "meta__capture_schema_version": np.asarray(
            CAPTURE_SCHEMA_VERSION, dtype=np.int64
        ),
        "meta__capture_scope_complete": np.asarray(False, dtype=np.bool_),
        "meta__capture_scope": np.asarray(
            "candidate fixture capture; independent review pending"
        ),
        "meta__fixture_publication_performed": np.asarray(False, dtype=np.bool_),
        "meta__golden_read_performed": np.asarray(False, dtype=np.bool_),
        "meta__golden_write_performed": np.asarray(False, dtype=np.bool_),
        "meta__manifest_write_performed": np.asarray(False, dtype=np.bool_),
        "meta__worker_sha256": sha256(WORKER_PATH),
        "meta__python_version": platform.python_version(),
        "meta__numpy_version": np.__version__,
        "meta__numba_version": modules.numba.__version__,
        "meta__platform": platform.platform(),
        "meta__system_byteorder": sys.byteorder,
        "meta__fixture_member_names": np.asarray(sorted(fixture)),
        "meta__fixture_member_count": np.asarray(19, dtype=np.int64),
        "meta__fixture_member_sha256": np.asarray(
            [fixture_hashes[name] for name in sorted(fixture)]
        ),
        "meta__fixture_schema_digest": np.asarray("0" * 64),
        "meta__payload_fingerprint": np.asarray("0" * 64),
        "meta__full_capture_schema_digest": np.asarray("0" * 64),
        "meta__full_capture_fingerprint": np.asarray("0" * 64),
        "identity__names": np.asarray(sorted(identities)),
        "identity__sha256_or_commit": np.asarray(
            [identities[name] for name in sorted(identities)]
        ),
        "identity__loaded_python_relative_paths": np.asarray(sorted(loaded_manifest)),
        "identity__loaded_python_sha256": np.asarray(
            [loaded_manifest[name] for name in sorted(loaded_manifest)]
        ),
        "identity__converted_member_names": np.asarray(sorted(converted["fields"])),
        "identity__converted_member_sha256": np.asarray(
            [
                converted["fields"][name]["sha256"]
                for name in sorted(converted["fields"])
            ]
        ),
        "identity__converted_words_sha256": converted["words_sha256"],
        "identity__converter_child_loaded_relative_paths": np.asarray(
            sorted(converted["loaded_staged_source_sha256"])
        ),
        "identity__converter_child_loaded_sha256": np.asarray(
            [
                converted["loaded_staged_source_sha256"][name]
                for name in sorted(converted["loaded_staged_source_sha256"])
            ]
        ),
        "source__structured_member_names": np.asarray(sorted(structured)),
        "source__structured_member_sha256": np.asarray(
            [array_sha256(structured[name]) for name in sorted(structured)]
        ),
        "source__abundance_roundtrip_bitwise": np.asarray(True, dtype=np.bool_),
        "source__depth_order_outer_to_inner": np.asarray(True, dtype=np.bool_),
        "configuration__effective_temperature": np.asarray(5778.0, dtype=np.float64),
        "configuration__log_surface_gravity": np.asarray(4.44, dtype=np.float64),
        "configuration__temperature_iteration_index": np.asarray(1, dtype=np.int64),
        "configuration__pressure_iteration_enabled": np.asarray(False, dtype=np.bool_),
        "configuration__molecules_enabled": np.asarray(True, dtype=np.bool_),
        "configuration__convection_enabled": np.asarray(False, dtype=np.bool_),
        "configuration__opacity_flags": np.asarray(_opacity_flags(modules)),
        "read_set__relative_paths": np.asarray(relative_reads),
        "read_set__sha256": np.asarray(
            [DATA_IDENTITIES[name][1] for name in relative_reads]
        ),
        "full__ion_stage_populations_by_packed_slot": full_actual,
        (
            "full__partition_normalized_population_over_mass_density_and_"
            "fractional_doppler_width"
        ): full_support,
        "full__fractional_doppler_widths": full_widths,
        "projected__ion_stage_populations_by_packed_slot": projected_actual,
        (
            "projected__partition_normalized_population_over_mass_density_and_"
            "fractional_doppler_width"
        ): projected_support,
        "projected__fractional_doppler_widths": projected_widths,
        "line__full_output": full_output,
        "line__projected_output": projected_output,
        "line__placeholder_output": placeholder_output,
        "line__full_output_sha256": EXPECTED_LINE_OUTPUT_SHA256,
        "line__projected_output_sha256": array_sha256(projected_output),
        "line__placeholder_output_sha256": array_sha256(placeholder_output),
        "line__full_projected_bitwise_equal": np.asarray(True, dtype=np.bool_),
        "line__placeholder_bitwise_equal": np.asarray(True, dtype=np.bool_),
        "placeholder__fixture_member_names": np.asarray(sorted(placeholder_fixture)),
        "placeholder__fixture_member_sha256": np.asarray(
            [
                array_sha256(placeholder_fixture[name])
                for name in sorted(placeholder_fixture)
            ]
        ),
        "placeholder__all_fixture_members_bitwise_equal": np.asarray(
            True, dtype=np.bool_
        ),
        "placeholder__rosseland_opacity": placeholder_model.rosseland_opacity,
        "placeholder__radiative_acceleration": (
            placeholder_model.radiative_acceleration
        ),
        "placeholder__convective_flux": placeholder_model.convective_flux,
        "placeholder__convective_velocity": (placeholder_model.convective_velocity),
    }
    evidence = _detached_mapping(evidence_values)
    payload_mapping = _detached_mapping(
        {
            **{f"fixture__{name}": value for name, value in fixture.items()},
            **{
                name: value
                for name, value in evidence.items()
                if name.startswith("payload__")
            },
        }
    )
    evidence["meta__fixture_schema_digest"] = np.asarray(mapping_schema_digest(fixture))
    evidence["meta__payload_fingerprint"] = np.asarray(
        mapping_fingerprint(payload_mapping)
    )
    evidence["meta__full_capture_schema_digest"] = np.asarray(
        mapping_schema_digest(_combined_mapping(fixture, evidence))
    )
    full_fingerprint_evidence = {
        name: value
        for name, value in evidence.items()
        if name != "meta__full_capture_fingerprint"
    }
    evidence["meta__full_capture_fingerprint"] = np.asarray(
        mapping_fingerprint(_combined_mapping(fixture, full_fingerprint_evidence))
    )
    evidence = _detached_mapping(evidence)
    if str(evidence["meta__fixture_schema_digest"]) != mapping_schema_digest(fixture):
        raise RuntimeError("fixture schema digest is inconsistent")
    if str(evidence["meta__payload_fingerprint"]) != mapping_fingerprint(
        payload_mapping
    ):
        raise RuntimeError("fixture payload fingerprint is inconsistent")
    if str(evidence["meta__full_capture_schema_digest"]) != mapping_schema_digest(
        _combined_mapping(fixture, evidence)
    ):
        raise RuntimeError("full capture schema digest is inconsistent")
    recomputed_full_fingerprint = mapping_fingerprint(
        _combined_mapping(
            fixture,
            {
                name: value
                for name, value in evidence.items()
                if name != "meta__full_capture_fingerprint"
            },
        )
    )
    if str(evidence["meta__full_capture_fingerprint"]) != recomputed_full_fingerprint:
        raise RuntimeError("full capture fingerprint is inconsistent")
    return InMemoryFixtureCapture(
        fixture_arrays=_detached_mapping(fixture),
        ephemeral_evidence=evidence,
    )


def summarize_capture(capture: InMemoryFixtureCapture) -> dict[str, Any]:
    """Return a compact JSON-safe account of an in-memory candidate."""

    fixture = capture.fixture_arrays
    evidence = capture.ephemeral_evidence
    _validate_fixture_mapping(fixture)
    identity_mapping = {
        str(name): str(value)
        for name, value in zip(
            evidence["identity__names"],
            evidence["identity__sha256_or_commit"],
            strict=True,
        )
    }
    return {
        "capture_scope_complete": bool(evidence["meta__capture_scope_complete"]),
        "fixture_member_count": len(fixture),
        "fixture_schema_digest": str(evidence["meta__fixture_schema_digest"]),
        "payload_fingerprint": str(evidence["meta__payload_fingerprint"]),
        "full_capture_schema_digest": str(evidence["meta__full_capture_schema_digest"]),
        "full_capture_fingerprint": str(evidence["meta__full_capture_fingerprint"]),
        "evidence_member_count": len(evidence),
        "worker_sha256": str(evidence["meta__worker_sha256"]),
        "fixture_schema": {
            name: {
                "shape": list(np.asarray(values).shape),
                "dtype": np.asarray(values).dtype.str,
                "sha256": array_sha256(values),
            }
            for name, values in fixture.items()
        },
        "full_array_hashes": {
            name: EXPECTED_FULL_ARRAY_HASHES[name]
            for name in sorted(EXPECTED_FULL_ARRAY_HASHES)
        },
        "projected_array_hashes": {
            name: EXPECTED_PROJECTED_ARRAY_HASHES[name]
            for name in sorted(EXPECTED_PROJECTED_ARRAY_HASHES)
        },
        "line_output_sha256": str(evidence["line__full_output_sha256"]),
        "full_projected_bitwise_equal": bool(
            evidence["line__full_projected_bitwise_equal"]
        ),
        "placeholder_fixture_bitwise_equal": bool(
            evidence["placeholder__all_fixture_members_bitwise_equal"]
        ),
        "placeholder_line_bitwise_equal": bool(
            evidence["line__placeholder_bitwise_equal"]
        ),
        "dynamic_read_set": [
            str(value) for value in evidence["read_set__relative_paths"]
        ],
        "loaded_python_source_count": int(
            evidence["identity__loaded_python_relative_paths"].size
        ),
        "staged_converter_dependency_hashes": {
            relative_path: identity_mapping[
                f"staged_converter_source__{relative_path}__sha256"
            ]
            for relative_path in STAGED_CONVERTER_DEPENDENCIES
        },
        "pinned_converter_dependency_hashes": {
            relative_path: identity_mapping[f"source__{relative_path}__sha256"]
            for relative_path in STAGED_CONVERTER_DEPENDENCIES
        },
        "converter_child_loaded_source_hashes": {
            str(relative_path): str(source_hash)
            for relative_path, source_hash in zip(
                evidence["identity__converter_child_loaded_relative_paths"],
                evidence["identity__converter_child_loaded_sha256"],
                strict=True,
            )
        },
        "fixture_publication_performed": bool(
            evidence["meta__fixture_publication_performed"]
        ),
        "golden_read_performed": bool(evidence["meta__golden_read_performed"]),
        "golden_write_performed": bool(evidence["meta__golden_write_performed"]),
        "manifest_write_performed": bool(evidence["meta__manifest_write_performed"]),
        "observed_row_role": "non_authoritative_packing_corroboration",
        "observed_row_index": NON_AUTHORITATIVE_OBSERVED_ROW_INDEX,
    }


def main() -> None:
    """Print one compact summary; never serialize the fixture or evidence."""

    if sys.argv[1:]:
        raise SystemExit("this worker accepts no paths or publication options")
    print(json.dumps(summarize_capture(build_fixture_capture()), sort_keys=True))


if __name__ == "__main__":
    main()
