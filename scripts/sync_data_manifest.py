#!/usr/bin/env python3
"""Synchronize exhaustive array metadata in the textbook data manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "data" / "MANIFEST.json"
PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
CHAPTER04_FIXTURE_SHA256 = (
    "351bba75dca1fa502f5cc2a108035f69f2e31c760a90133480f2e7fe31e45f79"
)
CHAPTER04_PUBLISHER_SHA256 = (
    "e1c7df9ecdd1222aaa340f527b1ba291ac61855d459f15c470ff562f6e38dd40"
)
CHAPTER04_ACCEPTANCE_SHA256 = (
    "6416e5890f29faf6defab48702ce76d972f40f1f37148fd9362cf3c433c8b76b"
)
CHAPTER04_CAPTURE_CONTRACT_SHA256 = (
    "ac170894a3ca5f7c8eda6dc1a3cd19688b3254e10c32a729f46b09c36344efff"
)
CHAPTER04_CONSTANTS_SHA256 = (
    "cf742ecc5181589e2b7f8b56c7b2d82bd203c9f303de263a5b1c863adaba40a0"
)
CHAPTER04_GOLDEN_BUILDER = "scripts/build_chapter04_payne_zero_goldens.py"

CHAPTER05_GOLDEN_BUILDER = "scripts/build_chapter05_payne_zero_goldens.py"
CHAPTER05_GOLDEN_DIRECTORY = (
    REPOSITORY_ROOT / "data" / "golden" / "payne_zero" / "chapter05"
)
CHAPTER05_PUBLICATION_ACCEPTANCE_PATH = (
    REPOSITORY_ROOT / "design" / "chapter05_publication_acceptance.json"
)
CHAPTER05_PUBLICATION_ACCEPTANCE_SHA256 = (
    "80bcfea8e6a817ac1a279b136a247cdd482af1b94af312e94dfa564ad40cafaf"
)
CHAPTER05_PUBLISHER_SHA256 = (
    "6e5ed476cbcd64b1599c9f369bf5ed746d9f561f9bd77c13db8b9ec1c1cdad81"
)
CHAPTER05_WORKER_SHA256 = (
    "429252d5fefd2b911ce4321578820aa67b505d5fe37b174cb647d4b6177d7389"
)
CHAPTER05_CAPTURE_CONTRACT_SHA256 = (
    "4198f76419f102efbb3468b5f2ed7ddca7ff6af776ffc566cfa4058b6164fdaf"
)
CHAPTER05_EXACT_SOURCE_CONTRACT_SHA256 = (
    "ec1c84519a898a454a408780bd6b36fa723fc7bade83a98e11eede1343bf2956"
)
CHAPTER05_PUBLISHER_CONTRACT_SHA256 = (
    "8a58866b315b2cdd1ca769436aa0ccdbed3b08e4ef180bd309179d6cc7e04c08"
)
CHAPTER05_ORACLE_ACCEPTANCE_SHA256 = (
    "892279cd2dfa850c3eabfb8ce94953e4723db5270da319bcedb9bc90c9597474"
)
CHAPTER05_DETERMINISTIC_NPZ_SHA256 = (
    "f4886766524d79623e648d28ab9d24215da42f8bf1f859f69381c546f9c96e49"
)
CHAPTER05_FIXTURE_SHA256 = (
    "ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae"
)
CHAPTER05_FIXTURE_PAYLOAD_DIGEST = (
    "4bcf0bbd8d61e58334c4c7ef6caaaf9ca47e6fb4536ad0098d5a541d540ec048"
)
CHAPTER05_RAW_SCHEMA_VERSION = 2
CHAPTER05_RAW_KEY_COUNT = 1161
CHAPTER05_RAW_SCHEMA_DIGEST = (
    "652c110dc79a6f6dfca6893bee35416289675b4920a5d0dcfe6b2cb262dacf3d"
)
CHAPTER05_PHYSICAL_PAYLOAD_FINGERPRINT = (
    "d223351fa2c51dc24a1b01896da9ab9a82fc475f4082c47fde34734d8dc03343"
)
CHAPTER05_FULL_CAPTURE_FINGERPRINT = (
    "3d2c131711e1c0dc6aa088892193bb24d41a76d005bc20dd1c42d3e84f66e656"
)
CHAPTER05_LOADED_SOURCE_COUNT = 52
CHAPTER05_LOADED_SOURCE_MANIFEST_DIGEST = (
    "aa7656f728b646282d809c774b59ee2b5112c0ac46c1382c2badcbbfcc85dc6b"
)
CHAPTER05_INVENTORY_MAPPING_DIGEST = (
    "b02a6a2896d3468d1052441f8def0841b608eb5d5816ccd72539a0ec982c452c"
)

CHAPTER06_FE_SUBSET_BUILDER = "scripts/build_chapter06_fe_record_subset.py"
CHAPTER06_FE_SUBSET_BUILDER_SHA256 = (
    "25bcf4662740155e8b08615b9522f3f4517e1a5ddc4627c68686620ccfff4d6c"
)
CHAPTER06_FE_SUBSET_SHA256 = (
    "bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04"
)
CHAPTER06_FE_SUBSET_BYTES = 8665
CHAPTER06_FE_SOURCE_ARCHIVE_SHA256 = (
    "4eafa927c02a4f74401523149a44e35239f2aaecb4a64f2905a4cd5530c2dde7"
)

CHAPTER05_READER_NAME = "chapter05_continuum_reader_cpu_float64.npz"
CHAPTER05_INTEGRATION_NAME = "chapter05_continuum_integration_cpu_float64.npz"
CHAPTER05_GOLDEN_ARCHIVES = {
    CHAPTER05_READER_NAME: {
        "sha256": ("84e3e764b2d9be8d409b0db0271decdd90e07b8e716f5c589fbdbb1b9e2ae39f"),
        "bytes": 1022580,
        "archive_kind": "continuum_reader",
        "archive_schema_version": 1,
        "key_count": 257,
        "schema_digest": (
            "058389fcd0e944dd4c1ad1208adbaac44d53ab35e61e5366d37a8a141ad91f88"
        ),
        "member_name_digest": (
            "f3c671b36472fedfbaf44693b2a1e40bac474e53717b18ecfb9316b2d2830ae3"
        ),
    },
    CHAPTER05_INTEGRATION_NAME: {
        "sha256": ("5ce6805353ed760dc42c42ce6bd59472a3b7a7443a486aded175caf755fbff73"),
        "bytes": 21608505,
        "archive_kind": "continuum_integration",
        "archive_schema_version": 1,
        "key_count": 1079,
        "schema_digest": (
            "e09a8932d97f8a2756aca2b779c4b8fdd822ca239197fd908ee575d09df081ca"
        ),
        "member_name_digest": (
            "e8dd3521a6747c080a7c4b7d9e3653fcac841f279f7b7a233cd4c0a5a8cde086"
        ),
        "reader_archive_sha256": (
            "84e3e764b2d9be8d409b0db0271decdd90e07b8e716f5c589fbdbb1b9e2ae39f"
        ),
        "reader_archive_schema_digest": (
            "058389fcd0e944dd4c1ad1208adbaac44d53ab35e61e5366d37a8a141ad91f88"
        ),
        "inventory_mapping_digest": CHAPTER05_INVENTORY_MAPPING_DIGEST,
    },
}

CHAPTER05_REPOSITORY_IDENTITIES = {
    "scripts/build_chapter05_payne_zero_goldens.py": CHAPTER05_PUBLISHER_SHA256,
    "scripts/chapter05_oracle_worker.py": CHAPTER05_WORKER_SHA256,
    "scripts/deterministic_npz.py": CHAPTER05_DETERMINISTIC_NPZ_SHA256,
    "design/chapter05_oracle_capture_contract.md": (CHAPTER05_CAPTURE_CONTRACT_SHA256),
    "design/chapter05_exact_source_contract.md": (
        CHAPTER05_EXACT_SOURCE_CONTRACT_SHA256
    ),
    "design/chapter05_publisher_contract.md": CHAPTER05_PUBLISHER_CONTRACT_SHA256,
    "design/chapter05_oracle_worker_acceptance.md": (
        CHAPTER05_ORACLE_ACCEPTANCE_SHA256
    ),
    "design/chapter05_publication_acceptance.json": (
        CHAPTER05_PUBLICATION_ACCEPTANCE_SHA256
    ),
    "data/fixtures/chapter05_continuum_states.npz": CHAPTER05_FIXTURE_SHA256,
}

# Populated only after an archive has passed its exact path-independent byte,
# schema, and NUL-delimited member-name gates.  Unit routing can therefore
# never bless a future namespace merely because it resembles a current one.
_VALIDATED_CHAPTER05_MEMBERS: dict[str, frozenset[str]] = {}

CHAPTER04_ENTRY_SPECS = (
    {
        "path": (
            "data/static/source_catalogs/lines/molecular_equilibrium_atmosphere.npz"
        ),
        "role": "static",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/source_catalogs/lines/"
            "molecular_equilibrium_atmosphere.npz"
        ),
        "source_sha256": (
            "971525641062d8cdb28ddb2955117627290ef223885695b5fd99088aa441a644"
        ),
        "scope": "Complete 170-record atmosphere molecular-equilibrium catalog.",
    },
    {
        "path": (
            "data/static/source_catalogs/lines/molecular_equilibrium_synthesis.npz"
        ),
        "role": "static",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/source_catalogs/lines/"
            "molecular_equilibrium_synthesis.npz"
        ),
        "source_sha256": (
            "3e8c1ea69fe672b9886bda38922f868c6d2ac2b43c4eb0d7750620241c238d28"
        ),
        "scope": "Complete 190-record synthesis molecular-equilibrium catalog.",
    },
    {
        "path": ("data/static/atmosphere_tables/molecular_equilibrium_tables.npz"),
        "role": "static",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/atmosphere_tables/"
            "molecular_equilibrium_tables.npz"
        ),
        "source_sha256": (
            "1e23fbfdca3062998fda0857ffd22fcf3909be505ed3288b422bf6b8d8e7bbbe"
        ),
        "scope": (
            "Exact atmosphere H2 partition table bundle shared by molecular "
            "chemistry and later opacity chapters."
        ),
    },
    {
        "path": "data/static/synthesis_tables/continuum_edge_grid.npz",
        "role": "static",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/synthesis_tables/"
            "continuum_edge_grid.npz"
        ),
        "source_sha256": (
            "11b77ecf752f04b37d34299b13c11aeccbc15bbbafde0d5324ca180e3b1d3f3e"
        ),
        "scope": (
            "Exact invariant continuum-edge arrays required by the synthesis "
            "structured-atmosphere builder; their opacity use is deferred."
        ),
    },
    {
        "path": "data/fixtures/chapter04_molecular_inputs.npz",
        "role": "fixture",
        "source": "data/fixtures/chapter03_atom_only_inputs.npz",
        "source_sha256": (
            "3ed0d65431fc9e284a77011b82267241b25cc56cdffa73e1bc86eec15f9b5219"
        ),
        "scope": (
            "Input-only six-depth molecular thermochemistry track plus "
            "independent limiting and exact branch-boundary controls; not a "
            "converged atmosphere."
        ),
    },
    {
        "path": (
            "data/golden/payne_zero/chapter04/"
            "chapter04_molecular_constants_cpu_float64.npz"
        ),
        "role": "golden",
        "source": "/Users/ysting/payne-zero",
        "source_commit": PAYNE_ZERO_COMMIT,
        "requires_optional_full_catalog": False,
        "scope": (
            "Single owner of the lossless atmosphere and synthesis molecular "
            "catalogs, their semantic alignment, exact branch-boundary "
            "witnesses, H2 probes, and capture provenance."
        ),
        "builder": CHAPTER04_GOLDEN_BUILDER,
        "fixture_sha256": CHAPTER04_FIXTURE_SHA256,
        "publisher_sha256": CHAPTER04_PUBLISHER_SHA256,
        "oracle_acceptance_sha256": CHAPTER04_ACCEPTANCE_SHA256,
        "capture_contract_sha256": CHAPTER04_CAPTURE_CONTRACT_SHA256,
        "capture_policy": "two fresh byte-identical capture sets",
        "sha256": CHAPTER04_CONSTANTS_SHA256,
        "bytes": 235675,
    },
    {
        "path": (
            "data/golden/payne_zero/chapter04/"
            "chapter04_atmosphere_molecular_state_cpu_float64.npz"
        ),
        "role": "golden",
        "source": "/Users/ysting/payne-zero",
        "source_commit": PAYNE_ZERO_COMMIT,
        "requires_optional_full_catalog": False,
        "scope": (
            "Single owner of exact atmosphere molecular-state routes, Newton "
            "histories, normalization handoffs, warm-start and energy modes, "
            "disabled behavior, scheduler inventory, and bridge witnesses."
        ),
        "builder": CHAPTER04_GOLDEN_BUILDER,
        "fixture_sha256": CHAPTER04_FIXTURE_SHA256,
        "publisher_sha256": CHAPTER04_PUBLISHER_SHA256,
        "oracle_acceptance_sha256": CHAPTER04_ACCEPTANCE_SHA256,
        "capture_contract_sha256": CHAPTER04_CAPTURE_CONTRACT_SHA256,
        "constants_archive_sha256": CHAPTER04_CONSTANTS_SHA256,
        "capture_policy": "two fresh byte-identical capture sets",
        "sha256": ("e0c80f66bf74776d29947c6ced204402c36678edce9b3bdddebd9bd79713ec2a"),
        "bytes": 1047808,
    },
    {
        "path": (
            "data/golden/payne_zero/chapter04/"
            "chapter04_synthesis_molecular_full_cpu_float64.npz"
        ),
        "role": "golden",
        "source": "/Users/ysting/payne-zero",
        "source_commit": PAYNE_ZERO_COMMIT,
        "requires_optional_full_catalog": False,
        "scope": (
            "Single owner of the exact full synthesis molecular EOS route, "
            "including its two ordered molecular solves, diagnostics, "
            "published state, input aliases, and call-chain provenance."
        ),
        "builder": CHAPTER04_GOLDEN_BUILDER,
        "fixture_sha256": CHAPTER04_FIXTURE_SHA256,
        "publisher_sha256": CHAPTER04_PUBLISHER_SHA256,
        "oracle_acceptance_sha256": CHAPTER04_ACCEPTANCE_SHA256,
        "capture_contract_sha256": CHAPTER04_CAPTURE_CONTRACT_SHA256,
        "constants_archive_sha256": CHAPTER04_CONSTANTS_SHA256,
        "capture_policy": "two fresh byte-identical capture sets",
        "sha256": ("c78393d8525bb637706b5f4dbb17aae7f24660e468bbbfd84dff659ceb8edf31"),
        "bytes": 471636,
    },
    {
        "path": (
            "data/golden/payne_zero/chapter04/"
            "chapter04_synthesis_molecular_fixed_cpu_float64.npz"
        ),
        "role": "golden",
        "source": "/Users/ysting/payne-zero",
        "source_commit": PAYNE_ZERO_COMMIT,
        "requires_optional_full_catalog": False,
        "scope": (
            "Single owner of the exact fixed-EOS synthesis routes with derived "
            "and supplied mass density, preserving their separate calls, "
            "outputs, aliases, and diagnostics without duplicating inputs."
        ),
        "builder": CHAPTER04_GOLDEN_BUILDER,
        "fixture_sha256": CHAPTER04_FIXTURE_SHA256,
        "publisher_sha256": CHAPTER04_PUBLISHER_SHA256,
        "oracle_acceptance_sha256": CHAPTER04_ACCEPTANCE_SHA256,
        "capture_contract_sha256": CHAPTER04_CAPTURE_CONTRACT_SHA256,
        "constants_archive_sha256": CHAPTER04_CONSTANTS_SHA256,
        "capture_policy": "two fresh byte-identical capture sets",
        "sha256": ("c4aeff4b3afba423e3ab9613bd0eef813b3487f7bf792e6b95bcbbf0db46ca12"),
        "bytes": 649052,
    },
    {
        "path": (
            "data/golden/payne_zero/chapter04/"
            "chapter04_molecular_public_mapping_cpu_float64.npz"
        ),
        "role": "golden",
        "source": "/Users/ysting/payne-zero",
        "source_commit": PAYNE_ZERO_COMMIT,
        "requires_optional_full_catalog": False,
        "scope": (
            "Single owner of the exact public structured-atmosphere molecular "
            "mapping, all 54 line-species lanes, no-ground partition policy, "
            "independent CO reconstruction, edge loading, and mutation trace."
        ),
        "builder": CHAPTER04_GOLDEN_BUILDER,
        "fixture_sha256": CHAPTER04_FIXTURE_SHA256,
        "publisher_sha256": CHAPTER04_PUBLISHER_SHA256,
        "oracle_acceptance_sha256": CHAPTER04_ACCEPTANCE_SHA256,
        "capture_contract_sha256": CHAPTER04_CAPTURE_CONTRACT_SHA256,
        "constants_archive_sha256": CHAPTER04_CONSTANTS_SHA256,
        "capture_policy": "two fresh byte-identical capture sets",
        "sha256": ("a40dcd105641e4620a7c8e9362f0270437e3fe87e7e1e9ed3209b9b566f32628"),
        "bytes": 590138,
    },
)

CHAPTER05_ENTRY_SPECS = (
    {
        "path": "data/static/atmosphere_tables/continuum_opacity_tables.npz",
        "role": "static",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/atmosphere_tables/"
            "continuum_opacity_tables.npz"
        ),
        "source_sha256": (
            "6fd4c556418870c28d3fcc9a050252af58ac4cc433cae979477355c8c7d593e3"
        ),
        "scope": (
            "Exact atmosphere continuum tables; includes actively consumed "
            "cross sections and loader-required fields owned by synthesis."
        ),
    },
    {
        "path": "data/static/atmosphere_tables/karzas_latter_tables.npz",
        "role": "static",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/atmosphere_tables/"
            "karzas_latter_tables.npz"
        ),
        "source_sha256": (
            "23805dc17c47af45b8ae63b2e278e1fb6c584a01c87d1eb3c31306e4555e6d15"
        ),
        "scope": "Exact atmosphere Karzas-Latter hydrogenic cross-section tables.",
    },
    {
        "path": "data/static/synthesis_tables/continuum_tables.npz",
        "role": "static",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/synthesis_tables/"
            "continuum_tables.npz"
        ),
        "source_sha256": (
            "406ea952ab8a849c0fee5d04d27882cb92184c30fcdcbaa901d71b8b310a823d"
        ),
        "scope": (
            "Exact synthesis continuum tables for the standard edge-triplet "
            "route and explicitly labeled sampled extension."
        ),
    },
    {
        "path": "data/fixtures/chapter05_continuum_states.npz",
        "role": "fixture",
        "source": "data/fixtures/chapter04_molecular_inputs.npz",
        "source_sha256": (
            "351bba75dca1fa502f5cc2a108035f69f2e31c760a90133480f2e7fe31e45f79"
        ),
        "scope": (
            "Four controlled six-depth continuum input regimes rebuilt by "
            "accepted Chapter 4 atmosphere and structured-synthesis routes; "
            "these are stage inputs, not converged stellar atmospheres or goldens."
        ),
    },
)

CHAPTER06_ENTRY_SPECS = (
    {
        "path": "data/static/atmosphere_tables/line_opacity_tables.npz",
        "role": "static",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/atmosphere_tables/"
            "line_opacity_tables.npz"
        ),
        "source_sha256": (
            "89f486122cb8939b23dc5423145a46d88a77df8daf57a1def35055b7b8205f16"
        ),
        "scope": (
            "Exact atmosphere coarse Harris source tables used to construct "
            "the independent 2001-point CPU Voigt-profile basis."
        ),
    },
    {
        "path": "data/static/synthesis_tables/line_profile_tables.npz",
        "role": "static",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/synthesis_tables/"
            "line_profile_tables.npz"
        ),
        "source_sha256": (
            "87b47fc76bed10455218f43c4b6686525b961002e72d6a5ef01255a08deb27d4"
        ),
        "scope": (
            "Exact synthesis line-profile archive. Chapter 6 consumes only "
            "the three Harris arrays; hydrogen/Stark members remain unopened "
            "and their teaching is deferred to Chapter 7."
        ),
    },
    {
        "path": "data/subsets/chapter06_fe_i_source_row_873702.npz",
        "role": "subset",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/source_catalogs/lines/"
            "atomic_source_lines_parsed.npz"
        ),
        "source_sha256": CHAPTER06_FE_SOURCE_ARCHIVE_SHA256,
        "source_row_index": 873702,
        "source_archive_bytes": 258021389,
        "source_archive_row_count": 1939975,
        "source_field_count": 17,
        "subset_schema_version": 1,
        "builder": CHAPTER06_FE_SUBSET_BUILDER,
        "builder_sha256": CHAPTER06_FE_SUBSET_BUILDER_SHA256,
        "sha256": CHAPTER06_FE_SUBSET_SHA256,
        "bytes": CHAPTER06_FE_SUBSET_BYTES,
        "requires_optional_full_catalog": True,
        "scope": (
            "Deterministic one-record teaching subset of accepted raw row "
            "873702: an ordinary, correction-free Fe I source record. It "
            "contains source fields and provenance only, never computed line "
            "opacity or a comparison golden."
        ),
    },
)

CHAPTER09_ENTRY_SPECS = (
    {
        "path": "data/static/synthesis_tables/transfer_tables.npz",
        "role": "static",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/synthesis_tables/"
            "transfer_tables.npz"
        ),
        "source_sha256": (
            "64f75de9af02697c0b97b7bbf919f6fed9d646622f18859eb5f66ff66e7f7a7b"
        ),
        "scope": (
            "Exact synthesis 51-point optical-depth grid, mean-intensity "
            "operator, and surface Eddington-flux weights used by Chapter 9."
        ),
    },
)

CHAPTER11_ENTRY_SPECS = (
    {
        "path": "data/static/atmosphere_tables/continuum_level_tables.npz",
        "role": "static",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/atmosphere_tables/"
            "continuum_level_tables.npz"
        ),
        "source_sha256": (
            "35a6839be4ff3dd824206c7a6b851b987132313374ede7ea5441f9d0bd69888f"
        ),
        "scope": (
            "Exact compact H/He/C/Mg/Al/Si/K/Ca level tables required by "
            "the Chapter 11 atmosphere-continuum pass."
        ),
    },
    {
        "path": "data/fixtures/chapter11_solar_seed.npz",
        "role": "fixture",
        "source": "/Users/ysting/payne-zero/examples/data/sun_structured_atmosphere.npz",
        "source_sha256": (
            "d686ea7107d60bf1707607e3d6377d283fb3eb7115c170ac2aeef54fbaa6abdb"
        ),
        "builder": "scripts/build_chapter11_inputs.py",
        "sha256": (
            "a14e32467dc381a6ceed8da362c6c5a7e118276a964e461faa3b8843f77374da"
        ),
        "bytes": 13273,
        "scope": (
            "Supplied 80-layer solar-like fixed-column seed and explicit "
            "previous-support controls; an input fixture, not a convergence claim."
        ),
    },
    {
        "path": "data/subsets/chapter11_observed_atomic_subset.npy",
        "role": "subset",
        "format": "npy",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/source_catalogs/lines/"
            "observed_atomic_lines.npy"
        ),
        "source_sha256": (
            "649b039474ca6cda6e9fe0dea3052e5d47c24320ba8be1f48cd2ec39b7bcf84f"
        ),
        "builder": "scripts/build_chapter11_inputs.py",
        "sha256": (
            "68ee3d3775cd8496b73d1500a490836f888c33364922d6541b1ec7fd8f603a5e"
        ),
        "bytes": 32896,
        "scope": (
            "Strongest 2048 packed observed atomic source rows, kept in "
            "wavelength order for exact compact atmosphere line selection."
        ),
    },
    {
        "path": "data/subsets/chapter11_observed_atomic_subset.provenance.npz",
        "role": "subset",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/source_catalogs/lines/"
            "observed_atomic_lines.npy"
        ),
        "source_sha256": (
            "649b039474ca6cda6e9fe0dea3052e5d47c24320ba8be1f48cd2ec39b7bcf84f"
        ),
        "builder": "scripts/build_chapter11_inputs.py",
        "sha256": (
            "2fe1647a39afb9634bf52d203cf89e8addb08686d9c0f5ad40a82d63971b02a6"
        ),
        "bytes": 18032,
        "scope": (
            "Pinned source rows and selection identity for the Chapter 11 "
            "observed atomic packed subset."
        ),
    },
    {
        "path": "data/subsets/chapter11_detailed_transition_subset.npz",
        "role": "subset",
        "source": (
            "/Users/ysting/payne-zero/source_data_files/source_catalogs/lines/"
            "detailed_transition_lines.npz"
        ),
        "source_sha256": (
            "57b3ff244698965b0c507c3ab23f767d85c37bf4705eb97f0a91c597e2330232"
        ),
        "builder": "scripts/build_chapter11_inputs.py",
        "sha256": (
            "49752ed1627134145f12768feb9e10f3c578ba986e1be90fc75c3a6348601b25"
        ),
        "bytes": 10766,
        "scope": (
            "Sixty-four strong ordinary detailed transitions in source order "
            "for the exact Chapter 11 XLINOP path."
        ),
    },
)

def _sha256_path(path: Path) -> str:
    """Return a file's SHA-256 identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chapter05_schema_digest(arrays: dict[str, np.ndarray]) -> str:
    """Hash exact names, NumPy dtypes, and shapes like the publisher."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        values = np.asarray(arrays[name])
        if values.dtype.hasobject:
            raise RuntimeError(f"{name} has forbidden object dtype")
        digest.update(name.encode())
        digest.update(values.dtype.str.encode())
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _chapter05_inventory_mapping_digest(arrays: dict[str, np.ndarray]) -> str:
    """Hash the exact raw-member ownership mapping like the publisher."""

    digest = hashlib.sha256()
    for member in (
        "inventory__raw_member_name",
        "inventory__disposition",
        "inventory__published_member",
    ):
        if member not in arrays:
            raise RuntimeError(f"integration archive lacks {member}")
        values = np.asarray(arrays[member])
        digest.update(member.encode())
        digest.update(values.dtype.str.encode())
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _chapter05_scalar(arrays: dict[str, np.ndarray], name: str) -> Any:
    """Return one required scalar without silently accepting a vector."""

    if name not in arrays:
        raise RuntimeError(f"Chapter 5 archive lacks required member {name}")
    values = np.asarray(arrays[name])
    if values.shape != ():
        raise RuntimeError(f"{name} must be scalar, found shape {values.shape}")
    return values.item()


def _validate_chapter05_repository_identities() -> None:
    """Reject drift in every file identity represented by the manifest."""

    for relative_path, expected_sha256 in CHAPTER05_REPOSITORY_IDENTITIES.items():
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            raise RuntimeError(
                f"required Chapter 5 identity is absent: {relative_path}"
            )
        actual_sha256 = _sha256_path(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{relative_path} has SHA-256 {actual_sha256}; "
                f"expected {expected_sha256}"
            )


def _validate_chapter05_archive(
    path: Path,
    name: str,
) -> dict[str, np.ndarray]:
    """Validate one accepted archive before exposing any member metadata."""

    expected = CHAPTER05_GOLDEN_ARCHIVES[name]
    actual_sha256 = _sha256_path(path)
    if actual_sha256 != expected["sha256"]:
        raise RuntimeError(
            f"{path} has SHA-256 {actual_sha256}; expected {expected['sha256']}"
        )
    actual_bytes = path.stat().st_size
    if actual_bytes != expected["bytes"]:
        raise RuntimeError(
            f"{path} has {actual_bytes} bytes; expected {expected['bytes']}"
        )

    with np.load(path, allow_pickle=False) as archive:
        if archive.files != sorted(archive.files):
            raise RuntimeError(f"{path} members are not lexically ordered")
        arrays = {
            member: np.array(archive[member], copy=True) for member in archive.files
        }

    names = tuple(arrays)
    if len(names) != expected["key_count"]:
        raise RuntimeError(
            f"{path} has {len(names)} members; expected {expected['key_count']}"
        )
    member_digest = hashlib.sha256("\0".join(names).encode()).hexdigest()
    if member_digest != expected["member_name_digest"]:
        raise RuntimeError(
            f"{path} member-name digest is {member_digest}; "
            f"expected {expected['member_name_digest']}"
        )
    schema_digest = _chapter05_schema_digest(arrays)
    if schema_digest != expected["schema_digest"]:
        raise RuntimeError(
            f"{path} schema digest is {schema_digest}; "
            f"expected {expected['schema_digest']}"
        )

    exact_scalars = {
        "meta__archive_kind": expected["archive_kind"],
        "meta__archive_schema_version": expected["archive_schema_version"],
        "meta__archive_key_count": expected["key_count"],
        "meta__archive_schema_digest": expected["schema_digest"],
        "meta__payne_zero_commit": PAYNE_ZERO_COMMIT,
        "meta__publisher_sha256": CHAPTER05_PUBLISHER_SHA256,
        "meta__worker_sha256": CHAPTER05_WORKER_SHA256,
        "meta__capture_contract_sha256": CHAPTER05_CAPTURE_CONTRACT_SHA256,
        "meta__exact_source_contract_sha256": (CHAPTER05_EXACT_SOURCE_CONTRACT_SHA256),
        "meta__publisher_contract_sha256": (CHAPTER05_PUBLISHER_CONTRACT_SHA256),
        "meta__deterministic_npz_sha256": CHAPTER05_DETERMINISTIC_NPZ_SHA256,
        "meta__oracle_acceptance_sha256": (CHAPTER05_ORACLE_ACCEPTANCE_SHA256),
        "meta__fixture_sha256": CHAPTER05_FIXTURE_SHA256,
        "meta__fixture_payload_digest": CHAPTER05_FIXTURE_PAYLOAD_DIGEST,
        "meta__raw_capture_schema_version": CHAPTER05_RAW_SCHEMA_VERSION,
        "meta__raw_capture_key_count": CHAPTER05_RAW_KEY_COUNT,
        "meta__raw_capture_schema_digest": CHAPTER05_RAW_SCHEMA_DIGEST,
        "meta__accepted_physical_payload_fingerprint": (
            CHAPTER05_PHYSICAL_PAYLOAD_FINGERPRINT
        ),
        "meta__accepted_full_capture_fingerprint": (CHAPTER05_FULL_CAPTURE_FINGERPRINT),
        "meta__loaded_pinned_python_source_count": (CHAPTER05_LOADED_SOURCE_COUNT),
        "meta__loaded_pinned_python_manifest_digest": (
            CHAPTER05_LOADED_SOURCE_MANIFEST_DIGEST
        ),
    }
    for member, expected_value in exact_scalars.items():
        actual_value = _chapter05_scalar(arrays, member)
        if actual_value != expected_value:
            raise RuntimeError(
                f"{path}:{member} is {actual_value!r}; expected {expected_value!r}"
            )

    if name == CHAPTER05_INTEGRATION_NAME:
        reader_sha256 = _chapter05_scalar(arrays, "meta__reader_archive_sha256")
        if reader_sha256 != expected["reader_archive_sha256"]:
            raise RuntimeError("integration archive has the wrong reader SHA-256")
        reader_schema = _chapter05_scalar(arrays, "meta__reader_archive_schema_digest")
        if reader_schema != expected["reader_archive_schema_digest"]:
            raise RuntimeError("integration archive has the wrong reader schema digest")
        if not bool(
            _chapter05_scalar(arrays, "meta__logical_raw_capture_coverage_complete")
        ):
            raise RuntimeError("integration archive claims incomplete raw coverage")
        inventory_digest = _chapter05_inventory_mapping_digest(arrays)
        if inventory_digest != expected["inventory_mapping_digest"]:
            raise RuntimeError(
                "integration inventory mapping digest is "
                f"{inventory_digest}; expected {expected['inventory_mapping_digest']}"
            )

    return arrays


def _chapter05_golden_specification(name: str) -> dict[str, object]:
    """Return one complete, reviewed post-publication manifest specification."""

    expected = CHAPTER05_GOLDEN_ARCHIVES[name]
    reader = name == CHAPTER05_READER_NAME
    scope = (
        "Single owner of the compact, interpretable continuum reader state, "
        "component banks, thresholds, synthesis-edge reconstruction, and "
        "pedagogical seam slices."
        if reader
        else (
            "Single owner of the exhaustive continuum integration evidence, "
            "full atmosphere products, deduplicated sampling grids, "
            "counterfactuals, and logical raw-capture ownership inventory."
        )
    )
    specification: dict[str, object] = {
        "path": f"data/golden/payne_zero/chapter05/{name}",
        "role": "golden",
        "source": "/Users/ysting/payne-zero",
        "source_commit": PAYNE_ZERO_COMMIT,
        "requires_optional_full_catalog": False,
        "scope": scope,
        "builder": CHAPTER05_GOLDEN_BUILDER,
        "archive_kind": expected["archive_kind"],
        "archive_schema_version": expected["archive_schema_version"],
        "key_count": expected["key_count"],
        "schema_digest": expected["schema_digest"],
        "member_name_digest": expected["member_name_digest"],
        "sha256": expected["sha256"],
        "bytes": expected["bytes"],
        "publisher_sha256": CHAPTER05_PUBLISHER_SHA256,
        "worker_sha256": CHAPTER05_WORKER_SHA256,
        "capture_contract_sha256": CHAPTER05_CAPTURE_CONTRACT_SHA256,
        "exact_source_contract_sha256": CHAPTER05_EXACT_SOURCE_CONTRACT_SHA256,
        "publisher_contract_sha256": CHAPTER05_PUBLISHER_CONTRACT_SHA256,
        "deterministic_npz_sha256": CHAPTER05_DETERMINISTIC_NPZ_SHA256,
        "oracle_acceptance_sha256": CHAPTER05_ORACLE_ACCEPTANCE_SHA256,
        "publication_acceptance_sha256": (CHAPTER05_PUBLICATION_ACCEPTANCE_SHA256),
        "fixture_sha256": CHAPTER05_FIXTURE_SHA256,
        "fixture_payload_digest": CHAPTER05_FIXTURE_PAYLOAD_DIGEST,
        "raw_capture_schema_version": CHAPTER05_RAW_SCHEMA_VERSION,
        "raw_capture_key_count": CHAPTER05_RAW_KEY_COUNT,
        "raw_capture_schema_digest": CHAPTER05_RAW_SCHEMA_DIGEST,
        "accepted_physical_payload_fingerprint": (
            CHAPTER05_PHYSICAL_PAYLOAD_FINGERPRINT
        ),
        "accepted_full_capture_fingerprint": CHAPTER05_FULL_CAPTURE_FINGERPRINT,
        "loaded_pinned_python_source_count": CHAPTER05_LOADED_SOURCE_COUNT,
        "loaded_pinned_python_manifest_digest": (
            CHAPTER05_LOADED_SOURCE_MANIFEST_DIGEST
        ),
        "capture_policy": "two fresh byte-identical raw and final capture sets",
    }
    if not reader:
        specification.update(
            {
                "reader_archive_sha256": expected["reader_archive_sha256"],
                "reader_archive_schema_digest": (
                    expected["reader_archive_schema_digest"]
                ),
                "inventory_mapping_digest": (expected["inventory_mapping_digest"]),
            }
        )
    return specification


def chapter05_golden_specs(
    archive_root: Path | None = None,
) -> tuple[dict[str, object], ...]:
    """Return two specs only for one complete, byte-exact archive pair.

    ``archive_root`` is a test/review seam: it changes where candidate bytes
    are read, never the canonical paths written into the manifest.
    """

    root = CHAPTER05_GOLDEN_DIRECTORY if archive_root is None else archive_root
    expected_names = tuple(CHAPTER05_GOLDEN_ARCHIVES)
    present = tuple((root / name).is_file() for name in expected_names)
    if not any(present):
        if root.is_dir():
            extras = sorted(path.name for path in root.iterdir())
            if extras:
                raise RuntimeError(
                    f"{root} contains unreviewed Chapter 5 publication files: {extras}"
                )
        return ()
    if not all(present):
        missing = [name for name, exists in zip(expected_names, present) if not exists]
        raise RuntimeError(
            f"partial Chapter 5 publication at {root}; missing {missing}"
        )
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != set(expected_names):
        raise RuntimeError(
            f"{root} must contain exactly {list(expected_names)}; "
            f"found {sorted(actual_names)}"
        )

    _validate_chapter05_repository_identities()
    arrays_by_name = {
        name: _validate_chapter05_archive(root / name, name) for name in expected_names
    }
    integration_reader_sha = _chapter05_scalar(
        arrays_by_name[CHAPTER05_INTEGRATION_NAME],
        "meta__reader_archive_sha256",
    )
    if integration_reader_sha != _sha256_path(root / CHAPTER05_READER_NAME):
        raise RuntimeError("integration-to-reader byte binding does not resolve")

    approved_members = {
        name: frozenset(arrays) for name, arrays in arrays_by_name.items()
    }
    previous_members = dict(_VALIDATED_CHAPTER05_MEMBERS)
    _VALIDATED_CHAPTER05_MEMBERS.update(approved_members)
    try:
        # Exercise every unit/axes/ownership route before a spec can reach main().
        for name, arrays in arrays_by_name.items():
            relative_path = f"data/golden/payne_zero/chapter05/{name}"
            for member, values in arrays.items():
                chapter05_golden_member_metadata(
                    relative_path,
                    member,
                    tuple(values.shape),
                )
    except Exception:
        _VALIDATED_CHAPTER05_MEMBERS.clear()
        _VALIDATED_CHAPTER05_MEMBERS.update(previous_members)
        raise
    return tuple(_chapter05_golden_specification(name) for name in expected_names)


TRANSFER_INPUT_UNITS = {
    "column_mass": "g cm^-2",
    "continuum_absorption_slab": "cm^2 g^-1",
    "continuum_scattering_slab": "cm^2 g^-1",
    "continuum_source_slab": "controlled source-function units (fixture)",
    "effective_temperature": "K",
    "frequency_hz": "Hz",
    "frequency_weights": "normalized quadrature weight",
    "h_over_kt": "s",
    "line_mass_absorption_coefficient_slab": "cm^2 g^-1",
    "planck_all": "controlled specific-intensity units (fixture)",
    "stimulated_all": "dimensionless",
    "target_integrated_eddington_flux": "controlled integrated-flux units (fixture)",
    "temperature": "K",
}

TRANSFER_TABLE_UNITS = {
    "__meta__": "UTF-8 JSON provenance bytes",
    "eddington_flux_operator": "dimensionless transfer operator",
    "mean_intensity_operator": "dimensionless transfer operator",
    "reference_column_mass": "g cm^-2",
    "second_moment_weights": "dimensionless quadrature weight",
    "surface_eddington_flux_weights": "dimensionless quadrature weight",
    "transfer_optical_depth_grid": "dimensionless",
}

GOLDEN_IDENTITY_UNITS = {
    "fixture_sha256": "SHA-256 hexadecimal identity",
    "payne_zero_commit": "Git commit identity",
    "transfer_tables_sha256": "SHA-256 hexadecimal identity",
}

CHAPTER03_SYNTHESIS_FIXTURE_UNITS = {
    "temperature": "K",
    "thermal_energy_ev": "eV",
    "thermal_energy_erg": "erg",
    "hc_over_kt": "cm",
    "natural_log_temperature": "natural logarithm of numerical temperature in K",
    "gas_pressure": "dyn cm^-2",
    "electron_density": "cm^-3",
    "total_nuclei_number_density": "cm^-3",
    "elemental_abundances": "linear relative number abundance",
    "ion_stage_count": "integer stage count",
    "ground_partition_table": (
        "dimensionless depth-specific loaded field; inactive in current "
        "runtime ground correction"
    ),
}

CHAPTER03_ATOM_ONLY_FIXTURE_UNITS = {
    "column_mass": "g cm^-2",
    "electron_density_seed": "cm^-3",
    "elemental_abundances": "linear relative number abundance",
    "gas_pressure": "dyn cm^-2",
    "microturbulence": "cm s^-1",
    "saha_atomic_number": "atomic number",
    "saha_charge_square_density": "cm^-3, charge-square weighted",
    "saha_electron_density": "cm^-3",
    "saha_gas_pressure": "dyn cm^-2",
    "saha_ion_stage_count": "integer requested stage count",
    "saha_population_mode": "source-native integer mode code",
    "saha_temperature": "K",
    "saha_total_nuclei_number_density": "cm^-3",
    "source_depth_indices": "zero-based source depth index",
    "temperature": "K",
}

CHAPTER04_MOLECULAR_FIXTURE_UNITS = {
    "atmosphere_h2_boundary_temperature": "K",
    "atmosphere_polynomial_boundary_temperature": "K",
    "column_mass": "g cm^-2",
    "electron_density_seed": "cm^-3; controlled upstream seed",
    "elemental_abundances": "linear relative number abundance",
    "gas_pressure": "dyn cm^-2",
    "microturbulence": "cm s^-1",
    "named_molecule_codes": "source-native base-100 molecule code",
    "pressure_control_gas_pressure": "dyn cm^-2",
    "pressure_control_temperature": "K",
    "source_abundance_fixture_sha256": "SHA-256 hexadecimal identity",
    "synthesis_named_h2_boundary_temperature": "K",
    "synthesis_polynomial_boundary_temperature": "K",
    "temperature": "K",
    "temperature_control_gas_pressure": "dyn cm^-2",
    "temperature_control_temperature": "K",
}

CHAPTER06_FE_SUBSET_RAW_FIELDS = {
    "stored_wavelength_nm",
    "raw_log_oscillator_strength",
    "species_code",
    "first_energy_column_cm",
    "second_energy_column_cm",
    "radiative_damping_log",
    "stark_damping_log",
    "van_der_waals_damping_log",
    "lower_principal_quantum_number",
    "upper_principal_quantum_number",
    "primary_isotope_number",
    "primary_isotope_log_correction",
    "secondary_isotope_log_correction",
    "energy_shift_field",
    "isotope_shift_units",
    "line_size",
    "line_category_tag",
}

CHAPTER06_FE_SUBSET_UNITS = {
    "builder_command": "portable regeneration command",
    "energy_shift_field": (
        "two fixed-width 5-byte energy-shift fields; parsed in 1e-3 cm^-1"
    ),
    "first_energy_column_cm": "cm^-1",
    "isotope_shift_units": "source units of 1e-4 nm",
    "line_category_tag": "fixed-width source line-category tag",
    "line_size": "source-native integer line-size routing code",
    "lower_principal_quantum_number": (
        "source-native lower principal-quantum-number field"
    ),
    "payne_zero_commit": "Git commit identity",
    "primary_isotope_log_correction": "dex correction to log10(gf)",
    "primary_isotope_number": "source-native isotope number",
    "radiative_damping_log": "log10 of numerical radiative damping in s^-1",
    "raw_log_oscillator_strength": "log10(gf)",
    "second_energy_column_cm": "cm^-1",
    "secondary_isotope_log_correction": "dex correction to log10(gf)",
    "source_archive_bytes": "source archive byte count",
    "source_archive_relative_path": "source archive relative path identity",
    "source_archive_row_count": "source archive row count",
    "source_archive_sha256": "SHA-256 hexadecimal identity",
    "source_field_count": "raw source field count",
    "source_row_index": "zero-based source archive row index",
    "species_code": "source-native atomic species and ion-stage code",
    "stark_damping_log": (
        "log10 of numerical Stark coefficient in cm^3 s^-1"
    ),
    "stored_wavelength_nm": "nm",
    "subset_role": "data-role declaration",
    "subset_schema_version": "subset schema version",
    "upper_principal_quantum_number": (
        "source-native upper principal-quantum-number field"
    ),
    "van_der_waals_damping_log": (
        "log10 of numerical van der Waals coefficient in cm^3 s^-1"
    ),
}

CONTINUUM_EDGE_GRID_UNITS = {
    "__meta__": "UTF-8 JSON provenance bytes",
    "continuum_edge_interval_width_squared_over_two_nm2": "nm^2",
    "continuum_edge_midpoint_wavelength_nm": "nm",
    "continuum_edge_sample_frequency_hz": "Hz",
    "continuum_edge_wavelength_nm": "nm",
    "continuum_edge_wavenumber_cm": "cm^-1; sign is interpolation metadata",
    "signed_continuum_edge_frequency_hz": ("Hz; sign is interpolation metadata"),
}

CONTINUUM_TABLE_UNITS = {
    "karzas_latter_log10_frequency_hz": "log10 of numerical frequency in Hz",
    "karzas_latter_total_log10_cross_section_cm2": (
        "log10 of numerical cross section in cm^2"
    ),
    "karzas_latter_angular_log10_cross_section_cm2": (
        "log10 of numerical cross section in cm^2"
    ),
    "karzas_latter_high_level_energy_offset_rydberg": "Rydberg",
    "hminus_boundfree_wavelength_nm": "nm",
    "hminus_boundfree_cross_section_cm2": (
        "stored numerical units of 1e-18 cm^2; consumers multiply by 1e-18"
    ),
    "hminus_freefree_inverse_wavelength_grid": (
        "source-native inverse-wavelength coordinate"
    ),
    "hminus_freefree_theta_grid": "dimensionless theta = 5040/T",
    "hminus_freefree_short_wavelength_table": (
        "source-native H-minus free-free coefficient"
    ),
    "hminus_freefree_long_wavelength_table": (
        "source-native H-minus free-free coefficient"
    ),
    "hydrogen_rayleigh_gavrila_main_table": (
        "source-native Gavrila polarizability coefficient"
    ),
    "hydrogen_rayleigh_gavrila_ab_table": (
        "source-native Gavrila polarizability coefficient"
    ),
    "hydrogen_rayleigh_gavrila_bc_table": (
        "source-native Gavrila polarizability coefficient"
    ),
    "hydrogen_rayleigh_gavrila_cd_table": (
        "source-native Gavrila polarizability coefficient"
    ),
    "hydrogen_rayleigh_gavrila_lyman_continuum_table": (
        "source-native Gavrila polarizability coefficient"
    ),
    "hydrogen_rayleigh_gavrila_lyman_frequency_ratio_grid": (
        "dimensionless frequency ratio"
    ),
    "coulomb_freefree_charge_log_offset": ("source-native logarithmic charge offset"),
    "coulomb_freefree_gaunt_table": "dimensionless Gaunt factor",
    "hot_metal_boundfree_transition_table": (
        "source-native mixed-unit transition records"
    ),
    "silicon_singly_ionized_peach_cross_section_table": (
        "natural logarithm of source-native Si II cross section"
    ),
    "silicon_singly_ionized_peach_threshold_frequencies_hz": "Hz",
    "silicon_singly_ionized_peach_natural_log_frequency_grid": (
        "natural logarithm of numerical frequency in Hz"
    ),
    "silicon_singly_ionized_peach_natural_log_temperature_grid": (
        "natural logarithm of numerical temperature in K"
    ),
    "ch_partition_table": "dimensionless partition function",
    "oh_partition_table": "dimensionless partition function",
    "ch_cross_section_table": "log10 of source-native CH cross section",
    "oh_cross_section_table": "log10 of source-native OH cross section",
    "hydrogen_molecule_h2_collision_table": (
        "log10 of source-native H2-H2 binary absorption coefficient"
    ),
    "hydrogen_molecule_he_collision_table": (
        "log10 of source-native H2-He binary absorption coefficient"
    ),
    "hydrogen_neutral_level_energy_cm": "cm^-1",
    "hydrogen_neutral_level_statistical_weight": "dimensionless",
}

CHAPTER05_SYNTHESIS_FIELD_UNITS = {
    "atmosphere_schema_version": "schema version",
    "temperature": "K",
    "column_mass": "g cm^-2",
    "gas_pressure": "dyn cm^-2",
    "electron_density": "cm^-3",
    "mass_density": "g cm^-3",
    "microturbulence": "cm s^-1",
    "hc_over_kt": "cm",
    "hydrogen_neutral_population": "cm^-3",
    "hydrogen_ionized_population": "cm^-3",
    "hydrogen_partition_normalized_ion_stage_populations": (
        "cm^-3 per partition function"
    ),
    "helium_neutral_population": "cm^-3",
    "helium_singly_ionized_population": "cm^-3",
    "molecular_hydrogen_population": "cm^-3",
    "carbon_partition_normalized_ion_stage_populations": (
        "cm^-3 per partition function"
    ),
    "magnesium_neutral_partition_normalized_population": (
        "cm^-3 per partition function"
    ),
    "aluminum_neutral_partition_normalized_population": (
        "cm^-3 per partition function"
    ),
    "silicon_neutral_partition_normalized_population": ("cm^-3 per partition function"),
    "iron_neutral_partition_normalized_population": ("cm^-3 per partition function"),
    "partition_normalized_populations": "cm^-3 per partition function",
    "ion_stage_populations": "cm^-3",
    "fractional_doppler_widths": "dimensionless v_D/c",
    "elemental_abundances": "linear relative number abundance",
    **CONTINUUM_EDGE_GRID_UNITS,
}

CHAPTER05_ATMOSPHERE_FIELD_UNITS = {
    "temperature": "K",
    "mass_density": "g cm^-3",
    "electron_density": "cm^-3",
    "gas_pressure": "dyn cm^-2",
    "hydrogen_partition_normalized_ion_stage_populations": (
        "cm^-3 per partition function"
    ),
    "hydrogen_neutral_population": "cm^-3",
    "hydrogen_ionized_population": "cm^-3",
    "helium_neutral_population": "cm^-3",
    "helium_singly_ionized_population": "cm^-3",
    "helium_neutral_partition_normalized_population": ("cm^-3 per partition function"),
    "helium_singly_ionized_partition_normalized_population": (
        "cm^-3 per partition function"
    ),
    "elemental_abundances_by_layer": "linear relative number abundance",
    "hydrogen_departure_coefficients": "dimensionless departure coefficient",
    "microturbulence": "cm s^-1",
    "ion_stage_populations_by_packed_slot": "cm^-3",
    "partition_normalized_populations_by_packed_slot": ("cm^-3 per partition function"),
    "ch_population": "cm^-3 per partition function",
    "oh_population": "cm^-3 per partition function",
}


def chapter05_continuum_fixture_unit(name: str) -> str:
    """Return one exact Chapter 5 stage-input representation."""

    direct = {
        "payne_zero_commit": "Git commit identity",
        "regime_names": "controlled stellar-regime labels",
        "source_chapter04_fixture_sha256": "SHA-256 hexadecimal identity",
        "source_continuum_edge_grid_sha256": "SHA-256 hexadecimal identity",
        "atmosphere_runner_opacity_flags": "source-native integer IFOP flag",
        "hminus_edge_frequency_hz": "Hz",
        "h2_policy_temperature_k": "K",
        "atmosphere_grid_boundary_temperature_k": "K",
        "synthesis_edge_probe_wavelength_nm": "nm",
    }
    if name in direct:
        return direct[name]
    if name.endswith("__effective_temperature"):
        return "K"
    _, owner, field = name.split("__", maxsplit=2)
    if owner == "input":
        return {
            "column_mass": "g cm^-2",
            "temperature": "K",
            "gas_pressure": "dyn cm^-2",
            "electron_density_seed": "cm^-3; controlled upstream seed",
            "microturbulence": "cm s^-1",
            "elemental_abundances": "linear relative number abundance",
        }[field]
    if owner == "atmosphere":
        return CHAPTER05_ATMOSPHERE_FIELD_UNITS[field]
    if owner == "synthesis":
        return CHAPTER05_SYNTHESIS_FIELD_UNITS[field]
    raise KeyError(name)


def molecular_catalog_unit(name: str) -> str:
    """Return one exact molecular-catalog field representation."""

    return {
        "molecule_count": "active molecule-record count",
        "equation_count": "active coupled-equation count",
        "component_count": "active component-entry count",
        "molecule_codes": "source-native base-100 molecule code",
        "equilibrium_coefficients": (
            "source-native mixed-unit equilibrium coefficients"
        ),
        "component_start_indices": "zero-based ragged component offset",
        "component_equation_indices": "zero-based equation index",
        "equation_species_codes": "source-native equation species code",
        "species_to_equation_index": "zero-based equation index lookup",
    }[name]


def atmosphere_eos_unit(name: str) -> str:
    """Return the physical representation for one atmosphere EOS table."""

    if name.endswith("_level_energy_cm") or name == "ionization_potential_cm":
        return "cm^-1"
    if name.endswith("_level_statistical_weight"):
        return "dimensionless statistical weight"
    return {
        "element_block_offsets": "one-based packed table index",
        "iron_group_partition_grid": "dimensionless partition function",
        "packed_level_metadata": "source-native packed integer partition metadata",
        "major_isotope_mass_amu": "amu",
        "isotope_records": (
            "source-native packed isotope records with mixed fields; "
            "not consumed by Chapter 3"
        ),
    }[name]


def synthesis_eos_unit(name: str) -> str:
    """Return the physical representation for one synthesis EOS table."""

    if name.endswith("_level_energy_cm") or name == "ionization_potential_cm":
        return "cm^-1"
    if name.endswith("_level_statistical_weight"):
        return "dimensionless statistical weight"
    return {
        "packed_partition_table": "source-native packed integer partition metadata",
        "iron_group_partition_grid": "dimensionless partition function",
        "iron_group_lower_potential_grid": "cm^-1",
        "iron_group_lower_potential_log_grid": "log10(cm^-1)",
        "element_block_offsets": "one-based packed table index",
        "partition_interpolation_scale": "dimensionless interpolation scale",
    }[name]


def chapter03_identity_unit(name: str) -> str | None:
    """Return a provenance unit for metadata embedded in Chapter 3 goldens."""

    if name == "payne_zero_commit":
        return "Git commit identity"
    if name.endswith("_sha256"):
        return "SHA-256 hexadecimal identity"
    if name.endswith("_version"):
        return "software version identity"
    if name == "numba_thread_count":
        return "thread count"
    if name == "numba_cache_state":
        return "cache-state description"
    if name == "torch_device":
        return "Torch device policy"
    if name == "torch_dtype":
        return "Torch dtype policy"
    return None


def chapter03_golden_unit(path: str, name: str) -> str:
    """Return one physical or interface representation in a Chapter 3 golden."""

    identity = chapter03_identity_unit(name)
    if identity is not None:
        return identity
    if path.endswith("chapter03_atmosphere_saha_outputs.npz"):
        if "_mode11_" in name:
            return "dimensionless ion-stage fraction per partition function"
        if "_mode12_" in name:
            return "dimensionless ion-stage fraction"
        if "_mode13_" in name:
            return "dimensionless partition function"
    if name.startswith("sentinel_"):
        return "exact interface sentinel; not a physical quantity"

    base_name = name.removeprefix("scalar_depth_")
    base_name = base_name.removeprefix("full_")
    base_name = base_name.removeprefix("fixed_")
    if base_name in {
        "electron_density",
        "fixed_input_electron_density",
        "input_electron_density",
        "total_nuclei_number_density",
        "charge_square_density",
        "hydrogen_neutral_population",
        "hydrogen_ionized_population",
        "helium_neutral_population",
        "helium_singly_ionized_population",
        "ion_stage_populations",
        "ion_stage_populations_by_packed_slot",
    }:
        return "cm^-3"
    if base_name in {
        "carbon_partition_normalized_ion_stage_populations",
        "hydrogen_partition_normalized_ion_stage_populations",
        "magnesium_neutral_partition_normalized_population",
        "aluminum_neutral_partition_normalized_population",
        "silicon_neutral_partition_normalized_population",
        "iron_neutral_partition_normalized_population",
        "partition_normalized_populations",
        "partition_normalized_populations_by_packed_slot",
    }:
        return "cm^-3 per partition function"
    if base_name == "mass_density":
        return "g cm^-3"
    if base_name == "atomic_specific_internal_energy":
        return "erg g^-1"
    if base_name == "fractional_doppler_widths":
        return "dimensionless v/c"
    if base_name == (
        "partition_normalized_population_over_mass_density_and_fractional_doppler_width"
    ):
        return "g^-1"
    if base_name == "eos_partition_functions":
        return "dimensionless partition function"
    if base_name == "eos_ion_stage_fractions_over_partition":
        return "dimensionless ion-stage fraction per partition function"
    raise KeyError(f"no Chapter 3 golden unit for {path}: {name}")


CHAPTER04_PUBLISHER_META_UNITS = {
    "meta__archive_kind": "publisher archive-kind identity",
    "meta__archive_schema_version": "publisher archive schema version",
    "meta__atmosphere_worker_sha256": "SHA-256 hexadecimal identity",
    "meta__capture_contract_sha256": "SHA-256 hexadecimal identity",
    "meta__constants_archive_sha256": "SHA-256 hexadecimal identity",
    "meta__deterministic_npz_sha256": "SHA-256 hexadecimal identity",
    "meta__fixture_sha256": "SHA-256 hexadecimal identity",
    "meta__oracle_acceptance_sha256": "SHA-256 hexadecimal identity",
    "meta__payne_zero_commit": "Git commit identity",
    "meta__publisher_sha256": "SHA-256 hexadecimal identity",
    "meta__synthesis_worker_sha256": "SHA-256 hexadecimal identity",
}

CHAPTER04_ORACLE_ENVIRONMENT_UNITS = {
    "cpu_only": "boolean CPU-only execution policy",
    "lc_all": "process locale setting",
    "mkl_dynamic": "MKL dynamic-threading environment setting",
    "mkl_num_threads": "MKL thread-count environment setting",
    "numba_cache_policy": "Numba cache-isolation policy",
    "numba_num_threads": "Numba thread-count environment setting",
    "numexpr_num_threads": "NumExpr thread-count environment setting",
    "numpy_float_dtype": "NumPy floating-point dtype policy",
    "omp_num_threads": "OpenMP thread-count environment setting",
    "openblas_num_threads": "OpenBLAS thread-count environment setting",
    "pythondontwritebytecode": "Python bytecode-write environment setting",
    "pythonhashseed": "Python hash-seed environment setting",
    "pythonnousersite": "Python user-site isolation environment setting",
    "tz": "process timezone setting",
    "veclib_maximum_threads": "Accelerate thread-count environment setting",
}

CHAPTER04_ORACLE_SHA256_NAMES = {
    "asset_atmosphere_tables_ionization_potential_tables_sha256",
    "asset_atmosphere_tables_iron_group_partition_tables_sha256",
    "asset_atmosphere_tables_isotope_tables_sha256",
    "asset_atmosphere_tables_molecular_equilibrium_tables_sha256",
    "asset_atmosphere_tables_packed_level_metadata_sha256",
    "asset_atmosphere_tables_special_partition_tables_sha256",
    "asset_source_catalogs_lines_molecular_equilibrium_atmosphere_sha256",
    "executed_atmosphere_source_sha256",
    "executed_source_sha256",
    "fixture_sha256",
    "source_atmosphere_io_sha256",
    "source_config_sha256",
    "source_equation_of_state_sha256",
    "source_molecular_data_sha256",
    "source_molecular_equilibrium_sha256",
    "source_population_layout_sha256",
    "source_run_setup_sha256",
    "source_runner_sha256",
    "source_runtime_state_sha256",
    "source_sha256__atmosphere_molecule_catalog",
    "source_sha256__atmosphere_schema",
    "source_sha256__atomic_masses",
    "source_sha256__continuum_edge_grid",
    "source_sha256__eos_tables",
    "source_sha256__equation_of_state_source",
    "source_sha256__molecular_equilibrium_source",
    "source_sha256__molecule_catalog",
    "source_sha256__pipeline_source",
    "source_synthesis_bridge_sha256",
}

CHAPTER04_ATMOSPHERE_ORACLE_IDENTITY_NAMES = {
    "asset_atmosphere_tables_ionization_potential_tables_sha256",
    "asset_atmosphere_tables_iron_group_partition_tables_sha256",
    "asset_atmosphere_tables_isotope_tables_sha256",
    "asset_atmosphere_tables_molecular_equilibrium_tables_sha256",
    "asset_atmosphere_tables_packed_level_metadata_sha256",
    "asset_atmosphere_tables_special_partition_tables_sha256",
    "asset_source_catalogs_lines_molecular_equilibrium_atmosphere_sha256",
    "blas_name",
    "blas_version",
    "environment_cpu_only",
    "environment_lc_all",
    "environment_mkl_dynamic",
    "environment_mkl_num_threads",
    "environment_numba_cache_policy",
    "environment_numba_num_threads",
    "environment_numexpr_num_threads",
    "environment_numpy_float_dtype",
    "environment_omp_num_threads",
    "environment_openblas_num_threads",
    "environment_pythondontwritebytecode",
    "environment_pythonhashseed",
    "environment_pythonnousersite",
    "environment_tz",
    "environment_veclib_maximum_threads",
    "executed_atmosphere_source_count",
    "executed_atmosphere_source_module_names",
    "executed_atmosphere_source_relative_paths",
    "executed_atmosphere_source_sha256",
    "lapack_name",
    "lapack_version",
    "numba_thread_count",
    "numba_version",
    "numpy_build_dependencies_json",
    "numpy_version",
    "platform",
    "platform_machine",
    "platform_release",
    "platform_system",
    "python_executable",
    "python_implementation",
    "python_version",
    "source_atmosphere_io_sha256",
    "source_config_sha256",
    "source_equation_of_state_sha256",
    "source_molecular_data_sha256",
    "source_molecular_equilibrium_sha256",
    "source_population_layout_sha256",
    "source_run_setup_sha256",
    "source_runner_sha256",
    "source_runtime_state_sha256",
    "source_synthesis_bridge_sha256",
    "system_byteorder",
}

CHAPTER04_SYNTHESIS_ORACLE_IDENTITY_NAMES = {
    "meta__blas_name",
    "meta__blas_version",
    "meta__device",
    "meta__dtype",
    "meta__environment__LC_ALL",
    "meta__environment__MKL_DYNAMIC",
    "meta__environment__MKL_NUM_THREADS",
    "meta__environment__NUMBA_NUM_THREADS",
    "meta__environment__NUMEXPR_NUM_THREADS",
    "meta__environment__OMP_NUM_THREADS",
    "meta__environment__OPENBLAS_NUM_THREADS",
    "meta__environment__PYTHONDONTWRITEBYTECODE",
    "meta__environment__PYTHONHASHSEED",
    "meta__environment__PYTHONNOUSERSITE",
    "meta__environment__TZ",
    "meta__environment__VECLIB_MAXIMUM_THREADS",
    "meta__executed_source_count",
    "meta__executed_source_module_names",
    "meta__executed_source_relative_paths",
    "meta__executed_source_sha256",
    "meta__fixture_sha256",
    "meta__lapack_name",
    "meta__lapack_version",
    "meta__numpy_version",
    "meta__payne_zero_commit",
    "meta__platform",
    "meta__python_version",
    "meta__source_sha256__atmosphere_molecule_catalog",
    "meta__source_sha256__atmosphere_schema",
    "meta__source_sha256__atomic_masses",
    "meta__source_sha256__continuum_edge_grid",
    "meta__source_sha256__eos_tables",
    "meta__source_sha256__equation_of_state_source",
    "meta__source_sha256__molecular_equilibrium_source",
    "meta__source_sha256__molecule_catalog",
    "meta__source_sha256__pipeline_source",
    "meta__system_byteorder",
    "meta__torch_num_interop_threads",
    "meta__torch_num_threads",
    "meta__torch_version",
}


def chapter04_oracle_identity_unit(name: str) -> str:
    """Return one explicitly scoped execution-provenance representation."""

    lowered = name.lower()
    if lowered.startswith("meta__"):
        lowered = lowered.removeprefix("meta__")
    if lowered.startswith("environment__"):
        key = lowered.removeprefix("environment__")
        return CHAPTER04_ORACLE_ENVIRONMENT_UNITS[key]
    if lowered.startswith("environment_"):
        key = lowered.removeprefix("environment_")
        return CHAPTER04_ORACLE_ENVIRONMENT_UNITS[key]
    if lowered in CHAPTER04_ORACLE_SHA256_NAMES:
        return "SHA-256 hexadecimal identity"
    return {
        "blas_name": "BLAS implementation identity",
        "blas_version": "BLAS version identity",
        "device": "Torch execution-device policy",
        "dtype": "Torch floating-point dtype policy",
        "executed_atmosphere_source_count": ("executed atmosphere source-module count"),
        "executed_atmosphere_source_module_names": (
            "Python module-name identities in execution order"
        ),
        "executed_atmosphere_source_relative_paths": (
            "repository-relative source paths in execution order"
        ),
        "executed_source_count": "executed synthesis source-module count",
        "executed_source_module_names": (
            "Python module-name identities in execution order"
        ),
        "executed_source_relative_paths": (
            "repository-relative source paths in execution order"
        ),
        "fixture_sha256": "SHA-256 hexadecimal identity",
        "lapack_name": "LAPACK implementation identity",
        "lapack_version": "LAPACK version identity",
        "numba_thread_count": "Numba thread count",
        "numba_version": "Numba version identity",
        "numpy_build_dependencies_json": ("UTF-8 JSON NumPy build-dependency identity"),
        "numpy_version": "NumPy version identity",
        "payne_zero_commit": "Git commit identity",
        "platform": "operating-system platform identity",
        "platform_machine": "machine-architecture identity",
        "platform_release": "operating-system release identity",
        "platform_system": "operating-system family identity",
        "python_executable": "Python executable path identity",
        "python_implementation": "Python implementation identity",
        "python_version": "Python version identity",
        "system_byteorder": "machine byte-order identity",
        "torch_num_interop_threads": "Torch inter-op thread count",
        "torch_num_threads": "Torch intra-op thread count",
        "torch_version": "Torch version identity",
    }[lowered]


CHAPTER04_ATMOSPHERE_PHYSICAL_DENSITY_FIELDS = {
    "actual_seed_override",
    "charge_square_density",
    "equation_densities_before_normalization",
    "input_charge_square_density",
    "input_electron_density",
    "ion_stage_populations_by_packed_slot",
    "layer_seed_equation_densities",
    "molecular_equation_densities",
    "molecular_equation_densities_after",
    "molecular_populations",
    "newton_preupdate_equation_densities",
    "newton_raw_corrections",
    "ordinary_continuation_seed",
    "output_electron_density",
    "physical_equation_densities",
    "physical_equation_densities_saved_before_fill",
    "previous_equation_densities_after",
    "previous_equation_densities_before",
    "previous_molecular_equation_densities",
    "raw_named_populations",
    "raw_populations_after",
    "raw_populations_before_fill",
    "saved_physical_rows_after",
    "saved_physical_rows_before",
    "saved_physical_rows_input",
    "total_nuclei_number_density",
}
CHAPTER04_ATMOSPHERE_NORMALIZED_DENSITY_FIELDS = {
    "normalized_named_populations",
    "normalized_populations_after",
    "normalized_populations_after_fill",
    "normalized_populations_before",
    "normalized_populations_before_fill",
    "partition_normalized_molecular_populations",
    "partition_normalized_populations_by_packed_slot",
}
CHAPTER04_ATMOSPHERE_TRANSFORMED_FIELDS = {
    "equation_densities_after_normalization",
    "transformed_equation_densities",
    "transformed_molecular_equation_densities",
}
CHAPTER04_ATMOSPHERE_CONSTANT_FIELDS = {
    "frozen_newton_constants",
    "population_constants",
}
CHAPTER04_ATMOSPHERE_COUNT_FIELDS = {
    "fill_call_count",
    "layer_call_count",
    "newton_iteration_count",
    "newton_np_linalg_direct_solve_branch_count",
    "newton_np_linalg_lstsq_call_count",
    "newton_np_linalg_lstsq_call_count_by_layer",
    "newton_np_linalg_solve_call_count",
    "newton_np_linalg_solve_call_count_by_layer",
    "solve_call_count",
    "temperature_iteration_cache_key_count",
}
CHAPTER04_ATMOSPHERE_INDEX_FIELDS = {
    "fractional_doppler_width_structural_infinity_slots",
    "layer_order",
    "newton_history_iteration_index",
    "newton_history_layer_index",
    "newton_history_layer_offsets",
}
CHAPTER04_ATMOSPHERE_MASK_FIELDS = {
    "fractional_doppler_width_structural_infinity_mask",
    "newton_converged",
    "newton_exhausted",
    "newton_linear_direct_solve_branch_mask",
    "newton_linear_lstsq_branch_mask",
    "newton_still_iterating_history",
    "pressure_iteration_enabled",
    "specific_internal_energy_mode_enabled",
    "temperature_iteration_cache_pops_itemp_present",
}


def chapter04_atmosphere_state_unit(name: str) -> str:
    """Return one exact atmosphere-route field representation."""

    if name in CHAPTER04_ATMOSPHERE_PHYSICAL_DENSITY_FIELDS:
        if "charge_square_density" in name:
            return "cm^-3, charge-square weighted"
        return "cm^-3"
    if name in CHAPTER04_ATMOSPHERE_NORMALIZED_DENSITY_FIELDS:
        return "cm^-3 per partition function"
    if name in CHAPTER04_ATMOSPHERE_TRANSFORMED_FIELDS:
        return (
            "source-native species-dependent N/U equation factor after "
            "partition and thermal normalization; not cm^-3"
        )
    if name in CHAPTER04_ATMOSPHERE_CONSTANT_FIELDS:
        return (
            "source-native molecule-dependent equilibrium factor with "
            "stoichiometry-dependent density powers"
        )
    if name in CHAPTER04_ATMOSPHERE_COUNT_FIELDS:
        return "integer call or iteration count"
    if name in CHAPTER04_ATMOSPHERE_INDEX_FIELDS:
        return "zero-based index or ragged history offset"
    if name in CHAPTER04_ATMOSPHERE_MASK_FIELDS:
        return "boolean branch or state mask"
    return {
        "column_mass": "g cm^-2",
        "direct_specific_internal_energy_reference": "erg g^-1",
        "fractional_doppler_widths": "dimensionless v/c",
        "gas_pressure": "dyn cm^-2",
        "major_isotope_mass_amu": "amu",
        "mass_density": "g cm^-3",
        "newton_relative_update_max_history": (
            "dimensionless maximum relative Newton update"
        ),
        "partition_normalized_population_over_mass_density_and_width": "g^-1",
        "population_mode": "source-native integer population-mode code",
        "postsolve_residual": "cm^-3 molecular conservation residual",
        "postsolve_row_scaled_residual": (
            "dimensionless row-scaled molecular conservation residual"
        ),
        "postsolve_row_scaled_residual_max": (
            "dimensionless maximum row-scaled residual"
        ),
        "specific_internal_energy": "erg g^-1",
        "specific_internal_energy_after": "erg g^-1",
        "specific_internal_energy_before": "erg g^-1",
        "stopping_relative_update_max": (
            "dimensionless maximum relative Newton update"
        ),
        "temperature": "K",
        "temperature_iteration_cache_value": (
            "source-native integer temperature-iteration cache key"
        ),
    }[name]


def chapter04_atmosphere_archive_unit(name: str) -> str:
    """Return one atmosphere golden member representation."""

    if name in CHAPTER04_PUBLISHER_META_UNITS:
        return CHAPTER04_PUBLISHER_META_UNITS[name]
    if name.startswith("oracle__"):
        field = name.removeprefix("oracle__")
        if field not in CHAPTER04_ATMOSPHERE_ORACLE_IDENTITY_NAMES:
            raise KeyError(f"unknown Chapter 4 atmosphere oracle identity {field}")
        return chapter04_oracle_identity_unit(field)
    for prefix in (
        "disabled_",
        "energy_",
        "full_",
        "handoff_",
        "mode12_",
        "mode2_",
    ):
        if name.startswith(prefix):
            return chapter04_atmosphere_state_unit(name.removeprefix(prefix))
    for prefix in ("pressure_control_", "temperature_control_"):
        if name.startswith(prefix):
            return chapter04_atmosphere_state_unit(name.removeprefix(prefix))
    if name.startswith("bridge_"):
        field = name.removeprefix("bridge_")
        return {
            "active_molecular_population_shape": (
                "ordered array-axis lengths (depth, active molecule)"
            ),
            "h2_all_zero_catalog_input": "cm^-3",
            "h2_all_zero_output": "cm^-3",
            "h2_mixed_catalog_input": "cm^-3",
            "h2_mixed_output": "cm^-3",
            "h2_no_catalog_output": "cm^-3",
            "h2_packed_neutral_hydrogen": "cm^-3",
            "h2_temperature": "K",
            "live_shape_error_message": "captured exception message",
            "live_shape_error_type": "Python exception-class identity",
            "padded_molecule_code_shape": (
                "ordered array-axis lengths (padded molecule catalog)"
            ),
        }[field]
    if name.startswith("priming_cache_"):
        field = name.removeprefix("priming_cache_")
        return {
            "additional_solve_count_during_schedule": "integer solve-call count",
            "cache_value_after_priming": (
                "source-native integer temperature-iteration cache key"
            ),
            "cache_value_after_schedule": (
                "source-native integer temperature-iteration cache key"
            ),
            "cache_value_before_priming": (
                "source-native integer temperature-iteration cache key"
            ),
            "cache_value_before_schedule": (
                "source-native integer temperature-iteration cache key"
            ),
            "job_count_during_schedule": "integer scheduled-job count",
            "molecular_solve_count_during_zero_code": "integer solve-call count",
            "populate_all_call_count": "integer populate-all call count",
            "schedule_job_cache_value_after": (
                "source-native integer temperature-iteration cache key"
            ),
            "schedule_job_cache_value_before": (
                "source-native integer temperature-iteration cache key"
            ),
            "schedule_job_solve_count_after": "cumulative integer solve-call count",
            "schedule_job_solve_count_before": "cumulative integer solve-call count",
            "schedule_job_temperature_iteration_index": (
                "zero-based temperature-iteration index"
            ),
            "schedule_solve_count_after": "cumulative integer solve-call count",
            "schedule_solve_count_before": "cumulative integer solve-call count",
            "temperature_iteration_index": "zero-based temperature-iteration index",
            "zero_code": "source-native zero molecular-population request code",
            "zero_code_call_count": "integer zero-code call count",
            "zero_code_mode": "source-native integer population-mode code",
            "zero_code_output_shape": "ordered output array-axis lengths",
        }[field]
    if name.startswith("schedule_inventory_"):
        field = name.removeprefix("schedule_inventory_")
        return {
            "atomic_job_count": "integer scheduled atomic-job count",
            "atomic_job_mask": "boolean atomic-job mask",
            "code": "source-native population request code",
            "job_count": "integer scheduled-job count",
            "mode": "source-native integer population-mode code",
            "molecular_job_count": "integer scheduled molecular-job count",
            "molecular_job_mask": "boolean molecular-job mask",
            "molecular_unique_code": "source-native unique molecule code",
            "molecular_unique_code_count": "integer unique molecule-code count",
            "molecular_unique_mode_pair": (
                "source-native ordered (molecule code, population mode) pair"
            ),
            "molecular_unique_packed_start_slot_one_based": (
                "one-based packed population start slot"
            ),
            "output_slots": "integer output-slot count",
            "packed_start_slot_one_based": ("one-based packed population start slot"),
            "packed_start_slot_zero_based": ("zero-based packed population start slot"),
            "target_index": "zero-based population target index",
            "target_name": "population target-name identity",
        }[field]
    raise KeyError(f"no Chapter 4 atmosphere golden unit for {name}")


CHAPTER04_SYNTHESIS_STATE_UNITS = {
    "aluminum_neutral_partition_normalized_population": (
        "cm^-3 per partition function"
    ),
    "carbon_partition_normalized_ion_stage_populations": (
        "cm^-3 per partition function"
    ),
    "eos__ion_stage_fractions_over_partition": (
        "dimensionless ion-stage fraction per partition function"
    ),
    "eos__partition_functions": "dimensionless partition function",
    "eos__partition_normalized_populations": "cm^-3 per partition function",
    "helium_neutral_population": "cm^-3",
    "helium_singly_ionized_population": "cm^-3",
    "hydrogen_ionized_population": "cm^-3",
    "hydrogen_neutral_population": "cm^-3",
    "hydrogen_partition_normalized_ion_stage_populations": (
        "cm^-3 per partition function"
    ),
    "ion_stage_populations": "cm^-3",
    "iron_neutral_partition_normalized_population": ("cm^-3 per partition function"),
    "magnesium_neutral_partition_normalized_population": (
        "cm^-3 per partition function"
    ),
    "mass_density": "g cm^-3",
    "partition_normalized_populations": "cm^-3 per partition function",
    "silicon_neutral_partition_normalized_population": ("cm^-3 per partition function"),
}

CHAPTER04_SYNTHESIS_CALL_UNITS = {
    "caller_file": "Python caller-file path identity",
    "caller_module": "Python caller-module identity",
    "caller_name": "Python caller-function identity",
    "chain_length": "integer observed molecular-call-chain length",
    "device": "Torch execution-device policy",
    "diag__equation_abundance": "linear relative number abundance",
    "diag__equation_count": "active coupled-equation count",
    "diag__equation_species_codes": "source-native equation species code",
    "diag__exhausted_mask": "boolean maximum-iteration exhaustion mask",
    "diag__iterations_completed": "integer Newton iteration count by depth",
    "diag__molecule_codes": "source-native base-100 molecule code",
    "diag__molecule_count": "active molecule-record count",
    "diag__natural_log_formation_constants": (
        "natural logarithm of source-native molecule-dependent formation factor"
    ),
    "diag__normalized_residual": (
        "dimensionless row-scaled molecular conservation residual"
    ),
    "diag__pre_replacement_log_term": (
        "natural logarithm of the source-native pre-replacement molecular term"
    ),
    "diag__pre_replacement_log_term_max": (
        "maximum natural logarithm of pre-replacement molecular term"
    ),
    "diag__pre_replacement_log_term_min": (
        "minimum natural logarithm of pre-replacement molecular term"
    ),
    "diag__pre_replacement_term_nonfinite_count": (
        "integer nonfinite pre-replacement term count by depth"
    ),
    "diag__pre_replacement_term_nonfinite_mask": (
        "boolean nonfinite pre-replacement term mask"
    ),
    "diag__residual": "cm^-3 molecular conservation residual",
    "diag__structure__active_molecule_mask": (
        "dimensionless 0/1 active-molecule multiplier"
    ),
    "diag__structure__component_multiplicity": (
        "dimensionless stoichiometric multiplicity"
    ),
    "diag__structure__electron_equation_index": ("zero-based electron equation index"),
    "diag__structure__full_component_multiplicity": (
        "dimensionless stoichiometric multiplicity"
    ),
    "diag__structure__full_inverse_electron_power": (
        "dimensionless inverse-electron exponent"
    ),
    "diag__structure__inverse_electron_power": (
        "dimensionless inverse-electron exponent"
    ),
    "diag__structure__negative_ion_flag": ("dimensionless 0/1 negative-ion flag"),
    "diag__total_particle_density": "cm^-3",
    "dtype": "Torch floating-point dtype policy",
    "input__electron_density": "cm^-3",
    "input__electron_density_member": (
        "NPZ member-name reference for electron density input"
    ),
    "input__elemental_abundances_member": (
        "NPZ member-name reference for linear elemental abundances"
    ),
    "input__gas_pressure_member": ("NPZ member-name reference for gas pressure"),
    "input__ion_formation_constants": (
        "source-native molecule-dependent ion formation factor with "
        "stoichiometry-dependent density powers"
    ),
    "input__temperature_member": ("NPZ member-name reference for temperature"),
    "max_iter": "integer Newton iteration limit",
    "molecules_path": "molecular-catalog filesystem path identity",
    "output__electron_density": "cm^-3",
    "output__equation_densities": "cm^-3",
    "output__molecular_populations": "cm^-3",
    "output__total_nuclei_number_density": "cm^-3",
    "tol": "dimensionless Newton relative-update tolerance",
}

CHAPTER04_GOLDEN_ALIAS_NAMES = {
    "chapter04_synthesis_molecular_full_cpu_float64.npz": {
        "alias__state__electron_density_member",
        "alias__state__molecular_equation_densities_member",
        "alias__state__molecular_populations_member",
        "alias__state__total_nuclei_number_density_member",
    },
    "chapter04_synthesis_molecular_fixed_cpu_float64.npz": {
        "derived__alias__state__electron_density_member",
        "derived__alias__state__molecular_equation_densities_member",
        "derived__alias__state__molecular_populations_member",
        "derived__alias__state__total_nuclei_number_density_member",
        "derived__alias__trace__internal_electron_density_member",
        "supplied__alias__state__electron_density_member",
        "supplied__alias__state__molecular_equation_densities_member",
        "supplied__alias__state__molecular_populations_member",
        "supplied__alias__state__total_nuclei_number_density_member",
        "supplied__alias__trace__internal_electron_density_member",
    },
    "chapter04_molecular_public_mapping_cpu_float64.npz": {
        "alias__co_reconstruction__grounded_population_member",
        "alias__co_reconstruction__reference_public_lane_member",
        "alias__edge_loader__continuum_edge_midpoint_wavelength_nm_member",
        "alias__edge_loader__continuum_edge_wavelength_nm_member",
        "alias__edge_loader__edge_interval_width_squared_over_two_nm2_member",
        "alias__edge_loader__signed_continuum_edge_frequency_hz_member",
        "alias__fixed_state__electron_density_member",
        "alias__fixed_state__molecular_equation_densities_member",
        "alias__fixed_state__molecular_populations_member",
        "alias__fixed_state__total_nuclei_number_density_member",
        "alias__ion_cube__after_member",
        "alias__ion_cube__before_member",
        "alias__line_population__no_ground_member",
        "alias__molecular_hydrogen__solved_code_101_member",
        "alias__partition_cube__after_member",
        "alias__partition_cube__before_member",
    },
}


def chapter04_synthesis_route_unit(name: str) -> str:
    """Return one full or fixed synthesis-route member representation."""

    if name in CHAPTER04_PUBLISHER_META_UNITS:
        return CHAPTER04_PUBLISHER_META_UNITS[name]
    if name.startswith("oracle__"):
        field = name.removeprefix("oracle__")
        if field not in CHAPTER04_SYNTHESIS_ORACLE_IDENTITY_NAMES:
            raise KeyError(f"unknown Chapter 4 synthesis oracle identity {field}")
        return chapter04_oracle_identity_unit(field)
    if name in (
        CHAPTER04_GOLDEN_ALIAS_NAMES[
            "chapter04_synthesis_molecular_full_cpu_float64.npz"
        ]
        | {
            alias.removeprefix("derived__").removeprefix("supplied__")
            for alias in CHAPTER04_GOLDEN_ALIAS_NAMES[
                "chapter04_synthesis_molecular_fixed_cpu_float64.npz"
            ]
        }
    ):
        return "NPZ member-name reference for a deduplicated route array"
    if name.startswith("input__"):
        return {
            "electron_density_seed": "cm^-3; controlled upstream seed",
            "elemental_abundances": "linear relative number abundance",
            "gas_pressure": "dyn cm^-2",
            "mass_density": "g cm^-3",
            "temperature": "K",
        }[name.removeprefix("input__")]
    if name.startswith("state__"):
        return CHAPTER04_SYNTHESIS_STATE_UNITS[name.removeprefix("state__")]
    if name.startswith(("call_0__", "call_1__")):
        field = name.split("__", 1)[1]
        return CHAPTER04_SYNTHESIS_CALL_UNITS[field]
    if name.startswith("trace__"):
        return {
            "internal_electron_density": "cm^-3",
            "molecular_call_count": "integer molecular-solver call count",
            "published_electron_from_call": (
                "zero-based molecular call chosen for published electron density"
            ),
            "published_electron_from_input": (
                "boolean fixed-EOS electron-density passthrough"
            ),
            "published_molecules_from_call": (
                "zero-based molecular call chosen for published molecule state"
            ),
            "route": "synthesis route identity",
        }[name.removeprefix("trace__")]
    raise KeyError(f"no Chapter 4 synthesis golden unit for {name}")


def chapter04_fixed_synthesis_unit(name: str) -> str:
    """Return a fixed-route field after interpreting its branch prefix."""

    if name in CHAPTER04_PUBLISHER_META_UNITS:
        return CHAPTER04_PUBLISHER_META_UNITS[name]
    if name.startswith("oracle__"):
        field = name.removeprefix("oracle__")
        if field not in CHAPTER04_SYNTHESIS_ORACLE_IDENTITY_NAMES:
            raise KeyError(f"unknown Chapter 4 synthesis oracle identity {field}")
        return chapter04_oracle_identity_unit(field)
    if name.startswith("input__"):
        return chapter04_synthesis_route_unit(name)
    for branch in ("derived__", "supplied__"):
        if name.startswith(branch):
            return chapter04_synthesis_route_unit(name.removeprefix(branch))
    raise KeyError(f"no Chapter 4 fixed synthesis golden unit for {name}")


def chapter04_constants_unit(name: str) -> str:
    """Return one constants/catalog/boundary member representation."""

    if name in CHAPTER04_PUBLISHER_META_UNITS:
        return CHAPTER04_PUBLISHER_META_UNITS[name]
    if name.startswith("atmosphere__"):
        field = name.removeprefix("atmosphere__")
        if field in CHAPTER04_ATMOSPHERE_ORACLE_IDENTITY_NAMES:
            return chapter04_oracle_identity_unit(field)
        if field.startswith("boundary_"):
            boundary = field.removeprefix("boundary_")
            if boundary in {
                "frozen_newton_constants_uint64_bits",
                "normalized_named_populations_uint64_bits",
                "population_constants_uint64_bits",
                "raw_named_populations_uint64_bits",
                "temperature_uint64_bits",
            }:
                return "exact IEEE-754 float64 bit pattern"
            if boundary in {
                "h2_catalog_active_branch_mask",
                "h2_catalog_inactive_branch_mask",
                "polynomial_active_branch_mask",
                "polynomial_inactive_branch_mask",
            }:
                return "boolean exact-boundary branch mask"
            return chapter04_atmosphere_state_unit(boundary)
        if field.startswith("h2_probe_"):
            probe = field.removeprefix("h2_probe_")
            if probe in {
                "catalog_equilibrium_constant_gated_uint64_bits",
                "helper_equilibrium_constant_ungated_uint64_bits",
                "interpolated_partition_uint64_bits",
                "temperature_uint64_bits",
            }:
                return "exact IEEE-754 float64 bit pattern"
            return {
                "catalog_active_branch_mask": "boolean H2 catalog branch mask",
                "catalog_equilibrium_constant_gated": (
                    "cm^3 H2 equilibrium factor after catalog temperature gate"
                ),
                "catalog_inactive_branch_mask": "boolean H2 catalog branch mask",
                "helper_equilibrium_constant_ungated": (
                    "cm^3 ungated H2 equilibrium factor"
                ),
                "helper_finite_positive_input_branch_mask": (
                    "boolean finite-positive-temperature branch mask"
                ),
                "interpolated_partition": "dimensionless H2 partition function",
                "partition_high_clamp_branch_mask": (
                    "boolean high-temperature partition-clamp mask"
                ),
                "partition_interpolation_branch_mask": (
                    "boolean partition-interpolation branch mask"
                ),
                "partition_low_clamp_branch_mask": (
                    "boolean low-temperature partition-clamp mask"
                ),
                "temperature": "K",
            }[probe]
        return {
            "named_molecule_catalog_indices": (
                "zero-based atmosphere molecule-catalog row index"
            ),
            "named_molecule_codes": "source-native base-100 molecule code",
        }[field]
    if name.startswith("synthesis__"):
        field = name.removeprefix("synthesis__")
        if field.startswith("meta__"):
            if field not in CHAPTER04_SYNTHESIS_ORACLE_IDENTITY_NAMES:
                raise KeyError(f"unknown Chapter 4 synthesis oracle identity {field}")
            return chapter04_oracle_identity_unit(field)
        if field.startswith("trace__"):
            return {
                "provisional_h2_fixed_eos_call_count": ("integer fixed-EOS call count"),
                "provisional_h2_molecular_solve_call_count": (
                    "integer molecular-solver call count"
                ),
                "route": "synthesis boundary route identity",
            }[field.removeprefix("trace__")]
        if field.startswith("alignment__"):
            alignment = field.removeprefix("alignment__")
            return {
                "atmosphere_only_count": "integer atmosphere-only record count",
                "atmosphere_only_rounded_code_keys": (
                    "molecule code rounded to the 1e-3 matching key"
                ),
                "atmosphere_only_row_indices": (
                    "zero-based atmosphere catalog row index"
                ),
                "atmosphere_shared_component_semantics": (
                    "source-native flattened component semantic species code"
                ),
                "atmosphere_shared_row_indices": (
                    "zero-based atmosphere catalog row index"
                ),
                "coefficient_mismatch_mask": (
                    "boolean shared-record coefficient mismatch mask"
                ),
                "component_mismatch_mask": (
                    "boolean shared-record component mismatch mask"
                ),
                "semantic_mismatch_count": (
                    "integer shared-record semantic mismatch count"
                ),
                "semantic_mismatch_mask": (
                    "boolean shared-record semantic mismatch mask"
                ),
                "shared_component_offsets": (
                    "zero-based ragged shared-component offset"
                ),
                "shared_count": "integer shared molecule-record count",
                "shared_molecule_codes": "source-native base-100 molecule code",
                "shared_rounded_code_keys": (
                    "molecule code rounded to the 1e-3 matching key"
                ),
                "shared_row_indices_differ_mask": (
                    "boolean shared-record row-order difference mask"
                ),
                "synthesis_only_count": "integer synthesis-only record count",
                "synthesis_only_molecule_codes": (
                    "source-native base-100 molecule code"
                ),
                "synthesis_only_rounded_code_keys": (
                    "molecule code rounded to the 1e-3 matching key"
                ),
                "synthesis_only_row_indices": (
                    "zero-based synthesis catalog row index"
                ),
                "synthesis_shared_component_semantics": (
                    "source-native flattened component semantic species code"
                ),
                "synthesis_shared_row_indices": (
                    "zero-based synthesis catalog row index"
                ),
            }[alignment]
        if field.startswith("catalog__"):
            catalog = field.removeprefix("catalog__")
            catalog_kind, catalog_field = catalog.split("__", 1)
            if catalog_kind not in {"atmosphere", "synthesis"}:
                raise KeyError(f"unknown Chapter 4 catalog namespace {catalog_kind}")
            if catalog_field == "hard_coded_molecular_atomic_masses_amu":
                if catalog_kind != "synthesis":
                    raise KeyError(
                        f"{catalog_field} is not an atmosphere catalog field"
                    )
                return "amu"
            if catalog_field == "hard_coded_molecular_atomic_masses_sha256":
                if catalog_kind != "synthesis":
                    raise KeyError(
                        f"{catalog_field} is not an atmosphere catalog field"
                    )
                return "SHA-256 hexadecimal identity"
            if catalog_field in {
                "active_component_mask",
                "active_equation_mask",
                "active_molecule_mask",
            }:
                if catalog_kind != "synthesis":
                    raise KeyError(
                        f"{catalog_field} is not an atmosphere catalog field"
                    )
                return "boolean active or inverse-electron sentinel mask"
            if catalog_field == "component_inverse_electron_sentinel_mask":
                return "boolean active or inverse-electron sentinel mask"
            if catalog_field == "component_semantic_species_codes":
                return "source-native component semantic species code"
            return molecular_catalog_unit(catalog_field)
        if field.startswith("boundary__"):
            boundary = field.removeprefix("boundary__")
            return {
                "provisional_h2_active_population_mask": (
                    "boolean nonzero provisional H2 population mask"
                ),
                "provisional_h2_electron_density": "cm^-3",
                "provisional_h2_equilibrium_factor": "cm^3 H2 formation factor",
                "provisional_h2_gas_pressure": "dyn cm^-2",
                "provisional_h2_molecular_exhausted_mask": (
                    "boolean maximum-iteration exhaustion mask"
                ),
                "provisional_h2_molecular_iterations": (
                    "integer Newton iteration count by depth"
                ),
                "provisional_h2_neutral_hydrogen": "cm^-3",
                "provisional_h2_population": "cm^-3",
                "provisional_h2_temperature": "K",
                "provisional_h2_temperature_bits": (
                    "exact IEEE-754 float64 bit pattern"
                ),
                "provisional_h2_temperature_branch_mask": (
                    "boolean exact-threshold branch mask"
                ),
                "provisional_h2_threshold_bits": ("exact IEEE-754 float64 bit pattern"),
                "synthesis_polynomial_direct_constants": (
                    "source-native molecule-dependent polynomial formation factor"
                ),
                "synthesis_polynomial_electron_density": "cm^-3",
                "synthesis_polynomial_gas_pressure": "dyn cm^-2",
                "synthesis_polynomial_mask": (
                    "boolean polynomial-constant molecule mask"
                ),
                "synthesis_polynomial_molecule_codes": (
                    "source-native base-100 molecule code"
                ),
                "synthesis_polynomial_temperature": "K",
                "synthesis_polynomial_temperature_bits": (
                    "exact IEEE-754 float64 bit pattern"
                ),
                "synthesis_polynomial_temperature_branch_mask": (
                    "boolean exact-threshold branch mask"
                ),
                "synthesis_polynomial_threshold_bits": (
                    "exact IEEE-754 float64 bit pattern"
                ),
                "synthesis_pre_newton_formation_constants": (
                    "source-native molecule-dependent formation factor"
                ),
                "synthesis_pre_newton_log_formation_constants": (
                    "natural logarithm of source-native formation factor"
                ),
            }[boundary]
    raise KeyError(f"no Chapter 4 constants golden unit for {name}")


def chapter04_public_mapping_unit(name: str) -> str:
    """Return one public-mapping golden member representation."""

    if name in CHAPTER04_PUBLISHER_META_UNITS:
        return CHAPTER04_PUBLISHER_META_UNITS[name]
    if name.startswith("oracle__"):
        field = name.removeprefix("oracle__")
        if field not in CHAPTER04_SYNTHESIS_ORACLE_IDENTITY_NAMES:
            raise KeyError(f"unknown Chapter 4 synthesis oracle identity {field}")
        return chapter04_oracle_identity_unit(field)
    if (
        name
        in CHAPTER04_GOLDEN_ALIAS_NAMES[
            "chapter04_molecular_public_mapping_cpu_float64.npz"
        ]
    ):
        return "NPZ member-name reference for a deduplicated public-route array"
    if name.startswith("input__"):
        return chapter04_synthesis_route_unit(name)
    if name.startswith("fixed_state__"):
        return CHAPTER04_SYNTHESIS_STATE_UNITS[name.removeprefix("fixed_state__")]
    if name.startswith("call_0__"):
        return CHAPTER04_SYNTHESIS_CALL_UNITS[name.removeprefix("call_0__")]
    if name.startswith("mapping__"):
        return {
            "molecule_code_offsets": (
                "zero-based ragged equilibrium-code offset by line species"
            ),
            "molecule_codes": "source-native base-100 equilibrium molecule code",
            "public_columns": "zero-based public atmosphere-cube column index",
            "species_codes": "source-native molecular line species code",
        }[name.removeprefix("mapping__")]
    if name.startswith("co_reconstruction__"):
        return {
            "catalog_index": "zero-based synthesis molecule-catalog row index",
            "component_atomic_masses_amu": "amu",
            "component_equation_indices": (
                "zero-based molecular equilibrium equation index"
            ),
            "component_species_codes": (
                "source-native component semantic species code"
            ),
            "equation_densities": "cm^-3",
            "equilibrium_code": "source-native base-100 equilibrium molecule code",
            "independent_population": "cm^-3 per molecular partition function",
            "leading_coefficient": ("eV, polynomial equilibrium exponent coefficient"),
            "line_species_code": "source-native molecular line species code",
            "molecular_mass_amu": "amu",
            "no_ground_neutral_partitions": "dimensionless partition function",
            "normalization": (
                "source-native 1.8786e20 sqrt((molecular mass in amu * K)^3) "
                "N/U normalization factor"
            ),
            "public_column_index": "zero-based public atmosphere-cube column index",
            "public_stage_index": "zero-based public atmosphere-cube stage index",
            "raw_discrimination_mask": (
                "boolean raw-population versus N/U discrimination mask"
            ),
            "raw_equilibrium_population": "cm^-3",
            "temperature": "K",
            "transformed_equation_densities": (
                "source-native species-dependent N/U equation factor after "
                "partition and thermal normalization; not cm^-3"
            ),
        }[name.removeprefix("co_reconstruction__")]
    if name.startswith("edge_loader__"):
        return {
            "continuum_edge_sample_frequency_hz": "Hz",
            "continuum_edge_wavenumber_cm": ("cm^-1; sign is interpolation metadata"),
        }[name.removeprefix("edge_loader__")]
    if name.startswith("line_population__"):
        return {
            "ground_discrimination_mask": (
                "boolean grounded versus no-ground partition-policy mask"
            ),
            "ground_discrimination_max_abs": ("cm^-3 per molecular partition function"),
            "grounded": "cm^-3 per molecular partition function",
            "independent_no_ground": "cm^-3 per molecular partition function",
            "public": "cm^-3 per molecular partition function",
        }[name.removeprefix("line_population__")]
    if name.startswith("partition__"):
        return {
            "bridge_cube": "dimensionless partition function",
            "elements_without_ground_floor": "zero-based element index",
            "grounded_cube": "dimensionless partition function",
            "without_ground_floor": "dimensionless partition function",
            "without_ground_floor_offsets": (
                "zero-based ragged partition-vector offset"
            ),
            "without_ground_floor_stage_counts": "integer ion-stage count",
        }[name.removeprefix("partition__")]
    if name.startswith("structured__"):
        field = name.removeprefix("structured__")
        if field in {
            "continuum_edge_interval_width_squared_over_two_nm2",
            "continuum_edge_midpoint_wavelength_nm",
            "continuum_edge_wavelength_nm",
            "signed_continuum_edge_frequency_hz",
        }:
            return CONTINUUM_EDGE_GRID_UNITS[field]
        if field in CHAPTER04_SYNTHESIS_STATE_UNITS:
            return CHAPTER04_SYNTHESIS_STATE_UNITS[field]
        return {
            "atmosphere_schema_version": "schema version",
            "column_mass": "g cm^-2",
            "electron_density": "cm^-3",
            "elemental_abundances": "linear relative number abundance",
            "fractional_doppler_widths": "dimensionless v/c",
            "gas_pressure": "dyn cm^-2",
            "hc_over_kt": "cm",
            "microturbulence": "cm s^-1",
            "molecular_hydrogen_population": "cm^-3",
            "temperature": "K",
        }[field]
    if name == "molecular_hydrogen__catalog_index":
        return "zero-based synthesis molecule-catalog row index"
    if name == "schema__structured_keyset":
        return "public structured-atmosphere member-name identity"
    if name.startswith("trace__"):
        return {
            "edge_grid_cache_present_before": (
                "boolean pre-call edge-grid cache state"
            ),
            "edge_grid_call_count": "integer continuum-edge loader call count",
            "edge_grid_caller": "Python caller-function identity",
            "edge_grid_path": "continuum-edge archive path identity",
            "fallback_resolve_call_count": (
                "integer fallback-path resolver call count"
            ),
            "fixed_eos_call_count": "integer fixed-EOS call count",
            "fixed_eos_caller": "Python caller-function identity",
            "independent_line_reconstruction_count": (
                "integer independent line-reconstruction count"
            ),
            "line_mapping_caller": "Python caller-function identity",
            "line_mapping_grounded_diagnostic_call_count": (
                "integer grounded diagnostic mapping call count"
            ),
            "line_mapping_production_call_count": (
                "integer production mapping call count"
            ),
            "molecular_solve_call_count": "integer molecular-solver call count",
            "partition_no_ground_call_count": (
                "integer no-ground partition-builder call count"
            ),
            "partition_no_ground_caller": "Python caller-function identity",
            "reused_fixed_molecular_arrays": (
                "boolean fixed-state molecular-array reuse invariant"
            ),
            "route": "public builder route identity",
        }[name.removeprefix("trace__")]
    raise KeyError(f"no Chapter 4 public mapping golden unit for {name}")


def chapter04_golden_unit(path: str, name: str) -> str:
    """Return an exhaustive, fail-closed Chapter 4 golden representation."""

    filename = Path(path).name
    expected_path = f"data/golden/payne_zero/chapter04/{filename}"
    if path != expected_path:
        raise KeyError(f"not a declared Chapter 4 golden path: {path}")
    aliases = CHAPTER04_GOLDEN_ALIAS_NAMES.get(filename, set())
    if name in aliases:
        return "NPZ member-name reference for a deduplicated route array"
    if name.startswith("alias__") or "__alias__" in name:
        raise KeyError(f"not a declared Chapter 4 alias in {filename}: {name}")
    if filename == "chapter04_molecular_constants_cpu_float64.npz":
        return chapter04_constants_unit(name)
    if filename == "chapter04_atmosphere_molecular_state_cpu_float64.npz":
        return chapter04_atmosphere_archive_unit(name)
    if filename == "chapter04_synthesis_molecular_full_cpu_float64.npz":
        return chapter04_synthesis_route_unit(name)
    if filename == "chapter04_synthesis_molecular_fixed_cpu_float64.npz":
        return chapter04_fixed_synthesis_unit(name)
    if filename == "chapter04_molecular_public_mapping_cpu_float64.npz":
        return chapter04_public_mapping_unit(name)
    raise KeyError(f"not a declared Chapter 4 golden path: {path}")


def _chapter05_archive_name(path: str) -> str:
    """Resolve one canonical Chapter 5 golden path without suffix matching."""

    filename = Path(path).name
    expected_path = f"data/golden/payne_zero/chapter05/{filename}"
    if path != expected_path or filename not in CHAPTER05_GOLDEN_ARCHIVES:
        raise KeyError(f"not a declared Chapter 5 golden path: {path}")
    return filename


def _chapter05_identity_unit(name: str) -> str | None:
    """Return an exact representation for reviewed provenance metadata."""

    lowered = name.lower()
    if (
        lowered.endswith("_sha256")
        or lowered.endswith("_digest")
        or "fingerprint" in lowered
    ):
        return "SHA-256 hexadecimal identity"
    if lowered.endswith("payne_zero_commit"):
        return "Git commit identity"
    if lowered.endswith("_schema_version") or lowered.endswith(
        "_archive_schema_version"
    ):
        return "schema version"
    if lowered.endswith("_version"):
        return "software version identity"
    if lowered.endswith("_count") or "_count_" in lowered:
        return "count"
    if (
        lowered.endswith("_verified")
        or lowered.endswith("_performed")
        or lowered.endswith("_complete")
        or lowered.endswith("_only")
    ):
        return "boolean"
    if lowered.endswith("_path"):
        return "filesystem path identity"
    if lowered.endswith(("_names", "_name")):
        return "ordered identifier"
    if lowered.startswith("meta__environment__"):
        return "controlled process-environment setting"
    if lowered in {
        "meta__archive_kind",
        "meta__work_dtype",
        "oracle__meta__capture_scope",
        "oracle__meta__environment__work_dtype",
        "oracle__meta__platform",
    }:
        return "runtime or archive identity"
    if lowered.startswith(("meta__", "oracle__identity__", "oracle__meta__")):
        return "runtime provenance identity"
    return None


def chapter05_golden_unit(path: str, name: str) -> str:
    """Return an exhaustive, membership-gated Chapter 5 representation."""

    filename = _chapter05_archive_name(path)
    if name not in _VALIDATED_CHAPTER05_MEMBERS.get(filename, frozenset()):
        raise KeyError(f"not a reviewed Chapter 5 member in {filename}: {name}")

    exact_units = {
        "line_reference__active_index": "zero-based index",
        "seam__molecular_boundary__cia_temperature_fraction": "dimensionless",
        "evidence__molecular_boundary__cia_temperature_table_index": (
            "zero-based index"
        ),
        "evidence__sampling_boundary__count_263_present": "boolean",
        "evidence__sampling_boundary__count_299_present": "boolean",
        "evidence__ifop19__bolometric_like_source": (
            "erg s^-1 cm^-2 sr^-1; bolometric sigma*T^4/pi convention"
        ),
    }
    if name in exact_units:
        return exact_units[name]

    identity = _chapter05_identity_unit(name)
    if identity is not None:
        return identity
    if name.startswith("alias__"):
        return "ordered identifier"
    if name in {
        "inventory__raw_member_name",
        "inventory__disposition",
        "inventory__published_member",
    }:
        return "ordered identifier"
    if name.endswith(
        (
            "_field_name",
            "_field_names",
            "_component_name",
            "_case_name",
            "_family",
            "_policy_label",
        )
    ):
        return "ordered identifier"
    if name.endswith("_temperature_uint64_bits"):
        return "exact IEEE-754 float64 bit pattern"
    if "packed_wavelength_index" in name:
        return "integer logarithmic-grid coordinate"
    if "line_reference__threshold" in name:
        return "cm^2 g^-1 with embedded 1e-3 and stimulated-emission division"
    if "cross_section_times_partition" in name:
        return "cm^2 times partition-function convention"
    if "interpolated_log10_coefficient" in name:
        return "log10 of source-native collision-induced absorption coefficient"
    if "freefree_prefactor" in name:
        return (
            "source-native Kramers free-free prefactor; combined with hc/kT "
            "to yield cm^2"
        )
    if "freefree_threshold" in name or name.endswith("hydrogen_tail_edge"):
        return "cm^-1"
    if name.endswith("hminus_freefree_rows"):
        return "source-native H-minus free-free coefficient rows"
    if name.endswith("rayleigh_factor"):
        return (
            "source-native Rayleigh polarizability factor; squared in the "
            "microscopic cross-section calculation"
        )
    if name.endswith("silicon_singly_ionized_peach_frequency_rows"):
        return "natural logarithm of source-native Si II cross section"
    if "cross_section" in name:
        return "cm^2"
    if "natural_log_frequency" in name:
        return "natural logarithm of numerical frequency in Hz"
    if "natural_log_temperature" in name:
        return "natural logarithm of numerical temperature in K"
    if (
        "frequency_weight" in name
        or name.endswith(("_frequency_hz", "_hz"))
        or name.endswith("signed_edge_flipped")
    ):
        return "Hz"
    if "wavenumber" in name:
        return "cm^-1"
    if "interval_width_squared_over_two_nm2" in name:
        return "nm^2"
    if "wavelength" in name or name.endswith("light_speed_nm_per_s"):
        if name.endswith("light_speed_nm_per_s"):
            return "nm s^-1"
        return "nm"
    if "temperature" in name:
        return "K"
    if "mass_density" in name:
        return "g cm^-3"
    if "electron_density" in name:
        return "cm^-3"
    if "rosseland_opacity" in name:
        return "cm^2 g^-1"
    if "equilibrium_constant" in name:
        return "cm^3"
    if "partition_normalized" in name:
        return "cm^-3 per partition function"
    if "population" in name or name.endswith(
        ("schema_h2_original", "schema_h2_perturbed")
    ):
        if "charge_square_population_sum" in name:
            return "cm^-3, charge-square weighted"
        return "cm^-3"
    if "stimulated_emission" in name:
        return "dimensionless"
    if "departure_coefficient" in name:
        return "dimensionless"
    if name.endswith(
        (
            "_basis",
            "_basis_sum",
            "_temperature_fraction",
            "_column_weight",
        )
    ):
        return "dimensionless"
    if "source_numerator" in name:
        return (
            "(cm^2 g^-1) times (erg s^-1 cm^-2 sr^-1 Hz^-1); "
            "absorption-weighted source numerator"
        )
    if "source" in name:
        return "erg s^-1 cm^-2 sr^-1 Hz^-1"
    if (
        "absorption" in name
        or "scattering" in name
        or name.endswith("h2_rayleigh_increment")
        or name.endswith("__component")
    ):
        return "cm^2 g^-1"
    if name.endswith("_maximum"):
        return "cm^2 g^-1"
    if name.endswith("_opacity_flags") or name.endswith("_runner_opacity_flags"):
        return "source-native integer IFOP flag"
    if name.endswith("_side"):
        return "source-native signed side indicator"
    if (
        name.endswith("_active")
        or name.endswith("_valid")
        or name.endswith("_mask")
        or name.endswith("_supplied")
        or name.endswith("_bit_equal")
        or name.endswith("_bit_invariant")
        or name.endswith("_is_zero")
        or name.endswith("_present")
        or name.endswith("_was_called")
        or name.endswith("_energy_first")
        or "_matches_" in name
        or "_active_" in name
    ):
        return "boolean"
    if name.endswith("_index") or name.endswith("_indices") or "_index_" in name:
        return "zero-based index"
    if name.endswith("_count") or "_count_" in name:
        return "count"
    raise KeyError(f"no Chapter 5 golden unit for {filename}: {name}")


def _chapter05_reader_axes(name: str, shape: tuple[int, ...]) -> list[str]:
    """Return semantic axes for one compact reader member."""

    if name.startswith(("meta__", "oracle__")):
        if not shape:
            return []
        if name.endswith("regime_names"):
            return ["regime"]
        if name.endswith("process_control_name") or name.endswith(
            "process_control_value"
        ):
            return ["process_control"]
        return ["provenance_entry"]
    if name == "axis__depth_index":
        return ["depth"]
    if name.startswith("axis__diagnostic__"):
        return ["diagnostic_family" if shape == (9,) else "diagnostic_frequency"]
    if name.startswith("axis__regime_"):
        return ["regime"]
    if name.startswith("atmosphere__component__"):
        if name.endswith(("_name",)):
            return [
                "scattering_component"
                if name.endswith("scattering_name")
                else "absorption_component"
            ]
        if len(shape) == 4:
            component_axis = (
                "absorption_component" if shape[1] == 14 else "scattering_component"
            )
            return ["regime", component_axis, "depth", "diagnostic_frequency"]
        return ["regime", "depth", "diagnostic_frequency"]
    if name.startswith("atmosphere__molecular_component__"):
        if name.endswith("_name"):
            return ["molecular_component"]
        if len(shape) == 4:
            return ["regime", "molecular_component", "depth", "diagnostic_frequency"]
        return ["regime", "depth", "diagnostic_frequency"]
    if name.startswith("atmosphere__compact__"):
        return ["regime", "depth", "diagnostic_frequency"]
    if name == "atmosphere__runner_opacity_flags":
        return ["ifop_slot"]
    if name.startswith("line_reference__"):
        suffix = name.removeprefix("line_reference__")
        return {
            "active_absorption": ["regime", "depth", "active_line_slot"],
            "active_count": ["regime"],
            "active_frequency_hz": ["regime", "active_line_slot"],
            "active_index": ["regime", "active_line_slot"],
            "active_scattering": ["regime", "depth", "active_line_slot"],
            "active_source": ["regime", "depth", "active_line_slot"],
            "active_valid": ["regime", "active_line_slot"],
            "packed_wavelength_index": ["line_reference_wavelength"],
            "threshold": ["regime", "depth", "line_reference_wavelength"],
            "wavelength_nm": ["line_reference_wavelength"],
        }[suffix]
    if name.startswith("reader_state__"):
        field = name.removeprefix("reader_state__")
        if field == "field_name":
            return ["reader_state_field"]
        axes = ["regime", "depth"]
        if len(shape) == 3:
            axes.append(
                "ion_stage"
                if field.endswith("ion_stage_populations")
                else "hot_metal_slot"
                if field == "hot_metal_populations"
                else "charge_square_species"
            )
        return axes
    if name.startswith("seam__ifop__"):
        return [] if not shape else ["depth", "diagnostic_frequency"]
    if name.startswith("seam__molecular_boundary__"):
        field = name.split("__")[-1]
        if field in {
            "ch_cross_section_times_partition",
            "oh_cross_section_times_partition",
        }:
            return ["temperature_boundary", "molecular_frequency"]
        if field == "ch_oh_temperature_k":
            return ["temperature_boundary"]
        if field == "cia_absorption":
            return ["cia_temperature_boundary", "cia_frequency"]
        if field.startswith("cia_"):
            return ["cia_frequency" if shape == (4,) else "cia_temperature_boundary"]
        return ["molecular_frequency"]
    if name.startswith("synthesis__edge__"):
        if not shape:
            return []
        if shape == (1020,):
            return ["packaged_edge_sample"]
        if shape in {(340,), (341,)}:
            return ["continuum_edge_interval" if shape == (340,) else "continuum_edge"]
        if shape == (11,):
            return ["edge_probe_wavelength"]
        if shape == (12,):
            return ["used_continuum_edge"]
        return ["requested_wavelength"]
    if name.startswith("synthesis__extension__"):
        if not shape:
            return []
        if name.endswith("_field_names"):
            return ["declared_field"]
        if shape == (4,):
            return ["regime"]
        if shape == (12,):
            return ["supported_extension_wavelength"]
        if len(shape) == 3:
            return ["regime", "depth", "supported_extension_wavelength"]
    if name.startswith("synthesis__seam__"):
        return ["regime"] if shape else []
    if name.startswith(("synthesis__standard__", "synthesis__diagnostic__")):
        if not shape:
            return []
        wavelength_axis = (
            "diagnostic_frequency"
            if name.startswith("synthesis__diagnostic__")
            else "requested_wavelength"
        )
        if name.endswith(("_component_name", "_case_name")):
            return [
                "synthesis_component"
                if name.endswith("_component_name")
                else "isolated_minor_case"
            ]
        if shape == (4,):
            return ["regime"]
        if len(shape) == 4:
            component_axis = (
                "isolated_minor_case"
                if "__isolated_minor__" in name
                else "synthesis_component"
            )
            return ["regime", component_axis, "depth", wavelength_axis]
        if len(shape) == 3:
            return ["regime", "depth", wavelength_axis]
        if len(shape) == 1:
            return [wavelength_axis]
    raise KeyError(f"no Chapter 5 reader axes for {name} with shape {shape}")


def _chapter05_integration_axes(name: str, shape: tuple[int, ...]) -> list[str]:
    """Return semantic axes for one exhaustive integration member."""

    if not shape:
        return []
    if name.startswith("alias__"):
        raise KeyError(f"Chapter 5 alias {name} must be scalar")
    if name.startswith(("meta__", "oracle__")):
        if name.endswith("regime_names"):
            return ["regime"]
        if name.endswith("process_control_name") or name.endswith(
            "process_control_value"
        ):
            return ["process_control"]
        return ["provenance_entry"]
    if name.startswith("inventory__"):
        return ["raw_capture_member"]
    if name.startswith("atmosphere_product__"):
        if name.endswith("grid_bank_index"):
            return ["regime"]
        return ["regime", "depth", "atmosphere_frequency"]
    if name.startswith("grid_bank__"):
        if name.endswith(("policy_index", "policy_label", "derived_frequency_sha256")):
            return ["sampling_policy"]
        return ["sampling_policy", "atmosphere_frequency"]
    if name == "sampling_boundary__grid_bank_index":
        return ["sampling_boundary"]
    if name.startswith("evidence__fixture_payload__"):
        return ["fixture_field"]
    if name.startswith("evidence__sampling_boundary__"):
        return ["sampling_boundary"]
    if name.startswith("evidence__ifop19__"):
        if shape == (20,):
            return ["ifop_slot"]
        if shape == (6,):
            return ["depth"]
        return ["depth", "diagnostic_frequency"]
    if name.startswith("evidence__molecular_boundary__"):
        if len(shape) == 2:
            return [
                "cia_temperature_boundary",
                "cia_frequency" if shape == (3, 4) else "cia_interpolation_sample",
            ]
        return [
            "h2_temperature_boundary" if shape == (8,) else "cia_temperature_boundary"
        ]
    if name.startswith("evidence__molecular_entry__"):
        if shape == (6,):
            return ["depth"]
        if shape == (3,):
            return ["owner_cutoff_temperature"]
        if shape == (3, 27):
            return ["owner_cutoff_temperature", "diagnostic_frequency"]
        return ["depth", "diagnostic_frequency"]
    if name.startswith("evidence__synthesis__extension__input__invariant__"):
        field = name.split("__")[-1]
        if shape == (6,):
            return ["peach_temperature"]
        if len(shape) == 1:
            return ["supported_extension_wavelength"]
        first_axis = {
            3: "edge",
            5: "hydrogen_low_level",
            9: "hydrogen_high_level",
            10: "helium_low_level",
            11: "hminus_freefree_coefficient",
            12: "supported_extension_wavelength",
            14: "magnesium_ionized_level",
            15: "magnesium_level",
            25: "carbon_level",
            28: "helium_high_level",
            33: "silicon_level",
            48: "iron_transition",
        }[shape[0]]
        second_axis = (
            "peach_temperature"
            if field == "silicon_singly_ionized_peach_frequency_rows"
            else "hminus_freefree_coefficient"
            if field == "hminus_freefree_rows"
            else "supported_extension_wavelength"
        )
        return [first_axis, second_axis]
    if name.startswith("evidence__synthesis__standard__trace__"):
        return ["continuum_edge_interval" if shape == (340,) else "unused_edge_sample"]
    if name == "evidence__synthesis__counterfactual__signed_edge_flipped":
        return ["continuum_edge"]
    if name.startswith("evidence__"):
        parts = name.split("__")
        if len(parts) > 2 and parts[1] in {
            "hot_dwarf",
            "solar_dwarf",
            "low_gravity_giant",
            "cool_molecule_rich",
        }:
            if "__activation__" in name:
                return [
                    "atmosphere_component"
                    if "__atmosphere__" in name
                    else "synthesis_component"
                ]
            if "__molecular_component__" in name:
                return ["depth", "diagnostic_frequency"]
            if "__extension__input__pops__" in name:
                return ["depth"] if len(shape) == 1 else ["depth", "ion_stage"]
            if "__counterfactual__" in name:
                return (
                    ["depth"] if len(shape) == 1 else ["depth", "requested_wavelength"]
                )
    raise KeyError(f"no Chapter 5 integration axes for {name} with shape {shape}")


def chapter05_golden_member_metadata(
    path: str,
    name: str,
    shape: tuple[int, ...],
) -> dict[str, object]:
    """Return unit, axes, and single-owner note for one reviewed member."""

    filename = _chapter05_archive_name(path)
    unit = chapter05_golden_unit(path, name)
    axes = (
        _chapter05_reader_axes(name, shape)
        if filename == CHAPTER05_READER_NAME
        else _chapter05_integration_axes(name, shape)
    )
    if len(axes) != len(shape):
        raise KeyError(f"{filename}:{name} shape {shape} has invalid axes {axes}")
    if name.startswith("alias__"):
        ownership = "integration alias; stored string resolves to a reader-owned member"
    elif name.startswith(("meta__", "oracle__")):
        ownership = (
            "common reviewed identity metadata"
            if filename == CHAPTER05_READER_NAME
            else "common reviewed identity metadata or integration-only provenance"
        )
    elif filename == CHAPTER05_READER_NAME:
        ownership = "reader golden owns this compact scientific member"
    else:
        ownership = "integration golden owns this exhaustive scientific member"
    return {"unit": unit, "axes": axes, "ownership": ownership}


def schema_units() -> dict[str, str]:
    """Return exact schema-v4 public units plus the archive version field."""

    schema_path = (
        REPOSITORY_ROOT / "data" / "static" / "schemas" / "atmosphere_schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    units = {field["name"]: field["unit"] for field in schema["required_arrays"]}
    units["atmosphere_schema_version"] = "schema version"
    return units


def schema_inventory() -> dict[str, dict[str, object]]:
    """Return every declarative schema-v4 public array."""

    schema_path = (
        REPOSITORY_ROOT / "data" / "static" / "schemas" / "atmosphere_schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return {
        field["name"]: {
            "shape": field["shape"],
            "unit": field["unit"],
        }
        for field in schema["required_arrays"]
    }


def unit_for(path: str, name: str) -> str:
    """Return the declared unit or representation for one stored array."""

    if "/golden/payne_zero/chapter15/" in path:
        if path.endswith("chapter15_verified_solar_spectrum_cpu_float64.npz"):
            return {
                "wavelength_nm": "nm",
                "flux_total": "source-native emergent flux density",
                "flux_continuum": "source-native emergent flux density",
                "normalized_flux": "dimensionless flux ratio",
            }[name]
        if name == "hydrogen_ionized_population":
            return "cm^-3"
        if name == "atmosphere_schema_version":
            return "schema version"
        return schema_units()[name]
    if path.endswith("continuum_level_tables.npz"):
        if name.endswith("_level_energy_cm"):
            return "cm^-1"
        if name.endswith("_level_statistical_weight"):
            return "dimensionless statistical weight"
        if name == "element_block_offsets":
            return "zero-based packed element-block offset"
        if name == "partition_interpolation_scale":
            return "source-native partition interpolation scale"
        raise KeyError(name)
    if path.endswith("chapter11_solar_seed.npz"):
        return {
            "column_mass": "g cm^-2",
            "convective_flux": "erg cm^-2 s^-1",
            "convective_velocity": "cm s^-1",
            "effective_temperature": "K",
            "electron_density": "cm^-3",
            "fixed_column_abundance_values": (
                "H/He linear number fraction; metals log10 number fraction"
            ),
            "fixture_role": "fixture-role declaration",
            "gas_pressure": "dyn cm^-2",
            "log_surface_gravity": "log10(cm s^-2)",
            "microturbulence": "cm s^-1",
            "opacity_flags": "source-native IFOP switch",
            "payne_zero_commit": "Git commit identity",
            "pressure_iteration_enabled": "boolean 0/1 switch",
            "previous_integrated_radiation_pressure": "dyn cm^-2",
            "previous_turbulent_pressure": "dyn cm^-2",
            "radiative_acceleration": "cm s^-2",
            "rosseland_opacity": "cm^2 g^-1",
            "source_sha256": "SHA-256 hexadecimal identity",
            "temperature": "K",
        }[name]
    if path.endswith("chapter11_observed_atomic_subset.provenance.npz"):
        return {
            "payne_zero_commit": "Git commit identity",
            "selection": "source-row selection declaration",
            "source_row_index": "zero-based source row index",
            "source_sha256": "SHA-256 hexadecimal identity",
        }[name]
    if path.endswith("chapter11_detailed_transition_subset.npz"):
        return {
            "continuum_species_slot": "one-based packed continuum species slot",
            "hydrogen_continuum_selector_index": (
                "source-native hydrogen continuum selector index"
            ),
            "line_limit": "source-native line-limit code",
            "line_type": "source-native detailed line-type code",
            "lower_excitation_cm": "cm^-1",
            "lower_hydrogen_level": "principal quantum number",
            "oscillator_strength": "dimensionless oscillator strength",
            "packed_species_slot": "one-based packed population slot",
            "packed_wavelength_index": "packed logarithmic wavelength index",
            "payne_zero_commit": "Git commit identity",
            "radiative_damping": "s^-1",
            "selection": "source-row selection declaration",
            "source_row_index": "zero-based source row index",
            "source_sha256": "SHA-256 hexadecimal identity",
            "stark_damping": "source-native Stark damping coefficient",
            "upper_hydrogen_level": "principal quantum number",
            "vacuum_wavelength_nm": "nm",
            "van_der_waals_damping": (
                "source-native van der Waals damping coefficient"
            ),
        }[name]
    if path.endswith("chapter02_schema_v4_minimal.npz"):
        return schema_units()[name]
    if path.endswith("chapter02_transfer_inputs.npz"):
        return TRANSFER_INPUT_UNITS[name]
    if path.endswith("radiative_transfer_tables.npz"):
        return TRANSFER_TABLE_UNITS[name]
    if path.endswith("transfer_tables.npz"):
        return TRANSFER_TABLE_UNITS[name]
    if path.endswith("chapter02_transfer_outputs.npz"):
        return GOLDEN_IDENTITY_UNITS.get(
            name, "exact transfer-accumulator units (comparison-only fixture)"
        )
    if path.endswith(
        (
            "special_partition_tables.npz",
            "iron_group_partition_tables.npz",
            "ionization_potential_tables.npz",
            "packed_level_metadata.npz",
            "isotope_tables.npz",
        )
    ):
        return atmosphere_eos_unit(name)
    if path.endswith("partition_saha_tables.npz"):
        return synthesis_eos_unit(name)
    if path.endswith("chapter03_synthesis_eos_state.npz"):
        return CHAPTER03_SYNTHESIS_FIXTURE_UNITS[name]
    if path.endswith("chapter03_atom_only_inputs.npz"):
        return CHAPTER03_ATOM_ONLY_FIXTURE_UNITS[name]
    if path.endswith("chapter04_molecular_inputs.npz"):
        return CHAPTER04_MOLECULAR_FIXTURE_UNITS[name]
    if path.endswith("chapter06_fe_i_source_row_873702.npz"):
        return CHAPTER06_FE_SUBSET_UNITS[name]
    if path.endswith(
        (
            "molecular_equilibrium_atmosphere.npz",
            "molecular_equilibrium_synthesis.npz",
        )
    ):
        return molecular_catalog_unit(name)
    if path.endswith("molecular_equilibrium_tables.npz"):
        return {
            "atomic_mass_amu": "amu",
            "h2_partition_function": "dimensionless partition function",
        }[name]
    if path.endswith(
        (
            "continuum_opacity_tables.npz",
            "karzas_latter_tables.npz",
            "continuum_tables.npz",
        )
    ):
        return CONTINUUM_TABLE_UNITS[name]
    if path.endswith("chapter05_continuum_states.npz"):
        return chapter05_continuum_fixture_unit(name)
    if path.endswith("continuum_edge_grid.npz"):
        return CONTINUUM_EDGE_GRID_UNITS[name]
    if path.endswith("line_opacity_tables.npz"):
        return "dimensionless Harris source coefficient"
    if path.endswith("line_profile_tables.npz"):
        return {
            "hydrogen_continuum_edges": "cm^-1",
            "radiative_damping_sums": "s^-1",
            "impact_electron_density_thresholds_cm3": "cm^-3",
            "stark_knm_table": (
                "parity-pinned hydrogen Stark K_nm coefficient (source convention)"
            ),
            "stark_probability_table": "dimensionless Stark correction",
            "stark_wing_correction_c": ("dimensionless Stark-wing fit coefficient"),
            "stark_wing_correction_d": ("dimensionless Stark-wing fit coefficient"),
            "stark_pressure_grid": "dimensionless pressure parameter",
            "stark_beta_grid": "dimensionless Stark offset beta",
            "harris_profile_h0_table": "dimensionless Harris H0 coefficient",
            "harris_profile_h1_table": "dimensionless Harris H1 coefficient",
            "harris_profile_h2_table": "dimensionless Harris H2 coefficient",
        }[name]
    if "/golden/payne_zero/chapter05/" in path:
        return chapter05_golden_unit(path, name)
    if "/golden/payne_zero/chapter04/" in path:
        return chapter04_golden_unit(path, name)
    if "/golden/payne_zero/chapter03_" in path:
        return chapter03_golden_unit(path, name)
    if path.endswith("atomic_masses.npz"):
        return "amu"
    raise KeyError(f"no unit registry for {path}")


def npz_inventory(path: Path, relative_path: str) -> dict[str, dict[str, object]]:
    """Return every NPZ member's shape, dtype, and declared unit."""

    with np.load(path, allow_pickle=False) as archive:
        inventory = {}
        for name in archive.files:
            values = np.asarray(archive[name])
            member_metadata: dict[str, object] = {
                "shape": list(np.asarray(archive[name]).shape),
                "dtype": str(np.asarray(archive[name]).dtype),
                "unit": unit_for(relative_path, name),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(values).tobytes(order="C")
                ).hexdigest(),
            }
            if "/golden/payne_zero/chapter05/" in relative_path:
                member_metadata.update(
                    chapter05_golden_member_metadata(
                        relative_path,
                        name,
                        tuple(values.shape),
                    )
                )
            if relative_path.endswith("chapter06_fe_i_source_row_873702.npz"):
                if name in CHAPTER06_FE_SUBSET_RAW_FIELDS:
                    if values.shape != (1,):
                        raise RuntimeError(
                            f"{relative_path}:{name} must have shape (1,)"
                        )
                    subset_axes = ["source_record"]
                    subset_ownership = (
                        "Chapter 6 teaching subset owns the exact raw value "
                        "extracted from source row 873702"
                    )
                else:
                    if values.shape != ():
                        raise RuntimeError(
                            f"{relative_path}:{name} provenance must be scalar"
                        )
                    subset_axes = []
                    subset_ownership = (
                        "Chapter 6 teaching subset provenance; not a computed output"
                    )
                member_metadata.update(
                    {
                        "axes": subset_axes,
                        "ownership": subset_ownership,
                    }
                )
            if relative_path.endswith("line_opacity_tables.npz"):
                member_metadata.update(
                    {
                        "axes": ["coarse_harris_offset"],
                        "ownership": ("Chapter 6 atmosphere Harris source authority"),
                    }
                )
            elif relative_path.endswith("line_profile_tables.npz"):
                table_axes = {
                    "hydrogen_continuum_edges": ["hydrogen_lower_level"],
                    "radiative_damping_sums": ["hydrogen_transition"],
                    "impact_electron_density_thresholds_cm3": [
                        "threshold_row",
                        "threshold_column",
                    ],
                    "stark_knm_table": [
                        "lower_principal_quantum_number",
                        "level_separation",
                    ],
                    "stark_probability_table": [
                        "transition_family",
                        "stark_pressure_parameter",
                        "stark_beta",
                    ],
                    "stark_wing_correction_c": [
                        "stark_pressure_parameter",
                        "transition_family",
                    ],
                    "stark_wing_correction_d": [
                        "stark_pressure_parameter",
                        "transition_family",
                    ],
                    "stark_pressure_grid": ["stark_pressure_parameter"],
                    "stark_beta_grid": ["stark_beta"],
                    "harris_profile_h0_table": ["doppler_offset"],
                    "harris_profile_h1_table": ["doppler_offset"],
                    "harris_profile_h2_table": ["doppler_offset"],
                }
                harris_names = {
                    "harris_profile_h0_table",
                    "harris_profile_h1_table",
                    "harris_profile_h2_table",
                }
                member_metadata.update(
                    {
                        "axes": table_axes[name],
                        "ownership": (
                            "Chapter 6 synthesis Harris authority"
                            if name in harris_names
                            else (
                                "Byte-identical loader member; hydrogen-profile "
                                "consumption and teaching deferred to Chapter 7"
                            )
                        ),
                    }
                )
            inventory[name] = member_metadata
        return inventory


def _atomic_write_manifest(text: str) -> None:
    """Replace the manifest atomically after every validation has passed."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=MANIFEST_PATH.parent,
            prefix=f".{MANIFEST_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(
            temporary_path,
            MANIFEST_PATH.stat().st_mode & 0o777,
        )
        os.replace(temporary_path, MANIFEST_PATH)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(*, chapter05_archive_root: Path | None = None) -> None:
    """Update hashes, provenance, and exhaustive array inventories."""

    chapter05_specs = chapter05_golden_specs(chapter05_archive_root)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    chapter05_paths = {
        f"data/golden/payne_zero/chapter05/{name}" for name in CHAPTER05_GOLDEN_ARCHIVES
    }
    existing_chapter05_paths = [
        entry["path"]
        for entry in manifest["entries"]
        if entry.get("path") in chapter05_paths
    ]
    if len(existing_chapter05_paths) != len(set(existing_chapter05_paths)):
        raise RuntimeError("data manifest contains duplicate Chapter 5 golden entries")
    if not chapter05_specs and existing_chapter05_paths:
        raise RuntimeError(
            "data manifest contains Chapter 5 golden entries without both archives"
        )

    static_specs = (
        *CHAPTER04_ENTRY_SPECS,
        *CHAPTER05_ENTRY_SPECS,
        *CHAPTER06_ENTRY_SPECS,
        *CHAPTER09_ENTRY_SPECS,
        *CHAPTER11_ENTRY_SPECS,
        *chapter05_specs,
    )
    chapter04_specs_by_path = {
        specification["path"]: specification for specification in static_specs
    }
    entries_by_path = {entry["path"]: entry for entry in manifest["entries"]}
    for specification in static_specs:
        if specification["path"] in entries_by_path:
            continue
        entry = {
            **specification,
            "format": specification.get("format", "npz"),
            "source_commit": specification.get(
                "source_commit",
                PAYNE_ZERO_COMMIT,
            ),
            "requires_optional_full_catalog": specification.get(
                "requires_optional_full_catalog",
                False,
            ),
            "arrays": {},
        }
        manifest["entries"].append(entry)
        entries_by_path[entry["path"]] = entry
    for entry in manifest["entries"]:
        relative_path = entry["path"]
        path = (
            chapter05_archive_root / Path(relative_path).name
            if (chapter05_archive_root is not None and relative_path in chapter05_paths)
            else REPOSITORY_ROOT / relative_path
        )
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        actual_bytes = path.stat().st_size
        specification = chapter04_specs_by_path.get(relative_path)
        if relative_path in chapter05_paths and specification is not None:
            entry.update(specification)
        if specification is not None and specification["role"] == "subset":
            entry.update(specification)
        if specification is not None and specification["role"] in {
            "golden",
            "subset",
        }:
            if actual_sha256 != specification["sha256"]:
                raise RuntimeError(
                    f"{relative_path} has SHA-256 {actual_sha256}; "
                    f"expected {specification['sha256']}"
                )
            if actual_bytes != specification["bytes"]:
                raise RuntimeError(
                    f"{relative_path} has {actual_bytes} bytes; "
                    f"expected {specification['bytes']}"
                )
        entry["sha256"] = actual_sha256
        entry["bytes"] = actual_bytes
        if (
            relative_path.endswith("chapter06_atmosphere_one_line_inputs.npz")
            or "/golden/payne_zero/chapter06/" in relative_path
        ):
            # These separately published Chapter 6 artifacts carry reviewed
            # per-member axes/ownership metadata and are synchronized by their
            # own publishers. Preserve that richer inventory during general
            # manifest refreshes.
            pass
        elif path.suffix == ".npz":
            entry["arrays"] = npz_inventory(path, relative_path)
        elif relative_path.endswith("atmosphere_schema.json"):
            entry["declared_arrays"] = schema_inventory()

        if relative_path.endswith("chapter02_schema_v4_minimal.npz"):
            entry["builder"] = "scripts/build_chapter02_fixture.py"
        elif relative_path.endswith("chapter02_transfer_inputs.npz"):
            entry["builder"] = "scripts/build_chapter02_transfer_fixture.py"
        elif relative_path.endswith("chapter02_transfer_outputs.npz"):
            entry["builder"] = "scripts/build_chapter02_transfer_oracle.py"
        elif relative_path.endswith(
            (
                "partition_saha_tables.npz",
                "chapter03_synthesis_eos_state.npz",
            )
        ):
            entry["builder"] = "scripts/build_chapter03_synthesis_eos_data.py"
        elif relative_path.endswith("chapter03_atom_only_inputs.npz"):
            entry["builder"] = "scripts/build_chapter03_atom_only_fixture.py"
        elif relative_path.endswith("chapter04_molecular_inputs.npz"):
            entry["builder"] = "scripts/build_chapter04_molecular_fixture.py"
        elif relative_path.endswith("chapter05_continuum_states.npz"):
            entry["builder"] = "scripts/build_chapter05_continuum_fixture.py"
        elif relative_path.endswith("chapter06_fe_i_source_row_873702.npz"):
            entry["builder"] = CHAPTER06_FE_SUBSET_BUILDER
        elif relative_path.endswith(
            (
                "chapter11_solar_seed.npz",
                "chapter11_observed_atomic_subset.npy",
                "chapter11_observed_atomic_subset.provenance.npz",
                "chapter11_detailed_transition_subset.npz",
            )
        ):
            entry["builder"] = "scripts/build_chapter11_inputs.py"
        elif "/golden/payne_zero/chapter04/" in relative_path:
            entry["builder"] = CHAPTER04_GOLDEN_BUILDER
        elif relative_path in chapter05_paths:
            entry["builder"] = CHAPTER05_GOLDEN_BUILDER
        elif "/golden/payne_zero/chapter03_" in relative_path:
            entry["builder"] = "scripts/build_chapter03_payne_zero_goldens.py"

    final_chapter05_entries = [
        entry for entry in manifest["entries"] if entry.get("path") in chapter05_paths
    ]
    expected_chapter05_count = 2 if chapter05_specs else 0
    if len(final_chapter05_entries) != expected_chapter05_count:
        raise RuntimeError(
            "data manifest must contain exactly two Chapter 5 golden entries "
            "if and only if both accepted archives exist"
        )

    serialized = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if MANIFEST_PATH.read_text(encoding="utf-8") != serialized:
        _atomic_write_manifest(serialized)
    try:
        display_path = MANIFEST_PATH.relative_to(REPOSITORY_ROOT)
    except ValueError:
        display_path = MANIFEST_PATH
    print(f"updated {display_path}")


if __name__ == "__main__":
    main()
