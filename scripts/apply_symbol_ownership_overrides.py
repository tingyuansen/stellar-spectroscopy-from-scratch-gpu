#!/usr/bin/env python3
"""Apply the reviewed semantic symbol registry to the coverage ledger.

The module table in ``COVERAGE.md`` remains the first, exhaustive visibility
mapping.  This file is the narrower semantic layer for modules whose public
surface mixes physics, composition, compatibility, diagnostics, unsupported
branches, caches, and publication.  Every reviewed module is bound to the
pinned raw inventory, its source SHA-256, its complete public descriptor
surface, and an exact ``(qualified_name, kind)`` ledger snapshot.  Explicit
branch contracts additionally freeze source signatures/fields and
branch-sensitive defaults.
"""

from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPOSITORY_ROOT / "audit/paynezero_symbol_coverage.json"
INVENTORY_PATH = REPOSITORY_ROOT / "audit/paynezero_symbols.json"
PRECISION = "reviewed_symbol_override"
PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
RAW_INVENTORY_CONTENT_SHA256 = (
    "94861595dfe59afcbe2c47b23c19d14e25d0474a6727ad23e3c4f3948cc1b4b0"
)
SEMANTIC_DISPOSITIONS = frozenset(
    {
        "taught",
        "composed",
        "plumbing-only",
        "compatibility-only",
        "diagnostic-only",
        "unsupported",
    }
)


def policy(
    disposition: str,
    reason: str,
) -> dict[str, str]:
    """Return a module-review policy."""

    if disposition not in SEMANTIC_DISPOSITIONS:
        raise ValueError(f"invalid semantic disposition: {disposition}")
    return {
        "semantic_disposition": disposition,
        "semantic_review_reason": reason,
    }


# These modules have mixed or branch-sensitive public surfaces.  The digest is
# SHA-256 over sorted ``qualified_name NUL kind LF`` records at pinned commit
# 9c44001feae40b85146630499e6f8a5fed42e5af.  A module-level policy is applied
# only after its complete public snapshot matches this registry.
REVIEWED_MODULE_SNAPSHOTS = {
    "payne_zero_atmosphere._numba_cache": (
        2,
        "9ab316c9e558e136c63ae3852fd82f10567ec8b238b512519aa1d8711b8cffad",
    ),
    "payne_zero_atmosphere.atmosphere_io": (
        23,
        "424d18b6d7991ade28cf5d20777cd45e50ef11c87be536073be3058bb4038f1c",
    ),
    "payne_zero_atmosphere.cli": (
        2,
        "28db4f785672d54a3511c8f25b96ff8243f04c45f047e87e8f1f746049103f6d",
    ),
    "payne_zero_atmosphere.config": (
        29,
        "15f6f4eec995355c10712fcf38e79550e4184b852b40a7faa8ac4784af3f73c9",
    ),
    "payne_zero_atmosphere.convection": (
        27,
        "5b87f78b4786d4fa847802f8d1f969994a71f8037a52ef5bf1ac2e98605ee488",
    ),
    "payne_zero_atmosphere.data_files": (
        6,
        "ef106a6884caa6e79c26af0cf292d80ac4c1ed4f5fc50666cb1a4b3ef37bc8e6",
    ),
    "payne_zero_atmosphere.direct_abundance": (
        69,
        "7eb0892de638cc60db8ea1236f9d6d0e45cdb0336e13fbba980dfdcac4899996",
    ),
    "payne_zero_atmosphere.install_runtime_data": (
        10,
        "72198915cb4559b0e4c9f24b6335443b9c49548b22d751c0bc7314811e9a7a27",
    ),
    "payne_zero_atmosphere.line_opacity": (
        6,
        "692bebba32bb04a5a35ca6f5865e704de16db065b3d6837dfb356755cc13b8f5",
    ),
    "payne_zero_atmosphere.line_selection": (
        8,
        "75deb5aefd8c2061855eb0a23153484983fe04e86a4f2a070c554546e31f7f83",
    ),
    "payne_zero_atmosphere.microturbulence": (
        1,
        "5ab2681a77464ce1bdf77fb4a233fc5af75d45946ccd13c1601e9f470be06733",
    ),
    "payne_zero_atmosphere.prewarm": (
        5,
        "43fdea803e79a482c8233d33e94f68fa107877efc1a9bbbb58a6938addf5aee9",
    ),
    "payne_zero_atmosphere.radiative_transfer": (
        12,
        "acecbe825d82d295ae52d04848d681d52c7daaad2cc0f39bcd217cfb360db82f",
    ),
    "payne_zero_atmosphere.rosseland_mean": (
        1,
        "c03ed30a709992f478cf5eb94d7ba2b0d6cf8949d926e643f33cab4f3d8ec957",
    ),
    "payne_zero_atmosphere.run_setup": (
        36,
        "84bc8a1051c45219163b1910734687a0c5edef129f400c1e2a2fc2b7eeca6886",
    ),
    "payne_zero_atmosphere.runner": (
        60,
        "4c09e1a0c32626a1849b3875e1e7fa6954287912839a782d73325bc6c3b1babe",
    ),
    "payne_zero_atmosphere.source_catalogs": (
        11,
        "48a019f9daa19d35d0374d1ddc62825585d5a41ca3d3d599fa9448b5936add7e",
    ),
    "payne_zero_atmosphere.synthesis_bridge": (
        8,
        "283db1e75854ce9fd3584073a56a269f7e8fd0cbc28d221ed5e84ae22fe7b127",
    ),
    "payne_zero_atmosphere.transfer_kernels": (
        3,
        "fbf6a86ea1174a3fa0325517bca64e26b9819e7a0bbe04b07e77efcbed646071",
    ),
    "payne_zero_atmosphere.warm_start": (
        37,
        "0baccd2990e40798699d5e66f9653be5507c9d467a793f37989abc24426c2f2f",
    ),
    "payne_zero_synthesis.api": (
        36,
        "8a5caff5b386495fc83b6ed2a2a9cb8194448f2c485f01b1f3891f76fe195dd5",
    ),
    "payne_zero_synthesis.atmosphere": (
        11,
        "5f872d60c5b4a8143b9abc0e41cfe875a28e2bf77769961f7a6939914e569061",
    ),
    "payne_zero_synthesis.atomic_lines": (
        41,
        "e2e527ef91c428e62c973c9eaac7478f3761c929286747e96b8f625984ecf1b7",
    ),
    "payne_zero_synthesis.cli": (
        1,
        "ab2283e0187b590940d230c45c5cf819e4c1cf14ad3281a50f82ff50cd4b7eba",
    ),
    "payne_zero_synthesis.device": (
        7,
        "1b6e04a91cd4d906c6eefd0c7f404314cada57f3fcc8366f02dc04812b4057d7",
    ),
    "payne_zero_synthesis.equation_of_state": (
        71,
        "a99655e6f7c0eed93415e1990be7afc4950bd38a418742d6970c766d2f8496d7",
    ),
    "payne_zero_synthesis.ground_partition_table": (
        5,
        "95ee78ed5eb1edda3476030da707cb3d35bde61b8c7fd5f95a00d3b63d4ef662",
    ),
    "payne_zero_synthesis.line_opacity": (
        61,
        "3ee42e56be5cfbdefb72044aaacf534d75215e08591913df11069eef1970b0c3",
    ),
    "payne_zero_synthesis.molecular_lines": (
        54,
        "cb17aa3e3bfcf6f4250c7f3e16cb011bb3571c11f9beb5623878762f774767fb",
    ),
    "payne_zero_synthesis.paths": (
        10,
        "fbf8596fda5c6f69ddc13663f075227e9a51ce6e40b864185d09e8a211f2d4e1",
    ),
    "payne_zero_synthesis.pipeline": (
        46,
        "af560b419e86c599574724812964d84f62ecee5768a22199b07ce7d1bad9e279",
    ),
    "payne_zero_synthesis.prewarm": (
        3,
        "e36dcbe130226bc3e45c78ee2393f50f4e25876bf58af7a055299206282d3cae",
    ),
    "payne_zero_synthesis.radiative_transfer": (
        18,
        "7c6585e56bebccf69405cf5cdd5d73eceefc88e03b3f89079bf960a86ba2f63a",
    ),
    "payne_zero_synthesis.source_catalog_molecular_compiler": (
        7,
        "83bd58124e34f7563ae124dfebb2b0f49afd36e19636e5b5dd57e363e30279a4",
    ),
    "payne_zero_synthesis.synthesis": (
        3,
        "c29631639b8978fe11cb00ac28379cf1aa4e6f24f222467b3ca4b290be8b229c",
    ),
    # The independent P0.3 audit found that these were the only modules still
    # contributing module-default records.  Their complete surfaces are now
    # reviewed; mixed public entry points are narrowed again below.
    "payne_zero_atmosphere.constants": (
        12,
        "70af2c9489c22d6d10f190ff98a158482b0528aee1d81a68ade858a57fe9cf69",
    ),
    "payne_zero_atmosphere.continuum_opacity": (
        123,
        "676b65f89f4f2df8e0dd816888d7f5ff3653936f837803043b5002a27d010c12",
    ),
    "payne_zero_atmosphere.convergence": (
        3,
        "6054c79c0cff3aced4298f76b826fc1efe84a5e8223ed2d4b1f69edcca5e8a2a",
    ),
    "payne_zero_atmosphere.doppler": (
        1,
        "ae3d1d242e60ea6899c4a3950fba6ffe3fa1d9ee99f26c083b8713304655096f",
    ),
    "payne_zero_atmosphere.equation_of_state": (
        46,
        "366c137d91eb073192e0ec01194c9d7186626fa4b10fd0faf84acbe6143e1a4e",
    ),
    "payne_zero_atmosphere.hydrogen_line_profile": (
        66,
        "ec6a31529a8d093f9db6b370e10585911bb11b9bfb0aa9b3ad154a4578fd96e0",
    ),
    "payne_zero_atmosphere.hydrostatic": (
        2,
        "d6a4b518fba58cde340300f9c05969df89ea5529d5917820031bcb93e71304e6",
    ),
    "payne_zero_atmosphere.line_catalog": (
        28,
        "4944b27f78a32b09c5b34cf5ca48e64b1fd6a942cb227424c914f557141ebc9b",
    ),
    "payne_zero_atmosphere.line_profile_math": (
        19,
        "835e22862d8ff6f019b1df5009f35e9a4159d62c7d8a69bd8dc11df660bc254d",
    ),
    "payne_zero_atmosphere.molecular_data": (
        16,
        "03fd90cd36d25957c01dfcc017d956477853ec9e33cdd3920af9f361d89630d9",
    ),
    "payne_zero_atmosphere.molecular_equilibrium": (
        21,
        "a2a4fbf8347c1088b5c6bec0e1225255a445669bfe0db42a0b83a539fb934d47",
    ),
    "payne_zero_atmosphere.population_layout": (
        11,
        "61781934bebfe8fb3b3ba320b8b8a7d2c8e97ebc2a3ea4a830eadc945df5de41",
    ),
    "payne_zero_atmosphere.radiative_pressure": (
        9,
        "b00a8c1aea52482e944fb370cfcfb98c6f6eed73fb244affc3c29f054d139646",
    ),
    "payne_zero_atmosphere.runtime_state": (
        24,
        "fe9413a24703b6be058a5076e0e590289123351c049505d4b2afed99d70bf364",
    ),
    "payne_zero_atmosphere.specific_internal_energy": (
        1,
        "209ac5eac68e461309d86d1eac7abcc7042786e06cdf0952bd6d6c4f64cb7674",
    ),
    "payne_zero_atmosphere.temperature_correction": (
        23,
        "9d635b15d9d424b9bd9f23f4a730b32af74aae6e32bf12702b49265dc40e3757",
    ),
    "payne_zero_synthesis.constants": (
        18,
        "d055622bbadf842481fc86c7c08eb7be81098284d1ddb2e752f2f1989a6517cc",
    ),
    "payne_zero_synthesis.continuum": (
        60,
        "63cac755729ef9cf1396dddbd75889dc14540a08dced4fc5cf8175c22c019192",
    ),
    "payne_zero_synthesis.hydrogen_lines": (
        51,
        "4fd777398bd45c10e19aaeddc6f06356a8367eb7325211d4a59a9c5eb6ab2021",
    ),
    "payne_zero_synthesis.molecular_equilibrium": (
        39,
        "e5d9a7d057282669e6e1263f972b25d243f0c75bc8be3a32837aab0a0b0fff4e",
    ),
}

# Exact source bytes for every module whose public surface receives a semantic
# module policy.  This is separate from the ledger surface snapshot above:
# source-only changes (including defaults and branch behavior) must also stop
# the build.
REVIEWED_SOURCE_SHA256 = {
    "payne_zero_atmosphere._numba_cache": "b8988812ea92fd5db1e7f092d06ce685e13fbda3b0b7910eba937eb7a4ddeb82",
    "payne_zero_atmosphere.atmosphere_io": "95c4d2cab230f6925e9404639ecb05b25af8c0c85755ac1ca70d760156a8683e",
    "payne_zero_atmosphere.cli": "62974d80798faefbe91f76f2785543303d8d9875866adb9617d21d50e33416ac",
    "payne_zero_atmosphere.config": "51e19846fb81c832ae57334faf3da2c1e4fc2ef9edf6e08467ef7296e4640b45",
    "payne_zero_atmosphere.constants": "ac1f1fbd345dc816eb3e70a8f97ebebc7a4c744fd2759b32ec19f8c88d987036",
    "payne_zero_atmosphere.continuum_opacity": "1ff81cf6acd974b495f734a7c464faa3c25823e5957e301e1606af07258c0e81",
    "payne_zero_atmosphere.convection": "9099af3ce97123a88cfee554cefb55b2b47a52085e3cb6cda19e6869e0fef9fd",
    "payne_zero_atmosphere.convergence": "6b4c674deda148baab6fd90e8a25eed2921581ce7d4bace489c024bc1c2748cb",
    "payne_zero_atmosphere.data_files": "bf89c32977fc2db0454cf597718d99b3f3d15487529ecddbacf717ad6dc245c2",
    "payne_zero_atmosphere.direct_abundance": "ec65683eb344c4c3fd77340c084e780f58c6401e77c9f0d6db05ef6753131445",
    "payne_zero_atmosphere.doppler": "e118a78bf5250ef5e1f77d652c9e78fbb7b92acf5c069f717faed7a3b3ea98f0",
    "payne_zero_atmosphere.equation_of_state": "719b316327fd6fb76dfa5267a2c9022d1384c269a0fa20d5b2ec733671ae3fa2",
    "payne_zero_atmosphere.hydrogen_line_profile": "6a48f43afee9e326d2f86282f22f44f5654e243a335cc0490c99f86c41451be0",
    "payne_zero_atmosphere.hydrostatic": "f59f7b807152b74f1cf85ed208c612454aa82f62369f5d7baebe3d1a46740fef",
    "payne_zero_atmosphere.install_runtime_data": "aa1b12da92d1857e15f43e960c4760e62237772128fb0a7143c024ba865d7d18",
    "payne_zero_atmosphere.line_catalog": "2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92",
    "payne_zero_atmosphere.line_opacity": "d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6",
    "payne_zero_atmosphere.line_profile_math": "9a5794140f00ff3c3fb6c2e3b28461bbc22b471f962d275055c066ad7f8acd15",
    "payne_zero_atmosphere.line_selection": "b2c62fdf5e1fe43f33022184bfeff88985b13331354e3c745c7dab3a6b634fef",
    "payne_zero_atmosphere.microturbulence": "3692062f1d6877e745ed84bba4fc2fdf04c60a7c52bc27856fc696416a0283cb",
    "payne_zero_atmosphere.molecular_data": "705c3072d79c8019c948ce0fa2c82052f232816d453e10a7c8e5fc5a8f5ce249",
    "payne_zero_atmosphere.molecular_equilibrium": "4c9665148a57fadf4837f193c9f4247fbab44b788e7fb76ef3cae7ebdb3c3d86",
    "payne_zero_atmosphere.population_layout": "36736f3095bfd7ee173e159dfde5a37aac995557893292b7bac8e794286a2ef0",
    "payne_zero_atmosphere.prewarm": "061162d18b569a248f4dcf158b205ffc53369340f35f25b9d535f9615a138d66",
    "payne_zero_atmosphere.radiative_pressure": "c61a256892282d9a0d6cb19714ea5ce6135f1b6f7f573e5761d216d552f321ec",
    "payne_zero_atmosphere.radiative_transfer": "df8970ca629487537a7c4849278eab5d755b527002d8fc58360c9264a3aa45db",
    "payne_zero_atmosphere.rosseland_mean": "91071248fd903e05322b7163d37566e9f894daefc1d7ba018d4850d362f1fc86",
    "payne_zero_atmosphere.run_setup": "de7cf08b936585dbcfa2e572c026fafa3f10282a99c27b834b62db0f3f2888c9",
    "payne_zero_atmosphere.runner": "05bd3d9976b20dd83259b3d77a88fdd9b1262bb11bd342008e7e5115e797a2d7",
    "payne_zero_atmosphere.runtime_state": "fae240ec00f6f89d7c2a7ef721ce6e6539be234e523291fd6e8a096d731430e8",
    "payne_zero_atmosphere.source_catalogs": "a9ea21735c9d4964b785d76c89c9fc976a30ed75f8b6f9d4f7c6aaa4e77dae36",
    "payne_zero_atmosphere.specific_internal_energy": "de06ba732ce1333d111a52223e39f5b4f80eece8cfc4ff2f30de9739e16d7ec5",
    "payne_zero_atmosphere.synthesis_bridge": "142a960b5e710823754b02766803b3c1dd8c48c9945fdfabe560b4ee7e1acb50",
    "payne_zero_atmosphere.temperature_correction": "67728389ba857511979d0f82ea59f0bf41ee635b8151ae26673dace02b195d21",
    "payne_zero_atmosphere.transfer_kernels": "50e759a085e6aefdb7819a3dbe3ef5e83405834f4b07e0a4de2f3c0e7354d3b9",
    "payne_zero_atmosphere.warm_start": "3a83af3d68be52a35bfc3f55f5912770661be8251cc28f28da3250b2e83e0ad3",
    "payne_zero_synthesis.api": "77718303c1e0052a520ece7fab277b3b1922c21d09b35a288596592d03310940",
    "payne_zero_synthesis.atmosphere": "06b79770e4d9472093655022d53ee7fddf7cc6727206f34c0f60c57151e2cf9b",
    "payne_zero_synthesis.atomic_lines": "0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0",
    "payne_zero_synthesis.cli": "2cb2fa1cb71b5ec4a15dee9564b622c3e3288032ba13df59ae3355062b107dad",
    "payne_zero_synthesis.constants": "ed58004196790f9fb4a2871044c9cd36bf7bc42046a923f9314f7b8ea7456798",
    "payne_zero_synthesis.continuum": "ab0d4eb771ee04101f6936253f633ed60d845e2816854a06b1b059e8b91dce1b",
    "payne_zero_synthesis.device": "22e769ebed60ad3a0f2060264247e469a99afd20ec5cadb69a01b6e5fa82ea3c",
    "payne_zero_synthesis.equation_of_state": "6497c29abb954e0b55d918cc22fa7b660952812c548faf1d7b1053345ef13562",
    "payne_zero_synthesis.ground_partition_table": "6950686c89ea51e301b4b11256d9413dad58d82741a51580f3547aa012ade832",
    "payne_zero_synthesis.hydrogen_lines": "81ab3ee2ca9ecd1994ddde8f01e09535c5b74f7beec5afe98a3c63b44677dcca",
    "payne_zero_synthesis.line_opacity": "639b95c3812f1a7d227b797fa89a4d6ef9725d5f0e1284f3d49cf86844278275",
    "payne_zero_synthesis.molecular_equilibrium": "df01757c160b2bff4390cc2148cff9d1ba6e5a2bc7cab4515b46f38e868d2714",
    "payne_zero_synthesis.molecular_lines": "14c9d07e431fa73e6d6938e9db2d11c6688e52348234e0aac37cc76e8be3dc32",
    "payne_zero_synthesis.paths": "2bca3284eb1765449ab3fc87439eb603e3941213d9c1205c71aee3fd1ad30b5d",
    "payne_zero_synthesis.pipeline": "465118980d73cbf549d29ee3f33adf82788708cc2b286e5dddb8eb288c933f22",
    "payne_zero_synthesis.prewarm": "d8831e4dd342a979d261325185460b564db78c9b4dc7e908e0cb9cb9c8e5ca86",
    "payne_zero_synthesis.radiative_transfer": "52e0d1a0c4a2713294ce1b43130c5d900e54c4cf1f8b2b05058fc2d6831ff62b",
    "payne_zero_synthesis.source_catalog_molecular_compiler": "b3e64b36f76228f9602490a927d7443dd701a36098573365da33e008436f633a",
    "payne_zero_synthesis.synthesis": "590e430b6582fbcf601a52b721d8f65073432903773a99238073a1d821fe0d0c",
}


REVIEWED_MODULE_POLICIES = {
    "payne_zero_atmosphere._numba_cache": policy(
        "plumbing-only", "persistent Numba-cache location and configuration"
    ),
    "payne_zero_atmosphere.atmosphere_io": policy(
        "compatibility-only",
        "fixed-column external deck compatibility wrapped around ModelAtmosphere",
    ),
    "payne_zero_atmosphere.cli": policy(
        "composed", "high-level physical solve and CLI publication workflow"
    ),
    "payne_zero_atmosphere.config": policy(
        "plumbing-only", "typed inputs, outputs, controls, and branch selectors"
    ),
    "payne_zero_atmosphere.convection": policy(
        "taught", "production convection reductions and returned state"
    ),
    "payne_zero_atmosphere.data_files": policy(
        "plumbing-only", "runtime table path and array loading boundary"
    ),
    "payne_zero_atmosphere.direct_abundance": policy(
        "taught", "direct-abundance initializer, provenance, and opt-in surrogate"
    ),
    "payne_zero_atmosphere.install_runtime_data": policy(
        "plumbing-only", "manifest-bound runtime-data installation"
    ),
    "payne_zero_atmosphere.line_opacity": policy(
        "taught", "atmosphere line allocation and deposition"
    ),
    "payne_zero_atmosphere.line_selection": policy(
        "taught", "atomic and molecular source-family selection"
    ),
    "payne_zero_atmosphere.microturbulence": policy(
        "taught", "standard atmosphere microturbulence prescription"
    ),
    "payne_zero_atmosphere.prewarm": policy(
        "diagnostic-only", "cache specialization and fresh-process verification"
    ),
    "payne_zero_atmosphere.radiative_transfer": policy(
        "taught", "depth-grid integration, remapping, and transfer tables"
    ),
    "payne_zero_atmosphere.rosseland_mean": policy(
        "taught", "production Rosseland mean and optical-depth step"
    ),
    "payne_zero_atmosphere.run_setup": policy(
        "plumbing-only", "seed normalization and physical-branch validation"
    ),
    "payne_zero_atmosphere.runner": policy(
        "composed", "physical-pass staging, iteration, convergence, and publication"
    ),
    "payne_zero_atmosphere.source_catalogs": policy(
        "plumbing-only", "catalog paths, status, checksums, and file validation"
    ),
    "payne_zero_atmosphere.synthesis_bridge": policy(
        "composed", "structured atmosphere schema assembly and publication"
    ),
    "payne_zero_atmosphere.transfer_kernels": policy(
        "taught", "compiled and parallel transfer accumulation kernels"
    ),
    "payne_zero_atmosphere.warm_start": policy(
        "taught", "learned initializer transforms, support, decode, and provenance"
    ),
    "payne_zero_synthesis.api": policy(
        "composed", "public atmosphere, initializer, and spectrum workflows"
    ),
    "payne_zero_synthesis.atmosphere": policy(
        "compatibility-only", "canonical schema loader plus bounded older aliases"
    ),
    "payne_zero_synthesis.atomic_lines": policy(
        "taught", "atomic grid, catalog decode, cache, and device conversion"
    ),
    "payne_zero_synthesis.cli": policy(
        "composed", "public label/archive command aliases and product workflow"
    ),
    "payne_zero_synthesis.device": policy(
        "taught", "CUDA/MPS/CPU and work/accumulation dtype contract"
    ),
    "payne_zero_synthesis.equation_of_state": policy(
        "taught", "Torch populations, electron closure, bridge, and table schemas"
    ),
    "payne_zero_synthesis.ground_partition_table": policy(
        "taught", "ordered ground-state partition corrections"
    ),
    "payne_zero_synthesis.line_opacity": policy(
        "taught", "ordinary, helium, and special atomic opacity deposits"
    ),
    "payne_zero_synthesis.molecular_lines": policy(
        "taught", "molecular catalog, invariants, chunking, cache, and deposits"
    ),
    "payne_zero_synthesis.paths": policy(
        "plumbing-only", "runtime table, source-catalog, and cache roots"
    ),
    "payne_zero_synthesis.pipeline": policy(
        "composed", "window invariants, structured bridge, and complete forward pass"
    ),
    "payne_zero_synthesis.prewarm": policy(
        "diagnostic-only", "window-cache prewarm and provenance"
    ),
    "payne_zero_synthesis.radiative_transfer": policy(
        "taught", "thermal source, optical-depth integration, and scattering transfer"
    ),
    "payne_zero_synthesis.source_catalog_molecular_compiler": policy(
        "taught", "text, TiO, and compiler-only H2O source conversion"
    ),
    "payne_zero_synthesis.synthesis": policy(
        "composed", "structured-atmosphere engine boundary"
    ),
    "payne_zero_atmosphere.constants": policy(
        "taught", "named physical constants and exact/reference value policy"
    ),
    "payne_zero_atmosphere.continuum_opacity": policy(
        "taught", "continuum state, tables, absorbers, scattering, and sampling grid"
    ),
    "payne_zero_atmosphere.convergence": policy(
        "composed", "declared fixed-point convergence measurements and limits"
    ),
    "payne_zero_atmosphere.doppler": policy(
        "taught", "atmosphere Doppler population-ratio update"
    ),
    "payne_zero_atmosphere.equation_of_state": policy(
        "taught", "atmosphere Saha populations, partition tables, and electron closure"
    ),
    "payne_zero_atmosphere.hydrogen_line_profile": policy(
        "taught", "atmosphere hydrogen-line setup and profile evaluation"
    ),
    "payne_zero_atmosphere.hydrostatic": policy(
        "taught", "hydrostatic gas, radiation, and turbulent-pressure balance"
    ),
    "payne_zero_atmosphere.line_catalog": policy(
        "plumbing-only", "selected and transition line-catalog binary boundaries"
    ),
    "payne_zero_atmosphere.line_profile_math": policy(
        "taught", "fast exponential and Voigt profile mathematics"
    ),
    "payne_zero_atmosphere.molecular_data": policy(
        "plumbing-only", "molecular-equilibrium source records and table schema"
    ),
    "payne_zero_atmosphere.molecular_equilibrium": policy(
        "taught", "atmosphere molecular equilibrium and state continuation"
    ),
    "payne_zero_atmosphere.population_layout": policy(
        "taught", "packed population slots and species work schedule"
    ),
    "payne_zero_atmosphere.radiative_pressure": policy(
        "taught", "radiative acceleration and pressure accumulation"
    ),
    "payne_zero_atmosphere.runtime_state": policy(
        "composed", "abundance, isotope, charge, and reusable runtime state"
    ),
    "payne_zero_atmosphere.specific_internal_energy": policy(
        "taught", "atomic specific-internal-energy reduction"
    ),
    "payne_zero_atmosphere.temperature_correction": policy(
        "taught", "frequency accumulation and atmosphere temperature correction"
    ),
    "payne_zero_synthesis.constants": policy(
        "taught", "named synthesis constants and precision policy"
    ),
    "payne_zero_synthesis.continuum": policy(
        "taught", "device continuum populations, invariants, opacity, and scattering"
    ),
    "payne_zero_synthesis.hydrogen_lines": policy(
        "taught", "Balmer and merged-series hydrogen opacity"
    ),
    "payne_zero_synthesis.molecular_equilibrium": policy(
        "taught", "Torch molecular equilibrium, diagnostics, and line populations"
    ),
}


def override(
    primary: str,
    *supporting: str,
    disposition: str,
    responsibility: str,
    gate: str,
    status: str,
) -> dict[str, Any]:
    """Return one complete explicit semantic override."""

    if disposition not in SEMANTIC_DISPOSITIONS:
        raise ValueError(f"invalid semantic disposition: {disposition}")
    return {
        "primary_location": primary,
        "supporting_locations": list(dict.fromkeys(supporting)),
        "semantic_disposition": disposition,
        "responsibility": responsibility,
        "gate": gate,
        "status": status,
    }


EXPLICIT_OVERRIDES: dict[str, dict[str, Any]] = {
    # Accepted Chapter 1/2 radiative primitives and the exact CPU parallel spine.
    "payne_zero_synthesis.radiative_transfer.planck_bnu": override(
        "chapter-1",
        "chapter-9",
        "chapter-10",
        disposition="taught",
        responsibility="exact thermal Planck source B_nu on [depth, wavelength]",
        gate="accepted Chapter 1 exact-source and render gate",
        status="integrated",
    ),
    "payne_zero_synthesis.radiative_transfer.integrate_optical_depth": override(
        "chapter-2",
        "chapter-9",
        "chapter-10",
        disposition="taught",
        responsibility="Torch wavelength-batched parabolic optical-depth integral",
        gate="accepted Chapter 2 exact integration parity",
        status="integrated",
    ),
    "payne_zero_atmosphere.radiative_transfer.parabolic_coefficients": override(
        "chapter-2",
        "chapter-9",
        disposition="taught",
        responsibility="shared one-dimensional parabolic interval coefficients",
        gate="accepted Chapter 2 coefficient/integral parity",
        status="integrated",
    ),
    "payne_zero_atmosphere.radiative_transfer.integrate_on_depth_grid": override(
        "chapter-2",
        "chapter-9",
        "chapter-12",
        disposition="taught",
        responsibility="NumPy float64 monotonic depth-grid parabolic integral",
        gate="accepted Chapter 2 exact integration parity",
        status="integrated",
    ),
    "payne_zero_atmosphere.transfer_kernels.accumulate_transfer_range_parallel": (
        override(
            "chapter-2",
            "chapter-9",
            "chapter-12",
            "chapter-13",
            disposition="taught",
            responsibility="prange transfer spine with chunk-private state and fixed reduction",
            gate="accepted Chapter 2 serial/parallel identity and thread matrix",
            status="integrated",
        )
    ),
    # Accepted Chapter 3 ground correction and structured bridge.
    **{
        f"payne_zero_synthesis.ground_partition_table.{name}": override(
            "chapter-3",
            disposition="taught",
            responsibility="ordered ground-state partition correction",
            gate="accepted Chapter 3 scalar/vector table parity",
            status="integrated",
        )
        for name in (
            "FIRST_RANGE_LABELS",
            "SECOND_RANGE_LABELS",
            "GROUND_PARTITION_TERMS",
            "ground_partition_value",
            "ground_partition_values",
        )
    },
    "payne_zero_synthesis.pipeline.load_atomic_masses": override(
        "chapter-3",
        "chapter-10",
        "appendix-b",
        disposition="plumbing-only",
        responsibility="99-element mass-table loader used by population and width bridges",
        gate="accepted Chapter 3 table identity",
        status="integrated",
    ),
    "payne_zero_synthesis.pipeline.compute_doppler_per_ion": override(
        "chapter-3",
        "chapter-6",
        "chapter-10",
        disposition="taught",
        responsibility="thermal plus microturbulent fractional Doppler widths",
        gate="accepted Chapter 3 exact width parity; Chapter 6 one-line reuse",
        status="integrated",
    ),
    "payne_zero_synthesis.pipeline.build_structured_atmosphere_from_columns": (
        override(
            "chapter-3",
            "chapter-4",
            "chapter-10",
            disposition="composed",
            responsibility="fixed-electron-density structured population-column bridge",
            gate="accepted Chapters 3/4 structured-column and molecular-state parity",
            status="integrated",
        )
    ),
    "payne_zero_synthesis.synthesis.build_structured_atmosphere_from_columns": (
        override(
            "chapter-3",
            "chapter-4",
            "chapter-10",
            disposition="composed",
            responsibility="public engine alias for the structured population-column bridge",
            gate="accepted Chapters 3/4 wrapper identity",
            status="integrated",
        )
    ),
    "payne_zero_synthesis.api.build_structured_atmosphere": override(
        "chapter-3",
        "chapter-4",
        "chapter-10",
        disposition="composed",
        responsibility="public structured-atmosphere builder from converged columns",
        gate="accepted Chapters 3/4 API bridge identity",
        status="integrated",
    ),
    # Chapter 4 molecular-state corrections retained from the first review.
    "payne_zero_synthesis.equation_of_state.molecular_seed_electron_density": (
        override(
            "chapter-4",
            "chapter-3",
            "chapter-10",
            disposition="taught",
            responsibility="molecular fixed-point electron-density seed",
            gate="accepted Chapter 4 atmosphere/synthesis molecular-state parity",
            status="integrated",
        )
    ),
    (
        "payne_zero_synthesis.equation_of_state."
        "molecular_ion_formation_constants_from_seed"
    ): override(
        "chapter-4",
        "chapter-3",
        "chapter-10",
        disposition="taught",
        responsibility="ion formation constants tied to the molecular seed state",
        gate="accepted Chapter 4 molecular formation-constant parity",
        status="integrated",
    ),
    (
        "payne_zero_synthesis.equation_of_state.PopulationState.molecular_populations"
    ): override(
        "chapter-4",
        "chapter-10",
        disposition="taught",
        responsibility="returned synthesis molecular populations by depth",
        gate="accepted Chapter 4 population-state schema and value parity",
        status="integrated",
    ),
    (
        "payne_zero_synthesis.equation_of_state."
        "PopulationState.molecular_equation_densities"
    ): override(
        "chapter-4",
        "chapter-10",
        disposition="taught",
        responsibility="returned molecular equation-density state",
        gate="accepted Chapter 4 state schema and continuation parity",
        status="integrated",
    ),
    (
        "payne_zero_atmosphere.runner.AtmospherePopulationState.molecular_state"
    ): override(
        "chapter-4",
        "chapter-11",
        "chapter-12",
        "chapter-13",
        "chapter-15",
        disposition="composed",
        responsibility="accepted molecular-equilibrium state carried through a physical pass",
        gate="accepted Chapter 4 state identity; later pass composition",
        status="integrated",
    ),
    "payne_zero_atmosphere.source_catalogs.molecular_equilibrium_catalog_path": (
        override(
            "chapter-4",
            "chapter-2",
            "chapter-11",
            "appendix-b",
            disposition="plumbing-only",
            responsibility="atmosphere molecular-equilibrium catalog path",
            gate="accepted Chapter 4 catalog identity and Appendix B provenance",
            status="integrated",
        )
    ),
    "payne_zero_atmosphere.config.AtmosphereInput.molecules_path": override(
        "chapter-4",
        "chapter-11",
        "chapter-13",
        "chapter-15",
        disposition="plumbing-only",
        responsibility="molecular-equilibrium catalog input, distinct from line catalogs",
        gate="accepted Chapter 4 catalog/state integration",
        status="integrated",
    ),
    "payne_zero_atmosphere.config.AtmosphereConfig.enable_molecules": override(
        "chapter-4",
        "chapter-11",
        "chapter-13",
        "chapter-15",
        disposition="composed",
        responsibility="molecular-equilibrium branch switch",
        gate="accepted Chapter 4 on/off state parity",
        status="integrated",
    ),
    "payne_zero_atmosphere.run_setup.RunSetup.molecules_enabled": override(
        "chapter-4",
        "chapter-11",
        "chapter-13",
        "chapter-15",
        disposition="composed",
        responsibility="resolved molecular-equilibrium branch state",
        gate="accepted Chapter 4 resolved-state identity",
        status="integrated",
    ),
    (
        "payne_zero_atmosphere.config.AtmosphereConfig."
        "molecular_convection_thermal_tracks_perturbation"
    ): override(
        "chapter-12",
        "chapter-11",
        "chapter-13",
        disposition="composed",
        responsibility="molecular thermal-state choice for convection perturbations",
        gate="Chapter 12 perturbation-state and convection derivative parity",
        status="planned",
    ),
    **{
        f"payne_zero_synthesis.pipeline.WindowInvariants.{name}": override(
            "chapter-8",
            "chapter-10",
            disposition="composed",
            responsibility="compiled molecular window invariant consumed by the full pipeline",
            gate="Chapter 8 molecular invariant parity; Chapter 10 composition",
            status="planned",
        )
        for name in ("molecular_lines", "molecular_invariants", "n_molecular")
    },
    **{
        f"payne_zero_atmosphere.molecular_data.{name}": override(
            "chapter-4",
            "appendix-b",
            disposition="taught",
            responsibility="accepted atmosphere molecular-equilibrium catalog boundary",
            gate="accepted Chapter 4 catalog parse/identity gate",
            status="integrated",
        )
        for name in (
            "MolecularEquilibriumCatalog",
            "find_default_molecular_equilibrium_catalog",
            "parse_molecular_equilibrium_record",
            "read_molecular_equilibrium_catalog",
        )
    },
    **{
        f"payne_zero_atmosphere.molecular_equilibrium.{name}": override(
            "chapter-4",
            "chapter-12",
            disposition="taught",
            responsibility="accepted atmosphere coupled molecular-equilibrium state",
            gate="accepted Chapter 4 state/energy/continuation parity",
            status="integrated",
        )
        for name in (
            "MolecularEquilibriumState",
            "compute_equilibrium_constants_for_layer",
            "compute_molecular_specific_internal_energy",
            "initialize_molecular_equilibrium_state",
            "populate_molecular_species",
            "restore_molecular_equation_density",
            "save_molecular_equation_density",
            "set_molecular_specific_internal_energy_mode",
            "solve_molecular_equilibrium",
            "solve_molecular_equilibrium_layer",
        )
    },
    # Chapter 5 molecular continuum and Chapter 7 hydrogen-profile consumers.
    **{
        f"payne_zero_atmosphere.continuum_opacity.{name}": override(
            "chapter-5",
            "chapter-11",
            disposition="taught",
            responsibility="accepted molecular continuum implementation",
            gate="accepted Chapter 5 component/slab parity",
            status="integrated",
        )
        for name in (
            "MolecularEquilibriumTables",
            "compute_molecular_continuum_opacity_columns",
            "compute_molecular_hydrogen_ion_opacity_columns",
            "compute_molecular_hydrogen_population",
            "load_molecular_equilibrium_tables",
        )
    },
    **{
        f"payne_zero_atmosphere.hydrogen_line_profile.{name}": override(
            "chapter-7",
            "chapter-11",
            disposition="taught",
            responsibility="hydrogen-profile molecular population helper",
            gate="Chapter 7 profile/deposit parity",
            status="planned",
        )
        for name in (
            "compute_hydrogen_molecule_population",
            "molecular_hydrogen_equilibrium_constant",
        )
    },
    # Production Rosseland ownership and microturbulence prescription/use split.
    "payne_zero_atmosphere.rosseland_mean.rosseland_mean_step": override(
        "chapter-12",
        "chapter-1",
        disposition="taught",
        responsibility="production harmonic Rosseland mean and optical-depth update",
        gate="Chapter 12 mean/depth parity; Chapter 1 concept only",
        status="planned",
    ),
    "payne_zero_atmosphere.microturbulence.standard_microturbulence": override(
        "chapter-11",
        "chapter-3",
        "chapter-6",
        disposition="taught",
        responsibility="standard atmosphere microturbulence profile, distinct from width use",
        gate="Chapter 11 profile parity; Chapters 3/6 consume supplied velocity",
        status="planned",
    ),
    # Atmosphere molecular source families and their shared deposit.
    **{
        f"payne_zero_atmosphere.config.AtmosphereInput.{name}": override(
            "chapter-7",
            "chapter-11",
            disposition="plumbing-only",
            responsibility="ordinary atomic source-catalog input",
            gate="Chapter 7 source decode/selection identity",
            status="planned",
        )
        for name in (
            "predicted_atomic_lines_path",
            "observed_atomic_lines_path",
            "high_excitation_lines_path",
            "selected_line_catalog_path",
            "detailed_line_catalog_path",
        )
    },
    **{
        f"payne_zero_atmosphere.config.AtmosphereInput.{name}": override(
            "chapter-8",
            "chapter-11",
            "appendix-b",
            disposition="taught",
            responsibility=responsibility,
            gate=gate,
            status="planned",
        )
        for name, responsibility, gate in (
            (
                "diatomic_lines_path",
                "standard converted atmosphere diatomic source input",
                "Chapter 8 atmosphere diatomic selection/deposit parity",
            ),
            (
                "titanium_oxide_lines_path",
                "standard converted atmosphere TiO source input",
                "Chapter 8 atmosphere TiO selection/deposit parity",
            ),
            (
                "water_lines_path",
                "standard converted atmosphere H2O source input",
                "Chapter 8 atmosphere water selection/deposit parity",
            ),
            (
                "h3plus_lines_path",
                "explicit opt-in atmosphere H3+ source; absent from default source mapping",
                "Chapter 8 opt-in status and Appendix B source provenance",
            ),
        )
    },
    "payne_zero_atmosphere.line_selection.read_diatomic_line_catalog": override(
        "chapter-8",
        "chapter-11",
        disposition="taught",
        responsibility="atmosphere diatomic/TiO converted-catalog reader",
        gate="Chapter 8 family decode and selected-word identity",
        status="planned",
    ),
    "payne_zero_atmosphere.line_selection.read_water_line_catalog": override(
        "chapter-8",
        "chapter-11",
        disposition="taught",
        responsibility="atmosphere water converted-catalog reader",
        gate="Chapter 8 water decode and selected-word identity",
        status="planned",
    ),
    "payne_zero_atmosphere.line_selection.select_water_line_words": override(
        "chapter-8",
        "chapter-11",
        disposition="taught",
        responsibility="atmosphere H2O line selector",
        gate="Chapter 8 water keep-test and selected-word parity",
        status="planned",
    ),
    "payne_zero_atmosphere.line_selection.generate_selected_lines": override(
        "chapter-8",
        "chapter-7",
        "chapter-11",
        disposition="composed",
        responsibility=(
            "fresh in-memory selector for atomic, diatomic, TiO, standard water, "
            "and opt-in H3+ source paths"
        ),
        gate="Chapters 7/8 per-family counts, bytes, and no-disk-cache identity",
        status="planned",
    ),
    "payne_zero_atmosphere.source_catalogs.source_line_paths": override(
        "chapter-8",
        "chapter-7",
        "chapter-11",
        "appendix-b",
        disposition="plumbing-only",
        responsibility=(
            "default atomic/diatomic/TiO/water mapping; H3+ is deliberately "
            "not a default key"
        ),
        gate="Chapter 8 two-lane family/status matrix and path identity",
        status="planned",
    ),
    "payne_zero_atmosphere.source_catalogs.atmosphere_source_catalog_paths": (
        override(
            "chapter-11",
            "chapter-7",
            "chapter-8",
            "appendix-b",
            disposition="composed",
            responsibility="complete default physical-pass source set",
            gate="Chapter 11 pass input manifest with Chapter 8 family statuses",
            status="planned",
        )
    ),
    "payne_zero_atmosphere.line_opacity.accumulate_selected_line_opacity": override(
        "chapter-7",
        "chapter-6",
        "chapter-8",
        "chapter-11",
        disposition="taught",
        responsibility="shared atomic and molecular selected-record opacity deposit",
        gate="Chapters 7/8 per-family deposit and thread parity",
        status="planned",
    ),
    "payne_zero_atmosphere.runner.prepare_opacity_state": override(
        "chapter-11",
        "chapter-7",
        "chapter-8",
        disposition="composed",
        responsibility=(
            "compose continuum with selected/detailed line catalogs; family "
            "semantics remain owned by Chapters 7/8"
        ),
        gate="Chapter 11 one-pass identity plus Chapters 7/8 branch gates",
        status="planned",
    ),
    # Synthesis molecular family status: text and TiO run; H2O is compiler-only.
    **{
        f"payne_zero_synthesis.source_catalog_molecular_compiler.{name}": override(
            "chapter-8",
            "appendix-b",
            disposition="taught",
            responsibility=responsibility,
            gate=gate,
            status="planned",
        )
        for name, responsibility, gate in (
            (
                "compile_molecular_text",
                (
                    "standard synthesis text-band compiler using exact "
                    "resolution spelling and cache boundary"
                ),
                "Chapter 8 scalar/Numba compilation and cache equality",
            ),
            (
                "compile_tio_schwenke",
                (
                    "standard synthesis TiO compiler using exact resolution "
                    "spelling and cache boundary"
                ),
                "Chapter 8 TiO decode/compile/runtime parity",
            ),
            (
                "compile_h2o_partridge",
                (
                    "working H2O compiler using exact resolution spelling; "
                    "not wired into the standard synthesis pipeline"
                ),
                "Chapter 8 compiler-only identity and explicit runtime-absence gate",
            ),
        )
    },
    # Full pipeline: exact source_path, keep_slabs, spectral_operator, and caches.
    "payne_zero_synthesis.pipeline.SynthesisPipeline": override(
        "chapter-10",
        "chapter-7",
        "chapter-8",
        disposition="composed",
        responsibility="complete device-resident synthesis from a supplied atmosphere",
        gate="Chapter 10 stage/full-spectrum/device parity",
        status="planned",
    ),
    "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__": override(
        "chapter-10",
        "chapter-7",
        "chapter-8",
        "appendix-d",
        disposition="composed",
        responsibility=(
            "complete window/device initialization; source_path is a "
            "comparison-only reference-source input, while standard LTE "
            "rebuilds Planck line source and zero line scattering"
        ),
        gate="Chapter 10 initialization, source rebuild/reference, and cache parity",
        status="planned",
    ),
    "payne_zero_synthesis.pipeline.SynthesisPipeline.run": override(
        "chapter-10",
        "chapter-9",
        "appendix-d",
        disposition="composed",
        responsibility=(
            "complete forward pass; keep_slabs is diagnostic-only and "
            "spectral_operator is interface-only before normalization/host transfer"
        ),
        gate=(
            "Chapter 10 full spectrum, default no-slab, diagnostic slab, crop, "
            "and joint total/continuum operator-order parity"
        ),
        status="planned",
    ),
    **{
        f"payne_zero_synthesis.pipeline.SpectrumResult.{name}": override(
            "chapter-10",
            "chapter-15",
            disposition="composed",
            responsibility="standard cropped public spectrum result",
            gate="Chapter 10 output shape/value/device-to-host parity",
            status="planned",
        )
        for name in (
            "wavelength_nm",
            "eddington_flux_total_per_frequency",
            "eddington_flux_continuum_per_frequency",
            "normalized_flux",
        )
    },
    "payne_zero_synthesis.pipeline.SpectrumResult": override(
        "chapter-10",
        "chapter-15",
        disposition="composed",
        responsibility="standard spectrum plus explicitly optional diagnostic fields",
        gate="Chapter 10 result schema, crop, and host-transfer parity",
        status="planned",
    ),
    **{
        f"payne_zero_synthesis.pipeline.SpectrumResult.{name}": override(
            "chapter-10",
            "appendix-d",
            disposition="diagnostic-only",
            responsibility="optional keep_slabs diagnostic host return",
            gate="Chapter 10 default-None and requested-slab crop/host-transfer gate",
            status="boundary",
        )
        for name in (
            "continuum_absorption",
            "continuum_scattering",
            "line_mass_absorption_coefficient",
            "line_source",
        )
    },
    **{
        f"payne_zero_synthesis.pipeline.SpectrumResult.{name}": override(
            "chapter-10",
            "appendix-d",
            disposition="diagnostic-only",
            responsibility="optional spectral-operator timing/name metadata",
            gate="Chapter 10 operator-order metadata gate",
            status="boundary",
        )
        for name in ("spectral_operator_seconds", "spectral_operator_name")
    },
    **{
        f"payne_zero_synthesis.pipeline.WindowInvariants.{name}": override(
            "chapter-10",
            disposition="composed",
            responsibility="complete synthesis window/device/cache invariant",
            gate="Chapter 10 invariant identity, crop, cache, and device parity",
            status="planned",
        )
        for name in (
            "key",
            "device",
            "dtype",
            "metal_chunk",
            "grid_obj",
            "synthesis_wavelength_nm",
            "wavelength_nm",
            "output_slice",
            "n_synthesis_wl",
            "n_wl",
            "continuum_tables",
            "transfer_tables",
        )
    },
    **{
        f"payne_zero_synthesis.pipeline.WindowInvariants.{name}": override(
            "chapter-7",
            "chapter-10",
            disposition="composed",
            responsibility="atomic/H/He forest invariant consumed by the full pipeline",
            gate="Chapter 7 family invariant parity; Chapter 10 composition",
            status="planned",
        )
        for name in (
            "n_atomic",
            "atomic_kernel_catalog",
            "has_metal",
            "has_helium",
            "has_hydrogen",
            "metal_invariant_chunks",
            "helium_invariants",
            "hydrogen_invariants_template",
        )
    },
    "payne_zero_synthesis.pipeline.WindowInvariants": override(
        "chapter-10",
        "chapter-7",
        "chapter-8",
        disposition="composed",
        responsibility="all atmosphere-independent window invariants and family subobjects",
        gate="Chapters 7/8 component parity and Chapter 10 cache composition",
        status="planned",
    ),
    "payne_zero_synthesis.pipeline.WindowInvariants.build_profile": override(
        "chapter-10",
        "appendix-d",
        disposition="diagnostic-only",
        responsibility="window-invariant build timing/profile diagnostic",
        gate="Chapter 10 diagnostic schema; never a physics acceptance value",
        status="boundary",
    ),
    "payne_zero_synthesis.pipeline.WINDOW_CONTEXT_SAMPLES": override(
        "chapter-10",
        disposition="taught",
        responsibility="symmetric internal native-grid context width",
        gate="Chapter 10 edge invariance and crop-before-host-transfer gate",
        status="planned",
    ),
    **{
        f"payne_zero_synthesis.pipeline.{name}": override(
            "chapter-10",
            "appendix-b",
            disposition="plumbing-only",
            responsibility="window-invariant device-cache control",
            gate="Chapter 10 cache on/off and cold/warm equality",
            status="planned",
        )
        for name in (
            "window_invariant_cache_enabled",
            "clear_window_invariant_cache",
            "window_invariants_for",
        )
    },
    # Public synthesis boundaries that carry the interface-only operator.
    **{
        f"payne_zero_synthesis.{module}.{name}": override(
            primary,
            *supporting,
            disposition="composed",
            responsibility=responsibility,
            gate=gate,
            status="planned",
        )
        for module, name, primary, supporting, responsibility, gate in (
            (
                "api",
                "synthesize",
                "chapter-10",
                ("appendix-c", "appendix-d"),
                (
                    "public archive/in-memory synthesis using resolution; "
                    "spectral_operator is an interface-only joint flux boundary"
                ),
                "Chapter 10 API/output and joint-operator order parity",
            ),
            (
                "api",
                "synthesize_from_labels",
                "chapter-15",
                ("chapter-10", "chapter-14", "appendix-c"),
                (
                    "label workflow with canonical r_grid, conflict-checked "
                    "resolution alias, and interface-only spectral_operator"
                ),
                "Chapter 15 end-to-end API alias/conflict/output parity",
            ),
            (
                "synthesis",
                "synthesize_structured_atmosphere",
                "chapter-10",
                ("chapter-15", "appendix-d"),
                (
                    "engine boundary with source_path=None, keep_slabs=False, "
                    "and optional spectral_operator forwarding"
                ),
                "Chapter 10 engine/default-boundary parity",
            ),
        )
    },
    "payne_zero_synthesis.atomic_lines.load_catalog": override(
        "chapter-7",
        "appendix-b",
        disposition="composed",
        responsibility="source-bound atomic catalog load with optional compiled-cache reuse",
        gate="Chapter 7 fresh/cache equality and stale-cache rejection",
        status="planned",
    ),
    "payne_zero_synthesis.molecular_lines.load_catalog": override(
        "chapter-8",
        "appendix-b",
        disposition="composed",
        responsibility="source-bound molecular catalog load with optional compiled-cache reuse",
        gate="Chapter 8 fresh/cache equality and stale-cache rejection",
        status="planned",
    ),
    # Readable/parity wing loop and restricted host accumulator live inside one symbol.
    "payne_zero_synthesis.line_opacity.accumulate_atomic": override(
        "chapter-7",
        "chapter-6",
        "chapter-10",
        "appendix-d",
        disposition="taught",
        responsibility=(
            "standard batched atomic deposit; wing_mode='loop' is parity-only "
            "and host_accumulator is restricted to unstimulated metal-only diagnostics"
        ),
        gate="Chapter 7 batched/loop parity and host-accumulator rejection matrix",
        status="planned",
    ),
    "payne_zero_synthesis.radiative_transfer.source_and_alpha": override(
        "chapter-9",
        "chapter-10",
        disposition="taught",
        responsibility=(
            "combine continuum/line extinction, scattering fraction, and "
            "thermal source after the standard LTE line source is supplied"
        ),
        gate="Chapter 9 source/extinction decomposition parity",
        status="planned",
    ),
    "payne_zero_synthesis.radiative_transfer.solve_spectrum": override(
        "chapter-9",
        "chapter-10",
        "appendix-d",
        disposition="taught",
        responsibility=(
            "stack total/continuum transfer; strict direct calls reject "
            "saturated surface cores while the standard pipeline explicitly "
            "uses the implemented continuation"
        ),
        gate="Chapter 9 total/continuum parity and saturated-core branch matrix",
        status="planned",
    ),
    "payne_zero_synthesis.device.resolve_runtime": override(
        "chapter-2",
        "chapter-10",
        "appendix-d",
        disposition="taught",
        responsibility=(
            "CUDA then MPS then CPU policy; fp32 on MPS, fp64 on CUDA/CPU, "
            "and explicit MPS-float64 rejection"
        ),
        gate=(
            "accepted Chapter 2 device/dtype policy; unavailable hardware is "
            "reported unavailable rather than passed"
        ),
        status="integrated",
    ),
    "payne_zero_synthesis.atomic_lines.Grid.resolution": override(
        "chapter-7",
        "chapter-10",
        "appendix-c",
        disposition="taught",
        responsibility="exact resolution field for the intrinsic geometric grid",
        gate="Chapter 7 grid field/spacing identity",
        status="planned",
    ),
    # Exact runner guards: only the three implemented rejection paths are claimed.
    "payne_zero_atmosphere.config.AtmosphereConfig.iterations": override(
        "chapter-13",
        "chapter-11",
        disposition="composed",
        responsibility="requested pass count; values below one are explicitly rejected",
        gate="_require_supported_run_setup iterations < 1 rejection",
        status="planned",
    ),
    "payne_zero_atmosphere.run_setup.RunSetup.iterations": override(
        "chapter-13",
        "chapter-11",
        disposition="composed",
        responsibility="resolved pass count; fewer than one is unsupported",
        gate="_require_supported_run_setup iterations < 1 rejection",
        status="planned",
    ),
    **{
        f"payne_zero_atmosphere.run_setup.{name}": override(
            "chapter-13",
            "chapter-11",
            disposition="unsupported",
            responsibility="turbulent-pressure configuration rejected by the exact runner",
            gate="_require_supported_run_setup setup.turbulence.enabled rejection",
            status="boundary",
        )
        for name in (
            "TurbulenceSettings",
            "TurbulenceSettings.enabled",
            "TurbulenceSettings.density_coefficient",
            "TurbulenceSettings.density_power",
            "TurbulenceSettings.sound_speed_fraction",
            "TurbulenceSettings.constant_velocity_km_s",
            "RunSetup.turbulence",
        )
    },
    "payne_zero_atmosphere.run_setup.opacity_flags_from_atmosphere": override(
        "chapter-11",
        "chapter-13",
        disposition="composed",
        responsibility="decode 20 opacity flags; flag 13 HLINOP is rejected later",
        gate="Chapter 11 decode identity and Chapter 13 flag-13 rejection",
        status="planned",
    ),
    "payne_zero_atmosphere.config.DEFAULT_OPACITY_FLAGS": override(
        "chapter-11",
        "chapter-13",
        disposition="plumbing-only",
        responsibility="standard opacity flag vector; HLINOP flag 13 is unsupported",
        gate="default identity and explicit flag-13 rejection",
        status="planned",
    ),
    "payne_zero_atmosphere.run_setup.resolve_run_setup": override(
        "chapter-11",
        "chapter-13",
        disposition="composed",
        responsibility=(
            "normalize run state; exact runner guards iterations, turbulent "
            "pressure, and HLINOP rather than every absent legacy spelling"
        ),
        gate="setup validation plus evidence-specific unsupported guard matrix",
        status="planned",
    ),
    "payne_zero_atmosphere.runner.run_atmosphere_model": override(
        "chapter-13",
        "chapter-11",
        "chapter-12",
        "chapter-15",
        disposition="composed",
        responsibility=(
            "full physical iteration and conditional publication; explicitly "
            "rejects iterations<1, turbulent pressure, and HLINOP flag 13"
        ),
        gate="trajectory/convergence/publication plus exact three-guard matrix",
        status="planned",
    ),
    # Diagnostics, debug handoffs, and publication are never mistaken for physics.
    "payne_zero_atmosphere.config.AtmosphereOutput.diagnostics_path": override(
        "appendix-d",
        "chapter-13",
        disposition="plumbing-only",
        responsibility=(
            "declared diagnostics API path; the pinned runner never writes this field"
        ),
        gate="source-use audit proves declaration-only and no filesystem product",
        status="boundary",
    ),
    "payne_zero_atmosphere.config.AtmosphereOutput.debug_state_path": override(
        "chapter-13",
        "appendix-d",
        disposition="diagnostic-only",
        responsibility="actual opt-in debug-state NPZ publication target",
        gate="Chapter 13 opt-in write/no-write and debug-state schema gate",
        status="boundary",
    ),
    "payne_zero_atmosphere.config.AtmosphereOutput.structured_atmosphere_path": (
        override(
            "chapter-13",
            "chapter-15",
            disposition="composed",
            responsibility="converged-product publication target",
            gate="Chapter 13 converged-only atomic publication gate",
            status="planned",
        )
    ),
    "payne_zero_atmosphere.runner.AtmosphereRunResult.diagnostics": override(
        "chapter-13",
        "appendix-d",
        disposition="diagnostic-only",
        responsibility="terminal iteration and publication diagnostics",
        gate="Chapter 13 diagnostic schema and failure-path gate",
        status="boundary",
    ),
    "payne_zero_atmosphere.runner.AtmosphereRunResult.iterations_completed": (
        override(
            "chapter-13",
            "appendix-d",
            disposition="diagnostic-only",
            responsibility="terminal iteration-count diagnostic",
            gate="Chapter 13 trajectory count identity",
            status="boundary",
        )
    ),
    "payne_zero_atmosphere.runner.AtmospherePopulationState.temperature_iteration_cache": (
        override(
            "chapter-13",
            "appendix-d",
            disposition="diagnostic-only",
            responsibility="iteration-local temperature-key cache",
            gate="Chapter 13 cache reuse/reset identity",
            status="boundary",
        )
    ),
    **{
        f"payne_zero_atmosphere.convection.{name}": override(
            "chapter-12",
            "appendix-d",
            disposition="diagnostic-only",
            responsibility="zero-flux diagnostic returned when convection is disabled",
            gate="Chapter 12 disabled-branch zero diagnostic identity",
            status="boundary",
        )
        for name in (
            "compute_disabled_convection_diagnostics",
            "DisabledConvectionDiagnostics",
            "DisabledConvectionDiagnostics.convective_flux",
            "DisabledConvectionDiagnostics.convective_velocity",
        )
    },
    **{
        f"payne_zero_atmosphere.synthesis_bridge.{name}": override(
            "chapter-13",
            "appendix-d",
            disposition="diagnostic-only",
            responsibility="debug-schema conversion or diagnostic publication",
            gate="Chapter 13 debug schema/version and no-product-confusion gate",
            status="boundary",
        )
        for name in (
            "structured_atmosphere_from_runtime_state",
            "structured_atmosphere_from_debug_npz",
            "save_structured_atmosphere_from_runtime_state",
            "save_structured_atmosphere_from_debug_npz",
        )
    },
    "payne_zero_atmosphere.synthesis_bridge.structured_atmosphere_from_packed_state": (
        override(
            "chapter-3",
            "chapter-4",
            "chapter-10",
            disposition="composed",
            responsibility="canonical packed physical-state to synthesis-schema bridge",
            gate="accepted Chapters 3/4 schema and population identity",
            status="integrated",
        )
    ),
    "payne_zero_atmosphere.synthesis_bridge.save_structured_atmosphere": override(
        "chapter-10",
        "chapter-13",
        "appendix-c",
        disposition="plumbing-only",
        responsibility="canonical structured-atmosphere NPZ serializer",
        gate="schema-v4 byte/value round trip",
        status="planned",
    ),
    "payne_zero_atmosphere.synthesis_bridge.save_product_structured_atmosphere": (
        override(
            "chapter-13",
            "chapter-15",
            disposition="composed",
            responsibility="publish only final fixed-column-quantized converged arrays",
            gate="Chapter 13 converged-only schema-v4 publication identity",
            status="planned",
        )
    ),
    # Canonical schema versus compatibility-only aliases and serializers.
    **{
        f"payne_zero_synthesis.atmosphere.{name}": override(
            "chapter-2",
            "chapter-10",
            disposition="taught",
            responsibility="canonical structured-atmosphere schema-v4 boundary",
            gate="accepted Chapter 2 schema/axis/dtype validation",
            status="integrated",
        )
        for name in (
            "ATMOSPHERE_SCHEMA_VERSION",
            "POPULATION_ION_STAGE_COUNT",
            "POPULATION_SPECIES_COUNT",
            "REQUIRED_ATMOSPHERE_ARRAYS",
            "load_atmosphere_npz",
            "validate_atmosphere_npz",
        )
    },
    **{
        f"payne_zero_synthesis.atmosphere.{name}": override(
            "chapter-13",
            "chapter-10",
            "appendix-c",
            disposition="plumbing-only",
            responsibility="optional converged-product provenance metadata schema",
            gate="Chapter 13 product metadata schema/identity",
            status="planned",
        )
        for name in (
            "ATMOSPHERE_PRODUCT_METADATA_SCHEMA_VERSION",
            "ATMOSPHERE_PRODUCT_METADATA_FIELDS",
            "load_atmosphere_product_metadata",
        )
    },
    **{
        f"payne_zero_synthesis.atmosphere.{name}": override(
            "appendix-c",
            "appendix-d",
            disposition="compatibility-only",
            responsibility="bounded older atmosphere schema/field alias input",
            gate="exact compatibility round-trip and rejection matrix",
            status="boundary",
        )
        for name in (
            "LEGACY_ATMOSPHERE_SCHEMA_VERSIONS",
            "LEGACY_ATMOSPHERE_ARRAY_ALIASES",
        )
    },
    **{
        f"payne_zero_atmosphere.atmosphere_io.{name}": override(
            "appendix-c",
            "appendix-d",
            "chapter-11",
            disposition="compatibility-only",
            responsibility="external fixed-column atmosphere-deck compatibility",
            gate="exact parse/format/quantization round trip",
            status="boundary",
        )
        for name in (
            "parse_atmosphere_deck",
            "read_atmosphere_deck",
            "format_atmosphere_deck",
            "write_atmosphere_deck",
        )
    },
    **{
        f"payne_zero_atmosphere.atmosphere_io.{name}": override(
            "chapter-2",
            "chapter-11",
            disposition="taught",
            responsibility="structured atmosphere state and exact derived coordinates",
            gate="accepted Chapter 2 field/unit/axis identity",
            status="integrated",
        )
        for name in (
            "ModelAtmosphere",
            "ModelAtmosphere.column_mass",
            "ModelAtmosphere.temperature",
            "ModelAtmosphere.gas_pressure",
            "ModelAtmosphere.electron_density",
            "ModelAtmosphere.microturbulence",
            "ModelAtmosphere.layers",
            "ModelAtmosphere.thermal_energy_erg",
            "ModelAtmosphere.h_over_kt",
            "ModelAtmosphere.hc_over_kt",
            "ModelAtmosphere.thermal_energy_ev",
            "ModelAtmosphere.natural_log_temperature",
        )
    },
    # Catalog cache/schema members are appendix plumbing, not repeated physics.
    **{
        f"payne_zero_synthesis.{module}.{name}": override(
            "appendix-b",
            f"chapter-{chapter}",
            disposition=disposition,
            responsibility=responsibility,
            gate=gate,
            status="boundary",
        )
        for module, chapter, name, disposition, responsibility, gate in (
            (
                "atomic_lines",
                7,
                "CACHE_SCHEMA",
                "plumbing-only",
                "atomic catalog cache schema",
                "cache identity/invalidation",
            ),
            (
                "atomic_lines",
                7,
                "CACHE_LOGIC_VERSION",
                "plumbing-only",
                "atomic catalog cache logic version",
                "cache identity/invalidation",
            ),
            (
                "atomic_lines",
                7,
                "LineCatalog.to_npz",
                "plumbing-only",
                "atomic compiled-catalog cache serializer",
                "cache round trip",
            ),
            (
                "atomic_lines",
                7,
                "LineCatalog.from_npz",
                "plumbing-only",
                "current atomic compiled-catalog cache-hit loader",
                "current-schema round trip and stale-cache rejection",
            ),
            (
                "molecular_lines",
                8,
                "CACHE_SCHEMA",
                "plumbing-only",
                "molecular catalog cache schema",
                "cache identity/invalidation",
            ),
            (
                "molecular_lines",
                8,
                "CACHE_LOGIC_VERSION",
                "plumbing-only",
                "molecular catalog cache logic version",
                "cache identity/invalidation",
            ),
            (
                "molecular_lines",
                8,
                "MolecularLineCatalog.to_npz",
                "plumbing-only",
                "molecular compiled-catalog cache serializer",
                "cache round trip",
            ),
            (
                "molecular_lines",
                8,
                "MolecularLineCatalog.from_npz",
                "plumbing-only",
                "current molecular compiled-catalog cache-hit loader",
                "current-schema round trip and stale-cache rejection",
            ),
            (
                "molecular_lines",
                8,
                "MolecularLineCatalog.from_mapping",
                "plumbing-only",
                "current in-memory molecular catalog constructor",
                "modern mapping schema and value identity",
            ),
        )
    },
    # Cache/prewarm and installation surfaces stay outside the causal physics.
    **{
        f"payne_zero_atmosphere._numba_cache.{name}": override(
            "chapter-2",
            "chapter-13",
            "appendix-b",
            disposition="plumbing-only",
            responsibility="stable Numba cache path/configuration",
            gate="accepted Chapter 2 cold/warm equality and cache-path identity",
            status="integrated",
        )
        for name in ("default_numba_cache_dir", "configure_numba_cache")
    },
    # API serialization and timing are publication/diagnostic surfaces.
    "payne_zero_synthesis.api.save_structured_atmosphere": override(
        "chapter-10",
        "chapter-13",
        "appendix-c",
        disposition="plumbing-only",
        responsibility="canonical structured-atmosphere NPZ serializer/validator",
        gate="schema-v4 round trip",
        status="planned",
    ),
    "payne_zero_synthesis.api.Spectrum.save_npz": override(
        "chapter-15",
        "appendix-c",
        disposition="plumbing-only",
        responsibility="public spectrum product serializer",
        gate="Chapter 15 output schema/round-trip identity",
        status="planned",
    ),
    "payne_zero_synthesis.api.InitializedAtmosphere.save_npz": override(
        "chapter-14",
        "appendix-c",
        disposition="plumbing-only",
        responsibility="initializer-state serializer with mandatory-closure metadata",
        gate="Chapter 14 provenance/schema round trip",
        status="planned",
    ),
    "payne_zero_synthesis.api.initialize_atmosphere_from_labels": override(
        "chapter-14",
        "chapter-15",
        disposition="composed",
        responsibility=(
            "select five-label, CNO8, or opt-in direct-abundance initializer "
            "and build a synthesis-ready but explicitly unconverged state"
        ),
        gate="Chapter 14 family routing, support, provenance, and mandatory-closure gate",
        status="planned",
    ),
    **{
        f"payne_zero_synthesis.api.{name}": override(
            "chapter-14",
            "chapter-15",
            disposition="composed",
            responsibility="initializer result explicitly distinguishes prediction from closure",
            gate="Chapter 14 false-converged/true-closure-required invariant",
            status="planned",
        )
        for name in (
            "InitializedAtmosphere",
            "InitializedAtmosphere.atmosphere_converged",
            "InitializedAtmosphere.atmosphere_closure_required",
            "LabelSpectrum.atmosphere_converged",
            "LabelSpectrum.atmosphere_closure_required",
        )
    },
    **{
        f"payne_zero_atmosphere.direct_abundance.{name}": override(
            "chapter-14",
            "appendix-d",
            disposition="composed",
            responsibility=responsibility,
            gate=gate,
            status="planned",
        )
        for name, responsibility, gate in (
            (
                "load_direct_abundance_initializer",
                "hash-bound experimental initializer requiring explicit opt-in",
                "Chapter 14 opt-in, checkpoint schema/hash, and support gate",
            ),
            (
                "build_direct_abundance_optimizer_surrogate",
                (
                    "explicitly opt-in unconverged optimizer surrogate; no "
                    "fitting implementation and exact atmosphere closure remains required"
                ),
                "Chapter 14 opt-in/provenance/mandatory-closure gate",
            ),
            (
                "direct_abundance_warm_start_deck",
                "opt-in quantized starting deck; never a final atmosphere",
                "Chapter 14 deck quantization and non-product gate",
            ),
            (
                "run_direct_abundance_atmosphere",
                "opt-in route that returns only after mandatory exact physical closure",
                "Chapter 14 convergence-required and insufficient-iteration rejection gate",
            ),
        )
    },
    "payne_zero_atmosphere.warm_start.warm_start_supported": override(
        "chapter-14",
        disposition="taught",
        responsibility="initializer support-domain predicate, not physical convergence",
        gate="Chapter 14 support-boundary and out-of-support rejection",
        status="planned",
    ),
    "payne_zero_atmosphere.warm_start.load_atmosphere_initializer": override(
        "chapter-14",
        "appendix-b",
        disposition="composed",
        responsibility=(
            "standard complete-state initializer loader used by support, "
            "deterministic-label, and emulator routes"
        ),
        gate="Chapter 14 checkpoint schema, family, coordinate, and provenance gate",
        status="planned",
    ),
    **{
        f"payne_zero_synthesis.api.{name}": override(
            "chapter-15",
            "appendix-d",
            disposition="diagnostic-only",
            responsibility="stage and end-to-end wall-time diagnostic",
            gate="timing presence/finite/nonnegative check, never numerical acceptance",
            status="boundary",
        )
        for name in (
            "ForwardTimings",
            "ForwardTimings.initializer_seconds",
            "ForwardTimings.population_bridge_seconds",
            "ForwardTimings.synthesis_seconds",
            "ForwardTimings.total_seconds",
            "Spectrum.seconds",
            "InitializedAtmosphere.timings",
            "LabelSpectrum.timings",
        )
    },
    "payne_zero_synthesis.source_catalog_molecular_compiler.logger": override(
        "appendix-d",
        disposition="diagnostic-only",
        responsibility="molecular compiler cache/progress logger",
        gate="diagnostic logging smoke; never a physics gate",
        status="boundary",
    ),
    # Independent P0.3 residual review: optional surrogate, diagnostic,
    # compatibility, unsupported, cache-control, and correction branches.
    **{
        f"payne_zero_atmosphere.continuum_opacity.{name}": override(
            "chapter-5",
            "appendix-d",
            "chapter-12",
            disposition="diagnostic-only",
            responsibility=(
                "default-off IFOP-19 Rosseland-opacity surrogate, distinct "
                "from standard continuum and the Chapter 12 production mean"
            ),
            gate=(
                "Chapter 5 explicit opt-in surrogate identity and "
                "Chapter 12 non-production-owner rejection"
            ),
            status="boundary",
        )
        for name in (
            "RosselandOpacityTable",
            "create_rosseland_opacity_table",
            "ingest_rosseland_opacity_table",
            "evaluate_rosseland_opacity",
            "compute_rosseland_continuum_opacity_columns",
        )
    },
    "payne_zero_synthesis.continuum.build_frequency_invariants": override(
        "chapter-5",
        "appendix-d",
        disposition="taught",
        responsibility=(
            "frequency-only continuum invariants; "
            "coulomb_table_energy_first=False is standard and True is the "
            "sampled diagnostic orientation"
        ),
        gate="accepted Chapter 5 False/True orientation parity and diagnostic separation",
        status="integrated",
    ),
    (
        "payne_zero_synthesis.molecular_equilibrium.solve_molecular_equilibrium"
    ): override(
        "chapter-4",
        "appendix-d",
        disposition="taught",
        responsibility=(
            "molecular equilibrium; return_diagnostics=False returns four "
            "physical arrays and True appends iteration/log-formation structure"
        ),
        gate="accepted Chapter 4 four-value/diagnostic-return schema and value parity",
        status="integrated",
    ),
    "payne_zero_atmosphere.line_catalog.decode_selected_line_words": override(
        "chapter-7",
        "appendix-c",
        disposition="plumbing-only",
        responsibility=(
            "selected-line decoder with detect_swapped_layout=True external "
            "compatibility detection; freshly generated native words pass False"
        ),
        gate="Chapter 7 native/swapped decode identity and generated-native fast-path gate",
        status="planned",
    ),
    "payne_zero_synthesis.hydrogen_lines.precompute_invariants": override(
        "chapter-7",
        "appendix-d",
        disposition="taught",
        responsibility=(
            "Balmer and merged-series invariant builder with an explicit "
            "NotImplementedError guard for unsupported Lyman records"
        ),
        gate="Chapter 7 Balmer parity plus exact n_lower<2 Lyman rejection",
        status="planned",
    ),
    "payne_zero_synthesis.hydrogen_lines.accumulate_hydrogen": override(
        "chapter-7",
        "appendix-d",
        disposition="taught",
        responsibility=(
            "hydrogen opacity with apply_stim=True standard and False "
            "stimulated-emission removal for diagnostic parity"
        ),
        gate="Chapter 7 stimulated/unstimulated factor identity",
        status="planned",
    ),
    **{
        f"payne_zero_atmosphere.equation_of_state.{name}": override(
            "chapter-3",
            "appendix-d",
            disposition="taught",
            responsibility=(
                "scalar or depth-batched Saha/partition evaluation with "
                "explicit unsupported population-mode guards"
            ),
            gate="accepted Chapter 3 scalar/batch parity and exact unsupported-mode rejection",
            status="integrated",
        )
        for name in ("saha_partition_depth", "saha_partition_depth_batch")
    },
    "payne_zero_atmosphere.line_profile_math.load_line_opacity_tables": override(
        "chapter-6",
        "appendix-b",
        disposition="plumbing-only",
        responsibility=(
            "current line-profile table loader with optional path and "
            "force_reload=False cache-control branch"
        ),
        gate="Chapter 6 default/explicit path and cold/warm/forced-reload identity",
        status="planned",
    ),
    "payne_zero_atmosphere.temperature_correction.apply_temperature_correction": (
        override(
            "chapter-12",
            "chapter-13",
            "appendix-d",
            disposition="taught",
            responsibility=(
                "frequency accumulation and final correction across mode, "
                "iteration, convection, smoothing, pressure, and remap branches"
            ),
            gate="Chapter 12 exact mode/iteration/convection/smoothing/pressure/remap matrix",
            status="planned",
        )
    ),
    **{
        f"payne_zero_atmosphere.temperature_correction.{name}": override(
            "chapter-12",
            "appendix-b",
            disposition=disposition,
            responsibility=responsibility,
            gate=gate,
            status="planned",
        )
        for name, disposition, responsibility, gate in (
            (
                "initialize_temperature_correction_state",
                "taught",
                "zeroed mutable state for frequency accumulation and correction",
                "Chapter 12 state shape/dtype/zero initialization",
            ),
            (
                "ingest_temperature_correction_rosseland_table",
                "plumbing-only",
                "validated Rosseland table ingest for the correction state",
                "Chapter 12 table shape/value/identity gate",
            ),
        )
    },
    # Current table/cache constructors are required implementation plumbing.
    **{
        qualified_name: override(
            primary,
            "appendix-b",
            disposition="plumbing-only",
            responsibility=responsibility,
            gate=gate,
            status=status,
        )
        for qualified_name, primary, responsibility, gate, status in (
            (
                "payne_zero_atmosphere.continuum_opacity.load_continuum_opacity_tables",
                "chapter-5",
                "current packaged continuum-table loader",
                "Chapter 5 manifest/schema/value identity",
                "integrated",
            ),
            (
                "payne_zero_atmosphere.continuum_opacity.load_karzas_latter_tables",
                "chapter-5",
                "current packaged Karzas-Latter table loader",
                "Chapter 5 manifest/schema/value identity",
                "integrated",
            ),
            (
                "payne_zero_atmosphere.continuum_opacity.load_continuum_level_tables",
                "chapter-5",
                "current packaged continuum level-table loader",
                "Chapter 5 manifest/schema/value identity",
                "integrated",
            ),
            (
                "payne_zero_atmosphere.hydrogen_line_profile.load_hydrogen_line_profile_tables",
                "chapter-7",
                "current packaged hydrogen-profile table loader",
                "Chapter 7 default/explicit path and table identity",
                "planned",
            ),
            (
                "payne_zero_atmosphere.line_profile_math.load_hydrogen_continuum_selector_table",
                "chapter-6",
                "current hydrogen-continuum selector-table loader",
                "Chapter 6 default/explicit path and table identity",
                "planned",
            ),
            (
                "payne_zero_atmosphere.runtime_state.load_major_isotope_masses_amu",
                "chapter-11",
                "current major-isotope mass-table loader",
                "Chapter 11 default/explicit path and table identity",
                "planned",
            ),
            (
                "payne_zero_synthesis.continuum.ContinuumTables.from_npz",
                "chapter-5",
                "current synthesis continuum NPZ-table loader",
                "accepted Chapter 5 schema/device/dtype identity",
                "integrated",
            ),
            (
                "payne_zero_synthesis.continuum.ContinuumTables.from_dict",
                "chapter-5",
                "current in-memory synthesis continuum table constructor",
                "accepted Chapter 5 schema/device/dtype identity",
                "integrated",
            ),
            (
                "payne_zero_synthesis.molecular_equilibrium.read_molecule_table",
                "chapter-4",
                "current synthesis molecular-equilibrium table loader",
                "accepted Chapter 4 source/schema/value identity",
                "integrated",
            ),
        )
    },
    # Second independent P0.3 residual scan: public algorithm, family,
    # environment, cache, correction, and ordering switches.
    "payne_zero_synthesis.molecular_lines.accumulate_molecular": override(
        "chapter-8",
        "chapter-10",
        "appendix-d",
        disposition="taught",
        responsibility=(
            "molecular opacity with apply_stim=True standard and False "
            "stimulated-emission omission; chunk_lines=None resolves the "
            "environment policy while an explicit value is clamped and used"
        ),
        gate=(
            "Chapter 8 stimulated/unstimulated parity plus environment/default/"
            "explicit chunk-size and clamp identity"
        ),
        status="planned",
    ),
    "payne_zero_synthesis.molecular_lines.molecular_chunk_lines": override(
        "chapter-8",
        "appendix-c",
        disposition="plumbing-only",
        responsibility=(
            "PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES resolver with "
            "missing/empty/invalid fallback and positive-integer clamp"
        ),
        gate="Chapter 8 environment absent/empty/invalid/zero/positive matrix",
        status="planned",
    ),
    "payne_zero_atmosphere.continuum_opacity.compute_continuum_opacity_columns": (
        override(
            "chapter-5",
            "chapter-11",
            "appendix-d",
            disposition="taught",
            responsibility=(
                "full continuum assembler: opacity_flags=None enables the "
                "20-family default, individual flags select opacity families, "
                "and IFOP-19 contributes only when flag 19 is active and a "
                "Rosseland table is supplied"
            ),
            gate=(
                "accepted Chapter 5 default/short/full flag-family matrix plus "
                "both IFOP-19 activation conditions"
            ),
            status="integrated",
        )
    ),
    **{
        f"payne_zero_atmosphere.continuum_opacity.{name}": override(
            "chapter-5",
            "chapter-11",
            disposition="taught",
            responsibility=responsibility,
            gate=gate,
            status="integrated",
        )
        for name, responsibility, gate in (
            (
                "compute_light_element_continuum_columns",
                (
                    "H/He continuum assembler with opacity_flags=None "
                    "20-family default and per-family selection"
                ),
                "accepted Chapter 5 default/short/full light-family flag matrix",
            ),
            (
                "compute_continuum_scattering_columns",
                (
                    "continuum scattering assembler with flag-family selection "
                    "and optional molecular-equilibrium table injection"
                ),
                "accepted Chapter 5 scattering flags and molecular-table parity",
            ),
        )
    },
    (
        "payne_zero_synthesis.equation_of_state.partition_functions_for_elements"
    ): override(
        "chapter-3",
        "chapter-4",
        disposition="taught",
        responsibility=(
            "selected-element partitions with apply_ground_partition=True "
            "ordinary ground floor and False molecular-bridge parity path"
        ),
        gate="accepted Chapters 3/4 ground-floor on/off partition parity",
        status="integrated",
    ),
    **{
        f"payne_zero_synthesis.equation_of_state.{name}": override(
            "chapter-3",
            "chapter-4",
            "chapter-10",
            disposition=disposition,
            responsibility=responsibility,
            gate=gate,
            status="integrated",
        )
        for name, disposition, responsibility, gate in (
            (
                "solve_electron_density",
                "taught",
                (
                    "electron closure with molecules=False atom-only and "
                    "molecules=True molecular closure; molecules_path selects "
                    "the molecular catalog"
                ),
                "accepted Chapters 3/4 atom-only/molecular closure and catalog routing",
            ),
            (
                "solve_population_state",
                "composed",
                (
                    "full population assembly with atom-only versus molecular "
                    "closure and the documented two-solve molecular seam"
                ),
                "accepted Chapters 3/4 molecules on/off state and two-solve identity",
            ),
            (
                "solve_population_state_at_electron_density",
                "composed",
                (
                    "fixed-public-electron population assembly with optional "
                    "molecular state; molecular internal electron is discarded "
                    "and molecules_path selects the catalog"
                ),
                "accepted Chapters 3/4 fixed-ne atom/molecule and electron-preservation gate",
            ),
        )
    },
    "payne_zero_atmosphere.radiative_transfer.load_radiative_transfer_tables": (
        override(
            "chapter-9",
            "appendix-b",
            disposition="plumbing-only",
            responsibility=(
                "current transfer-table loader: path=None selects the packaged "
                "archive and force_reload=False reuses the validated warm cache"
            ),
            gate="Chapter 9 default/explicit path and cold/warm/forced-reload identity",
            status="planned",
        )
    ),
    "payne_zero_synthesis.atomic_lines.parse_catalog": override(
        "chapter-7",
        "chapter-10",
        disposition="taught",
        responsibility=(
            "atomic source parser: catalog_path=None selects the default source, "
            "apply_iso_corr=True applies isotope corrections, and sort selects "
            "catalog or wavelength ordering"
        ),
        gate="Chapter 7 default/explicit source, isotope on/off, and sort-order matrix",
        status="planned",
    ),
    # CLI aliases remain exact without moving physics into the CLI.
    "payne_zero_synthesis.cli.main": override(
        "chapter-15",
        "appendix-c",
        disposition="composed",
        responsibility=(
            "label/archive CLI; --r-grid and --resolution aliases both store "
            "the exact resolution field"
        ),
        gate="CLI alias/conflict/product smoke",
        status="planned",
    ),
    "payne_zero_atmosphere.cli.main": override(
        "chapter-13",
        "chapter-15",
        "appendix-c",
        disposition="composed",
        responsibility="physical-solve CLI and converged-product publication",
        gate="workflow/guard/publication smoke",
        status="planned",
    ),
}

# Chapter 5's binding symbol-disposition authority contains the complete
# 46-object continuum surface.  Keeping this exact set beside the overrides
# prevents an accepted chapter from being represented by a hand-picked sample.
CHAPTER5_ATMOSPHERE_SYMBOLS = (
    "ContinuumOpacityTableError",
    "ContinuumOpacityTables",
    "KarzasLatterTables",
    "ContinuumLevelTables",
    "MolecularEquilibriumTables",
    "RosselandOpacityTable",
    "ContinuumAtmosphereState",
    "load_continuum_opacity_tables",
    "load_karzas_latter_tables",
    "load_continuum_level_tables",
    "load_molecular_equilibrium_tables",
    "build_continuum_atmosphere_state",
    "build_continuum_reference_wavelength_grid",
    "active_continuum_reference_frequencies",
    "assemble_continuum_line_selection_threshold",
    "compute_molecular_hydrogen_population",
    "compute_hydrogen_opacity_columns",
    "compute_helium_neutral_opacity_columns",
    "compute_hminus_opacity_columns",
    "compute_molecular_hydrogen_ion_opacity_columns",
    "compute_heminus_opacity_columns",
    "compute_helium_ionized_opacity_columns",
    "compute_light_element_continuum_columns",
    "compute_carbon_neutral_opacity_columns",
    "compute_magnesium_neutral_opacity_columns",
    "compute_silicon_neutral_opacity_columns",
    "compute_lukewarm_metal_opacity_columns",
    "compute_molecular_continuum_opacity_columns",
    "compute_hot_metal_opacity_columns",
    "compute_aluminum_neutral_opacity_columns",
    "compute_iron_neutral_opacity_columns",
    "create_rosseland_opacity_table",
    "ingest_rosseland_opacity_table",
    "evaluate_rosseland_opacity",
    "compute_rosseland_continuum_opacity_columns",
    "compute_continuum_opacity_columns",
    "compute_continuum_scattering_columns",
    "build_opacity_sampling_grid",
)
CHAPTER5_SYNTHESIS_SYMBOLS = (
    "ContinuumTables",
    "FrequencyInvariants",
    "build_pops",
    "pops_from_population_state",
    "build_frequency_invariants",
    "build_edge_sample_frequencies",
    "compute_sampled_continuum",
    "continuum",
)
CHAPTER5_PLUMBING_SYMBOLS = frozenset(
    {
        "payne_zero_atmosphere.continuum_opacity.ContinuumOpacityTableError",
        "payne_zero_atmosphere.continuum_opacity.ContinuumLevelTables",
        "payne_zero_atmosphere.continuum_opacity.load_continuum_opacity_tables",
        "payne_zero_atmosphere.continuum_opacity.load_karzas_latter_tables",
        "payne_zero_atmosphere.continuum_opacity.load_continuum_level_tables",
        "payne_zero_atmosphere.continuum_opacity.load_molecular_equilibrium_tables",
    }
)
CHAPTER5_COMPOSED_SYMBOLS = frozenset(
    {
        "payne_zero_atmosphere.continuum_opacity.build_continuum_atmosphere_state",
        "payne_zero_synthesis.continuum.build_pops",
        "payne_zero_synthesis.continuum.pops_from_population_state",
        "payne_zero_synthesis.continuum.continuum",
    }
)
CHAPTER5_DIAGNOSTIC_SYMBOLS = frozenset(
    {
        "payne_zero_atmosphere.continuum_opacity.RosselandOpacityTable",
        "payne_zero_atmosphere.continuum_opacity.create_rosseland_opacity_table",
        "payne_zero_atmosphere.continuum_opacity.ingest_rosseland_opacity_table",
        "payne_zero_atmosphere.continuum_opacity.evaluate_rosseland_opacity",
        (
            "payne_zero_atmosphere.continuum_opacity."
            "compute_rosseland_continuum_opacity_columns"
        ),
    }
)
CHAPTER5_ACCEPTED_SYMBOLS = frozenset(
    {
        *(
            f"payne_zero_atmosphere.continuum_opacity.{name}"
            for name in CHAPTER5_ATMOSPHERE_SYMBOLS
        ),
        *(
            f"payne_zero_synthesis.continuum.{name}"
            for name in CHAPTER5_SYNTHESIS_SYMBOLS
        ),
    }
)

for _qualified_name in sorted(CHAPTER5_ACCEPTED_SYMBOLS):
    if _qualified_name in CHAPTER5_PLUMBING_SYMBOLS:
        _disposition = "plumbing-only"
    elif _qualified_name in CHAPTER5_COMPOSED_SYMBOLS:
        _disposition = "composed"
    elif _qualified_name in CHAPTER5_DIAGNOSTIC_SYMBOLS:
        _disposition = "diagnostic-only"
    else:
        _disposition = "taught"
    _supporting_location = (
        "chapter-11"
        if _qualified_name.startswith("payne_zero_atmosphere.")
        else "chapter-10"
    )
    _existing_override = EXPLICIT_OVERRIDES.get(_qualified_name)
    if _existing_override is None:
        EXPLICIT_OVERRIDES[_qualified_name] = override(
            "chapter-5",
            _supporting_location,
            "appendix-b",
            disposition=_disposition,
            responsibility=(
                "complete accepted Chapter 5 continuum surface; exact execution "
                "lane, physical owner, reader disposition, and rationale are "
                "bound to the symbol-disposition authority"
            ),
            gate=(
                "accepted Chapter 5 exhaustive symbol-disposition join and "
                "component/lane parity"
            ),
            status="integrated",
        )
        continue
    _supporting_locations = []
    for _location in (
        *_existing_override["supporting_locations"],
        _supporting_location,
        "appendix-b",
    ):
        if _location != "chapter-5" and _location not in _supporting_locations:
            _supporting_locations.append(_location)
    EXPLICIT_OVERRIDES[_qualified_name] = {
        **_existing_override,
        "primary_location": "chapter-5",
        "supporting_locations": _supporting_locations,
        "semantic_disposition": _disposition,
        "status": "integrated",
    }

# The third independent residual scan named these branch-bearing public
# routines.  Their complete defaults and exact branch semantics are frozen
# below; none may inherit a broad module claim.
EXPLICIT_OVERRIDES.update(
    {
        "payne_zero_synthesis.continuum.compute_sampled_continuum": override(
            "chapter-5",
            "chapter-10",
            disposition="taught",
            responsibility=(
                "sampled diagnostic when frequency_invariants=None versus "
                "grid-validated materialized sampled extension when supplied"
            ),
            gate=(
                "accepted Chapter 5 calls without and with FrequencyInvariants, "
                "including orientation/grid rejection"
            ),
            status="integrated",
        ),
        (
            "payne_zero_atmosphere.continuum_opacity.compute_hminus_opacity_columns"
        ): override(
            "chapter-5",
            "chapter-11",
            disposition="taught",
            responsibility=(
                "H-minus opacity with unity LTE departure when "
                "hminus_departure_coefficient=None and an explicit depth "
                "multiplier with exact shape validation when supplied"
            ),
            gate=(
                "accepted Chapter 5 unity/explicit H-minus departure and "
                "shape-rejection parity"
            ),
            status="integrated",
        ),
        "payne_zero_atmosphere.convection.compute_convection": override(
            "chapter-12",
            "chapter-13",
            disposition="taught",
            responsibility=(
                "all-eight-arrays finite-difference thermodynamics versus "
                "ideal-gas fallback; convection iteration count, zero mixing "
                "length, overshoot, and top-layer zeroing remain distinct"
            ),
            gate=(
                "Chapter 12 complete/partial/absent derivative bundle, enabled/"
                "disabled, mixing-length, overshoot, and surface-zero matrix"
            ),
            status="planned",
        ),
        (
            "payne_zero_atmosphere.runner.compute_convection_finite_difference_samples"
        ): override(
            "chapter-12",
            "chapter-13",
            disposition="composed",
            responsibility=(
                "atomic-only versus required-state molecular perturbations, "
                "with saved/restored molecular state and optional molecular "
                "thermal-energy tracking across all four EOS samples"
            ),
            gate=(
                "Chapter 12 atomic/molecular, missing-state, tracking on/off, "
                "and exception-safe restoration matrix"
            ),
            status="planned",
        ),
        "payne_zero_atmosphere.runner.finalize_transfer_state": override(
            "chapter-12",
            "chapter-13",
            disposition="composed",
            responsibility=(
                "disabled or caller-supplied convection inputs versus internal "
                "finite-difference/convection construction; seed derivation, "
                "molecular tracking, radiation pressure, and turbulent pressure "
                "are route-specific"
            ),
            gate=(
                "Chapter 12 disabled/supplied/internal convection call trace "
                "and optional-state forwarding matrix"
            ),
            status="planned",
        ),
        "payne_zero_atmosphere.convergence.max_normalized_column_delta": override(
            "chapter-13",
            disposition="composed",
            responsibility=(
                "asymmetric before-column normalization by default versus "
                "symmetric max(before, after, floor) normalization when enabled"
            ),
            gate="Chapter 13 asymmetric/symmetric convergence-norm identity",
            status="planned",
        ),
        "payne_zero_atmosphere.prewarm.prewarm": override(
            "chapter-13",
            "appendix-c",
            disposition="diagnostic-only",
            responsibility=(
                "force=False reuses only a complete signature-, inventory-, "
                "kernel-, specialization-, and fresh-process-valid manifest; "
                "force=True rebuilds and revalidates"
            ),
            gate="Chapter 13 atmosphere prewarm reuse/force/stale matrix",
            status="planned",
        ),
        "payne_zero_synthesis.prewarm.prewarm": override(
            "chapter-10",
            "appendix-c",
            disposition="diagnostic-only",
            responsibility=(
                "force=False reuses only an identity- and artifact-valid window "
                "manifest; force=True bypasses reuse, clears in-process "
                "invariants, and rebuilds persistent products"
            ),
            gate="Chapter 10 synthesis prewarm reuse/force/stale matrix",
            status="planned",
        ),
        "payne_zero_synthesis.atomic_lines.load_catalog": override(
            "chapter-7",
            "chapter-10",
            "appendix-b",
            disposition="composed",
            responsibility=(
                "catalog_path/cache_dir default resolution; sort and isotope "
                "correction enter the cache key and parser; rebuild=False tries "
                "a valid cache then reparses while True always reparses"
            ),
            gate=(
                "Chapter 7 default/explicit paths, catalog/wavelength sort, "
                "isotope on/off, cache hit/corruption, and forced rebuild matrix"
            ),
            status="planned",
        ),
    }
)

# The completed Chapter 6--15 runtime spine now closes a narrower set of
# symbol-specific gates that predate the complete book.  Keep these promotions
# separate from the semantic ownership declarations above: each group names
# the exact public objects supported by one or more current runtime tests.
# High-level publication, full physical convergence, CLIs, prewarm workflows,
# data installation, and unexercised cold/warm loader matrices deliberately do
# not appear here.
VERIFIED_STATUS_PROMOTION_GROUPS: dict[str, dict[str, Any]] = {
    "chapter-7-8-line-and-source-runtime": {
        "evidence": (
            "tests/test_chapter06_runtime.py",
            "tests/test_chapter07_runtime.py",
            "tests/test_chapter08_runtime.py",
            "tests/test_chapter11_runtime.py",
        ),
        "symbols": frozenset(
            {
                "payne_zero_atmosphere.hydrogen_line_profile."
                "molecular_hydrogen_equilibrium_constant",
                "payne_zero_atmosphere.hydrogen_line_profile."
                "compute_hydrogen_molecule_population",
                "payne_zero_atmosphere.line_catalog.decode_selected_line_words",
                "payne_zero_atmosphere.line_opacity."
                "accumulate_selected_line_opacity",
                "payne_zero_atmosphere.line_selection."
                "read_diatomic_line_catalog",
                "payne_zero_atmosphere.line_selection.read_water_line_catalog",
                "payne_zero_atmosphere.line_selection.select_water_line_words",
                "payne_zero_atmosphere.line_selection.generate_selected_lines",
                "payne_zero_atmosphere.source_catalogs.source_line_paths",
                "payne_zero_atmosphere.source_catalogs."
                "atmosphere_source_catalog_paths",
                "payne_zero_synthesis.atomic_lines.Grid.resolution",
                "payne_zero_synthesis.hydrogen_lines.precompute_invariants",
                "payne_zero_synthesis.hydrogen_lines.accumulate_hydrogen",
                "payne_zero_synthesis.line_opacity.accumulate_atomic",
                "payne_zero_synthesis.molecular_lines.load_catalog",
                "payne_zero_synthesis.molecular_lines.accumulate_molecular",
                "payne_zero_synthesis.molecular_lines.molecular_chunk_lines",
                "payne_zero_synthesis.source_catalog_molecular_compiler."
                "compile_molecular_text",
                "payne_zero_synthesis.source_catalog_molecular_compiler."
                "compile_tio_schwenke",
                "payne_zero_synthesis.source_catalog_molecular_compiler."
                "compile_h2o_partridge",
            }
        ),
    },
    "chapter-9-transfer-runtime": {
        "evidence": ("tests/test_chapter09_runtime.py",),
        "symbols": frozenset(
            {
                "payne_zero_synthesis.radiative_transfer.source_and_alpha",
                "payne_zero_synthesis.radiative_transfer.solve_spectrum",
            }
        ),
    },
    "chapter-10-complete-synthesis-runtime": {
        "evidence": ("tests/test_chapter10_runtime.py",),
        "symbols": frozenset(
            {
                "payne_zero_synthesis.pipeline.WINDOW_CONTEXT_SAMPLES",
                "payne_zero_synthesis.pipeline.window_invariant_cache_enabled",
                "payne_zero_synthesis.pipeline.clear_window_invariant_cache",
                "payne_zero_synthesis.pipeline.window_invariants_for",
                "payne_zero_synthesis.pipeline.SpectrumResult",
                "payne_zero_synthesis.pipeline.SpectrumResult.wavelength_nm",
                "payne_zero_synthesis.pipeline.SpectrumResult."
                "eddington_flux_total_per_frequency",
                "payne_zero_synthesis.pipeline.SpectrumResult."
                "eddington_flux_continuum_per_frequency",
                "payne_zero_synthesis.pipeline.SpectrumResult.normalized_flux",
                "payne_zero_synthesis.pipeline.WindowInvariants",
                "payne_zero_synthesis.pipeline.WindowInvariants.key",
                "payne_zero_synthesis.pipeline.WindowInvariants.device",
                "payne_zero_synthesis.pipeline.WindowInvariants.dtype",
                "payne_zero_synthesis.pipeline.WindowInvariants.molecular_lines",
                "payne_zero_synthesis.pipeline.WindowInvariants.metal_chunk",
                "payne_zero_synthesis.pipeline.WindowInvariants.grid_obj",
                "payne_zero_synthesis.pipeline.WindowInvariants."
                "synthesis_wavelength_nm",
                "payne_zero_synthesis.pipeline.WindowInvariants.wavelength_nm",
                "payne_zero_synthesis.pipeline.WindowInvariants.output_slice",
                "payne_zero_synthesis.pipeline.WindowInvariants.n_synthesis_wl",
                "payne_zero_synthesis.pipeline.WindowInvariants.n_wl",
                "payne_zero_synthesis.pipeline.WindowInvariants.n_atomic",
                "payne_zero_synthesis.pipeline.WindowInvariants."
                "atomic_kernel_catalog",
                "payne_zero_synthesis.pipeline.WindowInvariants.has_metal",
                "payne_zero_synthesis.pipeline.WindowInvariants.has_helium",
                "payne_zero_synthesis.pipeline.WindowInvariants.has_hydrogen",
                "payne_zero_synthesis.pipeline.WindowInvariants.continuum_tables",
                "payne_zero_synthesis.pipeline.WindowInvariants.transfer_tables",
                "payne_zero_synthesis.pipeline.WindowInvariants."
                "metal_invariant_chunks",
                "payne_zero_synthesis.pipeline.WindowInvariants."
                "helium_invariants",
                "payne_zero_synthesis.pipeline.WindowInvariants."
                "hydrogen_invariants_template",
                "payne_zero_synthesis.pipeline.WindowInvariants."
                "molecular_invariants",
                "payne_zero_synthesis.pipeline.WindowInvariants.n_molecular",
                "payne_zero_synthesis.pipeline.SynthesisPipeline",
                "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__",
                "payne_zero_synthesis.pipeline.SynthesisPipeline.run",
                "payne_zero_synthesis.synthesis."
                "synthesize_structured_atmosphere",
            }
        ),
    },
    "chapter-11-atmosphere-input-and-opacity-runtime": {
        "evidence": ("tests/test_chapter11_runtime.py",),
        "symbols": frozenset(
            {
                "payne_zero_atmosphere.config.DEFAULT_OPACITY_FLAGS",
                "payne_zero_atmosphere.config.AtmosphereInput."
                "selected_line_catalog_path",
                "payne_zero_atmosphere.config.AtmosphereInput."
                "detailed_line_catalog_path",
                "payne_zero_atmosphere.config.AtmosphereInput."
                "predicted_atomic_lines_path",
                "payne_zero_atmosphere.config.AtmosphereInput."
                "observed_atomic_lines_path",
                "payne_zero_atmosphere.config.AtmosphereInput."
                "high_excitation_lines_path",
                "payne_zero_atmosphere.config.AtmosphereInput."
                "diatomic_lines_path",
                "payne_zero_atmosphere.config.AtmosphereInput."
                "titanium_oxide_lines_path",
                "payne_zero_atmosphere.config.AtmosphereInput.water_lines_path",
                "payne_zero_atmosphere.config.AtmosphereInput.h3plus_lines_path",
                "payne_zero_atmosphere.config.AtmosphereConfig.iterations",
                "payne_zero_atmosphere.microturbulence."
                "standard_microturbulence",
                "payne_zero_atmosphere.run_setup."
                "opacity_flags_from_atmosphere",
                "payne_zero_atmosphere.run_setup.resolve_run_setup",
                "payne_zero_atmosphere.run_setup.RunSetup.iterations",
                "payne_zero_atmosphere.runner.prepare_opacity_state",
                "payne_zero_atmosphere.runtime_state."
                "load_major_isotope_masses_amu",
            }
        ),
    },
    "chapter-12-13-iteration-component-runtime": {
        "evidence": (
            "tests/test_chapter12_runtime.py",
            "tests/test_chapter13_runtime.py",
        ),
        "symbols": frozenset(
            {
                "payne_zero_atmosphere.config.AtmosphereConfig."
                "molecular_convection_thermal_tracks_perturbation",
                "payne_zero_atmosphere.convection.compute_convection",
                "payne_zero_atmosphere.convergence."
                "max_normalized_column_delta",
                "payne_zero_atmosphere.rosseland_mean.rosseland_mean_step",
                "payne_zero_atmosphere.runner."
                "compute_convection_finite_difference_samples",
                "payne_zero_atmosphere.runner.finalize_transfer_state",
                "payne_zero_atmosphere.temperature_correction."
                "initialize_temperature_correction_state",
                "payne_zero_atmosphere.temperature_correction."
                "ingest_temperature_correction_rosseland_table",
                "payne_zero_atmosphere.temperature_correction."
                "apply_temperature_correction",
            }
        ),
    },
    "chapter-14-initializer-safety-runtime": {
        "evidence": ("tests/test_chapter14_runtime.py",),
        "symbols": frozenset(
            {
                "payne_zero_atmosphere.direct_abundance."
                "build_direct_abundance_optimizer_surrogate",
                "payne_zero_atmosphere.direct_abundance."
                "direct_abundance_warm_start_deck",
                "payne_zero_atmosphere.direct_abundance."
                "load_direct_abundance_initializer",
                "payne_zero_atmosphere.direct_abundance."
                "run_direct_abundance_atmosphere",
                "payne_zero_atmosphere.warm_start.warm_start_supported",
                "payne_zero_atmosphere.warm_start."
                "load_atmosphere_initializer",
            }
        ),
    },
    "chapter-10-15-public-api-runtime": {
        "evidence": (
            "tests/test_chapter10_runtime.py",
            "tests/test_chapter14_runtime.py",
            "tests/test_chapter15_runtime.py",
        ),
        "symbols": frozenset(
            {
                "payne_zero_atmosphere.synthesis_bridge."
                "save_structured_atmosphere",
                "payne_zero_synthesis.api.save_structured_atmosphere",
                "payne_zero_synthesis.api.synthesize",
                "payne_zero_synthesis.api.initialize_atmosphere_from_labels",
                "payne_zero_synthesis.api.synthesize_from_labels",
                "payne_zero_synthesis.api.InitializedAtmosphere",
                "payne_zero_synthesis.api.InitializedAtmosphere."
                "atmosphere_converged",
                "payne_zero_synthesis.api.InitializedAtmosphere."
                "atmosphere_closure_required",
                "payne_zero_synthesis.api.InitializedAtmosphere.save_npz",
                "payne_zero_synthesis.api.LabelSpectrum.atmosphere_converged",
                "payne_zero_synthesis.api.LabelSpectrum."
                "atmosphere_closure_required",
                "payne_zero_synthesis.atmosphere."
                "ATMOSPHERE_PRODUCT_METADATA_SCHEMA_VERSION",
                "payne_zero_synthesis.atmosphere."
                "ATMOSPHERE_PRODUCT_METADATA_FIELDS",
                "payne_zero_synthesis.atmosphere."
                "load_atmosphere_product_metadata",
            }
        ),
    },
}

_promoted_symbols: set[str] = set()
for _promotion_group, _promotion in VERIFIED_STATUS_PROMOTION_GROUPS.items():
    _symbols = set(_promotion["symbols"])
    _overlap = _promoted_symbols & _symbols
    if _overlap:
        raise RuntimeError(
            f"verified status promotion groups overlap at {sorted(_overlap)}"
        )
    for _qualified_name in sorted(_symbols):
        if _qualified_name not in EXPLICIT_OVERRIDES:
            raise RuntimeError(
                f"verified status promotion is not an explicit symbol: "
                f"{_qualified_name}"
            )
        if EXPLICIT_OVERRIDES[_qualified_name]["status"] != "planned":
            raise RuntimeError(
                f"verified status promotion no longer starts from planned: "
                f"{_promotion_group}: {_qualified_name}"
            )
        EXPLICIT_OVERRIDES[_qualified_name] = {
            **EXPLICIT_OVERRIDES[_qualified_name],
            "status": "verified",
        }
    _promoted_symbols.update(_symbols)

# Raw AST inventory descriptors freeze parameter order and keyword-only
# placement but do not serialize default expressions.  These source-spelled
# contracts make the branch defaults pedagogically and mechanically explicit;
# the exact module source SHA above binds the expressions back to source bytes.
REVIEWED_DEFAULT_CONTRACTS: dict[str, dict[str, Any]] = {
    "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__": {
        "parameters": [
            "self",
            "atmosphere",
            "source_path",
            "wl_start_nm",
            "wl_end_nm",
            "resolution",
            "tables_path",
            "transfer_tables_path",
            "continuum_tables_path",
            "molecular_lines",
            "device",
            "dtype",
            "metal_chunk",
            "window_invariants",
        ],
        "keyword_only_parameters": [],
        "defaults": {
            "source_path": "None",
            "wl_start_nm": "400.0",
            "wl_end_nm": "900.0",
            "resolution": "20000.0",
            "tables_path": '_SYNTHESIS_TABLE_DIR / "line_profile_tables.npz"',
            "transfer_tables_path": ('_SYNTHESIS_TABLE_DIR / "transfer_tables.npz"'),
            "continuum_tables_path": ('_SYNTHESIS_TABLE_DIR / "continuum_tables.npz"'),
            "molecular_lines": "True",
            "device": "None",
            "dtype": "None",
            "metal_chunk": "None",
            "window_invariants": "None",
        },
    },
    "payne_zero_synthesis.pipeline.SynthesisPipeline.run": {
        "parameters": ["self", "keep_slabs", "spectral_operator"],
        "keyword_only_parameters": [],
        "defaults": {"keep_slabs": "False", "spectral_operator": "None"},
    },
    "payne_zero_synthesis.line_opacity.accumulate_atomic": {
        "parameters": [
            "invariants",
            "state",
            "do_metal",
            "do_helium",
            "apply_stim",
            "wing_mode",
            "output_line_mass_absorption_coefficient",
            "host_accumulator",
        ],
        "keyword_only_parameters": [],
        "defaults": {
            "do_metal": "True",
            "do_helium": "True",
            "apply_stim": "True",
            "wing_mode": '"batched"',
            "output_line_mass_absorption_coefficient": "None",
            "host_accumulator": "None",
        },
    },
    "payne_zero_synthesis.continuum.build_frequency_invariants": {
        "parameters": [
            "continuum_tables",
            "frequencies_hz",
            "coulomb_table_energy_first",
        ],
        "keyword_only_parameters": [],
        "defaults": {"coulomb_table_energy_first": "False"},
    },
    "payne_zero_synthesis.molecular_equilibrium.solve_molecular_equilibrium": {
        "parameters": [
            "temperature",
            "gas_pressure",
            "electron_density",
            "elemental_abundances",
            "ion_formation_constants",
        ],
        "keyword_only_parameters": [
            "molecules_path",
            "device",
            "dtype",
            "max_iter",
            "tol",
            "chain_length",
            "return_diagnostics",
        ],
        "defaults": {
            "molecules_path": "None",
            "device": "None",
            "dtype": "DEFAULT_DTYPE",
            "max_iter": "200",
            "tol": "1e-3",
            "chain_length": "None",
            "return_diagnostics": "False",
        },
    },
    "payne_zero_atmosphere.line_catalog.decode_selected_line_words": {
        "parameters": ["words"],
        "keyword_only_parameters": ["detect_swapped_layout"],
        "defaults": {"detect_swapped_layout": "True"},
    },
    "payne_zero_synthesis.hydrogen_lines.precompute_invariants": {
        "parameters": [
            "catalog",
            "wavelength_grid_nm",
            "electron_density",
            "compute_device",
        ],
        "keyword_only_parameters": [],
        "defaults": {"compute_device": "None"},
    },
    "payne_zero_synthesis.hydrogen_lines.accumulate_hydrogen": {
        "parameters": ["invariants", "state", "apply_stim"],
        "keyword_only_parameters": [],
        "defaults": {"apply_stim": "True"},
    },
    "payne_zero_atmosphere.equation_of_state.saha_partition_depth": {
        "parameters": [],
        "keyword_only_parameters": [
            "temperature_k",
            "electron_density_cm3",
            "total_nuclei_number_density_cm3",
            "elemental_abundance",
            "atomic_number",
            "ion_stage_count",
            "population_mode",
            "charge_square_density_cm3",
        ],
        "defaults": {"charge_square_density_cm3": "None"},
    },
    "payne_zero_atmosphere.equation_of_state.saha_partition_depth_batch": {
        "parameters": [
            "temperature_k",
            "electron_density_cm3",
            "atomic_number",
            "ion_stage_count",
            "population_mode",
            "charge_square_density_cm3",
        ],
        "keyword_only_parameters": [],
        "defaults": {"charge_square_density_cm3": "None"},
    },
    "payne_zero_atmosphere.line_profile_math.load_line_opacity_tables": {
        "parameters": ["path"],
        "keyword_only_parameters": ["force_reload"],
        "defaults": {"path": "None", "force_reload": "False"},
    },
    "payne_zero_atmosphere.temperature_correction.apply_temperature_correction": {
        "parameters": ["state"],
        "keyword_only_parameters": [
            "mode",
            "frequency_weight",
            "column_mass",
            "total_opacity",
            "monochromatic_eddington_flux",
            "mean_intensity_minus_source",
            "monochromatic_optical_depth",
            "planck_source",
            "frequency_hz",
            "h_over_kt",
            "temperature_k",
            "stimulated_emission",
            "scattering_fraction",
            "target_integrated_eddington_flux",
            "effective_temperature",
            "frequency_count",
            "rosseland_optical_depth",
            "rosseland_opacity",
            "iteration_index",
            "convection_enabled",
            "convective_flux",
            "previous_convective_flux",
            "logarithmic_temperature_pressure_gradient",
            "adiabatic_gradient",
            "pressure_scale_height",
            "total_pressure",
            "mass_density",
            "log_density_temperature_derivative_at_constant_total_pressure",
            "heat_capacity",
            "mixing_length",
            "smooth_start_layer",
            "smooth_stop_layer",
            "smooth_left_weight",
            "smooth_center_weight",
            "smooth_right_weight",
            "integrated_radiation_pressure",
            "turbulent_pressure",
            "surface_gravity_cgs",
            "standard_log_tau_step",
            "standard_log_tau_start",
        ],
        "defaults": {
            "rosseland_optical_depth": "None",
            "rosseland_opacity": "None",
            "iteration_index": "1",
            "convection_enabled": "False",
            "convective_flux": "None",
            "previous_convective_flux": "None",
            "logarithmic_temperature_pressure_gradient": "None",
            "adiabatic_gradient": "None",
            "pressure_scale_height": "None",
            "total_pressure": "None",
            "mass_density": "None",
            "log_density_temperature_derivative_at_constant_total_pressure": "None",
            "heat_capacity": "None",
            "mixing_length": "1.0",
            "smooth_start_layer": "0",
            "smooth_stop_layer": "0",
            "smooth_left_weight": "0.3",
            "smooth_center_weight": "0.4",
            "smooth_right_weight": "0.3",
            "integrated_radiation_pressure": "None",
            "turbulent_pressure": "None",
            "surface_gravity_cgs": "1.0e4",
            "standard_log_tau_step": "0.125",
            "standard_log_tau_start": "-6.875",
        },
    },
    "payne_zero_synthesis.molecular_lines.molecular_chunk_lines": {
        "parameters": ["default"],
        "keyword_only_parameters": [],
        "defaults": {"default": "CHUNK_LINES"},
    },
    "payne_zero_synthesis.molecular_lines.accumulate_molecular": {
        "parameters": ["invariants", "state", "apply_stim", "chunk_lines"],
        "keyword_only_parameters": [],
        "defaults": {"apply_stim": "True", "chunk_lines": "None"},
    },
    "payne_zero_atmosphere.continuum_opacity.compute_continuum_opacity_columns": {
        "parameters": ["atmosphere", "frequency_hz"],
        "keyword_only_parameters": ["opacity_flags", "rosseland_table"],
        "defaults": {"opacity_flags": "None", "rosseland_table": "None"},
    },
    (
        "payne_zero_atmosphere.continuum_opacity."
        "compute_light_element_continuum_columns"
    ): {
        "parameters": ["atmosphere", "frequency_hz"],
        "keyword_only_parameters": ["opacity_flags"],
        "defaults": {"opacity_flags": "None"},
    },
    ("payne_zero_atmosphere.continuum_opacity.compute_continuum_scattering_columns"): {
        "parameters": ["atmosphere", "frequency_hz"],
        "keyword_only_parameters": ["opacity_flags", "molecular_tables"],
        "defaults": {"opacity_flags": "None", "molecular_tables": "None"},
    },
    ("payne_zero_synthesis.equation_of_state.partition_functions_for_elements"): {
        "parameters": ["temperature", "gas_pressure", "electron_density"],
        "keyword_only_parameters": [
            "tables",
            "elements",
            "nion",
            "apply_ground_partition",
        ],
        "defaults": {"nion": "1", "apply_ground_partition": "True"},
    },
    "payne_zero_synthesis.equation_of_state.solve_electron_density": {
        "parameters": ["temperature", "gas_pressure", "elemental_abundances"],
        "keyword_only_parameters": [
            "tables",
            "mean_nuclear_mass_amu",
            "electron_density_seed",
            "max_iter",
            "tol",
            "molecules",
            "molecules_path",
        ],
        "defaults": {
            "mean_nuclear_mass_amu": "None",
            "electron_density_seed": "None",
            "max_iter": "200",
            "tol": "1e-4",
            "molecules": "False",
            "molecules_path": "None",
        },
    },
    "payne_zero_synthesis.equation_of_state.solve_population_state": {
        "parameters": ["temperature", "gas_pressure", "elemental_abundances"],
        "keyword_only_parameters": [
            "tables",
            "mean_nuclear_mass_amu",
            "electron_density_seed",
            "max_iter",
            "tol",
            "molecules",
            "molecules_path",
        ],
        "defaults": {
            "mean_nuclear_mass_amu": "None",
            "electron_density_seed": "None",
            "max_iter": "200",
            "tol": "1e-4",
            "molecules": "False",
            "molecules_path": "None",
        },
    },
    (
        "payne_zero_synthesis.equation_of_state."
        "solve_population_state_at_electron_density"
    ): {
        "parameters": ["temperature", "gas_pressure", "elemental_abundances"],
        "keyword_only_parameters": [
            "tables",
            "electron_density",
            "mean_nuclear_mass_amu",
            "mass_density",
            "molecules",
            "molecules_path",
        ],
        "defaults": {
            "mean_nuclear_mass_amu": "None",
            "mass_density": "None",
            "molecules": "False",
            "molecules_path": "None",
        },
    },
    "payne_zero_atmosphere.radiative_transfer.load_radiative_transfer_tables": {
        "parameters": ["path"],
        "keyword_only_parameters": ["force_reload"],
        "defaults": {"path": "None", "force_reload": "False"},
    },
    "payne_zero_synthesis.atomic_lines.parse_catalog": {
        "parameters": ["grid", "catalog_path", "sort", "apply_iso_corr"],
        "keyword_only_parameters": [],
        "defaults": {
            "catalog_path": "None",
            "sort": '"catalog"',
            "apply_iso_corr": "True",
        },
    },
    "payne_zero_synthesis.continuum.compute_sampled_continuum": {
        "parameters": [
            "continuum_tables",
            "frequencies_hz",
            "pops",
            "frequency_invariants",
        ],
        "keyword_only_parameters": [],
        "defaults": {"frequency_invariants": "None"},
    },
    ("payne_zero_atmosphere.continuum_opacity.compute_hminus_opacity_columns"): {
        "parameters": ["atmosphere", "frequency_hz"],
        "keyword_only_parameters": [
            "hminus_departure_coefficient",
            "continuum_tables",
        ],
        "defaults": {
            "hminus_departure_coefficient": "None",
            "continuum_tables": "None",
        },
    },
    "payne_zero_atmosphere.convection.compute_convection": {
        "parameters": [],
        "keyword_only_parameters": [
            "rosseland_table",
            "column_mass",
            "rosseland_optical_depth",
            "temperature_k",
            "gas_pressure",
            "mass_density",
            "rosseland_opacity",
            "microturbulence",
            "absolute_radiation_pressure",
            "total_pressure",
            "surface_gravity_cgs",
            "target_integrated_eddington_flux",
            "mixing_length",
            "overshoot_weight",
            "convection_enabled",
            "zero_top_layer_count",
            "specific_internal_energy_plus_temperature",
            "specific_internal_energy_minus_temperature",
            "specific_internal_energy_plus_pressure",
            "specific_internal_energy_minus_pressure",
            "density_plus_temperature",
            "density_minus_temperature",
            "density_plus_pressure",
            "density_minus_pressure",
        ],
        "defaults": {
            "mixing_length": "1.0",
            "overshoot_weight": "1.0",
            "convection_enabled": "True",
            "zero_top_layer_count": "36",
            "specific_internal_energy_plus_temperature": "None",
            "specific_internal_energy_minus_temperature": "None",
            "specific_internal_energy_plus_pressure": "None",
            "specific_internal_energy_minus_pressure": "None",
            "density_plus_temperature": "None",
            "density_minus_temperature": "None",
            "density_plus_pressure": "None",
            "density_minus_pressure": "None",
        },
    },
    "payne_zero_atmosphere.convergence.max_normalized_column_delta": {
        "parameters": ["before", "after"],
        "keyword_only_parameters": ["floor", "symmetric"],
        "defaults": {"floor": "1.0e-300", "symmetric": "False"},
    },
    ("payne_zero_atmosphere.runner.compute_convection_finite_difference_samples"): {
        "parameters": [],
        "keyword_only_parameters": [
            "atmosphere",
            "runtime_state",
            "absolute_radiation_pressure",
            "rosseland_optical_depth",
            "temperature_iteration_seed",
            "temperature_iteration_cache",
            "molecules_enabled",
            "molecular_state",
            "molecular_thermal_energy_tracks_perturbation",
        ],
        "defaults": {
            "molecules_enabled": "False",
            "molecular_state": "None",
            "molecular_thermal_energy_tracks_perturbation": "False",
        },
    },
    "payne_zero_atmosphere.runner.finalize_transfer_state": {
        "parameters": ["transfer_accumulation"],
        "keyword_only_parameters": [
            "iteration_index",
            "temperature_iteration_seed",
            "convection_enabled",
            "convective_flux",
            "previous_convective_flux",
            "logarithmic_temperature_pressure_gradient",
            "adiabatic_gradient",
            "pressure_scale_height",
            "total_pressure",
            "log_density_temperature_derivative_at_constant_total_pressure",
            "heat_capacity",
            "mixing_length",
            "integrated_radiation_pressure",
            "turbulent_pressure",
            "molecular_convection_thermal_tracks_perturbation",
        ],
        "defaults": {
            "iteration_index": "1",
            "temperature_iteration_seed": "None",
            "convection_enabled": "False",
            "convective_flux": "None",
            "previous_convective_flux": "None",
            "logarithmic_temperature_pressure_gradient": "None",
            "adiabatic_gradient": "None",
            "pressure_scale_height": "None",
            "total_pressure": "None",
            "log_density_temperature_derivative_at_constant_total_pressure": "None",
            "heat_capacity": "None",
            "mixing_length": "1.0",
            "integrated_radiation_pressure": "None",
            "turbulent_pressure": "None",
            "molecular_convection_thermal_tracks_perturbation": "False",
        },
    },
    "payne_zero_atmosphere.prewarm.prewarm": {
        "parameters": [],
        "keyword_only_parameters": ["out_dir", "force"],
        "defaults": {"force": "False"},
    },
    "payne_zero_synthesis.prewarm.prewarm": {
        "parameters": [],
        "keyword_only_parameters": [
            "wavelength_start_nm",
            "wavelength_end_nm",
            "resolution",
            "force",
        ],
        "defaults": {"force": "False"},
    },
    "payne_zero_synthesis.atomic_lines.load_catalog": {
        "parameters": [
            "window",
            "grid",
            "catalog_path",
            "cache_dir",
            "sort",
            "apply_iso_corr",
            "rebuild",
        ],
        "keyword_only_parameters": [],
        "defaults": {
            "catalog_path": "None",
            "cache_dir": "None",
            "sort": '"catalog"',
            "apply_iso_corr": "True",
            "rebuild": "False",
        },
    },
}

# This is the deterministic residual of the pinned-source AST scan after the
# full explicit-symbol registry above is applied.  Every public callable here
# has at least one default argument.  It receives a location-preserving,
# callable-specific review source instead of silently inheriting a module
# policy.  Exact source bytes and the test-side AST fixed-point scan bind the
# defaults; these reasons state what the defaults can change.
DEFAULT_CALLABLE_REVIEWS = {
    "payne_zero_atmosphere.cli.solve_structured_atmosphere": (
        "initializer/composition presence selects five-label, CNO8, or direct "
        "abundance routing; alias conflicts, deterministic multistart, solve "
        "effort, seed, and jitter controls are distinct"
    ),
    "payne_zero_atmosphere.data_files.load_table_arrays": (
        "error_type changes only the public missing/malformed-table exception "
        "surface; it cannot change a successfully loaded numerical table"
    ),
    "payne_zero_atmosphere.equation_of_state.iterate_electron_density": (
        "max_iterations and tolerance are numerical termination controls for "
        "the same electron-closure equation"
    ),
    "payne_zero_atmosphere.equation_of_state.populate_species": (
        "molecular_state is a required-state validation conditioned on the "
        "separate molecules_enabled route, not an independent branch switch"
    ),
    "payne_zero_atmosphere.equation_of_state.populate_all_species": (
        "molecular_state is a required-state validation conditioned on the "
        "separate molecules_enabled route, not an independent branch switch"
    ),
    "payne_zero_atmosphere.hydrostatic.integrate_hydrostatic_pressure": (
        "pressure_constant is the continuous hydrostatic boundary term, not a "
        "discrete algorithm or compatibility route"
    ),
    "payne_zero_atmosphere.install_runtime_data.load_runtime_manifest": (
        "the default selects the packaged manifest; an explicit path changes "
        "data identity but not manifest parsing semantics"
    ),
    "payne_zero_atmosphere.install_runtime_data.verify_runtime_data": (
        "manifest_path selects data identity and include_public_metadata "
        "includes or skips the public-metadata verification surface"
    ),
    "payne_zero_atmosphere.install_runtime_data.install_initializer_assets": (
        "include_direct_xh selects the opt-in direct-abundance asset family; "
        "replace selects reject versus atomic replacement"
    ),
    "payne_zero_atmosphere.install_runtime_data.install_runtime_data": (
        "manifest_path selects data identity and replace selects reject versus "
        "atomic replacement of mismatched files"
    ),
    "payne_zero_atmosphere.install_runtime_data.main": (
        "argv=None consumes process arguments; an explicit sequence changes "
        "only the CLI invocation surface"
    ),
    "payne_zero_atmosphere.line_opacity.accumulate_transition_line_opacity": (
        "optional population/mass arrays enable special-transition inputs; a "
        "base slab selects allocate versus compose, and start/stop select a "
        "partial deposition window"
    ),
    "payne_zero_atmosphere.line_profile_math.fast_exponential_lookup": (
        "tables=None loads the canonical lookup; an explicit equivalent table "
        "is dependency injection for the same interpolation algorithm"
    ),
    "payne_zero_atmosphere.line_profile_math.evaluate_voigt_profile": (
        "basis=None builds the canonical basis; an explicit equivalent basis "
        "is dependency injection for the same Voigt algorithm"
    ),
    "payne_zero_atmosphere.line_selection.select_standard_line_words": (
        "frequency derivation, strength/slot/damping overrides, minimum ratio, "
        "and bin floor select exact ordinary versus diatomic/parity keep tests"
    ),
    "payne_zero_atmosphere.prewarm.main": (
        "argv=None consumes process arguments; an explicit sequence changes "
        "only the CLI invocation surface"
    ),
    "payne_zero_atmosphere.runner.prepare_population_state": (
        "setup selects resolve versus reuse, molecular thermal energy selects "
        "the ordinary versus injected parity input, and iteration index enters "
        "the population cache identity"
    ),
    ("payne_zero_atmosphere.runner.prepare_structured_handoff_population_state"): (
        "setup selects resolve versus reuse, molecular thermal energy selects "
        "the ordinary versus injected parity input, and iteration index enters "
        "the population cache identity"
    ),
    "payne_zero_atmosphere.runner.accumulate_transfer_state": (
        "start/stop select a frequency-block pass while a missing versus "
        "supplied correction state initializes versus continues accumulation"
    ),
    "payne_zero_atmosphere.runner.remap_finalized_iteration_state": (
        "missing convective/turbulent columns become zeros, supplied columns "
        "are remapped, completed_iterations changes metadata, and start/step "
        "select the continuous standard output grid"
    ),
    "payne_zero_atmosphere.runner.finalize_remapped_iteration": (
        "converged defaults false and diagnostics defaults empty; supplied "
        "values change the explicit terminal status and diagnostic payload"
    ),
    "payne_zero_atmosphere.source_catalogs.load_source_catalog_checksums": (
        "checksum_path=None selects the committed checksum manifest; an "
        "explicit path changes source identity only"
    ),
    "payne_zero_atmosphere.source_catalogs.verify_source_catalog_checksums": (
        "root/checksum_path defaults select committed runtime identities; "
        "explicit paths verify an alternate declared installation"
    ),
    "payne_zero_atmosphere.warm_start.compute_metal_log_number_abundances": (
        "absent offsets use bulk metallicity/alpha scaling; supplied per-element "
        "absolute abundances override those elements"
    ),
    "payne_zero_atmosphere.warm_start.compute_hydrogen_fraction": (
        "absent offsets use bulk metallicity/alpha scaling; supplied per-element "
        "absolute abundances alter the composition closure"
    ),
    "payne_zero_atmosphere.warm_start.select_warm_start_family": (
        "all CNO coordinates absent selects five-label; any relative or "
        "absolute C/N/O coordinate selects CNO8"
    ),
    "payne_zero_atmosphere.warm_start.resolve_cno8_labels": (
        "relative values override converted absolute values when consistent; "
        "otherwise C/N default to zero and O to alpha, with conflicts rejected"
    ),
    "payne_zero_atmosphere.warm_start.atmosphere_prediction_to_layer_table": (
        "microturbulence_km_s is a continuous predicted-table value, not a "
        "discrete initializer route"
    ),
    "payne_zero_atmosphere.warm_start.format_warm_start_deck": (
        "absolute offsets select per-element composition and title=None selects "
        "generated rather than caller publication text"
    ),
    "payne_zero_atmosphere.warm_start.AtmosphereInitializer.predict": (
        "optional C/N/O values only populate the already selected checkpoint "
        "feature vector; this method does not select the model family"
    ),
    "payne_zero_atmosphere.warm_start.deterministic_initializer_labels": (
        "CNO presence selects family, checkpoint_path selects source identity, "
        "max_trials adds deterministic jitter neighbors, and seed/scale/device "
        "control that ordered candidate computation"
    ),
    "payne_zero_atmosphere.warm_start.emulator_warm_start_model": (
        "CNO values select five-label/CNO8, family paths select checkpoints, "
        "initializer_label selects target versus nearby seed, and device/title "
        "control inference/provenance"
    ),
    "payne_zero_synthesis.atomic_lines.LineCatalog.to_torch": (
        "device and float/int dtype defaults select the standard backend and "
        "precision; explicit values change only the catalog tensor placement"
    ),
    "payne_zero_synthesis.device.to_dev": (
        "device=None selects the module runtime and dtype selects the explicit "
        "tensor cast; both are backend/precision controls"
    ),
    "payne_zero_synthesis.equation_of_state.EOSTables.from_npz": (
        "the default path selects packaged EOS inputs while device/dtype select "
        "the backend and precision of the loaded tables"
    ),
    "payne_zero_synthesis.equation_of_state.EOSTables.from_dict": (
        "device/dtype defaults select standard backend and precision for the "
        "same supplied in-memory EOS mapping"
    ),
    "payne_zero_synthesis.equation_of_state.populations": (
        "n_elements and max_ion are explicit algorithmic domain-truncation "
        "controls, not harmless tolerances"
    ),
    "payne_zero_synthesis.equation_of_state.load_state_from_inputs": (
        "the default path selects packaged EOS inputs while device/dtype select "
        "the backend and precision of the derived state"
    ),
    "payne_zero_synthesis.line_opacity.precompute_invariants": (
        "runtime_device=None resolves the standard backend; an explicit device "
        "fixes invariant placement and precision policy"
    ),
    ("payne_zero_synthesis.molecular_equilibrium.molecular_line_populations"): (
        "molecules_path=None selects the bundled molecular catalog; an explicit "
        "path changes the declared source identity"
    ),
    ("payne_zero_synthesis.molecular_equilibrium.all_molecular_line_populations"): (
        "partition_functions=None uses internal normalization while a supplied "
        "cube injects caller partitions; molecules_path selects catalog identity"
    ),
    (
        "payne_zero_synthesis.molecular_equilibrium."
        "molecular_line_populations_by_species_code"
    ): (
        "molecules_path=None selects the bundled molecular catalog; an explicit "
        "path changes the declared source identity"
    ),
    ("payne_zero_synthesis.molecular_equilibrium.molecular_equilibrium_metadata"): (
        "molecules_path selects bundled versus explicit catalog identity and "
        "the resolved path is part of the metadata cache key"
    ),
    "payne_zero_synthesis.molecular_lines.precompute_invariants": (
        "runtime_device=None resolves the standard backend; an explicit device "
        "fixes invariant placement and precision policy"
    ),
    "payne_zero_synthesis.prewarm.main": (
        "argv=None consumes process arguments; an explicit sequence changes "
        "only the CLI invocation surface"
    ),
    "payne_zero_synthesis.radiative_transfer.TransferTables.from_npz": (
        "device/dtype defaults select standard backend and work precision for "
        "the exact transfer-table archive"
    ),
    "payne_zero_synthesis.radiative_transfer.TransferTables.to": (
        "dtype=None preserves the current dtype while the required device moves "
        "the table; explicit dtype changes transfer precision"
    ),
    "payne_zero_synthesis.radiative_transfer.solve_scattering_source": (
        "sweeps controls the exact backward Gauss-Seidel iteration count and "
        "therefore the numerical source result"
    ),
    "payne_zero_synthesis.synthesis.compute_mean_nuclear_mass_amu": (
        "atomic_masses=None loads canonical masses while supplied masses change "
        "the physical reduction and provide the injection/parity route"
    ),
}

REVIEWED_DEFAULT_CONTRACTS_SHA256 = (
    "454f0fb32d68a4173ee2502bad6b600a2c568dfe97a1adebbe32ab4d6ab01449"
)

# Exact source segments and AST structures for the lexical defaults called out
# by the second independent audit.  The source segment preserves quote style;
# the AST dump independently freezes the expression tree.
EXACT_DEFAULT_SYNTAX = {
    "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__": {
        "tables_path": {
            "source_segment": '_SYNTHESIS_TABLE_DIR / "line_profile_tables.npz"',
            "ast_dump": (
                "BinOp(left=Name(id='_SYNTHESIS_TABLE_DIR', ctx=Load()), "
                "op=Div(), right=Constant(value='line_profile_tables.npz'))"
            ),
        },
        "transfer_tables_path": {
            "source_segment": '_SYNTHESIS_TABLE_DIR / "transfer_tables.npz"',
            "ast_dump": (
                "BinOp(left=Name(id='_SYNTHESIS_TABLE_DIR', ctx=Load()), "
                "op=Div(), right=Constant(value='transfer_tables.npz'))"
            ),
        },
        "continuum_tables_path": {
            "source_segment": '_SYNTHESIS_TABLE_DIR / "continuum_tables.npz"',
            "ast_dump": (
                "BinOp(left=Name(id='_SYNTHESIS_TABLE_DIR', ctx=Load()), "
                "op=Div(), right=Constant(value='continuum_tables.npz'))"
            ),
        },
    },
    "payne_zero_synthesis.line_opacity.accumulate_atomic": {
        "wing_mode": {
            "source_segment": '"batched"',
            "ast_dump": "Constant(value='batched')",
        }
    },
}
EXACT_DEFAULT_SYNTAX_SHA256 = (
    "70aa59048d72a23af70170dc87f0fadf2a865344d0831ffd78a23cb265d15433"
)


# These default-bearing callables already have complete symbol overrides.  The
# source hash and complete callable descriptor bind their exact defaults; their
# full override responsibility is the branch review.
EXPLICIT_DEFAULT_CALLABLES = frozenset(
    {
        "payne_zero_atmosphere.atmosphere_io.parse_atmosphere_deck",
        "payne_zero_atmosphere.cli.main",
        (
            "payne_zero_atmosphere.continuum_opacity."
            "compute_carbon_neutral_opacity_columns"
        ),
        (
            "payne_zero_atmosphere.continuum_opacity."
            "compute_helium_ionized_opacity_columns"
        ),
        (
            "payne_zero_atmosphere.continuum_opacity."
            "compute_helium_neutral_opacity_columns"
        ),
        ("payne_zero_atmosphere.continuum_opacity.compute_hot_metal_opacity_columns"),
        ("payne_zero_atmosphere.continuum_opacity.compute_hydrogen_opacity_columns"),
        (
            "payne_zero_atmosphere.continuum_opacity."
            "compute_magnesium_neutral_opacity_columns"
        ),
        (
            "payne_zero_atmosphere.continuum_opacity."
            "compute_molecular_continuum_opacity_columns"
        ),
        (
            "payne_zero_atmosphere.continuum_opacity."
            "compute_molecular_hydrogen_population"
        ),
        (
            "payne_zero_atmosphere.continuum_opacity."
            "compute_silicon_neutral_opacity_columns"
        ),
        ("payne_zero_atmosphere.continuum_opacity.create_rosseland_opacity_table"),
        ("payne_zero_atmosphere.convection.compute_disabled_convection_diagnostics"),
        (
            "payne_zero_atmosphere.direct_abundance."
            "build_direct_abundance_optimizer_surrogate"
        ),
        ("payne_zero_atmosphere.direct_abundance.direct_abundance_warm_start_deck"),
        ("payne_zero_atmosphere.direct_abundance.load_direct_abundance_initializer"),
        ("payne_zero_atmosphere.direct_abundance.run_direct_abundance_atmosphere"),
        (
            "payne_zero_atmosphere.hydrogen_line_profile."
            "compute_hydrogen_molecule_population"
        ),
        (
            "payne_zero_atmosphere.hydrogen_line_profile."
            "load_hydrogen_line_profile_tables"
        ),
        (
            "payne_zero_atmosphere.hydrogen_line_profile."
            "molecular_hydrogen_equilibrium_constant"
        ),
        ("payne_zero_atmosphere.line_opacity.accumulate_selected_line_opacity"),
        "payne_zero_atmosphere.line_selection.generate_selected_lines",
        "payne_zero_atmosphere.line_selection.select_water_line_words",
        "payne_zero_atmosphere.microturbulence.standard_microturbulence",
        ("payne_zero_atmosphere.molecular_equilibrium.solve_molecular_equilibrium"),
        "payne_zero_atmosphere.runner.prepare_opacity_state",
        "payne_zero_atmosphere.runtime_state.load_major_isotope_masses_amu",
        ("payne_zero_atmosphere.source_catalogs.atmosphere_source_catalog_paths"),
        ("payne_zero_atmosphere.source_catalogs.molecular_equilibrium_catalog_path"),
        "payne_zero_atmosphere.source_catalogs.source_line_paths",
        ("payne_zero_atmosphere.synthesis_bridge.save_product_structured_atmosphere"),
        (
            "payne_zero_atmosphere.synthesis_bridge."
            "save_structured_atmosphere_from_debug_npz"
        ),
        (
            "payne_zero_atmosphere.synthesis_bridge."
            "save_structured_atmosphere_from_runtime_state"
        ),
        ("payne_zero_atmosphere.synthesis_bridge.structured_atmosphere_from_debug_npz"),
        (
            "payne_zero_atmosphere.synthesis_bridge."
            "structured_atmosphere_from_packed_state"
        ),
        (
            "payne_zero_atmosphere.synthesis_bridge."
            "structured_atmosphere_from_runtime_state"
        ),
        "payne_zero_atmosphere.warm_start.load_atmosphere_initializer",
        "payne_zero_atmosphere.warm_start.warm_start_supported",
        "payne_zero_synthesis.api.build_structured_atmosphere",
        "payne_zero_synthesis.api.initialize_atmosphere_from_labels",
        "payne_zero_synthesis.api.synthesize",
        "payne_zero_synthesis.api.synthesize_from_labels",
        "payne_zero_synthesis.cli.main",
        "payne_zero_synthesis.continuum.ContinuumTables.from_dict",
        "payne_zero_synthesis.continuum.ContinuumTables.from_npz",
        "payne_zero_synthesis.continuum.build_pops",
        "payne_zero_synthesis.continuum.continuum",
        "payne_zero_synthesis.continuum.pops_from_population_state",
        "payne_zero_synthesis.device.resolve_runtime",
        "payne_zero_synthesis.molecular_lines.load_catalog",
        ("payne_zero_synthesis.pipeline.build_structured_atmosphere_from_columns"),
        "payne_zero_synthesis.pipeline.load_atomic_masses",
        "payne_zero_synthesis.pipeline.window_invariants_for",
        "payne_zero_synthesis.radiative_transfer.solve_spectrum",
        (
            "payne_zero_synthesis.source_catalog_molecular_compiler."
            "compile_h2o_partridge"
        ),
        (
            "payne_zero_synthesis.source_catalog_molecular_compiler."
            "compile_molecular_text"
        ),
        ("payne_zero_synthesis.source_catalog_molecular_compiler.compile_tio_schwenke"),
        ("payne_zero_synthesis.synthesis.build_structured_atmosphere_from_columns"),
        ("payne_zero_synthesis.synthesis.synthesize_structured_atmosphere"),
    }
)

# The union is the fixed point of a source-AST scan over every public callable
# in the 55 reviewed modules that has at least one positional or keyword-only
# default.  Tests reconstruct the set from the pinned source instead of
# trusting this registry to declare itself complete.
DEFAULT_BEARING_PUBLIC_CALLABLES = (
    frozenset(REVIEWED_DEFAULT_CONTRACTS)
    | frozenset(DEFAULT_CALLABLE_REVIEWS)
    | EXPLICIT_DEFAULT_CALLABLES
)

# A callable-level sentence cannot prove that every source default was reviewed.
# Build the lexical manifest directly from the pinned checkout, then join each
# exact ``(qualified_name, parameter)`` key to a manually classified effect.
# The resulting 456 records are embedded in the ledger and sealed below.
PINNED_PAYNE_ZERO_SOURCE_ROOT = REPOSITORY_ROOT.parent / "payne-zero"
DEFAULT_EFFECT_CATEGORIES = frozenset(
    {
        "physical",
        "data/source identity",
        "cache",
        "environment",
        "diagnostic",
        "compatibility",
        "parity/injection",
        "unsupported",
        "algorithmic",
        "continuous-value",
        "dependency-injection-with-no-branch",
    }
)


def _source_default_parameter_manifest() -> dict[str, dict[str, dict[str, str]]]:
    """Recover exact default source segments for the pinned 140-callable set."""

    manifest: dict[str, dict[str, dict[str, str]]] = {}
    expected = set(DEFAULT_BEARING_PUBLIC_CALLABLES)
    for qualified_module in sorted(REVIEWED_MODULE_SNAPSHOTS):
        package_name, module_name = qualified_module.split(".", maxsplit=1)
        source_path = (
            PINNED_PAYNE_ZERO_SOURCE_ROOT
            / package_name
            / f"{module_name.replace('.', '/')}.py"
        )
        if not source_path.is_file():
            raise RuntimeError(
                f"pinned source module is unavailable for default review: {source_path}"
            )
        source_text = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source_text)
        callables: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                callables.append((f"{qualified_module}.{node.name}", node))
                continue
            if not isinstance(node, ast.ClassDef):
                continue
            callables.extend(
                (f"{qualified_module}.{node.name}.{child.name}", child)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
        for qualified_name, node in callables:
            if qualified_name not in expected:
                continue
            defaults: dict[str, dict[str, str]] = {}
            positional = node.args.posonlyargs + node.args.args
            if node.args.defaults:
                for argument, default_node in zip(
                    positional[-len(node.args.defaults) :],
                    node.args.defaults,
                    strict=True,
                ):
                    source_default = ast.get_source_segment(source_text, default_node)
                    if source_default is None:
                        raise RuntimeError(
                            "cannot recover source default for "
                            f"{qualified_name}.{argument.arg}"
                        )
                    defaults[argument.arg] = {
                        "source_default": source_default,
                        "default_ast": ast.dump(
                            default_node,
                            include_attributes=False,
                        ),
                    }
            for argument, default_node in zip(
                node.args.kwonlyargs,
                node.args.kw_defaults,
                strict=True,
            ):
                if default_node is None:
                    continue
                source_default = ast.get_source_segment(source_text, default_node)
                if source_default is None:
                    raise RuntimeError(
                        "cannot recover source default for "
                        f"{qualified_name}.{argument.arg}"
                    )
                defaults[argument.arg] = {
                    "source_default": source_default,
                    "default_ast": ast.dump(
                        default_node,
                        include_attributes=False,
                    ),
                }
            manifest[qualified_name] = {
                parameter_name: defaults[parameter_name]
                for parameter_name in sorted(defaults)
            }
    if set(manifest) != expected:
        raise RuntimeError(
            "source-default callable fixed point changed: "
            f"missing={sorted(expected - set(manifest))}, "
            f"extra={sorted(set(manifest) - expected)}"
        )
    if sum(len(defaults) for defaults in manifest.values()) != 456:
        raise RuntimeError("source-default parameter fixed point changed")
    return {name: manifest[name] for name in sorted(manifest)}


SOURCE_DEFAULT_PARAMETER_MANIFEST = _source_default_parameter_manifest()


def _source_default_parameter_use_evidence() -> dict[str, dict[str, dict[str, Any]]]:
    """Recover exact load, predicate, and call-forward evidence for all defaults."""

    expected = set(DEFAULT_BEARING_PUBLIC_CALLABLES)
    evidence: dict[str, dict[str, dict[str, Any]]] = {}

    def contains_parameter(node: ast.AST, parameter_name: str) -> bool:
        return any(
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
            and child.id == parameter_name
            for child in ast.walk(node)
        )

    for qualified_module in sorted(REVIEWED_MODULE_SNAPSHOTS):
        package_name, module_name = qualified_module.split(".", maxsplit=1)
        relative_path = (
            Path(package_name) / f"{module_name.replace('.', '/')}.py"
        ).as_posix()
        source_path = PINNED_PAYNE_ZERO_SOURCE_ROOT / relative_path
        source_bytes = source_path.read_bytes()
        source_text = source_bytes.decode("utf-8")
        source_lines = source_text.splitlines()
        tree = ast.parse(source_text)
        callables: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                callables.append((f"{qualified_module}.{node.name}", node))
                continue
            if not isinstance(node, ast.ClassDef):
                continue
            callables.extend(
                (f"{qualified_module}.{node.name}.{child.name}", child)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )

        for qualified_name, callable_node in callables:
            if qualified_name not in expected:
                continue
            parameter_names = set(SOURCE_DEFAULT_PARAMETER_MANIFEST[qualified_name])
            uses_by_parameter: dict[str, list[dict[str, Any]]] = {
                parameter_name: [] for parameter_name in parameter_names
            }
            predicates_by_parameter: dict[str, list[dict[str, Any]]] = {
                parameter_name: [] for parameter_name in parameter_names
            }
            calls_by_parameter: dict[str, list[dict[str, Any]]] = {
                parameter_name: [] for parameter_name in parameter_names
            }
            attribute_predicates_by_parameter: dict[str, list[dict[str, Any]]] = {
                parameter_name: [] for parameter_name in parameter_names
            }
            flow_bindings_by_parameter: dict[str, list[dict[str, Any]]] = {
                parameter_name: [] for parameter_name in parameter_names
            }

            body_nodes: list[ast.AST] = []
            for statement in callable_node.body:
                body_nodes.extend(ast.walk(statement))
            parent_by_node: dict[ast.AST, ast.AST] = {
                child: parent
                for parent in ast.walk(callable_node)
                for child in ast.iter_child_nodes(parent)
            }

            # Follow simple local assignments so a predicate on a derived name
            # (for example ``chain_len`` derived from ``chain_length``) remains
            # source-verifiable evidence for the exact default parameter.
            flow_names_by_parameter = {
                parameter_name: {parameter_name} for parameter_name in parameter_names
            }

            def dotted_symbol(node: ast.AST) -> str | None:
                if isinstance(node, ast.Name):
                    return node.id
                if isinstance(node, ast.Attribute):
                    prefix = dotted_symbol(node.value)
                    return f"{prefix}.{node.attr}" if prefix is not None else None
                return None

            def loaded_symbols(node: ast.AST) -> set[str]:
                symbols = {
                    symbol
                    for child in ast.walk(node)
                    if isinstance(child, (ast.Name, ast.Attribute))
                    and isinstance(child.ctx, ast.Load)
                    if (symbol := dotted_symbol(child)) is not None
                }
                return symbols

            def stored_symbols(node: ast.AST) -> set[str]:
                return {
                    symbol
                    for child in ast.walk(node)
                    if isinstance(child, (ast.Name, ast.Attribute))
                    and isinstance(child.ctx, ast.Store)
                    if (symbol := dotted_symbol(child)) is not None
                }

            changed = True
            while changed:
                changed = False
                for node in body_nodes:
                    value: ast.AST | None = None
                    targets: list[ast.AST] = []
                    if isinstance(node, ast.Assign):
                        value = node.value
                        targets = list(node.targets)
                    elif isinstance(node, ast.AnnAssign) and node.value is not None:
                        value = node.value
                        targets = [node.target]
                    elif isinstance(node, ast.NamedExpr):
                        value = node.value
                        targets = [node.target]
                    if value is None:
                        continue
                    # A general call result is not treated as a transparent
                    # alias: its output may summarize many inputs and requires
                    # an explicit forwarded-target contract instead.  Local
                    # arithmetic, comparisons, and conditional selections do
                    # preserve a traceable parameter flow.
                    if isinstance(
                        value,
                        (
                            ast.Call,
                            ast.Dict,
                            ast.List,
                            ast.Set,
                            ast.Tuple,
                            ast.Lambda,
                            ast.ListComp,
                            ast.SetComp,
                            ast.DictComp,
                            ast.GeneratorExp,
                        ),
                    ):
                        continue
                    value_names = loaded_symbols(value)
                    target_names = set().union(
                        *(stored_symbols(target) for target in targets)
                    )
                    for parameter_name, flow_names in flow_names_by_parameter.items():
                        if not value_names & flow_names:
                            continue
                        value_snippet = ast.get_source_segment(source_text, value)
                        if value_snippet is None:
                            raise RuntimeError(
                                f"cannot recover flow source in {qualified_name}"
                            )
                        for target_name in sorted(target_names):
                            binding = {
                                "line": node.lineno,
                                "target": target_name,
                                "value": value_snippet,
                                "value_ast": ast.dump(
                                    value,
                                    include_attributes=False,
                                ),
                                "relation": (
                                    "direct"
                                    if parameter_name in value_names
                                    else "coupled"
                                ),
                                "snippet_sha256": hashlib.sha256(
                                    value_snippet.encode("utf-8")
                                ).hexdigest(),
                            }
                            if (
                                binding
                                not in flow_bindings_by_parameter[parameter_name]
                            ):
                                flow_bindings_by_parameter[parameter_name].append(
                                    binding
                                )
                        new_names = target_names - flow_names
                        if new_names:
                            flow_names.update(new_names)
                            changed = True

            def contains_parameter_flow(
                node: ast.AST,
                parameter_name: str,
            ) -> bool:
                return bool(
                    loaded_symbols(node) & flow_names_by_parameter[parameter_name]
                )

            def control_guards(
                node: ast.AST,
                parameter_name: str,
            ) -> list[dict[str, Any]]:
                guards: list[dict[str, Any]] = []
                child = node
                parent = parent_by_node.get(child)
                while parent is not None and parent is not callable_node:
                    predicate: ast.AST | None = None
                    polarity: str | None = None
                    kind: str | None = None
                    if isinstance(parent, ast.If):
                        predicate = parent.test
                        kind = "if"
                        if child in parent.body:
                            polarity = "true"
                        elif child in parent.orelse:
                            polarity = "false"
                    elif isinstance(parent, ast.IfExp):
                        predicate = parent.test
                        kind = "if_expression"
                        if child is parent.body:
                            polarity = "true"
                        elif child is parent.orelse:
                            polarity = "false"
                    elif isinstance(parent, ast.While):
                        predicate = parent.test
                        kind = "while"
                        if child in parent.body:
                            polarity = "true"
                        elif child in parent.orelse:
                            polarity = "false"
                    if (
                        predicate is not None
                        and polarity is not None
                        and contains_parameter_flow(predicate, parameter_name)
                    ):
                        predicate_snippet = ast.get_source_segment(
                            source_text,
                            predicate,
                        )
                        if predicate_snippet is None:
                            raise RuntimeError(
                                f"cannot recover guard source in {qualified_name}"
                            )
                        guards.append(
                            {
                                "line": predicate.lineno,
                                "kind": kind,
                                "polarity": polarity,
                                "relation": (
                                    "direct"
                                    if contains_parameter(
                                        predicate,
                                        parameter_name,
                                    )
                                    else "coupled"
                                ),
                                "snippet": predicate_snippet,
                                "snippet_sha256": hashlib.sha256(
                                    predicate_snippet.encode("utf-8")
                                ).hexdigest(),
                            }
                        )
                    child = parent
                    parent = parent_by_node.get(child)
                return sorted(
                    guards,
                    key=lambda item: (
                        item["line"],
                        item["kind"],
                        item["polarity"],
                    ),
                )

            def source_head(node: ast.AST | None) -> str | None:
                if node is None:
                    return None
                return ast.get_source_segment(source_text, node) or ast.unparse(node)

            for node in body_nodes:
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in parameter_names:
                        continue
                    snippet = source_lines[node.lineno - 1].strip()
                    use_record = {
                        "line": node.lineno,
                        "column": node.col_offset,
                        "snippet": snippet,
                        "snippet_sha256": hashlib.sha256(
                            snippet.encode("utf-8")
                        ).hexdigest(),
                        "control_guards": control_guards(node, node.id),
                    }
                    if use_record not in uses_by_parameter[node.id]:
                        uses_by_parameter[node.id].append(use_record)

                predicate: ast.AST | None = None
                predicate_kind: str | None = None
                body_head: ast.AST | None = None
                else_head: ast.AST | None = None
                if isinstance(node, ast.If):
                    predicate = node.test
                    predicate_kind = "if"
                    body_head = node.body[0] if node.body else None
                    else_head = node.orelse[0] if node.orelse else None
                elif isinstance(node, ast.IfExp):
                    predicate = node.test
                    predicate_kind = "if_expression"
                    body_head = node.body
                    else_head = node.orelse
                elif isinstance(node, ast.While):
                    predicate = node.test
                    predicate_kind = "while"
                    body_head = node.body[0] if node.body else None
                    else_head = node.orelse[0] if node.orelse else None
                elif isinstance(node, ast.Assert):
                    predicate = node.test
                    predicate_kind = "assert"
                elif isinstance(node, ast.BoolOp):
                    predicate = node
                    predicate_kind = "selection_expression"
                    selector = node.values[0]
                    remainder: ast.AST
                    if len(node.values) == 2:
                        remainder = node.values[1]
                    else:
                        remainder = ast.BoolOp(
                            op=node.op,
                            values=node.values[1:],
                        )
                    if isinstance(node.op, ast.Or):
                        body_head = selector
                        else_head = remainder
                    else:
                        body_head = remainder
                        else_head = selector
                if predicate is not None:
                    predicate_snippet = ast.get_source_segment(
                        source_text,
                        predicate,
                    )
                    if predicate_snippet is None:
                        raise RuntimeError(
                            f"cannot recover predicate source in {qualified_name}"
                        )
                    for parameter_name in parameter_names:
                        if not contains_parameter_flow(predicate, parameter_name):
                            continue
                        predicate_record = {
                            "line": predicate.lineno,
                            "kind": predicate_kind,
                            "relation": (
                                "direct"
                                if contains_parameter(predicate, parameter_name)
                                else "coupled"
                            ),
                            "snippet": predicate_snippet,
                            "snippet_sha256": hashlib.sha256(
                                predicate_snippet.encode("utf-8")
                            ).hexdigest(),
                            "body_head": (source_head(body_head)),
                            "else_head": (source_head(else_head)),
                        }
                        predicates_by_parameter[parameter_name].append(predicate_record)
                        if any(
                            "." in symbol
                            for symbol in (
                                loaded_symbols(predicate)
                                & flow_names_by_parameter[parameter_name]
                            )
                        ):
                            attribute_predicates_by_parameter[parameter_name].append(
                                predicate_record
                            )

                if not isinstance(node, ast.Call):
                    continue
                callee = ast.get_source_segment(source_text, node.func)
                call_snippet = ast.get_source_segment(source_text, node)
                if callee is None or call_snippet is None:
                    raise RuntimeError(
                        f"cannot recover call source in {qualified_name}"
                    )
                for position, argument in enumerate(node.args):
                    for parameter_name in parameter_names:
                        if not contains_parameter_flow(argument, parameter_name):
                            continue
                        calls_by_parameter[parameter_name].append(
                            {
                                "line": node.lineno,
                                "callee": callee,
                                "argument": f"position:{position}",
                                "relation": (
                                    "direct"
                                    if contains_parameter(argument, parameter_name)
                                    else "coupled"
                                ),
                                "snippet": call_snippet,
                                "snippet_sha256": hashlib.sha256(
                                    call_snippet.encode("utf-8")
                                ).hexdigest(),
                                "control_guards": control_guards(
                                    node,
                                    parameter_name,
                                ),
                            }
                        )
                for keyword in node.keywords:
                    for parameter_name in parameter_names:
                        if not contains_parameter_flow(
                            keyword.value,
                            parameter_name,
                        ):
                            continue
                        calls_by_parameter[parameter_name].append(
                            {
                                "line": node.lineno,
                                "callee": callee,
                                "argument": (
                                    keyword.arg
                                    if keyword.arg is not None
                                    else "**mapping"
                                ),
                                "relation": (
                                    "direct"
                                    if contains_parameter(
                                        keyword.value,
                                        parameter_name,
                                    )
                                    else "coupled"
                                ),
                                "snippet": call_snippet,
                                "snippet_sha256": hashlib.sha256(
                                    call_snippet.encode("utf-8")
                                ).hexdigest(),
                                "control_guards": control_guards(
                                    node,
                                    parameter_name,
                                ),
                            }
                        )

            evidence[qualified_name] = {}
            for parameter_name in sorted(parameter_names):
                parameter_uses = sorted(
                    uses_by_parameter[parameter_name],
                    key=lambda item: (item["line"], item["column"], item["snippet"]),
                )
                if not parameter_uses:
                    raise RuntimeError(
                        "default parameter has no source load site: "
                        f"{qualified_name}.{parameter_name}"
                    )
                evidence[qualified_name][parameter_name] = {
                    "source_path": relative_path,
                    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "definition_line": callable_node.lineno,
                    "definition_end_line": callable_node.end_lineno,
                    "parameter_flow_names": sorted(
                        symbol
                        for symbol in flow_names_by_parameter[parameter_name]
                        if "." not in symbol
                    ),
                    "parameter_flow_attributes": sorted(
                        symbol
                        for symbol in flow_names_by_parameter[parameter_name]
                        if "." in symbol
                    ),
                    "flow_bindings": sorted(
                        flow_bindings_by_parameter[parameter_name],
                        key=lambda item: (
                            item["line"],
                            item["target"],
                            item["value"],
                        ),
                    ),
                    "parameter_uses": parameter_uses,
                    "branch_predicates": sorted(
                        predicates_by_parameter[parameter_name],
                        key=lambda item: (
                            item["line"],
                            item["kind"],
                            item["snippet"],
                        ),
                    ),
                    "attribute_branch_predicates": sorted(
                        attribute_predicates_by_parameter[parameter_name],
                        key=lambda item: (
                            item["line"],
                            item["kind"],
                            item["snippet"],
                        ),
                    ),
                    "call_forwards": sorted(
                        calls_by_parameter[parameter_name],
                        key=lambda item: (
                            item["line"],
                            item["callee"],
                            item["argument"],
                            item["snippet"],
                        ),
                    ),
                }
    if set(evidence) != expected:
        raise RuntimeError("source-use evidence callable fixed point changed")
    if sum(len(records) for records in evidence.values()) != 456:
        raise RuntimeError("source-use evidence parameter fixed point changed")
    return {name: evidence[name] for name in sorted(evidence)}


_PINNED_AST_DEFAULT_PARAMETER_USE_EVIDENCE = _source_default_parameter_use_evidence()
SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE = json.loads(
    json.dumps(_PINNED_AST_DEFAULT_PARAMETER_USE_EVIDENCE)
)

# Every default parameter has a literal, source-specific semantic contract.
# The records below were reviewed at exact (qualified callable, parameter) keys;
# no category template, parameter-name classifier, or inherited responsibility
# supplies accepted prose.
DEFAULT_PARAMETER_SEMANTIC_FIELDS = (
    "effect_category",
    "branch_behavior",
    "default_route",
    "alternate_route",
    "validation_and_coupling",
    "consumer",
    "forwarded_to",
    "operation_signature",
    "shared_semantics_with",
    "observable_effect",
)

DEFAULT_PARAMETER_SEMANTICS_PATH = (
    REPOSITORY_ROOT / "audit/default_parameter_semantics.json"
)
DEFAULT_PARAMETER_SEMANTICS_AUTHORITY_SHA256 = (
    "f5a31c24b86b804a171258f0b342b3474a42a2cfda936635beb21177a462ae23"
)

EFFECT_ROLE_TO_CATEGORY = {
    "physical_quantity": "physical",
    "source_identity": "data/source identity",
    "cache_policy": "cache",
    "runtime_environment": "environment",
    "diagnostic_surface": "diagnostic",
    "compatibility_route": "compatibility",
    "parity_or_injection": "parity/injection",
    "unsupported_guard": "unsupported",
    "algorithm_control": "algorithmic",
    "continuous_equation": "continuous-value",
    "dependency_injection": "dependency-injection-with-no-branch",
}

EFFECT_ROLE_DESCRIPTION = {
    "physical_quantity": "physical state or equation",
    "source_identity": "file, table, or source identity",
    "cache_policy": "cache key, reuse, or rebuild policy",
    "runtime_environment": "runtime backend, precision, or environment policy",
    "diagnostic_surface": "diagnostic data, shape, or publication surface",
    "compatibility_route": "compatibility spelling, layout, or fallback route",
    "parity_or_injection": "parity path or injected dependency",
    "unsupported_guard": "supported-versus-rejected execution boundary",
    "algorithm_control": "loop, grid, ordering, tolerance, or algorithm route",
    "continuous_equation": "continuous equation coefficient or output coordinate",
    "dependency_injection": "canonical-versus-injected dependency",
}


def _category_for_effect_role(effect_role: str) -> str:
    """Return the fixed semantic category without consulting a registry."""

    if effect_role == "physical_quantity":
        return "physical"
    if effect_role == "source_identity":
        return "data/source identity"
    if effect_role == "cache_policy":
        return "cache"
    if effect_role == "runtime_environment":
        return "environment"
    if effect_role == "diagnostic_surface":
        return "diagnostic"
    if effect_role == "compatibility_route":
        return "compatibility"
    if effect_role == "parity_or_injection":
        return "parity/injection"
    if effect_role == "unsupported_guard":
        return "unsupported"
    if effect_role == "algorithm_control":
        return "algorithmic"
    if effect_role == "continuous_equation":
        return "continuous-value"
    if effect_role == "dependency_injection":
        return "dependency-injection-with-no-branch"
    raise RuntimeError(f"unknown source effect role: {effect_role}")


def _description_for_effect_role(effect_role: str) -> str:
    """Return the fixed effect label without consulting mutable prose."""

    if effect_role == "physical_quantity":
        return "physical state or equation"
    if effect_role == "source_identity":
        return "file, table, or source identity"
    if effect_role == "cache_policy":
        return "cache key, reuse, or rebuild policy"
    if effect_role == "runtime_environment":
        return "runtime backend, precision, or environment policy"
    if effect_role == "diagnostic_surface":
        return "diagnostic data, shape, or publication surface"
    if effect_role == "compatibility_route":
        return "compatibility spelling, layout, or fallback route"
    if effect_role == "parity_or_injection":
        return "parity path or injected dependency"
    if effect_role == "unsupported_guard":
        return "supported-versus-rejected execution boundary"
    if effect_role == "algorithm_control":
        return "loop, grid, ordering, tolerance, or algorithm route"
    if effect_role == "continuous_equation":
        return "continuous equation coefficient or output coordinate"
    if effect_role == "dependency_injection":
        return "canonical-versus-injected dependency"
    raise RuntimeError(f"unknown source effect role: {effect_role}")


def _read_default_parameter_fact_authority() -> dict[str, Any]:
    """Freshly read the source-anchored fact authority from its literal path."""

    authority_path = (
        Path(__file__).resolve().parents[1] / "audit/default_parameter_semantics.json"
    )
    raw = authority_path.read_bytes()
    # This seal protects repository reproducibility. Semantic validation below
    # does not trust this digest: it verifies each structured anchor against
    # the independently recomputed pinned AST.
    expected_raw_sha256 = (
        "f5a31c24b86b804a171258f0b342b3474a42a2cfda936635beb21177a462ae23"
    )
    actual_raw_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_raw_sha256 != expected_raw_sha256:
        raise RuntimeError(
            "default-parameter fact authority changed on disk: "
            f"expected {expected_raw_sha256}, got {actual_raw_sha256}"
        )
    authority = json.loads(raw)
    if set(authority) != {
        "authority_kind",
        "pinned_payne_zero_commit",
        "record_count",
        "records",
        "schema_version",
    }:
        raise RuntimeError("default-parameter fact authority schema changed")
    if authority["authority_kind"] != ("source_anchored_default_parameter_semantics"):
        raise RuntimeError("default-parameter fact authority kind changed")
    if authority["schema_version"] != 2:
        raise RuntimeError("default-parameter fact schema version changed")
    if authority["pinned_payne_zero_commit"] != PINNED_PAYNE_ZERO_COMMIT:
        raise RuntimeError("default-parameter fact source commit changed")
    if authority["record_count"] != 456 or len(authority["records"]) != 456:
        raise RuntimeError("default-parameter fact record count changed")
    return authority


def _anchor_from_source_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Choose the exact control/consumer/use node that anchors effect review."""

    if evidence["branch_predicates"]:
        item = evidence["branch_predicates"][0]
        return {
            "evidence_kind": "predicate",
            "line": item["line"],
            "node_kind": item["kind"],
            "relation": item["relation"],
            "snippet_sha256": item["snippet_sha256"],
        }
    if evidence["call_forwards"]:
        item = evidence["call_forwards"][0]
        return {
            "argument": item["argument"],
            "callee": item["callee"],
            "evidence_kind": "call",
            "line": item["line"],
            "relation": item["relation"],
            "snippet_sha256": item["snippet_sha256"],
        }
    item = evidence["parameter_uses"][-1]
    return {
        "column": item["column"],
        "evidence_kind": "use",
        "line": item["line"],
        "snippet_sha256": item["snippet_sha256"],
    }


def _forward_target_parameter_operations(
    target: dict[str, str],
) -> dict[str, Any]:
    """Recover exact downstream operations for one declared forwarding edge."""

    absolute_path = PINNED_PAYNE_ZERO_SOURCE_ROOT / target["source_path"]
    source_bytes = absolute_path.read_bytes()
    source_text = source_bytes.decode("utf-8")
    tree = ast.parse(source_text)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == target["callable"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"forward target callable is not unique: {target}")
    function = matches[0]
    parameter_names = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    if target["parameter"] not in parameter_names:
        raise RuntimeError(f"forward target parameter is absent: {target}")

    def symbol(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = symbol(node.value)
            return f"{prefix}.{node.attr}" if prefix is not None else None
        return None

    def loaded(node: ast.AST) -> set[str]:
        return {
            name
            for child in ast.walk(node)
            if isinstance(child, (ast.Name, ast.Attribute))
            and isinstance(child.ctx, ast.Load)
            if (name := symbol(child)) is not None
        }

    def stored(node: ast.AST) -> set[str]:
        return {
            name
            for child in ast.walk(node)
            if isinstance(child, (ast.Name, ast.Attribute))
            and isinstance(child.ctx, ast.Store)
            if (name := symbol(child)) is not None
        }

    nested_ranges = [
        (statement.lineno, statement.end_lineno)
        for statement in function.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    body_nodes = [
        child
        for statement in function.body
        for child in ast.walk(statement)
        if not any(
            start <= getattr(child, "lineno", -1) <= end for start, end in nested_ranges
        )
    ]
    flow_names = {target["parameter"]}
    changed = True
    while changed:
        changed = False
        for node in body_nodes:
            value: ast.AST | None = None
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                value = node.value
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                value = node.value
                targets = [node.target]
            elif isinstance(node, ast.NamedExpr):
                value = node.value
                targets = [node.target]
            if value is None or not (loaded(value) & flow_names):
                continue
            new_names = set().union(*(stored(item) for item in targets)) - flow_names
            if new_names:
                flow_names.update(new_names)
                changed = True

    parent_by_node = {
        child: parent
        for parent in ast.walk(function)
        for child in ast.iter_child_nodes(parent)
    }

    def guards(node: ast.AST) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        child = node
        parent = parent_by_node.get(child)
        while parent is not None and parent is not function:
            snippet: str | None = None
            polarity: str | None = None
            kind: str | None = None
            if isinstance(parent, (ast.If, ast.While)):
                snippet = ast.get_source_segment(source_text, parent.test)
                kind = "if" if isinstance(parent, ast.If) else "while"
                polarity = "true" if child in parent.body else "false"
            elif isinstance(parent, ast.IfExp):
                snippet = ast.get_source_segment(source_text, parent.test)
                kind = "if_expression"
                polarity = "true" if child is parent.body else "false"
            elif isinstance(parent, ast.ExceptHandler):
                exception = (
                    ast.get_source_segment(source_text, parent.type)
                    if parent.type is not None
                    else "BaseException"
                )
                snippet = f"except {exception}"
                kind = "except"
                polarity = "handler"
            if snippet is not None and polarity is not None and kind is not None:
                records.append(
                    {
                        "kind": kind,
                        "line": parent.lineno,
                        "polarity": polarity,
                        "snippet": snippet,
                        "snippet_sha256": hashlib.sha256(
                            snippet.encode("utf-8")
                        ).hexdigest(),
                    }
                )
            child = parent
            parent = parent_by_node.get(child)
        return sorted(
            records,
            key=lambda item: (item["line"], item["kind"], item["polarity"]),
        )

    small_callable = function.end_lineno - function.lineno <= 25
    operations: list[dict[str, Any]] = []
    operation_nodes = (
        ast.Assign,
        ast.AnnAssign,
        ast.Assert,
        ast.AugAssign,
        ast.BoolOp,
        ast.Call,
        ast.ExceptHandler,
        ast.If,
        ast.IfExp,
        ast.NamedExpr,
        ast.Return,
        ast.While,
    )
    for node in body_nodes:
        if not isinstance(node, operation_nodes):
            continue
        node_loads = loaded(node)
        related = bool(node_loads & flow_names)
        terminal_small = small_callable and isinstance(
            node,
            (ast.If, ast.Return, ast.ExceptHandler),
        )
        if not related and not terminal_small:
            continue
        snippet = ast.get_source_segment(source_text, node)
        if snippet is None:
            continue
        record = {
            "guard_path": guards(node),
            "line": node.lineno,
            "node_kind": type(node).__name__,
            "relation": (
                "direct"
                if target["parameter"] in node_loads
                else "coupled"
                if related
                else "target-control"
            ),
            "snippet": snippet,
            "snippet_sha256": hashlib.sha256(snippet.encode("utf-8")).hexdigest(),
        }
        if record not in operations:
            operations.append(record)
    operations.sort(
        key=lambda item: (
            item["line"],
            item["node_kind"],
            item["relation"],
            item["snippet"],
        )
    )
    if not operations:
        raise RuntimeError(f"forward target has no parameter operations: {target}")
    return {
        "callable": target["callable"],
        "definition_end_line": function.end_lineno,
        "definition_line": function.lineno,
        "operations": operations,
        "parameter": target["parameter"],
        "source_path": target["source_path"],
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def _require_pinned_source_line(
    source_path: str,
    line: int,
    expected_snippet: str,
) -> str:
    """Return one exact pinned line after checking its required operation."""

    lines = (
        (PINNED_PAYNE_ZERO_SOURCE_ROOT / source_path)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    actual = lines[line - 1].strip()
    if expected_snippet not in actual:
        raise RuntimeError(
            f"required source operation changed at {source_path}:{line}: {actual}"
        )
    return actual


def _normalized_source_expression(node: ast.AST | None) -> str:
    """Return one location-free, whitespace-normalized source expression."""

    if node is None:
        return "None"
    return " ".join(ast.unparse(node).split())


_CALLABLE_SOURCE_AST_CACHE: dict[
    tuple[str, int | None, str | None],
    tuple[str, ast.FunctionDef | ast.AsyncFunctionDef],
] = {}
_TERMINAL_CONTRACT_CACHE: dict[
    tuple[str, int | None, str | None],
    dict[str, Any],
] = {}
_TERMINAL_CONTRACT_ACTIVE: set[tuple[str, int | None, str | None]] = set()


def _callable_source_ast(
    source_path: str,
    *,
    definition_line: int | None = None,
    callable_name: str | None = None,
) -> tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Freshly recover one exact callable from the pinned checkout."""

    cache_key = (source_path, definition_line, callable_name)
    cached = _CALLABLE_SOURCE_AST_CACHE.get(cache_key)
    if cached is not None:
        return cached
    absolute_path = PINNED_PAYNE_ZERO_SOURCE_ROOT / source_path
    source_bytes = absolute_path.read_bytes()
    source_text = source_bytes.decode("utf-8")
    tree = ast.parse(source_text)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (definition_line is None or node.lineno == definition_line)
        and (callable_name is None or node.name == callable_name)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "source callable is not unique: "
            f"{source_path}:{definition_line}:{callable_name}"
        )
    result = (source_text, matches[0])
    _CALLABLE_SOURCE_AST_CACHE[cache_key] = result
    return result


def _callable_body_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.AST]:
    """Return callable-owned AST nodes while excluding nested definitions."""

    nested = [
        node
        for statement in function.body
        for node in ast.walk(statement)
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
        )
        and node is not function
    ]
    nested_nodes = {child for node in nested for child in ast.walk(node)}
    return [
        child
        for statement in function.body
        for child in ast.walk(statement)
        if child not in nested_nodes
    ]


def _source_node_symbol(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _source_node_symbol(node.value)
        return f"{prefix}.{node.attr}" if prefix is not None else node.attr
    return None


def _source_loaded_symbols(node: ast.AST) -> set[str]:
    return {
        symbol
        for child in ast.walk(node)
        if isinstance(child, (ast.Name, ast.Attribute))
        and isinstance(child.ctx, ast.Load)
        if (symbol := _source_node_symbol(child)) is not None
    }


def _assignment_parts(
    node: ast.AST,
) -> tuple[list[ast.AST], ast.AST | None]:
    if isinstance(node, ast.Assign):
        return list(node.targets), node.value
    if isinstance(node, ast.AnnAssign):
        return [node.target], node.value
    if isinstance(node, ast.AugAssign):
        return [node.target], node.value
    if isinstance(node, ast.NamedExpr):
        return [node.target], node.value
    return [], None


def _source_target_root(node: ast.AST) -> str | None:
    """Return the leftmost named object of one assignment/call target."""

    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _return_value_type(
    value: ast.AST | None,
    assignments: dict[str, list[ast.AST]],
    *,
    seen: frozenset[str] = frozenset(),
) -> tuple[str, int, list[str], int | None]:
    """Infer the observable source-level return type and arity."""

    if value is None or (isinstance(value, ast.Constant) and value.value is None):
        return "NoneType", 0, [], None
    if isinstance(value, ast.Tuple):
        element_types = [
            _return_value_type(item, assignments)[0] for item in value.elts
        ]
        return "tuple", len(value.elts), element_types, len(value.elts)
    if isinstance(value, ast.List):
        return (
            "list",
            1,
            [_return_value_type(item, assignments)[0] for item in value.elts],
            len(value.elts),
        )
    if isinstance(value, ast.Set):
        return (
            "set",
            1,
            [_return_value_type(item, assignments)[0] for item in value.elts],
            len(value.elts),
        )
    if isinstance(value, ast.Dict):
        return "dict", 1, [], len(value.keys)
    if isinstance(value, ast.Constant):
        return type(value.value).__name__, 1, [], None
    if isinstance(value, ast.Call):
        callee = _source_node_symbol(value.func) or _normalized_source_expression(
            value.func
        )
        return callee.rsplit(".", maxsplit=1)[-1], 1, [], None
    if isinstance(value, ast.Name) and value.id not in seen:
        assigned = assignments.get(value.id, [])
        inferred = {
            _return_value_type(
                candidate,
                assignments,
                seen=seen | {value.id},
            )[0]
            for candidate in assigned
        }
        if len(inferred) == 1:
            return next(iter(inferred)), 1, [], None
        return f"symbol:{value.id}", 1, [], None
    if isinstance(value, ast.IfExp):
        body_type = _return_value_type(value.body, assignments)[0]
        else_type = _return_value_type(value.orelse, assignments)[0]
        return f"conditional:{body_type}|{else_type}", 1, [], None
    return type(value).__name__, 1, [], None


def _statement_guarantees_exit(statement: ast.stmt) -> bool:
    if isinstance(statement, (ast.Return, ast.Raise)):
        return True
    if isinstance(statement, ast.If):
        return (
            bool(statement.body)
            and bool(statement.orelse)
            and _statement_guarantees_exit(statement.body[-1])
            and _statement_guarantees_exit(statement.orelse[-1])
        )
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return bool(statement.body) and _statement_guarantees_exit(statement.body[-1])
    if isinstance(statement, ast.Try):
        body_exits = bool(statement.body) and _statement_guarantees_exit(
            statement.body[-1]
        )
        handlers_exit = bool(statement.handlers) and all(
            handler.body and _statement_guarantees_exit(handler.body[-1])
            for handler in statement.handlers
        )
        final_exits = bool(statement.finalbody) and _statement_guarantees_exit(
            statement.finalbody[-1]
        )
        else_exits = not statement.orelse or _statement_guarantees_exit(
            statement.orelse[-1]
        )
        return final_exits or (body_exits and handlers_exit and else_exits)
    return False


def _terminal_contract_from_callable(
    source_path: str,
    *,
    definition_line: int | None = None,
    callable_name: str | None = None,
) -> dict[str, Any]:
    """Derive exact return-versus-mutation facts for one pinned callable."""

    cache_key = (source_path, definition_line, callable_name)
    cached = _TERMINAL_CONTRACT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if cache_key in _TERMINAL_CONTRACT_ACTIVE:
        return {"mutated_targets": []}
    _TERMINAL_CONTRACT_ACTIVE.add(cache_key)
    source_text, function = _callable_source_ast(
        source_path,
        definition_line=definition_line,
        callable_name=callable_name,
    )
    body_nodes = _callable_body_nodes(function)
    parent_by_node = {
        child: parent
        for parent in ast.walk(function)
        for child in ast.iter_child_nodes(parent)
    }
    parameter_names = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    assignments: dict[str, list[ast.AST]] = {}
    assignment_records: list[dict[str, Any]] = []
    mutated_targets: set[str] = set()
    for node in body_nodes:
        targets, value = _assignment_parts(node)
        if not targets or value is None:
            continue
        normalized_targets = [
            _normalized_source_expression(target) for target in targets
        ]
        assignment_records.append(
            {
                "operator": type(node).__name__,
                "targets": normalized_targets,
                "value": _normalized_source_expression(value),
            }
        )
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.setdefault(target.id, []).append(value)
            if (
                isinstance(target, (ast.Attribute, ast.Subscript))
                and _source_target_root(target) in parameter_names
            ):
                mutated_targets.add(
                    _normalized_source_expression(target).split("[", maxsplit=1)[0]
                )

    return_records: list[dict[str, Any]] = []
    for node in body_nodes:
        if not isinstance(node, ast.Return):
            continue
        value_type, arity, element_types, container_cardinality = _return_value_type(
            node.value,
            assignments,
        )
        return_annotation = (
            _normalized_source_expression(function.returns)
            if function.returns is not None
            else None
        )
        if return_annotation not in {None, "None"} and value_type not in {
            "NoneType",
            "tuple",
            "list",
            "set",
            "dict",
        }:
            value_type = return_annotation
        guards: list[dict[str, str]] = []
        child: ast.AST = node
        parent = parent_by_node.get(child)
        while parent is not None and parent is not function:
            if isinstance(parent, (ast.If, ast.While)):
                polarity = (
                    "true"
                    if child in parent.body
                    else "false"
                    if child in parent.orelse
                    else "nested"
                )
                guards.append(
                    {
                        "operator": type(parent).__name__,
                        "predicate": _normalized_source_expression(parent.test),
                        "polarity": polarity,
                    }
                )
            elif isinstance(parent, ast.IfExp):
                polarity = "true" if child is parent.body else "false"
                guards.append(
                    {
                        "operator": "IfExp",
                        "predicate": _normalized_source_expression(parent.test),
                        "polarity": polarity,
                    }
                )
            child = parent
            parent = parent_by_node.get(child)
        return_records.append(
            {
                "node_kind": "Return",
                "value": _normalized_source_expression(node.value),
                "arity": arity,
                "value_type": value_type,
                "element_types": element_types,
                "container_cardinality": container_cardinality,
                "guards": list(reversed(guards)),
            }
        )

    if not function.body or not _statement_guarantees_exit(function.body[-1]):
        return_records.append(
            {
                "node_kind": "ImplicitReturn",
                "value": "None",
                "arity": 0,
                "value_type": "NoneType",
                "element_types": [],
                "container_cardinality": None,
                "guards": [],
            }
        )

    non_none_returns = [
        record for record in return_records if record["value_type"] != "NoneType"
    ]
    declared_none = (
        function.returns is not None
        and _normalized_source_expression(function.returns) == "None"
    )
    if declared_none or not non_none_returns:
        # Propagate only proved writes in same-source callees. Formal-to-actual
        # substitution prevents read-only arrays passed to a None-returning
        # procedure from being mislabeled as mutated.
        for node in body_nodes:
            if not isinstance(node, ast.Call):
                continue
            callee_symbol = _source_node_symbol(node.func)
            if callee_symbol is None:
                continue
            callee_name = callee_symbol.rsplit(".", maxsplit=1)[-1]
            if callee_name == function.name:
                continue
            try:
                _, callee = _callable_source_ast(
                    source_path,
                    callable_name=callee_name,
                )
            except RuntimeError:
                continue
            callee_parameters = [
                argument.arg
                for argument in (
                    *callee.args.posonlyargs,
                    *callee.args.args,
                    *callee.args.kwonlyargs,
                )
            ]
            actual_by_formal = {
                formal: _normalized_source_expression(actual)
                for formal, actual in zip(callee_parameters, node.args)
            }
            actual_by_formal.update(
                {
                    keyword.arg: _normalized_source_expression(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
            )
            callee_terminal = _terminal_contract_from_callable(
                source_path,
                definition_line=callee.lineno,
            )
            for callee_target in callee_terminal["mutated_targets"]:
                formal_root = callee_target.split(".", maxsplit=1)[0].split(
                    "[",
                    maxsplit=1,
                )[0]
                actual = actual_by_formal.get(formal_root)
                if actual is None:
                    continue
                propagated = actual + callee_target[len(formal_root) :]
                try:
                    propagated_root = _source_target_root(
                        ast.parse(propagated, mode="eval").body
                    )
                except (SyntaxError, ValueError):
                    continue
                if propagated_root in parameter_names:
                    mutated_targets.add(propagated.split("[", maxsplit=1)[0])

    if non_none_returns and mutated_targets:
        effect_kind = "return_and_in_place_mutation"
    elif non_none_returns:
        effect_kind = "return_value"
    elif mutated_targets:
        effect_kind = "in_place_mutation"
    else:
        effect_kind = "side_effect_none"

    unique_returns: list[dict[str, Any]] = []
    for record in return_records:
        if record not in unique_returns:
            unique_returns.append(record)
    result = {
        "return_annotation": (
            _normalized_source_expression(function.returns)
            if function.returns is not None
            else None
        ),
        "return_node_count": len(return_records),
        "return_contracts": unique_returns,
        "return_contract_count": len(unique_returns),
        "effect_kind": effect_kind,
        "mutated_targets": sorted(mutated_targets),
        "mutated_target_count": len(mutated_targets),
        "assignment_count": len(assignment_records),
        "assignments": assignment_records,
    }
    _TERMINAL_CONTRACT_CACHE[cache_key] = result
    _TERMINAL_CONTRACT_ACTIVE.remove(cache_key)
    return result


def _cardinality_word(value: int) -> str:
    if value < 0:
        raise RuntimeError("operation cardinality cannot be negative")
    small = (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    )
    if value < len(small):
        return small[value]
    tens = (
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    )
    if value < 100:
        quotient, remainder = divmod(value, 10)
        return tens[quotient] + (f"-{small[remainder]}" if remainder else "")
    if value < 1000:
        quotient, remainder = divmod(value, 100)
        return (
            small[quotient]
            + "-hundred"
            + (f"-{_cardinality_word(remainder)}" if remainder else "")
        )
    quotient, remainder = divmod(value, 1000)
    return (
        _cardinality_word(quotient)
        + "-thousand"
        + (f"-{_cardinality_word(remainder)}" if remainder else "")
    )


def _location_free_operation_family(
    signature: dict[str, Any],
) -> dict[str, Any]:
    """Remove source locations while retaining every semantic operation value."""

    return {
        "schema_version": signature["schema_version"],
        "call_count": signature["call_count"],
        "calls": signature["calls"],
        "binding_count": signature["binding_count"],
        "bindings": signature["bindings"],
        "assignment_count": signature["assignment_count"],
        "assignments": signature["assignments"],
        "branch_count": signature["branch_count"],
        "branches": signature["branches"],
        "terminal_contract": signature["terminal_contract"],
        "downstream_target_count": signature["downstream_target_count"],
        "downstream_targets": [
            {key: value for key, value in target.items() if key != "source_path"}
            for target in signature["downstream_targets"]
        ],
        "concrete_quantity": {
            "returned_values": signature["concrete_quantity"]["returned_values"],
            "mutated_targets": signature["concrete_quantity"]["mutated_targets"],
        },
    }


def _strict_source_operation_signature(
    qualified_name: str,
    parameter_name: str,
    evidence: dict[str, Any],
    outcomes: dict[tuple[int, str], bool | None],
    forwarded_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the recurrence oracle from exact source operations and terminals."""

    source_text, function = _callable_source_ast(
        evidence["source_path"],
        definition_line=evidence["definition_line"],
    )
    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if source_hash != evidence["source_sha256"]:
        raise RuntimeError(
            f"operation-signature source changed for {qualified_name}.{parameter_name}"
        )
    # The existing parameter evidence follows captured defaults into nested
    # local callables and comprehensions. Match that exact source envelope for
    # parameter operations; terminal Return nodes are summarized separately
    # from the outer callable only.
    body_nodes = [child for statement in function.body for child in ast.walk(statement)]
    flow_names = set(evidence["parameter_flow_names"]) | set(
        evidence["parameter_flow_attributes"]
    )
    call_evidence = {
        (item["line"], item["callee"], item["snippet"])
        for item in evidence["call_forwards"]
    }
    calls: list[dict[str, Any]] = []
    for node in body_nodes:
        if not isinstance(node, ast.Call):
            continue
        callee = _normalized_source_expression(node.func)
        snippet = ast.get_source_segment(source_text, node)
        if snippet is None or (node.lineno, callee, snippet) not in call_evidence:
            continue
        affected_arguments = sorted(
            {
                item["argument"]
                for item in evidence["call_forwards"]
                if item["line"] == node.lineno
                and item["callee"] == callee
                and item["snippet"] == snippet
            }
        )
        calls.append(
            {
                "operator": "Call",
                "callee": callee,
                "positional_arguments": [
                    _normalized_source_expression(argument) for argument in node.args
                ],
                "keyword_arguments": [
                    {
                        "name": keyword.arg if keyword.arg is not None else "**mapping",
                        "value": _normalized_source_expression(keyword.value),
                    }
                    for keyword in node.keywords
                ],
                "affected_arguments": affected_arguments,
                "relation": (
                    "direct"
                    if parameter_name in _source_loaded_symbols(node)
                    else "coupled"
                ),
            }
        )

    relevant_predicate_lines = {item["line"] for item in evidence["branch_predicates"]}
    parent_by_node = {
        child: parent
        for parent in ast.walk(function)
        for child in ast.iter_child_nodes(parent)
    }

    def parameter_guarded(node: ast.AST) -> bool:
        parent = parent_by_node.get(node)
        while parent is not None and parent is not function:
            if getattr(parent, "lineno", None) in relevant_predicate_lines:
                return True
            parent = parent_by_node.get(parent)
        return False

    assignments: list[dict[str, Any]] = []
    for node in body_nodes:
        targets, value = _assignment_parts(node)
        if not targets or value is None:
            continue
        if not (_source_loaded_symbols(value) & flow_names or parameter_guarded(node)):
            continue
        assignments.append(
            {
                "operator": type(node).__name__,
                "targets": [
                    _normalized_source_expression(target) for target in targets
                ],
                "value": _normalized_source_expression(value),
            }
        )

    bindings = [
        {
            "operator": "derived_assignment",
            "target": item["target"],
            "value": " ".join(item["value"].split()),
            "value_ast": item["value_ast"],
            "relation": item["relation"],
        }
        for item in evidence["flow_bindings"]
    ]
    branches = [
        {
            "operator": item["kind"],
            "predicate": _normalized_source_expression(
                ast.parse(" ".join(item["snippet"].split()), mode="eval").body
            ),
            "relation": item["relation"],
            "default_outcome": outcomes[(item["line"], item["snippet"])],
            "true_head": _one_line(item["body_head"]),
            "false_head": _one_line(item["else_head"]),
        }
        for item in evidence["branch_predicates"]
    ]
    downstream = [
        {
            "source_path": target["source_path"],
            "callable": target["callable"],
            "parameter": target["parameter"],
            "operation_count": len(target["operation_facts"]["operations"]),
            "operations": [
                {
                    "operator": operation["node_kind"],
                    "relation": operation["relation"],
                    "value": _one_line(operation["snippet"]),
                }
                for operation in target["operation_facts"]["operations"]
            ],
            "terminal_contract": _terminal_contract_from_callable(
                target["source_path"],
                callable_name=target["callable"],
            ),
        }
        for target in forwarded_targets
    ]
    terminal = _terminal_contract_from_callable(
        evidence["source_path"],
        definition_line=evidence["definition_line"],
    )
    signature = {
        "schema_version": 1,
        "call_count": len(calls),
        "calls": calls,
        "binding_count": len(bindings),
        "bindings": bindings,
        "assignment_count": len(assignments),
        "assignments": assignments,
        "branch_count": len(branches),
        "branches": branches,
        "terminal_contract": terminal,
        "downstream_target_count": len(downstream),
        "downstream_targets": downstream,
        "concrete_quantity": {
            "source_parameter": parameter_name,
            "returned_values": [
                record["value"] for record in terminal["return_contracts"]
            ],
            "mutated_targets": terminal["mutated_targets"],
        },
    }
    location_free_family = _location_free_operation_family(signature)
    signature["location_free_family_sha256"] = hashlib.sha256(
        json.dumps(
            location_free_family,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return signature


def _terminal_contract_text(terminal: dict[str, Any]) -> str:
    """Render one concrete return/mutation contract without vague alternatives."""

    return_parts = [
        (
            f"{record['value_type']} with {_cardinality_word(record['arity'])} "
            f"value slot(s), source value `{record['value']}`"
        )
        for record in terminal["return_contracts"]
    ]
    if terminal["effect_kind"] == "in_place_mutation":
        return "returns `None` and updates " + ", ".join(
            f"`{target}`" for target in terminal["mutated_targets"]
        )
    if terminal["effect_kind"] == "side_effect_none":
        return "returns `None` after its exact side-effect calls"
    if terminal["effect_kind"] == "return_and_in_place_mutation":
        return (
            "returns "
            + "; ".join(return_parts)
            + " and also updates "
            + ", ".join(f"`{target}`" for target in terminal["mutated_targets"])
        )
    return "returns " + "; ".join(return_parts)


def _source_terminal_observable_effect(
    qualified_name: str,
    parameter_name: str,
    source_default: str,
    operation_signature: dict[str, Any],
) -> str:
    """Render exact local operations and the source terminal state."""

    terminal = operation_signature["terminal_contract"]
    branch_text = (
        ", ".join(
            (
                f"{branch['operator']} `{branch['predicate']}` -> "
                f"{branch['default_outcome']}"
            )
            for branch in operation_signature["branches"]
        )
        or "no parameter-dependent selector"
    )
    call_text = (
        "; ".join(
            (
                f"`{call['callee']}` positional={call['positional_arguments']} "
                f"keywords={call['keyword_arguments']} "
                f"binding={call['affected_arguments']}"
            )
            for call in operation_signature["calls"]
        )
        or "no parameter-forwarding call"
    )
    assignment_text = (
        "; ".join(
            f"`{item['operator']} {item['targets']} = {item['value']}`"
            for item in operation_signature["assignments"]
        )
        or "no parameter-controlled assignment"
    )
    downstream_text = "; ".join(
        (
            f"`{target['callable']}.{target['parameter']}` executes "
            f"{_cardinality_word(target['operation_count'])} operation(s) and "
            f"{_terminal_contract_text(target['terminal_contract'])}"
        )
        for target in operation_signature["downstream_targets"]
    )
    if downstream_text:
        downstream_text = " Downstream, " + downstream_text + "."
    concrete_targets = (
        operation_signature["concrete_quantity"]["mutated_targets"]
        or operation_signature["concrete_quantity"]["returned_values"]
    )
    concrete_text = ", ".join(f"`{value}`" for value in concrete_targets)
    if not concrete_text:
        concrete_text = "the explicit side effects named by the call records"
    return (
        f"In `{qualified_name}`, source parameter `{parameter_name}` controls the "
        f"concrete physical or numerical terminal quantity {concrete_text}. "
        f"With source default `{source_default}`, "
        f"the exact branch outcome is {branch_text}. "
        f"The source executes {_cardinality_word(operation_signature['call_count'])} "
        f"parameter call operation(s): {call_text}; and "
        f"{_cardinality_word(operation_signature['assignment_count'])} "
        f"parameter-controlled assignment(s): {assignment_text}. "
        f"The terminal contract {_terminal_contract_text(terminal)}." + downstream_text
    )


def _raw_effect_role_signal_from_source(
    qualified_name: str,
    parameter_name: str,
    effect_role: str,
    evidence: dict[str, Any],
) -> bool:
    """Return one raw semantic signal before exclusive-route exclusions.

    This helper is deliberately not an acceptance oracle.  Some source
    operations expose several low-level signals (for example a path entering
    an injected table call).  The total decision function below applies the
    mutually exclusive semantic route.  Public support is defined only as
    equality to that one result.
    """

    leaf = qualified_name.rsplit(".", maxsplit=1)[-1].lower()
    parameter = parameter_name.lower()
    source_facts = " ".join(
        [
            *(
                item["value"] if field_name == "flow_bindings" else item["snippet"]
                for field_name in (
                    "branch_predicates",
                    "call_forwards",
                    "flow_bindings",
                    "parameter_uses",
                )
                for item in evidence[field_name]
            ),
            *(item["callee"] for item in evidence["call_forwards"]),
            *(item["argument"] for item in evidence["call_forwards"]),
            *(item["target"] for item in evidence["flow_bindings"]),
            *(item["value"] for item in evidence["flow_bindings"]),
        ]
    ).lower()

    if effect_role == "runtime_environment":
        return parameter == "argv" or any(
            token in parameter for token in ("device", "dtype")
        )
    if effect_role == "unsupported_guard":
        return (
            parameter.startswith("enable_experimental")
            or parameter == "assert_no_saturated_core"
        ) and any(
            token in source_facts
            for token in (
                "enable_experimental",
                "require",
                "opt_in",
                "notimplemented",
                "saturated",
            )
        )
    if effect_role == "dependency_injection":
        return (
            parameter in {"basis", "tables"}
            and any(
                token in source_facts
                for token in (
                    "build_fast_exponential_tables",
                    "build_voigt_profile_basis",
                    "load_hydrogen_line_profile_tables",
                )
            )
            and any(token in source_facts for token in (" is none", " or build_"))
        )
    if effect_role == "diagnostic_surface":
        return (
            "diagnostic" in leaf
            or parameter
            in {
                "completed_iterations",
                "converged",
                "diagnostics",
                "keep_slabs",
                "output_line_mass_absorption_coefficient",
                "return_diagnostics",
                "selected_lines_output",
                "source",
                "title",
            }
            and (parameter != "source" or leaf == "parse_atmosphere_deck")
        )
    if effect_role == "cache_policy":
        return parameter in {
            "cache_dir",
            "force",
            "force_reload",
            "metal_chunk",
            "rebuild",
            "temperature_iteration_index",
        } and any(
            token in source_facts
            for token in (
                "cache",
                "chunk",
                "exists",
                "force",
                "iteration",
                "manifest",
                "rebuild",
            )
        )
    if effect_role == "compatibility_route":
        return (
            parameter
            in {
                "c_over_m",
                "carbon_over_m",
                "detect_swapped_layout",
                "error_type",
                "fe_over_h",
                "n_over_m",
                "nitrogen_over_m",
                "o_over_m",
                "oxygen_over_m",
                "resolution",
                "x_over_h",
            }
            and (parameter != "resolution" or leaf == "synthesize_from_labels")
            and any(
                token in source_facts
                for token in (
                    "alias",
                    "error",
                    "native",
                    "resolution",
                    "swapped",
                    "c_over_m",
                    "n_over_m",
                    "o_over_m",
                    "fe_over_h",
                    "x_over_h",
                    "[c/m]",
                    "[n/m]",
                    "[o/m]",
                    "[fe/h]",
                )
            )
        )
    if effect_role == "source_identity":
        return (
            any(
                token in parameter
                for token in (
                    "catalog_path",
                    "checkpoint_path",
                    "checksum_path",
                    "continuum_tables_path",
                    "edge_grid_path",
                    "five_label_path",
                    "generated_manifest_path",
                    "manifest_path",
                    "molecules_path",
                    "path",
                    "root",
                    "source_path",
                    "tables_path",
                    "transfer_tables_path",
                )
            )
            or parameter == "include_direct_xh"
        ) and any(
            token in source_facts
            for token in (
                "path",
                "root",
                "catalog",
                "checksum",
                "manifest",
                "np.load",
                "source",
                "tables",
            )
        )
    if effect_role == "continuous_equation":
        return (
            parameter
            in {
                "floor",
                "initializer_jitter_scale",
                "jitter_scale",
                "mixing_length",
                "pressure_constant",
                "smooth_center_weight",
                "smooth_left_weight",
                "smooth_right_weight",
                "standard_log_tau_start",
                "standard_log_tau_step",
            }
            and (
                parameter != "mixing_length"
                or leaf in {"apply_temperature_correction", "finalize_transfer_state"}
            )
            and any(
                token in source_facts
                for token in (
                    "float(",
                    "maximum",
                    "mixing_length",
                    "pressure",
                    "smooth",
                    "standard_log_tau",
                    "jitter",
                )
            )
        )
    if effect_role == "algorithm_control":
        algorithm_tokens = (
            "apply_ground_partition",
            "chunk",
            "chain_length",
            "convection_enabled",
            "energy_first",
            "entries_per_layer",
            "family",
            "enabled",
            "frequency_bin",
            "frequency_per_bin",
            "frequency_start",
            "frequency_stop",
            "include_",
            "index",
            "initializer",
            "iteration",
            "max_",
            "minimum_",
            "mode",
            "n_elements",
            "nion",
            "population_mode",
            "predicted",
            "perturbation",
            "r_grid",
            "replace",
            "resolution",
            "initializer_seed",
            "smooth_start",
            "smooth_stop",
            "sort",
            "sweep",
            "symmetric",
            "tol",
            "use_",
            "wavelength_start_index",
            "wavelength_stop_index",
            "wing_mode",
            "zero_top_layer_count",
        )
        return (
            any(token in parameter for token in algorithm_tokens)
            or (
                leaf == "molecular_chunk_lines"
                and parameter == "default"
                and "chunk" in source_facts
            )
            or (leaf == "deterministic_initializer_labels" and parameter == "seed")
        )
    if effect_role == "physical_quantity":
        return any(
            token in parameter
            for token in (
                "abundance",
                "alpha_enhancement",
                "carbon_enhancement",
                "apply_iso_corr",
                "apply_stim",
                "charge_square_density",
                "do_helium",
                "do_metal",
                "adiabatic_gradient",
                "departure_coefficient",
                "hydrogen_departure",
                "log_strength",
                "mass_density",
                "mean_nuclear_mass",
                "metallicity",
                "microturbulence",
                "mixing_length",
                "molecular_lines",
                "molecular_species",
                "molecules",
                "overshoot_weight",
                "nitrogen_enhancement",
                "oxygen_enhancement",
                "requested_maximum_velocity",
                "surface_gravity",
                "wavelength_end",
                "wavelength_start",
                "wl_end",
                "wl_start",
            )
        )
    if effect_role == "parity_or_injection":
        return any(
            token in parameter
            for token in (
                "accumulator",
                "atomic_masses",
                "base_line_",
                "catalog",
                "convective_",
                "density_minus",
                "density_plus",
                "electron_density_seed",
                "flags",
                "flux",
                "frequency_invariants",
                "heat_capacity",
                "integrated_radiation_pressure",
                "karzas_tables",
                "logarithmic_temperature_pressure_gradient",
                "molecular_populations",
                "molecular_state",
                "molecular_tables",
                "molecule_codes",
                "opacity_flags",
                "partition",
                "pops",
                "population_slot",
                "population_state",
                "pressure_scale_height",
                "previous_convective_flux",
                "rosseland_",
                "setup",
                "specific_internal_energy_",
                "spectral_operator",
                "stark_damping",
                "state",
                "tables",
                "temperature_correction_state",
                "thermal_energy",
                "total_pressure",
                "turbulent_pressure",
                "van_der_waals",
                "window_invariants",
            )
        )
    return False


def _source_derived_effect_role(
    qualified_name: str,
    parameter_name: str,
    evidence: dict[str, Any],
) -> str:
    """Classify one key through one total exclusive source-operation path."""

    required_lists = (
        "parameter_uses",
        "branch_predicates",
        "attribute_branch_predicates",
        "call_forwards",
        "flow_bindings",
    )
    if not isinstance(evidence, dict) or any(
        not isinstance(evidence.get(field_name), list) for field_name in required_lists
    ):
        raise RuntimeError(
            "source operations are absent or malformed for "
            f"{qualified_name}.{parameter_name}"
        )
    if not evidence["parameter_uses"]:
        raise RuntimeError(
            "source operations contain no exact parameter load for "
            f"{qualified_name}.{parameter_name}"
        )
    operation_text = " ".join(
        [
            *(item["snippet"] for item in evidence["parameter_uses"]),
            *(item["snippet"] for item in evidence["branch_predicates"]),
            *(item["snippet"] for item in evidence["call_forwards"]),
            *(item["value"] for item in evidence["flow_bindings"]),
        ]
    ).lower()
    if parameter_name.lower() not in operation_text:
        raise RuntimeError(
            "source operations do not contain the signature parameter for "
            f"{qualified_name}.{parameter_name}"
        )

    # The order encodes negative exclusions: dedicated unsupported,
    # constructor-injection, runtime, publication, cache, and compatibility
    # operations are removed before broader path/control/physical families.
    # Consequently the compatibility alias ``resolution`` cannot fall through
    # to algorithm control, and canonical constructor injection cannot fall
    # through to parity/injection.
    precedence = (
        "unsupported_guard",
        "dependency_injection",
        "runtime_environment",
        "diagnostic_surface",
        "cache_policy",
        "compatibility_route",
        "source_identity",
        "continuous_equation",
        "algorithm_control",
        "physical_quantity",
        "parity_or_injection",
    )
    for effect_role in precedence:
        if _raw_effect_role_signal_from_source(
            qualified_name,
            parameter_name,
            effect_role,
            evidence,
        ):
            return effect_role
    raise RuntimeError(
        "source operations do not determine an effect role for "
        f"{qualified_name}.{parameter_name}"
    )


def _effect_role_supported_by_source(
    qualified_name: str,
    parameter_name: str,
    effect_role: str,
    evidence: dict[str, Any],
) -> bool:
    """Return whether fresh operations yield exactly this one semantic role."""

    if effect_role not in EFFECT_ROLE_TO_CATEGORY:
        return False
    try:
        return (
            _source_derived_effect_role(
                qualified_name,
                parameter_name,
                evidence,
            )
            == effect_role
        )
    except RuntimeError:
        return False


def _validate_source_derived_effect_role(
    qualified_name: str,
    parameter_name: str,
    authority_role: str,
    evidence: dict[str, Any],
) -> None:
    """Reject every authority role except the one exclusive source result."""

    derived_role = _source_derived_effect_role(
        qualified_name,
        parameter_name,
        evidence,
    )
    if authority_role != derived_role:
        raise RuntimeError(
            f"effect role differs from exclusive source-operation role for "
            f"{(qualified_name, parameter_name)}: authority={authority_role}, "
            f"source={derived_role}"
        )


_UNKNOWN_SOURCE_VALUE = object()


def _source_symbol(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _source_symbol(node.value)
        return f"{prefix}.{node.attr}" if prefix is not None else None
    return None


def _evaluate_source_expression(
    expression: str,
    values: dict[str, Any],
) -> Any:
    """Evaluate the small pure AST subset used by default-route selectors."""

    node = ast.parse(" ".join(expression.split()), mode="eval").body

    def evaluate(item: ast.AST) -> Any:
        if isinstance(item, ast.Constant):
            return item.value
        if isinstance(item, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
            try:
                return ast.literal_eval(item)
            except (SyntaxError, ValueError):
                return _UNKNOWN_SOURCE_VALUE
        if isinstance(item, (ast.Name, ast.Attribute)):
            symbol = _source_symbol(item)
            return values.get(symbol, _UNKNOWN_SOURCE_VALUE)
        if isinstance(item, ast.UnaryOp):
            operand = evaluate(item.operand)
            if operand is _UNKNOWN_SOURCE_VALUE:
                return operand
            if isinstance(item.op, ast.Not):
                return not bool(operand)
            if isinstance(item.op, ast.USub):
                return -operand
            if isinstance(item.op, ast.UAdd):
                return +operand
            return _UNKNOWN_SOURCE_VALUE
        if isinstance(item, ast.BoolOp):
            if isinstance(item.op, ast.And):
                result: Any = True
                for operand in item.values:
                    result = evaluate(operand)
                    if result is _UNKNOWN_SOURCE_VALUE:
                        return result
                    if not bool(result):
                        return result
                return result
            if isinstance(item.op, ast.Or):
                result = False
                for operand in item.values:
                    result = evaluate(operand)
                    if result is _UNKNOWN_SOURCE_VALUE:
                        return result
                    if bool(result):
                        return result
                return result
        if isinstance(item, ast.Compare):
            left = evaluate(item.left)
            comparators = [evaluate(value) for value in item.comparators]
            if left is _UNKNOWN_SOURCE_VALUE or any(
                value is _UNKNOWN_SOURCE_VALUE for value in comparators
            ):
                return _UNKNOWN_SOURCE_VALUE
            values_to_compare = [left, *comparators]
            outcomes: list[bool] = []
            for operation, lhs, rhs in zip(
                item.ops,
                values_to_compare,
                values_to_compare[1:],
            ):
                if isinstance(operation, ast.Is):
                    outcomes.append(lhs is rhs)
                elif isinstance(operation, ast.IsNot):
                    outcomes.append(lhs is not rhs)
                elif isinstance(operation, ast.Eq):
                    outcomes.append(lhs == rhs)
                elif isinstance(operation, ast.NotEq):
                    outcomes.append(lhs != rhs)
                elif isinstance(operation, ast.Lt):
                    outcomes.append(lhs < rhs)
                elif isinstance(operation, ast.LtE):
                    outcomes.append(lhs <= rhs)
                elif isinstance(operation, ast.Gt):
                    outcomes.append(lhs > rhs)
                elif isinstance(operation, ast.GtE):
                    outcomes.append(lhs >= rhs)
                elif isinstance(operation, ast.In):
                    outcomes.append(lhs in rhs)
                elif isinstance(operation, ast.NotIn):
                    outcomes.append(lhs not in rhs)
                else:
                    return _UNKNOWN_SOURCE_VALUE
            return all(outcomes)
        if isinstance(item, ast.IfExp):
            selector = evaluate(item.test)
            if selector is _UNKNOWN_SOURCE_VALUE:
                return selector
            return evaluate(item.body if bool(selector) else item.orelse)
        if isinstance(item, ast.BinOp):
            left = evaluate(item.left)
            right = evaluate(item.right)
            if left is _UNKNOWN_SOURCE_VALUE or right is _UNKNOWN_SOURCE_VALUE:
                return _UNKNOWN_SOURCE_VALUE
            try:
                if isinstance(item.op, ast.Add):
                    return left + right
                if isinstance(item.op, ast.Sub):
                    return left - right
                if isinstance(item.op, ast.Mult):
                    return left * right
                if isinstance(item.op, ast.Div):
                    return left / right
                if isinstance(item.op, ast.FloorDiv):
                    return left // right
                if isinstance(item.op, ast.Mod):
                    return left % right
                if isinstance(item.op, ast.Pow):
                    return left**right
            except (TypeError, ValueError, ZeroDivisionError):
                return _UNKNOWN_SOURCE_VALUE
        if isinstance(item, ast.Call):
            callee = _source_symbol(item.func)
            arguments = [evaluate(argument) for argument in item.args]
            if any(value is _UNKNOWN_SOURCE_VALUE for value in arguments):
                return _UNKNOWN_SOURCE_VALUE
            functions = {
                "abs": abs,
                "bool": bool,
                "float": float,
                "int": int,
                "max": max,
                "min": min,
                "np.isfinite": lambda value: (
                    isinstance(value, (int, float))
                    and value == value
                    and value not in {float("inf"), float("-inf")}
                ),
            }
            function = functions.get(callee)
            if function is None:
                return _UNKNOWN_SOURCE_VALUE
            try:
                return function(*arguments)
            except (TypeError, ValueError):
                return _UNKNOWN_SOURCE_VALUE
        return _UNKNOWN_SOURCE_VALUE

    return evaluate(node)


def _default_predicate_outcomes(
    parameter_name: str,
    source_default: str,
    evidence: dict[str, Any],
) -> dict[tuple[int, str], bool | None]:
    values: dict[str, Any] = {}
    try:
        values[parameter_name] = ast.literal_eval(source_default)
    except (SyntaxError, ValueError):
        values[parameter_name] = _UNKNOWN_SOURCE_VALUE
    for binding in evidence["flow_bindings"]:
        value = _evaluate_source_expression(binding["value"], values)
        if value is not _UNKNOWN_SOURCE_VALUE:
            values[binding["target"]] = value
    outcomes: dict[tuple[int, str], bool | None] = {}
    for predicate in evidence["branch_predicates"]:
        expression = predicate["snippet"]
        if predicate["kind"] == "selection_expression":
            expression_node = ast.parse(
                " ".join(expression.split()),
                mode="eval",
            ).body
            if not isinstance(expression_node, ast.BoolOp):
                raise RuntimeError("selection expression is not a boolean operation")
            expression = ast.unparse(expression_node.values[0])
        value = _evaluate_source_expression(expression, values)
        outcomes[(predicate["line"], predicate["snippet"])] = (
            None if value is _UNKNOWN_SOURCE_VALUE else bool(value)
        )
    return outcomes


def _one_line(source: str | None) -> str:
    if source is None:
        return "<no explicit else; execution continues>"
    return " ".join(source.split())


def _guard_description(guards: list[dict[str, Any]]) -> str:
    if not guards:
        return "unguarded"
    return " guarded by " + ", ".join(
        f"L{guard['line']} `{_one_line(guard['snippet'])}`={guard['polarity']}"
        for guard in guards
    )


def _render_source_contract(
    qualified_name: str,
    parameter_name: str,
    fact: dict[str, Any],
    source_record: dict[str, str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Render prose deterministically from verified source/control-flow facts."""

    role = fact["effect_role"]
    category = _category_for_effect_role(role)
    source_default = source_record["source_default"]
    predicates = evidence["branch_predicates"]
    calls = evidence["call_forwards"]
    uses = evidence["parameter_uses"]
    rendered_forwarded_to = [
        {
            **target,
            "operation_facts": _forward_target_parameter_operations(target),
        }
        for target in fact["forwarded_to"]
    ]
    downstream_operations = [
        (target, operation)
        for target in rendered_forwarded_to
        for operation in target["operation_facts"]["operations"]
    ]
    outcomes = _default_predicate_outcomes(
        parameter_name,
        source_default,
        evidence,
    )
    load_sites = ", ".join(f"L{use['line']}:C{use['column']}" for use in uses)

    if predicates:
        branch_behavior = f"Exact parameter load site(s) {load_sites}. " + " ".join(
            (
                f"L{predicate['line']} {predicate['kind']} "
                f"({predicate['relation']}) tests "
                f"`{_one_line(predicate['snippet'])}`; true starts "
                f"`{_one_line(predicate['body_head'])}`, while "
                "false/fall-through starts "
                f"`{_one_line(predicate['else_head'])}`."
            )
            for predicate in predicates
        )
    elif calls:
        branch_behavior = " ".join(
            (
                f"L{call['line']} passes this value to `{call['callee']}` as "
                f"`{call['argument']}` in `{_one_line(call['snippet'])}`"
                f"{_guard_description(call['control_guards'])}."
            )
            for call in calls[:4]
        )
    else:
        branch_behavior = " ".join(
            f"L{use['line']}:C{use['column']} uses it in "
            f"`{_one_line(use['snippet'])}`"
            f"{_guard_description(use['control_guards'])}."
            for use in uses[:4]
        )

    default_parts: list[str] = []
    for predicate in predicates:
        outcome = outcomes[(predicate["line"], predicate["snippet"])]
        if outcome is None:
            default_parts.append(
                f"L{predicate['line']} `{_one_line(predicate['snippet'])}` also "
                "depends on runtime state, so the source retains both exact heads "
                f"`{_one_line(predicate['body_head'])}` and "
                f"`{_one_line(predicate['else_head'])}`"
            )
        else:
            selected = predicate["body_head"] if outcome else predicate["else_head"]
            default_parts.append(
                f"L{predicate['line']} `{_one_line(predicate['snippet'])}` is "
                f"{outcome}, selecting `{_one_line(selected)}`"
            )
    for call in calls[:4]:
        skipped = False
        for guard in call["control_guards"]:
            outcome = outcomes.get((guard["line"], guard["snippet"]))
            if outcome is not None and outcome != (guard["polarity"] == "true"):
                skipped = True
        if skipped:
            default_parts.append(
                f"the selected guard skips L{call['line']} "
                f"`{_one_line(call['snippet'])}`"
            )
        else:
            default_parts.append(
                f"L{call['line']} consumes it through "
                f"`{_one_line(call['snippet'])}` as "
                f"`{call['callee']}.{call['argument']}`"
                f"{_guard_description(call['control_guards'])}"
            )
    if not default_parts:
        default_parts.extend(
            f"L{use['line']}:C{use['column']} evaluates "
            f"`{_one_line(use['snippet'])}`"
            f"{_guard_description(use['control_guards'])}"
            for use in uses[:3]
        )
    default_route = (
        f"With exact default `{source_default}`, " + "; ".join(default_parts) + "."
    )
    if downstream_operations:
        default_route += " The declared target then executes exact operation(s) " + (
            "; ".join(
                f"`{target['callable']}.{target['parameter']}` L"
                f"{operation['line']} `{_one_line(operation['snippet'])}` "
                f"({operation['relation']})"
                for target, operation in downstream_operations[:8]
            )
            + "."
        )

    if predicates:
        alternate_route = (
            f"A caller value different from `{source_default}` is re-evaluated by "
            + "; ".join(
                (
                    f"L{predicate['line']} `{_one_line(predicate['snippet'])}`, "
                    f"whose true route begins `{_one_line(predicate['body_head'])}` "
                    f"and false route begins `{_one_line(predicate['else_head'])}`"
                )
                for predicate in predicates
            )
            + f"; its exact load site(s) are {load_sites}."
        )
    elif calls:
        alternate_route = (
            f"An explicit value replaces `{source_default}` at "
            + "; ".join(
                (
                    f"L{call['line']} `{_one_line(call['snippet'])}` as "
                    f"`{call['callee']}.{call['argument']}`"
                    f"{_guard_description(call['control_guards'])}"
                )
                for call in calls[:4]
            )
            + "."
        )
    else:
        alternate_route = (
            f"An explicit value replaces `{source_default}` in "
            + "; ".join(
                f"L{use['line']}:C{use['column']} `{_one_line(use['snippet'])}`"
                for use in uses[:4]
            )
            + "."
        )
    if downstream_operations:
        alternate_route += " The same explicit value is re-evaluated by target " + (
            "; ".join(
                f"`{target['callable']}` L{operation['line']} "
                f"`{_one_line(operation['snippet'])}`"
                for target, operation in downstream_operations[:8]
            )
            + "."
        )

    validation_calls = [
        call
        for call in calls
        if call["callee"].rsplit(".", maxsplit=1)[-1]
        in {
            "abs",
            "asarray",
            "bool",
            "device",
            "float",
            "int",
            "isfinite",
            "max",
            "min",
            "require",
            "validate",
        }
    ]
    validation_facts = [
        f"L{predicate['line']} `{_one_line(predicate['snippet'])}` "
        f"({predicate['relation']} {predicate['kind']})"
        for predicate in predicates
    ]
    validation_facts.extend(
        f"L{call['line']} `{_one_line(call['snippet'])}`" for call in validation_calls
    )
    if validation_facts:
        validation_and_coupling = (
            f"From exact load site(s) {load_sites}, validation/control facts are "
            + "; ".join(validation_facts)
            + "."
        )
    else:
        first_consumer = (
            f"L{calls[0]['line']} `{calls[0]['callee']}."
            f"{calls[0]['argument']}` in `{_one_line(calls[0]['snippet'])}`"
            if calls
            else (
                f"L{uses[0]['line']}:C{uses[0]['column']} "
                f"`{_one_line(uses[0]['snippet'])}`"
            )
        )
        validation_and_coupling = (
            f"Exact load site(s) {load_sites} have no parameter-dependent "
            "predicate or coercion call in the defining callable; the source "
            f"consumer is {first_consumer}, so schema/type errors arise there."
        )
    downstream_guards = [
        (target, operation, guard)
        for target, operation in downstream_operations
        for guard in operation["guard_path"]
    ]
    if downstream_guards:
        validation_and_coupling += " Downstream control/validation includes " + (
            "; ".join(
                f"`{target['callable']}` L{guard['line']} "
                f"`{_one_line(guard['snippet'])}`={guard['polarity']} around "
                f"L{operation['line']}"
                for target, operation, guard in downstream_guards[:8]
            )
            + "."
        )

    if calls:
        consumer = f"load site(s) {load_sites} -> " + "; ".join(
            (
                f"`{call['callee']}.{call['argument']}` at L{call['line']} via "
                f"`{_one_line(call['snippet'])}`"
                f"{_guard_description(call['control_guards'])}"
            )
            for call in calls[:4]
        )
    else:
        consumer = f"load site(s) {load_sites} -> " + "; ".join(
            f"L{use['line']}:C{use['column']} `{_one_line(use['snippet'])}`"
            f"{_guard_description(use['control_guards'])}"
            for use in uses[-3:]
        )
    if evidence["flow_bindings"]:
        consumer += "; exact derived assignment(s) " + "; ".join(
            f"L{binding['line']} `{binding['target']} = "
            f"{_one_line(binding['value'])}` ({binding['relation']})"
            for binding in evidence["flow_bindings"][:4]
        )
    if rendered_forwarded_to:
        consumer += "; verified target " + "; ".join(
            f"{target['callable']}.{target['parameter']} in {target['source_path']}"
            for target in rendered_forwarded_to
        )

    operation_signature = _strict_source_operation_signature(
        qualified_name,
        parameter_name,
        evidence,
        outcomes,
        rendered_forwarded_to,
    )

    exact_key = f"{qualified_name}\0{parameter_name}"
    if exact_key == (
        "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__\0source_path"
    ):
        required_binding = next(
            (
                binding
                for binding in evidence["flow_bindings"]
                if binding["target"] == "self._source_from_ref"
                and binding["value"] == "source_path is not None"
            ),
            None,
        )
        required_predicate = next(
            (
                predicate
                for predicate in predicates
                if predicate["line"] == 1145
                and predicate["snippet"] == "self._source_from_ref"
            ),
            None,
        )
        required_call = next(
            (
                call
                for call in calls
                if call["line"] == 1146 and call["callee"] == "np.load"
            ),
            None,
        )
        if not all((required_binding, required_predicate, required_call)):
            raise RuntimeError("source_path LTE route facts changed in pinned source")
        branch_behavior = (
            "L1144 assigns attribute `self._source_from_ref = source_path is not "
            "None`; coupled L1145 `if self._source_from_ref` guards L1146 "
            "`np.load(source_path)`. The false branch assigns both "
            "`self._line_source = None` and `self._line_scattering_ref = None`."
        )
        default_route = (
            "With exact default `None`, L1144 makes `self._source_from_ref` false, "
            "so L1145 skips `np.load`, and L1157–L1159 set both reference tensors "
            "to `None`. `SynthesisPipeline.run` therefore follows its standard "
            "non-reference route and rebuilds the LTE Planck line source."
        )
        alternate_route = (
            "A non-`None` path makes the attribute true, loads the NPZ at L1146, "
            "and materializes its `line_source` and `line_scattering` arrays as "
            "device tensors for the exact-reference source route."
        )
        validation_and_coupling = (
            "The exact selector is the attribute-mediated chain L1144 "
            "`source_path is not None` -> L1145 `self._source_from_ref`; path, "
            "archive-key, shape, dtype, and device errors arise only after the "
            "non-`None` route enters `np.load`/`torch.as_tensor`."
        )
        consumer = (
            "`self._source_from_ref`, guarded `np.load(source_path)`, "
            "`self._line_source`, and `self._line_scattering_ref`; downstream "
            "`SynthesisPipeline.run` selects reference tensors versus LTE Planck "
            "source construction"
        )

    downstream_source = " ".join(
        operation["snippet"] for _, operation in downstream_operations
    )
    if exact_key in {
        "payne_zero_atmosphere.line_profile_math.fast_exponential_lookup\0tables",
        "payne_zero_atmosphere.line_profile_math.evaluate_voigt_profile\0basis",
    }:
        constructor = (
            "build_fast_exponential_tables"
            if parameter_name == "tables"
            else "build_voigt_profile_basis"
        )
        assignment_target = "lookup" if parameter_name == "tables" else "profile_basis"
        if constructor not in " ".join(
            binding["value"] for binding in evidence["flow_bindings"]
        ):
            raise RuntimeError(
                f"canonical dependency constructor changed for {exact_key}"
            )
        default_route = (
            f"With exact default `None`, Python `or` short-circuits past the "
            f"false left operand and deterministically calls `{constructor}()`, "
            f"assigning the canonical result to `{assignment_target}`."
        )
        alternate_route = (
            f"A truthy injected `{parameter_name}` is the selected left operand, "
            f"so `{constructor}()` is not called; a falsey injected object still "
            "selects canonical construction."
        )
        validation_and_coupling = (
            "The selector is object truthiness, not unspecified runtime state. "
            f"The selected `{assignment_target}` supplies the exact lookup/profile "
            "arrays used by the remaining equations."
        )

    if exact_key == (
        "payne_zero_synthesis.molecular_lines.molecular_chunk_lines\0default"
    ):
        required = (
            'raw is None or raw == ""',
            "except ValueError",
            "return int(default)",
            "return max(1, value)",
        )
        if not all(fragment in downstream_source for fragment in required):
            raise RuntimeError("molecular chunk downstream facts changed")
        default_route = (
            "With exact fallback `CHUNK_LINES` (500,000), "
            "`PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES` absent or empty returns "
            "`int(default)`; invalid integer text reaches `except ValueError` and "
            "returns the same fallback; a valid environment integer returns "
            "`max(1, value)`."
        )
        alternate_route = (
            "An explicit fallback replaces 500,000 only for absent, empty, or "
            "invalid environment text. A valid environment value takes precedence "
            "and is clamped to at least one line."
        )
        validation_and_coupling = (
            "Target `_env_positive_int` distinguishes missing/empty text at L124, "
            "invalid integer text at L128–L129, and applies the positive clamp "
            "`max(1, value)` at L130."
        )

    if exact_key in {
        "payne_zero_synthesis.api.synthesize_from_labels\0r_grid",
        "payne_zero_synthesis.api.synthesize_from_labels\0resolution",
    }:
        required = (
            "20_000.0",
            "r_grid and resolution specify different values",
            "r_grid must be finite and positive",
            "return resolved",
        )
        if not all(fragment in downstream_source for fragment in required):
            raise RuntimeError("label-synthesis resolving-grid facts changed")
        if parameter_name == "r_grid":
            default_route = (
                "With both `r_grid=None` and compatibility alias "
                "`resolution=None`, `_resolved_r_grid` returns exactly 20,000.0."
            )
            alternate_route = (
                "A finite positive explicit `r_grid` becomes the synthesis "
                "resolving grid. Supplying both spellings is allowed only when "
                "their float values are exactly equal."
            )
            validation_and_coupling = (
                "Canonical `r_grid` is checked by `_resolved_r_grid`: unequal "
                "dual specification raises, then the selected value is "
                "float-coerced and rejected when non-finite or non-positive."
            )
        else:
            default_route = (
                "With compatibility alias `resolution=None`, canonical `r_grid` "
                "remains authoritative; if both are absent the exact fallback is "
                "20,000.0."
            )
            alternate_route = (
                "An explicit `resolution` supplies the compatibility spelling for "
                "the same resolving grid; when `r_grid` is also present their float "
                "values must be exactly equal."
            )
            validation_and_coupling = (
                "Compatibility alias `resolution` shares `_resolved_r_grid` "
                "validation: it must equal an accompanying canonical `r_grid`; "
                "the selected value is then rejected if non-finite or non-positive."
            )

    if exact_key == (
        "payne_zero_atmosphere.cli.solve_structured_atmosphere"
        "\0initializer_jitter_scale"
    ):
        required = (
            "jitter_scale must be finite and non-negative",
            "jitter_scale must be positive when retries are enabled",
            "projected_target + float(jitter_scale) * widths",
        )
        if not all(fragment in downstream_source for fragment in required):
            raise RuntimeError("initializer jitter downstream facts changed")
        default_route = (
            "The exact 0.01 default is finite, non-negative, and positive for the "
            "default two-trial wrapper, so it reaches deterministic retry-label "
            "construction."
        )
        alternate_route = (
            "A non-finite or negative scale raises immediately; zero is allowed "
            "only when `max_trials == 1`, and raises when retries are enabled."
        )
        validation_and_coupling = (
            "Target L1021 enforces finite/non-negative scale; coupled L1023 "
            "enforces positive scale for more than one trial. L1159 computes each "
            "retry as `projected_target + jitter_scale * widths * direction` and "
            "clips it to checkpoint bounds."
        )

    if exact_key == (
        "payne_zero_atmosphere.runner.finalize_transfer_state\0mixing_length"
    ):
        override_line = _require_pinned_source_line(
            "payne_zero_atmosphere/runner.py",
            1131,
            "mixing_length = setup.convection.mixing_length",
        )
        required = (
            "float(mixing_length) > 0.0 and velocity_coefficient > 0.0",
            "optical_thickness",
            "convection_denominator",
        )
        if not all(fragment in downstream_source for fragment in required):
            raise RuntimeError("forwarded mixing-length target facts changed")
        default_route = (
            "With convection disabled, exact 1.0 is forwarded to "
            "`apply_temperature_correction`. When internal convection is enabled, "
            f"L1131 `{override_line}` replaces the caller value before forwarding."
        )
        alternate_route = (
            "An explicit wrapper value reaches temperature correction only on the "
            "non-internal-convection route; internal convection instead uses "
            "`setup.convection.mixing_length`."
        )
        validation_and_coupling = (
            "The target uses mixing length in velocity, flux, and optical-thickness "
            "equations. Coupled L581 enters the convection-denominator equation "
            "only when both `mixing_length > 0` and `velocity_coefficient > 0`."
        )

    if exact_key == (
        "payne_zero_synthesis.molecular_equilibrium.solve_molecular_equilibrium"
        "\0return_diagnostics"
    ):
        required_lines = (
            (653, "if return_diagnostics:"),
            (654, "diag = {"),
            (663, "return ("),
            (670, "return heavy_nucleus_density"),
        )
        for line, snippet in required_lines:
            _require_pinned_source_line(
                "payne_zero_synthesis/molecular_equilibrium.py",
                line,
                snippet,
            )

    if exact_key == (
        "payne_zero_atmosphere.line_catalog.decode_selected_line_words"
        "\0detect_swapped_layout"
    ):
        required_lines = (
            (117, "_decode_selected_halves(packed_fields, swap_pairs=False)"),
            (118, "if detect_swapped_layout:"),
            (119, "_decode_selected_halves(packed_fields, swap_pairs=True)"),
            (120, "_selected_halfword_score(swapped)"),
            (121, "decoded = swapped"),
            (123, "return SelectedLineCatalog("),
        )
        for line, snippet in required_lines:
            _require_pinned_source_line(
                "payne_zero_atmosphere/line_catalog.py",
                line,
                snippet,
            )

    if exact_key == (
        "payne_zero_atmosphere.microturbulence.standard_microturbulence"
        "\0requested_maximum_velocity"
    ):
        required_lines = (
            (564, "requested_maximum_velocity == -99.0e5"),
            (581, "maximum_velocity = ("),
            (593, "maximum_velocity *= 1.0e5"),
            (595, "maximum_velocity = abs(requested_maximum_velocity)"),
            (601, "return base_profile * maximum_velocity / 1.83e5"),
        )
        for line, snippet in required_lines:
            _require_pinned_source_line(
                "payne_zero_atmosphere/microturbulence.py",
                line,
                snippet,
            )

    if exact_key in {
        "payne_zero_synthesis.device.resolve_runtime\0requested_device",
        "payne_zero_synthesis.device.resolve_runtime\0requested_dtype",
    }:
        required_lines = (
            (42, "runtime_device = ("),
            (47, "runtime_dtype = ("),
            (52, 'runtime_device.type == "mps"'),
            (56, "return runtime_device, runtime_dtype"),
        )
        for line, snippet in required_lines:
            _require_pinned_source_line(
                "payne_zero_synthesis/device.py",
                line,
                snippet,
            )

    if exact_key == (
        "payne_zero_synthesis.continuum.compute_sampled_continuum\0frequency_invariants"
    ):
        required = (
            "FrequencyInvariants layout mismatch",
            "FrequencyInvariants grid size mismatch",
            "batch_terms = frequency_invariants is not None",
            "grid_finalize",
            "return continuum_absorption, continuum_scattering",
        )
        if not all(fragment in downstream_source for fragment in required):
            raise RuntimeError("frequency-invariant downstream facts changed")

    if exact_key == (
        "payne_zero_synthesis.pipeline.SynthesisPipeline.run\0spectral_operator"
    ):
        required = (
            "spectral_operator.convolve_fluxes",
            "spectral_operator.output_wavelength_nm",
            "output_jacobian",
            "return (",
        )
        if not all(fragment in downstream_source for fragment in required):
            raise RuntimeError("spectral-operator downstream facts changed")

    if exact_key == (
        "payne_zero_synthesis.radiative_transfer.solve_spectrum"
        "\0assert_no_saturated_core"
    ):
        required = (
            "if assert_no_saturated_core:",
            "not assert_no_saturated_core and bool(saturated.any())",
            "_saturated_core_flux",
        )
        if not all(fragment in downstream_source for fragment in required):
            raise RuntimeError("saturated-core downstream facts changed")

    if exact_key == (
        "payne_zero_atmosphere.hydrogen_line_profile."
        "molecular_hydrogen_equilibrium_constant\0tables"
    ):
        required_route = next(
            (
                predicate
                for predicate in predicates
                if "load_hydrogen_line_profile_tables"
                in (
                    str(predicate["body_head"])
                    + " "
                    + str(predicate["else_head"])
                    + " "
                    + predicate["snippet"]
                )
            ),
            None,
        )
        if required_route is None:
            raise RuntimeError("hydrogen table-loader route changed")

    # One source-derived renderer owns all 456 terminal effects. The special
    # route blocks above may refine branch/default prose, but they cannot
    # replace the strict operation/return/mutation contract with hand-written
    # terminal claims.
    observable_effect = _source_terminal_observable_effect(
        qualified_name,
        parameter_name,
        source_default,
        operation_signature,
    )

    return {
        "effect_category": category,
        "branch_behavior": branch_behavior,
        "default_route": default_route,
        "alternate_route": alternate_route,
        "validation_and_coupling": validation_and_coupling,
        "consumer": consumer,
        "forwarded_to": rendered_forwarded_to,
        "operation_signature": operation_signature,
        "shared_semantics_with": None,
        "observable_effect": observable_effect,
    }


def _contracts_from_fact_authority(
    authority: dict[str, Any],
    evidence_registry: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Verify 456 source anchors and render their deterministic contracts."""

    expected_record_fields = {
        "category_anchor",
        "effect_role",
        "forwarded_to",
        "parameter_name",
        "qualified_name",
    }
    contracts: dict[str, dict[str, dict[str, Any]]] = {}
    previous_key: tuple[str, str] | None = None
    for fact in authority["records"]:
        if set(fact) != expected_record_fields:
            raise RuntimeError("default-parameter fact record shape changed")
        qualified_name = fact["qualified_name"]
        parameter_name = fact["parameter_name"]
        key = (qualified_name, parameter_name)
        if previous_key is not None and key <= previous_key:
            raise RuntimeError(
                "default-parameter fact records are not unique and sorted"
            )
        previous_key = key
        _category_for_effect_role(fact["effect_role"])
        try:
            evidence = evidence_registry[qualified_name][parameter_name]
            source_record = SOURCE_DEFAULT_PARAMETER_MANIFEST[qualified_name][
                parameter_name
            ]
        except KeyError as error:
            raise RuntimeError(
                f"default-parameter fact key is absent from pinned source: {key}"
            ) from error
        if fact["category_anchor"] != _anchor_from_source_evidence(evidence):
            raise RuntimeError(f"effect-role anchor differs from pinned AST for {key}")
        _validate_source_derived_effect_role(
            qualified_name,
            parameter_name,
            fact["effect_role"],
            evidence,
        )
        contracts.setdefault(qualified_name, {})[parameter_name] = (
            _render_source_contract(
                qualified_name,
                parameter_name,
                fact,
                source_record,
                evidence,
            )
        )
    expected_keys = {
        (qualified_name, parameter_name)
        for qualified_name, parameters in SOURCE_DEFAULT_PARAMETER_MANIFEST.items()
        for parameter_name in parameters
    }
    actual_keys = {
        (qualified_name, parameter_name)
        for qualified_name, parameters in contracts.items()
        for parameter_name in parameters
    }
    if actual_keys != expected_keys:
        raise RuntimeError("source-anchored semantic fact key set changed")
    return {name: contracts[name] for name in sorted(contracts)}


DEFAULT_PARAMETER_SEMANTICS_AUTHORITY = _read_default_parameter_fact_authority()
EXPLICIT_DEFAULT_PARAMETER_CONTRACTS = _contracts_from_fact_authority(
    DEFAULT_PARAMETER_SEMANTICS_AUTHORITY,
    _PINNED_AST_DEFAULT_PARAMETER_USE_EVIDENCE,
)

DEFAULT_PARAMETER_REVIEWS = {
    qualified_name: {
        parameter_name: {
            **source_record,
            **SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE[qualified_name][parameter_name],
            **EXPLICIT_DEFAULT_PARAMETER_CONTRACTS[qualified_name][parameter_name],
        }
        for parameter_name, source_record in defaults.items()
    }
    for qualified_name, defaults in SOURCE_DEFAULT_PARAMETER_MANIFEST.items()
}

# Every explicit override or default-bearing callable is fail-closed: it must
# exist and may never remain tagged as a broad reviewed-module policy.
REQUIRED_EXPLICIT_SYMBOLS = frozenset(EXPLICIT_OVERRIDES) | frozenset(
    DEFAULT_CALLABLE_REVIEWS
)


# Exact package export aliases.  Their physical placement is copied from the
# defining record after explicit symbol corrections; the alias itself is
# marked plumbing-only and checked against the complete ``__init__`` inventory.
ATMOSPHERE_EXPORT_MODULES = {
    "atmosphere_io": (
        "ModelAtmosphere",
        "format_atmosphere_deck",
        "linear_elemental_abundances",
        "parse_atmosphere_deck",
        "read_atmosphere_deck",
        "write_atmosphere_deck",
    ),
    "config": ("AtmosphereConfig", "AtmosphereInput", "AtmosphereOutput"),
    "continuum_opacity": (
        "ContinuumLevelTables",
        "ContinuumAtmosphereState",
        "ContinuumOpacityTables",
        "KarzasLatterTables",
        "MolecularEquilibriumTables",
        "RosselandOpacityTable",
        "active_continuum_reference_frequencies",
        "assemble_continuum_line_selection_threshold",
        "build_continuum_atmosphere_state",
        "build_continuum_reference_wavelength_grid",
        "build_opacity_sampling_grid",
        "compute_aluminum_neutral_opacity_columns",
        "compute_carbon_neutral_opacity_columns",
        "compute_continuum_opacity_columns",
        "compute_continuum_scattering_columns",
        "compute_heminus_opacity_columns",
        "compute_helium_ionized_opacity_columns",
        "compute_helium_neutral_opacity_columns",
        "compute_hminus_opacity_columns",
        "compute_hydrogen_opacity_columns",
        "compute_hot_metal_opacity_columns",
        "compute_iron_neutral_opacity_columns",
        "compute_light_element_continuum_columns",
        "compute_lukewarm_metal_opacity_columns",
        "compute_magnesium_neutral_opacity_columns",
        "compute_molecular_continuum_opacity_columns",
        "compute_molecular_hydrogen_population",
        "compute_molecular_hydrogen_ion_opacity_columns",
        "compute_rosseland_continuum_opacity_columns",
        "compute_silicon_neutral_opacity_columns",
        "create_rosseland_opacity_table",
        "evaluate_rosseland_opacity",
        "ingest_rosseland_opacity_table",
        "load_continuum_level_tables",
        "load_continuum_opacity_tables",
        "load_karzas_latter_tables",
        "load_molecular_equilibrium_tables",
    ),
    "convergence": (
        "deep_layer_relative_temperature_change",
        "max_normalized_column_delta",
    ),
    "convection": (
        "ConvectionFiniteDifferenceSamples",
        "ConvectionResult",
        "compute_convection",
        "integrate_geometric_depth_below_surface_km",
    ),
    "doppler": ("update_doppler_line_strength_factors",),
    "specific_internal_energy": ("compute_atomic_specific_internal_energy",),
    "equation_of_state": (
        "iterate_electron_density",
        "populate_all_species",
        "populate_species",
        "saha_partition_depth",
        "iron_group_partition_function",
        "SpecialPartitionTables",
        "load_ionization_potential_table_cm",
        "load_iron_group_partition_grid",
        "load_packed_level_metadata",
        "load_special_partition_tables",
    ),
    "line_catalog": (
        "LineTransitionCatalog",
        "SelectedLineCatalog",
        "read_line_transition_catalog",
        "read_selected_line_catalog",
    ),
    "line_selection": (
        "compute_doppler_population_ratio_max",
        "generate_selected_lines",
        "read_diatomic_line_catalog",
        "read_standard_line_catalog",
        "read_water_line_catalog",
        "select_standard_line_words",
        "select_water_line_words",
        "write_selected_line_words",
    ),
    "hydrogen_line_profile": (
        "HydrogenLineProfileEvaluator",
        "HydrogenLineProfileTableError",
        "HydrogenLineProfileTables",
        "HydrogenLineSetup",
        "compute_hydrogen_molecule_population",
        "load_hydrogen_line_profile_tables",
        "molecular_hydrogen_equilibrium_constant",
    ),
    "hydrostatic": ("integrate_hydrostatic_pressure", "update_total_pressure"),
    "line_opacity": (
        "LineOpacityState",
        "accumulate_selected_line_opacity",
        "accumulate_transition_line_opacity",
        "allocate_line_opacity_state",
    ),
    "line_profile_math": (
        "FastExponentialTables",
        "LineOpacityTables",
        "load_line_opacity_tables",
        "VoigtProfileBasis",
        "build_fast_exponential_tables",
        "build_selection_log_lookup",
        "build_voigt_profile_basis",
        "evaluate_voigt_profile",
        "fast_exponential_lookup",
        "load_hydrogen_continuum_selector_table",
    ),
    "microturbulence": ("standard_microturbulence",),
    "molecular_data": (
        "MolecularEquilibriumCatalog",
        "find_default_molecular_equilibrium_catalog",
        "parse_molecular_equilibrium_record",
        "read_molecular_equilibrium_catalog",
    ),
    "molecular_equilibrium": (
        "MolecularEquilibriumState",
        "compute_equilibrium_constants_for_layer",
        "compute_molecular_specific_internal_energy",
        "initialize_molecular_equilibrium_state",
        "populate_molecular_species",
        "restore_molecular_equation_density",
        "save_molecular_equation_density",
        "set_molecular_specific_internal_energy_mode",
        "solve_molecular_equilibrium",
        "solve_molecular_equilibrium_layer",
    ),
    "population_layout": (
        "PopulationJob",
        "atomic_population_slot_start",
        "decode_population_code",
        "ion_stage_count_for_atomic_number",
        "population_job_schedule",
    ),
    "radiative_pressure": (
        "RadiativePressureState",
        "accumulate_radiative_pressure",
        "initialize_radiative_pressure_state",
    ),
    "radiative_transfer": (
        "RadiativeTransferTables",
        "differentiate_on_depth_grid",
        "integrate_on_depth_grid",
        "load_radiative_transfer_tables",
        "remap_to_grid",
    ),
    "run_setup": ("RunSetup", "resolve_run_setup"),
    "rosseland_mean": ("rosseland_mean_step",),
    "runtime_state": (
        "AtmosphereRuntimeState",
        "build_runtime_state",
        "update_charge_square_density",
    ),
    "synthesis_bridge": (
        "infer_synthesis_source_catalog_root",
        "save_product_structured_atmosphere",
        "save_structured_atmosphere",
        "save_structured_atmosphere_from_debug_npz",
        "save_structured_atmosphere_from_runtime_state",
        "structured_atmosphere_from_debug_npz",
        "structured_atmosphere_from_packed_state",
        "structured_atmosphere_from_runtime_state",
    ),
    "temperature_correction": (
        "TemperatureCorrectionResult",
        "TemperatureCorrectionState",
        "apply_temperature_correction",
        "exponential_integral_approximation",
        "ingest_temperature_correction_rosseland_table",
        "initialize_temperature_correction_state",
    ),
    "warm_start": (
        "ALPHA_ELEMENT_ATOMIC_NUMBERS",
        "CNO8_FAMILY",
        "DEFAULT_CNO8_WEIGHTS_PATH",
        "DEFAULT_FIVE_LABEL_WEIGHTS_PATH",
        "FIVE_LABEL_FAMILY",
        "HELIUM_NUMBER_FRACTION",
        "SOLAR_METAL_LOG_ABUNDANCES_3_TO_99",
        "AtmosphereInitializer",
        "atmosphere_prediction_to_layer_table",
        "cno8_absolute_abundance_offsets",
        "compute_hydrogen_fraction",
        "compute_metal_log_number_abundances",
        "deterministic_initializer_labels",
        "emulator_warm_start_model",
        "load_atmosphere_initializer",
        "parse_abundance_offset",
        "resolve_cno8_labels",
        "select_warm_start_family",
        "warm_start_supported",
    ),
    "runner": (
        "IterationFinalization",
        "IterationRemap",
        "OpacityState",
        "AtmospherePopulationState",
        "AtmosphereRunResult",
        "TransferAccumulation",
        "accumulate_transfer_state",
        "compute_convection_finite_difference_samples",
        "finalize_transfer_state",
        "prepare_opacity_state",
        "prepare_population_state",
        "prepare_structured_handoff_population_state",
        "remap_finalized_iteration_state",
        "run_atmosphere_model",
        "finalize_remapped_iteration",
    ),
    "cli": ("solve_structured_atmosphere",),
}

SYNTHESIS_EXPORT_MODULES = {
    "api": (
        "ForwardTimings",
        "InitializedAtmosphere",
        "LabelSpectrum",
        "Spectrum",
        "build_structured_atmosphere",
        "initialize_atmosphere_from_labels",
        "save_structured_atmosphere",
        "synthesize",
        "synthesize_from_labels",
    ),
    "pipeline": (
        "clear_window_invariant_cache",
        "window_invariant_cache_enabled",
    ),
    "atmosphere": (
        "REQUIRED_ATMOSPHERE_ARRAYS",
        "load_atmosphere_npz",
        "load_atmosphere_product_metadata",
        "validate_atmosphere_npz",
    ),
}


def _flatten_export_targets(
    package: str,
    modules: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    return {
        f"{package}.__init__.{name}": f"{package}.{module}.{name}"
        for module, names in modules.items()
        for name in names
    }


API_ALIAS_TARGETS = {
    **_flatten_export_targets(
        "payne_zero_atmosphere",
        ATMOSPHERE_EXPORT_MODULES,
    ),
    **_flatten_export_targets(
        "payne_zero_synthesis",
        SYNTHESIS_EXPORT_MODULES,
    ),
}

REVIEWED_SOURCE_MANIFEST_SHA256 = (
    "b75d6f85f73d0ae14df5e908e785f46173681313407136cfa785804f9565eb32"
)
REVIEWED_PUBLIC_SURFACE_SHA256 = (
    "669a1afe23df89b4030aaf8ee1ad582409e771eff9b20b592ae20976fc60318b"
)
EXPLICIT_SOURCE_SURFACE_SHA256 = (
    "ffe805be645f5d1ee5938b3dc7797690759e63c6c9f7572b775d01987ff6611a"
)
EXPLICIT_SEMANTIC_REGISTRY_SHA256 = (
    "a1be7847609223216f779f116c40f122f8812f93e568069e368759c39e00377f"
)
DEFAULT_BEARING_CALLABLE_REGISTRY_SHA256 = (
    "74008ed126f8e3a07f100769304e251acfbddc96ba91812cc1240726624a59ee"
)
DEFAULT_CALLABLE_REVIEW_REGISTRY_SHA256 = (
    "c74c4d7f667da497b9f6e81919b339084bfb5d5aa1a8b3c4241cf80b5d386b99"
)
DEFAULT_CALLABLE_SOURCE_SURFACE_SHA256 = (
    "9947ad3a78ebb59fbe8d0f3e194bbb172e485099e24a2b9412f1a8ee362f5343"
)
SOURCE_DEFAULT_PARAMETER_MANIFEST_SHA256 = (
    "6ee425f2d8dffe2c16ef40071d5ebe61a7c4c389580c144b789abf42da02e560"
)
SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE_SHA256 = (
    "3256470cc72b73584318e90eaaf041ef7fae3fa7b20a33011698f69f70e85bf9"
)
EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256 = (
    "176164024984435354a95b7c0cada7a70ad972135167075e2ab41cd1669287e8"
)
DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256 = (
    "0cf4fefe0da1e13c4194890e9b91ef517d6410a4e70d071640a947550cf4eb65"
)
API_ALIAS_TARGET_REGISTRY_SHA256 = (
    "9589bbc3c4579ca80996f4f52649859d20ec146eff0b58dfd4af1347f8650964"
)
REVIEWED_MODULE_POLICY_CONTENT_SHA256 = (
    "6b810638845c6daedf7ab306cc8a5a4d2de3f53e3a03a9fb457a11b92940ce92"
)
ACCEPTED_ARTIFACT_MANIFEST_SHA256 = (
    "fd7232008a8c56ab47b5548220fce997ecb5f8ff26a53b68e08a6dac911d5d05"
)
COMPLETE_SEMANTIC_PROOF_SHA256 = (
    "9acb2dbb787718f0bd7ceac3066921412c580ddcd1aaea2f6b2b1b1b9b882f36"
)
ALIAS_SOURCE_MODULE_SHA256 = {
    "payne_zero_atmosphere.__init__": (
        "dbb7734cab4f3e98b9b88d1f1b5ec27afd02fd7fd003ba7931b43d3750049d61"
    ),
    "payne_zero_synthesis.__init__": (
        "7e94560ab51be3baa3950a93e8b6f998c849a6d1d78356c5f662f6cfaba927db"
    ),
}

ACCEPTED_CHAPTER_AUTHORITIES = {
    "chapter-1": (
        {
            "path": "content/Chapter01.ipynb",
            "sha256": "093d57068ac24fc89f1fc3e5069f515f0bd69af8e237f69d3e2688b387877499",
        },
        {
            "path": "tests/test_chapter01_exact_names.py",
            "sha256": "fa7afc9085b647aba77a6e75689aa97ef25aedabb28e635eb466eb21a8850054",
        },
    ),
    "chapter-2": (
        {
            "path": "design/chapter02_acceptance.md",
            "sha256": "3c3ed37f7f848b55238a0b19e0c3964fb64ed7deac1a15629861750fa4bc5bdb",
        },
        {
            "path": "design/chapter02_exact_source_contract.md",
            "sha256": "6c99e0c078feeb35265e9428fc8f0e1a5c373018935d48c28df3d1bf13f8bdd1",
        },
    ),
    "chapter-3": (
        {
            "path": "design/chapter03_acceptance.md",
            "sha256": "785693d5c20631e1ca0609e8db8739f9d18eae05e430436c809e5121fe363950",
        },
        {
            "path": "design/chapter03_exact_source_contract.md",
            "sha256": "47467dc34cef732b02306ed9d629929f9ac53ac2f52fa3ffba44afe6dfe0c6f7",
        },
    ),
    "chapter-4": (
        {
            "path": "design/chapter04_acceptance.md",
            "sha256": "ef4a1606ef8017dee706f1046444920499b4bc4b7d2623be93c450098f670fc0",
        },
        {
            "path": "design/chapter04_exact_source_contract.md",
            "sha256": "4964b42e53680bd478ffd23d8109fff910e12e91e07e6f925da3d6e2fb50e5cb",
        },
        {
            "path": "design/chapter04_ownership_data_audit.md",
            "sha256": "ddb9ca0f6fc4942bc07a271e993ebdac44a5644bd755a0c9d9a95f22b1d30b77",
        },
    ),
    "chapter-5": (
        {
            "path": "design/chapter05_final_acceptance.md",
            "sha256": "d94b81b561ce97d52a0f1d129da8d2acb91a8f89841bf3bc235e63b66f266d08",
        },
        {
            "path": "design/chapter05_exact_source_contract.md",
            "sha256": "ec1c84519a898a454a408780bd6b36fa723fc7bade83a98e11eede1343bf2956",
        },
        {
            "path": "design/chapter05_symbol_disposition.md",
            "sha256": "7fee0b1dbb1b2007608171e6f548bb84d4e99f7fe1d6e59ad83728008676654d",
        },
    ),
}

# Chapter 4's binding ownership table defines a complete narrow surface: every
# row whose corrected primary is Chapter 4 is represented here.  It includes
# both packages and therefore cannot collapse same-leaf routines.
CHAPTER4_ACCEPTED_SYMBOLS = {
    **{
        f"payne_zero_atmosphere.molecular_data.{name}": "taught"
        for name in (
            "MolecularEquilibriumCatalog",
            "find_default_molecular_equilibrium_catalog",
            "parse_molecular_equilibrium_record",
            "read_molecular_equilibrium_catalog",
        )
    },
    **{
        f"payne_zero_atmosphere.molecular_equilibrium.{name}": "taught"
        for name in (
            "MolecularEquilibriumState",
            "compute_equilibrium_constants_for_layer",
            "compute_molecular_specific_internal_energy",
            "initialize_molecular_equilibrium_state",
            "populate_molecular_species",
            "restore_molecular_equation_density",
            "save_molecular_equation_density",
            "set_molecular_specific_internal_energy_mode",
            "solve_molecular_equilibrium",
            "solve_molecular_equilibrium_layer",
        )
    },
    (
        "payne_zero_synthesis.equation_of_state.molecular_seed_electron_density"
    ): "taught",
    (
        "payne_zero_synthesis.equation_of_state."
        "molecular_ion_formation_constants_from_seed"
    ): "taught",
    (
        "payne_zero_synthesis.equation_of_state.PopulationState.molecular_populations"
    ): "taught",
    (
        "payne_zero_synthesis.equation_of_state."
        "PopulationState.molecular_equation_densities"
    ): "taught",
    (
        "payne_zero_atmosphere.runner.AtmospherePopulationState.molecular_state"
    ): "composed",
    (
        "payne_zero_atmosphere.source_catalogs.molecular_equilibrium_catalog_path"
    ): "plumbing-only",
    "payne_zero_atmosphere.config.AtmosphereInput.molecules_path": "plumbing-only",
    "payne_zero_atmosphere.config.AtmosphereConfig.enable_molecules": "composed",
    "payne_zero_atmosphere.run_setup.RunSetup.molecules_enabled": "composed",
}

# These tuples are transcribed from explicit source-interface or ownership
# surfaces in the frozen authorities.  ModelAtmosphere field leaves formerly
# accepted through incidental occurrences are deliberately absent.
ACCEPTED_ARTIFACT_SYMBOLS = {
    "chapter-1": {
        "payne_zero_synthesis.radiative_transfer.planck_bnu": "taught",
    },
    "chapter-2": {
        "payne_zero_atmosphere.radiative_transfer.integrate_on_depth_grid": "taught",
        "payne_zero_atmosphere.radiative_transfer.parabolic_coefficients": "taught",
        "payne_zero_atmosphere.transfer_kernels.accumulate_transfer_range_parallel": (
            "taught"
        ),
        "payne_zero_synthesis.atmosphere.ATMOSPHERE_SCHEMA_VERSION": "taught",
        "payne_zero_synthesis.atmosphere.POPULATION_ION_STAGE_COUNT": "taught",
        "payne_zero_synthesis.atmosphere.POPULATION_SPECIES_COUNT": "taught",
        "payne_zero_synthesis.atmosphere.REQUIRED_ATMOSPHERE_ARRAYS": "taught",
        "payne_zero_synthesis.atmosphere.load_atmosphere_npz": "taught",
        "payne_zero_synthesis.atmosphere.validate_atmosphere_npz": "taught",
        "payne_zero_synthesis.device.resolve_runtime": "taught",
        "payne_zero_synthesis.radiative_transfer.integrate_optical_depth": "taught",
    },
    "chapter-3": {
        "payne_zero_atmosphere.equation_of_state.saha_partition_depth": "taught",
        "payne_zero_atmosphere.equation_of_state.saha_partition_depth_batch": "taught",
        "payne_zero_atmosphere.synthesis_bridge.structured_atmosphere_from_packed_state": (
            "composed"
        ),
        "payne_zero_synthesis.api.build_structured_atmosphere": "composed",
        "payne_zero_synthesis.equation_of_state.partition_functions_for_elements": (
            "taught"
        ),
        "payne_zero_synthesis.equation_of_state.solve_electron_density": "taught",
        "payne_zero_synthesis.equation_of_state.solve_population_state": "composed",
        "payne_zero_synthesis.equation_of_state.solve_population_state_at_electron_density": (
            "composed"
        ),
        "payne_zero_synthesis.ground_partition_table.FIRST_RANGE_LABELS": "taught",
        "payne_zero_synthesis.ground_partition_table.GROUND_PARTITION_TERMS": "taught",
        "payne_zero_synthesis.ground_partition_table.SECOND_RANGE_LABELS": "taught",
        "payne_zero_synthesis.ground_partition_table.ground_partition_value": "taught",
        "payne_zero_synthesis.ground_partition_table.ground_partition_values": "taught",
        "payne_zero_synthesis.pipeline.build_structured_atmosphere_from_columns": (
            "composed"
        ),
        "payne_zero_synthesis.pipeline.compute_doppler_per_ion": "taught",
        "payne_zero_synthesis.synthesis.build_structured_atmosphere_from_columns": (
            "composed"
        ),
    },
    "chapter-4": CHAPTER4_ACCEPTED_SYMBOLS,
    "chapter-5": {
        qualified_name: (
            "plumbing-only"
            if qualified_name in CHAPTER5_PLUMBING_SYMBOLS
            else "composed"
            if qualified_name in CHAPTER5_COMPOSED_SYMBOLS
            else "diagnostic-only"
            if qualified_name in CHAPTER5_DIAGNOSTIC_SYMBOLS
            else "taught"
        )
        for qualified_name in sorted(CHAPTER5_ACCEPTED_SYMBOLS)
    },
}

# Exact authority locations for Chapters 1–4.  Each marker is checked on the
# declared line in a hash-bound authority.  Chapter 5 is parsed exhaustively
# from its binding symbol table by _chapter5_authority_evidence().
ACCEPTED_ARTIFACT_EVIDENCE = {
    "chapter-1": {
        "payne_zero_synthesis.radiative_transfer.planck_bnu": {
            "path": "tests/test_chapter01_exact_names.py",
            "line": 18,
            "marker": (
                "from payne_zero_synthesis.radiative_transfer import "
                "PLANCK_PREFACTOR, planck_bnu"
            ),
        },
    },
    "chapter-2": {
        "payne_zero_atmosphere.radiative_transfer.parabolic_coefficients": {
            "path": "design/chapter02_exact_source_contract.md",
            "line": 63,
            "marker": "def parabolic_coefficients(",
        },
        "payne_zero_atmosphere.radiative_transfer.integrate_on_depth_grid": {
            "path": "design/chapter02_exact_source_contract.md",
            "line": 70,
            "marker": "def integrate_on_depth_grid(",
        },
        ("payne_zero_atmosphere.transfer_kernels.accumulate_transfer_range_parallel"): {
            "path": "design/chapter02_exact_source_contract.md",
            "line": 226,
            "marker": "def accumulate_transfer_range_parallel(",
        },
        "payne_zero_synthesis.atmosphere.ATMOSPHERE_SCHEMA_VERSION": {
            "path": "design/chapter02_exact_source_contract.md",
            "line": 733,
            "marker": "ATMOSPHERE_SCHEMA_VERSION = 4",
        },
        "payne_zero_synthesis.atmosphere.POPULATION_ION_STAGE_COUNT": {
            "path": "design/chapter02_exact_source_contract.md",
            "line": 734,
            "marker": "POPULATION_ION_STAGE_COUNT = 6",
        },
        "payne_zero_synthesis.atmosphere.POPULATION_SPECIES_COUNT": {
            "path": "design/chapter02_exact_source_contract.md",
            "line": 735,
            "marker": "POPULATION_SPECIES_COUNT = 139",
        },
        "payne_zero_synthesis.atmosphere.REQUIRED_ATMOSPHERE_ARRAYS": {
            "path": "design/chapter02_exact_source_contract.md",
            "line": 757,
            "marker": "returns `REQUIRED_ATMOSPHERE_ARRAYS`",
        },
        "payne_zero_synthesis.atmosphere.load_atmosphere_npz": {
            "path": "design/chapter02_exact_source_contract.md",
            "line": 743,
            "marker": "def load_atmosphere_npz(",
        },
        "payne_zero_synthesis.atmosphere.validate_atmosphere_npz": {
            "path": "design/chapter02_exact_source_contract.md",
            "line": 753,
            "marker": "def validate_atmosphere_npz(",
        },
        "payne_zero_synthesis.device.resolve_runtime": {
            "path": "design/chapter02_exact_source_contract.md",
            "line": 462,
            "marker": "def resolve_runtime(",
        },
        "payne_zero_synthesis.radiative_transfer.integrate_optical_depth": {
            "path": "design/chapter02_exact_source_contract.md",
            "line": 381,
            "marker": "def integrate_optical_depth(",
        },
    },
    "chapter-3": {
        "payne_zero_atmosphere.equation_of_state.saha_partition_depth": {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 103,
            "marker": "| `saha_partition_depth` |",
        },
        "payne_zero_atmosphere.equation_of_state.saha_partition_depth_batch": {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 104,
            "marker": "| `saha_partition_depth_batch` |",
        },
        (
            "payne_zero_atmosphere.synthesis_bridge."
            "structured_atmosphere_from_packed_state"
        ): {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 117,
            "marker": "| `structured_atmosphere_from_packed_state` |",
        },
        "payne_zero_synthesis.api.build_structured_atmosphere": {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 935,
            "marker": "payne_zero_synthesis.api.build_structured_atmosphere",
        },
        ("payne_zero_synthesis.equation_of_state.partition_functions_for_elements"): {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 135,
            "marker": "| `partition_functions_for_elements` |",
        },
        "payne_zero_synthesis.equation_of_state.solve_electron_density": {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 136,
            "marker": "| `solve_electron_density` |",
        },
        "payne_zero_synthesis.equation_of_state.solve_population_state": {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 137,
            "marker": "| `solve_population_state` |",
        },
        (
            "payne_zero_synthesis.equation_of_state."
            "solve_population_state_at_electron_density"
        ): {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 138,
            "marker": "| `solve_population_state_at_electron_density` |",
        },
        "payne_zero_synthesis.ground_partition_table.FIRST_RANGE_LABELS": {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 390,
            "marker": "`FIRST_RANGE_LABELS` has 114 entries",
        },
        "payne_zero_synthesis.ground_partition_table.GROUND_PARTITION_TERMS": {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 383,
            "marker": "GROUND_PARTITION_TERMS[label] = (",
        },
        "payne_zero_synthesis.ground_partition_table.SECOND_RANGE_LABELS": {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 390,
            "marker": "`SECOND_RANGE_LABELS` has 168",
        },
        "payne_zero_synthesis.ground_partition_table.ground_partition_value": {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 372,
            "marker": "def ground_partition_value(",
        },
        "payne_zero_synthesis.ground_partition_table.ground_partition_values": {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 374,
            "marker": "def ground_partition_values(",
        },
        ("payne_zero_synthesis.pipeline.build_structured_atmosphere_from_columns"): {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 937,
            "marker": "pipeline.build_structured_atmosphere_from_columns",
        },
        "payne_zero_synthesis.pipeline.compute_doppler_per_ion": {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 139,
            "marker": "| `compute_doppler_per_ion` |",
        },
        ("payne_zero_synthesis.synthesis.build_structured_atmosphere_from_columns"): {
            "path": "design/chapter03_exact_source_contract.md",
            "line": 936,
            "marker": "synthesis.build_structured_atmosphere_from_columns",
        },
    },
}

# The Chapter 4 authority uses an exact package-export name followed by
# abbreviated names on one row.  Expand that row once into qualified symbols
# and bind every export to its exact defining object.  This prevents a repeated
# leaf such as ``solve_molecular_equilibrium`` from authorizing a symbol in the
# wrong module or package.
CHAPTER4_AUTHORITY_ROWS = {
    249: {
        "line_sha256": (
            "7e2e126fb9347d05bcaf40155afeb7631ac2eb8b1f8e93d4acb72447807ba71c"
        ),
        "marker": ("| `payne_zero_atmosphere.__init__.MolecularEquilibriumCatalog`"),
        "authority_symbols": tuple(
            f"payne_zero_atmosphere.__init__.{name}"
            for name in (
                "MolecularEquilibriumCatalog",
                "MolecularEquilibriumState",
                "compute_equilibrium_constants_for_layer",
                "compute_molecular_specific_internal_energy",
                "initialize_molecular_equilibrium_state",
                "find_default_molecular_equilibrium_catalog",
                "parse_molecular_equilibrium_record",
                "populate_molecular_species",
                "read_molecular_equilibrium_catalog",
                "restore_molecular_equation_density",
                "save_molecular_equation_density",
                "set_molecular_specific_internal_energy_mode",
                "solve_molecular_equilibrium",
                "solve_molecular_equilibrium_layer",
            )
        ),
    },
    250: {
        "line_sha256": (
            "f392b5b4fd3e295c6a921ea30f4fed8ff5898b37e323f90bd9b6deffd5d13d96"
        ),
        "marker": (
            "| `payne_zero_synthesis.equation_of_state.molecular_seed_electron_density`"
        ),
        "authority_symbols": (
            "payne_zero_synthesis.equation_of_state.molecular_seed_electron_density",
            "payne_zero_synthesis.equation_of_state."
            "molecular_ion_formation_constants_from_seed",
        ),
    },
    251: {
        "line_sha256": (
            "c9529987b4d570c0a72f9d37e82e1787ee9cc985c10acbca3503dd1ea0dff0b6"
        ),
        "marker": (
            "| `payne_zero_synthesis.equation_of_state."
            "PopulationState.molecular_populations`"
        ),
        "authority_symbols": (
            "payne_zero_synthesis.equation_of_state."
            "PopulationState.molecular_populations",
            "payne_zero_synthesis.equation_of_state."
            "PopulationState.molecular_equation_densities",
        ),
    },
    252: {
        "line_sha256": (
            "e0ee5b8e84ed5993d27f525023459d22583a36d345fff86e12e399d848887be5"
        ),
        "marker": (
            "| `payne_zero_atmosphere.runner.AtmospherePopulationState.molecular_state`"
        ),
        "authority_symbols": (
            "payne_zero_atmosphere.runner.AtmospherePopulationState.molecular_state",
        ),
    },
    253: {
        "line_sha256": (
            "ee762ef8a60b3b5827f41e72d30805adf033c01770b6b148752df5d3e7751726"
        ),
        "marker": (
            "| `payne_zero_atmosphere.source_catalogs."
            "molecular_equilibrium_catalog_path`"
        ),
        "authority_symbols": (
            "payne_zero_atmosphere.source_catalogs.molecular_equilibrium_catalog_path",
        ),
    },
    254: {
        "line_sha256": (
            "0c36ed8c2b91715b774e4deb53d70bffc0ffb8e0bfd6e143ab9c6776117a44a9"
        ),
        "marker": ("| `payne_zero_atmosphere.config.AtmosphereInput.molecules_path`"),
        "authority_symbols": (
            "payne_zero_atmosphere.config.AtmosphereInput.molecules_path",
            "payne_zero_atmosphere.config.AtmosphereConfig.enable_molecules",
            "payne_zero_atmosphere.run_setup.RunSetup.molecules_enabled",
        ),
    },
}


def _canonical_sha256(value: Any) -> str:
    """Hash one JSON-compatible value with a stable canonical encoding."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _effective_module_policies(
    ledger: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compose and freeze every module-policy semantic field before use."""

    coverage_fields: dict[str, dict[str, Any]] = {}
    for package_name, package in ledger["packages"].items():
        for module in package["modules"]:
            module_name = f"{package_name}.{module['module']}"
            if module_name not in REVIEWED_MODULE_POLICIES:
                continue
            coverage_fields[module_name] = {
                field_name: module[field_name]
                for field_name in (
                    "primary_location",
                    "supporting_locations",
                    "responsibility",
                    "gate",
                    "status",
                )
            }
    if set(coverage_fields) != set(REVIEWED_MODULE_POLICIES):
        raise RuntimeError("reviewed module policy/coverage registries differ")
    effective = {
        module_name: {
            **coverage_fields[module_name],
            **REVIEWED_MODULE_POLICIES[module_name],
        }
        for module_name in sorted(REVIEWED_MODULE_POLICIES)
    }
    _validate_registry_digest(
        "complete reviewed module semantic policy",
        effective,
        REVIEWED_MODULE_POLICY_CONTENT_SHA256,
    )
    return effective


def _chapter4_authority_evidence() -> dict[str, dict[str, Any]]:
    """Expand Chapter 4 rows into exact export-to-definition bindings."""

    relative_path = "design/chapter04_ownership_data_audit.md"
    lines = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8").splitlines()
    evidence: dict[str, dict[str, Any]] = {}
    for line_number, row in CHAPTER4_AUTHORITY_ROWS.items():
        exact_text = lines[line_number - 1]
        line_sha256 = hashlib.sha256(exact_text.encode("utf-8")).hexdigest()
        if line_sha256 != row["line_sha256"]:
            raise RuntimeError(
                f"Chapter 4 authority row changed at {relative_path}:{line_number}"
            )
        if row["marker"] not in exact_text:
            raise RuntimeError(
                f"Chapter 4 authority marker is missing at "
                f"{relative_path}:{line_number}"
            )
        for authority_symbol in row["authority_symbols"]:
            qualified_name = API_ALIAS_TARGETS.get(
                authority_symbol,
                authority_symbol,
            )
            if qualified_name in evidence:
                raise RuntimeError(
                    f"duplicate Chapter 4 authority target: {qualified_name}"
                )
            evidence[qualified_name] = {
                "authority_path": relative_path,
                "line_start": line_number,
                "line_end": line_number,
                "line_sha256": line_sha256,
                "marker": row["marker"],
                "authority_symbol": authority_symbol,
                "authority_binding": (
                    "package_export_alias"
                    if authority_symbol in API_ALIAS_TARGETS
                    else "direct_qualified_symbol"
                ),
                "definition_qualified_name": qualified_name,
            }
    if set(evidence) != set(CHAPTER4_ACCEPTED_SYMBOLS):
        raise RuntimeError(
            "Chapter 4 exhaustive authority extraction changed; "
            f"missing={sorted(set(CHAPTER4_ACCEPTED_SYMBOLS) - set(evidence))}, "
            f"extra={sorted(set(evidence) - set(CHAPTER4_ACCEPTED_SYMBOLS))}"
        )
    return evidence


def _chapter5_authority_evidence() -> dict[str, dict[str, Any]]:
    """Parse all 46 exact rows from the binding Chapter 5 symbol authority."""

    relative_path = "design/chapter05_symbol_disposition.md"
    lines = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8").splitlines()
    namespace: str | None = None
    evidence: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        if line == "## Atmosphere continuum public surface":
            namespace = "payne_zero_atmosphere.continuum_opacity"
            continue
        if line == "## Synthesis continuum public surface":
            namespace = "payne_zero_synthesis.continuum"
            continue
        if line.startswith("## ") and line not in {
            "## Atmosphere continuum public surface",
            "## Synthesis continuum public surface",
        }:
            namespace = None
            continue
        if namespace is None or not line.startswith("| `"):
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) != 5:
            raise RuntimeError(
                f"malformed Chapter 5 symbol row at {relative_path}:{line_number}"
            )
        source_symbol = columns[0].removeprefix("`").removesuffix("`")
        qualified_name = f"{namespace}.{source_symbol}"
        if qualified_name in evidence:
            raise RuntimeError(
                f"duplicate Chapter 5 authority symbol: {qualified_name}"
            )
        evidence[qualified_name] = {
            "authority_path": relative_path,
            "line_start": line_number,
            "line_end": line_number,
            "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
            "marker": f"| `{source_symbol}` |",
            "authority_namespace": namespace,
            "authority_symbol": source_symbol,
            "lane": columns[1],
            "physical_owner": columns[2],
            "reader_disposition": columns[3],
            "rationale": columns[4],
        }
    if set(evidence) != set(CHAPTER5_ACCEPTED_SYMBOLS):
        raise RuntimeError(
            "Chapter 5 exhaustive authority extraction changed; "
            f"missing={sorted(set(CHAPTER5_ACCEPTED_SYMBOLS) - set(evidence))}, "
            f"extra={sorted(set(evidence) - set(CHAPTER5_ACCEPTED_SYMBOLS))}"
        )
    if len(evidence) != 46:
        raise RuntimeError(
            f"Chapter 5 authority must contain exactly 46 symbols, got {len(evidence)}"
        )
    return evidence


def _accepted_artifact_join() -> dict[str, dict[str, Any]]:
    """Verify exact, location-aware Chapter 1–5 authority evidence."""

    manifest = {
        "authorities": ACCEPTED_CHAPTER_AUTHORITIES,
        "symbols": ACCEPTED_ARTIFACT_SYMBOLS,
        "evidence": ACCEPTED_ARTIFACT_EVIDENCE,
        "chapter4_scope": {
            "path": "design/chapter04_ownership_data_audit.md",
            "rows": CHAPTER4_AUTHORITY_ROWS,
            "qualified_names": sorted(CHAPTER4_ACCEPTED_SYMBOLS),
        },
        "chapter5_scope": {
            "path": "design/chapter05_symbol_disposition.md",
            "expected_count": 46,
            "qualified_names": sorted(CHAPTER5_ACCEPTED_SYMBOLS),
        },
    }
    _validate_registry_digest(
        "accepted Chapter 1-5 artifact manifest",
        manifest,
        ACCEPTED_ARTIFACT_MANIFEST_SHA256,
    )

    authority_lines: dict[str, list[str]] = {}
    for chapter_name, authorities in ACCEPTED_CHAPTER_AUTHORITIES.items():
        for authority in authorities:
            authority_path = REPOSITORY_ROOT / authority["path"]
            authority_bytes = authority_path.read_bytes()
            actual_sha256 = hashlib.sha256(authority_bytes).hexdigest()
            if actual_sha256 != authority["sha256"]:
                raise RuntimeError(
                    f"accepted authority hash changed for {authority['path']}: "
                    f"expected {authority['sha256']}, got {actual_sha256}"
                )
            authority_lines[authority["path"]] = authority_bytes.decode(
                "utf-8"
            ).splitlines()

    evidence_by_chapter = {
        **ACCEPTED_ARTIFACT_EVIDENCE,
        "chapter-4": _chapter4_authority_evidence(),
        "chapter-5": _chapter5_authority_evidence(),
    }
    join: dict[str, dict[str, Any]] = {}
    for chapter_name in ACCEPTED_CHAPTER_AUTHORITIES:
        expected_symbols = ACCEPTED_ARTIFACT_SYMBOLS[chapter_name]
        chapter_evidence = evidence_by_chapter[chapter_name]
        if set(chapter_evidence) != set(expected_symbols):
            raise RuntimeError(
                f"accepted authority evidence scope changed for {chapter_name}"
            )
        permitted_paths = {
            authority["path"]
            for authority in ACCEPTED_CHAPTER_AUTHORITIES[chapter_name]
        }
        for qualified_name, disposition in ACCEPTED_ARTIFACT_SYMBOLS[
            chapter_name
        ].items():
            evidence = dict(chapter_evidence[qualified_name])
            authority_path = evidence.get(
                "authority_path",
                evidence.pop("path", None),
            )
            line_start = int(evidence.get("line_start", evidence.pop("line", 0)))
            line_end = int(evidence.get("line_end", line_start))
            marker = evidence["marker"]
            if authority_path not in permitted_paths:
                raise RuntimeError(
                    f"accepted evidence path is not authoritative for "
                    f"{qualified_name}: {authority_path}"
                )
            lines = authority_lines[authority_path]
            if not 1 <= line_start <= line_end <= len(lines):
                raise RuntimeError(
                    f"accepted evidence location is invalid for {qualified_name}"
                )
            exact_text = "\n".join(lines[line_start - 1 : line_end])
            if marker not in exact_text:
                raise RuntimeError(
                    f"accepted exact evidence is missing for {qualified_name}: "
                    f"{authority_path}:{line_start}-{line_end} marker={marker}"
                )
            if ".npz" in marker:
                raise RuntimeError(
                    f"archive filename is not semantic evidence for {qualified_name}"
                )
            if chapter_name == "chapter-4":
                if evidence["definition_qualified_name"] != qualified_name:
                    raise RuntimeError(
                        f"Chapter 4 evidence targets the wrong qualified symbol: "
                        f"{qualified_name}"
                    )
                authority_symbol = evidence["authority_symbol"]
                expected_target = API_ALIAS_TARGETS.get(
                    authority_symbol,
                    authority_symbol,
                )
                if expected_target != qualified_name:
                    raise RuntimeError(
                        f"Chapter 4 authority binding changed for {qualified_name}: "
                        f"{authority_symbol} resolves to {expected_target}"
                    )
                expected_binding = (
                    "package_export_alias"
                    if authority_symbol in API_ALIAS_TARGETS
                    else "direct_qualified_symbol"
                )
                if evidence["authority_binding"] != expected_binding:
                    raise RuntimeError(
                        f"Chapter 4 authority binding kind changed for {qualified_name}"
                    )
            if chapter_name == "chapter-5":
                expected_namespace = qualified_name.rsplit(".", maxsplit=1)[0]
                expected_symbol = qualified_name.rsplit(".", maxsplit=1)[-1]
                if (
                    evidence["authority_namespace"] != expected_namespace
                    or evidence["authority_symbol"] != expected_symbol
                ):
                    raise RuntimeError(
                        f"Chapter 5 authority binding changed for {qualified_name}"
                    )
            line_sha256 = hashlib.sha256(exact_text.encode("utf-8")).hexdigest()
            declared_line_sha256 = evidence.get("line_sha256")
            if declared_line_sha256 is not None and declared_line_sha256 != line_sha256:
                raise RuntimeError(
                    f"accepted exact evidence hash changed for {qualified_name}"
                )
            if qualified_name in join:
                raise RuntimeError(
                    f"accepted authority symbol is duplicated: {qualified_name}"
                )
            join[qualified_name] = {
                "primary_location": chapter_name,
                "semantic_disposition": disposition,
                "status": "integrated",
                "accepted_authority_evidence": {
                    **evidence,
                    "authority_path": authority_path,
                    "line_start": line_start,
                    "line_end": line_end,
                    "line_sha256": line_sha256,
                    "qualified_name": qualified_name,
                    "chapter": chapter_name,
                },
            }
    return join


SEMANTIC_PROOF_RECORD_FIELDS = (
    "qualified_name",
    "kind",
    "mapping_precision",
    "primary_location",
    "supporting_locations",
    "semantic_disposition",
    "semantic_review_reason",
    "semantic_review_source",
    "source_spelling",
    "responsibility",
    "gate",
    "status",
    "alias_target",
    "source_module_sha256",
    "source_public_surface_sha256",
    "source_public_object_sha256",
    "reviewed_default_contract",
    "default_branch_review",
    "default_parameter_reviews",
    "accepted_authority_evidence",
)


def _semantic_proof_bundle(ledger: dict[str, Any]) -> dict[str, Any]:
    """Return the executable, duplicate-kind-preserving 1,501-record proof."""

    records = [
        symbol
        for package in ledger["packages"].values()
        for symbol in package["symbols"]
    ]
    projected_records = [
        {
            field_name: record.get(field_name)
            for field_name in SEMANTIC_PROOF_RECORD_FIELDS
        }
        for record in sorted(
            records,
            key=lambda item: (item["qualified_name"], item["kind"]),
        )
    ]
    return {
        "record_order": ["qualified_name", "kind"],
        "record_fields": list(SEMANTIC_PROOF_RECORD_FIELDS),
        "records": projected_records,
        "registries": {
            "reviewed_module_policies": _effective_module_policies(ledger),
            "reviewed_module_snapshots": REVIEWED_MODULE_SNAPSHOTS,
            "reviewed_source_sha256": REVIEWED_SOURCE_SHA256,
            "explicit_overrides": EXPLICIT_OVERRIDES,
            "required_explicit_symbols": sorted(REQUIRED_EXPLICIT_SYMBOLS),
            "reviewed_default_contracts": REVIEWED_DEFAULT_CONTRACTS,
            "default_bearing_public_callables": sorted(
                DEFAULT_BEARING_PUBLIC_CALLABLES
            ),
            "explicit_default_callables": sorted(EXPLICIT_DEFAULT_CALLABLES),
            "default_callable_reviews": DEFAULT_CALLABLE_REVIEWS,
            "source_default_parameter_manifest": SOURCE_DEFAULT_PARAMETER_MANIFEST,
            "source_default_parameter_use_evidence_sha256": (
                SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE_SHA256
            ),
            "default_parameter_semantics_authority": {
                "path": "audit/default_parameter_semantics.json",
                "sha256": DEFAULT_PARAMETER_SEMANTICS_AUTHORITY_SHA256,
                "schema_version": DEFAULT_PARAMETER_SEMANTICS_AUTHORITY[
                    "schema_version"
                ],
                "record_count": DEFAULT_PARAMETER_SEMANTICS_AUTHORITY["record_count"],
            },
            "default_parameter_reviews": DEFAULT_PARAMETER_REVIEWS,
            "exact_default_syntax": EXACT_DEFAULT_SYNTAX,
            "api_alias_targets": API_ALIAS_TARGETS,
            "accepted_chapter_authorities": ACCEPTED_CHAPTER_AUTHORITIES,
            "accepted_artifact_symbols": ACCEPTED_ARTIFACT_SYMBOLS,
            "accepted_artifact_evidence": ACCEPTED_ARTIFACT_EVIDENCE,
        },
    }


def validate_complete_semantic_proof(ledger: dict[str, Any]) -> None:
    """Validate accepted-authority tuples and the complete semantic proof hash."""

    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for package in ledger["packages"].values():
        for record in package["symbols"]:
            by_name[record["qualified_name"]].append(record)
    accepted_join = _accepted_artifact_join()
    missing = sorted(set(accepted_join) - set(by_name))
    if missing:
        raise RuntimeError(f"accepted artifact symbols are missing: {missing}")
    for qualified_name, expected in accepted_join.items():
        for record in by_name[qualified_name]:
            actual = {
                field_name: record.get(field_name)
                for field_name in (
                    "primary_location",
                    "semantic_disposition",
                    "status",
                    "accepted_authority_evidence",
                )
            }
            if actual != expected:
                raise RuntimeError(
                    f"accepted artifact ledger tuple changed for {qualified_name}: "
                    f"expected {expected}, got {actual}"
                )
    _validate_registry_digest(
        "complete 1,501-record semantic proof",
        _semantic_proof_bundle(ledger),
        COMPLETE_SEMANTIC_PROOF_SHA256,
    )


def _inventory_from_builder_arguments() -> dict[str, Any]:
    """Load the exact inventory selected by the canonical builder, if present."""

    inventory_argument: str | None = None
    for index, argument in enumerate(sys.argv):
        if argument == "--inventory" and index + 1 < len(sys.argv):
            inventory_argument = sys.argv[index + 1]
            break
        if argument.startswith("--inventory="):
            inventory_argument = argument.split("=", maxsplit=1)[1]
            break
    inventory_path = Path(inventory_argument) if inventory_argument else INVENTORY_PATH
    return json.loads(inventory_path.read_text(encoding="utf-8"))


def _inventory_modules(
    inventory: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index raw inventory modules by fully qualified module name."""

    return {
        f"{package_name}.{module['module']}": module
        for package_name, package in inventory["packages"].items()
        for module in package["modules"]
    }


def _module_public_surface(module: dict[str, Any]) -> dict[str, Any]:
    """Return the exact raw descriptor surface protected by semantic review."""

    return {
        "static_all_exports": module["static_all_exports"],
        "public_module_data": module["public_module_data"],
        "public_functions": module["public_functions"],
        "public_classes": module["public_classes"],
    }


def _inventory_objects(
    inventory: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index every locally defined public object and its exact descriptor."""

    objects: dict[str, dict[str, Any]] = {}
    for package_name, package in inventory["packages"].items():
        for module in package["modules"]:
            prefix = f"{package_name}.{module['module']}"
            for datum in module["public_module_data"]:
                objects[f"{prefix}.{datum['name']}"] = {
                    "kind": datum["kind"],
                    "descriptor": datum,
                }
            for function in module["public_functions"]:
                objects[f"{prefix}.{function['name']}"] = {
                    "kind": "public_function",
                    "descriptor": function,
                }
            for class_item in module["public_classes"]:
                class_name = f"{prefix}.{class_item['name']}"
                objects[class_name] = {
                    "kind": "public_class",
                    "descriptor": class_item,
                }
                constructor = class_item["constructor"]
                if constructor is not None:
                    objects[f"{class_name}.__init__"] = {
                        "kind": "constructor",
                        "descriptor": constructor,
                    }
                for field in class_item["fields"]:
                    objects[f"{class_name}.{field['name']}"] = {
                        "kind": "annotated_field",
                        "descriptor": field,
                    }
                for method in class_item["public_methods"]:
                    objects[f"{class_name}.{method['name']}"] = {
                        "kind": "public_method",
                        "descriptor": method,
                    }
    return objects


def _validate_registry_digest(
    name: str,
    value: Any,
    expected_digest: str,
) -> None:
    actual_digest = _canonical_sha256(value)
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"{name} changed: expected {expected_digest}, got {actual_digest}"
        )


def _fresh_process_default_parameter_snapshot(
    _builder_path: str = str(Path(__file__).resolve()),
) -> dict[str, Any]:
    """Load semantic ground truth from a clean interpreter and fresh source parse.

    The subprocess imports this file from disk, rereads and hashes every pinned
    source module, reparses all signatures/control flow, rereads the literal
    authority, reconstructs all downstream operations, rerenders all
    contracts, and emits the complete result.  It therefore does not inherit
    any caller-rebound module registry, digest, path, parser function, public
    evidence object, or private AST snapshot.
    """

    subprocess_module = __import__("subprocess")
    sys_module = __import__("sys")
    completed = subprocess_module.run(
        [
            sys_module.executable,
            _builder_path,
            "--emit-fresh-default-parameter-snapshot",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fresh default-parameter source subprocess failed: "
            f"{completed.stderr.strip()}"
        )
    try:
        snapshot = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "fresh default-parameter source subprocess returned invalid JSON"
        ) from error
    if set(snapshot) != {
        "authority",
        "contracts",
        "evidence",
        "manifest",
        "reviews",
        "source_root",
    }:
        raise RuntimeError("fresh default-parameter source snapshot shape changed")
    if snapshot["source_root"] != str(
        (Path(_builder_path).resolve().parent.parent.parent / "payne-zero").resolve()
    ):
        raise RuntimeError("fresh default-parameter source root changed")
    return snapshot


def _validate_default_parameter_reviews(
    objects: dict[str, dict[str, Any]],
    _fresh_snapshot_loader: Any = _fresh_process_default_parameter_snapshot,
) -> None:
    """Require one source-derived semantic contract for every exact default."""

    fresh_snapshot = _fresh_snapshot_loader()
    fresh_authority = fresh_snapshot["authority"]
    fresh_contracts = fresh_snapshot["contracts"]
    fresh_evidence = fresh_snapshot["evidence"]
    fresh_manifest = fresh_snapshot["manifest"]
    fresh_reviews = fresh_snapshot["reviews"]

    _validate_registry_digest(
        "source-default parameter manifest",
        SOURCE_DEFAULT_PARAMETER_MANIFEST,
        SOURCE_DEFAULT_PARAMETER_MANIFEST_SHA256,
    )
    _validate_registry_digest(
        "source-default parameter use evidence",
        SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE,
        SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE_SHA256,
    )
    _validate_registry_digest(
        "explicit default-parameter semantic contracts",
        EXPLICIT_DEFAULT_PARAMETER_CONTRACTS,
        EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256,
    )
    _validate_registry_digest(
        "default-parameter semantic review registry",
        DEFAULT_PARAMETER_REVIEWS,
        DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256,
    )
    # Semantic correctness does not come from any mutable public mapping or
    # digest. The expected structures came from the clean subprocess above,
    # which freshly read, hashed, and parsed the pinned source. No evidence
    # object or semantic helper supplied by this mutable caller is an oracle.
    if DEFAULT_PARAMETER_SEMANTICS_AUTHORITY != fresh_authority:
        raise RuntimeError(
            "loaded semantic authority differs from fresh source-anchored facts"
        )
    if SOURCE_DEFAULT_PARAMETER_MANIFEST != fresh_manifest:
        raise RuntimeError("source-default manifest differs from fresh pinned source")
    if SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE != fresh_evidence:
        raise RuntimeError("source-default use evidence differs from the pinned AST")
    if EXPLICIT_DEFAULT_PARAMETER_CONTRACTS != fresh_contracts:
        raise RuntimeError("semantic contracts differ from fresh source-fact rendering")
    if DEFAULT_PARAMETER_REVIEWS != fresh_reviews:
        raise RuntimeError(
            "default-parameter reviews differ from fresh source-fact rendering"
        )
    if set(SOURCE_DEFAULT_PARAMETER_MANIFEST) != set(DEFAULT_BEARING_PUBLIC_CALLABLES):
        raise RuntimeError("source-default manifest callable scope changed")
    if set(DEFAULT_PARAMETER_REVIEWS) != set(SOURCE_DEFAULT_PARAMETER_MANIFEST):
        raise RuntimeError("default-parameter review callable scope changed")
    if set(EXPLICIT_DEFAULT_PARAMETER_CONTRACTS) != set(
        SOURCE_DEFAULT_PARAMETER_MANIFEST
    ):
        raise RuntimeError("explicit default-contract callable scope changed")
    if (
        sum(len(defaults) for defaults in SOURCE_DEFAULT_PARAMETER_MANIFEST.values())
        != 456
    ):
        raise RuntimeError("source-default manifest parameter count changed")
    if sum(len(reviews) for reviews in DEFAULT_PARAMETER_REVIEWS.values()) != 456:
        raise RuntimeError("default-parameter review count changed")
    if (
        sum(
            len(contracts)
            for contracts in EXPLICIT_DEFAULT_PARAMETER_CONTRACTS.values()
        )
        != 456
    ):
        raise RuntimeError("explicit default-contract count changed")

    # Use only the fresh-process evidence below. In particular, the mutable
    # import-time private AST snapshot is never consulted by this final gate.
    recomputed_evidence = fresh_evidence

    source_target_cache: dict[
        tuple[str, str],
        tuple[list[str], list[str]],
    ] = {}

    def target_signature(
        source_path: str,
        callable_name: str,
    ) -> tuple[list[str], list[str]]:
        cache_key = (source_path, callable_name)
        cached = source_target_cache.get(cache_key)
        if cached is not None:
            return cached
        absolute_path = PINNED_PAYNE_ZERO_SOURCE_ROOT / source_path
        if not absolute_path.is_file():
            raise RuntimeError(f"forwarded target source does not exist: {source_path}")
        tree = ast.parse(absolute_path.read_text(encoding="utf-8"))
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == callable_name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "forwarded target callable is not unique in source: "
                f"{source_path}:{callable_name}"
            )
        node = matches[0]
        positional = [
            argument.arg for argument in (*node.args.posonlyargs, *node.args.args)
        ]
        keyword_only = [argument.arg for argument in node.args.kwonlyargs]
        source_target_cache[cache_key] = (positional, keyword_only)
        return positional, keyword_only

    source_fields = {
        "source_default",
        "default_ast",
        "source_path",
        "source_sha256",
        "definition_line",
        "definition_end_line",
        "parameter_flow_names",
        "parameter_flow_attributes",
        "flow_bindings",
        "parameter_uses",
        "branch_predicates",
        "attribute_branch_predicates",
        "call_forwards",
    }
    semantic_fields = set(DEFAULT_PARAMETER_SEMANTIC_FIELDS)
    expected_review_fields = source_fields | semantic_fields
    semantic_body_owners: dict[str, tuple[str, str]] = {}

    used_categories: set[str] = set()
    for qualified_name, source_defaults in SOURCE_DEFAULT_PARAMETER_MANIFEST.items():
        reviews = DEFAULT_PARAMETER_REVIEWS[qualified_name]
        if set(reviews) != set(source_defaults):
            raise RuntimeError(
                f"default-parameter review keys changed for {qualified_name}"
            )
        descriptor = objects[qualified_name]["descriptor"]
        source_parameters = set(descriptor["parameters"]) | set(
            descriptor["keyword_only_parameters"]
        )
        for parameter_name, source_record in source_defaults.items():
            if parameter_name not in source_parameters:
                raise RuntimeError(
                    f"source-default parameter is absent from descriptor: "
                    f"{qualified_name}.{parameter_name}"
                )
            if set(source_record) != {"source_default", "default_ast"}:
                raise RuntimeError(
                    f"source-default record shape changed for "
                    f"{qualified_name}.{parameter_name}"
                )
            actual_ast = ast.dump(
                ast.parse(source_record["source_default"], mode="eval").body,
                include_attributes=False,
            )
            if actual_ast != source_record["default_ast"]:
                raise RuntimeError(
                    f"source-default AST changed for {qualified_name}.{parameter_name}"
                )

            review = reviews[parameter_name]
            if set(review) != expected_review_fields:
                raise RuntimeError(
                    f"default-parameter review shape changed for "
                    f"{qualified_name}.{parameter_name}"
                )
            if {
                "source_default": review["source_default"],
                "default_ast": review["default_ast"],
            } != source_record:
                raise RuntimeError(
                    f"default-parameter source binding changed for "
                    f"{qualified_name}.{parameter_name}"
                )
            source_evidence = recomputed_evidence[qualified_name][parameter_name]
            if {
                field_name: review[field_name]
                for field_name in source_fields
                if field_name not in {"source_default", "default_ast"}
            } != source_evidence:
                raise RuntimeError(
                    "default-parameter source-use evidence changed for "
                    f"{qualified_name}.{parameter_name}"
                )

            semantic_contract = {
                field_name: review[field_name]
                for field_name in DEFAULT_PARAMETER_SEMANTIC_FIELDS
            }
            if semantic_contract != fresh_contracts[qualified_name][parameter_name]:
                raise RuntimeError(
                    "review differs from source-fact-rendered contract: "
                    f"{qualified_name}.{parameter_name}"
                )
            operation_signature = review["operation_signature"]
            if set(operation_signature) != {
                "schema_version",
                "call_count",
                "calls",
                "binding_count",
                "bindings",
                "assignment_count",
                "assignments",
                "branch_count",
                "branches",
                "terminal_contract",
                "downstream_target_count",
                "downstream_targets",
                "concrete_quantity",
                "location_free_family_sha256",
            }:
                raise RuntimeError(
                    "strict operation-signature shape changed for "
                    f"{qualified_name}.{parameter_name}"
                )
            if operation_signature["schema_version"] != 1:
                raise RuntimeError("strict operation-signature version changed")
            for count_name, list_name in (
                ("call_count", "calls"),
                ("binding_count", "bindings"),
                ("assignment_count", "assignments"),
                ("branch_count", "branches"),
                ("downstream_target_count", "downstream_targets"),
            ):
                if operation_signature[count_name] != len(
                    operation_signature[list_name]
                ):
                    raise RuntimeError(
                        f"strict operation {count_name} changed for "
                        f"{qualified_name}.{parameter_name}"
                    )
            terminal_contract = operation_signature["terminal_contract"]
            if set(terminal_contract) != {
                "return_annotation",
                "return_node_count",
                "return_contracts",
                "return_contract_count",
                "effect_kind",
                "mutated_targets",
                "mutated_target_count",
                "assignment_count",
                "assignments",
            }:
                raise RuntimeError(
                    "terminal operation-contract shape changed for "
                    f"{qualified_name}.{parameter_name}"
                )
            if terminal_contract["return_contract_count"] != len(
                terminal_contract["return_contracts"]
            ) or terminal_contract["mutated_target_count"] != len(
                terminal_contract["mutated_targets"]
            ):
                raise RuntimeError(
                    "terminal operation cardinality changed for "
                    f"{qualified_name}.{parameter_name}"
                )
            if terminal_contract["effect_kind"] not in {
                "return_value",
                "return_and_in_place_mutation",
                "in_place_mutation",
                "side_effect_none",
            }:
                raise RuntimeError(
                    "terminal effect kind changed for "
                    f"{qualified_name}.{parameter_name}"
                )
            family_payload = _location_free_operation_family(operation_signature)
            family_sha256 = hashlib.sha256(
                json.dumps(
                    family_payload,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if operation_signature["location_free_family_sha256"] != family_sha256:
                raise RuntimeError(
                    "location-free operation-family digest changed for "
                    f"{qualified_name}.{parameter_name}"
                )
            category = review["effect_category"]
            if category not in DEFAULT_EFFECT_CATEGORIES:
                raise RuntimeError(
                    f"invalid default-parameter effect category for "
                    f"{qualified_name}.{parameter_name}: {category}"
                )
            used_categories.add(category)

            semantic_digest = _canonical_sha256(semantic_contract)

            for field_name in (
                "branch_behavior",
                "default_route",
                "alternate_route",
                "validation_and_coupling",
                "consumer",
                "observable_effect",
            ):
                value = review[field_name]
                if not isinstance(value, str) or not value.strip():
                    raise RuntimeError(
                        f"empty {field_name} for {qualified_name}.{parameter_name}"
                    )
            if review["default_route"] == review["alternate_route"]:
                raise RuntimeError(
                    "default and alternate routes are identical for "
                    f"{qualified_name}.{parameter_name}"
                )
            effect_lower = review["observable_effect"].lower()
            if "results or mutations" in effect_lower or " supplies " in effect_lower:
                raise RuntimeError(
                    "terminal effect retains rejected generic wording for "
                    f"{qualified_name}.{parameter_name}"
                )

            branch_claim = review["branch_behavior"].lower()
            if review["branch_predicates"] and (
                "no branch" in branch_claim or "no-branch" in branch_claim
            ):
                raise RuntimeError(
                    "no-branch claim conflicts with source predicates for "
                    f"{qualified_name}.{parameter_name}"
                )

            shared_key = review["shared_semantics_with"]
            prior_owner = semantic_body_owners.get(semantic_digest)
            if prior_owner is not None:
                expected_shared = f"{prior_owner[0]}\0{prior_owner[1]}"
                if shared_key != expected_shared:
                    raise RuntimeError(
                        "duplicate semantic body lacks exact forwarding alias: "
                        f"{qualified_name}.{parameter_name}"
                    )
            else:
                semantic_body_owners[semantic_digest] = (
                    qualified_name,
                    parameter_name,
                )
                if shared_key is not None:
                    raise RuntimeError(
                        "shared_semantics_with names no identical prior contract: "
                        f"{qualified_name}.{parameter_name}"
                    )

            if category == "continuous-value":
                semantic_text = " ".join(
                    review[field_name]
                    for field_name in (
                        "default_route",
                        "alternate_route",
                        "validation_and_coupling",
                        "consumer",
                        "observable_effect",
                    )
                )
                for predicate in review["branch_predicates"]:
                    line_markers = (
                        f"L{predicate['line']}",
                        f"line-{predicate['line']}",
                        predicate["snippet"],
                    )
                    if not any(marker in semantic_text for marker in line_markers):
                        raise RuntimeError(
                            "continuous-value contract omits direct/coupled "
                            f"predicate L{predicate['line']} for "
                            f"{qualified_name}.{parameter_name}"
                        )

            forwarded_targets = review["forwarded_to"]
            if not isinstance(forwarded_targets, list):
                raise RuntimeError(
                    f"forwarded_to is not a list for {qualified_name}.{parameter_name}"
                )
            for target in forwarded_targets:
                if set(target) != {
                    "source_path",
                    "callable",
                    "parameter",
                    "operation_facts",
                }:
                    raise RuntimeError(
                        "forwarded target shape changed for "
                        f"{qualified_name}.{parameter_name}"
                    )
                base_target = {
                    field_name: target[field_name]
                    for field_name in ("source_path", "callable", "parameter")
                }
                if target["operation_facts"] != (
                    _forward_target_parameter_operations(base_target)
                ):
                    raise RuntimeError(
                        "forwarded target operations differ from pinned source for "
                        f"{qualified_name}.{parameter_name}"
                    )
                positional, keyword_only = target_signature(
                    target["source_path"],
                    target["callable"],
                )
                if target["parameter"] not in set(positional) | set(keyword_only):
                    raise RuntimeError(
                        f"forwarded target parameter is absent from source: {target}"
                    )
                forwarding_call_found = False
                for call in review["call_forwards"]:
                    if call["callee"].rsplit(".", maxsplit=1)[-1] != target["callable"]:
                        continue
                    argument = call["argument"]
                    if argument == target["parameter"]:
                        forwarding_call_found = True
                        break
                    if argument.startswith("position:"):
                        position = int(argument.split(":", maxsplit=1)[1])
                        if (
                            position < len(positional)
                            and positional[position] == target["parameter"]
                        ):
                            forwarding_call_found = True
                            break
                if not forwarding_call_found:
                    raise RuntimeError(
                        "claimed forwarded target lacks pinned call evidence: "
                        f"{qualified_name}.{parameter_name} -> {target}"
                    )

            for use in review["parameter_uses"]:
                if set(use) != {
                    "line",
                    "column",
                    "snippet",
                    "snippet_sha256",
                    "control_guards",
                }:
                    raise RuntimeError(
                        "parameter-use evidence shape changed for "
                        f"{qualified_name}.{parameter_name}"
                    )
                actual = hashlib.sha256(use["snippet"].encode("utf-8")).hexdigest()
                if actual != use["snippet_sha256"]:
                    raise RuntimeError(
                        "parameter-use snippet hash changed for "
                        f"{qualified_name}.{parameter_name}"
                    )
    if used_categories != set(DEFAULT_EFFECT_CATEGORIES):
        raise RuntimeError(
            "default-parameter effect category coverage changed: "
            f"expected {sorted(DEFAULT_EFFECT_CATEGORIES)}, "
            f"got {sorted(used_categories)}"
        )
    if len(semantic_body_owners) != 456:
        raise RuntimeError(
            "default-parameter semantic contracts are not 456 unique bodies"
        )


def _validate_inventory(
    inventory: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Fail before semantic assignment if source identity or signatures drift."""

    if inventory.get("payne_zero_commit") != PINNED_PAYNE_ZERO_COMMIT:
        raise RuntimeError(
            "Payne Zero commit changed: "
            f"expected {PINNED_PAYNE_ZERO_COMMIT}, "
            f"got {inventory.get('payne_zero_commit')}"
        )

    modules = _inventory_modules(inventory)
    if set(REVIEWED_SOURCE_SHA256) != set(REVIEWED_MODULE_SNAPSHOTS):
        raise RuntimeError("reviewed source/snapshot module registries differ")
    for module_name, expected_sha256 in REVIEWED_SOURCE_SHA256.items():
        module = modules.get(module_name)
        if module is None:
            raise RuntimeError(f"reviewed source module is missing: {module_name}")
        if module["sha256"] != expected_sha256:
            raise RuntimeError(
                f"reviewed module source SHA-256 changed for {module_name}: "
                f"expected {expected_sha256}, got {module['sha256']}"
            )

    source_manifest = {
        module_name: modules[module_name]["sha256"]
        for module_name in sorted(REVIEWED_SOURCE_SHA256)
    }
    _validate_registry_digest(
        "reviewed source manifest",
        source_manifest,
        REVIEWED_SOURCE_MANIFEST_SHA256,
    )
    reviewed_surfaces = {
        module_name: _module_public_surface(modules[module_name])
        for module_name in sorted(REVIEWED_SOURCE_SHA256)
    }
    _validate_registry_digest(
        "reviewed public signature/field surface",
        reviewed_surfaces,
        REVIEWED_PUBLIC_SURFACE_SHA256,
    )

    objects = _inventory_objects(inventory)
    missing_explicit_objects = sorted(set(EXPLICIT_OVERRIDES) - set(objects))
    if missing_explicit_objects:
        raise RuntimeError(
            f"explicit source descriptors are missing: {missing_explicit_objects}"
        )
    explicit_source_surface = {
        qualified_name: objects[qualified_name]
        for qualified_name in sorted(EXPLICIT_OVERRIDES)
    }
    _validate_registry_digest(
        "explicit source signature/field registry",
        explicit_source_surface,
        EXPLICIT_SOURCE_SURFACE_SHA256,
    )
    missing_default_callables = sorted(
        set(DEFAULT_BEARING_PUBLIC_CALLABLES) - set(objects)
    )
    if missing_default_callables:
        raise RuntimeError(
            f"default-bearing source descriptors are missing: "
            f"{missing_default_callables}"
        )
    _validate_registry_digest(
        "default-bearing callable registry",
        sorted(DEFAULT_BEARING_PUBLIC_CALLABLES),
        DEFAULT_BEARING_CALLABLE_REGISTRY_SHA256,
    )
    default_callable_source_surface = {
        qualified_name: objects[qualified_name]
        for qualified_name in sorted(DEFAULT_BEARING_PUBLIC_CALLABLES)
    }
    _validate_registry_digest(
        "default-bearing callable source surface",
        default_callable_source_surface,
        DEFAULT_CALLABLE_SOURCE_SURFACE_SHA256,
    )
    _validate_registry_digest(
        "default-callable review registry",
        DEFAULT_CALLABLE_REVIEWS,
        DEFAULT_CALLABLE_REVIEW_REGISTRY_SHA256,
    )
    if set(DEFAULT_CALLABLE_REVIEWS) & set(EXPLICIT_OVERRIDES):
        raise RuntimeError(
            "default-callable reviews must be disjoint from full explicit overrides"
        )
    if not set(REVIEWED_DEFAULT_CONTRACTS) <= set(EXPLICIT_OVERRIDES):
        raise RuntimeError(
            "exact default contracts must belong to full explicit overrides"
        )
    if set(DEFAULT_BEARING_PUBLIC_CALLABLES) != (
        set(REVIEWED_DEFAULT_CONTRACTS)
        | set(DEFAULT_CALLABLE_REVIEWS)
        | set(EXPLICIT_DEFAULT_CALLABLES)
    ):
        raise RuntimeError("default-bearing callable scope changed")
    _validate_default_parameter_reviews(objects)

    _validate_registry_digest(
        "reviewed default contract registry",
        REVIEWED_DEFAULT_CONTRACTS,
        REVIEWED_DEFAULT_CONTRACTS_SHA256,
    )
    _validate_registry_digest(
        "exact default source/AST syntax registry",
        EXACT_DEFAULT_SYNTAX,
        EXACT_DEFAULT_SYNTAX_SHA256,
    )
    for qualified_name, contract in REVIEWED_DEFAULT_CONTRACTS.items():
        descriptor = objects.get(qualified_name, {}).get("descriptor")
        if descriptor is None:
            raise RuntimeError(
                f"default contract source descriptor is missing: {qualified_name}"
            )
        for field_name in ("parameters", "keyword_only_parameters"):
            if descriptor.get(field_name) != contract[field_name]:
                raise RuntimeError(
                    f"default contract signature changed for {qualified_name} "
                    f"{field_name}: expected {contract[field_name]}, "
                    f"got {descriptor.get(field_name)}"
                )
    for qualified_name, syntax_by_parameter in EXACT_DEFAULT_SYNTAX.items():
        defaults = REVIEWED_DEFAULT_CONTRACTS[qualified_name]["defaults"]
        for parameter_name, syntax in syntax_by_parameter.items():
            source_segment = syntax["source_segment"]
            if defaults.get(parameter_name) != source_segment:
                raise RuntimeError(
                    f"exact default source segment changed for {qualified_name} "
                    f"{parameter_name}: expected {source_segment}, "
                    f"got {defaults.get(parameter_name)}"
                )
            actual_ast_dump = ast.dump(
                ast.parse(source_segment, mode="eval").body,
                include_attributes=False,
            )
            if actual_ast_dump != syntax["ast_dump"]:
                raise RuntimeError(
                    f"exact default AST changed for {qualified_name} "
                    f"{parameter_name}: expected {syntax['ast_dump']}, "
                    f"got {actual_ast_dump}"
                )

    _validate_registry_digest(
        "explicit semantic registry",
        EXPLICIT_OVERRIDES,
        EXPLICIT_SEMANTIC_REGISTRY_SHA256,
    )
    _validate_registry_digest(
        "API alias target registry",
        API_ALIAS_TARGETS,
        API_ALIAS_TARGET_REGISTRY_SHA256,
    )
    _accepted_artifact_join()

    for module_name, expected_sha256 in ALIAS_SOURCE_MODULE_SHA256.items():
        module = modules.get(module_name)
        if module is None or module["sha256"] != expected_sha256:
            raise RuntimeError(
                f"package alias source changed for {module_name}: "
                f"expected {expected_sha256}, "
                f"got {None if module is None else module['sha256']}"
            )
    for alias_name, target_name in API_ALIAS_TARGETS.items():
        alias_module = _module_name(alias_name)
        exported_name = _source_spelling(alias_name)
        if exported_name not in modules[alias_module]["static_all_exports"]:
            raise RuntimeError(f"API alias is absent from source exports: {alias_name}")
        if target_name not in objects:
            raise RuntimeError(
                f"API alias ultimate source descriptor is missing: {target_name}"
            )

    _validate_registry_digest(
        "canonical raw inventory content",
        inventory,
        RAW_INVENTORY_CONTENT_SHA256,
    )
    return modules, objects


def _module_name(qualified_name: str) -> str:
    parts = qualified_name.split(".")
    return ".".join(parts[:2])


def _source_spelling(qualified_name: str) -> str:
    return qualified_name.rsplit(".", maxsplit=1)[-1]


def _snapshot(records: list[dict[str, Any]]) -> tuple[int, str]:
    payload = "".join(
        f"{record['qualified_name']}\0{record['kind']}\n"
        for record in sorted(
            records,
            key=lambda item: (item["qualified_name"], item["kind"]),
        )
    ).encode("utf-8")
    return len(records), hashlib.sha256(payload).hexdigest()


def _apply_fields(
    record: dict[str, Any],
    fields: dict[str, Any],
    *,
    review_source: str,
) -> None:
    record.update(fields)
    record["mapping_precision"] = PRECISION
    record["semantic_review_source"] = review_source
    record["source_spelling"] = _source_spelling(record["qualified_name"])


def apply_overrides(
    ledger: dict[str, Any],
    *,
    inventory: dict[str, Any] | None = None,
) -> int:
    """Apply the fail-closed semantic registry and return reviewed record count."""

    source_inventory = (
        inventory if inventory is not None else _inventory_from_builder_arguments()
    )
    inventory_modules, inventory_objects = _validate_inventory(source_inventory)
    records = [
        symbol
        for package in ledger["packages"].values()
        for symbol in package["symbols"]
    ]
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_name[record["qualified_name"]].append(record)
        by_module[_module_name(record["qualified_name"])].append(record)

    if set(REVIEWED_MODULE_SNAPSHOTS) != set(REVIEWED_MODULE_POLICIES):
        raise RuntimeError("reviewed module policy/snapshot registries differ")
    effective_module_policies = _effective_module_policies(ledger)
    for module_name, expected in REVIEWED_MODULE_SNAPSHOTS.items():
        actual = _snapshot(by_module.get(module_name, []))
        if actual != expected:
            raise RuntimeError(
                f"reviewed module snapshot changed for {module_name}: "
                f"expected {expected}, got {actual}"
            )
        fields = effective_module_policies[module_name]
        for record in by_module[module_name]:
            _apply_fields(
                record,
                fields,
                review_source="reviewed_module_registry",
            )

    missing = sorted(set(EXPLICIT_OVERRIDES) - set(by_name))
    if missing:
        raise KeyError(f"coverage ledger is missing explicit overrides: {missing}")
    for qualified_name, fields in EXPLICIT_OVERRIDES.items():
        for record in by_name[qualified_name]:
            _apply_fields(
                record,
                fields,
                review_source="explicit_symbol_registry",
            )

    # A default-bearing callable that does not need a different owner still
    # receives its own exact semantic review.  This removes the last class of
    # branch-capable callables that could otherwise hide under a module phrase.
    missing_default_reviews = sorted(set(DEFAULT_CALLABLE_REVIEWS) - set(by_name))
    if missing_default_reviews:
        raise KeyError(
            "coverage ledger is missing default-callable reviews: "
            f"{missing_default_reviews}"
        )
    for qualified_name, review in DEFAULT_CALLABLE_REVIEWS.items():
        for record in by_name[qualified_name]:
            _apply_fields(
                record,
                {
                    "semantic_review_reason": review,
                    "default_branch_review": review,
                },
                review_source="explicit_default_callable_registry",
            )

    # Exact overrides keep their stronger source tag, but every default-bearing
    # callable carries a proof-visible branch review.  For these records the
    # explicit responsibility is the reviewed branch statement.
    explicit_default_names = set(REVIEWED_DEFAULT_CONTRACTS) | set(
        EXPLICIT_DEFAULT_CALLABLES
    )
    for qualified_name in explicit_default_names & set(by_name):
        if qualified_name in EXPLICIT_OVERRIDES:
            for record in by_name[qualified_name]:
                record["default_branch_review"] = EXPLICIT_OVERRIDES[qualified_name][
                    "responsibility"
                ]

    # The acceptance gate is the complete per-parameter registry below.  The
    # legacy callable-level text remains proof-visible for revision history but
    # is not evidence that any individual default was reviewed.
    missing_default_parameter_records = sorted(
        set(DEFAULT_PARAMETER_REVIEWS) - set(by_name)
    )
    if missing_default_parameter_records:
        raise KeyError(
            "coverage ledger is missing default-parameter reviews: "
            f"{missing_default_parameter_records}"
        )
    for qualified_name, reviews in DEFAULT_PARAMETER_REVIEWS.items():
        for record in by_name[qualified_name]:
            record["default_parameter_reviews"] = {
                parameter_name: dict(review)
                for parameter_name, review in reviews.items()
            }

    # Bind every package export to the corrected defining symbol.  Export
    # aliases are plumbing-only, but their owner/gate/status follow the target.
    expected_aliases = set(API_ALIAS_TARGETS) | {
        "payne_zero_synthesis.__init__.__version__"
    }
    actual_aliases = {
        record["qualified_name"]
        for record in records
        if record["kind"] == "public_export"
        and record["qualified_name"].split(".")[1] == "__init__"
    }
    if actual_aliases != expected_aliases:
        raise RuntimeError(
            "package export registry changed; "
            f"missing={sorted(actual_aliases - expected_aliases)}, "
            f"extra={sorted(expected_aliases - actual_aliases)}"
        )
    for alias_name, target_name in API_ALIAS_TARGETS.items():
        targets = by_name.get(target_name, [])
        if not targets:
            raise KeyError(f"API alias target is missing: {target_name}")
        target = next(
            (
                record
                for record in targets
                if record["kind"] not in {"public_export", "annotated_field"}
            ),
            targets[0],
        )
        alias_fields = {
            "primary_location": target["primary_location"],
            "supporting_locations": list(target["supporting_locations"]),
            "semantic_disposition": "plumbing-only",
            "responsibility": f"public API alias for {target_name}",
            "gate": f"API alias identity + {target['gate']}",
            "status": target["status"],
            "alias_target": target_name,
        }
        for record in by_name[alias_name]:
            _apply_fields(
                record,
                alias_fields,
                review_source="explicit_api_alias_registry",
            )
    for record in by_name["payne_zero_synthesis.__init__.__version__"]:
        _apply_fields(
            record,
            override(
                "appendix-c",
                disposition="plumbing-only",
                responsibility="public package version string",
                gate="API/version smoke",
                status="boundary",
            ),
            review_source="explicit_api_alias_registry",
        )

    explicit_review_sources = {
        "explicit_symbol_registry",
        "explicit_default_callable_registry",
    }
    not_explicit = sorted(
        qualified_name
        for qualified_name in REQUIRED_EXPLICIT_SYMBOLS
        if qualified_name not in by_name
        or any(
            record.get("semantic_review_source") not in explicit_review_sources
            for record in by_name[qualified_name]
        )
    )
    if not_explicit:
        raise RuntimeError(
            "required branch-sensitive symbols fell back from explicit review: "
            f"{not_explicit}"
        )

    accepted_join = _accepted_artifact_join()
    for qualified_name, expected in accepted_join.items():
        for record in by_name[qualified_name]:
            record["accepted_authority_evidence"] = expected[
                "accepted_authority_evidence"
            ]

    for record in records:
        if record["mapping_precision"] != PRECISION:
            continue
        qualified_name = record["qualified_name"]
        module_name = _module_name(qualified_name)
        source_object = inventory_objects.get(qualified_name)
        if source_object is None and qualified_name in API_ALIAS_TARGETS:
            source_object = inventory_objects[API_ALIAS_TARGETS[qualified_name]]
        if source_object is None:
            source_object = {
                "kind": "public_export",
                "descriptor": {
                    "name": _source_spelling(qualified_name),
                    "source_module": module_name,
                },
            }
        record["source_module_sha256"] = inventory_modules[module_name]["sha256"]
        record["source_public_surface_sha256"] = _canonical_sha256(
            _module_public_surface(inventory_modules[module_name])
        )
        record["source_public_object_sha256"] = _canonical_sha256(source_object)
        if qualified_name in REVIEWED_DEFAULT_CONTRACTS:
            record["reviewed_default_contract"] = REVIEWED_DEFAULT_CONTRACTS[
                qualified_name
            ]
        if record.get("semantic_disposition") not in SEMANTIC_DISPOSITIONS:
            raise RuntimeError(
                "reviewed symbol lacks a valid semantic disposition: "
                f"{record['qualified_name']} [{record['kind']}]"
            )
        if record.get("source_spelling") != _source_spelling(record["qualified_name"]):
            raise RuntimeError(
                f"source spelling drifted for {record['qualified_name']}"
            )
        if qualified_name in DEFAULT_BEARING_PUBLIC_CALLABLES:
            expected_reviews = DEFAULT_PARAMETER_REVIEWS[qualified_name]
            if record.get("default_parameter_reviews") != expected_reviews:
                raise RuntimeError(
                    f"default-bearing callable lacks complete parameter reviews: "
                    f"{qualified_name}"
                )

    validate_complete_semantic_proof(ledger)
    return sum(record["mapping_precision"] == PRECISION for record in records)


def main() -> None:
    """Rewrite only the reviewed symbol-level semantic fields."""

    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    count = apply_overrides(ledger)

    LEDGER_PATH.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"applied {count} reviewed symbol records")


def _emit_fresh_default_parameter_snapshot() -> None:
    """Emit clean-import semantic state for the final validator trust boundary."""

    snapshot = {
        "authority": DEFAULT_PARAMETER_SEMANTICS_AUTHORITY,
        "contracts": EXPLICIT_DEFAULT_PARAMETER_CONTRACTS,
        "evidence": _PINNED_AST_DEFAULT_PARAMETER_USE_EVIDENCE,
        "manifest": SOURCE_DEFAULT_PARAMETER_MANIFEST,
        "reviews": DEFAULT_PARAMETER_REVIEWS,
        "source_root": str(PINNED_PAYNE_ZERO_SOURCE_ROOT.resolve()),
    }
    sys.stdout.write(
        json.dumps(
            snapshot,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--emit-fresh-default-parameter-snapshot"]:
        _emit_fresh_default_parameter_snapshot()
    else:
        main()
