"""Assemble the accepted Chapter 6 synthesis observation entirely in memory.

This module has no command-line entry point, output path, serializer, manifest
mutation, authorization, or publication function.  It accepts only the
independently frozen 754-member result from
``chapter06_synthesis_oracle_worker`` and projects it into the proposed
comparison-only mapping described by Section 11 of the synthesis oracle plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts import chapter06_synthesis_oracle_worker as oracle_worker


ASSEMBLER_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = ASSEMBLER_PATH.parents[1]
PLAN_PATH = REPOSITORY_ROOT / "design/chapter06_synthesis_fixture_oracle_plan.md"
PLAN_REBIND_CANDIDATE_PATH = (
    REPOSITORY_ROOT / "design/chapter06_synthesis_plan_rebind_candidate.md"
)
PLAN_REBIND_AUDIT_PATH = (
    REPOSITORY_ROOT / "design/chapter06_synthesis_plan_rebind_independent_audit.md"
)

ACCEPTED_WORKER_SHA256 = (
    "36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68"
)
ACCEPTED_PLAN_SHA256 = (
    "413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856"
)
ACCEPTED_PLAN_REBIND_CANDIDATE_SHA256 = (
    "dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93"
)
ACCEPTED_PLAN_REBIND_AUDIT_SHA256 = (
    "9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7"
)
ACCEPTED_RAW_KEY_COUNT = 754
ACCEPTED_RAW_SCHEMA_DIGEST = (
    "d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178"
)
ACCEPTED_RAW_PHYSICAL_FINGERPRINT = (
    "51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc"
)
ACCEPTED_RAW_FULL_FINGERPRINT = (
    "8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893"
)
ACCEPTED_RAW_SCOPE = "accepted exhaustive Chapter 6 CPU one-line synthesis capture"

COMPACT_SCHEMA_VERSION = 1
COMPACT_SIZE_CEILING_BYTES = 4 * 1024 * 1024
REGIME_NAMES = oracle_worker.REGIME_NAMES
GRID_NAMES = tuple(oracle_worker.GRID_SPECS)

RAW_DISPOSITIONS = frozenset(
    {"final", "derived_digest_only", "intentionally_ephemeral"}
)

FINAL_LEDGER_FIELDS = {
    "lower_excitation_exponent": (
        "ledger__lower_excitation_exponent",
        "dimensionless",
    ),
    "fastex_weight": ("ledger__fastex_weight", "dimensionless"),
    "pre_excitation_strength": (
        "ledger__pre_excitation_strength",
        "cm^2 g^-1 pre-profile scale",
    ),
    "post_fastex_line_amplitude": (
        "ledger__post_fastex_line_amplitude",
        "cm^2 g^-1 pre-profile scale",
    ),
    "center_cutoff_work_float64": (
        "ledger__center_cutoff_work_float64",
        "cm^2 g^-1",
    ),
    "pre_cutoff_active": ("ledger__pre_cutoff_active", "boolean"),
    "post_cutoff_active": ("activity__post_cutoff_mask", "boolean"),
    "radiative_damping_term": (
        "ledger__radiative_damping_term",
        "dimensionless",
    ),
    "stark_damping_term": (
        "ledger__stark_damping_term",
        "dimensionless",
    ),
    "van_der_waals_damping_term": (
        "ledger__van_der_waals_damping_term",
        "dimensionless",
    ),
    "total_damping": ("ledger__total_damping", "dimensionless"),
    "damping_ratio": ("ledger__damping_ratio", "dimensionless"),
    "center_profile": ("ledger__center_profile", "dimensionless Harris value"),
    "gross_center_opacity_float32": (
        "ledger__gross_center_opacity_float32",
        "cm^2 g^-1",
    ),
    "net_center_opacity_float32": (
        "ledger__net_center_opacity_float32",
        "cm^2 g^-1",
    ),
    "stimulated_center_factor_float32": (
        "ledger__stimulated_center_factor_float32",
        "dimensionless",
    ),
    "wing_reach": ("ledger__wing_reach", "count"),
    "nonzero_count": ("ledger__nonzero_count", "count"),
}

EXPECTED_LEDGER_FIELDS = frozenset(
    {
        "selected_partition_normalized_population",
        "selected_fractional_doppler_width",
        "doppler_width_nm",
        "mass_density",
        "population_over_mass_and_fractional_width",
        "classical_strength_float32",
        "pre_excitation_strength",
        "lower_excitation_exponent",
        "fastex_weight",
        "post_fastex_line_amplitude",
        "center_continuum_float32",
        "center_cutoff_work_float64",
        "pre_cutoff_active",
        "post_cutoff_active",
        "radiative_damping_term",
        "stark_damping_term",
        "van_der_waals_damping_term",
        "total_damping",
        "damping_ratio",
        "center_profile",
        "center_opacity_work_float64",
        "wing_center_profile",
        "wing_profile_amplitude",
        "wing_continuum_float32",
        "wing_cutoff_work_float64",
        "wing_reach",
        "use_far_wing",
        "ten_doppler_steps",
        "doppler_offset_per_pixel",
        "far_wing_coefficient",
        "gross_center_opacity_float32",
        "net_center_opacity_float32",
        "stimulated_emission_factor_float32",
        "stimulated_center_factor_float32",
        "nonzero_count",
        "electron_density_float32",
        "collision_density_proxy_float32",
        "doubled_electron_stark_term",
        "doubled_collision_van_der_waals_term",
        "electron_perturbed_total_damping",
        "collision_perturbed_total_damping",
        "electron_total_damping_delta",
        "collision_total_damping_delta",
        "expected_stark_damping_delta",
        "expected_van_der_waals_damping_delta",
        "electron_perturbation_other_terms_unchanged",
        "collision_perturbation_other_terms_unchanged",
    }
)

RAW_META_NOT_COPIED = frozenset(
    {
        "meta__fixture_path",
        "meta__subset_path",
        "meta__pinned_data_root",
        "meta__golden_read_performed",
        "meta__golden_publication_performed",
        "meta__regime_names",
    }
)

INVARIANT_SUPPORT_FIELDS = frozenset(
    {
        "harris_profile_h0_table",
        "harris_profile_h1_table",
        "harris_profile_h2_table",
        "exponential_integer_table",
        "exponential_fraction_table",
    }
)

INVARIANT_GRID_FIELDS = frozenset(
    {"wavelength_grid", "n_wavelengths", "grid_resolution"}
)

CATALOG_UNITS = {
    "line_type": "type code",
    "atomic_number": "atomic number",
    "ion_stage": "one-based ion stage",
    "wavelength_nm": "nm",
    "index_wavelength_nm": "nm",
    "oscillator_strength": "dimensionless gf",
    "lower_excitation_cm": "cm^-1",
    "radiative_damping": "dimensionless normalized by 12.5664*frequency_hz",
    "stark_damping": "cm^3 normalized by 12.5664*frequency_hz",
    "van_der_waals_damping": "cm^3 normalized by 12.5664*frequency_hz",
    "raw_radiative_damping_log": "log10(s^-1)",
    "raw_stark_damping_log": "log10(cm^3 s^-1)",
    "raw_van_der_waals_damping_log": "log10(cm^3 s^-1)",
}

INVARIANT_UNITS = {
    "metal_catalog_index": "zero-based index",
    "metal_classical_strength": "cm^2",
    "metal_lower_excitation_cm": "cm^-1",
    "metal_radiative_damping": ("dimensionless normalized by 12.5664*frequency_hz"),
    "metal_stark_damping": "cm^3 normalized by 12.5664*frequency_hz",
    "metal_van_der_waals_damping": ("cm^3 normalized by 12.5664*frequency_hz"),
    "metal_wavelength_nm": "nm",
    "metal_population_ion_stage_index": "zero-based index",
    "metal_population_element_index": "zero-based index",
    "metal_center_index": "zero-based index",
    "metal_wing_index": "zero-based index",
    "metal_center_clamped": "zero-based index",
    "metal_wing_clamped": "zero-based index",
    "auto_catalog_index": "zero-based index",
    "auto_oscillator_strength": "dimensionless gf",
    "auto_lower_excitation_cm": "cm^-1",
    "auto_radiative_damping": "s^-1",
    "auto_stark_damping": "cm^3 s^-1",
    "auto_van_der_waals_damping": "cm^3 s^-1",
    "auto_wavelength_nm": "nm",
    "auto_population_ion_stage_index": "zero-based index",
    "auto_population_element_index": "zero-based index",
    "auto_center_index": "zero-based index",
    "auto_center_clamped": "zero-based index",
    "helium_classical_strength": "cm^2",
    "helium_lower_excitation_cm": "cm^-1",
    "helium_radiative_damping": ("dimensionless normalized by 12.5664*frequency_hz"),
    "helium_stark_damping": "cm^3 normalized by 12.5664*frequency_hz",
    "helium_van_der_waals_damping": ("cm^3 normalized by 12.5664*frequency_hz"),
    "helium_wavelength_nm": "nm",
    "helium_population_ion_stage_index": "zero-based index",
    "helium_population_element_index": "zero-based index",
    "helium_center_index": "zero-based index",
    "helium_line_type": "type code",
}


class CompactAssemblyError(RuntimeError):
    """Raised when raw ownership or comparison-only semantics drift."""


@dataclass(frozen=True)
class RawOwnership:
    """Disposition of one accepted raw-capture member."""

    raw_name: str
    disposition: str
    target: str
    reason: str


@dataclass(frozen=True)
class CompactMemberSpec:
    """Explicit schema description for one proposed comparison member."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    unit: str
    role: str


@dataclass(frozen=True)
class CompactAssembly:
    """Detached proposed mapping plus schema and exhaustive raw ownership."""

    arrays: dict[str, np.ndarray]
    schema: tuple[CompactMemberSpec, ...]
    raw_ownership: tuple[RawOwnership, ...]


def sha256(path: Path) -> str:
    """Return one regular file's SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_regular_nonsymlink(path: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise CompactAssemblyError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise CompactAssemblyError(f"{label} is not a regular file: {resolved}")
    return resolved


def _verify_rebind_authority() -> None:
    """Bind the exact accepted repaired-plan worker provenance."""

    if oracle_worker.DESIGN_PATH.resolve() != PLAN_PATH.resolve():
        raise CompactAssemblyError("scientific worker design path changed")
    accepted = {
        "repaired synthesis plan": (PLAN_PATH, ACCEPTED_PLAN_SHA256),
        "plan-rebind candidate": (
            PLAN_REBIND_CANDIDATE_PATH,
            ACCEPTED_PLAN_REBIND_CANDIDATE_SHA256,
        ),
        "plan-rebind independent audit": (
            PLAN_REBIND_AUDIT_PATH,
            ACCEPTED_PLAN_REBIND_AUDIT_SHA256,
        ),
    }
    for label, (path, expected_sha256) in accepted.items():
        resolved = _require_regular_nonsymlink(path, label)
        actual_sha256 = sha256(resolved)
        if actual_sha256 != expected_sha256:
            raise CompactAssemblyError(
                f"accepted {label} identity changed: {actual_sha256}; "
                f"expected {expected_sha256}"
            )


def _to_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise CompactAssemblyError("object dtype is forbidden")
    return np.array(array, copy=True, order="C")


def _scalar(arrays: Mapping[str, np.ndarray], name: str) -> Any:
    if name not in arrays:
        raise CompactAssemblyError(f"required scalar is missing: {name}")
    value = np.asarray(arrays[name])
    if value.shape != ():
        raise CompactAssemblyError(f"required scalar is not scalar: {name}")
    return value.item()


def _bitwise_equal(left: Any, right: Any) -> bool:
    first = np.asarray(left)
    second = np.asarray(right)
    return (
        first.dtype.str == second.dtype.str
        and first.shape == second.shape
        and np.ascontiguousarray(first).tobytes()
        == np.ascontiguousarray(second).tobytes()
    )


def _require_equal(left: Any, right: Any, label: str) -> None:
    if not _bitwise_equal(left, right):
        raise CompactAssemblyError(f"deduplication identity failed: {label}")


def member_sha256(value: Any) -> str:
    """Hash one array's dtype, shape, and contiguous bytes."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise CompactAssemblyError("object dtype is forbidden")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def schema_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash lexical member names, dtypes, and shapes."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise CompactAssemblyError(f"object dtype is forbidden: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    return digest.hexdigest()


def payload_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash all proposed comparison values except the self fingerprint."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        if name == "meta__compact_payload_fingerprint":
            continue
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise CompactAssemblyError(f"object dtype is forbidden: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def ownership_digest(ownership: tuple[RawOwnership, ...]) -> str:
    """Hash exact raw name, disposition, target, and reason."""

    digest = hashlib.sha256()
    for item in ownership:
        for value in (
            item.raw_name,
            item.disposition,
            item.target,
            item.reason,
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


class _Builder:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.specs: dict[str, CompactMemberSpec] = {}

    def add(
        self,
        name: str,
        value: Any,
        *,
        axes: tuple[str, ...] = (),
        unit: str = "metadata",
        role: str = "comparison-only",
    ) -> None:
        if name in self.arrays:
            raise CompactAssemblyError(f"duplicate compact member: {name}")
        array = _to_array(value)
        if len(axes) != array.ndim:
            raise CompactAssemblyError(
                f"axis rank does not match {name}: axes={axes}, shape={array.shape}"
            )
        self.arrays[name] = array
        self.specs[name] = CompactMemberSpec(
            name=name,
            dtype=array.dtype.str,
            shape=tuple(int(size) for size in array.shape),
            axes=axes,
            unit=unit,
            role=role,
        )

    def replace_scalar(self, name: str, value: Any) -> None:
        if name not in self.arrays or self.arrays[name].shape != ():
            raise CompactAssemblyError(f"cannot replace non-scalar member: {name}")
        replacement = _to_array(value)
        if replacement.shape != () or replacement.dtype != self.arrays[name].dtype:
            raise CompactAssemblyError(f"scalar replacement changed schema: {name}")
        self.arrays[name] = replacement

    def finish(self) -> tuple[dict[str, np.ndarray], tuple[CompactMemberSpec, ...]]:
        arrays = {name: _to_array(self.arrays[name]) for name in sorted(self.arrays)}
        specs = tuple(self.specs[name] for name in sorted(self.specs))
        return arrays, specs


def _validated_raw(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Copy and require the independently frozen scientific observation."""

    _verify_rebind_authority()
    if sha256(oracle_worker.WORKER_PATH) != ACCEPTED_WORKER_SHA256:
        raise CompactAssemblyError("accepted synthesis worker bytes changed")
    copied = oracle_worker.deterministic_result(raw)
    if len(copied) != ACCEPTED_RAW_KEY_COUNT:
        raise CompactAssemblyError(
            f"raw capture has {len(copied)} keys; expected {ACCEPTED_RAW_KEY_COUNT}"
        )
    actual_schema = oracle_worker._capture_schema_digest(copied)
    if actual_schema != ACCEPTED_RAW_SCHEMA_DIGEST:
        raise CompactAssemblyError(
            f"raw schema digest is {actual_schema}; "
            f"expected {ACCEPTED_RAW_SCHEMA_DIGEST}"
        )
    exact_scalars = {
        "meta__capture_schema_digest": ACCEPTED_RAW_SCHEMA_DIGEST,
        "meta__physical_payload_fingerprint": (ACCEPTED_RAW_PHYSICAL_FINGERPRINT),
        "meta__full_capture_fingerprint": ACCEPTED_RAW_FULL_FINGERPRINT,
        "meta__worker_sha256": ACCEPTED_WORKER_SHA256,
        "meta__design_sha256": ACCEPTED_PLAN_SHA256,
        "meta__capture_scope": ACCEPTED_RAW_SCOPE,
    }
    for name, expected in exact_scalars.items():
        if _scalar(copied, name) != expected:
            raise CompactAssemblyError(f"accepted raw identity changed: {name}")
    if not bool(_scalar(copied, "meta__capture_scope_complete")):
        raise CompactAssemblyError("raw capture scope is not accepted")
    if bool(_scalar(copied, "meta__golden_read_performed")):
        raise CompactAssemblyError("scientific worker claimed a golden read")
    if bool(_scalar(copied, "meta__golden_publication_performed")):
        raise CompactAssemblyError("scientific worker claimed golden publication")
    if (
        oracle_worker._fingerprint(copied, physical_payload_only=True)
        != ACCEPTED_RAW_PHYSICAL_FINGERPRINT
    ):
        raise CompactAssemblyError("raw physical fingerprint does not recompute")
    if (
        oracle_worker._fingerprint(copied, physical_payload_only=False)
        != ACCEPTED_RAW_FULL_FINGERPRINT
    ):
        raise CompactAssemblyError("raw full fingerprint does not recompute")
    try:
        oracle_worker._validate_complete_result(copied)
    except RuntimeError as error:
        raise CompactAssemblyError(
            f"accepted raw science validation failed: {error}"
        ) from error
    return copied


def _stack(raw: Mapping[str, np.ndarray], pattern: str) -> np.ndarray:
    return np.stack(
        [np.asarray(raw[pattern.format(regime=regime)]) for regime in REGIME_NAMES]
    )


def _mark(
    ownership: dict[str, RawOwnership],
    raw_name: str,
    disposition: str,
    target: str,
    reason: str,
) -> None:
    if raw_name in ownership:
        raise CompactAssemblyError(f"raw member received two owners: {raw_name}")
    if disposition not in RAW_DISPOSITIONS:
        raise CompactAssemblyError(f"unknown raw disposition: {disposition}")
    ownership[raw_name] = RawOwnership(
        raw_name=raw_name,
        disposition=disposition,
        target=target,
        reason=reason,
    )


def _copy_metadata(
    raw: Mapping[str, np.ndarray],
    builder: _Builder,
    ownership: dict[str, RawOwnership],
) -> None:
    for raw_name in sorted(name for name in raw if name.startswith("identity__")):
        unit = (
            "Git commit identity"
            if raw_name == "identity__payne_zero_commit"
            else "SHA-256 identity"
        )
        builder.add(raw_name, raw[raw_name], unit=unit)
        _mark(
            ownership,
            raw_name,
            "final",
            raw_name,
            "pinned source/table identity retained",
        )

    for raw_name in sorted(name for name in raw if name.startswith("meta__")):
        if raw_name in RAW_META_NOT_COPIED:
            if raw_name == "meta__regime_names":
                continue
            disposition = (
                "derived_digest_only"
                if raw_name.startswith("meta__golden_")
                else "intentionally_ephemeral"
            )
            _mark(
                ownership,
                raw_name,
                disposition,
                "",
                "machine-local path or already-validated worker lifecycle flag",
            )
            continue
        value = np.asarray(raw[raw_name])
        axes = () if value.ndim == 0 else ("metadata_entry",)
        builder.add(raw_name, value, axes=axes)
        _mark(
            ownership,
            raw_name,
            "final",
            raw_name,
            "accepted worker/source/environment metadata retained",
        )

    builder.add(
        "axis__regime_name",
        np.asarray(REGIME_NAMES),
        axes=("regime",),
        unit="regime label",
        role="axis",
    )
    builder.add(
        "axis__depth_index",
        np.arange(6, dtype=np.int64),
        axes=("depth",),
        unit="zero-based index",
        role="axis",
    )
    _mark(
        ownership,
        "meta__regime_names",
        "final",
        "axis__regime_name",
        "coalesced into the single regime axis",
    )

    builder.add(
        "meta__fixture_relative_path",
        np.asarray("data/fixtures/chapter05_continuum_states.npz"),
    )
    builder.add(
        "meta__subset_relative_path",
        np.asarray("data/subsets/chapter06_fe_i_source_row_873702.npz"),
    )
    builder.add(
        "meta__line_profile_table_relative_path",
        np.asarray("data/static/synthesis_tables/line_profile_tables.npz"),
    )
    builder.add(
        "meta__continuum_table_relative_path",
        np.asarray("data/static/synthesis_tables/continuum_tables.npz"),
    )
    builder.add(
        "meta__continuum_edge_grid_relative_path",
        np.asarray("data/static/synthesis_tables/continuum_edge_grid.npz"),
    )

    _mark(
        ownership,
        "fixture_payload__content_digest",
        "derived_digest_only",
        "meta__fixture_payload_digest",
        "duplicates the accepted fixture payload digest metadata",
    )
    _mark(
        ownership,
        "fixture_payload__field_names",
        "intentionally_ephemeral",
        "",
        "full upstream input schema does not belong in comparison output",
    )
    builder.add(
        "meta__subset_payload_digest",
        raw["subset_payload__content_digest"],
        unit="SHA-256 content digest",
    )
    _mark(
        ownership,
        "subset_payload__content_digest",
        "final",
        "meta__subset_payload_digest",
        "compact subset payload identity retained",
    )
    _mark(
        ownership,
        "subset_payload__field_names",
        "intentionally_ephemeral",
        "",
        "raw source-field inventory remains with the input subset",
    )


def _add_catalog(
    raw: Mapping[str, np.ndarray],
    builder: _Builder,
    ownership: dict[str, RawOwnership],
) -> None:
    record_fields = tuple(str(value) for value in raw["record__field_names"].tolist())
    if record_fields != oracle_worker.CATALOG_LINE_FIELDS:
        raise CompactAssemblyError("derived record field order changed")
    _mark(
        ownership,
        "record__field_names",
        "derived_digest_only",
        "meta__catalog_line_fields",
        "catalog metadata already records the field order",
    )
    for field_name in oracle_worker.CATALOG_LINE_FIELDS:
        record_name = f"record__{field_name}"
        catalog_name = f"catalog__{field_name}"
        _require_equal(raw[record_name], raw[catalog_name], field_name)
        target = f"mapping__{field_name}"
        builder.add(
            target,
            raw[catalog_name],
            axes=("line",),
            unit=CATALOG_UNITS[field_name],
        )
        _mark(
            ownership,
            catalog_name,
            "final",
            target,
            "one readable physical mapping field retained",
        )
        _mark(
            ownership,
            record_name,
            "derived_digest_only",
            target,
            "byte-identical duplicate of the retained catalog mapping",
        )

    builder.add(
        "mapping__helium_line_type",
        raw["catalog__helium_line_type"],
        axes=("helium_line",),
        unit="type code",
    )
    builder.add(
        "mapping__helium_line_center_cutoff_ratio",
        raw["catalog__helium_line_center_cutoff_ratio"],
        unit="dimensionless",
    )
    for name, target in (
        ("helium_line_type", "mapping__helium_line_type"),
        (
            "helium_line_center_cutoff_ratio",
            "mapping__helium_line_center_cutoff_ratio",
        ),
    ):
        _mark(
            ownership,
            f"catalog__{name}",
            "final",
            target,
            "required general-constructor support retained",
        )

    archive_hash = raw["identity__table__line_profile_tables__sha256"]
    builder.add(
        "support__line_profile_table_archive_sha256",
        archive_hash,
        unit="SHA-256 file identity",
    )
    for table_name in (
        "harris_profile_h0_table",
        "harris_profile_h1_table",
        "harris_profile_h2_table",
    ):
        raw_name = f"catalog__{table_name}"
        target = f"support__{table_name}_sha256"
        builder.add(
            target,
            np.asarray(member_sha256(raw[raw_name])),
            unit="SHA-256 member identity over dtype, shape, and bytes",
        )
        _mark(
            ownership,
            raw_name,
            "derived_digest_only",
            target,
            "full Harris table replaced by exact member identity",
        )


def _invariant_axis(field_name: str) -> tuple[str, ...]:
    if field_name.startswith("metal_"):
        return ("metal_line",)
    if field_name.startswith("auto_"):
        return ("auto_line",)
    if field_name.startswith("helium_"):
        return ("helium_line",)
    raise CompactAssemblyError(f"unknown invariant axis: {field_name}")


def _add_invariants(
    raw: Mapping[str, np.ndarray],
    builder: _Builder,
    ownership: dict[str, RawOwnership],
) -> None:
    canonical_prefix = "invariant__canonical__"
    coarse_prefix = "invariant__coarse__"

    _require_equal(
        raw[f"{canonical_prefix}wavelength_grid"],
        raw["grid__canonical__wavelength_nm"],
        "canonical invariant wavelength grid",
    )
    _require_equal(
        raw[f"{coarse_prefix}wavelength_grid"],
        raw["grid__coarse__wavelength_nm"],
        "coarse invariant wavelength grid",
    )

    builder.add(
        "invariant__n_wavelengths",
        raw[f"{canonical_prefix}n_wavelengths"],
        unit="count",
    )
    builder.add(
        "invariant__grid_resolution",
        raw[f"{canonical_prefix}grid_resolution"],
        unit="dimensionless resolving power",
    )
    for field_name in ("n_wavelengths", "grid_resolution"):
        _mark(
            ownership,
            f"{canonical_prefix}{field_name}",
            "final",
            f"invariant__{field_name}",
            "canonical invariant scalar retained",
        )
    _mark(
        ownership,
        f"{canonical_prefix}wavelength_grid",
        "derived_digest_only",
        "axis__wavelength_nm",
        "byte-identical duplicate of the single retained wavelength axis",
    )

    for field_name in (
        oracle_worker.METAL_INVARIANT_FIELDS
        + oracle_worker.AUTO_INVARIANT_FIELDS
        + oracle_worker.HELIUM_ARRAY_INVARIANT_FIELDS
    ):
        raw_name = f"{canonical_prefix}{field_name}"
        target = f"invariant__{field_name}"
        builder.add(
            target,
            raw[raw_name],
            axes=_invariant_axis(field_name),
            unit=INVARIANT_UNITS[field_name],
        )
        _mark(
            ownership,
            raw_name,
            "final",
            target,
            "canonical ordinary or exact empty-route invariant retained",
        )

    builder.add(
        "invariant__helium_cutoff",
        raw[f"{canonical_prefix}helium_cutoff"],
        unit="dimensionless",
    )
    _mark(
        ownership,
        f"{canonical_prefix}helium_cutoff",
        "final",
        "invariant__helium_cutoff",
        "general-constructor scalar retained",
    )

    for field_name in INVARIANT_SUPPORT_FIELDS:
        canonical_name = f"{canonical_prefix}{field_name}"
        coarse_name = f"{coarse_prefix}{field_name}"
        _require_equal(raw[canonical_name], raw[coarse_name], field_name)
        if field_name.startswith("harris_"):
            _require_equal(
                raw[canonical_name],
                raw[f"catalog__{field_name}"],
                f"catalog/invariant {field_name}",
            )
            target = f"support__{field_name}_sha256"
        else:
            target = f"support__{field_name}_sha256"
            builder.add(
                target,
                np.asarray(member_sha256(raw[canonical_name])),
                unit="SHA-256 member identity over dtype, shape, and bytes",
            )
        _mark(
            ownership,
            canonical_name,
            "derived_digest_only",
            target,
            "full numerical support table replaced by exact member identity",
        )
        _mark(
            ownership,
            coarse_name,
            "derived_digest_only",
            target,
            "byte-identical second-grid table copy deduplicated",
        )

    _mark(
        ownership,
        f"{coarse_prefix}wavelength_grid",
        "derived_digest_only",
        "coarse__grid_sha256",
        "coarse axis reduced to identity and scalar geometry",
    )
    for field_name in ("n_wavelengths", "grid_resolution"):
        target = (
            "coarse__grid_count"
            if field_name == "n_wavelengths"
            else "coarse__grid_resolution"
        )
        _mark(
            ownership,
            f"{coarse_prefix}{field_name}",
            "final",
            target,
            "coarse scalar geometry retained",
        )

    grid_specific = {
        "metal_center_index",
        "metal_wing_index",
        "metal_center_clamped",
        "metal_wing_clamped",
        "auto_center_index",
        "auto_center_clamped",
        "helium_center_index",
    }
    for field_name in (
        oracle_worker.METAL_INVARIANT_FIELDS
        + oracle_worker.AUTO_INVARIANT_FIELDS
        + oracle_worker.HELIUM_ARRAY_INVARIANT_FIELDS
    ):
        coarse_name = f"{coarse_prefix}{field_name}"
        canonical_name = f"{canonical_prefix}{field_name}"
        if field_name not in grid_specific:
            _require_equal(
                raw[coarse_name],
                raw[canonical_name],
                f"coarse/canonical invariant {field_name}",
            )
        _mark(
            ownership,
            coarse_name,
            "derived_digest_only",
            (
                "coarse__line_center_index"
                if field_name in grid_specific
                else f"invariant__{field_name}"
            ),
            "coarse-grid invariant validated then deduplicated",
        )


def _validate_continuum(
    raw: Mapping[str, np.ndarray],
    regime: str,
    grid_name: str,
) -> None:
    prefix = f"{regime}__{grid_name}"
    absorption = raw[f"{prefix}__continuum_absorption_float64"]
    scattering = raw[f"{prefix}__continuum_scattering_float64"]
    total = raw[f"{prefix}__continuum_total_float64"]
    line_input = raw[f"{prefix}__continuum_line_input_float32"]
    reconstructed = np.asarray(absorption) + np.asarray(scattering)
    if not np.array_equal(reconstructed, total):
        raise CompactAssemblyError(
            f"continuum sum does not reconstruct: {regime}/{grid_name}"
        )
    if not np.array_equal(np.asarray(total).astype(np.float32), line_input):
        raise CompactAssemblyError(
            f"continuum float32 fence does not reconstruct: {regime}/{grid_name}"
        )
    center = int(_scalar(raw, f"grid__{grid_name}__line_center_index"))
    wing = int(_scalar(raw, f"grid__{grid_name}__line_wing_index"))
    _require_equal(
        np.asarray(line_input)[:, center],
        raw[f"{prefix}__ledger__center_continuum_float32"],
        f"center continuum sample {regime}/{grid_name}",
    )
    _require_equal(
        np.asarray(line_input)[:, wing],
        raw[f"{prefix}__ledger__wing_continuum_float32"],
        f"wing continuum sample {regime}/{grid_name}",
    )


def _validate_route_deduplication(
    raw: Mapping[str, np.ndarray],
    regime: str,
    grid_name: str,
) -> None:
    prefix = f"{regime}__{grid_name}"
    for lifecycle in ("gross", "net"):
        _require_equal(
            raw[f"{prefix}__{lifecycle}_batched_float32"],
            raw[f"{prefix}__{lifecycle}_loop_float32"],
            f"{lifecycle} loop/batched {regime}/{grid_name}",
        )
    stimulation = raw[f"{prefix}__ledger__stimulated_emission_factor_float32"]
    for wing_mode in ("batched", "loop"):
        gross = raw[f"{prefix}__gross_{wing_mode}_float32"]
        net = raw[f"{prefix}__net_{wing_mode}_float32"]
        reconstructed = np.multiply(gross, stimulation, dtype=np.float32)
        _require_equal(
            reconstructed,
            net,
            f"gross/stimulation/net {regime}/{grid_name}/{wing_mode}",
        )


def _add_canonical_products(
    raw: Mapping[str, np.ndarray],
    builder: _Builder,
    ownership: dict[str, RawOwnership],
) -> None:
    wavelength = raw["grid__canonical__wavelength_nm"]
    builder.add(
        "axis__wavelength_nm",
        wavelength,
        axes=("wavelength",),
        unit="nm",
        role="axis",
    )
    for suffix, target, unit in (
        ("wavelength_nm", "axis__wavelength_nm", "nm"),
        ("sha256", "grid__canonical_sha256", "SHA-256 wavelength-axis identity"),
        (
            "line_center_index",
            "grid__canonical_line_center_index",
            "zero-based index",
        ),
        (
            "line_wing_index",
            "grid__canonical_line_wing_index",
            "zero-based index",
        ),
    ):
        raw_name = f"grid__canonical__{suffix}"
        if target != "axis__wavelength_nm":
            builder.add(target, raw[raw_name], unit=unit)
        _mark(
            ownership,
            raw_name,
            "final",
            target,
            "single canonical grid coordinate or identity retained",
        )

    gross = _stack(raw, "{regime}__canonical__gross_batched_float32")
    net = _stack(raw, "{regime}__canonical__net_batched_float32")
    builder.add(
        "opacity__gross_float32",
        gross,
        axes=("regime", "depth", "wavelength"),
        unit="cm^2 g^-1 sampled kappa_nu(c/lambda_i), not per nm",
    )
    builder.add(
        "opacity__net_float32",
        net,
        axes=("regime", "depth", "wavelength"),
        unit="cm^2 g^-1 sampled kappa_nu(c/lambda_i), not per nm",
    )

    for regime_index, regime in enumerate(REGIME_NAMES):
        prefix = f"{regime}__canonical"
        _validate_continuum(raw, regime, "canonical")
        _validate_route_deduplication(raw, regime, "canonical")
        for lifecycle in ("gross", "net"):
            batched_name = f"{prefix}__{lifecycle}_batched_float32"
            loop_name = f"{prefix}__{lifecycle}_loop_float32"
            target = f"opacity__{lifecycle}_float32[{regime_index}]"
            _mark(
                ownership,
                batched_name,
                "final",
                target,
                "one canonical stacked opacity slab retained",
            )
            _mark(
                ownership,
                loop_name,
                "derived_digest_only",
                target,
                "bit-identical loop duplicate discarded",
            )

    for field_name, (target, unit) in FINAL_LEDGER_FIELDS.items():
        values = _stack(
            raw,
            f"{{regime}}__canonical__ledger__{field_name}",
        )
        builder.add(
            target,
            values,
            axes=("regime", "depth"),
            unit=unit,
        )
        for regime_index, regime in enumerate(REGIME_NAMES):
            _mark(
                ownership,
                f"{regime}__canonical__ledger__{field_name}",
                "final",
                f"{target}[{regime_index}]",
                "comparison-only factor/branch ledger retained",
            )

    for sample_name, target in (
        ("center_continuum_float32", "continuum__center_float32"),
        ("wing_continuum_float32", "continuum__wing_float32"),
    ):
        builder.add(
            target,
            _stack(raw, f"{{regime}}__canonical__ledger__{sample_name}"),
            axes=("regime", "depth"),
            unit="cm^2 g^-1",
        )
        for regime_index, regime in enumerate(REGIME_NAMES):
            _mark(
                ownership,
                f"{regime}__canonical__ledger__{sample_name}",
                "final",
                f"{target}[{regime_index}]",
                "continuum cutoff sample retained",
            )

    continuum_kinds = (
        "continuum_absorption_float64",
        "continuum_scattering_float64",
        "continuum_total_float64",
        "continuum_line_input_float32",
    )
    for kind in continuum_kinds:
        target = f"continuum__canonical__{kind}_sha256"
        builder.add(
            target,
            np.asarray(
                [
                    member_sha256(raw[f"{regime}__canonical__{kind}"])
                    for regime in REGIME_NAMES
                ]
            ),
            axes=("regime",),
            unit="SHA-256 member identity over dtype, shape, and bytes",
        )
        for regime_index, regime in enumerate(REGIME_NAMES):
            _mark(
                ownership,
                f"{regime}__canonical__{kind}",
                "derived_digest_only",
                f"{target}[{regime_index}]",
                "full continuum slab discarded after reconstruction and digest",
            )

    stimulation_hash_target = "stimulation__canonical_factor_sha256"
    builder.add(
        stimulation_hash_target,
        np.asarray(
            [
                member_sha256(
                    raw[
                        f"{regime}__canonical__ledger__"
                        "stimulated_emission_factor_float32"
                    ]
                )
                for regime in REGIME_NAMES
            ]
        ),
        axes=("regime",),
        unit="SHA-256 member identity over dtype, shape, and bytes",
    )
    builder.add(
        "stimulation__canonical_batched_reconstruction_exact",
        np.ones(4, dtype=np.bool_),
        axes=("regime",),
        unit="boolean",
    )
    builder.add(
        "stimulation__canonical_loop_reconstruction_exact",
        np.ones(4, dtype=np.bool_),
        axes=("regime",),
        unit="boolean",
    )
    for regime_index, regime in enumerate(REGIME_NAMES):
        _mark(
            ownership,
            (f"{regime}__canonical__ledger__stimulated_emission_factor_float32"),
            "derived_digest_only",
            f"{stimulation_hash_target}[{regime_index}]",
            "full factor discarded after both exact net reconstructions",
        )


def _active_summary(
    values: np.ndarray, active: np.ndarray, operation: str
) -> np.ndarray:
    summaries = []
    for regime_index in range(len(REGIME_NAMES)):
        selected = values[regime_index][active[regime_index]]
        if selected.size == 0:
            raise CompactAssemblyError("coarse activity has no active depth")
        summaries.append(selected.min() if operation == "min" else selected.max())
    return np.asarray(summaries, dtype=values.dtype)


def _add_coarse_evidence(
    raw: Mapping[str, np.ndarray],
    builder: _Builder,
    ownership: dict[str, RawOwnership],
) -> None:
    coarse_grid = np.asarray(raw["grid__coarse__wavelength_nm"])
    coarse_center = int(_scalar(raw, "grid__coarse__line_center_index"))
    builder.add(
        "coarse__grid_count",
        np.asarray(coarse_grid.size, dtype=np.int64),
        unit="count",
    )
    builder.add(
        "coarse__grid_resolution",
        raw["invariant__coarse__grid_resolution"],
        unit="dimensionless resolving power",
    )
    builder.add(
        "coarse__first_wavelength_nm",
        np.asarray(coarse_grid[0], dtype=np.float64),
        unit="nm",
    )
    builder.add(
        "coarse__last_wavelength_nm",
        np.asarray(coarse_grid[-1], dtype=np.float64),
        unit="nm",
    )
    builder.add(
        "coarse__line_center_index",
        np.asarray(coarse_center, dtype=np.int64),
        unit="zero-based index",
    )
    builder.add(
        "coarse__line_center_wavelength_nm",
        np.asarray(coarse_grid[coarse_center], dtype=np.float64),
        unit="nm",
    )
    builder.add(
        "coarse__line_wing_index",
        raw["grid__coarse__line_wing_index"],
        unit="zero-based index",
    )
    builder.add(
        "coarse__grid_sha256",
        raw["grid__coarse__sha256"],
        unit="SHA-256 wavelength-axis identity",
    )
    for raw_name, target in (
        ("grid__coarse__wavelength_nm", "coarse__grid_sha256"),
        ("grid__coarse__sha256", "coarse__grid_sha256"),
        ("grid__coarse__line_center_index", "coarse__line_center_index"),
        ("grid__coarse__line_wing_index", "coarse__line_wing_index"),
    ):
        _mark(
            ownership,
            raw_name,
            "final",
            target,
            "coarse axis reduced to compact geometry and identity",
        )

    activity = _stack(raw, "{regime}__coarse__ledger__post_cutoff_active")
    reach = _stack(raw, "{regime}__coarse__ledger__wing_reach")
    nonzero = _stack(raw, "{regime}__coarse__ledger__nonzero_count")
    gross_peak = np.asarray(
        [
            np.asarray(raw[f"{regime}__coarse__gross_batched_float32"]).max()
            for regime in REGIME_NAMES
        ],
        dtype=np.float32,
    )
    net_peak = np.asarray(
        [
            np.asarray(raw[f"{regime}__coarse__net_batched_float32"]).max()
            for regime in REGIME_NAMES
        ],
        dtype=np.float32,
    )
    builder.add(
        "coarse__activity_mask",
        activity,
        axes=("regime", "depth"),
        unit="boolean",
    )
    builder.add(
        "coarse__gross_peak_float32",
        gross_peak,
        axes=("regime",),
        unit="cm^2 g^-1",
    )
    builder.add(
        "coarse__net_peak_float32",
        net_peak,
        axes=("regime",),
        unit="cm^2 g^-1",
    )
    builder.add(
        "coarse__wing_reach_minimum",
        _active_summary(reach, activity, "min"),
        axes=("regime",),
        unit="count",
    )
    builder.add(
        "coarse__wing_reach_maximum",
        _active_summary(reach, activity, "max"),
        axes=("regime",),
        unit="count",
    )
    builder.add(
        "coarse__nonzero_count_minimum",
        _active_summary(nonzero, activity, "min"),
        axes=("regime",),
        unit="count",
    )
    builder.add(
        "coarse__nonzero_count_maximum",
        _active_summary(nonzero, activity, "max"),
        axes=("regime",),
        unit="count",
    )
    for field_name, target in (
        ("post_cutoff_active", "coarse__activity_mask"),
        ("wing_reach", "coarse__wing_reach_minimum"),
        ("nonzero_count", "coarse__nonzero_count_minimum"),
    ):
        for regime_index, regime in enumerate(REGIME_NAMES):
            _mark(
                ownership,
                f"{regime}__coarse__ledger__{field_name}",
                "final",
                f"{target}[{regime_index}]",
                "compact coarse stability evidence retained",
            )

    continuum_kinds = (
        "continuum_absorption_float64",
        "continuum_scattering_float64",
        "continuum_total_float64",
        "continuum_line_input_float32",
    )
    for kind in continuum_kinds:
        target = f"continuum__coarse__{kind}_sha256"
        builder.add(
            target,
            np.asarray(
                [
                    member_sha256(raw[f"{regime}__coarse__{kind}"])
                    for regime in REGIME_NAMES
                ]
            ),
            axes=("regime",),
            unit="SHA-256 member identity over dtype, shape, and bytes",
        )
        for regime_index, regime in enumerate(REGIME_NAMES):
            _mark(
                ownership,
                f"{regime}__coarse__{kind}",
                "derived_digest_only",
                f"{target}[{regime_index}]",
                "coarse continuum discarded after reconstruction and digest",
            )

    stimulation_target = "stimulation__coarse_factor_sha256"
    builder.add(
        stimulation_target,
        np.asarray(
            [
                member_sha256(
                    raw[f"{regime}__coarse__ledger__stimulated_emission_factor_float32"]
                )
                for regime in REGIME_NAMES
            ]
        ),
        axes=("regime",),
        unit="SHA-256 member identity over dtype, shape, and bytes",
    )
    builder.add(
        "stimulation__coarse_batched_reconstruction_exact",
        np.ones(4, dtype=np.bool_),
        axes=("regime",),
        unit="boolean",
    )
    builder.add(
        "stimulation__coarse_loop_reconstruction_exact",
        np.ones(4, dtype=np.bool_),
        axes=("regime",),
        unit="boolean",
    )
    for regime_index, regime in enumerate(REGIME_NAMES):
        _mark(
            ownership,
            (f"{regime}__coarse__ledger__stimulated_emission_factor_float32"),
            "derived_digest_only",
            f"{stimulation_target}[{regime_index}]",
            "coarse full factor discarded after both exact reconstructions",
        )

    for regime_index, regime in enumerate(REGIME_NAMES):
        _validate_continuum(raw, regime, "coarse")
        _validate_route_deduplication(raw, regime, "coarse")
        for lifecycle, target in (
            ("gross", "coarse__gross_peak_float32"),
            ("net", "coarse__net_peak_float32"),
        ):
            for wing_mode in ("batched", "loop"):
                _mark(
                    ownership,
                    f"{regime}__coarse__{lifecycle}_{wing_mode}_float32",
                    ("final" if wing_mode == "batched" else "derived_digest_only"),
                    f"{target}[{regime_index}]",
                    (
                        "coarse slab reduced to peak summary"
                        if wing_mode == "batched"
                        else "bit-identical loop slab discarded"
                    ),
                )


def _classify_remaining_regime_members(
    raw: Mapping[str, np.ndarray],
    ownership: dict[str, RawOwnership],
) -> None:
    for regime in REGIME_NAMES:
        for state_family in ("continuum_state", "line_state"):
            prefix = f"{regime}__{state_family}__"
            for raw_name in sorted(name for name in raw if name.startswith(prefix)):
                _mark(
                    ownership,
                    raw_name,
                    "intentionally_ephemeral",
                    "",
                    "upstream computed input state is never copied to a golden",
                )

        for grid_name in GRID_NAMES:
            ledger_prefix = f"{regime}__{grid_name}__ledger__"
            actual_fields = {
                name.removeprefix(ledger_prefix)
                for name in raw
                if name.startswith(ledger_prefix)
            }
            if actual_fields != EXPECTED_LEDGER_FIELDS:
                raise CompactAssemblyError(
                    f"ledger inventory changed: {regime}/{grid_name}"
                )
            for field_name in sorted(actual_fields):
                raw_name = f"{ledger_prefix}{field_name}"
                if raw_name in ownership:
                    continue
                _mark(
                    ownership,
                    raw_name,
                    "intentionally_ephemeral",
                    "",
                    "audit-only state, perturbation, or detailed wing intermediate",
                )


def _classify_remaining_invariants(
    raw: Mapping[str, np.ndarray],
    ownership: dict[str, RawOwnership],
) -> None:
    for grid_name in GRID_NAMES:
        prefix = f"invariant__{grid_name}__"
        actual_fields = {
            name.removeprefix(prefix) for name in raw if name.startswith(prefix)
        }
        if actual_fields != set(oracle_worker.ATOMIC_INVARIANT_FIELDS):
            raise CompactAssemblyError(
                f"invariant field inventory changed: {grid_name}"
            )
    for raw_name in sorted(name for name in raw if name.startswith("invariant__")):
        if raw_name in ownership:
            continue
        _mark(
            ownership,
            raw_name,
            "derived_digest_only",
            "",
            "validated invariant duplicate not owned by compact comparison output",
        )


def _complete_ownership(
    raw: Mapping[str, np.ndarray],
    ownership: dict[str, RawOwnership],
) -> tuple[RawOwnership, ...]:
    for raw_name in sorted(name for name in raw if name.startswith("record__")):
        if raw_name not in ownership:
            raise CompactAssemblyError(f"unclassified record member: {raw_name}")
    for raw_name in sorted(name for name in raw if name.startswith("catalog__")):
        if raw_name not in ownership:
            raise CompactAssemblyError(f"unclassified catalog member: {raw_name}")
    missing = sorted(set(raw) - set(ownership))
    extra = sorted(set(ownership) - set(raw))
    if missing or extra:
        raise CompactAssemblyError(
            f"raw ownership is incomplete; missing={missing}, extra={extra}"
        )
    ordered = tuple(ownership[name] for name in sorted(ownership))
    if any(item.disposition not in RAW_DISPOSITIONS for item in ordered):
        raise CompactAssemblyError("raw ownership contains an invalid disposition")
    return ordered


def _add_candidate_metadata(
    builder: _Builder,
    raw: Mapping[str, np.ndarray],
    ownership: tuple[RawOwnership, ...],
) -> None:
    builder.add(
        "meta__archive_kind",
        np.asarray("synthesis_one_line_comparison_candidate"),
    )
    builder.add(
        "meta__compact_schema_version",
        np.asarray(COMPACT_SCHEMA_VERSION, dtype=np.int64),
        unit="schema version",
    )
    builder.add(
        "meta__raw_capture_key_count",
        np.asarray(ACCEPTED_RAW_KEY_COUNT, dtype=np.int64),
        unit="count",
    )
    builder.add(
        "meta__assembler_sha256",
        np.asarray(sha256(ASSEMBLER_PATH)),
        unit="SHA-256 source identity",
    )
    builder.add(
        "meta__raw_ownership_digest",
        np.asarray(ownership_digest(ownership)),
        unit="SHA-256 logical ownership identity",
    )
    for disposition in sorted(RAW_DISPOSITIONS):
        builder.add(
            f"meta__raw_ownership_count__{disposition}",
            np.asarray(
                sum(item.disposition == disposition for item in ownership),
                dtype=np.int64,
            ),
            unit="count",
        )
    builder.add(
        "meta__raw_ownership_complete",
        np.asarray(True, dtype=np.bool_),
        unit="boolean",
    )
    builder.add(
        "meta__publication_authorized",
        np.asarray(False, dtype=np.bool_),
        unit="boolean",
    )
    builder.add(
        "meta__golden_publication_performed",
        np.asarray(False, dtype=np.bool_),
        unit="boolean",
    )
    builder.add(
        "meta__compact_key_count",
        np.asarray(0, dtype=np.int64),
        unit="count",
    )
    builder.add(
        "meta__compact_schema_digest",
        np.asarray("0" * 64),
        unit="SHA-256 schema identity",
    )
    builder.add(
        "meta__compact_payload_fingerprint",
        np.asarray("0" * 64),
        unit="SHA-256 payload identity",
    )

    builder.replace_scalar(
        "meta__compact_key_count",
        np.asarray(len(builder.arrays), dtype=np.int64),
    )
    builder.replace_scalar(
        "meta__compact_schema_digest",
        np.asarray(schema_digest(builder.arrays)),
    )
    builder.replace_scalar(
        "meta__compact_payload_fingerprint",
        np.asarray(payload_fingerprint(builder.arrays)),
    )


def assemble_compact_candidate(
    raw: Mapping[str, np.ndarray],
) -> CompactAssembly:
    """Project one accepted raw observation into one comparison-only mapping."""

    raw_copy = _validated_raw(raw)
    builder = _Builder()
    ownership: dict[str, RawOwnership] = {}

    _copy_metadata(raw_copy, builder, ownership)
    _add_catalog(raw_copy, builder, ownership)
    _add_invariants(raw_copy, builder, ownership)
    _add_canonical_products(raw_copy, builder, ownership)
    _add_coarse_evidence(raw_copy, builder, ownership)
    _classify_remaining_regime_members(raw_copy, ownership)
    _classify_remaining_invariants(raw_copy, ownership)
    ordered_ownership = _complete_ownership(raw_copy, ownership)
    _add_candidate_metadata(builder, raw_copy, ordered_ownership)
    arrays, specs = builder.finish()
    assembly = CompactAssembly(
        arrays=arrays,
        schema=specs,
        raw_ownership=ordered_ownership,
    )
    validate_compact_candidate(assembly)
    return assembly


def validate_compact_candidate(assembly: CompactAssembly) -> None:
    """Require the complete in-memory comparison-only semantic contract."""

    arrays = assembly.arrays
    if list(arrays) != sorted(arrays):
        raise CompactAssemblyError("compact mapping is not in lexical order")
    if len(arrays) != int(_scalar(arrays, "meta__compact_key_count")):
        raise CompactAssemblyError("compact key count does not recompute")
    if schema_digest(arrays) != _scalar(arrays, "meta__compact_schema_digest"):
        raise CompactAssemblyError("compact schema digest does not recompute")
    if payload_fingerprint(arrays) != _scalar(
        arrays, "meta__compact_payload_fingerprint"
    ):
        raise CompactAssemblyError("compact payload fingerprint does not recompute")
    if bool(_scalar(arrays, "meta__publication_authorized")):
        raise CompactAssemblyError("compact candidate claimed publication authority")
    if bool(_scalar(arrays, "meta__golden_publication_performed")):
        raise CompactAssemblyError("compact candidate claimed golden publication")
    if not bool(_scalar(arrays, "meta__raw_ownership_complete")):
        raise CompactAssemblyError("raw ownership is not complete")

    specs = {spec.name: spec for spec in assembly.schema}
    if set(specs) != set(arrays):
        raise CompactAssemblyError("schema description does not cover the mapping")
    for name, value in arrays.items():
        array = np.asarray(value)
        spec = specs[name]
        if array.dtype.hasobject or not array.flags.c_contiguous:
            raise CompactAssemblyError(f"invalid compact array storage: {name}")
        if (
            spec.dtype != array.dtype.str
            or spec.shape != array.shape
            or len(spec.axes) != array.ndim
        ):
            raise CompactAssemblyError(f"schema description drifted: {name}")

    if arrays["axis__wavelength_nm"].shape != (6000,):
        raise CompactAssemblyError("canonical wavelength axis changed")
    for name in ("opacity__gross_float32", "opacity__net_float32"):
        value = arrays[name]
        if value.shape != (4, 6, 6000) or value.dtype != np.float32:
            raise CompactAssemblyError(f"canonical opacity contract changed: {name}")
        if np.any(~np.isfinite(value)) or np.any(value < 0.0):
            raise CompactAssemblyError(f"canonical opacity is invalid: {name}")
    if not np.array_equal(
        arrays["activity__post_cutoff_mask"],
        oracle_worker.EXPECTED_ACTIVITY_MASK,
    ):
        raise CompactAssemblyError("canonical activity mask changed")
    if not np.array_equal(
        arrays["coarse__activity_mask"],
        oracle_worker.EXPECTED_ACTIVITY_MASK,
    ):
        raise CompactAssemblyError("coarse activity mask changed")

    expected_empty_names = {
        f"invariant__{name}"
        for name in (
            oracle_worker.AUTO_INVARIANT_FIELDS
            + oracle_worker.HELIUM_ARRAY_INVARIANT_FIELDS
        )
    }
    if not expected_empty_names.issubset(arrays):
        raise CompactAssemblyError("an exact empty invariant is missing")
    if any(arrays[name].shape != (0,) for name in expected_empty_names):
        raise CompactAssemblyError("an exact empty invariant is not empty")

    forbidden_name_fragments = (
        "__continuum_state__",
        "__line_state__",
        "__gross_loop_",
        "__net_loop_",
        "stimulated_emission_factor_float32",
    )
    if any(
        fragment in name for name in arrays for fragment in forbidden_name_fragments
    ):
        raise CompactAssemblyError("compact mapping contains an ephemeral raw member")
    if any(
        np.asarray(value).shape in {(6, 6, 139), (6, 400), (4, 6, 400)}
        for value in arrays.values()
    ):
        raise CompactAssemblyError("compact mapping duplicated input or coarse slabs")
    if any(
        1001 in np.asarray(value).shape or 2001 in np.asarray(value).shape
        for value in arrays.values()
    ):
        raise CompactAssemblyError("compact mapping duplicated a numerical table")
    large_slabs = [
        name
        for name, value in arrays.items()
        if np.asarray(value).shape == (4, 6, 6000)
    ]
    if large_slabs != ["opacity__gross_float32", "opacity__net_float32"]:
        raise CompactAssemblyError("compact mapping owns the wrong dense slabs")
    total_bytes = sum(np.asarray(value).nbytes for value in arrays.values())
    if total_bytes > COMPACT_SIZE_CEILING_BYTES:
        raise CompactAssemblyError(
            f"compact mapping is {total_bytes} bytes; ceiling is "
            f"{COMPACT_SIZE_CEILING_BYTES}"
        )

    if len(assembly.raw_ownership) != ACCEPTED_RAW_KEY_COUNT:
        raise CompactAssemblyError("raw ownership count changed")
    raw_names = [item.raw_name for item in assembly.raw_ownership]
    if raw_names != sorted(raw_names) or len(raw_names) != len(set(raw_names)):
        raise CompactAssemblyError("raw ownership is not unique and lexical")
    for item in assembly.raw_ownership:
        if item.disposition not in RAW_DISPOSITIONS:
            raise CompactAssemblyError("raw ownership disposition changed")
        if item.disposition == "final":
            target_root = item.target.split("[", 1)[0]
            if target_root not in arrays:
                raise CompactAssemblyError(
                    f"final raw owner targets no compact member: {item.raw_name}"
                )
    if ownership_digest(assembly.raw_ownership) != _scalar(
        arrays, "meta__raw_ownership_digest"
    ):
        raise CompactAssemblyError("raw ownership digest does not recompute")


def summarize_compact_assembly(assembly: CompactAssembly) -> dict[str, Any]:
    """Return a compact JSON-safe review summary without serialization."""

    validate_compact_candidate(assembly)
    arrays = assembly.arrays
    counts = {
        disposition: sum(
            item.disposition == disposition for item in assembly.raw_ownership
        )
        for disposition in sorted(RAW_DISPOSITIONS)
    }
    return {
        "candidate_key_count": len(arrays),
        "candidate_array_bytes": sum(
            np.asarray(value).nbytes for value in arrays.values()
        ),
        "candidate_schema_digest": str(arrays["meta__compact_schema_digest"]),
        "candidate_payload_fingerprint": str(
            arrays["meta__compact_payload_fingerprint"]
        ),
        "raw_ownership_digest": str(arrays["meta__raw_ownership_digest"]),
        "raw_ownership_counts": counts,
        "raw_ownership_complete": bool(arrays["meta__raw_ownership_complete"]),
        "gross_shape": list(arrays["opacity__gross_float32"].shape),
        "net_shape": list(arrays["opacity__net_float32"].shape),
        "publication_authorized": bool(arrays["meta__publication_authorized"]),
        "golden_publication_performed": bool(
            arrays["meta__golden_publication_performed"]
        ),
    }


__all__ = [
    "ACCEPTED_PLAN_REBIND_AUDIT_SHA256",
    "ACCEPTED_PLAN_REBIND_CANDIDATE_SHA256",
    "ACCEPTED_PLAN_SHA256",
    "ACCEPTED_RAW_FULL_FINGERPRINT",
    "ACCEPTED_RAW_KEY_COUNT",
    "ACCEPTED_RAW_PHYSICAL_FINGERPRINT",
    "ACCEPTED_RAW_SCHEMA_DIGEST",
    "ACCEPTED_WORKER_SHA256",
    "COMPACT_SIZE_CEILING_BYTES",
    "CompactAssembly",
    "CompactAssemblyError",
    "CompactMemberSpec",
    "RawOwnership",
    "assemble_compact_candidate",
    "member_sha256",
    "ownership_digest",
    "payload_fingerprint",
    "schema_digest",
    "summarize_compact_assembly",
    "validate_compact_candidate",
]
