"""Executable Chapter 15 workflow-composition checkpoints.

This module composes exact source, initializer, schema-v4, synthesis, and
physical-solver boundaries from the staged repository.  It executes one
compact public label-to-spectrum workflow and keeps the much longer
independently converged atmosphere route separate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Mapping
import warnings

import numpy as np

from book.chapter15_teaching import (
    AcceptanceRow,
    AtmosphereSummary,
    SpectrumSummary,
    StellarRequest,
    WorkflowBoundary,
    acceptance_rows,
    canonical_digest,
    canonical_json,
    request_digest,
    stellar_request_from_mapping,
    summarize_atmosphere,
    summarize_spectrum,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
CASE_REQUESTS_PATH = (
    REPOSITORY_ROOT / "data/fixtures/chapter15_case_requests.json"
)
ARTIFACT_MANIFEST_PATH = REPOSITORY_ROOT / "data/chapter15_artifacts.json"
DATA_MANIFEST_PATH = REPOSITORY_ROOT / "data/MANIFEST.json"
VERIFIED_SOLAR_DIRECTORY = (
    REPOSITORY_ROOT / "data/golden/payne_zero/chapter15"
)
VERIFIED_SOLAR_ACCEPTANCE_PATH = (
    VERIFIED_SOLAR_DIRECTORY / "chapter15_verified_solar_acceptance.json"
)
VERIFIED_SOLAR_ATMOSPHERE_PATH = (
    VERIFIED_SOLAR_DIRECTORY
    / "chapter15_verified_solar_atmosphere_cpu_float64.npz"
)
VERIFIED_SOLAR_SPECTRUM_PATH = (
    VERIFIED_SOLAR_DIRECTORY
    / "chapter15_verified_solar_spectrum_cpu_float64.npz"
)
CHAPTER10_ATMOSPHERE_FIXTURE = (
    REPOSITORY_ROOT / "data/fixtures/chapter05_continuum_states.npz"
)
EXPECTED_CASE_NAMES = (
    "hot_dwarf",
    "solar_dwarf",
    "low_gravity_giant",
    "cool_molecule_rich",
)
EXPECTED_RUNNER_SYMBOLS = (
    "TransferAccumulation",
    "IterationFinalization",
    "IterationRemap",
    "AtmosphereRunResult",
    "finalize_transfer_state",
    "remap_finalized_iteration_state",
    "finalize_remapped_iteration",
    "run_atmosphere_model",
)
INITIALIZER_EXPORTS_REQUIRED_BY_SYNTHESIS_API = (
    "CNO8_FAMILY",
    "FIVE_LABEL_FAMILY",
    "emulator_warm_start_model",
    "linear_elemental_abundances",
    "select_warm_start_family",
)
SOURCE_HASHES = {
    "src/payne_zero_synthesis/api.py": (
        "77718303c1e0052a520ece7fab277b3b1922c21d09b35a288596592d03310940"
    ),
    "src/payne_zero_synthesis/atmosphere.py": (
        "06b79770e4d9472093655022d53ee7fddf7cc6727206f34c0f60c57151e2cf9b"
    ),
    "src/payne_zero_atmosphere/warm_start.py": (
        "3a83af3d68be52a35bfc3f55f5912770661be8251cc28f28da3250b2e83e0ad3"
    ),
    "src/payne_zero_atmosphere/direct_abundance.py": (
        "ec65683eb344c4c3fd77340c084e780f58c6401e77c9f0d6db05ef6753131445"
    ),
}


@dataclass(frozen=True)
class PublicInterfaceCheckpoint:
    """Exact public type fields and signatures."""

    forward_timing_fields: tuple[str, ...]
    initialized_atmosphere_fields: tuple[str, ...]
    label_spectrum_fields: tuple[str, ...]
    spectrum_fields: tuple[str, ...]
    initialize_signature: str
    synthesize_from_labels_signature: str
    synthesize_signature: str
    validation_signature: str
    metadata_signature: str
    load_signature: str


@dataclass(frozen=True)
class DependencyCheckpoint:
    """Present and absent seams, checked from the staged source."""

    initializer_export_names: tuple[str, ...]
    missing_initializer_exports: tuple[str, ...]
    runner_required_symbols: tuple[str, ...]
    missing_runner_symbols: tuple[str, ...]
    solve_structured_atmosphere_staged: bool
    exact_source_catalogs_ready: bool
    exact_source_catalog_blocker: str | None
    exploratory_boundary: WorkflowBoundary
    verified_boundary: WorkflowBoundary


@dataclass(frozen=True)
class InitializerRouteCheckpoint:
    """Exact ordinary/CNO family routing and candidate identity."""

    name: str
    requested_labels: dict[str, object]
    request_sha256: str
    routed_family: str
    candidate_count: int
    first_initializer_query: dict[str, float] | None
    projection_used: bool
    warning_messages: tuple[str, ...]
    candidates_sha256: str
    request_unchanged: bool


@dataclass(frozen=True)
class DirectMixtureCheckpoint:
    """Exact 81-to-97 direct-abundance handoff without decoder execution."""

    public_atomic_number_count: int
    solver_slot_count: int
    sentinel_slot_count: int
    iron_abundance: float
    selected_public_values: dict[int, float]
    selected_solver_values: dict[int, float]
    sentinels_inherit_iron: bool
    centidex_lattice_exact: bool
    mixture_sha256: str
    release_gate_passed: bool
    exact_closure_required: bool


@dataclass(frozen=True)
class SafetyTypeCheckpoint:
    """Executed public exploratory workflow and immutable safety fields."""

    probe_only: bool
    initializer_family: str
    initialized_flags: tuple[bool, bool]
    label_spectrum_flags: tuple[bool, bool]
    wavelength_count: int
    minimum_normalized_flux: float
    initializer_seconds: float
    population_bridge_seconds: float
    synthesis_seconds: float
    saved_validated_names: tuple[str, ...]
    loaded_field_count: int
    saved_product_role: str
    saved_initializer_family: str
    metadata_converged: bool
    metadata_closure_required: bool
    eligible_as_verified_physical_product: bool
    mapping_arrays_equal: bool


@dataclass(frozen=True)
class RegimeCapstoneRow:
    """One compact integration row with both workflow boundaries visible."""

    name: str
    request_sha256: str
    routed_family: str
    projection_used: bool
    molecular_lines: bool
    atmosphere: AtmosphereSummary
    spectrum: SpectrumSummary
    exploratory_status: str
    verified_status: str
    exploratory_acceptance: tuple[AcceptanceRow, ...]
    verified_acceptance: tuple[AcceptanceRow, ...]


@dataclass(frozen=True)
class FourRegimeCapstone:
    """The same compact report for hot, solar, giant, and cool states."""

    case_names: tuple[str, ...]
    wavelength_nm: np.ndarray
    rows: tuple[RegimeCapstoneRow, ...]
    backend: str
    dtype: str
    source_fixture_sha256: str
    compact_fixture_only: bool


@dataclass(frozen=True)
class ReproducibilityCheckpoint:
    """Canonical identity excluding nondeterministic timings."""

    provenance: dict[str, object]
    canonical_json: str
    sha256: str
    repeated_sha256: str
    repeatable: bool
    timing_fields_excluded: bool


@dataclass(frozen=True)
class VerifiedSolarCheckpoint:
    """Published exact-solar evidence, with untested gates left explicit."""

    status: str
    run_roles: tuple[str, ...]
    run_count: int
    converged_on_pass: int
    deep_temperature_change: float
    deep_temperature_change_threshold: float
    atmosphere_archive_sha256: str
    atmosphere_payload_sha256: str
    atmosphere_array_count: int
    atmosphere_depth_count: int
    minimum_temperature_k: float
    maximum_temperature_k: float
    minimum_column_mass: float
    maximum_column_mass: float
    atmosphere_archives_byte_identical: bool
    spectrum_archive_sha256: str
    spectrum_payload_sha256: str
    wavelength_count: int
    wavelength_start_nm: float
    wavelength_stop_nm: float
    minimum_normalized_flux: float
    maximum_normalized_flux: float
    spectrum_physical_arrays_bitwise_identical: bool
    catalogs_required_to_rebuild: bool
    catalogs_required_to_validate: bool
    acceptance: tuple[AcceptanceRow, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))


def _data_manifest_entries() -> dict[str, dict[str, object]]:
    payload = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("payne_zero_commit") != PINNED_PAYNE_ZERO_COMMIT:
        raise RuntimeError("global data manifest pinned commit changed")
    return {
        str(entry["path"]): entry
        for entry in payload["entries"]
    }


def configure_chapter15_runtime() -> None:
    """Select staged source/local data and verify every pinned input."""

    from book.chapter06_runtime import configure_local_data_paths

    configure_local_data_paths()
    from book.chapter10_runtime import configure_chapter10_runtime

    configure_chapter10_runtime()
    # Chapter 10's disposable shared data view now exposes immutable
    # atmosphere tables/emulator links beside its generated synthesis table
    # view, so both packages resolve one PAYNE_ZERO_DATA_ROOT.
    source_root = (REPOSITORY_ROOT / "src").resolve()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith(("payne_zero_atmosphere", "payne_zero_synthesis")):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            continue
        if not Path(module_path).resolve().is_relative_to(source_root):
            raise RuntimeError(f"{name} resolved outside staged source: {module_path}")

    manifest = _manifest()
    if manifest["pinned_payne_zero_commit"] != PINNED_PAYNE_ZERO_COMMIT:
        raise RuntimeError("Chapter 15 pinned commit changed")
    for relative, record in manifest["artifacts"].items():
        path = REPOSITORY_ROOT / relative
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Chapter 15 artifact identity changed: {relative}")
    for relative, expected in SOURCE_HASHES.items():
        if _sha256(REPOSITORY_ROOT / relative) != expected:
            raise RuntimeError(f"Chapter 15 source identity changed: {relative}")


@lru_cache(maxsize=1)
def verified_solar_checkpoint() -> VerifiedSolarCheckpoint:
    """Load the accepted three-run solar comparison without rerunning it."""

    configure_chapter15_runtime()
    record = json.loads(
        VERIFIED_SOLAR_ACCEPTANCE_PATH.read_text(encoding="utf-8")
    )
    if record.get("schema") != "chapter15_verified_solar_acceptance_v1":
        raise RuntimeError("unsupported Chapter 15 solar acceptance schema")
    if record.get("payne_zero_commit") != PINNED_PAYNE_ZERO_COMMIT:
        raise RuntimeError("verified solar pinned commit changed")

    manifest_entries = _data_manifest_entries()
    published_paths = (
        VERIFIED_SOLAR_ATMOSPHERE_PATH,
        VERIFIED_SOLAR_SPECTRUM_PATH,
        VERIFIED_SOLAR_ACCEPTANCE_PATH,
    )
    for path in published_paths:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        if relative not in manifest_entries:
            raise RuntimeError(f"verified solar artifact is unregistered: {relative}")
        if _sha256(path) != manifest_entries[relative]["sha256"]:
            raise RuntimeError(f"verified solar artifact identity changed: {relative}")

    outputs = record["outputs"]
    atmosphere_output = outputs["atmosphere"]
    spectrum_output = outputs["spectrum"]
    if _sha256(VERIFIED_SOLAR_ATMOSPHERE_PATH) != atmosphere_output["sha256"]:
        raise RuntimeError("verified solar atmosphere disagrees with acceptance record")
    if _sha256(VERIFIED_SOLAR_SPECTRUM_PATH) != spectrum_output["sha256"]:
        raise RuntimeError("verified solar spectrum disagrees with acceptance record")

    with np.load(VERIFIED_SOLAR_ATMOSPHERE_PATH, allow_pickle=False) as archive:
        atmosphere = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(VERIFIED_SOLAR_SPECTRUM_PATH, allow_pickle=False) as archive:
        spectrum = {name: np.asarray(archive[name]) for name in archive.files}

    evidence = record["evidence"]
    structural = evidence["structural_convergence"]
    atmosphere_parity = evidence["atmosphere_array_parity"]
    spectrum_parity = evidence["spectrum_array_parity"]
    flux_accepted = evidence["declared_flux_error_threshold"]["accepted"]
    hydrostatic_accepted = evidence["hydrostatic_residual"]["accepted"]
    ratio_accepted = evidence["normalized_flux_ratio"]["accepted"]
    schema_valid = bool(
        len(atmosphere) == 27
        and np.asarray(atmosphere["atmosphere_schema_version"]).item() == 4
        and np.asarray(atmosphere["temperature"]).shape == (80,)
        and all(np.all(np.isfinite(values)) for values in atmosphere.values())
    )
    spectrum_valid = bool(
        ratio_accepted
        and set(spectrum)
        == {
            "wavelength_nm",
            "flux_total",
            "flux_continuum",
            "normalized_flux",
        }
        and all(np.all(np.isfinite(values)) for values in spectrum.values())
        and np.all(np.diff(spectrum["wavelength_nm"]) > 0.0)
    )
    parity_valid = bool(
        atmosphere_parity["accepted"] and spectrum_parity["accepted"]
    )
    run_roles = tuple(str(item["role"]) for item in record["inputs"])
    provenance_valid = bool(
        run_roles
        == ("staged", "staged_repeat", "pinned_read_only_oracle")
        and record["status"] == "accepted_exact_three_run_array_parity"
    )
    accepted = acceptance_rows(
        workflow="verified",
        initializer_assets_valid=True,
        seed_valid=None,
        structural_convergence=bool(structural["accepted"]),
        flux_accepted=flux_accepted,
        hydrostatic_accepted=hydrostatic_accepted,
        optical_grid_accepted=None,
        schema_valid=schema_valid,
        spectrum_valid=spectrum_valid,
        golden_spectral_parity=parity_valid,
        provenance_valid=provenance_valid,
        blockers=(
            evidence["declared_flux_error_threshold"]["reason"],
            evidence["hydrostatic_residual"]["reason"],
            "no separately retained standard-optical-depth-grid acceptance record",
        ),
    )
    catalog_policy = record["catalog_policy"]
    wavelength = spectrum["wavelength_nm"]
    normalized = spectrum["normalized_flux"]
    temperature = atmosphere["temperature"]
    column_mass = atmosphere["column_mass"]
    atmosphere_archive_hashes = {
        str(item["atmosphere_archive_sha256"]) for item in record["inputs"]
    }
    spectrum_payload_hashes = {
        str(item["spectrum_physical_payload_sha256"])
        for item in record["inputs"]
    }
    return VerifiedSolarCheckpoint(
        status=str(record["status"]),
        run_roles=run_roles,
        run_count=int(atmosphere_parity["run_count"]),
        converged_on_pass=int(structural["converged_on_pass"]),
        deep_temperature_change=float(
            structural["deep_layer_relative_temperature_change"]
        ),
        deep_temperature_change_threshold=float(
            structural["maximum_deep_layer_relative_temperature_change"]
        ),
        atmosphere_archive_sha256=str(
            record["inputs"][0]["atmosphere_archive_sha256"]
        ),
        atmosphere_payload_sha256=str(
            atmosphere_output["payload_sha256"]
        ),
        atmosphere_array_count=len(atmosphere),
        atmosphere_depth_count=int(temperature.size),
        minimum_temperature_k=float(np.min(temperature)),
        maximum_temperature_k=float(np.max(temperature)),
        minimum_column_mass=float(np.min(column_mass)),
        maximum_column_mass=float(np.max(column_mass)),
        atmosphere_archives_byte_identical=(
            len(atmosphere_archive_hashes) == 1
        ),
        spectrum_archive_sha256=str(spectrum_output["sha256"]),
        spectrum_payload_sha256=str(
            spectrum_output["physical_payload_sha256"]
        ),
        wavelength_count=int(wavelength.size),
        wavelength_start_nm=float(wavelength[0]),
        wavelength_stop_nm=float(wavelength[-1]),
        minimum_normalized_flux=float(np.min(normalized)),
        maximum_normalized_flux=float(np.max(normalized)),
        spectrum_physical_arrays_bitwise_identical=(
            len(spectrum_payload_hashes) == 1
        ),
        catalogs_required_to_rebuild=bool(
            catalog_policy["required_to_rebuild_this_record"]
        ),
        catalogs_required_to_validate=bool(
            catalog_policy["required_to_validate_published_goldens"]
        ),
        acceptance=accepted,
    )


@lru_cache(maxsize=1)
def load_case_requests() -> tuple[StellarRequest, ...]:
    """Load the four exact requests with strict schema and order."""

    configure_chapter15_runtime()
    payload = json.loads(CASE_REQUESTS_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != "chapter15_case_requests_v1":
        raise RuntimeError("unsupported Chapter 15 request schema")
    requests = tuple(
        stellar_request_from_mapping(values) for values in payload["cases"]
    )
    if tuple(request.name for request in requests) != EXPECTED_CASE_NAMES:
        raise RuntimeError("Chapter 15 case order changed")
    return requests


def public_interface_checkpoint() -> PublicInterfaceCheckpoint:
    """Inspect exact staged public interfaces without calling blocked seams."""

    configure_chapter15_runtime()
    from payne_zero_synthesis import (
        ForwardTimings,
        InitializedAtmosphere,
        LabelSpectrum,
        Spectrum,
        initialize_atmosphere_from_labels,
        load_atmosphere_npz,
        load_atmosphere_product_metadata,
        synthesize,
        synthesize_from_labels,
        validate_atmosphere_npz,
    )

    return PublicInterfaceCheckpoint(
        forward_timing_fields=tuple(field.name for field in fields(ForwardTimings)),
        initialized_atmosphere_fields=tuple(
            field.name for field in fields(InitializedAtmosphere)
        ),
        label_spectrum_fields=tuple(
            field.name for field in fields(LabelSpectrum)
        ),
        spectrum_fields=tuple(field.name for field in fields(Spectrum)),
        initialize_signature=str(inspect.signature(initialize_atmosphere_from_labels)),
        synthesize_from_labels_signature=str(
            inspect.signature(synthesize_from_labels)
        ),
        synthesize_signature=str(inspect.signature(synthesize)),
        validation_signature=str(inspect.signature(validate_atmosphere_npz)),
        metadata_signature=str(
            inspect.signature(load_atmosphere_product_metadata)
        ),
        load_signature=str(inspect.signature(load_atmosphere_npz)),
    )


def dependency_checkpoint() -> DependencyCheckpoint:
    """Preflight code and full-catalog dependencies without substituting data."""

    configure_chapter15_runtime()
    atmosphere_package = importlib.import_module("payne_zero_atmosphere")
    runner = importlib.import_module("payne_zero_atmosphere.runner")
    missing_exports = tuple(
        name
        for name in INITIALIZER_EXPORTS_REQUIRED_BY_SYNTHESIS_API
        if not hasattr(atmosphere_package, name)
    )
    missing_runner = tuple(
        name for name in EXPECTED_RUNNER_SYMBOLS if not hasattr(runner, name)
    )
    cli_path = REPOSITORY_ROOT / "src/payne_zero_atmosphere/cli.py"
    solve_staged = bool(
        cli_path.is_file()
        and hasattr(atmosphere_package, "solve_structured_atmosphere")
    )
    catalog_blocker = None
    try:
        from payne_zero_atmosphere.source_catalogs import (
            atmosphere_source_catalog_paths,
        )

        atmosphere_source_catalog_paths()
        catalogs_ready = True
    except (FileNotFoundError, RuntimeError):
        catalogs_ready = False
        catalog_blocker = (
            "exact_catalog_data_unavailable: the full checksum-verified "
            "atmosphere source-catalog set is not installed"
        )
    exploratory_available = not missing_exports
    verified_available = not missing_runner and solve_staged and catalogs_ready
    return DependencyCheckpoint(
        initializer_export_names=INITIALIZER_EXPORTS_REQUIRED_BY_SYNTHESIS_API,
        missing_initializer_exports=missing_exports,
        runner_required_symbols=EXPECTED_RUNNER_SYMBOLS,
        missing_runner_symbols=missing_runner,
        solve_structured_atmosphere_staged=solve_staged,
        exact_source_catalogs_ready=catalogs_ready,
        exact_source_catalog_blocker=catalog_blocker,
        exploratory_boundary=WorkflowBoundary(
            mode="exploratory",
            entrypoint="initialize_atmosphere_from_labels / synthesize_from_labels",
            returned_type="InitializedAtmosphere / LabelSpectrum",
            available=exploratory_available,
            blocker=(
                None
                if exploratory_available
                else "initializer_export_seam_blocked: "
                + ", ".join(missing_exports)
            ),
        ),
        verified_boundary=WorkflowBoundary(
            mode="verified",
            entrypoint="solve_structured_atmosphere -> validate/load -> synthesize",
            returned_type="Path -> tuple[str, ...] / mapping -> Spectrum",
            available=verified_available,
            blocker=(
                None
                if verified_available
                else (
                    "exact_runner_seam_blocked: "
                    + ", ".join(
                        (
                            *missing_runner,
                            *(
                                ()
                                if solve_staged
                                else ("solve_structured_atmosphere",)
                            ),
                        )
                    )
                    if missing_runner or not solve_staged
                    else catalog_blocker
                )
            ),
        ),
    )


@lru_cache(maxsize=1)
def initializer_route_checkpoints() -> tuple[InitializerRouteCheckpoint, ...]:
    """Run exact family routing and deterministic initializer-query selection."""

    configure_chapter15_runtime()
    from payne_zero_atmosphere.warm_start import (
        deterministic_initializer_labels,
        select_warm_start_family,
    )

    results = []
    for request in load_case_requests():
        requested_before = request.canonical_payload()
        routed = select_warm_start_family(
            carbon_enhancement=request.c_over_m,
            nitrogen_enhancement=request.n_over_m,
            oxygen_enhancement=request.o_over_m,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            candidates = deterministic_initializer_labels(
                effective_temperature=request.effective_temperature,
                log_surface_gravity=request.log_surface_gravity,
                metallicity=request.metallicity,
                alpha_enhancement=request.alpha_enhancement,
                microturbulence_km_s=request.microturbulence_km_s,
                carbon_enhancement=request.c_over_m,
                nitrogen_enhancement=request.n_over_m,
                oxygen_enhancement=request.o_over_m,
                max_trials=2,
                seed=20260713,
                jitter_scale=0.01,
                device="cpu",
            )
        candidate_payload = {
            "name": request.name,
            "candidates": [
                None
                if candidate is None
                else {
                    key: float(value)
                    for key, value in sorted(candidate.items())
                }
                for candidate in candidates
            ],
        }
        results.append(
            InitializerRouteCheckpoint(
                name=request.name,
                requested_labels=request.label_payload(),
                request_sha256=request_digest(request),
                routed_family=routed,
                candidate_count=len(candidates),
                first_initializer_query=(
                    None if candidates[0] is None else dict(candidates[0])
                ),
                projection_used=candidates[0] is not None,
                warning_messages=tuple(str(item.message) for item in caught),
                candidates_sha256=canonical_digest(candidate_payload),
                request_unchanged=(
                    request.canonical_payload() == requested_before
                ),
            )
        )
    return tuple(results)


@lru_cache(maxsize=1)
def direct_mixture_checkpoint() -> DirectMixtureCheckpoint:
    """Run the strict exact direct-mixture completion and identity path."""

    configure_chapter15_runtime()
    from payne_zero_atmosphere.direct_abundance import (
        DIRECT_XH_ATOMIC_NUMBERS,
        DIRECT_XH_SENTINEL_ATOMIC_NUMBERS,
        complete_direct_abundance_vector,
        direct_abundance_mixture_sha256,
    )

    public_values = {
        int(atomic_number): -0.2
        for atomic_number in DIRECT_XH_ATOMIC_NUMBERS
    }
    public_values.update({6: -0.1, 12: 0.0, 26: -0.2})
    vector = complete_direct_abundance_vector(public_values)
    selected = (6, 12, 26)
    direct_manifest = json.loads(
        (
            REPOSITORY_ROOT
            / "data/static/atmosphere_emulator/direct_abundance/manifest.json"
        ).read_text(encoding="utf-8")
    )
    return DirectMixtureCheckpoint(
        public_atomic_number_count=len(DIRECT_XH_ATOMIC_NUMBERS),
        solver_slot_count=int(vector.size),
        sentinel_slot_count=len(DIRECT_XH_SENTINEL_ATOMIC_NUMBERS),
        iron_abundance=float(vector[26 - 3]),
        selected_public_values={
            atomic_number: public_values[atomic_number] for atomic_number in selected
        },
        selected_solver_values={
            atomic_number: float(vector[atomic_number - 3])
            for atomic_number in selected
        },
        sentinels_inherit_iron=bool(
            all(
                vector[atomic_number - 3] == vector[26 - 3]
                for atomic_number in DIRECT_XH_SENTINEL_ATOMIC_NUMBERS
            )
        ),
        centidex_lattice_exact=bool(
            np.allclose(
                vector / 0.01,
                np.rint(vector / 0.01),
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        mixture_sha256=direct_abundance_mixture_sha256(vector),
        release_gate_passed=bool(
            direct_manifest["release_gate"]["passed"]
        ),
        exact_closure_required=True,
    )


@lru_cache(maxsize=1)
def safety_type_checkpoint() -> SafetyTypeCheckpoint:
    """Execute and persist one exact compact label-to-spectrum workflow."""

    configure_chapter15_runtime()
    from payne_zero_synthesis import (
        initialize_atmosphere_from_labels,
        load_atmosphere_npz,
        load_atmosphere_product_metadata,
        synthesize_from_labels,
        validate_atmosphere_npz,
    )

    request = load_case_requests()[1]
    labels = request.label_payload()
    common = {
        "effective_temperature": request.effective_temperature,
        "log_surface_gravity": request.log_surface_gravity,
        "metallicity": request.metallicity,
        "alpha_enhancement": request.alpha_enhancement,
        "microturbulence_km_s": request.microturbulence_km_s,
        "c_over_m": request.c_over_m,
        "n_over_m": request.n_over_m,
        "o_over_m": request.o_over_m,
        "initializer_family": request.initializer_family,
        "molecular_lines": request.molecular_lines,
        "device": "cpu",
        "dtype": "float64",
    }
    initialized = initialize_atmosphere_from_labels(**common)
    label_spectrum = synthesize_from_labels(
        **common,
        wavelength_start_nm=498.95,
        wavelength_end_nm=499.15,
        resolution=20_000.0,
    )
    source = initialized.structured_atmosphere
    with tempfile.TemporaryDirectory(prefix="chapter15-safety-type-") as directory:
        path = Path(directory) / "initializer_probe.npz"
        saved_names = initialized.save_npz(path)
        validated_names = validate_atmosphere_npz(path)
        metadata = load_atmosphere_product_metadata(path)
        loaded = load_atmosphere_npz(path)
    arrays_equal = all(
        np.array_equal(np.asarray(source[name]), np.asarray(loaded[name]))
        for name in validated_names
    )
    return SafetyTypeCheckpoint(
        probe_only=False,
        initializer_family=initialized.initializer_family,
        initialized_flags=(
            initialized.atmosphere_converged,
            initialized.atmosphere_closure_required,
        ),
        label_spectrum_flags=(
            label_spectrum.atmosphere_converged,
            label_spectrum.atmosphere_closure_required,
        ),
        wavelength_count=int(label_spectrum.wavelength_nm.size),
        minimum_normalized_flux=float(np.min(label_spectrum.normalized_flux)),
        initializer_seconds=float(label_spectrum.timings.initializer_seconds),
        population_bridge_seconds=float(
            label_spectrum.timings.population_bridge_seconds
        ),
        synthesis_seconds=float(label_spectrum.timings.synthesis_seconds),
        saved_validated_names=tuple(saved_names),
        loaded_field_count=len(loaded),
        saved_product_role=str(metadata["atmosphere_product_role"]),
        saved_initializer_family=str(metadata["initializer_family"]),
        metadata_converged=bool(metadata["atmosphere_converged"]),
        metadata_closure_required=bool(
            metadata["atmosphere_closure_required"]
        ),
        eligible_as_verified_physical_product=bool(
            metadata["atmosphere_product_role"]
            != "learned_initializer_prediction"
            and metadata["atmosphere_converged"]
            and not metadata["atmosphere_closure_required"]
        ),
        mapping_arrays_equal=arrays_equal,
    )


def _tolerance_profile() -> Mapping[str, float]:
    return _manifest()["tolerance_profile"]


@lru_cache(maxsize=1)
def four_regime_capstone() -> FourRegimeCapstone:
    """Run exact compact schema and synthesis checks for all four regimes."""

    configure_chapter15_runtime()
    from book.chapter10_runtime import (
        REGIMES,
        load_regime_atmosphere,
        public_spectrum,
    )
    from payne_zero_synthesis.atmosphere import REQUIRED_ATMOSPHERE_ARRAYS

    dependencies = dependency_checkpoint()
    routes = {
        checkpoint.name: checkpoint
        for checkpoint in initializer_route_checkpoints()
    }
    profile = _tolerance_profile()
    rows = []
    wavelength = None
    for request in load_case_requests():
        atmosphere_mapping = load_regime_atmosphere(request.name)
        atmosphere = summarize_atmosphere(
            atmosphere_mapping,
            required_fields=REQUIRED_ATMOSPHERE_ARRAYS,
        )
        public = public_spectrum(request.name)
        spectrum = summarize_spectrum(
            wavelength_nm=public.wavelength_nm,
            flux_total=public.flux_total,
            flux_continuum=public.flux_continuum,
            normalized_flux=public.normalized_flux,
            ratio_rtol=profile["normalized_flux_ratio_rtol"],
            ratio_atol=profile["normalized_flux_ratio_atol"],
        )
        if wavelength is None:
            wavelength = np.asarray(public.wavelength_nm).copy()
        elif not np.array_equal(wavelength, public.wavelength_nm):
            raise RuntimeError("same-window capstone spectra use different grids")
        route = routes[request.name]
        exploratory_blockers = (
            ()
            if dependencies.exploratory_boundary.available
            else (str(dependencies.exploratory_boundary.blocker),)
        )
        verified_blockers = (
            ()
            if dependencies.verified_boundary.available
            else (str(dependencies.verified_boundary.blocker),)
        )
        schema_valid = bool(
            not atmosphere.missing_fields
            and atmosphere.all_depth_axes_match
            and atmosphere.all_public_arrays_float64
            and atmosphere.finite
            and atmosphere.column_mass_positive
            and atmosphere.column_mass_strictly_increasing
            and atmosphere.positive_thermodynamic_columns
        )
        spectrum_valid = bool(
            spectrum.wavelength_strictly_increasing
            and spectrum.arrays_aligned
            and spectrum.finite
            and spectrum.continuum_nonzero
            and spectrum.normalized_ratio_matches
        )
        rows.append(
            RegimeCapstoneRow(
                name=request.name,
                request_sha256=route.request_sha256,
                routed_family=route.routed_family,
                projection_used=route.projection_used,
                molecular_lines=request.molecular_lines,
                atmosphere=atmosphere,
                spectrum=spectrum,
                exploratory_status=(
                    "exact_label_workflow_available"
                    if dependencies.exploratory_boundary.available
                    else "compact_schema_fixture_synthesis_only; "
                    + str(dependencies.exploratory_boundary.blocker)
                ),
                verified_status=(
                    "exact_verified_workflow_available_not_run"
                    if dependencies.verified_boundary.available
                    else str(dependencies.verified_boundary.blocker)
                ),
                exploratory_acceptance=acceptance_rows(
                    workflow="exploratory",
                    initializer_assets_valid=True,
                    seed_valid=None,
                    structural_convergence=None,
                    flux_accepted=None,
                    hydrostatic_accepted=None,
                    optical_grid_accepted=None,
                    schema_valid=schema_valid,
                    spectrum_valid=spectrum_valid,
                    golden_spectral_parity=None,
                    provenance_valid=True,
                    blockers=exploratory_blockers,
                ),
                verified_acceptance=acceptance_rows(
                    workflow="verified",
                    initializer_assets_valid=True,
                    seed_valid=None,
                    structural_convergence=None,
                    flux_accepted=None,
                    hydrostatic_accepted=None,
                    optical_grid_accepted=None,
                    schema_valid=None,
                    spectrum_valid=None,
                    golden_spectral_parity=None,
                    provenance_valid=True,
                    blockers=verified_blockers,
                ),
            )
        )
    if tuple(REGIMES) != EXPECTED_CASE_NAMES:
        raise RuntimeError("Chapter 10 and Chapter 15 regime order diverged")
    return FourRegimeCapstone(
        case_names=EXPECTED_CASE_NAMES,
        wavelength_nm=np.asarray(wavelength, dtype=np.float64),
        rows=tuple(rows),
        backend="cpu",
        dtype="torch.float64",
        source_fixture_sha256=_sha256(CHAPTER10_ATMOSPHERE_FIXTURE),
        compact_fixture_only=True,
    )


@lru_cache(maxsize=1)
def reproducibility_checkpoint() -> ReproducibilityCheckpoint:
    """Build a stable complete identity and prove timing exclusion."""

    configure_chapter15_runtime()
    import torch

    capstone = four_regime_capstone()
    dependencies = dependency_checkpoint()
    artifacts = {
        relative: record["sha256"]
        for relative, record in sorted(_manifest()["artifacts"].items())
    }
    provenance: dict[str, object] = {
        "pinned_payne_zero_commit": PINNED_PAYNE_ZERO_COMMIT,
        "requests": {
            request.name: request_digest(request)
            for request in load_case_requests()
        },
        "artifacts": artifacts,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "backend": capstone.backend,
            "dtype": capstone.dtype,
            "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS", "unspecified"),
        },
        "synthesis": {
            "wavelength_start_nm": 468.6,
            "wavelength_end_nm": 656.6,
            "resolution": 20_000.0,
            "catalog_scope": "Chapter 10 compact manifest-owned source view",
            "cache_policy": "process invariant cache enabled unless explicitly disabled",
        },
        "availability": {
            "exploratory_available": (
                dependencies.exploratory_boundary.available
            ),
            "exploratory_blocker": dependencies.exploratory_boundary.blocker,
            "verified_available": dependencies.verified_boundary.available,
            "verified_blocker": dependencies.verified_boundary.blocker,
            "compact_fixture_only": capstone.compact_fixture_only,
        },
        "limitations": (
            "LTE",
            "one-dimensional static atmosphere",
            "local mixing-length convection",
            "fixed-thread atmosphere reproducibility target",
            "compact synthesis catalogs are not publication-grade coverage",
            "standard synthesis runtime omits the separate H2O compiler",
        ),
    }
    encoded = canonical_json(provenance)
    digest = canonical_digest(provenance)
    repeated = canonical_digest(json.loads(encoded))
    return ReproducibilityCheckpoint(
        provenance=provenance,
        canonical_json=encoded,
        sha256=digest,
        repeated_sha256=repeated,
        repeatable=digest == repeated,
        timing_fields_excluded=all(
            key not in encoded
            for key in (
                "initializer_seconds",
                "population_bridge_seconds",
                "synthesis_seconds",
                "total_seconds",
            )
        ),
    )
