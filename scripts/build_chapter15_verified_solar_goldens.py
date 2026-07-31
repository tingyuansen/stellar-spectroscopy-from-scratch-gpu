#!/usr/bin/env python3
"""Publish Chapter 15 solar goldens only after exact three-run parity.

The builder compares a staged run, an independent staged repeat, and a run
from the pinned read-only Payne Zero checkout.  Array identity is bitwise:
shape, dtype, and C-order payload bytes must agree for every member.  The
wall-clock ``seconds`` member is deliberately excluded from spectral identity
and is never written to the golden spectrum.

The resulting atmosphere is comparison-only.  Its existence records exact
solver convergence and reproducibility, but does not assert a flux-error or
hydrostatic-residual acceptance test that was not performed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Mapping

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.deterministic_npz import write_npz  # noqa: E402


PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"

DEFAULT_STAGED_DIRECTORY = REPOSITORY_ROOT / "audit/full_solve/solar_role_safe"
DEFAULT_REPEAT_DIRECTORY = REPOSITORY_ROOT / "audit/full_solve/solar_role_safe_repeat"
DEFAULT_PINNED_DIRECTORY = REPOSITORY_ROOT / "audit/full_solve/solar_pinned_oracle"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "data/golden/payne_zero/chapter15"

ATMOSPHERE_INPUT_NAME = "payne_zero_structured_atmosphere.npz"
SPECTRUM_INPUT_NAME = "spectrum_49895_49915.npz"
GOLDEN_ATMOSPHERE_NAME = "chapter15_verified_solar_atmosphere_cpu_float64.npz"
GOLDEN_SPECTRUM_NAME = "chapter15_verified_solar_spectrum_cpu_float64.npz"
ACCEPTANCE_NAME = "chapter15_verified_solar_acceptance.json"
BUILDER_PATH = "scripts/build_chapter15_verified_solar_goldens.py"

ATMOSPHERE_ARRAYS = (
    "temperature",
    "gas_pressure",
    "electron_density",
    "mass_density",
    "column_mass",
    "partition_normalized_populations",
    "ion_stage_populations",
    "fractional_doppler_widths",
    "hydrogen_neutral_population",
    "helium_neutral_population",
    "helium_singly_ionized_population",
    "molecular_hydrogen_population",
    "hydrogen_partition_normalized_ion_stage_populations",
    "carbon_partition_normalized_ion_stage_populations",
    "magnesium_neutral_partition_normalized_population",
    "aluminum_neutral_partition_normalized_population",
    "silicon_neutral_partition_normalized_population",
    "iron_neutral_partition_normalized_population",
    "hc_over_kt",
    "microturbulence",
    "elemental_abundances",
    "signed_continuum_edge_frequency_hz",
    "continuum_edge_wavelength_nm",
    "continuum_edge_midpoint_wavelength_nm",
    "continuum_edge_interval_width_squared_over_two_nm2",
    "hydrogen_ionized_population",
    "atmosphere_schema_version",
)
SPECTRUM_PHYSICAL_ARRAYS = (
    "wavelength_nm",
    "flux_total",
    "flux_continuum",
    "normalized_flux",
)
SPECTRUM_TIMING_ARRAYS = ("seconds",)

ATMOSPHERE_UNITS = {
    "temperature": "K",
    "gas_pressure": "dyne cm^-2",
    "electron_density": "cm^-3",
    "mass_density": "g cm^-3",
    "column_mass": "g cm^-2",
    "partition_normalized_populations": "cm^-3 per partition function",
    "ion_stage_populations": "cm^-3",
    "fractional_doppler_widths": "dimensionless v/c",
    "hydrogen_neutral_population": "cm^-3",
    "helium_neutral_population": "cm^-3",
    "helium_singly_ionized_population": "cm^-3",
    "molecular_hydrogen_population": "cm^-3",
    "hydrogen_partition_normalized_ion_stage_populations": (
        "cm^-3 per partition function"
    ),
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
    "hc_over_kt": "cm",
    "microturbulence": "cm s^-1",
    "elemental_abundances": "relative number abundance",
    "signed_continuum_edge_frequency_hz": "Hz",
    "continuum_edge_wavelength_nm": "nm",
    "continuum_edge_midpoint_wavelength_nm": "nm",
    "continuum_edge_interval_width_squared_over_two_nm2": "nm^2",
    "hydrogen_ionized_population": "cm^-3",
    "atmosphere_schema_version": "schema version",
}
SPECTRUM_UNITS = {
    "wavelength_nm": "nm",
    "flux_total": "source-native emergent flux density",
    "flux_continuum": "source-native emergent flux density",
    "normalized_flux": "dimensionless flux ratio",
}
ATMOSPHERE_SHAPES = {
    "temperature": (80,),
    "gas_pressure": (80,),
    "electron_density": (80,),
    "mass_density": (80,),
    "column_mass": (80,),
    "partition_normalized_populations": (80, 6, 139),
    "ion_stage_populations": (80, 6, 139),
    "fractional_doppler_widths": (80, 6, 139),
    "hydrogen_neutral_population": (80,),
    "helium_neutral_population": (80,),
    "helium_singly_ionized_population": (80,),
    "molecular_hydrogen_population": (80,),
    "hydrogen_partition_normalized_ion_stage_populations": (80, 2),
    "carbon_partition_normalized_ion_stage_populations": (80, 2),
    "magnesium_neutral_partition_normalized_population": (80,),
    "aluminum_neutral_partition_normalized_population": (80,),
    "silicon_neutral_partition_normalized_population": (80,),
    "iron_neutral_partition_normalized_population": (80,),
    "hc_over_kt": (80,),
    "microturbulence": (80,),
    "elemental_abundances": (99,),
    "signed_continuum_edge_frequency_hz": (341,),
    "continuum_edge_wavelength_nm": (341,),
    "continuum_edge_midpoint_wavelength_nm": (340,),
    "continuum_edge_interval_width_squared_over_two_nm2": (340,),
    "hydrogen_ionized_population": (80,),
    "atmosphere_schema_version": (1,),
}


class ParityError(RuntimeError):
    """Raised when prepared runs do not satisfy the publication contract."""


@dataclass(frozen=True)
class RunArtifacts:
    """The atmosphere and spectrum produced by one exact run."""

    role: str
    atmosphere: Path
    spectrum: Path


@dataclass(frozen=True)
class PublicationGate:
    """Read-only publication readiness result."""

    ready: bool
    status: str
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class PublicationResult:
    """Paths and record returned by a successful publication."""

    atmosphere: Path
    spectrum: Path
    acceptance: Path
    record: dict[str, object]


def sha256(path: Path) -> str:
    """Return the streaming SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _payload_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash member names, dtype/shape contracts, and array payload bytes."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        values = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(values.dtype.str.encode("ascii") + b"\0")
        digest.update(
            json.dumps(list(values.shape), separators=(",", ":")).encode("ascii")
            + b"\0"
        )
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validate_member_set(
    *,
    path: Path,
    arrays: Mapping[str, np.ndarray],
    required: Iterable[str],
    allowed_extra: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed_set = required_set | set(allowed_extra)
    missing = sorted(required_set - set(arrays))
    unexpected = sorted(set(arrays) - allowed_set)
    if missing or unexpected:
        raise ParityError(
            f"{path}: member contract changed; missing={missing}, "
            f"unexpected={unexpected}"
        )


def _compare_bitwise(
    *,
    kind: str,
    reference_role: str,
    reference: Mapping[str, np.ndarray],
    candidate_role: str,
    candidate: Mapping[str, np.ndarray],
    names: Iterable[str],
) -> None:
    for name in names:
        left = np.asarray(reference[name])
        right = np.asarray(candidate[name])
        if left.shape != right.shape:
            raise ParityError(
                f"{kind}:{name}: {reference_role} shape {left.shape} != "
                f"{candidate_role} shape {right.shape}"
            )
        if left.dtype.str != right.dtype.str:
            raise ParityError(
                f"{kind}:{name}: {reference_role} dtype {left.dtype.str} != "
                f"{candidate_role} dtype {right.dtype.str}"
            )
        if np.ascontiguousarray(left).tobytes(order="C") != np.ascontiguousarray(
            right
        ).tobytes(order="C"):
            raise ParityError(
                f"{kind}:{name}: {reference_role} and {candidate_role} "
                "payload bytes differ"
            )


def _validate_atmosphere(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    _validate_member_set(path=path, arrays=arrays, required=ATMOSPHERE_ARRAYS)
    for name, values in arrays.items():
        values = np.asarray(values)
        if values.shape != ATMOSPHERE_SHAPES[name]:
            raise ParityError(
                f"{path}:{name}: shape {values.shape} != " f"{ATMOSPHERE_SHAPES[name]}"
            )
        if name == "atmosphere_schema_version":
            if values.dtype != np.dtype(np.int32) or int(values[0]) != 4:
                raise ParityError(f"{path}: atmosphere schema is not v4")
            continue
        if values.dtype != np.dtype(np.float64):
            raise ParityError(f"{path}:{name}: expected float64")
        if not np.all(np.isfinite(values)):
            raise ParityError(f"{path}:{name}: non-finite values")
    column_mass = np.asarray(arrays["column_mass"], dtype=np.float64)
    if not np.all(column_mass > 0.0) or not np.all(np.diff(column_mass) > 0.0):
        raise ParityError(f"{path}: column mass must be positive and increasing")
    for name in ("temperature", "gas_pressure", "electron_density", "mass_density"):
        if not np.all(np.asarray(arrays[name]) > 0.0):
            raise ParityError(f"{path}:{name}: expected positive values")


def _validate_spectrum(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    _validate_member_set(
        path=path,
        arrays=arrays,
        required=SPECTRUM_PHYSICAL_ARRAYS,
        allowed_extra=SPECTRUM_TIMING_ARRAYS,
    )
    wavelength = np.asarray(arrays["wavelength_nm"])
    if wavelength.dtype != np.dtype(np.float64) or wavelength.shape != (8,):
        raise ParityError(f"{path}: expected eight float64 wavelengths")
    if not np.all(np.diff(wavelength) > 0.0):
        raise ParityError(f"{path}: wavelengths are not strictly increasing")
    for name in SPECTRUM_PHYSICAL_ARRAYS:
        values = np.asarray(arrays[name])
        if values.dtype != np.dtype(np.float64) or values.shape != wavelength.shape:
            raise ParityError(f"{path}:{name}: spectral array contract changed")
        if not np.all(np.isfinite(values)):
            raise ParityError(f"{path}:{name}: non-finite values")
    continuum = np.asarray(arrays["flux_continuum"])
    if np.any(continuum == 0.0):
        raise ParityError(f"{path}: zero continuum flux")
    ratio = np.asarray(arrays["flux_total"]) / continuum
    if not np.allclose(
        np.asarray(arrays["normalized_flux"]),
        ratio,
        rtol=3.0e-7,
        atol=0.0,
    ):
        raise ParityError(f"{path}: normalized flux is not total/continuum")
    if "seconds" in arrays:
        seconds = np.asarray(arrays["seconds"])
        if (
            seconds.dtype != np.dtype(np.float64)
            or seconds.shape != (1,)
            or not np.isfinite(seconds[0])
            or seconds[0] < 0.0
        ):
            raise ParityError(f"{path}: invalid non-identity timing member")


def publication_gate(runs: Iterable[RunArtifacts]) -> PublicationGate:
    """Report missing prepared inputs without creating any golden files."""

    missing: list[str] = []
    for run in runs:
        for kind, path in (
            ("atmosphere", run.atmosphere),
            ("spectrum", run.spectrum),
        ):
            if not path.is_file():
                missing.append(f"{run.role}:{kind}:{path}")
    if missing:
        return PublicationGate(
            ready=False,
            status="blocked_prepared_run_incomplete",
            blockers=tuple(missing),
        )
    return PublicationGate(
        ready=True,
        status="ready_for_exact_three_run_comparison",
        blockers=(),
    )


def _input_record(
    run: RunArtifacts,
    atmosphere: Mapping[str, np.ndarray],
    spectrum: Mapping[str, np.ndarray],
) -> dict[str, object]:
    timing = None
    if "seconds" in spectrum:
        timing = float(np.asarray(spectrum["seconds"])[0])
    physical_spectrum = {
        name: np.asarray(spectrum[name]) for name in SPECTRUM_PHYSICAL_ARRAYS
    }
    return {
        "role": run.role,
        "atmosphere_path": _display_path(run.atmosphere),
        "atmosphere_archive_sha256": sha256(run.atmosphere),
        "atmosphere_payload_sha256": _payload_digest(atmosphere),
        "spectrum_path": _display_path(run.spectrum),
        "spectrum_archive_sha256": sha256(run.spectrum),
        "spectrum_physical_payload_sha256": _payload_digest(physical_spectrum),
        "synthesis_seconds_observed_not_identity": timing,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(path)


def _array_records(
    arrays: Mapping[str, np.ndarray], units: Mapping[str, str]
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "shape": list(np.asarray(values).shape),
            "dtype": str(np.asarray(values).dtype),
            "unit": units[name],
            "sha256": _array_sha256(np.asarray(values)),
        }
        for name, values in sorted(arrays.items())
    }


def _acceptance_record(
    *,
    runs: tuple[RunArtifacts, ...],
    atmosphere_inputs: tuple[Mapping[str, np.ndarray], ...],
    spectrum_inputs: tuple[Mapping[str, np.ndarray], ...],
    golden_atmosphere: Path,
    golden_spectrum: Path,
) -> dict[str, object]:
    atmosphere_arrays = {
        name: np.asarray(atmosphere_inputs[0][name]) for name in ATMOSPHERE_ARRAYS
    }
    spectrum_arrays = {
        name: np.asarray(spectrum_inputs[0][name]) for name in SPECTRUM_PHYSICAL_ARRAYS
    }
    return {
        "schema": "chapter15_verified_solar_acceptance_v1",
        "status": "accepted_exact_three_run_array_parity",
        "payne_zero_commit": PINNED_PAYNE_ZERO_COMMIT,
        "builder": BUILDER_PATH,
        "case": {
            "effective_temperature_k": 5777.0,
            "log_surface_gravity_cgs": 4.44,
            "metallicity_dex": 0.0,
            "alpha_over_metal_dex": 0.0,
            "explicit_cno_labels": False,
            "microturbulence_km_s": 2.0,
            "initializer_family": "five_label",
            "initializer_selection": "auto",
            "wavelength_start_nm": 498.95,
            "wavelength_stop_nm": 499.15,
            "r_grid": 20000.0,
            "backend": "cpu",
            "dtype": "float64",
            "molecular_lines": True,
            "atmosphere_molecules": True,
            "convection": True,
            "iterations_per_trial": 15,
            "maximum_trials": 2,
            "random_seed": 20260713,
            "trial_jitter_fraction": 0.01,
            "minimum_iterations_before_convergence": 3,
            "required_consecutive_converged_iterations": 1,
        },
        "catalog_policy": {
            "full_source_catalogs_are_optional_textbook_data": True,
            "approximate_full_catalog_size_gb": 6.8,
            "required_to_rebuild_this_record": True,
            "required_to_validate_published_goldens": False,
            "bundled_in_chapter15_goldens": False,
        },
        "timing_policy": {
            "excluded_from_identity": list(SPECTRUM_TIMING_ARRAYS),
            "reason": (
                "wall-clock timings vary with runtime state and are not "
                "physical output arrays"
            ),
        },
        "evidence": {
            "structural_convergence": {
                "accepted": True,
                "publication_gate": (
                    "the exact runner writes the product structured-atmosphere "
                    "archive only after its declared convergence criterion passes"
                ),
                "converged_on_pass": 4,
                "deep_layer_relative_temperature_change": 4.778548e-4,
                "maximum_deep_layer_relative_temperature_change": 5.0e-4,
            },
            "atmosphere_array_parity": {
                "accepted": True,
                "comparison": "bitwise shape + dtype + C-order payload",
                "run_count": len(runs),
                "payload_sha256": _payload_digest(atmosphere_arrays),
                "arrays": _array_records(atmosphere_arrays, ATMOSPHERE_UNITS),
            },
            "spectrum_array_parity": {
                "accepted": True,
                "comparison": ("bitwise physical arrays; wall-clock seconds excluded"),
                "run_count": len(runs),
                "physical_payload_sha256": _payload_digest(spectrum_arrays),
                "arrays": _array_records(spectrum_arrays, SPECTRUM_UNITS),
            },
            "normalized_flux_ratio": {
                "accepted": True,
                "test": "normalized_flux ~= flux_total / flux_continuum",
                "rtol": 3.0e-7,
                "atol": 0.0,
            },
            "declared_flux_error_threshold": {
                "accepted": None,
                "status": "not_evaluated",
                "reason": (
                    "the prepared product archives do not retain an independently "
                    "reviewed flux-error acceptance record"
                ),
            },
            "hydrostatic_residual": {
                "accepted": None,
                "status": "not_evaluated",
                "reason": (
                    "no independent hydrostatic-residual threshold was evaluated "
                    "for this publication"
                ),
            },
        },
        "inputs": [
            _input_record(run, atmosphere, spectrum)
            for run, atmosphere, spectrum in zip(
                runs, atmosphere_inputs, spectrum_inputs, strict=True
            )
        ],
        "outputs": {
            "atmosphere": {
                "path": _display_path(golden_atmosphere),
                "role": "comparison-only golden",
                "sha256": sha256(golden_atmosphere),
                "payload_sha256": _payload_digest(atmosphere_arrays),
            },
            "spectrum": {
                "path": _display_path(golden_spectrum),
                "role": "timing-free comparison-only golden",
                "sha256": sha256(golden_spectrum),
                "physical_payload_sha256": _payload_digest(spectrum_arrays),
            },
        },
        "acceptance_scope": {
            "accepted_for": [
                "exact staged/repeat/pinned array reproducibility",
                "schema-v4 structured-atmosphere handoff",
                "finite wavelength-aligned spectrum",
                "normalized-flux internal ratio",
            ],
            "not_accepted_for": [
                "independent flux-closure threshold",
                "independent hydrostatic-residual threshold",
            ],
        },
    }


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    sort_keys: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _manifest_entry(
    *,
    path: Path,
    scope: str,
    acceptance_path: Path,
    arrays: Mapping[str, np.ndarray] | None = None,
    units: Mapping[str, str] | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": _display_path(path),
        "role": "golden",
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "format": path.suffix.removeprefix("."),
        "scope": scope,
        "source": "three exact local captures including pinned read-only oracle",
        "source_commit": PINNED_PAYNE_ZERO_COMMIT,
        "requires_optional_full_catalog": True,
        "builder": BUILDER_PATH,
    }
    if path != acceptance_path:
        entry["acceptance_record"] = _display_path(acceptance_path)
        entry["acceptance_sha256"] = sha256(acceptance_path)
    if arrays is not None:
        if units is None:
            raise ValueError("array units are required")
        entry["arrays"] = _array_records(arrays, units)
    return entry


def update_data_manifest(
    *,
    atmosphere_path: Path,
    spectrum_path: Path,
    acceptance_path: Path,
    atmosphere_arrays: Mapping[str, np.ndarray],
    spectrum_arrays: Mapping[str, np.ndarray],
    manifest_path: Path | None = None,
) -> None:
    """Add or replace the three accepted Chapter 15 artifact entries."""

    target = (
        REPOSITORY_ROOT / "data/MANIFEST.json"
        if manifest_path is None
        else Path(manifest_path)
    )
    manifest = json.loads(target.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ParityError("unsupported data manifest schema")
    entries = [
        _manifest_entry(
            path=atmosphere_path,
            scope=(
                "Comparison-only 80-layer solar schema-v4 atmosphere accepted "
                "by exact staged/repeat/pinned array parity; not a separate "
                "flux-closure or hydrostatic-residual acceptance."
            ),
            acceptance_path=acceptance_path,
            arrays=atmosphere_arrays,
            units=ATMOSPHERE_UNITS,
        ),
        _manifest_entry(
            path=spectrum_path,
            scope=(
                "Timing-free 498.95--499.15 nm solar spectrum accepted by exact "
                "staged/repeat/pinned physical-array parity."
            ),
            acceptance_path=acceptance_path,
            arrays=spectrum_arrays,
            units=SPECTRUM_UNITS,
        ),
        _manifest_entry(
            path=acceptance_path,
            scope=(
                "Role-honest Chapter 15 three-run parity and provenance record; "
                "flux-error and hydrostatic-residual acceptance remain unevaluated."
            ),
            acceptance_path=acceptance_path,
        ),
    ]
    owned_paths = {entry["path"] for entry in entries}
    manifest["entries"] = [
        entry for entry in manifest["entries"] if entry["path"] not in owned_paths
    ] + entries
    _atomic_write_json(target, manifest, sort_keys=False)


def validate_published(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, object]:
    """Validate accepted goldens without requiring source catalogs or runs."""

    atmosphere_path = output_directory / GOLDEN_ATMOSPHERE_NAME
    spectrum_path = output_directory / GOLDEN_SPECTRUM_NAME
    acceptance_path = output_directory / ACCEPTANCE_NAME
    missing = [
        str(path)
        for path in (atmosphere_path, spectrum_path, acceptance_path)
        if not path.is_file()
    ]
    if missing:
        raise ParityError(
            "published Chapter 15 artifacts missing: " + ", ".join(missing)
        )

    record = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if record.get("schema") != "chapter15_verified_solar_acceptance_v1":
        raise ParityError("unsupported Chapter 15 solar acceptance schema")
    if record.get("status") != "accepted_exact_three_run_array_parity":
        raise ParityError("Chapter 15 solar acceptance record is not accepted")
    if record.get("payne_zero_commit") != PINNED_PAYNE_ZERO_COMMIT:
        raise ParityError("Chapter 15 solar pinned commit changed")

    catalog_policy = record.get("catalog_policy", {})
    if not catalog_policy.get("full_source_catalogs_are_optional_textbook_data"):
        raise ParityError("Chapter 15 full-catalog optionality contract changed")
    if catalog_policy.get("required_to_validate_published_goldens"):
        raise ParityError("published golden validation must not require full catalogs")
    timing_policy = record.get("timing_policy", {})
    if timing_policy.get("excluded_from_identity") != ["seconds"]:
        raise ParityError("Chapter 15 spectral timing identity policy changed")

    evidence = record.get("evidence", {})
    if evidence.get("declared_flux_error_threshold", {}).get("accepted") is not None:
        raise ParityError("Chapter 15 record overclaims flux-error acceptance")
    if evidence.get("hydrostatic_residual", {}).get("accepted") is not None:
        raise ParityError("Chapter 15 record overclaims hydrostatic acceptance")

    atmosphere = _load_npz(atmosphere_path)
    spectrum = _load_npz(spectrum_path)
    _validate_atmosphere(atmosphere_path, atmosphere)
    _validate_spectrum(spectrum_path, spectrum)
    if "seconds" in spectrum:
        raise ParityError("published Chapter 15 spectrum contains wall-clock timing")

    outputs = record.get("outputs", {})
    expected_atmosphere = outputs.get("atmosphere", {})
    expected_spectrum = outputs.get("spectrum", {})
    if sha256(atmosphere_path) != expected_atmosphere.get("sha256"):
        raise ParityError("published Chapter 15 atmosphere archive hash changed")
    if sha256(spectrum_path) != expected_spectrum.get("sha256"):
        raise ParityError("published Chapter 15 spectrum archive hash changed")
    if _payload_digest(atmosphere) != expected_atmosphere.get("payload_sha256"):
        raise ParityError("published Chapter 15 atmosphere payload changed")
    if _payload_digest(spectrum) != expected_spectrum.get("physical_payload_sha256"):
        raise ParityError("published Chapter 15 spectrum payload changed")

    input_records = record.get("inputs")
    if not isinstance(input_records, list) or [
        item.get("role") for item in input_records
    ] != ["staged", "staged_repeat", "pinned_read_only_oracle"]:
        raise ParityError("Chapter 15 three-run provenance roles changed")
    atmosphere_payloads = {
        item.get("atmosphere_payload_sha256") for item in input_records
    }
    spectrum_payloads = {
        item.get("spectrum_physical_payload_sha256") for item in input_records
    }
    if atmosphere_payloads != {expected_atmosphere.get("payload_sha256")}:
        raise ParityError("Chapter 15 atmosphere input parity record changed")
    if spectrum_payloads != {expected_spectrum.get("physical_payload_sha256")}:
        raise ParityError("Chapter 15 spectrum input parity record changed")
    return record


def publish(
    *,
    runs: tuple[RunArtifacts, ...],
    output_directory: Path,
    update_manifest: bool = True,
    manifest_path: Path | None = None,
) -> PublicationResult:
    """Validate three prepared runs and publish deterministic goldens."""

    if tuple(run.role for run in runs) != (
        "staged",
        "staged_repeat",
        "pinned_read_only_oracle",
    ):
        raise ParityError("run roles/order must be staged, repeat, pinned oracle")
    gate = publication_gate(runs)
    if not gate.ready:
        raise ParityError(f"{gate.status}: " + "; ".join(gate.blockers))

    atmosphere_inputs = tuple(_load_npz(run.atmosphere) for run in runs)
    spectrum_inputs = tuple(_load_npz(run.spectrum) for run in runs)
    for run, atmosphere, spectrum in zip(
        runs, atmosphere_inputs, spectrum_inputs, strict=True
    ):
        _validate_atmosphere(run.atmosphere, atmosphere)
        _validate_spectrum(run.spectrum, spectrum)

    reference_atmosphere = atmosphere_inputs[0]
    reference_spectrum = spectrum_inputs[0]
    for run, atmosphere, spectrum in zip(
        runs[1:], atmosphere_inputs[1:], spectrum_inputs[1:], strict=True
    ):
        _compare_bitwise(
            kind="atmosphere",
            reference_role=runs[0].role,
            reference=reference_atmosphere,
            candidate_role=run.role,
            candidate=atmosphere,
            names=ATMOSPHERE_ARRAYS,
        )
        _compare_bitwise(
            kind="spectrum",
            reference_role=runs[0].role,
            reference=reference_spectrum,
            candidate_role=run.role,
            candidate=spectrum,
            names=SPECTRUM_PHYSICAL_ARRAYS,
        )

    atmosphere_arrays = {
        name: np.asarray(reference_atmosphere[name]) for name in ATMOSPHERE_ARRAYS
    }
    spectrum_arrays = {
        name: np.asarray(reference_spectrum[name]) for name in SPECTRUM_PHYSICAL_ARRAYS
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    atmosphere_path = output_directory / GOLDEN_ATMOSPHERE_NAME
    spectrum_path = output_directory / GOLDEN_SPECTRUM_NAME
    acceptance_path = output_directory / ACCEPTANCE_NAME

    with tempfile.TemporaryDirectory(
        dir=output_directory, prefix=".chapter15_publish_"
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary_atmosphere = temporary / GOLDEN_ATMOSPHERE_NAME
        temporary_spectrum = temporary / GOLDEN_SPECTRUM_NAME
        write_npz(temporary_atmosphere, atmosphere_arrays)
        write_npz(temporary_spectrum, spectrum_arrays)
        os.replace(temporary_atmosphere, atmosphere_path)
        os.replace(temporary_spectrum, spectrum_path)

    record = _acceptance_record(
        runs=runs,
        atmosphere_inputs=atmosphere_inputs,
        spectrum_inputs=spectrum_inputs,
        golden_atmosphere=atmosphere_path,
        golden_spectrum=spectrum_path,
    )
    _atomic_write_json(acceptance_path, record)
    if update_manifest:
        update_data_manifest(
            atmosphere_path=atmosphere_path,
            spectrum_path=spectrum_path,
            acceptance_path=acceptance_path,
            atmosphere_arrays=atmosphere_arrays,
            spectrum_arrays=spectrum_arrays,
            manifest_path=manifest_path,
        )
    return PublicationResult(
        atmosphere=atmosphere_path,
        spectrum=spectrum_path,
        acceptance=acceptance_path,
        record=record,
    )


def default_runs(
    *,
    staged_directory: Path = DEFAULT_STAGED_DIRECTORY,
    repeat_directory: Path = DEFAULT_REPEAT_DIRECTORY,
    pinned_directory: Path = DEFAULT_PINNED_DIRECTORY,
) -> tuple[RunArtifacts, ...]:
    """Return the fixed role ordering for the prepared captures."""

    return (
        RunArtifacts(
            "staged",
            staged_directory / ATMOSPHERE_INPUT_NAME,
            staged_directory / SPECTRUM_INPUT_NAME,
        ),
        RunArtifacts(
            "staged_repeat",
            repeat_directory / ATMOSPHERE_INPUT_NAME,
            repeat_directory / SPECTRUM_INPUT_NAME,
        ),
        RunArtifacts(
            "pinned_read_only_oracle",
            pinned_directory / ATMOSPHERE_INPUT_NAME,
            pinned_directory / SPECTRUM_INPUT_NAME,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged-directory", type=Path, default=DEFAULT_STAGED_DIRECTORY
    )
    parser.add_argument(
        "--repeat-directory", type=Path, default=DEFAULT_REPEAT_DIRECTORY
    )
    parser.add_argument(
        "--pinned-directory", type=Path, default=DEFAULT_PINNED_DIRECTORY
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="only report whether all three prepared captures exist",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate published goldens without source catalogs or audit runs",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="publish artifacts without changing data/MANIFEST.json",
    )
    args = parser.parse_args()
    if args.status and args.validate:
        parser.error("--status and --validate are mutually exclusive")
    if args.validate:
        record = validate_published(args.output_directory)
        print("Chapter 15 verified solar goldens: verified " f"({record['status']})")
        return
    runs = default_runs(
        staged_directory=args.staged_directory,
        repeat_directory=args.repeat_directory,
        pinned_directory=args.pinned_directory,
    )
    gate = publication_gate(runs)
    if args.status or not gate.ready:
        print(
            json.dumps(
                {
                    "status": gate.status,
                    "ready": gate.ready,
                    "blockers": list(gate.blockers),
                },
                indent=2,
                sort_keys=True,
            )
        )
        if args.status:
            return
        raise SystemExit(2)

    result = publish(
        runs=runs,
        output_directory=args.output_directory,
        update_manifest=not args.no_manifest,
    )
    print("Chapter 15 verified solar goldens: accepted " f"({result.record['status']})")


if __name__ == "__main__":
    main()
