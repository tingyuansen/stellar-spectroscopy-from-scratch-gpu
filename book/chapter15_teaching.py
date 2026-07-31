"""Transparent Chapter 15 workflow and acceptance composition.

The exact initializer, atmosphere solver, schema loader, and synthesizer live
in the staged Payne Zero packages.  This module owns only immutable requests,
editorial gate rows, and deterministic summaries of their public outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


_FAMILY_NAMES = {"auto", "five_label", "cno8", "direct_abundance"}
_NON_DEPTH_SCHEMA_FIELDS = {
    "elemental_abundances",
    "signed_continuum_edge_frequency_hz",
    "continuum_edge_wavelength_nm",
    "continuum_edge_midpoint_wavelength_nm",
    "continuum_edge_interval_width_squared_over_two_nm2",
}


@dataclass(frozen=True)
class StellarRequest:
    """One exact label/mixture request before initializer routing."""

    name: str
    effective_temperature: float
    log_surface_gravity: float
    metallicity: float
    alpha_enhancement: float
    microturbulence_km_s: float
    c_over_m: float | None
    n_over_m: float | None
    o_over_m: float | None
    initializer_family: str
    molecular_lines: bool

    def __post_init__(self) -> None:
        numeric = (
            self.effective_temperature,
            self.log_surface_gravity,
            self.metallicity,
            self.alpha_enhancement,
            self.microturbulence_km_s,
            self.c_over_m,
            self.n_over_m,
            self.o_over_m,
        )
        if not self.name:
            raise ValueError("request name must not be empty")
        if any(value is not None and not np.isfinite(value) for value in numeric):
            raise ValueError("all supplied stellar labels must be finite")
        if self.effective_temperature <= 0.0:
            raise ValueError("effective_temperature must be positive")
        if self.microturbulence_km_s < 0.0:
            raise ValueError("microturbulence_km_s must be non-negative")
        if self.initializer_family not in _FAMILY_NAMES:
            raise ValueError(
                "initializer_family must be auto, five_label, cno8, "
                "or direct_abundance"
            )

    def label_payload(self) -> dict[str, object]:
        """Return exact public ordinary/CNO labels without dropping nulls."""

        return {
            "effective_temperature": float(self.effective_temperature),
            "log_surface_gravity": float(self.log_surface_gravity),
            "metallicity": float(self.metallicity),
            "alpha_enhancement": float(self.alpha_enhancement),
            "microturbulence_km_s": float(self.microturbulence_km_s),
            "c_over_m": (
                None if self.c_over_m is None else float(self.c_over_m)
            ),
            "n_over_m": (
                None if self.n_over_m is None else float(self.n_over_m)
            ),
            "o_over_m": (
                None if self.o_over_m is None else float(self.o_over_m)
            ),
        }

    def canonical_payload(self) -> dict[str, object]:
        """Return every identity-bearing request field."""

        return {
            "name": self.name,
            **self.label_payload(),
            "initializer_family": self.initializer_family,
            "molecular_lines": bool(self.molecular_lines),
        }


@dataclass(frozen=True)
class AcceptanceRow:
    """One independent editorial acceptance gate."""

    gate: str
    passed: bool | None
    evidence: str


@dataclass(frozen=True)
class AtmosphereSummary:
    """Shape, dtype, and physical-boundary summary of one mapping."""

    required_field_count: int
    missing_fields: tuple[str, ...]
    depth_count: int
    all_depth_axes_match: bool
    all_public_arrays_float64: bool
    finite: bool
    column_mass_positive: bool
    column_mass_strictly_increasing: bool
    positive_thermodynamic_columns: bool


@dataclass(frozen=True)
class SpectrumSummary:
    """Public spectral semantics without introducing a result wrapper."""

    wavelength_count: int
    wavelength_strictly_increasing: bool
    arrays_aligned: bool
    finite: bool
    continuum_nonzero: bool
    normalized_ratio_matches: bool
    minimum_normalized_flux: float
    maximum_normalized_flux: float


@dataclass(frozen=True)
class WorkflowBoundary:
    """Availability of one exact public workflow boundary."""

    mode: str
    entrypoint: str
    returned_type: str
    available: bool
    blocker: str | None


def stellar_request_from_mapping(values: Mapping[str, object]) -> StellarRequest:
    """Parse one manifest request with no aliases or implicit CNO filling."""

    expected = {
        "name",
        "effective_temperature",
        "log_surface_gravity",
        "metallicity",
        "alpha_enhancement",
        "microturbulence_km_s",
        "c_over_m",
        "n_over_m",
        "o_over_m",
        "initializer_family",
        "molecular_lines",
    }
    observed = set(values)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"request fields differ; missing={missing}, extra={extra}")
    return StellarRequest(
        name=str(values["name"]),
        effective_temperature=float(values["effective_temperature"]),
        log_surface_gravity=float(values["log_surface_gravity"]),
        metallicity=float(values["metallicity"]),
        alpha_enhancement=float(values["alpha_enhancement"]),
        microturbulence_km_s=float(values["microturbulence_km_s"]),
        c_over_m=(
            None if values["c_over_m"] is None else float(values["c_over_m"])
        ),
        n_over_m=(
            None if values["n_over_m"] is None else float(values["n_over_m"])
        ),
        o_over_m=(
            None if values["o_over_m"] is None else float(values["o_over_m"])
        ),
        initializer_family=str(values["initializer_family"]),
        molecular_lines=bool(values["molecular_lines"]),
    )


def canonical_json(values: Mapping[str, object]) -> str:
    """Return the stable JSON representation used by Chapter 15 identities."""

    return json.dumps(
        dict(values),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_digest(values: Mapping[str, object]) -> str:
    """Hash a canonical mapping without timing or object-address state."""

    return hashlib.sha256(canonical_json(values).encode("utf-8")).hexdigest()


def request_digest(request: StellarRequest) -> str:
    """Return the complete immutable identity of one capstone request."""

    return canonical_digest(request.canonical_payload())


def summarize_atmosphere(
    atmosphere: Mapping[str, np.ndarray],
    *,
    required_fields: Sequence[str],
) -> AtmosphereSummary:
    """Summarize one schema mapping without calling a validator twice."""

    required = tuple(required_fields)
    missing = tuple(name for name in required if name not in atmosphere)
    arrays = [
        np.asarray(atmosphere[name])
        for name in required
        if name in atmosphere
    ]
    depth_count = (
        int(np.asarray(atmosphere["temperature"]).shape[0])
        if "temperature" in atmosphere
        and np.asarray(atmosphere["temperature"]).ndim >= 1
        else 0
    )
    depth_arrays = [
        np.asarray(atmosphere[name])
        for name in required
        if name in atmosphere and name not in _NON_DEPTH_SCHEMA_FIELDS
    ]
    axes_match = bool(
        depth_count > 0
        and depth_arrays
        and all(
            array.ndim >= 1 and array.shape[0] == depth_count
            for array in depth_arrays
        )
    )
    public_float64 = bool(
        arrays and all(array.dtype == np.float64 for array in arrays)
    )
    finite = bool(arrays and all(np.all(np.isfinite(array)) for array in arrays))
    column_mass = np.asarray(
        atmosphere.get("column_mass", np.asarray([], dtype=np.float64)),
        dtype=np.float64,
    )
    positive_names = (
        "temperature",
        "gas_pressure",
        "electron_density",
        "mass_density",
    )
    positive_thermodynamics = bool(
        all(
            name in atmosphere and np.all(np.asarray(atmosphere[name]) > 0.0)
            for name in positive_names
        )
    )
    return AtmosphereSummary(
        required_field_count=len(required),
        missing_fields=missing,
        depth_count=depth_count,
        all_depth_axes_match=axes_match,
        all_public_arrays_float64=public_float64,
        finite=finite,
        column_mass_positive=bool(
            column_mass.size > 0 and np.all(column_mass > 0.0)
        ),
        column_mass_strictly_increasing=bool(
            column_mass.size > 1 and np.all(np.diff(column_mass) > 0.0)
        ),
        positive_thermodynamic_columns=positive_thermodynamics,
    )


def summarize_spectrum(
    *,
    wavelength_nm: np.ndarray,
    flux_total: np.ndarray,
    flux_continuum: np.ndarray,
    normalized_flux: np.ndarray,
    ratio_rtol: float,
    ratio_atol: float,
) -> SpectrumSummary:
    """Check alignment, finiteness, wavelength order, and flux semantics."""

    wavelength = np.asarray(wavelength_nm, dtype=np.float64)
    total = np.asarray(flux_total, dtype=np.float64)
    continuum = np.asarray(flux_continuum, dtype=np.float64)
    normalized = np.asarray(normalized_flux, dtype=np.float64)
    aligned = (
        wavelength.ndim == 1
        and total.shape == wavelength.shape
        and continuum.shape == wavelength.shape
        and normalized.shape == wavelength.shape
    )
    finite = bool(
        aligned
        and all(
            np.all(np.isfinite(values))
            for values in (wavelength, total, continuum, normalized)
        )
    )
    continuum_nonzero = bool(aligned and np.all(continuum != 0.0))
    ratio_matches = bool(
        finite
        and continuum_nonzero
        and np.allclose(
            normalized,
            total / continuum,
            rtol=float(ratio_rtol),
            atol=float(ratio_atol),
        )
    )
    return SpectrumSummary(
        wavelength_count=int(wavelength.size),
        wavelength_strictly_increasing=bool(
            wavelength.size > 1 and np.all(np.diff(wavelength) > 0.0)
        ),
        arrays_aligned=aligned,
        finite=finite,
        continuum_nonzero=continuum_nonzero,
        normalized_ratio_matches=ratio_matches,
        minimum_normalized_flux=(
            float(np.min(normalized)) if normalized.size else float("nan")
        ),
        maximum_normalized_flux=(
            float(np.max(normalized)) if normalized.size else float("nan")
        ),
    )


def acceptance_rows(
    *,
    workflow: str,
    initializer_assets_valid: bool | None,
    seed_valid: bool | None,
    structural_convergence: bool | None,
    flux_accepted: bool | None,
    hydrostatic_accepted: bool | None,
    optical_grid_accepted: bool | None,
    schema_valid: bool | None,
    spectrum_valid: bool | None,
    golden_spectral_parity: bool | None,
    provenance_valid: bool | None,
    blockers: Sequence[str] = (),
) -> tuple[AcceptanceRow, ...]:
    """Assemble independent rows; unavailable evidence remains ``None``."""

    if workflow not in {"exploratory", "verified"}:
        raise ValueError("workflow must be exploratory or verified")
    blocked_text = "; ".join(str(value) for value in blockers) or "none"
    return (
        AcceptanceRow(
            "initializer support and asset integrity",
            initializer_assets_valid,
            "pinned local source, manifests, and checkpoints",
        ),
        AcceptanceRow(
            "seed finite, positive, and monotone",
            seed_valid,
            "exact initialized seed" if seed_valid is not None else blocked_text,
        ),
        AcceptanceRow(
            "structural fixed-point convergence",
            structural_convergence,
            (
                "Chapter 13 exact result"
                if structural_convergence is not None
                else blocked_text
            ),
        ),
        AcceptanceRow(
            "declared flux-error threshold",
            flux_accepted,
            "independent flux diagnostic" if flux_accepted is not None else blocked_text,
        ),
        AcceptanceRow(
            "hydrostatic residual",
            hydrostatic_accepted,
            (
                "independent hydrostatic diagnostic"
                if hydrostatic_accepted is not None
                else blocked_text
            ),
        ),
        AcceptanceRow(
            "standard optical-depth grid",
            optical_grid_accepted,
            "exact solver grid" if optical_grid_accepted is not None else blocked_text,
        ),
        AcceptanceRow(
            "schema-v4 mapping",
            schema_valid,
            (
                "25 canonical public arrays"
                if schema_valid is not None
                else blocked_text
            ),
        ),
        AcceptanceRow(
            "finite wavelength-aligned spectrum",
            spectrum_valid,
            (
                "Chapter 10 public flux fields and ratio"
                if spectrum_valid is not None
                else blocked_text
            ),
        ),
        AcceptanceRow(
            "golden spectral parity",
            golden_spectral_parity,
            (
                "reference-case golden"
                if golden_spectral_parity is not None
                else "no Chapter 15 golden physical product installed"
            ),
        ),
        AcceptanceRow(
            "provenance and limitations",
            provenance_valid,
            "canonical request/runtime/source digest plus named blockers",
        ),
    )
