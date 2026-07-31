"""Progressive Chapter 7 helpers for atomic catalogs and line forests.

The notebook keeps raw-source decoding and discrete catalog decisions on the
host, then calls the exact staged synthesis accumulator for the resulting
ordinary-line forest.  No helper in this module opens a golden output.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np

from book.chapter06_runtime import (
    LINE_CENTER_CUTOFF_RATIO,
    RAW_LINE_FIELDS,
    SYNTHESIS_LINE_TABLES,
    configure_local_data_paths,
    synthesis_line_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ATOMIC_TEACHING_SUBSET = (
    REPOSITORY_ROOT / "data/subsets/chapter07_atomic_catalog_subset.npz"
)
ATOMIC_TEACHING_SUBSET_SHA256 = (
    "d797e747d7f557d172505bbe546c0d025dbc2c7a4e0cce831a8bdbec94573e23"
)
PINNED_FULL_CATALOG_SHA256 = (
    "4eafa927c02a4f74401523149a44e35239f2aaecb4a64f2905a4cd5530c2dde7"
)
PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"


@dataclass(frozen=True)
class AtomicSourceCheckpoint:
    """Identity and route counts for the compact raw teaching catalog."""

    record_count: int
    source_row_index: np.ndarray
    source_catalog_sha256: str
    payne_zero_commit: str
    route_code: np.ndarray
    route_counts: dict[int, int]
    full_catalog_route_counts: dict[int, int]


@dataclass(frozen=True)
class RecordTransformCheckpoint:
    """Observable effects of the exact raw-to-physical record compiler."""

    isotope_changed_count: int
    default_radiative_count: int
    default_stark_count: int
    default_van_der_waals_count: int
    deuterium_source_row: int
    deuterium_log_gf_without_correction: float
    deuterium_log_gf_with_correction: float
    deuterium_strength_ratio: float
    maximum_stored_minus_derived_wavelength_nm: float


@dataclass(frozen=True)
class WindowSelectionCheckpoint:
    """Conservative line selection around one requested wavelength interval."""

    start_wavelength_nm: float
    end_wavelength_nm: float
    wavelength_nm: np.ndarray
    margin_nm: np.ndarray
    center_inside: np.ndarray
    selected_with_margin: np.ndarray
    selected_count: int
    center_inside_count: int
    outside_center_but_selected_count: int


@dataclass(frozen=True)
class CatalogLayoutCheckpoint:
    """The geometric wavelength grid and its sorted structure-of-arrays catalog."""

    wavelength_nm: np.ndarray
    catalog_mapping: dict[str, np.ndarray]
    type_segments: dict[int, tuple[int, int]]
    line_count: int
    center_index: np.ndarray
    on_grid_center_count: int
    unique_on_grid_center_count: int
    colliding_on_grid_record_count: int


@dataclass(frozen=True)
class ScatterAddCheckpoint:
    """A small collision that distinguishes accumulation from assignment."""

    columns: np.ndarray
    contributions: np.ndarray
    overwritten: np.ndarray
    scatter_added: np.ndarray
    private_buffers: np.ndarray
    reduced: np.ndarray


@dataclass(frozen=True)
class AtomicForestCheckpoint:
    """Exact ordinary-line synthesis deposits for one compact line forest."""

    regime: str
    wavelength_nm: np.ndarray
    continuum_opacity_cm2_per_g: np.ndarray
    gross_line_mass_absorption_coefficient: np.ndarray
    net_line_mass_absorption_coefficient: np.ndarray
    selected_line_count: int
    metal_line_count: int
    center_collision_count: int
    nonzero_count_per_depth: np.ndarray
    peak_net_opacity_cm2_per_g: float
    device: str
    work_dtype: str
    accumulation_dtype: str
    wing_mode: str


@dataclass(frozen=True)
class WingModeComparison:
    """Measured agreement between the batched and explicit ordinary wing walks."""

    batched: np.ndarray
    loop: np.ndarray
    maximum_absolute_difference: float
    maximum_relative_difference: float


@dataclass(frozen=True)
class HydrogenCheckpoint:
    """Fine structure, isotope separation, and density-dependent series merging."""

    hydrogen_wavelength_nm: float
    deuterium_wavelength_nm: float
    isotope_separation_nm: float
    component_offset_hz: np.ndarray
    component_weight: np.ndarray
    oscillator_strength_2_to_3: float
    electron_density_cm3: np.ndarray
    merge_wavenumber_cm: np.ndarray
    synthesis_minimum_supported_lower_level: int
    atmosphere_supported_lower_level_range: tuple[int, int]


@dataclass(frozen=True)
class AutoionizingCheckpoint:
    """The exact raw Shore--Fano parameters for one catalog resonance."""

    source_row: int
    wavelength_nm: float
    radiative_width: float
    shore_asymmetry: float
    shore_baseline: float
    reduced_frequency_offset: np.ndarray
    positive_profile_ratio: np.ndarray


def _scalar_text(value: np.ndarray) -> str:
    """Decode one scalar string stored in an NPZ without object arrays."""

    scalar = np.asarray(value).item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


@lru_cache(maxsize=1)
def _raw_subset() -> dict[str, np.ndarray]:
    """Load and validate the compact source-faithful catalog exactly once."""

    actual_sha256 = hashlib.sha256(ATOMIC_TEACHING_SUBSET.read_bytes()).hexdigest()
    if actual_sha256 != ATOMIC_TEACHING_SUBSET_SHA256:
        raise RuntimeError("the Chapter 7 atomic teaching subset changed")
    with np.load(ATOMIC_TEACHING_SUBSET, allow_pickle=False) as archive:
        missing = sorted(set(RAW_LINE_FIELDS).difference(archive.files))
        if missing:
            raise ValueError(
                "the Chapter 7 subset is missing raw fields: " + ", ".join(missing)
            )
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
    record_count = int(payload["source_row_index"].size)
    for name in RAW_LINE_FIELDS:
        if payload[name].shape != (record_count,):
            raise ValueError(f"raw field {name!r} has inconsistent shape")
    if _scalar_text(payload["source_catalog_sha256"]) != PINNED_FULL_CATALOG_SHA256:
        raise RuntimeError("the compact subset names the wrong full source catalog")
    if _scalar_text(payload["payne_zero_commit"]) != PINNED_PAYNE_ZERO_COMMIT:
        raise RuntimeError("the compact subset names the wrong implementation commit")
    return payload


def load_atomic_subset() -> dict[str, np.ndarray]:
    """Return defensive copies of the compact raw catalog."""

    return {name: values.copy() for name, values in _raw_subset().items()}


def build_atomic_records(*, apply_iso_corr: bool = True) -> dict[str, np.ndarray]:
    """Run the exact physical record compiler on the compact raw catalog."""

    configure_local_data_paths()
    from payne_zero_synthesis.atomic_lines import _build_records

    raw = _raw_subset()
    fields = {name: raw[name] for name in RAW_LINE_FIELDS}
    return {
        name: np.asarray(values).copy()
        for name, values in _build_records(
            fields,
            apply_iso_corr=apply_iso_corr,
        ).items()
    }


def atomic_source_checkpoint() -> AtomicSourceCheckpoint:
    """Summarize the teaching subset without pretending it is the full catalog."""

    raw = _raw_subset()
    records = build_atomic_records()
    route_code = np.asarray(records["line_type"], dtype=np.int64)
    route_counts = {
        int(code): int(np.count_nonzero(route_code == code))
        for code in np.unique(route_code)
    }
    return AtomicSourceCheckpoint(
        record_count=int(route_code.size),
        source_row_index=np.asarray(raw["source_row_index"], dtype=np.int64).copy(),
        source_catalog_sha256=_scalar_text(raw["source_catalog_sha256"]),
        payne_zero_commit=_scalar_text(raw["payne_zero_commit"]),
        route_code=route_code,
        route_counts=route_counts,
        full_catalog_route_counts={
            -6: 469,
            -3: 1_631,
            -2: 10,
            -1: 1_194,
            0: 1_936_350,
            1: 321,
        },
    )


def record_transform_checkpoint() -> RecordTransformCheckpoint:
    """Expose corrections and defaults before any wavelength-window selection."""

    raw = _raw_subset()
    corrected = build_atomic_records(apply_iso_corr=True)
    uncorrected = build_atomic_records(apply_iso_corr=False)
    correction = (
        np.asarray(raw["primary_isotope_log_correction"], dtype=np.float64)
        + np.asarray(raw["secondary_isotope_log_correction"], dtype=np.float64)
    )
    deuterium_index = int(np.flatnonzero(corrected["line_type"] == -2)[0])
    stored_minus_derived = (
        np.asarray(raw["stored_wavelength_nm"], dtype=np.float64)
        - np.asarray(corrected["wavelength_nm"], dtype=np.float64)
    )
    return RecordTransformCheckpoint(
        isotope_changed_count=int(np.count_nonzero(correction)),
        default_radiative_count=int(
            np.count_nonzero(raw["radiative_damping_log"] == 0.0)
        ),
        default_stark_count=int(np.count_nonzero(raw["stark_damping_log"] == 0.0)),
        default_van_der_waals_count=int(
            np.count_nonzero(raw["van_der_waals_damping_log"] == 0.0)
        ),
        deuterium_source_row=int(raw["source_row_index"][deuterium_index]),
        deuterium_log_gf_without_correction=float(
            uncorrected["log_oscillator_strength"][deuterium_index]
        ),
        deuterium_log_gf_with_correction=float(
            corrected["log_oscillator_strength"][deuterium_index]
        ),
        deuterium_strength_ratio=float(
            corrected["oscillator_strength"][deuterium_index]
            / uncorrected["oscillator_strength"][deuterium_index]
        ),
        maximum_stored_minus_derived_wavelength_nm=float(
            np.max(np.abs(stored_minus_derived))
        ),
    )


def _selection_margin_nm(
    records: dict[str, np.ndarray],
    start_wavelength_nm: float,
) -> np.ndarray:
    """Reproduce the exact line-size-to-window-margin transformation."""

    configure_local_data_paths()
    from payne_zero_synthesis.atomic_lines import _LINE_WINDOW_MARGINS_NM

    line_size = np.asarray(records["line_size"], dtype=np.int64)
    margin_class = np.minimum(8 - np.where(line_size > 0, line_size, 0), 7)
    hydrogen = (
        np.asarray(records["atomic_number"], dtype=np.int64) == 1
    ) | np.isin(records["line_type"], [-1, -2])
    margin_class = np.where(hydrogen, 1, margin_class)
    margin_class = np.clip(margin_class, 1, 7)
    red_scale = (
        1.0 if start_wavelength_nm <= 500.0 else start_wavelength_nm / 500.0
    )
    return np.asarray(_LINE_WINDOW_MARGINS_NM)[margin_class - 1] * red_scale


def window_selection_checkpoint(
    start_wavelength_nm: float = 498.95,
    end_wavelength_nm: float = 499.15,
) -> WindowSelectionCheckpoint:
    """Apply the exact inclusive line-window inequality."""

    if not end_wavelength_nm > start_wavelength_nm:
        raise ValueError("the wavelength interval must have positive width")
    configure_local_data_paths()
    from payne_zero_synthesis.atomic_lines import _line_window_mask

    records = build_atomic_records()
    wavelength = np.asarray(records["wavelength_nm"], dtype=np.float64)
    margin = _selection_margin_nm(records, start_wavelength_nm)
    center_inside = (wavelength >= start_wavelength_nm) & (
        wavelength <= end_wavelength_nm
    )
    selected = np.asarray(
        _line_window_mask(records, start_wavelength_nm, end_wavelength_nm),
        dtype=bool,
    )
    expected = (wavelength >= start_wavelength_nm - margin) & (
        wavelength <= end_wavelength_nm + margin
    )
    if not np.array_equal(selected, expected):
        raise RuntimeError("displayed line-window inequality diverged from production")
    return WindowSelectionCheckpoint(
        start_wavelength_nm=float(start_wavelength_nm),
        end_wavelength_nm=float(end_wavelength_nm),
        wavelength_nm=wavelength,
        margin_nm=margin,
        center_inside=center_inside,
        selected_with_margin=selected,
        selected_count=int(np.count_nonzero(selected)),
        center_inside_count=int(np.count_nonzero(center_inside)),
        outside_center_but_selected_count=int(np.count_nonzero(selected & ~center_inside)),
    )


def _catalog_support_tables() -> dict[str, np.ndarray]:
    """Load the exact profile-table fields required by the device constructor."""

    with np.load(SYNTHESIS_LINE_TABLES, allow_pickle=False) as archive:
        return {
            "helium_line_type": np.zeros(0, dtype=np.int64),
            "helium_line_center_cutoff_ratio": np.asarray(
                LINE_CENTER_CUTOFF_RATIO,
                dtype=np.float64,
            ),
            "harris_profile_h0_table": np.asarray(
                archive["harris_profile_h0_table"],
                dtype=np.float64,
            ).copy(),
            "harris_profile_h1_table": np.asarray(
                archive["harris_profile_h1_table"],
                dtype=np.float64,
            ).copy(),
            "harris_profile_h2_table": np.asarray(
                archive["harris_profile_h2_table"],
                dtype=np.float64,
            ).copy(),
        }


@lru_cache(maxsize=8)
def _catalog_layout_cached(
    start_wavelength_nm: float,
    end_wavelength_nm: float,
    resolution: float,
) -> CatalogLayoutCheckpoint:
    """Build a type-and-center-sorted catalog on one geometric grid."""

    configure_local_data_paths()
    from payne_zero_synthesis.atomic_lines import (
        Grid,
        LineCatalog,
        _assemble_catalog,
        _line_window_mask,
    )
    from payne_zero_synthesis.line_opacity import nearest_grid_indices

    records = build_atomic_records()
    grid = Grid(start_wavelength_nm, end_wavelength_nm, resolution)
    wavelength = grid.build()
    selected = _line_window_mask(
        records,
        start_wavelength_nm,
        end_wavelength_nm,
    )
    catalog = _assemble_catalog(
        records,
        selected,
        grid,
        wavelength,
        "type_center",
    )
    mapping = {
        name: np.asarray(getattr(catalog, name)).copy()
        for name in LineCatalog._FIELDS
    }
    mapping.update(_catalog_support_tables())
    center_index = nearest_grid_indices(
        wavelength,
        mapping["index_wavelength_nm"],
    )
    on_grid = (center_index >= 0) & (center_index < wavelength.size)
    unique_on_grid = np.unique(center_index[on_grid])
    return CatalogLayoutCheckpoint(
        wavelength_nm=wavelength,
        catalog_mapping=mapping,
        type_segments=dict(catalog.type_segments),
        line_count=len(catalog),
        center_index=np.asarray(center_index, dtype=np.int64),
        on_grid_center_count=int(np.count_nonzero(on_grid)),
        unique_on_grid_center_count=int(unique_on_grid.size),
        colliding_on_grid_record_count=int(
            np.count_nonzero(on_grid) - unique_on_grid.size
        ),
    )


def catalog_layout_checkpoint(
    start_wavelength_nm: float = 498.95,
    end_wavelength_nm: float = 499.15,
    resolution: float = 300_000.0,
) -> CatalogLayoutCheckpoint:
    """Return defensive arrays for one compact compiled catalog."""

    cached = _catalog_layout_cached(
        float(start_wavelength_nm),
        float(end_wavelength_nm),
        float(resolution),
    )
    return CatalogLayoutCheckpoint(
        wavelength_nm=cached.wavelength_nm.copy(),
        catalog_mapping={
            name: np.asarray(values).copy()
            for name, values in cached.catalog_mapping.items()
        },
        type_segments=dict(cached.type_segments),
        line_count=cached.line_count,
        center_index=cached.center_index.copy(),
        on_grid_center_count=cached.on_grid_center_count,
        unique_on_grid_center_count=cached.unique_on_grid_center_count,
        colliding_on_grid_record_count=cached.colliding_on_grid_record_count,
    )


def scatter_add_checkpoint() -> ScatterAddCheckpoint:
    """Show why coincident lines require addition and private reductions."""

    columns = np.asarray([2, 4, 4, 4, 7, 2], dtype=np.int64)
    contributions = np.asarray([1.0, 2.0, 0.5, 3.0, 4.0, 0.25])
    overwritten = np.zeros(9, dtype=np.float64)
    overwritten[columns] = contributions
    scatter_added = np.zeros_like(overwritten)
    np.add.at(scatter_added, columns, contributions)

    private_buffers = np.zeros((2, overwritten.size), dtype=np.float64)
    np.add.at(private_buffers[0], columns[:3], contributions[:3])
    np.add.at(private_buffers[1], columns[3:], contributions[3:])
    reduced = private_buffers[0] + private_buffers[1]
    return ScatterAddCheckpoint(
        columns=columns,
        contributions=contributions,
        overwritten=overwritten,
        scatter_added=scatter_added,
        private_buffers=private_buffers,
        reduced=reduced,
    )


@lru_cache(maxsize=8)
def _run_atomic_forest_cached(
    regime: str,
    start_wavelength_nm: float,
    end_wavelength_nm: float,
    resolution: float,
    runtime_device_name: str,
    wing_mode: str,
) -> AtomicForestCheckpoint:
    """Run the exact staged ordinary-line accumulator without a golden read."""

    if wing_mode not in {"batched", "loop"}:
        raise ValueError("wing_mode must be 'batched' or 'loop'")
    configure_local_data_paths()
    import torch
    from payne_zero_synthesis.line_opacity import (
        accumulate_atomic,
        precompute_invariants,
    )

    layout = _catalog_layout_cached(
        start_wavelength_nm,
        end_wavelength_nm,
        resolution,
    )
    compute_device = torch.device(runtime_device_name)
    state, continuum = synthesis_line_state(regime, layout.wavelength_nm)
    invariants = precompute_invariants(
        layout.catalog_mapping,
        layout.wavelength_nm,
        runtime_device=compute_device,
    )
    gross_tensor = accumulate_atomic(
        invariants,
        state,
        do_metal=True,
        do_helium=False,
        apply_stim=False,
        wing_mode=wing_mode,
    )
    net_tensor = accumulate_atomic(
        invariants,
        state,
        do_metal=True,
        do_helium=False,
        apply_stim=True,
        wing_mode=wing_mode,
    )
    gross = gross_tensor.detach().cpu().numpy().copy()
    net = net_tensor.detach().cpu().numpy().copy()
    metal_centers = invariants.metal_center_index.detach().cpu().numpy()
    on_grid = (metal_centers >= 0) & (metal_centers < layout.wavelength_nm.size)
    center_collision_count = int(
        np.count_nonzero(on_grid) - np.unique(metal_centers[on_grid]).size
    )
    return AtomicForestCheckpoint(
        regime=regime,
        wavelength_nm=layout.wavelength_nm.copy(),
        continuum_opacity_cm2_per_g=np.asarray(continuum).copy(),
        gross_line_mass_absorption_coefficient=gross,
        net_line_mass_absorption_coefficient=net,
        selected_line_count=layout.line_count,
        metal_line_count=int(invariants.metal_catalog_index.numel()),
        center_collision_count=center_collision_count,
        nonzero_count_per_depth=np.count_nonzero(net, axis=1).astype(np.int64),
        peak_net_opacity_cm2_per_g=float(np.max(net)),
        device=str(net_tensor.device),
        work_dtype=str(invariants.wavelength_grid.dtype),
        accumulation_dtype=str(net_tensor.dtype),
        wing_mode=wing_mode,
    )


def run_atomic_forest(
    regime: str = "solar_dwarf",
    *,
    start_wavelength_nm: float = 498.95,
    end_wavelength_nm: float = 499.15,
    resolution: float = 300_000.0,
    runtime_device: object = "cpu",
    wing_mode: str = "batched",
) -> AtomicForestCheckpoint:
    """Run one exact compact forest and return defensive NumPy arrays."""

    checkpoint = _run_atomic_forest_cached(
        regime,
        float(start_wavelength_nm),
        float(end_wavelength_nm),
        float(resolution),
        str(runtime_device),
        wing_mode,
    )
    return AtomicForestCheckpoint(
        regime=checkpoint.regime,
        wavelength_nm=checkpoint.wavelength_nm.copy(),
        continuum_opacity_cm2_per_g=checkpoint.continuum_opacity_cm2_per_g.copy(),
        gross_line_mass_absorption_coefficient=(
            checkpoint.gross_line_mass_absorption_coefficient.copy()
        ),
        net_line_mass_absorption_coefficient=(
            checkpoint.net_line_mass_absorption_coefficient.copy()
        ),
        selected_line_count=checkpoint.selected_line_count,
        metal_line_count=checkpoint.metal_line_count,
        center_collision_count=checkpoint.center_collision_count,
        nonzero_count_per_depth=checkpoint.nonzero_count_per_depth.copy(),
        peak_net_opacity_cm2_per_g=checkpoint.peak_net_opacity_cm2_per_g,
        device=checkpoint.device,
        work_dtype=checkpoint.work_dtype,
        accumulation_dtype=checkpoint.accumulation_dtype,
        wing_mode=checkpoint.wing_mode,
    )


def compare_forest_wing_modes(
    regime: str = "solar_dwarf",
) -> WingModeComparison:
    """Run both exact ordinary wing policies on the same compact catalog."""

    batched = run_atomic_forest(regime, wing_mode="batched")
    loop = run_atomic_forest(regime, wing_mode="loop")
    left = batched.net_line_mass_absorption_coefficient
    right = loop.net_line_mass_absorption_coefficient
    absolute = np.abs(left.astype(np.float64) - right.astype(np.float64))
    scale = np.maximum(np.abs(right.astype(np.float64)), 1.0e-30)
    relative = absolute / scale
    return WingModeComparison(
        batched=left,
        loop=right,
        maximum_absolute_difference=float(np.max(absolute)),
        maximum_relative_difference=float(np.max(relative)),
    )


def route_ledger() -> tuple[tuple[object, ...], ...]:
    """Return the exact catalog codes and their production dispositions."""

    present = atomic_source_checkpoint().route_counts
    rows = (
        (0, "ordinary atomic", "metal Voigt/Harris", "ordinary transition"),
        (3, "PRD tag", "ordinary LTE metal", "ordinary LTE transition"),
        (1, "autoionizing", "Shore--Fano", "Shore profile"),
        (-1, "H I", "Balmer-and-higher Stark engine", "H profile, levels 1..100"),
        (-2, "D I", "same series engine, isotope shift", "source-dependent route"),
        (-3, "He I", "helium profile + merge taper", "source-dependent route"),
        (-4, "He-3", "helium profile + merge taper", "source-dependent route"),
        (-6, "He II", "helium profile + merge taper", "source-dependent route"),
        (2, "COR tag", "parsed, then skipped", "parsed, then skipped"),
    )
    return tuple((*row, int(present.get(int(row[0]), 0))) for row in rows)


def hydrogen_checkpoint() -> HydrogenCheckpoint:
    """Evaluate the exact H-alpha fine structure and Inglis--Teller merge rule."""

    configure_local_data_paths()
    from payne_zero_synthesis.hydrogen_lines import (
        _fine_structure,
        _hf_nm,
        merge_wavenumber_by_depth,
    )

    records = build_atomic_records()
    h_index = int(np.flatnonzero(records["line_type"] == -1)[0])
    d_index = int(np.flatnonzero(records["line_type"] == -2)[0])
    offsets, weights = _fine_structure(2, 3)
    electron_density = np.logspace(10.0, 16.0, 121)
    merge = merge_wavenumber_by_depth(electron_density)
    return HydrogenCheckpoint(
        hydrogen_wavelength_nm=float(records["wavelength_nm"][h_index]),
        deuterium_wavelength_nm=float(records["wavelength_nm"][d_index]),
        isotope_separation_nm=float(
            records["wavelength_nm"][h_index] - records["wavelength_nm"][d_index]
        ),
        component_offset_hz=np.asarray(offsets, dtype=np.float64),
        component_weight=np.asarray(weights, dtype=np.float64),
        oscillator_strength_2_to_3=float(_hf_nm(2, 3)),
        electron_density_cm3=electron_density,
        merge_wavenumber_cm=np.asarray(merge, dtype=np.float64),
        synthesis_minimum_supported_lower_level=2,
        atmosphere_supported_lower_level_range=(1, 100),
    )


def autoionizing_checkpoint() -> AutoionizingCheckpoint:
    """Reconstruct one exact positive Shore--Fano profile ratio."""

    raw = _raw_subset()
    records = build_atomic_records()
    auto_index = int(np.flatnonzero(records["line_type"] == 1)[0])
    radiative_log = float(raw["radiative_damping_log"][auto_index])
    stark_log = float(raw["stark_damping_log"][auto_index])
    van_der_waals_log = float(raw["van_der_waals_damping_log"][auto_index])
    radiative_width = 10.0**radiative_log
    shore_asymmetry = (
        -(10.0 ** (-stark_log)) if stark_log > 0 else 10.0**stark_log
    )
    shore_baseline = 10.0**van_der_waals_log
    reduced_offset = np.linspace(-20.0, 20.0, 801)
    raw_ratio = (
        (shore_asymmetry * reduced_offset + shore_baseline)
        / (reduced_offset * reduced_offset + 1.0)
        / shore_baseline
    )
    return AutoionizingCheckpoint(
        source_row=int(raw["source_row_index"][auto_index]),
        wavelength_nm=float(records["wavelength_nm"][auto_index]),
        radiative_width=float(radiative_width),
        shore_asymmetry=float(shore_asymmetry),
        shore_baseline=float(shore_baseline),
        reduced_frequency_offset=reduced_offset,
        positive_profile_ratio=np.where(raw_ratio > 0.0, raw_ratio, 0.0),
    )
