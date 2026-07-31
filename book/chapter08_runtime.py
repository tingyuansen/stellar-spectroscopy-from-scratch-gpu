"""Progressive Chapter 8 helpers for molecular source compilation and opacity.

All reader-facing inputs live in this repository.  The multi-gigabyte source
catalogs are used only by the optional subset builder; the notebook validates
the compact copies by their pinned identities and never imports from the
external Payne Zero checkout.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from book.chapter06_runtime import (
    SYNTHESIS_LINE_TABLES,
    configure_local_data_paths,
    synthesis_line_state,
)
from book.chapter08_teaching import compile_text_rows_scalar


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
WINDOW_START_NM = 498.95
WINDOW_END_NM = 499.15
RESOLUTION = 300_000.0
MOLECULAR_MANIFEST = (
    REPOSITORY_ROOT / "data/static/molecular_sources/manifest.json"
)
TEXT_SUBSET = REPOSITORY_ROOT / "data/subsets/chapter08_molecular_text_subset.npz"
TIO_SUBSET = REPOSITORY_ROOT / "data/subsets/chapter08_tio_subset.npy"
H2O_SUBSET = REPOSITORY_ROOT / "data/subsets/chapter08_h2o_subset.npy"
DIATOMIC_SUBSET = (
    REPOSITORY_ROOT
    / "data/subsets/chapter08_atmosphere_diatomic_subset.npy"
)
H3PLUS_PATH_PROBE = (
    REPOSITORY_ROOT / "data/fixtures/chapter08_h3plus_path_probe.npy"
)
PROVENANCE = (
    REPOSITORY_ROOT / "data/subsets/chapter08_molecular_provenance.json"
)
LOCAL_INPUT_HASHES = {
    MOLECULAR_MANIFEST: (
        "65131b3c07f093f062afed5547875b969da1e607dc46e3120332758d5e1c32c6"
    ),
    TEXT_SUBSET: "c264db732dcab4f29cb29be2395f3f6af4af28749706cf4f967205bf4e3feea5",
    TIO_SUBSET: "204c2aa286b173c7a8125e7aa67139155522f7594acb44eabc1adac11bb6ab13",
    H2O_SUBSET: "26bf33be3859dcd3ed601f820a88f7bf66fe7fc132a69ada0ad2aefba2b43409",
    DIATOMIC_SUBSET: (
        "ebc43f107b8e046cf6494eb587a4234fa791e2d85b4a05817cd1935f969e45db"
    ),
    H3PLUS_PATH_PROBE: (
        "6557eb0c8c7042ee4d0858ae528b5b7faf2408b744fd784c4e0acc2d5017a59f"
    ),
    PROVENANCE: "9794448f59b0e6d9ac5382ed21022d5a960193d9f947bebbfda9d6749524164d",
}
COMPILED_FIELDS = (
    "center_index_1based",
    "classical_line_strength",
    "species_code",
    "lower_excitation_cm",
    "radiative_damping",
    "stark_damping",
    "van_der_waals_damping",
    "margin_class",
)
PHYSICAL_FIELDS = COMPILED_FIELDS[:-1]
TEXT_FIELDS = (
    "stored_wavelength_nm",
    "log_oscillator_strength",
    "first_energy_cm",
    "second_energy_cm",
    "source_code",
    "isotope_index",
    "radiative_damping_log_scaled",
    "upper_label_is_ground_state",
)


@dataclass(frozen=True)
class MolecularSourceCheckpoint:
    band_count: int
    full_text_record_count: int
    full_tio_record_count: int
    full_h2o_record_count: int
    full_diatomic_record_count: int
    compact_text_record_count: int
    compact_tio_record_count: int
    compact_h2o_record_count: int
    compact_diatomic_record_count: int
    local_hashes: dict[str, str]


@dataclass(frozen=True)
class PopulationAndBandCheckpoint:
    molecule_names: tuple[str, ...]
    species_code: np.ndarray
    population_column: np.ndarray
    co_wavelength_nm: np.ndarray
    co_classical_line_strength: np.ndarray
    co_population_cm3: np.ndarray


@dataclass(frozen=True)
class TextRecordCheckpoint:
    source_text: str
    parsed_values: tuple
    stored_wavelength_nm: float
    energy_wavelength_nm: float
    fallback_wavelength_nm: float
    pair_dispatch_species_code: int
    code_fallback_species_code: int
    missing_dispatch_is_none: bool


@dataclass(frozen=True)
class TextCompilerCheckpoint:
    scalar_line_count: int
    compiled_line_count: int
    band_order_exact: bool
    field_exact: dict[str, bool]
    dtypes: dict[str, str]
    scalar_seconds: float
    first_compiled_seconds: float
    compiled_cache_hit_seconds: float


@dataclass(frozen=True)
class ManifestRugCheckpoint:
    band_name: tuple[str, ...]
    manifest_index: np.ndarray
    wavelength_nm: np.ndarray
    line_count_by_band: np.ndarray
    concatenation_exact: bool


@dataclass(frozen=True)
class PackedCompilerCheckpoint:
    tio_isotope_index: np.ndarray
    tio_isotope_fraction: np.ndarray
    tio_vacuum_wavelength_nm: np.ndarray
    tio_air_wavelength_nm: np.ndarray
    tio_line_count: int
    h2o_sign_labels: tuple[str, ...]
    h2o_isotope_index: np.ndarray
    h2o_isotope_fraction: np.ndarray
    h2o_line_count: int
    tio_air_changed_fields: tuple[str, ...]
    h2o_air_changed_fields: tuple[str, ...]
    text_strength_coefficient: float
    packed_strength_coefficient: float
    fractional_coefficient_difference: float


@dataclass(frozen=True)
class AtmosphereFamilyCheckpoint:
    family_names: tuple[str, ...]
    source_count: np.ndarray
    selected_count: np.ndarray
    offset_examples: dict[str, np.ndarray]
    water_packed_species: np.ndarray
    h3plus_probe_is_scientific_catalog: bool


@dataclass(frozen=True)
class CatalogCheckpoint:
    text_line_count: int
    tio_line_count: int
    combined_line_count: int
    combined_fields: tuple[str, ...]
    species_codes: np.ndarray
    species_population_columns: np.ndarray
    first_tio_row: int
    concatenation_exact: bool
    center_reconstruction_max_abs_nm: float


@dataclass(frozen=True)
class CacheCheckpoint:
    cache_file_count: int
    first_reload_exact: bool
    cached_reload_exact: bool
    corrupt_cache_rebuilt_exact: bool
    combined_persistent_cache_fingerprints_manifest: bool


@dataclass(frozen=True)
class InvariantCheckpoint:
    field_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    devices: tuple[str, ...]
    local_resolving_power_min: float
    local_resolving_power_max: float


@dataclass(frozen=True)
class DopplerCheckpoint:
    species_code: np.ndarray
    species_mass_amu: np.ndarray
    population_column: np.ndarray
    doppler_fraction: np.ndarray
    population_doppler_ratio: np.ndarray
    thermal_width_heavier_is_smaller: bool
    microturbulent_widths_converge: bool


@dataclass(frozen=True)
class SparseOracleCheckpoint:
    center_index_1based: np.ndarray
    dense_sum: np.ndarray
    sparse_sum: np.ndarray
    maximum_absolute_difference: float
    maximum_relative_difference: float
    collision_pixel: int


@dataclass(frozen=True)
class MolecularOpacityCheckpoint:
    wavelength_nm: np.ndarray
    text_opacity: np.ndarray
    tio_opacity: np.ndarray
    combined_opacity: np.ndarray
    gross_combined_opacity: np.ndarray
    separate_sum_max_abs: float
    chunk_regrouping_max_abs: float
    stimulation_ratio_max_abs: float
    population_scale: np.ndarray
    integrated_opacity: np.ndarray
    standard_h2o_line_count: int
    compiler_only_h2o_line_count: int
    depth_index: int


def sha256(path: Path) -> str:
    """Return a streaming SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_local_inputs() -> None:
    """Fail closed when any self-contained Chapter 8 input changes."""

    for path, expected in LOCAL_INPUT_HASHES.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Chapter 8 input changed: {path.relative_to(REPOSITORY_ROOT)}")


@lru_cache(maxsize=1)
def load_manifest() -> dict:
    validate_local_inputs()
    return json.loads(MOLECULAR_MANIFEST.read_text())


def band_names() -> tuple[str, ...]:
    return tuple(entry["band"] for entry in load_manifest()["text_sources"])


@contextmanager
def molecular_source_cache(directory: Path):
    """Temporarily isolate exact compiler-value caches in a disposable path."""

    key = "PAYNE_ZERO_SYNTHESIS_MOLECULAR_SOURCE_CACHE_DIR"
    old_value = os.environ.get(key)
    os.environ[key] = str(directory)
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def source_checkpoint() -> MolecularSourceCheckpoint:
    validate_local_inputs()
    provenance = json.loads(PROVENANCE.read_text())
    manifest = load_manifest()
    with np.load(TEXT_SUBSET, allow_pickle=False) as archive:
        compact_text_count = sum(
            int(archive[f"{name}/stored_wavelength_nm"].size)
            for name in band_names()
        )
    return MolecularSourceCheckpoint(
        band_count=len(manifest["text_sources"]),
        full_text_record_count=sum(
            int(entry["line_count"]) for entry in manifest["text_sources"]
        ),
        full_tio_record_count=37_744_499,
        full_h2o_record_count=65_912_356,
        full_diatomic_record_count=12_488_322,
        compact_text_record_count=compact_text_count,
        compact_tio_record_count=int(np.load(TIO_SUBSET, mmap_mode="r").shape[0]),
        compact_h2o_record_count=int(np.load(H2O_SUBSET, mmap_mode="r").shape[0]),
        compact_diatomic_record_count=int(
            np.load(DIATOMIC_SUBSET, mmap_mode="r").shape[0]
        ),
        local_hashes={
            path.relative_to(REPOSITORY_ROOT).as_posix(): sha256(path)
            for path in LOCAL_INPUT_HASHES
        },
    )


def _grid_metadata(
    start_wavelength_nm: float = WINDOW_START_NM,
    resolution: float = RESOLUTION,
) -> tuple[float, int]:
    log_grid_ratio = math.log(1.0 + 1.0 / float(resolution))
    grid_origin_index = math.floor(
        math.log(float(start_wavelength_nm)) / log_grid_ratio
    )
    if math.exp(grid_origin_index * log_grid_ratio) < start_wavelength_nm:
        grid_origin_index += 1
    return log_grid_ratio, int(grid_origin_index)


def _compile_text(
    selected_bands: tuple[str, ...],
    start_wavelength_nm: float,
    end_wavelength_nm: float,
) -> dict[str, np.ndarray]:
    configure_local_data_paths()
    from payne_zero_synthesis.source_catalog_molecular_compiler import (
        compile_molecular_text,
    )

    with tempfile.TemporaryDirectory(prefix="chapter08-text-cache-") as directory:
        with molecular_source_cache(Path(directory)):
            return compile_molecular_text(
                TEXT_SUBSET,
                selected_bands,
                start_wavelength_nm,
                end_wavelength_nm,
                RESOLUTION,
                use_energy_level_wavelengths=True,
                include_predicted_lines=False,
            )


@lru_cache(maxsize=1)
def _compiled_components_internal() -> tuple[dict, dict, dict]:
    configure_local_data_paths()
    from payne_zero_synthesis.source_catalog_molecular_compiler import (
        compile_h2o_partridge,
        compile_molecular_text,
        compile_tio_schwenke,
    )

    with tempfile.TemporaryDirectory(prefix="chapter08-compiler-cache-") as directory:
        with molecular_source_cache(Path(directory)):
            text = compile_molecular_text(
                TEXT_SUBSET,
                band_names(),
                WINDOW_START_NM,
                WINDOW_END_NM,
                RESOLUTION,
                use_energy_level_wavelengths=True,
                include_predicted_lines=False,
            )
            tio = compile_tio_schwenke(
                TIO_SUBSET,
                WINDOW_START_NM,
                WINDOW_END_NM,
                RESOLUTION,
                use_vacuum_wavelengths=True,
            )
            h2o = compile_h2o_partridge(
                H2O_SUBSET,
                400.0,
                1130.0,
                RESOLUTION,
                use_vacuum_wavelengths=True,
            )
    return text, tio, h2o


def compiled_components() -> tuple[dict, dict, dict]:
    """Return defensive copies of text, TiO, and compiler-only H2O arrays."""

    return tuple(
        {field: np.asarray(values).copy() for field, values in component.items()}
        for component in _compiled_components_internal()
    )


def _mapping_from_components(*components: dict) -> dict[str, np.ndarray]:
    log_grid_ratio, grid_origin_index = _grid_metadata()
    mapping = {
        field: np.concatenate([np.asarray(component[field]) for component in components])
        for field in PHYSICAL_FIELDS
    }
    mapping["log_grid_ratio"] = np.asarray(log_grid_ratio)
    mapping["grid_origin_index"] = np.asarray(grid_origin_index)
    return mapping


def _catalog_from_components(*components: dict):
    configure_local_data_paths()
    from payne_zero_synthesis.molecular_lines import build_catalog_from_arrays

    return build_catalog_from_arrays(_mapping_from_components(*components))


def _wavelength_grid() -> np.ndarray:
    configure_local_data_paths()
    from payne_zero_synthesis.atomic_lines import Grid

    return Grid(WINDOW_START_NM, WINDOW_END_NM, RESOLUTION).build()


def _cool_state(wavelength_nm: np.ndarray) -> dict[str, np.ndarray]:
    from book.chapter05_runtime import load_regime_state

    state, _ = synthesis_line_state("cool_molecule_rich", wavelength_nm)
    upstream = load_regime_state("cool_molecule_rich", "synthesis")
    state["microturbulence"] = np.asarray(
        upstream["microturbulence"], dtype=np.float64
    )
    return state


def population_and_band_checkpoint() -> PopulationAndBandCheckpoint:
    co = _compile_text(("co_xx_band",), 999.5, 1000.5)
    log_grid_ratio, grid_origin_index = _grid_metadata(999.5)
    wavelength = np.exp(
        (
            co["center_index_1based"].astype(np.int64)
            - 1
            + grid_origin_index
        )
        * log_grid_ratio
    )
    state = _cool_state(_wavelength_grid())
    codes = np.asarray([240, 276, 366, 534], dtype=np.int64)
    columns = codes // 6 - 1
    return PopulationAndBandCheckpoint(
        molecule_names=("H2", "CO", "TiO", "H2O"),
        species_code=codes,
        population_column=columns,
        co_wavelength_nm=wavelength,
        co_classical_line_strength=np.asarray(
            co["classical_line_strength"], dtype=np.float64
        ),
        co_population_cm3=np.asarray(
            state["partition_normalized_populations"][:, 5, 45],
            dtype=np.float64,
        ),
    )


def text_record_checkpoint() -> TextRecordCheckpoint:
    configure_local_data_paths()
    from payne_zero_synthesis.source_catalog_molecular_compiler import (
        _dispatch_molecule,
        _molecular_text_wavelength_nm,
        _parse_text_line,
    )

    source_text = (
        f"{1000.0:10.4f}{-2.0:7.3f}{0.0:5.1f}{100.0:10.3f}"
        f"{1.0:5.1f}{10100.0:11.3f}{608:4d}{'LOW':<8}{'UP':<8}{12:2d}{650:4d}"
    )
    parsed = _parse_text_line(source_text)
    stored = float(parsed[0])
    energy_wavelength = _molecular_text_wavelength_nm(
        stored, float(parsed[3]), float(parsed[5]), True
    )
    fallback = _molecular_text_wavelength_nm(stored, 100.0, 100.0, True)
    pair = _dispatch_molecule(608, 12)
    code_fallback = _dispatch_molecule(608, 999)
    return TextRecordCheckpoint(
        source_text=source_text,
        parsed_values=parsed,
        stored_wavelength_nm=stored,
        energy_wavelength_nm=float(energy_wavelength),
        fallback_wavelength_nm=float(fallback),
        pair_dispatch_species_code=int(pair[0]),
        code_fallback_species_code=int(code_fallback[0]),
        missing_dispatch_is_none=_dispatch_molecule(9999, 999) is None,
    )


def _scalar_compile_all_bands() -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {field: [] for field in COMPILED_FIELDS}
    with np.load(TEXT_SUBSET, allow_pickle=False) as archive:
        for band in band_names():
            arrays = {
                field: np.asarray(archive[f"{band}/{field}"])
                for field in TEXT_FIELDS
            }
            compiled = compile_text_rows_scalar(
                **arrays,
                start_wavelength_nm=WINDOW_START_NM,
                end_wavelength_nm=WINDOW_END_NM,
                resolution=RESOLUTION,
                use_energy_level_wavelengths=True,
                include_predicted_lines=False,
            )
            for field in COMPILED_FIELDS:
                chunks[field].append(compiled[field])
    return {
        field: np.concatenate(values) if values else np.zeros(0)
        for field, values in chunks.items()
    }


@lru_cache(maxsize=1)
def text_compiler_checkpoint() -> TextCompilerCheckpoint:
    configure_local_data_paths()
    start = perf_counter()
    scalar = _scalar_compile_all_bands()
    scalar_seconds = perf_counter() - start

    from payne_zero_synthesis.source_catalog_molecular_compiler import (
        compile_molecular_text,
    )

    with tempfile.TemporaryDirectory(prefix="chapter08-timing-cache-") as directory:
        with molecular_source_cache(Path(directory)):
            start = perf_counter()
            compiled = compile_molecular_text(
                TEXT_SUBSET,
                band_names(),
                WINDOW_START_NM,
                WINDOW_END_NM,
                RESOLUTION,
                use_energy_level_wavelengths=True,
                include_predicted_lines=False,
            )
            first_seconds = perf_counter() - start
            start = perf_counter()
            cached = compile_molecular_text(
                TEXT_SUBSET,
                band_names(),
                WINDOW_START_NM,
                WINDOW_END_NM,
                RESOLUTION,
                use_energy_level_wavelengths=True,
                include_predicted_lines=False,
            )
            cache_hit_seconds = perf_counter() - start

    exact = {
        field: bool(np.array_equal(scalar[field], compiled[field]))
        for field in COMPILED_FIELDS
    }
    return TextCompilerCheckpoint(
        scalar_line_count=int(scalar["species_code"].size),
        compiled_line_count=int(compiled["species_code"].size),
        band_order_exact=all(
            np.array_equal(compiled[field], cached[field])
            for field in COMPILED_FIELDS
        ),
        field_exact=exact,
        dtypes={field: str(compiled[field].dtype) for field in COMPILED_FIELDS},
        scalar_seconds=float(scalar_seconds),
        first_compiled_seconds=float(first_seconds),
        compiled_cache_hit_seconds=float(cache_hit_seconds),
    )


@lru_cache(maxsize=1)
def manifest_rug_checkpoint() -> ManifestRugCheckpoint:
    all_wavelengths: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    counts: list[int] = []
    per_band_components: list[dict] = []
    log_grid_ratio, grid_origin_index = _grid_metadata()
    for manifest_index, band in enumerate(band_names()):
        component = _compile_text((band,), WINDOW_START_NM, WINDOW_END_NM)
        per_band_components.append(component)
        count = int(component["species_code"].size)
        counts.append(count)
        wavelength = np.exp(
            (
                component["center_index_1based"].astype(np.int64)
                - 1
                + grid_origin_index
            )
            * log_grid_ratio
        )
        all_wavelengths.append(wavelength)
        all_indices.append(np.full(count, manifest_index, dtype=np.int64))
    combined = _compiled_components_internal()[0]
    concatenation_exact = all(
        np.array_equal(
            combined[field],
            np.concatenate([component[field] for component in per_band_components]),
        )
        for field in COMPILED_FIELDS
    )
    return ManifestRugCheckpoint(
        band_name=band_names(),
        manifest_index=(
            np.concatenate(all_indices) if all_indices else np.zeros(0, np.int64)
        ),
        wavelength_nm=(
            np.concatenate(all_wavelengths)
            if all_wavelengths
            else np.zeros(0, np.float64)
        ),
        line_count_by_band=np.asarray(counts, dtype=np.int64),
        concatenation_exact=bool(concatenation_exact),
    )


def packed_compiler_checkpoint() -> PackedCompilerCheckpoint:
    configure_local_data_paths()
    from payne_zero_synthesis.constants import CLASSICAL_LINE_STRENGTH_COEFFICIENT
    from payne_zero_synthesis.source_catalog_molecular_compiler import (
        _H2O_ISOTOPE_FRACTIONS,
        _TIO_FINE_LOG_GRID,
        _TIO_ISOTOPE_FRACTIONS,
        _vacair,
        compile_h2o_partridge,
        compile_tio_schwenke,
    )

    tio_records = np.load(TIO_SUBSET, mmap_mode="r")
    tio_index = (
        np.abs(tio_records["isotope_species_code"].astype(np.int32)) - 8949
    )
    vacuum = np.exp(
        tio_records["wavelength_code"].astype(np.float64) * _TIO_FINE_LOG_GRID
    )
    air = np.asarray([_vacair(float(value)) for value in vacuum], dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="chapter08-tio-air-") as directory:
        with molecular_source_cache(Path(directory)):
            tio_air = compile_tio_schwenke(
                TIO_SUBSET,
                WINDOW_START_NM,
                WINDOW_END_NM,
                RESOLUTION,
                use_vacuum_wavelengths=False,
            )
            h2o_air = compile_h2o_partridge(
                H2O_SUBSET,
                400.0,
                1130.0,
                RESOLUTION,
                use_vacuum_wavelengths=False,
            )

    water = np.load(H2O_SUBSET, mmap_mode="r")
    energy = water["signed_lower_energy_code"].astype(np.int32)
    strength = water["signed_log_oscillator_strength_code"].astype(np.int32)
    water_index = np.where(
        (energy > 0) & (strength > 0),
        0,
        np.where(energy > 0, 1, np.where(strength > 0, 2, 3)),
    )
    _, tio, h2o = compiled_components()
    tio_changed_fields = tuple(
        field
        for field in COMPILED_FIELDS
        if not np.array_equal(tio[field], tio_air[field])
    )
    h2o_changed_fields = tuple(
        field
        for field in COMPILED_FIELDS
        if not np.array_equal(h2o[field], h2o_air[field])
    )
    packed_coefficient = 0.01502
    text_coefficient = float(CLASSICAL_LINE_STRENGTH_COEFFICIENT)
    return PackedCompilerCheckpoint(
        tio_isotope_index=tio_index,
        tio_isotope_fraction=np.asarray(_TIO_ISOTOPE_FRACTIONS, dtype=np.float64),
        tio_vacuum_wavelength_nm=vacuum,
        tio_air_wavelength_nm=air,
        tio_line_count=int(tio["species_code"].size),
        h2o_sign_labels=("+,+", "+,<=0", "<=0,+", "<=0,<=0"),
        h2o_isotope_index=water_index,
        h2o_isotope_fraction=np.asarray(
            _H2O_ISOTOPE_FRACTIONS, dtype=np.float64
        ),
        h2o_line_count=int(h2o["species_code"].size),
        tio_air_changed_fields=tio_changed_fields,
        h2o_air_changed_fields=h2o_changed_fields,
        text_strength_coefficient=text_coefficient,
        packed_strength_coefficient=packed_coefficient,
        fractional_coefficient_difference=(
            packed_coefficient / text_coefficient - 1.0
        ),
    )


@lru_cache(maxsize=1)
def atmosphere_family_checkpoint() -> AtmosphereFamilyCheckpoint:
    configure_local_data_paths()
    from payne_zero_atmosphere.line_profile_math import build_selection_log_lookup
    from payne_zero_atmosphere.line_selection import (
        _diatomic_log_strength_offsets,
        _titanium_oxide_log_strength_offsets,
        _water_line_derived_columns,
        read_diatomic_line_catalog,
        read_standard_line_catalog,
        read_water_line_catalog,
        select_standard_line_words,
        select_water_line_words,
    )

    diatomic = read_diatomic_line_catalog(DIATOMIC_SUBSET)
    tio = read_standard_line_catalog(TIO_SUBSET)
    water = read_water_line_catalog(H2O_SUBSET)
    h3probe = read_standard_line_catalog(H3PLUS_PATH_PROBE)
    all_codes = np.concatenate((diatomic[:, 0], tio[:, 0], water[:, 0]))
    packed_bins = np.linspace(
        int(all_codes.min()) - 2,
        int(all_codes.max()) + 2,
        64,
        dtype=np.int64,
    )
    step = math.log(1.0 + 1.0 / 2_000_000.0)
    frequency_per_bin = (
        2.99792458e17 / np.exp(packed_bins.astype(np.float64) * step)
    ).astype(np.float32)
    ratio_max = np.full((1006, packed_bins.size), 1.0e30, dtype=np.float32)
    lookup = build_selection_log_lookup()
    diatomic_offsets = _diatomic_log_strength_offsets(diatomic)
    tio_offsets = _titanium_oxide_log_strength_offsets(tio)
    selected_diatomic, _ = select_standard_line_words(
        diatomic,
        packed_continuum_wavelengths=packed_bins,
        selection_log_lookup=lookup,
        doppler_population_ratio_max=ratio_max,
        deepest_hc_over_kt=0.0,
        frequency_per_bin=frequency_per_bin,
        log_strength_offsets=diatomic_offsets,
        stark_damping_override=1,
    )
    selected_tio, _ = select_standard_line_words(
        tio,
        packed_continuum_wavelengths=packed_bins,
        selection_log_lookup=lookup,
        doppler_population_ratio_max=ratio_max,
        deepest_hc_over_kt=0.0,
        frequency_per_bin=frequency_per_bin,
        log_strength_offsets=tio_offsets,
        population_slot_override=895,
        stark_damping_override=1,
        van_der_waals_damping_override=9384,
    )
    selected_water = select_water_line_words(
        water,
        packed_continuum_wavelengths=packed_bins,
        selection_log_lookup=lookup,
        doppler_population_ratio_max=ratio_max,
        deepest_hc_over_kt=0.0,
        frequency_per_bin=frequency_per_bin,
    )
    selected_h3, _ = select_standard_line_words(
        h3probe,
        packed_continuum_wavelengths=packed_bins,
        selection_log_lookup=lookup,
        doppler_population_ratio_max=ratio_max,
        deepest_hc_over_kt=0.0,
        frequency_per_bin=frequency_per_bin,
        log_strength_offsets=np.full(h3probe.shape[0], -1272, dtype=np.int32),
        population_slot_override=895,
        stark_damping_override=1,
        van_der_waals_damping_override=9384,
    )
    water_species, _, water_strength = _water_line_derived_columns(water)
    return AtmosphereFamilyCheckpoint(
        family_names=("diatomic", "TiO", "H2O", "H3+ path probe"),
        source_count=np.asarray(
            [len(diatomic), len(tio), len(water), len(h3probe)], dtype=np.int64
        ),
        selected_count=np.asarray(
            [
                len(selected_diatomic),
                len(selected_tio),
                len(selected_water),
                len(selected_h3),
            ],
            dtype=np.int64,
        ),
        offset_examples={
            "diatomic": diatomic_offsets[:8].copy(),
            "TiO": np.unique(tio_offsets),
            "H2O": np.unique(water_strength - np.abs(water[:, 2])),
            "H3+": np.asarray([-1272], dtype=np.int32),
        },
        water_packed_species=np.unique(water_species),
        h3plus_probe_is_scientific_catalog=False,
    )


def feature_status_rows() -> tuple[dict[str, object], ...]:
    """Separate implementation availability from standard runtime wiring."""

    return (
        {
            "family": "text / diatomic",
            "atmosphere_selector_exists": True,
            "atmosphere_default_source": True,
            "synthesis_compiler_exists": True,
            "synthesis_standard_deposits": True,
        },
        {
            "family": "TiO",
            "atmosphere_selector_exists": True,
            "atmosphere_default_source": True,
            "synthesis_compiler_exists": True,
            "synthesis_standard_deposits": True,
        },
        {
            "family": "H2O",
            "atmosphere_selector_exists": True,
            "atmosphere_default_source": True,
            "synthesis_compiler_exists": True,
            "synthesis_standard_deposits": False,
        },
        {
            "family": "H3+",
            "atmosphere_selector_exists": True,
            "atmosphere_default_source": False,
            "synthesis_compiler_exists": False,
            "synthesis_standard_deposits": False,
        },
    )


def catalog_checkpoint() -> CatalogCheckpoint:
    text, tio, _ = compiled_components()
    catalog = _catalog_from_components(text, tio)
    expected = _mapping_from_components(text, tio)
    reconstructed = np.exp(
        (
            catalog.center_index_1based.astype(np.int64)
            - 1
            + catalog.grid_origin_index
        )
        * catalog.log_grid_ratio
    )
    return CatalogCheckpoint(
        text_line_count=int(text["species_code"].size),
        tio_line_count=int(tio["species_code"].size),
        combined_line_count=len(catalog),
        combined_fields=(
            "center_index_1based",
            "classical_line_strength",
            "species_code",
            "lower_excitation_cm",
            "radiative_damping",
            "stark_damping",
            "van_der_waals_damping",
            "center_index",
            "species_population_column",
            "wavelength_nm",
            "log_grid_ratio",
            "grid_origin_index",
            "unique_species_codes",
        ),
        species_codes=catalog.unique_species_codes.copy(),
        species_population_columns=(
            catalog.unique_species_codes.astype(np.int64) // 6 - 1
        ),
        first_tio_row=int(text["species_code"].size),
        concatenation_exact=all(
            np.array_equal(
                np.asarray(getattr(catalog, field)),
                np.asarray(expected[field]),
            )
            for field in PHYSICAL_FIELDS
        ),
        center_reconstruction_max_abs_nm=float(
            np.max(
                np.abs(
                    reconstructed
                    - np.asarray(catalog.wavelength_nm, dtype=np.float64)
                )
            )
        ),
    )


def _catalog_fields_equal(left, right) -> bool:
    return all(
        np.array_equal(getattr(left, field), getattr(right, field))
        for field in left._SERIALIZED_FIELDS
    ) and np.array_equal(left.unique_species_codes, right.unique_species_codes)


def cache_checkpoint() -> CacheCheckpoint:
    configure_local_data_paths()
    from payne_zero_synthesis.molecular_lines import load_catalog

    text, tio, _ = compiled_components()
    mapping = _mapping_from_components(text, tio)
    expected = _catalog_from_components(text, tio)
    with tempfile.TemporaryDirectory(prefix="chapter08-index-cache-") as directory:
        root = Path(directory)
        source_path = root / "compiled_window.npz"
        cache_dir = root / "derived"
        np.savez(source_path, **mapping)
        first = load_catalog(source_path, cache_dir=cache_dir)
        cached = load_catalog(source_path, cache_dir=cache_dir)
        cache_files = list(cache_dir.glob("molecular_lines_*.npz"))
        if len(cache_files) != 1:
            raise RuntimeError("derived catalog cache did not create one file")
        cache_files[0].write_bytes(b"not an npz")
        rebuilt = load_catalog(source_path, cache_dir=cache_dir)
        return CacheCheckpoint(
            cache_file_count=len(cache_files),
            first_reload_exact=_catalog_fields_equal(expected, first),
            cached_reload_exact=_catalog_fields_equal(expected, cached),
            corrupt_cache_rebuilt_exact=_catalog_fields_equal(expected, rebuilt),
            combined_persistent_cache_fingerprints_manifest=False,
        )


def _line_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SYNTHESIS_LINE_TABLES, allow_pickle=False) as archive:
        return tuple(
            np.asarray(archive[field], dtype=np.float64)
            for field in (
                "harris_profile_h0_table",
                "harris_profile_h1_table",
                "harris_profile_h2_table",
            )
        )


def _invariants_for_components(*components: dict):
    configure_local_data_paths()
    import torch
    from payne_zero_synthesis.molecular_lines import precompute_invariants

    return precompute_invariants(
        _catalog_from_components(*components),
        _wavelength_grid(),
        *_line_tables(),
        runtime_device=torch.device("cpu"),
    )


def invariant_checkpoint() -> InvariantCheckpoint:
    text, tio, _ = compiled_components()
    invariants = _invariants_for_components(text, tio)
    field_names = tuple(invariants.__dataclass_fields__)
    tensors = [getattr(invariants, field) for field in field_names]
    return InvariantCheckpoint(
        field_names=field_names,
        shapes=tuple(
            tuple(value.shape) if hasattr(value, "shape") else ()
            for value in tensors
        ),
        dtypes=tuple(
            str(value.dtype) if hasattr(value, "dtype") else type(value).__name__
            for value in tensors
        ),
        devices=tuple(
            str(value.device) if hasattr(value, "device") else "host scalar"
            for value in tensors
        ),
        local_resolving_power_min=float(
            invariants.local_resolving_power.min().item()
        ),
        local_resolving_power_max=float(
            invariants.local_resolving_power.max().item()
        ),
    )


def doppler_checkpoint() -> DopplerCheckpoint:
    configure_local_data_paths()
    import torch
    from payne_zero_synthesis.constants import (
        ATOMIC_MASS_GRAM,
        BOLTZMANN_ERG_PER_K,
        LIGHT_SPEED_CM_PER_S,
    )
    from payne_zero_synthesis.molecular_lines import (
        species_population_doppler_ratio,
    )

    text, tio, _ = compiled_components()
    invariants = _invariants_for_components(text, tio)
    state = _cool_state(_wavelength_grid())
    ratio, doppler = species_population_doppler_ratio(invariants, state)
    temperature = float(state["temperature"][0])
    thermal = np.sqrt(
        2.0
        * BOLTZMANN_ERG_PER_K
        * temperature
        / (np.asarray([2.0, 64.0]) * ATOMIC_MASS_GRAM)
    ) / LIGHT_SPEED_CM_PER_S
    turbulent = np.sqrt(
        (
            2.0
            * BOLTZMANN_ERG_PER_K
            * temperature
            / (np.asarray([2.0, 64.0]) * ATOMIC_MASS_GRAM)
        )
        + (1.0e8) ** 2
    ) / LIGHT_SPEED_CM_PER_S
    return DopplerCheckpoint(
        species_code=invariants.species_code.cpu().numpy(),
        species_mass_amu=invariants.species_mass_amu.cpu().numpy(),
        population_column=invariants.species_population_column.cpu().numpy(),
        doppler_fraction=doppler.cpu().numpy(),
        population_doppler_ratio=ratio.cpu().numpy(),
        thermal_width_heavier_is_smaller=bool(thermal[1] < thermal[0]),
        microturbulent_widths_converge=bool(
            abs(turbulent[0] - turbulent[1]) / turbulent[0] < 1.0e-4
        ),
    )


def _run_opacity(
    components: tuple[dict, ...],
    state: dict[str, np.ndarray],
    *,
    apply_stim: bool,
    chunk_lines: int,
) -> np.ndarray:
    configure_local_data_paths()
    from payne_zero_synthesis.molecular_lines import accumulate_molecular

    invariants = _invariants_for_components(*components)
    result = accumulate_molecular(
        invariants,
        state,
        apply_stim=apply_stim,
        chunk_lines=chunk_lines,
    )
    return result.detach().cpu().numpy()


@lru_cache(maxsize=1)
def sparse_oracle_checkpoint() -> SparseOracleCheckpoint:
    text, _, _ = compiled_components()
    collision_center = int(text["center_index_1based"][0])
    collision_rows = np.flatnonzero(
        text["center_index_1based"] == collision_center
    )
    order = collision_rows[
        np.argsort(text["classical_line_strength"][collision_rows])[::-1]
    ]
    selected_rows = order[:2]
    if selected_rows.size != 2:
        raise RuntimeError("compact molecular subset lost its center collision")
    tiny = {field: text[field][selected_rows] for field in COMPILED_FIELDS}
    state = _cool_state(_wavelength_grid())
    individual: list[np.ndarray] = []
    for index in range(2):
        one = {field: values[index : index + 1] for field, values in tiny.items()}
        individual.append(
            _run_opacity((one,), state, apply_stim=False, chunk_lines=1)
        )
    dense_sum = np.stack(individual, axis=1).sum(axis=1, dtype=np.float32)
    sparse_sum = _run_opacity(
        (tiny,), state, apply_stim=False, chunk_lines=2
    )
    difference = np.abs(dense_sum.astype(np.float64) - sparse_sum.astype(np.float64))
    scale = np.maximum(np.abs(dense_sum.astype(np.float64)), 1.0e-30)
    return SparseOracleCheckpoint(
        center_index_1based=tiny["center_index_1based"].copy(),
        dense_sum=dense_sum,
        sparse_sum=sparse_sum,
        maximum_absolute_difference=float(difference.max()),
        maximum_relative_difference=float((difference / scale).max()),
        collision_pixel=collision_center - 1,
    )


@lru_cache(maxsize=1)
def molecular_opacity_checkpoint() -> MolecularOpacityCheckpoint:
    configure_local_data_paths()
    from payne_zero_synthesis.constants import (
        BOLTZMANN_ERG_PER_K,
        LIGHT_SPEED_NM_PER_S,
        PLANCK_ERG_SECOND,
    )

    text, tio, h2o = compiled_components()
    wavelength = _wavelength_grid()
    state = _cool_state(wavelength)
    text_opacity = _run_opacity(
        (text,), state, apply_stim=True, chunk_lines=31
    )
    tio_opacity = _run_opacity(
        (tio,), state, apply_stim=True, chunk_lines=31
    )
    combined = _run_opacity(
        (text, tio), state, apply_stim=True, chunk_lines=500_000
    )
    regrouped = _run_opacity(
        (text, tio), state, apply_stim=True, chunk_lines=17
    )
    gross = _run_opacity(
        (text, tio), state, apply_stim=False, chunk_lines=500_000
    )
    temperature = np.asarray(state["temperature"], dtype=np.float64)
    frequency = LIGHT_SPEED_NM_PER_S / wavelength
    stimulation = 1.0 - np.exp(
        -PLANCK_ERG_SECOND
        * frequency[None, :]
        / (BOLTZMANN_ERG_PER_K * temperature[:, None])
    )
    expected_net = (
        gross.astype(np.float64) * stimulation
    ).astype(np.float32)
    scales = np.asarray([0.25, 0.5, 1.0, 2.0], dtype=np.float64)
    integrated: list[float] = []
    for scale in scales:
        scaled_state = {
            key: np.asarray(value).copy()
            for key, value in state.items()
        }
        scaled_state["partition_normalized_populations"][:, 5, :] *= scale
        opacity = _run_opacity(
            (text, tio),
            scaled_state,
            apply_stim=True,
            chunk_lines=500_000,
        )
        integrated.append(float(np.trapezoid(opacity[2], wavelength)))
    return MolecularOpacityCheckpoint(
        wavelength_nm=wavelength,
        text_opacity=text_opacity,
        tio_opacity=tio_opacity,
        combined_opacity=combined,
        gross_combined_opacity=gross,
        separate_sum_max_abs=float(
            np.max(
                np.abs(
                    text_opacity.astype(np.float64)
                    + tio_opacity.astype(np.float64)
                    - combined.astype(np.float64)
                )
            )
        ),
        chunk_regrouping_max_abs=float(
            np.max(
                np.abs(
                    regrouped.astype(np.float64)
                    - combined.astype(np.float64)
                )
            )
        ),
        stimulation_ratio_max_abs=float(
            np.max(
                np.abs(
                    expected_net.astype(np.float64)
                    - combined.astype(np.float64)
                )
            )
        ),
        population_scale=scales,
        integrated_opacity=np.asarray(integrated, dtype=np.float64),
        standard_h2o_line_count=0,
        compiler_only_h2o_line_count=int(h2o["species_code"].size),
        depth_index=2,
    )
