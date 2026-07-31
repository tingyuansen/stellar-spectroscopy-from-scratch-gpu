"""Chapter 10 integration helpers for the complete synthesis pipeline.

The helpers assemble a disposable source-catalog view from manifest-owned
Chapter 7/8 subsets, then call the exact staged Payne Zero implementation.
They do not read the external Payne Zero checkout and do not implement a
second synthesis pipeline.
"""

from __future__ import annotations

import atexit
from contextlib import redirect_stdout
from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import inspect
import io
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from time import perf_counter

import numpy as np

from book.chapter10_teaching import allocation_ledger


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
WINDOW_START_NM = 468.6
WINDOW_END_NM = 656.6
RESOLUTION = 20_000.0
ZOOM_START_NM = 498.95
ZOOM_END_NM = 499.15
REGIMES = (
    "hot_dwarf",
    "solar_dwarf",
    "low_gravity_giant",
    "cool_molecule_rich",
)
REGIME_LABELS = {
    "hot_dwarf": "hot dwarf",
    "solar_dwarf": "solar-like dwarf",
    "low_gravity_giant": "low-gravity giant",
    "cool_molecule_rich": "cool molecule-rich",
}

ATMOSPHERE_FIXTURE = (
    REPOSITORY_ROOT / "data/fixtures/chapter05_continuum_states.npz"
)
ATOMIC_SUBSET = (
    REPOSITORY_ROOT / "data/subsets/chapter07_atomic_catalog_subset.npz"
)
MOLECULAR_MANIFEST = (
    REPOSITORY_ROOT / "data/static/molecular_sources/manifest.json"
)
MOLECULAR_TEXT_SUBSET = (
    REPOSITORY_ROOT / "data/subsets/chapter08_molecular_text_subset.npz"
)
TIO_SUBSET = REPOSITORY_ROOT / "data/subsets/chapter08_tio_subset.npy"
MOLECULAR_EQUILIBRIUM = (
    REPOSITORY_ROOT
    / "data/static/source_catalogs/lines/molecular_equilibrium_synthesis.npz"
)
SYNTHESIS_EOS_STATE = (
    REPOSITORY_ROOT / "data/fixtures/chapter03_synthesis_eos_state.npz"
)

INPUT_HASHES = {
    ATMOSPHERE_FIXTURE: (
        "ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae"
    ),
    ATOMIC_SUBSET: (
        "d797e747d7f557d172505bbe546c0d025dbc2c7a4e0cce831a8bdbec94573e23"
    ),
    MOLECULAR_TEXT_SUBSET: (
        "c264db732dcab4f29cb29be2395f3f6af4af28749706cf4f967205bf4e3feea5"
    ),
    TIO_SUBSET: (
        "204c2aa286b173c7a8125e7aa67139155522f7594acb44eabc1adac11bb6ab13"
    ),
    SYNTHESIS_EOS_STATE: (
        "ecc2856d69c7d96bcdfb6d50988addcd06c68f6e492d9c1f08d21492984ad6c9"
    ),
}

STAGE_ORDER = (
    "continuum absorption + scattering",
    "float32 shared line slab",
    "ordinary/autoionizing metal chunks",
    "helium",
    "hydrogen with star-specific merge state",
    "molecular text + TiO",
    "LTE Planck line source + zero line scattering",
    "total/continuum transfer",
    "device-side requested-grid crop",
    "final host result construction",
)

_SOURCE_VIEW_OWNER: tempfile.TemporaryDirectory | None = None
_SOURCE_VIEW_ROOT: Path | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cleanup_source_view() -> None:
    global _SOURCE_VIEW_OWNER, _SOURCE_VIEW_ROOT
    if _SOURCE_VIEW_OWNER is not None:
        _SOURCE_VIEW_OWNER.cleanup()
    _SOURCE_VIEW_OWNER = None
    _SOURCE_VIEW_ROOT = None


atexit.register(_cleanup_source_view)


def _replace_default_argument(callable_object, name: str, value: Path) -> bool:
    """Replace one already-imported path default without replacing its API.

    Payne Zero resolves several packaged paths when a function is defined.
    A preceding chapter can therefore import the exact staged module while a
    different ``PAYNE_ZERO_DATA_ROOT`` is active.  Updating the environment
    later is insufficient because Python has already stored the old ``Path``
    inside ``__defaults__`` or ``__kwdefaults__``.

    Mutating the existing function preserves the public function and class
    objects held by earlier imports; only the stale path value is rebound.
    """

    function = getattr(callable_object, "__func__", callable_object)
    signature = inspect.signature(function)
    parameter = signature.parameters[name]
    if parameter.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        positional = [
            item
            for item in signature.parameters.values()
            if item.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        defaults = list(function.__defaults__ or ())
        default_start = len(positional) - len(defaults)
        default_index = positional.index(parameter) - default_start
        if default_index < 0:
            raise RuntimeError(f"{name} is not a defaulted parameter")
        changed = defaults[default_index] != value
        defaults[default_index] = value
        function.__defaults__ = tuple(defaults)
        return changed
    if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
        keyword_defaults = dict(function.__kwdefaults__ or {})
        changed = keyword_defaults.get(name) != value
        keyword_defaults[name] = value
        function.__kwdefaults__ = keyword_defaults
        return changed
    raise RuntimeError(f"{name} cannot carry a path default")


def _rebind_imported_runtime_paths(
    *,
    data_view: Path,
    cache_root: Path,
) -> None:
    """Rebind import-time paths in staged modules already present in-process.

    Clean imports need no intervention: they see the configured environment
    directly.  This narrow compatibility step exists for notebook/test
    processes that imported a staged package under an earlier chapter's data
    root.  It deliberately preserves module, class, and function identities.
    """

    synthesis_table_dir = data_view / "synthesis_tables"
    atmosphere_table_dir = data_view / "atmosphere_tables"
    atmosphere_emulator_dir = data_view / "atmosphere_emulator"

    synthesis_paths = sys.modules.get("payne_zero_synthesis.paths")
    synthesis_paths_changed = False
    if synthesis_paths is not None:
        synthesis_paths_changed = (
            synthesis_paths.SYNTHESIS_TABLE_DIR != synthesis_table_dir
            or synthesis_paths.PACKAGE_CACHE_ROOT != cache_root
        )
        synthesis_paths.SYNTHESIS_TABLE_DIR = synthesis_table_dir
        synthesis_paths.PACKAGE_CACHE_ROOT = cache_root

    eos = sys.modules.get("payne_zero_synthesis.equation_of_state")
    if eos is not None:
        eos_inputs = synthesis_table_dir / "partition_saha_inputs.npz"
        eos._DEFAULT_INPUTS = eos_inputs
        synthesis_paths_changed |= _replace_default_argument(
            eos.EOSTables.from_npz, "path", eos_inputs
        )
        synthesis_paths_changed |= _replace_default_argument(
            eos.load_state_from_inputs, "path", eos_inputs
        )

    continuum = sys.modules.get("payne_zero_synthesis.continuum")
    if continuum is not None:
        continuum_tables = synthesis_table_dir / "continuum_tables.npz"
        continuum._DEFAULT_CONTINUUM_TABLES = continuum_tables
        synthesis_paths_changed |= _replace_default_argument(
            continuum.ContinuumTables.from_npz, "path", continuum_tables
        )

    atomic_lines = sys.modules.get("payne_zero_synthesis.atomic_lines")
    if atomic_lines is not None:
        ionization_table = (
            synthesis_table_dir / "ionization_potential_lookup.npz"
        )
        if atomic_lines._IONIZATION_POTENTIAL_TABLE_PATH != ionization_table:
            synthesis_paths_changed = True
            atomic_lines._IONIZATION_POTENTIAL_TABLE_PATH = ionization_table
            atomic_lines._IONIZATION_POTENTIAL_TABLE = None

    pipeline = sys.modules.get("payne_zero_synthesis.pipeline")
    if pipeline is not None:
        line_tables = synthesis_table_dir / "line_profile_tables.npz"
        transfer_tables = synthesis_table_dir / "transfer_tables.npz"
        continuum_tables = synthesis_table_dir / "continuum_tables.npz"
        edge_table = synthesis_table_dir / "continuum_edge_grid.npz"
        atomic_mass_table = synthesis_table_dir / "atomic_masses.npz"
        pipeline._SYNTHESIS_TABLE_DIR = synthesis_table_dir
        pipeline._CONTINUUM_EDGE_TABLE = edge_table
        pipeline._ATOMIC_MASS_TABLE = atomic_mass_table
        synthesis_paths_changed |= _replace_default_argument(
            pipeline._build_edge_grid, "edge_table_path", edge_table
        )
        for callable_object in (
            pipeline.SynthesisPipeline.__init__,
            pipeline.window_invariants_for,
        ):
            synthesis_paths_changed |= _replace_default_argument(
                callable_object, "tables_path", line_tables
            )
            synthesis_paths_changed |= _replace_default_argument(
                callable_object, "transfer_tables_path", transfer_tables
            )
            synthesis_paths_changed |= _replace_default_argument(
                callable_object, "continuum_tables_path", continuum_tables
            )
        if synthesis_paths_changed:
            pipeline.clear_window_invariant_cache()
            if hasattr(pipeline._build_edge_grid, "_cache"):
                del pipeline._build_edge_grid._cache

    # Chapter 15 also uses atmosphere-side defaults.  Rebind them to the same
    # disposable data view so a prior atmosphere import cannot retain another
    # root's emulator or packed-table paths.
    atmosphere_paths_changed = False
    atmosphere_path_bindings = {
        "payne_zero_atmosphere.runtime_state": {
            "_ISOTOPE_TABLE_PATH": atmosphere_table_dir / "isotope_tables.npz",
        },
        "payne_zero_atmosphere.molecular_equilibrium": {
            "_MOLECULAR_TABLE_PATH": (
                atmosphere_table_dir / "molecular_equilibrium_tables.npz"
            ),
        },
        "payne_zero_atmosphere.hydrogen_line_profile": {
            "_DEFAULT_TABLE_PATH": (
                atmosphere_table_dir / "hydrogen_line_profile_tables.npz"
            ),
        },
        "payne_zero_atmosphere.line_profile_math": {
            "_DEFAULT_TABLE_PATH": (
                atmosphere_table_dir / "line_opacity_tables.npz"
            ),
        },
        "payne_zero_atmosphere.radiative_transfer": {
            "_DEFAULT_TRANSFER_TABLE_PATH": (
                atmosphere_table_dir / "radiative_transfer_tables.npz"
            ),
        },
        "payne_zero_atmosphere.synthesis_bridge": {
            "_DEFAULT_EDGE_GRID": (
                synthesis_table_dir / "continuum_edge_grid.npz"
            ),
        },
        "payne_zero_atmosphere.source_catalogs": {
            "BUNDLED_SOURCE_CATALOG_ROOT": data_view / "source_catalogs",
            "SOURCE_CATALOG_CHECKSUMS": (
                data_view / "source_catalogs/CHECKSUMS.sha256"
            ),
        },
    }
    for module_name, bindings in atmosphere_path_bindings.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attribute, value in bindings.items():
            atmosphere_paths_changed |= getattr(module, attribute) != value
            setattr(module, attribute, value)

    warm_start = sys.modules.get("payne_zero_atmosphere.warm_start")
    if warm_start is not None:
        five_label_path = atmosphere_emulator_dir / "five_label/checkpoint.pt"
        cno8_path = atmosphere_emulator_dir / "cno8/checkpoint.pt"
        warm_bindings = {
            "DEFAULT_EMULATOR_ASSET_DIR": atmosphere_emulator_dir,
            "DEFAULT_FIVE_LABEL_WEIGHTS_PATH": five_label_path,
            "DEFAULT_CNO8_WEIGHTS_PATH": cno8_path,
        }
        for attribute, value in warm_bindings.items():
            atmosphere_paths_changed |= getattr(warm_start, attribute) != value
            setattr(warm_start, attribute, value)
        if atmosphere_paths_changed:
            warm_start._load_atmosphere_initializer_cached.cache_clear()

    direct_abundance = sys.modules.get(
        "payne_zero_atmosphere.direct_abundance"
    )
    if direct_abundance is not None:
        direct_asset_dir = atmosphere_emulator_dir / "direct_abundance"
        direct_bindings = {
            "DEFAULT_DIRECT_XH_ASSET_DIR": direct_asset_dir,
            "DEFAULT_DIRECT_XH_CHECKPOINT_PATH": (
                direct_asset_dir / "checkpoint.pt"
            ),
            "DEFAULT_DIRECT_XH_MANIFEST_PATH": direct_asset_dir / "manifest.json",
        }
        direct_changed = False
        for attribute, value in direct_bindings.items():
            direct_changed |= getattr(direct_abundance, attribute) != value
            setattr(direct_abundance, attribute, value)
        if direct_changed:
            atmosphere_paths_changed = True
            direct_abundance._load_direct_abundance_initializer_cached.cache_clear()
            direct_abundance._build_direct_abundance_optimizer_surrogate_cached.cache_clear()

    atmosphere_package = sys.modules.get("payne_zero_atmosphere")
    if atmosphere_package is not None and warm_start is not None:
        atmosphere_package.DEFAULT_FIVE_LABEL_WEIGHTS_PATH = (
            warm_start.DEFAULT_FIVE_LABEL_WEIGHTS_PATH
        )
        atmosphere_package.DEFAULT_CNO8_WEIGHTS_PATH = (
            warm_start.DEFAULT_CNO8_WEIGHTS_PATH
        )


def _assert_staged_imports() -> None:
    staged_root = (REPOSITORY_ROOT / "src").resolve()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith(("payne_zero_synthesis", "payne_zero_atmosphere")):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            continue
        if not Path(module_path).resolve().is_relative_to(staged_root):
            raise RuntimeError(
                f"{name} resolved outside the staged source tree: {module_path}"
            )


def configure_chapter10_runtime() -> Path:
    """Configure immutable local tables and a disposable subset source view."""

    global _SOURCE_VIEW_OWNER, _SOURCE_VIEW_ROOT
    _assert_staged_imports()
    for path, expected in INPUT_HASHES.items():
        if _sha256(path) != expected:
            raise RuntimeError(f"Chapter 10 input identity changed: {path}")

    if _SOURCE_VIEW_ROOT is None:
        _SOURCE_VIEW_OWNER = tempfile.TemporaryDirectory(
            prefix="payne-zero-chapter10-"
        )
        working_root = Path(_SOURCE_VIEW_OWNER.name)
        source_root = working_root / "source_catalogs"
        synthesis_table_view = working_root / "data/synthesis_tables"
        (source_root / "lines").mkdir(parents=True)
        (source_root / "molecules").mkdir(parents=True)
        synthesis_table_view.mkdir(parents=True)
        (working_root / "data/source_catalogs").symlink_to(
            source_root.resolve(),
            target_is_directory=True,
        )
        links = {
            source_root / "lines/atomic_source_lines_parsed.npz": ATOMIC_SUBSET,
            source_root
            / "lines/molecular_equilibrium_synthesis.npz": MOLECULAR_EQUILIBRIUM,
            source_root / "molecules/manifest.json": MOLECULAR_MANIFEST,
            source_root
            / "molecules/molecular_band_lines.npz": MOLECULAR_TEXT_SUBSET,
            source_root / "molecules/titanium_oxide_lines.npy": TIO_SUBSET,
        }
        for destination, source in links.items():
            destination.symlink_to(source.resolve())
        checksums = []
        for destination, source in links.items():
            relative = destination.relative_to(source_root)
            checksums.append(f"{_sha256(source)}  {relative}")
        (source_root / "CHECKSUMS.sha256").write_text(
            "\n".join(checksums) + "\n",
            encoding="utf-8",
        )
        static_synthesis_tables = (
            REPOSITORY_ROOT / "data/static/synthesis_tables"
        )
        static_root = REPOSITORY_ROOT / "data/static"
        # Chapter 15 composes the atmosphere initializer with this synthesis
        # view.  Keep one shared PAYNE_ZERO_DATA_ROOT by exposing the immutable
        # atmosphere tables and emulator beside the generated synthesis table
        # view; only the directory entries are links, never copied data.
        (working_root / "data/atmosphere_tables").symlink_to(
            (static_root / "atmosphere_tables").resolve(),
            target_is_directory=True,
        )
        (working_root / "data/atmosphere_emulator").symlink_to(
            (static_root / "atmosphere_emulator").resolve(),
            target_is_directory=True,
        )
        for source in static_synthesis_tables.glob("*.npz"):
            if source.name == "partition_saha_tables.npz":
                continue
            (synthesis_table_view / source.name).symlink_to(source.resolve())
        with (
            np.load(
                static_synthesis_tables / "partition_saha_tables.npz",
                allow_pickle=False,
            ) as invariant_archive,
            np.load(SYNTHESIS_EOS_STATE, allow_pickle=False) as state_archive,
        ):
            partition_payload = {
                name: np.asarray(invariant_archive[name])
                for name in invariant_archive.files
            }
            partition_payload.update(
                {
                    name: np.asarray(state_archive[name])
                    for name in state_archive.files
                }
            )
        np.savez(
            synthesis_table_view / "partition_saha_inputs.npz",
            **partition_payload,
        )
        _SOURCE_VIEW_ROOT = source_root

    source_root = _SOURCE_VIEW_ROOT
    working_root = source_root.parent
    static_root = REPOSITORY_ROOT / "data/static"
    data_view = working_root / "data"
    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(data_view)
    os.environ["PAYNE_ZERO_ATMOSPHERE_DATA_ROOT"] = str(
        static_root / "atmosphere_tables"
    )
    os.environ["PAYNE_ZERO_SYNTHESIS_DATA_ROOT"] = str(
        data_view / "synthesis_tables"
    )
    os.environ["PAYNE_ZERO_SYNTHESIS_SOURCE_CATALOG_ROOT"] = str(source_root)
    os.environ["PAYNE_ZERO_SYNTHESIS_CACHE_DIR"] = str(working_root / "cache")
    os.environ["PAYNE_ZERO_SYNTHESIS_MOLECULAR_SOURCE_CACHE_DIR"] = str(
        working_root / "cache/molecular_source"
    )
    os.environ["PAYNE_ZERO_SYNTHESIS_MOLECULAR_COMPILED_CACHE_DIR"] = str(
        working_root / "cache/molecular_compiled"
    )

    staged_root = str((REPOSITORY_ROOT / "src").resolve())
    if staged_root in sys.path:
        sys.path.remove(staged_root)
    sys.path.insert(0, staged_root)
    _rebind_imported_runtime_paths(
        data_view=data_view,
        cache_root=working_root / "cache",
    )
    return source_root


@dataclass(frozen=True)
class FixtureCheckpoint:
    regimes: tuple[str, ...]
    depth_count: int
    schema_version: int
    required_array_count: int
    electron_density_surface_cm3: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class RuntimePolicyCheckpoint:
    default_device: str
    default_dtype: str
    cpu_default_dtype: str
    mps_float64_rejected: bool
    cuda_available: bool
    mps_available: bool


@dataclass(frozen=True)
class GridCheckpoint:
    requested_count: int
    synthesis_count: int
    context_each_side: int
    output_slice: slice
    interior_bitwise_exact: bool
    local_resolving_power_min: float
    local_resolving_power_max: float


@dataclass(frozen=True)
class InvariantCheckpoint:
    field_names: tuple[str, ...]
    atomic_line_count: int
    molecular_line_count: int
    has_metal: bool
    has_helium: bool
    has_hydrogen: bool
    metal_chunk: int
    device: str
    dtype: str
    build_profile: dict[str, float]


@dataclass(frozen=True)
class CacheCheckpoint:
    process_hit_same_object: bool
    disabled_build_new_object: bool
    disabled_values_equal: bool
    perturbed_key_is_distinct: bool
    clear_removes_entries: bool


@dataclass(frozen=True)
class HydrogenCheckpoint:
    template_depth_count: int
    hot_depth_count: int
    cool_depth_count: int
    hot_cool_state_differs: bool
    template_unchanged: bool


@dataclass(frozen=True)
class MemoryCheckpoint:
    depth_count: int
    synthesis_wavelength_count: int
    line_count: int
    float32_line_slab_bytes: int
    four_float64_depth_wavelength_slabs_bytes: int
    hypothetical_dense_depth_line_wavelength_bytes: int
    dense_depth_line_wavelength_allocated: bool
    host_state_fields: tuple[str, ...]
    device_state_fields: tuple[str, ...]


@dataclass(frozen=True)
class EndToEndCheckpoint:
    regime: str
    wavelength_nm: np.ndarray
    eddington_flux_total_per_frequency: np.ndarray
    eddington_flux_continuum_per_frequency: np.ndarray
    flux_total: np.ndarray
    flux_continuum: np.ndarray
    normalized_flux: np.ndarray
    continuum_absorption: np.ndarray
    continuum_scattering: np.ndarray
    line_absorption: np.ndarray
    line_source: np.ndarray
    stage_order: tuple[str, ...]
    seconds: float
    result_field_names: tuple[str, ...]
    spectrum_field_names: tuple[str, ...]


@dataclass(frozen=True)
class RoundTripCheckpoint:
    saved_field_count: int
    field_names_exact: bool
    arrays_exact: bool
    fixed_electron_density_seed: bool


@dataclass(frozen=True)
class TimingCheckpoint:
    labels: tuple[str, ...]
    seconds: np.ndarray
    outputs_equal: bool
    python: str
    numpy: str
    torch: str
    machine: str


@dataclass(frozen=True)
class FourRegimeCheckpoint:
    regimes: tuple[str, ...]
    wavelength_nm: np.ndarray
    normalized_flux: np.ndarray
    minimum_normalized_flux: np.ndarray
    process_seconds: np.ndarray
    backend: str
    dtype: str


def load_regime_atmosphere(regime: str) -> dict[str, np.ndarray]:
    """Load one input-only regime state and expose it as native schema v4."""

    if regime not in REGIMES:
        raise ValueError(f"unknown Chapter 10 regime: {regime}")
    prefix = f"{regime}__synthesis__"
    with np.load(ATMOSPHERE_FIXTURE, allow_pickle=False) as archive:
        atmosphere = {
            name.removeprefix(prefix): np.asarray(archive[name]).copy()
            for name in archive.files
            if name.startswith(prefix)
        }
    atmosphere["atmosphere_schema_version"] = np.asarray([4], np.int32)
    return atmosphere


@lru_cache(maxsize=1)
def fixture_checkpoint() -> FixtureCheckpoint:
    configure_chapter10_runtime()
    from payne_zero_synthesis.atmosphere import REQUIRED_ATMOSPHERE_ARRAYS

    atmospheres = [load_regime_atmosphere(name) for name in REGIMES]
    for atmosphere in atmospheres:
        missing = set(REQUIRED_ATMOSPHERE_ARRAYS).difference(atmosphere)
        if missing:
            raise RuntimeError(f"Chapter 10 fixture fields missing: {sorted(missing)}")
    return FixtureCheckpoint(
        regimes=REGIMES,
        depth_count=int(atmospheres[0]["temperature"].size),
        schema_version=4,
        required_array_count=len(REQUIRED_ATMOSPHERE_ARRAYS),
        electron_density_surface_cm3=np.asarray(
            [atmosphere["electron_density"][0] for atmosphere in atmospheres],
            np.float64,
        ),
        source_sha256=_sha256(ATMOSPHERE_FIXTURE),
    )


def runtime_policy_checkpoint() -> RuntimePolicyCheckpoint:
    configure_chapter10_runtime()
    import torch
    from payne_zero_synthesis.device import resolve_runtime

    default_device, default_dtype = resolve_runtime()
    _, cpu_dtype = resolve_runtime(torch.device("cpu"), None)
    rejected = False
    try:
        resolve_runtime(torch.device("mps"), torch.float64)
    except ValueError:
        rejected = True
    return RuntimePolicyCheckpoint(
        default_device=str(default_device),
        default_dtype=str(default_dtype),
        cpu_default_dtype=str(cpu_dtype),
        mps_float64_rejected=rejected,
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
    )


@lru_cache(maxsize=1)
def grid_checkpoint() -> GridCheckpoint:
    configure_chapter10_runtime()
    from payne_zero_synthesis.pipeline import (
        WINDOW_CONTEXT_SAMPLES,
        _window_grid_contract,
    )

    _, _, requested, synthesis, output_slice = _window_grid_contract(
        WINDOW_START_NM,
        WINDOW_END_NM,
        RESOLUTION,
    )
    resolving_power = 0.5 * (
        requested[1:] + requested[:-1]
    ) / np.diff(requested)
    return GridCheckpoint(
        requested_count=int(requested.size),
        synthesis_count=int(synthesis.size),
        context_each_side=WINDOW_CONTEXT_SAMPLES,
        output_slice=output_slice,
        interior_bitwise_exact=np.array_equal(
            requested, synthesis[output_slice]
        ),
        local_resolving_power_min=float(resolving_power.min()),
        local_resolving_power_max=float(resolving_power.max()),
    )


def _cpu_invariants():
    configure_chapter10_runtime()
    import torch
    from payne_zero_synthesis.pipeline import window_invariants_for

    return window_invariants_for(
        wl_start_nm=WINDOW_START_NM,
        wl_end_nm=WINDOW_END_NM,
        resolution=RESOLUTION,
        molecular_lines=True,
        runtime_device=torch.device("cpu"),
        work_dtype=torch.float64,
    )


@lru_cache(maxsize=1)
def invariant_checkpoint() -> InvariantCheckpoint:
    bundle = _cpu_invariants()
    return InvariantCheckpoint(
        field_names=tuple(field.name for field in fields(bundle)),
        atomic_line_count=bundle.n_atomic,
        molecular_line_count=bundle.n_molecular,
        has_metal=bundle.has_metal,
        has_helium=bundle.has_helium,
        has_hydrogen=bundle.has_hydrogen,
        metal_chunk=bundle.metal_chunk,
        device=str(bundle.device),
        dtype=str(bundle.dtype),
        build_profile=dict(bundle.build_profile),
    )


def cache_checkpoint() -> CacheCheckpoint:
    configure_chapter10_runtime()
    import torch
    import payne_zero_synthesis.pipeline as pipeline

    pipeline.clear_window_invariant_cache()
    first = _cpu_invariants()
    second = _cpu_invariants()
    os.environ["PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE"] = "1"
    try:
        disabled = _cpu_invariants()
    finally:
        os.environ.pop("PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE", None)
    perturbed = pipeline.window_invariants_for(
        wl_start_nm=WINDOW_START_NM,
        wl_end_nm=WINDOW_END_NM,
        resolution=RESOLUTION + 1.0,
        molecular_lines=True,
        runtime_device=torch.device("cpu"),
        work_dtype=torch.float64,
    )
    values_equal = (
        np.array_equal(first.wavelength_nm, disabled.wavelength_nm)
        and first.n_atomic == disabled.n_atomic
        and first.n_molecular == disabled.n_molecular
    )
    pipeline.clear_window_invariant_cache()
    empty = len(pipeline._WINDOW_INVARIANT_CACHE) == 0
    return CacheCheckpoint(
        process_hit_same_object=first is second,
        disabled_build_new_object=disabled is not first,
        disabled_values_equal=values_equal,
        perturbed_key_is_distinct=perturbed.key != first.key,
        clear_removes_entries=empty,
    )


@lru_cache(maxsize=1)
def hydrogen_checkpoint() -> HydrogenCheckpoint:
    configure_chapter10_runtime()
    import torch
    from payne_zero_synthesis.pipeline import SynthesisPipeline

    bundle = _cpu_invariants()
    template = bundle.hydrogen_invariants_template
    if template is None:
        raise RuntimeError("the teaching subset did not activate hydrogen")
    template_before = np.asarray(template.merge_wavenumber_by_depth).copy()
    hot = SynthesisPipeline(
        load_regime_atmosphere("hot_dwarf"),
        wl_start_nm=WINDOW_START_NM,
        wl_end_nm=WINDOW_END_NM,
        resolution=RESOLUTION,
        molecular_lines=True,
        device=torch.device("cpu"),
        dtype=torch.float64,
        window_invariants=bundle,
    )
    cool = SynthesisPipeline(
        load_regime_atmosphere("cool_molecule_rich"),
        wl_start_nm=WINDOW_START_NM,
        wl_end_nm=WINDOW_END_NM,
        resolution=RESOLUTION,
        molecular_lines=True,
        device=torch.device("cpu"),
        dtype=torch.float64,
        window_invariants=bundle,
    )
    hot_merge = np.asarray(hot.hydrogen_invariants.merge_wavenumber_by_depth)
    cool_merge = np.asarray(cool.hydrogen_invariants.merge_wavenumber_by_depth)
    template_after = np.asarray(template.merge_wavenumber_by_depth)
    return HydrogenCheckpoint(
        template_depth_count=int(template_before.size),
        hot_depth_count=int(hot_merge.size),
        cool_depth_count=int(cool_merge.size),
        hot_cool_state_differs=not np.array_equal(hot_merge, cool_merge),
        template_unchanged=np.array_equal(template_before, template_after),
    )


@lru_cache(maxsize=1)
def memory_checkpoint() -> MemoryCheckpoint:
    bundle = _cpu_invariants()
    atmosphere = load_regime_atmosphere("solar_dwarf")
    depth_count = int(atmosphere["temperature"].size)
    wavelength_count = int(bundle.n_synthesis_wl)
    line_count = int(bundle.n_atomic + bundle.n_molecular)
    allocation = allocation_ledger(
        depth_count=depth_count,
        wavelength_count=wavelength_count,
        line_count=line_count,
    )
    return MemoryCheckpoint(
        depth_count=depth_count,
        synthesis_wavelength_count=wavelength_count,
        line_count=line_count,
        float32_line_slab_bytes=allocation.float32_line_slab_bytes,
        four_float64_depth_wavelength_slabs_bytes=(
            allocation.four_float64_depth_wavelength_slabs_bytes
        ),
        hypothetical_dense_depth_line_wavelength_bytes=(
            allocation.hypothetical_dense_depth_line_wavelength_bytes
        ),
        dense_depth_line_wavelength_allocated=False,
        host_state_fields=(
            "partition_normalized_populations",
            "fractional_doppler_widths",
            "mass_density",
            "electron_density",
            "temperature",
            "hc_over_kt",
            "microturbulence",
        ),
        device_state_fields=(
            "column_mass",
            "temperature tensor",
            "continuum slabs",
            "shared float32 line slab",
            "line source/scattering",
            "flux vectors through crop",
        ),
    )


@lru_cache(maxsize=4)
def end_to_end_checkpoint(regime: str = "solar_dwarf") -> EndToEndCheckpoint:
    configure_chapter10_runtime()
    import torch
    from payne_zero_synthesis.api import _wrap
    from payne_zero_synthesis.pipeline import SpectrumResult, SynthesisPipeline

    bundle = _cpu_invariants()
    runner = SynthesisPipeline(
        load_regime_atmosphere(regime),
        source_path=None,
        wl_start_nm=WINDOW_START_NM,
        wl_end_nm=WINDOW_END_NM,
        resolution=RESOLUTION,
        molecular_lines=True,
        device=torch.device("cpu"),
        dtype=torch.float64,
        window_invariants=bundle,
    )
    started = perf_counter()
    result = runner.run(keep_slabs=True, spectral_operator=None)
    seconds = perf_counter() - started
    spectrum = _wrap(result, seconds)
    if not np.allclose(
        spectrum.normalized_flux,
        spectrum.flux_total / spectrum.flux_continuum,
        rtol=3.0e-7,
        atol=0.0,
    ):
        raise RuntimeError("public flux conversion changed the normalized ratio")
    return EndToEndCheckpoint(
        regime=regime,
        wavelength_nm=result.wavelength_nm.copy(),
        eddington_flux_total_per_frequency=(
            result.eddington_flux_total_per_frequency.copy()
        ),
        eddington_flux_continuum_per_frequency=(
            result.eddington_flux_continuum_per_frequency.copy()
        ),
        flux_total=spectrum.flux_total.copy(),
        flux_continuum=spectrum.flux_continuum.copy(),
        normalized_flux=spectrum.normalized_flux.copy(),
        continuum_absorption=result.continuum_absorption.copy(),
        continuum_scattering=result.continuum_scattering.copy(),
        line_absorption=result.line_mass_absorption_coefficient.copy(),
        line_source=result.line_source.copy(),
        stage_order=STAGE_ORDER,
        seconds=float(seconds),
        result_field_names=tuple(field.name for field in fields(SpectrumResult)),
        spectrum_field_names=tuple(field.name for field in fields(spectrum)),
    )


def public_spectrum(regime: str = "solar_dwarf"):
    """Call the exact five-field public API on a schema-v4 in-memory mapping."""

    configure_chapter10_runtime()
    from payne_zero_synthesis import synthesize

    return synthesize(
        load_regime_atmosphere(regime),
        wavelength_start_nm=WINDOW_START_NM,
        wavelength_end_nm=WINDOW_END_NM,
        resolution=RESOLUTION,
        molecular_lines=True,
        device="cpu",
        dtype="float64",
    )


def roundtrip_checkpoint() -> RoundTripCheckpoint:
    configure_chapter10_runtime()
    from payne_zero_synthesis import (
        build_structured_atmosphere,
        load_atmosphere_npz,
        save_structured_atmosphere,
    )
    from payne_zero_synthesis.atmosphere import REQUIRED_ATMOSPHERE_ARRAYS

    source = load_regime_atmosphere("solar_dwarf")
    electron_density_before = source["electron_density"].copy()
    atmosphere = build_structured_atmosphere(
        temperature=source["temperature"],
        column_mass=source["column_mass"],
        gas_pressure=source["gas_pressure"],
        electron_density=source["electron_density"],
        elemental_abundances=source["elemental_abundances"],
        microturbulence=source["microturbulence"],
        mass_density=source["mass_density"],
        molecular_lines=False,
        device="cpu",
        dtype="float64",
    )
    with tempfile.TemporaryDirectory(prefix="chapter10-atmosphere-") as directory:
        path = Path(directory) / "solar_schema_v4.npz"
        saved_names = save_structured_atmosphere(atmosphere, path)
        loaded = load_atmosphere_npz(path)
    public_names = tuple(sorted(REQUIRED_ATMOSPHERE_ARRAYS))
    return RoundTripCheckpoint(
        saved_field_count=len(saved_names),
        field_names_exact=tuple(sorted(saved_names)) == public_names,
        arrays_exact=all(
            np.array_equal(atmosphere[name], loaded[name]) for name in public_names
        ),
        fixed_electron_density_seed=np.array_equal(
            electron_density_before, loaded["electron_density"]
        ),
    )


def prewarm_checkpoint(*, force: bool = False) -> dict:
    """Run exact fixed-CPU-float64 prewarm without printing its JSON payload."""

    configure_chapter10_runtime()
    from payne_zero_synthesis.prewarm import prewarm

    with redirect_stdout(io.StringIO()):
        return prewarm(
            wavelength_start_nm=WINDOW_START_NM,
            wavelength_end_nm=WINDOW_END_NM,
            resolution=RESOLUTION,
            force=force,
        )


@lru_cache(maxsize=1)
def timing_checkpoint() -> TimingCheckpoint:
    configure_chapter10_runtime()
    import torch
    import payne_zero_synthesis.pipeline as pipeline
    from payne_zero_synthesis import synthesize

    cache_root = Path(os.environ["PAYNE_ZERO_SYNTHESIS_CACHE_DIR"]).resolve()
    source_view_parent = _SOURCE_VIEW_ROOT.parent.resolve()
    if not cache_root.is_relative_to(source_view_parent):
        raise RuntimeError("refusing to reset a cache outside the disposable view")
    if cache_root.exists():
        shutil.rmtree(cache_root)

    outputs = []
    elapsed = []
    for _label in ("cold source", "persistent warm", "process warm"):
        if _label != "process warm":
            pipeline.clear_window_invariant_cache()
        started = perf_counter()
        spectrum = synthesize(
            load_regime_atmosphere("solar_dwarf"),
            wavelength_start_nm=WINDOW_START_NM,
            wavelength_end_nm=WINDOW_END_NM,
            resolution=RESOLUTION,
            molecular_lines=True,
            device="cpu",
            dtype="float64",
        )
        elapsed.append(perf_counter() - started)
        outputs.append(spectrum.normalized_flux.copy())
    return TimingCheckpoint(
        labels=("cold source", "persistent warm", "process warm"),
        seconds=np.asarray(elapsed, np.float64),
        outputs_equal=all(np.array_equal(outputs[0], values) for values in outputs[1:]),
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        machine=platform.machine(),
    )


@lru_cache(maxsize=1)
def four_regime_checkpoint() -> FourRegimeCheckpoint:
    spectra = []
    seconds = []
    wavelength_nm = None
    for regime in REGIMES:
        spectrum = public_spectrum(regime)
        if wavelength_nm is None:
            wavelength_nm = spectrum.wavelength_nm.copy()
        elif not np.array_equal(wavelength_nm, spectrum.wavelength_nm):
            raise RuntimeError("same-window spectra returned different grids")
        spectra.append(spectrum.normalized_flux.copy())
        seconds.append(spectrum.seconds)
    normalized = np.stack(spectra)
    return FourRegimeCheckpoint(
        regimes=REGIMES,
        wavelength_nm=wavelength_nm,
        normalized_flux=normalized,
        minimum_normalized_flux=normalized.min(axis=1),
        process_seconds=np.asarray(seconds, np.float64),
        backend="cpu",
        dtype="torch.float64",
    )
