#!/usr/bin/env python3
"""Build the small, source-faithful molecular inputs used by Chapter 8.

The full molecular catalogs remain optional because they occupy several
gigabytes.  This builder reads them without modification and writes only:

* all 32 manifest bands, with a few original rows per band;
* one contiguous TiO slice near 499 nm;
* a small H2O set containing the local window and all four sign codes;
* one contiguous atmosphere-diatomic slice near 499 nm; and
* a clearly labelled explicit-path control for the H3+ routing branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np


PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
SOURCE_HASHES = {
    "manifest.json": "65131b3c07f093f062afed5547875b969da1e607dc46e3120332758d5e1c32c6",
    "molecular_band_lines.npz": (
        "39cb02310e109eca28190b41d82148173e8000ad6a968cc65ec14e6d2f6bd7df"
    ),
    "titanium_oxide_lines.npy": (
        "7def3a02828fba3a917581343f0cdbc11d97fb335a69c69098d348a344d38ca8"
    ),
    "water_lines.npy": (
        "3fd5f26886c8af68a1910606790358e4971b3d8831b3d0c1be493ca73f615442"
    ),
    "diatomic_lines.npy": (
        "3980781fa73d9b4e39d9fa5e53dbdf5ad326447753f6146d9040c33036a7ad67"
    ),
}
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
TEACHING_START_NM = 498.95
TEACHING_END_NM = 499.15
CO_FIGURE_START_NM = 999.5
CO_FIGURE_END_NM = 1000.5
FINE_LOG_STEP = math.log(1.0 + 1.0 / 2_000_000.0)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 without loading a full catalog into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{path} has SHA-256 {actual}, expected pinned identity {expected}"
        )


def energy_wavelength_nm(
    stored_wavelength_nm: np.ndarray,
    first_energy_cm: np.ndarray,
    second_energy_cm: np.ndarray,
) -> np.ndarray:
    """Mirror the standard energy-level wavelength policy vectorially."""

    difference = np.abs(np.abs(second_energy_cm) - np.abs(first_energy_cm))
    result = np.abs(stored_wavelength_nm).astype(np.float64, copy=True)
    use_energy = difference > 0.0
    result[use_energy] = 1.0e7 / difference[use_energy]
    return result


def choose_text_rows(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Choose original rows in source order, emphasizing the teaching window."""

    stored = np.asarray(arrays["stored_wavelength_nm"], np.float64)
    first = np.asarray(arrays["first_energy_cm"], np.float64)
    second = np.asarray(arrays["second_energy_cm"], np.float64)
    line_wavelength = energy_wavelength_nm(stored, first, second)
    standard = (
        (stored != 0.0)
        & (first >= 0.0)
        & (second >= 0.0)
        & (line_wavelength >= TEACHING_START_NM - 0.01)
        & (line_wavelength <= TEACHING_END_NM + 0.1)
        & (np.abs(stored) >= TEACHING_START_NM - 10.01)
        & (np.abs(stored) <= TEACHING_END_NM + 10.1)
    )
    candidates = np.flatnonzero(standard)
    selected: list[int] = []
    if candidates.size:
        log_gf = np.asarray(arrays["log_oscillator_strength"], np.float64)
        strongest = candidates[
            np.lexsort((candidates, -log_gf[candidates]))[:8]
        ]
        selected.extend(int(index) for index in strongest)

    # A second, narrow interval supplies a physically recognizable CO stick
    # spectrum without widening the standard 499 nm opacity calculation.
    co_figure = (
        (stored != 0.0)
        & (first >= 0.0)
        & (second >= 0.0)
        & (line_wavelength >= CO_FIGURE_START_NM - 0.01)
        & (line_wavelength <= CO_FIGURE_END_NM + 0.1)
        & (np.abs(stored) >= CO_FIGURE_START_NM - 10.01)
        & (np.abs(stored) <= CO_FIGURE_END_NM + 10.1)
    )
    co_candidates = np.flatnonzero(co_figure)
    if co_candidates.size:
        log_gf = np.asarray(arrays["log_oscillator_strength"], np.float64)
        strongest_co = co_candidates[
            np.lexsort((co_candidates, -log_gf[co_candidates]))[:6]
        ]
        selected.extend(int(index) for index in strongest_co)

    # Every manifest band remains represented even if it has no line in this
    # window.  The anchor is source-faithful and normally rejected by the
    # window compiler.
    nonzero = np.flatnonzero(stored != 0.0)
    selected.append(int(nonzero[0]) if nonzero.size else 0)

    # Preserve one predicted-line sign example when the source contains one.
    predicted = np.flatnonzero((first < 0.0) | (second < 0.0))
    if predicted.size:
        selected.append(int(predicted[0]))
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def build_text_subset(source: Path, destination: Path, manifest: dict) -> dict:
    payload: dict[str, np.ndarray] = {}
    band_rows: dict[str, list[int]] = {}
    with np.load(source, allow_pickle=False) as archive:
        for entry in manifest["text_sources"]:
            band = entry["band"]
            arrays = {
                field: np.asarray(archive[f"{band}/{field}"])
                for field in TEXT_FIELDS
            }
            rows = choose_text_rows(arrays)
            band_rows[band] = rows.tolist()
            for field, values in arrays.items():
                payload[f"{band}/{field}"] = np.asarray(values[rows])
            payload[f"__meta__/{band}/source_row_index"] = rows
    payload["__meta__/payne_zero_commit"] = np.asarray(PINNED_COMMIT)
    payload["__meta__/full_source_sha256"] = np.asarray(
        SOURCE_HASHES["molecular_band_lines.npz"]
    )
    payload["__meta__/selection"] = np.asarray(
        "source-order anchors plus up to eight strongest standard-call rows "
        "whose energy-derived centers lie in 498.94-499.25 nm, plus up to "
        "six strongest rows in 999.49-1000.60 nm for the CO band figure"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return {
        "path": str(destination),
        "sha256": sha256(destination),
        "bands": len(band_rows),
        "rows_by_band": band_rows,
        "total_rows": int(sum(len(rows) for rows in band_rows.values())),
    }


def contiguous_slice_near(
    records: np.ndarray,
    wavelength_nm: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    codes = records["wavelength_code"] if records.dtype.fields else records[:, 0]
    target_code = int(round(math.log(wavelength_nm) / FINE_LOG_STEP))
    center = int(np.searchsorted(codes, target_code, side="left"))
    start = max(0, min(center - count // 2, records.shape[0] - count))
    rows = np.arange(start, start + count, dtype=np.int64)
    return np.asarray(records[rows]).copy(), rows


def first_h2o_sign_rows(records: np.ndarray) -> np.ndarray:
    energy = records["signed_lower_energy_code"]
    strength = records["signed_log_oscillator_strength_code"]
    conditions = (
        (energy > 0) & (strength > 0),
        (energy > 0) & (strength <= 0),
        (energy <= 0) & (strength > 0),
        (energy <= 0) & (strength <= 0),
    )
    return np.asarray(
        [int(np.flatnonzero(condition)[0]) for condition in conditions],
        dtype=np.int64,
    )


def build(
    source_root: Path,
    atmosphere_lines_root: Path,
    repository_root: Path,
) -> None:
    source_root = source_root.resolve()
    atmosphere_lines_root = atmosphere_lines_root.resolve()
    repository_root = repository_root.resolve()

    sources = {
        "manifest.json": source_root / "manifest.json",
        "molecular_band_lines.npz": source_root / "molecular_band_lines.npz",
        "titanium_oxide_lines.npy": source_root / "titanium_oxide_lines.npy",
        "water_lines.npy": source_root / "water_lines.npy",
        "diatomic_lines.npy": atmosphere_lines_root / "diatomic_lines.npy",
    }
    for name, path in sources.items():
        verify_source(path, SOURCE_HASHES[name])

    manifest = json.loads(sources["manifest.json"].read_text())
    static_manifest = (
        repository_root / "data/static/molecular_sources/manifest.json"
    )
    static_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sources["manifest.json"], static_manifest)

    text_destination = (
        repository_root / "data/subsets/chapter08_molecular_text_subset.npz"
    )
    text_meta = build_text_subset(
        sources["molecular_band_lines.npz"],
        text_destination,
        manifest,
    )

    tio_full = np.load(sources["titanium_oxide_lines.npy"], mmap_mode="r")
    tio_subset, tio_rows = contiguous_slice_near(tio_full, 499.0, 64)
    tio_destination = repository_root / "data/subsets/chapter08_tio_subset.npy"
    np.save(tio_destination, tio_subset, allow_pickle=False)

    water_full = np.load(sources["water_lines.npy"], mmap_mode="r")
    water_local, water_local_rows = contiguous_slice_near(water_full, 499.0, 24)
    sign_rows = first_h2o_sign_rows(water_full)
    water_rows = np.unique(np.concatenate((water_local_rows, sign_rows)))
    water_subset = np.asarray(water_full[water_rows]).copy()
    water_destination = repository_root / "data/subsets/chapter08_h2o_subset.npy"
    np.save(water_destination, water_subset, allow_pickle=False)

    diatomic_full = np.load(sources["diatomic_lines.npy"], mmap_mode="r")
    diatomic_subset, diatomic_rows = contiguous_slice_near(
        diatomic_full, 499.0, 96
    )
    diatomic_destination = (
        repository_root / "data/subsets/chapter08_atmosphere_diatomic_subset.npy"
    )
    np.save(diatomic_destination, diatomic_subset, allow_pickle=False)

    # This control proves explicit-path routing only.  It is copied byte-for-byte
    # from two source records and must never be described as an H3+ source catalog.
    h3_probe_destination = (
        repository_root / "data/fixtures/chapter08_h3plus_path_probe.npy"
    )
    h3_probe_destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        h3_probe_destination,
        np.asarray(diatomic_subset[:2], dtype=np.int32),
        allow_pickle=False,
    )

    outputs = {
        "static_manifest": {
            "path": str(static_manifest),
            "sha256": sha256(static_manifest),
        },
        "text_subset": text_meta,
        "tio_subset": {
            "path": str(tio_destination),
            "sha256": sha256(tio_destination),
            "source_rows": [int(tio_rows[0]), int(tio_rows[-1])],
            "record_count": int(tio_rows.size),
        },
        "h2o_subset": {
            "path": str(water_destination),
            "sha256": sha256(water_destination),
            "source_rows": water_rows.tolist(),
            "sign_case_source_rows": sign_rows.tolist(),
            "record_count": int(water_rows.size),
        },
        "atmosphere_diatomic_subset": {
            "path": str(diatomic_destination),
            "sha256": sha256(diatomic_destination),
            "source_rows": [int(diatomic_rows[0]), int(diatomic_rows[-1])],
            "record_count": int(diatomic_rows.size),
        },
        "h3plus_explicit_path_probe": {
            "path": str(h3_probe_destination),
            "sha256": sha256(h3_probe_destination),
            "role": "fixture",
            "meaning": (
                "explicit-path routing control copied from two diatomic source "
                "records; not an H3+ scientific catalog"
            ),
        },
    }
    provenance = {
        "payne_zero_commit": PINNED_COMMIT,
        "builder": "scripts/build_chapter08_molecular_subsets.py",
        "teaching_window_nm": [TEACHING_START_NM, TEACHING_END_NM],
        "full_sources": {
            name: {
                "path": str(path),
                "sha256": SOURCE_HASHES[name],
                "size_bytes": path.stat().st_size,
            }
            for name, path in sources.items()
        },
        "outputs": outputs,
    }
    provenance_path = (
        repository_root / "data/subsets/chapter08_molecular_provenance.json"
    )
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/Users/ysting/payne-zero/source_data_files/source_catalogs/molecules"
        ),
    )
    parser.add_argument(
        "--atmosphere-lines-root",
        type=Path,
        default=Path(
            "/Users/ysting/payne-zero/source_data_files/source_catalogs/lines"
        ),
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args()
    build(
        arguments.source_root,
        arguments.atmosphere_lines_root,
        arguments.repository_root,
    )


if __name__ == "__main__":
    main()
