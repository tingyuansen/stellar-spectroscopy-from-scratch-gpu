"""Focused gates for the Chapter 15 three-run solar publisher."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_chapter15_verified_solar_goldens import (
    ACCEPTANCE_NAME,
    ATMOSPHERE_ARRAYS,
    ATMOSPHERE_INPUT_NAME,
    ATMOSPHERE_SHAPES,
    DEFAULT_OUTPUT_DIRECTORY,
    GOLDEN_ATMOSPHERE_NAME,
    GOLDEN_SPECTRUM_NAME,
    ParityError,
    RunArtifacts,
    SPECTRUM_INPUT_NAME,
    publication_gate,
    publish,
    sha256,
    validate_published,
)
from scripts.deterministic_npz import write_npz


def _atmosphere_arrays() -> dict[str, np.ndarray]:
    arrays = {
        name: np.ones(shape, dtype=np.float64)
        for name, shape in ATMOSPHERE_SHAPES.items()
        if name != "atmosphere_schema_version"
    }
    arrays["column_mass"] = np.geomspace(1.0e-4, 10.0, 80, dtype=np.float64)
    arrays["temperature"] = np.linspace(3700.0, 11400.0, 80, dtype=np.float64)
    arrays["gas_pressure"] = np.geomspace(10.0, 3.0e5, 80, dtype=np.float64)
    arrays["electron_density"] = np.geomspace(1.0e9, 2.0e16, 80, dtype=np.float64)
    arrays["mass_density"] = np.geomspace(1.0e-11, 4.0e-7, 80, dtype=np.float64)
    arrays["continuum_edge_wavelength_nm"] = np.linspace(
        1.0, 500000.0, 341, dtype=np.float64
    )
    arrays["continuum_edge_midpoint_wavelength_nm"] = np.linspace(
        1.5, 416666.0, 340, dtype=np.float64
    )
    arrays["atmosphere_schema_version"] = np.asarray([4], dtype=np.int32)
    assert set(arrays) == set(ATMOSPHERE_ARRAYS)
    return arrays


def _spectrum_arrays(seconds: float) -> dict[str, np.ndarray]:
    wavelength = np.linspace(498.96, 499.14, 8, dtype=np.float64)
    continuum = np.linspace(1.08e8, 1.07e8, 8, dtype=np.float64)
    normalized = np.linspace(0.72, 0.91, 8, dtype=np.float64)
    total = continuum * normalized
    return {
        "wavelength_nm": wavelength,
        "flux_total": total,
        "flux_continuum": continuum,
        "normalized_flux": total / continuum,
        "seconds": np.asarray([seconds], dtype=np.float64),
    }


def _prepared_runs(root: Path) -> tuple[RunArtifacts, ...]:
    roles = ("staged", "staged_repeat", "pinned_read_only_oracle")
    runs = []
    atmosphere = _atmosphere_arrays()
    for index, role in enumerate(roles):
        directory = root / role
        directory.mkdir(parents=True)
        atmosphere_path = directory / ATMOSPHERE_INPUT_NAME
        spectrum_path = directory / SPECTRUM_INPUT_NAME
        write_npz(atmosphere_path, atmosphere)
        write_npz(spectrum_path, _spectrum_arrays(1.0 + index))
        runs.append(RunArtifacts(role, atmosphere_path, spectrum_path))
    return tuple(runs)


def test_missing_pinned_oracle_is_an_explicit_nonwriting_gate(
    tmp_path: Path,
) -> None:
    runs = _prepared_runs(tmp_path)
    runs[-1].atmosphere.unlink()
    runs[-1].spectrum.unlink()
    gate = publication_gate(runs)
    assert not gate.ready
    assert gate.status == "blocked_prepared_run_incomplete"
    assert len(gate.blockers) == 2


def test_publish_ignores_only_timing_and_writes_role_honest_record(
    tmp_path: Path,
) -> None:
    runs = _prepared_runs(tmp_path / "runs")
    output = tmp_path / "data/golden/payne_zero/chapter15"
    manifest = tmp_path / "data/MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "payne_zero_commit": ("9c44001feae40b85146630499e6f8a5fed42e5af"),
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    result = publish(
        runs=runs,
        output_directory=output,
        update_manifest=True,
        manifest_path=manifest,
    )
    assert result.atmosphere.name == GOLDEN_ATMOSPHERE_NAME
    assert result.spectrum.name == GOLDEN_SPECTRUM_NAME
    assert result.acceptance.name == ACCEPTANCE_NAME

    with np.load(result.spectrum, allow_pickle=False) as spectrum:
        assert set(spectrum.files) == {
            "wavelength_nm",
            "flux_total",
            "flux_continuum",
            "normalized_flux",
        }
        assert "seconds" not in spectrum.files

    record = json.loads(result.acceptance.read_text(encoding="utf-8"))
    assert record["status"] == "accepted_exact_three_run_array_parity"
    assert record["timing_policy"]["excluded_from_identity"] == ["seconds"]
    assert [
        item["synthesis_seconds_observed_not_identity"] for item in record["inputs"]
    ] == [1.0, 2.0, 3.0]
    assert record["evidence"]["structural_convergence"]["accepted"] is True
    assert record["evidence"]["declared_flux_error_threshold"]["accepted"] is None
    assert record["evidence"]["hydrostatic_residual"]["accepted"] is None
    assert record["catalog_policy"]["full_source_catalogs_are_optional_textbook_data"]
    assert record["catalog_policy"]["required_to_rebuild_this_record"]
    assert not record["catalog_policy"]["required_to_validate_published_goldens"]

    entries = json.loads(manifest.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 3
    assert {Path(entry["path"]).name for entry in entries} == {
        GOLDEN_ATMOSPHERE_NAME,
        GOLDEN_SPECTRUM_NAME,
        ACCEPTANCE_NAME,
    }
    assert all(
        entry["builder"].endswith("verified_solar_goldens.py") for entry in entries
    )
    assert all(entry["requires_optional_full_catalog"] for entry in entries)
    validated = validate_published(output)
    assert validated["status"] == "accepted_exact_three_run_array_parity"


def test_physical_spectrum_difference_blocks_publication(tmp_path: Path) -> None:
    runs = _prepared_runs(tmp_path / "runs")
    changed = _spectrum_arrays(99.0)
    changed["normalized_flux"] = changed["normalized_flux"].copy()
    changed["normalized_flux"][3] += 1.0e-8
    write_npz(runs[-1].spectrum, changed)
    output = tmp_path / "output"

    with pytest.raises(ParityError, match="spectrum:normalized_flux.*payload"):
        publish(runs=runs, output_directory=output, update_manifest=False)
    assert not output.exists()


def test_atmosphere_payload_difference_blocks_publication(tmp_path: Path) -> None:
    runs = _prepared_runs(tmp_path / "runs")
    changed = _atmosphere_arrays()
    changed["temperature"] = changed["temperature"].copy()
    changed["temperature"][40] = np.nextafter(changed["temperature"][40], np.inf)
    write_npz(runs[1].atmosphere, changed)
    output = tmp_path / "output"

    with pytest.raises(ParityError, match="payload bytes differ"):
        publish(runs=runs, output_directory=output, update_manifest=False)
    assert not output.exists()


def test_repository_goldens_pass_catalog_free_validation() -> None:
    record = validate_published(DEFAULT_OUTPUT_DIRECTORY)
    assert record["evidence"]["atmosphere_array_parity"]["payload_sha256"] == (
        "d7cdb98e2e5e11f6d8eed4cb28d65e42bf64a0a15f53df917a0629a8efcb7e9a"
    )
    assert (
        record["evidence"]["spectrum_array_parity"]["physical_payload_sha256"]
        == "5e2b65add5326a9bfa0442216b8198b225305bc1ed0a05325858b34b2f345f27"
    )
    assert sha256(DEFAULT_OUTPUT_DIRECTORY / GOLDEN_ATMOSPHERE_NAME) == (
        "81d5163c755e0a755ca6f530ebf4ff06615efb64184b453135f7b59c42373ac3"
    )
    assert sha256(DEFAULT_OUTPUT_DIRECTORY / GOLDEN_SPECTRUM_NAME) == (
        "2131ab0c5a5681e568275a0da6054a6d7de65b82402aac10222050546dd15213"
    )
    assert sha256(DEFAULT_OUTPUT_DIRECTORY / ACCEPTANCE_NAME) == (
        "c7332dab6afc720df3c4f14b59185496edbeb1d5e93fc479c1c373baff4a66cd"
    )
