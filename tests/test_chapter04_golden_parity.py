"""Local-compute-before-archive parity gate for Chapter 4."""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

import book.chapter04_parity as parity_module
from book.chapter04_parity import compute_chapter04_local_parity_state


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_DIRECTORY = REPOSITORY_ROOT / "data" / "golden" / "payne_zero" / "chapter04"
ARCHIVE_NAMES = {
    "constants": "chapter04_molecular_constants_cpu_float64.npz",
    "atmosphere": "chapter04_atmosphere_molecular_state_cpu_float64.npz",
    "full": "chapter04_synthesis_molecular_full_cpu_float64.npz",
    "fixed": "chapter04_synthesis_molecular_fixed_cpu_float64.npz",
    "public": "chapter04_molecular_public_mapping_cpu_float64.npz",
}
EXPECTED_MEMBER_CLASSIFICATION = {
    "constants": {
        "all": (
            224,
            "f2ef207dd90ac50676a9d96223b0d5a69a2c4c36face548da5dd2f39f2187c39",
        ),
        "compared": (
            103,
            "054a02e0d2a23a9c6b65475d9d934f2fb13555069de09fe945f9246213b8e3f0",
        ),
        "alias": (
            0,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        "excluded": (
            121,
            "2c75b1f4b01413e1d1000823ecb7b1022fe1d5dd22afbd366db0420a267a499a",
        ),
        "archive_or_runtime_provenance": (
            102,
            "5d5c82ce31e31b58d5703681447eb3daf94454b3d9ad1ba8084f07d3a571ae17",
        ),
        "local_path_identity": (
            0,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        "oracle_trace_only": (
            19,
            "5c2190c44f41887cbcb163c33d0329367463d402915a716357de7b30fe706274",
        ),
    },
    "atmosphere": {
        "all": (
            392,
            "0945bad4cabba9fe454664c911bee177c67235844bb33088eff3bd2017dbfe05",
        ),
        "compared": (
            134,
            "ba73f25c4cb38571ffcf0ddc3a113ddd87cc27970db8211b4cd6a82d1bbf3a57",
        ),
        "alias": (
            0,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        "excluded": (
            258,
            "a0160b2ec88b71e0a7d0e4e9e3e3e9eef59a35e4eb7f7470dd47359e9d8a2b2c",
        ),
        "archive_or_runtime_provenance": (
            63,
            "cd2950c1593fb2c2bd727ba4af6bc1a4208ba85bf6e9ba60647fdeb6dbe1551e",
        ),
        "local_path_identity": (
            0,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        "oracle_trace_only": (
            195,
            "d10639db65590f935b0081754a85a320956bd8bfdb2dddde10285024788843fd",
        ),
    },
    "full": {
        "all": (
            161,
            "6101ab662e77d4bc397872c65182612178a2fbde21d3effc9b8420d91077ee58",
        ),
        "compared": (
            95,
            "9d3a9c3596937d86b5229b89adb27f099370ea68b54f395484831322bec89027",
        ),
        "alias": (
            11,
            "b68136d3a7edfb31adec2393dfb4533defeddcea23e6e17de0832ae778e4ad03",
        ),
        "excluded": (
            55,
            "0e548f9c96470d4e7b8c05e42166761931975cfb5790c838ac6a9625cc0fcede",
        ),
        "archive_or_runtime_provenance": (
            51,
            "83e0b2e01ce0c0c32aac5263d93c5134a2503aa5f15f6f2a43c4098da74b9752",
        ),
        "local_path_identity": (
            4,
            "271657e668981f6b5df8f4eca2218f3a57e07d1a2679b2ecd74cfdc10cdd6feb",
        ),
        "oracle_trace_only": (
            0,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
    },
    "fixed": {
        "all": (
            186,
            "b438e766e95be4daa0e1daacd3e5443cf56ee9d5f546f1e44abfbf83a20fc9a6",
        ),
        "compared": (
            113,
            "5aa609a013487385c06617156bfe20423c77eeb4f8a97a344eed6d4ac949062e",
        ),
        "alias": (
            18,
            "c93db66875ade31de6a2175c27b68f75ca9187947e5ac1de71e529d16437e905",
        ),
        "excluded": (
            55,
            "d9a08c3a51356a10d4777cbf2a802d5b74fdf4d3c4a9e9ae846fef51381d737f",
        ),
        "archive_or_runtime_provenance": (
            51,
            "83e0b2e01ce0c0c32aac5263d93c5134a2503aa5f15f6f2a43c4098da74b9752",
        ),
        "local_path_identity": (
            4,
            "b2188bd8c1226557dab57a0bce94dadcd40dd4f3ea282c2cbed65a71bbe42cf4",
        ),
        "oracle_trace_only": (
            0,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
    },
    "public": {
        "all": (
            208,
            "890a0c9170e5b6b2f0dbdc7deae6f0702627cd2ae55f8cd151e437787fa63da8",
        ),
        "compared": (
            123,
            "f54711a7d07850a0865af9c31db1ec11d4299a447b2d6b4d2c8f958aeed48a39",
        ),
        "alias": (
            20,
            "3a649c0528cf7d5972520cd2478b2ee8d44b7d9fdd4521c23982540ab8328b80",
        ),
        "excluded": (
            65,
            "b5bb17bb8648634a390cacb55d610473d5a9b554d4c51040da60ace7eac868ef",
        ),
        "archive_or_runtime_provenance": (
            51,
            "83e0b2e01ce0c0c32aac5263d93c5134a2503aa5f15f6f2a43c4098da74b9752",
        ),
        "local_path_identity": (
            2,
            "5f85edfe4ce6096da1732b6c2ebd7626be179dd94f1a87debb421c00b5a3637e",
        ),
        "oracle_trace_only": (
            12,
            "07a8937b6e5e2b0b1f15563b47971017bdc4107e53c8550677e7116a3e18502c",
        ),
    },
}
EXPECTED_ALIAS_AUTHORITY = {
    "full": (
        11,
        "e859787ac512b96c0c79a90fb96bcb3584f2667ddf211937d7f4c76f82503f5d",
    ),
    "fixed": (
        18,
        "7a9dd2f06df14e277a9910c5704112dc424d180e0cb9197caf1a8b479890c2b9",
    ),
    "public": (
        20,
        "ada73f18001cad2b60c9b019984d57854ab7dd5bfe9cdcf2534c92f8dd9ffebe",
    ),
}
EXPECTED_GLOBAL_ALIAS_AUTHORITY_SHA256 = (
    "6c6aac01f6122c4c484f8bc5544a8374f40af1697aec03a9d80616a18605665d"
)
REFERENCE_ARCHIVE_BASENAMES = frozenset(ARCHIVE_NAMES.values())


def _resolve_archive_directory() -> Path:
    """Prefer an explicit prepublication directory, then the published data."""

    explicit = os.environ.get("CHAPTER04_CANDIDATE_DIR")
    if explicit:
        directory = Path(explicit).expanduser().resolve()
        if not directory.is_dir():
            raise unittest.SkipTest(
                f"CHAPTER04_CANDIDATE_DIR does not exist: {directory}"
            )
        return directory
    if PUBLISHED_DIRECTORY.is_dir():
        return PUBLISHED_DIRECTORY.resolve()
    raise unittest.SkipTest(
        "Chapter 4 parity archives are unavailable: set "
        "CHAPTER04_CANDIDATE_DIR or publish data/golden/payne_zero/chapter04"
    )


def _load_event_kind(file: object) -> str:
    """Recognize a forbidden reference archive before its directory is known."""

    path_like = getattr(file, "name", file)
    return (
        "archive_open"
        if Path(os.fsdecode(path_like)).name in REFERENCE_ARCHIVE_BASENAMES
        else "local_data_open"
    )


def _load_archives(directory: Path) -> dict[str, dict[str, np.ndarray]]:
    """Open all five archives only after the caller supplies local state."""

    archives: dict[str, dict[str, np.ndarray]] = {}
    for label, filename in ARCHIVE_NAMES.items():
        path = directory / filename
        if not path.is_file():
            raise AssertionError(f"Chapter 4 archive is missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            archives[label] = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
    return archives


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _member(
    arrays: dict[str, np.ndarray],
    name: str,
) -> np.ndarray:
    """Resolve one compact-archive member or its declared lossless alias."""

    if name in arrays:
        return arrays[name]
    alias_names = [f"alias__{name}_member", f"{name}_member"]
    if "__" in name:
        branch, remainder = name.split("__", maxsplit=1)
        alias_names.append(f"{branch}__alias__{remainder}_member")
    for alias_name in alias_names:
        if alias_name not in arrays:
            continue
        authority = _scalar_text(arrays[alias_name])
        if "[:," in authority and authority.endswith("]"):
            member, index_text = authority[:-1].split("[:,", maxsplit=1)
            return arrays[member][:, int(index_text)]
        return arrays[authority]
    raise KeyError(f"neither {name!r} nor a declared alias is present")


def _authority_member_name(
    arrays: dict[str, np.ndarray],
    name: str,
) -> str:
    """Return the archive member that physically owns a requested value."""

    if name in arrays:
        return name
    alias_names = [f"alias__{name}_member", f"{name}_member"]
    if "__" in name:
        branch, remainder = name.split("__", maxsplit=1)
        alias_names.append(f"{branch}__alias__{remainder}_member")
    for alias_name in alias_names:
        if alias_name not in arrays:
            continue
        authority = _scalar_text(arrays[alias_name])
        return authority.split("[:,", maxsplit=1)[0]
    raise KeyError(f"no physical authority exists for {name!r}")


def _member_name_digest(names: set[str]) -> str:
    """Hash an exact lexical member-name inventory."""

    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def _alias_authority_digest(
    archive: dict[str, np.ndarray],
) -> tuple[int, str]:
    """Bind every compact alias name to its exact authority expression."""

    pairs = sorted(
        (name, _scalar_text(values))
        for name, values in archive.items()
        if name.endswith("_member")
    )
    payload = "\n".join(f"{name}\0{authority}" for name, authority in pairs).encode()
    return len(pairs), hashlib.sha256(payload).hexdigest()


def _locally_compared_member_registry(
    local: dict,
    archives: dict[str, dict[str, np.ndarray]],
) -> dict[str, set[str]]:
    """Enumerate every archive member reached by an independent local result."""

    compared = {label: set() for label in archives}

    compared["constants"].update(f"synthesis__{name}" for name in local["catalog"])
    compared["constants"].update(
        f"atmosphere__{name.replace('__', '_', 1)}"
        for name in local["atmosphere"]
        if name.startswith(("boundary__", "h2_probe__"))
    )
    compared["constants"].update(
        {
            "atmosphere__named_molecule_codes",
            "atmosphere__named_molecule_catalog_indices",
        }
    )
    compared["constants"].update(
        f"synthesis__{name}"
        for name in local["synthesis"]
        if name.startswith("boundary__")
    )
    compared["constants"].update(
        name for name in archives["constants"] if name.startswith("synthesis__trace__")
    )

    compared["atmosphere"].update(
        {
            "full_column_mass",
            "full_temperature",
            "full_gas_pressure",
            "full_input_electron_density",
        }
    )
    compared["atmosphere"].update(
        {
            "full_molecular_populations",
            "full_partition_normalized_molecular_populations",
            "full_equation_densities_after_normalization",
            "full_previous_molecular_equation_densities",
            "full_output_electron_density",
            "full_total_nuclei_number_density",
            "full_mass_density",
            "full_charge_square_density",
            "full_ion_stage_populations_by_packed_slot",
            "full_partition_normalized_populations_by_packed_slot",
            "full_fractional_doppler_widths",
            "full_partition_normalized_population_over_mass_density_and_width",
            "full_specific_internal_energy",
            "full_major_isotope_mass_amu",
            "full_temperature_iteration_cache_value",
            "full_population_constants",
            "full_fractional_doppler_width_structural_infinity_mask",
            "full_fractional_doppler_width_structural_infinity_slots",
            "full_layer_seed_equation_densities",
            "full_equation_densities_before_normalization",
            "full_newton_iteration_count",
            "full_newton_converged",
            "full_newton_exhausted",
            "full_postsolve_row_scaled_residual_max",
            "full_raw_populations_before_fill",
            "full_normalized_populations_after_fill",
            "full_physical_equation_densities_saved_before_fill",
            "full_transformed_molecular_equation_densities",
            "full_postsolve_residual",
        }
    )
    for route_prefix in (
        "handoff__",
        "disabled__",
        "temperature_control__",
        "pressure_control__",
        "mode2__",
        "mode12__",
        "energy__",
        "bridge__",
        "schedule_inventory__",
    ):
        compared["atmosphere"].update(
            name.replace("__", "_", 1)
            for name in local["atmosphere"]
            if name.startswith(route_prefix)
        )

    synthesis_routes = (
        ("full", "full", "full__state__", "state__", 2),
        ("derived", "fixed", "derived__state__", "derived__state__", 1),
        ("supplied", "fixed", "supplied__state__", "supplied__state__", 1),
    )
    for (
        branch,
        archive_label,
        local_prefix,
        archive_prefix,
        call_count,
    ) in synthesis_routes:
        archive = archives[archive_label]
        for name in local["synthesis"]:
            if name.startswith(local_prefix):
                requested = f"{archive_prefix}{name.removeprefix(local_prefix)}"
                compared[archive_label].add(_authority_member_name(archive, requested))
        for call_index in range(call_count):
            local_call_prefix = f"{branch}__call_{call_index}__"
            archive_call_prefix = (
                f"call_{call_index}__"
                if branch == "full"
                else f"{branch}__call_{call_index}__"
            )
            for name in local["synthesis"]:
                if not name.startswith(local_call_prefix):
                    continue
                call_name = name.removeprefix(local_call_prefix)
                if call_name in {"caller_file", "molecules_path"}:
                    continue
                compared[archive_label].add(
                    _authority_member_name(
                        archive,
                        f"{archive_call_prefix}{call_name}",
                    )
                )
        compared[archive_label].add(
            (
                "trace__molecular_call_count"
                if branch == "full"
                else f"{branch}__trace__molecular_call_count"
            )
        )
    compared["full"].update(
        {
            "input__temperature",
            "input__gas_pressure",
            "input__electron_density_seed",
            "input__elemental_abundances",
            "trace__route",
            "trace__published_electron_from_call",
            "trace__published_molecules_from_call",
        }
    )
    compared["fixed"].update(
        {
            "input__temperature",
            "input__gas_pressure",
            "input__electron_density_seed",
            "input__elemental_abundances",
            "supplied__input__mass_density",
            "derived__trace__route",
            "derived__trace__published_electron_from_input",
            "supplied__trace__route",
            "supplied__trace__published_electron_from_input",
        }
    )

    public = archives["public"]
    compared["public"].update(
        {
            "input__temperature",
            "input__gas_pressure",
            "input__electron_density_seed",
            "input__elemental_abundances",
            "schema__structured_keyset",
        }
    )
    for prefix in ("structured__", "fixed_state__", "call_0__"):
        for name in local["public"]:
            if not name.startswith(prefix):
                continue
            if name in {f"{prefix}caller_file", f"{prefix}molecules_path"}:
                continue
            compared["public"].add(_authority_member_name(public, name))
    for name in (
        "mapping__species_codes",
        "mapping__molecule_code_offsets",
        "mapping__molecule_codes",
        "mapping__public_columns",
        "partition__elements_without_ground_floor",
        "partition__without_ground_floor_stage_counts",
        "partition__without_ground_floor_offsets",
        "partition__without_ground_floor",
        "partition__grounded_cube",
        "partition__bridge_cube",
        "line_population__public",
        "line_population__independent_no_ground",
        "line_population__grounded",
        "line_population__ground_discrimination_mask",
        "line_population__ground_discrimination_max_abs",
        "co_reconstruction__line_species_code",
        "co_reconstruction__equilibrium_code",
        "co_reconstruction__public_stage_index",
        "co_reconstruction__public_column_index",
        "co_reconstruction__catalog_index",
        "co_reconstruction__component_equation_indices",
        "co_reconstruction__component_species_codes",
        "co_reconstruction__component_atomic_masses_amu",
        "co_reconstruction__molecular_mass_amu",
        "co_reconstruction__leading_coefficient",
        "co_reconstruction__temperature",
        "co_reconstruction__equation_densities",
        "co_reconstruction__transformed_equation_densities",
        "co_reconstruction__no_ground_neutral_partitions",
        "co_reconstruction__normalization",
        "co_reconstruction__raw_equilibrium_population",
        "co_reconstruction__reference_public_lane",
        "co_reconstruction__independent_population",
        "co_reconstruction__raw_discrimination_mask",
        "molecular_hydrogen__catalog_index",
        "molecular_hydrogen__solved_code_101",
        "partition_cube__before",
        "partition_cube__after",
        "ion_cube__before",
        "ion_cube__after",
    ):
        compared["public"].add(_authority_member_name(public, name))
    for name in (
        "signed_continuum_edge_frequency_hz",
        "continuum_edge_wavelength_nm",
        "continuum_edge_wavenumber_cm",
        "continuum_edge_sample_frequency_hz",
        "continuum_edge_midpoint_wavelength_nm",
        "edge_interval_width_squared_over_two_nm2",
    ):
        compared["public"].add(_authority_member_name(public, f"edge_loader__{name}"))
    compared["public"].update(
        {
            "trace__fixed_eos_call_count",
            "trace__molecular_solve_call_count",
            "trace__edge_grid_call_count",
            "trace__reused_fixed_molecular_arrays",
        }
    )
    return compared


def _assert_exact(
    testcase: unittest.TestCase,
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
) -> None:
    try:
        np.testing.assert_array_equal(actual, expected, strict=True)
    except AssertionError as error:
        raise AssertionError(f"exact parity failed for {label}: {error}") from error


class Chapter04GoldenParityTests(unittest.TestCase):
    """Require self-contained local science to reproduce the compact archives."""

    @classmethod
    def setUpClass(cls) -> None:
        original_load = np.load
        events: list[tuple[str, str]] = []

        def observed_load(file, *args, **kwargs):
            path_text = str(file)
            event = _load_event_kind(file)
            events.append((event, path_text))
            return original_load(file, *args, **kwargs)

        with mock.patch("numpy.load", side_effect=observed_load):
            cls.local = compute_chapter04_local_parity_state()
            events.append(("local_compute_complete", ""))
            cls.directory = _resolve_archive_directory()
            cls.archives = _load_archives(cls.directory)
        cls.load_events = events

    def test_local_compute_finishes_before_first_archive_open(self) -> None:
        event_names = [event for event, _ in self.load_events]
        completion_index = event_names.index("local_compute_complete")
        first_archive_index = event_names.index("archive_open")
        self.assertLess(completion_index, first_archive_index)
        self.assertGreater(
            sum(event == "local_data_open" for event in event_names),
            0,
        )
        for filename in REFERENCE_ARCHIVE_BASENAMES:
            self.assertEqual(
                _load_event_kind(Path("/unresolved/candidate/directory") / filename),
                "archive_open",
            )
        self.assertEqual(
            _load_event_kind(Path("/local/static/data") / "molecules.npz"),
            "local_data_open",
        )
        named_file_object = mock.Mock()
        named_file_object.name = str(
            Path("/unresolved/candidate/directory") / ARCHIVE_NAMES["constants"]
        )
        self.assertEqual(
            _load_event_kind(named_file_object),
            "archive_open",
        )
        self.assertEqual(
            _load_event_kind(
                os.fsencode(
                    Path("/unresolved/candidate/directory") / ARCHIVE_NAMES["public"]
                )
            ),
            "archive_open",
        )

    def test_local_compute_source_has_no_expected_result_dependency(self) -> None:
        source = inspect.getsource(parity_module)
        lowered = source.lower()
        for forbidden in (
            "chapter04_candidate_dir",
            "data/golden",
            "/private/tmp",
            "/users/",
            "chapter04_oracle_worker",
            "build_chapter04_payne_zero_goldens",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_every_scientific_output_is_detached_and_read_only(self) -> None:
        for section in ("inputs", "catalog", "atmosphere", "synthesis", "public"):
            for name, values in self.local[section].items():
                with self.subTest(section=section, name=name):
                    self.assertIsInstance(values, np.ndarray)
                    self.assertFalse(values.flags.writeable)
                    self.assertFalse(values.dtype.hasobject)

    def test_constants_links_and_input_identity(self) -> None:
        constants_path = self.directory / ARCHIVE_NAMES["constants"]
        constants_digest = hashlib.sha256(constants_path.read_bytes()).hexdigest()
        for label in ("atmosphere", "full", "fixed", "public"):
            self.assertEqual(
                _scalar_text(self.archives[label]["meta__constants_archive_sha256"]),
                constants_digest,
            )

        inputs = self.local["inputs"]
        atmosphere = self.archives["atmosphere"]
        for input_name, archive_name in (
            ("column_mass", "full_column_mass"),
            ("temperature", "full_temperature"),
            ("gas_pressure", "full_gas_pressure"),
            ("electron_density_seed", "full_input_electron_density"),
        ):
            _assert_exact(
                self,
                inputs[input_name],
                atmosphere[archive_name],
                f"atmosphere:{archive_name}",
            )
        for label in ("full", "fixed", "public"):
            archive = self.archives[label]
            for name in (
                "temperature",
                "gas_pressure",
                "electron_density_seed",
                "elemental_abundances",
            ):
                _assert_exact(
                    self,
                    inputs[name],
                    archive[f"input__{name}"],
                    f"{label}:input__{name}",
                )

    def test_complete_catalog_buffers_and_semantic_overlap(self) -> None:
        constants = self.archives["constants"]
        for name, values in self.local["catalog"].items():
            archive_name = f"synthesis__{name}"
            self.assertIn(archive_name, constants)
            _assert_exact(
                self,
                values,
                constants[archive_name],
                archive_name,
            )
        catalog = self.local["catalog"]
        self.assertEqual(int(catalog["alignment__shared_count"]), 170)
        self.assertEqual(
            int(catalog["alignment__synthesis_only_count"]),
            20,
        )
        self.assertEqual(
            int(catalog["alignment__atmosphere_only_count"]),
            0,
        )
        self.assertEqual(
            int(catalog["alignment__semantic_mismatch_count"]),
            0,
        )
        self.assertEqual(
            np.count_nonzero(catalog["alignment__shared_row_indices_differ_mask"]),
            64,
        )

    def test_reconstructible_atmosphere_boundary_and_h2_probes(self) -> None:
        constants = self.archives["constants"]
        local = self.local["atmosphere"]
        compared_count = 0
        for prefix in ("boundary__", "h2_probe__"):
            for name, values in local.items():
                if not name.startswith(prefix):
                    continue
                archive_name = f"atmosphere__{name.replace('__', '_', 1)}"
                self.assertIn(archive_name, constants)
                _assert_exact(
                    self,
                    values,
                    constants[archive_name],
                    archive_name,
                )
                compared_count += 1
        self.assertEqual(compared_count, 29)
        named_codes = self.local["inputs"]["named_molecule_codes"]
        atmosphere_codes = self.local["catalog"]["catalog__atmosphere__molecule_codes"]
        atmosphere_count = int(
            self.local["catalog"]["catalog__atmosphere__molecule_count"]
        )
        named_indices = np.asarray(
            [
                int(
                    np.flatnonzero(
                        np.isclose(
                            atmosphere_codes[:atmosphere_count],
                            code,
                            rtol=0.0,
                            atol=1.0e-8,
                        )
                    )[0]
                )
                for code in named_codes
            ],
            dtype=np.int64,
        )
        _assert_exact(
            self,
            named_codes,
            constants["atmosphere__named_molecule_codes"],
            "atmosphere:named molecule codes",
        )
        _assert_exact(
            self,
            named_indices,
            constants["atmosphere__named_molecule_catalog_indices"],
            "atmosphere:named molecule catalog indices",
        )

    def test_reconstructible_synthesis_boundary_probes(self) -> None:
        constants = self.archives["constants"]
        local = self.local["synthesis"]
        names = [name for name in local if name.startswith("boundary__")]
        self.assertEqual(len(names), 23)
        for name in names:
            archive_name = f"synthesis__{name}"
            self.assertIn(archive_name, constants)
            _assert_exact(
                self,
                local[name],
                constants[archive_name],
                archive_name,
            )
        self.assertEqual(
            _scalar_text(constants["synthesis__trace__route"]),
            "synthesis_boundary_probes",
        )
        self.assertEqual(
            int(constants["synthesis__trace__provisional_h2_fixed_eos_call_count"]),
            1,
        )
        self.assertEqual(
            int(
                constants["synthesis__trace__provisional_h2_molecular_solve_call_count"]
            ),
            1,
        )

    def test_atmosphere_final_state_continuation_and_structural_infinities(
        self,
    ) -> None:
        local = self.local["atmosphere"]
        archive = self.archives["atmosphere"]
        exact_members = {
            "molecular_populations": "full_molecular_populations",
            "partition_normalized_molecular_populations": (
                "full_partition_normalized_molecular_populations"
            ),
            "molecular_equation_densities": (
                "full_equation_densities_after_normalization"
            ),
            "previous_molecular_equation_densities": (
                "full_previous_molecular_equation_densities"
            ),
            "electron_density": "full_output_electron_density",
            "total_nuclei_number_density": ("full_total_nuclei_number_density"),
            "mass_density": "full_mass_density",
            "charge_square_density": "full_charge_square_density",
            "ion_stage_populations_by_packed_slot": (
                "full_ion_stage_populations_by_packed_slot"
            ),
            "partition_normalized_populations_by_packed_slot": (
                "full_partition_normalized_populations_by_packed_slot"
            ),
            "fractional_doppler_widths": "full_fractional_doppler_widths",
            "partition_normalized_population_over_mass_density_and_width": (
                "full_partition_normalized_population_over_mass_density_and_width"
            ),
            "specific_internal_energy": "full_specific_internal_energy",
            "major_isotope_mass_amu": "full_major_isotope_mass_amu",
            "temperature_iteration_cache_value": (
                "full_temperature_iteration_cache_value"
            ),
            "population_constants": "full_population_constants",
            "fractional_doppler_width_structural_infinity_mask": (
                "full_fractional_doppler_width_structural_infinity_mask"
            ),
            "fractional_doppler_width_structural_infinity_slots": (
                "full_fractional_doppler_width_structural_infinity_slots"
            ),
            "continuation_seed": "full_layer_seed_equation_densities",
            "continuation_solution": ("full_equation_densities_before_normalization"),
            "continuation_iteration_count": "full_newton_iteration_count",
            "continuation_converged": "full_newton_converged",
            "continuation_exhausted": "full_newton_exhausted",
            "continuation_residual_max_scaled": (
                "full_postsolve_row_scaled_residual_max"
            ),
        }
        for local_name, archive_name in exact_members.items():
            _assert_exact(
                self,
                local[local_name],
                archive[archive_name],
                f"atmosphere:{archive_name}",
            )
        for route_prefix in (
            "handoff__",
            "disabled__",
            "temperature_control__",
            "pressure_control__",
            "mode2__",
            "mode12__",
            "energy__",
            "bridge__",
            "schedule_inventory__",
        ):
            for local_name, values in local.items():
                if not local_name.startswith(route_prefix):
                    continue
                archive_name = local_name.replace("__", "_", 1)
                self.assertIn(archive_name, archive)
                _assert_exact(
                    self,
                    values,
                    archive[archive_name],
                    f"atmosphere:{archive_name}",
                )
        for local_name, archive_name in (
            (
                "molecular_populations",
                "full_raw_populations_before_fill",
            ),
            (
                "partition_normalized_molecular_populations",
                "full_normalized_populations_after_fill",
            ),
            (
                "previous_molecular_equation_densities",
                "full_equation_densities_before_normalization",
            ),
            (
                "previous_molecular_equation_densities",
                "full_physical_equation_densities_saved_before_fill",
            ),
            (
                "molecular_equation_densities",
                "full_transformed_molecular_equation_densities",
            ),
        ):
            _assert_exact(
                self,
                local[local_name],
                archive[archive_name],
                f"atmosphere:{archive_name}",
            )
        _assert_exact(
            self,
            local["continuation_residual_max_abs"],
            np.max(np.abs(archive["full_postsolve_residual"]), axis=1),
            "atmosphere:independent postsolve residual maximum",
        )
        self.assertTrue(np.all(local["continuation_seed_equal"]))
        self.assertFalse(np.any(local["continuation_exhausted"]))
        self.assertTrue(
            np.all(np.isposinf(local["fractional_doppler_widths"][:, [919, 927]]))
        )
        finite_mask = ~local["fractional_doppler_width_structural_infinity_mask"]
        self.assertTrue(
            np.all(np.isfinite(local["fractional_doppler_widths"][finite_mask]))
        )
        self.assertGreater(
            np.count_nonzero(local["specific_internal_energy"]),
            0,
        )

    def test_synthesis_full_and_fixed_public_states(self) -> None:
        local = self.local["synthesis"]
        branches = (
            ("full", self.archives["full"], "state__"),
            ("derived", self.archives["fixed"], "derived__state__"),
            ("supplied", self.archives["fixed"], "supplied__state__"),
        )
        for local_branch, archive, archive_prefix in branches:
            local_prefix = f"{local_branch}__state__"
            state_names = [
                name.removeprefix(local_prefix)
                for name in local
                if name.startswith(local_prefix)
            ]
            self.assertGreaterEqual(len(state_names), 18)
            for state_name in state_names:
                _assert_exact(
                    self,
                    local[f"{local_prefix}{state_name}"],
                    _member(archive, f"{archive_prefix}{state_name}"),
                    f"{local_branch}:state:{state_name}",
                )
        _assert_exact(
            self,
            local["supplied__input__mass_density"],
            self.archives["fixed"]["supplied__input__mass_density"],
            "fixed:supplied mass derived only from local inputs",
        )

    def test_synthesis_exact_molecular_calls_and_no_exhaustion(self) -> None:
        local = self.local["synthesis"]
        routes = (
            ("full", 2, self.archives["full"], ""),
            ("derived", 1, self.archives["fixed"], "derived__"),
            ("supplied", 1, self.archives["fixed"], "supplied__"),
        )
        for branch, count, archive, archive_prefix in routes:
            self.assertEqual(
                int(local[f"{branch}__molecular_call_count"]),
                count,
            )
            for call_index in range(count):
                local_prefix = f"{branch}__call_{call_index}__"
                archive_call_prefix = f"{archive_prefix}call_{call_index}__"
                call_names = [
                    name.removeprefix(local_prefix)
                    for name in local
                    if name.startswith(local_prefix)
                ]
                for call_name in call_names:
                    if call_name in {"caller_file", "molecules_path"}:
                        continue
                    _assert_exact(
                        self,
                        local[f"{local_prefix}{call_name}"],
                        _member(
                            archive,
                            f"{archive_call_prefix}{call_name}",
                        ),
                        f"{branch}:call_{call_index}:{call_name}",
                    )
                exhausted = local[f"{local_prefix}diag__exhausted_mask"]
                self.assertFalse(np.any(exhausted))
                populations = local[f"{local_prefix}output__molecular_populations"]
                equations = local[f"{local_prefix}output__equation_densities"]
                np.testing.assert_array_equal(populations[:, 190:], 0.0)
                np.testing.assert_array_equal(equations[:, 23:], 0.0)
            _assert_exact(
                self,
                local[f"{branch}__molecular_call_count"],
                archive[f"{archive_prefix}trace__molecular_call_count"],
                f"{branch}:trace:molecular_call_count",
            )

        # These are intentionally different electron-density concepts.
        _assert_exact(
            self,
            local["full__state__electron_density"],
            local["full__call_0__output__electron_density"],
            "full published electron comes from call zero",
        )
        self.assertFalse(
            np.array_equal(
                local["full__state__electron_density"],
                local["full__call_1__output__electron_density"],
            )
        )
        _assert_exact(
            self,
            local["derived__state__electron_density"],
            self.local["inputs"]["electron_density_seed"],
            "fixed published electron remains the supplied input",
        )
        self.assertFalse(
            np.array_equal(
                local["derived__state__electron_density"],
                local["derived__call_0__output__electron_density"],
            )
        )
        full = self.archives["full"]
        self.assertEqual(_scalar_text(full["trace__route"]), "full")
        self.assertEqual(int(full["trace__published_electron_from_call"]), 0)
        self.assertEqual(int(full["trace__published_molecules_from_call"]), 1)
        fixed = self.archives["fixed"]
        for branch, route in (
            ("derived", "fixed_derived_mass"),
            ("supplied", "fixed_supplied_mass"),
        ):
            self.assertEqual(
                _scalar_text(fixed[f"{branch}__trace__route"]),
                route,
            )
            self.assertTrue(
                bool(fixed[f"{branch}__trace__published_electron_from_input"])
            )

    def test_public_structured_mapping_lanes_h2_and_edges(self) -> None:
        local = self.local["public"]
        archive = self.archives["public"]
        structured_names = [
            name.removeprefix("structured__")
            for name in local
            if name.startswith("structured__")
        ]
        self.assertEqual(len(structured_names), 27)
        self.assertEqual(
            set(structured_names),
            set(archive["schema__structured_keyset"].tolist()),
        )
        for name in structured_names:
            _assert_exact(
                self,
                local[f"structured__{name}"],
                archive[f"structured__{name}"],
                f"public:structured:{name}",
            )

        for name in (
            "mapping__species_codes",
            "mapping__molecule_code_offsets",
            "mapping__molecule_codes",
            "mapping__public_columns",
            "partition__elements_without_ground_floor",
            "partition__without_ground_floor_stage_counts",
            "partition__without_ground_floor_offsets",
            "partition__without_ground_floor",
            "partition__grounded_cube",
            "partition__bridge_cube",
            "line_population__public",
            "line_population__independent_no_ground",
            "line_population__grounded",
            "line_population__ground_discrimination_mask",
            "co_reconstruction__line_species_code",
            "co_reconstruction__equilibrium_code",
            "co_reconstruction__public_stage_index",
            "co_reconstruction__public_column_index",
            "co_reconstruction__catalog_index",
            "co_reconstruction__component_equation_indices",
            "co_reconstruction__component_species_codes",
            "co_reconstruction__component_atomic_masses_amu",
            "co_reconstruction__molecular_mass_amu",
            "co_reconstruction__leading_coefficient",
            "co_reconstruction__temperature",
            "co_reconstruction__equation_densities",
            "co_reconstruction__transformed_equation_densities",
            "co_reconstruction__no_ground_neutral_partitions",
            "co_reconstruction__normalization",
            "co_reconstruction__raw_equilibrium_population",
            "co_reconstruction__reference_public_lane",
            "co_reconstruction__independent_population",
            "co_reconstruction__raw_discrimination_mask",
            "molecular_hydrogen__catalog_index",
            "molecular_hydrogen__solved_code_101",
            "line_population__ground_discrimination_max_abs",
        ):
            _assert_exact(
                self,
                local[name],
                _member(archive, name),
                f"public:{name}",
            )

        for name in (
            "partition_cube__before",
            "partition_cube__after",
            "ion_cube__before",
            "ion_cube__after",
        ):
            _assert_exact(
                self,
                local[name],
                _member(archive, name),
                f"public:{name}",
            )
        for name in (
            "signed_continuum_edge_frequency_hz",
            "continuum_edge_wavelength_nm",
            "continuum_edge_wavenumber_cm",
            "continuum_edge_sample_frequency_hz",
            "continuum_edge_midpoint_wavelength_nm",
            "edge_interval_width_squared_over_two_nm2",
        ):
            _assert_exact(
                self,
                local[f"edge_loader__{name}"],
                _member(archive, f"edge_loader__{name}"),
                f"public:edge:{name}",
            )

        columns = local["mapping__public_columns"]
        self.assertEqual(columns.shape, (54,))
        self.assertEqual(np.count_nonzero(columns <= 98), 51)
        np.testing.assert_array_equal(columns[columns > 98], [129, 130, 131])
        self.assertTrue(np.all(local["normalized_delta"][:, :5, :] == 0.0))
        self.assertTrue(
            np.all(local["normalized_delta"][:, 5, ~local["owned_stage5_mask"]] == 0.0)
        )
        _assert_exact(
            self,
            local["ion_cube__before"],
            local["ion_cube__after"],
            "public actual populations remain unchanged",
        )
        self.assertTrue(np.all(local["co_reconstruction__raw_discrimination_mask"]))
        self.assertEqual(int(local["fixed_eos_call_count"]), 1)
        self.assertEqual(int(local["molecular_solve_call_count"]), 1)
        self.assertEqual(int(local["edge_grid_call_count"]), 1)
        self.assertTrue(bool(local["reused_fixed_molecular_arrays"]))
        for local_name, archive_name in (
            ("fixed_eos_call_count", "trace__fixed_eos_call_count"),
            ("molecular_solve_call_count", "trace__molecular_solve_call_count"),
            ("edge_grid_call_count", "trace__edge_grid_call_count"),
            (
                "reused_fixed_molecular_arrays",
                "trace__reused_fixed_molecular_arrays",
            ),
        ):
            _assert_exact(
                self,
                local[local_name],
                archive[archive_name],
                f"public ownership:{archive_name}",
            )

    def test_public_fixed_state_and_owned_call_are_exactly_reconstructible(
        self,
    ) -> None:
        local = self.local["public"]
        archive = self.archives["public"]
        for prefix in ("fixed_state__", "call_0__"):
            names = [name for name in local if name.startswith(prefix)]
            self.assertGreater(len(names), 0)
            for name in names:
                if name in {
                    f"{prefix}caller_file",
                    f"{prefix}molecules_path",
                }:
                    continue
                _assert_exact(
                    self,
                    local[name],
                    _member(archive, name),
                    f"public:{name}",
                )

    def test_every_declared_lossless_alias_resolves(self) -> None:
        alias_count = 0
        global_pairs: list[tuple[str, str, str]] = []
        for archive_name in ("full", "fixed", "public"):
            archive = self.archives[archive_name]
            self.assertEqual(
                _alias_authority_digest(archive),
                EXPECTED_ALIAS_AUTHORITY[archive_name],
            )
            for name, values in archive.items():
                if not name.endswith("_member"):
                    continue
                authority = _scalar_text(values)
                global_pairs.append((archive_name, name, authority))
                if "[:," in authority and authority.endswith("]"):
                    member, index_text = authority[:-1].split(
                        "[:,",
                        maxsplit=1,
                    )
                    self.assertIn(member, archive)
                    index = int(index_text)
                    self.assertGreaterEqual(index, 0)
                    self.assertLess(index, archive[member].shape[1])
                    resolved = archive[member][:, index]
                else:
                    self.assertIn(authority, archive)
                    resolved = archive[authority]
                requested = name.removesuffix("_member")
                if "__alias__" in requested:
                    requested = requested.replace("__alias__", "__", 1)
                elif requested.startswith("alias__"):
                    requested = requested.removeprefix("alias__")
                _assert_exact(
                    self,
                    _member(archive, requested),
                    resolved,
                    f"{archive_name}:alias:{name}",
                )
                alias_count += 1
        self.assertEqual(alias_count, 49)
        global_payload = "\n".join(
            f"{archive}\0{name}\0{authority}"
            for archive, name, authority in sorted(global_pairs)
        ).encode()
        self.assertEqual(
            hashlib.sha256(global_payload).hexdigest(),
            EXPECTED_GLOBAL_ALIAS_AUTHORITY_SHA256,
        )
        redirected = dict(self.archives["public"])
        redirected["alias__line_population__no_ground_member"] = np.asarray(
            "line_population__grounded"
        )
        self.assertNotEqual(
            _alias_authority_digest(redirected),
            EXPECTED_ALIAS_AUTHORITY["public"],
        )

    def test_every_archive_member_is_fail_closed_and_classified(self) -> None:
        compared = _locally_compared_member_registry(
            self.local,
            self.archives,
        )
        for label, archive in self.archives.items():
            with self.subTest(archive=label):
                all_members = set(archive)
                alias_members = {
                    name for name in all_members if name.endswith("_member")
                }
                excluded_members = all_members - compared[label] - alias_members
                categories = {
                    "archive_or_runtime_provenance": set(),
                    "local_path_identity": set(),
                    "oracle_trace_only": set(),
                }
                for name in excluded_members:
                    if name.endswith(("caller_file", "molecules_path")):
                        categories["local_path_identity"].add(name)
                    elif name.startswith(("meta__", "oracle__")) or (
                        label == "constants"
                        and (
                            name.startswith("synthesis__meta__")
                            or (
                                name.startswith("atmosphere__")
                                and not name.startswith("atmosphere__boundary_")
                            )
                        )
                    ):
                        categories["archive_or_runtime_provenance"].add(name)
                    else:
                        categories["oracle_trace_only"].add(name)

                partitions = {
                    "all": all_members,
                    "compared": compared[label],
                    "alias": alias_members,
                    "excluded": excluded_members,
                    **categories,
                }
                expected = EXPECTED_MEMBER_CLASSIFICATION[label]
                for category, members in partitions.items():
                    expected_count, expected_digest = expected[category]
                    self.assertEqual(len(members), expected_count, category)
                    self.assertEqual(
                        _member_name_digest(members),
                        expected_digest,
                        category,
                    )
                self.assertFalse(compared[label] & alias_members)
                self.assertFalse(compared[label] & excluded_members)
                self.assertFalse(alias_members & excluded_members)
                self.assertEqual(
                    all_members,
                    compared[label] | alias_members | excluded_members,
                )
                self.assertEqual(
                    excluded_members,
                    set().union(*categories.values()),
                )
                for left_name, left_members in categories.items():
                    for right_name, right_members in categories.items():
                        if left_name >= right_name:
                            continue
                        self.assertFalse(
                            left_members & right_members,
                            f"{left_name} overlaps {right_name}",
                        )

    def test_oracle_only_traces_are_explicitly_excluded(self) -> None:
        self.assertEqual(
            set(self.local["excluded_oracle_trace_ownership"]),
            {
                "archive_publication_metadata",
                "runtime_environment_and_executed_source_provenance",
                "atmosphere_per_iteration_and_lifecycle_instrumentation",
                "synthesis_local_path_identity",
                "public_builder_trace_instrumentation",
            },
        )


if __name__ == "__main__":
    unittest.main()
