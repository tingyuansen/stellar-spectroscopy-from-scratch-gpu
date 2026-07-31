"""Pure Chapter 6 conversion from one raw Fe I row to atmosphere line words.

The converter is intentionally provenance-bound to raw synthesis-source row
873702.  It performs no catalog search, opens no optional upstream catalog,
and has no output path.  The lossy packed result is an atmosphere-kernel input;
it is not evidence that the synthesis and atmosphere catalogs share row
identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SOURCE_ROOT = (REPOSITORY_ROOT / "src").resolve()
for _module_name, _module in tuple(sys.modules.items()):
    if not _module_name.startswith("payne_zero_atmosphere"):
        continue
    _module_file = getattr(_module, "__file__", None)
    if _module_file is not None and not Path(_module_file).resolve().is_relative_to(
        LOCAL_SOURCE_ROOT
    ):
        raise RuntimeError(
            f"{_module_name} was imported from outside the textbook's staged "
            f"source tree: {_module_file}"
        )
_local_source_text = str(LOCAL_SOURCE_ROOT)
if _local_source_text in sys.path:
    sys.path.remove(_local_source_text)
sys.path.insert(0, _local_source_text)

from payne_zero_atmosphere.line_catalog import SelectedLineCatalog  # noqa: E402
from payne_zero_atmosphere.population_layout import (  # noqa: E402
    atomic_population_slot_start,
)


CANONICAL_SUBSET_PATH = (
    REPOSITORY_ROOT / "data/subsets/chapter06_fe_i_source_row_873702.npz"
)
CANONICAL_SUBSET_RELATIVE_PATH = "data/subsets/chapter06_fe_i_source_row_873702.npz"
CANONICAL_SUBSET_BYTES = 8_665
CANONICAL_SUBSET_SHA256 = (
    "bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04"
)
CONVERSION_VERSION = 1
SOURCE_ROW_INDEX = 873_702
SUBSET_SCHEMA_VERSION = 1
PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
SOURCE_ARCHIVE_RELATIVE_PATH = (
    "source_data_files/source_catalogs/lines/atomic_source_lines_parsed.npz"
)
SOURCE_ARCHIVE_PATH_TEXT = "/Users/ysting/payne-zero/" + SOURCE_ARCHIVE_RELATIVE_PATH
SOURCE_ARCHIVE_SHA256 = (
    "4eafa927c02a4f74401523149a44e35239f2aaecb4a64f2905a4cd5530c2dde7"
)
SOURCE_ARCHIVE_BYTES = 258_021_389
SOURCE_ARCHIVE_ROW_COUNT = 1_939_975
NON_AUTHORITATIVE_OBSERVED_ROW_INDEX = 780_108
MANIFEST_PATH = REPOSITORY_ROOT / "data/MANIFEST.json"
MANIFEST_SCHEMA_VERSION = 1
SUBSET_BUILDER_RELATIVE_PATH = "scripts/build_chapter06_fe_record_subset.py"
SUBSET_BUILDER_PATH = REPOSITORY_ROOT / SUBSET_BUILDER_RELATIVE_PATH
SUBSET_BUILDER_SHA256 = (
    "25bcf4662740155e8b08615b9522f3f4517e1a5ddc4627c68686620ccfff4d6c"
)

RATIO_LOG_STEP = np.log(1.0 + 1.0 / 2_000_000.0)
LIGHT_SPEED_NM_PER_SECOND = 2.99792458e17
CLASSICAL_LINE_STRENGTH_SCALE = 0.026538 / 1.77245 / LIGHT_SPEED_NM_PER_SECOND
DAMPING_SCALE = 1.0 / 12.5664 / LIGHT_SPEED_NM_PER_SECOND

RAW_FIELD_NAMES = (
    "stored_wavelength_nm",
    "raw_log_oscillator_strength",
    "species_code",
    "first_energy_column_cm",
    "second_energy_column_cm",
    "radiative_damping_log",
    "stark_damping_log",
    "van_der_waals_damping_log",
    "lower_principal_quantum_number",
    "upper_principal_quantum_number",
    "primary_isotope_number",
    "primary_isotope_log_correction",
    "secondary_isotope_log_correction",
    "energy_shift_field",
    "isotope_shift_units",
    "line_size",
    "line_category_tag",
)

RAW_FIELD_SPEC = {
    "stored_wavelength_nm": (np.dtype("<f8"), 499.0341),
    "raw_log_oscillator_strength": (np.dtype("<f8"), -0.86),
    "species_code": (np.dtype("<f8"), 26.0),
    "first_energy_column_cm": (np.dtype("<f8"), 53545.833),
    "second_energy_column_cm": (np.dtype("<f8"), 33507.123),
    "radiative_damping_log": (np.dtype("<f8"), 8.47),
    "stark_damping_log": (np.dtype("<f8"), -4.45),
    "van_der_waals_damping_log": (np.dtype("<f8"), -7.51),
    "lower_principal_quantum_number": (np.dtype("<i8"), 0),
    "upper_principal_quantum_number": (np.dtype("<i8"), 0),
    "primary_isotope_number": (np.dtype("<i8"), 0),
    "primary_isotope_log_correction": (np.dtype("<f8"), 0.0),
    "secondary_isotope_log_correction": (np.dtype("<f8"), 0.0),
    "energy_shift_field": (np.dtype("S10"), b"          "),
    "isotope_shift_units": (np.dtype("<f8"), 0.0),
    "line_size": (np.dtype("<i8"), 0),
    "line_category_tag": (np.dtype("S3"), b""),
}

PROVENANCE_SPEC = {
    "builder_command": (
        np.dtype("<U161"),
        "python scripts/build_chapter06_fe_record_subset.py --source-archive "
        "<optional-full-source>/source_data_files/source_catalogs/lines/"
        "atomic_source_lines_parsed.npz",
    ),
    "payne_zero_commit": (np.dtype("<U40"), PAYNE_ZERO_COMMIT),
    "source_archive_bytes": (np.dtype("<i8"), SOURCE_ARCHIVE_BYTES),
    "source_archive_relative_path": (
        np.dtype("<U70"),
        SOURCE_ARCHIVE_RELATIVE_PATH,
    ),
    "source_archive_row_count": (np.dtype("<i8"), SOURCE_ARCHIVE_ROW_COUNT),
    "source_archive_sha256": (np.dtype("<U64"), SOURCE_ARCHIVE_SHA256),
    "source_field_count": (np.dtype("<i8"), len(RAW_FIELD_NAMES)),
    "source_row_index": (np.dtype("<i8"), SOURCE_ROW_INDEX),
    "subset_role": (
        np.dtype("<U69"),
        "teaching subset: one immutable raw source record; no computed outputs",
    ),
    "subset_schema_version": (np.dtype("<i8"), SUBSET_SCHEMA_VERSION),
}

SELECTED_FIELD_DTYPES = {
    "packed_wavelength_index": np.dtype("<i4"),
    "packed_species_slot": np.dtype("<i2"),
    "lower_excitation_index": np.dtype("<i2"),
    "log_strength_index": np.dtype("<i2"),
    "radiative_damping_index": np.dtype("<i2"),
    "stark_damping_index": np.dtype("<i2"),
    "van_der_waals_damping_index": np.dtype("<i2"),
}


@dataclass(frozen=True)
class DecodedPhysicalLedger:
    """Kernel-facing physical values decoded from one packed line record."""

    reconstructed_wavelength_nm: float
    unquantized_wavelength_nm: float
    reconstructed_minus_unquantized_nm: float
    population_slot_one_based: int
    lower_excitation_cm: float
    oscillator_strength: float
    raw_radiative_damping_s_inverse: float
    raw_stark_damping_cm3_s_inverse: float
    raw_van_der_waals_damping_cm3_s_inverse: float
    classical_strength_cm2: float
    radiative_damping: float
    stark_damping_cm3: float
    van_der_waals_damping_cm3: float


def _array_sha256(values: np.ndarray) -> str:
    """Return the canonical C-byte identity of one array."""

    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the byte identity of one file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def packed_wavelength_code(wavelength_nm: float) -> np.int32:
    """Quantize one finite positive wavelength into the atmosphere code."""

    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError("wavelength must be finite and positive")
    code = int(np.floor(np.log(wavelength_nm) / RATIO_LOG_STEP + 0.5))
    limits = np.iinfo(np.int32)
    if code < limits.min or code > limits.max:
        raise ValueError("packed wavelength does not fit int32")
    return np.int32(code)


def tablog_code(positive_value: float) -> np.int16:
    """Quantize one finite positive value into a nonzero TABLOG index."""

    if not np.isfinite(positive_value) or positive_value <= 0.0:
        raise ValueError("TABLOG input must be finite and positive")
    code = int(np.floor(np.log10(positive_value) * 1000.0 + 16384.5))
    if code < 1 or code > np.iinfo(np.int16).max:
        raise ValueError("positive TABLOG index does not fit int16")
    return np.int16(code)


def validate_canonical_raw_row(
    raw_row: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate and defensively copy all 17 fields of canonical raw row 873702."""

    actual_names = set(raw_row)
    expected_names = set(RAW_FIELD_NAMES)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"canonical raw row fields changed; missing={missing!r}, extra={extra!r}"
        )

    validated: dict[str, np.ndarray] = {}
    for name in RAW_FIELD_NAMES:
        values = np.asarray(raw_row[name])
        expected_dtype, expected_value = RAW_FIELD_SPEC[name]
        if values.shape != (1,):
            raise ValueError(
                f"canonical raw field {name!r} has shape {values.shape}; expected (1,)"
            )
        if values.dtype != expected_dtype:
            raise ValueError(
                f"canonical raw field {name!r} has dtype {values.dtype}; "
                f"expected {expected_dtype}"
            )
        if values[0].item() != expected_value:
            raise ValueError(
                f"canonical raw field {name!r} has value "
                f"{values[0].item()!r}; expected {expected_value!r}"
            )
        validated[name] = np.array(values, copy=True)

    if validated["energy_shift_field"].tobytes(order="C") != b" " * 10:
        raise ValueError(
            "canonical energy_shift_field must retain ten literal space bytes"
        )
    if validated["line_category_tag"].tobytes(order="C") != b"\x00" * 3:
        raise ValueError(
            "canonical line_category_tag must retain three literal zero bytes"
        )
    return validated


def _load_verified_subset_archive(path: Path) -> dict[str, np.ndarray]:
    """Validate one archive's exact bytes and return all 27 copied members."""

    subset_path = Path(path)
    if not subset_path.is_file():
        raise FileNotFoundError(f"canonical Chapter 6 subset is missing: {subset_path}")
    if subset_path.stat().st_size != CANONICAL_SUBSET_BYTES:
        raise ValueError(
            f"canonical Chapter 6 subset has {subset_path.stat().st_size} bytes; "
            f"expected {CANONICAL_SUBSET_BYTES}"
        )
    actual_hash = _file_sha256(subset_path)
    if actual_hash != CANONICAL_SUBSET_SHA256:
        raise ValueError(
            f"canonical Chapter 6 subset has SHA-256 {actual_hash}; "
            f"expected {CANONICAL_SUBSET_SHA256}"
        )

    expected_members = set(RAW_FIELD_NAMES) | set(PROVENANCE_SPEC)
    with np.load(subset_path, allow_pickle=False) as archive:
        if tuple(archive.files) != tuple(sorted(expected_members)):
            raise ValueError(
                "canonical Chapter 6 subset member schema changed: "
                f"{tuple(archive.files)!r}"
            )
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}

    validate_canonical_raw_row({name: arrays[name] for name in RAW_FIELD_NAMES})
    for name, (expected_dtype, expected_value) in PROVENANCE_SPEC.items():
        values = arrays[name]
        if values.shape != ():
            raise ValueError(
                f"canonical provenance member {name!r} has shape "
                f"{values.shape}; expected scalar"
            )
        if values.dtype != expected_dtype:
            raise ValueError(
                f"canonical provenance member {name!r} has dtype "
                f"{values.dtype}; expected {expected_dtype}"
            )
        if values.item() != expected_value:
            raise ValueError(
                f"canonical provenance member {name!r} changed from "
                f"{expected_value!r} to {values.item()!r}"
            )
    return arrays


def _load_manifest_without_duplicate_keys(path: Path) -> dict[str, object]:
    """Read the canonical manifest while rejecting duplicate JSON keys."""

    manifest_path = Path(path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(
            f"canonical data manifest must be a regular nonsymlink file: {manifest_path}"
        )

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in data manifest: {key!r}")
            result[key] = value
        return result

    document = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(document, dict):
        raise ValueError("canonical data manifest root must be a JSON object")
    return document


def _validate_manifest_authority(
    manifest: Mapping[str, object],
    archive_members: Mapping[str, np.ndarray],
    *,
    builder_path: Path = SUBSET_BUILDER_PATH,
) -> None:
    """Bind the canonical subset to its unique manifest and builder record."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("canonical data manifest schema version changed")
    if manifest.get("payne_zero_commit") != PAYNE_ZERO_COMMIT:
        raise ValueError("canonical data manifest Payne Zero commit changed")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("canonical data manifest entries must be a list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("path") == CANONICAL_SUBSET_RELATIVE_PATH
    ]
    if len(matches) != 1:
        raise ValueError(
            "canonical subset must have exactly one manifest entry; "
            f"found {len(matches)}"
        )
    entry = matches[0]
    expected_entry_scalars = {
        "path": CANONICAL_SUBSET_RELATIVE_PATH,
        "role": "subset",
        "format": "npz",
        "sha256": CANONICAL_SUBSET_SHA256,
        "bytes": CANONICAL_SUBSET_BYTES,
        "source_commit": PAYNE_ZERO_COMMIT,
        "source": SOURCE_ARCHIVE_PATH_TEXT,
        "source_sha256": SOURCE_ARCHIVE_SHA256,
        "source_row_index": SOURCE_ROW_INDEX,
        "source_archive_bytes": SOURCE_ARCHIVE_BYTES,
        "source_archive_row_count": SOURCE_ARCHIVE_ROW_COUNT,
        "source_field_count": len(RAW_FIELD_NAMES),
        "subset_schema_version": SUBSET_SCHEMA_VERSION,
        "builder": SUBSET_BUILDER_RELATIVE_PATH,
        "builder_sha256": SUBSET_BUILDER_SHA256,
        "requires_optional_full_catalog": True,
    }
    for name, expected_value in expected_entry_scalars.items():
        if entry.get(name) != expected_value:
            raise ValueError(
                f"canonical subset manifest field {name!r} changed: {entry.get(name)!r}"
            )

    actual_builder = Path(builder_path)
    if actual_builder.is_symlink() or not actual_builder.is_file():
        raise ValueError(
            f"canonical subset builder must be a regular nonsymlink file: "
            f"{actual_builder}"
        )
    actual_builder_hash = _file_sha256(actual_builder)
    if actual_builder_hash != SUBSET_BUILDER_SHA256:
        raise ValueError(
            f"canonical subset builder SHA-256 changed: {actual_builder_hash}"
        )

    member_metadata = entry.get("arrays")
    if not isinstance(member_metadata, dict):
        raise ValueError("canonical subset manifest arrays must be an object")
    if set(member_metadata) != set(archive_members):
        raise ValueError(
            "canonical subset manifest member set changed; "
            f"missing={sorted(set(archive_members) - set(member_metadata))}, "
            f"extra={sorted(set(member_metadata) - set(archive_members))}"
        )
    for name, values in archive_members.items():
        metadata = member_metadata[name]
        if not isinstance(metadata, dict):
            raise ValueError(
                f"canonical subset manifest metadata for {name!r} is not an object"
            )
        expected_shape = list(np.asarray(values).shape)
        if metadata.get("shape") != expected_shape:
            raise ValueError(f"canonical subset manifest shape changed for {name!r}")
        if "dtype" not in metadata:
            raise ValueError(f"canonical subset manifest dtype is missing for {name!r}")
        try:
            declared_dtype = np.dtype(metadata.get("dtype"))
        except TypeError as error:
            raise ValueError(
                f"canonical subset manifest dtype is invalid for {name!r}"
            ) from error
        if declared_dtype != np.asarray(values).dtype:
            raise ValueError(f"canonical subset manifest dtype changed for {name!r}")
        if metadata.get("sha256") != _array_sha256(np.asarray(values)):
            raise ValueError(
                f"canonical subset manifest member SHA-256 changed for {name!r}"
            )


def load_verified_canonical_raw_row() -> dict[str, np.ndarray]:
    """Load only the unique manifest-bound canonical raw teaching row."""

    if CANONICAL_SUBSET_PATH.is_symlink():
        raise ValueError("canonical Chapter 6 subset path must not be a symlink")
    arrays = _load_verified_subset_archive(CANONICAL_SUBSET_PATH)
    manifest = _load_manifest_without_duplicate_keys(MANIFEST_PATH)
    _validate_manifest_authority(manifest, arrays)
    return validate_canonical_raw_row({name: arrays[name] for name in RAW_FIELD_NAMES})


def convert_raw_row_to_selected_line_catalog(
    raw_row: Mapping[str, np.ndarray],
) -> SelectedLineCatalog:
    """Convert the exact accepted raw Fe I row into one atmosphere record."""

    raw = validate_canonical_raw_row(raw_row)
    first_energy_cm = abs(float(raw["first_energy_column_cm"][0]))
    second_energy_cm = abs(float(raw["second_energy_column_cm"][0]))
    energy_separation_cm = abs(first_energy_cm - second_energy_cm)
    if energy_separation_cm <= 0.0:
        raise ValueError("the controlled line must have nonzero energy separation")

    wavelength_nm = 1.0e7 / energy_separation_cm
    lower_excitation_cm = min(first_energy_cm, second_energy_cm)
    log_strength = (
        float(raw["raw_log_oscillator_strength"][0])
        + float(raw["primary_isotope_log_correction"][0])
        + float(raw["secondary_isotope_log_correction"][0])
    )
    oscillator_strength = 10.0**log_strength
    radiative_damping = 10.0 ** float(raw["radiative_damping_log"][0])
    stark_damping = 10.0 ** float(raw["stark_damping_log"][0])
    van_der_waals_damping = 10.0 ** float(raw["van_der_waals_damping_log"][0])

    species_code = float(raw["species_code"][0])
    atomic_number = int(species_code + 1.0e-6)
    ion_fraction = species_code - atomic_number
    ion_stage = int(np.rint(ion_fraction * 100.0)) + 1 if ion_fraction > 1.0e-6 else 1
    population_slot_one_based = atomic_population_slot_start(atomic_number) + ion_stage
    packed_species_slot = population_slot_one_based * 10

    return SelectedLineCatalog(
        packed_wavelength_index=np.asarray(
            [packed_wavelength_code(wavelength_nm)], dtype=np.int32
        ),
        packed_species_slot=np.asarray([packed_species_slot], dtype=np.int16),
        lower_excitation_index=np.asarray(
            [tablog_code(lower_excitation_cm)], dtype=np.int16
        ),
        log_strength_index=np.asarray(
            [tablog_code(oscillator_strength)], dtype=np.int16
        ),
        radiative_damping_index=np.asarray(
            [tablog_code(radiative_damping)], dtype=np.int16
        ),
        stark_damping_index=np.asarray([tablog_code(stark_damping)], dtype=np.int16),
        van_der_waals_damping_index=np.asarray(
            [tablog_code(van_der_waals_damping)], dtype=np.int16
        ),
    )


def selected_line_member_hashes(
    selected_lines: SelectedLineCatalog,
) -> dict[str, str]:
    """Return C-byte hashes for the seven selected-record members."""

    _validate_selected_catalog_schema(selected_lines)
    return {
        name: _array_sha256(getattr(selected_lines, name))
        for name in SELECTED_FIELD_DTYPES
    }


def _validate_selected_catalog_schema(
    selected_lines: SelectedLineCatalog,
) -> None:
    """Require one exact-width selected-line record before packing or decoding."""

    for name, expected_dtype in SELECTED_FIELD_DTYPES.items():
        values = np.asarray(getattr(selected_lines, name))
        if values.shape != (1,):
            raise ValueError(
                f"selected-line member {name!r} has shape {values.shape}; expected (1,)"
            )
        if values.dtype != expected_dtype:
            raise ValueError(
                f"selected-line member {name!r} has dtype {values.dtype}; "
                f"expected {expected_dtype}"
            )


def pack_selected_line_words(
    selected_lines: SelectedLineCatalog,
) -> np.ndarray:
    """Pack one selected record into the native `(1, 4)` int32 payload."""

    _validate_selected_catalog_schema(selected_lines)
    halfwords = np.column_stack(
        [
            selected_lines.packed_species_slot,
            selected_lines.lower_excitation_index,
            selected_lines.log_strength_index,
            selected_lines.radiative_damping_index,
            selected_lines.stark_damping_index,
            selected_lines.van_der_waals_damping_index,
        ]
    )
    halfwords = np.ascontiguousarray(halfwords, dtype=np.int16)
    packed_tail = halfwords.view(np.int32).reshape(1, 3)
    return np.ascontiguousarray(
        np.column_stack([selected_lines.packed_wavelength_index, packed_tail]),
        dtype=np.int32,
    )


def decoded_physical_ledger(
    raw_row: Mapping[str, np.ndarray],
    selected_lines: SelectedLineCatalog,
) -> DecodedPhysicalLedger:
    """Decode the packed record in the float32 order used by the line kernel."""

    raw = validate_canonical_raw_row(raw_row)
    _validate_selected_catalog_schema(selected_lines)
    lookup = np.ascontiguousarray(
        10.0 ** ((np.arange(1, 32769, dtype=np.float64) - 16384.0) * 0.001),
        dtype=np.float32,
    )

    wavelength_nm = float(
        np.exp(float(selected_lines.packed_wavelength_index[0]) * RATIO_LOG_STEP)
    )
    lower_excitation = lookup[int(selected_lines.lower_excitation_index[0]) - 1]
    oscillator_strength = lookup[int(selected_lines.log_strength_index[0]) - 1]
    raw_radiative = lookup[int(selected_lines.radiative_damping_index[0]) - 1]
    raw_stark = lookup[int(selected_lines.stark_damping_index[0]) - 1]
    raw_van_der_waals = lookup[int(selected_lines.van_der_waals_damping_index[0]) - 1]

    wavelength_f32 = np.float32(wavelength_nm)
    classical_strength = np.float32(
        np.float32(CLASSICAL_LINE_STRENGTH_SCALE) * wavelength_f32 * oscillator_strength
    )
    damping_scale_f32 = np.float32(DAMPING_SCALE)
    radiative_damping = np.float32(raw_radiative * wavelength_f32 * damping_scale_f32)
    stark_damping = np.float32(raw_stark * wavelength_f32 * damping_scale_f32)
    van_der_waals_damping = np.float32(
        raw_van_der_waals * wavelength_f32 * damping_scale_f32
    )

    first_energy_cm = abs(float(raw["first_energy_column_cm"][0]))
    second_energy_cm = abs(float(raw["second_energy_column_cm"][0]))
    unquantized_wavelength_nm = 1.0e7 / abs(first_energy_cm - second_energy_cm)
    population_slot = abs(int(selected_lines.packed_species_slot[0])) // 10
    return DecodedPhysicalLedger(
        reconstructed_wavelength_nm=wavelength_nm,
        unquantized_wavelength_nm=unquantized_wavelength_nm,
        reconstructed_minus_unquantized_nm=(wavelength_nm - unquantized_wavelength_nm),
        population_slot_one_based=population_slot,
        lower_excitation_cm=float(lower_excitation),
        oscillator_strength=float(oscillator_strength),
        raw_radiative_damping_s_inverse=float(raw_radiative),
        raw_stark_damping_cm3_s_inverse=float(raw_stark),
        raw_van_der_waals_damping_cm3_s_inverse=float(raw_van_der_waals),
        classical_strength_cm2=float(classical_strength),
        radiative_damping=float(radiative_damping),
        stark_damping_cm3=float(stark_damping),
        van_der_waals_damping_cm3=float(van_der_waals_damping),
    )


__all__ = [
    "CANONICAL_SUBSET_BYTES",
    "CANONICAL_SUBSET_PATH",
    "CANONICAL_SUBSET_SHA256",
    "CONVERSION_VERSION",
    "DecodedPhysicalLedger",
    "NON_AUTHORITATIVE_OBSERVED_ROW_INDEX",
    "PAYNE_ZERO_COMMIT",
    "RAW_FIELD_NAMES",
    "RAW_FIELD_SPEC",
    "SELECTED_FIELD_DTYPES",
    "SOURCE_ROW_INDEX",
    "SUBSET_SCHEMA_VERSION",
    "convert_raw_row_to_selected_line_catalog",
    "decoded_physical_ledger",
    "load_verified_canonical_raw_row",
    "pack_selected_line_words",
    "packed_wavelength_code",
    "selected_line_member_hashes",
    "tablog_code",
    "validate_canonical_raw_row",
]
