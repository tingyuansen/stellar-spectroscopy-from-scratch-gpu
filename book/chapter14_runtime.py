"""Executable Chapter 14 checkpoints for learned atmosphere initializers.

The production warm-start and direct-abundance modules are staged byte for
byte in ``src``.  This module verifies their assets, separates Torch inference
from NumPy decoding, and exercises every initializer safety boundary that does
not depend on the still-incomplete shared atmosphere runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np

from book.chapter14_teaching import (
    DirectLayoutTrace,
    PcaDecodeTrace,
    decode_pca_coordinates,
    decode_profile_coordinates,
    direct_layout_trace,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
STATIC_ROOT = REPOSITORY_ROOT / "data/static"
EMULATOR_ROOT = STATIC_ROOT / "atmosphere_emulator"
PINNED_PAYNE_ZERO_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"
FIXTURE_PATH = REPOSITORY_ROOT / "data/fixtures/chapter14_initializer_inputs.npz"
GOLDEN_PATH = (
    REPOSITORY_ROOT
    / "data/golden/payne_zero/chapter14/chapter14_initializer_outputs.npz"
)
ARTIFACT_MANIFEST = REPOSITORY_ROOT / "data/chapter14_artifacts.json"

PINNED_SOURCE_SHA256 = {
    "src/payne_zero_atmosphere/warm_start.py": (
        "3a83af3d68be52a35bfc3f55f5912770661be8251cc28f28da3250b2e83e0ad3"
    ),
    "src/payne_zero_atmosphere/direct_abundance.py": (
        "ec65683eb344c4c3fd77340c084e780f58c6401e77c9f0d6db05ef6753131445"
    ),
}
PINNED_ASSET_SHA256 = {
    "data/static/atmosphere_emulator/release_manifest.json": (
        "fe8093a6c0260f2524efd35a974790797e2cc8f921e9dd109477f03d845a789c"
    ),
    "data/static/atmosphere_emulator/five_label/checkpoint.pt": (
        "c32717016d4f9047ab37bc17b6900faf9c514de3407292a07a745c35a50784e5"
    ),
    "data/static/atmosphere_emulator/cno8/checkpoint.pt": (
        "be97c0b729490b9e18f092c59f836adb421e2e99359b44436edc3ae5632f8a02"
    ),
    "data/static/atmosphere_emulator/direct_abundance/manifest.json": (
        "fb59da5e6bd3f8fcba06e0c4c284137e90aab5c4e93165daa74d8ce2ae268710"
    ),
    "data/static/atmosphere_emulator/direct_abundance/checkpoint.pt": (
        "1b8e1db1514956dfbf890eb5ae96e01bd918acfc86be538b6b77230332104243"
    ),
}
RUNNER_REQUIRED_SYMBOLS = (
    "TransferAccumulation",
    "IterationFinalization",
    "IterationRemap",
    "AtmosphereRunResult",
    "finalize_transfer_state",
    "remap_finalized_iteration_state",
    "finalize_remapped_iteration",
    "run_atmosphere_model",
)

FIVE_LABEL_EXAMPLE = {
    "effective_temperature": 5777.0,
    "log_surface_gravity": 4.44,
    "metallicity": -0.15,
    "alpha_enhancement": 0.10,
    "microturbulence_km_s": 1.8,
}
CNO8_EXAMPLE = {
    "effective_temperature": 4750.0,
    "log_surface_gravity": 2.25,
    "metallicity": -0.70,
    "alpha_enhancement": 0.25,
    "microturbulence_km_s": 2.1,
    "carbon_enhancement": 0.20,
    "nitrogen_enhancement": -0.10,
    "oxygen_enhancement": 0.35,
}


@dataclass(frozen=True)
class AssetIdentityCheckpoint:
    """Local source/checkpoint identity and release-manifest semantics."""

    source_sha256: dict[str, str]
    asset_sha256: dict[str, str]
    release: str
    five_label_features: tuple[str, ...]
    cno8_features: tuple[str, ...]
    exact_solver_is_final_authority: bool
    training_corpora_packaged: bool


@dataclass(frozen=True)
class TrainingDataCheckpoint:
    """Frozen split semantics needed to reproduce, not run, model fitting."""

    five_label_total: int
    five_label_train: int
    five_label_fit_validation: int
    five_label_internal_check: int
    cno8_total: int
    cno8_frozen_train: int
    cno8_fit_validation: int
    cno8_internal_check: int
    cno8_appended_training: int
    direct_total: int
    direct_train: int
    direct_fit_validation: int
    direct_internal_check: int
    direct_unused_external_gate: int
    direct_optimizer: str
    direct_learning_rate: float
    direct_weight_decay: float
    direct_seed: int


@dataclass(frozen=True)
class DecoderCheckpoint:
    """One exact MLP prediction and an independently staged PCA decode."""

    family: str
    labels: dict[str, float]
    checkpoint_feature_fields: tuple[str, ...]
    model_config: dict[str, object]
    feature_vector: np.ndarray
    standardized_feature_vector: np.ndarray
    standardized_coefficients: np.ndarray
    pca_trace: PcaDecodeTrace
    prediction: dict[str, np.ndarray]
    independently_decoded: dict[str, np.ndarray]
    maximum_absolute_decode_difference: float
    output_shapes: dict[str, tuple[int, ...]]
    output_dtypes: dict[str, str]
    acceleration_scale: float
    derivative_loss_weight: float
    optical_depth_loss_weight: float
    hydrostatic_loss_weight: float


@dataclass(frozen=True)
class CandidateCheckpoint:
    """Exact family routing, projection, and deterministic retry ordering."""

    five_label_family: str
    cno_relative_family: str
    cno_absolute_family: str
    in_support_candidates: tuple[dict[str, float] | None, ...]
    projected_candidates: tuple[dict[str, float] | None, ...]
    repeated_candidates: tuple[dict[str, float] | None, ...]
    projected_request_temperature: float
    first_projected_temperature: float
    deterministic: bool


@dataclass(frozen=True)
class WarmStartCheckpoint:
    """Raw decoded prediction and its exact fixed-column solver seed."""

    family: str
    atmosphere: object
    deck_text: str
    deck_sha256: str
    raw_prediction: dict[str, np.ndarray]
    quantized_prediction: dict[str, np.ndarray]
    maximum_relative_quantization_difference: dict[str, float]
    requested_effective_temperature: float
    parsed_effective_temperature: float
    requested_log_surface_gravity: float
    parsed_log_surface_gravity: float
    has_converged_field: bool


@dataclass(frozen=True)
class DirectMixtureCheckpoint:
    """Exact direct-abundance layouts, lattice, support, and hash."""

    public_xh: dict[int, float]
    exact_mixture: np.ndarray
    feature_vector: np.ndarray
    layout: DirectLayoutTrace
    mixture_sha256: str
    public_atomic_numbers: tuple[int, ...]
    sentinel_atomic_numbers: tuple[int, ...]
    public_values_quantized: bool


@dataclass(frozen=True)
class DirectSafetyCheckpoint:
    """Observed experimental gates and immutable surrogate provenance."""

    opt_in_rejected_by_default: bool
    incomplete_public_vector_rejected: bool
    unsupported_state_rejected: bool
    release_gate_passed: bool
    checkpoint_sha256: str
    manifest_sha256: str
    role: str
    exact_closure_required: bool
    is_final_atmosphere: bool
    realized_mixture_sha256: str
    deck_sha256: str
    surrogate_identity_sha256: str
    realized_mixture_writeable: bool
    public_deck_type: str
    public_deck_exposes_model: bool
    exact_trial_count: int


@dataclass(frozen=True)
class DirectSetEncoderCheckpoint:
    """Internal tensor shapes and an exact manual set-encoder reconstruction."""

    state_shape: tuple[int, ...]
    relative_abundance_shape: tuple[int, ...]
    element_embedding_shape: tuple[int, ...]
    token_input_shape: tuple[int, ...]
    response_law_shape: tuple[int, ...]
    summed_response_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    maximum_manual_output_difference: float
    paired_permutation_output_difference: float
    abundance_only_permutation_output_difference: float


@dataclass(frozen=True)
class ClosureSeamCheckpoint:
    """Honest status of the exact restart path inherited from Chapter 13."""

    runner_symbols_available: tuple[str, ...]
    runner_symbols_missing: tuple[str, ...]
    initializer_reader_core_executable: bool
    exact_restart_trajectory_executable: bool
    status: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_chapter14_runtime(*, require_built_artifacts: bool = False) -> Path:
    """Select staged source/data and fail closed on every immutable identity."""

    staged_root = SOURCE_ROOT.resolve()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("payne_zero_atmosphere"):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is not None and not Path(module_path).resolve().is_relative_to(
            staged_root
        ):
            raise RuntimeError(
                f"{name} resolved outside the staged source tree: {module_path}"
            )
    staged = str(staged_root)
    if staged in sys.path:
        sys.path.remove(staged)
    sys.path.insert(0, staged)
    os.environ["PAYNE_ZERO_DATA_ROOT"] = str(STATIC_ROOT)
    os.environ["PAYNE_ZERO_ATMOSPHERE_DATA_ROOT"] = str(
        STATIC_ROOT / "atmosphere_tables"
    )
    os.environ["PAYNE_ZERO_SYNTHESIS_DATA_ROOT"] = str(
        STATIC_ROOT / "synthesis_tables"
    )
    for relative, expected in {
        **PINNED_SOURCE_SHA256,
        **PINNED_ASSET_SHA256,
    }.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Chapter 14 immutable artifact changed: {relative}")
    if require_built_artifacts:
        manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("payne_zero_commit") != PINNED_PAYNE_ZERO_COMMIT:
            raise RuntimeError("Chapter 14 manifest names the wrong source commit")
        for path in (FIXTURE_PATH, GOLDEN_PATH):
            relative = str(path.relative_to(REPOSITORY_ROOT))
            if _sha256(path) != manifest["artifacts"][relative]["sha256"]:
                raise RuntimeError(f"Chapter 14 artifact identity changed: {relative}")
    return SOURCE_ROOT


def asset_identity_checkpoint() -> AssetIdentityCheckpoint:
    """Verify local files and read feature semantics from the release manifest."""

    configure_chapter14_runtime()
    release_manifest = json.loads(
        (EMULATOR_ROOT / "release_manifest.json").read_text(encoding="utf-8")
    )
    families = release_manifest["families"]
    return AssetIdentityCheckpoint(
        source_sha256={
            path: _sha256(REPOSITORY_ROOT / path) for path in PINNED_SOURCE_SHA256
        },
        asset_sha256={
            path: _sha256(REPOSITORY_ROOT / path) for path in PINNED_ASSET_SHA256
        },
        release=str(release_manifest["release"]),
        five_label_features=tuple(
            item["name"] for item in families["five_label"]["checkpoint_features"]
        ),
        cno8_features=tuple(
            item["name"] for item in families["cno8"]["checkpoint_features"]
        ),
        exact_solver_is_final_authority=bool(
            release_manifest["runtime_contract"]["exact_solver_is_final_authority"]
        ),
        training_corpora_packaged=any(
            path.name.startswith("strict_truth")
            or "training_corpora" in path.parts
            for path in EMULATOR_ROOT.rglob("*")
            if path.is_file()
        ),
    )


def training_data_checkpoint() -> TrainingDataCheckpoint:
    """Read fitting/checkpoint-selection roles from the two asset manifests."""

    configure_chapter14_runtime()
    import torch

    release = json.loads(
        (EMULATOR_ROOT / "release_manifest.json").read_text(encoding="utf-8")
    )
    direct_manifest = json.loads(
        (EMULATOR_ROOT / "direct_abundance/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    direct_checkpoint = torch.load(
        EMULATOR_ROOT / "direct_abundance/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    five = release["families"]["five_label"]["strict_truth"]
    cno = release["families"]["cno8"]["strict_truth"]
    direct = direct_manifest["training"]
    optimization = direct_checkpoint["training"]["optimization"]
    return TrainingDataCheckpoint(
        five_label_total=int(five["records"]),
        five_label_train=int(five["fit"]),
        five_label_fit_validation=int(five["fit_validation"]),
        five_label_internal_check=int(five["internal_check"]),
        cno8_total=int(cno["records"]),
        cno8_frozen_train=int(cno["frozen_fit"]),
        cno8_fit_validation=int(cno["frozen_fit_validation"]),
        cno8_internal_check=int(cno["frozen_internal_check"]),
        cno8_appended_training=int(cno["appended_parent_coverage_fit"]),
        direct_total=int(direct["total_records"]),
        direct_train=int(direct["train_records"]),
        direct_fit_validation=int(direct["fit_validation_records"]),
        direct_internal_check=int(direct["internal_check_records_unused_for_fit"]),
        direct_unused_external_gate=int(direct["unused_external_gate_records"]),
        direct_optimizer=str(optimization["optimizer"]),
        direct_learning_rate=float(optimization["learning_rate"]),
        direct_weight_decay=float(optimization["weight_decay"]),
        direct_seed=int(direct_checkpoint["training"]["seed"]),
    )


def _ordinary_checkpoint_path(family: str) -> Path:
    if family not in {"five_label", "cno8"}:
        raise ValueError("family must be five_label or cno8")
    return EMULATOR_ROOT / family / "checkpoint.pt"


def _ordinary_labels(family: str) -> dict[str, float]:
    return dict(FIVE_LABEL_EXAMPLE if family == "five_label" else CNO8_EXAMPLE)


def decoder_checkpoint(family: str = "five_label") -> DecoderCheckpoint:
    """Expose each side of exact Torch float32 to NumPy float64 decoding."""

    configure_chapter14_runtime()
    import torch

    from payne_zero_atmosphere.warm_start import (
        INITIALIZER_OUTPUT_FIELDS,
        _initializer_checkpoint_features,
        load_atmosphere_initializer,
    )

    labels = _ordinary_labels(family)
    initializer = load_atmosphere_initializer(
        checkpoint_path=_ordinary_checkpoint_path(family),
        device="cpu",
    )
    checkpoint = initializer.checkpoint
    fields = tuple(initializer.checkpoint_feature_fields)
    features = _initializer_checkpoint_features(
        **labels,
        checkpoint_feature_fields=fields,
    )
    label_mean = np.asarray(checkpoint["labels"]["mean"], dtype=np.float64)
    label_std = np.asarray(checkpoint["labels"]["std"], dtype=np.float64)
    standardized_features = (features - label_mean) / label_std
    with torch.no_grad():
        standardized_coefficients = (
            initializer.model(
                torch.as_tensor(
                    standardized_features[None, :],
                    dtype=torch.float32,
                    device="cpu",
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
    pca = checkpoint["pca"]
    trace = decode_pca_coordinates(
        standardized_coefficients,
        coefficient_mean=pca["coefficient_mean"],
        coefficient_std=pca["coefficient_std"],
        basis=pca["basis"],
        coordinate_mean=pca["coordinate_mean"],
        coordinate_std=pca["coordinate_std"],
    )
    independently_decoded_array = decode_profile_coordinates(
        trace.coordinates,
        effective_temperature=labels["effective_temperature"],
        rosseland_optical_depth=initializer.standard_rosseland_optical_depth,
        acceleration_scale=float(checkpoint["coordinates"]["acceleration_scale"]),
    )
    independent = {
        field: independently_decoded_array[:, index]
        for index, field in enumerate(INITIALIZER_OUTPUT_FIELDS)
    }
    prediction = initializer.predict(**labels)
    largest = max(
        float(np.max(np.abs(prediction[field] - independent[field])))
        for field in INITIALIZER_OUTPUT_FIELDS
    )
    loss = checkpoint["loss"]
    return DecoderCheckpoint(
        family=family,
        labels=labels,
        checkpoint_feature_fields=fields,
        model_config=dict(checkpoint["model"]["config"]),
        feature_vector=features,
        standardized_feature_vector=standardized_features,
        standardized_coefficients=standardized_coefficients[0],
        pca_trace=trace,
        prediction={name: values.copy() for name, values in prediction.items()},
        independently_decoded={
            name: values.copy() for name, values in independent.items()
        },
        maximum_absolute_decode_difference=largest,
        output_shapes={name: values.shape for name, values in prediction.items()},
        output_dtypes={name: str(values.dtype) for name, values in prediction.items()},
        acceleration_scale=float(checkpoint["coordinates"]["acceleration_scale"]),
        derivative_loss_weight=float(loss["derivative_weight"]),
        optical_depth_loss_weight=float(loss["optical_depth_weight"]),
        hydrostatic_loss_weight=float(loss["hydrostatic_weight"]),
    )


def candidate_checkpoint() -> CandidateCheckpoint:
    """Exercise exact family routing, support projection, and hashed jitters."""

    configure_chapter14_runtime()
    from payne_zero_atmosphere.warm_start import (
        deterministic_initializer_labels,
        select_warm_start_family,
    )

    five_family = select_warm_start_family()
    cno_relative = select_warm_start_family(carbon_enhancement=0.0)
    cno_absolute = select_warm_start_family(
        absolute_abundance_offsets={8: -0.2}
    )
    in_support = deterministic_initializer_labels(
        **FIVE_LABEL_EXAMPLE,
        max_trials=3,
        seed=20260713,
        jitter_scale=0.01,
        checkpoint_path=_ordinary_checkpoint_path("five_label"),
    )
    projected_temperature = 12000.0
    projected_kwargs = {
        **FIVE_LABEL_EXAMPLE,
        "effective_temperature": projected_temperature,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        projected = deterministic_initializer_labels(
            **projected_kwargs,
            max_trials=3,
            seed=20260713,
            jitter_scale=0.01,
            checkpoint_path=_ordinary_checkpoint_path("five_label"),
        )
        repeated = deterministic_initializer_labels(
            **projected_kwargs,
            max_trials=3,
            seed=20260713,
            jitter_scale=0.01,
            checkpoint_path=_ordinary_checkpoint_path("five_label"),
        )
    return CandidateCheckpoint(
        five_label_family=five_family,
        cno_relative_family=cno_relative,
        cno_absolute_family=cno_absolute,
        in_support_candidates=in_support,
        projected_candidates=projected,
        repeated_candidates=repeated,
        projected_request_temperature=projected_temperature,
        first_projected_temperature=float(projected[0]["effective_temperature"]),
        deterministic=projected == repeated,
    )


def _atmosphere_prediction(atmosphere: object) -> dict[str, np.ndarray]:
    fields = (
        "column_mass",
        "temperature",
        "gas_pressure",
        "electron_density",
        "rosseland_opacity",
        "radiative_acceleration",
    )
    return {
        field: np.asarray(getattr(atmosphere, field), np.float64).copy()
        for field in fields
    }


def warm_start_checkpoint(family: str = "five_label") -> WarmStartCheckpoint:
    """Decode, fixed-column-format, and parse one exact ordinary/CNO seed."""

    configure_chapter14_runtime()
    from payne_zero_atmosphere.warm_start import (
        emulator_warm_start_model,
        load_atmosphere_initializer,
    )

    labels = _ordinary_labels(family)
    initializer = load_atmosphere_initializer(
        checkpoint_path=_ordinary_checkpoint_path(family)
    )
    raw = initializer.predict(**labels)
    paths = (
        {"five_label_path": _ordinary_checkpoint_path(family)}
        if family == "five_label"
        else {"cno8_path": _ordinary_checkpoint_path(family)}
    )
    atmosphere, deck = emulator_warm_start_model(**labels, **paths)
    quantized = _atmosphere_prediction(atmosphere)
    differences = {}
    for field, raw_values in raw.items():
        denominator = np.maximum(np.abs(raw_values), np.finfo(np.float64).tiny)
        differences[field] = float(
            np.max(np.abs(quantized[field] - raw_values) / denominator)
        )
    return WarmStartCheckpoint(
        family=family,
        atmosphere=atmosphere,
        deck_text=deck,
        deck_sha256=hashlib.sha256(deck.encode("utf-8")).hexdigest(),
        raw_prediction={name: values.copy() for name, values in raw.items()},
        quantized_prediction=quantized,
        maximum_relative_quantization_difference=differences,
        requested_effective_temperature=labels["effective_temperature"],
        parsed_effective_temperature=float(
            atmosphere.metadata["effective_temperature"]
        ),
        requested_log_surface_gravity=labels["log_surface_gravity"],
        parsed_log_surface_gravity=float(
            atmosphere.metadata["log_surface_gravity"]
        ),
        has_converged_field=hasattr(atmosphere, "converged"),
    )


def _direct_public_input() -> tuple[dict[int, float], np.ndarray]:
    from payne_zero_atmosphere.direct_abundance import (
        retained_direct_abundance_mixture,
    )

    return retained_direct_abundance_mixture(
        iron_abundance_relative_to_hydrogen=-0.20,
        retained_abundance_relative_to_iron_by_atomic_number={
            6: 0.10,
            8: 0.20,
            12: 0.15,
        },
    )


def direct_mixture_checkpoint() -> DirectMixtureCheckpoint:
    """Build the strict 81-label input and authoritative 97-slot mixture."""

    configure_chapter14_runtime()
    from payne_zero_atmosphere.direct_abundance import (
        DIRECT_XH_ATOMIC_NUMBERS,
        DIRECT_XH_SENTINEL_ATOMIC_NUMBERS,
        _feature_vector,
        direct_abundance_mixture_sha256,
    )

    public_xh, exact_mixture = _direct_public_input()
    features = _feature_vector(
        effective_temperature=5600.0,
        log_surface_gravity=4.1,
        microturbulence_km_s=1.7,
        abundance_vector=exact_mixture,
    )
    layout = direct_layout_trace(
        DIRECT_XH_ATOMIC_NUMBERS,
        DIRECT_XH_SENTINEL_ATOMIC_NUMBERS,
        exact_mixture,
    )
    centidex_units = exact_mixture / 0.01
    return DirectMixtureCheckpoint(
        public_xh=public_xh,
        exact_mixture=exact_mixture,
        feature_vector=features,
        layout=layout,
        mixture_sha256=direct_abundance_mixture_sha256(exact_mixture),
        public_atomic_numbers=DIRECT_XH_ATOMIC_NUMBERS,
        sentinel_atomic_numbers=DIRECT_XH_SENTINEL_ATOMIC_NUMBERS,
        public_values_quantized=bool(
            np.allclose(centidex_units, np.rint(centidex_units), rtol=0.0, atol=1e-12)
        ),
    )


def direct_decoder_checkpoint() -> DecoderCheckpoint:
    """Run the experimental set encoder and the common float64 PCA decoder."""

    configure_chapter14_runtime()
    import torch

    from payne_zero_atmosphere.direct_abundance import (
        _feature_vector,
        load_direct_abundance_initializer,
    )
    from payne_zero_atmosphere.warm_start import INITIALIZER_OUTPUT_FIELDS

    public_xh, exact_mixture = _direct_public_input()
    labels = {
        "effective_temperature": 5600.0,
        "log_surface_gravity": 4.1,
        "microturbulence_km_s": 1.7,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        initializer = load_direct_abundance_initializer(enable_experimental=True)
    checkpoint = initializer.checkpoint
    features = _feature_vector(**labels, abundance_vector=exact_mixture)
    mean = np.asarray(checkpoint["labels"]["feature_mean"], np.float64)
    std = np.asarray(checkpoint["labels"]["feature_std"], np.float64)
    standardized = (features - mean) / std
    with torch.no_grad():
        standardized_coefficients = (
            initializer.model(
                torch.as_tensor(
                    standardized[None, :], dtype=torch.float32, device="cpu"
                )
            )
            .detach()
            .cpu()
            .numpy()
        )
    pca = checkpoint["pca"]
    trace = decode_pca_coordinates(
        standardized_coefficients,
        coefficient_mean=pca["coefficient_mean"],
        coefficient_std=pca["coefficient_std"],
        basis=pca["basis"],
        coordinate_mean=pca["coordinate_mean"],
        coordinate_std=pca["coordinate_std"],
    )
    decoded_array = decode_profile_coordinates(
        trace.coordinates,
        effective_temperature=labels["effective_temperature"],
        rosseland_optical_depth=checkpoint["coordinates"][
            "standard_rosseland_optical_depth"
        ],
        acceleration_scale=float(checkpoint["coordinates"]["acceleration_scale"]),
    )
    independent = {
        field: decoded_array[:, index]
        for index, field in enumerate(INITIALIZER_OUTPUT_FIELDS)
    }
    prediction = initializer.predict(
        **labels,
        abundance_by_atomic_number=public_xh,
    )
    largest = max(
        float(np.max(np.abs(prediction[field] - independent[field])))
        for field in INITIALIZER_OUTPUT_FIELDS
    )
    loss = checkpoint["training"]["loss"]
    return DecoderCheckpoint(
        family="direct_abundance",
        labels=labels,
        checkpoint_feature_fields=tuple(checkpoint["labels"]["feature_fields"]),
        model_config=dict(checkpoint["model"]["config"]),
        feature_vector=features,
        standardized_feature_vector=standardized,
        standardized_coefficients=standardized_coefficients[0],
        pca_trace=trace,
        prediction={name: values.copy() for name, values in prediction.items()},
        independently_decoded={
            name: values.copy() for name, values in independent.items()
        },
        maximum_absolute_decode_difference=largest,
        output_shapes={name: values.shape for name, values in prediction.items()},
        output_dtypes={name: str(values.dtype) for name, values in prediction.items()},
        acceleration_scale=float(checkpoint["coordinates"]["acceleration_scale"]),
        derivative_loss_weight=float(loss["derivative_weight"]),
        optical_depth_loss_weight=float(loss["optical_depth_weight"]),
        hydrostatic_loss_weight=float(loss["hydrostatic_weight"]),
    )


def direct_warm_start_checkpoint() -> WarmStartCheckpoint:
    """Internally verify the direct decoder's one fixed-column seed boundary."""

    configure_chapter14_runtime()
    from payne_zero_atmosphere.direct_abundance import (
        _direct_abundance_warm_start_model,
    )

    public_xh, _exact_mixture = _direct_public_input()
    labels = {
        "effective_temperature": 5600.0,
        "log_surface_gravity": 4.1,
        "microturbulence_km_s": 1.7,
    }
    raw = direct_decoder_checkpoint().prediction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        atmosphere, deck = _direct_abundance_warm_start_model(
            **labels,
            abundance_by_atomic_number=public_xh,
            enable_experimental=True,
        )
    quantized = _atmosphere_prediction(atmosphere)
    differences = {}
    for field, raw_values in raw.items():
        denominator = np.maximum(np.abs(raw_values), np.finfo(np.float64).tiny)
        differences[field] = float(
            np.max(np.abs(quantized[field] - raw_values) / denominator)
        )
    return WarmStartCheckpoint(
        family="direct_abundance",
        atmosphere=atmosphere,
        deck_text=deck,
        deck_sha256=hashlib.sha256(deck.encode("utf-8")).hexdigest(),
        raw_prediction={name: values.copy() for name, values in raw.items()},
        quantized_prediction=quantized,
        maximum_relative_quantization_difference=differences,
        requested_effective_temperature=labels["effective_temperature"],
        parsed_effective_temperature=float(
            atmosphere.metadata["effective_temperature"]
        ),
        requested_log_surface_gravity=labels["log_surface_gravity"],
        parsed_log_surface_gravity=float(
            atmosphere.metadata["log_surface_gravity"]
        ),
        has_converged_field=hasattr(atmosphere, "converged"),
    )


def direct_safety_checkpoint() -> DirectSafetyCheckpoint:
    """Exercise opt-in, support, provenance, deck-only, and surrogate gates."""

    configure_chapter14_runtime()
    from payne_zero_atmosphere.direct_abundance import (
        DIRECT_XH_ATOMIC_NUMBERS,
        build_direct_abundance_optimizer_surrogate,
        complete_direct_abundance_vector,
        direct_abundance_warm_start_deck,
        load_direct_abundance_initializer,
    )

    public_xh, _exact_mixture = _direct_public_input()
    opt_in_rejected = False
    try:
        load_direct_abundance_initializer()
    except RuntimeError:
        opt_in_rejected = True

    incomplete_rejected = False
    try:
        complete_direct_abundance_vector(
            {atomic_number: -0.2 for atomic_number in DIRECT_XH_ATOMIC_NUMBERS[:-1]}
        )
    except ValueError:
        incomplete_rejected = True

    unsupported_rejected = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        initializer = load_direct_abundance_initializer(enable_experimental=True)
        try:
            initializer.predict(
                effective_temperature=11000.0,
                log_surface_gravity=4.1,
                microturbulence_km_s=1.7,
                abundance_by_atomic_number=public_xh,
            )
        except ValueError:
            unsupported_rejected = True
        surrogate = build_direct_abundance_optimizer_surrogate(
            effective_temperature=5600.0,
            log_surface_gravity=4.1,
            microturbulence_km_s=1.7,
            abundance_by_atomic_number=public_xh,
            enable_experimental_optimizer_surrogate=True,
        )
        deck = direct_abundance_warm_start_deck(
            effective_temperature=5600.0,
            log_surface_gravity=4.1,
            microturbulence_km_s=1.7,
            abundance_by_atomic_number=public_xh,
            enable_experimental=True,
        )
    return DirectSafetyCheckpoint(
        opt_in_rejected_by_default=opt_in_rejected,
        incomplete_public_vector_rejected=incomplete_rejected,
        unsupported_state_rejected=unsupported_rejected,
        release_gate_passed=initializer.provenance.release_gate_passed,
        checkpoint_sha256=initializer.provenance.sha256,
        manifest_sha256=initializer.provenance.manifest_sha256,
        role=surrogate.role,
        exact_closure_required=surrogate.exact_closure_required,
        is_final_atmosphere=surrogate.is_final_atmosphere,
        realized_mixture_sha256=surrogate.realized_mixture_sha256,
        deck_sha256=surrogate.deck_sha256,
        surrogate_identity_sha256=surrogate.surrogate_identity_sha256,
        realized_mixture_writeable=bool(
            surrogate.realized_abundance_vector.flags.writeable
        ),
        public_deck_type=type(deck).__name__,
        public_deck_exposes_model=not isinstance(deck, str),
        exact_trial_count=1,
    )


def direct_set_encoder_checkpoint() -> DirectSetEncoderCheckpoint:
    """Rebuild the source model's token response and summed representation."""

    configure_chapter14_runtime()
    import torch

    from payne_zero_atmosphere.direct_abundance import (
        _feature_vector,
        load_direct_abundance_initializer,
    )

    public_xh, exact_mixture = _direct_public_input()
    labels = {
        "effective_temperature": 5600.0,
        "log_surface_gravity": 4.1,
        "microturbulence_km_s": 1.7,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        initializer = load_direct_abundance_initializer(enable_experimental=True)
    checkpoint = initializer.checkpoint
    feature = _feature_vector(**labels, abundance_vector=exact_mixture)
    standardized = (
        feature - np.asarray(checkpoint["labels"]["feature_mean"], np.float64)
    ) / np.asarray(checkpoint["labels"]["feature_std"], np.float64)
    values = torch.as_tensor(standardized[None, :], dtype=torch.float32)
    model = initializer.model

    def manual_output(
        state: object,
        relative: object,
        embedding: object,
    ) -> tuple[object, object, object, object]:
        batch_size, element_count = relative.shape
        expanded_state = state[:, None, :].expand(batch_size, element_count, 4)
        expanded_embedding = embedding[None, :, :].expand(
            batch_size, element_count, embedding.shape[1]
        )
        amplitude = relative[:, :, None]
        token_input = torch.cat(
            (
                expanded_state,
                expanded_embedding,
                amplitude,
                amplitude.square(),
            ),
            dim=2,
        )
        law = model.response_law(token_input).reshape(
            batch_size,
            element_count,
            2,
            model.response_law[-1].out_features // 2,
        )
        response = (
            amplitude * law[:, :, 0, :]
            + amplitude.square() * law[:, :, 1, :]
        )
        summed = response.sum(dim=1)
        output = model.decoder(torch.cat((state, summed), dim=1))
        return token_input, law, summed, output

    state = values[:, :4]
    relative = values[:, 4:]
    embedding = model.element_embedding
    with torch.no_grad():
        exact_output = model(values)
        token, law, summed, manual = manual_output(state, relative, embedding)
        permutation = torch.arange(relative.shape[1] - 1, -1, -1)
        _token, _law, _sum, paired = manual_output(
            state,
            relative[:, permutation],
            embedding[permutation],
        )
        _token, _law, _sum, mismatched = manual_output(
            state,
            relative[:, permutation],
            embedding,
        )
    return DirectSetEncoderCheckpoint(
        state_shape=tuple(state.shape),
        relative_abundance_shape=tuple(relative.shape),
        element_embedding_shape=tuple(embedding.shape),
        token_input_shape=tuple(token.shape),
        response_law_shape=tuple(law.shape),
        summed_response_shape=tuple(summed.shape),
        output_shape=tuple(exact_output.shape),
        maximum_manual_output_difference=float(
            torch.max(torch.abs(exact_output - manual)).item()
        ),
        paired_permutation_output_difference=float(
            torch.max(torch.abs(exact_output - paired)).item()
        ),
        abundance_only_permutation_output_difference=float(
            torch.max(torch.abs(exact_output - mismatched)).item()
        ),
    )


def closure_seam_checkpoint() -> ClosureSeamCheckpoint:
    """Report the inherited shared-runner seam without a false closure claim."""

    configure_chapter14_runtime()
    runner = importlib.import_module("payne_zero_atmosphere.runner")
    available = tuple(name for name in RUNNER_REQUIRED_SYMBOLS if hasattr(runner, name))
    missing = tuple(name for name in RUNNER_REQUIRED_SYMBOLS if not hasattr(runner, name))
    return ClosureSeamCheckpoint(
        runner_symbols_available=available,
        runner_symbols_missing=missing,
        initializer_reader_core_executable=True,
        exact_restart_trajectory_executable=not missing,
        status="ready" if not missing else "seam_blocked",
    )


def computed_artifact_arrays() -> dict[str, np.ndarray]:
    """Compute the compact arrays compared by the Chapter 14 verifier."""

    five = decoder_checkpoint("five_label")
    cno = decoder_checkpoint("cno8")
    direct = direct_decoder_checkpoint()
    five_seed = warm_start_checkpoint("five_label")
    cno_seed = warm_start_checkpoint("cno8")
    direct_seed = direct_warm_start_checkpoint()
    mixture = direct_mixture_checkpoint()
    arrays: dict[str, np.ndarray] = {}
    for prefix, checkpoint in (
        ("five", five),
        ("cno", cno),
        ("direct", direct),
    ):
        arrays[f"{prefix}_standardized_coefficients"] = (
            checkpoint.standardized_coefficients.copy()
        )
        for field, values in checkpoint.prediction.items():
            arrays[f"{prefix}_decoded_{field}"] = values.copy()
    for prefix, checkpoint in (("five", five_seed), ("cno", cno_seed)):
        for field, values in checkpoint.quantized_prediction.items():
            arrays[f"{prefix}_quantized_{field}"] = values.copy()
    for field, values in direct_seed.quantized_prediction.items():
        arrays[f"direct_quantized_{field}"] = values.copy()
    arrays["direct_exact_mixture"] = mixture.exact_mixture.copy()
    arrays["direct_feature_vector"] = mixture.feature_vector.copy()
    return arrays
