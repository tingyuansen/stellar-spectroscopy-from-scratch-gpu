"""Transparent calculations used while building the Chapter 14 initializer.

These helpers expose representation, ordering, and dtype transitions.  Exact
checkpoint loading, dispatch, deck quantization, and safety gates remain in
the byte-pinned Payne Zero modules exercised by ``chapter14_runtime``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PROFILE_FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)
COORDINATE_FIELDS = (
    "log10_column_mass_increment",
    "log10_temperature_relative_to_grey",
    "log10_gas_pressure",
    "log10_electron_density",
    "log10_rosseland_opacity",
    "asinh_radiative_acceleration",
)


@dataclass(frozen=True)
class ProfileRoundTrip:
    """One six-profile transform and inverse-transform comparison."""

    coordinates: np.ndarray
    reconstructed: np.ndarray
    maximum_relative_difference: np.ndarray


@dataclass(frozen=True)
class PcaDecodeTrace:
    """Arrays on both sides of the one Torch-to-NumPy inference boundary."""

    network_output_dtype: str
    coefficients_dtype: str
    standardized_coordinates_dtype: str
    coordinates_dtype: str
    coefficients: np.ndarray
    standardized_coordinates: np.ndarray
    coordinates: np.ndarray


@dataclass(frozen=True)
class PcaSentinelTrace:
    """Location reached by one nonzero coefficient in C-order reconstruction."""

    component_index: int
    requested_layer_index: int
    requested_field_index: int
    flattened_index: int
    recovered_layer_index: int
    recovered_field_index: int


@dataclass(frozen=True)
class FixedPointTrace:
    """A small contraction-map trajectory used to define a convergence basin."""

    values: np.ndarray
    residuals: np.ndarray
    contraction: float
    fixed_point: float


@dataclass(frozen=True)
class DirectLayoutTrace:
    """The public, network, and exact-mixture dimensions for one direct input."""

    public_atomic_numbers: np.ndarray
    sentinel_atomic_numbers: np.ndarray
    public_abundance_count: int
    network_feature_count: int
    exact_mixture_count: int
    sentinel_count: int
    iron_abundance: float
    maximum_sentinel_difference_from_iron: float


def grey_temperature(
    effective_temperature: float,
    rosseland_optical_depth: np.ndarray,
) -> np.ndarray:
    """Return the grey reference profile used by every initializer family."""

    depth = np.asarray(rosseland_optical_depth, dtype=np.float64)
    return float(effective_temperature) * (0.75 * (depth + 2.0 / 3.0)) ** 0.25


def encode_profile_coordinates(
    profile: np.ndarray,
    *,
    effective_temperature: float,
    rosseland_optical_depth: np.ndarray,
    acceleration_scale: float,
) -> np.ndarray:
    """Apply the six checkpoint transforms in declared column order."""

    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("profile must have shape (layers, 6)")
    if not np.all(np.isfinite(values)):
        raise ValueError("profile must be finite")
    if np.any(values[:, :5] <= 0.0) or np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError("the first five profiles must satisfy their constraints")
    scale = float(acceleration_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("acceleration_scale must be finite and positive")

    increments = np.diff(values[:, 0], prepend=0.0)
    grey = grey_temperature(effective_temperature, rosseland_optical_depth)
    coordinates = np.empty_like(values, dtype=np.float64)
    coordinates[:, 0] = np.log10(increments)
    coordinates[:, 1] = np.log10(values[:, 1] / grey)
    coordinates[:, 2:5] = np.log10(values[:, 2:5])
    coordinates[:, 5] = np.arcsinh(values[:, 5] / scale)
    return coordinates


def decode_profile_coordinates(
    coordinates: np.ndarray,
    *,
    effective_temperature: float,
    rosseland_optical_depth: np.ndarray,
    acceleration_scale: float,
) -> np.ndarray:
    """Apply the exact clipped inverse transforms used after PCA decoding."""

    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("coordinates must have shape (layers, 6)")
    grey = grey_temperature(effective_temperature, rosseland_optical_depth)
    decoded = np.empty_like(values, dtype=np.float64)
    decoded[:, 0] = np.cumsum(10.0 ** np.clip(values[:, 0], -30.0, 30.0))
    decoded[:, 1] = grey * 10.0 ** np.clip(values[:, 1], -3.0, 3.0)
    decoded[:, 2:5] = 10.0 ** np.clip(values[:, 2:5], -30.0, 30.0)
    decoded[:, 5] = float(acceleration_scale) * np.sinh(
        np.clip(values[:, 5], -20.0, 20.0)
    )
    return decoded


def profile_transform_round_trip(
    profile: np.ndarray,
    *,
    effective_temperature: float,
    rosseland_optical_depth: np.ndarray,
    acceleration_scale: float,
) -> ProfileRoundTrip:
    """Encode then decode six profiles and report one error per field."""

    original = np.asarray(profile, dtype=np.float64)
    coordinates = encode_profile_coordinates(
        original,
        effective_temperature=effective_temperature,
        rosseland_optical_depth=rosseland_optical_depth,
        acceleration_scale=acceleration_scale,
    )
    reconstructed = decode_profile_coordinates(
        coordinates,
        effective_temperature=effective_temperature,
        rosseland_optical_depth=rosseland_optical_depth,
        acceleration_scale=acceleration_scale,
    )
    denominator = np.maximum(np.abs(original), np.finfo(np.float64).tiny)
    difference = np.max(np.abs(reconstructed - original) / denominator, axis=0)
    return ProfileRoundTrip(
        coordinates=coordinates,
        reconstructed=reconstructed,
        maximum_relative_difference=difference,
    )


def decode_pca_coordinates(
    standardized_coefficients_float32: np.ndarray,
    *,
    coefficient_mean: np.ndarray,
    coefficient_std: np.ndarray,
    basis: np.ndarray,
    coordinate_mean: np.ndarray,
    coordinate_std: np.ndarray,
) -> PcaDecodeTrace:
    """Reproduce the exact CPU NumPy float64 half of the common decoder."""

    network_output = np.asarray(standardized_coefficients_float32)
    if network_output.dtype != np.float32:
        raise ValueError("network output must arrive as float32")
    if network_output.shape not in {(160,), (1, 160)}:
        raise ValueError("network output must contain 160 PCA coefficients")
    row = network_output.reshape(1, 160)
    coefficients = row * np.asarray(coefficient_std, np.float64) + np.asarray(
        coefficient_mean, np.float64
    )
    standardized_coordinates = coefficients @ np.asarray(basis, np.float64)
    flattened = standardized_coordinates * np.asarray(
        coordinate_std, np.float64
    ) + np.asarray(coordinate_mean, np.float64)
    coordinates = flattened.reshape(80, 6)
    return PcaDecodeTrace(
        network_output_dtype=str(network_output.dtype),
        coefficients_dtype=str(coefficients.dtype),
        standardized_coordinates_dtype=str(standardized_coordinates.dtype),
        coordinates_dtype=str(coordinates.dtype),
        coefficients=coefficients[0],
        standardized_coordinates=standardized_coordinates[0],
        coordinates=coordinates,
    )


def pca_sentinel_trace(
    *,
    component_index: int = 7,
    layer_index: int = 23,
    field_index: int = 4,
) -> PcaSentinelTrace:
    """Pin the stored `(160, 480)` basis orientation and C-order reshape."""

    component = int(component_index)
    layer = int(layer_index)
    field = int(field_index)
    if not 0 <= component < 160 or not 0 <= layer < 80 or not 0 <= field < 6:
        raise ValueError("sentinel indices are outside the decoder layout")
    flattened_index = 6 * layer + field
    coefficients = np.zeros((1, 160), dtype=np.float64)
    basis = np.zeros((160, 480), dtype=np.float64)
    coefficients[0, component] = 1.0
    basis[component, flattened_index] = 1.0
    reconstructed = (coefficients @ basis).reshape(80, 6)
    recovered = np.argwhere(reconstructed == 1.0)
    if recovered.shape != (1, 2):
        raise RuntimeError("PCA sentinel did not recover one coordinate")
    return PcaSentinelTrace(
        component_index=component,
        requested_layer_index=layer,
        requested_field_index=field,
        flattened_index=flattened_index,
        recovered_layer_index=int(recovered[0, 0]),
        recovered_field_index=int(recovered[0, 1]),
    )


def fixed_point_contraction_trace(
    initial_value: float,
    *,
    fixed_point: float = 1.0,
    contraction: float = 0.55,
    passes: int = 8,
) -> FixedPointTrace:
    """Iterate a scalar contraction to separate a good start from closure."""

    if not 0.0 <= abs(float(contraction)) < 1.0:
        raise ValueError("a contraction must have absolute slope below one")
    values = np.empty(int(passes) + 1, dtype=np.float64)
    residuals = np.empty(int(passes), dtype=np.float64)
    values[0] = float(initial_value)
    for index in range(int(passes)):
        next_value = float(fixed_point) + float(contraction) * (
            values[index] - float(fixed_point)
        )
        residuals[index] = next_value - values[index]
        values[index + 1] = next_value
    return FixedPointTrace(
        values=values,
        residuals=residuals,
        contraction=float(contraction),
        fixed_point=float(fixed_point),
    )


def direct_layout_trace(
    public_atomic_numbers: tuple[int, ...],
    sentinel_atomic_numbers: tuple[int, ...],
    exact_mixture: np.ndarray,
) -> DirectLayoutTrace:
    """Summarize one exact direct-abundance 81/84/97 handoff."""

    public = np.asarray(public_atomic_numbers, dtype=np.int64)
    sentinel = np.asarray(sentinel_atomic_numbers, dtype=np.int64)
    mixture = np.asarray(exact_mixture, dtype=np.float64)
    iron = float(mixture[26 - 3])
    sentinel_values = mixture[sentinel - 3]
    return DirectLayoutTrace(
        public_atomic_numbers=public,
        sentinel_atomic_numbers=sentinel,
        public_abundance_count=int(public.size),
        network_feature_count=4 + int(public.size) - 1,
        exact_mixture_count=int(mixture.size),
        sentinel_count=int(sentinel.size),
        iron_abundance=iron,
        maximum_sentinel_difference_from_iron=float(
            np.max(np.abs(sentinel_values - iron))
        ),
    )


def set_encoder_token_inputs(
    state_features: np.ndarray,
    element_embedding: np.ndarray,
    relative_abundances: np.ndarray,
) -> np.ndarray:
    """Construct the exact state/identity/linear/quadratic token inputs."""

    state = np.asarray(state_features, dtype=np.float32)
    embedding = np.asarray(element_embedding, dtype=np.float32)
    relative = np.asarray(relative_abundances, dtype=np.float32)
    if state.shape != (4,) or embedding.ndim != 2:
        raise ValueError("state must be (4,) and embedding must be two-dimensional")
    if relative.shape != (embedding.shape[0],):
        raise ValueError("one abundance must accompany every element embedding")
    expanded_state = np.broadcast_to(state, (relative.size, 4))
    amplitude = relative[:, None]
    return np.concatenate(
        (expanded_state, embedding, amplitude, amplitude**2),
        axis=1,
    )


def quantize_centidex(values: np.ndarray) -> np.ndarray:
    """Apply the direct initializer's exact NumPy 0.01-dex lattice rule."""

    quantized = np.round(np.asarray(values, np.float64) / 0.01) * 0.01
    quantized[quantized == 0.0] = 0.0
    return quantized
