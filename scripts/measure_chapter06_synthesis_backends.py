"""Measure Chapter 6 device results against the accepted CPU authority.

This command is deliberately read-only.  It computes the reader-built
one-record result before opening the comparison golden, repeats every available
backend twice, and writes one canonical JSON report to standard output.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from book.chapter06_runtime import (
    BOLTZMANN_ERG_PER_K,
    LIGHT_SPEED_NM_PER_S,
    PLANCK_ERG_SECOND,
    REPOSITORY_ROOT,
    run_synthesis_one_line,
    synthesis_line_state,
)


GOLDEN_PATH = (
    REPOSITORY_ROOT
    / "data/golden/payne_zero/chapter06/synthesis"
    / "chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz"
)
EXPECTED_GOLDEN_SHA256 = (
    "a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955"
)
REPEATS = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float32_ulp_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return ULP distance where two finite float32 values have the same sign."""

    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b) & (np.signbit(a) == np.signbit(b))
    distance = np.zeros(a.shape, dtype=np.uint64)
    if np.any(valid):
        a_bits = a[valid].view(np.uint32).astype(np.int64)
        b_bits = b[valid].view(np.uint32).astype(np.int64)
        distance[valid] = np.abs(a_bits - b_bits).astype(np.uint64)
    return distance


def _relative_error(observed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    observed64 = np.asarray(observed, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    union_nonzero = (observed64 != 0.0) | (reference64 != 0.0)
    result = np.zeros(observed64.shape, dtype=np.float64)
    denominator = np.maximum(np.abs(observed64), np.abs(reference64))
    np.divide(
        np.abs(observed64 - reference64),
        denominator,
        out=result,
        where=union_nonzero,
    )
    return result


def _frequency_integrals(
    slab: np.ndarray,
    wavelength_nm: np.ndarray,
) -> np.ndarray:
    frequency_hz = 2.99792458e17 / np.asarray(wavelength_nm, dtype=np.float64)
    order = np.argsort(frequency_hz)
    return np.trapezoid(
        np.asarray(slab, dtype=np.float64)[..., order],
        x=frequency_hz[order],
        axis=-1,
    )


def _quantity_metrics(
    observed: np.ndarray,
    reference: np.ndarray,
    *,
    wavelength_nm: np.ndarray,
    center_index: int,
) -> dict[str, Any]:
    observed_array = np.asarray(observed, dtype=np.float32)
    reference_array = np.asarray(reference, dtype=np.float32)
    difference = observed_array.astype(np.float64) - reference_array.astype(
        np.float64
    )
    relative = _relative_error(observed_array, reference_array)
    observed_integral = _frequency_integrals(observed_array, wavelength_nm)
    reference_integral = _frequency_integrals(reference_array, wavelength_nm)
    integral_relative = _relative_error(observed_integral, reference_integral)
    ulp = _float32_ulp_distance(observed_array, reference_array)
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "maximum_relative_error_union_nonzero": float(np.max(relative)),
        "maximum_float32_ulp_distance_same_sign_finite": int(np.max(ulp)),
        "zero_pattern_disagreement_count": int(
            np.count_nonzero((observed_array == 0.0) != (reference_array == 0.0))
        ),
        "maximum_center_absolute_error": float(
            np.max(
                np.abs(
                    difference[
                        ...,
                        int(center_index),
                    ]
                )
            )
        ),
        "maximum_frequency_integral_absolute_error": float(
            np.max(np.abs(observed_integral - reference_integral))
        ),
        "maximum_frequency_integral_relative_error_union_nonzero": float(
            np.max(integral_relative)
        ),
    }


def _mps_hardware() -> dict[str, Any]:
    information: dict[str, Any] = {
        "device_name": "Apple Metal Performance Shaders device",
        "architecture": platform.machine(),
        "driver_runtime": f"macOS {platform.mac_ver()[0]} / Metal",
    }
    try:
        completed = subprocess.run(
            ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        displays = json.loads(completed.stdout).get("SPDisplaysDataType", [])
        gpu = next(
            (
                item
                for item in displays
                if item.get("sppci_device_type") == "spdisplays_gpu"
            ),
            None,
        )
        if gpu is not None:
            information.update(
                device_name=str(
                    gpu.get("sppci_model", gpu.get("_name", information["device_name"]))
                ),
                gpu_cores=str(gpu.get("sppci_cores", "unknown")),
                metal_family=str(
                    gpu.get("spdisplays_mtlgpufamilysupport", "unknown")
                ),
            )
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        information["hardware_query"] = "unavailable"
    return information


def _backend_inventory() -> dict[str, dict[str, Any]]:
    import torch

    inventory: dict[str, dict[str, Any]] = {
        "cpu": {
            "available": True,
            "device_name": platform.processor() or platform.machine(),
            "architecture": platform.machine(),
            "driver_runtime": platform.platform(),
        },
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "unavailable"
            ),
            "architecture": (
                ".".join(str(value) for value in torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else "unavailable"
            ),
            "driver_runtime": (
                str(torch.version.cuda)
                if torch.cuda.is_available()
                else "unavailable"
            ),
        },
        "mps": {
            "available": bool(
                torch.backends.mps.is_built() and torch.backends.mps.is_available()
            ),
            "built": bool(torch.backends.mps.is_built()),
            **_mps_hardware(),
        },
    }
    return inventory


def _capture_backend(
    backend: str,
    *,
    regime_names: tuple[str, ...],
    wavelength_nm: np.ndarray,
) -> dict[str, Any]:
    import torch

    gross_rows = []
    net_rows = []
    activity_rows = []
    reach_rows = []
    stimulation_residuals = []
    contracts = []
    for regime in regime_names:
        result = run_synthesis_one_line(
            regime,
            wavelength_nm=wavelength_nm,
            runtime_device=backend,
        )
        gross_rows.append(result.gross_line_mass_absorption_coefficient)
        net_rows.append(result.net_line_mass_absorption_coefficient)
        activity_rows.append(result.activity_mask)
        reach_rows.append(result.wing_reach)
        state, _ = synthesis_line_state(regime, wavelength_nm)
        device = result.gross_line_mass_absorption_tensor.device
        work_dtype = torch.float32 if device.type == "mps" else torch.float64
        wavelength_tensor = torch.as_tensor(
            wavelength_nm,
            dtype=work_dtype,
            device=device,
        )
        temperature_tensor = torch.as_tensor(
            state["temperature"],
            dtype=work_dtype,
            device=device,
        )
        frequency_grid_hz = (LIGHT_SPEED_NM_PER_S / wavelength_tensor).to(
            torch.float32
        )
        photon_temperature_factor = (
            PLANCK_ERG_SECOND
            / (BOLTZMANN_ERG_PER_K * temperature_tensor)
        ).to(torch.float32)
        factor = 1.0 - torch.exp(
            -frequency_grid_hz[None, :]
            * photon_temperature_factor[:, None]
        )
        reconstructed_net = (
            result.gross_line_mass_absorption_tensor * factor
        ).detach().cpu().numpy()
        stimulation_residuals.append(
            float(
                np.max(
                    np.abs(
                        result.net_line_mass_absorption_coefficient.astype(
                            np.float64
                        )
                        - reconstructed_net.astype(np.float64)
                    )
                )
            )
        )
        contracts.append(
            {
                "regime": regime,
                "shape": list(result.gross_line_mass_absorption_coefficient.shape),
                "work_dtype": result.work_dtype,
                "accumulation_dtype": result.accumulation_dtype,
                "device": result.device,
                "metal_line_count": result.metal_line_count,
                "auto_line_count": result.auto_line_count,
                "helium_line_count": result.helium_line_count,
                "population_ion_stage_index": result.population_ion_stage_index,
                "population_element_index": result.population_element_index,
                "finite_nonnegative_gross": bool(
                    np.all(np.isfinite(gross_rows[-1]))
                    and np.all(gross_rows[-1] >= 0.0)
                ),
                "finite_nonnegative_net": bool(
                    np.all(np.isfinite(net_rows[-1]))
                    and np.all(net_rows[-1] >= 0.0)
                ),
            }
        )
    return {
        "gross": np.stack(gross_rows),
        "net": np.stack(net_rows),
        "activity": np.stack(activity_rows),
        "reach": np.stack(reach_rows),
        "maximum_stimulation_identity_absolute_error": float(
            max(stimulation_residuals)
        ),
        "contracts": contracts,
    }


def build_report() -> dict[str, Any]:
    """Return the complete read-only two-repeat backend measurement."""

    import torch

    if _sha256(GOLDEN_PATH) != EXPECTED_GOLDEN_SHA256:
        raise RuntimeError("the accepted Chapter 6 synthesis golden identity changed")
    with np.load(GOLDEN_PATH, allow_pickle=False) as archive:
        regime_names = tuple(str(value) for value in archive["axis__regime_name"])
        wavelength_nm = np.asarray(archive["axis__wavelength_nm"], dtype=np.float64)
        reference_gross = np.asarray(archive["opacity__gross_float32"], dtype=np.float32)
        reference_net = np.asarray(archive["opacity__net_float32"], dtype=np.float32)
        reference_activity = np.asarray(
            archive["activity__post_cutoff_mask"], dtype=bool
        )
        reference_reach = np.asarray(archive["ledger__wing_reach"], dtype=np.int64)
        center_index = int(archive["coarse__line_center_index"])

    inventory = _backend_inventory()
    measurements: dict[str, Any] = {}
    for backend, hardware in inventory.items():
        if not hardware["available"]:
            measurements[backend] = {"status": "unavailable", "hardware": hardware}
            continue
        repeats = [
            _capture_backend(
                backend,
                regime_names=regime_names,
                wavelength_nm=wavelength_nm,
            )
            for _ in range(REPEATS)
        ]
        repeat_gross_difference = np.asarray(repeats[1]["gross"]) - np.asarray(
            repeats[0]["gross"]
        )
        repeat_net_difference = np.asarray(repeats[1]["net"]) - np.asarray(
            repeats[0]["net"]
        )
        per_repeat = []
        for capture in repeats:
            per_repeat.append(
                {
                    "gross": _quantity_metrics(
                        capture["gross"],
                        reference_gross,
                        wavelength_nm=wavelength_nm,
                        center_index=center_index,
                    ),
                    "net": _quantity_metrics(
                        capture["net"],
                        reference_net,
                        wavelength_nm=wavelength_nm,
                        center_index=center_index,
                    ),
                    "activity_mask_disagreement_count": int(
                        np.count_nonzero(capture["activity"] != reference_activity)
                    ),
                    "reach_disagreement_count": int(
                        np.count_nonzero(capture["reach"] != reference_reach)
                    ),
                    "maximum_stimulation_identity_absolute_error": capture[
                        "maximum_stimulation_identity_absolute_error"
                    ],
                    "contracts": capture["contracts"],
                }
            )
        measurements[backend] = {
            "status": "measured",
            "hardware": hardware,
            "repeats": per_repeat,
            "repeat_variability": {
                "gross_maximum_absolute_difference": float(
                    np.max(np.abs(repeat_gross_difference))
                ),
                "gross_maximum_float32_ulp_distance": int(
                    np.max(
                        _float32_ulp_distance(
                            repeats[1]["gross"], repeats[0]["gross"]
                        )
                    )
                ),
                "net_maximum_absolute_difference": float(
                    np.max(np.abs(repeat_net_difference))
                ),
                "net_maximum_float32_ulp_distance": int(
                    np.max(
                        _float32_ulp_distance(repeats[1]["net"], repeats[0]["net"])
                    )
                ),
                "activity_mask_disagreement_count": int(
                    np.count_nonzero(
                        repeats[1]["activity"] != repeats[0]["activity"]
                    )
                ),
                "reach_disagreement_count": int(
                    np.count_nonzero(repeats[1]["reach"] != repeats[0]["reach"])
                ),
            },
        }

    return {
        "schema_version": 1,
        "scope": "Chapter 6 one-record synthesis backend comparison",
        "authority": {
            "path": str(GOLDEN_PATH.relative_to(REPOSITORY_ROOT)),
            "sha256": EXPECTED_GOLDEN_SHA256,
            "role": "CPU-only comparison golden",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "repeats_per_available_backend": REPEATS,
        "regimes": list(regime_names),
        "wavelength_count": int(wavelength_nm.size),
        "measurements": measurements,
    }


def main() -> None:
    print(json.dumps(build_report(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
