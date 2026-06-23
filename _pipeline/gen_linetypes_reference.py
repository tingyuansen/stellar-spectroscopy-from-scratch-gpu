#!/usr/bin/env python
"""Generate ground-truth reference for pykurucz's TYPE=1 (autoionizing) and
TYPE=81 (merged-continuum) LINE opacity, by driving pykurucz's OWN synthesis
and recording every call into the two leaf routines:

    synthe_py/engine/opacity.py
        _accumulate_autoionizing_profile   (TYPE=1, AUTOIONIZING)
        _accumulate_merged_continuum       (TYPE=81 -> CONTINUUM)

pykurucz is imported READ-ONLY. We monkeypatch the two leaf functions with
recording wrappers that:
  * snapshot every scalar/array INPUT the routine receives,
  * snapshot the OUTPUT buffer DELTA the routine produces (genuine ground truth).
The real (unmodified) routine still runs, so the synthesis is unchanged.

Window:  188-670 nm  (vacuum), hot star Teff=12000 K, logg=4.0.
  TYPE=1  : Al I autoionizing lines (193.23, 193.65 nm, ground-state).
  TYPE=81 : He I/II + H I merged-continuum lines (205.06, 260.05, 312.18,
            342.19, 364.55, 364.70, 367.98, 569.62, 663.40 nm);
            plus Al I TYPE=80 merged continuum at 207.13/207.61 nm.

Everything is packed into reference/linetypes.npz.  The verify script reloads
the recorded INPUTS, recomputes the buffer deltas from scratch (numpy only),
and compares to the recorded pykurucz OUTPUT deltas to machine precision.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

PYK = Path("/Users/ysting/pykurucz")
sys.path.insert(0, str(PYK))

BOOK = Path("/Users/ysting/Stellar_Spectroscopy_From_Scratch")
REF = BOOK / "reference"
REF.mkdir(exist_ok=True)

ATM_TXT = Path("/tmp/hot12000.atm")
ATM_NPZ = Path("/tmp/hot12000.npz")

WL_START, WL_END, RES = 188.0, 670.0, 100_000.0

import pykurucz  # noqa: E402
from synthe_py import config  # noqa: E402
from synthe_py.engine import opacity  # noqa: E402

# ----------------------------------------------------------------------------
# 1. Build the hot atmosphere + npz if not present.
# ----------------------------------------------------------------------------
if not ATM_TXT.exists():
    pykurucz.emulator_warmstart_atm(ATM_TXT, teff=12000.0, logg=4.0, mh=0.0,
                                    am=0.0, vturb=2.0)
if not ATM_NPZ.exists():
    import subprocess
    subprocess.run([sys.executable,
                    str(PYK / "synthe_py/tools/convert_atm_to_npz.py"),
                    str(ATM_TXT), str(ATM_NPZ)], check=True)

# ----------------------------------------------------------------------------
# 2. Recording wrappers around the two leaf routines.
# ----------------------------------------------------------------------------
auto_records: list[dict] = []
cont_records: list[dict] = []

_real_auto = opacity._accumulate_autoionizing_profile
_real_cont = opacity._accumulate_merged_continuum


def rec_auto(buffer, continuum_row, wavelength_grid, center_index,
             line_wavelength, kappa0, gamma_rad, gamma_stark, gamma_vdw,
             cutoff):
    # Run the REAL routine on the real buffer (keeps the synthesis correct).
    ret = _real_auto(buffer, continuum_row, wavelength_grid, center_index,
                     line_wavelength, kappa0, gamma_rad, gamma_stark,
                     gamma_vdw, cutoff)
    # Capture the routine's ISOLATED, pure deposit by re-running it on a fresh
    # zero buffer.  The routine only does `buffer[idx] += value` and reads
    # nothing from the buffer, so this gives the identical physics with no
    # (a+v)-a round-off contaminating the ground truth.
    probe = np.zeros_like(buffer)
    _real_auto(probe, continuum_row, wavelength_grid, center_index,
               line_wavelength, kappa0, gamma_rad, gamma_stark, gamma_vdw,
               cutoff)
    delta = probe
    auto_records.append(dict(
        continuum_row=np.asarray(continuum_row, dtype=np.float64).copy(),
        wavelength_grid=np.asarray(wavelength_grid, dtype=np.float64).copy(),
        center_index=int(center_index),
        line_wavelength=float(line_wavelength),
        kappa0=float(kappa0),
        gamma_rad=float(gamma_rad),
        gamma_stark=float(gamma_stark),
        gamma_vdw=float(gamma_vdw),
        cutoff=float(cutoff),
        delta=delta.copy(),
        ret=bool(ret),
    ))
    return ret


def rec_cont(buffer, continuum_row, wavelength_grid, center_index,
             line_wavelength, kappa, cutoff, merge_wavelength, tail_wavelength):
    # Run the REAL routine on the real buffer, then capture the isolated pure
    # deposit on a fresh zero buffer (see rec_auto for why).
    _real_cont(buffer, continuum_row, wavelength_grid, center_index,
               line_wavelength, kappa, cutoff, merge_wavelength, tail_wavelength)
    probe = np.zeros_like(buffer)
    _real_cont(probe, continuum_row, wavelength_grid, center_index,
               line_wavelength, kappa, cutoff, merge_wavelength, tail_wavelength)
    delta = probe
    # Record the GLOBAL ramp indices exactly as the routine computes them on the
    # full grid, so the from-scratch reproduction does not need to re-run
    # searchsorted over the (sliced) grid it ships.
    npts = buffer.size
    idx_start_g = max(int(center_index), 0)
    idx_merge_g = int(np.searchsorted(wavelength_grid, merge_wavelength, side="left"))
    idx_tail_g = min(int(np.searchsorted(wavelength_grid, tail_wavelength, side="right")),
                     npts)
    cont_records.append(dict(
        continuum_row=np.asarray(continuum_row, dtype=np.float64).copy(),
        wavelength_grid=np.asarray(wavelength_grid, dtype=np.float64).copy(),
        center_index=int(center_index),
        line_wavelength=float(line_wavelength),
        kappa=float(kappa),
        cutoff=float(cutoff),
        merge_wavelength=float(merge_wavelength),
        tail_wavelength=float(tail_wavelength),
        idx_start_g=idx_start_g,
        idx_merge_g=idx_merge_g,
        idx_tail_g=idx_tail_g,
        delta=delta.copy(),
    ))


opacity._accumulate_autoionizing_profile = rec_auto
opacity._accumulate_merged_continuum = rec_cont

# ----------------------------------------------------------------------------
# 3. Run a real synthesis on the window. (atomic lines only; no molecules)
#    The .spec / diagnostics outputs are throwaway synthesis byproducts; only
#    the recorded leaf-routine inputs/outputs matter, so write them to /tmp.
# ----------------------------------------------------------------------------
spec = Path("/tmp/linetypes_hot.spec")
cfg = config.SynthesisConfig.from_cli(
    spec_path=spec,
    diagnostics_path=None,
    atmosphere_path=ATM_TXT,
    atomic_catalog=PYK / "lines" / "gfallvac.latest",
    wl_start=WL_START, wl_end=WL_END, resolution=RES,
    velocity_microturb=2.0, vacuum=True, cutoff=1e-3,
    npz_path=ATM_NPZ,
    molecular_line_dirs=[], include_tio=False, include_h2o=False,
)
cfg.line_data.cache_directory = Path("/tmp/lt_cache")
cfg.log_level = "WARNING"

opacity.run_synthesis(cfg)

print(f"autoionizing (TYPE=1)  leaf calls recorded: {len(auto_records)}")
print(f"merged-continuum (81)  leaf calls recorded: {len(cont_records)}")

# Keep only calls that actually deposited opacity (delta != 0) -- those are the
# ones that pass the cutoff and genuinely contribute.
auto_nz = [r for r in auto_records if np.any(r["delta"] != 0.0)]
cont_nz = [r for r in cont_records if np.any(r["delta"] != 0.0)]
print(f"  autoionizing with nonzero deposit: {len(auto_nz)}")
print(f"  merged-cont  with nonzero deposit: {len(cont_nz)}")
for r in auto_nz:
    print(f"    AUTO  wl={r['line_wavelength']:.4f}nm  kappa0={r['kappa0']:.3e}"
          f"  npts_deposited={int(np.count_nonzero(r['delta']))}")
for r in cont_nz:
    print(f"    CONT  wl={r['line_wavelength']:.4f}nm  kappa={r['kappa']:.3e}"
          f"  wmerge={r['merge_wavelength']:.4f}  wtail={r['tail_wavelength']:.4f}"
          f"  npts_deposited={int(np.count_nonzero(r['delta']))}")

# ----------------------------------------------------------------------------
# 4. Pack the reference npz.  ALL contributing records are kept.  Per record we
#    store the input scalars plus the slices of wavelength / continuum the
#    routine actually reads, and the pykurucz OUTPUT delta over that same slice
#    (ground truth).  The verify script recomputes the delta from these inputs.
# ----------------------------------------------------------------------------
all_recs = auto_nz + cont_nz
wl_grid = all_recs[0]["wavelength_grid"]
assert all(np.array_equal(r["wavelength_grid"], wl_grid) for r in all_recs)
NPTS = wl_grid.size

# Per record, store only the slice of the inputs the profile actually reads.
# Each accumulation walks outward from the center until the value drops below
# continuum*cutoff, so storing the wavelength + continuum slice covering the
# deposited span plus a one-index margin on each side reproduces the routine
# exactly (the break check at the first sub-cutoff index is included).
def pack(records, prefix, fields, int_fields=()):
    out = {f"{prefix}_n": np.int64(len(records))}
    for i, r in enumerate(records):
        for k in fields:
            out[f"{prefix}{i}_{k}"] = np.float64(r[k])
        for k in int_fields:
            out[f"{prefix}{i}_{k}"] = np.int64(r[k])
        delta = r["delta"].astype(np.float64)
        nz = np.nonzero(delta)[0]
        ci = int(r["center_index"])
        # Slice covers the deposited span, the center, and a one-index margin on
        # each side so the cutoff break index (first sub-cutoff point) is stored.
        lo = max(min(int(nz[0]), ci) - 1, 0)
        hi = min(max(int(nz[-1]), ci) + 2, NPTS)
        # Sanity: every deposited point must lie inside the stored slice.
        assert lo <= int(nz[0]) and int(nz[-1]) < hi
        out[f"{prefix}{i}_slice_lo"] = np.int64(lo)
        out[f"{prefix}{i}_npts"] = np.int64(NPTS)
        out[f"{prefix}{i}_wl_slice"] = wl_grid[lo:hi].astype(np.float64).copy()
        out[f"{prefix}{i}_cont_slice"] = (
            r["continuum_row"][lo:hi].astype(np.float64).copy())
        out[f"{prefix}{i}_delta_vals"] = delta[lo:hi].copy()
    return out


auto_fields = ["center_index", "line_wavelength", "kappa0", "gamma_rad",
               "gamma_stark", "gamma_vdw", "cutoff"]
cont_fields = ["center_index", "line_wavelength", "kappa", "cutoff",
               "merge_wavelength", "tail_wavelength"]

cont_int_fields = ["idx_start_g", "idx_merge_g", "idx_tail_g"]

payload = {}
payload.update(pack(auto_nz, "auto", auto_fields))
payload.update(pack(cont_nz, "cont", cont_fields, int_fields=cont_int_fields))
payload["window"] = np.array([WL_START, WL_END, RES], dtype=np.float64)
payload["teff_logg"] = np.array([12000.0, 4.0], dtype=np.float64)

np.savez_compressed(REF / "linetypes.npz", **payload)
sz = (REF / "linetypes.npz").stat().st_size
print(f"\nfull grid {NPTS} pts; per-record input/output slices shipped")
print(f"wrote {REF/'linetypes.npz'}  ({sz/1024:.1f} KiB)")
