#!/usr/bin/env python
"""Generate the Lecture 10 (radiative equilibrium / temperature correction) reference.

This is run ONCE by the author against the read-only pykurucz tree; it produces
reference/tcorr_ref.npz, which the lecture notebook (and _pipeline/verify_tcorr.py)
load and compare to.  The notebook never imports pykurucz.

Deterministic configuration (matches the lecture's claims):
  * grey cold start, Teff=5770, logg=4.44  (atlas_py.physics.grey_start)
  * ONE atlas_py iteration (run_atlas, iterations=1)
  * CONTINUUM ONLY  -> IFOP(15)=0, IFOP(14)=0, IFOP(17)=0  (no 5 GB line data)
  * CONVECTION OFF  -> enable_convection=False (IFCONV=0)
  * SERIAL          -> n_workers=1 (bit-exact, no thread-order nondeterminism)

What we capture for that one iteration:
  INPUT grey atmosphere   : tau (TAUSTD), T, RHOX, P, XNE  (from grey_start)
  per-frequency continuum : freq_hz grid, rco quadrature weights,
                            acont (absorption), sigmac (scattering), scont (source)
                            -- these are the KAPP outputs (verified in Lecture 3),
                            treated as a GIVEN input here so L10 can focus on the
                            Rosseland mean + JOSH flux + temperature correction.
  TCORR accumulators      : flxrad, rjmins, rdabh, rdiagj  (post frequency loop)
  ROSS results            : abross (kappa_Rosseland), tauros (Rosseland tau)
  OUTPUT corrected model  : corrected T, RHOX (after the full run_atlas iteration,
                            i.e. AFTER the TAUSTD remap that closes the iteration)

The continuum opacity arrays are stored as float32 to keep the file small; this is
loss-free relative to how JOSH consumes them (it casts the iterate to float32
internally) and is the bottleneck of the achievable precision anyway.
"""
from __future__ import annotations

import os
# Pin numba threads BEFORE any numba-backed import so ThreadingPolicy.apply()
# (called inside run_atlas with n_workers=1) does not try to change it later.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/ysting/pykurucz")  # read-only reference code
BOOK = Path(__file__).resolve().parent.parent
REF = BOOK / "reference"
REF.mkdir(exist_ok=True)

TEFF = 5770.0
LOGG = 4.44


def _write_grey_continuum_atm(dest: Path) -> None:
    """Write a grey starting .atm with continuum-only IFOP flags."""
    import pykurucz
    from atlas_py.physics.grey_start import generate_grey_atm

    g = generate_grey_atm(teff=TEFF, logg=LOGG)
    n = g.tauros.size
    # 9-column DECK6: RHOX, T, P, XNE, ABROSS, ACCRAD, VTURB, FLXCNV, VCONV
    deck = np.column_stack([
        g.rhox, g.temperature, g.p_gas, g.xne, g.abross,
        g.accrad, g.vturb, g.flxcnv, g.vconv,
    ])
    pykurucz.write_atm_file(
        dest, TEFF, LOGG, deck, 0.0, mh=0.0, am=0.0,
        title="grey continuum-only start (Stellar Spectroscopy L10)",
    )
    # Flip IFOP to continuum-only: zero positions 14, 15, 17 (1-based) ->
    # indices 13, 14, 16 (0-based) -> HLINOP off, LINOP1 off, XLINOP off.
    text = dest.read_text(encoding="utf-8")
    cont_ifop = " OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0"
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("OPACITY IFOP"):
            lines[i] = cont_ifop
            break
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import logging
    logging.basicConfig(level=logging.WARNING)

    from atlas_py.physics.grey_start import generate_grey_atm
    from atlas_py.config import AtlasConfig, AtlasInput, AtlasOutput
    from atlas_py.engine.driver import run_atlas

    # --- grey starting model (the deterministic seed) ---
    grey = generate_grey_atm(teff=TEFF, logg=LOGG)

    atm_path = REF / "_tcorr_grey.atm"
    out_atm = REF / "_tcorr_out.atm"
    dbg_path = REF / "_tcorr_debug.npz"
    _write_grey_continuum_atm(atm_path)

    cfg = AtlasConfig(
        inputs=AtlasInput(atmosphere_path=atm_path),
        outputs=AtlasOutput(output_atm_path=out_atm, debug_state_path=dbg_path),
        iterations=1,
        enable_molecules=False,
        enable_convection=False,   # IFCONV = 0
        enable_scattering=True,
        n_workers=1,               # serial -> bit-exact
    )
    run_atlas(cfg)
    dbg = dict(np.load(dbg_path))

    # --- recompute the continuum-opacity grid the iteration consumed ---
    # We reproduce the driver's per-frequency continuum inputs by repeating the
    # iteration-1 setup (POPS/POPSALL from the grey start) and calling
    # kapcont_baseline with continuum-only ifop.  This is what JOSH/ROSS/TCORR ate.
    from atlas_py.io.atmosphere import load_atm
    from atlas_py.engine.driver import (
        _runtime_from_atm, _build_kapp_adapter, _parse_ifop_flags,
        _gravity_from_atm_metadata,
    )
    from atlas_py.physics.atlas_tables import load_atlas_tables
    from atlas_py.physics.kapcont import kapcont_baseline
    from atlas_py.physics.populations import pops
    from atlas_py.physics.popsall import popsall
    from atlas_py.physics.doppler import update_doppler_populations
    from atlas_py.physics.tcorr import init_tcorr

    atm = load_atm(atm_path)
    state = _runtime_from_atm(atm)
    ifop = _parse_ifop_flags(atm)
    gravity_cgs = _gravity_from_atm_metadata(atm)
    tables = load_atlas_tables()
    n_layers = atm.layers
    tcst = init_tcorr(n_layers)

    # iteration-1 EOS setup (driver lines ~994-1036): on itemp==1 P stays from the
    # input model; POPS/POPSALL run; DOPPLER updates.
    itemp = 1
    itemp_cache = {"pops_itemp": -1}
    prad = np.zeros(n_layers); pturb = np.zeros(n_layers)
    tk_erg = atm.tk
    state.chargesq[:] = 2.0 * state.xne
    excess = 2.0 * state.xne - state.p / np.maximum(tk_erg, 1e-300)
    state.chargesq[excess > 0] += 2.0 * excess[excess > 0]
    dummy = np.zeros((n_layers, 1))
    pops(code=0.0, mode=1, out=dummy, ifmol=False, ifpres=True,
         temperature_k=atm.temperature, tk_erg=atm.tk, state=state,
         itemp=itemp, itemp_cache=itemp_cache)
    popsall(temperature_k=atm.temperature, tk_erg=atm.tk, state=state,
            ifmol=False, ifpres=True, itemp=itemp, itemp_cache=itemp_cache)
    update_doppler_populations(tk_erg=atm.tk, vturb_cms=atm.vturb, state=state)

    adapter = _build_kapp_adapter(atm, state)
    wave_nm, rco, acont_all, sigmac_all, scont_all = kapcont_baseline(
        adapter=adapter, teff=TEFF, atlas_tables=tables, ifop=ifop, tcst=tcst,
    )
    freq_all = 2.99792458e17 / np.maximum(wave_nm, 1e-300)

    # Replay the frequency loop ONLY for ROSS, to capture the RAW (pre-TAUSTD-remap)
    # Rosseland opacity and optical-depth scale (the debug-state dump stores the
    # post-remap versions; the temperature correction consumes the raw ones).
    from atlas_py.physics.ross import ross_step
    from atlas_py.physics.josh import josh_depth_profiles
    T0 = np.asarray(atm.temperature, dtype=np.float64)
    rhox0 = np.asarray(atm.rhox, dtype=np.float64)
    hkt0 = 6.6256e-27 / np.maximum(T0 * 1.38054e-16, 1e-300)
    n0 = T0.size; nf0 = freq_all.size; zc = np.zeros(n0)
    ab_acc = np.zeros(n0)
    ross_step(ab_acc, mode=1, rcowt=0.0, bnu=zc, freq_hz=0.0, hkt=hkt0,
              temperature_k=T0, stim=np.ones(n0), abtot=np.ones(n0), numnu=nf0, rhox=rhox0)
    for inu in range(nf0):
        f = float(freq_all[inu]); rcowt = float(rco[inu])
        ehvkt = np.exp(-f * hkt0); stim = np.maximum(1.0 - ehvkt, 1e-300)
        bnu = 1.47439e-2 * ((f / 1e15) ** 3) * ehvkt / stim
        jr = josh_depth_profiles(
            ifscat=1, ifsurf=0, acont=acont_all[:, inu], scont=scont_all[:, inu],
            aline=zc, sline=bnu, sigmac=sigmac_all[:, inu], sigmal=zc,
            rhox=rhox0, bnu=bnu, freq_hz=f, wave_nm=float(wave_nm[inu]))
        ross_step(ab_acc, mode=2, rcowt=rcowt, bnu=bnu, freq_hz=f, hkt=hkt0,
                  temperature_k=T0, stim=stim, abtot=jr.abtot, numnu=nf0, rhox=rhox0)
    abross_raw, tauros_raw = ross_step(
        ab_acc, mode=3, rcowt=0.0, bnu=zc, freq_hz=0.0, hkt=hkt0,
        temperature_k=T0, stim=np.ones(n0), abtot=np.ones(n0), numnu=nf0, rhox=rhox0)

    # JOSH H-moment operator COEFH (maps the source vector onto H_nu on the fixed
    # xtau grid).  The shared josh_tables.npz (Lecture 8) ships only xtau/ch/coefj;
    # L10 also needs COEFH for the per-frequency flux, so ship it here.
    from atlas_py.physics.josh_tables_atlas12 import load_josh_tables
    josh_coefh = np.asarray(load_josh_tables()["coefh_matrix"], dtype=np.float64)

    out = REF / "tcorr_ref.npz"
    np.savez_compressed(
        out,
        # configuration scalars
        teff=np.float64(TEFF), logg=np.float64(LOGG),
        gravity_cgs=np.float64(gravity_cgs),
        tau1lg=np.float64(-6.875), steplg=np.float64(0.125),
        # JOSH H-moment operator (Lecture 8 ships only ch/coefj; L10 needs coefh too)
        josh_coefh=josh_coefh,
        # grey starting model (input)
        grey_tau=grey.tauros, grey_T=grey.temperature, grey_rhox=grey.rhox,
        grey_pgas=grey.p_gas, grey_xne=grey.xne,
        # EOS state at iteration 1 (the grey start, after POPS) -- T, RHOX, P, XNE
        T_in=np.asarray(atm.temperature, dtype=np.float64),
        rhox_in=np.asarray(atm.rhox, dtype=np.float64),
        p_in=np.asarray(state.p, dtype=np.float64),
        xne_in=np.asarray(state.xne, dtype=np.float64),
        # per-frequency continuum opacity inputs (KAPP, treated as GIVEN here)
        freq_hz=np.asarray(freq_all, dtype=np.float64),
        rco=np.asarray(rco, dtype=np.float64),
        acont=np.asarray(acont_all, dtype=np.float32),
        sigmac=np.asarray(sigmac_all, dtype=np.float32),
        scont=np.asarray(scont_all, dtype=np.float32),
        # TCORR accumulators (post frequency loop)
        flxrad_ref=dbg["flxrad_out"], rjmins_ref=dbg["rjmins_out"],
        rdabh_ref=dbg["rdabh_out"], rdiagj_ref=dbg["rdiagj_out"],
        # ROSS results (raw = pre-remap, what TCORR consumes; post = after TAUSTD remap)
        abross_ref=abross_raw, tauros_ref=tauros_raw,
        abross_remapped=dbg["abross_out"], tauros_remapped=dbg["tauros_out"],
        prad_ref=dbg["prad_out"],
        # correction diagnostics
        flxerr_ref=dbg["flxerr_out"], flxdrv_ref=dbg["flxdrv_out"],
        dtflux_ref=dbg["dtflux_out"], dtlamb_ref=dbg["dtlamb_out"],
        t1_ref=dbg["t1_out"],
        # OUTPUT corrected model (after one full run_atlas iteration + TAUSTD remap)
        T_out=dbg["temperature"], rhox_out=dbg["rhox"],
    )
    print(f"wrote {out}")
    d = np.load(out)
    print("arrays:", ", ".join(d.files))
    print(f"  n_freq={freq_all.size}  n_layers={n_layers}")
    print(f"  acont/sigmac/scont shape = {acont_all.shape} (float32)")
    sz = out.stat().st_size / 1e6
    print(f"  file size = {sz:.1f} MB")
    # cleanup scratch files
    for p in (atm_path, out_atm, dbg_path):
        p.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
