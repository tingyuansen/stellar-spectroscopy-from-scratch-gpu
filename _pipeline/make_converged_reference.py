#!/usr/bin/env python
"""Generate the Lecture 11 (convection + converged model) reference.

Run ONCE by the author against the read-only pykurucz tree; it produces
reference/converged_ref.npz, which the lecture notebook (and
_pipeline/verify_converged.py) load and compare to.  The notebook never imports
pykurucz.

Deterministic configuration (matches the lecture's claims):
  * grey cold start, Teff=5770, logg=4.44  (atlas_py.physics.grey_start)
  * run_atlas to CONVERGENCE  (Fortran checkconv.f90 deep-layer dT/T metric,
    dlntmax=1e-4 Fortran-faithful, min_iterations=5, 1 consecutive pass)
  * CONTINUUM ONLY  -> IFOP(15)=0, IFOP(14)=0, IFOP(17)=0  (no 5 GB line data)
  * CONVECTION ON   -> enable_convection=True (IFCONV=1), mixlth=1.25, overwt=0,
                       nconv=36 (top 36 layers forced non-convective)
  * SERIAL          -> n_workers=1, ATLAS_CONVEC_FD_PARALLEL=0  (bit-exact)

What we capture:
  CONVERGED model     : T, RHOX, P, XNE, ABROSS, TAUROS, FLXCNV  (final atmosphere)
  CONVERGENCE history : checkconv_dlnt (max|dT/T| over deep layers) per iteration,
                        and dTmax per iteration; number of iterations to converge.
  FIXED-POINT replay  : starting FROM the converged model, ONE single-iteration
                        run (continuum-only, convection on) captures every input
                        the from-scratch verifier needs:
      per-frequency continuum : freq_hz, rco, acont, sigmac, scont  (KAPP outputs,
                                treated as GIVEN, like Lecture 10)
      CONVEC FD samples       : edens1..4, rho1..4  (the EOS finite-difference
                                samples from re-running POPS at T,P +-0.1% --
                                treated as GIVEN so the verifier need not re-run
                                the full EOS), plus the convec scalar inputs
                                ptotal, rho, pradk, prad, abross, vturb
      CONVEC outputs (ground) : flxcnv, grdadb, dltdlp, hscale, dlrdlt, heatcp,
                                vconv, height  (the new physics to benchmark)
      ROSS/TCORR accumulators : flxrad, rjmins, rdabh, rdiagj
      result of the one step  : T_out, RHOX_out  (== converged model, fixed point)
      JOSH H operator         : josh_coefh (Lecture 8 ships only ch/coefj)

The continuum opacity arrays are stored float32 to keep the file small (JOSH
casts to float32 internally; this is the precision bottleneck anyway).
"""
from __future__ import annotations

import os
# Pin every thread pool to 1 BEFORE any numba-backed import (determinism).
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["ATLAS_CONVEC_FD_PARALLEL"] = "0"     # serial CONVEC FD -> bit-exact
os.environ["ATLAS_CHECKCONV_DLNTMAX"] = "1e-4"   # Fortran-faithful convergence

import logging
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/ysting/pykurucz")  # read-only reference code
BOOK = Path(__file__).resolve().parent.parent
REF = BOOK / "reference"
REF.mkdir(exist_ok=True)

TEFF = 5770.0
LOGG = 4.44


class _HistoryHandler(logging.Handler):
    """Capture the driver's per-iteration convergence diagnostics from INFO logs."""

    _ITER = re.compile(r"ATLAS iteration (\d+)/\d+:.*dTmax=([0-9.eE+-]+)")
    _CONV = re.compile(r"Convergence iteration (\d+)/\d+:.*checkconv_dlnt=([0-9.eEnaN+-]+)")

    def __init__(self) -> None:
        super().__init__()
        self.dtmax: dict[int, float] = {}
        self.dlnt: dict[int, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        m = self._ITER.search(msg)
        if m:
            self.dtmax[int(m.group(1))] = float(m.group(2))
        m = self._CONV.search(msg)
        if m:
            try:
                self.dlnt[int(m.group(1))] = float(m.group(2))
            except ValueError:
                self.dlnt[int(m.group(1))] = float("nan")


def _write_grey_continuum_atm(dest: Path) -> None:
    """Write a grey starting .atm with continuum-only IFOP flags."""
    import pykurucz
    from atlas_py.physics.grey_start import generate_grey_atm

    g = generate_grey_atm(teff=TEFF, logg=LOGG)
    deck = np.column_stack([
        g.rhox, g.temperature, g.p_gas, g.xne, g.abross,
        g.accrad, g.vturb, g.flxcnv, g.vconv,
    ])
    pykurucz.write_atm_file(
        dest, TEFF, LOGG, deck, 0.0, mh=0.0, am=0.0,
        title="grey continuum-only start (Stellar Spectroscopy L11)",
    )
    cont_ifop = " OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0"
    lines = dest.read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("OPACITY IFOP"):
            lines[i] = cont_ifop
            break
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_converge(atm_path: Path, out_atm: Path, dbg_path: Path):
    """Run the full grey->converged loop with convection on; return debug state."""
    from atlas_py.config import AtlasConfig, AtlasInput, AtlasOutput
    from atlas_py.engine.driver import run_atlas

    cfg = AtlasConfig(
        inputs=AtlasInput(atmosphere_path=atm_path),
        outputs=AtlasOutput(output_atm_path=out_atm, debug_state_path=dbg_path),
        iterations=40,                 # generous cap; loop early-stops on convergence
        enable_molecules=False,
        enable_convection=True,        # IFCONV = 1
        enable_scattering=True,
        n_workers=1,                   # serial -> bit-exact
        convergence_epsilon=1.0e-12,   # any >0 enables the early-stop gate
        convergence_min_iterations=5,
        convergence_consecutive=1,
        convergence_dlntmax=1.0e-4,    # Fortran-faithful (also via env above)
    )
    run_atlas(cfg)
    return dict(np.load(dbg_path))


def _replay_one_iteration(conv_atm_path: Path):
    """Single iteration FROM the converged model: capture every verifier input.

    Mirrors the driver's iteration-1 setup (EOS from the input model, POPS,
    DOPPLER), then the continuum opacity, the ROSS/JOSH frequency loop, the
    CONVEC FD samples + CONVEC kernel, and the resulting accumulators.
    """
    from atlas_py.io.atmosphere import load_atm
    from atlas_py.engine.driver import (
        _runtime_from_atm, _build_kapp_adapter, _parse_ifop_flags,
        _gravity_from_atm_metadata, _convec_fd_samples,
        _parse_pradk0_from_metadata,
    )
    from atlas_py.physics.atlas_tables import load_atlas_tables
    from atlas_py.physics.kapcont import kapcont_baseline
    from atlas_py.physics.populations import pops
    from atlas_py.physics.popsall import popsall
    from atlas_py.physics.doppler import update_doppler_populations
    from atlas_py.physics.tcorr import init_tcorr, rosstab_ingest
    from atlas_py.physics.ross import ross_step
    from atlas_py.physics.josh import josh_depth_profiles
    from atlas_py.physics.radiap import init_radiap, radiap_accumulate
    from atlas_py.physics.convec import convec, high_from_rhox

    atm = load_atm(conv_atm_path)
    state = _runtime_from_atm(atm)
    ifop = _parse_ifop_flags(atm)
    gravity_cgs = _gravity_from_atm_metadata(atm)
    tables = load_atlas_tables()
    n = atm.layers
    tcst = init_tcorr(n)
    itemp = 1
    itemp_cache = {"pops_itemp": -1}

    # iteration-1 EOS setup (driver lines ~994-1036).
    tk_erg = atm.tk
    state.chargesq[:] = 2.0 * state.xne
    excess = 2.0 * state.xne - state.p / np.maximum(tk_erg, 1e-300)
    state.chargesq[excess > 0] += 2.0 * excess[excess > 0]
    dummy = np.zeros((n, 1))
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

    T0 = np.asarray(atm.temperature, dtype=np.float64)
    rhox0 = np.asarray(atm.rhox, dtype=np.float64)
    hkt0 = 6.6256e-27 / np.maximum(T0 * 1.38054e-16, 1e-300)
    nf0 = freq_all.size
    zc = np.zeros(n)
    flux = 5.6697e-5 / 12.5664 * TEFF ** 4

    # ROSS + RADIAP + TCORR accumulators over the frequency loop.
    ab_acc = np.zeros(n)
    radst = init_radiap(n)
    from atlas_py.physics.tcorr import tcorr_step
    ross_step(ab_acc, mode=1, rcowt=0.0, bnu=zc, freq_hz=0.0, hkt=hkt0,
              temperature_k=T0, stim=np.ones(n), abtot=np.ones(n), numnu=nf0, rhox=rhox0)
    radiap_accumulate(radst, mode=1, rcowt=0.0, abtot=np.ones(n), hnu=zc, jnu=zc,
                      knu_surface=0.0, flux=flux, rhox=rhox0)
    tcorr_step(tcst, mode=1, rcowt=0.0, rhox=rhox0, abtot=np.ones(n), hnu=zc,
               jmins=zc, taunu=zc, bnu=zc, freq_hz=0.0, hkt=hkt0, temperature_k=T0,
               stim=np.ones(n), alpha=zc, flux=flux, teff=TEFF, numnu=nf0)

    for inu in range(nf0):
        f = float(freq_all[inu]); rcowt = float(rco[inu])
        ehvkt = np.exp(-f * hkt0); stim = np.maximum(1.0 - ehvkt, 1e-300)
        bnu = 1.47439e-2 * ((f / 1e15) ** 3) * ehvkt / stim
        jr = josh_depth_profiles(
            ifscat=1, ifsurf=0, acont=acont_all[:, inu], scont=scont_all[:, inu],
            aline=zc, sline=bnu, sigmac=sigmac_all[:, inu], sigmal=zc,
            rhox=rhox0, bnu=bnu, freq_hz=f, wave_nm=float(wave_nm[inu]))
        # Fortran safety clamp after JOSH (driver lines 1415-1418).
        if np.any(jr.hnu < 0.0):
            jr.hnu[:] = np.maximum(jr.hnu, 1e-99)
            jr.jnu[:] = np.maximum(jr.jnu, 1e-99)
            jr.snu[:] = np.maximum(jr.snu, 1e-99)
        radiap_accumulate(radst, mode=2, rcowt=rcowt, abtot=jr.abtot, hnu=jr.hnu,
                          jnu=jr.jnu, knu_surface=jr.knu_surface, freq_hz=f,
                          wave_nm=float(wave_nm[inu]), flux=flux, rhox=rhox0)
        tcorr_step(tcst, mode=2, rcowt=rcowt, rhox=rhox0, abtot=jr.abtot, hnu=jr.hnu,
                   jmins=jr.jmins, taunu=jr.taunu, bnu=bnu, freq_hz=f, hkt=hkt0,
                   temperature_k=T0, stim=stim, alpha=jr.alpha, flux=flux,
                   teff=TEFF, numnu=nf0)
        ross_step(ab_acc, mode=2, rcowt=rcowt, bnu=bnu, freq_hz=f, hkt=hkt0,
                  temperature_k=T0, stim=stim, abtot=jr.abtot, numnu=nf0, rhox=rhox0)

    abross_out, tauros_out = ross_step(
        ab_acc, mode=3, rcowt=0.0, bnu=zc, freq_hz=0.0, hkt=hkt0,
        temperature_k=T0, stim=np.ones(n), abtot=np.ones(n), numnu=nf0, rhox=rhox0)
    radiap_accumulate(radst, mode=3, rcowt=0.0, abtot=np.ones(n), hnu=zc, jnu=zc,
                      knu_surface=0.0, flux=flux, rhox=rhox0)
    prad_out = radst.prad.copy()
    flxrad_out = tcst.flxrad.copy(); rjmins_out = tcst.rjmins.copy()
    rdabh_out = tcst.rdabh.copy(); rdiagj_out = tcst.rdiagj.copy()
    rosstab_ingest(tcst, T0, state.p, abross_out)
    state.height = high_from_rhox(rhox=rhox0, rho=state.rho)

    # PTOTAL exactly as the driver builds it for CONVEC (driver lines 1520-1522).
    # The driver uses pradk0_prev = the PREVIOUS iteration's surface K-integral; for
    # iteration 1 it seeds that from the input .atm's PRADK line.  At the converged
    # fixed point this equals the current RADIAP pradk0; we use the .atm PRADK seed
    # to match the driver's iteration-1 path exactly.
    pradk0_prev = _parse_pradk0_from_metadata(atm)
    pcon = 0.0; pturb0 = 0.0
    pzero = pcon + float(pradk0_prev) + pturb0
    ptotal = gravity_cgs * rhox0 + pzero

    # CONVEC FD samples (re-run POPS at T,P +-0.1%); serial path.
    ed1, ed2, ed3, ed4, r1, r2, r3, r4 = _convec_fd_samples(
        atm=atm, state=state, pradk=radst.pradk, tauros=tauros_out,
        ifmol=False, itemp_seed=itemp * 10, itemp_cache=itemp_cache)

    cv = convec(
        tcst=tcst, rhox=rhox0, tauros=tauros_out, temperature_k=T0,
        gas_pressure=state.p, mass_density=state.rho, abross=abross_out,
        vturb=atm.vturb, pradk=radst.pradk, ptotal=ptotal,
        gravity_cgs=gravity_cgs, flux=flux, mixlth=1.25, overwt=0.0,
        ifconv=1, nconv=36,
        edens1=ed1, edens2=ed2, edens3=ed3, edens4=ed4,
        rho1=r1, rho2=r2, rho3=r3, rho4=r4)

    # TCORR mode 3 WITH convection -> corrected T, RHOX (== converged, fixed point).
    tcres = tcorr_step(
        tcst, mode=3, rcowt=0.0, rhox=rhox0, abtot=np.ones(n), hnu=zc, jmins=zc,
        taunu=zc, bnu=zc, freq_hz=0.0, hkt=hkt0, temperature_k=T0,
        stim=np.ones(n), alpha=zc, flux=flux, teff=TEFF, numnu=nf0,
        tauros=tauros_out, abross=abross_out, iter_index=1, ifconv=1,
        flxcnv=cv.flxcnv, flxcnv0=cv.flxcnv0, dltdlp=cv.dltdlp, grdadb=cv.grdadb,
        hscale=cv.hscale, ptotal=ptotal, rho=state.rho, dlrdlt=cv.dlrdlt,
        heatcp=cv.heatcp, mixlth=1.25,
        prad=prad_out, pturb=np.zeros(n), gravity_cgs=gravity_cgs,
        steplg=0.125, tau1lg=-6.875)

    # Close the iteration: remap (T+T1, RHOX+DRHOX) onto TAUSTD (driver lines 1651-74).
    from atlas_py.physics.josh_math import _map1 as josh_map1
    taustd = 10.0 ** (-6.875 + np.arange(n) * 0.125)
    T_step, _ = josh_map1(tauros_out, tcres.temperature, taustd)
    rhox_step, _ = josh_map1(tauros_out, tcres.rhox, taustd)

    from atlas_py.physics.josh_tables_atlas12 import load_josh_tables
    josh_coefh = np.asarray(load_josh_tables()["coefh_matrix"], dtype=np.float64)

    return dict(
        gravity_cgs=np.float64(gravity_cgs),
        # converged-model EOS state (the fixed-point INPUT)
        T_conv=T0, rhox_conv=rhox0, p_conv=np.asarray(state.p, dtype=np.float64),
        xne_conv=np.asarray(state.xne, dtype=np.float64),
        rho_conv=np.asarray(state.rho, dtype=np.float64),
        tauros_conv=tauros_out, abross_conv=abross_out, prad_conv=prad_out,
        vturb_conv=np.asarray(atm.vturb, dtype=np.float64),
        ptotal_conv=ptotal, pradk_conv=np.asarray(radst.pradk, dtype=np.float64),
        # per-frequency continuum opacity (KAPP outputs; GIVEN)
        freq_hz=np.asarray(freq_all, dtype=np.float64),
        rco=np.asarray(rco, dtype=np.float64),
        acont=np.asarray(acont_all, dtype=np.float32),
        sigmac=np.asarray(sigmac_all, dtype=np.float32),
        scont=np.asarray(scont_all, dtype=np.float32),
        # CONVEC FD EOS samples (GIVEN)
        edens1=ed1, edens2=ed2, edens3=ed3, edens4=ed4,
        rho1=r1, rho2=r2, rho3=r3, rho4=r4,
        # CONVEC outputs (ground truth to benchmark)
        flxcnv_ref=cv.flxcnv, flxcnv0_ref=cv.flxcnv0, grdadb_ref=cv.grdadb,
        dltdlp_ref=cv.dltdlp, hscale_ref=cv.hscale, dlrdlt_ref=cv.dlrdlt,
        heatcp_ref=cv.heatcp, vconv_ref=cv.vconv, height_ref=cv.height,
        # ROSS/TCORR accumulators
        flxrad_ref=flxrad_out, rjmins_ref=rjmins_out,
        rdabh_ref=rdabh_out, rdiagj_ref=rdiagj_out,
        # TCORR mode-3 outputs (with convection)
        t1_ref=tcres.t1, dtflux_ref=tcres.dtflux, dtlamb_ref=tcres.dtlamb,
        flxerr_ref=tcres.flxerr, cnvflx_ref=tcres.cnvflx,
        # result of ONE step from the converged model (fixed-point check target)
        T_step=T_step, rhox_step=rhox_step,
        josh_coefh=josh_coefh,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    hist = _HistoryHandler()
    logging.getLogger("atlas_py.engine.driver").addHandler(hist)
    logging.getLogger("atlas_py.engine.driver").setLevel(logging.INFO)

    grey_path = REF / "_conv_grey.atm"
    out_atm = REF / "_conv_out.atm"
    dbg_path = REF / "_conv_debug.npz"
    _write_grey_continuum_atm(grey_path)

    # --- full grey -> converged loop ---
    dbg = _run_converge(grey_path, out_atm, dbg_path)
    iters = sorted(hist.dlnt)
    dlnt_hist = np.array([hist.dlnt[i] for i in iters], dtype=np.float64)
    dtmax_hist = np.array([hist.dtmax.get(i, np.nan) for i in iters], dtype=np.float64)
    n_iter = len(iters)
    print(f"\nConverged in {n_iter} iterations.")
    print("  iter  dTmax       checkconv_dlnt")
    for i, dt, dl in zip(iters, dtmax_hist, dlnt_hist):
        print(f"  {i:3d}   {dt:.4e}   {dl:.4e}")

    # --- determinism: a second full run must match bit-for-bit ---
    hist2 = _HistoryHandler()
    logging.getLogger("atlas_py.engine.driver").addHandler(hist2)
    dbg2 = _run_converge(grey_path, REF / "_conv_out2.atm", REF / "_conv_debug2.npz")
    det_T = np.array_equal(dbg["temperature"], dbg2["temperature"])
    det_R = np.array_equal(dbg["rhox"], dbg2["rhox"])
    print(f"\nDeterminism: T identical={det_T}  RHOX identical={det_R}")

    # write converged model to a continuum-only .atm for the fixed-point replay
    import pykurucz
    conv_atm = REF / "_conv_converged.atm"
    deck = np.column_stack([
        dbg["rhox"], dbg["temperature"], dbg["p"], dbg["xne"], dbg["abross_out"],
        dbg["accrad_out"], dbg.get("vturb", np.zeros_like(dbg["rhox"])),
        dbg.get("flxcnv_out", np.zeros_like(dbg["rhox"])),
        dbg.get("vconv_out", np.zeros_like(dbg["rhox"])),
    ])
    pykurucz.write_atm_file(
        conv_atm, TEFF, LOGG, deck, 0.0, mh=0.0, am=0.0,
        title="converged continuum-only model (Stellar Spectroscopy L11)",
    )
    cont_ifop = " OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0"
    lines = conv_atm.read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("OPACITY IFOP"):
            lines[i] = cont_ifop
            break
    conv_atm.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- fixed-point replay ---
    rep = _replay_one_iteration(conv_atm)

    # converged-model arrays for the verifier's top-level benchmark
    rep.update(
        teff=np.float64(TEFF), logg=np.float64(LOGG),
        tau1lg=np.float64(-6.875), steplg=np.float64(0.125),
        n_iterations=np.int32(n_iter),
        dlnt_history=dlnt_hist, dtmax_history=dtmax_hist,
        # the FINAL converged model (== T_conv/rhox_conv, kept under canonical names)
        T_converged=dbg["temperature"], rhox_converged=dbg["rhox"],
        p_converged=dbg["p"], xne_converged=dbg["xne"],
        abross_converged=dbg["abross_out"],
        flxcnv_converged=dbg.get("flxcnv_out", np.zeros_like(dbg["rhox"])),
        determinism_T=np.int32(1 if det_T else 0),
        determinism_R=np.int32(1 if det_R else 0),
    )

    out = REF / "converged_ref.npz"
    np.savez_compressed(out, **rep)
    print(f"\nwrote {out}")
    d = np.load(out)
    print("arrays:", ", ".join(sorted(d.files)))
    print(f"  n_freq={rep['freq_hz'].size}  n_layers={rep['T_conv'].size}")
    print(f"  file size = {out.stat().st_size / 1e6:.1f} MB")

    # fixed-point residual (sanity): one step from converged should reproduce it
    rT = np.max(np.abs(rep["T_step"] - dbg["temperature"]) / np.abs(dbg["temperature"]))
    rR = np.max(np.abs(rep["rhox_step"] - dbg["rhox"]) / np.maximum(np.abs(dbg["rhox"]), 1e-300))
    print(f"  fixed-point (pykurucz self) T rel={rT:.3e}  RHOX rel={rR:.3e}")

    for p in (grey_path, out_atm, dbg_path, REF / "_conv_out2.atm",
              REF / "_conv_debug2.npz", conv_atm):
        p.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
