#!/usr/bin/env python
"""Generate read-only ground-truth reference for verify_convec_gaps.py.

Run ONCE against the read-only pykurucz tree.  Produces TWO npz files in the
book's reference/ dir:

  convec_gaps_inputs.npz   -- INPUTS ONLY (NO answer arrays): the exact state
                              fed to CONVEC FD + CONVEC, plus the atomic DATA
                              tables (NNN/POTION/PFTAB/LOCZ/SCALE) the Saha
                              engine reuses (these are atomic data, GIVEN).
  convec_gaps_truth.npz    -- GROUND TRUTH for comparison only:
                              GAP B: ed1..4, r1..4 (pykurucz _convec_fd_samples)
                              GAP A: flxcnv overshoot ON / OFF, flxcnv0/1, etc.

This script imports pykurucz; the verifier never does.  The verifier loads the
inputs npz, COMPUTES r1..4 (NELECT Saha solve) and the overshoot blend from
scratch, and compares to the truth npz.
"""
from __future__ import annotations

import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["ATLAS_CONVEC_FD_PARALLEL"] = "0"      # serial FD -> bit-exact
os.environ["ATLAS_POPS_PARALLEL"] = "0"
os.environ["ATLAS_CHECKCONV_DLNTMAX"] = "1e-4"

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/Users/ysting/pykurucz")
BOOK = Path("/Users/ysting/Stellar_Spectroscopy_From_Scratch")
REF = BOOK / "reference"
REF.mkdir(exist_ok=True)

TEFF = 5770.0
LOGG = 4.44


def _write_grey_continuum_atm(dest: Path) -> None:
    import pykurucz
    from atlas_py.physics.grey_start import generate_grey_atm
    g = generate_grey_atm(teff=TEFF, logg=LOGG)
    deck = np.column_stack([
        g.rhox, g.temperature, g.p_gas, g.xne, g.abross,
        g.accrad, g.vturb, g.flxcnv, g.vconv,
    ])
    pykurucz.write_atm_file(dest, TEFF, LOGG, deck, 0.0, mh=0.0, am=0.0,
                            title="grey continuum-only start (convec gaps ref)")
    cont_ifop = " OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0"
    lines = dest.read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("OPACITY IFOP"):
            lines[i] = cont_ifop
            break
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_converge(atm_path: Path, out_atm: Path, dbg_path: Path):
    from atlas_py.config import AtlasConfig, AtlasInput, AtlasOutput
    from atlas_py.engine.driver import run_atlas
    cfg = AtlasConfig(
        inputs=AtlasInput(atmosphere_path=atm_path),
        outputs=AtlasOutput(output_atm_path=out_atm, debug_state_path=dbg_path),
        iterations=40, enable_molecules=False, enable_convection=True,
        enable_scattering=True, n_workers=1,
        convergence_epsilon=1.0e-12, convergence_min_iterations=5,
        convergence_consecutive=1, convergence_dlntmax=1.0e-4,
    )
    run_atlas(cfg)
    return dict(np.load(dbg_path))


def main() -> int:
    import logging
    logging.basicConfig(level=logging.WARNING)
    import pykurucz
    grey_path = REF / "_cg_grey.atm"
    out_atm = REF / "_cg_out.atm"
    dbg_path = REF / "_cg_debug.npz"
    _write_grey_continuum_atm(grey_path)
    dbg = _run_converge(grey_path, out_atm, dbg_path)

    # write the converged model to a continuum-only .atm for the replay
    conv_atm = REF / "_cg_converged.atm"
    deck = np.column_stack([
        dbg["rhox"], dbg["temperature"], dbg["p"], dbg["xne"], dbg["abross_out"],
        dbg["accrad_out"], dbg.get("vturb", np.zeros_like(dbg["rhox"])),
        dbg.get("flxcnv_out", np.zeros_like(dbg["rhox"])),
        dbg.get("vconv_out", np.zeros_like(dbg["rhox"])),
    ])
    pykurucz.write_atm_file(conv_atm, TEFF, LOGG, deck, 0.0, mh=0.0, am=0.0,
                            title="converged continuum-only model (convec gaps ref)")
    cont_ifop = " OPACITY IFOP 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0"
    lines = conv_atm.read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("OPACITY IFOP"):
            lines[i] = cont_ifop
            break
    conv_atm.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- replay one iteration from the converged model up to the convec FD ----
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

    atm = load_atm(conv_atm)
    state = _runtime_from_atm(atm)
    ifop = _parse_ifop_flags(atm)
    gravity_cgs = _gravity_from_atm_metadata(atm)
    tables = load_atlas_tables()
    n = atm.layers
    tcst = init_tcorr(n)
    itemp = 1
    itemp_cache = {"pops_itemp": -1}

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
        adapter=adapter, teff=TEFF, atlas_tables=tables, ifop=ifop, tcst=tcst)
    freq_all = 2.99792458e17 / np.maximum(wave_nm, 1e-300)

    T0 = np.asarray(atm.temperature, dtype=np.float64)
    rhox0 = np.asarray(atm.rhox, dtype=np.float64)
    hkt0 = 6.6256e-27 / np.maximum(T0 * 1.38054e-16, 1e-300)
    nf0 = freq_all.size
    zc = np.zeros(n)
    flux = 5.6697e-5 / 12.5664 * TEFF ** 4

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
    rosstab_ingest(tcst, T0, state.p, abross_out)
    state.height = high_from_rhox(rhox=rhox0, rho=state.rho)

    pradk0_prev = _parse_pradk0_from_metadata(atm)
    pzero = 0.0 + float(pradk0_prev) + 0.0
    ptotal = gravity_cgs * rhox0 + pzero

    # ---- capture the EXACT INPUT STATE to convec FD (this is what the verifier
    #      will re-run NELECT from) BEFORE _convec_fd_samples mutates it ----
    in_T = T0.copy()
    in_P = np.asarray(state.p, dtype=np.float64).copy()
    in_xne = np.asarray(state.xne, dtype=np.float64).copy()
    in_xnatom = np.asarray(state.xnatom, dtype=np.float64).copy()
    in_rho = np.asarray(state.rho, dtype=np.float64).copy()
    in_chargesq = np.asarray(state.chargesq, dtype=np.float64).copy()
    in_wtmole = np.asarray(state.wtmole, dtype=np.float64).copy()
    in_xabund = np.asarray(state.xabund, dtype=np.float64).copy()  # (n,99)
    in_pradk = np.asarray(radst.pradk, dtype=np.float64).copy()
    in_tauros = np.asarray(tauros_out, dtype=np.float64).copy()
    in_tk = np.asarray(atm.tk, dtype=np.float64).copy()

    # ---- GAP B ground truth: pykurucz's serial convec FD samples ----
    ed1, ed2, ed3, ed4, r1, r2, r3, r4 = _convec_fd_samples(
        atm=atm, state=state, pradk=radst.pradk, tauros=tauros_out,
        ifmol=False, itemp_seed=itemp * 10, itemp_cache=itemp_cache)

    # ---- GAP A ground truth: CONVEC with overshoot ON (overwt>0) and OFF ----
    common = dict(
        tcst=tcst, rhox=rhox0, tauros=tauros_out, temperature_k=T0,
        gas_pressure=state.p, mass_density=state.rho, abross=abross_out,
        vturb=atm.vturb, pradk=radst.pradk, ptotal=ptotal,
        gravity_cgs=gravity_cgs, flux=flux, mixlth=1.25, ifconv=1, nconv=36,
        edens1=ed1, edens2=ed2, edens3=ed3, edens4=ed4,
        rho1=r1, rho2=r2, rho3=r3, rho4=r4)
    cv_off = convec(overwt=0.0, **common)
    cv_on = convec(overwt=1.0, **common)
    cv_on2 = convec(overwt=2.0, **common)  # second overwt to prove the blend formula

    # atomic data tables (GIVEN -> shipped in inputs)
    from atlas_py.physics.pfsaha import NNN, POTION, LOCZ, SCALE
    from atlas_py.physics.pfground import _get_pftab
    pftab = np.ascontiguousarray(_get_pftab(), dtype=np.float64)

    # special-partition level energies [cm^-1] and statistical weights g (ATLAS DATA
    # blocks): the light elements ATLAS sums level-by-level.  These are atomic DATA,
    # shipped in the inputs file so the Lecture-11 notebook stays self-contained.
    # Imported from the verifier (the single source of these literal constants).
    import importlib.util
    _vspec = importlib.util.spec_from_file_location(
        "verify_convec_gaps", Path(__file__).resolve().parent / "verify_convec_gaps.py")
    _vmod = importlib.util.module_from_spec(_vspec)
    _vspec.loader.exec_module(_vmod)
    _eg_names = ["EHYD", "GHYD", "EHE1", "GHE1", "EHE2", "GHE2", "EC1", "GC1",
                 "EC2", "GC2", "EMG1", "GMG1", "EMG2", "GMG2", "EAL1", "GAL1",
                 "ESI1", "GSI1", "ESI2", "GSI2", "ENA1", "GNA1", "EO1", "GO1",
                 "EB1", "GB1", "EK1", "GK1"]
    eg_tables = {nm: np.asarray(getattr(_vmod, nm), dtype=np.float64) for nm in _eg_names}

    # ================= write INPUTS (no answers) =================
    inputs = dict(
        teff=np.float64(TEFF), logg=np.float64(LOGG),
        gravity_cgs=np.float64(gravity_cgs),
        flux=np.float64(flux),
        mixlth=np.float64(1.25), nconv=np.int32(36),
        # convec-FD input state (the NELECT seeds)
        T=in_T, P=in_P, xne=in_xne, xnatom=in_xnatom, rho=in_rho,
        chargesq=in_chargesq, wtmole=in_wtmole, xabund=in_xabund,
        tk=in_tk, pradk=in_pradk, tauros=in_tauros,
        # convec scalar inputs (needed for GAP A overshoot height integral)
        rhox=rhox0, ptotal=ptotal, abross=abross_out,
        # GAP A blend INPUT: the pre-overshoot convective flux FLXCNV0 (same for
        # every OVERWT; reproduced bit-exactly by verify_converged.py's CONVEC
        # kernel).  The blend EXTENDS this; FLXCNV0 is an input, not the answer.
        flxcnv0=cv_off.flxcnv0,
        # atomic DATA tables (partition/ionization), GIVEN
        NNN=np.asarray(NNN, dtype=np.int64),
        POTION=np.asarray(POTION, dtype=np.float64),
        LOCZ=np.asarray(LOCZ, dtype=np.int64),
        SCALE=np.asarray(SCALE, dtype=np.float64),
        PFTAB=pftab,
        # special-partition level energies/weights (atomic DATA, light elements)
        **eg_tables,
    )
    out_in = REF / "convec_gaps_inputs.npz"
    np.savez_compressed(out_in, **inputs)

    # ================= write TRUTH (answers, comparison only) =================
    truth = dict(
        # GAP B
        edens1=ed1, edens2=ed2, edens3=ed3, edens4=ed4,
        rho1=r1, rho2=r2, rho3=r3, rho4=r4,
        # GAP A
        flxcnv_off=cv_off.flxcnv, flxcnv0_off=cv_off.flxcnv0,
        flxcnv1_off=cv_off.flxcnv1,
        flxcnv_on=cv_on.flxcnv, flxcnv0_on=cv_on.flxcnv0,
        flxcnv1_on=cv_on.flxcnv1,
        flxcnv_on2=cv_on2.flxcnv, flxcnv1_on2=cv_on2.flxcnv1,
        height=cv_on.height,
    )
    out_tr = REF / "convec_gaps_truth.npz"
    np.savez_compressed(out_tr, **truth)

    print(f"wrote {out_in}  ({out_in.stat().st_size/1e3:.0f} KB)")
    print(f"wrote {out_tr}  ({out_tr.stat().st_size/1e3:.0f} KB)")
    print("inputs:", ", ".join(sorted(inputs)))
    print("truth :", ", ".join(sorted(truth)))
    print(f"n_layers={n}")
    # quick overshoot sanity: where does overwt change flxcnv?
    diff = np.abs(cv_on.flxcnv - cv_off.flxcnv)
    print(f"overshoot ON vs OFF: max|dflxcnv|={diff.max():.3e}  "
          f"nlayers changed={np.sum(diff>0)}")
    print(f"flxcnv1_on nonzero layers: {np.sum(cv_on.flxcnv1>0)}")

    for p in (grey_path, out_atm, dbg_path, conv_atm):
        p.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
