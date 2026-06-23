#!/usr/bin/env python
"""Generate the ground-truth reference atmosphere + spectrum with pykurucz.

This runs pykurucz's OWN pipeline so the reference is genuinely its output:
  1. emulator_warmstart_atm  -> reference/sun.atm   (kurucz-a1 warm start)
  2. run_synthe_py           -> reference/sun_<win>.spec  (SYNTHE: opacity + RT)

pykurucz is imported READ-ONLY from PYKURUCZ_ROOT and never modified. The npz
cache is disabled so nothing is written into the pykurucz tree; all outputs go
under this book's reference/ directory.

Usage:
  python _pipeline/make_reference.py [--teff 5770] [--logg 4.44]
                                     [--wl-start 500] [--wl-end 510] [--R 300000]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

PYKURUCZ_ROOT = Path("/Users/ysting/pykurucz")
BOOK = Path(__file__).resolve().parent.parent
REF = BOOK / "reference"
REF.mkdir(exist_ok=True)

sys.path.insert(0, str(PYKURUCZ_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teff", type=float, default=5770.0)
    ap.add_argument("--logg", type=float, default=4.44)
    ap.add_argument("--mh", type=float, default=0.0)
    ap.add_argument("--am", type=float, default=0.0)
    ap.add_argument("--vturb", type=float, default=2.0)
    ap.add_argument("--wl-start", type=float, default=500.0)
    ap.add_argument("--wl-end", type=float, default=510.0)
    ap.add_argument("--R", type=float, default=300_000.0)
    ap.add_argument("--molecules", action="store_true",
                    help="include molecular lines (needs the ~5 GB data download)")
    args = ap.parse_args()

    import pykurucz

    tag = f"{int(args.teff)}_{args.logg:.2f}_{int(args.wl_start)}_{int(args.wl_end)}"
    atm = REF / "sun.atm"
    npz = REF / f"sun_{tag}.npz"
    spec = REF / f"sun_{tag}.spec"
    log = REF / f"sun_{tag}.log"

    print(f"[1/2] emulator warm-start -> {atm.name}", flush=True)
    t0 = time.perf_counter()
    pykurucz.emulator_warmstart_atm(
        atm, teff=args.teff, logg=args.logg, mh=args.mh, am=args.am, vturb=args.vturb,
    )
    print(f"      done in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"[2/2] SYNTHE  {args.wl_start}-{args.wl_end} nm  R={args.R:.0f}"
          f"  molecules={'on' if args.molecules else 'off'} -> {spec.name}", flush=True)
    t0 = time.perf_counter()
    pykurucz.run_synthe_py(
        atm,
        spec=spec, npz=npz, log_path=log,
        wl_start=args.wl_start, wl_end=args.wl_end, resolution=args.R,
        use_molecular_lines=args.molecules,
        npz_cache=False,            # do not write into the pykurucz tree
    )
    print(f"      done in {time.perf_counter()-t0:.1f}s", flush=True)

    # quick sanity peek
    import numpy as np
    data = np.loadtxt(spec)
    print(f"\nspec shape {data.shape}  cols=(wavelength, flux, continuum?)")
    print(f"wavelength range: {data[0,0]:.4f} .. {data[-1,0]:.4f} nm   ({len(data)} points)")
    print(f"flux  min/max: {data[:,1].min():.4e} / {data[:,1].max():.4e}")
    print(f"\nReference written:\n  {atm}\n  {npz}\n  {spec}")


if __name__ == "__main__":
    main()
