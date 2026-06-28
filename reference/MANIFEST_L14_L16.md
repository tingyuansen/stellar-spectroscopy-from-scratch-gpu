# L14-L16 Reference Bundle Manifest

This file classifies the arrays used by the line-blanketed capstone lectures so future edits do
not mistake a comparison target for computed state.

## Lecture 14: `leankurucz_<star>.npz`

Used by `_pipeline/build_lecture14.py` and checked by `_pipeline/verify_leankurucz.py`.

Physical input / static data:
- `atm_*`, `temperature`, `mass_density`, `electron_density`, `depth`, `gas_pressure`,
  `turbulent_velocity`, `teff`, `logg`, `xabund`: the atmosphere structure and scalar stellar
  parameters consumed by the synthesis half.
- `cat_*`, `f19_*`, `mol_*`, `molc_*`, `has_*`, `wavelength`: line lists, molecule lists,
  continuum-identification metadata, feature flags, and the wavelength grid.
- `frqedg`, `wledge`, `half_edge`, `delta_edge`, `freqset`, `bhyd`: continuum-edge and hydrogen
  table grids used by the in-notebook opacity engines.

Computed state currently loaded by L14:
- `population_per_ion`, `doppler_per_ion`, `xnf*`, `hckt`, `hkt`, `tkev`, `tk`, `tlog`, `xnatm`.
  These are EOS/population/Doppler inputs to the opacity engines. L14 consumes them; it does not
  yet recompute them from `(T, P, abundances)` inside the capstone.
- Solar-only `op_*` arrays are the grey-reference atmosphere-operator fixture. L14 computes the
  JOSH moments, Rosseland fold, radiation pressure, hydrostatic correction, and TCORR operator, but
  it does not rebuild the full-frequency opacity for that grey intermediate in the capstone.

Comparison-only targets:
- `continuum_absorption`, `continuum_scattering`, `line_opacity`, `line_scattering`,
  `line_source`, `slinec`, `flux_total`, `flux_continuum`.
- The lecture may load these only after computing its own opacity/spectrum, to report residuals.

Atmosphere provenance:
- `atm_source`, `atm_converged_from_scratch`, `atm_n_iter`, `atm_final_dlnt`,
  `atm_dlnt_history`, `atm_dtmax_history`, `atm_reason`.
- Current state: the Sun uses the Part-VI line-blanketed solar state (`base RHOX = 12.1439331`,
  `base T = 11425 K` in the pyk exact-LINOP reference). Hot dwarf, giant, and M dwarf remain
  documented emulator warm-start structures.

## Lecture 15: `lineblanket_ref.npz`

Used by `_pipeline/build_lecture15_gpu.py` and `_pipeline/verify_lineblanket.py`.

Physical input / static data:
- Stellar and structure scalars: `teff`, `logg`, `gravity_cgs`, `T`, `rhox`, `P`, `rho`,
  `ptotal`, `xne`, `vturb`, `hckt`.
- Frequency grid and continuum inputs: `freq_hz`, `waveset_nm`, `rco`, `acont`, `sigmac`,
  `scont`.
- Teaching-window line selection: `win_*`, `tabcont`, `iwavetab`, `ratiolg`, scale factors,
  and line-type flags.

Computed state currently loaded by L15:
- `win_xnfdop`, `win_dopple`, `win_xne`, `win_txnxn`, `win_hckt`: the per-line/per-depth state
  consumed by the teaching-window deposit. L16 is the lecture that rebuilds this class of state.

Comparison-only targets:
- `xlines_window_ref`, `xlines_fullgrid`, `ahline_fullgrid`, `T_step`, `rhox_step`, `abross`,
  `tauros`, `abross_raw`, `tauros_raw`, `prad`, `pradk`, `flxrad`, `flxcnv`, `grdadb`, `dltdlp`.

## Lecture 16: `converge_fromscratch_result.npz`

Used by `_pipeline/build_lecture16_gpu.py` as a compact result statement.

Computed result:
- `T`, `rhox`, `Tmed`, `base_T`, `base_RHOX`: the independent from-scratch line-blanketed solar
  convergence result (`base_RHOX` is 12.3-class). It is not the pyk exact-LINOP target; the pyk
  solar state used by L14 has `base RHOX = 12.1439331`.

Comparison arrays:
- `Ts`, `Rs`: reference solar temperature and column-mass arrays used to state residuals.

## Editing Rule

When tightening L14 into a stricter capstone, move arrays from "computed state currently loaded" to
computed-in-notebook code by reusing or inlining the L15/L16 machinery. Do not move comparison-only
targets earlier in the computed path, and do not call a lecture complete merely because a target
array is available in a bundle.
