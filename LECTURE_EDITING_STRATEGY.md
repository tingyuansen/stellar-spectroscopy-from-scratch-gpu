# Lecture Editing Strategy

This file is a working companion for editing the textbook lecture by lecture. It is not a replacement for the project passdown, plan, or build notes. Its purpose is narrower: keep the pedagogical strategy, material layout, and known editing pitfalls visible so later lectures can be revised in the same style as Chapter 1.

## Global Editing Standard

Write each lecture as self-contained graduate lecture notes for a first-year astronomy graduate student. Assume the reader is capable, but may have forgotten the details behind familiar names such as optical depth, LTE, Rosseland mean, Kirchhoff's law, or Eddington approximation.

The lecture should flow as physics, not as a code walkthrough. Introduce the physical need first, then the definition, then the equation, then the compact code that evaluates it. Code cells should be small enough to read at once and should support the surrounding argument.

Use PyTorch as the calculation language because it is GPU-native, but avoid making device, dtype, or tensor mechanics the main story unless they are pedagogically necessary for the calculation being shown. NumPy parity checks belong in separate verifier scripts, not as rendered textbook content.

Prefer preservation over deletion. The default edit should keep the lecture's concepts and useful code, but change their order, motivation, naming, and explanation so they read as coherent lecture notes. Delete only material that is project plumbing, duplicate exposition, misleading, uncited, or better protected in a separate verifier. When something is removed from the rendered lecture, record whether it was removed because it is not a physics concept, because it belongs in a later lecture, or because it should be represented by a cleaner physical sanity check.

Avoid project-internal language in the textbook. Do not mention `kgpu`, `pykurucz`, production internals, or precomputed tests in the rendered lecture unless a later lecture explicitly needs to distinguish a method table from a solved physical result. Do not touch `pykurucz_gpu` while it is still being developed separately.

## Lecture Editing Loop

1. Read the current lecture as a student would read it, not as a maintainer would diff it.
2. Identify the physical spine: what question starts the lecture, what obstacle appears next, and what new concept resolves it.
3. Move definitions to the first place where the concept is needed. Define important terms once, then explicitly reuse them.
4. Replace long derivations with short derivations only when the derivation teaches the idea. If a derivation interrupts the flow, give the physical reason and state the result clearly.
5. Use short paragraph labels for new terminology when helpful, for example `**Rosseland optical depth.**`, `**Why this mean?**`, or `**Hydrostatic equilibrium on an optical-depth grid.**`
6. Check every equation for defined variables, units, sign conventions, and whether it uses frequency or wavelength consistently.
7. Check every code cell for one clear purpose, understandable variable names, GPU-native tensor operations where natural, and no unnecessary redefinition.
8. Rebuild the notebook and rendered HTML, run the separate verifier when one exists, refresh the local reader, and grep for project leakage.

## Chapter 1 Layout

Chapter 1 now follows this sequence:

1. **The forward problem.** A synthetic spectrum maps `(T_eff, log g, composition)` to an emergent surface flux `F_lambda`. The chapter separates a model atmosphere from spectral synthesis.
2. **The simplifying assumptions.** Static, plane-parallel, grey, and LTE are introduced in plain language before any equations depend on them.
3. **Numerical conventions.** PyTorch is introduced only as the calculation language; Astropy supplies constants before the tensor calculation begins.
4. **Units and constants.** CGS units, `h`, `c`, `k`, photon energy, and `kT` set the radiation scale.
5. **The Planck function.** Specific intensity, `B_nu`, `B_lambda`, the frequency-to-wavelength conversion, and Rayleigh-Jeans/Wien limits are laid out before the first Planck code.
6. **Optical depth and the photosphere.** Mass extinction coefficient, mean free path, `d tau_lambda = -kappa_lambda rho dz`, attenuation `exp(-tau)`, column mass, and `d tau_lambda = kappa_lambda dm` establish why photons escape from optical depth of order unity.
7. **LTE and the source function.** The source function is formally defined here, not earlier. Kirchhoff's law then gives `S_lambda = B_lambda(T)` for LTE thermal emission.
8. **The grey atmosphere.** Grey opacity, radiative equilibrium, bolometric intensity moments, Eddington closure, and the grey `T^4` law build the temperature structure.
9. **The Hopf surface correction.** The constant Eddington boundary `q = 2/3` motivates the more flexible `q(tau)` form. The exponential grey-start fit is described as a modeling choice, not a new law.
10. **Rosseland optical depth.** `tau_Ross` is explicitly presented as the same optical-depth construction as `d tau_lambda = kappa_lambda dm`, with `kappa_lambda` replaced by the harmonic, diffusion-weighted Rosseland mean.
11. **Hydrostatic equilibrium.** `dP_total/dm = g` is combined with the already-defined Rosseland relation to get `dP_total/d tau_Ross = g/kappa_Ross`.
12. **Unit-opacity normalization.** `kappa_Ross = 1 cm^2 g^-1` is explained as a first-model normalization that attaches a pressure scale to the grey temperature law, not as a solar opacity claim.

## Chapter 1 Fact Audit

The current Chapter 1 physics is internally consistent:

- `F = sigma T_eff^4` defines effective temperature as a surface flux scale.
- `log g` is base-10 surface gravity in CGS units.
- `B_nu` and `B_lambda` are related by `B_lambda |d lambda| = B_nu |d nu|`, so `B_lambda = B_nu c/lambda^2`.
- Optical depth increases inward with the chosen outward `z` coordinate: `d tau_lambda = -kappa_lambda rho dz`.
- Pure attenuation gives `I = I_0 exp(-tau)`.
- Column mass satisfies `dm = -rho dz`, giving `d tau_lambda = kappa_lambda dm`.
- In LTE with true absorption, Kirchhoff's law gives the thermal source function `S_lambda = B_lambda(T)`. Scattering is correctly identified as a separate source-term problem.
- The grey moment definitions give `F = 4 pi H`. With Eddington closure `K = J/3` and radiative equilibrium, the compact result `(1/3) dJ/dtau = H` leads to `dT^4/dtau = (3/4) T_eff^4`.
- The Eddington boundary constant `C = 2/3` gives `T(tau = 2/3) = T_eff`, while the broader photospheric statement should remain "optical depth of order unity."
- The Rosseland mean is correctly described as a harmonic opacity average weighted by `partial B_lambda / partial T` or equivalently by the frequency form if used consistently.
- Hydrostatic equilibrium with column mass gives `P_total = g m` up to the top-boundary convention, and combining it with Rosseland optical depth gives `dP_total/dtau_Ross = g/kappa_Ross`.
- Radiation pressure `P_rad = 4 sigma T^4 / (3c)` is the isotropic blackbody/diffusion form. The Chapter 1 subtraction relative to the top boundary is a structural approximation for the first pressure scale.

Known caveat: the exact exponential Hopf coefficients `0.710 - 0.1331 exp(-3.4488 tau)` should not be over-cited unless a real source is added. In the chapter, it is safest to describe this as the smooth ATLAS/Kurucz grey-start fit used by the reference calculation and to emphasize that it is a surface-boundary modeling choice.

## Chapter 1 Retention Audit

The current Chapter 1 keeps the main original concepts:

- The forward problem from stellar parameters to emergent flux.
- The distinction between a model atmosphere and spectral synthesis.
- The simplifying assumptions: static, plane-parallel, grey, and LTE.
- CGS units and the radiation constants `h`, `c`, `k`.
- The Planck function in an overflow-safe algebraic form.
- Frequency-wavelength conversion between `B_nu` and `B_lambda`.
- Rayleigh-Jeans and Wien limits.
- Optical depth, mean free path, extinction, and column mass.
- LTE and the thermal source function.
- Kirchhoff's law.
- Grey-atmosphere temperature structure.
- The Eddington closure and the origin of the grey `T^4` law.
- The Hopf-function surface correction and the Kurucz/ATLAS grey-start fit.
- The 80-layer logarithmic optical-depth grid.
- Rosseland optical depth and the Rosseland mean opacity.
- Hydrostatic equilibrium and the pressure-column-mass scale.
- Radiation-pressure correction.
- The first solar atmosphere table and plots for temperature, pressure, and column mass.
- Practice exercises about Wien's law, photospheric temperature scale, and surface gravity.

The current Chapter 1 deliberately removed or relocated some original material:

- Rendered parity checks against `reference/L1.npz` were moved out of the lecture narrative and kept in `_pipeline/verify_atmosphere.py`.
- `pykurucz`, `kgpu`, reference-oracle language, and precomputed-test language were removed from the rendered textbook.
- The detailed device/dtype precision-budget discussion was reduced to a quiet numerical convention because it was distracting from the first physics lecture.
- The full lecture roadmap was removed from the opening because it made the chapter feel like a syllabus rather than a lecture.
- `RHOX`, `XNE`, and `ABROSS` names were removed from Chapter 1's rendered lecture because they are implementation/table-column names. The concepts remain as column mass, missing electron-density solution, and not-yet-computed opacity.
- The explicit Eddington-Barbier name is no longer in Chapter 1. The physical idea remains in the optical-depth discussion as "photosphere around optical depth unity" and in the grey boundary result `T(tau=2/3)=T_eff`. If strict concept preservation is desired, add only a short, non-derivational note that the formal Eddington-Barbier relation will later make this optical-depth sampling precise.
- The stimulated-emission forward pointer was removed. The Planck occupation-factor algebra remains, while stimulated emission is better introduced where line opacity is actually built.
- Kurucz/ATLAS provenance was softened. This keeps Chapter 1 self-contained but means exact provenance for the Hopf coefficients should be handled carefully if a citation is later added.

For later lectures, use this distinction: keep physical concepts and useful executable code; remove or relocate project plumbing, validation scaffolding, and premature references. If a removed item is a real concept rather than plumbing, either keep it in a shorter form or record where it will be taught.

## Chapter 2 Layout

Chapter 2 now follows this sequence:

1. **The equation-of-state need.** The atmosphere has temperature and pressure, but opacity needs ion stages, level populations, and free electrons.
2. **Boltzmann populations.** Energy levels, statistical weights, and the partition function are defined before the first partition-function plot.
3. **Saha ionization.** The stage-to-stage ratio is introduced with all variables and units, then evaluated for hydrogen as an interpretive scale calculation.
4. **Pressure ionization.** Debye lowering is explained as plasma screening, not as a code prescription.
5. **Charge conservation.** The electron density is solved by iterating Saha fractions and donated charge until `n_e` closes.
6. **Electron donors.** The lecture reads the solution physically: metals dominate the line-forming electron supply, while hydrogen wins in different depth regimes.
7. **Completing the atmosphere.** The solved `n_e` fills the electron-density column and is checked by charge closure.
8. **PFSAHA.** The full per-ion population engine is introduced as "partition functions plus Saha populations": `U`, `F/U`, and `Phi = n_Z F/U`.
9. **Table dispatch.** PFIRON, NNN, explicit light-element level sums, ground-state floors, occupation corrections, and log-space Saha ladders are explained as atomic-data machinery.
10. **Reading PFSAHA.** The rendered lecture uses physical sanity checks: stage sums, dominant ion stages, and stored-stage electron closure. Strict parity is kept in `_pipeline/verify_pfsaha.py`.

## Chapter 2 Fact And Retention Notes

Chapter 2 keeps the main original concepts:

- LTE Boltzmann populations and partition functions.
- Saha ionization balance and the electron-density dependence.
- Debye continuum lowering with a capped approximation.
- Fixed-point charge conservation for `n_e`.
- Metal electron donors in the solar line-forming layers.
- Completion of the atmosphere with electron density.
- PFSAHA per-ion partition functions, stored `F/U` factors, opacity population factors, PFIRON grids, NNN packed tables, explicit light-element level sums, ground-state floors, occupation corrections, and log-space Saha stabilization.

Chapter 2 deliberately moved or softened:

- Rendered NumPy/reference comparisons were replaced by charge-closure and stage-population checks; parity remains in `_pipeline/verify_pfsaha.py`.
- `pfsaha_truth.npz`, `pykurucz`, `kgpu`, production-engine, oracle, and documented-float-floor language were removed from the lecture.
- Exercises no longer ask for internal variable names as the main task; they ask for physical changes, limiting cases, and interpretation.
- The PFSAHA stored-stage electron closure is not exact at all depths because only the first six stages are stored in the lecture arrays; the photospheric stages close as expected, while the verifier owns exact table parity.

## Chapter 3 Layout

Chapter 3 now follows this sequence:

1. **Continuous opacity.** Mass extinction, column mass, true absorption, scattering, and the total continuum are defined locally.
2. **Numerical grid.** Depth columns and wavelength rows explain why the opacity is a depth-by-wavelength tensor.
3. **Stimulated emission.** The factor `1 - exp(-h nu/kT)` is introduced for true absorption only.
4. **H-minus abundance.** The H$^-$ Saha balance is derived at the level needed to show why the opacity tracks electron density.
5. **H-minus opacity.** John (1988) bound-free and free-free fits are introduced, including the threshold and the branchless clamp.
6. **Scattering.** Rayleigh and Thomson are defined and compared physically.
7. **Continuum formation.** The analytic total opacity is integrated over column mass to locate the 505 nm optical-depth surface.
8. **Tabulated engine.** The lecture explains why detailed synthesis uses tables: structured cross-sections, additional opacity sources, and edge-triplet interpolation.
9. **Table machinery.** The KAPP-style constants, H$^-$, H I, Coulomb, Rayleigh, helium, hot-star, metal, and molecular terms are assembled as source tensors.
10. **Budget and interpolation.** Source budgets are read physically, and the 3-point Lagrange reconstruction produces the tabulated continuum.
11. **Overlay.** The final plot compares the analytic continuum with the tabulated continuum, not with a rendered reference gate.

## Chapter 3 Fact And Retention Notes

Chapter 3 keeps the main original concepts:

- True absorption versus scattering.
- Stimulated-emission correction for absorption.
- H$^-$ Saha abundance, bound-free photodetachment, and free-free absorption.
- The John (1988) H$^-$ analytic fits and branchless threshold handling.
- Rayleigh scattering on neutral H and Thomson scattering on free electrons.
- Continuum optical depth from `d tau_lambda = kappa_lambda dm`.
- Tabulated H$^-$, H I Karzas-Latter, Coulomb free-free, Gavrila Rayleigh, helium, H$_2^+$, He$^-$, metal-edge, Si II, and HOTOP terms.
- Edge-triplet frequency sampling and 3-point Lagrange interpolation of `log10(kappa)`.
- Source-budget interpretation of the tabulated continuum.

Chapter 3 deliberately moved or softened:

- The analytic fp64-twin verifier and full-depth exact-engine assertion were removed from the rendered lecture. Strict KAPP parity stays in `_pipeline/verify_kapp.py`.
- Reference/oracle/production wording was replaced by "data bundle," "table convention," "tabulated engine," and "bookkeeping check."
- The final overlay now compares analytic and tabulated continua directly; it does not plot reference residuals in the textbook.
- The fp64 table-engine choice is explained as a stable way to handle discrete brackets and nearly equal exponential differences, not as a broad dtype lesson.
- The historical He II free-free source-order convention remains, but it is explained as a named table convention rather than implementation archaeology.

## Style Pitfalls Already Found

Avoid forward-looking meta sentences such as "the notes move between equations..." or "the path is deliberately narrow." Let the lecture itself carry the structure.

Avoid saying code cells are not a programming tutorial. The lecture should simply be self-contained: prose explains the physics, equations define the model, and code evaluates the model.

Avoid introducing the same term twice. In Chapter 1, source function is defined in LTE; Rosseland optical depth is defined once and then reused in hydrostatic equilibrium.

Avoid dropping a named concept before explaining why it appears. Optical depth, Rosseland mean, Hopf function, closure, and radiative equilibrium all need a one-paragraph motivation before the equation.

Avoid overexplaining PyTorch, NumPy, dtype, or setup mechanics in the textbook body. Those are execution details unless the lecture is explicitly about numerical precision or GPU-native formulation.

## Chapter 1 Problems and Replacement Patterns

Chapter 1 exposed several habits that later lecture edits should handle in both directions: remove the weak pattern and add the stronger teaching pattern.

**Lecture-opening lists.** Avoid starting with a long list of what the lecture will cover. Edit in a physical question or tension that makes the next concept necessary, then let the structure unfold from that question.

**Setup-first writing.** Avoid making imports, device choice, dtype, plotting defaults, or backend selection the intellectual entry point. Edit in a short numerical-conventions paragraph only when needed, then move quickly to the physical quantity being evaluated.

**PyTorch overexplanation.** Avoid turning the textbook into a PyTorch or NumPy tutorial. Edit in one clear reason for the computational choice: PyTorch is used because the same array expressions can run GPU-natively. Let code examples teach the physics, not the framework.

**Hand-written constants.** Avoid redefining constants by hand when a standard library can supply them. Edit in Astropy constants where practical, convert once to the working units, and keep the physical formula recognizable.

**Unmotivated prefactors.** Avoid dropping numerical prefactors without origin or units. Edit in the algebraic source of the prefactor, its units, and why the scaled form is numerically useful.

**Verifier material in the lecture.** Avoid parity tests, benchmark references, "precomputed" language, and reference-array discussion in the rendered textbook. Edit in physical sanity checks instead: monotonicity, limiting behavior, scale comparisons, and interpretable tables. Keep numerical parity in separate verifier scripts.

**Project-internal references.** Avoid `kgpu`, `pykurucz`, production-engine names, development branches, and implementation archaeology in self-contained lectures. Edit in generic physics language: "the atmosphere table," "the transfer solver," "the opacity calculation," or "the reference calculation" only when a reference is truly needed.

**Placeholder citations.** Avoid citation-like statements without real sources. Edit in a real bibliographic source or a real website link when citation is needed. Otherwise state the fact without pretending it is cited.

**Duplicate introductions.** Avoid introducing the same term twice as if it were new. Edit in one first-use definition, then later reuse it with phrases like "using the relation above" or "the same construction now becomes." Chapter 1 needed this for the source function and Rosseland optical depth.

**Named concepts without motivation.** Avoid dropping major terms before explaining why they appear. Edit in a short first-use explanation for effective temperature, surface gravity, static atmosphere, plane-parallel geometry, grey opacity, optical depth, extinction, column mass, source function, LTE, Kirchhoff's law, radiative equilibrium, closure, Eddington approximation, Hopf function, Rosseland mean, Rosseland optical depth, hydrostatic equilibrium, and radiation pressure.

**Factoid equations.** Avoid stating important equations as isolated facts. Edit in a three-step pattern: physical need, equation, interpretation. For example, `I = I_0 exp(-tau)`, the grey `T^4` law, `d tau_Ross = kappa_Ross dm`, and `dP/dtau_Ross = g/kappa_Ross` each need a short bridge before the equation and a short explanation after it.

**Overlong derivations.** Avoid derivations that are technically correct but interrupt the lecture. Edit in the useful insight: what assumption fixes the slope, what boundary condition fixes the offset, and what the result means physically. Keep full algebra only when the algebra is the lesson.

**Vague accuracy language.** Avoid phrases like "a more accurate analytic fit" unless the comparison target is named. Edit in the actual status of the expression: a boundary correction, an approximation, a modeling convention, or an ATLAS/Kurucz grey-start fit used by the reference calculation.

**Unexplained normalizations.** Avoid assumptions such as `kappa = 1` without saying what they do. Edit in the role of the normalization. In Chapter 1, `kappa_Ross = 1 cm^2 g^-1` only turns optical depth into column mass and pressure for a first model; it is not a solar opacity claim.

**Implementation-trivia exercises.** Avoid exercises about variable names, code plumbing, or backend mechanics. Edit in exercises that test a physical relation, change a physical parameter, compare limiting cases, or interpret a plotted/table value.

**Oversized code cells.** Avoid code cells that do several conceptual jobs at once. Edit in bite-size cells with one purpose, readable variable names, and minimal machinery. The prose before the cell should say what is being evaluated; the prose after should say how to read the output.

**Missing schematics.** Avoid leaving spatial, layered, or iterative ideas purely verbal when a simple schematic would make them clear. Edit in schematics for optical depth, depth grids, line formation heights, opacity windows, and iteration loops when they clarify the physics.

**Dependence on backward or forward references.** Avoid making a lecture understandable only if the reader remembers another lecture. Edit in enough local context for the current calculation to stand alone, while keeping any recap short and purposeful.

## Schematic Strategy

Use schematics only when they make a concept easier to follow. Chapter 1 currently uses:

- `resources/figures/s1_pipeline.png` for the forward-model chain.
- `resources/figures/s1_optical_depth.png` for photon escape and optical depth of order unity.
- `resources/figures/s1_atmosphere_structure.png` for the depth grid, temperature increase inward, and pressure increase inward.
- `resources/figures/s2_saha_boltzmann.png` for the relationship between Boltzmann level populations, Saha ion stages, and electron donors.
- `resources/figures/s3_hminus.png` for the H$^-$ photodetachment picture and why electron density matters.

Future schematics should clarify a physical relationship, not decorate a page.

## Verification Checklist

For edited chapters, the current expected checks are:

- Build source notebook with `_pipeline/build_lecture1_gpu.py`.
- Execute and render with `_pipeline/build.py 1`.
- Run `_pipeline/verify_atmosphere.py`.
- Refresh `http://127.0.0.1:8899/reader.html?ch=1`.
- Confirm rendered Chapter 1 contains no `pykurucz`, `kgpu`, or `precomputed` language.
- Build and render Chapter 2 with `_pipeline/build_lecture2_gpu.py` and `_pipeline/build.py 2`.
- Run `_pipeline/verify_pfsaha.py`.
- Build and render Chapter 3 with `_pipeline/build_lecture3_gpu.py` and `_pipeline/build.py 3`.
- Run `_pipeline/verify_kapp.py`.
- Grep rendered Chapters 2 and 3 after stripping image data for project/test language: `pykurucz`, `kgpu`, `precomputed`, `comparison cell`, `oracle`, `production`, `float floor`, `parity`, `benchmark`, and `diagnostic`.
- Confirm `pykurucz_gpu` remains untouched.

As later chapters are edited, append their layout and any fact-audit caveats here before moving on to the next lecture.
