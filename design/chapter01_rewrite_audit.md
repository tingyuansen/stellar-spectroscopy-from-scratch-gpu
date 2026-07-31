# Chapter 1 Rewrite Audit

This audit is the quality gate for the new textbook. It compares the previous
`content/Lecture1.ipynb` with the first generated `content/Chapter01.ipynb` and
records what must be preserved, corrected, and strengthened before Chapter 2
is drafted.

## Verdict

The generated Chapter 1 is executable but not yet acceptable as the style
prototype.

The central failure is not simply that the new chapter is shorter. It removes
several reasoning steps that a first-time reader needs, while simultaneously
inserting 151 lines of source code into five Markdown cells. The result asks
the reader to make larger conceptual jumps and then confronts implementation
objects before their purpose is clear.

The rewrite must recover the patient causal flow of the previous Lecture 1,
while changing its hardware story and names to the exact pinned Payne Zero
implementation.

## Measured Comparison

| measure | previous Lecture 1 | generated Chapter 1 |
| --- | ---: | ---: |
| prose words | 5,184 | 3,001 |
| Markdown cells | 19 | 23 |
| executable code cells | 12 | 11 |
| executable code lines | 196 | 162 |
| largest executable cell | 31 lines | 22 lines |
| Python lines embedded in Markdown | 0 | 151 |

The nominal executable-cell length improved, but that measurement hid the
large non-executable source dumps. Those dumps must be removed rather than
reclassified.

## What the Previous Lecture Did Well

### It began from an observable

The old opening did not begin with software objects. It began with a spectrum
as a record of photons and then asked what physical structure is required to
predict it. The atmosphere emerged as a necessity:

1. a spectrum contains continuum and lines;
2. those features form at different depths;
3. depth has no meaning without local temperature and pressure;
4. therefore the first object to construct is an atmosphere.

The new chapter must preserve this argument before mentioning APIs,
dataclasses, accelerators, or initializer families.

### It separated definitions from consequences

For `effective_temperature`, the previous lecture first defined
\(F=\sigma T_{\rm eff}^4\), then explained why \(T_{\rm eff}\) is not every
layer's temperature, then connected the gradient to wavelength-dependent
escape depth. Gravity and composition received the same treatment.

The generated chapter compresses these into bullets. The facts are present,
but the physical consequences are no longer developed.

### It derived optical depth rather than naming it

The previous lecture moved through:

1. mass extinction coefficient \(\kappa_\lambda\);
2. interaction probability per length \(\kappa_\lambda\rho\);
3. mean free path \(1/(\kappa_\lambda\rho)\);
4. sign convention \(d\tau_\lambda=-\kappa_\lambda\rho\,dz\);
5. differential attenuation;
6. its integral \(I=I_0e^{-\tau}\);
7. the physical meanings of thin, photospheric, and thick limits;
8. the change of coordinate to column mass;
9. the reason different wavelengths escape at different depths.

The new chapter retains equations but shortens the bridges. The rewrite must
restore the chain, including the ray-direction qualification
\(\tau_\lambda\sim\mu\).

### It prepared every approximation

The previous grey-atmosphere section explained why angular moments are
introduced, what bolometric means, why a closure is required, where the
factor \(4\pi\) enters, and which part of the temperature law is fixed by
radiative equilibrium versus the surface boundary.

The new chapter reaches the correct Eddington law too quickly. A novice could
copy the equations without understanding why three moments appear or why
\(K=J/3\) closes the system.

### It was honest about the first atmosphere

The old lecture repeatedly distinguished:

- a physical law from a modeling choice;
- a grey scaffold from a wavelength-dependent atmosphere;
- unit opacity from a computed Rosseland opacity;
- gas pressure from total and radiation pressure;
- a first atmosphere from a complete equation-of-state solution.

That level of qualification is the correct standard for the new book.

## Failures in the Generated Chapter

### 1. A parallel API was invented

The draft introduced:

- `StellarLabels.effective_temperature_K`;
- `StellarLabels.log10_surface_gravity_cm_s2`;
- `SpectrumResult.total_flux_per_nm`;
- `SpectrumResult.continuum_flux_per_nm`;
- `SpectrumWindow.start_wavelength_nm`;
- `AtmosphereRole`.

These are not Payne Zero's public names. Payne Zero uses:

- `effective_temperature`;
- `log_surface_gravity`;
- `metallicity`;
- `alpha_enhancement`;
- `microturbulence_km_s`;
- `wavelength_start_nm` and `wavelength_end_nm`;
- `Spectrum.flux_total`, `Spectrum.flux_continuum`, and
  `Spectrum.normalized_flux`;
- `InitializedAtmosphere.atmosphere_converged` and
  `InitializedAtmosphere.atmosphere_closure_required`.

The invented contracts must be deleted. Units belong in the prose, docstring,
contract table, and validation—not in renamed public fields that teach a
different system.

### 2. Source display replaced pedagogy

Five Markdown cells contain generated source excerpts of 36, 32, 8, 17, and
58 lines. The reader encounters complete dataclasses and validation methods
before needing them. The book also switches visual grammar from a notebook
conversation to a documentation dump.

New rule for Chapter 1:

- no source code inside Markdown fences;
- no entire class or multi-function excerpt;
- an executable code cell has one conceptual job;
- production code is shown only when its required physics and inputs have
  already been developed;
- a displayed exact Payne Zero function should normally fit in 10–25 lines;
- larger routines are taught by named stages across later cells and chapters.

### 3. The architecture arrived before the need

The initializer-versus-converged workflow is important, but the opening
schematic currently appears before the reader has a firm idea of what an
atmosphere contains. Chapter 1 should first establish atmosphere and synthesis
as physical objects. It can then show where an initializer sits and explain
the exact safety fields.

### 4. Several explanatory bridges were removed

The rewrite must restore at least:

- why a wavelength and frequency bin require a Jacobian;
- what the Planck factors mean physically;
- why the stable exponential form is algebraically identical;
- why optical depth is more useful than geometric kilometres;
- how a source function differs from attenuation alone;
- why LTE is local and does not make the whole emergent spectrum a blackbody;
- how true absorption differs from scattering;
- why angular moments are useful;
- why closure is mathematically necessary;
- why the Rosseland mean is harmonic and weighted by
  \(\partial B_\lambda/\partial T\);
- why opacity controls pressure at fixed optical depth;
- exactly which columns remain unknown after the grey exercise.

### 5. Exact implementation context was incomplete

Chapter 1 should not pretend that its unit-opacity atmosphere is Payne Zero's
production atmosphere solver. It should instead identify three exact
connections:

1. `payne_zero_synthesis.radiative_transfer.planck_bnu` implements the thermal
   \(B_\nu\) used by synthesis on a `[depth, wavelength]` Torch tensor.
2. `payne_zero_atmosphere.run_setup.standard_rosseland_optical_depth_grid`
   constructs the standard 80-layer outer-to-inner grid.
3. `AtmosphereInitializer.predict` uses the Eddington grey temperature as the
   multiplicative baseline for its learned temperature correction.

The unit-opacity hydrostatic calculation remains a declared controlled limit
for learning. Its arrays must use Payne Zero names (`temperature`,
`column_mass`, `gas_pressure`, `rosseland_opacity`) and must not be wrapped in
an invented production-looking dataclass.

## Required Narrative Sequence

The new Chapter 1 will use the following dependency order.

1. **The observable question.** What physical chain produces a stellar
   spectrum?
2. **The forward problem.** Define \(F_\lambda\), `effective_temperature`,
   `log_surface_gravity`, composition, and `microturbulence_km_s`.
3. **Why an atmosphere comes first.** Separate atmosphere structure from
   synthesis and list what a layer eventually contains.
4. **Model assumptions.** One-dimensional, plane-parallel, static, LTE, and
   hydrostatic, each with a physical meaning and failure horizon.
5. **Units and constants.** Introduce CGS and Payne Zero's exact versus
   parity-pinned constant tiers.
6. **Thermal scale.** Compare \(h\nu\) and \(kT\) numerically.
7. **Planck radiation.** Derive \(B_\nu\), derive the \(B_\lambda\) Jacobian,
   evaluate a scalar limit, then introduce the exact vectorized Torch
   `planck_bnu`.
8. **Photon escape.** Derive mean free path, optical depth, attenuation, and
   column mass; inspect limiting cases.
9. **Local emission.** Introduce the transfer equation, source function, LTE,
   and a constant-source slab.
10. **Grey radiative equilibrium.** Introduce moments one at a time, require a
    closure, derive the Eddington law, and identify the exact grey baseline in
    the initializer.
11. **One shared depth grid.** Motivate and define the Rosseland mean, then
    call the exact depth-grid function.
12. **A controlled hydrostatic limit.** Use unit `rosseland_opacity` to build
    `column_mass` and `gas_pressure`, with radiation-pressure bookkeeping.
13. **The production contract ahead.** Show the exact `ModelAtmosphere`
    field names and distinguish which are and are not yet computed.
14. **What was learned.** Reconstruct the complete argument in prose before
    exercises.

## Cell-Level Standard

Every code cell must answer one question that the preceding prose has already
posed.

Before a cell, state:

- what will be computed;
- the exact input names;
- units and shapes;
- why the chosen NumPy or Torch representation matches Payne Zero.

After a cell, state:

- what the numbers or figure mean physically;
- one check or limiting case;
- what the cell deliberately did not compute.

The first chapter may use local scalar expressions for explanation. Whenever
a corresponding Payne Zero name exists, the local expression must use that
name. The only callable production code displayed in the chapter will be
small exact functions drawn from the progressive Payne Zero package.

## Quantitative Figure Standard

The Payne Zero paper figures are the reference for typographic finish, visual
hierarchy, and restrained scientific color. The textbook adapts that language
to slower teaching figures rather than copying the paper's intentionally dense
multi-panel layouts.

- Prefer one panel and one physical claim per figure.
- Build a complicated comparison across successive figures or successive
  cells. Use multiple panels only when the relationship between panels is the
  concept being taught.
- Use a restrained, color-safe palette based on the paper's black/slate,
  orange, blue, green, and muted magenta, harmonized with the official
  website's navy, beige, and warm grey.
- Never rely on positional Matplotlib colors such as `C0` and `C3` in
  canonical chapter code.
- Use the paper's clear line hierarchy: a dark reference curve, a colored
  model curve, and lighter supporting guides. Line style must remain
  distinguishable in greyscale.
- Use consistent serif mathematical typography with a highly readable text
  face, consistent tick direction, quiet spines, and book-wide sizes.
- Put the physical quantity and unit on every axis. A logarithm label must
  make clear which quantity is inside the logarithm and what unit convention
  is being used.
- Prefer direct curve labels when they reduce eye travel. Otherwise order the
  legend physically and keep it away from the data.
- Use subtle major guides only when comparison needs them. Avoid dense
  default grids, decorative boxes, gradients, and rainbow color maps.
- Mark a physically meaningful location such as
  \(\tau_{\rm Ross}=2/3\) with a quiet annotation that does not cover the
  curve.
- Choose axis limits from the scientific question rather than accepting every
  numerical outlier.
- Use an explicit one-panel figure width, high render resolution, and
  constrained layout. Inspect for clipped labels, overlapping annotations,
  small tick text, and excessive whitespace.
- A caption states the relationship to notice. The following prose explains
  the physics; it does not merely say that the curve rises or falls.
- Conceptual schematics retain the official hand-sketched website aesthetic.
  Quantitative plots remain precise and typeset. The two figure classes must
  remain visually distinct.

## Acceptance Gates

Chapter 1 is ready only when all of the following hold:

- the prose is at least as explanatory as the previous Lecture 1, without
  copying obsolete all-Torch claims;
- no undefined mathematical symbol appears in an equation;
- every exact Payne Zero Python name is spelled as in the pinned source;
- no invented public class or field remains;
- no Python fence appears inside a Markdown cell;
- every executable cell is at most 35 lines and normally below 25;
- every code cell has a prose setup and a prose interpretation;
- `planck_bnu` and `standard_rosseland_optical_depth_grid` agree with the
  pinned source;
- all outputs execute without error;
- every quantitative plot passes the shared style and visual-inspection gate;
- the rendered page is inspected, not merely generated;
- a final read-through finds no backward dependency, unexplained hardware
  concept, or premature source dump.
