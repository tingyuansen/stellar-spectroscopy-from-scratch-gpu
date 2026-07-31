# Chapter 1 Pedagogical-Flow Audit

Scope: `book/chapters/chapter_01.py`, reviewed against the current `BIBLE.md` and the pacing
standard embodied by `content/Lecture1.ipynb`.

This is a narrative audit, not a physics- or API-correction pass. The chapter source was not
edited.

## Verdict

The chapter has a sound physical spine:

\[
\text{observed flux}
\rightarrow
\text{local thermal emission}
\rightarrow
\text{difficulty of escape}
\rightarrow
\text{local replenishment}
\rightarrow
\text{temperature with depth}
\rightarrow
\text{pressure with depth}.
\]

Sections 1.5–1.10 mostly create the need for their successors, and most numerical outputs receive
an immediate physical reading. The strongest transitions are:

- “The Planck function tells us the thermal radiation a layer can supply locally. It does not tell
  us whether a photon reaches space” (`chapter_01.py:535–537`);
- “Attenuation alone can only make a beam dimmer. A stellar atmosphere also emits”
  (`chapter_01.py:659–660`);
- “We now need a temperature at every depth” (`chapter_01.py:740`);
- “Temperature alone is not an atmosphere” (`chapter_01.py:1017`).

Those sentences are the Lecture 1 voice: each closes one result, names its insufficiency, and makes
the next concept necessary.

The chapter is not yet a fully successful style prototype because the same causal discipline is
not maintained at the opening, in the angular-moment derivation, or in the Rosseland-mean
introduction. The opening question is delayed by a syllabus-like objective list. Exact public
fields, learned-initializer architecture, safety metadata, and future atmosphere fields appear
before the reader needs them. Later, three angular moments and a harmonic-mean integral arrive
without the concrete example required by the BIBLE’s four-layer teaching rule. The closing
summarizes the scaffold but motivates Chapter 2 as a list of software topics rather than as the
solution to a specific failure of the calculation just built.

The required revision is therefore mostly one of sequencing and subtraction, not expansion. Keep
the middle physical arc, remove early implementation detours, insert two small conceptual bridges,
and make the final forward dependency causal.

## 1. Paragraph and Section Dependency Map

### Macro dependency spine

| Lines | Paragraph or section job | What it requires | What it should make necessary | Flow assessment |
| --- | --- | --- | --- | --- |
| 14–48 | Opening observable and governing question | Only ordinary experience of a plotted spectrum | A forward model and an atmosphere | Strong after line 31; weakened because learning objectives precede the observable |
| 53–91 | Define the forward problem, \(T_{\rm eff}\), gravity, composition, and microturbulence | Opening question | A depth-dependent state on which gravity and emission can act | Mostly strong; composition and microturbulence are previewed but not used |
| 96–151 | Exact label names, constant setup, and gravity calculation | Label definitions | A depth coordinate and pressure structure | The gravity result creates the right need; the exact-name table and unused labels slow the approach |
| 153–198 | Exact `Spectrum` fields and a three-point illustrative spectrum | Meaning of \(F_\lambda\) | Why an atmosphere must precede synthesis | Conceptually valid but misplaced; it interrupts the gravity-to-depth dependency |
| 200–216 | Separate atmosphere construction from spectral synthesis | Opening spectrum and label discussion | Model assumptions and a layer state | Strong and necessary |
| 220–245 | Forward schematic and learned-initializer workflow | Atmosphere/synthesis distinction | Supposedly the grey baseline | The schematic helps; the initializer, neural-model explanation, safety flags, and Chapter 14 pointer are premature |
| 249–276 | State one-dimensional, plane-parallel, static, LTE, and hydrostatic assumptions | Definition of atmosphere | A layer geometry and local radiation calculation | Strong, although LTE is necessarily only a preview here |
| 281–295 | Formation-depth schematic and full future field inventory | Would require opacity and optical depth | Thermal radiation and escape depth | Backward dependency: “optically thick,” strong-line formation, opacity, and several future fields have not been taught |
| 300–353 | Establish CGS, constants, and compare \(h\nu\) with \(kT\) | Labels and assumptions | A thermal radiation law | The photon/thermal comparison is useful; constant-tier policy and `PLANCK_PREFACTOR` arrive before the Planck equation needs them |
| 355–429 | Define intensity, Planck radiation, the spectral Jacobian, and stable algebra | Photon/thermal energy comparison | An exact thermal-radiation evaluation | Physically coherent; limits and concrete expectations should precede the production implementation |
| 431–529 | Display `planck_bnu`, form \(B_\lambda\), plot three temperatures, and inspect 505 nm | Planck equation and Jacobian | The problem of photon escape | Strong output–interpretation loop, but the 505 nm table largely repeats the plot and Torch terminology is underprepared |
| 533–653 | Derive mean free path, optical depth, attenuation, photospheric depth, and column mass | Local thermal emission | Local emission along an attenuating path | The chapter’s strongest complete dependency chain |
| 657–736 | Introduce source function, LTE source, transfer equation, and constant-source slab | Attenuation and optical depth | A temperature at every depth | Strong; the slab output is interpreted and directly creates the grey-atmosphere need |
| 738–843 | Introduce radiative equilibrium, \(J,H,K\), closure, and the Eddington grey law | LTE source and depth-dependent escape | One common depth coordinate and executable temperature profile | Correct arc, but too many abstractions arrive before a concrete angular example; the \(2/3\) boundary is asserted rather than earned |
| 848–919 | Motivate Rosseland opacity, define the mean, display the exact 80-layer constructor, and reconnect to the grey law | Wavelength-dependent escape and grey temperature | A numerical temperature array | Correct dependency, but the full integral precedes the simple harmonic-window example; source docstring commentary interrupts the physics |
| 921–1011 | Build, plot, and check the standard grid and grey temperature | Rosseland coordinate and grey law | A pressure scale | Strong equation–code–output–interpretation sequence |
| 1015–1090 | Derive hydrostatic pressure, declare unit opacity, introduce radiation pressure, and state the controlled limit | Column mass and temperature profile | Numerical `column_mass` and `gas_pressure` | Strong overall; radiation pressure should explicitly reuse the earlier \(K\) moment |
| 1093–1196 | Build arrays, print representative layers, plot pressure, and run identities | Hydrostatic controlled limit | An honest interface boundary | Code and checks are well scoped; the final radiation-pressure result is printed but never interpreted |
| 1200–1228 | Compare the four arrays with a full `ModelAtmosphere` | Completed scaffold | Exercises and a statement of what remains impossible | Appropriate implementation checkpoint and honest boundary |
| 1233–1274 | Exercises and further reading | Whole chapter | Reconstruction and transfer of learning | Exercises align with the chapter; further reading usefully remains outside the causal spine |
| 1278–1301 | Summary and Chapter 2 bridge | Whole chapter | Trustworthy numerical/interface discipline | Summary returns to the scaffold; forward bridge is too generic to make Chapter 2 feel inevitable |

### Dependency breaks

Four locations ask the reader to know something that the narrative has not yet supplied:

1. The formation-depth figure uses optical-thickness and line-formation language at
   `chapter_01.py:281–288`, before optical depth is defined at lines 533–563.
2. The constants section discusses numerical “parity,” a “branch,” and a “table boundary” at
   lines 307–325, before numerical parity is taught in Chapter 2 and before the Planck formula has
   appeared.
3. The exact Planck checkpoint uses “tensor,” broadcasting, axis insertion, dtype, and Torch
   execution at lines 416–438 without first defining a tensor as a multidimensional numerical
   array or stating that this chapter’s tensors are CPU float64.
4. The Rosseland harmonic-mean integral appears at lines 854–876 before a two-opacity-bin example
   has shown why transparent windows should dominate.

## 2. Weak Transitions and Premature Concepts

### High-priority flow problems

#### A. The chapter opens with a syllabus before it opens with the observable

The first reader-facing phrase after the title is “**Learning Objectives**” followed by eight
bullets (`chapter_01.py:18–29`). The observable begins only at “**Look at a high-resolution stellar
spectrum**” on line 31.

This reverses the BIBLE rule that a chapter opens with one physical question, tension, or
observable, and it weakens the strongest prose in the opening. Move the spectrum paragraph and
governing question directly under the title. Put a compact chapter contract after the reader has
accepted the problem. The current objectives can be shortened into that contract or moved to a
teacher-facing syllabus.

The chapter also lacks the required compact
“reads / writes / shape / units / dtype / device” contract. Adding it after the opening question
would satisfy the BIBLE without delaying the hook.

#### B. The gravity-created need for depth is interrupted by an output-API exercise

The excellent consequence
“**Pressure at a layer also depends on how much material lies above it. We therefore need a depth
coordinate**” (`chapter_01.py:148–151`) should lead immediately into what an atmosphere is.
Instead, “**Before building that structure, fix the meaning of the spectrum we ultimately want**”
(`chapter_01.py:153`) starts a field table and a synthetic three-point spectrum.

The field meanings are correct, and the middle sample is interpreted, but the insertion breaks the
newly created need. The reader has just been told that depth is missing; the next section should
give depth a physical home. Keep \(F_\lambda\), \(F_{\lambda,\mathrm c}\), and
\(f_{\rm norm}\) in the opening contract, but move the exact `Spectrum` field checkpoint to the end
of Section 1.2 or the final interface boundary. The three-point code cell can be cut: it does not
advance the construction of the atmosphere.

#### C. Learned initializers enter before the reader has built the object they initialize

“**Payne Zero supports two label-driven paths**” (`chapter_01.py:229`) introduces a learned model,
training examples, initializer proposal semantics, `InitializedAtmosphere`, and two safety flags
at lines 229–244. None is used in the chapter.

This is the clearest premature implementation detour. It makes the source architecture the subject
of the paragraph and creates a long forward reference to Chapter 14. Remove it from the early
narrative. At the final implementation checkpoint, one short note can say that the exact learned
initializer uses the grey relation as a baseline but remains unconverged and closure-required.
That is where the reader can understand what is being initialized and why the flags matter.

#### D. The formation-depth schematic precedes the mechanism it depicts

The caption begins “**Different wavelengths sample different ranges of depth**” and immediately
uses “optically thick,” strong lines, weak lines, and continuum formation
(`chapter_01.py:281–288`). The reader has not yet met mass extinction, mean free path, or optical
depth.

Move this figure to the end of Section 1.6, immediately after
\(d\tau_\lambda=\kappa_\lambda\,dm\) and the explanation that a strong line reaches optical depth
unity with less overlying mass. In that position the figure becomes a synthesis of an earned
result instead of a diagram that must be taken on faith.

The following inventory—“**Eventually, each layer will carry temperature, column mass, gas
pressure, electron density, mass density, microturbulence, populations, opacity, radiative
acceleration, and convective quantities**” (`chapter_01.py:291–295`) also exceeds the
just-in-time vocabulary. Retain only temperature, pressure, density, and composition here. The
full exact inventory already has a better home in Section 1.11.

#### E. Numerical constant policy is explained before the physical formula earns it

“**Payne Zero keeps two tiers of constants**” (`chapter_01.py:307`) and the detailed discussion of
parity-sensitive literals and `PLANCK_PREFACTOR` (`chapter_01.py:307–325`) appear before the Planck
function at line 367.

For the first pass, the reader needs only \(h\), \(c\), \(k\), their units, and the question
“how large is \(h\nu/kT\)?” Move the prefactor and constant-tier note to the exact `planck_bnu`
checkpoint after the Planck equation and stable algebra have been derived. Keep the Chapter 1 note
to two sentences; Chapter 2 owns the general constant-tier and parity policy.

The code output uses electron-volts at lines 341–345, but an electron-volt is never defined. Add
one sentence before the cell: one eV is the energy gained by one electron crossing a one-volt
potential, and here it is only a convenient energy unit.

#### F. Expected limiting behavior arrives after the implementation and plot

The chapter displays and executes the exact Planck function at lines 416–529. Only after the plot
does it name the Wien and Rayleigh–Jeans regimes (`chapter_01.py:505–509`).

For the BIBLE’s four-layer sequence, the limiting cases should come before the exact code:

1. ask what happens when \(h\nu\ll kT\) and \(h\nu\gg kT\);
2. derive the two simple limits;
3. predict the curve’s long- and short-wavelength behavior;
4. inspect `planck_bnu`;
5. use the plot as a check of those predictions.

The current plot interpretation is good, but moving the limits earlier turns it from observation
into verification.

At the implementation checkpoint, define a tensor in one sentence and state the runtime contract:
the inputs are CPU float64 Torch arrays in this chapter; the returned axes are
`[depth, wavelength]`. Full broadcasting, device, and performance mechanics belong to Chapter 2.

#### G. One implementation sentence creates an axis misconception

“**Later opacity and transfer arrays use the same depth-by-wavelength organization**”
(`chapter_01.py:439–440`) encourages the reader to infer a global layout. The exact synthesis
optical-depth path uses `[wavelength, depth]`, while `planck_bnu` returns
`[depth, wavelength]`.

Even in a flow audit this matters: Chapter 2 would have to make the reader unlearn a supposed
invariant. Replace the sentence with a narrative boundary: this function’s layout is exact and
function-specific; Chapter 2 will show where another kernel deliberately uses the opposite order.

#### H. The 505 nm table repeats the Planck plot without creating the escape problem

“**A smaller check isolates the temperature response at 505 nm**”
(`chapter_01.py:511–513`) produces a second output for the already interpreted claim that thermal
emission rises with temperature. Calling the three temperatures “three depths—or, in this
demonstration, three temperatures” also blurs the physical meaning of the depth axis.

Either cut this cell, or move its monotonic assertion into the array-construction cell before the
plot. Then let the plot interpretation end with the unresolved question: local brightness is known,
but which depth can escape? That gives Section 1.6 more momentum.

### Advanced concepts that need one more rung

#### I. Three angular moments arrive as a block

“**Instead, we compress the frequency-integrated—bolometric—intensity \(I(\mu)\) into angular
moments**” (`chapter_01.py:753–755`) is followed immediately by \(J\), \(H\), and \(K\), three
integrals, three interpretations, two moment equations, and a closure.

This is correct but denser than the Lecture 1 standard. Insert one concrete two- or three-ray
example:

- equal inward and outward rays give nonzero \(J\) but zero \(H\);
- weighting by \(\mu\) makes \(H\) retain net outward flow;
- weighting by \(\mu^2\) creates the pressure-like moment \(K\).

Then introduce the moments one at a time. Explicitly define
\(B(T)\equiv\int_0^\infty B_\nu(T)\,d\nu=\sigma T^4/\pi\) before writing
“\(J=B(T)\)” at line 780; otherwise \(B(T)\) looks like a new function unrelated to the
earlier \(B_\nu\).

The phrase “**The Eddington boundary condition supplies \(2/3\)**”
(`chapter_01.py:831–839`) is also too abrupt. Give a short no-incoming-radiation boundary argument
or label \(2/3\) explicitly as the adopted Eddington surface boundary and show what it fixes:
radiative equilibrium fixes the slope, while the boundary fixes the intercept.

#### J. The Rosseland integral precedes the transparent-window example

“**The appropriate representative opacity is therefore a weighted harmonic mean**”
(`chapter_01.py:854–856`) jumps directly to the full integral. The prose after the equation explains
the inverse weighting, but the reader has no numerical foothold.

Before the integral, compare two equal-weight bins with opacities 1 and 100:

- arithmetic mean: 50.5;
- harmonic mean: about 1.98.

Ask which result better represents diffusion through a transparent window. Then replace equal
weights by \(\partial B_\lambda/\partial T\) and write the integral. This restores the required
concrete-example-before-equation order.

The paragraph beginning “**The function's compact docstring names its immediate use in
`run_setup`**” (`chapter_01.py:890–893`) is implementation archaeology, not a physical
transition. Cut it. The exact function name, output ordering, shape, and dtype are sufficient at
this checkpoint.

The following learned-initializer paragraph (`chapter_01.py:905–907`) repeats the premature
initializer theme. Replace it with the immediate physical task: evaluate the grey law on the exact
80-layer coordinate. Mention the initializer connection only at the final boundary.

#### K. Radiation pressure is not connected back to the moment that prepared it

“**Radiation also carries momentum**” (`chapter_01.py:1070`) introduces
\(P_{\rm rad}=4\sigma T^4/(3c)\) as a new fact. Section 1.8 already identified \(K\) as the
radiation-pressure moment.

Add the missing causal bridge: the same angular weighting represented by \(K\) measures radiation
momentum; for an isotropic thermal field it reduces to \(aT^4/3=4\sigma T^4/(3c)\). This makes the
moment section pay off and prevents radiation pressure from feeling appended to the hydrostatic
calculation.

### Outputs and closing

#### L. The final numerical diagnostic is printed but not read

The last code cell prints
“`bottom radiation-pressure correction = ... of total`”
(`chapter_01.py:1193–1195`). The next paragraph starts Section 1.11 without interpreting that
number.

Add one short paragraph stating whether radiation pressure is negligible or material in this
solar-like controlled limit, why that result is plausible, and why hotter or lower-gravity
atmospheres may behave differently. The identity checks confirm algebra; the prose must interpret
the physics.

#### M. The forward bridge lists tools instead of exposing the next obstacle

The closing says:
“**Before adding that physics, we need rules for arrays, units, data provenance, numerical
integration, compilation, parallel loops, device precision, and reproducible checks**”
(`chapter_01.py:1295–1298`).

The list is accurate but feels like a curriculum detour. Make the dependency concrete:

- trustworthy now: analytic Planck, escape, grey-temperature, and unit-opacity hydrostatic
  relations on 80 layers;
- still impossible: physical populations, opacity, a schema-valid atmosphere handoff, and an
  emergent spectrum;
- immediate numerical obstacle: the depth integral must be repeated over many frequencies without
  changing its units, axis meaning, or arithmetic;
- therefore Chapter 2 must establish the exact parabolic integration, array/schema contract, and
  parity/timing discipline before later chapters add state.

That bridge returns directly to the opening question: the reader still cannot predict
\(F_\lambda\), but now knows precisely which trustworthy computational foundation must come next.

## 3. Recommended Reorder, Cuts, and Additions

### Reorder

1. Put the observable paragraph and governing question (`lines 31–48`) immediately after the
   title.
2. Follow the question with a compact chapter contract. Include the exact output names only as a
   boundary, not as a second narrative.
3. Let the gravity interpretation (`lines 148–151`) flow directly into the physical
   atmosphere/synthesis distinction (`lines 200–216`).
4. Move the formation-depth schematic (`lines 281–288`) to the end of Section 1.6, after optical
   depth and column mass.
5. Move Wien and Rayleigh–Jeans reasoning (`lines 505–509`, expanded slightly) before the exact
   `planck_bnu` source display.
6. Keep the exact `standard_rosseland_optical_depth_grid` checkpoint in Section 1.9, but place the
   two-bin Rosseland example before the integral.
7. Move the learned-initializer baseline and safety-field note to Section 1.11, after the reader can
   distinguish a scaffold, an initializer, and a physically accepted atmosphere.

### Cut or condense

- Cut the standalone three-point spectrum code (`lines 171–191`). Retain one equation and one
  compact output-contract row if needed.
- Condense the exact label table (`lines 96–112`) to the labels actually used in the chapter:
  `effective_temperature` and `log_surface_gravity`. Composition and microturbulence can remain a
  one-sentence horizon; their exact algebra belongs to Chapter 2.
- Cut the early learned-model architecture (`lines 229–244`).
- Cut the premature full atmosphere-field list (`lines 291–295`); retain the complete exact table
  at the Section 1.11 checkpoint.
- Condense the constant-tier policy (`lines 307–325`) and teach the general policy in Chapter 2.
- Cut the second 505 nm Planck output (`lines 511–529`) or fold its monotonic assertion into the
  first Planck calculation.
- Cut the source-docstring defense and cross-module use list (`lines 890–893`).
- Avoid repeating that the initializer uses the grey baseline in Sections 1.2 and 1.9; say it once
  at the closing implementation boundary.

### Add

1. A compact chapter contract after the opening question:

   - reads: elementary waves, energy, calculus, and ideal gas;
   - writes: `standard_rosseland_optical_depth`, `temperature`, `column_mass`,
     `gas_pressure`;
   - shapes: depth `(80,)`, Planck `[depth, wavelength]`;
   - units: CGS internally, nm at the spectrum boundary;
   - dtype/device: local NumPy float64 CPU; exact Planck demonstration Torch float64 CPU;
   - honest boundary: no EOS, physical opacity, populations, convergence, or spectrum.

2. A one-sentence definition of electron-volt before the photon-energy output.
3. A one-sentence definition of tensor and a tiny axis sketch before `planck_bnu`.
4. Rayleigh–Jeans and Wien predictions before the exact Planck implementation.
5. A concrete angular-ray example before \(J,H,K\).
6. An explicit definition of bolometric \(B(T)=\int B_\nu\,d\nu\).
7. A short physical explanation of the Eddington surface constant.
8. A two-bin arithmetic-versus-harmonic opacity example before the Rosseland integral.
9. A bridge from \(K\) to radiation pressure.
10. A prose interpretation of the bottom radiation-pressure fraction.
11. A final bridge framed as one unresolved calculation, not a list of software subjects.

### Exact-name placement rule

Keep exact Payne Zero names at four implementation checkpoints:

1. the compact input/output chapter contract;
2. the exact `planck_bnu` source and its function-specific layout;
3. the exact `standard_rosseland_optical_depth_grid` source and returned coordinate;
4. the final `ModelAtmosphere`/initializer boundary, including the convergence safety fields.

Between those checkpoints, use physical nouns and mathematical symbols. Repeating product names in
the physical derivation makes the chapter read like API documentation and weakens the “reader is
building the calculation” viewpoint.

## 4. Revised High-Level Act Structure

### Act I — Why a spectrum requires layers

**Opening question:** Why can one stellar temperature not directly produce the observed
wavelength-by-wavelength flux?

1. Begin with continuum plus absorption features.
2. Define the forward problem and \(F_\lambda\).
3. Define \(T_{\rm eff}\) through total flux and show why it is not a layer temperature.
4. Compute gravity and use the result to ask how much mass lies above each layer.
5. Define atmosphere versus synthesis.
6. State the five controlled assumptions.

**Visible result:** the gravity value and a causal schematic.

**Checkpoint:** labels and flux are understood, but no layer can emit or support weight because
temperature, escape depth, and column mass are missing.

### Act II — What a layer emits and what can escape

**Question:** What radiation can a layer supply locally, and how much of it survives to the
surface?

1. Introduce CGS and compare \(h\nu\) with \(kT\).
2. Build \(B_\nu\), the wavelength Jacobian, and the two asymptotic limits.
3. At the exact implementation checkpoint, inspect `planck_bnu`.
4. Plot and interpret the temperature dependence.
5. Derive extinction per length, mean free path, optical depth, attenuation, and column mass.
6. Move the formation-depth schematic here.
7. Show why attenuation alone is incomplete; introduce source function, LTE, and the slab solution.

**Visible results:** one Planck plot, one attenuation plot, and one slab plot.

**Checkpoint:** local supply and escape are understood, but the source varies with depth and no
temperature profile exists.

### Act III — How constant flux assigns temperature to depth

**Question:** What temperature profile can carry the required total flux through all layers?

1. Define radiative equilibrium and bolometric integration.
2. Use a concrete angular-ray example to motivate \(J\), \(H\), and \(K\).
3. Show why the moment equations need a closure.
4. Derive the grey slope and explain the surface boundary separately.
5. Motivate a shared depth coordinate with a two-bin transparent-window example.
6. Define the Rosseland mean and \(\tau_{\rm Ross}\).
7. At the exact checkpoint, construct the 80-layer grid.
8. Evaluate, plot, and check the grey temperature.

**Visible result:** the 80-layer temperature profile with its
\(\tau_{\rm Ross}=2/3\) check.

**Checkpoint:** temperature is assigned, but the layers still have no mass or pressure scale.

### Act IV — How gravity assigns pressure, and where the scaffold stops

**Question:** How much pressure is required to support the mass above each layer?

1. Derive hydrostatic balance in column mass.
2. Declare unit Rosseland opacity as a controlled normalization.
3. Reuse \(K\) to motivate radiation pressure.
4. Build `column_mass` and `gas_pressure`.
5. Read representative layers, the pressure profile, and the radiation-pressure fraction.
6. Compare the four arrays with the exact complete-atmosphere boundary.
7. Place the learned-initializer baseline and safety fields here as a short implementation note.

**Visible result:** a compact structure table and one pressure plot.

**Final answer to the opening question:** a spectrum cannot be predicted from labels alone. The
chapter has built the first temperature–pressure scaffold that an opacity and transfer calculation
would need, but it has not yet built the material populations, physical opacity, validated
atmosphere interface, or emergent flux.

**Forward dependency:** Chapter 2 makes the depth integral, array meanings, schema, numerical
agreement, and performance boundaries trustworthy before the book adds more physics.

## 5. Redundancy Audit

| Repeated idea | Current locations | Judgment | Recommendation |
| --- | --- | --- | --- |
| Continuum plus absorption features | 31–36; 153–168; 195–198; 200–215 | Repeated before the main construction begins | Keep the opening observable and one atmosphere/synthesis consequence; cut the synthetic spectrum cell |
| Exact public labels and outputs | 96–117; 153–168; 1200–1223 | Too much interface material before the physics | Keep a compact contract and the final exact boundary |
| Learned initializer and grey baseline | 229–244; 905–907 | Same future architecture explained twice | Move one concise statement to Section 1.11 |
| Formation at different depths | 80–82; 281–288; 630–652; 733–736 | The central idea is previewed before it is derived | Keep one opening hint, put the figure and full explanation after optical depth, and keep the slab consequence brief |
| Exact/reference constant policy | 307–325; 426–428 | General policy is taught twice and belongs mainly to Chapter 2 | State physical constants early; keep the pinned-prefactor distinction only at `planck_bnu` |
| Planck intensity grows with temperature | 465–503; 511–529 | Plot and table make the same claim | Keep the plot and fold the monotonic assertion into its calculation |
| LTE is local, not a global blackbody | 266–268; 673–683; 1282–1284 | Productive recurrence: preview, derivation, summary | Keep, but make the assumption preview one sentence |
| Photosphere near optical depth unity | 585–591; 630–634; 650–652; 944–951 | Mostly productive, but some phrasing repeats | Use attenuation for the quantitative meaning, the moved schematic for wavelength dependence, and the grey plot for the \(2/3\) boundary |
| Grey temperature rises inward | 930–951; 990–1010 | Output preview, plot, and analytic check overlap | Keep plot plus analytic check; combine the two setup paragraphs |
| Hydrostatic pressure rises inward | 1120–1144; 1148–1177 | Table and plot can serve distinct purposes, but current prose reads both identically | Use the table for scale and units; use the plot for monotonic structure; interpret the radiation fraction after the final check |
| The scaffold is incomplete | 291–295; 1088–1090; 1200–1228; 1288–1291 | Necessary honesty repeated four times | Use one early horizon, the full Section 1.11 contract, and one summary sentence |
| \(F=\sigma T_{\rm eff}^4\) | 71–80; 745–751 | Necessary reuse, not harmful duplication | In Section 1.8 label it explicitly as a recall used to set the constant flux, not a new derivation |

## Acceptance Criteria for the Flow Revision

The revised chapter will meet the Lecture 1 standard when:

- the first reader-facing paragraph is the observable and governing question;
- the required compact contract follows, rather than precedes, the hook;
- no schematic uses optical-depth language before optical depth is defined;
- no exact source name appears outside a compact contract or a genuine implementation/interface
  checkpoint;
- every advanced equation has an ordinary-language need and a concrete or limiting example before
  it;
- Planck limits are predictions checked by the plot, not terminology added afterward;
- \(J,H,K\), the Eddington boundary, and the Rosseland mean each gain one intermediate explanatory
  rung;
- every code output is followed by a physical interpretation and an honest statement of what it
  did not compute;
- the final paragraph explicitly answers the opening question at the level achieved;
- the Chapter 2 bridge names one concrete unresolved numerical/interface obstacle and explains why
  it must be solved before later physics can be added.

---

## Post-revision audit — 2026-07-30

### Resolved findings

- The observable now precedes all syllabus material (`chapter_01.py:18–34`), and the opening states
  both a durable central claim and the small atmosphere the reader will earn
  (`chapter_01.py:89–95`).
- The compact build/shape/dtype/device/honest-boundary contract now follows the hook rather than
  delaying it (`chapter_01.py:36–51`).
- The early synthetic `Spectrum` exercise and learned-initializer detour are gone. Atmosphere and
  synthesis are established first as physical tasks (`chapter_01.py:97–129`).
- The forward-problem schematic now follows the atmosphere/synthesis distinction, and its prose
  explicitly reads the arrows as a dependency order (`chapter_01.py:117–129`).
- Geometry and sign direction are established before the radiation and hydrostatic equations
  (`chapter_01.py:134–164`).
- Photon energy now creates the need for the Planck law, electron-volts are defined before output,
  and the output is read quantitatively (`chapter_01.py:169–231`).
- Rayleigh–Jeans and Wien behavior are predictions before `planck_bnu` is inspected and plotted
  (`chapter_01.py:263–278`). Tensor, dtype, device, and function-specific axis order are introduced
  just in time (`chapter_01.py:311–345`).
- The formation-depth schematic now appears only after mean free path, optical depth, column mass,
  and wavelength-dependent escape have been derived; the following paragraph interprets the
  drawing (`chapter_01.py:419–557`).
- The LTE slab closes attenuation’s missing-emission question, and its output creates the need for
  a depth-dependent temperature (`chapter_01.py:562–648`).
- The dense grey derivation now declares its mathematical altitude, begins with a concrete two-ray
  model, defines bolometric \(B(T)\), separates slope from boundary intercept, and explains the
  \(2/3\) constant (`chapter_01.py:658–782`).
- The Rosseland mean now follows a numerical two-window harmonic-mean example
  (`chapter_01.py:793–827`), and source-docstring archaeology has been removed.
- Radiation pressure now pays off the earlier \(K\)-moment discussion
  (`chapter_01.py:1034–1040`), and the final printed radiation-pressure fraction is immediately
  interpreted (`chapter_01.py:1159–1171`).
- The exact `ModelAtmosphere`, initializer-status, and `Spectrum` boundaries are delayed until the
  transparent scaffold exists (`chapter_01.py:1176–1226`).
- There is no detached exercise set. The worthwhile variations are embedded where they can be
  predicted and interpreted: three Planck temperatures, thin/thick slab limits, two opacity
  windows, and the unit-opacity hydrostatic limit.
- The chapter summary returns to the opening flux question, distinguishes computed from missing
  state, and the Chapter 2 link now names a genuine dependency: repeated depth integration without
  losing units, axes, or numerical order (`chapter_01.py:1246–1275`).

### Remaining flow blockers

#### P0

- None.

#### P1

- The strict code/output gate is still missed after the grey boundary check. The cell prints
  `T(tau_Ross=2/3)` and the inward-monotonic result (`chapter_01.py:939–954`), but Section 1.10
  begins immediately afterward. Add one sentence that reads the actual 5772 K result, confirms
  that both predicted invariants passed, and then states that temperature alone cannot support a
  layer.
- The five-row hydrostatic structure table is predicted before execution
  (`chapter_01.py:1084–1089`) but not read from its actual values afterward. The next paragraph
  merely announces that pressure spans many orders of magnitude and prepares another plot
  (`chapter_01.py:1112–1117`). Add a short reading of the outer, near-photospheric, and bottom rows
  before plotting; otherwise the table is a detached output.
- The summary introduces “correction” as a missing dependency
  (`chapter_01.py:1256–1259`) without having defined atmosphere temperature correction in the
  chapter. The close must contain no new concepts. Remove that word or name it earlier at the
  honest production boundary in plain language.

#### P2

- The first visible code cell mixes its one physical calculation with imports for later plots,
  Torch, and the not-yet-earned `PLANCK_PREFACTOR` (`chapter_01.py:196–223`). Move infrastructure
  imports to hidden setup or import the prefactor immediately before the exact Planck source
  checkpoint.
- The formation/escape transition is stated twice: “We still cannot say whether that radiation
  reaches space” (`chapter_01.py:411–414`) is immediately repeated by “It does not tell us whether
  a photon reaches space” (`chapter_01.py:421–423`). Keep the first as the causal bridge and let
  Section 1.6 begin directly with interaction probability.
- In the optical-depth section, scattering is named as part of extinction
  (`chapter_01.py:425–430`) before it is explained as redirection
  (`chapter_01.py:585–588`). Add the plain-language definition at first use.
- The direction cosine is written as \(\mu=\cos\theta\) (`chapter_01.py:516–520`) without stating
  that \(\theta\) is measured from the outward surface normal.

### New redundancy or premature-notation issues

- The exact `ModelAtmosphere` table is an appropriate late production checkpoint, but `metadata`
  and `fixed_column_abundance_values` (`chapter_01.py:1198–1199`) introduce external-format/deck
  vocabulary that the reader cannot use here. Either omit those two rows from the pedagogical
  table or add a single “file-boundary fields, deferred” grouping rather than two unexplained
  concepts.
- The Chapter 2 bridge is now causal, but its final roster—“scalar calculation, NumPy, Numba,
  parallel CPU work, and Torch devices” (`chapter_01.py:1268–1271`) introduces several names at
  once. This is acceptable as navigation, not explanation; Chapter 2 must begin by defining them
  from the depth-integral need and must not assume this preview taught them.

### Accept/revise verdict

**Revise, narrowly.** The chapter now meets the Agent4Astro/Lecture 1 causal-architecture standard
at section level, has no P0 blocker, uses its original figures at earned locations, contains no
detached exercise set, returns to the opening question, and gives Chapter 2 a genuine causal link.
Acceptance should wait for the three P1 fixes: interpret the grey-check output, read the structure
table before the pressure plot, and remove the new “correction” concept from the summary. The P2
items are polish and can be resolved in the same pass without changing the chapter’s architecture.
