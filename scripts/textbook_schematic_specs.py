"""Canonical prompt specifications for original textbook schematics.

The Payne Zero website's ``scripts/generate_physics_images.py`` established the
visual family.  This file adopts its auditable ``FigureSpec`` pattern and style
rules, but every textbook figure receives a new teaching claim, composition,
and label set.  Official website figures are visual references only.

The current rendered assets were created with Codex's built-in image-generation
tool.  This registry is backend-independent: it owns the prompts that should be
used when a figure is regenerated or revised.

Usage:
    python scripts/textbook_schematic_specs.py --list
    python scripts/textbook_schematic_specs.py ch01-forward-problem
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class FigureSpec:
    id: str
    chapter: int
    title: str
    prompt: str
    asset_path: str | None = None
    alt_text: str | None = None
    caption: str | None = None


STYLE = """
STYLE — use the Payne Zero website as a visual-family reference, not as a
composition to copy:
- Careful hand-drawn scientist's research-notebook illustration.
- Pure white background; no frame, title bar, or caption strip.
- Muted slate blue, soft charcoal, warm grey, and pale beige.
- Deep navy is reserved for the important dependency or direction.
- Slightly wobbly strokes and varied line weight, with a precise layout.
- Short, legible, correctly spelled English labels.
- Large explanatory objects and generous white space.
- Landscape composition unless the scientific idea requires otherwise.
- No photorealism, 3-D rendering, gradients, drop shadows, logos,
  watermarks, decorative filler, paragraphs, or invented numerical values.
- Return only the finished illustration.
"""


FIGURES = [
    FigureSpec(
        id="ch01-forward-problem",
        chapter=1,
        title="From a stellar photosphere to an emergent spectrum",
        prompt=f"""
Draw an original left-to-right explanation of the stellar-spectrum forward
problem. Show four large visual groups: (1) a small star with a thin
highlighted surface layer, labelled "stellar photosphere"; (2) an enlarged
stack of plane-parallel layers, labelled "atmosphere state"; (3) several
wavy rays passing upward through those layers with a small atom and opacity
mark, labelled "opacity + transfer"; and (4) a clean absorption spectrum on
axes, labelled "emergent spectrum". Connect the groups with one clear
deep-navy path. Beneath the first connection, add "physical conditions".
The star is context; the calculation begins in the one-dimensional layers.
Do not show atmosphere iteration, fitting, or a neural network.

{STYLE}
""",
    ),
    FigureSpec(
        id="ch01-formation-depth",
        chapter=1,
        title="Different wavelengths sample different atmospheric depths",
        prompt=f"""
Draw one enlarged side-on plane-parallel atmosphere as nine thin horizontal
layers. Label the upper edge "surface". At left, draw one downward arrow
labelled "optical depth increases inward". Show three upward escaping wavy
rays in distinct muted colors. Each begins at a visibly different
characteristic layer: a deeper ray labelled "continuum", a middle ray
labelled "weak line", and a higher ray labelled "strong line". All three
rays leave the same surface. The visual claim is that different wavelengths
sample different ranges of depth, not that photons follow exact tracks.
Do not include numerical values, axes, a stellar globe, or emission peaks.

{STYLE}
""",
    ),
    FigureSpec(
        id="ch02-ordered-depth",
        chapter=2,
        title="Independent wavelengths and ordered depth",
        prompt=f"""
Draw a rectangular wavelength-by-depth array as five horizontal rows,
labelled lambda 1 through lambda 5. Each row begins at a circle labelled
"surface seed" and advances through eight layer boxes toward an arrow
labelled "inward". Arrows connect consecutive layers within a row; no arrow
connects different rows. A vertical bracket at the left spans all five rows
and says "independent wavelengths". A horizontal bracket below the layer
boxes says "ordered depth". A tiny legend identifies each horizontal arrow
as "depends on previous layer". The composition must make the recurrence
direction and the independent batch axis unmistakable.

{STYLE}
""",
    ),
    FigureSpec(
        id="ch02-architecture",
        chapter=2,
        title="Actual CPU and device acceleration lanes",
        prompt=f"""
Draw three separated horizontal computation lanes. The first shows one CPU
depth column passing from NumPy to njit and then through an ordered loop,
labelled "atmosphere iteration — CPU only". The second shows many CPU
frequency chunks, one private accumulator per chunk, a prange bracket, and a
fixed-order reduction. The third shows a wavelength-by-depth tensor entering
"Torch synthesis" on a device with badges CUDA, MPS, and CPU, then a
spectrum. Add one small violet side card "initializer — Torch start only"
with a dotted arrow to a CPU atmosphere seed. Do not draw a GPU atmosphere
iteration path.

{STYLE}
""",
    ),
    FigureSpec(
        id="ch02-data-roles",
        chapter=2,
        title="Four data roles and one-way golden comparison",
        prompt=f"""
Draw four clearly separated trays. "static" contains an atomic table with
columns Z, level, and energy and is labelled "physical inputs". "subset"
contains clipped spectral line records with wavelength, species, and log gf
and is labelled "teaching slice". "fixture" contains a one-dimensional
stellar-atmosphere column with outer-to-inner direction and small T and P
tracks and is labelled "supplied upstream state". "golden" contains a
sealed check card and is labelled "comparison only". Solid arrows from
static, subset, and fixture enter a calculation. No arrow leads from golden
into the calculation; instead the calculation output enters a comparison
gate beside golden, and a separate solid arrow from golden enters that
comparison gate. A grey cache below has a dashed loop labelled "rebuildable".

{STYLE}
""",
    ),
    FigureSpec(
        id="ch02-populations",
        chapter=2,
        title="Actual and partition-normalized populations",
        prompt=f"""
Draw two equal-size transparent array cubes side by side. On both cubes,
label the diagonal direction "axis 0: depth" with extent D; label the
vertical direction "axis 1: ion stage" with zero-based indices 0 through 5;
and label the horizontal direction "axis 2: species" with zero-based indices
0 through 138. Label the left cube "actual population n", cm^-3, shape
(D, 6, 139). Between them place a card "divide by U". Label the right cube
"partition-normalized n/U", cm^-3, with the same shape. Beneath the right
cube add the warning "not a bound-level population". Show the small worked
relation n=10^12, U=4, n/U=2.5 x 10^11. End with the short statement
"same shape is not the same meaning".

{STYLE}
""",
    ),
    FigureSpec(
        id="ch03-levels-to-charge",
        chapter=3,
        title="Level counting, ion populations, and charge closure",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original explanation with five clearly separated scientific groups.
At left, show a small energy-level ladder with several horizontal levels and
the short labels "level energy E_i" and "weight g_i". Connect it to a compact
state-counting card labelled "partition U". Connect that card to an ion ladder
with three circles labelled "q = 0", "q = 1", and "q = 2", adjacent arrows
labelled "Saha ratios", and the short label "populations n_q(n_e)". Connect
the populated ion ladder to a final balance mark labelled "charge sum".

Below the space between the ion ladder and charge sum, add a distinct rounded
box labelled "electron density n_e". Draw one deep-navy arrow from "charge sum"
to this box, labelled "update n_e". Draw a second deep-navy arrow from the
electron-density box to a bracket spanning all three adjacent-stage Saha
relations, labelled "enters every Saha ratio". Neither feedback arrow may
terminate at only q = 0, q = 1, or q = 2. The feedback arrows
must not point to the level ladder or directly to "partition U". The forward path must remain levels
and weights to partition U to the complete ion ladder to charge sum. This is
one atom-only LTE dependency diagram, not an iterative atmosphere diagram.
Keep equations to the displayed symbols; do not add numerical values,
molecules, depth layers, spectra, or a raw code flowchart.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch03-levels-to-charge-v1.png"),
        alt_text=(
            "Original hand-sketched conceptual diagram in which level energies "
            "and statistical weights build a partition function, adjacent-stage "
            "Saha ratios set all ion populations using an explicit electron-"
            "density node, and the resulting charge sum updates that node."
        ),
        caption=(
            "<strong>Conceptual dependency schematic.</strong> Level energies "
            "and weights define the partition sum; partition functions enter "
            "the adjacent-stage Saha ladder; and the retained ion populations "
            "supply a charge sum that updates the explicit "
            "<em>n</em><sub>e</sub> node. That electron density enters every "
            "adjacent-stage ratio rather than one selected charge state. The "
            "feedback does not alter the static level data. A production "
            "evaluation may also apply its declared density-dependent "
            "partition corrections."
        ),
    ),
    FigureSpec(
        id="ch03-packed-to-public",
        chapter=3,
        title="Explicitly mapping the packed atmosphere state",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original left-to-right explanation with three large scientific groups.
At left, show a wide sparse array sheet labelled "packed atmosphere state" and
"shape (D, 1006)". Within each depth row, mark a few separated occupied slots
with clear gaps between them so the storage is visibly sparse and nonuniform.
In the middle, draw a narrow mapping funnel labelled "explicit index map" and
"gather + place". Several arrows must leave separated packed slots, cross
without merging into one contiguous block, and enter specific destinations in
the output.

At right, draw one transparent public array cube labelled "shape (D, 6, 139)".
Label its axes "depth D", "ion stage 0–5", and "species 0–138". Divide only
the species direction into a slate-blue region labelled "atomic species 0–98"
and a pale beige region labelled "unused tail 99–138 in this atom-only route".
Put the short warning "not a reshape" directly beneath the mapping funnel.
Add a small outlined callout pointing to the stage-index-5 plane, not to the
unused tail, labelled "later: selected synthetic line cells". The composition
must communicate lookup by slot meaning and explicit placement. It must not
imply that molecules are confined to columns 99–138. Do not draw a single
broad reshape arrow, equal-size source and destination grids, contiguous
source slices, copied molecule values, or invented numerical data.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch03-packed-to-public-v1.png"),
        alt_text=(
            "Original hand-sketched conceptual diagram showing separated values "
            "gathered from a sparse atmosphere array of shape D by 1006 and "
            "explicitly placed into a public depth-by-ion-stage-by-species array "
            "of shape D by 6 by 139; the atom-only route writes species columns "
            "0 through 98, leaves the tail 99 through 138 unused, and marks "
            "selected stage-index-5 cells that are later overloaded for "
            "synthetic molecular-line support."
        ),
        caption=(
            "<strong>Representation schematic.</strong> A decoder interprets "
            "the meaning of selected packed slots and explicitly places their "
            "values on the public axes; the conversion is not a raw reshape. "
            "The atom-only bridge writes species-axis columns 0–98 and leaves "
            "the tail 99–138 empty. That empty tail is not the complete "
            "molecular address rule: Chapter 4 will derive selected normalized "
            "stage-5 cells that are later overloaded for synthetic molecular-"
            "line support across the public species axis."
        ),
    ),
    FigureSpec(
        id="ch04-coupled-budgets",
        chapter=4,
        title="One CO molecule couples two elemental budgets",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape explanation for a senior-undergraduate stellar
spectroscopy textbook. On the left, show two separate shallow reservoirs:
one labelled "free carbon x_C" containing several small C tokens, and one
labelled "free oxygen x_O" containing several O tokens. In the center, show
one clearly paired C—O token labelled "CO population n_CO". Draw one
deep-navy arrow from each free reservoir into the paired CO token. On the
right, draw three compact ledger cards labelled "particle budget",
"carbon budget", and "oxygen budget". Draw an arrow from the same CO token
to all three ledger cards. On the carbon and oxygen cards, show one matching
C or O tally mark contributed by each CO token. Add the short statement
"one molecule, several coupled equations".

The visual claim is conservation coupling: forming one CO consumes one free
C and one free O and contributes to several residual rows. Do not show a
reaction rate, a time arrow, photons, a spectrum, a software flowchart, a
neural network, or numerical values.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch04-coupled-budgets-v1.png"),
        alt_text=(
            "Original hand-sketched schematic in which free carbon and oxygen "
            "tokens combine into one CO population that contributes "
            "simultaneously to particle, carbon, and oxygen conservation "
            "ledgers."
        ),
        caption=(
            "<strong>Why molecular equilibrium is coupled.</strong> Forming "
            "one CO molecule spends one free carbon nucleus and one free "
            "oxygen nucleus. The same molecular population therefore appears "
            "in the particle ledger and in both elemental ledgers, so those "
            "densities must be solved together."
        ),
    ),
    FigureSpec(
        id="ch04-newton-positivity",
        chapter=4,
        title="One Newton step, two exact positivity policies",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape flow diagram beginning with one card labelled
"conservation residual R", then fork immediately into two horizontal lanes.

The upper lane is labelled "atmosphere — ordinary number densities" and has
this exact left-to-right order: "analytic Jacobian J", "solve J delta = R",
"0.69 sign-history damping of delta", "candidate x - delta",
"absolute reflection + shared scale", and "positive returned density".

The lower lane is labelled "synthesis — ordinary number densities" and has
this exact left-to-right order: "jacrev Jacobian J",
"density-column-scaled solve", "0.69 sign-history damping of delta",
"candidate x - delta", "current density / 100 floor", and
"positive returned density".
The current-density/100 rule is the exact multiplicative 1% floor.

Use deep navy for the shared residual and atmosphere lane, and warm
beige-charcoal for the synthesis lane. The visual claim is that both backends
use ordinary density unknowns and damp the correction before forming a
candidate, while their Jacobian, scaling, and positivity policies differ.
Do not show logarithmic unknowns, gradient descent, a loss function, convergence
curves, code listings, or invented numerical results.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch04-newton-positivity-v1.png"),
        alt_text=(
            "Original hand-sketched conceptual Newton-flow schematic in which "
            "a conservation residual forks into atmosphere and synthesis "
            "density-space lanes; each lane damps the correction before "
            "forming a candidate, then applies its distinct positivity rule."
        ),
        caption=(
            "<strong>Conceptual map: shared physics, distinct update "
            "policy.</strong> Both "
            "backends solve conservation equations in ordinary number "
            "densities. They share sign-reversal damping, but the atmosphere "
            "route uses reflection and an order-sensitive shared scale while "
            "the synthesis route uses a multiplicative one-percent floor."
        ),
    ),
    FigureSpec(
        id="ch04-ordered-backends",
        chapter=4,
        title="Molecular Newton solves remain ordered through depth",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape diagram with two horizontal depth chains. The
upper chain is labelled "atmosphere — CPU float64". It contains five layer
cards labelled "depth 0", "depth 1", "depth 2", "...", and "depth D-1",
joined by one-way arrows from outer to inner depth. Place a small note inside
one representative upper card: "Numba local kernels"; place a small adjacent
card outside that box: "CPU linear solve".

The lower chain is labelled "synthesis — device-capable tensors" and has the
same five ordered depth cards and arrow direction. Inside one representative
lower card, stack the short labels "safe-log products", "jacrev", and
"scaled solve". After the final lower depth card, draw a separate bracket
over a small batch of population tokens labelled "vmap after depth solves".
Add one concise central note between the chains:
"previous converged depth seeds the next".

The visual claim is that both molecular Newton calculations are sequential in
depth even though their local kernels differ; vectorized population evaluation
occurs only after the chain. Do not draw arrows between different depths in
parallel, an all-depth Newton box, prange, a GPU atmosphere path, spectra,
timing numbers, or decorative hardware.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch04-ordered-backends-v1.png"),
        alt_text=(
            "Original hand-sketched diagram with ordered outer-to-inner depth "
            "chains for the CPU atmosphere and device-capable synthesis "
            "molecular solvers; Numba and CPU linear algebra appear locally "
            "above, jacrev inside one synthesis depth, and vmap only after all "
            "depth solves."
        ),
        caption=(
            "<strong>Depth is a recurrence, not a batch axis.</strong> Both "
            "molecular solvers seed each depth from the preceding converged "
            "layer. Numba accelerates local atmosphere arithmetic; synthesis "
            "uses automatic differentiation inside one depth. Its later "
            "population evaluation may be vectorized only after the ordered "
            "Newton chain has finished."
        ),
    ),
    FigureSpec(
        id="ch04-catalog-to-public-lane",
        chapter=4,
        title="Chemistry catalogs and the public synthetic line lane",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape explanation with two distinct paths that must not
merge by row number. At left, show a rounded slate-blue set labelled
"atmosphere catalog — 170 records" fully inside a larger pale set labelled
"synthesis catalog — 190 records". In the surrounding part of the larger
set, add "+20 synthesis-only records". Mark the shared region
"matched by molecule code + semantics", never by row.

Below those sets, show a separate card labelled
"54 line-list species codes". From this card, draw a deep-navy arrow through
a mapping card containing exactly
"column = species_code // 6 - 1"
and then into an exploded stack of six thin two-dimensional array sheets
labelled stage indices 0 through 5. Pull only the stage-5 sheet forward and
label it "normalized cube only — stage index 5". Its vertical axis is
"depth" and its horizontal axis is "species column", so the fixed-stage sheet
spans depth by species. On that sheet, mark several discrete narrow selected
columns rather than coloring the whole sheet, and label them
"54 selected columns". Add the quiet callout
"51 inside 0–98; 3 at 129–131". Beside the stack, place a separate protected
card labelled "actual ion-stage cube unchanged". Add one small worked trace
beside the mapping arrow, rendered verbatim:
"276 -> CO 608 -> [depth, 5, 45]".
Finish with the short warning "catalog row is not a public column".

The visual claim is that chemistry records are aligned by code and semantics,
while molecular line populations use a separate species-code mapping into
selected stage-5 cells of the normalized cube. The actual ion-stage cube stays
atomic. Do not confine the mapping to columns 99–138, do not highlight a
constant-depth sheet as stage 5, do not color an undifferentiated whole plane,
do not show a reshape arrow, do not equate 170 or 190 with public columns, and
do not invent molecule codes or values.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch04-catalog-to-public-lane-v1.png"),
        alt_text=(
            "Original hand-sketched schematic showing a 170-record atmosphere "
            "molecular catalog nested inside a 190-record synthesis catalog "
            "by code and semantics, plus a separate mapping from 54 line-list "
            "species codes into selected stage-index-5 cells of only the "
            "normalized public cube, while the actual ion-stage cube remains "
            "unchanged; the CO trace runs from 276 to 608 to depth, 5, 45."
        ),
        caption=(
            "<strong>Two identifiers, two jobs.</strong> Molecular-equilibrium "
            "records align by molecule code and record semantics: all 170 "
            "atmosphere records are shared and synthesis adds 20. Separately, "
            "54 line-list species codes select normalized molecular "
            "populations and place them in selected stage-index-5 cells at "
            "species-code-derived public columns: 51 inside 0–98 and three at "
            "129–131. The actual ion-stage cube remains atomic and unchanged. "
            "A catalog row is never a public address."
        ),
    ),
    FigureSpec(
        id="ch05-cross-section-to-opacity",
        chapter=5,
        title="From one microscopic interaction to opacity per gram",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape explanation with one left-to-right causal path.
At far left, show one wavy photon approaching a single absorber surrounded by
a small outlined effective area labelled "cross section sigma_nu" and
"cm^2". Next show a transparent cubic centimetre containing several identical
absorbers, labelled "number density n" and "cm^-3". Next show a short path
through that volume labelled "n sigma_nu" and "cm^-1"; a separate annotation
may state that the thin-slice interaction probability is approximately
"n sigma_nu ds". At far right, place
the path inside a small weighed parcel of stellar gas, with a division card
labelled "divide by mass density rho", leading to a final card labelled
"mass opacity kappa_nu = n sigma_nu / rho" and "cm^2 g^-1".

Use one deep-navy arrow through all four stages. The dimensional chain must
be exactly "cm^2 × cm^-3 = cm^-1" followed by
"cm^-1 ÷ (g cm^-3) = cm^2 g^-1". Never divide the dimensionless thin-slice
probability by mass density. Make the unit cancellation visually legible
without adding numerical values. The scientific claim is
probability for one particle becoming interaction per length and then per
gram. Do not draw a spectral line, a software pipeline, a neural network, a
star, or a radiative-transfer solution.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch05-cross-section-to-opacity-v2.png"),
        alt_text=(
            "Original hand-sketched schematic showing a photon cross section "
            "combined with absorber number density to form an inverse-length "
            "coefficient, then divided by mass density to give mass opacity "
            "in square centimetres per gram."
        ),
        caption=(
            "<strong>From one interaction to one gram of gas.</strong> A "
            "microscopic cross section becomes an inverse mean path after "
            "multiplication by absorber number density. A path length then "
            "turns that inverse length into a thin-slice probability, but the "
            "mass-opacity chain divides the inverse-length coefficient—not "
            "that dimensionless probability—by mass density."
        ),
    ),
    FigureSpec(
        id="ch05-absorption-vs-scattering",
        chapter=5,
        title="Absorption transfers energy; scattering redirects",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape fork beginning with one incoming wavy photon at
left. The upper branch is labelled "true absorption": the photon ends at a
small atom and the atom gains a warm-orange energy mark. From that absorption
event, fork to two sibling consequences: a tray labelled "kappa_abs" and a
separate card labelled "kappa_abs S" with subtitle "thermal source numerator".
There must be no arrow between those sibling outputs. The lower branch is
labelled "coherent scattering": the photon leaves in a new direction with the
same wavelength and the event feeds only a tray labelled "kappa_sca". On the
right, draw one arrow from the kappa_abs tray and one from the kappa_sca tray;
only those two arrows converge at a circled plus node, which points to a card
labelled "total extinction" and "kappa_abs + kappa_sca". Use no enclosing
brace. Keep the thermal source numerator entirely outside that extinction
path.

The claim is that both branches remove a photon from the original ray, but
only true absorption deposits energy and enters the LTE thermal numerator.
Do not imply scattering is zero opacity. Do not show frequency redistribution,
line profiles, numerical values, code, or a finished spectrum.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch05-absorption-vs-scattering-v3.png"),
        alt_text=(
            "Original hand-sketched fork in which true absorption deposits "
            "energy and produces sibling absorption-opacity and "
            "absorption-weighted source records, while coherent scattering "
            "redirects the photon and contributes only scattering opacity."
        ),
        caption=(
            "<strong>Two ways to leave the original ray.</strong> True "
            "absorption has two bookkeeping consequences: it contributes "
            "<em>κ</em><sub>abs</sub> to extinction and contributes its "
            "absorption-weighted source to the separate thermal numerator. "
            "Scattering still contributes to extinction, but it redirects "
            "rather than thermally creates a photon."
        ),
    ),
    FigureSpec(
        id="ch05-prange-columns",
        chapter=5,
        title="Why frequency columns can use prange",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape dependency diagram. At left, show one shared
read-only tray labelled "depth state + tables". Across the top, draw five
separate worker cards labelled "frequency 1" through "frequency 5", enclosed
by one horizontal bracket labelled "prange". A read-only data bus from the
shared tray fans upward to the five cards without crossing the array. Directly
below the cards, draw one depth-by-frequency opacity array with five vertical
frequency columns. Each worker has one short vertical arrow landing at the top
of its matching complete column; no connector may point at an individual
depth row. Use distinct column highlights spanning depth 0 through D-1.
Beneath the array, show a small ordered addition strip labelled
"component assembly outside prange".
Add the short statements "no shared writes" and "depth work stays inside one
frequency owner".

The claim is disjoint write ownership, not generic parallel speed. Do not
show depth layers running independently, a GPU, Torch, a fixed-order reduction
between frequency workers, or performance numbers.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch05-prange-columns-v4.png"),
        alt_text=(
            "Original hand-sketched diagram showing read-only atmosphere state "
            "feeding independent prange frequency workers, each of which "
            "writes one disjoint complete depth column before ordered "
            "component assembly."
        ),
        caption=(
            "<strong>The safe parallel axis is frequency.</strong> Every "
            "worker reads the same immutable depth state and tables but owns "
            "one matching complete vertical output column from depth 0 through "
            "depth <em>D</em>−1; no arrow assigns an individual depth row. Ordered "
            "component assembly remains outside the independent-frequency loop."
        ),
    ),
    FigureSpec(
        id="ch05-two-grids-edge-triplet",
        chapter=5,
        title="Direct atmosphere sampling and synthesis edge triplets",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape diagram that begins with one small input card
labelled "consumer views + owned tables fork conceptually" and forks into two
clearly separated computational lanes. The consumers retain separate declared
views, process sets, and table bundles; do not imply one shared mapping or one
shared table object. The upper lane is labelled
"atmosphere — CPU float64" and shows a long logarithmic wavelength ruler with
many qualitative marks, labelled "30,000 direct evaluations". Remove numerical
tick values; label only "shorter wavelength" and "longer wavelength". The
lower lane is labelled "synthesis — requested window" and shows a sparse
edge ruler, a highlighted set of only the used intervals, and arrows from
three samples per used interval to a denser requested-pixel ruler labelled
"log-opacity interpolation".

Enlarge two adjacent synthesis intervals in a clean callout. Label their three
exact black boundary ticks "lambda_L", "lambda_R", and "lambda_(R+1)".
Highlight the first interval and show its three sample dots labelled
"lambda_L + epsilon", "lambda_M", and "lambda_R - epsilon". Show the second
interval to the red-wavelength side of the exact lambda_R boundary. Point an
assignment arrow at the exact lambda_R boundary, never at a sample dot, and
write "lambda = lambda_R maps to [lambda_R, lambda_(R+1))". Use direction
arrows or a tiny red/blue wavelength cue so the one-sided offsets and the
red-side interval cannot be reversed.

The claim is two exact consumers of shared physical ideas, not a CPU-versus-
GPU approximation hierarchy. Do not imply that the atmosphere uses edge
interpolation, that synthesis evaluates every requested pixel physically, or
that the two product arrays should be equal.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch05-two-grids-edge-triplet-v4.png"),
        alt_text=(
            "Original hand-sketched schematic showing two consumer views and "
            "their owned tables forking conceptually into a direct "
            "thirty-thousand-point atmosphere grid and synthesis evaluation at "
            "three one-sided samples in each used continuum-edge interval "
            "followed by log-opacity interpolation to requested pixels."
        ),
        caption=(
            "<strong>Two exact grids, not interchangeable backends.</strong> "
            "Both routes consume declared state and tables, but their process "
            "sets remain distinct. The atmosphere evaluates every enabled "
            "atmosphere process directly on its 30,000-point CPU grid. "
            "Synthesis evaluates three one-sided "
            "samples only in used edge intervals and reconstructs positive "
            "opacity on the requested wavelength pixels. At an exact internal "
            "edge, right-sided search assigns the pixel to the next interval "
            "on the red-wavelength side."
        ),
    ),
    FigureSpec(
        id="ch06-smooth-background-narrow-line",
        chapter=6,
        title="A smooth background cannot explain one narrow opacity excess",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape scientific illustration that poses one question
without pretending to be a measured spectrum. Across the lower half, draw one
quiet, broad slate-blue curve labelled "continuum mass extinction". Above
that smooth background, add one narrow deep-navy peak at a single wavelength
and label it "line mass absorption". The two curves share one simple
horizontal wavelength guide and the implied same unit, but do not merge their
labels or processes. Mark one off-line wavelength with a small pale-beige
pointer and the line-center wavelength with a deep-navy pointer. From the
narrow peak, draw one restrained question arrow toward exactly two horizontal
bound energy levels at upper right, with one short label:
"which transition adds this opacity?"

The claim is only that a smooth continuum background cannot create a narrow
localized opacity excess, so a bound-bound interaction is now required. The
curves and axes are conceptual and carry no numerical scale. Do not draw an
observed flux dip, emergent intensity, a normalized spectrum, line wings,
catalogs, many lines, a Voigt profile, source code, a star, a GPU, numerical
ticks, or invented values.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch06-smooth-background-narrow-line-v1.png"
        ),
        alt_text=(
            "Original hand-sketched conceptual schematic with a smooth "
            "continuum mass-extinction curve, one narrow line "
            "mass-absorption excess at line center, an off-line marker, and "
            "a question arrow from the excess to two bound energy levels."
        ),
        caption=(
            "<strong>A narrow opacity excess needs a new interaction.</strong> "
            "The supplied continuum mass extinction changes smoothly across "
            "this conceptual wavelength interval. A localized line mass "
            "absorption contribution has the same units but a different "
            "physical origin. Neither curve is an emergent or observed flux."
        ),
    ),
    FigureSpec(
        id="ch06-two-levels-one-photon",
        chapter=6,
        title="Two bound levels define one resonant photon",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape scientific illustration with three connected
visual groups. At left, draw a simple atom as a small nucleus with one
bound electron orbit, beside the short note "bound before and after". In the
center, draw exactly two horizontal energy levels. Label the lower line
"lower level E_l" and place a small cluster of three particles on it. Label
the upper line "upper level E_u". Draw one deep-navy upward wavy arrow from
the lower level to the upper level, labelled exactly "photon h nu_l". At
right, draw a short wavelength ruler with one narrow vertical marker labelled
"lambda_l = c / nu_l". Connect the energy-level gap to that marker with one
quiet slate arrow.

The claim is that the level separation fixes the resonant photon and line
center, while absorption requires particles in the lower state. Keep the
energy spacing, particle count, and wavelength width visibly conceptual.
Do not draw line depth, broadening, damping wings, a catalog, a spectrum with
many lines, numerical values, or software objects.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch06-two-levels-one-photon-v1.png"),
        alt_text=(
            "Original hand-sketched schematic with particles on a lower bound "
            "energy level, one resonant photon lifting a particle to an upper "
            "bound level, and the same level separation connected to a narrow "
            "rest-wavelength marker."
        ),
        caption=(
            "<strong>Two levels define the line center.</strong> A bound "
            "particle can absorb only the photon energy that spans the lower "
            "and upper states. The drawing is conceptual: its level spacing, "
            "particle count, and wavelength-marker width are not to scale."
        ),
    ),
    FigureSpec(
        id="ch06-core-wings-convolution",
        chapter=6,
        title="Independent velocity and damping causes meet in one profile",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape scientific diagram with two clearly independent
input branches that meet only at one convolution mark. The upper-left branch
begins with small atoms carrying differently directed velocity arrows and the
label "thermal + microturbulent velocities". It leads to a compact symmetric
bell curve labelled "Gaussian core". The lower-left branch begins with one
excited atom beside three sparse perturbation symbols labelled
"radiative", "electron", and "neutral". It leads to a narrow central response
with long tails labelled "Lorentzian response". Route both branches into one
large hand-drawn convolution symbol. From it, draw one output profile with a
rounded center labelled "core" and extended low tails labelled "wings".
Beneath the two inputs and output, draw three equal-length pale-beige area
ribbons joined by a thin guide and label the guide "equal ideal area".

The claim is that velocity broadening and damping have different physical
causes but combine by convolution, redistributing a fixed ideal continuous
profile area. Do not imply that a finite production cutoff preserves the
deposited area. Do not draw a Voigt lookup table, branch thresholds, source
code, a catalog, multiple spectral lines, or numerical values.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch06-core-wings-convolution-v1.png"),
        alt_text=(
            "Original hand-sketched two-branch schematic in which thermal and "
            "microturbulent velocities make a Gaussian core, radiative and "
            "collisional damping make a Lorentzian response, and convolution "
            "produces one profile with a core and extended wings."
        ),
        caption=(
            "<strong>One profile, two independent causes.</strong> Velocities "
            "spread resonant frequencies into a Gaussian core; finite lifetime "
            "and charged or neutral perturbers create damping wings. Their "
            "convolution preserves the ideal continuous area. A later "
            "continuum-relative production cutoff is a separate numerical "
            "operation."
        ),
    ),
    FigureSpec(
        id="ch06-one-record-many-depths",
        chapter=6,
        title="One immutable transition changes with atmospheric depth",
        prompt=f"""
Use case: scientific-educational
Asset type: revision of an owned conceptual textbook schematic
Edit the supplied owned prior revision of the landscape scientific
illustration. Preserve its entire causal composition: one immutable line
record at left, a deep-navy bus into exactly six outer-to-inner atmospheric
layers, six differently shaped opacity rows sharing one wavelength-center
guide, and one depth-by-wavelength slab at right.

Make two precise changes only. First, on the left record card replace the
entry "E_l" with the exact, visibly unambiguous entry
"tilde E_l  [cm^-1]". Keep the other three entries "lambda_l", "gf", and
"damping data". Second, improve narrow-screen legibility: enlarge the record
entries and every repeated layer-state strip materially, tighten unused white
space between the record and the six layers, and keep the state tokens exactly
"T, rho, n_e, n_pert, n/U, delta". All six repeated strips must remain
readable and must not overlap their boxes.

The claim is that the source record and rest wavelength remain fixed while
the local atmosphere changes strength, width, and damping from layer to
layer. Row colors and shapes are conceptual, not numerical parity results.
Do not draw catalog selection, many-line accumulation, molecular bands,
radiative transfer, emergent flux, a GPU, or performance annotations.

{STYLE}
""",
        asset_path=("assets/schematics/textbook/ch06-one-record-many-depths-v2.png"),
        alt_text=(
            "Original hand-sketched schematic showing one fixed atomic-line "
            "record, including its stored lower-excitation wavenumber, feeding "
            "six atmospheric layers whose temperatures, densities, "
            "populations, collision conditions, and Doppler widths produce "
            "differently shaped opacity rows aligned into a "
            "depth-by-wavelength slab."
        ),
        caption=(
            "<strong>The record is fixed; the atmosphere is not.</strong> "
            "Every layer uses the same transition wavelength, stored "
            "lower-excitation wavenumber \\(\\tilde E_l=E_l/(hc)\\), "
            "oscillator strength, and damping data. Local population, "
            "density, temperature, collisions, and Doppler width change the "
            "resulting row of the depth-by-wavelength line-opacity slab. "
            "Shapes and colors here are conceptual."
        ),
    ),
    FigureSpec(
        id="ch07-catalog-to-grid",
        chapter=7,
        title="A raw atomic catalog becomes a computation-ready catalog",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw a left-to-right scientific story showing how a raw atomic line catalog
becomes a computation-ready catalog. At left, show a tall paper ledger with
many short rows and tiny symbols for wavelength, species, energy, strength,
and damping. In the middle, show three physical transformations rather than
software boxes: two energy levels setting a refined wavelength, an isotope
pair adjusting line strength, and an unfilled damping entry being completed
from atomic physics. Next, show a narrow illuminated wavelength window
selecting both line centers inside it and one broad-wing line whose center
lies just outside. At right, show the selected records reorganized into clean
parallel rails labelled only "type" and "center", leading toward a wavelength
grid.

The lesson is that raw fields are decoded, physically completed,
conservatively selected, and arranged for locality before deposition. Do not
imply that a line is discarded merely because its center is outside the
window; visibly preserve one outside-center record because its broad wing
reaches the window. The right-hand rails are data organization, not a
spectrum. Use only these labels: "raw records", "energy", "isotope",
"damping", "wavelength window", "type", and "center".

{STYLE}
""",
        asset_path="assets/schematics/textbook/ch07-catalog-to-grid-v1.png",
        alt_text=(
            "A raw atomic ledger passes through energy, isotope, damping, and "
            "conservative wavelength-window decisions before records are "
            "arranged by route type and grid center; one broad outside-center "
            "line is retained because its wing reaches the window."
        ),
        caption=(
            "<strong>From source rows to usable records.</strong> A raw row is "
            "decoded and physically completed before a conservative window "
            "keeps reachable wings and the selected records are organized for "
            "accumulation. The final rails indicate data locality, not a "
            "spectrum."
        ),
    ),
    FigureSpec(
        id="ch07-scatter-add-reduction",
        chapter=7,
        title="Overlapping line contributions add without races",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw several distinct narrow and broad line profiles at left, each sending a
curved arrow toward the same long wavelength-pixel grid. Make two or three
arrows meet the same pixel and place a clear plus sign beside that landing
point so the meaning is addition, never replacement. Below, show four short
private work ribbons labelled "private", computed independently and combined
by one reduction arrow into a ribbon labelled "sum". At right, show the
accumulated forest as a positive opacity comb labelled "line opacity".

The lesson is that coincident contributions must be scatter-added and CPU
parallel work is race-free when every worker owns a private accumulator before
a defined reduction. Every profile and opacity peak must stay above its
baseline. Use only "private", "sum", and "line opacity"; "private" may repeat.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch07-scatter-add-reduction-v1.png"
        ),
        alt_text=(
            "Several line profiles target shared wavelength pixels where their "
            "values add; separate private CPU opacity ribbons then reduce in a "
            "fixed order into one positive line-opacity forest."
        ),
        caption=(
            "<strong>No lost overlaps.</strong> Several records may target one "
            "wavelength cell, so their values add. Parallel CPU chunks instead "
            "write private slabs and a later ordered reduction combines them. "
            "The claim is no lost update, not identical floating-point "
            "association on every backend."
        ),
    ),
    FigureSpec(
        id="ch07-profile-routing",
        chapter=7,
        title="Different atomic record families require different profiles",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw one source catalog at left whose records fan into four physically
distinct opacity routes. "ordinary" shows a two-level atom over a symmetric
Doppler-core plus damping-wing profile. "H + D" shows closely spaced
fine-structure levels, an isotope pair, and charged perturbers over one broad
positive profile. "He" shows helium levels and charged perturbers over a
broad positive profile tapering gently into a continuum edge.
"autoionizing" shows a discrete level coupled to a shaded continuum over an
asymmetric but nonnegative profile. The four routes rejoin at one shared
baseline labelled "atomic opacity".

The lesson is that a route code selects profile physics; eligible routes add
to one opacity budget but must not be forced through one universal profile.
H and D should be close but not identical. Keep every profile positive. Use
only "ordinary", "H + D", "He", "autoionizing", and "atomic opacity".

{STYLE}
""",
        asset_path="assets/schematics/textbook/ch07-profile-routing-v1.png",
        alt_text=(
            "One atomic source catalog fans into ordinary, hydrogen and "
            "deuterium, helium, and autoionizing profile routes whose positive "
            "contributions rejoin one shared atomic-opacity budget."
        ),
        caption=(
            "<strong>One catalog, several profile families.</strong> Ordinary "
            "lines, H and D, helium, and autoionizing states keep the profile "
            "physics their source meanings require. They reunite only by "
            "addition to the atomic-opacity slab."
        ),
    ),
    FigureSpec(
        id="ch08-one-molecule-band",
        chapter=8,
        title="One molecular population feeds a band of transitions",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw one original left-to-right explanation of why one molecule produces many
nearby absorption lines. At left, show a simple two-atom molecule with one
label, "one molecule". In the middle, show three vertically separated
electronic energy shelves. On two shelves, draw a few vibrational sublevels;
on each vibrational sublevel, draw a small fan of rotational ticks. Use only
the short labels "electronic", "vibrational", and "rotational". Draw several
thin transition arrows between neighboring families of ticks. At right, turn
those arrows into a compact cluster of many downward absorption sticks on one
baseline labelled "molecular band".

The sole claim is one supplied species population can feed many transitions
because electronic, vibrational, and rotational structure is nested. The
spacings are conceptual, not measured. Do not show catalog files, quantum
numbers, numerical values, a spectrum continuum, radiative transfer, a star,
or a computer.

{STYLE}
""",
        asset_path="assets/schematics/textbook/ch08-one-molecule-band-v1.png",
        alt_text=(
            "One molecule branches into electronic shelves with vibrational "
            "sublevels and rotational ticks; many allowed transitions become "
            "a cluster of absorption sticks called a molecular band."
        ),
        caption=(
            "<strong>One population, many transitions.</strong> Electronic, "
            "vibrational, and rotational structure creates many nearby line "
            "centers from one molecular species. The level spacings and stick "
            "heights are conceptual."
        ),
    ),
    FigureSpec(
        id="ch08-encodings-to-record",
        chapter=8,
        title="Unlike molecular encodings converge on common fields",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw three source cards at left and one common structure-of-arrays record at
right. The first source card is a paper row labelled "text band" with tiny
column marks. The second is a 16-byte-style strip labelled "packed TiO" with
seven uneven field blocks. The third is a shorter strip labelled "signed H2O"
with one long block and two signed short blocks, visibly showing plus and
minus marks. Each source passes through its own small decoding funnel; do not
merge the funnels early. The three paths converge only at a clean set of
parallel array rails at right. Label those rails with exactly these compact
terms: "center", "strength", "species", "excitation", "damping", and
"margin".

The claim is semantic convergence after family-owned decoding, not one
universal parser and not byte-level archaeology. Do not show a GPU, opacity
profile, spectrum, manifest order, file tree, or invented numbers.

{STYLE}
""",
        asset_path="assets/schematics/textbook/ch08-encodings-to-record-v1.png",
        alt_text=(
            "Text-band, packed TiO, and signed H2O source records pass through "
            "separate decoders before converging on common center, strength, "
            "species, excitation, damping, and margin arrays."
        ),
        caption=(
            "<strong>Common fields, family-owned meanings.</strong> The three "
            "sources retain distinct decoding and isotope conventions before "
            "they publish the same compiled array roles."
        ),
    ),
    FigureSpec(
        id="ch08-host-to-device",
        chapter=8,
        title="Host compilation feeds bounded device deposition",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original left-to-right scientific workflow. At left, show a small
manifest sheet and two packed source strips beside a CPU chip, labelled
"host compile". From the CPU, send a neat bundle of parallel invariant arrays
through one arrow labelled "move once" to a generic compute device. Inside the
device area, show three successive bounded shapes: a modest block labelled
"line chunks", a thinner list labelled "surviving pairs", and several short
offset ribbons labelled "wing blocks". Their arrows end at a rectangular
depth-by-wavelength array labelled "(D, W) opacity". Show small plus signs
where contributions meet the array.

The claim is that parsing and exact discrete decisions stay on the host,
invariants move once, and bounded line, pair, and offset blocks add into one
device-resident opacity slab. Do not display CUDA, MPS, benchmarks, timing
numbers, a neural network, transfer, or emergent flux.

{STYLE}
""",
        asset_path="assets/schematics/textbook/ch08-host-to-device-v1.png",
        alt_text=(
            "Manifest and packed sources compile on the host, invariant arrays "
            "move once to a compute device, and bounded line chunks, surviving "
            "pairs, and wing blocks add into a depth-by-wavelength opacity slab."
        ),
        caption=(
            "<strong>Bound the temporary arrays.</strong> Host compilation "
            "settles discrete source meanings. Resident invariants then feed "
            "separately bounded line, pair, and wing-offset work before sparse "
            "addition into the \\((D,W)\\) slab."
        ),
    ),
    FigureSpec(
        id="ch08-two-lanes",
        chapter=8,
        title="Atmosphere and synthesis molecular families are wired differently",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw two clean horizontal lanes with generous separation. Label the upper
lane "atmosphere". In it, three solid source cards labelled "diatomic",
"TiO", and "H2O" flow into one opacity slab. Add a fourth card labelled
"H3+ explicit path" connected by a dashed optional branch. Label the lower
lane "synthesis". In it, two solid source cards labelled "text bands" and
"TiO" flow into one opacity slab. Place a separate H2O card beside this lane
labelled "compiler only", with a short line that stops before the opacity
slab. Place a small crossed-out H3+ card farther aside with no connecting
arrow.

The sole claim is exact runtime wiring: standard atmosphere includes
diatomic, TiO, and H2O; atmosphere H3+ requires an explicit path; standard
synthesis includes text bands and TiO; synthesis H2O has a compiler but is
not wired; synthesis H3+ has no source route. Keep every label verbatim and
short. Do not imply the two opacity slabs are byte-identical. Do not show
profiles, transfer, flux, file paths, or performance.

{STYLE}
""",
        asset_path="assets/schematics/textbook/ch08-two-lanes-v1.png",
        alt_text=(
            "The atmosphere lane uses standard diatomic, TiO, and H2O sources "
            "with an optional explicit H3+ path, while synthesis uses text "
            "bands and TiO; its H2O branch stops at compiler-only status and "
            "H3+ has no route."
        ),
        caption=(
            "<strong>Availability is not wiring.</strong> Both lanes compute "
            "molecular line opacity, but their standard sources differ. The "
            "diagram is a conceptual status map, not an equality claim between "
            "intermediate arrays."
        ),
    ),
    FigureSpec(
        id="ch09-rays-moments-boundaries",
        chapter=9,
        title="Rays, angular moments, and atmosphere boundaries",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw one enlarged side-on plane-parallel atmosphere as eight thin horizontal
layers. Label the white region above it "space", the upper material edge
"surface", and the lowest visible edge "deep layers". Make the first stored
layer visibly inside the material, with a small bracket labelled
"first stored layer". Draw several straight ray arrows crossing the layers:
two outward rays and one inward ray. Mark one outward angle with a small arc
from the outward surface normal and label it "mu = cos(theta)".

At the right, place three compact hand-drawn summary cards, each receiving
thin connector lines from the collection of rays. Label them exactly
"J: average intensity", "H: net outward flow", and
"K: direction-weighted intensity". Put a plus/minus direction cue beside H
so it is visibly a signed moment, not a second brightness average. Add one
short note near the top boundary: "no incident stellar ray from space".
Add one short note in the deepest layers: "diffusion limit".

The claim is that boundary conditions and angular weighting turn directional
intensities into different moments, and that the first stored layer is not
empty space. Do not show numerical quadrature nodes, transfer matrices,
source iteration, spectra, stars, software, or invented values.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch09-rays-moments-boundaries-v1.png"
        ),
        alt_text=(
            "Original hand-sketched plane-parallel atmosphere with outward and "
            "inward rays, the direction cosine mu, distinct space, surface, "
            "first-stored-layer, and deep-layer boundaries, and compact cards "
            "explaining the J, H, and K angular moments."
        ),
        caption=(
            "<strong>Conceptual angular-moment schematic.</strong> Rays carry "
            "directional intensity, while <em>J</em>, <em>H</em>, and "
            "<em>K</em> apply different angular weights. The upper boundary "
            "lies above the first stored material layer; the deep boundary "
            "approaches diffusion."
        ),
    ),
    FigureSpec(
        id="ch09-scattering-fixed-point",
        chapter=9,
        title="Scattering turns the source into a fixed point",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original clockwise dependency loop using four large rounded cards.
The cards, in loop order, are labelled exactly "complete source S",
"formal transfer", "mean intensity J", and "source correction". Use one
deep-navy arrow between every adjacent pair and one final deep-navy arrow from
"source correction" back to "complete source S". Beside the loop, place one
separate warm-grey card labelled "local thermal source" with exactly one
solid arrow entering "complete source S". Place a small slider-like mark on
that entering arrow labelled "1 - scattering fraction". Place a second
small mark on the return arrow from J through the correction labelled
"scattering fraction".

Under the loop write only "repeat to a static fixed point". The arrows must
look like an ordered numerical iteration, not a photon orbit or time cycle.
The thermal card enters once and must not be inside the loop. Do not show
atoms, photon trajectories, neural networks, clocks, convergence numbers,
code, or a finished spectrum.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch09-scattering-fixed-point-v1.png"
        ),
        alt_text=(
            "Original hand-sketched fixed-point loop in which a complete source "
            "undergoes formal transfer to produce mean intensity, which corrects "
            "the source, while the local thermal source enters once from outside."
        ),
        caption=(
            "<strong>Scattering makes the source implicit.</strong> A trial "
            "source creates a mean intensity through formal transfer; the "
            "scattered part of that field then corrects the source. Repetition "
            "solves a static numerical fixed point, not a time evolution."
        ),
    ),
    FigureSpec(
        id="ch09-total-continuum-stack",
        chapter=9,
        title="Total and continuum transfer share one atmosphere",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw one left-to-right scientific flow. At left, place one large card labelled
"same atmosphere state". It forks into two clearly separated horizontal
lanes. Label the upper lane "total" and show two small inputs,
"continuum opacity" and "line opacity", joining one card labelled
"transfer". Label the lower lane "continuum" and show "continuum opacity"
entering a separate card labelled "transfer"; beside the missing line input
write "line terms = 0". The two transfer cards must be equal in size and
visually parallel.

At the right, the upper lane ends in "total flux" and the lower lane ends in
"continuum flux". Only after those two flux cards, draw arrows from both into
one ratio card labelled exactly "normalized flux = total / continuum".
The two lanes must never merge before the ratio. Use deep navy for the common
atmosphere fork and muted distinct accents for the two lane outputs.

The claim is that normalized flux is a ratio of two physical transfer solves
through one supplied atmosphere, not a fitted baseline. Do not show catalog
parsing, source iteration, device hardware, cache boxes, an observed spectrum,
continuum fitting, or numerical values.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch09-total-continuum-stack-v1.png"
        ),
        alt_text=(
            "Original hand-sketched two-lane diagram in which one atmosphere "
            "feeds total transfer with continuum and line opacity and continuum "
            "transfer with line terms set to zero, with the two resulting fluxes "
            "meeting only in their normalized-flux ratio."
        ),
        caption=(
            "<strong>Two physical solves, one ratio.</strong> Total and "
            "continuum flux use the same atmosphere and transfer physics. The "
            "continuum branch removes line terms; only the two emergent fluxes "
            "meet when normalized flux is formed."
        ),
    ),
    FigureSpec(
        id="ch09-two-transfer-lanes",
        chapter=9,
        title="The synthesis and atmosphere transfer lanes",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw two restrained horizontal computation lanes with generous vertical
separation. The upper lane is labelled
"synthesis — Torch wavelength batch". Show a shallow rectangular array with
five wavelength rows. Within every row, arrows run through depth from
"surface" toward "inward"; no arrows connect different rows. The array enters
a card labelled "51-point source solve" and ends in two compact outputs
"surface total H_nu" and "surface continuum H_nu". Add small device badges
"CPU", "CUDA", and "MPS" beside this lane.

The lower lane is labelled
"atmosphere — Numba frequency chunks". Show three separate chunk cards under
one bracket labelled "prange over chunks". Inside every chunk, draw one
frequency row with ordered arrows through depth from surface to inward. Each
chunk ends in its own private accumulator tray. After the trays, show one
ordered merge card labelled "fixed chunk order". End with the depth-dependent
outputs "J_nu", "H_nu", and "J_nu - S_nu".

Use a shared vertical note between the lanes: "depth sweep stays ordered".
The claim is two exact axis and output contracts built from common transfer
physics. Do not imply GPU atmosphere iteration. Do not show cache keys,
catalog compilation, host transfers, benchmarks, memory sizes, or Chapter 10
pipeline orchestration.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch09-two-transfer-lanes-v1.png"
        ),
        alt_text=(
            "Original hand-sketched comparison of a Torch wavelength-batched "
            "synthesis transfer lane returning surface total and continuum "
            "Eddington fluxes and a Numba frequency-chunked atmosphere lane "
            "returning depth-dependent moments with private accumulators and "
            "fixed-order reduction; depth remains ordered in both."
        ),
        caption=(
            "<strong>Shared physics, distinct executable boundaries.</strong> "
            "Synthesis batches wavelengths on the selected Torch device and "
            "returns surface moments. Atmosphere transfer parallelizes private "
            "frequency chunks on the CPU, preserves each depth recurrence, and "
            "returns depth-dependent moments for later structure corrections."
        ),
    ),
    FigureSpec(
        id="ch10-window-versus-star-state",
        chapter=10,
        title="Reusable window state and changing stellar state",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape dependency diagram with a clear reusable middle.
At left, place six small input cards labelled exactly "wavelength window",
"catalogs", "tables", "device", "dtype", and "chunk policy". Their arrows
join one large slate-blue card labelled "reusable WindowInvariants". Put a
small circular reuse arrow around that large card.

At the right, draw two separate structured-atmosphere layer stacks labelled
"star A atmosphere" and "star B atmosphere". Each stack enters its own
rounded card labelled "star-dependent state". Draw one solid arrow from the
same reusable WindowInvariants card into each star-dependent card. Above the
two star cards, place one narrow card labelled "hydrogen template". Split it
into two leaves labelled "merge state A" and "merge state B", each entering
only the matching star-dependent card. Make the two merge-state leaves visibly
different while the template remains shared. Each star-dependent card ends in
its own small native-spectrum trace.

The claim is that grids, catalogs, tables, and invariant tensors are reused
when the complete window key is unchanged, while densities, populations,
Doppler widths, opacity, and the hydrogen merge boundary belong to each star.
Do not show a neural network, fitting, atmosphere convergence, cache files,
timings, device transfers, or invented numerical values.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch10-window-versus-star-state-v1.png"
        ),
        alt_text=(
            "Original hand-sketched dependency diagram in which window, "
            "catalog, table, device, dtype, and chunk policy build one reusable "
            "WindowInvariants object, while two structured atmospheres create "
            "separate star-dependent states and separate hydrogen merge states "
            "from one shared template."
        ),
        caption=(
            "<strong>Reuse stops at the stellar state.</strong> One complete "
            "window key owns reusable grids, catalogs, tables, and invariant "
            "tensors. Each atmosphere still supplies its own densities, "
            "populations, Doppler widths, opacity, and hydrogen merging state."
        ),
    ),
    FigureSpec(
        id="ch10-host-device-precision-map",
        chapter=10,
        title="The real host, device, and precision boundaries",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape host/device map in three horizontal zones. The top
zone is labelled "host — NumPy and control". It contains cards for
"source parsing", "schema + cache identity", "most star depth columns", and
"float64 discrete choices". The middle zone is labelled
"selected Torch device — CPU / CUDA / MPS". It contains a broad work path:
"kernel-specific uploads" to "continuum + profiles" to
"total / continuum transfer". Inside that middle path, draw one clearly
bounded warm-beige island labelled exactly "float32 line + scattering island".
Outside the island but still on device, place a card labelled
"device-side output_slice crop".

Use thin dashed arrows for small scalar, mask, or bracket control returning to
the host, and label one such arrow "small control read". Use thick solid
arrows only for full spectral results. From the device-side crop, draw exactly
three separate thick solid arrows crossing to a bottom-right host card:
"total H_nu", "continuum H_nu", and "normalized flux". Label that destination
"final host float64 result stage". Add the short warning
"star state is not one permanently resident tensor".

The claim is a mixed host/device architecture with deliberate precision
islands and three final result transfers, not one literal copy. Do not imply
GPU atmosphere iteration, forbid all host synchronization, show a monolithic
all-device star cube, add benchmarks, or include a neural network.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch10-host-device-precision-map-v1.png"
        ),
        alt_text=(
            "Original hand-sketched host and device map showing source, schema, "
            "control, and most stellar depth columns on the host; kernel-specific "
            "Torch uploads and broad work on the selected device; a bounded "
            "float32 line and scattering island; device-side cropping; small "
            "control reads; and three final float64 spectral result transfers."
        ),
        caption=(
            "<strong>A heterogeneous pipeline, not an all-device star "
            "object.</strong> Most depth state remains host-side until a kernel "
            "uploads what it needs. Large opacity and transfer work stays on "
            "the selected Torch device, with deliberate float32 islands, before "
            "three spectral tensors enter the final host-float64 result stage."
        ),
    ),
    FigureSpec(
        id="ch10-context-compute-crop",
        chapter=10,
        title="Compute with context, return the exact requested grid",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw one long horizontal wavelength strip divided into three contiguous
regions. The central, largest white-slate region is labelled
"exact requested samples W". The left pale-blue region contains sixteen small
ticks and is labelled "16 blue context samples". The right pale-beige region
contains sixteen small ticks and is labelled "16 red context samples".
Directly above the complete three-region strip, draw one long bracket labelled
"opacity + transfer on W + 32 samples".

Below it, draw the same complete strip entering a card labelled
"device-side output_slice". From that card, show only the central requested
region continuing to an output strip labelled
"public wavelength: exact original W samples". Make two small scissors or
crop marks remove only the context regions. Connect the input central region
to the output central region with thin vertical guides and add the short note
"interior preserved bitwise". Add a small ratio mark between adjacent native
ticks labelled "1 + 1 / resolution".

The claim is numerical context for edge-safe native synthesis followed by an
exact on-device crop. It is not padding returned to the user and not an
instrumental convolution. Do not change the central tick positions, blur or
resample the strip, show an observed-pixel grid, add a line-spread function,
or invent wavelength numbers.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch10-context-compute-crop-v1.png"
        ),
        alt_text=(
            "Original hand-sketched wavelength strip with sixteen blue and "
            "sixteen red context samples around an exact requested grid; all "
            "opacity and transfer work uses the expanded grid, after which a "
            "device-side output slice removes only the context and preserves "
            "the requested interior bitwise."
        ),
        caption=(
            "<strong>Context is computed, not returned.</strong> Native samples "
            "are grown outward from the exact requested endpoints, all physical "
            "work uses the expanded strip, and `output_slice` removes the 16+16 "
            "context samples on device without resampling the requested grid."
        ),
    ),
    FigureSpec(
        id="ch11-seed-gates-to-pass-state",
        chapter=11,
        title="A supplied seed crosses narrow gates before one atmosphere pass",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape dependency diagram for a senior-undergraduate
stellar-spectroscopy textbook. At left, show one tall stack of nine aligned
outer-to-inner tracks labelled collectively "supplied ModelAtmosphere".
Use only these short track labels: "column mass", "temperature",
"gas pressure", "electron density", "Rosseland opacity",
"radiative acceleration", "microturbulence", "convective flux", and
"convective velocity". A single deep-navy arrow carries the stack through a
compact gate labelled "validate shape + signs + order".

After validation, split the path into two explicitly labelled alternatives.
The upper branch crosses a double vertical gate labelled
"declared fixed-column quantization Q"; the lower branch is labelled
"already at the intended seed boundary". Rejoin both branches before one
large slate-blue card labelled "RunSetup". Add a short note beside it:
"pass 1 copies seed".

Below the main path, draw a separate muted-grey return path beginning at a
small card labelled "previous radiation support". Route it through a card
labelled "hydrostatic gas pressure" and point it toward a second small card
labelled "pass 2+". This return path must not enter the pass-1 seed. Do not
show opacity sampling, transfer, correction, convergence, neural networks,
schema v4, spectra, or invented values.

The sole claim is that nine aligned columns cross narrow validation and an
explicit optional numerical boundary before becoming RunSetup, while
hydrostatic pressure first uses previous radiation support on pass 2.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch11-seed-gates-to-pass-state-v1.png"
        ),
        alt_text=(
            "Conceptual hand-sketched diagram in which nine aligned "
            "ModelAtmosphere columns pass shape, sign, and ordering validation, "
            "optionally cross a declared fixed-column quantization gate, and "
            "form RunSetup for a copied first pass, while a separate grey path "
            "shows previous radiation support entering hydrostatic gas pressure "
            "only on pass two and later."
        ),
        caption=(
            "<strong>Conceptual seed-boundary schematic.</strong> Validation "
            "checks a supplied `ModelAtmosphere`; fixed-column quantization is "
            "crossed only at a declared numerical boundary. Pass 1 copies the "
            "seed. Hydrostatic gas pressure first consumes remapped radiation "
            "support on pass 2."
        ),
    ),
    FigureSpec(
        id="ch11-two-atmosphere-grids",
        chapter=11,
        title="The atmosphere uses two distinct coordinates",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape comparison with two large, clearly separated
scientific objects. At left, show a vertical stack of thin depth layers with
an inward arrow. Label the object "80-layer standard coordinate" and place
the formula "tau_R,std = 10^(-6.875 + 0.125 i)" beneath it. Add the short
note "outer to inner". Do not imply that this coordinate was obtained by
integrating the current opacity.

At right, show one long horizontal wavelength strip with many fine tick marks,
labelled "30,000 direct wavelength samples". Draw a wavelength arrow pointing
right and a frequency arrow directly below it pointing left. Beside the strip,
place five small aligned start cards labelled exactly
"Teff >= 30000", "13000 to <30000", "7250 to <13000",
"4500 to <7250", and "Teff < 4500". Each card may connect to a different
starting position on the same strip, but every branch must retain the label
"30,000 samples".

Between the two objects put a restrained note "different jobs, both exact".
Near the far right, draw a tiny crossed-out three-point edge symbol labelled
"not the synthesis grid". Do not show resolving power, context padding,
flux, opacity values, transfer, temperature correction, or a stellar globe.

The sole claim is that the fixed 80-layer standard Rosseland coordinate and
the effective-temperature-dependent 30,000-point atmosphere wavelength grid
are separate exact coordinates, and that frequency reverses the wavelength
ordering.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch11-two-atmosphere-grids-v1.png"
        ),
        alt_text=(
            "Conceptual hand-sketched comparison of the fixed outer-to-inner "
            "80-layer standard Rosseland coordinate and the separate direct "
            "30,000-sample atmosphere wavelength grid, whose starting coverage "
            "changes across five effective-temperature branches while its "
            "frequency arrow points opposite the wavelength arrow."
        ),
        caption=(
            "<strong>Conceptual coordinate schematic.</strong> The standard "
            "80-layer coordinate organizes seed microturbulence and later "
            "remapping. The direct 30,000-point wavelength grid organizes "
            "atmosphere opacity sampling; its coverage changes with effective "
            "temperature, while its sample count does not."
        ),
    ),
    FigureSpec(
        id="ch11-select-once-recompute-opacity",
        chapter=11,
        title="Reuse discrete catalog membership and rebuild continuous opacity",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape lifecycle diagram with generous whitespace. At the
upper left, place four small source cards labelled exactly "atomic",
"diatomic", "TiO", and "H2O". Their arrows enter one rounded card labelled
"first-pass selection". Beside it, place a separate card labelled
"detailed transitions" entering a narrow card labelled "load once". The
selection and load cards must end in two grey handle-shaped cards labelled
"selected catalog object" and "detailed catalog object".

In the lower half, draw two separate atmosphere cards labelled "state A" and
"state B". Each contains only the short labels "populations", "widths", and
"continuum". Draw arrows from both shared grey catalog objects into both
state lanes. State A ends in one amber depth-by-frequency slab labelled
"line opacity A"; state B ends in a visibly different amber slab labelled
"line opacity B". Put the short equation-like statement
"same objects, new values" between the two outputs.

Both opacity slabs then point separately to two blue cards labelled
"OpacityState A" and "OpacityState B". At the far right, draw one outgoing
deep-navy arrow labelled "Chapter 12 transfer reductions"; it must leave the
two OpacityState cards and exit the frame. Add a small dashed optional source
card beside the standard four labelled "H3+ explicit path"; it may enter
first-pass selection but must be visibly outside the standard group.

The sole claim is that discrete selected and detailed catalog objects are
created or loaded once and reused, while every changed atmosphere recomputes
its continuous line-opacity slab. Do not show cache files, a full atmosphere
iteration loop, correction, convergence, schema v4, GPU hardware, spectra,
timings, or invented numbers.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch11-select-once-recompute-opacity-v1.png"
        ),
        alt_text=(
            "Conceptual hand-sketched lifecycle diagram in which standard "
            "atomic, diatomic, TiO, and H2O sources create one selected catalog "
            "object, detailed transitions load into a second object, and two "
            "atmosphere states reuse both objects while producing different "
            "line-opacity slabs and separate OpacityState outputs; H3+ appears "
            "only as an explicit optional path."
        ),
        caption=(
            "<strong>Conceptual lifecycle schematic.</strong> Catalog "
            "membership is a discrete first-pass decision. The exact objects "
            "can be reused by a changed atmosphere, but populations and widths "
            "still require a newly accumulated line-opacity slab before each "
            "`OpacityState` enters Chapter 12."
        ),
    ),
    FigureSpec(
        id="ch12-frequency-private-reduction",
        chapter=12,
        title="Private frequency chunks reduce in a fixed order",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape computation diagram. At left, place one card
labelled "one OpacityState". Fan it into four contiguous horizontal lanes
labelled "frequency chunk 0", "frequency chunk 1", "frequency chunk 2",
and "frequency chunk C - 1". In every lane draw a short ordered loop of
frequency marks and a small vertical depth stack. Label every lane with the
three short statements "frequency loop ordered", "depth transfer ordered",
and "private writes isolated".

At the end of each lane, draw one separate private tray labelled
"8 depth columns + 1 scalar". The four trays must not share write arrows.
After all trays, draw one vertical funnel labelled
"ascending chunk reduction". Its arrows combine lanes explicitly from top
to bottom. End at one slate-blue card labelled "TransferAccumulation".

Above the fan, draw a separate narrow outlined ribbon labelled
"persistent correction history + lookup". Route it around, not through, a
small reset gate placed before the private trays. The per-pass accumulator
lanes cross the reset gate; the persistent ribbon bypasses it.

The sole claim is that parallelism exists across private contiguous frequency
chunks, each frequency and depth recurrence remains ordered, and combination
occurs only in a fixed ascending reduction. Do not show transfer equations,
convection, correction values, GPU hardware, spectra, timings, or invented
numbers.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch12-frequency-private-reduction-v1.png"
        ),
        alt_text=(
            "Conceptual hand-sketched diagram in which one OpacityState fans "
            "into private contiguous frequency chunks, each preserving ordered "
            "frequency and depth work, before eight depth-column accumulators "
            "and one scalar per chunk are combined in ascending chunk order; "
            "persistent correction history and lookup bypass the per-pass reset."
        ),
        caption=(
            "<strong>Conceptual reduction schematic.</strong> `prange` owns "
            "private contiguous frequency chunks. Every chunk preserves the "
            "formal depth recurrence and writes only its own accumulators; a "
            "fixed ascending reduction forms `TransferAccumulation` after the "
            "parallel region."
        ),
    ),
    FigureSpec(
        id="ch12-four-state-eos-transaction",
        chapter=12,
        title="Convection derivatives require a restored four-state transaction",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original landscape scientific transaction diagram. In the center,
place one large pale-beige card labelled "central atmosphere state". Around
it place four smaller cards in an orderly clockwise ring, labelled exactly
"1.001 T, P", "0.999 T, P", "T, 1.001 P", and "T, 0.999 P".
Number the surrounding cards 1 through 4 in that order. Draw arrows from the
central state through each perturbation sequentially, never in parallel.

Each perturbation passes through one narrow card labelled
"atomic or molecular closure" and deposits a compact pair labelled
"(e, rho) sample". After the fourth sample, draw one deep-navy return arrow
labelled "finally: restore central state" back to the central card.
Beside this return arrow add exactly two restrained amber warnings:
"charge_square_density not restored" and
"all-zero central energy recomputed".

At right, collect the eight sampled values into one card labelled
"c_P^src, chi_T, c_s, nabla_ad", then point to three successive cards
labelled "raw convective flux", "optional overshoot", and
"top-layer suppression". Make the calculation direction unmistakable.

The sole claim is that convection derivatives come from four ordered EOS
perturbations inside a restore transaction, with two exact source caveats,
before the raw/overshoot/suppressed flux sequence. Do not show a full
atmosphere pass loop, opacity sampling, transfer rays, correction, convergence,
spectra, code, or invented numerical results.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch12-four-state-eos-transaction-v1.png"
        ),
        alt_text=(
            "Conceptual hand-sketched four-state EOS transaction around one "
            "central atmosphere, with sequential temperature-plus, "
            "temperature-minus, pressure-plus, and pressure-minus closure "
            "samples, a finally restoration carrying two source caveats, and "
            "eight sampled values feeding thermodynamic derivatives and the "
            "raw, overshoot, then top-suppressed convection sequence."
        ),
        caption=(
            "<strong>Conceptual thermodynamic transaction.</strong> Four "
            "ordered closure evaluations provide the energy and density samples "
            "used by the convection derivatives. A `finally` path restores the "
            "central state, subject to the two explicitly audited source "
            "deltas, before convection is assembled in its exact order."
        ),
    ),
    FigureSpec(
        id="ch13-exact-pass-orbit",
        chapter=13,
        title="One complete atmosphere pass preserves state lifetimes",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original wide landscape orbit with fifteen numbered stations in one
unbroken clockwise order. Use a pale-beige inner lane for pass-local arrays
and a slate-blue outer lane for state carried to the next pass. Keep labels
short but render every station number and phrase exactly:
1 "seed copy / prior remap + optional hydrostatic Pgas"
2 "pass setup + carried surface constant"
3 "populations + widths + strengths"
4 "continuum + line opacity"
5 "build or reuse catalog handles"
6 "reset accumulators; keep history + lookup"
7 "transfer + fixed-order reduction"
8 "Rosseland finalization"
9 "radiation pressure + new surface constant"
10 "ingest current T, Pgas, kappa_R"
11 "four EOS samples + convection if enabled"
12 "temperature + column correction"
13 "disabled diagnostics if needed; complete remap"
14 "carry lookup + surface constant"
15 "structural / flux diagnostics; return or exit"

After station 15, one deep-navy return arrow must point to station 1 and be
labelled "unquantized IterationRemap". Put small reset glyphs only at station
6. At station 13, draw a small conditional side branch labelled
"convection disabled" entering "diagnostics", then returning before
"complete remap"; it must occur after station 12. Do not place fixed-column
quantization anywhere on the orbit or its return arrow.

The sole claim is exact source order and selective state lifetime across one
complete physical pass. Do not merge or renumber stations, show a neural
network, schema v4, synthesis, spectra, detailed equations, or numerical
results.

{STYLE}
""",
        asset_path="assets/schematics/textbook/ch13-exact-pass-orbit-v1.png",
        alt_text=(
            "Conceptual hand-sketched fifteen-station atmosphere-pass orbit "
            "from seed or prior remap through setup, populations, opacity, "
            "catalog lifecycle, accumulator reset, transfer, Rosseland and "
            "radiation finalization, lookup ingest, convection, correction, "
            "disabled diagnostics when needed, complete remap, carried state, "
            "and diagnostics, with the unquantized remap returning to the next "
            "pass and no quantization inside the loop."
        ),
        caption=(
            "<strong>Conceptual pass-orbit schematic.</strong> Pass-local "
            "arrays reset where the source resets them; catalog handles, lookup "
            "history, the surface constant, and the unquantized complete remap "
            "cross their declared boundaries. Fixed-column quantization remains "
            "outside this orbit."
        ),
    ),
    FigureSpec(
        id="ch13-terminal-quantization-gate",
        chapter=13,
        title="Terminal quantization separates iteration from publication",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original compact landscape boundary diagram. At left, show a small
clockwise loop labelled "physical iteration" whose return arrow carries
"unquantized IterationRemap". Let only an exit arrow from this loop pass
through a double vertical gate labelled "terminal boundary". Immediately
after the gate place one and only one card labelled
"Q = format then parse once".

From the quantized state, split three clearly different paths. The upper path
ends at "AtmosphereRunResult" and is labelled "terminal quantized columns".
The middle dashed path ends at "debug state" and is labelled
"last unquantized remap". Make clear that this dashed path branches from
before Q, not after it. The lower path crosses a gate labelled
"converged only", then a card "rebuild populations from quantized columns",
and ends at "schema-v4 physical product". Add a small note beside a blocked
lower path: "unconverged: no product".

The sole claim is that complete remap lives inside iteration, fixed-column
quantization occurs exactly once at the terminal boundary, debug keeps the
last unquantized remap, and a physical product is rebuilt and written only
after convergence. Do not show an initializer, synthesis, spectra, opacity
details, multiple Q gates, quantization on the loop return, or invented
values.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch13-terminal-quantization-gate-v1.png"
        ),
        alt_text=(
            "Conceptual hand-sketched terminal boundary in which the physical "
            "iteration returns an unquantized complete remap, exit crosses one "
            "format-then-parse quantization gate, result columns use the "
            "terminal quantized state, debug retains the last unquantized "
            "remap, and only convergence permits populations to be rebuilt "
            "from quantized columns into a schema-v4 physical product."
        ),
        caption=(
            "<strong>Conceptual terminal-boundary schematic.</strong> "
            "Iteration carries an unquantized complete remap. One terminal "
            "format/parse operation defines result columns; debug preserves the "
            "last unquantized remap, while schema v4 is rebuilt from quantized "
            "columns and promoted only after structural convergence."
        ),
    ),
    FigureSpec(
        id="ch14-initializer-to-closure",
        chapter=14,
        title="A learned initializer proposes a start; physical closure accepts it",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original wide landscape initializer-to-closure workflow. Begin with a
card labelled "requested stellar labels" containing only
"Teff, log g, mixture, microturbulence". Fan into three separate horizontal
lanes:

1. "five-label" -> "own normalization + MLP + PCA basis";
2. "CNO8: add C, N, O" -> "own normalization + MLP + PCA basis";
3. "direct abundance: 81 [X/H]" -> "own normalization + set encoder + PCA
   basis", with the direct lane marked "experimental; closure mandatory".

Each lane outputs "160 coefficients (family-specific)". Only then merge them
into one box labelled "same shape, not same basis" and "6 profiles x 80
layers". The six rows are "column mass", "temperature", "gas pressure",
"electron density", "Rosseland opacity", and "radiative acceleration".
Continue through "inverse physical transforms" to one amber card labelled
"initialized atmosphere" with the tag "proposal, not accepted".

From that sole proposal card, draw one ordered horizontal physical sequence:
"1 populations" -> "2 opacity" -> "3 transfer" -> "4 convection" ->
"5 correction + remap" -> "structural convergence + independent checks".
The successful path ends at "schema-v4 physical atmosphere"; the failed path
loops back to populations with the label "no: next pass".

The sole claim is that all three families own distinct learned decoder bundles
but target the same six-profile representation, and only exact iteration plus
acceptance gates can promote it. Do not show a shared network or shared PCA
basis. Do not show spectra, fitting, hardware, code, legacy names, or invented
accuracy values.

{STYLE}
""",
        asset_path=(
            "assets/schematics/textbook/ch14-initializer-to-closure-v2.png"
        ),
        alt_text=(
            "Conceptual hand-sketched workflow in which five-label, CNO8, and "
            "direct-abundance requests pass through separate family-owned "
            "normalization, network or set-encoder, and PCA-basis bundles. "
            "They share only the six-profile output representation, which "
            "becomes an initialized proposal before the exact population, "
            "opacity, transfer, convection, correction, remap, and convergence "
            "sequence can permit a schema-v4 physical atmosphere."
        ),
        caption=(
            "<strong>A learned start shortens the route; it does not replace "
            "the route.</strong> Five-label, CNO-aware, and experimental "
            "direct-abundance inputs use separate trained decoder bundles that "
            "end in the same six-profile representation. Every decoded "
            "atmosphere remains a proposal until the exact physical loop "
            "converges and its independent checks pass."
        ),
    ),
    FigureSpec(
        id="ch15-two-workflow-gates",
        chapter=15,
        title="The exploratory and verified workflows answer different questions",
        prompt=f"""
Use case: scientific-educational
Asset type: conceptual textbook schematic
Draw an original wide landscape two-lane workflow. Begin at one shared card
labelled "requested stellar labels + mixture" and split into an upper lane
labelled "exploratory" and a lower lane labelled "verified physical".

The upper lane must contain, in order, "learned initializer", an
"InitializedAtmosphere" card carrying "converged = False" and
"closure required = True", "synthesis", and "LabelSpectrum". Under this
lane write "fast, useful, not a physical-closure claim".

The lower lane must contain, in order, "learned initializer seed", a gate
labelled "full checksum-verified catalogs", a circular exact-atmosphere pass
loop labelled around its orbit "population", "opacity", "transfer",
"correction", and "remap", a gate labelled "converged + independent checks",
"schema-v4 physical atmosphere", "synthesis", and "Spectrum". Add a blocked
branch beneath the catalog gate labelled "catalogs absent: solve unavailable"
and a blocked branch beneath the convergence gate labelled
"not converged: no physical product".

The sole claim is that both workflows begin from the same requested labels
but only the lower lane may produce a verified physical atmosphere. Keep the
two returned spectrum types visually distinct. Do not show fitting, parameter
optimization, fabricated numerical values, legacy names, or GPU hardware.

{STYLE}
""",
        asset_path="assets/schematics/textbook/ch15-two-workflow-gates-v1.png",
        alt_text=(
            "Conceptual hand-sketched two-lane workflow beginning from the "
            "same requested stellar labels and mixture. The exploratory lane "
            "uses a learned initializer and synthesis to return a "
            "LabelSpectrum whose atmosphere is explicitly unconverged and "
            "requires physical closure. The verified lane requires full "
            "checksum-verified catalogs, repeated exact atmosphere passes, "
            "convergence and independent checks, and a schema-v4 physical "
            "atmosphere before synthesis returns Spectrum; missing catalogs "
            "or failed convergence block the physical product."
        ),
        caption=(
            "<strong>Two questions require two contracts.</strong> The upper "
            "lane is a fast exploratory calculation and keeps its unverified "
            "state visible in the returned type. The lower lane admits a "
            "physical product only after exact atmosphere iteration and "
            "independent gates; neither missing catalogs nor failed "
            "convergence can be hidden by a plausible initializer."
        ),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id", nargs="?")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list or args.id is None:
        for spec in FIGURES:
            print(f"{spec.id:28} chapter {spec.chapter:02d}  {spec.title}")
        return

    by_id = {spec.id: spec for spec in FIGURES}
    if args.id not in by_id:
        parser.error(f"unknown figure id: {args.id}")
    print(by_id[args.id].prompt.strip())


if __name__ == "__main__":
    main()
