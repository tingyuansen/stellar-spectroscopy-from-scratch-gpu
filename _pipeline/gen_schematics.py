#!/usr/bin/env python
"""Generate teaching schematics with Gemini 3 Pro image ('Nano Banana') for the
load-bearing concepts, saved to resources/figures/. Idempotent (skips existing).

Mirrors ~/notebooks/_pipeline/schematics/gen_schematics.py: GOOGLE_API_KEY from ~/.env,
model gemini-3-pro-image-preview, a shared style constant S appended to every prompt.
"""
import os
from pathlib import Path

for line in (Path.home() / ".env").read_text().splitlines():
    if line.strip().startswith("GOOGLE_API_KEY="):
        os.environ["GOOGLE_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
FIG = Path(__file__).resolve().parent.parent / "resources" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MODEL = "gemini-3-pro-image-preview"

S = (" STYLE: a polished, professional figure for a high-end physics textbook (think Quanta Magazine or a Nature "
     "explainer or Distill.pub). Plain white background, no outer border. Typography: an elegant modern geometric "
     "sans-serif (Inter / Helvetica Neue feel), crisp, consistently sized, comfortably spaced, never tiny or crowded. "
     "Palette: deep charcoal-ink strokes and labels (near #20242E); rounded rectangles with very soft pale slate fills "
     "(#EEF2F7) and thin clean outlines; a single warm amber accent (#E8A23D) reserved ONLY for the one most important "
     "element. Consistent thin stroke weights, arrows with neat tapered heads, careful alignment to an invisible grid, "
     "generous balanced whitespace, clear visual hierarchy. Flat, refined, restrained — NO gradients, NO drop shadows, "
     "NO 3-D, NO clip-art, NO decorative icons, NO robots, NO emoji. Any caption sits at the bottom centre in a smaller, "
     "lighter grey weight, not bold and not oversized. Vector-crisp and uncluttered.")

FIGS = {
 "s1_pipeline.png":
   "A left-to-right pipeline of five rounded boxes joined by arrows, turning a star's parameters into its spectrum. "
   "Box 1 'STELLAR PARAMETERS: Teff, log g, abundances'. Arrow to box 2 'MODEL ATMOSPHERE: T, P, density vs depth'. "
   "Arrow to box 3 'EQUATION OF STATE: ionization, electron density n_e'. Arrow to box 4 'OPACITY: continuum + lines, kappa(lambda)'. "
   "Arrow to box 5 'RADIATIVE TRANSFER: emergent flux F(lambda)' (this final box is the single amber element). "
   "Below box 5 a tiny absorption-line spectrum sketch. Caption 'from four numbers to a spectrum'." + S,
 "s1_optical_depth.png":
   "A 1-D plane-parallel stellar photosphere drawn as a horizontal stack of thin layers, an axis on the left labeled "
   "'optical depth tau' increasing downward (tau=0 at top, large at bottom). One dashed horizontal line across the middle "
   "marks 'tau = 2/3' (the single amber element) labeled 'the photosphere: where we see'. An upward arrow from near that "
   "line labeled 'escaping photon'; a short upward arrow starting deeper that stops, labeled 'reabsorbed'. "
   "Caption 'we see down to tau ~ 2/3'." + S,
 "s2_saha_boltzmann.png":
   "Two stacked concepts. TOP, labeled 'SAHA: ionization between stages': three boxes left-to-right 'neutral atom' -> "
   "'singly ionized + e-' -> 'doubly ionized + e-', each arrow labeled 'remove an electron', a small free-electron dot "
   "leaving at each step. BOTTOM, labeled 'BOLTZMANN: excitation within one ion': a small ladder of energy levels with "
   "populations shrinking upward, labeled 'n_i proportional to g_i exp(-E_i/kT)'. A side note (amber) 'in the Sun, metals "
   "supply the photospheric electrons'. Caption 'ionization and excitation set who absorbs'." + S,
 "s3_hminus.png":
   "The negative hydrogen ion H- as the dominant continuous opacity. Left: a hydrogen atom drawn as one proton (small "
   "circle '+') with one bound electron, capturing a SECOND, loosely-held electron, labeled 'H- : binds a 2nd electron "
   "by only 0.754 eV'. A wavy arrow 'photon' hits it; a right arrow shows the loose electron ejected, labeled "
   "'photo-detachment -> continuous absorption' (the photon arrow is the amber element). A small note 'abundance proportional "
   "to n_e (from the metals)'. Caption 'H- carries the visible continuum of cool stars'." + S,
 "s4_voigt.png":
   "Construction of the Voigt line profile as a convolution. Left: a narrow bell curve labeled 'Gaussian core "
   "(thermal + microturbulent Doppler)'. A convolution symbol (asterisk in a circle). Middle: a peaked curve with long "
   "tails labeled 'Lorentzian wings (natural + pressure broadening)'. An equals sign. Right: the combined profile labeled "
   "'VOIGT profile H(a,v)' (amber), with its core and wings annotated. Caption 'Doppler core convolved with damping wings'." + S,
 "s5_linelist.png":
   "Building total line opacity from a line list. Left: a small table 'LINE LIST' with columns 'wavelength, log gf, "
   "excitation, damping' and a few rows. An arrow 'sum each line's Voigt profile onto the grid'. Right: a plot with "
   "y-axis 'line opacity' and x-axis 'wavelength', showing many narrow opacity SPIKES pointing UPWARD from a near-zero "
   "baseline (opacity is LARGE at each line centre — upward peaks, NOT downward dips), of different heights, a 'forest' of "
   "upward spikes. A dashed horizontal threshold line near the baseline (amber) labeled 'CUTOFF: skip lines below 1e-3 of "
   "the continuum', with one faint short upward spike below it crossed out. Caption 'a million Voigt profiles, summed'." + S,
 "s5b_stark.png":
   "Stark broadening of hydrogen lines by the plasma microfield. CENTER: a single hydrogen atom (one proton '+' with "
   "one electron on a circular orbit) sitting among nearby charged particles — several small '+' ions and '-' electrons "
   "scattered around it, with thin arrows from them converging on the atom, labelled 'electric microfield of neighbouring "
   "ions and electrons (Holtsmark distribution)'. An arrow to the RIGHT shows the effect on one hydrogen energy level: a "
   "single level SPLITTING into several evenly-spaced sublevels, labelled 'linear Stark effect: the field shifts the levels "
   "(hydrogen is uniquely sensitive because its levels are degenerate)'. Far RIGHT: two line profiles drawn on the same "
   "axes for contrast — a NARROW peaked profile labelled 'a metal line: thermal Doppler + damping' and, overlaid, a much "
   "BROADER profile with long extended wings labelled 'a hydrogen Balmer line: broad Stark wings' (this broad profile is "
   "the single amber element). Use only horizontal text, no rotated or vertical labels. "
   "Caption 'the plasma microfield broadens hydrogen lines into broad wings'." + S,
 "s6_rt.png":
   "Line formation via the emergent flux. A plane-parallel photosphere with an optical-depth axis on the left; a short "
   "downward arrow on the right with the single word 'hotter' at its tip (use only horizontal text, NO rotated or vertical "
   "labels anywhere). TWO sightlines: one in the CONTINUUM reaching tau=2/3 in a deep, hot layer (label 'sees hot gas -> bright'); "
   "one IN A LINE, where the extra opacity makes tau=2/3 occur higher up in a cool layer (label 'sees cool gas -> dark') — "
   "this line sightline is the amber element. To the right, a small spectrum showing a bright continuum with one absorption "
   "dip aligned to the line sightline. Caption 'a line lets us see a higher, cooler layer (Eddington-Barbier)'." + S,
 "s7_josh.png":
   "A left-to-right flow of the JOSH moment radiative-transfer solver, five rounded boxes joined by arrows. "
   "Box 1 'OPACITY per depth: absorption, scattering, source B'. Arrow to box 2 'OPTICAL DEPTH: integrate over column mass'. "
   "Arrow to box 3 'MAP onto fixed grid: 51 Eddington depth points'. Arrow to box 4 'LAMBDA-ITERATION: S = (1-a) B + a J, "
   "scattering fraction a' with a small curved loop-back arrow above it labeled 'repeat until converged' (this box 4 is the "
   "single amber element, the heart of the method). Arrow to box 5 'SURFACE FLUX: weighted sum over depth'. "
   "Caption 'the moment solver: scattering handled by iteration'." + S,
 "s8_hydrostatic.png":
   "Building a grey model atmosphere from two numbers. On the left, two labelled inputs 'Teff' and 'log g'. "
   "An arrow to a vertical stack of plane-parallel layers (an optical-depth axis 'tau' increasing downward on the "
   "left of the stack). Two things are computed down the stack: a curved arrow labelled 'grey temperature law T(tau)' "
   "and, the amber element, a downward arrow labelled 'integrate dP/dtau = g/kappa' showing pressure P accumulating "
   "as the weight of the gas above. Output on the right: three small curves labelled 'T vs depth', 'P vs depth', "
   "'density vs depth' — ALL THREE rising/INCREASING to the right (deeper means hotter, higher pressure, and denser; "
   "every curve goes up toward the right, none decreasing). "
   "Caption 'from Teff and log g to the run of temperature and pressure with depth'." + S,
 "s9_tcorr.png":
   "The temperature-correction loop that enforces radiative equilibrium. A cycle of rounded boxes connected by arrows "
   "going clockwise: 'temperature T(tau)' -> 'compute the radiative flux at every depth (solve transfer)' -> "
   "'is the flux constant with depth?' -> 'correct the temperature: Delta T' -> back to 'temperature T(tau)'. "
   "A small inset plot shows flux-vs-depth NOT flat (sloping) becoming flat after iteration. The 'correct the "
   "temperature' box is the single amber element. Caption 'nudge T until the radiative flux is constant with depth "
   "(radiative equilibrium)'." + S,
 "s10_convection.png":
   "Why deep layers convect, and the converged model. A vertical atmosphere column split into two zones by a dashed "
   "line: the upper zone labelled 'radiative photosphere (where the spectral lines form)' with a straight upward arrow "
   "'energy carried by radiation'; the lower, deeper zone labelled 'convection zone' with small rounded rising and "
   "sinking blobs and the amber label 'energy carried by rising/sinking gas (mixing-length convection)'. A short "
   "caption arrow notes 'convection is negligible where the lines form, but sets the deep structure'. "
   "Caption 'in the deep layers convection carries the flux; iterating gives the converged model'." + S,
 "s11_molecules.png":
   "Molecular equilibrium and molecular bands in cool stars. LEFT, two stacked panels comparing gas at two "
   "temperatures. Top panel labelled 'HOT gas': two separate free atoms drawn as small labelled circles 'Ti' and 'O' "
   "moving apart. Bottom panel labelled 'COOL gas': the same two atoms bound together into one molecule labelled 'TiO'. "
   "A vertical two-way arrow links the panels, labelled 'formation balance: Ti + O <-> TiO, favoured when it is cool "
   "(a Saha-like equilibrium set by temperature and the dissociation energy D0)'. RIGHT: the spectral consequence on "
   "shared axes (x-axis 'wavelength', y-axis 'absorption'). A few WIDELY-SPACED tall narrow spikes labelled 'isolated "
   "atomic lines'; beside them a DENSE picket fence of many short lines that pile up at one edge into a sharp vertical "
   "step, labelled 'millions of molecular rovibrational lines blend into a BAND with a sharp bandhead' — this molecular "
   "band with its bandhead is the single amber element. Use only horizontal text, no rotated or vertical labels. "
   "Caption 'in cool stars atoms lock into molecules, whose bands reshape the spectrum'." + S,
 "s12_capstone.png":
   "A capstone figure: ONE pipeline produces the spectra of four very different stars. LEFT, a compact vertical "
   "pipeline of small labelled rounded boxes joined by downward arrows: 'stellar parameters (Teff, log g)' -> 'model "
   "atmosphere' -> 'opacity: continuum + atomic + molecular lines' -> 'radiative transfer' -> 'spectrum', with a thin "
   "bracket beside it labelled 'assembled from Lectures 1-12'. A horizontal arrow leads RIGHT to a small "
   "Hertzsprung-Russell diagram: horizontal axis 'temperature (hot on the LEFT, cool on the right)', vertical axis "
   "'luminosity', with four labelled points placed correctly — 'HOT DWARF' (upper left), 'SUN' (centre), 'GIANT' "
   "(upper right), 'M DWARF' (lower right). Next to each point a tiny spectrum thumbnail showing its signature: the hot "
   "dwarf one broad Balmer absorption trough; the Sun a forest of narrow metal lines; the giant a strong triplet dip "
   "(Mg b); the M dwarf a saw-toothed TiO molecular band with a sharp head — this M-dwarf TiO thumbnail is the single "
   "amber element. Use only horizontal text, no rotated or vertical labels. "
   "Caption 'one pipeline, built from the whole book, reproduces stars across the HR diagram'." + S,
}

for fn, prompt in FIGS.items():
    out = FIG / fn
    if out.exists():
        print("skip", fn); continue
    try:
        r = client.models.generate_content(
            model=MODEL, contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]))
        saved = 0
        for c in (r.candidates or []):
            for p in (getattr(c.content, "parts", None) or []):
                d = getattr(p, "inline_data", None)
                if d and d.data:
                    out.write_bytes(d.data); saved += 1
        print("ok " if saved else "NOIMG", fn, (out.stat().st_size // 1000 if saved else ""), "KB")
    except Exception as e:
        print("ERR", fn, type(e).__name__, str(e)[:120])
