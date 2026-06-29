/* Course manifest — parts, lectures, author, source files.
   Only built lectures are listed here (the reader links to these). The full
   planned arc lives in README.md; entries are appended as each lecture clears
   its accuracy + pedagogy audit. */
window.BOOK = {
  title: "Stellar Spectroscopy from Scratch",
  series: "self-contained torch/MPS/CUDA lectures, validated against shipped references",
  subtitle: "",
  parts: [
    {
      name: "Part I", title: "Foundations & Microphysics",
      blurb: "Treat the model atmosphere as given; set the foundations, the equation of state, and the continuous opacity.",
      chapters: [
        { n: 1, slug: "Lecture1", type: "html", title: "Overview & a First Model Atmosphere", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 2, slug: "Lecture2", type: "html", title: "The Equation of State", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 3, slug: "Lecture3", type: "html", title: "Continuous Opacity", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" }
      ]
    },
    {
      name: "Part II", title: "Line Opacity",
      blurb: "Build line opacity from a single profile to a million-line forest, then the hydrogen Stark wings.",
      chapters: [
        { n: 4, slug: "Lecture4", type: "html", title: "Line Opacity I: A Single Line", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 5, slug: "Lecture5", type: "html", title: "Line Opacity II: The Line List", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 6, slug: "Lecture6", type: "html", title: "Hydrogen Lines: Stark Broadening", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" }
      ]
    },
    {
      name: "Part III", title: "Radiative Transfer",
      blurb: "Let the light out: the formal solution, then the production JOSH moment solver.",
      chapters: [
        { n: 7, slug: "Lecture7", type: "html", title: "Radiative Transfer & the Emergent Spectrum", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 8, slug: "Lecture8", type: "html", title: "The JOSH Solver: Production Radiative Transfer", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" }
      ]
    },
    {
      name: "Part IV", title: "Building the Atmosphere",
      blurb: "Stop taking the model atmosphere as given: build its structure from hydrostatic and radiative equilibrium.",
      chapters: [
        { n: 9, slug: "Lecture9", type: "html", title: "Hydrostatic Equilibrium & Temperature Structure", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 10, slug: "Lecture10", type: "html", title: "Radiative Equilibrium & Temperature Correction", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 11, slug: "Lecture11", type: "html", title: "Convection & the Converged Atmosphere", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" }
      ]
    },
    {
      name: "Part V", title: "Cool Stars & Atmosphere-to-Spectrum Synthesis",
      blurb: "Beyond the warm Sun: cool stars form molecules and TiO bands carve their spectra; then the synthesis half — atmosphere in, spectrum out — is assembled and run across the HR diagram.",
      chapters: [
        { n: 12, slug: "Lecture12", type: "html", title: "Molecular Equilibrium & Molecular Bands", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 13, slug: "Lecture13", type: "html", title: "Molecular Chemistry: the Coupled Equilibrium and Continuous Opacity", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 14, slug: "Lecture14", type: "html", title: "A Spectrum from an Atmosphere, End to End", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" }
      ]
    },
    {
      name: "Part VI", title: "The Line-Blanketed Atmosphere — the Finale",
      blurb: "The model-atmosphere half: switch on the predicted line list, the wing-walk deposit kernel, the line-blanketed Rosseland mean, and the convergence machinery; then build the multi-element equation of state and convective heat capacity needed by the line-blanketed Sun.",
      chapters: [
        { n: 15, slug: "Lecture15", type: "html", title: "Line Blanketing: the True Model Atmosphere", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 16, slug: "Lecture16", type: "html", title: "The Full Equation of State: Species Slots & the Convective Heat Capacity", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" }
      ]
    }
  ]
};

// Flat list for navigation
window.BOOK.flat = window.BOOK.parts.flatMap(p => p.chapters.map(c => ({ ...c, part: p })));
window.BOOK.byN = Object.fromEntries(window.BOOK.flat.map(c => [c.n, c]));
