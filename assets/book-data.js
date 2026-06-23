/* Course manifest — parts, lectures, author, source files.
   Only built lectures are listed here (the reader links to these). The full
   planned arc lives in README.md; entries are appended as each lecture clears
   its accuracy + pedagogy audit. */
window.BOOK = {
  title: "Stellar Spectroscopy from Scratch",
  series: "rebuilding a synthetic stellar spectrum",
  subtitle: "",
  parts: [
    {
      name: "Part I", title: "Foundations",
      blurb: "Treat the model atmosphere as given; set the foundations and the equation of state.",
      chapters: [
        { n: 1, slug: "Lecture1", type: "html", title: "Overview & a First Model Atmosphere", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 2, slug: "Lecture2", type: "html", title: "The Equation of State", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" }
      ]
    },
    {
      name: "Part II", title: "Continuous Opacity",
      blurb: "Build the smooth continuous background — the physics, then the production engine.",
      chapters: [
        { n: 3, slug: "Lecture3", type: "html", title: "Continuous Opacity", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 4, slug: "Lecture4", type: "html", title: "The KAPP Continuum Engine", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" }
      ]
    },
    {
      name: "Part III", title: "Line Opacity",
      blurb: "Build line opacity from a single profile to a million-line forest, then the hydrogen Stark wings.",
      chapters: [
        { n: 5, slug: "Lecture5", type: "html", title: "Line Opacity I: A Single Line", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 6, slug: "Lecture6", type: "html", title: "Line Opacity II: The Line List", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 7, slug: "Lecture7", type: "html", title: "Hydrogen Lines: Stark Broadening", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" }
      ]
    },
    {
      name: "Part IV", title: "Radiative Transfer",
      blurb: "Let the light out: the formal solution, then the production JOSH moment solver.",
      chapters: [
        { n: 8, slug: "Lecture8", type: "html", title: "Radiative Transfer & the Emergent Spectrum", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" },
        { n: 9, slug: "Lecture9", type: "html", title: "The JOSH Solver: Production Radiative Transfer", lecturer: "Yuan-Sen Ting", affil: "Max Planck Institute for Astronomy & The Ohio State University" }
      ]
    }
  ]
};

// Flat list for navigation
window.BOOK.flat = window.BOOK.parts.flatMap(p => p.chapters.map(c => ({ ...c, part: p })));
window.BOOK.byN = Object.fromEntries(window.BOOK.flat.map(c => [c.n, c]));
