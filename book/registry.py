"""The single canonical registry for the textbook and web reader."""

from __future__ import annotations

from dataclasses import dataclass


AUTHOR = "Yuan-Sen Ting"
AFFILIATION = "Max Planck Institute for Astronomy & The Ohio State University"


@dataclass(frozen=True)
class Chapter:
    """One chapter's stable identity and current publication state."""

    number: int
    title: str
    module: str
    available: bool = False

    @property
    def slug(self) -> str:
        """Return the generated notebook/HTML basename."""

        return f"Chapter{self.number:02d}"


@dataclass(frozen=True)
class Part:
    """A named sequence of chapters with one conceptual purpose."""

    number: int
    title: str
    description: str
    chapters: tuple[Chapter, ...]


PARTS = (
    Part(
        1,
        "Foundations",
        "Learn what the model computes, build a first atmosphere, and turn equations into trustworthy code.",
        (
            Chapter(1, "From Starlight to a First Grey Atmosphere", "chapter_01", available=True),
            Chapter(
                2,
                "From Equations to Fast, Trustworthy Kernels and Explicit Data",
                "chapter_02",
                available=True,
            ),
        ),
    ),
    Part(
        2,
        "Thermochemical State",
        "Count atomic, ionic, electronic, and molecular particles in every atmospheric layer.",
        (
            Chapter(
                3,
                "Atoms, Ions, and Electrons",
                "chapter_03",
                available=True,
            ),
            Chapter(
                4,
                "Molecules and Coupled Equilibrium",
                "chapter_04",
                available=True,
            ),
        ),
    ),
    Part(
        3,
        "Opacity",
        "Build every non-redundant absorption and scattering source used by the working solver.",
        (
            Chapter(
                5,
                "Continuous Opacity and Scattering",
                "chapter_05",
                available=True,
            ),
            Chapter(6, "One Spectral Line", "chapter_06", available=True),
            Chapter(
                7,
                "Atomic Line Forests and Special Profiles",
                "chapter_07",
                available=True,
            ),
            Chapter(
                8,
                "Molecular Bands and Source Compilation",
                "chapter_08",
                available=True,
            ),
        ),
    ),
    Part(
        4,
        "Transfer and Synthesis",
        "Turn an atmospheric state into total, continuum, and normalized spectra.",
        (
            Chapter(
                9,
                "Radiative Transfer with Scattering",
                "chapter_09",
                available=True,
            ),
            Chapter(
                10,
                "GPU Synthesis from a Structured Atmosphere",
                "chapter_10",
                available=True,
            ),
        ),
    ),
    Part(
        5,
        "The Physical Atmosphere",
        "Iterate hydrostatic, radiative, and convective structure with NumPy and Numba.",
        (
            Chapter(
                11,
                "Starting and Blanketing an Atmosphere",
                "chapter_11",
                available=True,
            ),
            Chapter(
                12,
                "Radiation, Thermodynamics, and Convection",
                "chapter_12",
                available=True,
            ),
            Chapter(
                13,
                "Correction and the Full Numba Iteration",
                "chapter_13",
                available=True,
            ),
        ),
    ),
    Part(
        6,
        "Initialization and Complete Workflows",
        "Accelerate starts without confusing a prediction with a physically accepted atmosphere.",
        (
            Chapter(
                14,
                "Learned Initializers and Mandatory Physical Closure",
                "chapter_14",
                available=True,
            ),
            Chapter(
                15,
                "From Stellar Labels to a Verified Spectrum",
                "chapter_15",
                available=True,
            ),
        ),
    ),
)

CHAPTERS = tuple(chapter for part in PARTS for chapter in part.chapters)
BY_NUMBER = {chapter.number: chapter for chapter in CHAPTERS}

if tuple(chapter.number for chapter in CHAPTERS) != tuple(range(1, 16)):
    raise RuntimeError("chapter registry must contain the consecutive range 1..15")
if len({chapter.slug for chapter in CHAPTERS}) != len(CHAPTERS):
    raise RuntimeError("chapter slugs must be unique")
