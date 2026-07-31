"""Pedagogical and source-display gates for Chapter 1."""

from __future__ import annotations

import unittest

from book.chapters.chapter_01 import build_notebook


FORBIDDEN_PARALLEL_API_NAMES = (
    "StellarLabels",
    "SpectrumWindow",
    "SpectrumResult",
    "AtmosphereRole",
    "effective_temperature_K",
    "log10_surface_gravity_cm_s2",
    "total_flux_per_nm",
    "continuum_flux_per_nm",
)


class Chapter01BookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = build_notebook()
        cls.cells = cls.notebook["cells"]

    def test_no_source_dump_inside_markdown(self) -> None:
        markdown_sources = [
            cell["source"]
            for cell in self.cells
            if cell["cell_type"] == "markdown"
        ]
        self.assertFalse(any("```python" in source for source in markdown_sources))

    def test_visible_code_cells_are_bite_sized_and_bridged(self) -> None:
        visible_cells = [
            cell
            for cell in self.cells
            if "hide-input" not in cell.get("metadata", {}).get("tags", [])
        ]
        for index, cell in enumerate(visible_cells):
            if cell["cell_type"] != "code":
                continue
            with self.subTest(cell=index):
                self.assertLessEqual(len(cell["source"].splitlines()), 35)
                self.assertEqual(visible_cells[index - 1]["cell_type"], "markdown")
                self.assertEqual(visible_cells[index + 1]["cell_type"], "markdown")

    def test_exact_source_cells_are_small_and_identified(self) -> None:
        exact_cells = [
            cell
            for cell in self.cells
            if "exact-payne-zero-source"
            in cell.get("metadata", {}).get("tags", [])
        ]
        self.assertEqual(len(exact_cells), 2)
        self.assertTrue(all(len(cell["source"].splitlines()) <= 20 for cell in exact_cells))

    def test_no_invented_public_api_names_remain(self) -> None:
        complete_source = "\n".join(cell["source"] for cell in self.cells)
        for forbidden in FORBIDDEN_PARALLEL_API_NAMES:
            with self.subTest(name=forbidden):
                self.assertNotIn(forbidden, complete_source)

    def test_every_displayed_figure_is_closed_after_showing(self) -> None:
        code_source = "\n".join(
            cell["source"] for cell in self.cells if cell["cell_type"] == "code"
        )
        self.assertEqual(code_source.count("plt.show()"), 5)
        self.assertEqual(code_source.count("plt.close(figure)"), 5)

    def test_reader_path_is_self_contained(self) -> None:
        complete_source = "\n".join(cell["source"] for cell in self.cells)
        self.assertNotIn("/Users/ysting/payne-zero", complete_source)
        self.assertNotIn("/Users/ysting/Source_Files_Not_For_Review", complete_source)

    def test_chapter_closes_with_summary_and_causal_next_link(self) -> None:
        closing_source = self.cells[-1]["source"]
        self.assertIn("## 1.12 Chapter summary", closing_source)
        self.assertIn("### Next: make the calculation trustworthy and fast", closing_source)
        self.assertIn(
            "Chapter 2: From Equations to Fast, Trustworthy Kernels and Explicit",
            closing_source,
        )
        self.assertIn("](/reader.html?ch=2)", closing_source)

    def test_grey_scaffold_close_stays_physical_and_defers_late_apis(self) -> None:
        closing_index = next(
            index
            for index, cell in enumerate(self.cells)
            if "## 1.11 Read the scaffold as a causal chain" in cell["source"]
        )
        closing_source = "\n".join(
            cell["source"] for cell in self.cells[closing_index:]
        )
        for physical_limit in (
            "wavelength-dependent absorption",
            "opacity from the actual mixture",
            "flows and accelerations",
            "feedback that makes a stellar atmosphere self-consistent",
        ):
            with self.subTest(limit=physical_limit):
                self.assertIn(physical_limit, closing_source)
        for premature_name in (
            "ModelAtmosphere",
            "InitializedAtmosphere",
            "Spectrum",
            "atmosphere_product_role",
            "Payne Zero",
        ):
            with self.subTest(name=premature_name):
                self.assertNotIn(premature_name, closing_source)

    def test_useful_variations_are_not_deferred_to_exercises(self) -> None:
        markdown_source = "\n".join(
            cell["source"] for cell in self.cells if cell["cell_type"] == "markdown"
        )
        self.assertNotIn("## Exercises", markdown_source)
        self.assertNotIn("## Practice Exercises", markdown_source)


if __name__ == "__main__":
    unittest.main()
