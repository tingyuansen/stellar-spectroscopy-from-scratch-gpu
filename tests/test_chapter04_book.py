"""Pedagogy, execution-shape, and continuity gates for Chapter 4."""

from __future__ import annotations

import unittest

from book.chapters.chapter_04 import build_notebook


class Chapter04BookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = build_notebook()
        cls.cells = cls.notebook["cells"]
        cls.complete_source = "\n".join(cell["source"] for cell in cls.cells)
        cls.markdown_source = "\n".join(
            cell["source"] for cell in cls.cells if cell["cell_type"] == "markdown"
        )

    def test_no_source_dump_or_exercise_section_inside_markdown(self) -> None:
        self.assertNotIn("```python", self.markdown_source)
        self.assertNotIn("## Exercises", self.markdown_source)
        self.assertNotIn("## Practice Exercises", self.markdown_source)

    def test_every_python_cell_compiles_before_execution(self) -> None:
        for index, cell in enumerate(self.cells):
            if cell["cell_type"] != "code":
                continue
            with self.subTest(cell=index):
                compile(cell["source"], f"chapter04-cell-{index}", "exec")

    def test_exactly_seventeen_visible_bite_sized_cells_are_bridged(self) -> None:
        visible_cells = [
            cell
            for cell in self.cells
            if "hide-input" not in cell.get("metadata", {}).get("tags", [])
        ]
        code_indices = [
            index
            for index, cell in enumerate(visible_cells)
            if cell["cell_type"] == "code"
        ]
        self.assertEqual(len(code_indices), 17)
        for index in code_indices:
            with self.subTest(cell=index):
                lines = visible_cells[index]["source"].splitlines()
                self.assertLessEqual(len(lines), 35)
                self.assertTrue(all(len(line) <= 92 for line in lines))
                self.assertEqual(visible_cells[index - 1]["cell_type"], "markdown")
                self.assertEqual(visible_cells[index + 1]["cell_type"], "markdown")

    def test_three_movements_follow_the_coupled_root(self) -> None:
        for movement in (
            "Movement I — One molecule couples two budgets",
            "Movement II — Keep one coupled root positive through depth",
            "Movement III — Cross the atmosphere/synthesis boundary honestly",
        ):
            with self.subTest(movement=movement):
                self.assertIn(movement, self.markdown_source)

    def test_density_space_backend_and_public_lane_claims_are_explicit(self) -> None:
        for required in (
            "ordinary number-density coordinates",
            "neither backend uses logarithmic Newton unknowns",
            "pressure-scaled continuation",
            "`@njit(cache=True, nogil=True)`",
            "not a `prange` loop",
            "`jacrev`",
            "`vmap`",
            "fixed public builder",
            "partition_normalized_populations",
            "ion_stage_populations",
            "170",
            "190",
            "54",
            "276 → 608 → [depth, 5, 45]",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.complete_source)

    def test_reader_path_is_self_contained(self) -> None:
        self.assertNotIn("/Users/ysting/payne-zero", self.complete_source)
        self.assertNotIn(
            "/Users/ysting/Source_Files_Not_For_Review",
            self.complete_source,
        )

    def test_four_original_schematics_and_two_one_panel_figures_are_used(
        self,
    ) -> None:
        for schematic in (
            "ch04-coupled-budgets-v1.png",
            "ch04-newton-positivity-v1.png",
            "ch04-ordered-backends-v1.png",
            "ch04-catalog-to-public-lane-v1.png",
        ):
            self.assertIn(schematic, self.markdown_source)
        self.assertNotIn("payne-zero-website", self.complete_source)
        code_source = "\n".join(
            cell["source"] for cell in self.cells if cell["cell_type"] == "code"
        )
        self.assertEqual(code_source.count("single_panel()"), 2)
        self.assertEqual(code_source.count("plt.show()"), 2)
        self.assertEqual(code_source.count("plt.close(figure)"), 2)

    def test_chapter_closes_with_summary_and_causal_next_link(self) -> None:
        closing_source = self.cells[-1]["source"]
        self.assertIn("## 4.18 Chapter summary", closing_source)
        self.assertIn("Next: let the particles interact with light", closing_source)
        self.assertIn("Chapter 5: Continuous Opacity and Scattering", closing_source)
        self.assertIn("](/reader.html?ch=5)", closing_source)


if __name__ == "__main__":
    unittest.main()
