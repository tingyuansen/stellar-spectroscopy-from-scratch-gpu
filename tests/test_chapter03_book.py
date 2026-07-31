"""Pedagogy, execution-shape, and continuity gates for Chapter 3."""

from __future__ import annotations

import unittest

from book.chapters.chapter_03 import build_notebook


class Chapter03BookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = build_notebook()
        cls.cells = cls.notebook["cells"]
        cls.complete_source = "\n".join(cell["source"] for cell in cls.cells)
        cls.markdown_source = "\n".join(
            cell["source"]
            for cell in cls.cells
            if cell["cell_type"] == "markdown"
        )

    def test_no_source_dump_inside_markdown(self) -> None:
        self.assertNotIn("```python", self.markdown_source)

    def test_every_python_cell_compiles_before_execution(self) -> None:
        for index, cell in enumerate(self.cells):
            if cell["cell_type"] != "code":
                continue
            with self.subTest(cell=index):
                compile(cell["source"], f"chapter03-cell-{index}", "exec")

    def test_exactly_twenty_one_visible_bite_sized_cells_are_bridged(self) -> None:
        visible_cells = [
            cell
            for cell in self.cells
            if "hide-input" not in cell.get("metadata", {}).get("tags", [])
        ]
        visible_code_indices = [
            index
            for index, cell in enumerate(visible_cells)
            if cell["cell_type"] == "code"
        ]
        self.assertEqual(len(visible_code_indices), 21)
        for index in visible_code_indices:
            cell = visible_cells[index]
            with self.subTest(cell=index):
                lines = cell["source"].splitlines()
                self.assertLessEqual(len(lines), 35)
                self.assertTrue(all(len(line) <= 92 for line in lines))
                self.assertEqual(visible_cells[index - 1]["cell_type"], "markdown")
                self.assertEqual(visible_cells[index + 1]["cell_type"], "markdown")

    def test_main_text_contains_three_movements_and_no_exercises(self) -> None:
        self.assertNotIn("## Exercises", self.markdown_source)
        self.assertNotIn("## Practice Exercises", self.markdown_source)
        for movement in (
            "Movement I — Count states at one depth",
            "Movement II — Close the atomic gas",
            "Movement III — Move the state without changing its claim",
        ):
            with self.subTest(movement=movement):
                self.assertIn(movement, self.markdown_source)

    def test_reader_path_is_self_contained(self) -> None:
        self.assertNotIn("/Users/ysting/payne-zero", self.complete_source)
        self.assertNotIn(
            "/Users/ysting/Source_Files_Not_For_Review",
            self.complete_source,
        )

    def test_exact_names_layouts_and_claim_boundary_are_taught(self) -> None:
        for required in (
            "ion_stage_populations",
            "partition_normalized_populations",
            "electron_density",
            "total_nuclei_number_density",
            "mass_density",
            "fractional_doppler_widths",
            "specific_internal_energy",
            "(D,1006)",
            "(D,99,6)",
            "(D,6,139)",
            "solve_population_state",
            "solve_population_state_at_electron_density",
            "fixed-\\(n_e\\)",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.complete_source)

    def test_three_figures_are_shown_and_closed(self) -> None:
        code_source = "\n".join(
            cell["source"] for cell in self.cells if cell["cell_type"] == "code"
        )
        self.assertEqual(code_source.count("plt.show()"), 3)
        self.assertEqual(code_source.count("plt.close(figure)"), 3)

    def test_only_owned_chapter_three_schematics_are_referenced(self) -> None:
        self.assertIn("ch03-levels-to-charge-v1.png", self.markdown_source)
        self.assertIn("ch03-packed-to-public-v1.png", self.markdown_source)
        self.assertNotIn("payne-zero-website", self.complete_source)

    def test_chapter_closes_with_summary_and_causal_next_link(self) -> None:
        closing_source = self.cells[-1]["source"]
        self.assertIn("## 3.12 Chapter summary", closing_source)
        self.assertIn("### Next: let atoms bind into molecules", closing_source)
        self.assertIn("Chapter 4: Molecules and Coupled Equilibrium", closing_source)
        self.assertIn("](/reader.html?ch=4)", closing_source)


if __name__ == "__main__":
    unittest.main()
