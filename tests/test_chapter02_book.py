"""Pedagogical, execution-shape, and source-display gates for Chapter 2."""

from __future__ import annotations

import unittest

from book.chapters.chapter_02 import build_notebook


class Chapter02BookTests(unittest.TestCase):
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
        self.assertEqual(len(exact_cells), 3)
        self.assertTrue(all(len(cell["source"].splitlines()) <= 35 for cell in exact_cells))

    def test_main_text_contains_limits_instead_of_exercises(self) -> None:
        markdown_source = "\n".join(
            cell["source"]
            for cell in self.cells
            if cell["cell_type"] == "markdown"
        )
        self.assertNotIn("## Exercises", markdown_source)
        self.assertNotIn("## Practice Exercises", markdown_source)
        self.assertIn("If there is only one", markdown_source)
        self.assertIn("one-frequency problem", markdown_source)
        self.assertIn("Movement I — Make one kernel trustworthy", markdown_source)
        self.assertIn("Movement II — Preserve meaning around the kernel", markdown_source)

    def test_reader_path_is_self_contained(self) -> None:
        complete_source = "\n".join(cell["source"] for cell in self.cells)
        self.assertNotIn("/Users/ysting/payne-zero", complete_source)
        self.assertNotIn("/Users/ysting/Source_Files_Not_For_Review", complete_source)

    def test_exact_public_names_and_distinct_representations_are_taught(self) -> None:
        complete_source = "\n".join(cell["source"] for cell in self.cells)
        for required_name in (
            "integrate_on_depth_grid",
            "integrate_optical_depth",
            "resolve_runtime",
            "partition_normalized_populations",
            "ion_stage_populations",
        ):
            with self.subTest(name=required_name):
                self.assertIn(required_name, complete_source)

    def test_late_abundance_and_atmosphere_apis_are_deferred(self) -> None:
        complete_source = "\n".join(cell["source"] for cell in self.cells)
        for deferred_name in (
            "complete_direct_abundance_vector",
            "DIRECT_XH_ATOMIC_NUMBERS",
            "ModelAtmosphere",
            "validate_atmosphere_npz",
            "atmosphere_product_role",
            "schema-v4",
        ):
            with self.subTest(name=deferred_name):
                self.assertNotIn(deferred_name, complete_source)

    def test_every_quantitative_figure_is_closed_after_showing(self) -> None:
        code_source = "\n".join(
            cell["source"] for cell in self.cells if cell["cell_type"] == "code"
        )
        self.assertEqual(code_source.count("plt.show()"), 2)
        self.assertEqual(code_source.count("plt.close(figure)"), 2)

    def test_chapter_closes_with_summary_and_causal_next_link(self) -> None:
        closing_source = self.cells[-1]["source"]
        self.assertIn("## 2.13 Chapter summary", closing_source)
        self.assertIn("### Next: count atoms, ions, and electrons", closing_source)
        self.assertIn("Chapter 3: Atoms, Ions, and Electrons", closing_source)
        self.assertIn("](/reader.html?ch=3)", closing_source)


if __name__ == "__main__":
    unittest.main()
