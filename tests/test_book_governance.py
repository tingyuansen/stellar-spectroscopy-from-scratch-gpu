"""Book-wide gates that every published canonical chapter must satisfy."""

from __future__ import annotations

import importlib
import re
import unittest
from pathlib import Path

from book.registry import CHAPTERS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class BookGovernanceTests(unittest.TestCase):
    """Keep global promises enforceable as the chapter count grows."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.published = []
        for chapter in CHAPTERS:
            if not chapter.available:
                continue
            module = importlib.import_module(
                f"book.chapters.{chapter.module}"
            )
            notebook = module.build_notebook()
            cls.published.append((chapter, notebook["cells"]))

    def test_published_chapters_have_no_detached_exercise_section(self) -> None:
        detached_heading = re.compile(
            r"(?im)^#{1,6}\s*(?:practice\s+)?exercises?\b"
            r"|^#{1,6}\s*(?:problem\s+sets?|homework)\b"
        )
        for chapter, cells in self.published:
            markdown = "\n".join(
                cell["source"]
                for cell in cells
                if cell["cell_type"] == "markdown"
            )
            with self.subTest(chapter=chapter.number):
                self.assertIsNone(detached_heading.search(markdown))

    def test_published_chapters_are_self_contained(self) -> None:
        forbidden = (
            "/Users/ysting/payne-zero",
            "/Users/ysting/Source_Files_Not_For_Review",
        )
        for chapter, cells in self.published:
            source = "\n".join(cell["source"] for cell in cells)
            with self.subTest(chapter=chapter.number):
                for path in forbidden:
                    self.assertNotIn(path, source)

    def test_markdown_never_hides_large_python_source_blocks(self) -> None:
        for chapter, cells in self.published:
            markdown = "\n".join(
                cell["source"]
                for cell in cells
                if cell["cell_type"] == "markdown"
            )
            with self.subTest(chapter=chapter.number):
                self.assertNotIn("```python", markdown)

    def test_published_chapters_close_with_summary_and_next_dependency(self) -> None:
        for chapter, cells in self.published:
            closing = cells[-1]["source"]
            with self.subTest(chapter=chapter.number):
                self.assertIn("Chapter summary", closing)
                if chapter.number < 15:
                    self.assertIn("### Next:", closing)
                    self.assertIn(
                        f"](/reader.html?ch={chapter.number + 1})",
                        closing,
                    )

    def test_planned_next_links_render_an_in_construction_page(self) -> None:
        reader = (REPOSITORY_ROOT / "reader.html").read_text()
        builder = (REPOSITORY_ROOT / "scripts" / "build_book.py").read_text()
        self.assertIn("if (!meta.available)", reader)
        self.assertIn("Chapter in construction", reader)
        self.assertIn("window.BOOK.flat.map(chapter => [chapter.n, chapter])", builder)


if __name__ == "__main__":
    unittest.main()
