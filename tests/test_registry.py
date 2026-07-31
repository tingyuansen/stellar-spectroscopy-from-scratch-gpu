"""Structural tests for the one canonical chapter registry."""

from __future__ import annotations

import unittest

from book.registry import CHAPTERS, PARTS


class RegistryTests(unittest.TestCase):
    def test_chapter_numbers_are_consecutive(self) -> None:
        self.assertEqual(
            [chapter.number for chapter in CHAPTERS],
            list(range(1, 16)),
        )

    def test_every_part_has_chapters(self) -> None:
        self.assertTrue(all(part.chapters for part in PARTS))

    def test_slugs_are_unique(self) -> None:
        slugs = [chapter.slug for chapter in CHAPTERS]
        self.assertEqual(len(slugs), len(set(slugs)))


if __name__ == "__main__":
    unittest.main()
