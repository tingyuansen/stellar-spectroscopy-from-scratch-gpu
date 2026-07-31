"""Provenance gates for reader-facing conceptual schematics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.textbook_schematic_specs import FIGURES


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "assets" / "schematics" / "MANIFEST.json"
CHAPTER_ONE_SOURCE = REPOSITORY_ROOT / "book" / "chapters" / "chapter_01.py"
CHAPTER_TWO_SOURCE = REPOSITORY_ROOT / "book" / "chapters" / "chapter_02.py"


class SchematicProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_hashes_every_asset(self) -> None:
        for asset in self.manifest["assets"]:
            with self.subTest(path=asset["path"]):
                path = REPOSITORY_ROOT / asset["path"]
                self.assertTrue(path.is_file())
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, asset["sha256"])

    def test_textbook_assets_have_owned_prompts(self) -> None:
        prompt_ids = {spec.id for spec in FIGURES}
        textbook_assets = [
            asset
            for asset in self.manifest["assets"]
            if asset["role"] == "original_textbook_conceptual_schematic"
        ]
        self.assertGreaterEqual(len(textbook_assets), 2)
        for asset in textbook_assets:
            with self.subTest(path=asset["path"]):
                self.assertIn(asset["prompt_id"], prompt_ids)
                self.assertEqual(
                    asset["prompt_source"],
                    "scripts/textbook_schematic_specs.py",
                )

    def test_chapter_one_displays_only_original_textbook_schematics(self) -> None:
        source = CHAPTER_ONE_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("assets/schematics/official/", source)
        self.assertIn("assets/schematics/textbook/ch01-forward-problem-v1.png", source)
        self.assertIn("assets/schematics/textbook/ch01-formation-depth-v1.png", source)

    def test_chapter_two_displays_only_original_textbook_schematics(self) -> None:
        source = CHAPTER_TWO_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("assets/schematics/official/", source)
        for name in (
            "ch02-ordered-depth-v1.png",
            "ch02-architecture-v1.png",
            "ch02-data-roles-v1.png",
            "ch02-populations-v1.png",
        ):
            with self.subTest(name=name):
                self.assertIn(f"assets/schematics/textbook/{name}", source)

    def test_chapter_three_registry_has_exactly_two_owned_specs(self) -> None:
        chapter_three = {spec.id: spec for spec in FIGURES if spec.chapter == 3}
        self.assertEqual(
            set(chapter_three),
            {"ch03-levels-to-charge", "ch03-packed-to-public"},
        )
        expected_paths = {
            "ch03-levels-to-charge": (
                "assets/schematics/textbook/ch03-levels-to-charge-v1.png"
            ),
            "ch03-packed-to-public": (
                "assets/schematics/textbook/ch03-packed-to-public-v1.png"
            ),
        }
        for figure_id, spec in chapter_three.items():
            with self.subTest(figure_id=figure_id):
                self.assertEqual(spec.asset_path, expected_paths[figure_id])
                self.assertTrue(spec.alt_text)
                self.assertTrue(spec.caption)
                self.assertIn("<strong>", spec.caption)

    def test_chapter_three_prompts_freeze_the_scientific_claims(self) -> None:
        chapter_three = {spec.id: spec for spec in FIGURES if spec.chapter == 3}
        closure_prompt = chapter_three["ch03-levels-to-charge"].prompt
        self.assertIn('labelled "update n_e"', closure_prompt)
        self.assertIn('labelled "enters every Saha ratio"', closure_prompt)
        self.assertIn("Neither feedback arrow may", closure_prompt)
        self.assertIn("must not point to the level ladder", closure_prompt)

        mapping_prompt = " ".join(chapter_three["ch03-packed-to-public"].prompt.split())
        for required_text in (
            "shape (D, 1006)",
            "shape (D, 6, 139)",
            "atomic species 0–98",
            "unused tail 99–138 in this atom-only route",
            "later: selected synthetic line cells",
            "It must not imply that molecules are confined",
            "not a reshape",
            "explicit placement",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, mapping_prompt)

    def test_generated_chapter_three_assets_have_reviewed_provenance(self) -> None:
        chapter_three_assets = {
            asset["prompt_id"]: asset
            for asset in self.manifest["assets"]
            if asset.get("primary_chapter") == 3
        }
        self.assertEqual(
            set(chapter_three_assets),
            {"ch03-levels-to-charge", "ch03-packed-to-public"},
        )
        for prompt_id, asset in chapter_three_assets.items():
            with self.subTest(prompt_id=prompt_id):
                self.assertEqual(
                    asset["prompt_source"],
                    "scripts/textbook_schematic_specs.py",
                )
                self.assertEqual(
                    asset["role"],
                    "original_textbook_conceptual_schematic",
                )
                for source in asset["source_image_inputs"]:
                    self.assertEqual(source["role"], "owned_prior_revision")
                    self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(
                    asset["style_rule_source"],
                    "scripts/textbook_schematic_specs.py:STYLE",
                )
                self.assertEqual(asset["scientific_review"]["status"], "accepted")
                self.assertGreaterEqual(
                    len(asset["scientific_review"]["checks"]),
                    3,
                )

    def test_chapter_four_registry_has_exactly_four_owned_specs(self) -> None:
        chapter_four = {spec.id: spec for spec in FIGURES if spec.chapter == 4}
        expected_paths = {
            "ch04-coupled-budgets": (
                "assets/schematics/textbook/ch04-coupled-budgets-v1.png"
            ),
            "ch04-newton-positivity": (
                "assets/schematics/textbook/ch04-newton-positivity-v1.png"
            ),
            "ch04-ordered-backends": (
                "assets/schematics/textbook/ch04-ordered-backends-v1.png"
            ),
            "ch04-catalog-to-public-lane": (
                "assets/schematics/textbook/ch04-catalog-to-public-lane-v1.png"
            ),
        }
        self.assertEqual(set(chapter_four), set(expected_paths))
        for figure_id, spec in chapter_four.items():
            with self.subTest(figure_id=figure_id):
                self.assertEqual(spec.asset_path, expected_paths[figure_id])
                self.assertTrue(spec.alt_text)
                self.assertTrue(spec.caption)
                self.assertIn("<strong>", spec.caption)

    def test_chapter_four_prompts_freeze_scientific_boundaries(self) -> None:
        chapter_four = {spec.id: spec for spec in FIGURES if spec.chapter == 4}
        self.assertIn(
            "one molecule, several coupled equations",
            chapter_four["ch04-coupled-budgets"].prompt,
        )
        positivity = chapter_four["ch04-newton-positivity"].prompt
        self.assertIn("reflection + shared scale", positivity)
        self.assertIn("multiplicative 1% floor", positivity)
        self.assertIn("Do not show logarithmic unknowns", positivity)

        ordered = chapter_four["ch04-ordered-backends"].prompt
        self.assertIn("previous converged depth seeds the next", ordered)
        self.assertIn("vmap after depth solves", ordered)
        self.assertIn("Do not draw arrows between different depths in", ordered)

        mapping = " ".join(chapter_four["ch04-catalog-to-public-lane"].prompt.split())
        for required_text in (
            "atmosphere catalog — 170 records",
            "synthesis catalog — 190 records",
            "54 line-list species codes",
            "column = species_code // 6 - 1",
            "normalized cube only — stage index 5",
            "51 inside 0–98; 3 at 129–131",
            "actual ion-stage cube unchanged",
            "276 -> CO 608 -> [depth, 5, 45]",
            "catalog row is not a public column",
            "Do not confine the mapping to columns 99–138",
            "do not highlight a constant-depth sheet as stage 5",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, mapping)

    def test_generated_chapter_four_assets_have_reviewed_provenance(self) -> None:
        chapter_four_assets = {
            asset["prompt_id"]: asset
            for asset in self.manifest["assets"]
            if asset.get("primary_chapter") == 4
        }
        self.assertEqual(
            set(chapter_four_assets),
            {
                "ch04-coupled-budgets",
                "ch04-newton-positivity",
                "ch04-ordered-backends",
                "ch04-catalog-to-public-lane",
            },
        )
        for prompt_id, asset in chapter_four_assets.items():
            with self.subTest(prompt_id=prompt_id):
                self.assertEqual(
                    asset["prompt_source"],
                    "scripts/textbook_schematic_specs.py",
                )
                self.assertEqual(
                    asset["role"],
                    "original_textbook_conceptual_schematic",
                )
                self.assertEqual(
                    asset["style_rule_source"],
                    "scripts/textbook_schematic_specs.py:STYLE",
                )
                self.assertEqual(asset["source_image_inputs"], [])
                self.assertEqual(asset["scientific_review"]["status"], "accepted")
                self.assertGreaterEqual(
                    len(asset["scientific_review"]["checks"]),
                    4,
                )

    def test_chapter_five_registry_has_four_original_specs(self) -> None:
        chapter_five = {spec.id: spec for spec in FIGURES if spec.chapter == 5}
        expected_paths = {
            "ch05-cross-section-to-opacity": (
                "assets/schematics/textbook/ch05-cross-section-to-opacity-v2.png"
            ),
            "ch05-absorption-vs-scattering": (
                "assets/schematics/textbook/ch05-absorption-vs-scattering-v3.png"
            ),
            "ch05-prange-columns": (
                "assets/schematics/textbook/ch05-prange-columns-v4.png"
            ),
            "ch05-two-grids-edge-triplet": (
                "assets/schematics/textbook/ch05-two-grids-edge-triplet-v4.png"
            ),
        }
        self.assertEqual(set(chapter_five), set(expected_paths))
        for figure_id, spec in chapter_five.items():
            with self.subTest(figure_id=figure_id):
                self.assertEqual(spec.asset_path, expected_paths[figure_id])
                self.assertTrue(spec.alt_text)
                self.assertIn("<strong>", spec.caption)

    def test_chapter_five_assets_have_reviewed_provenance(self) -> None:
        assets = {
            asset["prompt_id"]: asset
            for asset in self.manifest["assets"]
            if asset.get("primary_chapter") == 5
        }
        self.assertEqual(
            set(assets),
            {
                "ch05-cross-section-to-opacity",
                "ch05-absorption-vs-scattering",
                "ch05-prange-columns",
                "ch05-two-grids-edge-triplet",
            },
        )
        for asset in assets.values():
            self.assertEqual(
                asset["role"],
                "original_textbook_conceptual_schematic",
            )
            self.assertIn(
                asset["scientific_review"]["status"],
                {"accepted", "accepted_with_caption_guard"},
            )
            self.assertGreaterEqual(
                len(asset["scientific_review"]["checks"]),
                4,
            )

    def test_chapter_six_registry_has_four_original_specs(self) -> None:
        chapter_six = {spec.id: spec for spec in FIGURES if spec.chapter == 6}
        expected_paths = {
            "ch06-smooth-background-narrow-line": (
                "assets/schematics/textbook/ch06-smooth-background-narrow-line-v1.png"
            ),
            "ch06-two-levels-one-photon": (
                "assets/schematics/textbook/ch06-two-levels-one-photon-v1.png"
            ),
            "ch06-core-wings-convolution": (
                "assets/schematics/textbook/ch06-core-wings-convolution-v1.png"
            ),
            "ch06-one-record-many-depths": (
                "assets/schematics/textbook/ch06-one-record-many-depths-v2.png"
            ),
        }
        self.assertEqual(set(chapter_six), set(expected_paths))
        for figure_id, spec in chapter_six.items():
            with self.subTest(figure_id=figure_id):
                self.assertEqual(spec.asset_path, expected_paths[figure_id])
                self.assertTrue(spec.alt_text)
                self.assertIn("<strong>", spec.caption)
        record = chapter_six["ch06-one-record-many-depths"]
        self.assertIn('"tilde E_l  [cm^-1]"', record.prompt)
        self.assertIn("stored lower-excitation wavenumber", record.caption)

    def test_chapter_six_text_and_output_provenance_is_hash_bound(self) -> None:
        specs = {spec.id: spec for spec in FIGURES if spec.chapter == 6}
        assets = {
            asset["prompt_id"]: asset
            for asset in self.manifest["assets"]
            if asset.get("primary_chapter") == 6
        }
        self.assertEqual(set(assets), set(specs))
        def digest(value: str) -> str:
            return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()

        for prompt_id, asset in assets.items():
            with self.subTest(prompt_id=prompt_id):
                spec = specs[prompt_id]
                self.assertEqual(
                    asset["text_binding_convention"],
                    (
                        "SHA-256 of UTF-8 FigureSpec field after strip(), "
                        "with no trailing newline"
                    ),
                )
                self.assertEqual(
                    asset["accepted_prompt_sha256"],
                    digest(spec.prompt),
                )
                self.assertEqual(
                    asset["accepted_alt_text_sha256"],
                    digest(spec.alt_text),
                )
                self.assertEqual(
                    asset["accepted_caption_sha256"],
                    digest(spec.caption),
                )
                # This is the registry snapshot recorded when Chapter 6 was
                # reviewed. Later chapters legitimately append new FigureSpec
                # records, so the historical digest must remain well formed
                # rather than being rewritten to equal the evolving file.
                self.assertRegex(
                    asset["prompt_registry_sha256_at_review"],
                    r"^[0-9a-f]{64}$",
                )
                self.assertRegex(asset["generated_at"], r"^\d{4}-\d{2}-\d{2}$")
                self.assertGreaterEqual(
                    len(asset["scientific_review"]["checks"]),
                    4,
                )


if __name__ == "__main__":
    unittest.main()
