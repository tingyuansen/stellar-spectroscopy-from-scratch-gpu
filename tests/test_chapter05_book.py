"""Pedagogy, execution-shape, and continuity gates for Chapter 5."""

from __future__ import annotations

import unittest

from book.chapters.chapter_05 import build_notebook


class Chapter05BookTests(unittest.TestCase):
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

    def test_no_source_dump_or_detached_exercise_section(self) -> None:
        self.assertNotIn("```python", self.markdown_source)
        for forbidden in (
            "## Exercises",
            "## Practice Exercises",
            "## Problem Set",
            "## Homework",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden.casefold(), self.markdown_source.casefold())

    def test_every_python_cell_compiles_before_execution(self) -> None:
        for index, cell in enumerate(self.cells):
            if cell["cell_type"] != "code":
                continue
            with self.subTest(cell=index):
                compile(cell["source"], f"chapter05-cell-{index}", "exec")

    def test_exactly_sixteen_visible_bite_sized_cells_are_bridged(self) -> None:
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
        self.assertEqual(len(code_indices), 16)
        for index in code_indices:
            with self.subTest(cell=index):
                lines = visible_cells[index]["source"].splitlines()
                self.assertLessEqual(len(lines), 35)
                self.assertTrue(all(len(line) <= 92 for line in lines))
                self.assertEqual(visible_cells[index - 1]["cell_type"], "markdown")
                self.assertEqual(visible_cells[index + 1]["cell_type"], "markdown")

    def test_four_movements_follow_one_continuum_question(self) -> None:
        for movement in (
            "Movement I — From one interaction to a light-particle background",
            "Movement II — Complete the physical budget",
            "Movement III — One physical budget, two exact grids",
            "Movement IV — Bind the exact consumers",
        ):
            with self.subTest(movement=movement):
                self.assertIn(movement, self.markdown_source)
        self.assertIn(
            "Between the narrow dips, why is the gas not transparent?",
            self.markdown_source,
        )

    def test_exact_route_boundaries_and_population_owners_are_explicit(self) -> None:
        required = (
            "ContinuumAtmosphereState",
            "27-field",
            "18-field",
            "partition-normalized",
            "actual ion",
            "30,000",
            "343",
            "344",
            "edge triplets",
            "coulomb_table_energy_first=False",
            "sampled diagnostic",
            "precomputed extension",
            "12-point",
            "roundoff",
            "continuum_absorption",
            "continuum_scattering",
            "continuum_source",
        )
        for phrase in required:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.complete_source)

    def test_light_depth_and_helium_roles_are_unambiguous(self) -> None:
        self.assertIn("light_depth = 3", self.complete_source)
        self.assertNotIn("[indices, 3]", self.complete_source)
        self.assertIn(
            "He I and He II supply bound-free terms and\nhigh-level tails",
            self.markdown_source,
        )
        self.assertIn("bare He III has no bound levels", self.markdown_source)

    def test_molecular_hydrogen_recomputation_is_not_blended_with_hminus(self) -> None:
        self.assertIn(
            "hydrogen_partition_normalized_ion_stage_populations[:, 0]",
            self.complete_source,
        )
        self.assertIn("hydrogen_neutral_population", self.complete_source)
        self.assertIn(
            r"\widetilde n_{\rm H\,I}^{\,{\rm local}}",
            self.complete_source,
        )
        self.assertIn("dimensionless H-ground departure", self.markdown_source)
        self.assertIn("table-derived association factor", self.markdown_source)
        self.assertIn(r"\mathrm{cm}^{-3})^2\mathrm{cm}^{3}", self.markdown_source)

    def test_consumer_views_tables_and_helium_population_are_not_blended(self) -> None:
        self.assertIn(
            "two declared consumer views and their owned immutable table bundles",
            self.complete_source.casefold(),
        )
        self.assertIn("not one shared mapping or", self.complete_source)
        self.assertIn("singly ionized He vectors", self.complete_source)

    def test_standalone_table_dtype_exception_is_executable(self) -> None:
        self.assertIn(
            'ContinuumTables.from_npz(\n    standalone_path, device="cpu", dtype=None',
            self.complete_source,
        )
        self.assertIn("standalone_tables.dtype", self.complete_source)
        self.assertIn("does not invoke the resolver", self.markdown_source)

    def test_sampled_neutral_metals_and_extension_only_grids_are_separated(self) -> None:
        self.assertIn(
            "full C I, Mg I, Al I, Si I, and Fe I",
            self.markdown_source,
        )
        self.assertIn("one frequency at a time", self.markdown_source)
        self.assertIn("N I, O I, Mg II, and Ca II", self.markdown_source)

    def test_standard_component_order_and_closure_are_visible(self) -> None:
        self.assertIn("absorption_component_names", self.complete_source)
        self.assertIn("scattering_component_names", self.complete_source)
        self.assertIn("absorption_residual_cm2_per_g", self.complete_source)
        self.assertIn("scattering_residual_cm2_per_g", self.complete_source)

    def test_constant_tiers_and_plot_depths_are_declared(self) -> None:
        self.assertIn("two constant tiers", self.markdown_source)
        self.assertIn("CODATA-exact", self.markdown_source)
        self.assertIn("uses depth index 4", self.markdown_source)
        self.assertIn("uses depth\nindex 3", self.markdown_source)

    def test_source_numerator_and_continuum_source_units_close(self) -> None:
        self.assertIn(
            r"erg s\(^{-1}\) cm\(^{-2}\) sr\(^{-1}\) Hz\(^{-1}\)",
            self.markdown_source,
        )
        self.assertIn(r"(\mathrm{cm}^{2}\,\mathrm{g}^{-1})", self.markdown_source)
        self.assertIn("restores the", self.markdown_source)

    def test_hminus_local_population_and_edge_share_binding_energy(self) -> None:
        self.assertIn(r"n_{\rm H^-}^{\rm local}", self.markdown_source)
        self.assertIn(r"2.4148\times10^{15}", self.markdown_source)
        self.assertIn("dimensionless departure coefficients", self.markdown_source)
        self.assertIn(r"h\nu_0=0.754209", self.markdown_source)

    def test_molecular_column_layer_and_disable_gates_are_taught(self) -> None:
        self.assertIn("min(temperature) < 9000 K", self.markdown_source)
        self.assertIn("all-warm column skips the whole composite", self.markdown_source)
        self.assertIn("enable_molecules=False", self.markdown_source)
        self.assertIn(
            "does not\ndisable the continuum-local",
            self.markdown_source,
        )

    def test_all_five_atmosphere_grid_start_indices_are_reproducible(self) -> None:
        for start_index in ("11601", "9599", "7027", "3577"):
            with self.subTest(start_index=start_index):
                self.assertIn(start_index, self.markdown_source)
        self.assertIn(r"T_{\rm eff}\geq30000", self.markdown_source)
        self.assertIn("equality belongs to the hotter interval", self.markdown_source)

    def test_frequency_native_probe_is_checked_bitwise_in_the_reader(self) -> None:
        self.assertIn(
            "diagnostic.frequency_hz.view(np.uint64)",
            self.complete_source,
        )

    def test_pinned_oracle_provenance_is_named_once_without_branding(self) -> None:
        self.assertEqual(self.markdown_source.count("Payne Zero"), 1)
        self.assertIn(
            "pinned Payne Zero comparison golden",
            self.markdown_source,
        )
        self.assertIn(
            "comparison_frequency_hz.view(np.uint64)",
            self.complete_source,
        )

    def test_reader_path_is_self_contained(self) -> None:
        self.assertNotIn("/Users/ysting/payne-zero", self.complete_source)
        self.assertNotIn(
            "/Users/ysting/Source_Files_Not_For_Review",
            self.complete_source,
        )

    def test_forward_fixture_role_is_honest(self) -> None:
        self.assertIn("controlled six-depth stage inputs", self.markdown_source)
        self.assertIn("solely to isolate continuum physics", self.markdown_source)
        self.assertIn("not\nevidence that a stellar atmosphere has converged", self.markdown_source)

    def test_four_original_schematics_and_six_one_panel_figures_are_used(
        self,
    ) -> None:
        for schematic in (
            "ch05-cross-section-to-opacity-v2.png",
            "ch05-absorption-vs-scattering-v3.png",
            "ch05-prange-columns-v4.png",
            "ch05-two-grids-edge-triplet-v4.png",
        ):
            self.assertIn(schematic, self.markdown_source)
        code_source = "\n".join(
            cell["source"]
            for cell in self.cells
            if cell["cell_type"] == "code"
        )
        self.assertEqual(code_source.count("_show_one_panel("), 7)
        self.assertEqual(code_source.count("single_panel()"), 1)
        self.assertEqual(code_source.count("plt.show()"), 1)
        self.assertEqual(code_source.count("plt.close(figure)"), 1)

    def test_chapter_closes_with_summary_and_causal_next_link(self) -> None:
        closing_source = self.cells[-1]["source"]
        self.assertIn("## 5.14 Chapter summary", closing_source)
        self.assertIn("### Next:", closing_source)
        self.assertIn("Chapter 6: One Spectral Line", closing_source)
        self.assertIn("](/reader.html?ch=6)", closing_source)


if __name__ == "__main__":
    unittest.main()
