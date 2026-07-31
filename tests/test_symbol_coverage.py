"""Reproducibility and semantic-ownership gates for the public-symbol ledger."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import scripts.apply_symbol_ownership_overrides as semantic_registry
from scripts.apply_symbol_ownership_overrides import (
    ACCEPTED_ARTIFACT_SYMBOLS,
    ACCEPTED_CHAPTER_AUTHORITIES,
    API_ALIAS_TARGETS,
    CHAPTER4_ACCEPTED_SYMBOLS,
    CHAPTER5_ACCEPTED_SYMBOLS,
    CHAPTER5_ATMOSPHERE_SYMBOLS,
    CHAPTER5_SYNTHESIS_SYMBOLS,
    DEFAULT_BEARING_PUBLIC_CALLABLES,
    DEFAULT_CALLABLE_REVIEWS,
    DEFAULT_EFFECT_CATEGORIES,
    DEFAULT_PARAMETER_REVIEWS,
    EXACT_DEFAULT_SYNTAX,
    EXPLICIT_DEFAULT_PARAMETER_CONTRACTS,
    EXPLICIT_DEFAULT_CALLABLES,
    PRECISION,
    REQUIRED_EXPLICIT_SYMBOLS,
    REVIEWED_DEFAULT_CONTRACTS,
    REVIEWED_MODULE_POLICIES,
    REVIEWED_MODULE_SNAPSHOTS,
    REVIEWED_SOURCE_SHA256,
    SEMANTIC_DISPOSITIONS,
    SOURCE_DEFAULT_PARAMETER_MANIFEST,
    SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE,
    VERIFIED_STATUS_PROMOTION_GROUPS,
    _accepted_artifact_join,
    apply_overrides,
    validate_complete_semantic_proof,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_SOURCE_ROOT = Path("/Users/ysting/payne-zero")
LEDGER_PATH = REPOSITORY_ROOT / "audit/paynezero_symbol_coverage.json"
REVIEWED_RECORD_COUNT = 1501
REVIEWED_QUALIFIED_NAME_COUNT = 1443
MODULE_DEFAULT_RECORD_COUNT = 0


def all_records(ledger: dict) -> list[dict]:
    """Flatten both package symbol lists without collapsing duplicate kinds."""

    return [
        symbol
        for package in ledger["packages"].values()
        for symbol in package["symbols"]
    ]


def default_bearing_public_callables_from_source(inventory: dict) -> set[str]:
    """Reconstruct the complete reviewed default-bearing surface from source."""

    reviewed_modules = set(REVIEWED_MODULE_SNAPSHOTS)
    discovered: set[str] = set()
    for package_name, package in inventory["packages"].items():
        for module in package["modules"]:
            qualified_module = f"{package_name}.{module['module']}"
            if qualified_module not in reviewed_modules:
                continue
            source_path = PINNED_SOURCE_ROOT / module["relative_path"]
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_"):
                        continue
                    if node.args.defaults or any(
                        default is not None for default in node.args.kw_defaults
                    ):
                        discovered.add(f"{qualified_module}.{node.name}")
                    continue
                if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                    continue
                for child in node.body:
                    if not isinstance(
                        child,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    ):
                        continue
                    is_constructor = child.name == "__init__"
                    if child.name.startswith("_") and not is_constructor:
                        continue
                    if child.args.defaults or any(
                        default is not None for default in child.args.kw_defaults
                    ):
                        discovered.add(f"{qualified_module}.{node.name}.{child.name}")
    return discovered


def default_parameter_manifest_from_source(
    inventory: dict,
) -> dict[str, dict[str, dict[str, str]]]:
    """Recover every exact default source segment and AST from pinned source."""

    reviewed = set(DEFAULT_BEARING_PUBLIC_CALLABLES)
    manifest: dict[str, dict[str, dict[str, str]]] = {}
    for package_name, package in inventory["packages"].items():
        for module in package["modules"]:
            qualified_module = f"{package_name}.{module['module']}"
            if qualified_module not in REVIEWED_MODULE_SNAPSHOTS:
                continue
            source_path = PINNED_SOURCE_ROOT / module["relative_path"]
            source_text = source_path.read_text(encoding="utf-8")
            tree = ast.parse(source_text)
            callables = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    callables.append((f"{qualified_module}.{node.name}", node))
                    continue
                if not isinstance(node, ast.ClassDef):
                    continue
                callables.extend(
                    (f"{qualified_module}.{node.name}.{child.name}", child)
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
            for qualified_name, node in callables:
                if qualified_name not in reviewed:
                    continue
                defaults: dict[str, dict[str, str]] = {}
                positional = node.args.posonlyargs + node.args.args
                if node.args.defaults:
                    for argument, default_node in zip(
                        positional[-len(node.args.defaults) :],
                        node.args.defaults,
                        strict=True,
                    ):
                        defaults[argument.arg] = {
                            "source_default": ast.get_source_segment(
                                source_text,
                                default_node,
                            ),
                            "default_ast": ast.dump(
                                default_node,
                                include_attributes=False,
                            ),
                        }
                for argument, default_node in zip(
                    node.args.kwonlyargs,
                    node.args.kw_defaults,
                    strict=True,
                ):
                    if default_node is None:
                        continue
                    defaults[argument.arg] = {
                        "source_default": ast.get_source_segment(
                            source_text,
                            default_node,
                        ),
                        "default_ast": ast.dump(
                            default_node,
                            include_attributes=False,
                        ),
                    }
                manifest[qualified_name] = {
                    parameter_name: defaults[parameter_name]
                    for parameter_name in sorted(defaults)
                }
    return {
        qualified_name: manifest[qualified_name] for qualified_name in sorted(manifest)
    }


class SymbolCoverageTests(unittest.TestCase):
    """Keep 1,501 public records visible and mixed branches semantically reviewed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        cls.inventory = json.loads(
            (REPOSITORY_ROOT / "audit/paynezero_symbols.json").read_text(
                encoding="utf-8"
            )
        )
        cls.records = all_records(cls.ledger)
        cls.by_name = defaultdict(list)
        for record in cls.records:
            cls.by_name[record["qualified_name"]].append(record)

    def one(self, qualified_name: str) -> dict:
        """Return the sole record, or a representative duplicate-kind record."""

        self.assertIn(qualified_name, self.by_name)
        return self.by_name[qualified_name][0]

    def raw_module(
        self,
        inventory: dict,
        qualified_module: str,
    ) -> dict:
        """Return one module descriptor from a raw inventory."""

        package_name, module_name = qualified_module.split(".", maxsplit=1)
        return next(
            module
            for module in inventory["packages"][package_name]["modules"]
            if module["module"] == module_name
        )

    def raw_callable(
        self,
        inventory: dict,
        qualified_name: str,
    ) -> dict:
        """Return one public function, constructor, or method descriptor."""

        package_name, module_name, *object_parts = qualified_name.split(".")
        module = self.raw_module(inventory, f"{package_name}.{module_name}")
        if len(object_parts) == 1:
            return next(
                function
                for function in module["public_functions"]
                if function["name"] == object_parts[0]
            )
        class_item = next(
            class_item
            for class_item in module["public_classes"]
            if class_item["name"] == object_parts[0]
        )
        if object_parts[1] == "__init__":
            return class_item["constructor"]
        return next(
            method
            for method in class_item["public_methods"]
            if method["name"] == object_parts[1]
        )

    def test_inventory_is_exhaustive_and_distinguishes_export_records(self) -> None:
        packages = self.ledger["packages"]
        self.assertEqual(
            sum(package["module_count"] for package in packages.values()),
            58,
        )
        declared_symbol_count = sum(
            package["symbol_count"] for package in packages.values()
        )
        self.assertEqual(declared_symbol_count, 1501)
        self.assertEqual(len(self.records), declared_symbol_count)
        qualified_kinds = {
            (symbol["qualified_name"], symbol["kind"]) for symbol in self.records
        }
        self.assertEqual(len(qualified_kinds), declared_symbol_count)
        # Fifty-eight in-module exports intentionally share a qualified name
        # with their function/class/constant definition but have a distinct kind.
        self.assertEqual(
            declared_symbol_count - len(self.by_name),
            58,
        )

    def test_semantic_review_counts_and_vocabulary_are_exact(self) -> None:
        reviewed = [
            record
            for record in self.records
            if record["mapping_precision"] == PRECISION
        ]
        defaults = [
            record
            for record in self.records
            if record["mapping_precision"] == "module_default"
        ]
        self.assertEqual(len(reviewed), REVIEWED_RECORD_COUNT)
        self.assertEqual(
            len({record["qualified_name"] for record in reviewed}),
            REVIEWED_QUALIFIED_NAME_COUNT,
        )
        self.assertEqual(len(defaults), MODULE_DEFAULT_RECORD_COUNT)
        self.assertEqual(
            Counter(record["semantic_disposition"] for record in reviewed),
            {
                "taught": 899,
                "plumbing-only": 357,
                "composed": 184,
                "diagnostic-only": 41,
                "compatibility-only": 13,
                "unsupported": 7,
            },
        )
        self.assertEqual(
            Counter(record["semantic_review_source"] for record in reviewed),
            {
                "reviewed_module_registry": 942,
                "explicit_api_alias_registry": 198,
                "explicit_symbol_registry": 309,
                "explicit_default_callable_registry": 52,
            },
        )
        self.assertEqual(
            Counter(record["status"] for record in reviewed),
            {
                "verified": 926,
                "integrated": 401,
                "implemented": 81,
                "boundary": 61,
                "planned": 32,
            },
        )
        for record in reviewed:
            with self.subTest(
                symbol=record["qualified_name"],
                kind=record["kind"],
            ):
                self.assertIn(
                    record["semantic_disposition"],
                    SEMANTIC_DISPOSITIONS,
                )
                self.assertEqual(
                    record["source_spelling"],
                    record["qualified_name"].rsplit(".", maxsplit=1)[-1],
                )
                self.assertTrue(record["responsibility"])
                self.assertTrue(record["gate"])
                self.assertTrue(record["status"])
                self.assertTrue(record["primary_location"])
                self.assertRegex(record["source_module_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(
                    record["source_public_surface_sha256"],
                    r"^[0-9a-f]{64}$",
                )
                self.assertRegex(
                    record["source_public_object_sha256"],
                    r"^[0-9a-f]{64}$",
                )

    def test_later_chapter_status_promotions_are_exact_and_evidence_bound(
        self,
    ) -> None:
        promoted: set[str] = set()
        self.assertEqual(len(VERIFIED_STATUS_PROMOTION_GROUPS), 7)
        for group_name, group in VERIFIED_STATUS_PROMOTION_GROUPS.items():
            with self.subTest(group=group_name):
                evidence = tuple(group["evidence"])
                symbols = set(group["symbols"])
                self.assertTrue(evidence)
                self.assertTrue(symbols)
                self.assertFalse(promoted & symbols)
                for evidence_path in evidence:
                    self.assertTrue((REPOSITORY_ROOT / evidence_path).is_file())
                    self.assertTrue(evidence_path.startswith("tests/test_chapter"))
                for qualified_name in symbols:
                    self.assertIn(qualified_name, self.by_name)
                    for record in self.by_name[qualified_name]:
                        self.assertEqual(record["status"], "verified")
                promoted.update(symbols)
        self.assertEqual(len(promoted), 105)

    def test_reviewed_module_snapshot_registry_is_complete(self) -> None:
        reviewed_module_records = [
            record
            for record in self.records
            if record.get("semantic_review_source") == "reviewed_module_registry"
        ]
        self.assertEqual(len(REVIEWED_MODULE_SNAPSHOTS), 55)
        self.assertEqual(set(REVIEWED_MODULE_SNAPSHOTS), set(REVIEWED_SOURCE_SHA256))
        self.assertEqual(
            sum(count for count, _digest in REVIEWED_MODULE_SNAPSHOTS.values()),
            1303,
        )
        # Exact symbol/default reviews replace the inherited module tag on
        # every branch-sensitive public callable.
        self.assertEqual(len(reviewed_module_records), 942)
        reviewed_modules = {
            ".".join(record["qualified_name"].split(".")[:2])
            for record in reviewed_module_records
        }
        self.assertTrue(reviewed_modules.issubset(REVIEWED_MODULE_SNAPSHOTS))

    def test_every_required_branch_symbol_is_explicit_not_fallback(self) -> None:
        self.assertEqual(len(REQUIRED_EXPLICIT_SYMBOLS), 335)
        for qualified_name in sorted(REQUIRED_EXPLICIT_SYMBOLS):
            with self.subTest(symbol=qualified_name):
                self.assertIn(qualified_name, self.by_name)
                for record in self.by_name[qualified_name]:
                    self.assertEqual(record["mapping_precision"], PRECISION)
                    self.assertIn(
                        record["semantic_review_source"],
                        {
                            "explicit_symbol_registry",
                            "explicit_default_callable_registry",
                        },
                    )

    def test_source_ast_default_callable_fixed_point_is_exhaustive(self) -> None:
        discovered = default_bearing_public_callables_from_source(self.inventory)
        source_defaults = default_parameter_manifest_from_source(self.inventory)
        self.assertEqual(len(discovered), 140)
        self.assertEqual(discovered, set(DEFAULT_BEARING_PUBLIC_CALLABLES))
        self.assertEqual(set(source_defaults), discovered)
        self.assertEqual(
            sum(len(defaults) for defaults in source_defaults.values()),
            456,
        )
        self.assertEqual(source_defaults, SOURCE_DEFAULT_PARAMETER_MANIFEST)
        self.assertEqual(set(DEFAULT_PARAMETER_REVIEWS), discovered)
        self.assertEqual(
            sum(len(reviews) for reviews in DEFAULT_PARAMETER_REVIEWS.values()),
            456,
        )
        self.assertEqual(
            sum(
                len(contracts)
                for contracts in EXPLICIT_DEFAULT_PARAMETER_CONTRACTS.values()
            ),
            456,
        )
        self.assertEqual(
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY["record_count"],
            456,
        )
        self.assertEqual(
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY,
            semantic_registry._read_default_parameter_fact_authority(),
        )
        self.assertEqual(
            SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE,
            semantic_registry._source_default_parameter_use_evidence(),
        )
        self.assertEqual(
            len(
                {
                    semantic_registry._canonical_sha256(contract)
                    for contracts in EXPLICIT_DEFAULT_PARAMETER_CONTRACTS.values()
                    for contract in contracts.values()
                }
            ),
            456,
        )
        self.assertEqual(
            {
                review["effect_category"]
                for reviews in DEFAULT_PARAMETER_REVIEWS.values()
                for review in reviews.values()
            },
            set(DEFAULT_EFFECT_CATEGORIES),
        )
        self.assertEqual(
            discovered,
            set(REVIEWED_DEFAULT_CONTRACTS)
            | set(DEFAULT_CALLABLE_REVIEWS)
            | set(EXPLICIT_DEFAULT_CALLABLES),
        )
        self.assertFalse(
            set(DEFAULT_CALLABLE_REVIEWS) & set(EXPLICIT_DEFAULT_CALLABLES)
        )
        for qualified_name in sorted(discovered):
            with self.subTest(symbol=qualified_name):
                for record in self.by_name[qualified_name]:
                    self.assertNotEqual(
                        record["semantic_review_source"],
                        "reviewed_module_registry",
                    )
                    self.assertEqual(
                        record["default_parameter_reviews"],
                        DEFAULT_PARAMETER_REVIEWS[qualified_name],
                    )
                    if qualified_name in DEFAULT_CALLABLE_REVIEWS:
                        self.assertEqual(
                            record["semantic_review_source"],
                            "explicit_default_callable_registry",
                        )
                        self.assertEqual(
                            record["semantic_review_reason"],
                            DEFAULT_CALLABLE_REVIEWS[qualified_name],
                        )
                        self.assertEqual(
                            record["default_branch_review"],
                            DEFAULT_CALLABLE_REVIEWS[qualified_name],
                        )

    def test_p03_named_default_parameter_semantics_are_exact(self) -> None:
        expected = {
            (
                "payne_zero_atmosphere.install_runtime_data.install_initializer_assets",
                "generated_manifest_path",
            ): (
                "DEFAULT_GENERATED_ASSET_MANIFEST",
                "data/source identity",
                ("choose and hash", "provenance"),
            ),
            (
                "payne_zero_atmosphere.install_runtime_data.install_initializer_assets",
                "include_direct_xh",
            ): ("True", "data/source identity", ("direct-[X/H]", "omits")),
            (
                "payne_zero_atmosphere.install_runtime_data.install_initializer_assets",
                "replace",
            ): ("False", "algorithmic", ("rejects", "atomic replacement")),
            (
                "payne_zero_atmosphere.warm_start.format_warm_start_deck",
                "metallicity",
            ): ("0.0", "physical", ("bulk metal abundance", "deck composition")),
            (
                "payne_zero_atmosphere.warm_start.format_warm_start_deck",
                "alpha_enhancement",
            ): ("0.0", "physical", ("alpha-element", "deck composition")),
            (
                "payne_zero_atmosphere.warm_start.format_warm_start_deck",
                "absolute_abundance_offsets",
            ): ("None", "physical", ("per-element", "overrides")),
            (
                "payne_zero_atmosphere.warm_start.format_warm_start_deck",
                "title",
            ): ("None", "diagnostic", ("canonical model title", "verbatim")),
            (
                "payne_zero_atmosphere.continuum_opacity."
                "compute_molecular_hydrogen_population",
                "hydrogen_departure_coefficient",
            ): ("None", "physical", ("unity LTE", "quadratically")),
            (
                "payne_zero_atmosphere.continuum_opacity."
                "compute_molecular_hydrogen_population",
                "tables",
            ): ("None", "parity/injection", ("canonical", "alternate-data")),
            (
                "payne_zero_atmosphere.hydrogen_line_profile."
                "compute_hydrogen_molecule_population",
                "hydrogen_departure_coefficient",
            ): ("None", "physical", ("unity LTE", "quadratically")),
            (
                "payne_zero_atmosphere.hydrogen_line_profile."
                "compute_hydrogen_molecule_population",
                "tables",
            ): ("None", "parity/injection", ("canonical", "alternate-data")),
            (
                "payne_zero_synthesis.molecular_lines.load_catalog",
                "cache_dir",
            ): ("None", "cache", ("package runtime", "explicit cache")),
            (
                "payne_zero_synthesis.molecular_lines.load_catalog",
                "rebuild",
            ): ("False", "cache", ("missing/stale/corrupt", "forces source")),
            (
                "payne_zero_synthesis.pipeline.window_invariants_for",
                "tables_path",
            ): (
                '_SYNTHESIS_TABLE_DIR / "line_profile_tables.npz"',
                "data/source identity",
                ("line-profile", "cache key"),
            ),
            (
                "payne_zero_synthesis.pipeline.window_invariants_for",
                "transfer_tables_path",
            ): (
                '_SYNTHESIS_TABLE_DIR / "transfer_tables.npz"',
                "data/source identity",
                ("radiative-transfer", "cache key"),
            ),
            (
                "payne_zero_synthesis.pipeline.window_invariants_for",
                "continuum_tables_path",
            ): (
                '_SYNTHESIS_TABLE_DIR / "continuum_tables.npz"',
                "data/source identity",
                ("continuum", "cache key"),
            ),
            (
                "payne_zero_synthesis.pipeline.window_invariants_for",
                "metal_chunk",
            ): (
                "None",
                "cache",
                (
                    "SynthesisPipeline.METAL_CHUNK",
                    "PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1",
                ),
            ),
            (
                "payne_zero_synthesis.source_catalog_molecular_compiler."
                "compile_molecular_text",
                "use_energy_level_wavelengths",
            ): ("False", "algorithmic", ("stored catalog", "energy-level-derived")),
            (
                "payne_zero_synthesis.source_catalog_molecular_compiler."
                "compile_molecular_text",
                "include_predicted_lines",
            ): ("False", "algorithmic", ("negative-energy", "retains")),
            (
                "payne_zero_atmosphere.convection."
                "compute_disabled_convection_diagnostics",
                "overshoot_weight",
            ): ("1.0", "diagnostic", ("forwarded", "endpoint diagnostic arrays")),
            (
                "payne_zero_atmosphere.convection."
                "compute_disabled_convection_diagnostics",
                "zero_top_layer_count",
            ): ("36", "diagnostic", ("top layers", "endpoint diagnostics")),
            (
                "payne_zero_atmosphere.synthesis_bridge."
                "save_product_structured_atmosphere",
                "molecular_lines",
            ): ("True", "physical", ("molecular chemistry", "atom-only")),
            (
                "payne_zero_atmosphere.synthesis_bridge."
                "save_product_structured_atmosphere",
                "source_catalog_root",
            ): ("None", "data/source identity", ("inferred", "explicit catalog")),
            (
                "payne_zero_atmosphere.synthesis_bridge."
                "save_product_structured_atmosphere",
                "device",
            ): ('"cpu"', "environment", ("execution backend", "builder")),
            (
                "payne_zero_atmosphere.synthesis_bridge."
                "save_product_structured_atmosphere",
                "dtype",
            ): ('"float64"', "environment", ("tensor precision", "builder")),
        }
        for (qualified_name, parameter_name), (
            source_default,
            category,
            _previous_narrative_markers,
        ) in expected.items():
            with self.subTest(symbol=qualified_name, parameter=parameter_name):
                review = DEFAULT_PARAMETER_REVIEWS[qualified_name][parameter_name]
                self.assertEqual(review["source_default"], source_default)
                self.assertEqual(review["effect_category"], category)
                semantic_text = " ".join(
                    review[field_name]
                    for field_name in (
                        "default_route",
                        "alternate_route",
                        "validation_and_coupling",
                        "consumer",
                        "observable_effect",
                    )
                )
                evidence = {
                    field_name: review[field_name]
                    for field_name in (
                        "branch_predicates",
                        "call_forwards",
                        "parameter_uses",
                    )
                }
                anchor = semantic_registry._anchor_from_source_evidence(evidence)
                fact = next(
                    item
                    for item in (
                        semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY[
                            "records"
                        ]
                    )
                    if item["qualified_name"] == qualified_name
                    and item["parameter_name"] == parameter_name
                )
                self.assertEqual(fact["category_anchor"], anchor)
                self.assertIn(f"L{anchor['line']}", semantic_text)
                self.assertEqual(
                    semantic_registry._source_derived_effect_role(
                        qualified_name,
                        parameter_name,
                        review,
                    ),
                    fact["effect_role"],
                )

    def test_fifth_audit_high_risk_routes_are_concrete(self) -> None:
        expected_source_facts = {
            (
                "payne_zero_atmosphere.microturbulence.standard_microturbulence",
                "requested_maximum_velocity",
            ): ("requested_maximum_velocity == -99.0e5", "abs"),
            (
                "payne_zero_atmosphere.line_catalog.decode_selected_line_words",
                "detect_swapped_layout",
            ): ("detect_swapped_layout", "swap_pairs=True"),
            (
                "payne_zero_synthesis.molecular_lines.molecular_chunk_lines",
                "default",
            ): (
                "PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES",
                "_env_positive_int",
            ),
            (
                "payne_zero_synthesis.device.resolve_runtime",
                "requested_device",
            ): ("requested_device is not None", "torch.device"),
            (
                "payne_zero_synthesis.device.resolve_runtime",
                "requested_dtype",
            ): ('runtime_device.type == "mps"', "torch.float64"),
            (
                "payne_zero_synthesis.molecular_equilibrium."
                "solve_molecular_equilibrium",
                "return_diagnostics",
            ): ("return_diagnostics", "iterations_completed"),
            (
                "payne_zero_atmosphere.temperature_correction."
                "apply_temperature_correction",
                "mixing_length",
            ): ("float(mixing_length) > 0.0", "velocity_coefficient"),
            (
                "payne_zero_atmosphere.warm_start.deterministic_initializer_labels",
                "jitter_scale",
            ): ("jitter_scale must be finite", "retries are enabled"),
            (
                "payne_zero_atmosphere.hydrogen_line_profile."
                "molecular_hydrogen_equilibrium_constant",
                "tables",
            ): (
                "load_hydrogen_line_profile_tables",
                "h2_partition_function",
            ),
            (
                "payne_zero_synthesis.api.synthesize_from_labels",
                "r_grid",
            ): ("_resolved_r_grid", "r_grid=r_grid"),
            (
                "payne_zero_synthesis.molecular_equilibrium."
                "solve_molecular_equilibrium",
                "chain_length",
            ): ("max(1, int(chain_length))", "depth_index % chain_len"),
            (
                "payne_zero_synthesis.pipeline.SynthesisPipeline.run",
                "spectral_operator",
            ): (
                "spectral_operator is not None",
                "_apply_spectral_operator_in_wavelength_density",
            ),
        }
        for (
            qualified_name,
            parameter_name,
        ), source_fragments in expected_source_facts.items():
            with self.subTest(symbol=qualified_name, parameter=parameter_name):
                review = DEFAULT_PARAMETER_REVIEWS[qualified_name][parameter_name]
                self.assertTrue(review["parameter_uses"])
                self.assertEqual(len(review["source_sha256"]), 64)
                source_fact_text = json.dumps(
                    {
                        "uses": review["parameter_uses"],
                        "bindings": review["flow_bindings"],
                        "predicates": review["branch_predicates"],
                        "calls": review["call_forwards"],
                        "targets": review["forwarded_to"],
                    },
                    sort_keys=True,
                ).replace('\\"', '"')
                rendered_text = " ".join(
                    review[field_name]
                    for field_name in semantic_registry.DEFAULT_PARAMETER_SEMANTIC_FIELDS
                    if isinstance(review[field_name], str)
                )
                for fragment in source_fragments:
                    self.assertIn(fragment, source_fact_text)
                    self.assertIn(fragment, rendered_text)

        dependency_injections = {
            (
                "payne_zero_atmosphere.line_profile_math.fast_exponential_lookup",
                "tables",
            ): "build_fast_exponential_tables",
            (
                "payne_zero_atmosphere.line_profile_math.evaluate_voigt_profile",
                "basis",
            ): "build_voigt_profile_basis",
            (
                "payne_zero_atmosphere.hydrogen_line_profile."
                "molecular_hydrogen_equilibrium_constant",
                "tables",
            ): "load_hydrogen_line_profile_tables",
        }
        for (
            qualified_name,
            parameter_name,
        ), constructor in dependency_injections.items():
            review = DEFAULT_PARAMETER_REVIEWS[qualified_name][parameter_name]
            self.assertEqual(
                review["effect_category"],
                "dependency-injection-with-no-branch",
            )
            source_fact_text = json.dumps(
                {
                    "bindings": review["flow_bindings"],
                    "predicates": review["branch_predicates"],
                    "uses": review["parameter_uses"],
                },
                sort_keys=True,
            )
            self.assertIn(constructor, source_fact_text)
            self.assertIn(constructor, review["default_route"])
            if constructor in {
                "build_fast_exponential_tables",
                "build_voigt_profile_basis",
            }:
                self.assertIn("deterministically", review["default_route"])
                self.assertIn("truthy injected", review["alternate_route"])
                self.assertIn("is not called", review["alternate_route"])
            else:
                self.assertIn("selecting", review["default_route"])

        rendered_text = json.dumps(
            EXPLICIT_DEFAULT_PARAMETER_CONTRACTS,
            sort_keys=True,
        )
        self.assertNotIn("<fall-through>", rendered_text)
        self.assertNotIn(
            "runtime state chooses between two <fall-through>",
            rendered_text,
        )

    def test_forwarded_high_risk_semantics_include_target_operations(self) -> None:
        required_contract_facts = {
            (
                "payne_zero_synthesis.molecular_lines.molecular_chunk_lines",
                "default",
            ): (
                "absent or empty",
                "invalid integer text",
                "max(1, value)",
                "500,000",
            ),
            (
                "payne_zero_synthesis.api.synthesize_from_labels",
                "r_grid",
            ): (
                "20,000.0",
                "exactly equal",
                "non-finite or non-positive",
            ),
            (
                "payne_zero_synthesis.api.synthesize_from_labels",
                "resolution",
            ): (
                "compatibility alias",
                "20,000.0",
                "exactly equal",
                "non-finite or non-positive",
            ),
            (
                "payne_zero_atmosphere.cli.solve_structured_atmosphere",
                "initializer_jitter_scale",
            ): (
                "finite, non-negative",
                "zero is allowed only",
                "jitter_scale * widths * direction",
            ),
            (
                "payne_zero_atmosphere.runner.finalize_transfer_state",
                "mixing_length",
            ): (
                "setup.convection.mixing_length",
                "mixing_length > 0",
                "optical-thickness",
            ),
            (
                "payne_zero_synthesis.continuum.compute_sampled_continuum",
                "frequency_invariants",
            ): (
                "FrequencyInvariants layout mismatch",
                "FrequencyInvariants grid size mismatch",
                "frequency_invariants.natural_log_frequency",
            ),
            (
                "payne_zero_synthesis.pipeline.SynthesisPipeline.run",
                "spectral_operator",
            ): (
                "spectral_operator.convolve_fluxes",
                "spectral_operator.output_wavelength_nm",
                "output_jacobian",
            ),
            (
                "payne_zero_synthesis.radiative_transfer.solve_spectrum",
                "assert_no_saturated_core",
            ): (
                "assert_no_saturated_core = True",
                "raise NotImplementedError",
                "_saturated_core_flux",
                "surface_eddington_flux_per_frequency",
            ),
        }
        for (
            qualified_name,
            parameter_name,
        ), fragments in required_contract_facts.items():
            review = DEFAULT_PARAMETER_REVIEWS[qualified_name][parameter_name]
            contract_text = " ".join(
                review[field_name]
                for field_name in (
                    "default_route",
                    "alternate_route",
                    "validation_and_coupling",
                    "observable_effect",
                )
            )
            for fragment in fragments:
                self.assertIn(fragment, contract_text)
            self.assertTrue(review["forwarded_to"])
        all_targets = [
            target
            for reviews in DEFAULT_PARAMETER_REVIEWS.values()
            for review in reviews.values()
            for target in review["forwarded_to"]
        ]
        self.assertEqual(len(all_targets), 8)
        for target in all_targets:
            facts = target["operation_facts"]
            self.assertEqual(len(facts["source_sha256"]), 64)
            self.assertTrue(facts["operations"])
            self.assertEqual(
                facts,
                semantic_registry._forward_target_parameter_operations(
                    {
                        field_name: target[field_name]
                        for field_name in (
                            "source_path",
                            "callable",
                            "parameter",
                        )
                    }
                ),
            )
        exact_terminal_signatures = {
            (
                "payne_zero_synthesis.molecular_equilibrium."
                "solve_molecular_equilibrium",
                "return_diagnostics",
            ): ("return_value", [("tuple", 5), ("tuple", 4)], []),
            (
                "payne_zero_atmosphere.line_catalog.decode_selected_line_words",
                "detect_swapped_layout",
            ): (
                "return_value",
                [("SelectedLineCatalog", 1), ("SelectedLineCatalog", 1)],
                [],
            ),
            (
                "payne_zero_atmosphere.microturbulence.standard_microturbulence",
                "requested_maximum_velocity",
            ): ("return_value", [("np.ndarray", 1)], []),
            (
                "payne_zero_synthesis.device.resolve_runtime",
                "requested_device",
            ): ("return_value", [("tuple", 2)], []),
            (
                "payne_zero_synthesis.device.resolve_runtime",
                "requested_dtype",
            ): ("return_value", [("tuple", 2)], []),
        }
        for (
            qualified_name,
            parameter_name,
        ), (
            effect_kind,
            return_schema,
            mutated_targets,
        ) in exact_terminal_signatures.items():
            terminal = DEFAULT_PARAMETER_REVIEWS[qualified_name][parameter_name][
                "operation_signature"
            ]["terminal_contract"]
            self.assertEqual(terminal["effect_kind"], effect_kind)
            self.assertEqual(
                [
                    (item["value_type"], item["arity"])
                    for item in terminal["return_contracts"]
                ],
                return_schema,
            )
            self.assertEqual(terminal["mutated_targets"], mutated_targets)

        atmosphere_signature = DEFAULT_PARAMETER_REVIEWS[
            "payne_zero_atmosphere.equation_of_state.iterate_electron_density"
        ]["max_iterations"]["operation_signature"]
        population_signature = DEFAULT_PARAMETER_REVIEWS[
            "payne_zero_synthesis.equation_of_state.solve_population_state"
        ]["max_iter"]["operation_signature"]
        molecular_signature = DEFAULT_PARAMETER_REVIEWS[
            "payne_zero_synthesis.molecular_equilibrium.solve_molecular_equilibrium"
        ]["max_iter"]["operation_signature"]
        atmosphere_terminal = atmosphere_signature["terminal_contract"]
        self.assertEqual(
            atmosphere_terminal["effect_kind"],
            "in_place_mutation",
        )
        self.assertEqual(
            {
                (record["value_type"], record["arity"])
                for record in atmosphere_terminal["return_contracts"]
            },
            {("NoneType", 0)},
        )
        self.assertEqual(
            atmosphere_terminal["mutated_targets"],
            [
                "state.charge_square_density",
                "state.electron_density",
                "state.ion_stage_populations_by_packed_slot",
                "state.mass_density",
                "state.total_nuclei_number_density",
            ],
        )
        population_terminal = population_signature["terminal_contract"]
        self.assertEqual(population_terminal["effect_kind"], "return_value")
        self.assertEqual(
            [
                (record["value_type"], record["arity"])
                for record in population_terminal["return_contracts"]
            ],
            [("PopulationState", 1)],
        )
        self.assertEqual(population_terminal["mutated_targets"], [])
        molecular_terminal = molecular_signature["terminal_contract"]
        self.assertEqual(molecular_terminal["effect_kind"], "return_value")
        self.assertEqual(
            [
                (record["value_type"], record["arity"])
                for record in molecular_terminal["return_contracts"]
            ],
            [("tuple", 5), ("tuple", 4)],
        )
        self.assertEqual(molecular_terminal["mutated_targets"], [])
        self.assertEqual(
            len(
                {
                    atmosphere_signature["location_free_family_sha256"],
                    population_signature["location_free_family_sha256"],
                    molecular_signature["location_free_family_sha256"],
                }
            ),
            3,
        )

    def test_source_path_none_selects_no_io_lte_route(self) -> None:
        review = DEFAULT_PARAMETER_REVIEWS[
            "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__"
        ]["source_path"]
        self.assertEqual(review["source_default"], "None")
        self.assertEqual(
            review["parameter_flow_attributes"],
            ["self._source_from_ref"],
        )
        self.assertIn(
            {
                "line": 1144,
                "target": "self._source_from_ref",
                "value": "source_path is not None",
                "value_ast": (
                    "Compare(left=Name(id='source_path', ctx=Load()), "
                    "ops=[IsNot()], comparators=[Constant(value=None)])"
                ),
                "relation": "direct",
                "snippet_sha256": (
                    "27aefa4dd52eec6b1f2615c0d8439b862bd388bf2b20a724b5d32523775a9222"
                ),
            },
            review["flow_bindings"],
        )
        predicate = next(
            item for item in review["branch_predicates"] if item["line"] == 1145
        )
        self.assertEqual(predicate["snippet"], "self._source_from_ref")
        load = next(
            item for item in review["call_forwards"] if item["callee"] == "np.load"
        )
        self.assertEqual(len(load["control_guards"]), 1)
        self.assertEqual(load["control_guards"][0]["line"], 1145)
        self.assertEqual(load["control_guards"][0]["polarity"], "true")
        self.assertEqual(
            load["control_guards"][0]["snippet"],
            "self._source_from_ref",
        )
        self.assertEqual(load["control_guards"][0]["relation"], "coupled")
        self.assertIn("skips `np.load`", review["default_route"])
        self.assertIn("LTE Planck", review["default_route"])
        self.assertIn("loads the NPZ", review["alternate_route"])

    def test_source_fact_authority_has_no_stock_semantic_prose(self) -> None:
        fresh_snapshot = semantic_registry._fresh_process_default_parameter_snapshot()
        authority = fresh_snapshot["authority"]
        fresh_evidence = fresh_snapshot["evidence"]
        self.assertEqual(authority["schema_version"], 2)
        self.assertEqual(authority["record_count"], 456)
        self.assertEqual(
            {field_name for record in authority["records"] for field_name in record},
            {
                "category_anchor",
                "effect_role",
                "forwarded_to",
                "parameter_name",
                "qualified_name",
            },
        )
        evidence_fields = {
            "binding": (
                "flow_bindings",
                {
                    "line",
                    "relation",
                    "snippet_sha256",
                    "target",
                    "value_ast",
                },
            ),
            "predicate": (
                "branch_predicates",
                {
                    "line",
                    "relation",
                    "snippet_sha256",
                },
            ),
            "call": (
                "call_forwards",
                {
                    "argument",
                    "callee",
                    "line",
                    "relation",
                    "snippet_sha256",
                },
            ),
            "use": (
                "parameter_uses",
                {
                    "column",
                    "line",
                    "snippet_sha256",
                },
            ),
        }
        for fact in authority["records"]:
            anchor = fact["category_anchor"]
            evidence_kind = anchor["evidence_kind"]
            source_field, projected_fields = evidence_fields[evidence_kind]
            expected_anchor = {
                field_name: value
                for field_name, value in anchor.items()
                if field_name != "node_kind"
            }
            matching = []
            for source_fact in fresh_evidence[fact["qualified_name"]][
                fact["parameter_name"]
            ][source_field]:
                projection = {
                    "evidence_kind": evidence_kind,
                    **{
                        field_name: source_fact[field_name]
                        for field_name in projected_fields
                    },
                }
                if projection == expected_anchor:
                    matching.append(source_fact)
            self.assertGreaterEqual(len(matching), 1, fact)
            if evidence_kind == "predicate":
                self.assertTrue(
                    any(
                        anchor["node_kind"] == source_fact["kind"]
                        for source_fact in matching
                    )
                )
        rendered = [
            contract
            for contracts in EXPLICIT_DEFAULT_PARAMETER_CONTRACTS.values()
            for contract in contracts.values()
        ]
        self.assertEqual(len(rendered), 456)
        narrative_fields = (
            "branch_behavior",
            "default_route",
            "alternate_route",
            "validation_and_coupling",
            "consumer",
            "observable_effect",
        )
        for field_name in narrative_fields:
            counts = Counter(contract[field_name] for contract in rendered)
            self.assertEqual(max(counts.values()), 1, field_name)
        all_narrative = [
            contract[field_name]
            for contract in rendered
            for field_name in narrative_fields
        ]
        rejected_stock_fields = (
            "forwarded or expression consumer",
            (
                "No separate predicate validates this parameter in the defining "
                "callable; the exact typed consumer/use evidence is authoritative."
            ),
            (
                "An explicit alternate replaces None in those exact callee "
                "arguments and changes the consumer input."
            ),
        )
        for stock_field in rejected_stock_fields:
            self.assertNotIn(stock_field, all_narrative)

        rejected_effect_conclusions = (
            "that argument-level change is the exact observable source route",
            "are the observable route difference",
        )
        effects = [contract["observable_effect"] for contract in rendered]
        for conclusion in rejected_effect_conclusions:
            self.assertFalse(
                any(conclusion in effect for effect in effects),
                conclusion,
            )

        def normalized_frame(
            text: str,
            qualified_name: str,
            parameter_name: str,
            consumer: str,
            contract: dict,
        ) -> str:
            normalized = text.lower()
            surface_tokens = {
                qualified_name,
                qualified_name.rsplit(".", maxsplit=1)[-1],
                qualified_name.rsplit(".", maxsplit=1)[-1].replace("_", " "),
                parameter_name,
                parameter_name.replace("_", " "),
                consumer,
            }
            for token in sorted(surface_tokens, key=len, reverse=True):
                normalized = re.sub(
                    rf"\b{re.escape(token.lower())}\b",
                    "<surface>",
                    normalized,
                )
            normalized = re.sub(
                r"\b(?:l|line)\s*\d+(?::c\d+)?\b",
                "<line>",
                normalized,
            )
            return " ".join(normalized.split())

        # Prove the detector removes cosmetic identity interpolation.
        self.assertEqual(
            normalized_frame(
                "pkg.one alpha consumer-one L12 `call(alpha)`",
                "pkg.one",
                "alpha",
                "consumer-one",
                {
                    "parameter_uses": [],
                    "branch_predicates": [],
                    "call_forwards": [],
                    "flow_bindings": [],
                },
            ),
            normalized_frame(
                "pkg.two beta consumer-two L99 `call(beta)`",
                "pkg.two",
                "beta",
                "consumer-two",
                {
                    "parameter_uses": [],
                    "branch_predicates": [],
                    "call_forwards": [],
                    "flow_bindings": [],
                },
            ),
        )
        frame_members: dict[str, list[tuple[str, str, tuple]]] = defaultdict(list)

        def independent_operation_family(contract: dict) -> tuple:
            signature = contract["operation_signature"]
            terminal = signature["terminal_contract"]
            return (
                signature["location_free_family_sha256"],
                signature["call_count"],
                tuple(
                    (
                        call["callee"],
                        tuple(call["positional_arguments"]),
                        tuple(
                            (item["name"], item["value"])
                            for item in call["keyword_arguments"]
                        ),
                        tuple(call["affected_arguments"]),
                    )
                    for call in signature["calls"]
                ),
                signature["binding_count"],
                tuple(
                    (item["target"], item["value"], item["value_ast"])
                    for item in signature["bindings"]
                ),
                signature["assignment_count"],
                tuple(
                    (tuple(item["targets"]), item["value"])
                    for item in signature["assignments"]
                ),
                signature["branch_count"],
                tuple(
                    (
                        item["operator"],
                        item["predicate"],
                        item["default_outcome"],
                    )
                    for item in signature["branches"]
                ),
                terminal["effect_kind"],
                tuple(
                    (
                        item["node_kind"],
                        item["value"],
                        item["arity"],
                        item["value_type"],
                    )
                    for item in terminal["return_contracts"]
                ),
                tuple(terminal["mutated_targets"]),
                tuple(
                    (
                        target["callable"],
                        target["parameter"],
                        target["operation_count"],
                        target["terminal_contract"]["effect_kind"],
                        tuple(
                            (
                                item["value"],
                                item["arity"],
                                item["value_type"],
                            )
                            for item in target["terminal_contract"]["return_contracts"]
                        ),
                        tuple(target["terminal_contract"]["mutated_targets"]),
                    )
                    for target in signature["downstream_targets"]
                ),
            )

        for (
            qualified_name,
            contracts,
        ) in EXPLICIT_DEFAULT_PARAMETER_CONTRACTS.items():
            for parameter_name, contract in contracts.items():
                source_contract = DEFAULT_PARAMETER_REVIEWS[qualified_name][
                    parameter_name
                ]
                frame = normalized_frame(
                    contract["observable_effect"],
                    qualified_name,
                    parameter_name,
                    contract["consumer"],
                    source_contract,
                )
                frame_members[frame].append(
                    (
                        qualified_name,
                        parameter_name,
                        independent_operation_family(source_contract),
                    )
                )
        recurrent_frames = {
            frame: members
            for frame, members in frame_members.items()
            if len(members) > 1
        }
        self.assertEqual(len(frame_members), 455)
        self.assertEqual(len(recurrent_frames), 1)
        self.assertEqual(
            sorted(len(members) for members in recurrent_frames.values()),
            [2],
        )
        for frame, members in frame_members.items():
            if len(members) == 1:
                continue
            families = {member[2] for member in members}
            self.assertEqual(
                len(families),
                1,
                (frame, [(qname, parameter) for qname, parameter, _ in members]),
            )
            evidence_families = {member[2][1:] for member in members}
            self.assertEqual(
                len(evidence_families),
                1,
                (frame, [(qname, parameter) for qname, parameter, _ in members]),
            )

        # Recompute every strict signature with a second AST implementation.
        # This code deliberately uses only the clean-process snapshot and fresh
        # pinned source text; it does not call the builder's signature, terminal,
        # evaluator, or normalization helpers.
        module_cache: dict[str, tuple[str, ast.Module]] = {}
        function_cache: dict[
            tuple[str, int | None, str | None],
            tuple[str, ast.FunctionDef | ast.AsyncFunctionDef],
        ] = {}
        terminal_cache: dict[
            tuple[str, int | None, str | None],
            dict,
        ] = {}
        terminal_active: set[tuple[str, int | None, str | None]] = set()
        unknown = object()

        def normalized(node: ast.AST | None) -> str:
            if node is None:
                return "None"
            return " ".join(ast.unparse(node).split())

        def module_source(source_path: str) -> tuple[str, ast.Module]:
            if source_path not in module_cache:
                source_text = (PINNED_SOURCE_ROOT / source_path).read_text(
                    encoding="utf-8"
                )
                module_cache[source_path] = (
                    source_text,
                    ast.parse(source_text),
                )
            return module_cache[source_path]

        def function_source(
            source_path: str,
            *,
            definition_line: int | None = None,
            callable_name: str | None = None,
        ) -> tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]:
            key = (source_path, definition_line, callable_name)
            if key not in function_cache:
                source_text, tree = module_source(source_path)
                matches = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                    and (definition_line is None or node.lineno == definition_line)
                    and (callable_name is None or node.name == callable_name)
                ]
                self.assertEqual(len(matches), 1, key)
                function_cache[key] = (source_text, matches[0])
            return function_cache[key]

        def symbol(node: ast.AST) -> str | None:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                prefix = symbol(node.value)
                return f"{prefix}.{node.attr}" if prefix else node.attr
            return None

        def loaded_symbols(node: ast.AST) -> set[str]:
            return {
                name
                for child in ast.walk(node)
                if isinstance(child, (ast.Name, ast.Attribute))
                and isinstance(child.ctx, ast.Load)
                if (name := symbol(child)) is not None
            }

        def assignment_parts(
            node: ast.AST,
        ) -> tuple[list[ast.AST], ast.AST | None]:
            if isinstance(node, ast.Assign):
                return list(node.targets), node.value
            if isinstance(node, ast.AnnAssign):
                return [node.target], node.value
            if isinstance(node, ast.AugAssign):
                return [node.target], node.value
            if isinstance(node, ast.NamedExpr):
                return [node.target], node.value
            return [], None

        def target_root(node: ast.AST) -> str | None:
            while isinstance(node, (ast.Attribute, ast.Subscript)):
                node = node.value
            return node.id if isinstance(node, ast.Name) else None

        def owned_body_nodes(
            function: ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> list[ast.AST]:
            nested = [
                node
                for statement in function.body
                for node in ast.walk(statement)
                if isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.Lambda,
                    ),
                )
                and node is not function
            ]
            excluded = {
                child for nested_scope in nested for child in ast.walk(nested_scope)
            }
            return [
                child
                for statement in function.body
                for child in ast.walk(statement)
                if child not in excluded
            ]

        def infer_return(
            value: ast.AST | None,
            assignments: dict[str, list[ast.AST]],
            seen: frozenset[str] = frozenset(),
        ) -> tuple[str, int, list[str], int | None]:
            if value is None or (
                isinstance(value, ast.Constant) and value.value is None
            ):
                return "NoneType", 0, [], None
            if isinstance(value, ast.Tuple):
                element_types = [
                    infer_return(item, assignments)[0] for item in value.elts
                ]
                return "tuple", len(value.elts), element_types, len(value.elts)
            if isinstance(value, ast.List):
                return (
                    "list",
                    1,
                    [infer_return(item, assignments)[0] for item in value.elts],
                    len(value.elts),
                )
            if isinstance(value, ast.Set):
                return (
                    "set",
                    1,
                    [infer_return(item, assignments)[0] for item in value.elts],
                    len(value.elts),
                )
            if isinstance(value, ast.Dict):
                return "dict", 1, [], len(value.keys)
            if isinstance(value, ast.Constant):
                return type(value.value).__name__, 1, [], None
            if isinstance(value, ast.Call):
                callee = symbol(value.func) or normalized(value.func)
                return callee.rsplit(".", maxsplit=1)[-1], 1, [], None
            if isinstance(value, ast.Name) and value.id not in seen:
                inferred = {
                    infer_return(
                        candidate,
                        assignments,
                        seen | {value.id},
                    )[0]
                    for candidate in assignments.get(value.id, [])
                }
                if len(inferred) == 1:
                    return next(iter(inferred)), 1, [], None
                return f"symbol:{value.id}", 1, [], None
            if isinstance(value, ast.IfExp):
                body_type = infer_return(value.body, assignments)[0]
                else_type = infer_return(value.orelse, assignments)[0]
                return (
                    f"conditional:{body_type}|{else_type}",
                    1,
                    [],
                    None,
                )
            return type(value).__name__, 1, [], None

        def guarantees_exit(statement: ast.stmt) -> bool:
            if isinstance(statement, (ast.Return, ast.Raise)):
                return True
            if isinstance(statement, ast.If):
                return (
                    bool(statement.body)
                    and bool(statement.orelse)
                    and guarantees_exit(statement.body[-1])
                    and guarantees_exit(statement.orelse[-1])
                )
            if isinstance(statement, (ast.With, ast.AsyncWith)):
                return bool(statement.body) and guarantees_exit(statement.body[-1])
            if isinstance(statement, ast.Try):
                body_exits = bool(statement.body) and guarantees_exit(
                    statement.body[-1]
                )
                handlers_exit = bool(statement.handlers) and all(
                    handler.body and guarantees_exit(handler.body[-1])
                    for handler in statement.handlers
                )
                final_exits = bool(statement.finalbody) and guarantees_exit(
                    statement.finalbody[-1]
                )
                else_exits = not statement.orelse or guarantees_exit(
                    statement.orelse[-1]
                )
                return final_exits or (body_exits and handlers_exit and else_exits)
            return False

        def independent_terminal(
            source_path: str,
            *,
            definition_line: int | None = None,
            callable_name: str | None = None,
        ) -> dict:
            key = (source_path, definition_line, callable_name)
            if key in terminal_cache:
                return terminal_cache[key]
            if key in terminal_active:
                return {"mutated_targets": []}
            terminal_active.add(key)
            source_text, function = function_source(
                source_path,
                definition_line=definition_line,
                callable_name=callable_name,
            )
            body_nodes = owned_body_nodes(function)
            parent_by_node = {
                child: parent
                for parent in ast.walk(function)
                for child in ast.iter_child_nodes(parent)
            }
            parameter_names = {
                argument.arg
                for argument in (
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                )
            }
            assignments: dict[str, list[ast.AST]] = {}
            assignment_records: list[dict] = []
            mutated_targets: set[str] = set()
            for node in body_nodes:
                targets, value = assignment_parts(node)
                if not targets or value is None:
                    continue
                assignment_records.append(
                    {
                        "operator": type(node).__name__,
                        "targets": [normalized(target) for target in targets],
                        "value": normalized(value),
                    }
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(value)
                    if (
                        isinstance(target, (ast.Attribute, ast.Subscript))
                        and target_root(target) in parameter_names
                    ):
                        mutated_targets.add(
                            normalized(target).split("[", maxsplit=1)[0]
                        )

            return_records: list[dict] = []
            annotation = (
                normalized(function.returns) if function.returns is not None else None
            )
            for node in body_nodes:
                if not isinstance(node, ast.Return):
                    continue
                value_type, arity, element_types, cardinality = infer_return(
                    node.value,
                    assignments,
                )
                if annotation not in {None, "None"} and value_type not in {
                    "NoneType",
                    "tuple",
                    "list",
                    "set",
                    "dict",
                }:
                    value_type = annotation
                guards: list[dict[str, str]] = []
                child: ast.AST = node
                parent = parent_by_node.get(child)
                while parent is not None and parent is not function:
                    if isinstance(parent, (ast.If, ast.While)):
                        polarity = (
                            "true"
                            if child in parent.body
                            else "false"
                            if child in parent.orelse
                            else "nested"
                        )
                        guards.append(
                            {
                                "operator": type(parent).__name__,
                                "predicate": normalized(parent.test),
                                "polarity": polarity,
                            }
                        )
                    elif isinstance(parent, ast.IfExp):
                        guards.append(
                            {
                                "operator": "IfExp",
                                "predicate": normalized(parent.test),
                                "polarity": (
                                    "true" if child is parent.body else "false"
                                ),
                            }
                        )
                    child = parent
                    parent = parent_by_node.get(child)
                return_records.append(
                    {
                        "node_kind": "Return",
                        "value": normalized(node.value),
                        "arity": arity,
                        "value_type": value_type,
                        "element_types": element_types,
                        "container_cardinality": cardinality,
                        "guards": list(reversed(guards)),
                    }
                )
            if not function.body or not guarantees_exit(function.body[-1]):
                return_records.append(
                    {
                        "node_kind": "ImplicitReturn",
                        "value": "None",
                        "arity": 0,
                        "value_type": "NoneType",
                        "element_types": [],
                        "container_cardinality": None,
                        "guards": [],
                    }
                )

            non_none_returns = [
                record
                for record in return_records
                if record["value_type"] != "NoneType"
            ]
            if annotation == "None" or not non_none_returns:
                for node in body_nodes:
                    if not isinstance(node, ast.Call):
                        continue
                    callee_symbol = symbol(node.func)
                    if callee_symbol is None:
                        continue
                    callee_name = callee_symbol.rsplit(".", maxsplit=1)[-1]
                    if callee_name == function.name:
                        continue
                    try:
                        _, callee = function_source(
                            source_path,
                            callable_name=callee_name,
                        )
                    except AssertionError:
                        continue
                    callee_parameters = [
                        argument.arg
                        for argument in (
                            *callee.args.posonlyargs,
                            *callee.args.args,
                            *callee.args.kwonlyargs,
                        )
                    ]
                    actual_by_formal = {
                        formal: normalized(actual)
                        for formal, actual in zip(
                            callee_parameters,
                            node.args,
                        )
                    }
                    actual_by_formal.update(
                        {
                            keyword.arg: normalized(keyword.value)
                            for keyword in node.keywords
                            if keyword.arg is not None
                        }
                    )
                    callee_terminal = independent_terminal(
                        source_path,
                        definition_line=callee.lineno,
                    )
                    for callee_target in callee_terminal["mutated_targets"]:
                        formal_root = callee_target.split(".", maxsplit=1)[0].split(
                            "[", maxsplit=1
                        )[0]
                        actual = actual_by_formal.get(formal_root)
                        if actual is None:
                            continue
                        propagated = actual + callee_target[len(formal_root) :]
                        try:
                            root = target_root(ast.parse(propagated, mode="eval").body)
                        except (SyntaxError, ValueError):
                            continue
                        if root in parameter_names:
                            mutated_targets.add(propagated.split("[", maxsplit=1)[0])

            if non_none_returns and mutated_targets:
                effect_kind = "return_and_in_place_mutation"
            elif non_none_returns:
                effect_kind = "return_value"
            elif mutated_targets:
                effect_kind = "in_place_mutation"
            else:
                effect_kind = "side_effect_none"
            unique_returns: list[dict] = []
            for record in return_records:
                if record not in unique_returns:
                    unique_returns.append(record)
            result = {
                "return_annotation": annotation,
                "return_node_count": len(return_records),
                "return_contracts": unique_returns,
                "return_contract_count": len(unique_returns),
                "effect_kind": effect_kind,
                "mutated_targets": sorted(mutated_targets),
                "mutated_target_count": len(mutated_targets),
                "assignment_count": len(assignment_records),
                "assignments": assignment_records,
            }
            terminal_cache[key] = result
            terminal_active.remove(key)
            return result

        def evaluate_expression(expression: str, values: dict[str, object]) -> object:
            node = ast.parse(" ".join(expression.split()), mode="eval").body

            def evaluate(item: ast.AST) -> object:
                if isinstance(item, ast.Constant):
                    return item.value
                if isinstance(item, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
                    try:
                        return ast.literal_eval(item)
                    except (SyntaxError, ValueError):
                        return unknown
                if isinstance(item, (ast.Name, ast.Attribute)):
                    item_symbol = symbol(item)
                    return values.get(item_symbol or "", unknown)
                if isinstance(item, ast.UnaryOp):
                    operand = evaluate(item.operand)
                    if operand is unknown:
                        return unknown
                    if isinstance(item.op, ast.Not):
                        return not bool(operand)
                    if isinstance(item.op, ast.USub):
                        return -operand  # type: ignore[operator]
                    if isinstance(item.op, ast.UAdd):
                        return +operand  # type: ignore[operator]
                    return unknown
                if isinstance(item, ast.BoolOp):
                    if isinstance(item.op, ast.And):
                        result: object = True
                        for operand_node in item.values:
                            result = evaluate(operand_node)
                            if result is unknown or not bool(result):
                                return result
                        return result
                    result = False
                    for operand_node in item.values:
                        result = evaluate(operand_node)
                        if result is unknown or bool(result):
                            return result
                    return result
                if isinstance(item, ast.Compare):
                    operands = [evaluate(item.left)] + [
                        evaluate(value) for value in item.comparators
                    ]
                    if any(value is unknown for value in operands):
                        return unknown
                    outcomes = []
                    for operation, left, right in zip(
                        item.ops,
                        operands,
                        operands[1:],
                    ):
                        if isinstance(operation, ast.Is):
                            outcomes.append(left is right)
                        elif isinstance(operation, ast.IsNot):
                            outcomes.append(left is not right)
                        elif isinstance(operation, ast.Eq):
                            outcomes.append(left == right)
                        elif isinstance(operation, ast.NotEq):
                            outcomes.append(left != right)
                        elif isinstance(operation, ast.Lt):
                            outcomes.append(left < right)  # type: ignore[operator]
                        elif isinstance(operation, ast.LtE):
                            outcomes.append(left <= right)  # type: ignore[operator]
                        elif isinstance(operation, ast.Gt):
                            outcomes.append(left > right)  # type: ignore[operator]
                        elif isinstance(operation, ast.GtE):
                            outcomes.append(left >= right)  # type: ignore[operator]
                        elif isinstance(operation, ast.In):
                            outcomes.append(left in right)  # type: ignore[operator]
                        elif isinstance(operation, ast.NotIn):
                            outcomes.append(left not in right)  # type: ignore[operator]
                        else:
                            return unknown
                    return all(outcomes)
                if isinstance(item, ast.IfExp):
                    selector = evaluate(item.test)
                    if selector is unknown:
                        return unknown
                    return evaluate(item.body if bool(selector) else item.orelse)
                if isinstance(item, ast.BinOp):
                    left = evaluate(item.left)
                    right = evaluate(item.right)
                    if left is unknown or right is unknown:
                        return unknown
                    try:
                        if isinstance(item.op, ast.Add):
                            return left + right  # type: ignore[operator]
                        if isinstance(item.op, ast.Sub):
                            return left - right  # type: ignore[operator]
                        if isinstance(item.op, ast.Mult):
                            return left * right  # type: ignore[operator]
                        if isinstance(item.op, ast.Div):
                            return left / right  # type: ignore[operator]
                        if isinstance(item.op, ast.FloorDiv):
                            return left // right  # type: ignore[operator]
                        if isinstance(item.op, ast.Mod):
                            return left % right  # type: ignore[operator]
                        if isinstance(item.op, ast.Pow):
                            return left**right  # type: ignore[operator]
                    except (TypeError, ValueError, ZeroDivisionError):
                        return unknown
                if isinstance(item, ast.Call):
                    callee = symbol(item.func)
                    arguments = [evaluate(argument) for argument in item.args]
                    if any(value is unknown for value in arguments):
                        return unknown
                    functions = {
                        "abs": abs,
                        "bool": bool,
                        "float": float,
                        "int": int,
                        "max": max,
                        "min": min,
                        "np.isfinite": lambda value: (
                            isinstance(value, (int, float))
                            and value == value
                            and value not in {float("inf"), float("-inf")}
                        ),
                    }
                    function = functions.get(callee or "")
                    if function is None:
                        return unknown
                    try:
                        return function(*arguments)
                    except (TypeError, ValueError):
                        return unknown
                return unknown

            return evaluate(node)

        def predicate_outcomes(
            parameter_name: str,
            source_default: str,
            evidence: dict,
        ) -> dict[tuple[int, str], bool | None]:
            values: dict[str, object] = {}
            try:
                values[parameter_name] = ast.literal_eval(source_default)
            except (SyntaxError, ValueError):
                values[parameter_name] = unknown
            for binding in evidence["flow_bindings"]:
                value = evaluate_expression(binding["value"], values)
                if value is not unknown:
                    values[binding["target"]] = value
            results = {}
            for predicate in evidence["branch_predicates"]:
                expression = predicate["snippet"]
                if predicate["kind"] == "selection_expression":
                    selector = ast.parse(
                        " ".join(expression.split()),
                        mode="eval",
                    ).body
                    self.assertIsInstance(selector, ast.BoolOp)
                    expression = ast.unparse(selector.values[0])
                value = evaluate_expression(expression, values)
                results[(predicate["line"], predicate["snippet"])] = (
                    None if value is unknown else bool(value)
                )
            return results

        def independent_signature(
            qualified_name: str,
            parameter_name: str,
            evidence: dict,
            review: dict,
        ) -> dict:
            source_text, function = function_source(
                evidence["source_path"],
                definition_line=evidence["definition_line"],
            )
            self.assertEqual(
                hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                evidence["source_sha256"],
            )
            body_nodes = [
                child for statement in function.body for child in ast.walk(statement)
            ]
            flow_names = set(evidence["parameter_flow_names"]) | set(
                evidence["parameter_flow_attributes"]
            )
            call_evidence = {
                (item["line"], item["callee"], item["snippet"])
                for item in evidence["call_forwards"]
            }
            calls = []
            for node in body_nodes:
                if not isinstance(node, ast.Call):
                    continue
                callee = normalized(node.func)
                snippet = ast.get_source_segment(source_text, node)
                if (
                    snippet is None
                    or (node.lineno, callee, snippet) not in call_evidence
                ):
                    continue
                calls.append(
                    {
                        "operator": "Call",
                        "callee": callee,
                        "positional_arguments": [
                            normalized(argument) for argument in node.args
                        ],
                        "keyword_arguments": [
                            {
                                "name": (
                                    keyword.arg
                                    if keyword.arg is not None
                                    else "**mapping"
                                ),
                                "value": normalized(keyword.value),
                            }
                            for keyword in node.keywords
                        ],
                        "affected_arguments": sorted(
                            {
                                item["argument"]
                                for item in evidence["call_forwards"]
                                if item["line"] == node.lineno
                                and item["callee"] == callee
                                and item["snippet"] == snippet
                            }
                        ),
                        "relation": (
                            "direct"
                            if parameter_name in loaded_symbols(node)
                            else "coupled"
                        ),
                    }
                )
            predicate_lines = {item["line"] for item in evidence["branch_predicates"]}
            parent_by_node = {
                child: parent
                for parent in ast.walk(function)
                for child in ast.iter_child_nodes(parent)
            }

            def parameter_guarded(node: ast.AST) -> bool:
                parent = parent_by_node.get(node)
                while parent is not None and parent is not function:
                    if getattr(parent, "lineno", None) in predicate_lines:
                        return True
                    parent = parent_by_node.get(parent)
                return False

            assignments = []
            for node in body_nodes:
                targets, value = assignment_parts(node)
                if not targets or value is None:
                    continue
                if not (loaded_symbols(value) & flow_names or parameter_guarded(node)):
                    continue
                assignments.append(
                    {
                        "operator": type(node).__name__,
                        "targets": [normalized(target) for target in targets],
                        "value": normalized(value),
                    }
                )
            bindings = [
                {
                    "operator": "derived_assignment",
                    "target": item["target"],
                    "value": " ".join(item["value"].split()),
                    "value_ast": item["value_ast"],
                    "relation": item["relation"],
                }
                for item in evidence["flow_bindings"]
            ]
            outcomes = predicate_outcomes(
                parameter_name,
                review["source_default"],
                evidence,
            )
            branches = [
                {
                    "operator": item["kind"],
                    "predicate": normalized(
                        ast.parse(
                            " ".join(item["snippet"].split()),
                            mode="eval",
                        ).body
                    ),
                    "relation": item["relation"],
                    "default_outcome": outcomes[(item["line"], item["snippet"])],
                    "true_head": (
                        "<no explicit else; execution continues>"
                        if item["body_head"] is None
                        else " ".join(item["body_head"].split())
                    ),
                    "false_head": (
                        "<no explicit else; execution continues>"
                        if item["else_head"] is None
                        else " ".join(item["else_head"].split())
                    ),
                }
                for item in evidence["branch_predicates"]
            ]
            downstream = []
            for target in review["forwarded_to"]:
                target_source, target_tree = module_source(target["source_path"])
                operations = []
                for operation in target["operation_facts"]["operations"]:
                    source_matches = [
                        node
                        for node in ast.walk(target_tree)
                        if getattr(node, "lineno", None) == operation["line"]
                        and type(node).__name__ == operation["node_kind"]
                        and ast.get_source_segment(target_source, node)
                        == operation["snippet"]
                    ]
                    self.assertGreaterEqual(
                        len(source_matches),
                        1,
                        (qualified_name, parameter_name, operation),
                    )
                    operations.append(
                        {
                            "operator": operation["node_kind"],
                            "relation": operation["relation"],
                            "value": " ".join(operation["snippet"].split()),
                        }
                    )
                downstream.append(
                    {
                        "source_path": target["source_path"],
                        "callable": target["callable"],
                        "parameter": target["parameter"],
                        "operation_count": len(operations),
                        "operations": operations,
                        "terminal_contract": independent_terminal(
                            target["source_path"],
                            callable_name=target["callable"],
                        ),
                    }
                )
            terminal = independent_terminal(
                evidence["source_path"],
                definition_line=evidence["definition_line"],
            )
            signature = {
                "schema_version": 1,
                "call_count": len(calls),
                "calls": calls,
                "binding_count": len(bindings),
                "bindings": bindings,
                "assignment_count": len(assignments),
                "assignments": assignments,
                "branch_count": len(branches),
                "branches": branches,
                "terminal_contract": terminal,
                "downstream_target_count": len(downstream),
                "downstream_targets": downstream,
                "concrete_quantity": {
                    "source_parameter": parameter_name,
                    "returned_values": [
                        record["value"] for record in terminal["return_contracts"]
                    ],
                    "mutated_targets": terminal["mutated_targets"],
                },
            }
            family = {
                "schema_version": signature["schema_version"],
                "call_count": signature["call_count"],
                "calls": signature["calls"],
                "binding_count": signature["binding_count"],
                "bindings": signature["bindings"],
                "assignment_count": signature["assignment_count"],
                "assignments": signature["assignments"],
                "branch_count": signature["branch_count"],
                "branches": signature["branches"],
                "terminal_contract": signature["terminal_contract"],
                "downstream_target_count": signature["downstream_target_count"],
                "downstream_targets": [
                    {
                        key: value
                        for key, value in target.items()
                        if key != "source_path"
                    }
                    for target in signature["downstream_targets"]
                ],
                "concrete_quantity": {
                    "returned_values": signature["concrete_quantity"][
                        "returned_values"
                    ],
                    "mutated_targets": signature["concrete_quantity"][
                        "mutated_targets"
                    ],
                },
            }
            signature["location_free_family_sha256"] = hashlib.sha256(
                json.dumps(
                    family,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            return signature

        independently_recomputed = 0
        for qualified_name, evidence_by_parameter in fresh_evidence.items():
            for parameter_name, evidence in evidence_by_parameter.items():
                review = fresh_snapshot["reviews"][qualified_name][parameter_name]
                actual_signature = independent_signature(
                    qualified_name,
                    parameter_name,
                    evidence,
                    review,
                )
                self.assertEqual(
                    actual_signature,
                    review["operation_signature"],
                    (qualified_name, parameter_name),
                )
                independently_recomputed += 1
        self.assertEqual(independently_recomputed, 456)

    def test_missing_required_symbol_and_module_drift_fail_closed(self) -> None:
        missing_ledger = deepcopy(self.ledger)
        target = "payne_zero_synthesis.radiative_transfer.integrate_optical_depth"
        for package in missing_ledger["packages"].values():
            package["symbols"] = [
                record
                for record in package["symbols"]
                if record["qualified_name"] != target
            ]
        with self.assertRaises((KeyError, RuntimeError)):
            apply_overrides(missing_ledger)

        drifted_ledger = deepcopy(self.ledger)
        pipeline = drifted_ledger["packages"]["payne_zero_synthesis"]["symbols"]
        pipeline.append(
            {
                "qualified_name": "payne_zero_synthesis.pipeline.new_public_branch",
                "kind": "public_function",
                "line": 9999,
                "mapping_precision": "module_default",
                "primary_location": "chapter-7",
                "supporting_locations": [],
                "responsibility": "synthetic drift",
                "gate": "none",
                "status": "planned",
            }
        )
        with self.assertRaisesRegex(RuntimeError, "reviewed module snapshot changed"):
            apply_overrides(drifted_ledger)

    def test_duplicate_export_and_definition_records_are_consistent(self) -> None:
        semantic_fields = (
            "mapping_precision",
            "primary_location",
            "supporting_locations",
            "semantic_disposition",
            "semantic_review_source",
            "source_spelling",
            "responsibility",
            "gate",
            "status",
            "source_module_sha256",
            "source_public_surface_sha256",
            "source_public_object_sha256",
            "reviewed_default_contract",
            "default_branch_review",
            "default_parameter_reviews",
            "accepted_authority_evidence",
        )
        duplicate_groups = {
            name: records for name, records in self.by_name.items() if len(records) > 1
        }
        self.assertEqual(
            sum(len(records) - 1 for records in duplicate_groups.values()),
            58,
        )
        for qualified_name, records in duplicate_groups.items():
            baseline = {field: records[0].get(field) for field in semantic_fields}
            for record in records[1:]:
                with self.subTest(
                    symbol=qualified_name,
                    kind=record["kind"],
                ):
                    self.assertEqual(
                        {field: record.get(field) for field in semantic_fields},
                        baseline,
                    )

    def test_complete_package_export_alias_registry_is_bound(self) -> None:
        aliases = [
            record
            for record in self.records
            if record["kind"] == "public_export"
            and record["qualified_name"].split(".")[1] == "__init__"
        ]
        self.assertEqual(len(aliases), 198)
        self.assertEqual(
            {record["qualified_name"] for record in aliases},
            set(API_ALIAS_TARGETS) | {"payne_zero_synthesis.__init__.__version__"},
        )
        for record in aliases:
            with self.subTest(alias=record["qualified_name"]):
                self.assertEqual(record["mapping_precision"], PRECISION)
                self.assertEqual(
                    record["semantic_review_source"],
                    "explicit_api_alias_registry",
                )
                self.assertEqual(
                    record["semantic_disposition"],
                    "plumbing-only",
                )
                if record["qualified_name"] in API_ALIAS_TARGETS:
                    target_name = API_ALIAS_TARGETS[record["qualified_name"]]
                    self.assertEqual(record["alias_target"], target_name)
                    target = self.one(target_name)
                    self.assertEqual(
                        record["primary_location"],
                        target["primary_location"],
                    )
                    self.assertEqual(record["status"], target["status"])

    def test_accepted_chapters_one_through_five_are_synchronized(self) -> None:
        expected = _accepted_artifact_join()
        self.assertEqual(len(expected), 97)
        self.assertEqual(
            set(expected),
            {
                qualified_name
                for symbols in ACCEPTED_ARTIFACT_SYMBOLS.values()
                for qualified_name in symbols
            },
        )
        self.assertEqual(
            set(ACCEPTED_CHAPTER_AUTHORITIES),
            {f"chapter-{number}" for number in range(1, 6)},
        )
        self.assertEqual(len(CHAPTER4_ACCEPTED_SYMBOLS), 23)
        self.assertEqual(len(CHAPTER5_ACCEPTED_SYMBOLS), 46)
        self.assertEqual(len(CHAPTER5_ATMOSPHERE_SYMBOLS), 38)
        self.assertEqual(len(CHAPTER5_SYNTHESIS_SYMBOLS), 8)
        self.assertEqual(
            {
                f"payne_zero_synthesis.continuum.{name}"
                for name in CHAPTER5_SYNTHESIS_SYMBOLS
            },
            {
                "payne_zero_synthesis.continuum.ContinuumTables",
                "payne_zero_synthesis.continuum.FrequencyInvariants",
                "payne_zero_synthesis.continuum.build_frequency_invariants",
                "payne_zero_synthesis.continuum.build_edge_sample_frequencies",
                "payne_zero_synthesis.continuum.build_pops",
                "payne_zero_synthesis.continuum.compute_sampled_continuum",
                "payne_zero_synthesis.continuum.continuum",
                "payne_zero_synthesis.continuum.pops_from_population_state",
            },
        )
        for qualified_name, values in expected.items():
            with self.subTest(symbol=qualified_name):
                for record in self.by_name[qualified_name]:
                    self.assertEqual(
                        {
                            "primary_location": record["primary_location"],
                            "semantic_disposition": record["semantic_disposition"],
                            "status": record["status"],
                            "accepted_authority_evidence": record[
                                "accepted_authority_evidence"
                            ],
                        },
                        values,
                    )

    def test_production_rosseland_and_microturbulence_owners_are_split(self) -> None:
        rosseland = self.one("payne_zero_atmosphere.rosseland_mean.rosseland_mean_step")
        self.assertEqual(rosseland["primary_location"], "chapter-12")
        self.assertNotEqual(rosseland["primary_location"], "chapter-5")
        self.assertIn("production", rosseland["responsibility"])

        width = self.one("payne_zero_synthesis.pipeline.compute_doppler_per_ion")
        standard = self.one(
            "payne_zero_atmosphere.microturbulence.standard_microturbulence"
        )
        self.assertEqual(width["primary_location"], "chapter-3")
        self.assertEqual(standard["primary_location"], "chapter-11")
        self.assertIn("distinct", standard["responsibility"])

    def test_atmosphere_and_synthesis_molecular_statuses_cannot_collapse(self) -> None:
        atmosphere_water = self.one(
            "payne_zero_atmosphere.config.AtmosphereInput.water_lines_path"
        )
        atmosphere_h3plus = self.one(
            "payne_zero_atmosphere.config.AtmosphereInput.h3plus_lines_path"
        )
        default_sources = self.one(
            "payne_zero_atmosphere.source_catalogs.source_line_paths"
        )
        synthesis_h2o = self.one(
            "payne_zero_synthesis.source_catalog_molecular_compiler."
            "compile_h2o_partridge"
        )
        selector = self.one(
            "payne_zero_atmosphere.line_selection.generate_selected_lines"
        )
        self.assertIn("standard", atmosphere_water["responsibility"])
        self.assertIn("opt-in", atmosphere_h3plus["responsibility"])
        self.assertIn("not a default key", default_sources["responsibility"])
        self.assertIn("not wired", synthesis_h2o["responsibility"])
        self.assertIn("water", selector["responsibility"])
        self.assertIn("H3+", selector["responsibility"])

    def test_pipeline_optional_and_diagnostic_boundaries_are_explicit(self) -> None:
        constructor = self.one(
            "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__"
        )
        run = self.one("payne_zero_synthesis.pipeline.SynthesisPipeline.run")
        accumulate = self.one("payne_zero_synthesis.line_opacity.accumulate_atomic")
        self.assertEqual(constructor["primary_location"], "chapter-10")
        self.assertIn("source_path", constructor["responsibility"])
        self.assertIn("Planck", constructor["responsibility"])
        self.assertIn("keep_slabs", run["responsibility"])
        self.assertIn("spectral_operator", run["responsibility"])
        self.assertIn("wing_mode='loop'", accumulate["responsibility"])
        self.assertIn("host_accumulator", accumulate["responsibility"])
        for field in (
            "continuum_absorption",
            "continuum_scattering",
            "line_mass_absorption_coefficient",
            "line_source",
            "spectral_operator_seconds",
            "spectral_operator_name",
        ):
            record = self.one(f"payne_zero_synthesis.pipeline.SpectrumResult.{field}")
            self.assertEqual(
                record["semantic_disposition"],
                "diagnostic-only",
            )
            self.assertEqual(record["status"], "boundary")

    def test_runner_unsupported_claims_name_only_implemented_guards(self) -> None:
        turbulence = self.one(
            "payne_zero_atmosphere.run_setup.TurbulenceSettings.enabled"
        )
        iterations = self.one("payne_zero_atmosphere.run_setup.RunSetup.iterations")
        flags = self.one(
            "payne_zero_atmosphere.run_setup.opacity_flags_from_atmosphere"
        )
        runner = self.one("payne_zero_atmosphere.runner.run_atmosphere_model")
        self.assertEqual(
            turbulence["semantic_disposition"],
            "unsupported",
        )
        self.assertIn("iterations < 1", iterations["gate"])
        self.assertIn("flag 13", flags["responsibility"])
        self.assertIn("iterations<1", runner["responsibility"])
        self.assertIn("turbulent pressure", runner["responsibility"])
        self.assertIn("HLINOP", runner["responsibility"])
        self.assertNotIn("NLTE", runner["responsibility"])
        self.assertNotIn("raw molecular", runner["responsibility"])

    def test_device_transfer_initializer_and_saturated_core_branches_are_explicit(
        self,
    ) -> None:
        runtime = self.one("payne_zero_synthesis.device.resolve_runtime")
        transfer = self.one("payne_zero_synthesis.radiative_transfer.solve_spectrum")
        initializer = self.one(
            "payne_zero_synthesis.api.initialize_atmosphere_from_labels"
        )
        closure = self.one(
            "payne_zero_synthesis.api.InitializedAtmosphere.atmosphere_closure_required"
        )
        direct = self.one(
            "payne_zero_atmosphere.direct_abundance."
            "build_direct_abundance_optimizer_surrogate"
        )
        self.assertEqual(runtime["primary_location"], "chapter-2")
        self.assertIn("MPS-float64 rejection", runtime["responsibility"])
        self.assertIn("unavailable", runtime["gate"])
        self.assertIn("saturated", transfer["responsibility"])
        self.assertIn("five-label", initializer["responsibility"])
        self.assertIn("direct-abundance", initializer["responsibility"])
        self.assertIn("closure", closure["responsibility"])
        self.assertIn("no fitting implementation", direct["responsibility"])

    def test_compatibility_diagnostics_cache_and_publication_are_separated(
        self,
    ) -> None:
        expected = {
            (
                "payne_zero_synthesis.atmosphere.LEGACY_ATMOSPHERE_ARRAY_ALIASES"
            ): "compatibility-only",
            ("payne_zero_synthesis.atomic_lines.LineCatalog.from_npz"): "plumbing-only",
            (
                "payne_zero_synthesis.molecular_lines.MolecularLineCatalog.from_npz"
            ): "plumbing-only",
            (
                "payne_zero_synthesis.molecular_lines.MolecularLineCatalog.from_mapping"
            ): "plumbing-only",
            (
                "payne_zero_atmosphere.warm_start.load_atmosphere_initializer"
            ): "composed",
            (
                "payne_zero_synthesis.pipeline.clear_window_invariant_cache"
            ): "plumbing-only",
            (
                "payne_zero_atmosphere.runner.AtmosphereRunResult.diagnostics"
            ): "diagnostic-only",
            (
                "payne_zero_atmosphere.synthesis_bridge."
                "structured_atmosphere_from_debug_npz"
            ): "diagnostic-only",
            (
                "payne_zero_atmosphere.synthesis_bridge."
                "save_product_structured_atmosphere"
            ): "composed",
            (
                "payne_zero_synthesis.atmosphere.load_atmosphere_product_metadata"
            ): "plumbing-only",
            (
                "payne_zero_atmosphere.convection."
                "compute_disabled_convection_diagnostics"
            ): "diagnostic-only",
            (
                "payne_zero_synthesis.source_catalog_molecular_compiler.logger"
            ): "diagnostic-only",
        }
        for qualified_name, disposition in expected.items():
            with self.subTest(symbol=qualified_name):
                self.assertEqual(
                    self.one(qualified_name)["semantic_disposition"],
                    disposition,
                )

    def test_independent_audit_residual_branches_are_explicit(self) -> None:
        rosseland_names = (
            "RosselandOpacityTable",
            "create_rosseland_opacity_table",
            "ingest_rosseland_opacity_table",
            "evaluate_rosseland_opacity",
            "compute_rosseland_continuum_opacity_columns",
        )
        for name in rosseland_names:
            with self.subTest(symbol=name):
                record = self.one(f"payne_zero_atmosphere.continuum_opacity.{name}")
                self.assertEqual(record["semantic_disposition"], "diagnostic-only")
                self.assertIn("default-off", record["responsibility"])
                self.assertIn("production mean", record["responsibility"])

        expected_text = {
            (
                "payne_zero_synthesis.continuum.build_frequency_invariants"
            ): "coulomb_table_energy_first=False",
            (
                "payne_zero_synthesis.molecular_equilibrium.solve_molecular_equilibrium"
            ): "return_diagnostics=False",
            (
                "payne_zero_atmosphere.line_catalog.decode_selected_line_words"
            ): "detect_swapped_layout=True",
            ("payne_zero_synthesis.hydrogen_lines.precompute_invariants"): "Lyman",
            (
                "payne_zero_synthesis.hydrogen_lines.accumulate_hydrogen"
            ): "apply_stim=True",
            (
                "payne_zero_atmosphere.equation_of_state.saha_partition_depth"
            ): "unsupported population-mode",
            (
                "payne_zero_atmosphere.equation_of_state.saha_partition_depth_batch"
            ): "unsupported population-mode",
            (
                "payne_zero_atmosphere.line_profile_math.load_line_opacity_tables"
            ): "force_reload=False",
            (
                "payne_zero_atmosphere.temperature_correction."
                "apply_temperature_correction"
            ): "smoothing",
        }
        for qualified_name, phrase in expected_text.items():
            with self.subTest(symbol=qualified_name):
                record = self.one(qualified_name)
                self.assertEqual(
                    record["semantic_review_source"],
                    "explicit_symbol_registry",
                )
                self.assertIn(phrase, record["responsibility"])

    def test_second_audit_branch_sensitive_routines_are_explicit(self) -> None:
        expected_phrases = {
            "payne_zero_synthesis.molecular_lines.accumulate_molecular": (
                "apply_stim=True",
                "False",
                "chunk_lines=None",
                "environment policy",
                "clamped",
            ),
            "payne_zero_synthesis.molecular_lines.molecular_chunk_lines": (
                "PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES",
                "missing/empty/invalid",
                "clamp",
            ),
            (
                "payne_zero_atmosphere.continuum_opacity."
                "compute_continuum_opacity_columns"
            ): (
                "opacity_flags=None",
                "individual flags",
                "flag 19",
                "Rosseland table",
            ),
            (
                "payne_zero_atmosphere.continuum_opacity."
                "compute_light_element_continuum_columns"
            ): ("opacity_flags=None", "per-family"),
            (
                "payne_zero_atmosphere.continuum_opacity."
                "compute_continuum_scattering_columns"
            ): ("flag-family", "molecular-equilibrium"),
            (
                "payne_zero_synthesis.equation_of_state."
                "partition_functions_for_elements"
            ): ("apply_ground_partition=True", "False", "molecular-bridge"),
            "payne_zero_synthesis.equation_of_state.solve_electron_density": (
                "molecules=False",
                "molecules=True",
                "molecules_path",
            ),
            "payne_zero_synthesis.equation_of_state.solve_population_state": (
                "atom-only",
                "molecular",
                "two-solve",
            ),
            (
                "payne_zero_synthesis.equation_of_state."
                "solve_population_state_at_electron_density"
            ): (
                "fixed-public-electron",
                "molecular internal electron",
                "molecules_path",
            ),
            (
                "payne_zero_atmosphere.radiative_transfer."
                "load_radiative_transfer_tables"
            ): ("path=None", "force_reload=False", "warm cache"),
            "payne_zero_synthesis.atomic_lines.parse_catalog": (
                "catalog_path=None",
                "apply_iso_corr=True",
                "sort",
                "wavelength ordering",
            ),
        }
        for qualified_name, phrases in expected_phrases.items():
            with self.subTest(symbol=qualified_name):
                record = self.one(qualified_name)
                self.assertEqual(
                    record["semantic_review_source"],
                    "explicit_symbol_registry",
                )
                for phrase in phrases:
                    self.assertIn(phrase, record["responsibility"])

    def test_third_audit_branch_sensitive_routines_are_explicit(self) -> None:
        expected_phrases = {
            "payne_zero_synthesis.continuum.compute_sampled_continuum": (
                "frequency_invariants=None",
                "grid-validated",
                "sampled extension",
            ),
            (
                "payne_zero_atmosphere.continuum_opacity.compute_hminus_opacity_columns"
            ): (
                "unity LTE departure",
                "depth multiplier",
                "shape validation",
            ),
            "payne_zero_atmosphere.convection.compute_convection": (
                "all-eight-arrays",
                "ideal-gas fallback",
                "zero mixing length",
                "overshoot",
                "top-layer zeroing",
            ),
            (
                "payne_zero_atmosphere.runner."
                "compute_convection_finite_difference_samples"
            ): (
                "atomic-only",
                "molecular perturbations",
                "saved/restored",
                "all four EOS samples",
            ),
            "payne_zero_atmosphere.runner.finalize_transfer_state": (
                "caller-supplied convection",
                "internal finite-difference",
                "radiation pressure",
                "turbulent pressure",
            ),
            "payne_zero_atmosphere.convergence.max_normalized_column_delta": (
                "asymmetric",
                "symmetric",
                "max(before, after, floor)",
            ),
            "payne_zero_atmosphere.prewarm.prewarm": (
                "force=False",
                "force=True",
                "fresh-process-valid",
            ),
            "payne_zero_synthesis.prewarm.prewarm": (
                "force=False",
                "force=True",
                "clears in-process invariants",
            ),
            "payne_zero_synthesis.atomic_lines.load_catalog": (
                "sort and isotope correction",
                "cache key",
                "rebuild=False",
                "True always reparses",
            ),
        }
        for qualified_name, phrases in expected_phrases.items():
            with self.subTest(symbol=qualified_name):
                record = self.one(qualified_name)
                self.assertEqual(
                    record["semantic_review_source"],
                    "explicit_symbol_registry",
                )
                self.assertEqual(
                    record["default_branch_review"],
                    record["responsibility"],
                )
                for phrase in phrases:
                    self.assertIn(phrase, record["responsibility"])

    def test_declared_diagnostics_and_written_debug_paths_are_distinct(self) -> None:
        diagnostics = self.one(
            "payne_zero_atmosphere.config.AtmosphereOutput.diagnostics_path"
        )
        debug = self.one(
            "payne_zero_atmosphere.config.AtmosphereOutput.debug_state_path"
        )
        self.assertEqual(diagnostics["semantic_disposition"], "plumbing-only")
        self.assertIn("never writes", diagnostics["responsibility"])
        self.assertIn("declaration-only", diagnostics["gate"])
        self.assertEqual(debug["semantic_disposition"], "diagnostic-only")
        self.assertIn("actual", debug["responsibility"])
        self.assertIn("write/no-write", debug["gate"])

    def test_branch_default_contracts_are_exact_and_embedded(self) -> None:
        self.assertEqual(len(REVIEWED_DEFAULT_CONTRACTS), 32)
        expected_defaults = {
            "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__": (
                "source_path",
                "None",
            ),
            "payne_zero_synthesis.pipeline.SynthesisPipeline.run": (
                "keep_slabs",
                "False",
            ),
            "payne_zero_synthesis.continuum.build_frequency_invariants": (
                "coulomb_table_energy_first",
                "False",
            ),
            (
                "payne_zero_synthesis.molecular_equilibrium.solve_molecular_equilibrium"
            ): ("return_diagnostics", "False"),
            "payne_zero_atmosphere.line_catalog.decode_selected_line_words": (
                "detect_swapped_layout",
                "True",
            ),
            "payne_zero_synthesis.hydrogen_lines.accumulate_hydrogen": (
                "apply_stim",
                "True",
            ),
            ("payne_zero_atmosphere.line_profile_math.load_line_opacity_tables"): (
                "force_reload",
                "False",
            ),
            "payne_zero_synthesis.molecular_lines.accumulate_molecular": (
                "apply_stim",
                "True",
            ),
            (
                "payne_zero_atmosphere.continuum_opacity."
                "compute_continuum_opacity_columns"
            ): ("opacity_flags", "None"),
            (
                "payne_zero_synthesis.equation_of_state."
                "partition_functions_for_elements"
            ): ("apply_ground_partition", "True"),
            "payne_zero_synthesis.equation_of_state.solve_electron_density": (
                "molecules",
                "False",
            ),
            (
                "payne_zero_atmosphere.radiative_transfer."
                "load_radiative_transfer_tables"
            ): ("force_reload", "False"),
            "payne_zero_synthesis.atomic_lines.parse_catalog": (
                "apply_iso_corr",
                "True",
            ),
            "payne_zero_synthesis.continuum.compute_sampled_continuum": (
                "frequency_invariants",
                "None",
            ),
            (
                "payne_zero_atmosphere.continuum_opacity.compute_hminus_opacity_columns"
            ): ("hminus_departure_coefficient", "None"),
            "payne_zero_atmosphere.convection.compute_convection": (
                "convection_enabled",
                "True",
            ),
            (
                "payne_zero_atmosphere.runner."
                "compute_convection_finite_difference_samples"
            ): ("molecules_enabled", "False"),
            "payne_zero_atmosphere.runner.finalize_transfer_state": (
                "convection_enabled",
                "False",
            ),
            "payne_zero_atmosphere.convergence.max_normalized_column_delta": (
                "symmetric",
                "False",
            ),
            "payne_zero_atmosphere.prewarm.prewarm": ("force", "False"),
            "payne_zero_synthesis.prewarm.prewarm": ("force", "False"),
            "payne_zero_synthesis.atomic_lines.load_catalog": (
                "rebuild",
                "False",
            ),
        }
        for qualified_name, (parameter, default) in expected_defaults.items():
            with self.subTest(symbol=qualified_name):
                contract = REVIEWED_DEFAULT_CONTRACTS[qualified_name]
                self.assertEqual(contract["defaults"][parameter], default)
                for record in self.by_name[qualified_name]:
                    self.assertEqual(record["reviewed_default_contract"], contract)

    def test_flagged_defaults_preserve_exact_source_text_and_ast(self) -> None:
        constructor = EXACT_DEFAULT_SYNTAX[
            "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__"
        ]
        self.assertEqual(
            constructor["tables_path"]["source_segment"],
            '_SYNTHESIS_TABLE_DIR / "line_profile_tables.npz"',
        )
        self.assertEqual(
            constructor["transfer_tables_path"]["source_segment"],
            '_SYNTHESIS_TABLE_DIR / "transfer_tables.npz"',
        )
        self.assertEqual(
            constructor["continuum_tables_path"]["source_segment"],
            '_SYNTHESIS_TABLE_DIR / "continuum_tables.npz"',
        )
        atomic = EXACT_DEFAULT_SYNTAX[
            "payne_zero_synthesis.line_opacity.accumulate_atomic"
        ]["wing_mode"]
        self.assertEqual(atomic["source_segment"], '"batched"')
        self.assertEqual(atomic["ast_dump"], "Constant(value='batched')")
        for qualified_name, parameters in EXACT_DEFAULT_SYNTAX.items():
            for parameter_name, syntax in parameters.items():
                with self.subTest(
                    symbol=qualified_name,
                    parameter=parameter_name,
                ):
                    self.assertEqual(
                        REVIEWED_DEFAULT_CONTRACTS[qualified_name]["defaults"][
                            parameter_name
                        ],
                        syntax["source_segment"],
                    )

    def test_raw_source_sha_and_signatures_fail_before_assignment(self) -> None:
        source_drift = deepcopy(self.inventory)
        self.raw_module(
            source_drift,
            "payne_zero_synthesis.pipeline",
        )["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            RuntimeError,
            "reviewed module source SHA-256 changed",
        ):
            apply_overrides(deepcopy(self.ledger), inventory=source_drift)

        signature_mutations = (
            (
                "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__",
                "source_path",
                "source_archive",
            ),
            (
                "payne_zero_synthesis.pipeline.SynthesisPipeline.run",
                "keep_slabs",
                "retain_slabs",
            ),
        )
        for qualified_name, old_name, new_name in signature_mutations:
            with self.subTest(symbol=qualified_name):
                signature_drift = deepcopy(self.inventory)
                descriptor = self.raw_callable(signature_drift, qualified_name)
                parameter_index = descriptor["parameters"].index(old_name)
                descriptor["parameters"][parameter_index] = new_name
                with self.assertRaisesRegex(
                    RuntimeError,
                    "reviewed public signature/field surface changed",
                ):
                    apply_overrides(
                        deepcopy(self.ledger),
                        inventory=signature_drift,
                    )

    def test_default_alias_and_semantic_registry_mutations_fail_closed(self) -> None:
        original_defaults = semantic_registry.REVIEWED_DEFAULT_CONTRACTS
        try:
            for qualified_name, parameter, replacement in (
                (
                    "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__",
                    "source_path",
                    "'source.npz'",
                ),
                (
                    "payne_zero_synthesis.pipeline.SynthesisPipeline.run",
                    "keep_slabs",
                    "True",
                ),
            ):
                with self.subTest(default=f"{qualified_name}.{parameter}"):
                    mutated = deepcopy(original_defaults)
                    mutated[qualified_name]["defaults"][parameter] = replacement
                    semantic_registry.REVIEWED_DEFAULT_CONTRACTS = mutated
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "reviewed default contract registry changed",
                    ):
                        apply_overrides(
                            deepcopy(self.ledger),
                            inventory=self.inventory,
                        )
        finally:
            semantic_registry.REVIEWED_DEFAULT_CONTRACTS = original_defaults

        original_aliases = semantic_registry.API_ALIAS_TARGETS
        try:
            mutated_aliases = dict(original_aliases)
            alias_name = sorted(mutated_aliases)[0]
            mutated_aliases[alias_name] = sorted(set(mutated_aliases.values()))[1]
            semantic_registry.API_ALIAS_TARGETS = mutated_aliases
            with self.assertRaisesRegex(
                RuntimeError,
                "API alias target registry changed",
            ):
                apply_overrides(deepcopy(self.ledger), inventory=self.inventory)
        finally:
            semantic_registry.API_ALIAS_TARGETS = original_aliases

        original_overrides = semantic_registry.EXPLICIT_OVERRIDES
        target = "payne_zero_synthesis.radiative_transfer.planck_bnu"
        mutations = {
            "semantic_disposition": "composed",
            "primary_location": "chapter-2",
            "gate": "mutated gate",
            "status": "planned",
        }
        try:
            for field_name, replacement in mutations.items():
                with self.subTest(explicit_field=field_name):
                    mutated_overrides = deepcopy(original_overrides)
                    mutated_overrides[target][field_name] = replacement
                    semantic_registry.EXPLICIT_OVERRIDES = mutated_overrides
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "explicit semantic registry changed",
                    ):
                        apply_overrides(
                            deepcopy(self.ledger),
                            inventory=self.inventory,
                        )
        finally:
            semantic_registry.EXPLICIT_OVERRIDES = original_overrides

    def test_default_callable_scope_and_review_mutations_fail_closed(self) -> None:
        original_reviews = semantic_registry.DEFAULT_CALLABLE_REVIEWS
        try:
            for qualified_name in sorted(original_reviews):
                with self.subTest(review=qualified_name):
                    mutated = dict(original_reviews)
                    mutated[qualified_name] = (
                        original_reviews[qualified_name] + " runtime mutation"
                    )
                    semantic_registry.DEFAULT_CALLABLE_REVIEWS = mutated
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "default-callable review registry changed",
                    ):
                        apply_overrides(
                            deepcopy(self.ledger),
                            inventory=self.inventory,
                        )
        finally:
            semantic_registry.DEFAULT_CALLABLE_REVIEWS = original_reviews

        original_scope = semantic_registry.DEFAULT_BEARING_PUBLIC_CALLABLES
        try:
            semantic_registry.DEFAULT_BEARING_PUBLIC_CALLABLES = frozenset(
                set(original_scope) - {"payne_zero_synthesis.atomic_lines.load_catalog"}
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "default-bearing callable registry changed",
            ):
                apply_overrides(deepcopy(self.ledger), inventory=self.inventory)
        finally:
            semantic_registry.DEFAULT_BEARING_PUBLIC_CALLABLES = original_scope

    def test_all_default_parameter_record_mutations_fail_closed(self) -> None:
        original_reviews = semantic_registry.DEFAULT_PARAMETER_REVIEWS
        objects = semantic_registry._inventory_objects(self.inventory)
        categories = sorted(DEFAULT_EFFECT_CATEGORIES)
        fresh_authority = semantic_registry._read_default_parameter_fact_authority()
        fresh_contracts = semantic_registry._contracts_from_fact_authority(
            fresh_authority,
            semantic_registry._PINNED_AST_DEFAULT_PARAMETER_USE_EVIDENCE,
        )

        # Exhaust every exact key against a fresh authority read and a new
        # deterministic render from the private pinned-AST facts.
        reviewed_keys = set()
        for qualified_name, reviews in original_reviews.items():
            for parameter_name, review in reviews.items():
                reviewed_keys.add((qualified_name, parameter_name))
                semantic_contract = {
                    field_name: review[field_name]
                    for field_name in semantic_registry.DEFAULT_PARAMETER_SEMANTIC_FIELDS
                }
                self.assertEqual(
                    semantic_contract,
                    semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS[
                        qualified_name
                    ][parameter_name],
                )
                self.assertEqual(
                    {
                        "source_default": review["source_default"],
                        "default_ast": review["default_ast"],
                    },
                    SOURCE_DEFAULT_PARAMETER_MANIFEST[qualified_name][parameter_name],
                )
                self.assertEqual(
                    semantic_contract,
                    fresh_contracts[qualified_name][parameter_name],
                )
        self.assertEqual(len(reviewed_keys), 456)

        qualified_name, parameter_name = sorted(reviewed_keys)[0]
        review = original_reviews[qualified_name][parameter_name]
        alternative_category = next(
            category for category in categories if category != review["effect_category"]
        )

        def replace_parameter(
            replacement_name: str | None,
            replacement_record: dict | None,
        ) -> dict:
            mutated = dict(original_reviews)
            parameters = dict(original_reviews[qualified_name])
            del parameters[parameter_name]
            if replacement_name is not None and replacement_record is not None:
                parameters[replacement_name] = replacement_record
            mutated[qualified_name] = parameters
            return mutated

        mutations = {
            "deleted": replace_parameter(None, None),
            "misnamed": replace_parameter(
                f"{parameter_name}_renamed",
                review,
            ),
            "reclassified": replace_parameter(
                parameter_name,
                {**review, "effect_category": alternative_category},
            ),
            "source-mutated": replace_parameter(
                parameter_name,
                {
                    **review,
                    "source_default": f"({review['source_default']})",
                },
            ),
        }
        try:
            for mutation_name, mutated in mutations.items():
                with self.subTest(mutation=mutation_name):
                    semantic_registry.DEFAULT_PARAMETER_REVIEWS = mutated
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "default-parameter semantic review registry changed",
                    ):
                        semantic_registry._validate_default_parameter_reviews(objects)
        finally:
            semantic_registry.DEFAULT_PARAMETER_REVIEWS = original_reviews

    def test_every_wrong_effect_role_fails_exclusive_source_classifier(
        self,
    ) -> None:
        roles = (
            "physical_quantity",
            "source_identity",
            "cache_policy",
            "runtime_environment",
            "diagnostic_surface",
            "compatibility_route",
            "parity_or_injection",
            "unsupported_guard",
            "algorithm_control",
            "continuous_equation",
            "dependency_injection",
        )
        fresh_snapshot = semantic_registry._fresh_process_default_parameter_snapshot()
        authority = deepcopy(fresh_snapshot["authority"])
        fresh_evidence = fresh_snapshot["evidence"]
        attempted = 0
        for fact in authority["records"]:
            qualified_name = fact["qualified_name"]
            parameter_name = fact["parameter_name"]
            evidence = fresh_evidence[qualified_name][parameter_name]
            expected_role = fact["effect_role"]
            self.assertEqual(
                semantic_registry._source_derived_effect_role(
                    qualified_name,
                    parameter_name,
                    evidence,
                ),
                expected_role,
            )
            supported_roles = [
                role
                for role in roles
                if semantic_registry._effect_role_supported_by_source(
                    qualified_name,
                    parameter_name,
                    role,
                    evidence,
                )
            ]
            self.assertEqual(supported_roles, [expected_role])
            empty_evidence = {
                **evidence,
                "parameter_uses": [],
                "branch_predicates": [],
                "attribute_branch_predicates": [],
                "call_forwards": [],
                "flow_bindings": [],
            }
            self.assertFalse(
                any(
                    semantic_registry._effect_role_supported_by_source(
                        qualified_name,
                        parameter_name,
                        role,
                        empty_evidence,
                    )
                    for role in roles
                )
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "no exact parameter load",
            ):
                semantic_registry._source_derived_effect_role(
                    qualified_name,
                    parameter_name,
                    empty_evidence,
                )
            for wrong_role in roles:
                if wrong_role == expected_role:
                    continue
                attempted += 1
                fact["effect_role"] = wrong_role
                with self.assertRaisesRegex(
                    RuntimeError,
                    "effect role differs from exclusive source-operation role",
                ):
                    semantic_registry._validate_source_derived_effect_role(
                        qualified_name,
                        parameter_name,
                        fact["effect_role"],
                        evidence,
                    )
                fact["effect_role"] = expected_role
        self.assertEqual(attempted, 4_560)
        named_roles = {
            (
                "payne_zero_synthesis.api.synthesize_from_labels",
                "resolution",
            ): "compatibility_route",
            (
                "payne_zero_atmosphere.line_profile_math.fast_exponential_lookup",
                "tables",
            ): "dependency_injection",
        }
        for (qualified_name, parameter_name), role in named_roles.items():
            evidence = fresh_evidence[qualified_name][parameter_name]
            self.assertEqual(
                semantic_registry._source_derived_effect_role(
                    qualified_name,
                    parameter_name,
                    evidence,
                ),
                role,
            )

    def test_default_parameter_manifest_and_template_substitution_fail_closed(
        self,
    ) -> None:
        objects = semantic_registry._inventory_objects(self.inventory)
        original_manifest = semantic_registry.SOURCE_DEFAULT_PARAMETER_MANIFEST
        original_reviews = semantic_registry.DEFAULT_PARAMETER_REVIEWS
        original_contracts = semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS
        original_review_digest = (
            semantic_registry.DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256
        )
        original_contract_digest = (
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256
        )
        qualified_name = (
            "payne_zero_atmosphere.install_runtime_data.install_initializer_assets"
        )
        parameter_name = "generated_manifest_path"
        try:
            mutated_manifest = deepcopy(original_manifest)
            mutated_manifest[qualified_name][parameter_name]["source_default"] = "None"
            semantic_registry.SOURCE_DEFAULT_PARAMETER_MANIFEST = mutated_manifest
            with self.assertRaisesRegex(
                RuntimeError,
                "source-default parameter manifest changed",
            ):
                semantic_registry._validate_default_parameter_reviews(objects)
        finally:
            semantic_registry.SOURCE_DEFAULT_PARAMETER_MANIFEST = original_manifest

        try:
            generic_reviews = deepcopy(original_reviews)
            generic_contracts = deepcopy(original_contracts)
            vague = {
                **generic_contracts[qualified_name][parameter_name],
                "branch_behavior": "category-level default behavior",
                "default_route": "Use the source default for this category.",
                "alternate_route": "Use an alternate value for this category.",
                "validation_and_coupling": "Apply ordinary validation.",
                "consumer": "the routine result",
                "observable_effect": "Changes the routine result.",
            }
            generic_contracts[qualified_name][parameter_name] = vague
            generic_reviews[qualified_name][parameter_name].update(vague)
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS = generic_contracts
            semantic_registry.DEFAULT_PARAMETER_REVIEWS = generic_reviews
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256 = (
                semantic_registry._canonical_sha256(generic_contracts)
            )
            semantic_registry.DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256 = (
                semantic_registry._canonical_sha256(generic_reviews)
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "semantic contracts differ from fresh source-fact rendering",
            ):
                semantic_registry._validate_default_parameter_reviews(objects)
        finally:
            semantic_registry.DEFAULT_PARAMETER_REVIEWS = original_reviews
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS = original_contracts
            semantic_registry.DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256 = (
                original_review_digest
            )
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256 = (
                original_contract_digest
            )

    def test_rebound_default_contract_attacks_fail_before_proof_sealing(
        self,
    ) -> None:
        objects = semantic_registry._inventory_objects(self.inventory)
        original_reviews = semantic_registry.DEFAULT_PARAMETER_REVIEWS
        original_contracts = semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS
        original_authority = semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY
        original_authority_path = semantic_registry.DEFAULT_PARAMETER_SEMANTICS_PATH
        original_role_categories = semantic_registry.EFFECT_ROLE_TO_CATEGORY
        original_role_descriptions = semantic_registry.EFFECT_ROLE_DESCRIPTION
        original_evidence = semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE
        original_digests = {
            "authority": (
                semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY_SHA256
            ),
            "reviews": semantic_registry.DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256,
            "contracts": (
                semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256
            ),
            "evidence": (
                semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE_SHA256
            ),
        }

        def bind_every_mutable_surface(
            reviews: dict,
            contracts: dict,
            authority: dict,
        ) -> None:
            semantic_registry.DEFAULT_PARAMETER_REVIEWS = reviews
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS = contracts
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY = authority
            semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE = deepcopy(
                original_evidence
            )
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_PATH = Path(
                "/tmp/hostile-default-parameter-semantics.json"
            )
            semantic_registry.EFFECT_ROLE_TO_CATEGORY = {
                role: "diagnostic" for role in semantic_registry.EFFECT_ROLE_TO_CATEGORY
            }
            semantic_registry.EFFECT_ROLE_DESCRIPTION = {
                role: "category-template effect"
                for role in semantic_registry.EFFECT_ROLE_DESCRIPTION
            }
            semantic_registry.DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256 = (
                semantic_registry._canonical_sha256(reviews)
            )
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256 = (
                semantic_registry._canonical_sha256(contracts)
            )
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY_SHA256 = (
                semantic_registry._canonical_sha256(authority)
            )
            semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE_SHA256 = (
                semantic_registry._canonical_sha256(
                    semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE
                )
            )

        try:
            # The exact fifth-audit generic substitution mutates both rendered
            # semantic surfaces and every published registry/digest.  Fresh
            # source-fact rendering must still reject it.
            generic_qname = "payne_zero_atmosphere.atmosphere_io.parse_atmosphere_deck"
            generic_parameter = "source"
            generic_reviews = deepcopy(original_reviews)
            generic_contracts = deepcopy(original_contracts)
            generic_authority = deepcopy(original_authority)
            vague_fields = {
                "branch_behavior": "Use the diagnostic category branch.",
                "default_route": "Use the source default for this category.",
                "alternate_route": "Use an alternate value for this category.",
                "validation_and_coupling": "Apply ordinary category validation.",
                "consumer": "the routine result",
                "observable_effect": "Changes the routine result.",
            }
            generic_contracts[generic_qname][generic_parameter].update(vague_fields)
            generic_reviews[generic_qname][generic_parameter].update(vague_fields)
            bind_every_mutable_surface(
                generic_reviews,
                generic_contracts,
                generic_authority,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "semantic contracts differ from fresh source-fact rendering",
            ):
                semantic_registry._validate_default_parameter_reviews(objects)

            # Reclassifying the chunk bound is rejected twice: directly by its
            # source-operation rule, and by the fresh on-disk authority after
            # all in-memory expectation surfaces have been rebound.
            qualified_name = (
                "payne_zero_synthesis.molecular_lines.molecular_chunk_lines"
            )
            parameter_name = "default"
            reclassified_reviews = deepcopy(original_reviews)
            reclassified_contracts = deepcopy(original_contracts)
            reclassified_authority = deepcopy(original_authority)
            reclassified_contracts[qualified_name][parameter_name][
                "effect_category"
            ] = "diagnostic"
            reclassified_reviews[qualified_name][parameter_name]["effect_category"] = (
                "diagnostic"
            )
            reclassified_fact = next(
                fact
                for fact in reclassified_authority["records"]
                if fact["qualified_name"] == qualified_name
                and fact["parameter_name"] == parameter_name
            )
            reclassified_fact["effect_role"] = "diagnostic_surface"
            with self.assertRaisesRegex(
                RuntimeError,
                "effect role differs from exclusive source-operation role",
            ):
                semantic_registry._contracts_from_fact_authority(
                    reclassified_authority,
                    semantic_registry._PINNED_AST_DEFAULT_PARAMETER_USE_EVIDENCE,
                )
            bind_every_mutable_surface(
                reclassified_reviews,
                reclassified_contracts,
                reclassified_authority,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "loaded semantic authority differs from fresh source-anchored facts",
            ):
                semantic_registry._validate_default_parameter_reviews(objects)

            # The independent private AST remains the evidence oracle even if
            # the public evidence object and its digest are changed together.
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY = original_authority
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_PATH = original_authority_path
            semantic_registry.EFFECT_ROLE_TO_CATEGORY = original_role_categories
            semantic_registry.EFFECT_ROLE_DESCRIPTION = original_role_descriptions
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY_SHA256 = (
                original_digests["authority"]
            )
            evidence_reviews = deepcopy(original_reviews)
            evidence_registry = deepcopy(original_evidence)
            evidence_registry[qualified_name][parameter_name]["parameter_uses"][0][
                "line"
            ] += 1
            evidence_reviews[qualified_name][parameter_name]["parameter_uses"][0][
                "line"
            ] += 1
            semantic_registry.DEFAULT_PARAMETER_REVIEWS = evidence_reviews
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS = original_contracts
            semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE = evidence_registry
            semantic_registry.DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256 = (
                semantic_registry._canonical_sha256(evidence_reviews)
            )
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256 = (
                original_digests["contracts"]
            )
            semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE_SHA256 = (
                semantic_registry._canonical_sha256(evidence_registry)
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "source-default use evidence differs from the pinned AST",
            ):
                semantic_registry._validate_default_parameter_reviews(objects)
        finally:
            semantic_registry.DEFAULT_PARAMETER_REVIEWS = original_reviews
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS = original_contracts
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY = original_authority
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_PATH = original_authority_path
            semantic_registry.EFFECT_ROLE_TO_CATEGORY = original_role_categories
            semantic_registry.EFFECT_ROLE_DESCRIPTION = original_role_descriptions
            semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE = original_evidence
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY_SHA256 = (
                original_digests["authority"]
            )
            semantic_registry.DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256 = (
                original_digests["reviews"]
            )
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256 = (
                original_digests["contracts"]
            )
            semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE_SHA256 = (
                original_digests["evidence"]
            )

    def test_fresh_process_rejects_private_ast_and_downstream_forgery(
        self,
    ) -> None:
        objects = semantic_registry._inventory_objects(self.inventory)
        original = {
            "private_evidence": (
                semantic_registry._PINNED_AST_DEFAULT_PARAMETER_USE_EVIDENCE
            ),
            "public_evidence": semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE,
            "contracts": semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS,
            "reviews": semantic_registry.DEFAULT_PARAMETER_REVIEWS,
            "authority": semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY,
            "manifest": semantic_registry.SOURCE_DEFAULT_PARAMETER_MANIFEST,
            "root": semantic_registry.PINNED_PAYNE_ZERO_SOURCE_ROOT,
            "authority_path": semantic_registry.DEFAULT_PARAMETER_SEMANTICS_PATH,
            "role_categories": semantic_registry.EFFECT_ROLE_TO_CATEGORY,
            "role_descriptions": semantic_registry.EFFECT_ROLE_DESCRIPTION,
            "target_parser": semantic_registry._forward_target_parameter_operations,
            "evidence_digest": (
                semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE_SHA256
            ),
            "contract_digest": (
                semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256
            ),
            "review_digest": (
                semantic_registry.DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256
            ),
            "authority_digest": (
                semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY_SHA256
            ),
        }
        forged_evidence = deepcopy(original["private_evidence"])
        qname = "payne_zero_atmosphere.atmosphere_io.parse_atmosphere_deck"
        parameter = "source"
        fabricated_binding = {
            "line": 9999,
            "target": "fabricated_temperature",
            "value": "source * 0",
            "value_ast": (
                "BinOp(left=Name(id='source', ctx=Load()), op=Mult(), "
                "right=Constant(value=0))"
            ),
            "relation": "direct",
            "snippet_sha256": semantic_registry.hashlib.sha256(
                b"source * 0"
            ).hexdigest(),
        }
        forged_evidence[qname][parameter]["flow_bindings"].append(fabricated_binding)
        forged_evidence[qname][parameter]["branch_predicates"].append(
            {
                "line": 9999,
                "kind": "if",
                "relation": "direct",
                "snippet": "source == 'fabricated'",
                "snippet_sha256": semantic_registry.hashlib.sha256(
                    b"source == 'fabricated'"
                ).hexdigest(),
                "body_head": "fabricated_temperature = source * 0",
                "else_head": None,
            }
        )
        attribute_qname = "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__"
        attribute_parameter = "source_path"
        forged_attribute = {
            "line": 9999,
            "kind": "if",
            "relation": "coupled",
            "snippet": "self._fabricated_source",
            "snippet_sha256": semantic_registry.hashlib.sha256(
                b"self._fabricated_source"
            ).hexdigest(),
            "body_head": "erase_reference_state()",
            "else_head": None,
        }
        forged_evidence[attribute_qname][attribute_parameter][
            "branch_predicates"
        ].append(forged_attribute)
        forged_evidence[attribute_qname][attribute_parameter][
            "attribute_branch_predicates"
        ].append(forged_attribute)

        forged_contracts = deepcopy(original["contracts"])
        forged_reviews = deepcopy(original["reviews"])
        forged_reviews[qname][parameter]["flow_bindings"].append(fabricated_binding)
        forged_reviews[qname][parameter]["branch_predicates"].append(
            forged_evidence[qname][parameter]["branch_predicates"][-1]
        )
        forged_contracts[qname][parameter]["observable_effect"] = (
            "Fabricated L9999 temperature state."
        )
        forged_reviews[qname][parameter]["observable_effect"] = (
            "Fabricated L9999 temperature state."
        )
        forged_reviews[attribute_qname][attribute_parameter][
            "branch_predicates"
        ].append(forged_attribute)
        forged_reviews[attribute_qname][attribute_parameter][
            "attribute_branch_predicates"
        ].append(forged_attribute)

        downstream_qname = "payne_zero_synthesis.continuum.compute_sampled_continuum"
        downstream_parameter = "frequency_invariants"
        fabricated_operation = {
            "guard_path": [],
            "line": 9999,
            "node_kind": "Call",
            "relation": "direct",
            "snippet": "erase_frequency_grid(frequency_invariants)",
            "snippet_sha256": semantic_registry.hashlib.sha256(
                b"erase_frequency_grid(frequency_invariants)"
            ).hexdigest(),
        }
        forged_contracts[downstream_qname][downstream_parameter]["forwarded_to"][0][
            "operation_facts"
        ]["operations"].append(fabricated_operation)
        forged_reviews[downstream_qname][downstream_parameter]["forwarded_to"][0][
            "operation_facts"
        ]["operations"].append(fabricated_operation)

        def forged_target_parser(target: dict) -> dict:
            facts = original["target_parser"](target)
            facts["operations"].append(fabricated_operation)
            return facts

        try:
            semantic_registry._PINNED_AST_DEFAULT_PARAMETER_USE_EVIDENCE = (
                forged_evidence
            )
            semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE = forged_evidence
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS = forged_contracts
            semantic_registry.DEFAULT_PARAMETER_REVIEWS = forged_reviews
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY = deepcopy(
                original["authority"]
            )
            semantic_registry.PINNED_PAYNE_ZERO_SOURCE_ROOT = Path(
                "/tmp/forged-payne-zero"
            )
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_PATH = Path(
                "/tmp/forged-authority.json"
            )
            semantic_registry.EFFECT_ROLE_TO_CATEGORY = {
                role: "diagnostic" for role in original["role_categories"]
            }
            semantic_registry.EFFECT_ROLE_DESCRIPTION = {
                role: "forged effect" for role in original["role_descriptions"]
            }
            semantic_registry._forward_target_parameter_operations = (
                forged_target_parser
            )
            semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE_SHA256 = (
                semantic_registry._canonical_sha256(forged_evidence)
            )
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256 = (
                semantic_registry._canonical_sha256(forged_contracts)
            )
            semantic_registry.DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256 = (
                semantic_registry._canonical_sha256(forged_reviews)
            )
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY_SHA256 = (
                semantic_registry._canonical_sha256(
                    semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY
                )
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "differs from the pinned AST",
            ):
                semantic_registry._validate_default_parameter_reviews(objects)
        finally:
            semantic_registry._PINNED_AST_DEFAULT_PARAMETER_USE_EVIDENCE = original[
                "private_evidence"
            ]
            semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE = original[
                "public_evidence"
            ]
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS = original[
                "contracts"
            ]
            semantic_registry.DEFAULT_PARAMETER_REVIEWS = original["reviews"]
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY = original[
                "authority"
            ]
            semantic_registry.SOURCE_DEFAULT_PARAMETER_MANIFEST = original["manifest"]
            semantic_registry.PINNED_PAYNE_ZERO_SOURCE_ROOT = original["root"]
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_PATH = original[
                "authority_path"
            ]
            semantic_registry.EFFECT_ROLE_TO_CATEGORY = original["role_categories"]
            semantic_registry.EFFECT_ROLE_DESCRIPTION = original["role_descriptions"]
            semantic_registry._forward_target_parameter_operations = original[
                "target_parser"
            ]
            semantic_registry.SOURCE_DEFAULT_PARAMETER_USE_EVIDENCE_SHA256 = original[
                "evidence_digest"
            ]
            semantic_registry.EXPLICIT_DEFAULT_PARAMETER_CONTRACTS_SHA256 = original[
                "contract_digest"
            ]
            semantic_registry.DEFAULT_PARAMETER_REVIEW_REGISTRY_SHA256 = original[
                "review_digest"
            ]
            semantic_registry.DEFAULT_PARAMETER_SEMANTICS_AUTHORITY_SHA256 = original[
                "authority_digest"
            ]

    def test_exact_default_syntax_mutations_fail_closed(self) -> None:
        original_syntax = semantic_registry.EXACT_DEFAULT_SYNTAX
        mutations = (
            (
                "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__",
                "tables_path",
                "source_segment",
                "_SYNTHESIS_TABLE_DIR / 'line_profile_tables.npz'",
            ),
            (
                "payne_zero_synthesis.line_opacity.accumulate_atomic",
                "wing_mode",
                "source_segment",
                "'batched'",
            ),
            (
                "payne_zero_synthesis.line_opacity.accumulate_atomic",
                "wing_mode",
                "ast_dump",
                "Name(id='batched', ctx=Load())",
            ),
        )
        try:
            for qualified_name, parameter, field_name, replacement in mutations:
                with self.subTest(
                    symbol=qualified_name,
                    parameter=parameter,
                    field=field_name,
                ):
                    mutated = deepcopy(original_syntax)
                    mutated[qualified_name][parameter][field_name] = replacement
                    semantic_registry.EXACT_DEFAULT_SYNTAX = mutated
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "exact default source/AST syntax registry changed",
                    ):
                        apply_overrides(
                            deepcopy(self.ledger),
                            inventory=self.inventory,
                        )
        finally:
            semantic_registry.EXACT_DEFAULT_SYNTAX = original_syntax

    def test_every_module_policy_semantic_field_is_digest_bound(self) -> None:
        self.assertEqual(len(REVIEWED_MODULE_POLICIES), 55)
        original_policies = semantic_registry.REVIEWED_MODULE_POLICIES
        target = "payne_zero_synthesis.molecular_lines"
        mutations = {
            "semantic_disposition": "composed",
            "semantic_review_reason": "runtime mutation",
            "primary_location": "chapter-99",
            "supporting_locations": ["appendix-z"],
            "responsibility": "runtime mutation",
            "gate": "runtime mutation",
            "status": "boundary",
        }
        try:
            for field_name, replacement in mutations.items():
                with self.subTest(field=field_name):
                    mutated = deepcopy(original_policies)
                    mutated[target][field_name] = replacement
                    semantic_registry.REVIEWED_MODULE_POLICIES = mutated
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "complete reviewed module semantic policy changed",
                    ):
                        apply_overrides(
                            deepcopy(self.ledger),
                            inventory=self.inventory,
                        )
        finally:
            semantic_registry.REVIEWED_MODULE_POLICIES = original_policies

    def test_accepted_artifact_authority_and_ledger_drift_fail_closed(self) -> None:
        original_authorities = semantic_registry.ACCEPTED_CHAPTER_AUTHORITIES
        try:
            mutated = deepcopy(original_authorities)
            mutated["chapter-3"][0]["sha256"] = "0" * 64
            semantic_registry.ACCEPTED_CHAPTER_AUTHORITIES = mutated
            with self.assertRaisesRegex(
                RuntimeError,
                "accepted Chapter 1-5 artifact manifest changed",
            ):
                _accepted_artifact_join()
        finally:
            semantic_registry.ACCEPTED_CHAPTER_AUTHORITIES = original_authorities

        target_authority = (
            REPOSITORY_ROOT / ACCEPTED_CHAPTER_AUTHORITIES["chapter-3"][0]["path"]
        )
        original_read_bytes = Path.read_bytes

        def drift_one_authority(path: Path) -> bytes:
            content = original_read_bytes(path)
            if path == target_authority:
                return content + b"\n"
            return content

        with patch.object(Path, "read_bytes", new=drift_one_authority):
            with self.assertRaisesRegex(
                RuntimeError,
                "accepted authority hash changed",
            ):
                _accepted_artifact_join()

        target_symbol = "payne_zero_synthesis.radiative_transfer.planck_bnu"
        for field_name, replacement in (
            ("primary_location", "chapter-99"),
            ("status", "planned"),
            ("accepted_authority_evidence", {}),
        ):
            with self.subTest(ledger_field=field_name):
                drifted_ledger = deepcopy(self.ledger)
                drifted_records = all_records(drifted_ledger)
                target_record = next(
                    record
                    for record in drifted_records
                    if record["qualified_name"] == target_symbol
                )
                target_record[field_name] = replacement
                with self.assertRaisesRegex(
                    RuntimeError,
                    "accepted artifact ledger tuple changed",
                ):
                    validate_complete_semantic_proof(drifted_ledger)

    def test_exact_authority_evidence_rejects_location_and_leaf_adversaries(
        self,
    ) -> None:
        original_evidence = semantic_registry.ACCEPTED_ARTIFACT_EVIDENCE
        original_manifest_digest = semantic_registry.ACCEPTED_ARTIFACT_MANIFEST_SHA256

        def rebound_manifest_digest() -> str:
            return semantic_registry._canonical_sha256(
                {
                    "authorities": semantic_registry.ACCEPTED_CHAPTER_AUTHORITIES,
                    "symbols": semantic_registry.ACCEPTED_ARTIFACT_SYMBOLS,
                    "evidence": semantic_registry.ACCEPTED_ARTIFACT_EVIDENCE,
                    "chapter4_scope": {
                        "path": "design/chapter04_ownership_data_audit.md",
                        "rows": semantic_registry.CHAPTER4_AUTHORITY_ROWS,
                        "qualified_names": sorted(
                            semantic_registry.CHAPTER4_ACCEPTED_SYMBOLS
                        ),
                    },
                    "chapter5_scope": {
                        "path": "design/chapter05_symbol_disposition.md",
                        "expected_count": 46,
                        "qualified_names": sorted(
                            semantic_registry.CHAPTER5_ACCEPTED_SYMBOLS
                        ),
                    },
                }
            )

        target = "payne_zero_synthesis.radiative_transfer.planck_bnu"
        mutations = (
            ("path", "content/Chapter01.ipynb"),
            ("line", 1),
            ("marker", "not an exact source marker"),
            ("marker", "continuum_tables.npz"),
        )
        try:
            for field_name, replacement in mutations:
                with self.subTest(evidence_field=field_name, value=replacement):
                    mutated = deepcopy(original_evidence)
                    mutated["chapter-1"][target][field_name] = replacement
                    semantic_registry.ACCEPTED_ARTIFACT_EVIDENCE = mutated
                    semantic_registry.ACCEPTED_ARTIFACT_MANIFEST_SHA256 = (
                        rebound_manifest_digest()
                    )
                    with self.assertRaises(RuntimeError):
                        _accepted_artifact_join()
        finally:
            semantic_registry.ACCEPTED_ARTIFACT_EVIDENCE = original_evidence
            semantic_registry.ACCEPTED_ARTIFACT_MANIFEST_SHA256 = (
                original_manifest_digest
            )

        # The same leaf exists in both packages.  A Chapter 4 atmosphere export
        # may not be rebound to the synthesis implementation.
        original_aliases = semantic_registry.API_ALIAS_TARGETS
        alias = "payne_zero_atmosphere.__init__.solve_molecular_equilibrium"
        try:
            mutated_aliases = dict(original_aliases)
            mutated_aliases[alias] = (
                "payne_zero_synthesis.molecular_equilibrium.solve_molecular_equilibrium"
            )
            semantic_registry.API_ALIAS_TARGETS = mutated_aliases
            with self.assertRaisesRegex(
                RuntimeError,
                "Chapter 4 exhaustive authority extraction changed",
            ):
                _accepted_artifact_join()
        finally:
            semantic_registry.API_ALIAS_TARGETS = original_aliases

        joined = _accepted_artifact_join()
        for qualified_name, fields in joined.items():
            with self.subTest(no_archive_filename=qualified_name):
                evidence = fields["accepted_authority_evidence"]
                self.assertNotIn(".npz", evidence["marker"])
                self.assertEqual(evidence["qualified_name"], qualified_name)
                self.assertRegex(evidence["line_sha256"], r"^[0-9a-f]{64}$")

    def test_complete_semantic_proof_rejects_one_field_record_changes(self) -> None:
        bundle = semantic_registry._semantic_proof_bundle(self.ledger)
        self.assertEqual(len(bundle["records"]), REVIEWED_RECORD_COUNT)
        target_symbol = "payne_zero_synthesis.molecular_lines.CHUNK_LINES"
        mutations = {
            "primary_location": "chapter-99",
            "supporting_locations": ["appendix-z"],
            "semantic_disposition": "composed",
            "responsibility": "runtime mutation",
            "gate": "runtime mutation",
            "status": "boundary",
            "source_public_object_sha256": "0" * 64,
        }
        for field_name, replacement in mutations.items():
            with self.subTest(field=field_name):
                drifted_ledger = deepcopy(self.ledger)
                target_record = next(
                    record
                    for record in all_records(drifted_ledger)
                    if record["qualified_name"] == target_symbol
                )
                target_record[field_name] = replacement
                with self.assertRaisesRegex(
                    RuntimeError,
                    "complete 1,501-record semantic proof changed",
                ):
                    validate_complete_semantic_proof(drifted_ledger)

        default_drift = deepcopy(self.ledger)
        default_record = next(
            record
            for record in all_records(default_drift)
            if record["qualified_name"]
            == "payne_zero_synthesis.atomic_lines.load_catalog"
        )
        default_record["default_branch_review"] = "runtime mutation"
        with self.assertRaisesRegex(
            RuntimeError,
            "complete 1,501-record semantic proof changed",
        ):
            validate_complete_semantic_proof(default_drift)

        parameter_drift = deepcopy(self.ledger)
        parameter_record = next(
            record
            for record in all_records(parameter_drift)
            if record["qualified_name"]
            == "payne_zero_synthesis.molecular_lines.load_catalog"
        )
        parameter_record["default_parameter_reviews"]["rebuild"][
            "observable_effect"
        ] += " runtime mutation"
        with self.assertRaisesRegex(
            RuntimeError,
            "complete 1,501-record semantic proof changed",
        ):
            validate_complete_semantic_proof(parameter_drift)

    def test_canonical_builder_rejects_raw_inventory_adversaries(self) -> None:
        mutations = []
        source_drift = deepcopy(self.inventory)
        self.raw_module(
            source_drift,
            "payne_zero_synthesis.pipeline",
        )["sha256"] = "0" * 64
        mutations.append(source_drift)

        signature_drift = deepcopy(self.inventory)
        run_descriptor = self.raw_callable(
            signature_drift,
            "payne_zero_synthesis.pipeline.SynthesisPipeline.run",
        )
        run_descriptor["parameters"].remove("keep_slabs")
        mutations.append(signature_drift)

        constructor_drift = deepcopy(self.inventory)
        constructor_descriptor = self.raw_callable(
            constructor_drift,
            "payne_zero_synthesis.pipeline.SynthesisPipeline.__init__",
        )
        constructor_descriptor["parameters"].remove("source_path")
        mutations.append(constructor_drift)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            for index, inventory in enumerate(mutations):
                with self.subTest(adversary=index):
                    inventory_path = temporary_root / f"inventory-{index}.json"
                    output_path = temporary_root / f"coverage-{index}.json"
                    inventory_path.write_text(
                        json.dumps(inventory),
                        encoding="utf-8",
                    )
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(REPOSITORY_ROOT / "scripts/build_symbol_coverage.py"),
                            "--inventory",
                            str(inventory_path),
                            "--ledger",
                            str(REPOSITORY_ROOT / "COVERAGE.md"),
                            "--output",
                            str(output_path),
                        ],
                        cwd=REPOSITORY_ROOT,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output_path.exists())

    def test_canonical_builder_reapplies_exact_semantic_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_paths = [
                Path(temporary_directory) / "coverage-first.json",
                Path(temporary_directory) / "coverage-second.json",
            ]
            results = []
            for output_path in output_paths:
                results.append(
                    subprocess.run(
                        [
                            sys.executable,
                            str(REPOSITORY_ROOT / "scripts/build_symbol_coverage.py"),
                            "--inventory",
                            str(REPOSITORY_ROOT / "audit/paynezero_symbols.json"),
                            "--ledger",
                            str(REPOSITORY_ROOT / "COVERAGE.md"),
                            "--output",
                            str(output_path),
                        ],
                        cwd=REPOSITORY_ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                )
            first_bytes, second_bytes = (
                output_path.read_bytes() for output_path in output_paths
            )
        for result in results:
            self.assertIn("1501 reviewed overrides", result.stdout)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_bytes, LEDGER_PATH.read_bytes())


if __name__ == "__main__":
    unittest.main()
