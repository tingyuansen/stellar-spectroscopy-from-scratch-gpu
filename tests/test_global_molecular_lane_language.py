"""Freeze the explicit atmosphere/synthesis molecular-lane contract."""

from __future__ import annotations

from pathlib import Path
import re

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# This is a deliberately closed authority set. Historical audits record why the
# contract changed; they are evidence, not current policy.
AUTHORITATIVE_ACTIVE_FILES = (
    "BIBLE.md",
    "PLAN.md",
    "PASSDOWN.md",
    "COVERAGE.md",
    "design/global_chapter_contracts.md",
    "design/part5_part6_atmosphere_brief.md",
)
EXCLUDED_IMMUTABLE_AUDIT_HISTORY = (
    "design/global_interface_molecular_repair_independent_audit.md",
    "design/global_molecular_lane_language_guard_independent_audit.md",
)

# Exact rows and prose handoffs protect the accepted two-lane matrix. These are
# intentionally literals, not an attempt to infer meaning from arbitrary prose.
REQUIRED_CONTRACT_ANCHORS = (
    (
        "bible-water-row",
        "BIBLE.md",
        "| water / H\\(_2\\)O | `water_lines_path` is in the standard high-level "
        "source set; the dedicated water selector feeds the common runtime deposit. "
        "| `compile_h2o_partridge` exists, but the standard `_compile_molecular` "
        "path does not invoke it; this is compiler-only, not standard runtime or "
        "public opt-in support. | Ch. 8 states and tests the cross-lane difference; "
        "Ch. 11 composes the active atmosphere path. |",
    ),
    (
        "bible-h3plus-row",
        "BIBLE.md",
        "| H\\(_3^+\\) | `AtmosphereInput.h3plus_lines_path` and the selector/runtime "
        "path exist, but `source_line_paths()` does not supply a default file; it is "
        "opt-in by an explicit path. | A species-mass entry alone does not make a "
        "feature: no standard source compiler or pipeline wiring supplies "
        "H\\(_3^+\\). | Ch. 8 owns the exact optional/absent boundary; Ch. 11 may "
        "compose the atmosphere option without reteaching it. |",
    ),
    (
        "bible-active-selectors",
        "BIBLE.md",
        "- converted atomic, diatomic, TiO, and water source selectors are active "
        "atmosphere paths and must\n  not be described as unsupported “raw molecular "
        "selectors”; atmosphere H\\(_3^+\\) is instead an\n  explicit-path opt-in "
        "boundary.",
    ),
    (
        "plan-resolved-lane-boundary",
        "PLAN.md",
        "1. **Resolved boundary:** Chapter 8 implements and verifies H2O compiler "
        "parity,\n   but the complete textbook pipeline preserves and documents the "
        "pinned\n   standard synthesis runtime's omission of H2O. It must not confuse "
        "this\n   synthesis-only compiler boundary with the standard atmosphere "
        "water\n   selection/deposition path.",
    ),
    (
        "passdown-resolved-lane-boundary",
        "PASSDOWN.md",
        "1. **Resolved boundary:** H2O compiler parity is retained and verified, "
        "while\n   the pinned standard synthesis pipeline continues to compile text "
        "bands plus\n   TiO rather than H2O. This synthesis boundary does not apply "
        "to the standard\n   atmosphere water selection/deposition path.",
    ),
    (
        "coverage-water-row",
        "COVERAGE.md",
        "| water / H2O | default runtime through `water_lines_path`, the dedicated "
        "water selector, and common selected-line deposit | compiler-only: "
        "`compile_h2o_partridge` exists, but standard `_compile_molecular` omits it "
        "| Ch. 8 exact cross-lane boundary; Ch. 11 active atmosphere composition |",
    ),
    (
        "coverage-h3plus-row",
        "COVERAGE.md",
        "| H3+ | opt-in runtime: explicit `AtmosphereInput.h3plus_lines_path`; no "
        "file is returned by `source_line_paths()` | absent from the standard source "
        "compiler and pipeline; a mass-table entry is not runtime wiring | Ch. 8 "
        "optional/absent boundary; Ch. 11 may compose the atmosphere option |",
    ),
    (
        "coverage-active-selector-row",
        "COVERAGE.md",
        "| converted diatomic/TiO/water selectors | active standard high-level "
        "runtime paths, not unsupported branches | Ch. 8, 11 | verified |",
    ),
    (
        "coverage-synthesis-h2o-row",
        "COVERAGE.md",
        "| H2O source compilation | compiler-only; omitted by standard "
        "`_compile_molecular` | implement compiler parity and test runtime omission "
        "| verified |",
    ),
    (
        "chapter-contract-exact-boundary",
        "design/global_chapter_contracts.md",
        "**Exact boundary.** The standard high-level atmosphere workflow supplies\n"
        "diatomic, TiO, and water source arrays. Its H\\(_3^+\\) selector/deposit "
        "path is\nruntime-capable only when an explicit `h3plus_lines_path` is "
        "supplied; the\nstandard `source_line_paths()` set supplies no H\\(_3^+\\) "
        "file. In synthesis,\ntext bands and TiO run by default when "
        "`molecular_lines=True`; the H\\(_2\\)O\ncompiler exists, but the pinned "
        "standard pipeline does not invoke it. The\nsynthesis source compiler and "
        "pipeline do not supply H\\(_3^+\\); a generic\nspecies-mass entry does not "
        "establish runtime support. The serial compiler is\nnot rewritten with "
        "`prange`; there is no `torch.compile` path.",
    ),
    (
        "chapter-contract-water-row",
        "design/global_chapter_contracts.md",
        "| water / H\\(_2\\)O | standard `water_lines_path`; dedicated selector then "
        "common selected runtime deposit | `compile_h2o_partridge` is compiler-only; "
        "`_compile_molecular` omits it | Ch. 8 boundary; Ch. 11 active atmosphere "
        "composition |",
    ),
    (
        "chapter-contract-h3plus-row",
        "design/global_chapter_contracts.md",
        "| H\\(_3^+\\) | explicit-path opt-in selector/runtime; no default file | no "
        "standard source compiler or pipeline wiring | Ch. 8 boundary; optional "
        "Ch. 11 composition |",
    ),
    (
        "atmosphere-brief-chapter11-boundary",
        "design/part5_part6_atmosphere_brief.md",
        "- The standard atmosphere route supports converted atomic, diatomic, TiO, "
        "and\n  water selection. Atmosphere H3+ is a separate explicit-path opt-in "
        "because\n  `source_line_paths()` supplies no default H3+ file.\n"
        "- `_require_supported_run_setup` rejects the exact turbulent-pressure and\n"
        "  HLINOP branches; it has no blanket raw-molecular-selector guard. Only "
        "the\n  synthesis H2O compiler is compiler-only and omitted from the "
        "standard\n  synthesis runtime; atmosphere water selection is active.",
    ),
    (
        "atmosphere-brief-failure-boundary",
        "design/part5_part6_atmosphere_brief.md",
        "- Only the exact turbulent-pressure and HLINOP guards above are described "
        "as\n  failing loudly. The separate synthesis H2O compiler remains unwired "
        "in the\n  standard synthesis runtime; this does not disable atmosphere "
        "water opacity.",
    ),
    (
        "atmosphere-brief-course-close",
        "design/part5_part6_atmosphere_brief.md",
        "- The standard atmosphere workflow includes converted water-line selection "
        "and\n  opacity deposition; atmosphere H3+ remains an explicit-path opt-in. "
        "The\n  separate synthesis H2O compiler is verified as a compiler but omitted "
        "from\n  the standard synthesis runtime.",
    ),
)

# These are the false formulations actually observed during the P0.2/P1
# repair. The list is closed: it does not claim general natural-language
# understanding.
FORBIDDEN_EXACT_CLAIMS = (
    (
        "legacy-water-runtime-ambiguity",
        "Water-line compilation exists as a data-preparation capability but is not "
        "silently enabled in the standard runtime.",
    ),
    (
        "legacy-verified-workflow-ambiguity",
        "Available water-line compilation does not imply the standard verified "
        "workflow includes water opacity.",
    ),
    (
        "unqualified-workflow-does-not-include-water",
        "The standard workflow does not include water opacity.",
    ),
    (
        "unqualified-runtime-without-h2o",
        "The standard runtime is without H2O opacity.",
    ),
    (
        "unqualified-pipeline-excludes-water",
        "The standard pipeline excludes water opacity.",
    ),
    (
        "unqualified-runtime-omits-water",
        "The standard runtime omits water opacity.",
    ),
)
FORBIDDEN_REGEX_CLAIMS = (
    (
        "raw-molecular-selector-failure-or-rejection",
        re.compile(
            r"\b(?:unsupported\s+)?raw[- ]molecular selectors?\b"
            r"[\s\S]{0,160}\b(?:fail(?:s|ed|ing)?|reject(?:s|ed|ing)?)\b",
            re.IGNORECASE,
        ),
        "Unsupported raw molecular selectors must fail loudly.",
    ),
)


def _load_active_authority() -> dict[str, str]:
    """Load only the explicitly named current authority documents."""

    return {
        relative_path: (REPOSITORY_ROOT / relative_path).read_text()
        for relative_path in AUTHORITATIVE_ACTIVE_FILES
    }


def _normalize_whitespace(text: str) -> str:
    """Make exact historical phrases insensitive only to Markdown wrapping."""

    return " ".join(text.split()).casefold()


def _contract_failures(documents: dict[str, str]) -> list[str]:
    """Return deterministic missing-anchor and known-false-claim failures."""

    failures = []
    for label, relative_path, anchor in REQUIRED_CONTRACT_ANCHORS:
        if anchor not in documents[relative_path]:
            failures.append(f"missing anchor {label} in {relative_path}")

    for relative_path, text in documents.items():
        normalized_text = _normalize_whitespace(text)
        for label, claim in FORBIDDEN_EXACT_CLAIMS:
            if _normalize_whitespace(claim) in normalized_text:
                failures.append(f"forbidden claim {label} in {relative_path}")
        for label, pattern, _sample in FORBIDDEN_REGEX_CLAIMS:
            if pattern.search(text):
                failures.append(f"forbidden claim {label} in {relative_path}")
    return failures


def test_authority_scope_is_exact_and_audit_history_is_excluded() -> None:
    """All six active authorities, and no immutable audits, define the policy."""

    assert AUTHORITATIVE_ACTIVE_FILES == (
        "BIBLE.md",
        "PLAN.md",
        "PASSDOWN.md",
        "COVERAGE.md",
        "design/global_chapter_contracts.md",
        "design/part5_part6_atmosphere_brief.md",
    )
    assert set(AUTHORITATIVE_ACTIVE_FILES).isdisjoint(EXCLUDED_IMMUTABLE_AUDIT_HISTORY)
    assert all(
        (REPOSITORY_ROOT / relative_path).is_file()
        for relative_path in (
            *AUTHORITATIVE_ACTIVE_FILES,
            *EXCLUDED_IMMUTABLE_AUDIT_HISTORY,
        )
    )


def test_live_authority_satisfies_the_explicit_molecular_lane_contract() -> None:
    """The accepted anchors are present and the known false claims are absent."""

    assert _contract_failures(_load_active_authority()) == []


@pytest.mark.parametrize(
    ("label", "relative_path", "anchor"),
    REQUIRED_CONTRACT_ANCHORS,
    ids=[record[0] for record in REQUIRED_CONTRACT_ANCHORS],
)
def test_each_required_anchor_is_a_failing_gate_when_altered(
    label: str,
    relative_path: str,
    anchor: str,
) -> None:
    """Changing any literal row or handoff must fail its named contract gate."""

    documents = _load_active_authority()
    assert documents[relative_path].count(anchor) == 1
    replacement = f"{anchor[:-1]}?"
    documents[relative_path] = documents[relative_path].replace(
        anchor,
        replacement,
        1,
    )

    assert f"missing anchor {label} in {relative_path}" in _contract_failures(documents)


@pytest.mark.parametrize(
    ("label", "claim"),
    FORBIDDEN_EXACT_CLAIMS,
    ids=[record[0] for record in FORBIDDEN_EXACT_CLAIMS],
)
def test_each_known_false_phrase_is_a_failing_gate(
    label: str,
    claim: str,
) -> None:
    """Reintroducing any historical false phrase must fail deterministically."""

    documents = _load_active_authority()
    target = "PLAN.md"
    documents[target] = f"{documents[target]}\n\n{claim}\n"

    assert f"forbidden claim {label} in {target}" in _contract_failures(documents)


@pytest.mark.parametrize(
    ("label", "pattern", "sample"),
    FORBIDDEN_REGEX_CLAIMS,
    ids=[record[0] for record in FORBIDDEN_REGEX_CLAIMS],
)
def test_each_known_false_regex_is_a_failing_gate(
    label: str,
    pattern: re.Pattern[str],
    sample: str,
) -> None:
    """Reintroducing the bounded raw-selector pattern must fail its gate."""

    assert pattern.search(sample)
    documents = _load_active_authority()
    target = "PASSDOWN.md"
    documents[target] = f"{documents[target]}\n\n{sample}\n"

    assert f"forbidden claim {label} in {target}" in _contract_failures(documents)
