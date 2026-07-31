# Global molecular-lane language guard — final independent audit

Date: 2026-07-30

## Decision

**P1 active-prose molecular-lane guard — REJECT.**

The repaired predicate closes both earlier synthesis-token masking defects and
passes its committed 46-case suite. It is still not a complete guard for the
stated language and authority boundary. Three independently reproduced gaps
remain:

1. periods inside an infix parenthetical (`e.g.`, `i.e.`, `Sec.`, or a decimal)
   disconnect the standard boundary from its negative water predicate;
2. plural standard boundaries (`runtimes`, `pipelines`, and `workflows`) are
   not matched; and
3. the allowlist omits active authoritative documents, most importantly
   `design/global_chapter_contracts.md`, even though `PLAN.md` explicitly calls
   it the authoritative chapter-to-chapter contract.

This is a nonblocking P1 test-quality decision. The independently accepted
P0.2 prose repair remains accepted at its recorded hashes. P0.3 was not
assessed.

## Scope and immutable inputs

Only the language guard and its candidate report were under decision:

| Input | Required SHA-256 | Observed |
| --- | --- | --- |
| `tests/test_global_molecular_lane_language.py` | `3e91a9a33d86e7fbcb4a046d42c544b1cd195137f9a0c2ebc6f909d408b817d8` | exact match |
| `design/global_interface_molecular_repair_candidate.md` | `6f9176bb8cc11e675fc2f441bd4465cb4f366bb95a3ee17bda1d3d23417261fd` | exact match |

The accepted P0.2 history in
`design/global_interface_molecular_repair_independent_audit.md` was read in
full through its two later P1 rejection addenda. That immutable history was
not edited or treated as active prose.

No test, candidate, active prose, accepted audit, or P0.3 artifact was changed
during this review.

## Committed verification

The supplied checks all pass:

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_global_molecular_lane_language.py

46 passed in 0.04s
```

```text
/Users/ysting/anaconda3/bin/ruff check \
  tests/test_global_molecular_lane_language.py

All checks passed!
```

```text
/Users/ysting/anaconda3/bin/ruff format --check \
  tests/test_global_molecular_lane_language.py

1 file already formatted
```

`git diff --check` is clean for the supplied test and candidate report.
Passing the committed cases is not sufficient evidence for the broader claim,
so the predicate was also invoked directly on fresh in-memory prose. No
repository input was mutated.

## What the repair now handles correctly

The following additional probes behaved correctly:

- commas and ordinary parentheticals between subject and predicate;
- em-dash and en-dash infix clauses;
- leading or infix `while`, `but`, `although`, and `and` clauses containing an
  unrelated synthesis token;
- preceding and following unrelated synthesis sentences;
- reordered supported predicates, including “Water opacity is what the
  standard atmosphere runtime omits,” “Absent from the standard atmosphere
  runtime is water opacity,” and passive `excluded`;
- multiple matched standard boundaries: any matched boundary lacking
  `synthesis` makes a negative-water unit fail closed;
- multiple synthesis-qualified standard boundaries;
- line wrapping between `standard atmosphere`, `workflow`, `water`, and
  `opacity`;
- singular and plural `water line`/`water-line` wording;
- plain `H2O` and the active Markdown spelling `H\(_2\)O`;
- explicit negative `standard ... synthesis ... runtime/pipeline/workflow`
  forms; and
- positive standard-atmosphere water claims.

The currently active files contain no Unicode-subscript `H₂O`; therefore the
predicate's failure to recognize that unused spelling is not a present
acceptance blocker. If that typography is introduced later, it must be added
deliberately.

## P1.1 — terminal-period splitting still masks infix claims

`SENTENCE_UNIT_SEPARATOR = re.compile(r"[.!?;]+")` treats every period as a
terminal sentence boundary. That is unsafe inside the exact kind of infix
parenthetical this guard must preserve.

Independent probes gave:

| Unqualified negative claim | Predicate result | Required |
| --- | --- | --- |
| `The standard atmosphere runtime (synthesis is documented, e.g. in Chapter 8) omits water opacity.` | accept | reject |
| `The standard atmosphere runtime (i.e. the atmosphere lane) excludes H2O.` | accept | reject |
| `The standard atmosphere runtime (version 1.0) omits water opacity.` | accept | reject |
| `The standard atmosphere runtime (see Sec. 8) omits H2O.` | accept | reject |

For the first claim, `_sentence_units` returns:

```text
The standard atmosphere runtime (synthesis is documented, e
g
in Chapter 8) omits water opacity
```

The first fragment has the standard atmosphere boundary but no water/negative
predicate. The last has water plus `omits` but no standard boundary. The
predicate therefore accepts the false claim. This is structurally the same
masking defect as the prior infix-clause rejection, now triggered by internal
punctuation rather than a conjunction.

A repair must protect abbreviations and decimal points, or otherwise identify
actual terminal sentence boundaries without severing a parenthetical subject
from its predicate. These four forms need committed regression cases.

## P1.2 — plural standard boundaries are invisible

`STANDARD_RUNTIME` ends in singular
`(?:runtime|pipeline|workflow)`. Consequently all three ordinary plural claims
below are accepted:

| Unqualified negative claim | Predicate result | Required |
| --- | --- | --- |
| `The standard atmosphere runtimes omit H2O.` | accept | reject |
| `The standard atmosphere pipelines exclude water opacity.` | accept | reject |
| `The standard atmosphere workflows are without H2O.` | accept | reject |

This is not a speculative spelling: plural standard workflows and pipelines
are normal English. The boundary should recognize
`runtimes?`, `pipelines?`, and `workflows?`, with positive and
synthesis-qualified plural controls so the change is not one-sided.

The negative vocabulary has a similar avoidable blind spot. Fresh probes of
“has no H2O,” “lacks water opacity,” “includes no water opacity,” “never
includes H2O,” and “cannot use water opacity” were all accepted for a standard
atmosphere boundary. At least these common negative predicates need explicit
disposition and tests; otherwise the function and candidate must not claim to
guard negative standard-water statements generally.

## P1.3 — active authority is outside the scan

The allowlist contains:

```text
BIBLE.md
PLAN.md
PASSDOWN.md
design/part5_part6_atmosphere_brief.md
```

All four files exist, and no immutable audit or candidate report is present.
The historical-record exclusion is intentional and correct.

The active side is incomplete. `PLAN.md` states that:

```text
design/global_chapter_contracts.md is the authoritative chapter-to-chapter
contract.
```

It also defines `COVERAGE.md` as the completeness ledger. Both are omitted.
Adding those two files only in memory to `ACTIVE_PROSE_FILES` immediately
exposed two units in the authoritative chapter contract that the supplied
predicate classifies as unqualified negative standard-water claims:

- the Chapter 8 exact-boundary paragraph beginning at line 418; and
- “explicit test that standard runtime omits `H\(_2\)O`” beginning at line
  431.

This audit does not decide whether those active sentences should be rewritten
or whether a deliberately context-aware policy should accept them. It does
show that the current guard cannot enforce its declared phrase-local policy
over the project's live authority because it never reads that authority.

There is a second scope hole: `_prose_blocks` drops every Markdown table row,
while the two-lane molecular truth is intentionally recorded in the Bible and
coverage matrices. If tables remain outside this prose guard, their molecular
cells need an equivalent explicit status assertion. Simply adding
`COVERAGE.md` to the file tuple would not protect its authoritative matrix.

Immutable audits must stay excluded. Active Bible, plan, passdown, global
chapter contract, detailed brief, and the relevant coverage claims must all be
covered either by this guard or by a clearly named equivalent guard.

## Precision note — a correct mixed sentence is rejected

The semicolon-separated form is accepted:

```text
The standard synthesis runtime omits H2O;
the standard atmosphere runtime includes water opacity.
```

The equally true comma/`while` form is rejected:

```text
The standard synthesis runtime omits H2O, while the standard atmosphere
runtime includes water opacity.
```

The current candidate intentionally fails closed on every standard boundary in
a negative-water sentence unit, including a boundary belonging only to a
positive clause. This does not create a false-atmosphere masking route, and a
semicolon is an available rewrite, so it is not the basis of this rejection.
It is nevertheless an over-broad behavior to retain as an explicit policy
test—or refine with clause-local attribution—rather than mistake for semantic
precision.

## Required closure evidence

Before this P1 guard can be accepted:

1. add regression cases for the four internal-period parentheticals;
2. reject plural standard runtime/pipeline/workflow claims unless each
   relevant matched boundary names synthesis;
3. disposition and test common negative forms such as `has no`, `lacks`,
   `includes no`, `never includes`, and `cannot use`;
4. include the explicitly authoritative global chapter contract and protect
   the relevant coverage/Bible matrix claims without scanning immutable audit
   history;
5. retain the existing unrelated-synthesis, multiple-boundary, line-wrapping,
   punctuation, active `H2O`/`H\(_2\)O`, valid synthesis-negative, and positive
   atmosphere controls; and
6. rerun focused pytest, Ruff check, Ruff format check, and `git diff --check`
   under a new immutable input identity.

## Final boundary

**P1 active-prose molecular-lane guard — REJECT** at:

- `tests/test_global_molecular_lane_language.py`
  `3e91a9a33d86e7fbcb4a046d42c544b1cd195137f9a0c2ebc6f909d408b817d8`;
  and
- `design/global_interface_molecular_repair_candidate.md`
  `6f9176bb8cc11e675fc2f441bd4465cb4f366bb95a3ee17bda1d3d23417261fd`.

This decision concerns only the nonblocking P1 guard. It does not reopen the
accepted P0.2 prose repair and does not assess P0.3.

---

# Repair re-audit — deterministic contract-anchor redesign

Date: 2026-07-30

## R1. Scope and supersession boundary

This addendum re-audits only the replacement P1 guard. It preserves the
rejection above as the correct decision for the former natural-language
predicate. The replacement deletes that parser and adopts a narrower,
deterministic contract: exact accepted anchors plus a closed set of historical
contradictions.

The supplied inputs match exactly:

| Input | Required and observed SHA-256 |
| --- | --- |
| `tests/test_global_molecular_lane_language.py` | `88e7cbffa3c89e189fb209492c06233ff6e814c3c0838e11f08bde09d6776ca7` |
| `design/global_interface_molecular_repair_candidate.md` | `707fc80a96299f41d1ca170f22d356ae6cad01b152d36017dbc667044693213c` |
| this independent audit before the addendum | `4a1a2a5443e6f03a23c1410b2605a0f9b26291c4f6faff80d6b531874d6a1108` |

Neither input was edited. P0.2 remains accepted at its prior exact evidence;
P0.3 was not assessed.

## R2. Exact authority set: pass

`AUTHORITATIVE_ACTIVE_FILES` is exactly, and in this order:

```text
BIBLE.md
PLAN.md
PASSDOWN.md
COVERAGE.md
design/global_chapter_contracts.md
design/part5_part6_atmosphere_brief.md
```

`_load_active_authority()` returned exactly those six keys, and all six paths
exist. This closes the prior omission of the coverage ledger and authoritative
global chapter contract.

The two immutable independent-audit records are named in
`EXCLUDED_IMMUTABLE_AUDIT_HISTORY`, are disjoint from the active tuple, and
exist. More importantly, the loader has no discovery or recursive scan: it
reads only the literal six-file authority tuple. Historical rejection language
therefore cannot become current policy accidentally.

Independent in-memory removal of each of the six files, one at a time, made
`test_authority_scope_is_exact_and_audit_history_is_excluded` fail. Adding an
audit to the active tuple would likewise fail the exact tuple assertion. The
scope is both complete and closed.

## R3. Exact contract anchors: pass

The replacement freezes fifteen literal rows and handoffs:

| Authority | Anchor count | Protected roles |
| --- | ---: | --- |
| `BIBLE.md` | 3 | water two-lane row; H3+ row; active-selector prohibition |
| `PLAN.md` | 1 | resolved synthesis-H2O versus atmosphere-water boundary |
| `PASSDOWN.md` | 1 | live resolved two-lane boundary |
| `COVERAGE.md` | 4 | water, H3+, active selectors, synthesis H2O role rows |
| `design/global_chapter_contracts.md` | 3 | exact prose boundary; water row; H3+ row |
| `design/part5_part6_atmosphere_brief.md` | 3 | Chapter 11 boundary; failure boundary; course close |

Every anchor occurs exactly once in its assigned file. The unmodified six-file
mapping returns no `_contract_failures`.

This design now protects Markdown matrices directly; it does not silently
discard table rows as the rejected prose parser did.

### R3.1 Independent mutation audit

The committed mutation changes only an anchor's final byte. I did not rely on
that proof. For each of the fifteen anchors, an independent in-memory mutation
changed a scientific role word inside the anchor, using changes such as:

```text
compiler-only -> runtime-enabled
opt-in        -> default-on
active        -> inactive
omission      -> inclusion
rather than   -> together with
water         -> argon
```

All fifteen mutations produced the corresponding
`missing anchor <label> in <path>` failure. No repository file was written.
Thus every row and prose handoff is a live gate, including every role row
requested in this re-audit.

## R4. Source fidelity of the pinned anchors: pass

The read-only Payne Zero checkout was independently verified at:

```text
9c44001feae40b85146630499e6f8a5fed42e5af
```

The four frozen roles agree with that source:

1. **Standard atmosphere water is active.**
   `payne_zero_atmosphere/source_catalogs.py:142-166` returns
   `water_lines_path`; `runner.py:599-646` forwards it;
   `line_selection.py:1136-1145` reads and selects it; and
   `runner.py:719-800` deposits the common selected catalog.
2. **Atmosphere H3+ is explicit-path opt-in.**
   `AtmosphereInput` exposes `h3plus_lines_path`, and
   `line_selection.py:1147-1161` implements its selector, while
   `source_line_paths()` returns no H3+ default.
3. **Synthesis H2O is compiler-only at the standard pipeline boundary.**
   `compile_h2o_partridge` exists at
   `source_catalog_molecular_compiler.py:960-1086`, but the only repository
   definition/call-site search result is that definition. The standard
   `_compile_molecular` path at `pipeline.py:1168-1261` compiles manifest text
   bands and TiO only.
4. **Converted atmosphere selectors are not an unsupported raw-selector
   branch.** `generate_selected_lines` has active atomic, diatomic, TiO, water,
   and optional H3+ routes. `_require_supported_run_setup` at
   `runner.py:2003-2020` rejects iteration count, turbulent pressure, and
   HLINOP; it has no blanket molecular-selector rejection.

The anchors therefore pin the accepted source behavior rather than merely
freezing mutually consistent prose.

## R5. Historical contradictions: pass

The guard rejects six closed exact claims after whitespace normalization and
case folding:

- the two historical ambiguous water-runtime formulations;
- “standard workflow does not include water”;
- “standard runtime is without H2O”;
- “standard pipeline excludes water”; and
- “standard runtime omits water.”

Independent mutation injected every exact claim into every active authority,
using uppercase and a line break between every word. All \(6\times6=36\)
injections produced the correct named failure. This proves the normalization
and all-file scan independently of the committed PLAN-only parameterization.

The bounded raw-selector regex was independently tested with:

```text
Unsupported raw molecular selectors must fail loudly.
Raw molecular selectors and unsupported line branches fail loudly.
An unsupported raw-molecular selector must be rejected loudly.
RAW MOLECULAR SELECTORS
MUST FAIL LOUDLY.
```

Every variant matched. Injecting every variant into every authority produced
all \(4\times6=24\) expected failures. The regex accepts singular/plural,
space/hyphen, case/wrapping, and fail/reject inflections required by the
historical loud-guard claim.

The current six authorities contain none of the closed contradictions.

## R6. Scope honesty: pass

The replacement no longer claims universal natural-language detection:

- the module docstring says it freezes the explicit molecular-lane contract;
- anchors are documented as intentional literals;
- forbidden exact claims are documented as the false formulations observed in
  the P0.2/P1 repair;
- the regex list is explicitly closed; and
- the candidate states that the guard does not understand arbitrary English,
  exhaust every paraphrase, or replace editorial review.

That is an honest and useful regression boundary. The internal-period,
plural-noun, negative-vocabulary, and mixed-clause counterexamples from the
earlier rejection are no longer falsely claimed as parsed language. Scientific
drift in the accepted contract rows fails deterministically, while novel prose
remains the responsibility of editorial and independent review.

Excluding immutable audits is therefore correct: they must retain rejected
historical sentences. The exact active tuple plus literal table/prose anchors
protects current authority without rewriting evidence history.

## R7. Focused verification

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_global_molecular_lane_language.py

24 passed in 0.44s
```

```text
/Users/ysting/anaconda3/bin/ruff check \
  tests/test_global_molecular_lane_language.py

All checks passed!
```

```text
/Users/ysting/anaconda3/bin/ruff format --check \
  tests/test_global_molecular_lane_language.py

1 file already formatted
```

`git diff --no-index --check /dev/null` emitted no whitespace error for either
untracked input. Its exit status was the expected `1` because each comparison
is an added file.

## R8. Final repaired verdict

**P1 deterministic molecular-lane contract-anchor guard — ACCEPT** at:

- `tests/test_global_molecular_lane_language.py`
  `88e7cbffa3c89e189fb209492c06233ff6e814c3c0838e11f08bde09d6776ca7`;
  and
- `design/global_interface_molecular_repair_candidate.md`
  `707fc80a96299f41d1ca170f22d356ae6cad01b152d36017dbc667044693213c`.

The rejection above remains the correct record for the superseded NLP-style
guard. This addendum accepts only the narrower deterministic replacement.
P0.2 remains accepted. P0.3 was not assessed.
