# Independent Audit — Global Goal Refinement

**Decision: ACCEPT**

The current governance set faithfully incorporates the user's latest direction. This
is an acceptance of the **rewritten goal, architecture, and review system**, not an
acceptance of the unfinished textbook. The authorities explicitly prevent such a
misreading: Chapters 6–15 are not yet eligible for chapter acceptance, and the
symbol-level completeness proof still has an open mixed-module/branch-sensitive
disposition gate.

## Audited snapshot

This bounded audit read only the following files. SHA-256 values were recomputed
immediately before the audit test:

| File | SHA-256 |
| --- | --- |
| `PLAN.md` | `ffec9f4582b110c7217930e1fbe8a23d90f3663cd9b74c0b5b21d787e25444ea` |
| `BIBLE.md` | `65387d009b732446252b5392afcbaa7a12fcb2c6db6083d9cf8eed0b6b1b36ac` |
| `PASSDOWN.md` | `a22afc714d2dc5c34723e6529f879db75e83ef4384263175d299b2073f027ba5` |
| `design/pedagogical_flow_rubric.md` | `4626b53dd9953df120398fd8762504b9cb1a429b8b0e6df4a1e5a049a433f35d` |
| `design/global_chapter_contracts.md` | `1ef69f8434af3f19a1545546aa1dad03b988316d19f686f44afab5d2bfdeafc2` |
| `COVERAGE.md` | `dbca7a801db525e467f5f0a6162591d0e057eeaa5056d1e690e8e137b1138b84` |
| `book/registry.py` | `b42b79e6722df1763234ed1a8f25d20030606e23257627d52083c94404a1db97` |
| `tests/test_book_governance.py` | `ce15bb9d4dc45930d4ff06e513c20904219ca3b03228b6c5c4dfa50d5df2efc3` |

The bounded executable check

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /Users/ysting/anaconda3/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_book_governance.py
```

passed: **5 passed in 0.11 s**.

## Requirement-by-requirement decision

| Latest requirement | Governing evidence | Decision |
| --- | --- | --- |
| Fifteen-chapter half-semester scope without material loss | `PLAN.md` declares fifteen reader-facing chapters and retains the 29-unit map as a lossless checklist. `design/global_chapter_contracts.md` supplies measured density gates for the exceptional sixteenth chapter. `book/registry.py` enforces consecutive Chapters 1–15. | Accept |
| Final-year undergraduate / first-year graduate level | `BIBLE.md` makes this the mission and defines the audience floor. `PASSDOWN.md` repeats it in the active refined goal. The pedagogical rubric asks whether a final-year undergraduate can follow without outside reading. | Accept |
| Assume only basic mathematics, physics, and ordinary Python | `BIBLE.md` lists the precise prerequisite floor and explicitly forbids assuming astronomy, quantum/statistical mechanics, spectroscopy, transfer, atmosphere physics, numerical methods, NumPy, Numba, Torch, GPUs, or machine learning. It requires advanced ideas to be built through physical question, example, mathematics, small code, and check. | Accept |
| Lecture-1 / Agent4Astro causal flow | `PLAN.md` states the causal spine. `design/pedagogical_flow_rubric.md` converts the reference style into seven required acts, paragraph-level and code-output gates, and neighbor/whole-book checks. `design/global_chapter_contracts.md` gives every chapter a question, output, check, close, and causal handoff. | Accept |
| No detached exercises; useful checks stay in the main text | The prohibition appears in all central authorities. The rubric requires prediction, execution, and immediate interpretation in place. The governance test rejects detached exercise-style headings in every published chapter. | Accept |
| Self-contained build from scratch | `BIBLE.md` defines “from scratch,” prohibits runtime imports from either external source tree, separates static/subset/fixture/golden roles, and requires comparison targets to be loaded only after the reader's calculation. The governance test rejects the two external absolute paths in published notebook sources. | Accept |
| Payne Zero named sparingly while exact production notation and interfaces are preserved | `BIBLE.md` distinguishes the narrative physical noun from the earned production boundary and requires exact public names, fields, keys, arguments, defaults, and axis orders. The global contract resolves interface-specific spellings such as `r_grid` versus `resolution` rather than inventing a cleaned-up notation. | Accept |
| Complete nonredundant atmosphere, synthesis, and learned-initializer/emulator scope; no fitting | `BIBLE.md` includes thermochemistry, continua, every active line family and honest boundary, transfer, physical iteration, remapping/convergence, five-label/CNO/direct-abundance initializers, GPU synthesis, Numba atmosphere work, data/caches, and reproducibility; it explicitly excludes spectrum fitting. `COVERAGE.md` assigns every atmosphere and synthesis module, feature, branch, and end-to-end gate. Chapter 14 owns the complete learned profile-transform/PCA/network initializer path and mandatory physical closure. | Accept as a scope contract; implementation remains incomplete |
| Original website-aesthetic schematics generated for the textbook | `BIBLE.md` makes the website read-only style evidence, forbids reuse as reader-facing assets, and requires original objects/composition, an owned prompt/specification, provenance, alt text, caption, and hash. `PLAN.md` requires review of the relevant website figure/prompt followed by newly generated artwork and mobile/native render review. | Accept |
| Professional, simple, usually one-panel plots | `BIBLE.md` specifies one panel/one claim, a paper-inspired color-safe palette and typography, units, inward ticks, white background, and rendered visual inspection. The rubric and global contract require immediate interpretation and reject dense dashboards/default styling. | Accept |
| Each chapter closes with a summary and causal next link | `BIBLE.md`, the rubric, and the global close template require this. The governance test enforces `Chapter summary`, `### Next:`, and the next reader URL for published Chapters 1–14. | Accept |
| Repeated local, neighbor, redundancy, coverage, and global zoom-out audits | `BIBLE.md` defines six review gates; `design/global_chapter_contracts.md` defines a ten-pass wave; `PASSDOWN.md` makes the same cadence operational and forbids direct subagent acceptance. | Accept |
| Clean, minimal, runnable repository | `PLAN.md` gives an explicit cleanup slice; `PASSDOWN.md` says Git history is the archive and the active tree contains only owned sources, data, tests, and products; `BIBLE.md` makes duplicate builders, mixed-role fixtures, filename inversions, environmental notebook failures, and unowned generated artifacts completion blockers. | Accept as a completion contract |
| Continuously readable local web edition | `PASSDOWN.md` records the live reader at `http://127.0.0.1:8765/`. `BIBLE.md` makes coherent/current local reading a completion condition. The registry supplies all fifteen navigation records, while the governance test verifies honest in-construction pages for unpublished chapters. | Accept |

## Actionable findings

### P0 — none in the rewritten goal or global architecture

No latest user requirement is absent or contradicted in the audited governance
set. In particular, the fifteen-chapter compression is not being achieved by
discarding the earlier 29-unit material, and the initializer is not being
misrepresented as a converged atmosphere.

### P1 — finish the open semantic coverage proof before Chapter 7 expansion

`PASSDOWN.md` correctly records the remaining P0.3 implementation gate:
mixed-module and branch-sensitive public symbols need semantic overrides beyond
module-level ownership. `COVERAGE.md` likewise says the generated public-symbol
inventory is required and that any unmapped symbol must fail the audit.

**Action:** complete and independently review those symbol-level dispositions
before Chapter 7 construction broadens the line-family surface. Require every
symbol to have one semantic role, one primary owner, one canonical textbook
symbol or explicit boundary disposition, and one parity gate. Do not convert
the present governance ACCEPT into a completeness claim until that ledger is
green.

### P1 — expand executable governance as chapters become publishable

The five passing tests enforce important structural promises—no detached
exercise headings, no external checkout paths, no Python source dumps in
Markdown, summary/next-link closure, and honest planned pages—but the other
accepted promises currently depend mostly on human review and ledgers.

**Action:** before final whole-book acceptance, add machine-readable gates for:

- exact registry/coverage ownership of every public symbol and branch;
- schematic prompt/provenance/hash/alt-text ownership;
- numerical-plot metadata plus rendered-review status;
- canonical data roles and absence of mixed-role bundles;
- current generated reader outputs versus canonical chapter sources;
- active-tree ownership and absence of superseded generated artifacts.

These tests should support, not replace, the paragraph-level and rendered
scientific reviews.

### P1 — preserve the explicit distinction between plan acceptance and book acceptance

`COVERAGE.md` marks Chapters 6–15 as not yet eligible, and most remaining
atmosphere/synthesis end-to-end gates are planned. The local reader's honest
in-construction pages are therefore the correct behavior.

**Action:** keep PASSDOWN and the reader synchronized after every atomic
chapter publication. Continue stating chapter acceptance, whole-book
retention, and final integration as three different statuses.

### P2 — make website-generator reuse auditable without creating a dependency

The authorities correctly demand original textbook artwork and identify the
website as a read-only visual reference. They specify review of the website
figure/prompt and a textbook-owned schematic specification, but do not yet
require a record of which website generation `.py` source informed a new
schematic.

**Action:** when the website's generator code materially informs a schematic
workflow, record its exact read-only path and hash in the visual provenance
record, together with the textbook-owned prompt/specification and generated
asset hash. Copy or adapt only the minimal generation machinery needed inside
this repository; do not make the website checkout a reader/runtime
dependency.

## Final assessment

The global goal is now appropriately ambitious and precise: a self-contained,
causally taught, executable stellar-spectroscopy course that constructs the
working atmosphere and synthesis calculation from basic foundations, reaches
the exact production boundary only after the physics is earned, and verifies
the result without turning into repository documentation. The plan should
remain in force. The next work is disciplined execution of it, with the open
symbol-coverage gate closed before the opacity surface expands.
