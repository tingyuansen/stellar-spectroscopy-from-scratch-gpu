# Independent audit of the global interface and molecular-contract repair

## 1. Audit choice, scope, and evidence

This is a separate audit rather than an appended section in
`design/global_coverage_zoomout_after_ch06_evidence.md`. The original report is
an immutable rejection at its own input hashes. Appending a later, partly
accepting decision would make it unclear which bytes each verdict governs.

This audit decides only whether P0.1 and P0.2 from that original report have
been repaired. It does **not** assess P0.3 or imply that the global coverage
proof is otherwise complete.

Exact evidence used:

| Evidence | SHA-256 or identity |
| --- | --- |
| pinned Payne Zero checkout `HEAD` | `9c44001feae40b85146630499e6f8a5fed42e5af` |
| original global rejection | `e4f08434604ccd308264e40ea0e934811c0d26cfbb3f7a6c15fca5c51cec8baf` |
| repair candidate | `929fe45de716f23894e6f762881d6022b2ebe869a43a35f7ccf14df62b3c1ff9` |
| repaired `BIBLE.md` | `65387d009b732446252b5392afcbaa7a12fcb2c6db6083d9cf8eed0b6b1b36ac` |
| repaired `COVERAGE.md` | `dbca7a801db525e467f5f0a6162591d0e057eeaa5056d1e690e8e137b1138b84` |
| repaired global chapter contracts | `1ef69f8434af3f19a1545546aa1dad03b988316d19f686f44afab5d2bfdeafc2` |
| atmosphere detailed brief searched for contradictions | `bc6162c56d60817500eba57a775ff42722825e9486af81dc590ae1e54dc6c301` |
| Chapter 6 Fe-record candidate audit searched for name drift | `7d42b47d2bbb2de2e981786fb53711e684feff5640a2ad7a4373ae8c4920159c` |
| Chapter 6 synthesis-oracle plan searched for name drift | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` |
| current `PLAN.md` | `77f76043d97a494c373aa1d4fe9133bda1d98041017c4c892881aee41cc3014e` |
| current `PASSDOWN.md` | `3ea9ea4ba9467062c0b0aae559ab7d90a777c4d29761b16e5ee066a8a4cba4e8` |

The source and the three repaired contract files were read only during this
audit.

## 2. Executive decision

| Repair | Decision | Reason |
| --- | --- | --- |
| P0.1 — exact intrinsic-grid spellings | **ACCEPT** | The three repaired global contracts now match every pinned API, engine, compiler, record, and CLI spelling checked. The intrinsic-grid/instrument distinction is explicit and correct. |
| P0.2 — exact two-lane molecular status and ownership | **REJECT, narrowly** | The repaired global matrices themselves are source-correct, but the retained atmosphere source/API brief still contains false or lane-ambiguous runtime statements at three separate locations. Because the global authority contract says detailed briefs remain the full field/API/source inventories, these are not harmless archival prose. |

P0.2 can close without changing the architecture. Remove or correct the stale
detailed-brief statements, update the repair candidate's incomplete
contradiction record, and rerun this bounded audit.

## 3. P0.1 independent source audit

### 3.1 Mathematical quantity and warning

The repaired documents consistently reserve
\(R_{\rm grid}=\lambda/\Delta\lambda\) for the intrinsic spacing of adjacent
model samples:

- `BIBLE.md:292-301,335-338`;
- `COVERAGE.md:50-62`; and
- `design/global_chapter_contracts.md:78-91`.

They also state that an instrumental resolving power or line-spread operator is
a separate downstream operation. This is supported by the CLI's own help text,
which calls the argument a “logarithmic wavelength-grid density” and explicitly
says it is not instrumental resolution
(`payne_zero_synthesis/cli.py:123-132`,
`prewarm.py:239-250`).

The formula is consistent with the implementation:
`Grid.ratio` is `1.0 + 1.0 / self.resolution`
(`atomic_lines.py:168-182`). Thus the exact source field called `resolution`
controls intrinsic adjacent-sample density; its name does not turn it into an
instrument model.

**Result: pass.**

### 3.2 Public label and archive/in-memory APIs

The repaired matrices distinguish the two public synthesis APIs correctly.

`synthesize` has:

```text
resolution: float = 20000.0
```

and no `r_grid` keyword (`payne_zero_synthesis/api.py:442-452`).

`synthesize_from_labels` has both:

```text
r_grid: float | None = None
resolution: float | None = None
```

(`api.py:732-752`). Its docstring calls `r_grid` the label API spelling and
`resolution` its alias (`api.py:756-760`). `_resolved_r_grid` behaves exactly
as the repaired contracts say:

- neither supplied → `20000.0`;
- only `r_grid` supplied → that value;
- only `resolution` supplied → that value;
- equal double specification → accepted; and
- unequal double specification → `ValueError("r_grid and resolution specify
  different values")`.

The last behavior was also executed directly under the pinned checkout, in
addition to reading `api.py:221-237`.

**Result: pass.**

### 3.3 Exact engine, grid, cache, and compiler spelling

Every checked lower boundary uses `resolution`, as the repaired matrices
require:

- `Grid.resolution`, plus serialized and reloaded catalog metadata:
  `atomic_lines.py:168-182,451-488`;
- molecular persistent-cache identity:
  `pipeline.py:704-783`;
- invariant-cache key and grid construction:
  `pipeline.py:800-890`;
- `SynthesisPipeline.__init__`:
  `pipeline.py:962-1005`;
- `synthesize_structured_atmosphere`:
  `synthesis.py:90-117`;
- `prewarm` and its manifest identity:
  `prewarm.py:134-163`;
- `compile_molecular_text`:
  `source_catalog_molecular_compiler.py:527-552`;
- `compile_tio_schwenke`:
  `source_catalog_molecular_compiler.py:817-844`; and
- `compile_h2o_partridge`:
  `source_catalog_molecular_compiler.py:960-987`.

An AST signature check found no unexpected `r_grid` parameter in these engine
or compiler interfaces.

**Result: pass.**

### 3.4 CLI aliases and destination behavior

Both synthesis CLIs define:

```text
"--r-grid", "--resolution", dest="resolution"
```

at `payne_zero_synthesis/cli.py:125-132` and
`prewarm.py:243-250`. Both spellings therefore land in `args.resolution`; they
are not two independently conflict-checked CLI values.

The main synthesis CLI then preserves the target API spelling:

- archive mode calls `synthesize(..., resolution=args.resolution)`
  (`cli.py:174-183`);
- label mode calls
  `synthesize_from_labels(..., r_grid=args.resolution)`
  (`cli.py:224-242`); and
- prewarm forwards `resolution=args.resolution`
  (`prewarm.py:251-258`).

The repaired contracts say only that the two flags are aliases stored as
`resolution`; they do not incorrectly promise API-style unequal-double-
specification validation at the CLI.

**Result: pass.**

### 3.5 P0.1 contradiction search

No universal “always use public `r_grid`” rule remains in `BIBLE.md`,
`COVERAGE.md`, or the global chapter contracts.

Two non-governing Chapter 6 evidence documents still use lowercase source-like
`r_grid` while describing a `Grid`/fixture construction:

- `design/chapter06_fe_record_candidate_audit.md:183,295-296,325`; and
- `design/chapter06_synthesis_fixture_oracle_plan.md:242-245`.

The actual Chapter 6 runtime and worker use `resolution`, and other Chapter 6
evidence uses mathematical `R_grid`. These stale audit-plan spellings do not
invalidate the repaired global P0.1 contract, but they should be changed to
\(R_{\rm grid}\) when the sentence is mathematical or `resolution` when it
names the exact `Grid` argument. They must not be copied into the reader-facing
chapter.

Old `_pipeline/build_lecture*.py` artifacts are not canonical chapter sources
and use mathematical `R_grid`, not a claimed public API keyword. They are
already governed by the planned active-tree cleanup and do not contradict
P0.1.

### P0.1 verdict

**ACCEPT.** No open P0 remains in the exact intrinsic-grid global contract.
The two Chapter 6 evidence-document spelling cleanups are P1 follow-through,
not a reason to retain the original global P0.1.

## 4. P0.2 independent source audit

### 4.1 Meaning of “default” in the atmosphere lane

The repaired `COVERAGE.md:64-68` correctly defines “default” as the standard
high-level workflow, not the all-`None` field defaults of the low-level
`AtmosphereInput` dataclass. This qualification is essential:

- low-level `AtmosphereInput` declares all source paths optional
  (`payne_zero_atmosphere/config.py:11-25`);
- the high-level solve creates an `AtmosphereInput` from
  `source_line_paths()` and enables molecules
  (`payne_zero_atmosphere/cli.py:414-430`).

The high-level resolver returns:

- predicted atomic;
- observed atomic;
- high-excitation atomic;
- diatomic;
- TiO;
- water; and
- detailed-transition paths

at `source_catalogs.py:142-166`. Its docstring says H3+ too, but its executable
mapping contains no `h3plus_lines_path`.

**Result: pass.**

### 4.2 Atmosphere ordinary, diatomic, TiO, water, and H3+ behavior

The repaired two-lane matrices match the executable selector:

| Family | Pinned atmosphere behavior |
| --- | --- |
| ordinary atomic | predicted, observed, and high-excitation source arrays go through the standard selector at `line_selection.py:1056-1099` |
| diatomic | `read_diatomic_line_catalog` plus `_diatomic_log_strength_offsets`, then the common selected-record path at `line_selection.py:1101-1113` |
| TiO | standard reader plus TiO strength/population/damping overrides at `line_selection.py:1115-1134` |
| water | dedicated `read_water_line_catalog`/`select_water_line_words` path at `line_selection.py:1136-1145` |
| H3+ | selector/deposit-capable only when an existing explicit `h3plus_lines_path` is supplied, at `line_selection.py:1147-1161`; no default path is returned |

`_generate_standard_selected_lines` forwards all supplied families and fails
only when line selection is enabled but neither a preselected catalog nor any
source path exists (`runner.py:599-646`). All admitted families become one
decoded `SelectedLineCatalog`.

The selected catalog is generated or loaded and deposited at
`runner.py:719-749`. When the detailed branch is enabled, its catalog is loaded
and deposited at `runner.py:751-782`. `run_atmosphere_model` keeps both objects
outside the iteration loop, passes them into later opacity calls, and replaces
the local references with the returned objects
(`runner.py:1634-1695`). This supports the repaired Chapter 11 select-once/reuse
contract.

**Result: pass.**

### 4.3 Synthesis defaults and family behavior

The repaired matrices also match the synthesis lane:

- `synthesize`, `synthesize_from_labels`, `synthesize_structured_atmosphere`,
  and `SynthesisPipeline` all default `molecular_lines=True`;
- atomic catalog compilation/routing occurs regardless of that flag
  (`pipeline.py:1591-1620`);
- molecular compilation and invariant creation occur only when the flag is
  true (`pipeline.py:1686-1716`);
- the standard molecular builder reads manifest-ordered text bands and calls
  `compile_molecular_text` and `compile_tio_schwenke`
  (`pipeline.py:1168-1261`);
- runtime molecular deposition consumes the resulting catalog at
  `pipeline.py:1404-1427`; and
- `molecular_lines=False` is an opt-out, not the default.

The standard molecular manifest contains the text-band list and packed TiO and
water source files. The pipeline's source identity and `_compile_molecular`
calls include only the text-band NPZ and TiO array
(`pipeline.py:722-770,1178-1259`).

`compile_h2o_partridge` is a real public module-level compiler
(`source_catalog_molecular_compiler.py:960-1086`), but the only synthesis-tree
occurrences are its definition and module `__all__` record. There is no
standard pipeline call and no high-level runtime flag or injection path that
turns it on. “Compiler-only” therefore means callable data-preparation code,
not a standard or high-level opt-in synthesis opacity path.

The synthesis tree contains one H3+ mass-table row
(`molecular_lines.py:92`). It has no compiler dispatch entry, standard source
compiler call, or pipeline source wiring. The repaired documents correctly say
that metadata alone is not runtime support.

**Result: pass.**

### 4.4 Chapter 7/8/10/11 ownership

The repaired ownership split is coherent and nonredundant:

- Chapter 7 owns ordinary atomic routing and the atmosphere lane's family-
  independent selected-record representation/common deposit;
- Chapter 8 owns the atmosphere molecular readers/corrections/selectors and
  the synthesis molecular formats/compilers/deposits, including default,
  opt-in, compiler-only, and absent statuses;
- Chapter 10 owns synthesis window-invariant cache composition; and
- Chapter 11 owns the standard atmosphere composition, first-pass
  generation/load, optional detailed-catalog load, and later object reuse.

Evidence appears consistently in:

- `BIBLE.md:303-314`;
- `COVERAGE.md:64-77,122-130,159-173`; and
- `design/global_chapter_contracts.md:365-447,525-545`.

This closes the original ownership ambiguity in the three repaired global
contracts. Chapter 11 composes but does not rederive Chapters 7–8.

**Result: pass.**

### 4.5 Corrected unsupported-boundary language

The repaired global documents no longer claim that the active converted
diatomic/TiO/water selectors are unsupported.

The exact central runner guard rejects only:

1. `iterations < 1`;
2. `setup.turbulence.enabled`; and
3. zero-based `opacity_flags[13] == 1`, the HLINOP branch

at `payne_zero_atmosphere/runner.py:2003-2020`.

The stale source docstring at `runner.py:1604-1613` says raw molecular selectors
are off, but executable source selection contradicts that sentence. The
repaired contracts correctly follow behavior rather than elevating the
docstring. They now state:

- turbulent pressure and HLINOP have explicit guards;
- NLTE is outside scientific scope, without inventing a guard; and
- converted molecular selectors are active, while H3+ is explicit-path
  opt-in.

**Result: pass in the repaired global contracts.**

## 5. Repo-wide contradiction audit

### 5.1 Blocking detailed-brief contradictions

The repair candidate reports only two remaining stale statements:
`design/part5_part6_atmosphere_brief.md:1359-1360` and line 2015. The
repository-wide search found an earlier duplicate pair in the same brief that
the candidate missed:

1. **Lines 709–710**
   - “Raw molecular selectors ... fail loudly” has no matching guard and
     contradicts the active converted molecular selector.
   - “Water-line compilation ... is not silently enabled in the standard
     runtime” is lane-ambiguous inside the atmosphere Chapter 11 brief. The
     atmosphere water path is standard; only synthesis H2O is compiler-only.
2. **Lines 1359–1360**
   - “unsupported raw molecular selectors ... must fail loudly” repeats the
     false guard claim.
3. **Line 2015**
   - “the standard verified workflow [does not include] water opacity” again
     omits the lane. It is false for the standard atmosphere workflow and true
     only for synthesis H2O.

These are not harmless solely because
`design/global_chapter_contracts.md:15-34` makes the global contract govern
reader-facing order and interfaces. The same authority section says the
detailed briefs remain the full field/API/source inventories, and the
atmosphere brief itself says its exact behavior refers to the pinned commit.
Future Chapter 11, 13, or 15 construction can therefore ingest these statements
as source facts.

Before P0.2 acceptance, each occurrence must be removed or made lane-exact:

- active converted atomic/diatomic/TiO/water selection is supported;
- atmosphere H3+ is explicit-path opt-in;
- only synthesis H2O is compiler-only/unwired; and
- only the actually guarded turbulent-pressure and HLINOP branches should be
  described as rejected by `_require_supported_run_setup`.

An inline “superseded by the global matrix” annotation at each occurrence would
be minimally sufficient, but replacing the false bullets is cleaner.

### 5.2 Nonblocking status drift

`PLAN.md:832-834` still presents synthesis H2O treatment as a decision to make.
The repaired contract has made that decision: preserve compiler parity and
document omission from the standard synthesis runtime. Update the known-
questions entry to a frozen boundary.

`PASSDOWN.md:328-329` states the correct source fact, but still calls it an
ambiguity. Its live status should be updated when the repair is accepted.

These are P1 state-maintenance issues. Neither changes the source verdict.

### 5.3 Why this blocks P0.2 but not P0.1

The P0.1 residuals are non-governing Chapter 6 audit-plan notation; the actual
runtime and global contracts already use `resolution`, and the reader-facing
outline uses mathematical notation.

The P0.2 residuals are different: they make false executable-branch claims
inside the retained atmosphere source/API inventory for the very chapters that
will teach those branches. They reproduce the original cross-lane confusion
and unsupported-guard invention. Leaving them in place would make the global
repair dependent on every future author noticing and resolving a hidden
contradiction. That is not an acceptable completeness contract.

## 6. Priority findings

### P0

1. Correct or explicitly supersede
   `design/part5_part6_atmosphere_brief.md:709-710,1359-1360,2015`.
   All three locations must distinguish active atmosphere water from
   compiler-only synthesis H2O and must stop claiming an unsupported raw-
   molecular-selector guard.
2. Amend the repair candidate's verification record, which currently says a
   repository-wide search found only the latter two locations. The unreported
   lines 709–710 are materially the same P0.2 contradiction.

### P1

1. Replace source-like lowercase `r_grid` with mathematical
   \(R_{\rm grid}\) or exact `resolution` in the two Chapter 6 evidence
   documents identified in Section 3.5 before their wording reaches the
   chapter.
2. Resolve the now-stale H2O “Known Question” in `PLAN.md:832-834`.
3. Update `PASSDOWN.md` after the bounded repairs are accepted.

### P2

1. Add a lightweight contract test or audit search that rejects unqualified
   statements combining “water/H2O,” “standard runtime,” and no product lane.
2. Add a similar guard against saying active converted molecular selectors
   fail loudly unless an exact source guard is cited.
3. Keep the pinned source docstring discrepancy documented as upstream
   evidence; do not edit the read-only Payne Zero checkout.

## 7. Final verdict

**P0.1 — ACCEPT** at:

- `BIBLE.md`
  `65387d009b732446252b5392afcbaa7a12fcb2c6db6083d9cf8eed0b6b1b36ac`;
- `COVERAGE.md`
  `dbca7a801db525e467f5f0a6162591d0e057eeaa5056d1e690e8e137b1138b84`;
  and
- `design/global_chapter_contracts.md`
  `1ef69f8434af3f19a1545546aa1dad03b988316d19f686f44afab5d2bfdeafc2`.

The interface names, aliases, CLI destination, default/conflict behavior, and
intrinsic-grid/instrument distinction match pinned commit
`9c44001feae40b85146630499e6f8a5fed42e5af`.

**P0.2 — REJECT** at the current repository state, despite the correctness of
the three repaired global matrices. The retained atmosphere field/API/source
brief still contradicts the active molecular runtime at lines
709–710, 1359–1360, and 2015. Those drafting hazards must be repaired or
explicitly superseded before P0.2 is accepted.

**P0.3 remains open and was not assessed.**

---

# Second independent re-audit: P0.2 follow-up

## A. Scope and supersession boundary

This delimited addendum preserves the first independent rejection above as the
decision for its exact evidence. It re-audits only the subsequently authorized
P0.2/P1 cleanup. It supersedes the earlier **P0.2 — REJECT** only if the hashes
below match; it does not alter the accepted P0.1 result and does not assess
P0.3.

The pinned Payne Zero checkout remains read-only at:

```text
9c44001feae40b85146630499e6f8a5fed42e5af
```

Exact follow-up inputs:

| Evidence | SHA-256 |
| --- | --- |
| updated repair candidate | `79c03b2d2b54ac56ca7e56605e023b795fa8ad55c5853750f6f1c92142a7f385` |
| repaired atmosphere brief | `647921155f926aab5f1faa7a4e6fe02b676534a92a304548778238fd356f7de6` |
| repaired Chapter 6 Fe-record audit | `390dea17fe65f71a1cbe377baa0ad5bf3d94901093f0373149660885851df7ef` |
| repaired Chapter 6 synthesis-oracle plan | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| repaired `PLAN.md` | `0cac1d62644976db2f3e89ecfc848182c8e53e4ee2ad1efe82bcb92e4dd80d7c` |
| repaired `PASSDOWN.md` | `2ea4936430923614914e786f16c8a25c82a8cca5c7ade5fd1dba21086739d7f4` |
| new active-prose test | `3a001cc080fd71bdd036a3f17c73627e1a35c3a00d468155a5d98410577e7dd3` |
| first independent audit before this addendum | `4fadca3f0946fc7ea844db72d1f483e62d24e793f01342f975fdff06e097d122` |

All claimed input hashes matched byte for byte.

## B. The three blocking brief sites

### B.1 Former lines 709–710: pass

The Chapter 11 failure-boundary section now says:

- the standard atmosphere route supports converted atomic, diatomic, TiO, and
  water selection;
- atmosphere H3+ is explicit-path opt-in because `source_line_paths()` supplies
  no default H3+ file;
- `_require_supported_run_setup` has no blanket raw-molecular-selector guard;
- its exact unsupported guards are turbulent pressure and HLINOP; and
- compiler-only synthesis H2O does not disable atmosphere water.

The repaired text is at
`design/part5_part6_atmosphere_brief.md:709-715`. It matches the pinned
`source_line_paths`, `generate_selected_lines`, and runner guard behavior
audited in Sections 4.1–4.5 above.

### B.2 Former lines 1359–1360: pass

The Chapter 13 failure section now keeps NLTE outside the scientific scope
without inventing a runtime guard. It explicitly calls converted
atomic/diatomic/TiO/water atmosphere selection active, makes H3+ an
explicit-path opt-in, and limits “failing loudly” to the exact turbulent-
pressure and HLINOP guards
(`part5_part6_atmosphere_brief.md:1361-1370`).

This closes the former false statement that unsupported raw molecular
selectors must fail loudly.

### B.3 Former line 2015: pass

The Chapter 15 boundary now states both lanes in one paragraph:

- the standard atmosphere workflow includes converted water selection and
  opacity deposition;
- atmosphere H3+ is explicit-path opt-in; and
- the separate synthesis H2O compiler is verified as a compiler but omitted
  from standard synthesis runtime
  (`part5_part6_atmosphere_brief.md:2024-2028`).

The sentence can no longer be read as excluding water opacity from the
atmosphere lane.

### B.4 Contradiction search: pass

A focused search over current active contracts, planning prose, the detailed
briefs, and canonical chapter sources found none of the three former claims:

```text
Raw molecular selectors and unsupported line branches fail loudly.
unsupported raw molecular selectors ... must fail loudly
Available water-line compilation ... standard verified workflow ... water opacity
```

Broader order-independent searches also found no active prose claiming that
raw molecular selectors fail/reject, or that a lane-unqualified standard
runtime omits water/H2O. Immutable rejection/audit records were not rewritten
or counted as current policy.

**P0.2 prose result: pass.**

## C. P1 follow-up

### C.1 Chapter 6 intrinsic-grid spelling: pass

The Fe-record audit now distinguishes the exact source boundary from
mathematics:

- exact `Grid.resolution = 300000` at
  `design/chapter06_fe_record_candidate_audit.md:180-184`;
- public archive API `resolution = 20000` at lines 293–297; and
- mathematical \(R_{\rm grid}\) for the two-density comparison at line 325.

The synthesis-oracle plan now calls the exact constructor field
`Grid.resolution=20000`
(`design/chapter06_synthesis_fixture_oracle_plan.md:240-245`) and already uses
`resolution=300000.0` in its constructor example.

Neither file retains source-like `` `r_grid=...` `` or
`` `r_grid = ...` `` wording. This closes the P1 notation drift identified in
the first audit without changing the accepted P0.1 matrix.

### C.2 Frozen H2O boundary in PLAN: pass

`PLAN.md:841-845` now marks the issue **Resolved boundary**:

- Chapter 8 verifies compiler parity;
- the complete textbook preserves the pinned standard synthesis runtime's H2O
  omission; and
- that synthesis-only boundary must not be confused with standard atmosphere
  water selection/deposition.

The substantive status is honest and matches the global matrices. The section
heading still says “Known Questions to Resolve,” but the explicit resolved
label prevents the item from being mistaken for an open scientific decision.
Moving resolved items to a separate subsection would be editorial P2 cleanup,
not a coverage defect.

### C.3 Live PASSDOWN status: pass

`PASSDOWN.md:338-343` likewise marks H2O as a resolved boundary and states the
two lanes exactly. It no longer presents the feature status as an unresolved
ambiguity.

### C.4 Candidate repair record: pass

The updated candidate now acknowledges that its initial search missed the
earlier lines 709–710, records all three corrected sites, preserves P0.3 as
open, and gives the exact follow-up file hashes
(`design/global_interface_molecular_repair_candidate.md:142-185`).

The source-status and repair-history claims are now honest.

## D. Focused verification

The requested focused suite was run exactly as:

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_global_molecular_lane_language.py \
  tests/test_symbol_coverage.py
```

Result:

```text
6 passed in 0.06s
```

The new test also passes:

```text
/Users/ysting/anaconda3/bin/ruff check \
  tests/test_global_molecular_lane_language.py

/Users/ysting/anaconda3/bin/ruff format --check \
  tests/test_global_molecular_lane_language.py
```

with `All checks passed!` and `1 file already formatted`.

`git diff --check` is clean for every follow-up input.

## E. Adversarial audit of the new language guard

### E.1 What the test does well

`tests/test_global_molecular_lane_language.py` intentionally scans an allowlist
of active planning prose:

- `BIBLE.md`;
- `PLAN.md`;
- `PASSDOWN.md`; and
- `design/part5_part6_atmosphere_brief.md`.

It ignores code fences, tables, and immutable audit history rather than trying
to rewrite records of earlier rejected states. It catches the two former
raw-molecular-selector formulations because they place `fail` after the
selector phrase. It also pins positive atmosphere facts in the drafting brief:
standard converted water and explicit-path H3+.

That scope is appropriate for a small active-prose regression guard.

### E.2 The negative-water regex is not yet robust

The test named
`test_negative_standard_water_runtime_claims_name_synthesis_lane` recognizes
only this narrow negative vocabulary:

```text
omit | omission | unwired | compiler-only | rather than
```

and recognizes only:

```text
standard runtime | standard pipeline | standard workflow
```

with optional `synthesis`, but not an intervening word such as `verified`.

An in-memory adversarial probe of the test—without changing any repository
file—gave:

| Synthetic active sentence | Guard result |
| --- | --- |
| former exact wording: “standard verified workflow includes water opacity” under a negative “does not imply” clause | **missed** |
| “The standard workflow does not include water opacity.” | **missed** |
| “The standard runtime is without H2O opacity.” | **missed** |
| “The standard pipeline excludes water opacity.” | **missed** |
| “The standard runtime omits water opacity.” | **missed** |

The last miss occurs because the regex matches `omit` but not `omits`. The
positive exact-string assertions do not solve this: a stale negative sentence
could be reintroduced elsewhere while the required positive sentence remains.

The candidate's claim at
`global_interface_molecular_repair_candidate.md:189-192` that the guard
generally requires negative standard-water claims to name synthesis is
therefore too broad.

This does **not** reopen P0.2: the current prose has been independently searched
and is correct. It is an open P1 verification-quality issue. Strengthen the
negative vocabulary to include at least:

```text
omit/omits/omitted
does not include/invoke/use
without
exclude/excludes/excluded
```

and allow qualifiers such as `verified` between `standard` and
`workflow/runtime/pipeline`. The unit test should include the three former
sentences as explicit synthetic regression cases, so future regex changes are
tested against the actual failure history.

## F. Priority and final decision

### P0

No P0.2 defect remains at the exact follow-up hashes. All three contradictory
brief sites are lane-exact, and current active prose agrees with the pinned
source and the repaired global matrices.

### P1

Strengthen `tests/test_global_molecular_lane_language.py` as described in
Section E.2, then narrow the candidate's guard claim or make the test satisfy
it. This is evidence hardening, not a reason to undo the prose repair.

### P2

1. Move the resolved H2O item out of PLAN's “Known Questions to Resolve”
   subsection when that section is next cleaned.
2. If the active-prose allowlist expands, add new authoritative drafting
   documents deliberately; do not scan immutable audit records and demand that
   historical rejections disappear.

## G. Second verdict

**P0.2 — ACCEPT** at:

- `design/part5_part6_atmosphere_brief.md`
  `647921155f926aab5f1faa7a4e6fe02b676534a92a304548778238fd356f7de6`;
- `design/chapter06_fe_record_candidate_audit.md`
  `390dea17fe65f71a1cbe377baa0ad5bf3d94901093f0373149660885851df7ef`;
- `design/chapter06_synthesis_fixture_oracle_plan.md`
  `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856`;
- `PLAN.md`
  `0cac1d62644976db2f3e89ecfc848182c8e53e4ee2ad1efe82bcb92e4dd80d7c`;
- `PASSDOWN.md`
  `2ea4936430923614914e786f16c8a25c82a8cca5c7ade5fd1dba21086739d7f4`;
  and
- updated repair candidate
  `79c03b2d2b54ac56ca7e56605e023b795fa8ad55c5853750f6f1c92142a7f385`.

The original P0.2 rejection above remains the correct record for its earlier
hashes. This addendum supersedes only that verdict for the exact repaired bytes
listed here.

The new test at
`3a001cc080fd71bdd036a3f17c73627e1a35c3a00d468155a5d98410577e7dd3`
passes but retains the P1 adversarial gap documented above.

**P0.3 remains open and was not assessed.**

---

# Final narrow re-audit: P1 language-guard hardening

## H. Scope and exact inputs

This final addendum audits only the P1 robustness defect in the active-prose
language guard. It does not reopen the P0.2 prose acceptance in Section G and
does not assess P0.3.

Exact inputs:

| Evidence | SHA-256 |
| --- | --- |
| hardened `tests/test_global_molecular_lane_language.py` | `d0a8a42e791435305131b79b5be8f3594d2734f20ff0566afe80cf8dc0b5176a` |
| updated repair candidate | `59f36a4addf9890b42022eb0a290feb3e79f1281b3fbe6b0698a4adb068443db` |
| independent audit before this addendum | `3e6fa493dcdbb9a144d6d0ce9981ceae199843e73a5909235219cb9c6389eeea` |

Both claimed repair-input hashes matched.

## I. Focused suite and requested grammar coverage

The focused test was run as:

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_global_molecular_lane_language.py
```

Result:

```text
26 passed in 0.02s
```

Ruff check, Ruff format check, and `git diff --check` also pass.

The exported predicate
`_is_unqualified_negative_standard_water_claim` was independently called with
each requested negative grammar. It correctly rejected all of these when no
synthesis lane was present:

- the exact former “does not imply the standard verified workflow includes
  water opacity” sentence;
- `omit`, `omits`, `omitted`, and `omission`;
- `does not include`, `does not invoke`, and `does not use`;
- `not included`, `not currently enabled`, and `not silently enabled`;
- `excludes`/`excluded`;
- `without`;
- `absent`;
- `unwired`;
- `compiler-only`;
- `rather than`; and
- both plain `H2O` and Markdown `H\(_2\)O`.

It also accepted the requested nonoffending forms:

- a negative claim whose `standard ... synthesis ... runtime/pipeline/workflow`
  phrase explicitly names the synthesis lane;
- positive standard-atmosphere water statements;
- a compiler-only statement that makes no standard-runtime claim; and
- an unrelated “does not imply” statement about cache reuse.

The new parameterized cases therefore close the earlier vocabulary and
`verified`-qualifier gaps.

## J. Immutable-audit exclusion: pass

`ACTIVE_PROSE_FILES` is an explicit allowlist:

```text
BIBLE.md
PLAN.md
PASSDOWN.md
design/part5_part6_atmosphere_brief.md
```

Neither the global rejection, either independent re-audit, nor the repair
candidate is scanned as current planning prose. This is intentional and
correct: an immutable audit must retain the rejected historical sentence so
the reason for the repair remains reviewable.

The allowlist still contains the active Bible, plan, live passdown, and the
specific detailed drafting brief that previously carried the contradiction.
It therefore excludes historical evidence without excluding the active prose
this narrow guard was designed to protect.

## K. Remaining adversarial defect: synthesis is not required in the lane phrase

The predicate ends with:

```python
and "synthesis" not in text.lower()
```

This asks only whether the word `synthesis` appears anywhere in the whole
wrapped prose block. It does not require the matched standard-runtime phrase—or
even the same claim—to identify the synthesis lane.

An in-memory adversarial probe, without editing an input file, produced:

| Prose block | Predicate result | Required result |
| --- | --- | --- |
| `Synthesis is discussed later. The standard atmosphere runtime omits water opacity.` | `False` | `True` — reject |
| `The synthesis H2O compiler exists. The standard atmosphere workflow does not include water opacity.` | `False` | `True` — reject |
| `The standard atmosphere pipeline excludes H2O. Synthesis uses a different catalog.` | `False` | `True` — reject |
| `The standard verified synthesis workflow omits H2O opacity.` | `False` | `False` — accept |

The first three are exactly the cross-lane error the guard is intended to
prevent. A nearby correct use of the word `synthesis` masks the false
atmosphere claim.

The candidate's statement that the hardened predicate requires an explicit
synthesis lane is therefore still too strong. It requires only an unscoped
synthesis token.

A robust repair should evaluate the `STANDARD_RUNTIME` match itself. At
minimum:

1. capture the matched `standard ... runtime/pipeline/workflow` phrase;
2. treat `standard ... atmosphere ...` negative-water phrases as offenders
   regardless of another sentence mentioning synthesis;
3. accept a negative standard-water claim only when `synthesis` qualifies that
   same runtime/pipeline/workflow phrase; and
4. add the three masking examples above to the parameterized rejection suite.

Requiring the source phrase `standard synthesis runtime` (or its qualified
equivalent such as `standard verified synthesis workflow`) is clearer and
safer than attempting paragraph-wide lane inference.

## L. Final P1 decision

**P1 language-guard hardening — REJECT** at:

- `tests/test_global_molecular_lane_language.py`
  `d0a8a42e791435305131b79b5be8f3594d2734f20ff0566afe80cf8dc0b5176a`;
  and
- `design/global_interface_molecular_repair_candidate.md`
  `59f36a4addf9890b42022eb0a290feb3e79f1281b3fbe6b0698a4adb068443db`.

The requested negative vocabulary, positive cases, formatting, and immutable-
audit scope all pass. The remaining blocker is narrower: the guard does not
actually require synthesis to qualify the negative standard-runtime claim.

**P0.2 remains accepted for the repaired prose at the hashes in Section G.**

**P0.3 remains open and was not assessed.**

---

# Final P1 masking-defect re-audit

## M. Scope and immutable inputs

This addendum re-audits only the repaired P1 masking defect from Section L.
It does not reopen the accepted P0.2 prose repair and does not assess P0.3.

The supplied immutable inputs match their expected SHA-256 identities:

- `tests/test_global_molecular_lane_language.py`:
  `df6ec7e0468b77deb9bc083c2fdd9172ec41985d2ce6faaec5ab034995f25050`;
- `design/global_interface_molecular_repair_candidate.md`:
  `bf6017def45d0576afd4fd2ceab0329b2324fe8d23b84b06a9f665e4e2e4f197`.

Neither input was edited during this audit.

## N. Focused verification

The committed suite and static checks pass:

- `PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q
  tests/test_global_molecular_lane_language.py`: `39 passed`;
- Ruff check: passed;
- Ruff format check: one file already formatted;
- `git diff --check` for both immutable inputs: clean.

The active-prose allowlist still contains only `BIBLE.md`, `PLAN.md`,
`PASSDOWN.md`, and `design/part5_part6_atmosphere_brief.md`. Immutable
candidate and audit records remain intentionally excluded.

The new tests correctly reject the supplied preceding/following sentence,
semicolon, `but`, `while`, and `and` examples. They also correctly accept the
supplied synthesis-qualified negatives and positive atmosphere facts.

## O. Remaining masking counterexample

The new predicate scopes `synthesis` correctly to each
`standard ... runtime`/`pipeline`/`workflow` match. However, it first applies
`CLAIM_SEPARATOR` unconditionally at every `but`, `while`, `whereas`, and
selected `and`. Those tokens are not always boundaries between complete
claims: they can introduce a parenthetical clause between a claim's subject
and predicate.

Independent in-memory probes, without editing an input, found:

| Claim | Produced | Required |
| --- | --- | --- |
| `The standard atmosphere runtime, while synthesis is discussed elsewhere, omits water opacity.` | accept | reject |
| `The standard atmosphere runtime—but synthesis is a separate topic—omits water opacity.` | accept | reject |
| `The standard atmosphere runtime—and synthesis is discussed elsewhere—omits water opacity.` | accept | reject |

For the first example, `_claim_segments` returns:

1. `The standard atmosphere runtime`
2. `synthesis is discussed elsewhere, omits water opacity`

The first segment has the standard atmosphere boundary but no water/negative
predicate. The second has synthesis, water, and `omits` but no standard
boundary. Consequently
`_is_unqualified_negative_standard_water_claim(...)` returns `False`, and the
false atmosphere-water omission is accepted.

This is the same P1 masking defect in an infix-clause position. The unrelated
synthesis text does not appear inside the standard-runtime phrase, but
segmentation disconnects that phrase from its predicate before the phrase-level
check runs.

A repair should add these infix cases to the rejection suite and avoid treating
an intervening parenthetical conjunction as a complete-claim boundary. The
phrase-local synthesis check is sound; the remaining issue is preserving the
main subject-predicate claim around an inserted clause.

## P. Final P1 verdict

**P1 language-guard hardening — REJECT** at the two hashes in Section M.

The committed 39 cases and ordinary clause-order variants pass, but the
predicate does not yet prevent an unrelated infix `while`/`but`/`and`
synthesis clause from masking a false atmosphere-water omission.

**P0.2 remains accepted.**

**P0.3 is being audited separately and was not assessed here.**
