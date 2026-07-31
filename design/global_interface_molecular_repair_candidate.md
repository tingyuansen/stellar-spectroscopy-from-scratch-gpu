# Global interface and molecular-contract repair candidate

## Scope and disposition

This candidate repairs only P0.1 and P0.2 from
`design/global_coverage_zoomout_after_ch06_evidence.md`, whose required input
SHA-256 is
`e4f08434604ccd308264e40ea0e934811c0d26cfbb3f7a6c15fca5c51cec8baf`.
It changes global contracts, not reader-facing chapters, generated coverage
inventories, source data, or the pinned implementation.

**Disposition: repair candidate; independent acceptance is still required.**

The read-only implementation oracle was independently checked at exact commit
`9c44001feae40b85146630499e6f8a5fed42e5af`; the checkout `HEAD` matched that
identity during this repair.

## P0.1 — exact intrinsic-grid interface spellings

The repaired contract separates the mathematical quantity from exact source
names:

| Boundary | Verified source behavior |
| --- | --- |
| mathematics | \(R_{\rm grid}=\lambda/\Delta\lambda\) is intrinsic adjacent-sample density, not instrumental resolving power |
| label API | `payne_zero_synthesis/api.py:221-237` defines `_resolved_r_grid`; `api.py:732-760` makes `r_grid` canonical and `resolution` an alias; unequal double specification raises |
| archive/in-memory API | `payne_zero_synthesis/api.py:442-452` exposes only `resolution` |
| grid and catalog records | `payne_zero_synthesis/atomic_lines.py:168-182` names `Grid.resolution`; `atomic_lines.py:451-488` serializes and reloads `resolution` |
| pipeline and window/cache helpers | `payne_zero_synthesis/pipeline.py:704-783`, `800-890`, and `962-1005` use `resolution` in the persistent molecular identity, invariant key, grid helper, and `SynthesisPipeline` constructor |
| engine helper | `payne_zero_synthesis/synthesis.py:90-117` exposes and forwards `resolution` |
| molecular compilers | `payne_zero_synthesis/source_catalog_molecular_compiler.py:527-552`, `817-844`, and `960-987` expose `resolution` for text, TiO, and H2O compilation |
| CLI | `payne_zero_synthesis/cli.py:123-132` and `prewarm.py:239-257` accept `--r-grid`/`--resolution` aliases with `dest="resolution"` |

The new matrices in `BIBLE.md`, `COVERAGE.md`, and
`design/global_chapter_contracts.md` preserve these spellings. They retain the
physical warning that an instrument/LSF operation is separate.

## P0.2 — exact two-lane family status

### Atmosphere evidence

- `payne_zero_atmosphere/source_catalogs.py:142-166`,
  `source_line_paths`, returns predicted, observed, high-excitation, diatomic,
  TiO, water, and detailed-transition paths. Despite its broader docstring, the
  returned mapping contains no `h3plus_lines_path`.
- `payne_zero_atmosphere/cli.py:414-430` constructs the standard
  `AtmosphereInput` from `source_line_paths()` and enables molecules.
- `payne_zero_atmosphere/config.py:11-25` exposes optional fields for all
  source families, including `h3plus_lines_path`.
- `payne_zero_atmosphere/runner.py:599-646` forwards every supplied ordinary
  atomic and molecular source path to `generate_selected_lines`.
- `payne_zero_atmosphere/line_selection.py:1012-1171` implements the active
  converted-catalog selectors: ordinary atomic at 1056-1099, diatomic at
  1101-1113, TiO at 1115-1134, water at 1136-1145, and optional H3+ at
  1147-1161. All return one decoded `SelectedLineCatalog`.
- `payne_zero_atmosphere/runner.py:719-800` selects or loads that common
  catalog and sends it to `accumulate_selected_line_opacity`.
- `payne_zero_atmosphere/runner.py:1634-1695` keeps both the selected and
  detailed-transition catalogs outside the iteration loop, passes them into
  each opacity preparation, and carries the returned objects into later
  iterations.
- `_require_supported_run_setup` at
  `payne_zero_atmosphere/runner.py:2003-2020` contains guards for iteration
  count, turbulent pressure, and HLINOP. It contains no raw-molecular-selector
  guard. Therefore the former global statement that raw molecular selectors
  are unsupported had no exact guard and has been removed. The stale
  `run_atmosphere_model` docstring at `runner.py:1604-1613` does not override
  the executable paths above.

### Synthesis evidence

- `payne_zero_synthesis/api.py:442-452` and
  `synthesis.py:90-117` default `molecular_lines=True`; ordinary atomic opacity
  is independent of that flag.
- `payne_zero_synthesis/pipeline.py:1591-1620` always loads/routes the atomic
  catalog. `pipeline.py:1686-1716` builds molecular invariants only when the
  flag is true.
- `payne_zero_synthesis/pipeline.py:1168-1261`, `_compile_molecular`, compiles
  only manifest-ordered text bands plus TiO and concatenates those arrays.
- The pinned
  `source_data_files/source_catalogs/molecules/manifest.json` lists text
  sources and packed TiO and water files, but pipeline cache/source identity at
  `pipeline.py:704-770` names only `molecular_band_lines.npz` and
  `titanium_oxide_lines.npy`.
- `compile_h2o_partridge` exists at
  `payne_zero_synthesis/source_catalog_molecular_compiler.py:960-1086`; no
  call from the standard pipeline was found. H2O is therefore compiler-only at
  this boundary.
- `payne_zero_synthesis/molecular_lines.py:44-111` has a generic H3+ mass-table
  row, but neither the compiler dispatch tables at
  `source_catalog_molecular_compiler.py:180-299` nor the standard pipeline
  supplies H3+ lines. A metadata row is not public runtime wiring.
- Runtime molecular deposition occurs at
  `payne_zero_synthesis/pipeline.py:1404-1427` for the catalog actually built
  above.

## Ownership frozen by the repair

- Chapter 7 owns ordinary atomic routing plus the atmosphere lane's
  family-independent selected-record and deposit machinery.
- Chapter 8 owns atmosphere molecular family readers/selectors and synthesis
  molecular formats/compilers/deposits, including the default, opt-in,
  compiler-only, and absent boundaries.
- Chapter 11 composes the standard atmosphere family set and owns first-pass
  selected-catalog creation/load, optional detailed-catalog load, and
  later-iteration object reuse. It does not rederive Chapters 7–8.
- Chapter 10 continues to own synthesis `WindowInvariants` cache composition;
  this does not move molecular family semantics out of Chapter 8.

## Files changed by the initial P0 repair

- `BIBLE.md`
- `COVERAGE.md`
- `design/global_chapter_contracts.md`
- this candidate report

At that stage, no user-facing chapter, generated inventory/ledger, data file,
test, plan, passdown, paper, or pinned Payne Zero source was changed. The
later, separately scoped P1 guard work changes only the test and this report,
as recorded below.

## Verification record

Contract hashes after this repair:

| File | SHA-256 |
| --- | --- |
| `BIBLE.md` | `65387d009b732446252b5392afcbaa7a12fcb2c6db6083d9cf8eed0b6b1b36ac` |
| `COVERAGE.md` | `dbca7a801db525e467f5f0a6162591d0e057eeaa5056d1e690e8e137b1138b84` |
| `design/global_chapter_contracts.md` | `1ef69f8434af3f19a1545546aa1dad03b988316d19f686f44afab5d2bfdeafc2` |

Verification performed:

- `git diff --check -- BIBLE.md COVERAGE.md
  design/global_chapter_contracts.md
  design/global_interface_molecular_repair_candidate.md`: clean.
- `PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q
  tests/test_symbol_coverage.py`: `3 passed`.
- A search of the three repaired global contracts found no surviving universal
  `r_grid` spelling rule.
- A search of the same contracts found no claim that active molecular
  selectors are unsupported; the sole matching phrase is the new prohibition
  against making that claim.
- The initial candidate's repository-wide search report was incomplete: it
  identified stale statements near
  `design/part5_part6_atmosphere_brief.md:1359-1360` and line 2015 but missed
  the materially equivalent pair formerly at lines 709-710. The independent
  re-audit correctly rejected P0.2 on all three sites.
- The follow-up repair was explicitly authorized to edit that brief. All three
  sites now say that converted atomic/diatomic/TiO/water atmosphere selection
  is active, atmosphere H3+ is explicit-path opt-in, only synthesis H2O is
  compiler-only/unwired, and only the exact turbulent-pressure and HLINOP
  guards fail loudly.
- A repaired repository-wide contradiction search and the follow-up hashes are
  recorded in the addendum below.

The candidate report's own SHA-256 is necessarily reported outside this file
after its final byte is written.

## Follow-up repair after independent P0.2 re-audit

The first independent re-audit accepted P0.1 but narrowly rejected P0.2 because
the retained atmosphere brief still contradicted the correct global matrices.
This follow-up changes no architecture, global matrix, accepted chapter, pinned
source, or independent audit.

It repairs:

- all three contradiction sites in
  `design/part5_part6_atmosphere_brief.md`;
- source-like lowercase `r_grid` drift in two Chapter 6 evidence documents;
- the stale H2O decision in `PLAN.md`; and
- the matching live-status ambiguity in `PASSDOWN.md`.

P0.3 remains open and is not claimed by this candidate. This follow-up remains
subject to a new independent re-audit.

### Follow-up identities

| File | SHA-256 |
| --- | --- |
| `design/part5_part6_atmosphere_brief.md` | `647921155f926aab5f1faa7a4e6fe02b676534a92a304548778238fd356f7de6` |
| `design/chapter06_fe_record_candidate_audit.md` | `390dea17fe65f71a1cbe377baa0ad5bf3d94901093f0373149660885851df7ef` |
| `design/chapter06_synthesis_fixture_oracle_plan.md` | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| `PLAN.md` | `0cac1d62644976db2f3e89ecfc848182c8e53e4ee2ad1efe82bcb92e4dd80d7c` |
| `PASSDOWN.md` | `2ea4936430923614914e786f16c8a25c82a8cca5c7ade5fd1dba21086739d7f4` |
| `tests/test_global_molecular_lane_language.py` | `88e7cbffa3c89e189fb209492c06233ff6e814c3c0838e11f08bde09d6776ca7` |

### Follow-up verification

- The replacement contract guard scans the exact six active authorities,
  excludes immutable audit history, pins the accepted two-lane rows and
  handoffs, and rejects only a closed list of previously observed false
  formulations.
- `PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q
  tests/test_global_molecular_lane_language.py`: `24 passed`.
- Ruff check and format check pass for the new test.
- The deterministic scan of all six active authorities finds none of the
  closed historical false phrases or the bounded raw-selector failure/rejection
  regex.
- The two Chapter 6 evidence files contain no remaining source-like
  `` `r_grid=...` `` or `` `r_grid = ...` `` spelling.
- `PLAN.md` and `PASSDOWN.md` contain no remaining H2O decision/ambiguity
  language.
- `git diff --check` is clean for every follow-up file.

As above, this report's final SHA-256 is reported externally after its last byte
is written.

### P1 deterministic contract-anchor guard

The independent audit correctly rejected the earlier English predicate. It
split on periods inside parentheticals, missed plural boundary nouns, omitted
active authority files, skipped tables, and therefore claimed substantially
more semantic coverage than it provided. That predicate and its adversarial
English suite have been deleted rather than expanded.

The replacement is intentionally smaller and explicit:

- its closed active-authority set is exactly `BIBLE.md`, `PLAN.md`,
  `PASSDOWN.md`, `COVERAGE.md`, `design/global_chapter_contracts.md`, and
  `design/part5_part6_atmosphere_brief.md`;
- it asserts fifteen literal accepted anchors, including exact matrix rows and
  exact prose handoffs;
- those anchors freeze standard atmosphere water selection/deposition,
  synthesis H2O compiler-only/unwired status, atmosphere H3+ explicit-path
  opt-in status, and active converted selectors not being unsupported
  branches;
- it rejects six exact historical false water-lane formulations after only
  whitespace/case normalization, plus one bounded regex for raw molecular
  selectors being described as failing or rejected; and
- direct in-memory mutation tests alter every required anchor and inject every
  forbidden phrase/regex sample, proving that each named gate fails without
  changing repository inputs.

The two relevant immutable independent audits are named only in an excluded
history tuple and are never scanned as current policy. The guard does not claim
to understand arbitrary English, exhaust every possible paraphrase, or replace
editorial review. Its responsibility is the exact accepted contract and the
closed set of contradictions already observed during this repair.

Focused verification at the test identity recorded above is:

```text
PYTHONPATH=src:. /Users/ysting/anaconda3/bin/python -m pytest -q \
  tests/test_global_molecular_lane_language.py
24 passed

/Users/ysting/anaconda3/bin/ruff check \
  tests/test_global_molecular_lane_language.py
All checks passed!

/Users/ysting/anaconda3/bin/ruff format --check \
  tests/test_global_molecular_lane_language.py
1 file already formatted
```

`git diff --no-index --check /dev/null` is clean for each untracked candidate
file. This P1 candidate changes only those two files. It makes no claim about
P0.3 and remains subject to independent acceptance.
