# Independent audit of the global symbol semantic repair

Status: **independent audit; candidate not modified**  
Decision for P0.3: **REJECT**  
Pinned Payne Zero source:
`9c44001feae40b85146630499e6f8a5fed42e5af`

## 1. Scope and decision

I audited, but did not edit, the following candidate:

| Object | Audited SHA-256 |
| --- | --- |
| semantic override builder | `ab7ad32d5efd761296d7d8cbd164728df3c95a77d1accb035c2a7768a95569fb` |
| focused semantic tests | `f46880c75892581ae1bba1da66fed0b510b9e1ffe1db613b0dda77e9157cc227` |
| canonical semantic ledger | `c53a22d4b4f54d911e3e6e3ed6781d975e82f5ea13f08248aa820b2a5e110f27` |
| candidate report | `50a751173f2151d41653859121a4b41320772644edd8f80a1a2d8fe59c31c549` |

The candidate is a substantial improvement over the earlier module-only
ledger. Its current JSON is deterministic, the explicit assignments named by
the global zoom-out are mostly source-faithful, and its reviewed-module
surface checks do reject missing, added, or reclassified public records.

It does **not**, however, close P0.3. There are three blocking defects:

1. the reviewed-module snapshots bind only `(qualified_name, kind)`, not the
   35 reviewed source files or their public signatures;
2. the claimed 552-record homogeneous remainder contains several concrete
   compatibility, diagnostic, optional, parity-only, and unsupported
   branches; and
3. several explicit `compatibility-only` dispositions describe current
   standard cache or initializer loaders rather than compatibility paths.

The first defect lets source semantics change while the canonical ledger
remains byte-identical. The second directly violates the re-audit condition
that no branch-sensitive public symbol may fall back to `module_default`.
The third means an explicit review is not yet necessarily a correct semantic
review.

This decision is only for the P0.3 symbol-semantic gate. It is not a rejection
of the fifteen-chapter architecture or of the separate interface and
molecular-family repairs.

## 2. Frozen evidence

The surrounding contracts used here were:

| Evidence | SHA-256 |
| --- | --- |
| triggering global zoom-out | `e4f08434604ccd308264e40ea0e934811c0d26cfbb3f7a6c15fca5c51cec8baf` |
| raw public-source inventory | `010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf` |
| `BIBLE.md` | `65387d009b732446252b5392afcbaa7a12fcb2c6db6083d9cf8eed0b6b1b36ac` |
| `PLAN.md` | `c0d0f2b705e1f68b94d178d8e212d2a6a9e5503d8a7868d365eacda40425d92c` |
| `COVERAGE.md` | `dbca7a801db525e467f5f0a6162591d0e057eeaa5056d1e690e8e137b1138b84` |
| global chapter contracts | `1ef69f8434af3f19a1545546aa1dad03b988316d19f686f44afab5d2bfdeafc2` |
| Chapter 6 causal outline | `1b66df5d548f2854f83289fcf9de5109058f1482a7b64aadaff3505d1f57e019` |
| Chapter 6 exact-source contract | `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1` |

The accepted Chapter 2–5 records consulted here have these identities:

| Acceptance record | SHA-256 |
| --- | --- |
| Chapter 2 | `3c3ed37f7f848b55238a0b19e0c3964fb64ed7deac1a15629861750fa4bc5bdb` |
| Chapter 3 | `785693d5c20631e1ca0609e8db8739f9d18eae05e430436c809e5121fe363950` |
| Chapter 4 | `ef4a1606ef8017dee706f1046444920499b4bc4b7d2623be93c450098f670fc0` |
| Chapter 5 | `d94b81b561ce97d52a0f1d129da8d2acb91a8f89841bf3bc235e63b66f266d08` |

The Payne Zero checkout reported the exact pinned commit. I rebuilt the entire
58-module raw AST inventory from the read-only checkout. The rebuilt bytes
were identical to `audit/paynezero_symbols.json`, including all per-module
source hashes.

For an additional compact check, the sorted
`module NUL source_sha256 LF` manifest has these digests:

| Source manifest | Modules | SHA-256 |
| --- | ---: | --- |
| 35 reviewed modules | 35 | `a6278b768a3f57d2a3435861d965861bbcf86a700d894b321485d570c9087390` |
| complete package inventory | 58 | `c2462222c5f041d71f52167e5576a7ccec3c07b593bf3842c4dcc99f1ecd8666` |

Every one of the 35 reviewed-module source hashes in the raw inventory matched
the corresponding pinned Python file. That fact is externally true at these
audited bytes; P0.1 below explains why the semantic builder does not enforce
it.

## 3. Mechanical checks that pass

### 3.1 Exact rebuild and counts

A clean run of:

```text
python scripts/build_symbol_coverage.py \
  --inventory audit/paynezero_symbols.json \
  --ledger COVERAGE.md \
  --output <temporary>/paynezero_symbol_coverage.json
```

reported 58 modules, 1,501 records, and 949 reviewed overrides. The temporary
JSON was byte-identical to the checked candidate and had SHA-256
`c53a22d4b4f54d911e3e6e3ed6781d975e82f5ea13f08248aa820b2a5e110f27`.

The independently counted surface is:

| Quantity | Result |
| --- | ---: |
| modules | 58 |
| `(qualified_name, kind)` records | 1,501 |
| distinct qualified names | 1,443 |
| reviewed records | 949 |
| reviewed qualified names | 899 |
| residual `module_default` records | 552 |
| residual qualified names | 544 |
| required explicit qualified names | 215 |
| records represented by those names | 224 |
| target-bound package aliases | 197 |
| package version export | 1 |

The candidate's registry digests also recompute:

| Registry | SHA-256 |
| --- | --- |
| 35 reviewed `(qualified_name, kind)` module snapshots | `0d497c39f89a11eee0562a551178f4bddd1e7a0bf16a1cd0439778ffc30ff2cc` |
| 215-name explicit registry | `2e293bd51463d93d77cfad4fb8e950b88f9df1299fc7e55dc2a47e175a8d119d` |
| 197 alias-target pairs | `a98d9d587d2139d522a7430aed58c18a0a8fb7a344d1fdeaf4c885b71e0d7af3` |
| 552 residual `(qualified_name, kind)` records | `aaf694b8a360281955201cfd2f1675ce151f85c158f2fa1298a64639fd1b173e` |

### 3.2 Disposition vocabulary and duplicate bindings

The 949 reviewed records use exactly the six permitted values:

| Disposition | Records |
| --- | ---: |
| `taught` | 445 |
| `composed` | 146 |
| `plumbing-only` | 297 |
| `compatibility-only` | 17 |
| `diagnostic-only` | 37 |
| `unsupported` | 7 |

All 58 duplicate export/definition groups have identical reviewed semantic
fields. All 215 required names resolve only through the explicit registry.

For the 197 target-bound package aliases, the current candidate has:

- the exact recorded target;
- the target's primary and supporting locations;
- the target's status;
- `API alias identity + <target gate>` as the composed gate; and
- `plumbing-only` as the alias's own disposition.

The version string is the 198th package export and has the intended Appendix C
boundary. The two synthesis cache controls are transitive imports through
`api.py`; their ultimate defining records in `pipeline.py` are the targets
recorded by the registry.

### 3.3 Focused and formatting checks

The following checks pass at the audited bytes:

```text
15 passed in 0.14s
ruff check: all checks passed
ruff format --check: 2 files already formatted
git diff --check: clean for the four candidate files
```

## 4. Source-level semantic sample

The explicit registry is mostly accurate for the concrete branches demanded
by Sections 5–7 of the global zoom-out:

| Required semantic split | Independent result |
| --- | --- |
| Planck and the three integration spines | Pass. Chapter 1/2 ownership, `integrated` status, array role, and NumPy/Torch/`prange` split match source and accepted chapters. |
| ground-partition scalar, vector, and tables | Pass. All five public objects are explicit Chapter 3 `taught` records with integrated gates. |
| structured builders versus the full pipeline | Pass. The three fixed-\(n_e\) builders are Chapter 3/4 composition; `SynthesisPipeline`, its constructor, and `run` remain Chapter 10 composition. |
| production Rosseland mean | Pass for `rosseland_mean_step`: Chapter 12 owns the production harmonic mean/depth update and Chapter 1 is conceptual support only. The separate Chapter 5 surrogate gap is P0.2 below. |
| width use versus standard microturbulence | Pass. `compute_doppler_per_ion` is Chapter 3/6 width use; `standard_microturbulence` is Chapter 11 construction. |
| two-lane molecules | Pass for the explicit rows. Atmosphere diatomic/TiO/water are standard; H3+ is opt-in and absent from `source_line_paths`; synthesis text/TiO run; the H2O compiler exists but is not in the standard synthesis compiler. |
| `source_path`, `keep_slabs`, `spectral_operator` | Pass. The source-reference path is comparison-only, ordinary LTE rebuilds Planck source with zero line scattering, slab returns are diagnostic, and the operator acts jointly before normalization and host transfer. |
| `wing_mode="loop"` and `host_accumulator` | Pass. The loop is the parity route; the host accumulator rejects helium or stimulated deposits and is restricted to unstimulated metal-only use. |
| saturated-core transfer | Pass. A strict direct call rejects; the standard pipeline explicitly requests the implemented continuation. |
| exact setup guards | Pass. The ledger names only `iterations < 1`, enabled turbulent pressure, and HLINOP flag 13 for `_require_supported_run_setup`; it does not falsely claim that this guard rejects every NLTE or raw-selector spelling. |
| schema, publication, and initializer closure | Mostly pass. Schema v4, final fixed-column publication, unconverged initializer metadata, direct-abundance opt-in, and mandatory closure are separated. The current-loader misclassifications are P0.3 below. |
| device policy and CLI/API spelling | Pass. CUDA/MPS/CPU priority, MPS float64 rejection, `resolution` versus `r_grid`, and both CLI aliases match source. |
| accepted Chapter 1–5 examples named by P1.1 | Pass. The named Planck, integration, ground-partition, structured-builder, molecular, continuum, Rosseland, and microturbulence assignments are synchronized. |

This pass list does not rescue P0.3. The gate requires every branch-sensitive
public object to be reviewed, not only the selected objects above.

## 5. Adversarial results

The following mutations were applied only to in-memory copies; no candidate
file was changed.

| Mutation | Result |
| --- | --- |
| delete a required explicit qualified name | rejected by reviewed-module snapshot |
| add a public name to a reviewed module | rejected by reviewed-module snapshot |
| change the kind of a reviewed-module record | rejected by reviewed-module snapshot |
| delete an explicit override registry entry | rejected because the required name falls back |
| add an override for a nonexistent name | rejected as a missing target |
| mutate a reviewed module's raw `sha256` | **accepted; rebuilt ledger remains byte-identical** |
| rename constructor `source_path` and remove method `keep_slabs` in the raw AST inventory | **accepted; rebuilt ledger remains byte-identical** |
| retarget a package alias to a wrong but existing definition | accepted by `apply_overrides`; a frozen-ledger rebuild comparison catches a one-file change, but there is no source import-binding check |
| change an explicit disposition to a different allowed value | accepted by `apply_overrides`; the frozen-ledger comparison catches a one-file change, but the semantic value has no independent source-derived check |
| give a residual default an invalid location, empty responsibility/gate, bogus status, and invalid disposition | accepted by `apply_overrides`; the frozen canonical JSON catches a checked-ledger mutation, but residual semantics themselves are not validated |

The positive snapshot failures are useful. The two byte-identical negative
cases are blocking: even the complete current rebuild test cannot see those
source changes because the builder discards exactly the fields that changed.

## 6. P0 findings

### P0.1 — The semantic review is not bound to the 35 reviewed source files

`REVIEWED_MODULE_SNAPSHOTS` is documented and implemented as a hash over only:

```text
qualified_name NUL kind LF
```

`build_symbol_coverage.py` then reduces the rich raw AST inventory to
`qualified_name`, `kind`, and `line` before applying semantic overrides. It
does not carry or validate:

- the module's `sha256`;
- public positional or keyword-only parameter names;
- defaults;
- decorators;
- summaries;
- source line extents; or
- public import bindings.

Therefore a body or signature can change without changing the reviewed
snapshot. The adversarial source-hash mutation and the
`source_path`/`keep_slabs` signature mutation both rebuilt to the exact
candidate SHA.

The raw inventory is correct at the frozen hash, but the builder does not
require that hash or its 35 source identities. A semantic proof cannot rely on
an unenforced statement in the candidate report.

**Required repair:** bind the semantic layer to the pinned raw inventory and
source bytes. At minimum:

1. require the exact pinned commit;
2. freeze and check the 35 per-module source SHA-256 values from the raw
   inventory;
3. freeze a signature fingerprint for every explicit public function,
   constructor, and method; and
4. make a source-hash or signature mutation fail before any module policy or
   explicit override is applied.

A hard-coded raw-inventory digest plus per-module/source-signature diagnostics
would also make failures easier to localize.

### P0.2 — The 552 residual records are not semantically homogeneous

The remainder occurs in only 20 modules, but “one chapter owns the module” is
not the same as “no member is branch-sensitive.” The following are concrete
counterexamples:

| Residual public object | Exact source behavior | Why fallback is invalid |
| --- | --- | --- |
| `continuum_opacity.RosselandOpacityTable`; `create_rosseland_opacity_table`; `ingest_rosseland_opacity_table`; `evaluate_rosseland_opacity`; `compute_rosseland_continuum_opacity_columns` | default-off IFOP-19 surrogate at `continuum_opacity.py:844,5709-5832` | The accepted Chapter 5 disposition explicitly calls this a separate test-only optional surrogate, not standard continuum and not the Chapter 12 production mean. All five names remain `module_default` with no disposition. |
| `synthesis.continuum.build_frequency_invariants` | `coulomb_table_energy_first=False` is the standard route; `True` is the sampled diagnostic route (`continuum.py:4960-5081`) | The accepted Chapter 5 exact contract makes this branch distinction explicit. Both export/definition records remain defaults. |
| `synthesis.molecular_equilibrium.solve_molecular_equilibrium` | `return_diagnostics=False` returns four physical arrays; `True` adds iteration counts, formation logs, and structure (`molecular_equilibrium.py:460-669`) | This accepted Chapter 4 diagnostic branch is precisely the kind of optional return that P0.3 requires to be reviewed. |
| `atmosphere.line_catalog.decode_selected_line_words` | `detect_swapped_layout=True` performs external byte-layout compatibility detection; fresh native words deliberately pass `False` (`line_catalog.py:99-124`) | Compatibility behavior is mixed into a Chapter 7 decoder but is not assigned `compatibility-only` support or an exact branch gate. |
| `synthesis.hydrogen_lines.precompute_invariants` | Lyman records raise `NotImplementedError`; the implemented optical path is Balmer (`hydrogen_lines.py:1054-1127`) | The candidate methodology says explicit unsupported guards were reviewed, yet this public guard is a default. |
| `synthesis.hydrogen_lines.accumulate_hydrogen` | `apply_stim=True` is standard; `False` divides the stimulated factor back out (`hydrogen_lines.py:1610,2055-2065`) | This is a parity/diagnostic branch analogous to the explicitly reviewed atomic `apply_stim` boundary, but it falls back. |
| `atmosphere.equation_of_state.saha_partition_depth` and `_batch` | public scalar/batch paths reject unsupported population modes (`equation_of_state.py:1132-1134,1270-1272`) | These accepted Chapter 3 public routines are integrated, but their unsupported-mode behavior has no semantic disposition or exact gate. |
| `atmosphere.line_profile_math.load_line_opacity_tables` | optional path plus `force_reload=False` cache-control branch (`line_profile_math.py:405`) | A cache-control branch remains inside an otherwise physics-owned Chapter 6 module default. |
| `atmosphere.temperature_correction.apply_temperature_correction` | mode, iteration, convection, smoothing, pressure, and remap branches share one public entry (`temperature_correction.py:313-917`) | The primary chapter is homogeneous, but the public branch surface is not. It needs one explicit reviewed responsibility/gate rather than unexamined inheritance. |

These examples are enough to disprove the candidate's residual-homogeneity
claim. They are not a demand for 552 hand-written paragraphs. A small number
of additional explicit public entries, plus a reviewed homogeneous policy for
the true field/table remainder, would close the gap compactly.

**Required repair:** enumerate every residual public function, constructor, or
method with an optional diagnostic, compatibility, unsupported, cache-control,
or parity branch. Give each one an explicit disposition, responsibility, exact
gate, and accepted/planned status. Then freeze the remaining homogeneous
surface and repeat this residual audit.

### P0.3 — Four explicit `compatibility-only` dispositions are semantically wrong

The following reviewed entries use an allowed word but not the correct role:

| Explicit entry | Candidate role | Pinned behavior |
| --- | --- | --- |
| `atomic_lines.LineCatalog.from_npz` | `compatibility-only` | This is the current compiled atomic-cache hit used by `load_catalog` (`atomic_lines.py:475,1140`), not an older-schema compatibility path. |
| `molecular_lines.MolecularLineCatalog.from_npz` | `compatibility-only` | This is the current compiled molecular-cache hit used by `load_catalog` (`molecular_lines.py:195,317`). |
| `molecular_lines.MolecularLineCatalog.from_mapping` | `compatibility-only` | Its docstring says “modern molecular catalog arrays,” and current source construction calls it (`molecular_lines.py:205,284`). |
| `atmosphere.warm_start.load_atmosphere_initializer` | `compatibility-only` | This loads the bundled current complete-state initializer and is called by standard support, deterministic-label, and emulator routes (`warm_start.py:811,982,1049,1254`). |

The three cache records are appropriate Appendix B `plumbing-only` entries.
The initializer loader is current Chapter 14 composition or plumbing, with
checkpoint schema/hash validation as its gate. None is compatibility-only.

This matters pedagogically: `compatibility-only` permits the main construction
to omit a path as an input boundary, whereas these loaders execute in the
working standard implementation.

## 7. P1 findings

### P1.1 — `diagnostics_path` and `debug_state_path` must not share one claim

The override comprehension gives both fields the responsibility “optional
diagnostic/debug output target” and the same opt-in write/no-write gate.
`debug_state_path` is written in `runner.py:1914-1944`.
`diagnostics_path` appears only in the dataclass declaration
(`config.py:33`) and is not written.

This exact distinction is already frozen in `PLAN.md` and `COVERAGE.md`:
`diagnostics_path` is declared but unwritten and must not be documented as a
file product. Split the records. Keep `diagnostics_path` as an unwritten API
boundary and `debug_state_path` as the actual diagnostic publication path.

### P1.2 — Alias and semantic-value hardening is weaker than surface hardening

The current frozen bundle catches a one-file change because the canonical JSON
must rebuild exactly. `apply_overrides` itself nevertheless accepts:

- a package alias retargeted to a different existing public object; and
- an explicit record changed from one allowed disposition to another.

The alias registry and explicit-name digests in the candidate report freeze
targets and names, but not the complete semantic override values in an
independently asserted contract. This did not create a wrong canonical alias
at the audited bytes; all current bindings pass. It is a maintenance weakness.

After repairing P0.1, add:

- a source-derived import/ultimate-definition binding check for all 197
  aliases; and
- a canonical digest of the complete explicit semantic registry, including
  disposition, locations, responsibility, gate, and status.

### P1.3 — Accepted-chapter synchronization tests are examples, not a complete join

The explicit P1.1 examples named by the global zoom-out are correct. The test
named `test_accepted_chapters_one_through_five_are_synchronized`, however,
checks only nine representative qualified names.

The missed Chapter 4 synthesis diagnostic branch and Chapter 5 surrogate
branch show why representative checks are insufficient. Generate the
accepted-status join from each accepted chapter's symbol disposition or
exact-source contract, then assert every joined qualified name and status.

## 8. P2 findings

1. The candidate says an alias “copies” the target gate. The implementation
   correctly uses `API alias identity + <target gate>`. State “composes” rather
   than “copies” to make the evidence exact.
2. The residual description should not call all 552 entries “fields,
   constants, table carriers, and kernels.” It includes complex public solvers,
   compatibility decoders, cache loaders, and optional-return functions.
3. Record the independently recomputable 35-source manifest digest separately
   from the existing reviewed public-surface digest; they protect different
   claims.

## 9. Repair and re-audit gate

P0.3 should be re-audited only after:

1. the builder rejects raw-inventory source-hash and public-signature drift;
2. all residual optional, diagnostic, compatibility, parity, cache-control,
   and unsupported public branches receive explicit semantic reviews;
3. the standard cache and initializer loaders are reclassified;
4. the unwritten diagnostics field is separated from the written debug path;
5. the canonical ledger again rebuilds byte-identically;
6. the 58-module, 1,501-record, explicit-name, alias, duplicate-kind, and
   six-value checks still pass; and
7. adversarial source-hash, signature, alias-target, disposition, location,
   gate, and status mutations all fail at a source-independent gate.

The repair does not require a new chapter or a prose entry for every dataclass
field. It requires binding the semantic review to the source bytes and moving
the demonstrably branch-sensitive exceptions out of the residual default.

## 10. Final verdict

**REJECT P0.3 — the global symbol inventory is not yet a complete semantic
coverage proof.**

- **P0:** reviewed-module identity ignores source hashes and signatures;
- **P0:** branch-sensitive public routines remain among the 552 defaults;
- **P0:** current standard loaders are misclassified as compatibility-only;
- **P1:** the unwritten diagnostics field is conflated with a written debug
  path;
- **P1:** alias/semantic-value mutation hardening is not source-derived; and
- **P1:** accepted-chapter status synchronization is sampled rather than
  exhaustively joined.

The deterministic inventory, current explicit branch rows, duplicate
consistency, package-alias coverage, and canonical rebuild are retained as
strong foundations for the repair.

<!-- BEGIN DELIMITED REPAIR RE-AUDIT: 2026-07-30 -->

---

# Repair re-audit of the global symbol semantic candidate

Re-audit date: 2026-07-30  
Status: **independent repair re-audit; candidate, builder, tests, and ledger
not modified**  
Decision for P0.3: **REJECT**  
Pinned Payne Zero source:
`9c44001feae40b85146630499e6f8a5fed42e5af`

## 11. Re-audit scope and exact identities

This appendix preserves the original rejection above. The exact pre-append
SHA-256 of this independent audit was
`b8cdd8f1a51be7225cd4886ac9fcddf5c5f22b17ac3f57cf30b14d4b91cc1fdd`.

The repaired candidate supplied for this re-audit has these identities:

| Object | Re-audited SHA-256 |
| --- | --- |
| semantic override builder | `ff9c72a74eb6931f38c9deaeccb29d2f146897cca0042044eba5be4105f961d7` |
| focused semantic tests | `9f7f53a89e84b88e29214e702485f109e95e667bca249b1bf462e3427c63afb4` |
| canonical semantic ledger | `edc6ef0429ffab641652712e1cc6b433b18ad40dcc2c1a60bfac70f8169caddb` |
| repaired candidate report | `c61ee08375cf8f006b61f278edee87f637e8dfee79f0f400bcaee2c7cdee4acc` |
| raw public-source inventory | `010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf` |

The repair closes most of the concrete defects named in the first audit. It
does not close the global gate, for two independent reasons:

1. multiple physically or operationally branch-sensitive public routines
   remain described only by broad module policies, so the 240-name explicit
   set is not exhaustive; and
2. the semantic policy that supplies 1,053 records is not digest-bound at
   all. Mutations of its disposition, locations, responsibility, gate, or
   status are accepted by `apply_overrides`.

The repaired candidate therefore has zero records *labelled*
`module_default`, but it does not yet prove zero unexamined branch-sensitive
defaults.

## 12. Reproducibility and count results

### 12.1 Source and ledger rebuilds

I rebuilt the complete raw inventory from the read-only Payne Zero checkout
at the pinned commit. The result covered 58 modules, was byte-identical to
`audit/paynezero_symbols.json`, and had SHA-256
`010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf`.

I then ran the canonical coverage builder against that inventory. It reported
58 mapped modules, 1,501 mapped public objects, and 1,501 reviewed overrides.
The rebuilt ledger was byte-identical to the candidate and had SHA-256
`edc6ef0429ffab641652712e1cc6b433b18ad40dcc2c1a60bfac70f8169caddb`.

The focused suite passes when invoked through the repository's Python module
path:

```text
21 passed in 0.62s
```

### 12.2 Exact semantic counts

| Quantity | Independent result |
| --- | ---: |
| modules in raw inventory | 58 |
| `(qualified_name, kind)` records | 1,501 |
| distinct qualified names | 1,443 |
| `reviewed_symbol_override` records | 1,501 |
| `module_default` records | 0 |
| module-policy records after replacement | 1,053 |
| explicit-symbol records | 250 |
| explicit qualified names | 240 |
| package-alias records | 198 |

The disposition counts exactly reproduce the candidate:

| Disposition | Records |
| --- | ---: |
| `taught` | 915 |
| `plumbing-only` | 352 |
| `composed` | 173 |
| `diagnostic-only` | 41 |
| `compatibility-only` | 13 |
| `unsupported` | 7 |

### 12.3 Fifty-five reviewed modules and the remaining three

The 55-module source manifest and complete descriptor surface are present and
validated before semantic assignment. The three other source modules are
justifiably handled outside that policy:

| Excluded source module | Public role |
| --- | --- |
| `payne_zero_atmosphere.__init__` | 182 static package exports; exact source SHA is frozen and aliases are handled by the complete alias registry |
| `payne_zero_synthesis.__init__` | 16 static package exports; exact source SHA is frozen and aliases are handled by the complete alias registry |
| `payne_zero_atmosphere.__main__` | zero public exports or definitions; still bound by the complete raw-inventory digest |

This part of the repair passes.

## 13. Original P0/P1 closure matrix

| Original defect or requested closure | Re-audit result |
| --- | --- |
| exact pinned commit and raw source SHA binding | **Pass.** A reviewed-module SHA mutation is rejected before assignment. |
| complete public descriptors, signatures, and annotated fields | **Pass.** Constructor, method, and field-descriptor mutations are rejected. |
| `source_path` and `keep_slabs` signatures/defaults | **Pass.** Both are present, embedded, and protected by the source surface, module SHA, and default registry digest. |
| twelve reviewed default contracts | **Pass semantically.** Every registered default value agrees with the pinned source; the two lexical spelling issues are P2.1 below. |
| 55 reviewed modules and three exclusions | **Pass.** See Section 12.3. |
| IFOP-19 surrogate versus production Rosseland mean | **Pass for the five repaired explicit objects.** |
| continuum Coulomb-table orientation | **Pass for `build_frequency_invariants`.** |
| molecular optional diagnostics | **Pass for `solve_molecular_equilibrium`.** |
| swapped selected-line layout | **Pass for `decode_selected_line_words`.** |
| hydrogen Lyman guard and stimulated-emission switch | **Pass for the two repaired hydrogen routines.** |
| atmosphere Saha population modes | **Pass for the scalar and batch routines.** |
| line-profile cache and temperature-correction branches | **Pass for the repaired explicit routines.** |
| current atomic/molecular cache loaders | **Pass.** They are now `plumbing-only`, not compatibility-only. |
| current atmosphere initializer loader | **Pass.** It is now Chapter 14 composition. |
| `diagnostics_path` versus `debug_state_path` | **Pass.** Declaration-only and actual debug publication are distinct. |
| 75 selected Chapter 1–5 tuples | **Internal pass only.** All 75 ledger tuples equal the selected explicit-registry tuples, but the join is not bound to the accepted chapter artifacts; see P1.1. |
| complete branch-sensitive review | **Fail.** Concrete unlisted branch-sensitive routines remain under generic policies; see P0.1. |
| source-independent semantic mutation rejection | **Fail for 1,053 module-policy records.** See P0.2. |

The corrected cache, initializer, diagnostics-path, IFOP-19, continuum-layout,
molecular-diagnostics, swapped-layout, hydrogen, Saha, and
temperature-correction rows should be retained in the next repair.

## 14. Adversarial re-audit

All source/inventory/registry adversaries below were in-memory or temporary;
no candidate file was changed.

| Mutation | Result |
| --- | --- |
| mutate a raw reviewed-module SHA | rejected |
| rename constructor parameter `source_path` | rejected by the reviewed public surface |
| rename method parameter `keep_slabs` | rejected by the reviewed public surface |
| mutate an annotated-field descriptor | rejected by the reviewed public surface |
| mutate unrelated raw-inventory metadata | rejected by the complete raw-content digest |
| mutate one default value in each of all twelve default contracts | all twelve rejected by the default-registry digest |
| delete or add an explicit registry entry | rejected |
| retarget or delete an alias | rejected by the alias-target digest |
| mutate an explicit disposition | rejected |
| mutate an explicit primary or supporting location | rejected |
| mutate an explicit responsibility | rejected |
| mutate an explicit gate | rejected |
| mutate an explicit status | rejected |
| mutate a module-policy disposition | **accepted** |
| mutate a module-policy primary or supporting location | **accepted** |
| mutate a module-policy responsibility | **accepted** |
| mutate a module-policy gate | **accepted** |
| mutate a module-policy status | **accepted** |

The current 55-entry `REVIEWED_MODULE_POLICIES` value has independently
computed canonical SHA-256
`f9e02d3aa496f69b23aa0936d69550f716ca862fbab8f1af04c49dd989684b46`.
No expected policy digest appears in the builder, and no validator is called
for it.

By contrast, the independently recomputed snapshot and required-name digests
do equal the values in the candidate report:

| Registry | SHA-256 |
| --- | --- |
| 55 reviewed ledger snapshots | `1465aee28881a29fd63dced37d566d07c15f7191a329f6c7c473c4137a9058f3` |
| 240 required explicit names | `7df112e834d1da8d7c98b38cc99d9d2c199e63dd7f46c5f9785ba23eb2c359ad` |

The candidate report's
`bee7f4a5cc7b68c5dc3cfe90fa8787bbb5b45b75961314672c7617573d0f794d`
“complete 1,501-record semantic proof” is not an enforced constant in the
builder or focused tests. It therefore does not close the accepted
module-policy mutations above.

## 15. P0 findings

### P0.1 — The 240-name explicit set is not an exhaustive branch set

Changing every remaining record's precision label from `module_default` to
`reviewed_symbol_override` does not make a broad per-module sentence an exact
semantic review. The following pinned public functions are concrete
counterexamples:

| Public routine still using `reviewed_module_registry` | Exact pinned branch | Generic ledger claim that misses it |
| --- | --- | --- |
| `payne_zero_synthesis.molecular_lines.accumulate_molecular` | `apply_stim=True` is the standard pipeline call; `False` omits the stimulated-emission factor. `chunk_lines=None` selects an environment-controlled chunk size, while an explicit value is clamped and used directly. | “GPU molecular catalog/invariants/chunked deposits”; “component/backend parity” |
| `payne_zero_atmosphere.continuum_opacity.compute_continuum_opacity_columns` | `opacity_flags=None` activates a 20-flag default vector; individual flags select physical opacity families. IFOP-19 contributes only when flag 19 is active **and** `rosseland_table` is not `None`. | “full atmosphere continuum, scattering, sampling grid, Rosseland lookup”; “component + slab parity” |
| `payne_zero_synthesis.equation_of_state.partition_functions_for_elements` | `apply_ground_partition=True` is the ordinary branch; the molecular bridge explicitly calls `False` for its no-ground-floor parity path. | “Torch Saha/populations, full-ne and fixed-ne states”; “state/bridge parity” |
| `payne_zero_synthesis.equation_of_state.solve_electron_density`, `solve_population_state`, and `solve_population_state_at_electron_density` | `molecules=False` and `True` select atom-only versus molecular closure/assembly, with `molecules_path` controlling the catalog. The Chapter 3 and 4 exact-source contracts explicitly distinguish these routes. | the same undifferentiated EOS module policy |
| `payne_zero_atmosphere.radiative_transfer.load_radiative_transfer_tables` | `path=None` chooses the packaged archive and `force_reload=False` selects warm-cache reuse; forced reload validates and replaces the cache. | “depth integration, differentiation, remap, tables”; “helper parity” |
| `payne_zero_synthesis.atomic_lines.parse_catalog` | `apply_iso_corr=True` controls isotope corrections, `sort` changes catalog ordering, and `catalog_path=None` selects the default source. | “source decoding, corrections, routing, margins, typed catalog”; “catalog identity” |

The molecular stimulated-emission omission is especially decisive. The
repair correctly made the equivalent hydrogen `apply_stim=True/False` branch
explicit, yet left the molecular branch generic even though the standard
pipeline calls it with `apply_stim=True`.

Likewise, the repair made five IFOP-19 helper objects explicit but left the
actual public continuum assembler—which interprets all opacity flags and the
two-condition IFOP-19 activation—generic. The helper rows do not substitute
for the assembler's public branch contract.

These are not source-signature gaps: their signatures are correctly frozen.
They are semantic-classification gaps. The source fingerprint proves which
function was reviewed, but a generic responsibility does not say what its
switches mean.

**Required repair:** repeat the callable-default scan over all 1,053
module-policy records and explicitly classify every physical, optional
diagnostic, compatibility, parity, cache, environment, unsupported, and
algorithm-selection branch. At minimum, the routines in the table above must
leave the generic module policy. Recompute the required-name set only after
that residual semantic audit.

### P0.2 — The 1,053-record module semantic policy is not fail-closed

`_validate_inventory` digest-checks:

- reviewed source identities and public descriptors;
- explicit source descriptors;
- the twelve default contracts;
- `EXPLICIT_OVERRIDES`;
- alias targets; and
- the self-derived 75-name subset.

It never digest-checks `REVIEWED_MODULE_POLICIES`, even though this registry
supplies the disposition, locations, responsibility, gate, and status for
1,053 final records.

I changed each of those six semantic fields in
`payne_zero_synthesis.molecular_lines` independently. Every mutation was
accepted by `apply_overrides` and propagated to
`accumulate_molecular`. Responsibility, gate, location, and status mutations
also preserve the suite's disposition counts, so updating the generated
ledger would make the canonical equality test tautologically agree.

This directly violates the requested source-independent mutation gate and
invalidates the claim that all 1,501 semantic mappings are frozen.

**Required repair:** add and validate a canonical digest of the complete
55-module policy registry before assignment. Add adversarial tests for every
semantic field on a module-policy-owned record, not only on an explicit
record. The final complete-proof digest should also be defined by executable
code and asserted independently rather than appearing only in the report.

## 16. P1 finding

### P1.1 — The “75-name accepted-status join” is self-derived, not an acceptance join

The builder constructs `accepted_join` by filtering `EXPLICIT_OVERRIDES` for:

```text
primary_location in chapter-1 ... chapter-5
and status == integrated
```

It then compares that value with a digest of the same manually authored
registry. The focused test repeats the same filter and compares the generated
ledger with those registry values.

No Chapter 1–5 acceptance record, exact-source contract, symbol-disposition
record, path, or SHA-256 is an input to this calculation. A search of the
builder and focused test finds no accepted-artifact reference. Therefore all
75 internal tuples pass, but a change in an accepted chapter contract is
invisible.

This also explains why integrated Chapter 3/4 callable contracts such as
`partition_functions_for_elements`, `solve_electron_density`,
`solve_population_state`, and
`solve_population_state_at_electron_density` can remain under a module policy
and outside the claimed join, despite being explicitly enumerated in the
Chapter 3 and Chapter 4 exact-source contracts.

**Required repair:** create a frozen, independently generated acceptance
manifest from the accepted Chapter 1–5 disposition/exact-source records.
Bind its source paths and hashes, then compare the ledger against that
manifest. A subset selected from the semantic registry itself is an internal
consistency check, not an acceptance join.

## 17. P2 findings

### P2.1 — Two “source-spelled” default contracts normalize quote style

All twelve registered defaults are semantically equal to the pinned source.
Two contracts are not lexically source-spelled as claimed:

- the three `SynthesisPipeline.__init__` table-path defaults use double-quoted
  filenames in source but single quotes in the registry; and
- `line_opacity.accumulate_atomic` uses `wing_mode="batched"` in source but
  records `"'batched'"`.

The module SHA still binds the exact bytes, so this is not a semantic P0.
Either preserve the exact source segments or describe these strings as
canonicalized expressions rather than source-spelled expressions.

### P2.2 — The complete-proof hash needs a named executable definition

The candidate report lists a “complete 1,501-record semantic proof” digest but
does not state its projection and the builder does not compute or assert it.
Define the exact fields, ordering, duplicate-kind treatment, and canonical
encoding next to the validator. This will make the number independently
recomputable and prevent it from being mistaken for an enforced gate.

## 18. Repair re-audit verdict

**REJECT P0.3 — the repaired ledger is reproducible and much better bound to
source, but it is not yet a complete or fail-closed semantic coverage proof.**

- **P0:** branch-sensitive public routines remain under generic module
  policies despite the zero-`module_default` count;
- **P0:** all semantic fields on the 1,053 module-policy records can be
  mutated without rejection;
- **P1:** the 75-name Chapter 1–5 check is derived from the semantic registry
  itself, not joined to accepted chapter evidence; and
- **P2:** two default contracts normalize rather than preserve source quote
  spelling, and the reported complete-proof digest lacks an executable
  definition.

The following repaired work passes and should not be redone: exact raw and
ledger rebuilds; 1,501/1,443 counts; 55-module source and descriptor binding;
the three-module exclusion rationale; all twelve defaults semantically;
the repaired IFOP-19, continuum-orientation, molecular-diagnostic,
swapped-layout, hydrogen, Saha, line-profile-cache, and
temperature-correction entries; the current cache/initializer
classifications; the diagnostics/debug-path split; alias binding; and the 75
internal registry-to-ledger tuples.

<!-- END DELIMITED REPAIR RE-AUDIT: 2026-07-30 -->

<!-- BEGIN DELIMITED SECOND-REPAIR RE-AUDIT: 2026-07-30 -->

---

# Second-repair re-audit of the global symbol semantic candidate

Re-audit date: 2026-07-30  
Status: **independent second-repair re-audit; candidate inputs not modified**  
Decision for P0.3: **REJECT**  
Pinned Payne Zero source:
`9c44001feae40b85146630499e6f8a5fed42e5af`

## 19. Scope and exact input identities

This appendix preserves both earlier rejection records. The exact SHA-256 of
this independent audit before the present append was
`17ec40ff377eabb8d8c0207f2c964c9824ddbc8a5b52806a80eec4274367e39e`.

| Object | Second-repair SHA-256 |
| --- | --- |
| semantic override builder | `6e4a0804772b32436c9b2cece1811406594bda6ec484879334ab58789da71503` |
| focused semantic tests | `a8fc9afb54826dc99f9a602255edbceebb181066bc232d8684d8a7d1053d702e` |
| canonical semantic ledger | `2623a693da7c12db5d50fe859606999562cbfd48bfeba12c7c5c31b2f507126e` |
| repaired candidate report | `64b39648141a46a021997721317a05fc2df496ab58df4d141244d2cf18cfecdc` |
| raw public-source inventory | `010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf` |

The second repair closes the module-policy mutation defect, implements an
executable complete-record proof, exactly repairs the previously named
default spellings, and makes every branch named by the immediately preceding
audit explicit.

It still does not close P0.3. A fresh residual source pass finds additional
public physical and algorithmic switches under broad module policies,
including one lane split already explicit in the accepted Chapter 5 contract.
The new external-authority join is byte-bound but uses bare leaf-name
occurrence as its extraction test; several of its 53 tuples are not actually
established by the cited text, and the extraction is not exhaustive.

## 20. Reproducibility and proof checks

### 20.1 Byte-identical rebuilds

I independently rebuilt the 58-module raw inventory from the read-only pinned
checkout. Its SHA-256 was
`010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf`,
and its bytes were identical to the checked inventory.

The canonical coverage build then reported 58 modules, 1,501 public-object
records, and 1,501 reviewed overrides. Its output was byte-identical to the
checked ledger and had SHA-256
`2623a693da7c12db5d50fe859606999562cbfd48bfeba12c7c5c31b2f507126e`.

The focused suite and static checks pass:

```text
27 passed in 2.83s
ruff check: all checks passed
ruff format --check: 2 files already formatted
Python compilation: pass
git diff --check on the four candidate inputs: pass
```

### 20.2 Exact counts

| Quantity | Independent result |
| --- | ---: |
| records | 1,501 |
| distinct qualified names | 1,443 |
| `module_default` records | 0 |
| module-policy records | 1,036 |
| explicit-symbol records | 267 |
| explicit qualified names | 251 |
| package-alias records | 198 |

The disposition counts reproduce the candidate exactly:

| Disposition | Records |
| --- | ---: |
| `taught` | 909 |
| `plumbing-only` | 354 |
| `composed` | 177 |
| `diagnostic-only` | 41 |
| `compatibility-only` | 13 |
| `unsupported` | 7 |

### 20.3 Enforced integrity digests

The following independently recomputed values equal the enforced constants:

| Proof object | SHA-256 |
| --- | --- |
| effective seven-field policies for all 55 modules | `b10f3dad89ba00e9c96b254f1ccb708d53c0d82cc84d2509946a2cd840d0a3b7` |
| accepted-artifact manifest | `f2cbf34acd60ea7ec5f8c87ee49a400705259927473f318d1919b7980e7f19ca` |
| complete 1,501-record semantic proof | `fe2c656289c8b67f10e72ad4780d9b2038462a908d8f3050519f5681b65341a0` |

Every effective module policy has exactly:

1. `primary_location`;
2. `supporting_locations`;
3. `responsibility`;
4. `gate`;
5. `status`;
6. `semantic_disposition`; and
7. `semantic_review_reason`.

The complete proof preserves duplicate kinds and all 17 declared record
fields. This repair is mechanically effective.

### 20.4 Exact source defaults

I extracted defaults independently with Python ASTs from the 23 pinned
functions/methods. All 23 complete default mappings are lexically identical
to `REVIEWED_DEFAULT_CONTRACTS`; there are zero mismatches.

The four specially asserted expression records also match both their exact
source segment and `ast.dump`:

- the three double-quoted `SynthesisPipeline.__init__` table paths; and
- `accumulate_atomic`'s exact `wing_mode="batched"` expression.

The prior quote-spelling P2 is closed.

## 21. New explicit contracts that pass

The eleven newly explicit names are source-faithful:

- molecular accumulation correctly distinguishes stimulated emission and the
  environment/explicit/clamped chunk policy;
- `molecular_chunk_lines` correctly records absent, empty, invalid, zero, and
  positive environment behavior;
- the full continuum assembler correctly records the default 20 flags,
  family selection, and the two-condition IFOP-19 activation;
- the light-family and scattering assemblers correctly record their flag and
  optional-table branches;
- selected-element partitions correctly distinguish ordinary ground-floor
  application from the molecular bridge's no-ground-floor route;
- all three synthesis population solvers correctly distinguish atom-only and
  molecular routes, including the two-solve full-state seam and fixed-public
  electron preservation;
- the transfer-table loader correctly records default/explicit paths and
  cold/warm/forced cache behavior; and
- the atomic parser correctly records default/explicit sources, isotope
  correction, and sort order.

Their exact descriptors and defaults are included in the enlarged source and
default registries. These repairs should be retained.

## 22. Adversarial results

All mutations were in-memory or temporary. No candidate input or accepted
authority was edited.

| Mutation | Result |
| --- | --- |
| each of the seven effective module-policy fields | all rejected by the policy digest |
| required complete-proof digest | rejected |
| one complete-proof record's location, support, disposition, responsibility, gate, status, or source-object identity | rejected |
| authority-manifest hash | rejected |
| simulated accepted-authority byte change | rejected by the per-file hash |
| accepted-symbol disposition registry | rejected by the manifest digest |
| accepted ledger location or status | rejected by the authority tuple check |
| raw public signature | rejected by the source surface |
| raw annotated-field descriptor | rejected by the source surface |
| reviewed default | rejected by the default-registry digest |
| exact quote or AST syntax | rejected by the exact-syntax digest |
| package alias target | rejected by the alias-target digest |

The mutation hardening requested after the second rejection passes. The
remaining blockers below concern whether the sealed semantic claims are
complete and whether the external texts actually support the extracted
tuples.

## 23. P0 finding

### P0.1 — Branch-sensitive public routines still inherit generic module claims

The new explicit set is not the fixed point of the residual audit. The
strongest counterexample is already a binding accepted-chapter contract:

| Public routine | Exact source/accepted behavior | Current generic ledger claim |
| --- | --- | --- |
| `payne_zero_synthesis.continuum.compute_sampled_continuum` | The accepted Chapter 5 exact-source contract distinguishes `frequency_invariants=None` as the sampled-diagnostic lane from an explicitly materialized `FrequencyInvariants` sampled extension. `_compute_at_freqs` follows separate scalar and materialized/batched branches. | `reviewed_module_registry`; “synthesis continuum populations, edge sampling, interpolation”; “component + slab parity”; no default contract |
| `payne_zero_atmosphere.convection.compute_convection` | Eight optional derivative arrays collectively select finite-difference thermodynamics versus the ideal-gas fallback; `convection_enabled` selects 30 correction iterations versus one; zero mixing length and top-layer rules create further exact exits. | `reviewed_module_registry`; “finite-difference thermodynamics and mixing length”; “derivative/flux parity”; no default contract |
| `payne_zero_atmosphere.runner.compute_convection_finite_difference_samples` | `molecules_enabled` requires/restores molecular state, while `molecular_thermal_energy_tracks_perturbation` changes whether every thermodynamic perturbation rewrites molecular thermal energy. | the undifferentiated runner policy “exact physical iteration and final product”; “one-pass + convergence” |
| `payne_zero_atmosphere.runner.finalize_transfer_state` | Disabled convection, caller-supplied convective arrays, and enabled internal finite-difference/convection construction are distinct routes. The molecular perturbation flag is forwarded only through the internally constructed route. | the same undifferentiated runner policy |
| `payne_zero_atmosphere.continuum_opacity.compute_hminus_opacity_columns` | `hminus_departure_coefficient=None` supplies the Chapter 5 unit-LTE factor; an explicit depth array changes the H-minus population and has its own shape rejection. | the generic continuum-module claim; no default contract |
| `payne_zero_synthesis.atomic_lines.load_catalog` | In addition to cache reuse, `rebuild=False/True` bypasses or forces parsing, while `sort` and `apply_iso_corr` are forwarded into the newly explicit parser and enter the cache key. | explicit but only “optional compiled-cache reuse”; its gate omits rebuild, isotope, and ordering branches; no default contract |

Two smaller residuals confirm that this is not confined to future convection:

- `convergence.max_normalized_column_delta(symmetric=False)` changes the
  normalization denominator when `True`; and
- both public `prewarm(force=False)` routines retain unreviewed cache-control
  switches under module policies.

The first table row is decisive on its own. The accepted Chapter 5 contract
at `design/chapter05_exact_source_contract.md` explicitly defines the two
`compute_sampled_continuum` lanes, and
`design/chapter05_symbol_disposition.md` requires one call without invariants
and one with invariants. A generic module phrase cannot be an exact semantic
review of that public optional branch.

The complete-proof hash correctly seals the current 1,501 records. It cannot
turn an incomplete baseline classification into a complete one.

**Required repair:** run the residual callable/default scan to a genuine fixed
point rather than adding only the examples named by the preceding audit.
Explicitly review at least the routines above and their cache/convergence
siblings. Require every remaining module-policy callable with a default or
optional input to have a documented reason why that input does not select a
physical, diagnostic, compatibility, parity, cache, environment,
unsupported, or algorithmic branch.

## 24. P1 findings

### P1.1 — The 53-symbol external join verifies leaf substrings, not exact authorities

The eleven authority-file hashes are correct. The semantic extraction is not.
For each qualified name, `_accepted_artifact_join` reduces the symbol to:

```text
qualified_name.rsplit(".", 1)[-1]
```

and accepts the tuple if that bare leaf occurs anywhere in the chapter's
combined authority text.

This creates concrete false evidence:

- Chapter 4's only occurrence of `molecular_state` is in the archive filename
  `chapter04_atmosphere_molecular_state_cpu_float64.npz`. It does not
  establish
  `payne_zero_atmosphere.runner.AtmospherePopulationState.molecular_state` as
  a composed integrated field.
- Chapter 2's only `ModelAtmosphere` occurrence is the type annotation on
  `linear_elemental_abundances`. The seven selected
  `ModelAtmosphere.<field>` leaf names occur in transfer signatures or the
  schema-v4 array table, not as an exact `ModelAtmosphere` field contract.
- Both atmosphere and synthesis `solve_molecular_equilibrium` entries can be
  satisfied by the same bare spelling. The check cannot identify which
  package/function an occurrence authorizes.
- The selected synthesis `PopulationState` molecular field leaves occur in a
  section explicitly titled “Atmosphere state”; the leaf check cannot prove
  class ownership.

Cryptographic pinning proves that these are the exact files inspected. It
does not make an incidental substring into semantic evidence for a qualified
symbol, chapter, disposition, or status.

**Required repair:** use exact qualified markers or a structured,
authority-owned manifest whose entries include the qualified name, chapter,
disposition, status, and source context. The extraction test must reject
ambiguous duplicate leaves and must point to the exact table row, signature,
or accepted checkpoint that establishes each tuple.

### P1.2 — The external extraction is selective rather than exhaustive

The binding Chapter 5 symbol-disposition authority contains 46 exact public
continuum symbols. `ACCEPTED_ARTIFACT_SYMBOLS["chapter-5"]` contains 12.
There is no extraction rule in the authority or validator explaining why
those 12 are the complete status join.

The omission is not limited to test-only or inactive helpers. The authority
marks all of the following synthesis objects visible or progressive and the
final acceptance closes the synthesis continuum lanes:

- `ContinuumTables`;
- `FrequencyInvariants`;
- `build_pops`;
- `pops_from_population_state`;
- `build_edge_sample_frequencies`;
- `compute_sampled_continuum`; and
- `continuum`.

All seven remain `verified` module-policy records rather than appearing in
the external status join; only `build_frequency_invariants` from the same
binding table is selected as `integrated`.

Thus the new 53-name mechanism is external and hash-bound, but it does not
close the original requirement for a complete accepted Chapter 1–5 status
join. It replaces a self-derived subset with a manually selected,
leaf-matched subset.

**Required repair:** define the extraction scope from each authority itself.
For Chapter 5, parse or transcribe every binding symbol-disposition row and
map its reader disposition to an explicit ledger status rule. Apply analogous
complete-surface rules to Chapters 1–4 instead of choosing examples.

## 25. P2 findings

There is no remaining default-spelling or executable-proof P2 at these bytes.
The 23 default contracts, four exact syntax records, seven-field policy
digest, and complete proof all independently recompute.

The accepted-authority marker weakness is P1 rather than P2 because it
permits semantically unsupported status/location tuples to pass.

## 26. Second-repair verdict

**REJECT P0.3 — the integrity system now robustly seals the ledger, but the
sealed semantic classification and accepted-artifact extraction are still
incomplete.**

- **P0:** accepted and future public branch switches remain under generic
  module policies, including the exact Chapter 5 sampled diagnostic/extension
  split and the central convection family;
- **P1:** the 53-symbol external join accepts incidental bare-leaf
  occurrences that do not establish exact qualified symbols;
- **P1:** the accepted-artifact selection is not exhaustive, including within
  the binding Chapter 5 symbol-disposition table; and
- **P2:** none beyond the P0/P1 findings at the audited bytes.

Retain without rework: exact raw and ledger reproducibility; all source
hash/signature/field/default bindings; the 251 existing explicit names; the
seven-field module-policy digest; alias hardening; current
cache/initializer/diagnostics classifications; exact default quote/AST
records; all requested mutation failures; and the executable 1,501-record
proof.

<!-- END DELIMITED SECOND-REPAIR RE-AUDIT: 2026-07-30 -->

<!-- BEGIN DELIMITED THIRD-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->

## 27. Third-repair audit scope and exact identities

This section is a new, preserved independent audit of the third repair. It
does not supersede or rewrite the preceding audit history. No candidate input,
accepted authority, chapter, Bible, coverage contract, or pinned Payne Zero
source was edited during the audit.

| Audited object | SHA-256 |
| --- | --- |
| pinned Payne Zero commit | `9c44001feae40b85146630499e6f8a5fed42e5af` |
| raw inventory | `010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf` |
| prior independent audit before this append | `c6071c4a16167c33d944fccd225ed4963a06290f838b01ff7d6fa7221e0bc6a1` |
| semantic override builder | `202e67db97be05c59ac4ddfa0631500f7479e669285fbea8ccc1fed3ff5ff049` |
| focused semantic tests | `aa3f4e8c14a7c849bc422b5041ceaefd99a55c3d71b69c64c7a5f3afd5be3ecc` |
| canonical semantic ledger | `913a80777f312a8f8caa81cca2712c640a99da8d2ef2445e8e8f064b35859fab` |
| third-repair candidate report | `9f216527da1519e39905fe18fb12f5900fa42adf90b5286f56f2a3f0f5d382ec` |

## 28. Independent reproducibility and complete-ledger checks

The raw inventory was rebuilt twice from the read-only Payne Zero checkout
with `scripts/audit_paynezero_source.py` and the pinned commit requirement.
Both fresh files were byte-identical to each other and to the checked raw
inventory, with SHA-256
`010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf`.
Each run found 58 modules.

The semantic ledger was then rebuilt twice from that inventory and
`COVERAGE.md` with `scripts/build_symbol_coverage.py`. Both fresh ledgers were
byte-identical to each other and to the checked ledger, with SHA-256
`913a80777f312a8f8caa81cca2712c640a99da8d2ef2445e8e8f064b35859fab`.
Each run reported 58 mapped modules, 1,501 mapped public objects, and 1,501
reviewed overrides.

An independent record walk confirmed:

- 1,501 records and 1,443 distinct qualified names;
- 1,501 `reviewed_symbol_override` records and zero `module_default` records;
- the exact disposition counts 899 taught, 357 plumbing-only, 184 composed,
  41 diagnostic-only, 13 compatibility-only, and 7 unsupported;
- 942 reviewed-module-policy records, 309 explicit-symbol records, 198
  explicit-package-API/version records, and 52 residual-default records;
- all source-module, public-surface, and public-object fingerprints against
  the raw inventory, including constructors, fields, duplicate kinds, and
  recursively resolved package exports;
- all 197 true package-export alias targets, plus the separately defined
  public `__version__` record; and
- the duplicate-kind-preserving 1,501-record proof digest
  `63d91360b55fc35c5372576ac7a01a1dbbcee759ebf12d1a94b2103cc59ef46a`.

The independently recomputed registry digests also match:

| Registry | SHA-256 |
| --- | --- |
| 286-name explicit source surface | `ffe805be645f5d1ee5938b3dc7797690759e63c6c9f7572b775d01987ff6611a` |
| explicit semantic registry | `17cf03748879f1f698efa6468ac7b8f9e2b85876c8813a5a96b2a1a1017e26e0` |
| 32 exact default contracts | `454f0fb32d68a4173ee2502bad6b600a2c568dfe97a1adebbe32ab4d6ab01449` |
| 140-name default-bearing registry | `74008ed126f8e3a07f100769304e251acfbddc96ba91812cc1240726624a59ee` |
| 49 residual-default reviews | `c74c4d7f667da497b9f6e81919b339084bfb5d5aa1a8b3c4241cf80b5d386b99` |
| accepted-authority manifest | `10c0e3bcd029834d97f480ed3e3e09d11b01126e36c116334f665e66bc28347b` |

## 29. Source-AST default fixed point

An independent AST scan covered every public function, constructor, and
public method in all 58 pinned modules. The three package entry modules
excluded by the builder contain no additional default-bearing public
callable, so the 55-module reviewed scan and the all-module scan both recover
the same 140 qualified names.

The candidate partition is syntactically exact and disjoint:

| Default review class | Qualified names |
| --- | ---: |
| exact `REVIEWED_DEFAULT_CONTRACTS` | 32 |
| `EXPLICIT_DEFAULT_CALLABLES` | 59 |
| callable-specific `DEFAULT_CALLABLE_REVIEWS` | 49 |
| total | 140 |

There are 163 duplicate-preserving ledger records for those 140 names, and
every one carries a non-empty `default_branch_review`. All 32 exact contracts
match the pinned source spelling, keyword-only set, default expressions, and
AST source segment; the 140 signatures contain 456 defaulted parameters in
total.

The nine branches singled out by the preceding audit are now correctly
represented: sampled continuum, H-minus departure, the main convection
routine, finite-difference convection samples, transfer finalization,
symmetric convergence, both forced-prewarm switches, and atomic
`load_catalog`. Those repairs should be retained.

The 140-name *syntactic* fixed point therefore passes. The semantic fixed
point does not: a non-empty generic responsibility string is accepted as the
review for each of the 59 `EXPLICIT_DEFAULT_CALLABLES`, even when it says
nothing about one or more default-selected physical, data, cache,
environment, diagnostic, or algorithmic routes.

## 30. Exact Chapter 1–5 authority join

The previous authority findings are closed. An independent parser, separate
from `_accepted_artifact_join`, verified every declared authority-file hash,
permitted path, exact line range, marker, line hash, namespace, symbol, and
ledger evidence object.

| Accepted surface | Qualified names | Ledger evidence records |
| --- | ---: | ---: |
| Chapter 1 | 1 | 1 |
| Chapter 2 | 11 | 11 |
| Chapter 3 | 16 | 21 |
| Chapter 4 | 23 | 23 |
| Chapter 5 | 46 | 53 |
| total | 97 | 109 |

All 46 Chapter 5 rows were reconstructed exhaustively from the two binding
namespace sections: 38 atmosphere-continuum symbols and 8
synthesis-continuum symbols. All six Chapter 4 ownership rows were
independently expanded, and their package exports resolve to the exact pinned
definition modules. Every one of the 97 qualified targets has a matching AST
definition or annotated field at the raw-inventory source line and module
hash.

No accepted marker is an archive filename. The only repeated leaf among the
97 targets is `build_structured_atmosphere_from_columns`, and its pipeline and
synthesis qualified definitions remain separate. Wrong-module leaf
substitution, duplicate target insertion, altered authority location, and
altered alias binding are rejected. The independently canonicalized 109
evidence records have digest
`c00f6b9a6532eece6d702d248d4c56fec1f0bb353898578faffa134f606678be`.

## 31. Adversarial and mechanical results

All adversarial changes were made to in-memory copies. The complete proof
rejected a mutation to every one of its 19 projected record fields, every one
of the 15 embedded authority-evidence fields, all three exact-default
contract containers, and each of the 132 exact default entries. Independent
source checks also rejected a changed module digest, public-surface digest,
public-object digest, source spelling, constructor descriptor, field
descriptor, alias target, authority path, line, marker, line hash, qualified
target, namespace, or duplicate evidence record.

Mechanical verification passes:

- `python -m pytest -q tests/test_symbol_coverage.py`: 31 passed;
- Ruff check: passed;
- Ruff format check: both Python files already formatted;
- `py_compile`: passed for the builder and focused tests;
- `git diff --check`: clean for the four candidate files; and
- two independent raw rebuilds and two independent semantic-ledger rebuilds:
  byte-identical.

These checks establish strong drift detection. They do not test whether a
generic sentence actually enumerates the semantics of every source default.

## 32. P0 finding

### P0.1 — The default-bearing partition is complete, but its semantic reviews are not

The repair replaced missing reviews with either one residual sentence or the
pre-existing symbol responsibility. Several such strings omit exact default
routes that the audit explicitly required the fixed point to cover.

| Public callable | Pinned default-selected behavior | Recorded `default_branch_review` |
| --- | --- | --- |
| `payne_zero_atmosphere.install_runtime_data.install_initializer_assets` | `generated_manifest_path=DEFAULT_GENERATED_ASSET_MANIFEST` selects the manifest used to choose and hash the installed assets and is returned as provenance. An explicit path changes that source identity. | Mentions only `include_direct_xh` and `replace`; `generated_manifest_path` is absent. |
| `payne_zero_atmosphere.continuum_opacity.compute_molecular_hydrogen_population` | `hydrogen_departure_coefficient=None` supplies unity, while an explicit array changes the returned H2 population quadratically; `tables=None` loads the canonical molecular-equilibrium table while an explicit table injects alternate/parity data. | Only “accepted molecular continuum implementation.” |
| `payne_zero_synthesis.molecular_lines.load_catalog` | `rebuild=False` attempts a valid cache and reparses a corrupt/stale cache; `rebuild=True` forces source parsing. `cache_dir=None` selects the package runtime cache. | Only “source-bound molecular catalog load with optional compiled-cache reuse.” The atomic sibling has the exact branch matrix, but this routine does not. |
| `payne_zero_synthesis.pipeline.window_invariants_for` | Three default table paths select source identities, `metal_chunk=None` selects `SynthesisPipeline.METAL_CHUNK` and enters the cache key, and `PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1` selects fresh construction. | Only “window-invariant device-cache control.” |
| `payne_zero_synthesis.source_catalog_molecular_compiler.compile_molecular_text` | `use_energy_level_wavelengths` selects stored versus energy-derived wavelengths; `include_predicted_lines` rejects or retains negative-energy predicted records; both enter the cache identity. | Only “standard synthesis text-band compiler using exact resolution spelling and cache boundary.” |
| `payne_zero_atmosphere.convection.compute_disabled_convection_diagnostics` | `overshoot_weight` and `zero_top_layer_count` are forwarded into the disabled-convection diagnostic computation and change the retained endpoint diagnostic arrays. | Only “zero-flux diagnostic returned when convection is disabled.” |
| `payne_zero_atmosphere.synthesis_bridge.save_product_structured_atmosphere` | `molecular_lines=True`, `source_catalog_root=None`, `device='cpu'`, and `dtype='float64'` select the product chemistry, data root, backend, and precision passed to the structured-atmosphere builder. | Only “publish only final fixed-column-quantized converged arrays.” |

The residual 49-name registry has the same defect in at least
`install_initializer_assets`. The 59-name explicit set has multiple
independent counterexamples spanning physical, cache, environment,
diagnostic, source-identity, parity/injection, and algorithmic behavior.
Therefore the problem is not a demand that prose repeat harmless scalar
arguments; it is the exact class of branch-capable omission the fixed-point
repair was supposed to eliminate.

The focused test only requires `default_branch_review` to be non-empty (and
then seals that text by digest). It consequently proves stability of these
generic strings, not their semantic completeness.

**Required repair:** do not treat membership in
`EXPLICIT_DEFAULT_CALLABLES` as sufficient by itself. For every one of the 140
AST-derived names, bind each defaulted parameter to either its exact
physical/data/cache/environment/diagnostic/compatibility/parity/unsupported/
algorithmic effect or an explicit, source-checked statement that it is only a
continuous value or dependency injection and creates no such branch. The
examples above need exact callable-specific reviews and gates. Add
per-parameter coverage data or equivalent structured tests so a generic
non-empty sentence cannot satisfy the fixed-point gate.

## 33. Third-repair verdict

**REJECT P0.3 — reproducibility, proof integrity, the exact 140-name AST
surface, all nine named repairs, and the 97-name Chapter 1–5 authority join
pass, but the claimed semantic fixed point is still incomplete.**

- **P0:** default-selected physical, data, cache, environment, diagnostic,
  parity/injection, and algorithmic routes remain absent from both the
  49-name residual reviews and the 59-name “already explicit” reviews;
- **P1:** none at these bytes; the exact and exhaustive authority-join repair
  closes the preceding P1 findings; and
- **P2:** none at these bytes; hashes, source bindings, exact contracts,
  adversarial proof coverage, tests, formatting, compilation, and
  deterministic rebuilds pass.

Retain without rework: the deterministic inventory and ledger, complete
proof, all source fingerprints and alias bindings, the 32 exact default
contracts, the nine named branch repairs, the complete Chapter 1–5 authority
join, and the expanded adversarial tests.

<!-- END DELIMITED THIRD-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->

<!-- BEGIN DELIMITED FOURTH-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->

## 34. Fourth-repair audit scope and exact identities

This section is a new, preserved independent audit of the fourth repair. It
does not supersede or rewrite the preceding audit history. No candidate input,
accepted authority, chapter, Bible, coverage contract, data file, or pinned
Payne Zero source was edited during the audit.

| Audited object | SHA-256 |
| --- | --- |
| pinned Payne Zero commit | `9c44001feae40b85146630499e6f8a5fed42e5af` |
| raw inventory | `010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf` |
| prior independent audit before this append | `2ec799126b07c4e7d9bde1e8f0342d498b47bbc3731724fa81bd10206a71628c` |
| semantic override builder | `689bddb683b5b66e62156cb90acb8f9ce50067185dc618175540cc98577479e3` |
| focused semantic tests | `082acd6b4b38991bb1358ab988829161a03055b3ddff87f013a736353710daed` |
| canonical semantic ledger | `02684b4412059137abbc0fed232a85a9e272dbd786fea97fba0609af705fa671` |
| fourth-repair candidate report | `629db62267afe68cd32eddafdef2ed417cdf913f19b6c18a70eadc9e92f699d0` |

## 35. Deterministic rebuild and complete-ledger checks

The raw inventory was rebuilt twice from the read-only pinned checkout. Both
fresh files were byte-identical to each other and to the checked inventory,
with SHA-256
`010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf`.
Each run found 58 modules.

The semantic ledger was rebuilt twice from those independent inventories and
`COVERAGE.md`. Both fresh files were byte-identical to each other and to the
checked ledger, with SHA-256
`02684b4412059137abbc0fed232a85a9e272dbd786fea97fba0609af705fa671`.
Each run reported 58 mapped modules, 1,501 mapped public objects, and 1,501
reviewed overrides.

An independent record walk confirmed:

- 1,501 records and 1,443 distinct qualified names;
- 1,501 `reviewed_symbol_override` records and zero `module_default` records;
- the exact disposition counts 899 taught, 357 plumbing-only, 184 composed,
  41 diagnostic-only, 13 compatibility-only, and 7 unsupported;
- 58 duplicate export/definition qualified names, with no mismatch across
  their semantic fields;
- 163 ledger records for the 140 default-bearing qualified names, all carrying
  the exact expected `default_parameter_reviews` mapping; the 23
  duplicate-kind default-bearing names carry identical mappings; and
- all 197 true package-alias targets plus the separate public `__version__`
  record.

The requested canonical digests independently recompute:

| Registry or proof | SHA-256 |
| --- | --- |
| canonical raw-inventory content | `94861595dfe59afcbe2c47b23c19d14e25d0474a6727ad23e3c4f3948cc1b4b0` |
| reviewed source manifest | `b75d6f85f73d0ae14df5e908e785f46173681313407136cfa785804f9565eb32` |
| reviewed public source surfaces | `669a1afe23df89b4030aaf8ee1ad582409e771eff9b20b592ae20976fc60318b` |
| 140-callable source surface | `9947ad3a78ebb59fbe8d0f3e194bbb172e485099e24a2b9412f1a8ee362f5343` |
| 140-callable registry | `74008ed126f8e3a07f100769304e251acfbddc96ba91812cc1240726624a59ee` |
| 456-parameter source-default manifest | `6ee425f2d8dffe2c16ef40071d5ebe61a7c4c389580c144b789abf42da02e560` |
| 456-record semantic registry | `4f6e8365043d150cd7321be6331d24b95bf46aefec0ad7d4b2c0080afb147153` |
| API-alias target registry | `9589bbc3c4579ca80996f4f52649859d20ec146eff0b58dfd4af1347f8650964` |
| accepted Chapter 1–5 artifact manifest | `10c0e3bcd029834d97f480ed3e3e09d11b01126e36c116334f665e66bc28347b` |
| complete executable semantic proof | `71bd5ac41963bfffcc70318a2a84f7b66368067898b03a297cd20ea477634fea` |

The proof now projects `default_parameter_reviews` for every ledger record and
embeds both the source-default manifest and the semantic registry. Those new
proof fields are present and hash-bound.

## 36. Independent 140-callable/456-parameter fixed point

An independent AST scan covered every public module-level function,
constructor, and public method in all 58 source modules. It did not use the
candidate's 140-name registry to select its scope. It recovered exactly 140
default-bearing qualified names and 456 exact
`(qualified_name, parameter, source_default, default_ast)` tuples. There are
no extra or missing names or parameters, every lexical default and
attribute-free AST matches the candidate manifest, and every default parameter
has at least one direct use in its defining callable body.

All eleven declared categories occur. The complete category census, including
how the explanation was obtained, is:

| Effect category | Records | Callable/parameter overrides | Global-template records |
| --- | ---: | ---: | ---: |
| `physical` | 109 | 10 | 99 |
| `data/source identity` | 67 | 7 | 60 |
| `cache` | 13 | 7 | 6 |
| `environment` | 58 | 2 | 56 |
| `diagnostic` | 13 | 3 | 10 |
| `compatibility` | 18 | 0 | 18 |
| `parity/injection` | 88 | 15 | 73 |
| `unsupported` | 4 | 0 | 4 |
| `algorithmic` | 70 | 13 | 57 |
| `continuous-value` | 13 | 1 | 12 |
| `dependency-injection-with-no-branch` | 3 | 2 | 1 |
| **total** | **456** | **60** | **396** |

The 60 override keys were traced to their exact source use, branch, or
pass-through site. The previously named counterexamples now have useful custom
records: initializer installation, warm-start deck formatting, both H2
population helpers, molecular-catalog cache behavior, window-invariant
identity, molecular text compilation, disabled-convection diagnostics, and
the structured-atmosphere product bridge. The nine preceding branch repairs
also remain represented: sampled-continuum materialization, H-minus departure,
the all-eight-array convection fallback, molecular finite-difference
sampling/restoration, transfer finalization, asymmetric/symmetric convergence,
both forced-prewarm routes, and the atomic-catalog parser/cache matrix.

The exact and exhaustive Chapter 1–5 join also remains valid: 97 qualified
names produce 109 duplicate-preserving ledger evidence records, partitioned as
1/11/16/23/46 qualified names across Chapters 1–5. Every ledger evidence
object equals the exact authority join.

## 37. P0 findings

### P0.1 — 396 records are category templates, not parameter-specific semantic reviews

The candidate report says every exact key is joined to a
“callable/parameter-specific explanation.” That is not how 396 of the 456
records are constructed.

`_default_parameter_effect` first assigns a category solely from the
source-spelled **parameter name**. It then selects one of eleven global
sentence templates. The template interpolates only the readable parameter
name and the callable's leaf name. Although the function accepts
`source_default`, it never uses that argument. A later wrapper prepends the
qualified key and lexical default.

An exhaustive comparison confirmed that all 396 non-override explanations are
byte-for-byte instances of those eleven templates. The qualified prefix makes
them unique strings, but it does not make their claimed effects
source-specific. Examples include:

- every compatibility default receives “compatibility spelling, validation,
  or public error surface,” without saying which of those three effects
  occurs;
- every non-overridden parity input receives “internal construction versus
  caller injection ... for the parity path,” without identifying the
  construction, coupling, ignored/required cases, or output effect;
- every non-overridden algorithmic input receives the same five-way
  “route, termination rule, domain, ordering, or discrete mode” disjunction;
- every data path receives “alternate declared data,” without naming the
  canonical fallback, file family, provenance, or cache identity; and
- every non-overridden diagnostic input receives a four-way
  “diagnostic, provenance, assertion, or retained-output surface” disjunction.

These are exactly the generic placeholders this audit was required to reject.
They frequently classify the broad family correctly, but they do not state
which behavior the exact source default selects.

Concrete source traces show material omissions:

- `standard_microturbulence.requested_maximum_velocity=-99.0e5` is a sentinel:
  equality selects a gravity/temperature-table interpolation, while every
  other value selects `abs(requested_maximum_velocity)`. Its physical template
  does not mention either route.
- `decode_selected_line_words.detect_swapped_layout=True` performs and scores
  a second pair-swapped decode; false skips that potentially very expensive
  pass. Its compatibility template does not identify the layout or the
  alternate decode.
- `molecular_chunk_lines.default=CHUNK_LINES` is the fallback passed through
  the `PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES` environment override and
  positive-integer clamp. Its algorithmic template identifies none of that
  behavior.
- `resolve_runtime.requested_dtype=None` chooses `DEFAULT_DTYPE` on MPS and
  `REFERENCE_DTYPE` elsewhere, and explicit MPS float64 raises. Its environment
  template does not state this coupled device/dtype route.
- `solve_molecular_equilibrium.return_diagnostics=False` changes the returned
  result shape; true constructs the exact iteration/species diagnostic
  mapping. Its diagnostic template does not say what is returned.

### P0.2 — Several `continuous-value` “no discrete route” claims are false

The continuous template goes beyond being vague: it asserts that the
parameter “does not by itself select a discrete implementation route.” Direct
source branches contradict that assertion.

- `apply_temperature_correction.mixing_length` is tested by
  `if float(mixing_length) > 0.0 and velocity_coefficient > 0.0`.
  `finalize_transfer_state.mixing_length` is forwarded into that exact
  calculation. Both records are templated as continuous values with no
  discrete route.
- `deterministic_initializer_labels.jitter_scale` rejects non-finite or
  negative values and has the coupled branch
  `max_trials > 1 and jitter_scale == 0.0`, which raises instead of producing
  retry labels. `solve_structured_atmosphere.initializer_jitter_scale` is
  forwarded to that routine. Both records carry the same false no-route
  assertion.

The remaining continuous records were traced as normalization floors,
hydrostatic constants, smoothing coefficients, or optical-depth-grid
coordinates. Even where the broad category is defensible, the global sentence
still omits the actual equation or output it changes.

All three `dependency-injection-with-no-branch` records were also traced.
`fast_exponential_lookup` chooses supplied tables or
`build_fast_exponential_tables`; `evaluate_voigt_profile` chooses a supplied
basis or `build_voigt_profile_basis`; and
`molecular_hydrogen_equilibrium_constant` chooses supplied profile tables or
`load_hydrogen_line_profile_tables`. The first two have useful custom prose.
The third is the same global “canonical tables dependency” template and does
not identify the hydrogen-profile loader, H2 partition interpolation, or
equilibrium result it changes.

### P0.3 — The adversaries seal text but do not validate its meaning

The 1,824 delete/misname/reclassify/source-mutation cases do execute and fail.
However, all four mutations for every record are expected to fail first with
“default-parameter semantic review registry changed”: the test mutates the
registry without recomputing its pinned digest. This is a strong tamper seal,
not a semantic category or explanation oracle.

The one digest-recomputed “generic” adversary contains the literal blacklist
terms `generic`, `placeholder`, and `has been reviewed`. The production
templates simply avoid those words.

Two additional in-memory adversaries establish the gap:

1. replacing the custom
   `install_initializer_assets.generated_manifest_path` explanation with the
   candidate's own `data/source identity` global template and recomputing the
   expected registry digest is **accepted** by
   `_validate_default_parameter_reviews`; and
2. falsely reclassifying
   `format_warm_start_deck.metallicity` from `physical` to `cache`, while
   leaving all eleven categories represented and recomputing the digest, is
   also **accepted**.

The validator checks exact keys, source binding, category vocabulary, a
qualified prefix, minimum length, and a short prohibited-word list. It does
not bind a category or explanation to source behavior. The proof then
faithfully seals whatever text passed that structural check.

## 38. Mechanical and retained-pass results

The focused suite and all requested static checks pass:

- `python -m pytest -q tests/test_symbol_coverage.py`: 34 passed;
- all 1,824 declared per-record mutations executed inside that suite;
- Ruff check: passed;
- Ruff format check: both Python files already formatted;
- `py_compile`: passed for the builder and focused tests;
- `git diff --check`: clean for the candidate files and this audit; and
- two raw rebuilds and two semantic-ledger rebuilds: byte-identical.

Retain without rework: deterministic inventory and ledger generation; the
exact 140-callable/456-parameter source manifest; all source, surface, object,
default, alias, authority, and proof fingerprints; duplicate-kind mapping
consistency; the complete 97-qualified-name/109-record authority join; the
nine previously repaired branch facts; the 60 genuinely custom records where
their source trace remains correct; the embedded per-record mappings; and the
expanded proof surface.

## 39. Fourth-repair verdict

**REJECT P0.3 — the syntactic 456-parameter fixed point and all integrity
machinery pass, but the semantic fixed point does not.**

- **P0:** 396 of 456 explanations are generated from eleven category-wide
  templates keyed primarily by parameter spelling, rather than stating the
  exact source-selected behavior;
- **P0:** at least four `continuous-value` records make a directly false claim
  that the parameter selects no discrete route;
- **P0:** digest-recomputed generic substitution and false category
  reclassification are accepted, so the 1,824 digest-mismatch adversaries do
  not establish semantic correctness;
- **P1:** none at these bytes; the exact Chapter 1–5 authority join remains
  closed; and
- **P2:** none at these bytes; reproducibility, source/default/alias/proof
  bindings, record counts, tests, formatting, compilation, and whitespace
  checks pass.

Required repair: replace the 396 category-template records with explicit
`(qualified_name, parameter)` reviews that name the concrete default route,
equation, fallback, validation condition, cache/source identity, returned
surface, or injected dependency found in the pinned source. Continuous-value
records must name every direct or coupled branch instead of asserting none.
Dependency-injection records must name the constructed dependency and the
observable effect of an injected object. Keep the exact source manifest and
integrity seals, but do not present a digest, qualified prefix, or prohibited
word list as evidence that the sealed prose is semantically source-faithful.

<!-- END DELIMITED FOURTH-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->

<!-- BEGIN DELIMITED FIFTH-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->

## 40. Fifth-repair independent audit — repair Section 14 (2026-07-30)

Status: **independent hostile re-audit; REJECT**

This section is a fresh audit of candidate Section 13. It does not inherit the
candidate author's acceptance claims. Sections 1–39 remain history. The
triggering fourth-repair audit was byte-identical at audit start:
`eafe4b5da285303f69dc3e59bd4b1553e9413e35f3f31db09b6637ce3b7c0e78`.

### Exact candidate bytes

| object | independently observed SHA-256 |
| --- | --- |
| pinned Payne Zero checkout commit | `9c44001feae40b85146630499e6f8a5fed42e5af` |
| semantic override builder | `5f16ef336e43cd9064aa781769b4d96984ab50d75fb9b442fc3e36e7d99a0e42` |
| focused semantic tests | `dc5ef87c67b9e04581a57febe90b5302ab1f07bb0efdb994e16d9fb01c407f81` |
| compact default-parameter authority | `142149023e566dcae55ae189a4eea52985aac729642cc87b5fc5ec87457f4692` |
| checked semantic coverage ledger | `a03253794a45b71225fd229f0bfe7ef0c7c02f7035cc132d4b4912511a30db8e` |
| candidate report | `8388eacc229ae83614ce2948bd63d863ca907afc0e85937e2a2bde134a35b011` |

The checkout was clean at the pinned commit. No source-tree write was made.
The ledger named “semantic ledger” in the candidate is the repository file
`audit/paynezero_symbol_coverage.json`.

### Retained structural and source-evidence passes

Independent JSON traversal and source-backed execution retain these results:

- 58 modules, 1,501 public records, 1,443 distinct qualified names, no
  `module_default` record, and no empty primary location;
- 942 reviewed-module records, 309 explicit-symbol records, 52
  explicit-default-callable records, and 198 exact API-alias records;
- exactly 140 distinct default-bearing callables and 456 distinct
  `(qualified_name, parameter_name)` contracts;
- a sorted, duplicate-free 456-record authority with 456 distinct complete
  semantic-body digests and no `shared_semantics_with` alias;
- 40 defining source files, 895 direct parameter-load records, 430 predicate
  records (302 direct and 128 coupled), and 1,052 call-consumer records (688
  direct and 364 coupled);
- eight declared forwarding edges, all resolving to a real target callable
  and parameter in the pinned source;
- the frozen category census of 109 physical, 67 data/source-identity, 13
  cache, 58 environment, 12 diagnostic, 19 compatibility, 88
  parity/injection, 5 unsupported, 69 algorithmic, 13 continuous-value, and 3
  dependency-injection-with-no-branch contracts; and
- the accepted Chapter 1–5 join remains 97 exact qualified names / 109 ledger
  records: Chapter 1 is 1/1, Chapter 2 is 11/11, Chapter 3 is 16/20, Chapter 4
  is 23/23, and Chapter 5 is 46/54 (qualified names / records).

The focused suite executed, rather than merely being read, and passed all 36
tests in 88.47 seconds. Its forged-line adversary fails against the private
pinned-AST reconstruction, and its no-branch attack fails against the
recomputed predicate evidence. The nine earlier branch repairs remain present.
Direct source inspection also confirms the fifth candidate's corrected
microturbulence sentinel, swapped-layout decode, chunk fallback/environment
clamp, coupled device/dtype policy, diagnostic return arity, mixing-length
threshold, jitter validation/retry, H-profile table-loader, `r_grid` alias,
chain restart, sampled-continuum invariant, spectral-operator, and
saturated-core descriptions.

Two fresh raw-source inventories were byte-identical to each other and to the
checked inventory, all at
`010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf`.
Ruff check, Ruff format check, Python compilation, and pre-append whitespace
checking passed.

These retained passes do not establish the required semantic fixed point.

### P0.1 — A source-specific record is directly false

The record for
`payne_zero_synthesis.pipeline.SynthesisPipeline.__init__.source_path`
states:

> The exact default `None` is consumed by L1146 `np.load(source_path)`.

The pinned source does the opposite. Line 1144 assigns
`self._source_from_ref = source_path is not None`; line 1145 branches on that
attribute. With the default `None`, the condition is false, `np.load` is not
called, and lines 1157–1159 set both `_line_source` and
`_line_scattering_ref` to `None`. Only a non-`None` alternate loads the exact
reference source/scattering arrays. The standard route later rebuilds the LTE
Planck line source.

The evidence record itself exposes why the semantic generator missed this:
it records the line-1144 use and line-1146 call but has an empty
`branch_predicates` list and only `["source_path"]` in
`parameter_flow_names`. Its simple-name flow does not follow the assignment
into `self._source_from_ref`, so the subsequent attribute predicate is absent.
The record then promotes a guarded alternate-only call into the claimed
default route. Its validation sentence also says no separate predicate exists
in the defining callable even though the line-1145 predicate is exactly what
selects the route.

This is not a stylistic objection. At least one of the 456 claimed exact
contracts gives the wrong runtime behavior, and the recomputed AST evidence is
not complete enough to detect it.

### P0.2 — Large parts of the authority remain stock templates

Literal storage in a JSON authority and uniqueness of each *whole* body do not
make every semantic field source-specific. Exact recurrence counts show:

- `branch_behavior = "forwarded or expression consumer"` appears in 243
  records spanning 83 callables;
- `validation_and_coupling = "No separate predicate validates this parameter
  in the defining callable; the exact typed consumer/use evidence is
  authoritative."` appears in 219 records spanning 79 callables; and
- `alternate_route = "An explicit alternate replaces None in those exact
  callee arguments and changes the consumer input."` appears verbatim in 117
  records spanning 50 callables and six effect categories.

The last group alone includes physical, data/source-identity, environment,
diagnostic, compatibility, and parity/injection parameters. It names neither
the alternate value nor its selected behavior. For example,
`payne_zero_atmosphere.cli.main.argv` does not say that `None` makes
`argparse` read the process command line while an explicit sequence replaces
that input. The wrapper records for `spectral_operator` do not carry the
native-grid identity versus total/continuum convolution and normalized-ratio
effect established at the pipeline target. The `wing_mode` record only says
that an alternate changes a callee input; it does not state batched standard
deposit versus loop parity.

Whole-body hashes are distinct because callable names, source snippets, and
consumer strings differ around these repeated fields. That property does not
close the fourth audit's requirement for concrete default, alternate,
validation, consumer, and observable-effect semantics at every exact key.
Candidate Section 13's assertion that no record contains global-template
prose is therefore false at the candidate bytes.

### P0.3 — Rebinding both semantic surfaces defeats the validator

Two independent in-memory adversaries started from a fresh import, preserved
all source/default evidence, and recomputed every affected published digest.
They modified no candidate file.

1. For
   `payne_zero_atmosphere.atmosphere_io.parse_atmosphere_deck.source`, the
   adversary replaced `branch_behavior`, `default_route`, `alternate_route`,
   `validation_and_coupling`, `consumer`, and `observable_effect` with
   category-level statements such as “use the source default for this
   diagnostic category” and “changes the routine result.” It updated both
   `EXPLICIT_DEFAULT_PARAMETER_CONTRACTS` and
   `DEFAULT_PARAMETER_REVIEWS`, rebound the exact key in
   `EXPECTED_DEFAULT_PARAMETER_EFFECT_CONTRACTS` to the new body hash, and
   recomputed all three registry digests. The generic body remained unique.
   `_validate_default_parameter_reviews` returned normally:
   `GENERIC_REBOUND_ACCEPTED`.
2. For
   `payne_zero_synthesis.molecular_lines.molecular_chunk_lines.default`, the
   adversary falsely changed `effect_category` from `algorithmic` to
   `diagnostic`, updated both main semantic mappings, rebound the independent
   expected `(category, body hash)` pair, and recomputed all three digests.
   `_validate_default_parameter_reviews` again returned normally:
   `FALSE_CATEGORY_REBOUND_ACCEPTED`.

The candidate tests rebind the main semantic digests but deliberately leave
the independent expected registry frozen. They prove that a second literal pin
detects drift from itself; they do not prove that the validator rejects a
category-template or false-category rebound around both claimed semantic
surfaces. Source evidence constrains locations and syntax, but no production
rule binds most effect categories or prose meanings to source behavior.

Forged AST evidence and a literal “no-branch” statement are correctly rejected
by separate source rules. Those useful guards do not rescue generic prose or a
false category that avoids the two narrow checks.

### P1.1 — The checked ledger is not a canonical byte rebuild

Two fresh canonical runs of:

```text
python scripts/build_symbol_coverage.py \
  --inventory audit/paynezero_symbols.json \
  --ledger COVERAGE.md \
  --output <temporary-path>
```

were byte-identical to each other at
`5f8752914f2d2d68f7b5dfad9ff0c048921ae8fecb009ae074f1b38fd54c9bf7`,
but neither matched the checked ledger at
`a03253794a45b71225fd229f0bfe7ef0c7c02f7035cc132d4b4912511a30db8e`.

The exact diff is eight accepted Chapter 5 `lane` values. The checked ledger
contains a literal U+2014 em dash, whereas the canonical builder's default
`json.dumps` serialization emits `\u2014`. The parsed JSON objects are equal,
which is why the focused test's object comparison passes, but the files are
not byte-identical. Candidate Section 13's explicit byte-rebuild claim is
false, and the checked generated artifact is not reproducible through its
declared canonical command at these bytes.

### Fifth-repair verdict

**REJECT P0.3.**

- **P0:** the `SynthesisPipeline.__init__.source_path` default contract states
  an alternate-only `np.load` call as the default route and misses the actual
  attribute-mediated branch;
- **P0:** hundreds of fields remain verbatim stock templates, including 117
  alternate-route statements that identify no concrete alternate behavior;
- **P0:** after both semantic mappings, the independent expected mapping, and
  all associated digests are rebound, the validator accepts both generic
  category prose and a known false category;
- **P1:** two canonical ledger builds agree with each other but not with the
  checked ledger, contradicting the candidate's byte-identical rebuild claim;
  and
- **retained:** the 1,501/1,443/0 ownership fixed point, 140/456 default
  surface, source/default/evidence hashes, 97-name/109-record authority join,
  nine earlier branch repairs, high-risk corrected routes, raw inventory
  reproducibility, 36 focused tests, formatting, compilation, and source-tree
  cleanliness remain valid.

Required repair: replace stock component sentences with concrete
key-specific behavior; trace branch flow through attributes and other guarded
derived state; correct `SynthesisPipeline.__init__.source_path`; add
adversaries that rebind *both* semantic expectation surfaces; bind categories
and required behaviors to independently reviewed source facts rather than a
second mutable hash registry; and make the declared canonical builder reproduce
the checked ledger bytes exactly.

<!-- END DELIMITED FIFTH-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->

<!-- BEGIN DELIMITED SIXTH-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->

## 41. Sixth-repair independent audit — repair Section 15 (2026-07-30)

Status: **independent hostile re-audit; REJECT**

This section audits candidate Section 14 from the exact sixth-repair bytes. It
does not reuse the candidate author's semantic conclusions. Sections 1–40
remain history. The triggering fifth-repair audit was unchanged at audit
start:
`2a6902a6feb7734e581a5db92d4cc8b192702204ac4a5378cd5044def374d9a1`.

### Exact candidate identities

| object | independently observed SHA-256 |
| --- | --- |
| pinned Payne Zero checkout commit | `9c44001feae40b85146630499e6f8a5fed42e5af` |
| semantic override builder | `34320c62b3eb27560890cc956885fed378258458885685916c42ed51404a816d` |
| focused semantic tests | `cce0fa9909fb0f4dc51e9b60879097ae894468e72b1aec158dab34b7a6942952` |
| schema-v2 source-fact authority | `f5a31c24b86b804a171258f0b342b3474a42a2cfda936635beb21177a462ae23` |
| checked semantic coverage ledger | `7f72ebc2bb49e87ad6caa2aff3215ed47e31626bbed94253395809ff407bc7c5` |
| candidate report | `cad01924ae3ee861f52286c4222678335e1988de3271abe0dd361f5f9c37beb6` |

The source checkout was clean at the pinned commit before and after the audit.
No Payne Zero, paper, chapter, governing, candidate, or data file was modified.

### Structural, authority, and reproducibility passes

Independent traversal retains the structural fixed points:

- 58 modules, 1,501 public records, 1,443 distinct qualified names, zero
  `module_default` records, and zero empty primary locations;
- 140 distinct default-bearing callables and exactly 456 default parameters;
- a schema-v2 authority containing 456 sorted, duplicate-free records and
  only `qualified_name`, `parameter_name`, `category_anchor`, `effect_role`,
  and `forwarded_to`;
- 235 call anchors, 207 predicate anchors, and 14 use anchors, each
  byte-matching the independently reconstructed pinned evidence selected by
  `_anchor_from_source_evidence`;
- 40 source files, 895 load records, 431 combined predicates (302 direct and
  129 coupled), one separately identified attribute predicate, 1,054 call
  records (688 direct and 366 coupled), and 242 derived assignments (135
  direct and 107 coupled);
- the 97-qualified-name / 109-record accepted Chapter 1–5 join: Chapter 1
  1/1, Chapter 2 11/11, Chapter 3 16/20, Chapter 4 23/23, and Chapter 5 46/54
  (qualified names / records);
- the eleven-role census remains exactly 109 physical, 67 source identity, 13
  cache, 58 runtime environment, 12 diagnostic, 19 compatibility, 88
  parity/injection, 5 unsupported, 69 algorithm control, 13 continuous
  equation, and 3 dependency injection records; and
- the nine earlier named branch repairs remain present.

The `SynthesisPipeline.__init__.source_path` repair itself passes. The evidence
now contains the L1144 binding to `self._source_from_ref`, the coupled L1145
attribute predicate, the true-polarity guard on L1146 `np.load`, and the
correct `None` no-I/O/LTE versus explicit-reference routes.

Fresh hostile probes also retain useful integrity properties:

- a category-anchor line mutation is rejected against the private AST;
- deleting the public `source_path` attribute binding/predicate evidence while
  rebinding its public digests is rejected against the private AST;
- a digest-recomputed `no-branch` replacement for the microturbulence sentinel
  is rejected by fresh source-fact rendering;
- the suite's generic-prose, ordinary false-category, path, public role-map,
  public description-map, registry, and digest rebounds are rejected by the
  literal fresh authority read; and
- forged public use evidence is rejected by the private source snapshot.

The focused suite passed all 38 tests in 51.81 seconds. Ruff check, Ruff format
check, Python compilation, and bounded whitespace checking passed.

Two new raw inventories were mutually and canonically byte-identical at
`010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf`.
Two new canonical coverage builds were byte-identical to each other and to the
checked ledger at
`7f72ebc2bb49e87ad6caa2aff3215ed47e31626bbed94253395809ff407bc7c5`.
The ledger has zero literal U+2014 bytes and exactly eight `\u2014` escapes.
The fifth audit's canonical-byte failure is closed.

These are genuine improvements, but they seal and reproduce a semantic
mapping whose rendered meanings are not yet correct.

### P0.1 — Two required dependency defaults are rendered incorrectly

The pinned source is unambiguous:

- `fast_exponential_lookup.tables=None` reaches line 155
  `lookup = tables or build_fast_exponential_tables()`. Because `None` is
  falsey, the default deterministically calls
  `build_fast_exponential_tables`; a truthy supplied object bypasses that
  constructor and supplies the two lookup arrays.
- `evaluate_voigt_profile.basis=None` reaches line 329
  `profile_basis = basis or build_voigt_profile_basis()`. The default
  deterministically builds the canonical basis; a truthy supplied basis
  bypasses construction and supplies the Gaussian/correction arrays.

Both schema-v2 contracts instead state that the exact `None` default “also
depends on runtime state” and that runtime state chooses between two
`<fall-through>` heads. Their observable-effect fields repeat that false
indeterminacy. The renderer sees the right source expressions but its partial
evaluator treats the right-hand constructor call as unknown and fails to
apply Python's known false-left short-circuit rule.

These are two of the three dependency-injection routes singled out by the
fourth and fifth audits. The focused test now checks only that the constructor
name occurs somewhere in source-fact text and rendered text; it does not check
that the rendered default selects that constructor. Thus the test passes
while the required default/alternate meaning has regressed.

The same evaluation defect is broader. Seventy-one contracts say that an
exact default retains two `<fall-through>` heads, and 65 observable effects
say runtime state chooses between them. For example, `None or {}` has a
deterministic right-hand empty-mapping result, not two unidentified runtime
routes. Exact source interpolation is not equivalent to evaluating the
default route.

### P0.2 — Previously required forwarded semantics have been discarded

The schema-v2 `forwarded_to` edges verify only that a call target and parameter
exist. The renderer does not incorporate the target parameter's semantics.
Several explicitly required high-risk records therefore regress to an
argument-level source tour:

- `molecular_chunk_lines.default` says only that `CHUNK_LINES` is passed to
  `_env_positive_int`. It omits that the environment variable's absent,
  empty, and invalid-text cases return the fallback, while a valid integer is
  clamped by `max(1, value)`.
- `synthesize_from_labels.r_grid` says only that `None` is passed to
  `_resolved_r_grid`. It omits the 20,000 dual-`None` fallback, exact-equality
  rule for dual specification, and finite-positive validation.
- `solve_structured_atmosphere.initializer_jitter_scale` says only that 0.01
  is float-coerced and forwarded. It omits the target's finite/non-negative
  guard, zero-with-retries error, and retry-displacement equation.
- `finalize_transfer_state.mixing_length` says only that 1.0 is forwarded to
  `apply_temperature_correction`. It omits the target's coupled positive
  threshold and the wrapper's internal-convection configuration override.

All four behaviors were concrete requirements of the fifth repair. A
forwarding edge is valuable evidence, but it cannot replace the observable
default, alternate, validation, and output effect at the public wrapper.

### P0.3 — The source-operation role oracle accepts other false categories

Candidate Section 14 claims categories are determined by concrete
source-operation facts. The production check is not unique: it is principally
a parameter-name/token classifier, and multiple roles can pass for the same
unchanged source anchor.

Two independent unchanged-anchor adversaries demonstrate the gap:

1. Changing
   `payne_zero_synthesis.api.synthesize_from_labels.resolution` from
   `compatibility_route` to `algorithm_control` is accepted by
   `_contracts_from_fact_authority` and renders category `algorithmic`. This
   spelling is the compatibility alias; `r_grid` is the canonical algorithmic
   control.
2. Changing
   `payne_zero_atmosphere.line_profile_math.fast_exponential_lookup.tables`
   from `dependency_injection` to `parity_or_injection` is also accepted and
   renders category `parity/injection`, despite the exact dedicated
   dependency-injection classification.

The final validator rejects a changed in-memory authority because it rereads
the frozen on-disk authority. That is a strong tamper seal, but it does not
show that the sealed role follows uniquely from the anchor. The candidate
test uses only `molecular_chunk_lines.default: algorithm -> diagnostic`, a
pair chosen so the token whitelist fails. It does not adversarially search the
other ten roles or require exactly one supported role per key.

Consequently, false-category rejection still depends on accepting the
authored authority as its own semantic oracle. The schema-v2 anchor proves
where a parameter is used, not why only one of several overlapping role
labels is correct.

### P0.4 — Full-string uniqueness hides rendered stock prose

All six narrative fields do have 456 distinct complete strings. That count is
achieved by interpolating unique line numbers, snippets, and consumers into a
small set of renderer templates. It is not evidence that every observable
effect states a source-specific physical or operational consequence:

- 211 of 456 observable effects end with the identical conclusion
  “that argument-level change is the exact observable source route”; and
- another 206 end with the identical branch-head/consumer conclusion that
  those items “are the observable route difference.”

These 417 records commonly stop at a callee argument or branch head rather
than explaining the selected data, equation, cache, return shape, compatibility
meaning, or downstream output. The regressions in P0.1 and P0.2 are concrete
examples. A recurrence census over whole interpolated strings therefore
cannot enforce the fourth audit's ban on category/global template semantics.

### Sixth-repair verdict

**REJECT P0.3.**

- **P0:** two required no-branch dependency defaults falsely say runtime state
  chooses their route instead of deterministically constructing canonical
  tables/basis for `None`;
- **P0:** the chunk environment/clamp, `r_grid`, jitter, and forwarded
  mixing-length contracts have regressed from concrete behavior to
  argument-level forwarding;
- **P0:** unchanged-anchor false categories are accepted for compatibility
  resolution and dependency tables, so the source-operation role check is
  not a unique semantic oracle;
- **P0:** 417 observable-effect records still terminate in two stock renderer
  conclusions; full-string uniqueness from interpolated source text does not
  establish source-specific meaning;
- **P1/P2:** none at these bytes; the canonical serialization defect is fixed,
  and structural, source, formatting, compilation, and whitespace gates pass;
  and
- **retained:** 58 modules; 1,501/1,443/0 ownership; 140/456 defaults; schema-v2
  anchor identity; the expanded evidence census; 97/109 accepted authority;
  nine named branch repairs; the corrected `source_path` attribute flow;
  ordinary tamper/rebound rejection; 38 focused tests; and byte-identical raw
  inventory and ledger rebuilds.

Required repair: evaluate Python short-circuit/default selectors accurately;
restore concrete semantics for every forwarded high-risk target rather than
stopping at its call edge; make role support exclusive or bind each role to an
independently reviewed operation-specific fact that competing roles cannot
satisfy; exhaustively adversarially try all ten wrong roles at all 456 keys;
and test semantic outcomes (canonical constructor, fallback, clamp, validation,
and observable result), not only the presence of source fragments or uniqueness
of rendered strings.

<!-- END DELIMITED SIXTH-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->

<!-- BEGIN DELIMITED SEVENTH-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->

## 42. Seventh-repair independent audit — repair Section 16 (2026-07-30)

Status: **independent hostile re-audit; REJECT**

This section audits candidate Section 15 from the exact seventh-repair bytes.
It does not reuse the candidate author's semantic conclusions. Sections 1–41
remain immutable history. The triggering sixth-repair audit was unchanged at
audit start:
`a6d82c10cdbd04e101e39a12fcf6bd600096d07c35b89ca0962e342ef09d97d5`.

### Exact candidate identities

| object | independently observed SHA-256 |
| --- | --- |
| pinned Payne Zero checkout commit | `9c44001feae40b85146630499e6f8a5fed42e5af` |
| semantic override builder | `c2e00e10caf93acac2dde2a47a65d3d7bb08bd17628f71917c93b0a6efea43a4` |
| focused semantic tests | `8ed79fc6fbe5cc0ed92b964af6671a2a3ee90d94adea4a48a1e4341f48d7565a` |
| schema-v2 source-fact authority | `f5a31c24b86b804a171258f0b342b3474a42a2cfda936635beb21177a462ae23` |
| checked semantic coverage ledger | `35314bcb7087592abce2ab9f589246262061049cfaa0dec9504796f1b70a9afe` |
| candidate report | `66051c6da616226334afb1a05c497d7f8bc5d9c035a437566a7e60b84bbcb4f9` |

The Payne Zero checkout was clean before and after review. No candidate,
governing, chapter, data, paper, or pinned-source object was modified.

### Retained structural, source, and reproducibility passes

Two independent raw-inventory builds were byte-identical to one another and
to `audit/paynezero_symbols.json` at
`010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf`.
Two independent canonical semantic builds were byte-identical to one another
and to the checked ledger at
`35314bcb7087592abce2ab9f589246262061049cfaa0dec9504796f1b70a9afe`.
Each build reported 58 modules, 1,501 mapped public objects, and 1,501
reviewed overrides.

Independent traversal retains:

- 1,501 records, 1,443 qualified names, zero `module_default` records, and
  1,501 `reviewed_symbol_override` records;
- 899 taught, 357 plumbing-only, 184 composed, 41 diagnostic-only, 13
  compatibility-only, and 7 unsupported records;
- 140 default-bearing callables with exactly 456 sorted, duplicate-free
  parameter facts;
- 40 source files, 895 parameter loads, 431 predicates (302 direct and 129
  coupled), one attribute-mediated predicate, 1,054 call forwards (688 direct
  and 366 coupled), and 242 derived bindings (135 direct and 107 coupled);
- 97 authority-qualified names and 109 joined records: Chapter 1 1/1,
  Chapter 2 11/11, Chapter 3 16/20, Chapter 4 23/23, and Chapter 5 46/54;
- the exact eleven-role census: 109 physical, 67 source identity, 13 cache,
  58 runtime environment, 12 diagnostic, 19 compatibility, 88
  parity/injection, 5 unsupported, 69 algorithm control, 13 continuous
  equation, and 3 dependency injection; and
- source-evidence, contract, merged-review, and complete-proof digests
  `3256470c…`, `001aacf2…`, `afb6b87d…`, and `43e2aac2…`.

All eight declared forwarding edges were independently reconstructed from the
pinned target source. Their 185 operations have the claimed
12/23/14/14/103/7/9/3 split for jitter, mixing length, `r_grid`, `resolution`,
frequency invariants, environment fallback, spectral operator, and
saturated-core guard respectively.

The concrete repairs for both `None or build_*()` routes are factually
correct: `None` deterministically constructs the canonical exponential tables
or Voigt basis, a truthy injected object bypasses construction, and a falsey
injected object follows Python `or` to the constructor. The chunk
missing/empty/invalid fallback and `max(1, value)` clamp, `r_grid` 20,000
fallback/equality/finite-positive checks, jitter finite/non-negative and
retry-displacement behavior, forwarded mixing-length override and equations,
`source_path=None` no-I/O LTE route, and the hydrogen-table loader route also
match the pinned source.

The focused suite passed all 40 tests in 220.78 seconds. Ruff check, Ruff
format check, bytecode-disabled compilation, and bounded whitespace checks
passed. The canonical ledger contains zero literal U+2014 bytes and exactly
eight `\u2014` escapes.

These retained passes do not establish the semantic exclusivity or prose
correctness required by P0.3.

### P0.1 — The 4,560 test measures precedence, not exclusive role support

An independent authority parse and exact 456-by-10 mutation reproduced the
advertised result: all 4,560 wrong authority labels are rejected by
`_validate_source_derived_effect_role`. Rebinding, cross-permuting, and
in-place corrupting the two public role-description/category maps does not
change that result.

The underlying predicate is not exclusive. Calling
`_effect_role_supported_by_source` for all eleven roles at every key finds 417
keys with one supported role and **39 keys with two supported roles**. The 39
collisions comprise:

| simultaneously supported roles | keys |
| --- | ---: |
| physical + source identity | 9 |
| source identity + parity/injection | 9 |
| cache + algorithm control | 5 |
| physical + algorithm control | 3 |
| diagnostic + algorithm control | 2 |
| parity/injection + algorithm control | 2 |
| parity/injection + dependency injection | 2 |
| physical + continuous equation | 2 |
| source identity + algorithm control | 2 |
| algorithm control + continuous equation | 1 |
| compatibility + algorithm control | 1 |
| physical + diagnostic | 1 |

The collision list's canonical SHA-256 is
`fd5885e38eaa7f9ee277ef381821e6f504a58fc52e7fcde1006065bae2d9b8dd`.
It includes both named repaired examples:
`synthesize_from_labels.resolution` supports compatibility and algorithm
control, while `fast_exponential_lookup.tables` supports dependency injection
and parity/injection. It also includes, among others,
`initializer_jitter_scale` (algorithm + continuous),
`include_direct_xh` (source identity + algorithm),
`temperature_iteration_index` (cache + algorithm),
`overshoot_weight` (physical + diagnostic), eight `molecules_path` keys
(physical + source identity), and the pipeline table paths (source identity +
parity/injection).

`_source_derived_effect_role` merely returns the first supported role in a
hard-coded precedence tuple. The hostile test then compares each wrong label
with that single return value; it never requires the other ten support
predicates to be false. Emptying all operation-evidence lists preserves 451 of
456 authority roles. Cyclically permuting complete evidence among keys
preserves 452 of 456. Conversely, neutralizing the parameter name makes 454
keys unclassifiable. The algorithm, physical, parity, and much of the runtime
and diagnostic logic is therefore a signature-token classifier, not an
exclusive consequence of source operations. Several nominal source checks
are tautological because the qualified and parameter names are inserted into
`source_facts` before those same tokens are sought.

Thus the exact 4,560 result is mechanically true but does not prove the
candidate's claimed exclusive source-derived oracle.

### P0.2 — Normalized prose still conflates different operations and effects

Applying the production normalizer independently to all 456
`observable_effect` fields yields **146** frames, not the candidate report's
exact claim of 145, with maximum recurrence 35. More importantly, the
repetitions are not confined to genuinely identical operations.

The 35-record maximum frame is only:

```text
Physical state or equation: <line> supplies <operation> to <operation>
through <operation>.
```

It combines unrelated effects including line-strength offsets, C/N/O/alpha
abundances, density and mean nuclear mass, microturbulence, molecular-line
switches, isotope correction, and wavelength bounds. The recurrence-29
parity frame similarly mixes molecular tables, convection finite-difference
states, damping overrides, flux operators, and population seeds. The
recurrence-28 runtime frame mixes CLI argument vectors, devices, and dtypes;
the recurrence-20 source frame mixes checkpoints, manifests, catalogs, edge
grids, molecular data, and opacity tables. These are renderer templates whose
identity tokens and snippets differ, not common observable effects.

Concrete source checks expose what those frames omit:

- `solve_molecular_equilibrium.return_diagnostics` changes the return from a
  four-tuple to a five-tuple containing the diagnostic mapping. Its rendered
  effect stops at assigning `diag` and never states the observable arity or
  return change.
- `decode_selected_line_words.detect_swapped_layout=True` computes a swapped
  decoding and adopts it only if its score exceeds the native decoding. The
  contract stops at the first swapped assignment; it omits the comparison and
  final chosen layout, while `False`'s important skipped decode/score work is
  likewise absent from the observable effect.
- the `requested_maximum_velocity=-99.0e5` sentinel selects a
  gravity/temperature-grid interpolation and scales the returned depth
  profile. Its rendered effect stops at the first `gravity_index` assignment
  versus `abs`, not the resulting velocity profile.
- the device/dtype records retain the MPS/float64 rejection, but duplicate
  line 52 as both an `if` and a `selection_expression` and present the two
  Boolean operands as route heads. This is AST-template narration, not a
  concise statement of the resolved backend/precision consequence.

The 185 forwarded operations are stored and hash-checked, but generic prose
renders only prefixes: at most eight target operations in default/alternate
routes and four in `observable_effect`. The 103-operation frequency-invariant
edge is the clearest case. Storing the remaining operations is not the same
as explaining their verified output effect.

The five newly special-cased forwarded contracts and the hydrogen loader are
good local repairs. They do not close the all-456 semantic acceptance
boundary.

### P0.3 — Rebound private evidence admits fabricated source facts

The literal authority reader correctly ignores the mutable public authority
path. A mutation of the first category anchor is rejected against the
on-disk authority, and the focused suite's public registry, role-map,
description-map, path, and digest substitutions also reject.

The supposedly private AST evidence is nevertheless a mutable module global,
and the validator treats it as its own recomputation oracle. A read-only
hostile probe:

1. retained the exact on-disk 456-record authority and its first anchor;
2. appended a nonexistent L9999 direct binding
   `fabricated_temperature = source * 0` to
   `parse_atmosphere_deck.source`;
3. rebound `_PINNED_AST_DEFAULT_PARAMETER_USE_EVIDENCE`, the public evidence,
   rendered contracts, merged reviews, and all their digests to the forged
   structures; and
4. invoked the full `_validate_default_parameter_reviews` over the canonical
   raw inventory.

Validation returned normally. The accepted fabricated observable effect
began:

```text
Diagnostic data, shape, or publication surface:
L9999 derives `fabricated_temperature` as `source * 0`
```

A combined probe also retained the real first anchors while appending a fake
`requested_device` branch, a fake second `source_path` attribute predicate,
and a fake `erase_frequency_grid(frequency_invariants)` downstream operation.
After rebinding the same expectation surfaces—and, for the downstream case,
the mutable target-operation parser—the full validator again returned
normally. The fake branch appeared in `default_route`, the attribute fact was
retained, and the fake downstream operation appeared in the rendered route.

This is not a digest attack: every rebound digest honestly described the
forged in-memory structures, while the unchanged pinned source described none
of them. The validation compares a mutable registry to itself instead of
freshly reparsing the pinned files at the final semantic boundary. The
candidate therefore fails the required forged branch/attribute/downstream
fact and both-surface rebound test.

### Seventh-repair verdict

**REJECT P0.3.**

- **P0:** the exact 4,560 wrong-label loop passes, but 39 keys satisfy two
  role-support predicates; precedence hides rather than removes the false-role
  collisions;
- **P0:** normalized prose still collapses physically and operationally
  different parameters into generic forwarding frames, with an independently
  observed 146 frames and maximum recurrence 35, and omits concrete return,
  layout, sentinel, and downstream consequences;
- **P0:** rebinding the mutable private AST evidence together with the public
  contracts/reviews/digests makes the full validator accept fabricated source
  facts that are absent from the pinned checkout;
- **P1:** the candidate report's exact 145-frame claim is false at these bytes
  (the executable normalizer produces 146); this is independently secondary
  to the semantic P0s;
- **P2:** none; formatting, compilation, source cleanliness, and canonical
  serialization pass; and
- **retained:** 58/1,501/1,443/0 ownership, 140/456 defaults, 97/109 authority
  join, the complete source-evidence census, all eight forwarding edges and
  185 stored operations, both corrected `None or build_*()` routes, the five
  special high-risk forwarding repairs, `source_path` LTE routing, exact
  canonical rebuild bytes, 40 focused tests, and the eleven-role census.

Required repair: make support itself exclusive rather than selecting the first
overlapping token rule; derive roles and effects from a fresh immutable source
parse at the final gate; reject any rebound private evidence even when every
published expectation and digest agrees with it; and replace repeated
argument-forwarding frames with factual observable consequences for each
parameter, including output arity, selected layout, physical returned state,
and the relevant complete downstream effect.

<!-- END DELIMITED SEVENTH-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->

<!-- BEGIN DELIMITED EIGHTH-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->

## 43. Eighth-repair independent audit — repair Section 17 (2026-07-30)

Status: **independent hostile re-audit; REJECT**

This section audits the exact eighth-repair candidate bytes and candidate
report Section 16. It does not reuse the candidate author's semantic
conclusions. Sections 1–42 remain immutable history. The triggering
seventh-repair audit was unchanged at audit start:
`99c9e9a9d43ac3d8bb0524125ae533d8e4a98d4639fb26f2f974c6fc86379b55`.

### Exact candidate identities

| object | independently observed SHA-256 |
| --- | --- |
| pinned Payne Zero checkout commit | `9c44001feae40b85146630499e6f8a5fed42e5af` |
| semantic override builder | `d3dd08a77a8b8d717c4b534fadac9c0b02c8ee70e1d4d623d1ddb0ceede0ec36` |
| focused semantic tests | `e3231c028c8ac9f5da2c8382a976311e0b68d228b65d34801878c61ee12448e0` |
| schema-v2 source-fact authority | `f5a31c24b86b804a171258f0b342b3474a42a2cfda936635beb21177a462ae23` |
| checked semantic coverage ledger | `c12f87df7510a247174a43c98ae22002b083a2673662fe46744caf6bd3fdc85c` |
| candidate report | `8da096d24c67d706bb2dc6b4459ae23ce43ab62624fa7dc08e63fce9f9f69f99` |

Every reviewed object was a regular single-link file. No candidate,
governing, chapter, data, paper, or pinned-source object was modified.

### Retained structural, source, and clean-process passes

Two independent raw-inventory builds were byte-identical to one another and
to `audit/paynezero_symbols.json` at
`010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf`.
Two independent canonical semantic builds in isolated temporary destinations
were byte-identical to one another and to the checked ledger at
`c12f87df7510a247174a43c98ae22002b083a2673662fe46744caf6bd3fdc85c`.
Each semantic build reported 58 modules, 1,501 public objects, and 1,501
reviewed overrides.

Independent traversal reproduced:

- 1,501 records, 1,443 qualified names, zero `module_default` records, and
  1,501 `reviewed_symbol_override` records;
- 899 taught, 357 plumbing-only, 184 composed, 41 diagnostic-only, 13
  compatibility-only, and 7 unsupported records;
- 140 sorted default-bearing callables and exactly 456 sorted,
  duplicate-free default-parameter records;
- 40 source files, 895 parameter loads, 431 predicates (302 direct and 129
  coupled), one attribute-mediated predicate, 1,054 call forwards (688 direct
  and 366 coupled), and 242 derived bindings (135 direct and 107 coupled);
- 97 authority-qualified names and 109 joined records, split 1/11/16/23/46
  across Chapters 1–5;
- the exact eleven-role census of 109 physical, 67 source identity, 13 cache,
  58 runtime environment, 12 diagnostic, 19 compatibility, 88
  parity/injection, 5 unsupported, 69 algorithm control, 13 continuous
  equation, and 3 dependency injection records; and
- source-evidence, contract, merged-review, and complete-proof digests
  `3256470c…`, `0494a4f6…`, `e3a0dd68…`, and `953d133c…`.

All eight declared downstream edges were independently rebuilt from the
pinned targets. Their exact operation counts remain
12/23/14/14/103/7/9/3, totaling 185. Direct source inspection retained the
concrete semantics for the four-versus-five molecular-equilibrium return,
native-versus-score-selected line-word layout, microturbulence sentinel,
resolved device/dtype pair, frequency-invariant opacity paths, spectral
operator, saturated-core policy, molecular chunk fallback, resolving grid,
retry jitter, mixing length, `source_path=None`, both `None or build_*()`
dependencies, and the hydrogen-table route.

The total role classifier also closes the preceding audit's exclusivity
defect. Independent enumeration found support cardinality exactly one for all
456 fresh records, zero supported roles for all 456 empty-evidence records,
and rejection of all 4,560 wrong roles. The authority anchors independently
split into 235 call, 207 predicate, and 14 use anchors.

The final validator's default-captured loader starts the literal on-disk
builder in a clean interpreter. The focused coherent attack rebinding private
and public AST evidence, manifest/authority surfaces, contracts, reviews,
paths, role maps, downstream parser, and affected digests rejected the L9999
binding/branch, fabricated attribute route, and fabricated downstream
operation against the fresh pinned parse. The critical source-fact,
456-by-11 role, full forgery, and two-build canonical tests passed:

```text
4 passed in 129.70s
```

These are substantial repairs. They do not establish the claimed terminal
semantic recurrence proof.

### P0.1 — The recurrent “family” omits the terminal operation it claims to prove

The aggressive normalizer census is mechanically reproducible: 456 effects
produce exactly 356 frames, maximum recurrence 6, with 64 recurrent frames.
The candidate test also obtains zero collisions under its declared family
tuple.

That tuple is not a complete source-operation or terminal-effect family. It
retains the broad effect category, one route-kind label, sets of
predicate-kind/relation pairs, sets of call and binding relations, and, only
for the eight forwarded records, target names plus sets of downstream
node-kind/relation pairs. It discards local callees, call arguments, binding
targets and values, operation hashes, operation cardinality, return schema,
and return-versus-mutation behavior. Different source operations therefore
become one family by construction.

An independent stricter operation comparison found different exact
source-operation signatures in 51 of the 64 recurrent groups. Even the very
coarse vector of local use/predicate/attribute/call/binding and downstream
operation counts differs within 31 of 64 recurrent groups. The location and
hash-complete signatures differ in all 64, as expected for distinct source
sites; the important result is that the candidate's relation-only family
erases concrete operational differences before testing them.

The normalized frame at SHA-256
`55d3ef1cbbb41f83d21cee66515c28e316e537bf1c4d2b8fcd7ade7b9fbc95b1`
is a direct counterexample. It groups:

| source contract | actual terminal behavior | local verified calls |
| --- | --- | ---: |
| `payne_zero_atmosphere.equation_of_state.iterate_electron_density.max_iterations` | returns `None`; mutates electron-density and ion-stage arrays in `AtmosphereRuntimeState` | 4 |
| `payne_zero_synthesis.equation_of_state.solve_population_state.max_iter` | returns one assembled `PopulationState` | 1 |
| `payne_zero_synthesis.molecular_equilibrium.solve_molecular_equilibrium.max_iter` | returns four arrays, or five values when diagnostics are requested | 1 |

All three are rendered as changing “the returned electron, atomic, and
molecular population solution” through the same `call flow direct` family.
That statement is directly false for the in-place atmosphere procedure and
does not distinguish either return schema. The normalization gate accepts the
group because it never models those terminal facts.

This is not an objection to cosmetic recurrence. It is a source-level false
claim in one of the 456 contracts and proof that the advertised one-family
test does not test the terminal semantic quantity required by P0.3.

### P0.2 — Most “terminal effects” remain one generic renderer

Exactly 439 of 456 observable effects still come from
`_generic_terminal_observable_effect`. The generic function selects a broad
subject and sink, counts or relation-collapses local AST facts, and ends with
one source witness. It does not inspect the callable's return statements or
mutation targets. Consequently “results or mutations” is used as an
undifferentiated placeholder precisely where the audit requires a concrete
changed quantity and terminal state.

The candidate's synthetic collision does not exercise a rejection path. The
test normalizes two invented strings, manually creates a two-element set
containing `("return-schema", "four-to-five tuple")` and
`("selected-layout", "native-or-score-chosen swap")`, and asserts that the
set length is two. Those labels are not derived from either source,
introduced into `frame_members`, or passed to a validator. The assertion
therefore demonstrates Python set cardinality, not that the recurrence
oracle detects a source-derived terminal-family collision.

The prose cleanup claim is also not exact under the requested lexical gate.
`observable_effect` still contains literal `supplies` twice:

- `synthesize_from_labels.resolution`; and
- `compute_sampled_continuum.frequency_invariants`.

Literal ` through ` and both old rejected suffixes occur zero times. This
secondary wording discrepancy does not cause the rejection; the false
return/mutation grouping does.

### Eighth-repair verdict

**REJECT P0.3.**

- **P0:** the 356-frame/64-recurrence test classifies local AST relation
  shapes, not terminal source-operation families; 51 recurrent groups contain
  multiple stricter operation signatures and 31 differ even in coarse
  evidence cardinality;
- **P0:** a concrete accepted recurrent frame conflates an in-place
  `None`-returning atmosphere mutation, a `PopulationState` return, and a
  four/five-value molecular-equilibrium return, while falsely describing all
  three as the same returned solution;
- **P0:** 439 of 456 effects remain generic sink prose that does not derive
  return schemas or mutated state, and the synthetic collision assertion is
  disconnected from the production recurrence acceptance path;
- **P1:** two observable effects retain literal `supplies` despite the stated
  cleanup target;
- **retained:** exact source and ledger rebuilds; 58/1,501/1,443/0 ownership;
  140/456 defaults; 97/109 authority join; all evidence and digest censuses;
  exclusive one-role support, zero empty-evidence support, and 4,560
  wrong-role rejections; the clean-process L9999/attribute/downstream forgery
  boundary; all 8/185 downstream operations; the named high-risk repairs;
  canonical serialization; and the critical focused test subset.

Required repair: derive and validate each observable effect against a fresh
terminal source summary that includes return arity/type or exact mutated
state, concrete call/binding identities and cardinality, and the complete
downstream terminal consequence. Recurrent prose may be shared only when
those terminal summaries are equal. The synthetic collision must enter the
same actual grouping-and-rejection function used for the 456 real records,
and the remaining generic wording fragments must be removed.

<!-- END DELIMITED EIGHTH-REPAIR INDEPENDENT RE-AUDIT: 2026-07-30 -->
