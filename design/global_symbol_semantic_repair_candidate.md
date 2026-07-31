# Global public-symbol semantic repair candidate

Status: **candidate for independent audit; not self-accepted**
Scope: P0.3 and the symbol-ledger parts of P1.1, P1.2, and P1.4 from the
post-Chapter-6 global zoom-out
Pinned Payne Zero source:
`9c44001feae40b85146630499e6f8a5fed42e5af`

## 1. Result

The 1,501-record public-source inventory remains exhaustive, but it is no
longer being used as if module visibility alone proved semantic ownership.
This candidate adds a fail-closed semantic review layer with exactly the six
permitted dispositions:

- `taught`;
- `composed`;
- `plumbing-only`;
- `compatibility-only`;
- `diagnostic-only`; and
- `unsupported`.

Every reviewed record also has an exact `source_spelling`, a responsibility,
a verification gate, a primary location, supporting locations, and a status.
The exact qualified name remains the authoritative source spelling; the new
field makes that binding mechanically testable.

The resulting ledger contains:

| category | records | distinct qualified names |
| --- | ---: | ---: |
| complete pinned inventory | 1,501 | 1,443 |
| semantically reviewed | 949 | 899 |
| residual homogeneous `module_default` | 552 | 544 |
| explicit symbol registry | 224 | 215 |
| exact package API aliases | 198 | 198 |
| reviewed-module policy after explicit replacements | 527 | 486 |

The apparent difference between records and qualified names is intentional.
There are 58 in-module export/definition pairs with the same qualified name
and different kinds. Tests require their semantic assignments to agree.

This is a candidate repair, not a claim that the global book plan is accepted.
The other global repairs and an independent zoom-out remain separate gates.

## 2. Selection methodology

The review began from all 58 modules and all 1,501 records in the raw AST
inventory. The pinned checkout was then inspected read-only for:

1. every concrete symbol and branch named by the global zoom-out;
2. modules with more than one pedagogical owner;
3. optional public parameters and feature switches;
4. `NotImplementedError` and explicit unsupported guards;
5. diagnostics, timing, debug returns, prewarm entry points, and test/parity
   branches;
6. compatibility schemas, alternate loaders, serializers, publication
   helpers, path resolution, and cache controls;
7. atmosphere versus synthesis molecular-family status;
8. package exports and CLI aliases; and
9. exact routines already accepted in Chapters 1–5.

There are two complementary registries.

First, 35 mixed or branch-sensitive modules are frozen by the SHA-256 of their
complete sorted `(qualified_name, kind)` public snapshot. A new or missing
public record makes the builder fail before it can inherit the module policy.
Within those frozen surfaces, 215 branch-sensitive qualified names receive
direct overrides. Every one of those names is also in
`REQUIRED_EXPLICIT_SYMBOLS`; none may fall back to the module review.

Second, all 198 package-level exports are frozen and bound to their defining
record. The alias copies the corrected target owner, gate, and status, while
its own disposition is `plumbing-only`. This avoids treating `__init__.py` as
one semantic chapter and prevents aliases from drifting away from definitions.
The package version string is the one non-target export and has its own compact
Appendix C disposition.

## 3. Complete reviewed-module counts

The exact module snapshots cover 730 records before direct symbol overrides.
They are divided as follows.

### Atmosphere package: 20 modules, 356 snapshot records

| module | records |
| --- | ---: |
| `_numba_cache` | 2 |
| `atmosphere_io` | 23 |
| `cli` | 2 |
| `config` | 29 |
| `convection` | 27 |
| `data_files` | 6 |
| `direct_abundance` | 69 |
| `install_runtime_data` | 10 |
| `line_opacity` | 6 |
| `line_selection` | 8 |
| `microturbulence` | 1 |
| `prewarm` | 5 |
| `radiative_transfer` | 12 |
| `rosseland_mean` | 1 |
| `run_setup` | 36 |
| `runner` | 60 |
| `source_catalogs` | 11 |
| `synthesis_bridge` | 8 |
| `transfer_kernels` | 3 |
| `warm_start` | 37 |

After direct overrides and the 182 atmosphere package aliases, 559 atmosphere
records are semantically reviewed and 384 retain `module_default`.

### Synthesis package: 15 modules, 374 snapshot records

| module | records |
| --- | ---: |
| `api` | 36 |
| `atmosphere` | 11 |
| `atomic_lines` | 41 |
| `cli` | 1 |
| `device` | 7 |
| `equation_of_state` | 71 |
| `ground_partition_table` | 5 |
| `line_opacity` | 61 |
| `molecular_lines` | 54 |
| `paths` | 10 |
| `pipeline` | 46 |
| `prewarm` | 3 |
| `radiative_transfer` | 18 |
| `source_catalog_molecular_compiler` | 7 |
| `synthesis` | 3 |

After direct overrides and the 16 synthesis package aliases, 390 synthesis
records are semantically reviewed and 168 retain `module_default`.

The final reviewed dispositions are:

| disposition | records |
| --- | ---: |
| `taught` | 445 |
| `composed` | 146 |
| `plumbing-only` | 297 |
| `compatibility-only` | 17 |
| `diagnostic-only` | 37 |
| `unsupported` | 7 |

## 4. Concrete P0.3 closures

### Accepted radiative and population primitives

- `payne_zero_synthesis.radiative_transfer.planck_bnu` is now Chapter 1,
  `taught`, and `integrated`.
- `payne_zero_synthesis.radiative_transfer.integrate_optical_depth`,
  `payne_zero_atmosphere.radiative_transfer.integrate_on_depth_grid`, and
  `payne_zero_atmosphere.transfer_kernels.accumulate_transfer_range_parallel`
  are now Chapter 2, `taught`, and `integrated`.
- Both scalar and vector ground-partition functions and their three public
  tables are Chapter 3, `taught`, and `integrated`.
- `pipeline.compute_doppler_per_ion` is Chapter 3 physical width use, while
  `standard_microturbulence` is the Chapter 11 standard profile. The ledger no
  longer conflates consuming a supplied velocity with constructing the
  atmosphere prescription.

### Structured builder versus complete synthesis pipeline

The three structured-column builders in `pipeline.py`, `synthesis.py`, and
`api.py` are Chapter 3/4 composition and are synchronized with the accepted
fixed-electron-density and molecular-state gates. In contrast,
`SynthesisPipeline`, its constructor, and its `run` method are Chapter 10
composition.

All 46 public `pipeline.py` records are explicitly reviewed. Atomic/H/He
invariants retain Chapter 7 family ownership, molecular invariants retain
Chapter 8 ownership, and window/device/cache/crop/result objects are owned by
Chapter 10. `WindowInvariants.build_profile` and optional returned slabs are
diagnostic rather than physics.

### Production Rosseland ownership

`rosseland_mean_step` is Chapter 12 production physics. Chapter 1 remains only
a controlled conceptual support location, and Chapter 5 is no longer its
implementation owner. This preserves Chapter 5's explicit exclusion of the
production Rosseland reduction.

### Molecular source-family status

The atmosphere and synthesis lanes can no longer collapse into one inherited
“molecular” row:

- atmosphere diatomic, TiO, and water paths are standard converted Chapter 8
  sources;
- atmosphere H3+ is explicitly opt-in and absent from the default source
  mapping;
- the shared atmosphere selected-record deposit is Chapter 7 general
  machinery with Chapter 8 molecular support;
- synthesis text-band and TiO compilers are standard Chapter 8 paths; and
- `compile_h2o_partridge` is a working compiler explicitly recorded as not
  wired into the standard synthesis pipeline.

`generate_selected_lines`, `source_line_paths`,
`atmosphere_source_catalog_paths`, and `prepare_opacity_state` have
responsibilities and gates that preserve the family-specific teaching owner
while allowing Chapter 11 to compose the selected catalogs.

### Optional synthesis boundaries

- `SynthesisPipeline.__init__` records `source_path` as comparison-only; the
  standard LTE route rebuilds the Planck line source and uses zero line
  scattering.
- `SynthesisPipeline.run` records `keep_slabs` as diagnostic-only and
  `spectral_operator` as the joint total/continuum, pre-normalization,
  pre-host-transfer interface. It does not claim an instrument or fitting
  implementation.
- The public API, engine wrapper, and label workflow carry the same spectral
  boundary without moving synthesis ownership into fitting.
- `accumulate_atomic` records the standard batched wing route,
  `wing_mode="loop"` as parity-only, and `host_accumulator` as the restricted
  unstimulated metal-only diagnostic.
- `solve_spectrum` records the strict direct-call saturated-core guard and the
  standard pipeline's explicit implemented continuation.

### Evidence-specific unsupported behavior

The runner ledger now claims exactly the implemented guards:

1. fewer than one iteration;
2. enabled turbulent pressure; and
3. opacity flag 13, HLINOP.

The seven turbulent-pressure settings/carriers are `unsupported` and
`boundary`. `resolve_run_setup`, `opacity_flags_from_atmosphere`,
`DEFAULT_OPACITY_FLAGS`, and `run_atmosphere_model` name the exact guards.
They do not claim that every NLTE or raw-selector spelling is independently
rejected.

### Diagnostics, compatibility, caches, and publication

The review separates:

- optional atmosphere diagnostic/debug paths and run-result diagnostics;
- disabled-convection zero diagnostics;
- prewarm manifests, timing records, build profiles, and compiler logging;
- canonical schema-v4 fields from bounded older schema aliases;
- external fixed-column deck compatibility;
- atomic and molecular cache schema/load/round-trip helpers;
- invariant-cache controls and stable Numba cache paths;
- debug-state conversion from converged-product publication; and
- product metadata from physical state.

`save_product_structured_atmosphere` is composed Chapter 13 publication of
final fixed-column-quantized arrays. Debug and live-runtime diagnostic
handoffs are `diagnostic-only`; they cannot masquerade as the product.

All package exports are bound to their targets, duplicate export/definition
records must agree, `synthesize_from_labels` retains canonical `r_grid` plus
the conflict-checked `resolution` alias, the public archive/in-memory API
retains `resolution`, and the CLI keeps both exact flag aliases.

### Initializer status

The public label initializer now explicitly distinguishes initializer success
from atmosphere convergence. Its result records retain
`atmosphere_converged=False` and
`atmosphere_closure_required=True`. Direct-abundance loaders and surrogates are
explicit opt-in starting states with hash/provenance gates; the optimizer
surrogate record states that no fitting implementation is owned here and that
physical closure remains mandatory.

## 5. Residual `module_default` interpretation

The remaining 552 records are not evidence-free omissions. They are the
homogeneous public fields, constants, table carriers, and kernels left after
branch-sensitive members were pulled into the direct registry. They occur in
20 modules:

- atmosphere: constants, the homogeneous remainder of continuum opacity,
  convergence, Doppler, atomic EOS, hydrogen profiles, hydrostatic balance,
  line catalogs, line-profile mathematics, molecular data/equilibrium,
  population layout, radiative pressure, runtime state, atomic internal
  energy, and temperature correction;
- synthesis: constants, continuum, hydrogen profiles, and molecular
  equilibrium.

Their module assignments remain appropriate because all remaining records in
each listed responsibility share the same primary causal lesson and gate. For
partially reviewed modules, the exceptions are explicit: for example,
production Rosseland ownership is not left inside continuum opacity; the
molecular catalog/equilibrium boundaries already taught in Chapter 4 are
explicit; and the hydrogen-profile population helpers handed to Chapter 7 are
explicit.

This is deliberately narrower than hand-labeling every homogeneous array field
with identical prose. The safety property comes from the combination of:

- exhaustive 1,501-record visibility;
- exact reviewed-module snapshots;
- a 215-name required explicit registry;
- complete package-alias binding;
- duplicate-kind consistency checks; and
- exact canonical rebuild equality.

## 6. Reproducibility and tests

The canonical ledger was rebuilt only through:

```text
python scripts/build_symbol_coverage.py \
  --inventory audit/paynezero_symbols.json \
  --ledger COVERAGE.md \
  --output audit/paynezero_symbol_coverage.json
```

The focused suite has 15 passing tests. It verifies:

- 58 modules and exactly 1,501 `(qualified_name, kind)` records;
- exact reviewed/default counts and the six-value disposition vocabulary;
- all 35 reviewed-module snapshots;
- all 215 explicit names and fail-closed behavior if one disappears;
- failure when a reviewed module gains an unreviewed public branch;
- consistency of all 58 duplicate export/definition records;
- all 198 package API aliases;
- accepted Chapter 1–5 examples;
- Rosseland and microturbulence ownership splits;
- the two-lane molecular status;
- pipeline optional/test-only branches;
- the exact runner guards;
- device/dtype, saturated-core, and initializer boundaries;
- compatibility/diagnostic/cache/publication separation; and
- byte-for-structure equality of a clean canonical rebuild.

Ruff check and Ruff format-check pass, and `git diff --check` reports no
whitespace errors for the authorized files.

## 7. Frozen identities

These hashes describe the final candidate bytes before this report:

| object | SHA-256 |
| --- | --- |
| triggering global zoom-out | `e4f08434604ccd308264e40ea0e934811c0d26cfbb3f7a6c15fca5c51cec8baf` |
| raw 1,501-record inventory | `010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf` |
| semantic override builder | `ab7ad32d5efd761296d7d8cbd164728df3c95a77d1accb035c2a7768a95569fb` |
| focused semantic tests | `f46880c75892581ae1bba1da66fed0b510b9e1ffe1db613b0dda77e9157cc227` |
| canonical semantic ledger | `c53a22d4b4f54d911e3e6e3ed6781d975e82f5ea13f08248aa820b2a5e110f27` |
| reviewed module snapshot registry | `0d497c39f89a11eee0562a551178f4bddd1e7a0bf16a1cd0439778ffc30ff2cc` |
| 215-name explicit registry | `2e293bd51463d93d77cfad4fb8e950b88f9df1299fc7e55dc2a47e175a8d119d` |
| 197 target-bound aliases | `a98d9d587d2139d522a7430aed58c18a0a8fb7a344d1fdeaf4c885b71e0d7af3` |
| 949 reviewed semantic records | `665c0a9c9d52bbabdf491fe338f89f711ea1298ad1109425cdb1efeab3e7f5ec` |
| 552 residual `(qualified_name, kind)` records | `aaf694b8a360281955201cfd2f1675ce151f85c158f2fa1298a64639fd1b173e` |

The alias digest covers the 197 aliases with definition targets; the package
version string is the independently reviewed 198th export.

## 8. Candidate disposition

**CANDIDATE — ready for independent audit.**

This repair closes the semantic-proof gap by making every identified mixed or
branch-sensitive public symbol either an exact explicit review or a member of
a hash-frozen reviewed surface, while keeping homogeneous residual plumbing
compact. It does not self-accept P0.3 and does not modify the Bible, global
contracts, coverage prose, chapters, data, or pinned Payne Zero source.

## 9. Independent-audit repair ledger

Status: **repaired candidate for independent re-audit; not self-accepted**
Triggering independent REJECT:
`b8cdd8f1a51be7225cd4886ac9fcddf5c5f22b17ac3f57cf30b14d4b91cc1fdd`

Sections 1–8 above preserve the history of the rejected revision. This repair
ledger supersedes their counts, residual-homogeneity claim, frozen identities,
and readiness statement.

### P0 source and signature binding

The semantic builder now validates the raw inventory before assigning any
module policy or explicit override. It requires:

1. pinned Payne Zero commit
   `9c44001feae40b85146630499e6f8a5fed42e5af`;
2. canonical raw-inventory content digest
   `94861595dfe59afcbe2c47b23c19d14e25d0474a6727ad23e3c4f3948cc1b4b0`;
3. all 55 reviewed module source SHA-256 values;
4. the complete public function, constructor, method, class, field, datum, and
   export descriptor surface of those 55 modules;
5. the exact source descriptor for every one of the 240 explicit names;
6. twelve source-spelled branch-default contracts, including
   `source_path=None`, `keep_slabs=False`,
   `coulomb_table_energy_first=False`, `return_diagnostics=False`,
   `detect_swapped_layout=True`, and both hydrogen `apply_stim` and Lyman
   boundaries; and
7. complete semantic, alias-target, and accepted-Chapters-1–5 registries.

Every reviewed ledger record now carries its source module SHA-256, complete
module public-surface fingerprint, and public-object fingerprint. The twelve
branch-sensitive entries also carry their reviewed default contract. A source
default edit changes the frozen module SHA even though the raw AST inventory
does not serialize default expressions.

The package alias proof is now two-sided. The complete 197-target registry is
digest-bound, every target must have an ultimate public source descriptor, and
the exact source SHA of both package `__init__.py` modules is frozen. Each alias
**composes** its own identity check with the target gate; it does not merely
copy the target gate.

### P0 residual repair

All 20 formerly residual modules now have exact ledger-surface, source-byte,
and public-descriptor snapshots. Consequently:

| category | records | distinct qualified names |
| --- | ---: | ---: |
| complete pinned inventory | 1,501 | 1,443 |
| semantically reviewed | 1,501 | 1,443 |
| residual `module_default` | 0 | 0 |
| reviewed-module snapshots before explicit replacement | 1,303 | — |
| reviewed-module policy after explicit replacement | 1,053 | — |
| explicit symbol registry | 250 | 240 |
| exact package API aliases | 198 | 198 |

The formerly ambiguous public branches are direct contracts:

- the five atmosphere IFOP-19 Rosseland table/create/ingest/evaluate/compute
  objects are `diagnostic-only`, default-off surrogates and are explicitly
  distinct from both standard continuum and the Chapter 12 production mean;
- synthesis `build_frequency_invariants` records the standard
  `coulomb_table_energy_first=False` route and the `True` sampled diagnostic;
- synthesis molecular equilibrium records its four-array standard return and
  optional diagnostics return;
- selected-line decoding records native generation versus swapped external
  compatibility detection;
- hydrogen invariant construction records the Balmer path and exact Lyman
  rejection, while hydrogen accumulation records stimulated and unstimulated
  parity;
- both public atmosphere Saha paths record their unsupported population-mode
  guards;
- the line-profile table loader records default/explicit paths and
  cold/warm/forced reload behavior; and
- temperature correction records its complete
  mode/iteration/convection/smoothing/pressure/remap branch matrix.

The disposition counts are now:

| disposition | records |
| --- | ---: |
| `taught` | 915 |
| `plumbing-only` | 352 |
| `composed` | 173 |
| `diagnostic-only` | 41 |
| `compatibility-only` | 13 |
| `unsupported` | 7 |

### P1 classification and join repairs

The current atomic `LineCatalog.from_npz`, molecular
`MolecularLineCatalog.from_npz`, and modern
`MolecularLineCatalog.from_mapping` paths are now `plumbing-only`, not
compatibility-only. The standard complete-state
`load_atmosphere_initializer` is now Chapter 14 composition, with its current
checkpoint schema/family/coordinate/provenance gate.

`AtmosphereOutput.diagnostics_path` is now explicitly a declared but unwritten
API boundary. `debug_state_path` is separately the actual opt-in diagnostic
NPZ publication path.

The accepted-Chapters-1–5 synchronization gate now joins all 75 integrated
explicit contracts, including the previously missed Chapter 4 diagnostic
return and Chapter 5 surrogate/orientation distinctions. It is no longer a
nine-name sample.

### Adversarial and reproducibility evidence

The focused suite now has 21 passing tests. In addition to the earlier gates,
it proves failure for:

- a reviewed raw-inventory module SHA mutation;
- `SynthesisPipeline.__init__.source_path` signature drift;
- `SynthesisPipeline.run.keep_slabs` signature drift;
- `source_path` and `keep_slabs` default-contract mutations;
- a package alias retarget; and
- explicit disposition, primary location, gate, or status mutation.

The raw-inventory SHA, `source_path`, and `keep_slabs` signature adversaries
are also passed through the canonical builder subprocess and are rejected
before an output ledger is written. A clean canonical rebuild is
byte-identical to the checked artifact. Ruff formatting, Ruff checks, Python
compilation, and the focused suite pass.

### Repaired frozen identities

| object | SHA-256 |
| --- | --- |
| raw inventory bytes | `010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf` |
| canonical raw-inventory content | `94861595dfe59afcbe2c47b23c19d14e25d0474a6727ad23e3c4f3948cc1b4b0` |
| original 35-source manifest | `a6278b768a3f57d2a3435861d965861bbcf86a700d894b321485d570c9087390` |
| expanded 55-source manifest | `b75d6f85f73d0ae14df5e908e785f46173681313407136cfa785804f9565eb32` |
| 55-module public signature/field surface | `669a1afe23df89b4030aaf8ee1ad582409e771eff9b20b592ae20976fc60318b` |
| 240-name source descriptor surface | `4eb980b3557266341ff5b61b425bfabf5edbc34d8d032bd351fc1ddea5fc5b99` |
| complete explicit semantic registry | `c27655d4d945b87839cb36162ae9a0d0a2fe3e20cd053ce54ab2c3de2ce7eb7d` |
| complete alias-target registry | `9589bbc3c4579ca80996f4f52649859d20ec146eff0b58dfd4af1347f8650964` |
| twelve default contracts | `71cccd37ed9bb84e564d2a851ade38e1c59866791ecfdfae0ad11d134d26cdad` |
| 75-name accepted-status join | `78ddc57aed9256cb604bf00b6a566614e97506606646893abb75e9031a73ec6d` |
| 55 reviewed ledger snapshots | `1465aee28881a29fd63dced37d566d07c15f7191a329f6c7c473c4137a9058f3` |
| 240 required explicit names | `7df112e834d1da8d7c98b38cc99d9d2c199e63dd7f46c5f9785ba23eb2c359ad` |
| complete 1,501-record semantic proof | `bee7f4a5cc7b68c5dc3cfe90fa8787bbb5b45b75961314672c7617573d0f794d` |
| semantic override builder | `ff9c72a74eb6931f38c9deaeccb29d2f146897cca0042044eba5be4105f961d7` |
| focused semantic tests | `9f7f53a89e84b88e29214e702485f109e95e667bca249b1bf462e3427c63afb4` |
| canonical semantic ledger | `edc6ef0429ffab641652712e1cc6b433b18ad40dcc2c1a60bfac70f8169caddb` |

This remains a candidate for independent re-audit. It does not self-accept
P0.3 and did not modify the independent audit, Bible, coverage contract,
chapters, data, or either pinned source project.

## 10. Second independent-audit repair ledger

Status: **repaired candidate for independent re-audit; not self-accepted**
Triggering independent REJECT:
`17ec40ff377eabb8d8c0207f2c964c9824ddbc8a5b52806a80eec4274367e39e`

Sections 1–9 preserve the history of the two rejected revisions. This ledger
supersedes their residual counts, accepted-chapter join, default spelling
claim, proof hash, and readiness statement.

### Remaining branch-sensitive routines

The second audit's source-level residuals are now explicit source-descriptor
and semantic contracts:

- synthesis molecular accumulation distinguishes `apply_stim=True` from the
  unstimulated path and `chunk_lines=None` environment resolution from an
  explicit clamped size; the environment resolver is also explicit;
- the atmosphere continuum assembler binds the default and per-family opacity
  flags, and records that IFOP-19 contributes only when flag 19 is active
  **and** a Rosseland table is supplied; its light-element and scattering
  family assemblers are explicit siblings;
- synthesis partitions distinguish the ordinary
  `apply_ground_partition=True` path from the molecular-bridge path;
- all three public synthesis EOS assemblers distinguish molecules off/on,
  molecular catalog routing, the two-solve seam, and preservation of the
  caller's fixed electron density where applicable;
- the atmosphere transfer-table loader binds default/explicit paths and
  cold/warm/forced reload behavior; and
- the atomic catalog parser binds default/explicit sources, isotope correction
  on/off, and catalog/wavelength ordering.

These changes increase the direct registry from 240 names and 250 records to
251 names and 267 records. The reviewed-module remainder decreases from 1,053
to 1,036 records. No record falls back to `module_default`.

### Complete module-policy integrity

The full effective policy of every one of the 55 reviewed modules is now
digest-bound before assignment. The protected content includes the module key
and all seven semantic fields:

1. semantic disposition;
2. semantic review reason;
3. primary location;
4. supporting locations;
5. responsibility;
6. gate; and
7. status.

The focused tests mutate every field independently at runtime. Every mutation
fails with a module-policy digest error instead of propagating to the ledger.
The complete module-policy content digest is
`b10f3dad89ba00e9c96b254f1ccb708d53c0d82cc84d2509946a2cd840d0a3b7`.

### Hash-bound accepted-artifact join

The former 75-name test was self-derived from the registry it was supposed to
check and has been removed. Its replacement pins the exact bytes of existing
Chapter 1–5 acceptance notebooks, tests, final-acceptance records, symbol
dispositions, and exact-source contracts. A static extraction of 53 symbols
from those frozen external authorities supplies the expected chapter,
disposition, and integrated status. It is not computed from
`EXPLICIT_OVERRIDES`.

The manifest digest is
`f2cbf34acd60ea7ec5f8c87ee49a400705259927473f318d1919b7980e7f19ca`.
Tests reject both a manifest hash mutation and a simulated byte change in an
authority file. Independent ledger mutations to an accepted symbol's location
or status fail at the join before the complete-proof digest is evaluated.

### Exact default spelling and AST structure

There are now 23 source-spelled branch-default contracts. The three synthesis
table paths preserve the exact double-quoted source expressions, for example
`_SYNTHESIS_TABLE_DIR / "line_profile_tables.npz"`, and atomic
`wing_mode` preserves exact `"batched"` spelling. A separate registry freezes
both the lexical segment and its complete expression AST:

`70aa59048d72a23af70170dc87f0fadf2a865344d0831ffd78a23cb265d15433`.

Single-quote normalization, another lexical edit, or an AST mutation is
rejected. The complete 23-contract digest is
`3cd1237df2e6c201689ffcac9ab119b8e18cf74eac77114bd41dd7de74dc99e9`.

### Executable 1,501-record proof

The reported complete proof is now an enforced digest, evaluated on every
application and canonical build. It preserves duplicate kinds and projects
all 1,501 records over qualified name, kind, precision, locations,
disposition, reason and review source, source spelling, responsibility, gate,
status, alias target, all three source fingerprints, and reviewed defaults.
The proof bundle also includes every relevant module, source, explicit,
default, exact-syntax, alias, and accepted-authority registry.

Its required digest is
`fe2c656289c8b67f10e72ad4780d9b2038462a908d8f3050519f5681b65341a0`.
Adversarial one-field changes to location, supporting locations, disposition,
responsibility, gate, status, or source-object identity all fail rather than
being reported as a valid proof.

### Superseding counts

| category | records | distinct qualified names |
| --- | ---: | ---: |
| complete pinned inventory | 1,501 | 1,443 |
| semantically reviewed | 1,501 | 1,443 |
| residual `module_default` | 0 | 0 |
| reviewed-module policy after explicit replacement | 1,036 | — |
| explicit symbol registry | 267 | 251 |
| exact package API aliases | 198 | 198 |

| disposition | records |
| --- | ---: |
| `taught` | 909 |
| `plumbing-only` | 354 |
| `composed` | 177 |
| `diagnostic-only` | 41 |
| `compatibility-only` | 13 |
| `unsupported` | 7 |

### Reproducibility evidence and frozen identities

The focused suite now has 27 passing tests. Ruff format and check pass; both
Python files compile; `git diff --check` is clean for the four authorized
files. A clean canonical builder invocation reports 58 mapped modules, 1,501
mapped public objects, and 1,501 reviewed overrides, then reproduces the
checked ledger byte-for-byte.

| object | SHA-256 |
| --- | --- |
| triggering independent audit, unchanged | `17ec40ff377eabb8d8c0207f2c964c9824ddbc8a5b52806a80eec4274367e39e` |
| 251-name source descriptor surface | `d1c1108dbead8cdc10e1afd3e0c0797164d16c831673054d7b9e2db23569cc42` |
| complete explicit semantic registry | `194a17f23eebed29d546170d935435d48bfd0d482f6a440bbc5420519b740693` |
| 23 reviewed default contracts | `3cd1237df2e6c201689ffcac9ab119b8e18cf74eac77114bd41dd7de74dc99e9` |
| exact default source/AST registry | `70aa59048d72a23af70170dc87f0fadf2a865344d0831ffd78a23cb265d15433` |
| complete reviewed-module semantic policies | `b10f3dad89ba00e9c96b254f1ccb708d53c0d82cc84d2509946a2cd840d0a3b7` |
| accepted Chapter 1–5 artifact manifest | `f2cbf34acd60ea7ec5f8c87ee49a400705259927473f318d1919b7980e7f19ca` |
| complete executable semantic proof | `fe2c656289c8b67f10e72ad4780d9b2038462a908d8f3050519f5681b65341a0` |
| semantic override builder | `6e4a0804772b32436c9b2cece1811406594bda6ec484879334ab58789da71503` |
| focused semantic tests | `a8fc9afb54826dc99f9a602255edbceebb181066bc232d8684d8a7d1053d702e` |
| canonical semantic ledger | `2623a693da7c12db5d50fe859606999562cbfd48bfeba12c7c5c31b2f507126e` |

This remains a candidate for independent re-audit. It does not self-accept
P0.3. The independent audit remains byte-identical, and this repair did not
modify the Bible, coverage contract, chapters, data, or either pinned source
project.

## 11. Third independent-audit repair ledger

Status: **repaired candidate for independent re-audit; not self-accepted**
Triggering independent REJECT:
`c6071c4a16167c33d944fccd225ed4963a06290f838b01ff7d6fa7221e0bc6a1`

Sections 1–10 are retained as revision history. This section supersedes their
branch-coverage counts, accepted-artifact join description, proof digest, and
readiness statement.

### Source-derived default-bearing fixed point

A deterministic AST sweep now reconstructs every public function,
constructor, and public method with at least one positional or keyword-only
default in the 55 reviewed pinned modules. The fixed point contains exactly
140 qualified callables:

- 32 callables have complete source-spelled default contracts;
- 59 additional callables already have full explicit symbol semantics; and
- 49 residual callables now have individual branch/default reviews rather than
  inherited module prose.

The three sets form the exact 140-name source surface. They are checked for
scope, source descriptors, registry content, and source-surface identity.
Every corresponding ledger record carries a non-empty
`default_branch_review`; none has `reviewed_module_registry` as its semantic
source. Duplicate export/definition records remain distinct and consistent.

The newly explicit source scan includes command and installation controls,
data and catalog resolution, line selection and opacity controls, runner
stages, warm-start selection, synthesis device/data routes, molecular metadata
and cache controls, and scattering sweeps. This closes the residual
branch-capable surface rather than treating the audit's named routines as a
finite checklist.

### Named branch contracts closed

The nine routines singled out by the third audit now have full default
contracts and branch-specific responsibilities:

1. `compute_sampled_continuum` distinguishes the scalar sampled diagnostic
   when `frequency_invariants=None` from the validated materialized sampled
   extension when invariants are supplied.
2. atmosphere H-minus opacity distinguishes unity LTE departure from a
   shape-validated supplied depth multiplier.
3. `compute_convection` distinguishes the all-eight-array finite-difference
   route from the ideal-gas fallback, along with enabled iterations, zero
   mixing length, overshoot, and top-layer zeroing.
4. `compute_convection_finite_difference_samples` distinguishes atom-only and
   required-state molecular perturbations, including state restoration and
   optional thermal-energy tracking.
5. `finalize_transfer_state` distinguishes disabled, caller-supplied, and
   internally constructed convection, including seed derivation and optional
   radiation/turbulent-pressure inputs.
6. the convergence norm distinguishes its default asymmetric denominator from
   the symmetric `max(before, after, floor)` denominator.
7. both atmosphere and synthesis prewarm routines distinguish manifest reuse
   from forced rebuild.
8. atomic `load_catalog` binds path/cache defaults, sort and isotope choices
   into cache/parser identity, and distinguishes cache reuse from forced
   reparsing.

### Exact, location-aware accepted-authority join

The accepted join now contains 97 exact qualified names:

| accepted surface | qualified names |
| --- | ---: |
| Chapter 1 | 1 |
| Chapter 2 | 11 |
| Chapter 3 | 16 |
| Chapter 4 binding ownership corrections | 23 |
| Chapter 5 exhaustive symbol-disposition table | 46 |

The former leaf-substring join is gone. Chapters 1–3 use an exact authority
path, line range, marker, and line hash per qualified name. Chapter 4 expands
the binding ownership rows into exact qualified symbols; package exports are
resolved through the frozen alias registry to their exact defining modules.
This prevents an atmosphere `solve_molecular_equilibrium` row from
authorizing the same leaf in the synthesis module. Chapter 5 parses all 38
atmosphere-continuum and all 8 synthesis-continuum rows from their two exact
namespace sections. The seven synthesis symbols previously absent from the
join are therefore included beside `build_frequency_invariants`.

Every accepted ledger record embeds its authority evidence. Archive filenames
are rejected as semantic evidence. Tests adversarially mutate authority path,
line, marker, line hash, alias target, qualified target, and ledger evidence;
wrong-module same-leaf authorization fails closed.

### Superseding counts

| category | records | distinct qualified names |
| --- | ---: | ---: |
| complete pinned inventory | 1,501 | 1,443 |
| semantically reviewed | 1,501 | 1,443 |
| residual `module_default` | 0 | 0 |
| reviewed-module policy after exact replacements | 942 | — |
| explicit symbol registry | 309 | 286 |
| explicit residual-default registry | 52 | 49 |
| exact package API aliases | 198 | 198 |
| default-bearing callable records | 163 | 140 |
| accepted-authority evidence records | 109 | 97 |

| disposition | records |
| --- | ---: |
| `taught` | 899 |
| `plumbing-only` | 357 |
| `composed` | 184 |
| `diagnostic-only` | 41 |
| `compatibility-only` | 13 |
| `unsupported` | 7 |

### Reproducibility evidence and frozen identities

The focused suite has 31 passing tests. Ruff check and format check pass; both
Python files compile; `git diff --check` is clean for the four authorized
files. Two clean canonical builder invocations are byte-identical and match
the checked ledger. Each reports 58 mapped modules, 1,501 mapped public
objects, and 1,501 reviewed overrides.

| object | SHA-256 |
| --- | --- |
| triggering independent audit, unchanged | `c6071c4a16167c33d944fccd225ed4963a06290f838b01ff7d6fa7221e0bc6a1` |
| 286-name source descriptor surface | `ffe805be645f5d1ee5938b3dc7797690759e63c6c9f7572b775d01987ff6611a` |
| complete explicit semantic registry | `17cf03748879f1f698efa6468ac7b8f9e2b85876c8813a5a96b2a1a1017e26e0` |
| 32 reviewed default contracts | `454f0fb32d68a4173ee2502bad6b600a2c568dfe97a1adebbe32ab4d6ab01449` |
| 140-name default-bearing source surface | `9947ad3a78ebb59fbe8d0f3e194bbb172e485099e24a2b9412f1a8ee362f5343` |
| 140-name default-bearing registry | `74008ed126f8e3a07f100769304e251acfbddc96ba91812cc1240726624a59ee` |
| 49-name residual-default review registry | `c74c4d7f667da497b9f6e81919b339084bfb5d5aa1a8b3c4241cf80b5d386b99` |
| accepted Chapter 1–5 artifact manifest | `10c0e3bcd029834d97f480ed3e3e09d11b01126e36c116334f665e66bc28347b` |
| complete executable semantic proof | `63d91360b55fc35c5372576ac7a01a1dbbcee759ebf12d1a94b2103cc59ef46a` |
| semantic override builder | `202e67db97be05c59ac4ddfa0631500f7479e669285fbea8ccc1fed3ff5ff049` |
| focused semantic tests | `aa3f4e8c14a7c849bc422b5041ceaefd99a55c3d71b69c64c7a5f3afd5be3ecc` |
| canonical semantic ledger | `913a80777f312a8f8caa81cca2712c640a99da8d2ef2445e8e8f064b35859fab` |

This remains a candidate for independent re-audit. It does not self-accept
P0.3. The independent audit remains byte-identical, and this repair did not
modify any authority, chapter, Bible, coverage contract, data file, or either
pinned source project.

## 12. Fourth independent-audit repair ledger

Status: **repaired candidate for independent re-audit; not self-accepted**
Triggering independent REJECT:
`2ec799126b07c4e7d9bde1e8f0342d498b47bbc3731724fa81bd10206a71628c`

Sections 1–11 are retained as revision history. This section supersedes their
default-review acceptance gate, semantic-proof digest, focused-test count, and
readiness statement. It does not alter the accepted 1,501/1,443/0 inventory,
the 140-callable AST fixed point, the nine previously repaired branch
contracts, or the exact 97-qualified-name/109-record Chapter 1–5 authority
join.

### The acceptance boundary is now 456 source defaults

The non-empty `default_branch_review` check is no longer the evidence of
semantic completeness. The legacy callable-level text remains in the ledger
only to preserve review history and the preceding proof surface.

The builder now parses the pinned read-only Payne Zero checkout and constructs
an exact, sorted source manifest for every defaulted parameter of every one of
the 140 public default-bearing callables. The fixed point is 456 records. Each
manifest record freezes:

- the exact qualified callable and source-spelled parameter name;
- the exact lexical default, including quote style and symbolic expressions;
  and
- the attribute-free AST of that default expression.

A separate structured semantic registry joins every one of those exact keys to
an effect category and a callable/parameter-specific explanation. The eleven
closed categories are `physical`, `data/source identity`, `cache`,
`environment`, `diagnostic`, `compatibility`, `parity/injection`,
`unsupported`, `algorithmic`, `continuous-value`, and
`dependency-injection-with-no-branch`. All eleven are exercised by the pinned
surface. Membership in `EXPLICIT_DEFAULT_CALLABLES`, a symbol responsibility,
or any other callable-level prose cannot satisfy this join.

Every default-bearing ledger record now embeds its complete
`default_parameter_reviews` mapping. Duplicate export/definition records carry
identical mappings. The complete 1,501-record proof projects the embedded
mapping and independently seals both the generated source manifest and the
semantic registry.

### Previously missing semantics are explicit

The records singled out by the rejecting audit now state the exact routes:

- `install_initializer_assets.generated_manifest_path` selects and hashes the
  installed initializer family and is retained as provenance;
  `include_direct_xh` selects the optional direct-[X/H] family; and `replace`
  distinguishes rejection from source-verified atomic replacement.
- all four defaults of `format_warm_start_deck` distinguish bulk metallicity,
  alpha enhancement, per-element absolute overrides, and generated versus
  caller-supplied publication titles.
- both public H2 population helpers distinguish unity LTE departure from the
  supplied quadratic depth multiplier and canonical molecular-equilibrium
  tables from alternate-data parity injection.
- molecular `load_catalog` distinguishes the package runtime cache from an
  explicit cache directory; `rebuild=False` accepts only a fingerprint-valid
  cache and reparses missing, stale, or corrupt content, while `True` forces
  source parsing.
- `window_invariants_for` binds all three table archives and `metal_chunk` to
  cache identity. Its `metal_chunk` record also names
  `PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1` as the independent
  environment route that bypasses reuse.
- `compile_molecular_text` distinguishes stored from energy-derived
  wavelengths and rejected from retained negative-energy predicted records;
  both choices enter compiled-cache identity.
- disabled-convection overshoot and top-layer zeroing explicitly change the
  retained endpoint diagnostic arrays.
- all four defaults of `save_product_structured_atmosphere` distinguish
  molecular versus atom-only chemistry, inferred versus explicit catalog
  roots, execution backend, and precision.

The structured records also retain the nine earlier named branch repairs at
parameter resolution: sampled-continuum scalar/materialized behavior, H-minus
LTE departure, the all-eight-array convection fallback, molecular
finite-difference sampling and restoration, transfer finalization,
asymmetric/symmetric convergence, both forced-prewarm routes, and the complete
atomic-catalog cache/parser matrix.

### Executable completeness and adversarial proof

The focused suite independently reparses the pinned source rather than trusting
the builder's registry to declare its own scope. It recovers the same 140
qualified callables, the same 456 parameter names, exact source segments, and
exact default ASTs.

The semantic validator requires exact record fields, exact manifest/review key
equality, all eleven categories, a callable-and-parameter-qualified
explanation prefix, and substantive text. It rejects generic markers even when
the adversarial test recomputes the expected registry digest around the
mutation.

An exhaustive adversarial loop applies four independent mutations to every one
of the 456 semantic records: deletion, parameter misnaming, category
reclassification, and source-default mutation. All 1,824 mutations fail
closed. Separate adversaries reject source-manifest drift and a digest-matched
generic placeholder. The complete proof also includes each embedded
`default_parameter_reviews` mapping, so ledger-only drift fails independently
of the registry checks.

### Reproducibility evidence and frozen identities

The focused suite has 34 passing tests. Ruff check and format check pass; both
Python files compile; `git diff --check` is clean for the four authorized
files.

Two clean raw-inventory builds from the read-only pinned checkout are
byte-identical to each other and to the checked inventory. Two semantic-ledger
builds from those independent inventories are byte-identical to each other and
to the checked ledger. Each reports 58 mapped modules, 1,501 mapped public
objects, and 1,501 reviewed overrides.

| object | SHA-256 |
| --- | --- |
| triggering independent audit, unchanged | `2ec799126b07c4e7d9bde1e8f0342d498b47bbc3731724fa81bd10206a71628c` |
| clean raw inventory | `010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf` |
| 140-callable/456-parameter source-default manifest | `6ee425f2d8dffe2c16ef40071d5ebe61a7c4c389580c144b789abf42da02e560` |
| 456-record structured semantic registry | `4f6e8365043d150cd7321be6331d24b95bf46aefec0ad7d4b2c0080afb147153` |
| complete executable semantic proof | `71bd5ac41963bfffcc70318a2a84f7b66368067898b03a297cd20ea477634fea` |
| semantic override builder | `689bddb683b5b66e62156cb90acb8f9ce50067185dc618175540cc98577479e3` |
| focused semantic tests | `082acd6b4b38991bb1358ab988829161a03055b3ddff87f013a736353710daed` |
| canonical semantic ledger | `02684b4412059137abbc0fed232a85a9e272dbd786fea97fba0609af705fa671` |

There are no known unresolved findings within this bounded repair. This remains
a candidate for independent re-audit and does not self-accept P0.3. The
triggering audit remains byte-identical, and this repair did not modify any
authority, chapter, Bible, plan, data manifest, or either external pinned
source project.

## 13. Fifth independent-audit repair ledger

Status: **repaired candidate for independent re-audit; not self-accepted**
Triggering independent REJECT:
`eafe4b5da285303f69dc3e59bd4b1553e9413e35f3f31db09b6637ce3b7c0e78`

Sections 1–12 remain revision history. This section supersedes their
default-parameter registry, category census, semantic-proof digest, focused
test count, file hashes, and readiness statement. The 1,501-record /
1,443-qualified-name / zero-`module_default` structural fixed point, the
140-callable default-bearing surface, and the exact Chapter 1–5 authority join
remain unchanged.

### One compact authority, 456 explicit contracts

The category-wide explanation generator is gone. The accepted semantic
authority is now
`audit/default_parameter_semantics.json`: one schema-versioned, sorted,
duplicate-free machine-readable file with exactly 456 records. Each record has
an exact `(qualified_name, parameter_name)` key and only these reviewed
semantic fields:

- effect category and branch behavior;
- concrete default and explicit/alternate routes;
- validation and coupled-branch conditions;
- exact consumer and any claimed forwarded target;
- `shared_semantics_with`; and
- the observable effect.

All 456 semantic bodies have distinct SHA-256 digests. All 456 are custom
records; no record inherits prose from a category, a parameter-name heuristic,
another callable, or a global template. No record currently uses
`shared_semantics_with`.

The Python builder no longer contains thousands of literal source snippets. It
hash-validates the compact authority, then reconstructs the source defaults
and all evidence from the pinned Payne Zero AST. A separate compact
`(category, semantic-body SHA-256)` registry remains literal in the builder
and is not loaded from the JSON authority. Rebinding the authority or the main
review digest around substituted prose or a false category therefore fails
before complete-proof sealing.

The authority is 5,995 lines / 678,603 bytes. During the repair the unrefactored
literal builder briefly reached 52,051 lines / 1,605,730 bytes; the final
builder is 7,884 lines / 337,784 bytes. The larger 66,147-line ledger is
generated audit output, not hand-maintained authority.

### Source-derived evidence and executable validation

At builder import/build time, the pinned source scan reconstructs, for every
one of the 456 exact keys:

- lexical default and attribute-free default AST;
- defining relative source path, source SHA-256, and definition line span;
- every direct parameter load with line, column, snippet, and snippet hash;
- directly parameterized and assignment-coupled `if`, conditional-expression,
  `while`, `assert`, and boolean-selection predicates;
- direct and assignment-coupled call-consumer arguments; and
- simple derived-name flow used to expose coupled predicates.

The resulting evidence covers 40 defining source files, 895 direct load sites,
430 predicate records (302 direct and 128 coupled), and 1,052 call-consumer
records (688 direct and 364 coupled). Eight explicit forwarding claims are
additionally resolved against the target source AST: the target callable must
be unique, the named target parameter must exist, and a matching keyword or
positional call edge must occur in the defining callable.

The validator requires exact source/evidence/semantic record shapes, exact
456-key equality, nonempty distinct default/alternate routes, all eleven
effect categories, 456 unique semantic bodies, and an exact independent
category/body expectation for every key. A no-branch claim is rejected when
the source evidence contains a direct or coupled predicate. Continuous-value
contracts must name the equation/output consumer and every direct/coupled
predicate by exact snippet or source line. Raw snippet hashes, source targets,
and target parameters are checked rather than trusted as prose.

### Re-reviewed routes singled out by Sections 37–39

The high-risk contracts now close the exact source behavior:

- the `-99.0e5` microturbulence sentinel selects the
  gravity/temperature-grid maximum, while other values use
  `abs(requested_maximum_velocity)`;
- swapped selected-line decoding runs the alternate halfword decode, scores it
  against native layout, and replaces fields only for a larger score;
- molecular chunk policy distinguishes the `CHUNK_LINES == 500_000`
  absent/empty/invalid-environment fallback, valid-environment integer clamp,
  explicit fallback override, and the higher-level explicit environment
  bypass;
- runtime device policy is CUDA then MPS then CPU when unspecified; dtype
  defaults to float32 on MPS and float64 otherwise, and MPS plus float64 is
  rejected;
- molecular-equilibrium diagnostics distinguish the four-object default return
  from the five-object diagnostic return;
- mixing length names the exact convection equation and positive threshold;
  transfer finalization also states when internally computed convection
  replaces the caller argument with configuration;
- initializer jitter names the finite/non-negative validation at L1021, the
  coupled zero-with-retries error at L1023, and the retry-displacement equation;
- the three dependency-injection contracts name
  `build_fast_exponential_tables`, `build_voigt_profile_basis`, and
  `load_hydrogen_line_profile_tables`, along with the injected arrays' effect
  on the returned exponential, Voigt profile, or H2 equilibrium constant;
- label synthesis distinguishes `r_grid` from its compatibility alias,
  resolves the dual-argument equality rule and 20,000 fallback, and rejects
  non-finite/non-positive grids;
- molecular `chain_length` names the clamped restart interval and modulo seed
  branch; sampled-continuum frequency invariants name scalar versus cached
  batched evaluation; the spectral operator names identity versus convolved
  wavelength/flux output; and saturated-core enforcement names strict
  unsupported-error versus fallback flux construction.

The source re-review changed four previously inherited classifications:
`synthesize_from_labels.r_grid` is algorithmic, its `resolution` spelling is
compatibility, sampled-continuum `frequency_invariants` is parity/injection,
and strict saturated-core enforcement is unsupported rather than diagnostic.
The frozen fifth-repair census is:

| effect category | exact records |
| --- | ---: |
| physical | 109 |
| data/source identity | 67 |
| cache | 13 |
| environment | 58 |
| diagnostic | 12 |
| compatibility | 19 |
| parity/injection | 88 |
| unsupported | 5 |
| algorithmic | 69 |
| continuous-value | 13 |
| dependency-injection-with-no-branch | 3 |
| **total** | **456** |

### Adversarial and reproducibility evidence

The focused suite has 36 passing tests. It exhaustively checks all 456 exact
keys against source default/AST, literal authority body, and independent
category/body hash. Representative structural deletion, misnaming,
reclassification, and source-default mutations fail the main digest. Stronger
adversaries recompute the affected main digests around:

- a category-template semantic substitution;
- a false category reclassification;
- an invented source-use line; and
- a no-branch claim for the microturbulence sentinel.

The first two fail the independent expected registry, the invented line fails
the pinned-AST comparison, and the no-branch attack still fails the source
predicate rule after both semantic expectation surfaces are adversarially
rebound.

Ruff check and format check pass; both Python files compile. Two canonical
semantic-ledger builds are byte-identical to the checked ledger, each reporting
1,501 reviewed overrides.

| object | SHA-256 |
| --- | --- |
| triggering independent audit, unchanged | `eafe4b5da285303f69dc3e59bd4b1553e9413e35f3f31db09b6637ce3b7c0e78` |
| compact 456-record semantic authority file | `142149023e566dcae55ae189a4eea52985aac729642cc87b5fc5ec87457f4692` |
| source-derived 456-record evidence registry | `8826f01ed70c75796d316435298a90c10c8b41c2213358f4aa62e87054cc4bf1` |
| loaded 456-record semantic contract mapping | `aef01a8feeb201ed46898e2929c81ac03b198e51ae0ad33ae06602cb531c8a9f` |
| independent expected category/body registry | `eff2ae6713302472c315a699a0e90079b4395f0107fae85a176dd29dd8b35e37` |
| merged source/evidence/semantic review registry | `39ca2d7a9e150d87ca171e6c7fefab0b4a316d7a870f7dfab36a8593a410088b` |
| complete executable semantic proof | `1df31068dc0920e46d715e3bcba236654db55779175efee9fb7914c5e19181b4` |
| semantic override builder | `5f16ef336e43cd9064aa781769b4d96984ab50d75fb9b442fc3e36e7d99a0e42` |
| focused semantic tests | `dc5ef87c67b9e04581a57febe90b5302ab1f07bb0efdb994e16d9fb01c407f81` |
| canonical semantic ledger | `a03253794a45b71225fd229f0bfe7ef0c7c02f7035cc132d4b4912511a30db8e` |

This is a bounded repair candidate for independent re-audit. It does not
self-accept P0.3. The triggering audit is byte-identical, and this repair did
not modify any authority chapter, Bible, plan, data manifest, or either
external pinned source project.

## 14. Sixth independent-audit repair ledger

Status: **repaired candidate for independent re-audit; not self-accepted**
Triggering independent REJECT:
`2a6902a6feb7734e581a5db92d4cc8b192702204ac4a5378cd5044def374d9a1`

Sections 1–13 are revision history. This section supersedes their
default-parameter authority schema, evidence census, semantic validation
design, proof digest, focused-test count, generated-ledger hash, and readiness
statement. The 1,501-record / 1,443-qualified-name / zero-`module_default`
ownership fixed point, 140-callable / 456-parameter default surface,
eleven-category census, and 97-qualified-name / 109-record Chapter 1–5
authority join remain closed.

### Schema-v2 source-fact authority

The authority is no longer a second prose surface. Its 456 sorted,
duplicate-free schema-v2 records contain only:

- the exact qualified callable and parameter;
- one exact AST evidence anchor with node kind, relation, line, snippet hash,
  and call/use metadata where applicable;
- a compact reviewed operation role;
- any source-resolved forwarding targets.

It contains no branch, default-route, alternate-route, validation, consumer,
or observable-effect narrative. The builder freshly reloads it from the
literal repository path, verifies the raw-file identity, verifies every anchor
against an independently reconstructed private pinned-AST snapshot, and checks
the reviewed role against concrete signature and source-operation facts.
Categories are returned by fixed control code, not a mutable role/category
registry.

All nine semantic fields are then rendered deterministically from verified
source facts. Branch and default routes use evaluated predicate outcomes and
guard polarities. Alternate routes name exact alternate branch heads or call
arguments. Validation names the exact predicates/coercions or states their
verified absence before a concrete source consumer. Consumers name exact load
sites, calls, forwarding targets, and derived assignments. Observable effects
are assembled from predicate outcomes, derived assignments, call arguments,
or exact expressions; uniqueness is not manufactured by inserting a callable
name into one shared sentence.

The exact recurrence census is:

| rendered field | distinct values | largest recurrence |
| --- | ---: | ---: |
| `branch_behavior` | 456 | 1 |
| `default_route` | 456 | 1 |
| `alternate_route` | 456 | 1 |
| `validation_and_coupling` | 456 | 1 |
| `consumer` | 456 | 1 |
| `observable_effect` | 456 | 1 |

The three rejected stock fields occur zero times: the 243-record generic
branch label, the 219-record generic validation sentence, and the 117-record
`None` alternate sentence have all been removed.

### Attribute-mediated `source_path` route

The evidence extractor now records derived attribute symbols, exact
assignment bindings, branch predicates over those attributes, and guard
polarity on downstream calls. For
`SynthesisPipeline.__init__.source_path`, it proves:

1. L1144 assigns
   `self._source_from_ref = source_path is not None`;
2. L1145 tests the derived attribute;
3. only the true guard reaches L1146 `np.load(source_path)`;
4. the `None` default makes the attribute false, skips all reference I/O, and
   leaves both source/scattering references `None`; and
5. `SynthesisPipeline.run` consequently follows the ordinary LTE Planck
   line-source construction, while a non-`None` path loads the reference NPZ
   arrays.

The source census retains all earlier evidence and adds the facts required to
close that missing flow:

| evidence kind | exact records |
| --- | ---: |
| parameter load sites | 895 |
| combined predicates | 431 (302 direct, 129 coupled) |
| separately identified attribute predicates | 1 coupled |
| call consumers | 1,054 (688 direct, 366 coupled) |
| derived assignment bindings | 242 (135 direct, 107 coupled) |

Thus the previous 430-predicate / 1,052-call surface remains present, while
the executable combined surface is larger rather than silently retaining the
known-incomplete counts.

### Hostile validation and source-operation roles

Semantic acceptance no longer depends on a rebound expected-body registry.
Validation freshly reconstructs expected contracts from the literal on-disk
facts and private AST evidence. The adversarial suite changes, together:

- both rendered semantic surfaces;
- the loaded authority;
- the public evidence object;
- the public role/category and role/description maps;
- the public authority path;
- every affected published digest.

The fifth-audit generic replacement for
`parse_atmosphere_deck.source` still fails because it differs from fresh
source-fact rendering. The false
`molecular_chunk_lines.default: algorithmic -> diagnostic` mutation first
fails the direct source-operation role check—the exact environment/chunk-bound
consumer has no diagnostic publication/shape operation—and then fails the
fresh on-disk authority comparison when all in-memory surfaces are rebound.
An independently forged source-use line still fails against the private AST.

### Canonical bytes and executable results

The generated ledger now uses the canonical serializer. It contains zero
literal U+2014 bytes and the eight required `\u2014` escapes. Two clean
canonical builds were byte-identical to each other and to the checked ledger,
all at
`7f72ebc2bb49e87ad6caa2aff3215ed47e31626bbed94253395809ff407bc7c5`.
Each reported 58 modules, 1,501 public records, and 1,501 reviewed overrides.

The focused semantic suite passed all 38 tests in 138.71 seconds. Ruff check,
Ruff format check, Python compilation, and bounded `git diff --check` all
passed.

| object | SHA-256 |
| --- | --- |
| triggering independent audit, unchanged | `2a6902a6feb7734e581a5db92d4cc8b192702204ac4a5378cd5044def374d9a1` |
| clean raw inventory | `010318f5804d5f18344ce07dccf2770f234c1de0cdf3294fec3d766054c65daf` |
| schema-v2 456-record fact authority | `f5a31c24b86b804a171258f0b342b3474a42a2cfda936635beb21177a462ae23` |
| source-derived evidence registry | `ea98ecc47f7e1a5e73fd0040cc7aafcf6b0bcecd47594071726309ad30e851e3` |
| deterministic 456-contract mapping | `c7f85610308df6fd3bd116e9f1a76c99edf54e0aa305a95ada532c8cefc39716` |
| merged source/evidence/semantic reviews | `a0e6c31c20affd465afddf2b087ff979f75bc6b9856def99498a8d16eb02caf0` |
| complete executable semantic proof | `b047f39b8ab7fc80001606c158e21092baf0391ea91c51eb0e69e2f3f7ff6bf7` |
| semantic override builder | `34320c62b3eb27560890cc956885fed378258458885685916c42ed51404a816d` |
| focused semantic tests | `cce0fa9909fb0f4dc51e9b60879097ae894468e72b1aec158dab34b7a6942952` |
| canonical semantic ledger | `7f72ebc2bb49e87ad6caa2aff3215ed47e31626bbed94253395809ff407bc7c5` |

This remains a bounded sixth-repair candidate for independent re-audit. It
does not self-accept P0.3. The independent audit was not modified, and this
repair did not edit governing/chapter/data files or either external pinned
source project.

## 15. Seventh independent-audit repair ledger

Status: **repaired candidate for independent re-audit; not self-accepted**
Triggering independent REJECT:
`a6d82c10cdbd04e101e39a12fcf6bd600096d07c35b89ca0962e342ef09d97d5`

Sections 1–14 are revision history. This section supersedes their treatment of
Python Boolean selection, downstream forwarding evidence, effect-role
validation, observable-effect recurrence, executable-test count, proof
digests, and generated-ledger hash. The retained fixed points remain 1,501
reviewed records, 1,443 qualified names, zero `module_default` records, 140
callables with 456 default parameters, 97 authority-qualified names with 109
joined records, and the unchanged eleven-category census.

### Exact short-circuit/default-dependency semantics

The source evaluator now implements Python `or` and `and` operand-return and
short-circuit behavior rather than reducing Boolean expressions to truth
values. A selection expression is evaluated at its left-hand selector before
the selected and rejected heads are rendered. Consequently:

- `fast_exponential_lookup.tables=None` deterministically selects
  `build_fast_exponential_tables()`;
- `evaluate_voigt_profile.basis=None` deterministically selects
  `build_voigt_profile_basis()`; and
- a truthy injected table or basis bypasses the canonical builder, while a
  falsey injected value follows Python `or` and still selects the builder.

Every former `<fall-through>` head is gone. An `if` without a source `else`
now says explicitly that execution continues, while a Boolean selector names
its actual first-operand and remaining-expression outcomes.

### Downstream source operations

Every one of the eight declared forwarding edges now carries a freshly parsed
`operation_facts` object for the exact target callable and parameter. The
object pins target source path and hash, definition span, node kind, direct or
coupled relation, exact line, snippet hash, and guard path. The eight edges
contain 185 verified target operations:

| target parameter | exact operations |
| --- | ---: |
| `deterministic_initializer_labels.jitter_scale` | 12 |
| `apply_temperature_correction.mixing_length` | 23 |
| `_resolved_r_grid.r_grid` | 14 |
| `_resolved_r_grid.resolution` | 14 |
| `_compute_at_freqs.frequency_invariants` | 103 |
| `_env_positive_int.default` | 7 |
| `_apply_spectral_operator_in_wavelength_density.spectral_operator` | 9 |
| `_solve_flux_rows.assert_no_saturated_core` | 3 |

Rendering incorporates those target operations into the route, validation,
consumer, and effect fields. Validation independently reconstructs the
operation facts from the pinned source and requires exact equality, so a
plausible prose claim without the target control/data-flow evidence cannot
pass.

The four high-risk wrapper contracts additionally pin and explain their exact
downstream behavior:

1. `molecular_chunk_lines.default` records the absent, empty, and invalid
   environment-value fallback to 500,000 and the final `max(1, value)` clamp.
2. `synthesize_from_labels.r_grid` records the exact 20,000 default, equality
   requirement when both grid spellings are supplied, finite-positive
   validation, and the resulting wavelength-grid sampling effect. The
   separate `resolution` compatibility spelling retains that distinct role.
3. `solve_structured_atmosphere.initializer_jitter_scale` records finite and
   non-negative validation, the zero-with-retries rejection, the default
   two-trial route, and the clipped label-width displacement equation.
4. `finalize_transfer_state.mixing_length` records the configuration
   forwarding and the target threshold/equation route that changes
   temperature corrections.

The other four wrappers retain their own exact downstream operation lists;
they were not accepted through a named-case exemption.

### Exclusive source-derived effect roles

The acceptance oracle is now one exclusive, source-derived classifier with
explicit precedence and negative exclusions across all eleven roles. It does
not accept a role merely because one token overlaps a broad set. In
particular, the unchanged
`synthesize_from_labels.resolution` anchor classifies only as compatibility,
not algorithm control, and `fast_exponential_lookup.tables` classifies only
as dependency injection, not parity/injection.

The exhaustive hostile test mutates each of the 456 authority facts through
each of the ten other roles—4,560 wrong-role authorities in total—while
leaving every other fact unchanged. Every mutation is rejected by the
production source-derived classifier without relying on checked-ledger
equality or an on-disk authority digest.

### Concrete observable effects and recurrence defense

The two stock effect conclusions previously covering 417 of 456 contracts now
occur zero times. Observable effects are assembled from verified selected and
rejected branch outcomes, exact builder calls or bypasses, derived
assignments, equations, state/output/arity/device/grid/cache mutations, target
operations, and exact use sites.

All six rendered semantic fields retain 456 distinct values with largest
recurrence one. A second test normalizes qualified names, parameters,
consumers, line/column numbers, numeric literals, and backticked source tokens
before counting frames. It first proves that synthetic unique-name
interpolation collapses to one frame, then checks the production contracts:
145 normalized frames, largest recurrence 35. Thus exact-string uniqueness
alone is no longer the recurrence defense.

### Canonical bytes and executable results

The source evidence census remains 895 parameter loads, 431 predicates (302
direct and 129 coupled), one separately identified coupled attribute
predicate, 1,054 call consumers (688 direct and 366 coupled), and 242 derived
bindings (135 direct and 107 coupled). All retained joins, anchors, source
paths, categories, generic/forged-anchor/attribute/no-branch adversaries, and
the exact `source_path` attribute-mediated route remain covered.

The focused semantic suite passed all 40 tests in 171.01 seconds, including
all 4,560 exclusive-role mutations. Ruff check, Ruff format check, Python
compilation, and bounded `git diff --check` all passed.

Two clean canonical builds were byte-identical to each other and to the
checked ledger. Each reported 58 modules, 1,501 public records, and 1,501
reviewed overrides. The JSON contains zero literal U+2014 bytes and exactly
eight canonical `\u2014` escapes.

| object | SHA-256 | lines | bytes |
| --- | --- | ---: | ---: |
| triggering independent audit, unchanged | `a6d82c10cdbd04e101e39a12fcf6bd600096d07c35b89ca0962e342ef09d97d5` | — | — |
| schema-v2 456-record fact authority | `f5a31c24b86b804a171258f0b342b3474a42a2cfda936635beb21177a462ae23` | 6,205 | 223,434 |
| source-derived evidence registry | `3256470cc72b73584318e90eaaf041ef7fae3fa7b20a33011698f69f70e85bf9` | — | — |
| deterministic 456-contract mapping | `001aacf28f0c66401cdcf5c43f33885b0125cfe0822ebd083060d6dc52fb80ae` | — | — |
| merged source/evidence/semantic reviews | `afb6b87d756dc9d4bff2fe2e9d67d347e59b2b1c83ce4120c364877b4fb718e9` | — | — |
| complete executable semantic proof | `43e2aac2994e068666e5dc1bf30ee17c387d0a11e8bba2b6c1081bfffb87b955` | — | — |
| semantic override builder | `c2e00e10caf93acac2dde2a47a65d3d7bb08bd17628f71917c93b0a6efea43a4` | 7,818 | 316,064 |
| focused semantic tests | `8ed79fc6fbe5cc0ed92b964af6671a2a3ee90d94adea4a48a1e4341f48d7565a` | 2,581 | 110,651 |
| canonical semantic ledger | `35314bcb7087592abce2ab9f589246262061049cfaa0dec9504796f1b70a9afe` | 78,767 | 6,081,592 |

This is a bounded seventh-repair candidate for independent re-audit. It does
not self-accept P0.3. The independent audit is byte-identical, and this repair
did not edit governing/chapter/data files or either external pinned source
project.

## 16. Eighth independent-audit repair ledger

Status: **repaired candidate for independent re-audit; not self-accepted**
Triggering independent REJECT:
`99c9e9a9d43ac3d8bb0524125ae533d8e4a98d4639fb26f2f974c6fc86379b55`

Sections 1–15 are revision history. This section supersedes their final-gate
trust model, effect-role support claim, normalized-effect census,
focused-test count, proof digests, generated-ledger hash, and readiness
statement. All retained ownership/default/join/category/source-evidence
counts, the eight forwarding edges with 185 operations, the high-risk
downstream routes, and canonical serialization requirements remain closed.

### A clean process owns the final semantic truth

`_validate_default_parameter_reviews` no longer treats any caller-supplied or
module-global AST/evidence object as its recomputation oracle. Each validation
starts a clean interpreter on the literal on-disk builder. That process:

1. resolves the pinned Payne Zero checkout independently of mutable caller
   paths;
2. rereads, hashes, and reparses the 40 source files;
3. reconstructs the 140-callable / 456-parameter source-default manifest and
   complete load/predicate/call/binding evidence;
4. rereads the literal schema-v2 authority and verifies all 456 anchors;
5. reparses all eight downstream targets and reconstructs their 185
   operations;
6. reclassifies roles and rerenders all 456 contracts; and
7. returns authority, manifest, evidence, contracts, and merged reviews as one
   fresh snapshot.

The mutable caller must equal that snapshot. The final validator never
consults the import-time private AST as an expected surface. The clean-process
loader is captured by the validator definition, so rebinding a module helper
or path does not redirect the trust boundary.

A combined hostile test appends a nonexistent L9999 binding and branch to
`parse_atmosphere_deck.source`, appends a nonexistent second attribute branch
to `SynthesisPipeline.__init__.source_path`, and appends
`erase_frequency_grid(frequency_invariants)` to the downstream operation
facts. It then coherently rebinds:

- both private and public evidence;
- both rendered semantic surfaces;
- authority and manifest objects;
- role/category and role/description maps;
- the pinned source root and authority path;
- the downstream target parser; and
- every affected published digest.

The full validator rejects the forged evidence against the clean-process
pinned AST. The older generic-template, category-reclassification,
public-evidence, forged-anchor, attribute, and no-branch adversaries remain
passing.

### One total role and zero empty-evidence acceptance

The operation classifier now fails before classification unless the evidence
has the complete structured list shape, a real parameter load, and an
operation containing the signature parameter. Emptying all operation lists
therefore leaves **zero** supported roles for every one of the 456 records.

For valid freshly parsed evidence, role support is defined only as equality to
the classifier's one total result. Direct enumeration of all eleven support
queries at every key gives:

| support cardinality | keys |
| ---: | ---: |
| 0 | 0 |
| 1 | 456 |
| 2 or more | 0 |

Thus the previous 39 dual-role support collisions are absent. The exhaustive
authority adversary still tries all ten wrong roles at each key—4,560
mutations—and all fail the production role validator. The test also asserts
support cardinality one directly from the clean-process evidence and asserts
that all eleven support queries are false for each corresponding empty
evidence object.

Authority anchors are independently projected from the fresh evidence
records. Each of the 235 call, 207 predicate, and 14 use anchors must be an
exact field-for-field source subset; predicate anchors must additionally match
the source node kind. No digest or frozen ledger supplies that equality.

### Terminal observable effects, not forwarding templates

The generic `supplies ... through ...` observable-effect renderer is removed.
Those two fragments and both previously rejected stock conclusions occur zero
times. Each effect now begins by naming the physical or operational quantity
being changed and the concrete returned or mutated state: abundance
coordinates, population solutions, species-specific continuum columns,
line-opacity accumulators, atmosphere labels/decks, convective corrections,
wavelength/flux products, caches, source artifacts, backend/precision pairs,
or another verified terminal family.

The effect then states whether the default selects deterministic guards,
runtime-coupled guards, derived state, calls, or direct expression flow. An
independently checkable operation-family clause records predicate kind and
relation, call relation, and derived-state relation. An exact source witness
pins the prose to one verified load. This witness preserves exact-contract
identity but is deliberately erased by the recurrence test; it is not the
semantic argument.

The previously omitted high-risk terminal effects now state and source-check:

- `return_diagnostics=False` returns the four-tuple of heavy-nucleus density,
  molecular populations, equation densities, and electron density, while
  `True` constructs and appends the diagnostic mapping as a fifth element;
- `detect_swapped_layout=False` performs only native decoding, while `True`
  also decodes and scores the swapped layout and adopts it only when its score
  exceeds the native score before constructing `SelectedLineCatalog`;
- the `requested_maximum_velocity=-99.0e5` sentinel bilinearly interpolates
  the temperature/gravity velocity grid, converts to cgs, and scales the
  remapped depth profile, while an explicit value supplies its absolute
  magnitude;
- `resolve_runtime` returns the resolved `(device, dtype)` pair, including
  backend-coupled precision and the MPS/float64 rejection;
- frequency invariants validate layout/grid length and drive cached
  vectorized H-minus, hydrogen, helium, metal, silicon, and scattering terms
  versus scalar-frequency recomputation, both returning absorption and
  scattering grids;
- a spectral operator converts to wavelength density, convolves total and
  continuum flux, replaces normalized flux/output wavelength, converts back
  with the output Jacobian, and publishes operator timing/name in
  `SpectrumResult`; and
- the saturated-core policy distinguishes strict rejection from computed
  saturated-row replacement in the returned surface-flux spectrum.

The molecular chunk fallback, 20,000 resolving grid, retry jitter, forwarded
mixing length, `source_path=None` LTE route, both Python `None or build_*()`
routes, and hydrogen-table loader retain their seventh-repair concrete
semantics.

### Independent recurrence proof

The recurrence test now aggressively removes:

- qualified callable, parameter, consumer, and callable-leaf spellings;
- snake-case and spaced variants;
- all identifiers independently extracted from source operations;
- every backticked snippet/witness;
- line/column markers; and
- numeric literals.

It observes exactly **356 normalized frames**, maximum recurrence **6**, with
64 recurrent frames. There is no acceptance threshold. For every recurrence,
the test independently constructs a family from effect category, terminal
route kind, predicate kind/relation set, call-relation set, derived-state
relation set, and complete downstream target operation-kind/relation set.
Every repeated frame has exactly one such source-operation family. A synthetic
four-to-five return-schema family and native-versus-score-selected layout
family are forced through the same cosmetic normalizer and correctly remain
an impermissible two-family collision.

All six rendered semantic fields still contain 456 distinct values with
largest exact recurrence one. The exact-string fixed point is retained, but
semantic recurrence acceptance no longer depends on it or on a frame-count
threshold.

### Canonical bytes and executable results

The focused suite passed all **41 tests in 367.38 seconds**. A post-format
critical subset covering the fresh anchor/normalization proof, 4,560 roles,
all downstream semantics, and the full coherent-forgery attack passed 4/4 in
30.60 seconds. Ruff check, Ruff format check, Python compilation, and bounded
`git diff --check` all pass.

Two final clean canonical builds are byte-identical to each other and to the
checked ledger. Each reports 58 modules, 1,501 public records, and 1,501
reviewed overrides. The checked JSON contains zero literal U+2014 bytes and
exactly eight canonical `\u2014` escapes.

| object | SHA-256 | lines | bytes |
| --- | --- | ---: | ---: |
| triggering independent audit, unchanged | `99c9e9a9d43ac3d8bb0524125ae533d8e4a98d4639fb26f2f974c6fc86379b55` | 2,273 | 122,931 |
| schema-v2 456-record fact authority, unchanged | `f5a31c24b86b804a171258f0b342b3474a42a2cfda936635beb21177a462ae23` | 6,205 | 223,434 |
| source-derived evidence registry | `3256470cc72b73584318e90eaaf041ef7fae3fa7b20a33011698f69f70e85bf9` | — | — |
| deterministic 456-contract mapping | `0494a4f6c008fcd6525fa4ddf1f2785161e4047703f9301bb537b255a9935f42` | — | — |
| merged source/evidence/semantic reviews | `e3a0dd68b933ed58174290f8c0d52d40c44876e6d1f7351bb50677950e4a3706` | — | — |
| complete executable semantic proof | `953d133ca3b3a527e6a63e68028edf2477c399ab83e367ac4e0063332bd3c4e9` | — | — |
| semantic override builder | `d3dd08a77a8b8d717c4b534fadac9c0b02c8ee70e1d4d623d1ddb0ceede0ec36` | 8,540 | 349,728 |
| focused semantic tests | `e3231c028c8ac9f5da2c8382a976311e0b68d228b65d34801878c61ee12448e0` | 3,148 | 132,773 |
| canonical semantic ledger | `c12f87df7510a247174a43c98ae22002b083a2673662fe46744caf6bd3fdc85c` | 78,767 | 5,985,981 |

This is a bounded eighth-repair candidate for independent re-audit. It does
not self-accept P0.3. The independent audit is byte-identical, and this repair
did not edit governing/chapter/data files or either external pinned source
project.

## 17. Ninth independent-audit repair ledger

Status: **repaired candidate for independent re-audit; not self-accepted**
Triggering independent REJECT:
`787554bceae4eb1c70ae41474daf7cb02e9a79ed77295f7e8d66fda1efebac3c`

Sections 1–16 remain revision history. This section supersedes their
observable-effect model, recurrence oracle and census, terminal-contract
claims, executable results, proof digests, generated-ledger hash, and final
readiness statement. The clean-process trust boundary, exact-one source role,
empty-evidence rejection, all wrong-role rejection, source/evidence counts,
and every earlier ownership/source/join gate remain closed.

### One strict source-derived operation signature per default

Each of the 456 default-parameter reviews now contains a structured
`operation_signature`. It is reconstructed from a fresh parse of the pinned
source and records:

- every relevant call's exact callee, operator, normalized positional values,
  normalized keyword names and values, affected argument bindings, relation,
  and exact call multiplicity;
- every derived binding's target, normalized value, AST value, relation, and
  exact binding multiplicity;
- every parameter-controlled assignment's operator, targets, normalized
  value, and exact assignment multiplicity;
- every predicate's operator, normalized expression, source relation,
  independently evaluated default outcome, true head, false head, and exact
  branch multiplicity;
- the outer callable's explicit and implicit return nodes, normalized return
  values, arity, source-level type, element types, container cardinality,
  guards, annotation, and complete assignment census;
- the distinction among returned values, caller-owned in-place mutation, and
  return-plus-mutation, including exact propagated mutation targets;
- every declared downstream target's exact operation count, operation
  operator/relation/value, and independently derived terminal contract; and
- the concrete source parameter, returned quantities, and mutated quantities.

The location-free family digest removes only locations and the source
parameter's cosmetic spelling. It retains every call value, binding value,
assignment value, branch outcome, downstream operation, terminal kind, return
value/type/arity, and mutation target. The former relation-only family is no
longer an acceptance oracle.

The hand-authored physical-subject and terminal-sink tables have been removed,
as have all special-case observable-effect overrides. One renderer now names
the source callable and parameter, the concrete returned or mutated
quantities, exact default branch outcomes, complete normalized call
arguments, complete controlled assignments, exact multiplicities, the
terminal value/mutation contract, and any downstream terminal contract.
Across all 456 records there are zero `results or mutations` fallbacks, zero
` supplies ` terminal phrases, and 456 distinct exact observable effects.

### The rejected terminal collision is structurally separated

The three records in the rejected frame now have different source-derived
families and terminal contracts:

| callable and parameter | terminal kind | exact terminal contract |
| --- | --- | --- |
| `payne_zero_atmosphere.equation_of_state.iterate_electron_density.max_iterations` | `in_place_mutation` | explicit/implicit `None`; updates exactly `state.charge_square_density`, `state.electron_density`, `state.ion_stage_populations_by_packed_slot`, `state.mass_density`, and `state.total_nuclei_number_density` |
| `payne_zero_synthesis.equation_of_state.solve_population_state.max_iter` | `return_value` | one `PopulationState`; no caller-owned mutation |
| `payne_zero_synthesis.molecular_equilibrium.solve_molecular_equilibrium.max_iter` | `return_value` | a five-value tuple on the diagnostic route or a four-value tuple otherwise; no caller-owned mutation |

Their location-free family digests are respectively
`57adccb92a6a944b0029514d0a2c638aa25f63f7e2551fc8adeb3de899b278a9`,
`3ef006f42a10b59b67b4eac66f9a67bc9333d32a7846b9aac5f4e75eac1a82e9`,
and
`0935fa0683e80d09a166b0e8c6eb3f3f6ce57d7c20e243c6c792acef85b80f11`.
The test asserts all three schemas, all five atmosphere mutation targets, and
the three-family separation directly.

Across the complete registry, terminal kinds are 410 `return_value`, 29
`return_and_in_place_mutation`, and 17 `in_place_mutation`. Nested local
function returns are excluded from the owning callable, list/set/dict returns
retain one return slot plus their container cardinality, annotations such as
`PopulationState` remain authoritative, implicit fallthrough is explicit,
and same-source procedure mutation is propagated only after formal-to-actual
argument substitution proves a caller-owned target.

### Non-vacuous recurrence proof

The new prose normalizer erases callable, callable-leaf, parameter, consumer,
line, and column identity. It deliberately preserves exact callees,
operations, argument/binding values, targets, return values, types, arities,
branch outcomes, and numeric values because those are semantic facts rather
than cosmetic identity.

The exact census is:

| normalized-effect property | value |
| --- | ---: |
| records | 456 |
| normalized frames | 455 |
| recurrent frames | 1 |
| records in the recurrent frame | 2 |
| maximum recurrence | 2 |
| recurrent frames with more than one strict family | 0 |

The one real recurrence is the pair
`prepare_population_state.molecular_thermal_energy_erg` and
`prepare_structured_handoff_population_state.molecular_thermal_energy_erg`.
Both independently reduce to the same strict family
`7433539d879d0728f5d733af380c7fd2a229c4911852d367d59885df2a07ef69`.
The test therefore exercises a real recurrent group rather than accepting a
vacuous zero-recurrence result or relying on a synthetic collision.

For every recurrent group, the test requires `family_count == 1` and separately
requires equality of all evidence counts, exact callees, normalized positional
and keyword arguments, affected bindings, binding targets/values, assignment
targets/values, predicates/outcomes, terminal kind/value/type/arity/mutation
targets, and downstream terminal schemas.

### Independent fresh-source recomputation

The recurrence test does not ask the builder to certify its own signature. A
second implementation inside `tests/test_symbol_coverage.py` starts from the
clean-process snapshot, rereads and reparses the pinned source, and
independently reconstructs all 456 signatures. It has its own:

- source/module/function caches and callable ownership traversal;
- call, binding, controlled-assignment, and branch extraction;
- pure default-expression evaluator and branch-outcome evaluator;
- explicit/implicit return, type, arity, container, and guard analysis;
- recursive same-source mutation proof with formal-to-actual substitution;
- downstream source-node verification and terminal analysis; and
- canonical location-free family serialization and SHA-256 computation.

The independent result must equal every field of every clean-process
`operation_signature`, including all counts and the family digest. It does not
call the production signature builder, terminal analyzer, evaluator,
normalizer, or digest helper.

### Canonical bytes and executable results

The full focused semantic suite passed all **41 tests in 1,647.13 seconds**.
After final Ruff formatting, the critical trust-boundary subset covering all
high-risk forwarded terminals, all 456 independently recomputed signatures,
all 4,560 wrong-role mutations, and the coherent private-AST/downstream
forgery passed **4/4 in 172.23 seconds**.

Ruff check, Ruff format check, Python compilation, and bounded
`git diff --check` pass. Two final post-format canonical builds were
byte-identical to the checked ledger, each reporting 1,501 reviewed
overrides. The checked JSON contains zero literal U+2014 bytes and exactly
eight canonical `\u2014` escapes.

| object | SHA-256 | lines | bytes |
| --- | --- | ---: | ---: |
| triggering independent audit, unchanged | `787554bceae4eb1c70ae41474daf7cb02e9a79ed77295f7e8d66fda1efebac3c` | 2,468 | 133,570 |
| schema-v2 456-record fact authority, unchanged | `f5a31c24b86b804a171258f0b342b3474a42a2cfda936635beb21177a462ae23` | 6,205 | 223,434 |
| source-derived evidence registry | `3256470cc72b73584318e90eaaf041ef7fae3fa7b20a33011698f69f70e85bf9` | — | — |
| deterministic 456-contract mapping | `176164024984435354a95b7c0cada7a70ad972135167075e2ab41cd1669287e8` | — | — |
| merged source/evidence/semantic reviews | `0cf4fefe0da1e13c4194890e9b91ef517d6410a4e70d071640a947550cf4eb65` | — | — |
| complete executable semantic proof | `eecd8272cc77f3a015243d6d453ec64dc14cebcfa5e2f09045aab0fd012c14b6` | — | — |
| semantic override builder | `138ed0af4791e8e14728259fee2bdc96c944478d6bbf4b1674127e964b4ed8a5` | 8,921 | 354,891 |
| focused semantic tests | `4de194e1170c92785fc1472449edcc30f76568327dd27a320ef9f12696146632` | 3,996 | 170,778 |
| canonical semantic ledger | `81891bdf35c0ed18898773d4cadf1c5341b268b36cc8adad616dc7088121cdcf` | 252,308 | 14,602,544 |

This is a bounded ninth-repair candidate for independent re-audit. It does
not self-accept P0.3. The triggering audit and compact authority are
byte-identical to their pre-repair bytes. This repair edited only the
authorized builder, focused tests, generated semantic ledger, and this
candidate report; it did not edit the governing audit, chapter, data, source,
or either external pinned Payne Zero project.
