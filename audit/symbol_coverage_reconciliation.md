# Public-symbol coverage reconciliation

Date: 2026-07-30  
Pinned Payne Zero commit: `9c44001feae40b85146630499e6f8a5fed42e5af`

## Authority rebind

`content/Chapter01.ipynb` is rebound to SHA-256
`093d57068ac24fc89f1fc3e5069f515f0bd69af8e237f69d3e2688b387877499`.
Before rebinding, the notebook was checked against
`book.chapters.chapter_01.build_notebook()`:

- all 50 cell types, IDs, tags, and normalized sources matched;
- all 17 code cells had execution counts;
- no stored error output was present.

The accepted-artifact manifest and complete semantic-proof digests were
recomputed only after that check.

## Status result

The old ledger contained 908 `planned` records. Their disposition and final
status are:

| Semantic disposition | Verified | Implemented | Still planned | Total |
| --- | ---: | ---: | ---: | ---: |
| taught | 515 | 0 | 2 | 517 |
| composed | 82 | 54 | 7 | 143 |
| plumbing-only | 189 | 23 | 21 | 233 |
| diagnostic-only | 2 | 4 | 2 | 8 |
| compatibility-only | 7 | 0 | 0 | 7 |
| **Total** | **795** | **81** | **32** | **908** |

The complete 1,501-record ledger changes from:

| Status | Before | After |
| --- | ---: | ---: |
| integrated | 401 | 401 |
| verified | 131 | 926 |
| implemented | 0 | 81 |
| planned | 908 | 32 |
| boundary | 61 | 61 |

No record was promoted to `integrated`: final whole-book review remains a
separate gate. Unsupported and intentionally excluded branches remain
`boundary`.

## Module grouping of the original 908 planned records

The entries below show only modules that contributed at least one old
`planned` record.

| Module | Verified | Implemented | Still planned |
| --- | ---: | ---: | ---: |
| `payne_zero_atmosphere.__init__` | 81 | 13 | 6 |
| `payne_zero_atmosphere.atmosphere_io` | 7 | 0 | 0 |
| `payne_zero_atmosphere.cli` | 0 | 1 | 1 |
| `payne_zero_atmosphere.config` | 24 | 0 | 1 |
| `payne_zero_atmosphere.constants` | 12 | 0 | 0 |
| `payne_zero_atmosphere.convection` | 23 | 0 | 0 |
| `payne_zero_atmosphere.convergence` | 3 | 0 | 0 |
| `payne_zero_atmosphere.data_files` | 6 | 0 | 0 |
| `payne_zero_atmosphere.direct_abundance` | 69 | 0 | 0 |
| `payne_zero_atmosphere.hydrogen_line_profile` | 65 | 0 | 1 |
| `payne_zero_atmosphere.hydrostatic` | 2 | 0 | 0 |
| `payne_zero_atmosphere.install_runtime_data` | 0 | 0 | 10 |
| `payne_zero_atmosphere.line_catalog` | 28 | 0 | 0 |
| `payne_zero_atmosphere.line_opacity` | 6 | 0 | 0 |
| `payne_zero_atmosphere.line_profile_math` | 17 | 0 | 2 |
| `payne_zero_atmosphere.line_selection` | 8 | 0 | 0 |
| `payne_zero_atmosphere.microturbulence` | 1 | 0 | 0 |
| `payne_zero_atmosphere.prewarm` | 0 | 4 | 1 |
| `payne_zero_atmosphere.radiative_pressure` | 9 | 0 | 0 |
| `payne_zero_atmosphere.radiative_transfer` | 9 | 0 | 1 |
| `payne_zero_atmosphere.rosseland_mean` | 1 | 0 | 0 |
| `payne_zero_atmosphere.run_setup` | 28 | 0 | 0 |
| `payne_zero_atmosphere.runner` | 3 | 52 | 1 |
| `payne_zero_atmosphere.runtime_state` | 1 | 0 | 0 |
| `payne_zero_atmosphere.source_catalogs` | 10 | 0 | 0 |
| `payne_zero_atmosphere.synthesis_bridge` | 1 | 1 | 1 |
| `payne_zero_atmosphere.temperature_correction` | 23 | 0 | 0 |
| `payne_zero_atmosphere.transfer_kernels` | 2 | 0 | 0 |
| `payne_zero_atmosphere.warm_start` | 37 | 0 | 0 |
| `payne_zero_synthesis.__init__` | 10 | 0 | 0 |
| `payne_zero_synthesis.api` | 26 | 0 | 1 |
| `payne_zero_synthesis.atmosphere` | 3 | 0 | 0 |
| `payne_zero_synthesis.atomic_lines` | 33 | 0 | 4 |
| `payne_zero_synthesis.cli` | 0 | 0 | 1 |
| `payne_zero_synthesis.constants` | 18 | 0 | 0 |
| `payne_zero_synthesis.device` | 6 | 0 | 0 |
| `payne_zero_synthesis.hydrogen_lines` | 51 | 0 | 0 |
| `payne_zero_synthesis.line_opacity` | 61 | 0 | 0 |
| `payne_zero_synthesis.molecular_lines` | 49 | 0 | 0 |
| `payne_zero_synthesis.paths` | 0 | 10 | 0 |
| `payne_zero_synthesis.pipeline` | 36 | 0 | 0 |
| `payne_zero_synthesis.prewarm` | 2 | 0 | 1 |
| `payne_zero_synthesis.radiative_transfer` | 16 | 0 | 0 |
| `payne_zero_synthesis.source_catalog_molecular_compiler` | 6 | 0 | 0 |
| `payne_zero_synthesis.synthesis` | 2 | 0 | 0 |

Most promotions follow the current module-level gates in `COVERAGE.md`. The
105 narrower symbol promotions are separately enumerated in
`VERIFIED_STATUS_PROMOTION_GROUPS`; they are supported by current Chapter
6--15 runtime tests. Package-export aliases inherit the exact target status.

## Intentionally unpromoted classes

All 32 remaining `planned` records belong to one of these evidence gaps:

1. **Runtime-data installer (10 records).** No dry-run/install/manifest
   mutation matrix currently exercises `install_runtime_data.py`.
2. **Unexercised loader matrices (8 records including aliases).** Default
   versus explicit path and cold/warm/forced-reload behavior is not completely
   gated for the hydrogen-continuum selector, line-opacity tables,
   hydrogen-profile tables, or radiative-transfer tables.
3. **Physical convergence and converged-only publication (6 records including
   aliases).** The full `run_atmosphere_model` trajectory, converged product
   writer, output publication field, and atmosphere CLI publication workflow
   require an accepted converged atmosphere trajectory.
4. **Public synthesis CLI (1 record).** Alias/conflict/product CLI behavior
   does not yet have its declared smoke gate.
5. **Prewarm workflows (2 records).** Component cache behavior is tested, but
   the complete reuse/force/stale prewarm-manifest matrices are not.
6. **Atomic parser/cache option matrices (4 records including duplicate export
   and definition records).** Compact catalog behavior is verified, but the
   full default/explicit source, sort, isotope, corruption, and forced-rebuild
   cross-product is not.
7. **`Spectrum.save_npz` (1 record).** The current capstone validates spectrum
   arrays and initialized-atmosphere serialization, not a spectrum-file
   round trip.

The 81 `implemented` records are also deliberately not called verified. They
cover the staged full runner/CLI/prewarm/publication seam and synthesis path
resolution whose remaining blockers are physical convergence or mixed-process
integration, not missing code.
