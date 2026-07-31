# Chapter 5 oracle worker detached acceptance record

Status: scientifically accepted capture worker; publication not authorized  
Acceptance date: 2026-07-30  
Independent reviewer task: `chapter05_oracle_reaudit`

## 1. Accepted identity tuple

This record applies only to the following exact local and upstream identity
tuple:

| identity | accepted value |
|---|---|
| Payne Zero commit | `9c44001feae40b85146630499e6f8a5fed42e5af` |
| input fixture SHA-256 | `ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae` |
| raw fixture-payload digest | `4bcf0bbd8d61e58334c4c7ef6caaaf9ca47e6fb4536ad0098d5a541d540ec048` |
| oracle worker SHA-256 | `429252d5fefd2b911ce4321578820aa67b505d5fe37b174cb647d4b6177d7389` |
| oracle capture contract SHA-256 | `4198f76419f102efbb3468b5f2ed7ddca7ff6af776ffc566cfa4058b6164fdaf` |
| frozen capture-schema version | `2` |
| capture key count | `1161` |
| capture-schema digest | `652c110dc79a6f6dfca6893bee35416289675b4920a5d0dcfe6b2cb262dacf3d` |
| loaded pinned Python source count | `52` |
| accepted physical-payload fingerprint | `d223351fa2c51dc24a1b01896da9ab9a82fc475f4082c47fde34734d8dc03343` |
| accepted full-capture fingerprint | `3d2c131711e1c0dc6aa088892193bb24d41a76d005bc20dd1c42d3e84f66e656` |

The physical fingerprint binds the raw fixture-content digest and all captured
physical inputs, grids, boundaries, counterfactuals, components,
intermediates, and outputs. The full fingerprint additionally binds the local
worker and contract hashes, environment and library metadata, pinned source
and table identities, schema metadata, and capture-scope metadata.

## 2. Independent fresh-cache executions

Both accepted executions were launched from
`/Users/ysting/stellar-spectroscopy-from-scratch-gpu` in independent
processes. Each invocation received a newly created empty Numba cache
directory.

Run A:

```bash
cache_dir=$(mktemp -d /tmp/chapter05-reaudit-v2.XXXXXX)
env \
  MKL_DYNAMIC=FALSE \
  MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  LC_ALL=C \
  PYTHONHASHSEED=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  TZ=UTC \
  NUMBA_CACHE_DIR="$cache_dir" \
  /Users/ysting/anaconda3/bin/python scripts/chapter05_oracle_worker.py
```

Run B:

```bash
cache_dir=$(mktemp -d /tmp/chapter05-reaudit-v2b.XXXXXX)
env \
  MKL_DYNAMIC=FALSE \
  MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  LC_ALL=C \
  PYTHONHASHSEED=0 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 \
  TZ=UTC \
  NUMBA_CACHE_DIR="$cache_dir" \
  /Users/ysting/anaconda3/bin/python scripts/chapter05_oracle_worker.py
```

The two processes returned identical accepted identities, physical
fingerprints, full fingerprints, and numerical summaries:

| summary field | accepted result in both runs |
|---|---|
| `capture_scope_complete` | `true` |
| `golden_publication_performed` | `false` |
| atmosphere product shape, every regime | `(6, 30000)` |
| standard synthesis product shape, every regime | `(6, 36)` |
| sampled diagnostic shape, every regime | `(6, 27)` |
| sampled extension shape, every regime | `(6, 12)` |
| sampling-boundary grid shape | `(8, 30000)` |
| representative line-reference threshold shape | `(6, 344)` |
| maximum absolute atmosphere-source residual | `8.673617379884035e-19` |
| maximum absolute synthesis-interpolation residual | `1.1641532182693481e-10` |
| IFOP-13-only scattering | exactly zero |
| CIA lower-column weights | `[0.0, 0.5, 0.0]` |
| CIA upper-column weights | `[1.0, 0.5, 1.0]` |

Successful completion also means that the worker's fail-closed assertions
passed for the exact 52-file source set before and after the capture lanes,
all eight active line-reference counts, component and per-layer regime
activation, molecular-entry and H2 ownership seams, IFOP 4/13 and IFOP 19,
the split synthesis minor terms, packaged edge-sample identity, unused-edge
call tracing, rich/trimmed H II ownership, stored-H2 invariance, signed-edge
invariance, all principal shapes and dtypes, the complete schema digest, and
all elementwise residual gates.

## 3. Negative process probes

### 3.1 Populated-cache rejection

Run A's now-populated `NUMBA_CACHE_DIR` was passed to a second otherwise
identical process. The process failed before loading the Payne Zero capture
modules with:

```text
OracleEnvironmentError: NUMBA_CACHE_DIR must be truly empty at process start
```

Disposition: pass. Cache reuse is fail-closed.

### 3.2 Symlink-cache rejection

An empty temporary directory was created and an unresolved symlink was pointed
at it. The symlink path, rather than its target, was supplied as
`NUMBA_CACHE_DIR` to:

```bash
env \
  MKL_DYNAMIC=FALSE MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 LC_ALL=C PYTHONHASHSEED=0 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TZ=UTC \
  NUMBA_CACHE_DIR="$cache_link" \
  /Users/ysting/anaconda3/bin/python -c \
  'from scripts.chapter05_oracle_worker import require_environment; require_environment()'
```

The process failed with:

```text
OracleEnvironmentError: NUMBA_CACHE_DIR must not be a symlink
```

Disposition: pass. The unresolved symlink check is effective.

## 4. Independent scientific verdict

The exact worker identified above is accepted as a deterministic and
scientifically complete Chapter 5 atmosphere-and-synthesis continuum capture
worker. The earlier blockers concerning incomplete full products, hidden
minor terms, missing physical seams, incomplete schema validation, local
capture provenance, raw fixture binding, post-execution source verification,
global-scale residual checks, inactive-source construction, cache symlinks,
and regime activation are closed.

This acceptance permits design and review of a publisher. It does not
authorize serialization or publication. A publisher must fail closed against
the complete accepted identity tuple above before writing any artifact and
must keep two products distinct:

1. a governance/test integration projection derived losslessly from the
   accepted ephemeral 1161-key capture; and
2. a small, explicit pedagogical allowlist for the reader golden.

The raw 1161-key capture need not itself be published. No archive, golden,
manifest, executable, or upstream file was created or modified during this
acceptance audit.

## 5. Re-acceptance rule

Any change to the worker, capture contract, fixture, pinned Payne Zero source
or data, schema version, key count, schema digest, physical fingerprint, or
full fingerprint invalidates this record. Re-acceptance requires new
independent empty-cache runs and new negative cache probes.
