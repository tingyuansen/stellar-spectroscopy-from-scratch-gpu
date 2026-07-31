# Chapter 5 publisher implementation acceptance

Status: **published; publisher, artifacts, authorization, and manifest independently accepted**

This record began as the separate acceptance of the publisher program. The
later candidate, authorization, publication, and post-publication manifest
decisions are recorded below so that the current state cannot be confused with
the historical prepublication gate.

## Four distinct gates

1. **Oracle-worker acceptance — accepted and frozen.** The publisher pins the
   accepted worker, capture contract, exact source contract, deterministic
   writer, fixture archive, raw schema, and raw fingerprints. Those identities
   define what may enter the publisher.
2. **Publisher-implementation acceptance — accepted after repair and fresh
   independent re-audit.** The accepted implementation retains the
   publication-safety boundary and now enforces the frozen physical
   deduplication contract with explicit, bitwise-gated semantic routes.
3. **Candidate-artifact acceptance — accepted.** The repaired reader and
   integration candidates passed an independent scientific, reconstruction,
   ownership, and deterministic-byte audit.
4. **Publication authorization and publication — accepted and completed.**
   The independently reviewed
   `design/chapter05_publication_acceptance.json` record authorized exactly
   those two candidates against the reviewed prepublication manifest. Atomic
   publication succeeded once. The synchronized exhaustive manifest then
   passed an independent post-publication audit.

## Stable implementation identities

| Item | SHA-256 |
|---|---|
| `scripts/build_chapter05_payne_zero_goldens.py` | `6e5ed476cbcd64b1599c9f369bf5ed746d9f561f9bd77c13db8b9ec1c1cdad81` |
| `tests/test_chapter05_golden_publisher.py` | `074975d36e8aa57f8173eebfe2e24eec6df52cff0a7fc8619ec4139d32144a43` |
| `design/chapter05_publisher_contract.md` | `8a58866b315b2cdd1ca769436aa0ccdbed3b08e4ef180bd309179d6cc7e04c08` |

The publisher pins these already accepted oracle identities:

- oracle worker:
  `429252d5fefd2b911ce4321578820aa67b505d5fe37b174cb647d4b6177d7389`
- oracle capture contract:
  `4198f76419f102efbb3468b5f2ed7ddca7ff6af776ffc566cfa4058b6164fdaf`
- exact source contract:
  `ec1c84519a898a454a408780bd6b36fa723fc7bade83a98e11eede1343bf2956`
- oracle-worker acceptance record:
  `892279cd2dfa850c3eabfb8ce94953e4723db5270da319bcedb9bc90c9597474`
- deterministic writer:
  `f4886766524d79623e648d28ab9d24215da42f8bf1f859f69381c546f9c96e49`
- fixture archive:
  `ef246acd1e7dbf0b6c781613dad8c67c1cfd2c1f27c4ce1b8639ce2318bbb7ae`

The accepted raw capture has schema version 2, 1,161 keys, schema digest
`652c110dc79a6f6dfca6893bee35416289675b4920a5d0dcfe6b2cb262dacf3d`,
fixture-payload digest
`4bcf0bbd8d61e58334c4c7ef6caaaf9ca47e6fb4536ad0098d5a541d540ec048`,
physical fingerprint
`d223351fa2c51dc24a1b01896da9ab9a82fc475f4082c47fde34734d8dc03343`,
and full fingerprint
`3d2c131711e1c0dc6aa088892193bb24d41a76d005bc20dd1c42d3e84f66e656`.

## Frozen output schemas and ownership

| Product | Key count | Schema digest |
|---|---:|---|
| reader | 257 | `058389fcd0e944dd4c1ad1208adbaac44d53ab35e61e5366d37a8a141ad91f88` |
| integration | 1,079 | `e09a8932d97f8a2756aca2b779c4b8fdd822ca239197fd908ee575d09df081ca` |

The exact source-name, disposition, and destination-member vectors are frozen
together by inventory-mapping digest
`b02a6a2896d3468d1052441f8def0841b608eb5d5816ccd72539a0ec982c452c`.
This closes the ambiguity left by validating only the vector lengths and
allowed disposition vocabulary.

## Verification evidence

The following command completed successfully twice, as two separate real
invocations against the accepted fixture and oracle:

```text
PYTHONPATH=.:src /Users/ysting/anaconda3/bin/python \
  scripts/build_chapter05_payne_zero_goldens.py \
  --full-capture --verify-only
```

It launched two independent child captures in distinct, initially absent cache
directories; compared their raw archives byte for byte; assembled both product
pairs independently; and compared each final archive byte for byte. The
temporary candidate identities were:

| Product | Bytes | Verify-only SHA-256 |
|---|---:|---|
| reader | 1,022,580 | `84e3e764b2d9be8d409b0db0271decdd90e07b8e716f5c589fbdbb1b9e2ae39f` |
| integration | 21,608,505 | `5ce6805353ed760dc42c42ce6bd59472a3b7a7443a486aded175caf755fbff73` |

These repaired reproducible candidate identities are also the installed
golden identities. Candidate-artifact acceptance and publication
authorization remained separate decisions; both subsequently passed.

Both invocations completed their own internal A/B raw-capture and final-product
agreement before yielding this same pair. These hashes first served as
reproducibility evidence and became final golden identities only after the two
later gates passed.

Additional stable-file checks:

- publisher tests: 29 passed, 1 skipped;
- cross-layer Chapter 5, manifest, and schematic tests: 114 passed, 1 skipped;
- Ruff: all checks passed;
- `git diff --check`: clean.

After the detached record was created, the CLI parser test was made
phase-aware: exact `--full-capture --publish` arguments are accepted only when
the strict gate validates, while every incomplete, forced, redirected, or
ungated publication form remains rejected. This test-only update does not
change publisher or artifact bytes.

The publisher is anchored to the external canonical workspace root
`/Users/ysting/stellar-spectroscopy-from-scratch-gpu`. Canonical
`--identity-only` succeeds. A byte-identical publisher copied to another tree
is rejected before NumPy or repository imports for both identity and
publication invocations, and creates no output. This prevents a relocated copy
from redefining repository identity or publication destinations.

The test suite includes direct same-schema mutations of both mappings cited in
the last P1 review: changing `reader` to `reader_alias`, and changing
`common_identity_metadata` to `integration`. Both are rejected by semantic
validation.

The repaired ownership map routes line-reference, diagnostic-frequency,
extension-frequency, selected-frequency, signed-edge, and nine per-regime
counterfactual products to reader-owned members or slices. The other 27
`FrequencyInvariants` fields, five remaining standard-trace fields, and flipped
signed-edge vector each have one physical integration owner. All declared
routes require exact dtype, shape, and bytes before coalescing or aliasing.
Logical reconstruction still returns all 1,161 raw members with the accepted
schema and both accepted fingerprints. The independent re-audit found only
documented semantically distinct equal-valued arrays, not remaining ownership
duplicates.

The subsequent publication-lifecycle P1 is closed in implementation by moving
authorization out of the publisher. Tests reject malformed hexadecimal
identities, unknown and duplicate fields, wrong publishers, wrong schemas,
wrong candidate hashes and sizes, changed manifests, symlink and nonregular
record paths, symlink candidates, injected publication functions, and direct
API calls without the record. Every negative direct-API case proves that the
destination parent is not created.

### Retired pre-repair candidate

The artifact review rejected the earlier integration schema with 1,178
members because it physically duplicated reader-owned and regime-invariant
scientific arrays. Its retired identities are retained only as audit history:

| Product | Bytes | Rejected SHA-256 |
|---|---:|---|
| reader | 1,022,580 | `661f4a1b89367438439034059fcd372adc00394971a01b43b87f5883958247d5` |
| integration | 21,848,543 | `bc0b8f6ef2dd6c767820a12cf42b5dd29ffe89e839c76ce19a16a0af5d643cf1` |

## Publication state

The publisher contains no in-source authorization hash or accepted final
artifact hashes. Instead, `publication_gate_ready()` strictly parses the
canonical detached JSON record, checks its regular-file and no-symlink
identity, and validates the current publisher, publisher contract, exact
candidate names/hashes/sizes/schema identities, and prepublication manifest.
The double-capture driver compares both final candidates with the record before
calling publication. The direct atomic API independently repeats the gate,
mandatory semantic validation, candidate byte checks, and manifest agreement
before writing and again around staging.

The canonical detached publication record now exists with SHA-256
`80bcfea8e6a817ac1a279b136a247cdd482af1b94af312e94dfa564ad40cafaf`.
Its mutation and no-write audit passed 26 negative cases. Atomic publication
installed:

- reader golden: 1,022,580 bytes,
  `84e3e764b2d9be8d409b0db0271decdd90e07b8e716f5c589fbdbb1b9e2ae39f`;
- integration golden: 21,608,505 bytes,
  `5ce6805353ed760dc42c42ce6bd59472a3b7a7443a486aded175caf755fbff73`.

`data/MANIFEST.json` now records all 1,336 members with exact shape, dtype,
axes, unit/convention, ownership, and member hash. An adversarial first
post-publication audit rejected six unit records and three semantic axes; the
narrow repair added field-level regression assertions for all nine. The
independent re-audit then accepted manifest SHA-256
`9073df69c3a2fbbb93769b8f232933101ac6cff5a6db54982d44f697fd238744`
with no P0, P1, or P2 finding.

The first candidate was rejected because the integration archive stored
physical copies of reader-owned line-reference, edge, and extension axes and
four copies of regime-invariant extension, signed-edge, and trace arrays. That
candidate remains retired. After successful publication the strict gate is
deliberately closed again: the live manifest no longer has the authorized
prepublication identity, so both CLI and direct republication fail without
writes.
