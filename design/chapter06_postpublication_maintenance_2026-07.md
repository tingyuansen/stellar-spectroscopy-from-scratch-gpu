# Chapter 6 postpublication maintenance record

Date: 2026-07-30

## Scope

This record documents a maintenance-only change after both Chapter 6 lane
artifacts were already published. It does not replace or amend either
publication authorization, record review, original postpublication audit,
manifest entry, or canonical artifact.

The historical records remain evidence for the exact code and repository state
that performed the one-time transitions. The live maintenance implementation
now verifies the resulting authority graph and artifacts without requiring the
whole textbook to remain in its prepublication state.

## Obsolete assumptions removed

1. The staged `payne_zero_atmosphere/__init__.py` is now byte-identical to the
   pinned Payne Zero initializer at
   `dbb7734cab4f3e98b9b88d1f1b5ec27afd02fd7fd003ba7931b43d3750049d61`.
   The converter gate records that exact identity while retaining the same
   three scientific converter dependencies and the same nineteen fixture
   arrays.
2. Later chapters may add manifest entries, chapter-scoped artifact records,
   and regular data files. The synthesis inventory still walks the entire data
   tree without following links, joins every legacy-manifest entry to exact
   bytes, rejects special nodes and unjustified empty directories, and includes
   later support files in its aggregate snapshot. It no longer misclassifies
   later chapter files as a Chapter 6 publication failure.
3. Canonical verification no longer expects authorization or the destination
   to be absent. Both publishers expose `--verify-published`, which validates
   the strict authorization and review JSON, realizes the two late-bound
   hashes, finds exactly one matching manifest entry, decodes the exact
   canonical archive, verifies scientific schema and payload identities, and
   performs no candidate reconstruction or mutation.
4. The detailed atmosphere-oracle test executes in a fresh interpreter with
   its accepted pre-Numba controls. Its outcome is independent of which other
   test initialized Numba or imported the staged atmosphere package first.

## Preserved scientific identities

The canonical scientific artifacts are unchanged:

- atmosphere fixture: 363,050 bytes,
  `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff`;
- synthesis comparison golden: 1,294,865 bytes,
  `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955`.

The atmosphere fixture still has nineteen arrays with schema digest
`f63ed611f144e6ca7aa2ad2663e15eb758fc540851fa4fc78b6ef819d7372698`
and payload fingerprint
`f30bc9f15ed483aa864a92399fb13b53f655e8add55397c6fec6465377644663`.
The synthesis golden still has 213 members with compact schema digest
`911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde`
and payload fingerprint
`e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b`.

No canonical data file, manifest byte, external Payne Zero source, or paper
source was modified by this maintenance.

## Verification commands

```text
python scripts/build_chapter06_atmosphere_fixture.py --verify-published
python scripts/build_chapter06_synthesis_golden.py --verify-published
python -m pytest -q tests/test_chapter06_*.py
```

Acceptance criterion: both verification commands report
`state = exact_registered` and `complete_validation = true`; the complete
Chapter 6 test set passes with no setup errors.
