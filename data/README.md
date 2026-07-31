# Textbook data roles

Every data product used by the executable book has one declared role:

- `static/`: immutable physical tables, schemas, and checkpoints;
- `subsets/`: checksum-bound teaching slices of larger source catalogs;
- `fixtures/`: explicit upstream computed state used to isolate a lesson;
- `golden/payne_zero/`: comparison-only outputs from the pinned implementation.

`MANIFEST.json` records identity, role, byte and per-array checksums, shapes,
dtypes, units, source provenance, builders, and scientific scope. A fixture is
never described as a quantity computed in the chapter that loads it. A golden
is never read until after the reader's calculation has finished.

The Chapter 3 synthesis EOS source bundle is intentionally split: invariant
partition/Saha arrays live in `static/synthesis_tables/`, while its extracted
80-depth thermodynamic state and depth-specific `ground_partition_table` live
in `fixtures/`. The exact constructor can merge those manifest-bound mappings
in memory without mislabeling computed fixture data as a static table.

`fixtures/chapter03_atom_only_inputs.npz` is the bounded six-depth integration
fixture. Its thermodynamic rows and 99-element composition are selected from
the extracted pinned state; its column mass, microturbulence, and four-point
one-element branch inputs are declared controls. Rebuild it with:

```text
python scripts/build_chapter03_atom_only_fixture.py
```

The Chapter 3 oracle builder reads the pinned checkout at commit
`9c44001feae40b85146630499e6f8a5fed42e5af` without modifying it. It writes,
in causal comparison order:

1. atmosphere scalar/batch Saha modes;
2. refreshed atom-only atmosphere state and atomic energy;
3. packed sentinel and physical bridge outputs;
4. CPU Torch float64 synthesis full/fixed states.

The fixed synthesis state receives the full closure's electron density
unchanged. Rebuild all four comparison-only archives with:

```text
python scripts/build_chapter03_payne_zero_goldens.py
```

Reader and test code computes the local result before opening any golden.

## Chapter 4 molecular comparison archives

Chapter 4 uses five CPU-float64 archives under
`golden/payne_zero/chapter04/`. Each scientific result has one owner:

1. `chapter04_molecular_constants_cpu_float64.npz` owns both lossless
   molecular catalogs, their semantic alignment, exact branch-boundary
   witnesses, H2 probes, and shared provenance.
2. `chapter04_atmosphere_molecular_state_cpu_float64.npz` owns the atmosphere
   molecular routes: complete Newton histories, physical-to-normalized
   handoffs, warm starts, energy mode, disabled behavior, schedule inventory,
   and the synthesis bridge.
3. `chapter04_synthesis_molecular_full_cpu_float64.npz` owns the ordered
   two-call full synthesis EOS route and its published state.
4. `chapter04_synthesis_molecular_fixed_cpu_float64.npz` owns both fixed-EOS
   branches, keeping derived-mass and supplied-mass calls distinct while
   storing their common inputs only once.
5. `chapter04_molecular_public_mapping_cpu_float64.npz` owns the public
   structured-atmosphere mapping, all 54 molecular line lanes, the no-ground
   partition policy, the independent CO reconstruction, edge loading, and
   mutation trace.

The constants archive has SHA-256
`cf742ecc5181589e2b7f8b56c7b2d82bd203c9f303de263a5b1c863adaba40a0`.
Each of the other four archives stores that identity in
`meta__constants_archive_sha256`; it does not duplicate constants-owned
catalog or alignment arrays.

Publication is deliberately separate from capture. The publisher starts
twelve isolated child processes: six routes in each of two fresh capture
sets, with a distinct empty Numba cache for every child and frozen process
controls in place before scientific imports. It assembles the five archives
twice and requires both the raw route archives and final archives to be
byte-identical. A prepublication audit may expose the verified final set in a
temporary directory. Point both comparison gates at that read-only candidate
without copying it into `data/`:

```text
CHAPTER04_CANDIDATE_DIR=/private/tmp/chapter04-verified-candidate-... \
python -m pytest -q tests/test_chapter04_manifest_prepublication.py

CHAPTER04_CANDIDATE_DIR=/private/tmp/chapter04-verified-candidate-... \
python -m pytest -q tests/test_chapter04_golden_parity.py
```

The manifest gate checks all five archive hashes and sizes, embedded commit,
fixture, publisher, acceptance-report, capture-contract, and constants
identities, and the physical or interface representation of every NPZ member.
The parity gate computes the local atmosphere, molecular EOS, and public
mapping first and only then opens the candidate. Both gates require
independent acceptance before publication.

After that acceptance, run the publisher one final time. It must reproduce
the reviewed identities while atomically installing the complete five-file
directory:

```text
python scripts/build_chapter04_payne_zero_goldens.py
```

Immediately synchronize the manifest and run the repository data and parity
tests:

```text
python scripts/sync_data_manifest.py
python -m pytest -q tests/test_chapter04_manifest_prepublication.py \
  tests/test_data_manifest.py tests/test_chapter04_golden_parity.py
```

While the repository archives are absent, the static specification tests
still run and the candidate-backed tests require
`CHAPTER04_CANDIDATE_DIR`. Once publication has completed, the same tests
default to the repository directory.

Parity code follows a strict causal rule: compute and retain every local
atmosphere, molecular EOS, and public-mapping result first; only after those
calculations finish may it open a comparison-only golden. A golden must never
seed, repair, reshape, or otherwise influence the calculation it is used to
check.
