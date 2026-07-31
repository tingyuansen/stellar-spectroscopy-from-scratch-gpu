# Chapter 4 deterministic golden candidate review

Status: **published; manifest-bound parity gates passed**  
Pinned implementation:
`9c44001feae40b85146630499e6f8a5fed42e5af`

This record begins only after the independent oracle-worker and publisher
entry gates passed. It distinguishes a reproducible candidate from a
published, manifest-bound comparison product.

## Double-capture result

The final publisher source used for this capture has SHA-256
`e1c7df9ecdd1222aaa340f527b1ba291ac61855d459f15c470ff562f6e38dd40`.
It ran six isolated scientific routes twice. Every route received a fresh
Numba cache and all twelve frozen pre-import process controls. The publisher
required:

- the six independently accepted in-memory result digests;
- the accepted worker, deterministic-writer, capture-contract, and
  worker-acceptance hashes;
- exact scientific route invariants and no exhausted layer;
- byte equality of all six raw NPZ captures between the two sets;
- byte equality of all five assembled NPZ archives between the two sets.

The successful `--verify-only` result was:

| Candidate archive | Bytes | SHA-256 |
| --- | ---: | --- |
| `chapter04_molecular_constants_cpu_float64.npz` | 235,675 | `cf742ecc5181589e2b7f8b56c7b2d82bd203c9f303de263a5b1c863adaba40a0` |
| `chapter04_atmosphere_molecular_state_cpu_float64.npz` | 1,047,808 | `e0c80f66bf74776d29947c6ced204402c36678edce9b3bdddebd9bd79713ec2a` |
| `chapter04_synthesis_molecular_full_cpu_float64.npz` | 471,636 | `c78393d8525bb637706b5f4dbb17aae7f24660e468bbbfd84dff659ceb8edf31` |
| `chapter04_synthesis_molecular_fixed_cpu_float64.npz` | 649,052 | `c4aeff4b3afba423e3ab9613bd0eef813b3487f7bf792e6b95bcbbf0db46ca12` |
| `chapter04_molecular_public_mapping_cpu_float64.npz` | 590,138 | `a40dcd105641e4620a7c8e9362f0270437e3fe87e7e1e9ed3209b9b566f32628` |

A second invocation materialized the same verified five-file set in
`/private/tmp/chapter04-verified-candidate-e1c7df9e` solely so the
prepublication manifest and unit registry can inspect exact member names,
shapes, and dtypes. That directory is not a textbook data dependency and is
not a golden location.

## Failure that the real gate caught

The first real `--verify-only` attempt failed during candidate assembly. A
cheap synthetic publisher test had incorrectly equated the full route's first
internal molecular-call electron input with the fixture electron seed. The
production call trace proved these are different quantities:

- the fixture seed remains an upstream declared input;
- full molecular call 0 receives the electron state produced inside charge
  closure;
- the published electron density is call 0's returned electron column;
- molecular call 1 receives that returned column;
- call 1's internal returned electron column remains distinct evidence.

The publisher and synthetic ownership test were corrected narrowly. A
targeted independent re-audit passed before the successful double capture.
Fixed-public-electron and public-builder ownership rules were unchanged.

## Publication completion

All publication gates subsequently passed:

1. the five manifest specifications bind the exact names, hashes, sizes,
   builder, fixture, accepted workers, and pinned commit;
2. the fail-closed unit registry routes all 1,171 members;
3. local computation finishes before any comparison archive opens, including
   guarded path, relative-path, byte-path, and file-object forms;
4. independent archive, manifest, and parity audits passed;
5. the final publisher reproduced the exact five identities above and
   atomically installed `data/golden/payne_zero/chapter04/`;
6. `data/MANIFEST.json` was synchronized immediately;
7. the repository manifest/parity gate passed 29 tests, and the complete
   adjacent Chapter 4 gate passed 131 tests.

The parity inventory now partitions all 1,171 members exactly: 568 locally
reconstructed scientific authorities, 49 name-to-authority aliases, and 554
explicit provenance, path-identity, or oracle-only lifecycle members. Chapter
4 may therefore claim pinned CPU-`float64` parity only for the routes and
members covered by that declared contract.
