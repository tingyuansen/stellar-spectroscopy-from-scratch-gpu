# Chapter 5 candidate-artifact acceptance

Status: **repaired candidate accepted; publication not authorized by this file**

This record accepts the scientific contents and physical ownership layout of
one reproducible Chapter 5 reader/integration pair. It is documentation, not
the machine-readable publication gate.

## Accepted identities

| Artifact | Bytes | Members | Kind | Schema digest | SHA-256 |
|---|---:|---:|---|---|---|
| reader | 1,022,580 | 257 | `continuum_reader` | `058389fcd0e944dd4c1ad1208adbaac44d53ab35e61e5366d37a8a141ad91f88` | `84e3e764b2d9be8d409b0db0271decdd90e07b8e716f5c589fbdbb1b9e2ae39f` |
| integration | 21,608,505 | 1,079 | `continuum_integration` | `e09a8932d97f8a2756aca2b779c4b8fdd822ca239197fd908ee575d09df081ca` | `5ce6805353ed760dc42c42ce6bd59472a3b7a7443a486aded175caf755fbff73` |

Both archives have schema version 1, are CPU-only `float64` comparison
products, and bind publisher SHA-256
`6e5ed476cbcd64b1599c9f369bf5ed746d9f561f9bd77c13db8b9ec1c1cdad81`.

## Independent acceptance evidence

The reviewer treated both NPZs as untrusted ZIP bytes before loading them.
The pair has the exact two-file set, lexical `ZIP_STORED` members, frozen ZIP
timestamps and permissions, no object arrays or path-like members, valid CRCs,
and no ZIP comments, extras, duplicate names, or trailing bytes. Rewriting
every member with the frozen deterministic writer reproduced both archives
byte for byte.

The integration archive binds the exact reader hash and reader schema. The
pair reconstructs all 1,161 logical raw members with exact shape, dtype, and
bytes and reproduces:

- raw schema digest
  `652c110dc79a6f6dfca6893bee35416289675b4920a5d0dcfe6b2cb262dacf3d`;
- physical fingerprint
  `d223351fa2c51dc24a1b01896da9ab9a82fc475f4082c47fde34734d8dc03343`;
- full fingerprint
  `3d2c131711e1c0dc6aa088892193bb24d41a76d005bc20dd1c42d3e84f66e656`.

All floating and complex logical arrays are finite. Main absorption and
scattering products are nonnegative and active in every full-product layer.
Atmosphere, synthesis, isolated-minor, source-numerator, and interpolation
closures pass. The reviewer also checked the exact threshold triplets, five
sampling grids, active line counts, sentinel and padding conventions, IFOP
routes, CH/OH/CIA seams, 19,900 K H2 table clamp, strict 20,000 K H2 owner
cutoff, traces, activation, counterfactuals, and component order.

Fresh Chapter 5 calculations reproduce the candidate bit for bit for compact
atmosphere output, line-reference thresholds, standard/diagnostic/extension
synthesis, and all four 30,000-point atmosphere products. The focused
acceptance suite reported 65 passed and 1 intentionally skipped test.

## Physical ownership

The 1,161 logical routes have exactly one disposition:

| Disposition | Count |
|---|---:|
| reader | 650 |
| reader alias | 71 |
| common identity metadata | 100 |
| integration | 326 |
| coalesced grid bank | 14 |

All 721 reader-owned logical routes appear in integration only as scalar
member-name aliases. The 326 integration routes resolve to 227 physical
descriptors; 33 regime-invariant fields each coalesce four raw routes after
bitwise equality checks. All 144 declared invariant raw routes were verified
before coalescing.

Remaining equal-valued arrays have distinct scientific meanings—for example,
separate zero-gate witnesses, expected-versus-observed checks, activation
records, and counterfactual equivalence tests. They are not duplicate owners.

## Publication state at acceptance

- `design/chapter05_publication_acceptance.json`: absent;
- published Chapter 5 golden directory: absent;
- Chapter 5 golden manifest entries: absent;
- prepublication manifest SHA-256:
  `d010889f565f4bbf739f2839fdd2f2e69fc9354f689a92388d940be0e1bf8a78`.

No file was published during this review. Publication requires a separately
reviewed canonical detached JSON record and a fresh publisher-side double
capture.
