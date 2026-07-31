# Chapter 5 publication-record review

Status: **accepted; exact publication record may authorize one canonical publication**

Reviewed record:
`design/chapter05_publication_acceptance.json`

Record SHA-256:
`80bcfea8e6a817ac1a279b136a247cdd482af1b94af312e94dfa564ad40cafaf`

## Scope

This review accepts the detached authorization record, not a future manifest
edit or an arbitrary candidate directory. The record binds:

- publisher SHA-256
  `6e5ed476cbcd64b1599c9f369bf5ed746d9f561f9bd77c13db8b9ec1c1cdad81`;
- publisher-contract SHA-256
  `8a58866b315b2cdd1ca769436aa0ccdbed3b08e4ef180bd309179d6cc7e04c08`;
- prepublication manifest SHA-256
  `d010889f565f4bbf739f2839fdd2f2e69fc9354f689a92388d940be0e1bf8a78`;
- the independently accepted reader and integration byte identities recorded
  in `design/chapter05_candidate_artifact_acceptance.md`;
- the one canonical Chapter 5 golden destination.

## Independent evidence

The reviewer verified strict duplicate-free JSON, the exact cycle-free schema,
all paths, hashes, byte sizes, kinds, key counts, schema digests, embedded
publisher identity, Payne Zero commit, and reader-to-integration binding.
`publication_gate_ready()` returned true against the canonical record,
accepted candidate, and unchanged prepublication manifest.

Twenty-six independent structural, identity, candidate, and symlink mutations
were exercised through the gate and direct publication API. Every invalid case
failed without creating a destination parent. A well-formed record with a
changed candidate hash or size remains parseable, as intended, but exact
candidate binding rejects the bytes before any write.

The safety boundary also rejects:

- the noncanonical `/tmp` alias in place of its resolved `/private/tmp` path;
- candidate, member, record, or parent symlinks;
- alternate or relative publication destinations;
- unknown, missing, duplicated, malformed, or changed record fields;
- force, replace, repair, merge, or partial-publication behavior.

Lifecycle coverage includes atomic first publication, identical-existing
no-op, race refusal, staging failure, and no-clobber behavior.

The lifecycle-aware publisher suite reported 29 passed and 1 intentionally
skipped test. Its test-file SHA-256 is
`074975d36e8aa57f8173eebfe2e24eec6df52cff0a7fc8619ec4139d32144a43`.

At review time, the Chapter 5 golden directory and manifest entries were
absent. No publication occurred during the review.
