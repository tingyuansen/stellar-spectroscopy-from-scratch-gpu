# Chapter 6 two-lane publisher-contract rebind candidate

Status: **CANDIDATE ONLY — independent contract-rebind audit required**

Date: 2026-07-30

Authorized write surface:

```text
design/chapter06_lane_artifact_publisher_contract.md
design/chapter06_lane_artifact_publisher_contract_rebind_candidate.md
```

No writer, worker, assembler, publisher implementation, publisher test,
candidate-byte record, authorization, data file, manifest, fixture, golden,
or external Payne Zero source was edited. This is a forward repair of design
locks only and grants no publication authority.

## 1. Candidate disposition

The two-lane contract now binds the independently accepted repaired synthesis
writer chain and the already accepted atmosphere candidate-byte record.

The exact contract transition is:

| contract state | SHA-256 |
| --- | --- |
| historical accepted contract, non-authoritative for the repaired synthesis chain | `9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774` |
| forward-rebound contract candidate | `d52d86512f6a576dc7ba167a4e6f7436368ad8552b13cd6dbd4ce7ed93cfd076` |

The candidate is `51,281` bytes and `895` lines. Its status is explicitly
`NOT AUTHORIZATION`.

The historical contract's independent audit,
`design/chapter06_lane_artifact_publisher_contract_independent_audit.md`,
SHA-256
`7f2517ad0abcca312dcf22785e483fe033519264d5513b16ebe6fa580d4521fd`,
audits only the historical contract bytes. It does not accept this candidate.

## 2. Exact before/after synthesis locks

The left column below is **non-authoritative provenance history only**. None
of its historical synthesis identities remains in the rebound contract.

| lock | historical non-authoritative value | rebound candidate value |
| --- | --- | --- |
| fixture/oracle plan | `d6e6ae1ef55b2fbb81610d29f1e197af794fd5bc3ac57429232ac9efc581b565` | `413d86b6cb671912418d3a19848e3953614d3ac294bf71d6464b9abd8fc11856` |
| phase-1 plan-rebind candidate | unbound | `dd64b85aa204bcac7e936de45e021b5ee2069046e50b47bf531b702959457b93` |
| phase-1 independent audit | unbound | `9441ab9128b6cb204a7c26088f441b9700fecebfc687a29539a3012063f235b7` |
| scientific worker | `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` | unchanged: `36275193118cb7a7c37e91f240e42bb30992d777856154825d6eba5211cc6c68` |
| worker tests | unbound | `1109739338095716dc9f2e2752b6693e74aeedef72b28556eaea4716b00a8189` |
| worker independent audit | `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` | unchanged: `a54689e0a83ff139b2a893effe91cfe90b1ebeda9bbd4730125029c618c84334` |
| raw full fingerprint | `33d1dec19544038367d4540ff38c1fc7d0b610081cfc0c91a2f08a4bc4cbbd7b` | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |
| compact assembler | `62b7aac3580d686183dd1d92e07b01d4710406e29acd944d3b7031889daca65a` | `583734a5843eb671e7ab7c4d598697bd53a569bef537b1806726b6bb360ca7a8` |
| compact-assembler tests | `53111433aa3082a58be5ec8b3da1a961f330eac1bb805ac26d1fc6625487e42d` | `25e371da6fa5c2f86dfd5b2e5847c054103944d3fc5b205036b99d1b875a0153` |
| phase-2 compact candidate | `41daee0a8b4239fd60a2e67b028afa1a0197ac3ae040a30f5dfd8795234b3550` | `54a9f327b7492897679e3e188d46dc4fb11f66727ed5a3e53a542cf382eac42c` |
| phase-2 compact audit | `a0530cd08d5f0ddcc96b51fdaab4520aa89e62ca65850cf855ad4ede22251a33` | `739854db2b5c4c0c0fe5e9db71d8a52958ce401ded7e7a80a8ab90e15172ddcb` |
| compact payload fingerprint | `ce5d1c1d46964eb99c6365ec83ff2e9873521085c629f68759fb8dceac3966f9` | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| deterministic writer | `3da5191d9d86d2df627c417e644b2e72eeed6adfd315258fc8a0a48eb5b6f9cb` | `57aa7147afee4a7366cb2a075715d3607fa20507c23c07ec978b0698368ae47b` |
| writer tests | `9601f8717a29ef51f32a62fd0a73c4d3db4b41c4f6ac08374be6149df4030bfc` | `7c41a74f9d2e38a23d988c990af4040ac262a8066cb3cd9feae4e29f0bdc0a4e` |
| writer candidate | `540cf57126df93ee34d02c3da446a6ef109b93a8b17d60514439e12a8f63fc71` | `6ab1f346a409b0302550a0923c35b71a84d6b2899f2c356070c8d76aa8145e5a` |
| writer independent audit | `b888b49226e8ca6407c8226a3c021efb88fa100623fef27dd62e9beba43f2535` | `467fdc810f14302dba80f0dd18ba34239dfedb7579b48280899f6f9b6e3b3653` |
| canonical candidate bytes | `1,294,865` | unchanged: `1,294,865` |
| canonical candidate SHA-256 | `b92e44a145a284d4d1c3611e32b7882bea7f28799d48e6b3017943ded2511850` | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |

The unchanged raw schema digest, physical fingerprint, compact schema digest,
raw-ownership digest, 754-member ownership partition, 213-member compact
schema, and fixed NPZ encoding remain bound by the repaired chain. The
archive identity changes only because four provenance members changed in the
accepted forward repair; the phase-3 independent audit proves the remaining
209 member payloads are byte-identical.

## 3. Exact edited trust-object paths

The synthesis column of Section 3.1 changed only these two path locks:

| trust object | historical non-authoritative path | rebound path |
| --- | --- | --- |
| writer candidate record | `design/chapter06_synthesis_compact_writer_candidate.md` | `design/chapter06_synthesis_writer_rebind_candidate.md` |
| writer acceptance | `design/chapter06_synthesis_compact_writer_independent_audit.md` | `design/chapter06_synthesis_writer_rebind_independent_audit.md` |

The synthesis candidate-byte acceptance path remains:

```text
design/chapter06_synthesis_candidate_byte_acceptance.md
```

Its current object is a historical `REJECT` at SHA-256
`474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1`.
It is not an acceptance of the repaired archive and remains unresolved.

No publisher, publisher-test, publisher-audit, authorization,
authorization-review, or postpublication path changed.

## 4. Atmosphere candidate-byte lock

The contract newly binds the already accepted read-only atmosphere record:

| object | exact SHA-256 |
| --- | --- |
| `design/chapter06_atmosphere_fixture_byte_acceptance.md` | `79b20c065f5f1b588c6796d6413a1b2aa5e1e5a60056968fc7913cd961e12fc6` |

That record freezes the unchanged nineteen-member, 363,050-byte archive at
SHA-256
`1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff`.
It remains read-only evidence, not a publisher acceptance or publication
authorization.

The record reviewed the historical contract at SHA-256 `9ee0029f…`. The
forward-rebound contract treats that immutable record as a previously
accepted upstream input and changes no atmosphere scientific, writer, cache,
schema, role, path, or byte rule. Because the record's own residual-scope
language conservatively names contract changes, the independent auditor of
this candidate must explicitly decide whether this forward carry is valid.
This candidate cannot accept itself.

Subject to that independent contract-rebind decision, only the synthesis
candidate-byte acceptance remains unresolved before the two publisher
implementations may be authored and separately audited.

## 5. Status and sequencing edits

The contract now says, consistently:

- both zero-argument deterministic writers are independently accepted;
- the atmosphere candidate-byte record is independently accepted;
- the synthesis candidate-byte gate remains unresolved;
- this contract rebind itself requires independent acceptance;
- publisher implementations may begin only after that contract acceptance
  and the synthesis candidate-byte acceptance;
- the fixed publication order remains atmosphere fixture first
  (\(M_0 \rightarrow M_1\)), synthesis golden second
  (\(M_1 \rightarrow M_2\)); and
- no accepted writer or candidate-byte record grants a canonical role,
  detached authorization, data write, or manifest mutation.

The acceptance checklist was split accordingly: the atmosphere byte gate is
checked, while the synthesis byte gate remains unchecked.

## 6. Preserved authority graph and publication mechanics

No dependency edge changed. The contract still freezes:

```text
accepted writers/sources ──> C ─┐
accepted publisher/contract ─> P ├─> A
M ───────────────────────────────┤
T ───────────────────────────────┘

A ──> R
(T, SHA256(A), SHA256(R)) ──> E
(M, E) ──> N
(A, R, E, N, artifact bytes) ──> Z
```

The two late-bound placeholders still occur exactly once each:

```text
__LATE_BOUND_AUTHORIZATION_SHA256__
__LATE_BOUND_RECORD_REVIEW_SHA256__
```

The repair did not change:

- strict duplicate-free authorization and review schemas;
- exact repository-relative identity versus absolute host-access typing;
- fixed atmosphere-first phase ordering;
- append-only manifest entry construction;
- current unsorted manifest order and exact JSON encoding;
- delete-last reconstruction of prepublication manifest bytes;
- literal late-bound placeholder substitution;
- exact no-follow path walking and retained device/inode checks;
- nested-directory `fsync` ordering;
- same-filesystem atomic no-replace installation;
- stable `data` directory locking;
- short-write, readback, and fresh-process validation;
- lane-specific cache rules;
- exact-existing validation-only no-op;
- partial-publication recovery states;
- inode-bound quarantine and separately reviewed cleanup;
- TOCTOU controls;
- adversarial matrix; or
- prohibition on publisher, data, manifest, fixture, and golden writes.

## 7. Exact consistency checks

Every newly bound synthesis file and the atmosphere byte record was hashed
from the live regular file. All matched the candidate contract:

| object group | checked files | result |
| --- | ---: | --- |
| repaired plan plus phase-1 records | `3` | exact |
| unchanged worker/tests/audit | `3` | exact |
| repaired assembler/tests plus phase-2 records | `4` | exact |
| repaired writer/tests plus phase-3 records | `4` | exact |
| atmosphere candidate-byte record | `1` | exact |
| total | `15` | exact |

Literal-count checks found each of the 17 exact synthesis values requested by
the repair once in the contract: plan, both phase-1 records, worker, worker
tests, worker audit, raw full fingerprint, assembler, assembler tests, both
phase-2 records, compact payload fingerprint, writer, writer tests, both
phase-3 records, and final archive SHA-256. The atmosphere candidate-byte
record hash occurs twice by design: once in its exact lock table and once in
the Section 3.1 gate-state prose.

Path checks found:

- the two repaired synthesis writer candidate/audit paths in both their
  exact-lock and trust-object rows;
- the exact atmosphere candidate-byte path;
- the still-unresolved synthesis candidate-byte path;
- all five `A -> R -> E -> N -> Z` forward edges; and
- both exact late-bound placeholders.

The live contract retained:

```text
current manifest SHA-256
  d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a

publisher authorization
  absent

canonical Chapter 6 atmosphere fixture
  absent

canonical Chapter 6 synthesis golden
  absent
```

## 8. Stale-literal scan

An exact scan of the rebound contract found zero occurrences of:

- historical plan `d6e6ae1…`;
- historical raw full fingerprint `33d1dec1…`;
- historical assembler/tests `62b7aac3…` / `53111433…`;
- historical compact candidate/audit `41daee0a…` / `a0530cd0…`;
- historical compact payload `ce5d1c1d…`;
- historical writer/tests `3da5191d…` / `9601f871…`;
- historical writer candidate/audit `540cf571…` / `b888b492…`;
- historical archive `b92e44a1…`; or
- historical synthesis writer-candidate/audit path literals.

Those values appear only in Section 2 and Section 3 of this candidate report,
where they are explicitly labelled non-authoritative provenance history.

## 9. Downstream invalidation

This candidate invalidates the previous contract's audit as authority for the
current contract bytes. A new independent read-only audit must accept or
reject SHA-256 `d52d8651…`.

The existing synthesis candidate-byte record remains a historical rejection
and cannot be refreshed in place. A different independent reviewer must
construct and review the exact repaired synthesis bytes and replace that
gate with a new acceptance at the same planned trust-object path only under
separate authorization.

No publisher implementations, publisher tests, publisher acceptances,
detached authorizations, authorization-record reviews, canonical artifacts,
manifest entries, or postpublication audits presently exist for either lane.
If any downstream draft elsewhere binds the historical contract hash or any
historical synthesis lock, it is stale and must fail closed; this candidate
does not silently refresh it.

The accepted atmosphere candidate-byte record was not edited. Its proposed
forward binding is specifically part of the independent decision requested
for this contract candidate, as described in Section 4.

## 10. Verification

Exact hash and identity checks:

```text
shasum -a 256 <15 accepted bound files>

all 15 matched their contract locks
```

Stale-literal scan:

```text
rg -n <historical synthesis hashes and old writer paths> \
  design/chapter06_lane_artifact_publisher_contract.md

no matches
```

Lock-count and graph/path checks:

```text
17 requested synthesis values: exactly one occurrence each
atmosphere candidate-byte hash: exactly two occurrences by design
repaired synthesis candidate/audit paths: present
atmosphere and synthesis candidate-byte paths: present
five forward authority edges: present
two late-bound placeholders: present exactly once each
```

Final whitespace checks:

```text
git diff --check -- \
  design/chapter06_lane_artifact_publisher_contract.md \
  design/chapter06_lane_artifact_publisher_contract_rebind_candidate.md

clean
```

## 11. Remaining gate

This report does not audit or accept its own contract change. The next action
is an independent read-only audit of:

- the exact rebound contract and this report;
- all live file/hash locks;
- the complete historical-to-live lock ledger;
- the atmosphere byte-record carry-forward;
- the unresolved synthesis byte gate;
- the unchanged `A -> R -> E -> N -> Z` graph;
- manifest encoding, ordering, durability, path, cache, recovery, and
  quarantine rules;
- stale-literal absence; and
- no publisher/data/manifest mutation.

Final disposition: **CANDIDATE ONLY — NOT AUTHORIZATION**.

---

## 12. Forward-cycle repair after independent rejection

Date: 2026-07-30

Status: **SUPERSEDING REPAIR CANDIDATE ONLY — independent audit required;
NOT AUTHORIZATION**

Sections 1–11 above are retained as immutable history of the rejected first
forward-rebind candidate. They are not the current proposed gate state. This
section records the narrow repair required by the independent rejection and
supersedes only the stale atmosphere carry-forward and its downstream status,
sequence, graph, checklist, and disposition assertions.

No synthesis plan, worker, assembler, writer, test, candidate, or audit was
edited. No atmosphere scientific source, worker, writer, candidate-byte
record, publisher, data file, manifest, fixture, golden, authorization,
review, or external tree was edited.

### 12.1 Exact rejected snapshot

The repair started from these exact immutable inputs:

| rejected-state object | SHA-256 |
| --- | --- |
| rejected forward-rebound contract | `d52d86512f6a576dc7ba167a4e6f7436368ad8552b13cd6dbd4ce7ed93cfd076` |
| this candidate report before the present append | `516d358a5a993f369fa8113bfcafea5fb3e7156287ba80f1e85bba783e3e2513` |
| independent rejection | `33493d0abf0a61ed6a7926f4a49ea6b932960b290a3ddbfa9384f7e7a6f7450d` |

The rejection accepted the mechanical synthesis rebind and publication
mechanics but found one blocking trust defect: the atmosphere byte record
`79b20c…` bound the old contract `9ee002…` and explicitly invalidated itself
when that contract changed. Carrying it as accepted into `d52d8651…` was
therefore stale. Rerunning the byte review while retaining its future hash
inside the contract would instead create a `contract <-> C` recursion.

### 12.2 Repaired contract candidate

The newly repaired contract candidate is:

```text
design/chapter06_lane_artifact_publisher_contract.md
  bytes
    54,997
  lines
    955
  SHA-256
    a663369c3851d89468a41436b8faddeba9d3dcbeba79a7254037734f4a5b3666
```

This is a candidate identity only. This report does not audit or accept it.

The repair changed only the following assertions:

1. The status now says both deterministic writers are independently accepted
   and both candidate-byte acceptances remain **UNRESOLVED**.
2. The atmosphere record at
   `design/chapter06_atmosphere_fixture_byte_acceptance.md`, SHA-256
   `79b20c065f5f1b588c6796d6413a1b2aa5e1e5a60056968fc7913cd961e12fc6`,
   is retained only as stale historical evidence that bound contract
   `9ee0029f228d31fac67cf3c669accf2b15416d4439305ef9a8e94d7c5bfec774`.
   It is not an accepted upstream lock.
3. The synthesis record at
   `design/chapter06_synthesis_candidate_byte_acceptance.md`, SHA-256
   `474e31821977a7e5063cbd99419f41581d7f68ac435574130866f914e042f7f1`,
   is retained only as the historical rejection of the old writer against the
   repaired plan. It accepts no repaired synthesis bytes.
4. Both existing candidate-byte trust-object paths remain fixed, but neither
   future acceptance hash is embedded in the contract.
5. The final rebound contract must be independently audited first.
6. Only afterward may each candidate-byte gate append a new independent
   re-audit at its existing path, preserving the old record and binding the
   exact independently accepted final contract.
7. Publisher implementation may begin only after **both** appended byte
   re-audits are independently final.
8. Later publisher implementation acceptances, detached authorizations, and
   authorization-record reviews bind both the final contract SHA-256 and the
   applicable final candidate-byte acceptance SHA-256.
9. The acceptance checklist now leaves the final-contract audit and both byte
   gates unchecked.
10. The final disposition no longer permits either lane to progress to
    publisher work on the strength of an old byte record.

Every repaired synthesis phase-1/phase-2/phase-3 file and hash, raw/compact
identity, archive identity, writer-level atmosphere identity, trust-object
path, role rule, and publication mechanic remains byte-for-byte as in the
rejected contract candidate except for context shifted by the assertions
above.

### 12.3 Cycle-free forward order

The repaired order is exact:

```text
final rebound contract
  -> independent contract audit
  -> new atmosphere C re-audit against that final contract
  -> new synthesis C re-audit against that final contract
  -> publisher implementations and independent publisher reviews
  -> detached authorization and authorization-record review
  -> realized manifest entry and postpublication manifest
  -> postpublication audit
```

The two byte re-audits may be conducted as separate gates after the contract
audit, but **both must close before either publisher implementation begins**.

Let `D` be the independently accepted final contract and `C` one lane's later
candidate-byte acceptance. The repaired contract contains the path and review
rules for `C`, but not `SHA256(C)`. The dependency is therefore:

```text
D -> C
(D, C, publisher implementation) -> P
(D, C, P, M, T) -> A
A -> R
(T, SHA256(A), SHA256(R)) -> E
(M, E) -> N
(A, R, E, N, artifact bytes) -> Z
```

Only the later `P`, `A`, and `R` objects bind both final `D` and final `C`
hashes. There is no `C -> D` edge and no `D <-> C` cycle. The established
`A -> R -> E -> N -> Z` late-binding repair remains unchanged.

### 12.4 Preserved exact live identities

The repaired contract retains the complete independently accepted synthesis
chain and these final in-memory identities:

| live accepted writer-level boundary | exact value |
| --- | --- |
| raw members | `754` |
| raw schema digest | `d9d2d2424e9503fae7cb796db995d5a2ed1c73a31e85a707a10e173b2ef95178` |
| raw physical fingerprint | `51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc` |
| raw full fingerprint | `8e75e3582a3624110086e4a28c630f8c01ae7efee3bb78d2fd500b3f4926b893` |
| compact members | `213` |
| compact array bytes | `1,235,275` |
| compact schema digest | `911d06d931b8210154761badfc923962793dfdb77905b1c02051100633190bde` |
| compact payload fingerprint | `e4eeb8b02fcbcf283ade84e39a492e92756f4c2c0be62951f9a7b697c419368b` |
| raw-ownership digest | `5594db24db26f726606078449a6e5e48f5b9d8d1be8732f5e676c43fd7382675` |
| synthesis archive bytes | `1,294,865` |
| synthesis archive SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |
| atmosphere scientific members | `19` |
| atmosphere scientific array bytes | `357,984` |
| atmosphere archive bytes | `363,050` |
| atmosphere archive SHA-256 | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` |

The live prepublication manifest remains:

```text
data/MANIFEST.json
  bytes
    1,087,741
  entries
    37
  SHA-256
    d8f30e25a415bd7b84fc486ffa878d56cbfb1af79ca3992bb8d3ec4976630a7a
```

Both canonical Chapter 6 lane artifacts, both publishers, both publisher
tests, both publisher acceptances, both detached authorizations, both record
reviews, and both postpublication audits remain absent.

### 12.5 Preservation and regression checks

The repaired contract retains:

- all seventeen repaired synthesis lock values exactly once each;
- all twenty path/hash table rows at their current exact files: eighteen
  authoritative writer-chain rows and two explicitly historical byte-record
  rows;
- all eleven trust-object paths per lane;
- the atmosphere `fixture` versus synthesis `golden` role asymmetry;
- both exact canonical destination paths;
- both exact late-bound placeholders once each;
- the append-only unsorted-manifest byte encoder and delete-last proof;
- nested-directory and parent durability;
- repository-relative identity versus absolute no-follow host access;
- lane-specific atmosphere-populated and synthesis-empty cache rules;
- closed source/data snapshots;
- exact-existing and partial-publication recovery;
- inode-bound quarantine and separately reviewed cleanup;
- atomic no-replace installation, TOCTOU checks, and fixed phase order; and
- the complete no-publication boundary.

Exact stale/recursion checks on the repaired contract require:

```text
full historical atmosphere byte-record SHA:
  one occurrence, only in explicitly stale history

full historical synthesis rejection SHA:
  one occurrence, only in explicitly rejected history

future authoritative atmosphere candidate-byte acceptance SHA:
  unresolved and absent from the contract

future authoritative synthesis candidate-byte acceptance SHA:
  unresolved and absent from the contract

both candidate-byte status assertions:
  UNRESOLVED

publisher implementation before both C gates:
  forbidden

D <-> C edge:
  absent

A -> R, A -> E, R -> E, E -> N, N -> Z:
  present

__LATE_BOUND_AUTHORIZATION_SHA256__:
  exactly once

__LATE_BOUND_RECORD_REVIEW_SHA256__:
  exactly once
```

The historical synthesis hashes and old synthesis writer-candidate/audit
paths listed in Sections 2 and 8 remain absent from the repaired contract.
They remain non-authoritative history in this report.

### 12.6 Remaining gate

This appended repair record does not accept its own contract candidate.
The next permitted action is a different agent's independent, read-only
audit of:

- exact repaired contract and report identities;
- all preserved phase-1/phase-2/phase-3 and atmosphere locks;
- both explicitly unresolved candidate-byte gates;
- stale `79b20c…` and rejected `474e318…` history;
- the `D -> C -> P/A` and `A -> R -> E -> N -> Z` graphs;
- trust paths, roles, manifest bytes, durability, path identity, cache,
  snapshot, recovery, and quarantine rules; and
- absence of publisher, authorization, data, and manifest mutation.

Final disposition: **REPAIR CANDIDATE ONLY — NOT INDEPENDENTLY AUDITED; NOT
AUTHORIZATION**.

## 13. Physical-fingerprint transcription repair after synthesis byte rejection

Status: **NARROW REPAIR CANDIDATE ONLY — NOT AUDITED OR ACCEPTED**

Sections 1–12 are preserved as the complete history of the forward-cycle
repair. Immediately before this append, the exact inputs were:

| preserved input | SHA-256 |
| --- | --- |
| previous publisher contract | `a663369c3851d89468a41436b8faddeba9d3dcbeba79a7254037734f4a5b3666` |
| previous rebind candidate report | `2fc24a7161916bfbc709e261d74c34be3bc754d15d967377f6c620fac7d478d4` |
| previous independent contract audit | `c4a4ca58d94ec71ec509238046afcb127e189ba0be98a96c9929488958a1c286` |
| atmosphere candidate-byte acceptance | `cd9be49b2436fc34a33275c54369c549ad6be40aeba2f713bdd033f028156669` |
| synthesis candidate-byte rejection | `df3714892cf60bb54d22743dcc444157246b8392a3ea907c039af1c196528c55` |

The synthesis byte gate found one exact contract contradiction. Section 2.1 of
the previous contract contained a 65-character transcription with one extra
`e` after `51371e5c0db1fa`. The repair is exactly:

```text
wrong:
51371e5c0db1fae7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc

right:
51371e5c0db1fa7eaefc1d4ef40c9c90393aac86239cbdb5b6e6c7f4b279b1fc
```

The corrected 64-character value is independently present in the live compact
assembler constant, the writer-imported raw summary field, the synthesis
phase-3 rebind, the writer independent audit, the previous contract audit, and
all four fresh raw observations recorded by the synthesis byte gate. This
repair transcribes that already accepted physical identity; it does not refresh
or reinterpret any scientific bytes.

The repaired contract candidate is:

```text
design/chapter06_lane_artifact_publisher_contract.md
  bytes
    54,996
  lines
    955
  SHA-256
    3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b
```

An exact semantic comparison with contract `a663369c…` is equal to replacing
the wrong literal above once with the right literal above once. No second
contract byte changed. All other accepted synthesis and atmosphere paths,
hashes, member counts, schemas, payload fingerprints, archive identities,
roles, graph edges, placeholder rules, manifest mechanics, durability,
recovery, quarantine, and no-publication boundaries remain byte-semantically
unchanged.

The trust consequences are fail-closed:

1. The prior contract audit `c4a4ca58…` is now historical because it binds
   previous contract `a663369c…`; it cannot accept `3a064f82…`.
2. The atmosphere acceptance `cd9be49b…` is now stale because it also binds
   previous contract `a663369c…`, even though its candidate bytes are not
   rejected by this transcription repair.
3. The synthesis record `df371489…` remains the authoritative rejection of
   the previous contract and grants no candidate-byte acceptance.
4. Both authoritative candidate-byte gates are therefore **UNRESOLVED**.
5. A different agent must first audit the exact repaired contract and this
   appended report.
6. Only after that new contract audit may the atmosphere and synthesis byte
   gates each rerun at their existing trust-object paths against the exact new
   contract and audit.
7. Both rerun byte gates must accept before either publisher implementation
   may begin.

No publisher, publisher test, publisher review, detached authorization,
authorization-record review, canonical artifact, data file, manifest entry,
cleanup record, or postpublication audit was created or modified by this
repair.

Final disposition: **ONE-FIELD CONTRACT REPAIR CANDIDATE — NEW CONTRACT AUDIT
AND BOTH CANDIDATE-BYTE RERUNS REQUIRED; NOT AUTHORIZATION**.
