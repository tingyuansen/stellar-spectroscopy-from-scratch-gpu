# Chapter 6 publisher threat-model adjudication

Date: 2026-07-30

Review mode: read-only design adjudication

Disposition: **REJECT the late syscall-entry substitution finding as an
acceptance-blocking interpretation of the frozen contract**

Authority: **NONE**

## 1. Decision

The isolated probes are technically correct: both reviewed publishers use a
staging *name* as the source of the final `link` or `replace`, and a wrapper
that runs arbitrary code literally at entry to `os.link` or `os.replace` can
substitute that name before invoking the real syscall. The resulting
mutate-before-reject traces are useful evidence of the limit of a
name-resolved syscall.

They are not, however, an in-scope cooperative-publisher or ordinary
filesystem-substitution race under the frozen contract. The wrapper has
arbitrary same-process execution at the authoritative syscall boundary. Read
as a simulation of an external actor, it represents a process with the same
account's unrestricted ability to ignore the retained `data` lock, discover
an invocation-private random staging name, unlink it, recreate it, and win the
single-instruction interval after the last validation. Section 12 explicitly
excludes a hostile process with unrestricted access to the same account from
the scientific reproducibility claim.

The exact scope judgment is therefore:

| question | judgment |
| --- | --- |
| Is the trace a real limitation of path-based `link`/`rename`? | **ACCEPT** |
| Is arbitrary code injected inside the publisher at syscall entry an in-scope threat actor? | **REJECT** |
| Is an external same-account process that ignores the shared lock and swaps the private stage in that exact interval in scope? | **REJECT** under Section 12 |
| Does the trace by itself reject an otherwise contract-conforming publisher? | **REJECT** |
| Does this adjudication accept either publisher or authorize publication? | **NO** |

The smallest honest Darwin path is to retain the frozen threat boundary and
accept immediate pre-syscall identity/byte checks, retained file and directory
descriptors, the common `data` lock, an atomic namespace syscall, retained
directory `fsync`, canonical-path rebind, and complete post-check as the
required publication mechanism. The documentation and the next independent
audit should say plainly that this prevents cooperative publisher races and
the contract's enumerated substitutions; it is not a security boundary
against arbitrary code executing as the repository owner.

Requiring a descriptor-bound primitive for *both* mutations is not a viable
Darwin acceptance rule. Darwin provides a useful descriptor-bound
create-if-absent primitive for the artifact, `fclonefileat`, but no documented
public primitive that atomically replaces an existing named manifest from an
already-open source descriptor. Imposing that unavailable primitive without
changing the promised platform support would make publication permanently
impossible.

## 2. Frozen evidence reviewed

All requested objects were read in full. Their exact identities at
adjudication time were:

| object | SHA-256 |
| --- | --- |
| `design/chapter06_lane_artifact_publisher_contract.md` | `3a064f8291a9b7436ed4bc585b36f3459c3b46b4c9c48bed0d0852c83cf3d65b` |
| complete atmosphere publisher audit history | `df58318960008ebdab127d9d54c3a88b2b295307a2a2f6c7c07496510ede14d6` |
| complete synthesis publisher audit history | `7790c4b7366a05456d3c5a947e6ad3b3890c16c07e7681e5acc84e86c409a15c` |

The exact repaired publisher bytes still match the snapshots named by those
histories:

| lane object | SHA-256 |
| --- | --- |
| atmosphere publisher | `d9cb38e5eacb3dbad66560dad63f13f77ff16f5aa58030b5c53f75a702913dfa` |
| atmosphere focused tests | `0653f399a3a1c7b5d1600a5bdbe9c9997e6210f0148f6833a973cffdc94a987a` |
| atmosphere repaired candidate report | `c55b8fd3653e6d89ecea5f1bcbb96b6b0a010452cf123216a02a6320f07de820` |
| synthesis publisher | `2a345fd389d2c8f8a97ebef8b4d418a1c81f5b49b94dd94e595ac95cb2e0479e` |
| synthesis focused tests | `a314d85569201c659c91308c9121e1f1762f4af72ee87f2ba84f93a8e198b7a1` |
| synthesis repaired candidate report | `89d38d06ff3868289207f9626f068038b1f523cb2c8b456a4e11ed340957d378` |

The accepted candidate-byte identities remain independent of this design
question:

| lane | bytes | members | archive SHA-256 |
| --- | ---: | ---: | --- |
| atmosphere fixture | `363,050` | `19` | `1b72767192672ac102a1b8e4b89b4d1d316a3da747d5197a9b99eaff38c86bff` |
| synthesis comparison golden | `1,294,865` | `213` | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |

No candidate, contract, audit history, plan, data file, manifest,
authorization, review, external source, or publisher implementation was
changed by this adjudication.

## 3. What the contract actually requires

Sections 9 and 10 specify a scientific publication transaction, not a
same-user adversarial security boundary:

1. The common exclusive lock is held on the stable canonical `data`
   directory for the complete file-plus-manifest phase.
2. The artifact stage is created exclusively, written and `fsync`ed, then
   read and validated.
3. An atomic create-if-absent syscall decides the target race. A target that
   wins between the earlier check and the syscall is never overwritten.
4. The manifest stage is written, `fsync`ed, read back, parsed, and compared
   with the exact intended bytes.
5. Immediately before replacement, the publisher rechecks the old manifest,
   artifact, authority objects, template, source/data state, and intended
   manifest.
6. The manifest replacement remains atomic; `data` is then `fsync`ed and the
   complete state is reopened and validated.

Section 12 then gives the controlling scope language. It requires
directory-relative no-follow access, retained directory descriptors, a stable
publication lock, immediate hash rechecks, atomic artifact and manifest
syscalls, and fresh readback. It says those controls close “cooperative
publisher races and ordinary filesystem substitution.” It immediately says
they do **not** claim that repository code can defeat “a hostile process with
unrestricted access to the same account”; such mutation is detected when
possible and is outside the scientific reproducibility claim.

Section 13 makes the operational scope concrete. It enumerates:

- both lanes contending through the same lock;
- a target created after the absence check, decided safely by no-replace;
- a parent swapped during staging, detected by retained identity;
- a manifest changed after authorization or while the candidate is staged,
  detected before replacement;
- symlink, alternate-root, nonregular, cross-device, multilink, durability,
  crash, quarantine, recovery, inventory, and source/data mutations.

It does not enumerate arbitrary same-process code execution or the power to
run attacker code between a validation call and the immediately following
syscall. The final sentence requiring negative cases to preserve pre-existing
canonical bytes applies to the required matrix, not to every operation an
arbitrary same-account program could perform. The latter interpretation would
contradict Section 12 and would be unbounded.

The distinction is observable:

| scenario | in scope? | required behavior |
| --- | --- | --- |
| second conforming atmosphere/synthesis publisher | yes | blocks on the same `data` descriptor lock |
| independent target creator races an absent artifact | yes | atomic no-replace loses safely |
| named parent or manifest changes before the immediate boundary check | yes | retained identity/hash check rejects |
| required source or non-target data changes | yes | closed snapshot rejects |
| crash at a named lifecycle point | yes | Section 11 recovery state |
| monkeypatched `os.link`/`os.replace` executes arbitrary code first | no | excluded same-process code execution |
| same-account process ignores the lock and swaps the random private stage after the final check | no | excluded hostile external mutation; detect afterward when possible |

If arbitrary in-process execution were in scope, protecting only the staging
name would not close the threat. The injected code could instead close or
replace retained descriptors, alter candidate buffers, patch validation
functions, invoke the real replacement directly, or rewrite the canonical
manifest without calling the publisher's mutation helper. No choice of Python
wrapper or hash placement can form a security boundary against code already
executing inside the process.

## 4. Classification of the audit findings

### 4.1 Atmosphere

The repaired atmosphere history freezes publisher `d9cb38e5…`. It records
that the former real blockers were closed:

- both lanes lock the same canonical `data` directory descriptor;
- artifact link, stage unlink, and manifest replacement use retained
  directory descriptors;
- the exact mutated directory inode is `fsync`ed;
- canonical name rebind then rejects a detached parent;
- early staging substitution, target races, manifest races, recovery,
  inventory, exact array semantics, source/data rechecks, and fresh-process
  validation pass.

Its only new P0 injects replacement of the staging name *inside a wrapper at
entry to the real mutation syscall*, after all production checks. That
experiment proves path resolution occurs in the syscall. It does not prove
failure of an enumerated cooperative race. Its classification as a contract
P0 is therefore **rejected**.

This does not rewrite the immutable audit or turn its `REJECT` into an
acceptance. A fresh independent, scope-aware audit must make any publisher
acceptance decision about the exact current bytes.

### 4.2 Synthesis

The original synthesis history demonstrated the same pathname limitation.
The repaired publisher `2a345fd3…` now retains its stage descriptor and
revalidates descriptor, name, inode, link count, mode, size, and exact bytes
inside a narrow helper immediately before `os.link`; it does the analogous
check immediately before `os.replace`. The repaired audit marks the original
substitution findings closed.

That does not make the repaired synthesis publisher acceptable. The same
audit independently finds an in-scope, deterministic P0: its real
non-`None`-authority path quarantines its own manifest temporary before
replacement. That is a reachability defect requiring repair. Its requested
canonical post-mutation parent/name rebinds are also finite, useful checks
already within the contract's explicit parent-race/readback model.

Thus this adjudication changes neither synthesis rejection nor its repair
queue. It only prevents a later audit from reopening the impossible
same-process syscall-entry criterion after the actual functional and
enumerated race defects are fixed.

## 5. Darwin primitive analysis

The host used for this adjudication is:

```text
macOS 26.5.2, build 25F84
Darwin 25.5.0, arm64
workspace filesystem: APFS
```

The local Command Line Tools SDK exposes these relevant public interfaces:

| interface | source identity | destination rule | atomic namespace property | useful here |
| --- | --- | --- | --- | --- |
| `linkat` / Python `os.link` | directory fd + source name | destination must be absent | atomic create-if-absent | yes, contract baseline; source remains name-resolved |
| `renameat` / Python `os.replace` | directory fd + source name | may replace existing destination | atomic replacement | yes, manifest baseline; source remains name-resolved |
| `renameatx_np` | directory fd + source name | `RENAME_EXCL` or `RENAME_SWAP` | atomic flagged rename | no source-fd binding |
| `fclonefileat` | **open source fd** | destination must be absent | atomic clone/create | useful for artifact only |
| `fcopyfile` | open source and destination fds | writes an already-open file | content copy, not atomic namespace replacement | not acceptable for manifest |
| `exchangedata` | two pathnames | both must exist | atomic data exchange where supported | path-resolved and documented unsupported on APFS |

The exact SDK evidence used was:

| SDK object | SHA-256 |
| --- | --- |
| `usr/include/sys/clonefile.h` | `1efe4ef4241c360905ecc40092245334725a9f1b208be8886ce87e2538b89cbd` |
| `usr/include/sys/stdio.h` | `a8078ae1cba1a46a66d0de6fe7f13bbc969cdb7e78fc2d4b9a5c5cdaaf16126b` |
| `usr/include/sys/fcntl.h` | `805fd8c695f8e5e1c327b6852382cc5533738bbfd8f18bc11f850531166e4fe8` |
| `usr/share/man/man2/clonefile.2` | `4d782d64dc79326de302ee78d7fdb631f260df28e379aee3b20d47ebce09bd74` |
| `usr/share/man/man2/rename.2` | `896c6a278dcfd14601147661cf2a4588d85b271ec8798bbec7913c1d17967033` |
| `usr/share/man/man2/exchangedata.2` | `2411f89255b9b23b4d69da9acf3b0d1a3db2615822c4609907374086666d021a` |

The Darwin headers provide no `AT_EMPTY_PATH` operation analogous to Linux's
fd-only link extension. They declare `fclonefileat(int srcfd, int dst_dirfd,
const char *dst, uint32_t flags)`, while every public rename variant takes a
source pathname.

### 5.1 Direct `fclonefileat` probe

An isolated `/private/tmp` probe on this host:

1. created a mode-`0600` source containing
   `descriptor-bound-clone`;
2. opened the source;
3. unlinked its only staging name;
4. called `fclonefileat(source_fd, destination_parent_fd, "artifact", 0)`;
5. called the same operation a second time with the destination present.

The exact results were:

```text
first call:
  return                         0
  source staging name absent     true
  destination bytes              b"descriptor-bound-clone"
  destination mode               0600
  destination link count         1

second call:
  return                         -1
  errno                          17 (EEXIST)
```

This proves that the current APFS host can create the artifact atomically from
the validated source descriptor without re-resolving a source name.
`fclonefileat` is therefore a legitimate optional artifact hardening path if:

- volume capability is explicitly tested and unsupported volumes fail before
  writes;
- the final one-link inode, exact bytes, mode, size, semantics, and parent
  durability are independently validated;
- the publisher contract's current hard-link-specific expected-inode and
  cleanup assertions are adjusted through a reviewed implementation design;
- no fallback silently weakens the reviewed primitive.

It is not a manifest solution. Its destination is create-only and an existing
`data/MANIFEST.json` yields `EEXIST`. Cloning to another temporary name merely
postpones the same name-resolved source problem to the later `rename`.

### 5.2 Why the manifest has no descriptor-bound equivalent

Within the documented Darwin interfaces above, no operation simultaneously:

1. identifies the new manifest by an already-open source descriptor;
2. atomically replaces an existing named `data/MANIFEST.json`; and
3. leaves no interval containing a partially written manifest.

`renameat` and `renameatx_np` provide the required atomic replacement but
resolve the source name. `fclonefileat` binds the source descriptor but
requires an absent destination. `fcopyfile` binds both descriptors but
changes the existing inode's contents rather than performing an atomic
namespace replacement. `exchangedata` is pathname-based, deprecated for this
use, and unsupported on APFS. `RENAME_SWAP` is also pathname-based; it may
preserve the old manifest under the other name, but it can still place a
substituted source at the canonical name and temporarily expose it.

Accordingly, a rule requiring descriptor-bound replacement of the existing
manifest must either:

- fail closed and declare this Darwin/APFS publication unsupported; or
- introduce a materially different security architecture, such as a
  separately privileged/capability-confined helper that prevents the
  repository-owning account from mutating the staging namespace.

Neither is the smallest implementation of the frozen scientific publication
contract.

## 6. Honest forward path

### 6.1 Required baseline

Keep the current atomicity. Do not replace `os.replace` with in-place
descriptor writes, copy/truncate, pre-delete, or check-then-overwrite.

For both publishers, the acceptance baseline should be:

1. fixed repository and destination;
2. strict detached authority and review;
3. same canonical `data` descriptor lock held through the full transaction
   and fresh validation;
4. create-exclusive/no-follow stage with exact mode;
5. retained stage descriptor where the implementation already has one;
6. descriptor-read exact bytes plus name/device/inode/link/mode/size rebind in
   the narrow helper immediately before mutation;
7. retained canonical parent descriptor revalidated immediately before the
   syscall;
8. atomic no-replace for the artifact and atomic replace for the manifest;
9. `fsync` of the exact retained mutated directory;
10. canonical-path rebind and complete fresh no-follow validation;
11. fail-closed recovery without destructive rollback.

This is the contract's explicitly bounded transaction. The late interval
between the last check and a pathname-based kernel operation is acknowledged,
not falsely claimed absent.

### 6.2 Artifact hardening

`fclonefileat` may replace the hard-link primitive as defense in depth on a
capability-proven Darwin volume, but it should not be a prerequisite for
accepting the manifest transaction and should not be installed as an
untested fallback. It improves the artifact source binding only. It cannot
justify a stronger claim for the manifest.

The smallest path is therefore to keep the currently reviewed hard-link
primitive under the explicit threat boundary unless another already-required
publisher repair makes the small, separately reviewed `fclonefileat` change
worthwhile.

### 6.3 Audit closure

The next independent reviews should:

- treat syscall-entry monkeypatch substitution as an out-of-scope
  defense-in-depth probe, not an acceptance gate;
- continue to require every explicit Section 13 mutation and exact
  unchanged-byte postcondition;
- require immediate checks to be visibly adjacent to the mutation helper;
- reject any helper that claims descriptor-bound replacement when it still
  names the source;
- state the Section 12 hostile same-account exclusion in the acceptance
  rationale.

For atmosphere, the immutable `df5831…` rejection remains the historical
decision. A new scope-aware independent audit may re-evaluate exact publisher
`d9cb38e5…`; this document does not accept it.

For synthesis, first repair the deterministic self-quarantine/reachability
blocker and the finite canonical post-mutation rebind issues recorded in
`7790c4…`. Then perform a new exact-byte audit. Do not spend another repair
cycle trying to construct a nonexistent descriptor-bound Darwin manifest
replacement.

## 7. Final disposition

**REJECT the audit scope, not the observed trace.**

The syscall-entry wrappers accurately demonstrate what arbitrary same-process
or unrestricted same-account code can do. The frozen contract explicitly does
not claim to withstand that actor. Treating the trace as an acceptance P0
silently expands a scientific reproducibility transaction into a hostile
same-user security boundary, contradicts Section 12, and requires a Darwin
manifest primitive that does not exist in the documented public API.

The useful and honest design is:

- preserve atomic no-replace artifact installation and atomic manifest
  replacement;
- preserve all immediate validations, retained descriptors, durability, and
  post-checks;
- optionally use capability-gated `fclonefileat` for artifact hardening;
- explicitly accept the path-resolved manifest source under the frozen
  hostile-account exclusion;
- close the remaining genuinely in-scope functional and enumerated race
  defects through fresh independent review.

**DESIGN ADJUDICATION ONLY.** This report grants no publisher acceptance,
detached authorization, authorization review, artifact installation, manifest
mutation, cleanup authority, or postpublication acceptance.
