# Chapter 6 schematics independent audit

Status: repair re-audit complete  
Audited: 2026-07-30; repair re-audited: 2026-07-30  
Disposition: **ACCEPT for notebook authoring**

The original snapshot below was rejected for two finite reasons:

1. schematic 3 labelled the immutable source record with conceptual energy
   `E_l`, whereas the accepted Chapter 6 outline requires the stored
   lower-excitation term wavenumber \(\tilde E_l=E_l/(hc)\);
2. the manifest did not yet record a generation date or an immutable binding
   to the exact prompt/caption/alt-text specification that produced each
   output.

The many-depth asset also needed a mobile-label correction or an explicit
minimum-width/zoom presentation contract. No other schematic needed a
scientific recomposition. The immutable history of that decision is retained
below; the repair re-audit in Section 7 closes every blocker.

No PNG, prompt registry, manifest, outline, chapter, or external website file
was edited by this audit. This report is the only file added.

## 1. Exact review snapshot

| authority | SHA-256 |
| --- | --- |
| `scripts/textbook_schematic_specs.py` | `420f1a72d9a796d9de02812b1790fdbd129d8f87304c9acce84a991496e4bf09` |
| `design/chapter06_causal_outline.md` | `1b66df5d548f2854f83289fcf9de5109058f1482a7b64aadaff3505d1f57e019` |
| `assets/schematics/MANIFEST.json` | `46b9c6053288e678e6b75f6dbdd8764ff05f9e7449ca7ce5c17a3b3c56e74bb5` |
| website style master copied as a read-only reference | `6c7061aef963b97adbe5795aecf8da1db5bca8f8ed3218049968b58c9a5793a2` |

Every Chapter 6 PNG was inspected with the original pixels preserved. A
separate temporary 420-pixel-wide rendering was also inspected to approximate
a narrow notebook/mobile viewport. Those temporary previews are not textbook
assets and are not evidence of a changed source image.

The current CLI-rendered prompt-text identities, including the emitted final
newline, are:

| prompt id | current prompt SHA-256 |
| --- | --- |
| `ch06-smooth-background-narrow-line` | `01f3b8b80ac203eb6c5e0ebdb0f2d5a841e20c7c1fea65ef99529829d8072795` |
| `ch06-two-levels-one-photon` | `854622c9aaa28286f933a6df2df942dc1535d6b9d10065782cdd6dff236d750f` |
| `ch06-core-wings-convolution` | `3db5d64893ca499143d82dc8c1729e3fed91243fa40229fa8db78b70a93aa431` |
| `ch06-one-record-many-depths` | `78cb74b655c11b55d63e3a9daceda1fc60b083719cf212980d66f1f4a2a9a334` |

These hashes describe the current registry, not necessarily the unrecorded
generation-time prompt bytes. That distinction is the provenance finding in
Section 5.

## 2. File-identity gate

All four manifest byte counts, dimensions, and output hashes match the actual
RGB, non-interlaced PNG files exactly.

| asset | actual dimensions | actual bytes | actual and manifest SHA-256 |
| --- | ---: | ---: | --- |
| `ch06-smooth-background-narrow-line-v1.png` | 1536 × 1024 | 1,106,031 | `9239713ac35d3c32b86067c7a613e560959500c1f84e9f35901659d6be2b7382` |
| `ch06-two-levels-one-photon-v1.png` | 1774 × 887 | 1,066,672 | `8c91501a6dad98564ae72fc44bd1840397223d7359184bc9af02cfebc6a82e09` |
| `ch06-core-wings-convolution-v1.png` | 1619 × 971 | 988,019 | `86ce5dec27b1de45e47e8484c5c4a11ea212a50a79739a26097ee6e6637f1ec4` |
| `ch06-one-record-many-depths-v1.png` | 1686 × 933 | 1,168,746 | `f68927c95ef0990e7a1a7817de0d012b3c5a8a9e276046c4db212b695752978e` |

There are exactly four Chapter 6 manifest entries and exactly four Chapter 6
`FigureSpec` entries. Their prompt ids and asset paths pair one-to-one. Each
spec has a nonempty prompt, alt text, and caption. Each manifest entry has the
correct chapter, role, generator, style-rule path, website-style reference,
reference-image hash, dimensions, byte count, output hash, and a
figure-specific scientific-review checklist.

No missing file, duplicate prompt id, stale output hash, or dimension mismatch
is present.

## 3. Asset-by-asset scientific audit

### 3.1 Smooth background and one narrow opacity excess: **ACCEPT**

Actual asset:
`assets/schematics/textbook/ch06-smooth-background-narrow-line-v1.png`

The rendered image satisfies the opening job:

- one broad, smooth slate curve is labelled exactly
  “continuum mass extinction”;
- one narrow deep-navy contribution is labelled exactly
  “line mass absorption”;
- the two processes are not joined by a slash or described as identical;
- off-line and line-center wavelengths are both marked;
- a single question arrow connects the narrow excess to exactly two bound
  levels;
- there are no numerical ticks, invented values, line wings, catalog,
  spectrum forest, source code, GPU, star, intensity, or flux.

The extinction-versus-absorption distinction is scientifically honest. The
line contribution is drawn as its own opacity component rather than as an
observed downward feature. The horizontal guide is labelled wavelength but has
no numerical scale, so it does not masquerade as a measurement.

The current caption is essential and correct: it says the quantities have the
same units but different physical origins and that neither is emergent or
observed flux. Use that guard beside the image.

At 420 pixels wide, the central question, both curve labels, and wavelength
labels remain readable. The pale-beige off-line label has the lowest contrast
in the set but remains legible on white. It should not be made lighter in any
future regeneration.

### 3.2 Two levels and one resonant photon: **ACCEPT**

Actual asset:
`assets/schematics/textbook/ch06-two-levels-one-photon-v1.png`

The image contains:

- a side atom labelled “bound before and after”;
- exactly two horizontal energy levels;
- three conceptual particles on the lower level;
- one upward wavy photon arrow;
- correctly spelled labels `lower level E_l`, `upper level E_u`, and
  `photon h nu_l`;
- one narrow wavelength marker labelled `lambda_l = c / nu_l`;
- one causal arrow from the energy-gap construction to the wavelength marker.

Here \(E_l\) and \(E_u\) are correct: this is the conceptual energy-level
picture used immediately before
\(h\nu_l=E_u-E_l\). It is not a source-record card and does not claim that the
catalog stores energy in erg. The caption properly declares spacing, particle
count, and marker width conceptual and not to scale.

There is no line depth, broadening, damping, catalog, many-line spectrum,
software object, or numerical value. All labels are clear at original and
420-pixel width.

### 3.3 Gaussian and Lorentzian causes meet by convolution: **ACCEPT**

Actual asset:
`assets/schematics/textbook/ch06-core-wings-convolution-v1.png`

The two causes remain visually independent until one convolution mark:

- thermal and microturbulent velocity arrows feed only the Gaussian branch;
- radiative, electron, and neutral symbols feed only the Lorentzian branch;
- both branches enter one large convolution symbol;
- the output has one labelled core and two labelled wings;
- the lower guide presents three ideal-profile shapes with the label
  “equal ideal area.”

The spelling of “microturbulent,” “Gaussian,” and “Lorentzian” is correct.
The image does not imply a simple weighted sum, and it includes no Harris
table, FASTEX rule, cutoff threshold, code, catalog, multiple lines, or
numerical values.

The caption makes the important honesty distinction: convolution preserves
the ideal continuous area, while a later continuum-relative production cutoff
is a separate numerical operation. The image itself does not display that
production machinery.

This is the densest of the first three assets. At 420 pixels wide, the branch
structure and primary labels remain readable; “radiative,” “electron,”
“neutral,” and the wing annotations are small but still distinguishable.
Render it at full notebook width and retain click-to-enlarge behavior on a
narrow viewport.

### 3.4 One immutable record through many layers: **REJECT**

Actual asset:
`assets/schematics/textbook/ch06-one-record-many-depths-v1.png`

Most of the composition is excellent:

- one record fans out to exactly six layers;
- outer-to-inner direction is explicit and correct;
- every layer shows \(T,\rho,n_e,n_{\rm pert},n/U,\delta\);
- six profiles vary in height, core width, and wing extent;
- all six share one vertical center guide;
- the rows align with a right-hand object labelled
  `line opacity (depth, wavelength)`;
- the image contains no selection tree, many-line accumulation, molecular
  band, radiative transfer, emergent flux, GPU, or performance claim.

The heatmap-like product can look quantitative in isolation, but the current
caption explicitly says its shapes and colors are conceptual rather than
parity data. That caption guard must remain.

The blocking label is the record entry `E_l`. The accepted causal outline at
line 1000 requires:

```text
lambda_l, tilde E_l, gf, gamma
```

The implementation record stores
\(\tilde E_l=E_l/(hc)\) in cm\(^{-1}\), not conceptual energy \(E_l\). Because
this is explicitly “one line record,” retaining `E_l` invites the exact
energy-versus-stored-wavenumber confusion the revised outline removed.

The mismatch exists in all three coupled places:

- prompt line 797 requests `E_l`;
- the PNG renders `E_l`;
- the caption says only “excitation,” without stating stored excitation
  wavenumber.

Required repair:

1. change the prompt entry to `tilde E_l` or another visually unambiguous
   rendering of \(\tilde E_l\);
2. regenerate the PNG so the record card uses that notation;
3. update the caption to say “stored lower-excitation wavenumber” while leaving
   the conceptual \(E_l\) notation in schematic 1 untouched;
4. update the manifest output identity and scientific checklist after visual
   review.

At 420 pixels wide, the overall fan-out, six profiles, center alignment,
outer/inner order, and slab remain clear, but the record entries and repeated
state tokens are not reliably readable. Since those labels are part of the
claimed teaching content, one of these must be demonstrated before acceptance:

- regenerate with materially larger record/state text and less unused space;
- use a responsive minimum-width container with horizontal scrolling and
  click-to-enlarge behavior, then document and inspect that rendered path;
- simplify repeated state strips while preserving the causal statement that
  every layer receives the same six state categories.

Alt text alone is not a substitute for visible labels, although the supplied
alt text is useful.

## 4. Four-image family audit

### Scientific and narrative continuity: **PASS except for schematic 3 notation**

The four pictures form the intended causal sequence:

```text
smooth continuum cannot explain a narrow opacity excess
→ two levels select one photon
→ velocities and damping create one profile
→ one fixed transition is evaluated through changing layers
```

No schematic pre-teaches catalog selection, sparse accumulation, line forests,
special profiles, radiative transfer, emergent flux, or production
optimization. “One line record” is the allowed single-transition object, not a
catalog tour.

Schematic 1's \(E_l,E_u\) and schematic 3's required \(\tilde E_l\) serve
different legitimate roles. The present failure is not that \(E_l\) is
forbidden everywhere; it is that the implementation-record picture currently
fails to mark the coordinate change.

### Visual-family consistency and originality: **PASS**

All four use:

- pure white background;
- hand-sketched charcoal/slate contours;
- deep navy for the causal path;
- pale beige and muted blue-grey fills;
- slightly irregular strokes;
- short direct labels;
- landscape composition and generous whitespace;
- no title strip, logo, watermark, photorealism, decorative scene, or invented
  number.

They visibly belong with the website-inspired scientific-notebook family.
They do not copy the style-master composition:

- schematic 0 is a curve-to-level question;
- schematic 1 is an atom-to-level-to-wavelength chain;
- schematic 2 is a two-branch convolution;
- schematic 3 is a one-to-many depth fan-out ending in a slab.

Schematic 3 reuses the visual vocabulary of a spiral paper record, but its
fan-out, state strips, aligned profiles, and slab are a new composition and a
new teaching claim. That is aesthetic reuse rather than figure copying.

### Whitespace, clipping, and native-resolution legibility: **PASS**

No original PNG has clipped strokes or labels, label overlap, accidental
frame, excessive legend, or illegible native-resolution text. Schematic 0 has
the most open upper-left region, but that space balances the upper-right
question and levels rather than looking like a failed crop. Schematics 2 and 3
are dense but orderly.

### Narrow mobile rendering: **PASS for 0–2; CONDITIONAL for 3**

At the inspected 420-pixel width:

- schematic 0 retains its complete question;
- schematic 1 is especially clean;
- schematic 2 retains its causal branches but benefits from zoom;
- schematic 3 loses reliable readability of the record and layer-state text.

The notebook should never hard-code a small raster display. Original pixels
are ample; the remaining question is responsive presentation and relative
label size.

## 5. Prompt and manifest completeness

### What is complete

The four current `FigureSpec` objects provide:

- unique ids and Chapter 6 ownership;
- complete generation prompts;
- exact asset paths;
- useful alt text;
- scientifically guarded captions.

The four current manifest objects provide:

- the same unique prompt ids and paths;
- generator identity;
- style-rule and website-aesthetic sources;
- the style-reference image hash;
- original-textbook conceptual role;
- exact output dimensions, bytes, and hashes;
- Chapter 6 ownership;
- figure-specific scientific-review checklists.

The hashes and dimensions are correct, and all rendered labels other than the
schematic 3 notation issue agree with the current prompt requests.

### Blocking provenance omissions

The causal outline requires every final asset to record its prompt/specification,
generator, date, source hash, output hash, caption guard, alt text, and
visual/scientific review. The current two-file arrangement does not bind all of
those items into an immutable generation record:

- no generation date or timestamp appears in a Chapter 6 manifest entry;
- no `prompt_sha256` or generation-time
  `prompt_source_sha256` appears;
- the manifest points to a mutable Python source path without recording which
  prompt bytes produced the current PNG;
- alt text and caption live in that mutable registry but are neither copied nor
  hashed in the manifest;
- the scientific checklist says a caption “must” provide a guard, but the
  manifest does not bind the exact accepted guard text.

The website source-image hash and PNG output hash are present, but they do not
answer whether the current prompt/caption/alt bytes are the generation-time
ones. The schematic 3 prompt/outline drift demonstrates why this binding
matters.

Required manifest repair for every Chapter 6 entry:

1. record `generated_at` or an equivalent generation date;
2. record a per-asset hash of the fully expanded prompt bytes, with the byte
   convention stated;
3. record the prompt-registry source hash used at generation or review;
4. bind the accepted alt text and caption, either as exact strings or hashes;
5. retain generator, style-reference input hash, PNG dimensions/bytes/hash,
   and independent scientific-review status;
6. after schematic 3 is regenerated, replace its output metadata and review
   checklist rather than carrying forward the old acceptance claim.

The current prompt hashes in Section 1 may seed this repair for unchanged
assets, but they must not be asserted as historical generation hashes without
evidence.

## 6. Acceptance gate

The following pass now and should be preserved:

- schematic 0 extinction-versus-absorption semantics;
- schematic 1 conceptual energy notation and resonant-photon chain;
- schematic 2 independent broadening causes and ideal-area guard;
- schematic 3 one-record/six-layer/shared-center/slab composition;
- absence of flux, catalogs, many-line machinery, source code, and performance
  leakage;
- exact output byte identities;
- original visual-family composition.

Before notebook authoring:

1. correct schematic 3 from record `E_l` to stored
   \(\tilde E_l\) in prompt, PNG, and caption;
2. demonstrate readable schematic 3 labels at the intended narrow responsive
   rendering;
3. complete and hash-bind the generation date, prompt, caption, and alt-text
   provenance for all four manifest entries;
4. re-run native-resolution label/science review and update the regenerated
   PNG's dimensions, bytes, and hash.

**Final disposition: REJECT the four-asset set for notebook authoring at the
snapshot above.** The first three PNGs are individually acceptable. The fourth
needs one scientifically meaningful notation correction, and the set needs a
complete immutable provenance record.

## 7. Repair re-audit

### 7.1 Scope and disposition

This section is a targeted independent review of the repair, not a rewrite of
the historical rejection above. No PNG, prompt specification, manifest,
reader JavaScript, reader CSS, outline, chapter, or website source was edited
by this re-audit. The only persistent change is this report.

All four original blockers are closed:

1. schematic 3 now distinguishes stored
   \(\tilde E_l=E_l/(hc)\) from conceptual \(E_l\) in its prompt, image,
   alt text, and caption;
2. its record card and repeated layer-state labels are materially larger and
   readable in the inspected 420-pixel rendering;
3. every Chapter 6 asset is date-bound and hash-bound to the accepted prompt,
   alt text, caption, prompt registry, and exact source image input;
4. the regenerated PNG's dimensions, byte count, and hash match the manifest,
   and the reader provides a full-resolution, two-axis-scrollable view.

**Repair disposition: ACCEPT the four-asset set for notebook authoring.**

The manifest deliberately still labels schematic 3
`repaired_pending_independent_reaudit`; this report supplies that independent
decision. Updating that workflow-state string, if desired, belongs to the
manifest owner and was outside this read-only audit.

### 7.2 Exact repaired snapshot

| authority | SHA-256 |
| --- | --- |
| `scripts/textbook_schematic_specs.py` | `07eee64c4af9586cd596e11dd1866c4e2536a202832e70dcb02a7ccc4d975535` |
| `design/chapter06_causal_outline.md` | `1b66df5d548f2854f83289fcf9de5109058f1482a7b64aadaff3505d1f57e019` |
| `assets/schematics/MANIFEST.json` | `019200a5af77d8221afd4cec62ad1fe03d60370a4357c488630c563649c3e303` |
| `assets/render.js` | `283333a4e527009b6f17aaf588d18325d2675b9e0633796e947ed1198d7d952b` |
| `assets/style.css` | `66b1e78d7876fbc5dc1b32efca9f0d395c4d9fe24f812dfeef963474b398dbed` |
| copied website style master | `6c7061aef963b97adbe5795aecf8da1db5bca8f8ed3218049968b58c9a5793a2` |

The source and copied website style-master files have the same final hash. The
current website generator source is identified by path in the manifest; its
current SHA-256 is
`70a76df7488e1c8a9fa137bf7efb9dd78a1af36419838a2b7f73995e360d6d57`.
The generated schematics depend on the separately hash-bound accepted
prompts, rather than claiming that this mutable generator script is a
generation transcript.

All actual output bytes, dimensions, and manifest values agree:

| asset | dimensions | bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `ch06-smooth-background-narrow-line-v1.png` | 1536 × 1024 | 1,106,031 | `9239713ac35d3c32b86067c7a613e560959500c1f84e9f35901659d6be2b7382` |
| `ch06-two-levels-one-photon-v1.png` | 1774 × 887 | 1,066,672 | `8c91501a6dad98564ae72fc44bd1840397223d7359184bc9af02cfebc6a82e09` |
| `ch06-core-wings-convolution-v1.png` | 1619 × 971 | 988,019 | `86ce5dec27b1de45e47e8484c5c4a11ea212a50a79739a26097ee6e6637f1ec4` |
| `ch06-one-record-many-depths-v2.png` | 1684 × 934 | 1,280,525 | `aae76239340b257f508e5f9b61e778c8cd29ef640825f70d9e767ffd75368e62` |

The original schematic-3 revision remains an owned source artifact, but it is
not active. A repository search over `assets`, `scripts`, `book`,
`reader.html`, and `index.html` finds no reference to
`ch06-one-record-many-depths-v1.png`. The registry and manifest both select
v2. The manifest retains only the old output hash
`f68927c95ef0990e7a1a7817de0d012b3c5a8a9e276046c4db212b695752978e`
as the immutable `owned_prior_revision` input to the edit.

### 7.3 Schematic-3 notation and narrow-view inspection

The 1684 × 934 source was inspected at native pixels. A separately generated
420 × 233 preview was then inspected beside a 420 × 232 rendering of v1; both
temporary previews were deleted after review.

The repaired record card visibly reads:

- `lambda_l`;
- `tilde E_l [cm^-1]`;
- `gf`;
- `damping data`.

This is the stored line-list quantity required by the causal outline, not a
claim that the record stores the conceptual energy \(E_l\). The caption makes
the conversion explicit:
\(\tilde E_l=E_l/(hc)\). The alt text independently calls the quantity a
stored lower-excitation wavenumber.

At 420 pixels, all six repeated strips can still be read as
`T, rho, n_e, n_pert, n/U, delta`; the record entries, `outer`, `inner`, and
`line opacity (depth, wavelength)` labels also remain identifiable. The
corresponding v1 preview has substantially smaller record and strip labels.
The repair therefore makes a material legibility improvement while preserving
exactly six layers, the shared center guide, and the depth-by-wavelength slab.
No overlap, clipping, invented parity values, or new scientific ambiguity was
introduced.

The accepted normalized text bindings, computed independently with the
manifest's declared
`SHA-256(UTF-8 FigureSpec field after strip())` convention, are:

| prompt id | accepted prompt | accepted alt text | accepted caption |
| --- | --- | --- | --- |
| `ch06-smooth-background-narrow-line` | `32e88db172afc392bbe66fab702ccf5c00215452f73ee14b06010638b10a1729` | `4195cf24fbc99578a9c1da69b4c73d6eddfc4d609792c86e021adfa68e7579b2` | `be24c0d494264268f1c4ee92f6f9c2d60b23153512f0830e830b779536f9d393` |
| `ch06-two-levels-one-photon` | `cbe115195cc88c9516c84e40f14ff9e1fe790df8446f0034b271cfdb6765842b` | `f782bd8cc03f7801d3a887e04887c0a4aa23436e6d7baa28a83e1396060b4d6d` | `8d8f3166b403be9953b141491ee267a75e619cfb36579947f5ebc6ba5949b8a8` |
| `ch06-core-wings-convolution` | `8462a44992c2e0a7e580aae33877e08c9ccad60d00828c62b5e80dcc70688433` | `8a572e68b0d2ff1a538ad2e62e8dea577c47b4fe05ddac084f294a4f1d5cf2af` | `586d53d05995dc2b9e24938a0d69fbdf23eeb122f0cdeafe335c754175beb90d` |
| `ch06-one-record-many-depths` | `64cfd8c938ab32cc2f7b0841e5ee2c2da9a3854939ca33ec4497064fe6c629c4` | `5012fc1de3f9ec3169f5fa49a4b2f34ae2a1fbe1931227a9df2a21c1853fa653` | `2958a759f6ac28627a2f6f53b4d3b8e678a449eccc0a731bbffbbde6d2b2d92d` |

All four entries record generation date `2026-07-30`, use registry hash
`07eee64c4af9586cd596e11dd1866c4e2536a202832e70dcb02a7ccc4d975535`,
and recompute to the values above. The first three bind the exact style-master
input hash
`6c7061aef963b97adbe5795aecf8da1db5bca8f8ed3218049968b58c9a5793a2`;
schematic 3 instead binds the exact owned-prior-revision hash
`f68927c95ef0990e7a1a7817de0d012b3c5a8a9e276046c4db212b695752978e`.

The provenance language is honest about the available evidence. The first
three entries explicitly call their prompts accepted canonical regeneration
prompts and state that they are not asserted to be verbatim generator-request
transcripts. Schematic 3 calls its text an accepted canonical edit prompt
reviewed against the final asset and its owned prior revision; it likewise
makes no claim of a verbatim transcript. Thus the hashes bind the reviewed,
reusable canonical specifications without silently upgrading them into
unavailable session logs.

### 7.4 Reader enlargement and accessibility evidence

The current reader adds the following concrete behavior to every prose image:

- `tabindex="0"`, `role="button"`, and an action-specific accessible label;
- click, Enter, and Space activation;
- a full-resolution overlay with `role="dialog"`, `aria-modal="true"`, and a
  labelled Close button;
- initial focus on Close;
- closure by Close, Escape, or the overlay background;
- focus restoration to the activating image with `preventScroll`;
- natural-width image presentation with unrestricted image `max-width`;
- horizontal and vertical overlay scrolling while page scrolling is locked.

The app browser backend was not exposed to this audit, so no second live
browser smoke test is claimed. Instead, a disposable DOM harness executed the
actual `assets/render.js` file. It independently dispatched Enter and click,
confirmed dialog creation and focus placement, dispatched the image load and
observed width `1684px`, then closed independently with Escape and Close and
confirmed source-focus restoration. It also asserted the actual CSS contracts
`overflow:auto`, `min-width:max-content`, `min-height:100%`,
`max-width:none`, and the page-scroll lock. The harness and all temporary
files were removed after the run. Its result was:

```text
PASS: Enter and click open; natural width is 1684px; Close and Escape restore focus; CSS enables two-axis dialog scrolling.
```

This is a targeted verification of the requested reader contract, not a claim
of a complete assistive-technology conformance audit.

### 7.5 Executed gates

```text
python -m pytest -q tests/test_schematic_provenance.py
..............                                                           [100%]
14 passed in 0.04s

node --check assets/render.js
```

`node --check` exited zero with no output. The provenance suite independently
recomputed the prompt, alt-text, caption, registry, output, dimension, and
Chapter 6 path bindings, including the v2-only schematic-3 registry path.
