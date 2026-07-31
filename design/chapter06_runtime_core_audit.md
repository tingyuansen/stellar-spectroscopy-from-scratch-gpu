# Chapter 6 runtime-core independent re-audit

Status: independent repaired-implementation review  
Disposition: **ACCEPT for Chapter 6 notebook authoring**  
Audited: 2026-07-30

## 1. Scope and immutable review snapshot

This re-audit reviewed, without editing the runtime or its tests:

- `book/chapter06_runtime.py`, 1332 lines, SHA-256
  `c011c542b72539c94537bd71def1ea0b63f1b69c1014107edd0c23821083a0fe`;
- `tests/test_chapter06_runtime.py`, 756 lines, SHA-256
  `0a0d3d38654f163a073a32dc6d7ce16e95d3a6b1122f9ed074f61f96a423808f`;
- `design/chapter06_exact_source_contract.md`, SHA-256
  `ffa099359649b62e0e97fbfd1b347989c83024f5f7ba86a19bb693bfc04d6ca1`;
- `design/chapter06_causal_outline.md`, SHA-256
  `e1ff327bcf6c2588708e7291b5af395b8cc4e23d665773cf75829b49068b65a3`;
- the accepted raw one-row subset, SHA-256
  `bb7ae01fe718c9bbeb0bec74cad1e9d1e7d47e7b63c6c5fb27cf6e5b3030fe04`;
- the staged atmosphere and synthesis sources named below.

The external Payne Zero tree remained read-only at commit
`9c44001feae40b85146630499e6f8a5fed42e5af`. The paper tree remained
read-only; `main.tex` retained SHA-256
`e11507b9150550b246f6664debf22e540aa92d8261eb40daabb594da91bd8e0d`.

Commands and outcomes:

```text
python -m pytest -q tests/test_chapter06_runtime.py
15 passed in 2.46s

ruff check book/chapter06_runtime.py tests/test_chapter06_runtime.py
All checks passed!

ruff format --check book/chapter06_runtime.py tests/test_chapter06_runtime.py
2 files already formatted
```

The post-acceptance test-only delta was inspected at the Doppler
microturbulence check. It changes the mass-table load from an implicit default
to:

```python
load_atomic_masses(
    REPOSITORY_ROOT / "data/static/synthesis_tables/atomic_masses.npz"
)
```

No runtime line, expected value, tolerance, physical formula, or production
call changed. The accepted runtime SHA-256 is unchanged.

No runtime, test, source, subset, fixture, static table, prospective golden,
external Payne Zero file, or paper file was modified by this review. This
report is the only file changed.

## 2. Executive verdict

The prior P0 physics, production-bridge, and source-origin findings are
repaired correctly. The runtime now:

- names the partition-normalized excitation factor honestly and exposes the
  subsequent \(gf\)-weighted factor;
- distinguishes the analytic direct-exponential/full-profile construction
  from exact production FASTEX and ordinary-metal shortcuts;
- reconstructs every one of the 24 regime-depth center deposits bitwise;
- retains the exact Torch tensor, device, and dtype contract;
- returns defensive record copies;
- treats Doppler widths as Chapter 3-owned inputs;
- exposes both Harris authorities and the exact synthesis wing policy;
- recovers exact cutoff masks, stimulation, loop/batched equality, and reach.

The formerly order-dependent grid import is now repaired:
`build_synthesis_wavelength_grid()` calls `configure_local_data_paths()`
before importing `Grid`. A genuine subprocess regression test calls this
helper first and proves that `atomic_lines.py` resolves under the textbook's
staged `src/` tree.

The disposition is therefore **ACCEPT for Chapter 6 notebook authoring**.
No P0 or P1 runtime-core blocker remains. Final Chapter 6 publication still
has the separate atmosphere-oracle, golden, backend, notebook, and render
gates declared by the exact source contract.

## 3. Localized source-origin repair verification

### P0-1 — Fresh first call now uses staged Payne Zero: PASS

Independent fresh-interpreter check:

```python
from pathlib import Path
import book.chapter06_runtime as chapter06

chapter06.build_synthesis_wavelength_grid()
import payne_zero_synthesis.atomic_lines as atomic_lines

print(Path(atomic_lines.__file__).resolve())
```

Observed:

```text
/Users/ysting/stellar-spectroscopy-from-scratch-gpu/src/payne_zero_synthesis/atomic_lines.py
```

The returned grid also retained its exact numerical contract:

```text
shape:    (6000,)
dtype:    float64
first:    495.0009387906341 nm
last:     504.9989209057178 nm
```

The new test uses `subprocess.run([sys.executable, "-c", code], ...)`, so it
does not inherit the parent test process's already-imported Payne Zero
modules. It calls the grid helper before importing
`payne_zero_synthesis.atomic_lines`, then asserts that the resolved file is
relative to `REPOSITORY_ROOT / "src"`. This closes the exact order-masking
failure from the first audit.

The existing fail-closed behavior for an already-loaded external module also
remains intact.

## 4. Prior P0/P1 repair ledger

### P0 — Honest population semantics: PASS

The obsolete false field `lower_level_population_cm3` is gone.
`StrengthCheckpoint` now reports, separately:

```text
excitation_weighted_partition_normalized_population_cm3
gf_weighted_excitation_factor_cm3
```

The numerical identity is exactly

```python
gf_weighted_excitation_factor_cm3 == (
    partition_normalized_population_cm3
    * lower_level_boltzmann_factor
    * oscillator_strength
)
```

The source field named `oscillator_strength` is used once as \(gf\); no
second statistical-weight factor appears.

### P0 — Analytic-to-production bridge: PASS

`gross_line_strength_checkpoint` is explicitly an analytic
direct-exponential reference. `dense_line_checkpoint` labels its selected
full-profile authority rather than presenting it as the optimized deposited
profile.

`synthesis_center_policy_checkpoint` exposes:

- the physical excitation exponent;
- direct and FASTEX weights;
- pre- and post-excitation strengths;
- pre- and post-FASTEX cutoff decisions;
- the exact damping ratio;
- full-Harris, shortcut, and selected center values;
- the selected branch;
- host and float32 classical strengths;
- the float32 reconstructed deposit and staged-kernel deposit.

For all 24 regime-depth pairs, the reconstructed float32 center value was
bitwise equal to `accumulate_atomic(..., apply_stim=False)`.

The maximum measured center-deposit cast effect over active depths was:

```text
absolute: 8.168161746979763e-08 cm2 g-1
relative: 3.971260638886659e-08
```

This keeps float32 deposition visible instead of hiding it behind a float64
comparison view.

### P0 — FASTEX policies: PASS numerically

Atmosphere scalar and compiled lookup values were bitwise equal on the audit
matrix. The two nominal half-step checks gave:

```text
x = 0.0005 -> exp(-0.001) = 0.999000499833375
x = 0.2385 -> exp(-0.239) = 0.7874148823726879
```

The synthesis CPU-float64 lane produced the same values. The MPS-compatible
CPU-float32 table evaluation produced `0.9990005` and `0.7874149`.

The distinct domain policies are preserved:

```text
x = -1: atmosphere 0.0; synthesis float64 exp(+1)
x = 0: both 1.0
x >= 1001: atmosphere 0.0; synthesis direct-exp branch
x = +inf: both 0.0
x = NaN: atmosphere 0.0; synthesis NaN
```

At `x=1000.999` and `x=1001`, all displayed values underflow to zero, but the
source branch ownership remains distinct. A deliberately sensitive
float32/f64 half-step (`x=1.2345`) also demonstrated that dtype-dependent
quantization is real and must not be normalized away:

```text
synthesis float64: 0.2911257425960852
synthesis float32: 0.29083478
```

### P0 — Harris seams and ordinary-metal shortcuts: PASS numerically

The independent audit evaluated exact values and one-sided neighbors at:

- \(a=0.2\);
- \(a=1.4\);
- \(a+|u|=3.2\);
- \(|u|=10\);
- \(a=100\);
- a 0.005-table-index half-step.

Across that matrix:

```text
maximum atmosphere scalar-versus-compiled difference: 0.0
maximum atmosphere scalar-versus-f64-compiled difference: 0.0
maximum synthesis full-versus-scalar-transcription difference: 0.0
```

The literal branch ownership is correct:

- \(a=0.2\) uses the full route;
- \(a=1.4\) alone remains in the blend;
- \(a+|u|=3.2\) remains in the blend;
- \(|u|=10\) remains table-based;
- \(a=100\) still receives the correction;
- values just above each boundary select the next route where specified.

The production shortcut is not conflated with the full evaluator. At
\((|u|,a)=(0.15,0.1)\):

```text
atmosphere full authority: 0.8797667615085327
synthesis full authority:  0.8792623223501963
synthesis ordinary wing:   0.8699247980349999
```

Thus both the cross-lane table difference and the synthesis low-damping
shortcut remain measurable.

The focused tests compare both sides of the seams but do not currently pin
every exact-equality point or the float32 half-step values. This is worth
strengthening during the required source-origin test repair; the current
runtime values themselves passed the complete independent seam matrix.

### P0 — Damping and cutoff ledger: PASS

The transformed record retains the exact normalized fields:

```text
radiative:     3.909296919359072e-08
Stark:         4.700008650504819e-21 cm3
van der Waals: 4.093536407068315e-24 cm3
```

For the six solar-dwarf depths, the independently checked final ratios were:

```text
0.00522449324  0.00580802268  0.0117496886
0.0694895979   0.633170867    8.53045124
```

Changing electron density changes only the Stark term; changing the neutral
collision proxy changes only the van der Waals term. The production center
checkpoint recovers:

```text
hot_dwarf pre-cutoff:  1 1 1 1 0 0
hot_dwarf post-FASTEX: 1 1 1 0 0 0
```

The other three regimes retain all six depths after both tests.

### P1 — Torch device, dtype, and state contract: PASS

`run_synthesis_one_line` retains cloned Torch tensors and returns NumPy copies
only as explicit comparison views. On the available CPU authority:

```text
work dtype:         torch.float64
accumulation dtype: torch.float32
cutoff dtype:       torch.float32
stimulation dtype:  torch.float32
device:             cpu
line counts:        metal 1, auto 0, helium 0
Fe I indices:       ion stage 0, element 25
center/wing index:  2434 / 2434
```

The net and gross tensors both remain Torch float32 on CPU. CUDA and MPS were
both unavailable on the audit host, so backend tolerance publication remains
a later integration gate rather than a claimed result.

### P1 — Exact stimulation lifecycle: PASS

For every cell in all four \(6\times6000\) products:

```python
net_tensor == gross_tensor * (
    1.0 - torch.exp(-frequency_float32 * photon_temperature_factor_float32)
)
```

held bitwise. The factor is applied exactly once, after the gross ordinary
line deposit.

### P1 — Activity, loop/batched equality, and reach: PASS

The exact activity masks are:

```text
hot_dwarf          1 1 1 0 0 0
solar_dwarf        1 1 1 1 1 1
low_gravity_giant  1 1 1 1 1 1
cool_molecule_rich 1 1 1 1 1 1
```

Loop and batched gross/net slabs were bitwise equal. The per-depth reaches
were:

```text
hot_dwarf           6   6   5   0   0   0
solar_dwarf        36  28  24  40  80 114
low_gravity_giant  16  19  22  19  20  32
cool_molecule_rich 15  21  30  73 163  73
```

For every active depth, `nonzero_count == 2*reach + 1`. An independent
profile/cutoff reconstruction found the contribution at `reach-1` at or
above the cutoff and the contribution at `reach` below it for every active
depth, confirming the first-below-cutoff inclusion. The focused controlled
case also reaches exactly `MAX_WING_PROFILE_STEPS`.

### P1 — Defensive record and Doppler ownership: PASS

The private transformed record is cached with read-only arrays. Public calls
return defensive copies. Mutating one returned oscillator-strength array did
not alter a later record or transition checkpoint.

The former ambiguous Doppler checkpoint is now
`stored_doppler_checkpoint`, with:

```text
width_source = "supplied Chapter 3 fractional_doppler_widths"
```

It verifies
\(\Delta\lambda_{\rm D}=\lambda_l\delta_l\) and
\(\Delta v_{\rm D}=c\delta_l\), while the existing Chapter 3 synthesis width
producer supplies the temperature/microturbulence perturbation. Increasing
microturbulence increases the stored width and lowers the damping ratio;
doubling the supplied width halves the core amplitude while preserving the
frequency-integrated analytic strength.

The test-only source for that perturbation is now explicit:

```text
data/static/synthesis_tables/atomic_masses.npz
file bytes / SHA-256:
1076 / d4739fef7e03964aea5a7b2604f9585fd9095c26c58f5b7d5d040aaafeb5d117
atomic_mass_amu shape / dtype / C-byte SHA-256:
(99,) / float64 / f25d343b8e73c53f67078b90d10a60f51906ea6db976c5f72003963f555a5d89
```

Passing that manifest-bound local path to `load_atomic_masses` removes
environment and import-order ownership from the test without changing the
runtime or the physical calculation. The six solar-dwarf Fe widths remain
strictly larger when microturbulence is doubled.

### P1 — Profile semantics and fail-closed normalization: PASS

The neutral field name `profile_h` is used for all three declared profile
authorities. The normalization checkpoint distinguishes:

```text
integral H(a,u) du = sqrt(pi)
integral phi_nu dnu = 1
```

and rejects integration limits at or below 12 Doppler widths. The dense
wavelength-grid values remain samples of \(\kappa_\nu(c/\lambda_i)\), with no
invented wavelength Jacobian.

## 5. Staged-source identity evidence

The following staged files are byte-identical to the corresponding files in
the pinned external Payne Zero checkout:

| source | SHA-256 |
| --- | --- |
| `src/payne_zero_atmosphere/line_profile_math.py` | `9a5794140f00ff3c3fb6c2e3b28461bbc22b471f962d275055c066ad7f8acd15` |
| `src/payne_zero_atmosphere/line_opacity.py` | `d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6` |
| `src/payne_zero_synthesis/atomic_lines.py` | `0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0` |
| `src/payne_zero_synthesis/line_opacity.py` | `639b95c3812f1a7d227b797fa89a4d6ef9725d5f0e1284f3d49cf86844278275` |

Runtime module paths now resolve under the textbook's staged `src/` root even
when the grid helper is the first source-using call.
`configure_local_data_paths` still rejects an already-loaded external module.

## 6. Acceptance scope and remaining integration gates

The runtime core is **ACCEPTED for notebook authoring** at the immutable
runtime/test hashes in Section 1. The focused 15-test suite, independent
fresh-interpreter probe, independent exact/one-sided FASTEX and Harris seam
matrix, all 24 production-center reconstructions, loop/batched comparisons,
and formatting/lint checks are green.

Final Chapter 6 publication remains separately blocked on the declared
80-depth atmosphere fixture/oracle, lane-specific accepted goldens, backend
measurements, notebook/render audits, and atomic publication review. Those
integration gates are not defects in this accepted runtime core.
