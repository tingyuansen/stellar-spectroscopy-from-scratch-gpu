# Chapter 6 synthesis backend-measurement candidate

Status: implementation candidate; explicit backend tolerances remain
unauthorized until independent review

## 1. Scope and identities

This candidate measures the already-accepted one-record synthesis construction
on every backend available on the present host. It does not create an alternate
golden. CPU remains the only numerical authority.

| object | identity |
| --- | --- |
| CPU comparison golden | `data/golden/payne_zero/chapter06/synthesis/chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz` |
| golden SHA-256 | `a4b1ffa44bf22d05bf4a680cb8e5d600f09f15ab4544cf4e6fa801641bc0a955` |
| measurement script | `scripts/measure_chapter06_synthesis_backends.py` |
| script SHA-256 | `2cc9671060cf82bf6b10e3ee42965253da5f689237b509c582f37177bc6fa546` |
| focused tests | `tests/test_chapter06_backend_measurement.py` |
| test SHA-256 | `d48ff8301eeaa40cbf5752ee908476067e08d780292fcda4a2fb396f3e6f2e07` |
| raw canonical JSON SHA-256 | `0a7c10d9106e76882d857cb386fcf11fba7f0c3b73bcac10bdcbafd833cea4ce` |
| raw canonical JSON bytes | `16,053` |

The raw JSON was generated in a disposable temporary path. It is reproducible
by running the hash-bound script; it is not a new repository data product.
The command opens the golden only after each reader-built result has been
computed.

## 2. Measurement environment

```text
Python       3.13.9
NumPy        2.3.5
Torch        2.11.0
host         macOS 26.5.2, arm64
OMP          1
OpenBLAS     1
MKL          1
Numba        1
```

Backend inventory:

| backend | status | device | architecture / runtime |
| --- | --- | --- | --- |
| CPU | measured | arm | arm64, macOS 26.5.2 |
| CUDA | unavailable | unavailable | unavailable |
| MPS | measured | Apple M4, 10 GPU cores | arm64, Metal family 4, macOS 26.5.2 |

“Unavailable” is not a pass and contributes no CUDA parity evidence.

## 3. Measurement definition

Each available backend ran the same four regimes twice:

1. hot dwarf;
2. solar dwarf;
3. low-gravity giant;
4. cool molecule-rich atmosphere.

Each capture contains gross and once-stimulated net arrays with axes
`(regime=4, depth=6, wavelength=6000)`. The report measures:

- maximum absolute error;
- maximum relative error over the union of nonzero cells, using
  `abs(observed-reference) / max(abs(observed), abs(reference))`;
- maximum float32 ULP distance where both values are finite and have the same
  sign;
- zero-pattern disagreements;
- activity-mask disagreements;
- wing-reach disagreements;
- maximum center-opacity absolute error;
- frequency-integrated profile absolute and relative errors;
- exact reconstruction of net opacity with the production Torch
  stimulated-emission arithmetic;
- repeat-to-repeat variability.

The frequency diagnostic sorts the decreasing
\(\nu=c/\lambda\) coordinate before trapezoidal integration. It is a sampled
diagnostic, not a claim that the cutoff deposit is analytically normalized.

## 4. CPU authority replay

Both CPU repeats were byte-exact against the accepted CPU arrays.

| metric | gross A/B | net A/B |
| --- | ---: | ---: |
| maximum absolute error | `0 / 0` | `0 / 0` |
| maximum relative error | `0 / 0` | `0 / 0` |
| maximum ULP distance | `0 / 0` | `0 / 0` |
| zero-pattern disagreements | `0 / 0` | `0 / 0` |
| maximum center absolute error | `0 / 0` | `0 / 0` |
| maximum frequency-integral absolute error | `0 / 0` | `0 / 0` |
| maximum frequency-integral relative error | `0 / 0` | `0 / 0` |

Both repeats also had:

```text
activity-mask disagreements        0
wing-reach disagreements           0
stimulation-identity max abs       0
gross A-versus-B max abs / ULP     0 / 0
net A-versus-B max abs / ULP       0 / 0
```

CPU used work `torch.float64` and accumulation `torch.float32`.

## 5. Apple M4 MPS observation

The two MPS repeats were identical to each other. Their errors relative to the
CPU authority were also identical:

| metric | gross A | gross B | net A | net B |
| --- | ---: | ---: | ---: | ---: |
| maximum absolute error | `1.430511474609375e-06` | same | `1.430511474609375e-06` | same |
| maximum relative error, union nonzero | `5.624739223027772e-07` | same | `5.680594200252181e-07` | same |
| maximum float32 ULP distance | `8` | same | `9` | same |
| zero-pattern disagreements | `0` | `0` | `0` | `0` |
| maximum center absolute error | `0` | `0` | `0` | `0` |
| maximum frequency-integral absolute error | `30898.639221191406` | same | `29503.630004882812` | same |
| maximum frequency-integral relative error | `2.0651056584025745e-07` | same | `2.053406650529883e-07` | same |

Both repeats had:

```text
activity-mask disagreements        0
wing-reach disagreements           0
stimulation-identity max abs       0
gross A-versus-B max abs / ULP     0 / 0
net A-versus-B max abs / ULP       0 / 0
```

Every MPS regime produced:

- shape `(6, 6000)`;
- work `torch.float32`;
- accumulation `torch.float32`;
- device `mps:0`;
- one ordinary metal record;
- zero autoionizing records;
- zero helium records;
- population ion-stage index `0`;
- population element index `25`;
- finite, nonnegative gross and net output.

The activity counts and exact reach arrays matched the CPU authority:

```text
active depths
  [3, 6, 6, 6]

hot dwarf reach
  [6, 6, 5, 0, 0, 0]

solar dwarf reach
  [36, 28, 24, 40, 80, 114]

low-gravity giant reach
  [16, 19, 22, 19, 20, 32]

cool molecule-rich reach
  [15, 21, 30, 73, 163, 73]
```

The nonzero counts likewise matched:

```text
hot dwarf
  [13, 13, 11, 0, 0, 0]

solar dwarf
  [73, 57, 49, 81, 161, 229]

low-gravity giant
  [33, 39, 45, 39, 41, 65]

cool molecule-rich
  [31, 43, 61, 147, 327, 147]
```

## 6. Candidate interpretation

The observation supports three claims only:

1. on this Apple M4 and software stack, the MPS float32-work route preserved
   the exact branch structure, sparse support, record mapping, output
   lifecycle, and finite/nonnegative contract;
2. the observed MPS numerical difference from the CPU float64-work authority
   was at most 9 float32 ULPs and \(5.681\times10^{-7}\) relative over the
   union of nonzero cells;
3. both MPS repeats were identical, so the measured repeat variability was
   zero in this two-repeat sample.

This does not establish a universal MPS tolerance, CUDA behavior, speedup, or
cross-backend bitwise equality.

## 7. Required independent review

Before the chapter may turn this observation into an executable parity gate,
an independent audit must:

1. re-read the measurement code for pre-golden computation and zero writes;
2. recompute the JSON on the available host;
3. verify every metric definition, especially ULP ordering, union-nonzero
   relative error, frequency-axis orientation, and exact stimulation
   reconstruction;
4. verify the four-regime structural arrays;
5. freeze separate explicit MPS gross and net `atol`/`rtol` values no smaller
   than repeat variability and no broader than the observed need;
6. retain CUDA as unavailable rather than passed;
7. add a regression test that separates structural equality from numerical
   tolerance.

No tolerance is authorized by this candidate report alone.
