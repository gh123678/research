# FP-ATTN-8X-001 first result: the network catches up, and we learn where it stops being trustworthy

Date: 2026-09-12.
Branch: `claude/FP-CENSUS-001`. Baseline: `a3e5a2b`.
Actor: Claude, under the user's instruction of 2026-09-12 ("好的").

## 1. What was run

```text
python -B evaluate_fp_attn_8x_001.py --tasks 12 --mixings 0.08,0.5 \
    --max-steps 12 --multiplier 8 --label network-at-8x \
    --output-dir results/FP-ATTN-8X-001/claude/formal
python -B analyze_fp_attn_8x_001.py
```

The literal attention network at `8x` certification (`131072 x 64 = 8,388,608`
items), twelve steps, both certificate arms, compared step by step against
`FP-ITER8X-001`'s numpy bundle. Torch `2.11.0+cpu`.

The network was the last component still on `1x`. Every result since
`FP-ATTN-ITER6-001` — the certificate repairs, the size ladder, the twelve- and
thirty-two-step iterations, the value measurement — had been numpy only.

## 2. The network reproduces numpy exactly

| step | `frozen` network | numpy | `empB` network | numpy | disagreements |
|---|---:|---:|---:|---:|---:|
| 1 | `42` | `42` | `44` | `44` | **`0`** |
| 4 | `40` | `40` | `42` | `42` | **`0`** |
| 8 | `29` | `29` | `34` | `34` | **`0`** |
| 12 | `22` | `22` | `26` | `26` | **`0`** |

**Zero set disagreements at every one of the twelve steps, in both arms**, on emitting
sets, not merely counts. `885` step entries carry `0` decision flips and `0` `eta`
flips against the same-process numpy comparator.

Both paths also reach the full horizon (`12` steps) with identical emission sequences:

```text
frozen : [42, 42, 40, 40, 38, 36, 33, 29, 28, 27, 25, 22]   network == numpy
empB   : [44, 44, 44, 42, 41, 38, 35, 34, 31, 29, 27, 26]   network == numpy
```

So the project's headline claim now rests on the network at `8x` over twelve steps,
not just on numpy.

## 3. Validity and soundness on the network path

| check | `frozen` | `empB` |
|---|---:|---:|
| emitted steps | `402` | `435` |
| componentwise degrading | `0` | `0` |
| non-positive gains | `0` | `0` |
| certificate violations | `0` | `0` |

`837` certified updates on the literal network, all valid. `H1` also holds
structurally: the two call sites whose results are bound to `certificate` and
`decision` take `q_literal` and never `q_numpy`, and every step's gap is strictly
positive (minimum `5.858e-08`), so no step's `Qhat` was silently replaced.

## 4. Drift does not accumulate over twelve steps

Worst `|dQ|` over all steps and arms: **`1.373e-05`**, well inside `ATOL = 1e-4`.
Per step (frozen): `1.08e-05, 6.27e-06, 7.18e-06, 1.37e-05, 1.00e-05, 9.25e-06,
9.66e-06, 8.05e-06, 6.48e-06, 9.77e-06, 4.57e-06, 1.17e-05` — flat, non-monotone, no
accumulation at five times the previous maximum network horizon.

## 5. The finding that matters: the network has a trust horizon

`H7` was registered as a floor (`≥ 20x` headroom) and it passes at `40.0x`. But the
number worth reading is the *trend*:

| step | min emitted gain | max `\|dQ\|` | headroom |
|---|---:|---:|---:|
| 1 | `6.910e-01` | `1.08e-05` | `64,221x` |
| 6 | `5.169e-02` | `9.25e-06` | `5,587x` |
| 10 | `3.738e-03` | `9.77e-06` | `383x` |
| 11 | `1.830e-04` | `4.57e-06` | **`40x`** |
| 12 | `6.487e-04` | `1.17e-05` | `56x` |

The margin shrinks geometrically while the `float32` gap stays flat at `~1.4e-05`.
Extrapolating that against `FP-HORIZON-001`'s numpy margin sequence, **post-hoc and
labelled as a projection**:

```text
step 16   min gain 1.947e-05   headroom  1.4x   decided by the certificate
step 17   min gain 8.105e-06   headroom  0.6x   decided by ARITHMETIC
...
step 25   min gain 1.513e-05   headroom  1.1x   back above 1 (the emitting set changed)
step 26   min gain 8.885e-06   headroom  0.6x   decided by ARITHMETIC
```

**The network can be trusted to reproduce numpy's eligibility decisions to about step
`17`.** Beyond that the certified improvements are smaller than the network's own
arithmetic resolution, so whether a record is eligible is decided by `float32`
rounding rather than by the certificate. Steps `21`–`25` recover above `1x` because
the emitting set changes and a larger margin reappears; from step `26` onward the
decisions are arithmetic-decided throughout.

This is a **limit on the network path specifically**, and it is not the same as the
iteration's reach. `FP-HORIZON-001` showed the numpy iteration still going at step
`32` on margins of `3.8e-7`; at that margin the network could not be trusted to make
the same decision at all.

The projection rests on two things: the numpy margin sequence out to `32` steps, and
a `float32` gap held fixed at this run's worst. Neither is a measurement of the
network beyond step `12`, and the report says so.

## 6. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` provenance | **PASS** (structural, plus `885/885` flagged entries) |
| `H2` validity and soundness | **PASS** (`837` emitted steps, `0` violations) |
| `H3` network reaches twelve steps | **PASS** |
| `H4` set agreement at every step | **PASS** (**`0`** disagreements) |
| `H5` zero `eta` flips | **PASS** |
| `H6` drift within `ATOL` | **PASS** (`1.373e-05`) |
| `H7` headroom above `20x` | **PASS** (`40.0x`) |

Construction checks **PASS**.

## 7. Where the line now stands

| result | path | certification | horizon |
|---|---|---|---|
| `FP-ATTN-ITER6-001` | network | `1x` | `6` |
| **`FP-ATTN-8X-001`** | **network** | **`8x`** | **`12`** |
| `FP-ITER8X-001` | numpy | `8x` | `12` |
| `FP-HORIZON-001` | numpy | `8x` | `32` |

The network is no longer behind on the tested configuration; what remains ahead of it
is the `32`-step horizon, and the trust-horizon projection says extending the network
there would be measuring arithmetic rather than the certificate.

## 8. What this does not establish

- **Nothing about the network beyond step `12`.** The trust horizon is a projection.
- **That the network's *values* match numpy's**, only its *decisions* and emitting
  sets. The `float32` gap of `~1.4e-05` is a real difference in the estimates; it
  simply does not change any decision within twelve steps.
- **That the certificate repairs help on the network path.** Both arms were run and
  both agree with numpy, but this task compares paths, not arms; the arm comparison
  at `8x` was done on numpy in `FP-SAMPLE-001` and `FP-ITER8X-001`.
- **Size versus realisation.** The `8x` batch is a fresh independent sample.
- **Same-actor throughout**, per the user's standing instruction.
