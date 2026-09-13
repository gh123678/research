# FP-ITER8X-001 first result: the iteration reaches twelve steps

Date: 2026-09-12.
Branch: `claude/FP-CENSUS-001`. Baseline: `d68d940`.
Actor: Claude, under the user's instruction of 2026-09-12 ("好的").

## 1. What was run

```text
python -B evaluate_fp_iter8x_001.py --tasks 12 --mixings 0.08,0.5 \
    --max-steps 12 --multiplier 8 --label iteration-at-8x \
    --output-dir results/FP-ITER8X-001/claude/formal
python -B analyze_fp_iter8x_001.py
```

Two arms, each with its own trajectory, on `8x` certification
(`131072 x 64 = 8,388,608` items) and a horizon of `12`:

- `frozen` — the sealed certificate on the larger sample, so **the data** is the only
  change;
- `empirical_bernstein` — the repair plus the data, so both levers act.

The `frozen` arm is the control. Without it a longer iteration could not be
attributed to more data rather than to a better bound, and those are exactly the two
explanations this line has been separating.

## 2. The headline: the horizon was reached, not exhausted

| step | `frozen@8x` | mean gain | min gain | `empB@8x` | mean gain | min gain | sealed `1x` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `42` | `2.084239` | `0.690987` | `44` | `2.013258` | `0.521629` | `22` |
| 2 | `42` | `1.652467` | `0.382176` | `44` | `1.608818` | `0.503701` | `20` |
| 3 | `40` | `1.084603` | `0.335048` | `44` | `1.031726` | `0.252349` | `15` |
| 4 | `40` | `0.666411` | `0.184902` | `42` | `0.654473` | `0.184902` | `12` |
| 5 | `38` | `0.428729` | `0.094834` | `41` | `0.405318` | `0.027370` | `12` |
| 6 | `36` | `0.269395` | `0.051690` | `38` | `0.267477` | `0.051690` | `9` |
| 7 | `33` | `0.184044` | `0.029867` | `35` | `0.176311` | `0.012192` | — |
| 8 | `29` | `0.125775` | `0.005956` | `34` | `0.120377` | `0.018112` | — |
| 9 | `28` | `0.090134` | `0.003628` | `31` | `0.087358` | `0.008948` | — |
| 10 | `27` | `0.065885` | `0.003738` | `29` | `0.064424` | `0.003738` | — |
| 11 | `25` | `0.047963` | `0.000183` | `27` | `0.049179` | `0.001558` | — |
| **12** | **`22`** | `0.035123` | `0.000649` | **`26`** | `0.036101` | `0.000649` | — |

**Both arms use all twelve steps and are still emitting at step 12** — `22` and `26`
of `48`. The sealed `1x` run stopped at six because six was its horizon; this one
stopped at twelve because twelve is *this* horizon. So the certified iteration has
**not been shown to stop at all**, and its true ceiling is beyond `12` and
unmeasured.

`H3` **PASS** (`33` and `35` route-records emit a seventh step). `H4` **PASS**: `33`
emit at step `7` against `9` at step `6` under `1x`.

## 3. Nothing is lost at any shared step

Against the sealed `1x` run, at every shared step:

| step | sealed `1x` | `frozen@8x` | gained | **lost** |
|---|---:|---:|---:|---:|
| 1 | `22` | `42` | `20` | **`0`** |
| 2 | `20` | `42` | `22` | **`0`** |
| 3 | `15` | `40` | `25` | **`0`** |
| 4 | `12` | `40` | `28` | **`0`** |
| 5 | `12` | `38` | `26` | **`0`** |
| 6 | `9` | `36` | `27` | **`0`** |

The larger sample does not trade records for reach: it adds `20`–`28` and drops
none. At step 1 alone, eligibility goes from `22/48` to `42/48`.

## 4. `H7` FALSIFIED, in the direction that favours the lever

I registered `H7` as "the step-6 minimum gain under `8x` is **lower** than the sealed
`0.019347`", reasoning that a tighter bound admits more records and a larger set
includes marginal ones, so the minimum should fall.

It rose instead: **`0.051690`**, more than `2.6x` the sealed value, while the
emitting set grew from `9` to `36`. So more data **widened the population and raised
the worst-case margin at the same time**. The intuition that a larger population must
degrade the minimum was wrong: the marginal records admitted by the tighter bound are
not the ones dragging the minimum down, because the whole trajectory is different.

This is the prediction I deliberately registered against the lever, and it failed
against the lever's disadvantage. Recorded as falsified, not reinterpreted.

`H6` **PASS**: the mean gain decays at every step beyond six, on both arms.

## 5. `H5` FALSIFIED, and the registration was wrong, not the implementation

I registered `H5` as "at every step, the repaired arm's emitting set contains the
frozen arm's", and justified it as a **theorem**: `LB_s = Î_s − E_Q‖Δπ_s‖₁` is
increasing in `E_Q`, so a smaller `E_Q` cannot remove eligibility.

**That justification is wrong across arms.** The monotonicity holds at a **fixed**
policy; the two arms carry different trajectories, so their step-`k` policies differ
and there is no theorem relating their emitting sets. `H5` failed at step `11`
(one route-record, `0.08/2 expected_finite`), which is exactly the expected
consequence of a mis-specified claim rather than a bug.

The claim **is** testable where the theorem applies — step `1`, where both arms start
from `π_0` with the same `q_hat` — and there it holds: `42 ⊆ 44`. The analyzer now
checks that form explicitly and reports the cross-arm count as an observation.

Cross-arm containment holds at `11/12` non-empty levels. That number is reported, and
it is not presented as a guarantee.

## 6. Validity holds at every step of both arms

| check | `frozen@8x` | `empB@8x` |
|---|---:|---:|
| emitted steps | `402` | `435` |
| componentwise degrading | `0` | `0` |
| non-positive total gains | `0` | `0` |
| certificate violations | `0` | `0` |
| abstentions without a frozen reason | `0` | `0` |

`H1`/`H2` **PASS**. `841` certified updates across the two arms, none degrading, none
violating the bound.

**But the tail is margin-starved.** The minimum gains at steps `11` and `12` are
`0.000183` and `0.000649` — three to four orders of magnitude below the `0.05` floor
that steps 4 and 5 cleared under `1x`. They are strictly positive and the
componentwise check passes, so the updates are valid; they are nonetheless running on
numerical dust. Any claim that the iteration "reaches 12" should carry that.

## 7. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` validity | **PASS** (`841` emitted steps, `0` degrading) |
| `H2` soundness | **PASS** (`0` certificate violations) |
| `H3` the iteration passes six steps | **PASS** (`33`/`35` seventh-step emissions) |
| `H4` step-7 population exceeds `1x`'s step 6 | **PASS** (`33` vs `9`) |
| `H5` repaired arm contains frozen arm | **FALSIFIED** (mis-registered; theorem holds at step 1) |
| `H6` mean gain keeps decaying past six | **PASS** |
| `H7` step-6 minimum is lower than `1x` | **FALSIFIED** (it is higher: `0.051690`) |

Construction checks **PASS**.

## 8. What this establishes, and what it does not

**Established**

- **The certified iteration reaches twelve steps on both arms** with `8x`
  certification, still emitting at the horizon. Combined with the record across the
  line, the certified horizon has gone `2 → 3 → 4 → 5 → 6 → **12**`.
- **The reach is attributable to the data**, not only to the repair: the `frozen` arm
  — the sealed certificate on the larger sample — also reaches twelve.
- Nothing is lost at any shared step; the larger sample adds `20`–`28` records per
  step and removes none.
- The worst-case margin at step 6 **improves** (`0.019347 → 0.051690`) while the
  population grows `9 → 36`.
- Validity and soundness hold at every one of `841` emitted updates.

**Not established**

- **Where it actually stops.** Twelve was the horizon and both arms used it. The true
  ceiling is beyond twelve and unmeasured; the `FP-ITER6-001` plateau at `9` was a
  horizon artifact in the same way, and this result does not rule out a similar or
  different shape further out.
- **That the tail margins are meaningful.** At `0.000183`, a step-11 emission is
  certified but numerically negligible. Whether such steps should count as
  "improvement" is a definitional question this task raises and does not answer.
- **Size versus realisation.** The `8x` batch is a fresh independent sample, not a
  superset of the sealed one, so the `1x → 8x` comparison crosses both. The
  `frozen@8x` arm bounds the certificate effect but not the sampler effect.
- **The network path is untested at `8x`.** The record here is numpy; the network is
  in sync only up to the `1x` six-step run.
- **Same-actor throughout**, per the user's standing instruction.
