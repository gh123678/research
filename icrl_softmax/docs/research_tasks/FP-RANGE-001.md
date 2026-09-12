# FP-RANGE-001: is the envelope ceiling reachable soundly? (a negative result)

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12 ("都做"), following
  the two levers `FP-TIGHT-001` priced.
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `7c0123f` (`claude/FP-CENSUS-001`).
- Execution branch: `claude/FP-CENSUS-001`.
- Result directory: `results/FP-RANGE-001/claude/`.
- Classification: long, conclusion-critical, single-actor under the standing user
  exception.

## Why this task exists

`FP-TIGHT-001` priced four levers on `E_Q` and found that the biggest measured one
was **deleting the envelope** (`−45%`), which is not a certificate at all: with the
range gone nothing bounds the tails. It labelled the arm unsound and excluded it
from the coverage audit. `FP-TIGHT-001` then predicted (`H6`) that correcting the
mean-step inequality would beat deleting the envelope — **and was wrong**, by
`−20%` against `−45%`. The census's reading, that `87–95%` of `s_x²` is the
worst-case assumption, was the right guide.

So the range looked like the remaining prize, and `FP-TIGHT-001` recorded it as
"priced, not delivered". This task asks whether it can be delivered **soundly**.

The natural construction is **truncation**: cap `|Y|` at a threshold `tau`, apply
Bernstein to the capped variable with range `2 tau`, and pay for the cap with a
tail bound. The price is explicit:

```text
|E[Y]| <= |mean_B(Y)|                     observed
        + (1/N) sum_{|Y_i|>tau} |Y_i|     the empirical tail mass, exact
        + sqrt(V_x * p)                   Cauchy-Schwarz, p = P(|Y| > tau)
        + bernstein(Var_B(Y'), 2 tau)     concentration of the capped variable
```

The middle two terms are **exactly the price of honesty**, and this task measures
whether the price leaves anything worth having.

## Research question

Can the envelope's `−45%` be realised by a certificate that actually bounds the
tails, and if not, at what sample size does truncation become worth its price?

## Falsifiable hypotheses

1. `H1 (soundness, mandatory)`: the data-range certificate covers the realized
   oracle error on all `48` route-records at `1x` — zero violations.
2. `H2 (pre-registered: the range lever loses to the inequality lever)`: the sound
   data-range repair's mean-`E_Q` reduction is **less negative** than the
   empirical-Bernstein repair's. Anchored on the pilot: `+10.2%` against `−16.7%`.
3. `H3 (pre-registered magnitude at 1x)`: the data-range reduction lies in
   `[+5%, +25%]`, i.e. it is **worse than the frozen certificate**, not better.
4. `H4 (the mechanism, pre-registered)`: the loss is carried by the tail terms, not
   by the concentration term. Specifically, at the binding pair the sum of the
   empirical tail mass and the Cauchy-Schwarz bias `sqrt(V_x p)` **exceeds** the
   concentration term `bernstein(Var_B(Y'), 2 tau)`. Reported per record.
5. `H5 (pre-registered: the bias alone exceeds the frozen radius)`: at `1x`,
   `sqrt(V_x p)` at the binding pair alone is larger than the frozen certificate's
   whole radius on that record. If true, no re-weighting of the decomposition can
   rescue the lever at this sample size.
6. `H6 (pre-registered: more data does not rescue it)`: at `8x`, the data-range
   repair improves **but still does not beat** the empirical-Bernstein repair at
   `8x`. The bias decays as `1/sqrt(N)`, the same rate as the term truncation was
   meant to remove, so the ordering is predicted to be stable across the ladder.

`H1` is mandatory. `H2`--`H6` are the substantive pre-registered predictions; a
falsified one is a result and is never reinterpreted.

### What a negative result is worth here

If `H3`, `H5` and `H6` all hold, the conclusion is that the `−45%` ceiling was an
**artifact of not paying for the tails**, and the largest sound lever is sample size
— which `FP-SAMPLE-001` independently established is worth `−58%` at a price
(`8x` certification). That is a decision-grade result: it stops effort being spent
on a number that cannot be collected.

## Frozen contract

### Protocol

Inherited verbatim; certification rungs `1x, 2x, 4x, 8x` using the vectorised
sampler validated by `FP-SAMPLE-001` (`H1` gate: worst TV `0.0091`).

### Scope: step 1 only

All `48` route-records at step 1 of the frozen `pi_0`. The iteration is not run.

### The change

One new function in the existing `fixed_policy_bernstein_certificate.py`,
`data_range_certificate`, plus a new evaluator. Sealed modules untouched.

### Two design choices that must be justified, not assumed

**The threshold comes from the other half.** `tau = max|Y|` over half A. Because
half A is independent of half B, every statement about half B holds
*conditionally on tau* with no union bound over a grid of candidate thresholds.
Optimising `tau` on half B would require such a union and would cost more than it
buys.

**The tail bound is the tight one.** The tail probability enters the bias under a
square root, so its bound dominates. Hoeffding on the indicator costs
`sqrt(log(1/δ)/(2N)) ≈ 0.014` at `N = 16369`, which alone makes the bias `0.15` —
three times the whole frozen radius. The Maurer-Pontil form's range term is
`(7/3)log(2/δ)/(N−1) ≈ 9.4e-4`, far tighter here because the indicator's observed
variance is essentially zero. The tighter bound is used **so that the negative
result is not an artifact of a weak choice**: the pilot measured `+155%` with
Hoeffding and `+10.2%` with Maurer-Pontil, and the honest conclusion has to rest on
the better one.

## Prohibited work

- No modification of any sealed module, sealed bundle, or closed task record.
- No change to any frozen formula, constant, tolerance or hypothesis after the run.
- No use of the unsound envelope deletion as if it were achievable; it appears only
  as a labelled ceiling.
- No re-tuning of `tau` on half B, and no union-bound-free claim that would require
  it.
- No reinterpretation of a falsified prediction.
- No `git add -A`.

## Acceptance criteria

1. `H1` coverage with zero violations, reported per record.
2. `H2`--`H6` each evaluated with evidence.
3. The price decomposition reported per record: empirical tail mass, Cauchy-Schwarz
   bias, concentration, and the fitted `tau`.
4. The ladder at `1x, 2x, 4x, 8x` with item counts.
5. Sealed module hashes verified unchanged.
6. All strict-JSON, finite, shape, seed and location checks pass.
7. Complete reproducibility evidence.
8. Same-actor derived verification recorded.
9. `ACTIVE_WORKSPACE.md` updated.

## Failure criteria

The construction fails if `H1` shows a coverage violation, if a sealed module
changes, or if the truncation's tail price is omitted from a reported number.

`H2`--`H6` failing is **not** a construction failure. If `H3` is falsified — if the
sound range repair actually beats the frozen certificate — the range lever is alive
and the programme should pursue it.

## Stopping conditions

Stop and report if:

- a coverage violation appears, since the construction would then not be sound;
- a sealed module's hash is found changed;
- execution would expand cost beyond the authorised scope.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-12), same-actor.
- Evidence: `docs/research_branches/FP-RANGE-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-12.
- Decision: "都做" — execute both levers identified by `FP-TIGHT-001`.
- Scope: this task and `FP-SAMPLE-001`, on `claude/FP-CENSUS-001`.

## Definition of done

- [x] `H1` coverage confirmed.
- [x] `H2`--`H6` each evaluated with evidence.
- [x] The price decomposition reported per record.
- [x] The ladder reported with item counts.
- [x] Sealed module hashes verified unchanged.
- [x] Same-actor derived verification recorded.
- [x] `ACTIVE_WORKSPACE.md` updated.

## Formal outcome (2026-09-12)

Route journal: `docs/research_branches/FP-RANGE-001/claude/first_result.md`.

| rung | items | `data_range` | vs frozen | `empirical_bernstein` | vs frozen |
|---|---:|---:|---:|---:|---:|
| `1x` | `1,048,576` | `0.2587` | **`+6.9%`** | `0.1955` | `−19.2%` |
| `2x` | `2,097,152` | `0.1827` | `−24.5%` | `0.1339` | `−44.7%` |
| `4x` | `4,194,304` | `0.1369` | `−43.4%` | `0.1018` | `−57.9%` |
| `8x` | `8,388,608` | `0.1055` | `−56.4%` | `0.0821` | `−66.1%` |

Same-sample references at `1x`: frozen `0.2420`, unsound ceiling `0.1356`
(`−44.0%`).

| hypothesis | verdict |
|---|---|
| `H1` soundness | **PASS** (`0/192` certificate-fits) |
| `H2` the range lever loses to the inequality lever | **PASS** |
| `H3` `1x` reduction in `[+5%, +25%]` | **PASS** (`+6.9%`) |
| `H4` the tail price exceeds the concentration term | **PASS** (`48/48`) |
| `H5` the bias alone exceeds the frozen radius | **FALSIFIED** (`0/48`, mean ratio `0.735`) |
| `H6` the ordering is stable at `8x` | **PASS** |

- **The `−45%` ceiling was an artifact of not paying for the tails.** A sound
  truncation certificate costs `+6.9%` at `1x` — **worse than the frozen
  certificate** — and never overtakes the plain inequality repair at any rung,
  including `8x`.
- The certificate **is** sound (`0` coverage violations over `192` fits). It is
  simply not an improvement, and the two facts are reported separately.
- **The mechanism is now exact**: the empirical tail mass is **exactly zero on all
  `48` route-records** (because `tau` is taken from the independent half), so the
  entire price lands in the Cauchy-Schwarz bias `sqrt(V_x p)` ≈ `0.0266`–`0.0542`,
  which is comparable to the frozen radius (`0.0485`) rather than larger. `H5` had
  predicted the bias would *exceed* the radius; it is falsified, and the loss is
  additive rather than dominated by one term.
- **The negative result does not rest on a weak choice.** The pilot measured `+155%`
  with a Hoeffding tail bound and `+10.2%` with Maurer-Pontil; the tighter bound is
  used and both numbers are recorded, so the conclusion is about the lever rather
  than about a badly implemented bound.
- **Closing the accounting on this line**:

  | lever | sound? | worth |
  |---|---|---|
  | correct the mean-step inequality | yes | `−15%` |
  | empirical Bernstein | yes | `−20%` |
  | delete the envelope | **no** | `−45%`, unreachable per this task |
  | `8x` certification data | yes | `−58%`, and `22` of `26` abstainers revived |

  Sample size is the only large lever that is both sound and collectable.
- **Not established**: nothing about the iteration (step 1 only), and the result is
  about **this** truncation construction — a different tail treatment is not ruled
  out by it.
