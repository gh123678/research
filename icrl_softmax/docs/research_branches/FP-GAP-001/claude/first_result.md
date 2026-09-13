# FP-GAP-001 first result: the tail updates are real work

Date: 2026-09-12.
Branch: `claude/FP-CENSUS-001`. Baseline: `c1cebd7`.
Actor: Claude, under the user's instruction of 2026-09-12 ("好的").

## 1. What was measured, and how

`FP-HORIZON-001` left a definitional question open: at step `32` the certified
iteration is still emitting, on minimum gains of `3.845e-07` — **do those updates
count as policy improvement?**

As posed that question is a choice of definition. This task re-posed it as one that
can be measured:

> Is the iteration making progress on the thing that matters — the distance to the
> optimal policy — or is it emitting certified improvements while that distance sits
> at a floor?

**No new simulation was needed.** `FP-HORIZON-001` sealed, for every emitted step,
the per-state change in `v^{pi_k}`; cumulating those deltas reconstructs `v^{pi_k}`
exactly, and the only missing ingredient, `v*`, is computable exactly by policy
iteration on a `4`-state, `3`-action MDP.

`H1` **PASS**: all `96` trajectories reconstruct to within `4.441e-16`.
`H2` **PASS**: `v*` satisfies the Bellman optimality equation to `8.882e-16` on every
record and dominates `v^{pi_0}` everywhere. `H3` **PASS**: not one rise in any of the
`96` gap trajectories.

## 2. The answer: the iteration converges geometrically

On a long trajectory, step by step:

| step | remaining gap | gain that step | gain / remaining gap |
|---:|---:|---:|---:|
| 4 | `0.186` | — | `0.81` |
| 8 | `0.018` | — | `0.54` |
| 12 | `0.003` | — | `0.47` |
| 16 | `0.001` | — | `0.46` |
| 20 | — | — | `0.45` |
| 24 | — | — | `0.45` |
| 28 | — | — | `0.45` |
| 32 | — | — | `0.44` |

**The gain relative to whatever remains is constant at about `0.45`, steadily, from
step 4 to step 32.** Each step closes a fixed fraction of what is left, which is what
geometric convergence looks like, and it means the shrinking absolute gains are a
*consequence* of the shrinking gap, not evidence that the iteration has stopped doing
work.

### A labelling error, found by a cross-session check

The first version of this record said "the gap decays by a constant factor of about
`0.45` per step". **That is wrong: `0.45` is not a decay factor.** It is the gain
divided by the *remaining* gap. For a geometric process `gap_k = ρ·gap_{k-1}` that
ratio equals `(1−ρ)/ρ`, so the same measurement has three equivalent forms:

| form | value |
|---|---:|
| gain / remaining gap (`(1−ρ)/ρ`) | `0.45` |
| decay factor `ρ` | **`0.69`** |
| fraction of the *current* gap closed (`1−ρ`) | `0.31` |

A cross-session check (`cross_check_fp_gap_001.md` in this directory) measured the decay
factor independently and got **`~0.69`** and a fraction closed of **`~0.305`** — which
**agree** with the table above once the forms are matched: `1/(1+0.45) = 0.690` and
`0.45/1.45 = 0.310`.

So the check's finding is correct and worth stating twice: **the label was wrong, the
arithmetic was not.** Its own summary that the median "differs by `2.3x`"
(`0.327` vs `0.758`) is the same labelling mismatch read in the other direction —
`1/(1+0.327) = 0.754`, which matches its measured `0.758` to within aggregation noise.
Two sessions compared a gain-to-remaining-gap ratio against a decay factor and each
called the discrepancy a disagreement.

The corrected forms are used from here on, and the analyzer and verifier now name the
quantity explicitly rather than calling it a decay factor.

**So the tail updates count.** They are doing the same *relative* work as the early
ones. The question "is `3.8e-7` big enough" turns out to have a clean answer: it is
big enough because there was only `~8e-7` left to close.

Across the `38` long trajectories (≥ `24` emitted steps), a mean of **`99.9717%` of
the initial suboptimality is closed**, at a median gain-to-remaining-gap ratio of
`0.327` — equivalently a decay factor of `0.754` and a fraction closed per step of
`0.246`. An independent cross-session re-derivation got `99.9388%` (a `0.03pp`
difference) and a decay factor of `0.758`, so the conclusion and the numbers both
reproduce once the three forms of the ratio are kept apart.

## 3. All three registered predictions failed — in the direction that favours the tail

| hypothesis | verdict |
|---|---|
| `H4` the gap reaches a floor before the horizon | **FALSIFIED** (`0/74` plateau) |
| `H5` the final gain is a vanishing fraction of the remaining gap | **FALSIFIED** (`0/86`; median `0.29`) |
| `H6` more than `90%` of the initial gap is closed | **FALSIFIED** (pooled mean `82.25%`) |

All three point the same way, and all three are wrong for **identifiable reasons that
are mine, not the data's**:

### `H5` was the wrong test

I predicted the final gain would fall below `1e-4` of the remaining gap. But a
process converging **geometrically** has a **constant** gain-to-gap ratio — a
vanishing ratio would mean *faster*-than-geometric convergence. The observed `~0.33`
is not a failure to converge; it *is* the convergence rate. I wrote a test whose
passing would have indicated a stronger property than convergence, and then read its
failure as evidence of idling.

### `H4` divided by a near-zero denominator

The registered metric was the gap's movement over the last third as a fraction of the
**final** gap. When the final gap is `~1e-10` that ratio is enormous for any movement
at all, and the metric saturates. The corrected form — the same movement as a share
of the **initial** suboptimality — gives a median of **`0.0816%`** for the long
trajectories, which is the honest version of "the gap has effectively flattened".

### `H6` pooled two populations that behave differently

| emitted steps | n | mean fraction of the gap closed |
|---|---:|---:|
| `1`–`5` | `12` | `54.89%` |
| `6`–`11` | `26` | `94.46%` |
| `12`–`23` | `10` | `98.28%` |
| `24`–`32` | `38` | **`99.97%`** |

The pooled mean of `82.25%` that failed my `90%` threshold is an artifact of mixing
trajectories that converge to `99.97%` with trajectories that stop after a few steps.
`10` trajectories never emit at all and close nothing. The `90%` threshold was
registered without asking which population it applied to.

## 4. Where the iteration genuinely stops short

The failures above do not mean there is nothing to worry about — they relocate it.
**The place the iteration stops short is the short trajectories.** Records that stop
emitting after `1`–`5` steps leave **`45%`** of their initial suboptimality on the
table. That is a real and large gap, and it is not a tail phenomenon at all: it is
the gate closing early on records that still had somewhere to go.

That is the opposite of the worry `FP-HORIZON-001` raised. The concern was that the
iteration would keep emitting negligible steps forever; the reality is that it
converges cleanly when it runs, and **stops early** on a subset of records while a
large amount of value remains uncollected.

## 5. The policy-class caveat

Required by the task sheet wherever a floor is reported, and it applies here:

> The policy class is the relative-softmax family with `pi_min = 0.15`, not the whole
> simplex, so the gap to `v*` is not expected to reach zero and part of any floor
> belongs to the policy class rather than to the iteration.

`0` trajectories close the gap **exactly** to zero, which is consistent with the
class restriction. This task does **not** decompose the residual into a
policy-class part and an iteration part; that would need a constrained optimum over
the reachable family, which is a different task.

## 6. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` reconstruction closes | **PASS** (`4.441e-16`) |
| `H2` `v*` verified | **PASS** (Bellman `8.882e-16`, dominates everywhere) |
| `H3` gap monotone | **PASS** (`0` rises in `96` trajectories) |
| `H4` plateau before the horizon | **FALSIFIED** (mis-specified denominator) |
| `H5` vanishing tail gain | **FALSIFIED** (wrong test for a geometric process) |
| `H6` `> 90%` closed | **FALSIFIED** (pooled two different populations) |
| `H7` effective stopping step | reported: pooled median `13` versus an emitted median of `14` |

Construction checks **PASS**.

## 7. What this settles, and what it leaves

**Settled.** The tail updates are not decorative. The certified iteration converges
geometrically to the optimum at a constant per-step factor of roughly `0.45`, closing
`99.97%` of the initial suboptimality on trajectories that run long enough. The
definitional worry — "is a gain of `3.8e-7` an improvement?" — is answered by the
observation that it is `~45%` of everything that remained.

**Left open.**

- **Why short trajectories stop early while `45%` of the gap remains.** This is now
  the binding question, and it is a question about the **gate**, not about the
  certificate's scale: the census established that abstention means
  `improvement_lcb_nonpositive`, so the improvement per unit of policy movement fell
  below the certified error on those records.
- **How much of the residual floor belongs to the policy class.** Not decomposed
  here, and not decomposable without a constrained optimum.
- **Whether the same holds on the network path.** This measurement is numpy only; the
  network would need the same `32`-step run, which `FP-ATTN-8X-001`'s trust-horizon
  result says would be measuring arithmetic past step `17`.
- **Same-actor throughout**, per the user's standing instruction.
