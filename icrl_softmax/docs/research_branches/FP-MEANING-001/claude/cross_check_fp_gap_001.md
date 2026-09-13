# Cross-session check of FP-GAP-001

Date: 2026-09-13.
Checker: the DSH session working in the main tree. **Not the session that wrote
`FP-GAP-001`**, and not the same tooling — that is the point of the exercise.

## Why this exists, and what it is not

`FP-GAP-001` was executed and committed to this same branch by a different Claude Code
session while `FP-MEANING-001` was being written. It answers the same question from the
realized side, and `FP-MEANING-001` depends on its conclusion: if the tail updates are
real work, then the certificate's evaporation is a statement about the *guarantee*
rather than about the iteration.

So its headline numbers carry weight in a claim I am making. That is a reason to
re-derive them rather than take them.

**This is not independent verification.** Both sessions are the same model, so a shared
conceptual error would survive; only a different actor closes that gap. What it *is*:
a second implementation, written without reference to the first, over the same sealed
bundle. That catches coding and aggregation errors, which is what it found.

Nothing in `FP-GAP-001`'s directory was modified. Its author fixes its own work.

## Method

From `results/FP-HORIZON-001/claude/formal/task_results.json`, for each of the 96
trajectories (48 route-records x 2 arms):

1. reconstruct `v^{pi_k}` by cumulating the sealed per-step `value_delta_vs_previous`
   on top of the recorded `start_policy_value`, and check the cumulation against the
   recorded `final_policy_value`;
2. compute `v*` by exact policy iteration on the `4x3` MDP and check the Bellman
   optimality residual;
3. recompute the gap sequence `||v* - v^{pi_k}||`, the per-step decay factor
   `gap_{k+1}/gap_k`, and the fraction of the initial suboptimality closed.

Both the max-norm and the sum-norm are used, because the norm choice changes the ratio
whenever the binding state moves.

### A bug this check had first, recorded because it is the instructive part

The first version cumulated the deltas **from zero** rather than from the recorded
start vector. The deltas are `v^{pi_k} - v^{pi_{k-1}}` and the chain begins at
`v^{pi_0}`, so that reconstructs `v^{pi_k} - v^{pi_0}` — a different vector.

It reported a large disagreement with `FP-GAP-001`: reconstruction error `2.629`,
`296` rises in the gap sequences, a decay factor of `1.03` (gaps *growing*), and only
`22.7%` of the initial suboptimality closed.

Every one of those numbers was absurd on its face — a certified iteration that closes
`22%` of the gap while its gap grows — and the absurdity is what identified the bug.
**A cross-check that reports a large disagreement should first ask whether it has
misread the data format.** It had.

## Result: the conclusion reproduces; one quoted number does not

| claim in `FP-GAP-001` | this check | agree? |
|---|---|---|
| `v^{pi_k}` reconstructs to `4.441e-16` over 96 trajectories | **`4.441e-16`** | **yes, exactly** |
| `v*` satisfies Bellman to `~1e-16` and dominates `v^{pi_0}` | `8.882e-16` worst, dominates everywhere | **yes** |
| not one rise in any gap trajectory | **`0` rises over 86 non-trivial trajectories** | **yes** |
| long trajectories (>= 24 steps): `38` | **`38`** | **yes, exactly** |
| long trajectories close `99.9717%` of the initial suboptimality | **`99.9388%`** | **yes** (0.03pp) |
| **median decay factor `0.327`** | **`0.758` max-norm, `0.760` sum-norm** | **no — differs by `2.3x`** |
| "the gap decays by a constant factor of about `0.45` per step" | constant `~0.69` on the longest trajectory | **no** |

**On the longest trajectory** (`0.08/1 expected_exact frozen`, 32 steps) the per-step
fraction of the gap closed settles to a constant, which is the qualitative claim and it
holds:

```text
step   4 : gap 0.116930 : fraction closed 0.4621
step   8 : gap 0.019544 : fraction closed 0.3305
step  12 : gap 0.004339 : fraction closed 0.3085
step  16 : gap 0.001005 : fraction closed 0.3056
step  20 : gap 0.000234 : fraction closed 0.3051
step  24 : gap 0.000055 : fraction closed 0.3050
step  28 : gap 0.000013 : fraction closed 0.3049
step  32 : gap 0.000003 : fraction closed 0.3049
```

A **constant** fraction closed per step — that is geometric convergence, and it is what
`FP-GAP-001` is arguing for. The constant is `~0.305`, not `0.45`.

### The internal inconsistency

`FP-GAP-001`'s section 2 states two things that cannot both hold:

> | 20 | — | — | `0.45` |
> **The gap decays by a constant factor of about `0.45` per step**

A `gain / gap` ratio of `0.45` means `45%` of the gap is closed each step, so the gap
**decays by a factor of `0.55`**. The text calls the fraction closed the decay factor.
One of the two statements is mislabelled; on this check's measurement the fraction
closed is `~0.305` and the decay factor is `~0.69`, so it is the *value* as well as the
label that differs from this check.

## What this changes, and what it does not

**Does not change the conclusion.** Geometric convergence is confirmed: a constant
per-step fraction of the gap is closed, stable across 28 steps, with zero rises and
`99.94%` of the initial suboptimality closed on long trajectories. **The tail updates
are real work**, and `FP-MEANING-001`'s reconciliation with `FP-GAP-001` stands.

**Does change one number.** The quoted median decay factor `0.327` and the quoted
`0.45` fraction closed do not reproduce here (`0.758` and `0.305`). The direction of the
error is conservative — the true convergence is *slower* than claimed, not faster — so
no claim is inflated by it. But it should be reconciled, and its author should do that,
not this session.

**Possible benign explanations this check cannot rule out**: a different trajectory
population for the median (this check uses all trajectories with >= 24 emitted steps and
takes the median of per-trajectory medians); a different definition of "gain" in the
`gain / gap` column; or a median taken over a different step window. The `0.327` is not
reproduced under either norm, but the check was not written to reverse-engineer the
other session's aggregation, and it does not claim the other session computed wrongly —
only that its number does not reproduce from the sealed bundle under a natural reading.

## Reproduce

`tmp/xcheck_fp_gap_001.py` in the main tree (untracked scratch). Read-only: it loads
the sealed bundle and writes nothing.
