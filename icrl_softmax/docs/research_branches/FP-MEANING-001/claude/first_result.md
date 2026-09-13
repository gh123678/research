# FP-MEANING-001 first result: the iteration outlives its own certificate

Date: 2026-09-12.
Branch: `claude/FP-CENSUS-001`. Baseline: `c1cebd7`.
Actor: Claude, under the user's instruction of 2026-09-12 ("好的").

## 1. What was asked, and why it is not merely definitional

`FP-HORIZON-001` left this open in its own task sheet:

> a gain of `4e-7` is certified, componentwise non-degrading and strictly positive,
> and whether that counts as "policy improvement" is a definitional question the line
> has not settled.

It is not purely definitional, because the decision rule requires
`min_s LB_s > 0`, where `LB_s = Î_s − E_Q‖Δπ_s‖₁` is a **proven** lower bound on the
true improvement at state `s`. So every emitted update carries a *certified*
improvement as well as a realized one, and the recorded `min_lb` is its size. Whether
`LB_s = 2e-16` counts as proving an improvement is answerable by measuring the
resolution of the arithmetic that computes it.

No new iteration was run. One new quantity was measured — the arithmetic floor — and
then `FP-HORIZON-001`'s 32-step trajectory was analysed.

## 2. `H1`: the floor is measured, and it matters that it was

| quantity | value |
|---|---:|
| `policy_quantities` vs value iteration, per record: **max** `\|Δv\|` | `2.287e-14` |
| median `\|Δv\|` over 24 records | `1.976e-14` |
| `float64` machine epsilon, for reference | `2.220e-16` |

The two routes share no code. Note what the measurement reveals: the floor is
**dominated by the value-iteration stopping tolerance, not by machine epsilon**, and
the two differ by two orders of magnitude. That gap is not academic — it moves the
answer by eight steps, as §4 shows. Asserting `eps` would have been the natural
shortcut and it would have been wrong by 100x.

## 3. `H2`: the realized tail gains are real

The smallest realized gain at step `32` is `3.845e-07`, which is **`1.7e7` times** the
measured floor. So the tail updates are not round-off: something genuinely improves,
and it improves by a quantity `float64` resolves easily.

`H2` **PASS**. This is the half of the answer that favours the iteration.

## 4. `H3`: the certified bound decays to nothing, and the iteration runs on

The certified bound `min_lb` decays geometrically by a mean factor of **`6.54` per
step** (`H4` **PASS**, registered range `[3, 10]`):

| step | `min_lb` (certified) | min gain (realized) | certified fraction | vs method floor |
|---|---:|---:|---:|---:|
| 1 | `3.165e-04` | `6.910e-01` | `4.58e-04` | `1.4e+10` |
| 10 | `2.772e-08` | `3.738e-03` | `7.42e-06` | `1.2e+06` |
| **18** | **`1.210e-14`** | `3.374e-06` | `3.59e-09` | **`5.3e-01`** |
| 26 | `1.771e-16` | `8.885e-06` | `1.99e-11` | `7.8e-03` |
| 32 | `2.164e-16` | `3.845e-07` | `5.63e-10` | `9.5e-03` |

**The certified bound crosses the measured floor at step `18`, and machine epsilon at
step `26`. The iteration is still emitting `12` route-records at step `32`.**

`H3` **PASS**. Past step `18` the iteration is certifying improvements whose *proven*
lower bound is at or below the disagreement between two independent value
computations — and nothing fails. Every validity check, every coverage check, every
componentwise non-degradation check still passes, because they are all tests of
*strict positivity* and `2e-16` is strictly positive.

**That is the finding: the certified iteration outlives the meaningfulness of its own
certificate.** It is not that the certificate is wrong; it is that it stops saying
anything, and no formal check in this line can detect that, because the property being
tested is `> 0` rather than `> something`.

`H5` **FALSIFIED**, narrowly: the certified fraction collapses by `5.9` orders of
magnitude against a registered minimum of `6`. Reported as falsified rather than
rounded up.

## 5. `H6`: the reach under each definition — and this task does not choose

| definition | first failing step | holds | last satisfying |
|---|---:|---:|---:|
| `D1` realized gain `> 0` | never | `32/32` | `32` |
| `D2` certified bound `> 0` | never | `32/32` | `32` |
| `D3` certified `>` method floor (`2.3e-14`) | **`18`** | `20/32` | `23` |
| `D4` certified `>` machine eps (`2.2e-16`) | **`26`** | `30/32` | `31` |
| `D5` certified `>` network `float32` gap (`1.4e-05`) | **`2`** | `5/32` | `6` |

`D1` and `D2` are what this line has been using: under them the iteration reaches `32`
and has not stopped.

`D3` fails at `18`, which is within two steps of the network's projected trust horizon
of `17` — two independent routes to roughly the same place, one from the certificate's
resolution and one from the network's.

`D5` fails at step `2`, which says that **the certified bound proves more than the
network can resolve only for the first few steps**; after that the network could not
confirm the certified improvement even in principle.

**A 'last satisfying step' column alone would mislead**, which is why the table reports
first failure and coverage: `min_lb` is a *minimum over the emitting set*, and that set
changes, so the series is not monotone. `D3` recovers above the floor at step `21`
before failing again at `24`.

**The task declines to choose.** Each row is defensible, they give first-failure steps
from `2` to never, and the choice is the user's. Choosing one after seeing this table
would be the post-hoc threshold selection the task sheet prohibits.

## 6. Reconciliation with `FP-GAP-001`, which asks the same question from the other side

While this task was being written, `FP-GAP-001` was executed and committed on this
same branch, re-posing the identical question as a measurement: *is the iteration
closing the distance to the optimal policy, or emitting certified improvements while
that distance sits at a floor?* Its answer is **the iteration converges
geometrically** — each step closes about `45%` of the remaining gap, stably from step
`4` to step `32`, and long trajectories close `99.97%` of the initial suboptimality.
Its conclusion is that **the tail updates are real work**, and that the genuine
shortfall is the *short* trajectories, which leave `45%` on the table.

**That is consistent with this task, and the two together say more than either
alone.**

| | `FP-GAP-001` | `FP-MEANING-001` |
|---|---|---|
| measures | the **realized** distance to `v*` | the **certified** lower bound `min_lb` |
| finds | geometric convergence, `~0.45`/step, stable to step 32 | geometric decay, `6.54 x`/step, to machine epsilon |
| verdict on the tail | real work, closing `45%` of what remains | real gain (`1.7e7 x` the floor) but **proven** bound `2.2e-16` |

The two are two halves of one statement:

> **The iteration keeps closing about `45%` of the remaining gap per step, so the tail
> updates are genuine improvements; but the certificate stops proving that around step
> `18`–`26`. The updates remain real work that the guarantee no longer covers.**

So `FP-GAP-001`'s "the tail updates count" and this task's "the certified bound
evaporates" are not in tension. The measured improvement is genuine and large
*relative to what remains*; what runs out is the **resolution of the guarantee**, not
the progress of the iteration. Both tasks agree on the realized side — `FP-GAP-001`
says the gain is `~45%` of the remaining gap, this task says the gain is `1.7e7 x` the
arithmetic floor — and they differ only in which quantity is being asked about.

The practical consequence is sharper than either report alone: **a reader told "the
iteration reaches 32 steps" is being told something true about the iteration and
something unexamined about the guarantee.** The right next question is not how far the
iteration goes, which `FP-GAP-001` has now answered, but whether the decision rule
should carry a magnitude floor.

## 7. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` floor measured, not assumed | **PASS** (`2.287e-14` max, `1.976e-14` median) |
| `H2` tail gains above the floor | **PASS** (`1.7e7 x`) |
| `H3` bound crosses a floor, iteration runs on | **PASS** (step `18`, still emitting at `32`) |
| `H4` geometric decay | **PASS** (mean factor `6.54`, range `[3, 10]`) |
| `H5` at least six orders of collapse | **FALSIFIED** (`5.9` orders) |
| `H6` reach under each definition | reported, deliberately unresolved |

Construction checks **PASS**.

## 8. What this settles, and what it does not

**Settled**

- Every emitted update in the 32-step trajectory is a **real** improvement
  (`1.7e7 x` the arithmetic floor at the tail): the realized half of the claim holds
  all the way.
- Every emitted update also carries a **certified** lower bound, and that bound decays
  `6.54 x` per step to `2.2e-16` — below the resolution of the arithmetic that computes
  it, from step `18` (vs two-method disagreement) or step `26` (vs machine epsilon).
- **The line's formal checks cannot see this**, because they test `> 0` and the bound
  remains strictly positive. This is a blind spot in the acceptance criteria, not a
  failure of any particular run.
- On the network path the certified bound exceeds the network's arithmetic resolution
  only for the first few steps (`D5` fails at step `2`).

**Not settled — and now a decision rather than an oversight**

- **Which definition of "improvement" this project claims.** The horizon under `D3` is
  `18`, under `D2` it is at least `32`, under `D5` it is `2`. The line has been
  implicitly reporting `D1`/`D2` and calling the result "reaches 32 steps"; a reader who
  assumed `D3` would take a very different message from the same data.
- **Whether the guarantee should carry a magnitude floor.** The decision rule could
  require `min_s LB_s > f` for a stated `f` rather than `> 0`. That is a change to the
  frozen rule and therefore a new task, not a reinterpretation of this one.
- **Same-actor throughout**, per the user's standing instruction.
