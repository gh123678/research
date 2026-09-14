# FP-ITER-BOUND-002 first result: the risk schedule is not the lever — and what actually stops the loop

> ## CORRECTIONS 2026-09-13 (independent audit), on top of a withdrawal
>
> **Withdrawn.** Every cell here uses `L12M(0.05)`, whose concentration step is not licensed
> (Hoeffding applied to a function computed from the same successor draws it averages over).
> The emission counts (`99 / 94 / 79`), the value gains (`2.901 / 2.920 / 2.833`) and the
> `H1`–`H7` table are therefore withdrawn; the comparison must be re-run with the repaired
> arm `L12S`.
>
> **Three claims narrowed or retracted (audit items 3a–3c).**
>
> 1. *"A non-uniform schedule cannot help"* is **too broad**. What is proved is that
>    `Σ_k log(1/δ_k)` is strictly convex on `Σ_k δ_k = δ_total`, so the equal allocation
>    uniquely minimises that sum — and, because the radius is monotone in `log(1/δ_k)`, it
>    also minimises `max_k log(1/δ_k)`. That is a statement about **this** objective, for a
>    union bound with a monotone radius. It is not a statement about every schedule for
>    every protocol.
> 2. *"Shrinking `K` is pure truncation"* is **wrong as written**. `K` sets the per-step
>    budget `δ_total/K`, which changes each certificate, hence which steps emit, hence the
>    policy trajectory itself. The measured mean-length ordering (`2.35 < 2.90 < 3.06`) is
>    the joint effect of the cap and the budget, not truncation alone; my `H2` registration
>    got the sign wrong for the same reason.
> 3. *"A confidence sequence is worse"* is **narrower than claimed**. What was compared is
>    one concrete normal-mixture boundary, `σ√((log n + 2log(1/δ))/n)`, against the
>    fixed-time union bound, `σ√((2log K + 2log(2/δ))/n)`. At this protocol's parameters
>    (`log n = 9.70` vs `2 log K = 5.55`) that one form loses. It does **not** follow that
>    no anytime-valid construction can win; see e.g.
>    [Howard et al., time-uniform Chernoff bounds](https://arxiv.org/pdf/1810.08240) for the
>    family this comparison did not cover.
>
> **What survives**: the two closed-form arguments above (as scoped), and `H4`'s
> decomposition of the stops by `E_Q/h` — though its population is `L12M`'s and must be
> recomputed with the sound arm.

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`. Baseline: `dfe5643`.
Actor: Claude, under the user's direct instruction of 2026-09-13 ("好").
Task: [FP-ITER-BOUND-002](../../research_tasks/FP-ITER-BOUND-002.md) (theory, hypotheses and bands
registered before the run).
Code: `evaluate_fp_iter_bound_002.py`, `analyze_fp_iter_bound_002.py`, `verify_fp_iter_bound_002.py`.

**Preliminary result, single actor.**

## 1. Two candidate levers, both closed by argument before any experiment

`FP-ITER-BOUND-001` located the cost at `δ_k = δ_total/K` and proposed two fixes. Both fail, and the
reason is worth recording:

**(a) A non-uniform schedule cannot help.** Under `Σ_k δ_k = δ_total` the union bound costs
`Σ_k log(1/δ_k)`, which is strictly convex in `δ_k`, so the equal allocation is the unique minimum —
and the certificate's radius is monotone in `log(1/δ_k)`, so the *worst* step (the one that decides
whether the trajectory stops) is also minimised by equal allocation. Any tilt makes some step worse.

**(b) An anytime-valid confidence sequence is worse here.** A normal-mixture CS has a boundary of order
`σ·sqrt((log n + 2 log(1/δ))/n)`, while fixed-sample-plus-union costs
`σ·sqrt((2 log K + 2 log(2/δ))/n)`. The comparison is `log n` against `2 log K`:

```text
this protocol:  n ≈ 16384 → log n = 9.70;   K = 16 → 2 log K = 5.55
```

The CS spends **4.15 more log units**. The crossover sits near `n ≲ K²`, and this protocol is nowhere
near it. A CS buys `log log K` in place of `log K`, but pays `log n` for it, and `log n` is the larger
number whenever the per-step sample is large.

## 2. So what was left: K itself

If the schedule cannot be improved, the only remaining term is K, which is a free parameter of the
*guarantee*, not of the estimator. The measurement: the same population and per-step data, with the
final arm `L12M(0.05)`, run at `K ∈ {4, 8, 16}` — **all three schedules share each step's batch**, so
the comparison is paired and free of sampler confound.

## 3. Results

| arm | `δ_k` | emitted | mean length | reached its `K` | mean value gain |
|---|---:|---:|---:|---:|---:|
| `frozen@K16` | `0.05/16` | `36` | `1.750` | `0` | `1.7341` |
| **`L12M(0.05)@K16`** | `0.05/16` | **`99`** | `3.062` | `0` | **`2.9014`** |
| `L12M(0.05)@K8` | `0.05/8` | `94` | `2.896` | `3` | `2.9199` |
| `L12M(0.05)@K4` | `0.05/4` | `79` | `2.354` | **`14`** | `2.8325` |

**Reducing K does not help.** Total emissions fall monotonically (`99 → 94 → 79`) and total value
follows (`2.901 → 2.920 → 2.833`); the mean trajectory length *falls* as K shrinks (`3.06 → 2.90 →
2.35`), which is pure truncation. The `K=4` cell reaches its horizon `14` times against `K=16`'s `0`,
but that is arithmetic — it is a shorter horizon — not a better loop.

## 4. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` (mandatory) no degradation, no coverage violation | **PASS** (`0`/`0` in all four cells) |
| `H2` length grows as `K` shrinks | **FALSIFIED — the sign is opposite** (`2.354 < 2.896 < 3.062`) |
| `H3` at least `15/48` reach `K=4` | **FALSIFIED** (`14/48`; misses by one) |
| `H4` at least `40%` of stops are "near-line" (`E_Q/h ≤ 1.25`) | **PASS** (`52%`) |
| `H5` the final arm beats FP-ITER-BOUND-001's `L12M(0.5)` at `K=16` | **PASS** (`99` vs `89`) |
| `H6` `K=4` buys `≥ 1.20×` the value | **FALSIFIED** (`0.976×`) |
| `H7` `K=8` and `K=16` lengths differ by `< 0.5` | **PASS** (`0.167`) |

`H2`'s sign error is mine and is worth stating plainly: I registered "a smaller `δ` per step ⇒ longer
trajectories" and forgot that a smaller K also caps the trajectory. Mean length is the wrong metric
for this comparison — the emission count is the right one, and it is monotone in the *opposite*
direction. Recorded as falsified, not re-interpreted.

## 5. What actually stops the loop — the useful number in this run

`H4` decomposes the `48` stops of `L12M(0.05)@K16` by how far the certificate was from letting the
step through:

```text
median E_Q/h at a stop : 1.2422
E_Q/h <= 1.25          : 25 stops (52%)   -- the bound is still (just) what stopped it
E_Q/h >  1.25          : 23 stops (48%)   -- the improvement genuinely ran out
```

**It is now a near-exact split.** Roughly half the remaining stops are still certificate-driven and
half are convergence-driven: the guaranteed improvement available at that state has become small
enough (`h` tiny) that no reasonable bound would admit it. Neither mechanism dominates, so the
"staircase" population is a mixture of two different phenomena, and future work should separate them
rather than treating "the loop stopped" as one fact.

## 6. The line, in one table

| lever | sound? | model-free? | `E_Q` at `c64k` | iteration emissions |
|---|---|---|---:|---:|
| frozen certificate | yes | yes | `0.12044` | `36` |
| `L1` known-`Q̂` envelope | yes | **yes** | `−11.0%` | — |
| `L12` `+` full risk budget | yes | **yes** | `−13.5%` | `66` |
| **`L12M(0.05)`** propagation | yes | **yes** | **`−39.6%`** | **`99`** |
| `L123` kernel propagation | yes | no (oracle) | `−41.5%` | `110` |

**The certified iteration went from `36` to `99` emitted steps (`2.75×`) and from `1.734` to `2.901`
mean value gain (`1.67×`), model-free, with `0` degradations and `0` coverage violations at every step
of every arm.** The kernel arm, which cannot support the behavioural-data claim, is worth another `11`
steps.

## 7. What this establishes, and what it does not

**Established**

- The multi-step risk schedule is **not** improvable by reallocation (convexity) and **not** improvable
  by a confidence sequence at these sample sizes (`log n > 2 log K`); both are closed by argument, with
  the crossover condition stated.
- Shortening the horizon does not extend the loop; it truncates it.
- The remaining stops are a **~50/50 mixture** of certificate-driven and convergence-driven, measured,
  not asserted.

**Not established**

- Whether the `48%` convergence-driven stops could be converted by a *different* update rule rather
  than a tighter bound — that is a question about the policy-improvement step, not the certificate.
- Nothing here crosses the `K` horizon: `0` trajectories reach `16` steps even with a `40%` tighter
  bound, and the honest statement is that the loop's length is set by the population's improvement
  margin, not by the protocol's budget.
- The `K=4`/`K=8` cells carry guarantees over `4`/`8` steps only and are quoted as such.
- **Same actor throughout**, and the replay verifier shares the executor's protocol model even though
  it shares no code path with the evaluator.

**Same-actor throughout**, per the user's standing instruction.
