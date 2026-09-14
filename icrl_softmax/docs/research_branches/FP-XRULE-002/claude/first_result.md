# FP-XRULE-002 first result: under the per-state rule the certified iteration has no stopping point within 12 steps

> **⛔ WITHDRAWN 2026-09-14 (third-party review, FAIL).** `L12S` is **not a licensed bound**. The
> split protects only the `k = 0` step: `V_0` is fixed given half A and `B ⊥ A`, so `T^B(V_0)` may
> be concentrated, but `V_1` itself depends on B, so every later step applies Hoeffding to a
> function computed from the same B — the `L12M` defect, entering one iteration later. Only
> `n_iter = 1` is licensed; the runs used `n_iter = 12`. See
> [`docs/derivations/FP-L12S-REVIEW-001-withdrawal.md`](../../../derivations/FP-L12S-REVIEW-001-withdrawal.md).
> The **measurements** in this file stand; any "sound/licensed" characterisation of `L12S` does not.

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`. Baseline: `583bcaa`.
Actor: Claude, under the user's direct instruction of 2026-09-13 ("继续").
Task: [FP-XRULE-002](../../research_tasks/FP-XRULE-002.md) (hypotheses and `K = 12` registered before
the run).
Upstream: [FP-XRULE-001](first_result.md), whose `K = 4` **saturated** — every per-state cell reached
the horizon in both families, so `K = 4` could not separate "the loop can go further" from "the loop
stops there".

**Preliminary result, single actor. Nothing here is VERIFIED.**

## 1. Design

Identical families, environments, seeds, routes, cells and rules as FP-XRULE-001; the only change is
`K: 4 → 12` (`δ_k = 0.05/12`). `evaluate_fp_xrule_001.run_family` reads its horizon from the module
global, so it is parameterised by assignment rather than by editing that file, keeping FP-XRULE-001's
recorded hash valid.

## 2. Results

| family | cell | emitted | mean simulated steps | reached `K=12` | mean value gain | items if alone | partial updates |
|---|---|---:|---:|---:|---:|---:|---:|
| **f1** (`S=6`) | `conj|frozen` | `31` | `1.323` | `0` | `0.9404` | `133,169,152` | `0%` |
| | `conj|L12S` | `115` | `2.198` | `0` | `3.3133` | `221,249,536` | `0%` |
| | `perstate|frozen` | `932` | `10.083` | `60` | `8.6893` | `1,015,021,568` | `95%` |
| | **`perstate|L12S`** | **`1053`** | `11.177` | **`76`** | **`9.6718`** | `1,125,122,048` | `89%` |
| **f2** (`S=4`) | `conj|frozen` | `389` | `4.906` | `14` | `5.8007` | `493,879,296` | `0%` |
| | `conj|L12S` | `416` | `5.188` | `14` | `6.4835` | `522,190,848` | `0%` |
| | `perstate|frozen` | `1109` | `11.615` | `90` | `8.1739` | `1,169,162,240` | `61%` |
| | **`perstate|L12S`** | **`1125`** | `11.760` | **`92`** | **`8.2779`** | `1,183,842,304` | `60%` |

Wall clock `26.0 min` for the whole eight-cell, two-family run at `K = 12`.

## 3. The finding: no natural stopping point

**`60 / 96` (F1) and `90 / 96` (F2) per-state trajectories run all the way to step 12**, and their mean
simulated length is `10.1` and `11.6` against a horizon of `12`. The loop is not self-terminating: at
every step up to 12 it still finds a certified, non-degrading improvement. **Its length is set by the
horizon the experimenter chooses, not by the certificate and not by the rule.**

That is a change in what limits the project. Across this line the reported bottlenecks have been, in
order: the bridge lemma (falsified, repaired), the certificate's constants (`L1`/`L12`, free), the
propagation lever (closed as a negative result after the audit), the multi-step risk schedule (closed),
and the update rule (per-state, a large win). **With the per-state rule the bound is no longer the
binding constraint, and what is left is the horizon** — an accounting decision, not a mathematical one.

Two supporting observations:

- **The conjunctive gate is still what kills the conjunctive arm**, not the information: at `K = 12` it
  still stops after `1.3` (F1) and `4.9` (F2) steps on average, with `0` and `14` trajectories reaching
  the horizon. The rule, not the data budget or the bound, decides.
- **The per-state advantage is genuinely from partial updating**: `95%` / `89%` (F1) and `61%` / `60%`
  (F2) of emitted steps updated strictly fewer than all `S` states. It is not "the same update with a
  looser gate".

## 4. Verdicts

| hypothesis | verdict |
|---|---|
| `H1`① no componentwise degradation, any cell | **PASS** (`0` in all eight) |
| `H1`② `E_Q ≥` realized `‖Q̂−Q^π‖∞`, every step | **PASS** (`0` violations in all eight) |
| `H2` at least one per-state trajectory reaches `K=12` in each family | **PASS** (F1 `60+76`, F2 `90+92`) |
| `H3` per-state mean simulated steps `≥ 8` in both families | **PASS** (`10.083 / 11.177`, `11.615 / 11.760`) |
| `H4` `perstate|L12S ≥ perstate|frozen` in value | **PASS** (`+11.3%` F1, `+1.3%` F2) |
| `H5` cost and same-total-data truncation reported | **PASS**, see §5 |
| `H6` `≥ 50%` of `perstate` emitted steps are partial updates | **PASS** (`95% / 89%`, `61% / 60%`) |

## 5. Cost, and the same-total-data comparison

Truncating every cell to `conj|frozen`'s shortest run in its family (`1` step in both):

| family | cell | items | truncated emitted | truncated value |
|---|---|---:|---:|---:|
| `f1` | `conj|frozen` | `133.2M` | `11` | `0.4085` |
| | `conj|L12S` | `221.2M` | `45` | `1.5493` |
| | `perstate|frozen` | `1,015.0M` | `96` | `3.1249` |
| | `perstate|L12S` | `1,125.1M` | `96` | `3.4611` |
| `f2` | `conj|frozen` | `493.9M` | `70` | `2.6748` |
| | `conj|L12S` | `522.2M` | `85` | `3.0824` |
| | `perstate|frozen` | `1,169.2M` | `96` | `3.4496` |
| | `perstate|L12S` | `1,183.8M` | `96` | `3.4697` |

Every ordering survives truncation. The cost column is part of the result: at `K = 12` the per-state
cells spend `7.6×` (F1) and `2.4×` (F2) what `conj|frozen` spends, because `conj|frozen` stops at once
and the per-state loop keeps working. **The per-state rule is not a cheaper way to the same place; it is
a way to a place the conjunctive rule never reaches, and it is paid for in data.**

## 6. Reproducing FP-XRULE-001's substitution finding at long `K`

At `K = 4` the certificate's marginal effect under the per-state rule was `+8.5%` (F1) and `+1.3%` (F2).
At `K = 12` it is `+11.3%` and `+1.3%`: the substitution pattern holds and does not vanish with a longer
horizon. Rule first, bound second — now measured at two horizons on two families.

## 7. What this establishes, and what it does not

**Established**

- The per-state certified iteration **does not terminate on its own within 12 steps** on either new
  family; the binding constraint has moved from the method to the horizon.
- The per-state advantage is mechanistically **partial updating**, quantified.
- The rule/bound substitution reproduces at `K = 12`.

**Not established**

- **Where it stops, if anywhere.** `K = 12` shows it has not stopped; it does not show it never stops.
  Extending further is now a cost decision rather than a scientific one, which is exactly the state the
  audit predicted ("proof closed, then horizon"), and the honest reading is that **further horizon
  extension alone has low value**.
- **Whether the value converges.** Emissions keep rising, but mean value gain `8.6893 → 9.6718` (F1) is
  a `11%` increase for a `10%` increase in steps — the marginal value per step is falling, and no
  convergence claim is made here.
- **No network cells** and **no third-party independence**; both remain open, and the second cannot be
  closed by this actor.

**Same-actor throughout.**

## 8. Verification

`verify_fp_xrule_002.py` replays both families from the sealed bundle. It differs structurally from the
FP-XRULE-001 verifier in one way that matters at this horizon: the certification batch belongs to a
`(record, step)` pair, not to a cell, so it is drawn **once per step and reused across the four cells**
(the earlier verifier re-drew it per cell — correct but `4×` wasteful, and at `K = 12` that is the
difference between 25 and 100 minutes).

| check | result over **`5,592` replayed steps** (`2,209` batch draws) |
|---|---|
| `C1` emission and `min_lb` reproduced from the replayed policy, both rule branches | `0` mismatches, max `\|Δmin_lb\|` `0.0` |
| `C2` `E_Q ≥` realized `‖Q̂−Q^π‖∞` | `0` violations, max `\|ΔE_Q\|` `0.0` |
| `C3` value deltas recomputed along the replayed chain | `0` degradations, max `\|Δ\|` `0.0` |
| `C4` sealed totals incl. `simulated_steps`, `items_if_run_alone`, `stopped_at` | `0` mismatches |
| `C5` the `L12S` cells ran the registered configuration; halves sum; risk sums to `δ_k` | `0` mismatches |

The `states_updated` figures quoted in §3 come from the bundle rather than from the replay; they are
consistent with the replayed `min_lb` in every step, since a state is updated exactly when its row has a
strictly positive lower bound.

