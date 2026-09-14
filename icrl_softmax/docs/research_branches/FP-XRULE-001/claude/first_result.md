# FP-XRULE-001 first result: both levers transfer to two new MDP families — and they are substitutes, not complements

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`. Baseline: `3dfa1e9`.
Actor: Claude, under the user's direct instruction of 2026-09-13 ("好的去做"), executing item 3 of the
independent audit's ordering.
Task: [FP-XRULE-001](../../research_tasks/FP-XRULE-001.md) (families, arms, `K`, hypotheses and bands
registered before the run).
Upstream: the audit's requirement that the two head-line mechanisms be re-tested on environments **no
earlier task has used**, with per-step budget, total data cost, and a same-total-cost truncation.

**Preliminary result, single actor. Nothing here is VERIFIED.**

## 1. Design

Two families that `FP-XFAM-001` registered and no task had used:

| family | states × actions | `GAP_BONUS` | `R*` | mixings | `task_index` |
|---|---|---|---|---|---|
| **F1** | `6 × 4` (24 pairs) | `0.5` | `1.5` | `0.08, 0.5` | `200..223` |
| **F2** | `4 × 3` (12 pairs) | `1.0` | `2.0` (`E ≈ 13.33`) | `0.2, 0.7` | `300..323` |

`96` route-records per family. Four cells sharing each step's batch, `K = 4` frozen,
`16384` chains × `64`, `δ_k = 0.05/4`:

```text
rule        ∈ {conj (conjunctive gate), perstate (per-state conservative update)}
certificate ∈ {frozen, L12S(f=0.9, p=0.05)}
```

The per-state rule is sound by the same argument as the conjunctive one: row-wise,
`A_s = Σ_a Δπ_s(a)Q^π(s,a) ≥ LB_s ≥ 0` for every state, so the performance-difference identity gives
`v^{π'}(s0) − v^π(s0) = E[Σ_t γ^t A_{s_t}] ≥ 0` componentwise.

## 2. Results

| family | cell | emitted | mean length | mean value gain | items if run alone | cov. | deg. |
|---|---|---:|---:|---:|---:|---:|---:|
| `f1` | `conj|frozen` | `39` | `1.365` | `1.2174` | `137,363,456` | `0` | `0` |
| | `conj|L12S` | `135` | `2.281` | `3.8532` | `229,638,144` | `0` | `0` |
| | `perstate|frozen` | `382` | `4.000` | `8.3968` | `402,653,184` | `0` | `0` |
| | **`perstate|L12S`** | **`384`** | `4.000` | **`9.1144`** | `402,653,184` | `0` | `0` |
| `f2` | `conj|frozen` | `251` | `3.135` | `5.8122` | `315,621,376` | `0` | `0` |
| | `conj|L12S` | `299` | `3.531` | `6.6341` | `355,467,264` | `0` | `0` |
| | `perstate|frozen` | `384` | `4.000` | `7.8099` | `402,653,184` | `0` | `0` |
| | **`perstate|L12S`** | **`384`** | `4.000` | **`7.9108`** | `402,653,184` | `0` | `0` |

Wall clock `12.1 min` for the whole eight-cell, two-family run.

**Both mechanisms transfer.** The per-state rule is worth `+589.7%` / `+136.5%` (F1, under `frozen` and
`L12S`) and `+34.4%` / `+19.2%` (F2); the tightened certificate is worth `+216.5%` / `+8.5%` (F1) and
`+14.1%` / `+1.3%` (F2). All eight cells run to the horizon under the per-state rule in both families.

## 3. The finding worth keeping: the two levers are substitutes

| family | rule | certificate effect (`L12S` over `frozen`) |
|---|---|---:|
| `f1` | `conj` | **`+216.5%`** |
| `f1` | `perstate` | `+8.5%` |
| `f2` | `conj` | **`+14.1%`** |
| `f2` | `perstate` | `+1.3%` |

The certificate's value collapses by one to two orders of magnitude once the per-state rule is in
place. That is not a defect of either lever — it is what "the binding constraint moved" looks like:
under the conjunctive gate most trajectories die at step 1 for gate reasons, and a tighter bound is
worth a great deal; under the per-state rule the loop already reaches the horizon, so there is very
little left for a tighter bound to buy. **Reporting either lever's headline without the other is
therefore misleading**, and the honest summary of this line is a *pair*: rule first, bound second.

## 4. Verdicts

| hypothesis | verdict |
|---|---|
| `H1`① no componentwise degradation, any cell | **PASS** (`0` in all eight) |
| `H1`② `E_Q ≥` realized `‖Q̂−Q^π‖∞`, every step | **PASS** (`0` violations in all eight) |
| `H2` per-state beats conjunctive in both families under both certificates | **PASS** (all four comparisons) |
| `H3` `L12S` beats `frozen` in both families under both rules (direction only) | **PASS** (all four; magnitudes `+216.5% … +1.3%`) |
| `H4` `perstate|L12S ≥ conj|frozen` in both families | **PASS** |
| `H5` cost and same-total-data truncation reported | **PASS**, see §5 |
| `H6` construction is dimension-generic | **PASS** — F1's `24` pairs and F2's `12` pairs both give `0` coverage violations in every cell |

## 5. Cost, and the same-total-data comparison

Per-step budget is identical across cells (`1,048,576` items), but the cells that stop early draw less.
Truncating every cell to the step count of the *shortest* cell in its family (F1: `1` step, F2: `1`):

| family | cell | items | truncated emitted | truncated value |
|---|---|---:|---:|---:|
| `f1` | `conj|frozen` | `137.4M` | `14` | `0.5421` |
| | `conj|L12S` | `229.6M` | `51` | `1.7801` |
| | `perstate|frozen` | `402.7M` | `96` | `3.2284` |
| | `perstate|L12S` | `402.7M` | `96` | `3.4804` |
| `f2` | `conj|frozen` | `315.6M` | `75` | `2.7627` |
| | `conj|L12S` | `355.5M` | `86` | `3.0946` |
| | `perstate|frozen` | `402.7M` | `96` | `3.4561` |
| | `perstate|L12S` | `402.7M` | `96` | `3.4703` |

**Every ordering survives the truncation**: at one step per record the per-state cells still lead, and
`L12S` still leads within each rule. The gains are therefore not bought with extra data — but the cost
column is part of the result, not a footnote: the per-state cells spend `2.9×` (`f1`) and `1.3×` (`f2`)
what `conj|frozen` spends, because they keep working instead of stopping.

## 6. What this establishes, and what it does not

**Established**

- Both mechanisms generalise **beyond the family they were found in**, including to a different
  state/action dimension (`6×4`, 24 pairs) and a different reward scale (`R* = 2.0`, envelope `13.33`).
- The certificate machinery is **dimension-generic**: `0` coverage violations on `24` pairs, where all
  earlier evidence was on `12`.
- The two levers are **substitutes**, measured on two families under both rules.

**Not established**

- **No network cells here.** This run uses the two array routes only; the network producer was tested on
  the original family and is not re-tested across families.
- **`K = 4`.** The per-state cells already reach this horizon in both families, so `K = 4` cannot
  separate "runs to the horizon" from "would run longer"; a longer `K` would be needed to see where the
  per-state loop actually stops, and that is a new cost decision.
- **The same-actor gap is unchanged**, and one of the four cells (`perstate|L12S` on F2) improves its
  rule's baseline by only `1.3%`, which is inside the range where a different seed schedule could
  matter; it is reported rather than emphasised.

**Same-actor throughout.**

## 7. Verification

`verify_fp_xrule_001.py` replays all four cells on both families from the sealed bundle: it rebuilds
each cell's policy chain, re-runs the family's array route on the replayed policy, re-draws each step's
batch, and re-derives the certificate, the **rule's own decision** (both branches, including
`states_updated`), the exact values and the cost fields.

| check | result over **`2,526` replayed steps** |
|---|---|
| `C1` emission and `min_lb` reproduced from the replayed policy | `0` mismatches, max `\|Δmin_lb\|` `0.0` |
| `C2` `E_Q ≥` realized `‖Q̂−Q^π‖∞` | `0` violations, max `\|ΔE_Q\|` `0.0` |
| `C3` value deltas recomputed along the replayed chain | `0` degradations, max `\|Δ\|` `0.0` |
| `C4` sealed totals incl. `simulated_steps`, `items_if_run_alone`, `stopped_at` | `0` mismatches |
| `C5` the `L12S` cells ran the registered configuration, halves sum, risk sums to `δ_k` | `0` mismatches |
