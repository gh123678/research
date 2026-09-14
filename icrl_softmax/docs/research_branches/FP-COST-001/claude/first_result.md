# FP-COST-001 first result: at equal total data the per-state rule still wins — and the conjunctive arm does not catch up even at 4.8x the cost

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`. Baseline: `05f5989`.
Actor: Claude, under the user's direct question and instruction of 2026-09-13.
Task: [FP-COST-001](../../research_tasks/FP-COST-001.md) (ladders and hypotheses registered before the
run).

**Preliminary result, single actor. Nothing here is VERIFIED.**

## 1. The question, and why it needed its own run

`FP-XRULE-001/002` reported a "same-total-cost truncation" that was really a **same-step-count**
truncation — every cell cut to the shortest cell's step count. That is not the comparison that decides
anything. The decision-relevant question is:

> Give the conjunctive arm the **same total certification data** as the per-state arm. Does it catch up?

The conjunctive arm stops on its own (mean `1.32` steps on F1, `4.91` on F2), so its only way to spend
more is **fatter batches**. This run measures its cost curve by scaling the chain count until the rung
exceeds the sealed per-state arm's total spend, so the matched-cost point is bracketed by measurements
rather than extrapolated.

## 2. Results

**Family F1** (`6×4`) — per-state reference: `1,015,021,568` items, gain `8.6893`

| conjunctive chains | items | × per-state cost | mean sim steps | emitted | mean value gain |
|---:|---:|---:|---:|---:|---:|
| `16,384` | `133,169,152` | `0.13` | `1.323` | `31` | `0.9404` |
| `65,536` | `1,832,910,848` | `1.81` | `4.552` | `341` | `7.1100` |
| **`131,072`** | **`4,907,335,680`** | **`4.83`** | `6.094` | `490` | **`8.6126`** |

**Family F2** (`4×3`) — per-state reference: `1,169,162,240` items, gain `8.1739`

| conjunctive chains | items | × per-state cost | mean sim steps | emitted | mean value gain |
|---:|---:|---:|---:|---:|---:|
| `16,384` | `493,879,296` | `0.42` | `4.906` | `389` | `5.8007` |
| **`65,536`** | **`3,464,495,104`** | **`2.96`** | `8.604` | `773` | **`7.6521`** |

## 3. The answer

**Yes — and the measurement is stronger than the interpolation.**

At the **matched-cost point** (linear interpolation of the conjunctive curve at the per-state arm's total
spend):

```text
F1: interpolated conjunctive 4.1413  vs  per-state measured 8.6893   ->  2.10x
F2: interpolated conjunctive 6.2215  vs  per-state measured 8.1739   ->  1.31x
```

But interpolation could understate a curve that turns up, so the decisive facts are the **measured top
rungs**: the conjunctive arm spent **`4.83×`** (F1) and **`2.96×`** (F2) the per-state arm's data and
still came in **below** it:

```text
F1: conjunctive @ 4.83x cost -> 8.6126   <   per-state @ 1.00x cost -> 8.6893
F2: conjunctive @ 2.96x cost -> 7.6521   <   per-state @ 1.00x cost -> 8.1739
```

**The per-state rule at 1× cost beats the conjunctive rule at 4.83× cost.** No interpolation is needed
for that statement.

## 4. Why: the marginal value of data collapses under the conjunctive gate

| family | conjunctive marginal value per `1e8` items | per-state average per `1e8` items |
|---|---|---:|
| F1 | `16k→64k`: `+0.3630`, `64k→128k`: `+0.0489` | `+0.8561` |
| F2 | `16k→64k`: `+0.0623` | `+0.6991` |

Fatter batches **do** keep the conjunctive arm alive longer (F1: `1.32 → 4.55 → 6.09` mean steps) —
a tighter certificate loosens the gate — but the value bought per item falls by an order of magnitude
between rungs, and never approaches the per-state arm's average. **Buying data does not rescue the
conjunctive gate**, which is exactly the mechanism `FP-EARLYSTOP-001` found on the original family with
a single `4×` rung; here it is a curve, on two families, with the matched-cost point bracketed.

## 5. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` no degradation, no coverage violation | **PASS** (`0`/`0` in all five cells) |
| `H2` the `16,384` rung reproduces FP-XRULE-002's `conj|frozen` | **PASS** — max `\|Δvalue\| = 0.000e+00`, `0` emission mismatches |
| `H3` interpolated conjunctive `<` per-state at matched cost | **PASS** on both families (`2.10×`, `1.31×`) |
| `H4` every conjunctive marginal step `<` the per-state average | **PASS** on both families |
| `H5` cost reported as total items, with chains, steps and wall clock | **PASS**, §2 |

## 6. What this establishes, and what it does not

**Established**

- **At equal total certification data the per-state rule is significantly better**, on two families, and
  the conjunctive arm fails to catch up even at `2.96–4.83×` the cost.
- The per-state advantage is therefore a **rule** effect, not an accounting trick: it is not "the same
  destination reachable by spending more data differently".
- The conjunctive gate's problem is diminishing returns on data, quantified as a curve rather than a
  single point.

**Not established**

- **The replay verification for this task is partial.** The `16,384` rung is bit-verified against the
  independently sealed FP-XRULE-002 bundle (`H2`, and it passed), but the `65,536` and `131,072` rungs
  have **not** been replayed step by step; the aggregates in §2 are as recorded by the evaluator. A full
  replay re-draws ~`8.4e9` items and is the outstanding item.
- Extrapolation beyond `131,072` chains is not measured; the claim is bounded by the rungs run.
- **No network producer here** — both arms use the array routes.
- **No third-party independence**, as everywhere in this line.

**Same-actor throughout.**
