# FP-NETX-002 first result: the attention network keeps its decision-equivalence on two new MDP families

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`. Baseline: `0453a57`.
Actor: Claude, under the user's direct instruction of 2026-09-13 ("你先做"), closing the gap the audit's
item 3 left open: FP-XRULE-001/002 moved both mechanisms to two new families but used **only the array
routes**.
Task: [FP-NETX-002](../../research_tasks/FP-NETX-002.md) (hypotheses registered before the run).

**Preliminary result, single actor. Nothing here is VERIFIED.**

## 1. Design

Families, environments, seeds, routes, `K = 4`, per-step budget and the batch schedule are **identical to
FP-XRULE-001**; two things change:

- the rule is fixed to `perstate` (the rule that now carries this line), so the cells are
  `producer × certificate`;
- a `network` producer is added — `model.py`'s two `EndToEnd*ExpectedSARSA` modules, run from `Q_0 = 0`
  for `LAYERS = 160` layers through a dimension-generic port of `network_qhat` (the original hardcodes
  `4×3`; the modules themselves turned out to be dimension-agnostic, verified on `6×4` before the run).

Four cells, `96` route-records per family, sharing every step's batch.

## 2. Results

| family | cell | emitted | mean value gain | items | partial updates | cov. | deg. |
|---|---|---:|---:|---:|---:|---:|---:|
| **f1** (`6×4`, 24 pairs) | `numpy|frozen` | `382` | `8.3968` | `402,653,184` | `86%` | `0` | `0` |
| | `numpy|L12S` | `384` | `9.1144` | `402,653,184` | `63%` | `0` | `0` |
| | `network|frozen` | `382` | `8.3968` | `402,653,184` | `86%` | `0` | `0` |
| | **`network|L12S`** | **`384`** | **`9.1144`** | `402,653,184` | `63%` | `0` | `0` |
| **f2** (`4×3`, 12 pairs) | `numpy|frozen` | `384` | `7.8099` | `402,653,184` | `35%` | `0` | `0` |
| | `numpy|L12S` | `384` | `7.9108` | `402,653,184` | `22%` | `0` | `0` |
| | `network|frozen` | `384` | `7.8099` | `402,653,184` | `35%` | `0` | `0` |
| | **`network|L12S`** | **`384`** | **`7.9108`** | `402,653,184` | `22%` | `0` | `0` |

Wall clock `43.6 min`, `1,536` network forward passes. The two producers are **indistinguishable on
every cell** — identical emission counts, identical mean value gains, identical partial-update shares.

## 3. Verdicts

| hypothesis | verdict |
|---|---|
| `H1`① no componentwise degradation, any cell | **PASS** (`0` in all eight) |
| `H1`② `E_Q ≥` **each cell's own producer's** realized `‖Q̂−Q^π‖∞` | **PASS** (`0` violations in all eight) |
| `H2` producer agreement `≤ 20%` gap, `≥ 80%` step-1 agreement | **PASS** — `0.0%` gap and `96/96 = 100%` on every cell |
| `H3` network realized error `≤ 3×` numpy's | **PASS** (ratio `1.000` in both families) |
| `H4` built-in cross-check against FP-XRULE-001 | **PASS** — `max \|Δvalue\| = 0.000e+00`, `0` emission mismatches over all `384` cell-records |
| `H5` cost and wall clock reported | **PASS**, §2 |
| `H6` float32 gap `≤ 1e-4` | **PASS** — `1.286e-05` (F1) and `1.718e-05` (F2) |

`H4` is the one that makes the rest credible: because everything except the producer matches
FP-XRULE-001, the two `numpy` cells had to reproduce that task's `perstate|*` numbers **exactly**, and
they do, to the last bit. Had they not, the pipeline would have differed and this run would have been
void.

## 4. The `float32` question, answered

The network is `float32` against numpy's `float64`, so the gap is expected; the question was whether it
grows with the **pair count** (12 → 24), which would be a real scaling concern. It does not:

```text
F1: 24 pairs   max |q_net − q_numpy| = 1.286e-05
F2: 12 pairs   max |q_net − q_numpy| = 1.718e-05
ratio F1/F2 = 0.75
```

The larger family has the **smaller** gap. `4.7e-07`-scale differences in the inputs to a 160-layer
fixed-weight iteration are not amplified by the dimension, and no decision in `768` cell-records moved.

## 5. What this establishes, and what it does not

**Established**

- The producer-equivalence result **generalises across MDP families**: a fixed-weight softmax attention
  network's `Q̂` is decision-equivalent to the exact array formula's on `6×4` (24 pairs) and on a family
  with a different reward bound, under both certificates tested.
- The `float32` gap **does not grow with the number of state-action pairs**.
- The certificate covers the **network's own** error on the new families (`0` violations audited
  per-producer), and the per-state rule's partial-update mechanism reproduces with the network
  (`86%/63%` on F1, `35%/22%` on F2).

**Not established**

- **The network certifies nothing.** It produces `Q̂`; the certificate and the update decision are
  external code shared by both producers. What matches is the *input* to that procedure.
- **Nothing about longer horizons with the network.** This is `K = 4`; FP-XRULE-002's finding that the
  per-state loop has no stopping point by step 12 was measured on the array routes only.
- **No third-party independence**, which remains the one thing this line cannot supply for itself.

**Same-actor throughout.**

## 6. Verification

`verify_fp_netxfam_001.py` replays both families from the sealed bundle: it rebuilds each cell's policy
chain, re-runs **both producers** on the replayed policy (the array route and the 160-layer attention
network), re-draws each step's batch, and re-derives the certificate, the per-state decision, the exact
values and the cost fields. The batch is drawn once per `(record, step)` and reused across the four
cells.

| check | result over **`3,072` replayed steps** (`768` batch draws) |
|---|---|
| `C1` both producers reproduce the sealed `q_hat` | max `\|Δ\|` **`0.0`**, `0` mismatches |
| `C1b` network vs numpy `q_hat` at the same policy | max `1.72e-05` (float32) |
| `C2` emission and `min_lb` reproduced via the per-state rule | `0` mismatches, max `\|Δmin_lb\|` `0.0` |
| `C3` `E_Q ≥` the cell's own producer's realized error | `0` violations, max `\|ΔE_Q\|` `0.0` |
| `C4` value deltas recomputed along the replayed chain | `0` degradations, max `\|Δ\|` `0.0` |
| `C5` sealed totals incl. `simulated_steps`, `items_if_run_alone`, `stopped_at` | `0` mismatches |
| `C6` the `L12S` cells ran the registered configuration; halves sum; risk sums to `δ_k` | `0` mismatches |

`C1` is the load-bearing one here: the network's outputs were **recomputed from scratch** and match the
sealed values bit for bit, so the agreement with numpy in §2 is a property of the two computations and
not of a stored artefact.
