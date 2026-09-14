# FP-RERUN-001 first result: the sound arm is dominated by the free one — the propagation lever closes

> **⛔ WITHDRAWN 2026-09-14 (third-party review, FAIL).** `L12S` is **not a licensed bound**. The
> split protects only the `k = 0` step: `V_0` is fixed given half A and `B ⊥ A`, so `T^B(V_0)` may
> be concentrated, but `V_1` itself depends on B, so every later step applies Hoeffding to a
> function computed from the same B — the `L12M` defect, entering one iteration later. Only
> `n_iter = 1` is licensed; the runs used `n_iter = 12`. See
> [`docs/derivations/FP-L12S-REVIEW-001-withdrawal.md`](../../../derivations/FP-L12S-REVIEW-001-withdrawal.md).
> The **measurements** in this file stand; any "sound/licensed" characterisation of `L12S` does not.

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`. Baseline: `b2c39a0`.
Actor: Claude, under the user's direct instruction of 2026-09-13 ("好去做"), executing item 3 of the
independent audit's ordering.
Task: [FP-RERUN-001](../../research_tasks/FP-RERUN-001.md) (hypotheses, arms and the `L12S` grid cell
registered before the run).
Upstream: [`FP-BOUND-002-l12m-withdrawal-and-split-repair.md`](../../derivations/FP-BOUND-002-l12m-withdrawal-and-split-repair.md).

**Preliminary result, single actor. Nothing here is VERIFIED.**

## 1. What was re-run and why

The audit found `L12M`'s concentration step unlicensed; the repair (`L12S`) splits each pair's
first-visit sample so the intervals come from half A and every propagation estimate from half B. On
step 1 at `c64k` the repaired arm is worth `−31.2%` against the withdrawn `−39.6%`.

This task puts the repaired arm back into the certified iteration, with the audit's three missing
instruments attached: **actual certification data cost**, **wall clock**, and a **per-step audit**.

Eight cells, all sharing each step's batch so the comparison is paired:

```text
producer ∈ {numpy, network (model.py literal net, Q_0 = 0, LAYERS = 160)}
lever    ∈ {frozen, L12, L12S(f=0.9, p=0.05), L123 (kernel, oracle reference)}
K = 16, 16384 chains × 64 per step, δ_k = 0.05/16, 48 record-routes, task_index 12..23
```

## 2. Results

| cell | emitted | mean length | mean value gain | items if run alone | × frozen | cov. viol. | degrading |
|---|---:|---:|---:|---:|---:|---:|---:|
| `numpy|frozen` | `36` | `1.750` | `1.7341` | `88,080,384` | `1.00` | `0` | `0` |
| `numpy|L12` | **`66`** | `2.375` | **`2.3771`** | `119,537,664` | `1.36` | `0` | `0` |
| `numpy|L12S` | `60` | `2.250` | `2.2590` | `113,246,208` | `1.29` | `0` | `0` |
| `numpy|L123` *(kernel)* | `110` | `3.292` | `2.9562` | `165,675,008` | `1.88` | `0` | `0` |
| `network|` all four | **identical to numpy, cell for cell** | | | | | `0` | `0` |

**Per-step certification budget (late addition, 2026-09-14 — see the `H5` row in §3):** every cell drew
`16,384 chains × 64 steps = ` **`1,048,576` items per simulated step**, recorded per step as
`items_this_step` and confirmed by the replay (`C5`). The "items if run alone" column above is that
number times the cell's simulated step count.

Total wall clock for the eight-cell run: **`650 s` (`10.8 min`)**.

**The sound propagation arm is dominated by the free one.** `L12` — the envelope fix plus the full
risk budget, no propagation, no split, no extra machinery — emits `66` steps and `2.377` value.
`L12S` emits `60` and `2.259`, and uses *less* data precisely because it stops sooner. The extra
machinery costs data and returns nothing.

## 3. Verdicts

| hypothesis | verdict |
|---|---|
| `H1`① no componentwise degradation in any cell | **PASS** (`0` in all eight) |
| `H1`② `E_Q ≥` each cell's own producer's realized `‖Q̂−Q^π‖∞` | **PASS** (`0` violations in all eight) |
| `H2` `L12S ≥ 1.30 × frozen` | **PASS** on both clauses — `1.67×` emissions, `1.30×` value (the value clause clears by `0.003`) |
| `H3` `L12S ≥ L12` | **FALSIFIED** — `60 < 66` |
| `H4` producer agreement | **PASS** — `0.0%` gap on every lever, `48/48` step-1 agreement on every lever |
| `H5` (as registered: **report** the per-step budget, the items-if-run-alone, and the same-total-cost truncation) | **PASS：交付要求满足** — scored item by item under the user's ruling of 2026-09-14. (b) items-if-run-alone: delivered in §2. (c) same-total-cost truncation: delivered in §4. **(a) per-step budget: ABSENT from this report as first written; supplied below and marked as a late addition** — it was in the bundle (`items_per_step`) and printed by the analyzer, but not stated in the text. The value threshold I had substituted for `H5` is **not a registered hypothesis**; it is recorded as a post-hoc observation only |
| `H6` `L12S ≤` the withdrawn `L12M`'s `99` | **INVALID — 任务条款冲突且判据不能检验所称目标.** It requires citing a withdrawn number while this task's prohibitions forbid citing one, and `60 ≤ 99` cannot test the batch consistency it was meant to test. Acceptance force revoked under the user's ruling; the original clause and its recorded value are retained for traceability. **The batch consistency of this run against the withdrawn `L12M` run remains unchecked and is NOT superseded by any other check** — in particular not by FP-COST-001's `H2`, which compares two *different* runs (FP-COST-001's own conjunctive 16k ladder against FP-XRULE-002) and says nothing about this task |

`H3` is the result. It was registered because the step-1 grid had `L12S` beating `L12` by `20%`; in the
loop the ordering reverses. The reason is visible in the two budgets: at step 1 the certificate runs at
`δ = 0.05`, in the loop at `δ_k = 0.05/16`, and the split's cost (an interval computed on a fraction of
the sample) does not shrink with `δ` while the propagation's gain does.

## 4. Cost, truncation, and what stops the loop

**Same-total-data comparison** (truncate every cell to the number of steps `frozen` actually simulated,
per record-route, i.e. a mean of `1.750` steps):

| cell | emitted | mean value gain |
|---|---:|---:|
| `frozen` (reference) | `36` | `1.7341` |
| `numpy|L12` | `49` | **`2.1175`** |
| `numpy|L12S` | `43` | `1.9302` |
| `network|L12S` | `43` | `1.9302` |

At equal total data `L12` still leads, and both beat `frozen`. So the iteration gain reported here is
**not** an artifact of spending more data — but the sound propagation arm contributes nothing to it.

**Stop decomposition** (median `E_Q/h` over the stops of each cell, and the share within `1.25` of the
line, i.e. stops the certificate is still responsible for):

```text
frozen   stops 48   median 1.7324   within 1.25: 15 (31%)
L12S     stops 48   median 1.4107   within 1.25: 19 (40%)
L123     stops 48   median 1.1849   within 1.25: 28 (58%)
```

The loop is still substantially certificate-driven at every arm, and the kernel arm — which cannot be
used for a behavioural-data claim — is the only one where a majority of stops are near-line. The
`L12 → L123` gap (`66 → 110` emissions) is what a sound model-free propagation would have to capture,
and `L12S` does not capture it.

## 5. What this closes

- **The propagation lever is closed as a negative result.** Both its forms are now measured: the
  unlicensed `L12M` (`99` emissions, withdrawn) and the licensed `L12S` (`60`, dominated by the free
  `L12` at `66`). The gain that made the lever look attractive was an artifact of the invalid
  concentration step.
- **What survives, and is what should be used**: `L1` (`−11.0%` on `E_Q`) and `L12` (`−13.5%`) — both
  free, both model-free, neither touching the defective step. In the loop, `L12` takes the certified
  iteration from `36` to `66` emitted steps (`1.83×`) and from `1.734` to `2.377` mean value gain
  (`1.37×`) at `1.36×` the data, with `0` degradations and `0` coverage violations.
- **The producer result is unchanged by the repair**: the attention network matches the array route
  cell for cell under every certificate tested, including the sound one.
- **Cost and wall clock are now on the record** for this run, as the audit required. They are *not*
  retroactively available for the earlier EARLYSTOP runs; that gap is stated rather than filled.

**Not established**: nothing about other MDP families or dimensions (audit item 4: that comes after
the proof work); no third-party independence; and the `L12 → L123` gap remains unexplained — a
different, cheaper propagation construction is not ruled out by this negative result, only these two.

**Same-actor throughout.**

## 6. Verification

`verify_fp_rerun_001.py` replays all eight cells from the sealed bundle alone: it rebuilds each cell's
policy chain from the recorded `η` selections, re-runs the **same producer** (array route or the literal
attention network) on the replayed policy, re-draws each step's batch from the step-indexed schedule,
and re-derives the certificate, the decision, the exact values and the cost fields.

| check | result over **`928` replayed steps** |
|---|---|
| `C1` producer output equals the sealed `q_hat` | max `\|Δ\|` **`0.0`**, `0` mismatches |
| `C2` `emitted ⟺ E_Q < h`, with `h` recomputed | `0` mismatches, max `\|Δh\|` `0.0` |
| `C3` `E_Q ≥` the cell's own producer's realized error | `0` violations, max `\|ΔE_Q\|` `0.0` |
| `C4` value deltas recomputed along the replayed chain | `0` degradations, max `\|Δ\|` `0.0` |
| `C5` sealed totals incl. `simulated_steps`, `items_if_run_alone`, `stopped_at` | `0` mismatches |
| `C6` the `L12S` cells really ran `(f=0.9, p=0.05)`, halves sum to the retained count, risk sums to `δ_step` | `0` mismatches |
| `C7` network vs numpy `q_hat` at the same replayed policy | max `1.06e-05` (float32) |

`C1` is the load-bearing one: the network's output reproduces the sealed values **bit for bit**, so the
replay is of the same computation and not merely of a similar one. `C6` is new and covers the repaired
arm's configuration, which the withdrawal made necessary.

