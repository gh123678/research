# FP-NET-BOUND-001 first result: the fixed-weight attention network reaches the exact computation's certified decisions, cell for cell

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`. Baseline: `4ff222e`.
Actor: Claude, under the user's direct instruction of 2026-09-13 ("好").
Task: [FP-NET-BOUND-001](../../research_tasks/FP-NET-BOUND-001.md) (hypotheses and bands registered
before the run).
Code: `evaluate_fp_net_bound_001.py`, `analyze_fp_net_bound_001.py`, `verify_fp_net_bound_001.py`.

**Preliminary result, single actor.**

## 1. Why this task exists

The project's question is whether a **fixed-weight softmax attention network** can turn in-context
experience into a **certified, non-degrading** policy improvement. Two lines had been built separately
and never met:

- the **certificate line**, which ended at `FP-ITER-BOUND-002` with a model-free bound `39.6%` tighter
  than the frozen one and a certified iteration of `99` steps against `36`;
- the **network line**, whose last statement (`FP-ATTN-8X-001`, `FP-CERTCHECK-001`) was agreement with
  the array formula under the *old, looser* certificate.

`FP-BOUND-001`/`002`/`003` and `FP-ITER-BOUND-001`/`002` were all measured on the numpy array routes.
This task runs the same loop with the **literal attention network** as the `Q̂` producer, and asks the
three questions that matter:

1. does the certificate still cover **the network's own** error, not numpy's?
2. does the tightened certificate transfer to the network?
3. does the network reach the **same certified decisions** as the exact computation?

## 2. Design

`48` record-routes (`task_index 12..23` × both mixings × both routes), `K = 16`, `16384` chains × `64`
per step, `δ_k = 0.05/K`. Eight cells share each step's certification batch and evolve independently:

```text
producer ∈ {numpy (sealed array routes), network (model.py, literal, from Q_0 = 0, LAYERS = 160)}
route    ∈ {expected_exact (masked softmax), expected_finite (finite softmax)}
lever    ∈ {frozen, L12M(0.05)}
```

The network is the structural `EndToEnd*SoftmaxExpectedSARSA` module with `(γ, α)` — no training, no
gradient step, weights fixed by construction. Every audit compares `E_Q` against the realized error of
**that cell's own `Q̂`**; numpy's error is never used as evidence about the network, or the reverse.

## 3. Results

| cell | emitted | mean length | reached `K` | mean value gain | coverage viol. | degrading |
|---|---:|---:|---:|---:|---:|---:|
| `numpy|frozen` | `36` | `1.750` | `0` | `1.7341` | `0` | `0` |
| `network|frozen` | `36` | `1.750` | `0` | `1.7341` | `0` | `0` |
| `numpy|L12M` | **`99`** | `3.062` | `0` | **`2.9014`** | `0` | `0` |
| `network|L12M` | **`99`** | `3.062` | `0` | **`2.9014`** | `0` | `0` |

**The two producers are indistinguishable at every level that matters.** Not merely close: the
per-record emission counts are identical on all `24` records, the totals are identical (`36` and `99`),
and the mean value gains agree to seven significant figures (`1.7340969749` vs `1.7340968201`; the
`1.5e-7` gap is float32 network arithmetic against float64 array arithmetic). The `E_Q/h` distributions
coincide exactly, median `0.8496` under `L12M` and `1.1416` under `frozen`, identical ranges.

## 4. Verdicts

| hypothesis | verdict |
|---|---|
| `H1`① no componentwise degradation in any cell | **PASS** (`0` in all four cells) |
| `H1`② `E_Q ≥` the **producer's own** realized `‖Q̂−Q^π‖∞` at every step | **PASS** (`0` violations in all four cells) |
| `H2` network emissions `≥ 1.10 ×` its frozen arm | **PASS** (`36 → 99`, `2.75×`) |
| `H3` network value gain `≥ 1.20 ×` its frozen arm | **PASS** (`1.673×`) |
| `H4` network vs numpy emission counts within `±20%` | **PASS** (`0.0%` gap at both levers) |
| `H5` step-1 decision agreement `≥ 80%` | **PASS** (`48/48 = 100%` at both levers) |
| `H6` network realized error `≤ 3 ×` numpy's | **PASS** (ratio `1.000`) |

## 5. What this answers

Read against the three questions in §1:

1. **The certificate covers the network.** `0` coverage violations across every step of every network
   cell, audited against the network's own realized error. The bound's premises (`|Y| ≤ E_eff`, iid
   first-visit samples, `Q̂` independent of the certification batch) are producer-agnostic, and the
   measurement confirms it: nothing about the certificate needed to know that a network produced `Q̂`.
2. **The tightening transfers.** The network's certified iteration goes `36 → 99` steps and
   `1.734 → 2.901` mean value gain, exactly the model-free gains the numpy line measured — with `0`
   degradations at every emitted step.
3. **The network reaches the exact computation's decisions.** `100%` step-1 agreement and identical
   per-record emission counts. On this population and protocol, the fixed-weight softmax attention
   network is not an approximation to the certified decision procedure — it **is** the certified
   decision procedure, up to float32 rounding.

**This is the project's headline question answered at the decision level**: a fixed-weight attention
network, run purely in-context over behaviour data, supports a certified non-degrading policy
improvement whose decisions match the exact array computation cell for cell — under a certificate
`39.6%` tighter than any available when the network line was last measured.

## 6. What this does not establish

- **The eight cells are still step-limited by the population, not by the producer.** `0` trajectories
  reach `K = 16`, exactly as in the numpy-only run, and `FP-ITER-BOUND-002` already showed that half
  the stops are convergence-driven rather than certificate-driven. The network inherits that limit; it
  does not remove it.
- **No new independence.** The network agreement is a *within-executor* reproduction of the array
  formula, and the array formula is what the certificate was derived against. A third party is still
  what this line lacks.
- **Nothing about other families or dimensions.** `task_index 12..23`, `4` states, `3` actions,
  `γ = 0.7`, `π_min = 0.15`. Cross-family and cross-dimension questions are `FP-XFAM-001`/`FP-NETX-001`
  territory and are untouched here.
- **The `network` producer is float32.** Its agreement with numpy is bounded by that, `~1e-6`; a
  certificate quoted to `1e-15` would not be reproducible by the network, and none is claimed.

**Same-actor throughout**, per the user's standing instruction.
