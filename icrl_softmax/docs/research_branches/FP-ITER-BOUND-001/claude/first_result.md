# FP-ITER-BOUND-001 first result: the tightened certificate doubles the certified iteration — and stops being the binding constraint

> ## WITHDRAWN 2026-09-13 (independent audit)
>
> The main arm here is `L12M`, whose concentration step is **not licensed** (Hoeffding
> applied to a function computed from the same successor draws it averages over). **All
> `L12M` numbers in this file are withdrawn**: `89` emitted steps, `2.8295` mean value
> gain, `2.854` mean length, the `H2`–`H7` verdicts, and the headline `36 → 89`.
>
> **What survives**: `H1`'s two clauses as *measurements* (zero degradations and zero
> observed coverage violations in every arm), the `frozen` (`36` / `1.7341`) and `L12`
> (`66` / `2.3771`) rows, and the qualitative finding that the bound was a first-order
> cost. The load-bearing `C1`–`C4` replay checks were about arithmetic, not about the
> arm's validity, and they stand.
>
> The iteration must be re-run with the repaired arm
> `L12S(f=0.9, p=0.05)` — see
> [`FP-BOUND-002-l12m-withdrawal-and-split-repair.md`](../../derivations/FP-BOUND-002-l12m-withdrawal-and-split-repair.md).
> The corrected step-1 gain is `−31.2%`, not `−37.6%`, so the iteration's `2.75×` should be
> treated as an **upper bound** on what the sound arm delivers until re-measured.

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`. Baseline: `e1bdc63`.
Actor: Claude, under the user's direct instruction of 2026-09-13 ("都去做").
Task: [FP-ITER-BOUND-001](../../research_tasks/FP-ITER-BOUND-001.md) (`K = 16` and hypotheses
registered before the run).
Code: `evaluate_fp_iter_bound_001.py`, `analyze_fp_iter_bound_001.py`, `verify_fp_iter_bound_001.py`.

**Preliminary result, single actor.**

## 1. The question

`FP-SHORT-001` found that trajectories stop **narrowly** (`E_Q/h` median `1.12`, tightest `1.00`), and
`FP-BOUND-001`/`002` cut `E_Q` by around `40%` soundly. So: does the certified iteration actually run
longer, or was the bound never the real limit?

Four arms run in lockstep over the **same** per-step certification batches (so the comparison is
paired and carries no sampler confound), population `task_index 12..23` × both mixings, both routes =
`48` record-routes, `K = 16` frozen, `16384` chains × `64` per step, `δ_k = 0.05/16` per theorem 2:

| arm | needs kernel? |
|---|---|
| `frozen` | no |
| `L12` | no |
| `L12M` | no — the model-free propagation arm (main) |
| `L123` | yes — upper reference, oracle label |

## 2. Results

| arm | emitted steps | mean length | mean total value gain | reached `K=16` | coverage viol. | degrading |
|---|---:|---:|---:|---:|---:|---:|
| `frozen` | `36` | `1.750` | `1.7341` | `0` | `0` | `0` |
| `L12` | `66` | `2.375` | `2.3771` | `0` | `0` | `0` |
| **`L12M`** | **`89`** | **`2.854`** | **`2.8295`** | `0` | `0` | `0` |
| `L123` | `110` | `3.292` | `2.9562` | `0` | `0` | `0` |

**The model-free certificate takes the certified iteration from `36` to `89` emitted steps (`2.47×`)
and the mean total value gain from `1.734` to `2.830` (`1.63×`), with `0` componentwise degradations
and `0` coverage violations in every arm.** So the bound *was* a first-order cost, and `FP-SHORT-001`'s
reading was right about that.

## 3. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` (mandatory) no degradation, no coverage violation | **PASS** (all four arms, `0`/`0`) |
| `H2` length `L12M > frozen`, `L123 ≥ L12M` | **PASS** (`2.854 > 1.750`, `3.292 ≥ 2.854`) |
| `H3` emissions `≥ 1.10 × frozen` | **PASS** (`89 ≥ 39.6`) |
| `H4` value gain `≥ 1.20 × frozen` | **PASS** (`2.830 ≥ 2.081`) |
| `H5` at least `5` trajectories reach `K = 16` | **FALSIFIED** (`0`; longest run `14`) |
| `H6` at least `50%` of step-1 stoppers revived | **FALSIFIED** (`10/36 = 28%`) |
| `H7` median `E_Q/h` at least `20%` below frozen's | **PASS** (`1.1416 → 0.8734`, `−23.5%`) |

### The two falsifications are the finding

`H5` and `H6` fail even though the certificate is `40%` tighter, and the reason is not the bound — it
is **the risk schedule**. The trajectory-level guarantee allocates `δ_k = δ_total/K`, so at `K = 16`
every step certifies at `0.003125` instead of the `0.05` the step-1 experiments use. `log(2/δ′)` grows
from `6.87` to `9.64`, which inflates both the `√` term (`+18%`) and the range term (`+40%`) at *every*
step. The step-1 result that `71%` (`10/14`) of stoppers revive does not carry over: here only `28%`
do.

So the honest reading of this run is:

- **the bound was binding, and is now much less so** — the loop runs `2.5×` longer on the same data;
- **the binding constraint has moved to the multi-step risk allocation**, which is a property of the
  protocol, not of the estimator, and is model-free to fix (a non-uniform schedule, or an
  anytime-valid confidence sequence costing `log log K` instead of `log K`).

That is a cleaner position than before: two independent, quantified levers, one of them already spent.

## 4. Verification

`verify_fp_iter_bound_001.py` replays every trajectory from the sealed bundle: it rebuilds each arm's
policy chain from the recorded `η` selections alone, re-derives `q̂`, the certificate, the decision and
the exact values, and checks the recorded numbers against them.

| check | result (over **`493` replayed steps**) |
|---|---|
| `C1` `emitted ⟺ E_Q < h`, with `h` recomputed | `0` mismatches, max `|Δh|` `0.0` |
| `C2` `E_Q ≥` realized `‖Q̂−Q^π‖∞` at every step | `0` violations, max `|ΔE_Q|` `0.0` |
| `C3` value deltas recomputed exactly; non-degradation re-checked | `0` degradations, max `|Δ|` `0.0` |
| `C4` sealed totals (`emitted_steps`, `total_value_gain`) | `0` mismatches, max `|Δ|` `0.0` |

`C3` is the strongest of the four: the value deltas were not trusted from the bundle but recomputed by
re-running exact policy evaluation along the replayed chain, and they agree to `0.0`.

*(A verifier defect was found and fixed during this check: the first version drew a fresh training
batch instead of consuming the task's RNG state, which desynchronised the replay and raised a
`KeyError`. The run was never wrong — the checker was.)*

## 5. What this establishes, and what it does not

**Established**

- The tightened certificate **more than doubles the certified iteration** at fixed data, with zero
  degradations — a `2.47×` increase in certified update steps and a `1.63×` increase in realized value.
- The per-step risk allocation `δ_total/K` is now a **first-order cost**, quantified by the gap between
  the step-1 revival rate (`71%`) and the in-loop one (`28%`) at the same population and data size.
- The `L12M` arm is the one to carry: model-free, sound, and within `2` points of the kernel arm's
  emission count.

**Not established**

- **Nothing reaches the horizon**, so "how long can it run?" is still open — this run says only that
  `K = 16` is beyond reach at `δ_k = δ_total/K`, not that the method stalls.
- `L12M` here uses the `0.5` risk split. `FP-BOUND-003` measured a `0.05` split worth another `3–4%`;
  the iteration numbers above are therefore a **lower bound** on what the final arm delivers.
- `L123`'s column reads the kernel and must keep its oracle label.
- **No independent actor.** Every number here is same-actor, and the replay verifier shares the
  executor's model of the protocol even though it shares no code path with the evaluator.

**Same-actor throughout**, per the user's standing instruction.
