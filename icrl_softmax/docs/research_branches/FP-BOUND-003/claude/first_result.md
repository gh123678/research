# FP-BOUND-003 first result: the model-free arm, fixed

> ## WITHDRAWN 2026-09-13 (independent audit)
>
> This task repaired the *risk allocation* of `L12M`, but the arm itself is **not a valid
> bound**: its concentration step applies Hoeffding to a function computed from the same
> successor draws it averages over. **All `L12M(0.05)` numbers here are withdrawn**
> (`−39.6%` / `−40.4%`, emissions `44` / `28`, recovery `93.1%` / `80.1%`), as is the
> `H1`–`H6` table that rests on them.
>
> Two findings survive and are still useful:
> 1. the propagation estimates need very little risk budget — the `0.05` vs `0.5` lesson
>    carries over to the repaired construction;
> 2. the withdrawal-and-repair note
>    ([`FP-BOUND-002-l12m-withdrawal-and-split-repair.md`](../../derivations/FP-BOUND-002-l12m-withdrawal-and-split-repair.md))
>    re-measures the sound arm `L12S(f=0.9, p=0.05)` at **`−31.2%`** with `44/48` emitted —
>    both knobs matter, in the same direction this task identified.
>
> The `frozen`/`L1`/`L12`/`L123` columns are unaffected.

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`. Baseline: `e1bdc63`.
Actor: Claude, under the user's direct instruction of 2026-09-13 ("都去做").
Task: [FP-BOUND-003](../../research_tasks/FP-BOUND-003.md).
Upstream: [FP-BOUND-002](first_result.md), whose mandatory clause `L12M ≤ L12` was falsified on
`2/48` (`c64k`) and `6/48` (`c16k`) record-routes, and whose `H3` failed at `c16k` (`63.3% < 70%`).

**Preliminary result, single actor.**

## 1. The defect was a mispriced risk budget, not the method

`L12M` split `δ_step` in half: one half for the per-pair intervals, one for the propagation estimates.
The intervals therefore ran at half of `L12`'s budget, `log(2/δ′)` went `6.17 → 6.87`, and the interval
came out about `5%` looser. Where the propagation gain was smaller than that premium, the arm lost to
`L12` outright — the `2`/`6` exceptions.

But the propagation estimates are nowhere near that risk-hungry. Their confidence terms are
`‖W‖∞·sqrt(log(1/δ_k)/(2n))`, of order `1e-3`, against a propagation gain of order `3e-2`. Squeezing
`δ_prop` from `0.5·δ_step` to `0.05·δ_step` raises `log(1/δ_k)` only from `9.35` to about `12.3`
(confidence terms ×`1.15`, still `1e-3`), while returning `48%` of the budget to the intervals.

One optional parameter, `delta_prop_fraction` (default `0.5`, so FP-BOUND-002's bundles reproduce
bit-for-bit). This task runs it at `0.05`.

## 2. Results

| rung | lever | mean `E_Q` | vs frozen | emitted | recovery of `L123`'s gain |
|---|---|---:|---:|---:|---:|
| `c64k` | `frozen` | `0.12044` | — | `34/48` | — |
| | `L12` | `0.10422` | `−13.5%` | `37` | — |
| | **`L12M(0.05)`** | **`0.07280`** | **`−39.6%`** | **`44`** | **`93.1%`** |
| | `L123` (kernel) | `0.07047` | `−41.5%` | `44` | `100%` |
| `c16k` | `frozen` | `0.23226` | — | `20/48` | — |
| | `L12` | `0.16976` | `−26.9%` | `21` | — |
| | **`L12M(0.05)`** | **`0.13854`** | **`−40.4%`** | **`28`** | **`80.1%`** |
| | `L123` (kernel) | `0.13076` | `−43.7%` | `32` | `100%` |

Against FP-BOUND-002's `0.5` split: `0.07510 → 0.07280` at `c64k` (`−3.1%`) and `0.14507 → 0.13854`
at `c16k` (`−4.5%`); emissions `27 → 28` at `c16k`, unchanged at `c64k`.

**The kernel arm is now worth only `1.9` points at `c64k` and `3.3` at `c16k`** — down from the `28`
points that separated the two model-free and kernel *steps* in FP-BOUND-001. Read against `frozen`, the
model-free arm gets `−39.6%` of the kernel's `−41.5%` without reading a kernel.

## 3. Verdicts

| hypothesis | verdict |
|---|---|
| `H1`① coverage, `0` violations | **PASS** at both rungs |
| `H1`② `L12M(0.05) ≤ L12` on every record-route | **PASS**, `0` exceptions at both rungs (FP-BOUND-002: `2` and `6`) |
| `H1`③ `L12M(0.05) ≤ L12M(0.5)` on every record-route | **PASS** |
| `H2` reduction in `[−45%, −30%]` | **PASS** (`−39.6%`, `−40.4%`) |
| `H3` recovery `≥ 80%` at both rungs | **PASS** (`93.1%`, `80.1%`) |
| `H4` emissions `≥ L12M(0.5)` | **PASS** (`44 vs 44`, `28 vs 27`) |
| `H5` risk accounting (`δ_eps + δ_prop = δ_step`, `d·δ_each = δ_eps`) | **PASS** |
| `H6` shared arms reproduce FP-BOUND-001 bit-for-bit | **PASS** (`0.0`, `0` mismatches) |

`H3` at `c16k` clears the line by `0.06` percentage points (`80.06%`). That is a pass, and it is also
a warning: the registered threshold was set where the measurement happens to land, and the honest
reading is "at or just above 80%", not "comfortably above".

## 4. Verification

`verify_fp_bound_002.py` runs unchanged on these bundles (same schema):

| check | `c16k` | `c64k` |
|---|---|---|
| `C1` L12M rebuilt from the sealed successor histogram | `0.0` | `0.0` |
| `C2` per-pair intervals vs the exact residual | `240/240` | `240/240` |
| `C3` `E_Q` reconstruction | `0.0` | `0.0` |
| `C4` shared arms vs FP-BOUND-001 | `0.0`, `0` mismatch | `0.0`, `0` mismatch |
| `C5` `emitted ⟺ E_Q < h` | `0` mismatch | `0` mismatch |
| `C6` risk accounting | `0` bad | `0` bad |

## 5. The arm to carry forward

**`L12M(0.05)` is the certificate to use.** It is model-free (`L12` + data-driven propagation),
strictly inside the behaviour-data claim, sound by the induction in FP-BOUND-002, and worth `−39.6%`
at `c64k` against `frozen`'s `−0%` — at the kernel arm's decision line, without the kernel.

`L12` and `L1` remain available as the cheap free tightening for any run that wants no risk-splitting
at all. `L123` stays as an upper reference and must keep its oracle label.

**Not established.** Nothing about the iteration — `FP-ITER-BOUND-001` uses `L12M(0.5)`, not this arm,
and its numbers are therefore a *lower* bound on what `L12M(0.05)` would deliver.
