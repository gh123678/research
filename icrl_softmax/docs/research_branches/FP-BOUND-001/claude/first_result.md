# FP-BOUND-001 first result: the certificate's conservatism, priced and partly collected

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`. Baseline: `fe97063`.
Actor: Claude, under the user's direct instruction of 2026-09-13 ("动界比动数据划算…做"), continuing
from the independent review #2 finding that `E_Q` runs **5–10× larger than the realized error**.
Task: [FP-BOUND-001](../../research_tasks/FP-BOUND-001.md) (hypotheses and bands pre-registered before
the confirmatory run).
Derivation: this file, §2. Code: `fixed_policy_tight_certificate.py`, `evaluate_fp_bound_001.py`.

**Preliminary result, single actor.** The confirmatory population is fresh (see §3); the actor is not.

## 1. The question

Review #2 measured `realized/E_Q` at a median of `0.10–0.19` across the sealed FP-CERTFIX-001 rungs:
the certificate is sound but its radius is roughly an order of magnitude larger than the error it
bounds. `FP-SHORT-001` then showed that the trajectories which stop early stop **narrowly**
(`E_Q/h` median `1.12`, tightest `1.00`). Together those say the binding constraint is the bound, not
the data — adding data only buys `1/√n`.

So: how much of the conservatism is removable **soundly**, and what does it buy at the decision line?

## 2. Three levers, three prices

The frozen scalar is `E_Q = max_x(|mean_x| + t_x)/(1−γ)` with
`t_x = √(2 V_x log(2/δ′)/n_x) + (7/3)·R·log(2/δ′)/(n_x−1)`, `R = 2E = 20`, `δ′ = δ_step/(2d)`.

**L1 — the envelope is worst-case over `Qhat`, but `Qhat` is known. (model-free, zero cost)**
`E = R* + γB + B` with `B = R*/(1−γ)` only assumes `‖Q̂‖∞ ≤ B`. But `Q̂` is an *input* the certificate
already holds, so the true range of `Y = r + γV̂(s′) − Q̂(s,a)` is deterministically at most

```text
E_eff = R* + γ‖V̂‖∞ + ‖Q̂‖∞,      V̂ = π·Q̂.
```

Nothing is estimated and no probability is spent — it is the same Maurer–Pontil theorem with a smaller
**true** range. This is the lever `FP-RANGE-001` looked for and did not find: that task tried to shrink
the range with a data-driven truncation `τ` and paid a Cauchy–Schwarz bias for it; shrinking it with
the *known* `Q̂` costs nothing. Measured `E`: `10 → 4.36` (range `20 → 8.7`).

**L2 — the implementation never spends half its risk budget. (model-free, zero cost)**
Maurer–Pontil Theorem 4 is **two-sided** at level `δ`, but the frozen code allocates `δ′ = δ_step/(2d)`
and *then* reasons about "two directions", spending only `d·δ′ = δ_step/2`. Using `δ′ = δ_step/d`
spends the budget exactly; `log(2/δ′)` falls `6.87 → 6.17`.

**L3 — the `1/(1−γ)` step throws away the per-pair structure. (NEEDS THE TRANSITION KERNEL)**
`Q̂ − Q^π = −(I − γP^π)^{-1}ρ` is an **identity**, not an inequality. With the per-pair interval
`|ρ_x| ≤ ε_x`:

```text
w         = (I − γK)^{-1} ε̄,        K(s,s′) = Σ_a π(a|s) P(s′|s,a),
ε̄(s)      = Σ_a π(a|s) ε(s,a),
|u(s,a)|  ≤ ε(s,a) + γ Σ_{s′} P(s′|s,a) w(s′),
```

which is `≤ max_x ε_x/(1−γ)` with equality only when `ε` is constant across pairs. The gain is exactly
the non-uniformity of `ε` (measured median `max/min` ratio **4.40**). **This reads the transition
kernel and must be labelled as such everywhere**: it is the same family as the pre-registered
oracle-kernel arm, it corroborates, and it cannot support any claim about learning from behavioural
data alone.

*Reference only, also needs the kernel:* the exact support range of `Y_x` from the MDP tables
(`support_range`) — quoted to show how much of the range lever L1 leaves on the table.

## 3. Pilot, then a fresh confirmatory population

The pilot ran on the **sealed** `step1_fv_c64k` bundle (`diagnose_fp_bound_slack.py`, exploratory) and
is what set the registered bands: L1 `−11.1%`, L1+L3 `−38.1%`, all three `−40.2%`, and the prediction
that the oracle range adds only `~7%` over L1.

The confirmatory run is on **`task_index 12..23` × both mixings — the 24 environments FP-EARLYSTOP-001
opened and the pilot never touched**, so the bands are genuinely out-of-sample. All five levers are
computed on the **same** certification batch, so the comparison is paired and carries no sampler
confound. Two rungs, `c16k` and `c64k` chains × `64`.

## 4. Results

| rung | lever | mean `E_Q` | vs frozen | step-1 emitted | coverage violations |
|---|---|---:|---:|---:|---:|
| `c64k` | `frozen` | `0.12044` | — | `34/48` | `0` |
| | `L1` (model-free) | `0.10717` | **`−11.0%`** | `37` | `0` |
| | `L12` (model-free) | `0.10422` | **`−13.5%`** | `37` | `0` |
| | `L123` | `0.07047` | **`−41.5%`** | **`44`** | `0` |
| | `support_range` | `0.09855` | `−18.2%` | `38` | `0` |
| `c16k` | `frozen` | `0.23226` | — | `20/48` | `0` |
| | `L1` (model-free) | `0.17829` | **`−23.2%`** | `20` | `0` |
| | `L12` (model-free) | `0.16976` | **`−26.9%`** | `21` | `0` |
| | `L123` | `0.13076` | **`−43.7%`** | **`32`** | `0` |
| | `support_range` | `0.14444` | `−37.8%` | `28` | `0` |

**The model-free levers are worth more exactly where the certificate is weakest.** `L1` is `−11.0%` at
`c64k` but `−23.2%` at `c16k`, because the range enters the *linear* term `(7/3)R log/n`, which
dominates the radius at small `n`. Free tightening that helps most in the small-sample regime is
precisely what the iteration needs.

**The decision consequence.** At `c64k` the frozen certificate abstains on `14` record-routes with
`E_Q/h` median `1.363`; under `L123` that median falls to **`0.771`** and **`10 of the 14`** revive.
At `c16k` the abstaining population is `28` with median `1.926`; `L123` brings it to `1.046` and revives
`12 of 28` — close, but the median does **not** cross the line.

## 5. Verdicts against the pre-registered bands

| hypothesis | verdict |
|---|---|
| `H1` soundness + monotone chain `frozen ≥ L1 ≥ L12 ≥ L123` | **PASS** — `0` coverage violations on all `240` record-route-lever cells; `0` monotonicity exceptions; `L123 ≤ max_x ε_x/(1−γ)` everywhere |
| `H2` model-free reduction in `[−25%, −5%]` | **PASS** at `c64k` (`−13.5%`), **FALSIFIED** at `c16k` (`−26.9%`, i.e. *stronger* than registered) |
| `H3` all-lever reduction in `[−55%, −20%]` | **PASS** at both rungs (`−41.5%`, `−43.7%`) |
| `H4` emission no fall and at least `+3` | **PASS** at both (`34→44`, `20→32`) |
| `H5` median `E_Q/h` under `L123` below `1.0` | **PASS** at `c64k` (`0.771`), **FALSIFIED** at `c16k` (`1.046`) |
| `H6` oracle range adds `≤ 10%` over `L1` | **PASS** at `c64k` (`−8.0%`), **FALSIFIED** at `c16k` (`−19.0%`) |
| `H7` combined `≥ 80%` of the sum of the three | **PASS** at both (`41.5%` vs `46.1%`; `43.7%` vs `51.0%`) |

**The three falsifications are one finding, not three.** `H2`, `H5` and `H6` all fail at `c16k` and
pass at `c64k` for the same reason: at small `n` the linear range term dominates, so anything that
shrinks the range buys more, and the registered bands — set from the pilot at `c64k` — were too
conservative for that regime. None of them is a failure of the construction; all three are recorded
as falsified rather than re-banded.

## 6. Verification

`verify_fp_bound_001.py` recomputes everything from the sealed bundle plus the task's MDP, and shares
no code path with the evaluator's arithmetic:

| check | result |
|---|---|
| `C1` per-pair coverage against the **exact** residual `ρ_x` (`240` record-route-lever cells × `12` pairs) | `240/240` pass, `0` violations |
| `C2` `E_Q` rebuilt from the sealed per-pair means/radii (`240` checks) | max `|Δ|` `0.0` |
| `C3` `K` and `(I−γK)^{-1}` rebuilt; identity `Q̂−Q^π = −(I−γP^π)^{-1}ρ` checked numerically | max `|Δ|` `0.0`; identity residual `3.9e-15` |
| `C4` envelope `y_range == 2(R* + γ‖V̂‖∞ + ‖Q̂‖∞)` | max `|Δ|` `0.0` |
| `C5` `h` recomputed; `emitted ⟺ E_Q < h` | max `|Δh|` `0.0`; `0` emission mismatches |
| `C6` risk accounting (`2d·δ_each` frozen, `d·δ_each` full-budget) | `0` bad |

`C3` is the check that matters most: the identity is the lemma L3 rests on, and it is confirmed to
`4e-15` on all `240` cells. `C1` is a far more sensitive coverage test than the usual sup-norm audit —
it checks every one of the `2,880` pair-level intervals, not just the maximum.

Determinism: both rungs were re-run after the bundle gained the per-pair arrays, and `summary.json` is
byte-identical before and after.

## 7. What this establishes, and what it does not

**Established**

- Three **provable** slack sources in the residual certificate, each located, priced and measured.
- **`L1` and `L2` are free**: model-free, zero probability cost, no new premise. They cut `E_Q` by
  `13.5%` at `c64k` and `26.9%` at `c16k`, and they should be adopted unconditionally in every
  downstream run.
- **`L123` cuts `E_Q` by `41.5%`** at the same data and takes step-1 emissions from `34/48` to `44/48`
  on environments nobody had tuned on, reviving `10` of the `14` abstainers.
- The **range lever is real and is captured without the kernel**: the oracle support range adds only
  `8%` over `L1` at `c64k`. `FP-RANGE-001`'s negative result was about *estimating* the range, not
  about the range being unavailable.
- The three levers compose nearly additively (`41.5%` combined against `46.1%` summed).

**Not established**

- **`L3` needs the transition kernel.** It is oracle-assisted, exactly like the pre-registered
  oracle-kernel arm, and every number in its column carries that label. It cannot support the
  project's central claim about learning from behavioural data.
- **No new guarantee kind.** `E_Q` is smaller; the claim is still `‖Q̂ − Q^π‖∞ ≤ E_Q` at risk `δ`.
  `0` coverage violations is a failure detector, not evidence — `FP-TIGHT-001` already showed an
  unsound arm passing it, which is why `C1`–`C3` are the load-bearing checks here.
- **Nothing about the multi-step trajectory.** This is step 1 on frozen `π_0`. `H5` says the *median*
  stop is now on the other side of the line at `c64k`; whether the certified iteration actually runs
  longer under `L123` is the next measurement, and the earlier `FP-HORIZON-001`/`FP-EARLYSTOP-001`
  populations are the natural place to look.
- **The model-free version of `L3` is untried.** The certification batch already contains iid draws of
  `s′ ~ P(·|s,a)` for every pair, so `Σ_{s′} P(s′|s,a) w(s′)` is estimable from the same data with its
  own confidence term — a certified fitted-value-iteration on the bound. If it works it would recover
  part of the `28`-point gap between `L12` and `L123` without the kernel. Registered as the natural
  next task, not attempted here.

**Same-actor throughout**, per the user's standing instruction.
