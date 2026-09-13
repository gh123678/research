# FP-BOUND-002 first result: the biggest lever no longer needs the kernel

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`. Baseline: `e1bdc63`.
Actor: Claude, under the user's direct instruction of 2026-09-13 ("都去做").
Task: [FP-BOUND-002](../../research_tasks/FP-BOUND-002.md) (hypotheses registered before the run).
Code: `fixed_policy_tight_certificate.propagation_data_driven`, `evaluate_fp_bound_002.py`.

**Preliminary result, single actor.**

## 1. The question

`FP-BOUND-001` left one big prize on the table. Its `L123` arm — exact propagation of the per-pair
residual interval through `(I−γP^π)^{-1}` — cut `E_Q` by `41.5%` at `c64k` and `43.7%` at `c16k`,
against `13.5%` / `26.9%` for the model-free `L12`. The difference, about `28` points, was paid for
by **reading the transition kernel**, which puts that arm in the same family as the pre-registered
oracle-kernel arm and disqualifies it from any claim about learning from behavioural data.

Can the same propagation be done from the certification batch alone?

## 2. Yes — the batch already contains the kernel it needs

For every pair `x = (s,a)` the batch holds iid draws of `s' ~ P(·|s,a)`. Those draws are what produced
the residuals; they equally estimate `E_{s'~P(·|s,a)}[f(s')]` for **any** `f`, with no kernel. So the
identity `Q̂ − Q^π = −(I−γP^π)^{-1}ρ` can be iterated on bounds:

```text
W_0        = max_x ε_x/(1−γ)                        (a valid bound on ‖w‖∞)
W_{k+1}(s) = Σ_a π(a|s) ε(s,a) + γ Σ_a π(a|s) [ T_x(W_k) + c_k(x) ]
T_x(f)     = (1/n_x) Σ_i f(s′_i)                    (the pair's own successor histogram)
c_k(x)     = ‖W_k‖∞ · sqrt(log(1/δ_k)/(2 n_x))      (one-sided Hoeffding)
```

`|w| ≤ W_k` holds for every `k` by induction (`|w(s)| ≤ ε̄(s) + γ Σ_a π(a|s) E[|w(s′)|]`), and one
final application bounds `|u|`. The iteration converges to `(I−γT̂)^{-1}ε̄ ≈ (I−γK)^{-1}ε̄` — the `L123`
limit — except that every step pays a confidence term for its own estimate. Samples are reused across
iterations, so the risk is union-bounded over `n_iter` rather than assumed fresh.

**Risk.** `δ_step` splits into `δ_eps` (per-pair intervals) and `δ_prop` (propagation estimates); the
latter splits again into `n_iter · d` pieces (one per iteration per pair) and `d` pieces (the final
estimate). `δ_eps + δ_prop = δ_step` exactly, checked per record-route.

## 3. Results (same environments, seeds and batch schedule as FP-BOUND-001)

| rung | lever | mean `E_Q` | vs frozen | step-1 emitted | coverage viol. |
|---|---|---:|---:|---:|---:|
| `c64k` | `frozen` | `0.12044` | — | `34/48` | `0` |
| | `L12` (model-free) | `0.10422` | `−13.5%` | `37` | `0` |
| | `L12M` (model-free) | `0.07510` | **`−37.6%`** | **`44`** | `0` |
| | `L123` (kernel) | `0.07047` | `−41.5%` | `44` | `0` |
| `c16k` | `frozen` | `0.23226` | — | `20/48` | `0` |
| | `L12` (model-free) | `0.16976` | `−26.9%` | `21` | `0` |
| | `L12M` (model-free) | `0.14507` | **`−37.5%`** | **`27`** | `0` |
| | `L123` (kernel) | `0.13076` | `−43.7%` | `32` | `0` |

**The model-free arm recovers `86.3%` of the kernel arm's gain at `c64k` and emits exactly as many
records (`44/48`, the same set).** At `c16k` it recovers `63.3%` and emits `27` against the kernel's
`32`.

## 4. Verdicts against the registered bands

| hypothesis | verdict |
|---|---|
| `H1`① coverage, `0` violations | **PASS** at both rungs |
| `H1`② `L12M ≤ L12` on **every** record-route | **FALSIFIED**: `2/48` exceptions at `c64k`, `6/48` at `c16k` |
| `H2` reduction in `[−45%, −15%]` | **PASS** (`−37.6%`, `−37.5%`) |
| `H3` recovery `≥ 70%` | **PASS** at `c64k` (`86.3%`), **FALSIFIED** at `c16k` (`63.3%`) |
| `H4` `L123 ≤ L12M` on `≥ 90%` of records | **PASS** (`48/48` at both rungs) |
| `H5` emissions `≥ L12 + 3` | **PASS** (`44` vs `37`; `27` vs `21`) |
| `H6` risk accounting exact | **PASS** |
| `H7` shared arms reproduce FP-BOUND-001 bit-for-bit | **PASS** (max `|ΔE_Q|` `0.0`, `0` decision mismatches, `48` record-routes × `4` arms) |

### The falsified mandatory clause, located

`H1`② is not an implementation error — the accounting checks out (`H6` PASS) — it is the **price of
being model-free**. `L12M` splits `δ_step` in half, so its per-pair intervals are computed at half of
`L12`'s risk budget: `log(2/δ′)` goes from `6.17` back to `6.87` and the interval is about `5%` looser.
On the records where the propagation gain is smaller than that premium, the trade loses:

```text
c64k exceptions:  mix=0.08 task=12 expected_exact   +3.93e-05  (+0.056%)
                  mix=0.08 task=23 expected_finite  +3.08e-04  (+0.481%)
```

Both are far below the arm's own gain (median `−2.5e-02`, min `−1.0e-01`), but a mandatory clause is a
mandatory clause, so it is recorded as **falsified, not excused**. The repair is obvious and is
measured in `FP-BOUND-003`: the propagation estimates do not need half the budget.

## 5. Verification

`verify_fp_bound_002.py`, recomputing from the sealed bundle alone:

| check | `c16k` | `c64k` |
|---|---|---|
| `C1` L12M rebuilt from the sealed successor histogram / sizes / intervals | max `|Δ|` `0.0` | max `|Δ|` `0.0` |
| `C2` per-pair intervals vs the **exact** residual (`240` cells × `12` pairs) | `240/240` | `240/240` |
| `C3` `E_Q` reconstruction (all non-propagation levers, `L123` included) | max `|Δ|` `0.0` | max `|Δ|` `0.0` |
| `C4` shared arms vs FP-BOUND-001 | `0.0`, `0` mismatches | `0.0`, `0` mismatches |
| `C5` `emitted ⟺ E_Q < h` | `0` mismatches | `0` mismatches |
| `C6` risk accounting | `0` bad | `0` bad |

`C1` is the load-bearing check: the entire model-free bound — iteration, confidence terms, final
propagation — is reproducible from the sealed numbers with **no kernel and no shared code path**.

*(One verifier defect was found and fixed in the process: the first version applied the uniform
`max ε/(1−γ)` formula to `L123` and reported a spurious `0.11` discrepancy; and its failure counter did
not include `C3`. Both fixed. The certificate was never wrong — the checker was.)*

## 6. What this establishes

- **The propagation lever is available without the kernel.** `−37.6%` at `c64k` from behaviour data
  alone, versus `−41.5%` with the kernel: the kernel buys `4` points out of `41`, not the `28` it
  appeared to buy in FP-BOUND-001 (because `L12`'s `13.5%` was the wrong comparison — most of the
  propagation gain is recoverable, just not for free).
- At `c64k` the model-free arm emits **the same `44/48` records** as the kernel arm, so on this
  population the oracle buys nothing at the decision line.
- The cost of being model-free is a slightly looser interval, not a weaker method; it shows up on
  `2/48` (`c64k`) and `6/48` (`c16k`) records as a `≤0.5%` relative loss.

**Not established.** Nothing about the iteration (this is step 1) — that is `FP-ITER-BOUND-001`.
`H3`'s `c16k` failure is real and unexplained beyond the sample-size dependence of the premium. And the
`support_range` reference arm and the oracle claims are untouched by any of this.

**Same-actor throughout**, per the user's standing instruction.
