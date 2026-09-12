# FP-TIGHT-001 first result

Date: 2026-09-12.
Branch: `claude/FP-CENSUS-001`. Baseline: `12ecfea`.
Actor: Claude, under the user's instruction of 2026-09-12 ("好的你去做"), closing the
work item "先松证书，再谈走多远".

## 1. What was run

```text
python -B evaluate_fp_tight_001.py --tasks 12 --mixings 0.08,0.5 \
    --label tightened-certificate --output-dir results/FP-TIGHT-001/claude/formal
python -B analyze_fp_tight_001.py
```

Step 1 of the frozen `pi_0`, all `24` records × `2` routes = `48` route-records,
which is where the `26` never-emitters live. The certification protocol, the
decision rule, the eta grid and the sealed modules are untouched; one **new** module
(`fixed_policy_bernstein_certificate.py`) supplies two sound repairs and one
labelled counterfactual. Whether a tighter certificate extends the *iteration* is a
different question and is **not** claimed here.

## 2. What was wrong with the frozen certificate

```text
E_Q = max_x ( |mean_B(x)| + r_x ) / (1 - gamma)
r_x = 2 * s_x * sqrt(log(1/delta_each) / N_B)
```

The second-moment step is a legitimate Hoeffding application (`Z = Y^2` lies in
`[0, (2B)^2]`). **The mean step is not.** It is documented as "Hoeffding's lemma on
the bounded residuals", but Hoeffding's lemma needs the *range* of `Y`, which is
`20`, not the data-estimated `s_x ~ 1.2`. The frozen step substitutes an estimated
scale for a range.

The repair is not a new assumption: it is **Bernstein's inequality**, which is
exactly the inequality that uses `E[Y^2] <= V` together with the range, and which
does not require `Y` to be sub-Gaussian with parameter `V`. Two sound arms were
built, differing in how much of the range they replace with observed variance.

## 3. `H1`/`H2`: the repairs are sound and monotone

| arm | coverage violations (of 48) |
|---|---:|
| `frozen` | `0` |
| `bernstein` | `0` |
| `empirical_bernstein` | `0` |

`H2` holds with `0` route-records losing eligibility — as it must, since
`LB_s = Î_s − E_Q‖Δπ_s‖₁` is increasing in `E_Q`.

**An epistemic warning that falls out of the counterfactual.** The arm with the
envelope *deleted* — provably not a certificate, since no high-confidence statement
survives — also had **`0` coverage violations on all `48` route-records**. So a clean
coverage audit is **not** evidence that a bound is valid: these bounds are loose
enough that even an unjustified one passes. That reinforces, with a concrete
number, the concern recorded in
`docs/2026-09-12-independent-verification-handoff.md` about what "0 violations"
does and does not establish.

## 4. Magnitude: the honest number, and a correction to my own earlier estimate

| arm | mean `E_Q` | vs frozen |
|---|---:|---:|
| `frozen` | `0.2426` | — |
| `bernstein` | `0.2061` | **`−15.0%`** |
| `empirical_bernstein` | `0.1940` | **`−20.0%`** |
| `counterfactual_no_envelope` | `0.1334` | `−45.0%` *(not a certificate)* |

`H3` **PASS** (`−15.0%` in the registered `[−25%, −10%]`), `H4` **PASS** (`−20.0%`
in `[−35%, −18%]`). Both bands were anchored on a `12`-route-record pilot taken
before the task sheet was written.

**A correction I owe the record.** During exploratory work I reported to the user
that empirical Bernstein would give about `−56%`. That figure came from a scratch
Bernstein radius using `t² = (2V + (2/3)Rt)·log(1/δ)/N` with the observed half-B
variance, and it was wrong twice over: the rigorous Maurer-Pontil form carries the
constant `7/3` and `log(2/δ)`, and the sealed risk spread over `3d = 36` bounds
rather than `2d = 24` both cost tightness. Separately, that diagnostic passed the
*envelope* where Bernstein requires the *range* (`20`, not `10`), under-counting the
range term by a factor of two. The measured answer is `−20%`. The pre-review records
the withdrawal; the `−56%` is not quoted again.

So the pre-draft's estimate — "tighten the constants, `E_Q: 0.22 → 0.08–0.12`,
emissions `22/48 → 35+/48`" — is **directionally right and numerically
over-optimistic**. The sound repair reaches `0.194`, and emissions go `22 → 30`.

## 5. `H5` FALSIFIED: the tightening *does* revive records

**`8` distinct route-records flip from abstention to emission**, each under both
sound arms (`16` arm-flips):

```text
0.08/3 exact   0.08/3 finite   0.08/9 exact   0.08/9 finite
0.5/2  exact   0.5/6 exact     0.5/11 exact   0.5/11 finite
```

Step-1 emissions go from `22/48` to `30/48`.

### Why I predicted wrong, stated precisely

I registered `H5` as "zero flips" from the arithmetic `sigma_min/E_Q = 1.244` versus
a threshold of `2.17`, i.e. a flip needs about `−43%`. **That arithmetic used the
abstainers' *mean* spread and then made a universal claim from it.** The flipped
records are the **high-spread tail**, not the average: they need far less than
`−43%`, and `−15%` to `−20%` is enough. The mean-sitting records (`σ ≈ 0.15–0.29`)
sit far below the threshold and do not move.

The lesson is the one this repository keeps re-learning: a mean is not a population,
and a prediction that says "none" must be checked against the tail, not the average.

## 6. `H6` FALSIFIED: the envelope really is the bigger lever

I predicted the opposite — that once the mean-step inequality is corrected, the
*range* term would become binding and correcting the inequality would be the bigger
move. The data says no:

```text
delete the envelope (unsound)   -45.0%
correct the inequality (sound)  -20.0%
```

So the census's "87–95% of `s_x²` is the worst-case assumption" reading **is** the
right guide to where the slack lives, and my contradiction of it was wrong. Recorded
as falsified, not reinterpreted.

The practical consequence is the important part: the largest *available* lever is to
make the **range** data-driven rather than to sharpen the inequality further. The
counterfactual prices that lever's ceiling at `−45%`, but the counterfactual is
unsound, so the real task is a range that is smaller **and** carries its own
confidence correction. That is the next task, not this one.

## 7. Post-hoc: the census criterion transfers, out of sample

Not pre-registered, and reported as such.

`FP-CENSUS-001` fitted `theta = 2.168689` to the ratio `sigma_min/E_Q` under the
**frozen** certificate, and there it misclassified `1/48`. It then failed to predict
the `FP-ITER6-001` dropout. Applying the same `theta`, unchanged, to the ratios of
the **new** certificate — which it was never fitted on:

| ratios | misclassifications |
|---|---:|
| frozen (in-sample) | `1/48` |
| empirical Bernstein (**out-of-sample**) | **`0/48`** |

Within the `26` abstainers under the new certificate the two groups are **disjoint**:

```text
flippers     sigma_min/E_Q in [2.5623, 3.0182]
non-flippers sigma_min/E_Q in [0.6494, 1.7371]
```

This resolves the census's apparent failure. The criterion is **not a predictor of
time** — it is the eligibility gate written in a form that does not depend on which
certificate is in force:

> a route-record emits when `sigma_min / E_Q` clears roughly `2.17`, where `E_Q` is
> whatever certificate is being used.

That is why it could not say *when* a record stops (the ratio barely moves along a
trajectory) and why it does say *whether* a record is eligible (a certificate change
moves the population across a fixed line). Per the census's own rule it is **not**
called a classifier on the strength of one population; the counts are given instead.

## 8. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` soundness, both repairs cover | **PASS** (`0` violations on `48/48`, both arms) |
| `H2` monotonicity | **PASS** (`0` lost) |
| `H3` Bernstein band | **PASS** (`−15.0%`) |
| `H4` empirical-Bernstein band | **PASS** (`−20.0%`) |
| `H5` no flips | **FALSIFIED** (`8` distinct records flip) |
| `H6` inequality beats envelope | **FALSIFIED** (envelope `−45%` vs inequality `−20%`) |
| `H7` sample size: reduction clause | **PASS** (`−64.0%`, needed `≤ −40%`) |
| `H7` sample size: flip clause | **NOT EXERCISED** (population contained no abstainer) |

Construction checks **PASS**.

## 9. The sample-size arm: the lever that actually works

A pilot arm with **`4x` the certification data** (`65536 x 64 = 4.2M` items instead
of `1.05M`) on a `3`-record subset (`6` route-records, mixing `0.08`):

| quantity | value |
|---|---:|
| mean `E_Q` at `1x` | `0.2029` |
| mean `E_Q` at `4x` (empirical Bernstein) | `0.0730` |
| reduction | **`−64.0%`** |
| coverage violations at `4x` | `0` |
| abstainers in the population | `0` |

**The reduction clause passes and overshoots**: `−64%` against a registered
threshold of `−40%`. This is the single largest sound lever measured anywhere in
this task — larger than the *unsound* envelope deletion (`−45%`).

**The flip clause is not exercised, and is not reported as falsified.** The subset
is records `0, 1, 2` of mixing `0.08`, and all three already emit at step 1, so the
population contains nothing that could flip. A `0` there would be a *vacuous
failure* — the same defect class as the vacuous *pass* this line already had to fix
in `verify_fp_attn_iter4_001_same_actor.py`. The analyzer now distinguishes
`NOT_EXERCISED` from `FALSIFIED` and records the population's composition instead.

So the honest reading is: the sample-size lever is **confirmed to shrink `E_Q` far
more than any constant repair**, and whether it revives abstainers is **untested**.
The arithmetic says it should — `−64%` exceeds the `−43%` flip requirement computed
in §5 — but that is arithmetic, not a measurement, and this record does not present
it as one.

**Why the subset, stated plainly.** The full `24`-record `4x` arm was started and
killed: at `~5.5 min/record` it needed about two hours, because the certification
generator is a Python loop over `4.2M` items per record. The subset was chosen for
cost, not for composition — which is exactly why its composition has to be reported
rather than glossed.

**A limitation of the comparison.** The `4x` batch is a fresh independent sample
drawn with the same seed schedule and more chains, **not** a superset of the sealed
batch, because the generator consumes the random stream per chain. The comparison
therefore mixes sample size with realisation. The effect measured is far larger than
the pilot-to-pilot scatter in the `1x` arm, but the confound is real and is recorded
rather than assuming it away.

---

## 10. What this establishes, and what it does not

**Established**

- The frozen mean step is not licensed by the inequality it names, and replacing it
  with Bernstein is **sound** (`H1`) and **worth `−15%`**; the empirical-Bernstein
  variant is worth `−20%`.
- That is **enough to revive `8` of the `26` never-emitting route-records**, taking
  step-1 emissions from `22/48` to `30/48`.
- The census's threshold is a **certificate-independent gate**, validated
  out-of-sample at `0/48`.
- The largest remaining lever is the **range**, not the inequality — measured, not
  argued.
- **The sample-size lever is the strongest of all**: `4x` certification data gives
  `−64%` on `E_Q`, soundly (zero coverage violations), which beats even the unsound
  envelope-deletion ceiling.
- A clean coverage audit is not evidence of validity: an unsound arm passed it.

**Not established**

- **Nothing about the iteration.** This is step 1 only. Whether `30` eligible records
  instead of `22` extends the certified iteration past six steps is untested.
- **No new guarantee.** `E_Q` is smaller; the claim is still
  `||Qhat - Q^pi||_inf <= E_Q` at risk `delta`, and the tightening does not improve
  what is being claimed, only how tight it is.
- **The range lever is priced, not delivered.** `−45%` is an unsound floor. A sound
  data-driven range has to carry its own confidence correction, and that cost is not
  yet measured.
- **The sample-size flip clause is untested.** `−64%` on `E_Q` is measured; whether
  it revives abstainers is arithmetic on a `3`-record subset with no abstainers in
  it.
- **Same-actor throughout**, per the user's standing instruction.
