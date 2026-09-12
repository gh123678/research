# FP-TIGHT-001: A soundly tightened certificate, and what it actually buys

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12 ("好的你去做"),
  closing the work item "先松证书，再谈走多远".
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `12ecfea` (`claude/FP-CENSUS-001`).
- Execution branch: `claude/FP-CENSUS-001`.
- Result directory: `results/FP-TIGHT-001/claude/`.
- Classification: long, conclusion-critical, single-actor under the standing user
  exception.

## Why this task exists

`FP-ITER6-001` and `FP-ATTN-ITER6-001` showed the certified iteration reaching six
steps on both implementations with the minimum gain down to `0.019347` — below the
`0.05` floor steps 4 and 5 cleared. `FP-CENSUS-001` then showed *why* records stop:
all `48` abstaining rows carry the single reason `improvement_lcb_nonpositive`, the
certificate never fails, and the discrimination between emitting and abstaining
records is carried by the within-state value spread rather than by the certified
error. The iteration is not running out of records to certify; it is running out of
margin.

So the obvious next move is to make `E_Q` smaller. The pre-draft proposal estimated
`E_Q: 0.22 -> 0.08-0.12` and emissions `22/48 -> 35+/48` by "writing the constants
tight" and "using per-cell observed scales". This task tests that, and the answer
turns out to be different from the estimate — which is the point of testing it.

### What the frozen certificate actually does

```text
E_Q = max_x ( |mean_B(x)| + r_x ) / (1 - gamma)
r_x = 2 * s_x * sqrt(log(1/delta_each) / N_B)
s_x = sqrt( mean_A(Y^2) + (2B)^2/2 * sqrt(2 log(1/delta_each) / N_A) )
```

Decomposed on sealed batches, the binding pair is the sparsely covered one, the
envelope term is 87-95% of `s_x^2`, and the frozen radius is 93% envelope while the
observed `max|Y|` is only `2.4-2.7` against an assumed range of `10`.

The second-moment step is a legitimate Hoeffding application (`Z = Y^2` lies in
`[0, (2B)^2]`). **The mean step is not.** It is documented as "Hoeffding's lemma on
the bounded residuals", but Hoeffding's lemma needs the *range* of `Y`, which is
`20`, not the data-estimated `s_x ~ 1.2`. The frozen step substitutes an estimated
scale for a range.

The repair is not a new assumption. It is **Bernstein's inequality**, which is
precisely the inequality that uses `E[Y^2] <= V` together with the range, and which
does not require `Y` to be sub-Gaussian with parameter `V`.

## Research question

How much can `E_Q` be reduced by sound repairs alone, and does that reduction cross
the threshold that `FP-CENSUS-001` measured for emission?

## Falsifiable hypotheses

1. `H1 (soundness, mandatory)`: both repairs' `E_Q` still bounds the realized oracle
   sup-error on all `48` route-records at step 1 — zero coverage violations.
2. `H2 (monotonicity, mandatory)`: every route-record that emitted under the frozen
   certificate still emits under both repairs. This is provable rather than
   empirical — `LB_s = Î_s − E_Q‖Δπ_s‖₁` is increasing in `E_Q` — so a single
   counterexample falsifies the implementation, not the mathematics.
3. `H3 (pre-registered magnitude, corrected inequality)`: the Bernstein repair's mean
   `E_Q` reduction over all `48` route-records lies in `[-25%, -10%]`, point estimate
   `-16%`.
4. `H4 (pre-registered magnitude, empirical Bernstein)`: the empirical-Bernstein
   repair's mean reduction lies in `[-35%, -18%]`, point estimate `-24%`.
5. `H5 (the substantive pre-registered prediction)`: **neither repair flips a single
   one of the `26` never-emitting route-records into emission.** The arithmetic that
   motivates this: the census measured the abstaining group at mean
   `sigma_min / E_Q = 1.244` and the emission threshold at `~2.17`, so a flip needs
   `E_Q` to fall by about `43%`. `H3`/`H4` predict at most `35%`.
6. `H6 (pre-registered: the bound's own framing is wrong about the biggest lever)`:
   the envelope term is `87-95%` of `s_x^2`, yet removing it entirely does **not**
   dominate replacing the mean-step inequality. Predicted because after either
   repair the *range* term becomes the binding half of the radius.
7. `H7 (pre-registered, sample size)`: on the secondary arm, multiplying the
   certification sample by `4` reduces mean `E_Q` by more than `40%`, and flips at
   least one never-emitting route-record. This is the lever the arithmetic says
   should work, and it is a **cost** lever, not a mathematical one.

`H1`, `H2` are mandatory. `H3`--`H7` are the substantive pre-registered predictions,
each reported `PASS` or `FALSIFIED` and never reinterpreted.

### Basis of the predictions, stated explicitly

`H3` and `H4` are anchored on a **pilot** of `12` route-records (`6` records × `2`
routes) drawn from the emitting groups, measured before this sheet was written:
`-16.1%` and `-24.1%`. They are therefore not blind extrapolations, and the pilot's
restriction to emitting records is why the bands are wide: the never-emitting
records may behave differently.

`H5` is **not** derived from the pilot. It is derived from the census's threshold
arithmetic and is a genuine prediction about the `26` records the pilot never
touched. It is also the prediction most likely to be wrong in an interesting way: if
`H3`/`H4` come in at the pessimistic end of their bands *and* some never-emitters
sit near the threshold, a small number could flip.

`H6` contradicts the natural reading of the census's own "84% from a worst-case
assumption" measurement, and is registered so that the contradiction is adjudicated
by data rather than by narrative.

`H7` is a **cost** prediction: it says the programme's next real lever is more
certification data, not a cleverer constant.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing `{0.08, 0.5}`,
`12` tasks per mixing, training trajectory `65536` transitions, certification
`16384 x 64 = 1048576` items, `gamma = 0.70`, `alpha = 0.65`, `R_star = 1.5`,
`delta = 0.05`, seed `20260911`, eta grid `1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`.

### Scope: step 1 only, on the frozen `pi_0`

Every measurement is at step 1 of `pi_0`, for all `24` records × `2` routes = `48`
route-records. That is where the `26` never-emitters live, so it is exactly where
the question is decided. Whether a tighter certificate extends the *iteration* is a
different question and a different task; nothing here claims otherwise.

### The change

One **new** module, `fixed_policy_bernstein_certificate.py`, providing two arms:

- `bernstein_certificate` — the frozen second-moment step, with the mean step
  replaced by Bernstein using the frozen envelope as the range. One changed
  ingredient: the inequality.
- `empirical_bernstein_certificate` — Maurer-Pontil at both steps, replacing the
  range `(2B)^2` by the observed variance of `Y^2` and the range `2(2B)` by the
  observed variance of `Y`.

Risk allocation is `delta/(3d)` per bound over `3d` bounds (one second-moment bound
plus one one-sided mean bound in each direction, since `||.||_inf` is two-sided),
summing to exactly `delta`.

**The sealed modules are not touched.** `fixed_policy_variance_certificate.py`,
`fixed_policy_expected_sarsa.py`, `fixed_policy_expected_sarsa_scaled.py` and
`model.py` must all be byte-identical; the new module imports the sealed one only to
reproduce its split rule and residual definition exactly, and every existing
verifier's hash checks must still pass.

### The decision rule is frozen

Both arms are scored through the **unmodified** `fs.improvement_for`, by passing a
certificate dict with a substituted `e_q`. No rule, threshold, eta grid or reason
order changes.

## Prohibited work

- No modification of any sealed module, sealed bundle, or closed task record other
  than verifier updates needed to keep integrity checks correct.
- No change to any frozen formula, constant, tolerance, eta grid, matrix, hypothesis
  or metric after the run.
- **No re-tuning of the new certificate after seeing the results.** The constants are
  those of the named inequalities; the `7/3` Maurer-Pontil constant and the
  `delta/(3d)` allocation are fixed in the code before the formal run.
- No claim that a tighter certificate is a better *guarantee*: a smaller `E_Q` is
  only useful if it still covers the realized error, which is what `H1` tests.
- No reinterpretation of a falsified prediction.
- No `git add -A`.

## Acceptance criteria

1. `H1` coverage: zero violations for both arms on all `48` route-records, with the
   realized error recorded per arm.
2. `H2` monotonicity: no route-record loses eligibility; any counterexample listed.
3. `H3`--`H7` each evaluated with evidence.
4. Per-record `E_Q` for all three arms, plus `sigma_min` and the census ratio, so any
   reader can re-derive the flip arithmetic.
5. The sealed modules' hashes verified byte-identical after the run.
6. All strict-JSON, finite, shape, seed and location checks pass.
7. Complete reproducibility evidence, including the new module's hash.
8. Same-actor derived verification recorded.
9. `ACTIVE_WORKSPACE.md` updated.

## Failure criteria

The construction fails if either arm produces a coverage violation, if a sealed
module's hash changes, if the decision rule is modified, or if a prediction is
quietly re-scoped after the fact.

`H3`--`H7` failing is **not** a construction failure. `H5` failing would be the most
useful outcome available: it would mean a sound tightening is enough to revive
records, and the programme should continue down this road. `H5` passing means the
constant is not the bottleneck and the answer is sample size.

## Stopping conditions

Stop and report if:

- a coverage violation appears in either arm, since that would mean the repair is
  not sound and no result below it can be trusted;
- a sealed module's hash is found changed;
- the comparison would require changing a frozen tolerance or parameter;
- execution would expand cost, publication or permissions beyond authorization.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-12), same-actor.
- Evidence: `docs/research_branches/FP-TIGHT-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-12.
- Decision: "好的你去做" — proceed with the certificate-tightening task as proposed.
- Scope: this task, on `claude/FP-CENSUS-001`.

## Definition of done

- [x] Coverage confirmed for both arms.
- [x] Monotonicity confirmed.
- [x] `H3`--`H7` each evaluated with evidence.
- [x] The flip arithmetic shown per record.
- [x] Sealed module hashes verified unchanged.
- [x] Same-actor derived verification recorded.
- [x] `ACTIVE_WORKSPACE.md` updated.

## Formal outcome (2026-09-12)

Route journal: `docs/research_branches/FP-TIGHT-001/claude/first_result.md`.

| arm | mean `E_Q` | vs frozen | coverage |
|---|---:|---:|---:|
| `frozen` | `0.2426` | — | `0/48` violations |
| `bernstein` (corrected inequality) | `0.2061` | `−15.0%` | `0/48` |
| `empirical_bernstein` | `0.1940` | `−20.0%` | `0/48` |
| `counterfactual_no_envelope` *(unsound)* | `0.1334` | `−45.0%` | `0/48` |
| `x4` certification *(subset pilot)* | `0.0730` from `0.2029` | `−64.0%` | `0/48` |

| hypothesis | verdict |
|---|---|
| `H1` both repairs cover | **PASS** (`0` violations on `48/48`) |
| `H2` monotonicity | **PASS** (`0` lost) |
| `H3` Bernstein band | **PASS** (`−15.0%` in `[−25%, −10%]`) |
| `H4` empirical-Bernstein band | **PASS** (`−20.0%` in `[−35%, −18%]`) |
| `H5` no flips | **FALSIFIED** — `8` distinct route-records flip, `22/48 → 30/48` |
| `H6` inequality beats envelope | **FALSIFIED** — envelope `−45%` vs inequality `−20%` |
| `H7` reduction clause | **PASS** (`−64.0%`, needed `≤ −40%`) |
| `H7` flip clause | **NOT EXERCISED** (population held no abstainer) |

- **The frozen mean step was not licensed by the inequality it names.** Hoeffding's
  lemma needs the range of `Y` (`20`), not the data-estimated `s_x ≈ 1.2`.
  Replacing it with Bernstein — same two ingredients, correct inequality — is sound
  and worth `−15%`; the empirical-Bernstein variant is worth `−20%`.
- **That revives `8` of the `26` never-emitting route-records**, taking step-1
  emissions from `22/48` to `30/48`. The pre-draft's `22 → 35+` was directionally
  right and numerically over-optimistic.
- **`H5` failed for a reason worth recording**: the `−43%` flip requirement was
  computed from the abstainers' **mean** spread and then used to make a universal
  claim. The flips are the high-spread tail; the mean-sitting records never move.
- **`H6` failed in the opposite direction to my prediction**: deleting the envelope
  (`−45%`) dominates correcting the inequality (`−20%`), so the census's "84% from
  the worst-case assumption" reading is the correct guide and my contradiction of it
  was wrong.
- **The sample-size lever is the strongest measured**: `4x` certification gives
  `−64%`, soundly, better than the unsound envelope ceiling. Its flip clause is
  untested because the cost-driven subset contained no abstainer — reported as
  `NOT_EXERCISED`, never as a vacuous failure.
- **An unsound arm passed the coverage audit** (`0/48`). A clean audit is therefore
  not evidence of validity; these bounds are loose enough that even an unjustified
  one passes. This is recorded because it bears directly on how much "0 violations"
  is worth elsewhere in the line.
- Post-hoc and labelled as such: the census's `theta = 2.168689`, applied to ratios
  from a certificate it was never fitted on, misclassifies **`0/48`** (versus `1/48`
  in-sample). The abstainers' two groups become **disjoint** under the new
  certificate (`[2.5623, 3.0182]` vs `[0.6494, 1.7371]`). The census's criterion is
  a **certificate-independent eligibility gate**, not a predictor of time — which is
  why it could not say when a record stops and does say whether it is eligible.
- **Scope**: step 1 only. Nothing here claims the tightened certificate extends the
  iteration.
- A correction owed to the record: an exploratory figure of `−56%` for empirical
  Bernstein was reported to the user before the run and was **wrong twice over**
  (non-rigorous Bernstein constant; the envelope used where the range belongs). The
  measured value is `−20%`. The pre-review records the withdrawal.
- Sealed modules byte-identical; every replayed verification record reproduced
  exactly.
