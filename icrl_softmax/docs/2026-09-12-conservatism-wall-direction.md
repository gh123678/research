# Research direction proposal: the conservatism wall of certified policy improvement

- Date: 2026-09-12.
- Author: drafted by the assistant at the user's request; **not** a GPT-authored
  task sheet.
- Status: **PROPOSAL**. It has no execution authority. Nothing in this document
  may be implemented or run until the user rules on the direction and a formal
  task sheet is authored under `AGENTS.md` sections 2--3.
- Predecessors: `FP-SCALE-002` (first certified improvement, 22/48),
  `FP-SCALE-001` (certificate-scale diagnosis), `FP-ESARSA-001` and
  `FP-ADV-001` (0/480 each), `FP-TU-001` (verified certificate baseline),
  `FP-KERN-001/002` (closed negative routes).
- Baseline for every number below: the sealed `FP-SCALE-002` formal records at
  `results/FP-SCALE-002/claude/formal/`, plus a from-source reconstruction of
  `Qhat` and the certificate described in section 4.

> **Status note added 2026-09-12 by a different session, after the draft was
> written.** The draft predates and does not reference three results that are now
> sealed and that bear directly on it. Nothing below has been altered; this note
> only says what has since been measured, so that section 3 and section 7 are read
> against current evidence rather than against the state of play at drafting time.
>
> - `FP-CENSUS-001` — the eligibility census over all `48` route-records and up to
>   five steps. It qualifies §3.4/§3.5 in one specific way: across every abstaining
>   step in the census the frozen reason is `improvement_lcb_nonpositive` and
>   **never** a certificate failure, and the discrimination between emitting and
>   abstaining records is carried by the within-state value spread (`2.6x`) rather
>   than by the certified error (`1.2x`). `E_Q` alone, as a predictor of
>   abstention, misclassifies `16/48`.
> - `FP-ITER6-001` and `FP-ATTN-ITER6-001` — a **sixth** certified step on both
>   implementations. Emissions are `22, 20, 15, 12, 12, 9`, so the plateau in §3
>   is one interval rather than a fixed point, and the minimum sixth-step gain
>   (`0.019347`) falls below the `0.05` floor that steps 4 and 5 cleared.
> - The census also measured the step-1 ratio `sigma_min / E_Q` as a **predictor**:
>   the best threshold misclassifies `1/48`, and no threshold on the step-5 ratio
>   separates the three records that stopped at step 6, because two of them had
>   ratios above the lowest surviving record. Evidence:
>   `docs/research_branches/FP-CENSUS-001/claude/first_result.md`.
>
> Consequently §3.3's coverage curve and §5.3's predicted coverage law should be
> read as **superseded in scope** by a six-step horizon, and §7's hypotheses may
> want a companion that is scored against step 6 rather than step 1. The five open
> rulings in section 10 are untouched by any of this and still require the user.

## 1. What question this proposal answers

`FP-SCALE-002` reports a positive result: 22 of 48 primary route-records emit a
certified, componentwise non-degrading, strictly improving relative-softmax
update. Read as an algorithmic contribution it is weak: the construction is
inherited, and the 22/48 figure is an artifact of one certificate scale on one
toy testbed.

Read as a measurement, the same records say something much more useful, and the
project has not said it:

> **Certified policy improvement on this testbed is not limited by the
> construction, by the data volume, or by the implementation. It is limited by
> a conservatism factor in the certificate that is one to two orders of
> magnitude, and that factor is structural to the certificate's shape.**

This proposal makes that statement the object of study instead of a footnote.

The shift in the research question is:

| | old question | proposed question |
|---|---|---|
| wording | can softmax do policy improvement? | where is the wall of certified policy improvement, what is it made of, and can it be moved? |
| deliverable | a positive emission rate | a measured wall, a decomposition, and a stated structural bound |
| role of 22/48 | the headline | one point on a curve |

## 2. Why this is worth doing now

Three items of evidence already exist and have never been assembled into one
statement:

1. **Three independent constructions emitted nothing.** `FP-ADV-001` 0/480,
   `FP-ESARSA-001` 0/480, `FP-SCALE-001` 0/1. The null is robust across
   constructions.
2. **One scale repair moved the number, and the diagnosis said why.** Moving to
   a dedicated certification batch took `E_Q` from a 22.47 floor to 0.98 and the
   worst-pair count from 14 to 40000, and *still* emitted nothing; only the
   variance-adaptive concentration argument emitted updates. `FP-SCALE-001`
   established quantitatively that emission needs `E_Q` below the within-state
   value spread, and that this ratio is invariant to widening the value
   spectrum.
3. **The conservatism is now measurable, not conjectured.** Section 3 measures
   it. The certificate's permitted error is 20 to 18000 times the realised
   error; the median is 84.

Point 3 is the missing sentence. `ACTIVE_WORKSPACE.md` currently records "22/48
emissions, 0 violations" and stops there. That framing invites the reader to ask
for a better algorithm. The measurement invites the reader to ask how far the
guarantee itself can be pushed, which is a question this project is unusually
well equipped to answer, because it owns a fully sealed and reproducible
certificate pipeline.

## 3. Measurements already taken (pre-draft, reproducible)

All figures below were produced by reconstructing `Qhat` and the certificate
from source and checking them bit-for-bit against the sealed records before use.
`E_Q` reproduced exactly on every one of the 48 primary route-records, and the
frozen emission rule reproduced 48/48 emit-or-abstain decisions from an
independently written implementation.

### 3.1 Conservatism

For each of the 192 (record x state) pairs, at the `eta` the frozen rule
actually selects, compare

- permitted error = `E_Q * ||pi_eta^+ - pi||_1`
- realised error = `|I_s(eta) - Ihat_s(eta)|`

| statistic | value |
|---|---|
| min | 19.8 |
| p25 | 47.8 |
| **median** | **83.6** |
| p75 | 186.9 |
| max | 12849.0 |
| share >= 10x | 100% |
| share >= 50x | 70% |

The certificate is never closer than a factor of 20 to the quantity it bounds.

### 3.2 Wall position

For each record, the critical `E_Q` is the largest certificate error at which
the frozen rule still emits on the frozen `eta` grid.

| statistic | current `E_Q` | critical `E_Q` |
|---|---|---|
| min | 0.1637 | 0.0630 |
| p25 | 0.2043 | 0.1598 |
| median | 0.2194 | 0.2106 |
| p75 | 0.2824 | 0.3421 |
| max | 0.4118 | 0.9368 |

**26 of 48 records sit on the wrong side of their own critical value.** The
ratio current/critical has median 1.0335, p25 1.0000, p75 1.8724: the median
record exceeds its own threshold by 3.3%, and the upper quartile by 87%. This
is a wall, not a gradual degradation: records do not emit less when `E_Q` rises,
they stop.

### 3.3 Coverage curve

Scaling `E_Q` by `k` and re-evaluating the frozen rule:

| k | median `E_Q` | emits | share |
|---|---|---|---|
| 1.000 | 0.2194 | 22/48 | 45.8% |
| 0.800 | 0.1755 | 30/48 | 62.5% |
| 0.673 | 0.1477 | 30/48 | 62.5% |
| 0.546 | 0.1198 | 35/48 | 72.9% |
| 0.500 | 0.1097 | 40/48 | 83.3% |
| 0.350 | 0.0768 | 44/48 | 91.7% |
| 0.300 | 0.0658 | 46/48 | 95.8% |
| **0.262** | 0.0575 | **48/48** | **100%** |

Two numbers matter:

- **full coverage needs `E_Q` reduced to a factor of 0.262** of its current
  value, i.e. by 3.8x;
- **the best constant bookkeeping available buys only 1.83x** (k=0.546), which
  reaches 35/48 — thirteen records are out of reach for any constant repair.

The coverage curve is not smooth at the top: between k=0.35 (44/48) and k=0.262
(48/48) the last four records fall away, and each of them is pinned by one
state's threshold. That is the signature of a wall made of individual state
thresholds rather than of an average.

### 3.4 What the conservatism is made of

For one record (`mix=0.08, task=1`, exact route), splitting the per-pair scale
`s_x^2 = mean_A(Y^2) + envelope term`:

| pair | `N_A` | data part `mean(Y^2)` | envelope part | envelope / data | envelope share of `s_x^2` |
|---|---|---|---|---|---|
| (0,0) | 120285 | 0.1316 | 0.5066 | 3.8x | 79.4% |
| (1,1) | 19578 | 0.0724 | 1.2557 | 17.4x | 94.6% |
| (2,0) | 94640 | 0.2507 | 0.5711 | 2.3x | 69.5% |
| (2,2) | 20378 | 0.0054 | 1.2308 | 228.7x | 99.6% |
| (3,0) | 13003 | 0.0315 | 1.5408 | 48.9x | 98.0% |
| (3,1) | 13025 | 0.0533 | 1.5395 | 28.9x | 96.7% |

Over all twelve pairs of this record the envelope share runs from 69.5% to
99.6%, median 93.2%. **The quantity the certificate calls "the residual
scale" is, for this configuration, almost entirely a constant times `B`, not a
property of the data.**

The same record, end to end:

| quantity | value |
|---|---|
| `E_Q` as sealed | 0.1924 |
| `E_Q` with the envelope term removed entirely | 0.0684 |
| realised max error `||Qhat - Q^pi||_inf` | 0.0063 |

**Removing the envelope term is worth 2.8x. After removing it, `E_Q` is still
11x the realised error.** That residual gap is produced by the concentration
step itself, not by the envelope, and no rearrangement of Hoeffding constants
touches it. This is the first sign that the conservatism has more than one
source; section 3.5 counts them.

### 3.5 The conservatism has three independent sources

On the same record the total conservatism decomposes exactly. Factorisation is
meaningful here because each factor touches a different term of the certificate.

| source | factor | basis |
|---|---|---|
| 1. the envelope term `B^2/2 * sqrt(2 log(1/delta_A)/N_A)` inside `s_x^2` | 2.81x | measured, section 3.4 |
| 2. constant bookkeeping and the risk split `delta_A`, `delta_B` | 1.83x | exact arithmetic |
| 3. the concentration step's own remaining slack | **5.9x** | residual after removing 1 and 2 |
| product | **30.5x** | equals sealed `E_Q` / realised error |

with the numbers behind it:

```text
realised max error                      0.0063
sealed E_Q                              0.1924     30.5x realised
after removing the envelope term        0.0684     10.9x realised
after also tightening the constants     0.0373      5.9x realised
```

Two readings of this table matter more than the table:

- **Source 3 is the largest single term**, and neither of the two popular
  levers touches it. A project that spends its budget re-deriving Hoeffding
  constants can at best move the total from 30.5x to 16.7x.
- **Even with sources 1 and 2 fully removed, `E_Q` stays 5.9x above the
  realised error.** Any claim that the wall is a bookkeeping artifact is
  therefore false on this record, and section 5.2 is the attempt to say what
  the 5.9x actually is.

### 3.6 Precision budget against the coverage requirement

Full coverage needs `E_Q` at 0.262 of its current value, i.e. a 3.8x reduction.
Sources 1 and 2 together supply 5.1x *if they multiply cleanly*, which would be
more than enough — but they do not multiply cleanly across records, because
coverage is pinned by whichever record has the smallest threshold rather than
by the median. The constant repair alone takes k to 0.546 (35/48) and the worst
record still needs 0.262. **After the best available constant repair, a further
2.1x is required, and source 3 is the only place it can come from.** That is
precisely why section 5.2, not another constant audit, is the load-bearing part
of this proposal.

## 4. Frozen protocol for the measurements this task would add

Nothing in this task needs new trajectories. The primary evidence base is the
existing sealed corpus and a bit-exact reconstruction, so the task is cheap and
its evidence is checkable by hand.

1. **Reconstruction.** Rebuild `Qhat`, the held-out residuals, the per-pair
   halves, the scales, the radii, the residual means, `E_Q`, and the emission
   decision for all 48 primary route-records from source. Every reconstructed
   quantity must match the sealed records bit-for-bit before any new statistic
   is reported. Any mismatch stops the task.
2. **Primary statistic.** The accuracy ratio
   `permitted error / realised error` at the selected `eta`, reported for all
   192 state-pairs as a full distribution, not a mean.
3. **Wall statistic.** Per-record critical `E_Q` by bisection on the frozen
   grid, and the multiplier `k` needed for 100% coverage.
4. **Decomposition.** Per-pair split of `s_x^2` into `mean(Y^2)` and the
   envelope term, each reported in absolute units.
5. **Sample-size sweep.** Recomputed certificates on disjoint subsamples of the
   sealed certification items at a frozen ladder of per-pair counts (for
   example 10000 / 20000 / 40000 / 80000), reporting `E_Q` and emission per
   rung. This is the only experiment that re-reads raw items; it needs no new
   rollouts.
6. **Environment sweep.** Re-run the frozen evaluator on a small ladder of
   value-separation parameters (for example the gap bonus at 0.25 / 0.5 / 1.0 /
   2.0), reporting coverage against the predicted threshold
   `E_Q < c * sigma_s^2 / A_s`. This is the only experiment with new rollouts;
   it must be smoke-sized and its grid frozen before execution.

## 5. Theory this task would attempt

The empirical measurement is not enough on its own; the useful claim is
structural, and structure is provable.

### 5.1 The exact threshold

Fix a state, the frozen relative-softmax family, and an error budget `e`. The
frozen rule needs

```text
Ihat_s(eta) - e * ||pi_eta^+ - pi||_1 > 0
```

Both terms depend on `eta` only through the tilted distribution, so the rule
certifies improvement exactly when

```text
e  <  max over eta  of   Ihat_s(eta) / ||pi_eta^+(.|s) - pi(.|s)||_1,
```

and the right-hand side is a functional of the state's own value vector and
policy row alone. Three consequences follow and are worth stating cleanly in
the task's own words:

1. a state with a flat value vector has a small threshold, and no amount of
   certificate tightening makes a flat state certifiable;
2. the numerator `Ihat_s(eta)` is invariant under adding a constant to the
   state's value vector, because `sum_a (pi_eta^+ - pi) = 0` kills the constant;
3. the denominator is **not** invariant, because `pi_eta^+` is a softmax and
   therefore not translation-equivariant. Raising the whole value vector
   changes the tilt, hence the `l1` distance, hence the threshold. This is the
   exact mechanism behind `FP-SCALE-001`'s measured asymmetry — the value
   spectrum widened 6x while `E_Q` rose 170x — and it is a statement about the
   threshold, not about the certificate.

### 5.2 The structural upper bound on the certificate

**Conjecture (to be proved or falsified).** Let the residuals be known only to
lie in `[-2B, 2B]`, and let the certificate be a function of the empirical
residuals, the split sizes, the risk `delta`, and `B`. Then no such certificate
can produce a simultaneous `E_Q` smaller than a quantity proportional to
`B * sqrt(log(1/delta) / N)`, and the proportionality constant is bounded below
by the requirement that the bound hold against a residual law putting mass
`O(1/N)` at `+-2B` — a law whose sample second moment is small with constant
probability while its true variance is not.

If this holds, it settles the question the project has been circling: **the
wall is not an implementation defect and cannot be removed by tighter
bookkeeping.** If it fails, the failure is itself the result, because it would
exhibit a certificate the project has not tried.

The proposal deliberately states this as a conjecture with a named adversarial
law, so that both outcomes are publishable and neither requires retuning.

### 5.3 Predicted coverage law

Section 5.1 makes the threshold — and therefore the whole coverage curve — a
closed-form functional of each state's value vector and policy row, computable
without any certificate. The prediction to freeze before the environment sweep
is:

> coverage at error budget `E_Q` equals, to within one record of the frozen
> corpus, the fraction of records whose state-threshold minimum exceeds `E_Q`.

This is a strong prediction because it uses no measured certificate at all. It
is also the sharpest available test of the claim that the wall lives in the
threshold structure rather than in the certificate. It must be written down and
committed before the sweep runs, and reported whichever way it goes.

## 6. Prohibited work

Inherited from the project's rules, restated because this task touches sealed
artifacts:

1. No modification of any file sealed by a closed task. Reconstruction reads;
   it does not rewrite.
2. No retuning of any frozen constant of `FP-SCALE-002`, including the risk
   split, the `eta` grid, the protocol parameters, or the emission rule.
3. No new claim about the usefulness of any construction. This task measures a
   certificate property; it does not emit updates and does not claim coverage
   improvements as achieved results.
4. No weakening of the certificate for the purpose of raising coverage. The `k`
   sweep reports a counterfactual curve and must be labelled as such
   everywhere; it is not a valid certificate at `k < 1`.
5. No seed shopping, no post-hoc selection of records, no new metric invented
   after seeing results.
6. No merge to `main`, no push, no publication, no external message without the
   user's separate approval.

## 7. Falsifiable hypotheses

| id | hypothesis | status now | falsified by |
|---|---|---|---|
| `W1` | The median accuracy ratio over the 192 state-pairs exceeds 50 | **already measured: 83.6** | a median below 10 |
| `W2` | The certificate's permitted error is bounded below by a structural constant times `B * sqrt(log(1/delta)/N)`, independent of the residual law | open, load-bearing | exhibiting a certificate meeting the same simultaneous guarantee with a strictly smaller constant at the same counts and risk |
| `W3` | The envelope term contributes more than half of `s_x^2` for most pairs | **already measured: 69.5%--99.6%, median 93.2%, on one record** | the envelope share below one quarter for a majority of pairs |
| `W4` | Coverage is predicted by the state-threshold distribution of section 5.1 | open; prediction must be frozen before the sweep | the prediction missing the measured coverage by more than one record in 48 |
| `W5` | There is a knee: coverage rises from below 50% to above 95% within a 3.5x range of `k`, then saturates | **already visible: 45.8% at k=1, 95.8% at k=0.3, flat to k=0** | coverage rising roughly linearly in `log k` up to `k=0`, with no saturation |
| `W6` | Full coverage requires at least a 3x reduction in `E_Q` | **already measured: 3.8x (k=0.262)** | a configuration reaching 48/48 with a reduction below 2x |
| `W7` | The best constant bookkeeping buys less than 2x | **already measured: 1.83x (k=0.546)** | a legitimate constant derivation yielding more than 2x |

`W2` is the only hypothesis with real risk of an OPEN verdict, and it is the one
that carries the result. `W1`, `W3`, `W5`, `W6`, `W7` are already settled from
the sealed corpus and are listed so that the formal task sheet does not
re-litigate them; `W4` is the bridge between the measurement and the theory.

## 8. Acceptance criteria

1. Reconstruction is bit-exact on all 48 primary route-records, and the check
   is reported as a count of matched quantities, not as a sentence.
2. The accuracy-ratio distribution is reported as a full five-number summary
   plus a histogram, with the `eta` used for each pair recorded.
3. Per-record critical `E_Q` and critical `k` are reported for all 48 records.
4. The per-pair envelope decomposition is reported in absolute units, with the
   per-pair `N_A` used, and sums back to the sealed `s_x^2` to within float
   tolerance.
5. The sample-size sweep reports `E_Q` per rung with the count rule applied
   identically at every rung, and states which rungs fail the rule.
6. The environment sweep states its prediction before execution, reports
   coverage against it, and reports any falsification as a result rather than
   dropping it.
7. `W2` ends as PROVED, FALSIFIED, or OPEN with a stated obstruction. An OPEN
   verdict must name the specific step that could not be closed.
8. Every claim about a measured number is reproducible from the sealed corpus
   by a second reader with no access to the author's intermediate files.
9. No sealed file is modified; a hash check proves it.
10. `ACTIVE_WORKSPACE.md` is updated, and no merge to `main` occurs without the
    user's approval.

## 9. What this would buy, stated honestly

**What it can claim if it succeeds**

- A measured, decomposed conservatism wall for certified policy improvement,
  with the wall's position, its two independent sources, and the precision
  budget required to cross it.
- A structural negative result: within the "bounded residuals plus
  concentration" family, the wall cannot be removed, only approached.
- A reusable, bit-exact reconstruction harness, which the project currently
  lacks and which makes every future claim cheaper to audit.
- An honest accounting of three zero-emission constructions and one positive
  construction as points on one curve, instead of four disconnected reports.

**What it cannot claim**

- Any new algorithm, any better policy, any scaled-up result, any claim that
  softmax policy improvement is useful in practice.
- Generality. The testbed is 4 states and 3 actions with a fixed policy. Every
  number is a property of this testbed plus the sealed corpus.
- Anything about what happens with a different certificate shape. The wall is
  measured for this shape only, and the proposal says so.

**Why this is nevertheless the right next step for this project**

The three items in section 2 already exist. Turning them into a single stated
result costs a small amount of computation and no new algorithmic risk, and it
converts four reports that each read as a partial failure into one report that
reads as a finding. The alternative — trying a fifth construction to chase a
higher emission rate — spends the project's remaining budget on the exact
question that `FP-SCALE-001` and the measurements above already answer in the
negative.

## 10. Roles and open rulings required from the user

Per `AGENTS.md`, the following are the user's rulings and are **not** decided by
this document:

1. Whether to open this direction at all, in place of a fifth construction.
2. Whether `W2` belongs in the same task or a separate theory-only task. It is
   the only item with real risk of an OPEN verdict.
3. Whether the environment sweep (section 4.6) is authorized, since it is the
   only part with new rollouts and therefore the only part with new compute.
4. Which actor authors the formal task sheet and which actor executes. The
   current single-actor exception under `FP-SCALE-002` makes every verification
   in this lineage same-actor, and that limitation should be stated in the new
   task as well if it is not lifted.
5. Whether the four sealed zero-emission results may be cited in a single
   combined statement, which would require a short reconciliation note.

## 11. Risks

| risk | mitigation |
|---|---|
| `W2` ends OPEN and the task produces no headline | the empirical wall measurement (sections 3.1--3.4) stands on its own and is already done |
| the environment sweep is too small to show the predicted trend | freeze the prediction in writing first; a clean falsification is a result |
| the reconstruction harness is mistaken for new science | acceptance criterion 1 and section 9's "cannot claim" list |
| the proposal is read as criticizing `FP-SCALE-002` | it is not: `FP-SCALE-002` produced exactly the artifact this task measures. Without a positive result there would be no non-degenerate point on the coverage curve |

## Appendix: reproducing every number in section 3

All statistics in section 3 come from:

- `results/FP-SCALE-002/claude/formal/task_results.json` (the sealed records),
- a reconstruction that rebuilds `Qhat` through
  `fixed_policy_expected_sarsa_scaled.run_route` and the certificate through
  `fixed_policy_variance_certificate.variance_adaptive_certificate`, using the
  same certification-batch construction as the sealed evaluator,
- an independently written implementation of the frozen emission rule, using
  the extended `eta` grid `(1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)` inherited
  from `FP-SCALE-001` v1.1, which is what `improvement_for` in the scaled
  module uses.

Checks that passed before any statistic was reported:

| check | result |
|---|---|
| reconstructed `e_q` equals sealed `e_q`, all 48 primary route-records | bit-exact |
| reconstructed `max abs(Qhat - Q^pi)` equals sealed `realized_q_sup_error` | bit-exact |
| reconstructed radii / residual means / scales equal sealed values | max deviation 0.000e+00 |
| independent emission rule reproduces sealed emit-or-abstain | 48/48 |

Anyone repeating this needs only the sealed records and the three modules named
above. The audit and measurement scripts used while drafting this proposal live
at the workspace root as `_audit_claims.py`, `_audit_claims2.py`, and
`_wall_measure.py`; they are scratch artifacts and carry no authority.

### Corrections made while drafting

Recorded because the same mistakes would otherwise be repeated:

1. An earlier audit compared a **(4,3) policy matrix** against a per-row softmax
   helper, which silently produced meaningless lower bounds and a spurious
   "7/48 records change" result. Corrected to a strict per-state evaluation,
   after which the emission rule reproduced 48/48 and the spurious effect
   vanished.
2. An earlier claim that **per-state localisation of the certificate error**
   (`E_Q(s)` instead of a global `E_Q`) would recover a median factor of 1.235
   was wrong. It recovers 1.000, because the binding pair always lies in the
   worst state, so `max_s E_Q(s) = E_Q` identically. **That lever is withdrawn.**
3. A first draft of this document described the residual gap after the constant
   repair as "roughly one third". Wrong: the constant repair reaches k=0.546
   and the worst record needs k=0.262, so the residual gap is a further 2.1x.
   Corrected in sections 3.3 and 3.6.
