# FP-SAMPLE-001 same-actor pre-review

Date: 2026-09-12.
Reviewer: Claude, acting as both executor and verifier under the user's standing
instruction of 2026-09-11 ("验证先不管") and the scope instruction of 2026-09-12
("都做").

**This is not an independent review.** There is no second actor.

Task under review: `docs/research_tasks/FP-SAMPLE-001.md` v1.0.

Outcome: `APPROVED`, with four conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| The sheet carries every element `AGENTS.md` section 3 requires | PASS |
| Each prediction's **basis** is stated per prediction | PASS |
| Falsifying `H3`--`H7` is declared non-invalidating | PASS |
| Prohibited work forbids using the new sampler where the sealed one is required | PASS |
| Stopping conditions make the sampler gate fatal to the whole ladder | PASS |
| Step-1-only scope stated; the iteration explicitly out of scope | PASS |

## 2. Condition 1 (mandatory): the sampler gate must run before the ladder is read

A vectorised sampler is the only way to reach `8x` in reasonable time, and the
reason it is *not* interchangeable with the sealed generator is precisely the defect
`FP-CENSUS-001`'s batch guard caught: `Generator.choice` with an explicit `p` does
not consume the random stream the way raw uniform draws do. A sampler that were fast
but wrong would shift `E_Q` at every rung and invalidate the entire curve, and the
curve would still look smooth and monotone — the failure mode with no symptom.

Condition: the gate is a mandatory hypothesis (`H1`), its numbers are reported rather
than asserted, its thresholds are registered before the run (total-variation
distance `< 0.02`, residual second moment within `2%`), and the analyzer **aborts
the interpretation** if it fails. Gate result: worst TV `0.0091`, worst `E[Y^2]`
deviation `0.9%`.

## 3. Condition 2 (mandatory): the sampler must not leak into the sealed path

The ladder needs a fresh independent sample; the sealed evaluators need the sealed
batch. Mixing them would silently change results that earlier tasks sealed.

Condition: no sealed evaluator may import the sampler. The verifier greps the four
step-1 evaluators for it, and the sampler module carries a self-test rather than
being wired into any protocol path.

## 4. Condition 3 (mandatory): the size/realisation confound is stated, not assumed away

Every rung is a fresh independent draw, **not** a superset of the sealed batch,
because the sampler consumes the stream differently. Size and realisation are
therefore not separated.

Condition: the task sheet states the confound and its direction. It is recorded
because it is the kind of caveat that is easy to omit when the result is clean, and
here the result is clean: the curve is monotone and the revival counts are monotone,
which is exactly when a reader is least likely to ask.

The direction argument is that the confound works *against* a clean monotone curve —
each larger rung is a different realisation, so a spurious monotone sequence is
harder to produce, not easier. That argument is recorded as an argument, not as a
proof.

## 5. Condition 4 (mandatory): `H6b` must be measured through the right baseline

The ladder computes only the two repaired certificates. It has **no frozen arm**, so
"`22` of the `26` abstainers revive" is only measurable by joining with
`FP-TIGHT-001`'s frozen bundle. The ladder's own `1x` rung already has the repair
applied and abstains on `18` route-records, not `26`; quietly re-basing the claim
onto that number would report a different, easier result.

Condition: the analyzer must load `FP-TIGHT-001`'s bundle, define the abstainer set
from its **frozen** arm, and report both baselines side by side so the difference is
visible. It does.

This condition is not hypothetical. `H5` failed for exactly this class of error —
see §6 — and the same mistake would have been invisible in `H6b`.

## 6. A prediction-anchoring error this review catches, in advance of the result

`H5`'s band was written as `[−70%, −50%]`, anchored on `FP-TIGHT-001`'s subset
measurement of `−64%`. That figure was measured against the **frozen certificate**,
so it contains both the repair and the size effect. The ladder's reductions are
measured against its **own `1x` rung**, which already has the repair, so they contain
the size effect alone. The two are not the same quantity and the band is not
comparable.

The anchor is left as registered — re-scoring a band after seeing the result is
worse than reporting it falsified — and the mismatch is documented in the result
record. The general rule this incident produces, recorded in the result: **when two
numbers are compared, state what each is measured against before writing the band.**

## 7. Inheritance fidelity

| element | status |
|---|---|
| protocol dimensions, mixing, gap, seed, eta grid | inherited from FP-SCALE-002 |
| certified quantity (`\|\|Qhat - Q^pi\|\|_inf`) | unchanged |
| certificate | `fixed_policy_bernstein_certificate.py`, as `FP-TIGHT-001` left it |
| decision rule | frozen code, untouched |
| certification sample | the only change: size, and a fresh independent draw |

## 8. Residual risks

| risk | assessment |
|---|---|
| The sampler is biased | `H1`, mandatory, with registered thresholds, and the curve is void if it fails. |
| The sampler leaks into a sealed path | Verifier greps the evaluators. |
| Size is confounded with realisation | Stated in the sheet and the result, with the direction argued. |
| `H6b` measured against the wrong baseline | Addressed by condition 4. |
| `H5` band mis-anchored | Recorded in advance; reported falsified rather than re-scored. |
| The straggler costs are read as predictions | Labelled an extrapolation, with a check that the fit reproduces the rung it was fitted on, and `617x` glossed as "at its floor". |
| Step-1-only scope overstated | Stated in the sheet and the result. |
| Same-actor verification | Not mitigated; disclosed. |

## 9. Verdict

**`APPROVED`**, contingent on the four conditions above being enforced executably,
which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.

## Objections

None.

## Addendum: a bug caught during execution

The first version of the straggler extrapolation fitted the **population mean**
`eps_res` and applied it to individual route-records. A record's `E_Q` is not the
population mean, and the error surfaced as the visible absurdity of predicting a
crossing at `6.3x` for a record already measured at `8x` without crossing. The fix
is a per-record fit, and the analyzer now reports whether each fit reproduces the
top rung it was fitted on.

Recorded here because it is the cheapest class of check available — a derived number
that contradicts a measurement already in hand is wrong — and because the check is
now automatic rather than depending on someone noticing.
