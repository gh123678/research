# FP-ITER8X-001 same-actor pre-review

Date: 2026-09-12.
Reviewer: Claude, acting as both executor and verifier under the user's standing
instruction of 2026-09-11 ("验证先不管") and the scope instruction of 2026-09-12
("好的").

**This is not an independent review.** There is no second actor.

Task under review: `docs/research_tasks/FP-ITER8X-001.md` v1.0.

Outcome: `APPROVED`, with four conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| The sheet carries every element `AGENTS.md` section 3 requires | PASS |
| The **basis** of each prediction is stated | PASS |
| Falsifying `H3`--`H7` is declared non-invalidating | PASS |
| The horizon is fixed before the run and not re-tuned after | PASS |
| Prohibited work forbids claiming the sampler reproduces the sealed batch | PASS |
| The `1x` comparison's confound (size **and** realisation) is stated | PASS |

## 2. Condition 1 (mandatory): the control arm is not decoration

The question is "does more data extend the iteration", and the tempting shortcut is
to run only the repaired certificate at `8x` and compare against the sealed `1x`
frozen run. That comparison changes **two** things at once — the certificate and the
data — and would not answer the question it claims to.

Condition: run both `frozen@8x` and `empirical_bernstein@8x`, with the `frozen` arm
being the sealed certificate on the larger sample, so the data effect is isolated.
Both arms are reported.

There is a second reason the `frozen` arm matters. The `8x` batch is a **fresh
independent sample**, not a superset of the sealed one, because the vectorised
sampler consumes the random stream differently. So any difference between
`frozen@8x` and the sealed `1x` run is size **plus** realisation, with the
certificate held fixed. The control therefore also makes the confound visible
instead of leaving it to be assumed away — and the task sheet states that the
confound is not separated.

## 3. Condition 2 (mandatory): the horizon must not be the binding constraint

The sealed runs stopped at six because six was the horizon, and nine route-records
used all of it. Running at a horizon of six again would make "it stopped at six"
uninterpretable.

Condition: horizon `12`, so the value under test has room to fail or succeed
visibly. The analyzer reports the **deepest** trajectory reached as well as the
per-step counts, so a horizon-limited result cannot be mistaken for an exhausted
one.

## 4. Condition 3 (mandatory): the containment claim must be checked, not assumed

`H5` asserts that at every step the repaired arm's emitting set contains the frozen
arm's. That is a theorem — `LB_s = Î_s − E_Q‖Δπ_s‖₁` is increasing in `E_Q`, and the
repair only lowers `E_Q` — so it is a check on the **implementation**, and a single
counterexample would mean the two arms are not differing only in their certificate.

Condition: check containment at every non-empty level, list counterexamples
individually, and separately confirm that the arms' trajectories are **not**
identical — otherwise containment would hold vacuously and the "control" would be
an illusion. Both checks are in the verifier.

## 5. Condition 4 (mandatory): one prediction is registered against the lever

`H7` predicts that the step-6 minimum gain under `8x` is **lower** than the sealed
`0.019347`. The reasoning is that a tighter bound admits more records, and a larger
emitting set includes marginal ones, so the minimum over the set should fall even
though the bound improved.

This is registered deliberately in the direction that argues against the lever being
tested. If `H7` passes, the "more data extends the iteration" story has to explain
why it also degrades the worst-case margin; if it fails, the lever looks better.
Writing it the other way round would have been easier and worth less.

## 6. Inheritance fidelity

| element | status |
|---|---|
| protocol, seed, eta grid, decision rule | inherited verbatim from FP-SCALE-002 |
| routes, residuals, ordered reasons, audit | frozen code, untouched |
| `pi_{k-1}` as the step-`k` target | inherited; the audit compares against the matching fixed point, per the FP-ITER2-001 fix |
| batch reuse across steps | inherited: one certification batch per record, reused at every step |
| certification size | `8x`, the change under test |
| horizon | `12`, so the change under test is not truncated |
| certificate | the arm; `frozen` is the control |

## 7. Residual risks

| risk | assessment |
|---|---|
| The result is attributed to the wrong lever | Addressed by condition 1's control arm. |
| A horizon-limited stop is read as exhaustion | Addressed by condition 2. |
| Containment holds vacuously | Addressed by condition 3's non-identity check. |
| Size is confounded with realisation | Not removable; stated in the sheet, and the control arm makes it observable. |
| `H7` is quietly flipped if it fails | Registered in the unflattering direction; reported as falsified if so. |
| Runtime exceeds budget | Priced by a one-record smoke: 27 s/record, so ~11 min for 24. |
| Same-actor verification | Not mitigated; disclosed. |

## 8. Verdict

**`APPROVED`**, contingent on the four conditions above being enforced executably,
which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.

## Objections

None.
