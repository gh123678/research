# FP-ITER4-001 same-actor pre-review

Date: 2026-09-11.
Reviewer: Claude, acting as both executor and verifier under the user
instruction of 2026-09-11.

**This is not an independent review.** There is no second actor.

Task under review: `docs/research_tasks/FP-ITER4-001.md` v1.0.

Outcome: `APPROVED`, with three conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| Task sheet carries every element `AGENTS.md` section 3 requires | PASS |
| The predictions' **basis** is stated per prediction | PASS |
| `H3`/`H7`/`H8`/`H9` failure is declared non-invalidating | PASS |
| Prohibited work forbids retuning and post-hoc reinterpretation | PASS |
| Stopping conditions cover `H1`/`H2` failure | PASS |

## 2. Condition 1 (mandatory): state extrapolation versus floor

The sheet marks `H7` and `H8` as **extrapolations** of the sealed trend, and
`H9` explicitly as **not** an extrapolation of a monotone law, because the
observed minimum-gain sequence `0.104197, 0.779902, 0.378197` is non-monotone.
`H9` is only a non-vacuity floor (`> 0.05`).

This matters for how a reader weighs a PASS. Two extrapolated predictions
holding is evidence for the trend; the floor holding only rules out
degeneration. The distinction is in the task sheet, not left to inference.

## 3. Condition 2 (mandatory): `H1` must prove the horizon change inert

`MAX_STEPS` lives in shared code, so raising it could perturb steps 1--3. `H1`
requires steps 1--3 to be **bit-identical** to the sealed `FP-ITER3-001` run
(decision, `eta`, `E_Q`), not merely close. The implementation checks this on a
subset before the full run and on all `90` entries afterwards.

## 4. Condition 3 (mandatory): separate the scientific corpus from the evaluators

This round changes a file that a previously sealed task recorded by hash. The
review therefore requires the distinction to be explicit:

- the **scientific corpus** — `fixed_policy_expected_sarsa.py`,
  `fixed_policy_expected_sarsa_scaled.py`,
  `fixed_policy_variance_certificate.py`, `model.py`,
  `verify_variance_adaptive_certificate.py` — must be byte-identical everywhere;
- the **shared task evaluators** evolve as horizons are extended.

A changed evaluator must be reported explicitly and paired with an inertness
proof, not silently tolerated and not left to fail an older task's report.
Section 5 of the plan makes repairing `FP-ATTN-ITER-001`'s record part of
closure.

## 5. Inheritance fidelity

| element | status |
|---|---|
| protocol dimensions, lengths, mixing, gap, seed | inherited from FP-ITER3-001 |
| certificate and decision code | same frozen modules, no fork |
| `pi_{k-1}` as step-`k` target | inherited |
| one training and one certification batch per record, reused at every step | inherited, digest-recorded |
| oracle audit, realized error against `Q^{pi_{k-1}}` | inherited |
| `MAX_STEPS` | the only change, `3` → `4` |

## 6. Residual risks

| risk | assessment |
|---|---|
| Horizon change perturbs earlier steps | Checked by `H1` on all 90 entries. |
| Shared-evaluator hash drift misreported | Addressed by condition 3 and by the closure repair. |
| Extrapolated predictions fail | Acceptable and informative; each is reported as PASS or FALSIFIED. |
| Same-actor verification | Not mitigated; disclosed. |

## 7. Verdict

**`APPROVED`**, contingent on the three conditions above being enforced
executably, which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.
