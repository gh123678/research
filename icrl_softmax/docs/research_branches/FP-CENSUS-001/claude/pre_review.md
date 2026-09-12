# FP-CENSUS-001 + FP-ITER6-001 same-actor pre-review

Date: 2026-09-12.
Reviewer: Claude, acting as both executor and verifier under the user instruction
of 2026-09-11 ("验证先不管") and the combined-scope instruction of 2026-09-12
("我的意思是1和2 一起做").

**This is not an independent review.** There is no second actor.

Tasks under review:

- `docs/research_tasks/FP-CENSUS-001.md` v1.0 (eligibility census);
- `docs/research_tasks/FP-ITER6-001.md` v1.0 (sixth certified step, scored against
  the census prediction).

Outcome: `APPROVED`, with four conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| Both sheets carry every element `AGENTS.md` section 3 requires | PASS |
| Each prediction's **basis** is stated per prediction | PASS |
| Failure of the strong hypotheses is declared non-invalidating | PASS |
| Prohibited work forbids retuning, post-hoc thresholds, prediction edits | PASS |
| Stopping conditions cover `H0` failure and a missing prediction file | PASS |
| Combined scope matches the user's instruction to run 1 and 2 together | PASS |

## 2. Condition 1 (mandatory): the census must sit on the sealed batches

The census re-derives `E_Q` at every step. If its certification batch differed
from the sealed one, `E_Q` would differ and the census would describe a different
iteration while looking identical. The mitigation is a hash-and-count check, not
an assertion: the generator is checked against `FP-SCALE-002`'s sealed
`cert_pair_counts` vectors on four records before the census is run.

This condition has already earned its place. The first implementation vectorised
the per-chain loop and **failed** the check on all four records
(`19531/19728`, `25167/25586`, `31673/31583`, `31871/32381`). Had the census been
run on the vectorised generator, every ratio would have been computed against a
certificate the sealed iteration never used, and nothing downstream would have
looked wrong. The literal loop is restored and the guard is kept.

## 3. Condition 2 (mandatory): `H0` before interpretation, and stronger than a flag

`FP-ITER5-001` is the ground truth for who emitted when. A census that merely
matched the emitted/abstained flag could still be measuring a stale `E_Q`. `H0`
therefore compares, per step, the decision, `E_Q` by **exact float equality**, the
selected `eta` and the ordered abstention reasons — plus the row count per route,
since the walk stops at the first abstention exactly as the sealed loop does.

Interpretation of any ratio is gated on `H0` passing at smoke (`8/8`) and at
formal (`48/48`). The smoke gate is not decorative: the plan aborts on failure.

## 4. Condition 3 (mandatory): the threshold rule must not become post hoc

`H2` and `H3` are separation claims, so the analysis scans **every** threshold the
feature can express and reports the minimum misclassification count. The
registered prior — written into the design before the run — is that the exploratory
step-1 ranges **overlap** (`emitted ≈ [2.17, 12.26]`, `blocked ≈ [0.57, 2.21]`), so
`H2` is expected to fail and the informative number is the count, not the verdict.
A threshold with misclassifications is never called a classifier.

For the sixth step, `theta` is frozen from the **step-1** scan, written to
`prediction_step6.json` with a UTC timestamp, and applied to the step-5 ratio
without refitting. Applying a step-1 threshold at step 5 is a deliberate stretch:
`H4` predicts the ratio decays, so the threshold may well be mis-scaled. That is
what makes `H6` a real test rather than a restatement. The prediction file's mtime
and hash are recorded, and the verifier checks it predates the sixth-step bundle.

## 5. Condition 4 (mandatory): the trivial predictor must be scored too

`FP-ITER5-001` found the emitting set unchanged from step 4 to step 5. The obvious
extrapolation — "it stays unchanged" — is a real competitor and is registered as
`P1` alongside the census's `P2`. `H6` requires `P2` to misclassify **strictly
fewer** than `P1`. Without `P1` there would be no way to distinguish "the ratio
predicts" from "nothing changed and any rule that predicts no change wins".

`H7` (`n6 < n5`) and `H7b` (`n6 == n5` with the identical set) are registered as
mutually exclusive competitors so that exactly one is `FALSIFIED`, whichever way
the data falls, including the case where neither holds (same count, different
set). `H7` is deliberately the same extrapolation that already failed once in
`FP-ITER5-001`, re-registered rather than quietly dropped.

## 6. Inheritance fidelity

| element | status |
|---|---|
| protocol dimensions, lengths, mixing, gap, seed | inherited from FP-ITER5-001 |
| certificate and decision code | same frozen modules, no fork, no reimplementation |
| `pi_{k-1}` as step-`k` target | inherited |
| one training and one certification batch per record, reused at every step | inherited, digest-recorded |
| oracle audit, realized error against `Q^{pi_{k-1}}` | inherited |
| `MAX_STEPS` on the numpy path | the only change, `5` → `6` |
| network path | **not** run in this round; disclosed as an outstanding gap |

The census adds `sigma_min` from `policy_quantities` and places it **only** under
`oracle_audit`. It is never passed to `variance_adaptive_certificate` or
`improvement_for`; the verifier greps the call sites to confirm it.

## 7. Residual risks

| risk | assessment |
|---|---|
| Census runs on a non-sealed batch | Checked executably; the first attempt failed and was reverted. |
| `H0` matches a flag but not the certificate | Mitigated: exact `E_Q`, `eta` and reason comparison. |
| Threshold chosen post hoc | Mitigated: full scan, registered prior, frozen `theta`. |
| `P2` reported without a baseline | Mitigated by condition 4. |
| A falsified prediction reinterpreted | Prohibited in both sheets; the analyzer emits PASS/FALSIFIED mechanically. |
| Numpy one step ahead of the network | Accepted and disclosed; a network six-step run is outstanding work. |
| Shared-evaluator hash drift | Bounded to `evaluate_fp_iter2_001.py`; affected verifiers re-run and re-sealed. |
| Same-actor verification | Not mitigated; disclosed. |

## 8. Verdict

**`APPROVED`**, contingent on the four conditions above being enforced
executably, which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.

## Objections

None. No task-level defect was found that would change the research question,
the acceptance criteria or the assignment.
