# FP-SCALE-002 same-actor pre-review

Date: 2026-09-11.
Reviewer: Claude, acting as both executor and verifier under the user
instruction of 2026-09-11 ("不管 codex 了，验证也交给你").

**This is not an independent review.** There is no second actor. `AGENTS.md`
section 4 presumes a reviewer distinct from the author; the user waived that for
this task. It is recorded so the check is auditable, not to claim independence.

Task under review: `docs/research_tasks/FP-SCALE-002.md` v1.0, with
`docs/superpowers/specs/2026-09-11-variance-adaptive-certificate-design.md` and
`docs/superpowers/plans/2026-09-11-variance-adaptive-certificate-plan.md`.

Outcome: `APPROVED`, with two mandatory conditions carried into the
implementation (sections 3 and 5).

## 1. Task/design/plan consistency

| check | result |
|---|---|
| Task sheet carries every element `AGENTS.md` section 3 requires | PASS |
| Design and plan agree with the research question and hypotheses | PASS |
| `H3`--`H5` failure is declared non-invalidating and forbids retuning | PASS |
| The prohibited-work list forbids any fitted constant and any item reuse | PASS |
| The stopping conditions cover a majority count-rule failure | PASS |

## 2. That the concentration argument is the only changed ingredient

Checked against the sealed `fixed_policy_expected_sarsa.py` on `main`:

| ingredient | status |
|---|---|
| exact grouped update, finite-logit scores, canonical memory | inherited unchanged |
| held-out residual definition | inherited unchanged |
| relative-softmax candidate, lower bound, strict `min_s LB_s > 0` rule | inherited unchanged |
| ordered non-emission reasons | inherited unchanged, copied verbatim |
| protocol dimensions, lengths, mixing, gap, seed | inherited from FP-SCALE-001 v1.1 |
| **sub-Gaussian parameter** | **replaced**: envelope `2B` to a two-half estimate |
| **per-pair count rule** | **changed**: `40000` subsample to a `5000`-per-half floor |

The count-rule change is a consequence, not an independent change: the frozen
mixture inversion that forced the `40000` subsample (its 15-point grid only
brackets inside a narrow, non-monotone band) is not used by the
variance-adaptive certificate at all. Verified: the new module imports no
mixture-inversion code.

## 3. Condition 1 (mandatory): no free constant

The predecessor prototype used an asserted `SAFETY = 1.1` inflation on the
estimated residual spread. That is not a theorem and is forbidden here. The
frozen construction instead bounds the **second moment** from half A with the
literal Hoeffding constant:

```text
Z_i = Y_i^2 in [0, (2B)^2]  =>  range/2 = (2B)^2 / 2,
E[Y^2 | x] <= mean_A(Y^2) + 2 B^2 sqrt(2 log(1/delta_A) / N_A) =: V_x,
```

and then uses `s_x = sqrt(V_x)` with the literal factor `2` from Hoeffding's
lemma in `r_x = 2 s_x sqrt(log(1/delta_B)/N_B)`.

Bounding the second moment rather than the variance is deliberate and is what
makes the statement valid without assuming the residual mean vanishes, which it
does not when `Qhat != Q^pi`. The verifier must reconstruct both constants by
hand on a fixture and must assert that no `SAFETY`-like symbol exists anywhere
in the implementation.

`delta_A = delta_B = delta/(2d)` is deterministic, so no risk is tuned.

## 4. No circularity, no reuse

The scale that certifies half B is computed only from half A. The halves are
disjoint and exhaustive by construction, and the split is a deterministic
function of item order, so using the realized `N_A` and `N_B` is valid. The
verifier must independently recompute the scales from A alone and the means from
B alone and require exact agreement.

## 5. Condition 2 (mandatory): the prototype is not evidence

The design probe emitted `31/48` with the asserted scale and `22/48` with the
rigorous one, but it has no sealed verifier and is not independently
reconstructed. Two consequences are carried into the implementation:

1. `H3`'s threshold is set from the **rigorous** prototype (`12` of `48`), not
   the optimistic one, so a pass cannot rest on the asserted constant.
2. Every report must state that the prototype is motivation only.

## 6. Oracle separation

| check | result |
|---|---|
| Certificate inputs: `q_hat`, policy, held-out fields, `R_star`, `gamma`, `delta` only | PASS |
| No oracle field in the certificate return value | PASS (verifier section S10) |
| Truth computed only inside `oracle_audit` | PASS |
| Certification batch cannot reach `Qhat` or hyperparameter selection | PASS: separate RNG stream, `Qhat` built before the batch is read |

## 7. `H5` attribution control

`envelope_control_exact` applies the `2B` envelope bound with the identical
per-pair split sizes and risk allocation on the same tasks and batches, so the
two certified errors are directly comparable. Reviewed and accepted as a
control; it makes no claim of being the sealed inherited implementation, and
the task says so.

## 8. Residual risks

| risk | assessment |
|---|---|
| The second-moment bound is looser than a variance bound | Real; quantified by the prototype (`31/48` to `22/48`) and absorbed by `H3`'s threshold. |
| Some pairs may fall below the `5000`-per-half floor | Real; smoke measured minimums of `78607`--`126108`, far above the floor, and a majority failure trips a stopping condition. |
| Same-actor verification | Not mitigated; disclosed. |
| Reduced matrix not comparable with the `480`-record protocol | Acknowledged; no cross-protocol numeric claim. |

## 9. Verdict

**`APPROVED`**, contingent on the two mandatory conditions in sections 3 and 5
being enforced executably, which the implementation and verifier do.

This is a same-actor approval and carries correspondingly less assurance than
an independent pre-review would.
