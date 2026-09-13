# FP-GAP-001 same-actor pre-review

Date: 2026-09-12.
Reviewer: Claude, acting as both executor and verifier under the user's standing
instruction of 2026-09-11 ("验证先不管") and the scope instruction of 2026-09-12
("好的").

**This is not an independent review.** There is no second actor.

Task under review: `docs/research_tasks/FP-GAP-001.md` v1.0.

Outcome: `APPROVED`, with four conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| The sheet carries every element `AGENTS.md` section 3 requires | PASS |
| The **basis** of each prediction is stated | PASS |
| Falsifying `H3`--`H6` is declared non-invalidating | PASS |
| Prohibited work forbids choosing a "meaningfulness" threshold post hoc | PASS |
| Prohibited work forbids decomposing the floor into policy-class and iteration parts | PASS |
| The measurement is over sealed data only; no new simulation is permitted | PASS |

## 2. Condition 1 (mandatory): the reconstruction must be proven to close

Every number in this task sits on `v^{pi_k}` reconstructed by cumulating sealed
per-step deltas. If the reconstruction drifted, the gap trajectory would be wrong in a
way that no internal consistency check would reveal — the gap would still be monotone
and would still look like convergence.

Condition: check `start_policy_value + sum(deltas) == final_policy_value` on all `96`
trajectories to floating tolerance, report the worst discrepancy, and make `H1`
mandatory with a stop condition. Result: worst discrepancy `4.441e-16`.

## 3. Condition 2 (mandatory): `v*` must be verified, not assumed

`v*` is the yardstick, and a wrong yardstick would make every gap number wrong while
looking entirely reasonable. Policy iteration is correct here but easy to get subtly
wrong — a mis-indexed transition array or a reward convention that does not match the
rest of the line would still converge to *something*.

Condition: verify `v*` against the Bellman optimality equation on every record
(`H2`, mandatory), and additionally check `v* >= v^{pi_0}` componentwise, which is a
property of the true optimum and would fail for a plausible-looking impostor. Results:
worst Bellman residual `8.882e-16`, zero dominance violations.

## 4. Condition 3 (mandatory): the policy-class caveat travels with every floor

The policy class is the relative-softmax family with `pi_min = 0.15`, not the whole
simplex. The gap to `v*` therefore **cannot** be expected to reach zero, and a plateau
is not by itself a failure of the iteration — part of it belongs to the class.

Condition: the caveat is stored in the bundle, printed wherever a floor is reported,
and asserted present by the verifier. Decomposing the residual is explicitly
prohibited, because it would need a constrained optimum over the reachable family and
this task does not compute one. A report that named a "floor" without the caveat would
be making a claim the data cannot support.

## 5. Condition 4 (mandatory): a mis-specified prediction is reported, not repaired

This task is unusually exposed to the temptation to rewrite a hypothesis after seeing
the answer, because the answer is a *number* and it is easy to argue about which
number should have been registered.

The pre-review therefore requires, in advance: if `H4`, `H5` or `H6` fails, the
registered verdict is reported as `FALSIFIED`, and any corrected metric is presented
as a **separate, clearly labelled post-hoc breakdown** that neither replaces the
registered verdict nor changes the headline hypothesis.

This condition earned its place. All three failed, and all three failed because of
errors on my side:

- `H5` tested for a **vanishing** gain-to-gap ratio, but a geometrically converging
  process has a **constant** one. The test asked for a stronger property than
  convergence and its failure was read as evidence of idling.
- `H4` measured movement against the **final** gap, which is `~1e-10`, so the ratio
  saturates for any movement at all.
- `H6` pooled trajectories that emit `1`–`5` steps (closing `54.89%`) with those that
  emit `24`–`32` (closing `99.97%`), and set a threshold without asking which
  population it applied to.

Keeping the registered verdicts visible is what makes the correction legible: a
reader can see that the pooled number was real and that the conditioned number is
different, rather than being shown only the version that works.

## 6. Inheritance fidelity

| element | status |
|---|---|
| source trajectory | `FP-HORIZON-001`'s sealed bundle, hash recorded in the output |
| `v*` | exact policy iteration on the same `build_task` MDPs |
| no new simulation | no sampler, no certification, no rollouts |
| sealed modules | untouched; hashes checked |

## 7. Residual risks

| risk | assessment |
|---|---|
| The reconstruction drifts | `H1`, mandatory, with the worst discrepancy reported. |
| `v*` is wrong | `H2`, verified two ways. |
| A floor is reported without the class caveat | Addressed by condition 3. |
| A failed hypothesis is recast as a corrected one | Addressed by condition 4, written before the result was known. |
| The gap to `v*` is read as achievable | The caveat, plus the report that `0` trajectories close it to zero. |
| Short and long trajectories are pooled | The length-conditioned breakdown is now a required output. |
| Same-actor verification | Not mitigated; disclosed. |

## 8. Verdict

**`APPROVED`**, contingent on the four conditions above being enforced executably,
which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.

## Objections

None.
