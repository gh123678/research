# FP-HORIZON-001 same-actor pre-review

Date: 2026-09-12.
Reviewer: Claude, acting as both executor and verifier under the user's standing
instruction of 2026-09-11 ("验证先不管") and the scope instruction of 2026-09-12
("好").

**This is not an independent review.** There is no second actor.

Task under review: `docs/research_tasks/FP-HORIZON-001.md` v1.0.

Outcome: `APPROVED`, with four conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| The sheet carries every element `AGENTS.md` section 3 requires | PASS |
| The **basis** of each prediction is stated | PASS |
| Falsifying `H3`--`H6`, `H8` is declared non-invalidating | PASS |
| Prohibited work forbids re-tuning the horizon after the result | PASS |
| The `frozen` control arm is retained and its purpose restated | PASS |
| The sampler confound is restated rather than inherited silently | PASS |

## 2. Condition 1 (mandatory): the horizon must be far enough that a stop means something

`FP-ITER8X-001` ran to `12` and both arms used all twelve steps. Running to `13` or
`16` would risk the same outcome and settle nothing.

Condition: horizon `32`. The observed decline at `8x` is `42, 42, 40, 40, 38, 36, 33,
29, 28, 27, 25, 22` — a mean of about `1.7` per step, so a linear extrapolation from
`22` reaches zero near step `25`. `32` therefore leaves about seven steps of margin
beyond the extrapolated stop, which is what makes "it stopped" a statement about the
method rather than about the budget.

The extrapolation is also the reason the prediction is not free: at `1x` the
population **plateaued** at `12` for two steps and the plateau turned out to be a
horizon artifact. A flattening is a live possibility here too.

## 3. Condition 2 (mandatory): the bundle must carry its own identity

`FP-HORIZON-001` reuses `FP-ITER8X-001`'s evaluator, which hard-coded its own task id
into the bundle. Left alone, this run's bundle would claim to be a different task's
result, and the hash-and-identity checks that other verifiers perform would have
nothing to compare against.

Condition: add a `--task-id` override and thread it through every place the id is
written, then assert in the verifier that the bundle carries `FP-HORIZON-001`. The
sealed `FP-ITER8X-001` bundle is already written and is unaffected.

## 4. Condition 3 (mandatory): "stopped" must be distinguished from "still going"

The headline is a termination claim, and the failure mode is a horizon-limited run
being read as a natural stop — the exact error `FP-ITER6-001`'s plateau induced and
`FP-ITER8X-001` corrected.

Condition: the analyzer reports the **last step at which any emission occurs**
alongside the deepest trajectory, and the verifier asserts that the `H3` verdict is
consistent with that number. If an emission occurs at the final step, the report must
say that termination has **not** been demonstrated and that the horizon is still
binding — not merely that `H3` failed.

## 5. Condition 4 (mandatory): the value claim must be per-state, not just totalled

`H8` asks what the trajectory bought. A large total with one degraded state is a
different and weaker claim than componentwise improvement, and the line has been
strict about that distinction since `FP-SCALE-002`.

Condition: report the cumulative gain **per state**, not only its sum, and have the
verifier recompute the gain from the recorded start and final value vectors rather
than from the analyzer's per-step sums — so the two derivations are independent of
each other within this task.

The verifier additionally checks that no record's start-to-final per-state value
decreases, which is the aggregate form of the per-step non-degradation claim and
catches a violation that happened to be offset within a single step.

## 6. Inheritance fidelity

| element | status |
|---|---|
| protocol, seed, eta grid, decision rule | inherited verbatim |
| routes, residuals, ordered reasons, audit | frozen code, untouched |
| certification | `8x`, as `FP-SAMPLE-001` established and `FP-ITER8X-001` used |
| arms | `frozen` (control) and `empirical_bernstein`, unchanged |
| horizon | `32`, the only change |

## 7. Residual risks

| risk | assessment |
|---|---|
| A horizon-limited stop is read as a natural one | Addressed by condition 3. |
| The bundle claims another task's identity | Addressed by condition 2. |
| A total-value claim hides a degraded state | Addressed by condition 4. |
| A plateau is mistaken for termination | The prediction's basis states the `1x` plateau precedent explicitly. |
| Size is confounded with realisation | Not removable; stated, and the control arm bounds the certificate effect. |
| Margin-zero emissions inflate the reach | The analyzer reports the minimum gain at every step, so a reach built on `1e-4` margins is visible; `FP-ITER8X-001` already showed margins of `0.000183` at step 11. |
| Same-actor verification | Not mitigated; disclosed. |

## 8. Verdict

**`APPROVED`**, contingent on the four conditions above being enforced executably,
which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.

## Objections

None.
