# FP-GAP-001: does the certified iteration close the optimality gap?

## Task metadata

- Created: 2026-09-12.
- Author: Claude, under the direct user instruction of 2026-09-12 ("好的"), closing the
  definitional question `FP-HORIZON-001` raised.
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `c1cebd7` (`claude/FP-CENSUS-001`).
- Execution branch: `claude/FP-CENSUS-001`.
- Result directory: `results/FP-GAP-001/claude/`.
- Classification: long, conclusion-critical, single-actor under the standing user
  exception.

## Why this task exists

`FP-HORIZON-001` found the certified iteration still emitting at step `32`, on
minimum gains of `3.845e-07`, and deliberately refused to choose a threshold:

> A gain of `3.8e-7` is certified, componentwise non-degrading and strictly positive,
> and whether that counts as "policy improvement" is a definitional question the line
> has not settled.

As posed, that question cannot be answered by experiment — it is a choice of
definition. **But it can be re-posed as one that can.** Instead of asking "is `3.8e-7`
big enough", ask:

> **Is the iteration making progress on the thing that matters — the distance to the
> optimal policy — or is it emitting certified improvements while that distance sits
> at a floor?**

That is measurable, and its answer does not depend on anyone's choice of threshold.
If the optimality gap is still falling at step `32`, the tail updates are doing real
work and the question of their size is cosmetic. If the gap has **plateaued** while
the per-step gains stay strictly positive, then the iteration is spending certified
steps inside a region it cannot leave, and "improvement" in the certified sense has
come apart from improvement in the decision-relevant sense. That would be a fact
about the method, not a matter of taste.

## Research question

Along the certified trajectory, how does the distance to the optimal value function
evolve, does it reach a floor, and what fraction of the initial suboptimality is
closed by the last step that emits?

## Method, and why it needs no new simulation

Everything required is already sealed.

`FP-HORIZON-001` recorded, for every emitted step of every trajectory,
`oracle_audit.value_delta_vs_previous` — the per-state change in `v^{pi_k}` — along
with `start_policy_value`. So `v^{pi_k}` is reconstructible exactly by cumulating the
deltas, and the reconstruction is checkable against the recorded
`final_policy_value`.

The missing ingredient is `v*`, the optimal value function, which is computable
**exactly** on a `4`-state, `3`-action MDP by policy iteration. So this task is a
measurement over sealed data plus a small exact solve per record: no new rollouts, no
new certification, no sampler.

### A distinction that must be kept

The policy class is the **relative-softmax family with `pi_min = 0.15`**, not the
simplex of all stochastic policies. So the gap to `v*` is **not** expected to reach
zero, and a plateau is not automatically a failure of the iteration — part of the
floor belongs to the policy class.

This task therefore reports the gap to `v*` as the measurement, states plainly that
the floor is a joint property of the policy class and the iteration, and **does not**
claim to decompose the two. Decomposing them would need a constrained optimum over
the reachable family, which is a different task and is not attempted here.

## Falsifiable hypotheses

1. `H1 (reconstruction, mandatory)`: for every one of the `96` trajectories,
   `start_policy_value + sum(value_delta_vs_previous) == final_policy_value` to
   floating tolerance. If this fails, the whole measurement is built on a
   mis-reconstructed trajectory and nothing below is interpretable.
2. `H2 (the optimal values, mandatory)`: `v*` satisfies the Bellman optimality
   equation at every state, and `v* >= v^{pi}` componentwise for every trajectory.
   Both are checked, not assumed.
3. `H3 (pre-registered: the gap is monotone)`: the optimality gap is non-increasing
   at every step of every trajectory. It must be, by construction — the updates are
   componentwise non-degrading — so this is a check on the reconstruction rather than
   a discovery, and a single counterexample falsifies the implementation.
4. `H4 (the headline, pre-registered)`: **the gap reaches a floor before the horizon.**
   For the majority of trajectories, the gap's remaining relative movement over the
   last third of the steps is below `1%` of the gap. Registered as the substantive
   claim: a `PASS` means the certified iteration spends its late steps inside a region
   it is not leaving.
5. `H5 (pre-registered)`: at the final emitting step, the per-step certified gain is
   a **vanishing fraction** of the remaining gap — registered as below `1e-4` for the
   majority of trajectories. This is the quantitative form of "the tail updates are
   real but decision-irrelevant".
6. `H6 (pre-registered: the fraction closed)`: the share of the initial suboptimality
   closed by the end, `(gap_0 - gap_final) / gap_0`, is reported per record with its
   distribution, and is registered as **above `90%`** on average. If the iteration
   closes almost all of the gap and then idles, the tail is decorative; if it closes
   only part of it, the iteration stopped short and the floor is doing the work.
7. `H7 (reporting obligation)`: the step at which the gap first falls within `1%` of
   its final value, per trajectory. That is the "effective stopping step", and it is
   the number a practitioner would want — as distinct from the step at which the
   certificate stops emitting, which `FP-HORIZON-001` showed does not arrive.

`H1`, `H2` are mandatory. `H3`--`H6` are pre-registered; each is reported `PASS` or
`FALSIFIED` and never reinterpreted.

### Basis of the predictions

`H4`, `H5` extrapolate the gain decay `FP-HORIZON-001` measured: mean gain falls
geometrically at roughly `x0.8` per step, from `2.08` at step 1 to `1.5e-4` at step
32, while the population settles onto long plateaus. A process whose per-step gains
decay geometrically while its emitting set is stable is a process converging to
something, and the registered claim is that the something is a **floor it reaches
early** rather than a limit it approaches indefinitely.

`H6` is the prediction that decides how to read the whole result, and it is
registered with a threshold rather than as a reporting obligation because the two
outcomes mean opposite things. Above `90%` closed: the iteration does its job and
then idles, so the tail is a certification artifact. Below `90%`: the iteration
**stops short** of what its own policy class can reach, and the binding constraint is
something this line has not yet identified.

`H7` has no threshold — it is a number to report, not a claim.

## Frozen contract

### Inputs, all sealed

- `results/FP-HORIZON-001/claude/formal/task_results.json` — the `32`-step, `8x`,
  two-arm trajectory.
- The task definitions in `fixed_policy_expected_sarsa_scaled.py` (`build_task`), used
  only to rebuild the same MDPs for the exact optimal-value solve.

### The measurement

Per record, per arm:
`v*` by policy iteration; `v^{pi_k}` by cumulating deltas; the gap trajectory
`gap_k = sum_s (v*_s - v^{pi_k}_s)`; the per-step gain relative to the remaining gap;
the share of the initial gap closed; and the effective stopping step.

### Prohibited work

- No new simulation, no sampler, no certification. This is a measurement over sealed
  data plus an exact solve.
- No modification of any sealed module, sealed bundle, or closed task record.
- No choice of a "meaningfulness" threshold presented as a result. Thresholds that
  appear are pre-registered above, and where none is registered the number is
  reported without a verdict.
- **No decomposition of the floor into "policy class" and "iteration" parts**, which
  would require a constrained optimum this task does not compute.
- No reinterpretation of a falsified prediction.
- No `git add -A`.

## Acceptance criteria

1. `H1` reconstruction verified on all `96` trajectories, with the worst discrepancy
   reported.
2. `H2` `v*` verified against the Bellman optimality equation and against `v^{pi_k}`
   componentwise.
3. `H3`--`H6` each evaluated with evidence.
4. `H7` reported per trajectory with its distribution.
5. The gap trajectory reported per arm, at every step.
6. The policy-class caveat stated wherever a floor is reported.
7. All strict-JSON, finite and location checks pass.
8. Same-actor derived verification recorded.
9. `ACTIVE_WORKSPACE.md` updated.

## Failure criteria

The construction fails if the reconstruction does not close, if `v*` fails the
Bellman optimality equation, if the gap is not monotone, or if a floor is reported
without the policy-class caveat.

`H4`--`H6` failing is **not** a construction failure. `H6` failing — less than `90%`
of the gap closed — would be the more consequential outcome: it would say the
certified iteration **stops short**, and would redirect the line from "how far" to
"why it stalls".

## Stopping conditions

Stop and report if:

- the reconstruction does not close, since nothing below would be interpretable;
- `v*` fails its own verification;
- a sealed module's hash is found changed;
- execution would expand cost beyond the authorised scope.

## Route assignment and verification

Single actor: Claude executes and verifies. By the user's instruction of
2026-09-11 ("验证先不管"), verification is not the focus; the derived checks are
recorded for completeness and no independent verification is claimed.

## Pre-review

- Status: `APPROVED` (2026-09-12), same-actor.
- Evidence: `docs/research_branches/FP-GAP-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-12.
- Decision: "好的" — make the definitional question into a measurable one.
- Scope: this task, on `claude/FP-CENSUS-001`.

## Definition of done

- [ ] `H1` reconstruction verified.
- [ ] `H2` `v*` verified.
- [ ] `H3`--`H6` each evaluated with evidence.
- [ ] `H7` effective stopping step reported.
- [ ] The policy-class caveat stated where a floor is reported.
- [ ] Same-actor derived verification recorded.
- [ ] `ACTIVE_WORKSPACE.md` updated.
