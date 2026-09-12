# FP-ITER3-001: A third certified step, and where iteration stops

## Task metadata

- Created: 2026-09-11.
- Author: Claude, under the direct user instruction of 2026-09-11 ("好的").
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `9b7cd4bf399d13b77418b3bba4884e9f9e98b611`
  (`main` after FP-ITER2-001).
- Execution branch: `main` (single actor; see the user ruling below).
- Result directory: `results/FP-ITER3-001/claude/`.
- Design:
  `docs/superpowers/specs/2026-09-11-third-certified-step-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-11-third-certified-step-plan.md`.
- Classification: long, conclusion-critical, single-actor under the standing
  user exception.

## Why this task exists

`FP-ITER2-001` showed the certified mechanism is not one-shot: `20` of the `22`
records that emitted once emitted a second time, every second step
componentwise non-degrading and strictly improving, value rising monotonically.

It also produced an **exploratory** observation that makes a sharper third test
possible. Between steps 1 and 2 the mean gain **fell** (`2.717707` →
`2.277997`) while the **minimum** gain **rose** (`0.104197` → `0.779902`). That
is consistent with step 2 being **selection-filtered**: the records that failed
were the ones whose previous step was certified most thinly, so survivors carry
more uniform improvement even as the average shrinks.

This task extends to `MAX_STEPS = 3`. Two questions become answerable that a
two-step run cannot answer:

1. **How far can iteration go?** Where does the attrition actually stop it?
2. **How does the gain decay?** If the mean keeps decaying and the minimum keeps
   rising, iteration is progressively filtering rather than progressively
   reaching an optimum — a distinction that matters for any control claim.

## Research question

Can a third relative-softmax step be certified by the same frozen certificate,
and does the attrition-and-gain pattern predicted by the selection-filtering
hypothesis hold?

## Falsifiable hypotheses

1. `H1 (step-limit monotonicity)`: raising `MAX_STEPS` from `2` to `3` does not
   change steps 1 or 2. Their decisions, selected `eta`, certified errors and
   gains must equal the sealed `FP-ITER2-001` values exactly. This is a
   **mandatory** check: extending the horizon must not perturb the earlier
   trajectory.
2. `H2 (step-2 reproduction)`: this run reproduces the sealed `FP-ITER2-001`
   `20` two-step records exactly.
3. `H3 (third step is certifiable)`: at least one record emits a third step.
4. `H4 (third step is valid)`: every emitted third step is componentwise
   non-degrading and strictly improving in total value.
5. `H5 (monotone value across three steps)`: no emitted step at any level
   degrades any state, so `V^{pi_3} >= V^{pi_2} >= V^{pi_1} >= V^{pi_0}`
   componentwise.
6. `H6 (pre-registered attrition prediction)`: the number of third-step
   emissions is **at most** the number of second-step emissions, and strictly
   fewer if the attrition pattern from step 1 to step 2 continues. Formally:
   `n3 <= n2`, with `n3 < n2` predicted and reported.
7. `H7 (pre-registered filtering prediction)`: the minimum third-step gain is
   **greater than** the minimum second-step gain (`0.779902`), continuing the
   filtering pattern.
8. `H8 (no certificate violations)`: across all three steps, every emitted
   certified error bounds the realized oracle error.

`H1`, `H2`, `H4`, `H8` are mandatory. `H3`, `H6`, `H7` are the substantive,
pre-registered predictions; `H6` or `H7` failing falsifies the
selection-filtering reading and must be reported as such, not reinterpreted.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing
`{0.08, 0.5}`, `12` tasks per mixing, training trajectory `65536` transitions,
certification `16384 x 64 = 1048576` items, `gamma = 0.70`, `alpha = 0.65`,
`160` layers, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`.

### The one structural change

`MAX_STEPS = 3` instead of `2`, in the otherwise unchanged FP-ITER2-001
iteration: identical training and certification batches at every step, the same
frozen certificate and decision code, `pi_{k-1}` as the target at step `k`,
nothing retuned. `H1` exists precisely to prove that the horizon change itself
is inert for the earlier steps.

### Certificate validity across steps

Unchanged from `FP-ITER2-001` and re-stated for completeness: the held-out
residual's conditional expectation
`E[Y_t(Q) | X_t = x] = (T_pi^X Q - Q)(x)` depends only on the transition kernel
and the policy being evaluated, not on the behaviour policy, so the frozen
martingale property and the variance-adaptive bound remain valid for every
`pi_{k-1}` using the same batch.

### Oracle audit

`V^{pi_0..pi_3}` from `policy_quantities` on the regenerated MDP, inside
`oracle_audit` only. Each step's realized error is measured against the fixed
point of the policy being evaluated at that step, `Q^{pi_{k-1}}` — the
correction identified as a bug in `FP-ITER2-001`.

## Prohibited work

- No modification of any sealed program, sealed result bundle, or closed task
  record; the FP-ITER2-001 and FP-SCALE-002 bundles are read-only references.
- No change to any frozen formula, constant, tolerance, eta grid, matrix,
  hypothesis, or metric after the run.
- No resampling between steps; the batches are identical at every step.
- No retuning to force a third emission.
- No reinterpretation of `H6`/`H7` after seeing the result: they are recorded
  in this sheet before the run.
- No claim of independent or reciprocal verification.

## Acceptance criteria

1. `H1` is verified: steps 1 and 2 are bit-identical to the sealed
   `FP-ITER2-001` record, including decisions, `eta`, `E_Q` and gains.
2. `H2` is verified: the `20` two-step records are reproduced exactly.
3. Batches are identical across all three steps, verified by digest equality.
4. Same frozen certificate and decision code at every step.
5. `H3`, `H6`, `H7` are evaluated and reported exactly, with their pre-registered
   predictions stated alongside the outcome.
6. Every emitted step passes componentwise non-degradation; every violation is
   listed individually.
7. Every non-emitting record carries its frozen ordered abstention reason.
8. Exact truth confined to `oracle_audit`.
9. All strict-JSON, duplicate-key, finite, shape, seed and location checks pass.
10. Complete reproducibility evidence.
11. Same-actor derived verification recorded, with the limitation stated.
12. `ACTIVE_WORKSPACE.md` is current.

## Failure criteria

The construction fails if `H1` or `H2` fails, since that would mean the horizon
change perturbed the earlier trajectory or the sealed result is not
reproducible; if the batches differ between steps; if the certificate code
differs; or if any violation is suppressed.

`H3`, `H6` or `H7` failing is **not** a construction failure. `H6`/`H7` failing
would falsify the selection-filtering reading, which is a valuable negative
result about how iteration behaves and must be reported as such.

## Stopping conditions

Stop affected work and notify the user if:

- `H1` or `H2` fails;
- a sealed file is found modified;
- the comparison would require changing a frozen parameter or tolerance;
- execution would expand cost, publication, external communication, or
  permissions beyond authorization.

## Route assignment and verification

Single actor: Claude executes and verifies, under the standing user instruction
of 2026-09-11. The verification is not independent and must be labelled as such,
recompute the comparison through a separate code path, replay the sealed
programs, and state the limitation.

## Pre-review

- Status: `APPROVED` (2026-09-11), same-actor.
- Evidence: `docs/research_branches/FP-ITER3-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-11.
- Decision: "好的" — proceed with the third certified step.
- Scope: this task only. Codex is out of scope by the user's instruction
  ("codex用不了不要管他"); independent verification remains outstanding and is
  not attempted here.

## Definition of done

- [x] Pre-review recorded with no unresolved objection.
- [x] `H1`/`H2` confirm the horizon change is inert and the sealed result
      reproduces (steps 1--2 bit-identical across 70 comparable entries).
- [x] `H1`--`H8` each evaluated with evidence.
- [x] Every abstention carries its frozen reason.
- [x] Same-actor derived verification recorded, with the limitation stated.
- [ ] `ACTIVE_WORKSPACE.md` is current.

## Formal outcome (2026-09-11)

Recorded here for the task index; the route journal at
`docs/research_branches/FP-ITER3-001/claude/first_result.md` holds the full
evidence.

| level | emissions | mean gain | minimum gain |
|---|---|---|---|
| step 1 | `22` (sealed `22`) | `2.71770715611645` (sealed) | `0.10419713678630303` (sealed) |
| step 2 | `20` (sealed `20`) | `2.2779972042823244` (sealed) | `0.7799019383638572` (sealed) |
| **step 3** | **`15`** | **`1.2869055664197442`** | **`0.37819659359698987`** |

- `15` route-records emitted all three certified steps; every emitted step is
  componentwise non-degrading and strictly improving.
- **`0` certificate violations, `0` non-degrading violations** across all three
  levels; the only stopping reason remains `improvement_lcb_nonpositive`.
- `H1`, `H2`, `H3`, `H4`, `H5`, `H6`, `H8` **PASS**.
- **`H7` FALSIFIED**: the minimum gain *fell* (`0.780` → `0.378`), so the
  selection-filtering reading of `FP-ITER2-001` was only half right. Attrition
  is real and monotone (`22 → 20 → 15`), but the margin profile does not improve
  monotonically, because the policy move itself creates new tight states. The
  falsification is reported as such and not reinterpreted.
- Sealed-file integrity re-checked across `FP-SCALE-002`, `FP-ITER2-001` and
  `FP-ATTN-001`: total DRIFT `0`.
- Same-actor derived verification: **PASS**.
