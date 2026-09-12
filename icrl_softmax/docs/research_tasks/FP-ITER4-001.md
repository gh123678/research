# FP-ITER4-001: A fourth certified step, and the shape of the decay

## Task metadata

- Created: 2026-09-11.
- Author: Claude, under the direct user instruction of 2026-09-11 ("第四步").
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `21b2ad75b232c8bdeed399dbd36dcc6d52ca1417`
  (`main` after FP-ATTN-ITER-001).
- Execution branch: `main` (single actor; see the user ruling below).
- Result directory: `results/FP-ITER4-001/claude/`.
- Design:
  `docs/superpowers/specs/2026-09-11-fourth-certified-step-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-11-fourth-certified-step-plan.md`.
- Classification: long, conclusion-critical, single-actor under the standing
  user exception.

## Why this task exists

The certified iteration now has three verified steps and the literal network has
been shown to carry all three identically to numpy. Three points describe a
trend but cannot distinguish candidate laws, and one prediction about that trend
has already been falsified once.

The sealed step 1--3 pattern (`FP-ITER3-001`, numpy, and identically
`FP-ATTN-ITER-001`, network):

| step | emissions | mean gain | minimum gain | maximum gain |
|---|---|---|---|---|
| 1 | `22` | `2.717707` | `0.104197` | `6.751510` |
| 2 | `20` | `2.277997` | `0.779902` | `3.823325` |
| 3 | `15` | `1.286906` | `0.378197` | `2.154661` |

Derived from the sealed record: emission ratios `0.9091, 0.7500`; mean-gain
ratios `0.8382, 0.5649`; the minimum gain is **not monotone**.

A fourth step makes two things possible that three cannot: whether the emission
attrition continues at a decelerating rate, and whether the mean gain's decay is
roughly geometric.

## Research question

Does a fourth certified step exist, and does the attrition-and-gain pattern
continue in the direction the first three steps indicate?

## Falsifiable hypotheses

1. `H1 (step-limit monotonicity, mandatory)`: raising `MAX_STEPS` from `3` to
   `4` does not change steps 1--3. Their decisions, selected `eta`, certified
   errors and gains must equal the sealed `FP-ITER3-001` values exactly.
2. `H2 (step-3 reproduction)`: this run reproduces the sealed `15` three-step
   route-records exactly.
3. `H3 (fourth step is certifiable)`: at least one route-record emits a fourth
   step, so the iteration does not terminate at three.
4. `H4 (fourth step is valid)`: every emitted fourth step is componentwise
   non-degrading and strictly improving in total value.
5. `H5 (monotone value across four steps)`: no emitted step at any level
   degrades any state, so
   `V^{pi_4} >= V^{pi_3} >= V^{pi_2} >= V^{pi_1} >= V^{pi_0}` componentwise.
6. `H6 (no certificate violations)`: across all four steps, every emitted
   certified error bounds the realized oracle error.
7. `H7 (pre-registered attrition prediction)`: `n4 < n3` (`15`), i.e. the
   emitting population continues to shrink.
8. `H8 (pre-registered mean-gain prediction)`: the mean fourth-step gain is
   **lower** than the mean third-step gain (`1.286906`), continuing the decay.
9. `H9 (pre-registered non-vacuity prediction)`: the minimum fourth-step gain is
   **strictly positive** and **not below `0.05`**, i.e. the fourth step is a real
   improvement rather than a numerical hair. This is deliberately weaker than a
   monotonicity claim, because the minimum gain was already shown to be
   non-monotone between steps 2 and 3 and no monotone law is proposed.

`H1`, `H2`, `H4`, `H6` are mandatory. `H3`, `H7`, `H8`, `H9` are the substantive
pre-registered predictions. Each of `H7`, `H8`, `H9` is reported as PASS or
FALSIFIED; a falsification is not reinterpreted.

### Basis of the predictions, stated explicitly

`H7` and `H8` **extrapolate** the sealed trend and are therefore genuine
predictions about data not yet seen.

`H9` is **not** an extrapolation of a monotone law: the observed minimum-gain
sequence `0.104197, 0.779902, 0.378197` is non-monotone, so no monotone trend is
asserted. `H9` only requires the fourth step to remain a non-vacuous
improvement, which is the weakest meaningful continuation claim and is testable
against the possibility that the iteration degenerates into numerically
irrelevant moves.

None of these predictions is derived from the step-4 data; all three are written
here before the run.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing
`{0.08, 0.5}`, `12` tasks per mixing, training trajectory `65536` transitions,
certification `16384 x 64 = 1048576` items, `gamma = 0.70`, `alpha = 0.65`,
`160` layers, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`.

### The one structural change

`MAX_STEPS = 4` instead of `3`, in the otherwise unchanged `FP-ITER3-001`
iteration: identical training and certification batches at every step, the same
frozen certificate and decision code, `pi_{k-1}` as the target at step `k`,
nothing retuned. `H1` exists to prove that the horizon change itself is inert
for the earlier steps.

### Certificate validity across steps

Unchanged and re-stated for completeness: the held-out residual's conditional
expectation `E[Y_t(Q) | X_t = x] = (T_pi^X Q - Q)(x)` depends only on the
transition kernel and the policy being evaluated, not on the behaviour policy,
so the frozen martingale property and the variance-adaptive bound remain valid
for every `pi_{k-1}` using the same batch.

### Oracle audit

`V^{pi_0..pi_4}` from `policy_quantities` on the regenerated MDP, inside
`oracle_audit` only. Each step's realized error is measured against the fixed
point of the policy being evaluated at that step.

## Prohibited work

- No modification of any sealed program, sealed result bundle, or closed task
  record.
- No change to any frozen formula, constant, tolerance, eta grid, matrix,
  hypothesis, or metric after the run.
- No resampling between steps.
- No retuning to force a fourth emission.
- No reinterpretation of `H7`/`H8`/`H9` after seeing the result.
- No claim of independent or reciprocal verification.

## Acceptance criteria

1. `H1`/`H2` confirm the horizon change is inert and the sealed run reproduces.
2. Batches identical across all four steps, verified by digest equality.
3. Same frozen certificate and decision code at every step.
4. `H3`, `H7`, `H8`, `H9` evaluated and reported exactly, with each prediction
   stated alongside its outcome.
5. Every emitted step passes componentwise non-degradation; every violation is
   listed individually.
6. Every non-emitting record carries its frozen ordered abstention reason.
7. Exact truth confined to `oracle_audit`.
8. All strict-JSON, duplicate-key, finite, shape, seed and location checks pass.
9. Complete reproducibility evidence.
10. Same-actor derived verification recorded, with the limitation stated.
11. `ACTIVE_WORKSPACE.md` is current.

## Failure criteria

The construction fails if `H1` or `H2` fails, if batches differ between steps,
if the certificate code differs from the frozen one, or if a violation is
suppressed.

`H3`, `H7`, `H8` or `H9` failing is **not** a construction failure. `H3` failing
would mean the iteration terminates at three, which is a substantive finding
about how far the certificate reaches. `H7`/`H8` failing would falsify the
extrapolated decay reading. `H9` failing would mean the fourth step is
numerically vacuous. Each would be reported as such.

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
- Evidence: `docs/research_branches/FP-ITER4-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-11.
- Decision: "第四步" — extend the certified iteration to a fourth step.
- Scope: this task only. Codex is out of scope by the user's instruction
  ("codex用不了不要管他"); independent verification remains outstanding and is
  not attempted here.

## Definition of done

- [x] Pre-review recorded with no unresolved objection.
- [x] `H1`/`H2` confirm the horizon change is inert (`90/90` entries identical)
      and the sealed result reproduces.
- [x] `H1`--`H9` each evaluated with evidence; all **PASS**.
- [x] Every abstention carries its frozen reason.
- [x] Same-actor derived verification recorded, with the limitation stated.
- [ ] `ACTIVE_WORKSPACE.md` is current.

## Formal outcome (2026-09-11)

Recorded here for the task index; the route journal at
`docs/research_branches/FP-ITER4-001/claude/first_result.md` holds the full
evidence.

| step | emissions | mean gain | minimum gain |
|---|---|---|---|
| 1 | `22` (sealed) | `2.71770715611645` (sealed) | `0.10419713678630303` (sealed) |
| 2 | `20` (sealed) | `2.2779972042823244` (sealed) | `0.7799019383638572` (sealed) |
| 3 | `15` (sealed) | `1.2869055664197442` (sealed) | `0.37819659359698987` (sealed) |
| **4** | **`12`** | **`0.807499658432434`** | **`0.1849020565541281`** |

- Horizon inertness: **all `90` step-1..3 entries bit-identical** to the sealed
  `FP-ITER3-001` run; `15` three-step routes reproduced; `0` reproduction
  failures.
- `12` route-records emitted **all four** certified steps; `105` step executions.
- **`0` certificate violations and `0` non-degrading violations**; the only
  stopping reason remains `improvement_lcb_nonpositive`.
- **All three pre-registered step-4 predictions PASS**: `H7` (`12 < 15`),
  `H8` (`0.8075 < 1.2869`), `H9` (`min4 = 0.1849 > 0.05`, so the fourth step is
  a real improvement rather than a vacuous move).
- Decay shape: emission ratios `0.9091, 0.7500, 0.8000`; mean-gain ratios
  `0.8382, 0.5649, 0.6266`. Both **dip at 2→3 and recover at 3→4**, so neither
  is monotone or cleanly geometric. Four points identify no functional form and
  none is claimed. The data support the **churn** reading from `FP-ITER3-001`
  rather than a smooth decay law.
- The fourth step has so far been executed on the **numpy** route only; the
  network has been verified through three steps.
- Same-actor derived verification: **PASS**.
