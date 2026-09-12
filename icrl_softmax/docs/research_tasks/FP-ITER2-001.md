# FP-ITER2-001: A second certified relative-softmax improvement step

## Task metadata

- Created: 2026-09-11.
- Author: Claude, under the direct user instruction of 2026-09-11 ("好的").
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `c8d9ac62b4ba17b5309431f7cc2218c407a3f9c1`
  (`main` after FP-ATTN-001).
- Execution branch: `main` (single actor; see the user ruling below).
- Result directory: `results/FP-ITER2-001/claude/`.
- Design:
  `docs/superpowers/specs/2026-09-11-second-certified-step-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-11-second-certified-step-plan.md`.
- Classification: long, conclusion-critical, single-actor under the standing
  user exception.

## Why this task exists

`FP-SCALE-002` produced the project's first certified policy improvement: `22`
of `48` route-records emitted an update that is componentwise non-degrading and
strictly improving in total value, with zero certificate violations, and
`FP-ATTN-001` then showed the literal attention networks reproduce those
decisions exactly.

That establishes the **existence** of the mechanism. It says nothing about
whether the mechanism can be applied **more than once**, which is what the
project's own roadmap calls repeated policy iteration and what any claim about
control would need. One certified step is a demonstration; two is the first
evidence of usability.

The starting point makes this a real test rather than a formality. Inspecting
the `22` certified emissions:

- `min_s LB_s` has median `0.027778` but the minimum is `0.000001`;
- **`18` of `22` have `min_s LB_s < 0.05`**, and `6` have it below `0.01`.

So most first steps were certified by a thin margin. Whether a second step can
be certified at all, and whether it still improves value, is genuinely unknown.

## Research question

Starting from each record whose first certified update was emitted, can a second
relative-softmax step be certified by the same frozen certificate, and does it
preserve componentwise non-degradation and continue to increase total value?

## Falsifiable hypotheses

1. `H1 (step-1 reproduction)`: re-deriving step 1 inside this task reproduces
   the sealed `FP-SCALE-002` first-step decision, selected `eta`, certified
   error and lower bounds exactly, so the iteration starts from the sealed
   state rather than a re-derived approximation.
2. `H2 (second step is certifiable at all)`: at least one of the `22` records
   emits a certified second step.
3. `H3 (second step is valid)`: every emitted second step satisfies
   `V^{pi_2} >= V^{pi_1}` componentwise in the oracle audit, with at least one
   strict improvement `sum_s (V^{pi_2}(s) - V^{pi_1}(s)) > 0`.
4. `H4 (monotone value)`: for every record that emits twice,
   `V^{pi_2} >= V^{pi_1} >= V^{pi_0}` componentwise, so no intermediate
   degradation occurs that cancels out.
5. `H5 (no certificate violations)`: across both steps, every emitted certified
   error bounds the realized oracle error.
6. `H6 (attrition is quantified)`: the number of records certifiable at step 2
   is reported exactly, with each non-emitting record's frozen abstention
   reason, so a negative result is as informative as a positive one.

`H1`, `H3`, `H5` are mandatory. `H2`, `H4`, `H6` are the substantive results;
`H2` failing is a valid and valuable negative result and forbids retuning.

## Frozen contract

### Protocol, inherited verbatim

`4` states, `3` actions, `pi_min = 0.15`, gap bonus `0.5`, mixing
`{0.08, 0.5}`, `12` tasks per mixing, training trajectory `65536` transitions,
certification `16384 x 64 = 1048576` items, `gamma = 0.70`, `alpha = 0.65`,
`160` layers, `R_star = 1.5`, `delta = 0.05`, seed `20260911`, eta grid
`1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`.

### The one structural change: the policy becomes the iterated object

For each record the batches are rebuilt exactly as `FP-SCALE-002` did, and then
the following is applied up to `MAX_STEPS = 2` times.

```text
pi_0 = the frozen behavior/target policy
for k = 1, 2:
    Q_k      = ExpectedSARSA(training batch, pi_{k-1})
    cert_k   = variance-adaptive residual certificate(Q_k, pi_{k-1}, cert batch)
    decide   = relative-softmax rule(pi_{k-1}, Q_k, cert_k)
    if decide abstains: stop, record the frozen reason
    pi_k     = decide.policy_plus
```

`Q_k` uses the **same frozen route code**, applied to policy `pi_{k-1}`. The
certificate and the decision rule are the **same frozen code**, applied to
`pi_{k-1}` and `Q_k`. Nothing is retuned.

### Why the certificate stays valid when the policy changes

The held-out residual is
`Y_t(Q) = R_{t+1} + gamma sum_b pi(b|S_{t+1}) Q(S_{t+1},b) - Q(S_t,A_t)`, and
its conditional expectation

```text
E[Y_t(Q) | X_t = x] = (T_pi^X Q - Q)(x)
```

depends only on the transition kernel and on the policy being evaluated, not on
the behavior policy that generated the data. So the frozen martingale property
and the two-half variance-adaptive bound remain valid for `pi_{k-1}` at every
step, using the same certification batch. This is stated so that the reuse of
one batch across steps is visibly justified rather than assumed.

### Frozen comparison set

The `22` route-records that emitted at step 1 under `FP-SCALE-002` are the
input set. Step-1 reproduction (`H1`) is checked on all `48` route-records, not
only those `22`, so the reproduction claim is not restricted to favourable
records.

### Oracle audit

`V^{pi_0}`, `V^{pi_1}`, `V^{pi_2}` come from `policy_quantities` on the
regenerated MDP, inside `oracle_audit` only. They are never inputs to the
estimator, the certificate, or the decision rule.

## Prohibited work

- No modification of any sealed program, sealed result bundle, or closed task
  record.
- No change to any frozen formula, constant, tolerance, eta grid, matrix,
  hypothesis, or metric after the full matrix is run.
- No resampling, lengthening, or refreshing of the training or certification
  batches between steps: both steps must use the identical batches, which is
  what makes single-batch iteration the claim being tested.
- No retuning of `eta`, the certificate risk, or the policy floor to force a
  second emission.
- No claim that a second-step abstention is a failure; it is a result to be
  quantified under `H6`.
- No claim of independent or reciprocal verification: single actor by ruling.

## Acceptance criteria

1. Step 1 reproduces the sealed `FP-SCALE-002` decision, `eta`, `E_Q` and lower
   bounds exactly on all `48` route-records, or every discrepancy is listed.
2. Both steps use the identical training and certification batches, verified by
   an executable equality check on the regenerated batches.
3. The certificate and decision code used at step 2 is the same frozen code as
   step 1, verified by module identity rather than by inspection alone.
4. `H2` is evaluated on the `22` frozen records with no retuning; the emitted
   count is reported exactly.
5. `H3`, `H4`, `H5` are evaluated per record from the oracle audit, and every
   violation is listed individually rather than summarised away.
6. Every non-emitting record carries its frozen ordered abstention reason.
7. Exact truth appears only inside `oracle_audit`.
8. All strict-JSON, duplicate-key, nonfinite, shape, seed, and location checks
   pass.
9. Complete reproducibility evidence: commands, environment, commits, hashes,
   anomalies, limitations.
10. Same-actor derived verification recorded, with the limitation stated.
11. `ACTIVE_WORKSPACE.md` is current.

## Failure criteria

The construction fails if step 1 does not reproduce the sealed result, if the
two steps use different batches or different certificate code, if any emitted
second step can degrade a state on the certificate event, or if a step-2
abstention is suppressed rather than reported.

`H2` failing is **not** a construction failure. It would mean the certificate is
precise enough for one step but not for iteration, which is a quantitative
statement about the gap between a demonstration and control, and would be the
task's headline result.

## Stopping conditions

Stop affected work and notify the user if:

- step 1 cannot reproduce the sealed result, which would indicate the frozen
  schedule is not reproducible as documented;
- a sealed file is found modified;
- the comparison would require changing a frozen parameter or tolerance;
- execution would expand cost, publication, external communication, or
  permissions beyond authorization.

## Route assignment and verification

Single actor: Claude executes and verifies, under the standing user instruction
of 2026-09-11 ("不管 codex 了，验证也交给你"). The verification is not
independent and must be labelled as such. It must recompute the comparison
through a separate code path, replay the sealed programs, and state the
limitation.

## Pre-review

- Status: `APPROVED` (2026-09-11), same-actor.
- Evidence: `docs/research_branches/FP-ITER2-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope)

- Date: 2026-09-11.
- Decision: "好的" — proceed with the second certified step as the next task.
- Scope: this task only. Codex is out of scope by the user's instruction
  ("codex用不了不要管他"); independent verification remains outstanding and is
  not attempted here.

## Definition of done

- [x] Pre-review recorded with no unresolved objection.
- [x] Step 1 reproduces the sealed result exactly (48/48, `E_Q`, `eta` and
      decision all exact).
- [x] `H1`--`H6` each evaluated with evidence; all **PASS**.
- [x] Every step-2 abstention carries its frozen reason
      (`improvement_lcb_nonpositive` only).
- [x] Same-actor derived verification recorded, with the limitation stated.
- [ ] `ACTIVE_WORKSPACE.md` is current.

## Formal outcome (2026-09-11)

Recorded here for the task index; the route journal at
`docs/research_branches/FP-ITER2-001/claude/first_result.md` holds the full
evidence.

- `48` route-records, `MAX_STEPS = 2`, identical batches at both steps.
- Step-1 reproduction: **48/48 exact**, `0` failures; step-1 emissions `22`,
  equal to the sealed `FP-SCALE-002` count.
- **`20` route-records emitted two steps**; `2` emitted one; `26` emitted none.
- **Step-2 survival `20/22 = 91%`**, with every emitted second step
  componentwise non-degrading and strictly improving.
- **`0` certificate violations, `0` non-degrading violations** across both
  steps.
- Mean gain `2.717707` → `2.277997`; **minimum** gain `0.104197` → `0.779902`.
- The only stopping reason observed is `improvement_lcb_nonpositive`.
- Exploratory (not pre-registered): the rising minimum together with the falling
  mean is consistent with step 2 being selection-filtered, keeping the records
  with real headroom and dropping the marginal ones. A third step is the natural
  next question.
- Same-actor derived verification: **PASS**.
- `main` unchanged beyond this task's own commits; no merge requested.
