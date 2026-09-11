# FP-SCALE-001: A certified relative-softmax improvement step at a reachable certificate scale

## Task metadata

- Created: 2026-09-11.
- Author: drafted by Claude under the direct user instruction of 2026-09-11
  ("不管 codex 了，验证也交给你"), which assigns both execution and verification
  to Claude for this task. This is a deviation from `AGENTS.md` section 7
  (reciprocal verification) and is recorded as a task-scoped user exception in
  "Objections and user rulings" below. It changes no scientific contract.
- Status: `DRAFT`.
- Task version: `1.0`.
- Predecessors: `FP-ESARSA-001` (`VERIFIED` by user exemption, single-route
  evidence), `FP-ADV-001` (`VERIFIED`, negative), `FP-TU-001` (`VERIFIED`,
  certificate baseline), `FP-KERN-001` / `FP-KERN-002` (`VERIFIED`, negative
  cross-state-generalization results, merged to `main` 2026-09-11).
- Scientific baseline: `467ebf591dde8421f7ed5324a18514fb5935d2f9` (`main`
  after the FP-KERN merges).
- Design:
  `docs/superpowers/specs/2026-09-11-reachable-certificate-scale-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-11-reachable-certificate-scale-plan.md`.
- Classification: long, conclusion-critical, single-actor under the user
  exception.
- Estimated resources: CPU-only, no new fee category. One smoke matrix plus
  one frozen formal run. The reduced matrix below is chosen so the bounded
  trajectory length is computationally feasible; the budget is recorded in the
  plan and must be re-measured in smoke before the formal run.

## Why this task exists: the measured obstruction

`FP-ADV-001` emitted `0/480` safe updates on all six routes and
`FP-ESARSA-001` emitted `0/480` on all three routes. A read-only diagnostic on
the sealed `FP-ESARSA-001` records (`results/FP-ESARSA-001/claude/`,
`task_results.json`, `expected_exact` route) localizes the obstruction:

| trajectory length | emitted | mean `E_Q` | mean realized `|Qhat-Q*|` | ratio |
|---|---|---|---|---|
| 1024 | 45 | 125.229 | 0.869 | 144 |
| 4096 | 118 | 62.876 | 0.351 | 179 |
| 16384 | 120 | 27.254 | 0.167 | 163 |

Three facts follow, and together they determine this task's design:

1. **The slack is the concentration radius, not the certificate form.** For the
   emitted records, the worst-pair empirical residual `max_x |Ybar_x|` is at
   most `2.4125`, while the worst-pair radius `max_x r_x` is `37.7612`. The
   certified `E_Q` is therefore almost entirely `r_x/(1-gamma)`; the empirical
   Bellman residual of `Qhat` is not the problem.
2. **The radius is set by the rarest held-out pair.** The frozen mixture
   radius is `r_x = 2B q_mix(N_x)/N_x`, which falls off like `N_x^{-1/2}`;
   a probe of the inherited verified inversion gives `r_x = 37.76` at
   `N_x = 1`, `4.15` at `N_x = 128`, `1.51` at `N_x = 1024`. The worst pair
   count in the sealed matrix was `14`, which alone accounts for the observed
   `E_Q` floor of about `22.5`. Note the artifact "this model is unhelpful" is
   really "this corpus cannot certify its rarest state-action pair".
3. **The inversion is not the culprit.** The frozen mixture radius is only
   about `1.32` to `1.41` times a calibrated two-sided sub-Gaussian radius at
   the same count and risk. Replacing the inversion cannot buy the factor of
   `150` that is needed.

Consequence: neither executing the frozen protocol again nor merely weakening
the guarantee type can succeed, because a `mu`-weighted (occupancy-weighted)
value guarantee has full support over the six states and its radius is still
the worst-pair radius. The obstruction is arithmetic, and it is about how many
observations each state-action pair actually receives.

## Research question

At a protocol whose per-pair held-out occupancy is large enough for the
verified time-uniform mixture certificate to be non-vacuous, can the
fixed-policy Expected SARSA construction of `FP-ESARSA-001` emit a certified
relative-softmax policy update, and does that emitted update measurably
improve value without degrading any state?

This task answers the usefulness question that `FP-ADV-001` and
`FP-ESARSA-001` both answered negatively at an inadequate scale. It does not
reopen their frozen parameters; it is a new protocol, frozen before execution
by the feasibility calculation below.

## Falsifiable hypotheses

1. `H1 (feasibility arithmetic)`: at the frozen per-pair held-out count
   `N_x >= 20000`, the inherited verified mixture radius satisfies
   `r_x/(1-gamma) <= 0.25` in every formal record. This is a checkable
   prediction, not an assumption: if it fails, the protocol is mis-specified
   and the task stops.
2. `H2 (construction carries over)`: the exact grouped and finite-logit
   Expected SARSA routes reproduce batch Expected SARSA within `1e-12` on the
   reduced fixture set, exactly as in `FP-ESARSA-001`.
3. `H3 (certificate validity)`: every emitted `E_Q` bounds the realized oracle
   `||Qhat-Q^pi||_infinity` in every audited record; zero certificate and
   residual-event violations.
4. `H4 (emission)`: at least one primary route emits a certified
   non-degrading relative-softmax update in at least `3` of the frozen formal
   records.
5. `H5 (real improvement)`: every emitted update satisfies
   `V^{pi_plus} >= V^pi` componentwise and `sum_s (V^{pi_plus}(s)-V^pi(s)) > 0`
   in the oracle audit, so at least one emission is a strict, not merely
   non-degrading, improvement.
6. `H6 (gain attribution)`: the emission rate at the reduced scale exceeds the
   `FP-ESARSA-001` emission rate, and the improvement is attributable to the
   per-pair occupancy increase rather than to any change in formula. Zero
   emissions is a valid negative result and forbids retuning.

`H1`--`H3` are mandatory construction and validity claims. `H4`--`H5` are the
usefulness claims. `H6` is the attribution claim.

## Frozen mathematical contract

The estimator, the three routes, the cross-fitted residual certificate, the
relative-softmax improvement rule, the ordered non-emission reasons, and the
oracle separation are inherited unchanged from `FP-ESARSA-001` (task sheet and
`docs/research_branches/fixed_policy_expected_sarsa_theory.md`). The only
frozen changes are the protocol scale and the state-action space, both fixed by
the feasibility arithmetic before any execution.

Explicitly unchanged: the exact grouped update, the finite-logit scores with
`zeta = xi = tau = 8`, the canonical one-token-per-pair memory, the grouped
`alpha/N_x^train` writeback, the held-out residual definition
`Y_t(Qhat) = R_{t+1} + gamma sum_b pi(b|S_{t+1}) Qhat(S_{t+1},b) - Qhat(S_t,A_t)`,
the mixture event with `d = |S||A|` groups at total risk `delta`, the
`E_Q = max_x (|Ybar_x| + r_x)/(1-gamma)` certificate, the eta grid
`1.0, 0.5, 0.2, 0.1, 0.05` in descending order, and every abstention reason.

### Frozen protocol change: state-action space and trajectory length

- States/actions: `4/3`, so `d = 12` pairs, deliberately fewer than the
  inherited `6/4` so a long trajectory can still give every pair a large count.
- Trajectory lengths: `32768` and `131072`, split contiguously in half, so the
  held-out half holds `16384` and `65536` transitions.
- Behavior policy minimum action probability: `0.15` (raised from `0.05`) so
  the per-pair occupancy is more even and the rarest pair is not starved.
- Tasks per cell: `12`.
- Matrix: `2` lengths x `2` mixing settings (`0.08`, `0.5`) x `1` reward-gap
  bonus (`0.5`) = `8` cells, `96` matched records. The gap bonus is fixed at
  the nontrivial value because the zero-bonus cells were the ones whose
  certificates were support-limited in `FP-ESARSA-001`.
- Everything else inherited: `gamma = 0.70`, `alpha = 0.65`, `160` layers,
  `R_star = 1.5` (so `B = 5`), `delta = 0.05`, `15` mixture components,
  `Q_0 = 0`, seed `20260911`.
- `H1` guard: if any formal record has a held-out pair count below `20000`,
  that record is recorded as `heldout_pair_support_missing` and cannot
  contribute to `H4`; if more than half the records are so excluded, `H4` is
  reported as unavailable rather than negative.

### Reduced compute budget

The dominant cost is the `160`-layer grouped iteration over the full
trajectory. At `131072` transitions, 160 layers, 96 records, three routes, the
plan estimates the cost and requires the smoke run to measure it before the
formal run; if projected wall time exceeds the task's stated budget, the task
stops and returns to `ACTIVE` with the measurement, and the protocol may only
be changed by a new frozen revision before any formal inspection.

## Routes

1. `expected_exact`: exact grouped fixed-policy Expected SARSA, primary.
2. `expected_finite`: finite-logit mask-free fixed-policy Expected SARSA,
   primary.
3. `sampled_exact`: exact grouped sampled SARSA using the observed next action,
   control.

## Ordered non-emission reasons

Inherited verbatim from `FP-ESARSA-001`; no reason may trigger a different eta
grid, a different route, a second risk budget, or an oracle fallback.

## Prohibited work

- No post-hoc change to any frozen formula, sharpness, eta rule, split, event,
  risk allocation, seed, matrix, hypothesis, or metric after smoke or formal
  inspection.
- No reuse or reinterpretation of `FP-ESARSA-001` or `FP-ADV-001` frozen
  numbers as evidence for this task; they are motivation only.
- No input-dependent equality mask or visited gate in `expected_finite`.
- No learned transition model, oracle certificate input, or validation leak
  into Q construction.
- No repeated policy iteration, no conditional-on-emission guarantee, and no
  output exploration-floor claim.
- No write to `main` during execution, no push, publication, external message,
  permission bypass, or new fee category without separate authorization.
- No claim of reciprocal verification: this task has a single actor by user
  ruling, and every report must say so.

## Acceptance criteria

1. The reduced fixture set verifies the inherited construction claims at
   `1e-12`, including self-loops, repeated visits, unvisited queries, and the
   absence of masks and gates in the finite route.
2. `H1` is checked per record and reported; any record failing it is excluded
   from `H4` with the frozen reason.
3. Every emitted `E_Q` equals the frozen formula, uses full held-out support,
   and bounds the realized oracle error in every audited record.
4. `H4` is evaluated on the frozen records with no retuning; the emitted count
   is reported exactly.
5. Every emitted policy is finite, strictly positive, row-normalized, changed,
   and componentwise non-degrading in the oracle audit; at least one emission
   is a strict improvement under `H5`.
6. Every abstention returns the input policy bit-for-bit with a frozen reason.
7. Training and held-out transitions are structurally separated; validation
   data cannot influence Q construction or hyperparameter selection.
8. Exact truth, realized errors, action gaps, improvements, values, and returns
   live only in `oracle_audit`.
9. All strict-JSON, duplicate-key, nonfinite, shape, range, identity, seed,
   configuration, and result-location checks pass.
10. The formal result contains exactly the frozen number of matched records,
    each with all three routes, reproducing the frozen generator identities and
    seed schedule.
11. All empirical violations, premise failures, and exclusions are enumerated
    and excluded from theorem evidence.
12. The compute budget is measured in smoke and the formal run is executed
    exactly once.
13. Complete reproducibility evidence is recorded: commands, environment,
    commits, raw hashes, failures, anomalies, and limitations.
14. Verification is performed by the same actor under the recorded user
    exception, and the report states that no independent reconstruction
    occurred. It must be executable and derived, never a restatement of the
    author's conclusions: an independent reconstruction of every frozen input,
    a replay of the sealed programs, and a from-scratch recomputation of the
    reported metrics.
15. `ACTIVE_WORKSPACE.md` is current; `main` is unchanged without separate user
    approval.

## Failure criteria

The mandatory theory fails if the held-out residuals are not martingale-valid,
the inherited mixture cannot be used at random visit counts with scale `2B`,
the Bellman residual does not imply the stated Q bound, the relative-softmax
lower bound is false, or an unstated model or oracle input is required.

The construction fails if the exact route differs from Expected SARSA, the
finite route uses hidden masks or gates, signed residuals are lost, policy
changes during evaluation, validation leaks into Q construction, a pair is
duplicated in an attention candidate set, or an emitted update can degrade any
state on the certificate event.

`H4` or `H5` may fail without invalidating the task, but then the result is a
verified negative usefulness result at a scale where the certificate is
non-vacuous, and the scale explanation above is falsified as the sole
obstruction. That outcome is scientifically valuable and must be reported as
such, without retuning.

## Stopping conditions

Stop affected work and notify the user if:

- `H1` fails in a way that indicates a mis-specified protocol rather than
  ordinary occupancy noise;
- the smoke measurement projects wall time beyond the task budget;
- implementation needs a prohibited mask, gate, oracle, or validation leak;
- smoke or formal output would be needed to choose a frozen parameter;
- an existing uncommitted edit overlaps an authorized path;
- execution would expand data transfer, cost, publication, merge, push,
  external communication, or permissions beyond authorization.

## Route assignments and verification

Under the direct user instruction of 2026-09-11, Claude executes and Claude
verifies this task. The verification is therefore not an independent
construction and must be labelled as such everywhere. To preserve as much
assurance as a single actor can, the verification step must:

- reconstruct every frozen input and generator identity from source rather than
  from the author's serialized summaries;
- replay the sealed programs and confirm exit codes;
- recompute every reported metric from the sealed raw records with an
  independent code path written for the purpose;
- re-derive the `H1` arithmetic from the inherited verified mixture code;
- report `PASS`, `FAIL`, or `OBJECTION` with evidence pointers, and state
  explicitly that no second actor reconstructed the route.

## Objections and user rulings

### Objection

- Status: `NONE`.
- Disputed clause: none.

### User ruling (single-actor execution and verification)

- Date: 2026-09-11.
- Decision: the user instructed that Codex no longer be involved and that
  verification also be assigned to Claude ("不管 codex 了，验证也交给你").
- Scope: execution and verification responsibility for this task only. It
  changes no scientific contract, no acceptance strength, and does not modify
  `AGENTS.md`. Because it waives the reciprocal-verification requirement of
  `AGENTS.md` section 7, this task's verification strength is strictly lower
  than a task closed with two independent actors, and that limitation must be
  stated in every report and in `ACTIVE_WORKSPACE.md`.

### User ruling (direction)

- Date: 2026-09-11.
- Decision: the user chose to pursue a reachable guarantee ("A. 换可达的保证")
  as the next scientific step, and approved merging the verified
  `FP-KERN-001` / `FP-KERN-002` tasks into `main`.
- Scope: direction only. The diagnostic arithmetic in "Why this task exists"
  was produced after that ruling and is recorded here as the frozen
  justification for the protocol scale.

## Definition of done

- [ ] `H1` arithmetic checked and reported.
- [ ] Smoke passes and the compute budget is measured.
- [ ] One frozen formal run executed exactly once.
- [ ] Every acceptance criterion has evidence.
- [ ] Verification replay recorded by the same actor, with the limitation
      stated.
- [ ] `ACTIVE_WORKSPACE.md` is current.
- [ ] The user approves any merge to `main`.
