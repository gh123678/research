# FP-KERN-001: Data-derived cross-state kernel feasibility

## Task metadata

- Created: 2026-09-09.
- Author: GPT.
- Status: `ACTIVE`.
- Task version: `1.0` (activated after Claude `APPROVED`; scientific
  hypotheses, routes, matrix, thresholds, and decision rule unchanged).
- Verified scientific baseline:
  `c579047950dfabb2600020cd2e53dd24b3e39c84`.
- Approved design commit:
  `f10d04ce1bec6d103c0e1e1f608db4d0d00a2b5e`.
- DRAFT task-definition baseline:
  `b7ef163f11eb5ee41499344c296587efd3516651`.
- Activation commit: recorded immediately after the activation seal.
- Common route execution-start commit: recorded before either route writes
  implementation code.
- Design:
  `docs/superpowers/specs/2026-09-09-kernel-state-generalization-feasibility-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-09-kernel-state-generalization-feasibility-plan.md`.
- GPT branch and worktree: `codex/FP-KERN-001` in
  `results/FP-KERN-001/codex_worktree/`.
- Planned Claude branch and worktree: `claude/FP-KERN-001` in an isolated
  task worktree created from the future common execution-start commit.
- GPT result directory: `results/FP-KERN-001/codex/`.
- Claude result directory: `results/FP-KERN-001/claude/`.
- Classification: long, conclusion-critical, multi-stage empirical task.
- Estimated resources: CPU-only; three trajectories for each of 480 records
  per independent route; expected wall time below six hours per route.

## User ruling and purpose

The user proposed generalizing the same action across different states by a
kernel, confirmed that no external state features exist, approved an empirical
feasibility study before a safety theorem, reviewed the written design, and on
2026-09-09 replied “确认”.

The task tests whether a state kernel learned solely from observed behavior of
the other actions can recover fixed-policy Q values for zero-count and
low-count target state-action pairs. It does not modify, rerun, or reinterpret
`FP-ADV-001`; that separate task and all of its uncommitted artifacts remain
untouched.

## Research question

Without external state features, does a leave-one-action-out empirical state
signature contain enough information to improve fixed-policy Q estimation and
action ordering for zero-count and low-count state-action pairs?

The current independently sampled random MDP family tests applicability to the
existing project. A hidden-cluster family, whose structure is unavailable to
the estimator, tests whether the proposed mechanism can exploit genuine
cross-state structure.

## Falsifiable hypotheses

1. In the hidden-cluster positive control, smaller leave-one-action-out
   signature distance predicts smaller true target-action Q difference.
2. In the hidden-cluster family, the primary kernel route improves zero-count
   Q RMSE over unconditional same-action pooling.
3. In the hidden-cluster family, the primary route improves `1-4` count Q RMSE
   over the local unpooled estimator.
4. In the hidden-cluster family, the primary route improves top-action
   accuracy in sparse states without materially increasing false-improvement
   decisions.
5. Repeating hypotheses 1--4 in the unchanged current random MDP family
   determines applicability; they are not assumed to pass.
6. The same-action anchor kernel may help sparse-positive pairs but cannot
   claim zero-count coverage.

Failure of the applicability hypotheses is a valid result. The task must
classify the outcome as `GENERAL_FEASIBLE`, `STRUCTURE_CONDITIONAL`,
`NOT_SUPPORTED`, or `INVALID` under the frozen decision rule.

## Scientific scope

### In scope

- Finite fixed-policy tabular MDPs with 6 states, 4 actions, and bounded edge
  rewards.
- The unchanged current unstructured `make_mdp`/`make_policy` family.
- One hidden two-cluster positive-control family with structure strength
  `0.90/0.10`, hidden cluster assignments, and the existing sticky-mixing
  convention.
- Three independent stationary-start trajectories per record: value,
  signature, and target.
- The unchanged exact V-first state-value estimator as a common nuisance
  estimate for all routes.
- Four routes: local unpooled, unconditional action pooling, same-action anchor
  kernel, and leave-one-action-out kernel.
- Exact true quantities only in a structurally separate oracle audit.
- Empirical Q error, action ordering, coverage, false improvement, and policy
  value diagnostics.

### Out of scope

- A safety certificate, confidence sequence, policy-nondegradation theorem,
  online policy control, repeated policy iteration, or manuscript claim.
- External state features, state-index geometry, true-kernel input, learned
  embeddings, low-rank factorization, neural representation learning, or an
  adaptive bandwidth grid.
- Changes to existing MDP, evaluator, certificate, formal-result, or manuscript
  files.
- Reuse of `FP-ADV-001` formal outcomes as training data or theorem evidence.
- Post-hoc route, bandwidth, environment, threshold, metric, or decision-rule
  selection.

## Frozen observable construction

Let `V_hat` be the common value estimate from the value trajectory. On the
signature trajectory define, for every visited pair,

```text
q_sig(s,b) = mean [R_{t+1} + gamma V_hat(S_{t+1}) | S_t=s, A_t=b].
```

For target action `a`, two states may be compared only when at least two common
non-target actions have positive signature counts. With
`B = R_star/(1-gamma)`, the leave-one-action-out distance is

```text
d_a(s,s_prime)^2
  = mean_{b != a, both visited}
      [(q_sig(s,b)-q_sig(s_prime,b))/(2B)]^2.
```

The bandwidth is the median of the finite positive eligible distances within
the record and target action. A missing or nonpositive median causes an
abstention. The primary weight is

```text
K_a(s,s_prime) = exp(-d_a(s,s_prime)^2/(2 h_a^2)).
```

On the target trajectory, with
`Y_t = R_{t+1} + gamma V_hat(S_{t+1})`, define

```text
Q_hat_kernel(s,a)
  = sum_s_prime K_a(s,s_prime)
        sum_{t:(S_t,A_t)=(s_prime,a)} Y_t
    / sum_s_prime K_a(s,s_prime) N_target(s_prime,a).
```

The pure implementation must derive all masks, distances, bandwidths, weights,
counts, and estimates from declared observable inputs. A nonpositive or
nonfinite denominator causes an explicit abstention.

The same-action anchor control applies an analogous scalar distance using the
target action on the signature trajectory. It is a sparse-positive control:
it must abstain whenever the target-trajectory pair count is zero or the target
state has no target-action signature observation.

## Frozen routes

1. `local_unpooled`: target-pair mean; unavailable at zero target count.
2. `action_only_pool`: count-weighted target-action mean across all states,
   with no similarity kernel.
3. `same_action_anchor_kernel`: same-action empirical similarity control;
   available only at positive target count and positive signature count.
4. `leave_one_action_out_kernel`: primary other-action state-signature route.

All routes share the same MDP, fixed policy, value estimate, and three
trajectories within a record.

## Hidden-cluster positive control

Each record contains two balanced clusters of three states with a random label
permutation. For cluster `c` and action `a`, sample prototype transition and
bounded edge-reward rows plus independent state-specific rows. Freeze

```text
P_struct(s,a) = 0.90 P_proto(c,a) + 0.10 P_independent(s,a)
P(s,a) = (1-mixing) point_mass(s) + mixing P_struct(s,a)
R(s,a,s_next)
  = 0.90 R_proto(c,a,s_next) + 0.10 R_independent(s,a,s_next).
```

Apply the unchanged nonnegative reward-gap bonus to action zero. Initial-state
probabilities retain the existing Dirichlet construction. Hidden cluster and
prototype values are oracle-audit fields only.

## Frozen formal protocol

- Environment families: current unstructured and hidden-cluster.
- Tasks per cell: 15.
- States/actions: 6/4.
- Trajectory lengths for each independent stream: 256, 1024, 4096, 16384.
- Mixing settings: 0.08, 0.50.
- Reward-gap bonuses: 0, 0.50.
- Minimum action probability: 0.05.
- Gamma/value iterations: 0.70/160.
- Seed: 20260909, with deterministic nonoverlapping substreams.
- Total records per route: `2 * 15 * 4 * 2 * 2 = 480`.

Before either independent formal run, each route must pass a labelled smoke
matrix with one task, both families, lengths 256 and 1024, both mixing values,
and both gap bonuses. Smoke can expose implementation defects but cannot change
the scientific contract. Each route may run the formal matrix exactly once
unless the user explicitly approves a documented exception.

## Metrics and decision rule

Report count bins `0`, `1-4`, `5-16`, and `17+`, where count always means the
target-trajectory pair count. The primary route never reads the target action's
signature-trajectory value, even when that auxiliary count is positive.
Comparisons use only pairs on which both compared routes have finite outputs;
coverage always uses all target pairs and cannot be restricted to successful
emissions.

Primary metrics are zero-count coverage, Q RMSE/MAE, pairwise action-order
accuracy, sparse-state top-action accuracy, false-improvement rate, and the
oracle return/componentwise value change of a descriptive greedy-with-floor
policy. That diagnostic policy is not a safety claim; incomplete route rows
retain the original policy row.

Use per-record paired differences and two-sided 95% Student-t intervals. The
hidden-cluster mechanism screen passes only if the primary route:

1. estimates at least 50% of signature-eligible zero-count pairs;
2. reduces mean zero-count RMSE by at least 10% versus `action_only_pool`, with
   the paired improvement interval excluding zero;
3. reduces mean `1-4` count RMSE by at least 10% versus `local_unpooled`, with
   the paired improvement interval excluding zero;
4. improves sparse-state top-action accuracy by at least 5 percentage points
   versus `action_only_pool`, with the paired interval excluding zero; and
5. has false-improvement rate no more than one percentage point above
   `action_only_pool` on common finite comparisons.

Apply the identical screen independently to the current family:

- `GENERAL_FEASIBLE`: both families pass;
- `STRUCTURE_CONDITIONAL`: hidden-cluster passes, current family does not;
- `NOT_SUPPORTED`: hidden-cluster fails;
- `INVALID`: implementation, provenance, isolation, or frozen-contract checks
  fail.

### Frozen metric clarifications after pre-review

The following rules close Claude's nonblocking pre-review cautions without
changing the five-item screen:

1. A record contributes a route-paired count-bin error difference only when
   both routes have at least one finite estimate in that bin. Empty-bin records
   are excluded from that paired mean, their exclusion count is reported, and
   an interval with fewer than two contributing records is unavailable and
   cannot pass a screen item.
2. A zero-target-count pair is `signature_eligible` when the primary route has
   at least one other state with two common observed non-target actions and the
   record/action has a finite positive bandwidth. Thus reasons
   `insufficient_common_actions` and `bandwidth_unavailable` are outside the
   eligibility denominator; `target_source_unavailable`, denominator failure,
   and estimate failure remain in the denominator and count as uncovered.
3. Median bandwidth uses sorted float64 distances. For an odd number it is the
   central value; for an even number it is the arithmetic mean of the two
   central values, matching `numpy.median`.
4. Effective sample size is observation-weighted. With one weight
   `K_a(s,s_prime)` for each of the `N_target(s_prime,a)` observations,

   ```text
   ESS(s,a)
     = [sum_s_prime K_a(s,s_prime) N_target(s_prime,a)]^2
       / sum_s_prime K_a(s,s_prime)^2 N_target(s_prime,a).
   ```

   A nonpositive or nonfinite ESS denominator follows the frozen denominator
   failure path.
5. Hypothesis 1 is a secondary falsifiable diagnostic and does not replace the
   five-item mechanism screen. Within each record/action, compute Spearman
   correlation between every eligible cross-state signature distance and the
   corresponding absolute true target-action Q difference. Average only
   finite per-record correlations and report a two-sided 95% Student-t
   interval. Hypothesis 1 passes in a family only when mean correlation is
   positive and the interval excludes zero; fewer than two finite records is
   an unavailable, nonpassing diagnostic.
6. The common `V_hat` is an independently estimated shared nuisance value. It
   reflects fixed-policy continuation, which includes all actions at successor
   states. Therefore “leave one action out” means the target action is absent
   from the *state signature coordinates*; it does not mean the shared value
   nuisance is mathematically independent of the target action. This common
   dependence is not direct target-trajectory leakage and must be stated in
   every theory/result note.
7. The inherited executable preflight is exactly
   `verify_fixed_policy_q_routes.py`, `verify_finite_sample_theorems.py`,
   `verify_visit_indexed_martingale_certificate.py`, and
   `verify_time_uniform_mixture_certificate.py`, followed by the new verifier
   and task-scoped Ruff.
8. Top-action accuracy includes only record/state rows for which both compared
   routes have finite estimates for all four actions. Each record contributes
   a paired top-action difference only when at least one such common complete
   sparse state exists. Empty records are excluded and counted; fewer than two
   contributing records makes the interval unavailable and nonpassing.
   Diagnostic-policy value remains separately defined: an incomplete route row
   retains the original policy row.

## Ordered abstention and error contract

The pure route uses the following order:

1. `shape_or_dtype_invalid`;
2. `count_inconsistent`;
3. `value_estimate_nonfinite`;
4. `insufficient_common_actions`;
5. `bandwidth_unavailable`;
6. `target_source_unavailable`;
7. `kernel_denominator_invalid`;
8. `estimate_nonfinite`;
9. `route_incomplete_for_policy`.

Reasons 4--7 are ordinary coverage abstentions when inputs are otherwise
valid. Invalid numeric or structural input cannot silently fall back to
uniform weights, an oracle neighbor, another bandwidth, or another route.

## Allowed work

### Shared read-only inputs

- `AGENTS.md`, `ACTIVE_WORKSPACE.md`, this task, its approved design and plan.
- Verified `FP-TU-001` code, task, theory, and result identities.
- The committed `FP-ADV-001` task/design only as motivation; no mutable or
  unsealed result is a scientific input.
- Task-scoped primary-source retrieval if a specific implementation assumption
  requires verification.

### GPT write scope on `codex/FP-KERN-001`

- `kernel_state_generalization.py`;
- `kernel_generalization_mdps.py`;
- `verify_kernel_state_generalization.py`;
- `evaluate_kernel_state_generalization.py`;
- `analyze_kernel_state_generalization.py`;
- `docs/research_branches/FP-KERN-001/codex/`;
- this task, its design, plan, and a compact `ACTIVE_WORKSPACE.md` pointer;
- `results/FP-KERN-001/codex/`.

### Claude write scope on `claude/FP-KERN-001`

- independent versions of the same five Python entry points;
- `docs/research_branches/FP-KERN-001/claude/`;
- `results/FP-KERN-001/claude/`.

Claude may record its theory, first result, formal result, and reciprocal
verification only within its assigned paths. It may not edit the task, design,
plan, workspace index, GPT evidence, or GPT branch.

## Prohibited work

- No research implementation or experiment before status `ACTIVE`.
- No true Q, V, transition, reward, occupancy, cluster label, realized error,
  or exact return in a kernel or estimator input.
- No changes to old code or old result directories.
- No learned bandwidth, result-selected neighbor count, alternative kernel,
  low-rank route, or metric substitution.
- No formal-output tuning, silent fallback, selective denominator, or dropping
  failed/abstained pairs from coverage.
- No access to the other route's first result, code conclusion, or formal
  output before both routes seal corresponding evidence.
- No force operation, history rewrite, merge to `main`, push, publication,
  external message, unsafe permission bypass, or new fee category.

## Expected artifacts

Each GPT and Claude route must provide its five implementation files, verifier,
evaluator, analyzer, theory/method note, first-result seal, formal-result seal,
commands, environment, hashes, raw records, summary, analysis, and final
verification report in its assigned paths.

Each ignored formal result directory must contain at least:

- `config.json`;
- `task_results.json`;
- `summary.json`;
- `analysis.json`;
- `environment.json`;
- `commands.log`;
- `checks.log`.

## Acceptance criteria

1. The implementation exactly matches the frozen four routes, formulas,
   support rules, bandwidth, and environment families.
2. All estimator inputs are observable and every oracle field is structurally
   separated and denied at the pure boundary.
3. Value, signature, and target trajectories use deterministic independent
   random-number streams and are reproducible from recorded seeds.
4. The primary signature excludes the target action and requires two common
   observed non-target actions.
5. Gaussian weights, bandwidths, counts, denominators, effective sample sizes,
   and estimates are finite and reconstructible from serialized observables.
6. Zero-count primary estimates use only other-state target-action
   observations; same-action control never claims unsupported zero coverage.
7. State-label permutation fixtures leave estimates equivariant under
   consistent MDP/trajectory relabelling.
8. Hidden clusters, true quantities, and exact returns never influence route
   selection, weights, bandwidths, estimates, or abstentions.
9. Every malformed input and ordinary abstention follows the frozen ordered
   reason contract without fallback.
10. The smoke matrix passes before the formal run and cannot change the task.
11. Each independent formal route produces exactly 480 records with the
    frozen seed, matrix, task identity, and route names.
12. All count-bin denominators, comparable sets, coverage, errors, orderings,
    policy diagnostics, and paired intervals are independently reconstructed.
13. The final feasibility classification follows the frozen five-item screen
    and cannot be replaced by a favorable cell or secondary metric.
14. All new verifiers, inherited relevant verifiers, strict JSON, schema,
    provenance, hash, and task-scoped Ruff checks pass.
15. Every formal command, environment, anomaly, failure, output hash,
    limitation, and acceptance judgment is recorded.
16. GPT and Claude independently construct the route from one common
    execution-start commit, seal before disclosure, and reciprocally reproduce
    each other's result.
17. Both reciprocal reports end `PASS`, or the task remains unverified pending
    user ruling.
18. `ACTIVE_WORKSPACE.md` is current and `main` remains unchanged without a
    separate user merge approval.

## Failure criteria

The construction is `INVALID` if it reads oracle information, leaks the target
action into the primary signature, changes the frozen bandwidth or matrix,
misstates zero coverage, silently falls back, uses dependent streams where
independence is declared, produces unreconstructible metrics, or cannot be
independently reproduced.

Scientific failure of the hidden-cluster screen yields `NOT_SUPPORTED` rather
than permission to tune the method. Failure only on the current family yields
`STRUCTURE_CONDITIONAL`. A passing current-family screen yields
`GENERAL_FEASIBLE` only if the hidden-cluster screen also passes.

## Stopping conditions

Stop affected work and notify the user if:

- Claude pre-review or final verification returns `OBJECTION`;
- the verified baseline, design commit, branch, or worktree differs;
- another edit overlaps an allowed path;
- implementation requires an oracle input, new model family, bandwidth,
  threshold, route, or metric;
- smoke failure would require a scientific-contract change;
- the formal command has already run and a rerun lacks user authorization;
- Claude is unavailable because of authentication, quota, permission, or
  environment failure;
- isolation, data-transfer, cost, publication, external communication, merge,
  or permission scope would expand beyond authorization.

## Route assignments and independence

GPT independently constructs proof-of-implementation, tests, code, smoke,
formal evidence, and its acceptance assessment on `codex/FP-KERN-001`.

Because this is a long task, the repository's standing user authorization
allows GPT to start local Claude Code automatically for task-scoped read-only
pre-review and, after activation, the isolated independent route. Claude must
return `APPROVED` or `OBJECTION` before implementation. It cannot redefine the
task or use a Codex subagent as a substitute.

Neither route may read the other's result before both blind seals exist. After
disclosure, GPT verifies Claude and Claude verifies GPT with executable,
evidence-linked reports ending exactly `PASS`, `FAIL`, or `OBJECTION`.

## Pre-review, objections, and evidence

### Claude read-only pre-review

- Date: 2026-09-09.
- Status: completed.
- Reviewed commit:
  `c17fa4f28f0ecfb9a08c3c1c8bb11634a7d73801`.
- Tool boundary: Claude Code 2.1.138 in `plan` mode with only `Read`, `Glob`,
  and `Grep`; no Bash, Python, write, edit, Git mutation, or experiment.
- Outcome: `APPROVED`.
- Evidence:
  `docs/research_branches/FP-KERN-001/codex/claude_pre_review.md`.
- Nonblocking cautions: eight metric/reconstruction clarifications, all closed
  in task version 0.3 without changing scientific content.

### Objection

- Status: none recorded.
- Disputed clause: none.
- Evidence: none.
- User ruling: not applicable.

### Execution evidence

- GPT route: not started.
- Claude route: not started.
- Reciprocal verification: not started.

## Definition of done

- [ ] No unresolved objection remains.
- [ ] Both independent routes are reproducible.
- [ ] Both reciprocal verification reports are recorded and pass.
- [ ] Every acceptance criterion has evidence.
- [ ] The classification follows the frozen decision rule.
- [ ] Differences are reconciled or ruled on by the user.
- [ ] `ACTIVE_WORKSPACE.md` is current.
- [ ] The user approved any merge to `main`.
