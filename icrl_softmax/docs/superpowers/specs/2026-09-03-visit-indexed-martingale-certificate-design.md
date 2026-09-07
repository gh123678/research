# Visit-indexed martingale fixed-policy certificate design

> Date: 2026-09-03
> Status: approved by the user on 2026-09-03
> Intended task: `FP-MART-001`
> Scientific code baseline: `main` commit `b4b2769`
> Task-definition baseline rule: commit the complete `DRAFT` contract, design,
> and plan, then record that commit's SHA while moving the task to `REVIEW`
> Execution-start rule: after Claude's approval is recorded, both routes begin
> from the same activation commit

## 1. Motivation

The completed fixed-policy stage gives uniform-in-layer Direct-Q bounds and
same-trajectory V-first no-split bounds. Its prior high-probability certificate
first lower-bounds every state and pair count using stationary occupancy and a
spectral Hoeffding radius. That sufficient condition fails in all 480 frozen
configurations even though observed full pair support rises from 2.5% and 85%
at trajectory lengths 256 and 1024 to 100% at 4096 and 16384. The corresponding
exact-route pathwise bounds are verified at the same rates.

The next bottleneck is therefore not the deterministic Direct-Q or V-first
recurrence. It is the conversion from a trajectory-level numerator bound to a
conditional residual through a conservative occupancy lower bound. This design
replaces that conversion with a visit-indexed martingale argument whose radius
depends directly on the observed number of visits to each required group.

The task does not attempt online control. Existing blockwise experiments have
roughly 30%--35% non-monotone policy blocks, so crossing the policy-change
boundary before fixing the fixed-policy certificate would mix two unresolved
problems.

## 2. Decision and alternatives

The required route is a finite-horizon, simultaneous-in-visit-count Hoeffding
boundary. For each residual group and each possible visit count
`1 <= k <= n`, risk is allocated before seeing the data. This incurs a
`log(n)` term but is elementary, auditable, and sufficient to remove the
occupancy denominator.

A variance-adaptive martingale boundary based on a valid Freedman or empirical
Bernstein construction is a secondary research route. It may become the
preferred numerical radius only if its filtration, observable variance proxy,
and risk allocation are proved without hidden oracle quantities. If two valid
boundaries are selected after looking at the trajectory, their risks must be
split in advance; taking an unadjusted post-hoc minimum is prohibited. Failure
of the variance-adaptive route is a reportable negative result and does not
invalidate the Hoeffding baseline.

A stitched or mixture confidence sequence with `log(log(n))` overhead is out of
the required scope. It may be documented as future work, but it is not an
acceptance gate for `FP-MART-001`.

## 3. Scope and probability contract

The frozen scope remains:

- a finite state-action space;
- one fixed policy with full population support on the target pairs;
- one frozen trajectory, retained under the stationary-start protocol for
  comparability with the completed stage;
- a deterministic edge reward satisfying `|R| <= R_star`;
- `B = R_star / (1 - gamma)`;
- fixed-context, synchronous Direct-Q and state-value iterations;
- exact matching or the existing finite-softmax matching rule.

The martingale proof may not need stationarity or spectral mixing, but this task
does not claim those extensions. Nonstationary starts, additional stochastic
reward noise, changing policies, layer-dependent scores, and fully online
control remain out of scope.

The certificate has a selective guarantee. If `Emit` is the trajectory-based
event that all required observed counts, algorithm contracts, and empirical
kernel margins pass, the claim must be written as

```text
P(Emit and certified error bound is violated) <= delta.
```

It must not be rewritten as `P(Emit) >= 1 - delta`, as a prior full-coverage
claim, or as a conditional coverage statement obtained by dividing by
`P(Emit)`. Missing support is an honest non-emission outcome.

## 4. Martingale objects

Let a transition be sampled in the order

```text
S_t -> A_t -> (R_{t+1}, S_{t+1}) -> A_{t+1}.
```

The proof must explicitly distinguish:

- a pre-action filtration containing the history through `S_t`;
- a post-action, pre-transition filtration containing the history through
  `(S_t, A_t)`.

State-value Bellman increments are centered with respect to the pre-action
filtration, so the conditional expectation integrates both the fixed-policy
action and the next transition. Pair Bellman and fixed-`V^pi` recovery
increments are centered with respect to the post-action filtration. Treating a
state residual as conditionally mean zero after conditioning on the sampled
action is prohibited unless a different target is derived and frozen.

For each state or pair group, define successive visit stopping times using an
indicator measurable in the appropriate pre-observation filtration, and index
the corresponding centered observations by `k`. The proof must not assume that
an arbitrary stopping time is predictable. It must instead verify the exact
measurability needed by the optional-skipping argument and show that the
skipped sequence remains a martingale-difference sequence under the stopped
filtration. It must then cover every deterministic count `1 <= k <= n`
simultaneously, so substituting the random final visit count requires no
conditional-independence assumption.

There are

```text
G = m + 2d
```

residual groups: `m` state Bellman groups, `d` pair Bellman groups, and `d`
fixed-`V^pi` recovery groups. Count indicators are not part of the new
concentration family because observed support is an emission condition rather
than a claimed high-probability event. Dependence among groups is allowed; one
global union allocation is used.

For the mandatory baseline, the target form is an explicit radius such as

```text
r_H(k, delta) = B * sqrt(2 * log(2 * G * n / delta) / k),
```

subject to a line-by-line verification of the actual conditional range and
constants. The final theorem must use the proved constant, not copy this target
form by assumption.

## 5. Computability and route composition

Certificate construction may use only:

- observed state and pair counts;
- observed empirical matching-kernel diagonals or margins;
- the declared reward bound;
- `gamma`, `alpha`, `beta`, iteration count, `delta`, `m`, `d`, and trajectory
  length;
- fixed algorithm-mode metadata.

It may not use true stationary occupancy, a transition matrix, a spectral gap,
`Q^pi`, `V^pi`, a true residual, or a true initial error. With zero
initialization, the computable initial-error upper bound is `B`. Exact truth may
be used only in a separately labelled audit section of formal experiment
outputs.

On the simultaneous martingale event:

- Direct-Q reuses the existing frozen empirical-operator recurrence with the
  visit-indexed pair Bellman radius;
- the V-first state stage uses the visit-indexed state Bellman radius;
- V-first no-split combines that state bound with the visit-indexed fixed-`V^pi`
  recovery radius using the existing deterministic `gamma`-Lipschitz argument;
- exact matching uses its exact empirical diagonal;
- finite-softmax uses the observed empirical diagonal or margin and preserves
  the existing explicit recovery leakage.

The probability step controls only fixed true targets. Data-dependent layer
iterates and early stopping remain covered by deterministic recurrence after
the frozen trajectory has fixed the empirical operators.

## 6. Components and data flow

The intended components are:

- `visit_indexed_martingale_certificate.py`: pure validation, risk allocation,
  radius, emission, and route-composition functions with no sampling, plotting,
  or file-writing side effects;
- `verify_visit_indexed_martingale_certificate.py`: deterministic fixtures for
  visit indexing, simultaneous-count arithmetic, route composition, rejection
  behavior, and strict JSON;
- `evaluate_visit_indexed_certificates.py`: reuse the frozen MDP, policy,
  trajectory, and existing estimators while adding the new certificate fields;
- `analyze_visit_indexed_certificates.py`: compare old common fields, summarize
  emission and audit coverage by trajectory length, and report all failure
  reasons;
- `docs/research_branches/visit_indexed_martingale_certificate_theory.md`: the
  formal filtration, theorem, proof, route corollaries, limitations, and source
  attribution;
- `docs/research_branches/visit_indexed_martingale_certificate_report.md`: the
  frozen 480-item results and route assessment.

The data flow is:

```text
frozen configuration and seed
  -> one frozen trajectory
  -> observed counts and empirical kernels
  -> visit-indexed residual radii
  -> existing deterministic Direct-Q/V-first recurrences
  -> route certificates and strict JSON
  -> paired regression and certificate analysis
```

The evaluator should call existing sampling and estimator functions rather than
forking their definitions. Existing result directories are read-only inputs.
New machine outputs go only below the task-specific result directory.
New fields live under a distinct `visit_indexed_certificate` namespace; the
existing `finite_sample_certificate`, route metrics, and legacy coverage flags
remain byte-for-byte or numerically regression-compatible.

## 7. Rejection and serialization contract

At minimum, the new certificate must distinguish:

- `missing_support` for any required zero observed count;
- `state_kernel_margin_nonpositive` and
  `pair_kernel_margin_nonpositive` for the relevant contraction gate;
- `algorithm_mode_mismatch` for an incompatible update or score rule;
- `risk_budget_invalid` for a malformed delta allocation;
- `numerical_nonfinite` for invalid arithmetic;
- `variance_adaptive_unavailable` when the optional tighter construction has no
  proved or computable inputs.

Failure reasons have deterministic ordering. Unavailable numerical fields are
serialized as JSON `null`; `NaN` and `Infinity` are prohibited. The output must
separate computable certificate fields from oracle audit fields so no consumer
can silently pass truth into certificate construction.

The new route status is named `selective_high_probability_certified` when it
emits. Non-emission and audit-only states use distinct names and do not
overwrite the completed stage's `high_probability_certified` or
`pathwise_bound_verified` meanings.

## 8. Frozen evaluation protocol

The formal matrix contains 480 matched comparisons:

- 30 independent tasks;
- `n_states = 6`, `n_actions = 4`;
- trajectory lengths 256, 1024, 4096, and 16384;
- mixing settings 0.08 and 0.5;
- reward gap bonuses 0 and 0.5;
- `pi_min = 0.05`, `beta = 8`;
- `gamma = 0.70`, `alpha = 0.65`, `delta = 0.05`;
- 160 evaluation iterations;
- seed `20260829`.

Every formal command must spell out these values rather than rely on CLI
defaults. No configuration, threshold, risk allocation, or stopping rule may be
tuned after inspecting formal results. Smoke tests use a separate output path
and cannot be reported as formal evidence.

The expected exact-route emission rates are tied to observed full pair support:
2.5%, 85.0%, 100%, and 100% at the four successive trajectory lengths. A
different result must be attributed to a specific theorem or implementation
gate; parameters may not be changed to restore the expectation. Direct-Q
softmax may emit less often when its empirical pair margin is nonpositive.

Emission means that the claimed probabilistic bound is valid and all route
assumptions are checkable on that trajectory. It does not mean that the bound
beats a deterministic range bound or is numerically useful. The analyzer must
report non-emission, emission, and bound-nontriviality as separate rates; a
vacuous but valid finite bound may not be relabelled as a certificate failure.

Empirical truth-based error coverage is reported for audit, including every
violation and its configuration. It is not used as a substitute for the proof,
and the task does not require zero violations merely because the theorem uses a
nominal delta.

## 9. Isolation and governance

`FP-MART-001` is a long task. GPT and Claude independently execute the same
frozen task, baseline, inputs, seed, and evaluation protocol:

- GPT branch: `codex/FP-MART-001`;
- Claude branch: `claude/FP-MART-001`;
- GPT results: `results/FP-MART-001/codex/`;
- Claude results: `results/FP-MART-001/claude/`.

The scientific code baseline remains `b4b2769`. GPT first commits the complete
`DRAFT` task, approved design, and implementation plan. The next metadata-only
commit records that task-definition baseline while moving the task to
`REVIEW`. After Claude returns `APPROVED`, GPT records the review and activates
the task in one activation commit. Both routes begin execution from that exact
activation commit in separate Git worktrees; neither actor switches branches
or writes results inside the other's worktree.

Before execution, GPT creates the formal task in `DRAFT`, moves it to `REVIEW`,
and Claude returns `APPROVED` or `OBJECTION` from a read-only review. Approval
is recorded before the task becomes `ACTIVE`. An objection moves the task to
`BLOCKED_BY_OBJECTION` and stops affected work for user ruling.

Each route must independently construct the theorem, implementation, verifier,
formal outputs, and evidence record. Neither route reads the other's first
result or conclusion before both first-result commits exist. After disclosure,
GPT reproduces Claude's route and Claude reproduces GPT's route. A Codex
sub-agent may assist GPT but cannot replace Claude's required independent work.

The verified but unmerged `DOC-001` branch is not part of this baseline or task.
Any later user-approved documentation integration must remain separate.

## 10. Acceptance criteria

The formal task is successful only if all required items hold:

1. The state, pair Bellman, and recovery filtrations, visit stopping times,
   martingale-difference properties, conditional ranges, and random-count
   substitution are proved explicitly.
2. One global event simultaneously covers all `G = m + 2d` groups and all
   counts `1 <= k <= n`, with total failure probability at most `delta`.
3. The emitted certificate has the selective probability semantics in Section
   3 and makes no prior full-coverage claim.
4. Certificate code uses no oracle quantity; zero-initialization error is upper
   bounded by `B`.
5. Direct-Q all-layer and V-first no-split exact/softmax bounds are derived from
   the new radii without changing the existing deterministic recurrences.
6. Missing support, nonpositive relevant margins, mode mismatch, invalid risk,
   and nonfinite arithmetic are deterministically rejected.
7. All outputs are strict JSON with a hard separation between certificate and
   audit-only fields.
8. The frozen 480-item run has zero mismatch in all common legacy fields using
   `math.isclose(rel_tol=1e-12, abs_tol=1e-12)` for numeric leaves and exact
   equality for nonnumeric leaves.
9. Exact-route emissions are 100% at lengths 4096 and 16384; all rates and any
   deviations from the four expected support rates are reported without tuning.
10. Existing fixed-policy verifiers and the new contract verifier pass on both
    implementations.
11. Both evidence routes record commits, environments, exact commands, raw
    outputs, failures, metrics, limitations, and criterion-by-criterion
    assessments.
12. GPT and Claude each return `PASS` when verifying the other route, unless the
    user records an explicit exception.

The optional variance-adaptive route is accepted only with its own complete
proof and tests. Its failure does not fail criteria 1--12 when the mandatory
Hoeffding route passes and the negative result is recorded accurately.

## 11. Failure and stopping conditions

Scientific failure includes:

- a claimed residual is not a martingale difference under the stated sampling
  order;
- random-count substitution relies on unproved conditional independence;
- the bound retains the old occupancy denominator or an equivalent hidden
  population-count requirement;
- a supposedly computable certificate uses a true model, occupancy, value, or
  residual;
- observed support is presented as a prior high-probability coverage guarantee;
- common legacy outputs cannot be reproduced from the frozen protocol.

A falsified scientific hypothesis is a valid negative research outcome when
the evidence is complete and reproducible. An implementation failure with a
sound frozen task returns the task from `VERIFYING` to `ACTIVE` for repair.

Stop affected work and notify the user when Claude returns `OBJECTION`, the two
routes cannot be isolated, an unexpected concurrent edit overlaps an allowed
path, the formal protocol would need post-hoc modification, the work expands to
online control or another excluded scope, or Claude is unavailable because of
authentication, quota, permission, or environment failure.

## 12. Resources and deliverables

The task is CPU-only and is expected to require two to four hours per route,
including theorem construction, implementation, the 480-item matrix, and
cross-verification. The user's standing authorization permits automatically
starting local Claude Code for this qualifying long task, but does not permit a
new fee category, unsafe permission bypass, publication, or a merge to `main`.

Required versioned deliverables are the formal task, implementation plan, pure
certificate module, verifier, evaluator, analyzer, theory document, report, and
small necessary figures. Raw and derived experimental outputs remain under the
ignored task-specific result directories. No merge to `main` occurs without
explicit user approval.
