# FP-MART-001: Visit-indexed martingale certificates for fixed-policy Q evaluation

## Task metadata

- Created: 2026-09-03
- Author: GPT
- Status: `DRAFT`
- Task version: `0.1`
- Scientific code baseline: `b4b2769c9b92007c5d6a65a1789149cd55ebd633`
- Task-definition baseline: recorded when this complete DRAFT moves to `REVIEW`
- Execution-start commit: recorded only after Claude pre-review returns `APPROVED`
- Design: `docs/superpowers/specs/2026-09-03-visit-indexed-martingale-certificate-design.md`
- Plan: `docs/superpowers/plans/2026-09-03-visit-indexed-martingale-certificate-plan.md`
- GPT branch and worktree: `codex/FP-MART-001` in the repository root
- Claude branch and worktree: `claude/FP-MART-001` in
  `C:\tmp\research-FP-MART-001-claude`
- GPT result directory:
  `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex\`
- Claude result directory:
  `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\claude\`
- Classification: long task requiring independent construction and
  cross-verification
- Estimated runtime and resources: two to four hours per route, CPU-only, two
  480-item formal runs, local Claude Code under the existing account, and no
  new fee category

## Research question

On one frozen fixed-policy trajectory, can the fixed state Bellman, pair
Bellman, and fixed-`V^pi` recovery residuals be controlled simultaneously by a
computable visit-indexed martingale boundary whose radius scales with each
observed count, and can that event replace the current occupancy-denominator
certificate in the existing Direct-Q and V-first no-split bounds without
claiming that full support itself occurs with high probability?

## Falsifiable hypotheses

1. Under the frozen sampling order, the `m` state residual sequences and two
   families of `d` pair residual sequences can be indexed by visits as
   martingale differences under explicitly stated stopped filtrations.
2. A finite-horizon union over `G = m + 2d` groups and every
   `1 <= k <= n` yields a simultaneous two-sided Hoeffding radius of order
   `B * sqrt(log(G*n/delta) / k)` with no stationary occupancy, transition
   matrix, or spectral factor.
3. Because the event is simultaneous in `k`, substituting each random observed
   final count is valid without conditioning on that count or assuming it is
   independent of the residuals.
4. On observed required support, the new radii compose with the unchanged
   deterministic Direct-Q all-layer and V-first no-split recurrences, including
   data-dependent early stopping and finite-softmax empirical-margin gates.
5. A route certificate can be computed using only observed counts, empirical
   matching diagonals, a declared reward bound, public hyperparameters, and
   algorithm metadata; true occupancy, kernels, values, residuals, and initial
   errors are unnecessary.
6. Under the frozen 480-item protocol, exact-route emission follows observed
   full pair support, with expected rates 2.5%, 85.0%, 100%, and 100% across the
   four trajectory lengths, while all common legacy fields have zero mismatch.
7. A variance-adaptive Freedman or empirical-Bernstein construction may be
   tighter, but it is valid only if its variance proxy is observable and its
   risk is preallocated. A negative result on this optional hypothesis does not
   reject hypotheses 1--6.

The mandatory scientific route fails if any residual lacks the stated
martingale property, the random-count substitution is invalid, the resulting
radius retains an occupancy denominator or hidden population count, or the
certificate requires an oracle quantity.

## Inputs and fixed protocol

### Sampling and theorem scope

- Finite state-action space and one fixed policy.
- Sampling order:
  `S_t -> A_t -> (R_{t+1}, S_{t+1}) -> A_{t+1}`.
- One frozen trajectory retained under the completed stationary-start protocol.
- Deterministic edge reward with `|R| <= R_star` and
  `B = R_star / (1 - gamma)`.
- Fixed-context synchronous Direct-Q and state-value iteration.
- Existing exact matching and finite-softmax matching definitions.
- Zero initialization, with computable initial-error bound `B`.
- Mandatory Hoeffding certificate receives the full declared `delta` and is the
  only primary route certificate.
- An optional variance-adaptive certificate is separately labelled and is not
  selected post hoc. Any combined minimum must preallocate `delta/2` to each
  valid constituent before the formal run.

The task excludes nonstationary starts, additional stochastic reward noise,
changing policies, layer-dependent scores, and online control even if part of
the martingale proof could later extend to those settings.

### Frozen formal matrix

- Tasks: 30.
- States: 6.
- Actions: 4.
- Trajectory lengths: 256, 1024, 4096, 16384.
- Mixing settings: 0.08, 0.5.
- Reward gap bonuses: 0, 0.5.
- Minimum action probability: 0.05.
- Softmax beta: 8.
- Gamma: 0.70.
- Alpha: 0.65.
- Evaluation iterations: 160.
- Certificate delta: 0.05.
- Seed: 20260829.
- Total matched records per route: 480.

Every formal command must spell out every value. No parameter, threshold, risk
allocation, or stopping rule may change after either route inspects formal
output.

### Frozen read-only regression input

The canonical baseline directory is
`C:\Users\Admin\Desktop\research\icrl_softmax\results\fixed_policy_finite_sample_certificates\`.
The required files and SHA-256 hashes are:

- `config.json`:
  `a2276eae06ba8689864cac2ad3d3d0e92b069014b2046d0180ee172bb8296ea0`;
- `task_results.json`:
  `c84329bd6b9fcd495789f2067f250ead28fec807193d1b69ae1038e9cc3b2f25`;
- `summary.json`:
  `49846c0825c859df28ff773164352d19f6cb86943c484ab411bac7f69434b5ae`.

The task-results file contains exactly 480 records. A hash or count mismatch is
a stopping condition, not permission to replace the baseline.

### Probability semantics

For the trajectory-based emission event `Emit`, the only required probability
claim is

```text
P(Emit and certified error bound is violated) <= delta.
```

The task prohibits claims that `P(Emit) >= 1 - delta`, that observed support is
a prior coverage guarantee, or that conditional coverage follows by dividing
by the unknown emission probability. Emission validity and numerical
nontriviality are separate metrics.

## Allowed work

### Shared read-only inputs

- Root governance files and the frozen task, design, and plan.
- Existing Python implementation and verifier files.
- Existing theory and comparison documents.
- The three hashed baseline result files and other old result directories.
- Local papers and read-only retrieval of primary sources for imported
  martingale results, with exact attribution recorded.

### GPT write scope on `codex/FP-MART-001`

- `visit_indexed_martingale_certificate.py`;
- `verify_visit_indexed_martingale_certificate.py`;
- `evaluate_visit_indexed_certificates.py`;
- `analyze_visit_indexed_certificates.py`;
- an additive, behavior-preserving change to
  `evaluate_fixed_policy_q_routes.py` or
  `fixed_policy_finite_sample_certificate.py` only if a minimal reusable public
  helper is required;
- `docs/research_branches/FP-MART-001/codex/`;
- `docs/research_branches/visit_indexed_martingale_certificate_theory.md`;
- `docs/research_branches/visit_indexed_martingale_certificate_report.md`;
- this task, its design and plan, and a compact `ACTIVE_WORKSPACE.md` pointer;
- `results/FP-MART-001/codex/`.

### Claude write scope on `claude/FP-MART-001`

- the same four new Python implementation/verification entry points;
- the same two existing Python files only for a minimal additive public helper;
- `docs/research_branches/FP-MART-001/claude/`;
- `results/FP-MART-001/claude/`.

Claude may record its theory, commands, results, limitations, and later
verification report in its assigned evidence directory. It may not edit the
task definition, design, plan, `ACTIVE_WORKSPACE.md`, GPT evidence, or GPT
branch.

## Prohibited work

- No implementation or experiment before task activation.
- No change to the fixed research question, hypotheses, protocol, risk budget,
  acceptance criteria, failure criteria, or route allocation without a GPT
  revision following user ruling.
- No write to `main`, the other actor's branch/worktree/result directory, any
  old result directory, `DOC-001`, `GOV-001`, the manuscript, or the archive.
- No reading of the other route's first result, report, result directory, or
  conclusion before both sealed first-result commits exist.
- No oracle quantity in certificate inputs and no hidden fallback from observed
  counts to true occupancy or spectral constants.
- No unadjusted post-hoc minimum across multiple probabilistic bounds.
- No tuning to reproduce an expected pass rate.
- No claim that empirical audit coverage proves the theorem.
- No expansion to stochastic rewards, nonstationary starts, blockwise policy
  improvement, or fully online control.
- No force push, history rewrite, merge to `main`, publication, external write,
  unsafe permission bypass, or new fee category.

## Expected artifacts

### Versioned GPT artifacts

- The four new Python files listed in the GPT scope.
- `docs/research_branches/FP-MART-001/codex/theory.md`.
- `docs/research_branches/FP-MART-001/codex/first_result.md`.
- `docs/research_branches/FP-MART-001/codex/verification_of_claude.md`.
- `docs/research_branches/visit_indexed_martingale_certificate_theory.md`.
- `docs/research_branches/visit_indexed_martingale_certificate_report.md`.
- Updated task evidence and `ACTIVE_WORKSPACE.md`.

### Versioned Claude artifacts

- Claude's independent versions of the four new Python files.
- `docs/research_branches/FP-MART-001/claude/theory.md`.
- `docs/research_branches/FP-MART-001/claude/first_result.md`.
- `docs/research_branches/FP-MART-001/claude/verification_of_codex.md`.

### Per-route ignored results

Each result directory contains:

- `config.json`;
- `task_results.json`;
- `summary.json`;
- `regression.json`;
- `environment.json`;
- `commands.log`;
- `checks.log`;
- any derived plots used by the route report.

Smoke outputs live below a clearly labelled `smoke/` subdirectory and cannot be
substituted for formal evidence.

## Acceptance criteria

1. Both routes explicitly derive the sampling filtrations, visit stopping
   times, measurability, martingale differences, conditional ranges, and
   optional-skipping step for all three residual families.
2. The mandatory theorem gives one event over all `G = m + 2d` residual groups
   and all `1 <= k <= n`, with total failure probability at most `delta`.
3. Random observed counts are substituted only through the simultaneous event,
   without conditional-independence or fixed-count assumptions.
4. The final theorem and outputs use the selective emission semantics exactly
   and make no prior full-support claim.
5. Certificate functions accept no oracle model, occupancy, value, residual, or
   true initial-error input; zero initialization uses `B`.
6. Direct-Q all-layer and V-first no-split exact/softmax bounds compose with the
   visit-indexed radii without changing the completed deterministic recurrence.
7. Required support, relevant empirical margins, algorithm mode, risk budget,
   and finite arithmetic are validated with deterministic ordered failures.
8. New fields use the `visit_indexed_certificate` namespace and the
   `selective_high_probability_certified` status without changing legacy status
   meanings.
9. All machine outputs are strict JSON, use `null` for unavailable values, and
   structurally separate certificate inputs from `oracle_audit` fields.
10. All existing fixed-policy verifiers and the new verifier pass on both
    branches.
11. Both smoke matrices pass strict-JSON, nonfinite, oracle-separation, and
    legacy-regression checks before either formal matrix begins.
12. Each formal result has exactly 480 records and exactly matches the frozen
    configuration and seed.
13. All legacy nonnumeric leaves match exactly; numeric leaves yield zero
    mismatches under `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`.
14. Exact-route emission is 100% at trajectory lengths 4096 and 16384. All four
    rates, nontriviality rates, and deviations from 2.5%, 85.0%, 100%, and 100%
    are reported without tuning.
15. Every empirical audit violation is reported with its configuration, and no
    empirical coverage statistic is used as proof.
16. Each route records exact commits, environment, commands, raw outputs,
    anomalies, metrics, limitations, and a numbered acceptance assessment.
17. GPT reproduces and verifies Claude's route, and Claude reproduces and
    verifies GPT's route, each ending in `PASS`, unless the user records an
    explicit exception.
18. The final synthesis explains every material route difference, updates
    `ACTIVE_WORKSPACE.md`, and leaves `main` unchanged pending user approval.

The optional variance-adaptive hypothesis is accepted only with its own proof,
tests, and risk semantics. Its failure does not fail the mandatory task when it
is recorded accurately and criteria 1--18 otherwise pass.

If a mandatory scientific hypothesis is falsified, the task may still reach a
verified negative conclusion instead of satisfying the positive-result items
above. That negative track requires all of the following:

1. both routes independently identify the same obstruction, or a material
   difference is resolved through cross-verification or user ruling;
2. the obstruction is given as a complete proof or deterministic counterexample
   under the frozen sampling order and inputs;
3. no implementation or output claims the rejected positive certificate;
4. all unaffected legacy verifiers and preservation checks pass;
5. both final verification reports return `PASS` for the stated negative
   conclusion and map evidence to the rejected hypotheses.

## Failure criteria

Scientific hypotheses 1--6 are rejected if:

- any required centered residual is not a martingale difference under the
  frozen sampling order;
- visit selection lacks the required measurability;
- random-count substitution depends on unproved conditioning or independence;
- the proved radius contains the old occupancy denominator or an equivalent
  hidden population-count condition;
- the claimed computable certificate needs a true kernel, occupancy, value,
  residual, or initial error;
- selective validity cannot be combined with either mandatory route.

Implementation or evidence fails if:

- an old verifier fails or legacy output mismatches;
- strict JSON, failure ordering, or oracle separation fails;
- a formal configuration differs from the frozen matrix;
- either route cannot reproduce its claimed outputs;
- a conclusion overstates the proof or empirical evidence.

A rigorous, reproducible negative scientific conclusion may complete the
research question, but it cannot be labelled as a positive certificate result.

## Stopping conditions

Stop affected work and notify the user if:

- Claude pre-review or final verification returns `OBJECTION`;
- a required input hash/count differs;
- another edit overlaps an allowed path or route isolation cannot be preserved;
- a formal run would require changing a frozen parameter, risk allocation, or
  stopping rule;
- a required change falls outside the allowed paths;
- source assumptions cannot be verified from primary evidence;
- the task expands to an excluded scientific scope;
- Claude is unavailable because of authentication, quota, permission, or
  environment failure;
- a new cost category, unsafe permission bypass, external write, publication,
  or merge would be required.

## Route assignments and independence

### GPT route

GPT independently constructs the proof, tests, implementation, smoke run,
formal run, and evidence in the GPT paths. Codex sub-agents may perform bounded
GPT-side support but cannot inspect sealed Claude results or replace Claude.

### Claude route

Claude independently constructs the same required result in its own branch,
worktree, evidence paths, and result directory. It may repair its own ordinary
implementation errors but may not redefine the task. It returns an objection
instead of changing a disputed requirement.

### Disclosure and cross-verification

Neither route reads the other's first result or conclusion before both sealed
first-result commits exist. After disclosure, each verifier inspects and
reproduces the other route without editing it and records `PASS`, `FAIL`, or
`OBJECTION` with checkable evidence.

## Execution evidence

### GPT route

- Commit and environment: not started.
- Commands: not started.
- Outputs: not started.
- Metrics and anomalies: not started.
- Conclusion and limitations: preliminary until cross-verification.
- Acceptance assessment: not started.

### Claude route

- Commit and environment: not started.
- Commands: not started.
- Outputs: not started.
- Metrics and anomalies: not started.
- Conclusion and limitations: preliminary until cross-verification.
- Acceptance assessment: not started.

## Objections and user rulings

### Objection

- Status: `NONE`.
- Disputed clause: none.
- Evidence: none.
- Validity impact: none.
- Options for user ruling: not applicable.

### User ruling

- Date: 2026-09-02 through 2026-09-03.
- Decision: the user selected the recommended visit-indexed martingale route,
  approved the scope/probability contract, module/isolation design, frozen
  evaluation and acceptance design, and then approved the written specification
  by instructing GPT to continue.
- Required GPT task revision: none.

## Quota or continuity handoff

- Current branch, commit, and task version: `codex/FP-MART-001`; DRAFT commit to
  be recorded at the REVIEW transition; task `0.1`.
- Completed, running, and pending work: design approved and committed; plan and
  DRAFT task being recorded; pre-review and all execution remain pending.
- Commands and outputs: design/self-review evidence is in Git history; no
  research implementation or experiment has run.
- Current findings and uncertainty: the visit-indexed Hoeffding route is the
  mandatory hypothesis; exact filtration constants and the optional
  variance-adaptive result remain to be constructed independently.
- Immutable boundaries: fixed scientific scope, hashes, 480-item protocol,
  risk semantics, route isolation, no oracle input, no tuning, and no `main`
  merge.
- Claude's exact next actions: perform only the read-only pre-review after the
  task enters `REVIEW`.
- GPT's required return verification: inspect and record the pre-review before
  activating either route.

## Verification reports

### GPT verifies Claude

- Status: not started.
- Reproduction or inspection performed: not started.
- Evidence: not started.
- Acceptance-criteria mapping: not started.
- Required next state: not started.

### Claude verifies GPT

- Status: not started.
- Reproduction or inspection performed: not started.
- Evidence: not started.
- Acceptance-criteria mapping: not started.
- Required next state: not started.

## Definition of done

- [ ] No unresolved objection remains.
- [ ] Both independent routes are reproducible.
- [ ] Both verification reports are recorded.
- [ ] Every acceptance criterion has evidence.
- [ ] Discrepancies are reconciled or ruled on by the user.
- [ ] `ACTIVE_WORKSPACE.md` is current.
- [ ] The user approved any merge into `main`.
