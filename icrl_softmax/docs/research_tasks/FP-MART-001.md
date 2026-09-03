# FP-MART-001: Visit-indexed martingale certificates for fixed-policy Q evaluation

## Task metadata

- Created: 2026-09-03
- Author: GPT
- Status: `VERIFIED`
- Task version: `1.0`
- Scientific code baseline: `b4b2769c9b92007c5d6a65a1789149cd55ebd633`
- Task-definition baseline: `2c5f59b25e20fb6db86cb9b308288dc47dc230ca`
- Execution-start commit:
  `c8ec7e5c3165930663e26a07f98b38cc9ec186ad`
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

- Implementation commit:
  `4cf6f50d69c5aa4937d181f758f546f9d2213c41`.
- Blind first-result seal commit:
  `e7c04111b7aec8c9dc5043883fcaa68cc158837b`.
- Environment, commands, output hashes, anomalies, smoke metrics, limitations,
  and numbered acceptance assessment:
  `docs/research_branches/FP-MART-001/codex/first_result.md`.
- Smoke outcome: 16 records; all five preflight verifiers, Ruff, strict JSON,
  complete matched-record regression, namespace audit, schema/oracle
  separation, and empirical audit passed. Legacy mismatch count, namespace
  problem count, schema problem count, and empirical violation count were all
  zero.
- Preliminary conclusion: positive mandatory construction; smoke emission was
  support-selective but every emitted bound remained nontriviality-negative
  under the preregistered `total_bound < B` metric.
- Formal 480-record result and exact hashes:
  `docs/research_branches/FP-MART-001/codex/formal_result.md`.
- Formal-evidence seal commit:
  `63b84fd4295598c3e020d7e489552470ac1763c4`.
- Independence: no Claude theory, code, output, result, or conclusion was read
  before the seal.

### Claude route

- Independent execution was launched from the common activation commit in the
  isolated Claude worktree after the user's explicit data-transfer approval.
- Blind first-result seal commit:
  `c5da2430d00e11414723d85cf42293dddae07183`.
- The Claude branch was clean at its seal. Its content remained undisclosed
  until after GPT's first-result and formal-evidence seals; both seal identities
  are now recorded, so reciprocal verification is authorized.
- Initial GPT verification outcome: `FAIL` at commit
  `7555fdbd4d7bb794be880978f207a22519e7329b`; the task definition remains
  valid, so Claude author repair was authorized.
- Repaired implementation/smoke seal:
  `e0343871fd313c80f4f49ffa773db10eb75667d6`.
- Repaired formal-evidence seal:
  `e3c37197905e3ced080197dba989cdab36b4d4e8`.
- GPT independently reproduced the repaired 480-record route exactly and found
  no remaining proof or implementation defect, but its second verification
  returned `FAIL` for incomplete smoke and acceptance evidence. The task is
  `ACTIVE` for a narrow Claude evidence-only repair; no new formal matrix is
  required.

## Claude read-only pre-review

- Date: 2026-09-03.
- Reviewed commit: `b386b62d4c6abf5aa7bb33e219e1af9fc55c93c1`.
- Outcome: `APPROVED`.
- Scope: task definition, design, implementation plan, shared fixed-policy
  theory, existing certificate/evaluator/verifier feasibility, repository
  state, and frozen baseline integrity. No implementation, experiment, branch
  mutation, network access, or sealed-result inspection was authorized.
- Repository checks: the worktree was clean on `codex/FP-MART-001`;
  `git diff --check 2c5f59b..HEAD` passed; the only files changed between the
  DRAFT baseline and REVIEW commit were this task record and
  `ACTIVE_WORKSPACE.md`.
- Frozen-input checks: the SHA-256 hashes of `config.json`,
  `task_results.json`, and `summary.json` matched the three values frozen in
  this task.
- Itemized result: all 18 checks passed, covering falsifiability; the sampling
  order and two filtration families; visit measurability and optional skipping;
  `G = m + 2d` accounting; simultaneous-in-count substitution; selective
  probability semantics; computable/oracle separation; mandatory and optional
  risk allocation; positive and verified-negative paths; hashes, record count,
  and tolerances; exact deliverables; lifecycle; common activation and isolated
  worktrees/results; first-result nondisclosure; role boundaries; acceptance,
  failure, and stopping criteria; resources; and executability without changing
  the frozen contract.
- Non-blocking cautions: derive the Hoeffding constant from the exact increment
  range instead of copying the candidate radius; pin the exact centered state
  Bellman residual before implementation; cite primary sources with their exact
  assumptions; pre-register categories for explaining emission-rate deviations;
  and use absolute canonical Windows paths for the Claude baseline and outputs.
- Validity assessment: no blocking defect or task-level objection was found;
  the five cautions are execution and verification checkpoints and require no
  task-definition revision.
- Session evidence: local Claude Code session
  `cce80670-85d5-4fbe-8489-55100c444d51`, completed successfully within the
  authorized read-only pre-review budget.

## Claude execution-launch authorization

- Date: 2026-09-03.
- Status: `AUTHORIZED`; the user explicitly approved the task-scoped data
  transfer and public-web source lookup on 2026-09-03.
- Completed safe setup: the isolated worktree and branch
  `claude/FP-MART-001` were created at the common activation commit
  `c8ec7e5c3165930663e26a07f98b38cc9ec186ad`; both route worktrees were clean
  at that commit.
- Prior blocker: the execution launch would send the designated private-repository
  research documents and code to the external model service used by local
  Claude Code and would permit source lookup on the public web. The standing
  authorization to launch local Claude does not, by itself, constitute explicit
  authorization for that data transfer.
- Safety outcome: the launch request was rejected before Claude read the
  execution materials or began proof, implementation, or experiments. No
  workaround was attempted.
- User ruling: `APPROVED`. The user explicitly replied “允许” after being told
  that the execution would send FP-MART-001-scoped private-repository research
  documents, code, and the three frozen baselines to the external Claude
  service and would permit task-scoped public-web lookup of primary sources.
- Remaining blocker: none. The authorization does not expand the frozen task,
  permit reading the GPT route, or permit publishing, merging, messaging,
  destructive actions, or new fee categories.

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

## Quota or continuity record

- Current branch and task version: `codex/FP-MART-001`; task `1.0`, status
  `VERIFIED`.
- Completed work: both routes independently sealed first results and formal
  evidence. GPT's initial Claude verification exposed ordinary defects; Claude
  repaired its own route and evidence. GPT then reproduced and passed the final
  Claude route, Claude independently reproduced and passed the GPT route, and
  GPT completed the shared theory, final report, and workspace update.
- Commands and outputs: exact GPT commands, environment, ignored-output hashes,
  anomalies, and metrics are in
  `docs/research_branches/FP-MART-001/codex/first_result.md`.
- Current findings and uncertainty: the mandatory observed-count Hoeffding
  route is verified. Primary-useful bounds occur only for V-first at length
  16384 in this grid. No observable variance-adaptive constituent was
  constructed; that remains an open construction problem rather than an
  impossibility result.
- Immutable boundaries: fixed scientific scope, hashes, 480-item protocol,
  risk semantics, route isolation, no oracle input, no tuning, and no `main`
  merge.
- Claude's exact next actions: none for this verified task.
- GPT's exact next actions: preserve the verified state; do not merge to
  `main` without explicit user approval. Any variance-adaptive continuation
  requires a new frozen task.

## Verification reports

### GPT verifies Claude

- Status: `FAIL` for sealed Claude commit
  `c5da2430d00e11414723d85cf42293dddae07183`; ordinary repair, not
  `OBJECTION`.
- Reproduction or inspection performed: all five verifiers and Ruff passed;
  the complete 480-record Claude output reproduced byte-identically for the
  deterministic core files; proof, code, schema, smoke, history, and ignored
  outputs were independently inspected.
- Evidence and full acceptance mapping:
  `docs/research_branches/FP-MART-001/codex/verification_of_claude.md`, commit
  `7555fdbd4d7bb794be880978f207a22519e7329b`.
- Blocking findings: an invalid written MGF iteration; true `mdp["R"]` maximum
  used as certificate `B`; out-of-horizon and inconsistent counts can emit;
  only 4/16 smoke records align while the gate passes; config/summary and
  per-group oracle audits are incomplete; and no pre-formal implementation
  seal or raw-output hashes exist.
- Required next state: `ACTIVE` for Claude author repair, then return to
  `VERIFYING` for GPT re-verification.

#### GPT verifies repaired Claude route

- Status: `FAIL` for repaired seals `e0343871fd313c80f4f49ffa773db10eb75667d6`
  and `e3c37197905e3ced080197dba989cdab36b4d4e8`; ordinary evidence repair, not
  `OBJECTION`.
- Reproduction performed: all five required verifiers and Ruff pass; a fresh
  480-record run matches all deterministic core hashes and passes every strict
  analyzer gate.
- Evidence: `docs/research_branches/FP-MART-001/codex/verification_of_claude_repair.md`.
- Remaining defects: the Claude repair-smoke record substituted
  `verify_theory.py` for `verify_end_to_end_sarsa.py`, lacked `checks.log`,
  omitted two existing smoke hashes, and did not provide a current numbered
  1--18 task assessment.
- Repair request:
  `docs/research_branches/FP-MART-001/codex/claude_evidence_repair_request.md`.
- Required next state: remain `ACTIVE` for the narrow evidence repair, then
  return to `VERIFYING` without rerunning the formal matrix.

#### GPT verifies Claude evidence repair

- Status: `PASS` for Claude evidence-repair commit
  `a559bd31769506565dd4503d8239d4a3f28ddf81`.
- Scope check: exactly two Claude evidence documents changed; no Python or
  formal raw-output file changed; branch clean; `git diff --check` passed.
- Evidence closure: all seven repaired-smoke artifacts and hashes are now
  recorded; Ruff and the exact five required verifiers pass; the post-formal
  timing anomaly is explicitly preserved; a current numbered 1--18 acceptance
  assessment is present.
- Final GPT evidence:
  `docs/research_branches/FP-MART-001/codex/verification_of_claude_evidence_repair.md`.
- Reproduction basis: GPT's fresh 480-record output remains byte-identical to
  Claude's deterministic formal core and every strict analyzer gate passes.
- Required next state: `VERIFYING`; Claude must now independently reproduce and
  verify the sealed GPT route.

### Claude verifies GPT

- Status: `PASS` at Claude commit
  `5999b889a3336866b3eec7099f74080b9b253ce5`.
- Frozen GPT seals: implementation
  `4cf6f50d69c5aa4937d181f758f546f9d2213c41`, first result
  `e7c04111b7aec8c9dc5043883fcaa68cc158837b`, formal evidence
  `63b84fd4295598c3e020d7e489552470ac1763c4`.
- Request:
  `docs/research_branches/FP-MART-001/codex/claude_verification_request.md`.
- Reproduction or inspection performed: independent proof/code/history audit,
  Ruff, all five verifiers, and a fresh frozen 480-record matrix plus strict
  analyzer in the Claude-owned verification directory.
- Evidence:
  `docs/research_branches/FP-MART-001/claude/verification_of_codex.md`.
- Acceptance-criteria mapping: criteria 1--17 `PASS`; criterion 18 was pending
  this final synthesis and workspace update and is now satisfied.
- Reproduction outcome: config, task records, and summary are byte-identical to
  GPT; regression differs only in the isolated output-directory field; all
  scientific metrics and gates agree.
- Required next state: completed by final synthesis; task is `VERIFIED`.

## Final synthesis

- Final synthesis and workspace-update commit:
  `0f5f822dc62b31cda608ea21c96b2e53fe6e3f45`.
- Shared verified theory:
  `docs/research_branches/visit_indexed_martingale_certificate_theory.md`.
- Shared verified report:
  `docs/research_branches/visit_indexed_martingale_certificate_report.md`.
- Mandatory outcome: positive. The observed-count Hoeffding event is valid and
  composes with all four fixed-policy routes under selective emission.
- Optional outcome: no valid observable variance-adaptive constituent was
  constructed; this is not an impossibility theorem.
- Route agreement: identical frozen protocol, radius constant, probability
  semantics, emission rates, usefulness rates, and zero-violation audits.
- Material route difference: equivalent stopped-process versus predictable
  transform proof presentations and different software decomposition/schema;
  all differences are reconciled in the shared report.
- Governance outcome: no unresolved objection; both reciprocal verification
  reports end `PASS`; `ACTIVE_WORKSPACE.md` is current; `main` is unchanged
  pending explicit user approval.

## Definition of done

- [x] No unresolved objection remains.
- [x] Both independent routes are reproducible.
- [x] Both verification reports are recorded.
- [x] Every acceptance criterion has evidence.
- [x] Discrepancies are reconciled in the final synthesis.
- [x] `ACTIVE_WORKSPACE.md` is current.
- [x] `main` remains unchanged; any future merge still requires explicit user
  approval.
