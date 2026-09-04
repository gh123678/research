# FP-TU-001: Time-uniform mixture certificates for fixed-policy Q evaluation

## Task metadata

- Created: 2026-09-04.
- Author: GPT.
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline:
  `28b71685ca05ae073cc847fdea230019e4bd63ea`.
- Approved design commit:
  `0c0c63dd8defbb606dc6f51190d914e33c9da526`.
- Task-definition baseline:
  `0a66c4c583678e3186af7c5dbcff02779436329c`.
- Execution-start commit: the `ACTIVE` activation commit containing the Claude
  pre-review record below; its exact SHA is recorded before either route seals
  a first-result commit.
- Design:
  `docs/superpowers/specs/2026-09-04-time-uniform-mixture-certificate-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-04-time-uniform-mixture-certificate-plan.md`.
- GPT branch and worktree: `codex/FP-TU-001` in the repository root.
- Claude branch and worktree: `claude/FP-TU-001` in
  `C:\tmp\research-FP-TU-001-claude` after activation.
- GPT result directory:
  `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-TU-001\codex\`.
- Claude result directory:
  `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-TU-001\claude\`.
- Classification: long, conclusion-critical task requiring independent
  construction and reciprocal verification.
- Estimated resources: three to five hours per route, CPU-only, two frozen
  480-record formal runs, local Claude Code under the existing account, and no
  new fee category.

## Research question

Under the frozen `FP-MART-001` fixed-policy protocol, can a preregistered
finite geometric mixture of exponential supermartingales replace the
count-wise Hoeffding union with a computable time-uniform residual boundary
that preserves the selective guarantee and no-oracle input contract, is no
wider for every integer count through 16384, and tightens the existing
Direct-Q and V-first no-split certificates without changing emission or legacy
outputs?

## Falsifiable hypotheses

1. For each of the `G = m + 2d` residual groups, the normalized stopped sum
   `Z_k = S_k/B` supports
   `exp(a Z_k - a^2 k/2)` as a nonnegative supermartingale for every fixed real
   `a` under the filtrations already verified by `FP-MART-001`.
2. The frozen 15-component, two-sided cosh mixture in the approved design has
   initial value one and, after a union over groups, yields one event uniform
   over all visit counts with total failure probability at most `delta`.
3. Its unique nonnegative crossing root can be computed conservatively from
   observed counts using stable log-domain arithmetic and the analytic
   line-stitching upper bracket, with no oracle input or outcome-dependent
   tuning.
4. For the frozen dimensions and risk budget, the resulting radius satisfies
   `r_mix(k) <= r_old(k; n)` for every frozen trajectory length `n` and every
   integer `1 <= k <= n`, with at least one strict inequality, and is
   nonincreasing through 16384. Here
   `r_old(k; n) = B sqrt(2 log(2 G n/delta)/k)` is exactly the legacy radius
   used by each record. The design's global `n_max = 16384` comparison remains
   a separate formula audit, not a substitute for this stricter per-record
   preservation check.
5. Replacing only the residual-radius primitive preserves every deterministic
   emission decision and makes every emitted Direct-Q and V-first exact or
   finite-softmax total bound nonincreasing.
6. The frozen 480-record run exactly preserves task identity, configuration,
   legacy fields, and exact-route emission rates of 2.5%, 85%, 100%, and 100%
   across the four trajectory lengths.
7. An observable transition-variance confidence construction may exist, but
   it enters this task only as a proof-or-counterexample feasibility study. It
   cannot enter the certificate without a complete observable contract and a
   separately allocated risk budget.

Hypotheses 1--6 form the mandatory positive route. If any is false, a complete
proof or deterministic counterexample may support a verified negative result;
weights, rates, thresholds, or protocol may not be retuned in response.

## Frozen theory and probability contract

For one residual group, the conditional range width is `2B`, where
`B = R_star/(1-gamma)` and `R_star` is declared before MDP sampling. Set
`J = 15` and, for `j = 0,...,14`,

```text
k_j = 2^j
w_j = (j+1)^(-2) / sum_{l=0}^{14} (l+1)^(-2)
L_j = log(2 G / (delta w_j))
a_j = sqrt(2 L_j / k_j)
```

The mandatory two-sided mixture is

```text
M(k,z) = sum_j w_j exp(-a_j^2 k/2) cosh(a_j z).
```

For integer `k >= 1`, `q_mix(k)` is the unique nonnegative root of
`M(k,q) = G/delta`, and

```text
r_mix(k) = B q_mix(k) / k.
```

The analytic audit boundary is

```text
q_j(k) = (L_j + a_j^2 k/2) / a_j
q_stitch(k) = min_j q_j(k)
r_stitch(k) = B q_stitch(k) / k.
```

It must be proved independently valid and used only as a deterministic upper
bracket and audit. It is never selected as the reported route certificate.

For every trajectory-based emission event `Emit`, the only probability claim
is

```text
P(Emit and an emitted bound is violated) <= delta.
```

The task does not claim high-probability support or conditional coverage given
emission. Random final counts are substituted only through the event already
uniform over all visit times.

## Numerical inversion contract

The implementation computes `log M(k,q)` with stable `logsumexp` and
`logcosh`. For each count it uses `q_lo = 0`,
`q_hi = q_stitch(k)`, and target `log(G/delta)`. It must establish

```text
log M(k,q_lo) < target <= log M(k,q_hi).
```

With `tol = 1e-12`, deterministic bisection runs for at most 200 iterations
and stops only when

```text
q_hi - q_lo <= tol * (1 + q_hi)
0 <= log M(k,q_hi) - target <= tol * (1 + abs(target)).
```

The returned value is the conservative upper endpoint. Bracketing,
convergence, finiteness, or conservativeness failure gives an ordered
deterministic non-emission reason; it never falls back silently to the old
radius.

## Inputs and fixed protocol

### Sampling and algorithm scope

- Finite state/action spaces, one fixed policy, one stationary-start
  trajectory, and sampling order
  `S_t -> A_t -> (R_{t+1}, S_{t+1}) -> A_{t+1}`.
- Deterministic bounded edge rewards with `|R| <= R_star`.
- Fixed-context synchronous Direct-Q and state-value iteration.
- Exact and finite-softmax matching, zero initialization, and the verified
  deterministic Direct-Q and V-first no-split recurrences.
- The same pre-action and post-action filtrations, visit stopping times, and
  fixed-target residual families as `FP-MART-001`.
- The mandatory mixture receives the full declared `delta` over all `G`
  groups and is the only reported new probabilistic certificate.

The task excludes policy changes, online control, policy improvement,
nonstationary starts, layer-dependent scores, general stochastic reward noise,
outcome-trained mixture constants, and unallocated post-hoc minima.

### Frozen formal matrix

- Tasks per cell: 30.
- States/actions: 6/4.
- Trajectory lengths: 256, 1024, 4096, 16384.
- Mixing settings: 0.08, 0.5.
- Reward-gap bonuses: 0, 0.5.
- Minimum action probability: 0.05.
- Softmax beta: 8.
- Gamma/alpha/iterations: 0.70/0.65/160.
- Certificate delta: 0.05.
- Seed: 20260829.
- Total matched records per route: 480.

Every formal command spells out every value. No design constant, threshold,
risk allocation, solver rule, or stopping rule may change after either route
inspects smoke or formal output.

### Frozen read-only regression input

The canonical baseline is
`C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex\`.
Its required files and SHA-256 hashes are:

- `config.json`:
  `bcc377b422711302163b976d5d5ece389e1e3ee8540d37817a8719bf9ec3bf8a`;
- `task_results.json`:
  `929e2f65689af850b65f000ee6675a8a87b3506c2e2a9600c07d28cf183c6d52`;
- `summary.json`:
  `fa13619755b627b9dcff281dc5e4bc9b4d0262b9a6592f2982012b53df2b34ed`.

The task-results file contains exactly 480 records. Any hash or count mismatch
is a stopping condition, not permission to replace the baseline.

## Allowed work

### Shared read-only inputs

- `AGENTS.md`, `ACTIVE_WORKSPACE.md`, this frozen task, its design and plan.
- Existing fixed-policy and visit-indexed implementations and verifiers.
- Completed `FP-MART-001` theory, reports, evidence, and frozen baseline files.
- Read-only retrieval of primary sources for imported probability results,
  with assumptions and mappings recorded.

### GPT write scope on `codex/FP-TU-001`

- `time_uniform_mixture_certificate.py`;
- `verify_time_uniform_mixture_certificate.py`;
- `evaluate_time_uniform_certificates.py`;
- `analyze_time_uniform_certificates.py`;
- a minimal additive, behavior-preserving change to an existing
  visit-indexed file only if required for a reusable public helper;
- `docs/research_branches/FP-TU-001/codex/`;
- `docs/research_branches/time_uniform_mixture_certificate_theory.md`;
- `docs/research_branches/time_uniform_mixture_certificate_report.md`;
- this task, its design and plan, and a compact `ACTIVE_WORKSPACE.md` pointer;
- `results/FP-TU-001/codex/`.

### Claude write scope on `claude/FP-TU-001`

- independent versions of the same four new Python entry points;
- the same narrowly allowed additive helper change;
- `docs/research_branches/FP-TU-001/claude/`;
- `results/FP-TU-001/claude/`.

Claude may record its own theory, commands, results, limitations, and later
verification report in its assigned evidence directory. It may not edit the
task, design, plan, workspace index, GPT evidence, or GPT branch.

## Prohibited work

- No implementation or experiment before the task becomes `ACTIVE`.
- No change to the frozen hypotheses, formulas, protocol, risk budget,
  acceptance criteria, failure criteria, or route assignments without a GPT
  revision following user ruling.
- No write to `main`, the other actor's branch/worktree/result directory, old
  result directories, the manuscript, or the archive.
- No reading of the other route's first result, code conclusion, or result
  directory before both first-result seals exist.
- No oracle kernel, occupancy, value, residual, realized route error, true
  reward maximum, or true initial error in certificate inputs.
- No learned mixture weights/rates, formal-output tuning, silent fallback,
  unadjusted certificate minimum, or use of empirical coverage as proof.
- No expansion to excluded scientific scope, unsafe permission bypass, force
  push, history rewrite, merge, publication, external message, or new fee
  category.

## Expected artifacts

### Versioned GPT artifacts

- The four new Python files in GPT scope.
- `docs/research_branches/FP-TU-001/codex/theory.md`.
- `docs/research_branches/FP-TU-001/codex/first_result.md`.
- `docs/research_branches/FP-TU-001/codex/formal_result.md`.
- `docs/research_branches/FP-TU-001/codex/verification_of_claude.md`.
- Shared theory and final report named in the GPT scope.
- Updated task evidence and `ACTIVE_WORKSPACE.md`.

### Versioned Claude artifacts

- Claude's independent versions of the four new Python files.
- `docs/research_branches/FP-TU-001/claude/theory.md`.
- `docs/research_branches/FP-TU-001/claude/first_result.md`.
- `docs/research_branches/FP-TU-001/claude/formal_result.md`.
- `docs/research_branches/FP-TU-001/claude/verification_of_codex.md`.

### Per-route ignored results

Each result directory contains exactly seven core artifacts:

- `config.json`;
- `task_results.json`;
- `summary.json`;
- `regression.json`;
- `environment.json`;
- `commands.log`;
- `checks.log`.

Clearly labelled smoke subdirectories are separate from formal evidence and
cannot replace it.

## Acceptance criteria

1. Both routes map the already verified filtrations, visit stopping times,
   measurability, martingale differences, and conditional range width to every
   mandatory mixture component without weakening assumptions.
2. The cosh mixture proof gives one event over all `G = m + 2d` groups and all
   visit counts with total failure probability at most `delta`.
3. Random observed counts and selective emission semantics are used exactly,
   with no conditional-support claim.
4. The analytic line-stitching boundary is separately proved, never selected,
   and gives a valid upper bracket for every count through 16384.
5. The pure certificate accepts no prohibited oracle input and derives `B`
   only from the declared pre-sampling reward bound.
6. Every count `1 <= k <= 16384` passes stable inversion, conservative-root,
   finiteness, `q_mix <= q_stitch`, and mixture-radius monotonicity checks.
7. For every `n` in `{256, 1024, 4096, 16384}` and every `1 <= k <= n`,
   `r_mix(k) <= r_old(k; n)` holds with at least one strict inequality overall;
   the separate global-`n_max` design audit also passes through 16384.
8. Direct-Q and V-first no-split exact/softmax composition uses the unchanged
   deterministic recurrences; every old emitted total bound is nonincreasing.
9. Required support, relevant empirical margins, fixed-context mode, risk,
   count consistency, and numeric behavior have ordered deterministic failure
   handling.
10. New fields live only below `time_uniform_certificate`; legacy namespaces,
    values, and status meanings remain unchanged.
11. Strict JSON rejects duplicate keys and nonfinite values, uses `null` for
    unavailable values, and structurally separates `oracle_audit`.
12. All existing fixed-policy and visit-indexed verifiers, the new verifier,
    Ruff, schema checks, and baseline-integrity checks pass on both routes.
13. Both smoke matrices pass before either formal matrix begins, and each route
    seals its implementation and first result before disclosure.
14. Each formal result has exactly 480 records and exactly matches the frozen
    configuration, seed, and task identity.
15. All legacy nonnumeric leaves match exactly; numeric leaves have zero
    mismatch under `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`.
16. Exact-route emission remains 2.5%, 85%, 100%, and 100%; all softmax,
    usefulness, radius-reduction, and deviation metrics are reported without
    tuning.
17. Every empirical audit violation is enumerated and is not used as theorem
    evidence.
18. The transition-variance study records either a complete observable
    proposal or an exact obstruction/gap and never enters the mandatory
    certificate.
19. Each route records exact commits, environment, commands, hashes,
    successful and failed runs, anomalies, limitations, and a numbered
    acceptance assessment.
20. GPT reproduces and verifies Claude, Claude reproduces and verifies GPT,
    both final reports end `PASS`, the final synthesis explains differences,
    and `ACTIVE_WORKSPACE.md` is current while `main` remains unchanged.

Failure of radius dominance is a scientific negative result, not permission
to retune the mixture. A valid negative completion requires a certified
failing count or proof, unchanged legacy behavior, all unaffected checks, full
evidence, and reciprocal `PASS` verification of the negative conclusion.

An increase in `<B` usefulness is a secondary empirical outcome. Its absence
does not reject a mathematically valid uniformly tighter radius.

## Failure criteria

The mandatory positive result fails if the mixture process is not a
supermartingale under a required residual filtration, group/sign risk exceeds
`delta`, random-count substitution is invalid, the analytic bracket is false,
the solver can understate the root, a prohibited input is required, or radius
dominance fails at any frozen count.

Implementation or evidence fails if an old verifier fails, legacy outputs or
emission decisions mismatch, strict JSON or oracle separation fails, a frozen
configuration differs, either route cannot reproduce its evidence, or a
conclusion overstates proof or empirical results.

A reproducible negative conclusion may complete the task only when both routes
independently establish the same obstruction or reconcile their difference,
no output claims the rejected certificate, all unaffected preservation checks
pass, and both reciprocal reports return `PASS` for that negative conclusion.

## Stopping conditions

Stop affected work and notify the user if:

- Claude pre-review or final verification returns `OBJECTION`;
- a frozen baseline hash or record count differs;
- the common baseline differs or route isolation cannot be preserved;
- another edit overlaps an allowed path;
- implementation would require a prohibited input, fallback, or formula
  change;
- numerical inversion cannot be certified under the frozen contract;
- a smoke or formal failure would require changing any frozen parameter;
- a required source assumption cannot be verified from primary evidence;
- Claude is unavailable through authentication, quota, permission, or
  environment failure;
- scope, cost, publication, external communication, merge, or permission would
  expand beyond existing authorization.

## Route assignments and independence

### GPT route

GPT independently constructs proof, tests, implementation, smoke, formal
evidence, and its acceptance assessment in GPT-owned paths. Codex subagents
cannot inspect sealed Claude results or replace the required Claude route.

### Claude route

Claude independently constructs the same mandatory result from the common
activation commit in its own worktree, branch, evidence directory, and result
directory. It may repair ordinary errors in its route but may not redefine the
task.

### Disclosure and reciprocal verification

Neither route reads the other's first result, implementation conclusion, or
formal output before both first-result commits are sealed. After disclosure,
each verifier inspects and independently reproduces the other route without
editing it and records exactly `PASS`, `FAIL`, or `OBJECTION` with checkable
evidence.

## Claude read-only pre-review

- Date: 2026-09-05.
- Status: `COMPLETED`.
- Reviewed commit: `b63557fff5f98f71228ad9ad66a345bbc7b45883`.
- Outcome: `APPROVED`.
- Tool boundary: local Claude Code 2.1.138, `plan` permission mode, and only
  `Read`, `Glob`, and `Grep`; no write/edit tool, Git mutation, experiment, or
  future route result was available.
- Session evidence: the first two non-TTY invocations returned no auditable
  text and were rejected as review evidence. A terminal-mode retry completed
  with exit code 0 in Codex terminal session 31798 and returned the approval
  below.
- Itemized result:
  1. lifecycle, task version, DRAFT/REVIEW commits, design identity, workspace
     pointer, and user ruling are consistent;
  2. the exponential-supermartingale constant matches the conditional width
     already verified by `FP-MART-001`;
  3. the cosh mixture starts at one and allocates exactly `delta/G` per group,
     while the factor two in `L_j` belongs to the signed line audit;
  4. the displayed `cosh` inequality proves
     `M(k,q_stitch(k)) >= G/delta`, and `M(k,0) < G/delta`;
  5. continuity and strict increase give a unique root, while the maintained
     upper endpoint makes the bisection result conservative;
  6. all-count dominance is coherent, falsifiable, and design-time feasible;
  7. radius monotonicity is an exhaustive, falsifiable numerical contract;
  8. the no-oracle interface is compatible with the verified visit-indexed
     certificate structure;
  9. existing route recurrences are monotone in the radius and their emission
     gates do not depend on the radius value;
  10. selective probability semantics and random-count substitution are
      preserved exactly;
  11. the three frozen baseline files exist, contain the expected protocol,
      and are appropriate for new legacy regression;
  12. branch, worktree, result, evidence, and nondisclosure boundaries satisfy
      the governance rules;
  13. all 20 acceptance items plus failure, stopping, positive, and negative
      tracks are testable and complete;
  14. the frozen formal command, existing helpers, six-verifier preflight, and
      seven core result artifacts are executable; and
  15. transition variance is safely isolated as feasibility-only work.
- Non-blocking cautions recorded for execution:
  1. GPT must independently verify baseline hashes, clean status, and REVIEW
     changed paths before activation;
  2. distinguish the design's global `n_max = 16384` audit from the legacy
     per-record trajectory-length radius;
  3. the new namespace audit must allow only
     `time_uniform_certificate`, treating `visit_indexed_certificate` as
     legacy;
  4. preflight must contain all five existing verifiers plus the new one;
  5. both routes must recompute, not cite, the design-time ratio values; and
  6. bisection must maintain its bracket invariant and satisfy both stopping
     tests within 200 iterations.
- GPT closure of cautions 1--2 before activation:
  - `git diff --check` passed from the DRAFT baseline to the REVIEW commit; the
    only changed files were `ACTIVE_WORKSPACE.md` and this task;
  - all three SHA-256 hashes exactly matched the frozen values and the task
    record count remained 480;
  - exhaustive formula checks gave maximum per-record radius ratios of
    `0.9573574`, `0.9263356`, `0.8975201`, and `0.8793887` for trajectory
    lengths 256, 1024, 4096, and 16384 respectively, always below one; and
  - the task now states the stricter per-record interpretation explicitly.
- Validity assessment: no blocking defect or task-level objection. The six
  cautions are implementation/verification checkpoints and require no change
  to the approved formulas, risk budget, protocol, or route allocation.

## Claude data-transfer authorization

- Date: 2026-09-05.
- Status: `AUTHORIZED`.
- User ruling: after being told that the designated `FP-TU-001` private task,
  design, plan, prior verified theory, and relevant code would be sent to the
  external Claude service for read-only pre-review, the user replied
  “允许”.
- Scope: this authorization covers only the task-scoped read-only pre-review.
  The later independent Claude construction would send a broader execution
  bundle and requires a separate explicit data-transfer authorization. This
  ruling does not authorize scope changes, external publication or messaging,
  unsafe permission bypass, merge, or a new fee category.

## Objections and user rulings

### Objection

- Status: `NONE`.
- Disputed clause: none.
- Evidence: none.
- Validity impact: none.
- Options for user ruling: not applicable.

### User ruling

- Date: 2026-09-04.
- Decision: the user approved the finite geometric mixture as mandatory, line
  stitching as audit-only, transition variance as feasibility-only, the
  frozen 480-record evaluation, the acceptance contract, and the written
  design at commit `0c0c63dd8defbb606dc6f51190d914e33c9da526`.
- Required GPT revision: none.

## Execution evidence

### GPT route

- Status: `NOT_STARTED_BEFORE_ACTIVATION`.

### Claude route

- Status: `PRE_REVIEW_APPROVED_EXECUTION_NOT_STARTED`; no implementation or
  execution-bundle transfer has started yet.

## Verification reports

### GPT verifies Claude

- Status: `PENDING`.

### Claude verifies GPT

- Status: `PENDING`.

## Definition of done

- [ ] No unresolved objection remains.
- [ ] Both independent routes are reproducible.
- [ ] Both reciprocal verification reports are recorded and pass.
- [ ] Every acceptance criterion has evidence.
- [ ] Differences are reconciled or ruled on by the user.
- [ ] `ACTIVE_WORKSPACE.md` is current.
- [ ] `main` remains unchanged unless the user separately approves a merge.
