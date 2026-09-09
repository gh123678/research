# FP-ADV-001: Fixed-policy action-gap certificates and one safe update

## Task metadata

- Created: 2026-09-08.
- Author: GPT.
- Status: `ACTIVE`.
- Task version: `1.2` (governance-only validation update; frozen scientific
  contract unchanged).
- Git and scientific baseline:
  `c579047950dfabb2600020cd2e53dd24b3e39c84`.
- Approved design commit:
  `0ff51967aa131e769c3bf63efc1c3bbe6c9030ad`.
- DRAFT task-definition baseline:
  `919c26f8f2dc90b093cc1f40c9b36d2be2b45b03`.
- Execution-start commit:
  `10a9a94e24ec92a59e7c756f9af6ce07b2f30e59`.
- Common route execution-start commit:
  `4078f6911cbfb4654205772685f49896e4e8cad2`.
- Design:
  `docs/superpowers/specs/2026-09-08-action-gap-safe-update-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-08-action-gap-safe-update-plan.md`.
- GPT branch and worktree: `codex/FP-ADV-001` in the repository root.
- Claude branch and worktree: `claude/FP-ADV-001` in an isolated worktree
  created from the common route execution-start commit.
- GPT result directory:
  `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-ADV-001\codex\`.
- Claude result directory:
  `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-ADV-001\claude\`.
- Classification: long, multi-stage, conclusion-critical task requiring
  independent construction and reciprocal verification.
- Estimated resources: four to six hours per route, CPU-only, two frozen
  480-record formal runs, local Claude Code under the existing account, and no
  new fee category.

## Research question

Under the verified `FP-TU-001` fixed-policy event, can observable pair-specific
recovery radii and successor-row differences certify selected action orderings
without requiring complete Q-table support, and can those orderings produce
one pointwise non-degrading policy update that preserves the exploration
floor?

## Falsifiable hypotheses

1. On the existing simultaneous event, the exact V-first recovery difference
   for two actions in one state is bounded by

   ```text
   r_sa + r_sb + 2 gamma E_V TV(P_bar_sa, P_bar_sb).
   ```

2. For finite-softmax recovery, normalized observed query weights define an
   effective successor row and a within-group mass `kappa_g`; the fixed-target
   error is bounded by `kappa_g r_g + 2B(1-kappa_g)`, yielding the pairwise
   uncertainty frozen below.
3. The receiver and donors can be selected after seeing estimates without new
   risk spending because all used statements are deterministic consequences
   of one event simultaneous over all pre-registered groups.
4. The local exact and softmax V-first penalties are no wider than their
   corresponding complete-Q global penalties whenever those controls emit.
5. Shifting the frozen fraction of transferable mass only along strictly
   positive action-gap lower bounds makes
   `T_{pi_plus}V^pi >= V^pi` componentwise and therefore
   `V^{pi_plus} >= V^pi` componentwise.
6. Unrelated missing state-action pairs do not block a valid local comparison;
   full state support and positive counts for the two compared actions are the
   mandatory support requirements.
7. The frozen 480-record reconstruction exactly preserves all FP-TU legacy
   identities, estimator metrics, certificate leaves, and task configuration.
8. At least one primary local V-first route emits a nonempty safe update in the
   frozen 480 records, and its update decisions weakly dominate the matching
   global V-first control record by record.

Hypotheses 1--7 are mandatory mathematical, software, and preservation claims.
Hypothesis 8 is the main empirical usefulness claim. If hypothesis 8 is false,
a complete zero-usefulness result is a valid verified negative outcome; no
formula, threshold, transfer fraction, route, or matrix may be retuned.

## Frozen theory and probability contract

### Reused event and semantics

The task uses the verified `FP-TU-001` event over
`G = m + 2d` groups: `m` state Bellman groups, `d` pair Bellman groups, and
`d` recovery groups. It allocates no new risk and does not select a minimum of
separately calibrated certificates.

For each emitted update, the only probability claim is

```text
P(EmitUpdate and (
    any used action ordering is false
    or exists s: V^{pi_plus}(s) < V^pi(s)
)) <= delta.
```

The task makes no claim that support or update emission occurs with
probability at least `1-delta`, and no claim conditional on emission.

### Exact V-first action gap

Let `B = R_star/(1-gamma)`. When the existing exact state route emits
`||V_hat-V^pi||_infinity <= E_V`, define, for visited pair `g=(s,a)`,

```text
q_hat_g = mean_{t:g_t=g} [R_{t+1} + gamma V_hat(S_{t+1})]
P_bar_g = empirical successor-state row for visits to g
r_g     = the FP-TU recovery radius at the observed count of g
```

For two actions in the same state,

```text
U_exact(s,a,b)
  = r_sa + r_sb
    + 2 gamma E_V TV(P_bar_sa, P_bar_sb)

LCB_exact(s,a,b)
  = q_hat(s,a) - q_hat(s,b) - U_exact(s,a,b).
```

Here `TV(p,q)=0.5||p-q||_1`. The proof must use
`|(p-q)^T e| <= TV(p,q)span(e)` and
`span(e) <= 2||e||_infinity`.

### Finite-softmax V-first action gap

For pair query `g`, derive the normalized verified one-hot softmax weights from
the observed group count, total trajectory length, and declared `beta`. Let
`kappa_g` be their total on-group mass and `P_bar_g^beta` their normalized
effective successor distribution. Define

```text
C_g = kappa_g r_g + 2B(1-kappa_g)

U_soft(s,a,b)
  = C_sa + C_sb
    + 2 gamma E_V^beta
        TV(P_bar_sa^beta, P_bar_sb^beta)

LCB_soft(s,a,b)
  = q_hat_beta(s,a) - q_hat_beta(s,b) - U_soft(s,a,b).
```

The `2B` contamination term follows only from every fixed-policy recovery
target lying in `[-B,B]`. The pure certificate cannot receive a true kernel,
true target, or oracle diagonal.

### Global controls

For any emitted complete-Q route bound `E_Q`, define

```text
LCB_global(s,a,b)
  = q_hat(s,a) - q_hat(s,b) - 2E_Q.
```

The controls are V-first no-split exact/softmax and Direct-Q exact/softmax.
They must use the same estimates and candidate-selection rule as the local
routes. The task does not introduce a local Direct-Q recurrence.

### Candidate selection and update

For every route and state, select

```text
a_star(s) = argmax_a q_hat(s,a),
```

with smallest-index tie breaking. A donor `b != a_star` is eligible only when
both pair counts are positive, the applicable `LCB(s,a_star,b) > 0`, and
`pi(b|s) > pi_min`.

Freeze `theta=1/2` and transfer

```text
eta(s,b) = theta [pi(b|s)-pi_min]
```

from each eligible donor to `a_star`. If no donor is eligible, return the
original policy and record an abstention. The update must preserve exact row
sums, nonnegativity, and `pi_plus(a|s) >= pi_min`.

On the reused event,

```text
(T_{pi_plus}V^pi)(s)-V^pi(s)
  = sum_b eta(s,b)[Q^pi(s,a_star)-Q^pi(s,b)]
  >= sum_b eta(s,b)LCB(s,a_star,b)
  >= 0.
```

Bellman monotonicity and contraction then yield
`V^{pi_plus} >= V^pi` componentwise.

## Inputs and fixed protocol

### Sampling and algorithm scope

- Finite state/action spaces and one fixed policy.
- One stationary-start on-policy trajectory with sampling order
  `S_t -> A_t -> (R_{t+1},S_{t+1}) -> A_{t+1}`.
- Deterministic edge rewards with predeclared `|R| <= R_star`.
- Fixed-context synchronous Direct-Q and state-value iteration from zero.
- Exact and finite-softmax matching and the unchanged no-split V-first
  recovery.
- The verified `FP-TU-001` filtrations, residual groups, mixture grid,
  numerical inversion, risk allocation, and selective semantics.
- Transfer fraction `theta=0.5` fixed before smoke or formal output.

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
- Transfer fraction: 0.5.
- Seed: 20260829.
- Total matched records per route: 480.

Every formal command spells out all values. No formula, threshold, reason
order, route, transfer rule, metric definition, or stopping rule may change
after either route inspects smoke or formal output.

### Frozen read-only regression input

The canonical baseline is
`C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-TU-001\codex\`.
It contains exactly 480 records. Required SHA-256 hashes are:

- `config.json`:
  `43dcb96b0f6f95e76f1c0b484d6375e3727dbb5609b16a8952a69e2ac0dddf3a`;
- `task_results.json`:
  `0e5eab39bf49894832c5ebcdd6f7b70c453fff6b9600b234617889f8f9fa79be`;
- `summary.json`:
  `565fc4d261a938d13350984bb97242e79fba42f014d11f807a18942517f4444f`.

Any hash or count mismatch is a stopping condition, not permission to replace
the baseline.

## Ordered status and abstention contract

Pair and update reasons use this fixed order:

1. `algorithm_mode_mismatch`;
2. `divergence_guard_triggered`;
3. `state_certificate_not_emitted`;
4. `candidate_pair_unvisited`;
5. `recovery_radius_unavailable`;
6. `attention_mass_invalid`;
7. `effective_transition_row_invalid`;
8. `numerical_nonfinite`;
9. `gap_lcb_nonpositive`;
10. `no_transferable_mass`.

The last two are ordinary abstentions. A local failure cannot invalidate a
valid comparison in another state. Numeric or input failure cannot silently
fall back to an oracle value, a different formula, or a global control.

## Allowed work

### Shared read-only inputs

- `AGENTS.md`, `ACTIVE_WORKSPACE.md`, this task, its design and plan.
- All verified `FP-MART-001` and `FP-TU-001` theory, reports, code, verifiers,
  and frozen baseline files.
- Existing fixed-policy and blockwise evaluators for formulas and regression.
- Task-scoped read-only retrieval of primary sources with exact assumption
  mapping.

### GPT write scope on `codex/FP-ADV-001`

- `action_gap_certificate.py`;
- `verify_action_gap_certificate.py`;
- `evaluate_action_gap_certificates.py`;
- `analyze_action_gap_certificates.py`;
- one minimal additive, behavior-preserving edit to
  `evaluate_fixed_policy_q_routes.py` only if needed to expose route estimates;
- `docs/research_branches/FP-ADV-001/codex/`;
- `docs/research_branches/action_gap_certificate_theory.md`;
- `docs/research_branches/action_gap_certificate_report.md`;
- this task, its design, plan, and a compact `ACTIVE_WORKSPACE.md` pointer;
- `results/FP-ADV-001/codex/`.

### Claude write scope on `claude/FP-ADV-001`

- independent versions of the same four new Python entry points;
- the same narrowly allowed evaluator helper change;
- `docs/research_branches/FP-ADV-001/claude/`;
- `results/FP-ADV-001/claude/`.

Claude may record its own theory, commands, results, limitations, and later
verification report in its assigned evidence directory. It may not edit the
task, design, plan, workspace index, GPT evidence, or GPT branch.

## Prohibited work

- No research implementation or experiment before status `ACTIVE`.
- No change to the frozen formulas, probability semantics, transfer rule,
  matrix, routes, hypotheses, or acceptance criteria without a GPT revision
  following user ruling.
- No write to `main`, the other route's branch/worktree/result directory, old
  result directories, the manuscript, archive, or the user-owned untracked
  learning record.
- No reading of the other route's first result, code conclusion, or formal
  output before both routes seal first and formal evidence.
- No true kernel, occupancy, value, Q table, gap, realized error, exact return,
  or oracle diagonal in certificate inputs.
- No post-hoc threshold, donor, route, transfer-fraction, or metric selection.
- No empirical audit as theorem evidence, silent fallback, force operation,
  history rewrite, merge, push, publication, external message, unsafe
  permission bypass, or new fee category.

## Expected artifacts

### Versioned GPT artifacts

- The four new Python files in GPT scope and any authorized helper edit.
- `docs/research_branches/FP-ADV-001/codex/theory.md`.
- `docs/research_branches/FP-ADV-001/codex/first_result.md`.
- `docs/research_branches/FP-ADV-001/codex/formal_result.md`.
- `docs/research_branches/FP-ADV-001/codex/reciprocal_verification.md`.
- Shared theory and report named in GPT scope.
- Updated task evidence and `ACTIVE_WORKSPACE.md`.

### Versioned Claude artifacts

- Claude's independent versions of the four new Python files and any
  authorized helper edit.
- `docs/research_branches/FP-ADV-001/claude/theory.md`.
- `docs/research_branches/FP-ADV-001/claude/first_result.md`.
- `docs/research_branches/FP-ADV-001/claude/formal_result.md`.
- `docs/research_branches/FP-ADV-001/claude/verification_of_codex.md`.

### Per-route ignored results

Each result directory contains exactly seven core artifacts:

- `config.json`;
- `task_results.json`;
- `summary.json`;
- `regression.json`;
- `environment.json`;
- `commands.log`;
- `checks.log`.

Labelled smoke directories are separate and cannot replace formal evidence.

## Acceptance criteria

1. Both routes correctly inherit the `FP-TU-001` simultaneous event, random
   count substitution, and selective semantics without risk resplitting.
2. The exact recovery decomposition and total-variation/span inequality imply
   the frozen exact local uncertainty.
3. The finite-softmax weights, within-group mass, effective successor row, and
   contamination decomposition imply the frozen softmax local uncertainty.
4. The pure module accepts no prohibited oracle input and derives `B` only
   from the predeclared reward bound.
5. Post-data receiver and donor selection remains covered by the simultaneous
   event; no conditional-coverage claim appears.
6. Every emitted donor has positive count, transferable mass, and a strictly
   positive LCB under the applicable route.
7. The policy update preserves finite values, exact row sums, nonnegativity,
   and the declared exploration floor.
8. The Bellman improvement lower bound is nonnegative in every state, and the
   pointwise policy-improvement implication is proved and tested.
9. Exact and softmax local penalties never exceed matching global V-first
   penalties when those controls exist.
10. Unrelated missing pairs do not block valid local comparisons; required
    state or selected-pair support produces ordered abstention.
11. Direct-Q controls use only the unchanged global `2E_Q` construction and
    do not claim a new local theorem.
12. New fields live only below `action_gap_certificate`; all FP-TU and older
    namespaces, values, and status meanings remain unchanged.
13. Strict JSON rejects duplicate keys and nonfinite values, uses `null` for
    unavailable values, and structurally separates `oracle_audit`.
14. All inherited verifiers, the new verifier, task-scoped Ruff, schema,
    provenance, and frozen-hash checks pass on both routes.
15. Both smoke matrices pass before formal evaluation, and each route seals
    implementation and first result before disclosure.
16. Each formal result contains exactly 480 records with the frozen seed,
    matrix, task identity, formulas, and transfer fraction.
17. Legacy nonnumeric leaves match exactly and numeric leaves have zero
    mismatch under `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`.
18. Every empirical false ordering, Bellman-bound violation, value decrease,
    and return decrease is enumerated and excluded from theorem evidence.
19. Each route records exact commits, environment, commands, hashes, failed
    and successful runs, anomalies, limitations, and a numbered acceptance
    assessment.
20. GPT reproduces and verifies Claude, Claude reproduces and verifies GPT,
    both reports end `PASS`, differences are reconciled, and
    `ACTIVE_WORKSPACE.md` is current while `main` remains unchanged.

## Failure criteria

The mandatory theory fails if the exact or softmax decomposition is false,
the existing event does not cover post-data selection, local/global dominance
is false under the frozen definitions, the Bellman implication needs an
unstated assumption, or any prohibited input or extra risk allocation is
required.

Implementation or evidence fails if validation can emit an invalid policy or
nonpositive gap, partial support is handled globally, legacy outputs mismatch,
strict JSON or oracle separation fails, a frozen configuration differs, or an
independent route cannot reproduce its evidence.

Hypothesis 8 may fail without invalidating the task. A verified negative
completion must preserve the frozen contract, report zero useful updates
without retuning, pass all unaffected checks, and receive reciprocal `PASS`
verification.

## Stopping conditions

Stop affected work and notify the user if:

- Claude pre-review or final verification returns `OBJECTION`;
- a frozen baseline hash or record count differs;
- the common activation baseline differs or route isolation cannot be
  preserved;
- another edit overlaps an allowed path;
- implementation would require an oracle input, new risk allocation, fallback,
  or formula change;
- the finite-softmax decomposition or pointwise policy theorem cannot be
  proved under the frozen assumptions;
- smoke or formal failure would require changing a frozen parameter;
- a primary-source assumption cannot be verified;
- Claude is unavailable because of authentication, quota, permission, or
  environment failure;
- independent-execution data transfer, cost, scope, merge, publication,
  external communication, or permissions would expand beyond authorization.

## Route assignments and independence

### GPT route

GPT independently constructs proof, tests, implementation, smoke, formal
evidence, and its acceptance assessment in GPT-owned paths. Codex subagents
cannot replace the required Claude route.

- Blind first-result seal:
  `996641d9ce2088e95c0bf2a0661e3b24b6e0d6fe`.
- Pre-seal evidence: proof and test-first implementation completed; all
  inherited/new verifiers, task-scoped Ruff, an eight-record smoke matrix,
  strict reconstruction, legacy preservation, policy validation, oracle
  separation, and local/global dominance checks passed.
- Formal evaluation: not run before the blind first-result seal.
- Frozen formal evaluator: run exactly once with 480 records; exited `PASS`.
- Post-formal ordinary fix:
  `440710c8c9c77ee8128d4abfd96919ffafe7991c` corrects only no-donor policy
  identity drift (maximum `6.94e-17`); serialized observable inputs were
  mechanically repaired and revalidated without rerunning the formal matrix.
- Formal-evidence seal:
  `5f190ebb78697acd3cd877c1d64898929bb88024`.
- Post-seal validation: after `913bd0e`, the analyzer was strengthened to
  replay the frozen seed schedule and independently reconstruct every action
  input, oracle audit, and action-summary row. Normal analysis is now
  read-only; evaluator output refuses non-empty directories; and identity
  repair is prehashed and atomic. The strengthened verifier, Ruff, fresh smoke,
  smoke analysis, and formal 480-record replay all pass with zero mismatches.
  These validation-only changes and the reciprocal-verification record were
  committed at `46cbf9bca8dab87161ffee7a1c38ead162d73ad6`; no formal record was
  rerun or changed.
- Preliminary empirical outcome: all six routes emitted zero updates in 480
  records; hypothesis 8 is falsified without retuning, while all Codex-route
  mandatory checks pass. The outcome remains preliminary pending reciprocal
  verification.

### Claude route

Claude independently constructs the same result from the common activation
commit in its own worktree, branch, evidence directory, and result directory.
Its blind first-result seal is `191e13ba5472ce4c183643008169236979c000ff` and
its formal-evidence seal is `191821b26b16e13de323fb31651343ffe1eb9656`.
It independently obtained the same zero-update result and repaired one summary
pipeline omission without rerunning the formal matrix. It may repair ordinary
defects in its route but cannot redefine the task.

### Disclosure and reciprocal verification

Neither route reads the other's first result, code conclusion, or formal
output before both first-result and formal-evidence seals exist. After
disclosure, each verifier inspects and independently reproduces the other route
without editing it and records exactly `PASS`, `FAIL`, or `OBJECTION`.

## Claude read-only pre-review

- Date: 2026-09-08.
- Status: `COMPLETED`.
- Reviewed commit: `a3a0340b3f63ed53eb00b5d5af244fabdbbbe75d`.
- Outcome: `APPROVED`.
- Tool boundary: Claude Code 2.1.138 in `plan` permission mode with only
  `Read`, `Glob`, and `Grep`; no file write, experiment, branch mutation,
  result generation, or future-route disclosure was available.
- Launch record: on 2026-09-08 GPT requested the task-scoped external Claude
  call with only `Read`, `Glob`, and `Grep`. The permission review connection
  ended before approval completed, so the process was rejected before Claude
  ran. That first attempt transferred no task material, created no Claude
  session, and produced no review result.
- Authorization closure and successful retry: after GPT identified the private
  task-scoped material, external Claude service, and read-only tool boundary,
  the user replied “好” on 2026-09-08. The retry used the same scope and
  completed with an auditable approval.
- Itemized result:
  1. lifecycle, baselines, approved design, user ruling, and GPT/Claude role
     boundaries are consistent with `AGENTS.md`;
  2. the exact V-first recovery decomposition matches the existing estimator
     and fixed-target recovery event;
  3. `|(p-q)^T e| <= TV(p,q)span(e)` gives the frozen factor
     `2 gamma E_V TV` with no missing constant;
  4. the finite-softmax within-group mass, off-group `2B` contamination, and
     effective successor row yield the frozen softmax uncertainty without an
     oracle input;
  5. all radii reuse the pre-paid `FP-TU-001` simultaneous event and introduce
     no hidden route or pair risk split;
  6. post-data receiver and donor selection is a deterministic consequence of
     that event and preserves the selective probability statement;
  7. exact and softmax local penalties are deterministically no wider than
     their matching global V-first controls;
  8. `theta=1/2` transfers preserve the simplex and exploration floor, and the
     Bellman identity implies pointwise policy non-degradation;
  9. support is correctly localized, abstentions are ordered, the pure
     interface is no-oracle, and strict-JSON/oracle separation has existing
     compatible infrastructure;
  10. the frozen 30-by-4-by-2-by-2 matrix, seed, baseline identities,
      deliverables, negative route, and runtime checks are executable;
  11. branch/result isolation, blind seals, nondisclosure, and separate
      independent-execution authorization are explicit; and
  12. all 20 acceptance criteria are checkable, while failure of the empirical
      usefulness hypothesis remains a valid verified negative result.
- Non-blocking execution cautions:
  1. make the receiver behavior with unvisited actions explicit in code and
     tests; donor eligibility already requires both compared counts positive,
     so safety is unaffected; and
  2. expect many short-prefix `state_certificate_not_emitted` abstentions
     because the state route requires full support; this is frozen behavior,
     not a defect.
- Validity assessment: no blocking task-definition or acceptance-basis defect
  exists. The task may become `ACTIVE` without changing its formulas, risk
  budget, routes, transfer rule, or evaluation protocol.

## Claude data-transfer authorization

### Read-only pre-review

- Status: `AUTHORIZED`.
- Prior user instruction: on 2026-09-08 the user replied “继续” after GPT
  stated that the next steps were the formal task, implementation plan, and
  Claude read-only pre-review. After GPT explicitly identified the private
  files, external service, and exclusions, the user replied “好” on
  2026-09-08.
- Scope: task-scoped governance, task, approved design, plan, inherited theory,
  relevant code, and frozen baseline identities may be sent to the external
  Claude service for read-only pre-review.
- Exclusions: implementation, experiments, route-result access, publication,
  external messaging, merge, push, unsafe permission bypass, or a new fee
  category.

### Independent execution

- Status: `AUTHORIZED`.
- Date: 2026-09-08.
- User ruling: after GPT identified the external Claude service, private
  task-scoped material, write and execution scope, frozen formal run, and
  exclusions, the user replied “好的”.
- Scope: task-scoped private code, task, design, plan, frozen baseline, and
  inherited verified theory may be sent to the external Claude service for an
  independent proof, tests, implementation, smoke evaluation, and exactly one
  frozen 480-record formal evaluation. Claude may write only its isolated
  `claude/FP-ADV-001` branch/worktree, assigned evidence directory, and assigned
  ignored result directory, and may perform task-scoped read-only primary-source
  retrieval.
- Exclusions: no merge to `main`, push, publication, external message, access to
  GPT's blind first result, unsafe permission bypass, new fee category, or
  scientific-contract change.

## Objections and user rulings

### Objection

- Status: `NONE`.
- Disputed clause: none.
- Evidence: none.
- Validity impact: none.
- Options for user ruling: not applicable.

### User ruling

- Date: 2026-09-08.
- Decision: the user selected `FP-ADV-001`, approved the V-first local action
  gap as primary, Direct-Q as control, one fixed-policy safe update, and the
  written design at commit
  `0ff51967aa131e769c3bf63efc1c3bbe6c9030ad`, then instructed GPT to continue.
- Required GPT task revision: none.

## Quota or continuity handoff

- Current branch and task version: `codex/FP-ADV-001`, version `1.2` in
  `ACTIVE`.
- Completed work: approved design and REVIEW, Claude pre-review `APPROVED`,
  GPT blind seal and one formal 480-record run, GPT representation-only repair,
  Claude blind/formal seals, and reciprocal disclosure.
- GPT evidence: formal zero-update result plus 480-record provenance/oracle
  replay, fresh smoke, hardened verifier, and Ruff all pass. The hardening and
  reciprocal-verification record were committed at
  `46cbf9bca8dab87161ffee7a1c38ead162d73ad6`.
- Claude evidence: formal zero-update result, zero legacy mismatches, isolated
  verifier/Ruff pass, and GPT's independent 17,280-state replay pass.
- Current blocker: Claude's reciprocal command-level verification of the GPT
  route is `FAIL` solely because its session denied Bash/sandbox execution of
  verifier, analyzer, and Ruff. Its read-only content review found no
  discrepancy, but governance does not allow that to count as `PASS`.
- Pending work: rerun those three read-only checks in a Claude session with the
  commands permitted, then record the final reciprocal decision and update the
  workspace status. No formal evaluator rerun is allowed.
- Immutable boundaries: probability semantics, formulas, transfer fraction,
  480-record protocol, route isolation, no oracle input, no tuning, and no
  `main` merge.

## Verification reports

### GPT verifies Claude

- Status: `PASS`.
- Reproduction or inspection performed: Claude verifier and Ruff passed in its
  isolated worktree; GPT replayed all 480 seed records and 17,280 route-state
  entries against Claude's formal artifacts with zero failures and max q-row
  difference 0.
- Evidence: `docs/research_branches/FP-ADV-001/codex/reciprocal_verification.md`,
  Claude seals `191e13b` and `191821b`, Claude `checks.log` and `regression.json`.
- Acceptance-criteria mapping: formal 480-record protocol, baseline
  preservation, route emissions, support/gates, oracle separation, and zero-
  update empirical conclusion all match; no discrepancy found.
- Required next state: retain `PASS` while awaiting Claude's executable
  reverse check.

### Claude verifies GPT

- Status: `FAIL` (environment blocker, not a scientific failure).
- Reproduction or inspection performed: Claude read the Codex implementation,
  formal evidence, hashes, and checks and found no content discrepancy, but
  two explicitly scoped read-only sessions denied Bash/sandbox execution of
  the verifier, analyzer, and Ruff.
- Evidence: `docs/research_branches/FP-ADV-001/codex/reciprocal_verification.md`.
- Acceptance-criteria mapping: content-level criteria pass by inspection;
  command-level independent reproduction remains unproven.
- Required next state: do not mark the task `VERIFIED`; rerun only the three
  read-only checks after the Claude environment permits them.

## Definition of done

- [ ] No unresolved objection remains.
- [ ] Both required routes are reproducible.
- [ ] Both reciprocal verification reports are recorded.
- [ ] Every acceptance criterion has evidence.
- [ ] Discrepancies are reconciled or ruled on by the user.
- [ ] `ACTIVE_WORKSPACE.md` is current.
- [ ] The user approved any merge into `main`.
