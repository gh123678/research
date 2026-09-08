# Fixed-policy action-gap certificate implementation plan

> Date: 2026-09-08
> Intended task: `docs/research_tasks/FP-ADV-001.md`
> Design: `docs/superpowers/specs/2026-09-08-action-gap-safe-update-design.md`
> Git and scientific baseline: `c579047950dfabb2600020cd2e53dd24b3e39c84`
> Constraint: no research implementation or experiment begins before Claude's
> read-only pre-review returns `APPROVED` and the task becomes `ACTIVE`.

## Task 1: Freeze and pre-review the research contract

Create `FP-ADV-001.md` in `DRAFT` with the exact formulas, frozen baseline
hashes, protocol, route ownership, positive and verified-negative outcomes,
acceptance criteria, failure criteria, and stopping rules. Commit the DRAFT
task and this plan together.

In a second commit, record the DRAFT commit, move the task to `REVIEW`, and
update `ACTIVE_WORKSPACE.md` with a compact pointer. Send only the task-scoped
governance, task, approved design, plan, inherited theory, relevant code, and
baseline identities to local Claude Code for a read-only pre-review. Require
exactly one outcome:

- `APPROVED`, with an itemized probability, algebra, software, protocol, and
  governance audit; or
- `OBJECTION`, with the disputed clause, evidence, validity impact, and user
  ruling options.

Record an objection as `BLOCKED_BY_OBJECTION` and stop. On approval, record the
full review, move the task to `ACTIVE`, and commit. This becomes the common
execution-start commit.

## Task 2: Create isolated execution worktrees

Keep the root worktree on `codex/FP-ADV-001`. Create the Claude branch
`claude/FP-ADV-001` and a separate worktree from the activation commit. Verify
that both routes resolve the same task version, design, plan, baseline hashes,
seed, formulas, transfer fraction, and protocol.

Create only the ignored canonical result roots:

- `results/FP-ADV-001/codex/`;
- `results/FP-ADV-001/claude/`.

Neither route may read the other's first-result evidence, code conclusion, or
output until both first-result and formal-evidence seals exist.

## Task 3: Independently establish the action-gap proof

Each route writes its own `theory.md` before implementation. The proof must
derive or inherit with exact references:

1. the `FP-TU-001` simultaneous event and selective probability statement;
2. the exact V-first fixed-target recovery decomposition;
3. `|(p-q)^T e| <= TV(p,q) span(e)` and `span(e) <= 2||e||_infinity`;
4. the exact local uncertainty `r_a+r_b+2 gamma E_V TV`;
5. normalized softmax query weights, diagonal mass, and effective successor
   rows;
6. the softmax on-group radius plus bounded off-group contamination;
7. the finite-softmax local uncertainty in the frozen task;
8. post-data action selection under one simultaneous event;
9. local-versus-global V-first penalty dominance;
10. the policy-transfer Bellman identity and pointwise improvement theorem.

Stop before implementation if any mandatory step fails. A complete proof of
failure or deterministic counterexample is the required negative route; the
formula may not be repaired after output inspection.

## Task 4: Write failing contract tests first

Independently create `verify_action_gap_certificate.py` and record the expected
missing-module failure before implementing `action_gap_certificate.py`.

Tests cover exact decomposition, total variation and span, common-offset and
disjoint-row extremes, softmax weights and direct matrix equivalence,
contamination bounds, simultaneous selection, local/global dominance, partial
support, policy simplex and exploration floor, tie-breaking, `theta=1/2`, the
Bellman improvement implication, ordered abstentions, strict JSON, and
no-oracle inputs.

## Task 5: Implement the pure certificate module

Independently create `action_gap_certificate.py`. Keep focused public helpers
for:

- exact and softmax effective successor distributions;
- total variation and validated local uncertainty calculations;
- global sup-norm controls;
- deterministic receiver and donor selection;
- probability transfers and policy validation;
- ordered status and abstention reasons;
- strict-JSON conversion.

The module performs no sampling, plotting, file writing, or truth computation.
It accepts only the observable and declared inputs frozen in the task.

## Task 6: Expose route estimates without changing legacy behavior

Use existing fixed-policy estimator functions to reconstruct the exact same
four route estimates. If an evaluator helper is required, add only one
task-authorized additive interface to `evaluate_fixed_policy_q_routes.py` and
prove that all legacy serialized leaves remain unchanged.

For each prefix, construct exact empirical successor rows and finite-softmax
effective rows from observed counts. Direct calculations and the pure helper
must agree under `1e-12` tolerance.

## Task 7: Build the paired evaluator and oracle separation

Create `evaluate_action_gap_certificates.py`. Reproduce the frozen 480
trajectories and estimator metrics, attach the unchanged FP-TU certificate,
then call the pure action-gap module independently for:

- V-first local exact;
- V-first local softmax;
- V-first global exact;
- V-first global softmax;
- Direct-Q global exact;
- Direct-Q global softmax.

Write all new observable outputs below `action_gap_certificate`. Compute true
action ordering, exact Bellman improvement, componentwise value improvement,
and exact return change only in `oracle_audit`; never pass them back into the
certificate builder.

## Task 8: Build strict regression and analysis

Create `analyze_action_gap_certificates.py` to:

- parse strict JSON and reject duplicate keys, NaN, and Infinity;
- verify the three frozen FP-TU baseline hashes and exactly 480 records;
- require exact identity of nonnumeric legacy leaves and `1e-12` numeric
  agreement;
- audit every exact and softmax formula from serialized observable inputs;
- require local/global dominance whenever a global control exists;
- validate every emitted transfer, policy row, and Bellman lower bound;
- summarize update, state, donor, transferred-mass, penalty-reduction, and
  exact/softmax agreement metrics;
- enumerate all oracle reversals or nonmonotone updates separately;
- write strict `regression.json`, `summary.json`, and `checks.log`.

## Task 9: Run and seal smoke evidence

Before formal evaluation, run task-scoped Ruff, all inherited verifiers, the
new verifier, and a small matrix spanning exact/softmax, both mixing and gap
settings, short/long prefixes, partial support, positive LCBs, and abstentions.

Require strict JSON, zero legacy mismatch, no oracle leakage, valid policy
rows, and local/global dominance. Record exact commands, environment, hashes,
failed and successful attempts, anomalies, metrics, limitations, and the
numbered acceptance assessment in `first_result.md`. Each route commits its
implementation and smoke evidence as a blind first-result seal.

## Task 10: Run the one frozen formal matrix

After the corresponding smoke seal, each route runs the formal command once:

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_action_gap_certificates.py
  --tasks 30
  --trajectory-lengths 256 1024 4096 16384
  --n-states 6
  --n-actions 4
  --pi-mins 0.05
  --betas 8
  --mixing 0.08 0.5
  --gap-bonuses 0 0.5
  --gamma 0.70
  --alpha 0.65
  --iterations 160
  --certificate-delta 0.05
  --transfer-fraction 0.5
  --seed 20260829
  --output-dir results/FP-ADV-001/codex
```

Claude changes only the final result directory to `claude`. A mechanical
interruption before complete readable output may be documented and restarted
with the identical command. A scientifically complete output is the sole
formal result and may not be rerun to improve metrics. Each route commits
`formal_result.md` with raw hashes and exact metrics before disclosure.

## Task 11: Disclose and cross-verify

After both routes seal first and formal results, GPT checks Claude's proof,
implementation, history, configuration, raw hashes, schema, and every
acceptance item, then performs a clean reproduction. Claude independently
performs the symmetric verification of GPT.

A task-definition defect is `OBJECTION`. A code, evidence, or inference defect
is `FAIL` and returns the affected route to `ACTIVE` for author repair. Each
verification report ends with exactly `PASS`, `FAIL`, or `OBJECTION` and cites
reproducible evidence.

## Task 12: Synthesize without erasing differences

After reciprocal verification, GPT writes the shared theory and report,
distinguishing theorem results, empirical outcomes, route differences,
negative findings, anomalies, and limitations. Move the task through
`VERIFYING` to `VERIFIED` only after both reports pass or the user records an
explicit exception.

Update `ACTIVE_WORKSPACE.md` and run final placeholder, scope, strict-JSON,
finite-value, verifier, regression, result-location, baseline, branch, and
clean-worktree audits. Do not merge to `main` or push without separate user
approval.
