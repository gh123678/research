# FP-KERN-001 implementation and evidence plan

Date: 2026-09-09

Task: `docs/research_tasks/FP-KERN-001.md`

Design:
`docs/superpowers/specs/2026-09-09-kernel-state-generalization-feasibility-design.md`

## 1. Governance and activation

1. Seal the DRAFT task and this plan on `codex/FP-KERN-001`.
2. Record the DRAFT commit, transition the task to `REVIEW`, and update the
   workspace pointer without changing scientific content.
3. Verify the branch, clean worktree, verified baseline, design identity,
   changed paths, and remote state.
4. Launch local Claude Code with read-only task-scoped inputs. Require an
   itemized `APPROVED` or `OBJECTION` response.
5. If `OBJECTION`, record it and stop. If `APPROVED`, close any nonblocking
   cautions, commit the unchanged or clarified REVIEW contract, then transition
   to `ACTIVE` in a separate common execution-start commit.
6. Create isolated Claude and GPT result/evidence directories from that common
   execution-start commit. Preserve nondisclosure until both route seals.

No implementation or experimental command may run before step 5 completes.

## 2. Pure-kernel tests before implementation

Create `verify_kernel_state_generalization.py` first with deterministic
fixtures for:

- leave-one-action-out exclusion of the target action;
- two-common-action support gating;
- identical, ordered, and degenerate distances;
- deterministic median bandwidth;
- Gaussian weight normalization and monotonicity;
- zero-count target estimation from other states;
- self-only reduction to the local pair mean;
- same-action control abstention at zero target count or missing target
  signature;
- state-label permutation equivariance;
- malformed shapes, noninteger/negative counts, nonfinite values, invalid
  policies, zero denominators, and reason ordering;
- rejection of every oracle/prohibited input name.

The verifier must fail before the pure module exists.

## 3. Pure estimator module

Implement `kernel_state_generalization.py` as a no-I/O module with small
functions for:

- strict observable-input validation;
- recovery-target aggregation;
- leave-one-action-out common-support masks;
- normalized distances and per-action median bandwidths;
- Gaussian weights and effective sample size;
- local, unconditional pool, same-action anchor, and primary estimates;
- explicit ordered abstention records;
- deterministic sparse-state diagnostic policy construction.

The pure module must not import MDP generators or oracle helpers. Run the new
verifier and task-scoped Ruff after each bounded implementation stage.

## 4. Hidden-cluster generator

Create `kernel_generalization_mdps.py` with:

- a narrow wrapper around the unchanged current `make_mdp`/`make_policy`;
- a two-balanced-cluster generator with random state-label permutation;
- frozen `0.90/0.10` transition and reward composition;
- the unchanged sticky-mixing form and action-zero reward bonus;
- deterministic p0 and stream seed derivation;
- strict probability-row, reward-bound, and reproducibility validation.

Extend the verifier with generator determinism, hidden-field separation,
cluster-balance, transition-simplex, reward-bound, and label-permutation tests.

## 5. Evaluator

Create `evaluate_kernel_state_generalization.py` that:

1. refuses a nonempty formal output directory;
2. records all CLI arguments and validates the exact task identity;
3. creates each MDP and fixed policy once per record;
4. derives nonoverlapping value/signature/target RNG substreams;
5. builds the unchanged exact V-first value estimate from the value stream;
6. serializes observable route inputs before any oracle audit;
7. evaluates all four routes from the same observable record;
8. attaches hidden structure and exact policy quantities only under
   `oracle_audit` after route outputs are frozen;
9. writes strict JSON, environment, commands, checks, and hashes atomically;
10. fails closed on invalid records without dropping them from coverage.

Use existing helpers through imports or copied minimal formulas only where the
task allows. Do not modify any existing evaluator.

## 6. Analyzer

Create `analyze_kernel_state_generalization.py` as read-only by default. It
must independently reconstruct:

- record count and frozen cell membership;
- three stream seeds and observable aggregate identities;
- every mask, distance, bandwidth, weight, denominator, effective sample size,
  route estimate, and abstention;
- oracle Q errors, action orderings, sparse-state top actions,
  false-improvement decisions, and diagnostic-policy values;
- count-bin and comparable-pair denominators;
- per-record paired differences and Student-t intervals;
- the five-item screen for each environment family;
- the final four-way classification.

Any mismatch is an analysis failure, not an excluded record.

## 7. GPT smoke and blind seal

Run the mandatory smoke matrix only after all unit checks pass:

```text
tasks=1
families=current_unstructured,hidden_cluster
trajectory_lengths=256,1024
mixing=0.08,0.50
gap_bonuses=0,0.50
n_states=6
n_actions=4
pi_min=0.05
gamma=0.70
iterations=160
seed=20260909
```

The smoke validates execution, schema, both families, all count/abstention
paths reachable in the fixture, strict analysis reconstruction, and oracle
separation. It cannot change scientific constants. Record commands and results
in `docs/research_branches/FP-KERN-001/codex/first_result.md`, then commit the
implementation and blind first-result seal before any Claude disclosure.

## 8. Claude independent route

After activation, start Claude from the identical common execution-start
commit on `claude/FP-KERN-001`. Provide the frozen task, design, plan, verified
baseline, and allowed code inputs, but no GPT implementation, smoke conclusion,
or result. Claude independently implements the five files, runs its verifier,
Ruff, smoke, seals its first result, and only then runs its one formal matrix.

If Claude encounters authentication, quota, permission, or environment
failure, record the blocker and notify the user. Do not replace the independent
route with a Codex subagent.

## 9. Frozen formal execution

Each route runs exactly one formal matrix after its blind implementation seal:

```text
tasks=15
families=current_unstructured,hidden_cluster
trajectory_lengths=256,1024,4096,16384
mixing=0.08,0.50
gap_bonuses=0,0.50
n_states=6
n_actions=4
pi_min=0.05
gamma=0.70
iterations=160
seed=20260909
records=480
```

Run the strict analyzer, all relevant inherited/new verifiers, task-scoped
Ruff, hash checks, and output inventory. Record failures and successes. A
formal rerun requires a documented user exception; ordinary defects that can
be repaired from saved observable artifacts must not regenerate the matrix.

Seal each route's formal result before disclosure.

## 10. Reciprocal verification and synthesis

After both formal seals:

1. GPT checks out or reads Claude's sealed route without editing it, runs its
   verifier/analyzer/Ruff, reconstructs all 480 records, and records exactly
   `PASS`, `FAIL`, or `OBJECTION`.
2. Claude performs the symmetric executable verification of GPT and writes its
   report only in its assigned evidence path.
3. Each original author repairs only its own implementation if verification
   returns `FAIL`; no formal matrix is rerun without user ruling.
4. Reconcile any metric or classification difference from raw observable
   evidence.
5. Transition `ACTIVE -> VERIFYING -> VERIFIED` only when both reciprocal
   reports pass and every acceptance item has evidence.
6. Update `ACTIVE_WORKSPACE.md` and present the verified conclusion. Do not
   merge or push without separate user approval.

## 11. Required evidence checklist

For each route record:

- exact branch, commits, baseline, task version, Python/Numpy versions;
- exact verifier, Ruff, smoke, evaluator, analyzer, and hash commands;
- output paths and SHA-256 identities;
- failed and successful runs, repairs, and anomalies;
- cell counts, count-bin denominators, coverage, RMSE/MAE, ordering accuracy,
  false-improvement rate, policy diagnostics, and paired intervals;
- hidden-cluster screen, current-family screen, and final classification;
- limitations, especially that feasibility is not a safety theorem;
- numbered acceptance-criteria assessment.
