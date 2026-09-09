# FP-KERN-002: Reused-record oracle and learnability diagnostic

## Task metadata

- Created: 2026-09-09.
- Author: GPT.
- Status: `REVIEW`.
- Task version: `0.2` (DRAFT sealed; scientific content unchanged).
- Predecessor and branch baseline:
  `403884ae6bde46c7c3578ae01d77422ed03faf05` (`FP-KERN-001` verified).
- Approved design commit:
  `be6b7213eeba08d3b9750a850ac2de99ff17be89`.
- DRAFT task-definition baseline:
  `e04db4c17c7648bc751bef1620ab0d01fe1cb3a3`.
- Activation and common execution-start commit: not created while status is
  `DRAFT`.
- Design:
  `docs/superpowers/specs/2026-09-09-kernel-reuse-oracle-learnability-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-09-kernel-reuse-oracle-learnability-plan.md`.
- GPT branch and worktree: `codex/FP-KERN-002` in
  `results/FP-KERN-002/codex_worktree/`.
- Planned Claude branch and worktree: `claude/FP-KERN-002` from the future
  common execution-start commit.
- Common host input directory: `results/FP-KERN-002/input/` outside both route
  worktrees.
- GPT result directory: `results/FP-KERN-002/codex/`.
- Claude result directory: `results/FP-KERN-002/claude/`.
- Classification: long because the conclusion is critical, the diagnostic is
  multi-stage, and independent construction is required.
- Estimated resources: CPU-only, no new trajectories, two deterministic
  analyses of 480 saved records; expected below one hour per route after
  implementation.

## User ruling and purpose

After `FP-KERN-001` rejected the fixed Gaussian estimator, the user asked
whether adding explicit structure would have practical value. GPT proposed a
three-level diagnosis: oracle peer headroom, true hidden-cluster usefulness,
and recovery of those clusters from observable other-action signatures. The
user approved the design and explicitly chose “复用现有记录”.

This task does not search for a better-performing method. It determines where
the predecessor failed while holding every trajectory and old result fixed.
It is empirical feasibility evidence only and cannot certify policy
nondegradation.

## Research question

On the sealed `FP-KERN-001` corpus, is there useful cross-state borrowing
headroom under favorable oracle peer selection, does the generator's actual
latent cluster support useful pooling, and can an observable leave-one-action-
out partition recover enough of that structure to pass the predecessor's
unchanged sparse-estimation screen?

## Falsifiable hypotheses

1. All 480 predecessor records, route outputs, metrics, intervals, and the
   `NOT_SUPPORTED` result reproduce exactly from the frozen source.
2. In `hidden_cluster`, `oracle_q_nearest2` passes the unchanged five-item
   screen, demonstrating favorable fixed-peer borrowing headroom.
3. In `hidden_cluster`, `oracle_generator_cluster` passes the screen,
   demonstrating that the injected latent structure is useful for target-Q
   pooling when membership is known.
4. In `hidden_cluster`, `observable_balanced_cluster` passes the screen,
   demonstrating conditionally practical learnability from allowed data.
5. Applying the observable route unchanged to `current_unstructured`
   distinguishes structure-conditional from general promise.

Scientific failure is valid. The task must report the three hidden-family
gates separately and apply the frozen ordered conclusion rule.

## Frozen input and provenance

The sole source corpus is the corrected canonical GPT output of
`FP-KERN-001`, sealed at
`1001d23273bdf29b92d9b84a3f3956e83819da4a` and finally verified at predecessor
commit `403884ae6bde46c7c3578ae01d77422ed03faf05`.

Required source files and SHA-256 values:

- `config.json`:
  `78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a`;
- `task_results.json`:
  `9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1`.

At activation they are copied byte-for-byte from the predecessor's canonical
result directory to the common host input directory and re-hashed. Both routes
consume exactly that copy. The input must contain 480 records, 240 per family,
with the predecessor's frozen seed and complete 15-task by four-length by
two-mixing by two-gap matrix in each family.

No trajectory may be generated, resampled, extended, shortened, or replaced.
No old result or implementation file may be modified.

## Input boundaries

Observable route inputs are limited to:

- `signature_q` and `signature_counts`;
- `target_sums` and `target_counts`;
- `current_policy`, `pi_min`, and `value_bound`; and
- task identity and non-scientific record metadata.

`oracle_audit.true_q` is permitted only for `oracle_q_nearest2` and evaluation.
`oracle_audit.generator.cluster_by_state` is permitted only for
`oracle_generator_cluster` and cluster-recovery evaluation. Exact values,
cluster labels, prototype fields, target-action signature coordinates,
realized errors, and old route outcomes are prohibited inputs to
`observable_balanced_cluster`.

## Stage 0 predecessor reproduction

Before any new result is interpreted, independently reconstruct the four old
routes from saved observable fields. Match every serialized estimate,
abstention, eligibility flag, distance, bandwidth, denominator, and effective
sample size exactly. Recompute all old metrics and intervals to absolute
tolerance `1e-12` and reproduce both family screens and `NOT_SUPPORTED`.

Any source, schema, record, matrix, route, or reproduction mismatch yields
`INVALID_INPUT` and stops the task.

## Frozen diagnostic routes

For any route source set `G(s,a)`, the estimate is

```text
Q_hat_G(s,a)
  = sum_{s' in G(s,a)} target_sums(s',a)
    / sum_{s' in G(s,a)} target_counts(s',a).
```

Positive-count targets include themselves. Zero-count targets use only other
states. A nonpositive or nonfinite denominator causes explicit abstention.

### `oracle_q_nearest2`

For each `(s,a)`, rank other states having positive target count for action
`a` by `abs(true_q(s,a)-true_q(s',a))`, then by state index. Select at most two
and include the target state only when its target count is positive. This
favorable route may use true Q only for peer selection and is not a universal
upper bound.

### `oracle_generator_cluster`

For hidden-cluster records only, pool the three states with the same true
`cluster_by_state` label. This route may not read true Q when constructing its
source set or estimate.

### `observable_balanced_cluster`

For each record and target action:

1. recompute all leave-one-action-out pair distances from common positive-count
   non-target signature actions;
2. replace unavailable pair distances with exactly `1.0`;
3. enumerate all ten unique partitions of six states into two groups of three;
4. score each partition by sorting its six squared within-group distances and
   taking their float64 arithmetic mean;
5. select only a unique exact minimum, otherwise abstain with
   `partition_tie`; and
6. pool target sufficient statistics within the selected group.

State zero appears in the first serialized group only to remove cluster-label
symmetry. No scientific tie is resolved by state index. The route is allowed
to know that the structural alternative has two balanced clusters, but never
their labels.

## Metrics and common sets

Reuse exactly the predecessor definitions for count bins, per-record RMSE,
common finite comparisons, sparse states, complete four-action top-action
rows, one-sided false improvement, and two-sided 95% Student-t intervals.

The zero-count coverage denominator for every route is the frozen set with
`target_counts == 0` and reconstructed predecessor
`signature_eligible == true`. Abstentions remain uncovered. Zero-count RMSE
uses all zero pairs on which the route and `action_only_pool` are finite.
`1-4`-count RMSE uses common finite pairs against `local_unpooled`.

Each route passes a family only when all five conditions hold:

1. eligible zero-count coverage is at least 50%;
2. zero-count RMSE falls by at least 10% versus `action_only_pool` and the
   paired improvement interval excludes zero;
3. `1-4`-count RMSE falls by at least 10% versus `local_unpooled` and the paired
   interval excludes zero;
4. sparse-state top-action accuracy improves by at least five percentage
   points versus `action_only_pool` and the paired interval excludes zero; and
5. false-improvement rate is no more than one percentage point above
   `action_only_pool` on common finite action differences.

Secondary hidden-family diagnostics are adjusted Rand index, same-cluster peer
precision, and the observable fraction of oracle-cluster RMSE improvement.
They cannot override a screen item.

## Frozen conclusion rule

Report `peer_headroom`, `generator_structure_useful`, and
`observable_structure_useful` separately, then apply this ordered rule:

1. `INVALID_INPUT` for any Stage 0, provenance, schema, isolation, or
   reconstruction failure;
2. `GENERAL_PROMISING` if the observable route passes both families;
3. `STRUCTURE_CONDITIONAL_PROMISING` if it passes hidden and fails current;
4. `REPRESENTATION_GAP` if it fails hidden but the generator-cluster route
   passes;
5. `GENERATOR_STRUCTURE_MISALIGNED` if both cluster routes fail hidden but the
   Q-nearest route passes; or
6. `NO_BORROWING_EVIDENCE` if all three diagnostic routes fail hidden.

Only an observable-route pass supports `PROMISING`. No category is a safety or
general identifiability theorem.

## Formal protocol

Stage 0 uses all 480 source records. The fixed smoke subset has `task_index=0`,
both families, lengths 256 and 1024, both mixing values, and both gap bonuses,
for exactly 16 records. After smoke and all checks pass, each independent route
generates one full 480-record diagnostic result. There is no formal trajectory
run and no scientific command-line parameter.

The common input is immutable. Each route's full result directory must be
absent or empty before its sole full-output generation. A rerun that replaces
diagnostic records requires user authorization.

## Allowed work

### Shared read-only inputs

- `AGENTS.md`, `ACTIVE_WORKSPACE.md`, this task, design, and plan;
- the verified `FP-KERN-001` task, code, evidence, and two frozen input files;
- verified predecessor scientific baselines needed by inherited checks.

### GPT write scope on `codex/FP-KERN-002`

- `analyze_kernel_reuse_diagnostics.py`;
- `verify_kernel_reuse_diagnostics.py`;
- `docs/research_branches/FP-KERN-002/codex/`;
- this task, design, plan, and compact workspace pointer; and
- `results/FP-KERN-002/codex/`.

### Claude write scope on `claude/FP-KERN-002`

- independent versions of the two Python entry points;
- `docs/research_branches/FP-KERN-002/claude/`; and
- `results/FP-KERN-002/claude/`.

The host-level `results/FP-KERN-002/input/` is a common immutable input, not a
route output.

## Prohibited work

- No implementation or analysis before status `ACTIVE`.
- No modification, regeneration, filtering, replacement, or selective loading
  of an old record.
- No new environment, trajectory, state feature, representation learner,
  adaptive bandwidth, cluster number, cluster balance, neighbor count,
  shrinkage coefficient, metric, threshold, or decision rule.
- No oracle value or label at the observable route boundary.
- No target-action signature coordinate in observable partition construction.
- No result-selected repair, favorable-cell claim, denominator restriction,
  silent fallback, or unrecorded abstention.
- No force operation, history rewrite, `main` merge, push, publication,
  external message, unsafe permission bypass, or new fee category.

## Expected artifacts

Each independent route provides its two implementation files, verifier-first
record, Stage 0 evidence, smoke evidence, blind implementation seal, full
result seal, method/result report, source manifest, strict result bundle,
commands, environment, hashes, anomalies, acceptance assessment, and final
reciprocal verification report.

Each route's full result bundle contains at least:

- `source_manifest.json`;
- `baseline_reproduction.json`;
- `diagnostic_records.json`;
- `summary.json`;
- `analysis.json`;
- `environment.json`;
- `commands.log`; and
- `checks.log`.

## Acceptance criteria

1. Both source files match the frozen hashes byte-for-byte.
2. Exactly 480 records and every frozen matrix cell are present once.
3. Stage 0 exactly reproduces old route internals and the predecessor result.
4. The old input and result directories remain unchanged.
5. Oracle-Q selects at most two positive-count peers by the frozen order.
6. Generator-cluster uses labels but not true Q for construction.
7. Observable clustering cannot receive an oracle or target-action signature
   input.
8. All ten balanced partitions, missing distance `1.0`, sorted scoring, unique
   minimum, and tie abstention are exact and deterministic.
9. Observable route output is equivariant under state relabelling whenever the
   optimum is unique; tied optima abstain.
10. Every estimate reconstructs from declared source sets, sums, and counts.
11. Every zero-count estimate uses only other-state target observations.
12. Coverage keeps every frozen eligible pair in the denominator.
13. All common sets, errors, actions, false improvements, intervals, cluster
    diagnostics, and screens independently reconstruct.
14. The ordered final classification is applied exactly.
15. Test-first, smoke, full reconstruction, source/output hash, inherited
    predecessor verifier, strict JSON, and task-scoped Ruff checks pass.
16. Commands, environment, failures, repairs, anomalies, hashes, limitations,
    and acceptance judgments are recorded.
17. GPT and Claude start from one activation commit, use one common input,
    seal before disclosure, and reciprocally reproduce the result.
18. Both reciprocal reports end `PASS`, or the task remains unverified pending
    user ruling.
19. `ACTIVE_WORKSPACE.md` is current and `main` remains unchanged without
    separate user approval.

## Failure and stopping conditions

`INVALID_INPUT` ends scientific interpretation when source identity, Stage 0,
schema, isolation, or reconstruction fails. Ordinary implementation failures
return to the original author without changing frozen science.

Stop affected work and notify the user if Claude pre-review returns
`OBJECTION`; a repair would change source data, peer count, cluster structure,
missing-distance rule, metrics, thresholds, or classification; another edit
overlaps an allowed path; the common input cannot be isolated; Claude is
unavailable; or any permission, cost, publication, communication, merge, or
rerun scope expands beyond authorization.

## Route assignments and independence

GPT and Claude independently implement the two entry points from the common
activation commit and consume the same frozen input. Neither reads the other's
implementation, smoke conclusion, or full result before both blind seals.
After disclosure, each performs executable reciprocal verification and records
`PASS`, `FAIL`, or `OBJECTION` with evidence.

## Pre-review, objections, and evidence

### Claude read-only pre-review

- Status: pending while task is `REVIEW`.
- Review target: the commit that transitions this frozen task to `REVIEW`;
  its exact identity is recorded with Claude's response.
- Outcome: not yet assigned.
- Evidence path:
  `docs/research_branches/FP-KERN-002/codex/claude_pre_review.md`.

### Objection

- Status: none recorded.
- User ruling: not applicable.

### Execution evidence

- Status: prohibited while task is `REVIEW`.

## Definition of done

- [ ] Claude pre-review is recorded with no unresolved objection.
- [ ] Source identity and Stage 0 reproduction pass.
- [ ] Both independent route artifacts are complete and reproducible.
- [ ] Both reciprocal verification reports end `PASS`.
- [ ] Every acceptance criterion has evidence.
- [ ] The ordered conclusion follows the frozen decision rule.
- [ ] Differences are explained or ruled on by the user.
- [ ] `ACTIVE_WORKSPACE.md` is current.
- [ ] `main` remains unchanged unless the user separately approves a merge.
