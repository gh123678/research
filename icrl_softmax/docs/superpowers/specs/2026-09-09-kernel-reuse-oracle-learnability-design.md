# FP-KERN-002: Reused-record oracle and learnability diagnostic

## Status and user decision

Design date: 2026-09-09.

The user approved a diagnostic of whether cross-state sharing has practical
potential and explicitly chose to reuse the sealed `FP-KERN-001` records
rather than generate new trajectories. This design freezes that diagnostic
before any new route is evaluated.

This is a diagnostic study, not a safety theorem or a policy-improvement
claim. Its purpose is to locate the failure observed in `FP-KERN-001` among
three possibilities:

1. the available cross-state target-action samples offer too little borrowing
   headroom even with oracle neighbor selection;
2. the injected hidden clusters do not align closely enough with target-action
   Q values for cluster pooling to help; or
3. useful latent structure exists but the observable other-action signature
   cannot recover it.

## Context

`FP-KERN-001` tested a leave-one-action-out state signature followed by a
fixed-bandwidth Gaussian kernel. Both independent routes classified that
construction as `NOT_SUPPORTED`. The corrected GPT route produced 100%
eligible zero-count coverage and positive record/action Spearman correlations,
but it missed the frozen zero-count RMSE and sparse top-action thresholds and
substantially worsened `1-4`-count RMSE.

That result rejects the particular signature-plus-Gaussian estimator. It does
not distinguish lack of cross-state headroom from generator misalignment or
representation failure. `FP-KERN-002` supplies that decomposition without
changing the data.

## Approaches considered

### Selected: sealed-record diagnostic ladder

Reconstruct the old result exactly, then evaluate oracle-Q peers, true hidden
clusters, and observable balanced clusters on the same saved target sums and
counts. This isolates the neighbor-selection question, avoids new random
variation, and is the fastest direct answer to the user's question.

### Rejected for this task: deterministic trajectory regeneration

Regenerating the same seeds could validate the old generator but would add no
information about the new structural routes. It would also create a new
formal-run identity and unnecessary opportunities for implementation drift.

### Deferred: new structured environments or adaptive estimators

Changing the environment, learning a flexible representation, choosing an
adaptive number of neighbors, or tuning shrinkage could be useful later, but
would combine mechanism discovery with method tuning. None is allowed in this
diagnostic.

## Frozen source data

The scientific predecessor is the verified `FP-KERN-001` task on
`codex/FP-KERN-001` at commit
`403884ae6bde46c7c3578ae01d77422ed03faf05`. The corrected formal-result seal
is `1001d23273bdf29b92d9b84a3f3956e83819da4a`.

The sole data source is the corrected canonical GPT record corpus:

`results/FP-KERN-001/codex/task_results.json`

Its frozen SHA-256 is:

`9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1`.

The matching configuration SHA-256 is:

`78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a`.

At task activation, GPT will copy the two files byte-for-byte into a common
`results/FP-KERN-002/input/` directory, write a source manifest, re-hash the
copy, and then freeze all three common-input files. GPT and Claude must consume
the same frozen copy and independently verify the manifest. The seven original
`FP-KERN-001` canonical artifacts and both old worktrees remain read-only.

The corpus contains exactly 480 records: 240 `current_unstructured` and 240
`hidden_cluster`, spanning the already-frozen task, trajectory-length, mixing,
and reward-gap matrix. No trajectory is generated, extended, resampled, or
replaced.

Allowed observable fields are `signature_q`, `signature_counts`,
`target_sums`, `target_counts`, `current_policy`, `pi_min`, and `value_bound`.
`true_q` and `cluster_by_state` are available only to the explicitly named
oracle routes and evaluation audit. No observable route may receive either
field.

## Stage 0: exact predecessor reproduction

Before evaluating a new route, independently reconstruct from the frozen
records:

- all four `FP-KERN-001` route outputs;
- count-bin coverage, RMSE, action ordering, sparse-state top-action accuracy,
  false-improvement rate, and paired intervals;
- the corrected family screen and `NOT_SUPPORTED` classification.

Reproduction must match every serialized route estimate and reason exactly and
all floating summaries to absolute tolerance `1e-12`. Source hashes, record
count, task identity, family counts, matrix cells, and route names must match
the frozen manifest. Any mismatch stops the task as `INVALID_INPUT`; no new
diagnostic result may be interpreted.

## Frozen diagnostic routes

All estimates use only the saved target sufficient statistics. For a selected
source set `G(s,a)`, define

```text
Q_hat_G(s,a)
  = sum_{s' in G(s,a)} target_sums(s',a)
    / sum_{s' in G(s,a)} target_counts(s',a).
```

States with zero target count contribute zero numerator and denominator. A
nonpositive or nonfinite denominator causes an explicit abstention. For a
positive-count target pair, every diagnostic source set includes the target
state itself; for a zero-count target pair, only other-state observations can
contribute.

### Existing controls

`local_unpooled`, `action_only_pool`, and the corrected
`leave_one_action_out_kernel` are reconstructed unchanged. Their old
serialized values are not silently reused as computed outputs; exact replay is
part of the evidence.

### Route A: `oracle_q_nearest2`

This non-deployable headroom route may read `oracle_audit.true_q` only to rank
peer states. For each `(s,a)`, among other states with positive saved target
count for action `a`, select up to two with smallest
`abs(true_q(s,a) - true_q(s',a))`. Break exact ties by state index. Combine
those peers with `s` itself when `target_counts(s,a) > 0` and apply the common
count-weighted formula.

The number two is frozen because the hidden generator has balanced
three-state clusters, so each state has two latent peers. This route measures
a favorable fixed-peer borrowing ceiling; it is not claimed to be a universal
upper bound over all estimators.

### Route B: `oracle_generator_cluster`

This route exists only for `hidden_cluster` records. It may read
`oracle_audit.generator.cluster_by_state`. Its source set is the three states
whose true generator cluster equals the target state's cluster. It may not
read true Q when constructing the estimate.

This route asks whether the actual injected latent structure is useful when
cluster membership is known perfectly. It is an oracle structural diagnostic,
not a deployable method.

### Route C: `observable_balanced_cluster`

This route receives only the observable fields. It is fit independently within
each record and target action.

1. Recompute every leave-one-action-out cross-state distance using only common
   positive-count non-target signature actions, exactly as in
   `FP-KERN-001`; a finite cross-state distance requires at least two such
   common actions.
2. Replace an unavailable cross-state distance by `1.0`, the conservative
   maximum normalized distance under the bounded construction.
3. Enumerate the ten unique balanced partitions of six labelled states into
   two groups of three. Requiring state zero to appear in the first serialized
   group removes only the duplicate cluster-label representation; it does not
   remove any partition.
4. Score a partition by sorting its six squared within-group pair distances in
   ascending float64 order and taking their arithmetic mean. Sorting makes the
   reduction independent of pair enumeration. Select the partition only when
   its score is the unique exact minimum. If two or more partitions have the
   same minimum float64 score, abstain with `partition_tie` rather than use a
   state-index-dependent scientific choice.
5. Estimate every `(s,a)` by count-weighted pooling within its selected group.

The route knows that the structural alternative contains two balanced
clusters. It never sees their labels. This is deliberately a favorable
learnability diagnostic; a positive result would justify a later task that
removes the known-balance assumption. It uses neither target-action signature
coordinates nor target outcomes when learning the partition. The independent
signature and target streams preserved in the source corpus prevent direct
fit/evaluation reuse.

## Metrics

The primary evaluation reuses the exact `FP-KERN-001` definitions:

- eligible zero-count coverage over all eligible pairs;
- zero-count RMSE versus `action_only_pool`;
- `1-4`-count RMSE versus `local_unpooled`;
- sparse-state top-action accuracy versus `action_only_pool`;
- one-sided false-improvement rate on common finite action differences; and
- per-record paired differences with two-sided 95% Student-t intervals.

For every route, the zero-count coverage denominator is exactly the frozen set
of pairs with `target_counts == 0` and reconstructed
`leave_one_action_out_kernel.signature_eligible == true` in `FP-KERN-001`.
An eligible pair on which a new route abstains remains uncovered. Zero-count
RMSE continues to use all zero-count pairs on which the new route and
`action_only_pool` are both finite, matching the predecessor's common-finite
comparison rule.

Each diagnostic route is screened with the same five thresholds:

1. at least 50% eligible zero-count coverage;
2. at least 10% zero-count RMSE reduction with paired interval excluding zero;
3. at least 10% `1-4`-count RMSE reduction with paired interval excluding
   zero;
4. at least five percentage points sparse-state top-action improvement with
   paired interval excluding zero; and
5. false-improvement rate at most one percentage point above
   `action_only_pool`.

For `hidden_cluster`, secondary cluster diagnostics use only record/actions on
which the observable partition has a unique minimum. `partition_tie`
record/actions are excluded and counted. Compute one adjusted Rand index over
the six state labels per emitted record/action, average the finite values
equally, and report a two-sided 95% Student-t interval; fewer than two values is
unavailable. Peer precision is micro-averaged over the twelve directed peer
assignments from every emitted record/action: the numerator counts assignments
whose generator labels agree and the denominator counts all such assignments.

Report oracle-benefit recovery separately for zero and `1-4` count bins. For
each bin, restrict to the exact record/pair set on which the observable route,
generator-cluster route, and relevant baseline are all finite. Within each
record compute all three RMSE values, then define

```text
recovery
  = mean(RMSE_baseline - RMSE_observable)
    / mean(RMSE_baseline - RMSE_generator_cluster).
```

The zero-count baseline is `action_only_pool`; the `1-4` baseline is
`local_unpooled`. A missing, nonfinite, or nonpositive denominator makes the
ratio unavailable. Do not clip the ratio. All three cluster diagnostics are
secondary and cannot override the five-item screen.

## Decision rule

Report the following three gates independently for the hidden-cluster family:

- `peer_headroom`: whether `oracle_q_nearest2` passes all five items;
- `generator_structure_useful`: whether `oracle_generator_cluster` passes all
  five items; and
- `observable_structure_useful`: whether
  `observable_balanced_cluster` passes all five items.

Then assign exactly one scoped conclusion using this ordered decision rule:

1. `INVALID_INPUT` if Stage 0, provenance, schema, isolation, or
   reconstruction fails;
2. `GENERAL_PROMISING` if `observable_balanced_cluster` passes in both
   families;
3. `STRUCTURE_CONDITIONAL_PROMISING` if the observable route passes in
   `hidden_cluster` and fails in `current_unstructured`;
4. `REPRESENTATION_GAP` if the observable route fails in `hidden_cluster` but
   `oracle_generator_cluster` passes in `hidden_cluster`;
5. `GENERATOR_STRUCTURE_MISALIGNED` if both cluster routes fail in
   `hidden_cluster` but `oracle_q_nearest2` passes in `hidden_cluster`; or
6. `NO_BORROWING_EVIDENCE` if all three diagnostic routes fail in
   `hidden_cluster`.

The observable route takes precedence when it passes because it is the only
deployable-data route; the Q-nearest construction is a favorable fixed-peer
diagnostic, not a mathematical upper bound. A favorable oracle result alone
cannot be described as practical feasibility. Only an observable-cluster pass
can support the word `PROMISING`, and even that remains a diagnostic rather
than a safety result.

## Isolation, implementation, and outputs

The future implementation is limited to two new entry points:

- `analyze_kernel_reuse_diagnostics.py`, containing pure reconstruction,
  route, metric, and reporting functions; and
- `verify_kernel_reuse_diagnostics.py`, containing independent fixtures and
  full-corpus checks.

No old Python file is modified. At activation GPT alone may create the common
input's two byte-identical copies and `source_manifest.json`; after its first
successful hash verification that directory becomes immutable to both routes.
GPT otherwise writes only on `codex/FP-KERN-002` under
the two new entry points, `docs/research_branches/FP-KERN-002/codex/`, the
formal task/design/plan/workspace pointer, and
`results/FP-KERN-002/codex/`. Claude uses `claude/FP-KERN-002` and its own
evidence/result directory.

Each route must write a strict-JSON result bundle containing at least:

- `source_manifest.json`;
- `baseline_reproduction.json`;
- `diagnostic_records.json`;
- `summary.json`;
- `analysis.json`;
- `environment.json`;
- `commands.log`; and
- `checks.log`.

For `current_unstructured`, `oracle_generator_cluster` must serialize one
route-level `not_applicable_family` status with no estimates or source sets and
must be omitted from the current-family screen.

## Verification

Required fixtures cover:

- exact source hashes and 480-record matrix identity;
- byte-preserving common-input copy;
- exact predecessor route and summary reconstruction;
- oracle-field rejection at the observable route boundary;
- target-action signature exclusion;
- deterministic two-neighbor selection and tie-breaking;
- enumeration of exactly ten balanced partitions;
- missing-distance replacement by exactly `1.0`;
- permutation-invariant unique-minimum selection and `partition_tie`
  abstention;
- state-label permutation equivariance, modulo the documented canonical label
  representation;
- zero-count estimates using only other-state target observations;
- explicit denominator abstention;
- count-bin, comparable-set, top-action, false-improvement, and interval
  reconstruction; and
- exact decision-table classification.

Because the task is conclusion-critical and multi-stage, GPT and Claude must
independently implement the analysis from one frozen task commit, consume the
same common input, seal results before disclosure, and reciprocally verify the
other route under `AGENTS.md`.

## Stopping conditions

Stop and report rather than adapting the study if:

- any source hash, schema, record count, family count, or baseline metric does
  not reproduce;
- the observable route requires true Q, hidden labels, target-action signature
  values, realized target errors, or another oracle input;
- a requested repair would change the source records, neighbor count, cluster
  count or balance, missing-distance value, metrics, thresholds, or decision
  rule;
- an old artifact or old implementation would need modification;
- Claude pre-review returns `OBJECTION` or either independent route cannot be
  isolated; or
- any rerun, external communication, fee, push, publication, or merge exceeds
  the user's authorization.

## Interpretation boundary

This design can determine whether a favorable, explicitly structured
three-state sharing model has empirical headroom and whether the old observable
signature can recover that structure. It cannot prove that all cross-state
generalization methods fail, establish identifiability without assumptions,
or certify policy nondegradation. A positive observable result would justify a
separate estimator-and-safety task; a negative result would close this specific
fixed-peer, balanced-cluster continuation without post-hoc tuning.
