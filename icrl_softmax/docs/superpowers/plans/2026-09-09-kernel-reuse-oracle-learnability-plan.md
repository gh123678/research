# FP-KERN-002 implementation and evidence plan

Date: 2026-09-09

Task: `docs/research_tasks/FP-KERN-002.md`

Design:
`docs/superpowers/specs/2026-09-09-kernel-reuse-oracle-learnability-design.md`

## 1. Governance and activation

1. Seal the DRAFT task and this plan on `codex/FP-KERN-002`.
2. Record the DRAFT commit, transition the task to `REVIEW`, and update the
   workspace pointer without changing the scientific contract.
3. Check branch, clean worktree, predecessor identities, design identity,
   source-artifact hashes, changed paths, and local remote-tracking state.
4. Start local Claude Code for task-scoped read-only pre-review. Require an
   itemized `APPROVED` or `OBJECTION` response.
5. If Claude returns `OBJECTION`, record it and stop for user ruling. If it
   returns `APPROVED`, incorporate only non-scientific clarifications, then
   transition `REVIEW -> ACTIVE` in a common execution-start commit.
6. Create `claude/FP-KERN-002` from that exact commit. Neither route may see
   the other's implementation or result before both blind seals.

No input copy, implementation, smoke analysis, or full-corpus diagnostic may
run before `ACTIVE`.

## 2. Freeze the common input

After activation, GPT copies only the sealed predecessor `config.json` and
`task_results.json` into the host-level common
`results/FP-KERN-002/input/` directory. Verify byte identity against the task's
SHA-256 values before either route reads the copy.

GPT writes `source_manifest.json` with source task, corrected formal seal, source
paths, sizes, hashes, record and family counts, matrix values, and copy time.
It then freezes all three common-input files. Both routes independently verify
the manifest. Make no change to any `FP-KERN-001` file.

## 3. Verifier-first implementation

Each route independently creates `verify_kernel_reuse_diagnostics.py` before
the analyzer. Fixtures must cover:

- exact source-manifest validation and 480-record identity;
- oracle-field rejection at the observable route boundary;
- target-action exclusion from each observable partition;
- exact two-neighbor oracle-Q selection and state-index tie order;
- all ten balanced partitions;
- unavailable distance replacement by exactly `1.0`;
- permutation-invariant sorted scoring;
- unique-minimum selection and `partition_tie` abstention;
- group count-weighted estimates, self inclusion for positive pairs, and
  other-state-only zero-count estimates;
- denominator failure and nonfinite-input rejection;
- adjusted Rand index and same-cluster peer diagnostics;
- predecessor metric reconstruction and the ordered final decision table.

The verifier must fail because the analyzer module is absent before its first
implementation run. Record that expected failure.

## 4. Pure reconstruction and route functions

Implement `analyze_kernel_reuse_diagnostics.py` with a pure layer that has no
file I/O and separate adapters for:

- strict record and observable-input validation;
- independent replay of the three predecessor controls;
- reconstruction of the predecessor signature distances and eligibility;
- `oracle_q_nearest2` source selection;
- `oracle_generator_cluster` source selection;
- exhaustive `observable_balanced_cluster` selection;
- count-weighted group estimation and abstention reasons;
- count bins, common finite sets, RMSE, top-action, one-sided
  false-improvement, Student-t intervals, cluster recovery, five-item screens,
  and final classification.

Oracle route functions receive oracle fields explicitly. The observable route
accepts one narrow typed mapping and rejects unknown oracle/prohibited keys.
Do not modify or import result values from old analyzers as trusted metrics.

## 5. Stage 0 and smoke gate

First run Stage 0 over all 480 records in read-only mode. It must match every
serialized predecessor estimate and reason exactly, match summary floats to
absolute tolerance `1e-12`, and recover the predecessor `NOT_SUPPORTED`
classification. Any mismatch stops the task as `INVALID_INPUT`.

After Stage 0 passes, run a 16-record smoke subset fixed by:

```text
task_index=0
families=current_unstructured,hidden_cluster
trajectory_lengths=256,1024
mixing=0.08,0.50
gap_bonuses=0,0.50
records=16
```

Smoke must exercise all three new routes, ordinary source-unavailable
abstention, strict JSON, metric reconstruction, and the output schema.
Deterministic verifier fixtures must exercise `partition_tie`; the frozen
16-record subset is not required to contain a naturally tied partition. Smoke
cannot change any scientific constant.

## 6. Independent Claude route

Claude starts from the identical activation commit and consumes the same
common input. It independently writes its verifier and analyzer, passes Stage
0 and smoke, and seals its implementation before running the full diagnostic.

Claude writes only under its assigned branch, code files, evidence directory,
and `results/FP-KERN-002/claude/`. If authentication, quota, permissions, or
environment prevent independent execution, record the blocker and notify the
user; do not substitute a Codex subagent.

## 7. One full reused-record diagnostic per route

After verifier, Stage 0, Ruff, and smoke pass, each route may generate its
full 480-record diagnostic bundle exactly once. The command exposes no
scientific tuning parameters: task identity, input hashes, two peers, two
balanced clusters, missing distance `1.0`, metrics, thresholds, and decision
rule are constants checked against the task.

Each route writes to a previously absent or empty route result directory. A
scientific rerun or output replacement requires a documented user exception.
A deterministic report reconstruction from sealed diagnostic records is
allowed only when it does not modify the records.

Before disclosure, run:

- the route's new verifier;
- task-scoped Ruff on the two new files;
- strict read-only full-bundle reconstruction;
- source and output SHA-256 checks; and
- the inherited `FP-KERN-001` verifier to demonstrate no regression.

Seal implementation, commands, anomalies, outputs, metrics, classification,
limitations, and acceptance judgments in the route's evidence directory.

## 8. Reciprocal verification and synthesis

After both blind full-result seals:

1. GPT runs Claude's verifier, full reconstruction, Ruff, and hash checks and
   writes a report ending exactly `PASS`, `FAIL`, or `OBJECTION`.
2. Claude performs the symmetric executable verification of GPT and writes an
   evidence-linked report with the same terminal vocabulary.
3. Each original author repairs only its own ordinary implementation defect.
   Frozen diagnostic outputs are not replaced without user authorization.
4. Reconcile every gate and the ordered final classification from raw saved
   records.
5. Transition `ACTIVE -> VERIFYING -> VERIFIED` only after both reports pass
   and every acceptance criterion has evidence.
6. Update `ACTIVE_WORKSPACE.md` and report the scoped scientific conclusion.
   Do not merge or push without separate user approval.

## 9. Required evidence

Each route records:

- branch, commits, baseline, design, task version, Python and NumPy versions;
- common source sizes and hashes;
- expected test-first failure and successful verifier, Ruff, Stage 0, smoke,
  full analysis, reconstruction, and hash commands;
- all abstention and anomaly counts;
- every count-bin denominator, coverage, RMSE, top-action,
  false-improvement, interval, cluster-recovery, and five-item result;
- the three independent hidden-family gates and ordered final classification;
- exact output paths and SHA-256 identities; and
- numbered acceptance-criteria judgments and interpretation limits.
