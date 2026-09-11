# FP-KERN-002 Claude route: blind implementation and smoke seal

Date: 2026-09-09.

Route: Claude independent executor on `claude/FP-KERN-002`, worktree
`results/FP-KERN-002/claude_worktree/`, from the common execution-start
commit `ffdf26b029efde08ea794454a7b5da890108c355`.

Blindness statement: no GPT FP-KERN-002 branch content, GPT implementation,
GPT result directory, or GPT conclusion was read before this seal. The only
GPT-authored FP-KERN-002 material read is the mandated read-only task set
(task, design, plan, activation metadata, and the recorded Claude
pre-review). The frozen common input and the verified FP-KERN-001
task/code/evidence were consumed as shared read-only inputs.

## Environment

- Python 3.13.9 (`C:\Users\Admin\anaconda3\python.exe`), NumPy 2.4.6,
  SciPy 1.16.3, Ruff 0.12.0, Windows 11.
- Prior environment blocker: resolved; Bash is usable in this fixed session
  (see `execution_blocker.md`, resolution section).

## Frozen common input verification (before any analysis)

- `input/config.json` SHA-256
  `78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a`
  (804 bytes) — matches the frozen task value.
- `input/task_results.json` SHA-256
  `9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1`
  (33805703 bytes) — matches the frozen task value.
- `source_manifest.json` validated against the frozen task constants:
  source task FP-KERN-001, verified commit
  `403884ae6bde46c7c3578ae01d77422ed03faf05`, corrected seal
  `1001d23273bdf29b92d9b84a3f3956e83819da4a`, 480 records, 240 per family,
  frozen matrix (15 tasks x 4 lengths x 2 mixing x 2 gaps), seed 20260909,
  `frozen_after_creation: true`.
- Record matrix independently checked: 480 records, sequential
  `record_index`, family counts 240/240, every frozen matrix cell present
  exactly once, all record seeds 20260909.

## Verifier-first record

`verify_kernel_reuse_diagnostics.py` was written before the analyzer.

Required initial failing run (module absent), recorded verbatim:

```text
ModuleNotFoundError: No module named 'analyze_kernel_reuse_diagnostics'
```

After implementing `analyze_kernel_reuse_diagnostics.py`:

```text
kernel reuse diagnostic checks passed (14 fixtures)
```

Fixtures cover: manifest validation and tamper rejection; frozen-input hash
and 480-record matrix identity; observable-boundary oracle-field rejection
(`true_q`, `oracle_audit`, `cluster_by_state`, `realized_error`,
`exact_return`); target-action exclusion from each observable partition;
oracle-Q two-peer selection with state-index tie order; exactly ten balanced
partitions with state zero in the first group; missing-distance replacement
by exactly `1.0` forcing the tie abstention; sorted-scoring unique-minimum
selection and `partition_tie`; state-permutation equivariance of partitions
and estimates; count-weighted group pooling with positive-self inclusion,
other-state-only zero-count pooling, and empty-denominator abstention;
generator-cluster source validation; adjusted Rand index (1.0, -0.5,
relabelling invariance) and 12-assignment peer precision; metric primitives
(median, ESS, mean CI, one-sided false improvement); the complete ordered
decision table including observable precedence; predecessor replay schema;
and the fixed 16-record smoke subset identity.

Repairs during bring-up (Claude's own scope, no scientific change): two
verifier fixture defects were fixed — a hand-computed peer-precision
expectation (4/12, not 8/12) and the `pool_estimate` fixture signature.
The analyzer was not changed to fit any fixture.

## Stage 0 exact predecessor reproduction (read-only, all 480 records)

Command:

```text
python analyze_kernel_reuse_diagnostics.py --input-dir <frozen input> --mode stage0
```

Result:

```text
PASS Stage 0 exact predecessor reproduction on 480 records; classification=NOT_SUPPORTED
```

- Every serialized estimate, abstention reason, eligibility flag, distance,
  bandwidth, denominator, effective sample size, common-action count, and
  diagnostic policy of all four predecessor routes was replayed from saved
  observable fields and compared with **exact equality** (bit-for-bit float
  equality after JSON round-trip); zero mismatches across 480 records.
- All old metrics and intervals (both family screens including the secondary
  Spearman diagnostic, and all descriptive route metrics) were recomputed and
  matched the sealed predecessor `summary.json` to absolute tolerance 1e-12;
  `analysis.json` classification/decision rule match; `NOT_SUPPORTED`
  reproduced.

## Task-scoped Ruff

```text
ruff check analyze_kernel_reuse_diagnostics.py verify_kernel_reuse_diagnostics.py
All checks passed!
```

## Fixed 16-record smoke

Subset: `task_index=0`, both families, lengths 256/1024, both mixing values,
both gap bonuses — exactly 16 records (fixture-verified). The smoke ran the
full pipeline (Stage 0 replay on the subset, all three diagnostic routes,
metrics, classification) and wrote a strict-JSON bundle to a temporary
directory, which was inspected and then deleted; the formal result directory
remained absent.

```text
PASS FP-KERN-002 smoke diagnostic on 16 records; classification=NO_BORROWING_EVIDENCE
```

Smoke exercised: all three routes; `not_applicable_family` serialization of
`oracle_generator_cluster` on the current family (8 records); ordinary
`target_source_unavailable` abstention (3 observable pairs); unique-minimum
partition selection; strict JSON round-trip of every bundle file
(`NaN`/`Infinity` rejected); and the output schema. The smoke classification
is a pipeline check only and carries no scientific weight; the frozen subset
is not required to contain a tie (`partition_tie` is covered by deterministic
fixtures).

## Inherited verifier (zero-mismatch legacy regression)

```text
python verify_kernel_state_generalization.py     -> kernel state-generalization checks passed
python verify_fixed_policy_q_routes.py           -> PASS
python verify_finite_sample_theorems.py          -> PASS
python verify_visit_indexed_martingale_certificate.py -> PASS
python verify_time_uniform_mixture_certificate.py     -> PASS
python analyze_kernel_state_generalization.py --result-dir <FP-KERN-001 canonical>  (read-only)
    -> PASS FP-KERN-001 strict analysis with 480 records; classification=NOT_SUPPORTED
```

No old file was modified; input hashes were re-verified unchanged after every
run (the analyzer self-checks this and aborts otherwise).

## Seal

Implementation, verifier, Stage 0, lint, smoke, and inherited-verifier
evidence are sealed by the `[claude]` implementation/smoke commit. The sole
full 480-record diagnostic will run exactly once afterwards into the empty
`results/FP-KERN-002/claude/` directory.
