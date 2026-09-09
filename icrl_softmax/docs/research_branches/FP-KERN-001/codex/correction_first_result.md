# FP-KERN-001 Codex corrective implementation and smoke seal

Date: 2026-09-09

User-ruling baseline:
`7fc2d53837b74f3af260583d0b1735391fcc736f`

Original GPT formal seal:
`640a3f8fb2d0f41b96eef8d9bb76fc5ef2e9b93b`

Status: user-authorized corrective implementation and smoke `PASS`;
corrective formal matrix not yet run.

## Authorized repair

No kernel, bandwidth, cross-state support rule, environment, random stream,
seed, matrix, threshold, or decision rule changed. The repair is restricted to
the post-verification discrepancies:

1. the target state now has distance zero and weight one whenever its target
   count is positive, even if its own signature has fewer than two observed
   non-target actions; cross-state distances still require two common observed
   non-target actions;
2. false improvement now counts only an estimated positive action difference
   whose true difference is nonpositive; and
3. the secondary Spearman diagnostic now computes one correlation within each
   record/action before averaging finite correlations.

While restructuring the self/source gates, nonpositive kernel denominators are
also labelled with the already-frozen `kernel_denominator_invalid` reason
instead of `target_source_unavailable`; this only restores the existing
ordered error contract and changes no route estimate or scientific constant.

## Test-first evidence

Three new deterministic regression checks were added before production code:

- sparse-positive self-only reduction when the target state lacks two
  non-target signature coordinates;
- one-sided false-improvement counting, including a reversed-order estimate
  that must not count; and
- separate positive and negative record/action Spearman correlations that
  would be obscured by pooling actions.

The first run failed before executing the tests with:

```text
ImportError: cannot import name '_false_improvement_counts' from
'analyze_kernel_state_generalization'
```

After the restricted implementation changes, the new verifier and task-scoped
Ruff both pass. All four inherited verifiers also pass.

## Corrective smoke

The 16-record smoke command was:

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_kernel_state_generalization.py --mode smoke --tasks 1 --families current_unstructured hidden_cluster --trajectory-lengths 256 1024 --mixing 0.08 0.50 --gap-bonuses 0.0 0.50 --n-states 6 --n-actions 4 --pi-min 0.05 --gamma 0.70 --alpha 0.65 --iterations 160 --seed 20260909 --output-dir results/FP-KERN-001/codex/correction_smoke
```

The evaluator produced exactly 16 records. The strict analyzer command

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_state_generalization.py --result-dir results/FP-KERN-001/codex/correction_smoke --write-results
```

regenerated all 16 records with zero mismatch. Its smoke classification was
`NOT_SUPPORTED`, which is not scientific evidence and caused no tuning or
further implementation change.

## Corrective smoke SHA-256

- `analysis.json`: `d2f92154d15bc6a43e5df06d45d7390326da559bba4805d4b271841252d15837`;
- `checks.log`: `c010d693fb7f20130ce7bc9cb4a9d746b976d0b5a050c6bda7a765a6cd865a68`;
- `commands.log`: `eec6cb36dc6dea57a7cde9878bc4f11f0f75d4ab598673fb61ba31cbf5f65cb4`;
- `config.json`: `91cee84019df3234355da32a8cea634102addded88f36dc4571052df47ca9c41`;
- `environment.json`: `056632654197415cbec07ce5ef5c59d0d50a7ebf6705ad012032b54c4047df48`;
- `summary.json`: `bdc1a6ceb009f5924bb69a8bf9f82f88bf82b7fc1374dbb0cb97ec262f7acbe7`;
- `task_results.json`: `ec0297d2736c4df980d78938a32e4966f3e864571be101268ce3b437cfb73d33`.

The original GPT formal artifacts remain unchanged in
`results/FP-KERN-001/codex/`. The authorized corrective formal output will be
written to a new empty subdirectory only after this implementation/smoke seal
is committed.
