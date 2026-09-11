# FP-KERN-001 Codex corrective formal result

Date: 2026-09-09

User ruling and repair seal:
`7fc2d53837b74f3af260583d0b1735391fcc736f`

Corrective implementation/smoke seal:
`5af3dc6134a13779908e87558941fa82ed0829eb`

Original GPT formal seal (preserved under `results/FP-KERN-001/codex/original_formal/`):
`640a3f8fb2d0f41b96eef8d9bb76fc5ef2e9b93b`

Status: corrective formal route `PASS` mechanically; frozen scientific
classification `NOT_SUPPORTED`; Claude final verification pending.

## Authorized exception and exact scope

The user explicitly authorized one corrective GPT formal rerun after reciprocal
disclosure identified three literal deviations from the frozen definitions.
The repair changed only:

1. positive target-count pairs retain self distance zero and self weight one,
   even when the target state's signature has fewer than two non-target
   coordinates; cross-state comparisons still require two common actions;
2. false improvement is the one-sided event `estimated difference > 0` and
   `true difference <= 0`; and
3. the secondary Spearman diagnostic is computed within each record/action and
   then averaged over finite correlations.

The denominator error label was aligned with the already-frozen ordered
abstention contract during the same gate refactor. No kernel, bandwidth,
cross-state support rule, environment generator, random stream, seed, formal
matrix, metric threshold, or decision rule was changed. The original formal
artifacts were moved byte-for-byte to `codex/original_formal/`; no old output
was overwritten.

## Corrective formal run

The sole authorized corrective formal command, run once after commit
`5af3dc6134a13779908e87558941fa82ed0829eb`, was:

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_kernel_state_generalization.py --mode formal --tasks 15 --families current_unstructured hidden_cluster --trajectory-lengths 256 1024 4096 16384 --mixing 0.08 0.50 --gap-bonuses 0.0 0.50 --n-states 6 --n-actions 4 --pi-min 0.05 --gamma 0.70 --alpha 0.65 --iterations 160 --seed 20260909 --output-dir results/FP-KERN-001/codex/corrected_formal
```

It exited zero and produced exactly 480 records. The seven corrected artifacts
were then promoted into the canonical `results/FP-KERN-001/codex/` directory;
the unchanged first-run artifacts remain in `codex/original_formal/`.

The post-run strict analyzer was run twice: once on the isolated corrected
directory with `--write-results`, and once read-only after promotion:

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_state_generalization.py --result-dir results/FP-KERN-001/codex/corrected_formal --write-results
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_state_generalization.py --result-dir results/FP-KERN-001/codex
```

Both runs independently regenerated all 480 records with zero reconstruction
mismatches. The second run did not modify the promoted artifacts.

## Corrected frozen screen

| Criterion | Current unstructured | Hidden cluster |
| --- | ---: | ---: |
| Signature-eligible zero-count coverage | 100.0% (183/183), pass | 100.0% (235/235), pass |
| Zero-count RMSE relative improvement vs action-only | 7.25%, fail | 1.47%, fail |
| Zero-count paired RMSE improvement 95% CI | [0.0117, 0.1275] | [-0.0313, 0.0546] |
| 1--4-count RMSE relative improvement vs local | -142.90%, fail | -118.39%, fail |
| 1--4-count paired RMSE improvement 95% CI | [-0.5237, -0.3668] | [-0.4845, -0.2902] |
| Sparse-state top-action improvement vs action-only | 2.64 pp, fail | 3.71 pp, fail |
| Top-action paired improvement 95% CI | [-0.47, 5.76] pp | [0.32, 7.10] pp |
| False-improvement rate | 20.51% vs 21.61%, pass | 17.51% vs 18.80%, pass |
| Family screen | FAIL | FAIL |

The corrective result remains `NOT_SUPPORTED`: both families pass coverage and
false-improvement control but fail the required zero-count RMSE, `1-4`-count
RMSE, and sparse top-action criteria. The zero-count estimates are unchanged
by construction. The self-weight correction improves sparse-positive estimates
slightly, but not enough to reach the frozen thresholds. The secondary
record/action Spearman diagnostic is positive in both families (current mean
0.2809, 95% CI [0.2568, 0.3050]; hidden mean 0.3163, 95% CI [0.2927,
0.3399]) and remains non-screen evidence.

## Verification after promotion

All of the following exited zero after the corrective formal analysis and
promotion:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_kernel_state_generalization.py
C:\Users\Admin\anaconda3\python.exe -B verify_fixed_policy_q_routes.py
C:\Users\Admin\anaconda3\python.exe -B verify_finite_sample_theorems.py
C:\Users\Admin\anaconda3\python.exe -B verify_visit_indexed_martingale_certificate.py
C:\Users\Admin\anaconda3\python.exe -B verify_time_uniform_mixture_certificate.py
C:\Users\Admin\anaconda3\Scripts\ruff.exe check kernel_state_generalization.py kernel_generalization_mdps.py verify_kernel_state_generalization.py evaluate_kernel_state_generalization.py analyze_kernel_state_generalization.py
```

The new verifier includes regression checks for all three corrected
definitions. The old seven artifacts in `codex/original_formal/` were
re-hashed after the move and remain byte-identical to the original seal.

## Corrected canonical artifact hashes

- `analysis.json`: `5fef6fd0f615b6301d274544af9c18e54fe04634cd6d5cee16ef929bf79d32a7`;
- `checks.log`: `a8666ad610f4fcc969dd589b0b2a8dbef94276e14268fea58afc824dd53780ed`;
- `commands.log`: `2c53a81f617d244a0965b48d4fdabdc7eafea28e7a973766750a1343558f37ee`;
- `config.json`: `78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a`;
- `environment.json`: `53602b573c9ae80948e5d0403d4d314a6f381e9ddda01beb32992e1a17f830d4`;
- `summary.json`: `0c5d1b039720acb3b3710846f1e490589a07f5eb7fd6f2ed859ba3520515b640`;
- `task_results.json`: `9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1`.

## Limitations and conclusion

This remains an empirical feasibility study only; it supplies no safety,
nondegradation, or policy-improvement theorem. The two independent routes now
agree under the literal frozen definitions: the observable signature has
positive similarity signal and high zero-count coverage, but the proposed
kernel estimator is not competitive in the required sparse regimes. The
corrective GPT route is ready for Claude's final reciprocal verification.
