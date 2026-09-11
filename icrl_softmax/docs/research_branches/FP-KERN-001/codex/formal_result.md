# FP-KERN-001 Codex blind formal result

Date: 2026-09-09

Implementation seal:
`3f57ce5019a6db150a91ea153fa9fcc834f0f9e8`

Common route baseline:
`28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c`

Status: Codex formal route `PASS` mechanically; frozen scientific
classification `NOT_SUPPORTED`; independent Claude result and reciprocal
verification pending.

The Codex formal evaluator was run exactly once after the implementation and
mandatory smoke had been blind-sealed. No Claude implementation, result, or
conclusion was read before this record was written. The smoke artifacts were
moved unchanged to `results/FP-KERN-001/codex_smoke/` so that the canonical
formal directory was empty before the one allowed formal run.

## Frozen formal run

Working directory:
`C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-KERN-001\codex_worktree\icrl_softmax`

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_kernel_state_generalization.py --mode formal --tasks 15 --families current_unstructured hidden_cluster --trajectory-lengths 256 1024 4096 16384 --mixing 0.08 0.50 --gap-bonuses 0.0 0.50 --n-states 6 --n-actions 4 --pi-min 0.05 --gamma 0.70 --alpha 0.65 --iterations 160 --seed 20260909 --output-dir results/FP-KERN-001/codex
```

The evaluator exited zero and wrote exactly 480 records. The strict analyzer
was then run once:

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_state_generalization.py --result-dir results/FP-KERN-001/codex --write-results
```

It independently regenerated all 480 records and reported zero reconstruction
mismatches.

## Frozen feasibility screen

| Criterion | Current unstructured | Hidden cluster |
| --- | ---: | ---: |
| Signature-eligible zero-count coverage | 100.0% (183/183), pass | 100.0% (235/235), pass |
| Zero-count RMSE relative improvement vs action-only | 7.25%, fail | 1.47%, fail |
| Zero-count paired RMSE improvement 95% CI | [0.0117, 0.1275] | [-0.0313, 0.0546] |
| 1--4-count RMSE relative improvement vs local | -154.41%, fail | -121.51%, fail |
| 1--4-count paired RMSE improvement 95% CI | [-0.5542, -0.3867] | [-0.4981, -0.3021] |
| Sparse-state top-action improvement vs action-only | 2.12 pp, fail | 2.65 pp, fail |
| Top-action paired improvement 95% CI | [-1.17, 5.42] pp | [-0.66, 5.96] pp |
| False-improvement rate | 38.13% vs 39.58%, pass | 34.39% vs 35.99%, pass |
| Family screen | FAIL | FAIL |

Both families fail the frozen five-part screen, so the decision rule returns
`NOT_SUPPORTED`. In particular, the primary kernel reconstructed all eligible
unseen actions and did not worsen the false-improvement diagnostic, but it did
not reach the required 10% zero-count RMSE gain and was substantially worse
than the local estimator on 1--4-count pairs. The hidden-structure positive
control also failed, so this result is stronger than merely saying that the
current unstructured generator lacks exploitable similarity.

The secondary observable-geometry diagnostic was positive in both families:
mean per-record Spearman correlation was 0.2170 with 95% CI
[0.1788, 0.2552] for current unstructured and 0.2355 with 95% CI
[0.1974, 0.2737] for hidden cluster. This correlation was not sufficient to
make the frozen kernel estimator competitive on sparse pairs.

## Verification

All of the following exited zero after the formal analysis:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_kernel_state_generalization.py
C:\Users\Admin\anaconda3\python.exe -B verify_fixed_policy_q_routes.py
C:\Users\Admin\anaconda3\python.exe -B verify_finite_sample_theorems.py
C:\Users\Admin\anaconda3\python.exe -B verify_visit_indexed_martingale_certificate.py
C:\Users\Admin\anaconda3\python.exe -B verify_time_uniform_mixture_certificate.py
ruff check kernel_state_generalization.py kernel_generalization_mdps.py verify_kernel_state_generalization.py evaluate_kernel_state_generalization.py analyze_kernel_state_generalization.py
```

## Formal artifact hashes

- `analysis.json`: `776656e2b1dd86e316ca7673d33b4f2f29fb31b511ad9d011ed6865e13cf5ecc`;
- `checks.log`: `a8666ad610f4fcc969dd589b0b2a8dbef94276e14268fea58afc824dd53780ed`;
- `commands.log`: `0a0887a8353e41d9250fda426736b36fb2fb4076dffd3d9959c1a7ad698886fd`;
- `config.json`: `78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a`;
- `environment.json`: `a2000e7d6e8ee47ec64bdb69ba4a3bac8e3da99b441d0896a70ac9ee43843f1b`;
- `summary.json`: `381b230e7d7d49f742109e041bb87ef5434be067cbd6d641b2b1a28b49d9fd96`;
- `task_results.json`: `d5392da064612aa6c62559d42cc9fa8d6d31960dc35cd5a44bbc57f59235793e`.

These are preliminary Codex-route findings until the independent Claude route
is blind-sealed and reciprocal verification is complete.
