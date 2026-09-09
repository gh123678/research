# FP-KERN-001 Codex blind first result

Date: 2026-09-09

Common route baseline:
`28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c`

Status: implementation and mandatory smoke `PASS`; formal matrix not run.

## Test-first record

The new verifier was created before either implementation module. Its first
execution failed as required with
`ModuleNotFoundError: kernel_generalization_mdps`. The pure kernel module and
hidden-cluster generator were then implemented until the verifier passed.

The final pre-seal implementation contains only the five task-authorized files:

- `kernel_state_generalization.py`;
- `kernel_generalization_mdps.py`;
- `verify_kernel_state_generalization.py`;
- `evaluate_kernel_state_generalization.py`;
- `analyze_kernel_state_generalization.py`.

No existing code, old result, manuscript, `FP-ADV-001` artifact, task formula,
route, matrix, threshold, or bandwidth was changed.

## Validation commands

Working directory:
`C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-KERN-001\codex_worktree\icrl_softmax`

The following commands exited zero:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_kernel_state_generalization.py
C:\Users\Admin\anaconda3\Scripts\ruff.exe check kernel_state_generalization.py kernel_generalization_mdps.py verify_kernel_state_generalization.py evaluate_kernel_state_generalization.py analyze_kernel_state_generalization.py
C:\Users\Admin\anaconda3\python.exe -B verify_fixed_policy_q_routes.py
C:\Users\Admin\anaconda3\python.exe -B verify_finite_sample_theorems.py
C:\Users\Admin\anaconda3\python.exe -B verify_visit_indexed_martingale_certificate.py
C:\Users\Admin\anaconda3\python.exe -B verify_time_uniform_mixture_certificate.py
```

The new verifier checks target-action exclusion, zero-count recovery,
same-action abstention, positive/even median behavior, ESS arithmetic, kernel
weight ordering, self-only reduction, state-label permutation equivariance,
oracle-key rejection, malformed inputs, degenerate-bandwidth abstention,
hidden-cluster determinism, cluster balance, probability rows, and reward
bounds. All checks pass.

Environment:

- Python 3.13.9;
- NumPy 2.4.6;
- SciPy 1.16.3;
- CPU-only Windows execution.

## Mandatory smoke

The 16-record smoke command was:

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_kernel_state_generalization.py --mode smoke --tasks 1 --families current_unstructured hidden_cluster --trajectory-lengths 256 1024 --mixing 0.08 0.50 --gap-bonuses 0.0 0.50 --n-states 6 --n-actions 4 --pi-min 0.05 --gamma 0.70 --alpha 0.65 --iterations 160 --seed 20260909 --output-dir results/FP-KERN-001/codex/smoke
```

It generated exactly `2 * 1 * 2 * 2 * 2 = 16` records. The strict analyzer
command

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_state_generalization.py --result-dir results/FP-KERN-001/codex/smoke --write-results
```

reconstructed all 16 records, all three seed streams, observable aggregates,
route outputs, hidden generator audit, true Q/V, and diagnostic policies with
zero mismatch. Repeating the evaluator against the nonempty smoke directory
failed before writing, as required.

Smoke classification was `NOT_SUPPORTED`. This is not formal evidence: only
eight records per family were present, multiple count-bin intervals were
underpowered, and the contract forbids changing the method in response. No
formula, threshold, environment, seed, route, or implementation was tuned.

## Smoke artifact hashes

- `analysis.json`: `d652e6b38297520b6cc3a6f01a6be317dfd8406f7c3f7731fb9ce0f42a0f02f7`;
- `checks.log`: `c010d693fb7f20130ce7bc9cb4a9d746b976d0b5a050c6bda7a765a6cd865a68`;
- `commands.log`: `111952e2fd7a3b2901a9fab99230e57329d1b55a65a563711ceb728540ce4f3c`;
- `config.json`: `91cee84019df3234355da32a8cea634102addded88f36dc4571052df47ca9c41`;
- `environment.json`: `41eb201274d73a85271e00367ac45fbbc3bed3198b9e57f5a28481821c51550a`;
- `summary.json`: `0e0920b5473f0651e79bd9f9fbcd9cefdbe085aa1749326b5e764df39a7f949b`;
- `task_results.json`: `e4119cc4c514abd02c07c05aae49895c3bd9531ff95fa635875ffa9688f04d97`.

The hash comparison is case-insensitive; all files were re-read from disk.

## Pre-formal assessment

Acceptance items 1--10 and the implementation parts of 12--15 pass in the
smoke scope. Formal record count, final screen, independent route, blind formal
seal, and reciprocal verification remain untested. No scientific feasibility
claim is made at this stage.
