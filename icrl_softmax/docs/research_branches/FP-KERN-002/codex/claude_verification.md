# FP-KERN-002 GPT verification of the Claude route

Date: 2026-09-09

Verifier: GPT on `codex/FP-KERN-002`

Claude blind implementation/smoke seal:
`a39323c8011acebfb651c8431d8598e1d1aee244`

Claude blind formal-result seal:
`718d77053801c6f9e3dd958513b7a06918a5e274`

Common execution-start commit:
`ffdf26b029efde08ea794454a7b5da890108c355`

## Scope and provenance

GPT did not read the Claude implementation or result before both routes had
blind formal-result seals. Claude's two commits were written by the principal
account from the exact path lists prepared by Claude because Claude's sandbox
could not create the Git worktree `index.lock`; the first content identity was
checked by Claude before its sole formal run, and the second commit contains
only its final report and independent reconstruction script. The permission
event is documented in the Claude evidence directory and did not alter source,
method, thresholds, outputs, or run count.

## Executable checks

GPT ran the following against the sealed Claude branch and host result bundle:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_kernel_reuse_diagnostics.py
C:\Users\Admin\anaconda3\python.exe -m ruff check analyze_kernel_reuse_diagnostics.py verify_kernel_reuse_diagnostics.py docs\research_branches\FP-KERN-002\claude\formal_output_check.py
C:\Users\Admin\anaconda3\python.exe -B docs\research_branches\FP-KERN-002\claude\formal_output_check.py
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_reuse_diagnostics.py --input-dir <common input> --result-dir <sealed Claude result> --mode stage0
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_reuse_diagnostics.py --input-dir <common input> --result-dir <sealed Claude result> --mode full
C:\Users\Admin\anaconda3\python.exe -B verify_kernel_state_generalization.py
C:\Users\Admin\anaconda3\python.exe -B verify_fixed_policy_q_routes.py
C:\Users\Admin\anaconda3\python.exe -B verify_finite_sample_theorems.py
C:\Users\Admin\anaconda3\python.exe -B verify_visit_indexed_martingale_certificate.py
C:\Users\Admin\anaconda3\python.exe -B verify_time_uniform_mixture_certificate.py
```

All commands passed. In particular:

- 14 verifier fixtures passed;
- task-scoped Ruff passed;
- Stage 0 exactly replayed 480 predecessor records and reproduced
  `NOT_SUPPORTED`;
- the full read-only recomputation reproduced 480 records and
  `NO_BORROWING_EVIDENCE`;
- the analyzer-independent `formal_output_check.py` passed all 506 check
  groups, reconstructing record identities, estimates, sources, peers,
  partitions, metrics, intervals, abstentions, screens, secondary diagnostics,
  and the ordered classification;
- all five inherited verifiers passed.

GPT independently recalculated SHA-256 for every entry declared in the Claude
`output_hashes.json`; all eight matched. The common input remained frozen at:

- `config.json`:
  `78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a`;
- `task_results.json`:
  `9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1`;
- `source_manifest.json`:
  `670648f7a2761d919f58b25881db45dc6d9d49d4fec25e307c6fe72bf8b31966`.

## Cross-route reconciliation

The independently implemented routes agree exactly, to displayed full
precision, on:

- all four false decision inputs and final classification
  `NO_BORROWING_EVIDENCE`;
- every family/route coverage denominator and numerator;
- every zero-count and 1--4-count RMSE mean, relative improvement, paired
  mean, confidence interval, and empty-record count;
- every top-action and false-improvement quantity;
- every five-item criterion vector and screen result;
- 960 emitted observable partitions, zero formal partition ties, ARI mean
  `0.1516203703703704`, interval
  `[0.1217019663212322, 0.1815387744195086]`, and peer precision
  `5656 / 11520 = 0.4909722222222222`;
- both unavailable oracle-benefit recovery ratios and their component means.

The bundle hashes are intentionally not identical across routes because the
independent implementations use different evidence schemas and field names
(for example `route_mean` versus `primary_mean`, and different manifest
wrappers). These serialization differences do not change any reconstructed
scientific value, screen, or classification.

## Acceptance judgment

Claude's route satisfies the frozen method, source, isolation, single-run,
reconstruction, hashing, and reporting requirements. No ordinary
implementation defect, result-selected change, unresolved anomaly, or task
objection was found. The sole remaining task-level requirement after this
report is Claude's symmetric executable verification of the GPT route.

PASS
