# FP-KERN-002 Codex verifier-first and smoke seal

Date: 2026-09-09

Branch: `codex/FP-KERN-002`

Common execution-start commit:
`ffdf26b029efde08ea794454a7b5da890108c355`

Common input manifest SHA-256:
`670648f7a2761d919f58b25881db45dc6d9d49d4fec25e307c6fe72bf8b31966`

## Test-first evidence

`verify_kernel_reuse_diagnostics.py` was created before the analyzer. Its
first run failed as required with:

```text
ModuleNotFoundError: No module named 'analyze_kernel_reuse_diagnostics'
```

After implementation, the following passed:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_kernel_reuse_diagnostics.py --input-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-KERN-002\input
C:\Users\Admin\anaconda3\Scripts\ruff.exe check analyze_kernel_reuse_diagnostics.py verify_kernel_reuse_diagnostics.py
```

The verifier covers the common input, ten balanced partitions, target-action
exclusion, prohibited oracle inputs, missing distance `1.0`, unique-minimum
and tie abstention, two oracle source constructions, zero-count source use,
ARI, one-sided false improvement, all decision-table branches, and a unique-
optimum state-permutation fixture.

## Stage 0

The first Stage 0 attempt reconstructed every one of the 480 serialized old
route records exactly, then stopped on the predecessor summary hash. Direct
strict-JSON comparison showed the reconstructed summary object and frozen
`summary.json` object were exactly equal. The difference was serialization:
the frozen Windows text file uses CRLF, whereas canonical logical JSON uses LF.

The repair records both identities separately and compares the reconstructed
logical payload against the canonical value:

- frozen predecessor summary file SHA-256:
  `0c5d1b039720acb3b3710846f1e490589a07f5eb7fd6f2ed859ba3520515b640`;
- canonical logical summary SHA-256:
  `e6327594ccca29e02eeb718849be09d7445a56f1ccbc2d53ee8129d79d2da9d0`.

No source, route, metric, threshold, or result changed. The corrected Stage 0
command passed:

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_reuse_diagnostics.py --mode stage0 --input-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-KERN-002\input
```

Result: 480 records, zero route mismatches, predecessor classification
`NOT_SUPPORTED`.

## Fixed smoke

The sole fixed 16-record smoke output was written under
`results/FP-KERN-002/codex/smoke/` using task index zero, both families,
lengths 256 and 1024, both mixing settings, and both gap bonuses.

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_reuse_diagnostics.py --mode smoke --input-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-KERN-002\input --output-dir results\FP-KERN-002\codex\smoke
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_reuse_diagnostics.py --mode verify --input-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-KERN-002\input --output-dir results\FP-KERN-002\codex\smoke
C:\Users\Admin\anaconda3\python.exe -B verify_kernel_state_generalization.py
```

All commands passed. The smoke classification is execution-only and has no
scientific standing. No full 480-record FP-KERN-002 diagnostic has run yet,
and no Claude implementation or result has been read.

## Scope judgment

The implementation modifies no predecessor file, creates no trajectory, and
keeps true Q and generator labels outside the observable route boundary. The
CRLF repair is an ordinary provenance-label correction and does not change the
frozen task. The route is ready for its blind implementation/smoke commit and
then the sole full reused-record diagnostic.
