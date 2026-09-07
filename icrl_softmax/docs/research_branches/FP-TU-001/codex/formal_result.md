# FP-TU-001 GPT formal result

Status: `SEALED_FORMAL_RESULT_PENDING_RECIPROCAL_VERIFICATION`.

The GPT route was already blind-sealed at
`00f7d89b09263da9338dd1e1e68f4a9787c2b8f7`.  This formal result was produced
without reading Claude's code, result, or conclusion.

## Frozen execution

Exactly one scientifically complete formal evaluation was run:

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_time_uniform_certificates.py --tasks 30 --trajectory-lengths 256 1024 4096 16384 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir results/FP-TU-001/codex
```

It exited 0 and produced 480 records.  The one formal analyzer command was:

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_time_uniform_certificates.py --result-dir results/FP-TU-001/codex --baseline-dir results/FP-MART-001/codex
```

It exited 0 with `formal=True`.  The analyzer verified all three frozen
baseline hashes before and after analysis, strict JSON, exactly seven formal
core artifacts, 480 matched records, and zero legacy leaf mismatches under
the frozen exact/numeric rules.

## Mandatory mathematical and numerical result

All 16384 counts passed finite stable inversion, conservative-root,
monotonic-radius, and stitch-bracket checks.  The largest bisection count was
42 and the maximum `q_mix/q_stitch` was 0.9959405625588716.  The maximum
mixture/legacy radius ratios were:

| Legacy trajectory length | Maximum ratio |
| ---: | ---: |
| 256 | 0.9573573993365391 |
| 1024 | 0.9263355819163681 |
| 4096 | 0.8975200574107253 |
| 16384 | 0.8793886652750217 |

Every comparison was at most one, and 21,760 horizon/count comparisons were
strictly smaller.  Across all observed formal residual groups the ratio
ranged from 0.7379884021918311 to 0.9492067139237118, with mean
0.8508755029787135.

Every old emitted route remained emitted, every old rejected route retained
the same ordered reasons, and every emitted total bound was nonincreasing.
Mean old-minus-new total-bound reductions among emitted records were:

| Route | Emitted | Mean reduction | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Direct exact | 345 | 5.867580794080362 | 1.1586208695694866 | 23.60475469209831 |
| Direct softmax | 312 | 36.33231682144782 | 1.7697373715405718 | 285.31873222610557 |
| V-first exact | 345 | 2.147345778082567 | 0.48993709483991843 | 8.297685254604062 |
| V-first softmax | 345 | 2.1574473975175636 | 0.4920543845329135 | 8.37192560979922 |

## Emission and descriptive usefulness

The exact-route emissions were unchanged at 3/120, 102/120, 120/120, and
120/120 for lengths 256, 1024, 4096, and 16384: exactly
2.5%, 85%, 100%, and 100%.  Direct-softmax emissions were 3, 74, 115, and 120;
V-first-softmax emissions matched the exact counts.

Primary `<B` usefulness counts were:

| Route | 256 | 1024 | 4096 | 16384 |
| --- | ---: | ---: | ---: | ---: |
| Direct exact | 0 | 0 | 0 | 0 |
| Direct softmax | 0 | 0 | 0 | 0 |
| V-first exact | 0 | 0 | 0 | 101 |
| V-first softmax | 0 | 0 | 0 | 58 |

Descriptive `<2B` counts were respectively 76 at length 16384 for Direct
exact; 91 and 120 at lengths 4096 and 16384 for V-first exact; and 70 and 120
for V-first softmax.  Direct softmax had none.  These are secondary empirical
outcomes and are not used to justify the probability theorem.

The oracle audit enumerated zero emitted-route violations and zero
fixed-target residual violations.  It is structurally separate and marked
`used_as_theorem_evidence=false`.

## Formal artifact hashes

- `checks.log`:
  `94f87599448d5e03f830621f9ee179d6ab2afcce889ca3fa89b83bb4a62265dd`.
- `commands.log`:
  `9c718f51904bae4d41c11f008f7571874779205870290c66699cf30cde3b6b41`.
- `config.json`:
  `43dcb96b0f6f95e76f1c0b484d6375e3727dbb5609b16a8952a69e2ac0dddf3a`.
- `environment.json`:
  `35277785974e2999936fdb58ecbc260107d4a6a8e4a2e6e4244ce37d8a27662d`.
- `regression.json`:
  `7c3d4d60734da4684e96cca604a7ebe443e87c0de14cc16b7440cf62d8556789`.
- `summary.json`:
  `565fc4d261a938d13350984bb97242e79fba42f014d11f807a18942517f4444f`.
- `task_results.json`:
  `0e5eab39bf49894832c5ebcdd6f7b70c453fff6b9600b234617889f8f9fa79be`.

## Failures and anomalies

- The preregistered test-first run failed as expected before the module
  existed; this is recorded in `first_result.md`.
- After formal analysis, one read-only inline metric-display command had a
  PowerShell quoting error and exited with a Python `SyntaxError`.  It did not
  alter any artifact.  A native read-only PowerShell extraction then produced
  the metrics recorded above.
- No formal evaluation or analyzer failure occurred, and the formal matrix
  was not rerun.

## Acceptance assessment

Acceptance items 1--19 pass on the GPT route: proof mapping, risk allocation,
random-count semantics, stitch validity, no-oracle inputs, exhaustive numeric
checks, radius dominance, unchanged recurrences, ordered failures, additive
namespace, strict JSON, six verifiers and Ruff, smoke-before-formal ordering,
the 480-record identity, zero legacy mismatch, frozen emissions, separated
oracle audit, transition-variance exclusion, and complete route evidence.

Item 20 remains pending: Claude's independently sealed formal route and both
reciprocal reproductions must pass before synthesis or `VERIFIED`.  Therefore
this document reports a strong positive GPT result, not a final project
conclusion.
