# FP-KERN-002 Codex blind formal result

Date: 2026-09-09

Branch: `codex/FP-KERN-002`

Frozen implementation and smoke commit:
`f37b9730aea4ba692b854dcfb89f8f3d17faa34e`

Common execution-start commit:
`ffdf26b029efde08ea794454a7b5da890108c355`

Common input manifest SHA-256:
`670648f7a2761d919f58b25881db45dc6d9d49d4fec25e307c6fe72bf8b31966`

## Blind execution

The frozen analyzer was run once on all 480 reused records. The formal output
directory did not exist before this run. No Claude implementation, output, or
conclusion had been read.

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_reuse_diagnostics.py --mode formal --input-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-KERN-002\input --output-dir results\FP-KERN-002\codex\formal
```

Result: `PASS`, 480 records, classification `NO_BORROWING_EVIDENCE`.

The strict bundle verifier, the task-specific verifier, task-scoped Ruff, and
the inherited kernel verifier all passed after the formal run. Stage 0 retained
zero predecessor-route mismatches and reproduced predecessor classification
`NOT_SUPPORTED`.

## Frozen-gate results

All four ordered decision inputs are false:

- `peer_headroom = false`;
- `generator_structure_useful = false`;
- `observable_structure_useful = false`;
- `observable_current_family_pass = false`.

The routes almost always found data to borrow, so lack of coverage is not the
failure mode:

- true-Q nearest-two coverage was 100% in both families;
- observable balanced-cluster coverage was 99.45% in the current family and
  96.60% in the hidden family;
- generator-cluster coverage was 94.89% in the hidden family.

However, no route passed all five frozen screens.

### True-Q nearest-two upper bound

Using true Q only to rank the two closest peers strongly improved zero-count
RMSE and top-action selection, but worsened RMSE when the target action already
had 1--4 observations:

- current family: zero-count RMSE improved 47.29%, while sparse RMSE worsened
  49.45%; top-action accuracy improved 26.13 percentage points;
- hidden family: zero-count RMSE improved 39.37%, while sparse RMSE worsened
  33.42%; top-action accuracy improved 23.82 percentage points.

Consequently the oracle upper-bound screen failed in both families.

### True generator cluster

On the hidden family, pooling within the true generator cluster covered 94.89%
of eligible zero-count actions, but zero-count RMSE worsened 0.46% and sparse
RMSE worsened 123.36%. Top-action accuracy improved 11.99 percentage points,
which was insufficient to offset the RMSE failures. The route failed its
screen.

### Observable balanced cluster

For the current family, zero-count RMSE improved 12.88%, but its paired
confidence interval crossed zero; sparse RMSE worsened 122.86%, and top-action
accuracy improved only 1.78 percentage points with a confidence interval that
also crossed zero.

For the hidden family, zero-count RMSE improved only 5.32% with a confidence
interval crossing zero; sparse RMSE worsened 97.54%. Top-action accuracy
improved 6.34 percentage points with a positive confidence interval, but the
route still failed the zero-count and sparse-RMSE screens.

The observable partition did not fail because of ties: it emitted all 960
record-action partitions with zero `partition_tie` cases. Its mean adjusted
Rand index against the hidden generator partition was 0.1516 (95% interval
0.1217 to 0.1815), and same-cluster peer precision was 49.10%.

## Interpretation and boundary

This result falsifies the practical claim tested here: unconditional
count-weighted reuse of the same action across states, using either the frozen
observable partition or either oracle source definition, does not satisfy the
five-screen policy-improvement diagnostic on this corpus.

It does not establish that every kernel or cross-state generalization method is
impossible. In particular, it does not test count-aware shrinkage, borrowing
only for zero-count actions, learned state representations, cross-fitting, or a
new estimator that retains the target-state estimate when local evidence is
available. Those are separate hypotheses and would require a new task. The
strong oracle gains on zero-count actions, combined with oracle harm in the
1--4 bin, point more directly to an estimator/gating problem than to simple
action inaccessibility.

## Artifact identity

Formal output: `results/FP-KERN-002/codex/formal/`

- `diagnostic_records.json` SHA-256:
  `90ae246bd4a970d9974e2841f3d6011851e4ca022b57c22bad555cc6ffb92a5f`;
- `summary.json` SHA-256:
  `7b52e13fdfc3375a3ab2f345bc64f9a680aaf5ac894e6a0739b9bb45d065b269`;
- `analysis.json` SHA-256:
  `dbb66d5547a0f3e60b54b2c5689aad89785c3d5d5b0b06a6f475b486663358ad`;
- `artifact_hashes.json` SHA-256:
  `88f3323822b655feded7785b74c234356e46d68a5d3037f8e625d313accf2722`.

This is a Codex blind result only. Under the repository governance rules it is
an initial result until the independent Claude route and reciprocal executable
verification are complete.
