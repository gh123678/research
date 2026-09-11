# FP-ESARSA-001 — Claude main route, first (smoke) result

Status: blind smoke seal, written **before** the frozen 480-record formal run
was inspected. This document is not a scientific claim; it is the sealed
labelled-smoke evidence required by the task plan.

## Preconditions

- Branch `claude/FP-ESARSA-001`, activation commit `e2859f2`. No push, no
  main merge.
- Executable contract: `verify_fixed_policy_expected_sarsa.py` reports
  `PASS fixed-policy Expected SARSA verification (13550 checks)`.
- `ruff check` clean on `fixed_policy_expected_sarsa.py`,
  `verify_fixed_policy_expected_sarsa.py`,
  `evaluate_fixed_policy_expected_sarsa.py`,
  `analyze_fixed_policy_expected_sarsa.py`.

## Smoke commands (labelled, non-frozen matrices)

```
C:/Users/Admin/anaconda3/python.exe -B evaluate_fixed_policy_expected_sarsa.py \
  --mode smoke --tasks 3 --trajectory-lengths 256 1024 \
  --output-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-ESARSA-001/claude/smoke \
  --label p4-smoke-1

C:/Users/Admin/anaconda3/python.exe -B evaluate_fixed_policy_expected_sarsa.py \
  --mode smoke --tasks 2 --trajectory-lengths 4096 16384 \
  --output-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-ESARSA-001/claude/smoke_long \
  --label p4-smoke-long
```

Both wrote `checks.log`, `commands.log`, `config.json`, `task_results.json`,
`regression.json`, `environment.json`. The smoke matrices are proper subsets
of the frozen seed schedule; no frozen parameter was changed.

## Observed smoke behaviour

- Generator-identity regression against the frozen FP-TU-001 baseline:
  `0 mismatches` in both runs (identity fields `seed_entropy`, `spawn_key`,
  `true_pair_occupancy_min`, `true_action_gap_min`, `true_action_gap_mean`;
  tolerance `1e-12`).
- Certificate emission only when the held-out suffix visits every one of the
  24 pairs; the `mixing=0.08` sticky cells fail `heldout_pair_support_missing`
  at lengths 256/1024, exactly as the ordered reason list prescribes.
- Full held-out support at lengths 4096/16384 emits certificates on every
  route (`16/16` in the long smoke), with no oracle certificate violation, no
  residual-event violation, and no oracle value decrease.
- `expected_finite` reports `contraction_premise_satisfied=False` whenever a
  pair is unvisited in training (empirical diagonal `0.0`), while the exact
  and sampled routes use the identity writeback kernel (diagonal `1.0`).
  Premise failures are reported, never oracle-repaired.
- No `safe_update_emitted` in either smoke subset. This is an expected
  consequence of the frozen mixture radius at these counts and is **not**
  interpreted here.

## Scope limits

- Smoke subsets are not the frozen matrix and carry no empirical hypothesis-8
  weight.
- Nothing in this document was used to choose or retune a frozen parameter.
- Results remain preliminary until the frozen formal run and, later, GPT
  post-seal acceptance (or an explicit user exemption).
