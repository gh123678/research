# FP-KERN-001 Claude route: formal result seal

Date: 2026-09-09. Branch: `claude/FP-KERN-001`. Task version 1.0 (`ACTIVE`).

- Verified scientific baseline: `c579047950dfabb2600020cd2e53dd24b3e39c84`.
- Common route execution-start commit: `28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c`.
- Claude blind implementation/smoke seal: `428af58c9511fd0c4438cbc1c97fceb0fdc632d6`
  (git HEAD at the formal run, recorded in `environment.json`).

This seal was written before inspecting any GPT route artifact. Blindness was
maintained throughout.

## Formal run (exactly one, no rerun, no tuning)

```text
python -B evaluate_kernel_state_generalization.py --formal --tasks 15 \
  --families current_unstructured hidden_cluster \
  --trajectory-lengths 256 1024 4096 16384 --mixing 0.08 0.50 \
  --gap-bonuses 0.0 0.50 --n-states 6 --n-actions 4 --pi-min 0.05 \
  --gamma 0.70 --alpha 0.65 --iterations 160 --seed 20260909 \
  --output-dir results/FP-KERN-001/claude
python -B analyze_kernel_state_generalization.py --result-dir results/FP-KERN-001/claude
```

- 480/480 records written; 0 record-level failures; strict JSON only.
- The canonical formal directory `results/FP-KERN-001/claude/` was verified
  absent/empty before the run; smoke output remained unchanged in
  `results/FP-KERN-001/claude_smoke/`.
- Evaluator preflight (in `checks.log`) re-ran the four inherited verifiers
  and the new verifier; all passed.

## Mechanical status after the formal run

All repeated after the formal run and analyzer:

```text
verify_fixed_policy_q_routes.py                  PASS
verify_finite_sample_theorems.py                 PASS
verify_visit_indexed_martingale_certificate.py   PASS
verify_time_uniform_mixture_certificate.py       PASS
verify_kernel_state_generalization.py            PASS (17 checks)
ruff check (five task files)                     All checks passed
analyzer                                        PASS, records=480
```

Analyzer integrity (all passed, zero failures):

- exact cell membership (2 families x 15 tasks x 4 lengths x 2 mixing x 2
  gap) and exactly 480 records with frozen route names;
- full per-record regeneration from serialized seed provenance (MDP/policy,
  three streams, aggregates, route outputs, oracle audit) with exact field
  comparison;
- bitwise V-first reconstruction from stored value aggregates; exact
  signature/target Y-sum reconstruction; exact route re-derivation from the
  pure module; diagnostic-policy status reconstruction;
- summary rows recomputed and matched; oracle separation of observable
  inputs and route outputs.

## Frozen screen outcomes

### Hidden-cluster family (positive control) — screen FAILED

| item | threshold | observed | pass |
|---|---|---|---|
| 1 zero-count coverage of signature-eligible pairs | >= 50% | 224/225 = 99.56% | yes |
| 2 zero-count RMSE vs `action_only_pool` | >= 10% reduction, paired CI excludes 0 | reduction 7.58% (0.7542 vs 0.8161); paired diff mean 0.0618, 95% CI [-0.0093, 0.1330]; n=66, excluded 174 | no |
| 3 `1-4` count RMSE vs `local_unpooled` | >= 10% reduction, CI excludes 0 | reduction -119.78% (0.6849 vs 0.3116); paired diff mean -0.3733, 95% CI [-0.4531, -0.2935]; n=117, excluded 123 | no |
| 4 sparse-state top-action accuracy vs `action_only_pool` | >= 5pp, CI excludes 0 | +4.56pp (0.4637 vs 0.4181), 95% CI [0.0102, 0.0811]; n=118, excluded 122 | no |
| 5 false-improvement rate vs `action_only_pool` | <= baseline + 1pp | 0.1799 vs 0.1948 | yes |

### Current unstructured family — screen FAILED

| item | threshold | observed | pass |
|---|---|---|---|
| 1 zero-count coverage | >= 50% | 227/227 = 100.0% | yes |
| 2 zero-count RMSE vs pool | >= 10%, CI excludes 0 | reduction 4.50% (0.9695 vs 1.0151); diff mean 0.0457, 95% CI [-0.0219, 0.1132]; n=64, excluded 176 | no |
| 3 `1-4` RMSE vs local | >= 10%, CI excludes 0 | reduction -112.51% (0.8429 vs 0.3966); diff mean -0.4462, 95% CI [-0.5300, -0.3625]; n=121, excluded 119 | no |
| 4 sparse top-action vs pool | >= 5pp, CI excludes 0 | +4.41pp (0.4548 vs 0.4107), 95% CI [0.0116, 0.0765]; n=121, excluded 119 | no |
| 5 false-improvement vs pool | <= baseline + 1pp | 0.1875 vs 0.1979 | yes |

### Secondary diagnostic (hypothesis 1, not a screen item)

Per-record Spearman correlation between eligible leave-one-action-out
signature distances and absolute true target-action Q differences:

- hidden_cluster: mean 0.3071, 95% CI [0.2829, 0.3314], 960 finite
  record/action correlations — positive, interval excludes zero (diagnostic
  passes);
- current_unstructured: mean 0.2833, 95% CI [0.2597, 0.3069], 960 finite —
  positive, interval excludes zero (diagnostic passes).

The distance ordering carries real signal in both families, yet the kernel
route still dilutes `1-4` count estimates relative to the local mean (item 3
fails decisively in both families), so the mechanism screen fails.

### Coverage totals (all target pairs, per family)

hidden_cluster: primary 224/269 zero-count, all 1-4/5-16/17+ emitted;
pool 269/269; local 0/269 zero-count (by construction); anchor 0/269
zero-count, 790/943 at 1-4, 1008/1036 at 5-16, 3506/3512 at 17+.
current_unstructured: primary 227/261 zero-count; pool 261/261; anchor
774/919 at 1-4, 1013/1037 at 5-16, 3538/3543 at 17+.

### Descriptive greedy-with-floor policy diagnostic (no safety claim)

Mean oracle return change (mu-weighted) / mean minimum componentwise change:

- hidden_cluster: local +1.088 / +0.352; pool +0.441 / -0.635; anchor
  +0.700 / -0.100; primary +0.499 / -0.550.
- current_unstructured: local +1.216 / +0.331; pool +0.583 / -0.698; anchor
  +0.810 / -0.143; primary +0.594 / -0.603.

Incomplete rows retained the original policy row
(`route_incomplete_for_policy`).

## Frozen classification

`NOT_SUPPORTED`: the hidden-cluster mechanism screen fails (items 2, 3, 4),
so the classification is `NOT_SUPPORTED` regardless of the current family.
No integrity, provenance, isolation, or frozen-contract check failed
(`integrity_failures` is empty), so the result is not `INVALID`.

## Environment

- Python 3.13.9 (`C:\Users\Admin\anaconda3\python.exe`), numpy 2.4.6,
  scipy 1.16.3, ruff 0.12.0; Windows 11 10.0.22631; CPU-only.
- Formal run created 2026-09-09T05:12:04Z (see `environment.json`).

## Formal artifact SHA-256 (`results/FP-KERN-001/claude/`)

- `analysis.json` (14729 B): `b3a7d1834772b3ed5f0bf4ffc45a1bb7eaea379b407e2c93e1f4a85837db4244`
- `checks.log` (3425 B): `cfe4ffd0f7c41681bed169ff14a9c5fb9d79c7b30e1a070aafafcdcb68fcd69f`
- `commands.log` (486 B): `eb1a5514b958efdf35d329b0996278ccb2cd1a1b66010d9daf3f1e8403213bf3`
- `config.json` (969 B): `dc6ee372ab7180dee5a6b427d4069d8220bf9593c643253c7d89f7cd62b53d28`
- `environment.json` (784 B): `c98bac6b1a2a59bb9ccc34d5be7aa157574eb1936f0257aa4b4a212273b9a768`
- `summary.json` (51819 B): `45314545712f0ddedea99fbf6e064a70c95d1fb279dff3745908f1f6ef254239`
- `task_results.json` (34153095 B): `a8cbf3ac790255224ce0ff4931af55551c82821be997b8145c9408da8e582921`

## Acceptance-criteria assessment (Claude route)

1. Frozen routes/formulas/support/bandwidth/families implemented exactly —
   verifier K1-K17 pin them; analyzer regenerates all 480 records exactly.
2. Estimator inputs observable-only; oracle keys rejected at the pure
   boundary (K11) and absent from route outputs (analyzer separation check).
3. Three streams derive from deterministic nonoverlapping substreams
   (`SeedSequence.spawn`); seeds recorded per record and replayed exactly.
4. Primary signature excludes the target action and requires two common
   observed non-target actions (K1, K2).
5. Weights, bandwidths, counts, denominators, ESS, and estimates are finite
   and reconstructible from serialized observables (analyzer reconstruction).
6. Zero-count primary estimates use only other-state observations (K7);
   anchor never emits at zero target count (K9).
7. State-label permutation equivariance verified (K10).
8. Hidden structure and true quantities never enter route selection,
   weights, bandwidths, estimates, or abstentions (K16, separation checks).
9. Ordered abstention contract without fallback (K3, K12, K13).
10. Smoke (16 records) passed before the formal run and changed no constant
    (first_result.md).
11. The formal route produced exactly 480 records with the frozen seed,
    matrix, task identity, and route names (evaluator and analyzer checks).
12. All count-bin denominators, comparable sets, coverage, errors,
    orderings, policy diagnostics, and paired intervals are independently
    reconstructed by the strict analyzer.
13. The classification follows the frozen five-item screen; no cell or
    secondary metric substituted.
14. All verifiers, strict JSON, provenance, hash, and task-scoped Ruff
    checks pass.
15. Commands, environment, the smoke p0 repair, hashes, and limitations are
    recorded here and in first_result.md.
16. Claude built the route from the common execution-start commit and sealed
    before disclosure; GPT-side and reciprocal verification are pending.
17. Reciprocal verification pending both seals.
18. `ACTIVE_WORKSPACE.md` and `main` untouched by this route.

## Limitations

- Feasibility evidence only; no safety certificate or nondegradation claim.
- The shared `V_hat` nuisance couples all actions at successor states (see
  the theory note's required statement); it is not target-trajectory leakage.
- Items 2-4 failures are scientific, not defects: the kernel route's
  zero-count accuracy gain over unconditional pooling is real but below the
  frozen 10% threshold with an interval covering zero, and kernel pooling
  strictly hurts `1-4` count pairs versus the local mean.
