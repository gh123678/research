# FP-MART-001 Claude repair — formal result

Date: 2026-09-03. Branch: `claude/FP-MART-001`.
Implementation seal (parent of this evidence): `e0343871fd313c80f4f49ffa773db10eb75667d6`.
The frozen 480-record formal matrix and the strict analyzer were run from that
exact clean commit. `first_result.md` and sealed commit `c5da243` remain the
historical first-result record; the sealed-era raw outputs were snapshotted at
`/tmp/sealed_root_backup/` before the formal rerun overwrote the canonical path.

## Environment

Same interpreter stack as recorded in `environment.json` of the formal run
(Python/NumPy/SciPy versions and git HEAD `e034387...` are in that file;
sha256 below).

## Commands

```
python -B evaluate_visit_indexed_certificates.py --tasks 30 --trajectory-lengths 256 1024 4096 16384 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8.0 --mixing 0.08 0.5 --gap-bonuses 0.0 0.5 --gamma 0.7 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-MART-001/claude
python -B analyze_visit_indexed_certificates.py --baseline-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/fixed_policy_finite_sample_certificates --new-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-MART-001/claude
```

## Raw-output hashes (formal, seven files)

- config.json: a2276eae06ba8689864cac2ad3d3d0e92b069014b2046d0180ee172bb8296ea0
- environment.json: b1f4a530ba0f16e4b96d6da36d6caa3ebec419eb11cc935f0e89e7d8ce23f5f6
- task_results.json: 365a0c979835890165c3ed1acc34938afcccaebef187af51780b165663d9afe5
- summary.json: bb052450e498e82fcfef594718d4552cd6d3d1161b96a100bb04456c8a0d614d
- commands.log: 350d131ad16c907f51254e217e21a9c3aa171b65b7faaa257e555a5a8a5c5e58
- checks.log: 0091ae7464bb3eafc6eae39f3ee56153895ee5c5adf9053c2196d68506b2591b
- regression.json: 9a88e6944d13cfc44d1c6e7d95f8f42a4b1799b1e01ad769bed73a3e7896ca7a

## Analyzer outcome

480/480 records compared; old_only/new_only keys 0; duplicate keys 0;
mismatch_count 0; max numeric difference 0.0 (bit-identical legacy fields);
baseline hash check, config/task-record/summary regression, and the
schema/oracle-provenance audit all pass; `PASS visit-indexed certificate
analysis`.

## Emission and usefulness (rates per route and trajectory length, 30 tasks/cell)

Emission (selective_high_probability_certified): direct_exact
0.025/0.85/1.0/1.0 at n=256/1024/4096/16384; direct_softmax
0.025/0.6167/0.9583/1.0; vfirst_nosplit_exact 0.025/0.85/1.0/1.0;
vfirst_nosplit_softmax 0.025/0.85/1.0/1.0.

Primary threshold `improves_over_zero_initialization := total_bound < B`
(among-emitted = overall, since non-emission is never counted as success):
direct_exact 0/0/0/0; direct_softmax 0/0/0/0; vfirst_nosplit_exact
0/0/0/0.4; vfirst_nosplit_softmax 0/0/0/0.0333.

Secondary threshold `below_two_B_range := total_bound < 2B`:
direct_exact 0/0/0/0.0583; direct_softmax 0/0/0/0;
vfirst_nosplit_exact 0/0/0.2833/1.0; vfirst_nosplit_softmax
0/0/0.1/1.0. Both thresholds are reported descriptively; none is selected.

## Audit violations and anomalies

- Oracle-audit bound violations: none (`audit_violations` empty).
- Per-group residual audit: no violations across all 480 records
  (`per_group_residual_violations` empty).
- Reward-declaration audit: `declaration_verified == True` on every record
  (declared `1 + gap_bonus` dominates the observed true reward max).
- Anomaly: the 16-record smoke configuration exhibited a deterministic 1-ulp
  difference (max 1.05e-11, within tolerance, zero mismatches) confined to the
  LAPACK-derived `right_hoeffding_inflation`/`omega_E` chain of the untouched
  legacy certificate path (see `repair_smoke.md`). The formal 480-record run
  reproduces the frozen baseline bit-identically (max difference 0.0); the
  anomaly therefore does not affect formal conclusions.

## Limitations

- Variance-adaptive constituent remains deterministically unavailable; no
  post-hoc minimum with the Hoeffding radius is formed.
- Usefulness thresholds are descriptive deterministic statements about emitted
  bounds, not tuned or selected claims.
- Per-group residual audit and oracle audit are empirical diagnostics.

## Acceptance assessment (numbered per the repair request)

1. Compensated supermartingale proof with E[M_n] <= 1 and the same constant: satisfied.
2. Narrowed variance-proxy conclusion: satisfied (theory and module text).
3. Declared reward bound; B derived in the pure module; true max only post-construction in `oracle_audit`: satisfied and verified on all 480 records.
4. Pure-module validation incl. original-integer counts, dimensions, sums, aggregation; counterexample n=8, [200,200], [100,100,100,100] rejected; failing-then-passing recorded: satisfied (27/27 tests).
5. Frozen 30-seed-per-cell smoke stride; all 16 smoke records baseline keys: satisfied.
6. Legacy config/task-record/176-summary-row preservation, strict JSON, schema/oracle-provenance separation, no baseline mutation: satisfied (0 mismatches, bit-identical).
7. Both descriptive thresholds reported without selection: satisfied.
8. Falsified statements corrected; historical first result preserved; new repair evidence written: satisfied.

## Acceptance assessment (frozen task FP-MART-001, numbered 1-18)

1. PASS. `theory.md` derives the sampling filtrations, visit stopping times,
   measurability, martingale differences, conditional ranges, and the
   optional-skipping step for all three residual families.
2. PASS. The mandatory theorem gives one event over all `G = m + 2d` residual
   groups and all `1 <= k <= n` with total failure probability at most `delta`,
   via the compensated supermartingale `M_t = exp(lambda*S_t - lambda^2 B^2
   J_{<t}/2)` with `E[M_n] <= 1` and pathwise `sum J <= k`.
3. PASS. Random observed counts are substituted only through the simultaneous
   event; no conditional-independence or fixed-count assumption (theory.md
   Section 5; enforced by the 27/27 contract tests).
4. PASS. The theorem and all outputs use the `selective_high_probability_certified`
   emission semantics exactly; no prior full-support claim is made.
5. PASS. Certificate functions accept no oracle model, occupancy, value,
   residual, or true initial-error input; `declared_reward_bound = 1 +
   gap_bonus` is declared before sampling and `B = declared/(1-gamma)` derived
   in the pure module; the true max is read only post-construction inside
   `oracle_audit.reward_declaration`, with `declaration_verified == True` on
   all 480 formal records.
6. PASS. Direct-Q all-layer and V-first no-split exact/softmax bounds compose
   with the visit-indexed radii without changing the completed deterministic
   recurrence; the formal run reproduces all legacy fields bit-identically
   (max numeric difference 0.0).
7. PASS. Required support, empirical margins, algorithm mode, risk budget, and
   finite arithmetic are validated with deterministic ordered failures;
   original-integer count, dimension, sum, and aggregation rejections are
   tested, including the mandated counterexample `n=8, [200,200],
   [100,100,100,100]`.
8. PASS. New fields use the `visit_indexed_certificate` namespace and the
   `selective_high_probability_certified` status; legacy status meanings are
   unchanged (176-row summary regression, zero mismatches).
9. PASS. All machine outputs are strict JSON (NaN/Infinity and duplicate-key
   rejection), use `null` for unavailable values, and structurally separate
   certificate inputs from `oracle_audit`; the schema/oracle-provenance audit
   passes on smoke and formal outputs.
10. PASS with disclosed timing anomaly. All five mandated verifiers
    (`verify_visit_indexed_martingale_certificate.py`,
    `verify_fixed_policy_q_routes.py`, `verify_finite_sample_theorems.py`,
    `verify_crossfit_markov_certificate.py`, `verify_end_to_end_sarsa.py`) now
    pass, but `verify_end_to_end_sarsa.py` was accidentally omitted before the
    formal run; the full exact five were run only during the later evidence
    repair, after formal seal `e3c3719`. `verify_theory.py` is additional only.
    Complete outputs are in `checks.log`; see `repair_smoke.md`.
11. PASS. The smoke strict-JSON, nonfinite, oracle-separation, and
    legacy-regression gates did pass before the formal matrix began; the
    anomaly in criterion 10 concerns verifier timing only, not these smoke
    gates, which were satisfied at the required point.
12. PASS. The formal result has exactly 480 records exactly matching the frozen
    configuration and seed 20260829 (config/task-record regression passes).
13. PASS. All legacy nonnumeric leaves match exactly; numeric leaves yield zero
    mismatches under `math.isclose(rel_tol=1e-12, abs_tol=1e-12)` (formal max
    difference 0.0; the smoke 1-ulp anomaly is within tolerance and disclosed).
14. PASS. Exact-route emission is 100% at trajectory lengths 4096 and 16384;
    all four rates (exact 0.025/0.85/1.0/1.0, softmax 0.025/0.6167/0.9583/1.0
    at n=256/1024/4096/16384, both vfirst routes 0.025/0.85/1.0/1.0),
    nontriviality rates, and deviations from 2.5%/85.0%/100%/100% are reported
    without tuning.
15. PASS. Every empirical audit violation would be reported with its
    configuration; `audit_violations` and `per_group_residual_violations` are
    empty across all 480 records, and no empirical coverage statistic is used
    as proof.
16. PASS. This route records exact commits (implementation seal `e034387`,
    formal evidence seal `e3c3719`), environment (`environment.json`),
    commands, raw-output hashes, anomalies, metrics, limitations, and numbered
    acceptance assessments in `repair_smoke.md` and this file.
17. PENDING. Reciprocal verification (GPT reproduces/verifies this route and
    Claude reproduces/verifies the GPT route, each ending in `PASS`) has not
    yet been completed.
18. PENDING. The final synthesis explaining material route differences and the
    `ACTIVE_WORKSPACE.md` update await the reciprocal verification and user
    approval.
