# FP-ADV-001 Claude route: formal result

Status: complete after the single frozen formal run.
Author: Claude Code, branch `claude/FP-ADV-001`, isolated worktree.
Seal commit (blind first result): `191e13b` (`[claude] seal FP-ADV-001
independent theory, implementation, and smoke evidence`).

## 1. Single formal run

Executed exactly once after the seal, from the worktree `icrl_softmax/`:

```text
python -B evaluate_action_gap_certificates.py --tasks 30 \
  --trajectory-lengths 256 1024 4096 16384 --n-states 6 --n-actions 4 \
  --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 \
  --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 \
  --transfer-fraction 0.5 --seed 20260829 \
  --output-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-ADV-001\claude
```

Result: exit 0, `wrote 480 matched comparisons`, `PASS FP-ADV-001 paired
evaluation`. No rerun was performed.

Analysis (rerunnable; does not regenerate records):

```text
python -B analyze_action_gap_certificates.py \
  --result-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-ADV-001\claude
```

Result: `PASS FP-ADV-001 analysis: records=480, formal=True`.

## 2. Regression against the frozen FP-TU-001 baseline

- Baseline hashes verified before and after every run and analysis
  (config `43dcb96b…`, task_results `0e5eab39…`, summary `565fc4d2…`).
- Full legacy regression over all 480 records, the config, and the summary
  after stripping only the additive `action_gap_certificate` namespace:
  **0 mismatches** (nonnumeric leaves exact; numeric leaves at
  `rel_tol=abs_tol=1e-12`).
- Baseline files unchanged during analysis (rehashed afterwards).

## 3. Formal artifact hashes (SHA-256, first 16 hex digits)

- `config.json` `1d065c23f8387113`
- `task_results.json` `1f1a9e774f8013b7` (full:
  `1f1a9e774f8013b75c22c3240d2be9bd676c79c2fa125716f622b9431d3057da`)
- `summary.json` `a632ae7a2311d931` (repaired; see anomaly below)
- `regression.json` `994a60a44739d67f`
- `environment.json` `f143b7099c767f2c`
- `commands.log`, `checks.log` appended by evaluator and analyzer.

## 4. Empirical findings (diagnostics, not theorem evidence)

- **Zero update emissions on all six routes in all 480 records**
  (120 per trajectory length).
- Abstention reasons (`route_reasons`): local routes abstain via
  `state_certificate_not_emitted` (13 records, exact local only, length-256
  cells where the state bound did not emit), `candidate_pair_unvisited`
  (122/135 records), `gap_lcb_nonpositive` (467 local exact), and
  `no_transferable_mass` (2). Global controls additionally report
  `recovery_radius_unavailable` (168 records) whenever the complete-Q bound
  `E_Q` did not emit (incomplete pair support).
- The dominant terminal reason is `gap_lcb_nonpositive`: even at length
  16384 the local uncertainties (roughly 2.7-4.4) exceed the estimated
  action differences (roughly 0.5-0.7 under gap bonus 0.5). The
  certificates are valid but too conservative to certify any ordering in
  this matrix. **Hypothesis 8 (empirical usefulness) fails; the theory and
  safety contract are unaffected** — the task explicitly permits a
  zero-useful-update negative completion.
- Dominance: all 6210 exact and 6210 softmax evaluated local penalties are
  within the matching global `2E_Q` penalty (evaluator hard-fail check plus
  analyzer recheck); weak update dominance holds vacuously (no global
  emission).
- Oracle audit (truth-only, structurally separated, never a certificate
  input): 0 records with false ordering, bound violation, Bellman
  violation, or value decrease — vacuously, since no update was emitted.

## 5. Anomaly and repair (summary pipeline)

- The sealed evaluator omitted the legacy visit-indexed per-row summary
  augmentation, so the first formal `summary.json` (hash `80250409…`)
  lacked the `visit_indexed_certificate` row key and failed the full
  summary regression (64 row-key mismatches; config and all 480 records
  matched with zero mismatches).
- Repair, without any evaluator rerun: the evaluator code was fixed to
  apply `augment_summary` from `evaluate_visit_indexed_certificates`
  between the legacy and time-uniform augmentations, and `summary.json`
  was regenerated once from the **untouched** formal `task_results.json`
  using the repaired deterministic pipeline. `task_results.json` hash
  `1f1a9e77…` is byte-identical before and after the repair (verified).
- After repair the analyzer reports `legacy regression mismatches=0` over
  config, records, and summary. The sealed smoke artifacts predate this
  defect's discovery and are unaffected in kind (smoke regression is
  record-level only); the smoke records remain valid evidence.

## 6. Acceptance-criteria assessment (formal phase)

Criteria 1-15 and 18-19: as in `first_result.md`, all PASS for this route.

16. PASS — the formal output contains exactly 480 records produced by the
    frozen seed 20260829, the frozen matrix, the frozen formulas, and
    transfer fraction 0.5 (analyzer revalidates every gate, uncertainty,
    LCB, transfer amount, and policy row from serialized observables).
17. PASS — full legacy regression: nonnumeric leaves match exactly,
    numeric leaves have zero mismatch at
    `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`.
18. PASS — the analyzer enumerates every empirical false ordering, bound
    violation, Bellman violation, value decrease, and return change under
    `oracle_audit` (all zero here) and marks them not theorem evidence.
20. PENDING — reciprocal verification of the GPT route has not started;
    this route never accessed GPT FP-ADV-001 material.

## 7. Limitations

- The empirical conclusion is a verified negative for usefulness on this
  matrix: no action ordering was certifiable at the frozen risk level, so
  the single safe update was everywhere abstained. Safety was never
  exercised against a realized update in the formal run.
- The bound validity is exactly the inherited selective statement:
  P(EmitUpdate and (any used ordering false or exists s with
  V^{pi_plus}(s) < V^pi(s))) <= delta. No conditional-on-emission coverage
  and no high-probability support claim.
- One fixed policy, one trajectory, one update; no online, repeated, or
  convergence claim.
