# FP-ADV-001 Codex blind first result

Date: 2026-09-08  
Route: `codex/FP-ADV-001`  
Status: `PASS` for proof, implementation, verifiers, and smoke; formal matrix not
yet run.  
Seal: the commit containing this file; its exact hash is recorded immediately
after creation in the task evidence.

## Frozen identity and independence

- Scientific baseline: `c579047950dfabb2600020cd2e53dd24b3e39c84`.
- Scientific activation: `10a9a94e24ec92a59e7c756f9af6ce07b2f30e59`.
- Common route execution baseline:
  `4078f6911cbfb4654205772685f49896e4e8cad2`.
- Task/design/plan were frozen before implementation. No Claude implementation,
  first result, conclusion, or formal output was read.
- The unrelated untracked learning record
  `docs/实验学习记录_恢复版.md` was preserved and excluded from every change and
  command target.

## Environment

- Windows 11, PowerShell.
- Python: `C:\Users\Admin\anaconda3\python.exe`, version 3.13.9.
- NumPy 2.4.6, SciPy 1.16.3, Matplotlib 3.10.6.
- CPU-only; no new dependency or fee category.

## Proof and implementation

The independent derivation is in `theory.md`. The route implements:

- `action_gap_certificate.py`: pure observable certificate and one
  `theta=0.5` update;
- `verify_action_gap_certificate.py`: formula, support, abstention, update,
  strict-JSON, and no-oracle contract tests;
- `evaluate_action_gap_certificates.py`: paired FP-TU reconstruction, route
  estimate capture, action-gap attachment, and post-certificate oracle audit;
- `analyze_action_gap_certificates.py`: strict parser, frozen regression,
  complete formula reconstruction, policy validation, local/global dominance,
  and enumerated oracle audit.

No legacy estimator file was changed. The evaluator reuses the existing public
estimator functions and independently checks the reconstructed Q estimates
against all serialized legacy Q metrics before adding the new namespace.

## Test-first and failed-attempt record

1. The new verifier was created before the module. Its first run failed as
   expected with `ModuleNotFoundError: No module named
   'action_gap_certificate'` and exit code 1.
2. After implementing the module, the verifier passed.
3. The first task-scoped Ruff run found two ordinary style defects: one unused
   import and one assigned lambda. Both were corrected without changing the
   frozen formulas or protocol; the next Ruff run passed.
4. No evaluator, schema, regression, strict-JSON, or numerical run failed.

## Exact commands

Unit and style checks:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_action_gap_certificate.py
C:\Users\Admin\anaconda3\python.exe -m ruff check action_gap_certificate.py verify_action_gap_certificate.py evaluate_action_gap_certificates.py analyze_action_gap_certificates.py
git diff --check
```

Smoke evaluation (the exact frozen seed schedule, with one task per cell and
two prefix lengths):

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_action_gap_certificates.py --tasks 1 --trajectory-lengths 256 16384 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --transfer-fraction 0.5 --seed 20260829 --output-dir results/FP-ADV-001/codex_smoke
C:\Users\Admin\anaconda3\python.exe -B analyze_action_gap_certificates.py --result-dir results/FP-ADV-001/codex_smoke --allow-smoke
```

The evaluator preflight also ran and passed:

- `verify_finite_sample_theorems.py`;
- `verify_fixed_policy_q_routes.py`;
- `verify_crossfit_markov_certificate.py`;
- `verify_end_to_end_sarsa.py`;
- `verify_visit_indexed_martingale_certificate.py`;
- `verify_action_gap_certificate.py`;
- `verify_time_uniform_mixture_certificate.py`.

## Smoke evidence

Location: ignored directory `results/FP-ADV-001/codex_smoke/`, containing the
required seven core files.

- Records: 8.
- Frozen FP-TU hashes: all three matched the task exactly; baseline count 480.
- Legacy comparison: 12,520 numeric and 3,748 nonnumeric leaves, zero mismatch.
- New certificate reconstruction: 18,180 numeric and 6,492 nonnumeric leaves,
  zero mismatch.
- Oracle input findings: 0.
- Policy/floor/row-sum/Bellman-LCB violations: 0.
- Local/global penalty dominance violations: 0.
- Local/global decision dominance violations: 0.
- Oracle false orderings, Bellman-bound violations, value decreases, and return
  decreases: all 0 because no smoke update emitted.
- Local exact/softmax updated-policy agreement: 8/8.
- Updates emitted: 0/8 on all six routes. This is recorded without retuning and
  is not treated as a formal conclusion.

## Acceptance assessment before formal evaluation

1. Inherited simultaneous event/no risk resplit: `PASS` by proof and serialized
   inherited bounds.
2. Exact decomposition/TV-span inequality: `PASS` by derivation and fixtures.
3. Softmax mass/effective row/contamination: `PASS`, including dense-weight and
   large-beta fixtures.
4. Pure observable interface and declared `B`: `PASS`; no forbidden oracle
   parameter or serialized input.
5. Post-data selection semantics: `PASS` by deterministic-event argument.
6. Positive-count, positive-LCB, transferable-mass eligibility: `PASS`.
7. Simplex, finiteness, nonnegativity, and floor: `PASS`.
8. Bellman lower bound and one-step improvement implication: `PASS` in proof,
   fixtures, and smoke validation.
9. Local penalty no wider than matching V-first global control: `PASS`.
10. Partial support localized and unvisited receiver handled explicitly:
    `PASS`.
11. Direct-Q remains global `2E_Q` only: `PASS`.
12. Additive namespace and legacy preservation: `PASS` on smoke records.
13. Strict JSON and structural oracle separation: `PASS`.
14. Inherited/new verifiers, Ruff, hashes, schema, and reconstruction: `PASS`.
15. Smoke-before-formal and blind seal: `PASS` when this commit is created.
16--20. Formal record count, full legacy regression, empirical audit, complete
    evidence, reciprocal verification, and synthesis: pending by lifecycle.

## Limitations and next step

The smoke matrix is deliberately too small for a usefulness conclusion and
emitted no update. The frozen 480-record command must now be run exactly once
from this sealed implementation. No formula, threshold, route, donor rule,
transfer fraction, metric, seed, or matrix may change in response to smoke or
formal output.
