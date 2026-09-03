# FP-MART-001 Claude route: sealed first result

> Author: Claude Code (auxiliary researcher), independent construction.
> Date: 2026-09-03. This file is sealed by the commit that introduces it;
> that commit's SHA is the sealed first-result commit reported to the
> principal researcher. Written without any knowledge of the GPT route's
> result or conclusion.

## 1. Verdict

POSITIVE for the mandatory route (hypotheses 1-6): the visit-indexed
martingale certificate is proved, implemented, and formally evaluated with
exact-route emission 2.5% / 85.0% / 100% / 100% at lengths 256 / 1024 / 4096 /
16384, zero legacy mismatch, and zero oracle-audit violations. The optional
variance-adaptive hypothesis 7 is a recorded negative within the oracle-free
input contract (`variance_adaptive_unavailable` on all 480 records).

## 2. Start point and isolation

- Branch: `claude/FP-MART-001`, worktree `C:\tmp\research-FP-MART-001-claude`.
- Execution-start commit: `c8ec7e5c3165930663e26a07f98b38cc9ec186ad`
  (verified at start; worktree clean; no other route's artifacts inspected).
- Frozen baseline verified before and after all runs:
  `config.json` `a2276eae...8296ea0`, `task_results.json`
  `c84329bd...9cc3b2f25`, `summary.json` `49846c08...f69434b5ae`; 480 records.

## 3. Environment

- Python 3.13.9 (Anaconda, MSC v.1929, 64 bit), `C:\Users\Admin\anaconda3\python.exe`
- numpy 2.4.6, scipy 1.16.3, matplotlib 3.10.6
- Windows 11 (10.0.22631); runs executed from
  `C:\tmp\research-FP-MART-001-claude\icrl_softmax` with `python -B`
- Full record: `results\FP-MART-001\claude\environment.json`

## 4. Artifacts

Tracked (this branch):
- `icrl_softmax/visit_indexed_martingale_certificate.py` (pure certificate module)
- `icrl_softmax/verify_visit_indexed_martingale_certificate.py` (20 contract tests)
- `icrl_softmax/evaluate_visit_indexed_certificates.py` (additive evaluator)
- `icrl_softmax/analyze_visit_indexed_certificates.py` (regression + analysis)
- `icrl_softmax/docs/research_branches/FP-MART-001/claude/theory.md`
- this file

No existing file was modified; the evaluator reuses the frozen pipeline by
import, so no additive helper in GPT-owned modules was needed.

Ignored results: `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\claude\`
with `config.json`, `task_results.json` (480 records), `summary.json`,
`regression.json`, `environment.json`, `commands.log`, `checks.log`, and the
labelled `smoke\` subdirectory (16 records; not formal evidence).

## 5. Commands (exact; see `commands.log`)

Verifiers (all PASS):
`python -B verify_finite_sample_theorems.py`,
`python -B verify_fixed_policy_q_routes.py`,
`python -B verify_crossfit_markov_certificate.py`,
`python -B verify_end_to_end_sarsa.py`,
`python -B verify_visit_indexed_martingale_certificate.py`.
The new verifier was run before the module existed and failed with
`ModuleNotFoundError` as required by test-first development.

Smoke:
`python -B evaluate_visit_indexed_certificates.py --tasks 2 --trajectory-lengths 256 1024 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\claude\smoke`
then the analyzer with `--smoke`.

Formal:
`python -B evaluate_visit_indexed_certificates.py --tasks 30 --trajectory-lengths 256 1024 4096 16384 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\claude`
`python -B analyze_visit_indexed_certificates.py --baseline-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\fixed_policy_finite_sample_certificates --new-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\claude`

## 6. Raw metrics

Emission rates (share of 120 records per length emitting
`selective_high_probability_certified`):

| route | 256 | 1024 | 4096 | 16384 |
|---|---|---|---|---|
| direct_exact | 0.0250 | 0.8500 | 1.0000 | 1.0000 |
| direct_softmax | 0.0250 | 0.6167 | 0.9583 | 1.0000 |
| vfirst_nosplit_exact | 0.0250 | 0.8500 | 1.0000 | 1.0000 |
| vfirst_nosplit_softmax | 0.0250 | 0.8500 | 1.0000 | 1.0000 |

Observed full pair support: 0.025, 0.85, 1.0, 1.0 (exactly the frozen
expectation); observed full state support: 0.8917, 1.0, 1.0, 1.0.

Deviation attribution (pre-registered categories): all non-emissions are
category A (missing observed support; 117/120 at n=256, 18/120 at n=1024) or
category B (pair_kernel_margin_nonpositive; 28 at n=1024, 5 at n=4096,
direct_softmax only). No category C/D/E event occurred.

Nontriviality (emitted bound strictly below the deterministic `2B` range):
essentially 0 below n=16384 except vfirst routes at 4096 (28.3% exact, 10.0%
softmax); at 16384: direct_exact 5.8%, direct_softmax 0%, both vfirst routes
100%. Bounds are valid but usually conservative; validity and nontriviality
are reported separately as required.

Oracle audit: 0 bound violations across all emitted route certificates (each
emitted total bound exceeds the true sup error on its record). Empirical audit
coverage is reported for audit only and is not used as proof.

Legacy regression: 480/480 keys aligned with the frozen baseline; 0
mismatches; maximum common numeric absolute difference 0.0 (bit-identical).

Variance-adaptive constituent: `unavailable` on all 480 records (no
observable fixed-target variance proxy; no post-hoc minimum taken).

Anomalies: none.

## 7. Acceptance-criteria assessment (task list 1-18)

1. PASS - theory.md Sections 2-4 derive both filtrations, visit indicators,
   measurability, MDS property after optional skipping (transform form), and
   exact conditional range width `2B` for all three families.
2. PASS - one event over `G = m + 2d` groups and all `1 <= k <= n`, total
   failure `<= delta` (theory.md Section 6; verifier test 1).
3. PASS - random observed counts substituted only inside the simultaneous
   event (theory.md Section 7; verifier test 5).
4. PASS - selective semantics `P(Emit and violation) <= delta` used verbatim
   (theory.md Section 9; no `P(Emit)` claim anywhere).
5. PASS - certificate functions take no oracle input; zero initialization
   uses `B` (verifier test 7, signature inspection; smoke oracle-separation
   check over all records).
6. PASS - Direct-Q all-layer and V-first no-split recurrences unchanged,
   radii substituted (theory.md Section 8; legacy regression 0 mismatch).
7. PASS - support, margins, mode, risk budget, and finite arithmetic are
   validated with deterministic ordered failures (module `_FAILURE_ORDER`;
   verifier tests 9, 11-15).
8. PASS - all new fields under `visit_indexed_certificate`; status
   `selective_high_probability_certified`; legacy statuses untouched
   (verifier test 16).
9. PASS - strict JSON (`allow_nan=False`, `parse_constant` rejection), null
   for unavailable values, `oracle_audit` structurally separate (verifier
   test 17; analyzer loads).
10. PASS - all four legacy verifiers plus the new verifier pass (Section 5).
11. PASS - smoke matrix passed strict-JSON, nonfinite, oracle-separation, and
    aligned legacy-regression checks before the formal run (checks.log).
12. PASS - 480 records, frozen configuration and seed (`config.json`).
13. PASS - zero mismatch on all legacy leaves; numeric tolerance
    `isclose(1e-12, 1e-12)` (achieved difference exactly 0.0).
14. PASS - exact-route emission 100% at 4096 and 16384; all four rates equal
    the expected 2.5/85/100/100 with zero deviation; softmax deviations
    attributed to categories A/B without tuning.
15. PASS - audit violations (none) reported per configuration; no empirical
    coverage statistic is used as proof.
16. PASS - this file records commits, environment, commands, raw outputs,
    anomalies, metrics, limitations, and this numbered assessment.
17. PENDING - cross-verification of the GPT route (post-disclosure phase).
18. PENDING - final synthesis is GPT's task after disclosure; `main`
    untouched.

## 8. Hypothesis assessment

1. Supported: all three residual families are visit-indexed MDS under the
   stated stopped filtrations (theory.md Sections 3-4).
2. Supported: simultaneous two-sided radius `B*sqrt(2*log(2*G*n/delta)/k)`,
   no occupancy/spectral factor (Sections 5-6; exact constant `2B` derived).
3. Supported: random-count substitution valid via simultaneity (Section 7).
4. Supported: composition with both routes including early stopping and
   empirical-margin gates (Section 8; formal run).
5. Supported: certificate uses only observed counts, observed diagonals,
   declared reward bound, and public hyperparameters (verifier signature
   test; oracle separation).
6. Supported: emission equals observed full pair support for exact routes
   (2.5/85/100/100); legacy fields reproduce with zero mismatch.
7. Negative (recorded, non-blocking): no oracle-free observable variance
   proxy exists for fixed-target residuals, so no valid Freedman or
   empirical-Bernstein constituent; reported as
   `variance_adaptive_unavailable` everywhere (theory.md Section 10).

## 9. Limitations

- `log(2Gn/delta)` union overhead; bounds are usually conservative at the
  frozen scale (see nontriviality rates).
- Scope: one fixed policy, frozen stationary-start trajectory, deterministic
  edge reward, finite state-action space; no stochastic reward noise,
  nonstationary starts, or online control.
- Web fetch of primary-source PDFs was blocked in this environment; the
  concentration argument is fully self-contained in theory.md, so no imported
  statement is load-bearing; bibliographic records are cited for attribution.

## 10. Sealing statement

This is the Claude route's sealed first result. The GPT route's branch,
results, reports, and conclusions were not read before this commit. Cross-
verification of the GPT route (acceptance criterion 17) proceeds only after
the GPT first-result commit exists.
