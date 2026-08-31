# Shared fixed-policy finite-sample theorem implementation plan

> Date: 2026-08-31  
> Design: `docs/superpowers/specs/2026-08-31-shared-fixed-policy-finite-sample-theorem-design.md`  
> Constraint: preserve the three existing formal result directories and the historical archive.

## Task 1: Freeze source-backed theorem constants

Read only primary sources already identified in the project and verify:

- the stationary, time-independent Markov Hoeffding form based on additive reversiblization;
- the right spectral inflation definition;
- the fact that the theorem applies to a finite union of pre-fixed bounded functions;
- the evidence boundary for nonstationary starts and random reward noise.

Record the exact theorem mapping in `docs/research_branches/shared_fixed_policy_finite_sample_theory.md`. Do not make the optional visit-indexed martingale rate a hard dependency.

Acceptance:

- every imported probability statement names its assumptions;
- edge absolute gap is not used;
- the main theorem remains valid without the optional sharper-rate section.

## Task 2: Add failing theorem-contract tests

Create `verify_finite_sample_theorems.py` with deterministic fixtures for:

1. shared radius arithmetic and one global delta allocation;
2. Direct-Q residual identity at (Q^\pi);
3. exact and finite-softmax all-layer recurrence;
4. state clipping nonexpansion;
5. same-sample V-first ghost-target decomposition;
6. finite-softmax recovery leakage;
7. missing-support and nonpositive-margin rejection;
8. exact-route beta independence;
9. strict JSON serialization;
10. a uniform fast-mixing fixture that passes the complete certificate.

Run the new verifier before implementation and confirm it fails only because the new module is absent.

## Task 3: Implement the pure certificate module

Create `fixed_policy_finite_sample_certificate.py` with focused, side-effect-free functions:

- `right_hoeffding_inflation`;
- `shared_event_radii`;
- `one_hot_diagonal_lower_bound`;
- `direct_q_uniform_bound`;
- `state_value_uniform_bound`;
- `vfirst_nosplit_bound`;
- `observed_ghost_residuals`;
- strict finite-value conversion and stable failure-reason handling.

Required behavior:

- never evaluate a kernel margin when its occupancy lower bound is nonpositive;
- exact matching bypasses beta entirely;
- unavailable numeric fields are `None`;
- assumptions, status and failure reasons are explicit;
- no sampling, plotting or file writing occurs in this module.

Run `python -B verify_finite_sample_theorems.py` until all new contracts pass.

## Task 4: Expose only the required chain diagnostics

Inspect `markov_coverage_certificate.py`. Reuse its stationary distributions and right spectral values. Modify it only if a small public interface is required for:

- state/pair/edge right-Hoeffding inflation;
- exact support metadata;
- detection of numerical support truncation.

Do not move route-specific theorem logic into this file. Re-run `verify_crossfit_markov_certificate.py` after any change.

## Task 5: Integrate route certificates and no-split softmax

Modify `evaluate_fixed_policy_q_routes.py` to:

- compute fixed-(Q^\pi), fixed-(V^\pi) and recovery ghost residuals;
- call the new certificate module;
- add `vfirst_nosplit_softmax` using the existing recovery function;
- save route-level optimization, statistical, propagation and leakage terms;
- save certificate status, failure reasons and pathwise slack;
- preserve all existing route values for identical seeds and configurations.

The evaluator must continue to accept an arbitrary output directory. Existing formal outputs are read-only baselines.

## Task 6: Regression and smoke matrix

Run:

- `python -B verify_fixed_policy_q_routes.py`;
- `python -B verify_crossfit_markov_certificate.py`;
- `python -B verify_finite_sample_theorems.py`;
- `python -B verify_end_to_end_sarsa.py`;
- the blockwise verifier or its smallest deterministic invocation;
- a two-task, two-length fixed-policy smoke run into `results/fixed_policy_finite_sample_certificates_smoke/`.

Acceptance:

- all contracts pass;
- strict JSON parses successfully;
- no NaN or infinity is emitted;
- every synthetic pathwise bound has nonnegative slack;
- no existing result directory is overwritten.

Delete the smoke directory after it has served its regression purpose, or move it to the historical archive if it contains diagnostic value.

## Task 7: Reproduce the formal matrix with new fields

Run the same formal matrix and seeds as the existing cross-fit scan:

- (n_S=6), (n_A=4), \(\pi_{\min}=0.05\), \(\beta=8\);
- trajectory lengths (256,1024,4096,16384);
- mixing (0.08,0.5);
- reward bonus (0,0.5);
- 30 independent tasks per cell;
- 160 evaluation iterations.

Write only to `results/fixed_policy_finite_sample_certificates/`.

Before accepting the run, compare all pre-existing routes against `results/fixed_policy_q_routes_crossfit/task_results.json` on their common keys. Any mismatch beyond floating-point tolerance must be explained and fixed before analysis.

## Task 8: Analyze evidence without tuning to pass rate

Produce summaries for:

- empirical error by route and trajectory length;
- high-probability certificate pass rate;
- pathwise ghost-bound verification rate;
- coverage and kernel margins;
- failure-reason frequencies;
- theoretical-bound to empirical-error ratios;
- no-split exact versus no-split softmax paired differences.

Generate only figures that support these questions. A zero high-probability pass rate is an admissible result and must not trigger threshold tuning.

## Task 9: Update theory and comparison documents

Create:

- `docs/research_branches/shared_fixed_policy_finite_sample_theory.md`.

Update:

- `docs/research_branches/branch_a_direct_q_theory.md`;
- `docs/research_branches/branch_b_v_first_theory.md`;
- `docs/research_branches/crossfit_markov_certificate_theory.md`;
- `docs/research_branches/direct_q_vs_v_first_report.md`;
- `ACTIVE_WORKSPACE.md`.

Required corrections:

- remove the claimed need for layerwise Direct-Q concentration;
- remove the claimed need for no-split stage independence;
- demote cross-fit gap from a necessary fixed-policy proof device to an optional proof/diagnostic route;
- retain the real open problems: conservative coverage, sharper visit-indexed rates, random rewards, nonstationary starts and control.

Do not modify the manuscript main text.

## Task 10: Final audit

Perform:

- placeholder and contradiction scans;
- strict JSON validation for all new output files;
- finite-value scans;
- full verifier/regression rerun;
- active-directory cache/tmp scan;
- confirmation that the historical archive was not modified;
- confirmation that the three old formal result directories are byte-preserved.

The directory is not a Git repository, so no commit step is available. Report this limitation explicitly in the handoff.
