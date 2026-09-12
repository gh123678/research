FP-TIGHT-001 same-actor derived verification
==============================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Coverage re-derived from the bundle
  PASS  48 route-records (found 48)
  PASS  frozen covers the realized error on every route-record (0 bad)
  PASS  bernstein covers the realized error on every route-record (0 bad)
  PASS  empirical_bernstein covers the realized error on every route-record (0 bad)
  PASS  every realized error is strictly positive (minimum 6.349e-03), so the audit is not comparing Qhat to itself
  PASS  the frozen arm certifies on every route-record (found {'certificate_emitted'})

2. Risk allocation sums to delta
  PASS  bernstein: delta_each = 0.0013888889, and 12 bounds of that size sum to delta
  PASS  empirical_bernstein: delta_each = 0.0013888889, and 12 bounds of that size sum to delta
  PASS  counterfactual_no_envelope: delta_each = 0.0020833333, and 12 bounds of that size match the frozen 2d split
  PASS  the counterfactual keeps the frozen 2d allocation, as it is the frozen construction with one term deleted rather than a new certificate

3. The unsound counterfactual is labelled, not hidden
  PASS  every counterfactual entry carries sound=false (found {False})
  INFO  the counterfactual floor fails coverage on 0/48 route-records; it is a diagnostic, never a guarantee.

4. Sealed corpus untouched
  PASS  SCIENCE fixed_policy_expected_sarsa.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_variance_certificate.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE model.py is byte-identical to the FP-ITER5-001 record
  INFO  evaluator evaluate_fp_attn_001.py matches this task's record
  INFO  evaluator evaluate_fp_iter2_001.py matches this task's record
  INFO  evaluator evaluate_fp_scale_002.py matches this task's record
  PASS  SCIENCE fixed_policy_expected_sarsa.py matches this task's own record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py matches this task's own record
  PASS  SCIENCE fixed_policy_variance_certificate.py matches this task's own record
  PASS  SCIENCE model.py matches this task's own record
  PASS  the new module does not import the frozen certificate module
  PASS  the frozen certificate module is still present

5. The sample-size arm, if present
  PASS  the x4 arm is present on every route-record
  PASS  the x4 arm still covers the realized error (0)

6. Analyzer determinism
  PASS  analyze_fp_tight_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

7. Replay of the affected sealed programs
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_policy_quantities_by_solve.py replays with exit 0 (got 0)
  PASS  verify_fp_iter4_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_iter5_001_same_actor.py replays with exit 0 (got 0)

==============================================================================
SUMMARY
  mean E_Q bernstein                    0.2061  -15.0%
  mean E_Q counterfactual_no_envelope   0.1334  -45.0%
  mean E_Q empirical_bernstein          0.1940  -20.0%
  mean E_Q frozen                       0.2426  +0.0%
  H1_coverage_violations      : {'bernstein': 0, 'empirical_bernstein': 0, 'frozen': 0}
  H2_lost_eligibility         : 0
  H3                          : PASS
  H4                          : PASS
  H5                          : FALSIFIED
  H6                          : FALSIFIED
  H7                          : PASS
  never-emitting at step 1    : 26
==============================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round. Step 1 only: nothing here says a
tighter certificate extends the iteration.
