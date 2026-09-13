FP-ITER8X-001 same-actor derived verification
========================================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Frozen inputs
  PASS  48 route-records (found 48)
  PASS  horizon 12 (found 12)
  PASS  multiplier 8 (found 8)
  PASS  certification is 8x the sealed chains (found 131072)

2. H1/H2 re-derived: validity and soundness
  PASS  frozen: 402 emitted steps, 0 degrading
  PASS  frozen: 0 non-positive gains
  PASS  frozen: 0 certificate violations
  PASS  frozen: 0 abstentions without a reason
  PASS  empirical_bernstein: 435 emitted steps, 0 degrading
  PASS  empirical_bernstein: 0 non-positive gains
  PASS  empirical_bernstein: 0 certificate violations
  PASS  empirical_bernstein: 0 abstentions without a reason
  PASS  every realized error is strictly positive (min 6.349e-03)

3. The two arms are genuinely separate trajectories
  PASS  the arms diverge on 20/48 route-records, so the control is real
  INFO  the arms are guaranteed to diverge wherever the certificate changes an eta choice; identical trajectories on some records are expected and are not evidence of a bug.

4. H3/H4 re-derived: the population trajectory
  PASS  H3: 68 seventh-step emissions across both arms
  PASS  the summary's H3 verdict matches a fresh re-derivation
  PASS  H4: step-7 frozen count 33 against the 1x step-6 count 9, matching the summary
  PASS  the deepest trajectory reaches 12 steps, so the horizon is not the binding constraint

5. H5 re-derived: containment across arms
  PASS  the containment theorem holds where it applies: step-1 frozen 42 subset of repaired 44
  PASS  the summary records that the theorem holds at step 1
  INFO  cross-arm containment holds at 11/12 non-empty levels; the 1 exception(s) are expected because the arms diverge, and are reported rather than suppressed.
        counterexample at step 11: [(0.08, 2, 'expected_finite')]

6. H7 re-derived: the step-6 minimum
  PASS  H7: step-6 minimum 0.05168977974515104 against the sealed 0.019347, matching the summary

7. The sealed 1x reference is intact, and the corpus is untouched
  PASS  the sealed 1x run emitted 9 at step 6 (expected 9)
  PASS  SCIENCE fixed_policy_expected_sarsa.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_variance_certificate.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE model.py is byte-identical to the FP-ITER5-001 record
  PASS  evaluate_fp_iter2_001.py is untouched and still uses only the sealed certificate
  PASS  evaluate_fp_attn_iter_001.py is untouched and still uses only the sealed certificate
  PASS  evaluate_fp_census_001.py is untouched and still uses only the sealed certificate

8. Analyzer determinism
  PASS  analyze_fp_iter8x_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

9. Replay of the foundation checks
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_policy_quantities_by_solve.py replays with exit 0 (got 0)

========================================================================================
SUMMARY
  emissions, frozen@8x : [42, 42, 40, 40, 38, 36, 33, 29, 28, 27, 25, 22]
  emissions, empB@8x   : [44, 44, 44, 42, 41, 38, 35, 34, 31, 29, 27, 26]
  emissions, sealed 1x : [22, 20, 15, 12, 12, 9] (horizon 6)
  deepest trajectory   : {'empirical_bernstein': 12, 'frozen': 12}
  seventh-step emissions: {'empirical_bernstein': 35, 'frozen': 33}
  H3                    : PASS
  H6                    : PASS
  H4                    : PASS
  H5                    : FALSIFIED
  H7                    : FALSIFIED
========================================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round. The 8x batch is a fresh independent
sample, not a superset of the sealed one, so size and realisation are not
separated; the frozen@8x arm exists to make that visible.
