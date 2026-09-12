FP-RANGE-001 same-actor derived verification
====================================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Frozen inputs
  PASS  48 route-records (found 48)
  PASS  the registered rungs (found [1, 2, 4, 8])
  PASS  item counts scale exactly with the rung

2. Coverage re-derived, every rung
  PASS  data_range: 192 certificate-fits, 0 coverage violations

3. The price decomposition is internally consistent
  PASS  every record carries a finite, positive price decomposition (0 bad)
  INFO  the empirical tail mass is exactly zero on 48/48 route-records, which is why the whole price lands in the Cauchy-Schwarz bias term.

4. The registered mechanism really was falsified
  PASS  H5 is falsified and the summary says so (0/48 bias-exceeds-radius)
  PASS  the summary records zero coverage violations

5. Sealed corpus untouched
  PASS  SCIENCE fixed_policy_expected_sarsa.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_variance_certificate.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE model.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_expected_sarsa.py matches this task's own record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py matches this task's own record
  PASS  SCIENCE fixed_policy_variance_certificate.py matches this task's own record
  PASS  SCIENCE model.py matches this task's own record
  PASS  evaluate_fp_iter2_001.py does not use the data-range certificate
  PASS  evaluate_fp_sample_001.py does not use the data-range certificate
  PASS  evaluate_fp_tight_001.py does not use the data-range certificate

6. Analyzer determinism
  PASS  analyze_fp_range_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

7. Replay of the foundation checks
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_policy_quantities_by_solve.py replays with exit 0 (got 0)

====================================================================================
SUMMARY
  1x items=  1048576 data_range=0.2587 (+6.9%)  empBern=-19.2%
  2x items=  2097152 data_range=0.1827 (-24.5%)  empBern=-44.7%
  4x items=  4194304 data_range=0.1369 (-43.4%)  empBern=-57.9%
  8x items=  8388608 data_range=0.1055 (-56.4%)  empBern=-66.1%
  unsound ceiling at 1x: -44.0%
  H2                  : PASS
  H6                  : PASS
  H3                  : PASS
  H4                  : PASS
  H5                  : FALSIFIED
====================================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round. Step 1 only; the ladder's rungs
are fresh independent samples, not supersets of the sealed batch.
