FP-SAMPLE-001 same-actor derived verification
====================================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Frozen inputs
  PASS  48 route-records (found 48)
  PASS  the registered rungs (found [1, 2, 4, 8])
  PASS  item counts scale exactly with the rung ({1: 1048576, 2: 2097152, 4: 4194304, 8: 8388608})

2. Coverage re-derived, every rung, both arms
  PASS  empirical_bernstein: 192 certificate-fits, 0 coverage violations
  PASS  bernstein: 192 certificate-fits, 0 coverage violations

3. The revival counts, re-derived from the bundles
  PASS  the FP-TIGHT-001 frozen baseline has 26 abstainers (found 26)
  revived by rung: {1: 8, 2: 13, 4: 18, 8: 22}
  PASS  the summary's revival counts match a fresh re-derivation
  PASS  the revival counts are non-decreasing ([8, 13, 18, 22])
  PASS  no route-record loses eligibility as data grows (0 lost)

4. The sampler gate re-runs
  PASS  fp_sample_vectorised_batch.py exits 0 (got 0)
  INFO  ==============================================================================
  INFO  worst discrepancy 0.0091; a sampler that were wrong would shift E_Q
  INFO  and invalidate the whole ladder, so this gate matters.

5. Sealed corpus untouched
  PASS  SCIENCE fixed_policy_expected_sarsa.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_variance_certificate.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE model.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_expected_sarsa.py matches this task's own record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py matches this task's own record
  PASS  SCIENCE fixed_policy_variance_certificate.py matches this task's own record
  PASS  SCIENCE model.py matches this task's own record
  PASS  evaluate_fp_iter2_001.py does not import the vectorised sampler
  PASS  evaluate_fp_attn_iter_001.py does not import the vectorised sampler
  PASS  evaluate_fp_tight_001.py does not import the vectorised sampler
  PASS  evaluate_fp_census_001.py does not import the vectorised sampler

6. Analyzer determinism
  PASS  analyze_fp_sample_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

7. Replay of the affected sealed programs
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_policy_quantities_by_solve.py replays with exit 0 (got 0)

====================================================================================
SUMMARY
  1x items=  1048576 mean E_Q=0.1955 (+0.0%)  revived= 8/26
  2x items=  2097152 mean E_Q=0.1339 (-31.5%)  revived=13/26
  4x items=  4194304 mean E_Q=0.1018 (-47.9%)  revived=18/26
  8x items=  8388608 mean E_Q=0.0821 (-58.0%)  revived=22/26
  H1_sampler_gate   : PASS
  H4                : PASS
  H6a               : PASS
  H6c               : PASS
  H7                : PASS
  H3                : PASS
  H5                : FALSIFIED
  H6b               : PASS
====================================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round. Step 1 only; the ladder's rungs
are fresh independent samples, not supersets of the sealed batch.
