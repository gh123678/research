FP-HORIZON-001 same-actor derived verification
============================================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Frozen inputs
  PASS  48 route-records (found 48)
  PASS  horizon 32 (found 32)
  PASS  the bundle carries this task's identity (found FP-HORIZON-001)
  PASS  certification is still 8x (found 131072 chains)

2. H1/H2 re-derived: validity and soundness
  PASS  frozen: all 751 emitted steps are componentwise non-degrading
  PASS  frozen: every emitted step has a strictly positive total gain
  PASS  frozen: zero certificate violations over 751 emitted steps
  PASS  frozen: every abstention carries a frozen reason
  PASS  empirical_bernstein: all 827 emitted steps are componentwise non-degrading
  PASS  empirical_bernstein: every emitted step has a strictly positive total gain
  PASS  empirical_bernstein: zero certificate violations over 827 emitted steps
  PASS  empirical_bernstein: every abstention carries a frozen reason

3. The stopping claim re-derived
  PASS  the deepest trajectory 32 matches the summary
  PASS  the last emitting step 32 matches the summary
  PASS  the H3 verdict is consistent with the data (last step 32, horizon 32)
  INFO  an emission occurs at the last step, so termination has NOT been demonstrated; the horizon is still binding.

4. The population is re-derived
  PASS  frozen: the emission sequence matches the summary
  INFO  frozen: 0 rises in the population
  PASS  empirical_bernstein: the emission sequence matches the summary
  INFO  empirical_bernstein: 0 rises in the population

5. The value gain is re-derived from the recorded values
  PASS  start-to-final per-state value never decreases on any record (0 exceptions)
  PASS  no trajectory loses total value (96 trajectories, min 0.000000)
  PASS  the 10 zero-gain trajectories are exactly the 10 that never emitted
  PASS  86 trajectories gain strictly positive total value

6. Sealed corpus untouched
  PASS  SCIENCE fixed_policy_expected_sarsa.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_variance_certificate.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE model.py is byte-identical to the FP-ITER5-001 record

7. Analyzer determinism
  PASS  analyze_fp_horizon_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

8. Replay of the foundation checks
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_policy_quantities_by_solve.py replays with exit 0 (got 0)

============================================================================================
SUMMARY
  emissions, frozen : [42, 42, 40, 40, 38, 36, 33, 29, 28, 27, 25, 22, 20, 20, 20, 20, 20, 20, 20, 20, 18, 18, 18, 18, 18, 18, 17, 14, 14, 12, 12, 12]
  emissions, empB   : [44, 44, 44, 42, 41, 38, 35, 34, 31, 29, 27, 26, 24, 23, 22, 22, 22, 22, 22, 22, 20, 20, 20, 20, 20, 20, 19, 16, 16, 14, 14, 14]
  deepest trajectory: {'empirical_bernstein': 32, 'frozen': 32}
  last emitting step: 32
  H3 terminates     : FALSIFIED
  H5 decay past 12  : PASS
  H6 positive gains : PASS
  total value gain, frozen              : mean 5.649893, worst state mean 0.838256
  total value gain, empirical_bernstein : mean 5.803923, worst state mean 0.860150
============================================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round. The 8x batch is a fresh independent
sample, not a superset of the sealed one.
