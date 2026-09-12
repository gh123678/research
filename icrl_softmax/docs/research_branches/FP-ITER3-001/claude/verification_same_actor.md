FP-ITER3-001 same-actor derived verification
======================================================================

1. Frozen inputs
  PASS  24 records (found 24)
  PASS  48 route-records (found 48)
  PASS  MAX_STEPS is 3 (found 3)
  PASS  config records the frozen horizon 3
  PASS  both frozen mixing settings

2. H1/H2: the horizon change is inert
  PASS  step-1 emissions 22 equals sealed 22
  PASS  step-2 emissions 20 equals sealed 20
  PASS  step-2 mean gain 2.2779972042823244 equals sealed 2.2779972042823244
  PASS  step-2 minimum gain 0.7799019383638572 equals sealed 0.7799019383638572
  PASS  step-1 mean gain 2.71770715611645 equals sealed 2.71770715611645
  PASS  compared every comparable step-1/step-2 entry (70 vs expected 70)
  PASS  at least one entry was compared
  PASS  steps 1 and 2 are bit-identical to the sealed FP-ITER2-001 run (0 differ)

3. Per-level attrition and validity
  PASS  15 third-step emissions (found 15)
  PASS  H6 PASS: n3=15 < n2=20
  PASS  step-3 minimum gain 0.37819659359698987
  PASS  H7 FALSIFIED as reported: min3=0.378197 < min2=0.779902
  PASS  every third-step gain is strictly positive
  PASS  the summary reports H7 as FALSIFIED rather than reinterpreting it

4. Violations recomputed
  PASS  zero certificate violations (0)
  PASS  zero non-degrading violations (0)

5. Attrition reasons from the frozen list
  PASS  all reasons frozen (offenders set())

6. No sealed file changed
  PASS  evaluate_fp_attn_001.py matches its recorded hash
  PASS  evaluate_fp_scale_002.py matches its recorded hash
  PASS  fixed_policy_expected_sarsa.py matches its recorded hash
  PASS  fixed_policy_expected_sarsa_scaled.py matches its recorded hash
  PASS  fixed_policy_variance_certificate.py matches its recorded hash
  PASS  model.py matches its recorded hash

7. Replay of the sealed programs
  PASS  analyze_fp_iter3_001.py replays with exit 0 (got 0)
  PASS  analyze_fp_iter2_001.py replays with exit 0 (got 0)
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_fp_scale_002_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_attn_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_iter2_001_same_actor.py replays with exit 0 (got 0)

======================================================================
SUMMARY
  route-records                : 48
  step-1 emissions             : 22 (sealed 22)
  step-2 emissions             : 20 (sealed 20)
  step-3 emissions             : 15
  emitted all three steps      : 15
  mean gain 1 / 2 / 3          : 2.717707 / 2.277997 / 1.286906
  min  gain 1 / 2 / 3          : 0.104197 / 0.779902 / 0.378197
  certificate violations       : 0
  non-degrading violations     : 0
  H6 attrition                 : PASS (n3 < n2)
  H7 filtering                 : FALSIFIED (min3 < min2)
======================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only. No second actor
reconstructed this route; this is not reciprocal verification.
