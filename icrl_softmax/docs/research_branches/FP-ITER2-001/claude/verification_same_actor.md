FP-ITER2-001 same-actor derived verification
====================================================================

1. Frozen inputs
  PASS  24 records (found 24)
  PASS  48 route-records (found 48)
  PASS  MAX_STEPS is 2
  PASS  both frozen mixing settings
  PASS  FP-SCALE-002 certification constants (16384 x 64)

2. Step 1 reproduces the sealed FP-SCALE-002 result
  PASS  step-1 reproduction checked on all 48 (found 48)
  PASS  every step-1 decision, eta and E_Q matches sealed (0 bad)
  PASS  step-1 emissions 22 equals the sealed 22

3. Attrition and validity recomputed
  PASS  20 route-records emit twice (found 20)
  PASS  one-step count 2 matches
  PASS  zero-step count 26 matches
  PASS  attrition accounting closes: two-step + one-step == step-1 emissions
  PASS  step-2 survival 20/22 exceeds half
  PASS  zero certificate violations (0 found)
  PASS  every emitted step is componentwise non-degrading (0 bad)
  PASS  22 step-1 gains recorded (found 22)
  PASS  20 step-2 gains recorded (found 20)
  PASS  every step-2 gain is strictly positive
  PASS  mean step-2 gain 2.277997 matches summary
  PASS  minimum step-2 gain 0.779902 matches summary

4. Attrition reasons are from the frozen list
  PASS  every abstention reason is in the frozen list (offenders: set())
  PASS  the only stopping reason observed is improvement_lcb_nonpositive (found {'improvement_lcb_nonpositive'})

5. H5: no sealed file changed
  PASS  evaluate_fp_attn_001.py byte-identical
  PASS  evaluate_fp_scale_002.py byte-identical
  PASS  fixed_policy_expected_sarsa.py byte-identical
  PASS  fixed_policy_expected_sarsa_scaled.py byte-identical
  PASS  fixed_policy_variance_certificate.py byte-identical
  PASS  model.py byte-identical

6. Replay of the sealed programs
  PASS  analyze_fp_iter2_001.py replays with exit 0 (got 0)
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_fp_scale_002_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_attn_001_same_actor.py replays with exit 0 (got 0)

====================================================================
SUMMARY
  route-records                 : 48
  step-1 emissions              : 22 (sealed 22)
  step-1 reproduction failures  : 0
  emitted two steps             : 20
  emitted one step              : 2
  emitted zero steps            : 26
  step-2 survival of step-1     : 20/22
  mean gain step 1 / step 2     : 2.717707 / 2.277997
  min  gain step 1 / step 2     : 0.104197 / 0.779902
  certificate violations        : 0
  non-degrading violations      : 0
====================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only. No second actor
reconstructed this route; this is not reciprocal verification.
