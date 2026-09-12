FP-ITER4-001 same-actor derived verification
========================================================================

1. Frozen inputs
  PASS  24 records (found 24)
  PASS  48 route-records (found 48)
  PASS  MAX_STEPS is 4 (found 4)
  PASS  config records the frozen horizon 4
  PASS  both frozen mixing settings

2. H1/H2: the horizon change is inert
  PASS  90 step-1..3 entries compared (found 90)
  PASS  steps 1-3 are bit-identical to the sealed FP-ITER3-001 run (0 differ)
  PASS  the summary records H1 as PASS

3. Per-level pattern recomputed
  PASS  step 1 emissions 22 equal the sealed 22
  PASS  step 2 emissions 20 equal the sealed 20
  PASS  step 3 emissions 15 equal the sealed 15
  PASS  12 fourth-step emissions (found 12)

4. The three pre-registered step-4 predictions
  PASS  H7 PASS: n4=12 < n3=15
  PASS  H8 PASS: mean4=0.807499658432434 < mean3=1.2869055664197442
  PASS  H9 PASS: min4=0.1849020565541281 > 0.05 (non-vacuous)
  PASS  the summary records all three predictions as PASS
  PASS  H3 PASS: a fourth step is certifiable

5. Validity and monotonicity
  PASS  zero certificate violations (0)
  PASS  zero non-degrading violations (0)
  PASS  every step-1 gain is strictly positive (22 emissions)
  PASS  every step-2 gain is strictly positive (20 emissions)
  PASS  every step-3 gain is strictly positive (15 emissions)
  PASS  every step-4 gain is strictly positive (12 emissions)

6. Attrition reasons and batches
  PASS  all reasons frozen (offenders set())
  PASS  each record has its own training batch digest
  PASS  every record records its certification batch digest
  PASS  no step-1 reproduction failure

7. No sealed file changed
  PASS  evaluate_fp_attn_001.py matches its recorded hash
  PASS  evaluate_fp_scale_002.py matches its recorded hash
  PASS  fixed_policy_expected_sarsa.py matches its recorded hash
  PASS  fixed_policy_expected_sarsa_scaled.py matches its recorded hash
  PASS  fixed_policy_variance_certificate.py matches its recorded hash
  PASS  model.py matches its recorded hash

8. Replay of the sealed programs
  PASS  analyze_fp_iter4_001.py replays with exit 0 (got 0)
  PASS  analyze_fp_iter3_001.py replays with exit 0 (got 0)
  PASS  verify_fp_iter3_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_attn_iter_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)

========================================================================
SUMMARY
  route-records            : 48
  step executions          : 105
  emitted all four steps   : 12
  step 1: emissions  22  mean gain 2.717707  min gain 0.104197
  step 2: emissions  20  mean gain 2.277997  min gain 0.779902
  step 3: emissions  15  mean gain 1.286906  min gain 0.378197
  step 4: emissions  12  mean gain 0.807500  min gain 0.184902
  certificate violations   : 0
  non-degrading violations : 0
  H1 inert horizon         : PASS (90/90 entries identical)
  H3 fourth step           : PASS (n4=12)
  H7 attrition  n4 < n3    : PASS (12 < 15)
  H8 mean decay            : PASS (0.807500 < 1.286906)
  H9 non-vacuity           : PASS (min4=0.184902 > 0.05)
========================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only. No second actor
reconstructed this route; this is not reciprocal verification.
