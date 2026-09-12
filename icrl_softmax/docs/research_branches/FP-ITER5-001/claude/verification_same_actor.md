FP-ITER5-001 same-actor derived verification
==========================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Frozen inputs
  PASS  numpy 24 records (found 24)
  PASS  network 24 records (found 24)
  PASS  numpy MAX_STEPS 5 (found 5)
  PASS  network MAX_STEPS 5 (found 5)
  PASS  ATOL is the frozen 1e-4 (found 0.0001)

2. H1/H2: both horizon changes inert
  PASS  numpy horizon inert: 105 entries compared, 0 mismatches
  PASS  network horizon inert: 105 entries compared, 0 mismatches

3. H3/H7: fifth step and attrition
  PASS  numpy fifth step emits 12 (found 12)
  PASS  network fifth step emits 12 (found 12)
  PASS  the summary records H7 as FALSIFIED rather than reinterpreting it
  PASS  per-level emissions [22, 20, 15, 12, 12] match the recorded plateau
  PASS  the step-4 and step-5 emitting sets are identical (12 records)

4. H8/H9/H10: predictions and drift
  PASS  H8 PASS: mean5=0.361474 < mean4=0.807500
  PASS  H9 PASS: min5=0.082395 > 0.05
  PASS  H10 PASS: step-5 network gap 4.567e-06 <= 1e-04
  PASS  step 1 network gap 1.076e-05 <= ATOL
  PASS  step 2 network gap 4.585e-06 <= ATOL
  PASS  step 3 network gap 7.176e-06 <= ATOL
  PASS  step 4 network gap 1.044e-05 <= ATOL
  PASS  step 5 network gap 4.567e-06 <= ATOL

5. H11: path agreement
  PASS  step 1 emission counts agree (22 == 22)
  PASS  step 2 emission counts agree (20 == 20)
  PASS  step 3 emission counts agree (15 == 15)
  PASS  step 4 emission counts agree (12 == 12)
  PASS  step 5 emission counts agree (12 == 12)
  PASS  105 decision comparisons available (found 105)
  PASS  no decision disagreement with numpy (0)
  PASS  no eta disagreement with numpy (0)

6. H4/H5/H6: validity on both paths
  PASS  numpy: zero certificate violations (0)
  PASS  numpy: zero non-degrading violations (0)
  PASS  numpy: step-1 gains all positive
  PASS  numpy: step-2 gains all positive
  PASS  numpy: step-3 gains all positive
  PASS  numpy: step-4 gains all positive
  PASS  numpy: step-5 gains all positive
  PASS  network: zero certificate violations (0)
  PASS  network: zero non-degrading violations (0)
  PASS  network: step-1 gains all positive
  PASS  network: step-2 gains all positive
  PASS  network: step-3 gains all positive
  PASS  network: step-4 gains all positive
  PASS  network: step-5 gains all positive

7. Corpus integrity
  PASS  evaluator evaluate_fp_attn_001.py matches its recorded hash
  PASS  evaluator evaluate_fp_scale_002.py matches its recorded hash
  PASS  SCIENCE fixed_policy_expected_sarsa.py matches its recorded hash
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py matches its recorded hash
  PASS  SCIENCE fixed_policy_variance_certificate.py matches its recorded hash
  PASS  SCIENCE model.py matches its recorded hash
  INFO  shared evaluator evaluate_fp_iter2_001.py differs from this task's record; it is not part of the scientific corpus.

8. Replay of the sealed programs
  PASS  analyze_fp_iter5_001.py replays with exit 0 (got 0)
  PASS  verify_fp_attn_iter_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_attn_iter4_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_iter4_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)

==========================================================================
SUMMARY
  emissions by step, numpy   : [22, 20, 15, 12, 12]
  emissions by step, network : [22, 20, 15, 12, 12]
  mean gains, numpy          : [2.717707, 2.277997, 1.286906, 0.8075, 0.361474]
  min gains, numpy           : [0.104197, 0.779902, 0.378197, 0.184902, 0.082395]
  network |dQ| by step       : ['1.076e-05', '4.585e-06', '7.176e-06', '1.044e-05', '4.567e-06']
  five-step routes, numpy    : 12
  five-step routes, network  : 12
  decision agreement         : 105/105
  H7 attrition               : FALSIFIED (plateau at 12)
  H8 mean decay              : PASS
  H9 non-vacuity             : PASS
  H10 drift                  : PASS
  H11 path agreement         : PASS
==========================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round.
