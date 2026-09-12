FP-ATTN-ITER4-001 same-actor derived verification
==========================================================================
NOTE: the user instructed on 2026-09-11 that verification is not the
focus of this round. These checks are recorded for completeness and are
same-actor only.

1. Frozen inputs
  PASS  24 records (found 24)
  PASS  48 route-records (found 48)
  PASS  MAX_STEPS is 4 (found 4)
  PASS  ATOL is the frozen 1e-4 (found 0.0001)
  PASS  expected_finite maps to the finite literal network

2. H1/H2: the network horizon change is inert
  PASS  90 step-1..3 entries compared (found 90)
  PASS  network steps 1-3 identical to sealed (0 differ)
  PASS  three-step routes 15 == sealed 15

3. H3: the network produced every Qhat
  PASS  producer set is exactly the literal network (found {'literal_attention_network'})
  PASS  105 network step executions (found 105)

4. H4/H5: four steps and decision agreement
  PASS  per-level emissions [22, 20, 15, 12] equal numpy [22, 20, 15, 12]
  PASS  12 fourth-step network emissions (found 12)
  PASS  105 step comparisons (found 105)
  PASS  all decisions match numpy (0 differ)
  PASS  all eta selections match (0 differ)

5. H6: drift does not accumulate at four steps
  PASS  step 1 max |dQ| 1.076e-05 <= ATOL 1e-04
  PASS  step 2 max |dQ| 4.585e-06 <= ATOL 1e-04
  PASS  step 3 max |dQ| 7.176e-06 <= ATOL 1e-04
  PASS  step 4 max |dQ| 1.044e-05 <= ATOL 1e-04
  PASS  the summary records H6 as PASS

6. H7/H8: flips enumerated, validity under the network
  PASS  summary flip count 0 equals record level 0
  PASS  zero certificate violations (0)
  PASS  zero non-degrading violations (0)
  PASS  all reasons frozen (offenders set())

7. Corpus integrity
  PASS  evaluator evaluate_fp_attn_001.py matches its recorded hash
  PASS  evaluator evaluate_fp_iter2_001.py matches its recorded hash
  PASS  evaluator evaluate_fp_scale_002.py matches its recorded hash
  PASS  SCIENCE fixed_policy_expected_sarsa.py matches its recorded hash
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py matches its recorded hash
  PASS  SCIENCE fixed_policy_variance_certificate.py matches its recorded hash
  PASS  SCIENCE model.py matches its recorded hash
  PASS  scientific corpus present

8. Replay of the sealed programs
  PASS  analyze_fp_attn_iter4_001.py replays with exit 0 (got 0)
  PASS  analyze_fp_iter4_001.py replays with exit 0 (got 0)
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)

==========================================================================
SUMMARY
  route-records                 : 48
  network step executions       : 105
  emissions by step (network)   : [22, 20, 15, 12]
  emissions by step (numpy)     : [22, 20, 15, 12]
  network four-step routes      : 12
  decision agreement vs numpy   : 105/105
  decision flips                : 0
  q-gap 1 / 2 / 3 / 4           : 1.076e-05 / 4.585e-06 / 7.176e-06 / 1.044e-05
  certificate violations        : 0
  non-degrading violations      : 0
==========================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round. No second actor reconstructed
this route.
