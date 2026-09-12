FP-ATTN-ITER-001 same-actor derived verification
========================================================================

1. Frozen inputs
  PASS  24 records (found 24)
  PASS  48 route-records (found 48)
  PASS  MAX_STEPS is 3 (found 3)
  PASS  ATOL is the frozen 1e-4 (found 0.0001)
  PASS  expected_exact maps to the masked literal network
  PASS  expected_finite maps to the finite literal network

2. H2: the network produced every step's Qhat
  PASS  every step's Qhat came from the literal network (found {'literal_attention_network'})
  PASS  90 step executions in total (found 90)

3. Network-vs-numpy agreement and the accumulation question
  PASS  step 1: max |Q_network - Q_numpy| 1.076e-05 <= ATOL
  PASS  step 2: max |Q_network - Q_numpy| 4.585e-06 <= ATOL
  PASS  step 3: max |Q_network - Q_numpy| 7.176e-06 <= ATOL
  PASS  summary step-1 max gap matches the record level (H1 anchor)
  PASS  the step-3 gap (7.176e-06) stays within ATOL, so the float32 difference does not accumulate past the frozen tolerance

4. H3/H4: three steps and decision agreement
  PASS  per-level emissions [22, 20, 15] equal the FP-ITER3-001 numpy baseline [22, 20, 15]
  PASS  15 network-driven three-step routes (found 15)
  PASS  90 step comparisons available (found 90)
  PASS  every decision matches the numpy baseline (0 disagree)
  PASS  every selected eta matches the numpy baseline (0 differ)

5. H5: flips are enumerated, not absorbed
  PASS  summary flip count 0 equals the record level 0
  PASS  every reported flip carries its Q gap and boundary margins

6. H6: validity under the network's own Qhat
  PASS  zero certificate violations (0)
  PASS  zero non-degrading violations (0)
  PASS  every emitted step at every level has a strictly positive total gain

7. Attrition reasons and batches
  PASS  all reasons frozen (offenders set())
  PASS  each record has its own training batch digest
  PASS  every record records its certification batch digest

8. H7: no sealed SCIENCE file changed
  PASS  evaluator evaluate_fp_attn_001.py matches its recorded hash
  PASS  evaluator evaluate_fp_scale_002.py matches its recorded hash
  PASS  SCIENCE fixed_policy_expected_sarsa.py matches its recorded hash
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py matches its recorded hash
  PASS  SCIENCE fixed_policy_variance_certificate.py matches its recorded hash
  PASS  SCIENCE model.py matches its recorded hash
  INFO  shared evaluator evaluate_fp_iter2_001.py changed after this task was sealed. It is not part of the scientific corpus; FP-ITER4-001 added a horizon value to its --max-steps choices. The evolution is proven inert: a re-run at the frozen horizon reproduces all 90 sealed step entries exactly.
  PASS  only the shared evaluator changed, and only by the documented horizon extension (changed: ['evaluate_fp_iter2_001.py'])

9. Finite route remains gate-free on this data
  PASS  the finite route still exposes no visited gate
  PASS  finite-route write attention remains strictly positive (no -inf mask)

10. Replay of the sealed programs
  PASS  analyze_fp_attn_iter_001.py replays with exit 0 (got 0)
  PASS  analyze_fp_iter3_001.py replays with exit 0 (got 0)
  PASS  verify_fp_iter3_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_attn_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)

========================================================================
SUMMARY
  route-records                    : 48
  step executions (network)        : 90
  emissions by step (network)      : [22, 20, 15]
  emissions by step (numpy)        : [22, 20, 15]
  network three-step routes        : 15
  max |Q_network - Q_numpy|        : 1.076e-05  (ATOL 1e-04)
  step gap 1 / 2 / 3               : 1.076e-05 / 4.585e-06 / 7.176e-06
  decision agreement vs numpy      : 90/90
  decision flips                   : 0
  certificate violations           : 0
  non-degrading violations         : 0
========================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only. No second actor
reconstructed this route; this is not reciprocal verification.
