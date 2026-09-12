FP-ITER6-001 same-actor derived verification
==========================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Frozen inputs
  PASS  24 records (found 24)
  PASS  numpy horizon 6 (found 6)
  PASS  certification dimensions are the frozen 16384 x 64 (found 16384 x 64)
  PASS  12 tasks per mixing (found 12)
  PASS  both mixings run (found [0.08, 0.5])

2. H1 re-derived from the bundles
  PASS  every sealed row was compared (117 == 117)
  PASS  117 sealed route-step rows in levels 1--5 (found 117)
  PASS  H1 re-derived: no mismatch (found 0)

3. Batch reuse is structural
  PASS  both batch builders are present (found ['certification_batch', 'training_batch'])
  PASS  no batch is rebuilt inside the step loop ([])

4. Validity on the sixth step
  PASS  at least one sixth-step emission exists (9)
  PASS  zero sixth-step degrading violations (0)
  PASS  zero sixth-step non-positive gains (0)
  PASS  zero certificate violations over six steps (0)
  PASS  every abstention carries a frozen reason (0 missing)

5. The scored prediction is the frozen one
  PASS  the census prediction file still exists
  PASS  the scored prediction is the frozen file (sha256 70b79f03dad1acbc...)
  PASS  the prediction predates the sixth-step bundle
  PASS  the recorded creation time is the file's own

6. Corpus integrity
  PASS  evaluator evaluate_fp_attn_001.py matches its recorded hash
  PASS  evaluator evaluate_fp_scale_002.py matches its recorded hash
  PASS  SCIENCE fixed_policy_expected_sarsa.py matches its recorded hash
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py matches its recorded hash
  PASS  SCIENCE fixed_policy_variance_certificate.py matches its recorded hash
  PASS  SCIENCE model.py matches its recorded hash
  PASS  only the shared evaluator changed, and only by the documented horizon extension (changed: [])
  PASS  SCIENCE fixed_policy_expected_sarsa.py is unchanged since FP-ITER5-001 was sealed
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py is unchanged since FP-ITER5-001 was sealed
  PASS  SCIENCE fixed_policy_variance_certificate.py is unchanged since FP-ITER5-001 was sealed
  PASS  SCIENCE model.py is unchanged since FP-ITER5-001 was sealed

7. Analyzer determinism
  PASS  analyze_fp_iter6_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

8. Replay of the affected sealed programs
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_fp_attn_iter_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_attn_iter4_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_iter4_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_iter5_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_census_001_same_actor.py replays with exit 0 (got 0)

==========================================================================
SUMMARY
  emissions by step      : [22, 20, 15, 12, 12, 9]
  emission deltas        : [-2, -5, -3, 0, -3]
  mean gains by step     : [2.717707, 2.277997, 1.286906, 0.8075, 0.361474, 0.167057]
  min gains by step      : [0.104197, 0.779902, 0.378197, 0.184902, 0.082395, 0.019347]
  H1_horizon_inert   : {'compared': 117, 'expected': 117, 'mismatches': 0}
  H2                 : PASS
  H3                 : PASS
  H4                 : PASS
  H7_attrition       : PASS
  H7b_plateau        : FALSIFIED
  H8_mean_decay      : PASS
  H9_non_vacuity     : FALSIFIED
  H5 P1 plateau rule     : FALSIFIED (3 misclassified)
  H6 P2 ratio rule       : FALSIFIED (3 misclassified)
==========================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round. The sixth step was run on the
numpy path only, so the network path is now one step behind.
