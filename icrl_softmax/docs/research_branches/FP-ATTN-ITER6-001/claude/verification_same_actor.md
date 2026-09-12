FP-ATTN-ITER6-001 same-actor derived verification
==========================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Frozen inputs
  PASS  24 records (found 24)
  PASS  network horizon 6 (found 6)
  PASS  ATOL is 1e-4 (found 0.0001)
  PASS  certification dimensions are 16384 x 64 (found 16384 x 64)
  PASS  12 tasks per mixing (found 12)
  INFO  torch 2.11.0+cpu

2. H1 re-derived from the bundles
  PASS  every sealed row compared (117 == 117)
  PASS  117 sealed rows in levels 1--5 (found 117)
  PASS  no decision/eta/E_Q/reason mismatch (0)
  PASS  every recorded Qhat gap reproduced exactly (0 mismatches)

3. Provenance: the decision is taken on the network's Qhat
  PASS  found both driving call sites (2)
  PASS  the certificate and the decision are both taken on the literal network's Qhat ([])
  PASS  the numpy comparator is still computed, so the confinement check is not vacuous
  PASS  every one of 129 steps records the literal network as producer (found {'literal_attention_network'})
  PASS  the sixth step exists (12 rows)

4. Batch reuse is structural
  PASS  both batch builders are present (found ['certification_batch', 'training_batch'])
  PASS  no batch rebuilt inside the step loop ([])

5. Validity over six network steps
  PASS  at least one sixth-step emission exists (9)
  PASS  zero sixth-step degrading violations (0)
  PASS  zero sixth-step non-positive gains (0)
  PASS  zero certificate violations over six steps (0)
  PASS  no emitted step degrades any state (0)
  PASS  every abstention carries a frozen reason (0 missing)

6. Path agreement, level by level
  PASS  step 1 emitting sets agree (22 vs 22)
  PASS  step 2 emitting sets agree (20 vs 20)
  PASS  step 3 emitting sets agree (15 vs 15)
  PASS  step 4 emitting sets agree (12 vs 12)
  PASS  step 5 emitting sets agree (12 vs 12)
  PASS  step 6 emitting sets agree (9 vs 9)
  PASS  zero decision flips against the comparator (0)
  PASS  zero eta flips against the comparator (0)
  PASS  the worst network-versus-numpy gap over all steps is 1.076e-05 <= 1e-04
  PASS  every step shows a nonzero float32 gap, so no numpy substitution (minimum 6.319e-08)

7. Corpus integrity
  PASS  evaluator evaluate_fp_attn_001.py matches its recorded hash
  PASS  evaluator evaluate_fp_scale_002.py matches its recorded hash
  PASS  SCIENCE fixed_policy_expected_sarsa.py matches its recorded hash
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py matches its recorded hash
  PASS  SCIENCE fixed_policy_variance_certificate.py matches its recorded hash
  PASS  SCIENCE model.py matches its recorded hash
  INFO  shared evaluator evaluate_fp_iter2_001.py changed after FP-ITER5-001 was sealed. It is not part of the scientific corpus; the horizon knob inside it was extended (FP-ITER6-001 added 6). The evolution is proven inert: a re-run at the frozen horizon reproduces all 117 sealed step entries exactly.
  INFO  evaluate_fp_attn_iter_001.py is recorded by no sealed bundle; its current sha256 is c932ca8d43dfa14241b7cd905089cea9f7b3ba1b8636b39ae24a17208e9751f5
  PASS  only documented task evaluators changed (changed: ['evaluate_fp_iter2_001.py'])

8. Analyzer determinism
  PASS  analyze_fp_attn_iter6_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

9. Replay of the affected sealed programs
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_fp_attn_iter_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_attn_iter4_001_same_actor.py replays with exit 0 (got 0)
  PASS  verify_fp_iter5_001_same_actor.py replays with exit 0 (got 0)

==========================================================================
SUMMARY
  emissions by step, network : [22, 20, 15, 12, 12, 9]
  emissions by step, numpy   : [22, 20, 15, 12, 12, 9]
  mean gains, network        : [2.717707, 2.277997, 1.286905, 0.807499, 0.361474, 0.167057]
  min gains, network         : [0.104197, 0.779902, 0.378197, 0.184902, 0.082395, 0.019347]
  max |dQ| by step           : ['1.076e-05', '4.585e-06', '7.176e-06', '1.044e-05', '4.567e-06', '6.814e-06']
  H1 network horizon inert   : {'compared': 117, 'expected': 117, 'gap_mismatches': 0, 'mismatches': 0}
  H2                         : PASS
  H3                         : PASS
  H4                         : PASS
  H8_count_matches_numpy     : PASS
  H9_mean_decay              : PASS
  H5_path_agreement_step6    : PASS
  H6_drift                   : PASS
  H7_no_flips                : PASS
==========================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round.
