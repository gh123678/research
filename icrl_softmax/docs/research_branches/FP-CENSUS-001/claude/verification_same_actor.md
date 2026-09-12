FP-CENSUS-001 same-actor derived verification
==========================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Frozen inputs
  PASS  24 records (found 24)
  PASS  census horizon 5 (found 5)
  PASS  certification dimensions are the frozen 16384 x 64 (found 16384 x 64)
  PASS  both mixings run (found [0.08, 0.5])
  PASS  12 tasks per mixing (found 12)

2. H0 re-derived from the bundles
  PASS  48 route-records in the census (found 48)
  PASS  no route differs in row count (found 0)
  PASS  no decision mismatch (found 0)
  PASS  no ordered-reason mismatch (found 0)
  PASS  no eta mismatch (found 0)
  PASS  no E_Q mismatch, by exact float equality (found 0)
  PASS  H0 re-derived exact on 48/48 route-records

3. Oracle confinement
  PASS  the certificate and rule are actually called (2 sites)
  PASS  no oracle name reaches a certificate or decision argument ([])
  PASS  the oracle side is still computed for the diagnostic
  PASS  the oracle block still declares its non-certificate purpose

4. The sealed reference is intact
  PASS  the sealed FP-ITER5-001 bundle has not changed since the census ran (0 drifts)
  INFO  limitation: this shows the reference has not changed SINCE the census, not that it was never touched before it.
  INFO  FP-ITER5-001 numpy bundle sha256 = 3673719c70b57da9086983b55e9c0db0fb2a26980f18a59f91bcc937f2135237

5. The step-6 prediction is frozen and self-consistent
  PASS  prediction_step6.json exists
  PASS  theta equals the lowest attaining step-1 threshold (2.168690 == 2.168690)
  PASS  theta's misclassification count matches the step-1 scan
  PASS  P2 covers the whole scoring population (12 == 12)
  PASS  P2's predicted emitter count is consistent with its own rows (12)
  PASS  P1's emitter list is exactly the step-5 emitters in the population (12 == 12)
  PASS  the prediction predates the sixth-step bundle
  INFO  prediction sha256 = 70b79f03dad1acbc689367ba3ea5af7123d875959c1d006bd7b035a9668961c7

6. The frozen-batch guard
  PASS  verify_census_batch_frozen.py exits 0 (got 0)
  INFO  PASS: the census batch reproduces the sealed pair-count vector exactly on all 4 probes

7. Corpus integrity
  PASS  SCIENCE fixed_policy_expected_sarsa.py matches the hash recorded at census time
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py matches the hash recorded at census time
  PASS  SCIENCE fixed_policy_variance_certificate.py matches the hash recorded at census time
  PASS  SCIENCE model.py matches the hash recorded at census time

8. Analyzer determinism
  PASS  analyze_fp_census_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

==========================================================================
SUMMARY
  H0 sealed reproduction : exact on 48/48
  H1                     : PASS
  H2                     : FALSIFIED
  H3                     : PASS
  H4                     : PASS
  H5                     : PASS
  H6                     : PASS
==========================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round.
