FP-ATTN-8X-001 same-actor derived verification
============================================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Frozen inputs
  PASS  48 route-records (found 48)
  PASS  horizon 12 (found 12)
  PASS  certification is 8x (found 131072 chains)
  PASS  ATOL is the frozen 1e-4 (found 0.0001)
  INFO  torch 2.11.0+cpu

2. Provenance: structural, not asserted
  PASS  found the driving call sites (2)
  PASS  the certificate and the decision are taken on the literal network's Qhat ([])
  PASS  the numpy comparator still exists, so the confinement check is not vacuous
  PASS  all 885 step entries record the literal network (found {'literal_attention_network'})
  PASS  every step shows a nonzero float32 gap, so no numpy substitution (minimum 5.858e-08)

3. Validity and soundness re-derived
  PASS  frozen: 402 emitted steps, all componentwise non-degrading
  PASS  frozen: every emitted step strictly improving
  PASS  frozen: zero certificate violations
  PASS  empirical_bernstein: 435 emitted steps, all componentwise non-degrading
  PASS  empirical_bernstein: every emitted step strictly improving
  PASS  empirical_bernstein: zero certificate violations
  PASS  frozen: 0 abstentions without a reason
  PASS  empirical_bernstein: 0 abstentions without a reason

4. Path agreement re-derived against the numpy bundle
  PASS  the set-disagreement count 0 matches the summary
  PASS  the H4 verdict is consistent with the re-derived count
  PASS  the eta-flip count 0 matches the summary

5. Drift and headroom re-derived
  PASS  the worst gap 1.373e-05 matches the summary and is within ATOL
  PASS  the worst headroom 40.0x matches the summary
  INFO  headroom is the quantity that decides whether path agreement is a statement about the certificate or about float32 arithmetic.

6. Sealed corpus untouched
  PASS  SCIENCE fixed_policy_expected_sarsa.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_variance_certificate.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE model.py is byte-identical to the FP-ITER5-001 record
  PASS  evaluate_fp_iter2_001.py still uses only the sealed certificate
  PASS  evaluate_fp_attn_iter_001.py still uses only the sealed certificate
  PASS  evaluate_fp_census_001.py still uses only the sealed certificate

7. Analyzer determinism
  PASS  analyze_fp_attn_8x_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

8. Replay of the foundation checks
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_policy_quantities_by_solve.py replays with exit 0 (got 0)

============================================================================================
SUMMARY
  network frozen                : [42, 42, 40, 40, 38, 36, 33, 29, 28, 27, 25, 22]
  numpy   frozen                : [42, 42, 40, 40, 38, 36, 33, 29, 28, 27, 25, 22]
  network empirical_bernstein   : [44, 44, 44, 42, 41, 38, 35, 34, 31, 29, 27, 26]
  numpy   empirical_bernstein   : [44, 44, 44, 42, 41, 38, 35, 34, 31, 29, 27, 26]
  deepest: network {'empirical_bernstein': 12, 'frozen': 12}, numpy {'empirical_bernstein': 12, 'frozen': 12}
  H3 reach          : PASS
  H4 set agreement  : PASS (0 disagreements)
  H5 eta flips      : PASS (0)
  H6 drift          : PASS (1.373e-05)
  H7 headroom       : PASS (40.0x)
============================================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round. The 8x batch is a fresh independent
sample, not a superset of the sealed one.
