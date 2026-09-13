FP-GAP-001 same-actor derived verification
============================================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Frozen inputs
  PASS  24 records (found 24)
  PASS  the source bundle's hash matches the one recorded at measurement time

2. H1 re-derived: every trajectory reconstructs
  PASS  96 trajectories inspected (found 96)
  PASS  worst reconstruction discrepancy 4.441e-16 <= 1e-12
  PASS  the summary's worst discrepancy matches

3. H2 re-derived: v* is optimal and dominates
  PASS  v* satisfies the Bellman optimality equation (worst residual 8.882e-16)
  PASS  v* dominates v^pi_0 everywhere (0 violations)

4. H3 re-derived: the gap is monotone
  PASS  no rise in any gap trajectory (0 rises)

5. The headline number, re-derived independently
  38 long trajectories (>= 24 emitted steps): mean fraction of the initial gap closed 99.9717%
  median (last gain / remaining gap) 0.327; the decay factor is 1/(1+that) = 0.753, which is what a converging process shows as a constant
  PASS  long trajectories close more than 99% of the gap (99.9717%)
  PASS  the decay factor is a stable interior value, not a vanishing one (0.327)

6. The mis-specified metrics are recorded as such
  PASS  the summary records H4, H5 and H6 as falsified rather than recasting them
  PASS  the length-conditioned breakdown is recorded alongside the pooled verdicts
  PASS  every floor statement carries the policy-class caveat

7. Sealed corpus untouched
  PASS  SCIENCE fixed_policy_expected_sarsa.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_variance_certificate.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE model.py is byte-identical to the FP-ITER5-001 record

8. Analyzer determinism
  PASS  analyze_fp_gap_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

9. Replay of the foundation checks
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_policy_quantities_by_solve.py replays with exit 0 (got 0)

============================================================================================
SUMMARY
  H1 reconstruction : 4.441e-16
  H2 v* optimal     : worst Bellman residual 8.882e-16
  H3 gap monotone   : 0 rises
  long trajectories : 38 closing 99.9717% of the gap
  median gain / remaining gap : 0.327 (decay factor 0.753)
  registered verdicts: H4 FALSIFIED, H5 FALSIFIED, H6 FALSIFIED
============================================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round. The floor is a joint property of the
policy class and the iteration; this task does not decompose the two.
