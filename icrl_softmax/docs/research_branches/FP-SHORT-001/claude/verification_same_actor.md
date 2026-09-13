FP-SHORT-001 same-actor derived verification
============================================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. Frozen inputs
  PASS  48 route-records (found 48)
  PASS  the frozen grid is the protocol grid (found [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
  PASS  the frozen grid is a descending PREFIX of the extended grid, which is what makes the inertness claim a theorem

2. H1 re-derived: the instrumented path is the frozen rule
  PASS  all 428 steps faithful (0 mismatches)
  PASS  the frozen-grid arm reproduces the sealed FP-ITER8X-001 frozen arm (0 divergences)

3. H2 re-derived: the extension is inert
  PASS  no emitting decision changes (0 counterexamples)

4. H5/H6 re-derived: the grid is not the constraint
  PASS  the two arms emit the same number of steps (402 vs 402)
  PASS  no trajectory changes length (0)
  PASS  the summary records both grid hypotheses as falsified

5. The gate diagnosis re-derived
  PASS  the short-trajectory count matches the summary (12)
  PASS  every short trajectory has E_Q/h >= 1, i.e. the margin really is the binding quantity (min 1.0009)
  PASS  exactly one state blocks in every short trajectory (distribution [1])
  PASS  the summary records H3 and H4 as passing

6. The extended grid did not leak into a protocol path
  PASS  evaluate_fp_iter2_001.py does not use the extended grid
  PASS  evaluate_fp_attn_iter_001.py does not use the extended grid
  PASS  evaluate_fp_iter8x_001.py does not use the extended grid
  PASS  evaluate_fp_attn_8x_001.py does not use the extended grid
  PASS  the frozen eta grid in the sealed module is unchanged

7. Sealed corpus untouched
  PASS  SCIENCE fixed_policy_expected_sarsa.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_variance_certificate.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE model.py is byte-identical to the FP-ITER5-001 record

8. Analyzer determinism
  PASS  analyze_fp_short_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

9. Replay of the foundation checks
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_policy_quantities_by_solve.py replays with exit 0 (got 0)

============================================================================================
SUMMARY
  short trajectories      : 12
  E_Q/h median            : 1.1211 (min 1.0009, max 1.7392)
  blocking-state counts   : [1]
  emitted steps frozen    : 402
  emitted steps extended  : 402
  H1_unfaithful           : 0
  H2_changed_decisions    : 0
  H3                      : PASS
  H4                      : PASS
  H5                      : FALSIFIED
  H6                      : FALSIFIED
============================================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round. The extended grid is a diagnostic,
not a proposed protocol.
