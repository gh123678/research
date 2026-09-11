FP-SCALE-002 same-actor derived verification
============================================================

1. Frozen inputs and sealed-module identity
  PASS  record count is 24 (found 24)
  PASS  config parses
  PASS  mixing settings match the frozen list
  PASS  tasks per cell is 12
  PASS  training length is 65536
  PASS  certification batch is 16384 x 64 = 1048576
  PASS  half-count floor is 5000
  PASS  delta is 0.05
  PASS  sealed FP-ESARSA-001 module is byte-identical to its sealed version
  PASS  certificate module is unchanged since the formal run

2. Certificate constants recomputed independently
  PASS  envelope 2B = 10.000000 matches R_star/(1-gamma)*2
  PASS  per-pair per-half risk is delta/(2d) = 0.00208333

3. E_Q recomputed from serialized residual means and radii
  PASS  every E_Q equals max_x(|Ybar_x| + r_x)/(1-gamma) (0 mismatches)
  PASS  every E_Q bounds the realized oracle error (0 violations)

4. Radius bound recomputed from the reported scales
  PASS  every radius is positive and below 2*s_x (0 anomalies)

5. Emission accounting recomputed
  PASS  attempted 48 matches summary
  PASS  emissions 22 matches summary
  PASS  non-degrading 22 matches summary
  PASS  strict 22 matches summary
  PASS  control emissions 1 matches summary
  PASS  violation count matches
  PASS  control/adaptive ratio range 3.448-5.457 matches summary
  PASS  the envelope control is worse in EVERY record (H5)
  PASS  H3: 22 emissions >= 12
  PASS  H4: every emission is non-degrading and strictly improving
  PASS  every selected eta [0.1, 1.0] lies in the frozen candidate grid
  PASS  selected eta [0.1, 1.0]: all within the inherited grid, so the downward extension was not the enabler in this run
  PASS  eta histogram accounts for every emission exactly once

6. Replay of the sealed programs
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  analyze_fp_scale_002.py replays with exit 0 (got 0)

============================================================
RESULT: PASS (all derived checks passed)
LIMITATION: same-actor derived verification only. No second actor
reconstructed this route; this is not reciprocal verification.
