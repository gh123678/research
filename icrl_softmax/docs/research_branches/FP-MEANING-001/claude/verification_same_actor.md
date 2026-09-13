FP-MEANING-001 same-actor derived verification
================================================================================================
NOTE: the user instructed that verification is not the focus of this
round. These checks are recorded for completeness and are same-actor only.

1. The floor re-derived from the raw value vectors
  PASS  the recorded gaps are reproduced from the value vectors
  PASS  the summary's floor 2.287e-14 is the max re-derived gap
  PASS  the two routes agree to 2.287e-14, so the floor is a floor
  PASS  no record agrees exactly, so the two methods are not the same code (minimum 1.643e-14)

2. The certified-bound crossings re-derived
  PASS  the crossing against the measured floor is step 18, matching the summary
  PASS  the crossing against machine epsilon is step 26, matching the summary
  PASS  the iteration is still emitting 12 route-records at step 32, so it runs past both crossings
  PASS  every recorded certified bound is strictly positive, so a '> 0' check passes throughout -- the blind spot this task identifies

3. The task made no definitional choice
  PASS  the summary records that no definition was chosen
  PASS  the reach table reports 5 definitions
  PASS  the definitions genuinely disagree (first-failure steps [2, 18, 26])
  INFO  the disagreement is the point: reporting a single horizon without naming the definition would hide it.

4. Sealed corpus untouched, and no iteration was re-run
  PASS  SCIENCE fixed_policy_expected_sarsa.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_expected_sarsa_scaled.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE fixed_policy_variance_certificate.py is byte-identical to the FP-ITER5-001 record
  PASS  SCIENCE model.py is byte-identical to the FP-ITER5-001 record
  PASS  evaluate_fp_iter8x_001.py is present and unmodified by this task
  PASS  evaluate_fp_attn_8x_001.py is present and unmodified by this task

5. Analyzer determinism
  PASS  analyze_fp_meaning_001.py replays with exit 0 (got 0)
  PASS  re-running the analyzer reproduces summary.json byte-for-byte

6. Replay of the foundation checks
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_policy_quantities_by_solve.py replays with exit 0 (got 0)

================================================================================================
SUMMARY
  measured floor        : 2.287e-14
  machine epsilon       : 2.220e-16
  certified bound step 1: 3.165e-04
  certified bound step 32: 2.164e-16
  first crossing vs floor: step 18
  first crossing vs eps  : step 26
  D1 realized gain > 0          : first failure None, holds 32/32
  D2 certified bound > 0        : first failure None, holds 32/32
  D3 certified > method floor   : first failure 18, holds 20/32
  D4 certified > machine eps    : first failure 26, holds 30/32
  D5 certified > float32 gap    : first failure 2, holds 5/32
================================================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only, and per the user's
instruction not the focus of this round. No new experiment was run; the
trajectory analysed is FP-HORIZON-001's.
