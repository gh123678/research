# Plan: FP-SCALE-002 variance-adaptive certificate

Date: 2026-09-11.
Task: `docs/research_tasks/FP-SCALE-002.md`.
Design: `docs/superpowers/specs/2026-09-11-variance-adaptive-certificate-design.md`.

## Stage 0: pre-review and freeze

1. Record the direction and single-actor rulings in the task sheet.
2. Write the same-actor pre-review, checking inheritance fidelity, that the
   concentration argument is the only changed ingredient, that every constant
   traces to a named inequality, and that the prototype's `SAFETY = 1.1`
   assertion is gone.
3. Freeze the task revision and open `claude/FP-SCALE-002`.

## Stage 1: tests first

1. Write `verify_variance_adaptive_certificate.py` before the implementation.
2. Sections required: sealed-module identity; no free scaling factor with each
   constant reconstructed by hand; two-half disjointness, exhaustiveness, and
   determinism; scale-depends-only-on-A and mean-depends-only-on-B; certificate
   formula; exact-route parity at `1e-12`; no mask and no visited gate;
   count-rule abstention with the frozen reason; decision rule emission,
   abstention, and bit-for-bit return; structural oracle separation; and a
   validity check that `E_Q` bounds a realized error computed from an
   independently solved model truth.
3. Record the red-first run.

## Stage 2: implementation

1. `fixed_policy_variance_certificate.py`: the two-half certificate with
   explicit Hoeffding constants and the `2B` envelope control for `H5`.
2. `evaluate_fp_scale_002.py`: the frozen evaluator with the three routes, the
   strict-JSON bundle, and the oracle audit.
3. Reuse the sealed `fixed_policy_expected_sarsa.py` by import; assert its
   SHA-256 in `environment.json` so a silent edit cannot pass unnoticed.
4. Confirm the inherited verifiers still pass unchanged.

## Stage 3: smoke

1. Run the labelled smoke matrix: `2` tasks, both mixing settings, real
   certification batches so the emission path is exercised.
2. Measure wall time and per-pair certification counts; project the formal
   cost and check it against budget.
3. Seal implementation and smoke with a `[claude]` commit before the formal
   run.

## Stage 4: formal run

1. Execute the frozen matrix exactly once into
   `results/FP-SCALE-002/claude/formal/`.
2. Post-run checks: strict-JSON reload, duplicate-key detection, finiteness,
   shape and range checks, oracle separation, and task-scoped Ruff.
3. Seal the formal result with raw hashes, commands, and the failure history.

## Stage 5: derived verification (same actor, by user ruling)

1. Recompute, from the sealed records only and through an independent code
   path: every scale, radius, `E_Q`, lower bound, emitted policy, emission
   count, and the `H5` attribution ratio.
2. Replay the sealed verifier and evaluator and confirm exit codes.
3. Compare against the sealed bundle and report any deviation.
4. Write `verification_same_actor.md` ending `PASS`, `FAIL`, or `OBJECTION`,
   stating explicitly that no second actor reconstructed the route.

## Stage 6: closure

1. Write the route report, theory note, and failure history.
2. Update `ACTIVE_WORKSPACE.md`.
3. Request user approval before any merge to `main`.

## Gates

- Gate A: pre-review recorded, no unresolved objection.
- Gate B: smoke passes and the projected formal cost is within budget.
- Gate C: exactly one formal execution.
- Gate D: derived verification recorded.
- Gate E: user approval before merge.
