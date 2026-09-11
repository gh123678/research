# Plan: FP-SCALE-001 implementation and evidence

Date: 2026-09-11.
Task: `docs/research_tasks/FP-SCALE-001.md`.
Design: `docs/superpowers/specs/2026-09-11-reachable-certificate-scale-design.md`.

## Stage 0: pre-review and freeze

1. Record the direction and single-actor rulings in the task sheet.
2. Re-derive the `H1` arithmetic from the inherited verified mixture code
   `time_uniform_mixture_certificate.py` and store the derivation as a route
   note. This is a prerequisite, not an experiment.
3. Run the pre-execution feasibility probe (occupancy and compute) and use its
   measurements to correct the protocol to v1.1. Record the probe scripts and
   outputs as route evidence.
4. Freeze the task revision and open the execution branch
   `claude/FP-SCALE-001`.

Gate A (passed 2026-09-11): `H1` arithmetic re-derived; two blocking v1.0
protocol defects found and repaired in v1.1 by pre-execution measurement. See
`docs/research_branches/FP-SCALE-001/claude/pre_review.md`.

## Stage 1: tests first

1. Write `verify_fixed_policy_expected_sarsa_scaled.py` before the
   implementation, reusing the `FP-ESARSA-001` verifier's section structure
   and adding the reduced-dimension fixtures:
   - `4x3` self-loop fixture where one Q entry serves two algebraic branches
     without duplication in either softmax candidate set;
   - repeated-visit exact residual mean and unchanged unvisited query;
   - finite-route head/read/writeback equality at `1e-12` with no mask or gate;
   - certificate formula fixture on a hand-computable record;
   - `H1` radius arithmetic fixture pinned to the inherited inversion.
2. Record the expected initial `ModuleNotFoundError` as red-first evidence.

## Stage 2: implementation

1. Add task-scoped code: `fixed_policy_expected_sarsa_scaled.py`,
   `evaluate_scaled_certificate_scale.py`, `analyze_scaled_certificate_scale.py`,
   plus reduced-dimension additions in `model.py`.
2. Reuse the inherited MDP/policy generator and trajectory sampler by
   parameterising dimensions and `pi_min`; do not fork their scientific logic.
   Any change to shared inherited files must be additive and must leave the
   inherited `6x4` defaults byte-equivalent.
3. Confirm the inherited verifiers still pass unchanged.

## Stage 3: smoke

1. Run a labelled smoke matrix: `1` task, both mixing settings, the fixed gap
   bonus, and a shortened certification batch, so smoke exercises the full code
   path without paying the full sampling cost.
2. Measure and record: rollout wall time per certification step, iteration
   wall time, memory, per-pair certification counts, the `H1` check, and the
   emission count.
3. Verify the projected formal wall time against the task budget. If it
   exceeds the budget, stop and return to `ACTIVE` with the measurement; do not
   shrink the matrix by choice.
4. Seal implementation and smoke evidence (`[claude]` commit) before the
   formal run.

## Stage 4: formal run

1. Execute the frozen formal matrix exactly once into
   `results/FP-SCALE-001/claude/`.
2. Run all post-run checks: strict-JSON reload, duplicate-key detection,
   nonfinite and shape checks, generator-identity regression against the
   inherited baseline, oracle separation, and task-scoped Ruff.
3. Seal the formal result (`[claude]` commit) with raw hashes, commands,
   environment, and the full failure history.

## Stage 5: derived verification (same actor, by user ruling)

1. Write an independent reconstruction path that reads only the sealed raw
   records and the frozen task, and recomputes:
   - every `r_x` from the inherited mixture inversion;
   - every `E_Q` from the frozen formula;
   - every emitted policy and every lower bound;
   - every reported metric, interval, and emission count;
   - the generator identity from the seed schedule.
2. Replay the sealed programs and confirm exit codes.
3. Compare the reconstruction against the sealed summary to `1e-12` where
   exact and report any deviation.
4. Write `verification_same_actor.md` ending `PASS`, `FAIL`, or `OBJECTION`,
   stating explicitly that no second actor reconstructed this route and that
   this is derived verification, not independent verification.

## Stage 6: closure

1. Write the route report, the main-route theory and report documents, and the
   failure history, including preserved failures.
2. Update `ACTIVE_WORKSPACE.md` with the result, the emission count, the `H1`
   check outcome, and the verification-strength limitation.
3. Request user approval before any merge to `main`.

## Budget and gates

- Gate A: `H1` arithmetic re-derived and recorded before any implementation.
- Gate B: smoke measured wall time within budget before the formal run.
- Gate C: exactly one formal execution; any rerun requires user authorization.
- Gate D: derived verification recorded before closure.
- Gate E: user approval before merge.
