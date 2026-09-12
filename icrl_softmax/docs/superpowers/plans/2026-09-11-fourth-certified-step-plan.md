# Plan: FP-ITER4-001 fourth certified step

Date: 2026-09-11.
Task: `docs/research_tasks/FP-ITER4-001.md`.
Design: `docs/superpowers/specs/2026-09-11-fourth-certified-step-design.md`.

## Stage 0: pre-review and freeze

1. Collect the sealed step 1--3 pattern from `FP-ITER3-001` and state, for each
   pre-registered prediction, whether it is an extrapolation or a floor.
2. Write the same-actor pre-review, including the mandatory `H1` inertness
   requirement and the evaluator-versus-science hash distinction.
3. Freeze the task and record it as `ACTIVE`.

## Stage 1: horizon extension

1. Add `4` to the frozen `ALLOWED_MAX_STEPS` tuple in the shared evaluator,
   leaving the default at `2` and keeping the knob an explicit frozen horizon
   rather than a duplicated code path.
2. Record in the comment that both `FP-ITER3-001` and `FP-ITER4-001` require the
   horizon change to be proven inert.
3. Lint.

## Stage 2: inertness check before the full run

1. Run a small subset at `--max-steps 4` and compare steps 1--3 against the
   sealed `FP-ITER3-001` bundle.
2. Only proceed if there are zero mismatches.

## Stage 3: full run

1. Execute the frozen matrix once at `--max-steps 4`:
   `12` tasks x `2` mixing settings.
2. Report per-level emissions, gains, and the four-step route count.

## Stage 4: analysis

1. `analyze_fp_iter4_001.py` recomputes from the sealed bundle with strict
   duplicate-key detection and evaluates `H1`--`H9`.
2. Report each pre-registered prediction as PASS or FALSIFIED, with the numbers.
3. Report the emission and mean-gain ratio sequences, since the shape of the
   decay is the scientific point of the extension.

## Stage 5: same-actor derived verification

1. Recompute the per-level pattern independently of the evaluator and analyzer.
2. Reconfirm the `90`-entry inertness comparison.
3. Confirm every tracked **scientific** file matches its recorded hash, and
   report any changed **evaluator** explicitly rather than silently.
4. Replay the sealed programs.
5. Write `verification_same_actor.md` ending `PASS`, `FAIL`, or `OBJECTION`,
   stating the limitation.

## Stage 6: closure and repair of the earlier record

1. Write the route journal with the decay shape and hypothesis verdicts.
2. Update the task sheet's definition of done and outcome section.
3. Update `ACTIVE_WORKSPACE.md`.
4. **Repair `FP-ATTN-ITER-001`'s verification**: the horizon extension changes
   the shared evaluator's hash, which that task's sealed-file check correctly
   flags. Update its verifier to keep the scientific-corpus checks strict and to
   pair the evaluator change with the inertness proof, then re-run and re-seal
   the regenerated report. Do not leave a committed report reading FAIL.
5. Request user approval before any merge beyond this task's own commits.

## Gates

- Gate A: pre-review recorded with no unresolved objection.
- Gate B: subset inertness check passes with zero mismatches.
- Gate C: the full matrix runs once.
- Gate D: same-actor derived verification recorded, and the `FP-ATTN-ITER-001`
  verification report re-sealed as PASS with the evaluator change documented.
