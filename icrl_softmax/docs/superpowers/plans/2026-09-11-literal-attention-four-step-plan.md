# Plan: FP-ATTN-ITER4-001 literal-attention four-step

Date: 2026-09-11.
Task: `docs/research_tasks/FP-ATTN-ITER4-001.md`.
Design: `docs/superpowers/specs/2026-09-11-literal-attention-four-step-design.md`.

## Stage 0: pre-review and freeze

1. Confirm from the sealed bundles that the network horizon is `3` while the
   numpy horizon is `4`, so the gap is real.
2. Audit which sealed `environment.json` files recorded
   `evaluate_fp_attn_iter_001.py`, so the impact of changing it is known before
   the change.
3. Write the same-actor pre-review and freeze the task as `ACTIVE`.

## Stage 1: horizon extension

1. Add `4` to the frozen `ALLOWED_MAX_STEPS` tuple in the literal evaluator,
   defaulting to `3` so the existing behaviour is unchanged.
2. Add a `--reference` option selecting the numpy baseline bundle (`iter3` =
   `FP-ITER3-001`, `iter4` = `FP-ITER4-001`), so step 4 is compared against the
   matching numpy horizon.
3. Parameterise the horizon and lint.

## Stage 2: inertness check before the full run

1. Re-run the network at `--max-steps 3 --reference iter3` over the full matrix.
2. Compare against the sealed `FP-ATTN-ITER-001` bundle on decision, `eta`,
   `E_Q` and the recorded `Qhat` gap, for all `90` step entries.
3. Only proceed on zero mismatches.

## Stage 3: full run

1. Execute the frozen matrix once at `--max-steps 4 --reference iter4`.
2. Report per-level emissions, gains and `Qhat` gaps, and the four-step route
   count.

## Stage 4: analysis

1. `analyze_fp_attn_iter4_001.py` recomputes from the sealed bundle with strict
   duplicate-key detection and evaluates `H1`--`H8`.
2. Report the per-step `Qhat` gap profile, since accumulation at four
   compositions is the substantive question.
3. Compare per-level emissions against the `FP-ITER4-001` numpy baseline.
4. Enumerate `flip_details` unconditionally.

## Stage 5: verification (recorded, not emphasised)

By the user's instruction of 2026-09-11 ("验证先不管"), verification is not the
focus of this round. The same-actor derived checks are still run and recorded:
recompute the comparison, reconfirm the `90`-entry inertness, check the producer
set, confirm the scientific corpus is byte-identical, replay the sealed
programs, and state the limitation.

## Stage 6: closure

1. Write the route journal with the per-step gaps and hypothesis verdicts.
2. Update the task sheet's definition of done and outcome section.
3. Update `ACTIVE_WORKSPACE.md`.
4. Confirm no earlier report was left failing by the shared-evaluator change; if
   one was, repair and re-seal it.

## Gates

- Gate A: pre-review recorded with no unresolved objection.
- Gate B: network inertness check passes with zero mismatches.
- Gate C: the full matrix runs once.
- Gate D: derived verification recorded; the scientific corpus confirmed
  byte-identical.
