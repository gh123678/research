# Plan: FP-ATTN-ITER-001 literal-attention iteration

Date: 2026-09-11.
Task: `docs/research_tasks/FP-ATTN-ITER-001.md`.
Design: `docs/superpowers/specs/2026-09-11-literal-attention-iteration-design.md`.

## Stage 0: pre-review and freeze

1. Confirm from the sealed bundles that the gap is real: no sealed artifact
   records a network-produced `Qhat` at step 2 or step 3.
2. Write the same-actor pre-review with the two mandatory conditions —
   executable provenance, and enumerated rather than absorbed disagreements.
3. Freeze the task and record it as `ACTIVE`.

## Stage 1: evaluator

1. Rebuild the frozen per-record state exactly as `FP-ITER3-001` does:
   regenerated MDP and policy, `mu_state`, one training batch, one
   certification batch, and their digests.
2. For each route, run the literal network for `160` layers from `Q_0 = 0` at
   each of up to three steps, using the **current** policy as the target.
3. Feed the network's `Qhat` straight into the frozen certificate, then the
   frozen decision rule, with no numpy result in that path.
4. Compute the numpy route additionally at the same step and same batches, for
   comparison only.
5. Serialise per step: producer, `Qhat` gap, literal and numpy status, both
   `eta` values, both `E_Q` values, both smallest lower bounds, the ordered
   abstention reasons, and the oracle audit for that step.
6. Oracle audit only: `V^{pi_k}` from `policy_quantities`, each step's realized
   error against `Q^{pi_{k-1}}`, and componentwise non-degradation.

## Stage 2: smoke

1. Run a labelled smoke subset at both frozen mixing settings.
2. Check that the network path completes three steps, that provenance is
   recorded, and that the per-step gaps are of the expected order.
3. Seal the evaluator before the full run.

## Stage 3: full run

1. Execute the frozen matrix once: `12` tasks x `2` mixing settings.
2. Report per-level emissions, three-step route count, decision agreement,
   flips, and violations.

## Stage 4: analysis

1. `analyze_fp_attn_iter_001.py` recomputes from the sealed bundle with strict
   duplicate-key detection and evaluates `H1`--`H6`.
2. Report the per-step gap profile, since the accumulation question is the point
   of the task.
3. Compare per-level emissions against the `FP-ITER3-001` baseline.
4. Enumerate `flip_details` unconditionally, so `H5` is satisfied by an
   exercised enumeration rather than an untested path.

## Stage 5: same-actor derived verification

1. Recompute the comparison independently of the evaluator and analyzer.
2. Confirm the producer set is exactly `{literal_attention_network}`.
3. Re-check the finite route's gate-free property on this data.
4. Replay the sealed programs and verify sealed-file byte-identity.
5. Write `verification_same_actor.md` ending `PASS`, `FAIL`, or `OBJECTION`,
   stating the within-author limitation.

## Stage 6: closure

1. Write the route journal with the per-step gaps and hypothesis verdicts.
2. Update the task sheet's definition of done and outcome section.
3. Update `ACTIVE_WORKSPACE.md`.
4. Request user approval before any merge beyond this task's own commits.

## Gates

- Gate A: pre-review recorded with no unresolved objection.
- Gate B: smoke completes three network steps with provenance recorded.
- Gate C: the full matrix runs once.
- Gate D: same-actor derived verification recorded.
