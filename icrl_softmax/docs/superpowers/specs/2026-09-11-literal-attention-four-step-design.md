# Design: four certified steps inside the literal attention network

Date: 2026-09-11.
Task: `docs/research_tasks/FP-ATTN-ITER4-001.md`.
Status: `ACTIVE`.

## 1. The gap

Two verified results did not align in horizon:

| result | horizon | code path |
|---|---|---|
| `FP-ITER4-001` | four certified steps, `12/48` | numpy |
| `FP-ATTN-ITER-001` | three certified steps, `15/48` | literal network |

The deepest certified result therefore rested on numpy, and the network claim
stopped one step short of it. The unexecuted inference was:

```text
network -> three certified steps    (established)
network -> four certified steps     (never executed)
```

## 2. Design decision

Run the network at `MAX_STEPS = 4` with the comparison baseline switched from the
sealed `FP-ATTN-ITER-001` three-step bundle to the sealed `FP-ITER4-001`
four-step numpy bundle, so step 4 is matched against the matching numpy horizon.

Nothing else moves: the network produces `Qhat` at every step, the identical
training and certification batches are reused, the same frozen certificate and
decision rule apply, `pi_{k-1}` is the target at step `k`, and the route→network
mapping and tolerance are inherited.

## 3. The risk, one composition deeper

`FP-ATTN-ITER-001` measured a per-step gap that stayed flat through three
compositions (`1.076e-05`, `4.585e-06`, `7.176e-06`). The obvious question was
whether a fourth composition would finally let it grow, especially since
`FP-ITER4-001`'s fourth-step emissions include margins as small as `0.185`.

Two outcomes, both informative:

- **flat**: the `float32` difference behaves as per-step rounding noise, and the
  network and numpy horizons align at four steps;
- **growing**: accumulation is real and bounded, which would say something
  quantitative about how deep the network path can be trusted.

The design does not assume flatness. `H6` names it explicitly, and `H7` requires
every disagreement to be serialised with its `E_Q`, both smallest lower bounds
and the `Qhat` gap, so a flip surfaces with its boundary margin.

## 4. Corpus integrity

The horizon knob lives in the shared literal evaluator, so extending it changes
that file. The design separates the two classes, following `FP-ITER4-001`'s
precedent:

- the **scientific corpus** must be byte-identical;
- the **task evaluators** evolve with the horizon.

Before relying on the extension, the design requires checking whether any sealed
bundle recorded the changed file, and pairing the change with an **inertness
proof** — a re-run at the frozen horizon `3` reproducing all `90` sealed step
entries.

## 5. What success and failure mean

- Success: the network emits four certified steps, decisions agree `105/105`, and
  the step-4 gap stays within `ATOL`. The network and numpy paths are then
  aligned at the deepest certified horizon.
- `H5` or `H6` failing: `float32` network arithmetic changes certified decisions
  or drifts beyond tolerance at four steps. That would bound the network path and
  would be the task's headline result.

## 6. Risks and mitigations

- **Compounding network error.** Named in `H6`, serialised per step, and
  compared against the matching numpy horizon.
- **Horizon perturbation.** Mitigated by `H1`'s exact-identity requirement over
  `90` entries, checked on a subset before the full run.
- **Shared-evaluator record drift.** Mitigated by auditing which sealed bundles
  recorded the file, and by the inertness proof.
- **Cost.** `105` network step executions plus sampling; measured at a few
  minutes.
- **Same-actor verification.** Disclosed and, per the user's instruction, not
  the focus of this round.

## 7. Out of scope

- Extending past four steps on either path.
- Repeated control, online control, or conditional-on-emission claims.
- Any claim of independent or reciprocal verification.
