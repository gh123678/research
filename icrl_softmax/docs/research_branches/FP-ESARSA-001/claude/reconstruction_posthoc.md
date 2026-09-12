# FP-ESARSA-001 post-hoc partial reconstruction

Date: 2026-09-11.
Script: `verify_fp_esarsa_001_reconstruction.py`
Sealed corpus: `results/FP-ESARSA-001/claude/task_results.json` (480 records)

## Why this was run

`FP-ESARSA-001` closed `VERIFIED` by explicit user exemption of the independent
GPT acceptance. Its task sheet states plainly that its evidence is
single-route and that no independent executable reconstruction was ever
performed. `FP-SCALE-001` and `FP-SCALE-002` then inherited its certificate,
routes, and decision rule, so the project's foundation rested on a result that
had only one route.

This reconstruction closes as much of that gap as the sealed evidence permits.
It was run after the FP-SCALE work, when the sealed corpus became readable from
`main`.

## Scope: what could and could not be checked

The sealed bundle serializes `q_hat`, `train_pair_counts`, the certificate
pieces (`residual_means`, `radii`, `e_q`, `epsilon_res`), the contraction block,
the improvement block, and the oracle audit. It does **not** serialize the
held-out trajectories.

| claim | checkable? | how |
|---|---|---|
| frozen seed schedule and generator identity | yes | reproduced from `SeedSequence(20260829)` and the frozen loop order |
| truth-based identity statistics | yes | recomputed `mu_pair` and action gaps from the regenerated MDP and policy |
| certificate formula `E_Q = max_x(|Ybar_x| + r_x)/(1-gamma)` | yes | recomputed from the serialized means and radii |
| mixture radii follow the frozen formula | yes | inverted the formula over counts `1..65536` and required a match |
| contraction diagonal and premise flag | yes | recomputed with the sealed `empirical_route_kernel` and `contraction_premise_satisfied` |
| emitted-policy validity | yes | positivity and row normalisation |
| claimed totals and oracle separation | yes | recomputed at record level |
| **residual means and therefore `E_Q` from first principles** | **no** | held-out trajectories were not serialized |

The last row is a real limitation. `E_Q` is verified only for internal
consistency with the radii and means it reports, not recomputed from data.
Claiming otherwise would be worse than reporting the gap.

## Result

**21 checks passed, 0 failed.**

What was positively reconstructed:

- all `480` records, with distinct `(mixing, gap, task, length)` keys;
- **every** generator identity reproduced exactly, including `seed_entropy`,
  `spawn_key`, `true_pair_occupancy_min`, `true_action_gap_min`, and
  `true_action_gap_mean` to `1e-12` — so the frozen schedule and the truth
  computation are confirmed against source, not taken on trust;
- `epsilon_res = max_x(|Ybar_x| + r_x)` and `E_Q = epsilon_res/(1-gamma)` in
  every emitted certificate, with `n_groups = 24`;
- **`31200` non-zero radii all equal the frozen formula at an integer held-out
  count** (counts `1..65536`, none unbracketed); implied held-out counts range
  `1..2843`;
- the contraction `diagonal_min` and the `premise_satisfied` flag reproduce
  from the sealed kernel and predicate;
- zero safe updates across the primary routes, zero certificate violations,
  zero oracle value decreases, zero residual-event violations, with per-route
  certificate counts agreeing between the record level and `summary.json`.

## Two failures that were mine, not the sealed result

The first run reported two failures which both turned out to be defects in this
reconstruction script, and both are recorded because they show the checks were
doing real work:

1. **Radii.** I compared each sealed radius against the inversion at the
   *training* pair count. The radii are held-out quantities, so the counts are
   different. Fixed by inverting the formula to recover the implied held-out
   count instead of assuming one.
2. **Contraction kernel.** I re-derived the finite-route kernel as an outer
   product; the sealed `one_hot_population_kernel` is a broadcast of the
   occupancy with the diagonal scaled by `exp(sharpness)`. Fixed by calling the
   sealed definition rather than a re-implementation.

## Status and limitation

This is a **same-actor, post-hoc, partial reconstruction**. It materially
strengthens confidence in `FP-ESARSA-001`'s construction, schedule, and
arithmetic, and it found no discrepancy of any kind.

It is **not** the independent second-route acceptance that task never received,
and it does **not** convert `FP-ESARSA-001` into a reciprocally verified task.
Its `VERIFIED` status still rests on the user's exemption ruling, and the task
sheet's disclosure of single-route evidence remains accurate. No status was
changed by this work.
