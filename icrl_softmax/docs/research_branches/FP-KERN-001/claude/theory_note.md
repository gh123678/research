# FP-KERN-001 Claude route: theory and method note

Date: 2026-09-09. Route: `claude/FP-KERN-001` from common route baseline
`28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c`. Task version 1.0 (`ACTIVE`).

This note is the Claude route's method statement, written before any formal
output was inspected. It interprets only the frozen task, design, plan, and
the closed pre-review clarifications.

## Construction

- Three independent stationary-start streams per record: value, signature,
  target. Starts are drawn from the true stationary distribution with each
  stream's own RNG substream (`task_seed.spawn` per length, then
  `length_seed.spawn(3)`).
- `V_hat` is the unchanged exact V-first estimator
  (`iterative_state_evaluation`, beta=None, alpha=0.65, 160 iterations,
  clipping bound `value_limit = reward_bound/(1-gamma)`,
  `reward_bound = 1 + gap_bonus`) applied to the value stream.
- All pair statistics are `Y = R + gamma * V_hat(S')` aggregates. Signature
  and target streams contribute counts and Y-sums per state-action pair;
  the value stream contributes state counts, reward sums, and
  state-transition counts, from which `vfirst_value_from_aggregates`
  reproduces the estimator bitwise (verifier K15).
- The four frozen routes are implemented exactly in
  `kernel_state_generalization.py`: `local_unpooled`, `action_only_pool`,
  `same_action_anchor_kernel`, `leave_one_action_out_kernel`, with the frozen
  Gaussian kernel, sorted-float64 `numpy.median` bandwidth over finite
  positive eligible distances, observation-weighted ESS
  `(sum K*N)^2 / sum K^2*N`, and the frozen nine-reason ordered abstention
  contract. There is no fallback: an abstaining pair emits no estimate.

## Interpretation choices frozen before output

- The self state enters the kernel sum with weight one
  (`d_a(s,s) = 0`, `K = 1`) and therefore reduces the primary estimate to the
  unpooled pair mean when only self has target observations; the self pair is
  never part of the bandwidth median (its distance is zero, not positive).
- `insufficient_common_actions` fires for a zero-count primary pair when no
  *other* state shares two observed non-target signature actions; a
  positive-count pair still emits through the self weight once bandwidth and
  denominator checks pass.
- `target_source_unavailable` means action `a` has no target-trajectory
  observations anywhere in the record; `kernel_denominator_invalid` covers a
  nonpositive/nonfinite weighted denominator despite global availability
  (e.g., no eligible neighbor holds observations).
- The anchor control shares the ordered contract: missing target-action
  signature at the target state maps to `insufficient_common_actions` (reason
  4), zero target count to `target_source_unavailable` (reason 6); it can
  never emit at zero target count.
- Signature eligibility for the zero-count coverage denominator follows
  clarification 2: a zero-count pair is eligible when another state has two
  common observed non-target actions and the record/action bandwidth is
  finite positive; later failures (`target_source_unavailable`,
  `kernel_denominator_invalid`, `estimate_nonfinite`) stay in the
  denominator.
- The p0 initial-state vector of the current family is renormalized to the
  simplex in float64 (float32 Dirichlet output sums to 1 within ~1e-7); the
  sticky-mixed transition and gap-bonus reward construction is bit-identical
  to the unchanged `make_mdp`.

## Shared-nuisance coupling statement (required by clarification 6)

The common `V_hat` is an independently estimated shared nuisance value. It
reflects fixed-policy continuation, which includes all actions at successor
states. "Leave one action out" means the target action is absent from the
state signature coordinates; it does not make the shared value nuisance
mathematically independent of the target action. This common dependence is
not direct target-trajectory leakage.

## Oracle separation

Estimator inputs are exactly: `n_states`, `n_actions`, `gamma`,
`reward_bound`, `policy`, `value_estimate`, `signature_counts`,
`signature_sums`, `target_counts`, `target_sums`. Unknown or oracle-named
keys (true Q/V, transitions, reward tensor, occupancies, cluster labels,
prototypes, exact returns) raise `KernelInputError` at the boundary. Hidden
cluster labels and prototypes exist only in a separate `hidden` mapping that
the evaluator attaches under `oracle_audit` after route outputs freeze.

## Environment families

- `current_unstructured`: unchanged `make_mdp`/`make_policy`.
- `hidden_cluster`: balanced 3/3 clusters via per-record label permutation;
  `P_struct = 0.90 P_proto + 0.10 P_independent`,
  `P = (1-mixing) point_mass(s) + mixing P_struct`,
  `R = 0.90 R_proto + 0.10 R_independent`, unchanged action-zero gap bonus;
  p0 retains the existing Dirichlet construction.

## Metrics and decision

Count bins 0 / 1-4 / 5-16 / 17+ on target-trajectory counts; per-record
paired differences with two-sided 95% Student-t intervals; the frozen
five-item screen per family; classification `GENERAL_FEASIBLE` /
`STRUCTURE_CONDITIONAL` / `NOT_SUPPORTED` / `INVALID`. Hypothesis 1
(per-record Spearman correlation between eligible signature distances and
absolute true target-action Q differences) is a secondary diagnostic, never
a screen substitute. The greedy-with-floor policy is descriptive; incomplete
route rows retain the original policy row and carry no safety claim.
