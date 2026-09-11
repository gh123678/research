# Data-derived cross-state kernel feasibility study

Date: 2026-09-09

Proposed task: `FP-KERN-001`

## 1. Context and decision

`FP-ADV-001` obtained a valid zero-update result. Its immediate empirical
bottlenecks were unvisited selected state-action pairs and positive but too
small visit counts, which left action-gap lower confidence bounds nonpositive.
The proposed response is to test whether observations of the same action in
other states can improve a target state's fixed-policy Q estimate.

The current MDP family does not provide semantic state features. Moreover,
`sample_mdp` independently draws each state-action transition row from a
Dirichlet distribution and each edge reward from a uniform distribution before
the existing sticky-transition and action-gap transformations. Therefore state
labels have no metric meaning, and the current generator does not guarantee
the cross-state structure required by kernel smoothing.

The study will not impose a kernel on state indices. It will construct a
data-derived, action-conditional kernel from the observed behavior of the
*other* actions in each state. It will test this method on both:

1. the unchanged current random MDP family, which answers whether the method is
   useful for the project's existing experimental regime; and
2. a hidden-cluster MDP positive control, which contains cross-state structure
   but exposes neither cluster labels nor state features to the estimator.

This separation makes a negative result interpretable. Success only on the
hidden-cluster family means kernel generalization is conditionally feasible but
the current MDP family lacks usable structure. Failure even on the positive
control rejects the proposed construction rather than the general idea of
structured generalization.

## 2. Approaches considered

Three approaches were considered before selecting the study routes.

1. **Same-action behavioral kernel.** Compare two states using empirical
   reward/transition or Q information for the target action itself. This has
   the weakest cross-action assumption, but it cannot estimate similarity when
   the target state-action pair is unvisited. It is retained as a low-count
   control.
2. **Leave-one-action-out state-signature kernel.** For target action `a`,
   compare states using their empirical Q signatures for actions `b != a`, then
   borrow target-action observations from similar states. This can estimate an
   unvisited `(s,a)` when the other actions provide enough signature support.
   It is the primary route because it directly tests the user's proposed
   cross-state, same-action generalization mechanism.
3. **Learned low-rank or latent-state model.** Factorize the complete Q table or
   transition tensor and impute missing entries. This could also cover zero
   visits, but it adds a substantially stronger structural model and is prone
   to overfitting with only 6 states and 4 actions. It is deferred unless the
   simpler kernel route shows a real structural signal.

## 3. Research question and falsifiable hypotheses

The primary question is:

> Without external state features, does a leave-one-action-out empirical state
> signature contain enough information to improve fixed-policy Q estimation
> and action ordering for zero-count and low-count state-action pairs?

The frozen hypotheses for the future formal task are:

1. In the hidden-cluster positive control, smaller leave-one-action-out
   signature distance predicts smaller true target-action Q difference.
2. In the hidden-cluster family, the primary kernel route improves zero-count
   Q error over unconditional same-action pooling and improves sparse-positive
   Q error over the unpooled estimator.
3. In the hidden-cluster family, the primary route improves action ordering in
   states containing a zero-count or sparse-positive action without increasing
   the false-improvement rate beyond the frozen tolerance.
4. The same three diagnostics on the unchanged current random MDP family
   determine applicability to the existing project; they are not assumed to
   pass.
5. The same-action behavioral kernel can improve some sparse-positive pairs
   but cannot claim zero-count coverage.

The task is empirical feasibility work. It makes no concentration, safety, or
policy-nondegradation claim. Oracle quantities are audit outputs only.

## 4. Data separation

Each sampled MDP and fixed policy will generate three independent
stationary-start trajectories of the same declared length:

- `value_trajectory`: constructs the unchanged exact V-first state-value
  estimate used in all routes;
- `signature_trajectory`: constructs the state signatures and kernel weights;
- `target_trajectory`: supplies the target-action recovery observations that
  are locally estimated or pooled.

All routes receive the same three trajectories. The unpooled and unconditional
pooling baselines therefore receive the same target observations and value
estimate as the kernel routes. Separate random-number streams are derived
deterministically from the task seed and recorded per trajectory.

The true transition kernel, reward tensor, stationary occupancy, true V, true
Q, true action gap, and exact policy return are unavailable to every estimator
and kernel builder. They are attached only after all route outputs are frozen
inside a structurally separate `oracle_audit` namespace.

## 5. Leave-one-action-out kernel

Let `V_hat` be the common value estimate from the value trajectory. On the
signature trajectory, define the observed recovery estimate

```text
q_sig(s,b) = mean [R_{t+1} + gamma V_hat(S_{t+1}) | S_t=s, A_t=b]
```

whenever the pair count is positive. For a target action `a`, define the common
signature support between states `s` and `s_prime` as

```text
C_a(s,s_prime)
  = {b != a : N_sig(s,b) > 0 and N_sig(s_prime,b) > 0}.
```

At least two common non-target actions are required. With the public value
scale `B = R_star / (1-gamma)`, define

```text
d_a(s,s_prime)^2
  = mean over b in C_a(s,s_prime)
      [(q_sig(s,b) - q_sig(s_prime,b)) / (2B)]^2.
```

The primary Gaussian bandwidth is the median of the finite positive eligible
distances for the record and target action. If no positive median exists, the
route abstains rather than substituting an oracle or tuned bandwidth. The
kernel is

```text
K_a(s,s_prime) = exp(-d_a(s,s_prime)^2 / (2 h_a^2)).
```

On the target trajectory, let `Y_t = R_{t+1} + gamma V_hat(S_{t+1})`. The
primary estimate is

```text
Q_hat_kernel(s,a)
  = sum_s_prime K_a(s,s_prime)
        sum_{t:(S_t,A_t)=(s_prime,a)} Y_t
    / sum_s_prime K_a(s,s_prime) N_target(s_prime,a).
```

The target state itself participates with weight one when it has observations.
When `(s,a)` has zero target count, only other states contribute. A zero or
nonfinite denominator produces an explicit abstention.

The same-action control uses an analogous signature derived from action `a`
on the signature trajectory. It must abstain when the target pair has no
signature observation and therefore cannot count as zero-visit recovery.

## 6. Baselines and routes

Every record contains four pre-registered routes:

1. `local_unpooled`: the ordinary target-trajectory pair mean; unavailable at
   zero count;
2. `action_only_pool`: count-weighted pooling of action `a` over all states,
   without a state-similarity kernel;
3. `same_action_anchor_kernel`: the low-count control described above;
4. `leave_one_action_out_kernel`: the primary route.

No bandwidth grid, post-hoc nearest-neighbor count, low-rank route, learned
embedding, or oracle-selected route is permitted in the first task.

## 7. Environment families

### Current unstructured family

The first family exactly reuses the existing `make_mdp` and `make_policy`
construction. It preserves the independently sampled transition/reward rows,
sticky-transition mixing settings, reward-gap bonuses, and policy floor. This
is the primary applicability test for the current research setting.

### Hidden-cluster positive control

The second family assigns the six labelled states to two balanced latent
clusters. For each cluster and action it samples a transition/reward prototype;
each member state is a frozen convex perturbation of that prototype. The
prototype weight is `0.90` and the independently sampled state-specific weight
is `0.10`, for both transition and edge-reward components. The transition row
then uses the existing sticky-mixture convention:

```text
P_struct(s,a) = 0.90 P_proto(cluster(s),a) + 0.10 P_independent(s,a)
P(s,a) = (1-mixing) point_mass(s) + mixing P_struct(s,a).
```

The bounded edge reward is

```text
R(s,a,s_next)
  = 0.90 R_proto(cluster(s),a,s_next)
    + 0.10 R_independent(s,a,s_next),
```

followed by the unchanged nonnegative reward-gap bonus on action zero. The two
clusters contain three states each and their state labels are randomly
permuted per record. Initial-state probabilities retain the existing
Dirichlet construction. Cluster assignments, prototypes, and perturbations
are hidden from all estimators and kernels and retained only in the oracle
audit.

The `0.90/0.10` structure strength is frozen and may not be retuned after smoke
or formal output. The positive control is not evidence that the current
unstructured family is smooth; it only verifies that the data-derived kernel
can detect and exploit a known but hidden shared structure.

## 8. Frozen feasibility matrix

The proposed formal matrix contains exactly 480 records:

- environment families: current unstructured and hidden-cluster;
- 15 tasks per cell;
- 6 states and 4 actions;
- trajectory lengths: 256, 1024, 4096, and 16384 for each of the three streams;
- mixing settings: 0.08 and 0.50;
- reward-gap bonuses: 0 and 0.50;
- `pi_min = 0.05`, `gamma = 0.70`, and 160 value iterations;
- seed: 20260909.

This is a new matrix, not a rerun or modification of `FP-ADV-001`. A labelled
smoke matrix may verify implementation and schema only. Kernel formulas,
environment parameters, metrics, thresholds, seed, and formal matrix are
frozen before any formal output is inspected.

## 9. Metrics and feasibility decision

Pair metrics are reported separately for target counts `0`, `1-4`, `5-16`,
and `17+`. Primary metrics are:

- estimator coverage among zero-count pairs with eligible signatures;
- Q RMSE and MAE against true fixed-policy Q;
- pairwise action-order accuracy;
- top-action accuracy in states containing a zero-count or `1-4` count pair;
- false-improvement rate, where an estimated positive action difference has a
  nonpositive true difference;
- oracle return and componentwise value change of a frozen greedy-with-floor
  diagnostic policy constructed from each complete route estimate.

The greedy-with-floor policy is descriptive only and carries no safety claim.
Routes with incomplete action rows abstain from that state and retain the old
policy row.

RMSE/MAE route comparisons use only target pairs for which both compared
routes emit finite estimates. Coverage is reported over all target pairs and
therefore cannot be hidden by this comparable-pair restriction. Action-order
and false-improvement comparisons likewise use the identical set of
route-common finite comparisons.

All uncertainty summaries use per-record paired differences and a two-sided
95% Student-t interval, matching the existing project convention. The primary
route passes the hidden-cluster mechanism screen only if:

1. it estimates at least 50% of signature-eligible zero-count pairs;
2. its mean zero-count RMSE is at least 10% lower than `action_only_pool`, with
   the paired interval for improvement excluding zero;
3. its mean `1-4` count RMSE is at least 10% lower than `local_unpooled`, with
   the paired interval for improvement excluding zero;
4. its sparse-state top-action accuracy improves by at least 5 percentage
   points over `action_only_pool`, with the paired interval for improvement
   excluding zero; and
5. its false-improvement rate is no more than one percentage point above the
   `action_only_pool` rate on their common finite comparisons.

The result classification is frozen as follows:

- `GENERAL_FEASIBLE`: the mechanism screen passes in both families;
- `STRUCTURE_CONDITIONAL`: it passes in the hidden-cluster family but not the
  current unstructured family;
- `NOT_SUPPORTED`: it fails the hidden-cluster mechanism screen;
- `INVALID`: implementation, provenance, route separation, or oracle-isolation
  checks fail.

No individual cell, bandwidth, or secondary metric can replace this decision
rule after formal output is inspected.

## 10. Implementation boundaries

The future formal task will create isolated modules for:

- hidden-cluster MDP generation;
- pure signature, distance, kernel, and pooled-estimator construction;
- deterministic fixtures and malformed-input tests;
- the paired evaluator and strict analyzer.

The implementation may reuse existing MDP, fixed-policy value-estimation,
policy-quantity, strict-JSON, and evidence helpers through narrow imports. It
must not change any existing evaluator, verified result, `FP-ADV-001` artifact,
manuscript, or user-owned untracked document.

Every kernel record must expose observed counts, common-action masks,
distances, bandwidth, normalized weights, effective sample size, estimate,
coverage/abstention reason, and no oracle fields. Oracle data must be stored
separately and checked against a deny-list at the pure estimator boundary.

## 11. Validation and failure handling

Deterministic verification must cover:

- identical signatures producing unit similarity;
- distant signatures receiving smaller weights;
- leave-one-action-out exclusion of the target action;
- zero-count target recovery from valid neighbors;
- same-action control abstention at missing target signature;
- exact reduction to the unpooled mean when only self weight is present;
- normalized finite weights and positive denominators;
- deterministic median-bandwidth behavior and all abstention paths;
- invariance to state relabelling when trajectory and MDP tensors are relabelled
  consistently;
- positive-control cluster recovery without exposing cluster labels;
- strict separation of estimator inputs and oracle audit;
- reproducible seed streams, exact record count, strict JSON, and task-scoped
  linting.

Malformed counts, shapes, policies, rewards, transition rows, missing common
actions, degenerate bandwidths, and nonfinite values must produce explicit
ordered errors or abstentions. There is no silent fallback to uniform weights,
an oracle neighbor, or a different bandwidth.

## 12. Governance and verification

`FP-KERN-001` is conclusion-critical and multi-stage, so it is a long task
under `AGENTS.md`. GPT and Claude Code must independently implement and execute
the same frozen task from one verified baseline, on `codex/FP-KERN-001` and
`claude/FP-KERN-001`, with outputs in their corresponding result directories.
Neither route may inspect the other's conclusion before both first and formal
results are sealed. Reciprocal executable verification is required before any
result is called verified.

The task must progress through `DRAFT -> REVIEW -> ACTIVE -> VERIFYING ->
VERIFIED`. Claude must perform read-only pre-review and return `APPROVED` before
implementation. A verified negative or structure-conditional result is a
successful scientific completion; it is not permission to retune the frozen
kernel or environment.
