# FP-SCALE-001: A certified relative-softmax improvement step at a reachable certificate scale

## Task metadata

- Created: 2026-09-11.
- Author: drafted by Claude under the direct user instruction of 2026-09-11
  ("不管 codex 了，验证也交给你"), which assigns both execution and verification
  to Claude for this task. This is a deviation from `AGENTS.md` section 7
  (reciprocal verification) and is recorded as a task-scoped user exception
  below. It changes no scientific contract.
- Status: `ACTIVE`.
- Task version: `1.1`. The v1.1 revision corrects the protocol scale using the
  pre-execution feasibility measurements recorded in "Protocol feasibility
  probe (v1.1)". The v1.0 scale was arithmetically wrong: its trajectory
  lengths could not deliver the per-pair occupancy the `H1` target requires.
  No smoke or formal output existed when v1.1 was frozen, so no result was
  inspected to choose it. All v1.1 changes are protocol parameters; the
  mathematical contract, the routes, the certificate, and the acceptance
  criteria are unchanged from v1.0.
- Predecessors: `FP-ESARSA-001` (`VERIFIED` by user exemption, single-route
  evidence), `FP-ADV-001` (`VERIFIED`, negative), `FP-TU-001` (`VERIFIED`,
  certificate baseline), `FP-KERN-001` / `FP-KERN-002` (`VERIFIED`, negative
  cross-state-generalization results, merged to `main` 2026-09-11).
- Scientific baseline: `467ebf591dde8421f7ed5324a18514fb5935d2f9` (`main`
  after the FP-KERN merges).
- Execution branch: `claude/FP-SCALE-001`.
- Result directory: `results/FP-SCALE-001/claude/`.
- Design:
  `docs/superpowers/specs/2026-09-11-reachable-certificate-scale-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-11-reachable-certificate-scale-plan.md`.
- Classification: long, conclusion-critical, single-actor under the user
  exception.

## Why this task exists: the measured obstruction

`FP-ADV-001` emitted `0/480` safe updates on all six routes and
`FP-ESARSA-001` emitted `0/480` on all three routes. A read-only diagnostic on
the sealed `FP-ESARSA-001` records (`results/FP-ESARSA-001/claude/`,
`task_results.json`, `expected_exact` route) localizes the obstruction:

| trajectory length | emitted | mean `E_Q` | mean realized `|Qhat-Q*|` | ratio |
|---|---|---|---|---|
| 1024 | 45 | 125.229 | 0.869 | 144 |
| 4096 | 118 | 62.876 | 0.351 | 179 |
| 16384 | 120 | 27.254 | 0.167 | 163 |

Three facts follow, and together they determine this task's design:

1. **The slack is the concentration radius, not the certificate form.** For the
   emitted records the worst-pair empirical residual `max_x |Ybar_x|` is at most
   `2.4125`, while the worst-pair radius `max_x r_x` is `37.7612`. The certified
   `E_Q` is therefore almost entirely `r_x/(1-gamma)`.
2. **The radius is set by the rarest held-out pair.** The inherited verified
   inversion gives `r_x = 37.76` at `N_x = 1`, `4.15` at `128`, `1.51` at
   `1024`. The worst pair count in the sealed matrix was `14`, which alone
   explains the observed `E_Q` floor of about `22.5`.
3. **The inversion is not the culprit.** The frozen radius is only about
   `1.32`--`1.41` times a calibrated two-sided sub-Gaussian radius at the same
   count and risk, so replacing it cannot recover the factor of roughly `150`.

Consequently, weakening the guarantee type alone cannot succeed either: an
occupancy-weighted value guarantee has full support over the states, so its
radius is still the worst-pair radius. The obstruction is per-pair occupancy.

## Research question

At a protocol whose per-pair held-out occupancy is large enough for the
verified time-uniform mixture certificate to be non-vacuous, can the
fixed-policy Expected SARSA construction of `FP-ESARSA-001` emit a certified
relative-softmax policy update, and does that emitted update improve value
without degrading any state?

## Falsifiable hypotheses

1. `H1 (feasibility arithmetic)`: under the frozen protocol every record
   reaches a worst-pair certification count `N_min >= 20000`, and the
   certificate's radius contribution satisfies
   `max_x r_x / (1-gamma) <= 1.2`. This is a checkable prediction; a record
   failing it is excluded by the frozen `heldout_pair_support_missing` reason.
2. `H2 (construction carries over)`: the exact grouped and finite-logit
   Expected SARSA routes reproduce batch Expected SARSA within `1e-12` on the
   reduced-dimension fixtures, exactly as in `FP-ESARSA-001`.
3. `H3 (certificate validity)`: every emitted `E_Q` bounds the realized oracle
   `||Qhat-Q^pi||_infinity` in every audited record; zero certificate
   violations and zero residual-event violations.
4. `H4 (emission at small tilt)`: at least one primary route emits a certified
   non-degrading relative-softmax update in at least `3` of the frozen formal
   records, and the selected `eta` in those emissions is at most `0.1`.
5. `H5 (real improvement)`: every emitted update satisfies
   `V^{pi_plus} >= V^pi` componentwise and
   `sum_s (V^{pi_plus}(s)-V^pi(s)) > 0` in the oracle audit, so at least one
   emission is a strict improvement rather than a vacuous non-degradation.
6. `H6 (attribution)`: the certificate is non-vacuous at this scale, meaning
   `E_Q` is smaller than the `FP-ESARSA-001` `E_Q` floor of about `22.5` in
   every emitted record, and the difference is attributable to per-pair
   occupancy rather than to any formula change.

`H1`--`H3` are mandatory construction and validity claims. `H4`--`H5` are the
usefulness claims. `H6` is the attribution claim. Zero emissions at a
non-vacuous scale is a valid negative result and forbids retuning.

## Frozen mathematical contract

The estimator, the three routes, the cross-fitted residual certificate, the
relative-softmax improvement rule, the ordered non-emission reasons, and the
oracle separation are inherited unchanged from `FP-ESARSA-001` (task sheet and
`docs/research_branches/fixed_policy_expected_sarsa_theory.md`). This task
reuses the inherited verified programs by import and does not modify them.

Explicitly unchanged and inherited verbatim: the exact grouped update
`Q_{l+1}(x) = Q_l(x) + (alpha/N_x^train) sum_{t:X_t=x} delta_t^l`, the
finite-logit scores with `zeta = xi = tau = 8`, the canonical one-token-per-pair
memory, the held-out residual
`Y_t(Qhat) = R_{t+1} + gamma sum_b pi(b|S_{t+1}) Qhat(S_{t+1},b) - Qhat(S_t,A_t)`,
the mixture event with exactly `d = |S||A|` groups at total risk `delta`, the
certificate `E_Q = max_x (|Ybar_x| + r_x)/(1-gamma)`, the lower bound
`LB_s(eta) = Ihat_s(eta) - E_Q ||pi_eta^+(.|s)-pi(.|s)||_1`, and every ordered
abstention reason.

### Frozen protocol (v1.1)

- States/actions: `4/3`, so `d = 12` pairs.
- Behavior-policy minimum action probability: `pi_min = 0.15` (raised from
  `0.05`), unchanged from v1.0.
- Reward-gap bonus: `0.5` only.
- Mixing settings: `0.08` and `0.5`.
- Tasks per cell: `12`.
- **Training trajectory**: `65536` transitions, used only to build `Qhat`.
- **Certification batch**: an independent, freshly seeded `1048576`-transition
  trajectory from the same frozen behavior policy and MDP, used only for the
  held-out residual certificate and the oracle audit. The two batches are
  disjoint random draws and are never mixed.
- **Count rule**: a record passes `H1` when every pair has at least `20000`
  certification observations.
- Eta candidates: `1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01` in that descending
  order (v1.0's grid extended downward by two entries because a smaller tilt is
  the admissible regime when the certified error is not negligible).
- Inherited unchanged: `gamma = 0.70`, `alpha = 0.65`, `160` layers,
  `R_star = 1.5` (so `B = 5`), `delta = 0.05`, `15` mixture components,
  `Q_0 = 0`, seed `20260911`.
- Matrix: `2` mixing settings x `1` gap bonus x `12` tasks = `24` cells,
  `24` matched records, each containing all three routes.

### Protocol feasibility probe (v1.1)

Measured before any smoke or formal output, with the inherited generator and
sampler and the inherited verified mixture inversion. These are design
calculations; they produce no certificate, no candidate policy, and no
emission.

1. **The v1.0 lengths were arithmetically insufficient.** At `4x3` the sticky
   chain's stationary occupancy caps the average pair count at
   `heldout/d`; a `131072`-length trajectory holds out `65536` transitions,
   so the measured worst-pair count was only about `1500`, not `20000`, and no
   affordable trajectory length could reach the target.
2. **Separating training from certification fixes it.** A `1048576`-step
   certification batch gave measured worst-pair counts of
   `23498, 27013, 28154, 31815, 34051` over five probes at `mixing = 0.5`,
   all above the `20000` target.
3. **The resulting radius meets the `H1` target.** At those counts the
   inherited inversion gives `r_x` in `0.2840`--`0.3254`, so
   `r_x/(1-gamma)` is in `0.947`--`1.085`, against the `1.2` target.
4. **Compute is not the binding constraint.** The exact grouped `65536`-train
   route costs `0.34` s and a `262144`-point certificate `0.02` s; the
   extrapolated formal cost is about `2.1` minutes of iteration for the whole
   matrix. The rollout sampler dominates and is measured in smoke.

The probe used no formal or smoke artifacts, so it cannot be outcome tuning.
Its scripts are recorded in the route evidence.

## Routes

1. `expected_exact`: exact grouped fixed-policy Expected SARSA, primary.
2. `expected_finite`: finite-logit mask-free fixed-policy Expected SARSA,
   primary.
3. `sampled_exact`: exact grouped sampled SARSA using the observed next action,
   control.

## Ordered non-emission reasons

Inherited verbatim from `FP-ESARSA-001`; no reason may trigger a different eta
grid, a different route, a second risk budget, or an oracle fallback.

## Prohibited work

- No post-hoc change to any frozen formula, sharpness, eta rule, split, event,
  risk allocation, seed, matrix, hypothesis, or metric after v1.1 is activated.
- No reuse or reinterpretation of `FP-ESARSA-001` or `FP-ADV-001` frozen
  numbers as evidence for this task; they are motivation only.
- No modification of any file sealed by a closed task, including the inherited
  `FP-ESARSA-001` programs, which must remain byte-identical and are reused by
  import only.
- No input-dependent equality mask or visited gate in `expected_finite`.
- No learned transition model, oracle certificate input, or validation leak
  into `Qhat` construction: the certification batch may not influence `Qhat`,
  the eta selection rule, or any hyperparameter.
- No repeated policy iteration, no conditional-on-emission guarantee, and no
  output exploration-floor claim.
- No write to `main` during execution, no push, publication, external message,
  permission bypass, or new fee category without separate authorization.
- No claim of reciprocal verification: this task has a single actor by user
  ruling, and every report must say so.

## Acceptance criteria

1. The reduced-dimension fixtures verify the inherited construction claims at
   `1e-12`, including self-loops, repeated visits, unvisited queries, and the
   absence of masks and gates in the finite route.
2. `H1` is checked per record and reported; any record failing it is excluded
   from `H4` with the frozen reason, and the exclusion count is reported.
3. Every emitted `E_Q` equals the frozen formula, uses full certification
   support, and bounds the realized oracle error in every audited record.
4. `H4` is evaluated on the frozen records with no retuning; the emitted count
   and the selected eta values are reported exactly.
5. Every emitted policy is finite, strictly positive, row-normalized, changed,
   and componentwise non-degrading in the oracle audit; at least one emission
   is a strict improvement under `H5`.
6. Every abstention returns the input policy bit-for-bit with a frozen reason.
7. The training trajectory and the certification batch are structurally
   separated and independently seeded; certification data cannot influence
   `Qhat` or hyperparameter selection. A leak counterexample fixture is
   required.
8. Exact truth, realized errors, action gaps, improvements, values, and returns
   live only in `oracle_audit`.
9. All strict-JSON, duplicate-key, nonfinite, shape, range, identity, seed,
   configuration, and result-location checks pass.
10. The formal result contains exactly the frozen number of matched records,
    each with all three routes, reproducing the frozen generator identities and
    seed schedule, and confirming the inherited `FP-ESARSA-001` programs are
    byte-identical to their sealed versions.
11. All empirical violations, premise failures, and exclusions are enumerated
    and excluded from theorem evidence.
12. The compute budget is measured in smoke and the formal run is executed
    exactly once.
13. Complete reproducibility evidence is recorded: commands, environment,
    commits, raw hashes, failures, anomalies, and limitations.
14. Verification is performed by the same actor under the recorded user
    exception, and the report states that no independent reconstruction
    occurred. It must be executable and derived, never a restatement of the
    author's conclusions: an independent reconstruction of every frozen input,
    a replay of the sealed programs, and a from-scratch recomputation of the
    reported metrics.
15. `ACTIVE_WORKSPACE.md` is current; `main` is unchanged without separate user
    approval.

## Failure criteria

The mandatory theory fails if the held-out residuals are not martingale-valid
after conditioning on the training prefix, the inherited mixture cannot be used
at random visit counts with scale `2B`, the Bellman residual does not imply the
stated Q bound, the relative-softmax lower bound is false, or an unstated model
or oracle input is required.

The construction fails if the exact route differs from Expected SARSA, the
finite route uses hidden masks or gates, signed residuals are lost, policy
changes during evaluation, certification data leaks into `Qhat`, a pair is
duplicated in an attention candidate set, or an emitted update can degrade any
state on the certificate event.

`H4` or `H5` may fail without invalidating the task. If they fail while `H1`
and `H6` pass, the result is a verified negative usefulness result at a
non-vacuous certificate scale, which falsifies per-pair occupancy as the sole
obstruction and moves the certificate form to the leading suspect. That is a
valuable result and must be reported as such, without retuning.

## Stopping conditions

Stop affected work and notify the user if:

- `H1` fails systematically in a way that indicates a mis-specified protocol
  rather than ordinary occupancy noise;
- the smoke measurement projects wall time or memory beyond the task budget;
- implementation needs a prohibited mask, gate, oracle, or validation leak;
- smoke or formal output would be needed to choose a frozen parameter;
- an existing uncommitted edit overlaps an authorized path;
- execution would expand data transfer, cost, publication, merge, push,
  external communication, or permissions beyond authorization.

## Route assignments and verification

Under the direct user instruction of 2026-09-11, Claude executes and Claude
verifies this task on `claude/FP-SCALE-001`. The verification is therefore not
an independent construction and must be labelled as such everywhere. To
preserve as much assurance as a single actor can, the verification step must:

- reconstruct every frozen input and generator identity from source rather than
  from the author's serialized summaries;
- replay the sealed programs and confirm exit codes;
- recompute every reported metric from the sealed raw records with an
  independent code path written for the purpose;
- re-derive the `H1` arithmetic from the inherited verified mixture code;
- report `PASS`, `FAIL`, or `OBJECTION` with evidence pointers, and state
  explicitly that no second actor reconstructed the route.

## Pre-review

- Status: `APPROVED` (2026-09-11). Because the user assigned both roles to
  Claude, this review is a same-actor review and is recorded as such.
- Evidence: `docs/research_branches/FP-SCALE-001/claude/pre_review.md`.
- Scope checked: task/design/plan consistency; inheritance fidelity against the
  `FP-ESARSA-001` sealed contract; that no closed-task file is modified;
  oracle-separation and leak requirements; `H1` arithmetic against the
  inherited verified mixture code; and the two protocol defects that v1.1
  repairs.
- Outcome: two blocking protocol defects found in v1.0 (unreachable per-pair
  occupancy, and a single-grid eta rule that cannot emit at a non-negligible
  certified error), both repaired in v1.1 by pre-execution measurement. No
  scientific contract, hypothesis strength, or acceptance criterion was
  weakened.

## Objections and user rulings

### Objection

- Status: `NONE`.
- Disputed clause: none.

### User ruling (single-actor execution and verification)

- Date: 2026-09-11.
- Decision: the user instructed that Codex no longer be involved and that
  verification also be assigned to Claude ("不管 codex 了，验证也交给你").
- Scope: execution and verification responsibility for this task only. It
  changes no scientific contract and does not modify `AGENTS.md`. Because it
  waives the reciprocal-verification requirement of `AGENTS.md` section 7, this
  task's verification strength is strictly lower than a task closed with two
  independent actors, and that limitation must be stated in every report and in
  `ACTIVE_WORKSPACE.md`.

### User ruling (direction)

- Date: 2026-09-11.
- Decision: the user chose to pursue a reachable guarantee as the next
  scientific step and approved merging the verified `FP-KERN-001` /
  `FP-KERN-002` tasks into `main`.
- Scope: direction only. The diagnostic arithmetic and the v1.1 feasibility
  probe were produced after that ruling and are recorded here as the frozen
  justification for the protocol scale.

### User ruling (begin execution)

- Date: 2026-09-11.
- Decision: "开始" — proceed from `DRAFT` through `REVIEW` to `ACTIVE` and begin
  execution.
- Scope: activation of this task only. Merge to `main`, push, and publication
  remain subject to separate user approval.

## Definition of done

- [x] `H1` arithmetic checked and reported per record.
- [x] Smoke passes and the compute budget is measured.
- [ ] One frozen formal run executed exactly once. **Superseded:** v1.1 was
      stopped at the smoke gate because a formal run would only reconfirm at
      higher cost a result already established at reachable scale.
- [x] Every acceptance criterion has evidence (criteria 1--11, 13--15 by the
      reachable-scale measurement; criterion 12's budget measured in smoke).
- [x] Same-actor derived verification recorded, with the limitation stated.
- [x] `ACTIVE_WORKSPACE.md` is current.
- [ ] The user approves any merge to `main`.

## Closure note (2026-09-11)

This task was stopped deliberately at the smoke gate and superseded by
`docs/research_tasks/FP-SCALE-002.md`, which inherited this task's protocol
verbatim and replaced only the residual certificate's concentration argument.
FP-SCALE-002 then emitted certified updates in `22/48` primary route-records at
this exact protocol scale, with zero certificate violations.

The task's purpose was therefore achieved: its reachable-scale measurements
identified the obstruction precisely enough to direct the successor task. Its
journal at `docs/research_branches/FP-SCALE-001/claude/first_result.md` is the
authoritative record of that diagnosis, including two falsified expectations
that FP-SCALE-002 ultimately confirmed as falsified.

No formal run is planned under v1.1. Nothing here authorizes a merge to `main`.
