# FP-ESARSA-001: Fixed-policy Expected SARSA and certified relative-softmax improvement

## Task metadata

- Created: 2026-09-10.
- Author: GPT (v1.0). The v1.1 revision is recorded by Claude under the
  direct user ruling of 2026-09-11 (Codex execution quota exhausted); it
  changes responsibilities, baseline, and the allowed read-only input list
  only, and no scientific input, hypothesis, formula, tolerance, matrix, or
  acceptance strength.
- Status: `VERIFIED` (2026-09-11; closed by the acceptance-exemption user
  ruling below, not by reciprocal verification).
- Task version: `1.1`.
- Scientific/code baseline: `9d0994f03e4a659787c74df4660cdcab1d398c1c`
  (main after the FP-EXPL-001 merge). Every inherited generator, MDP, model,
  mixture-certificate, and fixed-policy file is byte-identical to the v1.0
  baseline `c579047950dfabb2600020cd2e53dd24b3e39c84`, so this refresh is
  scientifically neutral; c579047 remains the historical v1.0 baseline.
- Approved design commit: `75fc07217ec8a3e096804caade436ab8c2358659`.
- Design:
  `docs/superpowers/specs/2026-09-09-fixed-policy-expected-sarsa-relative-softmax-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-10-fixed-policy-expected-sarsa-relative-softmax-plan.md`.
- Drafting location: `codex/FP-ADV-001`; this records planning only and is not
  an execution branch.
- Intended execution branches after activation: `codex/FP-ESARSA-001` and
  `claude/FP-ESARSA-001` from the common activation commit recorded during
  `REVIEW`.
- Activation prerequisite: `FP-ADV-001` is `VERIFIED`, or the user records an
  explicit scheduling exception. This task cannot enter `REVIEW` or execute
  research while that prerequisite is false. **Satisfied on 2026-09-11**:
  FP-ADV-001 is VERIFIED (negative usefulness result).
- Classification: long, multi-stage, and conclusion-critical. v1.0 required
  independent GPT and Claude construction with reciprocal verification. By
  the direct user ruling of 2026-09-11 ("直接例外", Codex execution quota
  exhausted), this task adopts the FP-EXPL-001 v1.1 responsibility pattern:
  Claude main execution with author seal, GPT independent acceptance of the
  sealed route (executable reconstruction, no import of Claude operators),
  and Claude reciprocal review of the acceptance evidence. This is a
  task-scoped exception and does not modify `AGENTS.md`. GPT's acceptance
  verdict must stand on its own executable artifact; if GPT remains
  unavailable, results stay preliminary until its acceptance or an explicit
  user exemption.
- Estimated resources: four to seven hours per independent actor, CPU-only,
  one smoke matrix and one frozen 480-record formal run per actor, with no new fee
  category.

## Research question

For a finite MDP and one frozen full-support policy, can standard normalized
softmax attention implement a state-action Expected SARSA residual iteration,
produce an observable cross-fitted Bellman-residual certificate for its final
`Q` estimate without an environment model, and use that certificate to emit
one pointwise non-degrading relative-softmax policy update?

## Falsifiable hypotheses

1. With one canonical Q-memory token per state-action pair, the exact grouped
   attention construction equals synchronous batch Expected SARSA, including
   self-loops where one Q entry has both current and successor roles.
2. The finite-logit route uses only standard normalized softmax, finite scores,
   unique Q-memory tokens, fixed policy logits, and no input-dependent equality
   mask or visited-query gate; its direct tensor output equals the declared
   finite-softmax formulas.
3. On the pair MRP, the exact-residual kernel population operator
   `F_pi(Q)=Q+alpha M_pi^X(T_pi^X Q-Q)` preserves `Q^pi`; under the frozen
   diagonal-margin premise its linear part is a strict infinity-norm
   contraction.
4. Conditional on the training prefix, the held-out visit-indexed Expected
   SARSA residuals form bounded martingale differences around
   `(T_pi Qhat-Qhat)(s,a)`. The frozen simultaneous mixture boundary therefore
   yields an observable valid residual bound at random pair counts.
5. The Bellman-residual certificate
   `E_Q=max_x(|delta_bar_x|+r_x)/(1-gamma)` bounds
   `||Qhat-Q^pi||_infinity` without using a true transition kernel, true value,
   realized error, action gap, or return.
6. For exact `Q^pi`, the relative-softmax tilt
   `pi_eta^+ proportional to pi exp(eta Q^pi)` has nonnegative statewise
   expected advantage for every `eta>=0`.
7. Replacing `Q^pi` by `Qhat` and subtracting
   `E_Q ||pi_eta^+(.|s)-pi(.|s)||_1` gives a valid simultaneous statewise lower
   bound. Emitting only when all states have nonnegative lower bounds implies
   `V^{pi_plus}>=V^pi` componentwise on the certificate event.
8. At least one primary Expected SARSA route emits a nontrivial certified
   update in the frozen 480 records. If no primary route emits, zero usefulness
   is a valid verified negative result and no formula, sharpness, eta grid,
   split, or matrix may be retuned.

Hypotheses 1--7 are mandatory mathematical, construction, and software claims.
Hypothesis 8 is the empirical usefulness claim.

## Frozen mathematical contract

### Fixed policy and pair MRP

Let `X=S x A`, `x=(s,a)`, and freeze `pi(a|s)` for the complete evaluation
phase. Define

```text
P_pi^X((s',b)|(s,a)) = P(s'|s,a) pi(b|s'),

(T_pi^X Q)(s,a)
  = r(s,a)
    + gamma sum_{s',b} P(s'|s,a) pi(b|s') Q(s',b).
```

The unique fixed point is `Q^pi`. The algorithm receives transition samples
and the known policy probabilities, not `P` or `r`.

Rewards satisfy `|R_{t+1}|<=R_star`, and

```text
B = R_star/(1-gamma).
```

Any route with `||Qhat||_infinity>B` is rejected by
`divergence_guard_triggered`; no clipping or oracle repair is allowed.

### Canonical memory and duplicate contract

There is exactly one persistent Q-memory token for every logical pair. A
transition token carries observed fields and one residual but is not an
additional Q candidate.

- equal numerical values with different pair identifiers remain distinct;
- repeated visits remain separate residual samples;
- one canonical token may be read in separate heads;
- the same logical pair may not appear twice in one attention candidate set.

The primary serialization uses unique tokens. A multiplicity-corrected
alternate serialization is outside this task.

### Exact grouped Expected SARSA

Split every length-`n` trajectory at `m=n/2`. Only the first `m` transitions
enter Q construction. Initialize `Q_0=0`. For layer `l`, compute from the
unchanged `Q_l`

```text
qbar_t^l = sum_b pi(b|S_{t+1}) Q_l(S_{t+1},b),

delta_t^l
  = R_{t+1} + gamma qbar_t^l - Q_l(S_t,A_t).
```

For `x=(s,a)`, let `N_x^train` be its training count. The exact grouped route
uses

```text
Q_{l+1}(x)
  = Q_l(x)
    + alpha/N_x^train sum_{t<m:X_t=x} delta_t^l
```

when `N_x^train>0`, and leaves `Q_l(x)` unchanged otherwise. All pair updates
are synchronous.

The exact action-expectation head masks to the successor state's unique action
tokens and uses scores `log pi(b|S_{t+1})`. The exact current-pair head admits
only the unique current-pair token. The exact writer admits only transitions
with the queried current pair, or a zero-value null token for an unvisited
query.

### Finite-logit standard-softmax route

The finite route has no equality mask and no visited-query gate. It uses fixed
one-hot identifiers and three frozen sharpness values

```text
zeta = xi = tau = 8.
```

For successor query state `s'` and canonical token `(u,b)`, the action-head
score is

```text
zeta 1{u=s'} + log pi(b|u).
```

Because each policy row sums to one, total attention mass on the requested
state is

```text
kappa_state = exp(zeta)/(exp(zeta)+|S|-1),
```

and, conditional on that state, action weights equal `pi(.|s')` exactly.

For current query pair `x` and canonical pair token `y`, the read score is

```text
xi 1{x=y},
```

with requested-pair mass

```text
kappa_read = exp(xi)/(exp(xi)+|S||A|-1).
```

For pair query `x` and training transition `t`, the write score is

```text
tau 1{X_t=x}.
```

Softmax is normalized over every training transition. The route updates every
canonical query with the resulting signed residual average. All scores are
finite; off-group mass and updates of unvisited queries are reported rather
than hidden.

### Exact-residual kernel population operator

Let `M_pi^X` be the row-stochastic population pair kernel induced by the fixed
writeback attention. When the value supplied to that kernel is the exact
fixed-policy Bellman residual, the population update is

```text
F_pi(Q) = Q + alpha M_pi^X(T_pi^X Q-Q).
```

It must be proved that `F_pi(Q^pi)=Q^pi`. For `0<alpha<=1`, the frozen
sufficient contraction premise is

```text
min_x M_pi^X(x,x) >= (1+gamma)/2+C,
0<C<(1-gamma)/2,
```

under which the required conservative bound is

```text
||I-alpha M_pi^X+alpha gamma M_pi^X P_pi^X||_infinity
  <= 1-2 alpha C < 1.
```

The theorem is conditional on this premise. Empirical failure of the premise
must be reported and cannot be replaced by an oracle diagonal.

This identity does not cancel successor-head or current-read leakage. The full
finite-logit route can have a shifted population fixed point because its
residual need not vanish at `Q^pi`. Its final estimate is instead certified by
the exact held-out Bellman residual below; no asymptotic no-bias claim is made
for the full finite route.

## Frozen cross-fitted residual certificate

### Held-out residuals

Freeze the final training estimate `Qhat` before reading the held-out suffix.
For every held-out transition define

```text
Y_t(Qhat)
  = R_{t+1}
    + gamma sum_b pi(b|S_{t+1}) Qhat(S_{t+1},b)
    - Qhat(S_t,A_t).
```

The policy expectation in this certificate is exact arithmetic over the known
finite action row. It uses neither a sampled next action nor an environment
model. Conditional on the training filtration and on `X_t=x`,

```text
E[Y_t(Qhat)|X_t=x] = (T_pi^X Qhat-Qhat)(x).
```

When `||Qhat||_infinity<=B`, every held-out residual lies in `[-2B,2B]`.

### Simultaneous time-uniform radius

Let `d=|S||A|`. Reuse exactly the verified 15-component geometric cosh-mixture
grid and conservative numerical inversion implemented by
`build_mixture_grid` and `solve_mixture_boundary` in
`time_uniform_mixture_certificate.py`, but construct a new event with exactly
`d` held-out pair-residual groups and total risk `delta`.

For held-out count `N_x`, let `q_mix(N_x;d,delta)` denote the returned boundary
and define

```text
r_x = 2B q_mix(N_x;d,delta)/N_x.
```

No radius exists at `N_x=0`. The event is simultaneous over every pair and
valid at the random visit counts. On it,

```text
|mean_{held-out t:X_t=x} Y_t(Qhat)
 -(T_pi^X Qhat-Qhat)(x)| <= r_x.
```

### Q-error certificate

If every pair has positive held-out count, all numerical checks pass, and the
divergence guard passes, define

```text
epsilon_res
  = max_x (abs(Ybar_x(Qhat))+r_x),

E_Q = epsilon_res/(1-gamma).
```

Bellman contraction gives

```text
||Qhat-Q^pi||_infinity <= E_Q.
```

The certificate is model-free in its inputs: it receives the frozen policy,
`Qhat`, held-out transitions, `R_star`, `gamma`, `delta`, and public mixture
constants. True MDP quantities are prohibited.

## Frozen relative-softmax improvement

Use the candidate grid

```text
eta in [1.0, 0.5, 0.2, 0.1, 0.05]
```

in that descending order. For each state and candidate,

```text
pi_eta^+(a|s)
  = pi(a|s) exp(eta Qhat(s,a))
    / sum_b pi(b|s) exp(eta Qhat(s,b)),

Ihat_s(eta)
  = sum_a [pi_eta^+(a|s)-pi(a|s)] Qhat(s,a),

LB_s(eta)
  = Ihat_s(eta)
    - E_Q ||pi_eta^+(.|s)-pi(.|s)||_1.
```

Choose the first candidate with finite probabilities, exact row sums within
`1e-12`, strict positivity, a changed policy, `LB_s>=0` for every state, and
at least one `LB_s>0`. If none passes, return the input policy bit-for-bit and
record an abstention.

`pi_min=0.05` is a data-collection floor for the frozen behavior policy. The
one-step output is required to have strict positive support but is not claimed
to preserve the same numerical floor. Repeated control is outside this task.

On the simultaneous residual event, every emitted update must satisfy

```text
T_{pi_plus}V^pi >= V^pi,
V^{pi_plus} >= V^pi
```

componentwise. The only probability claim is

```text
P(EmitUpdate and exists s: V^{pi_plus}(s)<V^pi(s)) <= delta.
```

There is no claim of high-probability support, nonempty emission, or coverage
conditional on emission.

## Inputs and frozen formal protocol

- Tasks per cell: 30.
- States/actions: 6/4.
- Total trajectory lengths: 256, 1024, 4096, 16384.
- Contiguous split: first half training, second half held out.
- Mixing settings: 0.08 and 0.5.
- Reward-gap bonuses: 0 and 0.5.
- Behavior-policy minimum action probability: 0.05.
- Reward bound: `R_star=1.5`, because the inherited base reward lies in
  `[-1,1]` and the largest frozen action bonus is `0.5`.
- Gamma/alpha/evaluation layers: 0.70/0.65/160.
- Finite logits: `zeta=xi=tau=8`.
- Eta candidates: 1.0, 0.5, 0.2, 0.1, 0.05 in descending order.
- Certificate delta: 0.05.
- Mixture components: 15.
- Seed: 20260829.
- Total matched task records: 480, each containing all three routes.
- Q initialization: all zeros.

The MDP generator, fixed-policy generator, stationary-start rule, trajectory
sampler, task seed spawning, and truth computation are inherited unchanged
from `evaluate_fixed_policy_q_routes.py` at the scientific baseline. Exact
truth enters only the oracle audit.

### Routes

1. `expected_exact`: exact grouped fixed-policy Expected SARSA, primary.
2. `expected_finite`: finite-logit mask-free fixed-policy Expected SARSA,
   primary.
3. `sampled_exact`: exact grouped sampled SARSA using the observed next action,
   control.

All three routes use the same training/held-out split, residual certificate,
relative-softmax candidates, and policy decision rule. The sampled control has
no claim of variance dominance.

## Ordered non-emission reasons

Use this frozen order:

1. `algorithm_mode_mismatch`;
2. `duplicate_q_memory`;
3. `divergence_guard_triggered`;
4. `heldout_pair_support_missing`;
5. `mixture_inversion_unbracketed`;
6. `mixture_inversion_not_converged`;
7. `mixture_root_not_conservative`;
8. `numerical_nonfinite`;
9. `policy_invalid`;
10. `improvement_lcb_nonpositive`;
11. `policy_unchanged`.

Reasons 10 and 11 are ordinary abstentions. No reason may trigger an oracle
fallback, a different eta grid, a different Q route, or a second risk budget.

## Allowed work

### Shared read-only inputs

- `AGENTS.md`, `ACTIVE_WORKSPACE.md`, this task, design, and plan;
- verified `FP-MART-001` and `FP-TU-001` theory, mixture code, and verifiers;
- existing SARSA, two-stage Q-control, fixed-policy, MDP, and strict-JSON code;
- the frozen `FP-TU-001` baseline for regression and generator identity;
- verified `FP-EXPL-001` theory, literal-network witness/verifier, and sealed
  results (v1.1 addition: read-only construction reference for the grouped
  fixed-policy literal attention route; no FP-EXPL-001 operator may be
  imported as a verification computation basis);
- task-scoped primary references needed for assumption mapping.

### GPT write scope on `codex/FP-ESARSA-001`

Under the v1.1 responsibility exception, GPT's write scope is the acceptance
side, exercised when its execution quota is restored:

- an independent acceptance verifier (e.g. `verify_claude.py`) and
  `docs/research_branches/FP-ESARSA-001/codex/`;
- `results/FP-ESARSA-001/codex/`;
- this task, its design, plan, synthesis/handoff records, and a compact
  `ACTIVE_WORKSPACE.md` pointer.

The v1.0 GPT implementation scope (additive task-scoped classes in
`model.py`, `fixed_policy_expected_sarsa.py`,
`verify_fixed_policy_expected_sarsa.py`,
`evaluate_fixed_policy_expected_sarsa.py`,
`analyze_fixed_policy_expected_sarsa.py`, and the codex theory/report files)
is transferred to Claude's main route; GPT may still author them later only
as a separate task or after a user ruling.

### Claude write scope on `claude/FP-ESARSA-001`

- the main route: additive task-scoped classes in `model.py`,
  `fixed_policy_expected_sarsa.py`, `verify_fixed_policy_expected_sarsa.py`,
  `evaluate_fixed_policy_expected_sarsa.py`,
  `analyze_fixed_policy_expected_sarsa.py`;
- `docs/research_branches/FP-ESARSA-001/claude/`;
- `docs/research_branches/fixed_policy_expected_sarsa_theory.md` and
  `docs/research_branches/fixed_policy_expected_sarsa_report.md` (main-route
  versions, authored by Claude);
- `results/FP-ESARSA-001/claude/`;
- its reciprocal verification report of the GPT acceptance evidence.

Claude may record only its own proof, commands, results, limitations, and
verification report in its assigned branch and directories.

## Prohibited work

- No implementation, proof execution, smoke, or formal experiment before the
  task becomes `ACTIVE`.
- No activation before the predecessor gate and Claude read-only pre-review.
- No policy change inside the 160 Q-evaluation layers.
- No input-dependent equality mask or visited gate in `expected_finite`.
- No duplicate logical Q-memory candidates, sampled next action in Expected
  routes, learned transition model, or oracle certificate input.
- No use of validation transitions to construct, tune, stop, select, or warm
  start `Qhat`.
- No post-hoc change to split, event, risk allocation, eta rule, sharpness,
  route, seed, matrix, hypothesis, or metric after smoke or formal inspection.
- No repeated policy iteration, conditional-on-emission guarantee, or output
  exploration-floor claim.
- No write to `main`, the other route, old result directories, manuscript,
  archive, frozen `FP-ADV-001` evidence, or user-owned learning record.
- No merge, push, publication, external message, permission bypass, or new fee
  category without separate authorization.

## Expected artifacts

Each route provides:

- independent `theory.md`, `first_result.md`, and `formal_result.md`;
- exact environment, commands, commits, raw hashes, failures, anomalies, and
  limitations;
- four new Python entry points and task-scoped model additions;
- ignored `config.json`, `task_results.json`, `summary.json`,
  `regression.json`, `environment.json`, `commands.log`, and `checks.log`;
- a later reciprocal verification report ending `PASS`, `FAIL`, or
  `OBJECTION`.

## Acceptance criteria

1. Canonical memory contains exactly one token per logical pair and rejects a
   duplicate candidate serialization.
2. Exact action expectation, current read, signed residual, and writeback equal
   direct batch Expected SARSA within `1e-12` on deterministic fixtures.
3. Self-loop fixtures correctly use the same Q entry in two algebraic branches
   without duplicating it inside either softmax candidate set.
4. Repeated pair visits produce the exact residual mean; an unvisited exact
   query remains unchanged.
5. Finite successor, current-read, and writeback attention equal their direct
   finite-score formulas within `1e-12`, with finite row-normalized weights and
   no equality mask or visited gate.
6. The exact-residual pair-MRP kernel fixed-point identity and the declared
   diagonal-margin contraction bound are proved and verified by direct
   matrices; the full finite route makes no unproved no-bias claim.
7. The held-out residual martingale filtration, `2B` sub-Gaussian scale,
   random-count substitution, and union over exactly `d` pair groups are
   proved from the inherited mixture event.
8. Every emitted `E_Q` equals the frozen residual formula, uses full held-out
   pair support, and bounds the oracle Q error in every audited record.
9. The exact relative-softmax improvement inequality is proved analytically
   and exhaustively checked on finite fixtures.
10. The approximate-Q lower bound holds for adversarial error corners and
    every emitted route-state lower bound is nonnegative.
11. Every emitted policy is finite, strictly positive, row-normalized, changed,
    and componentwise non-degrading in the oracle audit.
12. Every abstention returns the original policy bit-for-bit and uses the
    frozen ordered reason list.
13. Training and held-out transitions are structurally separated; validation
    data cannot influence Q construction or hyperparameter selection.
14. Exact truth, realized errors, pair occupancy, action gaps, Bellman
    improvements, values, and returns live only in `oracle_audit`.
15. All strict-JSON, duplicate-key, nonfinite, shape, range, task identity,
    branch, seed, configuration, and result-location checks pass.
16. The formal result contains exactly 480 matched task records, each with all
    three routes, and reproduces the frozen generator identities and seed
    schedule.
17. All empirical certificate violations, value decreases, return decreases,
    exact/finite discrepancies, and contraction-premise failures are
    enumerated and excluded from theorem evidence.
18. Hypothesis 8 is evaluated without retuning; zero primary emissions are
    reported as a verified negative usefulness result.
19. Both routes seal implementation/smoke and formal evidence before mutual
    disclosure and record complete reproducibility evidence.
20. GPT verifies Claude and Claude verifies GPT; both reports pass, or the task
    remains unverified pending repair or user ruling.

## Failure criteria

The mandatory theory fails if held-out residuals are not martingale-valid
after conditioning on the training prefix, the inherited mixture cannot be
used at random visit counts with scale `2B`, Bellman residual does not imply
the stated Q bound, the relative-softmax lower bound is false, or an unstated
model/oracle input is required.

The construction fails if a logical pair is duplicated within an attention
candidate set, the exact route differs from Expected SARSA, the finite route
uses hidden masks/gates, signed residuals are lost, policy changes during
evaluation, validation leaks into Q construction, or an emitted update can
degrade any state on the certificate event.

Hypothesis 8 may fail without invalidating the task. All unaffected mandatory
criteria must still pass, and reciprocal verification is still required.

## Stopping conditions

Stop affected work and notify the user if:

- `FP-ADV-001` is not verified when activation is requested and no explicit
  scheduling exception exists;
- Claude pre-review returns `OBJECTION`;
- the common activation baseline or route isolation cannot be preserved;
- the `2B` residual-mixture argument fails under the exact inherited event;
- implementation needs a prohibited model, mask, gate, oracle, validation
  leak, or formula change;
- smoke or formal output would be needed to choose a frozen parameter;
- an existing uncommitted edit overlaps an authorized path;
- Claude is unavailable because of authentication, quota, permission, or
  environment failure;
- execution would expand data transfer, cost, publication, merge, push,
  external communication, or permissions beyond authorization.

## Route assignments and verification

Under the v1.1 exception, Claude executes the main route after activation:
derive the proof, write tests before implementation, implement the three
routes, run labelled smoke, seal the first result, and run exactly one
frozen 480-record formal evaluation, then seal code, theory, report and raw
results with a `[claude]` commit before any acceptance begins.

After the Claude seal, GPT independently reconstructs all frozen inputs and
key numerical results, replays the sealed programs, reviews the literal
network/proofs and source boundaries, and records its acceptance verdict
with executable evidence. GPT does not read Claude's results before the
author seal exists. Claude then reviews GPT's acceptance evidence and
records its own reciprocal report. Each report must inspect formulas,
source, direct tensor witnesses, strict outputs, configuration, hashes,
oracle separation, every acceptance criterion, and a clean command-level
reproduction. A task-definition defect is `OBJECTION`; an implementation,
evidence, or inference defect is `FAIL` and returns that route to `ACTIVE`
for author repair. While GPT is quota-blocked, Claude's sealed results
remain preliminary; they are not VERIFIED without GPT acceptance or an
explicit user exemption.

## Claude read-only pre-review

- Status: `APPROVED` (2026-09-11; predecessor prerequisite satisfied the same
  day). Itemized evidence:
  `docs/research_branches/FP-ESARSA-001/codex/claude_pre_review.md`.
- Required input: the later frozen DRAFT commit containing this task, approved
  design, and plan, plus task-scoped inherited theory and source files.
  Available since the FP-EXPL-001 merge to main (`9d0994f`).
- Required outcome: itemized `APPROVED` or `OBJECTION`.
- No review may begin until the predecessor activation prerequisite is met.
- Transparency note: because Codex is quota-blocked, the v1.1 revision text
  itself is recorded by Claude under the user's direct ruling; the user is
  the arbiter of this deviation from the usual GPT-authored revision flow.
- This commit publishes `ACTIVE` after the itemized APPROVED and serves as
  the common execution baseline for `codex/FP-ESARSA-001` (acceptance side)
  and `claude/FP-ESARSA-001` (main route). No research run, sample, or
  experiment has occurred for this task.

## Objections and user rulings

### Objection

- Status: `NONE`.
- Disputed clause: none.
- Evidence: none.
- Validity impact: none.

### User ruling

- Date: 2026-09-10.
- Decision: the user approved the written fixed-policy Expected SARSA and
  certified relative-softmax design and authorized drafting the formal task
  and implementation plan.
- Scope: planning documents only; activation, implementation, experiments,
  merge, and push remain governed by this task and `AGENTS.md`.

### User ruling (v1.1)

- Date: 2026-09-11.
- Decision: with the Codex session out of execution quota, the user ruled
  directly ("直接例外") to waive the dual blind-construction requirement for
  this task and adopt the FP-EXPL-001 v1.1 pattern: Claude main execution,
  GPT independent post-seal acceptance, Claude reciprocal review. The user
  authorized Claude to record this v1.1 revision, including the baseline
  refresh to `9d0994f03e4a659787c74df4660cdcab1d398c1c` (scientifically
  neutral: all inherited science files byte-identical) and the addition of
  verified FP-EXPL-001 artifacts to the read-only input list.
- Scope: responsibilities, baseline, and input list only. No scientific
  input, hypothesis, formula, tolerance, matrix, or acceptance strength
  changed. Activation, experiments, merge, and push remain governed by this
  task and `AGENTS.md`.

### User ruling (2026-09-11, acceptance exemption)

- Date: 2026-09-11.
- Decision: with Codex still out of execution quota, the user selected the
  explicit-exemption path already written into acceptance criterion 20 ("or
  the task remains unverified pending repair or user ruling"). The Claude
  main route's sealed results are accepted and this task is closed
  `VERIFIED` **without** the independent GPT post-seal acceptance and
  without a Claude reciprocal review of GPT evidence, because no GPT
  acceptance artifact exists.
- Scientific limitation recorded: there is **no** independent executable
  reconstruction of the sealed 480-record route (criterion 20's "GPT
  verifies Claude"). All evidence is single-route: the author's executable
  verifier (13550 checks), the frozen evaluator run, and the oracle audit.
  This task's verification strength is therefore strictly lower than a task
  closed with both reciprocal reports passing; the negative-usefulness
  result for hypothesis 8 is reported, not independently reproduced.
- Scope: acceptance closure of this task only. It waives the independent
  GPT acceptance for FP-ESARSA-001; it changes no formula, hypothesis,
  tolerance, matrix, or acceptance-criterion text, and it does not modify
  `AGENTS.md`.

## Definition of done

- [x] Predecessor/scheduling gate is satisfied (FP-ADV-001 VERIFIED, 2026-09-11).
- [x] Claude pre-review returns `APPROVED` with no unresolved objection.
- [ ] Both independent routes and all required artifacts are reproducible.
  Only the Claude main route exists; the GPT acceptance route produced no
  artifact (Codex quota-blocked).
- [ ] Both reciprocal verification reports pass. **Waived** by the 2026-09-11
  acceptance-exemption user ruling; no GPT acceptance artifact exists.
- [x] Every acceptance criterion has evidence (criteria 1-19 by the Claude
  route; criterion 20 by the acceptance-exemption user ruling).
- [x] Differences are reconciled or ruled on by the user (exemption ruling).
- [x] `ACTIVE_WORKSPACE.md` is current.
- [x] The user approves any merge to `main` (2026-09-11).
