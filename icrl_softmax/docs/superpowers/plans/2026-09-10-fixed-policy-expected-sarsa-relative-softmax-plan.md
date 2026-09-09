# Fixed-policy Expected SARSA and relative-softmax implementation plan

> Date: 2026-09-10
> Intended task: `docs/research_tasks/FP-ESARSA-001.md`
> Design:
> `docs/superpowers/specs/2026-09-09-fixed-policy-expected-sarsa-relative-softmax-design.md`
> Scientific/code baseline: `c579047950dfabb2600020cd2e53dd24b3e39c84`
> Design commit: `75fc07217ec8a3e096804caade436ab8c2358659`
> Constraint: this plan is documentation only. No proof execution,
> implementation, smoke, or experiment begins until `FP-ADV-001` is verified
> or the user records a scheduling exception, Claude returns `APPROVED`, and
> `FP-ESARSA-001` becomes `ACTIVE`.

The `writing-plans` skill required by the brainstorming workflow is not
available in the current session. This document is a manual task-level plan
following the repository's existing plan structure and governance contract.

## Task 1: Close the predecessor and freeze the successor baseline

Finish only the remaining command-level reciprocal verification for
`FP-ADV-001`; do not rerun its formal evaluator or alter its frozen science.
Move the predecessor through `VERIFYING` to `VERIFIED` only when its own
requirements are satisfied.

Then create `codex/FP-ESARSA-001` and `claude/FP-ESARSA-001` from one recorded
baseline containing the approved design, DRAFT task, and this plan. Confirm
that none of the current uncommitted `FP-ADV-001` files or the user-owned
learning record enters the successor baseline accidentally.

## Task 2: Move the task through REVIEW without executing research

Record the DRAFT commit and move `FP-ESARSA-001` to `REVIEW`. Update
`ACTIVE_WORKSPACE.md` with a compact pointer only after the predecessor gate is
satisfied.

Send local Claude Code only the task-scoped governance, task, design, plan,
inherited mixture theory, relevant source, and exact baseline identities for a
read-only pre-review. Require one outcome:

- `APPROVED`, with itemized filtration, concentration, operator, attention,
  policy-improvement, software, protocol, and governance checks; or
- `OBJECTION`, with the disputed clause, evidence, validity impact, and user
  ruling options.

Record an objection as `BLOCKED_BY_OBJECTION` and stop. On approval, record the
review, freeze the activation commit, move the task to `ACTIVE`, and create the
two isolated execution worktrees and result roots.

## Task 3: Independently prove the mathematical chain

Before implementation, each route writes its own `theory.md` deriving:

1. the pair-MRP representation of `Q^pi`;
2. exact synchronous Expected SARSA from grouped residual means;
3. the finite successor-state mass, current-pair mass, and writeback weights;
4. the exact-residual kernel identity `F_pi(Q^pi)=Q^pi` and why it does not
   cancel finite successor/current-read leakage;
5. the declared diagonal-margin contraction bound;
6. conditional measurability of `Qhat` at the train/validation split;
7. the held-out visit-indexed martingale construction for each pair;
8. the `2B` sub-Gaussian scale and simultaneous 15-component mixture event;
9. the Bellman-residual implication
   `||Qhat-Q^pi||_infinity<=epsilon_res/(1-gamma)`;
10. exact relative-softmax statewise improvement;
11. the robust approximate-Q lower bound and pointwise policy theorem;
12. the selective probability statement and why eta selection spends no new
    risk on the simultaneous Q event.

Stop before coding if any mandatory step fails. A minimal deterministic
counterexample is the required negative evidence; formulas may not be repaired
after empirical output is inspected.

## Task 4: Write failing construction tests first

Create `verify_fixed_policy_expected_sarsa.py` before its implementation
module. Record the expected import or missing-symbol failure.

The first test group covers:

- canonical Q-memory uniqueness and duplicate rejection;
- exact policy attention from `log pi`;
- exact current-pair retrieval;
- terminal-free Expected SARSA residuals;
- self-loops where one Q token is read by two heads;
- repeated visits and exact residual averaging;
- unvisited exact queries and the zero null token;
- synchronous rather than within-batch sequential updates.

The second group covers direct finite-score matrices for the successor,
current-read, and pair-writer heads, including unvisited queries, finite row
sums, signed values, and the absence of equality masks or visited gates.

## Task 5: Implement the pure operator and literal attention modules

Create `fixed_policy_expected_sarsa.py` with pure validation and reference
functions for:

- canonical pair indexing and duplicate detection;
- exact and finite successor-policy aggregation;
- exact and finite current-pair reads;
- Expected and sampled SARSA residuals;
- exact grouped and finite pair-kernel writeback;
- synchronous route iteration and diagnostics;
- relative-softmax candidates and statewise estimated improvements.

Add only focused, task-scoped classes to `model.py`:

- `FixedPolicyActionExpectation`;
- `EndToEndMaskedSoftmaxExpectedSARSA`;
- `EndToEndFiniteSoftmaxExpectedSARSA`.

The tensor classes must call standard softmax and expose attention matrices for
direct verification. No trained parameters, transition model, equality mask
in the finite route, visited gate in the finite route, policy mutation, or
hidden Q copy is permitted.

## Task 6: Implement the held-out residual certificate

In `fixed_policy_expected_sarsa.py`, add a pure certificate builder that:

1. receives only `Qhat`, fixed policy, held-out transitions, `R_star`,
   `gamma`, `delta`, and declared metadata;
2. rejects `||Qhat||_infinity>B`, duplicate pair memory, malformed policy,
   nonfinite data, or a missing held-out pair;
3. computes exact held-out Expected SARSA residual means per pair;
4. invokes the frozen 15-component mixture grid with `d=|S||A|` groups;
5. applies radius `2B q_mix(N_x;d,delta)/N_x`;
6. returns `epsilon_res` and `E_Q=epsilon_res/(1-gamma)`;
7. keeps all truth-based quantities outside its function signature and output.

Extend the verifier with independent high-precision mixture roots, random
count fixtures, filtration-sensitive train/validation leak counterexamples,
direct Bellman solves, adversarial residual errors, ordered failure reasons,
and strict JSON.

## Task 7: Implement and verify the policy decision

For eta values `[1.0, 0.5, 0.2, 0.1, 0.05]` in descending order, build

```text
pi_eta^+ proportional to pi exp(eta Qhat).
```

Use stable logits `log pi + eta Qhat` and statewise normalization. Compute the
frozen `Ihat_s` and `LB_s` formulas. Accept the first finite, strictly positive,
row-normalized, changed policy whose lower bounds are nonnegative in all states
and strictly positive in at least one state. Otherwise return the original
policy bit-for-bit.

Test exact exponential-tilt improvement, adversarial Q-error corners, eta
tie-breaking, overflow resistance, invalid policies, zero Q rows, policy
identity on abstention, and direct pointwise value comparisons.

## Task 8: Build the matched evaluator with strict split isolation

Create `evaluate_fixed_policy_expected_sarsa.py`. Reuse the MDP generator,
policy generator, stationary-start rule, rollout, and child-seed schedule from
the frozen fixed-policy evaluator without changing legacy source behavior.

For each matched record:

1. generate one trajectory of the frozen total length;
2. expose only its first half to each Q-construction route;
3. seal each `Qhat` before passing the second half to the certificate builder;
4. run `expected_exact`, `expected_finite`, and `sampled_exact`;
5. construct and certify the relative-softmax policy independently per route;
6. serialize construction, attention, coverage, certificate, candidate, and
   decision fields below task-owned namespaces;
7. compute true Q error, Bellman residual, pair occupancy, contraction premise,
   componentwise value change, and return change only in `oracle_audit`.

The evaluator refuses a nonempty output directory, writes strict JSON, records
every explicit CLI constant, and never tunes or restarts a scientifically
complete formal run.

## Task 9: Build the independent analyzer

Create `analyze_fixed_policy_expected_sarsa.py` to reconstruct every route from
serialized observable inputs and verify:

- exact formulas and direct finite-score attention;
- train/held-out disjointness and seed identity;
- mixture roots, radii, residual bounds, and emitted `E_Q`;
- candidate order, LCBs, policy decision, and abstention identity;
- no certificate-to-oracle data flow;
- exact record and route counts;
- strict JSON without duplicate keys, NaN, or Infinity;
- all oracle violations and negative outcomes as separate descriptive fields.

It writes `regression.json`, `summary.json`, and `checks.log` without modifying
the evaluator's source records.

## Task 10: Run and seal smoke evidence

Each route independently runs all inherited relevant verifiers, the new
verifier, task-scoped Ruff, and a labelled smoke matrix spanning:

- exact, finite, and sampled routes;
- both mixing and reward-gap settings;
- missing and complete held-out support;
- self-loops and repeated pairs;
- certificate emission and ordinary abstention fixtures;
- finite-route unvisited-query leakage;
- valid and violated diagonal-margin fixtures.

Record exact commands, environment, hashes, failed and successful attempts,
metrics, anomalies, limitations, and acceptance assessment in
`first_result.md`. Commit the implementation and smoke evidence as a blind
first-result seal before formal evaluation.

## Task 11: Run the sole frozen formal evaluation

After its own smoke seal, each route runs exactly one formal evaluator command
with:

```text
--tasks 30
--trajectory-lengths 256 1024 4096 16384
--n-states 6
--n-actions 4
--pi-min 0.05
--mixing 0.08 0.5
--gap-bonuses 0 0.5
--reward-bound 1.5
--gamma 0.70
--alpha 0.65
--iterations 160
--state-sharpness 8
--read-sharpness 8
--kernel-sharpness 8
--eta-grid 1.0 0.5 0.2 0.1 0.05
--certificate-delta 0.05
--mixture-components 15
--seed 20260829
--output-dir results/FP-ESARSA-001/<actor>
```

The command must yield exactly 480 matched task records, each containing all
three routes. A
mechanical interruption before complete readable output may be documented and
restarted with the identical command. A scientifically complete output is the
sole formal result and may not be rerun to improve metrics.

Each route writes and commits `formal_result.md` with raw hashes, exact metrics,
all failures, negative findings, limitations, and a numbered acceptance
assessment before disclosure.

## Task 12: Disclose, reproduce, and synthesize

After both routes seal first and formal evidence, GPT reproduces Claude's proof,
implementation, formal configuration, hashes, and every acceptance criterion.
Claude independently performs the symmetric command-level verification of GPT.
Each report ends exactly `PASS`, `FAIL`, or `OBJECTION`.

An implementation or evidence defect is `FAIL` and returns the author route to
`ACTIVE` for repair. A task-definition defect is `OBJECTION` and requires user
ruling. Neither outcome authorizes retuning the frozen empirical contract.

After reciprocal `PASS`, GPT writes shared theory and report, distinguishing:

- exact construction identities;
- conditional population convergence;
- high-probability held-out residual certification;
- emitted and abstained updates;
- oracle return and value diagnostics;
- comparison with sampled SARSA and the prior V-first negative result;
- limitations of pair coverage, diagonal mass, finite logits, fixed policy,
  canonical memory, and one-step-only control.

Only then may the task move through `VERIFYING` to `VERIFIED` and update
`ACTIVE_WORKSPACE.md`. No merge to `main` or push occurs without separate user
approval.
