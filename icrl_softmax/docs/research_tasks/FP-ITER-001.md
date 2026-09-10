# FP-ITER-001: fixed-policy finite-softmax Expected SARSA iteration witness

## Metadata

- Date: 2026-09-10. Author: GPT. Version: 1.0. Status: `REVIEW`.
- Scientific baseline: `c579047950dfabb2600020cd2e53dd24b3e39c84`.
- Draft branch: `codex/CTRL-PREFLIGHT-001` until a dedicated execution branch
  can be created; branch creation is currently blocked by local Git metadata
  permissions. No implementation is authorized in this draft branch.
- Design: `docs/superpowers/specs/2026-09-10-fixed-policy-softmax-iteration-witness-design.md`.
- Predecessor: `CTRL-PREFLIGHT-001` is `VERIFIED` and supplies the finite-head
  interface comparison; `FP-ESARSA-001` remains an unactivated broader draft.
- Classification: bounded construction gate, CPU-only, estimated 30--60
  minutes per actor. It is conclusion-critical for the next control design,
  but contains no formal empirical matrix.

## Research question

With one frozen full-support policy `pi` on a two-state/two-action finite MDP,
can standard normalized softmax attention implement and repeatedly apply the
Expected SARSA evaluation operator, and can the resulting finite-logit error
be decomposed and bounded well enough to state an iteration-stability
condition?

## Falsifiable checks

1. The exact grouped route equals the direct synchronous Expected SARSA
   operator on every declared fixture, including self-loops and repeated
   visits.
2. The finite route's tensor output equals its independently written scalar
   finite-score formula to absolute error at most `1e-12`.
3. The finite route's discrepancy from the exact route is attributable to the
   declared attention stages: current read, successor state routing,
   successor action weighting, and pair writeback.
4. Repeated exact iterations match direct reference iterations and remain under
   the declared contraction condition.
5. Repeated finite iterations either satisfy a derived perturbation bound or
   produce a measured counterexample; no unconditional convergence claim is
   allowed.

## Frozen fixture and protocol

Use float64 and deterministic inputs only. The primary fixture has states
`{0,1}`, actions `{0,1}`, discount `gamma=0.7`, update step `alpha=0.5`, and
fixed policy

```text
pi = [[0.75, 0.25],
      [0.25, 0.75]]
```

Use a finite transition batch containing at least one self-loop, repeated
`(state, action)` visits, an unvisited pair, and both unequal and equal
successor-action values. The task author must freeze the complete rows,
initial `Q0`, number of iterations, and sharpness values before execution.
The direct reference, exact grouped route, and finite route consume the same
frozen rows and update synchronously from `Q_t` to `Q_{t+1}`.

The exact route may use declared grouping or equality routing as a mathematical
reference. The finite route may not silently use an equality mask, visited-query
gate, external Q indexing, or a hidden oracle copy. If a fixture requires one
of these for the exact comparison, record it as an external interface and do
not count it as a finite-network capability.

## Required implementation and evidence

The execution route may add only task-scoped files under
`docs/research_branches/FP-ITER-001/<actor>/` and ignored
`results/FP-ITER-001/<actor>/`, plus explicitly approved task-scoped modules
and verifiers. It must not alter `FP-ADV-001`, `FP-TU-001`, the old
`FP-ESARSA-001` draft, or the user's learning records.

Required artifacts:

- a pure direct Expected SARSA reference and Bellman fixed-point calculation;
- an exact grouped attention witness;
- a finite-logit attention witness exposing every attention matrix;
- a verifier that checks shape, normalization, signed residuals, synchronous
  writeback, self-loops, repeated visits, and unvisited-pair behavior;
- strict JSON with per-stage and per-iteration values;
- a report ending `PASS`, `FAIL`, or `OBJECTION`.

Every route records its commit, Python/Torch environment, command, successful
and failed checks, maximum errors, iteration traces, and limitations.

## Acceptance and stopping rules

Acceptance requires all five falsifiable checks to be decided with raw evidence.
The task returns to `ACTIVE` for an implementation defect and enters
`BLOCKED_BY_OBJECTION` for a task-definition defect. Stop if the finite route
needs an undeclared oracle, if exact and finite formulas cannot be separated,
if policy changes during evaluation, or if repeated finite updates violate the
stated bound without a documented counterexample.

The following outcomes are valid and must be reported plainly: exact witness
passes while finite stability fails; finite error is nonzero but bounded;
finite error grows with iteration; or the fixture has no nontrivial update.
None of these outcomes authorizes tuning the frozen protocol.

## Governance and next step

The task passed GPT's read-only self-review on 2026-09-10: no placeholder,
scope contradiction, undeclared oracle input, or acceptance ambiguity was
found. The user then authorized continuation. Lifecycle is now `REVIEW`.

The next required action is Claude's read-only pre-review. It must return
`APPROVED` with checks of the fixed-policy filtration, synchronous operator,
finite-score semantics, iteration bound, fixture freeze, artifact scope, and
governance rules, or return `OBJECTION` with the disputed clause and evidence.
No implementation or experiment may begin before `APPROVED` and a recorded
activation commit. GPT will independently verify the actor's raw outputs and
report. No merge, push, or main-branch change is authorized.

Claude pre-review on 2026-09-10 returned `APPROVED`; the itemized report is
`docs/research_branches/FP-ITER-001/codex/claude_review.md`. No file edit,
experiment, branch, commit, push, or unrelated-file access occurred during
the review. The task remains in `REVIEW` until a dedicated activation branch
and activation commit can be recorded.

The local environment currently exposes no callable Claude pre-review tool.
The earlier attempt to create the dedicated `codex/FP-ITER-001` branch was
rejected by the automatic approval review because the account usage limit was
reached; the draft therefore remains on `codex/CTRL-PREFLIGHT-001` and no
execution worktree has been created.
