# FP-ITER-001 Claude pre-review

- Date: 2026-09-10.
- Route: Claude Code CLI 2.1.138, read-only `plan` permission mode.
- Scope: `AGENTS.md`, `ACTIVE_WORKSPACE.md`, the FP-ITER-001 task and design,
  the CTRL-PREFLIGHT-001 report, and the cited sections of `model.py` and
  `verify_end_to_end_sarsa.py`.
- No file edit, experiment, branch, commit, push, or unrelated-file access was
  performed.

## Review checks

1. **Fixed policy and evaluation boundary — PASS.** The frozen policy is
   explicit, policy changes stop the task, and policy improvement is excluded.
2. **Operator separation — PASS.** Direct reference, exact grouped route, and
   finite route are independently named and testable; finite output is compared
   with an independently written scalar formula.
3. **Synchronous iteration and fixture — PASS.** Self-loops, repeated visits,
   an unvisited pair, equal and unequal successor-action values, float64, and
   frozen deterministic inputs are all required.
4. **Leakage and external interfaces — PASS.** Finite-route equality masks,
   visited gates, external Q indexing, and hidden oracle copies are prohibited;
   exact-route external interfaces must be declared.
5. **Stability claims — PASS.** The task permits only a checked perturbation or
   contraction condition, or a measured counterexample; it makes no unconditional
   convergence claim.
6. **Scope and governance — PASS.** The lifecycle, baseline, artifact scope,
   forbidden files, no-merge boundary, and reciprocal verification requirement
   are recorded.

## Non-blocking observations

- The dedicated `codex/FP-ITER-001` branch cannot currently be created because
  local Git metadata is not writable; the task records that it remains on the
  preparation branch until activation can be recorded safely.
- Claude execution with GPT verification differs from the short-task default,
  but the user's quota-saving authorization is recorded and takes precedence.

No task-level objection was found.

APPROVED
