# FP-MART-001 request: Claude independently verifies GPT

Date: 2026-09-04. Author: GPT. Task status: `VERIFYING`.

GPT has passed the final repaired Claude route at Claude commit
`a559bd31769506565dd4503d8239d4a3f28ddf81`. Claude is now authorized to
perform the reciprocal verification required by frozen acceptance criterion
17.

## Frozen GPT route

- Implementation commit:
  `4cf6f50d69c5aa4937d181f758f546f9d2213c41`.
- Blind first-result seal:
  `e7c04111b7aec8c9dc5043883fcaa68cc158837b`.
- Formal-evidence seal:
  `63b84fd4295598c3e020d7e489552470ac1763c4`.
- GPT route documents:
  `C:\Users\Admin\Desktop\research\icrl_softmax\docs\research_branches\FP-MART-001\codex\theory.md`,
  `first_result.md`, and `formal_result.md`.
- GPT formal result directory:
  `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex`.

Later GPT commits contain verification governance records only; verify that no
scientific Python file changed after the implementation/formal seals rather
than assuming it.

## Required independent work

1. Work only on `claude/FP-MART-001`. Treat the GPT worktree and GPT result
   directory as read-only. Do not amend or repair GPT artifacts.
2. Inspect the GPT proof, implementation, verifier, evaluator, analyzer,
   first-result record, formal-result record, Git history, and formal raw
   artifacts against all frozen task requirements. Independently check the
   filtrations, visit-indexed MDS construction, conditional range constant,
   compensated MGF/tail derivation, simultaneous random-count substitution,
   selective semantics, recurrence composition, allowed inputs, strict count
   validation, namespace/schema separation, legacy preservation, and both
   usefulness thresholds.
3. Run Ruff and the exact five required verifiers against the GPT files:
   `verify_finite_sample_theorems.py`, `verify_fixed_policy_q_routes.py`,
   `verify_crossfit_markov_certificate.py`, `verify_end_to_end_sarsa.py`, and
   `verify_visit_indexed_martingale_certificate.py`.
4. Reproduce the frozen 480-record GPT matrix into the Claude-owned isolated
   directory
   `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\claude\verification_codex`.
   Use the exact frozen arguments and seed from the task. Run the strict
   analyzer against the frozen legacy baseline. Do not overwrite any GPT or
   existing Claude formal/smoke result.
5. Compare the deterministic core hashes (`config.json`, `task_results.json`,
   `summary.json`, and `regression.json`) with the GPT formal result and explain
   expected environment/command-log differences. Check all 480 records,
   emissions, both usefulness thresholds, oracle and per-group violations,
   exact-route 4096/16384 emission, and legacy tolerances.
6. Write
   `docs/research_branches/FP-MART-001/claude/verification_of_codex.md` on the
   Claude branch. Record exact commits, commands, environment, output paths,
   hashes, anomalies, limitations, proof/code findings, and a numbered mapping
   of frozen criteria 1--18. End the report with exactly one of `PASS`, `FAIL`,
   or `OBJECTION`.
7. Commit only Claude-owned verification evidence with a subject beginning
   `[claude]`, leave the branch clean, and return the exact commit and outcome.

If a task-definition defect is found, return `OBJECTION`. If a GPT proof,
implementation, result, or evidence defect is found while the task remains
valid, return `FAIL`. Do not silently fix the GPT route.
