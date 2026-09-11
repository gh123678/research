# FP-EXPL-001 (v1.1) — Claude route failure and repair history

This file records all known failures and repairs on the Claude main route.
Nothing here is erased; older reports remain as historical records with a
dated correction note at the top.

## Current status (2026-09-11, probe boundary repair)

- GPT source review of the sealed Claude route found one ordinary boundary
  error: `witness.py` and `verify.py` both included `q_pi` (audit-only per
  frozen task section 4, never a network input) in the one-step **network**
  probe lists, together with `q_hat` and an arbitrary non-basis vector
  `[1, -1, 0.5, 0.5]`.
- Repair (this commit): probe lists reduced to the four standard basis
  vectors plus `Q0 = 0` in both scripts; audit-only `q_pi`/`q_hat`
  mathematics (population audit, Bellman residual, data-bias bound,
  three-term decomposition) untouched; `witness.py` code metadata now
  documents that `baseline_commit_at_start` denotes the run checkout at
  witness start (after a seal: the sealed run checkout), with the frozen
  scientific baseline `c710e32b...` and common ACTIVE publication commit
  `8c915c4b...` recorded explicitly under their own names.
- Effect: verifier one-step unit checks 8 -> 5 per equivalence group
  (groups C and D: 9 -> 6; total 1285 -> 1279, 0 failures, PASS). Ruff
  PASS. All scientific outputs (sampling, all four 65-row research traces,
  operators, fixed points, decompositions, bounds, literal network weights
  and first-step projections) are bit-identical to both the pre-repair
  sealed run and the original pre-seal preserved results; evidence:
  `results/FP-EXPL-001/claude/rerun_probe_boundary_repair_20260911T154700/scientific_invariance_evidence.json`.
- Preserved evidence: original pre-seal artifacts under
  `results/FP-EXPL-001/claude/preserved_pre_repair_20260911T154017/`;
  the successful but superseded rerun artifacts under
  `results/FP-EXPL-001/claude/archived_pre_probe_boundary_repair_20260911T154700/`;
  command logs under `rerun_20260911T154017/` and
  `rerun_probe_boundary_repair_20260911T154700/`.

## Earlier failures on this route (2026-09-11, pre-seal)

1. **String-to-list immutable snapshot TypeError** — an internal snapshot
   representation bug in the witness; fixed before any seal.
2. **Bool-as-error sampling checker** — a checker returned a boolean where a
   numeric error was expected, reporting 64 failures while exiting 0. Fixed;
   the failed `verification.json` of that run was subsequently overwritten
   by later runs, so only the repair logs remain as evidence of it.
3. **Data-bias proof and verifier exit-code defects** — `theory.md`'s
   data-bias proof and `verify.py`'s `[0,1)` draw check and nonzero FAIL
   exit were repaired in the same pre-seal session; verify/witness/ruff
   then passed (logs: `rerun_20260911T154017/`).
4. **Operational failures (no research-content effect)** — Bash EPERM/auth
   failures, wrong-working-directory exit 2, heredoc quoting exit 2, and a
   Python backslash-escape SyntaxError during document saves; the prior
   process was stopped during repeated heredoc save failures. These changed
   no research results.

## Provenance notes

- No author commit existed before this repair commit; earlier "sealed"
  references denote the successful but unsealed rerun now archived above.
- GPT's earlier execution of its acceptance script occurred before any
  author seal and is preflight only, not formal independent acceptance.
- The raw resume log `results/FP-EXPL-001/codex/claude_resume_author.raw.jsonl`
  is retained unmodified as raw evidence.
- Claude's reciprocal review of GPT's acceptance
  (`verification_of_other.md`) remains superseded/pending: GPT will replay
  the repaired route and send a corrected acceptance for Claude's executable
  reciprocal review.
