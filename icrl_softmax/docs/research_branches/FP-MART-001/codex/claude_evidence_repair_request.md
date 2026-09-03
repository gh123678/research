# FP-MART-001 Claude evidence-repair request

Date: 2026-09-03. Author: GPT. This request follows the GPT verification
`verification_of_claude_repair.md`, which found the repaired proof,
implementation, and 480-record result correct but returned `FAIL` for
incomplete route evidence.

This is an ordinary author-side evidence repair. Do not change the frozen task,
scientific implementation, formal configuration, or formal raw results, and do
not rerun or overwrite the canonical formal matrix.

## Required repair

1. From the current Claude branch, run Ruff and the exact five required
   verifiers: `verify_finite_sample_theorems.py`,
   `verify_fixed_policy_q_routes.py`, `verify_crossfit_markov_certificate.py`,
   `verify_end_to_end_sarsa.py`, and
   `verify_visit_indexed_martingale_certificate.py`. Preserve their complete
   output in the canonical repaired-smoke `checks.log`. `verify_theory.py` may
   be recorded as an additional check but cannot substitute for a required
   verifier.
2. Amend `repair_smoke.md` to state truthfully that
   `verify_end_to_end_sarsa.py` was omitted before the formal run and was run
   only during the later evidence repair. Do not rewrite this as a pre-formal
   event. Record the exact repair command, outcome, and hashes for all seven
   repaired-smoke artifacts: config, task results, summary, regression,
   environment, commands, and checks.
3. Add a current, numbered assessment of all 18 frozen task acceptance
   criteria to `repair_result.md` (or a clearly linked new Claude evidence
   document). Distinguish passed criteria from cross-verification and synthesis
   items that are still pending at the time of writing. Preserve the initial
   and first repair history.
4. Commit only the evidence correction with a subject beginning `[claude]`,
   leave the branch clean, and return the exact commit. No new formal run is
   required because the implementation and deterministic formal outputs have
   already reproduced exactly.

If this cannot be done without changing the task or scientific implementation,
return `OBJECTION` instead of modifying scope.
