# CTRL-PREFLIGHT-001: economical control-route preflight

- Date: 2026-09-10. Author: GPT. Version: 1.0. Status: `REVIEW`.
- Baseline: `db9d63043489f4ec5660e71c53e7e842a280d04f`.
- User authorization: after the comparison of directions, the user said
  “好的你去做”, authorizing dependency comparison and minimal validation.
- Classification: short diagnostic review, estimated 15--25 minutes, CPU only.
  No new control algorithm, convergence theorem, formal MDP matrix, training,
  or publication claim is implemented. GPT executes; Claude independently
  verifies the narrow reproducible diagnostics. This is not the multi-stage
  FP-ESARSA task and does not activate or change its frozen definitions.
- GPT branch: `codex/CTRL-PREFLIGHT-001`.
- Claude role: read-only pre-review and executable final review, no writes.
  GPT records the returned reports with provenance on its branch.
- Resource ceiling: two Claude Code calls, each at most USD 0.50 under the
  existing configured account; no new model/provider configuration. Report
  actual backend metadata rather than assuming a model from the CLI name.

## Questions and falsifiable checks

1. Which existing control candidate has fewer missing computation interfaces:
   TwoStageSoftmaxQControl, or fixed-policy Expected SARSA plus relative
   softmax? Compare actual source and draft, not method names.
2. Do existing literal sampled-SARSA and compact two-stage verifiers still pass?
   A failure is reported without changing the inherited code.
3. Does the finite successor head recover its *declared finite-score* average
   on two-state/two-action fixtures, and how does it differ from a grouped
   average? Test both candidate scoring rules, not only the favored route.
4. Reproduce the previously disclosed FP-ESARSA numerical no-emission preflight
   at its four frozen lengths. This is retrospective arithmetic checking,
   not a newly preregistered hypothesis or a new theorem.

## Inputs, protocol, and minimal diagnostic

Read-only inputs: AGENTS.md, ACTIVE_WORKSPACE.md, FP-ESARSA-001 task/design,
model.py, verify_end_to_end_sarsa.py, verify_two_stage_q_control.py,
time_uniform_mixture_certificate.py and imported dependencies, existing
construction documentation and manuscript method sections.

Run the two existing verifiers with Python -B and one new standalone diagnostic.
Use float64 and no random trajectory. The new diagnostic uses exactly these
Q matrices: [[0,0],[0,0]], [[1,-1],[-1,1]], [[-1,-1],[1,1]].
The fixed policy is [[0.75,0.25],[0.25,0.75]]. For each successor state and
zeta in [0,8,32], compare standard softmax weights on four unique memory
candidates to an independently written scalar-exp calculation:

- Two-stage scores: zeta*1{u=s'} + 8*Q(u,b).
- Fixed-policy scores: zeta*1{u=s'} + log pi(b|u).

Explicitly use query/key matrix products for the first implementation; the
reference uses loops and scalar arithmetic. Record normalization, requested-
state mass, finite/reference discrepancy, and finite/grouped discrepancy.
Require finite/reference errors <=1e-12; finite/grouped error need not vanish.
The uniform and separated-level cases are intentionally disclosed adversarial
fixtures, not representative empirical performance tests.

Also record current-read mass exp(8)/(exp(8)+3), and the uniform write mass
for a query absent from a three-transition context [0,0,3]. Repeated visits
remain separate residual observations, not duplicate Q-memory candidates.
These are interface diagnostics, not an end-to-end model or control run.

Reproduce the prior no-emission arithmetic for n=[256,1024,4096,16384],
d=24, delta=.05, gamma=.7, B=5. Record k=floor(n/(2d)), the analytic floor
2B/(1-gamma)*sqrt(2 log(d/delta)/k), and the existing numerical mixture
radius contribution at k. Do not claim this prohibits all other certificates.

## Outputs and allowed changes

GPT may add only:
- docs/research_tasks/CTRL-PREFLIGHT-001.md (this task);
- docs/research_branches/CTRL-PREFLIGHT-001/codex/diagnostic.py;
- docs/research_branches/CTRL-PREFLIGHT-001/codex/report.md;
- docs/research_branches/CTRL-PREFLIGHT-001/codex/claude_review.md;
- ignored results/CTRL-PREFLIGHT-001/codex/ (stdout/provenance);
- a compact task/status pointer in ACTIVE_WORKSPACE.md.

No existing algorithm, experiment, result, task contract, manuscript, or user
learning record may be modified. No push, merge, provider changes, or broad
experiments. New diagnostics print JSON to stdout and do not write files.
No implementation before pre-review APPROVED and status ACTIVE.

## Decision and acceptance

The report must list external operations (content lookup, action grouping,
fixed-policy input, missing-pair handling, policy output, sampling), reusable
components, missing literal witnesses, and remaining guarantee assumptions.
Recommend the next candidate by smallest missing interface scope *subject to*
the measured finite-head obstacle; do not treat code maturity as novelty.
An inconclusive/tied comparison is admissible. No empirical superiority claim.

Acceptance: accurate source references; all diagnostic rows recorded including
adverse results; finite-score reference errors <=1e-12; inherited checks
reported truthfully; no-oracle/source-preservation audit; explicit limitations
and one concrete next validation gate; Claude executable final report PASS.
The technical recommendation remains provisional, even when this diagnostic
task is VERIFIED. Independent verification validates evidence, not publication
value or an unbuilt architecture.

Stop affected work on task-level OBJECTION, overlapping changes, unavailable
Claude, or a required expansion beyond the frozen scope. A diagnostic failure
is a permissible finding; a reference/implementation error is repaired within
scope and both failed and successful checks are recorded.

## Review, lifecycle, and evidence

- Pre-review: NOT_EXECUTED; automatic approval review rejected transfer of
  private AGENTS.md and this new task to the configured external service.
  No process/session was started. Await explicit scoped user authorization.
  This is not a scientific OBJECTION; status remains REVIEW.
- Lifecycle: DRAFT sealed at ebe6bd3; entered REVIEW on 2026-09-10.
- Final verification: pending.
- Read-only comparison: docs/research_branches/CTRL-PREFLIGHT-001/codex/report.md.
- Review provenance: docs/research_branches/CTRL-PREFLIGHT-001/codex/claude_review.md.
- No merge to main is authorized or required for diagnostic completion.
