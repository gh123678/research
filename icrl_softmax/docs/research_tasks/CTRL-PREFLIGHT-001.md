# CTRL-PREFLIGHT-001: economical control-route preflight

- Date: 2026-09-10. Author: GPT. Version: 1.2. Status: `VERIFIED`.
- Baseline: `db9d63043489f4ec5660e71c53e7e842a280d04f`.
- User authorization: after the comparison of directions, the user said
  “好的你去做”, authorizing dependency comparison and minimal validation.
- Classification: short diagnostic review, estimated 15--25 minutes, CPU only.
  No new control algorithm, convergence theorem, formal MDP matrix, training,
  or publication claim is implemented. GPT defines and reviews; Claude executes
  the narrow reproducible diagnostics; GPT independently verifies them. This is not the multi-stage
  FP-ESARSA task and does not activate or change its frozen definitions.
- GPT branch: `codex/CTRL-PREFLIGHT-001`.
- Claude role: read-only pre-review, then scoped diagnostic implementation and
  executable verification of GPT's source comparison on `claude/CTRL-PREFLIGHT-001`.
  GPT records returned review provenance on its own branch, does not author
  Claude's artifacts, and independently reruns/inspects Claude's diagnostics.
- Resource authorization v1.2: user lifted the prior two-call and USD 1 limits
  for Claude on 2026-09-10 (ruling below). Necessary continuation and verification
  calls may use the existing configured account without another quota question.
  Remain economical; no new provider/model configuration or fee category.
  Report actual backend metadata rather than assuming a model from the CLI name.

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
construction documentation and manuscript method sections, and the GPT
source comparison in docs/research_branches/CTRL-PREFLIGHT-001/codex/report.md.

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
- docs/research_branches/CTRL-PREFLIGHT-001/codex/report.md;
- docs/research_branches/CTRL-PREFLIGHT-001/codex/claude_review.md;
- ignored results/CTRL-PREFLIGHT-001/codex/ (stdout/provenance);
- a compact task/status pointer in ACTIVE_WORKSPACE.md.

Claude may add only, within its isolated worktree and branch:
- docs/research_branches/CTRL-PREFLIGHT-001/claude/diagnostic.py;
- docs/research_branches/CTRL-PREFLIGHT-001/claude/report.md (including its
  executable verification of GPT's source comparison, ending PASS/FAIL/OBJECTION);
- ignored results/CTRL-PREFLIGHT-001/claude/ (stdout/provenance).

All paths above are relative to icrl_softmax/. The shared execution-start
commit is the GPT activation commit, containing unchanged scientific source
from the baseline plus this reviewed task and GPT's source comparison.
Claude must confirm branch/worktree and report its execution commit. It may
commit only its two specified documentation-directory artifacts, with a
[claude] subject. It must not change task status or any other tracked file.

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
and one concrete next validation gate; Claude executable source-review PASS
and GPT independent implementation/evidence-verification PASS.
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
  No process/session was started. This historical rejection was not a scientific
  OBJECTION. User subsequently authorized the scoped delegation below.
- User ruling on 2026-09-10: in response to the explicit scoped external-transfer
  request (governance, task, related source, diagnostic reports; two calls,
  total USD 1), user said: "你去思考然后提出要求，然后给claude去做就行".
  This authorizes GPT-defined requirements and Claude execution for this task,
  including necessary transmission to the existing configured Claude Code
  service. It is a task-specific exception to the default short-task executor
  assignment, not a waiver of pre-review, independent verification, or limits.
  No credentials, unrelated user files, archived results, or bulk repository
  upload are in scope. No new provider/model or fee category is authorized.
  Version 1.1 changes assignment/authorization only; numerical protocol and
  scientific acceptance tests are unchanged. Design self-review: no protocol
  expansion, no ambiguous writer support, finite/grouped errors kept separate.
- Lifecycle: DRAFT sealed at ebe6bd3; entered REVIEW on 2026-09-10.
- v1.1 pre-review: APPROVED on 2026-09-10, session
  8841d30e-fe26-4ef6-9a55-1a50b809b1f5; entered ACTIVE after receipt.
  No task-level objection. CLI-reported cost USD 0.141851; metadata labels
  kimi-k2.6 and k3 (not an independently authenticated model identity).
  Execution start is the commit recording this activation.
- Final verification: pending.
- Execution update: Claude call 2 stopped at error_max_budget_usd, session
  62b4189c-f1ce-4f93-9dfe-ce75a6c4f96b. Reported USD 0.533722 despite the
  configured USD 0.50 stop threshold; total reported USD 0.675573. Both calls
  are consumed. No new call without user ruling on this limit.
- Lifecycle continuation: ACTIVE -> VERIFYING for GPT inspection/rerun of
  Claude's partial artifacts -> ACTIVE after FAIL. This is an implementation
  defect, not a task-level OBJECTION: diagnostic.py line 150 omits /k in the
  numerical mixture contribution. No Claude final report or result commit.
- Historical blocker: CALL_COUNT_LIMIT after budget-interrupted execution.
  Exact failure, correct values, unchanged scope and author-repair handoff
  are recorded in codex/report.md. The 36 head rows and two inherited checks
  passed GPT independent checks, but the overall task is not VERIFIED.
- User ruling v1.2 on 2026-09-10: "好的你随便调用它他的额度无所谓".
  The prior call-count and monetary limits are lifted for necessary Claude
  continuation, repair and verification. The scientific protocol, authorized
  inputs/outputs, roles, acceptance criteria and no-push/no-merge boundaries
  are unchanged. This resource-only amendment resumes the already pre-reviewed
  task; it is not a new scientific task or permission for broader experiments.
  Claude must read this current task and GPT's handoff at their absolute paths
  in the GPT checkout (its frozen worktree retains v1.1). It must not update
  the task or GPT branch. Current blocker: none; author repair is pending.
- User clarification: "我的意思是你的额度得节省". Prefer Claude execution,
  debugging and report preparation; GPT retains research decisions and targeted
  independent verification, reusing unchanged evidence rather than repeating
  the entire audit. Verification requirements are not waived.
- Continuation at Claude commit 651fc6e: code repair and 36-row rerun PASS
  in GPT verification; four corrected contributions agree within 1.5e-14.
  ACTIVE -> VERIFYING -> ACTIVE for a report-only FAIL: sections 3.4-3.5
  incorrectly leave the analytic mixture inequality unproved while asserting
  the consequent unconditional no-emission result. GPT supplied the elementary
  square-completion proof for Claude to check independently. Narrow corrections
  also cover interface-existence wording, numeric tolerance and references.
  No new numerical protocol, algorithm, or task-level objection; awaiting
  author report repair, then targeted GPT diff verification.
- Report repair submitted at da6a72758537446bcd872f6dd7dc68fedd13a2a1;
  ACTIVE -> VERIFYING. Only Claude report.md changed relative to 651fc6e.
  GPT checked the complete diff, including the universal square-completion
  proof, numerical/analytic distinction, scope corrections and provenance.
  Final preservation audit and status recording follow; no new experiments.
- Final: VERIFYING -> VERIFIED on 2026-09-10 after both reciprocal verdicts
  PASS. Claude accepted report commit da6a72758537446bcd872f6dd7dc68fedd13a2a1;
  GPT final verification and acceptance checklist are in codex/report.md.
  Exact /k repair independently rerun, report proof independently inspected,
  allowed-file/source preservation checked, user learning record untouched.
  No unresolved objection; earlier failures remain recorded, not erased.
  Current blocker: none. Recommendation remains provisional; no new control
  algorithm, FP-ESARSA activation, main merge or push was performed.
- Read-only comparison: docs/research_branches/CTRL-PREFLIGHT-001/codex/report.md.
- Review provenance: docs/research_branches/CTRL-PREFLIGHT-001/codex/claude_review.md.
- No merge to main is authorized or required for diagnostic completion.
