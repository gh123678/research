# Active research workspace

## Current objective

Current construction gate: `docs/research_tasks/FP-ITER-001.md` (v1.1, `VERIFYING`). It narrows the
next step to a two-state/two-action fixed-policy Expected SARSA iteration
witness: direct reference, exact grouped attention, finite-logit attention,
and a per-iteration error/stability decomposition. The v1.0 review missed
unfrozen details. V1.1 freezes complete/missing batches, initial Q, 64 updates,
two sharpness triples and dual independent routes. The dedicated branch now
exists; v1.1 pre-review returned APPROVED. Both actors start independently from
the common activation commit, with reciprocal verification after blind seals.
Codex preliminary construction is now implemented at cb26a339, with 6,332
self-checks passing. Complete coverage at sharpness 8 contracts; missing-pair
write leakage remains. The complete fixture has zero population/data bias,
which limits that diagnostic. Claude's blind route is sealed at 4034498.
GPT's follow-up audit found unsupported M-sharp convergence claims and a
mislabeled scratch snapshot. Claude repaired both at f6feef6; GPT's replay
of the repair passes 1,665 checks plus 6,040 raw cross-checks and the scientific
review now returns PASS. All 24 aligned traces differ by at most 3.109e-15;
the repair changes none of their values. The task has resumed VERIFYING.
Claude's reciprocal review is running with read-only console evidence and
authored review records, omitting denied replay-file redirection. Existing
transfer authorization and global settings are unchanged. The task is not
VERIFIED until the reciprocal report is complete and passes.
See docs/research_branches/FP-ITER-001/codex/synthesis.md, verification_of_other.md
and handoff.md.
It does not activate the broader FP-ESARSA certificate
matrix, policy improvement, or online control.

## Prior verified diagnostic (CTRL-PREFLIGHT-001)

Prior diagnostic: `docs/research_tasks/CTRL-PREFLIGHT-001.md` (`VERIFIED`), an
economical source comparison and minimal finite-head diagnostic authorized on
2026-09-10. GPT branch: `codex/CTRL-PREFLIGHT-001`. Source comparison is recorded
in `docs/research_branches/CTRL-PREFLIGHT-001/codex/report.md`. User authorized
scoped external transfer and delegated execution: GPT defines requirements,
Claude pre-reviews and executes the diagnostic, GPT independently verifies.
Task pre-review returned APPROVED. After user-authorized continuation, Claude
repaired the missing /k and its report's analytic proof. Both reciprocal
verdicts are PASS; accepted Claude commit da6a727. The 36 head cases and
inherited checks pass, and the old certificate's no-emission obstruction is
confirmed analytically. See codex/report.md for preserved failures and final
acceptance. No current blocker. User requests conserving GPT quota: delegate
execution/report work to Claude, retain targeted independent GPT verification.
Provisional next gate: literal fixed-policy Expected SARSA construction with
explicit finite-error conditions; no overall control-route winner is claimed.
FP-ESARSA remains DRAFT; no new formal matrix or control algorithm was run.

## Verified action-gap task (FP-ADV-001)

Status: `FP-ADV-001` is `VERIFIED` as a negative usefulness result.  Claude's
task-scoped pre-review returned `APPROVED`; Codex and Claude then independently
sealed matching 480-record formal results, and both reciprocal verification
directions returned `PASS`.  All six routes emitted zero updates, so the safe
one-step theorem is retained while empirical usefulness hypothesis 8 is
falsified without retuning.

That task asked whether the verified `FP-TU-001` event can certify only
the action differences used by one policy update. V-first local exact and
finite-softmax bounds are primary; complete-Q V-first and Direct-Q bounds are
controls. The update moves half of each certified donor's mass above `pi_min`
and must guarantee `V^{pi_plus} >= V^pi` componentwise.

That task reused the frozen 480-record protocol and spends no new risk budget.
It excludes repeated control, new Direct-Q local theory, variance adaptation,
oracle inputs, outcome tuning, and conditional-on-emission claims.

`FP-TU-001` remains the verified certificate baseline. Its local merge and
remote synchronization are complete at `c579047950dfabb2600020cd2e53dd24b3e39c84`.

## Research governance

- Mode: GPT principal researcher; Claude Code auxiliary executor and independent verifier; user final arbiter.
- Canonical rules: `../AGENTS.md`.
- Active governance task: `docs/research_tasks/GOV-001.md` (`VERIFIED`).
- GPT branch: `codex/gpt-led-research-governance`.
- Claude role for GOV-001: read-only pre-review and final verification.
- Current blocker: none; GPT verification and Claude content/Git verification returned `PASS`.
- Next action: apply the verified governance rules to the next GPT-authored research task.
- Approved design: `docs/superpowers/specs/2026-08-31-gpt-led-claude-verified-research-governance-design.md`.

## Verified action-gap execution evidence

- Task: `docs/research_tasks/FP-ADV-001.md` (`VERIFIED`).
- Git and scientific baseline:
  `c579047950dfabb2600020cd2e53dd24b3e39c84`.
- Frozen DRAFT task-definition baseline:
  `919c26f8f2dc90b093cc1f40c9b36d2be2b45b03`.
- Common execution-start commit:
  `10a9a94e24ec92a59e7c756f9af6ce07b2f30e59`.
- Common route execution-start commit:
  `4078f6911cbfb4654205772685f49896e4e8cad2`.
- GPT branch: `codex/FP-ADV-001`.
- Claude role: read-only pre-review completed with `APPROVED`; isolated
  independent execution is explicitly authorized.
- GPT blind first-result seal:
  `996641d9ce2088e95c0bf2a0661e3b24b6e0d6fe`; all proof, verifier, Ruff,
  eight-record smoke, reconstruction, preservation, policy, separation, and
  dominance checks passed before the seal.
- GPT formal-evidence seal:
  `5f190ebb78697acd3cd877c1d64898929bb88024`; the sole 480-record formal
  evaluator run passed, and all six routes emitted zero updates. A bounded
  no-donor serialization defect was fixed at
  `440710c8c9c77ee8128d4abfd96919ffafe7991c` and mechanically repaired from
  saved observable inputs without rerunning the matrix.
- Claude isolated route: formal result sealed at
  `191821b26b16e13de323fb31651343ffe1eb9656` after blind seal
  `191e13ba5472ce4c183643008169236979c000ff`; GPT independently replayed all
  480 records and 17,280 route-state entries with zero failures.
- Claude executable reciprocal verification: `PASS` on 2026-09-10.  The
  verifier, strict read-only 480-record analyzer, and task-scoped Ruff all
  passed; no formal evaluator rerun or repository mutation occurred.  Claude
  recorded its own report on `claude/FP-ADV-001` at commit `42fb0cc`.
- Current blocker: none.
- Shared conclusion:
  `docs/research_branches/action_gap_certificate_report.md`.
- Next task: `docs/research_tasks/FP-ESARSA-001.md` remains `DRAFT`.  The next
  permitted action is task review; implementation or experiments must not
  begin before the governance lifecycle activates it.
- Design:
  `docs/superpowers/specs/2026-09-08-action-gap-safe-update-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-08-action-gap-safe-update-plan.md`.
- Prior verified task: `docs/research_tasks/FP-TU-001.md` (`VERIFIED`).

## Active implementation

- `action_gap_certificate.py`
- `verify_action_gap_certificate.py`
- `evaluate_action_gap_certificates.py`
- `analyze_action_gap_certificates.py`
- `evaluate_fixed_policy_q_routes.py`
- `fixed_policy_finite_sample_certificate.py`
- `verify_finite_sample_theorems.py`
- `analyze_fixed_policy_finite_sample_certificates.py`
- `verify_fixed_policy_q_routes.py`
- `crossfit_vfirst.py`
- `markov_coverage_certificate.py`
- `verify_crossfit_markov_certificate.py`
- `evaluate_blockwise_q_routes.py`
- `mdps.py`
- `visit_indexed_martingale_certificate.py`
- `evaluate_visit_indexed_certificates.py`
- `analyze_visit_indexed_certificates.py`
- `verify_visit_indexed_martingale_certificate.py`
- `time_uniform_mixture_certificate.py`
- `verify_time_uniform_mixture_certificate.py`
- `evaluate_time_uniform_certificates.py`
- `analyze_time_uniform_certificates.py`

## Active evidence

- `docs/research_branches/`
- `docs/superpowers/specs/2026-08-29-direct-q-v-first-balanced-exploration-design.md`
- `docs/superpowers/specs/2026-08-29-crossfit-markov-certificate-design.md`
- `docs/superpowers/specs/2026-08-31-shared-fixed-policy-finite-sample-theorem-design.md`
- `docs/superpowers/plans/2026-08-31-shared-fixed-policy-finite-sample-theorem-plan.md`
- `docs/research_branches/shared_fixed_policy_finite_sample_theory.md`
- `docs/superpowers/plans/2026-08-29-direct-q-v-first-balanced-exploration-plan.md`
- `docs/superpowers/plans/2026-08-29-crossfit-markov-certificate-plan.md`
- `results/fixed_policy_q_routes/`
- `results/fixed_policy_q_routes_crossfit/`
- `results/fixed_policy_finite_sample_certificates/`
- `results/blockwise_q_routes/`
- `docs/research_branches/visit_indexed_martingale_certificate_theory.md`
- `docs/research_branches/visit_indexed_martingale_certificate_report.md`
- `results/FP-MART-001/codex/`
- `results/FP-MART-001/claude/`
- `docs/superpowers/specs/2026-09-04-time-uniform-mixture-certificate-design.md`
- `docs/superpowers/plans/2026-09-04-time-uniform-mixture-certificate-plan.md`
- `docs/research_tasks/FP-TU-001.md`
- `docs/research_branches/time_uniform_mixture_certificate_theory.md`
- `docs/research_branches/time_uniform_mixture_certificate_report.md`
- `results/FP-TU-001/codex/`
- `results/FP-TU-001/claude/`
- `docs/research_tasks/FP-ADV-001.md`
- `docs/research_branches/action_gap_certificate_theory.md`
- `docs/research_branches/action_gap_certificate_report.md`
- `results/FP-ADV-001/codex/`
- `results/FP-ADV-001/claude/`

Each independent formal route contains 480 same-seed comparisons, strict-JSON
route certificates, zero-mismatch legacy regression, certificate/failure
summaries, exact/softmax analysis, and complete execution evidence. Exact
emission is 2.5%/85%/100%/100%; primary usefulness (`total_bound < B`) appears
only at length 16384 for V-first exact (40%) and V-first softmax (3.33%).

## Historical archive

Earlier diagnostics, generated outputs, smoke runs, superseded specs/plans, and temporary build trees were moved intact to:

`C:\Users\Admin\Desktop\research\_archive\icrl_softmax_history_20260831`

The archive is recoverable and remains outside the active project directory. Python and Ruff caches were deleted because they are reproducible.
