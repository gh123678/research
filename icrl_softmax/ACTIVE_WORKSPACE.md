# Active research workspace

## Current objective

Status: `FP-ADV-001` is `ACTIVE`. Claude's task-scoped read-only pre-review of
commit `a3a0340b3f63ed53eb00b5d5af244fabdbbbe75d` returned `APPROVED` on
2026-09-08 with 12 passed contract checks and two non-blocking implementation
cautions. The activation commit is the common execution start and will be
recorded exactly in the next evidence-only metadata commit.

The current task asks whether the verified `FP-TU-001` event can certify only
the action differences used by one policy update. V-first local exact and
finite-softmax bounds are primary; complete-Q V-first and Direct-Q bounds are
controls. The update moves half of each certified donor's mass above `pi_min`
and must guarantee `V^{pi_plus} >= V^pi` componentwise.

The task reuses the frozen 480-record protocol and spends no new risk budget.
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

## Active research task

- Task: `docs/research_tasks/FP-ADV-001.md` (`ACTIVE`).
- Git and scientific baseline:
  `c579047950dfabb2600020cd2e53dd24b3e39c84`.
- Frozen DRAFT task-definition baseline:
  `919c26f8f2dc90b093cc1f40c9b36d2be2b45b03`.
- Common execution-start commit: the current `ACTIVE` transition commit; exact
  identity will be recorded immediately afterward.
- GPT branch: `codex/FP-ADV-001`.
- Claude role: read-only pre-review completed with `APPROVED`; independent
  execution is not authorized.
- Current blocker: task validity is approved, but the private-material transfer
  required for Claude's independent proof, implementation, smoke, and formal
  route has not been authorized.
- Next action: record the exact activation commit, then obtain the separate
  independent-execution authorization before launching either blind route.
- Design:
  `docs/superpowers/specs/2026-09-08-action-gap-safe-update-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-08-action-gap-safe-update-plan.md`.
- Prior verified task: `docs/research_tasks/FP-TU-001.md` (`VERIFIED`).

## Active implementation

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

Each independent formal route contains 480 same-seed comparisons, strict-JSON
route certificates, zero-mismatch legacy regression, certificate/failure
summaries, exact/softmax analysis, and complete execution evidence. Exact
emission is 2.5%/85%/100%/100%; primary usefulness (`total_bound < B`) appears
only at length 16384 for V-first exact (40%) and V-first softmax (3.33%).

## Historical archive

Earlier diagnostics, generated outputs, smoke runs, superseded specs/plans, and temporary build trees were moved intact to:

`C:\Users\Admin\Desktop\research\_archive\icrl_softmax_history_20260831`

The archive is recoverable and remains outside the active project directory. Python and Ruff caches were deleted because they are reproducible.
