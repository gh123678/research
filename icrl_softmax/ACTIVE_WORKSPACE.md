# Active research workspace

## Current objective

Status: `FP-TU-001` is `VERIFIED`.  Both independent routes and reciprocal
reports pass; the user accepted the two documented environment/procedure
exceptions on 2026-09-07.  `main` remains unchanged.

The current task asks whether a preregistered finite geometric mixture of
exponential supermartingales can replace the verified count-wise Hoeffding
union while preserving the same fixed-policy assumptions, no-oracle inputs,
selective probability statement, and deterministic route recurrences.

The mandatory target is a time-uniform mixture radius no wider than the old
radius for every count through 16384. Analytic line stitching is audit-only;
observable transition variance is feasibility-only. The frozen 480-record
protocol is retained for paired evaluation after activation.

`FP-MART-001` remains the verified scientific baseline. Fully online control,
nonstationary starts, general stochastic reward noise, outcome-tuned mixture
constants, and post-hoc certificate selection remain outside scope.

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

- Task: `docs/research_tasks/FP-TU-001.md` (`VERIFIED`).
- Scientific baseline: `28b71685ca05ae073cc847fdea230019e4bd63ea`.
- Frozen task-definition baseline:
  `0a66c4c583678e3186af7c5dbcff02779436329c`.
- Common execution-start commit:
  `0ce18b4676f70ca0556496e804aa65563efa63da`.
- GPT branch: `codex/FP-TU-001`.
- Claude role: independent construction and reciprocal verification completed;
  final report commit
  `ae82bb8bd7a269c878697b19280a00c4ad2c2f23` ends `PASS`.
- Current blocker: none.  GPT reproduced Claude's repaired 480-record route;
  Claude independently reproduced GPT's verifier, hash, analyzer, and metric
  evidence; both reciprocal reports pass.
- Next action: await separate user approval before any merge to `main`, or
  begin the next GPT-authored research task on a new `codex/*` branch.
- Design:
  `docs/superpowers/specs/2026-09-04-time-uniform-mixture-certificate-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-04-time-uniform-mixture-certificate-plan.md`.
- Prior verified task: `docs/research_tasks/FP-MART-001.md` (`VERIFIED`).

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
