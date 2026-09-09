# Active research workspace

## Current objective

Status: `FP-KERN-001` is `ACTIVE` for a user-authorized corrective GPT rerun.
Both independent 480-record routes,
their blind seals, strict reconstructions, and reciprocal reports are
complete. GPT and Claude independently classify the frozen kernel route as
`NOT_SUPPORTED`: both families fail the zero-count RMSE, `1-4`-count RMSE,
and sparse-state top-action requirements while passing coverage and
false-improvement control.

GPT verification of Claude is `PASS` at `5e538ad06e1167c8c644db9db7f7f25573b470cb`;
Claude verification of GPT is `PASS` at
`f9666e9d54091e8d563f7c3f097d294ef3ee9ddd`. A principal audit nevertheless
found three literal frozen-definition deviations in the sealed GPT route
(self-weight gating, one-sided false-improvement definition, and
record/action Spearman granularity). They do not change the common scientific
classification. On 2026-09-09 the user explicitly authorized one documented
corrective GPT formal rerun, restricted to those three definitions with no
scientific tuning. The original formal artifacts remain preserved; Claude's
valid independent formal route will not be rerun.

The task asks whether a state kernel learned only from the observed behavior
of other actions can improve fixed-policy Q estimation and action ordering for
zero-count and low-count target pairs. The unchanged current random MDP family
tests project applicability; a hidden-cluster family with structure concealed
from the estimator is the positive control.

The frozen study has four routes and 480 records. It is empirical feasibility
work only: true Q, hidden clusters, and exact returns are oracle-audit outputs,
and no safety or policy-nondegradation theorem is claimed.

`FP-TU-001` remains the verified scientific baseline at
`c579047950dfabb2600020cd2e53dd24b3e39c84`. The separate unmerged
`FP-ADV-001` negative result motivates this task but is not an execution
baseline or formal input.

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

- Task: `docs/research_tasks/FP-KERN-001.md` (`ACTIVE`, corrective repair).
- Scientific baseline:
  `c579047950dfabb2600020cd2e53dd24b3e39c84`.
- Frozen DRAFT task-definition baseline:
  `b7ef163f11eb5ee41499344c296587efd3516651`.
- Activation commit:
  `2308c372eb47ce7c181f98caee952121f4e47644`.
- Common route execution-start commit:
  `28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c`.
- GPT branch and isolated worktree: `codex/FP-KERN-001` in
  `results/FP-KERN-001/codex_worktree/`.
- Claude route: formal seal `1a820467683d137e6527edbd99bb486000b2fc58`;
  classification `NOT_SUPPORTED`; GPT verification `PASS`.
- GPT route: formal seal `640a3f8fb2d0f41b96eef8d9bb76fc5ef2e9b93b`;
  classification `NOT_SUPPORTED`; Claude verification `PASS`.
- Current blocker: none; the user authorized the exact corrective rerun.
- Next action: add regression tests for the three frozen definitions, repair
  only those paths, pass smoke, seal the correction, run the one authorized
  480-record rerun, and obtain Claude verification of the corrected route.
- Design:
  `docs/superpowers/specs/2026-09-09-kernel-state-generalization-feasibility-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-09-kernel-state-generalization-feasibility-plan.md`.
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
- FP-KERN-001 implementation: `kernel_state_generalization.py`,
  `kernel_generalization_mdps.py`, `verify_kernel_state_generalization.py`,
  `evaluate_kernel_state_generalization.py`, and
  `analyze_kernel_state_generalization.py`.

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
- `docs/superpowers/specs/2026-09-09-kernel-state-generalization-feasibility-design.md`
- `docs/superpowers/plans/2026-09-09-kernel-state-generalization-feasibility-plan.md`
- `docs/research_tasks/FP-KERN-001.md`
- `docs/research_branches/FP-KERN-001/codex/formal_result.md`
- `docs/research_branches/FP-KERN-001/codex/verify_claude.md`
- `docs/research_branches/FP-KERN-001/claude/formal_result.md`
- `docs/research_branches/FP-KERN-001/claude/verify_codex.md`
- `results/FP-KERN-001/codex/`
- `results/FP-KERN-001/claude/`

Each independent formal route contains 480 same-seed comparisons, strict-JSON
route certificates, zero-mismatch legacy regression, certificate/failure
summaries, exact/softmax analysis, and complete execution evidence. Exact
emission is 2.5%/85%/100%/100%; primary usefulness (`total_bound < B`) appears
only at length 16384 for V-first exact (40%) and V-first softmax (3.33%).

## Historical archive

Earlier diagnostics, generated outputs, smoke runs, superseded specs/plans, and temporary build trees were moved intact to:

`C:\Users\Admin\Desktop\research\_archive\icrl_softmax_history_20260831`

The archive is recoverable and remains outside the active project directory. Python and Ruff caches were deleted because they are reproducible.
