# Active research workspace

## Current objective

Status: `FP-MART-001` verified by reciprocal reproduction on 2026-09-04.

The current verified extension closes a visit-indexed martingale certificate
for the two finite-sample fixed-policy routes on one frozen stationary
trajectory:

1. a uniform-in-layer Direct-Q bound from empirical contraction and the fixed residual at \(Q^\pi\);
2. a same-trajectory V-first no-split bound using a fixed-\(V^\pi\) ghost recovery target and deterministic \(\gamma\)-Lipschitz stability.

The certificate replaces unknown occupancy denominators by observed visit
counts under the selective guarantee
`P(Emit and bound violated) <= delta`. The mandatory Hoeffding route is
verified; no observable variance-adaptive constituent was established. Fully
online control, nonstationary starts, and general stochastic reward noise
remain outside scope.

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

- Task: `docs/research_tasks/FP-MART-001.md` (`VERIFIED`).
- Scientific code baseline: `b4b2769`.
- Frozen task-definition baseline: `2c5f59b`.
- GPT branch: `codex/FP-MART-001`.
- Claude role: independent construction on `claude/FP-MART-001` and independent
  verification of the GPT route, completed with `PASS` at `5999b889`.
- Current blocker: none. GPT's final verification of the repaired Claude route
  and Claude's verification of the GPT route both returned `PASS`; all 480
  records were independently reproduced.
- Next action: user review of the verified report and, only if explicitly
  approved, merge planning for `main`; otherwise open a new frozen task for a
  less conservative observable boundary.
- Design: `docs/superpowers/specs/2026-09-03-visit-indexed-martingale-certificate-design.md`.
- Plan: `docs/superpowers/plans/2026-09-03-visit-indexed-martingale-certificate-plan.md`.

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

Each independent formal route contains 480 same-seed comparisons, strict-JSON
route certificates, zero-mismatch legacy regression, certificate/failure
summaries, exact/softmax analysis, and complete execution evidence. Exact
emission is 2.5%/85%/100%/100%; primary usefulness (`total_bound < B`) appears
only at length 16384 for V-first exact (40%) and V-first softmax (3.33%).

## Historical archive

Earlier diagnostics, generated outputs, smoke runs, superseded specs/plans, and temporary build trees were moved intact to:

`C:\Users\Admin\Desktop\research\_archive\icrl_softmax_history_20260831`

The archive is recoverable and remains outside the active project directory. Python and Ruff caches were deleted because they are reproducible.
