# Active research workspace

## Current objective

Status: completed and formally reproduced on 2026-08-31.

Close two finite-sample fixed-policy results on one frozen stationary trajectory:

1. a uniform-in-layer Direct-Q bound from empirical contraction and the fixed residual at \(Q^\pi\);
2. a same-trajectory V-first no-split bound using a fixed-\(V^\pi\) ghost recovery target and deterministic \(\gamma\)-Lipschitz stability.

The two routes share state, pair, and edge-chain count/residual events. The required deliverables are explicit theorems, computable certificates, contract tests, and an updated comparison report. Fully online control, nonstationary starts, and general stochastic reward noise are outside the current scope.

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

- Task: `docs/research_tasks/FP-MART-001.md` (`ACTIVE`).
- Scientific code baseline: `b4b2769`.
- Frozen task-definition baseline: `2c5f59b`.
- GPT branch: `codex/FP-MART-001`.
- Claude role at this gate: independent construction on
  `claude/FP-MART-001`, followed by independent verification of the GPT route.
- Current blocker: Claude's read-only pre-review returned `APPROVED`, and the
  common activation commit plus isolated worktree exist, but execution requires
  the user's explicit authorization to send task-scoped private-repository
  materials to the external Claude service and permit public-web source lookup.
  No implementation or experiment has started.
- Next action: obtain and record that user ruling; if approved, start both
  routes from activation commit `c8ec7e5c` without disclosing either first
  result.
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

The new formal directory contains 480 same-seed comparisons, strict-JSON
route certificates, a zero-mismatch regression against the old 10 routes,
certificate/failure summaries, and exact-versus-softmax paired analysis.

## Historical archive

Earlier diagnostics, generated outputs, smoke runs, superseded specs/plans, and temporary build trees were moved intact to:

`C:\Users\Admin\Desktop\research\_archive\icrl_softmax_history_20260831`

The archive is recoverable and remains outside the active project directory. Python and Ruff caches were deleted because they are reproducible.
