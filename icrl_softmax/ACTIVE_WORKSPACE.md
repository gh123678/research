# Active research workspace

## Current objective

Close two finite-sample fixed-policy results on one frozen stationary trajectory:

1. a uniform-in-layer Direct-Q bound from empirical contraction and the fixed residual at \(Q^\pi\);
2. a same-trajectory V-first no-split bound using a fixed-\(V^\pi\) ghost recovery target and deterministic \(\gamma\)-Lipschitz stability.

The two routes share state, pair, and edge-chain count/residual events. The required deliverables are explicit theorems, computable certificates, contract tests, and an updated comparison report. Fully online control, nonstationary starts, and general stochastic reward noise are outside the current scope.

## Active implementation

- `evaluate_fixed_policy_q_routes.py`
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
- `docs/superpowers/plans/2026-08-29-direct-q-v-first-balanced-exploration-plan.md`
- `docs/superpowers/plans/2026-08-29-crossfit-markov-certificate-plan.md`
- `results/fixed_policy_q_routes/`
- `results/fixed_policy_q_routes_crossfit/`
- `results/blockwise_q_routes/`

## Historical archive

Earlier diagnostics, generated outputs, smoke runs, superseded specs/plans, and temporary build trees were moved intact to:

`C:\Users\Admin\Desktop\research\_archive\icrl_softmax_history_20260831`

The archive is recoverable and remains outside the active project directory. Python and Ruff caches were deleted because they are reproducible.
