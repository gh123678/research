# FP-TU-001 Claude route: repair evidence after GPT cross-verification FAIL

Route: Claude independent implementation on branch `claude/FP-TU-001`,
worktree `C:\tmp\research-FP-TU-001-claude`. The GPT route's verification of
the formal seal `0fa824fb4e565bf0fff8c589ecf4b0cf32c21f4d` returned `FAIL`
with four required repair items. This document seals the repair evidence. The
GPT route's code, evidence, and results were not read at any point; only the
four defect statements were acted on, within the Claude route's own files.

## Item 1: theory.md sections 3-4 width-to-range gap

Defect: section 3 derived "the normalized increment `xi_t / B` lies in
`[-1, 1]`" from a conditional range of width `2B`. Width `2B` does not imply
`[-B, B]`: centering shifts the support interval by the conditional mean,
which need not vanish, so the centered support can be asymmetric or partly
outside `[-B, B]`.

Repair: section 3 now states only what width gives — the normalized increment
is conditionally centered with support in an interval `[c, d]` with
`d - c <= 2` and `c <= 0 <= d`. Section 4 states and proves the conditional
Hoeffding lemma in interval form: centered `Y in [c, d]` with `d - c <= 2`
implies `E[exp(a Y) | F] <= exp(a^2 (d-c)^2 / 8) <= exp(a^2 / 2)`, via the
chord bound plus `phi''(u) <= 1/4` (Bernoulli-tilt variance). The downstream
supermartingale step is unchanged because it only ever used the proxy value
1. The primary-sources mapping is updated accordingly.

Verifier addition: `verify_conditional_mgf_fixtures` now includes asymmetric,
zero-mean, width-exactly-2 two-point fixtures whose support exceeds `[-1, 1]`
(`(-1.5, 0.5)`, `(-0.5, 1.5)`, `(-1.25, 0.75)`, `(-0.75, 1.25)`), asserting
`MGF <= exp(a^2/2)` for `a in [-6, 6]`. PASS.

## Item 2: mixture_grid returned the mutable cached dict

Defect: `_GRID_CACHE` stored and returned the same mutable dict, so a caller
mutating the returned grid (or its lists) would pollute every subsequent
certificate built from the cached grid.

Repairs in `time_uniform_mixture_certificate.py`:

1. `mixture_grid` now caches a frozen internal copy and returns an isolated
   per-call copy (outer dict plus each list; all leaf values are immutable
   numbers). Callers cannot pollute the internal frozen grid.
2. `mixture_root` validates a caller-supplied `grid` against the frozen
   `n_groups`/`delta` parameters (`grid["n_groups"] == n_groups` and
   `grid["delta"] == delta`, else `ValueError`). Existing failure-path
   fixtures (which keep `n_groups`/`delta` intact and perturb only
   `log_weights`) still reach the ordered `mixture_inversion_unbracketed`
   reason unchanged.
3. `mixture_root` rejects `max_iterations > INVERSION_MAX_ITERATIONS` (200)
   with `ValueError`; the frozen default and cap remain 200.

No frozen formula, constant, or default numeric changed: grid size 15,
`k_j = 2^j`, `w_j = (j+1)^{-2} / sum`, `L_j = log(2G/(delta w_j))`,
`a_j = sqrt(2 L_j / k_j)`, `tol = 1e-12`, default `max_iterations = 200` are
all byte-identical in behavior to the sealed run.

Verifier addition: `verify_grid_isolation_and_iteration_cap` mutates every
mutable field of a returned grid, refetches, and asserts the frozen grid is
unaffected and downstream `mixture_radius` results are unchanged; asserts
`ValueError` for grids with mismatched `n_groups` or `delta`; asserts
`ValueError` for `max_iterations = 201` and successful convergence at exactly
200. PASS.

## Item 3: transition-variance conclusion narrowed

Defect: `transition_variance_feasibility.md` claimed "no observable strict
tightening exists" and "there is nothing observable left to estimate", and
the corresponding test comments matched that claim. This overstates the
result: it rules out only structure-free uniform constructions.

Repair: the document now concludes only that (i) without extra structure
beyond the width-`2B` range, the uniform worst case over data-consistent laws
and all `V in [-B, B]^m` attains `B^2` exactly (aligned-corner saturation,
compatible with arbitrarily large counts, sandwiched by Popoviciu); and (ii)
the data-dependent confidence-set + Bellman-coupling tightening proof is
currently not completed. It explicitly does not claim that adaptive tightening
is impossible for all data, and records that unseen mass shrinks with sample
confidence (`(1-p)^n` escape probability and `log(1/delta)/n` ceilings both
decrease in `n`), so the unseen-mass configuration alone cannot block
data-dependent tightening at large `n`; the binding obstruction is the fully
observed aligned-corner configuration. The test comments were rewritten to
match, with added shrinkage assertions. The numeric content of the test is
unchanged and PASSes.

## Item 4: no formal rerun; prior run-2 anomaly retained for adjudication

The formal 480-record matrix was NOT rerun for this repair. The sealed formal
evidence (`formal_result.md`, artifacts under
`results/FP-TU-001/claude`, run at `450881dc…`) remains the only formal
record. As documented in `formal_result.md`, that evidence includes a second
evaluation under the identical frozen command, executed after a defect repair
in the evaluator's summary composition (run 1's analysis failed on missing
legacy summary leaves; the records are deterministic from the frozen seed
schedule and the defect touched only a derived aggregate). That second run
was a procedural anomaly relative to the "single formal run" reading of the
frozen task; it is retained, disclosed, and left for the user's adjudication.
Nothing in the present repair touches the evaluator, the analyzer, any
formula, parameter, seed, or metric-affecting code path; the module changes
above are isolation-and-validation-only and the documentation changes affect
no artifact, so the sealed artifact hashes remain the formal record.

## Verification after repair

Six-verifier suite, each `C:\Users\Admin\anaconda3\python.exe -B <script>`,
all PASS (run 2026-09-07, same environment as the sealed runs):

- `verify_finite_sample_theorems.py` PASS
- `verify_fixed_policy_q_routes.py` PASS
- `verify_crossfit_markov_certificate.py` PASS
- `verify_end_to_end_sarsa.py` PASS
- `verify_visit_indexed_martingale_certificate.py` PASS
- `verify_time_uniform_mixture_certificate.py` PASS (including the new
  asymmetric interval-Hoeffding fixtures, grid-isolation/consistency/cap
  regression tests, and narrowed transition-variance checks)

`python -m ruff check --no-cache` on the four new files
(`time_uniform_mixture_certificate.py`,
`verify_time_uniform_mixture_certificate.py`,
`evaluate_time_uniform_certificates.py`,
`analyze_time_uniform_certificates.py`): all checks passed.

## Repaired source SHA-256 (working tree at repair time)

- `time_uniform_mixture_certificate.py`
  `90a8d06a08e2a2d3b40ff025659650678b0e6ec05e65beb0080bc36ebe0c58fc`
- `verify_time_uniform_mixture_certificate.py`
  `32719980ba6446716857a1cc2621f3947016c1439871b2e304271fa174779a9a`
- `docs/research_branches/FP-TU-001/claude/theory.md`
  `204e0548bd766a5cb4b5b50337a1acfa25a662e04ee851706a53ab1f586991cc`
- `docs/research_branches/FP-TU-001/claude/transition_variance_feasibility.md`
  `0b6e480040e9b628ba78e23b9682eb5e1d045e13c0bbc6407cef82bdd478324d`

## Acceptance impact

- Item 1 (filtrations / width inheritance): repair strengthens the proof;
  the certificate statement is unchanged.
- Item 6 (inversion contract): unchanged; the iteration cap is now enforced
  as an upper bound as well as a default.
- Item 18 (transition-variance negative result): conclusion narrowed to the
  structure-free uniform worst case; still nothing enters the certificate.
- All other numbered items are unaffected; the formal artifact set and its
  hashes are unchanged.

## Follow-up repair (second review round, 2026-09-07)

The re-review of the repair seal found item 2 only partially closed:
`mixture_root` validated a caller-supplied `grid` on `n_groups`/`delta` only,
so a grid with matching `G`/`delta` but tampered `weights`, `log_weights`,
`log_terms`, `rates`, `half_squared_rates`, or `count_grid` was still
accepted, violating the field-by-field frozen-grid contract.

Repair in `time_uniform_mixture_certificate.py`: a caller-supplied grid is
now validated by `_require_frozen_grid` against a freshly computed
`mixture_grid(n_groups, delta)` reference — exact key set, list lengths, and
every value (exact for integers, `math.isclose(rel=1e-15, abs=1e-15)` for
floats, booleans rejected everywhere), with `ValueError` on any departure.
Frozen formulas, constants, and defaults are unchanged.

Verifier changes: the previous lower-bracket failure-path test no longer
tampers with the grid; it now monkeypatches `module.log_mixture` locally to
force `log M(k, 0) >= target` and still reaches the ordered
`mixture_inversion_unbracketed` reason (the upper-bracket path remains
monkeypatched via `stitch_boundary`, and `max_iterations=1` covers the
not-converged path). New per-field tamper rejection tests in
`verify_grid_isolation_and_iteration_cap`: each of `weights`, `log_weights`,
`log_terms`, `rates`, `half_squared_rates` tampered by value, truncation,
and extension; `count_grid` value and boolean tampering; scalar `delta`
tampering beyond machine precision; boolean `grid_size`; extra key; missing
key; non-mapping grid — all must raise `ValueError`; an exact copy is
accepted.

Verification after this follow-up (same environment, no formal rerun, no
artifact or formula change):

- `verify_time_uniform_mixture_certificate.py` PASS.
- `python -m ruff check --no-cache` on the four new files: all checks passed.

Repaired source SHA-256 (working tree at follow-up time):

- `time_uniform_mixture_certificate.py`
  `19343831400a34c5b281975445b09cb724646c9c164317f04b1436af702dccf3`
- `verify_time_uniform_mixture_certificate.py`
  `a5d67b8e8d6b6a1d51c6c90f80e5663cda8794d1672174a6117bbfe0b55d4890`
