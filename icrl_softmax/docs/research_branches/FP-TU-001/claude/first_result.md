# FP-TU-001 Claude route: blind first result (smoke seal)

Route: Claude independent implementation on branch `claude/FP-TU-001`,
worktree `C:\tmp\research-FP-TU-001-claude`, common activation commit
`0ce18b4676f70ca0556496e804aa65563efa63da` (recorded in
`environment.json:git_head_at_run`).

This document seals the implementation and the smoke evidence before any
inspection of the GPT route. The formal 480-record matrix has NOT been run;
it is plan Task 10 and happens only after both routes seal.

## Scope of this seal

- `icrl_softmax/time_uniform_mixture_certificate.py` — pure certificate module.
- `icrl_softmax/verify_time_uniform_mixture_certificate.py` — contract verifier.
- `icrl_softmax/evaluate_time_uniform_certificates.py` — additive evaluator.
- `icrl_softmax/analyze_time_uniform_certificates.py` — strict regression analyzer.
- `docs/research_branches/FP-TU-001/claude/theory.md` — independent proof.
- `docs/research_branches/FP-TU-001/claude/transition_variance_feasibility.md` —
  feasibility-only study (negative result; never in the certificate).
- Smoke output: `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-TU-001\claude\smoke\`
  (exactly seven core artifacts).

No task, design, plan, workspace, baseline, archive, or GPT file was modified.
No GPT post-activation branch, evidence, or result was read.

## Source SHA-256 at seal time

- `time_uniform_mixture_certificate.py`:
  `6c9b00ce10de80c2e0147ac29c7a579c5bfebc4166ff2a48206c0e8cadaa6de5`
- `verify_time_uniform_mixture_certificate.py`:
  `e257592a22743b838aafa8d17d3e9970444a4b3ed7f5fcb6322b63baa42e3301`
- `evaluate_time_uniform_certificates.py`:
  `60b34bb4d288df453d5c8f951f45b6eca5873da4c4f21d97eb03e5a43046d2dc`
- `analyze_time_uniform_certificates.py`:
  `fc4461aba879107460640dd781785ee4d6b045412a4d484e788214618386454a`
- `theory.md`:
  `21c371d39fb60e357a1ba3ca6440498a856da1adbfcd91f29896e8c7ba221cb5`
- `transition_variance_feasibility.md`:
  `5f58ef4c4c65414bd4debe44141408688e8604dcde9d8eacf49fadcee39d5101`

## Environment

From `smoke/environment.json`: Python 3.13.9
(`C:\Users\Admin\anaconda3\python.exe`), Windows-11-10.0.22631-SP0,
numpy 2.4.6, scipy 1.16.3, matplotlib 3.10.6, mpmath 1.3.0, run at
2026-09-06T15:41:09Z.

## Commands executed (chronological)

1. Failing-test-first record, before the module existed:

   ```text
   C:\Users\Admin\anaconda3\python.exe -B verify_time_uniform_mixture_certificate.py
   Traceback (most recent call last):
     File "C:\tmp\research-FP-TU-001-claude\icrl_softmax\verify_time_uniform_mixture_certificate.py", line 20, in <module>
       from time_uniform_mixture_certificate import (
       ...<15 lines>...
       )
   ModuleNotFoundError: No module named 'time_uniform_mixture_certificate'
   EXIT_CODE: 1
   ```

2. After implementing the module, the same command prints
   `PASS time-uniform mixture certificate contracts` (exit 0).
3. Six-verifier preflight (also re-executed by the evaluator and recorded in
   `smoke/checks.log`), each `python -B <script>`, all PASS:
   `verify_finite_sample_theorems.py`, `verify_fixed_policy_q_routes.py`,
   `verify_crossfit_markov_certificate.py`, `verify_end_to_end_sarsa.py`,
   `verify_visit_indexed_martingale_certificate.py`,
   `verify_time_uniform_mixture_certificate.py`.
4. `python -m ruff check` on the four new files: all checks passed.
5. Smoke evaluation (verbatim from `smoke/commands.log`):

   ```text
   C:\Users\Admin\anaconda3\python.exe -B evaluate_time_uniform_certificates.py --tasks 2 --trajectory-lengths 256 1024 4096 16384 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-TU-001/claude/smoke
   ```

   Output: `wrote 32 matched comparisons`; emissions direct_exact 22/32,
   direct_softmax 20/32, vfirst_nosplit_exact 22/32,
   vfirst_nosplit_softmax 22/32; `PASS time-uniform certificate evaluation`.
6. Smoke analysis:

   ```text
   C:\Users\Admin\anaconda3\python.exe -B analyze_time_uniform_certificates.py --result-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-TU-001/claude/smoke --mode smoke
   ```

   `PASS time-uniform certificate analysis`.

## Smoke artifact SHA-256

- `config.json` `206f92478d8dc8dcda0b21f558e5ea1234c10c6418976d36369ece921cc5cccd`
- `task_results.json` `3bf239951c68a1bf62211f0f07bb5202110713e6f287bdad335e437d2503cebd`
- `summary.json` `58e2ec66f24c0fb140345923ce2385e43cd79e2dbfd3c0b1b45418efed1c310c`
- `regression.json` `9f61863f3e27017f2225b0dbb5faef046a501330420f3780d2a5317054fd1407`
- `environment.json` `5a517906ae64e55e7b9415d02d18a6971afc4590fc042c164463045c6840d56e`

Frozen baseline verified before and after the run (hashes and 480-record
count unchanged): `config.json` bcc377b4…, `task_results.json` 929e2f65…,
`summary.json` fa136197… at
`C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex`.

## Smoke metrics (from `smoke/regression.json`)

- Legacy record regression: 0 mismatches under
  `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`; 32/32 records compared.
- Namespace audit: additions are exactly `time_uniform_certificate` in
  config, every record, and theorem-route summary rows.
- Schema audit: 0 problems (frozen grid/weights/log-terms, G = m+2d,
  per-group risk delta/G, total delta, no route-level resplit, pair/recovery
  radius equality, status/reason consistency, oracle separation).
- Recorded-radius recomputation: 1672 radii recomputed from the public module,
  0 problems; max mixture/old radius ratio 0.9437545; strict dominance seen.
- Exhaustive frozen count audit (recomputed independently by the analyzer):
  per-length max ratios 0.9573574 (n=256), 0.9263356 (n=1024),
  0.8975201 (n=4096), 0.8793887 (n=16384), all below 1; `r_mix`
  nonincreasing through 16384; the per-record checks imply the global
  `n_max = 16384` audit because `r_old(k; n)` increases in `n`.
- Emission/bound audit: 86 emitted route pairs compared; emission decisions
  identical to legacy in every record; every emitted total bound
  nonincreasing (min reduction 0.5607, mean reduction 9.8403).
- Empirical oracle audits: 0 violations (mixture and stitch residual events,
  emitted route bounds); diagnostics only, not theorem evidence.
- Per-length route rows: emission match rate 1.0 everywhere; at length 16384
  the V-first routes reach `<B` usefulness rates 0.875 (exact) and 0.625
  (softmax) with `<2B` rate 1.0; Direct-Q remains above `B` at these counts.

## Anomalies and failed runs

1. Expected pre-module verifier failure recorded above (failing test first).
2. During development the verifier exposed three defects in my own draft,
   all fixed before this seal: a dead memoization block in `mixture_root`;
   support-missing reasons conflated with inversion-failure reasons (now
   separated: count-based support completeness vs deterministic inversion
   failure); a verifier fixture whose scaled-rate grid could not break the
   upper bracket (replaced by a monkeypatched bracket to exercise the
   `mixture_inversion_unbracketed` path deterministically).
3. 270 legacy float leaves differ from the frozen baseline by at most
   1.046e-11 absolute (about 3e-15 relative, a few ulps) inside legacy
   statistical terms of magnitude ~2e3-4e3, e.g.
   `routes.vfirst_nosplit_softmax.finite_sample_certificate.state_value.high_probability.total_bound`.
   All pass the frozen comparator with zero mismatches. Attribution:
   reduction-order-level floating-point nondeterminism in the shared legacy
   numeric helpers across runs; the comparator exists precisely to absorb
   this scale.
4. Repo-wide `ruff check .` reports 9 pre-existing errors in out-of-scope
   legacy files (`analyze_fixed_policy_finite_sample_certificates.py`,
   `model.py`, `tools/*`); the four new files are clean. The pre-existing
   errors predate this task and were left untouched.

## Limitations

- Smoke covers 2 of 30 frozen tasks per cell (32 records); it is not the
  formal evidence and cannot replace it.
- Config and summary regression against the baseline are intentionally
  skipped in smoke mode (reduced task set); they are mandatory in formal mode.
- The transition-variance study concludes a deterministic obstruction (no
  observable strict tightening below `B^2` exists); it contributed nothing to
  the certificate.
- The certificate remains a fixed-policy, fixed-context, synchronous,
  single-trajectory statement under the frozen scope limits.

## Numbered acceptance assessment (smoke stage)

1. PASS — `theory.md` sections 1-3 map the verified FP-MART-001 filtrations,
   stopping times, measurability, martingale differences, and width `2B`
   without weakening; the module consumes only counts and declared bounds.
2. PASS — `theory.md` sections 4-6: fixed-rate supermartingale, cosh mixture
   with initial value one, Ville at `delta/G` per group, union to `delta`;
   verifier checks initial value and allocation numerically.
3. PASS — `theory.md` section 8; evaluator/analyzer record selective
   semantics (`P(Emit and violated) <= delta`), no support-probability claim,
   no division by emission probability.
4. PASS — `theory.md` section 10 proves independent stitch validity and
   `q_mix(k) <= q_stitch(k)`; verifier checks it for every k through 16384;
   the stitch is never selected as the reported certificate.
5. PASS — module signature excludes oracle quantities; `B` derives only from
   the declared `reward_bound`; schema audit's banned-key scan passes.
6. PASS — verifier `verify_exhaustive_counts` and `verify_high_precision_roots`
   (mpmath 60 dps): stable inversion, conservative root, finiteness,
   `q_mix <= q_stitch`, and monotonicity for every `1 <= k <= 16384`.
7. PASS — dominance anchors recomputed independently: max ratios 0.9573574 /
   0.9263356 / 0.8975201 / 0.8793887, all below 1 with strict inequalities;
   global `n_max` audit implied and recorded.
8. PASS — route recurrences unchanged (fixture cross-check against
   `build_visit_indexed_certificate` legacy totals); smoke emission/bound
   audit confirms identical emission and nonincreasing emitted totals.
9. PASS — ordered deterministic failure ranks including
   `mixture_inversion_unbracketed`, `mixture_inversion_not_converged`,
   `mixture_root_not_conservative`; rejection and ordering tests pass;
   inversion failure never falls back to the old radius.
10. PASS — namespace audit: exactly `time_uniform_certificate` added;
    zero legacy mismatches.
11. PASS — strict JSON loader rejects duplicate keys and nonfinite constants
    (tested); `null` for unavailable values; `oracle_audit` structurally
    separated inside `time_uniform_certificate`.
12. PASS (Claude route) — six verifiers, Ruff on new files, schema checks,
    and baseline-integrity checks all pass.
13. PASS (Claude half) — this smoke matrix passes and is sealed before any
    formal run and before reading the GPT route.
14. PENDING — formal 480-record matrix is Task 10, not yet run.
15. PASS at smoke scope — all compared legacy nonnumeric leaves exact; zero
    numeric mismatches under the frozen comparator.
16. REPORTED — smoke exact-route emission 0%/75%/100%/100% at
    256/1024/4096/16384 (2 tasks per cell; the preregistered
    2.5%/85%/100%/100% comparator applies to the formal 480-record run);
    softmax, usefulness, and radius-reduction metrics are in
    `regression.json:route_by_length`, untuned.
17. PASS — zero empirical audit violations; the enumeration machinery is in
    `regression.json` and treats audits as diagnostics only.
18. PASS — `transition_variance_feasibility.md` records the exact
    obstruction (uniform supremum equals `B^2`; unseen-mass invisibility);
    nothing from it enters the certificate.
19. PASS — this document records commits, environment, commands, hashes,
    successful and failed runs, anomalies, and limitations.
20. PENDING — reciprocal cross-verification occurs only after both seals.
