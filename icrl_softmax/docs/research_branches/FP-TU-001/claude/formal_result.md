# FP-TU-001 Claude route: formal result (480-record matrix)

Route: Claude independent implementation on branch `claude/FP-TU-001`,
worktree `C:\tmp\research-FP-TU-001-claude`, common activation commit
`0ce18b4676f70ca0556496e804aa65563efa63da`. The formal matrix ran at
`git_head_at_run = 450881dc62d6588f0542392df97705fcb92c989e` (the blind
first-result seal). The GPT route's code, evidence, and results were not
read at any point.

## Frozen command and run history

The one frozen formal command (plan Task 10, Claude arguments and result
directory):

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_time_uniform_certificates.py --tasks 30 --trajectory-lengths 256 1024 4096 16384 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-TU-001/claude
```

Analysis command:

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_time_uniform_certificates.py --result-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-TU-001/claude --mode formal
```

Run history, recorded in full:

1. Run 1 (evaluation): completed, 480 records, emissions
   direct_exact 345/480, direct_softmax 312/480, vfirst_nosplit_exact
   345/480, vfirst_nosplit_softmax 345/480; `PASS` from the evaluator.
2. Run 1 (analysis): FAILED. The summary regression found that my
   evaluator's `augment_summary` had replaced, rather than composed with,
   the legacy visit-indexed summary augmentation, so the legacy
   `visit_indexed_certificate` summary blocks were missing from new summary
   rows (acceptance item 10 requires legacy leaves preserved). Smoke had not
   caught this because smoke mode intentionally skips config/summary
   regression. Every other audit already passed on this run (records,
   namespace, schema, radii, emission/bound, zero oracle violations).
3. Repair: one-line composition fix in `evaluate_time_uniform_certificates.py`
   (apply the legacy `augment_summary` first, then add the
   `time_uniform_certificate` block). No formula, parameter, seed, radius,
   or metric-affecting code was touched; `summarize` and the record pipeline
   are unchanged, and the summary is a deterministic aggregate of the
   records. Ruff clean.
4. Run 2 (evaluation): identical frozen command, 480 records, identical
   emission counts (345/312/345/345), `PASS`.
5. Run 2 (analysis): `PASS time-uniform certificate analysis`.

The rerun was a defect repair before sealing, not metric tuning: the record
content is fixed by the frozen seed schedule, the defect was in a derived
aggregate only, and both runs are recorded here. The sealed evidence is
run 2; run 1's failed `regression.json` and log lines were overwritten by
the frozen pipeline's own file handling and are documented in this section.

## Seven core artifact SHA-256 (sealed)

- `config.json` `74c83a78a7ffcc14d427d999923fa3c304726900dfa607597a89ec3755262510`
- `task_results.json` `b898751859a4f8c459960585f3f143550bc2a4babf48f28baa8b77797bcd2ed9`
- `summary.json` `c7280185170b7041218676a3151404533a5a36431c42d3f52841577d5c6082e9`
- `regression.json` `dbf691f22e65652a2361cae729e0dfb561feb5912ebd5ca4bb2b1cd4d570ab3b`
- `environment.json` `c2619946d9333c5a2e167da3ff041bf6acb1dbb099c4d883fcede6684318a537`
- `commands.log` `91b726a4b002c0b05748f181bd369c35eef789ea42f6838b92557b57de220abf`
- `checks.log` `cd4b3b6beca52f04e6568057678b6d1d0fb4519a22c390b0db958d0b7858ea28`

Frozen baseline verified before and after both runs (hashes and 480-record
count unchanged): `config.json` bcc377b4…, `task_results.json` 929e2f65…,
`summary.json` fa136197… at
`C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex`.

## Environment

From `environment.json`: Python 3.13.9 (`C:\Users\Admin\anaconda3\python.exe`),
Windows-11-10.0.22631-SP0, numpy 2.4.6, scipy 1.16.3, matplotlib 3.10.6,
mpmath 1.3.0; sealed run created 2026-09-06T15:54:47Z. Six-verifier
preflight (all PASS) is recorded in `checks.log`.

## Formal metrics (from `regression.json`, all sections passed)

- Records: 480 configured, 480 actual, exact frozen task-key set.
- Legacy regression: 0 mismatches on records, config, and summary; maximum
  numeric difference 0.0 (bit-identical legacy leaves this run).
- Namespace audit: additions exactly `time_uniform_certificate`; schema
  audit: 0 problems in 480 records.
- Radius recomputation: 25,028 recorded radii recomputed from the public
  module with 0 problems; max mixture/old radius ratio 0.9492067; strict
  dominance observed.
- Exhaustive frozen count audit: per-length max radius ratios 0.9573574
  (n=256), 0.9263356 (n=1024), 0.8975201 (n=4096), 0.8793887 (n=16384), all
  below 1; `r_mix` nonincreasing through 16384; the global `n_max = 16384`
  audit is implied by the per-record checks.
- Emission/bound audit: 1,347 emitted route pairs; emission decisions
  identical to the legacy certificate in every record and route; every
  emitted total bound nonincreasing (min reduction 0.4899, mean 11.0209);
  recorded old totals match legacy totals exactly.
- Empirical oracle audits: 0 violations among all audited emissions
  (diagnostics only, not theorem evidence).

Per-length route summary (120 records each; emission / `<B` / `<2B` rates):

| length | route | emission | `<B` | `<2B` | mean reduction vs old |
|---|---|---|---|---|---|
| 256 | direct_exact | 0.025 | 0.0 | 0.0 | 14.9519 |
| 256 | direct_softmax | 0.025 | 0.0 | 0.0 | 31.5005 |
| 256 | vfirst_nosplit_exact | 0.025 | 0.0 | 0.0 | 5.4053 |
| 256 | vfirst_nosplit_softmax | 0.025 | 0.0 | 0.0 | 5.4341 |
| 1024 | direct_exact | 0.85 | 0.0 | 0.0 | 12.8131 |
| 1024 | direct_softmax | 0.6167 | 0.0 | 0.0 | 124.1870 |
| 1024 | vfirst_nosplit_exact | 0.85 | 0.0 | 0.0 | 4.4538 |
| 1024 | vfirst_nosplit_softmax | 0.85 | 0.0 | 0.0 | 4.4711 |
| 4096 | direct_exact | 1.0 | 0.0 | 0.0 | 3.7949 |
| 4096 | direct_softmax | 0.9583 | 0.0 | 0.0 | 12.9422 |
| 4096 | vfirst_nosplit_exact | 1.0 | 0.0 | 0.7583 | 1.4947 |
| 4096 | vfirst_nosplit_softmax | 1.0 | 0.0 | 0.5833 | 1.5032 |
| 16384 | direct_exact | 1.0 | 0.0 | 0.6333 | 1.8094 |
| 16384 | direct_softmax | 1.0 | 0.0 | 0.0 | 4.6916 |
| 16384 | vfirst_nosplit_exact | 1.0 | 0.8417 | 1.0 | 0.7581 |
| 16384 | vfirst_nosplit_softmax | 1.0 | 0.4833 | 1.0 | 0.7631 |

Exact-route emission rates 2.5% / 85% / 100% / 100% match the preregistered
expectations at every length; all deviations are attributed to observed pair
support, not tuning.

## Anomalies

1. Run-1 analysis failure and repair rerun, documented above.
2. In the smoke stage, 270 legacy float leaves had differed from the frozen
   baseline by at most 1.046e-11 absolute (a few ulps); the formal sealed run
   reproduces every legacy leaf bit-identically (max difference 0.0),
   confirming the smoke differences were reduction-order-level floating-point
   noise in shared legacy helpers, absorbed by the frozen comparator.
3. Repo-wide `ruff check .` still reports the same 9 pre-existing errors in
   out-of-scope legacy files; all four new files are clean.

## Limitations

- Fixed policy, fixed context, synchronous updates, one trajectory per task,
  deterministic bounded edge rewards; no other scope is claimed.
- The transition-variance feasibility study remains a separate negative
  result and contributed nothing to the certificate.
- The line-stitching boundary remains audit-only and was never selected.
- Empirical oracle audits are diagnostics; selective validity follows from
  the proof, not observed coverage.

## Numbered acceptance assessment (formal stage)

1. PASS — filtrations, stopping times, measurability, martingale
   differences, and width `2B` inherited without weakening (`theory.md`
   sections 1-3; verifier fixtures).
2. PASS — one cosh-mixture event over all `G = m + 2d = 54` groups and all
   visit counts at total failure probability `delta = 0.05` (`theory.md`
   sections 4-6; schema audit confirms per-group `delta/G` allocation and no
   route-level resplit in all 480 records).
3. PASS — random final counts substituted through the time-uniform event;
   selective `P(Emit and violated) <= delta` semantics recorded in every
   record; no support-probability claim.
4. PASS — stitch independently proved (`theory.md` section 10), verified as
   an upper bracket for every count through 16384, never selected.
5. PASS — the pure module accepts no oracle input; `B` derives only from the
   declared `1 + gap_bonus` reward bound; banned-key scan clean.
6. PASS — every `1 <= k <= 16384` passes stable inversion, conservative
   root, finiteness, `q_mix <= q_stitch`, and monotonicity (verifier plus
   independent analyzer recomputation; mpmath 60-dps cross-check).
7. PASS — `r_mix(k) <= r_old(k; n)` for every frozen `n` and `k`, with
   strict inequalities; anchors 0.9573574 / 0.9263356 / 0.8975201 /
   0.8793887 recomputed independently; global `n_max` audit implied and
   recorded.
8. PASS — unchanged deterministic recurrences; 1,347 emitted pairs compared;
   all emission decisions identical and all emitted total bounds
   nonincreasing versus the frozen baseline.
9. PASS — ordered deterministic failure handling exercised in the verifier
   (support, margins, mode, divergence, inversion branches) and observed in
   formal records (`pair_support_missing` 117/120 at length 256,
   `pair_kernel_margin_nonpositive` up to 28/120 at length 1024).
10. PASS — new fields only below `time_uniform_certificate`; zero legacy
    mismatches on records, config, and summary.
11. PASS — strict JSON rejects duplicate keys and nonfinite values
    (analyzer loader and verifier tests); `null` for unavailable values;
    `oracle_audit` structurally separated.
12. PASS (Claude route) — six verifiers, Ruff on new files, schema checks,
    baseline-integrity checks.
13. PASS — smoke matrix passed and was sealed at
    `450881dc62d6588f0542392df97705fcb92c989e` before this formal run.
14. PASS — exactly 480 records, exact frozen configuration, seed, and task
    identity; exact key set enforced.
15. PASS — all legacy nonnumeric leaves exact; numeric leaves have zero
    mismatch under `math.isclose(rel_tol=1e-12, abs_tol=1e-12)` (observed
    maximum difference 0.0).
16. PASS — exact-route emission 2.5% / 85% / 100% / 100% matches the
    preregistered rates; softmax, usefulness, radius-reduction, and
    deviation metrics are reported above and in `regression.json` without
    tuning.
17. PASS — zero empirical audit violations; enumerated in `regression.json`;
    not used as theorem evidence.
18. PASS — transition-variance obstruction recorded in
    `transition_variance_feasibility.md`; never entered the certificate.
19. PASS — commits, environment, commands, hashes, successful and failed
    runs, anomalies, and limitations recorded here and in
    `first_result.md`.
20. PENDING — reciprocal cross-verification follows only after the GPT
    route's seals exist.
