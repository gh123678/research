# FP-MART-001: Claude verification of the sealed GPT route

## Verification identity

- Verifier: Claude route.
- Date: 2026-09-04.
- GPT branch: `codex/FP-MART-001` (repository root).
- Frozen GPT seals verified:
  - implementation `4cf6f50d69c5aa4937d181f758f546f9d2213c41`;
  - blind first result `e7c04111b7aec8c9dc5043883fcaa68cc158837b`;
  - formal evidence `63b84fd4295598c3e020d7e489552470ac1763c4`.
- Common activation commit: `c8ec7e5c3165930663e26a07f98b38cc9ec186ad`.
- Verification request:
  `docs/research_branches/FP-MART-001/codex/claude_verification_request.md`.
- GPT evidence inspected read-only:
  `docs/research_branches/FP-MART-001/codex/theory.md`,
  `first_result.md`, `formal_result.md`, `verification_of_claude.md`,
  `verification_of_claude_repair.md`,
  `verification_of_claude_evidence_repair.md`.
- No GPT file, task file, Python file, or GPT result was edited; no command was
  rerun for this report beyond reading the already completed independent
  outputs at
  `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\claude\verification_codex`.

## Static findings (independent inspection)

- Seals and history confirmed: the GPT branch seals the implementation before
  the blind first result and the formal evidence after it, in the required
  order; no scientific Python change appears after the GPT implementation seal.
- Proof soundness: the sampling filtrations and stopped visit filtrations are
  explicit; all three residual families (state Bellman, pair Bellman,
  fixed-`V^pi` recovery) are martingale differences with conditional width
  `2B`; the compensated exponential supermartingale
  `M_t = exp(lambda S_t - lambda^2 B^2 sum J / 2)` with `E[M_n] <= 1` is
  correct; the `G = m + 2d` by `1 <= k <= n` union gives total failure
  probability at most `delta`; random-count substitution is valid through the
  simultaneous-in-`k` event without conditioning; selective emission semantics
  (`P(Emit and bound violated) <= delta`, no prior support claim) are exact;
  the radii compose with the unchanged deterministic Direct-Q all-layer and
  V-first no-split recurrences.
- Implementation soundness: certificate inputs use only observed counts,
  empirical matching diagonals, the declared reward bound
  `R_star = 1 + gap_bonus`, public hyperparameters, and algorithm metadata; no
  oracle quantity enters any certificate (the true reward maximum appears only
  in the post-construction `oracle_audit`); integer-count, dimension, horizon
  sum, and state/pair aggregation validation is enforced with deterministic
  ordered failures; the `visit_indexed_certificate` namespace, strict-JSON
  schema, and complete legacy config/task-record/summary audits are sound.
- No GPT defect and no frozen-task defect was found; no `OBJECTION` is raised.

## Reproduction record

Independent Claude-executed outputs already exist at
`results/FP-MART-001/claude/verification_codex/` (run from working directory
`C:\Users\Admin\Desktop\research\icrl_softmax`). This report records them
without rerunning:

- Ruff on the four route Python files: `All checks passed!`, exit code 0.
- The exact five required verifiers all exit code 0:
  `verify_finite_sample_theorems.py`,
  `verify_fixed_policy_q_routes.py`,
  `verify_crossfit_markov_certificate.py`,
  `verify_end_to_end_sarsa.py`,
  `verify_visit_indexed_martingale_certificate.py`
  (`PASS visit-indexed martingale certificate contracts`).
- Formal evaluator: wrote 480 matched comparisons; strict analyzer
  `ANALYZER_MODE: formal`, `ANALYZER_PASSED: True`.
- `LEGACY_MISMATCH_COUNT: 0`, `SCHEMA_PROBLEM_COUNT: 0`,
  `NAMESPACE_PROBLEM_COUNT: 0`, `EMPIRICAL_AUDIT_VIOLATION_COUNT: 0`.
- Emission totals: `direct_exact` 345/480, `direct_softmax` 312/480,
  `vfirst_nosplit_exact` 345/480, `vfirst_nosplit_softmax` 345/480
  (1,347 emitted route certificates in total).

Exact reproduction commands and path (from `commands.log`):

```text
WORKDIR: C:\Users\Admin\Desktop\research\icrl_softmax
EVALUATION: C:\Users\Admin\anaconda3\python.exe -B evaluate_visit_indexed_certificates.py --tasks 30 --trajectory-lengths 256 1024 4096 16384 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir results/FP-MART-001/claude/verification_codex
ANALYSIS: C:\Users\Admin\anaconda3\python.exe -B analyze_visit_indexed_certificates.py --baseline-dir results/fixed_policy_finite_sample_certificates --result-dir results/FP-MART-001/claude/verification_codex --mode formal
```

## Reproduced output hashes

SHA-256 of the seven files in the verification directory:

| File | SHA-256 |
| --- | --- |
| `config.json` | `bcc377b422711302163b976d5d5ece389e1e3ee8540d37817a8719bf9ec3bf8a` |
| `task_results.json` | `929e2f65689af850b65f000ee6675a8a87b3506c2e2a9600c07d28cf183c6d52` |
| `summary.json` | `fa13619755b627b9dcff281dc5e4bc9b4d0262b9a6592f2982012b53df2b34ed` |
| `regression.json` | `29f7d3711e4d3a7c61d525600c741becef8f5b9dea99ff713819e41a8e0b3b26` |
| `environment.json` | `8fa3fae1fee6221130e8f6df0575dca9b6a870dc14c4756c1943fe4395f063c9` |
| `commands.log` | `b29c061763bd7cdd98ddc19729d7e80842d8583de94d970b6ec3613e5ac567fa` |
| `checks.log` | `f1c40c8ea2a9c3111dc53c1a067eef85c2ae9f7ba3ccedd710c433dfaee20f95` |

Comparison against the sealed GPT formal hashes in
`codex/formal_result.md`:

- `config.json`, `task_results.json`, and `summary.json` are byte-identical to
  GPT's formal outputs (GPT recorded the same three hashes).
- `regression.json` differs from GPT's only in the recorded
  `new_result_directory` path; every gate outcome (480/480 compared records,
  zero mismatches, zero maximum numeric difference, passing config, summary,
  namespace, schema, emission-gate, and empirical audits) is identical, so it
  is semantically identical.
- `environment.json`, `commands.log`, and `checks.log` differ as expected:
  they record a different timestamp, git HEAD, working/output paths, and log
  text for an independent execution.

The frozen baseline hashes
(`config.json` `a2276eae...`, `task_results.json` `c84329bd...`,
`summary.json` `49846c08...`) were verified intact by the evaluator before and
after the run.

## Route-by-length emissions and usefulness

From the reproduced `summary.json` / `regression.json` `route_by_length`
(120 records per route per length):

| Length | Route | Emitted | Emission rate | `< B` useful | `< 2B` useful |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | Direct exact | 3/120 | 2.5% | 0/120 | 0/120 |
| 256 | Direct softmax | 3/120 | 2.5% | 0/120 | 0/120 |
| 256 | V-first exact | 3/120 | 2.5% | 0/120 | 0/120 |
| 256 | V-first softmax | 3/120 | 2.5% | 0/120 | 0/120 |
| 1024 | Direct exact | 102/120 | 85.0% | 0/120 | 0/120 |
| 1024 | Direct softmax | 74/120 | 61.667% | 0/120 | 0/120 |
| 1024 | V-first exact | 102/120 | 85.0% | 0/120 | 0/120 |
| 1024 | V-first softmax | 102/120 | 85.0% | 0/120 | 0/120 |
| 4096 | Direct exact | 120/120 | 100% | 0/120 | 0/120 |
| 4096 | Direct softmax | 115/120 | 95.833% | 0/120 | 0/120 |
| 4096 | V-first exact | 120/120 | 100% | 0/120 | 34/120 (28.33%) |
| 4096 | V-first softmax | 120/120 | 100% | 0/120 | 12/120 (10.0%) |
| 16384 | Direct exact | 120/120 | 100% | 0/120 | 7/120 (5.833%) |
| 16384 | Direct softmax | 120/120 | 100% | 0/120 | 0/120 |
| 16384 | V-first exact | 120/120 | 100% | 48/120 (40.0%) | 120/120 (100%) |
| 16384 | V-first softmax | 120/120 | 100% | 4/120 (3.33%) | 120/120 (100%) |

The primary preregistered usefulness metric
`improves_over_zero_initialization_rate` (`total_bound < B`) is serialized in
`summary.json` for every cell: zero everywhere except V-first exact 40% and
V-first softmax 3.33% at length 16384. The secondary descriptive
`< 2B` rates are the reconciled values recorded under the repair request's
dual-threshold reporting; neither threshold is selected post hoc. The
exact-route emission rates equal the preregistered support rates
2.5% / 85.0% / 100% / 100% with zero deviation at all four lengths, and the
100% exact-emission gate at lengths 4096 and 16384 passes.

Violations: zero oracle-audit violations among all 1,347 emitted certificates
(tolerance `1e-9`), zero per-group residual violations, and zero empirical
fixed-target residual violations across all 480 records.

## Limitations

- The certificate is per trajectory with selective semantics; it is not a
  simultaneous statement over the 480-record audit, and no empirical coverage
  statistic is used as proof.
- The Hoeffding radius is conservative: practically useful (`< B`) bounds
  appear only in the longest-trajectory V-first regime.
- The optional variance-adaptive route remains unavailable because no
  observable count-only variance proxy was proved or implemented from the
  frozen allowed inputs; this accurately recorded negative result does not
  reject the mandatory route.
- Environment, command-log, and checks-log hashes are execution-specific and
  cannot be byte-identical across independent runs.

## Acceptance-criteria mapping (frozen task FP-MART-001, numbered 1-18)

1. `PASS`: GPT `theory.md` derives the sampling filtrations, visit stopping
   times, measurability, martingale differences, conditional ranges (`2B`
   width), and the optional-skipping step for all three residual families.
2. `PASS`: one simultaneous event over all `G = m + 2d` groups and all
   `1 <= k <= n` with total failure probability at most `delta`, via the
   compensated exponential supermartingale.
3. `PASS`: random observed counts are substituted only through the
   simultaneous event; no conditional-independence or fixed-count assumption.
4. `PASS`: selective emission semantics are exact
   (`P(Emit and bound violated) <= delta`); no prior full-support claim.
5. `PASS`: certificate functions accept no oracle model, occupancy, value,
   residual, or true initial-error input; `B` derives from the predeclared
   public rule `R_star = 1 + gap_bonus`; zero initialization uses `B`.
6. `PASS`: Direct-Q all-layer and V-first no-split exact/softmax bounds compose
   with the visit-indexed radii without changing the completed deterministic
   recurrence; independent reproduction matches GPT's deterministic core
   byte-identically.
7. `PASS`: required support, empirical margins, algorithm mode, risk budget,
   and finite arithmetic are validated with deterministic ordered failures,
   including integer-count, dimension, horizon-sum, and aggregation checks.
8. `PASS`: new fields use the `visit_indexed_certificate` namespace and the
   `selective_high_probability_certified` status without changing legacy status
   meanings.
9. `PASS`: all machine outputs are strict JSON with `null` for unavailable
   values; certificate inputs are structurally separated from `oracle_audit`
   fields; schema and oracle-provenance problem counts are zero.
10. `PASS`: all five required verifiers and Ruff pass under independent Claude
    execution.
11. `PASS`: GPT's smoke gates (strict JSON, nonfinite, oracle separation,
    legacy regression, all 16 records baseline-aligned) passed before the GPT
    formal matrix began, as recorded in `first_result.md`.
12. `PASS`: the GPT formal result has exactly 480 records exactly matching the
    frozen configuration and seed 20260829, confirmed by byte-identical
    `config.json` and `task_results.json`.
13. `PASS`: all legacy nonnumeric leaves match exactly and numeric leaves yield
    zero mismatches (maximum numeric difference 0.0 for config, task records,
    and the summary).
14. `PASS`: exact-route emission is 100% at lengths 4096 and 16384; all four
    rates, both usefulness thresholds, and zero deviations from
    2.5%/85.0%/100%/100% are reported without tuning.
15. `PASS`: every empirical audit violation is enumerated; the count is zero
    and no empirical coverage statistic is used as proof.
16. `PASS`: GPT records exact commits, environment, commands, raw-output paths
    and hashes, anomalies, metrics, limitations, and a numbered acceptance
    assessment in `first_result.md` and `formal_result.md`.
17. `PASS`: reciprocal verification is now complete in both directions — GPT's
    verification of the repaired Claude route ended in `PASS`
    (`verification_of_claude_evidence_repair.md`, verified Claude commit
    `a559bd31769506565dd4503d8239d4a3f28ddf81`), and this report is Claude's
    verification of the sealed GPT route, ending in `PASS`.
18. `PENDING`: the final synthesis explaining material route differences and
    the `ACTIVE_WORKSPACE.md` update await GPT's final synthesis; `main`
    remains unchanged pending user approval.

PASS
