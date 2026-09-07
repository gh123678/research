# FP-TU-001 Claude verification of the GPT route (evidence stage)

- Date: 2026-09-07.
- Verifier: Claude Code, branch `claude/FP-TU-001`, writable worktree
  `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-TU-001\claude_worktree_writable`.
- Subject: GPT route, branch `codex/FP-TU-001`, checkout
  `C:\Users\Admin\Desktop\research`.
- Tool boundary: read-only against GPT paths plus local execution of verifiers,
  Ruff, hashing, metric extraction, and one analyzer reproduction on a
  temporary copy inside the Claude worktree. The formal evaluator was not
  rerun. No GPT file was modified. The temporary copy was deleted before this
  report was written.

## 1. GPT checkout state

- GPT HEAD: `f01a49da2bd044562154ea6c9b2014dbee420f4f` on `codex/FP-TU-001`.
- `git status --porcelain` at verification time showed uncommitted changes
  owned by GPT: modified `icrl_softmax/ACTIVE_WORKSPACE.md`,
  `icrl_softmax/docs/research_branches/FP-TU-001/codex/verification_of_claude.md`,
  `icrl_softmax/docs/research_tasks/FP-TU-001.md`, and untracked
  `icrl_softmax/docs/research_branches/time_uniform_mixture_certificate_report.md`
  and `.../time_uniform_mixture_certificate_theory.md`. These are GPT-scope
  files; they were left untouched and are recorded here as an anomaly for GPT
  to commit or explain. None of the seven formal artifacts, the four GPT
  Python entry points, or GPT's `FP-TU-001/codex/` evidence documents are in
  this uncommitted set, so the sealed evidence under verification is intact.
- The task record at HEAD states Claude verification was previously
  `BLOCKED_BY_ENVIRONMENT`; this session is the environment-repaired retry.

## 2. Six verifiers and Ruff on the GPT checkout

Commands run from `C:\Users\Admin\Desktop\research\icrl_softmax` with
`C:\Users\Admin\anaconda3\python.exe -B <script>`; each run twice (once for
output, once for clean exit capture):

| Verifier | Exit |
| --- | ---: |
| `verify_time_uniform_mixture_certificate.py` | 0 |
| `verify_finite_sample_theorems.py` | 0 |
| `verify_fixed_policy_q_routes.py` | 0 |
| `verify_crossfit_markov_certificate.py` | 0 |
| `verify_end_to_end_sarsa.py` | 0 |
| `verify_visit_indexed_martingale_certificate.py` | 0 |

All six print their `PASS` lines; spot-checked numeric residuals are at
machine precision (e.g. Bellman errors 1e-16, stationary formula error
9.7e-17).

Ruff:

- `python -m ruff check time_uniform_mixture_certificate.py
  verify_time_uniform_mixture_certificate.py
  evaluate_time_uniform_certificates.py analyze_time_uniform_certificates.py
  visit_indexed_martingale_certificate.py` -> `All checks passed!`, exit 0.
- Observation: a whole-tree `ruff check .` reports 9 pre-existing diagnostics
  (F401/F841/F541) in files outside the FP-TU-001 scope (`model.py`,
  `tools/build_mobile_tutorial_pdf.py`, `tools/md2pdf_latex.py`,
  `analyze_fixed_policy_finite_sample_certificates.py`). These predate the
  task, are not in either route's write scope, and do not affect acceptance
  item 12 as scoped to the task files. Recorded as an observation, not a
  failure.

## 3. Frozen baseline and sealed artifact integrity

SHA-256 computed locally over `results/FP-MART-001/codex/`:

- `config.json` = `bcc377b422711302163b976d5d5ece389e1e3ee8540d37817a8719bf9ec3bf8a` — matches frozen task value.
- `task_results.json` = `929e2f65689af850b65f000ee6675a8a87b3506c2e2a9600c07d28cf183c6d52` — matches.
- `summary.json` = `fa13619755b627b9dcff281dc5e4bc9b4d0262b9a6592f2982012b53df2b34ed` — matches.

SHA-256 over the seven canonical GPT artifacts in
`results/FP-TU-001/codex/`, each exactly matching the sealed values in GPT
`formal_result.md`:

- `checks.log` `94f87599448d5e03f830621f9ee179d6ab2afcce889ca3fa89b83bb4a62265dd`
- `commands.log` `9c718f51904bae4d41c11f008f7571874779205870290c66699cf30cde3b6b41`
- `config.json` `43dcb96b0f6f95e76f1c0b484d6375e3727dbb5609b16a8952a69e2ac0dddf3a`
- `environment.json` `35277785974e2999936fdb58ecbc260107d4a6a8e4a2e6e4244ce37d8a27662d`
- `regression.json` `7c3d4d60734da4684e96cca604a7ebe443e87c0de14cc16b7440cf62d8556789`
- `summary.json` `565fc4d261a938d13350984bb97242e79fba42f014d11f807a18942517f4444f`
- `task_results.json` `0e5eab39bf49894832c5ebcdd6f7b70c453fff6b9600b234617889f8f9fa79be`

Record count: `task_results.json` is a JSON list of exactly 480 records.

## 4. Compact metrics independently extracted from the sealed artifacts

From `regression.json` (status `PASS`):

- baseline record count 480, result record count 480, `formal_protocol: true`;
- `legacy_mismatch_count: 0` under rel/abs tolerance 1e-12;
- radius audit: 16384 counts checked, maximum bisection iterations 42,
  maximum `q_mix/q_stitch` 0.9959405625588716, 21760 strict comparisons,
  `passed: true`;
- maximum mixture/legacy radius ratio by horizon: 256 -> 0.9573573993365391,
  1024 -> 0.9263355819163681, 4096 -> 0.8975200574107253,
  16384 -> 0.8793886652750217 (all below one, matching the Claude pre-review
  closure values);
- observed radius ratio min/mean/max: 0.7379884021918311 /
  0.8508755029787135 / 0.9492067139237118;
- emissions by route and length: direct_exact 3/102/120/120, direct_softmax
  3/74/115/120, vfirst_nosplit_exact 3/102/120/120,
  vfirst_nosplit_softmax 3/102/120/120 — exact emissions exactly
  2.5%/85%/100%/100% as frozen;
- failure reasons across 480 records: `pair_support_missing` 540,
  `pair_kernel_margin_nonpositive` 33, `state_support_missing` 26 (counts are
  per route-family and need not sum to 480);
- mean total-bound reductions among emitted records: direct_exact
  5.867580794080362 (n=345), direct_softmax 36.33231682144782 (n=312),
  vfirst_nosplit_exact 2.147345778082567 (n=345),
  vfirst_nosplit_softmax 2.1574473975175636 (n=345);
- oracle audit: zero route violations, zero residual violations,
  `used_as_theorem_evidence: false`.

From `summary.json`: 176 cells; the 64 certificate-bearing cells
(16 route/length x 4 mixing/gap settings) reproduce the same emission counts
(3, 102, 120, 120 per exact route across lengths; direct_softmax 3/74/115/120),
with `all_emitted_bounds_nonincreasing: true` and zero oracle violations in
every inspected cell. Primary `<B` usefulness appears only at length 16384
(V-first exact 101, V-first softmax 58 in the sealed record audit; consistent
with the rates quoted in `ACTIVE_WORKSPACE.md`).

All extracted numbers agree with GPT `formal_result.md` to every printed
digit.

## 5. Analyzer reproduction on a temporary copy

Because `analyze_time_uniform_certificates.py` rewrites `regression.json` and
`checks.log` in its result directory, the seven canonical artifacts were
copied byte-for-byte to
`icrl_softmax/results/FP-TU-001/claude_verification_copy/` inside the Claude
worktree, and the analyzer was run from the GPT checkout:

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_time_uniform_certificates.py --result-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-TU-001/claude_worktree_writable/icrl_softmax/results/FP-TU-001/claude_verification_copy --baseline-dir results/FP-MART-001/codex
```

Output: `PASS FP-TU-001 analysis: records=480, formal=True`, exit code 0.
The analyzer independently re-verified the three frozen baseline hashes
before and after analysis, strict JSON, the seven-artifact layout, 480
records, frozen formal emissions, and zero legacy mismatches on the copy.

The entire temporary copy directory was then deleted (`rm -rf`), and the
`results/` skeleton created for the copy inside the worktree was removed.
Re-hashing the seven canonical GPT artifacts after the run returned exactly
the sealed values listed in section 3 — the canonical evidence is unchanged.

## 6. GPT theory review (`docs/research_branches/FP-TU-001/codex/theory.md`)

- Filtration inheritance is stated correctly: state Bellman residuals selected
  by `G_t`-measurable rules, pair/recovery residuals by `H_t`; optional
  skipping at predictable visit times; conditional width `2B` with
  `B = R_star/(1-gamma)`; `G = m + 2d` groups. No kernel, occupancy, true
  value, residual, route error, or initial error enters the certificate.
- The fixed-rate step uses conditional Hoeffding on `Z_k = S_k/B` giving
  `E[exp(a dZ)] <= exp(a^2/2)`; the cosh symmetrization mixes `+a_j` and
  `-a_j`; `M_0 = 1`; Ville at threshold `G/delta` plus a union over `G`
  groups gives total failure at most `delta`, uniform over all visit counts.
  Random final counts are substituted only through the already-uniform event;
  the only claim is `P(Emit and violation) <= delta`. Matches the frozen
  contract.
- Root existence/uniqueness: `M(k,q)` continuous, even, strictly increasing
  on `q >= 0`, `M(k,0) <= 1 < G/delta`, divergent as `q` grows — one positive
  root. Sound.
- Stitch validity: per-line Ville with allocation `delta w_j/(2G)`, lower
  envelope valid since crossing it crosses a preregistered line, and
  `cosh(x) >= e^x/2` gives `M(k, q_stitch) >= G/delta`, hence
  `q_mix <= q_stitch`. Stitch is audit/bracket only, never selected. Sound
  and consistent with the frozen formula `q_j(k) = (L_j + a_j^2 k/2)/a_j`.
- Composition section keeps the legacy recurrences (`rho^L B + (1-rho^L)
  r/(2 margin)`, `gamma U_V + r`, `+2B(1-d_min)` for softmax), monotone in
  the radius, with unchanged emission gates.
- Transition variance is recorded as an exact obstruction (no proved uniform
  observable confidence/optimization over `V in [-B,B]^m` including unseen
  successors), receives zero risk, and never enters the certificate —
  satisfying acceptance item 18.
- Sources (Howard-Ramdas-McAuliffe-Sekhon 2020; Hoeffding 1963; Azuma 1967)
  are cited with the correct roles.

## 7. GPT code review

`time_uniform_mixture_certificate.py`:

- `MixtureInversionError` (line 41) carries one of the frozen ordered reasons
  and rejects unknown reasons; `_FAILURE_ORDER` fixes the canonical ordering
  and `_canonical_reasons` sorts by it.
- `log_cosh`/`logsumexp` are overflow-stable and reject nonfinite values.
- `build_mixture_grid` enforces the frozen 15-component grid, `(j+1)^-2`
  weights, `L_j = log(2G/(delta w_j))`, `a_j = sqrt(2 L_j/k_j)`, and rejects
  non-canonical component counts; weights must sum to one within 1e-15.
- `_validated_grid` and `_require_frozen_grid` re-derive the frozen grid and
  reject any caller-supplied deviation (rel/abs 1e-15) — no outcome-tuned
  constants can enter.
- `solve_mixture_boundary` (line 196) brackets with `q_lo = 0`,
  `q_hi = q_stitch(k)`, target `log(G/delta)`; it raises
  `mixture_inversion_unbracketed` unless `log M(k,0) < target <= log M(k,q_hi)`,
  runs at most 200 bisection iterations (and refuses a larger cap), stops only
  when both frozen stopping tests hold (`mixture_inversion_not_converged`
  otherwise), re-checks the conservative bracket before returning
  (`mixture_root_not_conservative`), and returns the conservative upper
  endpoint. No path returns an understated root.
- `_family_summary` converts any inversion failure into `None` radii plus the
  ordered reason; `full_support` additionally requires all counts positive.
  `max_radius` is emitted only under full support.
- `_replace_uniform_radius` / `_replace_vfirst_radius`: any inversion reason
  forces `status = not_certified`, all bound fields `None`,
  `finite_bound_emitted = False`, with merged ordered failure reasons. Legacy
  (visit-indexed) emission decisions are computed first and the time-uniform
  layer never re-emits a rejected route; it only replaces the radius primitive
  on already-certified routes. There is no silent fallback to the old radius:
  the legacy radius appears only in `legacy_radius_by_group` audit fields.
- `build_time_uniform_certificate` records the risk allocation
  (`delta/G` per group, total `delta`, no route-level resplit, no post-hoc
  minimum), marks the stitch as `audit_and_upper_bracket_only`, and isolates
  the variance-adaptive block with `selected: false`, zero delta, and
  `affects_mandatory_status: false`.

`verify_time_uniform_mixture_certificate.py`:

- Contains independent high-precision root checks, the exhaustive all-count
  contract through 16384 against all four frozen horizons (with strictness
  witnessed), builder composition/preservation tests including ordered
  legacy failure reasons (`pair_support_missing`,
  `state_support_missing`, `algorithm_mode_mismatch`), strict-JSON
  serialization with no `NaN`/`Infinity` and no `oracle` key, input-failure
  rejections, and a post-hoc-minimum counterexample
  (`1-(1-delta)^2 > delta`).
- `test_ordered_inversion_nonemission_without_fallback` (line 256) forces an
  invalid upper bracket (stitch replaced by `1e-12`) and asserts
  deterministic non-emission with `mixture_inversion_unbracketed` on every
  family and route, `total_bound is None`, and no fallback — exactly the
  ordered-failure contract of acceptance item 9.

`analyze_time_uniform_certificates.py` verifies the three frozen baseline
hashes before and after analysis, compares legacy leaves with exact type
rules and `math.isclose(rel_tol=1e-12, abs_tol=1e-12)` for numerics, and
enforces the frozen formal emission vector `{256:3, 1024:102, 4096:120,
16384:120}` for every route family in formal mode.

## 8. Anomalies and observations

1. GPT checkout has uncommitted GPT-scope changes (task record, workspace
   index, `verification_of_claude.md`, and two untracked shared theory/report
   documents) at HEAD `f01a49da2bd044562154ea6c9b2014dbee420f4f`. None touch
   the sealed artifacts or the four GPT entry points; recorded for GPT to
   resolve, not a verification failure.
2. Whole-tree Ruff reports 9 pre-existing diagnostics in out-of-scope files
   (`model.py`, `tools/`, `analyze_fixed_policy_finite_sample_certificates.py`);
   all task-scoped files pass Ruff cleanly.
3. The analyzer rewrites `regression.json`/`checks.log` in place; verification
   therefore ran on a temporary copy (now deleted), and the canonical hashes
   were confirmed unchanged afterward.
4. One shell anomaly: an absolute-path `mkdir -p` under the worktree was
   refused by the sandbox (`Permission denied` on `/c/Users/Admin`); the same
   operation via a relative path from inside the worktree succeeded. No effect
   on evidence.

## 9. Evidence paths

- GPT sealed evidence: `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-TU-001\codex\`
  (seven artifacts, hashes above) and
  `icrl_softmax\docs\research_branches\FP-TU-001\codex\{theory,first_result,formal_result}.md`.
- Frozen baseline: `C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex\`.
- GPT code under review: `time_uniform_mixture_certificate.py`,
  `verify_time_uniform_mixture_certificate.py`,
  `analyze_time_uniform_certificates.py` in the GPT checkout.
- This report: `icrl_softmax/docs/research_branches/FP-TU-001/claude/verification_of_codex.md`
  on `claude/FP-TU-001`.

## 10. Interim status

Evidence collection is complete: all six verifiers and task-scoped Ruff pass
on the GPT checkout; baseline and sealed artifact hashes match the frozen
values; 480 records confirmed; compact metrics independently extracted and
digit-identical to GPT's formal result; the analyzer reproduced `PASS` on a
temporary copy that was deleted; theory and code review found the ordered
failure handling and no-silent-fallback contract correctly implemented. The
numbered acceptance assessment and final verdict are reserved for the
completed report.

## 11. Numbered acceptance assessment

Each item is judged at two levels. Route level asks whether GPT's route
(branch `codex/FP-TU-001`, sealed evidence, and GPT-owned documents at HEAD
`f01a49da2bd044562154ea6c9b2014dbee420f4f`) supplies the required evidence.
Project level asks whether the joint task contract is complete; project-level
items involve both routes and the user, and are marked as such rather than as
GPT-route defects.

1. Filtration, stopping-time, measurability, martingale-difference, and
   conditional-width inheritance (`2B`, `B = R_star/(1-gamma)`) are mapped to
   every mixture component without weakening (section 6, first bullet).
   Route level: PASS. Project level: also discharged by the Claude route;
   complete.
2. Cosh mixture gives one event over all `G = m + 2d` groups and all visit
   counts with total failure at most `delta` (Ville at `G/delta`, union over
   `G`, no post-hoc minimum). Route level: PASS (section 6). Project level:
   complete.
3. Random observed counts and selective emission are used exactly; the only
   claim is `P(Emit and violation) <= delta`, with no conditional-support
   claim (sections 6 and 7). Route level: PASS. Project level: complete.
4. Line-stitching boundary is separately proved valid (`cosh(x) >= e^x/2`
   argument), never selected, and brackets every count through 16384; the
   verifier's exhaustive all-count contract and the sealed radius audit
   (`q_mix/q_stitch <= 0.9959405625588716`) confirm it. Route level: PASS.
   Project level: complete.
5. No oracle input enters the certificate; `B` derives only from the declared
   pre-sampling reward bound; sealed artifacts show `oracle_audit`
   structurally separated and `used_as_theorem_evidence: false`. Route
   level: PASS. Project level: complete.
6. Every count `1 <= k <= 16384` passes stable inversion, conservative-root,
   finiteness, `q_mix <= q_stitch`, and monotonicity checks: the verifier's
   exhaustive contract passes and the sealed radius audit reports 16384
   counts checked, maximum 42 bisection iterations, `passed: true`. Route
   level: PASS. Project level: complete.
7. Per-record radius dominance holds: independently extracted maximum
   mixture/legacy ratios are 0.9573573993365391 / 0.9263355819163681 /
   0.8975200574107253 / 0.8793886652750217 for n = 256/1024/4096/16384, all
   strictly below one, matching the frozen pre-review closure values; the
   separate global-`n_max` audit passes in the verifier. Route level: PASS.
   Project level: complete.
8. Composition uses the unchanged deterministic recurrences; sealed
   `summary.json` shows `all_emitted_bounds_nonincreasing: true` and
   positive mean total-bound reductions among emitted records (section 4).
   Route level: PASS. Project level: complete.
9. Ordered deterministic failure handling is implemented and tested:
   `_FAILURE_ORDER` canonical ordering, inversion failures mapped to
   `not_certified` with `None` bounds, and the forced-unbracketed test
   asserting deterministic non-emission with no fallback on every family and
   route (section 7). Route level: PASS. Project level: complete.
10. New fields live only under `time_uniform_certificate`; legacy namespaces
    and values are unchanged, witnessed by `legacy_mismatch_count: 0` and
    exact-type leaf comparison in the analyzer. Route level: PASS. Project
    level: complete.
11. Strict JSON: analyzer enforces strict parsing, the verifier rejects
    `NaN`/`Infinity` and duplicate-key-adjacent constructs, `null` is used
    for unavailable values, and `oracle_audit` is structurally separate with
    no `oracle` key in certificates. Route level: PASS. Project level:
    complete.
12. All five pre-existing verifiers, the new mixture verifier, task-scoped
    Ruff, schema/strict-JSON checks, and baseline-integrity checks pass on
    the GPT checkout (sections 2 and 5); the 9 whole-tree Ruff diagnostics
    are pre-existing, out of scope, and unchanged. Route level: PASS.
    Project level: the same checks pass on the Claude route per its own
    sealed evidence and GPT's reproduction; complete.
13. GPT's smoke matrix passed before its formal run and both GPT seals
    (first-result `00f7d89b`, formal `6f73def5`) precede disclosure per the
    task record. Route level: PASS. Project level: Claude sealed
    analogously; complete.
14. The sealed formal result has exactly 480 records and matches the frozen
    configuration, seed 20260829, and task identity (sections 3 and 4).
    Route level: PASS. Project level: complete.
15. Legacy nonnumeric leaves match exactly and numeric leaves show zero
    mismatch under `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`;
    independently confirmed by the analyzer reproduction (`legacy_mismatch_count:
    0`). Route level: PASS. Project level: complete.
16. Exact-route emissions are exactly 2.5%/85%/100%/100% (3/102/120/120 per
    exact route across lengths); softmax, usefulness, radius-reduction, and
    deviation metrics are reported in the sealed artifacts without tuning
    (section 4). Route level: PASS. Project level: complete.
17. Empirical audit violations are enumerated (zero route and zero residual
    violations in the sealed oracle audit) and `used_as_theorem_evidence:
    false`. Route level: PASS. Project level: complete.
18. The transition-variance study records an exact obstruction (no proved
    uniform observable confidence/optimization over `V in [-B,B]^m`
    including unseen successors), receives zero risk, and never enters the
    certificate. Route level: PASS. Project level: complete.
19. GPT's route records commits, environment, commands, hashes, runs,
    anomalies, limitations, and a numbered assessment in its `theory.md`,
    `first_result.md`, and `formal_result.md`; this report provides the
    reciprocal numbered assessment. Route level: PASS. Project level:
    complete.
20. Reciprocal verification: GPT verifies Claude as `PASS` after repair and
    byte-for-byte reproduction; this report is the Claude verification of
    GPT. Route level (GPT route's obligations): PASS. Project level: two
    items remain transparent user-adjudication points, neither a GPT-route
    defect: (a) the documented duplicate Claude formal evaluation after a
    summary-only repair is a Claude-route history item awaiting final user
    ruling; (b) the GPT checkout carries uncommitted GPT-owned documentation
    (task record, workspace index, `verification_of_claude.md`, and two
    untracked shared documents) that is outside the sealed implementation
    and does not alter any verified evidence — GPT should commit or explain
    it. Final synthesis, workspace-index currency, and `main` unchanged are
    project-level closure steps beyond this report's scope.

## 12. Final verdict

GPT's route is reproducible: every verifier, hash, metric, and analyzer
check independently rerun or re-extracted in this session matches the sealed
evidence to every digit, and theory and code review found no task-level
defect. The two open items are project-level adjudication or housekeeping
points, not GPT-route defects: the duplicate Claude formal evaluation is a
documented Claude-route history item reserved for user ruling, and the
uncommitted GPT-owned documentation does not touch the sealed implementation
or evidence and does not change this verdict. No task-level defect remains.

PASS
