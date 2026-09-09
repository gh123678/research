# FP-KERN-001 Claude verification of the Codex route

Date: 2026-09-09. Verifier branch: `claude/FP-KERN-001` at Claude formal seal
`1a820467683d137e6527edbd99bb486000b2fc58`. Verified object: Codex route on
`codex/FP-KERN-001` at formal seal
`640a3f8fb2d0f41b96eef8d9bb76fc5ef2e9b93b`, worktree
`results/FP-KERN-001/codex_worktree/icrl_softmax`, formal outputs
`results/FP-KERN-001/codex/`.

Both blind formal seals existed before this verification began; the
nondisclosure period is over and reciprocal verification is the authorized
step. This was read-only verification: no Codex file was modified, the formal
evaluator was not rerun, the strict analyzer ran without `--write-results`,
and all artifact hashes were confirmed unchanged afterwards.

## 1. Provenance and isolation

- Codex worktree `git status` clean, `HEAD` = `640a3f8f...` (the seal), branch
  `codex/FP-KERN-001`.
- `git diff --stat 28c4ae0..640a3f8`: exactly eight added paths, zero
  modifications to pre-existing files — the five authorized task files plus
  `docs/research_branches/FP-KERN-001/codex/{theory,first_result,formal_result}.md`.
  No existing evaluator, verifier, MDP helper, old result, or task/design file
  was touched.
- Common route execution-start commit recorded identically in both routes:
  `28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c` (Codex `config.json`,
  `environment.json`, evaluator constant; Claude formal seal).
- Codex blind first-result seal `3f57ce5` (implementation + smoke) precedes
  the formal seal `640a3f8`; smoke artifacts were moved unchanged to
  `results/FP-KERN-001/codex_smoke/` so the canonical formal directory was
  empty before the single formal run (`_prepare_output` refuses a nonempty
  directory; `commands.log` records exactly one formal command;
  `checks.log` records one evaluator PASS and one analyzer PASS).

## 2. Hash verification (recomputed by Claude)

All seven formal artifact SHA-256 hashes recomputed from disk match
`formal_result.md` exactly (`analysis.json`, `checks.log`, `commands.log`,
`config.json`, `environment.json`, `summary.json`, `task_results.json`).
After the read-only analyzer run below, all seven were recomputed again and
remained identical — the analyzer did not mutate frozen artifacts.

As supplementary evidence, all seven smoke artifact hashes in
`results/FP-KERN-001/codex_smoke/` also match `first_result.md`, and the smoke
matrix contains exactly 16 records.

## 3. Executable verification (all run in the Codex worktree, `python -B`)

- `verify_kernel_state_generalization.py`: PASS ("kernel state-generalization
  checks passed"). Covers target-action exclusion, zero-count recovery,
  anchor abstention, median/ESS arithmetic, weight ordering, self-only
  reduction, permutation equivariance, oracle-key rejection, malformed
  inputs, degenerate bandwidth, and hidden-cluster generator determinism,
  balance, simplex, and reward bounds.
- `verify_fixed_policy_q_routes.py`: PASS.
- `verify_finite_sample_theorems.py`: PASS.
- `verify_visit_indexed_martingale_certificate.py`: PASS.
- `verify_time_uniform_mixture_certificate.py`: PASS.
- `ruff check` on the five task files: PASS ("All checks passed!").
- `analyze_kernel_state_generalization.py --result-dir results/FP-KERN-001/codex`
  (no `--write-results`): PASS — "strict analysis with 480 records;
  classification=NOT_SUPPORTED". The analyzer independently regenerated every
  record from the serialized seed components (MDP/policy, three trajectories,
  `V_hat`, observable aggregates, all four routes, oracle audit) and asserted
  nested equality at 1e-12 against the sealed record; any mismatch would have
  raised. Its internal before/after hash guard also passed.

## 4. Contract inspection of the five Codex files

Read in full: `kernel_state_generalization.py`,
`kernel_generalization_mdps.py`, `verify_kernel_state_generalization.py`,
`evaluate_kernel_state_generalization.py`,
`analyze_kernel_state_generalization.py`.

- **Frozen formulas**: `q_sig` conditional mean, common support
  `|C_a(s,s')| >= 2` with target action excluded, normalized distance
  `mean[((q_sig-q_sig')/(2B))^2]` with `B = R_star/(1-gamma)`, median of
  finite positive eligible distances (sorted float64, odd central value /
  even mean-of-two-centrals = `numpy.median` semantics), Gaussian kernel
  `exp(-d^2/(2 h^2))`, pooled estimate `sum K*Y_sum / sum K*N`, and
  observation-weighted ESS `(sum K*N)^2 / sum K^2*N` all match the frozen
  text exactly.
- **Routes**: all four frozen routes with the frozen names; local unpooled
  unavailable at zero count; pool count-weighted across states; anchor
  restricted to positive target count and never claiming zero coverage (523
  zero-count pairs all abstain); primary is the leave-one-action-out route.
- **RNG isolation**: per record, `SeedSequence([seed, family, length,
  mixing, gap, task]).spawn(5)` yields separate deterministic streams for
  MDP, policy, value, signature, and target; `seed_components` are serialized
  per record and replayed exactly by the analyzer. Three independent
  stationary-start trajectories per record as frozen.
- **Oracle boundary**: the pure module accepts only the seven declared
  observable keys and rejects prohibited fragments (`true`, `oracle`,
  `cluster`, `occupancy`, `stationary`, `exact_return`, `realized_error`,
  `kernel_matrix`); it imports only numpy. The evaluator attaches
  `oracle_audit` (true V/Q, returns, hidden generator fields) only after
  route outputs are built; the analyzer re-derives and compares both sides.
- **Matrix and thresholds**: `config.json` matches the frozen matrix (2
  families, 15 tasks, lengths 256/1024/4096/16384, mixing 0.08/0.50, gaps
  0/0.50, 6 states, 4 actions, pi_min 0.05, gamma 0.70, alpha 0.65, 160
  iterations, seed 20260909, count bins 0/1-4/5-16/17+, structure strength
  0.90/0.10); the evaluator hard-validates every frozen CLI value. 480
  records exactly.
- **Screen and decision rule**: the analyzer implements the frozen five-item
  screen per family (50% signature-eligible zero-count coverage; >=10%
  zero-count RMSE reduction vs pool with paired 95% Student-t interval
  excluding zero; >=10% 1-4 RMSE reduction vs local with interval; >=5pp
  sparse-state top-action gain with interval; false-improvement within 1pp),
  empty-bin exclusion counting, n<2 intervals unavailable-and-nonpassing,
  coverage over all target pairs, paired comparisons on common finite pairs,
  and the frozen four-way classification
  (`NOT_SUPPORTED` iff the hidden-cluster screen fails).
- **Stopping rules**: exactly one formal run, smoke sealed before the formal
  run, no tuning after any output; no rerun, fallback, or selective
  denominator found anywhere in the code or logs.

## 5. Route comparison and numerical differences

The two routes used different (both deterministic, nonoverlapping,
seed-recorded) RNG substream derivations, so the 480 MDP realizations differ
record-by-record; the task freezes the seed and matrix, not a bit-identical
stream scheme, so comparison is at the level of contract and conclusion.

Headline screen outcomes agree on every item in both families:

| Item | Codex current | Claude current | Codex hidden | Claude hidden |
|---|---|---|---|---|
| 1 coverage >= 50% | 100% (183/183) pass | 100% (227/227) pass | 100% (235/235) pass | 99.56% (224/225) pass |
| 2 zero-RMSE impr. | +7.25% fail (<10%) | +4.50% fail | +1.47% fail | +7.58% fail |
| 3 1-4 RMSE impr. | -154.4% fail | -112.5% fail | -121.5% fail | -119.8% fail |
| 4 top-action | +2.12pp fail | +4.41pp fail | +2.65pp fail | +4.56pp fail |
| 5 false-impr. | 38.13% vs 39.58% pass | 18.75% vs 19.79% pass | 34.39% vs 35.99% pass | 17.99% vs 19.48% pass |
| Screen | FAIL | FAIL | FAIL | FAIL |

Both routes classify `NOT_SUPPORTED`; both secondary Spearman diagnostics are
positive with intervals excluding zero (Codex 0.217/0.236, Claude
0.283/0.307). The failing margins are decisive (items 2 and 4 below
threshold, item 3 catastrophically negative), so no plausible reconstruction
difference can change the classification.

Four interpretation differences were identified and quantified against the
sealed records:

1. **Self-pair support gating (primary route)**. Codex applies the
   two-common-action support rule to the self pair, so the target state gets
   weight one only when its own leave-one-action-out signature has at least
   two visited non-target actions; Claude assigns the self pair weight one
   unconditionally ("participates with weight one when it has
   observations"). Consequence: 229 of 11520 pairs (2.0%; 223 at length 256,
   6 at 1024; all with finite record bandwidth) abstain
   (`insufficient_common_actions`) under Codex but would emit the
   local-mean-equivalent self-only estimate under Claude. Zero-count handling
   is identical in both routes (a zero-count pair is eligible only with an
   eligible *other* state in both). Impact: Claude's extra emitted pairs have
   primary error exactly equal to the local error, diluting item-3 paired
   differences slightly toward zero — consistent with the observed
   (-112.5%/-119.8% vs -154.4%/-121.5%) — and adding complete rows to item 4.
   No screen item or the classification can flip under either reading; the
   frozen text is genuinely ambiguous here and the difference should be
   recorded in the synthesis.
2. **False-improvement denominator convention**. Claude counts ordered
   action pairs and only the literal frozen event "estimated positive
   difference, true difference nonpositive"; Codex counts unordered pairs and
   sign disagreement in either direction. The per-pair numerator events
   coincide, so Codex's rates are about twice Claude's (observed ~2x). Item 5
   is a relative comparison of primary vs pool inside each route; both routes
   apply their convention consistently to both compared routes and both pass
   by wide margins. Absolute rates are not comparable across routes; no
   validity impact.
3. **Abstention-reason labeling**. Codex maps a zero/nonpositive kernel
   denominator to `target_source_unavailable` (this path never actually
   occurred in the sealed data — primary reasons are only `ok` and
   `insufficient_common_actions`) and checks the anchor zero-target-count
   condition before the signature-support condition, so a zero-count pair
   without signature support is labeled `target_source_unavailable` where the
   frozen reason ordering lists `insufficient_common_actions` earlier. Claude
   reserves `kernel_denominator_invalid` for denominator failure and follows
   the strict reason order for the anchor. All these reasons are ordinary
   coverage abstentions with identical eligibility/coverage numerics; only
   the labels differ. Minor contract-fidelity nit, no numerical effect.
4. **Spearman granularity**. Codex pools eligible distances across actions
   within a record (240 correlations per family); Claude computes per
   record/action (960 per family). Clarification 5 supports "within each
   record/action" for computation and "per-record" averaging; both readings
   are defensible, the diagnostic is secondary (not a screen item), and both
   routes find a significant positive correlation.

Observable-interface differences (Codex passes precomputed `signature_q`
means into the pure module; Claude passes raw sums and computes means
internally) are within the authorized freedom of independent implementation;
both derive everything from declared observables only.

## 6. Acceptance-criteria assessment (Codex route)

1. Frozen routes/formulas/support/bandwidth/families: verified by inspection
   and the passing contract verifier. 2. Observable-only inputs, oracle
   denied at the pure boundary: verified (input validation + reject tests +
   structural separation in the evaluator). 3. Independent deterministic
   streams, reproducible from recorded seeds: verified by full analyzer
   reconstruction. 4. Target action excluded from the primary signature with
   two-common-action gating: verified (code and exclusion test). 5. Finite,
   reconstructible weights/bandwidths/counts/denominators/ESS/estimates:
   verified by 480-record exact reconstruction. 6. Zero-count primary
   estimates use only other-state observations; anchor never emits at zero
   count: verified (code + 523/523 zero-count anchor abstentions). 7.
   State-label permutation equivariance: verifier test passes. 8. Hidden
   structure never influences routes: structural separation confirmed.
   9. Ordered abstention contract without fallback: confirmed, with the
   reason-label nit of section 5.3 (immaterial). 10. Smoke (16 records)
   passed and sealed before the formal run; hashes match. 11. Exactly 480
   records with frozen identity: confirmed by config validation, analyzer
   count check, and record-index check. 12. All denominators, comparable
   sets, coverage, errors, orderings, policy diagnostics, and intervals
   independently reconstructed: analyzer PASS with zero mismatches.
   13. Classification follows the frozen screen: `NOT_SUPPORTED` from the
   hidden-cluster failure, correctly not `INVALID`. 14. New verifier, four
   inherited verifiers, strict JSON (duplicate-key and nonfinite rejection),
   provenance, hash, and Ruff checks: all pass under my rerun. 15. Commands,
   environment, anomalies, hashes, limitations, and acceptance judgment are
   recorded in `first_result.md`/`formal_result.md`. 16. One common
   execution-start commit, blind seals before disclosure: git history
   confirms. 17-18: reciprocal verification (this report) and workspace
   update remain governed by the user.

## 7. Conclusion

The Codex route is mechanically sound, faithful to the frozen contract in all
validity-relevant respects, fully reproducible from its sealed artifacts
(reconstructed independently and exactly, without modifying them), and its
scientific conclusion `NOT_SUPPORTED` agrees with the independent Claude
route on every screen item in both families. All observed numerical
differences trace to authorized independent stream derivation plus four
documented interpretation differences, none of which can change any screen
item or the classification.

PASS
