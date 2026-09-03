# FP-MART-001 Claude repair — smoke evidence

Date: 2026-09-03. Branch: `claude/FP-MART-001` (worktree `C:\tmp\research-FP-MART-001-claude`).
This document records the corrected smoke validation of the eight mandated repairs in
`docs/research_branches/FP-MART-001/codex/claude_repair_request.md`. The historical
first-result record (`first_result.md`, commit `c5da243`) is preserved untouched.

## Repairs implemented

1. `theory.md` Section 5: raw MGF iteration replaced by a compensated exponential
   supermartingale `M_t = exp(lambda*S_t - lambda^2 B^2 J_{<t}/2)` with `E[M_n] <= 1`,
   pathwise `sum J <= k`, and the same two-sided constant; a two-step Rademacher
   counterexample (2.281 vs 2.184) documents why the raw iteration is invalid.
2. Optional variance-adaptive conclusion narrowed to: no valid nontrivial observable
   variance proxy was proved or implemented from the frozen allowed inputs (not an
   impossibility theorem); module text mirrors this.
3. True reward tensor removed from certificate inputs: `declared_reward_bound = 1 +
   gap_bonus` is declared before each MDP is sampled, passed into the pure module,
   and `B = declared/(1-gamma)` is derived there. The true full-table max is read
   only after certificate construction, inside the structurally separate
   `oracle_audit.reward_declaration`, which verifies the declaration on every record.
4. Pure-module validation of original integer counts (`operator.index`, bool
   rejected), vector dimensions, sums equal to trajectory length, pair/state
   dimension compatibility (`n_pairs % n_states == 0`), and per-state aggregation
   of pair counts; the counterexample `n=8, [200,200], [100,100,100,100]` is rejected.
5. Smoke seed allocation: every cell always spawns the frozen 30 task seeds and the
   smoke uses leading indices, so all 16 smoke records are baseline keys.
6. Legacy preservation: config, task-record, and 176-row summary regression against
   the frozen baseline `results/fixed_policy_finite_sample_certificates`
   (sha256 a2276eae/c84329bd/49846c08); strict JSON (NaN/Infinity and duplicate-key
   rejection); complete schema/oracle-provenance audit (no true/oracle/occupancy/
   spectral fragments outside `oracle_audit`); per-group residual audit against each
   group's own visit radius; read-only baseline inventory check retained.
7. Analyzer reports both deterministic descriptive thresholds without selecting:
   primary `improves_over_zero_initialization := total_bound < B`, secondary
   `below_two_B_range := total_bound < 2B`, each as among-emitted and overall rates.
8. Statements falsified by GPT verification corrected; this file and
   `repair_result.md` are the new repair evidence.

## Failing-then-passing record

Before the repair, seven new verifier tests failed against the sealed module
(non-original integer counts accepted, counterexample accepted, sum/aggregation
mismatches accepted, incompatible pair dimension accepted, declared-bound derivation
absent, smoke keys not required to be baseline keys). After the repair all 27 tests
in `verify_visit_indexed_martingale_certificate.py` pass, including the mandated
counterexample rejection.

## Commands

```
python -m ruff check visit_indexed_martingale_certificate.py evaluate_visit_indexed_certificates.py analyze_visit_indexed_certificates.py verify_visit_indexed_martingale_certificate.py
python -B verify_visit_indexed_martingale_certificate.py
python -B verify_fixed_policy_q_routes.py
python -B verify_finite_sample_theorems.py
python -B verify_crossfit_markov_certificate.py
python -B verify_theory.py
python -B evaluate_visit_indexed_certificates.py --tasks 2 --trajectory-lengths 256 1024 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8.0 --mixing 0.08 0.5 --gap-bonuses 0.0 0.5 --gamma 0.7 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-MART-001/claude/repair_smoke
python -B analyze_visit_indexed_certificates.py --smoke --baseline-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/fixed_policy_finite_sample_certificates --new-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-MART-001/claude/repair_smoke
```

## Smoke result

- Ruff: all checks passed. Five verifiers: all pass (27-test certificate contract
  included; `verify_theory.py` exit 0).
- Analyzer smoke output: `compared_records 16`, `new_only_keys 0`,
  `mismatch_count 0`, `duplicate_record_keys 0`,
  `max_common_numeric_abs_difference 1.0459e-11`; final checks all true
  (baseline hash check, config/task-record/summary regression, schema/oracle
  provenance audit) and `PASS visit-indexed certificate analysis`.
- All 16 records: `reward_declaration.declaration_verified == True`; per-group
  residual audit `violation_count == 0` over 48 visited groups per record.

## Known anomaly: 1-ulp spectral-path difference

Twenty legacy leaf paths differ from the frozen baseline by at most 1.05e-11, all
within the mandated regression tolerance (rel_tol = abs_tol = 1e-12, hence zero
mismatches). Every differing leaf sits in one dependency chain of the untouched
legacy certificate path: `certificate/edge/right_hoeffding_inflation` ->
`finite_sample/shared_event/omega_E` -> state-value `statistical_term`/`total_bound`
(exact and softmax) and the matching `vfirst_nosplit_*` leaves. The effect is
deterministic across reruns of the repaired build, the frozen seed/trajectory
identity is provably unchanged (`SeedSequence.spawn` child keys depend only on the
child index), and the legacy functions themselves are unmodified; a probe running
the sealed code in this same environment reproduces the baseline bit-identically.
The residual 1-ulp difference in the LAPACK `eigvalsh`-derived spectral inflation
could not be pinned to a source-code path and is recorded here as an open anomaly.
It does not affect any gate (zero mismatches) and does not weaken certificates
(certificate radii use the declared bound, which is conservative).

## Raw-output hashes (repair_smoke)

- config.json: 67fe72e3d210890d1429091fcbe4a43de0593e39025e2c0b7f387de1c3ef13bf
- environment.json: e4abe4ceb688d348cc715c17b1c0db89cefa88cad9cc4d8f7bb77fd5ba18b270
- task_results.json: 27faedf6286b4f756cb869180c661a4c4bf176e3717ae4c86df9970b8a9236ee
- commands.log: 414dc190fa92878e73fe79d646f961a4cb311bdfdc1f0e72f5db6b0b85d350d1

## Limitations

- Smoke covers 16 of 480 records (leading task seeds, lengths 256/1024 only).
- The variance-adaptive constituent remains deterministically unavailable; no
  post-hoc minimum with the Hoeffding radius is ever formed.
- Per-group residual audit is an empirical diagnostic, not a coverage guarantee.
