# FP-ADV-001 Claude route: blind first result

Status: sealed before the single formal 480-record run.
Author: Claude Code, branch `claude/FP-ADV-001`, isolated worktree.
Scope of this record: independent theory, implementation, verification, and
smoke evidence. No GPT FP-ADV-001 material was read. This file is written and
committed before the formal evaluator is run, exactly once.

## 1. Environment

- Windows 11 Pro 10.0.22631; bash shell; CPU-only.
- Python `C:\Users\Admin\anaconda3\python.exe` 3.13.9; numpy 2.4.6;
  scipy 1.16.3; matplotlib 3.10.6.
- Git HEAD at smoke runs: `4078f6911cbfb4654205772685f49896e4e8cad2`
  (common route execution-start commit; the four new files and this document
  are added by the seal commit).
- Baselines verified read-only by SHA-256 before and after every run:
  - FP-TU-001 `results/FP-TU-001/codex`: config `43dcb96b…`, task_results
    `0e5eab39…`, summary `565fc4d2…` (480 records).
  - FP-MART-001 `results/FP-MART-001/codex`: config `bcc377b4…`,
    task_results `929e2f65…`, summary `fa136197…`.

## 2. Commands (all from the worktree `icrl_softmax/` directory)

Preflight verifiers (run inside the evaluator before any record generation;
full logs in each smoke `checks.log`):

```text
python -B verify_finite_sample_theorems.py            -> PASS
python -B verify_fixed_policy_q_routes.py             -> PASS
python -B verify_crossfit_markov_certificate.py       -> PASS
python -B verify_end_to_end_sarsa.py                  -> PASS
python -B verify_visit_indexed_martingale_certificate.py -> PASS
python -B verify_time_uniform_mixture_certificate.py  -> PASS
python -B verify_action_gap_certificate.py            -> PASS (12 tests)
```

Task-scoped lint:

```text
python -m ruff check action_gap_certificate.py verify_action_gap_certificate.py \
  evaluate_action_gap_certificates.py analyze_action_gap_certificates.py
  -> All checks passed!
```

Smoke runs (evaluation plus strict analysis, artifacts under
`results/FP-ADV-001/claude/`, which is git-ignored):

```text
python -B evaluate_action_gap_certificates.py --tasks 1 --trajectory-lengths 256 \
  --output-dir ...\claude\smoke_tiny     -> PASS, 4 records
python -B analyze_action_gap_certificates.py --result-dir ...\claude\smoke_tiny
  -> PASS, matched-subset legacy regression mismatches=0
python -B evaluate_action_gap_certificates.py --tasks 3 \
  --trajectory-lengths 256 16384 --mixing 0.08 0.5 --gap-bonuses 0 0.5 \
  --output-dir ...\claude\smoke_matrix   -> PASS, 24 records
python -B analyze_action_gap_certificates.py --result-dir ...\claude\smoke_matrix
  -> PASS, matched-subset legacy regression mismatches=0
```

## 3. Smoke findings (not theorem evidence)

- 28 smoke records (4 + 24). Zero update emissions on all six routes.
- Abstention reasons are honest: at length 256 global controls report
  `recovery_radius_unavailable` (complete-Q bounds need full pair support) and
  local routes report `candidate_pair_unvisited` for unvisited receivers; at
  length 16384 all comparisons reach `gap_lcb_nonpositive` because the
  uncertainties (roughly 2.7-4.4 for local routes) exceed the estimated action
  differences (roughly 0.5-0.7) even with gap bonus 0.5. The certificates are
  conservative; the frozen contract permits a zero-update outcome.
- Dominance rechecks passed on every evaluated pair (216 exact + 216 softmax
  checked pairs in the 24-record matrix; analyzer recomputes every gate,
  uncertainty, LCB, transfer, and policy row from serialized observables).
- Oracle audit (truth-only, structurally separated): no false ordering, no
  bound violation, no Bellman violation, no value decrease — vacuously, since
  no update was emitted in smoke.
- Smoke records match the frozen FP-TU-001 baseline on the matched subset
  (same seed entropy, spawn key, length, task index) at
  `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`; nonnumeric leaves match
  exactly.

## 4. Acceptance-criteria assessment (Claude route, pre-formal)

1. PASS — theory section 1 maps (A1)-(A8) onto the verified FP-MART-001 /
   FP-TU-001 event; no new risk is spent, random counts substitute into the
   uniform-in-k event, and only the selective joint statement is claimed.
2. PASS — theory sections 2-3 prove the exact decomposition and TV/span
   bound; verifier tests 1, 2, 5 check the lemma, the identity, and 300
   random validity fixtures.
3. PASS — theory section 4 derives the one-hot softmax weights, kappa,
   effective row, and contamination; verifier tests 3-5 and 10 check the
   weights, rows, adversarial contamination, and route gates.
4. PASS — the pure module consumes only counts, histograms, estimates,
   policy, hyperparameters, and emitted bounds; `assert_no_oracle_keys`
   rejects prohibited keys (verifier test 12); `B = reward_bound/(1-gamma)`
   from the predeclared bound only.
5. PASS — theory section 6; every used quantity is a deterministic
   consequence of the single simultaneous event; no conditional-coverage or
   support claim is made.
6. PASS — donor gates require both counts positive, LCB > 0, and
   `pi(b|s) > pi_min` before eligibility (analyzer replays every gate).
7. PASS — row sums preserved exactly, floor strict (`theta in (0,1)`), values
   finite; verifier test 7 and the analyzer policy-row audit.
8. PASS — theory section 7 proves nonnegativity of the Bellman lower bound
   and the contraction implication; verifier test 8 checks the pointwise
   implication on 20 random MDPs with at least one emission.
9. PASS — theory section 5 proves U_exact <= 2E_Q^exact and
   U_soft <= 2E_Q^soft; the evaluator hard-fails on violation and the
   analyzer rechecks every pair.
10. PASS — only the two compared pairs need positive counts; unrelated
    missing pairs never enter; missing compared pairs give the ordered
    abstention `candidate_pair_unvisited` (verifier test 6).
11. PASS — Direct-Q controls are global-only with penalty `2 E_Q`; the pure
    module rejects `family="direct", scope="local"`.
12. PASS — new fields live only under `action_gap_certificate`; smoke
    matched-subset regression shows zero legacy mismatches.
13. PASS — strict JSON loading rejects duplicate keys and nonfinite
    constants; `null` marks unavailable values; `oracle_audit` is the only
    truth-based subtree and is excluded from certificate inputs.
14. PASS (this route) — all inherited verifiers, the new 12-test verifier,
    task-scoped Ruff, and frozen-hash checks pass; see section 2.
15. PASS (this route) — both smoke matrices pass before formal evaluation;
    this seal precedes the single formal run.
16. PENDING — the formal 480-record run happens exactly once after this
    seal; its evidence lands in `formal_result.md`.
17. PASS (smoke subset) / PENDING (full 480) — matched smoke leaves match
    the FP-TU-001 baseline exactly (nonnumeric) and at 1e-12 (numeric);
    the full-matrix regression runs on the formal output.
18. PASS — the analyzer enumerates every empirical false ordering, bound
    violation, Bellman violation, value decrease, and return change under
    `oracle_audit`, marked not theorem evidence.
19. PASS (this route) — commits, environment, commands, hashes, smoke
    outcomes, anomalies (none), and limitations are recorded here and in
    `formal_result.md`.
20. PENDING — reciprocal verification belongs to the post-formal phase; this
    route has not accessed the GPT route.

## 5. Limitations

- One fixed policy, one trajectory, one update; no online or convergence
  claim; the validity statement is exactly the inherited selective one.
- Smoke emitted no updates: the local uncertainties remain several times
  larger than the true action gaps at these scales, so honest abstention is
  the expected outcome in many cells. Hypothesis 8 (empirical usefulness)
  may fail without invalidating the theory.
- The exact local TV factor is informative only when the two observed
  successor rows overlap; nothing is claimed otherwise.
- Oracle audits are diagnostics over the realized truth and never enter any
  certificate input.
