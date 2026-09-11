# FP-ESARSA-001 — Claude route failure and repair history

This file records all known failures and repairs on the Claude main route.
Nothing here is erased; older reports remain as historical records with a
dated correction note at the top.

## Current status (2026-09-11)

### P2 tests-first: expected import failure (recorded, not a defect)

Per the task plan, `verify_fixed_policy_expected_sarsa.py` was written
*before* its implementation module. The first execution is therefore
expected to fail at import, and that failure is the tests-first evidence.

```
$ C:/Users/Admin/anaconda3/python.exe -B verify_fixed_policy_expected_sarsa.py
Traceback (most recent call last):
  File ".../verify_fixed_policy_expected_sarsa.py", line 44, in <module>
    import fixed_policy_expected_sarsa as es
ModuleNotFoundError: No module named 'fixed_policy_expected_sarsa'
```

This confirms the verifier exercises the implementation module and the three
task-scoped `model.py` classes rather than passing vacuously. The verifier is
retained unchanged as the executable contract; implementation then proceeds
until it reports `PASS`.

### P3 implementation: repairs found by the tests-first loop

Running the verifier against the first implementation exposed one
implementation defect and three verifier (contract) defects. All four were
repaired before any seal; the final verifier reports
`PASS fixed-policy Expected SARSA verification (13550 checks)`.

1. **Verifier fixture constant (contract defect).** S5's finite-route layer-1
   check wrote the layer-0 residual total as `1.25` (which happens to equal
   `sums0[0]`) instead of the true total `1.5`. The implementation matched the
   formula at the correct total. Repaired to use the residual total
   (`FIXTURE_REWARDS.sum()`), mirroring the standalone write-back check.
2. **Verifier eta-order budget (contract defect).** S11's excess-risk search
   used `e_q = 0.02 * gap`, under which the largest eta passes essentially
   always (empirically ~1 case in 4000), so the descending first-passage path
   was never exercised. Repaired to `0.1 * gap` (~1.8% hit rate), which does
   exercise the order check.
3. **Abstention "unchanged" test (implementation defect).** With `Q = 0` the
   candidate equals the input policy only up to floating-point round-off of
   `exp(log p)`, so a bit-exact comparison missed the `policy_unchanged` reason.
   Repaired the implementation to compare with a `1e-12` tolerance while still
   returning the input policy bit-for-bit on abstention.
4. **Verifier masked write-weight constant (contract defect).** S13 expected a
   flat `0.5` weight for every visited query, but the masked write is
   normalised per query, so a once-visited pair carries weight `1.0`. The
   masked route output already matched `EXPECTED_Q1`; repaired the expectation
   to `1 / len(matched rows)`.

### P3 status

- `fixed_policy_expected_sarsa.py` (pure-numpy reference: canonical memory,
  exact/finite/sampled routes, residual certificate, relative-softmax decision
  rule, strict JSON) and the three task-scoped `model.py` classes
  (`FixedPolicyActionExpectation`, `EndToEndMaskedSoftmaxExpectedSARSA`,
  `EndToEndFiniteSoftmaxExpectedSARSA`) are in place.
- `verify_fixed_policy_expected_sarsa.py` passes all 14 sections, 13550 checks.
  S8 (Monte Carlo, seed 20260911, 400 trials) reports 0 residual-event
  violations and 0 Q-certificate violations, worst `error / E_Q = 0.0120`.

### P4 evaluation: repairs found while writing the evaluator

The frozen evaluator `evaluate_fixed_policy_expected_sarsa.py` and analyzer
`analyze_fixed_policy_expected_sarsa.py` were then written and exercised on
labelled smoke subsets before the single formal run. Repairs made during P4:

1. **Ruff F401 (`math` unused).** Removed the unused import in the evaluator.
2. **Dead declared-reward-bound branch.** The first draft carried a
   no-op `if` on the per-cell declared reward bound. The frozen protocol uses
   a single `R_star=1.5` for the whole matrix, so the branch was deleted and
   `reward_bound` fixed to `1.5`.
3. **Analyzer duplicate-key rejection.** Added `object_pairs_hook` duplicate
   rejection to the analyzer's strict loader to cover acceptance criterion 15
   (strict-JSON integrity) on the sealed artifacts.

No implementation or contract defect was found during P4; the smoke and
formal runs both reproduced the frozen FP-TU-001 generator identities with
`0/480` mismatches.

### P4 status

- `evaluate_fixed_policy_expected_sarsa.py`, `analyze_fixed_policy_expected_sarsa.py`
  in place; `ruff` clean.
- Labelled smoke into `results/FP-ESARSA-001/claude/smoke{,_long}`; the blind
  smoke seal is `first_result.md`.
- Exactly one frozen 480-record formal run into
  `results/FP-ESARSA-001/claude/`: `283/480` certificates per route,
  `0/480` safe updates (hypothesis-8 verified negative), `0` oracle
  certificate/residual/value violations. Sealed evidence and hashes in
  `formal_result.md`; acceptance-criteria mapping in `report.md`.

## Provenance notes

- The Claude main route runs on branch `claude/FP-ESARSA-001`, created from
  activation commit `e2859f2`. The branch was pushed and fast-forwarded into
  `main` by explicit user authorization on 2026-09-11.
- Dual blind construction was waived by explicit user exception
  ("直接例外"): Claude main execution plus author seal. The independent GPT
  post-seal acceptance never occurred (Codex quota-blocked).
- Closure (2026-09-11): the user chose the explicit-exemption path written
  into acceptance criterion 20. The task closed `VERIFIED` on this route's
  single-side evidence alone, with no independent executable reconstruction.
  See the acceptance-exemption ruling in `docs/research_tasks/FP-ESARSA-001.md`.
