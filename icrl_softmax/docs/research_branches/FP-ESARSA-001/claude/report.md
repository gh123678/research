# FP-ESARSA-001 — Claude main route report

Route: `claude/FP-ESARSA-001` from activation commit `e2859f2`. This is the
Claude main-route report for the FP-ESARSA-001 task under the v1.1
responsibility exception (Claude main execution + author seal, GPT independent
post-seal acceptance, Claude reciprocal review). Preliminary until GPT
acceptance or an explicit user exemption.

## What was built

| artifact | role |
|---|---|
| `fixed_policy_expected_sarsa.py` | pure-numpy reference: canonical memory, exact/finite/sampled routes, residual certificate, relative-softmax decision, strict JSON |
| `model.py` (task-scoped additions) | `FixedPolicyActionExpectation`, `EndToEndMaskedSoftmaxExpectedSARSA`, `EndToEndFiniteSoftmaxExpectedSARSA` (literal torch attention) |
| `verify_fixed_policy_expected_sarsa.py` | executable contract, 14 sections, 13550 checks |
| `evaluate_fixed_policy_expected_sarsa.py` | frozen 480-record evaluator with split isolation, fresh-dir refusal, generator-identity regression, oracle separation |
| `analyze_fixed_policy_expected_sarsa.py` | model-free aggregator, strict JSON reload |
| `docs/research_branches/fixed_policy_expected_sarsa_theory.md` | consolidated 12-section theory |
| `docs/research_branches/fixed_policy_expected_sarsa_report.md` | main-route scientific report |
| `first_result.md`, `formal_result.md`, `failure_history.md` | route journal and sealed evidence |

## Process

`theory → tests-first → implementation → smoke → seal → formal`, as announced.
The tests-first loop wrote the verifier before the implementation; the
expected initial `ModuleNotFoundError` is recorded in `failure_history.md`,
along with the one implementation defect and three contract defects the loop
found and repaired. The frozen formal run was executed exactly once into
`results/FP-ESARSA-001/claude/`.

## Formal outcome

480 matched records, three routes each, generator identity `0/480` mismatches
against the frozen FP-TU-001 baseline. Certificates emitted on `283/480`
records per route; oracle certificate violations, residual-event violations,
and value decreases are all **0**. **No route emitted a safe
relative-softmax update**, so hypothesis 8 is a verified negative usefulness
result; no frozen parameter was retuned.

## Acceptance criteria (task sheet, 20 items)

1. Canonical memory, one token per pair, duplicate rejection — **PASS** (S1).
2. Exact head/read/residual/writeback equal batch Expected SARSA at `1e-12` — **PASS** (S2, S3, S13).
3. Self-loop uses one Q entry in two branches without duplication — **PASS** (S3).
4. Repeated-visit exact residual mean; unvisited exact query unchanged — **PASS** (S2).
5. Finite successor/read/writeback equal direct finite formulas; no mask/gate — **PASS** (S5, S13).
6. Kernel fixed point and diagonal-margin bound proved and verified on direct matrices; finite route claims no no-bias — **PASS** (S6, theory §4).
7. Martingale filtration, `2B` scale, random counts, union over `d` groups — **PASS** (theory §5–6, S7, S8).
8. Emitted `E_Q` equals the frozen formula, full held-out support, bounds oracle error — **PASS** (S9; formal 0 violations).
9. Exact relative-softmax improvement identity proved and exhaustively checked — **PASS** (S11, theory §7).
10. Lower bound valid at adversarial error corners; emitted LBs nonnegative — **PASS** (S11).
11. Emitted policies finite, positive, row-normalized, changed, non-degrading — **PASS** (S11 on fixtures; formal emitted none, so the claim holds vacuously and 0 oracle value decreases).
12. Abstention returns the policy bit-for-bit with the frozen reason list — **PASS** (S11; formal reasons only `heldout_pair_support_missing`, `improvement_lcb_nonpositive`).
13. Training/held-out structural separation — **PASS** (split assertion in the evaluator, S10 leak counterexample).
14. Truth confined to `oracle_audit` — **PASS** (record schema; S14 source boundary).
15. Strict-JSON, duplicate-key, nonfinite, shape, range, identity, branch, seed, config, result-location checks — **PASS** (S12; analyzer strict reload; `regression.json`).
16. Exactly 480 matched records, three routes, frozen identities and seed schedule — **PASS** (formal run; `regression.json` 0 mismatches).
17. Empirical violations and premise failures enumerated, excluded from theorem evidence — **PASS** (0 certificate/residual/value violations; 210 finite-route premise failures enumerated).
18. Hypothesis 8 without retuning; zero emissions reported as verified negative — **PASS**.
19. Sealed evidence before mutual disclosure — **PASS** for the Claude route (this seal).
20. GPT accepts Claude and Claude accepts GPT — **PENDING** (GPT quota-blocked; results preliminary).

## Verdict

`PASS` for the Claude main route (criteria 1–19); the task remains
unverified pending criterion 20 (GPT post-seal acceptance) or an explicit
user exemption.
