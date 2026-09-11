# FP-EXPL-001 (v1.1) — Claude reciprocal review of the GPT verification report

> **Current status note (2026-09-11):** this review refers to GPT's earlier
> pre-seal preflight execution and is **superseded/pending**. After GPT's
> source review found the one-step network probe boundary error, the Claude
> route was repaired and rerun (see `failure_history.md` and `report.md`).
> GPT will formally replay the repaired route and send a corrected
> acceptance; Claude's executable reciprocal review of that corrected
> acceptance is still to be performed. The text below is retained unchanged
> as the historical record of the earlier review.

This document is the Claude-side reciprocal review of
`docs/research_branches/FP-EXPL-001/codex/verification_of_other.md` (the GPT
verification of the Claude route). Per the review constraints, no Bash was
run, nothing was committed, and no scientific source or protocol file was
changed. This review is based exclusively on reading the following files:

- `docs/research_branches/FP-EXPL-001/codex/verification_of_other.md` (report under review)
- `docs/research_branches/FP-EXPL-001/codex/verify_claude.py` (GPT acceptance reference)
- `docs/research_branches/FP-EXPL-001/claude/witness.py` (Claude witness, in this worktree)
- `docs/research_branches/FP-EXPL-001/claude/verify.py` (Claude independent verifier, in this worktree)
- `results/FP-EXPL-001/claude/verification.json` (Claude verifier output)
- `results/FP-EXPL-001/codex/verification.json` (GPT acceptance output)

No command was executed by this reviewer; all judgments below are file-level
consistency checks between the report, the two scripts, and the two sealed
JSON artifacts. No command output is invented here.

## 1. Frozen inputs

The GPT acceptance reference (`verify_claude.py`) hard-codes
`GAMMA = 0.7`, `ALPHA = 0.5`, `SHARPNESS = 8.0`, target policy
`[[0.75, 0.25], [0.25, 0.75]]`, behavior threshold `0.5` (via
`ua < 0.5`), transition matrix
`[[0.75, 0.25], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]`, rewards
`[1.0, -0.5, 0.25, 0.25]` in pair order `00, 01, 10, 11`, sampler
`np.random.Generator(np.random.PCG64(20260911))`, 64 transitions from
initial state 0, exactly two scalar `rng.random()` draws per transition in
the order (action, then transition), and baseline commit
`8c915c4bf2bb9533e2374f5d2c91cf34e5c13c77`. Every one of these constants
matches the frozen inputs declared in `witness.py` and retyped in
`verify.py`, and the baseline matches the recorded baseline commit
(`8c915c4`). The report's claim that the independent reference reconstructs
the batch "from the task specification" is consistent with the code.

## 2. Independent replay claims

`verify_claude.py` imports only `argparse`, `json`, `pathlib`, `fractions`,
and `numpy`; it does not import any Claude module. The reconstruction in
`reconstruct()` was checked line by line against the Claude witness
formulas:

- `C0`, `S0`, `W0` match `exact_matrices` (indicator current, target-policy
  successor average, grouped-mean `1/n_x` writer).
- Finite `C[t,y] = (e^8 or 1)/(e^8 + 3)` matches softmax of logits
  `xi * 1{x_t = y}`; finite `S` numerator `(e^8 or 1) * pi` with denominator
  `e^8 + 1` matches softmax of `zeta * 1{u_t = state_y} + log pi` (the two
  matching-state pairs contribute `e^8 * (pi0 + pi1) = e^8`, the other two
  contribute `1`); finite `W` denominator `counts[y]*e^8 + 64 - counts[y]`
  matches softmax of `tau * 1{x = x_t}`.
- `G0 = I + alpha * W0 @ (gamma * S0 - C0)`, `b0 = alpha * W0 @ r`, and the
  finite analogues match `affine_maps`.
- `c_f = max row sum |Gf|` matches the infinity-norm definition used by both
  Claude scripts.
- The population audit `T_pi[x,y] = P[x, state_y] * pi[state_y, action_y]`
  and `q_pi = solve(I - gamma * T_pi, R)` match `population_q_pi`.
- The signed stage errors `e_current`, `e_successor`, `e_write` and the
  recursion `E_{k+1} = c_f E_k + ||Ff(q_k) - F0(q_k)||_inf` match the
  witness decomposition exactly.
- The direct grouped-mean update matches `direct_reference_step` (grouped
  residuals averaged by `1/len(group)`, no visitation-frequency multiplier).
- The conservative analytic certificate uses exact rational arithmetic:
  `1 + 1 + 1/2 + 1/6 + 1/24 = 65/24 > 8/3`, `(8/3)^8 = 16777216/6561
  > 2000`, and the rational bound `17/20 + (7/10)/2001 + 3/2003 +
  (17/10)(63/2063) ~= 0.9038 < 1`. These are exact rational facts that
  check out by hand; the derivation of the certificate itself is GPT's
  theory contribution and is outside the numerical evidence compared here.

The command-line adapter compares the reconstruction against the sealed
Claude `results.json` with tolerance `1e-12 * (1 + scale)` for point
quantities and `1e-10 * (1 + scale)` for traces, checks the Claude
verifier's own verdict, and writes
`results/FP-EXPL-001/codex/verification.json`. The output-path logic
(`parents[4]` of the script location) resolves to the main-repo results
directory where the sealed GPT artifact was in fact read. The check count is
internally consistent: 2 sampling checks + 13 matrix/fixed-point checks + 4
trace checks + 1 verifier-verdict check = 20, matching the sealed summary.

## 3. Reported differences

Every entry in the report's difference table was checked against the sealed
`results/FP-EXPL-001/codex/verification.json`:

| evidence | report | sealed JSON | consistent |
|---|---:|---:|:---:|
| C0, S0, W0 | 0 | 0.0, 0.0, 0.0 | yes |
| finite C, S, W | 3.33e-16 | 3.33e-16, 2.22e-16, 2.78e-17 | yes (report quotes the worst) |
| G0, b0 | 0 | 0.0, 0.0 | yes |
| Gf, bf | 3.33e-16 / 8.33e-17 | 3.3306690738754696e-16 / 8.326672684688674e-17 | yes |
| q_pi, q_hat | 0 | 0.0, 0.0 | yes |
| q_f,inf | 8.88e-16 | 8.881784197001252e-16 | yes |
| exact direct trace | 0 | 0.0 | yes |
| exact attention trace | 6.66e-16 | 6.661338147750939e-16 | yes |
| finite literal trace | 8.88e-16 | 8.881784197001252e-16 | yes |
| finite scalar trace | 1.33e-15 | 1.3322676295501878e-15 | yes |

Scalar claims also match: visit counts `[19, 16, 12, 17]` with minimum 12,
`c_f = 0.85` (sealed as `0.8500000000000001`), state values
`[1.7689620758483031, 1.4096806387225547]`, and data bias
`0.03806150093295335` (the last two from the sealed Claude
`verification.json`, group `I_fixed_points_decomposition`). The reported
Claude verifier tally "1285 checks, 0 failures, PASS" matches the sealed
Claude artifact, and the group check counts in that artifact sum to exactly
1285. The reported GPT tally "20 checks, 0 failures, PASS" matches the
sealed GPT artifact.

## 4. Source-boundary claims

The report's boundary statements were checked against the sources:

- "GPT did not import Claude modules": confirmed for `verify_claude.py`
  (imports listed above).
- "The Claude verifier is self-contained and does not import the witness":
  confirmed; `verify.py` imports only `json`, `math`, `os`, `platform`,
  `sys`, `numpy`, and re-derives everything independently (including a
  separately coded pure-Python literal attention network).
- "The witness uses the frozen behavior policy only for sampling":
  confirmed; `BEHAVIOR_PI` appears only in `sample_batch`.
- "the target policy only for successor averaging": confirmed;
  `TARGET_PI` feeds the operators, direct reference, and audit, never the
  sampler.
- "the transition matrix only for sampling and population audit":
  confirmed; `P_NEXT` appears only in `sample_batch`, `population_q_pi`,
  and the audit residual.
- Fixed projection weights, static role/position masks, dynamic Q only
  through the attention read/write stages: consistent with the
  `LiteralNetwork` code (weights built from frozen constants and one-hot
  identity features; masks depend only on token roles; Q enters via the
  memory-token field and stage outputs).
- No external Q lookup, visitation gate, resampling, policy update, or
  result-driven parameter change: nothing in either Claude script
  contradicts this; the coverage gate only stops the run before any Q
  update.
- The exact same-batch route uses declared equality masks only as its
  reference operator and is not presented as finite-network capability:
  consistent with both the code (`mode="exact"` branch) and the witness
  documentation.

The report's summary sentence that the signed three-term identity, affine
reconstruction, contraction bounds, fixed-point decomposition,
target-policy preservation, grouped-mean property, scratch clearing,
immutable fields, and input rejection all pass is supported by the sealed
Claude `verification.json` (all twelve groups report 0 failures).

## 5. Execution-provenance caveat

The report discloses that Claude's Bash tool was blocked by a local `EPERM`
and that GPT, not Claude, executed the two Claude-authored scripts from the
Claude worktree to produce the sealed evidence, before replaying the
results independently. This reviewer confirms the caveat is stated openly in
the report and is not contradicted by any artifact: the sealed Claude
`verification.json` records the interpreter `C:\Users\Admin\anaconda3\python.exe`
(Python 3.13.9, numpy 2.4.6), consistent with the GPT-side command lines
quoted in the report. The deviation concerns execution provenance only; it
does not change any frozen input, source file, or protocol step, and the
GPT acceptance route re-derives every sealed number independently, so the
numerical evidence does not rest on who pressed the keys. This reviewer
additionally notes it cannot re-execute anything (Read/Edit only), so the
execution claim itself is accepted as disclosed rather than independently
reproduced.

## 6. Limitations of this review

- No command was run; all "matches" above are file-content comparisons, not
  re-executions.
- `theory.md` and `report.md` (mentioned in the GPT report) and the sealed
  Claude `results.json` itself were not part of the required reading set;
  the GPT report's claims about them were assessed only through the two
  scripts and the two verification artifacts.
- The analytic uniform-contraction certificate's derivation (as opposed to
  its exact rational arithmetic, which checks out) lives in GPT's theory
  document and was not re-derived here.

## Status

The report's claims about frozen inputs, independent replay, reported
differences, and source boundaries are supported by the evidence files
listed above, and the execution-provenance deviation is disclosed in the
report rather than concealed. No discrepancy was found between the report
and the sealed artifacts.

**PASS**
