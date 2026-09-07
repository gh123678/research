# FP-MART-001 final verified report

Date: 2026-09-04. Status: `VERIFIED`. Main branch remains unchanged pending
explicit user approval.

## Result

The mandatory research question has a positive answer. On one frozen
fixed-policy trajectory, a visit-indexed martingale event replaces unknown
occupancy denominators by observed state and pair counts while retaining the
verified selective guarantee

```text
P(Emit and an emitted error bound is violated) <= delta.
```

The event is simultaneous over all `G = m + 2d` residual groups and every
visit count `1 <= k <= n`, with radius

```text
B sqrt(2 log(2 G n / delta) / k).
```

It composes with Direct-Q all-layer and V-first no-split exact/softmax bounds
without changing the previously verified deterministic recurrences. The
optional variance-adaptive constituent remains unavailable: neither route
constructed a valid observable count-only variance proxy.

## Formal evidence

Both routes used the frozen 480-record matrix with seed `20260829`. Both
preserved all legacy fields with zero mismatches, passed strict JSON and
schema/oracle-provenance audits, and reported zero empirical certificate or
per-group residual violations. Those zero violations are diagnostics, not a
replacement for the proof.

The two independent routes and the independent Claude reproduction of GPT
agree on every reported scientific rate:

| Length | Route | Emitted | Emission | `< B` | `< 2B` |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | Direct exact | 3/120 | 2.5% | 0 | 0 |
| 256 | Direct softmax | 3/120 | 2.5% | 0 | 0 |
| 256 | V-first exact | 3/120 | 2.5% | 0 | 0 |
| 256 | V-first softmax | 3/120 | 2.5% | 0 | 0 |
| 1024 | Direct exact | 102/120 | 85.0% | 0 | 0 |
| 1024 | Direct softmax | 74/120 | 61.7% | 0 | 0 |
| 1024 | V-first exact | 102/120 | 85.0% | 0 | 0 |
| 1024 | V-first softmax | 102/120 | 85.0% | 0 | 0 |
| 4096 | Direct exact | 120/120 | 100% | 0 | 0 |
| 4096 | Direct softmax | 115/120 | 95.8% | 0 | 0 |
| 4096 | V-first exact | 120/120 | 100% | 0 | 34/120 |
| 4096 | V-first softmax | 120/120 | 100% | 0 | 12/120 |
| 16384 | Direct exact | 120/120 | 100% | 0 | 7/120 |
| 16384 | Direct softmax | 120/120 | 100% | 0 | 0 |
| 16384 | V-first exact | 120/120 | 100% | 48/120 | 120/120 |
| 16384 | V-first softmax | 120/120 | 100% | 4/120 | 120/120 |

`< B` is the preregistered primary usefulness criterion: improvement over the
zero-initialization bound. `< 2B` is a secondary descriptive range threshold
added only to reconcile reporting conventions; neither threshold was selected
after observing results.

Exact-route emission equals the preregistered support rates
2.5% / 85% / 100% / 100%. Direct softmax additionally loses records to its
positive pair-margin gate. V-first softmax has no iterated pair recovery
operator, so it emits whenever the common support and state-stage gates pass.

The principal numerical conclusion is narrow but clear: under the conservative
Hoeffding radius, primary-useful bounds appear only at length 16384, where
V-first exact succeeds on 48/120 records and V-first softmax on 4/120. Direct-Q
never beats `B` in this grid. This supports a structural advantage for the
V-first composition, not a claim of online policy improvement.

## Independent-route comparison

The routes began from common activation commit
`c8ec7e5c3165930663e26a07f98b38cc9ec186ad` and remained blind until their
first results were sealed.

- GPT presents predictable visit selection through bounded visit stopping and
  integrates the certificate in one top-level builder. Claude presents the
  same concentration argument as a predictable truncated martingale transform
  and exposes separate event and route constructors. These are equivalent
  proof and software decompositions, not different probability claims.
- Both derive conditional range width `2B`, allocate one `delta` over `G*n`
  two-sided events, use the same observed-count radius, and compose the same
  deterministic route formulas.
- Their raw schemas and execution-specific hashes differ, but the frozen
  configuration, legacy comparisons, emission rates, usefulness rates, and
  audit conclusions agree. Claude's reproduction of GPT has byte-identical
  config, task records, and summary; its regression file differs only in the
  isolated output-directory field.
- Claude's first sealed route failed GPT review because of an invalid written
  raw-MGF iteration, an oracle reward-bound input, insufficient count
  validation, incomplete smoke alignment/audits, and missing evidence. Claude
  repaired these on its own branch. A later evidence-only repair truthfully
  records that one required verifier was first completed after the formal
  seal. The repaired scientific implementation and formal matrix reproduced
  exactly and passed final GPT verification.

## Verification record

- GPT implementation seal:
  `4cf6f50d69c5aa4937d181f758f546f9d2213c41`.
- GPT first-result seal:
  `e7c04111b7aec8c9dc5043883fcaa68cc158837b`.
- GPT formal-evidence seal:
  `63b84fd4295598c3e020d7e489552470ac1763c4`.
- Final repaired Claude evidence commit:
  `a559bd31769506565dd4503d8239d4a3f28ddf81`.
- GPT verification of final Claude route: `PASS`, recorded in
  `FP-MART-001/codex/verification_of_claude_evidence_repair.md`.
- Claude verification of GPT route: `PASS`, commit
  `5999b889a3336866b3eec7099f74080b9b253ce5`, recorded in
  `FP-MART-001/claude/verification_of_codex.md`.

All five required verifiers and Ruff passed on both branches. GPT reproduced
the repaired Claude deterministic formal core exactly; Claude independently
reproduced the GPT 480-record result and matched its deterministic scientific
core.

## Hypothesis and acceptance outcome

- Hypotheses 1--6: supported within the frozen scope.
- Hypothesis 7: verified negative for the attempted construction; no valid
  optional variance-adaptive constituent was proved or implemented.
- Frozen acceptance criteria 1--18: satisfied after reciprocal `PASS` reports,
  this synthesis, and the workspace update. There is no unresolved objection.
- No merge, publication, external message, or change to `main` was performed.

## Limitations and next research direction

The result is fixed-policy, stationary-start, finite-state/action, synchronous,
and deterministic-edge-reward. The probability statement is per trajectory,
not simultaneous over the 480-run audit. Its finite-horizon union over counts
adds a `log n` factor and is often too conservative for Direct-Q.

The strongest next step is a new, separately frozen task investigating an
observable time-uniform or variance-adaptive boundary that preserves the same
no-oracle input contract. It should not be treated as available until a valid
predictable variance process, complete risk allocation, and independent
verification are supplied.
