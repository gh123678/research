# FP-ATTN-ITER-001 same-actor pre-review

Date: 2026-09-11.
Reviewer: Claude, acting as both executor and verifier under the user
instruction of 2026-09-11.

**This is not an independent review.** There is no second actor.

Task under review: `docs/research_tasks/FP-ATTN-ITER-001.md` v1.0.

Outcome: `APPROVED`, with two conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| Task sheet carries every element `AGENTS.md` section 3 requires | PASS |
| Research question matches the identified gap | PASS |
| `H4` failure is declared non-invalidating and informative | PASS |
| Prohibited work forbids numpy substitution where the network must produce `Qhat` | PASS |
| Stopping conditions cover an infeasible network run | PASS |

## 2. The gap is real, not rhetorical

Verified by inspection of the sealed bundles on the current baseline:

- `FP-ATTN-001` ran the literal network for **one** step and compared decisions;
- `FP-ITER2-001` and `FP-ITER3-001` ran up to three steps but **only** through
  `fixed_policy_expected_sarsa_scaled.fs.run_route`, which is numpy;
- no sealed artifact records a network-produced `Qhat` at step 2 or step 3.

So the claim "the network carries the iteration" was genuinely unestablished.

## 3. Condition 1 (mandatory): provenance must be enforced, not asserted

The certificate at every step must be scored on the **network's** tensor output.
The implementation therefore:

- calls the network for `Qhat` at every step and passes that array straight to
  `variance_adaptive_certificate`, with no numpy `run_route` result in that path;
- records `qhat_producer = "literal_attention_network"` on every step;
- computes the numpy route **additionally**, solely as the comparison baseline.

The verifier must confirm the set of producers is exactly
`{literal_attention_network}`; it does.

## 4. Condition 2 (mandatory): disagreements must be enumerated, never absorbed

`FP-ATTN-001` measured a one-step `Qhat` gap of `1.076e-05`, and iteration feeds
each step's output into the next, so the gap could compound past a decision
boundary. Third-step margins are of order `0.378`, and some records are far
tighter.

The implementation must therefore serialise, for **every** step, the literal
status, the numpy status, both `E_Q` values, both smallest per-state lower
bounds and the `Qhat` gap, precisely so that a flip surfaces with its boundary
margin. `H5` is satisfied by an enumeration that is exercised and empty, not by
an untested path. The analyzer reports `decision_flips`, `eta_flips` and
`flip_details` unconditionally.

## 5. Inheritance fidelity

| element | status |
|---|---|
| protocol dimensions, lengths, mixing, gap, seed | inherited from FP-ITER3-001 |
| certificate and decision code | same frozen modules, no fork |
| route→network mapping | inherited from FP-ATTN-001 |
| `ATOL = 1e-4` on `Qhat` | inherited from FP-ATTN-001 |
| `MAX_STEPS = 3` | inherited from FP-ITER3-001 |
| batches identical at every step | required and digest-recorded |

## 6. Residual risks

| risk | assessment |
|---|---|
| `float32` network error compounds across steps and flips decisions | **The main risk.** Either outcome is informative; `H5` guarantees visibility. Measured outcome: it does not compound. |
| Network cost at `160` layers x 3 steps x 48 route-records | Measured about 1 s per network run, so about 90 s of network time plus sampling; comfortably in budget. |
| Same-actor verification | Not mitigated; disclosed. |
| The numpy baseline is also by this author | Disclosed: agreement rules out implementation drift, not a shared conceptual error. |

## 7. Verdict

**`APPROVED`**, contingent on the two mandatory conditions in sections 3 and 4
being enforced executably, which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.
