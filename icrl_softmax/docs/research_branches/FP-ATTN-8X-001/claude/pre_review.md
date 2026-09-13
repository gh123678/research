# FP-ATTN-8X-001 same-actor pre-review

Date: 2026-09-12.
Reviewer: Claude, acting as both executor and verifier under the user's standing
instruction of 2026-09-11 ("验证先不管") and the scope instruction of 2026-09-12
("好的").

**This is not an independent review.** There is no second actor.

Task under review: `docs/research_tasks/FP-ATTN-8X-001.md` v1.0.

Outcome: `APPROVED`, with four conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| The sheet carries every element `AGENTS.md` section 3 requires | PASS |
| The **basis** of each prediction is stated | PASS |
| Falsifying `H3`--`H7` is declared non-invalidating | PASS |
| Prohibited work forbids numpy substitution on the network path | PASS |
| The comparison reference is named exactly, not by description | PASS |
| The sampler confound is restated rather than inherited silently | PASS |

## 2. Condition 1 (mandatory): provenance must be proven structurally

The evaluator computes **two** certificates and **two** decisions per step — one on
the literal network's `Qhat`, one on the numpy comparator — because the comparator is
what the path comparison needs. A check that merely finds `q_literal` in the file
would pass even if the recorded decision were taken on numpy.

Condition: inspect the assignments whose **results are bound to** `certificate` and
`decision`, require `q_literal` among their arguments and `q_numpy` absent, and
separately confirm the comparator still exists so the confinement check is not
vacuous. Both are in the verifier. The bundle additionally records
`qhat_producer = literal_attention_network` on every step.

A second guard is required: every step's `q_hat_gap_vs_numpy` must be **strictly
positive**. An exact zero would mean the network output had been replaced by numpy's,
and it is the one failure the string field cannot detect, since the same code writes
both.

## 3. Condition 2 (mandatory): the horizon must match the numpy run being compared to

`H4` compares emitting sets step by step, which is only meaningful if both runs
cover the same steps. `FP-ITER8X-001` ran `12` steps at `8x` on numpy; a network run
at a different horizon would silently drop comparisons, and the loop bound would make
that failure look like agreement.

Condition: horizon `12`, the same `8x` certification, and the same two arms. The
analyzer loads `FP-ITER8X-001`'s bundle explicitly and reports per-step counts from
both sides, so a missing comparison is visible as a blank rather than as an
agreement.

## 4. Condition 3 (mandatory): agreement must be reported with its headroom

This is the condition that matters most, and it is the reason `H7` exists.

Through six steps at `1x` the paths agreed on every decision with `0` flips over `105`
comparisons. It would be easy to read that as "the paths agree", full stop. But the
margins thin as the iteration proceeds: `FP-ITER8X-001` recorded numpy minimum gains
of `0.000183` at step `11` and `0.000649` at step `12`, against an observed `float32`
gap of `~5e-6`. That is 30–100x of headroom, and it is the first time in this line
that the margin has come within two orders of magnitude of the arithmetic noise.

Condition: report `min emitted gain / max |dQ|` at **every** step, register a floor
(`H7`: worst headroom `≥ 20x`), and state in the report that once the headroom
approaches 1 the decision is being made by `float32` arithmetic rather than by the
certificate. Without this, a `PASS` on agreement would be read as unconditional when
it is in fact margin-limited, and the first flip would look like a bug rather than
like the arithmetic reaching the certificate's resolution.

## 5. Condition 4 (mandatory): a flip must be listed, never summarised

The informative outcome here is a late-step flip: it would establish that the
`float32` gap finally decides a marginal record, which is the first such event in
this line.

Condition: the analyzer lists every decision and `eta` flip individually with its
record, route and step, and the verifier re-derives the counts from the bundle rather
than trusting the summary. A suppressed or aggregated flip is a construction failure
under the task sheet's acceptance criteria.

## 6. Inheritance fidelity

| element | status |
|---|---|
| protocol, seed, eta grid, decision rule, `ATOL` | inherited verbatim |
| routes, residuals, ordered reasons, audit | frozen code, untouched |
| `pi_{k-1}` as the step-`k` target | inherited |
| one certification batch per record, reused at every step | inherited |
| certification | `8x`, from the sampler `FP-SAMPLE-001` validated |
| horizon | `12`, matching the numpy run |
| certificate | the arm; both arms run as on numpy |

## 7. Residual risks

| risk | assessment |
|---|---|
| A numpy `Qhat` drives a recorded decision | Addressed by condition 1, with the positive-gap guard. |
| Missing comparisons read as agreement | Addressed by condition 2. |
| Agreement read as unconditional | Addressed by condition 3. |
| A flip is hidden in an aggregate | Addressed by condition 4. |
| Drift accumulates over a five-times-longer horizon | This is `H6` and is the point of the run, not a risk to mitigate. |
| Size is confounded with realisation | Not removable; stated. |
| Runtime exceeds budget | Priced by a one-record smoke: 80 s/record, so ~32 min for 24. |
| Same-actor verification | Not mitigated; disclosed. |

## 8. Verdict

**`APPROVED`**, contingent on the four conditions above being enforced executably,
which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.

## Objections

None.
