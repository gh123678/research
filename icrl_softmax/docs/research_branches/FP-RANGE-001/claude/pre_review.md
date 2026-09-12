# FP-RANGE-001 same-actor pre-review

Date: 2026-09-12.
Reviewer: Claude, acting as both executor and verifier under the user's standing
instruction of 2026-09-11 ("验证先不管") and the scope instruction of 2026-09-12
("都做").

**This is not an independent review.** There is no second actor.

Task under review: `docs/research_tasks/FP-RANGE-001.md` v1.0.

Outcome: `APPROVED`, with three conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| The sheet carries every element `AGENTS.md` section 3 requires | PASS |
| The **basis** of each prediction is stated | PASS |
| Falsifying `H2`--`H6` is declared non-invalidating | PASS |
| Prohibited work forbids re-tuning `tau` on the evaluation half | PASS |
| The unsound ceiling is admissible only as a labelled reference | PASS |
| Step-1-only scope stated; the iteration explicitly out of scope | PASS |

## 2. Condition 1 (mandatory): the tail bound must be the tight one

The tail probability enters the bias under a **square root**, so the choice of tail
bound dominates the whole construction. Hoeffding on the indicator costs
`sqrt(log(1/δ)/(2N)) ≈ 0.014` at `N = 16369`, which alone makes the bias `0.15` —
three times the entire frozen radius. The Maurer-Pontil range term is
`(7/3)log(2/δ)/(N−1) ≈ 9.4e-4`, far tighter here because the indicator's observed
variance is essentially zero.

The difference is not cosmetic: the pilot measured `+155%` with Hoeffding and
`+10.2%` with Maurer-Pontil. A negative result extracted from the weaker bound would
be a statement about a bad implementation, not about the lever.

Condition: use the tighter bound, and record in the module and the result **both**
numbers, so the negative result is visibly not an artifact of the choice that was
available to be made badly.

## 3. Condition 2 (mandatory): `tau` must come from the other half

Truncation needs a threshold, and choosing it by optimising over half B would require
a union bound over the threshold grid — expensive, and the kind of cost that quietly
cancels the gain being chased.

Condition: take `tau = max|Y|` over half **A**. Because half A is independent of half
B, every statement about half B holds *conditionally on `tau`* with no union bound at
all. The module states this, and the analyzer reports the fitted `tau` per record so
the threshold is visible rather than buried.

## 4. Condition 3 (mandatory): the price must be decomposed, not just totalled

The whole question is whether the honest price eats the saving, and a single `E_Q`
number cannot answer it. `H4` and `H5` are registered as claims about **which term**
carries the loss, and they disagree with each other by design: `H4` says the tail
price as a whole exceeds the concentration term, `H5` says the bias *alone* beats the
frozen radius.

Condition: report the empirical tail mass, the Cauchy-Schwarz bias, the concentration
term and `tau` for the binding pair of every route-record, and have the verifier
confirm the decomposition is finite and positive rather than trusting the summary.

`H5` was written to be the stronger claim and it **failed** (`0/48`; mean ratio
`0.735`). That is the useful outcome: it narrows the mechanism from "the bias is
prohibitive" to "the bias is comparable and the terms add". `H4` passing while `H5`
fails is exactly the information a totalled number would have hidden.

## 5. Inheritance fidelity

| element | status |
|---|---|
| protocol, rungs, sampler | inherited from FP-SAMPLE-001, whose gate validated the sampler |
| certified quantity | unchanged |
| certificate arms | `fixed_policy_bernstein_certificate.py`, additive only |
| decision rule | frozen code, untouched |
| the change | one new function plus a new evaluator |

## 6. Residual risks

| risk | assessment |
|---|---|
| A weak tail bound manufactures the negative result | Addressed by condition 1; both numbers reported. |
| `tau` chosen on the evaluation half | Addressed by condition 2. |
| The negative result is over-read as "no truncation can work" | The result is about **this** construction; the sheet and the result both say so. |
| The unsound ceiling is read as achievable | Labelled, excluded from coverage claims, and reported only as a reference. |
| A falsified `H5` is quietly recast | Reported falsified with the ratio, and the verifier re-derives the count. |
| Step-1-only scope overstated | Stated in both the sheet and the result. |
| Same-actor verification | Not mitigated; disclosed. |

## 7. Verdict

**`APPROVED`**, contingent on the three conditions above being enforced executably,
which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.

## Objections

None.
