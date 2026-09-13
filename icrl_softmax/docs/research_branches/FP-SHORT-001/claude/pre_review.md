# FP-SHORT-001 same-actor pre-review

Date: 2026-09-12.
Reviewer: Claude, acting as both executor and verifier under the user's standing
instruction of 2026-09-11 ("验证先不管") and the scope instruction of 2026-09-12
("去做").

**This is not an independent review.** There is no second actor.

Task under review: `docs/research_tasks/FP-SHORT-001.md` v1.0.

Outcome: `APPROVED`, with four conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| The sheet carries every element `AGENTS.md` section 3 requires | PASS |
| The **basis** of each prediction is stated | PASS |
| Falsifying `H3`--`H6` is declared non-invalidating | PASS |
| Prohibited work forbids adopting the extended grid as a protocol | PASS |
| The three candidate mechanisms are each given a separable measurement | PASS |
| Stopping conditions make `H1`/`H2` failure fatal to the diagnosis | PASS |

## 2. Condition 1 (mandatory): the instrumented path must be proven to be the rule

The diagnosis uses a re-implemented decision function, because the sealed
`improvement_for` hardcodes the eta grid and cannot be handed a different one — the
same constraint the sealed module itself documents for `ETA_GRID`.

A re-implementation is exactly where a diagnosis can go wrong: measure a subtly
different rule, report a mechanism that the real rule does not have. The failure would
be silent, because a near-identical rule produces near-identical numbers.

Condition: assert equality with `fs.improvement_for` — status **and** selected `eta` —
at every step of every trajectory (`H1`, mandatory), and additionally require the
frozen-grid arm to reproduce `FP-ITER8X-001`'s sealed `frozen` arm step for step,
including `e_q`. That is a check against a **sealed bundle** rather than against
itself, so it would catch a divergence in the rule, the certificate, the sampler or
the trajectory walk.

## 3. Condition 2 (mandatory): the extension must be provably inert where it should be

The frozen eta values are a descending **prefix** of the extended grid, and the rule
scans in descending order taking the first passing candidate. So adding smaller
candidates cannot change any decision that the frozen grid already made: the scan
stops in the prefix.

That makes `H2` a **theorem**, and therefore a check on the implementation rather than
a discovery. A counterexample would mean the evaluated grid is not the prefix claimed,
or the scan order is wrong. It is registered as mandatory for exactly that reason.

Keeping the frozen values as a prefix with the additions appended is what makes the
comparison interpretable: any difference between the arms is attributable to the added
candidates alone, not to a re-ordered or re-scaled grid.

## 4. Condition 3 (mandatory): the exact gate margin must be the quantity reported

"The record nearly passed" is otherwise an impression. The exact gate margin

```text
h = max_eta min_s I_s / ||dpi_s||_1
```

is the largest certified error for which some candidate passes, so emission is exactly
`E_Q < h` and `E_Q / h` is the whole gate collapsed to one number. Its distance above
`1` is how narrowly the record missed, and it is the quantity that decides whether
mechanism 1 is in play.

Condition: report `h`, `E_Q` and their ratio for every short trajectory, register a
band (`H3`: `[1, 2]`) so the answer is a verdict rather than a table, and record the
smallest passing `eta` at the stopped step — which is the direct read on mechanism 3.

## 5. Condition 4 (mandatory): the grid extension must not be presented as a protocol

The eta grid is frozen protocol. Extending it is a protocol change that would require
its own task, its own justification and its own inertness argument for every sealed
result that used the grid.

The temptation here is real: if the grid turns out to be the binding constraint, the
extended grid is a one-line fix that visibly improves the headline number, and the
report would be much more satisfying if it ended "and so we extended the grid".

Condition: the extended grid is labelled a **diagnostic arm** in the task sheet, in the
config, in the analyzer output and in the result record; the report must not recommend
adopting it; and the verifier checks that no evaluator other than this one uses it.
If `H5` passes, the correct conclusion is "the grid bounds the iteration at the frozen
protocol", which is a statement about the protocol, not a licence to change it.

## 6. Inheritance fidelity

| element | status |
|---|---|
| protocol, seed, horizon, certification | inherited from `FP-ITER8X-001` exactly, so the regression check is meaningful |
| certificate | sealed variance-adaptive, identical in both arms |
| routes, residuals, ordered reasons | frozen code, untouched |
| the change | the eta grid, in one diagnostic arm only |

## 7. Residual risks

| risk | assessment |
|---|---|
| The re-implemented rule differs from the frozen one | Addressed by condition 1, two ways. |
| The extension is not inert where it should be | Addressed by condition 2. |
| The diagnosis reports an impression rather than a margin | Addressed by condition 3. |
| The diagnostic becomes a proposed protocol | Addressed by condition 4. |
| The short-trajectory population is too small to conclude from | The count is reported and every verdict carries its denominator. |
| Size is confounded with realisation | Inherited and unchanged; both arms share it, so it cannot drive the arm comparison. |
| Same-actor verification | Not mitigated; disclosed. |

## 8. Verdict

**`APPROVED`**, contingent on the four conditions above being enforced executably,
which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.

## Objections

None.
