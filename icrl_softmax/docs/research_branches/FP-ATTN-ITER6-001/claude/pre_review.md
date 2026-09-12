# FP-ATTN-ITER6-001 same-actor pre-review

Date: 2026-09-12.
Reviewer: Claude, acting as both executor and verifier under the user's standing
instruction of 2026-09-11 ("验证先不管") and the scope instruction of 2026-09-12,
in which the user selected "the network path is one step behind" from the list of
outstanding work.

**This is not an independent review.** There is no second actor.

Task under review: `docs/research_tasks/FP-ATTN-ITER6-001.md` v1.0.

Outcome: `APPROVED`, with three conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| The sheet carries every element `AGENTS.md` section 3 requires | PASS |
| Each prediction's **basis** is stated per prediction | PASS |
| Falsifying `H2`, `H5`--`H9` is declared non-invalidating | PASS |
| Prohibited work forbids numpy substitution and retuning | PASS |
| Stopping conditions cover the numpy-substitution case and budget | PASS |
| The task closes a gap its predecessor explicitly recorded | PASS |

## 2. Condition 1 (mandatory): the reference horizon must match the run's horizon

The evaluator's `--reference` selects which numpy bundle the network run is
compared against. It previously offered `iter3` and `iter4` only, and defaulted to
`iter3`. Running six network steps against a three-step numpy reference would
silently compare nothing at step 6 — the comparison loop is bounded by the shorter
of the two rows lists, so the step that this task exists to check would be dropped
without any error.

Condition: add the five- and six-step numpy bundles to `REFERENCE_BUNDLES`, keep the
default unchanged (so no earlier command's meaning shifts), and run this task with
`--reference iter6` explicitly. The analyzer independently asserts that the
configured reference is the six-step run.

## 3. Condition 2 (mandatory): prove the recorded decision uses the network's `Qhat`

The evaluator computes **two** certificates and **two** decisions per step: one on
the literal network's `Qhat` and one on the numpy comparator. A check that merely
finds `q_hat_literal` somewhere in the file would pass even if the recorded
decision were taken on numpy.

Condition: the verifier must inspect the call sites whose **result is bound to**
`certificate` and `decision` — the two that actually drive the iteration — and
require `q_hat_literal` to be among their arguments and `q_hat_numpy` to be absent.
It must also confirm the comparator still exists, so the confinement check is not
vacuous, and confirm that every step in the bundle records
`qhat_producer = literal_attention_network`.

A second, independent guard is added: every step's `q_hat_gap_vs_numpy` must be
**strictly positive**. An exact zero would mean the network output had been
replaced by the numpy one, and it is the one failure mode that the string
`qhat_producer` cannot detect, since that field is written by the same code that
would do the substituting.

## 4. Condition 3 (mandatory): the inertness proof must include the `Qhat` gaps

`FP-ITER5-001`'s network record stores `q_hat_gap_vs_numpy` per step. Those gaps
are float32-arithmetic artifacts: a rerun whose arithmetic differed by one ulp
anywhere would move them even if every decision still agreed. Comparing only
`update_emitted`, `eta` and `E_Q` would therefore under-test the inertness claim.

Condition: `H1` must reproduce the recorded gaps **exactly**, on all `117` sealed
route-step rows of levels 1--5, as a separate check from the decision comparison,
and report the two mismatch counts separately.

Note the constant: `117` is levels 1--5 (`48+22+20+15+12`). `105` is levels 1--4,
and quoting `105` produced a spurious failure in the sibling numpy task; both the
analyzer and the verifier compute the expected count from the sealed bundle rather
than hard-coding it.

## 5. Inheritance fidelity

| element | status |
|---|---|
| protocol dimensions, lengths, mixing, gap, seed | inherited from FP-ITER5-001 |
| certificate and decision code | same frozen modules, no fork, no reimplementation |
| `pi_{k-1}` as step-`k` target | inherited |
| one training and one certification batch per record, reused at every step | inherited, digest-recorded, structurally checked |
| oracle audit, realized error against `Q^{pi_{k-1}}` | inherited |
| `Qhat` producer on every step | the literal attention network, per `qhat_producer` |
| `MAX_STEPS` | the only change, `5` → `6` |

The network evaluator carries a provenance flag and a numpy comparator that the
numpy-only evaluator does not; neither is altered.

## 6. Corpus integrity

`evaluate_fp_attn_iter_001.py` is recorded by **no** sealed bundle, so this
horizon change cannot invalidate any earlier record's hash. `evaluate_fp_iter2_001.py`
**is** recorded by `FP-ATTN-ITER-001` and `FP-ATTN-ITER4-001`, but this task does
not touch it — it was already extended once by `FP-ITER6-001`, and that evolution
is documented in those records' verifiers. The verifier bounds any changed
evaluator to that pair and keeps the scientific-corpus hashes strict.

## 7. Residual risks

| risk | assessment |
|---|---|
| The six-step comparison silently drops step 6 | Mitigated by condition 1. |
| A numpy `Qhat` drives a recorded decision | Mitigated by condition 2 and the positive-gap guard. |
| Inertness under-tested by decision agreement alone | Mitigated by condition 3. |
| `float32` drift finally accumulates | This is the interesting outcome, not a risk to be mitigated; `H6` measures it. |
| Runtime exceeds budget | Priced by a two-record smoke before the formal run; abort if it does not fit. |
| Same-actor verification | Not mitigated; disclosed. |

## 8. Verdict

**`APPROVED`**, contingent on the three conditions above being enforced
executably, which the implementation and verifier do.

Same-actor approval; correspondingly less assurance than an independent review.

## Objections

None.
