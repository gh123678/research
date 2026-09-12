# FP-ATTN-ITER4-001 same-actor pre-review

Date: 2026-09-11.
Reviewer: Claude, acting as both executor and verifier under the user
instruction of 2026-09-11.

**This is not an independent review.** There is no second actor. By the user's
instruction of the same date ("验证先不管"), verification is also not the focus of
this round.

Task under review: `docs/research_tasks/FP-ATTN-ITER4-001.md` v1.0.

Outcome: `APPROVED`, with three conditions carried into the implementation.

## 1. Scope and consistency

| check | result |
|---|---|
| Task sheet carries every element `AGENTS.md` section 3 requires | PASS |
| The gap is real, not rhetorical | PASS (network sealed at 3, numpy at 4) |
| `H4`--`H7` failure is declared non-invalidating | PASS |
| Prohibited work forbids numpy substitution and flip suppression | PASS |
| Stopping conditions cover an infeasible four-step network run | PASS |

## 2. Condition 1 (mandatory): prove the network horizon change inert

The network horizon lives in shared code, so raising it could perturb network
steps 1--3. `H1` requires those steps to be **identical** to the sealed
`FP-ATTN-ITER-001` run on decision, `eta`, `E_Q` **and** the recorded `Qhat`
gap — the last of these matters because the gap is the quantity this task is
ultimately about. Checked on the full matrix before any four-step run.

## 3. Condition 2 (mandatory): compare step 4 against the matching baseline

The sealed `FP-ATTN-ITER-001` reference bundle contains only three steps. The
task therefore requires switching the comparison baseline to the sealed
`FP-ITER4-001` four-step numpy bundle, exposed as a `--reference` option, so that
step 4 is not silently uncompared.

## 4. Condition 3: audit the shared-evaluator impact before changing it

`FP-ITER4-001` established the pattern: the horizon knob lives in a shared
evaluator, so extending it changes a file some sealed record may track. The
review therefore required an audit of every sealed `environment.json` **before**
the change.

Audit result: **no sealed bundle recorded `evaluate_fp_attn_iter_001.py`.** The
bundle that would have (`FP-ATTN-ITER-001`) tracks
`evaluate_fp_iter4_001.py`-adjacent files but not the literal evaluator itself.
So this extension breaks no prior record, unlike `FP-ITER4-001`'s change to
`evaluate_fp_iter2_001.py`. The scientific-corpus checks remain strict
regardless, and the inertness proof is still required.

## 5. Inheritance fidelity

| element | status |
|---|---|
| protocol dimensions, lengths, mixing, gap, seed | inherited |
| certificate and decision code | same frozen modules, no fork |
| `pi_{k-1}` as step-`k` target | inherited |
| one training and one certification batch, reused at every step | inherited |
| route→network mapping | inherited from `FP-ATTN-001` |
| `ATOL = 1e-4` | inherited |
| `MAX_STEPS` | the only change, `3` → `4` |

## 6. Residual risks

| risk | assessment |
|---|---|
| Network `float32` error compounds at four compositions | The substantive question; named by `H6`, serialised per step, and compared against the matching numpy horizon. Either outcome is informative. |
| Horizon perturbation | Mitigated by `H1` over `90` entries. |
| Shared-evaluator record drift | Audited before the change; no sealed record affected. |
| Same-actor verification | Disclosed; per the user's instruction, not emphasised. |

## 7. Verdict

**`APPROVED`**, contingent on the three conditions above, which the
implementation satisfies.
