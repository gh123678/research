# FP-ATTN-ITER4-001 route journal

Branch: `main`. Single actor: Claude holds both execution and verification under
the standing user instructions of 2026-09-11.

## 1. The gap this task closes

Two verified results did not align in horizon:

| result | horizon | code path |
|---|---|---|
| `FP-ITER4-001` | **four** certified steps, `12/48` | numpy |
| `FP-ATTN-ITER-001` | **three** certified steps, `15/48` | literal network |

So the deepest certified result rested on numpy, and the network claim stopped
one step short of it. This task runs the network at `MAX_STEPS = 4`, with the
comparison baseline switched to `FP-ITER4-001` so step 4 is matched against the
matching numpy horizon.

## 2. Mandatory check: the network horizon change must be inert

Raising the network horizon from `3` to `4` must not perturb network steps 1--3.
`H1`/`H2` confirm it: **all `90` step-1..3 entries are identical** to the sealed
`FP-ATTN-ITER-001` run, including the recorded `Qhat` gaps, and the `15`
three-step route-records reproduce exactly. There are `0` reproduction failures.

## 3. Formal result

`24` records, `48` route-records, `105` network step executions.

| step | network emissions | numpy emissions | network mean gain | network min gain |
|---|---|---|---|---|
| 1 | `22` | `22` | `2.7177070297828965` | `0.10419714014227195` |
| 2 | `20` | `20` | `2.2779971698013703` | `0.7799017280406759` |
| 3 | `15` | `15` | `1.286905361647572` | `0.37819650415353206` |
| **4** | **`12`** | **`12`** | **`0.8074994690824351`** | **`0.18490214293638285`** |

| quantity | result |
|---|---|
| decision agreement vs numpy | **`105 / 105`** |
| decision flips | **`0`** |
| selected-`eta` flips | **`0`** |
| certificate violations | **`0`** |
| componentwise non-degrading violations | **`0`** |
| network four-step routes | **`12`** |
| producer set | `{literal_attention_network}` |

**Four certified steps are executable inside the literal attention network, and
the network reaches exactly the decisions the numpy route reaches.**

### The drift question at four steps

This was the substantive risk. `FP-ATTN-ITER-001` measured a per-step `Qhat` gap
that stayed flat through three steps (`1.076e-05`, `4.585e-06`, `7.176e-06`),
and the natural question was whether a fourth composition would finally let it
grow.

| step | max `\|Q_network - Q_numpy\|_inf` |
|---|---|
| 1 | `1.076e-05` |
| 2 | `4.585e-06` |
| 3 | `7.176e-06` |
| **4** | **`1.044e-05`** |

The step-4 gap is `1.044e-05`, of the same order as step 1 and still an order of
magnitude inside the frozen `ATOL = 1e-4`. **There is no accumulated drift at
four steps either.** The flatness is now observed across four compositions
rather than three, which strengthens the reading that the `float32` difference
behaves as per-step rounding noise rather than accumulated state error.

That explanation is offered as consistent with the measurement; the task claims
the measurement, not the mechanism.

### A small, expected numerical difference worth recording

The network's mean and minimum gains differ from numpy's at the fifth decimal or
so (e.g. step 1: `2.7177070297828965` network vs `2.71770715611645` numpy). This
is not a discrepancy: the two routes produce slightly different policies, and
`total_value_gain` is the *oracle* value change of the policy each route
actually produced. The emission counts, `eta` selections and decisions are
identical; only the measured gain of the resulting policy differs in the last
digits.

## 4. Hypothesis verdicts

| hypothesis | verdict |
|---|---|
| `H1` network horizon inert | **PASS** (`90/90` identical) |
| `H2` three-step reproduction | **PASS** (`15`) |
| `H3` network produced every `Qhat` | **PASS** (producer set exact) |
| `H4` fourth step certifiable in the network | **PASS** (`12`) |
| `H5` decisions agree with numpy | **PASS** (`105/105`) |
| `H6` no accumulated drift at step 4 | **PASS** (`1.044e-05 <= 1e-4`) |
| `H7` flips reported, never absorbed | **PASS** (enumeration exercised and empty) |
| `H8` validity under the network | **PASS** (`0` violations) |

## 5. Corpus integrity

The horizon knob lives in `evaluate_fp_attn_iter_001.py`, so extending it
changed that file. No sealed `environment.json` recorded it, so **no prior
report was affected** — confirmed by checking every sealed bundle's tracked file
list. All prior verifiers still exit `0`.

The scientific corpus remains byte-identical, and the inertness proof above
(`90/90`) discharges the same obligation that `FP-ITER4-001` discharged for the
numpy evaluator.

## 6. What the project now has

| question | answer | code path |
|---|---|---|
| can the network produce a certified improvement | yes, `22/48` | network |
| is it one-shot | no, `20/22` continue | network |
| how far does it go | **four** steps, `12/48` | **network and numpy agree** |

The network and numpy horizons are now aligned at four steps, so the deepest
certified result no longer depends on which code path is trusted. The
`float32`-versus-`float64` question is settled empirically for this matrix: it
changes no decision through four steps.

## 7. Verification

By the user's instruction of 2026-09-11 ("验证先不管"), verification was not the
focus of this round. The derived checks were still run and are recorded at
`docs/research_branches/FP-ATTN-ITER4-001/claude/verification_same_actor.md`;
they confirm the horizon inertness, the per-level pattern, zero violations, and
that the producer set is exactly the network.

**Limitation:** same-actor, and per the user's instruction not emphasised. No
second actor reconstructed this route, and every result in this line rests on
implementations by one author. Agreement between the network and numpy paths
rules out implementation drift between them, not a shared conceptual error.
