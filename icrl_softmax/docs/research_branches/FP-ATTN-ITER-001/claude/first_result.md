# FP-ATTN-ITER-001 route journal

Branch: `main`. Single actor: Claude holds both execution and verification under
the standing user instruction of 2026-09-11.

## 1. The gap this task closes

The project's claim is about a fixed-weight softmax attention network, but the
certified iteration had only ever run in numpy:

| result | what was verified | where |
|---|---|---|
| `FP-ATTN-001` | literal network reproduces **one** certified step | literal network |
| `FP-ITER2-001` | a **second** certified step exists | numpy only |
| `FP-ITER3-001` | a **third** certified step exists | numpy only |

So one inference step was still unexecuted:

```text
literal network -> one certified step       (FP-ATTN-001)
literal network -> three certified steps    (never executed until now)
```

This task executes it: at every iteration step the `Qhat` fed to the frozen
certificate and decision rule is the one the **literal network** produced.

## 2. Construction

For each of the `24` frozen records, with the **identical** training and
certification batches and the same frozen code at every step:

```text
pi_0 = frozen target policy
for k = 1, 2, 3:
    Q_k    = LITERAL NETWORK(training batch, pi_{k-1})   <-- the change
    cert_k = variance-adaptive certificate(Q_k, pi_{k-1}, cert batch)
    decide = relative-softmax rule(pi_{k-1}, Q_k, cert_k)
    if decide abstains: stop, record the frozen reason
    pi_k   = decide.policy_plus
```

Route mapping inherited from `FP-ATTN-001`: `expected_exact` →
`EndToEndMaskedSoftmaxExpectedSARSA`; `expected_finite` →
`EndToEndFiniteSoftmaxExpectedSARSA`. Each step records
`qhat_producer = "literal_attention_network"`, and the verifier confirms the set
of producers is exactly that with no numpy substitution anywhere.

## 3. Formal result

`24` records, `48` route-records, `90` network step executions.

| quantity | network | FP-ITER3-001 numpy baseline |
|---|---|---|
| emissions at step 1 | **22** | 22 |
| emissions at step 2 | **20** | 20 |
| emissions at step 3 | **15** | 15 |
| three-step route-records | **15** | 15 |

| quantity | result |
|---|---|
| decision agreement vs numpy | **90 / 90** |
| decision flips | **0** |
| selected-`eta` flips | **0** |
| certificate violations | **0** |
| componentwise non-degrading violations | **0** |
| max `\|Q_network - Q_numpy\|_inf` | **1.076e-05** (frozen `ATOL` 1e-04) |

**Three certified steps are executable inside the literal attention network,
and the network reaches exactly the same decisions as the numpy route.**

### The accumulation question, answered

This was the real risk. `FP-ATTN-001` measured a one-step gap of `1.076e-05`,
and iteration feeds each step's output into the next, so the `float32`-versus-
`float64` difference could in principle compound until it crossed a decision
boundary — the third-step margins are of order `0.378`, and some records are far
tighter.

It does **not** compound. The per-step maximum gap is:

| step | max `\|Q_network - Q_numpy\|_inf` |
|---|---|
| 1 | `1.076e-05` |
| 2 | `4.585e-06` |
| 3 | `7.176e-06` |

The gap is essentially **flat**, of the same order at every step, rather than
growing. That is why `90/90` decisions agree and there is not one flip to
report.

The likely reason is structural: both routes are fixed-point iterations of a
contraction, and each step starts again from `Q_0 = 0` on the same batch, so the
`float32` error behaves like a per-step rounding noise rather than an
accumulated state error. This is offered as an explanation consistent with the
measurement, not as a theorem; the task claims the measurement, not the
mechanism.

## 4. Hypothesis verdicts

| hypothesis | verdict |
|---|---|
| `H1` step 1 anchored to `FP-ATTN-001` | **PASS** (max step-1 gap `1.076e-05`) |
| `H2` the network produced every step's `Qhat` | **PASS** (`90/90` provenance) |
| `H3` three steps certifiable in the network | **PASS** (`15` routes) |
| `H4` decisions agree with numpy | **PASS** (`90/90`, no `eta` flips) |
| `H5` flips reported individually | **PASS** (none exist; the enumeration path is exercised and empty) |
| `H6` validity under the network's own `Qhat` | **PASS** (0 violations, all gains positive) |
| `H7` nothing sealed changes | **PASS** |

`H5` deserves a note: it is reported as PASS because the enumeration exists and
is empty, not because it was never tested. The record level carries, for every
step, the literal status, the numpy status, both `E_Q` values, both smallest
lower bounds and the `Qhat` gap, so a flip anywhere would have surfaced with its
boundary margin.

## 5. What this establishes

Combining the four tasks, the project's central claim now holds end to end:

| question | answer | evidence |
|---|---|---|
| can the attention network produce a certified improvement | yes, `22/48` | `FP-SCALE-002` + `FP-ATTN-001` |
| is it one-shot | no, `20/22` continue | `FP-ITER2-001` |
| how far does it go | three steps, `15/48` | `FP-ITER3-001` |
| does the **network** carry the iteration | yes, identically | **this task** |

The final row is what makes the earlier three apply to the network rather than
to a numpy reimplementation of its formulas. `60` of the `90` step executions
continued past the first step, so this is genuine multi-step network execution,
not a single step repeated.

## 6. Verification

`verify_fp_attn_iter_001_same_actor.py` — **PASS** — including recomputation of
the per-step gaps, confirmation that every step's `Qhat` came from the network,
the finite route's gate-free property re-checked on this data, replay of five
sealed programs, and byte-identity of all seven tracked sealed files.

Report: `docs/research_branches/FP-ATTN-ITER-001/claude/verification_same_actor.md`.

## 7. Limitation

Same-actor derived verification. No second actor reconstructed this route, and
`FP-ATTN-001`, `FP-ITER2-001`, `FP-ITER3-001` and this task all rest on
implementations by one author. The network-versus-numpy agreement is therefore a
**within-author** cross-check: it rules out implementation drift between the two
paths, but it cannot catch a conceptual error shared by both. Independent
verification remains the one outstanding item for the project's headline result.
