# FP-ITER5-001 route journal

Branch: `main`. Single actor: Claude holds both execution and verification under
the standing user instructions of 2026-09-11 ("好的第五步", with the earlier
"验证先不管" still in force).

## 1. What this task tests

Four certified steps are verified on both paths. A fifth step answers two open
questions: where the iteration actually stops, and whether the network-versus-
numpy `float32` gap stays flat. This round runs **both paths**, so they remain
aligned rather than the network falling a step behind again.

## 2. Mandatory checks: both horizon changes must be inert

Raising a horizon in shared code risks perturbing the earlier steps. Both were
checked at the previously frozen horizon:

| path | entries compared | mismatches |
|---|---|---|
| numpy (`--max-steps 4` vs sealed `FP-ITER4-001`) | `105` | **`0`** |
| network (`--max-steps 4` vs sealed `FP-ATTN-ITER4-001`, including the `Qhat` gaps) | `105` | **`0`** |

`H1`/`H2` **PASS**. Both horizon extensions are inert.

## 3. Formal result

`24` records, `48` route-records. The numpy path executed `120` step entries
(`22+20+15+12+12` at the first four levels plus level 5); the network path
executed the same structure with `105` comparisons available.

| step | numpy emissions | network emissions | mean gain | minimum gain | network `max\|dQ\|` |
|---|---|---|---|---|---|
| 1 | `22` | `22` | `2.717707` | `0.104197` | `1.076e-05` |
| 2 | `20` | `20` | `2.277997` | `0.779902` | `4.585e-06` |
| 3 | `15` | `15` | `1.286906` | `0.378197` | `7.176e-06` |
| 4 | `12` | `12` | `0.807500` | `0.184902` | `1.044e-05` |
| **5** | **`12`** | **`12`** | **`0.361474`** | **`0.082395`** | **`4.567e-06`** |

`12` route-records emitted all five steps on **both** paths. **`0` certificate
violations, `0` non-degrading violations** on either path.

## 4. The substantive finding: the population plateaus while gains keep decaying

Emission deltas: `−2, −5, −3, **0**`. After four steps of decline
(`22 → 20 → 15 → 12`), the fifth step emits **exactly the same 12
route-records**.

Verified directly rather than inferred: the set of emitting route-records at
step 4 and at step 5 is **identical** — nothing lost, nothing gained.

Meanwhile the mean gain keeps falling: `2.717707 → 2.277997 → 1.286906 →
0.807500 → 0.361474`, with ratios `0.8382, 0.5649, 0.6275, 0.4476`.

So the iteration has entered a **regime, not a cliff**: a stable set of
route-records continues to be certified improvement after improvement, with each
step delivering less. The minimum fifth-step gain is `0.0824`, still above the
pre-registered `0.05` floor but an order of magnitude below the first step's
`0.104`-scale margins in the later regime.

This is a different picture from the two candidate readings available before
this round:

- *"the population shrinks until nothing is left"* — not observed; it plateaued;
- *"the gains decay geometrically toward zero"* — consistent with the mean
  sequence, but the emission plateau means the iteration does **not** terminate
  by running out of emitters within five steps.

Both readings are now constrained by data rather than guessed at.

## 5. Pre-registered prediction outcomes

| prediction | registered | outcome |
|---|---|---|
| `H7`: `n5 < n4` | yes | **FALSIFIED** (`12`, not fewer) |
| `H8`: `mean5 < mean4` | yes | **PASS** (`0.3615 < 0.8075`) |
| `H9`: `min5 > 0.05` | yes | **PASS** (`0.0824`) |
| `H10`: step-5 network gap `<= ATOL` | yes | **PASS** (`4.567e-06`) |
| `H11`: both paths agree per level and per decision | yes | **PASS** |

**`H7` is reported as FALSIFIED and not reinterpreted.** It was a legitimate
extrapolation of a monotone decline across three intervals; the fourth interval
broke it. Its failure is what produced the plateau finding, so the prediction
earned its place even in failing.

`H11` deserves note: the two paths agree at **every** level (`22, 20, 15, 12,
12`) and on **all `105`** decision comparisons, with `0` disagreements.

## 6. The drift question at five steps

| step | network `max\|Q_network − Q_numpy\|_inf` |
|---|---|
| 1 | `1.076e-05` |
| 2 | `4.585e-06` |
| 3 | `7.176e-06` |
| 4 | `1.044e-05` |
| **5** | **`4.567e-06`** |

Still flat, still an order of magnitude inside `ATOL`. The gap is now observed
across **five** compositions with no sign of growth. Only the measurement is
claimed, not the mechanism.

## 7. Hypothesis verdicts

| hypothesis | verdict |
|---|---|
| `H1` numpy horizon inert | **PASS** (`105/105`) |
| `H2` network horizon inert | **PASS** (`105/105`, gaps included) |
| `H3` fifth step certifiable on both paths | **PASS** (`12` and `12`) |
| `H4` fifth step valid | **PASS** |
| `H5` monotone value across five steps | **PASS** (`0` violations) |
| `H6` no certificate violations | **PASS** |
| `H7` attrition prediction | **FALSIFIED** |
| `H8` mean-gain prediction | **PASS** |
| `H9` non-vacuity prediction | **PASS** |
| `H10` drift prediction | **PASS** |
| `H11` path agreement | **PASS** |

## 8. Corpus integrity

Extending the horizons touched two shared evaluators. The impact was **audited
before any change**:

| evaluator | recorded by |
|---|---|
| `evaluate_fp_iter2_001.py` | `FP-ATTN-ITER-001`, `FP-ATTN-ITER4-001` |
| `evaluate_fp_attn_iter_001.py` | no sealed bundle |

So this round's numpy-horizon change affected **two** sealed records, unlike
`FP-ITER4-001`'s single one. Each affected record's verifier was updated to keep
the scientific-corpus hash checks **strict** while reporting the evaluator
evolution explicitly, and each was re-run and re-sealed as `PASS`. No committed
report was left reading `FAIL`.

The scientific corpus is byte-identical, and the inertness checks above
discharge the same obligation `FP-ITER4-001` established.

## 9. What the project now has

| question | answer | code path |
|---|---|---|
| can the network produce a certified improvement | yes, `22/48` | network and numpy agree |
| is it one-shot | no, `20/22` continue | network and numpy agree |
| how far does it go | **five** steps, `12/48` | **network and numpy agree** |
| does the population keep shrinking | **no — it plateaus at `12` from step 4** | both paths |
| do gains keep decaying | yes, smoothly: `2.718 → … → 0.361` | both paths |
| does the `float32` gap accumulate | no, flat across five compositions | network vs numpy |

## 10. Verification

By the user's instruction of 2026-09-11 ("验证先不管"), verification is not the
focus of this round. The derived checks are recorded at
`docs/research_branches/FP-ITER5-001/claude/verification_same_actor.md`.

**Limitation:** same-actor. No second actor reconstructed this route, and every
result in this line rests on implementations by one author. Agreement between the
network and numpy paths rules out implementation drift between them, not a shared
conceptual error.
