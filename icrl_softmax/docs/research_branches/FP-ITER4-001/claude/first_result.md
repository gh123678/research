# FP-ITER4-001 route journal

Branch: `main`. Single actor: Claude holds both execution and verification under
the standing user instruction of 2026-09-11.

## 1. What this task tests

Three verified steps describe a trend but cannot distinguish candidate laws, and
one earlier prediction about that trend had already been falsified. This task
extends the horizon to `MAX_STEPS = 4` and **pre-registers** three step-4
predictions so the decay reading can be falsified rather than narrated.

## 2. Mandatory check: the horizon change must be inert

Raising `MAX_STEPS` from `3` to `4` must not perturb steps 1--3. `H1`/`H2`
confirm it: **all `90` step-1..3 entries are bit-identical** to the sealed
`FP-ITER3-001` run (decision, selected `eta`, `E_Q`), the `15` three-step
route-records reproduce exactly, and there are `0` step-1 reproduction failures.
The horizon extension is inert, so the fourth step is a genuine continuation.

## 3. Formal result

| step | emissions | mean gain | minimum gain | maximum gain |
|---|---|---|---|---|
| 1 | `22` | `2.71770715611645` | `0.10419713678630303` | `6.751510` |
| 2 | `20` | `2.2779972042823244` | `0.7799019383638572` | `3.823325` |
| 3 | `15` | `1.2869055664197442` | `0.37819659359698987` | `2.154661` |
| **4** | **`12`** | **`0.807499658432434`** | **`0.1849020565541281`** | — |

`12` route-records emitted **all four** certified steps. `105` step executions
in total. **`0` certificate violations and `0` non-degrading violations**; the
only stopping reason remains `improvement_lcb_nonpositive`.

### Pre-registered prediction outcomes

| prediction | registered | outcome |
|---|---|---|
| `H7`: `n4 < n3` | yes | **PASS** (`12 < 15`) |
| `H8`: `mean_gain_4 < mean_gain_3` | yes | **PASS** (`0.8075 < 1.2869`) |
| `H9`: `min_gain_4 > 0.05` | yes | **PASS** (`0.1849`) |

All three predictions held.

`H9` was deliberately the weakest meaningful continuation claim, not a
monotonicity claim, because the minimum-gain sequence was already known to be
non-monotone. It matters that it passed: the fourth step is a **real**
improvement, not a numerically vacuous move — the smallest fourth-step gain is
`0.185`, i.e. about `18%` of a unit of value, not a rounding artefact.

## 4. The shape of the decay, stated with its limits

Emission ratios between consecutive steps:

| interval | ratio |
|---|---|
| 1 → 2 | `0.9091` |
| 2 → 3 | `0.7500` |
| 3 → 4 | `0.8000` |

Mean-gain ratios:

| interval | ratio |
|---|---|
| 1 → 2 | `0.8382` |
| 2 → 3 | `0.5649` |
| 3 → 4 | `0.6266` |

Both sequences **dip at interval 2 → 3 and recover at 3 → 4**. Neither is
monotone, and neither is cleanly geometric. Four points still do not identify a
functional form, and this task claims none.

What four points do support:

- the emitting population shrinks at every step (`22 → 20 → 15 → 12`), by
  between `10%` and `25%` per step;
- the mean gain falls at every step (`2.718 → 2.278 → 1.287 → 0.807`), losing
  roughly `37%` over the three intervals on average;
- the per-step ratios of both quantities fluctuate rather than decay smoothly,
  which is consistent with the step-3 finding that iteration mixes attrition
  with **churn**: each policy move creates new tight states, so the population
  and the margin profile are re-drawn at every step rather than being filtered
  monotonically.

The churn reading, not a smooth decay law, is what the data now supports.

## 5. Hypothesis verdicts

| hypothesis | verdict |
|---|---|
| `H1` step-limit monotone (steps 1--3 unchanged) | **PASS** (`90/90`) |
| `H2` step-3 reproduction | **PASS** (`15`) |
| `H3` fourth step certifiable | **PASS** (`12`) |
| `H4` fourth step valid and strictly improving | **PASS** (`12/12`) |
| `H5` monotone value across four steps | **PASS** (`0` violations) |
| `H6` no certificate violations | **PASS** |
| `H7` attrition prediction | **PASS** |
| `H8` mean-gain decay prediction | **PASS** |
| `H9` non-vacuity prediction | **PASS** |

## 6. What the project now has

| question | answer | evidence |
|---|---|---|
| can the attention network produce a certified improvement | yes, `22/48` | `FP-SCALE-002`, `FP-ATTN-001` |
| is it one-shot | no, `20/22` continue | `FP-ITER2-001` |
| how far does it go | **four** steps, `12/48` | `FP-ITER3-001`, **this task** |
| does the **network** carry the iteration | yes, identically for three steps | `FP-ATTN-ITER-001` |

The fourth step has so far been executed on the **numpy** route only; the
network has been verified through three steps. Extending the network check to
four is the natural follow-up if it is wanted.

## 7. Verification

`verify_fp_iter4_001_same_actor.py` — **PASS** — including recomputation of the
per-level pattern, the `90`-entry inertness comparison, the three prediction
verdicts, validity and monotonicity, replay of five sealed programs, and
byte-identity of every tracked sealed file.

Report: `docs/research_branches/FP-ITER4-001/claude/verification_same_actor.md`.

## 8. Limitation

Same-actor derived verification. No second actor reconstructed this route, and
every result in this line of work rests on implementations by one author. A
shared conceptual error would not be caught. Independent verification remains
the one outstanding item for the project's headline result.
