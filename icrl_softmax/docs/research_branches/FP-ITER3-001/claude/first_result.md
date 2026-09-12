# FP-ITER3-001 route journal

Branch: `main`. Single actor: Claude holds both execution and verification under
the standing user instruction of 2026-09-11.

## 1. What this task tests

`FP-ITER2-001` established that the certificate is not one-shot (`20/22`
records emitted a second certified step). It also produced an *exploratory*
reading: between steps 1 and 2 the mean gain fell (`2.717707` → `2.277997`)
while the **minimum** gain rose (`0.104197` → `0.779902`), which looked like
selection filtering — survivors carrying more uniform improvement as the
thin-margin records drop out.

This task extended the horizon to `MAX_STEPS = 3` and **pre-registered** two
predictions so that the filtering reading could be falsified rather than
narrated:

- `H6`: `n3 < n2` (attrition continues);
- `H7`: `min_gain_step3 > min_gain_step2` (filtering continues).

Both were written into the task sheet before the run.

## 2. Mandatory check: the horizon change must be inert

Raising `MAX_STEPS` from `2` to `3` must not perturb steps 1 and 2. This is a
real risk, not a formality, because the evaluator is shared code.

`H1`/`H2` confirm it: with `--max-steps 3` the run reproduces the sealed
`FP-ITER2-001` values **exactly** —

| step | emissions | mean gain | minimum gain |
|---|---|---|---|
| 1 | `22` (= sealed `22`) | `2.71770715611645` (= sealed) | `0.10419713678630303` (= sealed) |
| 2 | `20` (= sealed `20`) | `2.2779972042823244` (= sealed) | `0.7799019383638572` (= sealed) |

A separate `--max-steps 2` replay of two records also matched the sealed bundle
on all four route-records. Step-1 reproduction failures: `0`. So the horizon
extension is inert, and the third step is a genuine continuation rather than a
different method.

## 3. Third-step result

| level | emissions | mean gain | minimum gain |
|---|---|---|---|
| step 1 | `22` | `2.71770715611645` | `0.10419713678630303` |
| step 2 | `20` | `2.2779972042823244` | `0.7799019383638572` |
| **step 3** | **`15`** | **`1.2869055664197442`** | **`0.37819659359698987`** |

`15` route-records emitted all three steps. Certificate violations `0`;
componentwise non-degrading violations `0`; the only stopping reason observed
remains `improvement_lcb_nonpositive`.

### Pre-registered prediction outcomes

| prediction | outcome |
|---|---|
| `H6`: `n3 < n2` | **PASS** (`15 < 20`) |
| `H7`: `min_gain_step3 > min_gain_step2` | **FALSIFIED** (`0.378 < 0.780`) |

`H7` is reported as falsified. It is **not** reinterpreted.

## 4. What the falsification means

The selection-filtering reading was **half right**.

- **Attrition is real and monotone**: `22 → 20 → 15`. Each step certifies fewer
  records, so iteration does progressively consume the population.
- **The "minimum gain keeps rising" part is false.** The minimum gain fell from
  `0.780` at step 2 to `0.378` at step 3.

The explanation is straightforward once stated: the active set at step 3 is not
a subset of the step-2 *emitters*, it is the set of records that emitted at
step 2, and a record can emit at step 2 with a small margin and then, after the
tilt, present a **new** binding state with thin headroom at step 3. So each
step can admit its own newly-thin records, and the minimum gain is not
monotone.

Consequently the correct description of iteration here is **both** filtering
and churn: the population shrinks, but its margin profile does not improve
monotonically, because the policy move itself creates new tight states.

### The decay shape, stated with its limit

Mean gain: `2.718 → 2.278 → 1.287`. The ratios are `0.838` and `0.565`. With
three points no functional form is established, and the task claims none. What
is established is that the mean gain decays and the emitting population
shrinks, so iteration is not free.

## 5. Hypothesis verdicts

| hypothesis | verdict |
|---|---|
| `H1` step-limit monotonicity (steps 1–2 unchanged) | **PASS** |
| `H2` step-2 reproduction of the sealed result | **PASS** |
| `H3` third step is certifiable | **PASS** (`15`) |
| `H4` third step valid and strictly improving | **PASS** (`15/15`) |
| `H5` value monotone across three steps | **PASS** (`0` violations) |
| `H6` attrition prediction `n3 < n2` | **PASS** |
| `H7` filtering prediction on the minimum gain | **FALSIFIED** |
| `H8` no certificate violations | **PASS** |

## 6. What this establishes overall

Three consecutive certified relative-softmax steps are possible on `15` of the
`48` frozen route-records, every emitted step componentwise non-degrading and
strictly improving, with zero certificate and zero non-degrading violations.
Value rises monotonically across all three steps.

Combined with `FP-ATTN-001`, the mechanism is now demonstrated to be
**iterable**, not merely possible — while the falsified `H7` shows the iteration
does not simply converge: it costs population and does not monotonically
improve the margin profile.

## 7. Verification

`verify_fp_iter3_001_same_actor.py` — see the report at
`docs/research_branches/FP-ITER3-001/claude/verification_same_actor.md`.

Sealed-file integrity was re-checked across all three prior sealed tasks
(`FP-SCALE-002`, `FP-ITER2-001`, `FP-ATTN-001`): **total DRIFT `0`**, with the
raw and LF-normalised line-ending forms distinguished explicitly rather than
hidden behind a normalisation.

## 8. Limitation

Same-actor derived verification. No second actor reconstructed this route, and
`FP-ATTN-001`'s literal-versus-numpy agreement, `FP-ITER2-001` and this task all
rest on implementations by one author. A shared conceptual error would not be
caught by their agreement. Independent verification remains outstanding for the
project's headline result.
