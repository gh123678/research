# FP-CENSUS-001 + FP-ITER6-001 first result

Date: 2026-09-12.
Branch: `claude/FP-CENSUS-001`.
Baseline: `c1e03dd4e5cd610cf5b8ee0eb57c6b1de0844d0d`.
Actor: Claude, executing and checking under the user's instruction of 2026-09-12
("我的意思是1和2 一起做") and the standing "验证先不管".

This is the route journal for the combined task: the eligibility census
(`FP-CENSUS-001`) and the sixth certified step (`FP-ITER6-001`).

---

# Part 1 — the eligibility census

## 1. What was run

```text
python -B evaluate_fp_census_001.py --tasks 2  --mixings 0.08,0.5 \
    --output-dir results/FP-CENSUS-001/claude/smoke        # gate: H0 exact 8/8
python -B evaluate_fp_census_001.py --tasks 12 --mixings 0.08,0.5 \
    --output-dir results/FP-CENSUS-001/claude/formal       # H0 exact 48/48
python -B analyze_fp_census_001.py --write-prediction
```

Environment: Python `3.13.9` (Anaconda), NumPy `2.4.6`,
`Windows-11-10.0.22631-SP0`. Frozen certification `16384 x 64 = 1048576` items,
`4` states, `3` actions, mixing `{0.08, 0.5}`, `12` tasks each, seed `20260911`.

## 2. The batch generator had to be pinned first, and the first attempt was wrong

The census re-derives `E_Q` at every step. If its certification batch were not the
one the sealed iteration used, `E_Q` would differ and the census would describe a
different process while looking completely normal.

The first implementation replaced the per-chain loop with a vectorised per-step
draw — logically the same order, and equivalent in distribution for independent
chains. It was checked against `FP-SCALE-002`'s sealed `cert_pair_counts` and
**failed on all four probed records**:

| mixing | task | vectorised min | sealed min | pair counts equal |
|---|---|---|---|---|
| `0.5` | `0` | `19531` | `19728` | no |
| `0.5` | `1` | `25167` | `25586` | no |
| `0.08` | `0` | `31673` | `31583` | no |
| `0.08` | `3` | `31871` | `32381` | no |

`Generator.choice` with an explicit `p` does not consume the underlying stream the
way a raw uniform draw does, so "equivalent for independent chains" is false at
the level of the realised sample. The literal per-chain loop was restored and now
reproduces the sealed **full pair-count vector** exactly on all four probes
(`verify_census_batch_frozen.py`, exit `0`).

This is recorded because it is the single most dangerous failure mode in this
task, and because the guard that caught it is now a permanent artifact rather
than a one-off check.

## 3. `H0`: the census reproduces the sealed iteration exactly

| comparison | result |
|---|---|
| route-records exact | `48/48` |
| decision mismatches | `0` |
| ordered-reason mismatches | `0` |
| `eta` mismatches | `0` |
| `E_Q` mismatches (exact float equality) | `0` |
| row-count disagreements | `0` |

Every ratio below is therefore measured on the sealed iteration's own decisions
and its own certificate.

## 4. The three groups

Labels come from the **sealed** `FP-ITER5-001` bundle, not from this task's own
decisions. Step-1 ratio distribution:

| group | n | min | q1 | median | q3 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| never emitted | `26` | `0.5715` | `0.8509` | `1.1321` | `2.0114` | `2.2080` | `1.3237` |
| dropped out | `10` | `2.1721` | `2.4583` | `2.5849` | `3.5719` | `4.6574` | `3.0927` |
| plateau | `12` | `2.3991` | `3.3120` | `3.7939` | `4.5939` | `12.2644` | `5.0171` |

`2` of the `24` records have their two routes in different groups, so the census
is reported per route-record and not per record.

## 5. Every abstention has the same frozen reason

Across all `48` abstaining rows — every group, every step level — there is exactly
**one** ordered reason, and it is never a certificate failure:

```text
never   (26 rows):  26  improvement_lcb_nonpositive
dropout (10 rows):  10  improvement_lcb_nonpositive
by step:  1 -> 26,  2 -> 2,  3 -> 5,  4 -> 3
```

No row reports `not_certified` or any other reason. The certificate never fails;
what fails is the decision rule's own lower bound
`LB_s = Î_s − E_Q · ||π_η⁺(.|s) − π(.|s)||₁ > 0`.

This is the census's most quotable structural finding: **the binding constraint is
the improvement-per-movement budget, not certification.**

## 6. Verdicts

| hypothesis | verdict | evidence |
|---|---|---|
| `H1` step-1 separation | **PASS** | emitter mean `4.1424` > abstainer mean `1.3237` |
| `H2` separation strength | **FALSIFIED** | best of `49` thresholds still misclassifies `1/48` (`2.1%`) |
| `H3` plateau vs drop-out | **PASS** | plateau mean `5.0171` > drop-out mean `3.0927` |
| `H4` ratio decay | **PASS** (majority reading) / **FALSIFIED** (literal reading) | see §7 |
| `H5` the binding term | **PASS** | abstainers' mean `E_Q` `0.2630` > emitters' `0.2185` |
| `H6` spread is a record property | **PASS** | mean CV(`sigma_min`) `0.0177` < mean CV(`E_Q`) `0.0264` |

### `H2` — the near miss, stated without inflation

The registered prior was that `H2` would fail because the exploratory step-1
ranges overlapped. It failed, and the informative number is how narrowly:

```text
fewest misclassifications  = 1 of 48  (2.1%)
attained on 2 thresholds spanning [2.168689, 2.243823]
chosen theta = 2.168689 -> misclassifies the single ABSTAINER with ratio 2.208020
lowest emitter ratio = 2.1721, highest abstainer ratio = 2.2080  -> interleaved
```

So `sigma_min / E_Q` orders the population almost perfectly but does **not**
separate it, and per acceptance criterion 5 it is **not** described as a
classifier anywhere in this record.

By step level the separation does not improve with depth, and one level is
**vacuous** — it has only one class and therefore cannot separate anything:

| step | rows | emitted | abstained | best threshold misclassifies |
|---|---:|---:|---:|---|
| 1 | `48` | `22` | `26` | `1/48` |
| 2 | `22` | `20` | `2` | `1/22` |
| 3 | `20` | `15` | `5` | `3/20` |
| 4 | `15` | `12` | `3` | `2/15` |
| 5 | `12` | `12` | `0` | **not a separation test (one class only)** |

The step-5 line is reported as vacuous rather than as a perfect `0/12`, which is
what a mechanical reading of the scan would have produced.

## 7. `H4` — a defect in the task record, reported as one

The task sheet phrases `H4` universally ("along each trajectory the ratio is
lower"). The frozen plan operationalises it as a majority claim. Both readings are
reported:

- literal reading: **FALSIFIED** — `8` of `20` trajectories did not fall;
- plan reading: **PASS** — `12/20` (`60%`) fell.

Mean change `−0.0531`, median `−0.0727`, range `[−0.7904, +0.9206]`. The
trajectories that rose are listed individually in the analyzer report. The
divergence between the two phrasings is a defect in this task's own record and is
not resolved by quoting only the favourable reading.

## 8. `H5`/`H6` — the substantive answer: it is the spread, not the error

`H5` passes literally: abstainers do have a larger mean `E_Q`. But the effect
sizes point the other way, and this is the census's real answer:

| quantity | abstainers | emitters | ratio |
|---|---:|---:|---:|
| mean `E_Q` | `0.2630` | `0.2185` | `1.204` |
| mean `sigma_min` | `0.3273` | `0.8554` | `0.383` |

- `E_Q` alone, scanned as a predictor of abstention, misclassifies `16/48`
  (`33%`) — versus `1/48` for the ratio. The error is **not** the discriminator.
- The within-state spread differs by a factor of `2.6`; the certified error by
  `1.2`.

Combined with §5 (no record ever fails certification), the census's answer to
"why these records?" is: **those records have too little action-relevant value
spread for any policy movement to pay for itself, not too loose a certificate.**

`H6` supports this reading: `sigma_min` is nearly a record constant
(mean CV `0.0177`, median `0.0146`) while `E_Q` moves more in relative terms
(mean CV `0.0264`). Note the tension: in *absolute* terms the spread's mean range
(`0.0331`) is larger than the error's (`0.0153`), so `H6` holds relatively, not
absolutely. Both numbers are recorded.

## 9. Separation quality summary

`sigma_min / E_Q` is a strong ordinal signal and a weak classifier:

| question | best attainable rule | misclassifications |
|---|---|---:|
| who emits at step 1? | `ratio > 2.1687` | `1/48` (`2.1%`) |
| who reaches step 5 (plateau vs drop-out)? | any threshold | `6/22` (`27.3%`) |

So the step-1 ratio predicts **whether a record starts** far better than it
predicts **how long it lasts** — `H3` passes on means while its own best
threshold still misses more than a quarter of the plateau/drop-out split.

---

## 10. The frozen prediction for the sixth step

`results/FP-CENSUS-001/claude/formal/prediction_step6.json`, created
`2026-09-12T11:23:59Z`, sha256
`70b79f03dad1acbc689367ba3ea5af7123d875959c1d006bd7b035a9668961c7`, written before
`FP-ITER6-001` was executed and not edited afterwards.

Two rules were registered:

- **P1 (the plateau carries on)**: every route-record that emitted at step 5 emits
  at step 6 — `12` predicted emitters.
- **P2 (the step-1 threshold)**: emit at step 6 iff `ratio(5) > theta`, with
  `theta = 2.168689` frozen from step 1 — `12` predicted emitters.

### A defect in this prediction, found before the outcome was known

The scoring population ("reached step 5") turns out to be **exactly** the `12`
step-5 emitters: no route-record emitted steps 1–4 and then abstained at step 5.
Every one of those `12` has a step-5 ratio in `[2.5922, 13.1850]`, all far above
`theta = 2.1687`.

Therefore **P1 and P2 make identical predictions**, and `H6` ("P2 misclassifies
strictly fewer than P1") cannot be satisfied by any outcome. The test is
degenerate.

This was detectable at the moment the prediction file was written — the census
bundle already contained the step-5 ratios — and it was not caught. It is recorded
here as a defect in the pre-registered design, not excused. `H6` will be reported
as `FALSIFIED` with the two counts shown to be equal, and that equality is
evidence of the degeneracy rather than of P2 being worse.

The consequence for interpretation is stated in advance: **the census's criterion
has no discriminating power on the survivor set**, because all survivors sit far
above the threshold. The sixth step can therefore test only whether the plateau
continues (`H5`, `H7`/`H7b`), not which rule predicts it.

---

# Part 2 — the sixth certified step

## 11. What was run

```text
python -B evaluate_fp_iter2_001.py --max-steps 6 --tasks 12 --mixings 0.08,0.5 \
    --label numpy-six-step --output-dir results/FP-ITER6-001/claude/numpy
python -B analyze_fp_iter6_001.py
```

One structural change from `FP-ITER5-001`: `ALLOWED_MAX_STEPS` in
`evaluate_fp_iter2_001.py` gained the value `6`. Nothing else changed — same
batches, same frozen certificate and decision code, same `pi_{k-1}` target, no
retuning. The network path was **not** run, per the user's scoping of the sixth
step to the numpy path; the network is now one step behind, and that gap is
stated rather than hidden.

Prediction frozen at `2026-09-12T11:23:59Z`; sixth-step bundle written
`2026-09-12T11:31:19Z`. The prediction predates the result.

## 12. `H1`: the horizon change is inert

| comparison | result |
|---|---|
| sealed route-step rows across levels 1--5 | `117` |
| rows compared | `117/117` |
| mismatches (decision, `eta`, `E_Q` exact, ordered reasons) | `0` |

Note the constant: levels 1--5 hold **117** rows (`48+22+20+15+12`), not `105`.
`105` is levels 1--4. Both the analyzer and the verifier initially used `105` and
reported a spurious `H1` failure; both were corrected to compute the expected
count from the sealed bundle instead of hard-coding it.

## 13. The plateau broke

| step | emissions | mean gain | minimum gain |
|---|---:|---:|---:|
| 1 | `22` | `2.717707` | `0.104197` |
| 2 | `20` | `2.277997` | `0.779902` |
| 3 | `15` | `1.286906` | `0.378197` |
| 4 | `12` | `0.807500` | `0.184902` |
| 5 | `12` | `0.361474` | `0.082395` |
| **6** | **`9`** | **`0.167057`** | **`0.019347`** |

Emission deltas: `−2, −5, −3, 0, **−3**`. The set is no longer fixed:

```text
retained 9, lost 3, gained 0
lost: mix=0.08 task=5  expected_exact
lost: mix=0.08 task=5  expected_finite
lost: mix=0.5  task=4  expected_finite
```

`FP-ITER5-001`'s plateau was one interval, not a fixed point. The attrition
prediction `H7` (`n6 < n5`) — which had already been falsified once at step 5 —
now **PASSES**; its competitor `H7b` is falsified. Both were registered in
advance precisely so that one of them would have to be reported as lost.

## 14. The margin is thinning fast

- Mean gain: `0.807500 → 0.361474 → 0.167057`, roughly `×0.5` per step.
- Minimum gain: `0.184902 → 0.082395 → 0.019347` — a factor of `4.3` in one step.

`H8` (mean decays) **PASSES**. `H9` (minimum gain strictly positive and not below
`0.05`, the *same* floor registered at step 5, not a loosened one) is
**FALSIFIED**: `0.019347 < 0.05`.

This is a substantive finding, not a bookkeeping failure. The certified margin on
the last surviving records is now about `1/4` of the floor that step 4 and step 5
cleared. If the decay continues at this rate, step 7 has little room.

## 15. Neither pre-registered rule predicted the break

Scoring population: the `12` route-records that reached step 5. Observed
sixth-step emitters among them: `9`.

| rule | predicted emitters | misclassified |
|---|---:|---:|
| **P1** the plateau carries on | `12` | `3` |
| **P2** `ratio(5) > 2.168689` | `12` | `3` |

`H5` **FALSIFIED**, `H6` **FALSIFIED**. Both rules made *identical* predictions
and misclassified the *same three* records. As recorded in §10, this was a
degenerate test: the step-5 ratios of the twelve survivors span
`[2.5922, 13.1850]`, every one of them far above `theta`.

### Why the ratio could not have caught them

Post-hoc, and labelled as such — this scan is a follow-up question, not the
registered test, and it is not used to revise `H5` or `H6`:

```text
best threshold on the step-5 ratio: misclassifies 1/12
lowest step-5 ratio among SURVIVORS: 2.6846
step-5 ratios of the DROP-OUTS     : 2.5922, 3.1312, 3.2501
```

Two of the three records that stopped had step-5 ratios **above** the lowest
surviving record. No threshold on this feature can separate them. The
`sigma_min / E_Q` criterion is therefore **descriptive of the sealed five-step
past but not predictive of the sixth step** — and the reason is structural, not a
matter of tuning: `H6` of the census already showed the ratio is nearly a
record-level constant (mean CV `0.018`), so it carries almost no information
about *when* a record stops.

## 16. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` horizon inert | **PASS** (`117/117`, `0` mismatches) |
| `H2` sixth step certifiable | **PASS** (`9` emissions) |
| `H3` sixth step valid | **PASS** (`9/9` non-degrading, strictly improving) |
| `H4` no certificate violations | **PASS** (`0` over all six steps) |
| `H5` P1 plateau rule | **FALSIFIED** (`3/12` misclassified) |
| `H6` P2 beats P1 | **FALSIFIED** (degenerate: `12` vs `12` predicted) |
| `H7` attrition `n6 < n5` | **PASS** (`9 < 12`) |
| `H7b` plateau `n6 == n5`, same set | **FALSIFIED** |
| `H8` mean gain decays | **PASS** (`0.167057 < 0.361474`) |
| `H9` non-vacuity floor `0.05` | **FALSIFIED** (`min6 = 0.019347`) |
| `H10` set-difference reporting | reported in §13 |

Construction checks: **PASS**. Falsified hypotheses are results, and the analyzer
now reports them on a separate line so that a falsified prediction can never be
read as a broken construction.

---

# What the combined task establishes

1. **Why these records emit** (`FP-CENSUS-001`). The ratio
   `sigma_min / E_Q` is a strong ordinal signal at step 1 — the best threshold
   misclassifies `1` of `48` (`2.1%`) — but not a separator, and it is therefore
   never called a classifier here. The separation is carried by the **spread**
   (`2.6×`), not by the certified error (`1.2×`), and **every** abstention in the
   entire census carries the same frozen reason, `improvement_lcb_nonpositive`:
   the certificate never fails, the improvement budget does.
2. **How far it reaches** (`FP-ITER6-001`). The plateau at `12` was not a fixed
   point. Step 6 emits `9`, and the surviving margin has fallen to `0.019`,
   below the floor that steps 4 and 5 cleared.
3. **When the criterion stops working.** It predicts the first step well and the
   sixth step not at all, and the reason is measurable: the ratio is nearly
   constant along a trajectory, so it cannot say when a record stops.

## Known gaps, stated plainly

- **Single actor.** All checks are same-actor derived verification. The user's
  standing instruction is that verification is not the focus of this round.
  Independent verification of this line remains outstanding, as it does for every
  result before it.
- **The network path is one step behind.** The sixth step is numpy only.
- **The `H6` prediction test was degenerate** and was not caught when the
  prediction was written. It is recorded as a defect in the design.
- **The `H4` phrasing in the task sheet is universal, the plan's operationalisation
  is a majority.** Both readings are reported.
- **`theta` applied across steps is a deliberate stretch**: a step-1 threshold was
  applied to step-5 ratios knowing `H4` predicts the ratio decays. That is why it
  was registered, and why its failure is informative rather than surprising.
- **The post-hoc step-5 scan is not pre-registered** and must not be quoted as if
  it were.

