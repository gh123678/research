# FP-SAMPLE-001 first result

Date: 2026-09-12.
Branch: `claude/FP-CENSUS-001`. Baseline: `15a1511`.
Actor: Claude, under the user's instruction of 2026-09-12 ("都做").

## 1. What was run

```text
python -B fp_sample_vectorised_batch.py                 # the mandatory sampler gate
python -B evaluate_fp_sample_001.py --tasks 12 --mixings 0.08,0.5 \
    --label certification-size-ladder --output-dir results/FP-SAMPLE-001/claude/formal
python -B analyze_fp_sample_001.py
```

Step 1 of the frozen `pi_0`, all `48` route-records, certification rungs
`1x, 2x, 4x, 8x`. The decision rule, the eta grid and the sealed modules are
untouched.

## 2. `H1` gate: the vectorised sampler is unbiased

The sealed generator loops per chain in Python; at `8x` that is `8.4M` items per
record and hours of CPU — an artifact of the loop, not of the science.

| record | pair-share TV distance | `E[Y^2]` sealed vs fast |
|---|---:|---:|
| `0.08/0` | `0.0027` | `0.08183` vs `0.08257` (`+0.9%`) |
| `0.08/5` | `0.0033` | `0.07583` vs `0.07590` (`+0.1%`) |
| `0.5/0` | `0.0012` | `0.16292` vs `0.16275` (`+0.1%`) |
| `0.5/4` | `0.0025` | `0.25163` vs `0.25139` (`+0.1%`) |

Worst discrepancy `0.0091` against a registered gate of `0.02`. The sampler draws
the same law; it does **not** reproduce the sealed batch, and is never used where
the sealed batch is required. Total speedup roughly `200x` (`16 s` for eight
certificates and eight batches up to `8.4M` items).

## 3. The cost curve

| rung | items | mean `E_Q` (empirical Bernstein) | vs `1x` | mean `E_Q` (Bernstein) | vs `1x` |
|---|---:|---:|---:|---:|---:|
| `1x` | `1,048,576` | `0.1955` | — | `0.2063` | — |
| `2x` | `2,097,152` | `0.1339` | **`−31.5%`** | `0.1424` | `−31.0%` |
| `4x` | `4,194,304` | `0.1018` | **`−47.9%`** | `0.1075` | `−47.9%` |
| `8x` | `8,388,608` | `0.0821` | **`−58.0%`** | `0.0853` | `−58.7%` |

Both certificate arms move together, so the gain is the sample size and not an
interaction with which inequality is used.

## 4. The answer: the abstainers revive, and cheaply

Against `FP-TIGHT-001`'s frozen baseline of `26` abstaining route-records:

| rung | step-1 emitting | of the `26` frozen abstainers, revived |
|---|---:|---:|
| `1x` | `30` | `8` |
| `2x` | `35` | `13` |
| `4x` | `40` | **`18`** |
| `8x` | `44` | **`22`** |

`H6a` (a flip at `2x` or below), `H6b` (`≥ 16` of `26` at `8x` — actual `22`) and
`H6c` (non-decreasing: `8, 13, 18, 22`) all **PASS**.

`H2` holds at every rung for both arms: `0` coverage violations across `192`
certificate-fits, so the reduction is bought without weakening the guarantee.

**The direction of the programme is settled.** `FP-TIGHT-001` found the sample-size
lever worth `−64%` on a subset with no abstainer in it and could not test the
consequence. This task shows the consequence is real: **doubling the certification
data from `1x` to `2x` revives five abstaining route-records**, and `8x` revives
`22` of `26`.

## 5. `H5` FALSIFIED, and it is my anchoring error

I registered the `4x` reduction band as `[−70%, −50%]`, anchored on
`FP-TIGHT-001`'s subset measurement of `−64%`. Actual: `−47.9%`.

The reason is a **baseline mismatch in the prediction, not a scientific surprise**.
`FP-TIGHT-001`'s `−64%` was measured against the **frozen certificate**, so it
contains *both* levers — the `−20%` repair and the size effect. The ladder's
reductions are measured against its **own `1x` rung**, which is already the repaired
certificate, so they contain the size effect alone. Composing them
(`1 − (1−0.20)(1−0.479) = −58%` against frozen) is broadly consistent with the
`−64%` subset figure.

This is the third prediction in this line sunk by a mis-specified baseline rather
than by the data, and the pattern is worth naming: **when two numbers are compared,
state what each is measured against before writing the band.** The `H5` band is
recorded as falsified and not re-scored.

## 6. Diminishing returns in the bound, not yet in the outcome

`H7` **PASS**: the drop from `2x` to `4x` is `−0.0321`, and from `4x` to `8x` is
`−0.0197`, so the curve is flattening.

But the revival count is still climbing at the same rate (`13 → 18 → 22`). **The
bound is flattening before the outcome is**, because each further reduction moves a
*different* part of the population across the line: the first records to revive are
the high-spread ones, and the remaining ones sit closer to the threshold. A
flattening `E_Q` curve is therefore not yet a signal to stop buying data.

## 7. The four stragglers, and what they would cost

Four route-records never revive, even at `8x`:

```text
0.08/4 expected_exact   0.08/4 expected_finite
0.5/5  expected_exact   0.5/5  expected_finite
```

Fitting each record's own `eps_res = a + b/sqrt(N)` over its four rungs (the shape
the Bernstein radius predicts: a floor plus a `1/sqrt(N)` term):

| route-record | `sigma_min` | needs `E_Q ≤` | fit reproduces `8x`? | fitted floor | `N / 1x` needed |
|---|---:|---:|---|---:|---:|
| `0.08/4` exact | `0.1856` | `0.0856` | yes | `0.0115` | `17.2x` |
| `0.08/4` finite | `0.1856` | `0.0856` | yes | `0.0169` | `45.5x` |
| `0.5/5` exact | `0.2121` | `0.0978` | yes | `0.0252` | `160.4x` |
| `0.5/5` finite | `0.2121` | `0.0978` | yes | `0.0272` | `617.5x` |

**Read this carefully, because the large numbers do not mean what they look like.**
The fitted floor is the part of `eps_res` that data does *not* shrink. Where the
floor sits near the needed value — `0.0272` against `0.0293` for the last row — the
enormous multiplier is the fit's way of saying *the record is at its floor*. The
honest reading for those two is **"not reachable by sample size"**, not "buy 617x
the data". The first two, with floors well below what they need, are genuinely
priceable at `17x`–`46x`.

This is an **extrapolation**, and the table reports whether the fit even reproduces
the top rung it was fitted on. All four say yes, which makes the shape defensible
inside the measured range; it says nothing about `617x` beyond it.

### A bug caught while writing this

The first version of this extrapolation fitted the **population mean** `eps_res`
and applied it to individual records. That is simply wrong — a record's `E_Q` is not
the population mean — and it produced the visible absurdity of predicting a crossing
at `6.3x` for a record already measured at `8x` without crossing. The per-record fit
fixed it. The absurdity is worth recording because a result that contradicts a
measurement already in hand is the cheapest possible check, and it was the fit, not
the measurement, that was wrong.

## 8. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` sampler gate | **PASS** (worst `0.0091` vs gate `0.02`) |
| `H2` coverage at every rung | **PASS** (`0/192`, both arms) |
| `H3` agreement with the sealed `1x` | **PASS** (`0.1955` vs `0.1940`, `0.8%`) |
| `H4` monotone in the rung | **PASS** (strictly decreasing) |
| `H5` `4x` band | **FALSIFIED** (`−47.9%`; band mis-anchored, §5) |
| `H6a` cheap flip | **PASS** |
| `H6b` revival at `8x` | **PASS** (`22` of `26`) |
| `H6c` non-decreasing revival | **PASS** (`8, 13, 18, 22`) |
| `H7` flattening | **PASS** |

Construction checks **PASS**.

## 9. What this establishes, and what it does not

**Established**

- **Certification size is the lever that works.** `2x` revives `5` more abstaining
  route-records, `8x` revives `22` of `26`, and step-1 emissions go `22/48` (frozen)
  → `30` → `35` → `40` → `44`.
- The gain is sound: `0` coverage violations at every rung.
- The bound flattens before the outcome does.
- Four stragglers remain, and at least two of them appear to be at a floor that
  sample size cannot move.

**Not established**

- **Nothing about the iteration.** This is step 1 only. Whether `44` eligible
  records instead of `22` extends the certified iteration past six steps is the next
  question and is untested.
- **Size is not separated from realisation.** Every rung is a fresh independent
  sample drawn by a vectorised sampler, not a superset of the sealed batch, because
  the sampler consumes the stream differently. The ladder is monotone in size and
  internally consistent, but the confound is real. It works *against* a clean
  monotone curve, so it does not explain the one observed.
- **The straggler costs are extrapolations**, and `617x` means "at its floor".
- **The `1x` rung is not the sealed batch.** It agrees with `FP-TIGHT-001`'s sealed
  `1x` to `0.8%`, which validates the sampler, but it is a different sample.
- **Same-actor throughout**, per the user's standing instruction.
