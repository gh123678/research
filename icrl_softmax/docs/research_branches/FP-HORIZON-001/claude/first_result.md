# FP-HORIZON-001 first result: the certified iteration does not stop

Date: 2026-09-12.
Branch: `claude/FP-CENSUS-001`. Baseline: `a95bbee`.
Actor: Claude, under the user's instruction of 2026-09-12 ("好").

## 1. What was run

```text
python -B evaluate_fp_iter8x_001.py --tasks 12 --mixings 0.08,0.5 \
    --max-steps 32 --multiplier 8 --task-id FP-HORIZON-001 \
    --label iteration-horizon-32 --output-dir results/FP-HORIZON-001/claude/formal
python -B analyze_fp_horizon_001.py
```

The `FP-ITER8X-001` configuration with the horizon raised from `12` to `32`, so the
iteration has room to stop on its own. Two arms, each with its own trajectory: the
sealed certificate on `8x` data, and the repaired certificate on `8x` data.

## 2. `H3` FALSIFIED: it is still going at step 32

**The certified iteration does not terminate within the horizon.** At step `32` both
arms are still emitting — `12` and `14` of `48` route-records — with the smallest
emitted gain at **`3.845e-07`**.

So the answer to "where does it stop" is: **not here, and not for any reason the
certificate provides.** It keeps finding componentwise non-degrading, strictly
positive, certified improvements indefinitely. `H6` **PASS** confirms those gains are
strictly positive rather than numerical noise, but `3.8e-7` is close enough to noise
that the distinction is technical rather than practical.

This is the outcome the task sheet flagged as the more interesting failure:

> `H3` failing means the iteration is still going at step `32` and has **not** been
> shown to terminate, which would be the more interesting outcome: a certified
> iteration that keeps finding tiny-but-valid improvements indefinitely, on margins
> near numerical dust.

## 3. The population moves in plateaus, not a taper

| step | `frozen` | `empB` |
|---|---:|---:|
| 1 | `42` | `44` |
| 6 | `36` | `38` |
| 12 | `22` | `26` |
| 13–20 | **`20` (eight steps)** | `24, 23, 22, 22, 22, 22, 22, 22` |
| 21–26 | **`18` (six steps)** | **`20` (six steps)** |
| 27–32 | `17, 14, 14, 12, 12, 12` | `19, 16, 16, 14, 14, 14` |

`H4` **PASS**: the population is non-increasing at every one of the 32 steps, with no
rises at all across both arms.

**The shape is a staircase, not a cliff and not a smooth taper.** Long plateaus — the
`frozen` arm holds `20` for eight consecutive steps and `18` for six — separated by
sudden drops.

That reframes the `1x` result. At `1x`, the population held `12` for two steps and
this line called it a plateau and went looking for a horizon artifact; it found one
by extending to `12`. Now, with 32 steps in view, **plateaus turn out to be the
normal shape of this process**, not an artifact of any particular horizon. The `1x`
plateau was real *as a plateau*; what was wrong was reading it as an endpoint.

## 4. Where the gains stop being meaningful

Not pre-registered, and labelled as such: "reaches 32" is only decision-relevant if
the gains are still worth having, so here is the reach at several floors.

| mean-gain floor | last step at or above it (`frozen`) | records emitting | (`empB`) | records |
|---|---:|---:|---:|---:|
| `0.05` | `10` | `27` | `10` | `29` |
| `0.01` | `16` | `20` | `16` | `22` |
| `1e-3` | `25` | `18` | `25` | `20` |
| `1e-4` | `32` | `12` | `32` | `14` |

The mean gain decays geometrically, roughly `×0.8` per step, and has not yet fallen
below `1e-4` at step `32`. The `0.05` floor that steps 4 and 5 cleared under `1x` is
last cleared at step **`10`**.

**This task does not choose a floor.** A gain of `3.8e-7` is certified, componentwise
non-degrading and strictly positive; whether that counts as "policy improvement" is a
definitional question the line has not settled, and picking a floor after seeing the
table would be exactly the post-hoc threshold selection the task sheet prohibits.
The table is offered so a reader can apply their own floor.

## 5. What the trajectory bought

`H8` — the value question, which the line had never measured.

| arm | total value gain per route-record (mean) | worst state (mean) | records improving **every** state |
|---|---:|---:|---:|
| `frozen@8x` | **`5.649893`** | `0.838256` | `42/48` |
| `empB@8x` | **`5.803923`** | `0.860150` | `44/48` |

Range of total gain across route-records: `[0.0, 12.568599]`. The records at `0.0` are
the ones that never emitted at all — the stragglers `FP-SAMPLE-001` identified as
sitting at a floor that sample size cannot move.

The verifier re-derives these from the recorded start and final per-state value
vectors, independently of the analyzer's per-step sums, and confirms that **no
record's start-to-final per-state value decreases** — the aggregate form of the
per-step non-degradation claim.

So the iteration is not merely long: it buys a mean total value gain of `5.65`–`5.80`
per route-record, with `42`–`44` of `48` records improved in **every** state.

## 6. Validity holds across the whole run

| check | `frozen@8x` | `empB@8x` |
|---|---:|---:|
| emitted steps | `751` | `827` |
| componentwise degrading | `0` | `0` |
| non-positive total gains | `0` | `0` |
| certificate violations | `0` | `0` |
| abstentions without a reason | `0` | `0` |

`H1`/`H2` **PASS**: `1,578` certified updates, none degrading, none violating the
bound that licenses them.

## 7. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` validity | **PASS** (`1,578` emitted steps, `0` degrading) |
| `H2` soundness | **PASS** (`0` certificate violations) |
| `H3` terminates before the horizon | **FALSIFIED** (still emitting at step `32`) |
| `H4` population non-increasing | **PASS** (no rises in either arm) |
| `H5` mean gain decays past twelve | **PASS** |
| `H6` every emitted gain strictly positive | **PASS** (smallest `3.845e-07`) |
| `H8` cumulative value gain | reported in §5 |

Construction checks **PASS**.

## 8. The certified horizon across this line

`2 → 3 → 4 → 5 → 6 → 12 → 32`, and the last three of those were **horizons**, not
findings. The sequence says more about where the experiments were stopped than about
the method. What the line can now say is narrower and firmer:

- the certified iteration reaches **at least 32 steps** at `8x` certification, on
  both the sealed and the repaired certificate;
- it does so while emitting `1,578` updates that are **all** valid;
- the population declines monotonically in a **staircase** of long plateaus;
- the per-step gains decay geometrically and are still above `1e-4` in the mean at
  step `32`, but the smallest is `3.8e-7`.

## 9. What this does not establish

- **That it ever stops.** At `8x` it has not, within 32 steps. Whether it converges,
  cycles, or continues indefinitely is open, and the next horizon would answer the
  same question one more time rather than a different one.
- **Whether the tail counts.** Steps beyond roughly `25` improve the mean value by
  less than `1e-3` per step. Whether those are "improvements" is a definition this
  line has not fixed, and it now matters.
- **Anything about the network path at `8x`.** The network is in sync only to the
  `1x` six-step run.
- **That size is separated from realisation.** The `8x` batch is a fresh independent
  sample, not a superset of the sealed one.
- **Same-actor throughout**, per the user's standing instruction.
