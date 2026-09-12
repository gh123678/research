# FP-RANGE-001 first result: the envelope ceiling is an artifact

Date: 2026-09-12.
Branch: `claude/FP-CENSUS-001`. Baseline: `7c0123f`.
Actor: Claude, under the user's instruction of 2026-09-12 ("都做").

## 1. What was run

```text
python -B evaluate_fp_range_001.py --tasks 12 --mixings 0.08,0.5 \
    --label data-driven-range --output-dir results/FP-RANGE-001/claude/formal
python -B analyze_fp_range_001.py
```

Step 1 of the frozen `pi_0`, all `48` route-records, certification rungs
`1x, 2x, 4x, 8x`. One new function (`data_range_certificate`), sealed modules
untouched, decision rule frozen.

## 2. The construction, and its honest price

`FP-TIGHT-001` priced **deleting the envelope** at `−45%` and marked it unsound:
with the range gone nothing bounds the tails. The natural sound replacement is
**truncation** — cap `|Y|` at `tau`, apply Bernstein with range `2 tau`, pay for the
cap:

```text
|E[Y]| <= |mean_B(Y)|                     observed
        + (1/N) sum_{|Y_i|>tau} |Y_i|     the empirical tail mass, exact
        + sqrt(V_x * p)                   Cauchy-Schwarz, p = P(|Y| > tau)
        + bernstein(Var_B(Y'), 2 tau)     concentration of the capped variable
```

Two design choices carry the result and are justified rather than assumed:

- **`tau` comes from the other half** (`max|Y|` over half A). Half A is independent
  of half B, so every statement about half B holds conditionally on `tau` with no
  union bound over a threshold grid. Optimising `tau` on half B would cost more than
  it buys.
- **The tight tail bound is used.** The tail probability enters the bias under a
  square root, so its bound dominates. Hoeffding on the indicator costs
  `sqrt(log(1/δ)/(2N)) ≈ 0.014` at `N = 16369`, which makes the bias `0.15` — three
  times the whole frozen radius. The Maurer-Pontil range term
  `(7/3)log(2/δ)/(N−1) ≈ 9.4e-4` is far tighter here because the indicator's
  observed variance is essentially zero. The pilot measured `+155%` with Hoeffding
  and `+10.2%` with Maurer-Pontil. **The negative result below rests on the better
  bound**, so it is not an artifact of a weak choice.

## 3. The answer: the ceiling cannot be collected

| rung | items | `data_range` | vs frozen | `empirical_bernstein` | vs frozen | `data_range` emits |
|---|---:|---:|---:|---:|---:|---:|
| `1x` | `1,048,576` | `0.2587` | **`+6.9%`** | `0.1955` | `−19.2%` | `17` |
| `2x` | `2,097,152` | `0.1827` | `−24.5%` | `0.1339` | `−44.7%` | `30` |
| `4x` | `4,194,304` | `0.1369` | `−43.4%` | `0.1018` | `−57.9%` | `35` |
| `8x` | `8,388,608` | `0.1055` | `−56.4%` | `0.0821` | `−66.1%` | `42` |

Same-sample references at `1x`: frozen `0.2420`, unsound ceiling `0.1356`
(`−44.0%`).

**Paying honestly for the tails turns `−44.0%` into `+6.9%`.** The sound
data-driven range is **worse than the frozen certificate** at `1x`, and it loses to
the plain inequality repair at every rung. At `8x` it reaches `−56.4%` — which is
still behind `empirical_bernstein` at `8x` (`−66.1%`), so the ordering never flips.

`H1` holds throughout: **`0` coverage violations over `192` certificate-fits**. The
construction is sound. It is simply not an improvement.

## 4. Why, exactly: the tail is empty and bounding it still costs

The decomposition is completely uniform across all `48` route-records:

- **the empirical tail mass is exactly `0.00000` everywhere** — because `tau` is
  taken from half A, no half-B sample ever exceeds it;
- **the concentration term is small** (`0.0084`–`0.0359`), as expected once the
  range is `2 tau ≈ 2.5–7` instead of `20`;
- **the entire price is the Cauchy-Schwarz bias** `sqrt(V_x · p_ub) ≈ ` `0.0266`–
  `0.0542`, which is what remains after the empirical tail comes back empty.

So the certificate is paying to certify that a tail it never observes is small, and
the cheapest sound way to say that costs about as much as the entire frozen radius.

`H4` **PASS** (`48/48`): the tail price exceeds the concentration term on every
record, so the loss is not coming from the concentration side.

`H5` **FALSIFIED**, and it is my registered mechanism that was wrong. I predicted
the bias alone would *exceed* the frozen radius at the same pair; it does not
(`0/48`; mean ratio `0.735`, max `0.759`). The bias is **comparable to**, not larger
than, the frozen radius. The loss comes from the two terms being **added**, not from
either one dominating. Recorded as falsified, not reinterpreted.

`H6` **PASS**: the ordering is stable across the whole ladder —
`1x dr+7% vs eb−19%`, `2x −25% vs −45%`, `4x −43% vs −58%`, `8x −56% vs −66%`. More
data improves both levers and never reverses their order, which is what the theory
predicts: the bias decays as `1/sqrt(N)`, the same rate as the term truncation was
meant to remove.

## 5. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` soundness | **PASS** (`0/192`) |
| `H2` the range lever loses to the inequality lever | **PASS** |
| `H3` `1x` reduction in `[+5%, +25%]` | **PASS** (`+6.9%`) |
| `H4` the tail price exceeds the concentration term | **PASS** (`48/48`) |
| `H5` the bias alone exceeds the frozen radius | **FALSIFIED** (`0/48`, ratio `0.735`) |
| `H6` the ordering is stable at `8x` | **PASS** |

Construction checks **PASS**.

## 6. What this closes

**The `−45%` ceiling was an artifact of not paying for the tails.** There is no
sound route to it at any affordable sample size: the price is a
`sqrt(V_x · p)` term that decays as `1/sqrt(N)`, the same rate as the term it was
meant to remove, so the ordering between "truncate and pay" and "correct the
inequality" is stable rather than closing.

Combined with `FP-SAMPLE-001`, the accounting on this line is now settled:

| lever | sound? | worth |
|---|---|---|
| correct the mean-step inequality | yes | `−15%` |
| empirical Bernstein | yes | `−20%` |
| delete the envelope | **no** | `−45%` — unreachable, see above |
| `8x` certification data | yes | `−58%` (and `22` of `26` abstainers revived) |

**Sample size is the only large lever that is both sound and collectable.** It is a
cost lever, and `FP-SAMPLE-001` measured its price.

**Not established**: nothing about the iteration; step 1 only. And `H5`'s failure
means the *mechanism* by which truncation loses is additive rather than dominated by
one term, which is a refinement rather than a closed question — a different tail
construction might pay a smaller price, and this result does not rule that out.

**Same-actor throughout**, per the user's standing instruction.
