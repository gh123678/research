# FP-ATTN-ITER6-001 first result

Date: 2026-09-12.
Branch: `claude/FP-CENSUS-001`.
Baseline: `0faffbd` (after the census and the numpy sixth step).
Actor: Claude, executing and checking under the user's standing "验证先不管" and the
scope instruction in which the user selected "the network path is one step behind"
from the list of outstanding work.

## 1. What was run

```text
# priced with a two-record smoke first: 2 records in 51.8 s
python -B evaluate_fp_attn_iter_001.py --tasks 1 --mixings 0.08,0.5 --max-steps 6 \
    --reference iter6 --output-dir results/FP-ATTN-ITER6-001/claude/smoke

# formal: 24 records, 8 min 25 s, bundle written 22:01:57
python -B evaluate_fp_attn_iter_001.py --tasks 12 --mixings 0.08,0.5 --max-steps 6 \
    --reference iter6 --label network-six-step \
    --output-dir results/FP-ATTN-ITER6-001/claude/network

python -B analyze_fp_attn_iter6_001.py
python -B verify_fp_attn_iter6_001_same_actor.py
```

Torch `2.11.0+cpu`. Frozen protocol, batches, certificate, decision rule and eta
grid are inherited verbatim from `FP-ITER5-001`. The one structural change is
`ALLOWED_MAX_STEPS` gaining the value `6`, plus the two new entries in
`REFERENCE_BUNDLES` so that a six-step network run can be compared against a
six-step numpy run.

## 2. The headline: the network is back in step with numpy

```text
emissions by step, network : [22, 20, 15, 12, 12, 9]
emissions by step, numpy   : [22, 20, 15, 12, 12, 9]
step-6 emitting sets       : 9 shared, 0 numpy-only, 0 network-only
```

The attention network carries a **sixth** certified step, it is the *same nine
route-records* the numpy path certified, and no record is gained or lost between
the paths. The two implementations are reconciled at the horizon where they were
previously one step apart.

## 3. `H1`: the horizon change is inert, gaps included

| comparison | result |
|---|---|
| sealed route-step rows, levels 1--5 | `117` |
| rows compared | `117/117` |
| decision / `eta` / `E_Q` / ordered-reason mismatches | `0` |
| **recorded `Qhat` gap mismatches** | **`0`** |

The gap check matters on its own. `q_hat_gap_vs_numpy` is a float32-arithmetic
artifact: a rerun whose arithmetic moved by one ulp anywhere would move it even if
every decision still agreed. All `117` sealed gaps are reproduced exactly, which is
a far stronger inertness statement than decision agreement alone.

## 4. Per-level outcome

| step | emissions | mean gain | minimum gain | max `\|dQ\|` vs numpy |
|---|---:|---:|---:|---:|
| 1 | `22` | `2.717707` | `0.104197` | `1.076e-05` |
| 2 | `20` | `2.277997` | `0.779902` | `4.585e-06` |
| 3 | `15` | `1.286905` | `0.378197` | `7.176e-06` |
| 4 | `12` | `0.807499` | `0.184902` | `1.044e-05` |
| 5 | `12` | `0.361474` | `0.082395` | `4.567e-06` |
| **6** | **`9`** | **`0.167057`** | **`0.019347`** | **`6.814e-06`** |

The `max |dQ|` column is aggregated over **every** row at each level, not only the
emitting ones, precisely so that levels 1--5 can be checked against the sealed
`FP-ITER5-001` headline figures: `1.076e-05, 4.585e-06, 7.176e-06, 1.044e-05,
4.567e-06`. They match exactly. An earlier draft of the analyzer aggregated only
emitting rows and reported `5.966e-06` at step 1; that was a different quantity and
not comparable, so it was corrected rather than quoted.

**The plateau break reproduces on the network path.** Emission deltas
`−2, −5, −3, 0, −3`; mean gain `0.361474 → 0.167057`; minimum gain
`0.082395 → 0.019347`, the same fall below the `0.05` floor that the numpy path
recorded.

## 5. Drift does not accumulate

```text
max |dQ| by step: 1.076e-05, 4.585e-06, 7.176e-06, 1.044e-05, 4.567e-06, 6.814e-06
```

Six compositions, all inside the frozen `ATOL = 1e-4`, and the sequence is flat
rather than growing — the sixth value is the *second smallest* of the six. `H6` was
registered as a flatness claim, not an extrapolation, and it holds. There is still
no evidence of `float32` accumulation across the certified iteration.

## 6. An honest detail: the mean gains differ in the sixth decimal

The network's step-3 and step-4 mean gains are `1.286905` and `0.807499` against
numpy's `1.286906` and `0.807500`. Minimum gains and every decision are identical.

This is expected and is not a disagreement about the science. The recorded value
gain is computed from `policy_plus = softmax-normalised π·exp(η·Qhat)`, so the
*policy itself* is a function of `Qhat`. The two paths select the same `eta` and
the same decision on every step, but their policies differ by the float32 gap
(≤ `1.076e-05`), so the oracle values of those policies differ at the same order.
Two consequences worth stating plainly:

- the per-step gain figures are **not** a path-agreement metric, and should not be
  read as one — the metric for that is the decision/`eta`/set comparison;
- any future claim of path agreement must be made on decisions and sets, which is
  how `H5`, `H7` and `H8` were formulated here.

## 7. Verdicts

| hypothesis | verdict | evidence |
|---|---|---|
| `H1` network horizon inert | **PASS** | `117/117` rows, `0` mismatches incl. gaps |
| `H2` sixth step certifiable | **PASS** | `9` emissions, all network-produced |
| `H3` sixth step valid | **PASS** | `9/9` non-degrading, strictly improving |
| `H4` no certificate violations | **PASS** | `0` over six steps; `0` at any level |
| `H5` path agreement at step 6 | **PASS** | identical sets, `9` shared, `0`/`0` |
| `H6` drift within `ATOL` | **PASS** | step-6 gap `6.814e-06 ≤ 1e-4` |
| `H7` no decision or `eta` flips | **PASS** | `0` and `0` over `129` step entries |
| `H8` count matches numpy | **PASS** | `9 == 9` |
| `H9` mean-gain decay | **PASS** | `0.167057 < 0.361474` |

Construction checks **PASS**; hypotheses are reported on a separate line from
construction checks, so a falsified prediction could never be read as a broken
build.

## 8. What this closes, and what it does not

**Closed**: the gap that `FP-ITER6-001` recorded. The line's headline claim — a
fixed-weight softmax attention network supporting certified, non-degrading policy
improvement — now rests on **six** steps on both implementations, with the paths
agreeing on every decision and every emitting set.

**Not closed:**

- **Single actor.** All checks remain same-actor derived verification. Independent
  verification of this entire line is still outstanding, as it is for every result
  before it.
- **`float32` is not proven safe in general.** Six flat gaps on four-state,
  three-action toy MDPs is evidence of no accumulation *here*, not a bound.
- **The sixth-step margin is thin.** `min6 = 0.019347`, already below the floor
  steps 4 and 5 cleared, on both paths.
- **The eligibility criterion still does not predict the break.** The census
  (`FP-CENSUS-001`) showed the step-5 ratio cannot separate the three records that
  stopped; running the network sixth step does not change that.
