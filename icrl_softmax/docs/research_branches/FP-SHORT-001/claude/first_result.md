# FP-SHORT-001 first result: records stop narrowly, and one state vetoes

Date: 2026-09-12.
Branch: `claude/FP-CENSUS-001`. Baseline: `32559ac`.
Actor: Claude, under the user's instruction of 2026-09-12 ("去做").

## 1. What was measured

`FP-GAP-001` found that records emitting only `1`–`5` steps leave a mean of `54.89%`
of their initial suboptimality uncollected, while long trajectories close `99.97%`.
This task asked **what specifically fails** at the stopping step, and separated three
candidate mechanisms. Both arms use the sealed certificate and the same `8x`
certification, so the `eta` grid is the only difference between them.

## 2. The regression checks hold

`H1` **PASS**: the instrumented decision path matches `fs.improvement_for` — status and
selected `eta` — at **all `428` steps**, and the frozen-grid arm reproduces
`FP-ITER8X-001`'s sealed `frozen` arm **step for step with `0` divergences**. The
diagnosis is therefore about the frozen rule, not about a re-implementation.

`H2` **PASS**: the extended grid changes **no** emitting decision anywhere. This was
registered as a theorem (the frozen values are a descending prefix, so the scan stops
before the additions) and it holds.

## 3. The answer: the records stop narrowly, and exactly one state vetoes

| route-record | emitted | `E_Q` | `h` | `E_Q / h` | blocking states | smallest passing `eta` |
|---|---:|---:|---:|---:|---:|---|
| `0.5/7` exact | `2` | `0.12541` | `0.12462` | **`1.006`** | `1` | none |
| `0.5/7` finite | `2` | `0.12725` | `0.12714` | **`1.001`** | `1` | none |
| `0.5/8` exact | `5` | `0.12254` | `0.12235` | `1.002` | `1` | none |
| `0.5/1` exact | `4` | `0.14235` | `0.13535` | `1.052` | `1` | none |
| `0.5/1` finite | `4` | `0.14548` | `0.13496` | `1.078` | `1` | none |
| `0.5/8` finite | `5` | `0.13288` | `0.12228` | `1.087` | `1` | none |
| `0.5/9` finite | `0` | `0.07104` | `0.06337` | `1.121` | `1` | none |
| `0.5/9` exact | `0` | `0.07283` | `0.06303` | `1.155` | `1` | none |
| `0.08/4` exact | `0` | `0.12366` | `0.08487` | `1.457` | `1` | none |
| `0.08/4` finite | `0` | `0.14200` | `0.08517` | `1.667` | `1` | none |
| `0.5/5` exact | `0` | `0.16373` | `0.09821` | `1.667` | `1` | none |
| `0.5/5` finite | `0` | `0.17062` | `0.09811` | `1.739` | `1` | none |

`h` is the **exact gate margin**: `h = max_eta min_s I_s / ‖Δπ_s‖₁`, the largest
certified error for which some candidate would pass. Emission is exactly `E_Q < h`, so
`E_Q / h` is the whole gate as one number.

- `H3` **PASS**: `E_Q / h` lies in `[1, 2]` for **`12/12`** short trajectories.
  Median `1.121`, and **two cases sit at `1.001` and `1.006`** — touching the line.
- `H4` **PASS**: **exactly one** state blocks in **`12/12`** cases. The blocking-state
  distribution is `{1: 12}`, with no case of two, three or four.

## 4. `H5`/`H6` FALSIFIED: the `eta` grid is not the constraint

| | total emitted steps |
|---|---:|
| frozen grid | `402` |
| extended grid (100× smaller `eta` available) | `402` |
| change | **`+0.0%`** |

**Extending the grid downward by a factor of `100` — six additional candidates down to
`1e-4` — lengthens exactly zero trajectories.** So mechanism 3 is out: the best
candidate is already inside the frozen grid, and `h` is not maximised by a smaller
tilt.

Both falsifications are informative and were registered with thresholds so that
"the grid is not it" would be a verdict rather than an impression. The grid was the one
mechanism this line could fix without touching the certificate, so its elimination
removes the cheapest remedy and points at the other two.

## 5. What the two live mechanisms actually say

**The stop is narrow.** Median `E_Q / h = 1.121` means the typical short record needs
its certified error reduced by about `12%` to pass; the two tightest need `0.1%`. At
the scaling `FP-SAMPLE-001` measured (the radius falls roughly as `N^{-1/2}` plus a
`1/N` term), a `12%` reduction is well within reach of a modest further increase in
certification data — `FP-SAMPLE-001` got `−31.5%` going from `1x` to `2x`.

**Exactly one state vetoes.** The rule requires `min_s LB_s > 0` — a conjunction over
`4` states — and in every one of the `12` cases a **single** state is the one that
fails, while the other three pass. That is a structural fact about the gate, and it
means the obstruction is not "the record cannot be improved" but "one state cannot be
improved **by the same policy tilt that the other three want**".

Those two readings suggest different remedies, and the data separates them: more data
buys a `12%` error reduction for the whole record, while a per-state treatment would
attack the veto directly. **This task does not choose between them** — it was scoped to
diagnose, and both are protocol-level changes that would need their own task.

## 6. Verdicts

| hypothesis | verdict |
|---|---|
| `H1` instrumented path is the frozen rule | **PASS** (`428/428`, and `0` divergences from the sealed bundle) |
| `H2` extension is inert where it should be | **PASS** (`0` counterexamples) |
| `H3` records stop narrowly | **PASS** (`12/12` in `[1, 2]`) |
| `H4` exactly one state blocks | **PASS** (`12/12`) |
| `H5` the grid is the binding constraint | **FALSIFIED** (`0` trajectories lengthened) |
| `H6` the extension is worth having | **FALSIFIED** (`+0.0%`) |

Construction checks **PASS**.

## 7. What this settles, and what it leaves

**Settled.** The early stops are not a grid artifact and not a gross certificate
failure. They are **narrow misses** — typically `12%`, sometimes `0.1%` — caused by a
**single state** failing a conjunctive gate that the other three states pass.

**Left open, and this task deliberately does not decide:**

- **Whether more data is the remedy.** The arithmetic says a `12%` reduction is
  reachable; measuring it needs a longer certification ladder than `FP-SAMPLE-001`
  ran, or a higher multiplier than `8x`. That is a protocol change with a compute
  cost.
- **Whether a per-state treatment is sound.** Updating only the states whose `LB_s > 0`
  and leaving the vetoing state unchanged would still be componentwise
  non-degrading — but it is a **change to the decision rule**, which is frozen, and it
  would need its own derivation, its own guarantee and its own inertness argument.
- **Whether the same one-state veto operates on the network path**, where the trust
  horizon of `FP-ATTN-8X-001` limits how far the comparison can run.
- **Same-actor throughout**, per the user's standing instruction.
