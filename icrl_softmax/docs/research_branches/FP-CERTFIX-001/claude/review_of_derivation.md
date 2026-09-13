# Independent review of FP-CERTFIX-001's derivation

Date: 2026-09-13.
Reviewer: the **DSH session in the main tree** — a different session from the one that
wrote the derivation and its implementation, and a different toolchain. Asked to review
by the user.

**Scope.** `docs/derivations/FP-CERTFIX-001-certificate-rederivation.md`, with Lemma A
and Theorem 2 named by the user as the core. I read the derivation, checked every lemma
on its own terms, and tested Lemma A computationally rather than by inspection.

**What this review is not.** Both sessions are the same model, so a shared conceptual
error would survive. This is a second *reading* and a second *implementation check*,
not independent verification by a different actor.

**Verdict: `OBJECTION`.** Lemma A is **false**, and Theorems 1 and 2 rest on it. The
remaining lemmas are correct. A correct repair is available and is given in §4 below.

---

## 1. Lemma A is false

**The claim.** Retaining the first `n` visits to a pair `x` within a chain, and
conditioning on `{N_x >= n}`, yields an iid `P_x^{⊗n}` sample.

**The counterexample.** States `{s, u, v}`; the pair of interest is `x = (s, a0)`.

```text
from s : successor u with probability 1/2, v with probability 1/2
from u : successor s with probability 1
from v : successor v with probability 1        (x is never visited again)
```

with the residual `Y = g(successor)` injective on `{u, v}`, so the residual reveals the
successor. Then

- `Y_1 = g(u)` implies the chain returns to `s`, so `τ_2 < ∞`;
- `Y_1 = g(v)` implies the chain sits at `v` forever, so `τ_2 = ∞`.

Hence `{N_x ≥ 2} = {Y_1 = g(u)}` exactly, and conditional on that event `Y_1` is
**degenerate**, not `P_x`.

**Measured**, `400,000` chains of length `40`, seed `20260913`
(`tmp/review_lemma_a.py`, read-only, writes nothing):

| quantity | Lemma A predicts | measured |
|---|---|---|
| `P(N_x ≥ 2)` | — | `0.4995` |
| `P(Y_1 = g(u) \| N_x ≥ 2)` | **`0.5000`** | **`1.0000`** |
| `P(Y_2 = g(u) \| N_x ≥ 2)` | `0.5000` | `0.5004` |

`Y_2` **is** `P_x`, which confirms the fresh-draw property (1) is correct. `Y_1` is not.
The conditional law is `δ_{g(u)} ⊗ P_x`, not `P_x^{⊗2}`.

This is not a tail case or a slow-mixing artifact: the dependence is exact and the
conditioning event is a deterministic function of the first sample.

## 2. Where the proof breaks, precisely

The step

```text
P(∩_{k<n} {Y_k ∈ B_k}, τ_n < ∞) = P_x(B_{n-1}) · P(∩_{k<n-1} {Y_k ∈ B_k}, τ_n < ∞)
```

does not follow. Written out, the step needs

```text
E[ 1{Y_{n-1} ∈ B_{n-1}} · 1{τ_n < ∞} | F_{τ_{n-1}} ] = P_x(B_{n-1}) · P(τ_n < ∞ | F_{τ_{n-1}}),
```

but the two factors are **dependent**. Property (1) licenses replacing
`1{Y_{n-1} ∈ B_{n-1}}` by `P_x(B_{n-1})` only on its own; the co-factor
`1{τ_n < ∞}` is **not** `F_{τ_{n-1}}`-measurable, because whether the chain returns to
`x` depends on the successor drawn at `τ_{n-1}` — that is, on `Y_{n-1}` itself. So the
indicator cannot be pulled out, and the product does not factor.

The `k = n` step of the proof is correct: `Y_1, …, Y_{n-1}` are `F_{τ_n}`-measurable
and `1{τ_n < ∞} ∈ F_{τ_n}`, so applying (1) at `τ_n` is legitimate. The error is
confined to the subsequent iteration, where the same argument is reused at `τ_{n-1}`
with an indicator that is no longer measurable.

## 3. A point in the document's favour, stated because it is easy to miss

§1 point 4 of the derivation **explicitly identifies this dependence**:

> `N_x` 依赖残差序列（后继决定下一步去哪里），所以"计数确定、程序确定"本身不蕴含条件独立

So the document knows that `N_x` is a function of the residual sequence, and it knows a
naive argument will not do. It then proposes a stopping-time argument as the bridge —
and the bridge is exactly what fails. The diagnosis is right and the repair does not
work; that is a narrower and more fixable failure than not having seen the problem.

## 4. A correct repair

Use **one sample per independent chain**, taken at that chain's **first** visit to `x`.

Why it works:

1. Condition on the first visit occurring. At `τ_1`, the successor is drawn fresh by
   (1), so `Y_1 | {τ_1 < ∞} ~ P_x` — this is the `k = 1` step of the document's own
   proof, which is valid, and the measurement above confirms it (`0.4995`).
2. Across chains the samples are independent, because the chains are.
3. The retained sample size `N = #{j : chain j visits x}` is a function of the chains'
   trajectories *up to* their first visits, hence **independent of the values**
   `{Y_j : chain j visits x}`.

So the retained collection is an iid `P_x` sample of **random** size `N`, with
`N ⊥ (Y_1, …, Y_N)`. Maurer–Pontil then applies for each `n`, and integrating over `N`
costs nothing:

```text
P(fail) = E[ P(fail | N) ] ≤ δ   since P(fail | N = n) ≤ δ for every n.
```

Under this repair the abstention reason becomes "fewer than `n` independent chains
visited `x`" rather than "the pair was visited fewer than `n` times".

**Cost, stated honestly.** The number of *chains* must satisfy `C · p_x >= n` where
`p_x` is the per-chain probability of visiting `x`, rather than the total visit count
satisfying `N_x >= n`. For the frozen configuration (`C = 16384` chains, `L = 64`, and
the sealed minimum pair count around `2e4` visits) the visit counts are of the same
order as the chain count, so this is a genuine but not prohibitive increase — and it is
the price of the guarantee being true.

**Not proposed**: keeping `n` visits per chain and correcting the dependence. That is a
regenerative-process problem with no cheap exact solution, and it is not needed.

## 5. What is correct

Checked in full, and these stand:

| item | verdict |
|---|---|
| §0 range `E = R* + γB + B = 10`, `2E = 20` | correct, and consistent with the sealed `ENVELOPE`/`Y_RANGE` |
| §0 "the law of `Y_x` does not depend on the behaviour policy" | correct |
| (1) the fresh-draw property | **correct** — independently confirmed above (`Y_2` measures `0.5004`) |
| §2 Lemma B (Maurer–Pontil, range-scaled) | correct as stated; the range scaling `R·(7/3)·log(2/δ)/(n−1)` is right |
| §2 the rejection of the old two-step construction | correct, and the reason is the right one: an upper bound on `E[Y²]` does not license using `sqrt(E[Y²])` as a Hoeffding range parameter |
| §3 Lemma C | correct; `T^π` is a `γ`-contraction in sup norm and the rearrangement is valid |
| §4 Lemma D | correct; the Hölder step is valid and the bound is uniform over candidates, so grid selection consumes no extra risk |
| §4 corollary (performance-difference identity) | correct; `A_s` is exactly the advantage of `π⁺` with respect to `Q^π`, and `min_s LB_s > 0 ⟹` componentwise strict improvement |
| §5 risk accounting `2d × δ_step/(2d) = δ_step` | correct |
| §6 the adaptive fix (a fresh batch per step) | **correct and important** — after one batch is reused across `K` steps, `π_{k−1}` is a function of that batch and the conditional-independence step genuinely fails. The per-step batch restores it, and the `H_{k−1}`-measurability argument is right |
| §6 "grid selection within the same step consumes no extra risk" | correct, by the uniformity in Lemma D |
| §6 the explicit statement that the guarantee is single-trajectory | correct and well done |
| §7 the `√2` diagnosis | **correct, and it is a bug in my own code** — see §6 below |
| §8 limitations | unusually honest; §8.2 in particular (zero violations is not evidence of validity) is exactly right and is a point this line had to learn the hard way |

## 5. This triggers the task's own pre-registered stop condition

The task sheet anticipated this outcome and wrote down what to do about it:

> 若桥梁引理在当前行为采样下无法证明（例如**停时论证不成立**），停止实现，回写推导缺口，
> 转入"预先固定每对 iid 直接采样（oracle 转移核）"的替代协议评估

The parenthetical case is the one that obtains. So `H1` is **not** adjudicated as
`PASS`, the bridge argument does not hold, and the pre-registered response is the
per-pair direct-sampling alternative.

**One constructive note on that fallback.** The task sheet's alternative samples
`s' ~ P(·|s,a)` directly from the transition kernel, which changes the data-access
claim — it needs the kernel, which an RL agent does not have. The repair in §4 above
instead stays entirely within **behaviour-chain data**: one sample per independent
chain at that chain's first visit to the pair. It is iid, it is unbiased, and it uses
nothing the frozen protocol does not already produce.

The two differ in what the guarantee is about, and the choice is the authors':

| repair | samples iid? | needs the kernel? | sample size |
|---|---|---|---|
| behaviour-chain, one per chain (§4) | yes | **no** | random, `N ⊥` values |
| pre-fixed per-pair direct sampling | yes | **yes** | fixed by construction |

If the point of the certificate is that it holds for a policy-improvement procedure
that only sees data, the first is the stronger statement; if the point is a clean
statistical baseline, the second is. Both are correct, and the review does not choose
between them.

### Per `AGENTS.md` §6, the authors fix their own work

I have not edited the derivation, the task sheet, the certificate modules or the sealed
bundles. This file is a review comment; the repair, the re-issued theorems and the
re-measurement belong to the session that owns `FP-CERTFIX-001`.

## 6. `§7`'s `√2` claim is correct, and it is my bug

The derivation states that in the earlier work the second-moment slack was written
`(E²/2)·sqrt(log(1/δ')/m)` where Hoeffding on `Z = Y² ∈ [0, E²]` requires
`(E²/2)·sqrt(2·log(1/δ')/m)` — too tight by `√2`.

I verified it. `fixed_policy_bernstein_certificate.py` line 263 (and line 386, the
same term inside `data_range_certificate`) reads

```python
slack = (ENVELOPE**2 / 2.0) * math.sqrt(math.log(1.0 / delta_each) / n_a)
```

while `fixed_policy_variance_certificate.py` line 171 reads

```python
slack = (envelope**2 / 2.0) * math.sqrt(2.0 * log_a / n_a)
```

Numerically, `E = 10`, `log(1/δ) = 4.6`, `N = 16000`:

| expression | value | ratio to correct |
|---|---:|---:|
| correct `E²·sqrt(log/(2N))` | `1.199631` | `1.000000` |
| sealed module | `1.199631` | **`1.000000`** |
| my module | `0.848268` | **`0.707107`** |

So **the sealed module is right and mine is too tight by exactly `√2`**, in the
two-second-moment-based arms only. The affected arms:

- `bernstein_certificate` — `FP-TIGHT-001`'s `H3` arm, published as `−15.0%`;
- `data_range_certificate` — `FP-RANGE-001`'s main arm.

Neither is a primary headline: `FP-TIGHT-001`'s substantive arm was
`empirical_bernstein` (the Maurer–Pontil form, which is correct), and `FP-RANGE-001`'s
conclusion was negative, which a *larger* `E_Q` can only reinforce. But the published
figures for those two arms are not consistent with a correct slack and must be
re-measured, and the module must be fixed. That is my work to do, not this review's.

## 7. What has to happen before the guarantee can be claimed

1. **Repair Lemma A** (§4 above) and re-issue Theorem 1 with the chain-replicated
   protocol. Until then, `‖Qhat − Q^π‖∞ ≤ E_Q` is *not* established at any stated
   confidence, and Theorem 2's union bound multiplies a per-step probability that has
   not been shown to be `δ_k`.
2. **Re-issue Theorem 2** on the repaired Theorem 1. Its own argument — fresh batch per
   step, `H_{k−1}`-measurability, union over `k` — is sound and does not need changing.
3. **State the randomized-sample-size consequence.** With the repair the sample size is
   random, and the certificate must carry it: report `N` per pair, and make the
   abstention condition `N < n` rather than "the pair was visited fewer than `n` times".
4. **Do not use the guarantee's failure to explain the experiments.** The empirical
   results (`FP-HORIZON-001`, `FP-GAP-001`, `FP-SHORT-001`) were produced under the old
   protocol and are unaffected by this objection — they are measurements, not
   applications of the theorem. The derivation's own §8.2 already says the zero-violation
   audits are not validity evidence, which keeps the two apart correctly.

## 8. Reviewer's summary

The document is unusually well constructed for this line: every inequality is named,
every premise is checked against the protocol, the risk accounting is auditable, and §8
states what is *not* covered with more candour than most published appendices. Lemma B,
C and D and the whole of Theorem 2's adaptive machinery are correct, and the `√2` catch
against my own module is a real defect found by reading code against its stated
inequality.

But the one lemma that carries the entire coverage claim is false, and the proof's own
text contains the observation that predicts why. Lemma A is not repairable as written;
it is replaceable by a chain-replicated protocol at a stated cost, and that replacement
is the shortest path from here to a guarantee that holds.
