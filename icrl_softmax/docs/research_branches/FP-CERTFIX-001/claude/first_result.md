# FP-CERTFIX-001 first result: valid certificates, and what validity costs

> **ERRATUM 2026-09-13 (independent review, OBJECTION).** The runs below used the
> first-n extraction whose bridge lemma (lemma A) was **proved false** by the
> independent review (`review_of_derivation.md` in this directory, exact
> counterexample). **Every guarantee statement in this file tied to that protocol is
> withdrawn.** The measurements (emission counts, zero violations, agreement checks)
> remain correct as measurements. The repaired protocol (lemma A': one first-visit
> sample per independent chain) is implemented in `fixed_policy_mp_certificate.py`
> (`*_firstvisit`) and re-run; see the v2 section appended below and the derivation's
> errata banner.

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001` (task executed under the user's direct 2026-09-13
authorization covering phases 1–4 of the 2026-09-13 GPT review plan; baseline
`a2df7f8`).
Actor: Claude (governance exception recorded in the task file).
Task: [FP-CERTFIX-001](../../research_tasks/FP-CERTFIX-001.md).
Derivation: [FP-CERTFIX-001-certificate-rederivation.md](../../derivations/FP-CERTFIX-001-certificate-rederivation.md).

**All results here are preliminary** (single-actor; independent derivation review and
independent re-run remain open, exactly as before this task).

## 1. What was broken and what this task did about it

| defect (evidence in `2026-09-13-review-audit-claude.md`) | fix | where |
|---|---|---|
| frozen mean step uses estimated scale as Hoeffding range | replaced by Maurer-Pontil with the KNOWN range `2E = 20` | derivation §2, `mp_certificate` |
| Bernstein arm's second-moment slack missing `√2` (too TIGHT, weaker than claimed) | split arm restores the correct `E²·sqrt(log(1/δ₁)/(2m))` | derivation §7, `split_bernstein_certificate` |
| per-pair sample size decided by realised visit counts, then halved | first-`n` protocol, `n` pre-registered; abstain if any pair short | lemma A, `fp_certfix_first_n.py` |
| one certification batch reused for 12 steps while the batch selected the policies | fresh independent batch every step, step-indexed seed schedule, `δ_k = 0.05/K` | theorem 2, `evaluate_fp_certfix_001.py` |

The literature alignment the review asked for: Maurer-Pontil Theorem 4 (iid, fixed
`n`, bounded) is now used exactly on its premises — iid+fixed-n come from lemma A
(first-`n` stopping-time argument), boundedness from the `divergence_guard`-enforced
`‖Q̂‖∞ ≤ B` and the known reward bound. The theorem statement was re-checked against
[the paper](https://arxiv.org/abs/0907.3740) (constants `√2` inside the root,
`7/3 (n−1)` linear term, `ln(2/δ)`).

## 2. Runs and verification

| rung | mode | n per pair | items drawn | verify |
|---|---|---:|---:|---|
| `smoke` | 2 records × 3 steps | 16,384 | 25.2M | PASS |
| `step1_n16k` | 48 records, step 1, δ=0.05 | 16,384 | 100.7M | PASS |
| `step1_n64k` | 48 records, step 1, δ=0.05 | 65,536 | 402.7M | PASS |
| `multi_n16k` | 48 records, K=12, δ_k=0.05/12 | 16,384 per step | 170.9M | PASS |

`verify_fp_certfix_001.py` recomputes from each sealed `task_results.json`: emission
counts three ways, risk accounting (`2d·δ_each = δ_step`, `K·δ_step = 0.05`), the MP
radius formula from sealed per-pair sample variances (188 steps checked, max
|Δ| < 1e-12), oracle audits, abstention reasons. All PASS, `failure_count = 0`.

## 3. Step-1 ladder: what the honest certificate emits

| protocol | per-pair data | first-step emitted |
|---|---:|---:|
| sealed frozen @ 1x (invalid mean step) | ~43k halves | 22 |
| sealed empB @ 1x (missing MP correction in RANGE arm; here: corrected) | ~43k halves | 30 |
| **mp @ n=16,384** | 16,384 | **28** |
| **split @ n=16,384** (minimal repair) | 16,384 | **13** |
| **mp @ n=65,536** | 65,536 | **36** |
| **split @ n=65,536** | 65,536 | **32** |
| sealed frozen @ 8x (invalid) | ~350k halves | 42 |
| sealed empB @ 8x (defects above) | ~350k halves | 44 |

The MP arm at n=16,384 emits 28/48 with **less than half** the per-pair data of the
sealed 1x runs — validity did not cost power at step 1; the old empB arm's 30/48 was
bought with a bound whose premises did not hold. Two environments (`0.5/7`,
`0.5/10`) abstain at n=65,536 because their minimum pair counts (51,189 / 59,505)
fall short of the pre-registered `n` — the protocol refusing rather than silently
using a random count is the intended behavior.

## 4. Multi-step under the trajectory-level guarantee

48 records, frozen horizon K=12, `δ_k = 0.05/12` per step, n=16,384 per pair per
step, fresh batch every step. mp arm per-step emissions:

```text
mp    : [23, 22, 22, 12, 12, 8, 5, 4, 3, 2, 2, 2]   total 117, still emitting 2 at step 12
split : [10, 10, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2]      total 45
```

- 0 certificate violations, 0 componentwise degradations, every abstention reasoned.
- mp mean total value gain per trajectory `3.415471`; split `1.403394`.
- 25/48 (mp) and 38/48 (split) trajectories never emit at all under this budget.
- The mp step-6/10/12 minimum realized gains (`0.051690`, `3.738e-03`, `6.487e-04`)
  coincide with the sealed ITER8X values — same dominant trajectories.

**Honest cost accounting.** The sealed 8x protocol reused one 8.4M-item batch across
12 steps (8.4M items per record total). This protocol drew 170.9M items for its 163
actually-simulated record-steps (~1M per record-step, i.e. ~12x per record if all
ran to 12) — and that is with per-pair n twenty times smaller than the 8x halves.
**A valid trajectory-level guarantee costs roughly an order of magnitude more data
than the invalid reuse protocol at comparable tightness**, plus the risk split
(`δ/12` per step tightens every step's log term slightly). This cost is a primary
result, not a footnote.

## 5. Verdicts on the task hypotheses

| hypothesis | verdict |
|---|---|
| H1 bridge lemma (first-n fixed-count iid) | derivation complete; implementation enforces exact counts (refuses otherwise) — **awaits independent derivation review** |
| H2 zero coverage violations | PASS as failure-detector (0 violations everywhere); NOT evidence of validity, per the task's own warning |
| H3 validity changes judgments | PASS as measurement: mp emits 28/48 at n=16k, 36/48 at n=64k; split 13/48 and 32/48; 25/48 mp trajectories never emit at δ/12 per step |
| H4 risk accounting exact | PASS (verified by re-adding every sealed step) |

## 6. What this does NOT establish

- The derivation has **not** been independently reviewed; one-actor risk is highest
  exactly here.
- The network path is untested under the new protocol (this run used the two numpy
  routes only) — Phase 3.
- Near-zero `LB` decisions have not had their high-precision recheck — Phase 3.
- All environments remain the 24 frozen ones — Phase 4.
- Whether the per-state remedy (update only `LB_s > 0` states) is sound — Phase 4.

## 7. Reproduce

```text
python evaluate_fp_certfix_001.py --output-dir results/FP-CERTFIX-001/claude/step1_n16k \
    --label step1_n16k --mode step1 --n-per-pair 16384 --chains 16384
python evaluate_fp_certfix_001.py --output-dir results/FP-CERTFIX-001/claude/step1_n64k \
    --label step1_n64k --mode step1 --n-per-pair 65536 --chains 65536
python evaluate_fp_certfix_001.py --output-dir results/FP-CERTFIX-001/claude/multi_n16k \
    --label multi_n16k --mode multi --n-per-pair 16384 --chains 16384 --max-steps 12
python verify_fp_certfix_001.py --results results/FP-CERTFIX-001/claude/<rung>
```

Versions and file hashes: each rung's `config.json` (sealed-file and new-file
SHA256). Code: commit `66fd826` on `claude/FP-CENSUS-001`.

---

## v2: results under the REPAIRED protocol (lemma A', first-visit sampling)

After the independent review proved lemma A false, the protocol was repaired
(chain-replicated first-visit sample; random per-pair `N_x` independent of values;
abstain if `N_x < 2000`) and everything was re-run. **Only this section carries
guarantee language.** The repaired step-1 ladder and multi run:

| rung | protocol | step-1 emitted | total steps | violations |
|---|---|---:|---:|---:|
| `step1_fv_c16k` | 16,384 chains (N_x ≈ 9k–16k/pair) | mp **22/48**, split 6/48 | — | 0 |
| `step1_fv_c64k` | 65,536 chains (N_x ≈ 36k–64k/pair) | mp **38/48**, split 32/48 | — | 0 |
| `multi_fv` | 16,384 chains/step, fresh each step, K=12, δ_k=0.05/12 | mp 16/48 | mp **76**, split 8 | 0 |

Read against the withdrawn table in §3, with the per-rung attribution the second
independent review (§4 of `review2_of_repair.md`) insisted on:

- at **16k chains** the first-visit arm retains `N_x ≈ 9k–16k` per pair, strictly
  below the withdrawn protocol's fixed 16,384, and the radius grows: **22 vs 28 is
  the real sample-size cost** of a valid bridge;
- at **64k chains** the valid arm emits **more** than the withdrawn one (38 vs 36).
  That difference is **not** a validity effect: the withdrawn run abstained on 2
  records (`mix=0.5, task=10`, both routes) for insufficient per-pair counts, while
  the first-visit arm counts chains and clears its threshold there. So
  "validity costs power" holds at 16k and reverses at 64k for an unrelated reason;
  the corrected statement is: **the cost of validity is a smaller per-pair sample
  at fixed chain count, worth about 6 emissions at 16k, and nothing at 64k.**

The valid certificate is looser than the withdrawn one at equal chain count — that
is the price of the guarantee being true — but it is far from vacuous, and the
split (minimal-repair) arm's collapse at 16k chains (6/48) quantifies exactly how
much the MP single-sample construction buys over the minimal repair.

All three rungs pass `verify_fp_certfix_001.py` (counts three ways, risk accounting
`2d·δ_each = δ_step`, MP radius recomputed from sealed per-pair variances AND
per-pair sizes, zero violations, zero degradations). The guarantee now rests on
lemma A' + theorem 2 as re-issued in the derivation; the review confirmed theorem
2's machinery unchanged.

---

## v3: the second confirmatory arm (oracle kernel), registered by user ruling

User ruling 2026-09-13 (option A) on the independent review #2 turned the task
sheet's **pre-registered fallback** into a first-class arm: `n` iid successors per
pair drawn straight from `P(·|s,a)` (`fp_sample_vectorised_batch.kernel_batch`,
`--extraction oracle_kernel`). It is a different sampling mechanism for the *same*
estimand (the residual law is `P_x` either way), so it is an independent
construction for lemma A' rather than a competing protocol. It reads the
transition kernel, so it corroborates but cannot replace the first-visit arm for
claims about learning from experience.

| rung | per-pair data | step-1 emitted (mp) | step-1 emitted (split) | violations |
|---|---|---:|---:|---:|
| `step1_ok_c64k` | 65,536 iid draws/pair | **41/48** | 34/48 | 0 |

At the same nominal per-pair budget the oracle arm emits 41/48 against the
first-visit arm's 38/48: expected, because its per-pair count is *fixed* at
65,536 whereas the chains retain `N_x ∈ [36k, 64k]`, and its batch has no
chain-internal correlation. It also draws **37.7M items for the whole step-1
ladder versus 402.7M for the chain protocol — 10.7× less**, since it never
simulates trajectories it will not use.

The review's decision-level comparison (different seed stream, so not the sealed
rung): **45/48 decisions agree**, and all 3 disagreements are
*first-visit abstains / oracle emits* — never the reverse. A biased first-visit
sample would have produced the reverse direction, so this is the evidence the
ruling asked for.

Both readings of the new rung pass: `verify_fp_certfix_001.py` (PASS,
`failure_count = 0`, 48 mp steps checked) and the reviewer's independent replay
(`review2_replay_sealed.py`, zero discrepancies on all 9 rungs).

**Standing caveat.** Review #1, review #2, the lemma A' proposal and both arms
were produced by the same actor. The ruling closes the *protocol* question, not
the *independence* question: every number in this file remains a preliminary
result until a third party re-derives or re-runs it.

---

## v3a: cross-check of the data-volume comparison, and the K=12 oracle rung

*(Appended by the executing session after reading v3; the text above is the
reviewer's and is left intact.)*

**One figure in v3 does not match the sealed bundles.** v3 compares "37.7M items
versus 402.7M — 10.7× less". The 402,653,184 items belong to `step1_n64k`, the
**withdrawn first-n rung**. The primary arm's corresponding rung is
`step1_fv_c64k` with **201,326,592** items. Sealed values:

| rung | extraction | items drawn | step-1 mp |
|---|---|---:|---:|
| `step1_fv_c16k` | first_visit (primary) | 50,331,648 | 22/48 |
| `step1_fv_c64k` | first_visit (primary) | 201,326,592 | 38/48 |
| `step1_oracle_n64k` | oracle_kernel (confirmation) | 37,748,736 | 41/48 |
| `step1_ok_c64k` | oracle_kernel (confirmation) | 37,748,736 | 41/48 |
| `step1_n16k` | first_n (**withdrawn**) | 100,663,296 | 28/48 |
| `step1_n64k` | first_n (**withdrawn**) | 402,653,184 | 36/48 |

So the honest saving of the confirmation arm over the **primary** arm at matched
nominal per-pair budget is **5.33×** (201.3M / 37.7M), not 10.7×; the larger ratio
only appears if the oracle arm is compared against a withdrawn rung.

**The K=12 oracle rung (this session).** `multi_oracle_n64k`, sealed and verified
(PASS, `failure_count = 0`, 371 mp radii recomputed):

| arm | step-1 emitted | total emitted steps (K=12) | items drawn |
|---|---:|---:|---:|
| oracle mp | 39/48 | **338** | 293,339,136 |
| oracle split | 32/48 | 237 | — |
| first_visit mp (primary) | 16/48 | 76 | 127,926,272 |
| first_visit split (primary) | 2/48 | 8 | — |

The oracle arm reaches 338 emitted steps against the primary arm's 76 at a
comparable data volume, because its per-pair count is fixed at 65,536 while the
chain protocol retains only the visiting chains. That difference is a
**data-access** effect (kernel versus chains), not a validity effect: both arms
estimate the same residual law, and the review's decision comparison shows the
disagreements run only in the direction a biased first-visit sample would not
produce.

**High-precision replay of this rung** (`hp_oracle_n64k`, float64 / `math.fsum` /
mpmath-50 over the rung's decision points): **646 decision points, 0 replay
mismatches, 0 flips**; 96 tight/abstain points also recomputed at 50 digits.
Smallest emitted margin **6.41e-10** — three orders above the measured rounding
envelope (~9e-16), so unlike the tail of the K=32 primary run, no oracle-arm
decision approaches the arithmetic floor.
