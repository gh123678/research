# FP-CERTCHECK-001 first result: network agrees step-for-step, and no sealed decision is arithmetic-dominated

> **ERRATUM 2026-09-13.** The underlying certificates came from the first-n protocol
> whose bridge lemma was falsified by independent review (see FP-CERTFIX-001's
> `review_of_derivation.md`). The network/numpy agreement and the high-precision
> recheck below are **protocol-agnostic measurements** (two producers through the same
> decision rule; arithmetic of the decision chain) and stand as measurements. Any
> reading of them as evidence about guarantee validity is withdrawn. Re-runs under the
> repaired lemma-A' protocol are reported in the v2 section appended below.

Date: 2026-09-13.
Branch: `claude/FP-CENSUS-001`（治理例外见任务单）。
Task: [FP-CERTCHECK-001](../../research_tasks/FP-CERTCHECK-001.md). Protocol and
certificates: FP-CERTFIX-001 (`docs/derivations/FP-CERTFIX-001-certificate-rederivation.md`).

**Preliminary results** (single-actor; independent review pending), as with the whole line.

## 1. What was asked

Phase 3 of the review plan: recompute under the corrected protocol with the network
and reference routes, and check near-zero decisions with high precision — answering
the question the withdrawn "step-17 / step-18" claims had asked with the wrong
quantities.

## 2. H1 — network route under the corrected protocol

48 records, K=12, `δ_k = 0.05/12`, n=16,384 per pair per step, fresh batch each
step; the literal attention networks produce `Qhat`; certificate, decision rule and
audits identical to the numpy run (`--producer network`).

| arm | numpy per-step emissions | network per-step emissions | decision divergences | eta divergences |
|---|---|---|---:|---:|
| mp | [23,22,22,12,12,8,5,4,3,2,2,2] | identical | **0** | **0** |
| split | [10,10,4,4,3,2,2,2,2,2,2,2] | identical | **0** | **0** |

Drift probe (`analyze_fp_certcheck_drift_001.py`, replaying every sealed step with
both producers on the same policy and train batch): worst sup-norm `q_hat` gap
**1.315e-05** across all 163 replayed record-steps, flat across steps (no
accumulation), within the frozen `1e-4` tolerance. As always: this excludes
implementation divergence, not shared conceptual error — the certificate's validity
rests on the derivation, not on this agreement.

## 3. H2/H3 — high-precision recheck of the decision chain

`analyze_fp_certcheck_hp_001.py` replays every one of the **246 sealed decision
points** of the FP-CERTFIX-001 multi run from the frozen seed schedule (regenerating
each step's training batch, certification batch and `q_hat`), and recomputes the
certificate and full eta-grid decision three ways: production float64, compensated
summation (`math.fsum`), and — for every tight (`|margin| < 1e-6`) or abstained point
(103 points) — mpmath at 50 digits.

| check | result |
|---|---|
| float64 replay vs seal (emission + eta) | **0 / 246 mismatches** |
| fsum recomputation vs float64 decision | **0 flips** |
| mpmath-50 vs float64 decision (103 tight points) | **0 flips** |
| support-shortfall abstentions reproduced | all confirmed (`heldout_pair_support_missing`) |
| max \|E_Q float64 − fsum\| | 2.2e-16 |
| max \|margin float64 − fsum\| (162 emitted) | 4.2e-17 |
| max \|margin float64 − mpmath50\| (tight) | **9.1e-16** |
| smallest emitted margin | 4.75e-10 |
| **headroom** | **5.2e5×** |

So within the measured 12 steps of this protocol, **no sealed decision is
arithmetic-dominated**, and the tightest one is five orders of magnitude above the
measured rounding envelope. This is the correct form of the answer the withdrawn
claims gestured at: it says nothing about steps beyond 12 (unmeasured), and it does
not use cross-quantity ratios as a proxy for the decision chain.

## 4. Verdicts

| hypothesis | verdict |
|---|---|
| H1 network step-for-step agreement, drift < 1e-4 | **PASS** (0 divergences, worst gap 1.315e-05, flat) |
| H2 fsum/mpmath recheck flips nothing | **PASS** (0/246, 0/103) |
| H3 smallest emitted margin ≥ 100× error envelope | **PASS** (5.2e5×) |

## 5. Boundaries

- 12 measured steps; nothing claimed beyond.
- The replay regenerates data from the frozen schedule, so this checks the decision
  chain's arithmetic, not the sampler's correctness (the sampler has its own
  FP-SAMPLE-001 gate).
- Both producers share the certificate and decision code; agreement does not
  validate the derivation. Independent review of the derivation is still open.

## 6. Artifacts

- `results/FP-CERTCHECK-001/claude/network_multi_n16k/` (network route, sealed)
- `results/FP-CERTCHECK-001/claude/drift/` (per-step gaps)
- `results/FP-CERTCHECK-001/claude/hp_recheck/` (`hp_recheck.json`,
  `hp_recheck_steps.json` with per-step pi/LB detail)
- Code: `analyze_fp_certcheck_hp_001.py`, `analyze_fp_certcheck_drift_001.py`,
  `evaluate_fp_certfix_001.py --producer network`; commit on `claude/FP-CENSUS-001`.

---

## v2: the same checks under the REPAIRED (lemma A') protocol

After the review's OBJECTION, the multi run was repeated under the first-visit
protocol (`results/FP-CERTFIX-001/claude/multi_fv/`), and all three checks were
repeated on it:

| check | v2 result |
|---|---|
| network vs numpy, 12 steps, both arms | **0 decision divergences, 0 eta divergences** (identical totals: mp 76, split 8) |
| drift probe (`drift_fv`) | worst gap **1.076e-05**, flat across steps, within `1e-4` |
| HP recheck (`hp_recheck_fv`) | **178 decision points, 0 replay mismatches, 0 flips** (104 points also at mpmath-50); smallest emitted margin **5.86e-10** |

The conclusion is unchanged and now attached to a protocol whose bridge lemma is
valid: within the measured 12 steps, no decision is arithmetic-dominated, and the
network reproduces every numpy decision.
