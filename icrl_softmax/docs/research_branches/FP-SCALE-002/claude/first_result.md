# FP-SCALE-002 route journal

Branch: `claude/FP-SCALE-001` (the execution branch named in the task sheet as
`claude/FP-SCALE-002` was not created; the work is recorded on the branch that
carries it, and this deviation is disclosed). Single actor: Claude holds both
execution and verification under the user instruction of 2026-09-11.

## 1. Gate A — pre-review

Same-actor pre-review recorded at
`docs/research_branches/FP-SCALE-002/claude/pre_review.md`: `APPROVED`, with two
mandatory conditions — no free scaling constant anywhere in the certificate, and
`H3`'s threshold taken from the rigorous prototype rather than the optimistic
one. Both are enforced executably.

## 2. Tests-first

`verify_variance_adaptive_certificate.py` was written before the implementation,
in eleven sections. Red-first evidence: the first runs failed with
`ModuleNotFoundError` for `fixed_policy_variance_certificate`, then with genuine
contract failures that the implementation had to satisfy (see section 4).

## 3. Implementation

- `fixed_policy_variance_certificate.py` — the two-half certificate with
  explicit Hoeffding constants, plus `envelope_control_certificate` for `H5`.
- `evaluate_fp_scale_002.py` — the frozen evaluator, three routes, strict-JSON
  bundle, oracle audit separated by construction.
- `fixed_policy_expected_sarsa_scaled.py` — one fix: `route_failure_reasons`
  now reads `failure_reasons` as well as `reasons`, so a certificate refusal
  can never be reported with an empty reason list.

The sealed `fixed_policy_expected_sarsa.py` is reused by import and is
unchanged; its SHA-256 is recorded in `environment.json`.

## 4. Defects found and repaired during the tests-first loop

Four checks failed on first contact with the implementation and each identified
a real problem, not a cosmetic one:

1. **Empty reason list.** The evaluator reported
   `certificate_status=not_certified` with `ordered_reasons=[]`, because the new
   certificate returns `failure_reasons` while the shared helper read `reasons`.
   A refusal with no recorded reason would have violated acceptance criterion 7.
   Repaired in `route_failure_reasons`.
2. **Brittle constant check.** The first version of the constant check matched
   source text (`"ENVELOPE**2 / 2.0"`), which failed for a cosmetic reason.
   Replaced by a semantic check that reconstructs both Hoeffding expressions by
   hand and requires exact agreement, which is strictly stronger.
3. **Fixture without full support.** `tiny_fixture` leaves two of four pairs
   empty, so the count floor correctly refused to certify and the formula checks
   were vacuous. Added `populated_fixture`, which populates every pair.
4. **Test vector without information.** The emission check used a Q vector whose
   second state had zero spread, so `min_s LB_s > 0` correctly failed. The rule
   requires strict positivity at **every** state; the fixture now carries a
   spread in both states.

Verifier status: **71 checks pass**, including a validity check that `E_Q`
bounds a realized error computed from a `3x2` model truth obtained by solving
the pair-MRP linear system — i.e. independently of the estimator.

## 5. Gate B — smoke

Two tasks, both mixing settings, real certification batches
(`65536 chains x 64` = `4194304` items per record).

| quantity | value |
|---|---|
| wall time | 216.5 s |
| records | 4 |
| primary route-records | 8 |
| **primary emissions** | **6 / 8 (75%)** |
| componentwise non-degrading | 6 / 6 |
| strict improvements | 6 / 6 |
| certificate violations | 0 |
| minimum per-pair count observed | 78607 (floor is 10000) |

Attribution control on the identical records: the `2B` envelope certificate
gives `E_Q` between `3.35x` and `6.92x` larger than the variance-adaptive
certificate and **never emits**, so the difference is attributable to the
concentration argument alone.

Projected formal cost: the smoke's per-task cost is dominated by sampling, so
the `24`-record matrix projects to roughly `22` minutes, within budget.

## 6. Gate C — formal run

Implementation and smoke were sealed at commit `c701af3` before the formal run
began. The formal matrix (2 mixing settings x 12 tasks, three routes each) was
then executed exactly once into `results/FP-SCALE-002/claude/formal/`.

### Formal result

| quantity | value |
|---|---|
| records | 24 |
| primary route-records | 48 |
| **primary emissions** | **22 / 48 (45.8%)** |
| componentwise non-degrading | 22 / 22 |
| strict improvements | 22 / 22 |
| **certificate violations** | **0** |
| control (envelope) emissions | 1 / 24 |
| count-rule failures | 0 |
| abstention reasons | `improvement_lcb_nonpositive` only, 26 occurrences |
| selected eta | `1.0` x 21, `0.1` x 1 |
| `E_Q` adaptive, mean | 0.2426 |
| `E_Q` envelope control, mean | 1.0744 |
| control/adaptive `E_Q` ratio | 3.448 -- 5.457 in all 24 records |
| mean value gain among emitted | 2.7177 |

### Hypothesis verdicts

| hypothesis | verdict |
|---|---|
| `H2` certificate validity (0 violations, control included) | **PASS** |
| `H3` emission >= 12 of 48 | **PASS** (22) |
| `H4` every emission non-degrading, at least one strict | **PASS** (22 / 22 strict) |
| `H5` envelope control worse in every record | **PASS** (min ratio 3.448 > 1) |
| `H6` no free constant | **PASS** by construction and by verifier sections S2 and S9 |

The `H5` result is the attribution: on identical tasks, batches, split sizes and
risk allocation, the only difference is the concentration argument, and the
variance-adaptive radius is smaller in **every one of the 24 records**, by
3.4x to 5.5x.

### A prediction of this journal that the formal run falsified

This journal's earlier prototype note expected the selected `eta` to be small.
The formal run selected `eta = 1.0` in 21 of 22 emissions and `eta = 0.1` once.
Both lie inside the **inherited** grid `{1.0, 0.5, 0.2, 0.1, 0.05}`, so the
FP-SCALE-001 v1.1 downward extension was **not** the enabler in this run. That
extension is therefore reported as unnecessary here rather than as a
contributor. It is recorded as a falsified expectation, not silently dropped.

## 7. Gate D — same-actor derived verification

`verify_fp_scale_002_same_actor.py` recomputes every reported quantity from the
sealed raw records through a separate code path, replays the sealed programs,
and checks the sealed-module hashes. Result: **PASS**, all derived checks.

It also **caught a defect in its own first version**: a check asserted that
every emission selected `eta = 1.0`. The recomputation showed `{1.0, 0.1}`, so
the assertion was wrong and was corrected. This is recorded because it is
evidence that the derived path is doing independent work rather than restating
the author's summary.

Full report:
`docs/research_branches/FP-SCALE-002/claude/verification_same_actor.md`.

**Limitation, stated plainly:** this is same-actor derived verification. No
second actor reconstructed this route, and nothing here is reciprocal
verification.

## 8. Status

Task complete at Gate D. `main` is unchanged and no merge has been requested.

