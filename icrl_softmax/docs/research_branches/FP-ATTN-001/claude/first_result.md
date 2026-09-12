# FP-ATTN-001 route journal

Branch: `main`. Single actor: Claude holds both execution and verification under
the standing user instruction of 2026-09-11.

## 1. The gap this task closes

The project's claim is about a fixed-weight softmax **attention network**. Every
formal emission, however, was computed by numpy code. Verified by inspection:

```text
evaluate_fixed_policy_expected_sarsa.py   imports no torch and no model
evaluate_fp_scale_002.py                  imports no torch and no model
fixed_policy_expected_sarsa_scaled.py     imports no torch and no model
```

The literal networks in `model.py` were referenced only by
`verify_fixed_policy_expected_sarsa.py`, and only on small fixtures. So the
chain backing the headline result was:

```text
literal network == numpy formula   (fixtures only)
numpy formula   -> 22/48 certified improvements
therefore literal network -> 22/48 (never executed)
```

This task executes the missing step.

## 2. Interface and timing probe

`EndToEndMaskedSoftmaxExpectedSARSA.forward` and
`EndToEndFiniteSoftmaxExpectedSARSA.forward` each apply **one** layer and return
`(q_new, diagnostics)`, so `160` layers means `160` calls. Both run
`160` layers on a real `65536`-transition record in about **1 second** on CPU
(`torch 2.11.0+cpu`), which made a full-matrix literal run cheap enough to be
the default rather than a subset.

## 3. A defect found and fixed during smoke

The evaluator initially read `fs.CERT_CHAINS` and `fs.CERT_CHAIN_LENGTH`, which
carry the **FP-SCALE-001** protocol (`262144 x 16`). `FP-SCALE-002` froze
`16384 x 64` in its own evaluator, and the sealed corpus was produced with
those. The regenerated certification batches therefore had `4x` more items than
the sealed run used, and the sealed-regeneration check failed with an `E_Q` gap
of about `0.13`.

Fixed by freezing `CERT_CHAINS = 16384` and `CERT_CHAIN_LENGTH = 64` locally in
this evaluator, with a comment recording why `fs` does not supply them.

This is exactly the class of error the task exists to catch, and it is recorded
rather than quietly corrected.

## 4. The sealed FP-SCALE-002 result is bit-exactly reproducible

While diagnosing section 3, the sealed code path was replayed verbatim on three
records. Result: `|dE_Q| = 0.0`, `max|dLB| = 0.0`, identical selected `eta` and
identical emission decisions. The sealed `FP-SCALE-002` formal result is
therefore **bit-exactly reproducible**, which had not been demonstrated before
for that task.

## 5. Full-matrix result

`24` records, `48` route-records, the frozen `FP-SCALE-002` protocol, the
literal networks in `float32` versus the numpy routes in `float64`.

| quantity | result |
|---|---|
| route-records compared | `48` |
| max `\|literal Q - numpy Q\|_inf` | **1.076e-05** (frozen `ATOL` 1e-04) |
| numpy emissions | **22 / 48** |
| **literal emissions** | **22 / 48** |
| decision flips | **0** |
| selected-eta flips | **0** |
| sealed regeneration failures | `0` |
| max sealed `E_Q` gap | `0.0` |
| max sealed lower-bound gap | `0.0` |
| layer-0 diagnostic gap | `0.0` on every reported field |
| non-finite literal `Qhat` | `0` |
| divergence-guard trips | `0` |

**The literal attention networks reproduce the certified improvement exactly.**
The inference step that had never been executed is now executed, and it agrees.

### Layer-0 diagnostics

The `0.0` gaps matter. They are not a loose agreement between two recursions
that happen to land nearby; the literal networks compute the same successor
expectation, the same retrieved current value, and the same signed residuals at
layer 0 as the numpy route, bit for bit in `float32`-representable terms. The
`1.076e-05` terminal gap is the accumulated `160`-layer rounding difference, and
it never reaches the decision boundary.

## 6. H5: the finite route is gate-free, checked executably

| check | result |
|---|---|
| finite route exposes no `visited` key | PASS |
| masked exact route does carry `visited` + null token | PASS |
| every finite-route write weight strictly positive (full support) | PASS |
| finite-route write attention row-normalised | PASS |
| masked route's write attention contains exact zeros (its mask) | PASS |
| this evaluator introduces no `-inf` mask of its own | PASS |

The asymmetry is real and documented rather than blurred: the **exact** route is
allowed an equality mask and an unvisited-query gate, and uses them; the
**finite** route uses neither and still produces the same decisions on this
matrix.

## 7. H6: nothing sealed was touched

All six files tracked in `environment.json` are byte-identical to their hashes
at this run, and both modules also still match their `FP-SCALE-002` sealed
hashes (`fixed_policy_expected_sarsa.py` in raw form, the certificate module in
LF-normalised form; the line-ending situation is recorded explicitly rather
than hidden behind a normalisation).

## 8. Hypothesis verdicts

| hypothesis | verdict |
|---|---|
| `H1` numeric agreement within `ATOL` | **PASS** (max `1.076e-05`) |
| `H2` per-layer execution | **PASS** (layer-0 gap `0.0` on every field) |
| `H3` decision agreement | **PASS** (22 vs 22, no flips) |
| `H4` no flips | **PASS** (0 flips, 0 eta flips) |
| `H5` finite route gate-free | **PASS** |
| `H6` sealed files untouched | **PASS** |

## 9. What this does and does not establish

**Does:** the certified relative-softmax improvement on the frozen
`FP-SCALE-002` matrix is produced by the literal softmax attention networks,
not merely by numpy code reimplementing their formulas. Zero records change
decision between the two.

**Does not:** it does not make `FP-SCALE-002` reciprocally verified. This is
same-actor derived verification; no second actor reconstructed either route.
The `float64`-numpy and `float32`-literal paths are two implementations by the
same author, so a shared conceptual error would not be caught by their
agreement. Independent verification by a second actor remains the outstanding
step for the project's headline result.

## 10. Verification

`verify_fp_attn_001_same_actor.py`: **PASS**, all checks, including three
replayed sealed programs. Report:
`docs/research_branches/FP-ATTN-001/claude/verification_same_actor.md`.

Ruff clean on all new modules.
