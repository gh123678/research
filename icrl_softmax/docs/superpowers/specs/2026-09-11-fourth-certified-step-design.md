# Design: a fourth certified step, and the shape of the decay

Date: 2026-09-11.
Task: `docs/research_tasks/FP-ITER4-001.md`.
Status: `ACTIVE`.

## 1. Why a fourth step

Three verified steps describe a trend but cannot distinguish candidate laws, and
one earlier prediction about that trend was already falsified
(`FP-ITER3-001`'s `H7`). Extending to four adds one more point to both the
attrition and the gain sequence, at a cost equal to the previous round.

The sealed step 1--3 pattern, from `FP-ITER3-001` (numpy) and identically
`FP-ATTN-ITER-001` (network):

| step | emissions | mean gain | minimum gain |
|---|---|---|---|
| 1 | 22 | 2.717707 | 0.104197 |
| 2 | 20 | 2.277997 | 0.779902 |
| 3 | 15 | 1.286906 | 0.378197 |

Emission ratios `0.9091, 0.7500`; mean-gain ratios `0.8382, 0.5649`; the minimum
gain is non-monotone.

## 2. The pre-registered predictions, and their basis

Three predictions are frozen before the run, with the basis of each stated
explicitly so that a reader can tell extrapolation from a floor:

- `H7`: `n4 < n3` (`15`). **Extrapolation** of the observed monotone attrition.
- `H8`: `mean_gain_4 < mean_gain_3` (`1.286906`). **Extrapolation** of the
  observed monotone mean-gain decay.
- `H9`: `min_gain_4 > 0.05`. **Not** an extrapolation of a monotone law, because
  the observed minimum-gain sequence `0.104197, 0.779902, 0.378197` is
  non-monotone. `H9` is the weakest meaningful continuation claim: it only asks
  whether the fourth step is a real improvement rather than a numerically
  vacuous move.

`H9` matters because a certificate that keeps emitting ever-smaller updates
would technically be "iterating" while being practically meaningless. Setting
the floor at `0.05` makes that failure mode visible.

## 3. The structural change, and why it needs proving

`MAX_STEPS = 4` instead of `3`, in the otherwise unchanged iteration. Raising a
horizon in shared code risks perturbing the earlier steps, so `H1` requires
steps 1--3 to be **bit-identical** to the sealed run. This is a genuine check,
not a formality, and it acquired extra weight in this round because the horizon
knob lives in a file that a previously sealed task recorded by hash.

## 4. A consequence worth designing for explicitly

The horizon knob is a parameter of the **shared** task evaluator, not of the
scientific corpus. Extending it changes that evaluator's hash, which an older
task's sealed-file record legitimately notices.

The design therefore distinguishes two classes:

- the **scientific corpus** — certificate, routes, model, and the certificate
  verifier — which must be byte-identical everywhere and forever;
- the **task evaluators**, which evolve as horizons are extended.

The required response is neither to suppress the change nor to ignore it, but to
pair it with an **inertness proof**: re-run at the frozen horizon and show the
earlier result reproduces exactly. That proof accompanies the change.

## 5. What success and failure mean

- Success: a fourth step exists, its gains are strictly positive and non-vacuous,
  and the attrition and mean-gain decay continue.
- `H3` failing would mean the iteration terminates at three, which is a
  substantive statement about how far the certificate reaches.
- `H7`/`H8` failing would falsify the extrapolated decay reading.
- `H9` failing would mean the fourth step is numerically vacuous.

Each outcome is reported as PASS or FALSIFIED; none is reinterpreted.

## 6. Risks and mitigations

- **Horizon perturbation.** Mitigated by `H1`'s bit-identity requirement.
- **Shared-evaluator hash drift.** Mitigated by separating the scientific corpus
  from the evaluators and by the inertness proof described in section 4.
- **Cost.** `105` step executions of a fast numpy iteration plus sampling;
  measured at a few minutes.
- **Same-actor verification.** Disclosed; not mitigated.

## 7. Out of scope

- Extending past four steps.
- Re-running the network iteration at four steps: the network is verified
  through three, and extending it is a separate task if wanted.
- Repeated control, online control, or conditional-on-emission claims.
- Any claim of independent or reciprocal verification.
