# FP-ATTN-001: Literal softmax-attention execution of the certified update

## Task metadata

- Created: 2026-09-11.
- Author: Claude, under the direct user instruction of 2026-09-11 ("好的先跑第一份").
- Status: `ACTIVE`.
- Task version: `1.0`.
- Scientific baseline: `ebf771c5bf40a48577ce5623840b8a06ef6b054e`
  (`main` after the FP-ESARSA-001 reconstruction).
- Execution branch: `main` (single-actor; see the user ruling below).
- Result directory: `results/FP-ATTN-001/claude/`.
- Design:
  `docs/superpowers/specs/2026-09-11-literal-attention-execution-design.md`.
- Plan:
  `docs/superpowers/plans/2026-09-11-literal-attention-execution-plan.md`.
- Classification: long, conclusion-critical, single-actor under the standing
  user exception.

## Why this task exists

The project's central claim is that a fixed-weight softmax **attention network**
can turn in-context experience into a better policy. Every formal result that
has produced an emission, however, was computed by numpy code. Verified by
inspection of the evaluators on the current baseline:

```text
evaluate_fixed_policy_expected_sarsa.py   imports no torch and no model
evaluate_fp_scale_002.py                  imports no torch and no model
fixed_policy_expected_sarsa_scaled.py     imports no torch and no model
```

The literal network classes `EndToEndMaskedSoftmaxExpectedSARSA` and
`EndToEndFiniteSoftmaxExpectedSARSA` in `model.py` are referenced **only** by
`verify_fixed_policy_expected_sarsa.py`, and only on small hand-built fixtures.
`FP-ESARSA-001`'s acceptance criteria asked the verifier to inspect the literal
network and direct tensor witnesses, and that acceptance was waived.

So the inference chain backing the project's headline result is:

```text
literal network  ==  numpy formula      (shown only on small fixtures)
numpy formula    ->  22/48 certified improvements
therefore literal network -> 22/48      (NEVER EXECUTED)
```

This task executes the missing step. It is the difference between "a numpy
program that reimplements attention formulas produced the improvement" and
"a softmax attention network produced the improvement", which is the question
the project exists to answer.

## Research question

Can the literal softmax attention networks execute the certified
relative-softmax improvement on the frozen `FP-SCALE-002` records, reproducing
the numpy route's `Qhat`, certified error `E_Q`, selected `eta`, and
emission/abstention decision?

## Falsifiable hypotheses

1. `H1 (numeric agreement)`: for every one of the `48` frozen route-records,
   the literal network's final `Qhat` agrees with the numpy route's `Qhat`
   within the frozen tolerance `ATOL = 1e-4` in infinity norm, and is finite
   and inside the divergence guard.
2. `H2 (per-layer execution)`: the layer-0 diagnostics of the literal network
   match the numpy route's layer-0 quantities within `ATOL`, so agreement is
   attributable to executing the same per-layer operator and not to two
   different recursions landing nearby.
3. `H3 (decision agreement)`: for every frozen record the literal network
   yields the same emission/abstention decision as the numpy route, and, where
   both emit, the same selected `eta`.
4. `H4 (no flips)`: zero records change decision. If any record does change
   decision, it must be reported individually with its `E_Q`, its smallest
   per-state lower bound, and its margin to the decision boundary, and it is
   reported as a **discrepancy**, never folded into a general claim of
   agreement.
5. `H5 (the finite route is gate-free)`: the literal finite network uses no
   input-dependent equality mask and no visited-query gate, checked
   executably on the source and on the tensor shapes it produces; the masked
   exact route, by contrast, does carry a visited gate and a null token, and
   this asymmetry is documented rather than blurred.
6. `H6 (nothing sealed is touched)`: every file sealed by a closed task is
   byte-identical to its sealed version after this task runs.

`H1`--`H3` are the mandatory construction claims. `H4` is the honesty claim.
`H5`--`H6` are boundary claims.

## Frozen contract

### Protocol, inherited verbatim

The `FP-SCALE-002` protocol: `4` states, `3` actions, `pi_min = 0.15`, gap
bonus `0.5`, mixing `{0.08, 0.5}`, `12` tasks per mixing, training trajectory
`65536` transitions, certification `16384 x 64 = 1048576` items,
`gamma = 0.70`, `alpha = 0.65`, `160` layers, `R_star = 1.5`, `delta = 0.05`,
seed `20260911`, eta grid `1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01`.

### Training batches, reproduced by seed

The `FP-SCALE-002` bundle does not serialize trajectories, so the training
batches are regenerated deterministically from the frozen schedule
(`fs.build_task` then `fs.training_batch`, seed `20260911`). The regeneration is
itself checked: the numpy route run on the regenerated batch must reproduce the
sealed `Qhat` for that record within `ATOL`. If it does not, the record is
reported as a reproduction failure and excluded from `H1`--`H3` rather than
silently dropped.

### Literal routes

1. `literal_masked_exact`: `EndToEndMaskedSoftmaxExpectedSARSA` with
   `gamma = 0.70`, `alpha = 0.65`, `160` layer applications, `Q_0 = 0`.
2. `literal_finite`: `EndToEndFiniteSoftmaxExpectedSARSA` with the same plus
   `zeta = xi = tau = 8`, `160` layer applications, `Q_0 = 0`.

Both run in `float32` (the networks' native dtype) under `torch.no_grad()`.
The numpy comparators are the inherited `expected_exact` and `expected_finite`
routes, which run in `float64`.

### Frozen tolerance

`ATOL = 1e-4` in infinity norm on `Qhat`, which is roughly three orders of
magnitude above the observed float32-versus-float64 gap (`3.2e-07` and
`2.7e-06` in the interface probe) and roughly two orders below the smallest
per-state lower-bound margin seen in the sealed matrix. It is frozen here,
before the full matrix is run.

A decision flip is **not** a tolerance failure: it is a legitimate outcome of
the frozen rule near its boundary and must be reported as such under `H4`.

### Certificate and decision

Both literal and numpy routes are scored with the **same frozen** certificate
and decision code (`fixed_policy_variance_certificate.variance_adaptive_certificate`
and `fixed_policy_expected_sarsa_scaled.improvement_for`), applied to the
certification batch of the corresponding record. The comparison is therefore
about the `Qhat` the network produces, not about two different decision rules.

## Prohibited work

- No modification of any sealed program, sealed result bundle, or task record
  from a closed task.
- No change to any frozen protocol parameter, tolerance, hypothesis, or metric
  after the full matrix is run.
- No re-running or amending the sealed `FP-SCALE-002` formal matrix; the
  literal run is a new, separate evaluation.
- No claim that a decision flip is "numerically equivalent"; flips are reported
  individually.
- No claim of independent or reciprocal verification: single actor by ruling.
- No merge to `main` beyond this task's own commits without separate approval
  for any *result* merge the user may request.

## Acceptance criteria

1. The literal networks execute `160` layers on every frozen record without
   non-finite values and without tripping the divergence guard.
2. `H1` is evaluated per record with the frozen tolerance, and the worst
   observed deviation is reported exactly.
3. `H2` compares layer-0 diagnostics against the numpy layer-0 quantities and
   reports per-field deviations.
4. `H3` reports, per record, the numpy decision, the literal decision, the
   numpy `E_Q`, the literal `E_Q`, and the selected `eta` for both.
5. `H4` lists every decision flip individually with the boundary margin, or
   states that there are none.
6. `H5` is checked executably: source inspection plus a tensor-level check that
   the finite route's attention is full-support and gate-free on this data.
7. `H6` verifies byte-identity of every sealed file against its recorded hash.
8. The regenerated numpy `Qhat` reproduces the sealed `Qhat` for every record
   within `ATOL`; any failure is listed.
9. All strict-JSON, finiteness, shape, seed, and result-location checks pass.
10. Complete reproducibility evidence is recorded: commands, environment,
    commits, raw hashes, anomalies, and limitations.
11. Same-actor derived verification is recorded, stating the limitation
    explicitly.
12. `ACTIVE_WORKSPACE.md` is current.

## Failure criteria

The construction fails if the literal networks cannot complete `160` layers on
the frozen records, if they produce non-finite values, if the numpy
regeneration does not reproduce the sealed `Qhat`, if the comparison uses
different certificate or decision code for the two sides, or if any decision
flip is suppressed rather than reported.

`H3`/`H4` failing is **not** a construction failure. A small number of flips
would be a legitimate, informative finding about how close the frozen matrix
sits to the decision boundary, and would be reported as the task's headline
result rather than treated as an error.

## Stopping conditions

Stop affected work and notify the user if:

- the literal networks cannot run at the frozen dimensions within budget;
- a sealed file is found to have been modified;
- the numpy regeneration cannot reproduce the sealed `Qhat` for most records,
  which would indicate the frozen schedule is not reproducible as documented;
- the comparison would require changing a frozen tolerance or protocol
  parameter;
- execution would expand cost, publication, external communication, or
  permissions beyond authorization.

## Route assignment and verification

Single actor: Claude executes and verifies, under the standing user instruction
of 2026-09-11 ("不管 codex 了，验证也交给你"). The verification is therefore
not independent and must be labelled as such everywhere. It must recompute the
comparison through a separate code path, replay the sealed programs, and state
the limitation.

## Pre-review

- Status: `APPROVED` (2026-09-11), same-actor.
- Evidence: `docs/research_branches/FP-ATTN-001/claude/pre_review.md`.

## Objections and user rulings

### Objection

- Status: `NONE`.

### User ruling (scope of this task)

- Date: 2026-09-11.
- Decision: "好的先跑第一份" — run the literal-attention execution first,
  ahead of the Codex independent-verification option.
- Scope: this task. The independent-verification option remains open and
  unstarted.

## Definition of done

- [ ] Pre-review recorded with no unresolved objection.
- [ ] Literal networks run on all `48` frozen route-records.
- [ ] `H1`--`H6` each evaluated with evidence.
- [ ] Decision flips listed individually, or none stated.
- [ ] Same-actor derived verification recorded, with the limitation stated.
- [ ] `ACTIVE_WORKSPACE.md` is current.
