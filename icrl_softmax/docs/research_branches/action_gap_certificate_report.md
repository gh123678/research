# FP-ADV-001 verified report

Date completed: 2026-09-10

Final status: `VERIFIED`

Outcome: mathematically valid, empirically zero-usefulness on the frozen
matrix

## Bottom line

The project can implement a safe one-step policy improvement from a fixed
policy without explicitly estimating a complete Q table: a strictly positive
local action-gap lower bound is enough to transfer probability mass from a
donor action to a selected receiver.  The proof is valid for both the exact
and finite-softmax V-first constructions and preserves the exploration floor.

However, the frozen 480-record experiment emitted no update on any route.
Thus hypothesis 8, which required at least one useful primary local update, is
falsified.  Hypotheses 1--7 and the associated safety, preservation, and
software claims passed.  No formula, threshold, transfer fraction, route, or
matrix was retuned after seeing the result.

## Frozen formal result

Both independent routes used the same 30-by-4-by-2-by-2 matrix, 480 matched
records, seed `20260829`, transfer fraction `0.5`, and the verified
`FP-TU-001` baseline.

| Route | Codex updates | Claude updates | Eligible donors | Updated states |
|---|---:|---:|---:|---:|
| V-first local exact | 0/480 | 0/480 | 0 | 0 |
| V-first local softmax | 0/480 | 0/480 | 0 | 0 |
| V-first global exact | 0/480 | 0/480 | 0 | 0 |
| V-first global softmax | 0/480 | 0/480 | 0 | 0 |
| Direct-Q global exact | 0/480 | 0/480 | 0 | 0 |
| Direct-Q global softmax | 0/480 | 0/480 | 0 | 0 |

In the Codex primary local routes, 135 records contained a selected unvisited
pair, 13 contained a non-emitted state certificate, and 467 contained a
nonpositive gap LCB.  These reason counts overlap.  Since every returned
policy was exactly the fixed input policy, false-ordering, Bellman-bound,
componentwise-value-decrease, and exact-return-decrease counts were all zero.

Local penalties were never wider than their matching complete-Q V-first
controls.  Decision dominance was vacuous because the controls also emitted
no update.

## Independent construction and reciprocal verification

- Codex blind first-result seal:
  `996641d9ce2088e95c0bf2a0661e3b24b6e0d6fe`.
- Codex formal-evidence seal:
  `5f190ebb78697acd3cd877c1d64898929bb88024`.
- Claude blind first-result seal:
  `191e13ba5472ce4c183643008169236979c000ff`.
- Claude formal-evidence seal:
  `191821b26b16e13de323fb31651343ffe1eb9656`.
- Claude reciprocal-report commit: `42fb0cc`.

GPT verified Claude with `PASS`: Claude's verifier and Ruff passed, and GPT
independently replayed all 480 records and 17,280 route-state entries with zero
failures and maximum q-row difference zero.

After the user's explicit authorization on 2026-09-10, Claude independently
verified the Codex route with `PASS` using only these three read-only commands:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_action_gap_certificate.py
C:\Users\Admin\anaconda3\python.exe -B analyze_action_gap_certificates.py --result-dir results/FP-ADV-001/codex
C:\Users\Admin\anaconda3\python.exe -m ruff check action_gap_certificate.py verify_action_gap_certificate.py evaluate_action_gap_certificates.py analyze_action_gap_certificates.py
```

The outputs were respectively `action-gap certificate checks passed`,
`PASS FP-ADV-001 strict analyzer with 480 records (formal_480)`, and
`All checks passed!`.  The formal evaluator was not rerun, and the reciprocal
session performed no file write, Git mutation, network retrieval, or access to
the successor task.

The execution entry point was Claude Code CLI 2.1.138, session
`7a2c8fbc-61f5-4611-9214-0207aa5cb600`.  Its machine-readable usage metadata
reported backend labels `kimi-k2.6` and `k3`.  The frozen task did not prescribe
a model identifier; this environment fact is retained for provenance rather
than silently normalized.

## Acceptance assessment

- Theory and probability contract (criteria 1--5): `PASS`.  Both exact and
  softmax decompositions follow from the inherited simultaneous event; there
  is no extra risk split or conditional-on-emission claim.
- Update validity and support (criteria 6--11): `PASS`.  The implementation
  enforces positive counts, positive LCBs, transferable mass, the simplex and
  exploration floor; localized abstention and unchanged Direct-Q controls are
  verified.
- Schema and preservation (criteria 12--17): `PASS`.  New fields are isolated,
  strict JSON and oracle separation pass, both formal routes contain exactly
  480 records, and the inherited baseline has zero numeric and nonnumeric
  mismatch.
- Empirical audit and evidence (criteria 18--19): `PASS`.  All oracle audit
  counts, commands, hashes, environments, anomalies, repairs, and limitations
  are recorded.  The one Codex no-donor serialization defect changed values by
  at most `6.94e-17`; it was repaired from saved observable inputs without a
  formal rerun and now returns the original policy exactly.
- Reciprocity and governance (criterion 20): `PASS`.  Both directions end in
  `PASS`, the zero-update conclusion agrees, and no unresolved objection or
  scientific discrepancy remains.  `main` has not been changed.

## Interpretation and next step

This task establishes the first part of the research program: the proposed
local softmax-style construction really can define a policy-improvement
operator, and an emitted update has a finite-sample non-degradation guarantee.
The negative experiment shows that this particular V-first certificate is too
conservative to improve the frozen cases, not that policy improvement is
impossible.

The successor `FP-ESARSA-001` task remains `DRAFT`.  Its proposed direction is
fixed-policy Expected SARSA: aggregate every repeated occurrence of the same
`(s,a)` group, evaluate the fixed behavior policy, and keep policy improvement
as a separate certified step.  It must pass task review before implementation
or experiments begin.

No merge to `main` was performed; any future merge still requires the user's
explicit approval.
