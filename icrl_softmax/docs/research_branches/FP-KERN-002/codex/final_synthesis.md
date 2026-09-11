# FP-KERN-002 final synthesis

Date: 2026-09-09

Final status: `VERIFIED`

Final classification: `NO_BORROWING_EVIDENCE`

## Independent route seals

- GPT implementation/smoke:
  `f37b9730aea4ba692b854dcfb89f8f3d17faa34e`;
- GPT formal result:
  `0815d0dbef3a8f7784438ac89e2df195d3cab00b`;
- GPT verification of Claude:
  `076939a70f5333f5a64a7a4360ea26e02a040b1d`;
- Claude implementation/smoke:
  `a39323c8011acebfb651c8431d8598e1d1aee244`;
- Claude formal result:
  `718d77053801c6f9e3dd958513b7a06918a5e274`;
- Claude verification of GPT:
  `1567603d5d8f8fd99c266c17a3741ae8fa30c7d3`.

Both routes started from
`ffdf26b029efde08ea794454a7b5da890108c355`, consumed the same frozen input,
sealed independently before disclosure, and performed reciprocal executable
verification afterward. Both reports end `PASS`.

## Reconciled evidence

The two independently implemented routes agree on every frozen screen,
aggregate metric, confidence interval, secondary diagnostic, abstention count,
gate, and the ordered classification. GPT's verification of Claude passed 14
fixtures, 506 analyzer-independent reconstruction groups, hash checks, Ruff,
and all inherited verifiers. Claude's verification of GPT passed 248,566
cross-route checks with zero failures.

The only route differences are harmless serialization choices: source-list
ordering, whether zero-contribution members are explicitly listed, floating
summation order below `2.98e-14` relative difference, field names/wrappers, and
Stage 0 identity notation. They change no estimate, downstream metric, screen,
or classification.

## Scientific conclusion

All routes achieved high eligible zero-count coverage, so simple action
inaccessibility was not the operative failure. Even the true-Q nearest-two
upper bound improved zero-count RMSE and top-action identification while
significantly worsening 1--4-count RMSE. True generator membership did not
make unconditional target-action pooling useful, and the observable balanced
partition recovered little hidden structure (mean ARI `0.1516203703703704`,
peer precision `0.4909722222222222`).

Therefore the tested practical claim is not supported: unconditional
count-weighted same-action borrowing across states, under the frozen peer and
partition rules, does not pass the five-item policy-improvement diagnostic on
this corpus.

This does not reject every form of cross-state generalization. Zero-only
borrowing, count-aware gating or shrinkage, learned representations, and
estimators that retain local evidence are outside this task and require a new
frozen hypothesis.

## Governance closure

There is no unresolved objection. Source identity, Stage 0, isolation,
single-run constraints, result reconstruction, hashes, reciprocal validation,
and all 19 acceptance criteria pass. The documented CRLF identity distinction
and Claude Git/session permission interruptions did not change any scientific
input or output. No merge to `main`, push, publication, or quota reset was
performed.
