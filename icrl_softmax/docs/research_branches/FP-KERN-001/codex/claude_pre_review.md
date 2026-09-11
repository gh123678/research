# Claude read-only pre-review of FP-KERN-001

Date: 2026-09-09

Reviewed commit: `c17fa4f28f0ecfb9a08c3c1c8bb11634a7d73801`

Tool boundary: local Claude Code 2.1.138, `plan` permission mode, and only
`Read`, `Glob`, and `Grep`. No Bash, Python, file write/edit, Git mutation, or
experiment was available.

## Decision

`APPROVED`

Claude found no blocking circularity, target-data leakage, unfair baseline,
invalid inference, unfrozen parameter, governance violation, or contradiction
between the task, design, plan, and workspace index.

## Itemized result

1. `DRAFT -> REVIEW`, branch isolation, and the verified baseline are valid;
   no implementation file existed before activation.
2. The no-feature premise matches the independently sampled tabular MDP
   generator; state labels have no metric meaning.
3. The primary distance excludes target-action signature coordinates and
   requires two common observed non-target actions.
4. Count unambiguously means target-trajectory pair count, while the three RNG
   streams are independently and reproducibly derived.
5. The hidden-cluster `0.90/0.10` convex construction preserves transition
   simplex and reward bounds; clusters and prototypes remain oracle-only.
6. The current MDP family is an unchanged applicability control.
7. Median bandwidth is deterministic, with no grid, learned selector, or
   oracle substitute.
8. True Q/V/kernel/reward/occupancy/cluster/return values are structurally
   isolated from estimator inputs and attached only after outputs freeze.
9. The formal arithmetic is 480 records and the smoke arithmetic is 16.
10. Comparable finite-pair restrictions do not conceal coverage failures, and
    all routes share the same target observations and nuisance value estimate.
11. Per-record paired Student-t summaries match project convention.
12. The five thresholds and four-way classification are pre-registered.
13. Ordered abstentions and no-fallback behavior are executable.
14. Allowed and prohibited paths preserve all existing evaluators, results,
    `FP-ADV-001`, and the manuscript.
15. Blind independent construction and reciprocal executable verification
    satisfy `AGENTS.md`.
16. All 18 acceptance items are executable after the clarifications below.

## Nonblocking cautions and closure

Claude requested exact rules for empty count buckets, the
`signature_eligible` denominator, even-sized medians, effective sample size,
the hypothesis-1 statistic, shared `V_hat` nuisance dependence, inherited
verifier names, and incomplete rows in top-action accuracy.

GPT closed all eight in `FP-KERN-001` task version 0.3. The closures freeze:

- common-nonempty per-record paired metrics and nonpassing intervals with fewer
  than two contributing records;
- a zero-count eligibility denominator that excludes only insufficient
  signature support and unavailable bandwidth while counting later failures;
- sorted-float64 `numpy.median` semantics;
- observation-weighted ESS `(sum K*N)^2 / sum K^2*N`;
- per-record Spearman correlation as a secondary, non-screen diagnostic;
- an explicit statement that shared `V_hat` is a nuisance coupling, not a
  target signature coordinate or target-trajectory leak;
- four named inherited verifiers; and
- complete common four-action rows for top-action comparisons.

These are reconstruction and reporting clarifications. They do not change the
kernel, environment families, matrix, hypotheses, thresholds, routes, or final
decision rule.

