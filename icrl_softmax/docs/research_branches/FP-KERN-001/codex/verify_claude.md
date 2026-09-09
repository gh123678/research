# FP-KERN-001 Codex verification of the Claude route

Date: 2026-09-09

Verifier branch: `codex/FP-KERN-001` at Codex formal seal
`640a3f8fb2d0f41b96eef8d9bb76fc5ef2e9b93b`.

Verified route: `claude/FP-KERN-001` at Claude formal seal
`1a820467683d137e6527edbd99bb486000b2fc58`, independently implemented from
common execution-start commit
`28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c`.

Both formal seals existed before this verification began. No Claude code or
result was inspected during the blind construction period.

## Provenance and scope

The Claude implementation/smoke seal is
`428af58c9511fd0c4438cbc1c97fceb0fdc632d6`; its formal-result seal is
`1a820467683d137e6527edbd99bb486000b2fc58`. Relative to the common execution
baseline, the implementation seal adds only the five task-authorized Python
entry points and two Claude evidence notes. The formal seal adds only
`docs/research_branches/FP-KERN-001/claude/formal_result.md`. Existing project
code, old results, the task, design, plan, `ACTIVE_WORKSPACE.md`, and `main`
were not changed.

The first-result evidence records a test-first failure before implementation,
all required smoke checks, and one disclosed mechanical repair: the unchanged
current generator's float32 Dirichlet `p0` was normalized in float64 before a
strict 1e-9 simplex validation. The verifier confirms that transition and
reward tensors remain bit-identical to the unchanged `make_mdp` construction.
Formal trajectories start from the exact stationary distribution, not `p0`,
so the normalization cannot affect any route input or reported scientific
metric.

## Executable verification

The following were run from the Claude worktree after the formal seal and all
exited zero:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_kernel_state_generalization.py
C:\Users\Admin\anaconda3\python.exe -B verify_fixed_policy_q_routes.py
C:\Users\Admin\anaconda3\python.exe -B verify_finite_sample_theorems.py
C:\Users\Admin\anaconda3\python.exe -B verify_visit_indexed_martingale_certificate.py
C:\Users\Admin\anaconda3\python.exe -B verify_time_uniform_mixture_certificate.py
C:\Users\Admin\anaconda3\python.exe -m ruff check kernel_state_generalization.py kernel_generalization_mdps.py verify_kernel_state_generalization.py evaluate_kernel_state_generalization.py analyze_kernel_state_generalization.py
C:\Users\Admin\anaconda3\python.exe -B analyze_kernel_state_generalization.py --result-dir results/FP-KERN-001/claude
```

The new verifier passed all 17 labelled contract checks. The strict analyzer
regenerated the MDP, fixed policy, three independent trajectory streams,
V-first estimate, observable aggregates, all route outputs, oracle audit,
metrics, intervals, and decision rule for all 480 records. It reported no
integrity failure and classification `NOT_SUPPORTED`.

## Artifact integrity

All seven hashes were recomputed from disk and match the Claude formal seal:

- `analysis.json`: `b3a7d1834772b3ed5f0bf4ffc45a1bb7eaea379b407e2c93e1f4a85837db4244`;
- `checks.log`: `cfe4ffd0f7c41681bed169ff14a9c5fb9d79c7b30e1a070aafafcdcb68fcd69f`;
- `commands.log`: `eb1a5514b958efdf35d329b0996278ccb2cd1a1b66010d9daf3f1e8403213bf3`;
- `config.json`: `dc6ee372ab7180dee5a6b427d4069d8220bf9593c643253c7d89f7cd62b53d28`;
- `environment.json`: `c98bac6b1a2a59bb9ccc34d5be7aa157574eb1936f0257aa4b4a212273b9a768`;
- `summary.json`: `45314545712f0ddedea99fbf6e064a70c95d1fb279dff3745908f1f6ef254239`;
- `task_results.json`: `a8cbf3ac790255224ce0ff4931af55551c82821be997b8145c9408da8e582921`.

## Contract inspection

The five Claude task files were inspected against task version 1.0 and the
approved design. The implementation matches the four frozen routes, target
action exclusion, two-common-nontarget-action cross-state gate, sorted-float64
median bandwidth, Gaussian weights, count-weighted target pooling, and
observation-weighted ESS. A target state with its own target observations
participates with self weight one; a zero-count target can use only eligible
other states. The same-action anchor never emits at zero target count.

The pure interface rejects oracle keys and derives its masks, distances,
weights, bandwidths, estimates, denominators, ESS, and abstentions entirely
from observable aggregates. True Q/V, transitions, rewards, occupancy, exact
returns, and hidden clusters are attached only in the later oracle-audit
namespace. Three stationary-start streams are deterministically independent
and their `SeedSequence` provenance is serialized and exactly replayed.

The analyzer implements the screen as frozen: zero-count coverage over the
signature-eligible denominator, per-record paired RMSE differences with
Student-t intervals, complete common four-action sparse rows, and the stated
false-improvement event on common finite comparisons. The secondary Spearman
diagnostic is computed within record/action as required and is not substituted
for a screen item.

## Scientific result

The hidden-cluster screen passes coverage and false-improvement control but
fails all three required improvement conditions:

- zero-count RMSE reduction is 7.58%, below 10%, with 95% CI crossing zero;
- `1-4`-count RMSE is 119.78% worse than local, with a strictly negative
  paired interval;
- sparse-state top-action gain is 4.56 percentage points, below the 5-point
  threshold.

The current-unstructured family has the same pass/fail pattern. Its zero-count
RMSE reduction is 4.50%, its `1-4`-count RMSE is 112.51% worse than local, and
its top-action gain is 4.41 points. Both families have positive secondary
signature-distance Spearman diagnostics, but those signals are insufficient
to pass the frozen estimator screen. The correct Claude-route classification
is therefore `NOT_SUPPORTED`, not `INVALID`.

## Cross-route note

Claude's formal numbers differ from Codex because each independent route used
a different deterministic, recorded nonoverlapping substream derivation. The
task fixes seed 20260909 and independence/reproducibility, not a single
bit-level spawn map. Both routes independently reconstruct exactly and agree
on every screen item's pass/fail status and the final classification.

Post-disclosure comparison also shows that the Claude implementation follows
three definitions more literally than the sealed Codex implementation:
self-weight inclusion for sparse positive pairs, the one-sided
false-improvement event, and record/action Spearman granularity. These do not
count against the Claude route; they are recorded for the principal
researcher's synthesis and Codex-route disposition.

The Claude route is independently reproducible, satisfies the frozen
validity-relevant contracts, and supports its sealed scientific conclusion.

PASS
