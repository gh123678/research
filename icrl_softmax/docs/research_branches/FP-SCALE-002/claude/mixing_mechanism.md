# Why the mixing setting dominates FP-SCALE-002 emission (read-only diagnostic)

Date: 2026-09-11.
Scope: read-only analysis of the sealed formal bundle. Runs no experiment,
writes no result file, and does not amend the sealed run.

## The observation being explained

The task-level breakdown of the sealed bundle is strongly non-uniform:

| mixing | primary emissions |
|---|---|
| `0.08` | 19 / 24 (0.792) |
| `0.5` | 3 / 24 (0.125) |

`FP-SCALE-002` froze no hypothesis about mixing, so this was recorded as an
exploratory observation. The question is whether it is an unexplained effect
that warrants a new task, or a consequence of the mechanism the project already
established.

## Candidate mechanism

`FP-SCALE-001` established that emission requires the certified error `E_Q` to
fall below the within-state, action-relevant value spread `sigma`, because the
policy-improvement signal is second order in the tilt while the error term is
first order. That predicts the controlling quantity is the ratio

```text
sigma_min / E_Q,   where sigma_min = min_s ptp_a q_pi(s, a)
```

computed from the sealed `oracle_audit_only.within_state_q_spread` and the
sealed per-route `e_q`. Both are already in the bundle, so this is a pure
re-analysis.

## Result

| mixing | mean `E_Q` | mean `sigma_min` | mean `sigma_min / E_Q` | range of the ratio |
|---|---|---|---|---|
| `0.08` | 0.2159 | 0.7402 | 3.687 | 0.572 -- 12.264 |
| `0.5` | 0.2693 | 0.3985 | 1.544 | 0.702 -- 2.562 |

The mixing effect is carried by the **numerator**, not the denominator: `E_Q`
moves only from `0.2159` to `0.2693` (a factor of `1.25`), while the minimum
within-state spread nearly halves from `0.7402` to `0.3985` (a factor of
`1.86`). Mixing controls how much of the action-dependent reward gap survives
into the fixed-policy value differences within a state.

That is mechanistically sensible: low mixing means the sticky self-loop keeps
the process in place, so the action taken matters more for the value
differences inside a state; high mixing washes the action choice out.

## How well the ratio predicts emission, stated precisely

| group | n | mean ratio | min | max |
|---|---|---|---|---|
| emitted | 22 | 4.142 | 2.1721 | 12.2644 |
| not emitted | 26 | 1.324 | 0.5715 | 2.2080 |

**There is no strictly separating threshold.** The two ranges overlap on
`[2.1721, 2.2080]`, and no single cut classifies all `48` records correctly;
a cut at `2.19` misclassifies exactly two. So the ratio is **directionally and
substantially predictive, not a sharp classifier**, and the mechanism is
supported rather than proven.

The residual scatter is not surprising: the emission decision is a strict
all-states condition, so the binding constraint can be a pair other than the
minimum-spread state, and the certificate itself is governed by the worst
certified pair. Those two effects were not separated here.

## Conclusion

The mixing effect is **explained in direction and magnitude by the mechanism
already established**, and it is not an independent phenomenon that needs a
new line of work. It is one more manifestation of the same requirement, that
`E_Q` must be small relative to the within-state value spread.

Accordingly **no new task is opened.** A follow-on task would be justified only
if it tested a genuinely new, falsifiable prediction. The obvious candidate
would be a pre-registered `sigma_min / E_Q` admission rule validated on fresh
records, but the honest assessment is that its value is modest: the controlling
quantity is already identified, the sealed data already supports it, and a
strict classifier would require separating the all-states condition from the
worst-pair certificate, which is a tightening exercise rather than a new
question.

## Limitation

Same actor as the execution, and read-only re-analysis of already-sealed data.
This is not independent verification, is not a pre-registered test, and adds no
acceptance-criterion evidence. It evaluates no out-of-sample claim.
