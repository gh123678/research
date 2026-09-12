# FP-ITER2-001 route journal

Branch: `main`. Single actor: Claude holds both execution and verification under
the standing user instruction of 2026-09-11.

## 1. What this task tests

`FP-SCALE-002` produced `22/48` certified policy improvements and `FP-ATTN-001`
showed the literal attention networks reproduce them. That establishes the
mechanism **exists**. It says nothing about whether it can be applied **more
than once**, which is what repeated policy iteration and any control claim need.

The starting point made this a real test: of the `22` certified first steps,
`18` had `min_s LB_s < 0.05` and `6` were below `0.01`, the smallest at
`0.000001`. Most first steps were certified by a thin margin.

## 2. Frozen construction

Both steps use the **identical** training and certification batches — no
resampling between steps — and the **same frozen** certificate and decision
code, applied to `pi_{k-1}` at step `k`. Nothing is retuned.

The reuse of one certification batch across steps is justified, not assumed: the
held-out residual's conditional expectation
`E[Y_t(Q) | X_t = x] = (T_pi^X Q - Q)(x)` depends only on the transition kernel
and the policy being evaluated, not on the behaviour policy that generated the
data. So the frozen martingale property and the variance-adaptive bound stay
valid for every `pi_{k-1}`.

## 3. Two bugs found and fixed during smoke

Both were in the evaluator, not in the method, and both are recorded.

**Bug 1: the per-route value chain was not reset.** `v_chain` was initialised
once per record and carried from the `expected_exact` route into the
`expected_finite` route, so the finite route compared its new value against the
*exact* route's final value. That produced deltas of `-1.099, -0.483, -0.477,
-0.541` and a spurious "componentwise non-degrading" failure for a step whose
true deltas are `+1.0125, +0.4620, +0.4851, +0.5140`.

This is worth naming precisely: the first smoke run appeared to show the
certified guarantee failing, which would have been a serious scientific result.
It was a bookkeeping error. The debug trace (`prev` equal to the previous
route's terminal value) is what identified it.

**Bug 2: the certificate audit used the wrong reference policy.** The realized
error was compared against `Q^{pi_0}` at every step, but the certificate at
step `k` bounds `||Q_k - Q^{pi_{k-1}}||`. That produced two spurious
"certificate violations" at step 2. The audit now tracks the fixed point of the
policy actually being evaluated.

## 4. Formal result

`24` records, `48` route-records, up to `MAX_STEPS = 2`.

| quantity | result |
|---|---|
| step-1 emissions | **22** (matches the sealed `FP-SCALE-002`) |
| step-1 reproduction failures | **0** (decision, `eta` and `E_Q` all exact) |
| **emitted two steps** | **20** |
| emitted one step | `2` |
| emitted zero steps | `26` |
| **step-2 survival of step-1** | **20 / 22 = 91%** |
| certificate violations | **0** |
| componentwise non-degrading violations | **0** |
| only stopping reason | `improvement_lcb_nonpositive` |
| mean gain, step 1 → step 2 | `2.717707` → `2.277997` |
| **minimum gain, step 1 → step 2** | **`0.104197` → `0.779902`** |

### Hypothesis verdicts

| hypothesis | verdict |
|---|---|
| `H1` step-1 reproduces sealed exactly | **PASS** (48/48) |
| `H2` a second step is certifiable | **PASS** (20 of 22) |
| `H3` second step valid and strictly improving | **PASS** (20/20 positive gains) |
| `H4` value monotone across both steps | **PASS** (0 violations) |
| `H5` no certificate violations | **PASS** |
| `H6` attrition quantified | **PASS** |

## 5. What the result means

**The certificate can be applied more than once.** The mechanism is not a
one-shot phenomenon: starting from the `22` certified steps, `20` were
certifiable a second time under the identical certificate and batches, and
every one of those second steps was componentwise non-degrading and strictly
improving. Value rises monotonically and never falls back.

This is the first evidence of **usability** rather than mere existence in the
project's history, and the gap between the two was the open question the
project's roadmap named.

### An exploratory observation, not a pre-registered hypothesis

The **minimum** gain *rose* from `0.104` at step 1 to `0.780` at step 2, while
the **mean** gain *fell* from `2.718` to `2.278`.

That is consistent with the two records that failed at step 2 being exactly the
ones whose first step was certified by the thinnest margin. Step 2 therefore
looks **selection-filtered**: it keeps the records with real headroom and drops
the marginal ones, so the surviving improvements are more uniform even as the
average shrinks.

FP-ITER2-001 froze no hypothesis about this, and the observation is recorded as
exploratory. It is also the natural next question: whether a third step shows
the same filtering, and whether the mean gain keeps decaying.

## 6. Verdicts on the acceptance criteria

| criterion | status |
|---|---|
| step-1 reproduction on all 48, discrepancies listed | PASS, none |
| identical batches at both steps | PASS (batch digests recorded per record) |
| same frozen certificate and decision code at step 2 | PASS (imported module identity) |
| `H2` evaluated on the frozen records, count reported | PASS (20) |
| `H3`--`H5` per record from the oracle audit, violations listed | PASS, none |
| every non-emitting record carries its frozen reason | PASS |
| exact truth confined to `oracle_audit` | PASS |
| strict-JSON, duplicate-key, finite, shape, seed, location checks | PASS |
| reproducibility evidence | PASS (`config.json`, `environment.json`, `commands.log`) |
| same-actor derived verification recorded | PASS |
| `ACTIVE_WORKSPACE.md` current | pending |

## 7. Limitation

Same-actor derived verification. No second actor reconstructed this route. Both
the literal-numpy agreement in `FP-ATTN-001` and this iteration result rest on
implementations by one author, so a shared conceptual error would not be caught.
Independent verification remains outstanding for the project's headline result.
