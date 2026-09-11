# FP-ESARSA-001 — Claude main route, frozen formal result

Status: sealed formal run, exactly one execution of the frozen 480-record
matrix. At seal time, preliminary pending GPT post-seal acceptance or an
explicit user exemption.

Closure note (2026-09-11): this route was never independently reconstructed.
The task closed `VERIFIED` on 2026-09-11 by an explicit user exemption of the
independent GPT acceptance (see the acceptance-exemption ruling in
`docs/research_tasks/FP-ESARSA-001.md`). The seal-time status above is
preserved as the historical record.

## Command

```
WORKDIR: C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-ESARSA-001\claude_worktree\icrl_softmax
EVALUATION: C:\Users\Admin\anaconda3\python.exe -B evaluate_fixed_policy_expected_sarsa.py \
  --mode formal \
  --output-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-ESARSA-001/claude \
  --label formal-frozen-480
ANALYSIS: C:\Users\Admin\anaconda3\python.exe -B analyze_fixed_policy_expected_sarsa.py \
  --result-dir C:/Users/Admin/Desktop/research/icrl_softmax/results/FP-ESARSA-001/claude --mode formal
```

Preflight `verify_fixed_policy_expected_sarsa.py` passed inside the run
(`checks.log`, 13550 checks). `ruff check` clean on all four route scripts.

## Frozen configuration

30 tasks/cell, 6 states, 4 actions, lengths 256/1024/4096/16384, contiguous
half split, mixing {0.08, 0.5}, gap bonuses {0, 0.5}, pi_min 0.05,
R_star 1.5, gamma 0.70, alpha 0.65, 160 layers, zeta=xi=tau=8, eta grid
(1.0, 0.5, 0.2, 0.1, 0.05), delta 0.05, 15 mixture components, seed 20260829.

## Generator identity

480/480 records reproduced the frozen FP-TU-001 baseline identity fields
(`seed_entropy`, `spawn_key`, `true_pair_occupancy_min`,
`true_action_gap_min`, `true_action_gap_mean`) within `1e-12`
(`regression.json`, `mismatch_count = 0`). Seed schedule: 120 spawned cells.

## Results

| route | certificates | safe updates | max oracle error | min bound slack |
|---|---|---|---|---|
| expected_exact   | 283/480 | 0/480 | 1.667 | 22.30 |
| expected_finite  | 283/480 | 0/480 | 2.187 | 22.32 |
| sampled_exact    | 283/480 | 0/480 | 1.491 | 22.39 |

- Certificate emission is support-limited: `0/30` at length 256, partial at
  1024 (`0.033`–`0.533`), `0.967`–`1.0` at 4096, `1.0` at 16384.
- `E_Q` among emitted certificates ranges `22.47`–`134.01` (mean `57.69`),
  shrinking with length (mean `~27.3` at 16384).
- Ordered reason rates: `heldout_pair_support_missing` 0.41,
  `improvement_lcb_nonpositive` 0.59; no other reason fires.
- Every emitted certificate bounds the oracle error: **0** oracle certificate
  violations, **0** residual-event violations, **0** oracle value decreases,
  reward bound satisfied in all 480 records.
- Contraction premise (empirical writeback diagonal): exact/sampled `480/480`
  satisfied (identity writeback); finite route `270/480` satisfied, failing
  exactly where a training pair is unvisited at lengths 256/1024. Premise
  failures are reported, not oracle-repaired.

## Hypothesis 8 (empirical usefulness)

**Negative.** No primary route emitted a safe relative-softmax update in any
of the 480 frozen records. Per the task sheet this is a valid verified
negative usefulness result: the frozen formula, sharpness, eta grid, split,
and matrix were not retuned, and every mandatory mathematical and
construction claim still holds on the emitted certificates.

## Artifacts and hashes (sha256)

| file | sha256 |
|---|---|
| config.json | `0cc8fe7b571192aae0f5855356c20f0e5b08789197afb45bd2e241bacc10db8d` |
| task_results.json | `3e9b3fa1a1ae528024d36b866f2f85cc57b27de8004b62cbdf51d0718d2a0e28` |
| summary.json | `cac93ef2695065ed4f761043a7ddf2a04331e703b69aa02ab2950239da3fce6a` |
| regression.json | `637c6d47f7b159da9a938bd86f41774f66659eef327be4dee2bf5cdc3e395f7a` |
| environment.json | `824e029131839fdff150fefced28282237eb6fab133a53833587a25b4c39e00f` |
| commands.log | `f58c22ef5c7260a7eb1ced18716107f415e94b401cda6fbad94072b5badcabbf` |
| checks.log | `bbf3091d082ce33ee9e026edd8770e4b02dc352fb2e74534032d1985b9d0e0bd` |

Python 3.13.9, numpy 2.4.6, Windows 11.

## Limitations

- The certificate radius is dominated by the simultaneous `d=24` mixture
  event at moderate held-out counts, so `E_Q` (22–134) dwarfs the realized
  Q error (≤ 2.19): the bound is valid but loose, and the relative-softmax
  lower bound stays nonpositive. No sharpness, radius, split, or eta was
  changed to force an emission.
- The finite route makes no no-bias claim; its final estimate is certified
  only by the exact held-out Bellman residual.
- Closed on 2026-09-11 by an explicit user exemption of the independent GPT
  acceptance; no independent executable reconstruction of this route was
  performed. See the closure note at the top of this file.
