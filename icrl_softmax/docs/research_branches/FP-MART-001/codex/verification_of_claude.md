# FP-MART-001: GPT verification of the sealed Claude route

## Verification identity

- Verifier: GPT principal-researcher route.
- Date: 2026-09-03.
- Claude branch: `claude/FP-MART-001`.
- Sealed Claude commit verified:
  `c5da2430d00e11414723d85cf42293dddae07183`.
- Common parent:
  `c8ec7e5c3165930663e26a07f98b38cc9ec186ad`.
- Claude worktree state before and after reproduction: clean.
- Scope check: the sealed commit added only the four allowed Python entry
  points and two files below the Claude evidence directory. It did not modify
  the frozen task, design, plan, workspace index, GPT evidence, legacy code, or
  old result directories.
- Verification status: `FAIL`, due to repairable proof, implementation, smoke,
  and evidence defects. This is not an `OBJECTION`; the frozen task remains
  valid.

## Reproduction performed

From
`C:\tmp\research-FP-MART-001-claude\icrl_softmax`, GPT reran all four legacy
verifiers and Claude's 20-test verifier. Every process exited zero. The four
Claude Python files also passed Ruff with `--no-cache`.

The exact formal reproduction commands were:

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_visit_indexed_certificates.py --tasks 30 --trajectory-lengths 256 1024 4096 16384 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex\verification_claude

C:\Users\Admin\anaconda3\python.exe -B analyze_visit_indexed_certificates.py --baseline-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\fixed_policy_finite_sample_certificates --new-dir C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex\verification_claude
```

The reproduction preserved the three frozen baseline hashes and produced 480
records. The deterministic core outputs were byte-identical to the sealed
Claude result directory:

| Reproduced file | SHA-256 |
| --- | --- |
| `config.json` | `a2276eae06ba8689864cac2ad3d3d0e92b069014b2046d0180ee172bb8296ea0` |
| `task_results.json` | `d5d5e9174fe5e4571397ad4c34319a667641ede372fa2cc8066b4063204f0ab3` |
| `summary.json` | `4f76af8ca9777d3c9b52a1de998671847f59cd893821a9505c9a3a91a42ffe03` |
| `regression.json` | `7aeab043586009c77488ed774e449b401a2d379a56572201e8f94974496633fc` |
| `environment.json` | `4dd2de670d94fb5bead17896047674d0dcdc3e3df6ce72fda4b151dd15f234e0` |
| `commands.log` | `68119111474de1d365886e36a59bb20eb6216022f851bcd345e8363291aff922` |

The environment and command hashes appropriately differ from the original
because the reproduction timestamp, recorded HEAD, and output path differ.

Positive reproduced facts are substantial:

- all 480 formal task keys align with the baseline;
- 506,715 legacy task-record leaves match, with zero mismatch and maximum
  numeric difference zero;
- exact-route emissions are `3, 102, 120, 120` of 120 by length;
- Direct-softmax emissions are `3, 74, 115, 120`;
- both V-first routes have the exact-route emission counts;
- all recorded optional variance objects are unavailable;
- independent normalized comparison against GPT gives maximum emitted-bound
  differences of `7.11e-15` for Direct exact, `5.33e-13` for Direct softmax,
  `1.78e-15` for V-first exact, and `5.33e-15` for V-first softmax.

These facts independently support the deterministic recurrence structure and
the predicted support/margin emission pattern. They do not cure the failures
below.

## Failure 1: the written MGF iteration is invalid

Claude's `theory.md`, lines 139--145, directly states

```text
E exp(lambda sum D_t)
  <= E exp(lambda^2 sum (2 B J_t)^2 / 8).
```

That intermediate inequality does not follow when later predictable selectors
`J_t` depend on earlier increments: the random compensator and the preceding
exponential can be correlated. A two-step predictable Rademacher construction
at `lambda=B=1` gives approximately `2.281` on the claimed left side and
`2.184` on the claimed right side.

The final radius is recoverable without changing its constant. The author must
instead define

```text
M_t = exp(lambda sum_{u<t} D_u
          - lambda^2 B^2 sum_{u<t} J_u / 2),
```

prove `E[M_n] <= 1`, and then use the pathwise bound `sum J_u <= k` to obtain
`E exp(lambda sum D_u) <= exp(lambda^2 B^2 k/2)`. Until this compensation step
is written correctly, acceptance criteria 1 and 2 are not proved by the Claude
route.

## Failure 2: a true reward table enters the certificate

Claude's evaluator, lines 424--441, computes

```text
reward_limit = max(abs(mdp["R"]))
value_limit = reward_limit / (1-gamma)
```

from the complete true MDP reward tensor and passes that value into the primary
certificate. The frozen protocol permits only the predeclared public rule
`R_star = 1 + gap_bonus`, with `B = R_star/(1-gamma)`.

The leakage is not merely nominal. Across Claude's formal outputs:

| Gap bonus | Required public `B` | Claude certificate `B` range |
| ---: | ---: | ---: |
| 0.0 | 3.3333333333 | 3.2427398364--3.3330865701 |
| 0.5 | 5.0000000000 | 4.0959713856--4.9974721670 |

All 480 records use a task-dependent true-model value rather than the public
bound. The numeric bounds are therefore oracle-assisted and do not satisfy
acceptance criteria 5 and 9, even though the true full-table maximum is a valid
mathematical bound for this synthetic MDP. The source must be removed, the
public rule must be serialized, and the formal matrix must be rerun.

## Failure 3: impossible visit counts can emit a certificate

The pure module converts counts using `int()` and checks only nonnegativity and
vector length. It does not require original integer inputs, either count vector
to sum to the trajectory length, state counts to agree with state-aggregated
pair counts, or individual counts to lie in the theorem's domain `k <= n`.

The following deterministic counterexample was reproduced against the sealed
module:

```text
trajectory_length = 8
state_counts = [200, 200]
pair_counts = [100, 100, 100, 100]
```

It returned `event_valid=true`, pair radius `2.008843708304334`, and a Direct
exact status of `selective_high_probability_certified`. This invokes a radius
at `k=100` even though the theorem only unions over `1 <= k <= 8`. The public
`hoeffding_radius` correctly rejects the same domain error, but
`shared_visit_event` bypasses that check. This is a real validity risk and fails
criterion 7.

## Failure 4: smoke alignment was not actually checked

Claude's smoke run used two seeds per cell as the seed-spawn stride. Only four
of its 16 records share frozen baseline keys; twelve are `new_only`. The smoke
regression file records exactly `compared_records: 4` and
`new_only_keys: 12`, yet reports `passed: true` because the analyzer only
requires a nonempty key intersection in smoke mode.

Consequently the stated 16/16 aligned smoke regression is false, and the formal
run began without the required passing aligned smoke gate. The repair must
always spawn the frozen 30 seeds per cell and select the requested leading task
indices, then require every smoke key to be a baseline key. Criterion 11 fails.

## Failure 5: legacy config/summary and schema checks are incomplete

The analyzer compares only `task_results.json`. It never compares the legacy
config or summary. It then replaces `summary.json` with a new dictionary,
whereas the frozen summary is a 176-row list. Thus the zero-mismatch claim is
true for legacy task-record leaves but false as a claim of complete legacy
output preservation.

The analyzer also lacks a full namespace/schema audit capable of detecting the
hidden provenance of `value_bound`, and its residual audit compares only a
global residual supremum with a worst-group radius rather than checking every
group against its own radius. The repair must preserve the old summary rows,
add new fields only under `visit_indexed_certificate`, compare all old config,
record, and summary leaves, reject duplicate keys, and audit per-group
residuals. Criteria 9 and 13 fail; criteria 15 and 16 remain incomplete.

## Failure 6: execution evidence was not sealed in the required order

The sealed Claude commit has the activation commit as its direct parent and
adds implementation, proof, smoke conclusions, and formal conclusions at once.
There is no preceding commit sealing a passing implementation and smoke result
before the formal run, as required by plan Tasks 9 and 10. The original formal
`environment.json` records HEAD at the activation commit, not a committed
implementation. The ignored raw outputs are named by path but their hashes are
absent from the sealed first-result evidence.

The branch is reproducible, but its history cannot establish that the formal
output was first inspected only after a passing implementation seal, nor can
the sealed document content-address the ignored results. Criterion 16 therefore
fails. The repair must first commit code/tests/proof, run from that exact commit,
then commit a result record containing exact paths and hashes.

## Reporting differences requiring reconciliation

Claude defines nontriviality as `total_bound < 2B`; GPT preregistered
`total_bound < B`, the computable zero-initialization error bound. The frozen
task did not choose a threshold, so this is not itself a task objection. The
rates cannot be compared without relabelling. Under the common `<B` rule,
Claude's current numbers reduce to zero for both Direct routes and all lengths,
zero for both V-first routes below length 16384, then 48/120 for V-first exact
and 4/120 for V-first softmax at 16384, exactly matching GPT's counts. The
repair and final synthesis should report `<B` as the primary common metric and
may retain `<2B` as a clearly labelled secondary metric.

Claude's optional-route text says that no oracle-free observable variance proxy
exists. The supplied analysis establishes only that no such proxy was proved
or implemented from the frozen allowed inputs. The repaired text must use that
narrower conclusion; this optional issue does not reject the mandatory route.

Differences in `not_emitted` versus `not_certified`, generic versus route-
specific missing-support names, and namespace nesting are nonblocking if their
semantics remain explicit and the full schema audit passes.

## Acceptance-criteria mapping

| Criterion | Verification of sealed Claude route |
| ---: | --- |
| 1 | `FAIL`: correct filtration and range setup, but invalid MGF iteration remains in the written proof. |
| 2 | `FAIL`: risk arithmetic is correct, but currently depends on the proof gap. |
| 3 | `PASS_COMPONENT`: simultaneous random-count lookup is correctly stated once the MGF step is repaired. |
| 4 | `PASS`: selective semantics are exact; no support or conditional-coverage overclaim. |
| 5 | `FAIL`: true reward table determines certificate `B`. |
| 6 | `PASS`: recurrences agree with the frozen deterministic result after normalizing the differing `B`. |
| 7 | `FAIL`: invalid and out-of-horizon counts can emit. |
| 8 | `PASS`: record additions use the required namespace and selective status. |
| 9 | `FAIL`: hidden oracle provenance and incomplete schema audit. |
| 10 | `PASS_COMPONENT`: all five verifiers pass, but they omit the failing contracts. |
| 11 | `FAIL`: only 4/16 smoke records align while the gate reports success. |
| 12 | `PASS`: formal result contains the exact 480 frozen task keys and seed. |
| 13 | `FAIL`: task records match, but config/summary preservation is not fully analyzed and summary is replaced. |
| 14 | `PASS_COMPONENT`: emission counts and deviations are correct; nontriviality needs a common label. |
| 15 | `PARTIAL`: independent audit found zero violations, but Claude's own per-group audit is incomplete. |
| 16 | `FAIL`: missing pre-formal seal and raw-output hashes; runtime HEAD is the activation commit. |
| 17 | `FAIL`: this reciprocal verification report requires author repair and re-verification. |
| 18 | `PENDING`: synthesis and workspace update must wait for repair and both final passes. |

## Required author repair

Claude must repair only its own branch and allowed paths. At minimum it must:

1. replace the invalid MGF step by the compensated exponential
   supermartingale and narrow the optional variance claim;
2. derive `B` solely from the public `R_star = 1 + gap_bonus` rule;
3. validate integer counts, horizon totals, dimensions, and state/pair
   consistency with new failing-then-passing tests;
4. preserve the frozen 30-seed stride in smoke and require all 16 keys to align;
5. preserve and regress every old config, task-record, and summary leaf, add
   complete namespace/oracle/per-group audits, and record both nontriviality
   labels without selection;
6. commit the repaired implementation before rerunning smoke and formal, then
   seal exact result paths, hashes, metrics, anomalies, and acceptance mapping
   in a second commit.

The mandatory scientific hypothesis is not rejected; the current Claude
implementation and evidence are not yet acceptable.

FAIL
