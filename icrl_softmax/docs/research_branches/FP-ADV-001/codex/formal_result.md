# FP-ADV-001 Codex formal result

Date: 2026-09-08  
Route: `codex/FP-ADV-001`  
Status: `PRELIMINARY VERIFIED-NEGATIVE CANDIDATE`; Codex route checks pass, but
Claude disclosure and reciprocal verification are still pending.  
Blind first-result seal: `996641d9ce2088e95c0bf2a0661e3b24b6e0d6fe`.  
Post-formal ordinary implementation fix:
`440710c8c9c77ee8128d4abfd96919ffafe7991c`.

## Formal protocol and command

The frozen evaluator was run exactly once, after the blind first-result seal:

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_action_gap_certificates.py --tasks 30 --trajectory-lengths 256 1024 4096 16384 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --transfer-fraction 0.5 --seed 20260829 --output-dir results/FP-ADV-001/codex
```

It produced 480 matched records and exited successfully. The unchanged
inherited certificate route emission counts were:

- Direct exact: 345/480;
- Direct softmax: 312/480;
- V-first no-split exact: 345/480;
- V-first no-split softmax: 345/480.

The strict analyzer command then passed:

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_action_gap_certificates.py --result-dir results/FP-ADV-001/codex
```

## Detected representation defect and bounded repair

The first analysis showed that all six action-gap routes correctly abstained,
but the builder always recomputed the selected receiver as one minus the other
row entries. With no eligible donor this caused a representation-only change
of at most `6.938893903907228e-17` instead of returning the input policy
literally. All 2,880 route records were affected. No route emitted, and no
ordering, mass transfer, estimator, bound, trajectory, threshold, or empirical
conclusion changed.

This ordinary implementation defect was fixed by preserving the original row
when a state has no eligible donor, with a new exact-identity regression test.
The already saved observable inputs were then used to repair only the new
action-gap serialization and its identity-policy oracle consequences:

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_action_gap_certificates.py --result-dir results/FP-ADV-001/codex --repair-no-update-identity
```

The frozen 480-record evaluator was not rerun. The repair reported exactly
2,880 corrected no-update policy serializations, after which the strict
analyzer passed again. Every no-update policy now equals its original policy
element by element, and exact/softmax updated-policy agreement is 480/480.

## Frozen preservation and structural validation

- Frozen FP-TU baseline count: 480.
- All three required FP-TU SHA-256 hashes matched.
- Legacy config: 40 numeric and 28 nonnumeric leaves compared, zero mismatch.
- Legacy task results: 762,882 numeric and 213,083 nonnumeric leaves compared,
  zero mismatch.
- Legacy summary: 4,966 numeric and 1,384 nonnumeric leaves compared, zero
  mismatch.
- New pure-certificate reconstruction: 1,120,685 numeric and 363,996
  nonnumeric leaves compared, zero mismatch.
- Oracle input findings: 0.
- Invalid policies, exploration-floor failures, row-sum failures, or negative
  Bellman LCBs: 0.
- Local/global penalty-dominance violations: 0.
- Local/global decision-dominance violations: 0.
- Strict JSON, duplicate-key rejection, finite serialization, route order,
  task identity, formal matrix, result inventory, and provenance: `PASS`.

## Formal empirical result

Every action-gap route abstained on every record:

| Route | Update records | Eligible donors | Updated states |
|---|---:|---:|---:|
| V-first local exact | 0/480 | 0 | 0 |
| V-first local softmax | 0/480 | 0 | 0 |
| V-first global exact | 0/480 | 0 | 0 |
| V-first global softmax | 0/480 | 0 | 0 |
| Direct-Q global exact | 0/480 | 0 | 0 |
| Direct-Q global softmax | 0/480 | 0 | 0 |

For the primary local routes, 135 records included a selected unvisited pair;
13 records included a non-emitted state certificate; and 467 records included
nonpositive local action-gap LCBs. These route-level reason counts may overlap.
No threshold or route was changed after observing them.

Because no ordering was used and every returned policy is exactly the original
policy, all enumerated oracle counts are zero for all six routes:

- false orderings: 0;
- Bellman-bound violations: 0;
- componentwise value decreases: 0;
- exact-return decreases: 0.

The local penalties were always no wider than matching complete-Q V-first
controls whenever comparable. The claimed decision dominance holds vacuously:
global controls also emitted no update.

## Hypothesis assessment

1. Exact V-first local formula: supported by proof, unit fixtures, full
   reconstruction, and zero validation mismatch.
2. Finite-softmax effective-row and contamination formula: supported by proof,
   dense-weight/large-beta fixtures, and full reconstruction.
3. Post-data selection under the reused simultaneous event: supported as a
   deterministic consequence; no risk resplit is serialized.
4. Local penalty no wider than matching global V-first control: supported with
   zero formal violations.
5. Pointwise safe-update theorem: supported by proof and tests; no formal update
   was emitted, so the empirical policy audit is identity-only.
6. Localized support handling: supported by tests and serialized state-local
   outcomes.
7. Exact FP-TU preservation: supported by the zero-mismatch full regression.
8. At least one useful primary local update: falsified, with 0/480 emissions on
   both primary routes. This is the frozen valid negative outcome.

## Final Codex-route artifact hashes

Ignored result directory:
`C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-ADV-001\codex\`.

- `config.json`:
  `0e1c7990d93d9cd61f37214535b5d8291a33ab2997bba968b3d341352124c5d5`;
- `task_results.json`:
  `198dd141852d73a85349a2f310332be41551c623e659816a639ee8c276f1b464`;
- `summary.json`:
  `9d438717693363e996c9db1242257a3931af57febb4c827e1a463d9c5faa0675`;
- `regression.json`:
  `b1a2a6541d3b6f94e2eb36fb1cc5beb427606c34d13a592382e0f8e70b2649b8`;
- `environment.json`:
  `406ea53429e6191a2e3d79c0b1f80f0d079c4fad0bad59aaee84bcc963492cf9`;
- `commands.log`:
  `8ad536f9806226766a2644ee5d39e7a31e1a630b536320b904cde75a93a3be9a`;
- `checks.log`:
  `d3399c1b799f83f5b752426f8f3b1d1dd92547a33682342cc54f32f152ea5bbf`.

## Post-seal validation hardening

After the formal matrix and its representation-only repair, the Codex working
tree added validation-only safeguards without rerunning or changing the 480
records:

- the analyzer now reconstructs every MDP, policy, trajectory, estimator,
  inherited bound, certificate input, oracle audit, and action-summary row from
  the frozen seed schedule;
- normal analysis is read-only, evaluator output refuses non-empty directories,
  and the one-time identity repair requires a task-results prehash and uses
  atomic JSON replacement;
- count inputs must retain an integer source dtype; the verifier adds exhaustive
  finite-distribution span checks, multi-donor conservation, and malformed,
  nonfinite, and float-count rejection cases.

The strengthened verifier and Ruff pass. A fresh eight-record smoke evaluation
and read-only smoke analysis pass. The strengthened analyzer replays all 480
formal records and passes with zero provenance, oracle, formula, policy,
dominance, and summary mismatches. The formal matrix was not rerun. The
post-hardening evidence hashes are:

- `regression.json`:
  `2e964320910261e24ed0e70fbde785f139830234c465087c2a258e01d15cc77e`;
- `commands.log`:
  `13284d79e26da87153d01e34e8e8edbd3cbab5925fc425ac799c3b8deef4ece0`;
- `checks.log`:
  `efc2a07ed0416005c80019f95e94ce3bfcf5728d7688cf01cfaca4a917a5d0a6`.

The code hardening and reciprocal-verification record were committed at
`46cbf9bca8dab87161ffee7a1c38ead162d73ad6`. The unrelated user-owned
untracked learning record remains untouched.

## Acceptance status

Codex-route acceptance criteria 1--19 pass with the evidence above, except that
criterion 15 also requires the independent Claude smoke seal and criterion 19
requires Claude's corresponding evidence. Criterion 20—reciprocal
reproduction, both final reports, reconciliation, and workspace completion—is
pending. The task must remain `ACTIVE`; this result cannot be called verified
until the independent route is disclosed and both verifiers pass.
