# FP-MART-001 GPT route: blind formal result

## Identity and status

- Status: `PRELIMINARY_POSITIVE`, pending reciprocal verification.
- Common execution-start commit:
  `c8ec7e5c3165930663e26a07f98b38cc9ec186ad`.
- Certificate implementation commit:
  `4cf6f50d69c5aa4937d181f758f546f9d2213c41`.
- GPT first-result seal commit:
  `e7c04111b7aec8c9dc5043883fcaa68cc158837b`.
- Repository HEAD recorded by the formal run:
  `fef32e1ef85a4607de72c00acefda4f030eb24fa`.
- Formal-evidence seal commit: this document's commit; its SHA is recorded in
  the frozen task evidence immediately after creation.
- Blindness: no Claude theory, code, output, result, or conclusion was read
  before this formal result was generated, analyzed, and written.
- This was the first and only GPT formal run. No frozen parameter, seed, risk
  allocation, threshold, or stopping rule was changed after output inspection.

## Exact formal commands

Working directory:
`C:\Users\Admin\Desktop\research\icrl_softmax`.

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_visit_indexed_certificates.py --tasks 30 --trajectory-lengths 256 1024 4096 16384 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir results/FP-MART-001/codex

C:\Users\Admin\anaconda3\python.exe -B analyze_visit_indexed_certificates.py --baseline-dir results/fixed_policy_finite_sample_certificates --result-dir results/FP-MART-001/codex --mode formal
```

The matrix contains exactly 480 records: 30 task seeds, four trajectory
lengths, two mixing settings, two reward-gap settings, and one value for every
other frozen parameter. The environment was recorded at
`2026-09-03T08:02:57.835284+00:00` on Windows 11 with Python 3.13.9, NumPy
2.4.6, SciPy 1.16.3, and Matplotlib 3.10.6.

## Gate results

The evaluator verified the three frozen baseline hashes and the required 480
baseline records before sampling, then verified the full baseline directory
inventory again after evaluation. All five preflight verifiers passed.

The strict formal analyzer returned `passed: true`:

- actual and configured records: 480;
- matched legacy records: 480/480;
- legacy task-record mismatch count: zero;
- legacy config mismatch count: zero;
- legacy summary mismatch count: zero;
- maximum legacy numeric absolute difference: zero for all three comparisons;
- namespace problem count: zero;
- new schema/oracle-separation problem count: zero;
- exact-route high-length emission gate: passed;
- empirical fixed-target residual or emitted-route violation count: zero;
- all files parsed and serialized as strict JSON.

The baseline hashes remained:

- `config.json`:
  `a2276eae06ba8689864cac2ad3d3d0e92b069014b2046d0180ee172bb8296ea0`;
- `task_results.json`:
  `c84329bd6b9fcd495789f2067f250ead28fec807193d1b69ae1038e9cc3b2f25`;
- `summary.json`:
  `49846c0825c859df28ff773164352d19f6cb86943c484ab411bac7f69434b5ae`.

## Emission and usefulness results

`Nontrivial` means the preregistered, descriptive condition
`emitted total_bound < B`. It is not part of selective validity.

| Length | Route | Emitted | Emission rate | Nontrivial | Mean emitted bound |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | Direct exact | 3/120 | 2.500% | 0/120 | 66.6630 |
| 256 | Direct softmax | 3/120 | 2.500% | 0/120 | 140.4449 |
| 256 | V-first exact | 3/120 | 2.500% | 0/120 | 30.3746 |
| 256 | V-first softmax | 3/120 | 2.500% | 0/120 | 31.2992 |
| 1024 | Direct exact | 102/120 | 85.000% | 0/120 | 55.0496 |
| 1024 | Direct softmax | 74/120 | 61.667% | 0/120 | 544.2354 |
| 1024 | V-first exact | 102/120 | 85.000% | 0/120 | 22.2990 |
| 1024 | V-first softmax | 102/120 | 85.000% | 0/120 | 23.7701 |
| 4096 | Direct exact | 120/120 | 100.000% | 0/120 | 21.0677 |
| 4096 | Direct softmax | 115/120 | 95.833% | 0/120 | 70.6491 |
| 4096 | V-first exact | 120/120 | 100.000% | 0/120 | 9.1599 |
| 4096 | V-first softmax | 120/120 | 100.000% | 0/120 | 9.9862 |
| 16384 | Direct exact | 120/120 | 100.000% | 0/120 | 9.8917 |
| 16384 | Direct softmax | 120/120 | 100.000% | 0/120 | 25.3071 |
| 16384 | V-first exact | 120/120 | 100.000% | 48/120 | 4.4291 |
| 16384 | V-first softmax | 120/120 | 100.000% | 4/120 | 5.0870 |

The exact-route rates are exactly the preregistered support rates: 2.5%, 85%,
100%, and 100%. Every deviation from those four reference rates is zero. This
agreement was not produced by tuning: exact emission is mechanically equivalent
to observed full pair support under the frozen gates.

The aggregate emitted counts are:

- Direct exact: 345/480;
- Direct softmax: 312/480;
- V-first exact: 345/480;
- V-first softmax: 345/480.

Direct-softmax lost another 28 records at length 1024 and five records at
length 4096 to a nonpositive empirical pair-kernel margin. V-first softmax does
not iterate the pair recovery operator and therefore correctly needs no pair
contraction-margin gate; its state stage passed whenever full pair support was
present in these data.

The practically useful result appears only in the longest trajectory regime:
V-first exact improves on `B` in 48/120 records (40%), and V-first softmax in
4/120 records (3.33%). Direct-Q remains above `B` in every emitted record. Thus
the formal evidence supports the predicted structural advantage of the
V-first composition, while also showing that the Hoeffding route is generally
conservative.

## Empirical audits and probability interpretation

Across all 480 records there were zero fixed-target residual event violations
and zero violations among all 1,347 emitted route certificates at tolerance
`1e-9`. These are diagnostic observations only. They do not prove the theorem,
do not estimate conditional coverage, and do not change the required statement

```text
P(Emit and certified error bound is violated) <= delta
```

for each frozen trajectory record.

## Formal ignored-output hashes

Canonical directory:
`C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `checks.log` | 3848 | `358b4d35e9460642672e9b147dc000f67c5567c0f8794200f6ade7d8677e8177` |
| `commands.log` | 615 | `6167fd1ffcd12e1e4c6eec95b721a775c6f3ec652856e5fdc249ebc519ddafe7` |
| `config.json` | 1438 | `bcc377b422711302163b976d5d5ece389e1e3ee8540d37817a8719bf9ec3bf8a` |
| `environment.json` | 2229 | `5b260049cdda2386f3eaa9a8752519b5fbd05e234d33990a2490001ee4594a05` |
| `regression.json` | 12442 | `404a52bf6ed0468f95c43b1804a384486e2d45d2c1709e942aefaeed6f4dfa71` |
| `summary.json` | 240831 | `fa13619755b627b9dcff281dc5e4bc9b4d0262b9a6592f2982012b53df2b34ed` |
| `task_results.json` | 26345474 | `929e2f65689af850b65f000ee6675a8a87b3506c2e2a9600c07d28cf183c6d52` |

## Acceptance assessment after the GPT formal run

- Criteria 1--10: `PASS_GPT`; Claude evidence remains undisclosed.
- Criterion 11: `PASS_GPT`; Claude smoke evidence remains undisclosed.
- Criterion 12: `PASS_GPT`; exactly 480 frozen records and seed schedule.
- Criterion 13: `PASS_GPT`; task records, config, and summary have zero legacy
  mismatch.
- Criterion 14: `PASS_GPT`; all emission, nontriviality, and deviation rates are
  reported, with 100% exact emission at both required high lengths.
- Criterion 15: `PASS_GPT`; every audit violation is enumerated and the count is
  zero without using coverage as proof.
- Criterion 16: `PASS_GPT`; exact versions, environment, commands, raw-output
  paths and hashes, anomalies, metrics, conclusion, and limitations are
  recorded.
- Criteria 17--18: `PENDING`; reciprocal reproduction, final synthesis, and
  workspace update have not begun. `main` remains unchanged.

## Limitations and next step

The certificate remains scoped to the frozen fixed-policy deterministic-reward
protocol, is per trajectory rather than simultaneous over the 480-record audit,
and is often numerically loose. The optional variance-adaptive route remains
unavailable because counts alone do not provide its required observable
variance process.

The next action is to commit this blind formal evidence, confirm the independent
Claude first-result seal without first reading its contents, and only then
start reciprocal reproduction and comparison.
