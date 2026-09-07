# FP-MART-001 GPT route: sealed first result

## Identity and independence

- Result status: `PRELIMINARY_POSITIVE` for the mandatory visit-indexed
  Hoeffding construction.
- Common execution-start commit:
  `c8ec7e5c3165930663e26a07f98b38cc9ec186ad`.
- Exact implementation commit used by the sealed smoke run:
  `4cf6f50d69c5aa4937d181f758f546f9d2213c41`.
- First-result seal commit: this document's commit; its SHA is recorded in the
  frozen task evidence immediately after the commit is created.
- Independence statement: the GPT route did not read Claude's theory, code,
  output, first result, conclusion, or working tree before sealing this result.
  Codex sub-agents were used only for bounded GPT-side theorem, protocol, and
  code-map checks and do not count as Claude or as final independent
  verification.
- Formal 480-record output and both cross-verification reports were not
  inspected when this first result was sealed.

## Preliminary conclusion

The mandatory construction succeeds under the frozen scope. For each of the
state Bellman, pair Bellman, and fixed-`V^pi` recovery residual families, the
appropriate visit indicator is predictable under the stated pre-observation
filtration. A directly constructed exponential supermartingale, stopped at the
bounded `k`th-visit time, gives the two-sided radius

```text
r_H(k, delta) = B sqrt(2 log(2 G n / delta) / k),
G = m + 2d.
```

A union over every group and every `1 <= k <= n` permits pathwise lookup at the
random observed final count. On the resulting single event, the unchanged
Direct-Q and V-first no-split deterministic recurrences give the four exact and
finite-softmax route bounds. Therefore the supported claim is exactly

```text
P(Emit and certified error bound is violated) <= delta.
```

The construction makes no high-probability claim about support or emission and
does not claim conditional coverage. The pure certificate receives no true
kernel, occupancy, spectral quantity, value, residual, or true initial error.

The proof is in `theory.md`. Its imported concentration ingredients are traced
to the primary papers by Hoeffding (1963), Azuma (1967), and Freedman (1975).
Freedman's inequality is not used by the mandatory result: the allowed inputs
do not contain a proved observable predictable-variance proxy, so the optional
route is explicitly recorded as unavailable and receives no risk budget.

## Test-first and implementation evidence

The verifier was created before the certificate module. Its first run failed as
expected with `ModuleNotFoundError: No module named
'visit_indexed_martingale_certificate'`. After implementation it passed. A
later review added count-consistency and metadata-type rejection tests; the new
count test first failed with `AssertionError: expected ValueError`, then passed
after the certificate began rejecting incompatible state and pair counts.

The exact implementation commit contains these new files:

- `visit_indexed_martingale_certificate.py`;
- `verify_visit_indexed_martingale_certificate.py`;
- `evaluate_visit_indexed_certificates.py`;
- `analyze_visit_indexed_certificates.py`;
- `docs/research_branches/FP-MART-001/codex/theory.md`.

The final sealed smoke run executed all five preflight verifiers. Each exited
zero:

1. `verify_finite_sample_theorems.py`;
2. `verify_fixed_policy_q_routes.py`;
3. `verify_crossfit_markov_certificate.py`;
4. `verify_end_to_end_sarsa.py`;
5. `verify_visit_indexed_martingale_certificate.py`.

The four new Python entry points also passed Ruff, and the staged change passed
`git diff --cached --check`.

## Sealed smoke commands and environment

Working directory:
`C:\Users\Admin\Desktop\research\icrl_softmax`.

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_visit_indexed_certificates.py --tasks 2 --trajectory-lengths 256 4096 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir results/FP-MART-001/codex/smoke

C:\Users\Admin\anaconda3\python.exe -B analyze_visit_indexed_certificates.py --baseline-dir results/fixed_policy_finite_sample_certificates --result-dir results/FP-MART-001/codex/smoke --mode smoke
```

The environment record was created at
`2026-09-03T07:56:20.319124+00:00` and identifies Windows 11, Python 3.13.9,
NumPy 2.4.6, SciPy 1.16.3, and Matplotlib 3.10.6. It records the exact
implementation commit above and the common activation commit.

Before any baseline record was sampled, the evaluator verified the frozen
baseline hashes and the required count of 480 records. It verified the complete
baseline directory inventory again after evaluation and found it unchanged.

## Smoke results

The smoke matrix produced 16 matched records: two task seeds, two trajectory
lengths, two mixing values, two gap bonuses, and all other formal settings
fixed. Its seed schedule uses the frozen 30-seed stride per cell, so every smoke
record has the same spawn key as its formal baseline counterpart.

| Length | Route | Emitted | Pair support | Nontrivial | Oracle violations |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | Direct exact | 0/8 | 0/8 | 0/8 | 0 |
| 256 | Direct softmax | 0/8 | 0/8 | 0/8 | 0 |
| 256 | V-first exact | 0/8 | 0/8 | 0/8 | 0 |
| 256 | V-first softmax | 0/8 | 0/8 | 0/8 | 0 |
| 4096 | Direct exact | 8/8 | 8/8 | 0/8 | 0 |
| 4096 | Direct softmax | 7/8 | 8/8 | 0/8 | 0 |
| 4096 | V-first exact | 8/8 | 8/8 | 0/8 | 0 |
| 4096 | V-first softmax | 8/8 | 8/8 | 0/8 | 0 |

At length 256 every route rejected because pair support was missing. At length
4096 the one Direct-softmax rejection had a nonpositive empirical pair-kernel
margin. All other supported routes emitted finite bounds. No emitted route or
fixed-target residual audit violated its corresponding bound at tolerance
`1e-9`.

The preregistered usefulness metric was
`emitted total_bound < B`. Its smoke rate was zero for every route. Thus the
smoke results support validity and implementation behavior but do not yet show
a practically sharper guarantee than the zero-initialization bound. This
looseness is reported separately and is not relabelled as a validity failure.

The strict analyzer passed with:

- 16/16 matched baseline records;
- zero legacy mismatches under exact nonnumeric comparison and
  `math.isclose(rel_tol=1e-12, abs_tol=1e-12)` for numeric leaves;
- maximum common numeric absolute difference
  `4.305888978706207e-12`, still within the specified relative tolerance;
- zero namespace problems;
- zero new-schema or oracle-separation problems;
- zero empirical audit violations;
- strict JSON throughout.

## Sealed ignored-output hashes

Canonical smoke directory:
`C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex\smoke`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `checks.log` | 3847 | `e5c2f9457632d8931484622f82eaf666deb78ad094650af85ba41c86a229cf27` |
| `commands.log` | 614 | `2b2ebb54ac4772e9f34be5fe46c28a2e8e8d67aefadd5ccd9a24cd8caba15ca7` |
| `config.json` | 1414 | `e535cc7a5f68d74f655fc08a5f07cbd9edbb54a2adee5a9ff850642395c37fe6` |
| `environment.json` | 2229 | `de52558372d417b4c73b660947236de1d10903685c497853563efb4979dec148` |
| `regression.json` | 6846 | `ea662719d7b7106263879e83c80f7d2587f21833a59f312f140869243510c20f` |
| `summary.json` | 117181 | `32bbac6a415bf1b117ebeaf328bb525437f7417ed007e5e63eb120036a6acae2` |
| `task_results.json` | 872553 | `ab0c26d94dd9db5d055f44cdb067fe12d3efd0a5e0721ad55c5c11ffd554dbf6` |

## Anomalies and repairs before sealing

1. The initial test-first missing-module failure and the later count-validation
   failure were expected red-phase evidence; both are resolved.
2. The first exploratory smoke evaluator spawned only the requested two seeds
   per cell. That changed later cells' spawn keys, so the analyzer correctly
   rejected the comparison: only four records aligned and twelve were
   new-only. The implementation was repaired to always spawn the frozen 30
   seeds per cell and select the requested leading task indices. The sealed run
   then aligned all 16 records with zero mismatches.
3. Ruff initially reported three unused imports. They were removed; the sealed
   check passes.
4. A bounded GPT-side auxiliary implementation review hit its Codex usage limit
   before returning findings. It is not counted as a completed review or as
   Claude verification. The primary GPT agent manually reviewed the complete
   implementation, strengthened count consistency and namespace/oracle gates,
   and reran the verifier, linter, evaluator, and analyzer successfully.

No anomaly changed a frozen scientific parameter, probability budget, seed,
acceptance rule, or stopping rule. No failed exploratory output is used as
evidence.

## Numbered acceptance assessment at first-result seal

| Criterion | GPT-route assessment |
| ---: | --- |
| 1 | `PASS_GPT`; all three filtrations, stopping times, ranges, and stopping argument are explicit. Claude route pending. |
| 2 | `PASS_GPT`; one `G*n` event with total failure at most `delta`. Claude route pending. |
| 3 | `PASS_GPT`; random counts enter only by simultaneous lookup. Claude route pending. |
| 4 | `PASS_GPT`; selective semantics are exact and support is not given a prior guarantee. Claude route pending. |
| 5 | `PASS_GPT`; the pure interface contains no oracle input and initializes the error bound at `B`. Claude route pending. |
| 6 | `PASS_GPT`; the frozen Direct-Q and V-first recurrences are composed unchanged. Claude route pending. |
| 7 | `PASS_GPT`; support, margins, mode, divergence, input consistency, and finite arithmetic have deterministic gates. Claude route pending. |
| 8 | `PASS_GPT`; all additions are under `visit_indexed_certificate` and use the new selective status. Claude route pending. |
| 9 | `PASS_GPT`; strict JSON and structural oracle separation pass. Claude route pending. |
| 10 | `PASS_GPT`; all five GPT preflight verifiers pass. Claude branch pending. |
| 11 | `PASS_GPT`; GPT smoke passes all required audits. Claude smoke pending. |
| 12 | `PENDING`; neither formal result is admitted at this seal. |
| 13 | `PASS_SMOKE_ONLY`; formal 480-record regression pending. |
| 14 | `PENDING`; formal emission and usefulness rates pending. |
| 15 | `PASS_SMOKE_ONLY`; zero smoke violations are fully reported, formal audit pending. |
| 16 | `PARTIAL`; GPT smoke commit, environment, commands, outputs, anomalies, metrics, and limitations are recorded; formal and Claude evidence pending. |
| 17 | `PENDING`; disclosure and reciprocal reproduction have not begun. |
| 18 | `PENDING`; final synthesis and workspace update follow cross-verification; `main` remains unchanged. |

## Limitations and next step

- The theorem is scoped only to the frozen fixed-policy, deterministic-edge-
  reward, fixed-context synchronous setting.
- The guarantee is per trajectory record, not joint over the 480-record audit.
- Smoke bounds are valid but vacuous relative to `B` under the preregistered
  usefulness metric.
- No optional variance-adaptive certificate is available from counts alone.
- The result remains preliminary until both routes seal formal evidence and
  reciprocal verification ends in `PASS` or the user records an exception.

The next GPT action is the exact frozen 480-record formal run from the sealed
implementation, followed by strict analysis. Claude's result remains
undisclosed until both first-result seals exist.
