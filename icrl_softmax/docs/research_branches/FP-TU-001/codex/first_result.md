# FP-TU-001 GPT blind first result

Status: `SEALED_PRE_FORMAL_INITIAL_RESULT`.

This result was constructed without reading Claude's implementation, first
result, or result directory.  It is preliminary until the frozen formal run
and reciprocal verification are complete.

## Identity and environment

- Common execution start:
  `0ce18b4676f70ca0556496e804aa65563efa63da`.
- GPT implementation commit:
  `4656fd30b24e410241990665f93d93bf2ece564b`.
- Branch: `codex/FP-TU-001`.
- Python: 3.13.9 at `C:\Users\Admin\anaconda3\python.exe`.
- NumPy/SciPy/Matplotlib: 2.4.6/1.16.3/3.10.6.
- Ruff: 0.12.0.
- Platform: Windows 11, CPU-only.

## Test-first evidence

Before `time_uniform_mixture_certificate.py` existed, this command was run:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_time_uniform_mixture_certificate.py
```

It exited 1 with the expected
`ModuleNotFoundError: No module named 'time_uniform_mixture_certificate'`.
After implementation, the same command passed.  The verifier exhaustively
checks all counts 1 through 16384, independently recomputes selected roots at
60-decimal precision, enumerates bounded centered two-point MGF fixtures, and
checks composition, validation, JSON, and the unallocated-minimum
counterexample.

The five inherited verifiers all exited 0:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_finite_sample_theorems.py
C:\Users\Admin\anaconda3\python.exe -B verify_fixed_policy_q_routes.py
C:\Users\Admin\anaconda3\python.exe -B verify_crossfit_markov_certificate.py
C:\Users\Admin\anaconda3\python.exe -B verify_end_to_end_sarsa.py
C:\Users\Admin\anaconda3\python.exe -B verify_visit_indexed_martingale_certificate.py
```

Ruff also passed on all four new entry points.

## Smoke run

The final-code smoke command was:

```text
C:\Users\Admin\anaconda3\python.exe -B evaluate_time_uniform_certificates.py --tasks 1 --trajectory-lengths 256 1024 4096 16384 --n-states 6 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --gamma 0.70 --alpha 0.65 --iterations 160 --certificate-delta 0.05 --seed 20260829 --output-dir results/FP-TU-001/codex/smoke
```

It produced 16 matched records and passed.  The analyzer command was:

```text
C:\Users\Admin\anaconda3\python.exe -B analyze_time_uniform_certificates.py --result-dir results/FP-TU-001/codex/smoke --baseline-dir results/FP-MART-001/codex
```

It passed strict JSON, frozen-baseline integrity, additive namespace,
emission preservation, nonincreasing emitted bounds, oracle separation, and
the complete count audit.  Smoke emissions were 11/16 Direct exact, 9/16
Direct softmax, 11/16 V-first exact, and 11/16 V-first softmax.  No emitted
route or fixed-target residual audit violation occurred.

The exhaustive formula audit found:

| Legacy horizon | maximum `r_mix/r_old` |
| ---: | ---: |
| 256 | 0.9573573993365391 |
| 1024 | 0.9263355819163681 |
| 4096 | 0.8975200574107253 |
| 16384 | 0.8793886652750217 |

The maximum `q_mix/q_stitch` was 0.9959405625588716 and the largest bisection
count was 42, below the frozen cap of 200.  Across observed smoke groups,
`r_mix/r_old` ranged from 0.7379884021918313 to 0.9407923658991243, with mean
0.8501884876211377.

## Artifact hashes

- `time_uniform_mixture_certificate.py`:
  `47806c55f644b1618f9f32df50525257d7a8c2d44b5425507eba366145714fbc`.
- `verify_time_uniform_mixture_certificate.py`:
  `a884436eed06d472e01e0502be0867c219f649b43b98687df93c59eaebd12655`.
- `evaluate_time_uniform_certificates.py`:
  `9de2041eef3b45000358c8b2340015660b05fca07857fe9b1505f4d70c3c8813`.
- `analyze_time_uniform_certificates.py`:
  `4dd7e84e1b52c07e26068d36ec69e0d0ac89ecb01f315cb47effa9deae23fe5d`.
- Smoke `config.json`:
  `56ec771550d4168c1fbfd109368b93abd4b02e21d8cfe6aa457019bc99d37c1b`.
- Smoke `task_results.json`:
  `a7e2d85cba6ad69a4ee623ff72c548e7a5f2365758bdc6686074a362b1a6a980`.
- Smoke `summary.json`:
  `c9f7871e54768229f291f480c3f7e5a998ad92f49defbaae6fd84201236f9e47`.
- Smoke `regression.json`:
  `0e91171047f0a74151e4d30852c30a82455bf4032a42bcde446ec1c881e41f40`.

All seven smoke core artifacts are under
`results/FP-TU-001/codex/smoke/`; this ignored directory is separate from the
future canonical formal result.

## Transition-variance assessment

No complete observable construction was established.  The unresolved proof
obligation is a simultaneously valid observable transition-row confidence
set plus a uniform optimization of the nonlinear variance functional over all
`V in [-B,B]^m`, including unseen successors.  Counts alone do not give the
predictable variance of the unobservable fixed-target residual sequence.
Accordingly this idea has zero allocated risk and is not part of the mandatory
certificate.

## Numbered acceptance assessment

1. Filtration and stopping-time mapping: pass in the independent theory.
2. Cosh-mixture group/time guarantee: pass in theory and verifier.
3. Random-count and selective semantics: pass.
4. Independent stitch proof and bracket: pass.
5. No-oracle pure interface: pass; builder delegates only deterministic legacy validation.
6. Counts 1--16384 inversion, finiteness, monotonicity, and stitch dominance: pass.
7. Per-record and global-horizon old-radius dominance: pass.
8. Unchanged recurrences and nonincreasing smoke bounds: pass.
9. Ordered support/margin/mode/risk/count handling: pass for tested paths.
10. Additive `time_uniform_certificate` namespace: pass.
11. Strict JSON and oracle separation: pass.
12. Six verifiers and Ruff: pass.
13. GPT smoke before formal and blind first seal: pass when this evidence commit is created.
14. Frozen 480-record identity: pending formal run.
15. Full legacy zero-mismatch regression: pending formal run.
16. Frozen exact emissions and full descriptive metrics: pending formal run.
17. Smoke empirical violations enumerated and excluded from proof: pass.
18. Transition variance feasibility gap recorded and excluded: pass.
19. Initial commands, environment, hashes, failures, limits, and judgments: pass.
20. Reciprocal reproduction, synthesis, and final status: pending both formal routes.

## Limitations and anomalies

- This is a smoke result, not the frozen 480-record conclusion.
- The first Claude execution attempt was unable to write or run local tools;
  it produced no files and is not treated as a scientific result.  A scoped
  writable retry was started independently.
- The evaluator deliberately reruns the unchanged legacy generator and adds
  the new certificate afterward; the formal analyzer will require all legacy
  leaves to match the frozen `FP-MART-001` baseline.
