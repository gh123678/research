# FP-TU-001 GPT post-formal defensive repair

After both formal seals and before Claude's reciprocal verification, GPT
self-review found that a simulated numerical inversion failure raised an
exception instead of returning the frozen ordered non-emission reason.  The
normal frozen grid cannot trigger this path in the sealed 480-record result,
so the formal data and scientific metrics are unchanged.

The repair adds typed inversion reasons for unbracketed, nonconverged,
nonconservative, and nonfinite cases; propagates them through state, pair, and
V-first routes; returns `not_certified` with all unavailable numeric fields
set to `null`; and preserves old radii only as clearly labelled audit values,
never as a fallback.  A monkeypatched invalid bracket now exercises the full
builder and confirms ordered non-emission on every route.

Validation after repair:

```text
C:\Users\Admin\anaconda3\python.exe -B verify_time_uniform_mixture_certificate.py
time-uniform mixture certificate checks passed

C:\Users\Admin\anaconda3\python.exe -m ruff check --no-cache time_uniform_mixture_certificate.py verify_time_uniform_mixture_certificate.py evaluate_time_uniform_certificates.py analyze_time_uniform_certificates.py
All checks passed!
```

Repaired source hashes:

- `time_uniform_mixture_certificate.py`:
  `ee8fb9e6b269e4d04d1c86a1c298a2f3a37e57eba9853eaf8dc6f2accaa13990`.
- `verify_time_uniform_mixture_certificate.py`:
  `18d527594d73b3f4ef0e1ca9a9c40374a4752527d4d9c3d59d0716bae6396a0a`.

No evaluator, analyzer, formal configuration, result artifact, formula,
weight, rate, threshold, solver default, or reported metric changed.  The
formal matrix was not rerun.
