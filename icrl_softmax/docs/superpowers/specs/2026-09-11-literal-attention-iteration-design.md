# Design: three certified iteration steps inside the literal attention network

Date: 2026-09-11.
Task: `docs/research_tasks/FP-ATTN-ITER-001.md`.
Status: `ACTIVE`.

## 1. The gap

Three prior results established the pieces separately:

| result | what it verified | which code path |
|---|---|---|
| `FP-ATTN-001` | the literal network reproduces **one** certified step | literal network |
| `FP-ITER2-001` | a **second** certified step exists, `20/22` | numpy |
| `FP-ITER3-001` | a **third** certified step exists, `15/48` | numpy |

The project's claim is about the **network**, so the second and third steps were
established in the wrong code path as far as that claim is concerned. The
unexecuted inference is:

```text
network -> one certified step       (established)
network -> three certified steps    (never executed)
```

## 2. Design decision

Make the network produce `Qhat` at **every** iteration step, and let the frozen
certificate and decision rule operate on that network output unchanged. Nothing
else moves: same protocol, same batches, same certificate, same decision rule,
same route→network mapping, same tolerance.

```text
pi_0 = frozen target policy
for k = 1, 2, 3:
    Q_k    = LITERAL NETWORK(training batch, pi_{k-1})
    cert_k = variance_adaptive certificate(Q_k, pi_{k-1}, cert batch)
    decide = relative-softmax rule(pi_{k-1}, Q_k, cert_k)
    if abstain: stop and record the frozen reason
    pi_k   = decide.policy_plus
```

The numpy route is computed **additionally** at each step, purely as the
comparison baseline. It never feeds the iteration.

## 3. The risk this task is really about

`FP-ATTN-001` measured a one-step `Qhat` gap of `1.076e-05` between `float32`
network arithmetic and `float64` numpy arithmetic. Iteration feeds each step's
output into the next, so that difference could **compound**. Third-step
emission margins are of order `0.378`, and some records are much tighter.

Two outcomes, both valuable:

- **agreement**: the network carries the iteration, and the project's claim is
  established in the code path it is actually about;
- **flips**: `float32` network arithmetic changes certified decisions under
  iteration. That would be a substantive finding, and it would also bound how
  far the network path can be trusted.

The design therefore does not assume agreement. It serialises, for every step,
the literal status, the numpy status, both `E_Q` values, both smallest per-state
lower bounds and the `Qhat` gap, so a flip surfaces with its boundary margin
rather than being absorbed into an average.

## 4. Why an inert-looking step is still worth executing

It would be reasonable to argue that `FP-ATTN-001` plus `FP-ITER3-001` already
imply the result. They do not, for two reasons:

1. the composition is over a **nonlinear** iteration with a decision boundary,
   so agreement at step 1 does not imply agreement at step 3;
2. the certificate is a threshold rule — a small `E_Q` change can flip an
   emission whenever a record sits near `min_s LB_s = 0`.

Both are empirical questions, and this task answers them.

## 5. What success and failure mean

- Success (`H4`): `90/90` decisions agree, `15` three-step routes, zero
  violations. The network carries the iteration identically to numpy.
- Failure of `H4`: disagreements are listed individually with their margins, and
  the task reports that `float32` network arithmetic changes certified decisions
  under iteration. The numpy iteration results remain valid; the network path
  acquires a measured limit.

## 6. Risks and mitigations

- **Accumulation.** The main risk, handled by explicit per-step gap
  serialisation and by `H5`'s enumeration requirement.
- **Provenance drift.** A numpy call could silently replace the network. Handled
  by passing the network tensor straight into the certificate and by a
  `qhat_producer` field the verifier checks.
- **Cost.** `160` layers x `3` steps x `48` route-records, measured at about one
  second per network run, plus the sampling cost; comfortably in budget.
- **Within-author cross-check.** The numpy baseline is by the same author, so
  agreement rules out implementation drift but not a shared conceptual error.
  Disclosed; not mitigated.

## 7. Out of scope

- Any change to a closed task's frozen parameters, results, or reports.
- Extending past three steps.
- Repeated control, online control, or conditional-on-emission claims.
- Any claim of independent or reciprocal verification.
