"""Executable mathematical and interface checks for the literal witness."""

import json
import math
import sys

import numpy as np

import witness as w


class Checks:
    def __init__(self):
        self.count, self.failures, self.maxima = 0, [], {}

    def close(self, label, left, right, single=False):
        left, right = np.asarray(left), np.asarray(right)
        error = w.norm(left-right)
        tol = 1e-12 if single else 1e-10*(1+max(w.norm(left), w.norm(right)))
        self.count += 1
        self.maxima[label] = max(self.maxima.get(label, 0), error)
        if not np.isfinite(error) or error > tol:
            self.failures.append({"check": label, "error": error, "tolerance": tol})

    def truth(self, label, condition):
        self.count += 1
        if not condition:
            self.failures.append({"check": label})

    def bound(self, label, left, right):
        error = max(0., float(left-right))
        self.close(label, error, 0.)

    def rejects(self, label, call):
        try:
            call()
        except ValueError:
            self.truth(label, True)
        else:
            self.truth(label, False)


def main():
    check = Checks()
    data = w.run()
    check.truth("four exact sequences", len(data["exact"]) == 4)
    check.truth("eight finite sequences", len(data["finite"]) == 8)
    check.close("fixed policy", data["protocol"]["pi"], [[.75, .25], [.25, .75]], single=True)
    for label, rows in w.BATCHES.items():
        for exact in (True, False):
            for sharp in ((0,) if exact else (0, 8)):
                matrices = w.scalar_probabilities(rows, (sharp,)*3, exact=exact)
                g, b = w.affine(rows, matrices)
                net = w.Literal(rows, (sharp,)*3, exact=exact)
                vectors = [np.zeros(4), *np.eye(4), w.INITIALS["A"]]
                for q in vectors:
                    h = net.initialize(q)
                    hn, tape = net.step(h, tape=True)
                    actual = hn[:4, w.Q]
                    reference = w.direct(rows, q) if exact else w.scalar_finite(rows, q, (sharp,)*3)
                    check.close("one-step reference", actual, reference, single=True)
                    check.close("basis affine", actual, g@q+b, single=True)
                    immutable = [i for i in range(w.DIM) if i != w.Q]
                    check.close("scratch clear and immutable fields", hn[:, immutable], h[:, immutable], single=True)
                    check.close("nonmemory Q unchanged", hn[4:, w.Q], h[4:, w.Q], single=True)
                    check.close("null token unchanged", hn[-1], np.zeros(w.DIM), single=True)
                    expected_residual = rows[:, 2]+w.GAMMA*matrices[1]@q-matrices[0]@q
                    check.close("signed residual", tape["H_before_write"][4:4+len(rows), w.DELTA],
                                expected_residual, single=True)
                    for spec in net.heads:
                        stage = spec["stage"]
                        p = tape[stage]["probabilities"]
                        check.close("probability normalization", p.sum(axis=1), np.ones(net.length), single=True)
                        check.truth("finite scores", bool(np.isfinite(tape[stage]["scores_before_static_mask"]).all()))
                        check.truth("nonnegative probabilities", bool((p >= 0).all()))
                        check.close("masked probabilities zero", p[~spec["allowed"]], 0, single=True)
                        if not exact:
                            expected = np.zeros_like(spec["allowed"])
                            expected[:, -1] = True
                            if stage == "writer":
                                expected[:4, -1] = False
                                expected[:4, 4:4+len(rows)] = True
                            else:
                                expected[4:4+len(rows), -1] = False
                                expected[4:4+len(rows), :4] = True
                            check.truth("only positional finite masks", np.array_equal(expected, spec["allowed"]))
                    check.close("C matrix", tape["current"]["probabilities"][4:4+len(rows), :4], matrices[0], single=True)
                    check.close("S matrix", tape["successor"]["probabilities"][4:4+len(rows), :4], matrices[1], single=True)
                    check.close("W matrix", tape["writer"]["probabilities"][:4, 4:4+len(rows)], matrices[2], single=True)
                    if label == "M" and exact:
                        check.close("absent exact coordinate", actual[3], q[3], single=True)
                if not exact and label == "M":
                    check.close("absent finite writer is uniform", matrices[2][3], np.full(len(rows), 1/len(rows)), single=True)

    for key, route in data["exact"].items():
        label, init = key.split("/")
        check.truth("exact 65 steps", len(route["trace"]) == 65)
        check.close("exact norm", route["c"], .85 if label == "C" else 1, single=True)
        if label == "C":
            check.close("empirical fixed point", route["G"]@route["q_hat"]+route["b"], route["q_hat"])
        else:
            check.truth("missing empirical point unavailable", route["q_hat"] is None and bool(route["q_hat_unavailable_reason"]))
        for record in route["trace"]:
            check.close("exact repeated reference", record["q_direct"], record["q_exact"])
            if label == "C":
                check.bound("exact contraction bound", record["empirical_distance"], record["contraction_bound"])
            else:
                check.close("missing repeated invariant", record["q_exact"][3], w.INITIALS[init][3], single=True)

    for key, route in data["finite"].items():
        label, init, sharp = key.split("/")
        rows = w.BATCHES[label]
        exact = data["exact"][f"{label}/{init}"]
        check.truth("finite 65 steps", len(route["trace"]) == 65)
        check.close("conditional policy exact", route["conditional_policy_error"], 0, single=True)
        c = route["matrices"]["C"]
        s = route["matrices"]["S"]
        wm = route["matrices"]["W"]
        for record in route["trace"]:
            k, qf, qe = record["step"], record["q_finite"], record["q_exact"]
            check.close("finite repeated scalar", qf, record["q_scalar"])
            check.close("signed telescope", sum(record["stage_errors"].values()), record["same_input_update_gap"])
            check.bound("stage triangle bound", w.norm(record["same_input_update_gap"]), record["stage_triangle_bound"])
            check.bound("finite horizon bound", w.norm(qf-qe), record["error_bound"])
            check.close("same-input actual reference", record["same_input_update_gap"],
                        w.scalar_finite(rows, qe, (int(sharp),)*3)-w.direct(rows, qe))
            if k < w.STEPS:
                check.close("literal affine recurrence", route["trace"][k+1]["q_finite"], route["G"]@qf+route["b"])
                check.close("bound recurrence", route["trace"][k+1]["error_bound"],
                            route["c"]*record["error_bound"]+w.norm(record["same_input_update_gap"]))
                # One-step absolute tolerance at every trajectory point, separately
                # from the accumulated relative tolerance of the traces.
                hn, _ = w.Literal(rows, (int(sharp),)*3).step(w.Literal(rows, (int(sharp),)*3).initialize(qf))
                check.close("trajectory one-step scalar", hn[:4, w.Q], w.scalar_finite(rows, qf, (int(sharp),)*3), single=True)
            if record["population_error_decomposition"] is not None:
                dec = record["population_error_decomposition"]
                check.close("population signed decomposition", dec["finite_iteration"]+dec["attention_bias"]+dec["batch_discrepancy"], dec["total"])
                check.bound("empirical transient bound", record["empirical_distance"], record["empirical_distance_bound"])
            check.close("residual reconstruction", record["residual"], rows[:, 2]+w.GAMMA*s@qf-c@qf)
        if route["c"] < 1:
            check.close("finite fixed point solve", route["G"]@route["q_f_inf"]+route["b"], route["q_f_inf"])
        else:
            check.truth("noncontractive no uniqueness claim", route["q_f_inf"] is None and bool(route["q_f_inf_unavailable_reason"]))
        if label == "C" and route["c"] < 1:
            check.bound("steady attention bound", route["attention_bias"], route["attention_bias_bound"])
            check.close("batch component", exact["batch_discrepancy"], exact["q_hat"]-data["q_pi_audit_only"])
        if int(sharp) == 0:
            check.close("uniform writer", wm, np.full_like(wm, 1/len(rows)), single=True)
            check.close("zero sharpness invariant differences", route["trace"][-1]["q_finite"]-route["trace"][-1]["q_finite"][0],
                        w.INITIALS[init]-w.INITIALS[init][0])

    pop_t = np.array([[w.AUDIT_P[x, u]*w.PI[u, a] for u in range(2) for a in range(2)] for x in range(4)])
    check.close("population Bellman solve", w.AUDIT_R+w.GAMMA*pop_t@data["q_pi_audit_only"], data["q_pi_audit_only"])
    for bad in (np.zeros((2, 2)), np.array([[.8, .8], [.2, .8]]), np.full((2, 2), np.nan), np.ones((2, 3))):
        check.rejects("malformed pi rejected", lambda bad=bad: w.Literal(w.BATCH_C, pi=bad))
    check.rejects("duplicate identity rejected", lambda: w.Literal(w.BATCH_C, memory_ids=(0, 0, 2, 3)))
    check.rejects("nonfinite Q rejected", lambda: w.Literal(w.BATCH_C).initialize(np.full(4, math.inf)))
    check.rejects("nonfinite rows rejected", lambda: w.Literal(np.full((6, 4), np.nan)))
    # Strict serialization must reject every nonfinite numeric value.
    json.loads(json.dumps(data, default=w.serializable, allow_nan=False))
    check.truth("strict JSON roundtrip", True)
    output = {"status": "PASS" if not check.failures else "FAIL", "checks": check.count,
              "failures": check.failures, "max_errors": check.maxima, "environment": w.environment()}
    print(json.dumps(output, allow_nan=False))
    return bool(check.failures)


if __name__ == "__main__":
    sys.exit(main())
