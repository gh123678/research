"""FP-ITER-001 v1.1: explicit fixed projections, independent scalar audit."""

import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

BASELINE = "6d2376968aeb9379c978bd1a7af7929b70fdeb09"
PI = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=np.float64)
GAMMA, ALPHA, STEPS = 0.7, 0.5, 64
BATCH_C = np.array([
    [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, -0.5, 1],
    [1, 0, 0.25, 0], [1, 0, 0.25, 1], [1, 1, 0.75, 0],
], dtype=np.float64)
BATCHES = {"C": BATCH_C, "M": BATCH_C[:5].copy()}
INITIALS = {"Z": np.zeros(4), "A": np.array([1., -1., 0.5, 0.5])}
AUDIT_P = np.array([[.75, .25], [.25, .75], [.5, .5], [.75, .25]])
AUDIT_R = np.array([1., -.5, .25, .75])
# Row-token representation. All fields except the last three are immutable
# during a network step, apart from Q which is changed only by writer attention.
DIM = 15
LOGPI, ONE, REWARD, Q, CURRENT, SUCCESSOR, DELTA = range(8, 15)


def norm(x):
    return float(np.max(np.abs(x)))


def validate(rows, q, pi=PI, memory_ids=(0, 1, 2, 3)):
    rows, q, pi = map(lambda a: np.asarray(a, dtype=np.float64), (rows, q, pi))
    if rows.ndim != 2 or rows.shape[1] != 4 or len(rows) == 0 or q.shape != (4,):
        raise ValueError("nonempty N-by-4 rows and four Q coordinates required")
    if pi.shape != (2, 2) or not np.all(np.isfinite(pi)) or np.any(pi <= 0):
        raise ValueError("finite positive 2-by-2 policy required")
    if not np.allclose(pi.sum(axis=1), 1, atol=1e-14, rtol=0):
        raise ValueError("policy rows must sum to one")
    if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(q)):
        raise ValueError("nonfinite input")
    if np.any(~np.isin(rows[:, [0, 1, 3]], [0, 1])):
        raise ValueError("state and action identities must be binary integers")
    if len(memory_ids) != 4 or len(set(memory_ids)) != 4:
        raise ValueError("duplicate or missing canonical memory identity")
    if tuple(memory_ids) != (0, 1, 2, 3):
        raise ValueError("canonical memory positions must be 00,01,10,11")
    return rows, q, pi


def direct(rows, q, pi=PI):
    """Scalar synchronous batch Expected SARSA; no attention/projection calls."""
    rows, q, pi = validate(rows, q, pi)
    out = q.copy()
    for s in range(2):
        for a in range(2):
            residuals = []
            for st, act, reward, nxt in rows:
                if int(st) == s and int(act) == a:
                    expectation = sum(float(pi[int(nxt), b]) * float(q[2*int(nxt)+b])
                                      for b in range(2))
                    residuals.append(float(reward) + GAMMA*expectation - float(q[2*s+a]))
            if residuals:
                out[2*s+a] += ALPHA*sum(residuals)/len(residuals)
    return out


def scalar_probabilities(rows, sharp, pi=PI, exact=False):
    """Loop-built scalar score reference; independent of H and W projections."""
    rows, _, pi = validate(rows, np.zeros(4), pi)
    xi, zeta, tau = sharp
    n = len(rows)
    c, s, w = np.zeros((n, 4)), np.zeros((n, 4)), np.zeros((4, n))
    for t, (st, act, _, nxt) in enumerate(rows):
        pair = 2*int(st)+int(act)
        cweights = [float(y == pair) if exact else math.exp(xi*int(y == pair))
                    for y in range(4)]
        sweights = [float(pi[u, b]) * (float(u == int(nxt)) if exact
                    else math.exp(zeta*int(u == int(nxt))))
                    for u in range(2) for b in range(2)]
        for y in range(4):
            c[t, y] = cweights[y]/sum(cweights)
            s[t, y] = sweights[y]/sum(sweights)
    for pair in range(4):
        weights = [float(2*int(st)+int(act) == pair) if exact else
                   math.exp(tau*int(2*int(st)+int(act) == pair))
                   for st, act, _, _ in rows]
        if sum(weights):
            for t in range(n):
                w[pair, t] = weights[t]/sum(weights)
    return c, s, w


def scalar_finite(rows, q, sharp, pi=PI):
    """Evaluates scalar sums without using the literal attention implementation."""
    rows, q, pi = validate(rows, q, pi)
    c, s, w = scalar_probabilities(rows, sharp, pi)
    d = [float(rows[t, 2]) + GAMMA*sum(s[t, y]*q[y] for y in range(4))
         - sum(c[t, y]*q[y] for y in range(4)) for t in range(len(rows))]
    return np.array([q[x] + ALPHA*sum(w[x, t]*d[t] for t in range(len(rows)))
                     for x in range(4)])


class Literal:
    """Three standard attention heads plus signed ReLU and scratch projection."""

    def __init__(self, rows, sharp=(8., 8., 8.), pi=PI, exact=False,
                 memory_ids=(0, 1, 2, 3)):
        self.rows, _, self.pi = validate(rows, np.zeros(4), pi, memory_ids)
        self.rows, self.pi = self.rows.copy(), self.pi.copy()
        self.sharp = tuple(map(float, sharp))
        if len(self.sharp) != 3 or not np.all(np.isfinite(self.sharp)):
            raise ValueError("three finite sharpness values required")
        self.n, self.exact = len(rows), exact
        self.length = 4+self.n+1
        self.template = np.zeros((self.length, DIM))
        for x in range(4):
            self.template[x, x] = 1
            self.template[x, 4+x//2] = 1
            self.template[x, LOGPI] = math.log(self.pi[x//2, x % 2])
            self.template[x, ONE] = 1
        for t, (st, act, reward, nxt) in enumerate(self.rows):
            row = self.template[4+t]
            row[2*int(st)+int(act)] = 1
            row[6+int(nxt)] = 1
            row[ONE], row[REWARD] = 1, reward
        self.heads = self.make_heads()
        self.ff1 = np.zeros((DIM, 2))
        self.ff1[REWARD, 0], self.ff1[CURRENT, 0] = 1, -1
        self.ff1[SUCCESSOR, 0] = GAMMA
        self.ff1[:, 1] = -self.ff1[:, 0]
        self.ff2 = np.zeros((2, DIM))
        self.ff2[0, DELTA], self.ff2[1, DELTA] = 1, -1
        self.clear = np.eye(DIM)
        self.clear[[CURRENT, SUCCESSOR, DELTA], [CURRENT, SUCCESSOR, DELTA]] = 0

    def make_heads(self):
        result = []
        for stage, sharp, dk, field in zip(
            ["current", "successor", "writer"], self.sharp, [4, 3, 4],
            [CURRENT, SUCCESSOR, Q], strict=True,
        ):
            wq, wk = np.zeros((DIM, dk)), np.zeros((DIM, dk))
            wv, wo = np.zeros((DIM, 1)), np.zeros((1, DIM))
            if stage == "successor":
                wq[6:8, :2] = (0 if self.exact else sharp)*math.sqrt(dk)*np.eye(2)
                wk[4:6, :2] = np.eye(2)
                wq[ONE, 2], wk[LOGPI, 2] = math.sqrt(dk), 1
            else:
                wq[:4] = (0 if self.exact else sharp)*math.sqrt(dk)*np.eye(4)
                wk[:4] = np.eye(4)
            wv[DELTA if stage == "writer" else Q, 0] = 1
            wo[0, field] = ALPHA if stage == "writer" else 1
            allowed = np.zeros((self.length, self.length), dtype=bool)
            allowed[:, -1] = True
            queries = range(4) if stage == "writer" else range(4, 4+self.n)
            keys = list(range(4, 4+self.n)) if stage == "writer" else list(range(4))
            for query in queries:
                eligible = keys
                if self.exact:
                    if stage == "successor":
                        nxt = int(self.rows[query-4, 3])
                        eligible = [key for key in keys if key//2 == nxt]
                    elif stage == "current":
                        pair = 2*int(self.rows[query-4, 0])+int(self.rows[query-4, 1])
                        eligible = [key for key in keys if key == pair]
                    else:
                        eligible = [key for key in keys if
                                    2*int(self.rows[key-4, 0])+int(self.rows[key-4, 1]) == query]
                if eligible:
                    allowed[query, -1] = False
                    allowed[query, eligible] = True
            result.append({"stage": stage, "WQ": wq, "WK": wk, "WV": wv,
                           "WO": wo, "allowed": allowed})
        return result

    def initialize(self, q):
        _, q, _ = validate(self.rows, q, self.pi)
        h = self.template.copy()
        h[:4, Q] = q
        return h

    def head(self, h, spec):
        queries, keys, values = h@spec["WQ"], h@spec["WK"], h@spec["WV"]
        scores = queries@keys.T/math.sqrt(queries.shape[1])
        masked = np.where(spec["allowed"], scores, -np.inf)
        unnormalized = np.exp(masked-masked.max(axis=1, keepdims=True))
        probabilities = unnormalized/unnormalized.sum(axis=1, keepdims=True)
        output = probabilities@values@spec["WO"]
        return h+output, {"queries": queries, "keys": keys, "values": values,
                          "scores_before_static_mask": scores,
                          "probabilities": probabilities, "output": output}

    def step(self, h, tape=False):
        if h.shape != (self.length, DIM) or not np.all(np.isfinite(h)):
            raise ValueError("invalid prompt matrix")
        record = {"H_input": h.copy()} if tape else {}
        for spec in self.heads[:2]:
            h, evidence = self.head(h, spec)
            if tape:
                record[spec["stage"]] = evidence
        hidden = np.maximum(h@self.ff1, 0)
        h = h+hidden@self.ff2
        if tape:
            record["relu_hidden"], record["H_before_write"] = hidden, h.copy()
        h, evidence = self.head(h, self.heads[2])
        if tape:
            record["writer"] = evidence
        h = h@self.clear
        if tape:
            record["H_output"] = h.copy()
        return h, record

    def weights(self):
        return {"heads": self.heads, "FF1": self.ff1, "FF2": self.ff2,
                "scratch_clear": self.clear, "template": self.template,
                "layout": {"pair": [0, 1, 2, 3], "state": [4, 5], "next_state": [6, 7],
                           "log_pi": LOGPI, "one": ONE, "reward": REWARD, "Q": Q,
                           "current": CURRENT, "successor": SUCCESSOR, "delta": DELTA}}


def affine(rows, matrices):
    c, s, w = matrices
    return np.eye(4)+ALPHA*w@(GAMMA*s-c), ALPHA*w@rows[:, 2]


def error_terms(rows, q, matrices, exact_matrices):
    c, s, w = matrices
    c0, s0, w0 = exact_matrices
    terms = {
        "current": -ALPHA*w@(c-c0)@q,
        "successor": ALPHA*GAMMA*w@(s-s0)@q,
        "write": ALPHA*(w-w0)@(rows[:, 2]+GAMMA*s0@q-c0@q),
    }
    return terms


def environment():
    here = Path(__file__).resolve()
    project = here.parents[4]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    paths = [here, here.with_name("verify.py")]
    return {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
            "platform": platform.platform(), "dtype": "float64", "device": "CPU",
            "git_commit": commit, "execution_baseline": BASELINE,
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths if p.exists()}}


def run():
    population_transition = np.array([
        [AUDIT_P[x, u]*PI[u, b] for u in range(2) for b in range(2)]
        for x in range(4)])
    q_pi = np.linalg.solve(np.eye(4)-GAMMA*population_transition, AUDIT_R)
    result = {"task": "FP-ITER-001", "version": "1.1", "actor": "codex",
              "environment": environment(), "protocol": {"pi": PI, "gamma": GAMMA,
              "alpha": ALPHA, "steps": STEPS, "batches": BATCHES, "initials": INITIALS,
              "sharpness": [0, 8], "audit_P": AUDIT_P, "audit_r": AUDIT_R},
              "q_pi_audit_only": q_pi, "exact": {}, "finite": {}}
    for label, rows in BATCHES.items():
        exact_matrices = scalar_probabilities(rows, (0, 0, 0), exact=True)
        g0, b0 = affine(rows, exact_matrices)
        q_hat = np.linalg.solve(np.eye(4)-g0, b0) if label == "C" else None
        for initial_label, initial in INITIALS.items():
            exact_net = Literal(rows, exact=True)
            h0 = exact_net.initialize(initial)
            q_direct, q_exact = initial.copy(), initial.copy()
            exact_records, exact_tape = [], None
            for k in range(STEPS+1):
                exact_records.append({"step": k, "q_direct": q_direct.copy(),
                                      "q_exact": q_exact.copy(),
                                      "reference_gap": norm(q_direct-q_exact),
                                      "population_distance": norm(q_exact-q_pi),
                                      "empirical_distance": None if q_hat is None else norm(q_exact-q_hat),
                                      "contraction_bound": None if q_hat is None else
                                      .85**k*norm(initial-q_hat)})
                if k < STEPS:
                    q_direct = direct(rows, q_direct)
                    h0, t = exact_net.step(h0, tape=(k == 0))
                    if k == 0:
                        exact_tape = t
                    q_exact = h0[:4, Q].copy()
            result["exact"][f"{label}/{initial_label}"] = {
                "G": g0, "b": b0, "c": float(np.linalg.norm(g0, ord=np.inf)),
                "matrices": dict(zip(["C0", "S0", "W0"], exact_matrices, strict=True)),
                "q_hat": q_hat, "q_hat_unavailable_reason": None if q_hat is not None
                else "Missing pair 11 is preserved; no unique full-table empirical fixed point.",
                "batch_discrepancy": None if q_hat is None else q_hat-q_pi,
                "weights": exact_net.weights(), "first_step_tape": exact_tape,
                "trace": exact_records}
            for sharp in (0, 8):
                triple = (sharp, sharp, sharp)
                matrices = scalar_probabilities(rows, triple)
                c, s, w = matrices
                g, b = affine(rows, matrices)
                cf, rho = float(np.linalg.norm(g, ord=np.inf)), float(max(abs(np.linalg.eigvals(g))))
                q_inf = np.linalg.solve(np.eye(4)-g, b) if cf < 1 else None
                bias = None if q_hat is None or q_inf is None else norm(q_inf-q_hat)
                bias_bound = None if bias is None else norm(g@q_hat+b-q_hat)/(1-cf)
                net = Literal(rows, triple)
                hf = net.initialize(initial)
                q_scalar, error_bound, trace, finite_tape = initial.copy(), 0., [], None
                for k in range(STEPS+1):
                    qf = hf[:4, Q].copy()
                    qe = exact_records[k]["q_exact"]
                    terms = error_terms(rows, qe, matrices, exact_matrices)
                    same_gap = g@qe+b-(g0@qe+b0)
                    decomposition = None
                    if bias is not None:
                        decomposition = {"finite_iteration": qf-q_inf,
                                         "attention_bias": q_inf-q_hat,
                                         "batch_discrepancy": q_hat-q_pi,
                                         "total": qf-q_pi}
                    trace.append({"step": k, "q_finite": qf, "q_scalar": q_scalar.copy(),
                                  "scalar_gap": norm(qf-q_scalar), "q_exact": qe,
                                  "finite_exact_gap": norm(qf-qe), "error_bound": error_bound,
                                  "bound_slack": error_bound-norm(qf-qe),
                                  "same_input_update_gap": same_gap,
                                  "stage_errors": terms,
                                  "stage_triangle_bound": sum(norm(v) for v in terms.values()),
                                  "current_read": c@qf, "successor_expectation": s@qf,
                                  "residual": rows[:, 2]+GAMMA*s@qf-c@qf,
                                  "population_distance": norm(qf-q_pi),
                                  "empirical_distance": None if q_hat is None else norm(qf-q_hat),
                                  "empirical_distance_bound": None if bias is None else
                                  cf**k*norm(initial-q_inf)+bias,
                                  "population_error_decomposition": decomposition})
                    if k < STEPS:
                        error_bound = cf*error_bound+norm(same_gap)
                        q_scalar = scalar_finite(rows, q_scalar, triple)
                        hf, t = net.step(hf, tape=(k == 0))
                        if k == 0:
                            finite_tape = t
                policy_errors = [abs(s[t, 2*u+a]/sum(s[t, 2*u:2*u+2])-PI[u, a])
                                 for t in range(len(rows)) for u in range(2) for a in range(2)]
                result["finite"][f"{label}/{initial_label}/{sharp}"] = {
                    "sharpness": triple, "G": g, "b": b, "c": cf, "spectral_radius": rho,
                    "eigenvalues": [[float(v.real), float(v.imag)] for v in np.linalg.eigvals(g)],
                    "matrices": dict(zip(["C", "S", "W"], matrices, strict=True)),
                    "conditional_policy_error": max(policy_errors),
                    "q_f_inf": q_inf, "q_f_inf_unavailable_reason": None if q_inf is not None
                    else "Infinity-norm contraction is not established; no unique-limit claim.",
                    "attention_bias": bias, "attention_bias_bound": bias_bound,
                    "weights": net.weights(), "first_step_tape": finite_tape, "trace": trace}
    return result


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    print(json.dumps(run(), default=serializable, allow_nan=False, ensure_ascii=False))
