"""FP-EXPL-001 Claude route verifier: independent re-derivation and checks.

Self-contained independent re-implementation of the frozen FP-EXPL-001 v1.1
protocol (no import of witness.py or any other actor's code). Regenerates the
frozen exploration batch from numpy PCG64(20260911) with exactly two scalar
draws per transition, recomputes the direct grouped-mean reference, the exact
grouped-attention operator, the finite operator as an independently written
scalar formula, and a separately coded pure-Python literal fixed-weight
softmax attention network; then checks coverage, sampling reconstruction,
policy preservation, both equivalences, affine reconstruction from the four
standard basis vectors, scratch clearing, immutable-field preservation, stage
errors, all finite-step bounds, and (when applicable) fixed-point residuals
and the three-term decomposition. Writes strict JSON check counts, failures
and maximum deviations to stdout and to
results/FP-EXPL-001/claude/verification.json. Exit code is 0 when the
verdict is PASS or COVERAGE_FAILURE (a protocol stop event, not a verifier
failure) and 1 when the verdict is FAIL; an unexpected exception exits
nonzero via the interpreter.
Run from icrl_softmax:
python -B docs/research_branches/FP-EXPL-001/claude/verify.py
"""

import json
import math
import os
import platform
import sys

import numpy as np

# Frozen inputs, retyped from docs/research_tasks/FP-EXPL-001.md v1.1.
GAMMA = 0.7
ALPHA = 0.5
STEPS = 64
PI = ((0.75, 0.25), (0.25, 0.75))  # target policy
PI_B = ((0.5, 0.5), (0.5, 0.5))  # behavior policy (sampler only)
REW = (1.0, -0.5, 0.25, 0.25)  # by pair 00,01,10,11
P_NEXT = ((0.75, 0.25), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25))  # sampler/audit only
PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))
INIT_STATE = 0
BATCH = 64
SEED = 20260911
Q0 = (0.0, 0.0, 0.0, 0.0)
XI = ZETA = TAU = 8.0
TOL1 = 1e-12
TOLR = 1e-10
C0 = 1.0 - ALPHA * (1.0 - GAMMA)  # 0.85

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, os.pardir, os.pardir, os.pardir,
    )
)
RESULTS_DIR = os.path.join(ROOT, "results", "FP-EXPL-001", "claude")


class Reject(ValueError):
    pass


def pidx(s, a):
    if s not in (0, 1) or a not in (0, 1):
        raise Reject("bad state/action")
    return 2 * s + a


def check_pi(pi):
    if len(pi) != 2 or any(len(row) != 2 for row in pi):
        raise Reject("pi shape")
    for row in pi:
        for v in row:
            if not math.isfinite(v):
                raise Reject("pi nonfinite")
            if v <= 0.0:
                raise Reject("pi must have full support")
        if abs(sum(row) - 1.0) > TOL1:
            raise Reject("pi row not normalized")


def check_q(q):
    if len(q) != 4 or any(not math.isfinite(v) for v in q):
        raise Reject("bad q")


def check_rows(rows):
    if not rows:
        raise Reject("empty batch")
    for s, a, r, u in rows:
        pidx(s, a)
        if u not in (0, 1):
            raise Reject("bad next state")
        if not math.isfinite(r):
            raise Reject("nonfinite reward")


# --- frozen sampler (independently coded; exactly two draws per transition) --


def sample_batch():
    rng = np.random.Generator(np.random.PCG64(SEED))
    s = INIT_STATE
    out = []
    draws = 0
    for _ in range(BATCH):
        ua = float(rng.random())
        draws += 1
        a = 0 if ua < PI_B[s][0] else 1
        ut = float(rng.random())
        draws += 1
        x = pidx(s, a)
        u = 0 if ut < P_NEXT[x][0] else 1
        out.append((s, a, REW[x], u, ua, ut))
        s = u
    return out, draws


def softmax(logits):
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    z = sum(e)
    return [v / z for v in e]


# --- direct reference (grouped residuals, no attention) ---------------------


def direct_step(q, rows):
    check_q(q)
    check_rows(rows)
    out = list(q)
    for x in range(4):
        ds = []
        for s, a, r, u in rows:
            if pidx(s, a) == x:
                vsucc = sum(PI[u][b] * q[pidx(u, b)] for b in (0, 1))
                ds.append(r + GAMMA * vsucc - q[x])
        if ds:
            out[x] = q[x] + ALPHA * sum(ds) / len(ds)
    return out


# --- exact grouped matrices (declared equality structure) -------------------


def exact_mats(rows):
    n = len(rows)
    c0 = [[0.0] * 4 for _ in range(n)]
    s0 = [[0.0] * 4 for _ in range(n)]
    w0 = [[0.0] * n for _ in range(4)]
    r = [row[2] for row in rows]
    for t, (s, a, _, u) in enumerate(rows):
        c0[t][pidx(s, a)] = 1.0
        for b in (0, 1):
            s0[t][pidx(u, b)] = PI[u][b]
    for x in range(4):
        match = [t for t, (s, a, _, _) in enumerate(rows) if pidx(s, a) == x]
        if not match:
            raise Reject(f"pair {PAIRS[x]} absent from covered batch")
        for t in match:
            w0[x][t] = 1.0 / len(match)
    return c0, s0, w0, r


# --- finite operator, independent scalar formula ----------------------------


def finite_weights(rows, xi, zeta, tau):
    n = len(rows)
    x_ids = [pidx(s, a) for s, a, _, _ in rows]
    u_ids = [u for _, _, _, u in rows]
    logpi = [math.log(PI[s][a]) for s, a in PAIRS]
    c = [softmax([xi if x_ids[t] == y else 0.0 for y in range(4)])
         for t in range(n)]
    s = [softmax([(zeta if u_ids[t] == PAIRS[y][0] else 0.0) + logpi[y]
                  for y in range(4)])
         for t in range(n)]
    w = [softmax([tau if x == x_ids[t] else 0.0 for t in range(n)])
         for x in range(4)]
    r = [row[2] for row in rows]
    return c, s, w, r


def matvec(m, v):
    return [sum(row[j] * v[j] for j in range(len(v))) for row in m]


def finite_scalar_step(q, rows, xi, zeta, tau):
    check_q(q)
    c, s, w, r = finite_weights(rows, xi, zeta, tau)
    sq = matvec(s, q)
    cq = matvec(c, q)
    n = len(rows)
    out = list(q)
    for x in range(4):
        acc = sum(w[x][t] * (r[t] + GAMMA * sq[t] - cq[t]) for t in range(n))
        out[x] = q[x] + ALPHA * acc
    return out


# --- separately coded literal attention network (pure python) ---------------
# Fields per token, same derived architecture, independently coded:
#   const, role in {mem,ctx,null}, pair one-hot(4), state one-hot(2),
#   action one-hot(2), reward, q, logpi, scratch cur/succ/res/wd.


def make_tokens(rows, q0):
    tokens = []
    for x, (s, a) in enumerate(PAIRS):
        tokens.append({
            "role": "mem", "pair": [1.0 if j == x else 0.0 for j in range(4)],
            "state": [1.0, 0.0] if s == 0 else [0.0, 1.0],
            "action": [1.0, 0.0] if a == 0 else [0.0, 1.0],
            "reward": 0.0, "q": float(q0[x]), "logpi": math.log(PI[s][a]),
            "cur": 0.0, "succ": 0.0, "res": 0.0, "wd": 0.0,
        })
    for s, a, r, u in rows:
        tokens.append({
            "role": "ctx", "pair": [1.0 if j == pidx(s, a) else 0.0
                                    for j in range(4)],
            "state": [1.0, 0.0] if u == 0 else [0.0, 1.0],
            "action": [1.0, 0.0] if a == 0 else [0.0, 1.0],
            "reward": r, "q": 0.0, "logpi": 0.0,
            "cur": 0.0, "succ": 0.0, "res": 0.0, "wd": 0.0,
        })
    tokens.append({
        "role": "null", "pair": [0.0] * 4, "state": [0.0, 0.0],
        "action": [0.0, 0.0], "reward": 0.0, "q": 0.0, "logpi": 0.0,
        "cur": 0.0, "succ": 0.0, "res": 0.0, "wd": 0.0,
    })
    return tokens


IMMUTABLE_KEYS = ("role", "pair", "state", "action", "reward", "logpi")
SCRATCH_KEYS = ("cur", "succ", "res", "wd")


def literal_net_step(tokens, rows, xi, zeta, tau, mode, capture=False):
    """One synchronous update through explicit fixed projection weights.

    mode 'finite': normalized softmax over dot-product scores of declared
    one-hot identity features (plus frozen log pi bias in stage 2).
    mode 'exact': declared equality probabilities (exact route only).
    Static role masks only: ctx queries read mem keys; mem queries read ctx
    keys; inactive queries read the null token (whose value is zero).
    """
    n = len(rows)
    mem_idx = list(range(4))
    ctx_idx = [4 + t for t in range(n)]
    null_idx = 4 + n

    def attend(active, keys, score_fn, value_fn, exact_probs=None):
        outs = {}
        probs_record = {}
        for qi in range(len(tokens)):
            if qi in active:
                ks = keys
                if mode == "exact":
                    probs = exact_probs(qi)
                else:
                    probs = softmax([score_fn(qi, kj) for kj in ks])
            else:
                ks = [null_idx]
                probs = [1.0]
            outs[qi] = sum(p * value_fn(kj) for p, kj in zip(probs, ks))
            probs_record[qi] = (list(ks), list(probs))
        return outs, probs_record

    # stage 1: current-pair read; score xi * onehot(x_t).onehot(pair_y)
    s1, _p1 = attend(
        ctx_idx, mem_idx,
        lambda qi, kj: xi * sum(
            tokens[qi]["pair"][j] * tokens[kj]["pair"][j] for j in range(4)),
        lambda kj: tokens[kj]["q"],
        exact_probs=lambda qi: [
            1.0 if kj == pidx(rows[qi - 4][0], rows[qi - 4][1]) else 0.0
            for kj in mem_idx],
    )
    for t in ctx_idx:
        tokens[t]["cur"] += s1[t]  # residual connection
    # stage 2: successor read; score zeta * onehot(u_t).onehot(state_y) + logpi
    s2, _p2 = attend(
        ctx_idx, mem_idx,
        lambda qi, kj: zeta * sum(
            tokens[qi]["state"][j] * tokens[kj]["state"][j] for j in range(2))
        + tokens[kj]["logpi"],
        lambda kj: tokens[kj]["q"],
        exact_probs=lambda qi: [
            PI[PAIRS[kj][0]][PAIRS[kj][1]]
            if PAIRS[kj][0] == rows[qi - 4][3] else 0.0
            for kj in mem_idx],
    )
    for t in ctx_idx:
        tokens[t]["succ"] += s2[t]
    # stage 3: fixed linear feed-forward residual map on every token
    for tok in tokens:
        tok["res"] = tok["res"] + tok["reward"] + GAMMA * tok["succ"] - tok["cur"]
    # stage 4: pair writeback; score tau * onehot(pair_x).onehot(x_t)
    x_ids = [pidx(s, a) for s, a, _, _ in rows]
    counts = [sum(1 for t in range(n) if x_ids[t] == x) for x in range(4)]

    def exact_write_probs(qi):
        if counts[qi] == 0:
            return None  # absent-row null mask: handled by caller
        return [
            (1.0 / counts[qi]) if x_ids[kj - 4] == qi else 0.0
            for kj in ctx_idx
        ]

    s4, p4 = attend(
        [x for x in mem_idx if mode == "finite" or counts[x] > 0],
        ctx_idx,
        lambda qi, kj: tau * sum(
            tokens[qi]["pair"][j] * tokens[kj]["pair"][j] for j in range(4)),
        lambda kj: tokens[kj]["res"],
        exact_probs=exact_write_probs,
    )
    for x in mem_idx:
        tokens[x]["wd"] += s4.get(x, 0.0)
    # stage 5: fixed linear maps: q += alpha * wd; reset scratch fields
    captured = None
    if capture:
        # snapshot before the reset: after reset the scratch fields are zero
        captured = {
            "res_before_reset": [tok["res"] for tok in tokens],
            "wd_before_reset": [tok["wd"] for tok in tokens],
            "stage4_probs_keys": {str(k): v for k, v in p4.items()},
        }
    for tok in tokens:
        tok["q"] = tok["q"] + ALPHA * tok["wd"]
        tok["cur"] = tok["succ"] = tok["res"] = tok["wd"] = 0.0
    out_q = [tokens[x]["q"] for x in mem_idx]
    if capture:
        captured["post_reset_scratch_zero"] = all(
            tok["cur"] == 0.0 and tok["succ"] == 0.0
            and tok["res"] == 0.0 and tok["wd"] == 0.0 for tok in tokens)
        return out_q, captured
    return out_q, None


# --- affine maps ------------------------------------------------------------


def affine(rows, xi=None, zeta=None, tau=None):
    if xi is None:
        c, s, w, r = exact_mats(rows)
    else:
        c, s, w, r = finite_weights(rows, xi, zeta, tau)
    n = len(rows)
    g = [[0.0] * 4 for _ in range(4)]
    for x in range(4):
        for y in range(4):
            acc = sum(w[x][t] * (GAMMA * s[t][y] - c[t][y]) for t in range(n))
            g[x][y] = (1.0 if x == y else 0.0) + ALPHA * acc
    b = [ALPHA * sum(w[x][t] * r[t] for t in range(n)) for x in range(4)]
    return g, b


def norm_inf_mat(g):
    return max(sum(abs(v) for v in row) for row in g)


def norm_inf_vec(v):
    return max(abs(x) for x in v)


def sub(u, v):
    return [a - b for a, b in zip(u, v)]


def add3(a, b, c):
    return [x + y + z for x, y, z in zip(a, b, c)]


# ---------------------------------------------------------------------------


def main():
    check_pi(PI)
    check_pi(PI_B)
    results = {
        "schema": "FP-EXPL-001/claude-verification/v1",
        "task_id": "FP-EXPL-001",
        "task_version": "1.1",
        "actor": "claude",
        "verifier": "self-contained independent re-implementation",
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
        },
        "command": "python -B docs/research_branches/FP-EXPL-001/claude/verify.py",
        "groups": {},
    }
    groups = results["groups"]

    def new_group(name):
        groups[name] = {"checks": 0, "failures": 0, "max_error": 0.0,
                        "details": []}
        return groups[name]

    def record(g, err, tol, label):
        g["checks"] += 1
        g["max_error"] = max(g["max_error"], abs(err))
        if abs(err) > tol:
            g["failures"] += 1
            g["details"].append({"label": label, "error": err, "tol": tol})

    # ---- A. sampling: regeneration, reconstruction, coverage -----------------
    ga = new_group("A_sampling_and_coverage")
    trans, draws = sample_batch()
    trans2, draws2 = sample_batch()
    record(ga, 0.0 if trans == trans2 and draws == draws2 else 1.0, 0.0,
           "two independent regenerations agree exactly (draws and trajectory)")
    record(ga, draws - 2 * BATCH, 0.0, "exactly two rng draws per transition")
    record(ga, trans[0][0] - INIT_STATE, 0.0, "initial state is 0")
    for t, (s, a, r, u, ua, ut) in enumerate(trans):
        a2 = 0 if ua < PI_B[s][0] else 1
        x = pidx(s, a)
        u2 = 0 if ut < P_NEXT[x][0] else 1
        record(ga, a - a2, 0.0, f"action threshold reconstruction t={t}")
        record(ga, u - u2, 0.0, f"transition threshold reconstruction t={t}")
        record(ga, r - REW[x], 0.0, f"reward mapping t={t}")
        record(ga, 0.0 if 0.0 <= ua < 1.0 and 0.0 <= ut < 1.0 else 1.0, 0.0,
               f"raw draws in [0,1) (numpy rng.random half-open range) t={t}")
        if t + 1 < len(trans):
            record(ga, trans[t + 1][0] - u, 0.0, f"trajectory continuity t={t}")
    counts = [sum(1 for s, a, _, _, _, _ in trans if pidx(s, a) == x)
              for x in range(4)]
    ga["details"].append({"label": "visit counts by pair 00,01,10,11",
                          "counts": counts})
    record(ga, 0.0 if min(counts) >= 1 else 1.0, 0.0,
           "H1 coverage: min n_x >= 1 (COVERAGE_FAILURE stop rule otherwise)")
    covered = min(counts) >= 1
    if not covered:
        results["summary"] = {
            "verdict": "COVERAGE_FAILURE",
            "note": "batch rejected before Q iteration per task section 3",
            "counts": counts,
        }
        json.dump(results, sys.stdout, allow_nan=False, indent=1)
        sys.stdout.write("\n")
        return

    rows = [(s, a, r, u) for s, a, r, u, _ua, _ut in trans]
    n = len(rows)

    # ---- B. matrix/probability structure -------------------------------------
    gb = new_group("B_probability_structure")
    c0, s0, w0, r0 = exact_mats(rows)
    for x in range(4):
        record(gb, sum(w0[x]) - 1.0, TOL1, f"W0 rowsum pair {x}")
        for t in range(n):
            if pidx(rows[t][0], rows[t][1]) == x:
                record(gb, w0[x][t] * counts[x] - 1.0, TOL1,
                       f"W0 grouped mean 1/n_x pair {x} t={t} (no visitation-"
                       "frequency multiplier)")
            else:
                record(gb, w0[x][t], TOL1, f"W0 zero off-group pair {x} t={t}")
    for t in range(n):
        record(gb, sum(c0[t]) - 1.0, TOL1, f"C0 rowsum t={t}")
        record(gb, sum(s0[t]) - 1.0, TOL1, f"S0 rowsum t={t}")
    c, s, w, rw = finite_weights(rows, XI, ZETA, TAU)
    for t in range(n):
        record(gb, sum(c[t]) - 1.0, TOL1, f"C rowsum t={t}")
        record(gb, sum(s[t]) - 1.0, TOL1, f"S rowsum t={t}")
    for x in range(4):
        record(gb, sum(w[x]) - 1.0, TOL1, f"W rowsum pair {x}")

    # ---- fixed points used by several groups ---------------------------------
    g0, b0 = affine(rows)
    gf, bf = affine(rows, XI, ZETA, TAU)
    q_hat = np.linalg.solve(np.eye(4) - np.array(g0), np.array(b0)).tolist()
    ppi = np.zeros((4, 4))
    for x in range(4):
        for y, (sy, by) in enumerate(PAIRS):
            ppi[x, y] = P_NEXT[x][sy] * PI[sy][by]
    q_pi = np.linalg.solve(
        np.eye(4) - GAMMA * ppi, np.array(REW)).tolist()
    v_pi = [sum(PI[s][b] * q_pi[pidx(s, b)] for b in (0, 1)) for s in (0, 1)]
    c_f = norm_inf_mat(gf)

    # One-step network probes: the four standard basis vectors plus Q0 = 0
    # only. Task section 4 makes q_pi audit-only and never a network input;
    # q_hat and arbitrary non-basis vectors are likewise not probed through
    # the network. q_hat/q_pi above remain in use for the audit-only
    # fixed-point, Bellman-residual, data-bias and decomposition checks.
    probes = [[1.0 if j == i else 0.0 for j in range(4)] for i in range(4)]
    probes += [[0.0] * 4]

    # ---- C. H2a: exact grouped attention == direct reference -------------------
    gc = new_group("C_H2a_exact_equals_direct")
    for q in probes:
        tokens = make_tokens(rows, q)
        out, _ = literal_net_step(tokens, rows, 0.0, 0.0, 0.0, "exact")
        ref = direct_step(q, rows)
        record(gc, norm_inf_vec(sub(out, ref)), TOL1, "one-step probe")
    q_d = list(Q0)
    tokens = make_tokens(rows, Q0)
    worst = 0.0
    trace_exact_direct = [list(Q0)]
    for _k in range(STEPS):
        q_d = direct_step(q_d, rows)
        trace_exact_direct.append(list(q_d))
        q_a, _ = literal_net_step(tokens, rows, 0.0, 0.0, 0.0, "exact")
        for a, b in zip(q_d, q_a):
            worst = max(worst, abs(a - b)
                        / (TOLR * (1.0 + max(abs(a), abs(b)))))
    record(gc, worst, 1.0, "64-update trace (scaled)")

    # ---- D. H2b: literal finite attention == scalar formula ---------------------
    gd = new_group("D_H2b_literal_equals_scalar")
    for q in probes:
        tokens = make_tokens(rows, q)
        out, _ = literal_net_step(tokens, rows, XI, ZETA, TAU, "finite")
        ref = finite_scalar_step(q, rows, XI, ZETA, TAU)
        record(gd, norm_inf_vec(sub(out, ref)), TOL1, "one-step probe")
    q_s = list(Q0)
    tokens = make_tokens(rows, Q0)
    worst = 0.0
    trace_finite = [list(Q0)]
    for _k in range(STEPS):
        q_s = finite_scalar_step(q_s, rows, XI, ZETA, TAU)
        trace_finite.append(list(q_s))
        q_l, _ = literal_net_step(tokens, rows, XI, ZETA, TAU, "finite")
        for a, b in zip(q_s, q_l):
            worst = max(worst, abs(a - b)
                        / (TOLR * (1.0 + max(abs(a), abs(b)))))
    record(gd, worst, 1.0, "64-update trace (scaled)")

    # ---- E. H3: signed stage errors telescope -------------------------------------
    ge = new_group("E_H3_telescoping")
    for k in range(STEPS):
        qk = trace_exact_direct[k]
        cq = matvec(c, qk)
        c0q = matvec(c0, qk)
        sq = matvec(s, qk)
        s0q = matvec(s0, qk)
        e_cur = [-ALPHA * sum(w[x][t] * (cq[t] - c0q[t]) for t in range(n))
                 for x in range(4)]
        e_suc = [ALPHA * GAMMA * sum(w[x][t] * (sq[t] - s0q[t]) for t in range(n))
                 for x in range(4)]
        d0 = [r0[t] + GAMMA * s0q[t] - c0q[t] for t in range(n)]
        e_wri = [ALPHA * sum((w[x][t] - w0[x][t]) * d0[t] for t in range(n))
                 for x in range(4)]
        f_f = finite_scalar_step(qk, rows, XI, ZETA, TAU)
        f_0 = direct_step(qk, rows)
        resid = norm_inf_vec(sub(add3(e_cur, e_suc, e_wri), sub(f_f, f_0)))
        record(ge, resid, TOL1, f"telescoping identity k={k}")

    # ---- F. affine reconstruction on the four standard basis vectors -------------
    gfz = new_group("F_affine_basis")
    f0_zero = direct_step([0.0] * 4, rows)
    for i in range(4):
        e_i = [1.0 if j == i else 0.0 for j in range(4)]
        lhs = sub(direct_step(e_i, rows), f0_zero)
        rhs = [g0[x][i] for x in range(4)]
        record(gfz, norm_inf_vec(sub(lhs, rhs)), TOL1, f"F0 basis e{i}")
    record(gfz, norm_inf_vec(sub(f0_zero, b0)), TOL1, "F0(0)=b0")
    ff_zero = finite_scalar_step([0.0] * 4, rows, XI, ZETA, TAU)
    for i in range(4):
        e_i = [1.0 if j == i else 0.0 for j in range(4)]
        lhs = sub(finite_scalar_step(e_i, rows, XI, ZETA, TAU), ff_zero)
        rhs = [gf[x][i] for x in range(4)]
        record(gfz, norm_inf_vec(sub(lhs, rhs)), TOL1, f"Ff basis e{i}")
    record(gfz, norm_inf_vec(sub(ff_zero, bf)), TOL1, "Ff(0)=bf")
    u1 = [0.3, -1.2, 0.7, 2.1]
    u2 = [-0.4, 0.9, 1.1, -0.6]
    lhs = sub(finite_scalar_step(u1, rows, XI, ZETA, TAU),
              finite_scalar_step(u2, rows, XI, ZETA, TAU))
    rhs = matvec(gf, sub(u1, u2))
    record(gfz, norm_inf_vec(sub(lhs, rhs)), TOL1, "Ff affinity probe")

    # ---- G. H4: exact contraction; finite contraction diagnostic -----------------
    gg = new_group("G_H4_contraction")
    record(gg, norm_inf_mat(g0) - C0, TOL1,
           "||G0||_inf equals 1-alpha(1-gamma)=0.85 on the covered batch")
    e0v = norm_inf_vec(sub(Q0, q_hat))
    for k in range(STEPS + 1):
        lhs = norm_inf_vec(sub(trace_exact_direct[k], q_hat))
        rhs = (C0 ** k) * e0v
        record(gg, max(0.0, lhs - rhs), TOLR * (1.0 + max(lhs, rhs)),
               f"exact contraction bound k={k} (one-sided)")
    gg["details"].append({
        "label": "finite operator contraction sufficient statistic",
        "c_f": c_f,
        "c_f_lt_1": c_f < 1.0,
        "rho_Gf_numerical_diagnostic": float(
            max(abs(np.linalg.eigvals(np.array(gf))))),
        "note": "c_f >= 1 would not imply divergence; rho is a floating-point "
        "diagnostic only",
    })

    # ---- H. perturbation bounds -----------------------------------------------------
    gh = new_group("H_perturbation_bounds")
    e_bound = 0.0
    for k in range(STEPS):
        qk = trace_exact_direct[k]
        delta = norm_inf_vec(sub(
            finite_scalar_step(qk, rows, XI, ZETA, TAU), direct_step(qk, rows)))
        e_bound = c_f * e_bound + delta
        actual = norm_inf_vec(sub(trace_finite[k + 1], trace_exact_direct[k + 1]))
        record(gh, max(0.0, actual - e_bound),
               TOLR * (1.0 + max(actual, e_bound)),
               f"E bound k+1={k + 1} (one-sided)")

    # ---- I. fixed points, data bias, decomposition, H5 --------------------------------
    gi = new_group("I_fixed_points_decomposition")
    resid_qh = norm_inf_vec(sub(direct_step(q_hat, rows), q_hat))
    record(gi, resid_qh, TOLR * (1.0 + norm_inf_vec(q_hat)), "F0(q_hat)=q_hat")
    bellman = []
    for x, (_sx, _ax) in enumerate(PAIRS):
        succ = sum(P_NEXT[x][sp] * sum(PI[sp][b] * q_pi[pidx(sp, b)]
                                       for b in (0, 1)) for sp in (0, 1))
        bellman.append(REW[x] + GAMMA * succ)
    record(gi, norm_inf_vec(sub(bellman, q_pi)),
           TOLR * (1.0 + norm_inf_vec(q_pi)),
           "population Bellman residual of q_pi (audit only)")
    f0_qpi = direct_step(q_pi, rows)
    resid_f0_qpi = norm_inf_vec(sub(f0_qpi, q_pi))
    data_lhs = norm_inf_vec(sub(q_hat, q_pi))
    data_rhs = resid_f0_qpi / (1.0 - C0)
    record(gi, max(0.0, data_lhs - data_rhs),
           TOLR * (1.0 + max(data_lhs, data_rhs)),
           "data-bias audit bound ||q_hat-q_pi|| <= ||F0(q_pi)-q_pi||/(1-c0) "
           "(one-sided)")
    gi["details"].append({
        "label": "H5 metrics",
        "V_pi_by_state": v_pi,
        "abs_state_value_difference": abs(v_pi[0] - v_pi[1]),
        "data_bias_norm_inf": data_lhs,
        "note": "state values must differ (proved); data bias reported as-is, "
        "not a success condition",
    })
    record(gi, 1.0 if abs(v_pi[0] - v_pi[1]) <= 1e-6 else 0.0, 0.0,
           "H5: the two true state values differ")
    if c_f < 1.0:
        q_f = np.linalg.solve(np.eye(4) - np.array(gf), np.array(bf)).tolist()
        resid_qf = norm_inf_vec(sub(
            finite_scalar_step(q_f, rows, XI, ZETA, TAU), q_f))
        record(gi, resid_qf, TOLR * (1.0 + norm_inf_vec(q_f)),
               "Ff(q_f_inf)=q_f_inf")
        gap = norm_inf_vec(sub(
            finite_scalar_step(q_hat, rows, XI, ZETA, TAU), q_hat))
        ss = norm_inf_vec(sub(q_f, q_hat)) - gap / (1.0 - c_f)
        record(gi, max(0.0, ss), TOLR * (1.0 + norm_inf_vec(q_f)),
               "steady-state bound (one-sided)")
        base = norm_inf_vec(sub(Q0, q_f))
        for k in range(STEPS + 1):
            lhs = norm_inf_vec(sub(trace_finite[k], q_f))
            rhs = (c_f ** k) * base
            record(gi, max(0.0, lhs - rhs), TOLR * (1.0 + max(lhs, rhs)),
                   f"trajectory bound k={k} (one-sided)")
        for k in range(STEPS + 1):
            qk = trace_finite[k]
            t123 = add3(sub(qk, q_f), sub(q_f, q_hat), sub(q_hat, q_pi))
            record(gi, norm_inf_vec(sub(sub(qk, q_pi), t123)), TOL1,
                   f"three-term decomposition identity k={k}")
            lhs = norm_inf_vec(sub(qk, q_pi))
            rhs = sum(norm_inf_vec(v) for v in
                      (sub(qk, q_f), sub(q_f, q_hat), sub(q_hat, q_pi)))
            record(gi, max(0.0, lhs - rhs), TOLR * (1.0 + max(lhs, rhs)),
                   f"norm triangle upper bound k={k} (one-sided)")
    else:
        gi["details"].append({
            "label": "finite fixed-point analysis unavailable",
            "reason": f"c_f = {c_f} >= 1; sufficient contraction condition "
            "fails; not a divergence claim; null per task section 5",
        })

    # ---- J. policy preservation (successor average uses target_pi) -------------------
    gj = new_group("J_policy_preservation")
    worst = 0.0
    beh = float("inf")
    for t in range(n):
        for u in (0, 1):
            denom = sum(s[t][pidx(u, b)] for b in (0, 1))
            for b in (0, 1):
                cond = s[t][pidx(u, b)] / denom
                worst = max(worst, abs(cond - PI[u][b]))
                beh = min(beh, abs(cond - PI_B[u][b]))
    record(gj, worst, TOL1, "successor conditional equals target_pi")
    record(gj, 1.0 if beh <= 1e-3 else 0.0, 0.0,
           "successor conditional is NOT the behavior policy (0.5, 0.5)")

    # ---- K. scratch clearing, immutable fields, capture consistency --------------------
    gk = new_group("K_network_integrity")
    tokens = make_tokens(rows, Q0)
    imm0 = [{k: (list(v) if isinstance(v, list) else v)
             for k, v in tok.items()} for tok in tokens]
    scratch_worst = 0.0
    ctx_q_worst = 0.0
    cap = None
    for k in range(1, STEPS + 1):
        _, cap = literal_net_step(tokens, rows, XI, ZETA, TAU, "finite",
                                  capture=True)
        scratch_worst = max(
            scratch_worst,
            max(abs(tok[key]) for tok in tokens for key in SCRATCH_KEYS))
        ctx_q_worst = max(ctx_q_worst,
                          max(abs(tokens[t]["q"]) for t in range(4, 4 + n + 1)))
    record(gk, scratch_worst, 0.0, "scratch fields exactly zero after every "
           "one of the 64 updates (fixed clearing map)")
    record(gk, ctx_q_worst, 0.0, "only the four memory tokens carry Q")
    imm_worst = 0.0
    for tok, before in zip(tokens, imm0):
        for key in IMMUTABLE_KEYS:
            after_v = tok[key]
            before_v = before[key]
            if isinstance(after_v, str):
                imm_worst = max(imm_worst,
                                0.0 if after_v == before_v else 1.0)
            elif isinstance(after_v, float):
                imm_worst = max(imm_worst, abs(after_v - before_v))
            else:
                imm_worst = max(
                    imm_worst,
                    max(abs(a - b) for a, b in zip(after_v, before_v)))
    record(gk, imm_worst, 0.0, "immutable prompt fields bit-identical after "
           "64 updates")
    # capture consistency on the final captured update
    sq = matvec(s, trace_finite[STEPS - 1])
    cq = matvec(c, trace_finite[STEPS - 1])
    d_ref = [rw[t] + GAMMA * sq[t] - cq[t] for t in range(n)]
    err_res = max(abs(cap["res_before_reset"][4 + t] - d_ref[t])
                  for t in range(n))
    record(gk, err_res, TOL1, "captured pre-reset residual == scalar d_t")
    err_wd = 0.0
    for x in range(4):
        ks, probs = cap["stage4_probs_keys"][str(x)]
        wd_from_capture = sum(p * cap["res_before_reset"][kj]
                              for p, kj in zip(probs, ks))
        err_wd = max(err_wd, abs(wd_from_capture - cap["wd_before_reset"][x]))
    record(gk, err_wd, TOL1, "writer probs x captured residual == wd")
    err_upd = max(abs(trace_finite[STEPS][x] - trace_finite[STEPS - 1][x]
                      - ALPHA * cap["wd_before_reset"][x]) for x in range(4))
    record(gk, err_upd, TOL1, "q update == q + alpha * captured wd")
    record(gk, 0.0 if cap["post_reset_scratch_zero"] else 1.0, 0.0,
           "post-reset scratch zeroed")

    # ---- L. rejection of malformed inputs ----------------------------------------------
    gm = new_group("L_input_rejection")
    rejection_cases = [
        ("pi wrong shape", lambda: check_pi(((0.5, 0.5),))),
        ("pi negative", lambda: check_pi(((1.25, -0.25), (0.25, 0.75)))),
        ("pi unnormalized", lambda: check_pi(((0.8, 0.25), (0.25, 0.75)))),
        ("pi zero entry", lambda: check_pi(((1.0, 0.0), (0.25, 0.75)))),
        ("q nonfinite", lambda: check_q((0.0, float("nan"), 1.0, 2.0))),
        ("q wrong length", lambda: check_q((0.0, 1.0))),
        ("reward nonfinite", lambda: check_rows(((0, 0, float("inf"), 0),))),
        ("bad state index", lambda: check_rows(((2, 0, 1.0, 0),))),
        ("empty batch", lambda: check_rows(())),
    ]
    for label, fn in rejection_cases:
        try:
            fn()
            record(gm, 1.0, 0.0, f"NOT rejected: {label}")
        except Reject:
            record(gm, 0.0, 0.0, f"rejected: {label}")

    # ---- summary ----------------------------------------------------------------------
    total_checks = sum(g["checks"] for g in groups.values())
    total_failures = sum(g["failures"] for g in groups.values())
    results["summary"] = {
        "total_checks": total_checks,
        "total_failures": total_failures,
        "max_error_by_group": {k: g["max_error"] for k, g in groups.items()},
        "coverage_counts": counts,
        "c_f": c_f,
        "verdict": "PASS" if total_failures == 0 else "FAIL",
        "tolerances": {"one_step_and_probabilities": TOL1,
                       "repeated_traces_solves_decompositions":
                           "1e-10*(1+max_abs(left,right)); inequalities add "
                           "the tolerance on the right side",
                       "raw_draws_and_discrete_trajectory": "exact equality"},
    }
    text = json.dumps(results, allow_nan=False, indent=1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "verification.json"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    sys.stdout.write(text + "\n")
    if results["summary"]["verdict"] == "FAIL":
        # A failed verification must be visible to the calling process:
        # exit nonzero. (COVERAGE_FAILURE is a protocol stop event reported
        # in the JSON, not a verifier failure; it returns above with 0.)
        sys.exit(1)


if __name__ == "__main__":
    main()
