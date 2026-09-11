"""FP-EXPL-001 Claude main-route witness: fixed exploration batch, grouped-mean Q iteration.

Independent implementation containing:
- the frozen sampler: numpy.random.Generator(numpy.random.PCG64(20260911)),
  64 sequential transitions from initial state 0, exactly two scalar
  rng.random() draws per transition in the task order (action, then
  transition); no extra draws, no resampling, no shuffling, no seed choice;
- the coverage diagnostic: min n_x >= 1 over the four state-action pairs,
  checked before any Q update; on failure the batch is rejected with
  COVERAGE_FAILURE and no Q iteration is run;
- the direct grouped-mean reference F0 (plain loops, no attention machinery);
- the exact grouped-attention operator (declared equality masks, which are
  not counted as a finite-network capability);
- the literal finite-softmax attention witness: prompt matrix H, explicit
  fixed WQ/WK/WV/WO matrices, scaled-dot-product normalized softmax
  attention, residual connections, fixed linear feed-forward maps and a
  fixed scratch-clearing projection; static token-role masks only, no
  content-equality mask and no visitation gate in finite mode;
- an independently written scalar loop implementation of the finite operator;
- the frozen protocol: Q0 = 0, alpha = 0.5, gamma = 0.7, xi = zeta = tau = 8,
  64 synchronous updates on the single frozen batch;
- signed stage-error decomposition, affine maps, fixed points, audit-only
  population comparison and all bounds.

No visitation-frequency multiplier is used anywhere; the behavior policy is
used only by the sampler, the successor average uses target_pi only, and the
true transition matrix P is used only by the sampler and the audit.

Writes strict JSON evidence (no NaN/Infinity literals) to stdout and to
results/FP-EXPL-001/claude/results.json.
Run from icrl_softmax:
python -B docs/research_branches/FP-EXPL-001/claude/witness.py
"""

import hashlib
import json
import math
import os
import platform
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Frozen inputs (task FP-EXPL-001 v1.1). Do not retune.
# ---------------------------------------------------------------------------

GAMMA = 0.7
ALPHA = 0.5
STEPS = 64
TARGET_PI = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=np.float64)
BEHAVIOR_PI = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
REWARDS = np.array([1.0, -0.5, 0.25, 0.25], dtype=np.float64)  # by pair 00,01,10,11
P_NEXT = np.array(
    [[0.75, 0.25], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25]], dtype=np.float64
)  # rows by pair, columns next state; sampler and audit only
INITIAL_STATE = 0
BATCH_LENGTH = 64
SEED = 20260911
Q0 = np.zeros(4, dtype=np.float64)
XI = 8.0
ZETA = 8.0
TAU = 8.0
CANONICAL_PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))  # order 00, 01, 10, 11

TOL_ONE_STEP = 1e-12
TOL_REPEATED = 1e-10
C0_BOUND = 1.0 - ALPHA * (1.0 - GAMMA)  # 0.85, proved for any covered batch

ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
    )
)
RESULTS_DIR = os.path.join(ROOT, "results", "FP-EXPL-001", "claude")


class InputReject(ValueError):
    """Raised when malformed or nonfinite inputs are rejected."""


def pair_index(s, a):
    if s not in (0, 1) or a not in (0, 1):
        raise InputReject(f"invalid state/action ({s}, {a})")
    return 2 * s + a


def validate_policy(pi):
    arr = np.asarray(pi, dtype=np.float64)
    if arr.shape != (2, 2):
        raise InputReject("policy must have shape (2, 2)")
    if not np.all(np.isfinite(arr)):
        raise InputReject("policy contains nonfinite entries")
    if np.any(arr <= 0.0):
        raise InputReject("policy must have full support (log pi is used)")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=TOL_ONE_STEP, rtol=0.0):
        raise InputReject("policy rows must sum to 1")
    return arr


def validate_q(q):
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape != (4,):
        raise InputReject("q must have shape (4,)")
    if not np.all(np.isfinite(arr)):
        raise InputReject("q contains nonfinite entries")
    return arr


def validate_rows(rows):
    out = []
    for row in rows:
        s, a, r, u = row
        pair_index(s, a)
        if u not in (0, 1):
            raise InputReject(f"invalid next state {u}")
        if not math.isfinite(r):
            raise InputReject("nonfinite reward")
        out.append((int(s), int(a), float(r), int(u)))
    if len(out) == 0:
        raise InputReject("empty batch")
    return out


# ---------------------------------------------------------------------------
# Frozen sampler. Exactly two scalar draws per transition, in task order.
# ---------------------------------------------------------------------------


def sample_batch():
    """Return (transitions, draw_count) for the single frozen batch.

    Per transition, in this exact order (task FP-EXPL-001 v1.1 section 3):
      1. u_action = rng.random(); a = 0 iff u_action < 0.5
      2. u_transition = rng.random();
         s_next = 0 iff u_transition < P_next_state[2*s+a, 0]
    then record (s, a, rewards[2*s+a], s_next, u_action, u_transition) and
    continue from s_next. The generator is never used for anything else.
    """
    rng = np.random.Generator(np.random.PCG64(SEED))
    s = INITIAL_STATE
    transitions = []
    draws = 0
    for _ in range(BATCH_LENGTH):
        u_action = float(rng.random())
        draws += 1
        a = 0 if u_action < float(BEHAVIOR_PI[s, 0]) else 1
        u_transition = float(rng.random())
        draws += 1
        x = pair_index(s, a)
        s_next = 0 if u_transition < float(P_NEXT[x, 0]) else 1
        transitions.append(
            (s, a, float(REWARDS[x]), s_next, u_action, u_transition)
        )
        s = s_next
    return transitions, draws


def coverage_counts(transitions):
    counts = [0, 0, 0, 0]
    for s, a, _r, _u, _ua, _ut in transitions:
        counts[pair_index(s, a)] += 1
    return counts


# ---------------------------------------------------------------------------
# Exact and finite formula matrices.
# ---------------------------------------------------------------------------


def softmax_rows(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def exact_matrices(rows, pi):
    """C0[t,y]=1{x_t=y}; S0[t,(u,b)]=1{u_t=u}pi(b|u); W0[x,t]=1{x=x_t}/n_x."""
    pi = validate_policy(pi)
    rows = validate_rows(rows)
    n = len(rows)
    c0 = np.zeros((n, 4))
    s0 = np.zeros((n, 4))
    w0 = np.zeros((4, n))
    r = np.zeros(n)
    for t, (s, a, rw, u) in enumerate(rows):
        c0[t, pair_index(s, a)] = 1.0
        for b in (0, 1):
            s0[t, pair_index(u, b)] = pi[u, b]
        r[t] = rw
    for x in range(4):
        match = [t for t, (s, a, _, _) in enumerate(rows) if pair_index(s, a) == x]
        if not match:
            raise InputReject(f"pair {CANONICAL_PAIRS[x]} absent from batch")
        for t in match:
            w0[x, t] = 1.0 / len(match)
    return c0, s0, w0, r


def finite_matrices(rows, pi, xi, zeta, tau):
    """C[t,y]=softmax_y(xi 1{x_t=y}); S[t,(u,b)]=softmax(zeta 1{u_t=u}+log pi);
    W[x,t]=softmax_t(tau 1{x=x_t}). Indicator products are dot products of
    declared one-hot identity features; there is no visited-query gate and no
    content-equality mask in the finite route."""
    pi = validate_policy(pi)
    rows = validate_rows(rows)
    for name, v in (("xi", xi), ("zeta", zeta), ("tau", tau)):
        if not math.isfinite(v):
            raise InputReject(f"nonfinite sharpness {name}")
    n = len(rows)
    x_ids = [pair_index(s, a) for s, a, _, _ in rows]
    u_ids = [u for _, _, _, u in rows]
    r = np.array([rw for _, _, rw, _ in rows], dtype=np.float64)
    logpi_pair = np.array(
        [math.log(pi[s, a]) for (s, a) in CANONICAL_PAIRS], dtype=np.float64
    )
    logits_c = np.array(
        [[xi * (1.0 if x_ids[t] == y else 0.0) for y in range(4)] for t in range(n)]
    )
    logits_s = np.array(
        [
            [
                zeta * (1.0 if u_ids[t] == CANONICAL_PAIRS[y][0] else 0.0)
                + logpi_pair[y]
                for y in range(4)
            ]
            for t in range(n)
        ]
    )
    logits_w = np.array(
        [[tau * (1.0 if x == x_ids[t] else 0.0) for t in range(n)] for x in range(4)]
    )
    return softmax_rows(logits_c), softmax_rows(logits_s), softmax_rows(logits_w), r


def exact_operator(q, mats):
    c0, s0, w0, r = mats
    q = validate_q(q)
    return q + ALPHA * (w0 @ (r + GAMMA * (s0 @ q) - c0 @ q))


def finite_operator(q, fmats):
    c, s, w, r = fmats
    q = validate_q(q)
    return q + ALPHA * (w @ (r + GAMMA * (s @ q) - c @ q))


def affine_maps(mats):
    c, s, w, r = mats
    g = np.eye(4) + ALPHA * (w @ (GAMMA * s - c))
    b = ALPHA * (w @ r)
    return g, b


# ---------------------------------------------------------------------------
# Direct reference: grouped residuals without any attention implementation.
# ---------------------------------------------------------------------------


def direct_reference_step(q, rows, pi):
    pi = validate_policy(pi)
    rows = validate_rows(rows)
    q = validate_q(q)
    out = q.copy()
    for x, (_sx, _ax) in enumerate(CANONICAL_PAIRS):
        residuals = []
        for s, a, rw, u in rows:
            if pair_index(s, a) == x:
                succ = pi[u, 0] * q[pair_index(u, 0)] + pi[u, 1] * q[pair_index(u, 1)]
                residuals.append(rw + GAMMA * succ - q[x])
        if residuals:
            out[x] = q[x] + ALPHA * sum(residuals) / len(residuals)
    return out


# ---------------------------------------------------------------------------
# Independently written scalar loop implementation of the finite operator.
# ---------------------------------------------------------------------------


def finite_scalar_step(q, rows, pi, xi, zeta, tau):
    pi = validate_policy(pi)
    rows = validate_rows(rows)
    q = validate_q(q)
    n = len(rows)
    x_ids = [pair_index(s, a) for s, a, _, _ in rows]
    u_ids = [u for _, _, _, u in rows]
    rewards = [rw for _, _, rw, _ in rows]
    logpi_pair = [math.log(pi[s, a]) for (s, a) in CANONICAL_PAIRS]
    c_rows = []
    s_rows = []
    for t in range(n):
        lc = [xi if x_ids[t] == y else 0.0 for y in range(4)]
        mc = max(lc)
        ec = [math.exp(v - mc) for v in lc]
        zc = sum(ec)
        c_rows.append([v / zc for v in ec])
        ls = [
            (zeta if u_ids[t] == CANONICAL_PAIRS[y][0] else 0.0) + logpi_pair[y]
            for y in range(4)
        ]
        ms = max(ls)
        es = [math.exp(v - ms) for v in ls]
        zs = sum(es)
        s_rows.append([v / zs for v in es])
    w_rows = []
    for x in range(4):
        lw = [tau if x == x_ids[t] else 0.0 for t in range(n)]
        mw = max(lw)
        ew = [math.exp(v - mw) for v in lw]
        zw = sum(ew)
        w_rows.append([v / zw for v in ew])
    out = [0.0] * 4
    for x in range(4):
        acc = 0.0
        for t in range(n):
            succ = sum(s_rows[t][y] * q[y] for y in range(4))
            cur = sum(c_rows[t][y] * q[y] for y in range(4))
            acc += w_rows[x][t] * (rewards[t] + GAMMA * succ - cur)
        out[x] = q[x] + ALPHA * acc
    return np.array(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Literal attention witness.
#
# Embedding fields (DIM=19):
#   0 const, 1 role_mem, 2 role_ctx, 3 role_null,
#   4..7 pair one-hot, 8..9 ustate one-hot, 10..11 action one-hot,
#   12 reward, 13 q, 14 logpi,
#   15 scratch_current, 16 scratch_successor, 17 scratch_residual,
#   18 scratch_writeback
# Tokens: 0..3 memory (one per canonical pair), 4..4+N-1 context (batch rows),
#   4+N null.
# Stages per update:
#   1 current-pair read:  context queries, memory keys; logit xi * 1{x_t=y}
#   2 successor read:     context queries, memory keys; logit zeta 1{u_t=u}+log pi
#   3 residual map:       fixed linear feed-forward d = r + gamma*succ - cur
#   4 pair writeback:     memory queries, context keys; logit tau * 1{x=x_t}
#   5 q update + reset:   fixed linear maps q += alpha*wd; scratch fields zeroed
# Static role/position masks: context queries see memory keys, memory queries
# see context keys, inactive queries read the null token (value zero). Masks
# never depend on observed pair equality in finite mode.
# ---------------------------------------------------------------------------

F_CONST = 0
F_ROLE_MEM = 1
F_ROLE_CTX = 2
F_ROLE_NULL = 3
F_PAIR = 4  # 4..7
F_UST = 8  # 8..9
F_ACT = 10  # 10..11
F_R = 12
F_Q = 13
F_LOGPI = 14
F_CUR = 15
F_SUCC = 16
F_RES = 17
F_WD = 18
DIM = 19

SCRATCH_FIELDS = (F_CUR, F_SUCC, F_RES, F_WD)
IMMUTABLE_FIELDS = (
    F_CONST,
    F_ROLE_MEM,
    F_ROLE_CTX,
    F_ROLE_NULL,
    F_PAIR,
    F_PAIR + 1,
    F_PAIR + 2,
    F_PAIR + 3,
    F_UST,
    F_UST + 1,
    F_ACT,
    F_ACT + 1,
    F_R,
    F_LOGPI,
)

FIELD_LEGEND = {
    "0": "const(1)",
    "1": "role_memory",
    "2": "role_context",
    "3": "role_null",
    "4..7": "pair one-hot (memory: own pair; context: observed pair x_t)",
    "8..9": "state one-hot (memory: own state; context: next state u_t)",
    "10..11": "action one-hot (memory: own action; context: observed action a_t)",
    "12": "reward r_t (context)",
    "13": "dynamic Q value (memory)",
    "14": "log pi(b|u) of own pair (memory)",
    "15": "scratch current-pair read (context)",
    "16": "scratch successor expectation (context)",
    "17": "scratch TD residual (context)",
    "18": "scratch aggregated writeback (memory)",
}


class Stage:
    def __init__(self, wq, wk, wv, wo, scale, allowed, exact_probs):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.scale = scale
        self.allowed = allowed  # static role/position boolean mask
        self.exact_probs = exact_probs  # declared equality route (exact mode)


class LiteralNetwork:
    """Fixed-weight normalized-softmax attention witness for one batch."""

    def __init__(self, rows, pi, mode, xi=0.0, zeta=0.0, tau=0.0):
        self.pi = validate_policy(pi)
        self.rows = validate_rows(rows)
        if mode not in ("finite", "exact"):
            raise InputReject("mode must be 'finite' or 'exact'")
        self.mode = mode
        self.xi, self.zeta, self.tau = float(xi), float(zeta), float(tau)
        self.n = len(self.rows)
        self.null_token = 4 + self.n
        self.n_tokens = self.null_token + 1
        self.h = self._build_prompt(np.zeros(4))
        self._build_stages()
        self.m3 = np.eye(DIM)
        self.m3[F_RES, F_R] = 1.0
        self.m3[F_RES, F_SUCC] = GAMMA
        self.m3[F_RES, F_CUR] = -1.0
        self.m5 = np.eye(DIM)
        self.m5[F_Q, F_WD] = ALPHA
        self.p_reset = np.eye(DIM)
        for f in SCRATCH_FIELDS:
            self.p_reset[f, f] = 0.0

    def _build_prompt(self, q0):
        q0 = validate_q(q0)
        h = np.zeros((self.n_tokens, DIM))
        h[:, F_CONST] = 1.0
        for x, (s, a) in enumerate(CANONICAL_PAIRS):
            h[x, F_ROLE_MEM] = 1.0
            h[x, F_PAIR + pair_index(s, a)] = 1.0
            h[x, F_UST + s] = 1.0
            h[x, F_ACT + a] = 1.0
            h[x, F_LOGPI] = math.log(self.pi[s, a])
            h[x, F_Q] = q0[x]
        for t, (s, a, rw, u) in enumerate(self.rows):
            tok = 4 + t
            h[tok, F_ROLE_CTX] = 1.0
            h[tok, F_PAIR + pair_index(s, a)] = 1.0
            h[tok, F_UST + u] = 1.0
            h[tok, F_ACT + a] = 1.0
            h[tok, F_R] = rw
        h[self.null_token, F_ROLE_NULL] = 1.0
        return h

    def set_q(self, q):
        """Prompt-initialization interface: load dynamic Q into memory tokens."""
        q = validate_q(q)
        self.h[0:4, F_Q] = q
        self.h[0:4, F_WD] = 0.0
        self.h[4:, F_CUR] = 0.0
        self.h[4:, F_SUCC] = 0.0
        self.h[4:, F_RES] = 0.0

    def current_q(self):
        """Positional extraction of the four Q-memory outputs."""
        return self.h[0:4, F_Q].copy()

    def _build_stages(self):
        ntok = self.n_tokens
        mem = list(range(4))
        ctx = [4 + t for t in range(self.n)]
        null = self.null_token

        def role_mask(active, keys):
            m = np.zeros((ntok, ntok), dtype=bool)
            for qtok in range(ntok):
                if qtok in active:
                    for ktok in keys:
                        m[qtok, ktok] = True
                else:
                    m[qtok, null] = True
            return m

        # Stage 1: current-pair read, logit xi * 1{x_t = y} (dot of one-hots).
        wq1 = np.zeros((4, DIM))
        wk1 = np.zeros((4, DIM))
        for j in range(4):
            wq1[j, F_PAIR + j] = 2.0 * self.xi  # sqrt(d_k)=2 folded in
            wk1[j, F_PAIR + j] = 1.0
        wv1 = np.zeros((1, DIM))
        wv1[0, F_Q] = 1.0
        wo1 = np.zeros((DIM, 1))
        wo1[F_CUR, 0] = 1.0
        # Stage 2: successor read, logit zeta * 1{u_t = u_y} + log pi(b_y|u_y).
        rt3 = math.sqrt(3.0)
        wq2 = np.zeros((3, DIM))
        wk2 = np.zeros((3, DIM))
        wq2[0, F_UST + 0] = rt3 * self.zeta
        wq2[1, F_UST + 1] = rt3 * self.zeta
        wq2[2, F_CONST] = rt3
        wk2[0, F_UST + 0] = 1.0
        wk2[1, F_UST + 1] = 1.0
        wk2[2, F_LOGPI] = 1.0
        wv2 = np.zeros((1, DIM))
        wv2[0, F_Q] = 1.0
        wo2 = np.zeros((DIM, 1))
        wo2[F_SUCC, 0] = 1.0
        # Stage 4: pair writeback, logit tau * 1{x = x_t}.
        wq4 = np.zeros((4, DIM))
        wk4 = np.zeros((4, DIM))
        for j in range(4):
            wq4[j, F_PAIR + j] = 2.0 * self.tau
            wk4[j, F_PAIR + j] = 1.0
        wv4 = np.zeros((1, DIM))
        wv4[0, F_RES] = 1.0
        wo4 = np.zeros((DIM, 1))
        wo4[F_WD, 0] = 1.0

        allowed1 = role_mask(ctx, mem)
        allowed2 = role_mask(ctx, mem)
        allowed4 = role_mask(mem, ctx)

        # Declared equality probabilities for the exact reference route only.
        x_ids = [pair_index(s, a) for s, a, _, _ in self.rows]
        u_ids = [u for _, _, _, u in self.rows]
        p1 = np.zeros((ntok, ntok))
        p2 = np.zeros((ntok, ntok))
        p4 = np.zeros((ntok, ntok))
        for t in range(self.n):
            p1[4 + t, x_ids[t]] = 1.0
            for y, (sy, by) in enumerate(CANONICAL_PAIRS):
                if sy == u_ids[t]:
                    p2[4 + t, y] = self.pi[sy, by]
        for x in range(4):
            match = [t for t in range(self.n) if x_ids[t] == x]
            if match:
                for t in match:
                    p4[x, 4 + t] = 1.0 / len(match)
            else:
                p4[x, null] = 1.0  # absent-row null mask; null value is zero
        for qtok in range(ntok):
            if qtok not in ctx:
                p1[qtok, null] = 1.0
                p2[qtok, null] = 1.0
            if qtok not in mem:
                p4[qtok, null] = 1.0

        self.stage1 = Stage(wq1, wk1, wv1, wo1, 0.5, allowed1, p1)
        self.stage2 = Stage(wq2, wk2, wv2, wo2, 1.0 / rt3, allowed2, p2)
        self.stage4 = Stage(wq4, wk4, wv4, wo4, 0.5, allowed4, p4)

    def _attention(self, h, st):
        q = h @ st.wq.T
        k = h @ st.wk.T
        v = h @ st.wv.T
        scores = (q @ k.T) * st.scale
        if self.mode == "finite":
            masked = np.where(st.allowed, scores, -np.inf)
            probs = softmax_rows(masked)
        else:
            probs = st.exact_probs
        out = probs @ v
        return {"queries": q, "keys": k, "values": v, "scores": scores,
                "probs": probs, "out": out}

    def step(self, capture=False):
        h = self.h
        a1 = self._attention(h, self.stage1)
        h = h + a1["out"] @ self.stage1.wo.T
        a2 = self._attention(h, self.stage2)
        h = h + a2["out"] @ self.stage2.wo.T
        h = h @ self.m3.T
        a4 = self._attention(h, self.stage4)
        h = h + a4["out"] @ self.stage4.wo.T
        h = h @ self.m5.T
        exposed = None
        if capture:
            # Snapshot BEFORE the scratch-reset projection: p_reset zeroes
            # F_CUR/F_SUCC/F_RES/F_WD, so a post-reset snapshot would show
            # only zeros.
            ctx = [4 + t for t in range(self.n)]
            stage4_value_read = a4["values"][ctx, 0]
            captured_residual = h[ctx, F_RES]
            exposed = {
                "stage1_current_read": a1,
                "stage2_successor_read": a2,
                "stage4_writeback": a4,
                "residual_field_before_scratch_reset": self.h_res_snapshot(h),
                "residual_capture_note": (
                    "snapshot taken after the residual feed-forward map, the "
                    "writeback and the q update, before the fixed scratch-reset "
                    "projection; scratch_residual therefore holds the TD "
                    "residual d_t that the stage-4 writer actually read"
                ),
                "captured_residual_vs_stage4_value_read_max_abs": float(
                    np.max(np.abs(captured_residual - stage4_value_read))
                ),
            }
        h = h @ self.p_reset.T
        self.h = h
        return self.current_q(), exposed

    @staticmethod
    def h_res_snapshot(h):
        return h[:, [F_R, F_CUR, F_SUCC, F_RES, F_WD, F_Q]].tolist()

    def run(self, steps, q0, capture_updates=(1,)):
        self.h = self._build_prompt(q0)
        trace = [self.current_q()]
        captured = {}
        for k in range(1, steps + 1):
            _, exposed = self.step(capture=k in capture_updates)
            trace.append(self.current_q())
            if exposed is not None:
                captured[f"update_{k}_from_step_{k - 1}"] = exposed
        return trace, captured


# ---------------------------------------------------------------------------
# Fixed points, audit population solution, and bounds.
# ---------------------------------------------------------------------------


def spectral_radius(g):
    return float(max(abs(np.linalg.eigvals(g))))


def solve_fixed_point(g, b):
    return np.linalg.solve(np.eye(4) - g, b)


def population_q_pi():
    """Audit-only population fixed-policy Bellman solution (never a network input)."""
    ppi = np.zeros((4, 4))
    for x in range(4):
        for y, (sy, by) in enumerate(CANONICAL_PAIRS):
            ppi[x, y] = P_NEXT[x, sy] * TARGET_PI[sy, by]
    return np.linalg.solve(np.eye(4) - GAMMA * ppi, REWARDS)


def repeated_tol(a, b):
    return TOL_REPEATED * (1.0 + max(abs(a), abs(b)))


# ---------------------------------------------------------------------------
# JSON helpers.
# ---------------------------------------------------------------------------


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def git_head():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        )
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001 - evidence must not crash on git absence
        return f"unavailable: {exc}"


# ---------------------------------------------------------------------------
# Protocol driver.
# ---------------------------------------------------------------------------


def run_protocol():
    evidence = {
        "schema": "FP-EXPL-001/claude-witness/v1",
        "task_id": "FP-EXPL-001",
        "task_version": "1.1",
        "task_status": "ACTIVE",
        "actor": "claude",
        "route": "main author route (user ruling: Claude executes, GPT accepts)",
        "code": {
            "baseline_commit_at_start": git_head(),
            "baseline_commit_at_start_meaning": (
                "git HEAD of the run checkout recorded at witness start. "
                "Because the author seals the code before rerunning, this "
                "denotes the sealed run checkout, NOT the frozen scientific "
                "baseline; it is kept under its original field name for "
                "schema continuity and documented here."
            ),
            "frozen_scientific_baseline_commit": (
                "c710e32b24d77085adea134c3f470773037ffbb1"
            ),
            "common_active_publication_commit": (
                "8c915c4bf2bb9533e2374f5d2c91cf34e5c13c77"
            ),
            "witness_sha256": file_sha256(os.path.abspath(__file__)),
            "verify_sha256": file_sha256(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify.py")
            ),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "compute": "CPU float64",
        },
        "command": "python -B docs/research_branches/FP-EXPL-001/claude/witness.py",
        "frozen_inputs": {
            "gamma": GAMMA,
            "alpha": ALPHA,
            "target_pi": TARGET_PI,
            "behavior_pi": BEHAVIOR_PI,
            "rewards_by_pair_00_01_10_11": REWARDS,
            "P_next_state_by_pair": P_NEXT,
            "canonical_pairs": [list(p) for p in CANONICAL_PAIRS],
            "initial_state": INITIAL_STATE,
            "batch_length": BATCH_LENGTH,
            "seed": SEED,
            "Q0": Q0,
            "updates": STEPS,
            "xi": XI,
            "zeta": ZETA,
            "tau": TAU,
            "pair_order": "00,01,10,11",
        },
        "token_layout": FIELD_LEGEND,
        "tolerances": {
            "one_step_and_probabilities_abs": TOL_ONE_STEP,
            "repeated_traces_solves_decompositions": "1e-10*(1+max_abs(left,right))",
            "inequalities": "tolerance added on the right side",
            "raw_draws_and_discrete_trajectory": "exact equality across routes",
        },
    }
    failures = []

    # ---- frozen sampling, reconstruction, coverage -------------------------
    transitions, draw_count = sample_batch()
    counts = coverage_counts(transitions)
    recon_max = 0.0
    for t, (s, a, r, u, ua, ut) in enumerate(transitions):
        a2 = 0 if ua < float(BEHAVIOR_PI[s, 0]) else 1
        x = pair_index(s, a)
        u2 = 0 if ut < float(P_NEXT[x, 0]) else 1
        recon_max = max(
            recon_max,
            abs(a - a2),
            abs(u - u2),
            abs(r - float(REWARDS[x])),
        )
        if t + 1 < len(transitions):
            recon_max = max(recon_max, abs(transitions[t + 1][0] - u))
    recon_max = max(recon_max, abs(transitions[0][0] - INITIAL_STATE))
    evidence["sampling"] = {
        "seed": SEED,
        "rng": "numpy.random.Generator(numpy.random.PCG64(seed))",
        "initial_state": INITIAL_STATE,
        "batch_length": BATCH_LENGTH,
        "total_rng_draws": draw_count,
        "draws_per_transition": 2,
        "draw_order": "u_action then u_transition per transition",
        "transitions": [
            {
                "t": t,
                "s": s,
                "a": a,
                "r": r,
                "s_next": u,
                "u_action": ua,
                "u_transition": ut,
            }
            for t, (s, a, r, u, ua, ut) in enumerate(transitions)
        ],
        "visit_counts_by_pair_00_01_10_11": counts,
        "min_count": min(counts),
        "coverage_pass": bool(min(counts) >= 1),
        "threshold_reconstruction_max_abs_error": recon_max,
    }
    if min(counts) < 1:
        evidence["status"] = "COVERAGE_FAILURE"
        evidence["missing_pairs"] = [
            list(CANONICAL_PAIRS[x]) for x in range(4) if counts[x] == 0
        ]
        evidence["note"] = (
            "per task section 3 the batch Q iteration is stopped; no resampling, "
            "no batch enlargement; construction completion is not claimed"
        )
        evidence["failed_runs"] = failures
        evidence["witness_self_verdict"] = "COVERAGE_FAILURE"
        return evidence

    rows = [(s, a, r, u) for s, a, r, u, _ua, _ut in transitions]

    # ---- operators and affine maps -----------------------------------------
    em = exact_matrices(rows, TARGET_PI)
    fm = finite_matrices(rows, TARGET_PI, XI, ZETA, TAU)
    g0, b0 = affine_maps(em)
    gf, bf = affine_maps(fm)
    c_f = float(np.linalg.norm(gf, ord=np.inf))
    norm_g0 = float(np.linalg.norm(g0, ord=np.inf))
    evidence["operators"] = {
        "exact": {
            "C0": em[0],
            "S0": em[1],
            "W0": em[2],
            "r": em[3],
            "G0": g0,
            "b0": b0,
            "norm_inf_G0": norm_g0,
            "c0_proved_bound": C0_BOUND,
            "rho_G0_numerical_diagnostic": spectral_radius(g0),
        },
        "finite": {
            "C": fm[0],
            "S": fm[1],
            "W": fm[2],
            "r": fm[3],
            "Gf": gf,
            "bf": bf,
            "c_f": c_f,
            "rho_Gf_numerical_diagnostic": spectral_radius(gf),
            "spectral_note": (
                "rho(Gf) is a floating-point numerical diagnostic only; it "
                "does not by itself prove spectral properties, a unique fixed "
                "point, or convergence; c_f >= 1 would not imply divergence"
            ),
        },
    }

    # ---- fixed points and audit ---------------------------------------------
    q_hat = solve_fixed_point(g0, b0)
    q_pi = population_q_pi()
    v_pi = np.array(
        [sum(TARGET_PI[s, b] * q_pi[pair_index(s, b)] for b in (0, 1)) for s in (0, 1)]
    )
    r_bar = np.array(
        [
            sum(TARGET_PI[s, b] * REWARDS[pair_index(s, b)] for b in (0, 1))
            for s in (0, 1)
        ]
    )
    f0_qpi_resid = float(np.max(np.abs(exact_operator(q_pi, em) - q_pi)))
    data_bias_lhs = float(np.max(np.abs(q_hat - q_pi)))
    data_bias_rhs = f0_qpi_resid / (1.0 - C0_BOUND)
    fixed_points = {
        "q_hat_empirical": {
            "value": q_hat,
            "residual_inf": float(
                np.max(np.abs(exact_operator(q_hat, em) - q_hat))
            ),
            "justification": (
                "coverage passed; ||G0||_inf = 1-alpha(1-gamma) = 0.85 < 1 "
                "(proved for any covered batch), so F0 has a unique fixed point"
            ),
        },
        "q_pi_audit": {
            "value": q_pi,
            "residual_inf": float(
                np.max(
                    np.abs(
                        q_pi
                        - (
                            REWARDS
                            + GAMMA
                            * (
                                P_NEXT
                                @ np.array(
                                    [
                                        TARGET_PI[sp, 0] * q_pi[pair_index(sp, 0)]
                                        + TARGET_PI[sp, 1] * q_pi[pair_index(sp, 1)]
                                        for sp in (0, 1)
                                    ]
                                )
                            )
                        )
                    )
                )
            ),
            "V_pi_by_state": v_pi,
            "V_pi_state_difference": float(v_pi[0] - v_pi[1]),
            "target_mean_immediate_reward_by_state": r_bar,
            "nondegeneracy_argument": (
                "if V_pi(0) == V_pi(1) == v, the Bellman equation would force "
                "r_bar(s) = (1-gamma) v for both states, but r_bar = "
                "(0.625, 0.25); hence the two true state values differ"
            ),
            "note": (
                "audit-only population solution computed from the true MDP; "
                "never a network input; P and q_pi are auditor-only objects"
            ),
        },
        "data_bias": {
            "lhs_norm_inf_q_hat_minus_q_pi": data_bias_lhs,
            "rhs_audit_bound": data_bias_rhs,
            "rhs_definition": "||F0(q_pi)-q_pi||_inf / (1-c0), c0 = 0.85",
            "F0_q_pi_residual_inf": f0_qpi_resid,
            "margin_rhs_minus_lhs": data_bias_rhs - data_bias_lhs,
            "note": (
                "deterministic audit upper bound using ground truth; not a "
                "learner-computable statistical certificate; the data bias "
                "itself is reported as-is and may be nonzero or zero"
            ),
        },
        "q_f_inf": None,
        "q_f_inf_status": None,
    }
    if c_f < 1.0:
        q_f = solve_fixed_point(gf, bf)
        fixed_points["q_f_inf"] = {
            "value": q_f,
            "residual_inf": float(np.max(np.abs(finite_operator(q_f, fm) - q_f))),
            "justification": (
                f"c_f = ||Gf||_inf = {c_f:.12g} < 1: Gf is an inf-norm "
                "contraction, so Ff has a unique fixed point and iterates "
                "converge geometrically (Banach)"
            ),
        }
        fixed_points["q_f_inf_status"] = "available"
    else:
        fixed_points["q_f_inf_status"] = "unavailable"
        fixed_points["q_f_inf_unavailable_reason"] = (
            f"c_f = {c_f:.12g} >= 1: the contraction sufficient condition "
            "fails; this is not a divergence claim; fixed-point quantities "
            "are null per task section 5"
        )
    evidence["fixed_points"] = fixed_points

    # ---- literal network description ----------------------------------------
    net_desc_net = LiteralNetwork(rows, TARGET_PI, "finite", XI, ZETA, TAU)
    net_desc = {
        "n_tokens": net_desc_net.n_tokens,
        "null_token": net_desc_net.null_token,
        "prompt_H": net_desc_net.h.tolist(),
        "masks": {
            "stage1_allowed": net_desc_net.stage1.allowed.tolist(),
            "stage2_allowed": net_desc_net.stage2.allowed.tolist(),
            "stage4_allowed": net_desc_net.stage4.allowed.tolist(),
            "note": (
                "static role/position masks; context queries read memory, "
                "memory queries read context, inactive queries read the null "
                "token; no dependence on observed pair equality"
            ),
        },
        "stages": {
            "stage1_current_read": {
                "WQ": net_desc_net.stage1.wq,
                "WK": net_desc_net.stage1.wk,
                "WV": net_desc_net.stage1.wv,
                "WO": net_desc_net.stage1.wo,
                "scale": net_desc_net.stage1.scale,
            },
            "stage2_successor_read": {
                "WQ": net_desc_net.stage2.wq,
                "WK": net_desc_net.stage2.wk,
                "WV": net_desc_net.stage2.wv,
                "WO": net_desc_net.stage2.wo,
                "scale": net_desc_net.stage2.scale,
            },
            "stage4_writeback": {
                "WQ": net_desc_net.stage4.wq,
                "WK": net_desc_net.stage4.wk,
                "WV": net_desc_net.stage4.wv,
                "WO": net_desc_net.stage4.wo,
                "scale": net_desc_net.stage4.scale,
            },
        },
        "feedforward_maps": {
            "M3_residual": net_desc_net.m3,
            "M5_q_update": net_desc_net.m5,
            "P_reset_scratch": net_desc_net.p_reset,
        },
        "weights_depend_on": (
            "dimensions, frozen target_pi, gamma, alpha and frozen sharpness "
            "only; never on dynamic Q, sampled rewards, visit counts or the "
            "true transition matrix; rewards and observed transitions enter "
            "only as immutable prompt data"
        ),
        "no_visitation_frequency_multiplier": (
            "the writeback is the grouped mean (W0 rows are 1/n_x indicators, "
            "finite W rows are normalized softmax weights); no n_x/N or "
            "visitation-frequency factor multiplies the update"
        ),
    }
    evidence["literal_network"] = net_desc

    # ---- sequences ------------------------------------------------------------
    q = Q0.copy()
    direct_trace = [q.copy()]
    for _ in range(STEPS):
        q = direct_reference_step(q, rows, TARGET_PI)
        direct_trace.append(q.copy())

    net_e = LiteralNetwork(rows, TARGET_PI, "exact")
    exact_trace, _ = net_e.run(STEPS, Q0, capture_updates=())

    net_f = LiteralNetwork(rows, TARGET_PI, "finite", XI, ZETA, TAU)
    finite_trace, captured = net_f.run(STEPS, Q0, capture_updates=(1,))

    q = Q0.copy()
    scalar_trace = [q.copy()]
    for _ in range(STEPS):
        q = finite_scalar_step(q, rows, TARGET_PI, XI, ZETA, TAU)
        scalar_trace.append(q.copy())

    evidence["sequences"] = [
        {
            "id": "exact-direct",
            "route": "exact",
            "implementation": "direct grouped-mean reference (no attention)",
            "q_trace_k0_to_k64": np.array(direct_trace),
        },
        {
            "id": "exact-attention",
            "route": "exact",
            "implementation": "literal attention with declared equality masks",
            "q_trace_k0_to_k64": np.array(exact_trace),
        },
        {
            "id": "finite-literal",
            "route": "finite",
            "implementation": "literal fixed-weight softmax attention network",
            "q_trace_k0_to_k64": np.array(finite_trace),
        },
        {
            "id": "finite-scalar",
            "route": "finite",
            "implementation": "independently written scalar formula loops",
            "q_trace_k0_to_k64": np.array(scalar_trace),
        },
    ]
    evidence["first_step_projections"] = {
        "note": (
            "full literal projections (queries, keys, values, unmasked scores, "
            "normalized probabilities, outputs) of the finite network at "
            "update 1 from Q0, plus the pre-reset residual snapshot"
        ),
        "captured": to_jsonable(captured),
    }

    # ---- immutable-field / scratch integrity over the full run ---------------
    net_chk = LiteralNetwork(rows, TARGET_PI, "finite", XI, ZETA, TAU)
    net_chk.set_q(Q0)
    imm0 = net_chk.h[:, IMMUTABLE_FIELDS].copy()
    scratch_max = 0.0
    ctx_q_max = 0.0
    for _ in range(STEPS):
        net_chk.step()
        scratch_max = max(
            scratch_max, float(np.max(np.abs(net_chk.h[:, SCRATCH_FIELDS])))
        )
        ctx_q_max = max(ctx_q_max, float(np.max(np.abs(net_chk.h[4:, F_Q]))))
    imm_max = float(np.max(np.abs(net_chk.h[:, IMMUTABLE_FIELDS] - imm0)))
    evidence["network_integrity"] = {
        "immutable_fields_max_abs_change_over_64_updates": imm_max,
        "scratch_fields_max_abs_after_each_update": scratch_max,
        "context_tokens_q_field_max_abs": ctx_q_max,
        "note": (
            "immutable prompt fields (roles, identity one-hots, reward, logpi, "
            "const) must be bit-identical; scratch fields are zeroed by the "
            "fixed P_reset projection after each update; only the four memory "
            "Q fields carry dynamic state between updates"
        ),
    }

    # ---- per-step signed stage-error decomposition and perturbation bound -----
    c0m, s0m, w0m, r0m = em
    ccm, ssm, wwm, rrm = fm
    per_step = []
    e_bound = [0.0]
    margins = []
    for k in range(STEPS):
        qk = direct_trace[k]
        e_cur = -ALPHA * (wwm @ ((ccm - c0m) @ qk))
        e_suc = ALPHA * GAMMA * (wwm @ ((ssm - s0m) @ qk))
        e_wri = ALPHA * ((wwm - w0m) @ (rrm + GAMMA * (s0m @ qk) - c0m @ qk))
        delta = finite_operator(qk, fm) - exact_operator(qk, em)
        tele_res = float(np.max(np.abs((e_cur + e_suc + e_wri) - delta)))
        d_inf = float(np.max(np.abs(delta)))
        e_bound.append(c_f * e_bound[-1] + d_inf)
        actual = float(np.max(np.abs(finite_trace[k + 1] - direct_trace[k + 1])))
        margins.append(e_bound[-1] - actual)
        per_step.append(
            {
                "k": k,
                "e_current": e_cur,
                "e_successor": e_suc,
                "e_write": e_wri,
                "e_current_inf": float(np.max(np.abs(e_cur))),
                "e_successor_inf": float(np.max(np.abs(e_suc))),
                "e_write_inf": float(np.max(np.abs(e_wri))),
                "delta_inf": d_inf,
                "triangle_gap": float(
                    np.max(np.abs(e_cur))
                    + np.max(np.abs(e_suc))
                    + np.max(np.abs(e_wri))
                    - d_inf
                ),
                "telescoping_residual": tele_res,
                "E_kplus1": e_bound[-1],
                "actual_error_kplus1": actual,
                "bound_margin_kplus1": margins[-1],
            }
        )
    decomposition = {
        "definition": (
            "e_current = -alpha W (C-C0) q; e_successor = alpha gamma W (S-S0) q; "
            "e_write = alpha (W-W0)(r + gamma S0 q - C0 q); evaluated at "
            "q = q_exact,k; Ff(q)-F0(q) = e_current+e_successor+e_write exactly"
        ),
        "E_recursion": "E_0 = 0; E_(k+1) = c_f E_k + ||Ff(q_exact,k)-F0(q_exact,k)||_inf",
        "per_step": per_step,
        "max_telescoping_residual": float(
            max(p["telescoping_residual"] for p in per_step)
        ),
        "min_bound_margin": float(min(margins)),
        "E_64": e_bound[-1],
        "actual_error_64": float(
            np.max(np.abs(finite_trace[STEPS] - direct_trace[STEPS]))
        ),
    }
    evidence["stage_error_decomposition"] = decomposition

    # ---- exact contraction trajectory bound ------------------------------------
    e0 = float(np.max(np.abs(Q0 - q_hat)))
    exact_contraction = []
    for k in range(STEPS + 1):
        lhs = float(np.max(np.abs(direct_trace[k] - q_hat)))
        rhs = (C0_BOUND**k) * e0
        exact_contraction.append(
            {"k": k, "lhs": lhs, "rhs": rhs, "margin": rhs - lhs}
        )
    evidence["exact_contraction_bound"] = {
        "statement": "||q_exact,k - q_hat||_inf <= 0.85^k ||Q0 - q_hat||_inf",
        "min_margin": float(min(row["margin"] for row in exact_contraction)),
        "per_k": exact_contraction,
    }

    # ---- finite fixed-point bounds and three-term decomposition (if c_f < 1) ---
    if c_f < 1.0:
        q_f = np.array(fixed_points["q_f_inf"]["value"])
        gap = float(np.max(np.abs(finite_operator(q_hat, fm) - q_hat)))
        ss_rhs = gap / (1.0 - c_f)
        ss_lhs = float(np.max(np.abs(q_f - q_hat)))
        traj = []
        for k in range(STEPS + 1):
            lhs = float(np.max(np.abs(finite_trace[k] - q_f)))
            rhs = (c_f**k) * float(np.max(np.abs(Q0 - q_f)))
            traj.append({"k": k, "lhs": lhs, "rhs": rhs, "margin": rhs - lhs})
        decomp = []
        for k in range(STEPS + 1):
            t1 = finite_trace[k] - q_f
            t2 = q_f - q_hat
            t3 = q_hat - q_pi
            total = finite_trace[k] - q_pi
            ident = float(np.max(np.abs(total - (t1 + t2 + t3))))
            norm_sum = float(
                np.max(np.abs(t1)) + np.max(np.abs(t2)) + np.max(np.abs(t3))
            )
            decomp.append(
                {
                    "k": k,
                    "transient_inf": float(np.max(np.abs(t1))),
                    "finite_fixed_point_shift_inf": float(np.max(np.abs(t2))),
                    "data_bias_inf": float(np.max(np.abs(t3))),
                    "total_error_inf": float(np.max(np.abs(total))),
                    "identity_residual": ident,
                    "norm_triangle_margin": norm_sum - float(np.max(np.abs(total))),
                }
            )
        evidence["finite_fixed_point_analysis"] = {
            "status": "available",
            "steady_state_bound": {
                "statement": "||q_f,inf - q_hat|| <= ||Ff(q_hat)-q_hat||/(1-c_f)",
                "lhs": ss_lhs,
                "rhs": ss_rhs,
                "margin": ss_rhs - ss_lhs,
            },
            "trajectory_bound": {
                "statement": "||q_finite,k - q_f,inf|| <= c_f^k ||Q0 - q_f,inf||",
                "min_margin": float(min(row["margin"] for row in traj)),
                "per_k": traj,
            },
            "signed_decomposition_vs_q_pi": {
                "statement": (
                    "q_finite,k - q_pi = (q_finite,k - q_f,inf) + "
                    "(q_f,inf - q_hat) + (q_hat - q_pi); the signed vector "
                    "identity is exact; the norm sum is only an upper bound"
                ),
                "max_identity_residual": float(
                    max(row["identity_residual"] for row in decomp)
                ),
                "min_norm_triangle_margin": float(
                    min(row["norm_triangle_margin"] for row in decomp)
                ),
                "per_k": decomp,
            },
        }
    else:
        evidence["finite_fixed_point_analysis"] = {
            "status": "unavailable",
            "reason": fixed_points["q_f_inf_unavailable_reason"],
            "steady_state_bound": None,
            "trajectory_bound": None,
            "signed_decomposition_vs_q_pi": None,
        }

    # ---- witness self-checks ---------------------------------------------------
    # One-step network probes: the four standard basis vectors plus Q0 = 0
    # only. Task section 4 makes q_pi audit-only and never a network input;
    # q_hat and arbitrary non-basis vectors are likewise not probed through
    # the network. q_hat and q_pi remain in use below only for the audit-only
    # population, data-bias and decomposition mathematics (direct formula
    # application, not network inputs).
    probes = [np.eye(4)[i] for i in range(4)] + [
        np.zeros(4),
    ]
    h2a_one = 0.0
    h2b_one = 0.0
    for qp in probes:
        net_ep = LiteralNetwork(rows, TARGET_PI, "exact")
        net_ep.set_q(qp)
        net_ep.step()
        h2a_one = max(
            h2a_one,
            float(
                np.max(np.abs(net_ep.current_q() - direct_reference_step(qp, rows, TARGET_PI)))
            ),
        )
        net_fp = LiteralNetwork(rows, TARGET_PI, "finite", XI, ZETA, TAU)
        net_fp.set_q(qp)
        net_fp.step()
        h2b_one = max(
            h2b_one,
            float(
                np.max(
                    np.abs(
                        net_fp.current_q()
                        - finite_scalar_step(qp, rows, TARGET_PI, XI, ZETA, TAU)
                    )
                )
            ),
        )

    h2a_trace = 0.0
    h2b_trace = 0.0
    for k in range(STEPS + 1):
        for i in range(4):
            a, b = direct_trace[k][i], exact_trace[k][i]
            h2a_trace = max(h2a_trace, abs(a - b) / repeated_tol(a, b))
            a, b = finite_trace[k][i], scalar_trace[k][i]
            h2b_trace = max(h2b_trace, abs(a - b) / repeated_tol(a, b))

    # affine reconstruction on the four standard basis vectors
    affine_max = 0.0
    f0_zero = exact_operator(np.zeros(4), em)
    ff_zero = finite_operator(np.zeros(4), fm)
    affine_max = max(affine_max, float(np.max(np.abs(f0_zero - b0))))
    affine_max = max(affine_max, float(np.max(np.abs(ff_zero - bf))))
    for i in range(4):
        e_i = np.eye(4)[i]
        affine_max = max(
            affine_max,
            float(np.max(np.abs((exact_operator(e_i, em) - f0_zero) - g0[:, i]))),
            float(np.max(np.abs((finite_operator(e_i, fm) - ff_zero) - gf[:, i]))),
        )

    # policy preservation: successor conditional equals target_pi, not behavior
    policy_err = 0.0
    target_vs_behavior = float("inf")
    for t in range(len(rows)):
        for u in (0, 1):
            denom = sum(ssm[t, pair_index(u, b)] for b in (0, 1))
            for b in (0, 1):
                cond = ssm[t, pair_index(u, b)] / denom
                policy_err = max(policy_err, abs(cond - TARGET_PI[u, b]))
                target_vs_behavior = min(
                    target_vs_behavior, abs(cond - BEHAVIOR_PI[u, b])
                )

    # grouped mean: W0 row weights are exactly 1/n_x on matches (no frequency factor)
    grouped_mean_max = 0.0
    for x in range(4):
        for t in range(len(rows)):
            if pair_index(rows[t][0], rows[t][1]) == x:
                grouped_mean_max = max(
                    grouped_mean_max, abs(w0m[x, t] * counts[x] - 1.0)
                )

    # rejection checks
    rejections = []
    for label, fn in (
        ("malformed_pi_shape", lambda: validate_policy([[0.5, 0.5]])),
        ("malformed_pi_negative", lambda: validate_policy([[1.25, -0.25],
                                                          [0.25, 0.75]])),
        ("malformed_pi_unnormalized", lambda: validate_policy([[0.8, 0.25],
                                                               [0.25, 0.75]])),
        ("malformed_pi_zero_entry", lambda: validate_policy([[1.0, 0.0],
                                                             [0.25, 0.75]])),
        ("nonfinite_q", lambda: validate_q([0.0, float("nan"), 1.0, 2.0])),
        ("nonfinite_reward", lambda: validate_rows([(0, 0, float("inf"), 0)])),
        ("bad_state", lambda: validate_rows([(2, 0, 1.0, 0)])),
        ("empty_batch", lambda: validate_rows([])),
    ):
        try:
            fn()
            rejections.append({"check": label, "rejected": False})
            failures.append(f"rejection check failed: {label}")
        except InputReject:
            rejections.append({"check": label, "rejected": True})

    residual_capture_err = 0.0
    for upd in captured.values():
        residual_capture_err = max(
            residual_capture_err,
            float(upd["captured_residual_vs_stage4_value_read_max_abs"]),
        )

    fin_analysis = evidence["finite_fixed_point_analysis"]
    checks = {
        "H1_coverage": {
            "counts": counts,
            "min_count": min(counts),
            "pass": bool(min(counts) >= 1),
        },
        "sampling_integrity": {
            "total_rng_draws": draw_count,
            "expected_draws": 2 * BATCH_LENGTH,
            "threshold_reconstruction_max_abs_error": recon_max,
            "pass": bool(draw_count == 2 * BATCH_LENGTH and recon_max == 0.0),
        },
        "H2a_direct_equals_exact_attention": {
            "max_one_step_abs_error": h2a_one,
            "max_trace_scaled_error": h2a_trace,
            "pass": bool(h2a_one <= TOL_ONE_STEP and h2a_trace <= 1.0),
        },
        "H2b_literal_finite_equals_scalar": {
            "max_one_step_abs_error": h2b_one,
            "max_trace_scaled_error": h2b_trace,
            "pass": bool(h2b_one <= TOL_ONE_STEP and h2b_trace <= 1.0),
        },
        "H3_stage_errors_and_bounds": {
            "max_telescoping_residual": decomposition["max_telescoping_residual"],
            "min_perturbation_bound_margin": decomposition["min_bound_margin"],
            "pass": bool(
                decomposition["max_telescoping_residual"] <= TOL_ONE_STEP
                and decomposition["min_bound_margin"] >= -TOL_ONE_STEP
            ),
        },
        "H4_contraction": {
            "norm_inf_G0": norm_g0,
            "c0_proved": C0_BOUND,
            "min_exact_contraction_margin": evidence["exact_contraction_bound"][
                "min_margin"
            ],
            "c_f": c_f,
            "c_f_lt_1": bool(c_f < 1.0),
            "pass": bool(
                abs(norm_g0 - C0_BOUND) <= TOL_ONE_STEP
                and evidence["exact_contraction_bound"]["min_margin"] >= -TOL_ONE_STEP
            ),
        },
        "H5_state_values_and_data_bias": {
            "V_pi_by_state": v_pi,
            "abs_state_difference": float(abs(v_pi[0] - v_pi[1])),
            "data_bias_norm_inf": data_bias_lhs,
            "audit_bound_rhs": data_bias_rhs,
            "audit_bound_margin": data_bias_rhs - data_bias_lhs,
            "note": (
                "the two true state values must differ (proved); the data bias "
                "magnitude is a reported metric, not a success condition"
            ),
            "pass": bool(
                abs(v_pi[0] - v_pi[1]) > 1e-6
                and data_bias_lhs <= data_bias_rhs + repeated_tol(
                    data_bias_lhs, data_bias_rhs
                )
            ),
        },
        "affine_basis_reconstruction": {
            "max_abs_error": affine_max,
            "pass": bool(affine_max <= TOL_ONE_STEP),
        },
        "policy_preservation": {
            "successor_conditional_vs_target_pi_max_error": policy_err,
            "min_abs_distance_conditional_vs_behavior_0p5": target_vs_behavior,
            "pass": bool(
                policy_err <= TOL_ONE_STEP and target_vs_behavior > 1e-3
            ),
        },
        "grouped_mean_no_frequency_multiplier": {
            "max_abs_w0_times_n_x_minus_1": grouped_mean_max,
            "pass": bool(grouped_mean_max <= TOL_ONE_STEP),
        },
        "network_integrity": {
            "immutable_fields_max_abs_change": imm_max,
            "scratch_max_abs_after_updates": scratch_max,
            "context_q_max_abs": ctx_q_max,
            "residual_capture_max_abs_error": residual_capture_err,
            "pass": bool(
                imm_max == 0.0
                and scratch_max == 0.0
                and ctx_q_max == 0.0
                and residual_capture_err <= TOL_ONE_STEP
            ),
        },
        "input_rejection": {
            "cases": rejections,
            "pass": all(r["rejected"] for r in rejections),
        },
    }
    if c_f < 1.0:
        checks["finite_fixed_point_bounds"] = {
            "q_f_residual_inf": fixed_points["q_f_inf"]["residual_inf"],
            "steady_state_margin": fin_analysis["steady_state_bound"]["margin"],
            "trajectory_min_margin": fin_analysis["trajectory_bound"]["min_margin"],
            "decomposition_max_identity_residual": fin_analysis[
                "signed_decomposition_vs_q_pi"
            ]["max_identity_residual"],
            "min_norm_triangle_margin": fin_analysis[
                "signed_decomposition_vs_q_pi"
            ]["min_norm_triangle_margin"],
            "pass": bool(
                fixed_points["q_f_inf"]["residual_inf"]
                <= repeated_tol(
                    fixed_points["q_f_inf"]["residual_inf"], 0.0
                )
                and fin_analysis["steady_state_bound"]["margin"]
                >= -repeated_tol(
                    fin_analysis["steady_state_bound"]["lhs"],
                    fin_analysis["steady_state_bound"]["rhs"],
                )
                and fin_analysis["trajectory_bound"]["min_margin"] >= -TOL_REPEATED
                and fin_analysis["signed_decomposition_vs_q_pi"][
                    "max_identity_residual"
                ]
                <= TOL_ONE_STEP
                and fin_analysis["signed_decomposition_vs_q_pi"][
                    "min_norm_triangle_margin"
                ]
                >= -TOL_ONE_STEP
            ),
        }
    else:
        checks["finite_fixed_point_bounds"] = {
            "status": "unavailable",
            "reason": fixed_points["q_f_inf_unavailable_reason"],
            "pass": True,
            "note": "not applicable; null per task section 5, not a divergence claim",
        }
    evidence["h_checks"] = checks
    evidence["failed_runs"] = failures
    evidence["witness_self_verdict"] = (
        "PASS"
        if all(c.get("pass", False) for c in checks.values()) and not failures
        else "FAIL"
    )
    return evidence


def main():
    evidence = run_protocol()
    text = json.dumps(to_jsonable(evidence), allow_nan=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
