"""FP-INDEP-001 / claude: independent re-implementation of the full pipeline.

SELF-CONTAINED by construction: this module imports ONLY numpy and the standard library.
Every operator is written from the task sheet's spec appendix (§10 of
docs/research_tasks/FP-INDEP-001.md), not from any baseline module. Baseline modules
(fixed_policy_*, fp_*, evaluate_*, model.py, mdps.py, verify_*) are never imported here.

Frozen protocol (v4 of the task sheet): original family 4 states x 3 actions,
gamma=0.70, R*=1.5, GAP_BONUS=0.5, PI_MIN=0.15; mixing in {0.08, 0.5}, task_index 12..23;
algorithms expected_exact/expected_finite; arms conj_n16k / conj_n64k / perstate_n16k;
producers numpy (float64) and network (float32 spec operator, NOT model.py);
K=12, delta_total=0.05, delta_step=delta_total/K; certification batches 65536 chains x 64
in 32768 chunks, seed vector [20260911, 77531, round(100*mixing), task_index, step];
first-visit retention with min_visits=2000; frozen envelope certificate (value_bound=5,
envelope=10, y_range=20, delta_dir=delta_step/(2*12), MP_CONSTANT=7/3, ddof=1);
eta grid (1,.5,.2,.1,.05,.02,.01); trajectories stop at their first non-emission.

    python evaluate.py --output-dir <results/FP-INDEP-001/claude/formal>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

TASK_ID = "FP-INDEP-001"
ACTOR = "claude"
BASELINE = "1da7162abb662142332eaf4493209f73f41fb5c4"
TASK_SHEET = Path(__file__).resolve().parents[4] / "docs" / "research_tasks" / "FP-INDEP-001.md"

# §10.1 constants, frozen
S, A, D = 4, 3, 12
GAMMA = 0.70
REWARD_BOUND = 1.5
GAP_BONUS = 0.5
PI_MIN = 0.15
SEED = 20260911
SALT = 77531
TRAIN_LENGTH = 65536
CERT_CHAINS = 65536
CHAIN_LENGTH = 64
CHUNK = 32768
MIN_VISITS = 2000
DELTA_TOTAL = 0.05
K = 12
DELTA_STEP = DELTA_TOTAL / K
ETA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
ALPHA = 0.65
LAYERS = 160
ZETA = XI = TAU = 8.0
VALUE_BOUND = 5.0
# §10.8 frozen envelope
CERT_VALUE_BOUND = REWARD_BOUND / (1.0 - GAMMA)          # 5
CERT_ENVELOPE = REWARD_BOUND + GAMMA * CERT_VALUE_BOUND + CERT_VALUE_BOUND  # 10
Y_RANGE = 2.0 * CERT_ENVELOPE                            # 20
MP_CONSTANT = 7.0 / 3.0
DELTA_DIR = DELTA_STEP / (2.0 * D)
NUMERIC_PAUSE = 1e-10        # §6: |margin| at or below this pauses the run
MIXINGS = (0.08, 0.5)
TASK_INDICES = tuple(range(12, 24))
ALGORITHMS = ("expected_exact", "expected_finite")
ARMS = ("conj_n16k", "conj_n64k", "perstate_n16k")
PRODUCERS = ("numpy", "network")
ARM_CHAINS = {"conj_n16k": 16384, "conj_n64k": 65536, "perstate_n16k": 16384}
REASON_ORDER = ("divergence_guard_triggered", "heldout_pair_support_missing",
                "numerical_nonfinite")


def sha256_lf(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    except OSError:
        return None


def h(*arrays) -> str:
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        m.update(str(a.dtype.str).encode())
        m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# §10.2 environment
# --------------------------------------------------------------------------- #
def build_env(mixing: float, task_index: int):
    rng = np.random.default_rng([SEED, int(round(100 * mixing)), int(task_index)])
    P0 = rng.dirichlet(np.ones(S), size=(S, A)).astype(np.float32)
    R = rng.uniform(-1.0, 1.0, size=(S, A, S)).astype(np.float32)
    p0 = rng.dirichlet(np.ones(S)).astype(np.float32)
    P0 = np.asarray(P0, dtype=np.float64)
    P0 /= P0.sum(axis=2, keepdims=True)
    sticky = np.zeros_like(P0)
    for s in range(S):
        sticky[s, :, s] = 1.0
    P = (1.0 - mixing) * sticky + mixing * P0
    R = np.asarray(R, dtype=np.float64)
    R[:, 0, :] += GAP_BONUS
    policy = np.full((S, A), PI_MIN, dtype=np.float64)
    preferred = rng.integers(0, A, size=S)
    policy[np.arange(S), preferred] = 1.0 - (A - 1) * PI_MIN
    return {"P": P, "R": R, "p0": p0, "nS": S, "nA": A, "gamma": GAMMA}, policy, rng


# --------------------------------------------------------------------------- #
# §10.3 exact quantities (audit channel and starting quantities)
# --------------------------------------------------------------------------- #
def exact_quantities(mdp, policy):
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    gamma = float(mdp["gamma"])
    p_pi = np.einsum("sa,san->sn", policy, P)
    r_sa = np.sum(P * R, axis=2)
    r_pi = np.sum(policy * r_sa, axis=1)
    v_pi = np.linalg.solve(np.eye(S) - gamma * p_pi, r_pi)
    q_pi = r_sa + gamma * np.einsum("san,n->sa", P, v_pi)
    system = p_pi.T - np.eye(S)
    rhs = np.zeros(S)
    system[-1] = 1.0
    rhs[-1] = 1.0
    mu = np.linalg.solve(system, rhs)
    mu = np.maximum(mu, 0.0)
    mu = mu / mu.sum()
    return {"v_pi": v_pi, "q_pi": q_pi, "mu": mu, "r_sa": r_sa, "p_pi": p_pi}


def _value_of(policy, mdp):
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    r_sa = np.einsum("sap,sap->sa", P, R)
    p_pi = np.einsum("sa,sap->sp", policy, P)
    r_pi = np.einsum("sa,sa->s", policy, r_sa)
    return np.linalg.solve(np.eye(S) - GAMMA * p_pi, r_pi)


def optimal_values(mdp):
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    r_sa = np.einsum("sap,sap->sa", P, R)
    greedy = np.full((S, A), 1.0 / A)
    for _ in range(1000):
        q = r_sa + GAMMA * np.einsum("sap,p->sa", P, _value_of(greedy, mdp))
        best = np.argmax(q, axis=1)     # numpy convention: lowest index of the max
        new = np.zeros_like(greedy)
        new[np.arange(S), best] = 1.0
        if np.array_equal(new, greedy):
            break
        greedy = new
    return _value_of(greedy, mdp), greedy


# --------------------------------------------------------------------------- #
# §10.4 training batch (float32 reward storage, per the spec's dtype note)
# --------------------------------------------------------------------------- #
def training_batch(mdp, policy, mu, rng):
    start = int(rng.choice(S, p=mu))
    P = mdp["P"]
    R = mdp["R"]
    n = TRAIN_LENGTH
    states = np.zeros(n + 1, dtype=np.int64)
    actions = np.zeros(n + 1, dtype=np.int64)
    rewards32 = np.zeros(n + 1, dtype=np.float32)
    states[0] = start
    for t in range(n):
        a = int(rng.choice(A, p=policy[states[t]]))
        actions[t] = a
        following = int(rng.choice(S, p=P[states[t], a]))
        states[t + 1] = following
        rewards32[t + 1] = R[states[t], a, following]   # float64 value stored into float32
    actions[n] = int(rng.choice(A, p=policy[states[n]]))
    return {
        "states": states[:n],
        "actions": actions[:n],
        "rewards": np.asarray(rewards32, dtype=np.float64)[1 : n + 1],
        "next_states": states[1 : n + 1],
        "next_actions": actions[1 : n + 1],
    }


# --------------------------------------------------------------------------- #
# §10.5 certification batch (pinned call sequence)
# --------------------------------------------------------------------------- #
def certification_batch(mdp, policy, mu, seed_parts, chains=CERT_CHAINS,
                        chain_length=CHAIN_LENGTH):
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    rng = np.random.default_rng(seed_parts)
    policy_cdf = np.cumsum(policy, axis=1)
    policy_cdf[:, -1] = 1.0
    total = chains * chain_length
    states = np.empty(total, dtype=np.int32)
    actions = np.empty(total, dtype=np.int32)
    rewards = np.empty(total, dtype=np.float64)
    next_states = np.empty(total, dtype=np.int32)
    done = 0
    while done < chains:
        size = min(CHUNK, chains - done)
        current = rng.choice(S, size=size, p=mu).astype(np.int32)
        for step in range(chain_length):
            uniform = rng.random(size)
            chosen = (uniform[:, None] > policy_cdf[current]).sum(axis=1)
            np.clip(chosen, 0, A - 1, out=chosen)
            following_uniform = rng.random(size)
            rows = P[current, chosen]
            following = (following_uniform[:, None] > np.cumsum(rows, axis=1)).sum(axis=1)
            np.clip(following, 0, S - 1, out=following)
            following = following.astype(np.int32)
            base = (done + np.arange(size)) * chain_length + step
            states[base] = current
            actions[base] = chosen.astype(np.int32)
            rewards[base] = R[current, chosen, following]
            next_states[base] = following
            current = following
        done += size
    return {"states": states, "actions": actions, "rewards": rewards,
            "next_states": next_states, "next_actions": actions}


# --------------------------------------------------------------------------- #
# §10.6 first-visit retention
# --------------------------------------------------------------------------- #
def first_visit(batch, chain_length, max_chains=None):
    flat = np.asarray(batch["states"], dtype=np.int64) * A + np.asarray(batch["actions"],
                                                                        dtype=np.int64)
    total = flat.size
    if total % chain_length != 0:
        raise ValueError("batch length is not a multiple of chain_length")
    n_chains = total // chain_length
    if max_chains is not None:
        n_chains = min(n_chains, int(max_chains))
    grid = flat[: n_chains * chain_length].reshape(n_chains, chain_length)
    keep: list[int] = []
    counts = np.zeros(D, dtype=np.int64)
    for pair in range(D):
        mask = grid == pair
        has = mask.any(axis=1)
        counts[pair] = int(has.sum())
        if counts[pair]:
            first_pos = mask[has].argmax(axis=1)
            chain_ids = np.flatnonzero(has)
            keep.extend((chain_ids * chain_length + first_pos).tolist())
    keep_arr = np.sort(np.asarray(keep, dtype=np.int64))
    reduced = {key: np.asarray(value)[keep_arr] for key, value in batch.items()}
    return reduced, counts


# --------------------------------------------------------------------------- #
# §10.7 residuals and §10.8 frozen envelope certificate
# --------------------------------------------------------------------------- #
def residuals_for(q_hat, policy, batch):
    q_hat = np.asarray(q_hat, dtype=np.float64)
    policy = np.asarray(policy, dtype=np.float64)
    st = np.asarray(batch["states"], dtype=np.int64)
    ac = np.asarray(batch["actions"], dtype=np.int64)
    rw = np.asarray(batch["rewards"], dtype=np.float64)
    ns = np.asarray(batch["next_states"], dtype=np.int64)
    successor = (policy[ns] * q_hat[ns]).sum(axis=1)
    return rw + GAMMA * successor - q_hat[st, ac]


def certificate(q_hat, policy, reduced, counts):
    """Frozen-envelope MP certificate. Returns (e_q or None, ordered_reasons, detail)."""
    q_hat = np.asarray(q_hat, dtype=np.float64)
    reasons: list[str] = []
    if not np.all(np.isfinite(q_hat)):
        reasons.append("numerical_nonfinite")
    elif float(np.max(np.abs(q_hat))) > CERT_VALUE_BOUND:
        reasons.append("divergence_guard_triggered")
    if not reasons:
        for pair in range(D):
            if counts[pair] < MIN_VISITS:
                reasons.append("heldout_pair_support_missing")
                break
    ordered = sorted(set(reasons), key=REASON_ORDER.index) if reasons else []
    if ordered:
        return None, ordered, None
    resid = residuals_for(q_hat, policy, reduced)
    flat = np.asarray(reduced["states"], dtype=np.int64) * A + np.asarray(
        reduced["actions"], dtype=np.int64)
    log_term = math.log(2.0 / DELTA_DIR)
    means = np.zeros(D)
    radii = np.zeros(D)
    variances = np.zeros(D)
    for pair in range(D):
        y = resid[flat == pair]
        n = int(counts[pair])
        v = float(np.var(y, ddof=1))
        means[pair] = float(np.mean(y))
        variances[pair] = v
        radii[pair] = math.sqrt(2.0 * max(v, 0.0) * log_term / n) + (
            MP_CONSTANT * Y_RANGE * log_term / max(n - 1, 1))
    eps = np.abs(means) + radii
    e_q = float(np.max(eps)) / (1.0 - GAMMA)
    return e_q, [], {"means": means, "vars": variances, "radii": radii, "eps": eps}


# --------------------------------------------------------------------------- #
# §10.9 decision rules
# --------------------------------------------------------------------------- #
def relative_softmax_candidate(policy, q_hat, eta):
    policy = np.asarray(policy, dtype=np.float64)
    q_hat = np.asarray(q_hat, dtype=np.float64)
    logits = np.log(policy) + float(eta) * q_hat
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def conjunctive_decision(policy, q_hat, e_q):
    for eta in ETA_GRID:
        cand = relative_softmax_candidate(policy, q_hat, eta)
        delta = cand - policy
        lb = (delta * q_hat).sum(axis=1) - e_q * np.abs(delta).sum(axis=1)
        if float(lb.min()) > 0.0:
            return {"emitted": True, "policy_plus": cand, "eta_selected": float(eta),
                    "lb_by_state": lb, "states_updated": S}
    return {"emitted": False, "policy_plus": policy.copy(), "eta_selected": None,
            "lb_by_state": np.zeros(S), "states_updated": 0}


def perstate_decision(policy, q_hat, e_q):
    new_policy = policy.copy()
    lb_by_state = np.zeros(S)
    etas: list[Any] = []
    updated = 0
    for s in range(S):
        chosen = None
        chosen_lb = 0.0
        for eta in ETA_GRID:
            cand = relative_softmax_candidate(policy, q_hat, eta)
            delta_row = cand[s] - policy[s]
            lb = float((delta_row * q_hat[s]).sum()) - e_q * float(np.abs(delta_row).sum())
            if lb > 0.0:
                chosen = float(eta)
                chosen_lb = lb
                new_policy[s] = cand[s]
                break
        etas.append(chosen)
        lb_by_state[s] = chosen_lb
        if chosen is not None:
            updated += 1
    return {"emitted": updated > 0, "policy_plus": new_policy, "eta_selected": etas,
            "lb_by_state": lb_by_state, "states_updated": updated}


# --------------------------------------------------------------------------- #
# §10.10 producers
# --------------------------------------------------------------------------- #
def array_exact(policy, train):
    """§10.10.A: synchronous grouped Expected SARSA, exact action expectation."""
    q = np.zeros((S, A), dtype=np.float64)
    st = np.asarray(train["states"], np.int64)
    ac = np.asarray(train["actions"], np.int64)
    rw = np.asarray(train["rewards"], np.float64)
    ns = np.asarray(train["next_states"], np.int64)
    current = st * A + ac
    counts = np.bincount(current, minlength=D).astype(np.float64)
    policy = np.asarray(policy, dtype=np.float64)
    for _ in range(LAYERS):
        qbar = (policy[ns] * q[ns]).sum(axis=1)
        residuals = rw + GAMMA * qbar - q[st, ac]
        accumulated = np.bincount(current, weights=residuals, minlength=D)
        update = np.zeros(D)
        visited = counts > 0
        update[visited] = ALPHA * accumulated[visited] / counts[visited]
        q = q + update.reshape(S, A)
        if not np.all(np.isfinite(q)) or float(np.max(np.abs(q))) > VALUE_BOUND:
            raise RuntimeError("producer divergence: protocol anomaly, stop the trajectory")
    return q


def array_finite(policy, train):
    """§10.10.B: finite-logit grouped Expected SARSA (full-support, no gate)."""
    q = np.zeros((S, A), dtype=np.float64)
    st = np.asarray(train["states"], np.int64)
    ac = np.asarray(train["actions"], np.int64)
    rw = np.asarray(train["rewards"], np.float64)
    ns = np.asarray(train["next_states"], np.int64)
    current = st * A + ac
    m = float(st.size)
    policy = np.asarray(policy, dtype=np.float64)
    log_pi = np.log(policy).reshape(-1)
    e_tau = math.exp(TAU)
    e_xi = math.exp(XI)
    for _ in range(LAYERS):
        flat = q.reshape(-1)
        # read: read_x = (e^xi * q_x + (sum_y q_y - q_x)) / (e^xi + D - 1)
        total = float(flat.sum())
        reads_flat = (e_xi * flat + (total - flat)) / (e_xi + D - 1.0)
        reads = reads_flat[current]
        # successor: per next state s, softmax over tokens with score zeta*1{u=s}+log pi
        # (the scores depend only on the next state, so precompute S of them per layer)
        token_state = np.arange(D) // A
        scores = log_pi[None, :] + ZETA * (token_state[None, :] == np.arange(S)[:, None])
        scores -= scores.max(axis=1, keepdims=True)
        w = np.exp(scores)
        w /= w.sum(axis=1, keepdims=True)
        succ_by_state = w @ flat                       # (S,)
        successor = succ_by_state[ns]
        residuals = rw + GAMMA * successor - reads
        # write-back
        counts = np.bincount(current, minlength=D).astype(np.float64)
        sums = np.bincount(current, weights=residuals, minlength=D)
        total_r = float(residuals.sum())
        denom = counts * e_tau + (m - counts)
        update = ALPHA * (e_tau * sums + (total_r - sums)) / denom
        q = q + update.reshape(S, A)
        if not np.all(np.isfinite(q)) or float(np.max(np.abs(q))) > VALUE_BOUND:
            raise RuntimeError("producer divergence: protocol anomaly, stop the trajectory")
    return q


def network_masked(policy, train):
    """§10.10.C: masked exact attention operator, float32, numpy implementation."""
    q = np.zeros((S, A), dtype=np.float32)
    st = np.asarray(train["states"], np.int64)
    ac = np.asarray(train["actions"], np.int64)
    ns = np.asarray(train["next_states"], np.int64)
    rw = np.asarray(train["rewards"], dtype=np.float32)
    pi = np.asarray(policy, dtype=np.float32)
    pair_axis = np.arange(D)
    for _ in range(LAYERS):
        qf = q.reshape(-1)
        cp = st * A + ac
        current_q = qf[cp]                                # exact singleton read
        # action-expectation head: attention[t,(u,b)] = 1{u=ns_t} * pi[b|u]
        qbar = (pi[ns] * qf.reshape(S, A)[ns]).sum(axis=1)
        residuals = rw + np.float32(GAMMA) * qbar - current_q
        # write-back over sources (transitions + null), softmax on the source axis:
        # visited pair -> uniform mean over its matched transitions; unvisited -> 0.
        counts = np.bincount(cp, minlength=D).astype(np.float32)
        sums = np.bincount(cp, weights=residuals, minlength=D)
        update = np.zeros(D, dtype=np.float32)
        visited = counts > 0
        update[visited] = np.float32(ALPHA) * sums[visited] / counts[visited]
        q = (qf + update).reshape(S, A)
        if not np.all(np.isfinite(q)) or float(np.max(np.abs(q))) > VALUE_BOUND:
            raise RuntimeError("producer divergence: protocol anomaly, stop the trajectory")
    return q.astype(np.float64)


def network_finite(policy, train):
    """§10.10.D: finite full-support attention operator, float32, numpy implementation."""
    q = np.zeros((S, A), dtype=np.float32)
    st = np.asarray(train["states"], np.int64)
    ac = np.asarray(train["actions"], np.int64)
    ns = np.asarray(train["next_states"], np.int64)
    rw = np.asarray(train["rewards"], dtype=np.float32)
    pi = np.asarray(policy, dtype=np.float32)
    log_pi = np.log(pi).reshape(-1)
    token_state = np.arange(D) // A
    pair_axis = np.arange(D)
    e_tau = np.float32(math.exp(TAU))
    m = st.size
    for _ in range(LAYERS):
        qf = q.reshape(-1)
        cp = st * A + ac
        # successor: softmax over tokens with score zeta*1{u=ns_t} + log pi(token)
        succ_mask = (token_state[None, :] == ns[:, None]).astype(np.float32)
        succ_scores = log_pi[None, :] + np.float32(ZETA) * succ_mask
        succ_scores -= succ_scores.max(axis=1, keepdims=True)
        w = np.exp(succ_scores)
        successor = (w * qf[None, :]).sum(axis=1) / w.sum(axis=1)
        # read: score xi*1{y = cp_t}
        read_mask = (pair_axis[None, :] == cp[:, None])
        read_scores = np.float32(XI) * read_mask.astype(np.float32)
        read_scores -= read_scores.max(axis=1, keepdims=True)
        wr = np.exp(read_scores)
        read = (wr * qf[None, :]).sum(axis=1) / wr.sum(axis=1)
        residuals = rw + np.float32(GAMMA) * successor - read
        # write-back: w_t(x) = exp(tau*1{cp_t=x}) / (N_x e^tau + (m - N_x))
        counts = np.bincount(cp, minlength=D).astype(np.float32)
        match = (cp[:, None] == np.arange(D)[None, :])
        numerator = np.where(match, e_tau, np.float32(1.0))
        denominator = counts * e_tau + (np.float32(m) - counts)
        write_attention = numerator / denominator[None, :]
        update = np.float32(ALPHA) * (write_attention.astype(np.float32).T
                                      @ residuals.astype(np.float32))
        q = (qf + update).reshape(S, A)
        if not np.all(np.isfinite(q)) or float(np.max(np.abs(q))) > VALUE_BOUND:
            raise RuntimeError("producer divergence: protocol anomaly, stop the trajectory")
    return q.astype(np.float64)


def produce(algorithm, producer, policy, train):
    if producer == "numpy":
        return array_exact(policy, train) if algorithm == "expected_exact" else array_finite(policy, train)
    return network_masked(policy, train) if algorithm == "expected_exact" else network_finite(policy, train)


# --------------------------------------------------------------------------- #
# main matrix
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke only: first N task indices")
    args = ap.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke = bool(args.limit)

    started_utc = datetime.now(timezone.utc).isoformat()
    t_start = time.time()
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    batch_hashes: dict[str, str] = {}
    unique_batches: set[tuple] = set()
    items_unique = 0
    net_forwards = 0
    anomalies: list[dict] = []

    idx_list = TASK_INDICES[: args.limit] if args.limit else TASK_INDICES
    for mixing in MIXINGS:
        for task_index in idx_list:
            t0 = time.time()
            mdp, behaviour, rng = build_env(mixing, task_index)
            eq0 = exact_quantities(mdp, behaviour)
            mu = eq0["mu"]
            v_star, _ = optimal_values(mdp)
            train = training_batch(mdp, behaviour, mu, rng)
            manifest.append({
                "mixing": float(mixing), "task_index": int(task_index),
                "mdp_hash": h(np.asarray(mdp["P"]), np.asarray(mdp["R"])),
                "policy0_hash": h(behaviour),
                "train_hash": h(train["states"], train["actions"], train["rewards"],
                                train["next_states"], train["next_actions"]),
                "v_star_sum": float(np.sum(v_star)),
                "v0_sum": float(np.sum(eq0["v_pi"])),
            })

            # per-record step-major execution: one batch per step, shared by all live cells
            cell_state: dict[tuple, dict[str, Any]] = {}
            for algorithm in ALGORITHMS:
                for arm in ARMS:
                    for producer in PRODUCERS:
                        cell_state[(algorithm, arm, producer)] = {
                            "pi": behaviour.copy(),
                            "v_chain": [eq0["v_pi"].copy()],
                            "q_ref": eq0["q_pi"].copy(),
                            "steps": [],
                            "live": True,
                        }
            batches: dict[int, dict[str, Any]] = {}
            reduced_cache: dict[tuple, Any] = {}

            for step in range(1, K + 1):
                live = [key for key, stt in cell_state.items() if stt["live"]]
                if not live:
                    break
                if step not in batches:
                    raw = certification_batch(
                        mdp, behaviour, mu,
                        [SEED, SALT, int(round(100 * mixing)), int(task_index), step])
                    batches[step] = raw
                    ukey = (round(float(mixing), 6), int(task_index), step)
                    if ukey not in unique_batches:
                        unique_batches.add(ukey)
                        items_unique += CERT_CHAINS * CHAIN_LENGTH
                    batch_hashes[f"{mixing}|{task_index}|{step}"] = h(
                        raw["states"][:1000], raw["next_states"][:1000])
                raw = batches[step]

                for chain_count in {ARM_CHAINS[k[1]] for k in live}:
                    if (step, chain_count) not in reduced_cache:
                        reduced_cache[(step, chain_count)] = first_visit(
                            raw, CHAIN_LENGTH, max_chains=chain_count)

                for algorithm, arm, producer in live:
                    stt = cell_state[(algorithm, arm, producer)]
                    qh = produce(algorithm, producer, stt["pi"], train)
                    if producer == "network":
                        net_forwards += 1
                    chain_count = ARM_CHAINS[arm]
                    reduced, counts = reduced_cache[(step, chain_count)]
                    realized = float(np.max(np.abs(qh - stt["q_ref"])))
                    e_q, reasons, cert_detail = certificate(qh, stt["pi"], reduced, counts)
                    if e_q is None:
                        decision = {"emitted": False, "policy_plus": stt["pi"].copy(),
                                    "eta_selected": ([] if arm == "perstate_n16k" else None),
                                    "lb_by_state": np.zeros(S), "states_updated": 0}
                        if arm == "perstate_n16k":
                            decision["eta_selected"] = [None] * S
                    else:
                        decision = (perstate_decision(stt["pi"], qh, e_q)
                                    if arm == "perstate_n16k"
                                    else conjunctive_decision(stt["pi"], qh, e_q))
                        reasons = [] if decision["emitted"] else ["improvement_lcb_nonpositive"]
                    lb = np.asarray(decision["lb_by_state"], dtype=np.float64)
                    entry: dict[str, Any] = {
                        "step": step, "emitted": bool(decision["emitted"]),
                        "eta_selected": decision["eta_selected"],
                        "states_updated": int(decision["states_updated"]),
                        "e_q": e_q,
                        "min_lb": float(lb.min()) if lb.size else None,
                        "min_pair_count_observed": int(counts.min()),
                        "ordered_reasons": reasons,
                        "oracle_audit": {
                            "purpose": "truth-based audit only; never a certificate input",
                            "realized_q_sup_error_vs_current_target_pi": realized,
                            "certificate_violation": bool(e_q is not None and float(e_q) < realized),
                        },
                    }
                    # §6 boundary guard
                    margin = None if e_q is None else float(e_q) - realized
                    loc = {"mixing": float(mixing), "task_index": int(task_index),
                           "algorithm": algorithm, "arm": arm, "producer": producer,
                           "step": step}
                    if margin is not None and (not math.isfinite(margin) or margin < 0.0
                                               or abs(margin) <= NUMERIC_PAUSE):
                        anomalies.append({**loc, "kind": "h1_margin_anomaly",
                                          "margin": margin, "e_q": e_q,
                                          "realized": realized})
                    if decision["emitted"]:
                        nxt = np.asarray(decision["policy_plus"], dtype=np.float64)
                        eq_next = exact_quantities(mdp, nxt)
                        v_next = eq_next["v_pi"]
                        delta_v = v_next - stt["v_chain"][-1]
                        entry["oracle_audit"].update({
                            "value_delta_vs_previous": delta_v.tolist(),
                            "componentwise_nondegrading": bool(float(np.min(delta_v)) >= -1e-12),
                            "total_value_gain": float(np.sum(delta_v)),
                        })
                        if float(np.min(delta_v)) < -1e-12 or not np.all(np.isfinite(delta_v)):
                            anomalies.append({**loc, "kind": "h1_degrading_step",
                                              "min_delta": float(np.min(delta_v))})
                        stt["v_chain"].append(v_next)
                        stt["q_ref"] = eq_next["q_pi"]
                        stt["pi"] = nxt
                    stt["steps"].append(entry)
                    if not decision["emitted"]:
                        stt["live"] = False
                # free the raw batch for this step: the reduced arrays are what we keep
                batches.pop(step, None)

            if anomalies:
                (out_dir / "h1_anomaly.json").write_text(json.dumps(
                    {"task_id": TASK_ID, "actor": ACTOR, "aborted": True,
                     "anomalies": anomalies, "location": anomalies[-1],
                     "next_step": ("section 6 read-only high-precision recomputation from the "
                                   "same sealed inputs; no re-run with a new seed")},
                    indent=2, sort_keys=True), encoding="utf-8")
                raise SystemExit(f"H1/numerical anomaly: {anomalies[-1]} -- all formal sampling stopped")

            routes: dict[str, Any] = {}
            for algorithm in ALGORITHMS:
                arms_out: dict[str, Any] = {}
                for arm in ARMS:
                    producers_out: dict[str, Any] = {}
                    for producer in PRODUCERS:
                        stt = cell_state[(algorithm, arm, producer)]
                        v_final = stt["v_chain"][-1]
                        init_gap = float(np.sum(v_star - stt["v_chain"][0]))
                        producers_out[producer] = {
                            "steps": stt["steps"],
                            "emitted_steps": sum(1 for s in stt["steps"] if s["emitted"]),
                            "initial_v_sum": float(np.sum(stt["v_chain"][0])),
                            "final_v_sum": float(np.sum(v_final)),
                            "initial_suboptimality": init_gap,
                            "fraction_gap_closed": (float(np.sum(v_final - stt["v_chain"][0]))
                                                    / init_gap if init_gap > 0 else 1.0),
                            "final_gap_per_state": (v_star - v_final).tolist(),
                        }
                    arms_out[arm] = producers_out
                routes[algorithm] = arms_out
            records.append({"task_id": TASK_ID, "mixing": float(mixing),
                            "task_index": int(task_index),
                            "environment_is_new": True,
                            "wall_seconds": time.time() - t0, "routes": routes})
            print(f"  {mixing}/{task_index} ({time.time() - t0:.1f}s)", flush=True)
            _partial(out_dir, records, len(MIXINGS) * len(idx_list))

    wall = time.time() - t_start
    bundle = {
        "task_id": TASK_ID, "actor": ACTOR, "baseline": BASELINE,
        "arms": list(ARMS), "routes": list(ALGORITHMS), "producers": list(PRODUCERS),
        "task_indices": [int(t) for t in idx_list], "mixings": list(MIXINGS),
        "chains_per_step": CERT_CHAINS, "chain_length": CHAIN_LENGTH,
        "delta_step": DELTA_STEP, "delta_total": DELTA_TOTAL, "horizon": K,
        "min_visits": MIN_VISITS, "eta_grid": list(ETA_GRID),
        "salt": SALT, "seed": SEED, "smoke": smoke,
        "record_count": len(records), "net_forwards": net_forwards,
        "items_unique": items_unique, "unique_batches": len(unique_batches),
        "wall_seconds": wall,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    (out_dir / "task_results.json").write_text(json.dumps(bundle, indent=2, sort_keys=True),
                                               encoding="utf-8")
    (out_dir / "input_manifest.json").write_text(json.dumps({
        "task_id": TASK_ID, "seed": SEED, "salt": SALT, "records": manifest,
        "batch_hashes": batch_hashes,
        "run_metadata": {
            "task_sheet_path": str(TASK_SHEET),
            "task_sheet_sha256_lf": sha256_lf(TASK_SHEET),
            "baseline": BASELINE,
            "code_sha256_lf": {Path(__file__).name: sha256_lf(Path(__file__))},
            "platform": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "platform": platform.platform()},
            "command": " ".join([sys.executable, *sys.argv]),
            "argv": list(sys.argv), "started_utc": started_utc,
        },
        "note": "no baseline module is imported; every operator is written from the spec",
    }, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps({
        "task_id": TASK_ID, "actor": ACTOR, "baseline": BASELINE,
        "arms": list(ARMS), "routes": list(ALGORITHMS), "producers": list(PRODUCERS),
        "horizon": K, "chains_per_step": CERT_CHAINS, "chain_length": CHAIN_LENGTH,
        "delta_step": DELTA_STEP, "min_visits": MIN_VISITS, "eta_grid": list(ETA_GRID),
        "salt": SALT, "seed": SEED, "smoke": smoke,
    }, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps({
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
    }, indent=2, sort_keys=True), encoding="utf-8")

    # §6.1 seal-time round-trip integrity check
    reloaded = json.loads((out_dir / "task_results.json").read_text(encoding="utf-8"))
    mismatches = _roundtrip_check(bundle, reloaded)
    (out_dir / "seal_integrity.json").write_text(json.dumps(
        {"float_fields_compared": mismatches[0], "mismatches": mismatches[1],
         "passed": mismatches[1] == 0}, indent=2), encoding="utf-8")
    if mismatches[1]:
        raise SystemExit(f"seal round-trip integrity check failed: {mismatches[1]} mismatches")

    print(json.dumps({"records": len(records), "net_forwards": net_forwards,
                      "items_unique": items_unique, "unique_batches": len(unique_batches),
                      "wall_total_s": round(wall, 1)}, indent=2))


def _roundtrip_check(bundle, reloaded):
    """§6.1: every float/int/bool decision-critical field must equal its in-memory value."""
    compared = 0
    bad = 0
    for r_new, r_old in zip(reloaded["records"], bundle["records"]):
        for algorithm, arms in r_new["routes"].items():
            for arm, producers in arms.items():
                for producer, payload in producers.items():
                    old_steps = r_old["routes"][algorithm][arm][producer]["steps"]
                    for s_new, s_old in zip(payload["steps"], old_steps):
                        compared += 1
                        if s_new["emitted"] != s_old["emitted"]:
                            bad += 1
                        if s_new["states_updated"] != s_old["states_updated"]:
                            bad += 1
                        if s_new["ordered_reasons"] != s_old["ordered_reasons"]:
                            bad += 1
                        if s_new["min_pair_count_observed"] != s_old["min_pair_count_observed"]:
                            bad += 1
                        e_new, e_old = s_new["e_q"], s_old["e_q"]
                        if (e_new is None) != (e_old is None):
                            bad += 1
                        elif e_new is not None and float(e_new) != float(e_old):
                            bad += 1
                        for a_new, a_old in zip(
                                s_new["eta_selected"] if isinstance(s_new["eta_selected"], list)
                                else [s_new["eta_selected"]],
                                s_old["eta_selected"] if isinstance(s_old["eta_selected"], list)
                                else [s_old["eta_selected"]]):
                            if (a_new is None) != (a_old is None):
                                bad += 1
                            elif a_new is not None and float(a_new) != float(a_old):
                                bad += 1
                            elif a_new is not None and a_new not in ETA_GRID:
                                bad += 1
                        if s_new["min_lb"] is not None and s_old["min_lb"] is not None and (
                                float(s_new["min_lb"]) != float(s_old["min_lb"])):
                            bad += 1
    return compared, bad


def _partial(out_dir, records, planned):
    (out_dir / "task_results.partial.json").write_text(json.dumps(
        {"task_id": TASK_ID, "actor": ACTOR, "partial": True,
         "records_done": len(records), "planned_records": planned,
         "records": records}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
