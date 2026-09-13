"""FP-CERTCHECK-001: high-precision recheck of every sealed decision point.

For each step entry sealed in an FP-CERTFIX-001 multi run, this script replays
the trajectory from the frozen seed schedule -- regenerating the training batch,
the step's certification batch, and re-running the route for ``q_hat`` -- and
recomputes the certificate and the full eta-grid decision THREE ways:

1. float64 replay: must reproduce the sealed decision exactly (emission, eta,
   min_lb to 1e-15), or the seal is not reproducible;
2. ``math.fsum`` replay: compensated (exactly rounded) summation for the
   residual mean/variance and all dot products, float64 otherwise;
3. mpmath 50-digit replay: treats the regenerated float64 inputs as exact and
   carries all arithmetic at 50 digits -- the reference value.

A decision is arithmetic-dominated if (2) or (3) flips the emitted/abstained
verdict or changes the selected eta relative to (1). This answers the question
the withdrawn step-17/step-18 claims asked the wrong way: not "does some margin
ratio cross some floor" but "does any ACTUAL sealed decision change when the
decision chain is computed exactly".

The check seals per step: policy snapshot, q_hat sup-difference vs the original
producer, E_Q, and every eta's per-state LB from each of the three computations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_mp_certificate as mc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import kernel_batch, vectorised_batch  # noqa: E402

import mpmath as mp  # noqa: E402

MP_DPS = 50
ETA_GRID = fs.ETA_CANDIDATES
GAMMA = fs.GAMMA
TIGHT_MARGIN = 1e-6


def fsum_mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def fsum_var(values: list[float], mean: float) -> float:
    return math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)


def cert_mp_fsum(
    q_hat: np.ndarray, policy: np.ndarray, batch: dict, n_unused: int, delta_step: float
) -> tuple[float, np.ndarray]:
    """MP certificate with compensated summation (derivation section 2).

    Per-pair sample size is the retained size in ``batch`` (first-visit
    protocol: random ``N_x``, valid at every realised value).
    """
    residuals = mc.residuals_for(q_hat, policy, batch)
    flat = batch["states"].astype(np.int64) * fs.N_ACTIONS + batch["actions"]
    d = fs.N_STATES * fs.N_ACTIONS
    delta_dir = delta_step / (2.0 * d)
    log_term = math.log(2.0 / delta_dir)
    radii = np.zeros(d)
    means = np.zeros(d)
    for pair in range(d):
        y = [float(v) for v in residuals[np.flatnonzero(flat == pair)]]
        n = len(y)
        mean = fsum_mean(y)
        var = fsum_var(y, mean)
        means[pair] = mean
        radii[pair] = math.sqrt(2.0 * max(var, 0.0) * log_term / n) + (
            mc.MP_CONSTANT * mc.Y_RANGE * log_term / max(n - 1, 1)
        )
    eps_res = float(np.max(np.abs(means) + radii))
    return eps_res / (1.0 - GAMMA), radii


def cert_split_fsum(
    q_hat: np.ndarray, policy: np.ndarray, batch: dict, n_unused: int, delta_step: float
) -> float:
    """Split-Bernstein certificate (derivation section 7), compensated sums;
    order-split halves of each pair's retained first-visit sample."""
    residuals = mc.residuals_for(q_hat, policy, batch)
    flat = batch["states"].astype(np.int64) * fs.N_ACTIONS + batch["actions"]
    d = fs.N_STATES * fs.N_ACTIONS
    delta_1 = delta_step / (2.0 * d)
    delta_2 = delta_step / (2.0 * d)
    radii = np.zeros(d)
    means = np.zeros(d)
    for pair in range(d):
        idx = np.flatnonzero(flat == pair)
        m = idx.size // 2
        y_a = [float(v) for v in residuals[idx[:m]]]
        y_b = [float(v) for v in residuals[idx[m:]]]
        m2 = math.fsum(v * v for v in y_a) / m
        slack = (mc.ENVELOPE**2) * math.sqrt(math.log(1.0 / delta_1) / (2.0 * m))
        v_x = m2 + slack
        radii[pair] = mc._bernstein_t(v_x, mc.Y_RANGE, math.log(2.0 / delta_2), idx.size - m)
        means[pair] = fsum_mean(y_b)
    eps_res = float(np.max(np.abs(means) + radii))
    return eps_res / (1.0 - GAMMA)


def cert_mp_mpmath(
    q_hat: np.ndarray, policy: np.ndarray, batch: dict, n: int, delta_step: float
) -> mp.mpf:
    """MP certificate at 50 digits; float64 inputs treated as exact."""
    with mp.workdps(MP_DPS):
        residuals = mc.residuals_for(q_hat, policy, batch)
        flat = batch["states"].astype(np.int64) * fs.N_ACTIONS + batch["actions"]
        d = fs.N_STATES * fs.N_ACTIONS
        delta_dir = mp.mpf(delta_step) / (2 * d)
        log_term = mp.log(2 / delta_dir)
        eps_res = mp.mpf(0)
        for pair in range(d):
            ys = [mp.mpf(float(v)) for v in residuals[np.flatnonzero(flat == pair)]]
            n = len(ys)
            mean = mp.fsum(ys) / n
            var = mp.fsum([(y - mean) ** 2 for y in ys]) / (n - 1)
            radius = mp.sqrt(2 * var * log_term / n) + (
                mp.mpf(mc.MP_CONSTANT) * mc.Y_RANGE * log_term / (n - 1)
            )
            eps_res = max(eps_res, abs(mean) + radius)
        return eps_res / (1 - mp.mpf(GAMMA))


def cert_split_mpmath(
    q_hat: np.ndarray, policy: np.ndarray, batch: dict, n: int, delta_step: float
) -> mp.mpf:
    """Split-Bernstein certificate at 50 digits (fixed-point solve included)."""
    with mp.workdps(MP_DPS):
        residuals = mc.residuals_for(q_hat, policy, batch)
        flat = batch["states"].astype(np.int64) * fs.N_ACTIONS + batch["actions"]
        d = fs.N_STATES * fs.N_ACTIONS
        delta_1 = mp.mpf(delta_step) / (2 * d)
        delta_2 = mp.mpf(delta_step) / (2 * d)
        eps_res = mp.mpf(0)
        for pair in range(d):
            idx = np.flatnonzero(flat == pair)
            m = idx.size // 2
            y_a = [mp.mpf(float(v)) for v in residuals[idx[:m]]]
            y_b = [mp.mpf(float(v)) for v in residuals[idx[m:]]]
            m2 = mp.fsum([v * v for v in y_a]) / m
            slack = mp.mpf(mc.ENVELOPE) ** 2 * mp.sqrt(mp.log(1 / delta_1) / (2 * m))
            v_x = m2 + slack
            log2 = mp.log(2 / delta_2)
            t = mp.sqrt(2 * v_x * log2 / (idx.size - m))
            for _ in range(300):
                nxt = mp.sqrt((2 * v_x + mp.mpf(4) / 3 * mc.Y_RANGE * t) * log2 / (idx.size - m))
                if abs(nxt - t) <= mp.mpf("1e-40") * max(mp.mpf(1), t):
                    break
                t = nxt
            mean_b = mp.fsum(y_b) / (idx.size - m)
            eps_res = max(eps_res, abs(mean_b) + t)
        return eps_res / (1 - mp.mpf(GAMMA))


def candidate_lbs_float64(
    policy: np.ndarray, q_hat: np.ndarray, e_q: float
) -> list[tuple[float, np.ndarray]]:
    out = []
    for eta in ETA_GRID:
        cand = fs.es.relative_softmax_candidate(policy, q_hat, eta)
        delta_pi = cand - policy
        i_hat = (delta_pi * q_hat).sum(axis=1)
        lb = i_hat - e_q * np.abs(delta_pi).sum(axis=1)
        out.append((float(eta), lb))
    return out


def candidate_lbs_fsum(
    policy: np.ndarray, q_hat: np.ndarray, e_q: float
) -> list[tuple[float, np.ndarray]]:
    """Candidate LBs with compensated dot products; softmax still float64."""
    out = []
    for eta in ETA_GRID:
        cand = fs.es.relative_softmax_candidate(policy, q_hat, eta)
        delta_pi = cand - policy
        lbs = []
        for s in range(policy.shape[0]):
            i_hat = math.fsum(
                float(delta_pi[s, a]) * float(q_hat[s, a])
                for a in range(policy.shape[1])
            )
            l1 = math.fsum(abs(float(delta_pi[s, a])) for a in range(policy.shape[1]))
            lbs.append(i_hat - e_q * l1)
        out.append((float(eta), np.asarray(lbs)))
    return out


def candidate_lbs_mpmath(
    policy: np.ndarray, q_hat: np.ndarray, e_q: mp.mpf
) -> list[tuple[float, list[mp.mpf]]]:
    """Full 50-digit softmax + dot products; float64 inputs treated as exact."""
    out = []
    with mp.workdps(MP_DPS):
        n_s, n_a = policy.shape
        for eta in ETA_GRID:
            lbs = []
            for s in range(n_s):
                logits = [
                    mp.log(mp.mpf(float(policy[s, a]))) + mp.mpf(eta) * mp.mpf(float(q_hat[s, a]))
                    for a in range(n_a)
                ]
                m = max(logits)
                w = [mp.exp(l - m) for l in logits]
                tot = mp.fsum(w)
                cand = [wi / tot for wi in w]
                dpi = [cand[a] - mp.mpf(float(policy[s, a])) for a in range(n_a)]
                i_hat = mp.fsum(
                    dpi[a] * mp.mpf(float(q_hat[s, a])) for a in range(n_a)
                )
                l1 = mp.fsum(abs(dpi[a]) for a in range(n_a))
                lbs.append(i_hat - e_q * l1)
            out.append((float(eta), lbs))
    return out


def first_passage(lbs_per_eta) -> tuple[float | None, float | None]:
    for eta, lbs in lbs_per_eta:
        arr = np.asarray([float(v) for v in lbs], dtype=np.float64)
        if float(arr.min()) > 0.0:
            return float(eta), float(arr.min())
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.results / "task_results.json").read_text(encoding="utf-8"))
    n_per_pair = int(data.get("n_per_pair") or 0)  # unused under first-visit
    extraction = data.get("extraction", "first_visit")
    chains = int(data["chains_per_step"])
    delta_step = float(data["delta_step"])
    salt = int(data["task_salt"])
    routes = list(data["routes"])
    arms = list(data["arms"])

    rows_out: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    replay_mismatches: list[dict[str, Any]] = []
    n_points = 0
    tight_points = 0
    margins_all: list[float] = []

    for rec in data["records"]:
        mixing = float(rec["mixing"])
        task_index = int(rec["task_index"])
        mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
        exact0 = policy_quantities(mdp, behaviour)
        mu_state = np.asarray(exact0["mu_state"], dtype=np.float64)
        train = fs.training_batch(mdp, behaviour, mu_state, rng)
        for route in routes:
            for arm in arms:
                sealed_steps = rec["routes"][route][arm]["steps"]
                current = behaviour.copy()
                for s in sealed_steps:
                    step_index = int(s["step"])
                    q_hat = np.asarray(
                        fs.run_route(route, current, train)["q_hat"],
                        dtype=np.float64,
                    ).reshape(fs.N_STATES, fs.N_ACTIONS)
                    if extraction == "oracle_kernel":
                        # Second confirmation arm: re-draw the same kernel sample.
                        reduced, counts = kernel_batch(
                            mdp,
                            step_seed_parts(
                                fs.SEED, salt, mixing, task_index, step_index
                            ),
                            int(data.get("n_per_pair") or 65536),
                        )
                    else:
                        raw = vectorised_batch(
                            mdp,
                            behaviour,
                            mu_state,
                            step_seed_parts(fs.SEED, salt, mixing, task_index, step_index),
                            chains,
                            64,
                        )
                        reduced, counts = first_visit_batch(raw, 64)
                    # (1) float64 replay via the production module
                    min_visits = int(data.get("min_visits", 2000))
                    cert = mc.mp_certificate_firstvisit(
                        q_hat,
                        current,
                        reduced,
                        min_visits=min_visits,
                        delta_step=delta_step,
                    ) if arm == "mp" else mc.split_bernstein_firstvisit(
                        q_hat,
                        current,
                        reduced,
                        min_visits=min_visits,
                        delta_step=delta_step,
                    )
                    if cert["e_q"] is None:
                        ok = (not bool(s["update_emitted"])) and any(
                            "heldout_pair_support_missing" in r_
                            for r_ in s.get("ordered_reasons", [])
                        )
                        rows_out.append(
                            {
                                "record": f"{mixing}/{task_index}/{route}/{arm}",
                                "step": step_index,
                                "not_certified_confirmed": bool(ok),
                                "min_pair_count_observed": int(counts.min()),
                            }
                        )
                        if not ok:
                            replay_mismatches.append(
                                {
                                    "record": f"{mixing}/{task_index}/{route}/{arm}",
                                    "step": step_index,
                                    "kind": "heldout_shortfall_not_matched",
                                }
                            )
                        break
                    e_q64 = float(cert["e_q"])
                    lbs64 = candidate_lbs_float64(current, q_hat, e_q64)
                    eta64, margin64 = first_passage(lbs64)
                    # reproducibility vs seal
                    sealed_emitted = bool(s["update_emitted"])
                    sealed_eta = s["eta_selected"]
                    if (eta64 is not None) != sealed_emitted or (
                        eta64 is not None
                        and abs(eta64 - float(sealed_eta)) > 1e-15
                    ):
                        replay_mismatches.append(
                            {
                                "record": f"{mixing}/{task_index}/{route}/{arm}",
                                "step": step_index,
                                "sealed": [sealed_emitted, sealed_eta],
                                "replay": [eta64 is not None, eta64],
                            }
                        )
                    # (2) fsum replay
                    if arm == "mp":
                        e_q_fs, _ = cert_mp_fsum(
                            q_hat, current, reduced, n_per_pair, delta_step
                        )
                    else:
                        e_q_fs = cert_split_fsum(
                            q_hat, current, reduced, n_per_pair, delta_step
                        )
                    lbs_fs = candidate_lbs_fsum(current, q_hat, e_q_fs)
                    eta_fs, margin_fs = first_passage(lbs_fs)
                    if (eta_fs is not None) != (eta64 is not None) or (
                        eta_fs is not None and abs(eta_fs - eta64) > 1e-15
                    ):
                        flips.append(
                            {
                                "record": f"{mixing}/{task_index}/{route}/{arm}",
                                "step": step_index,
                                "kind": "fsum",
                                "float64": [eta64, margin64],
                                "fsum": [eta_fs, margin_fs],
                            }
                        )
                    # (3) mpmath only where the decision is tight
                    margin_for_gate = margin64 if margin64 is not None else None
                    near = (
                        margin_for_gate is not None and abs(margin_for_gate) < TIGHT_MARGIN
                    ) or (eta64 is None)
                    mp_detail = None
                    if near:
                        tight_points += 1
                        if arm == "mp":
                            e_q_mp = cert_mp_mpmath(
                                q_hat, current, reduced, n_per_pair, delta_step
                            )
                        else:
                            e_q_mp = cert_split_mpmath(
                                q_hat, current, reduced, n_per_pair, delta_step
                            )
                        lbs_mp = candidate_lbs_mpmath(current, q_hat, e_q_mp)
                        eta_mp, margin_mp = first_passage(lbs_mp)
                        mp_detail = {
                            "e_q": float(e_q_mp),
                            "eta": eta_mp,
                            "margin": margin_mp,
                        }
                        if (eta_mp is not None) != (eta64 is not None) or (
                            eta_mp is not None and abs(eta_mp - eta64) > 1e-15
                        ):
                            flips.append(
                                {
                                    "record": f"{mixing}/{task_index}/{route}/{arm}",
                                    "step": step_index,
                                    "kind": "mpmath50",
                                    "float64": [eta64, margin64],
                                    "mpmath": [eta_mp, margin_mp],
                                }
                            )
                    n_points += 1
                    if margin64 is not None:
                        margins_all.append(abs(margin64))
                    rows_out.append(
                        {
                            "record": f"{mixing}/{task_index}/{route}/{arm}",
                            "step": step_index,
                            "sealed_emitted": sealed_emitted,
                            "sealed_eta": sealed_eta,
                            "sealed_min_lb": s["min_lb"],
                            "e_q_float64": e_q64,
                            "e_q_fsum": float(e_q_fs),
                            "eta_float64": eta64,
                            "margin_float64": margin64,
                            "eta_fsum": eta_fs,
                            "margin_fsum": margin_fs,
                            "mpmath": mp_detail,
                            "lbs_by_eta_float64": {
                                f"{eta:g}": [float(v) for v in lbs]
                                for eta, lbs in lbs64
                            },
                        }
                    )
                    if eta64 is None:
                        break
                    # advance the replayed policy with the sealed/replayed eta
                    cand = fs.es.relative_softmax_candidate(current, q_hat, eta64)
                    current = np.asarray(cand, dtype=np.float64)

    margins_all.sort()
    out = {
        "source_results": str(args.results),
        "decision_points_checked": n_points,
        "tight_points_mpmath": tight_points,
        "replay_mismatches": replay_mismatches,
        "decision_flips": flips,
        "smallest_emitted_margins": margins_all[:10],
        "verdict": (
            "no sealed decision is arithmetic-dominated"
            if not flips and not replay_mismatches
            else "FLIPS OR REPLAY MISMATCHES FOUND -- see lists"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hp_recheck.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    (args.output_dir / "hp_recheck_steps.json").write_text(
        json.dumps(rows_out, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in out.items() if k not in ("replay_mismatches", "decision_flips")}, indent=2, sort_keys=True, default=float))
    print("replay_mismatches:", len(replay_mismatches), "flips:", len(flips))


if __name__ == "__main__":
    main()
