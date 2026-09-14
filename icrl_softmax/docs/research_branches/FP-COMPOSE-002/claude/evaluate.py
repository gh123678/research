"""FP-COMPOSE-002 / claude: network x perstate x L12 at K=12.

Same protocol as FP-COMPOSE-001 with exactly ONE variable changed: the horizon, K=4 -> K=12.
In particular delta_step is held at 0.0125 rather than becoming delta_total/K; making it
0.05/12 would inflate log(2/delta_dir) and hence the radius and E_Q, confounding the horizon
effect with the risk-budget effect -- which is the very quantity under measurement. The
consequence, delta_total = K * delta_step = 0.15, is a declared cost, not a hidden change.
The frozen y_range formula is kept unchanged on purpose: changing the range bound and the
horizon together would make the result unattributable.

Because delta_step and the step-indexed seed vector are unchanged, steps 1..4 of this run
have IDENTICAL inputs to FP-COMPOSE-001 and must reproduce its sealed records bit for bit
(pre-registered prediction P0). Any discrepancy is an implementation or environment defect,
not a scientific result.

The L12 scalar, the candidate grid, the row-wise per-state decision and the audit solve are
implemented here rather than imported, and the baseline modules that define the objects under
test (model.py's attention networks, the array routes, the family MDP generator, the training
batch, the sampler and the first-visit extractor) are READ, never modified.

    python evaluate.py --output-dir <results/FP-COMPOSE-002/claude/formal>
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

PROJECT = Path(__file__).resolve().parents[4]          # .../icrl_softmax
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa as es  # noqa: E402
import torch  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_xfam_001 import (  # noqa: E402
    FAMILIES, build_family_task, run_route, training_batch, vectorised_batch_generic,
)
from fp_certfix_first_n import first_visit_batch  # noqa: E402
from model import (  # noqa: E402
    EndToEndFiniteSoftmaxExpectedSARSA, EndToEndMaskedSoftmaxExpectedSARSA,
)

TASK_ID = "FP-COMPOSE-002"
ACTOR = "claude"
BASELINE = "2840cbfdc07808c2e268b37b24c6eee676903fdf"
SEED = 20260911
SALT = 90417
LAYERS = 160
ALPHA = 0.65
ZETA = XI = TAU = 8.0
K = 12
CHAINS = 16384
CHAIN_LENGTH = 64
MIN_VISITS = 2000
DELTA_STEP = 0.0125                 # HELD FIXED from FP-COMPOSE-001 (the design point)
DELTA_TOTAL = K * DELTA_STEP        # = 0.15, the declared arithmetic consequence
ETA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
ROUTES = ("expected_exact", "expected_finite")
PRODUCERS = ("numpy", "network")
CELLS = tuple(f"{p}|perstate|L12" for p in PRODUCERS)
DIAG_THRESHOLD = 1e-10          # task sheet section 6: a trigger, NOT a tolerance
NUMERIC_PAUSE = 1e-10           # section 6: |margin| at or below this pauses the run

# Section 3: the manifest records the task-sheet hash and the hash of every invoked code
# file. Unlike FP-COMPOSE-001 the task sheet is tracked in this branch, but the content hash
# is recorded all the same so the frozen definition can be identified exactly.
TASK_SHEET = PROJECT / "docs" / "research_tasks" / "FP-COMPOSE-002.md"
CALLED_CODE = ("model.py", "fixed_policy_expected_sarsa.py",
               "evaluate_fixed_policy_q_routes.py", "evaluate_fp_xfam_001.py",
               "fp_certfix_first_n.py",
               "docs/research_branches/FP-COMPOSE-002/claude/evaluate.py")


def file_sha256_lf(path: Path):
    """SHA256 with CRLF normalised to LF, per section 3 ('code hashes are unified LF')."""
    try:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    except OSError:
        return None


def run_metadata(argv, started_utc) -> dict:
    import os
    return {
        "task_sheet_path": str(TASK_SHEET),
        "task_sheet_sha256_lf": file_sha256_lf(TASK_SHEET),
        "task_sheet_tracked_in_git": False,
        "task_sheet_note": ("the frozen task sheet exists only as an untracked file in the "
                            "authoring worktree; the content hash is the integrity anchor"),
        "baseline": BASELINE,
        "code_sha256_lf": {name: file_sha256_lf(PROJECT / name) for name in CALLED_CODE},
        "library_versions": {"numpy": np.__version__, "torch": torch.__version__,
                             "python": sys.version.split()[0]},
        "threads": {"torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")},
        "command": " ".join([sys.executable, *argv]),
        "argv": list(argv),
        "cwd": os.getcwd(),
        "started_utc": started_utc,
    }


# --------------------------------------------------------------------------- #
# Independent numeric kernels
# --------------------------------------------------------------------------- #
def solve_qpi_vpi(mdp, policy, gamma):
    """Exact Q^pi and v^pi by direct linear solve -- the audit's own truth.

    Q^pi = (I - gamma P^pi)^-1 r^pi on the pair space; v^pi(s) = sum_a pi(a|s)Q(s,a).
    Used instead of policy_quantities so the audit does not depend on it.
    """
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    pi = np.asarray(policy, dtype=np.float64)
    S, A = pi.shape
    r_pi = (P * R).sum(axis=2)                                  # (S,A)
    Ppi = np.einsum("sat,tb->satb", P, pi).reshape(S * A, S * A)
    Q = np.linalg.solve(np.eye(S * A) - gamma * Ppi, r_pi.reshape(-1)).reshape(S, A)
    v = (pi * Q).sum(axis=1)
    return Q, v


def optimal_v(mdp, gamma, tol=1e-12, iters=2000):
    """v* by value iteration -- audit only."""
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    S, A = P.shape[0], P.shape[1]
    v = np.zeros(S)
    for _ in range(iters):
        q = (P * (R + gamma * v[None, None, :])).sum(axis=2)
        nv = q.max(axis=1)
        if np.max(np.abs(nv - v)) < tol:
            v = nv
            break
        v = nv
    return v


def l12_scalar(q_hat, policy, residuals, members, delta_step, gamma, r_star):
    """The L12 certificate scalar, written here (task sheet section 3).

    Magnitude bound only: eps_x = |mean_x| + radius_x, E_Q = max_x eps_x/(1-gamma).
    No kernel, no propagation, no central interval.
    """
    S, A = policy.shape
    d = S * A
    v_hat = (policy * q_hat).sum(axis=1)
    y_range = 2.0 * (r_star + gamma * float(np.max(np.abs(v_hat)))
                     + float(np.max(np.abs(q_hat))))
    delta_dir = float(delta_step) / d
    log_term = math.log(2.0 / delta_dir)
    means = np.zeros(d); radii = np.zeros(d); vars_ = np.zeros(d); sizes = np.zeros(d, dtype=int)
    for x in range(d):
        y = residuals[members[x]]
        n = int(y.size)
        sizes[x] = n
        v = float(np.var(y, ddof=1)) if n > 1 else 0.0
        vars_[x] = v
        means[x] = float(np.mean(y))
        radii[x] = math.sqrt(2.0 * max(v, 0.0) * log_term / n) + (
            (7.0 / 3.0) * y_range * log_term / max(n - 1, 1))
    eps = np.abs(means) + radii
    return {"e_q": float(np.max(eps)) / (1.0 - gamma), "means": means, "radii": radii,
            "vars": vars_, "sizes": sizes, "eps": eps, "y_range": y_range,
            "delta_dir": delta_dir, "delta_step": float(delta_step)}


def perstate_rows(policy, q_hat, e_q):
    """Row-wise first passage on the eta grid; every candidate uses the OLD pi.

    Returns the update mask, each row's chosen eta (None if it does not pass), the
    full candidate LB table, and pi_after. Written here, not imported.
    """
    S = policy.shape[0]
    rows_eta: list[Any] = []
    mask = np.zeros(S, dtype=bool)
    chosen_lb = np.zeros(S)
    table: list[list[float]] = []
    pi_after = policy.copy()
    for s in range(S):
        lb_row = []
        pick = None
        for eta in ETA_GRID:
            cand = es.relative_softmax_candidate(policy, q_hat, eta)
            d_row = cand[s] - policy[s]
            lb = float((d_row * q_hat[s]).sum()) - e_q * float(np.abs(d_row).sum())
            lb_row.append(lb)
            if pick is None and lb > 0.0:
                pick = float(eta)
                chosen_lb[s] = lb
                pi_after[s] = cand[s]
        table.append(lb_row)
        rows_eta.append(pick)
        mask[s] = pick is not None
    return {"pi_after": pi_after, "mask": mask, "rows_eta": rows_eta,
            "chosen_lb": chosen_lb, "lb_table": table,
            "emitted": bool(mask.any()), "states_updated": int(mask.sum())}


def make_networks(fam):
    return {
        "expected_exact": EndToEndMaskedSoftmaxExpectedSARSA(gamma=fam["gamma"], alpha=ALPHA),
        "expected_finite": EndToEndFiniteSoftmaxExpectedSARSA(
            gamma=fam["gamma"], alpha=ALPHA, zeta=ZETA, xi=XI, tau=TAU),
    }


def network_qhat(net, policy, train):
    st = torch.as_tensor(train["states"], dtype=torch.long)
    ac = torch.as_tensor(train["actions"], dtype=torch.long)
    rw = torch.as_tensor(train["rewards"], dtype=torch.float32)
    ns = torch.as_tensor(train["next_states"], dtype=torch.long)
    pi = torch.as_tensor(policy, dtype=torch.float32)
    q = torch.zeros(policy.shape, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(LAYERS):
            q, _ = net(q, st, ac, rw, ns, pi)
    return q.detach().numpy().astype(np.float64)


def h(*arrays) -> str:
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        m.update(str(a.dtype.str).encode()); m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def _partial(out_dir: Path, records, planned_records: int, extra=None) -> None:
    """Crash/accommodation flush: the formal matrix runs about half an hour, so the
    accumulated records are written after every task. The authoritative artefact is still
    task_results.json, written once at the very end."""
    payload = {"task_id": TASK_ID, "actor": ACTOR, "partial": True,
               "records_done": len(records), "planned_records": planned_records,
               "records": records}
    if extra:
        payload.update(extra)
    (out_dir / "task_results.partial.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke only: use the first N task indices per family")
    ap.add_argument("--chains", type=int, default=0,
                    help="smoke only: override CHAINS; 0 keeps the frozen protocol value")
    args = ap.parse_args()
    chains = int(args.chains) if args.chains else CHAINS
    smoke = bool(args.limit or args.chains)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # The design point of this task: delta_step is a FROZEN INPUT here, not derived from K.
    delta_step = DELTA_STEP
    assert abs(DELTA_TOTAL - K * delta_step) < 1e-15
    wall = {"total": 0.0, "sampling": 0.0, "producer": 0.0, "certificate_decision": 0.0,
            "audit": 0.0, "audit_recompute": 0.0}
    net_forwards = 0
    audit_recomputes = 0
    t_start = time.time()
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    unique_batches: set[tuple] = set()
    items_unique = 0
    items_alone = {c: 0 for c in CELLS}
    anomalies: list[dict] = []
    planned_records = sum(
        len(FAMILIES[f]["mixings"]) * (args.limit or len(FAMILIES[f]["task_indices"]))
        for f in ("f1", "f2"))
    started_utc = datetime.now(timezone.utc).isoformat()

    for fam_name in ("f1", "f2"):
        fam = FAMILIES[fam_name]
        nets = make_networks(fam)
        S, A = fam["n_states"], fam["n_actions"]
        d = S * A
        for mixing in fam["mixings"]:
            idx_list = fam["task_indices"][:args.limit] if args.limit else fam["task_indices"]
            for task_index in idx_list:
                t0 = time.time()
                mdp, behaviour, rng = build_family_task(fam, mixing, task_index)
                mu = np.asarray(policy_quantities(mdp, behaviour)["mu_state"], dtype=np.float64)
                train = training_batch(mdp, behaviour, mu, rng)
                P = np.asarray(mdp["P"], dtype=np.float64)
                q0, v0 = solve_qpi_vpi(mdp, behaviour, fam["gamma"])
                vstar = optimal_v(mdp, fam["gamma"])
                gap_denom = float(np.sum(vstar) - np.sum(v0))
                manifest.append({"family": fam_name, "mixing": float(mixing),
                                 "task_index": int(task_index),
                                 "mdp_hash": h(P, np.asarray(mdp["R"])),
                                 "train_hash": h(train["states"], train["actions"],
                                                 train["rewards"], train["next_states"]),
                                 "policy0_hash": h(behaviour),
                                 "v_star_sum": float(np.sum(vstar)),
                                 "v0_sum": float(np.sum(v0))})
                routes: dict[str, Any] = {}
                for route in ROUTES:
                    state = {c: {"pi": behaviour.copy(),
                                 "v": v0.copy(), "qref": q0.copy(),
                                 "steps": []} for c in CELLS}
                    for step in range(1, K + 1):
                        live = [c for c in CELLS if state[c]["steps"] == []
                                or state[c]["steps"][-1]["emitted"]]
                        if not live:
                            break
                        t_s = time.time()
                        raw = vectorised_batch_generic(
                            mdp, behaviour, mu,
                            [SEED, SALT, int(round(100 * mixing)), int(task_index), step],
                            chains, fam, CHAIN_LENGTH)
                        reduced, counts = first_visit_batch(
                            raw, CHAIN_LENGTH, n_states=S, n_actions=A)
                        key = (fam_name, round(float(mixing), 6), int(task_index), step)
                        if key not in unique_batches:
                            unique_batches.add(key)
                            items_unique += chains * CHAIN_LENGTH
                        wall["sampling"] += time.time() - t_s
                        bh = h(raw["states"][:1000], raw["next_states"][:1000])

                        for cell in live:
                            producer = cell.split("|")[0]
                            st = state[cell]
                            t_p = time.time()
                            if producer == "numpy":
                                qh = run_route(route, fam, st["pi"], train).reshape(S, A)
                            else:
                                qh = network_qhat(nets[route], st["pi"], train)
                                net_forwards += 1
                            wall["producer"] += time.time() - t_p

                            # Audit-only producer comparison under the SAME policy. Section 9
                            # requires this extra recompute to be carried separately and not
                            # folded into the main-run producer or data-efficiency figures.
                            t_r = time.time()
                            if producer == "numpy":
                                qh_np_audit = qh
                            else:
                                qh_np_audit = run_route(route, fam, st["pi"],
                                                        train).reshape(S, A)
                                audit_recomputes += 1
                            wall["audit_recompute"] += time.time() - t_r

                            t_c = time.time()
                            flat = (np.asarray(reduced["states"], np.int64) * A
                                    + np.asarray(reduced["actions"], np.int64))
                            members = {x: np.flatnonzero(flat == x) for x in range(d)}
                            resid = _residuals(qh, st["pi"], reduced, fam["gamma"])
                            if min(int(members[x].size) for x in range(d)) < MIN_VISITS:
                                cert = None
                                reasons = ["heldout_pair_support_missing"]
                            else:
                                cert = l12_scalar(qh, st["pi"], resid, members,
                                                  delta_step, fam["gamma"], fam["reward_bound"])
                                reasons = []
                            if cert is None:
                                dec = {"emitted": False, "pi_after": st["pi"].copy(),
                                       "mask": np.zeros(S, bool), "rows_eta": [None] * S,
                                       "chosen_lb": np.zeros(S), "lb_table": [], "states_updated": 0}
                            else:
                                dec = perstate_rows(st["pi"], qh, cert["e_q"])
                                reasons = [] if dec["emitted"] else ["improvement_lcb_nonpositive"]
                            wall["certificate_decision"] += time.time() - t_c

                            t_a = time.time()
                            realized = float(np.max(np.abs(qh - st["qref"])))
                            e_q = None if cert is None else cert["e_q"]
                            entry: dict[str, Any] = {
                                "step": step, "producer": producer, "route": route,
                                "pi_before": st["pi"].tolist(), "q_hat": qh.tolist(),
                                "pair_sizes": (None if cert is None else cert["sizes"].tolist()),
                                "pair_means": (None if cert is None else cert["means"].tolist()),
                                "pair_vars": (None if cert is None else cert["vars"].tolist()),
                                "pair_radii": (None if cert is None else cert["radii"].tolist()),
                                "pair_eps": (None if cert is None else cert["eps"].tolist()),
                                "e_q": e_q, "y_range": (None if cert is None else cert["y_range"]),
                                "delta_dir": (None if cert is None else cert["delta_dir"]),
                                "delta_step": delta_step, "delta_total": DELTA_TOTAL,
                                "lb_table": dec["lb_table"], "rows_eta": dec["rows_eta"],
                                "update_mask": dec["mask"].tolist(),
                                "states_updated": dec["states_updated"],
                                "pi_after": np.asarray(dec["pi_after"]).tolist(),
                                "emitted": bool(dec["emitted"]),
                                "ordered_reasons": reasons,
                                "audit": {
                                    "q_pi_audit": st["qref"].tolist(),
                                    "v_audit": st["v"].tolist(),
                                    "realized_sup_error": realized,
                                    "covers": bool(e_q is not None and float(e_q) >= realized),
                                    "safety_margin": (None if e_q is None
                                                      else float(e_q) - realized),
                                    "min_chosen_lb_over_updated": (
                                        float(np.min(dec["chosen_lb"][dec["mask"]]))
                                        if dec["mask"].any() else None),
                                },
                                "items_this_step": chains * CHAIN_LENGTH,
                                "batch_hash": bh,
                                "producer_gap_same_policy": float(np.max(np.abs(qh - qh_np_audit))),
                            }
                            if dec["emitted"]:
                                q1, v1 = solve_qpi_vpi(mdp, np.asarray(dec["pi_after"],
                                                                       dtype=np.float64), fam["gamma"])
                                dv = v1 - st["v"]
                                entry["audit"].update({
                                    "v_after": v1.tolist(),
                                    "value_delta": dv.tolist(),
                                    "componentwise_nondegrading": bool(float(np.min(dv)) >= 0.0),
                                    "total_value_gain": float(np.sum(dv)),
                                    "closure_fraction": float(np.sum(dv)) / gap_denom,
                                })
                                st["v"] = v1
                                st["qref"] = q1
                                st["pi"] = np.asarray(dec["pi_after"], dtype=np.float64)
                            wall["audit"] += time.time() - t_a
                            st["steps"].append(entry)
                            items_alone[cell] += chains * CHAIN_LENGTH

                            # Sections 6 and 7: an H1 anomaly or a point sitting on the
                            # numerical boundary stops ALL of this actor's formal sampling
                            # immediately, at the located step, so that the only work done
                            # afterwards is the read-only high-precision diagnosis.
                            loc = {"family": fam_name, "mixing": float(mixing),
                                   "task_index": int(task_index), "route": route,
                                   "producer": producer, "step": step}
                            m = entry["audit"]["safety_margin"]
                            if m is not None and not math.isfinite(m):
                                anomalies.append({**loc, "kind": "non_finite_margin", "value": m})
                            elif m is not None and m < 0.0:
                                anomalies.append({**loc, "kind": "h1_coverage_violation",
                                                  "e_q": e_q, "realized_sup_error": realized,
                                                  "margin": m})
                            elif m is not None and abs(m) <= NUMERIC_PAUSE:
                                anomalies.append({**loc, "kind": "numeric_boundary_pause",
                                                  "margin": m})
                            if dec["emitted"]:
                                dv = np.asarray(entry["audit"]["value_delta"], np.float64)
                                if not np.all(np.isfinite(dv)):
                                    anomalies.append({**loc, "kind": "non_finite_value_delta"})
                                elif float(np.min(dv)) < 0.0:
                                    anomalies.append({**loc, "kind": "h1_degrading_step",
                                                      "min_value_delta": float(np.min(dv))})
                            if anomalies:
                                _partial(out_dir, records, planned_records,
                                         {"aborted": True, "anomalies": anomalies})
                                (out_dir / "h1_anomaly.json").write_text(json.dumps(
                                    {"task_id": TASK_ID, "actor": ACTOR, "aborted": True,
                                     "anomalies": anomalies, "wall": wall,
                                     "location": loc,
                                     "next_step": ("section 6 read-only high-precision "
                                                   "recomputation from the same sealed "
                                                   "inputs; no re-run with a new seed")},
                                    indent=2, sort_keys=True), encoding="utf-8")
                                raise SystemExit(
                                    f"H1/numerical anomaly at {loc}: {anomalies[-1]} -- "
                                    f"all formal sampling stopped per task sheet sections 6/7")
                    routes[route] = {
                        c: {"steps": state[c]["steps"],
                            "emitted_steps": sum(1 for e in state[c]["steps"] if e["emitted"]),
                            "simulated_steps": len(state[c]["steps"]),
                            "v_final": state[c]["v"].tolist(),
                            "closure_fraction": float(np.sum(state[c]["v"]) - np.sum(v0)) / gap_denom,
                            "gap_denom": gap_denom}
                        for c in CELLS}
                records.append({"task_id": TASK_ID, "family": fam_name,
                                "mixing": float(mixing), "task_index": int(task_index),
                                "wall_seconds": time.time() - t0, "routes": routes})
                print(f"  {fam_name} {mixing}/{task_index} ({time.time() - t0:.1f}s)", flush=True)
                _partial(out_dir, records, planned_records)

    wall["total"] = time.time() - t_start
    bundle = {
        "task_id": TASK_ID, "actor": ACTOR, "baseline": BASELINE,
        "cells": list(CELLS), "producers": list(PRODUCERS), "routes": list(ROUTES),
        "families": ["f1", "f2"], "horizon": K, "chains": chains, "smoke": smoke,
        "chain_length": CHAIN_LENGTH, "salt": SALT, "seed": SEED,
        "min_visits": MIN_VISITS, "delta_total": DELTA_TOTAL, "delta_step": delta_step,
        "eta_grid": list(ETA_GRID), "record_count": len(records),
        "net_forwards": net_forwards,
        "cost": {"unique_batches": len(unique_batches), "items_unique": items_unique,
                 "items_if_run_alone": items_alone, "wall": wall,
                 "audit_recomputes": audit_recomputes,
                 "note": ("'audit_recompute' wall and count are the extra same-policy numpy "
                          "recompute used only by D4; they draw no new data and are excluded "
                          "from the producer and data-efficiency figures."),
                 "items_per_batch": chains * CHAIN_LENGTH},
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    (out_dir / "task_results.json").write_text(json.dumps(bundle, indent=2, sort_keys=True),
                                               encoding="utf-8")
    (out_dir / "input_manifest.json").write_text(json.dumps(
        {"task_id": TASK_ID, "seed": SEED, "salt": SALT, "records": manifest,
         "step_batch_hashes": {
             f"{r['family']}|{r['mixing']}|{r['task_index']}|{route}|{cell}|{e['step']}":
                 e["batch_hash"]
             for r in records for route, per_cell in r["routes"].items()
             for cell, payload in per_cell.items() for e in payload["steps"]},
         "run_metadata": run_metadata(sys.argv, started_utc),
         "note": "MDP/train/policy hashes are LF-normalised where text; arrays are hashed "
                 "with dtype/shape/bytes in native layout."}, indent=2, sort_keys=True),
        encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(
        {"task_id": TASK_ID, "actor": ACTOR, "baseline": BASELINE, "cells": list(CELLS),
         "horizon": K, "chains": chains, "chain_length": CHAIN_LENGTH, "salt": SALT,
         "delta_total": DELTA_TOTAL, "delta_step": delta_step, "min_visits": MIN_VISITS,
         "eta_grid": list(ETA_GRID), "smoke": smoke}, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(
        {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
         "torch": torch.__version__,
         "threads": run_metadata(sys.argv, started_utc)["threads"]},
        indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"records": len(records), "net_forwards": net_forwards,
                      "audit_recomputes": audit_recomputes,
                      "items_unique": items_unique,
                      "wall_total_s": round(wall["total"], 1)}, indent=2))


def _residuals(q_hat, policy, batch, gamma):
    st = np.asarray(batch["states"], np.int64)
    ac = np.asarray(batch["actions"], np.int64)
    rw = np.asarray(batch["rewards"], np.float64)
    ns = np.asarray(batch["next_states"], np.int64)
    pi = np.asarray(policy, np.float64)
    return rw + gamma * (pi[ns] * q_hat[ns]).sum(axis=1) - q_hat[st, ac]


if __name__ == "__main__":
    main()
