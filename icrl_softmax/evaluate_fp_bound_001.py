"""FP-BOUND-001 evaluator: price the three sound tightenings on FRESH environments.

The confirmatory population is `task_index 12..23` x `mixing in {0.08, 0.5}` = the
`24` environments `FP-EARLYSTOP-001` opened and this task's pilot never touched,
so the pre-registered bands are genuinely out-of-sample. All levers are computed
on the SAME certification batch, so the comparison is paired and carries no
sampler confound.

Levers (see `fixed_policy_tight_certificate.py` for the derivations):

    frozen          the current MP certificate                      (model-free)
    L1              known-Qhat envelope                             (model-free)
    L12             L1 + the full risk budget                       (model-free)
    L123            L12 + exact propagation                         (NEEDS the kernel)
    support_range   L1 + the oracle support range                   (reference only)

Nothing about the decision rule, the eta grid or `delta_total` is changed: every
lever produces a scalar `E_Q` and is handed to the frozen `fs.improvement_for`.

Also recorded per record-route: the exact gate margin `h = max_eta min_s
I_s/||dpi_s||_1`, so the analyzer can express every lever on the decision's own
scale as `E_Q/h` (the FP-SHORT-001 yardstick).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa as es  # noqa: E402
import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_tight_certificate as tc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402

TASK_ID = "FP-BOUND-001"
TASK_SALT = 90417
DELTA_TOTAL = 0.05
CHAIN_LENGTH = 64
FRESH_TASKS = tuple(range(12, 24))  # FP-EARLYSTOP-001's environments
MIN_VISITS = 2000
LEVERS = ("frozen", "L1", "L12", "L123", "support_range")

SEALED_FILES = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "fixed_policy_bernstein_certificate.py",
    "fixed_policy_mp_certificate.py",
    "model.py",
)
NEW_FILES = (
    "fixed_policy_tight_certificate.py",
    "evaluate_fp_bound_001.py",
    "diagnose_fp_bound_slack.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def gate_margin(policy: np.ndarray, q_hat: np.ndarray) -> float:
    """Exact `h = max_eta min_s I_s/||dpi_s||_1`; emission is exactly `E_Q < h`.

    A state with `dpi_s == 0` contributes a hard zero: its `LB_s` is exactly 0,
    which fails the strict `min_s LB_s > 0`, so that eta can never pass.
    """
    best = 0.0
    for eta in fs.ETA_CANDIDATES:
        cand = es.relative_softmax_candidate(policy, q_hat, eta)
        dpi = cand - policy
        i_hat = (dpi * q_hat).sum(axis=1)
        l1 = np.abs(dpi).sum(axis=1)
        ratios = np.where(l1 > 0.0, i_hat / np.where(l1 > 0.0, l1, 1.0), 0.0)
        best = max(best, float(np.min(ratios)))
    return best


def run(args) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for mixing in fs.MIXING:
        for task_index in FRESH_TASKS:
            mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact0 = policy_quantities(mdp, behaviour)
            mu = np.asarray(exact0["mu_state"], dtype=np.float64)
            train = fs.training_batch(mdp, behaviour, mu, rng)
            q_pi = np.asarray(exact0["q_pi"], dtype=np.float64)
            routes: dict[str, Any] = {}
            for route in ("expected_exact", "expected_finite"):
                q_hat = np.asarray(
                    fs.run_route(route, behaviour, train)["q_hat"], dtype=np.float64
                ).reshape(fs.N_STATES, fs.N_ACTIONS)
                raw = vectorised_batch(
                    mdp,
                    behaviour,
                    mu,
                    step_seed_parts(fs.SEED, TASK_SALT, mixing, task_index, 1),
                    int(args.chains),
                    CHAIN_LENGTH,
                )
                reduced, counts = first_visit_batch(raw, CHAIN_LENGTH)
                realized = float(np.max(np.abs(q_hat - q_pi)))
                h = gate_margin(behaviour, q_hat)
                arms: dict[str, Any] = {}
                for lever in LEVERS:
                    cert = tc.certificate(
                        q_hat,
                        behaviour,
                        reduced,
                        min_visits=MIN_VISITS,
                        delta_step=DELTA_TOTAL,
                        lever=lever,
                        transition=np.asarray(mdp["P"], dtype=np.float64),
                        reward=np.asarray(mdp["R"], dtype=np.float64),
                    )
                    decision = fs.improvement_for(behaviour, q_hat, cert)
                    e_q = cert["e_q"]
                    arms[lever] = {
                        "status": decision["status"],
                        "certificate_status": cert["status"],
                        "failure_reasons": cert["failure_reasons"],
                        "e_q": e_q,
                        "epsilon_res": float(np.max(cert["epsilon_res_by_pair"]))
                        if cert["status"] == "certificate_emitted"
                        else None,
                        "delta_each": cert["delta_each"],
                        "y_range": cert["y_range"],
                        "eta_selected": decision["eta_selected"],
                        "min_lb": float(np.min(decision["lb_by_state"])),
                        "emitted": decision["status"] == "safe_update_emitted",
                        "covers_realized": bool(e_q is not None and float(e_q) >= realized),
                        "e_q_over_h": (
                            float(e_q) / h if e_q is not None and h > 0 else None
                        ),
                        "pair_sizes": cert["pair_sizes"].tolist(),
                        "residual_means": cert["residual_means"].tolist(),
                        "radii": cert["radii"].tolist(),
                        "uniform_bound": (
                            cert["propagation"]["uniform_bound"]
                            if cert.get("propagation")
                            else None
                        ),
                    }
                    if lever == "L123" and cert.get("propagation"):
                        arms[lever]["propagation"] = cert["propagation"]
                routes[route] = {
                    "q_hat": q_hat.tolist(),
                    "realized_q_sup_error": realized,
                    "gate_margin_h": h,
                    "pair_counts": counts.tolist(),
                    "arms": arms,
                }
            records.append(
                {
                    "task_id": TASK_ID,
                    "mixing": float(mixing),
                    "task_index": int(task_index),
                    "routes": routes,
                }
            )
    return {
        "task_id": TASK_ID,
        "label": args.label,
        "fresh_tasks": list(FRESH_TASKS),
        "mixings": list(fs.MIXING),
        "record_count": len(records),
        "chains": int(args.chains),
        "chain_length": CHAIN_LENGTH,
        "min_visits": MIN_VISITS,
        "delta_step": DELTA_TOTAL,
        "levers": list(LEVERS),
        "task_salt": TASK_SALT,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--chains", type=int, required=True)
    args = parser.parse_args()

    out = run(args)
    out["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_results.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "label": args.label,
                "chains": int(args.chains),
                "chain_length": CHAIN_LENGTH,
                "min_visits": MIN_VISITS,
                "delta_step": DELTA_TOTAL,
                "fresh_tasks": list(FRESH_TASKS),
                "mixings": list(fs.MIXING),
                "levers": list(LEVERS),
                "sealed_file_hashes": {n: sha256(PROJECT / n) for n in SEALED_FILES},
                "new_file_hashes": {n: sha256(PROJECT / n) for n in NEW_FILES},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps(
            {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "task_id": TASK_ID,
        "records": out["record_count"],
        "chains": int(args.chains),
        "emitted_by_lever": {
            lever: sum(
                1
                for r in out["records"]
                for route in r["routes"].values()
                if route["arms"][lever]["emitted"]
            )
            for lever in LEVERS
        },
        "mean_e_q_by_lever": {
            lever: float(
                np.mean(
                    [
                        route["arms"][lever]["e_q"]
                        for r in out["records"]
                        for route in r["routes"].values()
                        if route["arms"][lever]["e_q"] is not None
                    ]
                )
            )
            for lever in LEVERS
        },
        "coverage_violations_by_lever": {
            lever: sum(
                1
                for r in out["records"]
                for route in r["routes"].values()
                if route["arms"][lever]["e_q"] is not None
                and not route["arms"][lever]["covers_realized"]
            )
            for lever in LEVERS
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
