"""FP-BOUND-002 evaluator: the model-free propagation arm (L12M), same protocol.

Runs on exactly the same environments, seeds and batch schedule as FP-BOUND-001,
with one arm added. The four shared arms therefore must reproduce FP-BOUND-001's
sealed bundles number-for-number, which is checked by
`verify_fp_bound_002.py` -- so the new arm is compared against arms that are
themselves cross-validated rather than re-derived.

Shared helpers are imported from `evaluate_fp_bound_001` on purpose: that file is
left byte-identical so its recorded `new_file_hashes` stay valid.

    python evaluate_fp_bound_002.py --output-dir results/FP-BOUND-002/claude/c64k \
        --label c64k --chains 65536
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_tight_certificate as tc  # noqa: E402
from evaluate_fp_bound_001 import (  # noqa: E402
    CHAIN_LENGTH,
    DELTA_TOTAL,
    FRESH_TASKS,
    MIN_VISITS,
    SEALED_FILES,
    TASK_SALT,
    gate_margin,
    sha256,
)
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402

TASK_ID = "FP-BOUND-002"
LEVERS = ("frozen", "L1", "L12", "L123", "L12M")
NEW_FILES = (
    "fixed_policy_tight_certificate.py",
    "evaluate_fp_bound_002.py",
    "evaluate_fp_bound_001.py",
    "diagnose_fp_bound_slack.py",
)


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
                        "e_q_over_h": (float(e_q) / h if e_q is not None and h > 0 else None),
                        "pair_sizes": cert["pair_sizes"].tolist(),
                        "residual_means": cert["residual_means"].tolist(),
                        "radii": cert["radii"].tolist(),
                        "uniform_bound": (
                            cert["propagation"]["uniform_bound"]
                            if cert.get("propagation")
                            else None
                        ),
                    }
                    if cert.get("propagation"):
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
