"""Independent replay of the sealed FP-CERTFIX-001 rungs (review #2).

No re-sampling: this reads each sealed ``task_results.json`` and re-derives, from
the stored per-pair certificate numbers plus a DETERMINISTIC reconstruction of
``q_hat``, (a) the certificate's E_Q, (b) the frozen decision (eta selection,
emission, LB values), and (c) the risk accounting. Then it compares everything
against what the sealed file claims.

The reconstruction of q_hat is itself checked against the sealed
``oracle_audit.realized_q_sup_error_vs_current_target_pi`` field, so a wrong
reconstruction cannot silently "agree" with the sealed decisions.

Written by the designated independent reviewer; does not import or call
``verify_fp_certfix_001.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

def _repo_root() -> Path:
    for cand in Path(__file__).resolve().parents:
        if (cand / "icrl_softmax" / "fixed_policy_expected_sarsa_scaled.py").exists():
            return cand
    raise RuntimeError("repository root not found")


ROOT = _repo_root()
PROJ = ROOT / "icrl_softmax"
sys.path.insert(0, str(PROJ))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402

RUNGS = [
    "smoke_fv", "step1_fv_c16k", "step1_fv_c64k", "multi_fv",
    "step1_ok_c64k",
    "smoke", "step1_n16k", "step1_n64k", "multi_n16k",
]
BASE = PROJ / "results" / "FP-CERTFIX-001" / "claude"


def main() -> int:
    print("independent replay of sealed FP-CERTFIX-001 rungs")
    print("=" * 90)
    bad = 0
    for rung in RUNGS:
        d = BASE / rung
        if not (d / "task_results.json").exists():
            print(f"{rung:<16} MISSING")
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        res = json.loads((d / "task_results.json").read_text(encoding="utf-8"))
        extraction = res.get("extraction", cfg.get("extraction", "first_n"))
        delta_step = float(res["delta_step"])
        emitted = {"mp": 0, "split": 0}
        n_rec = 0
        max_e_q_rel = 0.0
        max_realized_abs = 0.0
        decision_mismatch = 0
        lb_mismatch = 0
        risk_bad = 0
        eta_mismatch = 0
        viol = 0
        for rec in res["records"]:
            mixing = float(rec["mixing"])
            task_index = int(rec["task_index"])
            mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact0 = policy_quantities(mdp, behaviour)
            mu_state = np.asarray(exact0["mu_state"], dtype=np.float64)
            train = fs.training_batch(mdp, behaviour, mu_state, rng)
            q_pi = np.asarray(exact0["q_pi"], dtype=np.float64)
            n_rec += 1
            for route in res["routes"]:
                q_hat = np.asarray(
                    fs.run_route(route, behaviour, train)["q_hat"], dtype=np.float64
                ).reshape(fs.N_STATES, fs.N_ACTIONS)
                for arm in res["arms"]:
                    steps = rec["routes"][route][arm]["steps"]
                    if not steps:
                        continue
                    s0 = steps[0]
                    stored_realized = s0["oracle_audit"][
                        "realized_q_sup_error_vs_current_target_pi"
                    ]
                    realized = float(np.max(np.abs(q_hat - q_pi)))
                    max_realized_abs = max(
                        max_realized_abs, abs(realized - stored_realized)
                    )
                    if s0["status"] != "safe_update_emitted":
                        continue
                    means = np.asarray(s0["cert_means"], dtype=np.float64)
                    radii = np.asarray(s0["cert_radii"], dtype=np.float64)
                    e_q_re = float(np.max(np.abs(means) + radii)) / (1.0 - fs.GAMMA)
                    stored_e_q = float(s0["e_q"])
                    max_e_q_rel = max(max_e_q_rel, abs(e_q_re - stored_e_q) / stored_e_q)
                    dec = fs.improvement_for(
                        behaviour, q_hat, {"status": "certificate_emitted", "e_q": e_q_re}
                    )
                    if dec["status"] != s0["status"]:
                        decision_mismatch += 1
                        continue
                    if dec["eta_selected"] != s0["eta_selected"]:
                        eta_mismatch += 1
                    lb = np.asarray(dec["lb_by_state"], dtype=np.float64)
                    if abs(float(lb.min()) - float(s0["min_lb"])) > 1e-9:
                        lb_mismatch += 1
                    if s0["oracle_audit"]["certificate_violation"]:
                        viol += 1
                    emitted[arm] += 1
                    # risk accounting
                    de = float(s0["delta_each"])
                    if arm == "mp":
                        ok = abs(2.0 * 12 * de - delta_step) < 1e-15
                    else:
                        ok = abs(12 * 2.0 * de - delta_step) < 1e-15
                    if not ok:
                        risk_bad += 1
        print(
            f"{rung:<16} extraction={extraction:<11} records={n_rec:>3}  "
            f"step1 emitted mp={emitted['mp']:>2} split={emitted['split']:>2}  "
            f"max|dE_Q|rel={max_e_q_rel:.2e}  max|d realized|={max_realized_abs:.2e}"
        )
        print(
            f"{'':<16} decision mismatch={decision_mismatch} eta mismatch={eta_mismatch} "
            f"lb mismatch={lb_mismatch} risk accounting bad={risk_bad} "
            f"certificate violations={viol}"
        )
        if decision_mismatch or eta_mismatch or lb_mismatch or risk_bad or viol:
            bad += 1
        if max_e_q_rel > 1e-12 or max_realized_abs > 1e-12:
            bad += 1
    print("=" * 90)
    print(f"rungs with any discrepancy: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
