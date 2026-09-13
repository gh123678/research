"""Review #2 auxiliaries: conservatism of the repaired certificate, and the
independence of the retained count N from the retained values on real data.

(a) realized ||Qhat - Q^pi||_inf / E_Q for every EMITTED step of the sealed
    first-visit rungs: how much headroom the certificate actually had.
(b) max |corr(N_x, mean_x)| across replications, from the coverage probe.
"""

from __future__ import annotations

import json
from pathlib import Path

def _repo_root() -> Path:
    for cand in Path(__file__).resolve().parents:
        if (cand / "icrl_softmax" / "fixed_policy_expected_sarsa_scaled.py").exists():
            return cand
    raise RuntimeError("repository root not found")


ROOT = _repo_root()
BASE = ROOT / "icrl_softmax" / "results" / "FP-CERTFIX-001" / "claude"
TMP = ROOT / "tmp"


def conservatism():
    print("(a) realized sup-error / E_Q over emitted steps (sealed first-visit rungs)")
    for rung in ("step1_fv_c16k", "step1_fv_c64k", "multi_fv", "step1_n16k", "step1_n64k"):
        path = BASE / rung / "task_results.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text(encoding="utf-8"))
        ratios, e_qs, rho_ratios = [], [], []
        for rec in res["records"]:
            for route in res["routes"]:
                for arm in res["arms"]:
                    if arm != "mp":
                        continue
                    for s in rec["routes"][route][arm]["steps"]:
                        if s["status"] != "safe_update_emitted":
                            continue
                        r = s["oracle_audit"]["realized_q_sup_error_vs_current_target_pi"]
                        ratios.append(r / s["e_q"])
                        e_qs.append(s["e_q"])
                        rho_ratios.append(s["e_q"] / max(r, 1e-300))
        if not ratios:
            continue
        ratios.sort()
        e_qs.sort()
        n = len(ratios)
        print(
            f"  {rung:<14} n={n:>4}  realized/E_Q: min {ratios[0]:.4f}  median {ratios[n // 2]:.4f}"
            f"  max {ratios[-1]:.4f}   E_Q median {e_qs[n // 2]:.4f}"
            f"   max(E_Q/realized) {max(rho_ratios):.1f}x"
        )


def drift():
    print()
    print("(b) independence of the retained count N_x from the retained values")
    path = TMP / "review2_lemma_A_prime.json"
    if not path.exists():
        print("  coverage probe output missing")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data.get("P2P3_real", []):
        dr = row["drift_by_pair"]
        cors = [abs(v["corr_n_mean"]) for v in dr.values() if v["corr_n_mean"] == v["corr_n_mean"]]
        nmin = min(v["n_min"] for v in dr.values())
        nmax = max(v["n_max"] for v in dr.values())
        print(
            f"  mixing={row['mixing']} task={row['task_index']:>2}  "
            f"N in [{nmin}, {nmax}]  max|corr(N, mean)| = {max(cors):.3f}"
            f"   (12 pairs, {row['reps']} replications)"
        )


if __name__ == "__main__":
    conservatism()
    drift()
