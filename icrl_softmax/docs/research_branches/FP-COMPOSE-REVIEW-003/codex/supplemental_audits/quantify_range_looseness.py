"""Quantify how loose the FROZEN L12 range formula is against the tighter span bound.

Read-only analysis of the sealed FP-COMPOSE-001 bundle. No new arm is run: this reports an
algebraic property of two bounds, not a counterfactual decision.
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from evaluate_fp_xfam_001 import FAMILIES, build_family_task  # noqa: E402

b = json.loads((PROJECT / "results/FP-COMPOSE-001/claude/formal/task_results.json")
               .read_text(encoding="utf-8"))
rows = []
for rec in b["records"]:
    fam = FAMILIES[rec["family"]]
    Rstar, gam = fam["reward_bound"], fam["gamma"]
    mdp, beh, rng = build_family_task(fam, float(rec["mixing"]), int(rec["task_index"]))
    R = np.asarray(mdp["R"], float)
    for route, cells in rec["routes"].items():
        for c, s in cells.items():
            for e in s["steps"]:
                if e["pair_sizes"] is None:
                    continue
                pi = np.asarray(e["pi_before"], float)
                qh = np.asarray(e["q_hat"], float)
                vh = np.einsum("sa,sa->s", pi, qh)
                spanV = float(vh.max() - vh.min())
                Y = R + gam * vh[None, None, :] - qh[:, :, None]
                rows.append(dict(
                    fam=rec["family"], step=int(e["step"]), Rstar=Rstar, gam=gam,
                    spanV=spanV, rngY=float((Y.max(axis=2) - Y.min(axis=2)).max()),
                    mid=2 * Rstar + gam * spanV, proof=2 * Rstar / (1 - gam),
                    sup2=2 * float(np.abs(Y).max()), code=float(e["y_range"])))

print("=== frozen L12 y_range vs the tighter span bound the proof doc already derives ===")
for fam in ("f1", "f2"):
    m = [r for r in rows if r["fam"] == fam]
    code = np.array([r["code"] for r in m])
    mid = np.array([r["mid"] for r in m])
    rngY = np.array([r["rngY"] for r in m])
    proof = m[0]["proof"]
    steps = np.array([r["step"] for r in m])
    print()
    print("%s  R*=%.1f  gamma=%.2f   proof final bound 2R*/(1-g)=%.3f"
          % (fam, m[0]["Rstar"], m[0]["gam"], proof))
    print("  exact range(Y)                    max %.3f   violations of proof bound: %d"
          % (rngY.max(), int((rngY > proof + 1e-9).sum())))
    print("  tighter span bound 2R*+g*span(V)  median %.3f  max %.3f  violations: %d"
          % (np.median(mid), mid.max(), int((rngY > mid + 1e-9).sum())))
    print("  frozen code y_range               median %.3f  max %.3f"
          % (np.median(code), code.max()))
    print("  LOOSENESS code / tighter          median %.2fx  max %.2fx"
          % (np.median(code / mid), (code / mid).max()))
    by_step = [round(float(np.median(code[steps == s] / mid[steps == s])), 2)
               for s in (1, 2, 3, 4)]
    print("  by step 1..4                      %s" % by_step)

print()
print("=== share of the L12 radius contributed by the range term, at the argmax pair ===")
bern_l, yrt_l = [], []
for rec in b["records"]:
    for route, cells in rec["routes"].items():
        for c, s in cells.items():
            for e in s["steps"]:
                if e["pair_sizes"] is None:
                    continue
                n = np.asarray(e["pair_sizes"], float)
                v = np.asarray(e["pair_vars"], float)
                eps = np.asarray(e["pair_eps"], float)
                lt = np.log(2.0 / e["delta_dir"])
                x = int(np.argmax(eps))
                bern_l.append(float(np.sqrt(2 * v * lt / n)[x]))
                yrt_l.append(float(((7 / 3) * e["y_range"] * lt / np.maximum(n - 1, 1))[x]))
bern = np.array(bern_l)
yrt = np.array(yrt_l)
print("Bernstein term median %.5f | range term median %.5f | range share of radius median %.3f"
      % (np.median(bern), np.median(yrt), np.median(yrt / (bern + yrt))))
print("steps where range term > Bernstein term: %.3f" % float((yrt > bern).mean()))
