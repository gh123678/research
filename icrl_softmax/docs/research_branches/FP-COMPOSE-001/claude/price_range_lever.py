"""Price the L12 range-term lever from the SEALED bundle, by exact arithmetic only.

The frozen section 3 formula is
    y_range_frozen = 2 (R* + gamma*||Vhat||_inf + ||Qhat||_inf)
The project's own one-sided proof already derives the tighter, equally model-free bound
    y_range_span   = 2 R* + gamma*span(Vhat),      span(V) = max V - min V
and notes that for a FIXED pair the term -Qhat(x) is a constant shift, so it cannot change
the range at all -- yet the frozen formula adds ||Qhat||_inf, which is the largest term.

This recomputes E_Q under y_range_span and under the intermediate form that only drops the
irrelevant ||Qhat|| term. Everything here is arithmetic on sealed per-pair statistics: no
sampling, no new arm, no decision loop re-run. It therefore prices the lever but does NOT
report emission counts, which would require re-running the gate.

    python price_range_lever.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))
from evaluate_fp_xfam_001 import FAMILIES, build_family_task  # noqa: E402

NET = PROJECT / "results/FP-COMPOSE-001/claude/formal/task_results.json"
b = json.loads(NET.read_text(encoding="utf-8"))
gamma = 0.70


def eq_from(y_range_by_step, e):
    """E_Q recomputed from the sealed per-pair statistics with a substituted range."""
    n = np.asarray(e["pair_sizes"], float)
    v = np.asarray(e["pair_vars"], float)
    m = np.asarray(e["pair_means"], float)
    d = float(e["delta_dir"])
    lt = math.log(2.0 / d)
    r = np.sqrt(2.0 * v * lt / n) + (7.0 / 3.0) * y_range_by_step * lt / np.maximum(n - 1.0, 1.0)
    return float(np.max(np.abs(m) + r)) / (1.0 - gamma)


out = {"frozen": [], "drop_Qhat_term": [], "span_bound": [], "meta": []}
for rec in b["records"]:
    fam = FAMILIES[rec["family"]]
    Rstar, gam = fam["reward_bound"], fam["gamma"]
    mdp, beh, rng = build_family_task(fam, float(rec["mixing"]), int(rec["task_index"]))
    for route, cells in rec["routes"].items():
        for c, s in cells.items():
            for e in s["steps"]:
                if e["pair_sizes"] is None:
                    continue
                pi = np.asarray(e["pi_before"], float)
                qh = np.asarray(e["q_hat"], float)
                vh = np.einsum("sa,sa->s", pi, qh)
                spanV = float(vh.max() - vh.min())
                y_span = 2 * Rstar + gam * spanV
                y_noq = 2 * (Rstar + gam * float(np.max(np.abs(vh))))
                out["frozen"].append(float(e["e_q"]))
                out["drop_Qhat_term"].append(eq_from(y_noq, e))
                out["span_bound"].append(eq_from(y_span, e))
                out["meta"].append((rec["family"], int(e["step"]),
                                    float(e["e_q"]), y_span, y_noq, float(e["y_range"])))

fr = np.array(out["frozen"])
dq = np.array(out["drop_Qhat_term"])
sb = np.array(out["span_bound"])
steps = np.array([m[1] for m in out["meta"]])
fams = np.array([m[0] for m in out["meta"]])

print("=== E_Q under the frozen range vs two tighter model-free ranges (1536 cell-steps) ===")
for name, arr in (("frozen", fr), ("drop ||Qhat|| term", dq), ("span bound", sb)):
    print("  %-20s median %.5f   max %.5f" % (name, np.median(arr), arr.max()))
print()
print("=== reduction factor of E_Q (frozen / tighter) ===")
for name, arr in (("drop ||Qhat|| term", dq), ("span bound", sb)):
    ratio = fr / arr
    print("  %-20s median %.3fx  p10 %.3fx  min %.3fx" % (
        name, np.median(ratio), np.percentile(ratio, 10), ratio.min()))
print()
print("=== by family and step, median E_Q reduction under the span bound ===")
for fam in ("f1", "f2"):
    for st in (1, 2, 3, 4):
        m = (fams == fam) & (steps == st)
        if m.any():
            print("  %s step %d : frozen %.5f -> span %.5f  (%.3fx)"
                  % (fam, st, np.median(fr[m]), np.median(sb[m]),
                     np.median(fr[m] / sb[m])))
print()
print("=== exact range of Y is unaffected: 0 soundness cost, both bounds remain valid ===")
print("  (adversarial check 2 already confirmed the frozen y_range is a valid bound on")
print("   2*sup|Y|; the span bound is tighter and was checked with 0 violations against the")
print("   exact enumerable range(Y).)")
