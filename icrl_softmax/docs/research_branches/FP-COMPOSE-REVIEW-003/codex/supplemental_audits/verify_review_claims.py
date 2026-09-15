"""Check each claim in the reviewer's report against the sealed data. Verify, then fix."""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
B = PROJECT / "results/FP-COMPOSE-002/claude/formal/task_results.json"
A = PROJECT / "results/FP-COMPOSE-002/claude/formal/analysis.json"
b = json.loads(B.read_text(encoding="utf-8"))
a = json.loads(A.read_text(encoding="utf-8"))

print("### issue 1: tail contribution indices")
m = a["H3"]["mean_closure_by_step"]
print("  mean_closure_by_step[0..11] =", [round(x, 4) for x in m])
print("  index 0 is step 1, so mean[i] is the value AFTER step i+1")
print("  my labels:  k=5..12 := mean[11]-mean[4] = %.4f  (actually steps 6..12)"
      % ((m[11] - m[4]) * 100))
print("              k=6..12 := mean[11]-mean[5] = %.4f  (actually steps 7..12)"
      % ((m[11] - m[5]) * 100))
print("  correct:    steps 6..12 := mean[11]-mean[4] = %.4f points" % ((m[11] - m[4]) * 100))
print("              steps 5..12 := mean[11]-mean[3] = %.4f points" % ((m[11] - m[3]) * 100))
print("  reviewer said 4.309 and 7.839 -> %s"
      % ("CONFIRMED" if abs((m[11] - m[4]) * 100 - 4.309) < 0.01
         and abs((m[11] - m[3]) * 100 - 7.839) < 0.01 else "MISMATCH"))

print()
print("### issue 4: is '192/192 positive every step' cumulative or per-step?")
# recompute per-step increments per route-record
inc = {}
for rec in b["records"]:
    for route, cs in rec["routes"].items():
        curves = []
        for c, s in cs.items():
            denom = s["gap_denom"]
            v0 = np.asarray(s["steps"][0]["audit"]["v_audit"], float).sum()
            cur, out = 0.0, []
            for e in s["steps"]:
                if e["emitted"]:
                    cur = (np.asarray(e["audit"]["v_after"], float).sum() - v0) / denom
                out.append(cur)
            out += [cur] * (12 - len(out))
            curves.append(np.asarray(out))
        row = np.mean(curves, axis=0)
        for k in range(1, 12):
            inc.setdefault(k + 1, []).append(float(row[k] - row[k - 1]))
print("  step  per-record positive increments / n")
for k in sorted(inc):
    v = np.asarray(inc[k])
    print("   %2d    %3d / %d   (fraction %.4f)" % (k, int((v > 0).sum()), v.size,
                                                    float((v > 0).mean())))
print("  reviewer said step 6 -> 191/192 and step 12 -> 180/192 -> %s"
      % ("CONFIRMED" if int((np.asarray(inc[6]) > 0).sum()) == 191
         and int((np.asarray(inc[12]) > 0).sum()) == 180 else "MISMATCH"))

print()
print("### issue 3: the task313 counterexample")
found = False
for rec in b["records"]:
    if rec["family"] == "f2" and abs(rec["mixing"] - 0.2) < 1e-9 and rec["task_index"] == 313:
        for route, cs in rec["routes"].items():
            for c, s in cs.items():
                if route != "expected_finite" or not c.startswith("numpy"):
                    continue
                last = s["steps"][-1]
                dv = np.asarray(last["audit"]["value_delta"], float)
                print("  route=%s cell=%s steps=%d closure=%.4f gap_denom=%.4f"
                      % (route, c, len(s["steps"]), s["closure_fraction"], s["gap_denom"]))
                print("  final step %d: min increment %.6e  emitted=%s"
                      % (last["step"], dv.min(), last["emitted"]))
                print("  -> still closed only %.2f%% of the initial gap, %.2f%% remains"
                      % (100 * s["closure_fraction"], 100 * (1 - s["closure_fraction"])))
                found = True
print("  counterexample found:", found)

print()
print("### issue 3b: does a tiny MIN increment imply a small TOTAL increment?")
x = []
for rec in b["records"]:
    for route, cs in rec["routes"].items():
        for c, s in cs.items():
            for e in s["steps"]:
                if not e["emitted"]:
                    continue
                dv = np.asarray(e["audit"]["value_delta"], float)
                x.append((float(dv.min()), float(dv.sum()), float(e["audit"]["total_value_gain"])))
x = np.asarray(x)
tiny = x[x[:, 0] <= 1e-6]
print("  steps with min-increment <= 1e-6: %d" % tiny.shape[0])
print("  their TOTAL gain: median %.3e  min %.3e  max %.3e"
      % (np.median(tiny[:, 1]), tiny[:, 1].min(), tiny[:, 1].max()))
print("  how many of them have total gain > 1e-4: %d / %d"
      % (int((tiny[:, 1] > 1e-4).sum()), tiny.shape[0]))

print()
print("### record error: unique batches")
print("  items_unique / items_per_batch = %.3f -> %d batches"
      % (b["cost"]["items_unique"] / b["cost"]["items_per_batch"],
         round(b["cost"]["items_unique"] / b["cost"]["items_per_batch"])))
print("  report said 1152 (that is the UPPER LIMIT 96*12); reviewer said 1136")
print("  cost['unique_batches'] field =", b["cost"]["unique_batches"])

print()
print("### reviewer's positive checks")
print("  steps evaluated = %d (reviewer 4534) -> %s"
      % (a["H1"]["evaluated_steps"], a["H1"]["evaluated_steps"] == 4534))
print("  coverage violations = %d, degrading = %d (reviewer 0/0)"
      % (a["H1"]["coverage_violations"], a["H1"]["componentwise_degrading_steps"]))
print("  H0 identical steps = %d (reviewer 1536)" % a["H0"]["identical_steps"])
print("  H4 mismatched = %d (reviewer 192/192 identical)" % a["H4"]["mismatched"])
print("  final mean closure = %.4f%% (reviewer 89.3123%%)"
      % (100 * a["H3"]["mean_closure_by_step"][-1]))
