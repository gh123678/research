"""Characterise the tail value deltas properly, instead of generalising from 12 points.

first_result.md claimed the seam "fades" at the tail. All 12 near-zero points sit in ONE
task (f2/0.2/task 303) at steps 11-12, so that phrasing generalises from 0.27 percent of the
emitted steps in 1 of 96 task-instances. This computes the actual distribution so the claim
can be stated at the right scope.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
B = PROJECT / "results/FP-COMPOSE-002/claude/formal/task_results.json"
b = json.loads(B.read_text(encoding="utf-8"))

per_step_min = []          # per emitted step: the smallest value delta over states
per_step_max = []
cells = set()
near = []
for rec in b["records"]:
    for route, cs in rec["routes"].items():
        for c, s in cs.items():
            hit = False
            for e in s["steps"]:
                if not e["emitted"]:
                    continue
                dv = np.asarray(e["audit"]["value_delta"], float)
                per_step_min.append(float(dv.min()))
                per_step_max.append(float(dv.max()))
                if float(dv.min()) <= 1e-8:
                    near.append((rec["family"], rec["mixing"], rec["task_index"], route, c,
                                 e["step"], float(dv.min())))
                    hit = True
            if hit:
                cells.add((rec["family"], rec["mixing"], rec["task_index"]))

mn = np.asarray(per_step_min)
mx = np.asarray(per_step_max)
print("emitted steps with a value delta: %d" % mn.size)
print("min over steps  : %.3e   median %.3e   max %.3e" % (mn.min(), np.median(mn), mn.max()))
print()
print("how many emitted steps have min-value-delta below a threshold:")
for t in (1e-6, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
    k = int((mn <= t).sum())
    print("   <= %.0e : %5d  (%.3f%%)" % (t, k, 100.0 * k / mn.size))
print()
print("task-instances containing at least one near-zero point: %d of %d"
      % (len(cells), 96))
print("distinct (task, route, cell) groups affected: %d"
      % len({(a, b_, c_, d, e_) for a, b_, c_, d, e_, f_, g_ in near}))
print()
print("distribution of the per-step MIN delta by step (median across cells):")
steps = np.arange(1, 13)
# rebuild with step labels
bystep = {k: [] for k in steps}
bystep_max = {k: [] for k in steps}
for rec in b["records"]:
    for route, cs in rec["routes"].items():
        for c, s in cs.items():
            for e in s["steps"]:
                if not e["emitted"]:
                    continue
                dv = np.asarray(e["audit"]["value_delta"], float)
                bystep[e["step"]].append(float(dv.min()))
                bystep_max[e["step"]].append(float(dv.max()))
for k in steps:
    v = np.asarray(bystep[k]); w = np.asarray(bystep_max[k])
    print("  step %2d  n=%4d  min %.3e  p50 %.3e  max %.3e  frac<=1e-6 %.4f"
          % (k, v.size, v.min(), np.median(v), w.max(),
             float((v <= 1e-6).mean()) if v.size else 0.0))
