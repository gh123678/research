"""Why does the mixing setting dominate FP-SCALE-002 emission? (read-only)

The sealed formal bundle emits at 19/24 for mixing 0.08 and 3/24 for mixing 0.5.
This diagnostic asks whether that is explained by two measurable quantities the
sealed records already carry:

  need  : the certificate must be tight enough, i.e. E_Q below the within-state
          value spread of the limiting state;
  margin: the tilt must produce a positive improvement, i.e. some state's
          Ihat_s divided by its TV_s must exceed E_Q.

If low mixing simply widens the value spread relative to E_Q, the effect is
explained and needs no new task. If it is not explained, the effect is a genuine
open question.

Read-only: reads the sealed bundle, writes nothing, runs no experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent / "icrl_softmax"
BUNDLE = PROJECT / "results" / "FP-SCALE-002" / "claude" / "formal" / "task_results.json"
PRIMARY = ("variance_adaptive_exact", "variance_adaptive_finite")

bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))

rows = []
for record in bundle["records"]:
    spreads = np.asarray(record["oracle_audit_only"]["within_state_q_spread"], dtype=float)
    min_spread = float(spreads.min())
    means = np.asarray(record["oracle_audit_only"]["q_pi"], dtype=float)  # placeholder
    del means
    for name in PRIMARY:
        entry = record["routes"][name]
        e_q = entry["e_q"]
        if e_q is None:
            continue
        lb = np.asarray(entry["lb_by_state"], dtype=float)
        rows.append(
            {
                "mixing": float(record["mixing"]),
                "task": int(record["task_index"]),
                "route": name,
                "emitted": bool(entry["update_emitted"]),
                "e_q": float(e_q),
                "min_spread": min_spread,
                "spread_over_e_q": min_spread / float(e_q),
                "eta": entry["eta_selected"],
                "best_lb": float(lb.max()),
                "min_lb": float(lb.min()),
            }
        )

print(f"sealed primary route-records with a certificate: {len(rows)}")
print()

for mixing in (0.08, 0.5):
    subset = [r for r in rows if r["mixing"] == mixing]
    emitted = [r for r in subset if r["emitted"]]
    print(f"mixing {mixing}: {len(emitted)}/{len(subset)} emitted")
    print(
        f"  E_Q                      mean {np.mean([r['e_q'] for r in subset]):.4f}"
        f"   emitted {np.mean([r['e_q'] for r in emitted]):.4f}"
        if emitted
        else f"  E_Q                      mean {np.mean([r['e_q'] for r in subset]):.4f}"
    )
    print(f"  min within-state spread  mean {np.mean([r['min_spread'] for r in subset]):.4f}")
    print(
        f"  spread/E_Q ratio         mean {np.mean([r['spread_over_e_q'] for r in subset]):.3f}"
        f"   min {np.min([r['spread_over_e_q'] for r in subset]):.3f}"
        f"   max {np.max([r['spread_over_e_q'] for r in subset]):.3f}"
    )
    if emitted:
        print(
            f"  spread/E_Q, emitted only mean {np.mean([r['spread_over_e_q'] for r in emitted]):.3f}"
        )
    print(
        f"  best max_s LB            mean {np.mean([r['best_lb'] for r in subset]):.5f}"
    )
    print()

# Does spread/E_Q separate emitted from non-emitted?
emitted_ratio = [r["spread_over_e_q"] for r in rows if r["emitted"]]
blocked_ratio = [r["spread_over_e_q"] for r in rows if not r["emitted"]]
print("Separation test on spread/E_Q = (min within-state q spread) / E_Q")
print(f"  emitted     n={len(emitted_ratio):>3}  mean {np.mean(emitted_ratio):.3f}  "
      f"min {np.min(emitted_ratio):.3f}")
print(f"  not emitted n={len(blocked_ratio):>3}  mean {np.mean(blocked_ratio):.3f}  "
      f"max {np.max(blocked_ratio):.3f}")
threshold = np.mean(emitted_ratio)
tp = sum(1 for r in rows if r["emitted"] and r["spread_over_e_q"] >= threshold)
fn = sum(1 for r in rows if r["emitted"] and r["spread_over_e_q"] < threshold)
tn = sum(1 for r in rows if not r["emitted"] and r["spread_over_e_q"] < threshold)
fp = sum(1 for r in rows if not r["emitted"] and r["spread_over_e_q"] >= threshold)
print(f"  at threshold {threshold:.3f}: emitted correctly above {tp}/{tp + fn}, "
      f"blocked correctly below {tn}/{tn + fp}")
print()

# Does spread/E_Q itself depend on mixing?
mix_low = [r["spread_over_e_q"] for r in rows if r["mixing"] == 0.08]
mix_high = [r["spread_over_e_q"] for r in rows if r["mixing"] == 0.5]
print("Is the mixing effect carried by spread/E_Q?")
print(f"  mixing 0.08 spread/E_Q mean {np.mean(mix_low):.3f}")
print(f"  mixing 0.5  spread/E_Q mean {np.mean(mix_high):.3f}")
print(f"  ratio {np.mean(mix_low) / np.mean(mix_high):.3f}")
print()
print("e_q by mixing:")
print(f"  0.08 mean E_Q {np.mean([r['e_q'] for r in rows if r['mixing'] == 0.08]):.4f}")
print(f"  0.5  mean E_Q {np.mean([r['e_q'] for r in rows if r['mixing'] == 0.5]):.4f}")
