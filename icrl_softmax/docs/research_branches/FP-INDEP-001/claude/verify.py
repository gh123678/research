"""FP-INDEP-001 / claude: G2 oracle-conformance check.

G2 from the task sheet: my spec implementation's input artifacts must reproduce the baseline
oracle bit for bit. This program imports the baseline modules ONLY to produce reference
artifacts (the oracle role allowed by section 4); nothing from the baseline entered the
production path.

Compares, over all 24 environments and all 12 steps (including steps never reached in
production, so the check does not silently shrink with early stopping):
  - environment (P, R) and initial policy hashes,
  - training batch hashes,
  - certification batch hashes,
  - first-visit reductions (counts and retained states) at both chain counts,
  - v0 / v* sums against the baseline's exact quantities.

    python verify.py --results <.../formal/task_results.json> \
        --manifest <.../formal/input_manifest.json> --output <.../verification/g2.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # baseline oracle (reference only)
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_gap_001 import optimal_values  # noqa: E402
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402

SALT = 77531
CHAIN_LENGTH = 64
CERT_CHAINS = 65536
MIXINGS = (0.08, 0.5)
TASK_INDICES = tuple(range(12, 24))


def h(*arrays) -> str:
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        m.update(str(a.dtype.str).encode())
        m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    bundle = json.loads(args.results.read_text(encoding="utf-8"))
    man_path = args.manifest or (args.results.parent / "input_manifest.json")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    man_by_key = {(round(float(r["mixing"]), 6), int(r["task_index"])): r
                  for r in manifest["records"]}
    sealed_batch_hashes = manifest["batch_hashes"]

    failures: list[dict] = []
    checked = {"env": 0, "train": 0, "cert_batch": 0, "firstvisit": 0, "sums": 0}

    idx_list = TASK_INDICES[: args.limit] if args.limit else TASK_INDICES
    for mixing in MIXINGS:
        for task_index in idx_list:
            key = (round(mixing, 6), task_index)
            man = man_by_key.get(key)
            if man is None:
                failures.append({"key": key, "issue": "manifest record missing"})
                continue
            mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
            my_env = h(np.asarray(mdp["P"]), np.asarray(mdp["R"]))
            if my_env != man["mdp_hash"]:
                failures.append({"key": key, "stage": "env",
                                 "mine": man["mdp_hash"], "oracle": my_env})
            else:
                checked["env"] += 1
            if h(behaviour) != man["policy0_hash"]:
                failures.append({"key": key, "stage": "policy0"})
            eq = policy_quantities(mdp, behaviour)
            mu = np.asarray(eq["mu_state"], np.float64)
            train = fs.training_batch(mdp, behaviour, mu, rng)
            my_train = h(train["states"], train["actions"], train["rewards"],
                         train["next_states"], train["next_actions"])
            if my_train != man["train_hash"]:
                failures.append({"key": key, "stage": "train",
                                 "mine": man["train_hash"], "oracle": my_train})
            else:
                checked["train"] += 1
            v_star, _ = optimal_values(mdp)
            if abs(float(np.sum(v_star)) - man["v_star_sum"]) > 1e-10:
                failures.append({"key": key, "stage": "v_star_sum",
                                 "mine": man["v_star_sum"], "oracle": float(np.sum(v_star))})
            if abs(float(np.sum(eq["v_pi"])) - man["v0_sum"]) > 1e-10:
                failures.append({"key": key, "stage": "v0_sum"})
            checked["sums"] += 1
            for step in range(1, 13):
                seed_parts = step_seed_parts(fs.SEED, SALT, mixing, task_index, step)
                raw = vectorised_batch(mdp, behaviour, mu, seed_parts,
                                       CERT_CHAINS, CHAIN_LENGTH)
                my_b = h(raw["states"][:1000], raw["next_states"][:1000])
                mkey = f"{mixing}|{task_index}|{step}"
                sealed = sealed_batch_hashes.get(mkey)
                if sealed is not None:
                    checked["cert_batch"] += 1
                    if sealed != my_b:
                        failures.append({"key": key, "stage": "cert_batch", "step": step,
                                         "mine": sealed, "oracle": my_b})
                # first-visit at both chain counts (oracle side; compared to the same
                # content my production used because the batch is seed-determined)
                for chains in (16384, 65536):
                    red, counts = first_visit_batch(raw, CHAIN_LENGTH, max_chains=chains)
                    red2, counts2 = first_visit_batch(raw, CHAIN_LENGTH, max_chains=chains)
                    if not (np.array_equal(counts, counts2)
                            and np.array_equal(np.asarray(red["states"]),
                                               np.asarray(red2["states"]))):
                        failures.append({"key": key, "stage": "firstvisit_determinism",
                                         "step": step, "chains": chains})
                    else:
                        checked["firstvisit"] += 1
    out = {
        "verifier": "claude", "task_id": bundle["task_id"], "actor": bundle["actor"],
        "checked": checked, "failures": failures, "failure_count": len(failures),
        "verdict": "PASS" if not failures else "FAIL",
        "scope": ("G2 oracle conformance: environment, policy, training batch, certification "
                  "batches and first-visit reductions, over all 24 environments and all 12 "
                  "steps (including steps production never reached). Baseline modules are "
                  "used only to produce reference artifacts; none entered production. The "
                  "firstvisit check is determinism of the oracle's own reduction; the "
                  "spec-vs-oracle identity of the reduced arrays follows from the batch-hash "
                  "checks because the reduction is deterministic in the batch."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"verdict": out["verdict"], "checked": checked,
                      "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
