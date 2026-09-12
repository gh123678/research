"""Guard: the census certification batch is the one the sealed iteration used.

`FP-SCALE-002` sealed `cert_pair_counts` and `min_cert_count_observed` per record.
Those are a fingerprint of the realised sample, so they detect any change in the
generator's draw order -- including changes that are "equivalent for independent
chains" in distribution but not in realisation.

This exists because the first census implementation WAS changed that way (a
vectorised per-step rewrite) and produced different counts on every probed record.
The guard is what caught it. It is kept as a permanent artifact so that the census
cannot silently drift onto a different batch later.

Exit 0 means every probed record matches the sealed pair-count vector exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import evaluate_fp_census_001 as census  # noqa: E402
import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402

SEALED = (
    PROJECT / "results" / "FP-SCALE-002" / "claude" / "formal" / "task_results.json"
)
PROBES = ((0.5, 0), (0.5, 1), (0.08, 0), (0.08, 3))


def main() -> int:
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
    }

    mismatched: list[str] = []
    for mixing, task_index in PROBES:
        mdp, policy, _ = fs.build_task(task_index=task_index, mixing=mixing)
        exact = policy_quantities(mdp, policy)
        mu_state = np.asarray(exact["mu_state"], dtype=np.float64)
        batch = census.certification_batch(
            mdp, policy, mu_state, [fs.SEED, 9001, int(round(mixing * 100)), task_index]
        )
        counts = fs.pair_counts(batch["states"], batch["actions"])
        record = by_key[(mixing, task_index)]
        sealed_counts = np.asarray(record["cert_pair_counts"], dtype=np.int64)
        same = bool(np.array_equal(counts, sealed_counts))
        print(
            f"mixing={mixing} task={task_index}: census min={int(counts.min())} "
            f"sealed min={int(record['min_cert_count_observed'])} "
            f"pairCountsMatch={same}"
        )
        if not same:
            mismatched.append(f"({mixing}, {task_index})")

    print()
    if mismatched:
        print(
            "FAIL: the census batch differs from the sealed one on "
            f"{len(mismatched)} of {len(PROBES)} probes: {mismatched}"
        )
        return 1
    print(
        f"PASS: the census batch reproduces the sealed pair-count vector exactly "
        f"on all {len(PROBES)} probes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
