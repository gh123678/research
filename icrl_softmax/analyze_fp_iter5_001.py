"""FP-ITER5-001 analyzer: fifth certified step on both the numpy and network paths.

Recomputes the per-level pattern for both paths, evaluates H1--H11, and compares
the two paths against each other.

Usage:
    python -B analyze_fp_iter5_001.py --numpy-dir <dir> --network-dir <dir> [--write-results]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TASK_ID = "FP-ITER5-001"
PRIMARY = ("expected_exact", "expected_finite")
PROJECT = Path(__file__).resolve().parent
NUMPY_SEALED4 = PROJECT / "results" / "FP-ITER4-001" / "claude" / "formal" / "task_results.json"
NET_SEALED4 = (
    PROJECT / "results" / "FP-ATTN-ITER4-001" / "claude" / "formal" / "task_results.json"
)


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key: {key}")
        seen.add(key)
        out[key] = value
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)


def blocks_of(bundle: dict[str, Any]):
    return [
        (record, route, record["routes"][route])
        for record in bundle["records"]
        for route in PRIMARY
    ]


def per_level(steps, max_steps: int, key: str = "oracle_audit") -> dict[int, dict]:
    out: dict[int, dict] = {}
    for level in range(1, max_steps + 1):
        gains = [
            s[key]["total_value_gain"]
            for _, _, _, s in steps
            if s["step"] == level and s["update_emitted"]
        ]
        gaps = [
            s["q_hat_gap_vs_numpy"]
            for _, _, _, s in steps
            if s["step"] == level and "q_hat_gap_vs_numpy" in s
        ]
        out[level] = {
            "emissions": len(gains),
            "mean_gain": (sum(gains) / len(gains)) if gains else None,
            "min_gain": min(gains) if gains else None,
            "max_q_gap": max(gaps) if gaps else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numpy-dir", type=Path, required=True)
    parser.add_argument("--network-dir", type=Path, required=True)
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()

    np_bundle = load(args.numpy_dir / "task_results.json")
    net_bundle = load(args.network_dir / "task_results.json")
    np_steps = [(r, route, b, s) for r, route, b in blocks_of(np_bundle) for s in b["steps"]]
    net_steps = [
        (r, route, b, s) for r, route, b in blocks_of(net_bundle) for s in b["steps"]
    ]
    np_max = int(np_bundle["max_steps"])
    net_max = int(net_bundle["max_steps"])
    atol = float(net_bundle["atol"])

    np_level = per_level(np_steps, np_max)
    net_level = per_level(net_steps, net_max)

    # H1: numpy inertness vs sealed FP-ITER4-001.
    h1 = {"compared": 0, "mismatches": 0}
    if NUMPY_SEALED4.exists():
        sealed = load(NUMPY_SEALED4)
        by_key = {(float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]}
        for record, route, block in blocks_of(np_bundle):
            ref = by_key.get((float(record["mixing"]), int(record["task_index"])))
            if ref is None:
                continue
            ref_steps = ref["routes"][route]["steps"]
            for i in range(min(4, len(block["steps"]), len(ref_steps))):
                h1["compared"] += 1
                a, b = block["steps"][i], ref_steps[i]
                if (
                    a["update_emitted"] != b["update_emitted"]
                    or a["eta_selected"] != b["eta_selected"]
                    or a["e_q"] != b["e_q"]
                ):
                    h1["mismatches"] += 1

    # H2: network inertness vs sealed FP-ATTN-ITER4-001 (including Q gaps).
    h2 = {"compared": 0, "mismatches": 0}
    if NET_SEALED4.exists():
        sealed = load(NET_SEALED4)
        by_key = {(float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]}
        for record, route, block in blocks_of(net_bundle):
            ref = by_key.get((float(record["mixing"]), int(record["task_index"])))
            if ref is None:
                continue
            ref_steps = ref["routes"][route]["steps"]
            for i in range(min(4, len(block["steps"]), len(ref_steps))):
                h2["compared"] += 1
                a, b = block["steps"][i], ref_steps[i]
                if (
                    a["update_emitted"] != b["update_emitted"]
                    or a["eta_selected"] != b["eta_selected"]
                    or a["e_q"] != b["e_q"]
                    or a["q_hat_gap_vs_numpy"] != b["q_hat_gap_vs_numpy"]
                ):
                    h2["mismatches"] += 1

    # Validity on both paths.
    def violations(steps):
        cert = [
            (r["mixing"], r["task_index"], route, s["step"])
            for r, route, _, s in steps
            if s["oracle_audit"].get("certificate_violation")
        ]
        nondeg = [
            (r["mixing"], r["task_index"], route, s["step"])
            for r, route, _, s in steps
            if s["update_emitted"]
            and min(s["oracle_audit"]["value_delta_vs_previous"]) < -1e-12
        ]
        return cert, nondeg

    np_cert, np_nondeg = violations(np_steps)
    net_cert, net_nondeg = violations(net_steps)

    # H11: path agreement per level.
    path_levels = {
        level: {
            "numpy": np_level[level]["emissions"],
            "network": net_level[level]["emissions"],
            "match": np_level[level]["emissions"] == net_level[level]["emissions"],
        }
        for level in range(1, min(np_max, net_max) + 1)
    }
    net_comp = [
        (r, route, c)
        for r, route, b in blocks_of(net_bundle)
        for c in b["reference_comparisons"]
    ]
    net_disagree = [c for c in net_comp if not c[2]["decision_match"]]

    n4 = np_level[4]["emissions"]
    n5 = np_level[5]["emissions"]
    mean4 = np_level[4]["mean_gain"]
    mean5 = np_level[5]["mean_gain"]
    min5 = np_level[5]["min_gain"]
    gap5 = net_level[5]["max_q_gap"]

    summary = {
        "task_id": TASK_ID,
        "numpy_label": np_bundle.get("label"),
        "network_label": net_bundle.get("label"),
        "max_steps": {"numpy": np_max, "network": net_max},
        "atol": atol,
        "numpy_per_level": {str(k): v for k, v in np_level.items()},
        "network_per_level": {str(k): v for k, v in net_level.items()},
        "numpy_five_step_routes": sum(
            1 for _, _, b in blocks_of(np_bundle) if b["emitted_steps"] >= np_max
        ),
        "network_five_step_routes": sum(
            1 for _, _, b in blocks_of(net_bundle) if b["emitted_steps"] >= net_max
        ),
        "H1_numpy_inert": {
            **h1,
            "outcome": "PASS" if h1["compared"] > 0 and not h1["mismatches"] else "FAIL",
        },
        "H2_network_inert": {
            **h2,
            "outcome": "PASS" if h2["compared"] > 0 and not h2["mismatches"] else "FAIL",
        },
        "H3_fifth_step_both": {
            "numpy": n5,
            "network": net_level[5]["emissions"],
            "outcome": (
                "PASS" if n5 > 0 and net_level[5]["emissions"] > 0 else "FALSIFIED"
            ),
        },
        "H4_fifth_valid": {
            "numpy_min_gain": min5,
            "network_min_gain": net_level[5]["min_gain"],
            "outcome": (
                "PASS"
                if (min5 or 0) > 0 and (net_level[5]["min_gain"] or 0) > 0
                else "FALSIFIED"
            ),
        },
        "H5_monotone_value": bool(not np_nondeg and not net_nondeg),
        "H6_no_certificate_violations": bool(not np_cert and not net_cert),
        "H7_attrition_prediction": {
            "prediction": "n5 < n4 on the numpy path",
            "n4": n4,
            "n5": n5,
            "outcome": "PASS" if n5 < n4 else "FALSIFIED",
        },
        "H8_mean_gain_prediction": {
            "prediction": "mean5 < mean4 on the numpy path",
            "mean_gain_step4": mean4,
            "mean_gain_step5": mean5,
            "outcome": (
                "PASS" if (mean4 is not None and mean5 is not None and mean5 < mean4) else "FALSIFIED"
            ),
        },
        "H9_non_vacuity_prediction": {
            "prediction": "min5 > 0.05",
            "min_gain_step5": min5,
            "outcome": "PASS" if (min5 is not None and min5 > 0.05) else "FALSIFIED",
        },
        "H10_drift_prediction": {
            "prediction": "step-5 network Q gap <= ATOL",
            "max_step5_q_gap": gap5,
            "atol": atol,
            "outcome": "PASS" if (gap5 is not None and gap5 <= atol) else "FALSIFIED",
        },
        "H11_path_agreement": {
            "per_level": {str(k): v for k, v in path_levels.items()},
            "decision_comparisons": len(net_comp),
            "decision_disagreements": len(net_disagree),
            "outcome": (
                "PASS"
                if all(v["match"] for v in path_levels.values()) and not net_disagree
                else "FALSIFIED"
            ),
        },
        "numpy_certificate_violations": len(np_cert),
        "network_certificate_violations": len(net_cert),
        "numpy_nondegrading_violations": len(np_nondeg),
        "network_nondegrading_violations": len(net_nondeg),
    }
    analysis = {
        "task_id": TASK_ID,
        "reconstruction": "recomputed from both bundles with strict duplicate-key detection",
        "numpy_gains_by_level": {
            str(level): [
                s["oracle_audit"]["total_value_gain"]
                for _, _, _, s in np_steps
                if s["step"] == level and s["update_emitted"]
            ]
            for level in range(1, np_max + 1)
        },
        "network_q_gaps_by_level": {
            str(level): [
                s["q_hat_gap_vs_numpy"]
                for _, _, _, s in net_steps
                if s["step"] == level
            ]
            for level in range(1, net_max + 1)
        },
        "artifacts": {
            "numpy": {
                name: sha256(args.numpy_dir / name)
                for name in ("task_results.json", "config.json", "environment.json")
                if (args.numpy_dir / name).exists()
            },
            "network": {
                name: sha256(args.network_dir / name)
                for name in ("task_results.json", "config.json", "environment.json")
                if (args.network_dir / name).exists()
            },
        },
    }

    if args.write_results:
        for directory in (args.numpy_dir, args.network_dir):
            (directory / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
            )
            (directory / "analysis.json").write_text(
                json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
            )

    print(f"{TASK_ID}")
    for label, levels in (("numpy", np_level), ("network", net_level)):
        print(f"  {label}:")
        for level, row in levels.items():
            print(
                f"    step {level}: emissions={row['emissions']:>3} "
                f"mean_gain={row['mean_gain']} min_gain={row['min_gain']}"
                + (f" max|dQ|={row['max_q_gap']:.3e}" if row["max_q_gap"] else "")
            )
    print(
        f"  five-step routes: numpy={summary['numpy_five_step_routes']} "
        f"network={summary['network_five_step_routes']}"
    )
    for key in (
        "H1_numpy_inert",
        "H2_network_inert",
        "H3_fifth_step_both",
        "H4_fifth_valid",
        "H5_monotone_value",
        "H6_no_certificate_violations",
        "H7_attrition_prediction",
        "H8_mean_gain_prediction",
        "H9_non_vacuity_prediction",
        "H10_drift_prediction",
        "H11_path_agreement",
    ):
        entry = summary[key]
        outcome = entry.get("outcome") if isinstance(entry, dict) else entry
        print(f"  {key}: {outcome}")


if __name__ == "__main__":
    main()
