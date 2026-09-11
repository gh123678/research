"""Analyzer for the FP-ESARSA-001 formal results (Claude main route).

Reads ``task_results.json`` and ``config.json`` from a result directory and
writes ``summary.json``. Every reported quantity is model-free (it is built
from route outputs, certificates, and the ordered reason lists) except the
fields explicitly taken from ``oracle_audit``, which are labelled ``oracle_``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_fixed_policy_q_routes import write_json

ROUTES = ("expected_exact", "expected_finite", "sampled_exact")


def load_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate,
    )


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _cell_key(record: dict[str, Any]) -> tuple[int, float, float]:
    return (
        int(record["trajectory_length"]),
        float(record["mixing"]),
        float(record["gap_bonus"]),
    )


def summarize_route(records: list[dict[str, Any]], route: str) -> dict[str, Any]:
    certificates = [record["routes"][route] for record in records]
    audits = [record["oracle_audit"]["routes"][route] for record in records]
    emitted = [item for item in certificates if item["status"] == "certificate_emitted"]
    safe = [item for item in certificates if item["improvement"]["status"] == "safe_update_emitted"]
    reasons = Counter(
        reason for item in certificates for reason in item["failure_reasons"]
    )
    eta_counts = Counter(
        item["improvement"]["eta_selected"]
        for item in safe
        if item["improvement"]["eta_selected"] is not None
    )
    e_q = [float(item["certificate"]["e_q"]) for item in emitted]
    bound_slack = [
        audit["bound_slack"] for audit in audits if audit["bound_slack"] is not None
    ]
    return {
        "records": len(records),
        "certificate_emitted": len(emitted),
        "certificate_emission_rate": len(emitted) / len(records),
        "safe_update_emitted": len(safe),
        "safe_update_rate": len(safe) / len(records),
        "mean_e_q_among_emitted": _mean(e_q),
        "min_e_q_among_emitted": None if not e_q else min(e_q),
        "failure_reason_rates": {
            reason: count / len(records) for reason, count in sorted(reasons.items())
        },
        "eta_selected_counts": {str(eta): count for eta, count in sorted(eta_counts.items())},
        "contraction_premise_satisfied_rate": sum(
            item["contraction"]["premise_satisfied"] for item in certificates
        )
        / len(records),
        "oracle_certificate_violations": sum(
            audit["certificate_violation"] is True for audit in audits
        ),
        "oracle_residual_event_violations": sum(
            audit["residual_event_violation"] is True for audit in audits
        ),
        "oracle_value_decreases": sum(
            audit["oracle_value_decrease"] is True for audit in audits
        ),
        "oracle_audited_updates": sum(audit["update_emitted"] for audit in audits),
        "oracle_bound_slack_min": None if not bound_slack else min(bound_slack),
        "oracle_max_actual_q_sup_error": max(
            audit["actual_q_sup_error"] for audit in audits
        ),
    }


def summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_cell_key(record)].append(record)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        length, mixing, gap_bonus = key
        selected = grouped[key]
        row: dict[str, Any] = {
            "trajectory_length": length,
            "mixing": mixing,
            "gap_bonus": gap_bonus,
            "records": len(selected),
        }
        for route in ROUTES:
            route_summary = summarize_route(selected, route)
            row[route] = {
                "certificate_emission_rate": route_summary["certificate_emission_rate"],
                "safe_update_rate": route_summary["safe_update_rate"],
                "mean_e_q_among_emitted": route_summary["mean_e_q_among_emitted"],
                "contraction_premise_satisfied_rate": route_summary[
                    "contraction_premise_satisfied_rate"
                ],
                "oracle_certificate_violations": route_summary["oracle_certificate_violations"],
                "oracle_value_decreases": route_summary["oracle_value_decreases"],
            }
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    records = load_json(result_dir / "task_results.json")
    config = load_json(result_dir / "config.json")

    summary: dict[str, Any] = {
        "task_id": config.get("task_id", "FP-ESARSA-001"),
        "mode": config.get("mode", args.mode),
        "record_count": len(records),
        "routes": {route: summarize_route(records, route) for route in ROUTES},
        "cells": summary_rows(records),
        "cross_route": {
            "mean_exact_vs_sampled_exact_q_sup_gap": _mean(
                [float(r["cross_route"]["expected_exact_vs_sampled_exact_q_sup_gap"]) for r in records]
            ),
            "mean_exact_vs_finite_q_sup_gap": _mean(
                [float(r["cross_route"]["expected_exact_vs_expected_finite_q_sup_gap"]) for r in records]
            ),
        },
        "integrity": {
            "reward_bound_satisfied_all": all(
                r["oracle_audit"]["reward_bound"]["satisfied"] for r in records
            ),
            "any_oracle_certificate_violation": any(
                r["oracle_audit"]["any_certificate_violation"] for r in records
            ),
            "any_oracle_residual_event_violation": any(
                r["oracle_audit"]["any_residual_event_violation"] for r in records
            ),
            "any_oracle_value_decrease": any(
                r["oracle_audit"]["any_value_decrease"] for r in records
            ),
            "any_contraction_premise_failure": any(
                r["oracle_audit"]["any_contraction_premise_failure"] for r in records
            ),
        },
    }
    write_json(result_dir / "summary.json", summary)

    print(f"analyzed {len(records)} records from {result_dir}")
    for route in ROUTES:
        row = summary["routes"][route]
        print(
            f"{route:18s} cert_rate={row['certificate_emission_rate']:.4f} "
            f"update_rate={row['safe_update_rate']:.4f} "
            f"oracle_violations={row['oracle_certificate_violations']} "
            f"oracle_value_decreases={row['oracle_value_decreases']}"
        )
    print(f"integrity: {summary['integrity']}")
    print("PASS fixed-policy Expected SARSA analysis")


if __name__ == "__main__":
    main()
